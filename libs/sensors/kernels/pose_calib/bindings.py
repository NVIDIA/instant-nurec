# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Python bindings for pose calibration kernel operations.

This module provides thin wrappers around hand-written CUDA pose calibration kernels.
Both the forward and backward passes run in CUDA with analytic Jacobians for the
backward (gradients of embed_weights only).
"""

from typing import Any, Optional, Tuple

import torch

from torch import Tensor

import libs.sensors.libpose_calib_cuda_cc as pose_calib_cuda  # type: ignore # pycena: skip


# ============================================================================
# Autograd Functions
# ============================================================================


class ComputePosesAndTimestampsFunction(torch.autograd.Function):
    """Autograd function for fused pose calibration and rolling shutter interpolation."""

    @staticmethod
    def forward(
        ctx,
        T_sensor_world_startend_allviews: Tensor,
        embed_weights: Optional[Tensor],
        frame_idx: Tensor,
        rect_points_lb: Optional[Tensor],
        resolution: Optional[Tensor],
        timestamps_startend_us_allviews: Tensor,
        shutter_type: int,
        enable_calib: bool,
    ) -> Tuple[Tensor, Tensor]:
        T_sensor_world_startend_allviews = T_sensor_world_startend_allviews.contiguous()
        assert not T_sensor_world_startend_allviews.requires_grad, (
            "Gradients w.r.t. T_sensor_world_startend_allviews not supported. Detach before passing."
        )
        embed_weights = embed_weights.contiguous() if embed_weights is not None else None
        frame_idx = frame_idx.contiguous()
        timestamps_startend_us_allviews = timestamps_startend_us_allviews.contiguous()

        batch_size = frame_idx.shape[0]
        device = T_sensor_world_startend_allviews.device

        has_subsampling = rect_points_lb is not None and resolution is not None
        if has_subsampling:
            assert rect_points_lb is not None
            assert resolution is not None
            rect_points_lb = rect_points_lb.contiguous()
            resolution = resolution.contiguous()

        T_out = torch.empty((batch_size, 2, 4, 4), device=device, dtype=torch.float32)
        timestamps_out = torch.empty((batch_size, 2), device=device, dtype=torch.int64)
        # Scratch for saved intermediates: per-thread (5 float4 = 20 floats).
        # Layout: q_calib_{s,e}, q_out_{s,e}, (t0, t1, 0, 0).
        # Skipped in inference (embed_weights.requires_grad=False): forward
        # avoids ~5 float4 scatter writes per thread (~0.6 us). Note: we can't
        # use torch.is_grad_enabled() here because PyTorch disables it inside
        # Function.forward; requires_grad on the tensor is preserved.
        need_scratch = enable_calib and embed_weights is not None and embed_weights.requires_grad
        scratch = (
            torch.empty((batch_size, 5, 4), device=device, dtype=torch.float32)
            if need_scratch
            else torch.empty(0, device=device, dtype=torch.float32)
        )

        if batch_size > 0:
            dummy_f32 = torch.empty(0, device=device, dtype=torch.float32)
            pose_calib_cuda.forward(
                batch_size,
                T_sensor_world_startend_allviews,
                embed_weights if embed_weights is not None else dummy_f32,
                frame_idx,
                rect_points_lb if rect_points_lb is not None else dummy_f32,
                resolution if resolution is not None else dummy_f32,
                timestamps_startend_us_allviews,
                shutter_type,
                enable_calib,
                has_subsampling,
                T_out,
                timestamps_out,
                scratch,
            )

        _sentinel = torch.empty(0, device=device, dtype=torch.float32)
        ctx.save_for_backward(
            T_sensor_world_startend_allviews,
            embed_weights if embed_weights is not None else _sentinel,
            frame_idx,
            rect_points_lb if rect_points_lb is not None else _sentinel,
            resolution if resolution is not None else _sentinel,
            scratch,
        )
        ctx.batch_size = batch_size
        ctx.shutter_type = shutter_type
        ctx.enable_calib = enable_calib
        ctx.has_subsampling = has_subsampling

        return T_out, timestamps_out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Any
    ) -> Tuple[Optional[Tensor], Optional[Tensor], None, None, None, None, None, None]:
        if not ctx.enable_calib:
            return None, None, None, None, None, None, None, None

        grad_T_out, _ = grad_outputs
        (
            T_allviews,
            embed_weights,
            frame_idx,
            rect_points_lb,
            resolution,
            scratch,
        ) = ctx.saved_tensors

        # Sentinels (numel==0) mean the original was None
        if embed_weights.numel() == 0:
            return None, None, None, None, None, None, None, None

        grad_T_out = grad_T_out.contiguous()

        # Zero-initialized output gradient buffer (atomicAdd accumulates into this)
        grad_embed_weights = torch.zeros_like(embed_weights)

        dummy_f32 = torch.empty(0, device=embed_weights.device, dtype=torch.float32)

        pose_calib_cuda.backward(
            ctx.batch_size,
            T_allviews,
            embed_weights,
            frame_idx,
            rect_points_lb if rect_points_lb.numel() > 0 else dummy_f32,
            resolution if resolution.numel() > 0 else dummy_f32,
            ctx.shutter_type,
            ctx.has_subsampling,
            grad_T_out,
            grad_embed_weights,
            scratch,
        )

        return None, grad_embed_weights, None, None, None, None, None, None


# ============================================================================
# Public API
# ============================================================================


def compute_poses_and_timestamps(
    T_sensor_world_startend_allviews: Tensor,
    embed_weights: Optional[Tensor],
    frame_idx: Tensor,
    rect_points_lb: Optional[Tensor],
    resolution: Optional[Tensor],
    timestamps_startend_us_allviews: Tensor,
    shutter_type: int,
    enable_calib: bool,
) -> Tuple[Tensor, Tensor]:
    """Compute interpolated poses and timestamps with optional calibration refinement.

    Fused pose calibration and rolling shutter interpolation. Fully differentiable.

    Args:
        T_sensor_world_startend_allviews: (V, 2, 4, 4) start/end poses for all views
        embed_weights: Embedding weights for calibration refinement
        frame_idx: (N,) frame indices into the views
        rect_points_lb: Optional (N, 2, 2) rect points for subsampling
        resolution: Optional (N, 2) resolution for subsampling
        timestamps_startend_us_allviews: (V, 2) start/end timestamps per view in microseconds
        shutter_type: Shutter type enum value
        enable_calib: Whether to apply calibration refinement

    Returns:
        T_out: (N, 2, 4, 4) interpolated start/end poses per sample
        timestamps_out: (N, 2) interpolated start/end timestamps per sample
    """
    assert 1 <= shutter_type <= 5, f"Invalid shutter_type: {shutter_type}"

    batch_size = frame_idx.shape[0]
    device = T_sensor_world_startend_allviews.device

    if batch_size == 0:
        return (
            torch.empty((0, 2, 4, 4), device=device, dtype=torch.float32),
            torch.empty((0, 2), device=device, dtype=torch.int64),
        )

    # Validate input shapes
    num_views = T_sensor_world_startend_allviews.shape[0]

    assert T_sensor_world_startend_allviews.dim() == 4 and T_sensor_world_startend_allviews.shape[1:] == (2, 4, 4), (
        f"T_sensor_world_startend_allviews must have shape (V, 2, 4, 4), got {T_sensor_world_startend_allviews.shape}"
    )

    assert timestamps_startend_us_allviews.dim() == 2 and timestamps_startend_us_allviews.shape == (num_views, 2), (
        f"timestamps_startend_us_allviews must have shape ({num_views}, 2), got {timestamps_startend_us_allviews.shape}"
    )

    assert frame_idx.dim() == 1, f"frame_idx must be 1D, got {frame_idx.dim()}D"

    if enable_calib:
        assert embed_weights is not None, "embed_weights must be provided if enable_calib is True"
        assert embed_weights.dim() == 2 and embed_weights.shape[1] == 9, (
            f"embed_weights must have shape (V, 9), got {embed_weights.shape}"
        )

    if rect_points_lb is not None:
        assert rect_points_lb.dim() == 3 and rect_points_lb.shape[1:] == (2, 2), (
            f"rect_points_lb must have shape (N, 2, 2), got {rect_points_lb.shape}"
        )
        assert rect_points_lb.shape[0] == batch_size, (
            f"rect_points_lb batch size {rect_points_lb.shape[0]} != frame_idx batch size {batch_size}"
        )
    if resolution is not None:
        assert resolution.dim() == 2 and resolution.shape[1] == 2, (
            f"resolution must have shape (N, 2), got {resolution.shape}"
        )
        assert resolution.shape[0] == batch_size, (
            f"resolution batch size {resolution.shape[0]} != frame_idx batch size {batch_size}"
        )

    assert (rect_points_lb is None) == (resolution is None), (
        "rect_points_lb and resolution must both be provided or both be None"
    )

    return ComputePosesAndTimestampsFunction.apply(
        T_sensor_world_startend_allviews,
        embed_weights,
        frame_idx,
        rect_points_lb,
        resolution,
        timestamps_startend_us_allviews,
        shutter_type,
        enable_calib,
    )


__all__ = [
    "compute_poses_and_timestamps",
]
