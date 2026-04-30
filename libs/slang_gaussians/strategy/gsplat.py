# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from libs.slang_gaussians.interface import gsplat_strategy_cuda  # type: ignore


def update_gradient_buffers(
    positions: torch.Tensor,
    params_grad: torch.Tensor,
    ray_origin: torch.Tensor,
    grad_norm_accum: torch.Tensor,
    grad_norm_denom: torch.Tensor,
) -> None:
    """
    Update gradient accumulation buffers for densification.

    This computes distance-scaled gradient norms for each Gaussian that was hit
    (has non-zero gradients) and accumulates them into the provided buffers.

    Args:
        positions: [N, 3] Gaussian positions
        params_grad: [N, 3] Gradients of positions from backpropagation
        ray_origin: [3] Camera/ray origin (assumes single origin for batch)
        grad_norm_accum: [N, 1] Accumulated gradient norms (modified in-place)
        grad_norm_denom: [N, 1] Accumulator count (modified in-place)
    """
    n_gaussians = positions.size(0)
    assert positions.dim() == 2 and positions.size(1) == 3, f"positions must be [N, 3], got {positions.shape}"
    assert params_grad.dim() == 2 and params_grad.size(1) == 3, f"params_grad must be [N, 3], got {params_grad.shape}"
    assert ray_origin.dim() == 1 and ray_origin.size(0) == 3, f"ray_origin must be [3], got {ray_origin.shape}"
    assert grad_norm_accum.dim() == 2 and grad_norm_accum.size(1) == 1, (
        f"grad_norm_accum must be [N, 1], got {grad_norm_accum.shape}"
    )
    assert grad_norm_denom.dim() == 2 and grad_norm_denom.size(1) == 1, (
        f"grad_norm_denom must be [N, 1], got {grad_norm_denom.shape}"
    )
    assert params_grad.size(0) == n_gaussians, "params_grad must have same batch size as positions"
    assert grad_norm_accum.size(0) == n_gaussians, "grad_norm_accum must have same batch size as positions"
    assert grad_norm_denom.size(0) == n_gaussians, "grad_norm_denom must have same batch size as positions"
    assert (
        positions.is_cuda
        and params_grad.is_cuda
        and ray_origin.is_cuda
        and grad_norm_accum.is_cuda
        and grad_norm_denom.is_cuda
    ), "Tensors must be on CUDA device"
    assert (
        positions.dtype == torch.float32
        and params_grad.dtype == torch.float32
        and ray_origin.dtype == torch.float32
        and grad_norm_accum.dtype == torch.float32
    ), "Tensors (except grad_norm_denom) must be float32"
    assert grad_norm_denom.dtype == torch.int32, "grad_norm_denom must be int32"

    if n_gaussians == 0:
        return

    # Ensure contiguous
    positions = positions.contiguous()
    params_grad = params_grad.contiguous()
    ray_origin = ray_origin.contiguous()
    grad_norm_accum = grad_norm_accum.contiguous()
    grad_norm_denom = grad_norm_denom.contiguous()

    threads_per_block = 256

    gsplat_strategy_cuda.update_gradient_buffers(
        positions,
        params_grad,
        ray_origin,
        grad_norm_accum,
        grad_norm_denom,
        threads_per_block,
    )
