# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Python wrappers for Slang quaternion operations with full PyTorch autograd support.

This module provides Python bindings to the high-performance Slang quaternion kernels
with automatic differentiation via PyTorch's autograd system.

All functions follow the same conventions:
- Quaternion format: float4 in xyzw order (x, y, z, w)
- Unit quaternions represent rotations in 3D space
- All operations support automatic differentiation
"""

from typing import Any

import torch

from torch import Tensor

from libs.geometry.kernels.interface import libquaternion_slang_cc as quaternion_slang
from libs.slang_utils.utils import div_up


# Slang module constants
_THREADS_PER_BLOCK = 256
_MODULE_NAME = "quaternion"


class QuatSlerpFunction(torch.autograd.Function):
    """Spherical linear interpolation between quaternions."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor, t: float) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        count = q1.shape[0]
        result = torch.empty_like(q1)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_slerp_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (q1,)),
                (q2, (q2,)),
                t,
                (result, (result,)),
            )

        ctx.save_for_backward(q1, q2, result)
        ctx.count = count
        ctx.t = t
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        q1, q2, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_q1 = torch.empty_like(q1)
        grad_q2 = torch.empty_like(q2)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_slerp_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (grad_q1,)),
                (q2, (grad_q2,)),
                ctx.t,
                (result, (grad_result,)),
            )

        return grad_q1, grad_q2, None  # None for t (not differentiable)


def quat_slerp(q1: Tensor, q2: Tensor, t: float | Tensor) -> Tensor:
    """Spherical linear interpolation between quaternions (GPU accelerated, differentiable).

    Args:
        q1: (..., 4) start quaternion(s) in xyzw format
        q2: (..., 4) end quaternion(s) in xyzw format
        t: Interpolation parameter [0, 1] (scalar)

    Returns:
        (..., 4) interpolated quaternion(s)
    """
    assert q1.shape == q2.shape, f"Shape mismatch: {q1.shape} vs {q2.shape}"
    if isinstance(t, Tensor):
        assert t.numel() == 1, "t must be a scalar"
        t = t.item()

    original_shape = q1.shape
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    result_flat = QuatSlerpFunction.apply(q1_flat, q2_flat, float(t))
    return result_flat.reshape(original_shape)
