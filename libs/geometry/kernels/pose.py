# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Python wrappers for Slang pose operations with PyTorch autograd support.

This module provides Python bindings to the high-performance Slang pose kernels
with automatic differentiation via PyTorch's autograd system.

The main pose.slang file contains the core SE3Pose and trajectory implementations.
This module provides differentiable wrappers around pose_kernels.slang functions.

All functions expect batched torch tensors with shape (N, D) where N is the batch size.
For single inputs, add a batch dimension: tensor.reshape(1, -1)
"""

from typing import Any

import torch

from libs.geometry.kernels.interface import libpose_kernels_slang_cc as pose_kernels_slang
from libs.slang_utils.utils import div_up


# Kernel launch configuration
_THREADS_PER_BLOCK = 256


def _calculate_grid_size(batch_size: int) -> int:
    """Calculate grid size for CUDA kernel launch."""
    return div_up(batch_size, _THREADS_PER_BLOCK)


# ============================================================================
# Autograd Function Wrappers for Differentiable Operations
# ============================================================================


class SE3PoseFromMatrixFunction(torch.autograd.Function):
    """Differentiable SE3 pose to 4x4 matrix conversion."""

    @staticmethod
    def forward(ctx, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = matrix.contiguous()
        N = matrix.shape[0]
        translation = torch.empty((N, 3), device=matrix.device, dtype=matrix.dtype)
        rotation = torch.empty((N, 4), device=matrix.device, dtype=matrix.dtype)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.se3pose_from_matrix_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (matrix, (matrix,)),
                (translation, (translation,)),
                (rotation, (rotation,)),
            )

        ctx.save_for_backward(matrix, translation, rotation)
        ctx.N = N
        return translation, rotation

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_translation = grad_outputs[0]
        grad_rotation = grad_outputs[1]
        matrix = ctx.saved_tensors[0]
        translation = ctx.saved_tensors[1]
        rotation = ctx.saved_tensors[2]
        grad_translation = grad_translation.contiguous()
        grad_rotation = grad_rotation.contiguous()
        grad_matrix = torch.empty_like(matrix)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.se3pose_from_matrix_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (matrix, (grad_matrix,)),
                (translation, (grad_translation,)),
                (rotation, (grad_rotation,)),
            )

        return grad_matrix


class SE3PoseToInverseMatrixFunction(torch.autograd.Function):
    """Differentiable SE3 pose to inverse 4x4 matrix conversion."""

    @staticmethod
    def forward(ctx, translation: torch.Tensor, rotation: torch.Tensor, wxyz_format: bool = False) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        N = translation.shape[0]
        result = torch.empty((N, 16), device=translation.device, dtype=translation.dtype)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.se3pose_to_inverse_matrix_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (translation,)),
                (rotation, (rotation,)),
                (result, (result,)),
                wxyz_format,
            )

        ctx.save_for_backward(translation, rotation, result)
        ctx.N = N
        ctx.wxyz_format = wxyz_format
        return result.reshape(N, 4, 4)

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, result = ctx.saved_tensors
        grad_result = grad_result.contiguous().reshape(ctx.N, 16)
        grad_translation = torch.empty_like(translation)
        grad_rotation = torch.empty_like(rotation)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.se3pose_to_inverse_matrix_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (grad_translation,)),
                (rotation, (grad_rotation,)),
                (result, (grad_result,)),
                ctx.wxyz_format,
            )

        return grad_translation, grad_rotation, None


def se3pose_from_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 4x4 transformation matrices to SE3 poses (batched, GPU accelerated, differentiable).

    Args:
        matrix: (N, 4, 4) or (N, 16) transformation matrices

    Returns:
        (N, 3) translation vectors
        (N, 4) quaternions in xyzw format
    """
    if matrix.shape[1] == 4:
        # turn matrix from (N, 4, 4) to (N, 16)
        matrix = matrix.reshape(matrix.shape[0], -1)
    return SE3PoseFromMatrixFunction.apply(matrix)


def se3pose_to_inverse_matrix(
    translation: torch.Tensor, rotation: torch.Tensor, wxyz_format: bool = False
) -> torch.Tensor:
    """Convert SE3 poses to inverse 4x4 transformation matrices (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors
        rotation: (N, 4) quaternions in xyzw format
        wxyz_format: If true, input is (w,x,y,z) and will be converted to (x,y,z,w)
    """
    return SE3PoseToInverseMatrixFunction.apply(translation, rotation, wxyz_format)


