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


class SE3PoseTransformPointFunction(torch.autograd.Function):
    """Differentiable SE3 point transformation."""

    @staticmethod
    def forward(ctx, translation: torch.Tensor, rotation: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        point = point.contiguous()
        N = point.shape[0]
        result = torch.empty_like(point)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.se3pose_transform_point_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (translation,)),
                (rotation, (rotation,)),
                (point, (point,)),
                (result, (result,)),
            )

        ctx.save_for_backward(translation, rotation, point, result)
        ctx.N = N
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, point, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_translation = torch.empty_like(translation)
        grad_rotation = torch.empty_like(rotation)
        grad_point = torch.empty_like(point)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.se3pose_transform_point_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (grad_translation,)),
                (rotation, (grad_rotation,)),
                (point, (grad_point,)),
                (result, (grad_result,)),
            )

        return grad_translation, grad_rotation, grad_point


class SE3PoseTransformDirectionFunction(torch.autograd.Function):
    """Differentiable SE3 direction transformation."""

    @staticmethod
    def forward(ctx, translation: torch.Tensor, rotation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        direction = direction.contiguous()
        N = direction.shape[0]
        result = torch.empty_like(direction)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.se3pose_transform_direction_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (translation,)),
                (rotation, (rotation,)),
                (direction, (direction,)),
                (result, (result,)),
            )

        ctx.save_for_backward(translation, rotation, direction, result)
        ctx.N = N
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, direction, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_translation = torch.empty_like(translation)
        grad_rotation = torch.empty_like(rotation)
        grad_direction = torch.empty_like(direction)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.se3pose_transform_direction_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (grad_translation,)),
                (rotation, (grad_rotation,)),
                (direction, (grad_direction,)),
                (result, (grad_result,)),
            )

        return grad_translation, grad_rotation, grad_direction


class SE3PoseInverseTransformPointFunction(torch.autograd.Function):
    """Differentiable SE3 inverse point transformation."""

    @staticmethod
    def forward(ctx, translation: torch.Tensor, rotation: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        point = point.contiguous()
        N = point.shape[0]
        result = torch.empty_like(point)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.se3pose_inverse_transform_point_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (translation,)),
                (rotation, (rotation,)),
                (point, (point,)),
                (result, (result,)),
            )

        ctx.save_for_backward(translation, rotation, point, result)
        ctx.N = N
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, point, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_translation = torch.empty_like(translation)
        grad_rotation = torch.empty_like(rotation)
        grad_point = torch.empty_like(point)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.se3pose_inverse_transform_point_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (grad_translation,)),
                (rotation, (grad_rotation,)),
                (point, (grad_point,)),
                (result, (grad_result,)),
            )

        return grad_translation, grad_rotation, grad_point


class SE3PoseInverseTransformDirectionFunction(torch.autograd.Function):
    """Differentiable SE3 inverse direction transformation."""

    @staticmethod
    def forward(ctx, translation: torch.Tensor, rotation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        direction = direction.contiguous()
        N = direction.shape[0]
        result = torch.empty_like(direction)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.se3pose_inverse_transform_direction_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (translation,)),
                (rotation, (rotation,)),
                (direction, (direction,)),
                (result, (result,)),
            )

        ctx.save_for_backward(translation, rotation, direction, result)
        ctx.N = N
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, direction, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_translation = torch.empty_like(translation)
        grad_rotation = torch.empty_like(rotation)
        grad_direction = torch.empty_like(direction)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.se3pose_inverse_transform_direction_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (grad_translation,)),
                (rotation, (grad_rotation,)),
                (direction, (grad_direction,)),
                (result, (grad_result,)),
            )

        return grad_translation, grad_rotation, grad_direction


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


class SE3PoseToMatrixFunction(torch.autograd.Function):
    """Differentiable SE3 pose to 4x4 matrix conversion."""

    @staticmethod
    def forward(ctx, translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        N = translation.shape[0]
        result = torch.empty((N, 16), device=translation.device, dtype=translation.dtype)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.se3pose_to_matrix_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (translation,)),
                (rotation, (rotation,)),
                (result, (result,)),
            )

        ctx.save_for_backward(translation, rotation, result)
        ctx.N = N
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
            pose_kernels_slang.se3pose_to_matrix_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (translation, (grad_translation,)),
                (rotation, (grad_rotation,)),
                (result, (grad_result,)),
            )

        return grad_translation, grad_rotation


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


class TrajectoryTransformPoint2PosesFunction(torch.autograd.Function):
    """Differentiable 2-pose trajectory point transformation."""

    @staticmethod
    def forward(
        ctx,
        trans0: torch.Tensor,
        rot0: torch.Tensor,
        time0: torch.Tensor,
        trans1: torch.Tensor,
        rot1: torch.Tensor,
        time1: torch.Tensor,
        point: torch.Tensor,
        query_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans0 = trans0.contiguous()
        rot0 = rot0.contiguous()
        time0 = time0.contiguous()
        trans1 = trans1.contiguous()
        rot1 = rot1.contiguous()
        time1 = time1.contiguous()
        point = point.contiguous()
        query_time = query_time.contiguous()
        N = point.shape[0]
        result_point = torch.empty_like(point)
        result_out_of_bounds = torch.empty((N,), device=point.device, dtype=torch.float32)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.trajectory_transform_point_2poses_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (trans0, (trans0,)),
                (rot0, (rot0,)),
                (time0, (time0,)),
                (trans1, (trans1,)),
                (rot1, (rot1,)),
                (time1, (time1,)),
                (point, (point,)),
                (query_time, (query_time,)),
                (result_point, (result_point,)),
                (result_out_of_bounds, (result_out_of_bounds,)),
            )

        ctx.save_for_backward(
            trans0, rot0, time0, trans1, rot1, time1, point, query_time, result_point, result_out_of_bounds
        )
        ctx.N = N
        return result_point, result_out_of_bounds

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result_point = grad_outputs[0]
        # grad_result_out_of_bounds = grad_outputs[1]  # out_of_bounds is no_diff
        trans0, rot0, time0, trans1, rot1, time1, point, query_time, result_point, result_out_of_bounds = (
            ctx.saved_tensors
        )
        grad_result_point = grad_result_point.contiguous()
        grad_trans0 = torch.empty_like(trans0)
        grad_rot0 = torch.empty_like(rot0)
        grad_time0 = torch.empty_like(time0)
        grad_trans1 = torch.empty_like(trans1)
        grad_rot1 = torch.empty_like(rot1)
        grad_time1 = torch.empty_like(time1)
        grad_point = torch.empty_like(point)
        grad_query_time = torch.empty_like(query_time)
        grad_result_out_of_bounds = torch.empty_like(result_out_of_bounds)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.trajectory_transform_point_2poses_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (trans0, (grad_trans0,)),
                (rot0, (grad_rot0,)),
                (time0, (grad_time0,)),
                (trans1, (grad_trans1,)),
                (rot1, (grad_rot1,)),
                (time1, (grad_time1,)),
                (point, (grad_point,)),
                (query_time, (grad_query_time,)),
                (result_point, (grad_result_point,)),
                (result_out_of_bounds, (grad_result_out_of_bounds,)),
            )

        return grad_trans0, grad_rot0, grad_time0, grad_trans1, grad_rot1, grad_time1, grad_point, grad_query_time


class TrajectoryGetRotation2PosesFunction(torch.autograd.Function):
    """Differentiable 2-pose trajectory rotation query."""

    @staticmethod
    def forward(
        ctx,
        trans0: torch.Tensor,
        rot0: torch.Tensor,
        time0: torch.Tensor,
        trans1: torch.Tensor,
        rot1: torch.Tensor,
        time1: torch.Tensor,
        query_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans0 = trans0.contiguous()
        rot0 = rot0.contiguous()
        time0 = time0.contiguous()
        trans1 = trans1.contiguous()
        rot1 = rot1.contiguous()
        time1 = time1.contiguous()
        query_time = query_time.contiguous()
        N = trans0.shape[0]
        result_quat = torch.empty((N, 4), device=trans0.device, dtype=torch.float32)
        result_out_of_bounds = torch.empty((N,), device=trans0.device, dtype=torch.float32)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.trajectory_get_rotation_2poses_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (trans0, (trans0,)),
                (rot0, (rot0,)),
                (time0, (time0,)),
                (trans1, (trans1,)),
                (rot1, (rot1,)),
                (time1, (time1,)),
                (query_time, (query_time,)),
                (result_quat, (result_quat,)),
                (result_out_of_bounds, (result_out_of_bounds,)),
            )

        ctx.save_for_backward(trans0, rot0, time0, trans1, rot1, time1, query_time, result_quat, result_out_of_bounds)
        ctx.N = N
        return result_quat, result_out_of_bounds

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result_quat = grad_outputs[0]
        # grad_result_out_of_bounds = grad_outputs[1]  # out_of_bounds is no_diff
        trans0, rot0, time0, trans1, rot1, time1, query_time, result_quat, result_out_of_bounds = ctx.saved_tensors
        grad_result_quat = grad_result_quat.contiguous()
        grad_trans0 = torch.empty_like(trans0)
        grad_rot0 = torch.empty_like(rot0)
        grad_time0 = torch.empty_like(time0)
        grad_trans1 = torch.empty_like(trans1)
        grad_rot1 = torch.empty_like(rot1)
        grad_time1 = torch.empty_like(time1)
        grad_query_time = torch.empty_like(query_time)
        grad_result_out_of_bounds = torch.empty_like(result_out_of_bounds)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.trajectory_get_rotation_2poses_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (trans0, (grad_trans0,)),
                (rot0, (grad_rot0,)),
                (time0, (grad_time0,)),
                (trans1, (grad_trans1,)),
                (rot1, (grad_rot1,)),
                (time1, (grad_time1,)),
                (query_time, (grad_query_time,)),
                (result_quat, (grad_result_quat,)),
                (result_out_of_bounds, (grad_result_out_of_bounds,)),
            )

        return grad_trans0, grad_rot0, grad_time0, grad_trans1, grad_rot1, grad_time1, grad_query_time


class TrajectoryTransformPoint1PoseFunction(torch.autograd.Function):
    """Differentiable 1-pose trajectory point transformation."""

    @staticmethod
    def forward(
        ctx,
        trans: torch.Tensor,
        rot: torch.Tensor,
        time: torch.Tensor,
        point: torch.Tensor,
        query_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans = trans.contiguous()
        rot = rot.contiguous()
        time = time.contiguous()
        point = point.contiguous()
        query_time = query_time.contiguous()
        N = point.shape[0]
        result_point = torch.empty_like(point)
        result_out_of_bounds = torch.empty((N,), device=point.device, dtype=torch.float32)

        blocks = _calculate_grid_size(N)
        if blocks > 0:
            pose_kernels_slang.trajectory_transform_point_1pose_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (trans, (trans,)),
                (rot, (rot,)),
                (time, (time,)),
                (point, (point,)),
                (query_time, (query_time,)),
                (result_point, (result_point,)),
                (result_out_of_bounds, (result_out_of_bounds,)),
            )

        ctx.save_for_backward(trans, rot, time, point, query_time, result_point, result_out_of_bounds)
        ctx.N = N
        return result_point, result_out_of_bounds

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result_point = grad_outputs[0]
        # grad_result_out_of_bounds = grad_outputs[1]  # out_of_bounds is no_diff
        trans, rot, time, point, query_time, result_point, result_out_of_bounds = ctx.saved_tensors
        grad_result_point = grad_result_point.contiguous()
        grad_trans = torch.empty_like(trans)
        grad_rot = torch.empty_like(rot)
        grad_time = torch.empty_like(time)
        grad_point = torch.empty_like(point)
        grad_query_time = torch.empty_like(query_time)
        grad_result_out_of_bounds = torch.empty_like(result_out_of_bounds)

        blocks = _calculate_grid_size(ctx.N)
        if blocks > 0:
            pose_kernels_slang.trajectory_transform_point_1pose_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (trans, (grad_trans,)),
                (rot, (grad_rot,)),
                (time, (grad_time,)),
                (point, (grad_point,)),
                (query_time, (grad_query_time,)),
                (result_point, (grad_result_point,)),
                (result_out_of_bounds, (grad_result_out_of_bounds,)),
            )

        return grad_trans, grad_rot, grad_time, grad_point, grad_query_time


# ============================================================================
# Public API - Differentiable Wrapper Functions
# ============================================================================


def se3pose_transform_point(translation: torch.Tensor, rotation: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Transform points using SE3 poses (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors
        rotation: (N, 4) quaternions in xyzw format
        point: (N, 3) points to transform

    Returns:
        (N, 3) transformed points
    """
    return SE3PoseTransformPointFunction.apply(translation, rotation, point)


def se3pose_transform_direction(
    translation: torch.Tensor, rotation: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Transform directions using SE3 poses (rotation only, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors (not used for directions)
        rotation: (N, 4) quaternions in xyzw format
        direction: (N, 3) directions to transform

    Returns:
        (N, 3) transformed directions
    """
    return SE3PoseTransformDirectionFunction.apply(translation, rotation, direction)


def se3pose_inverse_transform_point(
    translation: torch.Tensor, rotation: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    """Apply inverse SE3 transformation to points (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors
        rotation: (N, 4) quaternions in xyzw format
        point: (N, 3) points to transform

    Returns:
        (N, 3) inverse transformed points
    """
    return SE3PoseInverseTransformPointFunction.apply(translation, rotation, point)


def se3pose_inverse_transform_direction(
    translation: torch.Tensor, rotation: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Apply inverse SE3 transformation to directions (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors (not used for directions)
        rotation: (N, 4) quaternions in xyzw format
        direction: (N, 3) directions to transform

    Returns:
        (N, 3) inverse transformed directions
    """
    return SE3PoseInverseTransformDirectionFunction.apply(translation, rotation, direction)


def se3pose_to_matrix(translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Convert SE3 poses to 4x4 transformation matrices (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors
        rotation: (N, 4) quaternions in xyzw format

    Returns:
        (N, 4, 4) transformation matrices
    """
    return SE3PoseToMatrixFunction.apply(translation, rotation)


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


def trajectory_transform_point_2poses(
    trans0: torch.Tensor,
    rot0: torch.Tensor,
    time0: torch.Tensor,
    trans1: torch.Tensor,
    rot1: torch.Tensor,
    time1: torch.Tensor,
    point: torch.Tensor,
    query_time: torch.Tensor,
) -> dict:
    """Transform points using 2-pose trajectories (with extrapolation, batched, GPU accelerated, differentiable).

    Args:
        trans0: (N, 3) translations of first poses
        rot0: (N, 4) rotations of first poses
        time0: (N,) times of first poses
        trans1: (N, 3) translations of second poses
        rot1: (N, 4) rotations of second poses
        time1: (N,) times of second poses
        point: (N, 3) points to transform
        query_time: (N,) times to query trajectories at

    Returns:
        dict with 'point' (N, 3) tensor and 'out_of_bounds' (N,) boolean tensor
    """
    result_point, result_out_of_bounds = TrajectoryTransformPoint2PosesFunction.apply(
        trans0, rot0, time0, trans1, rot1, time1, point, query_time
    )
    return {"point": result_point, "out_of_bounds": result_out_of_bounds > 0.5}


def trajectory_get_rotation_2poses(
    trans0: torch.Tensor,
    rot0: torch.Tensor,
    time0: torch.Tensor,
    trans1: torch.Tensor,
    rot1: torch.Tensor,
    time1: torch.Tensor,
    query_time: torch.Tensor,
) -> dict:
    """Get rotations from 2-pose trajectories (with extrapolation, batched, GPU accelerated, differentiable).

    Args:
        trans0: (N, 3) translations of first poses
        rot0: (N, 4) rotations of first poses
        time0: (N,) times of first poses
        trans1: (N, 3) translations of second poses
        rot1: (N, 4) rotations of second poses
        time1: (N,) times of second poses
        query_time: (N,) times to query trajectories at

    Returns:
        dict with 'quat' (N, 4) tensor and 'out_of_bounds' (N,) boolean tensor
    """
    result_quat, result_out_of_bounds = TrajectoryGetRotation2PosesFunction.apply(
        trans0, rot0, time0, trans1, rot1, time1, query_time
    )
    return {"quat": result_quat, "out_of_bounds": result_out_of_bounds > 0.5}


def trajectory_transform_point_1pose(
    trans: torch.Tensor, rot: torch.Tensor, time: torch.Tensor, point: torch.Tensor, query_time: torch.Tensor
) -> dict:
    """Transform points using 1-pose trajectories (batched, GPU accelerated, differentiable).

    With only one pose, always returns that pose (can't interpolate/extrapolate).

    Args:
        trans: (N, 3) translations of poses
        rot: (N, 4) rotations of poses
        time: (N,) times of poses
        point: (N, 3) points to transform
        query_time: (N,) times to query trajectories at

    Returns:
        dict with 'point' (N, 3) tensor and 'out_of_bounds' (N,) boolean tensor
    """
    result_point, result_out_of_bounds = TrajectoryTransformPoint1PoseFunction.apply(
        trans, rot, time, point, query_time
    )
    return {"point": result_point, "out_of_bounds": result_out_of_bounds > 0.5}


def frame_transform_poses_tquat(
    tquat_poses: torch.Tensor,
    rotation: tuple[float, float, float, float],
    translation: tuple[float, float, float],
    scale: float,
) -> torch.Tensor:
    assert tquat_poses.ndim == 2 and tquat_poses.shape[-1] == 7, (
        f"Expected tquat_poses to have shape (N, 7), got {tquat_poses.shape}"
    )
    assert tquat_poses.device != "cpu", "frame_transform_poses_tquat does not support CPU tensors"
    tquat_poses = tquat_poses.contiguous()
    N = tquat_poses.shape[0]
    result = torch.empty_like(tquat_poses)

    blocks = div_up(N, _THREADS_PER_BLOCK)
    if blocks > 0:
        pose_kernels_slang.frame_transform_poses_tquat_kernel(
            (_THREADS_PER_BLOCK, 1, 1),
            (blocks, 1, 1),
            tquat_poses,
            result,
            rotation,
            translation,
            scale,
        )

    return result
