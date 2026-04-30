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


class QuatNormalizeSafeFunction(torch.autograd.Function):
    """Normalize quaternion(s) with safety checks."""

    @staticmethod
    def forward(ctx, quat: Tensor) -> Tensor:
        quat = quat.contiguous()
        count = quat.shape[0]
        result = torch.empty_like(quat)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_normalize_safe_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (quat, (quat,)),
                (result, (result,)),
            )

        ctx.save_for_backward(quat, result)
        ctx.count = count
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        quat, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_quat = torch.empty_like(quat)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_normalize_safe_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (quat, (grad_quat,)),
                (result, (grad_result,)),
            )

        return grad_quat


class QuatConjugateFunction(torch.autograd.Function):
    """Compute quaternion conjugate."""

    @staticmethod
    def forward(ctx, quat: Tensor) -> Tensor:
        quat = quat.contiguous()
        count = quat.shape[0]
        result = torch.empty_like(quat)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_conjugate_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (quat, (quat,)),
                (result, (result,)),
            )

        ctx.save_for_backward(quat, result)
        ctx.count = count
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        quat, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_quat = torch.empty_like(quat)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_conjugate_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (quat, (grad_quat,)),
                (result, (grad_result,)),
            )

        return grad_quat


class QuatMultiplyFunction(torch.autograd.Function):
    """Multiply two quaternions using Hamilton product."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        count = q1.shape[0]
        result = torch.empty_like(q1)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_multiply_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (q1,)),
                (q2, (q2,)),
                (result, (result,)),
            )

        ctx.save_for_backward(q1, q2, result)
        ctx.count = count
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
            quaternion_slang.quat_multiply_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (grad_q1,)),
                (q2, (grad_q2,)),
                (result, (grad_result,)),
            )

        return grad_q1, grad_q2


class QuatRotateVectorFunction(torch.autograd.Function):
    """Apply quaternion rotation to vector(s)."""

    @staticmethod
    def forward(ctx, quat: Tensor, vec: Tensor) -> Tensor:
        quat = quat.contiguous()
        vec = vec.contiguous()
        count = vec.shape[0]
        result = torch.empty_like(vec)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_rotate_vector_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (quat, (quat,)),
                (vec, (vec,)),
                (result, (result,)),
            )

        ctx.save_for_backward(quat, vec, result)
        ctx.count = count
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        quat, vec, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_quat = torch.empty_like(quat)
        grad_vec = torch.empty_like(vec)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_rotate_vector_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (quat, (grad_quat,)),
                (vec, (grad_vec,)),
                (result, (grad_result,)),
            )

        return grad_quat, grad_vec


class QuatToMatrixFunction(torch.autograd.Function):
    """Convert quaternion(s) to 3x3 rotation matrix/matrices."""

    @staticmethod
    def forward(ctx, quat: Tensor) -> Tensor:
        quat = quat.contiguous()
        count = quat.shape[0]
        result_flat = torch.empty(count, 9, dtype=quat.dtype, device=quat.device)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_to_matrix_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (quat, (quat,)),
                (result_flat, (result_flat,)),
            )

        result = result_flat.reshape(count, 3, 3)
        ctx.save_for_backward(quat, result_flat)
        ctx.count = count
        ctx.original_shape = quat.shape[:-1]
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        quat, result_flat = ctx.saved_tensors
        grad_result_flat = grad_result.reshape(ctx.count, 9).contiguous()
        grad_quat = torch.empty_like(quat)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_to_matrix_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (quat, (grad_quat,)),
                (result_flat, (grad_result_flat,)),
            )

        return grad_quat


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


class QuatSlerpBatchedFunction(torch.autograd.Function):
    """Batched spherical linear interpolation with per-element t values."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        t = t.contiguous()
        count = q1.shape[0]
        result = torch.empty_like(q1)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_slerp_batched_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (q1,)),
                (q2, (q2,)),
                (t, (t,)),
                (result, (result,)),
            )

        ctx.save_for_backward(q1, q2, t, result)
        ctx.count = count
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        q1, q2, t, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_q1 = torch.empty_like(q1)
        grad_q2 = torch.empty_like(q2)
        grad_t = torch.empty_like(t)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_slerp_batched_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (grad_q1,)),
                (q2, (grad_q2,)),
                (t, (grad_t,)),
                (result, (grad_result,)),
            )

        return grad_q1, grad_q2, grad_t


class QuatLerpFunction(torch.autograd.Function):
    """Linear interpolation between quaternions."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor, t: float) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        count = q1.shape[0]
        result = torch.empty_like(q1)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_lerp_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (q1,)),
                (q2, (q2,)),
                t,
                (result, (result,)),
            )

        ctx.save_for_backward(q1, q2, result)
        ctx.t = t
        ctx.count = count
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
            quaternion_slang.quat_lerp_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (grad_q1,)),
                (q2, (grad_q2,)),
                ctx.t,
                (result, (grad_result,)),
            )

        return grad_q1, grad_q2, None


class QuatFromAxisAngleFunction(torch.autograd.Function):
    """Convert axis-angle representation to quaternion."""

    @staticmethod
    def forward(ctx, axis: Tensor, angle: Tensor) -> Tensor:
        axis = axis.contiguous()
        angle = angle.contiguous()
        count = axis.shape[0]
        quat = torch.empty(count, 4, dtype=axis.dtype, device=axis.device)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_from_axis_angle_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (axis, (axis,)),
                (angle, (angle,)),
                (quat, (quat,)),
            )

        ctx.save_for_backward(axis, angle, quat)
        ctx.count = count
        return quat

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_quat = grad_outputs[0]
        axis, angle, quat = ctx.saved_tensors
        grad_quat = grad_quat.contiguous()
        grad_axis = torch.empty_like(axis)
        grad_angle = torch.empty_like(angle)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_from_axis_angle_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (axis, (grad_axis,)),
                (angle, (grad_angle,)),
                (quat, (grad_quat,)),
            )

        return grad_axis, grad_angle


class QuatAngularDistanceFunction(torch.autograd.Function):
    """Compute angular distance between quaternions."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        count = q1.shape[0]
        distance = torch.empty(count, 1, dtype=q1.dtype, device=q1.device)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_angular_distance_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (q1,)),
                (q2, (q2,)),
                (distance, (distance,)),
            )

        ctx.save_for_backward(q1, q2, distance)
        ctx.count = count
        return distance.squeeze(-1)

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_distance = grad_outputs[0].unsqueeze(-1)
        q1, q2, distance = ctx.saved_tensors
        grad_distance = grad_distance.contiguous()
        grad_q1 = torch.empty_like(q1)
        grad_q2 = torch.empty_like(q2)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_angular_distance_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (grad_q1,)),
                (q2, (grad_q2,)),
                (distance, (grad_distance,)),
            )

        return grad_q1, grad_q2


class SO3ExpFunction(torch.autograd.Function):
    """SO(3) exponential map: angular velocity vector to quaternion."""

    @staticmethod
    def forward(ctx, omega: Tensor) -> Tensor:
        omega = omega.contiguous()
        count = omega.shape[0]
        result = torch.empty(count, 4, dtype=omega.dtype, device=omega.device)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.so3_exp_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (omega, (omega,)),
                (result, (result,)),
            )

        ctx.save_for_backward(omega, result)
        ctx.count = count
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        omega, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_omega = torch.empty_like(omega)

        blocks = div_up(ctx.count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.so3_exp_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (omega, (grad_omega,)),
                (result, (grad_result,)),
            )

        return grad_omega


class QuatManifoldInterpFunction(torch.autograd.Function):
    """Manifold interpolation using SO(3) operations."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor, t: float) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        count = q1.shape[0]
        result = torch.empty_like(q1)

        blocks = div_up(count, _THREADS_PER_BLOCK)
        if blocks > 0:
            quaternion_slang.quat_manifold_interp_kernel(
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
            quaternion_slang.quat_manifold_interp_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (q1, (grad_q1,)),
                (q2, (grad_q2,)),
                ctx.t,
                (result, (grad_result,)),
            )

        return grad_q1, grad_q2, None  # None for t (not differentiable)


# ============================================================================
# Public API - User-Friendly Functions with Batch Shape Support
# ============================================================================


def quat_normalize_safe(quat: Tensor) -> Tensor:
    """Normalize quaternion(s) with safety checks (GPU accelerated, differentiable).

    Handles near-zero quaternions by returning identity quaternion.

    Args:
        quat: (..., 4) quaternion(s) in xyzw format

    Returns:
        (..., 4) normalized quaternion(s)
    """
    original_shape = quat.shape
    quat_flat = quat.reshape(-1, 4)
    result_flat = QuatNormalizeSafeFunction.apply(quat_flat)
    return result_flat.reshape(original_shape)


def quat_conjugate(q: Tensor) -> Tensor:
    """Compute quaternion conjugate (GPU accelerated, differentiable).

    Args:
        q: (..., 4) quaternion(s) in xyzw format

    Returns:
        (..., 4) conjugate quaternion(s)
    """
    original_shape = q.shape
    q_flat = q.reshape(-1, 4)
    result_flat = QuatConjugateFunction.apply(q_flat)
    return result_flat.reshape(original_shape)


def quat_inverse(q: Tensor) -> Tensor:
    """Compute quaternion inverse (GPU accelerated, differentiable).

    For unit quaternions, inverse equals conjugate.

    Args:
        q: (..., 4) quaternion(s) in xyzw format (assumed normalized)

    Returns:
        (..., 4) inverse quaternion(s)
    """
    return quat_conjugate(q)


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Multiply two quaternions using Hamilton product (GPU accelerated, differentiable).

    Result is q1 * q2 (apply q2 first, then q1).

    Args:
        q1: (..., 4) first quaternion(s) in xyzw format
        q2: (..., 4) second quaternion(s) in xyzw format

    Returns:
        (..., 4) product quaternion(s)
    """
    assert q1.shape == q2.shape, f"Shape mismatch: {q1.shape} vs {q2.shape}"

    original_shape = q1.shape
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    result_flat = QuatMultiplyFunction.apply(q1_flat, q2_flat)
    return result_flat.reshape(original_shape)


def quat_rotate_vector(q: Tensor, v: Tensor) -> Tensor:
    """Apply quaternion rotation to vector(s) (GPU accelerated, differentiable).

    Uses the efficient formula: q * v * q^(-1)

    Args:
        q: (..., 4) quaternion(s) in xyzw format
        v: (..., 3) vector(s) to rotate

    Returns:
        (..., 3) rotated vector(s)
    """
    assert q.shape[:-1] == v.shape[:-1], f"Batch shape mismatch: {q.shape[:-1]} vs {v.shape[:-1]}"

    original_shape = v.shape
    q_flat = q.reshape(-1, 4)
    v_flat = v.reshape(-1, 3)
    result_flat = QuatRotateVectorFunction.apply(q_flat, v_flat)
    return result_flat.reshape(original_shape)


def quat_to_matrix(quat: Tensor) -> Tensor:
    """Convert quaternion(s) to 3x3 rotation matrix/matrices (GPU accelerated, differentiable).

    Args:
        quat: (..., 4) quaternion(s) in xyzw format

    Returns:
        (..., 3, 3) rotation matrix/matrices
    """
    original_shape = quat.shape[:-1]
    quat_flat = quat.reshape(-1, 4)
    result = QuatToMatrixFunction.apply(quat_flat)
    return result.reshape(*original_shape, 3, 3)


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


def quat_slerp_batched(q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
    """Batched spherical linear interpolation with per-element t (GPU accelerated, differentiable).

    Unlike quat_slerp which only supports scalar t, this supports per-element interpolation
    parameters for batched operations.

    Args:
        q1: (N, 4) start quaternions in xyzw format
        q2: (N, 4) end quaternions in xyzw format
        t: (N,) interpolation parameters in [0, 1]

    Returns:
        (N, 4) interpolated quaternions
    """
    assert q1.shape == q2.shape, f"Shape mismatch: {q1.shape} vs {q2.shape}"
    assert q1.shape[0] == t.shape[0], f"Batch size mismatch: {q1.shape[0]} vs {t.shape[0]}"

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    t = t.reshape(-1, 1)  # (N, 1) for kernel
    return QuatSlerpBatchedFunction.apply(q1, q2, t)


def quat_lerp(q1: Tensor, q2: Tensor, t: float) -> Tensor:
    """Linear interpolation between quaternions (GPU accelerated, differentiable).

    Simple linear interpolation in quaternion space. Note: does NOT produce
    unit quaternions unless inputs are unit and t is 0 or 1.

    Args:
        q1: (..., 4) start quaternion(s) in xyzw format
        q2: (..., 4) end quaternion(s) in xyzw format
        t: Interpolation parameter [0, 1]

    Returns:
        (..., 4) interpolated quaternion(s)
    """
    original_shape = q1.shape
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    result_flat = QuatLerpFunction.apply(q1_flat, q2_flat, t)
    return result_flat.reshape(original_shape)


def quat_from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    """Convert axis-angle representation to quaternion (GPU accelerated, differentiable).

    Args:
        axis: (..., 3) rotation axis (does not need to be normalized)
        angle: (...,) rotation angle in radians

    Returns:
        (..., 4) quaternion in xyzw format
    """
    original_shape = axis.shape[:-1]
    axis_flat = axis.reshape(-1, 3)
    angle_flat = angle.reshape(-1, 1) if angle.dim() > 0 else angle.unsqueeze(0).unsqueeze(1)
    result_flat = QuatFromAxisAngleFunction.apply(axis_flat, angle_flat)
    return result_flat.reshape(*original_shape, 4)


def quat_angular_distance(q1: Tensor, q2: Tensor) -> Tensor:
    """Compute angular distance between quaternions (GPU accelerated, differentiable).

    Returns the geodesic distance on the unit quaternion manifold.

    Args:
        q1: (..., 4) quaternion(s) in xyzw format
        q2: (..., 4) quaternion(s) in xyzw format

    Returns:
        (...,) angular distance in radians
    """
    original_shape = q1.shape[:-1]
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    result_flat = QuatAngularDistanceFunction.apply(q1_flat, q2_flat)
    return result_flat.reshape(original_shape)


def quat_identity(
    shape: tuple = (), dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")
) -> Tensor:
    """Create identity quaternion(s).

    Args:
        shape: Shape of the batch dimensions
        dtype: Data type
        device: Device to create tensor on (defaults to CUDA)

    Returns:
        (*shape, 4) identity quaternion(s) [0, 0, 0, 1]
    """
    quat = torch.zeros(shape + (4,), dtype=dtype, device=device)
    quat[..., 3] = 1.0
    return quat


def so3_exp(omega: Tensor) -> Tensor:
    """SO(3) exponential map: angular velocity vector to quaternion (GPU accelerated, differentiable).

    Args:
        omega: (..., 3) angular velocity vector(s)

    Returns:
        (..., 4) quaternion(s) in xyzw format
    """
    original_shape = omega.shape[:-1]
    omega_flat = omega.reshape(-1, 3)
    result_flat = SO3ExpFunction.apply(omega_flat)
    return result_flat.reshape(*original_shape, 4)


def quat_manifold_interp(q1: Tensor, q2: Tensor, t: float | Tensor) -> Tensor:
    """Manifold interpolation using SO(3) operations (GPU accelerated, differentiable).

    Composed operation: q1 * exp(t * log(q1^-1 * q2))
    Uses a single optimized kernel instead of multiple operations.

    Args:
        q1: (..., 4) start quaternion(s) in xyzw format
        q2: (..., 4) end quaternion(s) in xyzw format
        t: Interpolation parameter [0, 1] (scalar only for now)

    Returns:
        (..., 4) interpolated quaternion(s)
    """
    if not isinstance(t, (int, float)):
        raise ValueError("Manifold interpolation currently only supports scalar t parameter")

    original_shape = q1.shape
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    result_flat = QuatManifoldInterpFunction.apply(q1_flat, q2_flat, float(t))
    return result_flat.reshape(original_shape)
