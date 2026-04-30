// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Quaternion CUDA kernels - entry point for Python bindings via slangtorch
// This file includes dependencies and defines CUDA kernel wrappers around quaternion device functions

// Include extensions first for DiffTensorView operations
#include <slang_utils/extensions.slang>

// Include quaternion device functions
#include "quaternion.slang"

// CUDA Kernel wrappers for Python bindings
// These kernels wrap the quaternion namespace functions for use via slangtorch

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_normalize_safe_kernel(
    DiffTensorView quats_in,    // (N, 4)
    DiffTensorView quats_out    // (N, 4)
) {
    uint count = quats_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q = quats_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q_norm = quaternion::normalizeSafe(q);
    quats_out.storeVecOnce(uint2(tid, 0), q_norm);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_conjugate_kernel(
    DiffTensorView quats_in,
    DiffTensorView quats_out
) {
    uint count = quats_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q = quats_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q_conj = quaternion::conjugate(q);
    quats_out.storeVecOnce(uint2(tid, 0), q_conj);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_inverse_kernel(
    DiffTensorView quats_in,
    DiffTensorView quats_out
) {
    uint count = quats_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q = quats_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q_inv = quaternion::inverse(q);
    quats_out.storeVecOnce(uint2(tid, 0), q_inv);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_inverse_safe_kernel(
    DiffTensorView quats_in,
    DiffTensorView quats_out
) {
    uint count = quats_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q = quats_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q_inv = quaternion::inverseSafe(q);
    quats_out.storeVecOnce(uint2(tid, 0), q_inv);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_multiply_kernel(
    DiffTensorView quats1_in,
    DiffTensorView quats2_in,
    DiffTensorView quats_out
) {
    uint count = quats1_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q1 = quats1_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q2 = quats2_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q_prod = quaternion::multiply(q1, q2);
    quats_out.storeVecOnce(uint2(tid, 0), q_prod);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_rotate_vector_kernel(
    DiffTensorView quats_in,
    DiffTensorView vecs_in,
    DiffTensorView vecs_out
) {
    uint count = quats_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q = quats_in.loadVecOnce<4>(uint2(tid, 0));
    const float3 v = vecs_in.loadVecOnce<3>(uint2(tid, 0));
    const float3 v_rot = quaternion::rotateVector(q, v);
    vecs_out.storeVecOnce(uint2(tid, 0), v_rot);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_to_matrix_kernel(
    DiffTensorView quats_in,
    DiffTensorView matrices_out  // (N, 9) - flattened 3x3 matrices
) {
    uint count = quats_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q = quats_in.loadVecOnce<4>(uint2(tid, 0));
    float3x3 mat = quaternion::toMatrix(q);
    
    // Store matrix in row-major order
    matrices_out.storeMatrixOnce(uint2(tid, 0), mat);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_slerp_kernel(
    DiffTensorView quats1_in,
    DiffTensorView quats2_in,
    no_diff float t,
    DiffTensorView quats_out
) {
    uint count = quats1_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q1 = quats1_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q2 = quats2_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q_interp = quaternion::slerp(q1, q2, t);
    quats_out.storeVecOnce(uint2(tid, 0), q_interp);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_slerp_batched_kernel(
    DiffTensorView quats1_in,
    DiffTensorView quats2_in,
    DiffTensorView t_in,  // (N,) per-element interpolation parameters
    DiffTensorView quats_out
) {
    uint count = quats1_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q1 = quats1_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q2 = quats2_in.loadVecOnce<4>(uint2(tid, 0));
    float t = t_in.loadVecOnce<1>(uint2(tid, 0))[0];
    const float4 q_interp = quaternion::slerp(q1, q2, t);
    quats_out.storeVecOnce(uint2(tid, 0), q_interp);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_lerp_kernel(
    DiffTensorView quats1_in,
    DiffTensorView quats2_in,
    no_diff float t,
    DiffTensorView quats_out
) {
    uint count = quats1_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q1 = quats1_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q2 = quats2_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q_interp = quaternion::lerp(q1, q2, t);
    quats_out.storeVecOnce(uint2(tid, 0), q_interp);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_from_axis_angle_kernel(
    DiffTensorView axes_in,
    DiffTensorView angles_in,
    DiffTensorView quats_out
) {
    uint count = axes_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float3 axis = axes_in.loadVecOnce<3>(uint2(tid, 0));
    float angle = angles_in.loadVecOnce<1>(uint2(tid, 0))[0];
    const float4 q = quaternion::convertAngleToQuat(axis, angle);
    quats_out.storeVecOnce(uint2(tid, 0), q);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_angular_distance_kernel(
    DiffTensorView quats1_in,
    DiffTensorView quats2_in,
    DiffTensorView distances_out
) {
    uint count = quats1_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    
    const float4 q1 = quats1_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q2 = quats2_in.loadVecOnce<4>(uint2(tid, 0));
    float dist = quaternion::angular_distance(q1, q2);
    distances_out.storeVecOnce(uint2(tid, 0), dist);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void quat_manifold_interp_kernel(
    DiffTensorView quats1_in,
    DiffTensorView quats2_in,
    no_diff float t,
    DiffTensorView quats_out
) {
    uint count = quats1_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;

    const float4 q1 = quats1_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q2 = quats2_in.loadVecOnce<4>(uint2(tid, 0));
    const float4 q_interp = quaternion::manifoldInterp(q1, q2, t);
    quats_out.storeVecOnce(uint2(tid, 0), q_interp);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void so3_exp_kernel(
    DiffTensorView omega_in,    // (N, 3)
    DiffTensorView quats_out    // (N, 4)
) {
    uint count = omega_in.size(0);
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;

    const float3 omega = omega_in.loadVecOnce<3>(uint2(tid, 0));
    const float4 q = quaternion::so3Exp(omega);
    quats_out.storeVecOnce(uint2(tid, 0), q);
}
