// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Test kernels for pose functionality
// These kernels provide CUDA kernel wrappers to test pose functionality from Python via slangtorch

#include <slang_utils/extensions.slang>
#include "pose.slang"

// ============================================================================
// SE3Pose CUDA Kernels
// ============================================================================

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void se3pose_transform_point_kernel(
    DiffTensorView translation,
    DiffTensorView rotation,
    DiffTensorView point,
    DiffTensorView result
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= point.size(0)) return;
    
    float3 trans = translation.loadVecOnce<3>(uint2(idx, 0));
    float4 rot = rotation.loadVecOnce<4>(uint2(idx, 0));
    float3 pt = point.loadVecOnce<3>(uint2(idx, 0));
    
    float3 res = se3pose_transform_point(trans, rot, pt);
    
    result.storeVecOnce(uint2(idx, 0), res);
}

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void se3pose_transform_direction_kernel(
    DiffTensorView translation,
    DiffTensorView rotation,
    DiffTensorView direction,
    DiffTensorView result
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= direction.size(0)) return;
    
    float3 trans = translation.loadVecOnce<3>(uint2(idx, 0));
    float4 rot = rotation.loadVecOnce<4>(uint2(idx, 0));
    float3 dir = direction.loadVecOnce<3>(uint2(idx, 0));
    
    float3 res = se3pose_transform_direction(trans, rot, dir);
    
    result.storeVecOnce(uint2(idx, 0), res);
}

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void se3pose_inverse_transform_point_kernel(
    DiffTensorView translation,
    DiffTensorView rotation,
    DiffTensorView point,
    DiffTensorView result
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= point.size(0)) return;
    
    float3 trans = translation.loadVecOnce<3>(uint2(idx, 0));
    float4 rot = rotation.loadVecOnce<4>(uint2(idx, 0));
    float3 pt = point.loadVecOnce<3>(uint2(idx, 0));
    
    float3 res = se3pose_inverse_transform_point(trans, rot, pt);
    
    result.storeVecOnce(uint2(idx, 0), res);
}

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void se3pose_inverse_transform_direction_kernel(
    DiffTensorView translation,
    DiffTensorView rotation,
    DiffTensorView direction,
    DiffTensorView result
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= direction.size(0)) return;
    
    float3 trans = translation.loadVecOnce<3>(uint2(idx, 0));
    float4 rot = rotation.loadVecOnce<4>(uint2(idx, 0));
    float3 dir = direction.loadVecOnce<3>(uint2(idx, 0));
    
    float3 res = se3pose_inverse_transform_direction(trans, rot, dir);
    
    result.storeVecOnce(uint2(idx, 0), res);
}


[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void se3pose_to_matrix_kernel(
    DiffTensorView translation, // (N, 3)
    DiffTensorView rotation,    // (N, 4)
    DiffTensorView result       // (N, 16)
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= translation.size(0)) return;
    
    float3 trans = translation.loadVecOnce<3>(uint2(idx, 0));
    float4 rot = rotation.loadVecOnce<4>(uint2(idx, 0));
    
    float4x4 mat = se3pose_to_matrix(trans, rot);
    
    // Store as 4x4 matrix
    result.storeMatrixOnce(uint2(idx, 0), mat);
}

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void se3pose_from_matrix_kernel(
    DiffTensorView matrix,
    DiffTensorView translation,
    DiffTensorView rotation
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= matrix.size(0)) return;
    
    float4x4 mat = matrix.loadMatrixOnce<4, 4>(uint2(idx, 0));
    float3 trans;
    float4 rot;
    se3pose_from_matrix(mat, trans, rot);
    translation.storeVecOnce(uint2(idx, 0), float3(trans));
    rotation.storeVecOnce(uint2(idx, 0), rot);
}

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void se3pose_to_inverse_matrix_kernel(
    DiffTensorView translation, // (N, 3)
    DiffTensorView rotation,    // (N, 4)
    DiffTensorView result,      // (N, 16)
    no_diff bool wxyz_format = false
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= translation.size(0)) return;
    
    float3 trans = translation.loadVecOnce<3>(uint2(idx, 0));
    float4 rot = rotation.loadVecOnce<4>(uint2(idx, 0));
    
    float4x4 mat = se3pose_to_inverse_matrix(trans, rot, wxyz_format);
    
    result.storeMatrixOnce(uint2(idx, 0), mat);
}

// ============================================================================
// Trajectory CUDA Kernels with 2 Control Poses
// ============================================================================

// Helper struct to return transform result with out-of-bounds flag
struct TransformResult : IDifferentiable {
    float3 point;
    no_diff bool out_of_bounds;  // true if extrapolated beyond control points
}

// Helper struct to return rotation with out-of-bounds flag
struct RotationResult : IDifferentiable {
    float4 quat;
    no_diff bool out_of_bounds;  // true if extrapolated beyond control points
}

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void trajectory_transform_point_2poses_kernel(
    DiffTensorView trans0,
    DiffTensorView rot0,
    DiffTensorView time0,
    DiffTensorView trans1,
    DiffTensorView rot1,
    DiffTensorView time1,
    DiffTensorView point,
    DiffTensorView query_time,
    DiffTensorView result_point,
    DiffTensorView result_out_of_bounds
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= point.size(0)) return;
    
    float3 t0 = trans0.loadVecOnce<3>(uint2(idx, 0));
    float4 r0 = rot0.loadVecOnce<4>(uint2(idx, 0));
    float t0_time = time0.loadVecOnce<1>(uint2(idx, 0))[0];
    float3 t1 = trans1.loadVecOnce<3>(uint2(idx, 0));
    float4 r1 = rot1.loadVecOnce<4>(uint2(idx, 0));
    float t1_time = time1.loadVecOnce<1>(uint2(idx, 0))[0];
    float3 pt = point.loadVecOnce<3>(uint2(idx, 0));
    float qt = query_time.loadVecOnce<1>(uint2(idx, 0))[0];
    
    TransformResult res = trajectory_transform_point_2poses(
        t0, r0, t0_time,
        t1, r1, t1_time,
        pt, qt
    );
    
    result_point.storeVecOnce(uint2(idx, 0), res.point);
    result_out_of_bounds[idx] = res.out_of_bounds ? 1.0 : 0.0;
}

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void trajectory_get_rotation_2poses_kernel(
    DiffTensorView trans0,
    DiffTensorView rot0,
    DiffTensorView time0,
    DiffTensorView trans1,
    DiffTensorView rot1,
    DiffTensorView time1,
    DiffTensorView query_time,
    DiffTensorView result_quat,
    DiffTensorView result_out_of_bounds
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= trans0.size(0)) return;
    
    float3 t0 = trans0.loadVecOnce<3>(uint2(idx, 0));
    float4 r0 = rot0.loadVecOnce<4>(uint2(idx, 0));
    float t0_time = time0.loadVecOnce<1>(uint2(idx, 0))[0];
    float3 t1 = trans1.loadVecOnce<3>(uint2(idx, 0));
    float4 r1 = rot1.loadVecOnce<4>(uint2(idx, 0));
    float t1_time = time1.loadVecOnce<1>(uint2(idx, 0))[0];
    float qt = query_time.loadVecOnce<1>(uint2(idx, 0))[0];
    
    RotationResult res = trajectory_get_rotation_2poses(
        t0, r0, t0_time,
        t1, r1, t1_time,
        qt
    );
    
    result_quat.storeVecOnce(uint2(idx, 0), res.quat);
    result_out_of_bounds[idx] = res.out_of_bounds ? 1.0 : 0.0;
}

// Transform a point using a 2-pose trajectory at a specific time
// Always extrapolates beyond bounds (fully differentiable)
[Differentiable]
TransformResult trajectory_transform_point_2poses(
    float3 trans0,
    float4 rot0,
    float time0,
    float3 trans1,
    float4 rot1,
    float time1,
    float3 point,
    float query_time
) {
    // Check if out of bounds
    float t_min = min(time0, time1);
    float t_max = max(time0, time1);
    bool out_of_bounds = (query_time < t_min) || (query_time > t_max);
    
    // Always interpolate/extrapolate (differentiable for all alpha)
    // Handle degenerate case where time0 == time1
    float alpha = 0.0;
    if (time1 > time0) {
        alpha = (query_time - time0) / (time1 - time0);
    }
    
    // Linear interpolation for translation (works for extrapolation too)
    // Work directly with the input vectors to maintain autodiff chain
    float3 trans_interp = lerp(trans0, trans1, alpha);
    
    // SLERP for rotation (works for extrapolation too)
    float4 rot_interp = quaternion::slerp(rot0, rot1, alpha);
    
    // Now apply the interpolated transformation
    // T(p) = R(p) + t
    float3 rotated = quaternion::rotateVector(rot_interp, point);
    float3 transformed = rotated + trans_interp;
    
    TransformResult result;
    result.point = transformed;
    result.out_of_bounds = out_of_bounds;
    return result;
}

// Get rotation from a 2-pose trajectory at a specific time
// Always extrapolates beyond bounds (fully differentiable)
[Differentiable]
RotationResult trajectory_get_rotation_2poses(
    float3 trans0,
    float4 rot0,
    float time0,
    float3 trans1,
    float4 rot1,
    float time1,
    float query_time
) {
    // Check if out of bounds
    float t_min = min(time0, time1);
    float t_max = max(time0, time1);
    bool out_of_bounds = (query_time < t_min) || (query_time > t_max);
    
    // Always interpolate/extrapolate (differentiable for all alpha)
    // Handle degenerate case where time0 == time1
    float alpha = 0.0;
    if (time1 > time0) {
        alpha = (query_time - time0) / (time1 - time0);
    }
    
    // SLERP for rotation (works for extrapolation too)
    float4 rot = quaternion::slerp(rot0, rot1, alpha);
    
    RotationResult result;
    result.quat = rot;
    result.out_of_bounds = out_of_bounds;
    return result;
}

// ============================================================================
// Trajectory CUDA Kernels with 1 Control Pose
// ============================================================================

[CUDAKernel]
[AutoPyBindCUDA]
[Differentiable]
void trajectory_transform_point_1pose_kernel(
    DiffTensorView trans,
    DiffTensorView rot,
    DiffTensorView time,
    DiffTensorView point,
    DiffTensorView query_time,
    DiffTensorView result_point,
    DiffTensorView result_out_of_bounds
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= point.size(0)) return;
    
    float3 t = trans.loadVecOnce<3>(uint2(idx, 0));
    float4 r = rot.loadVecOnce<4>(uint2(idx, 0));
    float t_time = time.loadVecOnce<1>(uint2(idx, 0))[0];
    float3 pt = point.loadVecOnce<3>(uint2(idx, 0));
    float qt = query_time.loadVecOnce<1>(uint2(idx, 0))[0];
    
    TransformResult res = trajectory_transform_point_1pose(
        t, r, t_time,
        pt, qt
    );
    
    result_point.storeVecOnce(uint2(idx, 0), res.point);
    result_out_of_bounds[idx] = res.out_of_bounds ? 1.0 : 0.0;
}

// Transform a point using a 1-pose trajectory at a specific time
// With only one pose, always returns that pose (can't extrapolate)
[Differentiable]
TransformResult trajectory_transform_point_1pose(
    float3 trans,
    float4 rot,
    float time,
    float3 point,
    float query_time
) {
    // Always out of bounds if query_time != time (can't interpolate with 1 pose)
    bool out_of_bounds = (query_time != time);
    
    // Apply the transformation: T(p) = R(p) + t
    // Work directly with inputs to maintain autodiff chain
    float3 rotated = quaternion::rotateVector(rot, point);
    float3 transformed = rotated + trans;
    
    TransformResult result;
    result.point = transformed;
    result.out_of_bounds = out_of_bounds;
    return result;
}


// GPU implementation mirroring FrameConversion.transform_poses (but in tquat format).
[CUDAKernel]
[AutoPyBindCUDA]
void frame_transform_poses_tquat_kernel(
    TensorView<float> input_poses,   // [N, 7] (tx, ty, tz, qx, qy, qz, qw)
    TensorView<float> output_poses,  // [N, 7]
    float4 rotation,                 // quaternion (xyzw)
    float3 translation,
    float scale
) {
    uint idx = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (idx >= input_poses.size(0)) {
        return;
    }

    float3 input_translation = input_poses.loadVec<3>(uint2(idx, 0));
    float4 input_rotation = input_poses.loadVec<4>(uint2(idx, 3));
    
    float4 new_rotation = quaternion::multiply(rotation, input_rotation);
    float3 rotated_t = quaternion::rotateVector(rotation, input_translation);
    float3 new_translation = scale * (rotated_t + translation);

    output_poses.storeVec(uint2(idx, 0), new_translation);
    output_poses.storeVec(uint2(idx, 3), new_rotation);
}