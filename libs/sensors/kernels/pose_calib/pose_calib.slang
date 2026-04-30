// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Pose calibration kernels for fused pose calibration and rolling shutter interpolation.
//
// This module provides a fused kernel that combines:
// 1. Pose calibration (applying learned calibration deltas)
// 2. Rolling shutter interpolation (adjusting poses for subsampled regions)

#include <geometry/kernels/rotation.slang>

#include <slang_utils/extensions.slang>

#include <sensors/kernels/cameras/camera.slang>
#include <sensors/kernels/cameras/interface.slang>
#include <sensors/kernels/common/interpolation.slang>

// Load 4x4 matrix from tensor at [base_idx.x, base_idx.y, :, :]
// Assumes row-major storage: tensor shape is [..., 4, 4] with rows contiguous
[Differentiable]
float4x4 load_matrix4x4(DiffTensorView tensor, uint2 base_idx) {
    float4 row0 = tensor.loadVecOnce<4>(uint4(base_idx.x, base_idx.y, 0, 0));
    float4 row1 = tensor.loadVecOnce<4>(uint4(base_idx.x, base_idx.y, 1, 0));
    float4 row2 = tensor.loadVecOnce<4>(uint4(base_idx.x, base_idx.y, 2, 0));
    float4 row3 = tensor.loadVecOnce<4>(uint4(base_idx.x, base_idx.y, 3, 0));
    return float4x4(row0, row1, row2, row3);
}

// Non-differentiable overload for TensorView (no gradient flow)
float4x4 load_matrix4x4(TensorView<float> tensor, uint2 base_idx) {
    float4 row0 = tensor.loadVec<4>(uint4(base_idx.x, base_idx.y, 0, 0));
    float4 row1 = tensor.loadVec<4>(uint4(base_idx.x, base_idx.y, 1, 0));
    float4 row2 = tensor.loadVec<4>(uint4(base_idx.x, base_idx.y, 2, 0));
    float4 row3 = tensor.loadVec<4>(uint4(base_idx.x, base_idx.y, 3, 0));
    return float4x4(row0, row1, row2, row3);
}

// Store 4x4 matrix to tensor at [base_idx.x, base_idx.y, :, :]
// Assumes row-major storage: tensor shape is [..., 4, 4] with rows contiguous
[Differentiable]
void store_matrix4x4(DiffTensorView tensor, uint2 base_idx, float4x4 mat) {
    tensor.storeVecOnce<4>(uint4(base_idx.x, base_idx.y, 0, 0), mat[0]);
    tensor.storeVecOnce<4>(uint4(base_idx.x, base_idx.y, 1, 0), mat[1]);
    tensor.storeVecOnce<4>(uint4(base_idx.x, base_idx.y, 2, 0), mat[2]);
    tensor.storeVecOnce<4>(uint4(base_idx.x, base_idx.y, 3, 0), mat[3]);
}

// Inner implementation with compile-time constants for enable_calib and has_subsampling.
// Using compile-time specialization eliminates runtime branch overhead.
// When HAS_SUBSAMPLING=false, rect_points_lb/resolution/shutter_type are unused (can be dummies).
[Differentiable]
void compute_poses_and_timestamps_impl<let ENABLE_CALIB : bool, let HAS_SUBSAMPLING : bool>(
    uint tid,
    TensorView<float> T_sensor_world_startend_allviews,
    DiffTensorView embed_weights,
    TensorView<int32_t> frame_idx,
    TensorView<float> rect_points_lb,
    TensorView<float> resolution,
    TensorView<int64_t> timestamps_startend_us_allviews,
    int shutter_type,
    DiffTensorView T_out,
    TensorView<int64_t> timestamps_out,
) {
    // --- Index into allviews using frame_idx ---
    int fidx = frame_idx[uint(tid)];

    // Load 4x4 start/end poses from T_sensor_world_startend_allviews[fidx, 0/1, :, :]
    float4x4 T_start_mat = load_matrix4x4(T_sensor_world_startend_allviews, uint2(fidx, 0));
    float4x4 T_end_mat = load_matrix4x4(T_sensor_world_startend_allviews, uint2(fidx, 1));

    // Convert 4x4 matrices to SE3 (translation + quaternion)
    transform::SE3 T_start = transform::convertMatrixToSE3(T_start_mat);
    transform::SE3 T_end = transform::convertMatrixToSE3(T_end_mat);

    // Load timestamps
    int64_t ts_start = timestamps_startend_us_allviews[uint2(fidx, 0)];
    int64_t ts_end = timestamps_startend_us_allviews[uint2(fidx, 1)];

    // --- Apply calibration (compile-time decision) ---
    if (ENABLE_CALIB) {
        // Load pose_delta [9]: first 3 = translation offset, last 6 = 6D rotation offset from identity
        // Note: Use loadVecEx instead of loadVecOnce to ensure proper gradient accumulation
        // when multiple batch elements reference the same frame_idx
        float3 dx = embed_weights.loadVecEx<3>(uint2(fidx, 0));
        float3 col1_offset = embed_weights.loadVecEx<3>(uint2(fidx, 3));
        float3 col2_offset = embed_weights.loadVecEx<3>(uint2(fidx, 6));

        // Convert 6D rotation offset to 3x3 rotation matrix
        float3x3 dR = rotation6dToMatrix(col1_offset, col2_offset);

        // Convert 3x3 rotation matrix to quaternion
        float4 dq = quaternion::fromMatrix(dR);

        // Create calibration delta SE3
        transform::SE3 pose_delta = transform::SE3.from_vec(dx, dq);

        // Apply calibration: T = T * delta (right multiplication)
        T_start = T_start * pose_delta;
        T_end = T_end * pose_delta;
    }

    // --- Apply rolling shutter interpolation (compile-time decision) ---
    if (HAS_SUBSAMPLING) {
        // Load rect corners: rect_points_lb[tid, :, :] is [2, 2] (start/end corners, x/y)
        float2 rect_start = rect_points_lb.loadVec<2>(uint3(tid, 0, 0));
        float2 rect_end = rect_points_lb.loadVec<2>(uint3(tid, 1, 0));

        // Load resolution: resolution[tid, :] is [2] (width, height)
        float2 res = resolution.loadVec<2>(uint2(tid, 0));
        uint2 res_uint = uint2(uint(res.x), uint(res.y));

        // Compute relative times [0, 1] based on rect position and shutter type
        float2 t_rel = float2(
            no_diff compute_relative_frame_time(rect_start, res_uint, ShutterType(shutter_type)),
            no_diff compute_relative_frame_time(rect_end, res_uint, ShutterType(shutter_type))
        );
        // Ensure t_start <= t_end
        if (t_rel.x > t_rel.y) {
            t_rel = float2(t_rel.y, t_rel.x);
        }

        // `interpolate_pose_2poses` interpolate poses using SLERP (rotation) + lerp (translation).
        // This is equivalent to Lie group (exp/log) interpolation in transform::interpolate but more GPU-efficient
        // due to less branch divergence in warps and math operations.
        transform::SE3 T_start_orig = T_start;
        transform::SE3 T_end_orig = T_end;
        T_start = interpolate_pose_2poses(T_start_orig, T_end_orig, t_rel.x);
        T_end = interpolate_pose_2poses(T_start_orig, T_end_orig, t_rel.y);

        // Interpolate timestamps
        int64_t ts_start_orig = ts_start;
        int64_t ts_end_orig = ts_end;
        ts_start = ts_start_orig + int64_t(float(ts_end_orig - ts_start_orig) * t_rel.x);
        ts_end = ts_start_orig + int64_t(float(ts_end_orig - ts_start_orig) * t_rel.y);
    }

    // --- Convert SE3 back to 4x4 matrix and store outputs ---
    // Note: If downstream consumers accept 7D (3D trans + 4D quaternion) format directly,
    // we could avoid this matrix conversion and output tquat instead.
    float4x4 T_start_out = transform::convertSE3ToMatrix(T_start);
    float4x4 T_end_out = transform::convertSE3ToMatrix(T_end);
    store_matrix4x4(T_out, uint2(tid, 0), T_start_out);
    store_matrix4x4(T_out, uint2(tid, 1), T_end_out);

    // Store timestamps
    timestamps_out[uint2(tid, 0)] = ts_start;
    timestamps_out[uint2(tid, 1)] = ts_end;
}

// ============================================================================
// Specialized kernel entry points (2x2 combinations)
// ============================================================================

// Kernel: no calibration, no subsampling
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void compute_poses_and_timestamps_kernel(
    no_diff uint batch_size,
    TensorView<float> T_sensor_world_startend_allviews,
    TensorView<int32_t> frame_idx,
    TensorView<int64_t> timestamps_startend_us_allviews,
    DiffTensorView T_out,
    TensorView<int64_t> timestamps_out,
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= batch_size) return;

    // Dummy TensorViews - never accessed since HAS_SUBSAMPLING=false
    TensorView<float> dummy_rect;
    TensorView<float> dummy_res;
    DiffTensorView<float> dummy_embed_weights; // since ENABLE_CALIB == false, we don't read from embed_weights

    compute_poses_and_timestamps_impl</*ENABLE_CALIB=*/false, /*HAS_SUBSAMPLING=*/false>(
        tid, T_sensor_world_startend_allviews, dummy_embed_weights, frame_idx,
        dummy_rect, dummy_res, timestamps_startend_us_allviews,
        0, T_out, timestamps_out);
}

// Kernel: no calibration, with subsampling
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void compute_poses_and_timestamps_subsample_kernel(
    no_diff uint batch_size,
    TensorView<float> T_sensor_world_startend_allviews,
    TensorView<int32_t> frame_idx,
    TensorView<float> rect_points_lb,
    TensorView<float> resolution,
    TensorView<int64_t> timestamps_startend_us_allviews,
    no_diff int shutter_type,
    DiffTensorView T_out,
    TensorView<int64_t> timestamps_out,
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= batch_size) return;

    DiffTensorView<float> dummy_embed_weights; // since ENABLE_CALIB == false, we don't read from embed_weights

    compute_poses_and_timestamps_impl</*ENABLE_CALIB=*/false, /*HAS_SUBSAMPLING=*/true>(
        tid, T_sensor_world_startend_allviews, dummy_embed_weights, frame_idx,
        rect_points_lb, resolution, timestamps_startend_us_allviews,
        shutter_type, T_out, timestamps_out);
}

// Kernel: with calibration, no subsampling
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void compute_poses_and_timestamps_calib_kernel(
    no_diff uint batch_size,
    TensorView<float> T_sensor_world_startend_allviews,
    DiffTensorView embed_weights,
    TensorView<int32_t> frame_idx,
    TensorView<int64_t> timestamps_startend_us_allviews,
    DiffTensorView T_out,
    TensorView<int64_t> timestamps_out,
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= batch_size) return;

    // Dummy TensorViews - never accessed since HAS_SUBSAMPLING=false
    TensorView<float> dummy_rect;
    TensorView<float> dummy_res;

    compute_poses_and_timestamps_impl</*ENABLE_CALIB=*/true, /*HAS_SUBSAMPLING=*/false>(
        tid, T_sensor_world_startend_allviews, embed_weights, frame_idx,
        dummy_rect, dummy_res, timestamps_startend_us_allviews,
        0, T_out, timestamps_out);
}

// Kernel: with calibration, with subsampling
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void compute_poses_and_timestamps_calib_subsample_kernel(
    no_diff uint batch_size,
    TensorView<float> T_sensor_world_startend_allviews,
    DiffTensorView embed_weights,
    TensorView<int32_t> frame_idx,
    TensorView<float> rect_points_lb,
    TensorView<float> resolution,
    TensorView<int64_t> timestamps_startend_us_allviews,
    no_diff int shutter_type,
    DiffTensorView T_out,
    TensorView<int64_t> timestamps_out,
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= batch_size) return;

    compute_poses_and_timestamps_impl</*ENABLE_CALIB=*/true, /*HAS_SUBSAMPLING=*/true>(
        tid, T_sensor_world_startend_allviews, embed_weights, frame_idx,
        rect_points_lb, resolution, timestamps_startend_us_allviews,
        shutter_type, T_out, timestamps_out);
}
