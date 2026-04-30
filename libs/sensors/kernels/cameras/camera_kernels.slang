// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Camera CUDA kernels - entry point for Python bindings via slangtorch
// These kernels are thin wrappers around camera device functions defined in camera.slang

// Include extensions for DiffTensorView operations
#include <slang_utils/extensions.slang>

// Include shared trajectory utilities
#include <sensors/kernels/common/trajectory.slang>

// Include camera device functions
#include <sensors/kernels/cameras/camera.slang>

[CUDAKernel]
[AutoPyBindCUDA]
void generate_image_points_kernel(
    int width,
    int height,
    TensorView<float> output // [H*W, 2]
) {
    uint idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (idx >= width * height) return;

    uint x = idx % width;
    uint y = idx / width;
    output.storeVec(uint2(idx, 0), float2(float(x)+0.5f, float(y)+0.5f));
}

// ============================================================================
// Helper Functions for Loading Struct Parameters from Tensors
// ============================================================================

// Load OpenCVPinholeCameraModelParameters from a single packed tensor
// Layout: (18,) [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4, resolution_x, resolution_y]
// Uses a single DiffTensorView for efficiency - avoids overhead of 5 separate tensor views
// Uses loadVecUniform/loadUniform since all threads load the same indices (uniform access pattern)
// This ensures correct gradient accumulation in the backward pass via WaveActiveSum reduction
[Differentiable]
OpenCVPinholeCameraModelParameters load_opencv_pinhole_params(
    DiffTensorView intrinsics  // (18,) packed intrinsics tensor
) {
    OpenCVPinholeCameraModelParameters params;
    params.focal_length = intrinsics.loadVecUniform<2>(0);       // [0:2] fx, fy
    params.principal_point = intrinsics.loadVecUniform<2>(2);    // [2:4] cx, cy
    params.radial_coeffs[0] = intrinsics.loadUniform(4);         // [4:10] k1-k6
    params.radial_coeffs[1] = intrinsics.loadUniform(5);
    params.radial_coeffs[2] = intrinsics.loadUniform(6);
    params.radial_coeffs[3] = intrinsics.loadUniform(7);
    params.radial_coeffs[4] = intrinsics.loadUniform(8);
    params.radial_coeffs[5] = intrinsics.loadUniform(9);
    params.tangential_coeffs[0] = intrinsics.loadUniform(10);    // [10:12] p1, p2
    params.tangential_coeffs[1] = intrinsics.loadUniform(11);
    params.thin_prism_coeffs[0] = intrinsics.loadUniform(12);    // [12:16] s1-s4
    params.thin_prism_coeffs[1] = intrinsics.loadUniform(13);
    params.thin_prism_coeffs[2] = intrinsics.loadUniform(14);
    params.thin_prism_coeffs[3] = intrinsics.loadUniform(15);
    params.resolution = intrinsics.loadVecUniform<2>(16);      // [16:18] resolution_x, resolution_y
    return params;
}

// Load OpenCVFisheyeCameraModelParameters from a single packed tensor
// Layout: (11,) [cx, cy, fx, fy, k1, k2, k3, k4, ab, resolution_x, resolution_y]
// Uses a single DiffTensorView for efficiency - avoids overhead of 5 separate tensor views
// Uses loadVecUniform/loadUniform since all threads load the same indices (uniform access pattern)
// This ensures correct gradient accumulation in the backward pass via WaveActiveSum reduction
[Differentiable]
OpenCVFisheyeCameraModelParameters load_opencv_fisheye_params(
    DiffTensorView intrinsics,  // (11,) packed intrinsics tensor
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm
) {
    OpenCVFisheyeCameraModelParameters params;
    params.principal_point = intrinsics.loadVecUniform<2>(0);    // [0:2] cx, cy
    params.focal_length = intrinsics.loadVecUniform<2>(2);       // [2:4] fx, fy
    params.forward_poly[0] = intrinsics.loadUniform(4);          // [4:8] k1-k4
    params.forward_poly[1] = intrinsics.loadUniform(5);
    params.forward_poly[2] = intrinsics.loadUniform(6);
    params.forward_poly[3] = intrinsics.loadUniform(7);
    params.approx_backward_factor = intrinsics.loadUniform(8); // [8:9] approx_backward_factor
    params.resolution = intrinsics.loadVecUniform<2>(9);      // [9:11] resolution_x, resolution_y
    params.max_angle = max_angle;
    params.newton_iterations = newton_iterations;
    params.min_2d_norm = min_2d_norm;
    return params;
}

// FTheta intrinsics tensor layout offsets (must match parameters.py _FTHETA_* constants)
// Layout: [pp(2), fw_poly(N), bw_poly(N), A(4), Ainv(4)]
// where N = MAX_POLYNOMIAL_TERMS
static const uint FTHETA_PP_OFFSET = 0;
static const uint FTHETA_FW_OFFSET = FTHETA_PP_OFFSET + 2;
static const uint FTHETA_BW_OFFSET = FTHETA_FW_OFFSET + MAX_POLYNOMIAL_TERMS;
static const uint FTHETA_A_OFFSET = FTHETA_BW_OFFSET + MAX_POLYNOMIAL_TERMS;
static const uint FTHETA_AINV_OFFSET = FTHETA_A_OFFSET + 4;

// Number of differentiable floats in FThetaCameraModelParameters:
// pp(2) + fw_poly(6) + bw_poly(6) + A(4) + Ainv(4) = 22
static const uint FTHETA_NUM_DIFF_PARAMS = 2 + 2 * MAX_POLYNOMIAL_TERMS + 4 + 4;

// Load FThetaCameraModelParameters from a single packed tensor.
// Uses loadBatchUniform to batch all 22 uniform loads into a single shared-memory
// reduction in the backward pass, reducing global atomics from O(warps) to O(blocks).
[Differentiable]
FThetaCameraModelParameters load_ftheta_params(
    DiffTensorView intrinsics,
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm
) {
    float raw[FTHETA_NUM_DIFF_PARAMS] = intrinsics.loadBatchUniform<FTHETA_NUM_DIFF_PARAMS>(FTHETA_PP_OFFSET);

    FThetaCameraModelParameters params;
    params.reference_poly = FThetaCameraModelParameters::FThetaPolynomialType(reference_poly);
    params.principal_point = float2(raw[FTHETA_PP_OFFSET], raw[FTHETA_PP_OFFSET + 1]);
    for (uint i = 0; i < MAX_POLYNOMIAL_TERMS; i++) {
        params.fw_poly[i] = raw[FTHETA_FW_OFFSET + i];
        params.bw_poly[i] = raw[FTHETA_BW_OFFSET + i];
    }
    params.A[0][0] = raw[FTHETA_A_OFFSET];
    params.A[0][1] = raw[FTHETA_A_OFFSET + 1];
    params.A[1][0] = raw[FTHETA_A_OFFSET + 2];
    params.A[1][1] = raw[FTHETA_A_OFFSET + 3];
    params.Ainv[0][0] = raw[FTHETA_AINV_OFFSET];
    params.Ainv[0][1] = raw[FTHETA_AINV_OFFSET + 1];
    params.Ainv[1][0] = raw[FTHETA_AINV_OFFSET + 2];
    params.Ainv[1][1] = raw[FTHETA_AINV_OFFSET + 3];
    params.fw_poly_degree = fw_poly_degree;
    params.bw_poly_degree = bw_poly_degree;
    params.max_angle = max_angle;
    params.newton_iterations = newton_iterations;
    params.min_2d_norm = min_2d_norm;
    return params;
}

// Number of differentiable floats in BivariateWindshieldModelParameters:
// h_poly(6) + v_poly(15) = 21
static const uint BIVARIATE_NUM_DIFF_PARAMS =
    BivariateWindshieldModelParameters::MAX_H_POLYNOMIAL_TERMS
    + BivariateWindshieldModelParameters::MAX_V_POLYNOMIAL_TERMS;

// Load BivariateWindshieldModelParameters for the distort (projection) direction.
// Packed tensor layout: [h_poly(6), v_poly(15), h_poly_inv(6), v_poly_inv(15)]
// FORWARD reference -> distort uses h_poly/v_poly at offsets [0:21]
// BACKWARD reference -> distort uses h_poly_inv/v_poly_inv at offsets [21:42]
// Uses loadBatchUniform to batch all 21 loads into one shared-memory backward reduction.
[Differentiable]
BivariateWindshieldModelParameters load_bivariate_windshield_distort_params(
    DiffTensorView distortion_coeffs,
    no_diff int reference_polynomial,
    no_diff uint h_poly_degree,
    no_diff uint v_poly_degree
) {
    BivariateWindshieldModelParameters params;
    params.h_poly_degree = h_poly_degree;
    params.v_poly_degree = v_poly_degree;

    uint base = (reference_polynomial == 0) ? 0 : BIVARIATE_NUM_DIFF_PARAMS;

    float raw[BIVARIATE_NUM_DIFF_PARAMS] = distortion_coeffs.loadBatchUniform<BIVARIATE_NUM_DIFF_PARAMS>(base);

    for (uint i = 0; i < BivariateWindshieldModelParameters::MAX_H_POLYNOMIAL_TERMS; i++) {
        params.h_poly[i] = raw[i];
    }
    for (uint i = 0; i < BivariateWindshieldModelParameters::MAX_V_POLYNOMIAL_TERMS; i++) {
        params.v_poly[i] = raw[BivariateWindshieldModelParameters::MAX_H_POLYNOMIAL_TERMS + i];
    }
    return params;
}

// Load BivariateWindshieldModelParameters for the undistort (backprojection) direction.
// Uses the inverse polynomial pair relative to the distort loader.
// FORWARD reference -> undistort uses h_poly_inv/v_poly_inv at offsets [21:42]
// BACKWARD reference -> undistort uses h_poly/v_poly at offsets [0:21]
// Uses loadBatchUniform to batch all 21 loads into one shared-memory backward reduction.
[Differentiable]
BivariateWindshieldModelParameters load_bivariate_windshield_undistort_params(
    DiffTensorView distortion_coeffs,
    no_diff int reference_polynomial,
    no_diff uint h_poly_degree,
    no_diff uint v_poly_degree
) {
    BivariateWindshieldModelParameters params;
    params.h_poly_degree = h_poly_degree;
    params.v_poly_degree = v_poly_degree;
    uint base = (reference_polynomial == 0) ? BIVARIATE_NUM_DIFF_PARAMS : 0;

    float raw[BIVARIATE_NUM_DIFF_PARAMS] = distortion_coeffs.loadBatchUniform<BIVARIATE_NUM_DIFF_PARAMS>(base);

    for (uint i = 0; i < BivariateWindshieldModelParameters::MAX_H_POLYNOMIAL_TERMS; i++) {
        params.h_poly[i] = raw[i];
    }
    for (uint i = 0; i < BivariateWindshieldModelParameters::MAX_V_POLYNOMIAL_TERMS; i++) {
        params.v_poly[i] = raw[BivariateWindshieldModelParameters::MAX_H_POLYNOMIAL_TERMS + i];
    }
    return params;
}

// Helper functions (compute_relative_frame_time, is_in_bounds) are now in camera.slang

// ============================================================================
// Primary Kernel: Project world points with rolling shutter support
// ============================================================================

// This kernel projects world points to image coordinates with full rolling shutter
// compensation. It iteratively refines the pose based on the projection location.
//
// Template parameters:
//   Projection: ICameraProjection - camera projection model
//   ExternalDist: IExternalDistortion - external distortion model
//
// Algorithm:
//   1. Start with initial pose guess (typically at t=0.5)
//   2. Transform world point to camera space
//   3. Apply external distortion and camera projection
//   4. Compute relative time based on projected location and shutter type
//   5. Refine pose using computed time
//   6. Iterate until convergence or max iterations
//
// Note: This is a generic helper function called by concrete wrappers below.
// Generic functions cannot be entry points in slangtorch.
[Differentiable]
void project_world_points_shutter_pose_kernel<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    Projection projection,                  // Camera projection parameters
    ExternalDist external_distortion,       // External distortion parameters
    uint2 resolution,                       // (width, height)
    ShutterType shutter_type,               // Rolling shutter behavior
    DiffTensorView trajectory_control_poses_translation,  // (K, 3) control pose translations
    DiffTensorView trajectory_control_poses_rotation,     // (K, 4) control pose rotations (quaternions)
    no_diff TensorView<float> trajectory_control_times,              // (K,) control times [0, 1]
    uint trajectory_control_count,          // Number of control poses
    no_diff int max_iterations,                     // Max rolling shutter iterations
    no_diff float stop_mean_error_px,               // Stopping criterion for mean pixel error
    no_diff float stop_delta_mean_error_px,         // Stopping criterion for pixel error change
    no_diff float initial_relative_time,            // Initial guess for rolling shutter convergence [0, 1]
    no_diff int64_t start_timestamp_us,             // Start timestamp in microseconds (for absolute timestamp output)
    no_diff int64_t end_timestamp_us,               // End timestamp in microseconds (for absolute timestamp output)
    DiffTensorView world_points,           // (N, 3) input
    DiffTensorView image_points,           // (N, 2) output
    no_diff TensorView<bool> valid_flags,  // (N,) bool output
    no_diff TensorView<int64_t> timestamps_us,        // (N,) int64 output - absolute timestamp in microseconds
    no_diff TensorView<float> poses_translation,      // (N, 3) float output - per-point pose translation
    no_diff TensorView<float> poses_rotation          // (N, 4) float output - per-point pose rotation (quaternion)
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = world_points.size(0);
    if (tid >= count) return;

    // No trajectory poses: emit invalid result to avoid out-of-bounds pose access
    if (trajectory_control_count == 0) {
        image_points.storeVecOnce(uint2(tid, 0), float2(0.0, 0.0));
        valid_flags.store(tid, false);
        timestamps_us.store(tid, start_timestamp_us);
        poses_translation.storeVec(uint2(tid, 0), float3(0.0, 0.0, 0.0));
        poses_rotation.storeVec(uint2(tid, 0), float4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Load world point (differentiable — gradient flows back to world_points)
    float3 world_point = world_points.loadVecOnce<3>(uint2(tid, 0));

    // Pre-load the two control poses from DiffTensorView (differentiable loads).
    // The device function works on plain float3/float4 so it can use
    // [BackwardDerivativeOf]. Slang auto-diffs these loads/stores.
    float3 trans0 = trajectory_control_poses_translation.loadVecEx<3>(uint2(0, 0));
    float3 trans1 = trajectory_control_poses_translation.loadVecEx<3>(uint2(min(trajectory_control_count - 1, 1u), 0));
    float4 rot0 = trajectory_control_poses_rotation.loadVecEx<4>(uint2(0, 0));
    float4 rot1 = trajectory_control_poses_rotation.loadVecEx<4>(uint2(min(trajectory_control_count - 1, 1u), 0));

    // Call device function with custom backward — rolling shutter loop
    // context is eliminated, only the single converged step is auto-diff'd.
    RollingShutterProjectionResult rs_result = project_world_point_rolling_shutter(
        projection, external_distortion, world_point,
        trans0, trans1, rot0, rot1,
        trajectory_control_times, trajectory_control_count,
        resolution, shutter_type, max_iterations,
        stop_mean_error_px, stop_delta_mean_error_px,
        initial_relative_time);

    // Reconstruct final interpolated pose for output (no_diff stores)
    float relative_time = rs_result.relative_time;
    TrajectoryInterval interval = no_diff find_trajectory_interval(
        relative_time, trajectory_control_times, trajectory_control_count);
    float3 final_trans = detach(lerp(trans0, trans1, interval.alpha));
    float4 final_rot = detach(quaternion::slerp(rot0, rot1, interval.alpha));

    // Store results
    image_points.storeVecOnce(uint2(tid, 0), rs_result.image_point);
    valid_flags.store(tid, rs_result.valid);
    timestamps_us.store(tid, start_timestamp_us + int64_t(relative_time * float(end_timestamp_us - start_timestamp_us)));
    poses_translation.storeVec(uint2(tid, 0), final_trans);
    poses_rotation.storeVec(uint2(tid, 0), final_rot);
}

// ============================================================================
// Rolling shutter projection with custom backward
// ============================================================================
//
// Architecture:
//   DiffTensorView can't be wrapped in DifferentialPair (doesn't implement
//   IDifferentiable), so [BackwardDerivativeOf] can't target the kernel directly.
//   Instead we factor the computation into device functions that use plain
//   differentiable types:
//
//   Kernel (auto-diff'd by Slang):
//     loads from DiffTensorView → calls device function → stores to DiffTensorView
//
//   Device function [Differentiable] + [BackwardDerivativeOf]:
//     takes float3/float4 poses, world_point, projection, external_distortion
//     forward: full rolling shutter convergence loop
//     backward: re-run convergence non-diff, differentiate single converged step
//
//   This eliminates ~14x of primal context by avoiding loop unrolling in backward.
// ============================================================================

struct RollingShutterProjectionResult : IDifferentiable {
    float2 image_point;
    no_diff bool valid;
    no_diff float relative_time;
};

// Single-step differentiable projection at an interpolated pose.
// Slang auto-diffs this — only ~178 floats of primal context.
[Differentiable]
CameraProjectResult interpolate_and_project<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    Projection projection,
    ExternalDist external_distortion,
    float3 world_point,
    float3 trans0,
    float3 trans1,
    float4 rot0,
    float4 rot1,
    no_diff float alpha
) {
    float3 interp_trans = lerp(trans0, trans1, alpha);
    float4 interp_rot = quaternion::slerp(rot0, rot1, alpha);
    SE3Pose pose = SE3Pose.from_translation_quaternion(interp_trans, interp_rot);
    float3 camera_point = pose.inverse_transform_point(world_point);

    // We can skip the check for camera_point.z <= 0.0 here because it is guaranteed by
    // the validity check in bwd(project_world_point_rolling_shutter).
    // This simplifies the autodiff graph for the projection step.

    float3 camera_ray = normalize(camera_point);
    float3 distorted_ray = external_distortion.distort(camera_ray);
    return projection.camera_rays_to_image_points(distorted_ray);
}

// Full rolling shutter projection for a single world point.
// The forward runs the convergence loop normally.
[Differentiable]
RollingShutterProjectionResult project_world_point_rolling_shutter<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    Projection projection,
    ExternalDist external_distortion,
    float3 world_point,
    float3 trans0,
    float3 trans1,
    float4 rot0,
    float4 rot1,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff uint2 resolution,
    no_diff ShutterType shutter_type,
    no_diff int max_iterations,
    no_diff float stop_mean_error_px,
    no_diff float stop_delta_mean_error_px,
    no_diff float initial_relative_time
) {
    // Note: This kernel has a custom backward (below).
    // project_world_point_rolling_shutter_bwd
    // Changes to this kernel must be reflected in the custom backward.
    // Consolidation into a single kernel is limited by compiler limitations

    RollingShutterProjectionResult result;
    result.image_point = float2(0.0, 0.0);
    result.valid = false;
    result.relative_time = initial_relative_time;

    float2 prev_image_point = float2(0.0, 0.0);

    int iter_limit = min(max_iterations, MAX_ROLLING_SHUTTER_ITERATIONS);
    [MaxIters(MAX_ROLLING_SHUTTER_ITERATIONS)]
    for (int iter = 0; iter < iter_limit; iter++) {
        TrajectoryInterval interval = no_diff find_trajectory_interval(
            result.relative_time, trajectory_control_times, trajectory_control_count);

        float3 trans = lerp(trans0, trans1, interval.alpha);
        float4 rot = quaternion::slerp(rot0, rot1, interval.alpha);

        SE3Pose pose = SE3Pose.from_translation_quaternion(trans, rot);
        float3 camera_point = pose.inverse_transform_point(world_point);

        if (camera_point.z <= 0.0) {
            result.valid = false;
            break;
        }

        float3 camera_ray = normalize(camera_point);
        float3 distorted_ray = external_distortion.distort(camera_ray);
        CameraProjectResult proj_result = projection.camera_rays_to_image_points(distorted_ray);
        result.image_point = proj_result.image_point;

        if (!proj_result.valid || !is_in_bounds(result.image_point, resolution)) {
            result.valid = false;
            break;
        }

        result.valid = true;

        float new_relative_time = no_diff compute_relative_frame_time(
            result.image_point, resolution, shutter_type);

        if (shutter_type == ShutterType.GLOBAL) break;

        float2 pixel_diff = result.image_point - prev_image_point;
        float pixel_error = sqrt(pixel_diff.x * pixel_diff.x + pixel_diff.y * pixel_diff.y);
        if (iter > 0 && pixel_error < stop_delta_mean_error_px) break;

        float time_diff = abs(new_relative_time - result.relative_time);
        float approx_pixel_change = time_diff * float(max(resolution.x, resolution.y));
        if (approx_pixel_change < stop_mean_error_px) break;

        prev_image_point = result.image_point;
        result.relative_time = new_relative_time;
    }

    return result;
}

// Custom backward: re-runs convergence non-differentiably, then calls
// bwd_diff(interpolate_and_project) for the single converged step.
[BackwardDerivativeOf(project_world_point_rolling_shutter)]
void project_world_point_rolling_shutter_bwd<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    inout DifferentialPair<Projection> projection,
    inout DifferentialPair<ExternalDist> external_distortion,
    inout DifferentialPair<float3> world_point,
    inout DifferentialPair<float3> trans0,
    inout DifferentialPair<float3> trans1,
    inout DifferentialPair<float4> rot0,
    inout DifferentialPair<float4> rot1,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff uint2 resolution,
    no_diff ShutterType shutter_type,
    no_diff int max_iterations,
    no_diff float stop_mean_error_px,
    no_diff float stop_delta_mean_error_px,
    no_diff float initial_relative_time,
    RollingShutterProjectionResult.Differential d_result
) {
    // --- Phase 1: Non-differentiable convergence to find converged time ---
    float relative_time = initial_relative_time;
    bool valid = false;
    float2 current_image_point = float2(0.0, 0.0);
    float2 prev_image_point = float2(0.0, 0.0);
    TrajectoryInterval last_interval;

    int iter_limit = min(max_iterations, MAX_ROLLING_SHUTTER_ITERATIONS);
    for (int iter = 0; iter < iter_limit; iter++) {
        last_interval = find_trajectory_interval(
            relative_time, trajectory_control_times, trajectory_control_count);

        float3 trans = lerp(trans0.p, trans1.p, last_interval.alpha);
        float4 rot = quaternion::slerp(rot0.p, rot1.p, last_interval.alpha);

        SE3Pose pose = SE3Pose.from_translation_quaternion(trans, rot);
        float3 camera_point = pose.inverse_transform_point(world_point.p);

        if (camera_point.z <= 0.0) { valid = false; break; }

        float3 camera_ray = normalize(camera_point);
        float3 distorted_ray = external_distortion.p.distort(camera_ray);
        CameraProjectResult proj_result = projection.p.camera_rays_to_image_points(distorted_ray);
        current_image_point = proj_result.image_point;

        if (!proj_result.valid || !is_in_bounds(current_image_point, resolution)) {
            valid = false; break;
        }

        valid = true;

        float new_relative_time = compute_relative_frame_time(
            current_image_point, resolution, shutter_type);

        if (shutter_type == ShutterType.GLOBAL) break;

        float2 pixel_diff = current_image_point - prev_image_point;
        float pixel_error = sqrt(pixel_diff.x * pixel_diff.x + pixel_diff.y * pixel_diff.y);
        if (iter > 0 && pixel_error < stop_delta_mean_error_px) break;

        float time_diff = abs(new_relative_time - relative_time);
        float approx_pixel_change = time_diff * float(max(resolution.x, resolution.y));
        if (approx_pixel_change < stop_mean_error_px) break;

        prev_image_point = current_image_point;
        relative_time = new_relative_time;
    }

    if (!valid) return;

    // --- Phase 2: Differentiate single converged step ---
    // Reuse the interval from the last iteration — this is the interval that
    // produced the forward's image_point, so gradients match exactly.
    CameraProjectResult.Differential d_proj;
    d_proj.image_point = d_result.image_point;

    bwd_diff(interpolate_and_project)(
        projection, external_distortion, world_point,
        trans0, trans1, rot0, rot1,
        last_interval.alpha,
        d_proj);
}

// ============================================================================
// Helper Kernel: Project camera rays to image points
// ============================================================================

// Projects camera rays (already in camera frame) to image coordinates.
// No pose transformation - assumes rays are in camera space at capture time.
// Uses device function from camera.slang.
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void camera_rays_to_image_points_kernel<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    Projection projection,                  // Camera projection parameters
    ExternalDist external_distortion,       // External distortion parameters
    DiffTensorView camera_rays,            // (N, 3) input - normalized direction vectors
    DiffTensorView image_points,           // (N, 2) output
    no_diff TensorView<bool> valid_flags   // (N,) bool output
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = camera_rays.size(0);
    if (tid >= count) return;

    // Load camera ray
    float3 camera_ray = camera_rays.loadVecOnce<3>(uint2(tid, 0));

    // Use device function from camera.slang
    CameraProjectResult result = project_camera_ray(camera_ray, projection, external_distortion);

    // Store results
    image_points.storeVecOnce(uint2(tid, 0), result.image_point);
    valid_flags.store(tid, result.valid);
}

// ============================================================================
// Back-Projection Kernel: Image points to camera rays
// ============================================================================

// Back-projects image points to camera rays (normalized direction vectors).
// No pose transformation - returns rays in camera frame.
// Uses device function from camera.slang.
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void image_points_to_camera_rays_kernel<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    Projection projection,                  // Camera projection parameters
    ExternalDist external_distortion,       // External distortion parameters
    DiffTensorView image_points,           // (N, 2) input
    DiffTensorView camera_rays             // (N, 3) output - normalized direction vectors
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = image_points.size(0);
    if (tid >= count) return;

    // Load image point
    float2 image_point = image_points.loadVecOnce<2>(uint2(tid, 0));

    // Use device function from camera.slang
    float3 camera_ray = backproject_to_camera_ray(image_point, projection, external_distortion);

    // Store result
    camera_rays.storeVecOnce(uint2(tid, 0), camera_ray);
}

// ============================================================================
// Convenience Kernel: Project world points with mean pose
// ============================================================================

// Projects world points using the mean pose of a dynamic pose.
// Faster than full rolling shutter but less accurate.
//
// Algorithm:
//   1. Compute mean pose (at t=0.5)
//   2. Transform all points using mean pose
//   3. Apply projection
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void project_world_points_mean_pose_kernel<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    Projection projection,                  // Camera projection parameters
    ExternalDist external_distortion,       // External distortion parameters
    uint2 resolution,                       // (width, height)
    DiffTensorView trajectory_control_poses_translation,  // (K, 3) control pose translations
    DiffTensorView trajectory_control_poses_rotation,     // (K, 4) control pose rotations
    no_diff TensorView<float> trajectory_control_times,              // (K,) control times [0, 1]
    uint trajectory_control_count,          // Number of control poses
    no_diff int64_t start_timestamp_us,             // Start timestamp in microseconds (for absolute timestamp output)
    no_diff int64_t end_timestamp_us,               // End timestamp in microseconds (for absolute timestamp output)
    DiffTensorView world_points,           // (N, 3) input
    DiffTensorView image_points,           // (N, 2) output
    no_diff TensorView<bool> valid_flags,  // (N,) bool output
    no_diff TensorView<int64_t> timestamps_us,        // (N,) int64 output - absolute timestamp in microseconds
    no_diff TensorView<float> poses_translation,      // (N, 3) float output - per-point mean pose translation
    no_diff TensorView<float> poses_rotation          // (N, 4) float output - per-point mean pose rotation (quaternion)
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = world_points.size(0);
    if (tid >= count) return;

    // Compute mean timestamp for mean pose (at t=0.5)
    float mean_time = 0.5;
    int64_t mean_timestamp_us = int64_t(start_timestamp_us + mean_time * float(end_timestamp_us - start_timestamp_us));

    // No trajectory poses: emit invalid result to avoid out-of-bounds pose access
    if (trajectory_control_count == 0) {
        image_points.storeVecOnce(uint2(tid, 0), float2(0.0, 0.0));
        valid_flags.store(tid, false);
        timestamps_us.store(tid, mean_timestamp_us);
        poses_translation.storeVec(uint2(tid, 0), float3(0.0, 0.0, 0.0));
        poses_rotation.storeVec(uint2(tid, 0), float4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Load world point
    float3 world_point = world_points.loadVecOnce<3>(uint2(tid, 0));

    // Find trajectory interval and interpolate pose
    TrajectoryInterval interval = find_trajectory_interval(
        mean_time, trajectory_control_times, trajectory_control_count);

    float3 trans0 = trajectory_control_poses_translation.loadVecEx<3>(uint2(interval.idx0, 0));
    float3 trans1 = trajectory_control_poses_translation.loadVecEx<3>(uint2(interval.idx1, 0));
    float4 rot0 = trajectory_control_poses_rotation.loadVecEx<4>(uint2(interval.idx0, 0));
    float4 rot1 = trajectory_control_poses_rotation.loadVecEx<4>(uint2(interval.idx1, 0));

    float3 trans = lerp(trans0, trans1, interval.alpha);
    float4 rot = quaternion::slerp(rot0, rot1, interval.alpha);

    // Create mean SE3Pose
    SE3Pose mean_pose = SE3Pose.from_translation_quaternion(trans, rot);

    // Transform world point to camera frame
    float3 camera_point = mean_pose.inverse_transform_point(world_point);

    // Check if point is behind camera
    if (camera_point.z <= 0.0) {
        image_points.storeVecOnce(uint2(tid, 0), float2(0.0, 0.0));
        valid_flags.store(tid, false);
        timestamps_us.store(tid, mean_timestamp_us);
        poses_translation.storeVec(uint2(tid, 0), trans);
        poses_rotation.storeVec(uint2(tid, 0), rot);
        return;
    }

    // Apply external distortion
    float3 camera_ray = normalize(camera_point);
    float3 distorted_ray = external_distortion.distort(camera_ray);

    // Apply camera projection
    CameraProjectResult proj_result = projection.camera_rays_to_image_points(distorted_ray);

    // Check bounds
    bool valid = proj_result.valid && is_in_bounds(proj_result.image_point, resolution);

    // Store results
    image_points.storeVecOnce(uint2(tid, 0), proj_result.image_point);
    valid_flags.store(tid, valid);
    timestamps_us.store(tid, mean_timestamp_us);
    poses_translation.storeVec(uint2(tid, 0), trans);
    poses_rotation.storeVec(uint2(tid, 0), rot);
}

// ============================================================================
// Back-Projection Kernel: Image points to world rays with static pose
// ============================================================================

// Back-projects image points to world rays using a single static pose.
// All rays use the same pose (no interpolation).
//
// Algorithm:
//   1. Back-project image point to camera ray
//   2. Transform camera ray to world using static pose
//   3. Return world ray (origin + direction)
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void image_points_to_world_rays_static_pose_kernel<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    Projection projection,                  // Camera projection parameters
    ExternalDist external_distortion,       // External distortion parameters
    float3 pose_translation,                // Static pose translation (differentiable)
    float4 pose_rotation,                   // Static pose rotation quaternion (differentiable)
    no_diff int64_t timestamp_us,                     // Static timestamp in microseconds (for absolute timestamp output)
    DiffTensorView image_points,           // (N, 2) input
    DiffTensorView world_rays,             // (N, 6) output [origin.xyz, direction.xyz]
    no_diff TensorView<int64_t> timestamps_us,        // (N,) int64 output - absolute timestamp in microseconds
    no_diff TensorView<float> poses_translation,      // (N, 3) float output - per-point pose translation
    no_diff TensorView<float> poses_rotation          // (N, 4) float output - per-point pose rotation (quaternion)
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = image_points.size(0);
    if (tid >= count) return;

    // Load image point
    float2 image_point = image_points.loadVecOnce<2>(uint2(tid, 0));

    // Create static pose
    SE3Pose pose = SE3Pose.from_translation_quaternion(pose_translation, pose_rotation);

    // Back-project to camera ray (with distortion)
    float3 distorted_ray = projection.image_points_to_camera_rays(image_point);

    // Remove external distortion
    float3 camera_ray = external_distortion.undistort(distorted_ray);
    camera_ray = normalize(camera_ray);

    // Transform to world frame
    // Ray origin is the camera position in world frame
    float3 ray_origin = pose_translation;

    // Ray direction is the rotated camera ray
    float3 ray_direction = pose.transform_direction(camera_ray);

    // Store world ray
    world_rays.storeVecOnce(uint2(tid, 0), ray_origin);
    world_rays.storeVecOnce(uint2(tid, 3), ray_direction);

    // Store timestamp and pose
    timestamps_us.store(tid, timestamp_us);
    poses_translation.storeVec(uint2(tid, 0), pose_translation);
    poses_rotation.storeVec(uint2(tid, 0), pose_rotation);
}

// ============================================================================
// Back-Projection Kernel: Image points to world rays with rolling shutter
// ============================================================================

// Back-projects image points to world rays with rolling shutter support.
// Each ray gets a pose interpolated based on its image location.
//
// Algorithm:
//   1. Compute relative time based on image location and shutter type
//   2. Get interpolated pose at that time
//   3. Back-project image point to camera ray
//   4. Transform to world using interpolated pose
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void image_points_to_world_rays_shutter_pose_kernel<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    Projection projection,                  // Camera projection parameters
    ExternalDist external_distortion,       // External distortion parameters
    uint2 resolution,                       // (width, height)
    ShutterType shutter_type,               // Rolling shutter behavior
    DiffTensorView trajectory_control_poses_translation,  // (K, 3) control pose translations
    DiffTensorView trajectory_control_poses_rotation,     // (K, 4) control pose rotations
    no_diff TensorView<float> trajectory_control_times,              // (K,) control times [0, 1]
    uint trajectory_control_count,          // Number of control poses
    no_diff int64_t start_timestamp_us,             // Start timestamp in microseconds (for absolute timestamp output)
    no_diff int64_t end_timestamp_us,               // End timestamp in microseconds (for absolute timestamp output)
    DiffTensorView image_points,        // (N, 2) input
    no_diff bool does_generate_elements,   // True if elements are generated, false if read from elements tensor
    DiffTensorView world_rays,             // (N, 6) output [origin.xyz, direction.xyz]
    no_diff TensorView<int64_t> timestamps_us,        // (N,) int64 output - absolute timestamp in microseconds
    no_diff TensorView<float> poses_translation,      // (N, 3) float output - per-point pose translation
    no_diff TensorView<float> poses_rotation          // (N, 4) float output - per-point pose rotation (quaternion)
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = does_generate_elements ? (resolution.x * resolution.y) : image_points.size(0);
    if (tid >= count) return;

    // No trajectory poses: emit zero rays to avoid out-of-bounds pose access
    if (trajectory_control_count == 0) {
        world_rays.storeVecOnce(uint2(tid, 0), float3(0.0, 0.0, 0.0));
        world_rays.storeVecOnce(uint2(tid, 3), float3(0.0, 0.0, 0.0));
        timestamps_us.store(tid, start_timestamp_us);
        poses_translation.storeVec(uint2(tid, 0), float3(0.0, 0.0, 0.0));
        poses_rotation.storeVec(uint2(tid, 0), float4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Load image point
    float2 image_point;
    if (does_generate_elements) {
        int y = tid / resolution.x;
        int x = tid - y * resolution.x;
        // Add 0.5 to convert from pixel indices to pixel centers (continuous coordinates)
        image_point = 0.5f + float2(float(x), float(y));
    } else {
        image_point = image_points.loadVecOnce<2>(uint2(tid, 0));
    }

    // Compute relative time from image location and shutter type (non-differentiable)
    float relative_time = no_diff compute_relative_frame_time(image_point, resolution, shutter_type);

    // Find trajectory interval and interpolate pose
    TrajectoryInterval interval = find_trajectory_interval(
        relative_time, trajectory_control_times, trajectory_control_count);

    float3 trans0 = trajectory_control_poses_translation.loadVecEx<3>(uint2(interval.idx0, 0));
    float3 trans1 = trajectory_control_poses_translation.loadVecEx<3>(uint2(interval.idx1, 0));
    float4 rot0 = trajectory_control_poses_rotation.loadVecEx<4>(uint2(interval.idx0, 0));
    float4 rot1 = trajectory_control_poses_rotation.loadVecEx<4>(uint2(interval.idx1, 0));

    float3 trans = lerp(trans0, trans1, interval.alpha);
    float4 rot = quaternion::slerp(rot0, rot1, interval.alpha);

    // Create interpolated pose
    SE3Pose pose = SE3Pose.from_translation_quaternion(trans, rot);

    // Back-project to camera ray (with distortion)
    float3 distorted_ray = projection.image_points_to_camera_rays(image_point);

    // Remove external distortion
    float3 camera_ray = external_distortion.undistort(distorted_ray);
    camera_ray = normalize(camera_ray);

    // Transform to world frame
    // Ray origin is the camera position in world frame
    float3 ray_origin = trans;

    // Ray direction is the rotated camera ray
    float3 ray_direction = pose.transform_direction(camera_ray);

    // Store world ray
    world_rays.storeVecOnce(uint2(tid, 0), ray_origin);
    world_rays.storeVecOnce(uint2(tid, 3), ray_direction);

    // Store timestamp and pose
    timestamps_us.store(tid, start_timestamp_us + int64_t(relative_time * float(end_timestamp_us - start_timestamp_us)));
    poses_translation.storeVec(uint2(tid, 0), trans);
    poses_rotation.storeVec(uint2(tid, 0), rot);
}

// ============================================================================
// Concrete Wrapper Kernels (non-generic versions for slangtorch compatibility)
// ============================================================================
// slangtorch does not support generic kernels, so we provide concrete
// instantiations for each combination of projection and distortion types.

// ----------------------------------------------------------------------------
// project_world_points_shutter_pose: Concrete wrappers
// ----------------------------------------------------------------------------

// OpenCV Pinhole + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_shutter_pose_opencv_pinhole_no_external(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    uint2 resolution,
    int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    uint trajectory_control_count,
    no_diff int max_iterations,
    no_diff float stop_mean_error_px,
    no_diff float stop_delta_mean_error_px,
    no_diff float initial_relative_time,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    NoExternalDistortion external_distortion;
    project_world_points_shutter_pose_kernel<OpenCVPinholeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        max_iterations, stop_mean_error_px, stop_delta_mean_error_px,
        initial_relative_time,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Pinhole + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int max_iterations,
    no_diff float stop_mean_error_px,
    no_diff float stop_delta_mean_error_px,
    no_diff float initial_relative_time,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    project_world_points_shutter_pose_kernel<OpenCVPinholeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        max_iterations, stop_mean_error_px, stop_delta_mean_error_px,
        initial_relative_time,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Fisheye + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_shutter_pose_opencv_fisheye_no_external(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int max_iterations,
    no_diff float stop_mean_error_px,
    no_diff float stop_delta_mean_error_px,
    no_diff float initial_relative_time,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    project_world_points_shutter_pose_kernel<OpenCVFisheyeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        max_iterations, stop_mean_error_px, stop_delta_mean_error_px,
        initial_relative_time,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Fisheye + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int max_iterations,
    no_diff float stop_mean_error_px,
    no_diff float stop_delta_mean_error_px,
    no_diff float initial_relative_time,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    project_world_points_shutter_pose_kernel<OpenCVFisheyeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        max_iterations, stop_mean_error_px, stop_delta_mean_error_px,
        initial_relative_time,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// F-Theta + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_shutter_pose_ftheta_no_external(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int max_iterations,
    no_diff float stop_mean_error_px,
    no_diff float stop_delta_mean_error_px,
    no_diff float initial_relative_time,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    project_world_points_shutter_pose_kernel<FThetaCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        max_iterations, stop_mean_error_px, stop_delta_mean_error_px,
        initial_relative_time,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// F-Theta + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_shutter_pose_ftheta_bivariate_windshield(
    DiffTensorView intrinsics, // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int max_iterations,
    no_diff float stop_mean_error_px,
    no_diff float stop_delta_mean_error_px,
    no_diff float initial_relative_time,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    project_world_points_shutter_pose_kernel<FThetaCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        max_iterations, stop_mean_error_px, stop_delta_mean_error_px,
        initial_relative_time,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// ----------------------------------------------------------------------------
// camera_rays_to_image_points: Concrete wrappers using generic kernel
// ----------------------------------------------------------------------------

// OpenCV Pinhole + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void camera_rays_to_image_points_opencv_pinhole_no_external(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView camera_rays,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    NoExternalDistortion external_distortion;
    camera_rays_to_image_points_kernel<OpenCVPinholeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, camera_rays, image_points, valid_flags
    );
}

// OpenCV Pinhole + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView camera_rays,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    camera_rays_to_image_points_kernel<OpenCVPinholeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, camera_rays, image_points, valid_flags
    );
}

// OpenCV Fisheye + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void camera_rays_to_image_points_opencv_fisheye_no_external(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView camera_rays,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    camera_rays_to_image_points_kernel<OpenCVFisheyeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, camera_rays, image_points, valid_flags
    );
}

// OpenCV Fisheye + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void camera_rays_to_image_points_opencv_fisheye_bivariate_windshield(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView camera_rays,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    camera_rays_to_image_points_kernel<OpenCVFisheyeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, camera_rays, image_points, valid_flags
    );
}

// F-Theta + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void camera_rays_to_image_points_ftheta_no_external(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView camera_rays,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    camera_rays_to_image_points_kernel<FThetaCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, camera_rays, image_points, valid_flags
    );
}

// F-Theta + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void camera_rays_to_image_points_ftheta_bivariate_windshield(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView camera_rays,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    camera_rays_to_image_points_kernel<FThetaCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, camera_rays, image_points, valid_flags
    );
}

// ----------------------------------------------------------------------------
// image_points_to_camera_rays: Concrete wrappers using generic kernel
// ----------------------------------------------------------------------------

// OpenCV Pinhole + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_camera_rays_opencv_pinhole_no_external(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView image_points,
    DiffTensorView camera_rays
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    NoExternalDistortion external_distortion;
    image_points_to_camera_rays_kernel<OpenCVPinholeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, image_points, camera_rays
    );
}

// OpenCV Pinhole + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView image_points,
    DiffTensorView camera_rays
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    image_points_to_camera_rays_kernel<OpenCVPinholeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, image_points, camera_rays
    );
}

// OpenCV Fisheye + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_camera_rays_opencv_fisheye_no_external(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView image_points,
    DiffTensorView camera_rays
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    image_points_to_camera_rays_kernel<OpenCVFisheyeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, image_points, camera_rays
    );
}

// OpenCV Fisheye + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_camera_rays_opencv_fisheye_bivariate_windshield(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView image_points,
    DiffTensorView camera_rays
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    image_points_to_camera_rays_kernel<OpenCVFisheyeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, image_points, camera_rays
    );
}

// F-Theta + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_camera_rays_ftheta_no_external(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView image_points,
    DiffTensorView camera_rays
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    image_points_to_camera_rays_kernel<FThetaCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, image_points, camera_rays
    );
}

// F-Theta + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_camera_rays_ftheta_bivariate_windshield(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView image_points,
    DiffTensorView camera_rays
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    image_points_to_camera_rays_kernel<FThetaCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, image_points, camera_rays
    );
}

// ----------------------------------------------------------------------------
// project_world_points_mean_pose: Concrete wrappers
// ----------------------------------------------------------------------------

// OpenCV Pinhole + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_mean_pose_opencv_pinhole_no_external(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    uint2 resolution,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    NoExternalDistortion external_distortion;
    project_world_points_mean_pose_kernel<OpenCVPinholeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution,
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Pinhole + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    project_world_points_mean_pose_kernel<OpenCVPinholeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution,
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Fisheye + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_mean_pose_opencv_fisheye_no_external(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    uint2 resolution,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    project_world_points_mean_pose_kernel<OpenCVFisheyeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution,
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Fisheye + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_mean_pose_opencv_fisheye_bivariate_windshield(
    DiffTensorView intrinsics,  // (16,) packed intrinsics [cx, cy, fx, fy, k1-k4, dk1-dk4, ab1-ab4]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    project_world_points_mean_pose_kernel<OpenCVFisheyeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution,
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// F-Theta + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_mean_pose_ftheta_no_external(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    uint2 resolution,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    project_world_points_mean_pose_kernel<FThetaCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution,
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// F-Theta + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void project_world_points_mean_pose_ftheta_bivariate_windshield(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView image_points,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_distort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    project_world_points_mean_pose_kernel<FThetaCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution,
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        world_points, image_points, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// ----------------------------------------------------------------------------
// image_points_to_world_rays_static_pose: Concrete wrappers
// ----------------------------------------------------------------------------

// OpenCV Pinhole + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_static_pose_opencv_pinhole_no_external(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView pose_translation,
    DiffTensorView pose_rotation,
    no_diff int64_t timestamp_us,
    DiffTensorView image_points,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    NoExternalDistortion external_distortion;
    float3 trans = pose_translation.loadVecUniform<3>(0);
    float4 rot = pose_rotation.loadVecUniform<4>(0);
    image_points_to_world_rays_static_pose_kernel<OpenCVPinholeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, trans, rot, timestamp_us, image_points, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Pinhole + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView pose_translation,
    DiffTensorView pose_rotation,
    no_diff int64_t timestamp_us,
    DiffTensorView image_points,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    float3 trans = pose_translation.loadVecUniform<3>(0);
    float4 rot = pose_rotation.loadVecUniform<4>(0);
    image_points_to_world_rays_static_pose_kernel<OpenCVPinholeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, trans, rot, timestamp_us, image_points, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Fisheye + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_static_pose_opencv_fisheye_no_external(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView pose_translation,
    DiffTensorView pose_rotation,
    no_diff int64_t timestamp_us,
    DiffTensorView image_points,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    float3 trans = pose_translation.loadVecUniform<3>(0);
    float4 rot = pose_rotation.loadVecUniform<4>(0);
    image_points_to_world_rays_static_pose_kernel<OpenCVFisheyeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, trans, rot, timestamp_us, image_points, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Fisheye + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView pose_translation,
    DiffTensorView pose_rotation,
    no_diff int64_t timestamp_us,
    DiffTensorView image_points,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    float3 trans = pose_translation.loadVecUniform<3>(0);
    float4 rot = pose_rotation.loadVecUniform<4>(0);
    image_points_to_world_rays_static_pose_kernel<OpenCVFisheyeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, trans, rot, timestamp_us, image_points, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// F-Theta + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_static_pose_ftheta_no_external(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView pose_translation,
    DiffTensorView pose_rotation,
    no_diff int64_t timestamp_us,
    DiffTensorView image_points,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    float3 trans = pose_translation.loadVecUniform<3>(0);
    float4 rot = pose_rotation.loadVecUniform<4>(0);
    image_points_to_world_rays_static_pose_kernel<FThetaCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, trans, rot, timestamp_us, image_points, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// F-Theta + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    DiffTensorView pose_translation,
    DiffTensorView pose_rotation,
    no_diff int64_t timestamp_us,
    DiffTensorView image_points,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    float3 trans = pose_translation.loadVecUniform<3>(0);
    float4 rot = pose_rotation.loadVecUniform<4>(0);
    image_points_to_world_rays_static_pose_kernel<FThetaCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, trans, rot, timestamp_us, image_points, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// ----------------------------------------------------------------------------
// image_points_to_world_rays_shutter_pose: Concrete wrappers
// ----------------------------------------------------------------------------

// OpenCV Pinhole + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView image_points,
    no_diff bool does_generate_elements,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    NoExternalDistortion external_distortion;
    image_points_to_world_rays_shutter_pose_kernel<OpenCVPinholeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        image_points, does_generate_elements, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Pinhole + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield(
    DiffTensorView intrinsics,  // (18,) packed intrinsics [fx, fy, cx, cy, k1-k6, p1, p2, s1-s4, resolution_x, resolution_y]
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView image_points,
    no_diff bool does_generate_elements,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVPinholeCameraModelParameters projection = load_opencv_pinhole_params(intrinsics);
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    image_points_to_world_rays_shutter_pose_kernel<OpenCVPinholeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        image_points, does_generate_elements, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Fisheye + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView image_points,
    no_diff bool does_generate_elements,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    image_points_to_world_rays_shutter_pose_kernel<OpenCVFisheyeCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        image_points, does_generate_elements, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// OpenCV Fisheye + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield(
    DiffTensorView intrinsics,  // (11,) packed intrinsics [cx, cy, fx, fy, k1-k4, ab, resolution_x, resolution_y]
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView image_points,
    no_diff bool does_generate_elements,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    OpenCVFisheyeCameraModelParameters projection = load_opencv_fisheye_params(
        intrinsics, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    image_points_to_world_rays_shutter_pose_kernel<OpenCVFisheyeCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        image_points, does_generate_elements, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// F-Theta + No External Distortion
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_shutter_pose_ftheta_no_external(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView image_points,
    no_diff bool does_generate_elements,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    NoExternalDistortion external_distortion;
    image_points_to_world_rays_shutter_pose_kernel<FThetaCameraModelParameters, NoExternalDistortion>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        image_points, does_generate_elements, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

// F-Theta + Bivariate Windshield
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield(
    DiffTensorView intrinsics,  // (22,) packed [principal_point(2), fw_poly(6), bw_poly(6), A(4), Ainv(4)]
    no_diff int reference_poly,
    no_diff uint fw_poly_degree,
    no_diff uint bw_poly_degree,
    no_diff float max_angle,
    no_diff int newton_iterations,
    no_diff float min_2d_norm,
    DiffTensorView ext_distortion_coeffs,  // (40,) packed [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
    no_diff int ext_reference_polynomial,
    no_diff uint ext_h_poly_degree,
    no_diff uint ext_v_poly_degree,
    uint2 resolution,
    no_diff int shutter_type,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView image_points,
    no_diff bool does_generate_elements,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    FThetaCameraModelParameters projection = load_ftheta_params(
        intrinsics, reference_poly, fw_poly_degree, bw_poly_degree, max_angle, newton_iterations, min_2d_norm
    );
    BivariateWindshieldModelParameters external_distortion = load_bivariate_windshield_undistort_params(
        ext_distortion_coeffs, ext_reference_polynomial, ext_h_poly_degree, ext_v_poly_degree
    );
    image_points_to_world_rays_shutter_pose_kernel<FThetaCameraModelParameters, BivariateWindshieldModelParameters>(
        projection, external_distortion, resolution, ShutterType(shutter_type),
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        image_points, does_generate_elements, world_rays,
        timestamps_us, poses_translation, poses_rotation
    );
}

