// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Camera device functions - core projection and transformation logic
// These functions can be called from CUDA kernels or fused kernels (e.g., gsplat)

#pragma once

// Include pose interfaces from geometry module
#include <geometry/kernels/pose.slang>

// Include common sensor utilities
#include <sensors/kernels/common/interface.slang>
#include <sensors/kernels/common/interpolation.slang>

// Include camera interfaces and parameters
#include <sensors/kernels/cameras/interface.slang>
#include <sensors/kernels/cameras/parameters.slang>

// ============================================================================
// Constants
// ============================================================================

// Maximum iterations for rolling shutter convergence (compile-time bound for autodiff)
static const int MAX_ROLLING_SHUTTER_ITERATIONS = 12;

// ============================================================================
// Output Structs for Device Functions
// ============================================================================

// Note: CameraProjectResult is defined in interface.slang

// ============================================================================
// Device Function: Compute relative frame time from shutter type
// ============================================================================

// Computes relative frame time [0,1] based on shutter type and image coordinates.
// Used for rolling shutter pose interpolation.
//
// Parameters:
//   image_point: (x, y) pixel coordinates
//   resolution: (width, height) sensor resolution
//   shutter_type: Rolling shutter direction or global
//
// Returns:
//   Relative time in [0, 1]
//
// Note: This function is intentionally NOT differentiable because floor/ceil are discontinuous.
// Call sites should use no_diff when calling this function.
//
float compute_relative_frame_time(float2 image_point, uint2 resolution, ShutterType shutter_type) {
    // Floor/ceil the continuous image points to the row/column index following the image coordinate
    // convention that index defines the top left corner of each pixel, e.g., the first pixel's
    // u/v-range is [0.0, 1.0). Division by (resolution - 1) maps exactly to [0, 1] range where
    // first scanline is at t=0 and last scanline is at t=1.

    // Avoid division by zero in degenerate resolutions (use max(res-1, 1) for divisor)
    float safe_width_divisor = float(max(resolution.x, 2u) - 1u);
    float safe_height_divisor = float(max(resolution.y, 2u) - 1u);

    if (shutter_type == ShutterType.GLOBAL) {
        return 0.0; // All pixels captured at frame start (exposure trigger time)
    } else if (shutter_type == ShutterType.ROLLING_TOP_TO_BOTTOM) {
        return floor(image_point.y) / safe_height_divisor;
    } else if (shutter_type == ShutterType.ROLLING_BOTTOM_TO_TOP) {
        return (float(resolution.y) - ceil(image_point.y)) / safe_height_divisor;
    } else if (shutter_type == ShutterType.ROLLING_LEFT_TO_RIGHT) {
        return floor(image_point.x) / safe_width_divisor;
    } else { // ROLLING_RIGHT_TO_LEFT
        return (float(resolution.x) - ceil(image_point.x)) / safe_width_divisor;
    }
}

// ============================================================================
// Device Function: Check if image point is within sensor bounds
// ============================================================================

[Differentiable]
bool is_in_bounds(float2 image_point, uint2 resolution) {
    return image_point.x >= 0.0 && image_point.x < float(resolution.x) &&
           image_point.y >= 0.0 && image_point.y < float(resolution.y);
}

// Note: interpolate_pose_2poses is provided by common/trajectory.slang

// ============================================================================
// Device Function: Project camera ray to image point
// ============================================================================

// Projects a camera ray to image coordinates using the given projection model.
// Applies external distortion before camera projection.
//
// Parameters:
//   camera_ray: Normalized direction vector in camera frame
//   projection: Camera projection parameters (implements ICameraProjection)
//   external_distortion: External distortion parameters (implements IExternalDistortion)
//
// Returns:
//   CameraProjectResult with image point and validity flag
//
[Differentiable]
CameraProjectResult project_camera_ray<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    float3 camera_ray,
    Projection projection,
    ExternalDist external_distortion
) {
    // Apply external distortion
    float3 distorted_ray = external_distortion.distort(camera_ray);

    // Apply camera projection
    return projection.camera_rays_to_image_points(distorted_ray);
}

// ============================================================================
// Device Function: Back-project image point to camera ray
// ============================================================================

// Back-projects an image point to a camera ray (normalized direction).
// Removes external distortion after inverse camera projection.
//
// Parameters:
//   image_point: (x, y) pixel coordinates
//   projection: Camera projection parameters (implements ICameraProjection)
//   external_distortion: External distortion parameters (implements IExternalDistortion)
//
// Returns:
//   Normalized camera ray in camera frame
//
[Differentiable]
float3 backproject_to_camera_ray<
    Projection: ICameraProjection,
    ExternalDist: IExternalDistortion
>(
    float2 image_point,
    Projection projection,
    ExternalDist external_distortion
) {
    // Back-project to camera ray (with distortion)
    float3 distorted_ray = projection.image_points_to_camera_rays(image_point);

    // Remove external distortion
    float3 camera_ray = external_distortion.undistort(distorted_ray);

    return normalize(camera_ray);
}

