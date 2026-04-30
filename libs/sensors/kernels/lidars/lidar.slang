// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// LiDAR device functions - core projection and transformation logic
// These functions can be called from CUDA kernels or fused kernels

#pragma once

// Include pose interfaces from geometry module
#include <geometry/kernels/pose.slang>

// Include common sensor utilities
#include <sensors/kernels/common/interface.slang>
#include <sensors/kernels/common/interpolation.slang>

// Include LiDAR interfaces
#include <sensors/kernels/lidars/interface.slang>

// Include shared constants
#include <slang_utils/constants.slang>

// ============================================================================
// Constants
// ============================================================================

static const float LIDAR_PI = constants::pi<float>();
static const float LIDAR_TWO_PI = 2.0 * constants::pi<float>();

// Maximum iterations for rolling shutter convergence (compile-time bound for autodiff)
static const int MAX_ROLLING_SHUTTER_ITERATIONS = 32;

// ============================================================================
// Output Structs for Device Functions
// ============================================================================

typedef WorldRay LidarWorldRay;

struct LidarInverseProjectResult {
    float2 sensor_angles;  // (elevation, azimuth)
    bool valid;
};

// ============================================================================
// Device Function: Sensor rays to sensor angles
// ============================================================================

// Converts a sensor ray to sensor angles (elevation, azimuth).
// Simple coordinate conversion (Cartesian to spherical), no pose transformation.
//
// Parameters:
//   sensor_ray: Normalized direction vector in sensor frame
//
// Returns:
//   float2 (elevation, azimuth) in radians
//
[Differentiable]
float2 sensor_rays_to_sensor_angles(float3 sensor_ray) {
    // Convert to spherical coordinates
    float xy_norm = sqrt(sensor_ray.x * sensor_ray.x + sensor_ray.y * sensor_ray.y);
    float elevation = atan2(sensor_ray.z, xy_norm);
    float azimuth = atan2(sensor_ray.y, sensor_ray.x);
    
    return float2(elevation, azimuth);
}

// ============================================================================
// Device Function: Sensor angles to sensor rays
// ============================================================================

// Converts sensor angles (elevation, azimuth) to a sensor ray.
// Simple coordinate conversion (spherical to Cartesian), no pose transformation.
//
// Parameters:
//   sensor_angles: float2 (elevation, azimuth) in radians
//
// Returns:
//   Normalized direction vector in sensor frame
//
[Differentiable]
float3 sensor_angles_to_sensor_rays(float2 sensor_angles) {
    float elevation = sensor_angles.x;
    float azimuth = sensor_angles.y;
    
    float cos_elev = cos(elevation);
    float3 sensor_ray = float3(
        cos_elev * cos(azimuth),
        cos_elev * sin(azimuth),
        sin(elevation)
    );
    
    return sensor_ray;
}

// ============================================================================
// Device Function: Elements to sensor angles
// ============================================================================

// Converts element indices (row, col) to sensor angles (elevation, azimuth).
// No pose transformation, just angle lookup.
//
// Parameters:
//   row, col: Element indices
//   n_rows, n_columns: Sensor dimensions
//   row_elevation: Elevation angle for this row (radians)
//   column_azimuth: Base azimuth angle for this column (radians)
//   row_azimuth_offset: Per-row azimuth offset (radians), 0 if not used
//   has_row_offsets: Whether row offsets are applied
//
// Outputs:
//   valid: Set to false if element indices are out of bounds
//
// Returns:
//   float2 (elevation, azimuth) in radians
//
[Differentiable]
float2 elements_to_sensor_angles(
    int row,
    int col,
    int n_rows,
    int n_columns,
    float row_elevation,
    float column_azimuth,
    float row_azimuth_offset,
    bool has_row_offsets,
    out bool valid
) {
    // Bounds check - set valid flag for out-of-bounds elements
    if (row < 0 || row >= n_rows || col < 0 || col >= n_columns) {
        valid = false;
        return float2(0.0, 0.0);
    }
    
    valid = true;
    
    float elevation = row_elevation;
    float azimuth = column_azimuth;
    
    // Apply row offset if present
    if (has_row_offsets) {
        azimuth += row_azimuth_offset;
        // Normalize to [-PI, PI]
        azimuth = fmod(azimuth + LIDAR_PI, LIDAR_TWO_PI);
        if (azimuth < 0.0) azimuth += LIDAR_TWO_PI;
        azimuth -= LIDAR_PI;
    }
    
    return float2(elevation, azimuth);
}

// ============================================================================
// Device Function: Azimuth to relative time
// ============================================================================

// Computes relative frame time [0,1] from azimuth angle.
// Used for rolling shutter pose interpolation.
//
// Parameters:
//   azimuth: Azimuth angle in radians
//   spinning_direction: 0 = CW, 1 = CCW
//   fov_horiz_start_rad: Start of horizontal FOV
//   fov_horiz_span_rad: Span of horizontal FOV
//
// Returns:
//   Relative time in [0, 1]
//
[Differentiable]
float azimuth_to_relative_time(
    float azimuth,
    SpinningDirection spinning_direction,
    float fov_horiz_start_rad,
    float fov_horiz_span_rad
) {
    // Normalize to [0, TWO_PI)
    float normalized_az = fmod(azimuth, LIDAR_TWO_PI);
    if (normalized_az < 0.0) normalized_az += LIDAR_TWO_PI;
    
    // Guard against division by zero (degenerate FOV)
    float relative_az = (fov_horiz_span_rad == 0.0) ? 0.0 : (normalized_az - fov_horiz_start_rad) / fov_horiz_span_rad;
    if (spinning_direction == SpinningDirection.COUNTERCLOCKWISE) {
        relative_az = 1.0 - relative_az;
    }
    return clamp(relative_az, 0.0, 1.0);
}

// Note: interpolate_pose_2poses is provided by common/trajectory.slang

// ============================================================================
// Device Function: Element to world ray (for fused kernels)
// ============================================================================

// Generates a world ray for a spinning LiDAR element with rolling shutter support.
// This is the main device function for use in fused kernels (e.g., gsplat).
// Takes row/column indices and pre-loaded angle parameters directly.
//
// Parameters:
//   row, col: Element indices
//   n_rows, n_columns: Sensor dimensions (for bounds checking)
//   row_elevation: Elevation angle for this row (radians), pre-loaded from buffer
//   column_azimuth: Base azimuth angle for this column (radians), pre-loaded from buffer
//   row_azimuth_offset: Per-row azimuth offset (radians), 0 if not used
//   has_row_offsets: Whether row offsets are applied
//   spinning_direction: CW or CCW (unused for forward projection, kept for API consistency)
//   fov_horiz_start_rad, fov_horiz_span_rad: Horizontal FOV bounds (unused for forward projection)
//   control_trans0, control_trans1: Translation of control poses
//   control_rot0, control_rot1: Rotation quaternions of control poses
//   control_time0, control_time1: Times of control poses [0, 1]
//
// Outputs:
//   valid: Set to false if element indices are out of bounds
//
// Returns:
//   LidarWorldRay with origin and direction in world frame
//
[Differentiable]
LidarWorldRay element_to_world_ray(
    int row,
    int col,
    int n_rows,
    int n_columns,
    float row_elevation,
    float column_azimuth,
    float row_azimuth_offset,
    bool has_row_offsets,
    SpinningDirection spinning_direction,
    float fov_horiz_start_rad,
    float fov_horiz_span_rad,
    float3 control_trans0,
    float3 control_trans1,
    float4 control_rot0,
    float4 control_rot1,
    float control_time0,
    float control_time1,
    out bool valid
) {
    // Convert element to sensor angles (also checks bounds)
    float2 sensor_angles = elements_to_sensor_angles(
        row, col, n_rows, n_columns,
        row_elevation, column_azimuth, row_azimuth_offset,
        has_row_offsets,
        valid
    );
    
    // Compute relative time from column index directly.
    // Column index represents the exact time ordering of the scan - column 0 is captured first,
    // column (n_columns-1) is captured last. This is more accurate than computing from azimuth
    // because row azimuth offsets mean different rows have different azimuths for the same column.
    float relative_time = (n_columns > 1) ? float(col) / float(n_columns - 1) : 0.0f;
    
    // Interpolate pose
    SE3Pose pose = interpolate_pose_2poses(
        relative_time,
        control_trans0, control_trans1,
        control_rot0, control_rot1,
        control_time0, control_time1
    );
    
    // Generate sensor ray from angles
    float3 sensor_ray = sensor_angles_to_sensor_rays(sensor_angles);
    
    // Transform to world frame
    LidarWorldRay result;
    result.origin = pose.translation;
    result.direction = pose.transform_direction(sensor_ray);
    
    return result;
}

// ============================================================================
// Device Function: Generate spinning lidar ray (from sensor angles)
// ============================================================================

// Generates a world ray for a spinning LiDAR element with rolling shutter support.
// Handles pose interpolation internally based on azimuth-derived relative time.
//
// NOTE: This function computes relative time from azimuth angle. For forward projection
// where element indices (row, col) are available, prefer element_to_world_ray() which
// uses the column index directly for more accurate time computation. This function
// is intended for cases where only sensor_angles are available (e.g., synthetic ray generation).
//
// Parameters:
//   sensor_angles: (elevation, azimuth) in radians
//   spinning_direction: CW or CCW
//   fov_horiz_start_rad, fov_horiz_span_rad: Horizontal FOV bounds
//   control_trans0, control_trans1: Translation of control poses
//   control_rot0, control_rot1: Rotation quaternions of control poses
//   control_time0, control_time1: Times of control poses [0, 1]
//
// Returns:
//   LidarWorldRay with origin and direction in world frame
//
[Differentiable]
LidarWorldRay generate_spinning_lidar_ray(
    float2 sensor_angles,
    SpinningDirection spinning_direction,
    float fov_horiz_start_rad,
    float fov_horiz_span_rad,
    float3 control_trans0,
    float3 control_trans1,
    float4 control_rot0,
    float4 control_rot1,
    float control_time0,
    float control_time1
) {
    float elevation = sensor_angles.x;
    float azimuth = sensor_angles.y;
    
    // Compute relative time from azimuth
    float relative_time = azimuth_to_relative_time(
        azimuth, spinning_direction, fov_horiz_start_rad, fov_horiz_span_rad
    );
    
    // Interpolate pose
    SE3Pose pose = interpolate_pose_2poses(
        relative_time,
        control_trans0, control_trans1,
        control_rot0, control_rot1,
        control_time0, control_time1
    );
    
    // Generate sensor ray from angles
    float3 sensor_ray = sensor_angles_to_sensor_rays(sensor_angles);
    
    // Transform to world frame
    LidarWorldRay result;
    result.origin = pose.translation;
    result.direction = pose.transform_direction(sensor_ray);
    
    return result;
}

// ============================================================================
// Device Function: Single-step inverse projection
// ============================================================================

// Result struct for single-step inverse projection
struct LidarInverseProjectStepResult {
    float2 sensor_angles;  // (elevation, azimuth)
    float new_relative_time;  // Updated relative time based on computed azimuth
    bool valid;  // Whether the point is within FOV
};

// Performs a single step of inverse projection given a pose.
// This is the core building block for iterative rolling shutter refinement.
// Used by the 2-pose device function for fused kernels.
//
// NOTE: This device function computes relative_time from azimuth alone (using FOV parameters).
// For maximum accuracy with row-offset sensors, use inverse_project_spinning_lidar_kernel()
// which has access to the full projection struct and accounts for row offsets when finding
// the column index. The difference is typically small for sensors with small row offsets.
//
// Parameters:
//   world_point: Point in world coordinates
//   pose: SE3Pose at current relative time
//   spinning_direction: 0 = CW, 1 = CCW
//   fov_horiz_start_rad, fov_horiz_span_rad: Horizontal FOV bounds
//   fov_vert_start_rad, fov_vert_span_rad: Vertical FOV bounds
//
// Returns:
//   LidarInverseProjectStepResult with angles, new relative time, and validity
//
[Differentiable]
LidarInverseProjectStepResult inverse_project_step(
    float3 world_point,
    SE3Pose pose,
    SpinningDirection spinning_direction,
    float fov_horiz_start_rad,
    float fov_horiz_span_rad,
    float fov_vert_start_rad,
    float fov_vert_span_rad
) {
    LidarInverseProjectStepResult result;
    result.sensor_angles = float2(0.0, 0.0);
    result.new_relative_time = 0.5;
    result.valid = false;
    
    // Transform world point to sensor frame
    float3 sensor_point = pose.inverse_transform_point(world_point);
    
    // Convert to sensor angles
    float2 current_angles = sensor_rays_to_sensor_angles(normalize(sensor_point));
    float elevation = current_angles.x;
    float azimuth = current_angles.y;
    
    // Check if within FOV
    // Check vertical FOV
    if (elevation < fov_vert_start_rad || elevation > fov_vert_start_rad + fov_vert_span_rad) {
        return result;
    }
    
    // Check horizontal FOV
    float normalized_az = fmod(azimuth - fov_horiz_start_rad, LIDAR_TWO_PI);
    if (normalized_az < 0.0) normalized_az += LIDAR_TWO_PI;
    normalized_az += fov_horiz_start_rad;
    if (normalized_az > fov_horiz_start_rad + fov_horiz_span_rad) {
        return result;
    }
    
    result.sensor_angles = current_angles;
    result.valid = true;
    
    // Compute new relative time from azimuth
    result.new_relative_time = azimuth_to_relative_time(
        azimuth, spinning_direction, fov_horiz_start_rad, fov_horiz_span_rad
    );
    
    return result;
}

// ============================================================================
// Device Function: Inverse project spinning lidar (2-pose trajectory)
// ============================================================================

// Projects a world point to sensor angles with rolling shutter support.
// Uses iterative refinement since the correct pose depends on the azimuth angle.
// This version is optimized for 2-pose trajectories (common case for fused kernels).
//
// Parameters:
//   world_point: Point in world coordinates
//   spinning_direction: 0 = CW, 1 = CCW
//   fov_horiz_start_rad, fov_horiz_span_rad: Horizontal FOV bounds
//   fov_vert_start_rad, fov_vert_span_rad: Vertical FOV bounds
//   control_trans0, control_trans1: Translation of control poses
//   control_rot0, control_rot1: Rotation quaternions of control poses
//   control_time0, control_time1: Times of control poses [0, 1]
//   max_iterations: Maximum rolling shutter iterations
//   stop_mean_relative_time_error: Stopping criterion for absolute error
//   stop_delta_mean_relative_time_error: Stopping criterion for error change
//
// Returns:
//   LidarInverseProjectResult with sensor angles and validity flag
//
[Differentiable]
LidarInverseProjectResult inverse_project_spinning_lidar(
    float3 world_point,
    SpinningDirection spinning_direction,
    float fov_horiz_start_rad,
    float fov_horiz_span_rad,
    float fov_vert_start_rad,
    float fov_vert_span_rad,
    float3 control_trans0,
    float3 control_trans1,
    float4 control_rot0,
    float4 control_rot1,
    float control_time0,
    float control_time1,
    int max_iterations,
    float stop_mean_relative_time_error,
    float stop_delta_mean_relative_time_error
) {
    LidarInverseProjectResult result;
    result.sensor_angles = float2(0.0, 0.0);
    result.valid = false;
    
    // Initialize relative time
    float relative_time = 0.5;  // Start at middle of frame
    float prev_time_diff = 1.0;
    
    // Rolling shutter iteration (clamp to MaxIters bound for correct adjoint)
    int iter_limit = min(max_iterations, MAX_ROLLING_SHUTTER_ITERATIONS);
    [MaxIters(MAX_ROLLING_SHUTTER_ITERATIONS)]
    for (int iter = 0; iter < iter_limit; iter++) {
        // Interpolate pose
        SE3Pose pose = interpolate_pose_2poses(
            relative_time,
            control_trans0, control_trans1,
            control_rot0, control_rot1,
            control_time0, control_time1
        );
        
        // Perform single projection step
        LidarInverseProjectStepResult step = inverse_project_step(
            world_point, pose,
            spinning_direction,
            fov_horiz_start_rad, fov_horiz_span_rad,
            fov_vert_start_rad, fov_vert_span_rad
        );
        
        if (!step.valid) {
            result.valid = false;
            break;
        }
        
        result.sensor_angles = step.sensor_angles;
        result.valid = true;
        
        // Check convergence
        float time_diff = abs(step.new_relative_time - relative_time);
        
        if (time_diff < stop_mean_relative_time_error) {
            break;  // Converged
        }
        
        float delta_time_diff = abs(time_diff - prev_time_diff);
        if (iter > 0 && delta_time_diff < stop_delta_mean_relative_time_error) {
            break;  // Converged - error not changing significantly
        }
        
        prev_time_diff = time_diff;
        relative_time = step.new_relative_time;
    }
    
    return result;
}

