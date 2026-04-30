// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// LiDAR CUDA kernels - entry point for Python bindings via slangtorch
// These kernels are thin wrappers around LiDAR device functions defined in lidar.slang

// Include extensions for DiffTensorView operations
#include <slang_utils/extensions.slang>

// Include shared trajectory utilities
#include <sensors/kernels/common/trajectory.slang>

// Include LiDAR device functions
#include <sensors/kernels/lidars/lidar.slang>

// Include LiDAR parameter structures
#include <sensors/kernels/lidars/parameters.slang>

// ============================================================================
// Helper Functions for Loading Struct Parameters from Tensors
// ============================================================================

// Load RowOffsetStructuredSpinningLidarProjection from tensor views
// This creates the projection struct with all necessary tensor views for gradient flow
[Differentiable]
RowOffsetStructuredSpinningLidarProjection load_row_offset_structured_spinning_params(
    no_diff int n_rows,
    no_diff int n_columns,
    DiffTensorView row_elevations_rad,
    DiffTensorView column_azimuths_rad,
    DiffTensorView row_azimuth_offsets_rad,
    no_diff bool has_row_offsets,
    no_diff float spinning_frequency_hz,
    no_diff SpinningDirection spinning_direction,
    no_diff float fov_horiz_start_rad,
    no_diff float fov_horiz_span_rad,
    no_diff float fov_vert_start_rad,
    no_diff float fov_vert_span_rad,
    no_diff TensorView<int> angles_to_columns_map,
    no_diff bool has_angles_map,
    no_diff int angles_to_columns_map_resolution_factor,
    no_diff int angles_to_columns_map_height,
    no_diff int angles_to_columns_map_width
) {
    RowOffsetStructuredSpinningLidarProjection projection;
    projection.n_rows = n_rows;
    projection.n_columns = n_columns;
    projection.row_elevations_rad = row_elevations_rad;
    projection.column_azimuths_rad = column_azimuths_rad;
    projection.row_azimuth_offsets_rad = row_azimuth_offsets_rad;
    projection.has_row_offsets = has_row_offsets;
    projection.spinning_frequency_hz = spinning_frequency_hz;
    projection.spinning_direction = spinning_direction;
    projection.fov_horiz_start_rad = fov_horiz_start_rad;
    projection.fov_horiz_span_rad = fov_horiz_span_rad;
    projection.fov_vert_start_rad = fov_vert_start_rad;
    projection.fov_vert_span_rad = fov_vert_span_rad;
    projection.angles_to_columns_map = angles_to_columns_map;
    projection.has_angles_map = has_angles_map;
    projection.angles_to_columns_map_resolution_factor = angles_to_columns_map_resolution_factor;
    projection.angles_to_columns_map_height = angles_to_columns_map_height;
    projection.angles_to_columns_map_width = angles_to_columns_map_width;
    return projection;
}

// ============================================================================
// Generic Kernel Functions
// ============================================================================
// These generic functions work with any LiDAR projection type implementing ILidarProjection.
// slangtorch does not support generic kernels directly, so concrete wrappers are provided below.

// ============================================================================
// Generic: Element indices to world rays with rolling shutter
// ============================================================================

// Generates world rays for spinning LiDAR with rolling shutter support.
// Each element (row, col) maps to a specific sensor angle, and the sensor
// pose is interpolated based on column index (which directly represents capture time).
//
// Note: This is a generic helper function called by concrete wrappers below.
// Generic functions cannot be entry points in slangtorch.
[Differentiable]
void generate_spinning_lidar_rays_kernel<Projection: ILidarProjection>(
    Projection projection,
    DiffTensorView trajectory_control_poses_translation,  // (K, 3) control pose translations
    DiffTensorView trajectory_control_poses_rotation,     // (K, 4) control pose rotations
    no_diff TensorView<float> trajectory_control_times,   // (K,) control times [0, 1]
    no_diff uint trajectory_control_count,                // Number of control poses
    no_diff int64_t start_timestamp_us,                   // Start timestamp in microseconds
    no_diff int64_t end_timestamp_us,                     // End timestamp in microseconds
    DiffTensorView elements,               // (N, 2) [row, col] int input
    no_diff bool does_generate_elements,   // True if elements are generated, false if read from elements tensor
    DiffTensorView world_rays,             // (N, 6) [origin.xyz, direction.xyz] output
    no_diff TensorView<int64_t> timestamps_us,        // (N,) int64 output - absolute timestamp in microseconds
    no_diff TensorView<float> poses_translation,      // (N, 3) float output - per-ray interpolated pose translations
    no_diff TensorView<float> poses_rotation          // (N, 4) float output - per-ray interpolated pose rotations (quaternions)
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = does_generate_elements ? uint(projection.get_n_rows() * projection.get_n_columns()) : elements.size(0);
    if (tid >= count) return;

    // Load element indices (non-differentiable)
    int row, col;
    if (does_generate_elements) {
        row = tid / projection.get_n_columns();
        col = tid - row * projection.get_n_columns();
    } else {
        int2 element = int2(elements.loadVecOnce<2>(uint2(tid, 0)));
        row = element.x;
        col = element.y;
    }
    
    // Handle invalid cases with zero output
    bool valid = (trajectory_control_count > 0);
    
    // Get sensor angles using projection interface (also checks bounds)
    bool element_valid;
    float2 sensor_angles = projection.elements_to_sensor_angles(int2(row, col), element_valid);
    valid = valid && element_valid;
    
    // Compute relative time from column index directly.
    // Column index represents the exact time ordering of the scan - column 0 is captured first,
    // column (n_columns-1) is captured last. This is more accurate than computing from azimuth
    // because row azimuth offsets mean different rows have different azimuths for the same column.
    int n_cols = projection.get_n_columns();
    float relative_time = (n_cols > 1) 
        ? float(col) / float(n_cols - 1) 
        : 0.0f;
    
    // Find trajectory interval for pose interpolation using shared helper
    TrajectoryInterval interval = find_trajectory_interval(
        relative_time, trajectory_control_times, trajectory_control_count);
    
    // Load control poses for this interval - DIFFERENTIABLE
    // Use loadVecEx since interval indices are not related to thread idx
    float3 trans0 = valid ? trajectory_control_poses_translation.loadVecEx<3>(uint2(interval.idx0, 0)) : float3(0.0);
    float3 trans1 = valid ? trajectory_control_poses_translation.loadVecEx<3>(uint2(interval.idx1, 0)) : float3(0.0);
    float4 rot0 = valid ? trajectory_control_poses_rotation.loadVecEx<4>(uint2(interval.idx0, 0)) : float4(0.0, 0.0, 0.0, 1.0);
    float4 rot1 = valid ? trajectory_control_poses_rotation.loadVecEx<4>(uint2(interval.idx1, 0)) : float4(0.0, 0.0, 0.0, 1.0);
    
    // Interpolate pose using alpha from interval
    float3 trans = lerp(trans0, trans1, interval.alpha);
    float4 rot = quaternion::slerp(rot0, rot1, interval.alpha);
    
    // Generate sensor ray from angles using projection interface
    float3 sensor_ray = projection.sensor_angles_to_sensor_rays(sensor_angles);
    
    // Transform direction to world frame using quaternion rotation
    float3 world_direction = quaternion::rotateVector(rot, sensor_ray);
    
    // Final output
    float3 origin = valid ? trans : float3(0.0);
    float3 direction = valid ? world_direction : float3(0.0);
    
    // Store world ray using differentiable stores
    world_rays.storeVecOnce(uint2(tid, 0), origin);
    world_rays.storeVecOnce(uint2(tid, 3), direction);
    
    // Store timestamp and pose (non-differentiable outputs)
    int64_t timestamp = valid ? start_timestamp_us + int64_t(relative_time * float(end_timestamp_us - start_timestamp_us)) : start_timestamp_us;
    timestamps_us.store(tid, timestamp);
    poses_translation.storeVec(uint2(tid, 0), valid ? trans : float3(0.0, 0.0, 0.0));
    poses_rotation.storeVec(uint2(tid, 0), valid ? rot : float4(0.0, 0.0, 0.0, 1.0));
}

// ============================================================================
// Generic: Element indices to sensor angles
// ============================================================================

// Converts element indices to sensor angles (elevation, azimuth).
// No pose transformation, just angle lookup via projection interface.
// Used for preprocessing, analysis, and debugging.
//
// Outputs:
//   sensor_angles: (N, 2) [elevation, azimuth] float - angles for each element
//   valid_flags: (N,) bool - false if element indices are out of bounds
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void elements_to_sensor_angles_kernel<Projection: ILidarProjection>(
    Projection projection,
    DiffTensorView elements,               // (N, 2) [row, col] int input
    DiffTensorView sensor_angles,          // (N, 2) [elevation, azimuth] float output
    no_diff TensorView<bool> valid_flags   // (N,) bool output - false if out of bounds
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = elements.size(0);
    if (tid >= count) return;

    // Load element indices
    int row = int(elements.load(uint2(tid, 0)));
    int col = int(elements.load(uint2(tid, 1)));
    
    // Get sensor angles using projection interface (returns validity via out param)
    bool valid;
    float2 result = projection.elements_to_sensor_angles(int2(row, col), valid);
    
    // Store results
    sensor_angles.storeVecOnce(uint2(tid, 0), result);
    valid_flags.store(tid, valid);
}

// ============================================================================
// Generic: Inverse projection - World points to sensor angles with rolling shutter
// ============================================================================

// Projects world points to sensor angles with rolling shutter support.
// Uses iterative refinement since the correct pose depends on the azimuth angle.
// Properly handles multi-pose trajectories by finding the correct interval at each iteration.
//
// NOTE: Unlike forward projection which uses column index directly for relative time,
// inverse projection must compute relative time from azimuth since we don't know
// the column until after projection. For this lookup, while this implementation computes directly
// from azimuth via FOV parameters. Both approaches should converge to the same result through iteration.
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void inverse_project_spinning_lidar_kernel<Projection: ILidarProjection>(
    Projection projection,
    DiffTensorView trajectory_control_poses_translation,  // (K, 3) control pose translations
    DiffTensorView trajectory_control_poses_rotation,     // (K, 4) control pose rotations
    no_diff TensorView<float> trajectory_control_times,   // (K,) control times [0, 1]
    no_diff uint trajectory_control_count,                // Number of control poses
    no_diff int max_iterations,                           // Max rolling shutter iterations
    no_diff float stop_mean_relative_time_error,          // Stopping criterion for mean error
    no_diff float stop_delta_mean_relative_time_error,    // Stopping criterion for error change
    no_diff int64_t start_timestamp_us,                   // Start timestamp in microseconds
    no_diff int64_t end_timestamp_us,                     // End timestamp in microseconds
    DiffTensorView world_points,           // (N, 3) world coordinates input
    DiffTensorView sensor_angles,          // (N, 2) [elevation, azimuth] output
    no_diff TensorView<bool> valid_flags,  // (N,) bool output
    no_diff TensorView<int64_t> timestamps_us,        // (N,) int64 output - absolute timestamp in microseconds
    no_diff TensorView<float> poses_translation,      // (N, 3) float output - per-point interpolated pose translations
    no_diff TensorView<float> poses_rotation          // (N, 4) float output - per-point interpolated pose rotations (quaternions)
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = world_points.size(0);
    if (tid >= count) return;

    // No trajectory poses: emit invalid result to avoid out-of-bounds pose access
    if (trajectory_control_count == 0) {
        sensor_angles.storeVecOnce(uint2(tid, 0), float2(0.0, 0.0));
        valid_flags.store(tid, false);
        timestamps_us.store(tid, start_timestamp_us);
        poses_translation.storeVec(uint2(tid, 0), float3(0.0, 0.0, 0.0));
        poses_rotation.storeVec(uint2(tid, 0), float4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Load world point
    float3 world_point = world_points.loadVecOnce<3>(uint2(tid, 0));
    
    // Initialize result
    float2 result_angles = float2(0.0, 0.0);
    bool result_valid = false;
    
    // Initialize relative time
    float relative_time = 0.5;  // Start at middle of frame
    float prev_time_diff = 1.0;
    
    // Track final interpolated pose for output
    float3 final_trans = float3(0.0, 0.0, 0.0);
    float4 final_rot = float4(0.0, 0.0, 0.0, 1.0);
    
    // Rolling shutter iteration with proper multi-pose handling (clamp to MaxIters bound for correct adjoint)
    int iter_limit = min(max_iterations, MAX_ROLLING_SHUTTER_ITERATIONS);
    [MaxIters(MAX_ROLLING_SHUTTER_ITERATIONS)]
    for (int iter = 0; iter < iter_limit; iter++) {
        // Find trajectory interval using shared helper
        TrajectoryInterval interval = find_trajectory_interval(
            relative_time, trajectory_control_times, trajectory_control_count);
        
        // Load control poses for this interval
        // Use loadVecEx since interval indices are not related to thread idx
        float3 trans0 = trajectory_control_poses_translation.loadVecEx<3>(uint2(interval.idx0, 0));
        float3 trans1 = trajectory_control_poses_translation.loadVecEx<3>(uint2(interval.idx1, 0));
        float4 rot0 = trajectory_control_poses_rotation.loadVecEx<4>(uint2(interval.idx0, 0));
        float4 rot1 = trajectory_control_poses_rotation.loadVecEx<4>(uint2(interval.idx1, 0));
        
        // Interpolate pose using alpha from interval
        float3 trans = lerp(trans0, trans1, interval.alpha);
        float4 rot = quaternion::slerp(rot0, rot1, interval.alpha);
        
        // Store for final output
        final_trans = trans;
        final_rot = rot;
        
        // Create SE3Pose for transformation
        SE3Pose pose = SE3Pose.from_translation_quaternion(trans, rot);
        
        // Transform world point to sensor frame
        float3 sensor_point = pose.inverse_transform_point(world_point);
        
        // Convert to sensor angles using projection interface
        float3 sensor_ray = normalize(sensor_point);
        float2 angles = projection.sensor_rays_to_sensor_angles(sensor_ray);
        
        // Check if in FOV
        bool in_fov = projection.is_in_fov(angles);
        
        if (!in_fov) {
            result_valid = false;
            break;
        }
        
        result_angles = angles;
        result_valid = true;
        
        // Compute new relative time from sensor angles (accounts for row offsets)
        // This matches forward projection which uses column index directly
        float new_relative_time = projection.sensor_angles_to_relative_time(angles);
        
        // Check convergence
        float time_diff = abs(new_relative_time - relative_time);
        
        if (time_diff < stop_mean_relative_time_error) {
            break;  // Converged
        }
        
        float delta_time_diff = abs(time_diff - prev_time_diff);
        if (iter > 0 && delta_time_diff < stop_delta_mean_relative_time_error) {
            break;  // Converged - error not changing significantly
        }
        
        prev_time_diff = time_diff;
        relative_time = new_relative_time;
    }
    
    // Store results
    sensor_angles.storeVecOnce(uint2(tid, 0), result_angles);
    valid_flags.store(tid, result_valid);
    timestamps_us.store(tid, start_timestamp_us + int64_t(relative_time * float(end_timestamp_us - start_timestamp_us)));
    poses_translation.storeVec(uint2(tid, 0), final_trans);
    poses_rotation.storeVec(uint2(tid, 0), final_rot);
}

// ============================================================================
// Generic: Sensor rays to sensor angles
// ============================================================================

// Converts sensor rays to sensor angles (elevation, azimuth).
// Simple coordinate conversion, no pose transformation.
// Input rays must be normalized.
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void sensor_rays_to_sensor_angles_kernel<Projection: ILidarProjection>(
    Projection projection,
    DiffTensorView sensor_rays,            // (N, 3) normalized direction input
    DiffTensorView sensor_angles           // (N, 2) [elevation, azimuth] output
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = sensor_rays.size(0);
    if (tid >= count) return;

    // Load sensor ray
    float3 sensor_ray = sensor_rays.loadVecOnce<3>(uint2(tid, 0));
    
    // Use projection interface
    float2 result = projection.sensor_rays_to_sensor_angles(sensor_ray);
    
    // Store result
    sensor_angles.storeVecOnce(uint2(tid, 0), result);
}

// ============================================================================
// Generic: Sensor angles to sensor rays
// ============================================================================

// Converts sensor angles (elevation, azimuth) to sensor rays.
// Simple coordinate conversion, no pose transformation.
//
// Note: This is a generic helper function called by concrete wrappers below.
[Differentiable]
void sensor_angles_to_sensor_rays_kernel<Projection: ILidarProjection>(
    Projection projection,
    DiffTensorView sensor_angles,          // (N, 2) [elevation, azimuth] input
    DiffTensorView sensor_rays             // (N, 3) normalized direction output
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = sensor_angles.size(0);
    if (tid >= count) return;

    // Load sensor angles
    float2 angles = sensor_angles.loadVecOnce<2>(uint2(tid, 0));
    
    // Use projection interface
    float3 result = projection.sensor_angles_to_sensor_rays(angles);
    
    // Store result (already normalized)
    sensor_rays.storeVecOnce(uint2(tid, 0), result);
}

// ============================================================================
// Concrete Wrapper Kernels (non-generic versions for slangtorch compatibility)
// ============================================================================
// slangtorch does not support generic kernels, so we provide concrete
// instantiations for each LiDAR projection type.

// ----------------------------------------------------------------------------
// generate_spinning_lidar_rays: Concrete wrapper for RowOffsetStructuredSpinning
// ----------------------------------------------------------------------------

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void generate_spinning_lidar_rays_row_offset(
    no_diff int n_rows,
    no_diff int n_columns,
    DiffTensorView row_elevations_rad,
    DiffTensorView column_azimuths_rad,
    DiffTensorView row_azimuth_offsets_rad,
    no_diff int has_row_offsets,
    no_diff float spinning_frequency_hz,
    no_diff int spinning_direction,
    no_diff float fov_horiz_start_rad,
    no_diff float fov_horiz_span_rad,
    no_diff float fov_vert_start_rad,
    no_diff float fov_vert_span_rad,
    no_diff TensorView<int> angles_to_columns_map,
    no_diff int has_angles_map,
    no_diff int angles_to_columns_map_resolution_factor,
    no_diff int angles_to_columns_map_height,
    no_diff int angles_to_columns_map_width,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView elements,
    no_diff bool does_generate_elements,
    DiffTensorView world_rays,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    RowOffsetStructuredSpinningLidarProjection projection = load_row_offset_structured_spinning_params(
        n_rows, n_columns,
        row_elevations_rad, column_azimuths_rad, row_azimuth_offsets_rad,
        has_row_offsets != 0,
        spinning_frequency_hz, SpinningDirection(spinning_direction),
        fov_horiz_start_rad, fov_horiz_span_rad,
        fov_vert_start_rad, fov_vert_span_rad,
        angles_to_columns_map, has_angles_map != 0, angles_to_columns_map_resolution_factor,
        angles_to_columns_map_height, angles_to_columns_map_width
    );
    generate_spinning_lidar_rays_kernel<RowOffsetStructuredSpinningLidarProjection>(
        projection,
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        start_timestamp_us, end_timestamp_us,
        elements, does_generate_elements, world_rays, timestamps_us, poses_translation, poses_rotation
    );
}

// ----------------------------------------------------------------------------
// elements_to_sensor_angles: Concrete wrapper for RowOffsetStructuredSpinning
// ----------------------------------------------------------------------------

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void elements_to_sensor_angles_row_offset(
    no_diff int n_rows,
    no_diff int n_columns,
    DiffTensorView row_elevations_rad,
    DiffTensorView column_azimuths_rad,
    DiffTensorView row_azimuth_offsets_rad,
    no_diff int has_row_offsets,
    no_diff float spinning_frequency_hz,
    no_diff int spinning_direction,
    no_diff float fov_horiz_start_rad,
    no_diff float fov_horiz_span_rad,
    no_diff float fov_vert_start_rad,
    no_diff float fov_vert_span_rad,
    no_diff TensorView<int> angles_to_columns_map,
    no_diff int has_angles_map,
    no_diff int angles_to_columns_map_resolution_factor,
    no_diff int angles_to_columns_map_height,
    no_diff int angles_to_columns_map_width,
    DiffTensorView elements,
    DiffTensorView sensor_angles,
    no_diff TensorView<bool> valid_flags
) {
    RowOffsetStructuredSpinningLidarProjection projection = load_row_offset_structured_spinning_params(
        n_rows, n_columns,
        row_elevations_rad, column_azimuths_rad, row_azimuth_offsets_rad,
        has_row_offsets != 0,
        spinning_frequency_hz, SpinningDirection(spinning_direction),
        fov_horiz_start_rad, fov_horiz_span_rad,
        fov_vert_start_rad, fov_vert_span_rad,
        angles_to_columns_map, has_angles_map != 0, angles_to_columns_map_resolution_factor,
        angles_to_columns_map_height, angles_to_columns_map_width
    );
    elements_to_sensor_angles_kernel<RowOffsetStructuredSpinningLidarProjection>(
        projection, elements, sensor_angles, valid_flags
    );
}

// ----------------------------------------------------------------------------
// inverse_project_spinning_lidar: Concrete wrapper for RowOffsetStructuredSpinning
// ----------------------------------------------------------------------------

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void inverse_project_spinning_lidar_row_offset(
    no_diff int n_rows,
    no_diff int n_columns,
    DiffTensorView row_elevations_rad,
    DiffTensorView column_azimuths_rad,
    DiffTensorView row_azimuth_offsets_rad,
    no_diff int has_row_offsets,
    no_diff float spinning_frequency_hz,
    no_diff int spinning_direction,
    no_diff float fov_horiz_start_rad,
    no_diff float fov_horiz_span_rad,
    no_diff float fov_vert_start_rad,
    no_diff float fov_vert_span_rad,
    no_diff TensorView<int> angles_to_columns_map,
    no_diff int has_angles_map,
    no_diff int angles_to_columns_map_resolution_factor,
    no_diff int angles_to_columns_map_height,
    no_diff int angles_to_columns_map_width,
    DiffTensorView trajectory_control_poses_translation,
    DiffTensorView trajectory_control_poses_rotation,
    no_diff TensorView<float> trajectory_control_times,
    no_diff uint trajectory_control_count,
    no_diff int max_iterations,
    no_diff float stop_mean_relative_time_error,
    no_diff float stop_delta_mean_relative_time_error,
    no_diff int64_t start_timestamp_us,
    no_diff int64_t end_timestamp_us,
    DiffTensorView world_points,
    DiffTensorView sensor_angles,
    no_diff TensorView<bool> valid_flags,
    no_diff TensorView<int64_t> timestamps_us,
    no_diff TensorView<float> poses_translation,
    no_diff TensorView<float> poses_rotation
) {
    RowOffsetStructuredSpinningLidarProjection projection = load_row_offset_structured_spinning_params(
        n_rows, n_columns,
        row_elevations_rad, column_azimuths_rad, row_azimuth_offsets_rad,
        has_row_offsets != 0,
        spinning_frequency_hz, SpinningDirection(spinning_direction),
        fov_horiz_start_rad, fov_horiz_span_rad,
        fov_vert_start_rad, fov_vert_span_rad,
        angles_to_columns_map, has_angles_map != 0, angles_to_columns_map_resolution_factor,
        angles_to_columns_map_height, angles_to_columns_map_width
    );
    inverse_project_spinning_lidar_kernel<RowOffsetStructuredSpinningLidarProjection>(
        projection,
        trajectory_control_poses_translation, trajectory_control_poses_rotation,
        trajectory_control_times, trajectory_control_count,
        max_iterations, stop_mean_relative_time_error, stop_delta_mean_relative_time_error,
        start_timestamp_us, end_timestamp_us,
        world_points, sensor_angles, valid_flags,
        timestamps_us, poses_translation, poses_rotation
    );
}

// ----------------------------------------------------------------------------
// sensor_rays_to_sensor_angles: Concrete wrapper for RowOffsetStructuredSpinning
// ----------------------------------------------------------------------------

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void sensor_rays_to_sensor_angles_row_offset(
    no_diff int n_rows,
    no_diff int n_columns,
    DiffTensorView row_elevations_rad,
    DiffTensorView column_azimuths_rad,
    DiffTensorView row_azimuth_offsets_rad,
    no_diff int has_row_offsets,
    no_diff float spinning_frequency_hz,
    no_diff int spinning_direction,
    no_diff float fov_horiz_start_rad,
    no_diff float fov_horiz_span_rad,
    no_diff float fov_vert_start_rad,
    no_diff float fov_vert_span_rad,
    no_diff TensorView<int> angles_to_columns_map,
    no_diff int has_angles_map,
    no_diff int angles_to_columns_map_resolution_factor,
    no_diff int angles_to_columns_map_height,
    no_diff int angles_to_columns_map_width,
    DiffTensorView sensor_rays,
    DiffTensorView sensor_angles
) {
    RowOffsetStructuredSpinningLidarProjection projection = load_row_offset_structured_spinning_params(
        n_rows, n_columns,
        row_elevations_rad, column_azimuths_rad, row_azimuth_offsets_rad,
        has_row_offsets != 0,
        spinning_frequency_hz, SpinningDirection(spinning_direction),
        fov_horiz_start_rad, fov_horiz_span_rad,
        fov_vert_start_rad, fov_vert_span_rad,
        angles_to_columns_map, has_angles_map != 0, angles_to_columns_map_resolution_factor,
        angles_to_columns_map_height, angles_to_columns_map_width
    );
    sensor_rays_to_sensor_angles_kernel<RowOffsetStructuredSpinningLidarProjection>(
        projection, sensor_rays, sensor_angles
    );
}

// ----------------------------------------------------------------------------
// sensor_angles_to_sensor_rays: Concrete wrapper for RowOffsetStructuredSpinning
// ----------------------------------------------------------------------------

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void sensor_angles_to_sensor_rays_row_offset(
    no_diff int n_rows,
    no_diff int n_columns,
    DiffTensorView row_elevations_rad,
    DiffTensorView column_azimuths_rad,
    DiffTensorView row_azimuth_offsets_rad,
    no_diff int has_row_offsets,
    no_diff float spinning_frequency_hz,
    no_diff int spinning_direction,
    no_diff float fov_horiz_start_rad,
    no_diff float fov_horiz_span_rad,
    no_diff float fov_vert_start_rad,
    no_diff float fov_vert_span_rad,
    no_diff TensorView<int> angles_to_columns_map,
    no_diff int has_angles_map,
    no_diff int angles_to_columns_map_resolution_factor,
    no_diff int angles_to_columns_map_height,
    no_diff int angles_to_columns_map_width,
    DiffTensorView sensor_angles,
    DiffTensorView sensor_rays
) {
    RowOffsetStructuredSpinningLidarProjection projection = load_row_offset_structured_spinning_params(
        n_rows, n_columns,
        row_elevations_rad, column_azimuths_rad, row_azimuth_offsets_rad,
        has_row_offsets != 0,
        spinning_frequency_hz, SpinningDirection(spinning_direction),
        fov_horiz_start_rad, fov_horiz_span_rad,
        fov_vert_start_rad, fov_vert_span_rad,
        angles_to_columns_map, has_angles_map != 0, angles_to_columns_map_resolution_factor,
        angles_to_columns_map_height, angles_to_columns_map_width
    );
    sensor_angles_to_sensor_rays_kernel<RowOffsetStructuredSpinningLidarProjection>(
        projection, sensor_angles, sensor_rays
    );
}

// ============================================================================
// Simple Kernels (non-generic, pure math operations)
// ============================================================================
// These kernels don't need projection data - they're just coordinate conversions.
// Kept as simple non-generic versions for efficiency.

// Simple sensor rays to sensor angles (no projection needed)
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void sensor_rays_to_sensor_angles_kernel(
    DiffTensorView sensor_rays,            // (N, 3) normalized direction input
    DiffTensorView sensor_angles           // (N, 2) [elevation, azimuth] output
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = sensor_rays.size(0);
    if (tid >= count) return;

    // Load sensor ray
    float3 sensor_ray = sensor_rays.loadVecOnce<3>(uint2(tid, 0));
    
    // Use device function (pure math: Cartesian to spherical)
    float2 result = sensor_rays_to_sensor_angles(sensor_ray);
    
    // Store result
    sensor_angles.storeVecOnce(uint2(tid, 0), result);
}

// Simple sensor angles to sensor rays (no projection needed)
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void sensor_angles_to_sensor_rays_kernel(
    DiffTensorView sensor_angles,          // (N, 2) [elevation, azimuth] input
    DiffTensorView sensor_rays             // (N, 3) normalized direction output
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint count = sensor_angles.size(0);
    if (tid >= count) return;

    // Load sensor angles
    float2 angles = sensor_angles.loadVecOnce<2>(uint2(tid, 0));
    
    // Use device function (pure math: spherical to Cartesian)
    float3 result = sensor_angles_to_sensor_rays(angles);
    
    // Store result (already normalized)
    sensor_rays.storeVecOnce(uint2(tid, 0), result);
}
