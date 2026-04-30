// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// LiDAR parameter structures implementing projection interface

#include <sensors/kernels/lidars/interface.slang>
#include <slang_utils/constants.slang>
#include <slang_utils/extensions.slang>

// ============================================================================
// Constants
// ============================================================================

static const float PI = constants::pi<float>();
static const float TWO_PI = 2.0 * constants::pi<float>();
static const float MAX_DIFF = 1e10;

// ============================================================================
// LiDAR Projection Implementation
// ============================================================================

// Structured spinning LiDAR projection (covers both basic and row-offset cases)
// Compatible with Hesai P128, Waymo, Pandar, and basic structured LiDARs
// IDifferentiable inherited from ILidarProjection - allows gradient flow through struct members
struct RowOffsetStructuredSpinningLidarProjection : ILidarProjection {
    // Structure (non-differentiable configuration)
    no_diff int n_rows;
    no_diff int n_columns;

    // Geometry - differentiable tensor views for gradient flow
    DiffTensorView row_elevations_rad;      // (n_rows,) elevation angles - differentiable
    DiffTensorView column_azimuths_rad;     // (n_columns,) azimuth angles - differentiable
    DiffTensorView row_azimuth_offsets_rad; // (n_rows,) azimuth offsets - differentiable
    no_diff bool has_row_offsets;           // Whether row offsets are present

    // Spinning behavior - non-differentiable configuration
    no_diff float spinning_frequency_hz;       // Rotation frequency (0.0 for non-spinning)
    no_diff SpinningDirection spinning_direction;  // CW or CCW

    // Field of view - non-differentiable configuration
    no_diff float fov_horiz_start_rad;  // Start of horizontal FOV
    no_diff float fov_horiz_span_rad;   // Span of horizontal FOV
    no_diff float fov_vert_start_rad;   // Start of vertical FOV
    no_diff float fov_vert_span_rad;    // Span of vertical FOV

    // Inverse projection optimization (optional precomputed 2D map) - non-differentiable
    // Map shape: (resolution_factor * n_rows, resolution_factor * n_columns)
    // Indexed by (elevation_idx, azimuth_idx) -> column index
    no_diff TensorView<int> angles_to_columns_map;     // Optional: 2D precomputed map
    no_diff bool has_angles_map;
    no_diff int angles_to_columns_map_resolution_factor;
    no_diff int angles_to_columns_map_height;  // resolution_factor * n_rows
    no_diff int angles_to_columns_map_width;   // resolution_factor * n_columns

    // Helper: Normalize angle to [-PI, PI]
    [Differentiable]
    float normalize_angle(float angle) {
        // Use fmod for robust handling of large values
        angle = fmod(angle + PI, TWO_PI);
        if (angle < 0.0) angle += TWO_PI;
        return angle - PI;
    }

    // Helper: Spherical to Cartesian conversion
    [Differentiable]
    float3 spherical_to_cartesian(float elevation, float azimuth) {
        float cos_elev = cos(elevation);
        return float3(
            cos_elev * cos(azimuth),
            cos_elev * sin(azimuth),
            sin(elevation)
        );
    }

    // Helper: Cartesian to spherical conversion
    [Differentiable]
    float2 cartesian_to_spherical(float3 ray) {
        float xy_norm = sqrt(ray.x * ray.x + ray.y * ray.y);
        float elevation = atan2(ray.z, xy_norm);
        float azimuth = atan2(ray.y, ray.x);
        return float2(elevation, azimuth);
    }

    // ========================================================================
    // ILidarProjection Interface Implementation
    // ========================================================================

    // Get number of columns (for relative time calculation)
    int get_n_columns() {
        return n_columns;
    }

    // Get number of rows (for element generation)
    int get_n_rows() {
        return n_rows;
    }

    [Differentiable]
    float2 sensor_rays_to_sensor_angles(float3 sensor_ray) {
        return cartesian_to_spherical(sensor_ray);
    }

    [Differentiable]
    float3 sensor_angles_to_sensor_rays(float2 sensor_angles) {
        float elevation = sensor_angles.x;
        float azimuth = sensor_angles.y;
        return spherical_to_cartesian(elevation, azimuth);
    }

    [Differentiable]
    float2 elements_to_sensor_angles(int2 element, out bool valid) {
        int row = element.x;
        int col = element.y;

        // Bounds check - set valid flag and return zero angles for out-of-bounds elements
        if (row < 0 || row >= n_rows || col < 0 || col >= n_columns) {
            valid = false;
            return float2(0.0, 0.0);
        }

        valid = true;

        // Get base elevation and azimuth - use .load() for DiffTensorView
        float elevation = row_elevations_rad.load(uint(row));
        float azimuth = column_azimuths_rad.load(uint(col));

        // Apply row offset if present
        if (has_row_offsets) {
            azimuth += row_azimuth_offsets_rad.load(uint(row));
            azimuth = normalize_angle(azimuth);
        }

        return float2(elevation, azimuth);
    }

    // ========================================================================
    // Additional Helper Functions
    // ========================================================================

    // Compute relative frame time [0,1] from azimuth angle
    // Used for rolling shutter pose interpolation
    [Differentiable]
    float azimuth_to_relative_time(float azimuth) {
        // Normalize azimuth to [0, 2*PI) using fmod for robust handling
        float normalized_az = fmod(azimuth, TWO_PI);
        if (normalized_az < 0.0) normalized_az += TWO_PI;

        // Compute relative position within FOV
        float relative_az = (normalized_az - fov_horiz_start_rad) / fov_horiz_span_rad;

        // Handle spinning direction
        if (spinning_direction == SpinningDirection.COUNTERCLOCKWISE) {
            relative_az = 1.0 - relative_az;
        }

        // Clamp to [0, 1]
        return clamp(relative_az, 0.0, 1.0);
    }

    // Find nearest row index for a given elevation angle.
    // Returns discrete index - not differentiable by nature.
    int find_nearest_row(float elevation) {
        float min_diff = MAX_DIFF;
        int best_row = 0;
        
        for (int row = 0; row < n_rows; row++) {
            float row_elev = row_elevations_rad.load(uint(row));
            float diff = abs(elevation - row_elev);
            if (diff < min_diff) {
                min_diff = diff;
                best_row = row;
            }
        }
        
        return best_row;
    }

    // Find nearest column index for a given azimuth angle.
    // Returns discrete index - not differentiable by nature.
    int find_nearest_column(float azimuth) {
        // Use precomputed map if available - O(1) lookup (preferred path)
        if (has_angles_map) {
            // Normalize azimuth to [0, TWO_PI) using fmod
            float normalized_az = fmod(azimuth, TWO_PI);
            if (normalized_az < 0.0) normalized_az += TWO_PI;
            
            // Compute map index: map covers [0, TWO_PI) with map_size entries
            // map_size = n_columns * angles_to_columns_map_resolution_factor
            int map_size = n_columns * angles_to_columns_map_resolution_factor;
            int index = int((normalized_az / TWO_PI) * float(map_size));
            
            // Clamp to valid range (handles edge case where normalized_az == TWO_PI)
            index = clamp(index, 0, map_size - 1);
            
            return angles_to_columns_map.load(uint(index));
        }

        // Fallback: Linear search for nearest column - O(n)
        // TODO: If column_azimuths_rad is monotonically sorted (typical for LiDARs),
        // this could be optimized to O(log n) with binary search. However, the O(1)
        // map lookup above is the preferred path for performance-critical code.
        float min_diff = MAX_DIFF;
        int best_col = 0;
        
        for (int col = 0; col < n_columns; col++) {
            float col_az = column_azimuths_rad.load(uint(col));
            float diff = abs(normalize_angle(azimuth - col_az));
            if (diff < min_diff) {
                min_diff = diff;
                best_col = col;
            }
        }
        
        return best_col;
    }

    // Compute relative frame time [0,1] from sensor angles
    // This accounts for row azimuth offsets when finding the column
    // Used for inverse projection to ensure consistency with forward projection
    float sensor_angles_to_relative_time(float2 sensor_angles) {
        float elevation = sensor_angles.x;
        float azimuth = sensor_angles.y;
        
        // Use 2D map if available - O(1) lookup
        if (has_angles_map) {
            // Compute relative angles from FOV start
            // Vertical: relative to fov_vert_start, going down (cw convention)
            float rel_elevation = fov_vert_start_rad - elevation;
            if (rel_elevation < 0.0) rel_elevation = 0.0;
            
            // Horizontal: relative to fov_horiz_start
            float rel_azimuth;
            if (spinning_direction == SpinningDirection.COUNTERCLOCKWISE) {
                rel_azimuth = azimuth - fov_horiz_start_rad;
            } else {
                rel_azimuth = fov_horiz_start_rad - azimuth;
            }
            // Normalize to [0, 2*PI)
            rel_azimuth = fmod(rel_azimuth, TWO_PI);
            if (rel_azimuth < 0.0) rel_azimuth += TWO_PI;
            
            // Compute map indices with nearest-neighbor rounding (+0.5)
            // Map resolution in radians
            float map_res_vert = fov_vert_span_rad / float(angles_to_columns_map_height - 1);
            float map_res_horiz = fov_horiz_span_rad / float(angles_to_columns_map_width - 1);
            
            int vert_idx = int(rel_elevation / map_res_vert + 0.5);
            int horiz_idx = int(rel_azimuth / map_res_horiz + 0.5);
            
            // Clamp to valid range
            vert_idx = clamp(vert_idx, 0, angles_to_columns_map_height - 1);
            horiz_idx = clamp(horiz_idx, 0, angles_to_columns_map_width - 1);
            
            // 2D lookup: map[vert_idx, horiz_idx]
            int flat_idx = vert_idx * angles_to_columns_map_width + horiz_idx;
            int col = angles_to_columns_map.load(uint(flat_idx));
            
            // Compute relative time from column index
            return (n_columns > 1) ? float(col) / float(n_columns - 1) : 0.0f;
        }
        
        // Fallback: manual row/column search
        // Find the row from elevation
        int row = find_nearest_row(elevation);
        
        // Adjust azimuth for row offset to get the base column azimuth
        float adjusted_azimuth = azimuth;
        if (has_row_offsets) {
            adjusted_azimuth -= row_azimuth_offsets_rad.load(uint(row));
        }
        
        // Find the column from the adjusted azimuth
        int col = find_nearest_column(adjusted_azimuth);
        
        // Compute relative time from column index
        return (n_columns > 1) ? float(col) / float(n_columns - 1) : 0.0f;
    }

    // Check if sensor angles are within valid FOV
    // Uses "cw convention": fov_vert_start_rad is the TOP (highest elevation),
    // and the valid range is [fov_vert_start_rad - fov_vert_span_rad, fov_vert_start_rad]
    [Differentiable]
    bool is_in_fov(float2 sensor_angles) {
        float elevation = sensor_angles.x;
        float azimuth = sensor_angles.y;

        // Check vertical FOV (cw convention: start at top, go down)
        // Valid range: [fov_vert_start_rad - fov_vert_span_rad, fov_vert_start_rad]
        if (elevation > fov_vert_start_rad || elevation < fov_vert_start_rad - fov_vert_span_rad) {
            return false;
        }

        // Check horizontal FOV (handle wraparound)
        // Normalize azimuth relative to fov_horiz_start
        float rel_azimuth;
        if (spinning_direction == SpinningDirection.COUNTERCLOCKWISE) {
            rel_azimuth = azimuth - fov_horiz_start_rad;
        } else {
            rel_azimuth = fov_horiz_start_rad - azimuth;
        }
        // Normalize to [0, 2*PI)
        rel_azimuth = fmod(rel_azimuth, TWO_PI);
        if (rel_azimuth < 0.0) rel_azimuth += TWO_PI;

        if (rel_azimuth > fov_horiz_span_rad) {
            return false;
        }

        return true;
    }
};

