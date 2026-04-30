// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

// LiDAR projection interfaces and common types

// Spinning direction enum
enum SpinningDirection : int {
    CLOCKWISE = 0,         // "cw" in Python
    COUNTERCLOCKWISE = 1   // "ccw" in Python
};

// Interface for LiDAR projection models
// Defines the conversion between elements/rays/angles
// Extends IDifferentiable to enable gradient flow through generic functions
interface ILidarProjection : IDifferentiable {
    // Get number of columns (for relative time calculation)
    // Column index directly represents capture time ordering
    int get_n_columns();

    // Get number of rows (for element generation)
    int get_n_rows();

    // Convert sensor rays to sensor angles (elevation, azimuth)
    // sensor_ray must be normalized
    [Differentiable]
    float2 sensor_rays_to_sensor_angles(float3 sensor_ray);

    // Convert sensor angles to sensor rays
    [Differentiable]
    float3 sensor_angles_to_sensor_rays(float2 sensor_angles);

    // Convert element indices (row, col) to sensor angles
    // Returns validity via out parameter (false if element is out of bounds)
    [Differentiable]
    float2 elements_to_sensor_angles(int2 element, out bool valid);

    // Compute relative frame time [0,1] from sensor angles
    // This accounts for row azimuth offsets when finding the column
    // Used for inverse projection to ensure consistency with forward projection
    float sensor_angles_to_relative_time(float2 sensor_angles);

    // Compute relative frame time [0,1] from azimuth angle alone
    // Fallback for cases where elevation is not available
    [Differentiable]
    float azimuth_to_relative_time(float azimuth);

    // Check if sensor angles are within valid FOV
    [Differentiable]
    bool is_in_fov(float2 sensor_angles);
};

