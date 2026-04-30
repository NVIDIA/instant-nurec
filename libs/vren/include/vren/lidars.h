// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "rolling_shutter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

enum class SpinningDirection {
    CLOCK_WISE,
    COUNTER_CLOCK_WISE
};

struct BaseSpinningLidarModelParameters {
    //
    // Represents parameters common to all spinning lidar models
    //

    float spinning_frequency_hz;
    SpinningDirection spinning_direction;
};

struct BaseStructuredSpinningLidarModelParameters : BaseSpinningLidarModelParameters {
    //
    // Represents parameters for a structured spinning lidar model.
    //
    // A structured lidar model consists of a fixed number of rows x columns point measurements per frame.
    //
    // Representation
    //     sensor_angle.x --> elevation
    //     sensor_angle.y --> azimuth
    //

    int n_rows;    // number of rows
    int n_columns; // number of columns

    float fov_horiz_start_rad; // horizontal angle that is measured "first" in each spin [around z axis, relative to x axis] [radians]
    float fov_horiz_span_rad;  // span of the horizontal field of view [radians in [0, 2π]]

    float fov_vert_start_rad; // vertical angle that is measured "first" in each spin [around y axis, relative to z axis] [radians]
    float fov_vert_span_rad;  // span of the vertical field of view [radians in [0, 2π]]
};

struct RowOffsetStructuredSpinningLidarModelParameters : BaseStructuredSpinningLidarModelParameters {
    //
    // Represents parameters for a row offset structured spinning lidar model.
    //

    // elevation angles: elevation angle of each row, constant for each column [around y axis, relative to x axis] [(Nrows,) radians]
    float* _row_elevations_rad = nullptr; // device buffer of size N_ROWS

    // azimuth angles: azimuth angle of each column [around z axis, relative to x axis] [(Ncolumns,) radians]
    float* _column_azimuths_rad = nullptr; // device buffer of size N_COLUMNS

    // azimuth angle offsets for each row [around z axis, relative to x axis] [(Nrows,) radians]
    float* _row_azimuth_offsets_rad = nullptr; // device buffer of size N_ROWS

    // TODO: consider if rolling shutter & tiling implementation can be generalized to all StructuredSpinningLidarModels

    // derived variables for rolling shutter
    int* _angles_to_columns_map = nullptr; // device buffer of size (n_rows*factor) * (n_columns*factor)
    int angles_to_columns_map_resolution_factor;
    float map_resolution_horiz_rad; // horizental (azimuth) size of each map cell measured in radians
    float map_resolution_vert_rad;  // vertical (elevation) size of each map cell measured in radians

    // derived variables for LiDAR tiling
    int* _cdf_elevation      = nullptr; // device buffer of size elevation_cdf_resolution + 1
    int* _cdf_dense_ray_mask = nullptr; // device buffer of size (azimuth_cdf_resolution + 1), (elevation_cdf_resolution + 1)
    uint32_t elevation_cdf_resolution;
    uint32_t azimuth_cdf_resolution;

    int* _tiles_to_elements_map = nullptr; // device buffer of size n_rows * n_columns * 2
    int* _tiles_pack_info       = nullptr; // device buffer of size n_bins_azimuth * n_bins_elevation * 2
    int n_bins_azimuth;
    int n_bins_elevation;
    int max_pts_per_tile;
};
