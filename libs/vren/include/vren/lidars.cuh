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

#include "lidars.h"

#include "rolling_shutter.cuh"
#include <ku/helper_math.cuh>

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------------------------

// Lidar models

template <class DerivedLidarModel>
struct BaseLidarModel {

    struct SensorAngleReturn {
        float2 sensor_angle;
        bool valid_flag;
    };

    struct SensorRayReturn {
        float3 sensor_ray;
        bool valid_flag;
    };

    struct WorldPointToSensorAngleReturn {
        float2 sensor_angle;
        bool valid_flag;
        int64_t timestamp_us;
        Pose3 T_world_sensor;
    };

    struct WorldRayReturn {
        Ray3 world_ray;
        int64_t timestamp_us;
        Pose3 T_sensor_world;
    };
};

struct RowOffsetStructuredSpinningLidarModel : BaseLidarModel<RowOffsetStructuredSpinningLidarModel> {
    RowOffsetStructuredSpinningLidarModelParameters parameters;

    RowOffsetStructuredSpinningLidarModel(RowOffsetStructuredSpinningLidarModelParameters const& parameters)
        : parameters(parameters) {}

    // Retrieves the azimuth and elevation angles for elements in the structured lidar model. Elements are given as (row, column) indices.
    inline __device__ float2 element_to_sensor_angle(const int2& element) const {
        // elements: N x 2 array of (row, column) indices
        // reconstruct angles from model parameterization
        const float elevation_rad = parameters._row_elevations_rad[element.x];
        const float azimuth_rad   = parameters._column_azimuths_rad[element.y] + parameters._row_azimuth_offsets_rad[element.x];
        return {
            __normalize_angle(elevation_rad),
            __normalize_angle(azimuth_rad),
        };
    }

    // Computes normalized 3d sensor ray directions for elements in the structured lidar model. Elements are given as (row, column) indices.
    inline __device__ float3 element_to_sensor_ray(const int2& element) const {
        const auto sensor_angle = element_to_sensor_angle(element);
        return sensor_angle_to_sensor_ray(sensor_angle).sensor_ray;
    }

    // Computes 3d sensor points for elements in the structured lidar model. Elements are given as (row, column) indices.
    inline __device__ float3 element_to_sensor_point(const int2& element, float element_distance) const {
        const auto sensor_ray = element_to_sensor_ray(element);
        return sensor_ray * element_distance;
    }

    // Computes the elevation and azimuth angles for normalized 3d sensor rays.
    inline __device__ SensorAngleReturn sensor_ray_to_sensor_angle(float3 sensor_ray, bool normalized = true) const {
        if (!normalized) {
            sensor_ray = sensor_ray / length(sensor_ray);
        }

        const auto elevation_rad = asinf(sensor_ray.z);
        const auto azimuth_rad   = atan2f(sensor_ray.y, sensor_ray.x);
        const auto sensor_angle  = make_float2(elevation_rad, azimuth_rad);

        return {
            sensor_angle,
            __is_valid_angle(sensor_angle),
        };
    }

    // Computes the sensor rays for elevation/azimuth angles.
    inline __device__ SensorRayReturn sensor_angle_to_sensor_ray(const float2& sensor_angle) const {
        // sensor_angles: N x 2 array of elevation and azimuth angles
        const float elevation_rad = sensor_angle.x;
        const float azimuth_rad   = sensor_angle.y;
        const float cos_elevation = cosf(elevation_rad);
        const float x             = cosf(azimuth_rad) * cos_elevation;
        const float y             = sinf(azimuth_rad) * cos_elevation;
        const float z             = sinf(elevation_rad);

        return {
            {x, y, z},
            __is_valid_angle(sensor_angle),
        };
    }

    inline __device__ float sensor_angle_relative_frame_time(const float2& sensor_angle) const {
        // ARGS: sensor_angles: N x 2 array of elevation and azimuth angles in radians
        // NOTE: all sensor angles are assumed to be in the vertical fov of the sensor
        assert(parameters._angles_to_columns_map != nullptr);

        auto const elevations_rad = sensor_angle.x;
        auto const azimuths_rad   = sensor_angle.y;

        auto const relative_elevation_rad =
            relative_angle(parameters.fov_vert_start_rad, elevations_rad, SpinningDirection::CLOCK_WISE);
        auto const relative_azimuth_rad =
            relative_angle(parameters.fov_horiz_start_rad, azimuths_rad, parameters.spinning_direction);

        // Check that all angles are in the foV.
        // Lidar model is only precise up to float32 so we need to add some epsilon for when comparing in float64
        assert(relative_elevation_rad <= parameters.fov_vert_span_rad + 3.f * std::numeric_limits<float>::epsilon());
        assert(relative_azimuth_rad <= parameters.fov_horiz_span_rad + 3.f * std::numeric_limits<float>::epsilon());

        auto const n_pts_horiz = parameters.n_columns * parameters.angles_to_columns_map_resolution_factor; // N_columns
        auto const n_pts_vert  = parameters.n_rows * parameters.angles_to_columns_map_resolution_factor;    // N_rows

        // Compute the relative frame times by dividing the angles with the resolution of the map
        // Clip the indices to ensure that they are in valid range
        auto const horizontal_idx =
            std::clamp(static_cast<int>(
                           relative_azimuth_rad / parameters.map_resolution_horiz_rad + .5f), // = (azimuth + res/2) / res
                       0, n_pts_horiz - 1);
        auto const vertical_idx =
            std::clamp(static_cast<int>(
                           relative_elevation_rad / parameters.map_resolution_vert_rad + .5f), // = (elevation + res/2) / res
                       0, n_pts_vert - 1);

        // Grab the corresponding column index from the map
        const int* angles_to_columns_map = parameters._angles_to_columns_map;
        const auto column_index          = angles_to_columns_map[vertical_idx * n_pts_horiz + horizontal_idx];

        // Compute the relative frame time using the column-associated relative time
        return static_cast<float>(column_index) / (parameters.n_columns - 1);
    }

    template <size_t N_ROLLING_SHUTTER_ITERATIONS>
    inline __device__ WorldPointToSensorAngleReturn
    world_point_to_image_point_shutter_pose(float3 const& world_point, RollingShutterParameters const& rolling_shutter_parameters, float margin_factor) const {
        return world_point_to_sensor_angle_shutter_pose<N_ROLLING_SHUTTER_ITERATIONS>(world_point, RollingShutter(rolling_shutter_parameters));
    }

    template <size_t N_ROLLING_SHUTTER_ITERATIONS>
    inline __device__ WorldPointToSensorAngleReturn
    world_point_to_sensor_angle_shutter_pose(float3 const& world_point, RollingShutter const& rolling_shutter) const {
        const auto T_world_sensor_s = rolling_shutter.pose_start();
        const auto T_world_sensor_e = rolling_shutter.pose_end();

        // Do initial transformations using both start and end pose to determine all candidate
        // points and take union of valid projections as iteration starting points.
        auto const [sensor_angle_s, valid_s] = sensor_ray_to_sensor_angle(T_world_sensor_s.transform_point(world_point), false);
        auto const [sensor_angle_e, valid_e] = sensor_ray_to_sensor_angle(T_world_sensor_e.transform_point(world_point), false);

        // This selection prefers points at the start-of-frame pose over end-of-frame points
        // - the optimization will determine the final timestamp for each point
        auto init_sensor_angle = float2{};

        if (valid_s) {
            init_sensor_angle = sensor_angle_s;
        } else if (valid_e) {
            init_sensor_angle = sensor_angle_e;
        } else {
            // No valid projection at start or finish -> mark point as invalid. Still
            // return projection result at end of frame to be consistent with ncore, as
            // this will be condensed at the python interface level
            return {
                sensor_angle_e,
                false,
                (int64_t)rolling_shutter.timestamp_end_us(),
                T_world_sensor_e,
            };
        }

        // For valid image points, compute the new timestamp and project again
        auto sensor_angle_rs_prev = init_sensor_angle;
        auto valid_rs_prev        = true;
        auto relative_frame_time  = float{};
        auto T_world_sensor_rs    = Pose3{};

#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            relative_frame_time = sensor_angle_relative_frame_time(sensor_angle_rs_prev);
            T_world_sensor_rs   = rolling_shutter.interpolate_shutter_pose(relative_frame_time);
            auto const [sensor_angle_rs, valid_rs] =
                sensor_ray_to_sensor_angle(T_world_sensor_rs.transform_point(world_point), false);
            sensor_angle_rs_prev = sensor_angle_rs;
            valid_rs_prev        = valid_rs;
        }

        const auto timestamp_us_rs = rolling_shutter.interpolate_timestamps_us(relative_frame_time);

        return {
            sensor_angle_rs_prev,
            valid_rs_prev,
            timestamp_us_rs,
            T_world_sensor_rs,
        };
    }

    inline __device__ WorldRayReturn element_to_world_ray_shutter_pose(const int2& element, RollingShutter const& rolling_shutter) const {
        auto sensor_ray = element_to_sensor_ray(element);

        const auto relative_frame_time = __get_element_timestamp(element);

        // need to invert poses *before* interpolation
        auto const T_sensor_world_start = rolling_shutter.pose_start().inverse();
        auto const T_sensor_world_end   = rolling_shutter.pose_end().inverse();

        auto const T_sensor_world = Pose3{(1.f - relative_frame_time) * T_sensor_world_start.t + relative_frame_time * T_sensor_world_end.t,
                                          unitquat_slerp(T_sensor_world_start.q, T_sensor_world_end.q, relative_frame_time)};

        return {
            T_sensor_world.transform_local_ray(sensor_ray),
            rolling_shutter.interpolate_timestamps_us(relative_frame_time),
            T_sensor_world,
        };
    }

    inline __device__ int2 sensor_angle_to_tile_index(const float2 angle) const {
        const int32_t resolution = parameters.elevation_cdf_resolution;

        const float elevation_rad = angle.x;
        const float azimuth_rad   = angle.y;

        const float relative_elevation_normalized =
            relative_angle(parameters.fov_vert_start_rad, elevation_rad, SpinningDirection::CLOCK_WISE) /
            parameters.fov_vert_span_rad * resolution;
        const float relative_azimuth_normalized =
            relative_angle(parameters.fov_horiz_start_rad, azimuth_rad, parameters.spinning_direction) /
            parameters.fov_horiz_span_rad * parameters.n_bins_azimuth;

        const int32_t azimuths_index   = static_cast<int>(relative_azimuth_normalized) % parameters.n_bins_azimuth; // azimuths is periodic
        const int32_t elevations_index = parameters._cdf_elevation[std::clamp(static_cast<int>(relative_elevation_normalized), 0, resolution - 1)];

        return {elevations_index, azimuths_index};
    }

    // Output range is [0, 2π)
    // NOTE(qi): It is important to ensure 0 is closed
    static inline __device__ float relative_angle(float ref_angle_rad, float angle_rad, SpinningDirection direction) {
        const float M_2PI = 2.f * M_PI; // 2π constant
        // Compute the relative angle between two angles in radians
        //    ref_angle_rad: reference angle in radians
        //    angle_rad: angle to compute the relative angle to
        //    direction: spinning direction of the lidar
        const float relative_angle_rad = (direction == SpinningDirection::CLOCK_WISE)
                                             ? fmod(ref_angle_rad - angle_rad, M_2PI)
                                             : fmod(angle_rad - ref_angle_rad, M_2PI);
        // output range [0, 2π)
        return (relative_angle_rad < 0) ? (relative_angle_rad + M_2PI) : relative_angle_rad;
    }

    // Get the timestamp of the elements
    inline __device__ float __get_element_timestamp(const int2& element) const {
        // Get the row and column indices
        const auto& column = element.y;
        // Get the timestamp of the elements
        return (float)column / (parameters.n_columns - 1);
    }

    // Checks if a sensor angle is valid within the sensor's field of view
    inline __device__ bool __is_valid_angle(const float2& sensor_angle) const {
        const float elevation_rad          = sensor_angle.x;
        const float azimuth_rad            = sensor_angle.y;
        const float relative_elevation_rad = elevation_rad - parameters.fov_vert_start_rad;
        const float relative_azimuth_rad   = relative_angle(parameters.fov_horiz_start_rad, azimuth_rad, parameters.spinning_direction);
        return (relative_elevation_rad <= parameters.fov_vert_span_rad + 3.f * std::numeric_limits<float>::epsilon()) &&
               (relative_azimuth_rad <= parameters.fov_horiz_span_rad + 3.f * std::numeric_limits<float>::epsilon());
    }

    // Normalizes angle to the interval (-π, π]
    static inline __device__ float __normalize_angle(float angle_rad) {
        const float M_3PI = 3.f * M_PI; // 3π constant
        const float M_2PI = 2.f * M_PI; // 2π constant
        // branch-less execution of azimuth wrapping
        // NOTE(qi): avoid using fmod() if possible for numerical stability
        if (-M_3PI < angle_rad && angle_rad <= M_3PI) {
            angle_rad = (angle_rad > M_PI) ? angle_rad - M_2PI : angle_rad;
            angle_rad = (angle_rad <= -M_PI) ? angle_rad + M_2PI : angle_rad;
            return angle_rad;
        } else {
            angle_rad = fmod(angle_rad + M_PI, M_2PI);
            angle_rad = (angle_rad <= 0) ? (angle_rad + M_2PI) : angle_rad;
            return angle_rad - M_PI;
        }
    }
};
