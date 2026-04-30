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

#include <ku/helper_math.cuh>

// ---------------------------------------------------------------------------------------------
struct Ray3 {
    // Struct encapsulating a ray with origin/direction in 3D space
    float3 org;
    float3 dir;
};

struct Pose3 {
    // Transforms from a source to a target frame rigidly (rotation / translation)
    // equivalent to a 4x4 SE3 matrix
    // [ mat3x3(q), t ]
    // [         0, 1 ]
    // where q is a unit quaternion and t is a translation vector
    // and mat3x3(q) is the 3x3 rotation matrix corresponding to the quaternion q
    // and 0 is a 3x1 zero vector
    float3 t; // position in target-frame coordinates
    float4 q; // source->target rotation, encoded as quaternion = { x, y, z, w }

    inline __device__ __host__ float3 transform_point(float3 const& point) const {
        return apply_quaternion(q, point) + t;
    }

    inline __device__ __host__ float3 transform_direction(float3 const& direction) const {
        return apply_quaternion(q, direction) /* + (0,0,0) omitted*/;
    }

    inline __device__ __host__ Ray3 transform_local_ray(float3 const& local_ray_dir) const {
        return {
            t,                                 // origin is (0, 0, 0) in local coordinates
            apply_quaternion(q, local_ray_dir) // do not apply translation
        };
    }

    inline __device__ __host__ Pose3 inverse() const {
        auto const q_inv = conjugate_quaternion(q);
        auto const t_inv = apply_quaternion(q_inv, make_float3(-t.x, -t.y, -t.z));
        return {t_inv, q_inv};
    }
};

struct RollingShutter {
    RollingShutterParameters parameters;

    __host__ __device__ RollingShutter(RollingShutterParameters const& parameters)
        : parameters(parameters) {
    }

    template <typename TensorPosesT, typename TensorTimestampsT>
    __host__ __device__ RollingShutter(
        TensorPosesT& const T_sensor_worlds,   // (2, 7) tensor-like tquat poses [f32]
        TensorTimestampsT& const timestamps_us // (2,) tensor-like timestamps [int64]
    ) {
        // Read poses from tensor and invert them (RollingShutterParameters expects T_world_sensor)
        auto const T_sensor_world_start_tquat = T_sensor_worlds[0];
        auto const T_sensor_world_end_tquat   = T_sensor_worlds[1];

        Pose3 const T_sensor_world_start{
            make_float3(T_sensor_world_start_tquat[0], T_sensor_world_start_tquat[1], T_sensor_world_start_tquat[2]),
            make_float4(T_sensor_world_start_tquat[3], T_sensor_world_start_tquat[4], T_sensor_world_start_tquat[5], T_sensor_world_start_tquat[6])};
        Pose3 const T_sensor_world_end{
            make_float3(T_sensor_world_end_tquat[0], T_sensor_world_end_tquat[1], T_sensor_world_end_tquat[2]),
            make_float4(T_sensor_world_end_tquat[3], T_sensor_world_end_tquat[4], T_sensor_world_end_tquat[5], T_sensor_world_end_tquat[6])};

        // Invert to get T_world_sensor (RollingShutter stores T_world_sensors)
        Pose3 const T_world_sensor_start = T_sensor_world_start.inverse();
        Pose3 const T_world_sensor_end   = T_sensor_world_end.inverse();

        // Store T_world_sensors in RollingShutterParameters
        parameters.T_world_sensors[0]  = T_world_sensor_start.t.x;
        parameters.T_world_sensors[1]  = T_world_sensor_start.t.y;
        parameters.T_world_sensors[2]  = T_world_sensor_start.t.z;
        parameters.T_world_sensors[3]  = T_world_sensor_start.q.x;
        parameters.T_world_sensors[4]  = T_world_sensor_start.q.y;
        parameters.T_world_sensors[5]  = T_world_sensor_start.q.z;
        parameters.T_world_sensors[6]  = T_world_sensor_start.q.w;
        parameters.T_world_sensors[7]  = T_world_sensor_end.t.x;
        parameters.T_world_sensors[8]  = T_world_sensor_end.t.y;
        parameters.T_world_sensors[9]  = T_world_sensor_end.t.z;
        parameters.T_world_sensors[10] = T_world_sensor_end.q.x;
        parameters.T_world_sensors[11] = T_world_sensor_end.q.y;
        parameters.T_world_sensors[12] = T_world_sensor_end.q.z;
        parameters.T_world_sensors[13] = T_world_sensor_end.q.w;

        // Store timestamps in RollingShutterParameters
        parameters.timestamps_us[0] = timestamps_us[0];
        parameters.timestamps_us[1] = timestamps_us[1];
    }

    inline __device__ __host__ float3 t_start() const {
        auto const& T = parameters.T_world_sensors;
        return make_float3(T[0], T[1], T[2]);
    }

    inline __device__ __host__ float4 q_start() const {
        auto const& T = parameters.T_world_sensors;
        return make_float4(T[3], T[4], T[5], T[6]); // xyzw representation
    }

    inline __device__ __host__ float3 t_end() const {
        auto const& T = parameters.T_world_sensors;
        return make_float3(T[7], T[8], T[9]);
    }

    inline __device__ __host__ float4 q_end() const {
        auto const& T = parameters.T_world_sensors;
        return make_float4(T[10], T[11], T[12], T[13]); // xyzw representation
    }

    inline __device__ __host__ Pose3 pose_start() const {
        return Pose3{t_start(), q_start()};
    }

    inline __device__ __host__ Pose3 pose_end() const {
        return Pose3{t_end(), q_end()};
    }

    inline __device__ __host__ const uint64_t& timestamp_start_us() const {
        return parameters.timestamps_us[0];
    }

    inline __device__ __host__ const uint64_t& timestamp_end_us() const {
        return parameters.timestamps_us[1];
    }

    inline __device__ __host__ float3 interpolate_shutter_pose_t(float relative_frame_time) const {
        // Interpolate a pose linearly for a relative frame time
        return (1.f - relative_frame_time) * t_start() + relative_frame_time * t_end();
    }

    inline __device__ __host__ float4 interpolate_shutter_pose_q(float relative_frame_time) const {
        // Interpolate a pose linearly for a relative frame time
        auto const q_s  = q_start();
        auto const q_e  = q_end();
        auto const q_rs = unitquat_slerp(q_s, q_e, relative_frame_time);
        return q_rs;
    }

    // inline __device__ __host__ glm::mat4x3 interpolate_shutter_pose_se3(float relative_frame_time) const {
    //     const auto q = interpolate_shutter_pose_q(relative_frame_time);
    //     return glm::mat4x3{
    //         glm::mat3_cast(glm::fquat{q.w, q.x, q.y, q.z}),
    //     };
    // }

    // Interpolate a pose linearly for a relative frame time
    inline __device__ __host__ Pose3 interpolate_shutter_pose(float relative_frame_time) const {
        auto const t_rs = interpolate_shutter_pose_t(relative_frame_time);
        auto const q_rs = interpolate_shutter_pose_q(relative_frame_time);
        return Pose3{t_rs, q_rs};
    }

    // Interpolate timestamps linearly for a relative frame time
    inline __device__ int64_t interpolate_timestamps_us(float relative_frame_time) const {
        auto const& t_start_us = parameters.timestamps_us[0];
        auto const& t_end_us   = parameters.timestamps_us[1];
        return t_start_us + int64_t(relative_frame_time * (t_end_us - t_start_us));
    }

    inline __device__ Ray3 sensor_ray_to_world_ray(float relative_frame_time, float3 const& sensor_ray) const {
        auto const T_world_sensor = interpolate_shutter_pose(relative_frame_time);
        auto const T_sensor_world = T_world_sensor.inverse();
        return T_sensor_world.transform_local_ray(sensor_ray);
    }

    inline __device__ __host__ float3 sensor_world_position(float relative_frame_time) const {
        auto const T_world_sensor = interpolate_shutter_pose(relative_frame_time);
        return apply_quaternion(conjugate_quaternion(T_world_sensor.q), -T_world_sensor.t);
    }
};

inline __device__ Pose3
interpolate_pose(float relative_frame_time, Pose3 const& T_world_sensor_start, Pose3 const& T_world_sensor_end) {
    // Interpolate a pose linearly for a relative frame time
    auto const t_rs = (1.f - relative_frame_time) * T_world_sensor_start.t + relative_frame_time * T_world_sensor_end.t;
    auto const q_rs = unitquat_slerp(T_world_sensor_start.q, T_world_sensor_end.q, relative_frame_time); // xyzw representation
    return {t_rs, q_rs};
}

inline __device__ int64_t interpolate_timestamp_us(float relative_frame_time, std::array<uint64_t, 2> const& timestamps_us) {
    // Interpolate timestamp linearly for a relative frame time (rounding to integer)
    auto const& t_start = timestamps_us[0];
    auto const& t_end   = timestamps_us[1];
    return t_start + int64_t(relative_frame_time * (t_end - t_start));
}
