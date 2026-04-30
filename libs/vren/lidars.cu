// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "utils.h"

#include <vren/lidars.cuh>

#include <c10/cuda/CUDAStream.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

void inline __device__ store(torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t> pose_accessor, Pose3 const& pose) {
    // Helper function to store an encoded 3d pose in a tensor accessor
    pose_accessor[0] = pose.t.x;
    pose_accessor[1] = pose.t.y;
    pose_accessor[2] = pose.t.z;
    pose_accessor[3] = pose.q.x;
    pose_accessor[4] = pose.q.y;
    pose_accessor[5] = pose.q.z;
    pose_accessor[6] = pose.q.w;
}

void inline __device__ store(torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t> point2_accessor, float2 const& point) {
    // Helper function to store an encoded 2d point a tensor accessor
    point2_accessor[0] = point.x;
    point2_accessor[1] = point.y;
}

void inline __device__ store(torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t> ray_accessor, Ray3 const& ray) {
    // Helper function to store an encoded 3d ray a tensor accessor
    ray_accessor[0] = ray.org.x;
    ray_accessor[1] = ray.org.y;
    ray_accessor[2] = ray.org.z;
    ray_accessor[3] = ray.dir.x;
    ray_accessor[4] = ray.dir.y;
    ray_accessor[5] = ray.dir.z;
}

template <class LidarModel>
__global__ void elements_to_sensor_rays_kernel(
    LidarModel lidar_model,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> const elements,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sensor_rays) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= elements.size(0)) // Point does not exist
        return;

    auto const element    = elements[point_idx];
    auto const sensor_ray = lidar_model.element_to_sensor_ray(make_int2(element[0], element[1]));

    auto local_sensor_ray = sensor_rays[point_idx];
    local_sensor_ray[0]   = sensor_ray.x;
    local_sensor_ray[1]   = sensor_ray.y;
    local_sensor_ray[2]   = sensor_ray.z;
}

torch::Tensor elements_to_sensor_rays_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& elements) {

    auto const elements_arg = torch::TensorArg{elements, "elements", 2};
    torch::checkAllContiguous(__func__, {elements_arg});

    auto const N_points = elements.size(0);
    torch::checkSize(__func__, elements_arg, {N_points, 2});

    auto sensor_rays = torch::zeros({N_points, 3}, elements.options().dtype(torch::kFloat32));

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
        elements_to_sensor_rays_kernel<<<blocks, threads, 0, stream>>>(
            lidar_model,
            elements.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            sensor_rays.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return sensor_rays;
}

template <class LidarModel>
__global__ void elements_to_sensor_points_kernel(
    LidarModel lidar_model,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> const elements,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> const distances,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sensor_points) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= elements.size(0)) // Point does not exist
        return;

    auto const element      = elements[point_idx];
    auto const sensor_point = lidar_model.element_to_sensor_point(make_int2(element[0], element[1]), (float)distances[point_idx]);

    auto local_sensor_point = sensor_points[point_idx];
    local_sensor_point[0]   = sensor_point.x;
    local_sensor_point[1]   = sensor_point.y;
    local_sensor_point[2]   = sensor_point.z;
}

torch::Tensor elements_to_sensor_points_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& elements, const torch::Tensor& distances) {

    auto const elements_arg  = torch::TensorArg{elements, "elements", 2};
    auto const distances_arg = torch::TensorArg{distances, "distances", 3};

    torch::checkAllContiguous(__func__, {elements_arg, distances_arg});

    auto const N_points = elements.size(0);
    torch::checkSize(__func__, elements_arg, {N_points, 2});
    torch::checkSize(__func__, distances_arg, {N_points});

    auto sensor_points = torch::zeros({N_points, 3}, elements.options().dtype(torch::kFloat32));

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
        elements_to_sensor_points_kernel<<<blocks, threads, 0, stream>>>(
            lidar_model,
            elements.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            distances.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            sensor_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return sensor_points;
}

template <class LidarModel>
__global__ void elements_to_sensor_angles_kernel(
    LidarModel lidar_model,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> const elements,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sensor_angles) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= elements.size(0)) // Point does not exist
        return;

    auto const element      = elements[point_idx];
    auto const sensor_angle = lidar_model.element_to_sensor_angle(make_int2(element[0], element[1]));

    auto local_sensor_angle = sensor_angles[point_idx];
    local_sensor_angle[0]   = sensor_angle.x;
    local_sensor_angle[1]   = sensor_angle.y;
}

torch::Tensor elements_to_sensor_angles_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& elements) {

    auto const elements_arg = torch::TensorArg{elements, "elements", 2};
    torch::checkAllContiguous(__func__, {elements_arg});

    auto const N_points = elements.size(0);
    torch::checkSize(__func__, elements_arg, {N_points, 2});

    auto sensor_angles = torch::zeros({N_points, 2}, elements.options().dtype(torch::kFloat32));

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
        elements_to_sensor_angles_kernel<<<blocks, threads, 0, stream>>>(
            lidar_model,
            elements.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            sensor_angles.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return sensor_angles;
}

template <class LidarModel>
__global__ void sensor_rays_to_sensor_angles_kernel(
    LidarModel lidar_model,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const sensor_rays,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sensor_angles,
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid_flag) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= sensor_rays.size(0)) // Point does not exist
        return;

    auto const sensor_ray = sensor_rays[point_idx];
    auto const ret        = lidar_model.sensor_ray_to_sensor_angle(make_float3(sensor_ray[0], sensor_ray[1], sensor_ray[2]));

    auto local_sensor_angle = sensor_angles[point_idx];
    local_sensor_angle[0]   = ret.sensor_angle.x;
    local_sensor_angle[1]   = ret.sensor_angle.y;

    valid_flag[point_idx] = ret.valid_flag;
}

std::tuple<torch::Tensor, torch::Tensor> sensor_rays_to_sensor_angles_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& sensor_rays) {

    auto const sensor_rays_arg = torch::TensorArg{sensor_rays, "sensor_rays", 2};
    torch::checkAllContiguous(__func__, {sensor_rays_arg});

    auto const N_rays = sensor_rays.size(0);
    torch::checkSize(__func__, sensor_rays_arg, {N_rays, 3});

    auto sensor_angles = torch::zeros({N_rays, 2}, sensor_rays.options());
    auto valid_flags   = torch::zeros({N_rays}, sensor_rays.options().dtype(torch::kBool));

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_rays + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
        sensor_rays_to_sensor_angles_kernel<<<blocks, threads, 0, stream>>>(
            lidar_model,
            sensor_rays.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sensor_angles.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            valid_flags.packed_accessor32<bool, 1, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return {sensor_angles, valid_flags};
}

template <class LidarModel>
__global__ void sensor_angles_to_sensor_rays_kernel(
    LidarModel lidar_model,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const sensor_angles,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sensor_rays,
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid_flag) {

    auto const ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= sensor_angles.size(0)) // Ray does not exist
        return;

    auto const sensor_angle = sensor_angles[ray_idx];
    auto const ret          = lidar_model.sensor_angle_to_sensor_ray(make_float2(sensor_angle[0], sensor_angle[1]));

    auto local_sensor_ray = sensor_rays[ray_idx];
    local_sensor_ray[0]   = ret.sensor_ray.x;
    local_sensor_ray[1]   = ret.sensor_ray.y;
    local_sensor_ray[2]   = ret.sensor_ray.z;

    valid_flag[ray_idx] = ret.valid_flag;
}

std::tuple<torch::Tensor, torch::Tensor> sensor_angles_to_sensor_rays_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& sensor_angles) {

    auto const sensor_angles_arg = torch::TensorArg{sensor_angles, "sensor_angles", 2};
    torch::checkAllContiguous(__func__, {sensor_angles_arg});

    auto const N_rays = sensor_angles.size(0);
    torch::checkSize(__func__, sensor_angles_arg, {N_rays, 2});

    auto sensor_rays = torch::zeros({N_rays, 3}, sensor_angles.options().dtype(torch::kFloat32));
    auto valid_flags = torch::zeros({N_rays}, sensor_angles.options().dtype(torch::kBool));

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_rays + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
        sensor_angles_to_sensor_rays_kernel<<<blocks, threads, 0, stream>>>(
            lidar_model,
            sensor_angles.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sensor_rays.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            valid_flags.packed_accessor32<bool, 1, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return {sensor_rays, valid_flags};
}

template <class LidarModel>
__global__ void sensor_angles_relative_frame_times_kernel(
    LidarModel lidar_model,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const sensor_angles,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> relative_frame_times) {

    auto const ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= sensor_angles.size(0)) // Ray does not exist
        return;

    auto const sensor_angle        = sensor_angles[ray_idx];
    auto const relative_frame_time = lidar_model.sensor_angle_relative_frame_time(make_float2(sensor_angle[0], sensor_angle[1]));
    relative_frame_times[ray_idx]  = relative_frame_time;
}

torch::Tensor sensor_angles_relative_frame_times_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor& sensor_angles) {

    auto const sensor_angles_arg = torch::TensorArg{sensor_angles, "sensor_angles", 3};
    torch::checkAllContiguous(__func__, {sensor_angles_arg});

    auto const N_points = sensor_angles.size(0);
    torch::checkSize(__func__, sensor_angles_arg, {N_points, 2});

    if (parameters._angles_to_columns_map == nullptr) {
        throw std::runtime_error("[vren]: \"angles_to_columns_map\" not set for \"sensor_angles_relative_frame_times\", "
                                 "likely due to LiDAR model not being initialized with rolling shutter information.");
    }

    auto relative_frame_times = torch::zeros({N_points}, sensor_angles.options().dtype(torch::kFloat32));

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
        sensor_angles_relative_frame_times_kernel<<<blocks, threads, 0, stream>>>(
            lidar_model,
            sensor_angles.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            relative_frame_times.packed_accessor32<float, 1, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return relative_frame_times;
}

template <size_t N_ROLLING_SHUTTER_ITERATIONS, class LidarModel>
__global__ void world_points_to_sensor_angles_shutter_pose_kernel(
    LidarModel lidar_model, RollingShutter rolling_shutter,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const world_points,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sensor_angles,
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid_flag,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> timestamps_us,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> T_world_sensors) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= world_points.size(0))
        return; // Ray does not exist

    auto const local_world_point = world_points[point_idx];
    auto const world_point       = make_float3(local_world_point[0], local_world_point[1], local_world_point[2]);

    const auto ret = lidar_model.template world_point_to_sensor_angle_shutter_pose<N_ROLLING_SHUTTER_ITERATIONS>(world_point, rolling_shutter);

    valid_flag[point_idx]    = ret.valid_flag;
    timestamps_us[point_idx] = ret.timestamp_us;
    store(sensor_angles[point_idx], ret.sensor_angle);
    store(T_world_sensors[point_idx], ret.T_world_sensor);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> world_points_to_sensor_angles_shutter_pose_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    RollingShutterParameters const rolling_shutter_parameters,
    const torch::Tensor& world_points) {

    auto const world_points_arg = torch::TensorArg{world_points, "world_points", 1};
    torch::checkAllContiguous(__func__, {world_points_arg});

    auto const N_rays = world_points.size(0);
    torch::checkSize(__func__, world_points_arg, {N_rays, 3});

    if (parameters._angles_to_columns_map == nullptr) {
        throw std::runtime_error("[vren]: \"angles_to_columns_map\" not set for \"world_points_to_sensor_angles_shutter_pose\", "
                                 "likely due to LiDAR model not being initialized with rolling shutter information.");
    }

    auto sensor_angles   = torch::zeros({N_rays, 2}, world_points.options());
    auto valid_flag      = torch::zeros({N_rays}, world_points.options().dtype(torch::kBool));
    auto timestamps_us   = torch::zeros({N_rays}, world_points.options().dtype(torch::kInt64));
    auto T_world_sensors = torch::zeros({N_rays, 7}, world_points.options());

    const auto rolling_shutter = RollingShutter(rolling_shutter_parameters);

    // fixed number of rolling-shutter iterations - same as in NCore
    auto constexpr N_ROLLING_SHUTTER_ITERATIONS = 10;

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_rays + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

        world_points_to_sensor_angles_shutter_pose_kernel<N_ROLLING_SHUTTER_ITERATIONS><<<blocks, threads, 0, stream>>>(
            lidar_model, rolling_shutter,
            world_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            sensor_angles.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            valid_flag.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            T_world_sensors.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return {sensor_angles, valid_flag, timestamps_us, T_world_sensors};
}

// Kernel that reads rolling shutter parameters from tensor global memory
template <class LidarModel>
__global__ void elements_to_world_rays_shutter_pose_kernel(
    LidarModel lidar_model,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const T_sensor_worlds, // (2, 7) tquat poses
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const timestamps_us, // (2,) timestamps
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> const elements,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> world_rays,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> timestamps_us_out,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> T_sensor_worlds_out) {

    RollingShutter rolling_shutter(T_sensor_worlds, timestamps_us);

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= elements.size(0)) // Point does not exist
        return;

    auto const local_element = elements[point_idx];
    auto const element       = make_int2(local_element[0], local_element[1]);

    auto local_world_ray      = world_rays[point_idx];
    auto& local_timestamp_us  = timestamps_us_out[point_idx];
    auto local_T_sensor_world = T_sensor_worlds_out[point_idx];

    auto const ret = lidar_model.element_to_world_ray_shutter_pose(element, rolling_shutter);

    store(local_world_ray, ret.world_ray);
    local_timestamp_us = ret.timestamp_us;
    store(local_T_sensor_world, ret.T_sensor_world);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> elements_to_world_rays_shutter_pose_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor T_sensor_worlds, // (2, 7) tensor with tquat poses
    const torch::Tensor timestamps_us,   // (2,) tensor with int64 timestamps
    const torch::Tensor& elements) {
    auto const elements_arg        = torch::TensorArg{elements, "elements", 1};
    auto const T_sensor_worlds_arg = torch::TensorArg{T_sensor_worlds, "T_sensor_worlds", 2};
    auto const timestamps_us_arg   = torch::TensorArg{timestamps_us, "timestamps_us", 3};

    torch::checkAllContiguous(__func__, {elements_arg, T_sensor_worlds_arg, timestamps_us_arg});

    auto const N_rays = elements.size(0);
    torch::checkSize(__func__, elements_arg, {N_rays, 2});
    torch::checkSize(__func__, T_sensor_worlds_arg, {2, 7});
    torch::checkSize(__func__, timestamps_us_arg, {2});

    auto world_rays          = torch::zeros({N_rays, 6}, elements.options().dtype(torch::kFloat32));
    auto timestamps_us_out   = torch::zeros({N_rays}, elements.options().dtype(torch::kInt64));
    auto T_sensor_worlds_out = torch::zeros({N_rays, 7}, elements.options().dtype(torch::kFloat32));

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_rays + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
        elements_to_world_rays_shutter_pose_kernel<<<blocks, threads, 0, stream>>>(
            lidar_model,
            T_sensor_worlds.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            elements.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            world_rays.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            timestamps_us_out.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            T_sensor_worlds_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return {world_rays, timestamps_us_out, T_sensor_worlds_out};
}

template <class LidarModel>
__global__ void sensor_angles_to_tile_indices_kernel(
    LidarModel lidar_model,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const sensor_angles,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> tile_indices) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= sensor_angles.size(0)) // Point does not exist
        return;

    auto const sensor_angle = sensor_angles[point_idx];

    auto const tile_index = lidar_model.sensor_angle_to_tile_index(make_float2(sensor_angle[0], sensor_angle[1]));

    auto local_tile_index = tile_indices[point_idx];
    local_tile_index[0]   = tile_index.x;
    local_tile_index[1]   = tile_index.y;
}

torch::Tensor sensor_angles_to_tile_indices_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor& sensor_angles) {

    auto const sensor_angles_arg = torch::TensorArg{sensor_angles, "sensor_angles", 3};
    torch::checkAllContiguous(__func__, {sensor_angles_arg});

    auto const N_points = sensor_angles.size(0);
    torch::checkSize(__func__, sensor_angles_arg, {N_points, 2});

    if (parameters._cdf_elevation == nullptr) {
        throw std::runtime_error("[vren]: \"cdf_elevation\" not set for \"sensor_angles_to_tile_indices\", "
                                 "likely due to LiDAR model not being initialized with tiling information.");
    }
    if (parameters._tiles_pack_info == nullptr || parameters._tiles_to_elements_map == nullptr) {
        throw std::runtime_error("[vren]: \"tiles_pack_info\" or \"tiles_to_elements_map\" not set for \"sensor_angles_to_tile_indices\", "
                                 "likely due to LiDAR model not being initialized with tiling information.");
    }

    auto tile_indices = torch::zeros({N_points, 2}, sensor_angles.options().dtype(torch::kInt32));

    auto launchKernel = [&](auto const& lidar_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
        sensor_angles_to_tile_indices_kernel<<<blocks, threads, 0, stream>>>(
            lidar_model,
            sensor_angles.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            tile_indices.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launchKernel(RowOffsetStructuredSpinningLidarModel(parameters));

    return tile_indices;
}

__global__ void normalize_sensor_angles_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const sensor_angles,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> normalized_sensor_angles) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= sensor_angles.size(0)) // Point does not exist
        return;

    auto const sensor_angle            = sensor_angles[point_idx];
    auto local_normalized_sensor_angle = normalized_sensor_angles[point_idx];
    local_normalized_sensor_angle[0]   = RowOffsetStructuredSpinningLidarModel::__normalize_angle(sensor_angle[0]);
    local_normalized_sensor_angle[1]   = RowOffsetStructuredSpinningLidarModel::__normalize_angle(sensor_angle[1]);
}

torch::Tensor normalize_sensor_angles_cu(
    const torch::Tensor& sensor_angles) {

    auto const sensor_angles_arg = torch::TensorArg{sensor_angles, "sensor_angles", 1};
    torch::checkAllContiguous(__func__, {sensor_angles_arg});

    auto const N_points = sensor_angles.size(0);
    torch::checkSize(__func__, sensor_angles_arg, {N_points, 2});

    auto normalized_sensor_angles = torch::zeros({N_points, 2}, sensor_angles.options().dtype(torch::kFloat32));

    auto const threads = dim3(256);
    auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    normalize_sensor_angles_kernel<<<blocks, threads, 0, stream>>>(
        sensor_angles.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        normalized_sensor_angles.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return normalized_sensor_angles;
}

__global__ void relative_sensor_angles_kernel(
    float sensor_start_angle, SpinningDirection sensor_direction,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> const sensor_angles,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> relative_sensor_angles) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= sensor_angles.size(0)) // Point does not exist
        return;

    relative_sensor_angles[point_idx] = RowOffsetStructuredSpinningLidarModel::relative_angle(sensor_start_angle, sensor_angles[point_idx], sensor_direction);
}

torch::Tensor relative_sensor_angles_cu(
    float sensor_start_angle,
    SpinningDirection sensor_direction,
    const torch::Tensor& sensor_angles) {

    auto const sensor_angles_arg = torch::TensorArg{sensor_angles, "sensor_angles", 1};
    torch::checkAllContiguous(__func__, {sensor_angles_arg});

    auto const N_points = sensor_angles.size(0);
    torch::checkSize(__func__, sensor_angles_arg, {N_points});

    auto relative_sensor_angles = torch::zeros({N_points}, sensor_angles.options().dtype(torch::kFloat32));

    auto const threads = dim3(256);
    auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    relative_sensor_angles_kernel<<<blocks, threads, 0, stream>>>(
        sensor_start_angle,
        sensor_direction,
        sensor_angles.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        relative_sensor_angles.packed_accessor32<float, 1, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return relative_sensor_angles;
}
