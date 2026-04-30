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

#include <vren/cameras.cuh>
#include <vren/overload_visitor.h>

#include <c10/cuda/CUDAStream.h>

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

template <class CameraModel>
__global__ void camera_rays_to_image_points_kernel(
    CameraModel camera_model,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const cam_rays,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> image_points,
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid_flag) {

    auto const ray_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ray_idx >= cam_rays.size(0))
        // Ray does not exist
        return;

    auto const cam_ray              = cam_rays[ray_idx];
    auto const [image_point, valid] = camera_model.camera_ray_to_image_point(make_float3(cam_ray[0],
                                                                                         cam_ray[1],
                                                                                         cam_ray[2]));

    auto local_image_point = image_points[ray_idx];
    local_image_point[0]   = image_point.x;
    local_image_point[1]   = image_point.y;

    valid_flag[ray_idx] = valid;
}

std::tuple<torch::Tensor, torch::Tensor> camera_rays_to_image_points_cu(
    CameraModelParametersVariant const camera_model_parameters,
    torch::Tensor const cam_rays) {
    auto const cam_rays_arg = torch::TensorArg{cam_rays, "cam_rays", 1};

    torch::checkAllContiguous(__func__, {cam_rays_arg});

    auto const N_rays = cam_rays.size(0);

    torch::checkSize(__func__, cam_rays_arg, {N_rays, 3});

    auto image_points = torch::zeros({N_rays, 2}, cam_rays.options());
    auto valid_flag   = torch::zeros({N_rays}, cam_rays.options().dtype(torch::kBool));

    auto launchKernel = [&](auto const& camera_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_rays + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

        camera_rays_to_image_points_kernel<<<blocks, threads, 0, stream>>>(
            camera_model,
            cam_rays.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            image_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            valid_flag.packed_accessor32<bool, 1, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    std::visit(
        OverloadVisitor{
            [&](OpenCVPinholeCameraModelParameters const& params) { launchKernel(OpenCVPinholeCameraModel(params)); },
            [&](OpenCVFisheyeCameraModelParameters const& params) { launchKernel(OpenCVFisheyeCameraModel(params)); },
            [&](FThetaCameraModelParameters const& params) { launchKernel(FThetaCameraModel(params)); },
        },
        camera_model_parameters);

    return {image_points, valid_flag};
}

template <size_t N_ROLLING_SHUTTER_ITERATIONS, class CameraModel>
__global__ void world_points_to_image_points_shutter_pose_kernel(
    CameraModel camera_model,
    RollingShutter rolling_shutter,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const world_points,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> image_points,
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> valid_flag,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> timestamps_us,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> T_world_sensors) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= world_points.size(0))
        // Ray does not exist
        return;

    auto& local_valid_flag    = valid_flag[point_idx];
    auto& local_timestamp_us  = timestamps_us[point_idx];
    auto local_image_point    = image_points[point_idx];
    auto local_T_world_sensor = T_world_sensors[point_idx];

    auto const T_world_sensor_start = rolling_shutter.pose_start();
    auto const T_world_sensor_end   = rolling_shutter.pose_end();

    auto const local_world_point = world_points[point_idx];
    auto const world_point       = make_float3(local_world_point[0], local_world_point[1], local_world_point[2]);

    // Always perform transformation using start pose
    auto const [image_point_start, valid_start] = camera_model.camera_ray_to_image_point(
        T_world_sensor_start.transform_point(world_point));

    if (camera_model.parameters.shutter_type == ShutterType::GLOBAL) {
        // Exit early if we have a global shutter sensor
        local_valid_flag   = valid_start;
        local_timestamp_us = rolling_shutter.timestamp_start_us();
        store(local_image_point, image_point_start);
        store(local_T_world_sensor, T_world_sensor_start);
        return;
    }

    // Do initial transformations using both start and end poses to determine all candidate
    // points and take union of valid projections as iteration starting points
    auto const [image_point_end, valid_end] = camera_model.camera_ray_to_image_point(
        T_world_sensor_end.transform_point(world_point));

    // This selection prefers points at the start-of-frame pose over end-of-frame points
    // - the optimization will determine the final timestamp for each point
    auto init_image_point = float2{};
    if (valid_start) {
        init_image_point = image_point_start;
    } else if (valid_end) {
        init_image_point = image_point_end;
    } else {
        // No valid projection at start or finish -> mark point as invalid. Still
        // return projection result at end of frame to be consistent with ncore, as
        // this will be condensed at the python interface level
        local_valid_flag   = false;
        local_timestamp_us = rolling_shutter.timestamp_end_us();
        store(local_image_point, image_point_end);
        store(local_T_world_sensor, T_world_sensor_end);
        return;
    }

    // Compute the new timestamp and project again
    auto image_points_rs_prev = init_image_point;
    auto valid_rs_prev        = true;
    auto relative_frame_time  = float{};
    auto T_world_sensor_rs    = Pose3{};

#pragma unroll
    for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
        relative_frame_time = camera_model.shutter_relative_frame_time(image_points_rs_prev);
        T_world_sensor_rs   = rolling_shutter.interpolate_shutter_pose(relative_frame_time);

        auto const [image_point_rs, valid_rs] = camera_model.camera_ray_to_image_point(
            T_world_sensor_rs.transform_point(world_point));
        image_points_rs_prev = image_point_rs;
        valid_rs_prev        = valid_rs;
    }

    local_valid_flag   = valid_rs_prev;
    local_timestamp_us = rolling_shutter.interpolate_timestamps_us(relative_frame_time);

    store(local_image_point, image_points_rs_prev);
    store(local_T_world_sensor, T_world_sensor_rs);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> world_points_to_image_points_shutter_pose_cu(
    CameraModelParametersVariant const camera_model_parameters,
    RollingShutterParameters const rolling_shutter_parameters,
    torch::Tensor const world_points) {
    auto const world_points_arg = torch::TensorArg{world_points, "world_points", 1};

    torch::checkAllContiguous(__func__, {world_points_arg});

    auto const N_rays = world_points.size(0);

    torch::checkSize(__func__, world_points_arg, {N_rays, 3});

    auto image_points    = torch::zeros({N_rays, 2}, world_points.options());
    auto valid_flag      = torch::zeros({N_rays}, world_points.options().dtype(torch::kBool));
    auto timestamps_us   = torch::zeros({N_rays}, world_points.options().dtype(torch::kInt64));
    auto T_world_sensors = torch::zeros({N_rays, 7}, world_points.options());

    // fixed number of rolling-shutter iterations - same as in NCore
    auto constexpr N_ROLLING_SHUTTER_ITERATIONS = 10;

    auto const rolling_shutter = RollingShutter(rolling_shutter_parameters);

    auto launchKernel = [&](auto const& camera_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_rays + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

        world_points_to_image_points_shutter_pose_kernel<N_ROLLING_SHUTTER_ITERATIONS><<<blocks, threads, 0, stream>>>(
            camera_model, rolling_shutter,
            world_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            image_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            valid_flag.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
            timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            T_world_sensors.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    std::visit(
        OverloadVisitor{
            [&](OpenCVPinholeCameraModelParameters const& params) { launchKernel(OpenCVPinholeCameraModel(params)); },
            [&](OpenCVFisheyeCameraModelParameters const& params) { launchKernel(OpenCVFisheyeCameraModel(params)); },
            [&](FThetaCameraModelParameters const& params) { launchKernel(FThetaCameraModel(params)); },
        },
        camera_model_parameters);

    return {image_points, valid_flag, timestamps_us, T_world_sensors};
}

// Kernel that reads rolling shutter parameters from tensor global memory
template <class CameraModel>
__global__ void image_points_to_world_points_shutter_pose_kernel(
    CameraModel camera_model,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const T_sensor_worlds, // (2, 7) tquat poses
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const timestamps_us, // (2,) timestamps
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const image_points,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> ray_timestamps_us,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_T_sensor_worlds) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= image_points.size(0))
        return; // Point does not exist

    auto const local_image_point = image_points[point_idx];
    auto const image_point       = make_float2(local_image_point[0], local_image_point[1]);

    // Read poses from tensor (they are already T_sensor_world, no inversion needed)
    auto const T_sensor_world_start_tquat = T_sensor_worlds[0];
    auto const T_sensor_world_end_tquat   = T_sensor_worlds[1];

    Pose3 const T_sensor_world_start{
        make_float3(T_sensor_world_start_tquat[0], T_sensor_world_start_tquat[1], T_sensor_world_start_tquat[2]),
        make_float4(T_sensor_world_start_tquat[3], T_sensor_world_start_tquat[4], T_sensor_world_start_tquat[5], T_sensor_world_start_tquat[6])};
    Pose3 const T_sensor_world_end{
        make_float3(T_sensor_world_end_tquat[0], T_sensor_world_end_tquat[1], T_sensor_world_end_tquat[2]),
        make_float4(T_sensor_world_end_tquat[3], T_sensor_world_end_tquat[4], T_sensor_world_end_tquat[5], T_sensor_world_end_tquat[6])};

    // Read timestamps
    std::array<uint64_t, 2> timestamps_array = {static_cast<uint64_t>(timestamps_us[0]), static_cast<uint64_t>(timestamps_us[1])};

    auto const [ray, ray_timestamp_us, ray_T_sensor_world] =
        camera_model.image_point_to_world_ray_shutter_pose(
            image_point, T_sensor_world_start, T_sensor_world_end, timestamps_array);

    store(rays[point_idx], ray);

    ray_timestamps_us[point_idx] = ray_timestamp_us;

    store(ray_T_sensor_worlds[point_idx], ray_T_sensor_world);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> image_points_to_world_points_shutter_pose_cu(
    CameraModelParametersVariant const camera_model_parameters,
    torch::Tensor const T_sensor_worlds, // (2, 7) tensor with tquat poses
    torch::Tensor const timestamps_us,   // (2,) tensor with int64 timestamps
    torch::Tensor const image_points) {
    auto const image_points_arg    = torch::TensorArg{image_points, "image_points", 1};
    auto const T_sensor_worlds_arg = torch::TensorArg{T_sensor_worlds, "T_sensor_worlds", 2};
    auto const timestamps_us_arg   = torch::TensorArg{timestamps_us, "timestamps_us", 3};

    torch::checkAllContiguous(__func__, {image_points_arg, T_sensor_worlds_arg, timestamps_us_arg});

    auto const N_points = image_points.size(0);

    torch::checkSize(__func__, image_points_arg, {N_points, 2});
    torch::checkSize(__func__, T_sensor_worlds_arg, {2, 7});
    torch::checkSize(__func__, timestamps_us_arg, {2});

    // Allocate output tensors
    auto world_rays                = torch::zeros({N_points, 6}, image_points.options());
    auto world_ray_timestamps_us   = torch::zeros({N_points}, image_points.options().dtype(torch::kInt64));
    auto world_ray_T_sensor_worlds = torch::zeros({N_points, 7}, image_points.options());

    auto launchKernel = [&](auto const& camera_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

        image_points_to_world_points_shutter_pose_kernel<<<blocks, threads, 0, stream>>>(
            camera_model,
            T_sensor_worlds.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            image_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            world_rays.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            world_ray_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            world_ray_T_sensor_worlds.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    std::visit(
        OverloadVisitor{
            [&](OpenCVPinholeCameraModelParameters const& params) { launchKernel(OpenCVPinholeCameraModel(params)); },
            [&](OpenCVFisheyeCameraModelParameters const& params) { launchKernel(OpenCVFisheyeCameraModel(params)); },
            [&](FThetaCameraModelParameters const& params) { launchKernel(FThetaCameraModel(params)); },
        },
        camera_model_parameters);

    return {world_rays, world_ray_timestamps_us, world_ray_T_sensor_worlds};
}

template <class CameraModel>
__global__ void image_points_to_camera_rays_kernel(
    CameraModel camera_model,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> const image_points,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cam_rays) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= image_points.size(0))
        // Point does not exist
        return;

    auto const image_point = image_points[point_idx];
    auto const camera_ray  = camera_model.image_point_to_camera_ray(make_float2(image_point[0],
                                                                                image_point[1]));

    auto local_camera_ray = cam_rays[point_idx];
    local_camera_ray[0]   = camera_ray.x;
    local_camera_ray[1]   = camera_ray.y;
    local_camera_ray[2]   = camera_ray.z;
}

torch::Tensor image_points_to_camera_rays_cu(
    CameraModelParametersVariant const camera_model_parameters,
    torch::Tensor const image_points) {
    auto const image_points_arg = torch::TensorArg{image_points, "image_points", 1};

    torch::checkAllContiguous(__func__, {image_points_arg});

    auto const N_points = image_points.size(0);

    torch::checkSize(__func__, image_points_arg, {N_points, 2});

    auto camera_rays = torch::zeros({N_points, 3}, image_points.options());

    auto launchKernel = [&](auto const& camera_model) {
        auto const threads = dim3(256);
        auto const blocks  = dim3((N_points + threads.x - 1) / threads.x);
        auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

        image_points_to_camera_rays_kernel<<<blocks, threads, 0, stream>>>(
            camera_model,
            image_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            camera_rays.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    std::visit(
        OverloadVisitor{
            [&](OpenCVPinholeCameraModelParameters const& params) { launchKernel(OpenCVPinholeCameraModel(params)); },
            [&](OpenCVFisheyeCameraModelParameters const& params) { launchKernel(OpenCVFisheyeCameraModel(params)); },
            [&](FThetaCameraModelParameters const& params) { launchKernel(FThetaCameraModel(params)); },
        },
        camera_model_parameters);

    return camera_rays;
}

std::array<float, 7 * 2> invert_tquat_poses_cu(std::array<float, 7 * 2> const& poses) {
    // Helper function to perform pose pair inversion at the API-level
    auto const pose_start_inv = Pose3{
        make_float3(poses[0], poses[1], poses[2]),
        make_float4(poses[3], poses[4], poses[5], poses[6])}
                                    .inverse();
    auto const pose_end_inv = Pose3{
        make_float3(poses[7], poses[8], poses[9]),
        make_float4(poses[10], poses[11], poses[12], poses[13])}
                                  .inverse();
    return {
        pose_start_inv.t.x, pose_start_inv.t.y, pose_start_inv.t.z, pose_start_inv.q.x, pose_start_inv.q.y, pose_start_inv.q.z, pose_start_inv.q.w,
        pose_end_inv.t.x, pose_end_inv.t.y, pose_end_inv.t.z, pose_end_inv.q.x, pose_end_inv.q.y, pose_end_inv.q.z, pose_end_inv.q.w};
}
