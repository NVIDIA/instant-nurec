// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <vren/cameras.h>
#include <vren/lidars.h>
#include <vren/rolling_shutter.h>

#include <torch/extension.h>

#include <pybind11/numpy.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

std::vector<std::optional<torch::Tensor>> ray_aabb_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor aabbs_min,
    const torch::Tensor aabbs_max,
    const bool compute_hits_flag,
    const bool compute_hits_t);

std::vector<torch::Tensor> ray_sphere_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits);

// Returns:
// - intersections_cnt, intersections_tracks_idx, intersections_ts if with_intersections_ts is True
// - intersections_cnt, intersections_tracks_idx if with_intersections_ts is False
std::vector<torch::Tensor> ray_cuboidtracks_intersection_cu(
    torch::Tensor const rays_o,               // N_rays x 3 (3d world positions)
    torch::Tensor const rays_d,               // N_rays x 3 (normalized 3d world directions)
    torch::Tensor const rays_timestamps_us,   // N_rays (per ray timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_ray,
    bool const with_intersections_ts);

torch::Tensor point_cuboidtracks_intersection_cu(
    torch::Tensor const points,               // N_points x 3 (3d world positions)
    torch::Tensor const timestamps_us,        // N_points or 1 (per point or global timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    bool const return_dense_mask); // If true (N_points x N_tracks) mask is returned else (N_points x 1) and true if point is inside any cuboidtrack

std::vector<torch::Tensor> point_cuboidtracks_intersection_interpolate_pose_cu(
    torch::Tensor const points,               // N_points x 3 (3d world positions)
    torch::Tensor const points_timestamps_us, // N_points (per point timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses);

// Returns:
// - intersections_cnt, intersections_tracks_idx, intersections_ts if with_intersections_ts is True
// - intersections_cnt, intersections_tracks_idx if with_intersections_ts is False
std::vector<torch::Tensor> ray_cuboidtracks_rolling_shutter_intersection_cu(
    torch::Tensor const pixel_idxs,   // N_rays x 2 (pixel indices of rays, i in [0, width-1] / j in [0, height-1])
    torch::Tensor const camera_rays,  // N_rays x 3 (camera-space rays)
    torch::Tensor const camera_poses, // 2 x 7  [[x, y, z], [quat_x, quat_y, quat_z, quat_w]] for start/end pose
    int64_t const camera_timestamp_start_us,
    int64_t const camera_timestamp_end_us,
    int32_t const w,                          // image resolution (width)
    int32_t const h,                          // image resolution (height)
    int32_t const shutter_type,               // ROLLING_TOP_TO_BOTTOM = 1, ROLLING_LEFT_TO_RIGHT = 2, ROLLING_BOTTOM_TO_TOP = 3,
                                              // ROLLING_RIGHT_TO_LEFT = 4, GLOBAL = 5
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_ray,
    bool const with_intersections_ts); // If true, sort intersections by depth (near to far); if False, keep unsorted (faster)

std::vector<torch::Tensor> ray_cuboidtracks_intersection_transform_filter_cu(
    torch::Tensor const rays_o,               // N_rays x 3 (3d world positions)
    torch::Tensor const rays_d,               // N_rays x 3 (normalized 3d world directions)
    torch::Tensor const rays_timestamps_us,   // N_rays (per ray timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses);

std::vector<torch::Tensor> ray_cuboidtracks_intersection_transform_filter_backward_cu(
    torch::Tensor const dL_drays_cuboid_o,     // N_intersections x 3 (gradient of 3d local positions)
    torch::Tensor const dL_drays_cuboid_d,     // N_intersections x 3 (gradient of 3d local directions)
    torch::Tensor const rays_cuboid_o,         // N_intersections x 3 (3d local positions)
    torch::Tensor const rays_cuboid_d,         // N_intersections x 3 (normalized 3d local directions)
    torch::Tensor const rays_timestamps_us,    // N_intersections (per ray timestamp)
    torch::Tensor const rays_pose_idx,         // N_intersections (pose index of the ray-cuboid intersection)
    torch::Tensor const intersection_idx,      // N_intersections x 2
    torch::Tensor const tracks_poses,          // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us); // (N_total_poses) containing per-pose timestamps

torch::Tensor ray_samples_in_distranges_masks_cu(
    torch::Tensor const rays_samples_packinfo,    // N_rays x 2 (per ray sample packinfo [sample_start_idx, N_samples_of_ray])
    torch::Tensor const rays_samples_t,           // N_total_samples (distances of individual ray samples)
    torch::Tensor const rays_distranges_packinfo, // N_rays x 2 (per ray distranges packinfo [distrange_start_idx, N_distranges_of_ray])
    torch::Tensor const rays_distranges_ts);      // N_total_distranges x 2 (distranges of individual rays)

std::vector<torch::Tensor> alpha_composite_train_fw_cu(
    const torch::Tensor alphas,
    const torch::Tensor rgbs,
    const torch::Tensor ts,
    const torch::Tensor pack_info,
    const float transmittance_threshold);

std::vector<torch::Tensor> alpha_composite_train_bw_cu(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddistance,
    const torch::Tensor dL_drgb,
    const torch::Tensor dL_dws,
    const torch::Tensor alphas,
    const torch::Tensor rgbs,
    const torch::Tensor ws,
    const torch::Tensor ts,
    const torch::Tensor pack_info,
    const torch::Tensor opacity,
    const torch::Tensor distance,
    const torch::Tensor rgb,
    const float transmittance_threshold);

void alpha_composite_test_fw_cu(
    const torch::Tensor alphas,
    const torch::Tensor rgbs,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const float transmittance_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor distance,
    torch::Tensor rgb);

std::vector<torch::Tensor> weights_from_alphas_fw_cu(
    const torch::Tensor alphas,
    const torch::Tensor pack_info,
    const float transmittance_threshold);

torch::Tensor ray_samples_visibility_masks_cu(
    const torch::Tensor alphas,
    const torch::Tensor pack_info,
    const float transmittance_threshold,
    const float alpha_threshold);

torch::Tensor weights_from_alphas_bw_cu(
    const torch::Tensor dL_dws,
    const torch::Tensor dL_dopacity,
    const torch::Tensor alphas,
    const torch::Tensor ws,
    const torch::Tensor opacity,
    const torch::Tensor pack_info,
    const float transmittance_threshold);

std::vector<torch::Tensor> cuboidtracks_frame_poses_interpolation_cu(
    torch::Tensor const frame_timestamps_us,   // frame start and end timestamp : 2 [int64]
    torch::Tensor const tracks_packinfo,       // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,          // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us); // (N_total_poses) containing per-pose timestamps

std::array<float, 7 * 2> invert_tquat_poses_cu(std::array<float, 7 * 2> const& poses);

// Camera-specific types

std::tuple<torch::Tensor, torch::Tensor>
camera_rays_to_image_points_cu(
    const CameraModelParametersVariant camera_model_parameters,
    const torch::Tensor cam_rays);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
world_points_to_image_points_shutter_pose_cu(
    const CameraModelParametersVariant camera_model_parameters,
    const RollingShutterParameters rolling_shutter_parameters,
    const torch::Tensor world_points);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
image_points_to_world_points_shutter_pose_cu(
    const CameraModelParametersVariant camera_model_parameters,
    const torch::Tensor T_sensor_worlds, // (2, 7) tensor with tquat poses
    const torch::Tensor timestamps_us,   // (2,) tensor with int64 timestamps
    const torch::Tensor image_points);

torch::Tensor
image_points_to_camera_rays_cu(
    const CameraModelParametersVariant camera_model_parameters,
    const torch::Tensor image_points);

// LiDAR specific types

torch::Tensor elements_to_sensor_rays_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& elements);

torch::Tensor elements_to_sensor_angles_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& elements);

torch::Tensor elements_to_sensor_points_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& elements, const torch::Tensor& distances);

std::tuple<torch::Tensor, torch::Tensor>
sensor_rays_to_sensor_angles_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& sensor_rays);

std::tuple<torch::Tensor, torch::Tensor>
sensor_angles_to_sensor_rays_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters, const torch::Tensor& sensor_angles);

torch::Tensor
sensor_angles_relative_frame_times_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor& sensor_angles);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
world_points_to_sensor_angles_shutter_pose_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    RollingShutterParameters const rolling_shutter_parameters,
    const torch::Tensor& world_points);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
elements_to_world_rays_shutter_pose_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor T_sensor_worlds, // (2, 7) tensor with tquat poses
    const torch::Tensor timestamps_us,   // (2,) tensor with int64 timestamps
    const torch::Tensor& elements);

torch::Tensor sensor_angles_to_tile_indices_cu(
    RowOffsetStructuredSpinningLidarModelParameters const parameters,
    const torch::Tensor& sensor_angles);

torch::Tensor normalize_sensor_angles_cu(
    const torch::Tensor& sensor_angles);

torch::Tensor relative_sensor_angles_cu(
    float sensor_start_angle,
    SpinningDirection sensor_direction,
    const torch::Tensor& sensor_angles);

torch::Tensor lidar_seg_ensemble_cu(
    torch::Tensor points,
    torch::Tensor result,
    unsigned char ignore_label);
