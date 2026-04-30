// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "intersection.cuh"

#include "utils.h"

#include <ku/helper_math.cuh>

#include <ku/binary_search.cuh>
#include <ku/common.cuh>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <tuple>

template <typename BBoxTimed>
inline __device__ auto worldray_bboxtimed_intersection(float3 const& ray_o_world, float3 const& ray_d_world,
                                                       float ray_t, BBoxTimed const& bboxtimed) {
    // returns tuple of
    // - float2 t1t2 (intersection times)
    // - float3 ray_o_bbox (ray origin in bbox space)
    // - float3 ray_d_bbox (ray direction in bbox space)

    // grab timed bbox with start/end bbox->world pose
    auto const dim     = make_float3(bboxtimed[0], bboxtimed[1], bboxtimed[2]);
    auto const c_start = make_float3(bboxtimed[3], bboxtimed[4], bboxtimed[5]);
    auto const q_start = make_float4(bboxtimed[6], bboxtimed[7], bboxtimed[8], bboxtimed[9]);
    auto const c_end   = make_float3(bboxtimed[10], bboxtimed[11], bboxtimed[12]);
    auto const q_end   = make_float4(bboxtimed[13], bboxtimed[14], bboxtimed[15], bboxtimed[16]);

    // interpolate bbox to ray time and create world->bbox transformation
    auto const c_raytime = (1.f - ray_t) * c_start + ray_t * c_end;
    auto const R_raytime = transpose_matrix(unitquat_rotmatrix(unitquat_slerp(q_start, q_end, ray_t)));

    // transform world-ray to bbox-ray
    auto const ray_o_bbox = apply_matrix(R_raytime, ray_o_world - c_raytime);
    auto const ray_d_bbox = apply_matrix(R_raytime, ray_d_world);

    // perform intersection with aabb at origin
    auto const aabb_min = make_float3(-dim.x / 2.f, -dim.y / 2.f, -dim.z / 2.f);
    auto const aabb_max = make_float3(dim.x / 2.f, dim.y / 2.f, dim.z / 2.f);
    auto const t1t2     = ray_aabb_intersect(ray_o_bbox, 1.f / ray_d_bbox, aabb_min, aabb_max);

    return std::make_tuple(t1t2, ray_o_bbox, ray_d_bbox);
}

template <typename PoseTimed>
inline __device__ auto worldray_bboxtimed_intersection_backward(float3 const& d_ray_o_bbox, float3 const& d_ray_d_bbox,
                                                                float3 const& ray_o_bbox, float3 const& ray_d_bbox,
                                                                float ray_t, PoseTimed const& posetimed) {
    auto const c_start = make_float3(posetimed[0], posetimed[1], posetimed[2]);
    auto const q_start = make_float4(posetimed[3], posetimed[4], posetimed[5], posetimed[6]);
    auto const c_end   = make_float3(posetimed[7], posetimed[8], posetimed[9]);
    auto const q_end   = make_float4(posetimed[10], posetimed[11], posetimed[12], posetimed[13]);

    // interpolate bbox to ray time and create world->bbox transformation
    auto const c_raytime = (1.f - ray_t) * c_start + ray_t * c_end;
    auto const R_raytime = unitquat_rotmatrix(unitquat_slerp(q_start, q_end, ray_t));

    // compute ray-origin and ray-direction gradient
    auto const d_ray_o_world = apply_matrix(R_raytime, d_ray_o_bbox);
    auto const d_ray_d_world = apply_matrix(R_raytime, d_ray_d_bbox);

    // compute pose gradient
    auto const d_R_raytime = apply_matrix(
        R_raytime, cross(d_ray_d_bbox, ray_d_bbox) + cross(d_ray_o_bbox, ray_o_bbox));

    auto const phi       = log_q0invq1(q_start, q_end);
    auto const phi_angle = length(phi);

    Mat3 alphaJJ;
    if (abs(phi_angle) < 1.0e-6) {
        auto const J_left    = identity() + 0.5f * skew_symmetric(phi * ray_t);
        auto const Jinv_left = identity() - 0.5f * skew_symmetric(phi);
        alphaJJ              = ray_t * (J_left * Jinv_left);
    } else {
        auto const sin_phi_angle = sinf(phi_angle * ray_t);
        auto const cos_phi_angle = cosf(phi_angle * ray_t);
        auto const phi_axis      = phi / phi_angle;

        auto const alpha_J_left = (sin_phi_angle / phi_angle) * identity() +
                                  (ray_t - sin_phi_angle / phi_angle) * outer_product(phi_axis) +
                                  ((1 - cos_phi_angle) / phi_angle) * skew_symmetric(phi_axis);

        auto const phi_hangle     = phi_angle / 2;
        auto const cot_phi_hangle = 1 / tanf(phi_hangle);

        auto const Jinv_left = phi_hangle * cot_phi_hangle * identity() +
                               (1 - phi_hangle * cot_phi_hangle) * outer_product(phi_axis) -
                               phi_hangle * skew_symmetric(phi_axis);

        alphaJJ = alpha_J_left * Jinv_left;
    }

    auto const R_start = unitquat_rotmatrix(q_start);

    // TODO: Accelerate this.
    auto const Q         = R_start * alphaJJ * transpose_matrix(R_start);
    auto const d_R_end   = apply_matrix(transpose_matrix(Q), d_R_raytime);
    auto const d_R_start = d_R_raytime - d_R_end;

    auto const d_t_start = -d_ray_o_world * (1 - ray_t);
    auto const d_t_end   = -d_ray_o_world * ray_t;
    auto const d_T_start = d_R_start + cross(c_start, d_t_start);
    auto const d_T_end   = d_R_end + cross(c_end, d_t_end);

    return std::make_tuple(d_ray_d_world, d_ray_o_world, d_t_start, d_T_start, d_t_end, d_T_end);
}

inline __device__ float rolling_shutter_time(int16_t const i, int16_t const j,
                                             int32_t const w, int32_t const h,
                                             int32_t const shutter_type) {
    // computes relative frame time [0,1] given pixel index, image resolution, and rolling-shutter type

    switch (shutter_type) {
    case 1: // ROLLING_TOP_TO_BOTTOM
        return float(j) / (h - 1);
    case 2: // ROLLING_LEFT_TO_RIGHT
        return float(i) / (w - 1);
    case 3: // ROLLING_BOTTOM_TO_TOP
        return (h - 1 - float(j)) / (h - 1);
    case 4: // ROLLING_RIGHT_TO_LEFT
        return (w - 1 - float(i)) / (w - 1);
    }

    return 0.f; // GLOBAL / fallback - stick to time of first pose
}

inline __device__ auto rolling_shutter_camera_ray_to_world(
    int16_t const pixel_i,
    int16_t const pixel_j,
    float3 const camera_ray, // 3d ray direction in camera frame
    float3 const t_start,    // camera-to-world at start of frame [t, q]
    float4 const q_start,
    float3 const t_end, // camera-to-world at end of frame [t, q]
    float4 const q_end,
    int32_t const w,
    int32_t const h,
    int32_t const shutter_type) {
    // determine relative frame time from shutter type and pixel location
    auto const relative_frame_time = rolling_shutter_time(pixel_i, pixel_j, w, h, shutter_type);

    // rolling-shutter sensor->world pose for the ray
    auto const t_raytime = (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
    auto const R_raytime = unitquat_rotmatrix(unitquat_slerp(q_start, q_end, relative_frame_time));

    float3 const ray_o_world = t_raytime;
    float3 const ray_d_world = apply_matrix(R_raytime, camera_ray);

    return std::make_tuple(ray_o_world, ray_d_world, relative_frame_time);
}

// Kernel / entry points for CuboidTracks APIs
template <typename scalar_t>
__global__ void ray_cuboidtracks_intersection_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_o,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_d,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const rays_timestamps_us,

    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,

    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,

    int32_t const max_intersections_per_ray,

    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> intersections_cnt,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> intersections_ts,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> intersections_tracks_idx,
    bool const with_intersections_ts) {

    auto const ray_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0))
        // track of block does not exist
        return;

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0];
    auto const N_track_poses   = track_packinfo[1];
    auto const N_total_poses   = tracks_poses.size(0);

    if (N_track_poses <= 1)
        // track has no poses
        return;

    // load the track's timestamps into shared memory to accelerate binary search
    // using all threads (irrespective of whether the local ray exists)
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads(); // all timestamps of the block-associated track are now loaded to shared memory

    if (ray_idx >= rays_o.size(0))
        // local ray doesn't exist
        return;

    // check if ray's timestamp is in the track's valid time-range and exit early if not
    auto const ray_timestamp_us = rays_timestamps_us[ray_idx];

    if (ray_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < ray_timestamp_us)
        return;

    // ray's time is in the tracks' valid time-range AND we have N_track_poses > 1 -> determine track poses for linear interpolation
    auto const interpolation_end_idx   = binary_search_interp(ray_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1; // note: this is guaranteed to be in the valid range of poses due to the check above

    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx],
               local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    // linear interpolation parameter [0,1]
    auto const t_interp = scalar_t(ray_timestamp_us - local_interpolation_start_timestamp_us) / scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    auto const local_interpolation_start_pose = tracks_poses[track_start_idx + interpolation_start_idx],
               local_interpolation_end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // collect timed-cuboid data for intersection computation
    auto const cuboidtimed = std::array<scalar_t, 17>{cuboids_dims[track_idx][0],
                                                      cuboids_dims[track_idx][1],
                                                      cuboids_dims[track_idx][2],

                                                      local_interpolation_start_pose[0],
                                                      local_interpolation_start_pose[1],
                                                      local_interpolation_start_pose[2],
                                                      local_interpolation_start_pose[3],
                                                      local_interpolation_start_pose[4],
                                                      local_interpolation_start_pose[5],
                                                      local_interpolation_start_pose[6],

                                                      local_interpolation_end_pose[0],
                                                      local_interpolation_end_pose[1],
                                                      local_interpolation_end_pose[2],
                                                      local_interpolation_end_pose[3],
                                                      local_interpolation_end_pose[4],
                                                      local_interpolation_end_pose[5],
                                                      local_interpolation_end_pose[6]};

    // grab timed ray
    auto const ray_o_world = make_float3(rays_o[ray_idx][0], rays_o[ray_idx][1], rays_o[ray_idx][2]);
    auto const ray_d_world = make_float3(rays_d[ray_idx][0], rays_d[ray_idx][1], rays_d[ray_idx][2]);

    // perform time-based cuboid intersection
    auto const t1t2 = std::get<0>(worldray_bboxtimed_intersection(ray_o_world, ray_d_world, t_interp, cuboidtimed));

    if (t1t2.y > 0.f) { // if ray intersects the cuboid
        auto const cnt = atomicAdd(&intersections_cnt[ray_idx], 1);
        if (cnt < max_intersections_per_ray) {
            if (with_intersections_ts) {
                intersections_ts[ray_idx][cnt][0] = fmaxf(t1t2.x, 0.f);
                intersections_ts[ray_idx][cnt][1] = t1t2.y;
            }
            intersections_tracks_idx[ray_idx][cnt] = track_idx;
        }
    }
}

std::vector<torch::Tensor> ray_cuboidtracks_intersection_cu(
    torch::Tensor const rays_o,             // N_rays x 3 (3d world positions)
    torch::Tensor const rays_d,             // N_rays x 3 (normalized 3d world directions)
    torch::Tensor const rays_timestamps_us, // N_rays (per ray timestamp)

    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps

    torch::Tensor const cuboids_dims, // (N_tracks x 3) cuboid x/y/z extents (in local track frame)

    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_ray,
    bool const with_intersections_ts) {

    auto rays_o_arg             = torch::TensorArg{rays_o, "rays_o", 1};
    auto rays_d_arg             = torch::TensorArg{rays_d, "rays_d", 2};
    auto rays_timestamps_us_arg = torch::TensorArg{rays_timestamps_us, "rays_timestamps_us", 3};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 4};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 5};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 6};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 7};

    torch::checkAllSameType(__func__, {rays_o_arg, rays_d_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, rays_timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {rays_o_arg, rays_d_arg, rays_timestamps_us_arg,
                                      tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                      cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {rays_o_arg, rays_d_arg, rays_timestamps_us_arg,
                                         tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                         cuboids_dims_arg});

    auto const N_rays = rays_o.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, rays_o_arg, {N_rays, 3});
    torch::checkSize(__func__, rays_d_arg, {N_rays, 3});
    torch::checkSize(__func__, rays_timestamps_us_arg, {N_rays});

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    auto intersections_cnt = torch::zeros({N_rays}, torch::dtype(torch::kInt32).device(rays_o.device()));
    // If not needed put some arbitrary tensor instead
    auto intersections_ts         = with_intersections_ts ? torch::full({N_rays, max_intersections_per_ray, 2}, -1., rays_o.options()) : torch::empty({0, 0, 0}, rays_o.options());
    auto intersections_tracks_idx = torch::full({N_rays, max_intersections_per_ray}, -1, torch::dtype(torch::kInt32).device(rays_o.device()));

    if (N_rays == 0 || max_track_n_poses == 0) {
        if (with_intersections_ts) {
            return {intersections_cnt, intersections_tracks_idx, intersections_ts};
        } else {
            return {intersections_cnt, intersections_tracks_idx};
        }
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_rays + threads - 1) / threads, N_tracks);

    // Ensure entire function runs on the same stream and device as data
    c10::cuda::CUDAGuard device_guard(rays_o.device());
    auto const current_stream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::CUDAStreamGuard stream_guard(current_stream);
    auto const stream = current_stream.stream();

    auto const smem = max_track_n_poses * sizeof(int64_t); // allocate enought shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(rays_o.scalar_type(), "ray_cuboidtracks_intersection_cu", ([&] {
                                   ray_cuboidtracks_intersection_kernel<<<blocks, threads, smem, stream>>>(
                                       rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),

                                       max_intersections_per_ray,

                                       intersections_cnt.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                       intersections_ts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       intersections_tracks_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(), with_intersections_ts);
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (with_intersections_ts) {
        return {intersections_cnt, intersections_tracks_idx, intersections_ts};
    } else {
        return {intersections_cnt, intersections_tracks_idx};
    }
}

template <typename scalar_t>
__global__ void point_cuboidtracks_intersection_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const points,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const timestamps_us,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> point_cuboidtracks_intersection_mask,
    bool const return_dense_mask) {

    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0)) {
        return;
    }

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0];
    auto const N_track_poses   = track_packinfo[1];
    auto const N_total_poses   = tracks_poses.size(0);

    if (N_track_poses <= 1) {
        // track has no poses
        return;
    }

    // Load track timestamps into shared memory
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads();

    // Once the loading is done, exit if point index is out of bounds
    if (point_idx >= points.size(0))
        return;

    // Get point timestamp (handle both per-point and global timestamp cases)
    auto const point_timestamp_us = timestamps_us.size(0) == 1 ? timestamps_us[0] : timestamps_us[point_idx];
    auto const point_world        = make_float3(points[point_idx][0], points[point_idx][1], points[point_idx][2]);

    // Check if point's timestamp is in track's time range and exit early if not
    if (point_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < point_timestamp_us)
        return;

    // Find interpolation poses
    auto const interpolation_end_idx   = binary_search_interp(point_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1;

    // Calculate interpolation parameter
    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx];
    auto const local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    auto const t_interp = scalar_t(point_timestamp_us - local_interpolation_start_timestamp_us) /
                          scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    // Get interpolated poses in tquat
    auto const start_pose = tracks_poses[track_start_idx + interpolation_start_idx];
    auto const end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // Interpolate pose
    auto const c_start = make_float3(start_pose[0], start_pose[1], start_pose[2]);
    auto const c_end   = make_float3(end_pose[0], end_pose[1], end_pose[2]);
    auto const q_start = make_float4(start_pose[3], start_pose[4], start_pose[5], start_pose[6]);
    auto const q_end   = make_float4(end_pose[3], end_pose[4], end_pose[5], end_pose[6]);

    auto const c_interp = (1.f - t_interp) * c_start + t_interp * c_end;
    auto const R_interp = transpose_matrix(unitquat_rotmatrix(unitquat_slerp(q_start, q_end, t_interp)));

    // Transform point to local space
    auto const point_local = apply_matrix(R_interp, point_world - c_interp);

    // Check if point is inside cuboid bounds
    auto const dims = make_float3(cuboids_dims[track_idx][0],
                                  cuboids_dims[track_idx][1],
                                  cuboids_dims[track_idx][2]);

    auto const half_dims = dims * 0.5f;

    bool const is_inside = point_local.x >= -half_dims.x && point_local.x <= half_dims.x &&
                           point_local.y >= -half_dims.y && point_local.y <= half_dims.y &&
                           point_local.z >= -half_dims.z && point_local.z <= half_dims.z;

    if (is_inside) {
        if (return_dense_mask) {
            point_cuboidtracks_intersection_mask[point_idx][track_idx] = true;
        } else {
            point_cuboidtracks_intersection_mask[point_idx][0] = true;
        }
    }
}

torch::Tensor point_cuboidtracks_intersection_cu(
    torch::Tensor const points,               // N_points x 3 (3d world positions)
    torch::Tensor const timestamps_us,        // N_points or 1 (per point or global timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses,
    bool const return_dense_mask = true) { // If true (N_points x N_tracks) mask is returned else (N_points x 1) and true if point is inside any cuboidtrack

    auto points_arg        = torch::TensorArg{points, "points", 1};
    auto timestamps_us_arg = torch::TensorArg{timestamps_us, "timestamps_us", 2};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 3};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 4};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 5};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 6};

    torch::checkAllSameType(__func__, {points_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {points_arg, timestamps_us_arg, tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg, cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {points_arg, timestamps_us_arg, tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg, cuboids_dims_arg});

    auto const N_points = points.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, points_arg, {N_points, 3});
    TORCH_CHECK(timestamps_us.size(0) == N_points || timestamps_us.size(0) == 1,
                "timestamps_us must have size N_points or 1, but got ", timestamps_us.size(0));

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});
    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    // Create output tensor with appropriate dimensions based on return_dense_mask
    auto point_cuboidtracks_intersection_mask = torch::zeros(
        {N_points, return_dense_mask ? N_tracks : 1},
        torch::dtype(torch::kBool).device(points.device()));

    if (N_points == 0 || max_track_n_poses == 0) {
        return point_cuboidtracks_intersection_mask;
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_points + threads - 1) / threads, N_tracks);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    auto const smem    = max_track_n_poses * sizeof(int64_t); // allocate enought shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "point_cuboidtracks_intersection_cu", ([&] {
                                   point_cuboidtracks_intersection_kernel<<<blocks, threads, smem, stream>>>(
                                       points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       point_cuboidtracks_intersection_mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
                                       return_dense_mask);
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return point_cuboidtracks_intersection_mask;
}

template <typename scalar_t>
__global__ void point_cuboidtracks_intersection_interpolate_pose_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const points,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const points_timestamps_us,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> interpolated_tracks_pose,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> interpolated_tracks_idx) {
    auto const point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0))
        // track of block does not exist
        return;

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0], N_track_poses = track_packinfo[1];

    if (N_track_poses <= 1)
        // track has no poses
        return;

    // load the track's timestamps into shared memory to accelerate binary search
    // using all threads (irrespective of whether the local ray exists)
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads(); // all timestamps of the block-associated track are now loaded to shared memory

    if (point_idx >= points.size(0))
        // local ray doesn't exist
        return;

    // check if point's timestamp is in the track's valid time-range and exit early if not
    auto const point_timestamp_us = points_timestamps_us[point_idx];

    if (point_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < point_timestamp_us)
        return;

    // point's time is in the tracks' valid time-range -> determine track poses for linear interpolation
    auto const interpolation_end_idx   = binary_search_interp(point_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1; // note: this is guaranteed to be in the valid range of poses due to the check above

    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx],
               local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    // linear interpolation parameter [0,1]
    auto const t_interp = scalar_t(point_timestamp_us - local_interpolation_start_timestamp_us) / scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    auto const local_interpolation_start_pose = tracks_poses[track_start_idx + interpolation_start_idx],
               local_interpolation_end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // grab timed bbox with start/end bbox->world pose
    auto const cuboid_dim = make_float3(cuboids_dims[track_idx][0], cuboids_dims[track_idx][1], cuboids_dims[track_idx][2]);
    auto const c_start    = make_float3(local_interpolation_start_pose[0], local_interpolation_start_pose[1], local_interpolation_start_pose[2]);
    auto const q_start    = make_float4(local_interpolation_start_pose[3], local_interpolation_start_pose[4], local_interpolation_start_pose[5], local_interpolation_start_pose[6]);
    auto const c_end      = make_float3(local_interpolation_end_pose[0], local_interpolation_end_pose[1], local_interpolation_end_pose[2]);
    auto const q_end      = make_float4(local_interpolation_end_pose[3], local_interpolation_end_pose[4], local_interpolation_end_pose[5], local_interpolation_end_pose[6]);

    // interpolate bbox to point time and create world->bbox transformation
    auto const c_pointtime = (1.f - t_interp) * c_start + t_interp * c_end;
    auto const q_pointtime = unitquat_slerp(q_start, q_end, t_interp);
    auto const R_pointtime = transpose_matrix(unitquat_rotmatrix(q_pointtime));

    // transform world-point to bbox-point
    auto const point_world = make_float3(points[point_idx][0], points[point_idx][1], points[point_idx][2]);
    auto const point_bbox  = apply_matrix(R_pointtime, point_world - c_pointtime);

    if (point_bbox > -cuboid_dim / 2.f && point_bbox < cuboid_dim / 2.f) {
        // point is inside the cuboid -> record the track pose and index
        interpolated_tracks_pose[point_idx][0] = c_pointtime.x;
        interpolated_tracks_pose[point_idx][1] = c_pointtime.y;
        interpolated_tracks_pose[point_idx][2] = c_pointtime.z;
        interpolated_tracks_pose[point_idx][3] = q_pointtime.x;
        interpolated_tracks_pose[point_idx][4] = q_pointtime.y;
        interpolated_tracks_pose[point_idx][5] = q_pointtime.z;
        interpolated_tracks_pose[point_idx][6] = q_pointtime.w;
        interpolated_tracks_idx[point_idx]     = track_idx;
    }
}

std::vector<torch::Tensor> point_cuboidtracks_intersection_interpolate_pose_cu(
    torch::Tensor const points,               // N_points x 3 (3d world positions)
    torch::Tensor const points_timestamps_us, // N_points (per point timestamp)
    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps
    torch::Tensor const cuboids_dims,         // (N_tracks x 3) cuboid x/y/z extents (in local track frame)
    int32_t const max_track_n_poses) {

    auto points_arg               = torch::TensorArg{points, "points", 1};
    auto points_timestamps_us_arg = torch::TensorArg{points_timestamps_us, "points_timestamps_us", 2};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 3};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 4};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 5};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 6};

    torch::checkAllSameType(__func__, {points_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, points_timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {points_arg, points_timestamps_us_arg,
                                      tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                      cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {points_arg, points_timestamps_us_arg,
                                         tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                         cuboids_dims_arg});

    auto const N_points = points.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, points_arg, {N_points, 3});
    torch::checkSize(__func__, points_timestamps_us_arg, {N_points});

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    auto interpolated_tracks_pose = torch::zeros({N_points, 7}, points.options());
    auto interpolated_tracks_idx  = torch::full({N_points}, -1, torch::dtype(torch::kInt32).device(points.device()));

    if (N_points == 0 || max_track_n_poses == 0) {
        return {interpolated_tracks_pose, interpolated_tracks_idx};
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_points + threads - 1) / threads, N_tracks);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    auto const smem    = max_track_n_poses * sizeof(int64_t); // allocate enought shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "point_cuboidtracks_intersection_interpolate_pose_cu", ([&] {
                                   point_cuboidtracks_intersection_interpolate_pose_kernel<<<blocks, threads, smem, stream>>>(
                                       points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       points_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       interpolated_tracks_pose.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       interpolated_tracks_idx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {interpolated_tracks_pose, interpolated_tracks_idx};
}

template <typename scalar_t>
__global__ void ray_cuboidtracks_rolling_shutter_intersection_kernel(
    torch::PackedTensorAccessor32<int16_t, 2, torch::RestrictPtrTraits> const pixel_idxs, // i in [0, width-1], j in [0, height-1]
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const camera_rays,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const camera_poses,
    int64_t const camera_timestamp_start_us,
    int64_t const camera_timestamp_end_us,
    int32_t const w,            // image resolution (width)
    int32_t const h,            // image resolution (height)
    int32_t const shutter_type, // ROLLING_TOP_TO_BOTTOM = 1, ROLLING_LEFT_TO_RIGHT = 2, ROLLING_BOTTOM_TO_TOP = 3,
                                // ROLLING_RIGHT_TO_LEFT = 4, GLOBAL = 5

    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,

    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,

    int32_t const max_intersections_per_ray,

    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> intersections_cnt,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> intersections_ts,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> intersections_tracks_idx,
    bool const with_intersections_ts) {
    auto const ray_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0))
        // track of block does not exist
        return;

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0];
    auto const N_track_poses   = track_packinfo[1];
    auto const N_total_poses   = tracks_poses.size(0);

    if (N_track_poses <= 1)
        // track has no poses
        return;

    // load the track's timestamps into shared memory to accelerate binary search
    // using all threads (irrespective of whether the local ray exists)
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads(); // all timestamps of the block-associated track are now loaded to shared memory

    if (ray_idx >= pixel_idxs.size(0))
        // local ray doesn't exist
        return;

    // grab and transform camera-ray to world-ray via rolling-shutter time
    auto const [ray_o_world, ray_d_world, relative_frame_time] = rolling_shutter_camera_ray_to_world(
        pixel_idxs[ray_idx][0],
        pixel_idxs[ray_idx][1],
        make_float3(camera_rays[ray_idx][0], camera_rays[ray_idx][1], camera_rays[ray_idx][2]),
        make_float3(camera_poses[0][0], camera_poses[0][1], camera_poses[0][2]),
        make_float4(camera_poses[0][3], camera_poses[0][4], camera_poses[0][5], camera_poses[0][6]),
        make_float3(camera_poses[1][0], camera_poses[1][1], camera_poses[1][2]),
        make_float4(camera_poses[1][3], camera_poses[1][4], camera_poses[1][5], camera_poses[1][6]),
        w, h, shutter_type);

    // map frame-relative time to absolute time
    auto const ray_timestamp_us = camera_timestamp_start_us + int64_t(relative_frame_time * (camera_timestamp_end_us - camera_timestamp_start_us));

    // check if ray's timestamp is in the track's valid time-range and exit early if not
    if (ray_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < ray_timestamp_us)
        return;

    // ray's time is in the tracks' valid time-range AND we have N_track_poses > 1 -> determine track poses for linear interpolation
    auto const interpolation_end_idx   = binary_search_interp(ray_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1; // note: this is guaranteed to be in the valid range of poses due to the check above

    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx],
               local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    // linear interpolation parameter [0,1]
    auto const t_interp = scalar_t(ray_timestamp_us - local_interpolation_start_timestamp_us) / scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    // load poses to interpolate from global memory
    auto const local_interpolation_start_pose = tracks_poses[track_start_idx + interpolation_start_idx],
               local_interpolation_end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // collect timed-cuboid data for intersection computation
    auto const cuboidtimed = std::array<scalar_t, 17>{cuboids_dims[track_idx][0],
                                                      cuboids_dims[track_idx][1],
                                                      cuboids_dims[track_idx][2],

                                                      local_interpolation_start_pose[0],
                                                      local_interpolation_start_pose[1],
                                                      local_interpolation_start_pose[2],
                                                      local_interpolation_start_pose[3],
                                                      local_interpolation_start_pose[4],
                                                      local_interpolation_start_pose[5],
                                                      local_interpolation_start_pose[6],

                                                      local_interpolation_end_pose[0],
                                                      local_interpolation_end_pose[1],
                                                      local_interpolation_end_pose[2],
                                                      local_interpolation_end_pose[3],
                                                      local_interpolation_end_pose[4],
                                                      local_interpolation_end_pose[5],
                                                      local_interpolation_end_pose[6]};

    // perform time-based cuboid intersection
    auto const t1t2 = std::get<0>(worldray_bboxtimed_intersection(ray_o_world, ray_d_world, t_interp, cuboidtimed));

    if (t1t2.y > 0.f) { // if ray intersects the cuboid
        auto const cnt = atomicAdd(&intersections_cnt[ray_idx], 1);
        if (cnt < max_intersections_per_ray) {
            if (with_intersections_ts) {
                intersections_ts[ray_idx][cnt][0] = fmaxf(t1t2.x, 0.f);
                intersections_ts[ray_idx][cnt][1] = t1t2.y;
            }
            intersections_tracks_idx[ray_idx][cnt] = track_idx;
        }
    }
}

std::vector<torch::Tensor> ray_cuboidtracks_rolling_shutter_intersection_cu(
    torch::Tensor const pixel_idxs,   // N_rays x 2 (pixel indices of rays, i in [0, width-1] / j in [0, height-1])
    torch::Tensor const camera_rays,  // N_rays x 3 (camera-space rays)
    torch::Tensor const camera_poses, // 2 x 7  [[x, y, z], [quat_x, quat_y, quat_z, quat_w]] for start/end pose
    int64_t const camera_timestamp_start_us,
    int64_t const camera_timestamp_end_us,
    int32_t const w,            // image resolution (width)
    int32_t const h,            // image resolution (height)
    int32_t const shutter_type, // ROLLING_TOP_TO_BOTTOM = 1, ROLLING_LEFT_TO_RIGHT = 2, ROLLING_BOTTOM_TO_TOP = 3,
                                // ROLLING_RIGHT_TO_LEFT = 4, GLOBAL = 5

    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps

    torch::Tensor const cuboids_dims, // (N_tracks x 3) cuboid x/y/z extents (in local track frame)

    int32_t const max_track_n_poses,
    int32_t const max_intersections_per_ray,
    bool const with_intersections_ts) {

    auto pixel_idxs_arg   = torch::TensorArg{pixel_idxs, "pixel_idxs", 1};
    auto camera_rays_arg  = torch::TensorArg{camera_rays, "camera_rays", 2};
    auto camera_poses_arg = torch::TensorArg{camera_poses, "camera_poses", 3};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 4};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 5};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 6};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 7};

    torch::checkAllSameType(__func__, {camera_rays_arg, camera_poses_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, pixel_idxs_arg, torch::kInt16);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {pixel_idxs_arg, camera_rays_arg, camera_poses_arg,
                                      tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                      cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {pixel_idxs_arg, camera_rays_arg, camera_poses_arg,
                                         tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                         cuboids_dims_arg});

    auto const N_rays = camera_rays.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, pixel_idxs_arg, {N_rays, 2});
    torch::checkSize(__func__, camera_rays_arg, {N_rays, 3});
    torch::checkSize(__func__, camera_poses_arg, {2, 7});

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    auto intersections_cnt = torch::zeros({N_rays}, torch::dtype(torch::kInt32).device(pixel_idxs.device()));
    // If not needed put some arbitrary tensor instead
    auto intersections_ts         = with_intersections_ts ? torch::full({N_rays, max_intersections_per_ray, 2}, -1., camera_rays.options()) : torch::empty({0, 0, 0}, camera_rays.options());
    auto intersections_tracks_idx = torch::full({N_rays, max_intersections_per_ray}, -1, torch::dtype(torch::kInt32).device(pixel_idxs.device()));

    if (N_rays == 0 || max_track_n_poses == 0) {
        if (with_intersections_ts) {
            return {intersections_cnt, intersections_tracks_idx, intersections_ts};
        } else {
            return {intersections_cnt, intersections_tracks_idx};
        }
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_rays + threads - 1) / threads, N_tracks);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    auto const smem    = max_track_n_poses * sizeof(int64_t); // allocate enought shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(camera_rays.scalar_type(), "ray_cuboidtracks_rolling_shutter_intersection_cu", ([&] {
                                   ray_cuboidtracks_rolling_shutter_intersection_kernel<<<blocks, threads, smem, stream>>>(
                                       pixel_idxs.packed_accessor32<int16_t, 2, torch::RestrictPtrTraits>(),
                                       camera_rays.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       camera_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       camera_timestamp_start_us, camera_timestamp_end_us, w, h, shutter_type,

                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),

                                       max_intersections_per_ray,

                                       intersections_cnt.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                       intersections_ts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       intersections_tracks_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       with_intersections_ts);
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (with_intersections_ts) {
        return {intersections_cnt, intersections_tracks_idx, intersections_ts};
    } else {
        return {intersections_cnt, intersections_tracks_idx};
    }
}

template <typename scalar_t>
__global__ void ray_cuboidtracks_intersection_transform_filter_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_o,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_d,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const rays_timestamps_us,

    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,

    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const cuboids_dims,

    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> intersections_rays_cuboid_o_full,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> intersections_rays_cuboid_d_full,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> intersections_rays_ts_full,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> intersections_rays_pose_idx_full,
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> intersections_flags_full) {
    auto const ray_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    auto const track_idx = blockIdx.y;

    if (track_idx >= tracks_packinfo.size(0))
        // track of block does not exist
        return;

    auto const track_packinfo  = tracks_packinfo[track_idx];
    auto const track_start_idx = track_packinfo[0];
    auto const N_track_poses   = track_packinfo[1];
    auto const N_total_poses   = tracks_poses.size(0);

    if (N_track_poses <= 1)
        // track has no poses
        return;

    // load the track's timestamps into shared memory to accelerate binary search
    // using all threads (irrespective of whether the local ray exists)
    extern __shared__ int64_t shared_track_timestamps_us[];

    for (auto i = threadIdx.x; i < N_track_poses; i += blockDim.x)
        shared_track_timestamps_us[i] = tracks_timestamps_us[track_start_idx + i];

    __syncthreads(); // all timestamps of the block-associated track are now loaded to shared memory

    if (ray_idx >= rays_o.size(0))
        // local ray doesn't exist
        return;

    // check if ray's timestamp is in the track's valid time-range and exit early if not
    auto const ray_timestamp_us = rays_timestamps_us[ray_idx];

    if (ray_timestamp_us < shared_track_timestamps_us[0] ||
        shared_track_timestamps_us[N_track_poses - 1] < ray_timestamp_us)
        return;

    // ray's time is in the tracks' valid time-range AND we have N_track_poses > 1 -> determine track poses for linear interpolation.
    auto const interpolation_end_idx   = binary_search_interp(ray_timestamp_us, shared_track_timestamps_us, N_track_poses);
    auto const interpolation_start_idx = interpolation_end_idx - 1; // note: this is guaranteed to be in the valid range of poses due to the check above

    auto const local_interpolation_start_timestamp_us = shared_track_timestamps_us[interpolation_start_idx];
    auto const local_interpolation_end_timestamp_us   = shared_track_timestamps_us[interpolation_end_idx];

    // linear interpolation parameter [0,1]
    auto const t_interp = scalar_t(ray_timestamp_us - local_interpolation_start_timestamp_us) / scalar_t(local_interpolation_end_timestamp_us - local_interpolation_start_timestamp_us);

    auto const local_interpolation_start_pose = tracks_poses[track_start_idx + interpolation_start_idx];
    auto const local_interpolation_end_pose   = tracks_poses[track_start_idx + interpolation_end_idx];

    // collect timed-cuboid data for intersection computation
    auto const cuboidtimed = std::array<scalar_t, 17>{cuboids_dims[track_idx][0],
                                                      cuboids_dims[track_idx][1],
                                                      cuboids_dims[track_idx][2],

                                                      local_interpolation_start_pose[0],
                                                      local_interpolation_start_pose[1],
                                                      local_interpolation_start_pose[2],
                                                      local_interpolation_start_pose[3],
                                                      local_interpolation_start_pose[4],
                                                      local_interpolation_start_pose[5],
                                                      local_interpolation_start_pose[6],

                                                      local_interpolation_end_pose[0],
                                                      local_interpolation_end_pose[1],
                                                      local_interpolation_end_pose[2],
                                                      local_interpolation_end_pose[3],
                                                      local_interpolation_end_pose[4],
                                                      local_interpolation_end_pose[5],
                                                      local_interpolation_end_pose[6]};

    // grab timed ray
    auto const ray_o_world = make_float3(rays_o[ray_idx][0], rays_o[ray_idx][1], rays_o[ray_idx][2]);
    auto const ray_d_world = make_float3(rays_d[ray_idx][0], rays_d[ray_idx][1], rays_d[ray_idx][2]);

    // perform time-based cuboid intersection
    auto const [t1t2, ray_o_bbox, ray_d_bbox] = worldray_bboxtimed_intersection(ray_o_world, ray_d_world, t_interp, cuboidtimed);

    // TODO: align into single kernel with ray_cuboidtracks_intersection_kernel?
    if (t1t2.y > 0.f) { // record intersection if ray intersects the cuboid
        intersections_rays_cuboid_o_full[ray_idx][track_idx][0] = ray_o_bbox.x;
        intersections_rays_cuboid_o_full[ray_idx][track_idx][1] = ray_o_bbox.y;
        intersections_rays_cuboid_o_full[ray_idx][track_idx][2] = ray_o_bbox.z;
        intersections_rays_cuboid_d_full[ray_idx][track_idx][0] = ray_d_bbox.x;
        intersections_rays_cuboid_d_full[ray_idx][track_idx][1] = ray_d_bbox.y;
        intersections_rays_cuboid_d_full[ray_idx][track_idx][2] = ray_d_bbox.z;
        intersections_rays_ts_full[ray_idx][track_idx][0]       = t1t2.x;
        intersections_rays_ts_full[ray_idx][track_idx][1]       = t1t2.y;
        intersections_rays_pose_idx_full[ray_idx][track_idx]    = track_start_idx + interpolation_start_idx;
        intersections_flags_full[ray_idx][track_idx]            = true;
    }
}

std::vector<torch::Tensor> ray_cuboidtracks_intersection_transform_filter_cu(
    torch::Tensor const rays_o,             // N_rays x 3 (3d world positions)
    torch::Tensor const rays_d,             // N_rays x 3 (normalized 3d world directions)
    torch::Tensor const rays_timestamps_us, // N_rays (per ray timestamp)

    torch::Tensor const tracks_packinfo,      // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,         // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us, // (N_total_poses) containing per-pose timestamps

    torch::Tensor const cuboids_dims, // (N_tracks x 3) cuboid x/y/z extents (in local track frame)

    int32_t const max_track_n_poses) {

    auto rays_o_arg             = torch::TensorArg{rays_o, "rays_o", 1};
    auto rays_d_arg             = torch::TensorArg{rays_d, "rays_d", 2};
    auto rays_timestamps_us_arg = torch::TensorArg{rays_timestamps_us, "rays_timestamps_us", 3};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 4};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 5};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 6};

    auto cuboids_dims_arg = torch::TensorArg{cuboids_dims, "cuboids_dims", 7};

    torch::checkAllSameType(__func__, {rays_o_arg, rays_d_arg, tracks_poses_arg, cuboids_dims_arg});
    torch::checkScalarType(__func__, rays_timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {rays_o_arg, rays_d_arg, rays_timestamps_us_arg,
                                      tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                      cuboids_dims_arg});
    torch::checkAllContiguous(__func__, {rays_o_arg, rays_d_arg, rays_timestamps_us_arg,
                                         tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg,
                                         cuboids_dims_arg});

    auto const N_rays = rays_o.size(0), N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, rays_o_arg, {N_rays, 3});
    torch::checkSize(__func__, rays_d_arg, {N_rays, 3});
    torch::checkSize(__func__, rays_timestamps_us_arg, {N_rays});

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    torch::checkSize(__func__, cuboids_dims_arg, {N_tracks, 3});

    // allocate buffers for *full* cartesian product of potential ray x tracks intersections
    auto intersections_rays_cuboid_o_full = torch::zeros({N_rays, N_tracks, 3}, rays_o.options());
    auto intersections_rays_cuboid_d_full = torch::zeros({N_rays, N_tracks, 3}, rays_o.options());
    auto intersections_rays_ts_full       = torch::zeros({N_rays, N_tracks, 2}, rays_o.options());
    auto intersections_rays_pose_idx_full = torch::full({N_rays, N_tracks}, -1, torch::dtype(torch::kInt32).device(rays_o.device()));
    auto intersections_flags_full         = torch::zeros({N_rays, N_tracks}, torch::dtype(torch::kBool).device(rays_o.device()));

    if (N_rays == 0 || max_track_n_poses == 0) {
        return {intersections_rays_cuboid_o_full, intersections_rays_cuboid_d_full, intersections_rays_ts_full, intersections_rays_pose_idx_full, intersections_flags_full};
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_rays + threads - 1) / threads, N_tracks);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();
    auto const smem    = max_track_n_poses * sizeof(int64_t); // allocate enough shared memory for all timestamps of the longest track

    AT_DISPATCH_FLOATING_TYPES(rays_o.scalar_type(), "ray_cuboidtracks_intersection_transform_filter_cu", ([&] {
                                   ray_cuboidtracks_intersection_transform_filter_kernel<<<blocks, threads, smem, stream>>>(
                                       rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       cuboids_dims.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),

                                       intersections_rays_cuboid_o_full.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       intersections_rays_cuboid_d_full.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       intersections_rays_ts_full.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       intersections_rays_pose_idx_full.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       intersections_flags_full.packed_accessor32<bool, 2, torch::RestrictPtrTraits>());
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {intersections_rays_cuboid_o_full, intersections_rays_cuboid_d_full, intersections_rays_ts_full, intersections_rays_pose_idx_full, intersections_flags_full};
}

template <typename scalar_t>
__global__ void ray_cuboidtracks_intersection_transform_filter_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const dL_drays_cuboid_o,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const dL_drays_cuboid_d,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_cuboid_o,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_cuboid_d,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const rays_timestamps_us,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> const rays_pose_idx,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const intersection_idx,

    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,

    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_drays_o,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_drays_d,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dtracks_poses) {
    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= dL_drays_cuboid_o.size(0))
        return;

    auto const pose_idx = rays_pose_idx[idx];
    auto const ray_idx  = intersection_idx[idx][0];

    // linear interpolation parameter [0,1]
    auto const pose_start_timestamps_us = tracks_timestamps_us[pose_idx],
               pose_end_timestamps_us   = tracks_timestamps_us[pose_idx + 1];
    auto const t_interp                 = scalar_t(rays_timestamps_us[ray_idx] - pose_start_timestamps_us) /
                          scalar_t(pose_end_timestamps_us - pose_start_timestamps_us);

    auto const local_interpolation_start_pose = tracks_poses[pose_idx],
               local_interpolation_end_pose   = tracks_poses[pose_idx + 1];

    // collect timed-cuboid data for intersection computation
    auto const posetimed = std::array<scalar_t, 14>{local_interpolation_start_pose[0],
                                                    local_interpolation_start_pose[1],
                                                    local_interpolation_start_pose[2],
                                                    local_interpolation_start_pose[3],
                                                    local_interpolation_start_pose[4],
                                                    local_interpolation_start_pose[5],
                                                    local_interpolation_start_pose[6],

                                                    local_interpolation_end_pose[0],
                                                    local_interpolation_end_pose[1],
                                                    local_interpolation_end_pose[2],
                                                    local_interpolation_end_pose[3],
                                                    local_interpolation_end_pose[4],
                                                    local_interpolation_end_pose[5],
                                                    local_interpolation_end_pose[6]};

    // Compute gradient
    auto const d_ray_o_cuboid = make_float3(dL_drays_cuboid_o[idx][0], dL_drays_cuboid_o[idx][1], dL_drays_cuboid_o[idx][2]);
    auto const d_ray_d_cuboid = make_float3(dL_drays_cuboid_d[idx][0], dL_drays_cuboid_d[idx][1], dL_drays_cuboid_d[idx][2]);
    auto const ray_o_cuboid   = make_float3(rays_cuboid_o[idx][0], rays_cuboid_o[idx][1], rays_cuboid_o[idx][2]);
    auto const ray_d_cuboid   = make_float3(rays_cuboid_d[idx][0], rays_cuboid_d[idx][1], rays_cuboid_d[idx][2]);

    auto const [d_ray_d_world, d_ray_o_world, d_t_start, d_T_start, d_t_end, d_T_end] =
        worldray_bboxtimed_intersection_backward(d_ray_o_cuboid, d_ray_d_cuboid, ray_o_cuboid, ray_d_cuboid, t_interp, posetimed);

    // Accumulate gradients to global memory
    atomicAdd(&dL_drays_o[ray_idx][0], d_ray_o_world.x);
    atomicAdd(&dL_drays_o[ray_idx][1], d_ray_o_world.y);
    atomicAdd(&dL_drays_o[ray_idx][2], d_ray_o_world.z);
    atomicAdd(&dL_drays_d[ray_idx][0], d_ray_d_world.x);
    atomicAdd(&dL_drays_d[ray_idx][1], d_ray_d_world.y);
    atomicAdd(&dL_drays_d[ray_idx][2], d_ray_d_world.z);
    atomicAdd(&dL_dtracks_poses[pose_idx][0], d_t_start.x);
    atomicAdd(&dL_dtracks_poses[pose_idx][1], d_t_start.y);
    atomicAdd(&dL_dtracks_poses[pose_idx][2], d_t_start.z);
    atomicAdd(&dL_dtracks_poses[pose_idx][3], d_T_start.x);
    atomicAdd(&dL_dtracks_poses[pose_idx][4], d_T_start.y);
    atomicAdd(&dL_dtracks_poses[pose_idx][5], d_T_start.z);
    atomicAdd(&dL_dtracks_poses[pose_idx + 1][0], d_t_end.x);
    atomicAdd(&dL_dtracks_poses[pose_idx + 1][1], d_t_end.y);
    atomicAdd(&dL_dtracks_poses[pose_idx + 1][2], d_t_end.z);
    atomicAdd(&dL_dtracks_poses[pose_idx + 1][3], d_T_end.x);
    atomicAdd(&dL_dtracks_poses[pose_idx + 1][4], d_T_end.y);
    atomicAdd(&dL_dtracks_poses[pose_idx + 1][5], d_T_end.z);
    // (6th column intentionally left zero otherwise pytorch will complain)
}

std::vector<torch::Tensor> ray_cuboidtracks_intersection_transform_filter_backward_cu(
    torch::Tensor const dL_drays_cuboid_o,  // N_intersections x 3 (gradient of 3d local positions)
    torch::Tensor const dL_drays_cuboid_d,  // N_intersections x 3 (gradient of 3d local directions)
    torch::Tensor const rays_cuboid_o,      // N_intersections x 3 (3d local positions)
    torch::Tensor const rays_cuboid_d,      // N_intersections x 3 (normalized 3d local directions)
    torch::Tensor const rays_timestamps_us, // N_rays (per ray timestamp)
    torch::Tensor const rays_pose_idx,      // N_intersections (pose index of the ray-cuboid intersection)
    torch::Tensor const intersection_idx,   // N_intersections x 2

    torch::Tensor const tracks_poses,        // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us // (N_total_poses) containing per-pose timestamps
) {

    auto dL_drays_cuboid_o_arg    = torch::TensorArg{dL_drays_cuboid_o, "dL_drays_cuboid_o", 1};
    auto dL_drays_cuboid_d_arg    = torch::TensorArg{dL_drays_cuboid_d, "dL_drays_cuboid_d", 2};
    auto rays_cuboid_o_arg        = torch::TensorArg{rays_cuboid_o, "rays_cuboid_o", 3};
    auto rays_cuboid_d_arg        = torch::TensorArg{rays_cuboid_d, "rays_cuboid_d", 4};
    auto rays_timestamps_us_arg   = torch::TensorArg{rays_timestamps_us, "rays_timestamps_us", 5};
    auto rays_pose_idx_arg        = torch::TensorArg{rays_pose_idx, "rays_pose_idx", 6};
    auto intersection_idx_arg     = torch::TensorArg{intersection_idx, "intersection_idx", 7};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 8};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 9};

    torch::checkAllSameType(__func__, {dL_drays_cuboid_o_arg, dL_drays_cuboid_d_arg, rays_cuboid_o_arg, rays_cuboid_d_arg});
    torch::checkScalarType(__func__, rays_timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, intersection_idx_arg, torch::kInt32);
    torch::checkScalarType(__func__, rays_pose_idx_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {dL_drays_cuboid_o_arg, dL_drays_cuboid_d_arg, rays_cuboid_o_arg,
                                      rays_cuboid_d_arg, rays_timestamps_us_arg, rays_pose_idx_arg,
                                      tracks_poses_arg, tracks_timestamps_us_arg});
    torch::checkAllContiguous(__func__, {dL_drays_cuboid_o_arg, dL_drays_cuboid_d_arg, rays_cuboid_o_arg,
                                         rays_cuboid_d_arg, rays_timestamps_us_arg, rays_pose_idx_arg,
                                         tracks_poses_arg, tracks_timestamps_us_arg});

    auto const N_intersections = rays_cuboid_o.size(0), N_total_poses = tracks_poses.size(0);
    auto const N_rays = rays_timestamps_us.size(0);

    torch::checkSize(__func__, dL_drays_cuboid_o_arg, {N_intersections, 3});
    torch::checkSize(__func__, dL_drays_cuboid_d_arg, {N_intersections, 3});
    torch::checkSize(__func__, rays_cuboid_o_arg, {N_intersections, 3});
    torch::checkSize(__func__, rays_cuboid_d_arg, {N_intersections, 3});
    torch::checkSize(__func__, rays_timestamps_us_arg, {N_rays});
    torch::checkSize(__func__, rays_pose_idx_arg, {N_intersections});
    torch::checkSize(__func__, intersection_idx_arg, {N_intersections, 2});

    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    // allocate buffers for sparse gradients
    auto dL_drays_o       = torch::zeros({N_rays, 3}, rays_cuboid_o.options());
    auto dL_drays_d       = torch::zeros({N_rays, 3}, rays_cuboid_o.options());
    auto dL_dtracks_poses = torch::zeros({N_total_poses, 7}, tracks_poses.options());

    if (N_intersections == 0) {
        return {dL_drays_o, dL_drays_d, dL_dtracks_poses};
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_intersections + threads - 1) / threads);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES(dL_drays_cuboid_o.scalar_type(), "ray_cuboidtracks_intersection_transform_filter_backward_cu", ([&] {
                                   ray_cuboidtracks_intersection_transform_filter_backward_kernel<<<blocks, threads, 0, stream>>>(
                                       dL_drays_cuboid_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       dL_drays_cuboid_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_cuboid_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_cuboid_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                                       rays_pose_idx.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                       intersection_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),

                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       dL_drays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       dL_drays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       dL_dtracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dL_drays_o, dL_drays_d, dL_dtracks_poses};
}

template <typename scalar_t>
__global__ void ray_samples_in_distranges_masks_kernel(
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const rays_samples_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> const rays_samples_t,

    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const rays_distranges_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const rays_distranges_ts,

    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> rays_samples_cover_mask) {
    auto const ray_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ray_idx >= rays_samples_packinfo.size(0))
        return;

    // Process all samples of this ray by checking against all distance ranges
    auto const ray_samples_start_idx = rays_samples_packinfo[ray_idx][0];
    auto const ray_Nsamples          = rays_samples_packinfo[ray_idx][1];

    auto const ray_distranges_start_idx = rays_distranges_packinfo[ray_idx][0];
    auto const ray_Ndistranges          = rays_distranges_packinfo[ray_idx][1];

    for (auto sample_i = 0; sample_i < ray_Nsamples; ++sample_i) {
        auto const sample_t = rays_samples_t[ray_samples_start_idx + sample_i];
        for (auto distrange_i = 0; distrange_i < ray_Ndistranges; ++distrange_i) {
            auto const distrang_start_t = rays_distranges_ts[ray_distranges_start_idx + distrange_i][0],
                       distrang_end_t   = rays_distranges_ts[ray_distranges_start_idx + distrange_i][1];

            if (distrang_start_t <= sample_t && sample_t <= distrang_end_t) {
                // sample is within distrange -> record valid cover and continue with next sample of ray
                // (this write doesn't need to be atomic, as only a single thread evaluates this sample)
                rays_samples_cover_mask[ray_samples_start_idx + sample_i] = true;
                break;
            }
        }
    }
}

torch::Tensor ray_samples_in_distranges_masks_cu(
    torch::Tensor const rays_samples_packinfo, // N_rays x 2 (per ray sample packinfo [sample_start_idx, N_samples_of_ray])
    torch::Tensor const rays_samples_t,        // N_total_samples (distances of individual ray samples)

    torch::Tensor const rays_distranges_packinfo, // N_rays x 2 (per ray distranges packinfo [distrange_start_idx, N_distranges_of_ray])
    torch::Tensor const rays_distranges_ts        // N_total_distranges x 2 (distranges of individual rays)
) {

    auto rays_samples_packinfo_arg = torch::TensorArg{rays_samples_packinfo, "rays_samples_packinfo", 1};
    auto rays_samples_t_arg        = torch::TensorArg{rays_samples_t, "rays_samples_t", 2};

    auto rays_distranges_packinfo_arg = torch::TensorArg{rays_distranges_packinfo, "rays_distranges_packinfo", 3};
    auto rays_distranges_ts_arg       = torch::TensorArg{rays_distranges_ts, "rays_distranges_ts", 4};

    torch::checkAllSameType(__func__, {rays_samples_t_arg, rays_distranges_ts_arg});
    torch::checkScalarType(__func__, rays_samples_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, rays_distranges_packinfo_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {rays_samples_packinfo_arg, rays_samples_t_arg, rays_distranges_packinfo_arg,
                                      rays_distranges_ts_arg});
    torch::checkAllContiguous(__func__, {rays_samples_packinfo_arg, rays_samples_t_arg, rays_distranges_packinfo_arg,
                                         rays_distranges_ts_arg});

    auto const N_rays = rays_samples_packinfo.size(0), N_total_samples = rays_samples_t.size(0), N_total_distranges = rays_distranges_ts.size(0);

    torch::checkSize(__func__, rays_samples_packinfo_arg, {N_rays, 2});
    torch::checkSize(__func__, rays_samples_t_arg, {N_total_samples});
    torch::checkSize(__func__, rays_distranges_packinfo_arg, {N_rays, 2});
    torch::checkSize(__func__, rays_distranges_ts_arg, {N_total_distranges, 2});

    // allocate buffer for cover mask of each sample
    auto rays_samples_cover_mask = torch::zeros({N_total_samples}, torch::dtype(torch::kBool).device(rays_samples_t.device()));

    if (N_rays == 0) {
        return rays_samples_cover_mask;
    }

    // each thread processes a single ray
    auto const threads = 256l;
    auto const blocks  = dim3((N_rays + threads - 1) / threads);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES(rays_samples_t.scalar_type(), "ray_samples_in_distranges_masks_cu", ([&] {
                                   ray_samples_in_distranges_masks_kernel<<<blocks, threads, 0, stream>>>(
                                       rays_samples_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       rays_samples_t.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),

                                       rays_distranges_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       rays_distranges_ts.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),

                                       rays_samples_cover_mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>());
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return rays_samples_cover_mask;
}
