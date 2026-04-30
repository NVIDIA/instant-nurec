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

#include <ku/helper_math.cuh>

#include <ku/binary_search.cuh>
#include <ku/common.cuh>

#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void frame_poses_interpolation_kernel(
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const frame_timestamps_us,

    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> const tracks_packinfo,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> const tracks_poses,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> const tracks_timestamps_us,

    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> valid_tracks_cnt,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> valid_tracks_idx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> valid_tracks_start_poses,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> valid_tracks_end_poses) {

    auto const track_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= tracks_packinfo.size(0))
        return;

    auto const track_packinfo     = tracks_packinfo[track_idx];
    const int32_t track_start_idx = tracks_packinfo[track_idx][0];
    const int32_t n_track_poses   = tracks_packinfo[track_idx][1];

    if (n_track_poses <= 1)
        return;

    const int64_t start_timestamp_us = frame_timestamps_us[0];
    const int64_t end_timestamp_us   = frame_timestamps_us[1];

    if ((tracks_timestamps_us[track_start_idx] > start_timestamp_us) ||
        (tracks_timestamps_us[track_start_idx + n_track_poses - 1] < end_timestamp_us))
        return;

    const int32_t start_interpolation_end_idx   = binary_search_interp(start_timestamp_us, &tracks_timestamps_us[track_start_idx], n_track_poses) + track_start_idx;
    const int32_t start_interpolation_start_idx = start_interpolation_end_idx - 1;

    const int32_t end_interpolation_end_idx   = binary_search_interp(end_timestamp_us, &tracks_timestamps_us[start_interpolation_start_idx], n_track_poses - (start_interpolation_start_idx - track_start_idx)) + start_interpolation_start_idx;
    const int32_t end_interpolation_start_idx = end_interpolation_end_idx - 1;

    const int32_t idx = atomicAdd(&valid_tracks_cnt[0], 1);

    valid_tracks_idx[idx][0] = track_idx; // mapping idx
    valid_tracks_idx[idx][1] = track_idx; // unique id [currently same as mapping id]

    auto const interpolate_pose = [](auto startPose,
                                     auto endPose,
                                     scalar_t alpha,
                                     auto interpPose) {
        auto const c_start = make_float3(startPose[0], startPose[1], startPose[2]);
        auto const q_start = make_float4(startPose[3], startPose[4], startPose[5], startPose[6]);

        auto const c_end = make_float3(endPose[0], endPose[1], endPose[2]);
        auto const q_end = make_float4(endPose[3], endPose[4], endPose[5], endPose[6]);

        auto const c_interp = (1.f - alpha) * c_start + alpha * c_end;
        auto const q_interp = unitquat_slerp(q_start, q_end, alpha);

        interpPose[0] = c_interp.x;
        interpPose[1] = c_interp.y;
        interpPose[2] = c_interp.z;

        interpPose[3] = q_interp.x;
        interpPose[4] = q_interp.y;
        interpPose[5] = q_interp.z;
        interpPose[6] = q_interp.w;
    };

    const int64_t start_interpolation_start_timestamp_us = tracks_timestamps_us[start_interpolation_start_idx];
    interpolate_pose(
        tracks_poses[start_interpolation_start_idx],
        tracks_poses[start_interpolation_end_idx],
        scalar_t(start_timestamp_us - start_interpolation_start_timestamp_us) / scalar_t(tracks_timestamps_us[start_interpolation_end_idx] - start_interpolation_start_timestamp_us),
        valid_tracks_start_poses[idx]);

    const int64_t end_interpolation_start_timestamp_us = tracks_timestamps_us[end_interpolation_start_idx];
    interpolate_pose(
        tracks_poses[end_interpolation_start_idx],
        tracks_poses[end_interpolation_end_idx],
        scalar_t(end_timestamp_us - end_interpolation_start_timestamp_us) / scalar_t(tracks_timestamps_us[end_interpolation_end_idx] - end_interpolation_start_timestamp_us),
        valid_tracks_end_poses[idx]);
}

std::vector<torch::Tensor> cuboidtracks_frame_poses_interpolation_cu(
    torch::Tensor const frame_timestamps_us, // 2

    torch::Tensor const tracks_packinfo,     // (N_tracks x 2) with [track_start_idx, N_track_poses] each
    torch::Tensor const tracks_poses,        // (N_total_poses x 7) containing quat-encoded SE3 pose each [translation, normalized quaternion]
    torch::Tensor const tracks_timestamps_us // (N_total_poses) containing per-pose timestamps
) {

    auto frame_timestamps_us_arg = torch::TensorArg{frame_timestamps_us, "frame_timestamps_us", 1};

    auto tracks_packinfo_arg      = torch::TensorArg{tracks_packinfo, "tracks_packinfo", 2};
    auto tracks_poses_arg         = torch::TensorArg{tracks_poses, "tracks_poses", 3};
    auto tracks_timestamps_us_arg = torch::TensorArg{tracks_timestamps_us, "tracks_timestamps_us", 4};

    torch::checkScalarType(__func__, frame_timestamps_us_arg, torch::kLong);
    torch::checkScalarType(__func__, tracks_packinfo_arg, torch::kInt32);
    torch::checkScalarType(__func__, tracks_timestamps_us_arg, torch::kLong);

    torch::checkAllSameGPU(__func__, {frame_timestamps_us_arg, tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg});
    torch::checkAllContiguous(__func__, {frame_timestamps_us_arg, tracks_packinfo_arg, tracks_poses_arg, tracks_timestamps_us_arg});

    auto const N_tracks = tracks_packinfo.size(0), N_total_poses = tracks_poses.size(0);

    torch::checkSize(__func__, frame_timestamps_us_arg, {2});

    torch::checkSize(__func__, tracks_packinfo_arg, {N_tracks, 2});
    torch::checkSize(__func__, tracks_poses_arg, {N_total_poses, 7});
    torch::checkSize(__func__, tracks_timestamps_us_arg, {N_total_poses});

    auto valid_tracks_cnt         = torch::zeros({1}, torch::dtype(torch::kInt32).device(tracks_poses.device()));
    auto valid_tracks_idx         = torch::full({N_tracks, 2}, -1, torch::dtype(torch::kInt32).device(tracks_poses.device()));
    auto valid_tracks_start_poses = torch::zeros({N_tracks, 7}, tracks_poses.options());
    auto valid_tracks_end_poses   = torch::zeros({N_tracks, 7}, tracks_poses.options());

    if (N_tracks == 0) {
        return {valid_tracks_cnt, valid_tracks_idx, valid_tracks_start_poses, valid_tracks_end_poses};
    }

    auto const threads = 256l; // N threads cooperate in processing a single track within each block
    auto const blocks  = dim3((N_tracks + threads - 1) / threads, 1);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES(tracks_poses.scalar_type(), "cuboidtracks_frame_poses_interpolation_cu", ([&] {
                                   frame_poses_interpolation_kernel<<<blocks, threads, 0, stream>>>(
                                       frame_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       tracks_packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       tracks_timestamps_us.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),

                                       valid_tracks_cnt.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                       valid_tracks_idx.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                       valid_tracks_start_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       valid_tracks_end_poses.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
                               }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {valid_tracks_cnt, valid_tracks_idx, valid_tracks_start_poses, valid_tracks_end_poses};
}
