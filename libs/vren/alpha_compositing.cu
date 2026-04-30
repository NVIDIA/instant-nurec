// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "utils.h"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <c10/cuda/CUDAStream.h>

#define _SWITCH_CALL_KERNEL_N_FEATS(N_feats)                                       \
    switch (N_feats) {                                                             \
    case 0: _CALL_KERNEL_N_FEATS(0); break;                                        \
    case 1: _CALL_KERNEL_N_FEATS(1); break;                                        \
    case 2: _CALL_KERNEL_N_FEATS(2); break;                                        \
    case 3: _CALL_KERNEL_N_FEATS(3); break;                                        \
    case 4: _CALL_KERNEL_N_FEATS(4); break;                                        \
    case 8: _CALL_KERNEL_N_FEATS(8); break;                                        \
    case 16: _CALL_KERNEL_N_FEATS(16); break;                                      \
    case 32: _CALL_KERNEL_N_FEATS(32); break;                                      \
    default:                                                                       \
        std::cout << "[AlphaCompositing] unsupported number of features ("         \
                  << N_feats << "). Must be in [0,1,2,3,4,8,16,32]." << std::endl; \
        assert(false);                                                             \
    }

template <typename scalar_t>
__global__ void weights_from_alphas_fw_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_info,
    const scalar_t transmittance_threshold,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> total_samples,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> opacity,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ws) {
    const int32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= opacity.size(0))
        return;

    const int32_t start_idx = pack_info[ray_idx][0];
    const int32_t N_samples = pack_info[ray_idx][1];

    // front to back compositing
    int32_t samples        = 0;
    scalar_t transmittance = 1.0f;

    while (samples < N_samples) {
        const int32_t s  = start_idx + samples;
        const scalar_t w = alphas[s] * transmittance; // weight of the sample point
        opacity[ray_idx] += w;
        ws[s] = w;
        transmittance *= 1.0f - alphas[s];
        samples++;
        if (transmittance <= transmittance_threshold)
            break; // ray has enough opacity
    }
    total_samples[ray_idx] = samples;
}

std::vector<torch::Tensor> weights_from_alphas_fw_cu(
    const torch::Tensor alphas,
    const torch::Tensor pack_info,
    const float transmittance_threshold) {
    const int32_t N_rays = pack_info.size(0), N = alphas.size(0);

    auto opacity       = torch::zeros({N_rays}, alphas.options());
    auto ws            = torch::zeros({N}, alphas.options());
    auto total_samples = torch::zeros({N_rays}, torch::dtype(torch::kInt32).device(alphas.device()));

    const int32_t threads = 512, blocks = (N_rays + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, alphas.scalar_type(), "weights_from_alphas_fw_cu",
                                   ([&] {
                                       weights_from_alphas_fw_kernel<scalar_t><<<blocks, threads>>>(
                                           alphas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           pack_info.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                           transmittance_threshold,
                                           total_samples.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                                           opacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           ws.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
                                   }));

    return {total_samples, opacity, ws};
}

template <typename scalar_t>
__global__ void weights_from_alphas_bw_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dws,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dopacity,
    scalar_t* __restrict__ dL_dws_times_ws,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> opacity,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_info,
    const scalar_t transmittance_threshold,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dalphas) {
    const int32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= opacity.size(0))
        return;

    const int32_t start_idx = pack_info[ray_idx][0];
    const int32_t N_samples = pack_info[ray_idx][1];

    // front to back compositing
    int32_t samples        = 0;
    scalar_t O             = opacity[ray_idx];
    scalar_t transmittance = 1.0f;

    // compute prefix sum of dL_dws * ws
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           dL_dws_times_ws + start_idx,
                           dL_dws_times_ws + start_idx + N_samples,
                           dL_dws_times_ws + start_idx);
    scalar_t dL_dws_times_ws_sum = dL_dws_times_ws[start_idx + N_samples - 1];

    while (samples < N_samples) {
        const int32_t s  = start_idx + samples;
        const scalar_t w = alphas[s] * transmittance;
        transmittance *= 1.0f - alphas[s];
        // gradient of the alphas w.r.t. to the loss
        dL_dalphas[s] = (dL_dopacity[ray_idx] * (1 - O) +                                        // gradient from opacity
                         transmittance * dL_dws[s] - (dL_dws_times_ws_sum - dL_dws_times_ws[s])) // gradient from ws
                        / fmaxf(1 - alphas[s], 1e-10f);                                          // guard the division
        samples++;
        if (transmittance <= transmittance_threshold)
            break; // ray has enough opacity
    }
}

torch::Tensor weights_from_alphas_bw_cu(
    const torch::Tensor dL_dws,
    const torch::Tensor dL_dopacity,
    const torch::Tensor alphas,
    const torch::Tensor ws,
    const torch::Tensor opacity,
    const torch::Tensor pack_info,
    const float transmittance_threshold) {
    const int32_t N_rays = pack_info.size(0), N = alphas.size(0);

    auto dL_dalphas      = torch::zeros({N}, alphas.options());
    auto dL_dws_times_ws = dL_dws * ws; // auxiliary input

    const int32_t threads = 512, blocks = (N_rays + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, alphas.scalar_type(), "weights_from_alphas_bw_cu",
                                   ([&] {
                                       weights_from_alphas_bw_kernel<scalar_t><<<blocks, threads>>>(
                                           dL_dws.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           dL_dopacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           dL_dws_times_ws.data_ptr<scalar_t>(),
                                           alphas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           opacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           pack_info.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                           transmittance_threshold,
                                           dL_dalphas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
                                   }));

    return dL_dalphas;
}

template <typename scalar_t>
__global__ void ray_samples_visibility_masks_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> packinfo,
    const scalar_t transmittance_threshold,
    const scalar_t alpha_threshold,
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> mask) {

    const int32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= packinfo.size(0))
        return;

    const int32_t start_idx = packinfo[ray_idx][0];
    const int32_t N_samples = packinfo[ray_idx][1];

    // front to back compositing
    int32_t samples        = 0;
    scalar_t transmittance = 1.0f;

    while (samples < N_samples) {
        const int32_t s              = start_idx + samples;
        const scalar_t current_alpha = alphas[s];
        samples++;

        transmittance *= 1.0f - current_alpha;

        if (current_alpha <= alpha_threshold)
            continue; // ray doesn't have enough density so it will be ignored

        // Sample contributes to the volume rendering
        mask[s] = true;

        if (transmittance <= transmittance_threshold)
            break; // ray has enough opacity so the following samples will be skipped
    }
}

torch::Tensor ray_samples_visibility_masks_cu(
    const torch::Tensor alphas,          // Alpha value of each sample [N_samples]
    const torch::Tensor packinfo,        // (N_rays x 2) with [ray_start_idx, N_sample_for_this_ray] each
    const float transmittance_threshold, // Threshold of the accumulated transmitance
    const float alpha_threshold) {       // Threshold on the alpha value of each sample

    auto alphas_arg   = torch::TensorArg{alphas, "alphas", 1};
    auto packinfo_arg = torch::TensorArg{packinfo, "packinfo", 2};

    torch::checkAllSameGPU(__func__, {alphas_arg, packinfo_arg});
    torch::checkAllContiguous(__func__, {alphas_arg, packinfo_arg});

    const int32_t N_rays = packinfo.size(0), N = alphas.size(0);

    auto mask = torch::zeros({N}, torch::dtype(torch::kBool).device(alphas.device()));

    auto const threads   = 256;
    const int32_t blocks = (N_rays + threads - 1) / threads;
    auto const stream    = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, alphas.scalar_type(), "ray_samples_visibility_masks_cu",
                                   ([&] {
                                       ray_samples_visibility_masks_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                           alphas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           packinfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                           transmittance_threshold,
                                           alpha_threshold,
                                           mask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>());
                                   }));

    return mask;
}

template <typename scalar_t, int N_feats>
__global__ void alpha_composite_train_fw_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> feats,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ts,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_info,
    const scalar_t transmittance_threshold,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> total_samples,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> opacity,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> distance,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> feat,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ws) {
    const int32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= opacity.size(0))
        return;

    const int32_t start_idx = pack_info[ray_idx][0];
    const int32_t N_samples = pack_info[ray_idx][1];

    // front to back compositing
    int32_t samples        = 0;
    scalar_t transmittance = 1.0f;

    while (samples < N_samples) {
        const int32_t s  = start_idx + samples;
        const scalar_t w = alphas[s] * transmittance; // weight of the sample point

#pragma unroll
        for (int32_t i = 0; i < N_feats; ++i) {
            feat[ray_idx][i] += w * feats[s][i];
        }
        distance[ray_idx] += w * ts[s];
        opacity[ray_idx] += w;
        ws[s] = w;
        transmittance *= 1.0f - alphas[s];
        samples++;

        if (transmittance <= transmittance_threshold)
            break; // ray has enough opacity
    }
    total_samples[ray_idx] = samples;
}

std::vector<torch::Tensor> alpha_composite_train_fw_cu(
    const torch::Tensor alphas,
    const torch::Tensor feats,
    const torch::Tensor ts,
    const torch::Tensor pack_info,
    const float transmittance_threshold) {
    const int32_t N_rays = pack_info.size(0), N = alphas.size(0), N_feats = feats.size(1);

    auto opacity       = torch::zeros({N_rays}, alphas.options());
    auto distance      = torch::zeros({N_rays}, alphas.options());
    auto feat          = torch::zeros({N_rays, N_feats}, alphas.options());
    auto ws            = torch::zeros({N}, alphas.options());
    auto total_samples = torch::zeros({N_rays}, torch::dtype(torch::kLong).device(alphas.device()));

    const int32_t threads = 512, blocks = (N_rays + threads - 1) / threads;

#undef _CALL_KERNEL_N_FEATS
#define _CALL_KERNEL_N_FEATS(N_feats)                                                                               \
    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, alphas.scalar_type(), "alpha_composite_train_fw_cu",    \
                                   ([&] {                                                                           \
                                       alpha_composite_train_fw_kernel<scalar_t, N_feats><<<blocks, threads>>>(     \
                                           alphas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),       \
                                           feats.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),        \
                                           ts.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),           \
                                           pack_info.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),     \
                                           transmittance_threshold,                                                 \
                                           total_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(), \
                                           opacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),      \
                                           distance.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),     \
                                           feat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),         \
                                           ws.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());          \
                                   }))

    _SWITCH_CALL_KERNEL_N_FEATS(N_feats);

    return {total_samples, opacity, distance, feat, ws};
}

template <typename scalar_t, int N_feats, int N_feats_or_1 = std::max(N_feats, 1)>
__global__ void alpha_composite_train_bw_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dopacity,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_ddistance,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dfeat,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dws,
    scalar_t* __restrict__ dL_dws_times_ws,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> feats,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ts,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_info,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> opacity,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> distance,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> feat,
    const scalar_t transmittance_threshold,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dalphas,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dfeats) {
    const int32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= opacity.size(0))
        return;

    const int32_t start_idx = pack_info[ray_idx][0];
    const int32_t N_samples = pack_info[ray_idx][1];

    // front to back compositing
    int32_t samples = 0;
    scalar_t f[N_feats_or_1]; // Should be at least 1-sized, since zero-sized arrays are not allowed.
#pragma unroll
    for (int32_t i = 0; i < N_feats; ++i) {
        f[i] = 0.0f;
    }
    scalar_t O = opacity[ray_idx], D = distance[ray_idx];
    scalar_t transmittance = 1.0f, d = 0.0f;
    // compute prefix sum of dL_dws * ws
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           dL_dws_times_ws + start_idx,
                           dL_dws_times_ws + start_idx + N_samples,
                           dL_dws_times_ws + start_idx);
    scalar_t dL_dws_times_ws_sum = dL_dws_times_ws[start_idx + N_samples - 1];

    while (samples < N_samples) {
        const int32_t s  = start_idx + samples;
        const scalar_t w = alphas[s] * transmittance;

#pragma unroll
        for (int32_t i = 0; i < N_feats; ++i) {
            f[i] += w * feats[s][i];
        }
        d += w * ts[s];
        transmittance *= 1.0f - alphas[s];

        // compute gradients by math...
        scalar_t dL_dalphas_dfeat = 0.0f;
#pragma unroll
        for (int32_t i = 0; i < N_feats; ++i) {
            dL_dfeats[s][i] = dL_dfeat[ray_idx][i] * w;
            dL_dalphas_dfeat += dL_dfeat[ray_idx][i] * (feats[s][i] * transmittance - (feat[ray_idx][i] - f[i]));
        }
        // gradient of the alphas w.r.t. to the loss
        dL_dalphas[s] = (dL_dalphas_dfeat +                                                      // gradients from feat
                         dL_dopacity[ray_idx] * (1 - O) +                                        // gradient from opacity
                         dL_ddistance[ray_idx] * (ts[s] * transmittance - (D - d)) +             // gradient from distance
                         transmittance * dL_dws[s] - (dL_dws_times_ws_sum - dL_dws_times_ws[s])) // gradient from ws
                        / fmaxf(1 - alphas[s], 1e-10f);                                          // guard the dicision

        if (transmittance <= transmittance_threshold)
            break; // ray has enough opacity
        samples++;
    }
}

std::vector<torch::Tensor> alpha_composite_train_bw_cu(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddistance,
    const torch::Tensor dL_dfeat,
    const torch::Tensor dL_dws,
    const torch::Tensor alphas,
    const torch::Tensor feats,
    const torch::Tensor ws,
    const torch::Tensor ts,
    const torch::Tensor pack_info,
    const torch::Tensor opacity,
    const torch::Tensor distance,
    const torch::Tensor feat,
    const float transmittance_threshold) {
    const int32_t N = alphas.size(0), N_rays = pack_info.size(0), N_feats = feats.size(1);

    auto dL_dalphas = torch::zeros({N}, alphas.options());
    auto dL_dfeats  = torch::zeros({N, N_feats}, alphas.options());

    auto dL_dws_times_ws = dL_dws * ws; // auxiliary input

    const int32_t threads = 512, blocks = (N_rays + threads - 1) / threads;

#undef _CALL_KERNEL_N_FEATS
#define _CALL_KERNEL_N_FEATS(N_feats)                                                                               \
    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, alphas.scalar_type(), "alpha_composite_train_bw_cu",    \
                                   ([&] {                                                                           \
                                       alpha_composite_train_bw_kernel<scalar_t, N_feats><<<blocks, threads>>>(     \
                                           dL_dopacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),  \
                                           dL_ddistance.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(), \
                                           dL_dfeat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),     \
                                           dL_dws.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),       \
                                           dL_dws_times_ws.data_ptr<scalar_t>(),                                    \
                                           alphas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),       \
                                           feats.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),        \
                                           ts.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),           \
                                           pack_info.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),     \
                                           opacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),      \
                                           distance.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),     \
                                           feat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),         \
                                           transmittance_threshold,                                                 \
                                           dL_dalphas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),   \
                                           dL_dfeats.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());   \
                                   }))

    _SWITCH_CALL_KERNEL_N_FEATS(N_feats);

    return {dL_dalphas, dL_dfeats};
}

template <typename scalar_t, int N_feats>
__global__ void alpha_composite_test_fw_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> feats,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> ts,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> hits_t,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> alive_indices,
    const scalar_t transmittance_threshold,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> N_eff_samples,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> opacity,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> distance,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> feat) {
    const int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0))
        return;

    if (N_eff_samples[n] == 0) { // no hit
        alive_indices[n] = -1;
        return;
    }

    const size_t r = alive_indices[n]; // ray index

    // front to back compositing
    int32_t s              = 0;
    scalar_t transmittance = 1 - opacity[r];

    while (s < N_eff_samples[n]) {
        const scalar_t w = alphas[n][s] * transmittance;

#pragma unroll
        for (int32_t i = 0; i < N_feats; ++i) {
            feat[r][i] += w * feats[n][s][i];
        }
        distance[r] += w * ts[n][s];
        opacity[r] += w;
        transmittance *= 1.0f - alphas[n][s];

        if (transmittance <= transmittance_threshold) { // ray has enough opacity
            alive_indices[n] = -1;
            break;
        }
        s++;
    }
}

void alpha_composite_test_fw_cu(
    const torch::Tensor alphas,
    const torch::Tensor feats,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    torch::Tensor alive_indices,
    const float transmittance_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor distance,
    torch::Tensor feat) {
    const int32_t N_rays = alive_indices.size(0), N_feats = feats.size(1);

    const int32_t threads = 512, blocks = (N_rays + threads - 1) / threads;

#undef _CALL_KERNEL_N_FEATS
#define _CALL_KERNEL_N_FEATS(N_feats)                                                                               \
    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, alphas.scalar_type(), "alpha_composite_test_fw_cu",     \
                                   ([&] {                                                                           \
                                       alpha_composite_test_fw_kernel<scalar_t, N_feats><<<blocks, threads>>>(      \
                                           alphas.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),       \
                                           feats.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),        \
                                           ts.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),           \
                                           hits_t.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),       \
                                           alive_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), \
                                           transmittance_threshold,                                                 \
                                           N_eff_samples.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(), \
                                           opacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),      \
                                           distance.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),     \
                                           feat.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());        \
                                   }))

    _SWITCH_CALL_KERNEL_N_FEATS(N_feats);
}
