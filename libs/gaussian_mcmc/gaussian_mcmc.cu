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
#include <ATen/cuda/CUDAContext.h>
#include <ku/helper_math.cuh>

__global__ void compute_relocation_kernel(
    int N,
    float min_opacity,
    float* __restrict__ opacities,
    float* __restrict__ scales,
    int* __restrict__ ratios,
    float* __restrict__ binoms,
    int n_max,
    float* new_opacities,
    float* new_scales) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;

    int n_idx       = ratios[idx];
    float denom_sum = 0.0f;

    // compute new opacity
    const float opacity     = max(opacities[idx], min_opacity);
    const float new_opacity = 1.0f - powf(1.0f - opacity, 1.0f / n_idx);
    new_opacities[idx]      = new_opacity;

    // compute new scale
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            float bin_coeff = binoms[(i - 1) * n_max + k];
            float term      = (powf(-1.0f, k) * rsqrtf(static_cast<float>(k + 1))) *
                         powf(new_opacity, k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
    const float coeff = (opacity / denom_sum);
#pragma unroll
    for (int i = 0; i < 3; ++i)
        new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
}

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor_cu(
    torch::Tensor opacities,
    torch::Tensor scales,
    torch::Tensor ratios,
    torch::Tensor binoms,
    const int max_num_gaussians,
    const float min_opacity) {
    const int32_t num_gaussians = opacities.size(0);

    torch::Tensor new_opacities = torch::empty_like(opacities);
    torch::Tensor new_scales    = torch::empty_like(scales);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256, blocks = (num_gaussians + threads - 1) / threads;

    if (num_gaussians) {
        compute_relocation_kernel<<<blocks, threads, 0, stream>>>(
            num_gaussians,
            min_opacity,
            opacities.data_ptr<float>(),
            scales.data_ptr<float>(),
            ratios.data_ptr<int>(),
            binoms.data_ptr<float>(),
            max_num_gaussians,
            new_opacities.data_ptr<float>(),
            new_scales.data_ptr<float>());
    }

    return std::make_tuple(new_opacities, new_scales);
}

template <typename scalar_t>
__global__ void quat_scale_to_covariance_kernel(
    const scalar_t* __restrict__ quats,  // [N, 4] (wxyz or xyzw)
    const scalar_t* __restrict__ scales, // [N, 3]
    scalar_t* __restrict__ covars,       // [N, 3, 3]
    int N,
    bool wxyz_format) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;

    // Move the pointer to the current particle
    const scalar_t* quats_ptr  = quats + idx * 4;
    const scalar_t* scales_ptr = scales + idx * 3;

    float4 quat;
    if (wxyz_format) {
        // w, x, y, z
        quat = make_float4(quats_ptr[1], quats_ptr[2], quats_ptr[3], quats_ptr[0]);
    } else {
        // x, y, z, w (default)
        quat = make_float4(quats_ptr[0], quats_ptr[1], quats_ptr[2], quats_ptr[3]);
    }
    const float3 scale = make_float3(scales_ptr[0], scales_ptr[1], scales_ptr[2]);

    Mat3 covar = quat_scale_to_covar(quat, scale);

    scalar_t* covars_ptr = covars + idx * 9;
#pragma unroll
    for (uint32_t i = 0; i < 3; i++) {
        covars_ptr[i * 3]     = covar[i].x;
        covars_ptr[i * 3 + 1] = covar[i].y;
        covars_ptr[i * 3 + 2] = covar[i].z;
    }
}

torch::Tensor quat_scale_to_covariance_cu(
    torch::Tensor quats,
    torch::Tensor scales,
    std::string quaternion_format) {
    auto quats_arg  = torch::TensorArg{quats, "quats", 1};
    auto scales_arg = torch::TensorArg{scales, "scales", 2};

    torch::checkAllSameType(__func__, {quats_arg, scales_arg});
    torch::checkAllSameGPU(__func__, {quats_arg, scales_arg});
    torch::checkAllContiguous(__func__, {quats_arg, scales_arg});

    const int32_t N_gaussians = quats.size(0);

    torch::checkSize(__func__, quats_arg, {N_gaussians, 4});
    torch::checkSize(__func__, scales_arg, {N_gaussians, 3});

    torch::Tensor covars = torch::empty({N_gaussians, 3, 3}, quats.options());

    auto const threads = 256l;
    auto const blocks  = dim3((N_gaussians + threads - 1) / threads);
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

    bool wxyz_format = (quaternion_format == "wxyz");

    if (N_gaussians) {
        AT_DISPATCH_FLOATING_TYPES(quats.scalar_type(), "quat_scale_to_covariance_cu", ([&] {
                                       quat_scale_to_covariance_kernel<<<blocks, threads, 0, stream>>>(
                                           quats.data_ptr<scalar_t>(),
                                           scales.data_ptr<scalar_t>(),
                                           covars.data_ptr<scalar_t>(),
                                           N_gaussians,
                                           wxyz_format);
                                   }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return covars;
}
