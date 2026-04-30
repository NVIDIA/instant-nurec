// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Hand-written CUDA translation of strategy/gsplat.slang
// update_gradient_buffers_kernel: accumulates distance-scaled gradient norms
// for GSplat densification decisions (split/clone).

#include "gsplat_cuda.h"

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <ku/common.cuh>
#include <ku/common_host.h>
#include <torch/extension.h>

__global__ void update_gradient_buffers_kernel(
    int const n,
    float const* __restrict__ positions,
    float const* __restrict__ params_grad,
    float const* __restrict__ ray_origin,
    float* __restrict__ grad_norm_accum,
    int* __restrict__ grad_norm_denom) {

    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }

    // Load gradient (contiguous [N, 3] layout)
    float3 const grad = *reinterpret_cast<float3 const*>(params_grad + tid * 3);

    // Check if this Gaussian was hit (any non-zero gradient component)
    if (grad.x == 0.0f && grad.y == 0.0f && grad.z == 0.0f) {
        return;
    }

    // Load position and ray origin
    float3 const pos    = *reinterpret_cast<float3 const*>(positions + tid * 3);
    float3 const origin = *reinterpret_cast<float3 const*>(ray_origin);

    // Compute distance to camera
    float const dx       = pos.x - origin.x;
    float const dy       = pos.y - origin.y;
    float const dz       = pos.z - origin.z;
    float const distance = sqrtf(dx * dx + dy * dy + dz * dz);

    // Scale gradient by distance and compute norm, divided by 2
    float const sx        = grad.x * distance;
    float const sy        = grad.y * distance;
    float const sz        = grad.z * distance;
    float const grad_norm = sqrtf(sx * sx + sy * sy + sz * sz) * 0.5f;

    // Accumulate (each Gaussian processed by exactly one thread, no atomics needed)
    grad_norm_accum[tid] += grad_norm;
    grad_norm_denom[tid] += 1;
}

void update_gradient_buffers_cuda(
    torch::Tensor const positions,
    torch::Tensor const params_grad,
    torch::Tensor const ray_origin,
    torch::Tensor grad_norm_accum,
    torch::Tensor grad_norm_denom,
    int threads_per_block) {

    CHECK_INPUT(positions);
    CHECK_INPUT(params_grad);
    CHECK_INPUT(ray_origin);
    CHECK_INPUT(grad_norm_accum);
    CHECK_INPUT(grad_norm_denom);

    // Shape validation
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                "positions must be [N, 3], got ", positions.sizes());
    TORCH_CHECK(params_grad.dim() == 2 && params_grad.size(1) == 3,
                "params_grad must be [N, 3], got ", params_grad.sizes());
    TORCH_CHECK(ray_origin.dim() == 1 && ray_origin.size(0) == 3,
                "ray_origin must be [3], got ", ray_origin.sizes());
    TORCH_CHECK(grad_norm_accum.dim() == 2 && grad_norm_accum.size(1) == 1,
                "grad_norm_accum must be [N, 1], got ", grad_norm_accum.sizes());
    TORCH_CHECK(grad_norm_denom.dim() == 2 && grad_norm_denom.size(1) == 1,
                "grad_norm_denom must be [N, 1], got ", grad_norm_denom.sizes());
    TORCH_CHECK(params_grad.size(0) == positions.size(0),
                "params_grad must have same batch size as positions");
    TORCH_CHECK(grad_norm_accum.size(0) == positions.size(0),
                "grad_norm_accum must have same batch size as positions");
    TORCH_CHECK(grad_norm_denom.size(0) == positions.size(0),
                "grad_norm_denom must have same batch size as positions");

    // Dtype validation
    TORCH_CHECK(positions.dtype() == torch::kFloat32, "positions must be float32");
    TORCH_CHECK(params_grad.dtype() == torch::kFloat32, "params_grad must be float32");
    TORCH_CHECK(ray_origin.dtype() == torch::kFloat32, "ray_origin must be float32");
    TORCH_CHECK(grad_norm_accum.dtype() == torch::kFloat32, "grad_norm_accum must be float32");
    TORCH_CHECK(grad_norm_denom.dtype() == torch::kInt32, "grad_norm_denom must be int32");

    int const n = positions.size(0);
    if (n == 0) {
        return;
    }

    int const blocks = div_round_up(static_cast<uint32_t>(n), static_cast<uint32_t>(threads_per_block));

    c10::cuda::OptionalCUDAGuard const device_guard(torch::device_of(positions));
    auto stream = c10::cuda::getCurrentCUDAStream();

    update_gradient_buffers_kernel<<<blocks, threads_per_block, 0, stream>>>(
        n,
        positions.data_ptr<float>(),
        params_grad.data_ptr<float>(),
        ray_origin.data_ptr<float>(),
        grad_norm_accum.data_ptr<float>(),
        grad_norm_denom.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
