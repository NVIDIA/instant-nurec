// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "losses_cuda.h"

#include <c10/cuda/CUDAGuard.h>

#include "losses_cuda.cuh"

// DISPATCH 3: Gaussian Regularization Losses
// ============================================================================

__global__ void gaussian_losses_forward_kernel(
    int N_scales, int N_densities, int N_z_scales, int N_oob,
    float scale_factor, float density_factor, float z_scale_factor, float z_scale_threshold, float oob_factor,
    const float* __restrict__ gaussian_visibility,
    const float3* __restrict__ gaussian_scales,
    const float* __restrict__ gaussian_densities,
    const float3* __restrict__ gaussian_z_scales,
    const float3* __restrict__ oob_positions,
    const float3* __restrict__ oob_cuboid_dims,
    float* __restrict__ scale_loss,
    float* __restrict__ density_loss,
    float* __restrict__ z_scale_loss,
    float* __restrict__ oob_loss) {
    int ti = blockIdx.x * LOSSES_BLOCK_THREADS + threadIdx.x;

    // Scale Loss: sum(exp(preact)) * factor
    if (scale_factor >= 0 && ti < N_scales) {
        float scale_l = 0.0f;
        if (scale_factor > 0) {
            float3 s = gaussian_scales[ti];
            float sx = expf(s.x);
            float sy = expf(s.y);
            float sz = expf(s.z);
            scale_l  = (sx + sy + sz) * scale_factor * gaussian_visibility[ti];
        }
        scale_loss[ti] = scale_l;
    }

    // Density Loss: |density| * factor
    if (density_factor >= 0 && ti < N_densities) {
        float density_l = 0.0f;
        if (density_factor > 0) {
            density_l = fabsf(gaussian_densities[ti]) * density_factor * gaussian_visibility[ti];
        }
        density_loss[ti] = density_l;
    }

    // Z-Scale Loss: max(0, exp(preact_z) - threshold) * factor
    if (z_scale_factor >= 0 && ti < N_z_scales) {
        float z_scale_l = 0.0f;
        if (z_scale_factor > 0) {
            float z_scale = expf(gaussian_z_scales[ti].z); // z-component
            z_scale_l     = fmaxf(0.0f, z_scale - z_scale_threshold) * z_scale_factor;
        }
        z_scale_loss[ti] = z_scale_l;
    }

    // Out-of-Bound Loss: sum(relu(|pos| - half_dims)) * factor
    if (oob_factor >= 0 && ti < N_oob) {
        float oob_l = 0.0f;
        if (oob_factor > 0) {
            float3 pos = oob_positions[ti];
            float3 hd  = oob_cuboid_dims[ti];
            float lx   = fmaxf(0.0f, fabsf(pos.x) - hd.x * 0.5f);
            float ly   = fmaxf(0.0f, fabsf(pos.y) - hd.y * 0.5f);
            float lz   = fmaxf(0.0f, fabsf(pos.z) - hd.z * 0.5f);
            oob_l      = (lx + ly + lz) * oob_factor;
        }
        oob_loss[ti] = oob_l;
    }
}

__global__ void gaussian_losses_backward_kernel(
    int N_scales, int N_densities, int N_z_scales, int N_oob,
    float scale_factor, float density_factor, float z_scale_factor, float z_scale_threshold, float oob_factor,
    const float* __restrict__ gaussian_visibility,
    const float3* __restrict__ gaussian_scales,
    const float* __restrict__ gaussian_densities,
    const float3* __restrict__ gaussian_z_scales,
    const float3* __restrict__ oob_positions,
    const float3* __restrict__ oob_cuboid_dims,
    const float* __restrict__ grad_scale_loss,
    const float* __restrict__ grad_density_loss,
    const float* __restrict__ grad_z_scale_loss,
    const float* __restrict__ grad_oob_loss,
    float3* __restrict__ grad_gaussian_scales,
    float* __restrict__ grad_gaussian_densities,
    float3* __restrict__ grad_gaussian_z_scales,
    float3* __restrict__ grad_oob_positions) {
    int ti = blockIdx.x * LOSSES_BLOCK_THREADS + threadIdx.x;

    // Scale backward: d(exp(s)*factor*vis)/ds = exp(s)*factor*vis
    if (scale_factor > 0 && ti < N_scales) {
        float3 s                 = gaussian_scales[ti];
        float gl                 = grad_scale_loss[ti];
        float scale              = scale_factor * gl * gaussian_visibility[ti];
        grad_gaussian_scales[ti] = make_float3(expf(s.x) * scale, expf(s.y) * scale, expf(s.z) * scale);
    }

    // Density backward: d(|d|*factor*vis)/dd = sign(d)*factor*vis
    if (density_factor > 0 && ti < N_densities) {
        grad_gaussian_densities[ti] =
            sign_f(gaussian_densities[ti]) * density_factor * grad_density_loss[ti] * gaussian_visibility[ti];
    }

    // Z-Scale backward: d max(0, exp(z)-thr)/dz = exp(z) if exp(z) > thr, else 0
    // NOTE: grad_gaussian_z_scales must be zero-initialized by the caller —
    // only the z-component is written when above threshold, x/y remain zero.
    if (z_scale_factor > 0 && ti < N_z_scales) {
        float z_preact = gaussian_z_scales[ti].z;
        float z_scale  = expf(z_preact);
        if (z_scale > z_scale_threshold) {
            float3 g                   = grad_gaussian_z_scales[ti];
            g.z                        = z_scale * z_scale_factor * grad_z_scale_loss[ti];
            grad_gaussian_z_scales[ti] = g;
        }
    }

    // Out-of-Bound backward: d relu(|p|-hd)/dp = sign(p) if |p| > hd, else 0
    // NOTE: grad_oob_positions must be zero-initialized by the caller —
    // only components where |pos| > half_dim are written, others remain zero.
    if (oob_factor > 0 && ti < N_oob) {
        float gl    = grad_oob_loss[ti];
        float scale = oob_factor * gl;
        float3 pos  = oob_positions[ti];
        float3 hd   = oob_cuboid_dims[ti];
        float3 grad = grad_oob_positions[ti];
        if (fabsf(pos.x) > hd.x * 0.5f)
            grad.x = sign_f(pos.x) * scale;
        if (fabsf(pos.y) > hd.y * 0.5f)
            grad.y = sign_f(pos.y) * scale;
        if (fabsf(pos.z) > hd.z * 0.5f)
            grad.z = sign_f(pos.z) * scale;
        grad_oob_positions[ti] = grad;
    }
}

void gaussian_losses_forward_cuda(
    int N_scales, int N_densities, int N_z_scales, int N_oob,
    float scale_factor, float density_factor, float z_scale_factor, float z_scale_threshold, float oob_factor,
    torch::Tensor const gaussian_visibility,
    torch::Tensor const gaussian_scales,
    torch::Tensor const gaussian_densities,
    torch::Tensor const gaussian_z_scales,
    torch::Tensor const oob_positions,
    torch::Tensor const oob_cuboid_dims,
    torch::Tensor const scale_loss,
    torch::Tensor const density_loss,
    torch::Tensor const z_scale_loss,
    torch::Tensor const oob_loss) {
    CHECK_INPUT(gaussian_visibility);
    CHECK_INPUT(gaussian_scales);
    CHECK_INPUT(gaussian_densities);
    CHECK_INPUT(gaussian_z_scales);
    CHECK_INPUT(oob_positions);
    CHECK_INPUT(oob_cuboid_dims);
    CHECK_INPUT(scale_loss);
    CHECK_INPUT(density_loss);
    CHECK_INPUT(z_scale_loss);
    CHECK_INPUT(oob_loss);

    at::cuda::CUDAGuard device_guard(gaussian_scales.device());
    int N = max(max(N_scales, N_densities), max(N_z_scales, N_oob));
    if (N == 0)
        return;
    dim3 threads(LOSSES_BLOCK_THREADS);
    dim3 blocks(div_round_up(N, LOSSES_BLOCK_THREADS));

    gaussian_losses_forward_kernel<<<blocks, threads>>>(
        N_scales, N_densities, N_z_scales, N_oob,
        scale_factor, density_factor, z_scale_factor, z_scale_threshold, oob_factor,
        gaussian_visibility.data_ptr<float>(),
        reinterpret_cast<const float3*>(gaussian_scales.data_ptr<float>()),
        gaussian_densities.data_ptr<float>(),
        reinterpret_cast<const float3*>(gaussian_z_scales.data_ptr<float>()),
        reinterpret_cast<const float3*>(oob_positions.data_ptr<float>()),
        reinterpret_cast<const float3*>(oob_cuboid_dims.data_ptr<float>()),
        scale_loss.data_ptr<float>(),
        density_loss.data_ptr<float>(),
        z_scale_loss.data_ptr<float>(),
        oob_loss.data_ptr<float>());
}

void gaussian_losses_backward_cuda(
    int N_scales, int N_densities, int N_z_scales, int N_oob,
    float scale_factor, float density_factor, float z_scale_factor, float z_scale_threshold, float oob_factor,
    torch::Tensor const gaussian_visibility,
    torch::Tensor const gaussian_scales,
    torch::Tensor const gaussian_densities,
    torch::Tensor const gaussian_z_scales,
    torch::Tensor const oob_positions,
    torch::Tensor const oob_cuboid_dims,
    torch::Tensor const grad_scale_loss,
    torch::Tensor const grad_density_loss,
    torch::Tensor const grad_z_scale_loss,
    torch::Tensor const grad_oob_loss,
    torch::Tensor const grad_gaussian_scales,
    torch::Tensor const grad_gaussian_densities,
    torch::Tensor const grad_gaussian_z_scales,
    torch::Tensor const grad_oob_positions) {
    CHECK_INPUT(gaussian_visibility);
    CHECK_INPUT(gaussian_scales);
    CHECK_INPUT(gaussian_densities);
    CHECK_INPUT(gaussian_z_scales);
    CHECK_INPUT(oob_positions);
    CHECK_INPUT(oob_cuboid_dims);
    CHECK_INPUT(grad_scale_loss);
    CHECK_INPUT(grad_density_loss);
    CHECK_INPUT(grad_z_scale_loss);
    CHECK_INPUT(grad_oob_loss);
    CHECK_INPUT(grad_gaussian_scales);
    CHECK_INPUT(grad_gaussian_densities);
    CHECK_INPUT(grad_gaussian_z_scales);
    CHECK_INPUT(grad_oob_positions);

    at::cuda::CUDAGuard device_guard(gaussian_scales.device());
    int N = max(max(N_scales, N_densities), max(N_z_scales, N_oob));
    if (N == 0)
        return;
    dim3 threads(LOSSES_BLOCK_THREADS);
    dim3 blocks(div_round_up(N, LOSSES_BLOCK_THREADS));

    gaussian_losses_backward_kernel<<<blocks, threads>>>(
        N_scales, N_densities, N_z_scales, N_oob,
        scale_factor, density_factor, z_scale_factor, z_scale_threshold, oob_factor,
        gaussian_visibility.data_ptr<float>(),
        reinterpret_cast<const float3*>(gaussian_scales.data_ptr<float>()),
        gaussian_densities.data_ptr<float>(),
        reinterpret_cast<const float3*>(gaussian_z_scales.data_ptr<float>()),
        reinterpret_cast<const float3*>(oob_positions.data_ptr<float>()),
        reinterpret_cast<const float3*>(oob_cuboid_dims.data_ptr<float>()),
        grad_scale_loss.data_ptr<float>(),
        grad_density_loss.data_ptr<float>(),
        grad_z_scale_loss.data_ptr<float>(),
        grad_oob_loss.data_ptr<float>(),
        reinterpret_cast<float3*>(grad_gaussian_scales.data_ptr<float>()),
        grad_gaussian_densities.data_ptr<float>(),
        reinterpret_cast<float3*>(grad_gaussian_z_scales.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_oob_positions.data_ptr<float>()));
}

// ============================================================================
