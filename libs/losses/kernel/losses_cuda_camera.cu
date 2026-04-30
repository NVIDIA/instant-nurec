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

// DISPATCH 1: Camera Losses
// ============================================================================

__global__ void camera_losses_forward_kernel(
    int B, int H, int W,
    float rgb_factor, float bg_factor,

    const int32_t* __restrict__ rgb_flags,
    const float3* __restrict__ rgb_gt,
    const float3* __restrict__ rgb_pred,
    const float* __restrict__ bg_pred,
    float* __restrict__ rgb_loss,
    float* __restrict__ bg_loss) {
    int ti = blockIdx.x * LOSSES_BLOCK_THREADS + threadIdx.x;
    int HW = H * W;
    int z  = ti / HW;
    if (z >= B)
        return;

    int x = ti % W;
    int y = (ti / W) % H;

    // Load flags once, reused for both losses
    int rgb_f = 0;
    if (rgb_factor > 0 || bg_factor > 0) {
        rgb_f = rgb_flags[((z * H + y) * W + x)];
    }

    // RGB L1 Loss
    if (rgb_factor >= 0) {
        float rgb_l = 0.0f;
        if (rgb_factor > 0) {
            if ((rgb_f & LOSSES_FLAG_RGB_LABEL) != 0 && (rgb_f & LOSSES_FLAG_INVALID) == 0) {
                int pixel = (z * H + y) * W + x;
                float3 p  = rgb_pred[pixel];
                float3 g  = rgb_gt[pixel];
                rgb_l     = (fabsf(p.x - g.x) + fabsf(p.y - g.y) + fabsf(p.z - g.z)) * rgb_factor;
            }
        }
        rgb_loss[z * HW + y * W + x] = rgb_l;
    }

    // Background MSE Loss
    if (bg_factor >= 0) {
        float bg_l = 0.0f;
        if (bg_factor > 0) {
            if ((rgb_f & LOSSES_FLAG_INVALID) == 0 && (rgb_f & LOSSES_FLAG_DIFIXED) == 0 && (rgb_f & LOSSES_FLAG_SYNTHETIC) == 0) {
                float p = fminf(fmaxf(bg_pred[ti], 0.0f), 1.0f);
                float g = ((rgb_f & LOSSES_FLAG_SKY_SEMANTIC) != 0) ? 0.0f : 1.0f;
                bg_l    = (p - g) * (p - g) * bg_factor;
            }
        }
        bg_loss[ti] = bg_l;
    }
}

__global__ void camera_losses_backward_kernel(
    int B, int H, int W,
    float rgb_factor, float bg_factor,

    const int32_t* __restrict__ rgb_flags,
    const float3* __restrict__ rgb_gt,
    const float3* __restrict__ rgb_pred,
    const float* __restrict__ bg_pred,
    const float* __restrict__ grad_rgb_loss,
    const float* __restrict__ grad_bg_loss,
    float3* __restrict__ grad_rgb_pred,
    float* __restrict__ grad_bg_pred) {
    int ti = blockIdx.x * LOSSES_BLOCK_THREADS + threadIdx.x;
    int HW = H * W;
    int z  = ti / HW;
    if (z >= B)
        return;

    int x = ti % W;
    int y = (ti / W) % H;

    int rgb_f = 0;
    if (rgb_factor > 0 || bg_factor > 0) {
        rgb_f = rgb_flags[((z * H + y) * W + x)];
    }

    // RGB L1 backward: d|x|/dx = sign(x)
    // NOTE: grad_rgb_pred must be zero-initialized by the caller — threads where
    // the flag check fails do not write, relying on the initial zero value.
    if (rgb_factor > 0) {
        if ((rgb_f & LOSSES_FLAG_RGB_LABEL) != 0 && (rgb_f & LOSSES_FLAG_INVALID) == 0) {
            int pixel            = (z * H + y) * W + x;
            float gl             = grad_rgb_loss[z * HW + y * W + x];
            float scale          = rgb_factor * gl;
            float3 p             = rgb_pred[pixel];
            float3 g             = rgb_gt[pixel];
            grad_rgb_pred[pixel] = make_float3(sign_f(p.x - g.x) * scale,
                                               sign_f(p.y - g.y) * scale,
                                               sign_f(p.z - g.z) * scale);
        }
    }

    // Background MSE backward: d(clamp(p,0,1)-g)^2/dp = 2*(clamp(p,0,1)-g) * d_clamp/dp
    // NOTE: grad_bg_pred must be zero-initialized by the caller.
    if (bg_factor > 0) {
        if ((rgb_f & LOSSES_FLAG_INVALID) == 0 && (rgb_f & LOSSES_FLAG_DIFIXED) == 0 && (rgb_f & LOSSES_FLAG_SYNTHETIC) == 0) {
            float p_raw = bg_pred[ti];
            float p     = fminf(fmaxf(p_raw, 0.0f), 1.0f);
            float g     = ((rgb_f & LOSSES_FLAG_SKY_SEMANTIC) != 0) ? 0.0f : 1.0f;
            // Gradient of clamp: 1 if p_raw in [0,1], 0 otherwise
            float clamp_grad = (p_raw >= 0.0f && p_raw <= 1.0f) ? 1.0f : 0.0f;
            grad_bg_pred[ti] = 2.0f * (p - g) * bg_factor * grad_bg_loss[ti] * clamp_grad;
        }
    }
}

void camera_losses_forward_cuda(
    int B_rgb, int H_rgb, int W_rgb,
    float rgb_factor, float bg_factor,

    torch::Tensor const rgb_flags,
    torch::Tensor const rgb_gt,
    torch::Tensor const rgb_pred,
    torch::Tensor const bg_pred,
    torch::Tensor const rgb_loss,
    torch::Tensor const bg_loss) {
    CHECK_INPUT(rgb_flags);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_pred);
    CHECK_INPUT(bg_pred);
    CHECK_INPUT(rgb_loss);
    CHECK_INPUT(bg_loss);

    at::cuda::CUDAGuard device_guard(rgb_pred.device());
    int N = B_rgb * H_rgb * W_rgb;
    dim3 threads(LOSSES_BLOCK_THREADS);
    dim3 blocks(div_round_up(N, LOSSES_BLOCK_THREADS));

    camera_losses_forward_kernel<<<blocks, threads>>>(
        B_rgb, H_rgb, W_rgb,
        rgb_factor, bg_factor,

        rgb_flags.data_ptr<int32_t>(),
        reinterpret_cast<const float3*>(rgb_gt.data_ptr<float>()),
        reinterpret_cast<const float3*>(rgb_pred.data_ptr<float>()),
        bg_pred.data_ptr<float>(),
        rgb_loss.data_ptr<float>(),
        bg_loss.data_ptr<float>());
}

void camera_losses_backward_cuda(
    int B_rgb, int H_rgb, int W_rgb,
    float rgb_factor, float bg_factor,

    torch::Tensor const rgb_flags,
    torch::Tensor const rgb_gt,
    torch::Tensor const rgb_pred,
    torch::Tensor const bg_pred,
    torch::Tensor const grad_rgb_loss,
    torch::Tensor const grad_bg_loss,
    torch::Tensor const grad_rgb_pred,
    torch::Tensor const grad_bg_pred) {
    CHECK_INPUT(rgb_flags);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_pred);
    CHECK_INPUT(bg_pred);
    CHECK_INPUT(grad_rgb_loss);
    CHECK_INPUT(grad_bg_loss);
    CHECK_INPUT(grad_rgb_pred);
    CHECK_INPUT(grad_bg_pred);

    at::cuda::CUDAGuard device_guard(rgb_pred.device());
    int N = B_rgb * H_rgb * W_rgb;
    dim3 threads(LOSSES_BLOCK_THREADS);
    dim3 blocks(div_round_up(N, LOSSES_BLOCK_THREADS));

    camera_losses_backward_kernel<<<blocks, threads>>>(
        B_rgb, H_rgb, W_rgb,
        rgb_factor, bg_factor,

        rgb_flags.data_ptr<int32_t>(),
        reinterpret_cast<const float3*>(rgb_gt.data_ptr<float>()),
        reinterpret_cast<const float3*>(rgb_pred.data_ptr<float>()),
        bg_pred.data_ptr<float>(),
        grad_rgb_loss.data_ptr<float>(),
        grad_bg_loss.data_ptr<float>(),
        reinterpret_cast<float3*>(grad_rgb_pred.data_ptr<float>()),
        grad_bg_pred.data_ptr<float>());
}

// ============================================================================
