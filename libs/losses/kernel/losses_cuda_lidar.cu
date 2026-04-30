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

// DISPATCH 2: LiDAR Losses
// ============================================================================

__global__ void lidar_losses_forward_kernel(
    int B, int H, int W,
    float lidar_factor, float bg_lidar_factor, float intensity_factor, float raydrop_factor,

    const int32_t* __restrict__ lidar_flags,
    const float* __restrict__ lidar_gt,
    const float* __restrict__ intensity_gt,
    const float* __restrict__ raydrop_gt,
    const float* __restrict__ lidar_pred,
    const float* __restrict__ bg_lidar_pred,
    const float* __restrict__ intensity_pred,
    const float* __restrict__ raydrop_pred,
    float* __restrict__ lidar_loss,
    float* __restrict__ bg_lidar_loss,
    float* __restrict__ intensity_loss,
    float* __restrict__ raydrop_loss) {
    int ti = blockIdx.x * LOSSES_BLOCK_THREADS + threadIdx.x;
    int HW = H * W;
    int z  = ti / HW;
    if (z >= B)
        return;

    int x = ti % W;
    int y = (ti / W) % H;

    // Load flags once, reused for all four losses
    int lidar_f = 0;
    if (lidar_factor > 0 || bg_lidar_factor > 0 || intensity_factor > 0 || raydrop_factor > 0) {
        lidar_f = lidar_flags[((z * H + y) * W + x)];
    }

    int idx_4d   = ((z * H + y) * W + x); // Linear index for [B,H,W] shaped outputs
    int idx_bhw1 = idx_4d;                // Same as idx_4d since last dim is 1

    // LiDAR L1 Loss
    if (lidar_factor >= 0) {
        float lidar_l = 0.0f;
        if (lidar_factor > 0) {
            if ((lidar_f & LOSSES_FLAG_INVALID) == 0 && (lidar_f & LOSSES_FLAG_DROPPED) == 0) {
                float p = lidar_pred[idx_bhw1];
                float g = lidar_gt[idx_bhw1];
                lidar_l = fabsf(p - g) * lidar_factor;
            }
        }
        lidar_loss[idx_4d] = lidar_l;
    }

    // Background LiDAR MSE Loss
    if (bg_lidar_factor >= 0) {
        float bg_lidar_l = 0.0f;
        if (bg_lidar_factor > 0) {
            if ((lidar_f & LOSSES_FLAG_INVALID) == 0 && (lidar_f & LOSSES_FLAG_DROPPED) == 0) {
                float p    = fminf(fmaxf(bg_lidar_pred[ti], 0.0f), 1.0f);
                float g    = ((lidar_f & LOSSES_FLAG_SKY_SEMANTIC) != 0) ? 0.0f : 1.0f;
                bg_lidar_l = (p - g) * (p - g) * bg_lidar_factor;
            }
        }
        bg_lidar_loss[ti] = bg_lidar_l;
    }

    // Intensity MSE Loss
    if (intensity_factor >= 0) {
        float intensity_l = 0.0f;
        if (intensity_factor > 0) {
            if ((lidar_f & LOSSES_FLAG_INVALID) == 0 && (lidar_f & LOSSES_FLAG_DROPPED) == 0) {
                float p     = intensity_pred[idx_bhw1];
                float g     = intensity_gt[idx_bhw1];
                intensity_l = (p - g) * (p - g) * intensity_factor;
            }
        }
        intensity_loss[idx_4d] = intensity_l;
    }

    // Raydrop MSE Loss (only checks LOSSES_FLAG_INVALID, not LOSSES_FLAG_DROPPED)
    if (raydrop_factor >= 0) {
        float raydrop_l = 0.0f;
        if (raydrop_factor > 0) {
            if ((lidar_f & LOSSES_FLAG_INVALID) == 0) {
                float p   = raydrop_pred[idx_bhw1];
                float g   = raydrop_gt[idx_bhw1];
                raydrop_l = (p - g) * (p - g) * raydrop_factor;
            }
        }
        raydrop_loss[idx_4d] = raydrop_l;
    }
}

__global__ void lidar_losses_backward_kernel(
    int B, int H, int W,
    float lidar_factor, float bg_lidar_factor, float intensity_factor, float raydrop_factor,

    const int32_t* __restrict__ lidar_flags,
    const float* __restrict__ lidar_gt,
    const float* __restrict__ intensity_gt,
    const float* __restrict__ raydrop_gt,
    const float* __restrict__ lidar_pred,
    const float* __restrict__ bg_lidar_pred,
    const float* __restrict__ intensity_pred,
    const float* __restrict__ raydrop_pred,
    const float* __restrict__ grad_lidar_loss,
    const float* __restrict__ grad_bg_lidar_loss,
    const float* __restrict__ grad_intensity_loss,
    const float* __restrict__ grad_raydrop_loss,
    float* __restrict__ grad_lidar_pred,
    float* __restrict__ grad_bg_lidar_pred,
    float* __restrict__ grad_intensity_pred,
    float* __restrict__ grad_raydrop_pred) {
    int ti = blockIdx.x * LOSSES_BLOCK_THREADS + threadIdx.x;
    int HW = H * W;
    int z  = ti / HW;
    if (z >= B)
        return;

    int x = ti % W;
    int y = (ti / W) % H;

    int lidar_f = 0;
    if (lidar_factor > 0 || bg_lidar_factor > 0 || intensity_factor > 0 || raydrop_factor > 0) {
        lidar_f = lidar_flags[((z * H + y) * W + x)];
    }

    int idx_4d   = ((z * H + y) * W + x);
    int idx_bhw1 = idx_4d;

    // LiDAR L1 backward
    // NOTE: all grad_*_pred buffers must be zero-initialized by the caller —
    // threads where flag checks fail do not write, relying on initial zero values.
    if (lidar_factor > 0) {
        if ((lidar_f & LOSSES_FLAG_INVALID) == 0 && (lidar_f & LOSSES_FLAG_DROPPED) == 0) {
            float p                   = lidar_pred[idx_bhw1];
            float g                   = lidar_gt[idx_bhw1];
            grad_lidar_pred[idx_bhw1] = sign_f(p - g) * lidar_factor * grad_lidar_loss[idx_4d];
        }
    }

    // Background LiDAR MSE backward
    if (bg_lidar_factor > 0) {
        if ((lidar_f & LOSSES_FLAG_INVALID) == 0 && (lidar_f & LOSSES_FLAG_DROPPED) == 0) {
            float p_raw            = bg_lidar_pred[ti];
            float p                = fminf(fmaxf(p_raw, 0.0f), 1.0f);
            float g                = ((lidar_f & LOSSES_FLAG_SKY_SEMANTIC) != 0) ? 0.0f : 1.0f;
            float clamp_grad       = (p_raw >= 0.0f && p_raw <= 1.0f) ? 1.0f : 0.0f;
            grad_bg_lidar_pred[ti] = 2.0f * (p - g) * bg_lidar_factor * grad_bg_lidar_loss[ti] * clamp_grad;
        }
    }

    // Intensity MSE backward
    if (intensity_factor > 0) {
        if ((lidar_f & LOSSES_FLAG_INVALID) == 0 && (lidar_f & LOSSES_FLAG_DROPPED) == 0) {
            float p                       = intensity_pred[idx_bhw1];
            float g                       = intensity_gt[idx_bhw1];
            grad_intensity_pred[idx_bhw1] = 2.0f * (p - g) * intensity_factor * grad_intensity_loss[idx_4d];
        }
    }

    // Raydrop MSE backward (only checks LOSSES_FLAG_INVALID, not LOSSES_FLAG_DROPPED)
    if (raydrop_factor > 0) {
        if ((lidar_f & LOSSES_FLAG_INVALID) == 0) {
            float p                     = raydrop_pred[idx_bhw1];
            float g                     = raydrop_gt[idx_bhw1];
            grad_raydrop_pred[idx_bhw1] = 2.0f * (p - g) * raydrop_factor * grad_raydrop_loss[idx_4d];
        }
    }
}

void lidar_losses_forward_cuda(
    int B_lidar, int H_lidar, int W_lidar,
    float lidar_factor, float bg_lidar_factor, float intensity_factor, float raydrop_factor,

    torch::Tensor const lidar_flags,
    torch::Tensor const lidar_gt,
    torch::Tensor const intensity_gt,
    torch::Tensor const raydrop_gt,
    torch::Tensor const lidar_pred,
    torch::Tensor const bg_lidar_pred,
    torch::Tensor const intensity_pred,
    torch::Tensor const raydrop_pred,
    torch::Tensor const lidar_loss,
    torch::Tensor const bg_lidar_loss,
    torch::Tensor const intensity_loss,
    torch::Tensor const raydrop_loss) {
    CHECK_INPUT(lidar_flags);
    CHECK_INPUT(lidar_gt);
    CHECK_INPUT(intensity_gt);
    CHECK_INPUT(raydrop_gt);
    CHECK_INPUT(lidar_pred);
    CHECK_INPUT(bg_lidar_pred);
    CHECK_INPUT(intensity_pred);
    CHECK_INPUT(raydrop_pred);
    CHECK_INPUT(lidar_loss);
    CHECK_INPUT(bg_lidar_loss);
    CHECK_INPUT(intensity_loss);
    CHECK_INPUT(raydrop_loss);

    at::cuda::CUDAGuard device_guard(lidar_pred.device());
    int N = B_lidar * H_lidar * W_lidar;
    dim3 threads(LOSSES_BLOCK_THREADS);
    dim3 blocks(div_round_up(N, LOSSES_BLOCK_THREADS));

    lidar_losses_forward_kernel<<<blocks, threads>>>(
        B_lidar, H_lidar, W_lidar,
        lidar_factor, bg_lidar_factor, intensity_factor, raydrop_factor,

        lidar_flags.data_ptr<int32_t>(),
        lidar_gt.data_ptr<float>(),
        intensity_gt.data_ptr<float>(),
        raydrop_gt.data_ptr<float>(),
        lidar_pred.data_ptr<float>(),
        bg_lidar_pred.data_ptr<float>(),
        intensity_pred.data_ptr<float>(),
        raydrop_pred.data_ptr<float>(),
        lidar_loss.data_ptr<float>(),
        bg_lidar_loss.data_ptr<float>(),
        intensity_loss.data_ptr<float>(),
        raydrop_loss.data_ptr<float>());
}

void lidar_losses_backward_cuda(
    int B_lidar, int H_lidar, int W_lidar,
    float lidar_factor, float bg_lidar_factor, float intensity_factor, float raydrop_factor,

    torch::Tensor const lidar_flags,
    torch::Tensor const lidar_gt,
    torch::Tensor const intensity_gt,
    torch::Tensor const raydrop_gt,
    torch::Tensor const lidar_pred,
    torch::Tensor const bg_lidar_pred,
    torch::Tensor const intensity_pred,
    torch::Tensor const raydrop_pred,
    torch::Tensor const grad_lidar_loss,
    torch::Tensor const grad_bg_lidar_loss,
    torch::Tensor const grad_intensity_loss,
    torch::Tensor const grad_raydrop_loss,
    torch::Tensor const grad_lidar_pred,
    torch::Tensor const grad_bg_lidar_pred,
    torch::Tensor const grad_intensity_pred,
    torch::Tensor const grad_raydrop_pred) {
    CHECK_INPUT(lidar_flags);
    CHECK_INPUT(lidar_gt);
    CHECK_INPUT(intensity_gt);
    CHECK_INPUT(raydrop_gt);
    CHECK_INPUT(lidar_pred);
    CHECK_INPUT(bg_lidar_pred);
    CHECK_INPUT(intensity_pred);
    CHECK_INPUT(raydrop_pred);
    CHECK_INPUT(grad_lidar_loss);
    CHECK_INPUT(grad_bg_lidar_loss);
    CHECK_INPUT(grad_intensity_loss);
    CHECK_INPUT(grad_raydrop_loss);
    CHECK_INPUT(grad_lidar_pred);
    CHECK_INPUT(grad_bg_lidar_pred);
    CHECK_INPUT(grad_intensity_pred);
    CHECK_INPUT(grad_raydrop_pred);

    at::cuda::CUDAGuard device_guard(lidar_pred.device());
    int N = B_lidar * H_lidar * W_lidar;
    dim3 threads(LOSSES_BLOCK_THREADS);
    dim3 blocks(div_round_up(N, LOSSES_BLOCK_THREADS));

    lidar_losses_backward_kernel<<<blocks, threads>>>(
        B_lidar, H_lidar, W_lidar,
        lidar_factor, bg_lidar_factor, intensity_factor, raydrop_factor,

        lidar_flags.data_ptr<int32_t>(),
        lidar_gt.data_ptr<float>(),
        intensity_gt.data_ptr<float>(),
        raydrop_gt.data_ptr<float>(),
        lidar_pred.data_ptr<float>(),
        bg_lidar_pred.data_ptr<float>(),
        intensity_pred.data_ptr<float>(),
        raydrop_pred.data_ptr<float>(),
        grad_lidar_loss.data_ptr<float>(),
        grad_bg_lidar_loss.data_ptr<float>(),
        grad_intensity_loss.data_ptr<float>(),
        grad_raydrop_loss.data_ptr<float>(),
        grad_lidar_pred.data_ptr<float>(),
        grad_bg_lidar_pred.data_ptr<float>(),
        grad_intensity_pred.data_ptr<float>(),
        grad_raydrop_pred.data_ptr<float>());
}

// ============================================================================
