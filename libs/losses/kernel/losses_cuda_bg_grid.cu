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

__global__ void background_grid_losses_forward_kernel(
    int B_tex, int D_tex, int H_tex, int W_tex, int C_tex,
    int B_gc, int D_gc, int H_gc, int W_gc,
    int B_gf, int D_gf, int H_gf, int W_gf,
    float bg_tex_factor,
    float grid_drift_camera_factor, float grid_drift_frame_factor,
    float grid_camera_tv_factor, float grid_frame_tv_factor,
    const float* __restrict__ bg_tex,
    const float* __restrict__ grids_camera,
    const float* __restrict__ grids_frame,
    float* __restrict__ bg_tex_loss,
    float* __restrict__ grids_drift_loss,
    float* __restrict__ grid_camera_tv_loss,
    float* __restrict__ grid_frame_tv_loss) {
    int ti = blockIdx.x * LOSSES_BLOCK_THREADS + threadIdx.x;

    int numel_gc  = B_gc * D_gc * H_gc * W_gc;
    int numel_gf  = B_gf * D_gf * H_gf * W_gf;
    int numel_tex = B_tex * D_tex * H_tex * W_tex * C_tex;

    // Sky env map TV loss
    if (ti < numel_tex) {
        float tv = 0.0f;
        if (bg_tex_factor > 0) {
            int WC  = W_tex * C_tex;
            int HWC = H_tex * WC;
            int r   = ti;
            int bd  = r / HWC;
            r -= bd * HWC;
            int h = r / WC;
            r -= h * WC;
            int w = r / C_tex;
            r -= w * C_tex;
            int c = r;

            float center = bg_tex[ti]; // [bd, h, w, c] contiguous
            float delta;
            float norm;

            // Depth dimension (for CUBEMAP: D_tex == 6)
            if (D_tex > 1) {
                int b = bd / D_tex;
                int d = bd % D_tex;
                if (d < D_tex - 1) {
                    int neighbor_idx = ((b * D_tex + d + 1) * H_tex + h) * WC + w * C_tex + c;
                    delta            = center - bg_tex[neighbor_idx];
                    norm             = 1.0f / ((float)B_tex * (D_tex - 1) * H_tex * W_tex * C_tex);
                    tv += delta * delta * norm;
                }
            }

            // Height dimension: ternary avoids warp divergence at cubemap boundary
            if (H_tex > 1) {
                int h_next       = (D_tex == 1) ? (h + 1) % H_tex : min(h + 1, H_tex - 1);
                int neighbor_idx = bd * HWC + h_next * WC + w * C_tex + c;
                delta            = (D_tex == 1 || h < H_tex - 1) ? center - bg_tex[neighbor_idx] : 0.0f;
                norm             = 1.0f / ((float)B_tex * D_tex * (H_tex - 1) * W_tex * C_tex);
                tv += delta * delta * norm;
            }

            // Width dimension: ternary avoids warp divergence at cubemap boundary
            if (W_tex > 1) {
                int w_next       = (D_tex == 1) ? (w + 1) % W_tex : min(w + 1, W_tex - 1);
                int neighbor_idx = bd * HWC + h * WC + w_next * C_tex + c;
                delta            = (D_tex == 1 || w < W_tex - 1) ? center - bg_tex[neighbor_idx] : 0.0f;
                norm             = 1.0f / ((float)B_tex * D_tex * H_tex * (W_tex - 1) * C_tex);
                tv += delta * delta * norm;
            }
        }
        bg_tex_loss[ti] = tv;
    }

    // Grid camera drift loss
    if (grid_drift_camera_factor > 0 && ti < numel_gc) {
        grids_drift_loss[ti] = compute_grid_drift_loss(ti, D_gc, H_gc, W_gc, grid_drift_camera_factor, grids_camera);
    } else if (ti < numel_gc) {
        grids_drift_loss[ti] = 0.0f;
    }

    // Grid camera TV loss
    if (grid_camera_tv_factor > 0 && ti < numel_gc) {
        grid_camera_tv_loss[ti] = compute_grid_total_variation_spatial(ti, D_gc, H_gc, W_gc, grid_camera_tv_factor, grids_camera);
    } else if (ti < numel_gc) {
        grid_camera_tv_loss[ti] = 0.0f;
    }

    // Grid frame drift loss
    if (grid_drift_frame_factor > 0 && ti < numel_gf) {
        grids_drift_loss[numel_gc + ti] = compute_grid_drift_loss(ti, D_gf, H_gf, W_gf, grid_drift_frame_factor, grids_frame);
    } else if (ti < numel_gf) {
        grids_drift_loss[numel_gc + ti] = 0.0f;
    }

    // Grid frame TV loss
    if (grid_frame_tv_factor > 0 && ti < numel_gf) {
        grid_frame_tv_loss[ti] = compute_grid_total_variation_spatial(ti, D_gf, H_gf, W_gf, grid_frame_tv_factor, grids_frame);
    } else if (ti < numel_gf) {
        grid_frame_tv_loss[ti] = 0.0f;
    }
}

// ============================================================================
// DISPATCH 4 BACKWARD
// ============================================================================

__global__ void background_grid_losses_backward_kernel(
    int B_tex, int D_tex, int H_tex, int W_tex, int C_tex,
    int B_gc, int D_gc, int H_gc, int W_gc,
    int B_gf, int D_gf, int H_gf, int W_gf,
    float bg_tex_factor,
    float grid_drift_camera_factor, float grid_drift_frame_factor,
    float grid_camera_tv_factor, float grid_frame_tv_factor,
    const float* __restrict__ bg_tex,
    const float* __restrict__ grids_camera,
    const float* __restrict__ grids_frame,
    const float* __restrict__ grad_bg_tex_loss,
    const float* __restrict__ grad_grids_drift_loss,
    const float* __restrict__ grad_grid_camera_tv_loss,
    const float* __restrict__ grad_grid_frame_tv_loss,
    float* __restrict__ grad_bg_tex,
    float* __restrict__ grad_grids_camera,
    float* __restrict__ grad_grids_frame) {
    int ti = blockIdx.x * LOSSES_BLOCK_THREADS + threadIdx.x;

    int numel_gc  = B_gc * D_gc * H_gc * W_gc;
    int numel_gf  = B_gf * D_gf * H_gf * W_gf;
    int numel_tex = B_tex * D_tex * H_tex * W_tex * C_tex;

    // Sky env map TV backward
    if (ti < numel_tex && bg_tex_factor > 0) {
        int WC  = W_tex * C_tex;
        int HWC = H_tex * WC;
        int r   = ti;
        int bd  = r / HWC;
        r -= bd * HWC;
        int h = r / WC;
        r -= h * WC;
        int w = r / C_tex;
        r -= w * C_tex;
        int c = r;

        float center      = bg_tex[ti];
        float gl          = grad_bg_tex_loss[ti];
        float grad_center = 0.0f;

        // Depth dimension
        if (D_tex > 1) {
            int b = bd / D_tex;
            int d = bd % D_tex;
            if (d < D_tex - 1) {
                int neighbor_idx = ((b * D_tex + d + 1) * H_tex + h) * WC + w * C_tex + c;
                float neighbor   = bg_tex[neighbor_idx];
                float norm       = 1.0f / ((float)B_tex * (D_tex - 1) * H_tex * W_tex * C_tex);
                float grad_val   = 2.0f * (center - neighbor) * norm * gl;
                grad_center += grad_val;
                atomicAdd(&grad_bg_tex[neighbor_idx], -grad_val);
            }
        }

        // Height dimension: ternary avoids warp divergence at cubemap boundary
        if (H_tex > 1) {
            int h_next        = (D_tex == 1) ? (h + 1) % H_tex : min(h + 1, H_tex - 1);
            int neighbor_idx  = bd * HWC + h_next * WC + w * C_tex + c;
            bool has_neighbor = (D_tex == 1 || h < H_tex - 1);
            float neighbor    = bg_tex[neighbor_idx];
            float norm        = 1.0f / ((float)B_tex * D_tex * (H_tex - 1) * W_tex * C_tex);
            float grad_val    = has_neighbor ? 2.0f * (center - neighbor) * norm * gl : 0.0f;
            grad_center += grad_val;
            if (has_neighbor) {
                atomicAdd(&grad_bg_tex[neighbor_idx], -grad_val);
            }
        }

        // Width dimension: ternary avoids warp divergence at cubemap boundary
        if (W_tex > 1) {
            int w_next        = (D_tex == 1) ? (w + 1) % W_tex : min(w + 1, W_tex - 1);
            int neighbor_idx  = bd * HWC + h * WC + w_next * C_tex + c;
            bool has_neighbor = (D_tex == 1 || w < W_tex - 1);
            float neighbor    = bg_tex[neighbor_idx];
            float norm        = 1.0f / ((float)B_tex * D_tex * H_tex * (W_tex - 1) * C_tex);
            float grad_val    = has_neighbor ? 2.0f * (center - neighbor) * norm * gl : 0.0f;
            grad_center += grad_val;
            if (has_neighbor) {
                atomicAdd(&grad_bg_tex[neighbor_idx], -grad_val);
            }
        }

        // Direct write for center element (no atomicAdd needed since each thread owns its ti)
        atomicAdd(&grad_bg_tex[ti], grad_center);
    }

    // Grid camera: fused drift + TV backward (shared index decomposition)
    if (ti < numel_gc && (grid_drift_camera_factor > 0 || grid_camera_tv_factor > 0)) {
        float grad_drift = (grid_drift_camera_factor > 0) ? grad_grids_drift_loss[ti] : 0.0f;
        float grad_tv    = (grid_camera_tv_factor > 0) ? grad_grid_camera_tv_loss[ti] : 0.0f;
        compute_grid_drift_and_tv_backward(
            ti, D_gc, H_gc, W_gc,
            grid_drift_camera_factor, grid_camera_tv_factor,
            grids_camera, grad_drift, grad_tv, grad_grids_camera);
    }

    // Grid frame: fused drift + TV backward
    if (ti < numel_gf && (grid_drift_frame_factor > 0 || grid_frame_tv_factor > 0)) {
        float grad_drift = (grid_drift_frame_factor > 0) ? grad_grids_drift_loss[numel_gc + ti] : 0.0f;
        float grad_tv    = (grid_frame_tv_factor > 0) ? grad_grid_frame_tv_loss[ti] : 0.0f;
        compute_grid_drift_and_tv_backward(
            ti, D_gf, H_gf, W_gf,
            grid_drift_frame_factor, grid_frame_tv_factor,
            grids_frame, grad_drift, grad_tv, grad_grids_frame);
    }
}

void background_grid_losses_forward_cuda(
    int B_tex, int D_tex, int H_tex, int W_tex, int C_tex,
    int B_gc, int D_gc, int H_gc, int W_gc,
    int B_gf, int D_gf, int H_gf, int W_gf,
    float bg_tex_factor,
    float grid_drift_camera_factor, float grid_drift_frame_factor,
    float grid_camera_tv_factor, float grid_frame_tv_factor,
    torch::Tensor const bg_tex,
    torch::Tensor const grids_camera,
    torch::Tensor const grids_frame,
    torch::Tensor const bg_tex_loss,
    torch::Tensor const grids_drift_loss,
    torch::Tensor const grid_camera_tv_loss,
    torch::Tensor const grid_frame_tv_loss) {
    CHECK_INPUT(bg_tex);
    CHECK_INPUT(grids_camera);
    CHECK_INPUT(grids_frame);
    CHECK_INPUT(bg_tex_loss);
    CHECK_INPUT(grids_drift_loss);
    CHECK_INPUT(grid_camera_tv_loss);
    CHECK_INPUT(grid_frame_tv_loss);

    at::cuda::CUDAGuard device_guard(bg_tex.device());
    int numel_tex = B_tex * D_tex * H_tex * W_tex * C_tex;
    int numel_gc  = B_gc * D_gc * H_gc * W_gc;
    int numel_gf  = B_gf * D_gf * H_gf * W_gf;
    int N         = max(numel_tex, max(numel_gc, numel_gf));
    if (N == 0)
        return;
    dim3 threads(LOSSES_BLOCK_THREADS);
    dim3 blocks(div_round_up(N, LOSSES_BLOCK_THREADS));

    background_grid_losses_forward_kernel<<<blocks, threads>>>(
        B_tex, D_tex, H_tex, W_tex, C_tex,
        B_gc, D_gc, H_gc, W_gc,
        B_gf, D_gf, H_gf, W_gf,
        bg_tex_factor,
        grid_drift_camera_factor, grid_drift_frame_factor,
        grid_camera_tv_factor, grid_frame_tv_factor,
        bg_tex.data_ptr<float>(),
        grids_camera.data_ptr<float>(),
        grids_frame.data_ptr<float>(),
        bg_tex_loss.data_ptr<float>(),
        grids_drift_loss.data_ptr<float>(),
        grid_camera_tv_loss.data_ptr<float>(),
        grid_frame_tv_loss.data_ptr<float>());
}

void background_grid_losses_backward_cuda(
    int B_tex, int D_tex, int H_tex, int W_tex, int C_tex,
    int B_gc, int D_gc, int H_gc, int W_gc,
    int B_gf, int D_gf, int H_gf, int W_gf,
    float bg_tex_factor,
    float grid_drift_camera_factor, float grid_drift_frame_factor,
    float grid_camera_tv_factor, float grid_frame_tv_factor,
    torch::Tensor const bg_tex,
    torch::Tensor const grids_camera,
    torch::Tensor const grids_frame,
    torch::Tensor const grad_bg_tex_loss,
    torch::Tensor const grad_grids_drift_loss,
    torch::Tensor const grad_grid_camera_tv_loss,
    torch::Tensor const grad_grid_frame_tv_loss,
    torch::Tensor const grad_bg_tex,
    torch::Tensor const grad_grids_camera,
    torch::Tensor const grad_grids_frame) {
    CHECK_INPUT(bg_tex);
    CHECK_INPUT(grids_camera);
    CHECK_INPUT(grids_frame);
    CHECK_INPUT(grad_bg_tex_loss);
    CHECK_INPUT(grad_grids_drift_loss);
    CHECK_INPUT(grad_grid_camera_tv_loss);
    CHECK_INPUT(grad_grid_frame_tv_loss);
    CHECK_INPUT(grad_bg_tex);
    CHECK_INPUT(grad_grids_camera);
    CHECK_INPUT(grad_grids_frame);

    at::cuda::CUDAGuard device_guard(bg_tex.device());
    int numel_tex = B_tex * D_tex * H_tex * W_tex * C_tex;
    int numel_gc  = B_gc * D_gc * H_gc * W_gc;
    int numel_gf  = B_gf * D_gf * H_gf * W_gf;
    int N         = max(numel_tex, max(numel_gc, numel_gf));
    if (N == 0)
        return;
    dim3 threads(LOSSES_BLOCK_THREADS);
    dim3 blocks(div_round_up(N, LOSSES_BLOCK_THREADS));

    background_grid_losses_backward_kernel<<<blocks, threads>>>(
        B_tex, D_tex, H_tex, W_tex, C_tex,
        B_gc, D_gc, H_gc, W_gc,
        B_gf, D_gf, H_gf, W_gf,
        bg_tex_factor,
        grid_drift_camera_factor, grid_drift_frame_factor,
        grid_camera_tv_factor, grid_frame_tv_factor,
        bg_tex.data_ptr<float>(),
        grids_camera.data_ptr<float>(),
        grids_frame.data_ptr<float>(),
        grad_bg_tex_loss.data_ptr<float>(),
        grad_grids_drift_loss.data_ptr<float>(),
        grad_grid_camera_tv_loss.data_ptr<float>(),
        grad_grid_frame_tv_loss.data_ptr<float>(),
        grad_bg_tex.data_ptr<float>(),
        grad_grids_camera.data_ptr<float>(),
        grad_grids_frame.data_ptr<float>());
}
