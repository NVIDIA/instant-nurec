// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Shared constants and device functions for CUDA loss kernels.
// Included by individual per-dispatch .cu files.
//
// Constants have #ifndef defaults that can be overridden by including a generated
// constants_cuda.h header before this file. When building from NRE, Bazel
// generates that header from constants.py (same source of truth as Slang).
// The defaults below are stubs for standalone builds.

#pragma once

// Per-pixel/point flag bits for fused camera/lidar loss kernels.
// Defaults are stubs for standalone builds.
// Override via -D compiler flags when building from a host project
// (e.g., NRE passes its RayFlags values via Bazel copts from constants.py).
#ifndef LOSSES_FLAG_RGB_LABEL
#define LOSSES_FLAG_RGB_LABEL 1
#endif
#ifndef LOSSES_FLAG_DROPPED
#define LOSSES_FLAG_DROPPED 64
#endif
#ifndef LOSSES_FLAG_INVALID
#define LOSSES_FLAG_INVALID 256
#endif
#ifndef LOSSES_FLAG_SKY_SEMANTIC
#define LOSSES_FLAG_SKY_SEMANTIC 4
#endif
#ifndef LOSSES_FLAG_DIFIXED
#define LOSSES_FLAG_DIFIXED 512
#endif
#ifndef LOSSES_FLAG_SYNTHETIC
#define LOSSES_FLAG_SYNTHETIC 1024
#endif

// Bilateral grid affine transformation matrix dimensions.
#ifndef LOSSES_GRID_NUM_ROWS
#define LOSSES_GRID_NUM_ROWS 3
#endif
#ifndef LOSSES_GRID_NUM_COLS
#define LOSSES_GRID_NUM_COLS 4
#endif
#ifndef LOSSES_GRID_NUM_CHANNELS
#define LOSSES_GRID_NUM_CHANNELS (LOSSES_GRID_NUM_ROWS * LOSSES_GRID_NUM_COLS)
#endif

// Block dimensions for loss kernels.
#ifndef LOSSES_BLOCK_THREADS
#define LOSSES_BLOCK_THREADS 256
#endif

// sign function returning 0 for 0 (matching abs() subgradient at zero)
__device__ __forceinline__ float sign_f(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

// NOTE: Use __device__ __forceinline__ (not static __device__) for helpers.
// static __device__ prevents the compiler from optimizing across call sites
// in the same kernel, losing performance vs __forceinline__ which guarantees
// the function body is inlined at every call site.

// TODO: Fuse compute_grid_drift_loss + compute_grid_total_variation_spatial
// into a single helper function. Both iterate over the same 12 channel
// positions (GRID_NUM_ROWS * GRID_NUM_COLS == GRID_NUM_CHANNELS), reading
// the same center grid values at each position. A fused function would share
// the 12 center reads and the (b,d,h,w) index decomposition, eliminating
// ~12 redundant global memory reads per thread.

// Helper: Compute grid drift loss for a single element
// grid is [B*LOSSES_GRID_NUM_ROWS*LOSSES_GRID_NUM_COLS, D, H, W], indexed as [bij, d, h, w]
__device__ __forceinline__ float compute_grid_drift_loss(
    int ti,
    int D_grid, int H_grid, int W_grid,
    float grid_drift_factor,
    const float* __restrict__ grid) {
    float grid_drift_loss = 0.0f;
    if (grid_drift_factor > 0) {
        int HW  = H_grid * W_grid;
        int DHW = D_grid * HW;

        int r = ti;
        int b = r / DHW;
        r -= b * DHW;
        int d = r / HW;
        r -= d * HW;
        int h = r / W_grid;
        r -= h * W_grid;
        int w = r;

        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < LOSSES_GRID_NUM_ROWS; ++i) {
#pragma unroll
            for (int j = 0; j < LOSSES_GRID_NUM_COLS; ++j) {
                int bij        = b * LOSSES_GRID_NUM_ROWS * LOSSES_GRID_NUM_COLS + i * LOSSES_GRID_NUM_COLS + j;
                float mij      = grid[((bij * D_grid + d) * H_grid + h) * W_grid + w];
                float identity = (i == j) ? 1.0f : 0.0f;
                float diff     = mij - identity;
                sum += diff * diff;
            }
        }

        grid_drift_loss = sqrtf(sum) * grid_drift_factor;
    }
    return grid_drift_loss;
}

// Helper: Compute grid total variation spatial loss for a single element
__device__ __forceinline__ float compute_grid_total_variation_spatial(
    int ti,
    int D_grid, int H_grid, int W_grid,
    float tv_spatial_factor,
    const float* __restrict__ grid) {
    float tv_loss = 0.0f;
    if (tv_spatial_factor > 0) {
        int HW  = H_grid * W_grid;
        int DHW = D_grid * HW;

        int r = ti;
        int b = r / DHW;
        r -= b * DHW;
        int d = r / HW;
        r -= d * HW;
        int h = r / W_grid;
        r -= h * W_grid;
        int w = r;

#pragma unroll
        for (int c = 0; c < LOSSES_GRID_NUM_CHANNELS; ++c) {
            int bc  = b * LOSSES_GRID_NUM_CHANNELS + c;
            float v = grid[((bc * D_grid + d) * H_grid + h) * W_grid + w];

            if (d + 1 < D_grid) {
                float u = grid[((bc * D_grid + (d + 1)) * H_grid + h) * W_grid + w];
                tv_loss += (u - v) * (u - v) * D_grid / (D_grid - 1) * tv_spatial_factor;
            }

            if (h + 1 < H_grid) {
                float u = grid[((bc * D_grid + d) * H_grid + (h + 1)) * W_grid + w];
                tv_loss += (u - v) * (u - v) * H_grid / (H_grid - 1) * tv_spatial_factor;
            }

            if (w + 1 < W_grid) {
                float u = grid[((bc * D_grid + d) * H_grid + h) * W_grid + (w + 1)];
                tv_loss += (u - v) * (u - v) * W_grid / (W_grid - 1) * tv_spatial_factor;
            }
        }
        tv_loss /= (float)LOSSES_GRID_NUM_CHANNELS;
    }
    return tv_loss;
}

// Helper: Fused backward for grid drift + TV spatial loss for a single element.
// Combines both operations to share index decomposition and grid reads.
// Key optimizations:
//   1. Grid drift: cache 12 matrix entries in registers (single read), reuse for S and grad
//   2. Grid TV: accumulate self-gradient in register, single atomicAdd per channel instead of 3
//   3. Fused drift+TV: shared (b,d,h,w) decomposition
__device__ __forceinline__ void compute_grid_drift_and_tv_backward(
    int ti,
    int D_grid, int H_grid, int W_grid,
    float grid_drift_factor,
    float tv_spatial_factor,
    const float* __restrict__ grid,
    float grad_drift_loss_val,
    float grad_tv_loss_val,
    float* __restrict__ grad_grid) {
    int HW  = H_grid * W_grid;
    int DHW = D_grid * HW;

    int r = ti;
    int b = r / DHW;
    r -= b * DHW;
    int d = r / HW;
    r -= d * HW;
    int h = r / W_grid;
    r -= h * W_grid;
    int w = r;

    // --- Grid drift backward ---
    // Cache all 12 matrix entries in registers (single read from global memory)
    if (grid_drift_factor > 0) {
        float cached_mij[LOSSES_GRID_NUM_ROWS * LOSSES_GRID_NUM_COLS];
        int cached_idx[LOSSES_GRID_NUM_ROWS * LOSSES_GRID_NUM_COLS];
        float S = 0.0f;

#pragma unroll
        for (int i = 0; i < LOSSES_GRID_NUM_ROWS; ++i) {
#pragma unroll
            for (int j = 0; j < LOSSES_GRID_NUM_COLS; ++j) {
                int k          = i * LOSSES_GRID_NUM_COLS + j;
                int bij        = b * LOSSES_GRID_NUM_ROWS * LOSSES_GRID_NUM_COLS + k;
                int idx        = ((bij * D_grid + d) * H_grid + h) * W_grid + w;
                float mij      = grid[idx];
                cached_mij[k]  = mij;
                cached_idx[k]  = idx;
                float identity = (i == j) ? 1.0f : 0.0f;
                float diff     = mij - identity;
                S += diff * diff;
            }
        }

        if (S >= 1e-12f) {
            float scale = grid_drift_factor * grad_drift_loss_val / sqrtf(S);
#pragma unroll
            for (int i = 0; i < LOSSES_GRID_NUM_ROWS; ++i) {
#pragma unroll
                for (int j = 0; j < LOSSES_GRID_NUM_COLS; ++j) {
                    int k          = i * LOSSES_GRID_NUM_COLS + j;
                    float identity = (i == j) ? 1.0f : 0.0f;
                    atomicAdd(&grad_grid[cached_idx[k]], (cached_mij[k] - identity) * scale);
                }
            }
        }
    }

    // --- Grid TV spatial backward ---
    // For each of 12 channels: load center value once, then check 3 neighbor directions.
    // Accumulate self-gradient in register, write once via atomicAdd per channel.
    if (tv_spatial_factor > 0) {
        float grad_tv_base = 2.0f * tv_spatial_factor / (float)LOSSES_GRID_NUM_CHANNELS * grad_tv_loss_val;

#pragma unroll
        for (int c = 0; c < LOSSES_GRID_NUM_CHANNELS; ++c) {
            int bc             = b * LOSSES_GRID_NUM_CHANNELS + c;
            int v_idx          = ((bc * D_grid + d) * H_grid + h) * W_grid + w;
            float v            = grid[v_idx];
            float grad_v_accum = 0.0f; // Accumulate self-gradient in register

            if (d + 1 < D_grid) {
                int u_idx  = ((bc * D_grid + (d + 1)) * H_grid + h) * W_grid + w;
                float u    = grid[u_idx];
                float grad = (u - v) * (float)D_grid / (float)(D_grid - 1) * grad_tv_base;
                grad_v_accum -= grad;
                atomicAdd(&grad_grid[u_idx], grad);
            }

            if (h + 1 < H_grid) {
                int u_idx  = ((bc * D_grid + d) * H_grid + (h + 1)) * W_grid + w;
                float u    = grid[u_idx];
                float grad = (u - v) * (float)H_grid / (float)(H_grid - 1) * grad_tv_base;
                grad_v_accum -= grad;
                atomicAdd(&grad_grid[u_idx], grad);
            }

            if (w + 1 < W_grid) {
                int u_idx  = ((bc * D_grid + d) * H_grid + h) * W_grid + (w + 1);
                float u    = grid[u_idx];
                float grad = (u - v) * (float)W_grid / (float)(W_grid - 1) * grad_tv_base;
                grad_v_accum -= grad;
                atomicAdd(&grad_grid[u_idx], grad);
            }

            // Single atomicAdd for accumulated self-gradient (was up to 3 separate atomicAdds)
            if (grad_v_accum != 0.0f) {
                atomicAdd(&grad_grid[v_idx], grad_v_accum);
            }
        }
    }
}

//
// Dispatches 1-4 replace the Slang auto-differentiated kernels in losses.slang
// with hand-written forward and backward passes.
//
// TODO: Fuse forward+backward into single kernels for dispatches 1-4.
//
// Loss kernels are terminal nodes in the computation graph — nothing downstream
// reads the per-element loss values (they are summed to a scalar for .backward()).
// This means at forward time we already know backward will follow and the upstream
// gradient is 1.0 (after sum, with normalization baked into the factor). A fused
// kernel can: read inputs once -> compute loss -> keep values in registers ->
// compute gradients -> write scalar loss sum (atomicAdd) + input gradients in one
// pass. This is NOT possible for non-terminal kernels (e.g. rendering, post-
// processing) where forward outputs are consumed by downstream operations before
// backward is triggered.
//
// Benefits: halves kernel launches per dispatch, eliminates per-element loss
// tensor allocation and global memory write/read roundtrip, each input value read
// exactly once instead of twice (once in forward, once re-read in backward).
//
// Best candidates in order of expected impact:
//   - Dispatch 3 (gaussian regularization): pure element-wise ops, no atomics in
//     backward, trivially fusible. Forward reads scales/densities, backward needs
//     the same values — fusing keeps them in registers.
//   - Dispatches 1-2 (camera/lidar): also element-wise per-pixel, no neighbor
//     interactions. sign() and clamp() derivatives are cheap to compute alongside
//     the forward value.
//   - Dispatch 4 (bg_grid): harder because backward uses atomicAdd for neighbor
//     gradient scattering. Fusing would mean atomicAdd in the "forward" kernel,
//     mixing read-write patterns. Still possible but less clean.
//
// Requires changing the Layer 1 interface (CudaLossesFunction) to return
// pre-reduced scalar sums instead of per-element loss tensors, and producing
// input gradients as additional forward outputs with a no-op backward.

// DISPATCH 4: Background Texture & Grid Regularization Losses
// ============================================================================
