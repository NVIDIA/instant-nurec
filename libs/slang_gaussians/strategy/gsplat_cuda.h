// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <torch/extension.h>

/*!
 * Update gradient accumulation buffers for GSplat densification.
 *
 * For each Gaussian with non-zero position gradients, computes:
 *   distance = length(position - ray_origin)
 *   grad_norm = length(grad * distance) * 0.5
 * and accumulates into grad_norm_accum / grad_norm_denom.
 *
 * @param positions        [N, 3] Gaussian positions (float32)
 * @param params_grad      [N, 3] Position gradients (float32)
 * @param ray_origin       [3] Camera/ray origin (float32)
 * @param grad_norm_accum  [N, 1] Accumulated gradient norms, modified in-place (float32)
 * @param grad_norm_denom  [N, 1] Accumulator count, modified in-place (int32)
 * @param threads_per_block Number of CUDA threads per block
 */
void update_gradient_buffers_cuda(
    torch::Tensor const positions,
    torch::Tensor const params_grad,
    torch::Tensor const ray_origin,
    torch::Tensor grad_norm_accum,
    torch::Tensor grad_norm_denom,
    int threads_per_block);
