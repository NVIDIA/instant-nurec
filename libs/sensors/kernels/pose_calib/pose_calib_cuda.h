// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <ku/common_host.h>
#include <torch/extension.h>

// Duplicated from ku/common.cuh (which is __host__ __device__ and cannot be
// included in a host-only header). Keep in sync if the shared version changes.
inline uint32_t div_round_up(uint32_t val, uint32_t divisor) {
    return (val + divisor - 1) / divisor;
}

// ShutterType enum values (must match sensors/kernels/cameras/interface.slang)
enum class ShutterType : int {
    ROLLING_TOP_TO_BOTTOM = 1,
    ROLLING_LEFT_TO_RIGHT = 2,
    ROLLING_BOTTOM_TO_TOP = 3,
    ROLLING_RIGHT_TO_LEFT = 4,
    GLOBAL                = 5,
};

/*!
 * Forward pass: compute calibrated, optionally rolling-shutter-interpolated poses.
 *
 * @param batch_size              Number of samples
 * @param T_startend_allviews     [V, 2, 4, 4] float32 start/end poses for all views
 * @param embed_weights           [V, 9] float32 calibration embeddings (ignored when !enable_calib)
 * @param frame_idx               [N] int32 frame indices into views
 * @param rect_points_lb          [N, 2, 2] float32 rect corners (ignored when !has_subsampling)
 * @param resolution              [N, 2] float32 sensor resolution (ignored when !has_subsampling)
 * @param timestamps_startend     [V, 2] int64 start/end timestamps per view
 * @param shutter_type            ShutterType enum value
 * @param enable_calib            Whether to apply calibration deltas
 * @param has_subsampling         Whether to apply rolling-shutter interpolation
 * @param T_out                   [N, 2, 4, 4] float32 output poses (pre-allocated)
 * @param timestamps_out          [N, 2] int64 output timestamps (pre-allocated)
 */
void pose_calib_forward_cuda(
    int batch_size,
    const torch::Tensor& T_startend_allviews,
    const torch::Tensor& embed_weights,
    const torch::Tensor& frame_idx,
    const torch::Tensor& rect_points_lb,
    const torch::Tensor& resolution,
    const torch::Tensor& timestamps_startend,
    int shutter_type,
    bool enable_calib,
    bool has_subsampling,
    const torch::Tensor& T_out,
    const torch::Tensor& timestamps_out,
    const torch::Tensor& scratch);

/*!
 * Backward pass: compute gradients of embed_weights from grad_T_out.
 *
 * Only called when enable_calib=true (embed_weights is the sole differentiable
 * input). Re-runs the forward chain to regenerate intermediates, then
 * propagates gradients backward through each stage via analytic Jacobians.
 *
 * timestamps_startend is not needed here (timestamps are non-differentiable
 * and not read by the backward kernel).
 *
 * @param batch_size              Number of samples
 * @param T_startend_allviews     [V, 2, 4, 4] float32 start/end poses (not differentiable)
 * @param embed_weights           [V, 9] float32 calibration embeddings (forward values)
 * @param frame_idx               [N] int32 frame indices into views
 * @param rect_points_lb          [N, 2, 2] float32 rect corners (ignored when !has_subsampling)
 * @param resolution              [N, 2] float32 sensor resolution (ignored when !has_subsampling)
 * @param shutter_type            ShutterType enum value
 * @param has_subsampling         Whether rolling-shutter interpolation was applied
 * @param grad_T_out              [N, 2, 4, 4] float32 incoming gradient on output poses
 * @param grad_embed_weights      [V, 9] float32 output gradient buffer (must be zero-initialized)
 */
void pose_calib_backward_cuda(
    int batch_size,
    const torch::Tensor& T_startend_allviews,
    const torch::Tensor& embed_weights,
    const torch::Tensor& frame_idx,
    const torch::Tensor& rect_points_lb,
    const torch::Tensor& resolution,
    int shutter_type,
    bool has_subsampling,
    const torch::Tensor& grad_T_out,
    const torch::Tensor& grad_embed_weights,
    const torch::Tensor& scratch);
