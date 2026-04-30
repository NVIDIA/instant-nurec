// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

[ForceInline]
[Differentiable]
float smooth_l1(float diff, float beta) {
    float abs_diff = abs(diff);
    if (beta <= 0.0f) {
        return abs_diff;
    }
    if (abs_diff < beta) {
        return 0.5f * diff * diff / beta;
    }
    return abs_diff - 0.5f * beta;
}

// Helper for PPISP smoothness loss across cameras.
// Computes one element of the smoothness loss for a given (pair, col).
[Differentiable]
float smoothness_loss_across_cams(
    no_diff float beta,
    no_diff int64_t src,
    no_diff int64_t dst,
    no_diff uint col,
    DiffTensorView<float> values  // [num_frames, values_dim]
) {
    float src_val = values.loadOnce(uint2((uint)src, col));
    float dst_val = values.loadOnce(uint2((uint)dst, col));
    return smooth_l1(src_val - dst_val, beta);
}
