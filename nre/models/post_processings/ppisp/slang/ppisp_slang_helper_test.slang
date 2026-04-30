// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// Test kernels for PPISP helper functions.
// These kernels expose internal helper functions for unit testing.

import "ppisp.slang";

// ============================================================================
// Test kernel for apply_color_correction_rg
// ============================================================================

// Test kernel that wraps apply_color_correction_rg for a single (H, rg) pair.
// Same interface as the helper function: (float3x3 H, float2 rg) -> float2
//
// Interface:
//   H: Homography matrix, shape [3, 3]
//   rg: Input RG chromaticity, shape [2]
//   out_rg: Output RG chromaticity, shape [2]
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void ppisp_test_apply_color_correction_rg(
    DiffTensorView<float> H,     // [3, 3] - homography matrix
    DiffTensorView<float> rg,    // [2] - input RG chromaticity
    DiffTensorView<float> out_rg // [2] - output RG chromaticity
) {
    // Single thread kernel
    if (cudaThreadIdx().x != 0 || cudaBlockIdx().x != 0) return;

    // Load homography matrix
    float3x3 H_mat = float3x3(
        float3(H[uint2(0, 0)], H[uint2(0, 1)], H[uint2(0, 2)]),
        float3(H[uint2(1, 0)], H[uint2(1, 1)], H[uint2(1, 2)]),
        float3(H[uint2(2, 0)], H[uint2(2, 1)], H[uint2(2, 2)])
    );

    // Load input rg
    float2 rg_in = float2(rg[uint1(0)], rg[uint1(1)]);

    // Apply color correction using the helper function
    float2 rg_out = apply_color_correction_rg(H_mat, rg_in);

    // Store output
    out_rg.storeOnce(uint1(0), rg_out.x);
    out_rg.storeOnce(uint1(1), rg_out.y);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void ppisp_test_inverse_ppf(
    DiffTensorView<float> raw_params, // [7] - CRF raw params
    DiffTensorView<float> y,          // [1] - input y
    DiffTensorView<float> out_x       // [1] - output x
) {
    if (cudaThreadIdx().x != 0 || cudaBlockIdx().x != 0) return;

    CRFParams params;
    params.x0_offset.raw = raw_params[uint1(0)];
    params.y0.raw = raw_params[uint1(1)];
    params.y1_fract.raw = raw_params[uint1(2)];
    params.toe_length.raw = raw_params[uint1(3)];
    params.shoulder_length.raw = raw_params[uint1(4)];
    params.shoulder_overshoot.raw = raw_params[uint1(5)];
    params.gamma.raw = raw_params[uint1(6)];

    // Curve points computed via shared helper (matches PiecewisePowerFunction.crf_curve_points).
    CurvePoints curve = compute_curve_points(params);

    float y_val = y[uint1(0)];
    float x_val = inverse_ppf(curve, y_val);
    out_x.storeOnce(uint1(0), x_val);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void ppisp_test_compute_curve_points(
    DiffTensorView<float> raw_params, // [7] - CRF raw params
    DiffTensorView<float> out_curve   // [11] - output curve points
) {
    if (cudaThreadIdx().x != 0 || cudaBlockIdx().x != 0) return;

    CRFParams params;
    params.x0_offset.raw = raw_params[uint1(0)];
    params.y0.raw = raw_params[uint1(1)];
    params.y1_fract.raw = raw_params[uint1(2)];
    params.toe_length.raw = raw_params[uint1(3)];
    params.shoulder_length.raw = raw_params[uint1(4)];
    params.shoulder_overshoot.raw = raw_params[uint1(5)];
    params.gamma.raw = raw_params[uint1(6)];

    CurvePoints curve = compute_curve_points(params);
    out_curve.storeOnce(uint1(0), curve.x0);
    out_curve.storeOnce(uint1(1), curve.y0);
    out_curve.storeOnce(uint1(2), curve.slope_p0);
    out_curve.storeOnce(uint1(3), curve.y0_pre_gamma);
    out_curve.storeOnce(uint1(4), curve.slope_line);
    out_curve.storeOnce(uint1(5), curve.gamma);
    out_curve.storeOnce(uint1(6), curve.x1);
    out_curve.storeOnce(uint1(7), curve.y1);
    out_curve.storeOnce(uint1(8), curve.slope_p1);
    out_curve.storeOnce(uint1(9), curve.shoulder_x);
    out_curve.storeOnce(uint1(10), curve.shoulder_y);
}

// ============================================================================
// Tests for crf_curve_points helpers
// ============================================================================

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void ppisp_test_crf_curve_points(
    no_diff uint camera_idx,
    no_diff uint channel_idx,
    DiffTensorView<float> crf_params,  // [num_cameras, 3, 7]
    DiffTensorView<float> out_curve    // [11]
) {
    if (cudaThreadIdx().x != 0 || cudaBlockIdx().x != 0) return;

    CurvePoints curve = crf_curve_points(crf_params, camera_idx, channel_idx);
    out_curve.storeOnce(uint1(0), curve.x0);
    out_curve.storeOnce(uint1(1), curve.y0);
    out_curve.storeOnce(uint1(2), curve.slope_p0);
    out_curve.storeOnce(uint1(3), curve.y0_pre_gamma);
    out_curve.storeOnce(uint1(4), curve.slope_line);
    out_curve.storeOnce(uint1(5), curve.gamma);
    out_curve.storeOnce(uint1(6), curve.x1);
    out_curve.storeOnce(uint1(7), curve.y1);
    out_curve.storeOnce(uint1(8), curve.slope_p1);
    out_curve.storeOnce(uint1(9), curve.shoulder_x);
    out_curve.storeOnce(uint1(10), curve.shoulder_y);
}
