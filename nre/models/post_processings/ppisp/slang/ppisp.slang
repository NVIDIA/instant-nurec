// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

#include "extensions.slang"

struct VignettingParams : IDifferentiable {
    float2 optical_center;
    vector<float, NUM_VIGNETTING_ALPHA_TERMS> alphas;
};

struct PositiveParameter : IDifferentiable {
    float raw;
    [Differentiable] float forward() { return log(1.0 + exp(raw)); }
};

struct ClampedParameter : IDifferentiable {
    float raw;
    [Differentiable] float forward() { return 1.0 / (1.0 + exp(-raw)); }
};

struct CRFParams : IDifferentiable {
    PositiveParameter x0_offset;
    ClampedParameter y0;
    ClampedParameter y1_fract;
    PositiveParameter toe_length;
    PositiveParameter shoulder_length;
    PositiveParameter shoulder_overshoot;
    PositiveParameter gamma;
};

struct CurvePoints : IDifferentiable {
    float x0;
    float y0;
    float slope_p0;
    float y0_pre_gamma;
    float slope_line;
    float gamma;
    float x1;
    float y1;
    float slope_p1;
    float shoulder_x;
    float shoulder_y;
};

// Compute log-space coefficients for y = a * x^b with slope m at (x0, y0).
// Matches PiecewisePowerFunction.solve_ln_a_b logic from ppisp.py.
[Differentiable]
float2 solve_ln_a_b(float x0, float y0, float m) {
    float b = (m * x0) / y0;
    float ln_a = log(max(y0, EPSILON)) - b * log(max(x0, EPSILON));
    return float2(ln_a, b);
}


// Invert piecewise power function for target y using curve points.
// Matches PiecewisePowerFunction.inverse_ppf in ppisp.py.
[ForceInline]
[Differentiable]
float inverse_ppf(CurvePoints curve, float y) {
    float y_safe = max(y, EPSILON);

    // Toe region: y = a*x^b -> x = exp((log(y) - ln_a) / b)
    float2 ln_a_b_toe = solve_ln_a_b(curve.x0, curve.y0, curve.slope_p0);
    float toe_x = exp((log(y_safe) - ln_a_b_toe.x) / ln_a_b_toe.y);

    // Middle region: y = (m*x + b)^gamma -> x = ((y^(1/gamma) - y0_pre_gamma) / slope_line) + x0
    float middle_x = (pow(y_safe, 1.0 / curve.gamma) - curve.y0_pre_gamma) / curve.slope_line + curve.x0;

    // Shoulder region: y = shoulder_y - a*(shoulder_x - x)^b
    float2 ln_a_b_shoulder =
        solve_ln_a_b(curve.shoulder_x - curve.x1, curve.shoulder_y - curve.y1, curve.slope_p1);
    float shoulder_diff = max(curve.shoulder_y - y, EPSILON);
    float shoulder_x = curve.shoulder_x - exp((log(shoulder_diff) - ln_a_b_shoulder.x) / ln_a_b_shoulder.y);

    // Above dynamic range (y > 1.0): return x where curve intersects y = 1.0
    float above_diff = max(curve.shoulder_y - 1.0, EPSILON);
    float above_x = curve.shoulder_x - exp((log(above_diff) - ln_a_b_shoulder.x) / ln_a_b_shoulder.y);

    if (y >= 0.0 && y < curve.y0) {
        return toe_x;
    }
    if (y >= curve.y0 && y < curve.y1) {
        return middle_x;
    }
    if (y >= curve.y1 && y <= 1.0) {
        return shoulder_x;
    }
    if (y > 1.0) {
        return above_x;
    }
    return 0.0;
}

[Differentiable]
VignettingParams get_vignetting_params(DiffTensorView vignetting_params, uint camera_idx,
                                       uint channel_idx) {
    VignettingParams params;
    params.optical_center = vignetting_params.loadVecUniform<2>(uint3(camera_idx, channel_idx, 0));
    params.alphas = vignetting_params.loadVecUniform<NUM_VIGNETTING_ALPHA_TERMS>(uint3(camera_idx, channel_idx, 2));

    return params;
}

[Differentiable]
float3x3 get_homography(DiffTensorView color_params, uint frame_idx) {
    return float3x3(
        color_params.loadVecUniform<3>(uint2(frame_idx, 0)),
        color_params.loadVecUniform<3>(uint2(frame_idx, 3)),
        float3(color_params.loadVecUniform<2>(uint2(frame_idx, 6)), 1.0));
}

// Apply homography to 2D RG chromaticity with perspective division
// Equivalent to Python ColorCorrection.apply_color_correction_rg
// Input: 2D point [r, g], Output: transformed [r', g'] after homography
[ForceInline]
[Differentiable]
float2 apply_color_correction_rg(float3x3 H, float2 rg) {
    float3 p = float3(rg.x, rg.y, 1.0f);
    float3 t = mul(H, p);
    float inv_z = 1.0f / (t.z + EPSILON);
    return float2(t.x * inv_z, t.y * inv_z);
}

[Differentiable]
CRFParams get_crf_params(DiffTensorView crf_params, uint camera_idx, uint channel_idx) {
    CRFParams params;
    params.x0_offset.raw = crf_params.loadUniform(uint3(camera_idx, channel_idx, 0));
    params.y0.raw = crf_params.loadUniform(uint3(camera_idx, channel_idx, 1));
    params.y1_fract.raw = crf_params.loadUniform(uint3(camera_idx, channel_idx, 2));
    params.toe_length.raw = crf_params.loadUniform(uint3(camera_idx, channel_idx, 3));
    params.shoulder_length.raw = crf_params.loadUniform(uint3(camera_idx, channel_idx, 4));
    params.shoulder_overshoot.raw = crf_params.loadUniform(uint3(camera_idx, channel_idx, 5));
    params.gamma.raw = crf_params.loadUniform(uint3(camera_idx, channel_idx, 6));

    return params;
}

[ForceInline]
[Differentiable]
CurvePoints compute_curve_points(CRFParams params) {
    CurvePoints curve;
    float x0_offset = params.x0_offset.forward();
    float y0 = params.y0.forward();
    float y1_fract = params.y1_fract.forward();
    float toe_length = params.toe_length.forward();
    float shoulder_length = params.shoulder_length.forward();
    float shoulder_overshoot = params.shoulder_overshoot.forward();
    float gamma = max(params.gamma.forward(), 0.1);

    // Compute p0 = (x0, y0)
    curve.x0 = x0_offset * (1.0 + toe_length);
    curve.y0 = y0;

    // The slope at p0 is given by the line passing through (toe_length, 0) and p0
    curve.slope_p0 = y0 / x0_offset;

    // Before applying gamma, p0 and p1 are connected with a straight line
    curve.y0_pre_gamma = pow(y0, 1.0 / gamma);
    curve.slope_line = curve.slope_p0 / (gamma * pow(curve.y0_pre_gamma, gamma - 1.0));
    curve.gamma = gamma;

    // Find p1 = (x1, y1)
    curve.y1 = y0 + (1.0 - y0) * y1_fract;
    float y1_pre_gamma = pow(curve.y1, 1.0 / gamma);
    curve.x1 = curve.x0 + (y1_pre_gamma - curve.y0_pre_gamma) / curve.slope_line;

    // The slope at p1 is given from p1
    curve.slope_p1 = gamma * curve.slope_line * pow(y1_pre_gamma, gamma - 1.0);

    // The shoulder point is given by extending from p1 along slope_p1 and intersecting y = 1 + shoulder_overshoot
    float remaining_y = 1.0 - curve.y1;
    curve.shoulder_y = 1.0 + remaining_y * shoulder_overshoot;
    float shoulder_intercept_x_offset = (curve.shoulder_y - curve.y1) / curve.slope_p1;
    curve.shoulder_x = curve.x1 + shoulder_intercept_x_offset * (1.0 + shoulder_length);

    return curve;
}

[ForceInline]
[Differentiable]
CurvePoints crf_curve_points(DiffTensorView crf_params, uint camera_idx, uint channel_idx) {
    // Wrapper for raw CRF params -> curve points (used by losses and tests).
    CRFParams params = get_crf_params(crf_params, camera_idx, channel_idx);
    return compute_curve_points(params);
}

[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void ppisp(no_diff uint batch_size,
           no_diff uint num_cameras,
           no_diff uint num_frames,
           DiffTensorView exposure_params,    // Shape: [num_frames]
           DiffTensorView vignetting_params,  // Shape: [num_cameras, 3, 5]
           DiffTensorView color_params,       // Shape: [num_frames, 8]
           DiffTensorView crf_params,         // Shape: [num_cameras, 3, 7]
           DiffTensorView rgb_in,             // Shape: [batch_size, 3] for RGB channels (input)
           DiffTensorView rgb_out,            // Shape: [batch_size, 3] for RGB channels (output)
           no_diff TensorView<float2> uv_coords,
           no_diff TensorView<int16_t> camera_idcs,
           no_diff TensorView<int32_t> frame_idcs
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= batch_size) {
        return;
    }

    float3 rgb = rgb_in.loadVecOnce<3>(uint2(tid, 0));

    float2 uv = no_diff uv_coords[tid];

    // get frame index
    int camera_idx = no_diff camera_idcs[tid];
    int frame_idx = no_diff frame_idcs[tid];

    // check if frame or camera index is negative which will skip certain computations
    no_diff bool has_frame = frame_idx > -1;
    no_diff bool has_camera = camera_idx > -1;

    if (has_frame) {
        float exposure_offset = exposure_params.loadUniform(no_diff uint(frame_idx));

        // Exposure compensation
        rgb = rgb * exp2(exposure_offset);
    }

    if (has_camera) {
        // Vignetting
        [ForceUnroll]
        for (uint i = 0; i < 3; i++) {
            VignettingParams vig_params = get_vignetting_params(vignetting_params, no_diff uint(camera_idx), i);
            float2 delta = uv - vig_params.optical_center;
            float r2 = dot(delta, delta);
            float falloff = 1.0;
            float r2_pow = r2;
            [ForceUnroll]
            for (uint j = 0; j < NUM_VIGNETTING_ALPHA_TERMS; j++) {
                falloff += vig_params.alphas[j] * r2_pow;
                r2_pow *= r2;
            }
            falloff = clamp(falloff, 0.0, 1.0);
            rgb[i] = rgb[i] * falloff;
        }
    }

    if (has_frame) {
        // Color correction
        float3x3 homography = get_homography(color_params, no_diff uint(frame_idx));
        // Convert RGB to Red-Green-Intensity (RGI) space
        float intensity = rgb.x + rgb.y + rgb.z;
        float3 rgi = float3(rgb.x, rgb.y, intensity);
        rgi = mul(homography, rgi);
        rgi = rgi * (intensity / (rgi.z + EPSILON));
        rgb = float3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);
    }

    if (has_camera) {
        // CRF
        [ForceUnroll]
        for (uint i = 0; i < 3; i++) {
            CurvePoints curve = crf_curve_points(crf_params, no_diff uint(camera_idx), i);
            // Apply CRF
            float x = rgb[i];
            if (x < 0.0) {  // Below toe point
                rgb[i] = 0.0;
            } else if (x < curve.x0) {  // Toe region
                float2 ln_a_b = solve_ln_a_b(curve.x0, curve.y0, curve.slope_p0);
                x = max(x, EPSILON);
                rgb[i] = exp(ln_a_b.x + ln_a_b.y * log(x));
            } else if (x < curve.x1) {  // Middle region
                rgb[i] = pow(curve.y0_pre_gamma + curve.slope_line * (x - curve.x0), curve.gamma);
            } else if (x < curve.shoulder_x) {  // Shoulder region
                float2 ln_a_b = solve_ln_a_b(curve.shoulder_x - curve.x1, curve.shoulder_y - curve.y1, curve.slope_p1);
                float shoulder_x_offset = max(curve.shoulder_x - x, EPSILON);
                rgb[i] = curve.shoulder_y - exp(ln_a_b.x + ln_a_b.y * log(shoulder_x_offset));
            } else { // Above shoulder point
                rgb[i] = 1.0;
            }
        }
        rgb = clamp(rgb, 0.0, 1.0);
    }
    else {
        rgb = clamp(rgb, 0.0, 1.0);
        rgb = pow(rgb, 1.0 / 2.2);
    }

    rgb_out.storeVecOnce(uint2(tid, 0), rgb);
}
