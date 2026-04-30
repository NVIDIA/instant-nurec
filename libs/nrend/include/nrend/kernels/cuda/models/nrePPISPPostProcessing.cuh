// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <cuda_fp16.h>
#include <nrend/renderer/renderParameters.h>
#include <tiny-cuda-nn/vec.h>

// Dynamic PPISP post-processing reading parameters via binding indices
// TParams must define static constexpr int BufferIndices for each parameter
template <typename TParams>
struct NREPPISPPostProcessing {

    static constexpr float epsilon = 1e-5f;

    static inline __device__ void solve_ln_a_b(float& b, float& ln_a, float x0, float y0, float m) {
        b    = (m * x0) / y0;
        ln_a = __logf(y0 < epsilon ? epsilon : y0) - b * log(x0 < epsilon ? epsilon : x0);
    }

    template <typename TRay>
    static inline __device__ void eval(TRay& ray,
                                       const nrend::RenderParameters& renderParams,
                                       nrend::MemoryHandles trainParams,
                                       const tcnn::ivec2* sensorsIdsPtr) {

        // This kernel assumes the following:
        static_assert(TParams::NUM_CHANNELS == 3);
        static_assert(TParams::NUM_VIGNETTING_OPTICAL_CENTER == 2);
        static_assert(TParams::NUM_VIGNETTING_ALPHA_TERMS == 3);
        static_assert(TParams::NUM_CRF_PARAMS == 7);
        static_assert(TParams::NUM_HOMOGRAPHY_PARAMS == 8);

        // Extract camera and frame
        int cam   = sensorsIdsPtr ? sensorsIdsPtr[0].x : 0;
        int frame = sensorsIdsPtr ? sensorsIdsPtr[0].y : 0;

        // If camera or frame index is out-of-bounds, do nothing
        if (cam < 0 || cam >= TParams::NUM_CAMERAS || frame < 0 || frame >= TParams::TOTAL_FRAMES) {
            return;
        }

        // Get RGB value (already normalized) and compute pixel coordinates normalized
        tcnn::vec3 rgb = ray.features.vec.xyz();

        const tcnn::vec2 uv{
            (float)((threadIdx.x + blockDim.x * blockIdx.x) / renderParams.frameResolution.x),
            (float)((threadIdx.y + blockDim.y * blockIdx.y) / renderParams.frameResolution.y)};

        // Frame params: exposure (1) + homography (8)
        const __half* framePack = trainParams.bufferPtr<const __half>(TParams::FrameParamsBufferIndex);
        int fbase               = frame * 9;
        float exposure_offset   = __half2float(framePack[fbase + 0]);

        // Apply exposure
        rgb = rgb * exp2f(exposure_offset);

        // Sensor-channel params: vignetting optical_center (2) and alpha (3) and CRF (7)
        const __half* sensorPack = trainParams.bufferPtr<const __half>(TParams::SensorParamsBufferIndex);

        // Vignetting
#pragma unroll
        for (int ch = 0; ch < 3; ++ch) {
            int sbase = (cam * 3 + ch) * 12;

            tcnn::vec2 center = tcnn::vec2(
                __half2float(sensorPack[sbase + 0]),
                __half2float(sensorPack[sbase + 1]));

            tcnn::vec2 delta = uv - center;

            float r2      = tcnn::dot(delta, delta);
            float falloff = 1.f;
            float r2_pow  = r2; // Polynomial terms are: r^2, r^4, r^6

#pragma unroll
            for (int j = 2; j < 5; ++j) {
                float alpha = __half2float(sensorPack[sbase + j]);
                falloff += alpha * r2_pow;
                r2_pow *= r2;
            }
            falloff = tcnn::clamp(falloff, 0.f, 1.f);

            rgb[ch] *= falloff;
        }

        // Get homography for color correction (transposed as TCNN is column-major and Slang is row-major)
        tcnn::mat3 h;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            h[i % 3][i / 3] = __half2float(framePack[fbase + 1 + i]);
        }
        h[2][2] = 1.f;

        // Color correction apply (match Slang implementation)
        float intensity   = rgb.x + rgb.y + rgb.z;
        tcnn::vec3 rgi    = tcnn::vec3(rgb.x, rgb.y, intensity);
        tcnn::vec3 rgi_tr = h * rgi;
        float scale       = intensity / (rgi_tr.z + epsilon);
        tcnn::vec3 rgi_tc = rgi_tr * scale;
        rgb               = tcnn::vec3(rgi_tc.x, rgi_tc.y, rgi_tc.z - rgi_tc.x - rgi_tc.y);

        // CRF
#pragma unroll
        for (int ch = 0; ch < 3; ++ch) {
            int sbase = (cam * 3 + ch) * 12;

            // Raw CRF parameters
            float x0_offset_raw          = __half2float(sensorPack[sbase + 5]);
            float y0_raw                 = __half2float(sensorPack[sbase + 6]);
            float y1_fract_raw           = __half2float(sensorPack[sbase + 7]);
            float toe_length_raw         = __half2float(sensorPack[sbase + 8]);
            float shoulder_length_raw    = __half2float(sensorPack[sbase + 9]);
            float shoulder_overshoot_raw = __half2float(sensorPack[sbase + 10]);
            float gamma_raw              = __half2float(sensorPack[sbase + 11]);

            // Apply parameter transforms (softplus for positive params, sigmoid for fractions)
            float x0_offset          = __logf(__expf(x0_offset_raw) + 1.f);
            float y0                 = 1.f / (1.f + __expf(-y0_raw));
            float y1_fract           = 1.f / (1.f + __expf(-y1_fract_raw));
            float toe_length         = __logf(__expf(toe_length_raw) + 1.f);
            float shoulder_length    = __logf(__expf(shoulder_length_raw) + 1.f);
            float shoulder_overshoot = __logf(__expf(shoulder_overshoot_raw) + 1.f);
            float gamma              = __logf(__expf(gamma_raw) + 1.f);

            // CRF compute curve points
            float x0                          = x0_offset * (1.f + toe_length);
            float slope_p0                    = y0 / x0_offset;
            float y0_pre_gamma                = __powf(y0, 1.f / gamma);
            float slope_line                  = slope_p0 / (gamma * __powf(y0_pre_gamma, gamma - 1.f));
            float y1                          = y0 + (1.f - y0) * y1_fract;
            float y1_pre_gamma                = __powf(y1, 1.f / gamma);
            float x1                          = x0 + (y1_pre_gamma - y0_pre_gamma) / slope_line;
            float slope_p1                    = gamma * slope_line * __powf(y1_pre_gamma, gamma - 1.f);
            float remaining_y                 = 1.f - y1;
            float shoulder_y                  = 1.f + remaining_y * shoulder_overshoot;
            float shoulder_intercept_x_offset = (shoulder_y - y1) / slope_p1;
            float shoulder_x                  = x1 + shoulder_intercept_x_offset * (1.f + shoulder_length);

            // CRF forward apply
            float x = rgb[ch];
            float y;
            if (x < 0.f) {
                y = 0.f;
            } else if (x < x0) {
                float b, ln_a;
                solve_ln_a_b(b, ln_a, x0, y0, slope_p0);
                float x_safe = fmaxf(x, epsilon);
                y            = __expf(ln_a + b * __logf(x_safe));
            } else if (x < x1) {
                y = __powf(y0_pre_gamma + slope_line * (x - x0), gamma);
            } else if (x < shoulder_x) {
                float b, ln_a;
                solve_ln_a_b(b, ln_a, (shoulder_x - x1), (shoulder_y - y1), slope_p1);
                float shoulder_x_offset = fmaxf(shoulder_x - x, epsilon);
                y                       = shoulder_y - __expf(ln_a + b * __logf(shoulder_x_offset));
            } else {
                y = 1.f;
            }

            rgb[ch] = y;
        }

        // Clamp output rgb to [0, 1]
        rgb = tcnn::clamp(rgb, 0.f, 1.f);

        // Write back
        ray.features.vec.xyz() = rgb;
    }
};
