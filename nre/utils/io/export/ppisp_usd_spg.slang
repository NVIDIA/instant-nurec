// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// PPISP (Physically Plausible Image Signal Processing) SPG Shader
//
// Implements the ISP pipeline for USD RenderProducts:
// 1. Exposure compensation
// 2. Vignetting correction (per-channel)
// 3. Color correction via homography
// 4. Camera Response Function (per-channel)
//
// NOTE: All parameters use flat naming to match USD inputs: attributes (UsdShade-compatible).
// SPG requires the Slang struct field names to match USD input names.

// Flat parameter struct matching USD inputs: naming
struct PPISPParams
{
    // Exposure
    float exposureOffset;

    // Vignetting R channel
    float2 vignettingCenterR;
    float vignettingAlpha1R;
    float vignettingAlpha2R;
    float vignettingAlpha3R;

    // Vignetting G channel
    float2 vignettingCenterG;
    float vignettingAlpha1G;
    float vignettingAlpha2G;
    float vignettingAlpha3G;

    // Vignetting B channel
    float2 vignettingCenterB;
    float vignettingAlpha1B;
    float vignettingAlpha2B;
    float vignettingAlpha3B;

    // Color homography (3x3 matrix, row-major)
    float3x3 colorHomography;

    // CRF R channel
    float crfX0R;
    float crfY0R;
    float crfSlopeP0R;
    float crfY0PreGammaR;
    float crfSlopeLineR;
    float crfGammaR;
    float crfX1R;
    float crfY1R;
    float crfSlopeP1R;
    float crfShoulderXR;
    float crfShoulderYR;

    // CRF G channel
    float crfX0G;
    float crfY0G;
    float crfSlopeP0G;
    float crfY0PreGammaG;
    float crfSlopeLineG;
    float crfGammaG;
    float crfX1G;
    float crfY1G;
    float crfSlopeP1G;
    float crfShoulderXG;
    float crfShoulderYG;

    // CRF B channel
    float crfX0B;
    float crfY0B;
    float crfSlopeP0B;
    float crfY0PreGammaB;
    float crfSlopeLineB;
    float crfGammaB;
    float crfX1B;
    float crfY1B;
    float crfSlopeP1B;
    float crfShoulderXB;
    float crfShoulderYB;
};

[[vk::binding(0, 1)]] ParameterBlock<PPISPParams> g_Params;
[[vk::binding(1, 1)]] Texture2D<float4> g_InTex;
[[vk::binding(2, 1)]] RWTexture2D<float4> g_OutTex;

// Solve for power function parameters: y = exp(ln_a + b * ln(x))
float2 solveLnAB(float x0, float y0, float m)
{
    float b = (m * x0) / max(y0, 1.0e-5);
    float lnA = log(max(y0, 1.0e-5)) - b * log(max(x0, 1.0e-5));
    return float2(lnA, b);
}

// Apply vignetting falloff for a single channel
float applyVignetting(float value, float2 uv, float2 opticalCenter, float alpha1, float alpha2, float alpha3)
{
    float2 delta = uv - opticalCenter;
    float r2 = dot(delta, delta);

    float falloff = 1.0;
    float r2Pow = r2;

    falloff += alpha1 * r2Pow;
    r2Pow *= r2;
    falloff += alpha2 * r2Pow;
    r2Pow *= r2;
    falloff += alpha3 * r2Pow;

    falloff = clamp(falloff, 0.0, 1.0);
    return value * falloff;
}

// Apply CRF for a single channel using pre-computed curve points
float applyCRF(float x, float x0, float y0, float slopeP0, float y0PreGamma, float slopeLine,
               float gamma, float x1, float y1, float slopeP1, float shoulderX, float shoulderY)
{
    if (x < 0.0)
    {
        // Below toe point
        return 0.0;
    }
    else if (x < x0)
    {
        // Toe region - power function
        float2 lnAB = solveLnAB(x0, y0, slopeP0);
        x = max(x, 1.0e-5);
        return exp(lnAB.x + lnAB.y * log(x));
    }
    else if (x < x1)
    {
        // Middle region - gamma curve
        return pow(y0PreGamma + slopeLine * (x - x0), gamma);
    }
    else if (x < shoulderX)
    {
        // Shoulder region - inverse power function
        float2 lnAB = solveLnAB(shoulderX - x1, shoulderY - y1, slopeP1);
        float shoulderXOffset = max(shoulderX - x, 1.0e-5);
        return shoulderY - exp(lnAB.x + lnAB.y * log(shoulderXOffset));
    }
    else
    {
        // Above shoulder point
        return 1.0;
    }
}

// Apply color correction via homography in RGI space
float3 applyColorCorrection(float3 rgb, float3x3 homography)
{
    // Convert RGB to Red-Green-Intensity (RGI) space
    float intensity = rgb.x + rgb.y + rgb.z;
    float3 rgi = float3(rgb.x, rgb.y, intensity);

    // Apply homography
    rgi = mul(homography, rgi);

    // Normalize and convert back to RGB
    rgi = rgi * (intensity / (rgi.z + 1.0e-5));
    return float3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ppispProcess(uint3 tid : SV_DispatchThreadID)
{
    uint w = 0, h = 0;
    g_InTex.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    // Load input pixel
    float4 pixel = g_InTex.Load(int3(tid.xy, 0));
    float3 rgb = pixel.rgb;

    // Compute normalized UV coordinates [0, 1]
    float2 uv = float2(tid.x + 0.5, tid.y + 0.5) / float2(w, h);

    // 1. Exposure compensation
    rgb = rgb * exp2(g_Params.exposureOffset);

    // 2. Vignetting correction (per-channel)
    rgb.r = applyVignetting(rgb.r, uv, g_Params.vignettingCenterR,
                            g_Params.vignettingAlpha1R, g_Params.vignettingAlpha2R, g_Params.vignettingAlpha3R);
    rgb.g = applyVignetting(rgb.g, uv, g_Params.vignettingCenterG,
                            g_Params.vignettingAlpha1G, g_Params.vignettingAlpha2G, g_Params.vignettingAlpha3G);
    rgb.b = applyVignetting(rgb.b, uv, g_Params.vignettingCenterB,
                            g_Params.vignettingAlpha1B, g_Params.vignettingAlpha2B, g_Params.vignettingAlpha3B);

    // 3. Color correction via homography
    rgb = applyColorCorrection(rgb, g_Params.colorHomography);

    // 4. Camera Response Function (per-channel)
    rgb.r = applyCRF(rgb.r, g_Params.crfX0R, g_Params.crfY0R, g_Params.crfSlopeP0R,
                     g_Params.crfY0PreGammaR, g_Params.crfSlopeLineR, g_Params.crfGammaR,
                     g_Params.crfX1R, g_Params.crfY1R, g_Params.crfSlopeP1R,
                     g_Params.crfShoulderXR, g_Params.crfShoulderYR);
    rgb.g = applyCRF(rgb.g, g_Params.crfX0G, g_Params.crfY0G, g_Params.crfSlopeP0G,
                     g_Params.crfY0PreGammaG, g_Params.crfSlopeLineG, g_Params.crfGammaG,
                     g_Params.crfX1G, g_Params.crfY1G, g_Params.crfSlopeP1G,
                     g_Params.crfShoulderXG, g_Params.crfShoulderYG);
    rgb.b = applyCRF(rgb.b, g_Params.crfX0B, g_Params.crfY0B, g_Params.crfSlopeP0B,
                     g_Params.crfY0PreGammaB, g_Params.crfSlopeLineB, g_Params.crfGammaB,
                     g_Params.crfX1B, g_Params.crfY1B, g_Params.crfSlopeP1B,
                     g_Params.crfShoulderXB, g_Params.crfShoulderYB);

    // Clamp final output
    rgb = clamp(rgb, 0.0, 1.0);

    // Write output with original alpha
    g_OutTex[tid.xy] = float4(rgb, pixel.a);
}
