-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: LicenseRef-NvidiaProprietary
--
-- NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
-- property and proprietary rights in and to this material, related
-- documentation and any modifications thereto. Any use, reproduction,
-- disclosure or distribution of this material and related documentation
-- without an express license agreement from NVIDIA CORPORATION or
-- its affiliates is strictly prohibited.

-- PPISP (Physically Plausible Image Signal Processing) SPG Launcher
--
-- Binds PPISP parameters and dispatches the compute shader for
-- USD RenderProduct post-processing.
--
-- NOTE: Uses flat parameter names matching USD inputs: attributes (UsdShade-compatible).

function ppispProcess(inputs, outputs, params)
    local in_rgba = inputs["HdrColor"]
    assert(in_rgba and in_rgba.rank == 2, "HdrColor input must be a 2D texture")

    -- Output texture mirrors input shape and dtype
    local height = in_rgba.shape[1]
    local width = in_rgba.shape[2]
    outputs["PPISPColor"] = slang.empty({height, width}, in_rgba.dtype)

    -- Pass params directly to preserve __fullName for shader reflection matching.
    -- slang.float3x3() now automatically flattens nested matrix values from USD params.
    local homographyMatrix = params["colorHomography"] and slang.float3x3(params["colorHomography"])
        or slang.float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    -- Helper to get vignetting center with default
    local function getVignettingCenter(suffix)
        local p = params["vignettingCenter" .. suffix]
        return p and slang.float2(p) or slang.float2(0.5, 0.5)
    end

    return slang.dispatch({
        bind = {
            slang.ParameterBlock(
                -- Exposure
                slang.float(params["exposureOffset"] or 0.0),

                -- Vignetting R channel
                getVignettingCenter("R"),
                slang.float(params["vignettingAlpha1R"] or 0.0),
                slang.float(params["vignettingAlpha2R"] or 0.0),
                slang.float(params["vignettingAlpha3R"] or 0.0),

                -- Vignetting G channel
                getVignettingCenter("G"),
                slang.float(params["vignettingAlpha1G"] or 0.0),
                slang.float(params["vignettingAlpha2G"] or 0.0),
                slang.float(params["vignettingAlpha3G"] or 0.0),

                -- Vignetting B channel
                getVignettingCenter("B"),
                slang.float(params["vignettingAlpha1B"] or 0.0),
                slang.float(params["vignettingAlpha2B"] or 0.0),
                slang.float(params["vignettingAlpha3B"] or 0.0),

                -- Color homography
                homographyMatrix,

                -- CRF R channel
                slang.float(params["crfX0R"] or 0.1),
                slang.float(params["crfY0R"] or 0.1),
                slang.float(params["crfSlopeP0R"] or 1.0),
                slang.float(params["crfY0PreGammaR"] or 0.1),
                slang.float(params["crfSlopeLineR"] or 1.0),
                slang.float(params["crfGammaR"] or 2.2),
                slang.float(params["crfX1R"] or 0.9),
                slang.float(params["crfY1R"] or 0.9),
                slang.float(params["crfSlopeP1R"] or 1.0),
                slang.float(params["crfShoulderXR"] or 1.2),
                slang.float(params["crfShoulderYR"] or 1.0),

                -- CRF G channel
                slang.float(params["crfX0G"] or 0.1),
                slang.float(params["crfY0G"] or 0.1),
                slang.float(params["crfSlopeP0G"] or 1.0),
                slang.float(params["crfY0PreGammaG"] or 0.1),
                slang.float(params["crfSlopeLineG"] or 1.0),
                slang.float(params["crfGammaG"] or 2.2),
                slang.float(params["crfX1G"] or 0.9),
                slang.float(params["crfY1G"] or 0.9),
                slang.float(params["crfSlopeP1G"] or 1.0),
                slang.float(params["crfShoulderXG"] or 1.2),
                slang.float(params["crfShoulderYG"] or 1.0),

                -- CRF B channel
                slang.float(params["crfX0B"] or 0.1),
                slang.float(params["crfY0B"] or 0.1),
                slang.float(params["crfSlopeP0B"] or 1.0),
                slang.float(params["crfY0PreGammaB"] or 0.1),
                slang.float(params["crfSlopeLineB"] or 1.0),
                slang.float(params["crfGammaB"] or 2.2),
                slang.float(params["crfX1B"] or 0.9),
                slang.float(params["crfY1B"] or 0.9),
                slang.float(params["crfSlopeP1B"] or 1.0),
                slang.float(params["crfShoulderXB"] or 1.2),
                slang.float(params["crfShoulderYB"] or 1.0)
            ),
            -- Texture SRV / UAV pair
            slang.Texture2D(in_rgba),
            slang.RWTexture2D(outputs["PPISPColor"]),
        },
    })
end