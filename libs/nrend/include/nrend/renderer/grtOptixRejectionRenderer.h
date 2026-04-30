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

#include <nrend/renderer/grtOptixRenderer.h>

namespace nrend {

class GRTOptixRejectionRenderer final : public GRTOptixRenderer {

public:
    static constexpr char name[] = "3dgrt-rejection-optix-nrend";

    GRTOptixRejectionRenderer(const nlohmann::json& rendererState, const Logger& logger)
        : GRTOptixRenderer(rendererState, logger, true) {

        m_settings.pipelineType                  = Settings::RejectionSampling;
        m_settings.pipelineKBufferSize           = 7;
        m_settings.pipelineNumSamples            = 7;
        m_settings.primitiveType                 = Settings::TransformedAabb;
        m_settings.primitiveDensityScaleClamping = true;

        initializeSettings(rendererState, logger);
    }
    virtual ~GRTOptixRejectionRenderer() = default;

    bool supportVersion(const ModelVersion& version,
                        RenderingParameters::RendererHints rendererHint,
                        RenderingParameters::OptFlags renderFlags) const override {
        return ((rendererHint == RenderingParameters::RendererHints::RendererDefault) ||
                (rendererHint == RenderingParameters::RendererHints::RendererQuality)) &&
               GRUTRenderer::supportVersion(version, rendererHint, renderFlags);
    }
};

} // namespace nrend
