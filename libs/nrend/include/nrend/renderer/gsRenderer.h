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

#include <nrend/renderer/gutRenderer.h>
namespace nrend {

class GSRenderer final : public GUTRenderer {

public:
    static constexpr char name[] = "3dgs-nrend";

    GSRenderer(const nlohmann::json& rendererState, const Logger& logger)
        : GUTRenderer(rendererState, logger, true) {

        // setup default settings for 3dgs
        m_settings.perRayFeatures = false;
        m_settings.globalZOrder   = true;
        m_settings.renderMode     = Settings::RenderMode::Splat;

        initializeSettings(rendererState, logger);
    };
    virtual ~GSRenderer() = default;

    bool supportVersion(const ModelVersion& version,
                        RenderingParameters::RendererHints rendererHint,
                        RenderingParameters::OptFlags renderFlags) const override {
        return ((rendererHint == RenderingParameters::RendererHints::RendererDefault) ||
                (rendererHint == RenderingParameters::RendererHints::RendererFastest)) &&
               GRUTRenderer::supportVersion(version, rendererHint, renderFlags);
    }
};

} // namespace nrend
