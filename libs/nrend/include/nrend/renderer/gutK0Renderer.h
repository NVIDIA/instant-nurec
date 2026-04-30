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

class GUTK0Renderer final : public GUTRenderer {

public:
    static constexpr char name[] = "3dgut-k0-nrend";

    GUTK0Renderer(const nlohmann::json& rendererState, const Logger& logger)
        : GUTRenderer(rendererState, logger, true) {
        m_settings.kBufferSize = 0;
        initializeSettings(rendererState, logger);
    };
    virtual ~GUTK0Renderer() = default;

    bool supportVersion(const ModelVersion& version,
                        RenderingParameters::RendererHints rendererHint,
                        RenderingParameters::OptFlags renderFlags) const override {
        return ((rendererHint == RenderingParameters::RendererHints::RendererDefault) ||
                (rendererHint == RenderingParameters::RendererHints::RendererFast)) &&
               GRUTRenderer::supportVersion(version, rendererHint, renderFlags);
    }
};

} // namespace nrend
