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

#include <nrend/models/modelVersion.h>
#include <nrend/renderer/renderer.h>

namespace nrend {

class NRendererImplementation : public NRenderer {

protected:
    ModelVersion m_modelVersion;

public:
    NRendererImplementation(const nlohmann::json& rendererState, const Logger& logger)
        : NRenderer(logger) {}
    virtual ~NRendererImplementation() = default;

    virtual bool supportVersion(const ModelVersion& version,
                                RenderingParameters::RendererHints rendererHint,
                                RenderingParameters::OptFlags renderFlags) const = 0;

    virtual Status initialize(const ModelVersion& version,
                              const nlohmann::json& modelState,
                              const RenderingParameters& renderParams) = 0;

    virtual Status getModelVersion(int& versionMajor,
                                   int& versionMinor,
                                   int& versionPatch,
                                   const char*& modelName) const override {
        versionMajor = m_modelVersion.number().major;
        versionMinor = m_modelVersion.number().minor;
        versionPatch = m_modelVersion.number().patch;
        modelName    = m_modelVersion.model().c_str();
        return Status();
    }
};

} // namespace nrend
