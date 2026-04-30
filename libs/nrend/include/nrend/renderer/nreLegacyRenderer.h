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

#include <nrend/renderer/rendererImplementation.h>

namespace nrend {

class NRELegacyRenderer final : public NRendererImplementation {
    void* m_impl = nullptr;

    void release();

public:
    static constexpr char name[]                           = "nre-nerf-legacy";
    static constexpr ModelVersion::Number minVersionNumber = {2, 0, 0};
    static constexpr ModelVersion::Number maxVersionNumber = {999, 999, 999};

    NRELegacyRenderer(const nlohmann::json& rendererState, const Logger& logger);
    virtual ~NRELegacyRenderer() {
        release();
    }

    bool supportVersion(const ModelVersion& version,
                        RenderingParameters::RendererHints /*rendererHint*/,
                        RenderingParameters::OptFlags renderFlags) const override {
        const bool validVersion = version.is("nerf") &&
                                  (version.number() >= minVersionNumber) &&
                                  (version.number() < maxVersionNumber);
        return validVersion && !(renderFlags & RenderingParameters::OptDifferentiable);
    }

    Status initialize(const ModelVersion& version,
                      const nlohmann::json& modelState,
                      const RenderingParameters& renderParams) override;

    /// march the scene according to the given camera and composite the result into the given cuda arrays
    Status renderForward(const RenderParameters& params,
                         const tcnn::vec3* wordlRayOriginCudaPtr,
                         const tcnn::vec3* worldRayDirectionCudaPtr,
                         const TTimestamp* worldRayTimestampCudaPtr,
                         const tcnn::ivec2* sensorsIdsPtr,
                         const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                         const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                         const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                         uint32_t* instanceIdCudaPtr,
                         float* worldHitDistanceCudaPtr,
                         tcnn::vec3* worldHitNormalCudaPtr,
                         tcnn::vec4* radianceDensityCudaPtr,
                         void* extendedFeaturesCudaPtr,
                         void* sceneDataCudaPtr,
                         ForwardContext** forwardContext,
                         int cudaDeviceIndex,
                         cudaStream_t cudaStream) const override;
};

} // namespace nrend
