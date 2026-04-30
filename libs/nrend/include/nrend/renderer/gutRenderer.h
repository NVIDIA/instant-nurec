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

#include <nrend/renderer/grutRenderer.h>
#include <nrend/renderer/gutRendererParameters.h>

namespace nrend {

class GUTRenderer : public GRUTRenderer {

    enum KernelEntryPoints {
        PreProcessParticlesKernelEntryPoint,
        ProjectOnTilesKernelEntryPoint,
        ExpandTileProjectionsKernelEntryPoint,
        RenderLidarKernelEntryPoint,
        RenderCameraKernelEntryPoint,
        PrepareSceneKernelEntryPoint,
        NumForwardEntryPoints,
        RenderBackwardCameraKernelEntryPoint = NumForwardEntryPoints,
        RenderBackwardLidarKernelEntryPoint,
        ProjectBackwardKernelEntryPoint,
        PrepareSceneBackwardKernelEntryPoint,
        NumKernelEntryPoints
    };

    mutable uint32_t m_renderKernelIndex = 0; ///< index of the kernel in the kernel definitions table

protected:
    struct Settings {
        bool perRayFeatures         = false;
        bool globalZOrder           = false;
        uint32_t kBufferSize        = 16;
        bool tightOpacityBounding   = true;
        bool rectBounding           = true;
        bool tileCulling            = true;
        float nearClipDistance      = 0.2f;
        float farClipDistanceCamera = std::numeric_limits<float>::max();
        float farClipDistanceLidar  = std::numeric_limits<float>::max();
        bool nearFarZCulling        = true; ///< true for backward compatibility
        enum RenderMode {
            KBuffer,
            Splat,
            Undefined
        } renderMode               = KBuffer;
        bool enableWarpAtomicOptim = true;
        bool enableRayBasedCulling = true;
        tcnn::ivec2 cameraTileSize = tcnn::ivec2(GUTParameters::DefaultTiling::BlockX, GUTParameters::DefaultTiling::BlockY);
        tcnn::ivec2 lidarTileSize  = tcnn::ivec2(GUTParameters::DefaultTiling::BlockX, GUTParameters::DefaultTiling::BlockY);
    } m_settings;

    struct ProjectionSettings {
        int nRollingShutterIterations = 5;
        int dim                       = 3;
        float alpha                   = 1.0f;
        float beta                    = 2.f;
        float kappa                   = 0.f;
        float imageMarginFactor       = 0.1f;
        bool requireAllSigmaPoints    = true;
        float minProjectedRayRadius   = 0.5477225575051661f; // √0.3
    } m_projectionSettings;

    virtual void initializeSettings(const nlohmann::json& rendererState, const Logger& logger);

public:
    static constexpr char name[]                           = "3dgut-nrend";
    static constexpr ModelVersion::Number minVersionNumber = {0, 2, 234};
    static constexpr ModelVersion::Number maxVersionNumber = {999, 999, 999};

    GUTRenderer(const nlohmann::json& rendererState, const Logger& logger, bool defaultSettings = false);
    virtual ~GUTRenderer();

    bool supportVersion(const ModelVersion& version,
                        RenderingParameters::RendererHints rendererHint,
                        RenderingParameters::OptFlags renderFlags) const override {
        return ((rendererHint == RenderingParameters::RendererHints::RendererDefault) ||
                (rendererHint == RenderingParameters::RendererHints::RendererFastQuality)) &&
               GRUTRenderer::supportVersion(version, rendererHint, renderFlags);
    }

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

    Status renderBackward(const RenderParameters& params,
                          const tcnn::vec3* wordlRayOriginCudaPtr,
                          const tcnn::vec3* worldRayDirectionCudaPtr,
                          const nrend::TTimestamp* worldRayTimestampCudaPtr,
                          const tcnn::ivec2* sensorsIdsPtr,
                          const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                          const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                          const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                          uint32_t* instanceIdCudaPtr,
                          const float* worldHitDistanceCudaPtr,
                          const float* worldHitDistanceGradientCudaPtr,
                          const tcnn::vec3* worldHitNormalCudaPtr,
                          const tcnn::vec3* worldHitNormalGradientCudaPtr,
                          const tcnn::vec4* radianceDensityCudaPtr,
                          const tcnn::vec4* radianceDensityGradientCudaPtr,
                          const void* extendedFeaturesCudaPtr,
                          const void* extendedFeaturesGradientCudaPtr,
                          tcnn::vec3* wordlRayOriginGradientCudaPtr,
                          tcnn::vec3* worldRayDirectionGradientCudaPtr,
                          ForwardContext* forwardContext,
                          int cudaDeviceIndex,
                          cudaStream_t cudaStream) const override;

    Status prepareSceneForward(const RenderParameters& params,
                               const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                               const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                               const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                               void* sceneDensityCudaPtr,
                               void* sceneFeaturesCudaPtr,
                               void* sceneExtendedFeaturesCudaPtr,
                               void* sceneSensorExtendedFeaturesCudaPtr,
                               void* sceneDataCudaPtr,
                               uint32_t& sceneSize,
                               ForwardContext** forwardContext,
                               int cudaDeviceIndex,
                               cudaStream_t cudaStream) const override;

    Status prepareSceneBackward(const RenderParameters& params,
                                const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                                const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                const void* sceneFeaturesCudaPtr,
                                const void* sceneExtendedFeaturesCudaPtr,
                                const void* sceneSensorExtendedFeaturesCudaPtr,
                                const void* sceneDensityGradientCudaPtr,
                                const void* sceneFeaturesGradientCudaPtr,
                                const void* sceneExtendedFeaturesGradientCudaPtr,
                                const void* sceneSensorExtendedFeaturesGradientCudaPtr,
                                ForwardContext* forwardContext,
                                int cudaDeviceIndex,
                                cudaStream_t cudaStream) const override;

public:
    Status registerKernelDefinitions(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        const KernelDefinitionsTable& kernelDefinitionsTable,
        KernelOpts kernelOpts,
        const Logger& logger) const override;

    Status configureCompiledKernels(
        const std::vector<std::unique_ptr<RtcKernel>>& compiledKernels,
        KernelOpts kernelOpts,
        const Logger& logger) const override;

private:
    bool validVersionNumber(const ModelVersion::Number& versionNumber) const override {
        if (versionNumber <= ModelVersion::Number{0, 0, 0}) {
            return true; // no actual version number is available (e.g., when executed in sandbox) - assume valid
        }
        return versionNumber >= minVersionNumber && versionNumber < maxVersionNumber;
    }

    inline bool backwardProjectionEnabled() const {
        return m_settings.renderMode == Settings::Splat;
    }

    inline bool linearProjectionEnabled() const {
        // linear projection is enabled only for gsplat since 3dgrut repo does not implement it
        return m_settings.renderMode == Settings::Splat;
    }

    /// number of precomputed features to pass from projection to render
    int numPrecomputedFeatures(SensorType sensorType) const;

    static inline Settings::RenderMode parseRenderMode(const std::string& renderModeStr,
                                                       const Logger& logger) {
        if (renderModeStr == "kbuffer") {
            return Settings::KBuffer;
        } else if (renderModeStr == "splat") {
            return Settings::Splat;
        }
        LOG_ERROR(logger, "GUTRenderer : undefined render mode %s.", renderModeStr.c_str());
        return Settings::Undefined;
    }
};

} // namespace nrend
