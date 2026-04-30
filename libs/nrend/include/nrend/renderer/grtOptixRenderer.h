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
#include <nrend/utils/optix/optixAccelerationStructure.h>

#include <memory>

namespace nrend {
class GRTOptixRenderer : public GRUTRenderer {

protected:
    enum KernelEntryPoints {
        PreProcessParticlesKernelEntryPoint,
        BuildParticlesPrimitiveKernelEntryPoint,
        NumKernelEntryPoints
    };
    mutable uint32_t m_kernelIndex                 = 0; ///< index of the kernel in the kernel definitions table
    mutable uint32_t m_renderPipelineIndex         = 0; ///< index of the forward tracing pipeline in the kernel definitions table
    mutable uint32_t m_renderPipelineBwdIndex      = 0; ///< index of the backward tracing pipeline in the kernel definitions table
    mutable uint32_t m_renderLidarPipelineIndex    = 0; ///< index of the forward lidar tracing pipeline in the kernel definitions table
    mutable uint32_t m_renderLidarPipelineBwdIndex = 0; ///< index of the backward lidar tracing pipeline in the kernel definitions table

    mutable bool m_staticAS = false;                                                                    ///< whether to use static acceleration structures
    mutable std::vector<std::unique_ptr<std::vector<OptixAccelerationStructure>>> m_staticPerDevicesAS; ///< static acceleration structures per device

    struct Settings {
        enum PipelineType {
            Reference,
            ReferenceInstance,
            RejectionSampling,
            NumPipelineTypes
        };
        // The 3 pipelines are :
        // - pipelineType : pipeline type to use (during inference)
        // - pipelineTypeFwd : forward tracing pipeline (during optimization)
        // - pipelineTypeBwd : backward tracing pipeline (during optimization)
        PipelineType pipelineType    = ReferenceInstance;
        PipelineType pipelineTypeFwd = pipelineType;
        PipelineType pipelineTypeBwd = pipelineTypeFwd;
        uint32_t pipelineKBufferSize = 10;
        uint32_t pipelineNumSamples  = pipelineKBufferSize;
        enum PrimitiveType {
            Tetrahedra,
            Diamond,
            Octahedron,
            Icosahedron,
            Aabb,
            Trisurfel,
            Rhombus,
            Sphere,
            TransformedAabb,
            NumPrimitiveTypes
        } primitiveType                    = TransformedAabb;
        bool primitiveDensityScaleClamping = true;
        float nearClipDistance             = 0.2f;
        float farClipDistanceCamera        = std::numeric_limits<float>::max();
        float farClipDistanceLidar         = std::numeric_limits<float>::max();
    } m_settings;

    struct ParticlesPrimitiveInfos {
        uint32_t dataSizePerParticles              = 0;
        uint32_t extDataSizePerParticles           = 0;
        uint32_t numIndicesPerParticles            = 0;
        OptixAccelerationStructure::InputType type = OptixAccelerationStructure::InputType::NumInputTypes;
        uint32_t buildASFlag                       = OptixAccelerationStructure::DefaultBuild | OptixAccelerationStructure::FastTrace;
    } m_particlesPrimitiveInfos;

    virtual void initializeSettings(const nlohmann::json& rendererState, const Logger& logger);

public:
    static constexpr char name[]                           = "3dgrt-optix-nrend";
    static constexpr ModelVersion::Number minVersionNumber = {0, 2, 234}; ///< GRT may render GUT models
    static constexpr ModelVersion::Number maxVersionNumber = {999, 999, 999};

    GRTOptixRenderer(const nlohmann::json& rendererState, const Logger& logger, bool defaultSettings = false);
    virtual ~GRTOptixRenderer();

    bool supportVersion(const ModelVersion& version,
                        RenderingParameters::RendererHints rendererHint,
                        RenderingParameters::OptFlags renderFlags) const override {
        return ((rendererHint == RenderingParameters::RendererHints::RendererDefault) ||
                (rendererHint == RenderingParameters::RendererHints::RendererHighestQuality)) &&
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

public:
    Status registerKernelDefinitions(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        const KernelDefinitionsTable& kernelDefinitionsTable,
        KernelOpts kernelOpts,
        const Logger& logger) const override;

private:
    bool validVersionNumber(const ModelVersion::Number& versionNumber) const override {
        if (versionNumber <= ModelVersion::Number{0, 0, 0}) {
            return true; // no actual version number is available (e.g., when executed in sandboxed unit tests) - assume valid
        }
        return versionNumber >= minVersionNumber && versionNumber < maxVersionNumber;
    }

    Status setupOptixPipelineTracer(bool backward,
                                    bool lidar,
                                    OptixPipelineOptions& optixPipelineOptions,
                                    std::string& optixPipelineTracerSourceCode,
                                    KernelOpts kernelOpts,
                                    const Logger& logger) const;

    static inline Settings::PipelineType pipelineTypeFromStr(const std::string& pipelineTypeStr) {
        if (pipelineTypeStr == "reference") {
            return Settings::Reference;
        } else if (pipelineTypeStr == "reference_instance") {
            return Settings::ReferenceInstance;
        } else if (pipelineTypeStr == "rejection_sampling") {
            return Settings::RejectionSampling;
        } else {
            return Settings::NumPipelineTypes;
        }
    }

    static inline Settings::PrimitiveType primitiveTypeFromStr(const std::string& primitiveTypeStr) {
        if (primitiveTypeStr == "transformed_aabb") {
            return Settings::TransformedAabb;
        } else if (primitiveTypeStr == "tetrahedra") {
            return Settings::Tetrahedra;
        } else if (primitiveTypeStr == "diamond") {
            return Settings::Diamond;
        } else if (primitiveTypeStr == "octahedron") {
            return Settings::Octahedron;
        } else if (primitiveTypeStr == "icosahedron") {
            return Settings::Icosahedron;
        } else if (primitiveTypeStr == "aabb") {
            return Settings::Aabb;
        } else if (primitiveTypeStr == "trisurfel") {
            return Settings::Trisurfel;
        } else if (primitiveTypeStr == "rhombus") {
            return Settings::Rhombus;
        } else if (primitiveTypeStr == "sphere") {
            return Settings::Sphere;
        }
        return Settings::NumPrimitiveTypes;
    }

    static ParticlesPrimitiveInfos particlePrimitiveInfos(Settings::PrimitiveType);
    static std::string particlePrimitiveDefinition(Settings::PrimitiveType,
                                                   bool densityScaleClamping);
};

} // namespace nrend
