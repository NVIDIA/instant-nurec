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

#include <nrend/renderer/rendererImplementation.h>
#include <nrend/utils/cuda/cudaKernelResources.h>

namespace nrend {

class NREModel;

class GRUTRenderer : public NRendererImplementation, public KernelDefinitionsProvider {

protected:
    struct OutputSettings {
        bool enableFeatures               = true;
        bool enableExtendedFeatures       = true;
        bool enableSensorExtendedFeatures = true;
        bool enableNormals                = true;
        bool enableRayGradients           = false;
    };
    OutputSettings m_cameraOutputSettings;
    RenderingFeaturesLayout m_cameraRenderingFeaturesLayout;
    OutputSettings m_lidarOutputSettings;
    RenderingFeaturesLayout m_lidarRenderingFeaturesLayout;

    RenderingSceneDataLayout m_renderingSceneDataLayout;

    std::unique_ptr<NREModel> m_modelPtr;                       ///< model definition
    std::unique_ptr<CudaKernelResources> m_cudaKernelResources; ///< cached (mutable) cuda kernel resources

    KernelOpts m_optFlags = KernelOpts::None;

protected:
    void initializeOutputSettings(const nlohmann::json& rendererState, const Logger& logger);

public:
    GRUTRenderer(const nlohmann::json& rendererState, const Logger& logger);
    virtual ~GRUTRenderer() = default;

    virtual bool supportVersion(const ModelVersion& version,
                                RenderingParameters::RendererHints rendererHint,
                                RenderingParameters::OptFlags renderFlags) const override;

    virtual Status renderingFeaturesLayout(SensorType sensorType,
                                           RenderingFeaturesLayout& featuresLayout) const;

    virtual Status renderingSceneDataLayout(uint32_t& sceneDataSize,
                                            RenderingSceneDataLayout& sceneDataLayout) const;

    virtual Status sceneLayout(SensorType sensorType,
                               uint32_t& sceneSize,
                               uint32_t& sceneDensitySize,
                               uint32_t& featureSize,
                               uint32_t& extendedFeaturesSize,
                               uint32_t& sensorExtendedFeaturesSize,
                               bool& halfPrecision) const;

    virtual Status initialize(const ModelVersion& version,
                              const nlohmann::json& modelState,
                              const RenderingParameters& renderParams) override;

    virtual Status updateModelParameters(const NamedParameterDefinitionsSpan& namedParametersDefinition,
                                         bool gradients,
                                         bool copy,
                                         int cudaDeviceIndex,
                                         cudaStream_t cudaStream) override;

    virtual Status detachModelParameters(bool gradients, bool copy, int cudaDeviceIndex, cudaStream_t cudaStream) override;

public:
    Status processKernelMemory(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const override;

protected:
    virtual bool validVersionNumber(const ModelVersion::Number& versionNumber) const = 0;

    inline bool initialized() const {
        return m_cudaKernelResources.get();
    }
};

} // namespace nrend
