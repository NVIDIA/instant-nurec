// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/renderer/grutRenderer.h>

#include <nrend/models/nreGaussiansCompositeModel.h>
#include <nrend/models/nreGaussiansPrimitiveModel.h>
#include <nrend/models/nreShGaussianModel.h>

nrend::GRUTRenderer::GRUTRenderer(const nlohmann::json& rendererState, const Logger& logger)
    : NRendererImplementation(rendererState, logger) {
}

void nrend::GRUTRenderer::initializeOutputSettings(const nlohmann::json& rendererState, const Logger& logger) {
    if (!rendererState.empty()) {
        if (rendererState.contains("outputs")) {
            const auto& outputsSettingsConfig = rendererState["outputs"];
            if (outputsSettingsConfig.contains("camera")) {
                const auto& cameraOutputsSettingsConfig             = outputsSettingsConfig["camera"];
                m_cameraOutputSettings.enableFeatures               = cameraOutputsSettingsConfig.value("enable_features", m_cameraOutputSettings.enableFeatures);
                m_cameraOutputSettings.enableExtendedFeatures       = cameraOutputsSettingsConfig.value("enable_extended_features", m_cameraOutputSettings.enableExtendedFeatures);
                m_cameraOutputSettings.enableSensorExtendedFeatures = cameraOutputsSettingsConfig.value("enable_sensor_extended_features", m_cameraOutputSettings.enableSensorExtendedFeatures);
                m_cameraOutputSettings.enableNormals                = cameraOutputsSettingsConfig.value("enable_normals", m_cameraOutputSettings.enableNormals);
                m_cameraOutputSettings.enableRayGradients           = cameraOutputsSettingsConfig.value("enable_ray_gradients", m_cameraOutputSettings.enableRayGradients);
            }
            if (outputsSettingsConfig.contains("lidar")) {
                const auto& lidarOutputsSettingsConfig             = outputsSettingsConfig["lidar"];
                m_lidarOutputSettings.enableFeatures               = lidarOutputsSettingsConfig.value("enable_features", m_lidarOutputSettings.enableFeatures);
                m_lidarOutputSettings.enableExtendedFeatures       = lidarOutputsSettingsConfig.value("enable_extended_features", m_lidarOutputSettings.enableExtendedFeatures);
                m_lidarOutputSettings.enableSensorExtendedFeatures = lidarOutputsSettingsConfig.value("enable_sensor_extended_features", m_lidarOutputSettings.enableSensorExtendedFeatures);
                m_lidarOutputSettings.enableNormals                = lidarOutputsSettingsConfig.value("enable_normals", m_lidarOutputSettings.enableNormals);
                m_lidarOutputSettings.enableRayGradients           = lidarOutputsSettingsConfig.value("enable_ray_gradients", m_lidarOutputSettings.enableRayGradients);
            }
            LOG_INFO(logger, "GRUTRenderer : camera output settings : %s | %s | %s | %s | %s",
                     m_cameraOutputSettings.enableFeatures ? "Features ON" : "Features OFF",
                     m_cameraOutputSettings.enableExtendedFeatures ? "Extended Features ON" : "Extended Features OFF",
                     m_cameraOutputSettings.enableSensorExtendedFeatures ? "Sensor Extended Features ON" : "Sensor Extended Features OFF",
                     m_cameraOutputSettings.enableNormals ? "Normals ON" : "Normals OFF",
                     m_cameraOutputSettings.enableRayGradients ? "Ray Gradients ON" : "Ray Gradients OFF");
            LOG_INFO(logger, "GRUTRenderer : lidar output settings : %s | %s | %s | %s | %s",
                     m_lidarOutputSettings.enableFeatures ? "Features ON" : "Features OFF",
                     m_lidarOutputSettings.enableExtendedFeatures ? "Extended Features ON" : "Extended Features OFF",
                     m_lidarOutputSettings.enableSensorExtendedFeatures ? "Sensor Extended Features ON" : "Sensor Extended Features OFF",
                     m_lidarOutputSettings.enableNormals ? "Normals ON" : "Normals OFF",
                     m_lidarOutputSettings.enableRayGradients ? "Ray Gradients ON" : "Ray Gradients OFF");
        }
    }
}

bool nrend::GRUTRenderer::supportVersion(const ModelVersion& version,
                                         RenderingParameters::RendererHints /*rendererHint*/,
                                         RenderingParameters::OptFlags renderFlags) const {
    const bool supportedModelInstance =
        version.isInstance(NREGaussiansPrimitiveModel::name) ||
        version.isInstance(NRESHGaussianModel::name) ||
        version.isInstance(NREGaussiansCompositeModel::name);
    return supportedModelInstance && version.is("nre") && validVersionNumber(version.number());
}

nrend::Status nrend::GRUTRenderer::initialize(const ModelVersion& version,
                                              const nlohmann::json& modelState,
                                              const RenderingParameters& renderParams) {
    if (!supportVersion(version, renderParams.rendererHint, renderParams.opts)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GRUTRenderer : unsupported model version %s.", version.str().c_str());
    }

    m_modelVersion = version;
    m_optFlags     = static_cast<KernelOpts>((renderParams.opts & RenderingParameters::OptDifferentiable ? KernelOpts::Differentiable : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptLinearRGB ? KernelOpts::LinearRGB : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptDisableFeatures ? KernelOpts::DisableFeatures : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptDisableExtendedFeatures ? KernelOpts::DisableExtendedFeatures : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptDisableSensorExtendedFeatures ? KernelOpts::DisableSensorExtendedFeatures : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptDisableBackground ? KernelOpts::DisableBackground : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptDisablePostProcessings ? KernelOpts::DisablePostProcessings : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptDisableNormals ? KernelOpts::DisableNormals : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptDisableRayGradients ? KernelOpts::DisableRayGradients : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptEnableParticleCumulatedWeights ? KernelOpts::EnableCumulatedWeights : KernelOpts::None) |
                                         (renderParams.opts & RenderingParameters::OptEnableParticleVisibility ? KernelOpts::EnableVisibility : KernelOpts::None));

    if (!modelState.contains("nre_data")) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GRUTRenderer : cannot create renderer from JSON : no nre_data header.");
    }

    if (modelState["nre_data"].contains("config") && modelState["nre_data"].contains("state_dict")) {
        auto modelPtr = NREModel::createFromJSON(modelState["nre_data"]["config"], m_logger, modelState["nre_data"]["state_dict"], ".");
        if (!dynamic_cast<INREParticlesModel*>(modelPtr)) {
            delete modelPtr;
            RETURN_ERROR(m_logger, ErrorCode::BadInput, "GRUTRenderer : cannot create renderer from JSON.");
        }
        m_modelPtr.reset(modelPtr);
        if (renderParams.trackInstancesStrUIds.size) {
            m_modelPtr->initializeTrackInstances(renderParams.trackInstancesStrUIds, m_logger);
        }
        const NREModel::FeaturesLayout nreFeaturesLayout = m_modelPtr->featuresLayout();
        m_cameraRenderingFeaturesLayout                  = RenderingFeaturesLayout{
            !(m_optFlags & KernelOpts::DisableFeatures) && m_cameraOutputSettings.enableFeatures ? nreFeaturesLayout.baseFeaturesDim : 0,
            !(m_optFlags & KernelOpts::DisableExtendedFeatures) && m_cameraOutputSettings.enableExtendedFeatures ? nreFeaturesLayout.extendedFeaturesDim : 0,
            !(m_optFlags & KernelOpts::DisableSensorExtendedFeatures) && m_cameraOutputSettings.enableSensorExtendedFeatures ? nreFeaturesLayout.cameraExtendedFeaturesDim : 0,
            !(m_optFlags & KernelOpts::DisableNormals) && m_cameraOutputSettings.enableNormals,
            !(m_optFlags & KernelOpts::DisableRayGradients) && m_cameraOutputSettings.enableRayGradients};
        LOG_DEBUG(m_logger, "GRUTRenderer : camera rendering layout : Features = %d, Extended Features = %d, Sensor Extended Features = %d, Normals = %s, Ray Gradients = %s",
                  m_cameraRenderingFeaturesLayout.baseFeaturesDim,
                  m_cameraRenderingFeaturesLayout.extendedFeaturesDim,
                  m_cameraRenderingFeaturesLayout.sensorExtendedFeaturesDim,
                  (m_cameraRenderingFeaturesLayout.computeNormals ? "ON" : "OFF"),
                  (m_cameraRenderingFeaturesLayout.computeRayGradients ? "ON" : "OFF"));
        m_lidarRenderingFeaturesLayout = RenderingFeaturesLayout{
            !(m_optFlags & KernelOpts::DisableFeatures) && m_lidarOutputSettings.enableFeatures ? nreFeaturesLayout.baseFeaturesDim : 0,
            !(m_optFlags & KernelOpts::DisableExtendedFeatures) && m_lidarOutputSettings.enableExtendedFeatures ? nreFeaturesLayout.extendedFeaturesDim : 0,
            !(m_optFlags & KernelOpts::DisableSensorExtendedFeatures) && m_lidarOutputSettings.enableSensorExtendedFeatures ? nreFeaturesLayout.lidarExtendedFeaturesDim : 0,
            !(m_optFlags & KernelOpts::DisableNormals) && m_lidarOutputSettings.enableNormals,
            !(m_optFlags & KernelOpts::DisableRayGradients) && m_lidarOutputSettings.enableRayGradients};
        LOG_DEBUG(m_logger, "GRUTRenderer : lidar rendering layout : Features = %d, Extended Features = %d, Sensor Extended Features = %d, Normals = %s, Ray Gradients = %s",
                  m_lidarRenderingFeaturesLayout.baseFeaturesDim,
                  m_lidarRenderingFeaturesLayout.extendedFeaturesDim,
                  m_lidarRenderingFeaturesLayout.sensorExtendedFeaturesDim,
                  (m_lidarRenderingFeaturesLayout.computeNormals ? "ON" : "OFF"),
                  (m_lidarRenderingFeaturesLayout.computeRayGradients ? "ON" : "OFF"));

        // Initialize scene data layout
        const bool enableCumulatedWeights = m_optFlags & KernelOpts::EnableCumulatedWeights;
        const bool enableVisibility       = m_optFlags & KernelOpts::EnableVisibility;
        m_renderingSceneDataLayout        = RenderingSceneDataLayout{};
        m_renderingSceneDataLayout.cumulatedWeights =
            enableCumulatedWeights ? RenderingSceneDataLayout::Span{m_renderingSceneDataLayout.count(), 1} : RenderingSceneDataLayout::Span{};
        m_renderingSceneDataLayout.visibility =
            enableVisibility ? RenderingSceneDataLayout::Span{m_renderingSceneDataLayout.count(), 1} : RenderingSceneDataLayout::Span{};
        LOG_DEBUG(m_logger,
                  "GRUTRenderer : scene data layout : Cumulated Weights = %s, Visibility = %s",
                  enableCumulatedWeights ? "ON" : "OFF",
                  enableVisibility ? "ON" : "OFF");
    } else {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GRUTRenderer : cannot create renderer from JSON : no config or state_dict entries.");
    }

    m_cudaKernelResources = std::make_unique<CudaKernelResources>();

    return Status();
}

nrend::Status nrend::GRUTRenderer::processKernelMemory(
    const KernelMemoryBindings& memoryBindings,
    KernelMemoryBindings::BindingsFlag bindingsFlag,
    const std::vector<std::unique_ptr<KernelMemory>>& memory,
    ProcessMemoryFlag processFlag,
    uint64_t processQueueHandle,
    const Logger& logger) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRUTRenderer : cannot process kernel memory, not initialized.");
    }

    if (!m_modelPtr) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRUTRenderer : cannot process kernel memory, uninitialized model.");
    }
    CHECK_STATUS_RETURN(m_modelPtr->processKernelMemory(memoryBindings, bindingsFlag, memory, processFlag, processQueueHandle, logger));

    return Status();
}

nrend::Status nrend::GRUTRenderer::updateModelParameters(const NamedParameterDefinitionsSpan& namedParametersDefinitions,
                                                         bool gradients,
                                                         bool copy,
                                                         int cudaDeviceIndex,
                                                         cudaStream_t cudaStream) {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRUTRenderer : cannot update model parameters, not initialized.");
    }

    auto cudaResource = m_cudaKernelResources->update(
        this,
        m_optFlags,
        cudaStream,
        cudaDeviceIndex,
        namedParametersDefinitions,
        gradients ? KernelMemoryBindings::ParameterGradients : KernelMemoryBindings::Parameters,
        !copy,
        m_logger);
    if (!cudaResource) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRUTRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }
    // FIXME : resources concurrent management
    // const long resourcesPtrUseCount = cudaResource->resourcesUseCount();
    // if ((!gradients && resourcesPtrUseCount > 2) || (resourcesPtrUseCount > 3)) {
    //     RETURN_ERROR(m_logger, ErrorCode::InvalidResource,
    //                "GUTRenderer : already locked resources (%d) have been updated on the device %d.",
    //                 resourcesPtrUseCount, cudaDeviceIndex);
    // }
    return Status();
}

nrend::Status nrend::GRUTRenderer::detachModelParameters(bool gradients, bool copy, int cudaDeviceIndex, cudaStream_t cudaStream) {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRUTRenderer : cannot detach model parameters, not initialized.");
    }

    return m_cudaKernelResources->detach(
        this,
        cudaStream,
        cudaDeviceIndex,
        gradients ? KernelMemoryBindings::ParameterGradients : KernelMemoryBindings::Parameters,
        copy,
        m_logger);
}

nrend::Status nrend::GRUTRenderer::renderingFeaturesLayout(SensorType sensorType,
                                                           RenderingFeaturesLayout& featuresLayout) const {
    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRUTRenderer : cannot get rendering features layout, not initialized.");
    }
    if (sensorType == SensorType::Camera) {
        featuresLayout = m_cameraRenderingFeaturesLayout;
    } else {
        featuresLayout = m_lidarRenderingFeaturesLayout;
    }
    return Status();
}

nrend::Status nrend::GRUTRenderer::renderingSceneDataLayout(uint32_t& sceneDataSize,
                                                            RenderingSceneDataLayout& sceneDataLayout) const {
    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRUTRenderer : cannot get rendering scene data layout, not initialized.");
    }
    INREParticlesModel* particlesModelPtr = dynamic_cast<INREParticlesModel*>(m_modelPtr.get());
    sceneDataSize                         = particlesModelPtr ? particlesModelPtr->numParticles() : 0;
    sceneDataLayout                       = m_renderingSceneDataLayout;
    return Status();
}

nrend::Status nrend::GRUTRenderer::sceneLayout(SensorType sensorType,
                                               uint32_t& sceneSize,
                                               uint32_t& sceneDensitySize,
                                               uint32_t& featureSize,
                                               uint32_t& extendedFeaturesSize,
                                               uint32_t& sensorExtendedFeaturesSize,
                                               bool& halfPrecision) const {
    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GRUTRenderer : cannot get scene layout, not initialized.");
    }

    INREParticlesModel* particlesModelPtr = dynamic_cast<INREParticlesModel*>(m_modelPtr.get());
    sceneSize                             = particlesModelPtr ? particlesModelPtr->numParticles() : 0;

    // Size of the density parameters is currently hardcoded
    // 12 floats (3 for position, 1 for density, 4 for quaternion, 3 for scale, 1 for padding)
    // TODO : find a way to sync with the particle model
    sceneDensitySize = 12;

    if (sensorType == SensorType::Camera) {
        featureSize                = m_cameraRenderingFeaturesLayout.baseFeaturesDim;
        extendedFeaturesSize       = m_cameraRenderingFeaturesLayout.extendedFeaturesDim;
        sensorExtendedFeaturesSize = m_cameraRenderingFeaturesLayout.sensorExtendedFeaturesDim;
        halfPrecision              = false;
    } else {
        featureSize                = m_lidarRenderingFeaturesLayout.baseFeaturesDim;
        extendedFeaturesSize       = m_lidarRenderingFeaturesLayout.extendedFeaturesDim;
        sensorExtendedFeaturesSize = m_lidarRenderingFeaturesLayout.sensorExtendedFeaturesDim;
        halfPrecision              = false;
    }

    return Status();
}