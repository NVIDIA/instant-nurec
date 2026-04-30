// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/models/nreParticlesModel.h>
#include <nrend/renderer/gutRenderer.h>
#include <nrend/renderer/gutRendererParameters.h>
#include <nrend/sensors/sensors.h>
#include <nrend/utils/deviceLaunchesLogger.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

using namespace tcnn;

namespace {

using namespace nrend;

static inline tcnn::ivec2 sensorTileGridResolution(const SensorProjectionModel& sensorModel, const tcnn::ivec2& tileSize, const tcnn::ivec2& resolution) {
    if (sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
        return tcnn::ivec2{tileSize.x * sensorModel.nreHesaiP128LidarParams.elevationNBins,
                           tileSize.y * sensorModel.nreHesaiP128LidarParams.azimuthNBins};
    }
    return tcnn::ivec2{resolution.x, resolution.y};
}

// identify tiles start/end indices in the sorted tile/depth keys buffer
__global__ void computeSortedTileRangeIndices(
    int numKeys,
    const uint64_t* __restrict__ sortedTileDepthKeys,
    uvec2* __restrict__ tileRangeIndices) {

    const int keyIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (keyIdx >= numKeys) {
        return;
    }

    const uint32_t tileIdx = sortedTileDepthKeys[keyIdx] >> 32;
    const bool validTile   = tileIdx != GUTParameters::InvalidTileIdx;
    if (keyIdx == 0) {
        if (validTile) {
            tileRangeIndices[tileIdx].x = keyIdx;
        }
    } else {
        const uint32_t prevKeyTileIdx = sortedTileDepthKeys[keyIdx - 1] >> 32;
        if (prevKeyTileIdx != tileIdx) {
            if (prevKeyTileIdx != GUTParameters::InvalidTileIdx) {
                tileRangeIndices[prevKeyTileIdx].y = keyIdx;
            }
            if (validTile) {
                tileRangeIndices[tileIdx].x = keyIdx;
            }
        }
    }
    if (validTile && (keyIdx == numKeys - 1)) {
        tileRangeIndices[tileIdx].y = numKeys;
    }
}

inline uint32_t getMSBExclusive(uint32_t x) {
#if defined(_MSC_VER)
    unsigned long index;
    if (_BitScanReverse(&index, x)) {
        return index + 1; // Make it exclusive by adding 1
    }
    return 0; // Handle x == 0 case
#else
    if (x == 0)
        return 0;
    return 32 - __builtin_clz(x); // Already exclusive
#endif
}

// TODO : per-stream n context cache
struct GutRenderForwardContext final : public NRenderer::ForwardContext {

    uint64_t queueHandle = 0u;

    ScopedCudaBuffer unsortedTileDepthKeys;
    ScopedCudaBuffer sortedTileDepthKeys;
    ScopedCudaBuffer sortedTileRangeIndices;
    ScopedCudaBuffer unsortedTileParticleIdx;
    ScopedCudaBuffer sortedTileParticleIdx;
    ScopedCudaBuffer sortingWorkingBuffer;

    ScopedCudaBuffer parameterMemoryHandles;
    std::vector<KernelBindedTransientMemory> transientParameters;

    uint32_t numParticles                 = 0u;
    uint32_t numParticleTileIntersections = 0u; ///< number of particle/tile intersections

    ScopedCudaBuffer particlesTilesCount;                            ///< number of intersected tiles per particles uint32_t [Nx1]
    ScopedCudaBuffer particlesTilesOffset;                           ///< cumulative sum of particle tiles count uint32_t [Nx1]
    ScopedCudaBuffer particlesProjectedPosition;                     ///< projected particles center Nx2
    ScopedCudaBuffer particlesProjectedConicOpacity;                 ///< projected particles conic and opacity Nx4
    ScopedCudaBuffer particlesProjectedExtent;                       ///< projected particles extent Nx2
    ScopedCudaBuffer particlesGlobalDepth;                           ///< particles global depth
    ScopedCudaBuffer particlesPrecomputedFeatures;                   ///< precomputed particle features float [NxFeaturesDim]
    mutable ScopedCudaBuffer particlesProjectedPositionGradient;     ///< projected particles center Nx2
    mutable ScopedCudaBuffer particlesProjectedConicOpacityGradient; ///< projected particles conic and opacity Nx4
    mutable ScopedCudaBuffer particlesGlobalDepthGradient;           ///< particles global depth
    mutable ScopedCudaBuffer particlesPrecomputedFeaturesGradient;   ///< precomputed particle features float [NxFeaturesDim]
    ScopedCudaBuffer scanningWorkingBuffer;                          ///< working buffer to compute the cumulative sum of particles/tiles intersections number

    // clang-format off
    GutRenderForwardContext(uint64_t queueHandle)
        : queueHandle(queueHandle),
          unsortedTileDepthKeys(queueHandle),
          sortedTileDepthKeys(queueHandle),
          sortedTileRangeIndices(queueHandle),
          unsortedTileParticleIdx(queueHandle),
          sortedTileParticleIdx(queueHandle),
          sortingWorkingBuffer(queueHandle),
          parameterMemoryHandles(queueHandle),
          particlesTilesCount(queueHandle),
          particlesTilesOffset(queueHandle),
          particlesProjectedPosition(queueHandle),
          particlesProjectedConicOpacity(queueHandle),
          particlesProjectedExtent(queueHandle),
          particlesGlobalDepth(queueHandle),
          particlesPrecomputedFeatures(queueHandle),
          particlesProjectedPositionGradient(queueHandle),
          particlesProjectedConicOpacityGradient(queueHandle),
          particlesGlobalDepthGradient(queueHandle),
          particlesPrecomputedFeaturesGradient(queueHandle),
          scanningWorkingBuffer(queueHandle) {}
    // clang-format on
    ~GutRenderForwardContext() = default;

    Status updateTileSortingBuffers(
        const ivec2& tileGrid,
        int numKeys,
        cudaStream_t stream,
        const Logger& logger) {

        RETURN_ERROR_IF(queueHandle != reinterpret_cast<uint64_t>(stream), logger,
                        ErrorCode::BadInput, "GUTRenderer : invalid queue handle.");

        const bool uptodate = unsortedTileDepthKeys.size() >= sizeof(uint64_t) * numKeys;
        if (!uptodate) {
            CHECK_STATUS_RETURN(unsortedTileDepthKeys.resize(sizeof(uint64_t) * numKeys, logger));
            CHECK_STATUS_RETURN(sortedTileDepthKeys.resize(sizeof(uint64_t) * numKeys, logger));
            CHECK_STATUS_RETURN(unsortedTileParticleIdx.resize(sizeof(uint32_t) * numKeys, logger));
            CHECK_STATUS_RETURN(sortedTileParticleIdx.resize(sizeof(uint32_t) * numKeys, logger));
            size_t sortingWorkingBufferSize = 0;
            CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(nullptr,
                                                              sortingWorkingBufferSize,
                                                              static_cast<const uint64_t*>(unsortedTileDepthKeys.data()),
                                                              static_cast<uint64_t*>(sortedTileDepthKeys.data()),
                                                              static_cast<const uint32_t*>(unsortedTileParticleIdx.data()),
                                                              static_cast<uint32_t*>(sortedTileParticleIdx.data()),
                                                              numKeys,
                                                              0, 32 + getMSBExclusive(tileGrid.x * tileGrid.y),
                                                              stream),
                              logger);
            CHECK_STATUS_RETURN(sortingWorkingBuffer.resize(sortingWorkingBufferSize, logger));
        }
        if (numKeys) {
            CHECK_STATUS_RETURN(sortedTileRangeIndices.resize(sizeof(uvec2) * tileGrid.x * tileGrid.y, logger));
            CUDA_CHECK_RETURN(cudaMemsetAsync(sortedTileRangeIndices.data(), 0, tileGrid.x * tileGrid.y * sizeof(uvec2), stream), logger);
        }
        return Status();
    }

    inline Status updateParameterMemoryHandlesBuffer(
        const KernelMemoryPtrVec& parameterMemoryPtrVec,
        const Logger& logger) {
        std::vector<uint64_t> parameterMemoryHandlesVec(parameterMemoryPtrVec.size());
        std::transform(parameterMemoryPtrVec.begin(), parameterMemoryPtrVec.end(), parameterMemoryHandlesVec.begin(),
                       [](const KernelMemoryPtr& ptr) { return ptr ? ptr->handle() : 0; });
        for (const auto& transientParameter : transientParameters) {
            RETURN_ERROR_IF((transientParameter.memoryBindingIndex == KernelMemoryBindings::InvalidMemoryIndex) || !transientParameter.memory,
                            logger,
                            ErrorCode::BadInput,
                            "GUTRenderer : cannot update parameter memory handles buffer, transient parameter memory binding index is invalid.");
            parameterMemoryHandlesVec[transientParameter.memoryBindingIndex] = transientParameter.memory->handle();
        }
        return parameterMemoryHandles.setFromHost(parameterMemoryHandlesVec.data(), parameterMemoryHandlesVec.size() * sizeof(uint64_t), logger);
    }

    inline Status updateParticlesWorkingBuffers(
        int numParticles,
        cudaStream_t cudaStream,
        const Logger& logger) {

        RETURN_ERROR_IF(queueHandle != reinterpret_cast<uint64_t>(cudaStream), logger,
                        ErrorCode::BadInput, "GUTRenderer : invalid queue handle.");

        const bool uptodate = particlesTilesCount.size() >= numParticles * sizeof(uint32_t);
        if (!uptodate) {
            CHECK_STATUS_RETURN(particlesTilesCount.resize(numParticles * sizeof(uint32_t),
                                                           logger));
            CHECK_STATUS_RETURN(particlesTilesOffset.resize(numParticles * sizeof(uint32_t),
                                                            logger));
            size_t scanningWorkingBufferSize = 0;
            CUDA_CHECK_RETURN(
                cub::DeviceScan::InclusiveSum(
                    nullptr,
                    scanningWorkingBufferSize,
                    static_cast<const uint32_t*>(particlesTilesCount.data()),
                    static_cast<uint32_t*>(particlesTilesOffset.data()),
                    numParticles,
                    cudaStream),
                logger);

            CHECK_STATUS_RETURN(scanningWorkingBuffer.resize(
                scanningWorkingBufferSize,
                logger));

            CHECK_STATUS_RETURN(particlesProjectedPosition.resize(
                numParticles * sizeof(vec2),
                logger));
            CHECK_STATUS_RETURN(particlesProjectedConicOpacity.resize(
                numParticles * sizeof(vec4),
                logger));
            CHECK_STATUS_RETURN(particlesProjectedExtent.resize(
                numParticles * sizeof(ivec2),
                logger));
            CHECK_STATUS_RETURN(particlesGlobalDepth.resize(
                numParticles * sizeof(float),
                logger));
        }

        return Status();
    }

    inline Status updateParticlesProjectionGradientBuffers(
        int numParticles,
        cudaStream_t cudaStream,
        const Logger& logger) const {

        RETURN_ERROR_IF(queueHandle != reinterpret_cast<uint64_t>(cudaStream), logger,
                        ErrorCode::BadInput, "GUTRenderer : invalid queue handle.");

        CHECK_STATUS_RETURN(particlesProjectedPositionGradient.enlarge(
            numParticles * sizeof(vec2),
            logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesProjectedPositionGradient.data(), 0, numParticles * sizeof(vec2), cudaStream), logger);

        CHECK_STATUS_RETURN(particlesProjectedConicOpacityGradient.resize(
            numParticles * sizeof(vec4),
            logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesProjectedConicOpacityGradient.data(), 0, numParticles * sizeof(vec4), cudaStream), logger);

        CHECK_STATUS_RETURN(particlesGlobalDepthGradient.resize(
            numParticles * sizeof(float),
            logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesGlobalDepthGradient.data(), 0, numParticles * sizeof(float), cudaStream), logger);

        return Status();
    }

    inline Status
    updateParticlesFeaturesBuffer(
        int featuresSize,
        cudaStream_t /*cudaStream*/,
        const Logger& logger) {

        CHECK_STATUS_RETURN(particlesPrecomputedFeatures.enlarge(
            featuresSize * sizeof(float),
            logger));

        return Status();
    }

    inline Status updateParticlesFeaturesGradientBuffer(
        uint32_t featuresSize,
        cudaStream_t cudaStream,
        const Logger& logger) const {

        RETURN_ERROR_IF(queueHandle != reinterpret_cast<uint64_t>(cudaStream), logger,
                        ErrorCode::BadInput, "GUTRenderer : invalid queue handle.");

        const size_t newSize = featuresSize * sizeof(float);

        CHECK_STATUS_RETURN(particlesPrecomputedFeaturesGradient.enlarge(
            newSize,
            logger));

        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesPrecomputedFeaturesGradient.data(), 0, newSize, cudaStream), logger);

        return Status();
    }
};

struct GutPrepareSceneForwardContext final : public NRenderer::ForwardContext {

    uint64_t queueHandle = 0u;

    ScopedCudaBuffer validParticleCountIdx;
    uint32_t numValidParticles = 0u;

    GutPrepareSceneForwardContext(uint64_t queueHandle)
        : queueHandle(queueHandle), validParticleCountIdx(queueHandle) {}
    ~GutPrepareSceneForwardContext() = default;

    Status initialize(
        uint32_t numParticles,
        cudaStream_t stream,
        const Logger& logger) {

        RETURN_ERROR_IF(queueHandle != reinterpret_cast<uint64_t>(stream), logger,
                        ErrorCode::BadInput, "GUTRenderer : invalid queue handle.");

        // + 1 for the total number of valid particles stored in the first element
        CHECK_STATUS_RETURN(validParticleCountIdx.resize((numParticles + 1) * sizeof(uint32_t), logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(validParticleCountIdx.data(), 0, sizeof(uint32_t), stream), logger);

        return Status();
    }
};

} // namespace

nrend::GUTRenderer::GUTRenderer(const nlohmann::json& rendererState, const Logger& logger, bool defaultSettings)
    : GRUTRenderer(rendererState, logger) {
    if (!defaultSettings) {
        initializeSettings(rendererState, logger);
    }
}

void nrend::GUTRenderer::initializeSettings(const nlohmann::json& rendererState, const Logger& logger) {
    if (!rendererState.empty()) {
        initializeOutputSettings(rendererState, logger);
        m_settings.perRayFeatures = rendererState.value("per_ray_features", m_settings.perRayFeatures);
        m_settings.globalZOrder   = rendererState.value("global_z_order", m_settings.globalZOrder);
        if (rendererState.contains("culling")) {
            const auto& cullingSettingsConfig = rendererState["culling"];
            m_settings.tightOpacityBounding   = cullingSettingsConfig.value("tight_opacity_bounding", m_settings.tightOpacityBounding);
            m_settings.rectBounding           = cullingSettingsConfig.value("rect_bounding", m_settings.rectBounding);
            m_settings.tileCulling            = cullingSettingsConfig.value("tile_based", m_settings.tileCulling);
            m_settings.nearClipDistance       = cullingSettingsConfig.value("near_clip_distance", m_settings.nearClipDistance);

            // Load separate camera and lidar far clip distances with backward compatibility
            if (cullingSettingsConfig.contains("far_clip_distance_camera") || cullingSettingsConfig.contains("far_clip_distance_lidar")) {
                m_settings.farClipDistanceCamera = cullingSettingsConfig.value("far_clip_distance_camera", m_settings.farClipDistanceCamera);
                m_settings.farClipDistanceLidar  = cullingSettingsConfig.value("far_clip_distance_lidar", m_settings.farClipDistanceLidar);
            } else if (cullingSettingsConfig.contains("far_clip_distance")) {
                // Backward compatibility: use the same value for both camera and lidar
                float farClipDistance            = cullingSettingsConfig.value("far_clip_distance", std::numeric_limits<float>::max());
                m_settings.farClipDistanceCamera = farClipDistance;
                m_settings.farClipDistanceLidar  = farClipDistance;
            }

            m_settings.nearFarZCulling       = cullingSettingsConfig.value("near_far_z_culling", m_settings.nearFarZCulling);
            m_settings.enableRayBasedCulling = cullingSettingsConfig.value("enable_ray_based_culling", m_settings.enableRayBasedCulling);
        }
        if (rendererState.contains("projection")) {
            const auto& projectionSettingsConfig           = rendererState["projection"];
            m_projectionSettings.nRollingShutterIterations = projectionSettingsConfig.value("n_rolling_shutter_iterations", m_projectionSettings.nRollingShutterIterations);
            m_projectionSettings.dim                       = projectionSettingsConfig.value("ut_dim", m_projectionSettings.dim);
            m_projectionSettings.alpha                     = projectionSettingsConfig.value("ut_alpha", m_projectionSettings.alpha);
            m_projectionSettings.beta                      = projectionSettingsConfig.value("ut_beta", m_projectionSettings.beta);
            m_projectionSettings.kappa                     = projectionSettingsConfig.value("ut_kappa", m_projectionSettings.kappa);
            m_projectionSettings.imageMarginFactor         = projectionSettingsConfig.value("image_margin_factor", m_projectionSettings.imageMarginFactor);
            m_projectionSettings.requireAllSigmaPoints     = projectionSettingsConfig.value("ut_require_all_sigma_points", m_projectionSettings.requireAllSigmaPoints);
            m_projectionSettings.minProjectedRayRadius     = projectionSettingsConfig.value("min_projected_ray_radius", m_projectionSettings.minProjectedRayRadius);
        }
        if (rendererState.contains("render")) {
            const auto& renderSettingsConfig = rendererState["render"];
            m_settings.renderMode            = parseRenderMode(renderSettingsConfig.value<std::string>("mode", "kbuffer"), logger);
            m_settings.kBufferSize           = renderSettingsConfig.value("k_buffer_size", m_settings.kBufferSize);
            m_settings.enableWarpAtomicOptim = renderSettingsConfig.value("enable_warp_atomic_optim", m_settings.enableWarpAtomicOptim);
        }
        if (rendererState.contains("tiling")) {
            const auto& tilingConfig = rendererState["tiling"];
            if (tilingConfig.contains("camera")) {
                const auto& camera        = tilingConfig["camera"];
                m_settings.cameraTileSize = tcnn::ivec2{camera.value("tile_width", m_settings.cameraTileSize.x), camera.value("tile_height", m_settings.cameraTileSize.y)};
                LOG_INFO(logger, "Camera tiling size: %dx%d", m_settings.cameraTileSize.x, m_settings.cameraTileSize.y);
            }
            if (tilingConfig.contains("lidar")) {
                const auto& lidar        = tilingConfig["lidar"];
                m_settings.lidarTileSize = tcnn::ivec2{lidar.value("tile_size_elevation", m_settings.lidarTileSize.x), lidar.value("tile_size_azimuth", m_settings.lidarTileSize.y)};
                LOG_INFO(logger, "LiDAR tiling size: %dx%d", m_settings.lidarTileSize.x, m_settings.lidarTileSize.y);
            }
        }
    }
}

nrend::GUTRenderer::~GUTRenderer() {}

nrend::Status nrend::GUTRenderer::configureCompiledKernels(
    const std::vector<std::unique_ptr<RtcKernel>>& compiledKernels,
    KernelOpts kernelOpts,
    const Logger& logger) const {

    if (m_renderKernelIndex >= compiledKernels.size()) {
        return Status(ErrorCode::BadInput, "GUTRenderer : invalid render kernel index.");
    }

    CudaRtcKernel* cudaRtcKernelPtr = dynamic_cast<CudaRtcKernel*>(compiledKernels[m_renderKernelIndex].get());
    if (!cudaRtcKernelPtr) {
        return Status(ErrorCode::InvalidResource, "GUTRenderer : invalid CUDA kernel.");
    }

    // NOTE: This exhibits different behaviors depending on different architectures
    // it is safer to make it no-op for now but keep the supporting code to allow
    // post-compile config.
    // E.G. on A40, A6000 we see massive improvements by favoring shared-mem over
    // L1 and simultaneously decreasing register allocations which launch_bounds.
    // On H100 we see a different behaviors for lidar and camera with an overall
    // nullified result.
    // Here's how one would adjust the memory profile configurations
    // if (kernelOpts & KernelOpts::Differentiable && m_settings.renderMode == Settings::KBuffer) {
    //     CHECK_STATUS_RETURN(cudaRtcKernelPtr->setKernelCacheConfig(
    //         RenderBackwardCameraKernelEntryPoint,
    //         CU_FUNC_CACHE_PREFER_SHARED,
    //         logger));
    //     CHECK_STATUS_RETURN(cudaRtcKernelPtr->setKernelCacheConfig(
    //         RenderBackwardLidarKernelEntryPoint,
    //         CU_FUNC_CACHE_PREFER_SHARED,
    //         logger));
    // }

    return Status();
}

nrend::Status nrend::GUTRenderer::registerKernelDefinitions(
    const KernelMemoryBindings& memoryBindings,
    const KernelSourceCodeTable& sourceCodeTable,
    const KernelDefinitionsTable& kernelDefinitionsTable,
    KernelOpts kernelOpts,
    const Logger& logger) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot register resource, not initialized.");
    }

    if (kernelOpts != m_optFlags) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : inconsistent render options.");
    }

    if (!m_modelPtr) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot register resource, uninitialized model.");
    }
    CHECK_STATUS_RETURN(m_modelPtr->registerKernelResources(memoryBindings, sourceCodeTable, kernelOpts, logger));

    const NREModel::FeaturesLayout nreFeaturesLayout = m_modelPtr->featuresLayout();

    const bool cameraEnableFeatures               = !(kernelOpts & KernelOpts::DisableFeatures) && m_cameraOutputSettings.enableFeatures && (nreFeaturesLayout.baseFeaturesDim > 0);
    const bool cameraEnableExtendedFeatures       = !(kernelOpts & KernelOpts::DisableExtendedFeatures) && m_cameraOutputSettings.enableExtendedFeatures && (nreFeaturesLayout.extendedFeaturesDim > 0);
    const bool cameraEnableSensorExtendedFeatures = !(kernelOpts & KernelOpts::DisableSensorExtendedFeatures) && m_cameraOutputSettings.enableSensorExtendedFeatures && (nreFeaturesLayout.cameraExtendedFeaturesDim > 0);
    const bool cameraEnableNormals                = !(kernelOpts & KernelOpts::DisableNormals) && m_cameraOutputSettings.enableNormals;
    const bool cameraEnableRayGradients           = !(kernelOpts & KernelOpts::DisableRayGradients) && m_cameraOutputSettings.enableRayGradients;

    const bool lidarEnableFeatures               = !(kernelOpts & KernelOpts::DisableFeatures) && m_lidarOutputSettings.enableFeatures && (nreFeaturesLayout.baseFeaturesDim > 0);
    const bool lidarEnableExtendedFeatures       = !(kernelOpts & KernelOpts::DisableExtendedFeatures) && m_lidarOutputSettings.enableExtendedFeatures && (nreFeaturesLayout.extendedFeaturesDim > 0);
    const bool lidarEnableSensorExtendedFeatures = !(kernelOpts & KernelOpts::DisableSensorExtendedFeatures) && m_lidarOutputSettings.enableSensorExtendedFeatures && (nreFeaturesLayout.lidarExtendedFeaturesDim > 0);
    const bool lidarEnableNormals                = !(kernelOpts & KernelOpts::DisableNormals) && m_lidarOutputSettings.enableNormals;
    const bool lidarEnableRayGradients           = !(kernelOpts & KernelOpts::DisableRayGradients) && m_lidarOutputSettings.enableRayGradients;

    const std::string projectorTemplate = R"(
        #include <nrend/renderer/gutRendererParameters.h>

        namespace nrend {{
        namespace GUTParameters {{

        struct CameraTiling {{
            static constexpr uint32_t BlockX    = {CameraTilingBlockX};
            static constexpr uint32_t BlockY    = {CameraTilingBlockY};
            static constexpr uint32_t BlockSize = BlockX * BlockY;
            static constexpr uint32_t NumWarps  = BlockSize / nrend::GUTParameters::WarpSize;
            static constexpr bool EnableRayBasedCulling = false;
        }};

        struct LidarTiling {{
            static constexpr uint32_t BlockX    = {LidarTilingBlockX};
            static constexpr uint32_t BlockY    = {LidarTilingBlockY};
            static constexpr uint32_t BlockSize = BlockX * BlockY;
            static constexpr uint32_t NumWarps  = BlockSize / nrend::GUTParameters::WarpSize;
            static constexpr bool EnableRayBasedCulling = {EnableRayBasedCulling};
        }};

        }}
        }}

        struct TGUTProjectorParamsBase
        {{
            static constexpr float MinProjectedRayRadiusSq       = {MinProjectedRayRadiusSq};
            static constexpr float AlphaThreshold                = {GUTModelClassAlias}::Particles::AlphaThreshold;
            static constexpr bool TightOpacityBounding           = {TightOpacityBounding};
            static constexpr bool RectBounding                   = {RectBounding};
            static constexpr bool TileCulling                    = {TileCulling};
            static constexpr bool PerRayParticleFeatures         = {PerRayParticleFeatures};
            static constexpr float NearClipDistance              = {NearClipDistance};
            static constexpr bool NearFarZCulling                = {NearFarZCulling};
            static constexpr float MaxDepthValue                 = {MaxDepthValue};
            static constexpr bool GlobalZOrder                   = {GlobalZOrder};
            static constexpr bool BackwardProjection             = {BackwardProjection};
            static constexpr bool EnableLinearProjection         = {EnableLinearProjection};
            static constexpr uint32_t SceneDataDim               = {ProjectorSceneDataDim};
            static constexpr int32_t SceneDataVisibilityOffset   = {ProjectorSceneDataVisibilityOffset};
        }};

        struct TGUTCameraProjectorParams : TGUTProjectorParamsBase
        {{
            static constexpr float FarClipDistance = {FarClipDistanceCamera};
        }};

        struct TGUTLidarProjectorParams : TGUTProjectorParamsBase
        {{
            static constexpr float FarClipDistance = {FarClipDistanceLidar};
        }};

        struct TGUTProjectionParams
        {{
            static constexpr int NRollingShutterIterations = {NRollingShutterIterations};
            static constexpr int D                         = {D};
            static constexpr float Alpha                   = {Alpha};
            static constexpr float Beta                    = {Beta};
            static constexpr float Kappa                   = {Kappa};
            static constexpr float Delta                   = {Delta}; ///< sqrt(Alpha*Alpha*(D+Kappa))
            static constexpr float ImageMarginFactor       = {ImageMarginFactor};
            static constexpr bool RequireAllSigmaPoints    = {RequireAllSigmaPoints};
        }};

        #include <nrend/kernels/cuda/renderers/gutProjector.cuh>

        using TGUTCameraProjector = GUTProjector<{GUTModelClassAlias}::Particles,
                                                  TGUTCameraProjectorParams,
                                                  TGUTProjectionParams,
                                                  nrend::GUTParameters::CameraTiling,
                                                  {CameraEnableFeatures},
                                                  {CameraEnableExtendedFeatures},
                                                  {CameraEnableCameraExtendedFeatures},
                                                  false /*CameraEnableLidarExtendedFeatures*/>;
        using TGUTLidarProjector = GUTProjector<{GUTModelClassAlias}::Particles,
                                                TGUTLidarProjectorParams,
                                                TGUTProjectionParams,
                                                nrend::GUTParameters::LidarTiling,
                                                {LidarEnableFeatures},
                                                {LidarEnableExtendedFeatures},
                                                false /*LidarEnableCameraExtendedFeatures*/,
                                                {LidarEnableLidarExtendedFeatures}>;
    )";
    const std::string projectorDefinition =
        fmt::format(projectorTemplate,
                    fmt::arg("CameraTilingBlockX", m_settings.cameraTileSize.x),
                    fmt::arg("CameraTilingBlockY", m_settings.cameraTileSize.y),
                    fmt::arg("LidarTilingBlockX", m_settings.lidarTileSize.x),
                    fmt::arg("LidarTilingBlockY", m_settings.lidarTileSize.y),
                    fmt::arg("EnableRayBasedCulling", m_settings.enableRayBasedCulling),
                    fmt::arg("MinProjectedRayRadiusSq", m_projectionSettings.minProjectedRayRadius * m_projectionSettings.minProjectedRayRadius),
                    fmt::arg("TightOpacityBounding", m_settings.tightOpacityBounding),
                    fmt::arg("RectBounding", m_settings.rectBounding),
                    fmt::arg("TileCulling", m_settings.tileCulling),
                    fmt::arg("PerRayParticleFeatures", m_settings.perRayFeatures),
                    fmt::arg("NearClipDistance", m_settings.nearClipDistance),
                    fmt::arg("FarClipDistanceCamera", m_settings.farClipDistanceCamera),
                    fmt::arg("FarClipDistanceLidar", m_settings.farClipDistanceLidar),
                    fmt::arg("NearFarZCulling", m_settings.nearFarZCulling),
                    fmt::arg("MaxDepthValue", std::numeric_limits<float>::max()),
                    fmt::arg("GlobalZOrder", m_settings.globalZOrder),
                    fmt::arg("BackwardProjection", backwardProjectionEnabled()),
                    fmt::arg("EnableLinearProjection", linearProjectionEnabled()),
                    fmt::arg("ProjectorSceneDataDim", m_renderingSceneDataLayout.count()),
                    fmt::arg("ProjectorSceneDataVisibilityOffset", m_renderingSceneDataLayout.visibility.offset),
                    fmt::arg("NRollingShutterIterations", m_projectionSettings.nRollingShutterIterations),
                    fmt::arg("D", m_projectionSettings.dim),
                    fmt::arg("Alpha", m_projectionSettings.alpha),
                    fmt::arg("Beta", m_projectionSettings.beta),
                    fmt::arg("Kappa", m_projectionSettings.kappa),
                    fmt::arg("Delta", std::sqrt(m_projectionSettings.alpha * m_projectionSettings.alpha * (m_projectionSettings.dim + m_projectionSettings.kappa))),
                    fmt::arg("ImageMarginFactor", m_projectionSettings.imageMarginFactor),
                    fmt::arg("RequireAllSigmaPoints", m_projectionSettings.requireAllSigmaPoints),
                    fmt::arg("GUTModelClassAlias", m_modelPtr->cudaCallPrefix()),
                    fmt::arg("CameraEnableFeatures", cameraEnableFeatures),
                    fmt::arg("CameraEnableExtendedFeatures", cameraEnableExtendedFeatures),
                    fmt::arg("CameraEnableCameraExtendedFeatures", cameraEnableSensorExtendedFeatures),
                    fmt::arg("LidarEnableFeatures", lidarEnableFeatures),
                    fmt::arg("LidarEnableExtendedFeatures", lidarEnableExtendedFeatures),
                    fmt::arg("LidarEnableLidarExtendedFeatures", lidarEnableSensorExtendedFeatures));

    std::string rendererDefinition;
    switch (m_settings.renderMode) {
    case Settings::KBuffer: {
        const std::string rendererTemplate = R"(
        struct TGUTRendererParams
        {{
            static constexpr bool PerRayParticleFeatures    = {PerRayParticleFeatures};
            static constexpr float MinProjectedRayRadius    = {MinProjectedRayRadius};
            static constexpr int KHitBufferSize             = {KHitBufferSize};
            static constexpr bool EnableWarpAtomicOptim     = {EnableWarpAtomicOptim};
            static constexpr uint32_t SceneDataDim          = {SceneDataDim};
            static constexpr int32_t SceneDataWeightsOffset = {SceneDataWeightsOffset};
        }};

        #include <nrend/kernels/cuda/renderers/gutKBufferRenderer.cuh>

        using TGUTCameraRenderer = GUTKBufferRenderer<{GUTModelClassAlias}::Particles,
                                                      TGUTRendererParams,
                                                      nrend::GUTParameters::CameraTiling,
                                                      {CameraEnableFeatures},
                                                      {CameraEnableExtendedFeatures},
                                                      {CameraEnableCameraExtendedFeatures},
                                                      false /*CameraEnableLidarExtendedFeatures*/,
                                                      {CameraEnableNormals},
                                                      {CameraEnableRayGradients},
                                                      false /*Backward*/>;
        using TGUTCameraBackwardRenderer = GUTKBufferRenderer<{GUTModelClassAlias}::Particles,
                                                               TGUTRendererParams,
                                                               nrend::GUTParameters::CameraTiling,
                                                               {CameraEnableFeatures},
                                                               {CameraEnableExtendedFeatures},
                                                               {CameraEnableCameraExtendedFeatures},
                                                               false /*CameraEnableLidarExtendedFeatures*/,
                                                               {CameraEnableNormals},
                                                               {CameraEnableRayGradients},
                                                               true /*Backward*/>;

        using TGUTLidarRenderer = GUTKBufferRenderer<{GUTModelClassAlias}::Particles,
                                                      TGUTRendererParams, nrend::GUTParameters::LidarTiling,
                                                      {LidarEnableFeatures},
                                                      {LidarEnableExtendedFeatures},
                                                      false /*LidarEnableCameraExtendedFeatures*/,
                                                      {LidarEnableLidarExtendedFeatures},
                                                      {LidarEnableNormals},
                                                      {LidarEnableRayGradients},
                                                      false /*Backward*/>;
        using TGUTLidarBackwardRenderer = GUTKBufferRenderer<{GUTModelClassAlias}::Particles,
                                                             TGUTRendererParams, nrend::GUTParameters::LidarTiling,
                                                             {LidarEnableFeatures},
                                                             {LidarEnableExtendedFeatures},
                                                             false /*LidarEnableCameraExtendedFeatures*/,
                                                             {LidarEnableLidarExtendedFeatures},
                                                             {LidarEnableNormals},
                                                             {LidarEnableRayGradients},
                                                             true /*Backward*/>;
        )";
        rendererDefinition =
            fmt::format(rendererTemplate,
                        fmt::arg("PerRayParticleFeatures", m_settings.perRayFeatures),
                        fmt::arg("MinProjectedRayRadius", m_projectionSettings.minProjectedRayRadius),
                        fmt::arg("KHitBufferSize", m_settings.kBufferSize),
                        fmt::arg("EnableWarpAtomicOptim", m_settings.enableWarpAtomicOptim),
                        fmt::arg("SceneDataDim", m_renderingSceneDataLayout.count()),
                        fmt::arg("SceneDataWeightsOffset", m_renderingSceneDataLayout.cumulatedWeights.offset),
                        fmt::arg("GUTModelClassAlias", m_modelPtr->cudaCallPrefix()),
                        fmt::arg("CameraEnableFeatures", cameraEnableFeatures),
                        fmt::arg("CameraEnableExtendedFeatures", cameraEnableExtendedFeatures),
                        fmt::arg("CameraEnableCameraExtendedFeatures", cameraEnableSensorExtendedFeatures),
                        fmt::arg("CameraEnableNormals", cameraEnableNormals),
                        fmt::arg("CameraEnableRayGradients", cameraEnableRayGradients),
                        fmt::arg("LidarEnableFeatures", lidarEnableFeatures),
                        fmt::arg("LidarEnableExtendedFeatures", lidarEnableExtendedFeatures),
                        fmt::arg("LidarEnableLidarExtendedFeatures", lidarEnableSensorExtendedFeatures),
                        fmt::arg("LidarEnableNormals", lidarEnableNormals),
                        fmt::arg("LidarEnableRayGradients", lidarEnableRayGradients));

    } break;
    case Settings::Splat: {
        if (m_settings.perRayFeatures) {
            RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : splat render mode does not support per ray feature evaluation.");
        }
        const std::string rendererTemplate = R"(
        struct TGUTRendererParams
        {{
            static constexpr bool ProjectRay                = {ProjectRay};
            static constexpr bool GlobalZOrder              = {GlobalZOrder};
            static constexpr uint32_t SceneDataDim          = {SceneDataDim};
            static constexpr int32_t SceneDataWeightsOffset = {SceneDataWeightsOffset};
        }};

        #include <nrend/kernels/cuda/renderers/gutSplatRenderer.cuh>

        using TGUTCameraRenderer = GUTSplatRenderer<{GUTModelClassAlias}::Particles,
                                                     TGUTRendererParams,
                                                     nrend::GUTParameters::CameraTiling,
                                                     {CameraEnableFeatures},
                                                     {CameraEnableExtendedFeatures},
                                                     {CameraEnableCameraExtendedFeatures},
                                                     false /*CameraEnableLidarExtendedFeatures*/,
                                                     {CameraEnableNormals},
                                                     {CameraEnableRayGradients},
                                                     false /*Backward*/>;
        using TGUTCameraBackwardRenderer = GUTSplatRenderer<{GUTModelClassAlias}::Particles,
                                                             TGUTRendererParams,
                                                             nrend::GUTParameters::CameraTiling,
                                                             {CameraEnableFeatures},
                                                             {CameraEnableExtendedFeatures},
                                                             {CameraEnableCameraExtendedFeatures},
                                                             false /*CameraEnableLidarExtendedFeatures*/,
                                                             {CameraEnableNormals},
                                                             {CameraEnableRayGradients},
                                                             true /*Backward*/>;

        using TGUTLidarRenderer = GUTSplatRenderer<{GUTModelClassAlias}::Particles,
                                                     TGUTRendererParams,
                                                     nrend::GUTParameters::LidarTiling,
                                                     {LidarEnableFeatures},
                                                     {LidarEnableExtendedFeatures},
                                                     false /*LidarEnableCameraExtendedFeatures*/,
                                                     {LidarEnableLidarExtendedFeatures},
                                                     {LidarEnableNormals},
                                                     {LidarEnableRayGradients},
                                                     false /*Backward*/>;
        using TGUTLidarBackwardRenderer = GUTSplatRenderer<{GUTModelClassAlias}::Particles,
                                                             TGUTRendererParams,
                                                             nrend::GUTParameters::LidarTiling,
                                                             {LidarEnableFeatures},
                                                             {LidarEnableExtendedFeatures},
                                                             false /*LidarEnableCameraExtendedFeatures*/,
                                                             {LidarEnableLidarExtendedFeatures},
                                                             {LidarEnableNormals},
                                                             {LidarEnableRayGradients},
                                                             true /*Backward*/>;
        )";
        rendererDefinition =
            fmt::format(rendererTemplate,
                        fmt::arg("ProjectRay", true),
                        fmt::arg("GlobalZOrder", m_settings.globalZOrder),
                        fmt::arg("SceneDataDim", m_renderingSceneDataLayout.count()),
                        fmt::arg("SceneDataWeightsOffset", m_renderingSceneDataLayout.cumulatedWeights.offset),
                        fmt::arg("GUTModelClassAlias", m_modelPtr->cudaCallPrefix()),
                        fmt::arg("CameraEnableFeatures", cameraEnableFeatures),
                        fmt::arg("CameraEnableExtendedFeatures", cameraEnableExtendedFeatures),
                        fmt::arg("CameraEnableCameraExtendedFeatures", cameraEnableSensorExtendedFeatures),
                        fmt::arg("CameraEnableNormals", cameraEnableNormals),
                        fmt::arg("CameraEnableRayGradients", cameraEnableRayGradients),
                        fmt::arg("LidarEnableFeatures", lidarEnableFeatures),
                        fmt::arg("LidarEnableExtendedFeatures", lidarEnableExtendedFeatures),
                        fmt::arg("LidarEnableLidarExtendedFeatures", lidarEnableSensorExtendedFeatures),
                        fmt::arg("LidarEnableNormals", lidarEnableNormals),
                        fmt::arg("LidarEnableRayGradients", lidarEnableRayGradients));

    } break;
    default:
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : undefined render mode.");
    }

    const std::string sourceCodeTemplate = R"(

        {TGUTProjectorDefinition}

        {TGUTRendererDefinition}

        using TGUTModel = {GUTModelClassAlias};

        static constexpr bool SRGBModel = {SRGBModel};
        static constexpr bool SRGBOutput = {SRGBOutput};
        static constexpr bool Differentiable = {Differentiable};

        #include <nrend/kernels/cuda/renderers/gutRenderer.cuh>
    )";

    std::vector<const char*> entryPointNames(NumForwardEntryPoints);
    entryPointNames[PreProcessParticlesKernelEntryPoint]   = "preProcessParticles";
    entryPointNames[ProjectOnTilesKernelEntryPoint]        = "projectOnTiles";
    entryPointNames[ExpandTileProjectionsKernelEntryPoint] = "expandTileProjections";
    entryPointNames[RenderLidarKernelEntryPoint]           = "renderLidar";
    entryPointNames[RenderCameraKernelEntryPoint]          = "renderCamera";
    entryPointNames[PrepareSceneKernelEntryPoint]          = "prepareScene";
    if (m_optFlags & KernelOpts::Differentiable) {
        entryPointNames.resize(NumKernelEntryPoints);
        entryPointNames[RenderBackwardLidarKernelEntryPoint]  = "renderBackwardLidar";
        entryPointNames[RenderBackwardCameraKernelEntryPoint] = "renderBackwardCamera";
        entryPointNames[ProjectBackwardKernelEntryPoint]      = "projectBackward";
        entryPointNames[PrepareSceneBackwardKernelEntryPoint] = "prepareSceneBackward";
    }

    m_renderKernelIndex = kernelDefinitionsTable.registerKernel({KernelDefinition::CudaKernel,
                                                                 CudaKernelOptions{entryPointNames},
                                                                 fmt::format(sourceCodeTemplate,
                                                                             fmt::arg("TGUTProjectorDefinition", projectorDefinition),
                                                                             fmt::arg("TGUTRendererDefinition", rendererDefinition),
                                                                             fmt::arg("GUTModelClassAlias", m_modelPtr->cudaCallPrefix()),
                                                                             fmt::arg("SRGBModel", !(m_optFlags & KernelOpts::DisablePostProcessings)),
                                                                             fmt::arg("SRGBOutput", !(m_optFlags & KernelOpts::LinearRGB)),
                                                                             fmt::arg("Differentiable", m_optFlags & KernelOpts::Differentiable))});

    return Status();
}

nrend::Status nrend::GUTRenderer::renderForward(const RenderParameters& params,
                                                const vec3* wordlRayOriginCudaPtr,
                                                const vec3* worldRayDirectionCudaPtr,
                                                const TTimestamp* worldRayTimestampCudaPtr,
                                                const ivec2* sensorsIdsPtr,
                                                const ivec2* activeTrackInstancesIdsCudaPtr,
                                                const TTrackInstancePose* activeTrackInstancesStartPoseCudaPtr,
                                                const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                                uint32_t* instanceIdCudaPtr,
                                                float* worldHitDistanceCudaPtr,
                                                vec3* worldHitNormalCudaPtr,
                                                vec4* radianceDensityCudaPtr,
                                                void* extendedFeaturesCudaPtr,
                                                void* sceneDataCudaPtr,
                                                ForwardContext** forwardContext,
                                                int cudaDeviceIndex,
                                                cudaStream_t cudaStream) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render forward, not initialized.");
    }

    if (forwardContext) {
        *forwardContext = nullptr;
    }

    // prepare the cuda resources required for rendering
    auto cudaResource = m_cudaKernelResources->prepare(this, m_optFlags, cudaStream, cudaDeviceIndex, m_logger);
    if (!cudaResource) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render");

    // FIXME : resources concurrent management
    // if (cudaResource->resourcesUseCount() > 2) {
    //     RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render, cuda resource on device %d is already in use.", cudaDeviceIndex);
    // }
    std::unique_ptr<GutRenderForwardContext> gutForwardContextPtr =
        std::make_unique<GutRenderForwardContext>(reinterpret_cast<uint64_t>(cudaStream));

    INREParticlesModel* particlesModelPtr = dynamic_cast<INREParticlesModel*>(m_modelPtr.get());
    RETURN_ERROR_IF(!particlesModelPtr, m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render, model is not an INREParticlesModel.");

    const KernelMemoryPtrVec* parameterMemoryPtrVec = cudaResource->memoryPtrVec(KernelMemoryBindings::BindingsFlag::Parameters);
    if (!parameterMemoryPtrVec) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource,
                     "GUTRenderer : cannot render, no parameter memory on device %d.", cudaDeviceIndex);
    }

    uint32_t numParticlesToPreProcess = 0;
    {
        const auto prepapreProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/prepare-particles"};
        CHECK_STATUS_RETURN(particlesModelPtr->prepareParticlesParameters(params.numActiveTrackInstances,
                                                                          activeTrackInstancesIdsCudaPtr,
                                                                          *parameterMemoryPtrVec,
                                                                          gutForwardContextPtr->transientParameters,
                                                                          gutForwardContextPtr->numParticles,
                                                                          numParticlesToPreProcess,
                                                                          cudaStream,
                                                                          m_logger));
    }
    if (gutForwardContextPtr->numParticles == 0) {
        LOG_WARN(m_logger, "GUTRenderer : no particles to render.");
        return Status();
    }

    CHECK_STATUS_RETURN(gutForwardContextPtr->updateParameterMemoryHandlesBuffer(*parameterMemoryPtrVec, m_logger));

    if (numParticlesToPreProcess > 0) {
        const auto preprocessProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/preprocess-particles"};
        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_renderKernelIndex, PreProcessParticlesKernelEntryPoint,
            div_round_up<int>(numParticlesToPreProcess, GUTParameters::LinearLaunchSize), GUTParameters::LinearLaunchSize, /*shmem=*/0, cudaStream, m_logger,
            // start of kernel arguments //
            numParticlesToPreProcess,
            params.numActiveTrackInstances,
            activeTrackInstancesIdsCudaPtr,
            params.sensorState.startTimestamp + (params.sensorState.endTimestamp - params.sensorState.startTimestamp) / 2,
            activeTrackInstancesStartPoseCudaPtr,
            activeTrackInstancesEndPoseCudaPtr,
            gutForwardContextPtr->parameterMemoryHandles.data()));
    }

    const SensorType sensorType          = sensorIsLidar(params.sensorModel) ? SensorType::Lidar : SensorType::Camera;
    const tcnn::ivec2 tileSize           = sensorType == SensorType::Lidar ? m_settings.lidarTileSize : m_settings.cameraTileSize;
    const tcnn::ivec2 tileGridResolution = sensorTileGridResolution(params.sensorModel, tileSize, params.frameTileResolution);
    const ivec2 tileGrid{div_round_up(tileGridResolution.x, tileSize.x), div_round_up(tileGridResolution.y, tileSize.y)};
    // Sanity check for LiDAR tile size
    if (sensorType == SensorType::Lidar) {
        if (params.sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
            if (params.sensorModel.nreHesaiP128LidarParams.maxPtsPerTile != tileSize.x * tileSize.y) {
                RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot render, LiDAR maximum points per tile <%d> does not match the tile size <%d, %d>.",
                             params.sensorModel.nreHesaiP128LidarParams.maxPtsPerTile, tileSize.x, tileSize.y);
            }
        } else {
            RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot render, LiDAR model type is not supported.");
        }
    }

    const uint32_t numParticles = gutForwardContextPtr->numParticles;

    CHECK_STATUS_RETURN(gutForwardContextPtr->updateParticlesWorkingBuffers(numParticles, cudaStream, m_logger));
    if (!m_settings.perRayFeatures) {
        CHECK_STATUS_RETURN(
            gutForwardContextPtr->updateParticlesFeaturesBuffer(numParticles * numPrecomputedFeatures(sensorType), cudaStream, m_logger));
    }

    if (m_settings.renderMode == Settings::Splat && worldHitNormalCudaPtr) {
        LOG_WARN(m_logger, "GUTRenderer : splat render mode does not support world hit normal output.");
    }

    // TODO : check if using cudaGraph may help (is it even supported with RTC kernel ?)
    {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/project"};

        const TPose sensorPose = interpolatedPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f);

        // clang-format off
        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_renderKernelIndex, ProjectOnTilesKernelEntryPoint,
            div_round_up(numParticles, GUTParameters::LinearLaunchSize),  GUTParameters::LinearLaunchSize,  /*shmem=*/0,  cudaStream, m_logger,
            // start of kernel arguments //
            tileGrid,
            numParticles,
            params.frameResolution,
            params.frameTileOffset,
            params.objectToWorldTransform,
            params.sensorModel,
            params.worldToObjectTransform * tcnn::vec4(poseInverse(sensorPose).slice<0,3>(), 1.f),
            // FIXME : work directly in sensor space to avoid all intermediate transforms
            poseToMat(sensorPose) * tcnn::mat4(params.objectToWorldTransform),
            params.sensorState,
            gutForwardContextPtr->particlesTilesCount.data(),
            gutForwardContextPtr->particlesProjectedPosition.data(),
            gutForwardContextPtr->particlesProjectedConicOpacity.data(),
            gutForwardContextPtr->particlesProjectedExtent.data(),
            gutForwardContextPtr->particlesGlobalDepth.data(),
            gutForwardContextPtr->particlesPrecomputedFeatures.data(),
            reinterpret_cast<float*>(sceneDataCudaPtr),
            gutForwardContextPtr->parameterMemoryHandles.data())
        );
        // clang-format on
    }

    deviceLaunchesLogger.push("render/prepare-expand");

    // inplace cumulative sum over list of number of intersected tiles per particles
    size_t scanningWorkingBufferSize = gutForwardContextPtr->scanningWorkingBuffer.size();
    // TODO : check if using not inplace version has perf benefits
    CUDA_CHECK_RETURN(
        cub::DeviceScan::InclusiveSum(
            gutForwardContextPtr->scanningWorkingBuffer.data(),
            scanningWorkingBufferSize,
            static_cast<const uint32_t*>(gutForwardContextPtr->particlesTilesCount.data()),
            static_cast<uint32_t*>(gutForwardContextPtr->particlesTilesOffset.data()),
            numParticles,
            cudaStream),
        m_logger);

    // fetch total number of particle/tile intersections to launch and resize the sorting buffers
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(&gutForwardContextPtr->numParticleTileIntersections,
                        static_cast<uint32_t*>(gutForwardContextPtr->particlesTilesOffset.data()) + numParticles - 1,
                        sizeof(uint32_t),
                        cudaMemcpyDeviceToHost,
                        cudaStream),
        m_logger);
    cudaStreamSynchronize(cudaStream);
    if (gutForwardContextPtr->numParticleTileIntersections == 0) {
        // setup the context for backward pass
        if (forwardContext && (m_optFlags & KernelOpts::Differentiable)) {
            *forwardContext = dynamic_cast<ForwardContext*>(gutForwardContextPtr.release());
        }
        return Status();
    }

    // sorting buffers allocation
    CHECK_STATUS_RETURN(
        gutForwardContextPtr->updateTileSortingBuffers(tileGrid, gutForwardContextPtr->numParticleTileIntersections, cudaStream, m_logger));

    deviceLaunchesLogger.pop("render/prepare-expand");

    {
        const auto expandProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/expand"};
        // clang-format off
        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_renderKernelIndex, ExpandTileProjectionsKernelEntryPoint,
            div_round_up(numParticles, GUTParameters::LinearLaunchSize), GUTParameters::LinearLaunchSize, /*shmem=*/0, cudaStream, m_logger,
            // start of kernel arguments //
            tileGrid,
            numParticles,
            params.sensorModel,
            params.sensorState,
            gutForwardContextPtr->particlesTilesOffset.data(),
            gutForwardContextPtr->particlesProjectedPosition.data(),
            gutForwardContextPtr->particlesProjectedConicOpacity.data(),
            gutForwardContextPtr->particlesProjectedExtent.data(),
            gutForwardContextPtr->particlesGlobalDepth.data(),
            gutForwardContextPtr->parameterMemoryHandles.data(),
            gutForwardContextPtr->unsortedTileDepthKeys.data(),
            gutForwardContextPtr->unsortedTileParticleIdx.data())
        );
        // clang-format on
    }

    deviceLaunchesLogger.push("render/sort");

    // Sort complete list of (duplicated) Gaussian indices by keys
    size_t sortingWorkingBufferSize = gutForwardContextPtr->sortingWorkingBuffer.size();
    CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
                          gutForwardContextPtr->sortingWorkingBuffer.data(),
                          sortingWorkingBufferSize,
                          static_cast<const uint64_t*>(gutForwardContextPtr->unsortedTileDepthKeys.data()),
                          static_cast<uint64_t*>(gutForwardContextPtr->sortedTileDepthKeys.data()),
                          static_cast<const uint32_t*>(gutForwardContextPtr->unsortedTileParticleIdx.data()),
                          static_cast<uint32_t*>(gutForwardContextPtr->sortedTileParticleIdx.data()),
                          gutForwardContextPtr->numParticleTileIntersections,
                          0, 32 + getMSBExclusive(tileGrid.x * tileGrid.y), cudaStream),
                      m_logger);

    // Compute the tile range indices in the sorted keys
    linear_kernel(
        computeSortedTileRangeIndices,
        /*shmem=*/0,
        cudaStream,
        gutForwardContextPtr->numParticleTileIntersections,
        static_cast<const uint64_t*>(gutForwardContextPtr->sortedTileDepthKeys.data()),
        static_cast<uvec2*>(gutForwardContextPtr->sortedTileRangeIndices.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);

    deviceLaunchesLogger.pop("render/sort");

    {
        const auto renderProfile                 = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render/render"};
        const KernelEntryPoints renderEntryPoint = sensorIsLidar(params.sensorModel) ? RenderLidarKernelEntryPoint : RenderCameraKernelEntryPoint;
        // clang-format off
        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_renderKernelIndex, renderEntryPoint,
            dim3{static_cast<uint32_t>(tileGrid.x), static_cast<uint32_t>(tileGrid.y), 1u},
            dim3{static_cast<uint32_t>(tileSize.x), static_cast<uint32_t>(tileSize.y), 1u},  /*shmem=*/0,  cudaStream,  m_logger,
            // start of kernel arguments //
            params,
            gutForwardContextPtr->sortedTileRangeIndices.data(),
            gutForwardContextPtr->sortedTileParticleIdx.data(),
            wordlRayOriginCudaPtr,
            worldRayDirectionCudaPtr,
            worldRayTimestampCudaPtr,
            sensorsIdsPtr,
            instanceIdCudaPtr,
            worldHitDistanceCudaPtr,
            worldHitNormalCudaPtr,
            radianceDensityCudaPtr,
            extendedFeaturesCudaPtr,
            sceneDataCudaPtr,
            gutForwardContextPtr->particlesProjectedPosition.data(),
            gutForwardContextPtr->particlesProjectedConicOpacity.data(),
            gutForwardContextPtr->particlesPrecomputedFeatures.data(),
            gutForwardContextPtr->parameterMemoryHandles.data())
        );
        // clang-format on
    }

    // setup the context for backward pass
    if (forwardContext && (m_optFlags & KernelOpts::Differentiable)) {
        *forwardContext = dynamic_cast<ForwardContext*>(gutForwardContextPtr.release());
    }

    return Status();
}

nrend::Status nrend::GUTRenderer::renderBackward(const RenderParameters& params,
                                                 const vec3* wordlRayOriginCudaPtr,
                                                 const vec3* worldRayDirectionCudaPtr,
                                                 const nrend::TTimestamp* worldRayTimestampCudaPtr,
                                                 const ivec2* sensorsIdsPtr,
                                                 const ivec2* activeTrackInstancesIdsCudaPtr,
                                                 const TTrackInstancePose* activeTrackInstancesStartPoseCudaPtr,
                                                 const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                                 uint32_t* instanceIdCudaPtr,
                                                 const float* worldHitDistanceCudaPtr,
                                                 const float* worldHitDistanceGradientCudaPtr,
                                                 const vec3* worldHitNormalCudaPtr,
                                                 const vec3* worldHitNormalGradientCudaPtr,
                                                 const vec4* radianceDensityCudaPtr,
                                                 const vec4* radianceDensityGradientCudaPtr,
                                                 const void* extendedFeaturesCudaPtr,
                                                 const void* extendedFeaturesGradientCudaPtr,
                                                 vec3* wordlRayOriginGradientCudaPtr,
                                                 vec3* worldRayDirectionGradientCudaPtr,
                                                 ForwardContext* forwardContextPtr,
                                                 int cudaDeviceIndex,
                                                 cudaStream_t cudaStream) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render backward, not initialized.");
    }

    if ((!(m_optFlags & KernelOpts::Differentiable))) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime, "GUTRenderer : cannot call renderBackward(Lidar|Camera), not initialized as differentiable.");
    }

    // FIXME : LIDAR does not support frame-tile rendering
    if ((params.sensorModel.modelType == TSensorModel::ModelType::RowOffsetStructuredSpinningLidarModel) &&
        (std::abs(params.frameTileOffset.x) > 0 || std::abs(params.frameTileOffset.y) > 0)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot render backward, LIDAR does not support tile-frame rendering.");
    }

    auto cudaResource = m_cudaKernelResources->prepare(this, m_optFlags, cudaStream, cudaDeviceIndex, m_logger);
    if (!cudaResource) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }
    // FIXME : resources concurrent management
    // if (cudaResource->resourcesUseCount() > 3) {
    //     RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render backward, cuda resource on device %d, already in use.", cudaDeviceIndex);
    // }
    const GutRenderForwardContext* gutForwardContextPtr = dynamic_cast<const GutRenderForwardContext*>(forwardContextPtr);
    if (!gutForwardContextPtr || (gutForwardContextPtr->queueHandle != reinterpret_cast<uint64_t>(cudaStream))) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot render backward, invalid forward context on device %d.", cudaDeviceIndex);
    }

    if (gutForwardContextPtr->numParticleTileIntersections == 0) {
        LOG_WARN(m_logger, "GUTRenderer : not backpropagating, no particle/tile intersections.");
        return Status();
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render-backward");

    const SensorType sensorType          = sensorIsLidar(params.sensorModel) ? SensorType::Lidar : SensorType::Camera;
    const KernelEntryPoints entryPoint   = sensorIsLidar(params.sensorModel) ? RenderBackwardLidarKernelEntryPoint : RenderBackwardCameraKernelEntryPoint;
    const tcnn::ivec2 tileSize           = sensorType == SensorType::Lidar ? m_settings.lidarTileSize : m_settings.cameraTileSize;
    const tcnn::ivec2 tileGridResolution = sensorTileGridResolution(params.sensorModel, tileSize, params.frameTileResolution);
    const ivec2 tileGrid{div_round_up(tileGridResolution.x, tileSize.x), div_round_up(tileGridResolution.y, tileSize.y)};
    // Sanity check for LiDAR tile size
    if (sensorType == SensorType::Lidar) {
        if (params.sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
            if (params.sensorModel.nreHesaiP128LidarParams.maxPtsPerTile != tileSize.x * tileSize.y) {
                RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot render, LiDAR maximum points per tile <%d> does not match the tile size <%d, %d>.",
                             params.sensorModel.nreHesaiP128LidarParams.maxPtsPerTile, tileSize.x, tileSize.y);
            }
        } else {
            RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot render, LiDAR model type is not supported.");
        }
    }

    const uint32_t numParticles = gutForwardContextPtr->numParticles;

    if (!m_settings.perRayFeatures) {
        CHECK_STATUS_RETURN(
            gutForwardContextPtr->updateParticlesFeaturesGradientBuffer(numParticles * numPrecomputedFeatures(sensorType), cudaStream, m_logger));
    }

    if (m_settings.renderMode == Settings::Splat) {
        CHECK_STATUS_RETURN(
            gutForwardContextPtr->updateParticlesProjectionGradientBuffers(numParticles, cudaStream, m_logger));
    }

    {
        const auto renderProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward/render"};

        // clang-format off
        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_renderKernelIndex, entryPoint,
            dim3{static_cast<uint32_t>(tileGrid.x), static_cast<uint32_t>(tileGrid.y), 1u},
            dim3{static_cast<uint32_t>(tileSize.x), static_cast<uint32_t>(tileSize.y), 1u}, /*shmem=*/0, cudaStream, m_logger,
            // start of kernel arguments //
            params,
            gutForwardContextPtr->sortedTileRangeIndices.data(),
            gutForwardContextPtr->sortedTileParticleIdx.data(),
            wordlRayOriginCudaPtr,
            worldRayDirectionCudaPtr,
            worldRayTimestampCudaPtr,
            sensorsIdsPtr,
            instanceIdCudaPtr,
            worldHitDistanceCudaPtr,
            worldHitDistanceGradientCudaPtr,
            worldHitNormalCudaPtr,
            worldHitNormalGradientCudaPtr,
            radianceDensityCudaPtr,
            radianceDensityGradientCudaPtr,
            extendedFeaturesCudaPtr,
            extendedFeaturesGradientCudaPtr,
            wordlRayOriginGradientCudaPtr,
            worldRayDirectionGradientCudaPtr,
            gutForwardContextPtr->particlesProjectedPosition.data(),
            gutForwardContextPtr->particlesProjectedConicOpacity.data(),
            gutForwardContextPtr->particlesPrecomputedFeatures.data(),
            gutForwardContextPtr->parameterMemoryHandles.data(),
            gutForwardContextPtr->particlesProjectedPositionGradient.data(),
            gutForwardContextPtr->particlesProjectedConicOpacityGradient.data(),
            gutForwardContextPtr->particlesPrecomputedFeaturesGradient.data(),
            cudaResource->memoryHandlesPtr(KernelMemoryBindings::BindingsFlag::ParameterGradients))
        );
        // clang-format on
    }

    if (!m_settings.perRayFeatures) {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward/project"};

        const TPose sensorPose = interpolatedPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f);

        // clang-format off
        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_renderKernelIndex, ProjectBackwardKernelEntryPoint,
            div_round_up(numParticles, GUTParameters::LinearLaunchSize), GUTParameters::LinearLaunchSize, /*shmem=*/0, cudaStream, m_logger,
            // start of kernel arguments //
            numParticles,
            params.frameResolution,
            params.sensorModel,
            params.worldToObjectTransform * tcnn::vec4(poseInverse(sensorPose).slice<0,3>(), 1.f),
            // FIXME : work directly in sensor space to avoid all intermediate transforms
            poseToMat(sensorPose) * tcnn::mat4(params.objectToWorldTransform),
            gutForwardContextPtr->particlesTilesCount.data(),
            gutForwardContextPtr->parameterMemoryHandles.data(),
            gutForwardContextPtr->particlesProjectedPositionGradient.data(),
            gutForwardContextPtr->particlesProjectedConicOpacityGradient.data(),
            gutForwardContextPtr->particlesPrecomputedFeatures.data(),
            gutForwardContextPtr->particlesPrecomputedFeaturesGradient.data(),
            cudaResource->memoryHandlesPtr(KernelMemoryBindings::BindingsFlag::ParameterGradients))
        );
        // clang-format on
    }

    return Status();
}

nrend::Status nrend::GUTRenderer::prepareSceneForward(const RenderParameters& params,
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
                                                      cudaStream_t cudaStream) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot prepare scene, not initialized.");
    }

    if (forwardContext) {
        *forwardContext = nullptr;
    }

    // prepare the cuda resources required for rendering
    auto cudaResource = m_cudaKernelResources->prepare(this, m_optFlags, cudaStream, cudaDeviceIndex, m_logger);
    if (!cudaResource) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("prepare-scene");

    if (m_settings.renderMode == Settings::Splat) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot prepare scene, splat render mode is not supported.");
    }

    // FIXME : resources concurrent management
    // if (cudaResource->resourcesUseCount() > 2) {
    //     RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render, cuda resource on device %d is already in use.", cudaDeviceIndex);
    // }
    std::unique_ptr<GutPrepareSceneForwardContext> gutForwardContextPtr =
        std::make_unique<GutPrepareSceneForwardContext>(reinterpret_cast<uint64_t>(cudaStream));

    INREParticlesModel* particlesModelPtr = dynamic_cast<INREParticlesModel*>(m_modelPtr.get());
    assert(particlesModelPtr && "GUTRenderer : cannot prepare scene, model is not an INREParticlesModel.");

    const uint64_t* parameterMemoryPtr = cudaResource->memoryHandlesPtr(KernelMemoryBindings::BindingsFlag::Parameters);
    if (!parameterMemoryPtr) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource,
                     "GUTRenderer : cannot prepare scene, no parameter memory on device %d.", cudaDeviceIndex);
    }

    const SensorType sensorType          = sensorIsLidar(params.sensorModel) ? SensorType::Lidar : SensorType::Camera;
    const tcnn::ivec2 tileSize           = sensorType == SensorType::Lidar ? m_settings.lidarTileSize : m_settings.cameraTileSize;
    const tcnn::ivec2 tileGridResolution = sensorTileGridResolution(params.sensorModel, tileSize, params.frameTileResolution);
    const ivec2 tileGrid{div_round_up(tileGridResolution.x, tileSize.x), div_round_up(tileGridResolution.y, tileSize.y)};
    // Sanity check for LiDAR tile size
    if (sensorType == SensorType::Lidar) {
        if (params.sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
            if (params.sensorModel.nreHesaiP128LidarParams.maxPtsPerTile != tileSize.x * tileSize.y) {
                RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot prepare scene, LiDAR maximum points per tile <%d> does not match the tile size <%d, %d>.",
                             params.sensorModel.nreHesaiP128LidarParams.maxPtsPerTile, tileSize.x, tileSize.y);
            }
        } else {
            RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot prepare scene, LiDAR model type is not supported.");
        }
    }

    const uint32_t numParticles = particlesModelPtr->numParticles();
    CHECK_STATUS_RETURN(gutForwardContextPtr->initialize(numParticles, cudaStream, m_logger));

    // TODO : check if using cudaGraph may help (is it even supported with RTC kernel ?)
    {
        const auto prepareProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "prepare-scene/prepare"};
        const TPose sensorPose    = interpolatedPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f);

        // clang-format off
        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_renderKernelIndex, PrepareSceneKernelEntryPoint,
            div_round_up(numParticles, GUTParameters::LinearLaunchSize),  GUTParameters::LinearLaunchSize,  /*shmem=*/0,  cudaStream, m_logger,
            // start of kernel arguments //
            tileGrid,
            numParticles,
            params.frameResolution,
            params.frameTileOffset,
            params.objectToWorldTransform,
            params.sensorModel,
            params.worldToObjectTransform * tcnn::vec4(poseInverse(sensorPose).slice<0,3>(), 1.f),
            // FIXME : work directly in sensor space to avoid all intermediate transforms
            poseToMat(sensorPose) * tcnn::mat4(params.objectToWorldTransform),
            params.sensorState,
            gutForwardContextPtr->validParticleCountIdx.data(),
            sceneDensityCudaPtr,
            sceneFeaturesCudaPtr,
            sceneExtendedFeaturesCudaPtr,
            sceneSensorExtendedFeaturesCudaPtr,
            sceneDataCudaPtr,
            parameterMemoryPtr)
        );
        // clang-format on

        // fetch total number of valid particles
        CUDA_CHECK_RETURN(
            cudaMemcpyAsync(&gutForwardContextPtr->numValidParticles,
                            static_cast<uint32_t*>(gutForwardContextPtr->validParticleCountIdx.data()),
                            sizeof(uint32_t),
                            cudaMemcpyDeviceToHost,
                            cudaStream),
            m_logger);
        cudaStreamSynchronize(cudaStream);
        sceneSize = gutForwardContextPtr->numValidParticles;
    }

    // setup the context for backward pass
    if (forwardContext && (m_optFlags & KernelOpts::Differentiable)) {
        *forwardContext = dynamic_cast<ForwardContext*>(gutForwardContextPtr.release());
    }

    return Status();
}

nrend::Status nrend::GUTRenderer::prepareSceneBackward(const RenderParameters& params,
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
                                                       ForwardContext* forwardContextPtr,
                                                       int cudaDeviceIndex,
                                                       cudaStream_t cudaStream) const {

    if (!initialized()) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot prepare scene backward, not initialized.");
    }

    if ((!(m_optFlags & KernelOpts::Differentiable))) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime, "GUTRenderer : cannot call prepareSceneBackward, not initialized as differentiable.");
    }

    // FIXME : LIDAR does not support frame-tile rendering
    if ((params.sensorModel.modelType == TSensorModel::ModelType::RowOffsetStructuredSpinningLidarModel) &&
        (std::abs(params.frameTileOffset.x) > 0 || std::abs(params.frameTileOffset.y) > 0)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot prepare scene backward, LIDAR does not support tile-frame rendering.");
    }

    auto cudaResource = m_cudaKernelResources->prepare(this, m_optFlags, cudaStream, cudaDeviceIndex, m_logger);
    if (!cudaResource) {
        RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot get cuda resource on the device %d.", cudaDeviceIndex);
    }
    // FIXME : resources concurrent management
    // if (cudaResource->resourcesUseCount() > 3) {
    //     RETURN_ERROR(m_logger, ErrorCode::InvalidResource, "GUTRenderer : cannot render backward, cuda resource on device %d, already in use.", cudaDeviceIndex);
    // }
    const GutPrepareSceneForwardContext* gutForwardContextPtr = dynamic_cast<const GutPrepareSceneForwardContext*>(forwardContextPtr);
    if (!gutForwardContextPtr || (gutForwardContextPtr->queueHandle != reinterpret_cast<uint64_t>(cudaStream))) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot prepare scene backward, invalid forward context on device %d.", cudaDeviceIndex);
    }

    if (gutForwardContextPtr->numValidParticles == 0) {
        LOG_WARN(m_logger, "GUTRenderer : not backpropagating, no valid particles.");
        return Status();
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("prepare-scene-backward");

    const SensorType sensorType          = sensorIsLidar(params.sensorModel) ? SensorType::Lidar : SensorType::Camera;
    const tcnn::ivec2 tileSize           = sensorType == SensorType::Lidar ? m_settings.lidarTileSize : m_settings.cameraTileSize;
    const tcnn::ivec2 tileGridResolution = sensorTileGridResolution(params.sensorModel, tileSize, params.frameTileResolution);
    const ivec2 tileGrid{div_round_up(tileGridResolution.x, tileSize.x), div_round_up(tileGridResolution.y, tileSize.y)};
    // Sanity check for LiDAR tile size
    if (sensorType == SensorType::Lidar) {
        if (params.sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
            if (params.sensorModel.nreHesaiP128LidarParams.maxPtsPerTile != tileSize.x * tileSize.y) {
                RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot prepare scene backward, LiDAR maximum points per tile <%d> does not match the tile size <%d, %d>.",
                             params.sensorModel.nreHesaiP128LidarParams.maxPtsPerTile, tileSize.x, tileSize.y);
            }
        } else {
            RETURN_ERROR(m_logger, ErrorCode::BadInput, "GUTRenderer : cannot prepare scene backward, LiDAR model type is not supported.");
        }
    }

    const uint32_t numValidParticles = gutForwardContextPtr->numValidParticles;

    if (!m_settings.perRayFeatures) {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "prepare-scene-backward/project"};

        const TPose sensorPose = interpolatedPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f);

        CHECK_STATUS_RETURN(cudaResource->launchCudaKernel(
            m_renderKernelIndex, PrepareSceneBackwardKernelEntryPoint,
            div_round_up(numValidParticles, GUTParameters::LinearLaunchSize), GUTParameters::LinearLaunchSize, /*shmem=*/0, cudaStream, m_logger,
            // start of kernel arguments //
            numValidParticles,
            params.frameResolution,
            params.sensorModel,
            params.worldToObjectTransform * tcnn::vec4(poseInverse(sensorPose).slice<0, 3>(), 1.f),
            // FIXME : work directly in sensor space to avoid all intermediate transforms
            poseToMat(sensorPose) * tcnn::mat4(params.objectToWorldTransform),
            gutForwardContextPtr->validParticleCountIdx.data(),
            sceneFeaturesCudaPtr,
            sceneExtendedFeaturesCudaPtr,
            sceneSensorExtendedFeaturesCudaPtr,
            cudaResource->memoryHandlesPtr(KernelMemoryBindings::BindingsFlag::Parameters),
            sceneDensityGradientCudaPtr,
            sceneFeaturesGradientCudaPtr,
            sceneExtendedFeaturesGradientCudaPtr,
            sceneSensorExtendedFeaturesGradientCudaPtr,
            cudaResource->memoryHandlesPtr(KernelMemoryBindings::BindingsFlag::ParameterGradients)));
    }

    return Status();
}

int nrend::GUTRenderer::numPrecomputedFeatures(SensorType sensorType) const {
    // (model base + extended features + object space hitDistance if backwardProjection is enabled)
    const RenderingFeaturesLayout& renderingFeaturesLayout = sensorType == SensorType::Camera ? m_cameraRenderingFeaturesLayout : m_lidarRenderingFeaturesLayout;
    return renderingFeaturesLayout.baseFeaturesDim + renderingFeaturesLayout.extendedFeaturesDim + renderingFeaturesLayout.sensorExtendedFeaturesDim + (backwardProjectionEnabled() ? 1 : 0);
}
