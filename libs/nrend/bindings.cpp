// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <cuda_bf16.h>
#include <random>

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <nrend/iNRenderer.h>
#include <nrend/utils/logger.h>
#include <pybind11/functional.h>

#include <vren/cameras.h>
#include <vren/lidars.h>

#ifndef NREND_MAX_LOG_LEVEL
// Do not change this value, it is controlled by Bazel as either:
// - Warning (for no-internal builds)
// - Debug (for internal builds)
#define NREND_MAX_LOG_LEVEL nrend::LoggerParameters::Warning
#endif

#define NREND_TEST_FTHETA_REGULAFALSI 0

int32_t computeWindshieldPolyOrder(int32_t maxDegree, std::vector<float> const& poly) {
    int32_t polyDeg = 0;
    for (int32_t deg = 0; deg < maxDegree; deg++) {
        const int32_t numTerms = (deg + 1) * (deg + 2) / 2;
        if (static_cast<size_t>(numTerms) == poly.size()) {
            polyDeg = deg;
            break;
        } else if (static_cast<size_t>(numTerms) > poly.size()) {
            throw std::runtime_error("windshield polynomial has invalid number of coefficients");
        }
    }
    return polyDeg;
};

using namespace nrend;

class NRendererForwardContextWrapper {

    struct ScopedHandle {
        INRenderer::RenderingContextHandle _handle;
        ScopedHandle(INRenderer::RenderingContextHandle handle)
            : _handle(handle) {}
        ~ScopedHandle() { INRenderer::destroyRenderingContext(_handle); };
    };

public:
    NRendererForwardContextWrapper(INRenderer::RenderingContextHandle handle = INRenderer::InvalidRenderingContextHandle)
        : m_ctxHandlePtr(new ScopedHandle(handle)) {}
    ~NRendererForwardContextWrapper() = default;

    INRenderer::RenderingContextHandle handle() const {
        return m_ctxHandlePtr ? m_ctxHandlePtr->_handle : INRenderer::InvalidRenderingContextHandle;
    }

    void reset() { return m_ctxHandlePtr.reset(); }

private:
    std::shared_ptr<ScopedHandle> m_ctxHandlePtr;
};

class NRendererProfiler final {
    struct CudaTimer {
        const std::string _tag;
        cudaStream_t _stream;
        cudaEvent_t _start = 0, _stop = 0;
        bool _valid = false, _stopped = false;

        CudaTimer(const std::string& tag, cudaStream_t stream)
            : _tag(tag), _stream(stream) {
            _valid = cudaEventCreate(&_start) == cudaSuccess;
            _valid = _valid && (cudaEventCreate(&_stop) == cudaSuccess);
            if (_valid) {
                cudaEventRecord(_start, _stream);
            }
        }
        // non-copyable because of the cudaEvents destruction
        CudaTimer(const CudaTimer&)            = delete;
        CudaTimer& operator=(const CudaTimer&) = delete;

        ~CudaTimer() {
            if (_stop) {
                cudaEventDestroy(_stop);
            }
            if (_start) {
                cudaEventDestroy(_start);
            }
        }

        void stop() {
            if (_valid && !_stopped) {
                cudaEventRecord(_stop, _stream);
                _stopped = true;
            }
        }

        float collect() {
            float milliseconds = 1e09f;
            if (_valid) {
                stop();
                cudaEventSynchronize(_stop);
                cudaEventElapsedTime(&milliseconds, _start, _stop);
            }
            return milliseconds;
        }
    };

    bool m_enabled;
    std::mt19937 m_rgen;
    std::uniform_real_distribution<float> m_rand;
    const float m_frequency;
    // CudaTimer are non-copyable
    std::vector<std::unique_ptr<CudaTimer>> m_currentTimers;
    std::map<std::string, std::pair<double, double>> m_timersStat;
    std::string m_scopedTags;

    inline void pushScopedTag(const char* tag, int uid) {
        if (tag) {
            const std::string fullTag = uid >= 0 ? std::string(tag) + "-" + std::to_string(uid) : std::string(tag);
            m_scopedTags += ("/" + fullTag);
        }
    }

    inline void popScopedTag(const char* tag, int uid) {
        if (tag) {
            const std::string fullTag = uid >= 0 ? std::string(tag) + "-" + std::to_string(uid) : std::string(tag);
            const size_t lastTagPos   = m_scopedTags.rfind("/" + fullTag);
            if (lastTagPos != std::string::npos) {
                m_scopedTags.erase(lastTagPos);
            }
        }
    }

public:
    NRendererProfiler(float frequency)
        : m_enabled(false), m_rgen(0), m_rand(0.f, 1.f), m_frequency(frequency) {
        resetTimersStat();
    }

    inline void initializeProfile(const char* tag = nullptr, int uid = -1) {
        if (m_enabled) {
            pushScopedTag(tag, uid);
        }
    }

    inline void finalizeProfile(const char* tag = nullptr, int uid = -1) {
        if (m_enabled) {
            for (auto& timer : m_currentTimers) {
                const std::string strTag = timer->_tag;
                auto it                  = m_timersStat.find(strTag);
                if (it != m_timersStat.end()) {
                    it->second.first += timer->collect();
                    it->second.second += 1.0;
                } else {
                    m_timersStat.insert({strTag, {static_cast<double>(timer->collect()), 1.0}});
                }
            }
            m_currentTimers.clear();
            popScopedTag(tag, uid);
        }
    }

    inline void start(const char* tag, int /*deviceIndex*/, cudaStream_t stream) {
        if (m_enabled) {
            m_currentTimers.emplace_back(new CudaTimer(tag ? m_scopedTags + "/" + tag : m_scopedTags, stream));
        }
    }

    inline void stop(const char* tag, int /*deviceIndex*/, cudaStream_t) {
        if (m_enabled) {
            const std::string fullTag = tag ? m_scopedTags + "/" + tag : m_scopedTags;
            for (size_t i = m_currentTimers.size(); i > 0; --i) {
                CudaTimer* timer = m_currentTimers[i - 1].get();
                if (fullTag == timer->_tag) {
                    timer->stop();
                    return;
                }
            }
        }
    }

    inline const std::map<std::string, float> collect(bool reset) {
        std::map<std::string, float> stats;
        for (const auto& tag : m_timersStat) {
            stats[tag.first] = static_cast<float>(tag.second.first / tag.second.second);
        }
        if (reset) {
            resetTimersStat();
        }
        return stats;
    }

    inline void resetTimersStat() {
        m_timersStat.clear();
        if (m_frequency > 0.f) {
            m_enabled = (1.0f - m_rand(m_rgen)) < m_frequency;
        }
    }

    class Scoped {
        NRendererProfiler& _profiler;
        const char* _tag;
        const int _uid;
        int _deviceIndex;
        cudaStream_t _stream;

    public:
        Scoped(NRendererProfiler& profiler,
               const char* tag     = nullptr,
               int deviceIndex     = 0,
               cudaStream_t stream = 0,
               int uid             = -1)
            : _profiler(profiler), _tag(tag), _uid(uid), _deviceIndex(deviceIndex), _stream(stream) {
            _profiler.initializeProfile(_tag, _uid);
            _profiler.start(nullptr, _deviceIndex, _stream);
        }
        ~Scoped() {
            _profiler.stop(nullptr, _deviceIndex, _stream);
            _profiler.finalizeProfile(_tag, _uid);
        }
    };
};

struct TorchDeviceMemoryAllocator {
    static inline nrend::ErrorCode allocAsync(void*& ptr, size_t size, uint64_t stream, const nrend::Logger& logger) {
        ptr = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, reinterpret_cast<cudaStream_t>(stream));
        if (ptr == nullptr) {
            LOG_ERROR(logger, "TorchDeviceMemoryAllocator: Failed to allocate %zu bytes", size);
            return nrend::ErrorCode::OutOfMemory;
        }
        return nrend::ErrorCode::None;
    }

    static inline nrend::ErrorCode freeAsync(void* ptr, uint64_t, const nrend::Logger&) {
        c10::cuda::CUDACachingAllocator::raw_delete(ptr);
        return nrend::ErrorCode::None;
    }

    static inline nrend::ErrorCode free(void* ptr, const nrend::Logger&) {
        c10::cuda::CUDACachingAllocator::raw_delete(ptr);
        return nrend::ErrorCode::None;
    }
};

class NRendererWrapper {

    INRenderer::RendererHandle m_rendererHandle = INRenderer::InvalidRendererHandle;
    struct Version {
        int major        = 0;
        int minor        = 0;
        int patch        = 0;
        const char* name = "";
    } m_modelVersion;

    struct Parameters {
        // NOTE: We do not use INRenderer::RenderParameters::defaultHitTransmittance as the value is too low and only useful for the Kit. This value
        //      controls whether a ray hit is considered as a front face hit or not. Only the front face hit will produce hit depth. In NRE we want
        //      every ray hit to produce a hit depth, so we set it to 1.f.
        float defaultHitTransmittance = 1.f;

        void fromVersion(const Version& version) {
            // NOTE: 0.2.668 is the first version to support the defaultHitTransmittance parameter
            if (version.major <= 0 && version.minor <= 2 && version.patch < 668) {
                defaultHitTransmittance = INRenderer::RenderParameters::defaultHitTransmittance;
            }
        }
    } m_parameters;

    RenderingFeaturesLayout m_renderingCameraFeaturesLayout;
    RenderingFeaturesLayout m_renderingLidarFeaturesLayout;
    bool m_hasSceneData = false;
    NRendererProfiler m_profiler;
    int m_logLevel;

    static void NREND_LOGGER_CB logCallback(uint8_t level, const char* msg, void* data) {
        std::ostream& stream = (level > LoggerParameters::Error) ? std::cout : std::cerr;
        stream << "[NuRec::NRend][" << LoggerParameters::levelToString(level) << "] ::: "
               << msg << std::flush << std::endl;
    }

    static void NREND_LOGGER_CB deviceLaunchCallback(bool start,
                                                     const char* tag,
                                                     int deviceIndex,
                                                     uint64_t deviceQueue,
                                                     void* data) {
        NRendererProfiler* loggerPtr = reinterpret_cast<NRendererProfiler*>(data);
        if (loggerPtr) {
            if (start) {
                loggerPtr->start(tag, deviceIndex, reinterpret_cast<cudaStream_t>(deviceQueue));
            } else {
                loggerPtr->stop(tag, deviceIndex, reinterpret_cast<cudaStream_t>(deviceQueue));
            }
        }
    }

public:
    using SensorModelVariant = std::variant<INRenderer::SensorProjectionModel,
                                            OpenCVPinholeCameraModelParameters,
                                            OpenCVFisheyeCameraModelParameters,
                                            FThetaCameraModelParameters,
                                            RowOffsetStructuredSpinningLidarModelParameters>;

    NRendererWrapper(py::buffer modelPyBuffer,
                     py::buffer renderSettingsPyBuffer,
                     py::list trackInstancesStrUIdList,
                     int rendererHint,
                     int logLevel,
                     float profilingFrequency,
                     bool differentiable,
                     bool computeNormals,
                     bool computeParticlesCumulatedWeights,
                     bool computeParticlesVisibility)
        : m_renderingCameraFeaturesLayout{0, 0, 0, false, false}, m_renderingLidarFeaturesLayout{0, 0, 0, false, false}, m_profiler(profilingFrequency), m_logLevel(logLevel) {

        m_logLevel = std::min(logLevel, static_cast<int>(NREND_MAX_LOG_LEVEL));

        // convert the track instances UId map list[str] into a vector of c-string
        std::vector<std::string> trackInstancesStrUIdsCache;
        trackInstancesStrUIdsCache.reserve(trackInstancesStrUIdList.size());
        std::vector<const char*> trackInstancesStrUIds;
        trackInstancesStrUIds.reserve(trackInstancesStrUIdsCache.size());
        for (const auto& instanceId : trackInstancesStrUIdList) {
            trackInstancesStrUIdsCache.push_back(instanceId.cast<std::string>());
            trackInstancesStrUIds.push_back(trackInstancesStrUIdsCache.back().c_str());
        }

        // convert packed dictionnary buffer to a buffer ptr/size
        py::buffer_info modelPyBufferInfos          = modelPyBuffer.request();
        py::buffer_info renderSettingsPyBufferInfos = renderSettingsPyBuffer.request();
        const bool validRenderSettings              = renderSettingsPyBufferInfos.size > 1; //< encoding an empty dict as a buffer of size 1

        // build rendering flags from input parameters
        const auto renderingOptFlags = static_cast<RenderingParameters::OptFlags>(
            RenderingParameters::OptNREReferential |
            (differentiable ? RenderingParameters::OptDifferentiable : RenderingParameters::OptNone) |
            (computeNormals ? RenderingParameters::OptNone : RenderingParameters::OptDisableNormals) |
            (computeParticlesCumulatedWeights ? RenderingParameters::OptEnableParticleCumulatedWeights : RenderingParameters::OptNone) |
            (computeParticlesVisibility ? RenderingParameters::OptEnableParticleVisibility : RenderingParameters::OptNone));

        m_hasSceneData = computeParticlesCumulatedWeights || computeParticlesVisibility;

        auto nrendStatus = INRenderer::create(
            {reinterpret_cast<const char*>(modelPyBufferInfos.ptr), static_cast<size_t>(modelPyBufferInfos.size)},
            {reinterpret_cast<const char*>(validRenderSettings ? renderSettingsPyBufferInfos.ptr : nullptr),
             static_cast<size_t>(validRenderSettings ? renderSettingsPyBufferInfos.size : 0)},
            {static_cast<RenderingParameters::RendererHints>(rendererHint),
             renderingOptFlags,
             TrackInstancesUIdsSpan{trackInstancesStrUIds.size(), trackInstancesStrUIds.data()}},
            LoggerParameters{static_cast<uint8_t>(m_logLevel),
                             logCallback,
                             nullptr,
                             profilingFrequency > 0.f ? deviceLaunchCallback : nullptr,
                             profilingFrequency > 0.f ? &m_profiler : nullptr},
            m_rendererHandle);

        if (NREND_SUCCESS(nrendStatus)) {
            nrendStatus = INRenderer::getModelVersion(m_rendererHandle, m_modelVersion.major, m_modelVersion.minor, m_modelVersion.patch, m_modelVersion.name);
            // setup specific parameters based on the model version for backward compatibility
            m_parameters.fromVersion(m_modelVersion);
        }
        if (NREND_SUCCESS(nrendStatus)) {
            nrendStatus = INRenderer::renderingFeaturesLayout(m_rendererHandle, INRenderer::SensorType::Camera, m_renderingCameraFeaturesLayout);
        }
        if (NREND_SUCCESS(nrendStatus)) {
            nrendStatus = INRenderer::renderingFeaturesLayout(m_rendererHandle, INRenderer::SensorType::Lidar, m_renderingLidarFeaturesLayout);
        }
        if (NREND_SUCCESS(nrendStatus)) {
            nrendStatus = INRenderer::setDeviceAllocator({&TorchDeviceMemoryAllocator::allocAsync,
                                                          &TorchDeviceMemoryAllocator::freeAsync,
                                                          &TorchDeviceMemoryAllocator::free});
        }
        if (NREND_FAILED(nrendStatus) && valid()) {
            INRenderer::destroy(m_rendererHandle);
            m_rendererHandle = INRenderer::InvalidRendererHandle;
        }
    }

    bool
    valid() const {
        return (m_rendererHandle != INRenderer::InvalidRendererHandle);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, NRendererForwardContextWrapper, bool>
    render(int id,
           int frameWidth,
           int frameHeight,
           py::tuple frameTileInfos, //< (frameTileOffsetX, frameTileOffsetY, frameTileWidth, frameTileHeight)
           INRenderer::TTimestamp startTimestamp,
           INRenderer::TTimestamp endTimestamp,
           torch::Tensor rayOrigins,
           torch::Tensor rayDirections,
           torch::Tensor rayTimestamps,
           SensorModelVariant sensorModelVariant,
           torch::Tensor sensorsIds,
           torch::Tensor sensorsStartPose,
           torch::Tensor sensorsEndPose,
           int numActiveTrackInstances,
           torch::Tensor activeTrackInstancesIds,
           torch::Tensor activeTrackInstancesStartPose,
           torch::Tensor activeTrackInstancesEndPose) {

        auto cudaDeviceIndex    = rayOrigins.get_device();
        cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

        NRendererProfiler::Scoped profiling(m_profiler, "render", cudaDeviceIndex, cudaStream, sensorModelVariant.index());

        const int frameTileOffsetX = frameTileInfos[0].cast<int>();
        const int frameTileOffsetY = frameTileInfos[1].cast<int>();
        const int frameTileWidth   = frameTileInfos[2].cast<int>();
        const int frameTileHeight  = frameTileInfos[3].cast<int>();

        const bool renderingLidar                              = std::holds_alternative<RowOffsetStructuredSpinningLidarModelParameters>(sensorModelVariant);
        const RenderingFeaturesLayout& renderingFeaturesLayout = renderingLidar ? m_renderingLidarFeaturesLayout : m_renderingCameraFeaturesLayout;
        const int rayExtendedFeaturesDim                       = renderingFeaturesLayout.extendedFeaturesDim + renderingFeaturesLayout.sensorExtendedFeaturesDim;

        // Query scene data layout each render call since numElements may change
        uint32_t renderingSceneDataSize = 0;
        RenderingSceneDataLayout sceneDataLayout;
        if (m_hasSceneData) {
            INRenderer::renderingSceneDataLayout(m_rendererHandle, renderingSceneDataSize, sceneDataLayout);
        }

        auto floatTensorOptions                                 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cudaDeviceIndex);
        torch::Tensor radianceDensity                           = torch::zeros({frameTileHeight, frameTileWidth, renderingFeaturesLayout.baseFeaturesDim + 1}, floatTensorOptions);
        torch::Tensor objectRayHitDistance                      = torch::zeros({frameTileHeight, frameTileWidth, 1}, floatTensorOptions);
        torch::Tensor objectRayHitNormal                        = renderingFeaturesLayout.computeNormals ? torch::zeros({frameTileHeight, frameTileWidth, 3}, floatTensorOptions) : torch::empty({0}, floatTensorOptions);
        torch::Tensor rayExtendedFeatures                       = torch::zeros({frameTileHeight, frameTileWidth, rayExtendedFeaturesDim}, floatTensorOptions);
        torch::Tensor sceneData                                 = m_hasSceneData ? torch::zeros({renderingSceneDataSize, sceneDataLayout.count()}, floatTensorOptions) : torch::empty({0}, floatTensorOptions);
        INRenderer::RenderingContextHandle forwardContextHandle = INRenderer::InvalidRenderingContextHandle;

        ErrorCode nrendStatus = ErrorCode::None;

        // Check if frameTileInfos are out of bounds
        if (frameTileOffsetX + frameTileWidth > frameWidth || frameTileOffsetY + frameTileHeight > frameHeight) {
            if (m_logLevel >= LoggerParameters::Error) {
                logCallback(LoggerParameters::Error, "frameTileInfos are out of bounds", nullptr);
            }
            nrendStatus = ErrorCode::BadInput;
        }

        // Check if rayOrigins tensor shape is compatible with frameTileInfos
        if (rayOrigins.numel() != frameTileHeight * frameTileWidth * 3) {
            if (m_logLevel >= LoggerParameters::Error) {
                logCallback(LoggerParameters::Error, "rayOrigins tensor size is not compatible with frameTileInfos", nullptr);
            }
            nrendStatus = ErrorCode::BadInput;
        }

        if (valid() && NREND_SUCCESS(nrendStatus)) {

            INRenderer::RenderParameters renderParameters;
            renderParameters.id                      = id;
            renderParameters.frameResolution         = INRenderer::Vec2{static_cast<float>(frameWidth), static_cast<float>(frameHeight)};
            renderParameters.frameTileOffset         = INRenderer::Vec2{static_cast<float>(frameTileOffsetX), static_cast<float>(frameTileOffsetY)};
            renderParameters.frameTileResolution     = INRenderer::IVec2{frameTileWidth, frameTileHeight};
            renderParameters.sensorModel             = toSensorModel(sensorModelVariant);
            renderParameters.sensorState             = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
            renderParameters.hitTransmittance        = m_parameters.defaultHitTransmittance;
            renderParameters.objectAABB              = INRenderer::BoundingBox{INRenderer::Vec3{-1e06f, -1e06f, -1e06f}, INRenderer::Vec3{1e06f, 1e06f, 1e06f}};
            renderParameters.worldToObjectTransform  = INRenderer::Mat4x3::identity();
            renderParameters.objectToWorldTransform  = INRenderer::Mat4x3::identity();
            renderParameters.colorCorrectionMatrix   = INRenderer::Mat4x3::identity();
            renderParameters.objectInstanceIds       = INRenderer::UVec4{0u, 0u, 0u, 0u};
            renderParameters.numActiveTrackInstances = numActiveTrackInstances;

            if (renderParameters.sensorModel.modelType == INRenderer::SensorProjectionModel::Unsupported) {
                if (m_logLevel >= LoggerParameters::Error) {
                    logCallback(LoggerParameters::Error, "Unsupported sensor model", nullptr);
                }
                nrendStatus = ErrorCode::BadInput;
            }

            if (NREND_SUCCESS(nrendStatus)) {
                nrendStatus = INRenderer::render(m_rendererHandle,
                                                 renderParameters,
                                                 reinterpret_cast<const INRenderer::Vec3*>(voidDataPtr(rayOrigins)),
                                                 reinterpret_cast<const INRenderer::Vec3*>(voidDataPtr(rayDirections)),
                                                 reinterpret_cast<const INRenderer::TTimestamp*>(voidDataPtr(rayTimestamps)),
                                                 reinterpret_cast<const INRenderer::IVec2*>(voidDataPtr(sensorsIds)),
                                                 reinterpret_cast<const INRenderer::IVec2*>(voidDataPtr(activeTrackInstancesIds)),
                                                 reinterpret_cast<const INRenderer::TTrackInstancePose*>(voidDataPtr(activeTrackInstancesStartPose)),
                                                 reinterpret_cast<const INRenderer::TTrackInstancePose*>(voidDataPtr(activeTrackInstancesEndPose)),
                                                 nullptr, //< instance ids buffer
                                                 reinterpret_cast<float*>(voidDataPtr(objectRayHitDistance)),
                                                 renderingFeaturesLayout.computeNormals ? reinterpret_cast<INRenderer::Vec3*>(voidDataPtr(objectRayHitNormal)) : nullptr,
                                                 reinterpret_cast<INRenderer::Vec4*>(voidDataPtr(radianceDensity)),
                                                 rayExtendedFeaturesDim > 0 ? reinterpret_cast<float*>(voidDataPtr(rayExtendedFeatures)) : nullptr,
                                                 m_hasSceneData ? reinterpret_cast<float*>(voidDataPtr(sceneData)) : nullptr,
                                                 cudaDeviceIndex,
                                                 reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream),
                                                 &forwardContextHandle);
            }
        }

        return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, NRendererForwardContextWrapper, bool>(
            radianceDensity,
            objectRayHitDistance,
            objectRayHitNormal,
            rayExtendedFeatures,
            sceneData,
            NRendererForwardContextWrapper(forwardContextHandle),
            NREND_SUCCESS(nrendStatus));
    }

    ~NRendererWrapper() {
        INRenderer::destroy(m_rendererHandle);
    }

    std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>, bool>
    renderBackward(int id,
                   int frameWidth,
                   int frameHeight,
                   py::tuple frameTileInfos, //< (frameTileOffsetX, frameTileOffsetY, frameTileWidth, frameTileHeight)
                   INRenderer::TTimestamp startTimestamp,
                   INRenderer::TTimestamp endTimestamp,
                   torch::Tensor rayOrigins,
                   torch::Tensor rayDirections,
                   std::map<std::string, torch::Tensor> differentiatedParametersStateDict,
                   torch::Tensor rayTimestamps,
                   SensorModelVariant sensorModelVariant,
                   torch::Tensor sensorsIds,
                   torch::Tensor sensorsStartPose,
                   torch::Tensor sensorsEndPose,
                   int numActiveTrackInstances,
                   torch::Tensor activeTrackInstancesIds,
                   torch::Tensor activeTrackInstancesStartPose,
                   torch::Tensor activeTrackInstancesEndPose,
                   torch::Tensor rayRadianceDensity,
                   torch::Tensor rayRadianceDensityGradient,
                   torch::Tensor rayHitDistance,
                   torch::Tensor rayHitDistanceGradient,
                   torch::Tensor rayHitNormal,
                   torch::Tensor rayHitNormalGradient,
                   torch::Tensor rayExtendedFeatures,
                   torch::Tensor rayExtendedFeaturesGradient,
                   NRendererForwardContextWrapper& forwardContext) {

        const int cudaDeviceIndex = rayOrigins.get_device();
        cudaStream_t cudaStream   = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

        NRendererProfiler::Scoped profiling(m_profiler, "render-backward", cudaDeviceIndex, cudaStream, sensorModelVariant.index());

        const int frameTileOffsetX = frameTileInfos[0].cast<int>();
        const int frameTileOffsetY = frameTileInfos[1].cast<int>();
        const int frameTileWidth   = frameTileInfos[2].cast<int>();
        const int frameTileHeight  = frameTileInfos[3].cast<int>();

        const bool renderingLidar                              = std::holds_alternative<RowOffsetStructuredSpinningLidarModelParameters>(sensorModelVariant);
        const RenderingFeaturesLayout& renderingFeaturesLayout = renderingLidar ? m_renderingLidarFeaturesLayout : m_renderingCameraFeaturesLayout;

        auto floatTensorOptions             = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cudaDeviceIndex);
        torch::Tensor raysOriginGradient    = renderingFeaturesLayout.computeRayGradients ? torch::zeros_like(rayOrigins) : torch::empty({0}, floatTensorOptions);
        torch::Tensor raysDirectionGradient = renderingFeaturesLayout.computeRayGradients ? torch::zeros_like(rayDirections) : torch::empty({0}, floatTensorOptions);
        std::map<std::string, torch::Tensor> parametersGradientStateDict;
        std::vector<NamedParameterDefinition> parametersGradientStateDictDefs;
        parametersGradientStateDictDefs.reserve(differentiatedParametersStateDict.size());
        for (const auto& paramDictEntry : differentiatedParametersStateDict) {
            auto tensorGradient                               = torch::zeros_like(paramDictEntry.second);
            parametersGradientStateDict[paramDictEntry.first] = tensorGradient;
            parametersGradientStateDictDefs.push_back({paramDictEntry.first.c_str(),
                                                       ParameterDefinition{tensorGradient.nbytes(),
                                                                           voidDataPtr(tensorGradient),
                                                                           ParameterDefinition::Buffer}});
        }

        ErrorCode nrendStatus = ErrorCode::None;

        // Check if frameTileInfos are out of bounds
        if (frameTileOffsetX + frameTileWidth > frameWidth || frameTileOffsetY + frameTileHeight > frameHeight) {
            if (m_logLevel >= LoggerParameters::Error) {
                logCallback(LoggerParameters::Error, "frameTileInfos are out of bounds", nullptr);
            }
            nrendStatus = ErrorCode::BadInput;
        }

        // Check if rayOrigins tensor shape is compatible with frameTileInfos
        if (rayOrigins.numel() != frameTileHeight * frameTileWidth * 3) {
            if (m_logLevel >= LoggerParameters::Error) {
                logCallback(LoggerParameters::Error, "rayOrigins tensor size is not compatible with frameTileInfos", nullptr);
            }
            nrendStatus = ErrorCode::BadInput;
        }

        if (valid() && NREND_SUCCESS(nrendStatus)) {

            INRenderer::RenderParameters renderParameters;
            renderParameters.id                      = id;
            renderParameters.frameResolution         = INRenderer::Vec2{static_cast<float>(frameWidth), static_cast<float>(frameHeight)};
            renderParameters.frameTileOffset         = INRenderer::Vec2{static_cast<float>(frameTileOffsetX), static_cast<float>(frameTileOffsetY)};
            renderParameters.frameTileResolution     = INRenderer::IVec2{frameTileWidth, frameTileHeight};
            renderParameters.sensorModel             = toSensorModel(sensorModelVariant);
            renderParameters.sensorState             = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
            renderParameters.hitTransmittance        = m_parameters.defaultHitTransmittance;
            renderParameters.objectAABB              = INRenderer::BoundingBox{INRenderer::Vec3{-1e06f, -1e06f, -1e06f}, INRenderer::Vec3{1e06f, 1e06f, 1e06f}};
            renderParameters.worldToObjectTransform  = INRenderer::Mat4x3::identity();
            renderParameters.objectToWorldTransform  = INRenderer::Mat4x3::identity();
            renderParameters.colorCorrectionMatrix   = INRenderer::Mat4x3::identity();
            renderParameters.objectInstanceIds       = INRenderer::UVec4{0u};
            renderParameters.numActiveTrackInstances = numActiveTrackInstances;

            if (renderParameters.sensorModel.modelType == INRenderer::SensorProjectionModel::Unsupported) {
                if (m_logLevel >= LoggerParameters::Error) {
                    logCallback(LoggerParameters::Error, "Unsupported sensor model", nullptr);
                }
                nrendStatus = ErrorCode::BadInput;
            }

            if (NREND_SUCCESS(nrendStatus)) {
                nrendStatus = INRenderer::updateModelParameters(
                    m_rendererHandle,
                    NamedParameterDefinitionsSpan{parametersGradientStateDictDefs.size(), parametersGradientStateDictDefs.data()},
                    true /*gradient*/,
                    false /*copy*/,
                    cudaDeviceIndex,
                    reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream));
            }

            if (NREND_SUCCESS(nrendStatus)) {
                const bool hasExtendedFeatures = renderingFeaturesLayout.extendedFeaturesDim + renderingFeaturesLayout.sensorExtendedFeaturesDim > 0;
                nrendStatus                    = INRenderer::renderBackward(
                    m_rendererHandle,
                    renderParameters,
                    reinterpret_cast<const INRenderer::Vec3*>(voidDataPtr(rayOrigins)),
                    reinterpret_cast<const INRenderer::Vec3*>(voidDataPtr(rayDirections)),
                    reinterpret_cast<const INRenderer::TTimestamp*>(voidDataPtr(rayTimestamps)),
                    reinterpret_cast<const INRenderer::IVec2*>(voidDataPtr(sensorsIds)),
                    reinterpret_cast<const INRenderer::IVec2*>(voidDataPtr(activeTrackInstancesIds)),
                    reinterpret_cast<const INRenderer::TTrackInstancePose*>(voidDataPtr(activeTrackInstancesStartPose)),
                    reinterpret_cast<const INRenderer::TTrackInstancePose*>(voidDataPtr(activeTrackInstancesEndPose)),
                    nullptr, //< instance ids buffer
                    reinterpret_cast<float*>(voidDataPtr(rayHitDistance)),
                    reinterpret_cast<float*>(voidDataPtr(rayHitDistanceGradient)),
                    renderingFeaturesLayout.computeNormals ? reinterpret_cast<INRenderer::Vec3*>(voidDataPtr(rayHitNormal)) : nullptr,
                    renderingFeaturesLayout.computeNormals ? reinterpret_cast<INRenderer::Vec3*>(voidDataPtr(rayHitNormalGradient)) : nullptr,
                    reinterpret_cast<INRenderer::Vec4*>(voidDataPtr(rayRadianceDensity)),
                    reinterpret_cast<INRenderer::Vec4*>(voidDataPtr(rayRadianceDensityGradient)),
                    hasExtendedFeatures ? reinterpret_cast<float*>(voidDataPtr(rayExtendedFeatures)) : nullptr,
                    hasExtendedFeatures ? reinterpret_cast<float*>(voidDataPtr(rayExtendedFeaturesGradient)) : nullptr,
                    renderingFeaturesLayout.computeRayGradients ? reinterpret_cast<INRenderer::Vec3*>(voidDataPtr(raysOriginGradient)) : nullptr,
                    renderingFeaturesLayout.computeRayGradients ? reinterpret_cast<INRenderer::Vec3*>(voidDataPtr(raysDirectionGradient)) : nullptr,
                    cudaDeviceIndex,
                    reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream),
                    forwardContext.handle());
            } else {
                nrendStatus = ErrorCode::InvalidResource;
            }

            if (NREND_SUCCESS(nrendStatus)) {
                nrendStatus = INRenderer::detachModelParameters(m_rendererHandle,
                                                                true /*gradient*/,
                                                                false /*copy*/,
                                                                cudaDeviceIndex,
                                                                reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream));
            }
        }

        return std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>, bool>(
            raysOriginGradient, raysDirectionGradient, parametersGradientStateDict, NREND_SUCCESS(nrendStatus));
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, NRendererForwardContextWrapper, bool>
    prepareScene(int id,
                 int frameWidth,
                 int frameHeight,
                 py::tuple frameTileInfos, //< (frameTileOffsetX, frameTileOffsetY, frameTileWidth, frameTileHeight)
                 INRenderer::TTimestamp startTimestamp,
                 INRenderer::TTimestamp endTimestamp,
                 SensorModelVariant sensorModelVariant,
                 torch::Tensor sensorsStartPose,
                 torch::Tensor sensorsEndPose,
                 int numActiveTrackInstances,
                 torch::Tensor activeTrackInstancesIds,
                 torch::Tensor activeTrackInstancesStartPose,
                 torch::Tensor activeTrackInstancesEndPose) {

        auto cudaDeviceIndex    = activeTrackInstancesIds.get_device();
        cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

        NRendererProfiler::Scoped profiling(m_profiler, "prepare-scene", cudaDeviceIndex, cudaStream, sensorModelVariant.index());

        ErrorCode nrendStatus = ErrorCode::None;

        const int frameTileOffsetX = frameTileInfos[0].cast<int>();
        const int frameTileOffsetY = frameTileInfos[1].cast<int>();
        const int frameTileWidth   = frameTileInfos[2].cast<int>();
        const int frameTileHeight  = frameTileInfos[3].cast<int>();

        // Check if frameTileInfos are out of bounds
        if (frameTileOffsetX + frameTileWidth > frameWidth || frameTileOffsetY + frameTileHeight > frameHeight) {
            if (m_logLevel >= LoggerParameters::Error) {
                logCallback(LoggerParameters::Error, "frameTileInfos are out of bounds", nullptr);
            }
            nrendStatus = ErrorCode::BadInput;
        }

        uint32_t sceneSize                       = 0;
        uint32_t sceneDensitySize                = 0;
        uint32_t sceneFeaturesSize               = 0;
        uint32_t sceneExtendedFeaturesSize       = 0;
        uint32_t sceneSensorExtendedFeaturesSize = 0;
        bool halfPrecisionScene                  = false;

        torch::TensorOptions floatTensorOptions = torch::TensorOptions().dtype(halfPrecisionScene ? torch::kFloat16 : torch::kFloat32).device(torch::kCUDA, cudaDeviceIndex);

        torch::Tensor sceneDensityTensor;
        torch::Tensor sceneFeaturesTensor;
        torch::Tensor sceneExtendedFeaturesTensor;
        torch::Tensor sceneSensorExtendedFeaturesTensor;
        torch::Tensor sceneDataTensor;

        if (NREND_SUCCESS(nrendStatus)) {
            const bool renderingLidar = std::holds_alternative<RowOffsetStructuredSpinningLidarModelParameters>(sensorModelVariant);
            nrendStatus               = INRenderer::sceneLayout(m_rendererHandle,
                                                  renderingLidar ? INRenderer::SensorType::Lidar : INRenderer::SensorType::Camera,
                                                                sceneSize,
                                                                sceneDensitySize,
                                                                sceneFeaturesSize,
                                                                sceneExtendedFeaturesSize,
                                                                sceneSensorExtendedFeaturesSize,
                                                                halfPrecisionScene);
            if (NREND_SUCCESS(nrendStatus)) {
                sceneDensityTensor                = torch::empty({sceneSize, sceneDensitySize}, floatTensorOptions);
                sceneFeaturesTensor               = torch::empty({sceneSize, sceneFeaturesSize}, floatTensorOptions);
                sceneExtendedFeaturesTensor       = torch::empty({sceneSize, sceneExtendedFeaturesSize}, floatTensorOptions);
                sceneSensorExtendedFeaturesTensor = torch::empty({sceneSize, sceneSensorExtendedFeaturesSize}, floatTensorOptions);
            }
        }

        if (NREND_SUCCESS(nrendStatus)) {
            // Query scene data layout each render call since numElements may change
            uint32_t renderingSceneDataSize = 0;
            RenderingSceneDataLayout sceneDataLayout;
            if (m_hasSceneData) {
                nrendStatus = INRenderer::renderingSceneDataLayout(m_rendererHandle, renderingSceneDataSize, sceneDataLayout);
            }
            if (NREND_SUCCESS(nrendStatus)) {
                sceneDataTensor = torch::zeros({renderingSceneDataSize, sceneDataLayout.count()}, floatTensorOptions);
            }
        }

        INRenderer::RenderingContextHandle forwardContextHandle = INRenderer::InvalidRenderingContextHandle;

        if (valid() && NREND_SUCCESS(nrendStatus)) {

            INRenderer::RenderParameters renderParameters;
            renderParameters.id                      = id;
            renderParameters.frameResolution         = INRenderer::Vec2{static_cast<float>(frameWidth), static_cast<float>(frameHeight)};
            renderParameters.frameTileOffset         = INRenderer::Vec2{static_cast<float>(frameTileOffsetX), static_cast<float>(frameTileOffsetY)};
            renderParameters.frameTileResolution     = INRenderer::IVec2{frameTileWidth, frameTileHeight};
            renderParameters.sensorModel             = toSensorModel(sensorModelVariant);
            renderParameters.sensorState             = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
            renderParameters.hitTransmittance        = m_parameters.defaultHitTransmittance;
            renderParameters.objectAABB              = INRenderer::BoundingBox{INRenderer::Vec3{-1e06f, -1e06f, -1e06f}, INRenderer::Vec3{1e06f, 1e06f, 1e06f}};
            renderParameters.worldToObjectTransform  = INRenderer::Mat4x3::identity();
            renderParameters.objectToWorldTransform  = INRenderer::Mat4x3::identity();
            renderParameters.colorCorrectionMatrix   = INRenderer::Mat4x3::identity();
            renderParameters.objectInstanceIds       = INRenderer::UVec4{0u, 0u, 0u, 0u};
            renderParameters.numActiveTrackInstances = numActiveTrackInstances;

            if (renderParameters.sensorModel.modelType == INRenderer::SensorProjectionModel::Unsupported) {
                if (m_logLevel >= LoggerParameters::Error) {
                    logCallback(LoggerParameters::Error, "Unsupported sensor model", nullptr);
                }
                nrendStatus = ErrorCode::BadInput;
            }

            if (NREND_SUCCESS(nrendStatus)) {

                uint32_t validSceneSize = 0;
                nrendStatus             = INRenderer::prepareScene(m_rendererHandle,
                                                                   renderParameters,
                                                                   reinterpret_cast<const INRenderer::IVec2*>(voidDataPtr(activeTrackInstancesIds)),
                                                                   reinterpret_cast<const INRenderer::TTrackInstancePose*>(voidDataPtr(activeTrackInstancesStartPose)),
                                                                   reinterpret_cast<const INRenderer::TTrackInstancePose*>(voidDataPtr(activeTrackInstancesEndPose)),
                                                                   voidDataPtr(sceneDensityTensor),
                                                                   voidDataPtr(sceneFeaturesTensor),
                                                                   voidDataPtr(sceneExtendedFeaturesTensor),
                                                                   voidDataPtr(sceneSensorExtendedFeaturesTensor),
                                                                   voidDataPtr(sceneDataTensor),
                                                                   validSceneSize,
                                                                   cudaDeviceIndex,
                                                                   reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream),
                                                                   &forwardContextHandle);
                if (NREND_SUCCESS(nrendStatus)) {
                    sceneDensityTensor.resize_({validSceneSize, sceneDensitySize});
                    sceneFeaturesTensor.resize_({validSceneSize, sceneFeaturesSize});
                    sceneExtendedFeaturesTensor.resize_({validSceneSize, sceneExtendedFeaturesSize});
                    sceneSensorExtendedFeaturesTensor.resize_({validSceneSize, sceneSensorExtendedFeaturesSize});
                }
            }
        }

        return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, NRendererForwardContextWrapper, bool>(
            sceneDensityTensor,
            sceneFeaturesTensor,
            sceneExtendedFeaturesTensor,
            sceneSensorExtendedFeaturesTensor,
            sceneDataTensor,
            NRendererForwardContextWrapper(forwardContextHandle),
            NREND_SUCCESS(nrendStatus));
    }

    std::tuple<std::map<std::string, torch::Tensor>, bool>
    prepareSceneBackward(int id,
                         int frameWidth,
                         int frameHeight,
                         py::tuple frameTileInfos, //< (frameTileOffsetX, frameTileOffsetY, frameTileWidth, frameTileHeight)
                         INRenderer::TTimestamp startTimestamp,
                         INRenderer::TTimestamp endTimestamp,
                         std::map<std::string, torch::Tensor> differentiatedParametersStateDict,
                         SensorModelVariant sensorModelVariant,
                         torch::Tensor sensorsStartPose,
                         torch::Tensor sensorsEndPose,
                         int numActiveTrackInstances,
                         torch::Tensor activeTrackInstancesIds,
                         torch::Tensor activeTrackInstancesStartPose,
                         torch::Tensor activeTrackInstancesEndPose,
                         torch::Tensor sceneDensityGradient,
                         torch::Tensor sceneFeatures,
                         torch::Tensor sceneFeaturesGradient,
                         torch::Tensor sceneExtendedFeatures,
                         torch::Tensor sceneExtendedFeaturesGradient,
                         torch::Tensor sceneSensorExtendedFeatures,
                         torch::Tensor sceneSensorExtendedFeaturesGradient,
                         NRendererForwardContextWrapper& forwardContext) {

        const int cudaDeviceIndex = sceneDensityGradient.get_device();
        cudaStream_t cudaStream   = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

        NRendererProfiler::Scoped profiling(m_profiler, "prepare-scene-backward", cudaDeviceIndex, cudaStream, sensorModelVariant.index());

        std::map<std::string, torch::Tensor> parametersGradientStateDict;
        std::vector<NamedParameterDefinition> parametersGradientStateDictDefs;
        parametersGradientStateDictDefs.reserve(differentiatedParametersStateDict.size());
        for (const auto& paramDictEntry : differentiatedParametersStateDict) {
            auto tensorGradient                               = torch::zeros_like(paramDictEntry.second);
            parametersGradientStateDict[paramDictEntry.first] = tensorGradient;
            parametersGradientStateDictDefs.push_back({paramDictEntry.first.c_str(),
                                                       ParameterDefinition{tensorGradient.nbytes(),
                                                                           voidDataPtr(tensorGradient),
                                                                           ParameterDefinition::Buffer}});
        }

        ErrorCode nrendStatus = ErrorCode::None;

        // Check if frameTileInfos are out of bounds
        const int frameTileOffsetX = frameTileInfos[0].cast<int>();
        const int frameTileOffsetY = frameTileInfos[1].cast<int>();
        const int frameTileWidth   = frameTileInfos[2].cast<int>();
        const int frameTileHeight  = frameTileInfos[3].cast<int>();

        if (frameTileOffsetX + frameTileWidth > frameWidth || frameTileOffsetY + frameTileHeight > frameHeight) {
            if (m_logLevel >= LoggerParameters::Error) {
                logCallback(LoggerParameters::Error, "frameTileInfos are out of bounds", nullptr);
            }
            nrendStatus = ErrorCode::BadInput;
        }

        if (valid() && NREND_SUCCESS(nrendStatus)) {

            INRenderer::RenderParameters renderParameters;
            renderParameters.id                      = id;
            renderParameters.frameResolution         = INRenderer::Vec2{static_cast<float>(frameWidth), static_cast<float>(frameHeight)};
            renderParameters.frameTileOffset         = INRenderer::Vec2{static_cast<float>(frameTileOffsetX), static_cast<float>(frameTileOffsetY)};
            renderParameters.frameTileResolution     = INRenderer::IVec2{frameTileWidth, frameTileHeight};
            renderParameters.sensorModel             = toSensorModel(sensorModelVariant);
            renderParameters.sensorState             = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
            renderParameters.hitTransmittance        = m_parameters.defaultHitTransmittance;
            renderParameters.objectAABB              = INRenderer::BoundingBox{INRenderer::Vec3{-1e06f, -1e06f, -1e06f}, INRenderer::Vec3{1e06f, 1e06f, 1e06f}};
            renderParameters.worldToObjectTransform  = INRenderer::Mat4x3::identity();
            renderParameters.objectToWorldTransform  = INRenderer::Mat4x3::identity();
            renderParameters.colorCorrectionMatrix   = INRenderer::Mat4x3::identity();
            renderParameters.objectInstanceIds       = INRenderer::UVec4{0u};
            renderParameters.numActiveTrackInstances = numActiveTrackInstances;

            if (renderParameters.sensorModel.modelType == INRenderer::SensorProjectionModel::Unsupported) {
                if (m_logLevel >= LoggerParameters::Error) {
                    logCallback(LoggerParameters::Error, "Unsupported sensor model", nullptr);
                }
                nrendStatus = ErrorCode::BadInput;
            }

            if (NREND_SUCCESS(nrendStatus)) {
                nrendStatus = INRenderer::updateModelParameters(
                    m_rendererHandle,
                    NamedParameterDefinitionsSpan{parametersGradientStateDictDefs.size(), parametersGradientStateDictDefs.data()},
                    true /*gradient*/,
                    false /*copy*/,
                    cudaDeviceIndex,
                    reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream));
            }

            if (NREND_SUCCESS(nrendStatus)) {
                nrendStatus = INRenderer::prepareSceneBackward(
                    m_rendererHandle,
                    renderParameters,
                    reinterpret_cast<const INRenderer::IVec2*>(voidDataPtr(activeTrackInstancesIds)),
                    reinterpret_cast<const INRenderer::TTrackInstancePose*>(voidDataPtr(activeTrackInstancesStartPose)),
                    reinterpret_cast<const INRenderer::TTrackInstancePose*>(voidDataPtr(activeTrackInstancesEndPose)),
                    voidDataPtr(sceneFeatures),
                    voidDataPtr(sceneExtendedFeatures),
                    voidDataPtr(sceneSensorExtendedFeatures),
                    voidDataPtr(sceneDensityGradient),
                    voidDataPtr(sceneFeaturesGradient),
                    voidDataPtr(sceneExtendedFeaturesGradient),
                    voidDataPtr(sceneSensorExtendedFeaturesGradient),
                    cudaDeviceIndex,
                    reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream),
                    forwardContext.handle());
            } else {
                nrendStatus = ErrorCode::InvalidResource;
            }

            if (NREND_SUCCESS(nrendStatus)) {
                nrendStatus = INRenderer::detachModelParameters(m_rendererHandle,
                                                                true /*gradient*/,
                                                                false /*copy*/,
                                                                cudaDeviceIndex,
                                                                reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream));
            }
        }

        return std::tuple<std::map<std::string, torch::Tensor>, bool>(parametersGradientStateDict, NREND_SUCCESS(nrendStatus));
    }

    bool updateModelParameters(std::map<std::string, torch::Tensor>& parametersStateDict, bool deepCopy) {
        if (!valid()) {
            return false;
        }

        if (parametersStateDict.empty()) {
            return true;
        }

        int cudaDeviceIndex = -1;
        std::vector<NamedParameterDefinition> namedParametersDefinition;
        for (auto& paramDictEntry : parametersStateDict) {
            const int deviceIndex = paramDictEntry.second.get_device();
            namedParametersDefinition.push_back({paramDictEntry.first.c_str(),
                                                 ParameterDefinition{paramDictEntry.second.nbytes(),
                                                                     voidDataPtr(paramDictEntry.second),
                                                                     deviceIndex == -1 ? ParameterDefinition::Value : ParameterDefinition::Buffer}});
            // check consistency of device
            if (deviceIndex != -1) {
                if (cudaDeviceIndex == -1) {
                    cudaDeviceIndex = deviceIndex;
                }
                // fail if gpu tensors are not on the same device
                if (cudaDeviceIndex != deviceIndex) {
                    return false;
                }
            }
        }

        if (cudaDeviceIndex == -1) {
            if (m_logLevel >= LoggerParameters::Error) {
                logCallback(LoggerParameters::Error, "invalid device index [-1]", nullptr);
            }
            return false;
        }

        cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

        NRendererProfiler::Scoped profiling(m_profiler, "update-model-parameters", cudaDeviceIndex, cudaStream);

        const auto nrendStatus = INRenderer::updateModelParameters(
            m_rendererHandle,
            NamedParameterDefinitionsSpan{namedParametersDefinition.size(), namedParametersDefinition.data()},
            false /*gradient*/,
            deepCopy,
            cudaDeviceIndex,
            reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream));
        return NREND_SUCCESS(nrendStatus);
    }

    bool detachModelParameters(int cudaDeviceIndex, bool deepCopy) {
        if (!valid()) {
            return false;
        }
        cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);
        const auto nrendStatus  = INRenderer::detachModelParameters(
            m_rendererHandle,
            false /*gradient*/,
            deepCopy,
            cudaDeviceIndex,
            reinterpret_cast<INRenderer::DeviceQueueHandle>(cudaStream));
        return NREND_SUCCESS(nrendStatus);
    }

    std::map<std::string, float> collectProfilings() {
        return m_profiler.collect(true);
    }

private:
    inline void* voidDataPtr(torch::Tensor& tensor) {
        if (tensor.size(0) == 0) {
            return nullptr;
        }
        switch (tensor.scalar_type()) {
        case torch::kFloat32:
            return tensor.contiguous().data_ptr<float>();
        case torch::kHalf:
            return tensor.contiguous().data_ptr<torch::Half>();
        case torch::kInt32:
            return tensor.contiguous().data_ptr<int32_t>();
        case torch::kInt64:
            return tensor.contiguous().data_ptr<int64_t>();
        case torch::kUInt32:
            return tensor.contiguous().data_ptr<uint32_t>();
        case torch::kUInt64:
            return tensor.contiguous().data_ptr<uint64_t>();
        default:
            throw std::runtime_error{"NRendererWrapper :: Unknown precision torch->void"};
        }
    }

    inline INRenderer::SensorProjectionModel toSensorModel(const SensorModelVariant& cameraModelVariant) {
        struct ExternalDistortionVisitor {
            INRenderer::SensorProjectionModel& cameraModelParams;
            ExternalDistortionVisitor(INRenderer::SensorProjectionModel& params)
                : cameraModelParams(params) {}

            INRenderer::SensorProjectionModel operator()(const std::monostate& params) const {
                cameraModelParams.externalDistortionType = INRenderer::SensorProjectionModel::EmptyExternalDistortionModel;
                return cameraModelParams;
            }
            INRenderer::SensorProjectionModel operator()(const BivariateWindshieldModelParameters& params) const {
                cameraModelParams.externalDistortionType                                      = INRenderer::SensorProjectionModel::BivariateWindshieldDistortion;
                cameraModelParams.bivariateWindshieldDistortionParameters.horizontalPolyOrder = computeWindshieldPolyOrder(INRenderer::BivariateWindshieldDistortionParameters::N_MAX_POLY_ORDER, params.horizontal_poly);
                cameraModelParams.bivariateWindshieldDistortionParameters.verticalPolyOrder   = computeWindshieldPolyOrder(INRenderer::BivariateWindshieldDistortionParameters::N_MAX_POLY_ORDER, params.vertical_poly);

                if (params.horizontal_poly_buffer == nullptr || params.vertical_poly_buffer == nullptr) {
                    throw std::runtime_error{"NRendererWrapper :: BivariateWindshieldModel expects non-null polynomial buffers"};
                }
                cameraModelParams.bivariateWindshieldDistortionParameters.horizontalPoly = reinterpret_cast<const float*>(params.horizontal_poly_buffer);
                cameraModelParams.bivariateWindshieldDistortionParameters.verticalPoly   = reinterpret_cast<const float*>(params.vertical_poly_buffer);

                return cameraModelParams;
            }
        };

        struct SensorModelVisitor {

            INRenderer::SensorProjectionModel operator()(INRenderer::SensorProjectionModel params) const {
                return params;
            }

            INRenderer::SensorProjectionModel operator()(OpenCVPinholeCameraModelParameters ocvPinholeParams) const {
                INRenderer::SensorProjectionModel params;
                params.shutterType = static_cast<INRenderer::SensorProjectionModel::ShutterType>(ocvPinholeParams.shutter_type);
                if (ocvPinholeParams.is_perfect_pinhole()) {
                    params.modelType = INRenderer::SensorProjectionModel::PerspectiveModel;
                    static_assert(sizeof(ocvPinholeParams.principal_point) == sizeof(INRenderer::Vec2), "NuRec::NREND : typing size mismatch");
                    params.perspectiveParams.principalPoint = *reinterpret_cast<const INRenderer::Vec2*>(ocvPinholeParams.principal_point.data());
                    static_assert(sizeof(ocvPinholeParams.focal_length) == sizeof(INRenderer::Vec2), "NuRec::NREND : typing size mismatch");
                    params.perspectiveParams.focalLength = *reinterpret_cast<const INRenderer::Vec2*>(ocvPinholeParams.focal_length.data());
                } else {
                    params.modelType = INRenderer::SensorProjectionModel::OpenCVPinholeModel;
                    params.ocvPinholeParams.nominalResolution =
                        INRenderer::Vec2{static_cast<float>(ocvPinholeParams.resolution[0]), static_cast<float>(ocvPinholeParams.resolution[1])};
                    static_assert(sizeof(ocvPinholeParams.principal_point) == sizeof(INRenderer::Vec2), "NuRec::NREND : typing size mismatch");
                    params.ocvPinholeParams.principalPoint = *reinterpret_cast<const INRenderer::Vec2*>(ocvPinholeParams.principal_point.data());
                    static_assert(sizeof(ocvPinholeParams.focal_length) == sizeof(INRenderer::Vec2), "NuRec::NREND : typing size mismatch");
                    params.ocvPinholeParams.focalLength = *reinterpret_cast<const INRenderer::Vec2*>(ocvPinholeParams.focal_length.data());
                    static_assert(sizeof(ocvPinholeParams.radial_coeffs) == sizeof(INRenderer::TVec<float, 6>), "NuRec::NREND : typing size mismatch");
                    params.ocvPinholeParams.radialCoeffs = *reinterpret_cast<const INRenderer::TVec<float, 6>*>(ocvPinholeParams.radial_coeffs.data());
                    static_assert(sizeof(ocvPinholeParams.tangential_coeffs) == sizeof(INRenderer::Vec2), "NuRec::NREND : typing size mismatch");
                    params.ocvPinholeParams.tangentialCoeffs = *reinterpret_cast<const INRenderer::Vec2*>(ocvPinholeParams.tangential_coeffs.data());
                    static_assert(sizeof(ocvPinholeParams.thin_prism_coeffs) == sizeof(INRenderer::Vec4), "NuRec::NREND : typing size mismatch");
                    params.ocvPinholeParams.thinPrismCoeffs = *reinterpret_cast<const INRenderer::Vec4*>(ocvPinholeParams.thin_prism_coeffs.data());
                    std::visit(ExternalDistortionVisitor(params), ocvPinholeParams.external_distortion_parameters);
                }
                return params;
            }

            INRenderer::SensorProjectionModel operator()(OpenCVFisheyeCameraModelParameters ocvFisheyeParams) const {
                INRenderer::SensorProjectionModel params;
                params.shutterType = static_cast<INRenderer::SensorProjectionModel::ShutterType>(ocvFisheyeParams.shutter_type);
                params.modelType   = INRenderer::SensorProjectionModel::OpenCVFisheyeModel;
                params.ocvFisheyeParams.nominalResolution =
                    INRenderer::Vec2{static_cast<float>(ocvFisheyeParams.resolution[0]), static_cast<float>(ocvFisheyeParams.resolution[1])};
                static_assert(sizeof(ocvFisheyeParams.principal_point) == sizeof(INRenderer::Vec2), "NuRec::NREND : typing size mismatch");
                params.ocvFisheyeParams.principalPoint = *reinterpret_cast<const INRenderer::Vec2*>(ocvFisheyeParams.principal_point.data());
                static_assert(sizeof(ocvFisheyeParams.focal_length) == sizeof(INRenderer::Vec2), "NuRec::NREND : typing size mismatch");
                params.ocvFisheyeParams.focalLength = *reinterpret_cast<const INRenderer::Vec2*>(ocvFisheyeParams.focal_length.data());
                static_assert(sizeof(ocvFisheyeParams.radial_coeffs) == sizeof(INRenderer::Vec4), "NuRec::NREND : typing size mismatch");
                params.ocvFisheyeParams.radialCoeffs = *reinterpret_cast<const INRenderer::Vec4*>(ocvFisheyeParams.radial_coeffs.data());
                params.ocvFisheyeParams.maxAngle     = ocvFisheyeParams.max_angle;
                std::visit(ExternalDistortionVisitor(params), ocvFisheyeParams.external_distortion_parameters);
                return params;
            }

            INRenderer::SensorProjectionModel operator()(FThetaCameraModelParameters fthetaParams) const {
                INRenderer::SensorProjectionModel params;
                params.shutterType = static_cast<INRenderer::SensorProjectionModel::ShutterType>(fthetaParams.shutter_type);
                params.modelType   = INRenderer::SensorProjectionModel::FThetaModel;
                params.fthetaParams.nominalResolution =
                    INRenderer::Vec2{static_cast<float>(fthetaParams.resolution[0]), static_cast<float>(fthetaParams.resolution[1])};
                static_assert(sizeof(fthetaParams.principal_point) == sizeof(INRenderer::Vec2), "NuRec::NREND : typing size mismatch");
                params.fthetaParams.principalPoint = *reinterpret_cast<const INRenderer::Vec2*>(fthetaParams.principal_point.data());
                params.fthetaParams.referencePoly  = static_cast<INRenderer::FThetaProjectionParameters::PolynomialType>(fthetaParams.reference_poly);
#if NREND_TEST_FTHETA_REGULAFALSI
                if (params.fthetaParams.referencePoly == INRenderer::FThetaProjectionParameters::PolynomialType::PIXELDIST_TO_ANGLE) {
                    params.fthetaParams.referencePoly = INRenderer::FThetaProjectionParameters::PolynomialType::PIXELDIST_TO_ANGLE_RF;
                }
#endif
                static_assert(sizeof(fthetaParams.pixeldist_to_angle_poly) == sizeof(INRenderer::TVec<float, INRenderer::FThetaProjectionParameters::PolynomialDegree>), "NuRec::NREND : typing size mismatch");
                params.fthetaParams.pixeldistToAnglePoly = *reinterpret_cast<const INRenderer::TVec<float, INRenderer::FThetaProjectionParameters::PolynomialDegree>*>(fthetaParams.pixeldist_to_angle_poly.data());
                static_assert(sizeof(fthetaParams.angle_to_pixeldist_poly) == sizeof(INRenderer::TVec<float, INRenderer::FThetaProjectionParameters::PolynomialDegree>), "NuRec::NREND : typing size mismatch");
                params.fthetaParams.angleToPixeldistPoly = *reinterpret_cast<const INRenderer::TVec<float, INRenderer::FThetaProjectionParameters::PolynomialDegree>*>(fthetaParams.angle_to_pixeldist_poly.data());
                params.fthetaParams.maxAngle             = fthetaParams.max_angle;
                params.fthetaParams.linear_cde           = *reinterpret_cast<const INRenderer::TVec<float, 3>*>(fthetaParams.linear_cde.data());
                std::visit(ExternalDistortionVisitor(params), fthetaParams.external_distortion_parameters);
                return params;
            }

            INRenderer::SensorProjectionModel operator()(RowOffsetStructuredSpinningLidarModelParameters rowOffsetLidarParams) const {
                INRenderer::SensorProjectionModel params;
                // Sanity check to ensure the lidar model is properly initialized with all the required information
                params.modelType = INRenderer::SensorProjectionModel::Unsupported;
                if (rowOffsetLidarParams._angles_to_columns_map == nullptr) {
                    logCallback(LoggerParameters::Error, "The LiDAR sensor model is missing rolling shutter information, likely due to the model not being initialized.", nullptr);
                    return params;
                }
                if (rowOffsetLidarParams._cdf_elevation == nullptr) {
                    logCallback(LoggerParameters::Error, "The LiDAR sensor model is missing the elevation CDF table, likely due to improper initialization with tiling information.", nullptr);
                    return params;
                }
                if (rowOffsetLidarParams._tiles_pack_info == nullptr || rowOffsetLidarParams._tiles_to_elements_map == nullptr) {
                    logCallback(LoggerParameters::Error, "The LiDAR sensor model is missing tiling information, likely due to the model not being properly initialized.", nullptr);
                    return params;
                }
                params.shutterType                                              = INRenderer::SensorProjectionModel::ShutterType::Undefined;
                params.modelType                                                = INRenderer::SensorProjectionModel::RowOffsetStructuredSpinningLidarModel;
                params.nreHesaiP128LidarParams.spin                             = (INRenderer::RowOffsetStructuredSpinningLidarProjectionParameters::SpinningDirection)rowOffsetLidarParams.spinning_direction;
                params.nreHesaiP128LidarParams.nRows                            = rowOffsetLidarParams.n_rows;
                params.nreHesaiP128LidarParams.nColumns                         = rowOffsetLidarParams.n_columns;
                params.nreHesaiP128LidarParams.fovStart                         = INRenderer::Vec2{rowOffsetLidarParams.fov_horiz_start_rad, rowOffsetLidarParams.fov_vert_start_rad};
                params.nreHesaiP128LidarParams.fovSpan                          = INRenderer::Vec2{rowOffsetLidarParams.fov_horiz_span_rad, rowOffsetLidarParams.fov_vert_span_rad};
                params.nreHesaiP128LidarParams.azimuthNBins                     = rowOffsetLidarParams.n_bins_azimuth;
                params.nreHesaiP128LidarParams.elevationNBins                   = rowOffsetLidarParams.n_bins_elevation;
                params.nreHesaiP128LidarParams.maxPtsPerTile                    = rowOffsetLidarParams.max_pts_per_tile;
                params.nreHesaiP128LidarParams.tilesPackInfo                    = reinterpret_cast<const INRenderer::IVec2*>(rowOffsetLidarParams._tiles_pack_info);
                params.nreHesaiP128LidarParams.tilesToElementsMap               = reinterpret_cast<const INRenderer::IVec2*>(rowOffsetLidarParams._tiles_to_elements_map);
                params.nreHesaiP128LidarParams.elevationCDFResolution           = rowOffsetLidarParams.elevation_cdf_resolution;
                params.nreHesaiP128LidarParams.azimuthCDFResolution             = rowOffsetLidarParams.azimuth_cdf_resolution;
                params.nreHesaiP128LidarParams.elevationCDFTable                = reinterpret_cast<const int*>(rowOffsetLidarParams._cdf_elevation);
                params.nreHesaiP128LidarParams.denseRayMaskCDFTable             = reinterpret_cast<const int*>(rowOffsetLidarParams._cdf_dense_ray_mask);
                params.nreHesaiP128LidarParams.angleToColumnMapResolutionFactor = rowOffsetLidarParams.angles_to_columns_map_resolution_factor;
                params.nreHesaiP128LidarParams.mapResolution                    = INRenderer::Vec2{rowOffsetLidarParams.map_resolution_horiz_rad, rowOffsetLidarParams.map_resolution_vert_rad};
                params.nreHesaiP128LidarParams.angleToColumnMap                 = reinterpret_cast<const int*>(rowOffsetLidarParams._angles_to_columns_map);
                return params;
            }
        };
        return std::visit(SensorModelVisitor{}, cameraModelVariant);
    }

    inline INRenderer::TSensorState toSensorState(INRenderer::TTimestamp startTs,
                                                  torch::Tensor sensorsStartPose,
                                                  INRenderer::TTimestamp endTs,
                                                  torch::Tensor sensorsEndPose) {
        const torch::Tensor sensorsStartPoseHost = sensorsStartPose.cpu();
        auto startInvPosePtr                     = reinterpret_cast<const INRenderer::TSensorPose*>(sensorsStartPoseHost.data_ptr<float>());
        const torch::Tensor sensorsEndPoseHost   = sensorsEndPose.cpu();
        auto endInvPosePtr                       = reinterpret_cast<const INRenderer::TSensorPose*>(sensorsEndPoseHost.data_ptr<float>());
        return INRenderer::TSensorState{
            startTs,
            startInvPosePtr ? INRenderer::TSensorState::poseInverse(*startInvPosePtr) : INRenderer::TSensorState::poseIdentity(),
            endTs,
            endInvPosePtr ? INRenderer::TSensorState::poseInverse(*endInvPosePtr) : INRenderer::TSensorState::poseIdentity()};
    }
};

// Single Python extension module for NRend
PYBIND11_MODULE(libnrend_cc, m) {
    m.def("set_rtc_cache_dir", [](const std::string& dir) { INRenderer::setRTCCacheDirectory(dir.c_str()); });

    m.def("set_rtc_include_dir", [](const std::string& dir, bool append, bool extra) { INRenderer::setRTCIncludeDirectory(dir.c_str(), append, extra); });

    m.def("device_memory_usage", []() -> size_t { 
        size_t usage;
        if (NREND_SUCCESS(INRenderer::devicesMemoryUsage(usage))) {
            return usage;
        }
        return 0u; });

    py::class_<NRendererWrapper>(m, "NRendererWrapper")
        .def(py::init<py::buffer, py::buffer, py::list, int, int, float, bool, bool, bool, bool>())
        .def("valid", &NRendererWrapper::valid)
        .def("render", &NRendererWrapper::render)
        .def("render_backward", &NRendererWrapper::renderBackward)
        .def("prepare_scene", &NRendererWrapper::prepareScene)
        .def("prepare_scene_backward", &NRendererWrapper::prepareSceneBackward)
        .def("update_model_parameters", &NRendererWrapper::updateModelParameters)
        .def("detach_model_parameters", &NRendererWrapper::detachModelParameters)
        .def("collect_profilings", &NRendererWrapper::collectProfilings);

    py::class_<NRendererForwardContextWrapper>(m, "NRendererForwardContextWrapper")
        .def(py::init())
        .def("reset", &NRendererForwardContextWrapper::reset);

    // ------------------------------------------------------------------------------------------------------------
    // Sensor models

    py::class_<INRenderer::SensorProjectionModel>(m, "NRendererSensorProjectionModel", py::module_local())
        .def(py::init<>());
}
