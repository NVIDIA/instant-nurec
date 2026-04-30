// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/iNRenderer.h>
#include <nrend/kernelResources/rtcKernelConfig.h>
#include <nrend/renderer/renderer.h>
#include <nrend/utils/cuda/cudaMemoryAllocator.h>

// FIXME : this is required for TCNN within NRE
#ifndef TCNN_CMRC
#include <tiny-cuda-nn/rtc_kernel.h>
#endif

#define NREND_CONSISTENT_DEFINITION(var) \
    static_assert(sizeof(nrend::INRenderer::var) == sizeof(nrend::var), "INRenderer::" #var " must be castable to " #var);

NREND_CONSISTENT_DEFINITION(OrthographicProjectionParameters)
NREND_CONSISTENT_DEFINITION(PerspectiveProjectionParameters)
NREND_CONSISTENT_DEFINITION(OpenCVPinholeProjectionParameters)
NREND_CONSISTENT_DEFINITION(OpenCVFisheyeProjectionParameters)
NREND_CONSISTENT_DEFINITION(FThetaProjectionParameters)
NREND_CONSISTENT_DEFINITION(RowOffsetStructuredSpinningLidarProjectionParameters)
NREND_CONSISTENT_DEFINITION(GeneralizedProjectionParameters)
NREND_CONSISTENT_DEFINITION(BivariateWindshieldDistortionParameters)
NREND_CONSISTENT_DEFINITION(SensorProjectionModel)

nrend::INRenderer::TSensorPose nrend::INRenderer::TSensorState::poseInverse(const TSensorPose& pose) {
    static_assert(sizeof(TSensorPose) == sizeof(nrend::TSensorPose));
    const nrend::TSensorPose iPose = nrend::poseInverse({pose.elems[0], pose.elems[1], pose.elems[2], pose.elems[3], pose.elems[4], pose.elems[5], pose.elems[6]});
    return {iPose[0], iPose[1], iPose[2], iPose[3], iPose[4], iPose[5], iPose[6]};
}

nrend::ErrorCode nrend::INRenderer::create(const MsgPackData& modelSpecificationData,
                                           const MsgPackData& renderSpecificationData,
                                           const RenderingParameters& renderParameters,
                                           const LoggerParameters& loggerParameters,
                                           RendererHandle& handle) {

    auto rendererPtr = NRenderer::loadFromMsgPackData(modelSpecificationData,
                                                      renderSpecificationData,
                                                      renderParameters,
                                                      Logger(loggerParameters));
    handle           = rendererPtr ? reinterpret_cast<RendererHandle>(rendererPtr) : InvalidRendererHandle;

    return rendererPtr ? ErrorCode::None : ErrorCode::BadInput;
}

void nrend::INRenderer::destroy(RendererHandle handle) {
    if (handle != InvalidRendererHandle) {
        delete reinterpret_cast<NRenderer*>(handle);
    }
}

nrend::ErrorCode nrend::INRenderer::render(RendererHandle handle,
                                           const RenderParameters& params,
                                           const Vec3* worldRayOriginCudaPtr,
                                           const Vec3* worldRayDirectionCudaPtr,
                                           const TTimestamp* worldRayTimestampCudaPtr,
                                           const IVec2* sensorsIdsCudaPtr,
                                           const IVec2* activeTrackInstancesIdsCudaPtr,
                                           const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                           const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                           uint32_t* instanceIdCudaPtr,
                                           float* worldHitDistanceCudaPtr,
                                           Vec3* worldHitNormalCudaPtr,
                                           Vec4* radianceDensityCudaPtr,
                                           void* extendedFeaturesCudaPtr,
                                           void* sceneDataCudaPtr,
                                           int deviceIndex,
                                           DeviceQueueHandle deviceQueue,
                                           RenderingContextHandle* context) {

    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }

    CudaCheckDeviceGuard cudaDeviceGuard(deviceIndex);
    if (!cudaDeviceGuard.check()) {
        return ErrorCode::Runtime;
    }

    static_assert(sizeof(RenderParameters) == sizeof(nrend::RenderParameters),
                  "INRenderer::Parameters must be castable to RenderParameters");

    return reinterpret_cast<const NRenderer*>(handle)->renderForward(reinterpret_cast<const nrend::RenderParameters&>(params),
                                                                     reinterpret_cast<const tcnn::vec3*>(worldRayOriginCudaPtr),
                                                                     reinterpret_cast<const tcnn::vec3*>(worldRayDirectionCudaPtr),
                                                                     worldRayTimestampCudaPtr,
                                                                     reinterpret_cast<const tcnn::ivec2*>(sensorsIdsCudaPtr),
                                                                     reinterpret_cast<const tcnn::ivec2*>(activeTrackInstancesIdsCudaPtr),
                                                                     reinterpret_cast<const nrend::TTrackInstancePose*>(activeTrackInstancesPoseCudaPtr),
                                                                     reinterpret_cast<const nrend::TTrackInstancePose*>(activeTrackInstancesEndPoseCudaPtr),
                                                                     instanceIdCudaPtr,
                                                                     worldHitDistanceCudaPtr,
                                                                     reinterpret_cast<tcnn::vec3*>(worldHitNormalCudaPtr),
                                                                     reinterpret_cast<tcnn::vec4*>(radianceDensityCudaPtr),
                                                                     reinterpret_cast<void*>(extendedFeaturesCudaPtr),
                                                                     reinterpret_cast<void*>(sceneDataCudaPtr),
                                                                     reinterpret_cast<NRenderer::ForwardContext**>(context),
                                                                     deviceIndex,
                                                                     reinterpret_cast<cudaStream_t>(deviceQueue));
}

nrend::ErrorCode nrend::INRenderer::renderBackward(RendererHandle handle,
                                                   const RenderParameters& params,
                                                   const Vec3* worldRayOriginCudaPtr,
                                                   const Vec3* worldRayDirectionCudaPtr,
                                                   const TTimestamp* worldRayTimestampCudaPtr,
                                                   const IVec2* sensorsIdsCudaPtr,
                                                   const IVec2* activeTrackInstancesIdsCudaPtr,
                                                   const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                                   const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                                   uint32_t* instanceIdCudaPtr,
                                                   float* worldHitDistanceCudaPtr,
                                                   const float* worldHitDistanceGradientCudaPtr,
                                                   const Vec3* worldHitNormalCudaPtr,
                                                   const Vec3* worldHitNormalGradientCudaPtr,
                                                   const Vec4* radianceDensityCudaPtr,
                                                   const Vec4* radianceDensityGradientCudaPtr,
                                                   const void* extendedFeaturesCudaPtr,
                                                   const void* extendedFeaturesGradientCudaPtr,
                                                   Vec3* wordlRayOriginGradientCudaPtr,
                                                   Vec3* worldRayDirectionGradientCudaPtr,
                                                   int deviceIndex,
                                                   DeviceQueueHandle deviceQueue,
                                                   RenderingContextHandle context) {

    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }

    CudaCheckDeviceGuard cudaDeviceGuard(deviceIndex);
    if (!cudaDeviceGuard.check()) {
        return ErrorCode::Runtime;
    }

    return reinterpret_cast<const NRenderer*>(handle)->renderBackward(reinterpret_cast<const nrend::RenderParameters&>(params),
                                                                      reinterpret_cast<const tcnn::vec3*>(worldRayOriginCudaPtr),
                                                                      reinterpret_cast<const tcnn::vec3*>(worldRayDirectionCudaPtr),
                                                                      reinterpret_cast<const nrend::TTimestamp*>(worldRayTimestampCudaPtr),
                                                                      reinterpret_cast<const tcnn::ivec2*>(sensorsIdsCudaPtr),
                                                                      reinterpret_cast<const tcnn::ivec2*>(activeTrackInstancesIdsCudaPtr),
                                                                      reinterpret_cast<const nrend::TTrackInstancePose*>(activeTrackInstancesPoseCudaPtr),
                                                                      reinterpret_cast<const nrend::TTrackInstancePose*>(activeTrackInstancesEndPoseCudaPtr),
                                                                      instanceIdCudaPtr,
                                                                      worldHitDistanceCudaPtr,
                                                                      worldHitDistanceGradientCudaPtr,
                                                                      reinterpret_cast<const tcnn::vec3*>(worldHitNormalCudaPtr),
                                                                      reinterpret_cast<const tcnn::vec3*>(worldHitNormalGradientCudaPtr),
                                                                      reinterpret_cast<const tcnn::vec4*>(radianceDensityCudaPtr),
                                                                      reinterpret_cast<const tcnn::vec4*>(radianceDensityGradientCudaPtr),
                                                                      reinterpret_cast<const void*>(extendedFeaturesCudaPtr),
                                                                      reinterpret_cast<const void*>(extendedFeaturesGradientCudaPtr),
                                                                      reinterpret_cast<tcnn::vec3*>(wordlRayOriginGradientCudaPtr),
                                                                      reinterpret_cast<tcnn::vec3*>(worldRayDirectionGradientCudaPtr),
                                                                      reinterpret_cast<NRenderer::ForwardContext*>(context),
                                                                      deviceIndex,
                                                                      reinterpret_cast<cudaStream_t>(deviceQueue));
}

nrend::ErrorCode nrend::INRenderer::sceneLayout(RendererHandle handle,
                                                SensorType sensorType,
                                                uint32_t& sceneSize,
                                                uint32_t& sceneDensitySize,
                                                uint32_t& featureSize,
                                                uint32_t& extendedFeaturesSize,
                                                uint32_t& sensorExtendedFeaturesSize,
                                                bool& halfPrecision) {
    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }
    return reinterpret_cast<const NRenderer*>(handle)->sceneLayout(static_cast<nrend::SensorType>(sensorType),
                                                                   sceneSize,
                                                                   sceneDensitySize,
                                                                   featureSize,
                                                                   extendedFeaturesSize,
                                                                   sensorExtendedFeaturesSize,
                                                                   halfPrecision);
}

nrend::ErrorCode nrend::INRenderer::prepareScene(RendererHandle handle,
                                                 const RenderParameters& params,
                                                 const IVec2* activeTrackInstancesIdsCudaPtr,
                                                 const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                                 const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                                 void* sceneDensityCudaPtr,
                                                 void* sceneFeaturesCudaPtr,
                                                 void* sceneExtendedFeaturesCudaPtr,
                                                 void* sceneSensorExtendedFeaturesCudaPtr,
                                                 void* sceneDataCudaPtr,
                                                 uint32_t& sceneSize,
                                                 int deviceIndex,
                                                 DeviceQueueHandle deviceQueue,
                                                 RenderingContextHandle* forwardContext) {

    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }

    CudaCheckDeviceGuard cudaDeviceGuard(deviceIndex);
    if (!cudaDeviceGuard.check()) {
        return ErrorCode::Runtime;
    }

    return reinterpret_cast<const NRenderer*>(handle)->prepareSceneForward(reinterpret_cast<const nrend::RenderParameters&>(params),
                                                                           reinterpret_cast<const tcnn::ivec2*>(activeTrackInstancesIdsCudaPtr),
                                                                           reinterpret_cast<const nrend::TTrackInstancePose*>(activeTrackInstancesPoseCudaPtr),
                                                                           reinterpret_cast<const nrend::TTrackInstancePose*>(activeTrackInstancesEndPoseCudaPtr),
                                                                           sceneDensityCudaPtr,
                                                                           sceneFeaturesCudaPtr,
                                                                           sceneExtendedFeaturesCudaPtr,
                                                                           sceneSensorExtendedFeaturesCudaPtr,
                                                                           sceneDataCudaPtr,
                                                                           sceneSize,
                                                                           reinterpret_cast<NRenderer::ForwardContext**>(forwardContext),
                                                                           deviceIndex,
                                                                           reinterpret_cast<cudaStream_t>(deviceQueue));
}

nrend::ErrorCode nrend::INRenderer::prepareSceneBackward(RendererHandle handle,
                                                         const RenderParameters& params,
                                                         const IVec2* activeTrackInstancesIdsCudaPtr,
                                                         const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                                                         const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                                         const void* sceneFeaturesCudaPtr,
                                                         const void* sceneExtendedFeaturesCudaPtr,
                                                         const void* sceneSensorExtendedFeaturesCudaPtr,
                                                         const void* sceneDensityGradientCudaPtr,
                                                         const void* sceneFeaturesGradientCudaPtr,
                                                         const void* sceneExtendedFeaturesGradientCudaPtr,
                                                         const void* sceneSensorExtendedFeaturesGradientCudaPtr,
                                                         int deviceIndex,
                                                         DeviceQueueHandle deviceQueue,
                                                         RenderingContextHandle forwardContext) {
    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }

    CudaCheckDeviceGuard cudaDeviceGuard(deviceIndex);
    if (!cudaDeviceGuard.check()) {
        return ErrorCode::Runtime;
    }

    return reinterpret_cast<const NRenderer*>(handle)->prepareSceneBackward(reinterpret_cast<const nrend::RenderParameters&>(params),
                                                                            reinterpret_cast<const tcnn::ivec2*>(activeTrackInstancesIdsCudaPtr),
                                                                            reinterpret_cast<const nrend::TTrackInstancePose*>(activeTrackInstancesPoseCudaPtr),
                                                                            reinterpret_cast<const nrend::TTrackInstancePose*>(activeTrackInstancesEndPoseCudaPtr),
                                                                            sceneFeaturesCudaPtr,
                                                                            sceneExtendedFeaturesCudaPtr,
                                                                            sceneSensorExtendedFeaturesCudaPtr,
                                                                            sceneDensityGradientCudaPtr,
                                                                            sceneFeaturesGradientCudaPtr,
                                                                            sceneExtendedFeaturesGradientCudaPtr,
                                                                            sceneSensorExtendedFeaturesGradientCudaPtr,
                                                                            reinterpret_cast<NRenderer::ForwardContext*>(forwardContext),
                                                                            deviceIndex,
                                                                            reinterpret_cast<cudaStream_t>(deviceQueue));
}

void nrend::INRenderer::destroyRenderingContext(RenderingContextHandle context) {
    if (context != InvalidRenderingContextHandle) {
        delete reinterpret_cast<NRenderer::ForwardContext*>(context);
    }
}

nrend::ErrorCode nrend::INRenderer::getModelVersion(RendererHandle handle,
                                                    int& versionMajor,
                                                    int& versionMinor,
                                                    int& versionPatch,
                                                    const char*& modelName) {

    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }
    return reinterpret_cast<const NRenderer*>(handle)->getModelVersion(versionMajor, versionMinor, versionPatch, modelName);
}

nrend::ErrorCode nrend::INRenderer::updateModelParameters(RendererHandle handle,
                                                          const NamedParameterDefinitionsSpan& namedParametersDefinitions,
                                                          bool gradients,
                                                          bool copy,
                                                          int deviceIndex,
                                                          DeviceQueueHandle deviceQueue) {

    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }

    CudaCheckDeviceGuard cudaDeviceGuard(deviceIndex);
    if (!cudaDeviceGuard.check()) {
        return ErrorCode::Runtime;
    }

    return reinterpret_cast<NRenderer*>(handle)->updateModelParameters(namedParametersDefinitions,
                                                                       gradients,
                                                                       copy,
                                                                       deviceIndex,
                                                                       reinterpret_cast<cudaStream_t>(deviceQueue));
}

nrend::ErrorCode nrend::INRenderer::detachModelParameters(RendererHandle handle,
                                                          bool gradients,
                                                          bool copy,
                                                          int deviceIndex,
                                                          DeviceQueueHandle deviceQueue) {

    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }

    CudaCheckDeviceGuard cudaDeviceGuard(deviceIndex);
    if (!cudaDeviceGuard.check()) {
        return ErrorCode::Runtime;
    }

    return reinterpret_cast<NRenderer*>(handle)->detachModelParameters(gradients,
                                                                       copy,
                                                                       deviceIndex,
                                                                       reinterpret_cast<cudaStream_t>(deviceQueue));
}

nrend::ErrorCode nrend::INRenderer::setRTCCacheDirectory(const char* directory) {
    RtcKernelConfig::setCacheDirectory(directory);
    return ErrorCode::None;
}

nrend::ErrorCode nrend::INRenderer::setRTCIncludeDirectory(const char* directory,
                                                           bool append,
                                                           bool extra) {
    RtcKernelConfig::setIncludeDirectory(directory, append, extra);
// FIXME : this is required for TCNN within NRE
#ifndef TCNN_CMRC
    if (!append && !extra) {
        tcnn::rtc_set_include_dir(directory);
    }
#endif
    return ErrorCode::None;
}

nrend::ErrorCode nrend::INRenderer::setDeviceAllocator(const DeviceMemoryAllocator& allocator) {
    CudaMemoryAllocator::get().setAllocator(allocator);
    return ErrorCode::None;
}

nrend::ErrorCode nrend::INRenderer::devicesMemoryUsage(size_t& usage) {
    usage = CudaMemoryAllocator::get().currentlyAllocated();
    return ErrorCode::None;
}

nrend::ErrorCode nrend::INRenderer::renderingFeaturesLayout(RendererHandle handle,
                                                            SensorType sensorType,
                                                            RenderingFeaturesLayout& featuresLayout) {
    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }
    return reinterpret_cast<const NRenderer*>(handle)->renderingFeaturesLayout(
        static_cast<nrend::SensorType>(sensorType),
        featuresLayout);
}

nrend::ErrorCode nrend::INRenderer::renderingSceneDataLayout(RendererHandle handle,
                                                             uint32_t& sceneDataSize,
                                                             RenderingSceneDataLayout& sceneDataLayout) {
    if (handle == InvalidRendererHandle) {
        return ErrorCode::BadInput;
    }
    return reinterpret_cast<const NRenderer*>(handle)->renderingSceneDataLayout(sceneDataSize, sceneDataLayout);
}
