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

#include <nrend/modelParameters.h>
#include <nrend/renderer/renderParameters.h>
#include <nrend/renderingParameters.h>
#include <nrend/utils/status.h>

#include <cuda_runtime.h>

namespace nrend {
class NRenderer {
public:
    class ForwardContext {
    public:
        ForwardContext()          = default;
        virtual ~ForwardContext() = default;
    };

public:
    NRenderer(const Logger& logger)
        : m_logger(logger) {}
    virtual ~NRenderer() = default;

    /// load
    static NRenderer* loadFromMsgPackData(MsgPackData modelData,
                                          MsgPackData renderSettingsData,
                                          const RenderingParameters& renderParams = RenderingParameters{},
                                          Logger logger                           = Logger());

    /// get the rendering features layout
    virtual Status renderingFeaturesLayout(SensorType,
                                           RenderingFeaturesLayout& featuresLayout) const {
        // default features layout corresponds to the legacy renderer (ensuring backward compatibility)
        featuresLayout = RenderingFeaturesLayout{};
        return Status();
    }

    /// get the rendering scene data layout
    virtual Status renderingSceneDataLayout(uint32_t& sceneDataSize,
                                            RenderingSceneDataLayout& sceneDataLayout) const {
        // default scene data layout is empty
        sceneDataSize   = 0;
        sceneDataLayout = RenderingSceneDataLayout{};
        return Status();
    }

    /// march the scene according to the given camera and composite the result into the given cuda arrays
    inline Status render(const RenderParameters& params,
                         const tcnn::vec3* wordlRayOriginCudaPtr,
                         const tcnn::vec3* worldRayDirectionCudaPtr,
                         const nrend::TTimestamp* worldRayTimestampCudaPtr,
                         const tcnn::ivec2* sensorsIdsCudaPtr,
                         const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                         const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                         const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                         uint32_t* instanceIdCudaPtr,
                         float* worldHitDistanceCudaPtr,
                         tcnn::vec3* worldHitNormalCudaPtr,
                         tcnn::vec4* radianceDensityCudaPtr,
                         void* extendedFeaturesCudaPtr,
                         int cudaDeviceIndex,
                         cudaStream_t cudaStream) const {

        return renderForward(params,
                             wordlRayOriginCudaPtr,
                             worldRayDirectionCudaPtr,
                             worldRayTimestampCudaPtr,
                             sensorsIdsCudaPtr,
                             activeTrackInstancesIdsCudaPtr,
                             activeTrackInstancesPoseCudaPtr,
                             activeTrackInstancesEndPoseCudaPtr,
                             instanceIdCudaPtr,
                             worldHitDistanceCudaPtr,
                             worldHitNormalCudaPtr,
                             radianceDensityCudaPtr,
                             extendedFeaturesCudaPtr,
                             nullptr,
                             nullptr,
                             cudaDeviceIndex,
                             cudaStream);
    };

    virtual Status renderForward(const RenderParameters& params,
                                 const tcnn::vec3* wordlRayOriginCudaPtr,
                                 const tcnn::vec3* worldRayDirectionCudaPtr,
                                 const nrend::TTimestamp* worldRayTimestampCudaPtr,
                                 const tcnn::ivec2* sensorsIdsCudaPtr,
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
                                 cudaStream_t cudaStream) const = 0;

    virtual Status renderBackward(const RenderParameters& params,
                                  const tcnn::vec3* wordlRayOriginCudaPtr,
                                  const tcnn::vec3* worldRayDirectionCudaPtr,
                                  const nrend::TTimestamp* worldRayTimestampCudaPtr,
                                  const tcnn::ivec2* sensorsIdsCudaPtr,
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
                                  cudaStream_t cudaStream) const {
        RETURN_ERROR(m_logger, ErrorCode::NotImplemented, "NRenderer : renderBackward is not implemented.");
    }

    virtual Status sceneLayout(SensorType,
                               uint32_t& sceneSize,
                               uint32_t& sceneDensitySize,
                               uint32_t& featureSize,
                               uint32_t& extendedFeaturesSize,
                               uint32_t& sensorExtendedFeaturesSize,
                               bool& halfPrecision) const {
        RETURN_ERROR(m_logger, ErrorCode::NotImplemented, "NRenderer : sceneLayout is not implemented.");
    }

    virtual Status prepareSceneForward(const RenderParameters& params,
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
        RETURN_ERROR(m_logger, ErrorCode::NotImplemented, "NRenderer : prepareSceneForward is not implemented.");
    };

    virtual Status prepareSceneBackward(const RenderParameters& params,
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
                                        cudaStream_t cudaStream) const {
        RETURN_ERROR(m_logger, ErrorCode::NotImplemented, "NRenderer : prepareSceneBackward is not implemented.");
    };

    virtual Status updateModelParameters(const NamedParameterDefinitionsSpan& namedParametersDefinition,
                                         bool gradients,
                                         bool copy,
                                         int cudaDeviceIndex,
                                         cudaStream_t cudaStream) {
        RETURN_ERROR(m_logger, ErrorCode::NotImplemented, "NRenderer : updateModelParameters is not implemented.");
    }

    virtual Status detachModelParameters(bool gradients, bool copy, int cudaDeviceIndex, cudaStream_t cudaStream) {
        RETURN_ERROR(m_logger, ErrorCode::NotImplemented, "NRenderer : detachModelParameters is not implemented.");
    }

    virtual Status getModelVersion(int& versionMajor, int& versionMinor, int& versionPatch, const char*& modelName) const = 0;

protected:
    Logger m_logger;
};

} // namespace nrend
