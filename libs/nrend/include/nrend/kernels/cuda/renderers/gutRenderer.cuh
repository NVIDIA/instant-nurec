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

#include <nrend/kernels/cuda/common/rayPayload.cuh>
#include <nrend/renderer/gutRendererParameters.h>

__global__ void preProcessParticles(uint32_t numParticles,
                                    uint32_t numActiveTrackInstances,
                                    const tcnn::ivec2* __restrict__ activeTrackInstancesIdsCudaPtr,
                                    nrend::TTimestamp timestamp,
                                    const nrend::TTrackInstancePose* __restrict__ activeTrackInstancesStartPoseCudaPtr,
                                    const nrend::TTrackInstancePose* __restrict__ activeTrackInstancesEndPoseCudaPtr,
                                    const uint64_t* __restrict__ parameterMemoryHandles) {
    TGUTModel::preprocess(numParticles,
                          numActiveTrackInstances,
                          activeTrackInstancesIdsCudaPtr,
                          timestamp,
                          activeTrackInstancesStartPoseCudaPtr,
                          activeTrackInstancesEndPoseCudaPtr,
                          {parameterMemoryHandles});
}

__global__ void projectOnTiles(tcnn::ivec2 tileGrid,
                               uint32_t numParticles,
                               tcnn::vec2 resolution,
                               tcnn::vec2 offset,
                               tcnn::mat4x3 objectToWorldMatrix,
                               nrend::TSensorModel sensorModel,
                               tcnn::vec3 sensorPosition,
                               tcnn::mat4x3 sensorViewMatrix,
                               nrend::TSensorState sensorShutterState,
                               uint32_t* __restrict__ particlesTilesCountPtr,
                               tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                               tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                               tcnn::ivec2* __restrict__ particlesProjectedExtentPtr,
                               float* __restrict__ particlesGlobalDepthPtr,
                               float* __restrict__ particlesPrecomputedFeaturesPtr,
                               float* __restrict__ sceneDataPtr,
                               const uint64_t* __restrict__ parameterMemoryHandles) {

    const bool lidarSensor = sensorIsLidar(sensorModel);
    if (lidarSensor) {
        TGUTLidarProjector::eval(tileGrid,
                                 numParticles,
                                 resolution,
                                 offset,
                                 objectToWorldMatrix,
                                 sensorModel,
                                 sensorPosition,
                                 sensorViewMatrix,
                                 sensorShutterState,
                                 particlesTilesCountPtr,
                                 particlesProjectedPositionPtr,
                                 particlesProjectedConicOpacityPtr,
                                 particlesProjectedExtentPtr,
                                 particlesGlobalDepthPtr,
                                 reinterpret_cast<TGUTLidarProjector::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesPtr),
                                 sceneDataPtr,
                                 {parameterMemoryHandles});
    } else {
        TGUTCameraProjector::eval(tileGrid,
                                  numParticles,
                                  resolution,
                                  offset,
                                  objectToWorldMatrix,
                                  sensorModel,
                                  sensorPosition,
                                  sensorViewMatrix,
                                  sensorShutterState,
                                  particlesTilesCountPtr,
                                  particlesProjectedPositionPtr,
                                  particlesProjectedConicOpacityPtr,
                                  particlesProjectedExtentPtr,
                                  particlesGlobalDepthPtr,
                                  reinterpret_cast<TGUTCameraProjector::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesPtr),
                                  sceneDataPtr,
                                  {parameterMemoryHandles});
    }
}

__global__ void expandTileProjections(tcnn::ivec2 tileGrid,
                                      uint32_t numParticles,
                                      nrend::TSensorModel sensorModel,
                                      nrend::TSensorState sensorState,
                                      const uint32_t* __restrict__ particlesTilesOffsetPtr,
                                      const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                      const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                      const tcnn::ivec2* __restrict__ particlesProjectedExtentPtr,
                                      const float* __restrict__ particlesGlobalDepthPtr,
                                      const uint64_t* __restrict__ parameterMemoryHandles,
                                      uint64_t* __restrict__ unsortedTileDepthKeysPtr,
                                      uint32_t* __restrict__ unsortedTileParticleIdxPtr) {

    const bool lidarSensor = sensorIsLidar(sensorModel);
    if (lidarSensor) {
        TGUTLidarProjector::expand(tileGrid,
                                   numParticles,
                                   sensorModel,
                                   sensorState,
                                   particlesTilesOffsetPtr,
                                   particlesProjectedPositionPtr,
                                   particlesProjectedConicOpacityPtr,
                                   particlesProjectedExtentPtr,
                                   particlesGlobalDepthPtr,
                                   {parameterMemoryHandles},
                                   unsortedTileDepthKeysPtr,
                                   unsortedTileParticleIdxPtr);
    } else {
        TGUTCameraProjector::expand(tileGrid,
                                    numParticles,
                                    sensorModel,
                                    sensorState,
                                    particlesTilesOffsetPtr,
                                    particlesProjectedPositionPtr,
                                    particlesProjectedConicOpacityPtr,
                                    particlesProjectedExtentPtr,
                                    particlesGlobalDepthPtr,
                                    {parameterMemoryHandles},
                                    unsortedTileDepthKeysPtr,
                                    unsortedTileParticleIdxPtr);
    }
}

__global__ void renderLidar(nrend::RenderParameters params,
                            const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                            const uint32_t* __restrict__ sortedTileDataPtr,
                            const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                            const tcnn::vec3* __restrict__ worldRayDirectionPtr,
                            const nrend::TTimestamp* __restrict__ worldRayTimestampCudaPtr,
                            const tcnn::ivec2* __restrict__ sensorsIdsPtr,
                            uint32_t* __restrict__ instanceIdPtr,
                            float* __restrict__ worldHitDistancePtr,
                            tcnn::vec3* __restrict__ worldHitNormalPtr,
                            float* __restrict__ radianceDensityPtr,
                            void* __restrict__ extendedFeaturesPtr,
                            void* __restrict__ sceneDataPtr,
                            const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                            const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                            const float* __restrict__ particlesPrecomputedFeaturesPtr,
                            const uint64_t* __restrict__ parameterMemoryHandles) {

    auto ray = initializeRay<TGUTLidarRenderer::TRayPayload, true>(
        params, wordlRayOriginPtr, worldRayDirectionPtr, worldRayTimestampCudaPtr, instanceIdPtr, worldHitDistancePtr);

    TGUTLidarRenderer::eval(params,
                            ray,
                            sortedTileRangeIndicesPtr,
                            sortedTileDataPtr,
                            reinterpret_cast<float*>(sceneDataPtr),
                            particlesProjectedPositionPtr,
                            particlesProjectedConicOpacityPtr,
                            reinterpret_cast<const TGUTLidarRenderer::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesPtr),
                            {parameterMemoryHandles});

    TGUTModel::eval(params, ray, {parameterMemoryHandles}, sensorsIdsPtr);

    // NB : finalize ray is not differentiable (has to be no-op when used in a differentiable renderer)
    finalizeRay<SRGBModel, SRGBOutput, Differentiable>(
        ray,
        params,
        wordlRayOriginPtr,
        instanceIdPtr,
        worldHitDistancePtr,
        worldHitNormalPtr,
        reinterpret_cast<tcnn::vec<TGUTLidarRenderer::TRayPayload::BaseFeatDim + 1>*>(radianceDensityPtr),
        extendedFeaturesPtr);
}

__global__ void renderCamera(nrend::RenderParameters params,
                             const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                             const uint32_t* __restrict__ sortedTileDataPtr,
                             const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                             const tcnn::vec3* __restrict__ worldRayDirectionPtr,
                             const nrend::TTimestamp* __restrict__ worldRayTimestampCudaPtr,
                             const tcnn::ivec2* __restrict__ sensorsIdsPtr,
                             uint32_t* __restrict__ instanceIdPtr,
                             float* __restrict__ worldHitDistancePtr,
                             tcnn::vec3* __restrict__ worldHitNormalPtr,
                             float* __restrict__ radianceDensityPtr,
                             void* __restrict__ extendedFeaturesPtr,
                             void* __restrict__ sceneDataPtr,
                             const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                             const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                             const float* __restrict__ particlesPrecomputedFeaturesPtr,
                             const uint64_t* __restrict__ parameterMemoryHandles) {

    auto ray = initializeRay<TGUTCameraRenderer::TRayPayload, false>(
        params, wordlRayOriginPtr, worldRayDirectionPtr, worldRayTimestampCudaPtr, instanceIdPtr, worldHitDistancePtr);

    TGUTCameraRenderer::eval(params,
                             ray,
                             sortedTileRangeIndicesPtr,
                             sortedTileDataPtr,
                             reinterpret_cast<float*>(sceneDataPtr),
                             particlesProjectedPositionPtr,
                             particlesProjectedConicOpacityPtr,
                             reinterpret_cast<const TGUTCameraRenderer::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesPtr),
                             {parameterMemoryHandles});

    TGUTModel::eval(params, ray, {parameterMemoryHandles}, sensorsIdsPtr);

    // NB : finalize ray is not differentiable (has to be no-op when used in a differentiable renderer)
    finalizeRay<SRGBModel, SRGBOutput, Differentiable>(
        ray,
        params,
        wordlRayOriginPtr,
        instanceIdPtr,
        worldHitDistancePtr,
        worldHitNormalPtr,
        reinterpret_cast<tcnn::vec<TGUTCameraRenderer::TRayPayload::BaseFeatDim + 1>*>(radianceDensityPtr),
        extendedFeaturesPtr);
}

__global__
__launch_bounds__(nrend::GUTParameters::LidarTiling::BlockSize, 2) void renderBackwardLidar(nrend::RenderParameters params,
                                                                                            const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                                                                            const uint32_t* __restrict__ sortedTileDataPtr,
                                                                                            const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                                                                                            const tcnn::vec3* __restrict__ worldRayDirectionPtr,
                                                                                            const nrend::TTimestamp* __restrict__ worldRayTimestampCudaPtr,
                                                                                            const tcnn::ivec2* __restrict__ sensorsIdsPtr,
                                                                                            uint32_t* __restrict__ instanceIdPtr,
                                                                                            const float* __restrict__ worldHitDistancePtr,
                                                                                            const float* __restrict__ worldHitDistanceGradientPtr,
                                                                                            const tcnn::vec3* __restrict__ worldHitNormalPtr,
                                                                                            const tcnn::vec3* __restrict__ worldHitNormalGradientPtr,
                                                                                            const float* __restrict__ radianceDensityPtr,
                                                                                            const float* __restrict__ radianceDensityGradientPtr,
                                                                                            const void* __restrict__ extendedFeaturesPtr,
                                                                                            const void* __restrict__ extendedFeaturesGradientPtr,
                                                                                            tcnn::vec3* __restrict__ wordlRayOriginGradientPtr,
                                                                                            tcnn::vec3* __restrict__ worldRayDirectionGradientPtr,
                                                                                            const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                                                            const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                                                                            const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                                                                            const uint64_t* __restrict__ parameterMemoryHandles,
                                                                                            tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                                                                                            tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                                                                                            float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                                                                                            const uint64_t* __restrict__ parameterGradientMemoryHandles) {
    auto ray = initializeBackwardRay<TGUTLidarRenderer::TRayPayloadBackward, true>(
        params,
        wordlRayOriginPtr,
        worldRayDirectionPtr,
        worldRayTimestampCudaPtr,
        instanceIdPtr,
        worldHitDistancePtr,
        worldHitDistanceGradientPtr,
        worldHitNormalPtr,
        worldHitNormalGradientPtr,
        reinterpret_cast<const tcnn::vec<TGUTLidarRenderer::TRayPayloadBackward::BaseFeatDim + 1>*>(radianceDensityPtr),
        reinterpret_cast<const tcnn::vec<TGUTLidarRenderer::TRayPayloadBackward::BaseFeatDim + 1>*>(radianceDensityGradientPtr),
        extendedFeaturesPtr,
        extendedFeaturesGradientPtr);

    TGUTModel::evalBackward(params, ray, {parameterMemoryHandles}, {parameterGradientMemoryHandles}, sensorsIdsPtr);

    TGUTLidarBackwardRenderer::eval(params,
                                    ray,
                                    sortedTileRangeIndicesPtr,
                                    sortedTileDataPtr,
                                    nullptr, // sceneDataPtr - not used in backward pass
                                    particlesProjectedPositionPtr,
                                    particlesProjectedConicOpacityPtr,
                                    reinterpret_cast<const TGUTLidarRenderer::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesPtr),
                                    {parameterMemoryHandles},
                                    particlesProjectedPositionGradPtr,
                                    particlesProjectedConicOpacityGradPtr,
                                    reinterpret_cast<TGUTLidarRenderer::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesGradPtr),
                                    {parameterGradientMemoryHandles});

    finalizeBackwardRay<TGUTLidarRenderer::TRayPayloadBackward>(ray, params, wordlRayOriginGradientPtr, worldRayDirectionGradientPtr);
}

__global__ void renderBackwardCamera(nrend::RenderParameters params,
                                     const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                     const uint32_t* __restrict__ sortedTileDataPtr,
                                     const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                                     const tcnn::vec3* __restrict__ worldRayDirectionPtr,
                                     const nrend::TTimestamp* __restrict__ worldRayTimestampCudaPtr,
                                     const tcnn::ivec2* __restrict__ sensorsIdsPtr,
                                     uint32_t* __restrict__ instanceIdPtr,
                                     const float* __restrict__ worldHitDistancePtr,
                                     const float* __restrict__ worldHitDistanceGradientPtr,
                                     const tcnn::vec3* __restrict__ worldHitNormalPtr,
                                     const tcnn::vec3* __restrict__ worldHitNormalGradientPtr,
                                     const float* __restrict__ radianceDensityPtr,
                                     const float* __restrict__ radianceDensityGradientPtr,
                                     const void* __restrict__ extendedFeaturesPtr,
                                     const void* __restrict__ extendedFeaturesGradientPtr,
                                     tcnn::vec3* __restrict__ wordlRayOriginGradientPtr,
                                     tcnn::vec3* __restrict__ worldRayDirectionGradientPtr,
                                     const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                     const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                     const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                     const uint64_t* __restrict__ parameterMemoryHandles,
                                     tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                                     tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                                     float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                                     const uint64_t* __restrict__ parameterGradientMemoryHandles) {
    auto ray = initializeBackwardRay<TGUTCameraRenderer::TRayPayloadBackward, false>(
        params,
        wordlRayOriginPtr,
        worldRayDirectionPtr,
        worldRayTimestampCudaPtr,
        instanceIdPtr,
        worldHitDistancePtr,
        worldHitDistanceGradientPtr,
        worldHitNormalPtr,
        worldHitNormalGradientPtr,
        reinterpret_cast<const tcnn::vec<TGUTCameraRenderer::TRayPayloadBackward::BaseFeatDim + 1>*>(radianceDensityPtr),
        reinterpret_cast<const tcnn::vec<TGUTCameraRenderer::TRayPayloadBackward::BaseFeatDim + 1>*>(radianceDensityGradientPtr),
        extendedFeaturesPtr,
        extendedFeaturesGradientPtr);

    TGUTModel::evalBackward(params, ray, {parameterMemoryHandles}, {parameterGradientMemoryHandles}, sensorsIdsPtr);

    TGUTCameraBackwardRenderer::eval(params,
                                     ray,
                                     sortedTileRangeIndicesPtr,
                                     sortedTileDataPtr,
                                     nullptr, // sceneDataPtr - not used in backward pass
                                     particlesProjectedPositionPtr,
                                     particlesProjectedConicOpacityPtr,
                                     reinterpret_cast<const TGUTCameraRenderer::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesPtr),
                                     {parameterMemoryHandles},
                                     particlesProjectedPositionGradPtr,
                                     particlesProjectedConicOpacityGradPtr,
                                     reinterpret_cast<TGUTCameraRenderer::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesGradPtr),
                                     {parameterGradientMemoryHandles});

    finalizeBackwardRay<TGUTCameraRenderer::TRayPayloadBackward>(ray, params, wordlRayOriginGradientPtr, worldRayDirectionGradientPtr);
}

__global__ void projectBackward(uint32_t numParticles,
                                tcnn::vec2 resolution,
                                nrend::TSensorModel sensorModel,
                                tcnn::vec3 sensorPosition,
                                tcnn::mat4x3 sensorViewMatrix,
                                const uint32_t* __restrict__ particlesTilesCountPtr,
                                const uint64_t* __restrict__ parameterMemoryHandles,
                                const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                                const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                                const uint64_t* __restrict__ parameterGradientMemoryHandles) {

    const bool lidarSensor = sensorIsLidar(sensorModel);
    if (lidarSensor) {
        TGUTLidarProjector::evalBackward(numParticles,
                                         resolution,
                                         sensorModel,
                                         sensorPosition,
                                         sensorViewMatrix,
                                         particlesTilesCountPtr,
                                         {parameterMemoryHandles},
                                         particlesProjectedPositionGradPtr,
                                         particlesProjectedConicOpacityGradPtr,
                                         reinterpret_cast<const TGUTLidarProjector::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesPtr),
                                         reinterpret_cast<const TGUTLidarProjector::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesGradPtr),
                                         {parameterGradientMemoryHandles});
    } else {
        TGUTCameraProjector::evalBackward(numParticles,
                                          resolution,
                                          sensorModel,
                                          sensorPosition,
                                          sensorViewMatrix,
                                          particlesTilesCountPtr,
                                          {parameterMemoryHandles},
                                          particlesProjectedPositionGradPtr,
                                          particlesProjectedConicOpacityGradPtr,
                                          reinterpret_cast<const TGUTCameraProjector::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesPtr),
                                          reinterpret_cast<const TGUTCameraProjector::TPrecomputedFeaturesVec*>(particlesPrecomputedFeaturesGradPtr),
                                          {parameterGradientMemoryHandles});
    }
}

__global__ void prepareScene(tcnn::ivec2 tileGrid,
                             uint32_t numParticles,
                             tcnn::vec2 resolution,
                             tcnn::vec2 offset,
                             tcnn::mat4x3 objectToWorldMatrix,
                             nrend::TSensorModel sensorModel,
                             tcnn::vec3 sensorPosition,
                             tcnn::mat4x3 sensorViewMatrix,
                             nrend::TSensorState sensorShutterState,
                             uint32_t* __restrict__ particlesCountIdxPtr,
                             float* __restrict__ particlesDensityPtr,
                             float* __restrict__ particlesFeaturesPtr,
                             float* __restrict__ particlesExtendedFeaturesPtr,
                             float* __restrict__ particlesSensorExtendedFeaturesPtr,
                             float* __restrict__ sceneDataPtr,
                             const uint64_t* __restrict__ parameterMemoryHandles) {

    const bool lidarSensor = sensorIsLidar(sensorModel);
    if (lidarSensor) {
        TGUTLidarProjector::prepare(tileGrid,
                                    numParticles,
                                    resolution,
                                    offset,
                                    objectToWorldMatrix,
                                    sensorModel,
                                    sensorPosition,
                                    sensorViewMatrix,
                                    sensorShutterState,
                                    particlesCountIdxPtr,
                                    particlesDensityPtr,
                                    particlesFeaturesPtr,
                                    particlesExtendedFeaturesPtr,
                                    particlesSensorExtendedFeaturesPtr,
                                    sceneDataPtr,
                                    {parameterMemoryHandles});
    } else {
        TGUTCameraProjector::prepare(tileGrid,
                                     numParticles,
                                     resolution,
                                     offset,
                                     objectToWorldMatrix,
                                     sensorModel,
                                     sensorPosition,
                                     sensorViewMatrix,
                                     sensorShutterState,
                                     particlesCountIdxPtr,
                                     particlesDensityPtr,
                                     particlesFeaturesPtr,
                                     particlesExtendedFeaturesPtr,
                                     particlesSensorExtendedFeaturesPtr,
                                     sceneDataPtr,
                                     {parameterMemoryHandles});
    }
}

__global__ void prepareSceneBackward(uint32_t numParticles,
                                     tcnn::vec2 resolution,
                                     nrend::TSensorModel sensorModel,
                                     tcnn::vec3 sensorPosition,
                                     tcnn::mat4x3 sensorViewMatrix,
                                     uint32_t* __restrict__ particlesCountIdxPtr,
                                     const void* __restrict__ particlesFeaturesPtr,
                                     const void* __restrict__ particlesExtendedFeaturesPtr,
                                     const void* __restrict__ particlesSensorExtendedFeaturesPtr,
                                     const uint64_t* __restrict__ parameterMemoryHandles,
                                     const void* __restrict__ particlesDensityGradPtr,
                                     const void* __restrict__ particlesFeaturesGradPtr,
                                     const void* __restrict__ particlesExtendedFeaturesGradPtr,
                                     const void* __restrict__ particlesSensorExtendedFeaturesGradPtr,
                                     const uint64_t* __restrict__ parameterGradientMemoryHandles) {

    const bool lidarSensor = sensorIsLidar(sensorModel);
    if (lidarSensor) {
        TGUTLidarProjector::prepareBackward(numParticles,
                                            resolution,
                                            sensorModel,
                                            sensorPosition,
                                            sensorViewMatrix,
                                            particlesCountIdxPtr,
                                            particlesFeaturesPtr,
                                            particlesExtendedFeaturesPtr,
                                            particlesSensorExtendedFeaturesPtr,
                                            {parameterMemoryHandles},
                                            particlesDensityGradPtr,
                                            particlesFeaturesGradPtr,
                                            particlesExtendedFeaturesGradPtr,
                                            particlesSensorExtendedFeaturesGradPtr,
                                            {parameterGradientMemoryHandles});
    } else {
        TGUTCameraProjector::prepareBackward(numParticles,
                                             resolution,
                                             sensorModel,
                                             sensorPosition,
                                             sensorViewMatrix,
                                             particlesCountIdxPtr,
                                             particlesFeaturesPtr,
                                             particlesExtendedFeaturesPtr,
                                             particlesSensorExtendedFeaturesPtr,
                                             {parameterMemoryHandles},
                                             particlesDensityGradPtr,
                                             particlesFeaturesGradPtr,
                                             particlesExtendedFeaturesGradPtr,
                                             particlesSensorExtendedFeaturesGradPtr,
                                             {parameterGradientMemoryHandles});
    }
}
