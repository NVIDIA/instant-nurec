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

#include <nrend/kernels/cuda/common/rayPayloadBackward.cuh>
#include <nrend/kernels/cuda/sensors/sensorsProjection.cuh>
#include <nrend/renderer/gutRendererParameters.h>

struct GUTSplatRendererDummyParams {
    static constexpr bool ProjectRay                = true;
    static constexpr bool GlobalZOrder              = true;
    static constexpr uint32_t SceneDataDim          = 0;
    static constexpr int32_t SceneDataWeightsOffset = -1;
};

template <typename Particles,
          typename Params                   = GUTSplatRendererDummyParams,
          typename TilingParams             = nrend::GUTParameters::DefaultTiling,
          bool EnableFeatures               = true,
          bool EnableExtendedFeatures       = true,
          bool EnableCameraExtendedFeatures = true,
          bool EnableLidarExtendedFeatures  = false,
          bool EnableNormals                = false,
          bool EnableRayGradients           = false,
          bool Backward                     = false>
struct GUTSplatRenderer : Params {

    static constexpr int EnabledFeaturesDim = EnableFeatures ? Particles::FeaturesDim : 0;
    static constexpr int EnabledExtendedFeaturesDim =
        (EnableExtendedFeatures ? Particles::ExtendedFeaturesDim : 0) +
        (EnableCameraExtendedFeatures ? Particles::CameraExtendedFeaturesDim : 0) +
        (EnableLidarExtendedFeatures ? Particles::LidarExtendedFeaturesDim : 0);

    using TFeaturesVec            = tcnn::vec<EnabledFeaturesDim ? EnabledFeaturesDim : 1>;
    using TExtendedFeaturesVec    = tcnn::vec<EnabledExtendedFeaturesDim ? EnabledExtendedFeaturesDim : 1>;
    using TRayPayload             = RayPayload<EnabledFeaturesDim, EnabledExtendedFeaturesDim, EnableNormals>;
    using TRayPayloadBackward     = RayPayloadBackward<EnabledFeaturesDim, EnabledExtendedFeaturesDim, EnableNormals, EnableRayGradients>;
    using TPrecomputedFeaturesVec = tcnn::vec<EnabledFeaturesDim + EnabledExtendedFeaturesDim + 1>;

    struct ParticleData {
        uint32_t idx;
        tcnn::vec2 projectedPosition;
        tcnn::vec4 projectedConicOpacity;
        TPrecomputedFeaturesVec features;
    };

    template <typename TRayPayload>
    static inline __device__ void processHitParticle(
        TRayPayload& ray,
        const tcnn::vec2& rayProjectedPosition,
        const ParticleData& hitParticleData,
        float hitParticleAlpha,
        const Particles& particles,
        tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
        tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
        TPrecomputedFeaturesVec* __restrict__ particlesPrecomputedFeaturesGradPtr) {

        if constexpr (Backward) {
            float hitAlphaGrad = 0.f;

            if constexpr (EnabledFeaturesDim) {
                TFeaturesVec particleFeaturesGradientVec = TFeaturesVec::zero();
                particles.featuresIntegrateBwd(hitParticleAlpha,
                                               hitAlphaGrad,
                                               nrend::sliceVec<0, EnabledFeaturesDim>(hitParticleData.features),
                                               particleFeaturesGradientVec,
                                               nrend::sliceVec<0, EnabledFeaturesDim>(ray.features.vec),
                                               nrend::sliceVec<0, EnabledFeaturesDim>(ray.featuresGradient.vec));
#pragma unroll
                for (int i = 0; i < EnabledFeaturesDim; ++i) {
                    atomicAdd(&(particlesPrecomputedFeaturesGradPtr[hitParticleData.idx][i]), particleFeaturesGradientVec[i]);
                }
            }

            if constexpr (EnabledExtendedFeaturesDim) {
                particles.template extendedFeaturesIntegrateBwdToBuffer<EnableExtendedFeatures,
                                                                        EnableCameraExtendedFeatures,
                                                                        EnableLidarExtendedFeatures,
                                                                        false,
                                                                        false>(
                    hitParticleAlpha,
                    hitAlphaGrad,
                    hitParticleData.idx,
                    particles.template extendedFeaturesFromBuffer<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                        hitParticleData.idx, ray.direction),
                    nrend::sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.features.vec),
                    nrend::sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.featuresGradient.vec));
            }

            tcnn::vec2 projectedPositionGradient     = tcnn::vec2::zero();
            tcnn::vec4 projectedConicOpacityGradient = tcnn::vec4::zero();
            float hitTGradient                       = 0.f;

            particles.densityProcessHitBwdToConic(rayProjectedPosition,
                                                  hitParticleAlpha,
                                                  hitAlphaGrad,
                                                  ray.transmittanceBackward,
                                                  ray.transmittanceGradient,
                                                  hitParticleData.features[Particles::FeaturesDim], // hitDistance
                                                  hitTGradient,
                                                  ray.hitT,
                                                  ray.hitTGradient,
                                                  hitParticleData.projectedPosition,
                                                  projectedPositionGradient,
                                                  hitParticleData.projectedConicOpacity,
                                                  projectedConicOpacityGradient);

            atomicAdd(&(particlesProjectedPositionGradPtr[hitParticleData.idx].x), projectedPositionGradient.x);
            atomicAdd(&(particlesProjectedPositionGradPtr[hitParticleData.idx].y), projectedPositionGradient.y);

            atomicAdd(&(particlesProjectedConicOpacityGradPtr[hitParticleData.idx].x), projectedConicOpacityGradient.x);
            atomicAdd(&(particlesProjectedConicOpacityGradPtr[hitParticleData.idx].y), projectedConicOpacityGradient.y);
            atomicAdd(&(particlesProjectedConicOpacityGradPtr[hitParticleData.idx].z), projectedConicOpacityGradient.z);
            atomicAdd(&(particlesProjectedConicOpacityGradPtr[hitParticleData.idx].w), projectedConicOpacityGradient.w);

            atomicAdd(&particlesPrecomputedFeaturesGradPtr[hitParticleData.idx][EnabledFeaturesDim + EnabledExtendedFeaturesDim], hitTGradient);

            ray.transmittance *= (1.0f - hitParticleAlpha);

        } else {
            const float hitWeight =
                particles.densityIntegrateHit(hitParticleAlpha,
                                              ray.transmittance,
                                              hitParticleData.features[EnabledFeaturesDim + EnabledExtendedFeaturesDim], // hitDistance
                                              ray.hitT);

            if constexpr (EnabledExtendedFeaturesDim) {
                particles.template extendedFeaturesIntegrateFwd<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                    hitWeight,
                    particles.template extendedFeaturesFromBuffer<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(hitParticleData.idx, ray.direction),
                    nrend::sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.features.vec));
            }

            if constexpr (EnabledFeaturesDim) {
                particles.featureIntegrateFwd(hitWeight,
                                              nrend::sliceVec<0, EnabledFeaturesDim>(hitParticleData.features),
                                              nrend::sliceVec<0, EnabledFeaturesDim>(ray.features.vec));
            }
        }
        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            ray.kill();
        }
    }

    template <typename TRay>
    static inline __device__ void eval(const nrend::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                       float* __restrict__ sceneDataPtr,
                                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                       const TPrecomputedFeaturesVec* __restrict__ particlesPrecomputedFeaturesPtr,
                                       nrend::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr                = nullptr,
                                       tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr            = nullptr,
                                       TPrecomputedFeaturesVec* __restrict__ particlesPrecomputedFeaturesGradPtr = nullptr,
                                       nrend::MemoryHandles parametersGradient                                   = {}) {

        using namespace nrend;

        const uint32_t tileIdx                     = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t tileThreadIdx               = threadIdx.y * blockDim.x + threadIdx.x;
        const tcnn::uvec2 tileParticleRangeIndices = sortedTileRangeIndicesPtr[tileIdx];
        uint32_t tileNumParticlesToProcess         = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t tileNumBlocksToProcess      = tcnn::div_round_up(tileNumParticlesToProcess, TilingParams::BlockSize);

        tcnn::vec2 rayProjectedPosition;
        // project the ray to get the uv position of the thread : permits to take into account potential ray jittering
        if (Params::ProjectRay && (params.sensorModel.modelType == TSensorModel::PerspectiveModel)) {
            // NB : splatting does not support rolling shutter
            // FIXME : ray is first projected in object space, then transformed back to world space
            const tcnn::vec3 tStart = params.sensorState.startPose.slice<0, 3>();
            const tcnn::quat qStart = tcnn::quat{
                params.sensorState.startPose[6], // w
                params.sensorState.startPose[3], // x
                params.sensorState.startPose[4], // y
                params.sensorState.startPose[5]  // z
            };
            const tcnn::vec3 camRay = tcnn::to_mat3(qStart) * params.objectToWorldTransform * tcnn::vec4(ray.origin + 10.f * ray.direction, 1.f) + tStart;
            if (!nrend::projectPoint(params.sensorModel.perspectiveParams,
                                     params.frameResolution,
                                     camRay,
                                     1.0f, ///< arbitrary high tolerance (assuming the input ray is guaranteed to project on the sensor)
                                     rayProjectedPosition)) {
                ray.kill();
            }
        } else {
            rayProjectedPosition = tcnn::vec2(threadIdx.x + blockDim.x * blockIdx.x,
                                              threadIdx.y + blockDim.y * blockIdx.y) +
                                   0.5f;
        }

        __shared__ ParticleData prefetchedParticlesData[TilingParams::BlockSize];

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }

        if constexpr (EnabledFeaturesDim) {
            particles.initializeFeatures(parameters);
            if constexpr (Backward) {
                particles.initializeFeaturesGradient(parametersGradient);
            }
        }

        if constexpr (EnabledExtendedFeaturesDim) {
            particles.template initializeExtendedFeatures<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                parameters);
            if constexpr (Backward) {
                particles.template initializeExtendedFeaturesGradient<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                    parametersGradient);
            }
        }

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= TilingParams::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * TilingParams::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {
                        particleIdx,
                        particlesProjectedPositionPtr[particleIdx],
                        particlesProjectedConicOpacityPtr[particleIdx],
                        particlesPrecomputedFeaturesPtr[particleIdx]};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; ray.isAlive() && j < min(TilingParams::BlockSize, tileNumParticlesToProcess); j++) {

                const ParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }

                float particleAlpha;
                if ((particleData.features[Particles::FeaturesDim] > ray.tMinMax.x) &&
                    (particleData.features[Particles::FeaturesDim] < ray.tMinMax.y) &&
                    particles.densityConicHit(rayProjectedPosition,
                                              particleData.projectedPosition,
                                              particleData.projectedConicOpacity,
                                              particleAlpha)) {

                    if constexpr (Params::SceneDataWeightsOffset >= 0) {
                        if (sceneDataPtr) {
                            float accumulatedAlpha = particleAlpha;
                            reduceActiveWarpSumToBufferScalar(accumulatedAlpha,
                                                              &sceneDataPtr[particleData.idx * Params::SceneDataDim + Params::SceneDataWeightsOffset],
                                                              tileThreadIdx);
                        }
                    }

                    // TODO : block / warp atomic add
                    processHitParticle(ray,
                                       rayProjectedPosition,
                                       particleData,
                                       particleAlpha,
                                       particles,
                                       particlesProjectedPositionGradPtr,
                                       particlesProjectedConicOpacityGradPtr,
                                       particlesPrecomputedFeaturesGradPtr);
                }
            }
        }
    }
};
