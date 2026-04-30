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
#include <nrend/renderer/gutRendererParameters.h>
#include <nrend/utils/nreVec.h>

struct GUTKBufferRendererDummyParams {
    static constexpr bool PerRayParticleFeatures    = false;
    static constexpr float MinProjectedRayRadius    = 1.0f;
    static constexpr int KHitBufferSize             = 0;
    static constexpr bool EnableWarpAtomicOptim     = false;
    static constexpr uint32_t SceneDataDim          = 0;
    static constexpr int32_t SceneDataWeightsOffset = -1;
};

template <bool HasNormals>
struct GUTKBufferHitParticle {
    static constexpr float InvalidHitT = -1.0f;
    int idx                            = -1;
    float hitT                         = InvalidHitT;
    float alpha                        = 0.0f;
    nrend::OptionalVec<3, float, HasNormals> normal;
};

template <int K, bool HasNormals>
struct GUTKBufferHitParticleKBuffer {
    using HitParticle = GUTKBufferHitParticle<HasNormals>;

    __device__ GUTKBufferHitParticleKBuffer() {
        m_numHits = 0;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            m_kbuffer[i] = HitParticle();
        }
    }

    // insert a new hit into the kbuffer.
    // if the buffer is full overwrite the closest entry
    inline __device__ void insert(HitParticle& hitParticle) {
        const bool isFull = full();
        if (isFull) {
            m_kbuffer[0].hitT = HitParticle::InvalidHitT;
        } else {
            m_numHits++;
        }
#pragma unroll
        for (int i = K - 1; i >= 0; --i) {
            if (hitParticle.hitT > m_kbuffer[i].hitT) {
                const HitParticle tmp = m_kbuffer[i];
                m_kbuffer[i]          = hitParticle;
                hitParticle           = tmp;
            }
        }
    }

    inline __device__ const HitParticle& operator[](int i) const {
        return m_kbuffer[i];
    }

    inline __device__ const uint32_t& numHits() const {
        return m_numHits;
    }

    inline __device__ bool full() const {
        return m_numHits == K;
    }

    inline __device__ const HitParticle& closestHit(const HitParticle&) const {
        return m_kbuffer[0];
    }

private:
    HitParticle m_kbuffer[K];
    uint32_t m_numHits;
};

template <bool HasNormals>
struct GUTKBufferHitParticleKBuffer<0, HasNormals> {
    using HitParticle = GUTKBufferHitParticle<HasNormals>;
    constexpr inline __device__ void insert(HitParticle& hitParticle) const {}
    constexpr inline __device__ HitParticle operator[](int) const {
        return HitParticle();
    }
    constexpr inline __device__ uint32_t numHits() const { return 0; }
    constexpr inline __device__ bool full() const { return true; }
    constexpr inline __device__ const HitParticle& closestHit(const HitParticle& hitParticle) const {
        return hitParticle;
    }
};

template <typename Particles,
          typename Params                   = GUTKBufferRendererDummyParams,
          typename TilingParams             = nrend::GUTParameters::DefaultTiling,
          bool EnableFeatures               = true,
          bool EnableExtendedFeatures       = true,
          bool EnableCameraExtendedFeatures = true,
          bool EnableLidarExtendedFeatures  = false,
          bool EnableNormals                = false,
          bool EnableRayGradients           = false,
          bool Backward                     = false>
struct GUTKBufferRenderer : Params {

    static constexpr int EnabledFeaturesDim = EnableFeatures ? Particles::FeaturesDim : 0;
    static constexpr int EnabledExtendedFeaturesDim =
        (EnableExtendedFeatures ? Particles::ExtendedFeaturesDim : 0) +
        (EnableCameraExtendedFeatures ? Particles::CameraExtendedFeaturesDim : 0) +
        (EnableLidarExtendedFeatures ? Particles::LidarExtendedFeaturesDim : 0);

    using TFeaturesVec            = tcnn::vec<EnabledFeaturesDim ? EnabledFeaturesDim : 1>;
    using TExtendedFeaturesVec    = tcnn::vec<EnabledExtendedFeaturesDim ? EnabledExtendedFeaturesDim : 1>;
    using TRayPayload             = RayPayload<EnabledFeaturesDim, EnabledExtendedFeaturesDim, EnableNormals>;
    using TRayPayloadBackward     = RayPayloadBackward<EnabledFeaturesDim, EnabledExtendedFeaturesDim, EnableNormals, EnableRayGradients>;
    using TPrecomputedFeaturesVec = tcnn::vec<EnabledFeaturesDim + EnabledExtendedFeaturesDim ? EnabledFeaturesDim + EnabledExtendedFeaturesDim : 1>;
    using HitParticleKBuffer      = GUTKBufferHitParticleKBuffer<Params::KHitBufferSize, EnableNormals>;
    using HitParticle             = typename HitParticleKBuffer::HitParticle;
    using DensityRawParameters    = typename Particles::DensityRawParameters;
    using DensityParameters       = typename Particles::DensityParameters;

    struct PrefetchedParticleData {
        uint32_t idx;
        DensityParameters densityParameters;
    };
    struct PrefetchedFullParticleData {
        uint32_t idx;
        DensityParameters densityParameters; //< TODO : check if we need the matrix
        tcnn::vec4 quaternion;
    };

    template <typename TRayPayload>
    static inline __device__ void processHitParticle(
        TRayPayload& ray,
        const HitParticle& hitParticle,
        const Particles& particles,
        const TPrecomputedFeaturesVec* __restrict__ particleFeatures,
        TPrecomputedFeaturesVec* __restrict__ particleFeaturesGradient) {

        using namespace nrend;

        if constexpr (Backward) {
            float hitAlphaGrad = 0.f;
            if constexpr (Params::PerRayParticleFeatures) {
                // TODO : support ray direction gradient brackward
                if constexpr (EnabledFeaturesDim) {
                    particles.template featuresIntegrateBwdToBuffer<false>(
                        ray.direction,
                        ray.directionGradient.ptr(),
                        hitParticle.alpha,
                        hitAlphaGrad,
                        hitParticle.idx,
                        particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                        sliceVec<0, EnabledFeaturesDim>(ray.features.vec),
                        sliceVec<0, EnabledFeaturesDim>(ray.featuresGradient.vec));
                }

                if constexpr (EnabledExtendedFeaturesDim) {
                    particles.template extendedFeaturesIntegrateBwdToBuffer<EnableExtendedFeatures,
                                                                            EnableCameraExtendedFeatures,
                                                                            EnableLidarExtendedFeatures,
                                                                            false,
                                                                            false>(
                        ray.direction,
                        ray.directionGradient.ptr(),
                        hitParticle.alpha,
                        hitAlphaGrad,
                        hitParticle.idx,
                        particles.template extendedFeaturesFromBuffer<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                            hitParticle.idx, ray.direction),
                        sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.features.vec),
                        sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.featuresGradient.vec));
                }

            } else {
                if constexpr (EnabledFeaturesDim) {
                    TFeaturesVec particleFeaturesGradientVec = TFeaturesVec::zero();
                    particles.featuresIntegrateBwd(hitParticle.alpha,
                                                   hitAlphaGrad,
                                                   sliceVec<0, EnabledFeaturesDim>(particleFeatures[hitParticle.idx]),
                                                   particleFeaturesGradientVec,
                                                   sliceVec<0, EnabledFeaturesDim>(ray.features.vec),
                                                   sliceVec<0, EnabledFeaturesDim>(ray.featuresGradient.vec));
#pragma unroll
                    for (int i = 0; i < EnabledFeaturesDim; ++i) {
                        atomicAdd(&(particleFeaturesGradient[hitParticle.idx][i]), particleFeaturesGradientVec[i]);
                    }
                }

                if constexpr (EnabledExtendedFeaturesDim) {
                    TExtendedFeaturesVec extendedFeaturesGradientVec = TExtendedFeaturesVec::zero();
                    particles.template extendedFeaturesIntegrateBwdToVec<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                        hitParticle.alpha,
                        hitAlphaGrad,
                        sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(particleFeatures[hitParticle.idx]),
                        extendedFeaturesGradientVec,
                        sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.features.vec),
                        sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.featuresGradient.vec));

#pragma unroll
                    for (int i = 0; i < EnabledExtendedFeaturesDim; ++i) {
                        atomicAdd(&(particleFeaturesGradient[hitParticle.idx][EnabledFeaturesDim + i]), extendedFeaturesGradientVec[i]);
                    }
                }
            }

            particles.template densityProcessHitBwdToBuffer<false>(ray.origin,
                                                                   ray.originGradient.ptr(),
                                                                   ray.direction,
                                                                   ray.directionGradient.ptr(),
                                                                   // FIXME : correct sensor specific support (eg LIDAR)
                                                                   Params::MinProjectedRayRadius * ray.spread,
                                                                   hitParticle.idx,
                                                                   hitParticle.alpha,
                                                                   hitAlphaGrad,
                                                                   ray.transmittanceBackward,
                                                                   ray.transmittanceGradient,
                                                                   hitParticle.hitT,
                                                                   ray.hitT,
                                                                   ray.hitTGradient,
                                                                   hitParticle.normal.ptr(),
                                                                   ray.normal.ptr(),
                                                                   ray.normalGradient.ptr());

            ray.transmittance *= (1.0f - hitParticle.alpha);

        } else {
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha,
                                              ray.transmittance,
                                              hitParticle.hitT,
                                              ray.hitT,
                                              hitParticle.normal.ptr(),
                                              ray.normal.ptr());

            if constexpr (EnabledExtendedFeaturesDim) {
                particles.template extendedFeaturesIntegrateFwd<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                    hitWeight,
                    particles.template extendedFeaturesFromBuffer<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                        hitParticle.idx, ray.direction),
                    sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.features.vec));
            }

            if constexpr (EnabledFeaturesDim) {
                TFeaturesVec particleFeaturesVec;
                if constexpr (Params::PerRayParticleFeatures) {
                    particleFeaturesVec = particles.featuresFromBuffer(hitParticle.idx, ray.direction);
                } else {
                    particleFeaturesVec = particleFeatures[hitParticle.idx];
                }
                particles.featureIntegrateFwd(hitWeight,
                                              particleFeaturesVec,
                                              sliceVec<0, EnabledFeaturesDim>(ray.features.vec));
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
                                       const tcnn::vec2* __restrict__ /*particlesProjectedPositionPtr*/,
                                       const tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityPtr*/,
                                       const TPrecomputedFeaturesVec* __restrict__ particlesPrecomputedFeaturesPtr,
                                       nrend::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/            = nullptr,
                                       tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/        = nullptr,
                                       TPrecomputedFeaturesVec* __restrict__ particlesPrecomputedFeaturesGradPtr = nullptr,
                                       nrend::MemoryHandles parametersGradient                                   = {}) {

        using namespace nrend;

        const uint32_t tileIdx                                  = threadSensorRayTileIdx(params.sensorModel);
        const uint32_t tileThreadIdx                            = threadIdx.y * blockDim.x + threadIdx.x;
        const tcnn::uvec2 tileParticleRangeIndices              = sortedTileRangeIndicesPtr[tileIdx];
        uint32_t tileNumParticlesToProcess                      = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t tileNumBlocksToProcess                   = tcnn::div_round_up(tileNumParticlesToProcess, TilingParams::BlockSize);
        const TPrecomputedFeaturesVec* particleFeaturesBuffer   = Params::PerRayParticleFeatures ? nullptr : particlesPrecomputedFeaturesPtr;
        TPrecomputedFeaturesVec* particleFeaturesGradientBuffer = (Params::PerRayParticleFeatures || !Backward) ? nullptr : particlesPrecomputedFeaturesGradPtr;

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        if constexpr (EnabledFeaturesDim) {
            particles.initializeFeatures(parameters);
            if constexpr (Backward && Params::PerRayParticleFeatures) {
                particles.initializeFeaturesGradient(parametersGradient);
            }
        }
        if constexpr (EnabledExtendedFeaturesDim) {
            particles.template initializeExtendedFeatures<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(parameters);
            if constexpr (Backward) {
                particles.template initializeExtendedFeaturesGradient<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(parametersGradient);
            }
        }

        if constexpr (Params::EnableWarpAtomicOptim && Backward && (Params::KHitBufferSize == 0)) {
            evalBackwardNoKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                  sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        } else {
            evalKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                        sortedTileParticleIdxPtr, sceneDataPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer);
        }
    }

    template <typename TRay>
    static inline __device__ void evalKBuffer(TRay& ray,
                                              Particles& particles,
                                              const tcnn::uvec2& tileParticleRangeIndices,
                                              uint32_t tileNumBlocksToProcess,
                                              uint32_t tileNumParticlesToProcess,
                                              const uint32_t tileThreadIdx,
                                              const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                              float* __restrict__ sceneDataPtr,
                                              const TPrecomputedFeaturesVec* __restrict__ particleFeaturesBuffer,
                                              TPrecomputedFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        using namespace nrend;
        __shared__ PrefetchedParticleData prefetchedParticlesData[TilingParams::BlockSize];

        HitParticleKBuffer hitParticleKBuffer;

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= TilingParams::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * TilingParams::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; ray.isAlive() && j < min(TilingParams::BlockSize, tileNumParticlesToProcess); j++) {

                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }

                HitParticle hitParticle;
                hitParticle.idx = particleData.idx;
                if (particles.densityHit(ray.origin,
                                         ray.direction,
                                         // FIXME : correct sensor specific support (eg LIDAR)
                                         Params::MinProjectedRayRadius * ray.spread,
                                         particleData.densityParameters,
                                         hitParticle.alpha,
                                         hitParticle.hitT,
                                         hitParticle.normal.ptr()) &&
                    (hitParticle.hitT > ray.tMinMax.x) &&
                    (hitParticle.hitT < ray.tMinMax.y)) {

                    if constexpr (Params::SceneDataWeightsOffset >= 0) {
                        if (sceneDataPtr) {
                            float accumulatedAlpha = hitParticle.alpha;
                            reduceActiveWarpSumToBufferScalar(accumulatedAlpha,
                                                              &sceneDataPtr[particleData.idx * Params::SceneDataDim + Params::SceneDataWeightsOffset],
                                                              tileThreadIdx);
                        }
                    }

                    if (hitParticleKBuffer.full()) {
                        processHitParticle(ray,
                                           hitParticleKBuffer.closestHit(hitParticle),
                                           particles,
                                           particleFeaturesBuffer,
                                           particleFeaturesGradientBuffer);
                    }
                    hitParticleKBuffer.insert(hitParticle);
                }
            }
        }

        if constexpr (Params::KHitBufferSize > 0) {
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                processHitParticle(ray,
                                   hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i],
                                   particles,
                                   particleFeaturesBuffer,
                                   particleFeaturesGradientBuffer);
            }
        }
    }

    template <typename TRay>
    static inline __device__ void evalBackwardNoKBuffer(TRay& ray,
                                                        Particles& particles,
                                                        const tcnn::uvec2& tileParticleRangeIndices,
                                                        uint32_t tileNumBlocksToProcess,
                                                        uint32_t tileNumParticlesToProcess,
                                                        const uint32_t tileThreadIdx,
                                                        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                        const TPrecomputedFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                        TPrecomputedFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        static_assert(Backward && (Params::KHitBufferSize == 0), "Optimized path for backward pass with no KBuffer");

        using namespace nrend;
        __shared__ PrefetchedFullParticleData prefetchedFullParticlesData[TilingParams::BlockSize];

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= TilingParams::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * TilingParams::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    DensityRawParameters densityRawParameters                    = particles.fetchDensityRawParameters(particleIdx);
                    prefetchedFullParticlesData[tileThreadIdx].densityParameters = particles.densityParametersFromRaw(densityRawParameters);
                    prefetchedFullParticlesData[tileThreadIdx].quaternion =
                        {densityRawParameters.quaternion.x, densityRawParameters.quaternion.y, densityRawParameters.quaternion.z, densityRawParameters.quaternion.w};
                    prefetchedFullParticlesData[tileThreadIdx].idx = particleIdx;
                } else {
                    prefetchedFullParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedFullParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; j < min(TilingParams::BlockSize, tileNumParticlesToProcess); j++) {

                const PrefetchedFullParticleData particleData = prefetchedFullParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    ray.kill();
                }

                if (__all_sync(nrend::GUTParameters::WarpMask, !ray.isAlive())) {
                    break;
                }

                DensityRawParameters densityRawParametersGrad;
                densityRawParametersGrad.density    = 0.0f;
                densityRawParametersGrad.position   = tcnn::vec3(0.0f);
                densityRawParametersGrad.quaternion = tcnn::vec4(0.0f);
                densityRawParametersGrad.scale      = tcnn::vec3(0.0f);

                TPrecomputedFeaturesVec featuresGrad = TPrecomputedFeaturesVec::zero();
                static_assert(TRayPayload::BaseFeatDim == EnabledFeaturesDim, "FeaturesDim mismatch");
                static_assert(TRayPayload::ExtFeatDim == EnabledExtendedFeaturesDim, "ExtendedFeaturesDim mismatch");

                HitParticle hitParticle;
                hitParticle.idx = particleData.idx;

                bool validHit = false;
                if (ray.isAlive()) {
                    validHit = particles.densityHit(ray.origin,
                                                    ray.direction,
                                                    // FIXME : correct sensor specific support (eg LIDAR)
                                                    Params::MinProjectedRayRadius * ray.spread,
                                                    particleData.densityParameters,
                                                    hitParticle.alpha,
                                                    hitParticle.hitT,
                                                    hitParticle.normal.ptr()) &&
                               (hitParticle.hitT > ray.tMinMax.x) &&
                               (hitParticle.hitT < ray.tMinMax.y);
                }
                if (validHit) {

                    float hitAlphaGrad = 0.f;
                    if constexpr (EnabledFeaturesDim) {
                        if constexpr (Params::PerRayParticleFeatures) {
                            // FIXME : support atomic warp sum for per ray particle features
                            particles.template featuresIntegrateBwdToBuffer<false>(ray.direction,
                                                                                   ray.directionGradient.ptr(),
                                                                                   hitParticle.alpha,
                                                                                   hitAlphaGrad,
                                                                                   hitParticle.idx,
                                                                                   particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                                                                                   sliceVec<0, EnabledFeaturesDim>(ray.features.vec),
                                                                                   sliceVec<0, EnabledFeaturesDim>(ray.featuresGradient.vec));
                        } else {
                            particles.featuresIntegrateBwd(hitParticle.alpha,
                                                           hitAlphaGrad,
                                                           sliceVec<0, EnabledFeaturesDim>(particleFeaturesBuffer[hitParticle.idx]),
                                                           sliceVec<0, EnabledFeaturesDim>(featuresGrad),
                                                           sliceVec<0, EnabledFeaturesDim>(ray.features.vec),
                                                           sliceVec<0, EnabledFeaturesDim>(ray.featuresGradient.vec));
                        }
                    }
                    if constexpr (EnabledExtendedFeaturesDim) {
                        if constexpr (Params::PerRayParticleFeatures) {
                            // FIXME : support atomic warp sum for per ray particle features
                            particles.template extendedFeaturesIntegrateBwdToBuffer<EnableExtendedFeatures,
                                                                                    EnableCameraExtendedFeatures,
                                                                                    EnableLidarExtendedFeatures,
                                                                                    false,
                                                                                    false>(
                                ray.direction,
                                ray.directionGradient.ptr(),
                                hitParticle.alpha,
                                hitAlphaGrad,
                                hitParticle.idx,
                                particles.template extendedFeaturesFromBuffer<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                                    hitParticle.idx, ray.direction),
                                sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.features.vec),
                                sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.featuresGradient.vec));
                        } else {
                            particles.template extendedFeaturesIntegrateBwdToVec<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                                hitParticle.alpha,
                                hitAlphaGrad,
                                sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(particleFeaturesBuffer[hitParticle.idx]),
                                sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.features.vec),
                                sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(ray.featuresGradient.vec),
                                sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(featuresGrad));
                        }
                    }

                    particles.densityProcessHitBwdToRawParameters(ray.origin,
                                                                  ray.originGradient.ptr(),
                                                                  ray.direction,
                                                                  ray.directionGradient.ptr(),
                                                                  // FIXME : correct sensor specific support (eg LIDAR)
                                                                  Params::MinProjectedRayRadius * ray.spread,
                                                                  particleData.idx,
                                                                  hitParticle.alpha,
                                                                  hitAlphaGrad,
                                                                  ray.transmittanceBackward,
                                                                  ray.transmittanceGradient,
                                                                  hitParticle.hitT,
                                                                  ray.hitT,
                                                                  ray.hitTGradient,
                                                                  particleData.densityParameters,
                                                                  particleData.quaternion,
                                                                  densityRawParametersGrad,
                                                                  hitParticle.normal.ptr(),
                                                                  ray.normal.ptr(),
                                                                  ray.normalGradient.ptr());

                    ray.transmittance *= (1.0f - hitParticle.alpha);
                    if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                        ray.kill();
                    }
                }

                if (__all_sync(nrend::GUTParameters::WarpMask, !validHit)) {
                    continue;
                }

                if constexpr (EnabledFeaturesDim + EnabledExtendedFeaturesDim) {
                    if constexpr (!Params::PerRayParticleFeatures) {
                        // integrate both features and extended features at the same time
                        reduceWarpSumToBuffer<EnabledFeaturesDim + EnabledExtendedFeaturesDim>(featuresGrad,
                                                                                               &particleFeaturesGradientBuffer[particleData.idx],
                                                                                               tileThreadIdx);
                    }
                }

                particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
            }
        }
    }
};
