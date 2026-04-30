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

#include <nrend/kernels/cuda/common/nreStdUtils.cuh>
#include <nrend/kernels/cuda/common/rayPayloadBackward.cuh>
#include <nrend/utils/nreVec.h>

#include <optix.h>

/* Reference Optix Tracer :

    Reference optix tracer is using the anyHit shader to collect the K (up-to 16) closest hits per traversal.
*/

struct GRTReferenceOptixTracerDummyParams {
    static constexpr int KHitBufferSize           = 0;
    static constexpr bool InstancePrimitive       = false;
    static constexpr bool InstanceIdAsOpacity     = false;
    static constexpr bool DensityScaleClamping    = true;
    static constexpr uint32_t IndicesPerPrimitive = 0;
    static constexpr uint32_t OptixTraceRayFlags  = 0;
    static constexpr float NearDistance           = 0.f;
    static constexpr float FarDistance            = 0.f;
};

template <typename Particles,
          typename Params                   = GRTReferenceOptixTracerDummyParams,
          bool EnableFeatures               = true,
          bool EnableExtendedFeatures       = true,
          bool EnableCameraExtendedFeatures = true,
          bool EnableLidarExtendedFeatures  = false,
          bool EnableNormals                = false,
          bool EnableRayGradients           = false,
          bool kBackward                    = false>
struct GRTReferenceOptixTracer : Params {

    static constexpr bool Backward          = kBackward;
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
    using DensityRawParameters    = typename Particles::DensityRawParameters;
    using DensityParameters       = typename Particles::DensityParameters;

    struct HitParticle {
        static constexpr float InvalidHitT          = -1.0f;
        static constexpr float FarDistance          = Params::FarDistance;
        static constexpr uint32_t InvalidParticleId = 0xFFFFFFFF;
        uint32_t idx                                = InvalidParticleId;
        float hitT                                  = InvalidHitT;
        float alpha                                 = 0.0f;
        nrend::OptionalVec<3, float, EnableNormals> normal;
    };

    template <typename TRayPayload>
    static inline __device__ void processHitParticle(
        TRayPayload& ray,
        const HitParticle& hitParticle,
        const Particles& particles) {

        using namespace nrend;

        if constexpr (Backward) {
            float hitAlphaGrad = 0.f;
            if constexpr (EnabledFeaturesDim) {
                particles.template featuresIntegrateBwdToBuffer<false>(ray.direction,
                                                                       ray.directionGradient.ptr(),
                                                                       hitParticle.alpha,
                                                                       hitAlphaGrad,
                                                                       hitParticle.idx,
                                                                       particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                                                                       sliceVec<0, TRayPayload::BaseFeatDim>(ray.features.vec),
                                                                       sliceVec<0, TRayPayload::BaseFeatDim>(ray.featuresGradient.vec));
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
                    sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.features.vec),
                    sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.featuresGradient.vec));
            }

            particles.template densityProcessHitBwdToBuffer<false>(ray.origin,
                                                                   ray.originGradient.ptr(),
                                                                   ray.direction,
                                                                   ray.directionGradient.ptr(),
                                                                   // FIXME : ray spread not supported
                                                                   0.f, // ray.spread,
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
                    sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.features.vec));
            }

            if constexpr (EnabledFeaturesDim) {
                TFeaturesVec particleFeaturesVec;
                particleFeaturesVec = particles.featuresFromBuffer(hitParticle.idx, ray.direction);

                particles.featureIntegrateFwd(hitWeight,
                                              particleFeaturesVec,
                                              sliceVec<0, TRayPayload::BaseFeatDim>(ray.features.vec));
            }
        }

        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            ray.kill();
        }
    }

    template <bool KnownHitDistance, bool KnownHitAlpha, typename TRay>
    static inline __device__ bool validateAndProcessHit(TRay& ray,
                                                        HitParticle& hitParticle,
                                                        const Particles& particles,
                                                        float* __restrict__ sceneDataPtr = nullptr) {

        // FIXME : avoid recomputing the full hit (we already have the distance, just need the alpha and the normal)
        if (!particles.template densityHit<KnownHitDistance, KnownHitAlpha>(ray.origin,
                                                                            ray.direction,
                                                                            0.f, // FIXME : support ray spread
                                                                            particles.fetchDensityParameters(hitParticle.idx),
                                                                            hitParticle.alpha,
                                                                            hitParticle.hitT,
                                                                            hitParticle.normal.ptr())) {
            return false;
        }
        if constexpr (Params::SceneDataWeightsOffset >= 0) {
            if (sceneDataPtr) {
                atomicAdd(&sceneDataPtr[hitParticle.idx * Params::SceneDataDim + Params::SceneDataWeightsOffset], hitParticle.alpha);
            }
        }
        processHitParticle(ray,
                           hitParticle,
                           particles);
        return true;
    }

    template <typename TRay>
    static inline __device__ void raygen(OptixTraversableHandle traversableHandle,
                                         const nrend::RenderParameters& params,
                                         TRay& ray,
                                         float* __restrict__ sceneDataPtr,
                                         nrend::MemoryHandles parameters,
                                         nrend::MemoryHandles parametersGradient = {}) {

        static_assert(Params::KHitBufferSize <= 16, "ReferenceOptixTracer : KHitBufferSize must be <= 16");

        using namespace nrend;

        Particles particles;
        particles.initializeDensity(parameters);
        particles.initializeDensityGradient(parametersGradient);

        if constexpr (EnabledFeaturesDim) {
            particles.initializeFeatures(parameters);
            particles.initializeFeaturesGradient(parametersGradient);
        }

        if constexpr (EnabledExtendedFeaturesDim) {
            particles.template initializeExtendedFeatures<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                parameters);
            particles.template initializeExtendedFeaturesGradient<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                parametersGradient);
        }

        constexpr float epsT    = 1e-9f;
        uint32_t hitCount       = 0;
        float traceStartT       = max(ray.tMinMax.x, Params::NearDistance);
        const float traceEndT   = min(ray.tMinMax.y + epsT, Params::FarDistance);
        uint32_t traversedCount = 0;

        while (ray.isAlive()) {

            if constexpr (Params::KHitBufferSize > 0) {

                tcnn::uvec<Params::KHitBufferSize> hitSampleParticleIds = {HitParticle::InvalidParticleId};
                tcnn::uvec<Params::KHitBufferSize> hitSampleDistances   = {__float_as_uint(HitParticle::FarDistance)};

                // Trace the ray against our scene hierarchy
                // clang-format off
                optixTrace(traversableHandle,
                           *reinterpret_cast<float3*>(&ray.origin),
                           *reinterpret_cast<float3*>(&ray.direction),
                           traceStartT + epsT,
                           traceEndT,
                           0.0f, // rayTime : ignored when no motion
                           OptixVisibilityMask(255),
                           Params::OptixTraceRayFlags,
                           0, // SBT offset   -- See SBT discussion
                           1, // SBT stride   -- See SBT discussion
                           0 // missSBTIndex -- See SBT discussion
                           #if GRTReferenceOptixTracer_KHitBufferSize > 15  
                           ,hitSampleParticleIds[15], hitSampleDistances[15] 
                           #endif 
                           #if GRTReferenceOptixTracer_KHitBufferSize > 14
                           ,hitSampleParticleIds[14], hitSampleDistances[14]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 13
                           ,hitSampleParticleIds[13], hitSampleDistances[13]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 12
                           ,hitSampleParticleIds[12], hitSampleDistances[12]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 11
                           ,hitSampleParticleIds[11], hitSampleDistances[11]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 10
                           ,hitSampleParticleIds[10], hitSampleDistances[10]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 9
                           ,hitSampleParticleIds[9], hitSampleDistances[9]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 8
                           ,hitSampleParticleIds[8], hitSampleDistances[8]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 7
                           ,hitSampleParticleIds[7], hitSampleDistances[7]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 6
                           ,hitSampleParticleIds[6], hitSampleDistances[6]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 5
                           ,hitSampleParticleIds[5], hitSampleDistances[5]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 4
                           ,hitSampleParticleIds[4], hitSampleDistances[4]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 3
                           ,hitSampleParticleIds[3], hitSampleDistances[3]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 2
                           ,hitSampleParticleIds[2], hitSampleDistances[2]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 1
                           ,hitSampleParticleIds[1], hitSampleDistances[1]
                           #endif
                           ,hitSampleParticleIds[0], hitSampleDistances[0]
                );
                // clang-format on

#pragma unroll
                for (int i = 0; i < Params::KHitBufferSize; ++i) {
                    if (ray.isAlive() && (hitSampleParticleIds[i] != HitParticle::InvalidParticleId)) {
                        traversedCount++;
                        HitParticle hitParticle;
                        hitParticle.idx  = hitSampleParticleIds[i];
                        hitParticle.hitT = __uint_as_float(hitSampleDistances[i]);
                        // NB : traceStartT is the hit distance to the primitive (not to the particle)
                        traceStartT = fmaxf(traceStartT, hitParticle.hitT);
                        if (validateAndProcessHit<true, false>(ray, hitParticle, particles, sceneDataPtr)) {
                            if (hitCount == 0) {
                                ray.tMinMax.x = traceStartT;
                            };
                            ray.tMinMax.y = traceStartT;
                            ++hitCount;
                        }
                    }
                }

                // last hit particle is invalid, kill the ray
                if (hitSampleParticleIds[Params::KHitBufferSize - 1] == HitParticle::InvalidParticleId) {
                    ray.kill();
                }
            } else {
                optixTraverse(traversableHandle,
                              *reinterpret_cast<float3*>(&ray.origin),
                              *reinterpret_cast<float3*>(&ray.direction),
                              traceStartT + epsT,
                              traceEndT,
                              0.0f, // rayTime : ignored when no motion
                              OptixVisibilityMask(255),
                              Params::OptixTraceRayFlags,
                              0, // SBT offset   -- See SBT discussion
                              1, // SBT stride   -- See SBT discussion
                              0  // missSBTIndex -- See SBT discussion
                );
                if (optixHitObjectIsHit()) {
                    traversedCount++;
                    HitParticle hitParticle;
                    hitParticle.idx  = optixHitObjectPrimitiveIndex();
                    hitParticle.hitT = optixHitObjectGetRayTmax();
                    // NB : traceStartT is the hit distance to the primitive (not to the ray)
                    traceStartT = fmaxf(traceStartT, hitParticle.hitT);
                    if (validateAndProcessHit<true, false>(ray, hitParticle, particles, sceneDataPtr)) {
                        if (hitCount == 0) {
                            ray.tMinMax.x = traceStartT;
                        };
                        ray.tMinMax.y = traceStartT;
                        ++hitCount;
                    }
                } else {
                    ray.kill();
                }
            }
        }
    }

    static inline __device__ void intersect(nrend::MemoryHandles parameters) {
        float hitDistance;
        bool intersect = false;
        if constexpr (Params::InstancePrimitive) {
            intersect = Particles::densityCanonicalRayHitDistance(optixGetObjectRayOrigin(),
                                                                  optixGetObjectRayDirection(),
                                                                  optixGetRayTmin(),
                                                                  optixGetRayTmax(),
                                                                  hitDistance);
        } else {
            Particles particles;
            particles.initializeDensity(parameters);
            intersect = particles.densityRayHit(optixGetWorldRayOrigin(),
                                                optixGetWorldRayDirection(),
                                                optixGetPrimitiveIndex(),
                                                optixGetRayTmin(),
                                                optixGetRayTmax(),
                                                hitDistance);
        }
        if (intersect) {
            optixReportIntersection(hitDistance, 0);
        }
    }

    static inline __device__ uint32_t optixHitObjectPrimitiveIndex() {
        return Params::InstancePrimitive ? optixHitObjectGetInstanceIndex() : optixHitObjectGetPrimitiveIndex() / Params::IndicesPerPrimitive;
    }

    static inline __device__ uint32_t optixPrimitiveIndex() {
        return Params::InstancePrimitive ? optixGetInstanceIndex() : optixGetPrimitiveIndex() / Params::IndicesPerPrimitive;
    }

    static inline __device__ void anyhit(nrend::MemoryHandles) {
        if constexpr (Params::KHitBufferSize > 0) {
            struct RayHit {
                unsigned int particleId;
                float distance;
            } hit = RayHit{optixPrimitiveIndex(), optixGetRayTmax()};

#define compareAndSwapHitPayloadValue(hit, i_id, i_distance)                      \
    if constexpr (Params::KHitBufferSize > i_id / 2) {                            \
        const float distance = __uint_as_float(optixGetPayload_##i_distance##()); \
        if (hit.distance < distance) {                                            \
            optixSetPayload_##i_distance##(__float_as_uint(hit.distance));        \
            const uint32_t id = optixGetPayload_##i_id##();                       \
            optixSetPayload_##i_id##(hit.particleId);                             \
            hit.distance   = distance;                                            \
            hit.particleId = id;                                                  \
        }                                                                         \
    }

            if (hit.distance < __uint_as_float(optixGetPayload_1())) {
                compareAndSwapHitPayloadValue(hit, 30, 31);
                compareAndSwapHitPayloadValue(hit, 28, 29);
                compareAndSwapHitPayloadValue(hit, 26, 27);
                compareAndSwapHitPayloadValue(hit, 24, 25);
                compareAndSwapHitPayloadValue(hit, 22, 23);
                compareAndSwapHitPayloadValue(hit, 20, 21);
                compareAndSwapHitPayloadValue(hit, 18, 19);
                compareAndSwapHitPayloadValue(hit, 16, 17);
                compareAndSwapHitPayloadValue(hit, 14, 15);
                compareAndSwapHitPayloadValue(hit, 12, 13);
                compareAndSwapHitPayloadValue(hit, 10, 11);
                compareAndSwapHitPayloadValue(hit, 8, 9);
                compareAndSwapHitPayloadValue(hit, 6, 7);
                compareAndSwapHitPayloadValue(hit, 4, 5);
                compareAndSwapHitPayloadValue(hit, 2, 3);
                compareAndSwapHitPayloadValue(hit, 0, 1);

                // ignore all inserted hits, except if the last one
                if (__uint_as_float(optixGetPayload_1()) > optixGetRayTmax()) {
                    optixIgnoreIntersection();
                }
            }

#undef compareAndSwapHitPayloadValue
        }
    }
};
