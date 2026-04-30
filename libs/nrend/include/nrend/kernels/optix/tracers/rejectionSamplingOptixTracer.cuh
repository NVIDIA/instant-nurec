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

#include <nrend/kernels/cuda/common/random.cuh>
#include <nrend/kernels/cuda/primitives/particleAsPrimitivesUtils.cuh>
#include <nrend/kernels/optix/tracers/referenceOptixTracer.cuh>

/* Reference Optix Tracer with opacity encoded in the instance id.

    - the particle opacity is encoded in the instance id AS of the instance primitive to prevent fetching the density particle data.
    - the opacity is quantized with a maximium of OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID values (only for the forward pass)

    - support a K buffer up-to 10
    - support only instance primitives
*/

template <typename Particles,
          typename Params                   = GRTReferenceOptixTracerDummyParams,
          bool EnableFeatures               = true,
          bool EnableExtendedFeatures       = true,
          bool EnableCameraExtendedFeatures = true,
          bool EnableLidarExtendedFeatures  = false,
          bool EnableNormals                = false,
          bool EnableRayGradients           = false,
          bool kBackward                    = false>
struct GRTRejectionSamplingOptixTracer : Params {

    using TBase               = GRTReferenceOptixTracer<Particles,
                                          Params,
                                          EnableFeatures,
                                          EnableExtendedFeatures,
                                          EnableCameraExtendedFeatures,
                                          EnableLidarExtendedFeatures,
                                          EnableNormals,
                                          EnableRayGradients,
                                          kBackward>;
    using TRayPayload         = typename TBase::TRayPayload;
    using TRayPayloadBackward = typename TBase::TRayPayloadBackward;

    template <typename TRayPayload, typename HitParticle>
    static inline __device__ void processSampleHit(
        TRayPayload& ray,
        const HitParticle& hitParticle,
        const Particles& particles,
        float* __restrict__ sceneDataPtr = nullptr) {

        using namespace nrend;

        if constexpr (kBackward) {
            float hitAlphaGrad = 0.f;
            if constexpr (TBase::EnabledFeaturesDim) {
                particles.template featuresIntegrateBwdToBuffer<false, true>(
                    ray.direction,
                    ray.directionGradient.ptr(),
                    hitParticle.alpha,
                    hitAlphaGrad,
                    hitParticle.idx,
                    particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                    sliceVec<0, TRayPayload::BaseFeatDim>(ray.features.vec),
                    sliceVec<0, TRayPayload::BaseFeatDim>(ray.featuresGradient.vec));
            }

            if constexpr (TBase::EnabledExtendedFeaturesDim) {
                particles.template extendedFeaturesIntegrateBwdToBuffer<EnableExtendedFeatures,
                                                                        EnableCameraExtendedFeatures,
                                                                        EnableLidarExtendedFeatures,
                                                                        false,
                                                                        true>(
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

            particles.template densityProcessHitBwdToBuffer<false, true>(
                ray.origin,
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
                &hitParticle.normal,
                ray.normal.ptr(),
                ray.normalGradient.ptr());

            ray.transmittance -= hitParticle.alpha;

        } else {
            const float hitWeight =
                particles.template densityIntegrateHit<true>(hitParticle.alpha,
                                                             ray.transmittance,
                                                             hitParticle.hitT,
                                                             ray.hitT,
                                                             hitParticle.normal.ptr(),
                                                             ray.normal.ptr());

            if constexpr (TBase::EnabledExtendedFeaturesDim) {
                particles.template extendedFeaturesIntegrateFwd<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                    hitWeight,
                    particles.template extendedFeaturesFromBuffer<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                        hitParticle.idx, ray.direction),
                    sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.features.vec));
            }

            if constexpr (TBase::EnabledFeaturesDim) {
                typename TBase::TFeaturesVec particleFeaturesVec;
                particleFeaturesVec = particles.featuresFromBuffer(hitParticle.idx, ray.direction);

                particles.featureIntegrateFwd(hitWeight,
                                              particleFeaturesVec,
                                              sliceVec<0, TRayPayload::BaseFeatDim>(ray.features.vec));
            }

            if constexpr (Params::SceneDataWeightsOffset >= 0) {
                if (sceneDataPtr) {
                    atomicAdd(&sceneDataPtr[hitParticle.idx * Params::SceneDataDim + Params::SceneDataWeightsOffset], hitParticle.alpha);
                }
            }
        }
    }

    template <typename TRay>
    static inline __device__ void raygen(OptixTraversableHandle traversableHandle,
                                         const nrend::RenderParameters& params,
                                         TRay& ray,
                                         float* __restrict__ sceneDataPtr,
                                         nrend::MemoryHandles parameters,
                                         nrend::MemoryHandles parametersGradient = {}) {

        // TODO : support backward pass (current implementation does not take into account the particles occlusions which lead to poor results)
        static_assert(!kBackward, "RejectionSamplingInstanceOptixTracer : Backward pass not supported");
        static_assert(Params::KHitBufferSize <= 15, "RejectionSamplingInstanceOptixTracer : KHitBufferSize must be <= 15");

        using namespace nrend;

        Particles particles;
        particles.initializeDensity(parameters);
        particles.initializeDensityGradient(parametersGradient);

        if constexpr (TBase::EnabledFeaturesDim) {
            particles.initializeFeatures(parameters);
            particles.initializeFeaturesGradient(parametersGradient);
        }

        if constexpr (TBase::EnabledExtendedFeaturesDim) {
            particles.template initializeExtendedFeatures<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                parameters);
            particles.template initializeExtendedFeaturesGradient<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(
                parametersGradient);
        }

        constexpr float epsT    = 1e-9f;
        const float traceStartT = max(ray.tMinMax.x, Params::NearDistance);
        const float traceEndT   = min(ray.tMinMax.y + epsT, Params::FarDistance);

        if constexpr (Params::KHitBufferSize > 0) {
#pragma unroll
            for (int i = 0; i < Params::NumSamples; i += Params::KHitBufferSize) {

                tcnn::uvec<Params::KHitBufferSize> hitSampleParticleIds = {TBase::HitParticle::InvalidParticleId};
                tcnn::uvec<Params::KHitBufferSize> hitSampleDistances   = {__float_as_uint(TBase::HitParticle::FarDistance)};

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
                           0, // missSBTIndex -- See SBT discussion
                           ray.rndSeed
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
                for (int j = 0; j < Params::KHitBufferSize; ++j) {
                    if ((j + i < Params::NumSamples) && (hitSampleParticleIds[j] != TBase::HitParticle::InvalidParticleId)) {
                        typename TBase::HitParticle hitParticle;
                        hitParticle.idx   = hitSampleParticleIds[j];
                        hitParticle.hitT  = __uint_as_float(hitSampleDistances[j]);
                        hitParticle.alpha = 1.f / Params::NumSamples;
                        if constexpr (EnableNormals) {
                            // TODO : replace with normal computation
                            particles.template densityHit<true, true>(ray.origin,
                                                                      ray.direction,
                                                                      0.f, // FIXME : support ray spread
                                                                      particles.fetchDensityParameters(hitParticle.idx),
                                                                      hitParticle.alpha,
                                                                      hitParticle.hitT,
                                                                      hitParticle.normal.ptr());
                        }
                        processSampleHit(ray, hitParticle, particles, sceneDataPtr);
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < Params::NumSamples; ++i) {
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
                              0, // missSBTIndex -- See SBT discussion
                              ray.rndSeed);
                if (optixHitObjectIsHit()) {
                    typename TBase::HitParticle hitParticle;
                    hitParticle.idx   = TBase::optixHitObjectPrimitiveIndex();
                    hitParticle.hitT  = optixHitObjectGetRayTmax();
                    hitParticle.alpha = 1.f / Params::NumSamples;
                    if constexpr (EnableNormals) {
                        // TODO : replace with normal computation
                        particles.template densityHit<true, true>(ray.origin,
                                                                  ray.direction,
                                                                  0.f, // FIXME : support ray spread
                                                                  particles.fetchDensityParameters(hitParticle.idx),
                                                                  hitParticle.alpha,
                                                                  hitParticle.hitT,
                                                                  hitParticle.normal.ptr());
                    }
                    processSampleHit(ray, hitParticle, particles, sceneDataPtr);
                }
            }
        }
        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            ray.kill();
        }
    }

    static inline __device__ void intersect(nrend::MemoryHandles parameters) {
        static_assert(Params::InstancePrimitive, "ReferenceInstanceOptixTracer : only support instance primitives");
        const uint32_t particleId = TBase::optixPrimitiveIndex();
        float particleDensity;
        if constexpr (Params::InstanceIdAsOpacity) {
            particleDensity = nrend::instanceIdAsOpacity(optixGetInstanceId());
        } else {
            Particles particles;
            particles.initializeDensity(parameters);
            particleDensity = particles.fetchDensity(particleId);
        }
        float hitDistance;
        float alpha;
        const bool intersect = Particles::densityCanonicalRayHitAlpha(optixGetObjectRayOrigin(),
                                                                      optixGetObjectRayDirection(),
                                                                      optixGetRayTmin(),
                                                                      optixGetRayTmax(),
                                                                      alpha,
                                                                      hitDistance,
                                                                      particleDensity,
                                                                      Params::DensityScaleClamping);
        if (intersect) {

            uint32_t rndSeed = optixGetPayload_0();

            if constexpr (Params::KHitBufferSize > 0) {
                float maxMinSampleDistance = -TBase::HitParticle::FarDistance;

#define payloadRejectionSampling(i_id, i_distance)                                      \
    if constexpr (Params::KHitBufferSize > (i_id - 1) / 2) {                            \
        const float sampleDistance = __uint_as_float(optixGetPayload_##i_distance##()); \
        if ((hitDistance < sampleDistance) && (alpha > rnd(rndSeed))) {                 \
            optixSetPayload_##i_id##(particleId);                                       \
            optixSetPayload_##i_distance##(__float_as_uint(hitDistance));               \
            maxMinSampleDistance = fmaxf(maxMinSampleDistance, hitDistance);            \
        } else {                                                                        \
            maxMinSampleDistance = fmaxf(maxMinSampleDistance, sampleDistance);         \
        }                                                                               \
    }

                payloadRejectionSampling(1, 2);
                payloadRejectionSampling(3, 4);
                payloadRejectionSampling(5, 6);
                payloadRejectionSampling(7, 8);
                payloadRejectionSampling(9, 10);
                payloadRejectionSampling(11, 12);
                payloadRejectionSampling(13, 14);
                payloadRejectionSampling(15, 16);
                payloadRejectionSampling(17, 18);
                payloadRejectionSampling(19, 20);
                payloadRejectionSampling(21, 22);
                payloadRejectionSampling(23, 24);
                payloadRejectionSampling(25, 26);
                payloadRejectionSampling(27, 28);
                payloadRejectionSampling(29, 30);

#undef payloadRejectionSampling

                if (maxMinSampleDistance > 0.f) {
                    optixReportIntersection(maxMinSampleDistance, 0);
                }

            } else {
                if (alpha > rnd(rndSeed)) {
                    optixReportIntersection(hitDistance, 0);
                }
            }

            optixSetPayload_0(rndSeed);
        }
    }

    static inline __device__ void anyhit(nrend::MemoryHandles) {
    }
};
