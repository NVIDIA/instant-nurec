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
struct GRTReferenceInstanceOptixTracer : Params {

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

    template <typename TRay>
    static inline __device__ void raygen(OptixTraversableHandle traversableHandle,
                                         const nrend::RenderParameters& params,
                                         TRay& ray,
                                         float* __restrict__ sceneDataPtr,
                                         nrend::MemoryHandles parameters,
                                         nrend::MemoryHandles parametersGradient = {}) {

        static_assert(Params::KHitBufferSize <= 10, "ReferenceInstanceOptixTracer : KHitBufferSize must be <= 10");

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
        uint32_t hitCount       = 0;
        float traceStartT       = max(ray.tMinMax.x, Params::NearDistance);
        const float traceEndT   = min(ray.tMinMax.y + epsT, Params::FarDistance);
        uint32_t traversedCount = 0;

        while (ray.isAlive()) {

            if constexpr (Params::KHitBufferSize > 0) {

                tcnn::uvec<Params::KHitBufferSize> hitSampleParticleIds = {TBase::HitParticle::InvalidParticleId};
                tcnn::uvec<Params::KHitBufferSize> hitSampleDistances   = {__float_as_uint(TBase::HitParticle::FarDistance)};
                tcnn::uvec<Params::KHitBufferSize> hitSampleAlpha       = {__float_as_uint(0.f)};

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
                           #if GRTReferenceOptixTracer_KHitBufferSize > 9
                           ,hitSampleParticleIds[9], hitSampleDistances[9], hitSampleAlpha[9]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 8
                           ,hitSampleParticleIds[8], hitSampleDistances[8], hitSampleAlpha[8]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 7
                           ,hitSampleParticleIds[7], hitSampleDistances[7], hitSampleAlpha[7]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 6
                           ,hitSampleParticleIds[6], hitSampleDistances[6], hitSampleAlpha[6]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 5
                           ,hitSampleParticleIds[5], hitSampleDistances[5], hitSampleAlpha[5]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 4
                           ,hitSampleParticleIds[4], hitSampleDistances[4], hitSampleAlpha[4]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 3
                           ,hitSampleParticleIds[3], hitSampleDistances[3], hitSampleAlpha[3]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 2
                           ,hitSampleParticleIds[2], hitSampleDistances[2], hitSampleAlpha[2]
                           #endif
                           #if GRTReferenceOptixTracer_KHitBufferSize > 1
                           ,hitSampleParticleIds[1], hitSampleDistances[1], hitSampleAlpha[1]
                           #endif
                           ,hitSampleParticleIds[0], hitSampleDistances[0], hitSampleAlpha[0]
                );
                // clang-format on

#pragma unroll
                for (int i = 0; i < Params::KHitBufferSize; ++i) {
                    if (ray.isAlive() && (hitSampleParticleIds[i] != TBase::HitParticle::InvalidParticleId)) {
                        traversedCount++;
                        typename TBase::HitParticle hitParticle;
                        hitParticle.idx   = hitSampleParticleIds[i];
                        hitParticle.hitT  = __uint_as_float(hitSampleDistances[i]);
                        hitParticle.alpha = __uint_as_float(hitSampleAlpha[i]);
                        // NB : traceStartT is the hit distance to the primitive (not to the particle)
                        traceStartT = fmaxf(traceStartT, hitParticle.hitT);
                        // In backward we have to consider unquantized alpha
                        const bool validHit = TBase::template validateAndProcessHit<true, !kBackward>(ray, hitParticle, particles, sceneDataPtr);
                        if (validHit) {
                            if (hitCount == 0) {
                                ray.tMinMax.x = traceStartT;
                            };
                            ray.tMinMax.y = traceStartT;
                            ++hitCount;
                        }
                    }
                }

                // last hit particle is invalid, kill the ray
                if (hitSampleParticleIds[Params::KHitBufferSize - 1] == TBase::HitParticle::InvalidParticleId) {
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
                    typename TBase::HitParticle hitParticle;
                    hitParticle.idx   = TBase::optixHitObjectPrimitiveIndex();
                    hitParticle.hitT  = optixHitObjectGetRayTmax();
                    hitParticle.alpha = __uint_as_float(optixHitObjectGetAttribute_0());
                    // NB : traceStartT is the hit distance to the primitive (not to the particle)
                    traceStartT = fmaxf(traceStartT, hitParticle.hitT);
                    // In backward we have to consider unquantized alpha
                    const bool validHit = TBase::template validateAndProcessHit<true, !kBackward>(ray, hitParticle, particles, sceneDataPtr);
                    if (validHit) {
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
        float particleDensity;
        if constexpr (Params::InstanceIdAsOpacity) {
            particleDensity = nrend::instanceIdAsOpacity(optixGetInstanceId());
        } else {
            Particles particles;
            particles.initializeDensity(parameters);
            particleDensity = particles.fetchDensity(TBase::optixPrimitiveIndex());
        }
        float hitDistance;
        float alpha;
        static_assert(Params::InstancePrimitive, "ReferenceInstanceOptixTracer : only support instance primitives");
        const bool intersect = Particles::densityCanonicalRayHitAlpha(optixGetObjectRayOrigin(),
                                                                      optixGetObjectRayDirection(),
                                                                      optixGetRayTmin(),
                                                                      optixGetRayTmax(),
                                                                      alpha,
                                                                      hitDistance,
                                                                      particleDensity,
                                                                      Params::DensityScaleClamping);
        if (intersect) {
            optixReportIntersection(hitDistance, 0, __float_as_uint(alpha));
        }
    }

    static inline __device__ void anyhit(nrend::MemoryHandles parameters) {
        if constexpr (Params::KHitBufferSize > 0) {
            struct RayHit {
                unsigned int particleId;
                float distance;
                uint32_t alphaPayload;
            } hit = RayHit{TBase::optixPrimitiveIndex(), optixGetRayTmax(), optixGetAttribute_0()};

#define compareAndSwapHitPayloadValue(hit, i_id, i_distance, i_alpha)             \
    if constexpr (Params::KHitBufferSize > i_id / 3) {                            \
        const float distance = __uint_as_float(optixGetPayload_##i_distance##()); \
        if (hit.distance < distance) {                                            \
            const uint32_t swappedId = optixGetPayload_##i_id##();                \
            optixSetPayload_##i_id##(hit.particleId);                             \
            hit.particleId = swappedId;                                           \
            optixSetPayload_##i_distance##(__float_as_uint(hit.distance));        \
            hit.distance                       = distance;                        \
            const uint32_t swappedAlphaPayload = optixGetPayload_##i_alpha##();   \
            optixSetPayload_##i_alpha##(hit.alphaPayload);                        \
            hit.alphaPayload = swappedAlphaPayload;                               \
        }                                                                         \
    }

            if (hit.distance < __uint_as_float(optixGetPayload_1())) {
                compareAndSwapHitPayloadValue(hit, 27, 28, 29);
                compareAndSwapHitPayloadValue(hit, 24, 25, 26);
                compareAndSwapHitPayloadValue(hit, 21, 22, 23);
                compareAndSwapHitPayloadValue(hit, 18, 19, 20);
                compareAndSwapHitPayloadValue(hit, 15, 16, 17);
                compareAndSwapHitPayloadValue(hit, 12, 13, 14);
                compareAndSwapHitPayloadValue(hit, 9, 10, 11);
                compareAndSwapHitPayloadValue(hit, 6, 7, 8);
                compareAndSwapHitPayloadValue(hit, 3, 4, 5);
                compareAndSwapHitPayloadValue(hit, 0, 1, 2);

                // ignore all inserted hits, except if the last one
                if (__uint_as_float(optixGetPayload_1()) > optixGetRayTmax()) {
                    optixIgnoreIntersection();
                }
            }

#undef compareAndSwapHitPayloadValue
        }
    }
};
