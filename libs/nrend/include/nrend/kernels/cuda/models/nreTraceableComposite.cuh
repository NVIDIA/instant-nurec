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

#include <nrend/kernels/cuda/common/nreSimpleTuple.cuh>
#include <nrend/kernels/cuda/common/nreStdUtils.cuh>
#include <nrend/kernels/cuda/common/rayPayload.cuh>

// Sample implementation of NRETraceableCompositeParams
struct NRETraceableCompositeDefaultParams {

public:
    static constexpr uint8_t NumPrimitives = 1;

    static constexpr uint8_t NumBackgroundPrimitives = 1;
    static constexpr uint16_t NumPrimitiveInstances  = 1;

    static constexpr uint16_t MaxPrimitiveInstancesBatchSize = 16;

    static const uint16_t primitiveInstancesPackedIdx[NumPrimitiveInstances];

    static constexpr float minTransmittance = nrend::RenderParameters::defaultMinTransmittance;
};
const uint16_t NRETraceableCompositeDefaultParams::primitiveInstancesPackedIdx[NumPrimitiveInstances] = {0};
template <typename TParams,
          typename TAppearanceEmbedding,
          typename TBackground,
          typename TPostProcessings,
          typename... TPrimitives>
struct NRETraceableComposite {

    static constexpr int32_t WarpSize  = 32;
    static constexpr uint32_t WarpMask = 0xFFFFFFFF;

public:
    using TRayPayload = RayPayload<3>;

    // members maybe static iff they do not have any non-static member
    TAppearanceEmbedding m_appearanceEmbedding;
    TBackground m_background;

    using TPrimitivesTuple = NreSimpleTuple<TParams::NumPrimitives, TPrimitives...>;
    static const TPrimitivesTuple m_primitives;

    static constexpr float MaxHitDistance                 = 1e09f;
    static constexpr uint16_t PrimitiveInstancesBatchSize = TParams::NumPrimitiveInstances > 0 ? TParams::MaxPrimitiveInstancesBatchSize : TParams::NumBackgroundPrimitives;

    struct PrimitiveInstanceHit {

        static constexpr uint16_t InvalidInstanceId     = 0x03FF;
        static constexpr uint16_t InvalidActiveTrackIdx = 0xFFFF;
        static constexpr uint16_t InvalidPackedIdx      = 0xFFFF;

        static inline __device__ uint16_t primitiveIdToPackedIdx(uint8_t primitiveId) {
            return (static_cast<uint16_t>(primitiveId) << 10) | InvalidInstanceId;
        }

        tcnn::vec3 localDirection;
        tcnn::vec3 localOrigin;

        float stepHitMaxDistance = MaxHitDistance;
        float stepHitDistance    = MaxHitDistance;
        float stepHitDelta       = 0.0f;

        uint16_t packedIdx      = InvalidPackedIdx;
        uint16_t activeTrackIdx = InvalidActiveTrackIdx;

        inline __device__ uint8_t primitiveId() const { return static_cast<uint8_t>((packedIdx & 0xFC00) >> 10); }
        inline __device__ uint16_t instanceId() const { return packedIdx & InvalidInstanceId; }
    };
    PrimitiveInstanceHit m_primitiveInstanceHits[PrimitiveInstancesBatchSize];
    // primitive instance hit distance past the PrimitiveInstancesBatchSize
    float m_nextPrimitiveInstanceMinStepHitDistance = 0.0f;

    // helper functions to call initializeSteps on a given primitiveInstance
    template <uint8_t PrimitiveId = 0, typename TPrimitive = typename TPrimitivesTuple::THead, typename nreEnableIf<(PrimitiveId >= TParams::NumPrimitives), bool>::type = true>
    static inline __device__ tcnn::vec2 initializePrimitiveInstanceSteps(TPrimitive, const tcnn::vec3&, const tcnn::vec3&, nrend::TTimestamp, const tcnn::vec2&, uint32_t&, nrend::MemoryHandles, uint8_t, uint16_t) {
        __builtin_unreachable();
        return tcnn::vec2::zero();
    }
    template <uint8_t PrimitiveId = 0, typename TPrimitive = typename TPrimitivesTuple::THead, typename nreEnableIf<(PrimitiveId < TParams::NumPrimitives), bool>::type = true>
    static inline __device__ tcnn::vec2 initializePrimitiveInstanceSteps(TPrimitive primitive, const tcnn::vec3& rayOrigin, const tcnn::vec3& rayDirection, nrend::TTimestamp rayTimestamp, const tcnn::vec2& rayMinMax, uint32_t& rndSeed, nrend::MemoryHandles parameters, uint8_t primitiveId, uint16_t instanceId) {

        if (primitiveId == PrimitiveId) {
            return primitive.get().initializeSteps(rayOrigin, rayDirection, rayTimestamp, rayMinMax, rndSeed, parameters, instanceId);
        } else {
            return initializePrimitiveInstanceSteps<PrimitiveId + 1>(primitive.next(), rayOrigin, rayDirection, rayTimestamp, rayMinMax, rndSeed, parameters, primitiveId, instanceId);
        }
    }

    // helper functions to call step on a given primitiveInstance
    template <uint8_t PrimitiveId = 0, typename TPrimitive = typename TPrimitivesTuple::THead, typename nreEnableIf<(PrimitiveId >= TParams::NumPrimitives), bool>::type = true>
    static inline __device__ tcnn::vec2 stepPrimitiveInstance(TPrimitive, const tcnn::vec3&, const tcnn::vec3&, const tcnn::vec2&, uint32_t&, nrend::MemoryHandles, uint8_t, uint16_t) {
        __builtin_unreachable();
        return tcnn::vec2::zero();
    }
    template <uint8_t PrimitiveId = 0, typename TPrimitive = typename TPrimitivesTuple::THead, typename nreEnableIf<(PrimitiveId < TParams::NumPrimitives), bool>::type = true>
    static inline __device__ tcnn::vec2 stepPrimitiveInstance(TPrimitive primitive, const tcnn::vec3& rayOrigin, const tcnn::vec3& rayDirection, const tcnn::vec2& rayMinMax, uint32_t& rndSeed, nrend::MemoryHandles parameters, uint8_t primitiveId, uint16_t instanceId) {

        if (primitiveId == PrimitiveId) {
            return primitive.get().step(rayOrigin, rayDirection, rayMinMax, rndSeed, parameters, instanceId);
        } else {
            return stepPrimitiveInstance<PrimitiveId + 1>(primitive.next(), rayOrigin, rayDirection, rayMinMax, rndSeed, parameters, primitiveId, instanceId);
        }
    }

    // helper functions to call evalAt on a given primitiveInstance
    template <uint8_t PrimitiveId = 0, typename TPrimitive = typename TPrimitivesTuple::THead, typename TEvalAppearanceEmbedding = TAppearanceEmbedding, typename nreEnableIf<(PrimitiveId >= TParams::NumPrimitives), bool>::type = true>
    static inline __device__ tcnn::vec4 evalPrimitiveInstanceAt(TPrimitive, const tcnn::vec3&, const tcnn::vec3&, float, const TEvalAppearanceEmbedding&, nrend::MemoryHandles, uint8_t, uint16_t) {
        __builtin_unreachable();
        return tcnn::vec4::zero();
    }
    template <uint8_t PrimitiveId = 0, typename TPrimitive = typename TPrimitivesTuple::THead, typename TEvalAppearanceEmbedding = TAppearanceEmbedding, typename nreEnableIf<(PrimitiveId < TParams::NumPrimitives), bool>::type = true>
    static inline __device__ tcnn::vec4 evalPrimitiveInstanceAt(TPrimitive primitive, const tcnn::vec3& position, const tcnn::vec3& direction, nrend::TTimestamp timeStamp, const TEvalAppearanceEmbedding& embedding, nrend::MemoryHandles parameters, uint8_t primitiveId, uint16_t instanceId) {

        if (primitiveId == PrimitiveId) {
            return primitive.get().evalAt(position, direction, timeStamp, instanceId, embedding, parameters);
        } else {
            return evalPrimitiveInstanceAt<PrimitiveId + 1>(primitive.next(), position, direction, timeStamp, embedding, parameters, primitiveId, instanceId);
        }
    }

    inline __device__ void insertPrimitiveInstanceHit(PrimitiveInstanceHit& hit) {

#pragma unroll
        for (uint16_t i = 0; i < PrimitiveInstancesBatchSize; ++i) {

            if (hit.stepHitDistance < m_primitiveInstanceHits[i].stepHitDistance) {
                PrimitiveInstanceHit swapHit = m_primitiveInstanceHits[i];
                m_primitiveInstanceHits[i]   = hit;
                hit                          = swapHit;
            }
        }
    }

    static inline __device__ tcnn::quat vec4ToQuat(const tcnn::vec4& xyzw) {
        return tcnn::quat(xyzw.w, xyzw.x, xyzw.y, xyzw.z);
    }

    inline __device__ void fetchPrimitiveInstanceTransform(nrend::TTimestamp startTs,
                                                           nrend::TTimestamp endTs,
                                                           const nrend::TTrackInstancePose& startPose,
                                                           const nrend::TTrackInstancePose& endPose,
                                                           nrend::TTimestamp ts,
                                                           tcnn::mat3& rotation,
                                                           tcnn::vec3& position) {

        const float alpha = endTs > startTs ? static_cast<float>(ts - startTs) / static_cast<float>(endTs - startTs) : 1.0f;

        const tcnn::vec3 startPosition   = startPose.slice<0, 3>();
        const tcnn::quat startQuaternion = vec4ToQuat(startPose.slice<3, 4>());
        const tcnn::vec3 endPosition     = endPose.slice<0, 3>();
        const tcnn::quat endQuaternion   = vec4ToQuat(endPose.slice<3, 4>());

        position = tcnn::mix(startPosition, endPosition, alpha);
        rotation = tcnn::transpose(tcnn::to_mat3(tcnn::slerp(startQuaternion, endQuaternion, alpha)));
    }

    inline __device__ void fetchPrimitiveInstanceTransform(const nrend::TTrackInstancePose& pose,
                                                           tcnn::mat3& rotation,
                                                           tcnn::vec3& position) {

        position = pose.slice<0, 3>();
        rotation = tcnn::transpose(tcnn::to_mat3(vec4ToQuat(pose.slice<3, 4>())));
    }

    inline __device__ void resetPrimitiveInstanceHit(nrend::TTimestamp rayTimestamp,
                                                     const tcnn::vec2& rayMinMax,
                                                     uint32_t& rndSeed,
                                                     nrend::MemoryHandles parameters,
                                                     PrimitiveInstanceHit& primitiveInstanceHit) {

        // compute the intersection with the instance extent
        const tcnn::vec2 stepHitBounds = initializePrimitiveInstanceSteps(m_primitives,
                                                                          primitiveInstanceHit.localOrigin,
                                                                          primitiveInstanceHit.localDirection,
                                                                          rayTimestamp,
                                                                          rayMinMax,
                                                                          rndSeed,
                                                                          parameters,
                                                                          primitiveInstanceHit.primitiveId(),
                                                                          primitiveInstanceHit.instanceId());

        const tcnn::vec2 stepHit = stepPrimitiveInstance(m_primitives,
                                                         primitiveInstanceHit.localOrigin,
                                                         primitiveInstanceHit.localDirection,
                                                         stepHitBounds,
                                                         rndSeed,
                                                         parameters,
                                                         primitiveInstanceHit.primitiveId(),
                                                         primitiveInstanceHit.instanceId());

        primitiveInstanceHit.stepHitMaxDistance = stepHitBounds.y;
        primitiveInstanceHit.stepHitDistance    = stepHit.x >= stepHitBounds.y ? MaxHitDistance : stepHit.x;
        primitiveInstanceHit.stepHitDelta       = stepHit.x >= stepHitBounds.y ? 0.0f : stepHit.y;

        // insertion sort
        insertPrimitiveInstanceHit(primitiveInstanceHit);

        // primitiveInstanceHit contains the last removed hit
        m_nextPrimitiveInstanceMinStepHitDistance = fminf(primitiveInstanceHit.stepHitDistance, m_nextPrimitiveInstanceMinStepHitDistance);
    }

    inline __device__ void resetPrimitiveInstanceHits(const tcnn::vec3& rayOrigin,
                                                      const tcnn::vec3& rayDirection,
                                                      nrend::TTimestamp rayTimestamp,
                                                      const tcnn::vec2& rayMinMax,
                                                      uint32_t& rndSeed,
                                                      nrend::MemoryHandles parameters,
                                                      nrend::TTimestamp startTimestamp,
                                                      nrend::TTimestamp endTimestamp,
                                                      int numActiveTrackInstances,
                                                      const tcnn::ivec2* __restrict__ trackInstancesIds,
                                                      const nrend::TTrackInstancePose* __restrict__ trackInstanceStartPoseCudaPtr,
                                                      const nrend::TTrackInstancePose* __restrict__ trackInstanceEndPoseCudaPtr) {

        m_nextPrimitiveInstanceMinStepHitDistance = MaxHitDistance;

#pragma unroll
        for (uint8_t i = 0; i < TParams::NumBackgroundPrimitives; ++i) {

            PrimitiveInstanceHit primitiveInstanceHit;
            primitiveInstanceHit.packedIdx      = PrimitiveInstanceHit::primitiveIdToPackedIdx(i);
            primitiveInstanceHit.localDirection = rayDirection;
            primitiveInstanceHit.localOrigin    = rayOrigin;

            resetPrimitiveInstanceHit(rayTimestamp, rayMinMax, rndSeed, parameters, primitiveInstanceHit);
        }

        for (uint8_t i = 0; i < numActiveTrackInstances; ++i) {

            const tcnn::ivec2 trackMappingIds = trackInstancesIds ? trackInstancesIds[i] : tcnn::ivec2{-1, -1};
            if ((trackMappingIds.x < 0) || (trackMappingIds.x >= TParams::NumPrimitiveInstances)) {
                continue;
            }

            PrimitiveInstanceHit primitiveInstanceHit;
            primitiveInstanceHit.packedIdx = TParams::primitiveInstancesPackedIdx[trackMappingIds.x];
            if (primitiveInstanceHit.packedIdx == PrimitiveInstanceHit::InvalidPackedIdx) {
                continue;
            }
            primitiveInstanceHit.activeTrackIdx = trackMappingIds.y;

            // fetch the interpolated instance transform for the current ray
            tcnn::mat3 instanceRotation;
            tcnn::vec3 instancePosition;

            if (trackInstanceStartPoseCudaPtr && trackInstanceEndPoseCudaPtr) {
                fetchPrimitiveInstanceTransform(startTimestamp,
                                                endTimestamp,
                                                trackInstanceStartPoseCudaPtr[i],
                                                trackInstanceEndPoseCudaPtr[i],
                                                rayTimestamp,
                                                instanceRotation,
                                                instancePosition);
            } else {
                fetchPrimitiveInstanceTransform(trackInstanceStartPoseCudaPtr ? trackInstanceStartPoseCudaPtr[i] : trackInstanceEndPoseCudaPtr[i],
                                                instanceRotation,
                                                instancePosition);
            }

            primitiveInstanceHit.localDirection = instanceRotation * rayDirection;
            primitiveInstanceHit.localOrigin    = instanceRotation * (rayOrigin - instancePosition);

            resetPrimitiveInstanceHit(rayTimestamp, rayMinMax, rndSeed, parameters, primitiveInstanceHit);
        }
    }

    inline __device__ tcnn::vec2 step(const tcnn::vec3& rayOrigin,
                                      const tcnn::vec3& rayDirection,
                                      nrend::TTimestamp rayTimestamp,
                                      const tcnn::vec2& rayMinMax,
                                      uint32_t& rndSeed,
                                      nrend::MemoryHandles parameters,
                                      nrend::TTimestamp startTimestamp,
                                      nrend::TTimestamp endTimestamp,
                                      int numActiveTrackInstances,
                                      const tcnn::ivec2* __restrict__ trackInstancesIds,
                                      const nrend::TTrackInstancePose* __restrict__ trackInstanceStartPoseCudaPtr,
                                      const nrend::TTrackInstancePose* __restrict__ trackInstanceEndPoseCudaPtr) {

        // first entry hit contains the previous closest hit : step the first entry
        // NB : has to be done before potential reset (after reset the first entry hit is the currently closest)
        if (m_primitiveInstanceHits[0].stepHitDistance < MaxHitDistance) {

            // step closest primitive instance to the next hit
            const tcnn::vec2 nextStepHit = stepPrimitiveInstance(m_primitives,
                                                                 m_primitiveInstanceHits[0].localOrigin,
                                                                 m_primitiveInstanceHits[0].localDirection,
                                                                 // FIXME NRE : in theory we should step from the current position (rayMinMax.x)
                                                                 tcnn::vec2{
                                                                     m_primitiveInstanceHits[0].stepHitDistance + m_primitiveInstanceHits[0].stepHitDelta,
                                                                     m_primitiveInstanceHits[0].stepHitMaxDistance},
                                                                 rndSeed,
                                                                 parameters,
                                                                 m_primitiveInstanceHits[0].primitiveId(),
                                                                 m_primitiveInstanceHits[0].instanceId());

            // not allowed to march farther than the current closest, out of batch, primitive instance (m_nextPrimitiveInstanceMinStepHitDistance)
            const float stepHitMaxDistance = fminf(m_primitiveInstanceHits[0].stepHitMaxDistance, m_nextPrimitiveInstanceMinStepHitDistance);

            m_primitiveInstanceHits[0].stepHitDistance = nextStepHit.x >= stepHitMaxDistance ? MaxHitDistance : nextStepHit.x;
            m_primitiveInstanceHits[0].stepHitDelta    = nextStepHit.x >= MaxHitDistance ? 0.0f : nextStepHit.y;

#pragma unroll
            for (uint16_t i = 1; i < PrimitiveInstancesBatchSize; ++i) {

                if (m_primitiveInstanceHits[i].stepHitDistance < m_primitiveInstanceHits[i - 1].stepHitDistance) {
                    PrimitiveInstanceHit swapHit   = m_primitiveInstanceHits[i];
                    m_primitiveInstanceHits[i]     = m_primitiveInstanceHits[i - 1];
                    m_primitiveInstanceHits[i - 1] = swapHit;
                }
            }
        }

        // reset the current batch of primitiveInstanceHits (always during the first call to step)
        else if (m_nextPrimitiveInstanceMinStepHitDistance < MaxHitDistance) {

            resetPrimitiveInstanceHits(rayOrigin,
                                       rayDirection,
                                       rayTimestamp,
                                       rayMinMax,
                                       rndSeed, parameters,
                                       startTimestamp,
                                       endTimestamp,
                                       numActiveTrackInstances,
                                       trackInstancesIds,
                                       trackInstanceStartPoseCudaPtr,
                                       trackInstanceEndPoseCudaPtr);
        }

        // first entry hit contains the current closest
        // FIXME : in theory we should step up-to the next closest (setting delta as min(delta, next closest - current closest)
        return tcnn::vec2{m_primitiveInstanceHits[0].stepHitDistance, m_primitiveInstanceHits[0].stepHitDelta};
    }

    template <typename TEvalAppearanceEmbedding>
    inline tcnn::vec4 __device__ evalAt(const tcnn::vec3& position,
                                        const tcnn::vec3& direction,
                                        nrend::TTimestamp timestamp,
                                        const TEvalAppearanceEmbedding& embedding,
                                        nrend::MemoryHandles parameters,
                                        uint8_t primitiveId,
                                        uint16_t instanceId) {

        return evalPrimitiveInstanceAt(m_primitives,
                                       position,
                                       direction,
                                       timestamp,
                                       embedding,
                                       parameters,
                                       primitiveId,
                                       instanceId);
    }

    template <typename TEvalAppearanceEmbedding>
    inline tcnn::vec3 __device__ evalAtInfinity(const tcnn::vec3& direction,
                                                nrend::TTimestamp timestamp,
                                                const TEvalAppearanceEmbedding& embedding,
                                                nrend::MemoryHandles parameters) {

        return m_background.eval(direction, timestamp, embedding, parameters);
    }

    inline void __device__ march(const nrend::RenderParameters& params,
                                 TRayPayload& ray,
                                 nrend::MemoryHandles parameters,
                                 const tcnn::ivec2* __restrict__ trackInstancesIds,
                                 const nrend::TTrackInstancePose* __restrict__ trackInstanceStartPoseCudaPtr,
                                 const nrend::TTrackInstancePose* __restrict__ trackInstanceEndPoseCudaPtr,
                                 const tcnn::ivec2* __restrict__ sensorsIdsPtr) {

        m_appearanceEmbedding.eval(ray.timestamp, parameters, ray.idx, sensorsIdsPtr);

        bool skipStep = false;
        tcnn::vec2 stepHit;

        while (true) {

            if (ray.isAlive() && !skipStep) {
                stepHit = step(ray.origin,
                               ray.direction,
                               ray.timestamp,
                               ray.tMinMax,
                               ray.rndSeed,
                               parameters,
                               params.sensorState.startTimestamp,
                               params.sensorState.endTimestamp,
                               params.numActiveTrackInstances,
                               trackInstancesIds,
                               trackInstanceStartPoseCudaPtr,
                               trackInstanceEndPoseCudaPtr);
                if (stepHit.x >= ray.tMinMax.y) {
                    ray.kill();
                }
            }

            if (__all_sync(WarpMask, !ray.isAlive())) {
                break;
            }

            // synchronize the warp on the same primitive : the primitive of the lane with the smallest hit distance
            int evaluatedPrimitiveId = m_primitiveInstanceHits[0].primitiveId();
            int warpPrimitiveSynchronized;
            float closestHitDistance = ray.isAlive() ? m_primitiveInstanceHits[0].stepHitDistance : MaxHitDistance;
            if (!__match_all_sync(WarpMask, evaluatedPrimitiveId, &warpPrimitiveSynchronized)) {
                // butterfly min reduction over the hit distance
#pragma unroll
                for (int mask = 1; mask < WarpSize; mask *= 2) {
                    const float bfClosestHitDistance = __shfl_xor_sync(WarpMask, closestHitDistance, mask);
                    const int bfPrimitiveId          = __shfl_xor_sync(WarpMask, evaluatedPrimitiveId, mask);
                    if ((bfClosestHitDistance < closestHitDistance) || ((bfClosestHitDistance == closestHitDistance) && (bfPrimitiveId < evaluatedPrimitiveId))) {
                        closestHitDistance   = bfClosestHitDistance;
                        evaluatedPrimitiveId = bfPrimitiveId;
                    }
                }
            }
            skipStep = m_primitiveInstanceHits[0].primitiveId() != evaluatedPrimitiveId;

            // evaluate all thread at the same primitive id
            // (NB : inputs are wrong for threads which closest primitive is not the evaluated one, result is ignored anyway)
            const tcnn::vec4 radianceDensity = evalAt(
                m_primitiveInstanceHits[0].localOrigin + m_primitiveInstanceHits[0].stepHitDistance * m_primitiveInstanceHits[0].localDirection,
                m_primitiveInstanceHits[0].localDirection,
                ray.timestamp,
                m_appearanceEmbedding,
                parameters,
                evaluatedPrimitiveId,
                m_primitiveInstanceHits[0].instanceId());

            // skip computation of dead threads or threads which evaluated primitive is not the closest
            if (!ray.isAlive() || skipStep) {
                continue;
            }

            const float transmittance = __expf(-radianceDensity.w * stepHit.y);
            const float weight        = (1.0f - transmittance) * ray.transmittance;

            ray.features.vec += weight * radianceDensity.xyz();
            ray.hitT += weight * stepHit.x;
            ray.transmittance *= transmittance;
            ray.tMinMax.x = stepHit.x + stepHit.y;
            if (ray.transmittance < TParams::minTransmittance) {
                ray.kill();
            }
        }

        // mark the ray as front hit if the traversed volume is sufficiently opaque
        if (ray.isValid() && (ray.transmittance < params.hitTransmittance)) {
            ray.hitFront();
        }

        // process the ray background hits (need all threads to be active for potential coopvec operations)
        const bool needBackEvaluation = ray.isValid() && !ray.hasBackHit();
        if (__any_sync(0xFFFFFFFF, needBackEvaluation)) {
            const tcnn::vec3 backgroundRadiance = evalAtInfinity(ray.direction, ray.timestamp, m_appearanceEmbedding, parameters);
            if (needBackEvaluation) {
                ray.features.vec += ray.transmittance * backgroundRadiance;
                ray.transmittance = 0.0f; ///< background has been hit, set the transmittance to 0
            }
        }

        // post process the ray output
        if (ray.isValid()) {
            TPostProcessings::eval(ray, params, parameters, sensorsIdsPtr);
        }
    }
};
