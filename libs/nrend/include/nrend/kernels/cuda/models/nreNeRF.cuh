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

#include <nrend/kernels/cuda/common/rayPayload.cuh>

struct NRENeRFDefaultParams {
    static constexpr float minTransmittance = nrend::RenderParameters::defaultMinTransmittance;
};

template <typename TAccelerationStructure,
          typename TFeatureVolume,
          typename TGeometry,
          typename TTexture,
          typename TAppearanceEmbedding,
          typename TBackground,
          typename TPostProcessings,
          typename TParams = NRENeRFDefaultParams>
struct NRENeRF {
    using TRayPayload = RayPayload<3>;

    static constexpr uint16_t MarchDummyInstanceId = 0;

    TAccelerationStructure m_accelerationStructure;
    TFeatureVolume m_featureVolume;
    TGeometry m_geometry;
    TTexture m_texture;
    TAppearanceEmbedding m_appearanceEmbedding;
    TBackground m_background;

    inline __device__ tcnn::vec2 initializeSteps(const tcnn::vec3& rayOrigin,
                                                 const tcnn::vec3& rayDirection,
                                                 nrend::TTimestamp rayTimestamp,
                                                 const tcnn::vec2& rayMinMax,
                                                 uint32_t& rndSeed,
                                                 nrend::MemoryHandles parameters,
                                                 uint16_t instanceId = 0) {

        return m_accelerationStructure.initialize(rayOrigin, rayDirection, rayTimestamp, rayMinMax, rndSeed, parameters, instanceId);
    }

    inline __device__ tcnn::vec2 step(const tcnn::vec3& rayOrigin,
                                      const tcnn::vec3& rayDirection,
                                      const tcnn::vec2& rayMinMax,
                                      uint32_t& rndSeed,
                                      nrend::MemoryHandles parameters,
                                      uint8_t instanceId = 0) {

        return m_accelerationStructure.step(rayOrigin, rayDirection, rayMinMax, rndSeed, parameters);
    }

    template <typename TEvalAppearanceEmbedding>
    inline __device__ tcnn::vec4 evalAt(const tcnn::vec3& position,
                                        const tcnn::vec3& direction,
                                        nrend::TTimestamp timestamp,
                                        uint16_t instanceId,
                                        const TEvalAppearanceEmbedding& embedding,
                                        nrend::MemoryHandles parameters) {

        const tcnn::vec4 normalizedPos = m_accelerationStructure.contract_position(position, instanceId);
        const auto features            = m_featureVolume.eval(normalizedPos.xyz(), timestamp, instanceId, parameters);
        tcnn::vec4 radianceDensity;
        radianceDensity.w     = m_geometry.eval(features, normalizedPos.w);
        radianceDensity.xyz() = m_texture.eval(features, embedding, normalizedPos, direction, parameters);
        return radianceDensity;
    }

    template <typename TEvalAppearanceEmbedding>
    inline __device__ tcnn::vec3 evalAtInfinity(const tcnn::vec3& direction,
                                                nrend::TTimestamp timestamp,
                                                const TEvalAppearanceEmbedding& embedding,
                                                nrend::MemoryHandles parameters) {

        return m_background.eval(direction, timestamp, embedding, parameters);
    }

    inline __device__ void march(const nrend::RenderParameters& params,
                                 TRayPayload& ray,
                                 nrend::MemoryHandles parameters,
                                 const tcnn::ivec2* __restrict__ /*trackInstancesIds*/,
                                 const nrend::TTrackInstancePose* __restrict__ /*trackInstancesStartPoseCudaPtr*/,
                                 const nrend::TTrackInstancePose* __restrict__ /*trackInstancesEndPoseCudaPtr*/,
                                 const tcnn::ivec2* __restrict__ sensorsIdsPtr) {

        m_appearanceEmbedding.eval(ray.timestamp, parameters, ray.idx, sensorsIdsPtr);

        ray.tMinMax = initializeSteps(ray.origin, ray.direction, ray.timestamp, ray.tMinMax, ray.rndSeed, parameters);

        while (true) {

            float stepDt = 0.0f;

            if (ray.isAlive()) {
                const tcnn::vec2 stepHit = step(ray.origin, ray.direction, ray.tMinMax, ray.rndSeed, parameters);
                ray.tMinMax.x            = stepHit.x;
                stepDt                   = stepHit.y;
                if (ray.tMinMax.x >= ray.tMinMax.y) {
                    ray.kill();
                }
            }

            if (__all_sync(0xFFFFFFFF, !ray.isAlive())) {
                break;
            }

            const tcnn::vec4 radianceDensity = evalAt(
                ray.origin + ray.direction * ray.tMinMax.x,
                ray.direction,
                ray.timestamp,
                MarchDummyInstanceId,
                m_appearanceEmbedding,
                parameters);

            // All threads in the warp must execute the above MLPs for coherence reasons.
            // Starting from here, it's fine to skip computation.
            if (!ray.isAlive()) {
                continue;
            }

            const float transmittance = __expf(-radianceDensity.w * stepDt);
            const float weight        = (1.0f - transmittance) * ray.transmittance;

            ray.features.vec += weight * radianceDensity.xyz();
            ray.hitT += weight * ray.tMinMax.x;
            ray.transmittance *= transmittance;
            ray.tMinMax.x += stepDt;

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
