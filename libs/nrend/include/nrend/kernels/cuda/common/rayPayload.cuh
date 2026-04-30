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

#include <nrend/kernels/cuda/common/nreColorUtils.cuh>
#include <nrend/kernels/cuda/common/random.cuh>
#include <nrend/kernels/cuda/sensors/sensorsRay.cuh>
#include <nrend/renderer/renderParameters.h>
#include <nrend/utils/nreVec.h>

template <int BaseFeatN = 3, int ExtFeatN = 0, bool THasNormals = false>
struct RayPayload {
    static constexpr uint32_t BaseFeatDim       = BaseFeatN;
    static constexpr bool HasBaseFeatures       = BaseFeatN > 0;
    static constexpr uint32_t ExtFeatDim        = ExtFeatN;
    static constexpr uint32_t FeatDim           = BaseFeatDim + ExtFeatDim;
    static constexpr bool HasFeatures           = FeatDim > 0;
    static constexpr bool HasNormals            = THasNormals;
    static constexpr uint32_t InvalidRayIdx     = -1U;
    static constexpr uint32_t InvalidInstanceId = 0;

    nrend::TTimestamp timestamp;
    tcnn::vec3 origin;
    tcnn::vec3 direction;
    float spread; // spread of the ray cone (diameter at unit distance)
    tcnn::vec2 tMinMax;
    float hitT;
    nrend::OptionalVec<3, float, HasNormals> normal;
    float transmittance;
    enum {
        Default             = 0,
        Valid               = 1 << 0,
        Alive               = 1 << 2,
        BackHit             = 1 << 3,
        BackHitProxySurface = 1 << 4,
        FrontHit            = 1 << 5
    };
    uint32_t flags;
    uint32_t idx;
    uint32_t rndSeed;
    nrend::OptionalVec<FeatDim, float, HasFeatures> features;

    __device__ __inline__ bool isAlive() const {
        return flags & Alive;
    }
    __device__ __inline__ void kill() {
        flags &= ~Alive;
    }
    __device__ __inline__ bool isValid() const {
        return flags & Valid;
    }
    __device__ __inline__ bool hasBackHit() const {
        return flags & BackHit;
    }
    __device__ __inline__ bool isFrontHit() const {
        return flags & FrontHit;
    }
    __device__ __inline__ void hitFront() {
        flags |= FrontHit;
    }
};

template <typename RayPayloadT, bool ThreadSensorRayIdx = false>
__device__ __inline__ RayPayloadT initializeRay(const nrend::RenderParameters& params,
                                                const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                                                const tcnn::vec3* __restrict__ worldRayDirectionPtr,
                                                const nrend::TTimestamp* __restrict__ worldRayTimestampPtr,
                                                const uint32_t* __restrict__ instanceIdPtr,
                                                const float* __restrict__ worldHitDistancePtr,
                                                const tcnn::vec2* __restrict__ worldHitDistanceBoundsPtr = nullptr) {
    RayPayloadT ray;
    ray.flags = RayPayloadT::Default;

    const tcnn::ivec2 uv = tcnn::ivec2{static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x),
                                       static_cast<int>(threadIdx.y + blockDim.y * blockIdx.y)};
    if constexpr (ThreadSensorRayIdx) {
        ray.idx = nrend::threadSensorRayIdx<RayPayloadT::InvalidRayIdx>(params.sensorModel, params.frameTileResolution, uv, params.frameTileOffset);
        if (ray.idx == RayPayloadT::InvalidRayIdx) {
            return ray;
        }
    } else {
        if ((uv.x >= params.frameTileResolution.x) || (uv.y >= params.frameTileResolution.y)) {
            return ray;
        }
        ray.idx = uv.x + params.frameTileResolution.x * uv.y;
    }
    ray.hitT          = 0.0f;
    ray.transmittance = 1.0f;
    if constexpr (RayPayloadT::HasNormals) {
        ray.normal.vec = tcnn::vec3(0.0f);
    }
    ray.rndSeed = tea(params.id, ray.idx * 786433);
    if constexpr (RayPayloadT::HasFeatures) {
        ray.features.vec = tcnn::vec<RayPayloadT::FeatDim>::zero();
    }

    ray.timestamp = worldRayTimestampPtr ? worldRayTimestampPtr[ray.idx] : (params.sensorState.startTimestamp + (params.sensorState.endTimestamp - params.sensorState.startTimestamp) / 2);
    ray.origin    = params.worldToObjectTransform * tcnn::vec4(wordlRayOriginPtr[ray.idx], 1.0f);
    ray.direction = worldRayDirectionPtr[ray.idx];
    ray.spread    = tcnn::length(ray.direction);
    if (ray.spread < 1e-6f) {
        return ray;
    }
    ray.direction            = tcnn::mat3(params.worldToObjectTransform) * ray.direction;
    float worldToObjectScale = tcnn::length(ray.direction);
    if (worldToObjectScale > 1e-06f) {
        ray.direction /= worldToObjectScale;
        worldToObjectScale /= ray.spread;
    }

    ray.tMinMax   = params.objectAABB.ray_intersect(ray.origin, ray.direction);
    ray.tMinMax.x = fmaxf(ray.tMinMax.x, 0.0f);
    if (worldHitDistanceBoundsPtr) {
        const tcnn::vec2 objectHitDistanceBounds = worldHitDistanceBoundsPtr[ray.idx] * worldToObjectScale;
        ray.tMinMax.x                            = fmaxf(ray.tMinMax.x, objectHitDistanceBounds.x);
        ray.tMinMax.y                            = fminf(ray.tMinMax.y, objectHitDistanceBounds.y);
    }
    if (ray.tMinMax.y > ray.tMinMax.x) {
        const uint32_t instanceId = instanceIdPtr ? instanceIdPtr[ray.idx] : RayPayloadT::InvalidInstanceId;
        // we have a valid instance id <=> we have a valid world hit distance (BackHit)
        if (instanceId != RayPayloadT::InvalidInstanceId) {
            // ignore back hit on the given proxy objects
            const bool backHitProxySurface = (params.objectInstanceIds.x == instanceId) ||
                                             (params.objectInstanceIds.y == instanceId) ||
                                             (params.objectInstanceIds.z == instanceId) ||
                                             (params.objectInstanceIds.w == instanceId);
            if (backHitProxySurface) {
                ray.flags |= RayPayloadT::BackHitProxySurface;
            } else {
                ray.flags |= RayPayloadT::BackHit;
                ray.tMinMax.y = fminf(ray.tMinMax.y, worldToObjectScale * worldHitDistancePtr[ray.idx]);
            }
        }
    }

    if (ray.tMinMax.y > ray.tMinMax.x) {
        ray.flags |= RayPayloadT::Valid | RayPayloadT::Alive;
    }

    return ray;
}

template <bool ColorTransformSRGBInput, bool ColorTransformSRGBOutput, bool Differentiable, typename TRayPayload>
__device__ __inline__ void finalizeRay(const TRayPayload& ray,
                                       const nrend::RenderParameters& params,
                                       const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                                       uint32_t* __restrict__ instanceIdPtr,
                                       float* __restrict__ worldHitDistancePtr,
                                       tcnn::vec3* __restrict__ worldHitNormalPtr,
                                       tcnn::vec<TRayPayload::BaseFeatDim + 1>* __restrict__ radianceDensityPtr,
                                       void* __restrict__ extendedFeaturesPtr             = nullptr,
                                       tcnn::vec2* __restrict__ worldHitDistanceBoundsPtr = nullptr) {
    if (!ray.isValid()) {
        return;
    }

    if constexpr (TRayPayload::BaseFeatDim > 0) {
        static_assert(TRayPayload::BaseFeatDim == 3, "Only RGB radiance is supported for now");
        if constexpr (Differentiable) {
            radianceDensityPtr[ray.idx].xyz() = ray.features.vec;
        } else {
            // radiance is alpha pre-multiplied : unmultiply to apply transforms
            tcnn::vec<3> rayRadiance = nrend::sliceVec<0, 3>(ray.features.vec); // / tcnn::max(1.f - ray.transmittance, 1e-08f);
            if (ColorTransformSRGBInput) {
                rayRadiance = tcnn::vec3{srgbToLinear(rayRadiance[0]), srgbToLinear(rayRadiance[1]), srgbToLinear(rayRadiance[2])};
            }
            rayRadiance = params.colorCorrectionMatrix * tcnn::vec4(rayRadiance, 1.0f);
            if (ColorTransformSRGBOutput) {
                rayRadiance =
                    tcnn::vec3{linearToSrgb(rayRadiance[0]), linearToSrgb(rayRadiance[1]), linearToSrgb(rayRadiance[2])};
            }
            radianceDensityPtr[ray.idx].xyz() = radianceDensityPtr[ray.idx].xyz() * ray.transmittance + rayRadiance; // tcnn::mix(rayRadiance, radianceDensityPtr[ray.idx].xyz(), ray.transmittance);
        }
    }

    if constexpr (TRayPayload::ExtFeatDim > 0) {
        if constexpr (Differentiable) {
            static_cast<tcnn::vec<TRayPayload::ExtFeatDim>*>(extendedFeaturesPtr)[ray.idx] = nrend::sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.features.vec);
        } else {
            static_cast<tcnn::vec<TRayPayload::ExtFeatDim>*>(extendedFeaturesPtr)[ray.idx] =
                static_cast<tcnn::vec<TRayPayload::ExtFeatDim>*>(extendedFeaturesPtr)[ray.idx] * ray.transmittance +
                nrend::sliceVec<TRayPayload::BaseFeatDim, TRayPayload::ExtFeatDim>(ray.features.vec);
        }
    }

    if constexpr (Differentiable) {
        radianceDensityPtr[ray.idx][TRayPayload::BaseFeatDim] = 1.0f - ray.transmittance;
    } else {
        // alpha = front_alpha + back_alpha * (1 - front_alpha) = back_alpha + front_alpha * (1 - back_alpha)
        radianceDensityPtr[ray.idx][TRayPayload::BaseFeatDim] +=
            (1.0f - ray.transmittance) * (1.0f - radianceDensityPtr[ray.idx][TRayPayload::BaseFeatDim]);
    }

    // FIXME : depth compositing with proxy surfaces is ill-defined
    //         (worldHitDistancePtr is set to the proxy surface distance)
    if (ray.isFrontHit() || (ray.flags & TRayPayload::BackHitProxySurface)) {
        const tcnn::vec3 objectTarget = ray.origin + ray.direction * ray.hitT;
        const float worldHitDistance  = tcnn::length(params.objectToWorldTransform * tcnn::vec4(objectTarget, 1.0f) - wordlRayOriginPtr[ray.idx]);
        worldHitDistancePtr[ray.idx]  = worldHitDistance;
        if (worldHitDistanceBoundsPtr) {
            const float objectToWorldScale       = worldHitDistance / ray.hitT;
            worldHitDistanceBoundsPtr[ray.idx].x = ray.tMinMax.x * objectToWorldScale;
            worldHitDistanceBoundsPtr[ray.idx].y = ray.tMinMax.y * objectToWorldScale;
        }
        if (instanceIdPtr && !(ray.flags & TRayPayload::BackHitProxySurface) && (params.objectInstanceIds.x != TRayPayload::InvalidInstanceId)) {
            // set the instance id as the first associated object
            instanceIdPtr[ray.idx] = params.objectInstanceIds.x;
        }

        if constexpr (TRayPayload::HasNormals) {
            if (worldHitNormalPtr) {
                // store world-space normal; transform because ray.normal is in object space
                // => equivalent to transpose(inverse(objectToWorld)) * normal
                const tcnn::vec3 worldNormal = nrend::mul<float, 3, 3>(ray.normal.vec, tcnn::mat3(params.worldToObjectTransform));
                if constexpr (Differentiable) {
                    worldHitNormalPtr[ray.idx] = worldNormal;
                } else {
                    worldHitNormalPtr[ray.idx] = worldHitNormalPtr[ray.idx] * ray.transmittance + worldNormal;
                }
            }
        }
    }
}
