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

template <int BaseFeatN         = 3,
          int ExtFeatN          = 0,
          bool HasNormals       = false,
          bool THasRayGradients = false>
struct RayPayloadBackward : public RayPayload<BaseFeatN, ExtFeatN, HasNormals> {
    static constexpr bool HasFeatures     = BaseFeatN + ExtFeatN > 0;
    static constexpr bool HasRayGradients = THasRayGradients;

    float transmittanceBackward;
    float hitTGradient;
    float transmittanceGradient;
    nrend::OptionalVec<3, float, HasNormals> normalGradient;
    nrend::OptionalVec<BaseFeatN + ExtFeatN, float, HasFeatures> featuresGradient;
    nrend::OptionalVec<3, float, HasRayGradients> originGradient;    ///< output ray origin gradient
    nrend::OptionalVec<3, float, HasRayGradients> directionGradient; ///< output ray direction gradient
};

template <typename RayPayloadT, bool ThreadSensorRayIdx = false>
__device__ __inline__ RayPayloadT initializeBackwardRay(const nrend::RenderParameters& params,
                                                        const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                                                        const tcnn::vec3* __restrict__ worldRayDirectionPtr,
                                                        const nrend::TTimestamp* __restrict__ worldRayTimestampPtr,
                                                        const uint32_t* __restrict__ instanceIdPtr,
                                                        const float* __restrict__ worldHitDistancePtr,
                                                        const float* __restrict__ worldHitDistanceGradientPtr,
                                                        const tcnn::vec3* __restrict__ worldHitNormalPtr,
                                                        const tcnn::vec3* __restrict__ worldHitNormalGradientPtr,
                                                        const tcnn::vec<RayPayloadT::BaseFeatDim + 1>* __restrict__ featuresDensityPtr,
                                                        const tcnn::vec<RayPayloadT::BaseFeatDim + 1>* __restrict__ featuresDensityGradientPtr,
                                                        const void* __restrict__ extendedFeaturesPtr,
                                                        const void* __restrict__ extendedFeaturesGradientPtr,
                                                        const tcnn::vec2* __restrict__ worldHitDistanceBoundsPtr = nullptr) {

    // NB : no backpropagation through the forward ray initialization / finalization
    RayPayloadT ray = initializeRay<RayPayloadT, ThreadSensorRayIdx>(params,
                                                                     wordlRayOriginPtr,
                                                                     worldRayDirectionPtr,
                                                                     worldRayTimestampPtr,
                                                                     instanceIdPtr,
                                                                     worldHitDistancePtr,
                                                                     worldHitDistanceBoundsPtr);

    if (ray.isAlive()) {
        ray.hitT                                                      = worldHitDistancePtr[ray.idx];
        const tcnn::vec<RayPayloadT::BaseFeatDim + 1> featuresDensity = featuresDensityPtr[ray.idx];
        ray.transmittanceBackward                                     = 1.f - featuresDensity[RayPayloadT::BaseFeatDim];
        if constexpr (RayPayloadT::BaseFeatDim > 0) {
            nrend::sliceVec<0, RayPayloadT::BaseFeatDim>(ray.features.vec) =
                nrend::sliceVec<0, RayPayloadT::BaseFeatDim>(featuresDensity);
        }
        if constexpr (RayPayloadT::ExtFeatDim > 0) {
            nrend::sliceVec<RayPayloadT::BaseFeatDim, RayPayloadT::ExtFeatDim>(ray.features.vec) =
                static_cast<const tcnn::vec<RayPayloadT::ExtFeatDim>*>(extendedFeaturesPtr)[ray.idx];
        }
        if constexpr (RayPayloadT::HasNormals) {
            ray.normal.vec = worldHitNormalPtr[ray.idx];
        }

        ray.hitTGradient                                                  = worldHitDistanceGradientPtr[ray.idx];
        const tcnn::vec<RayPayloadT::FeatDim + 1> featuresDensityGradient = featuresDensityGradientPtr[ray.idx];
        ray.transmittanceGradient                                         = -1.f * featuresDensityGradient[RayPayloadT::BaseFeatDim];
        if constexpr (RayPayloadT::BaseFeatDim > 0) {
            nrend::sliceVec<0, RayPayloadT::BaseFeatDim>(ray.featuresGradient.vec) =
                nrend::sliceVec<0, RayPayloadT::BaseFeatDim>(featuresDensityGradient);
        }
        if constexpr (RayPayloadT::ExtFeatDim > 0) {
            nrend::sliceVec<RayPayloadT::BaseFeatDim, RayPayloadT::ExtFeatDim>(ray.featuresGradient.vec) =
                static_cast<const tcnn::vec<RayPayloadT::ExtFeatDim>*>(extendedFeaturesGradientPtr)[ray.idx];
        }
        if constexpr (RayPayloadT::HasNormals) {
            ray.normalGradient.vec = worldHitNormalGradientPtr[ray.idx];
        }

        if constexpr (RayPayloadT::HasRayGradients) {
            ray.originGradient.vec    = tcnn::vec3::zero();
            ray.directionGradient.vec = tcnn::vec3::zero();
        }
    }

    return ray;
}

template <typename RayPayloadT>
__device__ __inline__ void finalizeBackwardRay(RayPayloadT& ray,
                                               const nrend::RenderParameters&,
                                               tcnn::vec3* __restrict__ wordlRayOriginGradientPtr,
                                               tcnn::vec3* __restrict__ worldRayDirectionGradientPtr) {
    if constexpr (RayPayloadT::HasRayGradients) {
        if (ray.isValid()) {
            if (wordlRayOriginGradientPtr) {
                wordlRayOriginGradientPtr[ray.idx] += ray.originGradient.vec;
            }
            if (worldRayDirectionGradientPtr) {
                // input ray direction is multiplied by the ray spread (non-zero)
                worldRayDirectionGradientPtr[ray.idx] += ray.directionGradient.vec / ray.spread;
            }
        }
    }
}
