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

#include <nrend/kernels/cuda/models/nreAppearanceEmbedding.cuh>
#include <nrend/renderer/renderParameters.h>

template <typename TParticles,
          typename TAppearanceEmbedding,
          typename TBackground,
          bool EnableBackground,
          typename TPostProcessings,
          bool EnablePostProcessings,
          bool SaturateRadiance>
struct NREGaussiansPrimitive {

    using Particles = TParticles;

    static inline __device__ void preprocess(uint32_t /*numParticles*/,
                                             uint32_t /*numActiveTrackInstances*/,
                                             const tcnn::ivec2* /*activeTrackInstancesIdsCudaPtr*/,
                                             nrend::TTimestamp /*timestamp*/,
                                             const nrend::TTrackInstancePose* /*activeTrackInstancesPoseCudaPtr*/,
                                             const nrend::TTrackInstancePose* /*activeTrackInstancesEndPoseCudaPtr*/,
                                             nrend::MemoryHandles /*parameterMemoryHandles*/) {
        // NOOP
    }

    template <typename TRay>
    static inline __device__ void eval(const nrend::RenderParameters& params,
                                       TRay& ray,
                                       nrend::MemoryHandles parameters,
                                       const tcnn::ivec2* __restrict__ sensorsIdsPtr) {

        // mark the ray as front hit if the traversed volume is sufficiently opaque
        if (ray.isValid() && (ray.transmittance < params.hitTransmittance)) {
            ray.hitFront();
        }

        // process the ray background hits (need all threads to be active for potential coopvec operations)
        if constexpr (TRay::HasBaseFeatures && TBackground::Enabled && EnableBackground) {
            const bool needBackEvaluation = ray.isValid() && !ray.hasBackHit();
            bool evalBackground           = needBackEvaluation;
            if constexpr (TBackground::RequireThreadSync) {
                evalBackground = __any_sync(0xFFFFFFFF, needBackEvaluation);
            }
            if (evalBackground) {
                TAppearanceEmbedding appearanceEmbedding;
                appearanceEmbedding.eval(ray.timestamp, parameters, ray.idx, sensorsIdsPtr);
                const tcnn::vec<TRay::FeatDim> backFeatures = TBackground().eval(ray.direction, ray.timestamp, appearanceEmbedding, parameters);
                if (needBackEvaluation) {
                    ray.features.vec += ray.transmittance * backFeatures;
                    ray.transmittance = 0.0f; ///< background has been hit, set the transmittance to 0
                }
            }
        }

        // post process the ray output
        if constexpr (TRay::HasBaseFeatures && EnablePostProcessings) {
            if (ray.isValid()) {
                TPostProcessings::eval(ray, params, parameters, sensorsIdsPtr);
            }
        }

        if constexpr (TRay::HasBaseFeatures && SaturateRadiance) {
            if (ray.isValid()) {
                nrend::sliceVec<0, TRay::BaseFeatDim>(ray.features.vec) =
                    tcnn::clamp(nrend::sliceVec<0, TRay::BaseFeatDim>(ray.features.vec), 0.f, 1.f);
            }
        }
    }

    template <typename TRay>
    static inline __device__ void evalBackward(const nrend::RenderParameters& params,
                                               TRay& ray,
                                               nrend::MemoryHandles parameters,
                                               nrend::MemoryHandles parametersGradient,
                                               const tcnn::ivec2* __restrict__ /*sensorsIdsPtr*/) {
        // NB : Gaussian primitives are not differentiated through NRend
    }
};
