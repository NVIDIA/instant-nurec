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

#include <nrend/renderer/renderParameters.h>

template <typename TParticles>
struct NREShGaussian {

    using Particles            = TParticles;
    using DensityRawParameters = typename Particles::DensityRawParameters;

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
                                       const tcnn::ivec2* __restrict__ /*sensorsIdsPtr*/) {
        if (ray.isValid()) {

            // mark the ray as front hit if the traversed volume is sufficiently opaque
            if (ray.transmittance < params.hitTransmittance) {
                ray.hitFront();
            }
        }
    }

    template <typename TRay>
    static inline __device__ void evalBackward(const nrend::RenderParameters& params,
                                               TRay& ray,
                                               nrend::MemoryHandles parameters,
                                               nrend::MemoryHandles parametersGradient,
                                               const tcnn::ivec2* __restrict__ /*sensorsIdsPtr*/) {
        // NOOP
    }

    static inline __device__ void deform(nrend::TTimestamp /*timestamp*/,
                                         uint16_t /*instanceId*/,
                                         nrend::MemoryHandles /*parameters*/,
                                         DensityRawParameters& /*densityParams*/) {
        // NOOP : static gaussians do not deform
    }
};