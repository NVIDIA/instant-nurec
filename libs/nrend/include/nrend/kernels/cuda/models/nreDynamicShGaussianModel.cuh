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

#include <nrend/kernels/cuda/models/nreInstancesExtent.cuh>
#include <nrend/kernels/cuda/models/nreShGaussianModel.cuh>

#include <nrend/utils/nreVec.h>

struct NREDeformableShGaussianParams {
    static constexpr bool UseDeformNetwork            = false;
    static constexpr bool DeformPositions             = false;
    static constexpr bool DeformRotations             = false;
    static constexpr bool DeformRotationsFromIdentity = false;
    static constexpr bool DeformScales                = false;
};

template <typename TParticles, typename TInstanceExtent, typename TDeformNetwork, typename TParams>
struct NREDeformableShGaussian {

    using Particles            = TParticles;
    using DensityRawParameters = typename Particles::DensityRawParameters;

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

    static inline __device__ void deform(nrend::TTimestamp timestamp,
                                         uint16_t instanceId,
                                         nrend::MemoryHandles parameters,
                                         DensityRawParameters& densityParams) {
        if constexpr (TParams::UseDeformNetwork) {
            const tcnn::vec3 normalizedPos = TInstanceExtent::contract_position(densityParams.position, instanceId);
            const auto deformation         = TDeformNetwork().eval(normalizedPos, timestamp, instanceId, parameters);
            if constexpr (TParams::DeformPositions) {
                constexpr int kOffset = 0;
                densityParams.position += tcnn::vec3(nrend::sliceVec<kOffset, 3>(deformation));
            }
            if constexpr (TParams::DeformRotations) {
                constexpr int kOffset = TParams::DeformPositions ? 3 : 0;
                tcnn::quat deformQuat = tcnn::quat{deformation[3 + kOffset], deformation[0 + kOffset], deformation[1 + kOffset], deformation[2 + kOffset]};
                if constexpr (TParams::DeformRotationsFromIdentity) {
                    deformQuat.w += 1.0f;
                }
                const tcnn::quat deformedQuaternion = nrend::mul(
                    // Note : the quaternion deformation is in xyzw format
                    tcnn::normalize(deformQuat),
                    tcnn::quat{densityParams.quaternion[0], densityParams.quaternion[1], densityParams.quaternion[2], densityParams.quaternion[3]});
                densityParams.quaternion = tcnn::vec4{deformedQuaternion.w, deformedQuaternion.x, deformedQuaternion.y, deformedQuaternion.z};
            }
            if constexpr (TParams::DeformScales) {
                constexpr int kOffset = (TParams::DeformPositions ? 3 : 0) + (TParams::DeformRotations ? 4 : 0);
                // TODO : generic scale activation
                densityParams.scale = densityParams.scale * tcnn::exp(tcnn::vec3(nrend::sliceVec<kOffset, 3>(deformation)));
            }
        }
    }
};
