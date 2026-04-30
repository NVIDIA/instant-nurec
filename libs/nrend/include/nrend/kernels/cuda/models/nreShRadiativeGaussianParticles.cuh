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

struct NREShRadiativeGaussianParticlesInternalDummyParams {
    static constexpr int DensityRawParametersBufferIndex                  = -1;
    static constexpr int DensityRawParametersGradientBufferIndex          = -1;
    static constexpr int FeaturesRawParametersBufferIndex                 = -1;
    static constexpr int FeaturesRawParametersGradientBufferIndex         = -1;
    static constexpr int ExtendedFeaturesRawParametersBufferIndex         = -1;
    static constexpr int ExtendedFeaturesRawParametersGradientBufferIndex = -1;
    static constexpr int GlobalParametersValueBufferIndex                 = -1;
    static constexpr int FeatureShDegreeValueOffset                       = -1;
};

template <typename TBuffer, bool TDifferentiable>
struct NREShRadiativeGaussianParticlesBuffer {
    TBuffer* ptr = nullptr;
};

template <typename TBuffer>
struct NREShRadiativeGaussianParticlesBuffer<TBuffer, true> {
    TBuffer* ptr     = nullptr;
    TBuffer* gradPtr = nullptr;
};

template <typename TBuffer, bool TDifferentiable, bool Enabled>
struct NREShRadiativeGaussianParticlesOptionalBuffer {
};

template <typename TBuffer, bool TDifferentiable>
struct NREShRadiativeGaussianParticlesOptionalBuffer<TBuffer, TDifferentiable, true> : NREShRadiativeGaussianParticlesBuffer<TBuffer, TDifferentiable> {
};

struct NREShRadiativeGaussianParticlesExternalDummyParams {
    static constexpr int FeaturesDim                 = 0;
    static constexpr int FeaturesParametersDim       = 0;
    static constexpr float AlphaThreshold            = .0f;
    static constexpr float MinTransmittanceThreshold = nrend::RenderParameters::defaultMinTransmittance;
};

template <typename TDensityRawParameters,
          typename TDensityParameters,
          typename TFeaturesParameters,
          typename TFeaturesType = float,
          typename Params        = NREShRadiativeGaussianParticlesInternalDummyParams,
          typename ExtParams     = NREShRadiativeGaussianParticlesExternalDummyParams,
          bool TDifferentiable   = true>
struct NREShRadiativeGaussianVolumetricFeaturesParticles : Params, public ExtParams {

    constexpr static uint32_t WarpSize = 32;
    constexpr static uint32_t WarpMask = 0xFFFFFFFF;

    struct DensityRawParameters {
        tcnn::vec3 position;
        float density;
        tcnn::vec4 quaternion;
        tcnn::vec3 scale;
        float padding;
    };
    using DensityParameters = TDensityParameters;

    __forceinline__ __device__ void initializeDensity(nrend::MemoryHandles parameters) {
        static_assert(sizeof(DensityRawParameters) == sizeof(TDensityRawParameters), "DensityRawParameters size mismatch");
        m_densityRawParameters.ptr =
            parameters.bufferPtr<TDensityRawParameters>(Params::DensityRawParametersBufferIndex);
    }

    __forceinline__ __device__ void initializeDensityGradient(nrend::MemoryHandles parametersGradient) {
        if constexpr (TDifferentiable) {
            m_densityRawParameters.gradPtr =
                parametersGradient.bufferPtr<TDensityRawParameters>(Params::DensityRawParametersGradientBufferIndex);
        }
    };

    __forceinline__ __device__ DensityRawParameters fetchDensityRawParameters(uint32_t particleIdx) const {
        static_assert(sizeof(DensityRawParameters) == sizeof(TDensityRawParameters), "DensityRawParameters size mismatch");
        TDensityRawParameters densityRawParameters = particleDensityFetchParametersRaw(particleIdx,
                                                                                       {{m_densityRawParameters.ptr, nullptr, false}});
        return reinterpret_cast<const DensityRawParameters&>(densityRawParameters);
    }

    __forceinline__ __device__ void fetchDensityRawParametersGradBwd(uint32_t particleIdx,
                                                                     const DensityRawParameters& densityRawParameters) const {
        if constexpr (TDifferentiable) {
            DensityRawParameters* densityRawParametersGrad = reinterpret_cast<DensityRawParameters*>(m_densityRawParameters.gradPtr);
            atomicAdd(&densityRawParametersGrad[particleIdx].position.x, densityRawParameters.position.x);
            atomicAdd(&densityRawParametersGrad[particleIdx].position.y, densityRawParameters.position.y);
            atomicAdd(&densityRawParametersGrad[particleIdx].position.z, densityRawParameters.position.z);
            atomicAdd(&densityRawParametersGrad[particleIdx].density, densityRawParameters.density);
            atomicAdd(&densityRawParametersGrad[particleIdx].quaternion.x, densityRawParameters.quaternion.x);
            atomicAdd(&densityRawParametersGrad[particleIdx].quaternion.y, densityRawParameters.quaternion.y);
            atomicAdd(&densityRawParametersGrad[particleIdx].quaternion.z, densityRawParameters.quaternion.z);
            atomicAdd(&densityRawParametersGrad[particleIdx].quaternion.w, densityRawParameters.quaternion.w);
            atomicAdd(&densityRawParametersGrad[particleIdx].scale.x, densityRawParameters.scale.x);
            atomicAdd(&densityRawParametersGrad[particleIdx].scale.y, densityRawParameters.scale.y);
            atomicAdd(&densityRawParametersGrad[particleIdx].scale.z, densityRawParameters.scale.z);
        }
    }

    template <bool exclusiveGradient>
    __forceinline__ __device__ void fetchDensityRawParametersBwd(uint32_t particleIdx, const DensityRawParameters& densityRawParameters) {
        if constexpr (TDifferentiable) {
            particleDensityFetchParametersRawBwd(particleIdx,
                                                 {{m_densityRawParameters.ptr, m_densityRawParameters.gradPtr, exclusiveGradient}},
                                                 reinterpret_cast<const TDensityRawParameters&>(densityRawParameters));
        }
    }

    __forceinline__ __device__ DensityParameters fetchDensityParameters(uint32_t particleIdx) const {
        return particleDensityParameters(particleIdx, {m_densityRawParameters.ptr, nullptr});
    }

    __forceinline__ __device__ tcnn::vec3 fetchPosition(uint32_t particleIdx) const {
        return *(reinterpret_cast<const tcnn::vec3*>(&m_densityRawParameters.ptr[particleIdx].position_1));
    }

    __forceinline__ __device__ float fetchDensity(uint32_t particleIdx) const {
        return m_densityRawParameters.ptr[particleIdx].density_1;
    }

    static __forceinline__ __device__ DensityParameters densityParametersFromRaw(const DensityRawParameters& densityRawParameters) {

        return particleDensityParametersFromRaw(
            reinterpret_cast<const TDensityRawParameters&>(densityRawParameters));
    }

    static __forceinline__ __device__ const tcnn::vec3& position(const DensityParameters& parameters) {
        return *(reinterpret_cast<const tcnn::vec3*>(&parameters.position_0));
    }

    static __forceinline__ __device__ const tcnn::vec3& scale(const DensityParameters& parameters) {
        return *(reinterpret_cast<const tcnn::vec3*>(&parameters.scale_0));
    }

    static __forceinline__ __device__ const tcnn::mat3& rotation(const DensityParameters& parameters) {
        // slang uses row-major order (tcnn uses column-major order), so we return the rotation (not transposed)
        return *(reinterpret_cast<const tcnn::mat3*>(&parameters.rotationT_0));
    }

    static __forceinline__ __device__ const float& opacity(const DensityParameters& parameters) {
        return parameters.density_0;
    }

    static __forceinline__ __device__ float canonicalScale(const DensityParameters& parameters, bool densityScaleClamping) {
        return particleCanonicalScale(parameters, densityScaleClamping);
    }

    static __forceinline__ __device__ const tcnn::vec3 fromCanonical(const DensityParameters& parameters,
                                                                     const tcnn::vec3& point) {
        const auto canonicalPoint = particleFromCanonical(parameters, *reinterpret_cast<const float3*>(&point));
        return *reinterpret_cast<const tcnn::vec3*>(&canonicalPoint);
    }

    template <bool KnownHitDistance = false, bool KnownHitAlpha = false>
    static __forceinline__ __device__ bool densityHit(const tcnn::vec3& rayOrigin,
                                                      const tcnn::vec3& rayDirection,
                                                      float raySpread,
                                                      const DensityParameters& parameters,
                                                      float& alpha,
                                                      float& hitDistance,
                                                      tcnn::vec3* normal = nullptr) {
        if constexpr (KnownHitDistance && KnownHitAlpha) {
            if (normal != nullptr) {
                return particleDensityHitNormal(*reinterpret_cast<const float3*>(&rayOrigin),
                                                *reinterpret_cast<const float3*>(&rayDirection),
                                                raySpread,
                                                parameters,
                                                alpha,
                                                hitDistance,
                                                normal != nullptr,
                                                reinterpret_cast<float3*>(normal));
            }
            return true;
        } else if constexpr (KnownHitDistance) {
            return particleDensityHitAlpha(*reinterpret_cast<const float3*>(&rayOrigin),
                                           *reinterpret_cast<const float3*>(&rayDirection),
                                           raySpread,
                                           parameters,
                                           &alpha,
                                           &hitDistance,
                                           normal != nullptr,
                                           reinterpret_cast<float3*>(normal));
        } else {
            // No specific handling of case KnownHitAlpha && !KnownHitDistance
            return particleDensityHit(*reinterpret_cast<const float3*>(&rayOrigin),
                                      *reinterpret_cast<const float3*>(&rayDirection),
                                      raySpread,
                                      parameters,
                                      &alpha,
                                      &hitDistance,
                                      normal != nullptr,
                                      reinterpret_cast<float3*>(normal));
        }
    }

    template <bool Sampling = false>
    static __forceinline__ __device__ float densityIntegrateHit(float alpha,
                                                                float& transmittance,
                                                                float hitDistance,
                                                                float& integratedhitDistance,
                                                                const tcnn::vec3* normal     = nullptr,
                                                                tcnn::vec3* integratedNormal = nullptr) {

        if constexpr (Sampling) {
            return particleDensityIntegrateSampleHit(alpha,
                                                     &transmittance,
                                                     hitDistance,
                                                     &integratedhitDistance,
                                                     normal != nullptr,
                                                     normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(normal),
                                                     reinterpret_cast<float3*>(integratedNormal));
        } else {
            return particleDensityIntegrateHit(alpha,
                                               &transmittance,
                                               hitDistance,
                                               &integratedhitDistance,
                                               normal != nullptr,
                                               normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(normal),
                                               reinterpret_cast<float3*>(integratedNormal));
        }
    }

    __forceinline__ __device__ float densityProcessHitFwdFromBuffer(const tcnn::vec3& rayOrigin,
                                                                    const tcnn::vec3& rayDirection,
                                                                    float raySpread,
                                                                    uint32_t particleIdx,
                                                                    float& transmittance,
                                                                    float& integratedhitDistance,
                                                                    tcnn::vec3* integratedNormal = nullptr) const {
        return particleDensityProcessHitFwdFromBuffer(*reinterpret_cast<const float3*>(&rayOrigin),
                                                      *reinterpret_cast<const float3*>(&rayDirection),
                                                      raySpread,
                                                      particleIdx,
                                                      {{m_densityRawParameters.ptr, nullptr, true}},
                                                      &transmittance,
                                                      &integratedhitDistance,
                                                      integratedNormal != nullptr,
                                                      reinterpret_cast<float3*>(integratedNormal));
    }

    template <bool exclusiveGradient, bool Sampling = false>
    __forceinline__ __device__ void densityProcessHitBwdToBuffer(const tcnn::vec3& rayOrigin,
                                                                 tcnn::vec3* rayOriginGradient,
                                                                 const tcnn::vec3& rayDirection,
                                                                 tcnn::vec3* rayDirectionGradient,
                                                                 float raySpread,
                                                                 uint32_t particleIdx,
                                                                 float alpha,
                                                                 float alphaGrad,
                                                                 float& transmittance,
                                                                 float& transmittanceGrad,
                                                                 float hitDistance,
                                                                 float& integratedhitDistance,
                                                                 float& integratedhitDistanceGrad,
                                                                 const tcnn::vec3* normal         = nullptr,
                                                                 tcnn::vec3* integratedNormal     = nullptr,
                                                                 tcnn::vec3* integratedNormalGrad = nullptr

    ) const {
        if constexpr (TDifferentiable) {
            if constexpr (Sampling) {
                particleDensityProcessSampleHitBwdToBuffer(*reinterpret_cast<const float3*>(&rayOrigin),
                                                           *reinterpret_cast<const float3*>(&rayDirection),
                                                           rayOriginGradient != nullptr && rayDirectionGradient != nullptr,
                                                           reinterpret_cast<float3*>(rayOriginGradient),
                                                           reinterpret_cast<float3*>(rayDirectionGradient),
                                                           raySpread,
                                                           particleIdx,
                                                           {{m_densityRawParameters.ptr, m_densityRawParameters.gradPtr, exclusiveGradient}},
                                                           alpha,
                                                           alphaGrad,
                                                           &transmittance,
                                                           &transmittanceGrad,
                                                           hitDistance,
                                                           &integratedhitDistance,
                                                           &integratedhitDistanceGrad,
                                                           normal != nullptr,
                                                           normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(normal),
                                                           reinterpret_cast<float3*>(integratedNormal),
                                                           reinterpret_cast<float3*>(integratedNormalGrad));
            } else {
                particleDensityProcessHitBwdToBuffer(*reinterpret_cast<const float3*>(&rayOrigin),
                                                     *reinterpret_cast<const float3*>(&rayDirection),
                                                     rayOriginGradient != nullptr && rayDirectionGradient != nullptr,
                                                     reinterpret_cast<float3*>(rayOriginGradient),
                                                     reinterpret_cast<float3*>(rayDirectionGradient),
                                                     raySpread,
                                                     particleIdx,
                                                     {{m_densityRawParameters.ptr, m_densityRawParameters.gradPtr, exclusiveGradient}},
                                                     alpha,
                                                     alphaGrad,
                                                     &transmittance,
                                                     &transmittanceGrad,
                                                     hitDistance,
                                                     &integratedhitDistance,
                                                     &integratedhitDistanceGrad,
                                                     normal != nullptr,
                                                     normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(normal),
                                                     reinterpret_cast<float3*>(integratedNormal),
                                                     reinterpret_cast<float3*>(integratedNormalGrad));
            }
        }
    }

    static __forceinline__ __device__ void densityProcessHitBwdToRawParameters(const tcnn::vec3& rayOrigin,
                                                                               tcnn::vec3* rayOriginGradient,
                                                                               const tcnn::vec3& rayDirection,
                                                                               tcnn::vec3* rayDirectionGradient,
                                                                               float raySpread,
                                                                               uint32_t particleIdx,
                                                                               float alpha,
                                                                               float alphaGrad,
                                                                               float& transmittance,
                                                                               float& transmittanceGrad,
                                                                               float hitDistance,
                                                                               float& integratedhitDistance,
                                                                               float& integratedhitDistanceGrad,
                                                                               const DensityParameters& densityParameters,
                                                                               const tcnn::vec4& quaternion,
                                                                               DensityRawParameters& densityRawParametersGrad,
                                                                               const tcnn::vec3* normal         = nullptr,
                                                                               tcnn::vec3* integratedNormal     = nullptr,
                                                                               tcnn::vec3* integratedNormalGrad = nullptr

    ) {
        if constexpr (TDifferentiable) {
            particleDensityProcessHitBwdToRawParameters(*reinterpret_cast<const float3*>(&rayOrigin),
                                                        *reinterpret_cast<const float3*>(&rayDirection),
                                                        rayOriginGradient != nullptr && rayDirectionGradient != nullptr,
                                                        reinterpret_cast<float3*>(rayOriginGradient),
                                                        reinterpret_cast<float3*>(rayDirectionGradient),
                                                        raySpread,
                                                        particleIdx,
                                                        densityParameters,
                                                        *reinterpret_cast<const float4*>(&quaternion),
                                                        reinterpret_cast<TDensityRawParameters*>(&densityRawParametersGrad),
                                                        alpha,
                                                        alphaGrad,
                                                        &transmittance,
                                                        &transmittanceGrad,
                                                        hitDistance,
                                                        &integratedhitDistance,
                                                        &integratedhitDistanceGrad,
                                                        normal != nullptr,
                                                        normal == nullptr ? make_float3(0, 0, 0) : *reinterpret_cast<const float3*>(normal),
                                                        reinterpret_cast<float3*>(integratedNormal),
                                                        reinterpret_cast<float3*>(integratedNormalGrad));
        }
    }

    template <bool useAtomicAdd = true>
    __forceinline__ __device__ void processHitBwdUpdateDensityGradient(uint32_t particleIdx, DensityRawParameters& densityRawParameters, uint32_t tileThreadIdx) {
        if constexpr (TDifferentiable) {
#pragma unroll
            for (int offset = WarpSize >> 1; offset > 0; offset >>= 1) {
                densityRawParameters.position.x += __shfl_down_sync(WarpMask, densityRawParameters.position.x, offset);
                densityRawParameters.position.y += __shfl_down_sync(WarpMask, densityRawParameters.position.y, offset);
                densityRawParameters.position.z += __shfl_down_sync(WarpMask, densityRawParameters.position.z, offset);
                densityRawParameters.density += __shfl_down_sync(WarpMask, densityRawParameters.density, offset);
                densityRawParameters.quaternion.x += __shfl_down_sync(WarpMask, densityRawParameters.quaternion.x, offset);
                densityRawParameters.quaternion.y += __shfl_down_sync(WarpMask, densityRawParameters.quaternion.y, offset);
                densityRawParameters.quaternion.z += __shfl_down_sync(WarpMask, densityRawParameters.quaternion.z, offset);
                densityRawParameters.quaternion.w += __shfl_down_sync(WarpMask, densityRawParameters.quaternion.w, offset);
                densityRawParameters.scale.x += __shfl_down_sync(WarpMask, densityRawParameters.scale.x, offset);
                densityRawParameters.scale.y += __shfl_down_sync(WarpMask, densityRawParameters.scale.y, offset);
                densityRawParameters.scale.z += __shfl_down_sync(WarpMask, densityRawParameters.scale.z, offset);
            }

            // First thread in the warp performs the atomic add
            if constexpr (useAtomicAdd) {
                if ((tileThreadIdx & (WarpSize - 1)) == 0) {
                    fetchDensityRawParametersGradBwd(particleIdx, densityRawParameters);
                }
            }
        }
    }

    template <typename TVec3>
    __forceinline__ __device__ bool densityRayHit(const TVec3& rayOrigin,
                                                  const TVec3& rayDirection,
                                                  uint32_t particleIdx,
                                                  float minHitDistance,
                                                  float maxHitDistance,
                                                  float& hitDistance,
                                                  tcnn::vec3* normal = nullptr) const {
        return particleDensityRayHit(*reinterpret_cast<const float3*>(&rayOrigin),
                                     *reinterpret_cast<const float3*>(&rayDirection),
                                     particleIdx,
                                     {{m_densityRawParameters.ptr, nullptr, true}},
                                     minHitDistance,
                                     maxHitDistance,
                                     &hitDistance,
                                     normal != nullptr,
                                     reinterpret_cast<float3*>(normal));
    }

    template <typename TVec3>
    static __forceinline__ __device__ bool densityCanonicalRayHitDistance(const TVec3& canonicalRayOrigin,
                                                                          const TVec3& canonicalUnormalizedRayDirection,
                                                                          float minHitDistance,
                                                                          float maxHitDistance,
                                                                          float& hitDistance

    ) {
        return particleDensityCanonicalRayHitDistance(*reinterpret_cast<const float3*>(&canonicalRayOrigin),
                                                      *reinterpret_cast<const float3*>(&canonicalUnormalizedRayDirection),
                                                      minHitDistance,
                                                      maxHitDistance,
                                                      &hitDistance);
    }

    template <typename TVec3>
    static __forceinline__ __device__ bool densityCanonicalRayHitAlpha(const TVec3& canonicalRayOrigin,
                                                                       const TVec3& canonicalUnormalizedRayDirection,
                                                                       float minHitDistance,
                                                                       float maxHitDistance,
                                                                       float& alpha,
                                                                       float& hitDistance,
                                                                       float particleDensity,
                                                                       bool densityScaleClamping

    ) {
        return particleDensityCanonicalRayHitAlpha(*reinterpret_cast<const float3*>(&canonicalRayOrigin),
                                                   *reinterpret_cast<const float3*>(&canonicalUnormalizedRayDirection),
                                                   minHitDistance,
                                                   maxHitDistance,
                                                   &alpha,
                                                   &hitDistance,
                                                   particleDensity,
                                                   densityScaleClamping);
    }

    static __forceinline__ __device__ tcnn::vec3 densityIncidentDirection(const DensityParameters& parameters,
                                                                          const tcnn::vec3& sourcePosition)

    {
        const auto incidentDirection = particleDensityIncidentDirection(parameters, *reinterpret_cast<const float3*>(&sourcePosition));
        return *reinterpret_cast<const tcnn::vec3*>(&incidentDirection);
    }

    template <bool exclusiveGradient>
    __forceinline__ __device__ void densityIncidentDirectionBwdToBuffer(uint32_t particlesIdx,
                                                                        const tcnn::vec3& sourcePosition,
                                                                        const tcnn::vec3& incidedentDirectionGrad)

    {
        if constexpr (TDifferentiable) {
            particleDensityIncidentDirectionBwdToBuffer(particlesIdx,
                                                        {{m_densityRawParameters.ptr, m_densityRawParameters.gradPtr, exclusiveGradient}},
                                                        *reinterpret_cast<const float3*>(&sourcePosition),
                                                        *reinterpret_cast<const float3*>(&incidedentDirectionGrad));
        }
    }

    static __forceinline__ __device__ bool densityPerspectiveConicProjection(const DensityParameters& parameters,
                                                                             const tcnn::vec2& resolution,
                                                                             const tcnn::vec2& nearFarClipDistances,
                                                                             const tcnn::vec2& focalLength,
                                                                             const tcnn::vec2& principalPoint,
                                                                             const tcnn::mat4x3& sensorView,
                                                                             const tcnn::vec3& sensorPosition,
                                                                             tcnn::vec3& incidentRay,
                                                                             tcnn::vec2& projectedCenter,
                                                                             tcnn::vec3& projectedCovariance,
                                                                             float& projectedOpacity

    ) {
        return particleDensityPerspectiveConicProjection(parameters,
                                                         *reinterpret_cast<const float2*>(&resolution),
                                                         *reinterpret_cast<const float2*>(&nearFarClipDistances),
                                                         *reinterpret_cast<const float2*>(&focalLength),
                                                         *reinterpret_cast<const float2*>(&principalPoint),
                                                         // slang matrices are row-major (tcnn uses column-major) : sensorView transposed
                                                         *reinterpret_cast<const Matrix<float, 4 /*rows*/, 3 /*columns*/>*>(&sensorView),
                                                         *reinterpret_cast<const float3*>(&sensorPosition),
                                                         reinterpret_cast<float3*>(&incidentRay),
                                                         reinterpret_cast<float2*>(&projectedCenter),
                                                         reinterpret_cast<float3*>(&projectedCovariance),
                                                         &projectedOpacity);
    }

    static __forceinline__ __device__ bool densityConicHit(const tcnn::vec2& rayProjectedPosition,
                                                           const tcnn::vec2& projectedCenter,
                                                           const tcnn::vec4& projectedConicOpacity,
                                                           float& alpha) {
        return particleDensityConicHit(
            *reinterpret_cast<const float2*>(&rayProjectedPosition),
            *reinterpret_cast<const float2*>(&projectedCenter),
            *reinterpret_cast<const float4*>(&projectedConicOpacity),
            &alpha);
    }

    static __forceinline__ __device__ void densityProcessHitBwdToConic(const tcnn::vec2& rayProjectedPosition,
                                                                       float alpha,
                                                                       float alphaGrad,
                                                                       float& transmittance,
                                                                       float& transmittanceGrad,
                                                                       float hitDistance,
                                                                       float& hitDistanceGrad,
                                                                       float& integratedhitDistance,
                                                                       float& integratedhitDistanceGrad,
                                                                       const tcnn::vec2& projectedPosition,
                                                                       tcnn::vec2& projectedPositionGrad,
                                                                       const tcnn::vec4& projectedConicOpacity,
                                                                       tcnn::vec4& projectedConicOpacityGrad)

    {
        if constexpr (TDifferentiable) {
            particleDensityProcessHitBwdToConic(
                *reinterpret_cast<const float2*>(&rayProjectedPosition),
                alpha,
                alphaGrad,
                &transmittance,
                &transmittanceGrad,
                hitDistance,
                &hitDistanceGrad,
                &integratedhitDistance,
                &integratedhitDistanceGrad,
                *reinterpret_cast<const float2*>(&projectedPosition),
                reinterpret_cast<float2*>(&projectedPositionGrad),
                *reinterpret_cast<const float4*>(&projectedConicOpacity),
                reinterpret_cast<float4*>(&projectedConicOpacityGrad));
        }
    }

    template <bool exclusiveGradient>
    __forceinline__ __device__ void densityPerspectiveConicProjectionBwdToBuffer(uint32_t particleIdx,
                                                                                 const tcnn::vec2& resolution,
                                                                                 const tcnn::vec2& nearFarClipDistances,
                                                                                 const tcnn::vec2& focalLength,
                                                                                 const tcnn::vec2& principalPoint,
                                                                                 const tcnn::mat4x3& sensorView,
                                                                                 const tcnn::vec3& sensorPosition,
                                                                                 float covarianceDilation,
                                                                                 const tcnn::vec2& projectedPositionGrad,
                                                                                 const tcnn::vec4& projectedConicOpacityGrad,
                                                                                 float hitDistanceGrad) const {
        if constexpr (TDifferentiable) {
            particleDensityPerspectiveConicProjectionBwdToBuffer(
                particleIdx,
                {{m_densityRawParameters.ptr, m_densityRawParameters.gradPtr, exclusiveGradient}},
                *reinterpret_cast<const float2*>(&resolution),
                *reinterpret_cast<const float2*>(&nearFarClipDistances),
                *reinterpret_cast<const float2*>(&focalLength),
                *reinterpret_cast<const float2*>(&principalPoint),
                // slang matrices are row-major (tcnn uses column-major) : sensorView transposed
                *reinterpret_cast<const Matrix<float, 4 /*rows*/, 3 /*columns*/>*>(&sensorView),
                *reinterpret_cast<const float3*>(&sensorPosition),
                covarianceDilation,
                *reinterpret_cast<const float2*>(&projectedPositionGrad),
                *reinterpret_cast<const float4*>(&projectedConicOpacityGrad),
                hitDistanceGrad);
        }
    }

    using FeaturesRawParameters = typename tcnn::tvec<TFeaturesType, ExtParams::FeaturesParametersDim>;
    using TFeaturesVec          = typename tcnn::vec<ExtParams::FeaturesDim>;

    inline __device__ void initializeFeatures(nrend::MemoryHandles parameters) {
        static_assert(ExtParams::FeaturesDim == 3, "Hardcoded 3-dimensional radiance because of Slang-Cuda interop");
        m_featureRawParameters.ptr = parameters.bufferPtr<TFeaturesType>(Params::FeaturesRawParametersBufferIndex);
        m_featureActiveShDegree    = *reinterpret_cast<int*>(parameters.bufferPtr<uint8_t>(Params::GlobalParametersValueBufferIndex) + Params::FeatureShDegreeValueOffset);
    };

    inline __device__ void initializeFeaturesGradient(nrend::MemoryHandles parametersGradient) {
        if constexpr (TDifferentiable) {
            m_featureRawParameters.gradPtr = parametersGradient.bufferPtr<TFeaturesType>(Params::FeaturesRawParametersGradientBufferIndex);
        }
    };

    __forceinline__ __device__ FeaturesRawParameters featuresRawParametersFromBuffer(uint32_t particleIdx) const {
        return reinterpret_cast<const FeaturesRawParameters*>(m_featureRawParameters.ptr)[particleIdx];
    }

    __forceinline__ __device__ TFeaturesVec featuresFromBuffer(uint32_t particleIdx,
                                                               const tcnn::vec3& incidentDirection) const {
        const auto features = particleFeaturesFromBuffer(particleIdx,
                                                         {{m_featureRawParameters.ptr, nullptr, true}, m_featureActiveShDegree},
                                                         *reinterpret_cast<const float3*>(&incidentDirection));
        return *reinterpret_cast<const TFeaturesVec*>(&features);
    }

    template <bool exclusiveGradient>
    __forceinline__ __device__ void featuresBwdToBuffer(uint32_t particleIdx,
                                                        const TFeaturesVec& features,
                                                        const TFeaturesVec& featuresGrad,
                                                        const tcnn::vec3& incidentDirection,
                                                        tcnn::vec3& incidentDirectionGrad) const {
        if constexpr (TDifferentiable) {
            particleFeaturesBwdToBuffer(particleIdx,
                                        {{m_featureRawParameters.ptr, m_featureRawParameters.gradPtr, exclusiveGradient}, m_featureActiveShDegree},
                                        *reinterpret_cast<const float3*>(&featuresGrad),
                                        *reinterpret_cast<const float3*>(&incidentDirection),
                                        reinterpret_cast<float3*>(&incidentDirectionGrad));
        }
    }

    static __forceinline__ __device__ void featureIntegrateFwd(float weight,
                                                               const TFeaturesVec& features,
                                                               TFeaturesVec& integratedFeatures) {

        particleFeaturesIntegrateFwd(weight,
                                     *reinterpret_cast<const float3*>(&features),
                                     reinterpret_cast<float3*>(&integratedFeatures));
    }

    __forceinline__ __device__ void featuresIntegrateFwdFromBuffer(const tcnn::vec3& incidentDirection,
                                                                   float weight,
                                                                   uint32_t particleIdx, TFeaturesVec integratedFeatures) const {

        particleFeaturesIntegrateFwdFromBuffer(*reinterpret_cast<const float3*>(&incidentDirection),
                                               weight,
                                               particleIdx,
                                               {{m_featureRawParameters.ptr, nullptr, true}, m_featureActiveShDegree},
                                               reinterpret_cast<float3*>(&integratedFeatures));
    }

    static __forceinline__ __device__ void featuresIntegrateBwd(float alpha,
                                                                float& alphaGrad,
                                                                const TFeaturesVec& features,
                                                                TFeaturesVec& featuresGrad,
                                                                TFeaturesVec& integratedFeatures,
                                                                TFeaturesVec& integratedFeaturesGrad) {
        if (TDifferentiable) {
            particleFeaturesIntegrateBwd(alpha,
                                         &alphaGrad,
                                         *reinterpret_cast<const float3*>(&features),
                                         reinterpret_cast<float3*>(&featuresGrad),
                                         reinterpret_cast<float3*>(&integratedFeatures),
                                         reinterpret_cast<float3*>(&integratedFeaturesGrad));
        }
    }

    template <bool exclusiveGradient, bool Sampling = false>
    __forceinline__ __device__ void featuresIntegrateBwdToBuffer(const tcnn::vec3& incidentDirection,
                                                                 tcnn::vec3* incidentDirectionGrad,
                                                                 float alpha,
                                                                 float& alphaGrad,
                                                                 uint32_t particleIdx,
                                                                 const TFeaturesVec& features,
                                                                 TFeaturesVec& integratedFeatures,
                                                                 TFeaturesVec& integratedFeaturesGrad) const {

        if constexpr (TDifferentiable) {
            if constexpr (Sampling) {
                particleFeaturesIntegrateSampleBwdToBuffer(*reinterpret_cast<const float3*>(&incidentDirection),
                                                           incidentDirectionGrad != nullptr,
                                                           reinterpret_cast<float3*>(incidentDirectionGrad),
                                                           alpha,
                                                           &alphaGrad,
                                                           particleIdx,
                                                           {{m_featureRawParameters.ptr, m_featureRawParameters.gradPtr, exclusiveGradient}, m_featureActiveShDegree},
                                                           *reinterpret_cast<const float3*>(&features),
                                                           reinterpret_cast<float3*>(&integratedFeatures),
                                                           reinterpret_cast<float3*>(&integratedFeaturesGrad));
            } else {
                particleFeaturesIntegrateBwdToBuffer(*reinterpret_cast<const float3*>(&incidentDirection),
                                                     incidentDirectionGrad != nullptr,
                                                     reinterpret_cast<float3*>(incidentDirectionGrad),
                                                     alpha,
                                                     &alphaGrad,
                                                     particleIdx,
                                                     {{m_featureRawParameters.ptr, m_featureRawParameters.gradPtr, exclusiveGradient}, m_featureActiveShDegree},
                                                     *reinterpret_cast<const float3*>(&features),
                                                     reinterpret_cast<float3*>(&integratedFeatures),
                                                     reinterpret_cast<float3*>(&integratedFeaturesGrad));
            }
        }
    }

    static constexpr int ExtendedFeaturesDim  = ExtParams::ExtendedFeaturesDim;
    static constexpr bool HasExtendedFeatures = ExtendedFeaturesDim > 0;
    using ExtendedFeaturesRawParameters       = typename tcnn::tvec<TFeaturesType, HasExtendedFeatures ? ExtParams::ExtendedFeaturesParametersDim : 1>;
    using TExtendedFeaturesArr                = FixedArray<float, HasExtendedFeatures ? ExtendedFeaturesDim : 1>; //< defined by Slang compiler

    static constexpr int CameraExtendedFeaturesDim  = ExtParams::CameraExtendedFeaturesDim;
    static constexpr bool HasCameraExtendedFeatures = CameraExtendedFeaturesDim > 0;
    using CameraExtendedFeaturesRawParameters       = typename tcnn::tvec<TFeaturesType, HasCameraExtendedFeatures ? ExtParams::CameraExtendedFeaturesParametersDim : 1>;
    using TCameraExtendedFeaturesArr                = FixedArray<float, HasCameraExtendedFeatures ? CameraExtendedFeaturesDim : 1>; //< defined by Slang compiler

    static constexpr int LidarExtendedFeaturesDim  = ExtParams::LidarExtendedFeaturesDim;
    static constexpr bool HasLidarExtendedFeatures = LidarExtendedFeaturesDim > 0;
    using LidarExtendedFeaturesRawParameters       = typename tcnn::tvec<TFeaturesType, HasLidarExtendedFeatures ? ExtParams::LidarExtendedFeaturesParametersDim : 1>;
    using TLidarExtendedFeaturesArr                = FixedArray<float, HasLidarExtendedFeatures ? LidarExtendedFeaturesDim : 1>; //< defined by Slang compiler

    template <bool EnableExtendedFeatures,
              bool EnableCameraExtendedFeatures,
              bool EnableLidarExtendedFeatures>
    inline __device__ void initializeExtendedFeatures(nrend::MemoryHandles parameters) {
        if constexpr (HasExtendedFeatures && EnableExtendedFeatures) {
            m_extendedFeaturesParameters.ptr = parameters.bufferPtr<TFeaturesType>(Params::ExtendedFeaturesRawParametersBufferIndex);
        }
        if constexpr (HasCameraExtendedFeatures && EnableCameraExtendedFeatures) {
            m_cameraExtendedFeaturesParameters.ptr = parameters.bufferPtr<TFeaturesType>(Params::CameraExtendedFeaturesRawParametersBufferIndex);
        }
        if constexpr (HasLidarExtendedFeatures && EnableLidarExtendedFeatures) {
            m_lidarExtendedFeaturesParameters.ptr = parameters.bufferPtr<TFeaturesType>(Params::LidarExtendedFeaturesRawParametersBufferIndex);
        }
    };

    template <bool EnableExtendedFeatures,
              bool EnableCameraExtendedFeatures,
              bool EnableLidarExtendedFeatures>
    inline __device__ void initializeExtendedFeaturesGradient(nrend::MemoryHandles parametersGradient) {
        if constexpr (TDifferentiable) {
            if constexpr (HasExtendedFeatures && EnableExtendedFeatures) {
                m_extendedFeaturesParameters.gradPtr = parametersGradient.bufferPtr<TFeaturesType>(Params::ExtendedFeaturesRawParametersGradientBufferIndex);
            }
            if constexpr (HasCameraExtendedFeatures && EnableCameraExtendedFeatures) {
                m_cameraExtendedFeaturesParameters.gradPtr = parametersGradient.bufferPtr<TFeaturesType>(Params::CameraExtendedFeaturesRawParametersGradientBufferIndex);
            }
            if constexpr (HasLidarExtendedFeatures && EnableLidarExtendedFeatures) {
                m_lidarExtendedFeaturesParameters.gradPtr = parametersGradient.bufferPtr<TFeaturesType>(Params::LidarExtendedFeaturesRawParametersGradientBufferIndex);
            }
        }
    };

    __forceinline__ __device__ ExtendedFeaturesRawParameters extendedFeaturesRawParametersFromBuffer(uint32_t particleIdx) const {
        return reinterpret_cast<const ExtendedFeaturesRawParameters*>(m_extendedFeaturesParameters.ptr)[particleIdx];
    }

    __forceinline__ __device__ CameraExtendedFeaturesRawParameters cameraExtendedFeaturesRawParametersFromBuffer(uint32_t particleIdx) const {
        return reinterpret_cast<const CameraExtendedFeaturesRawParameters*>(m_cameraExtendedFeaturesParameters.ptr)[particleIdx];
    }

    __forceinline__ __device__ LidarExtendedFeaturesRawParameters lidarExtendedFeaturesRawParametersFromBuffer(uint32_t particleIdx) const {
        return reinterpret_cast<const LidarExtendedFeaturesRawParameters*>(m_lidarExtendedFeaturesParameters.ptr)[particleIdx];
    }

    template <bool EnableExtendedFeatures,
              bool EnableCameraExtendedFeatures,
              bool EnableLidarExtendedFeatures,
              bool exclusiveGradient,
              typename TEnabledExtendedFeaturesVec>
    __forceinline__ __device__ void extendedFeaturesBwdToBuffer(uint32_t particleIdx,
                                                                const TEnabledExtendedFeaturesVec& features,
                                                                const TEnabledExtendedFeaturesVec& featuresGrad,
                                                                const tcnn::vec3& incidentDirection,
                                                                tcnn::vec3& incidentDirectionGrad) const {
        if constexpr (TDifferentiable) {

            if constexpr (HasExtendedFeatures && EnableExtendedFeatures) {
                particleExtendedFeaturesBwdToBuffer(particleIdx,
                                                    {m_extendedFeaturesParameters.ptr, m_extendedFeaturesParameters.gradPtr, exclusiveGradient},
                                                    *reinterpret_cast<const TExtendedFeaturesArr*>(featuresGrad.data()),
                                                    *reinterpret_cast<const float3*>(&incidentDirection),
                                                    reinterpret_cast<float3*>(&incidentDirectionGrad));
            }
            if constexpr (HasCameraExtendedFeatures && EnableCameraExtendedFeatures) {
                constexpr int kOffset = EnableExtendedFeatures ? ExtendedFeaturesDim : 0;
                particleCameraExtendedFeaturesBwdToBuffer(particleIdx,
                                                          {m_cameraExtendedFeaturesParameters.ptr, m_cameraExtendedFeaturesParameters.gradPtr, exclusiveGradient},
                                                          *reinterpret_cast<const TCameraExtendedFeaturesArr*>(featuresGrad.data() + kOffset),
                                                          *reinterpret_cast<const float3*>(&incidentDirection),
                                                          reinterpret_cast<float3*>(&incidentDirectionGrad));
            }
            if constexpr (HasLidarExtendedFeatures && EnableLidarExtendedFeatures) {
                constexpr int kOffset = (EnableExtendedFeatures ? ExtendedFeaturesDim : 0) +
                                        (EnableCameraExtendedFeatures ? CameraExtendedFeaturesDim : 0);
                particleLidarExtendedFeaturesBwdToBuffer(particleIdx,
                                                         {m_lidarExtendedFeaturesParameters.ptr, m_lidarExtendedFeaturesParameters.gradPtr, exclusiveGradient},
                                                         *reinterpret_cast<const TLidarExtendedFeaturesArr*>(featuresGrad.data() + kOffset),
                                                         *reinterpret_cast<const float3*>(&incidentDirection),
                                                         reinterpret_cast<float3*>(&incidentDirectionGrad));
            }
        }
    }

    template <bool EnableExtendedFeatures,
              bool EnableCameraExtendedFeatures,
              bool EnableLidarExtendedFeatures,
              int EnabledExtendedFeaturesDim = (EnableExtendedFeatures ? ExtendedFeaturesDim : 0) +
                                               (EnableCameraExtendedFeatures ? CameraExtendedFeaturesDim : 0) +
                                               (EnableLidarExtendedFeatures ? LidarExtendedFeaturesDim : 0)>
    __forceinline__ __device__ tcnn::vec<EnabledExtendedFeaturesDim> extendedFeaturesFromBuffer(
        uint32_t particleIdx,
        const tcnn::vec3& incidentDirection) const {

        static_assert(EnabledExtendedFeaturesDim > 0, "EnabledExtendedFeaturesDim must be greater than 0");

        using TExtendedFeaturesVec    = tcnn::vec<EnabledExtendedFeaturesDim>;
        TExtendedFeaturesVec features = TExtendedFeaturesVec::zero();

        if constexpr (HasExtendedFeatures && EnableExtendedFeatures) {
            particleExtendedFeaturesFromBuffer(particleIdx,
                                               {m_extendedFeaturesParameters.ptr, nullptr, true},
                                               *reinterpret_cast<const float3*>(&incidentDirection),
                                               reinterpret_cast<TExtendedFeaturesArr*>(features.data()));
        }
        if constexpr (HasCameraExtendedFeatures && EnableCameraExtendedFeatures) {
            constexpr int kOffset = EnableExtendedFeatures ? ExtendedFeaturesDim : 0;
            particleCameraExtendedFeaturesFromBuffer(particleIdx,
                                                     {m_cameraExtendedFeaturesParameters.ptr, nullptr, true},
                                                     *reinterpret_cast<const float3*>(&incidentDirection),
                                                     reinterpret_cast<TCameraExtendedFeaturesArr*>(features.data() + kOffset));
        }
        if constexpr (HasLidarExtendedFeatures && EnableLidarExtendedFeatures) {
            constexpr int kOffset = (EnableExtendedFeatures ? ExtendedFeaturesDim : 0) +
                                    (EnableCameraExtendedFeatures ? CameraExtendedFeaturesDim : 0);
            particleLidarExtendedFeaturesFromBuffer(particleIdx,
                                                    {m_lidarExtendedFeaturesParameters.ptr, nullptr, true},
                                                    *reinterpret_cast<const float3*>(&incidentDirection),
                                                    reinterpret_cast<TLidarExtendedFeaturesArr*>(features.data() + kOffset));
        }
        return features;
    }

    template <bool EnableExtendedFeatures,
              bool EnableCameraExtendedFeatures,
              bool EnableLidarExtendedFeatures,
              typename TEnabledExtendedFeaturesVec>
    static __forceinline__ __device__ void extendedFeaturesIntegrateFwd(float weight,
                                                                        const TEnabledExtendedFeaturesVec& features,
                                                                        TEnabledExtendedFeaturesVec& integratedFeatures) {

        if constexpr (HasExtendedFeatures && EnableExtendedFeatures) {
            particleExtendedFeaturesIntegrateFwd(weight,
                                                 *reinterpret_cast<const TExtendedFeaturesArr*>(features.data()),
                                                 reinterpret_cast<TExtendedFeaturesArr*>(integratedFeatures.data()));
        }
        if constexpr (HasCameraExtendedFeatures && EnableCameraExtendedFeatures) {
            constexpr int kOffset = EnableExtendedFeatures ? ExtendedFeaturesDim : 0;
            particleCameraExtendedFeaturesIntegrateFwd(weight,
                                                       *reinterpret_cast<const TCameraExtendedFeaturesArr*>(features.data() + kOffset),
                                                       reinterpret_cast<TCameraExtendedFeaturesArr*>(integratedFeatures.data() + kOffset));
        }
        if constexpr (HasLidarExtendedFeatures && EnableLidarExtendedFeatures) {
            constexpr int kOffset = (EnableExtendedFeatures ? ExtendedFeaturesDim : 0) +
                                    (EnableCameraExtendedFeatures ? CameraExtendedFeaturesDim : 0);
            particleLidarExtendedFeaturesIntegrateFwd(weight,
                                                      *reinterpret_cast<const TLidarExtendedFeaturesArr*>(features.data() + kOffset),
                                                      reinterpret_cast<TLidarExtendedFeaturesArr*>(integratedFeatures.data() + kOffset));
        }
    }

    template <bool EnableExtendedFeatures,
              bool EnableCameraExtendedFeatures,
              bool EnableLidarExtendedFeatures,
              bool exclusiveGradient,
              bool Sampling,
              typename TEnabledExtendedFeaturesVec>
    __forceinline__ __device__ void extendedFeaturesIntegrateBwdToBuffer(const tcnn::vec3& incidentDirection,
                                                                         tcnn::vec3* incidentDirectionGrad,
                                                                         float alpha,
                                                                         float& alphaGrad,
                                                                         uint32_t particleIdx,
                                                                         const TEnabledExtendedFeaturesVec& features,
                                                                         TEnabledExtendedFeaturesVec& integratedFeatures,
                                                                         TEnabledExtendedFeaturesVec& integratedFeaturesGrad) const {

        if constexpr (TDifferentiable) {

            if constexpr (HasExtendedFeatures && EnableExtendedFeatures) {
                if constexpr (Sampling) {
                    particleExtendedFeaturesIntegrateSampleBwdToBuffer(*reinterpret_cast<const float3*>(&incidentDirection),
                                                                       incidentDirectionGrad != nullptr,
                                                                       reinterpret_cast<float3*>(incidentDirectionGrad),
                                                                       alpha,
                                                                       &alphaGrad,
                                                                       particleIdx,
                                                                       {m_extendedFeaturesParameters.ptr, m_extendedFeaturesParameters.gradPtr, exclusiveGradient},
                                                                       *reinterpret_cast<const TExtendedFeaturesArr*>(features.data()),
                                                                       reinterpret_cast<TExtendedFeaturesArr*>(integratedFeatures.data()),
                                                                       reinterpret_cast<TExtendedFeaturesArr*>(integratedFeaturesGrad.data()));
                } else {
                    particleExtendedFeaturesIntegrateBwdToBuffer(*reinterpret_cast<const float3*>(&incidentDirection),
                                                                 incidentDirectionGrad != nullptr,
                                                                 reinterpret_cast<float3*>(incidentDirectionGrad),
                                                                 alpha,
                                                                 &alphaGrad,
                                                                 particleIdx,
                                                                 {m_extendedFeaturesParameters.ptr, m_extendedFeaturesParameters.gradPtr, exclusiveGradient},
                                                                 *reinterpret_cast<const TExtendedFeaturesArr*>(features.data()),
                                                                 reinterpret_cast<TExtendedFeaturesArr*>(integratedFeatures.data()),
                                                                 reinterpret_cast<TExtendedFeaturesArr*>(integratedFeaturesGrad.data()));
                }
            }
            if constexpr (HasCameraExtendedFeatures && EnableCameraExtendedFeatures) {
                constexpr int kOffset = EnableExtendedFeatures ? ExtendedFeaturesDim : 0;
                if constexpr (Sampling) {
                    particleCameraExtendedFeaturesIntegrateSampleBwdToBuffer(
                        *reinterpret_cast<const float3*>(&incidentDirection),
                        incidentDirectionGrad != nullptr,
                        reinterpret_cast<float3*>(incidentDirectionGrad),
                        alpha,
                        &alphaGrad,
                        particleIdx,
                        {m_cameraExtendedFeaturesParameters.ptr, m_cameraExtendedFeaturesParameters.gradPtr, exclusiveGradient},
                        *reinterpret_cast<const TCameraExtendedFeaturesArr*>(features.data() + kOffset),
                        reinterpret_cast<TCameraExtendedFeaturesArr*>(integratedFeatures.data() + kOffset),
                        reinterpret_cast<TCameraExtendedFeaturesArr*>(integratedFeaturesGrad.data() + kOffset));
                } else {
                    particleCameraExtendedFeaturesIntegrateBwdToBuffer(
                        *reinterpret_cast<const float3*>(&incidentDirection),
                        incidentDirectionGrad != nullptr,
                        reinterpret_cast<float3*>(incidentDirectionGrad),
                        alpha,
                        &alphaGrad,
                        particleIdx,
                        {m_cameraExtendedFeaturesParameters.ptr, m_cameraExtendedFeaturesParameters.gradPtr, exclusiveGradient},
                        *reinterpret_cast<const TCameraExtendedFeaturesArr*>(features.data() + kOffset),
                        reinterpret_cast<TCameraExtendedFeaturesArr*>(integratedFeatures.data() + kOffset),
                        reinterpret_cast<TCameraExtendedFeaturesArr*>(integratedFeaturesGrad.data() + kOffset));
                }
            }
            if constexpr (HasLidarExtendedFeatures && EnableLidarExtendedFeatures) {
                constexpr int kOffset = (EnableExtendedFeatures ? ExtendedFeaturesDim : 0) +
                                        (EnableCameraExtendedFeatures ? CameraExtendedFeaturesDim : 0);
                if constexpr (Sampling) {
                    particleLidarExtendedFeaturesIntegrateSampleBwdToBuffer(
                        *reinterpret_cast<const float3*>(&incidentDirection),
                        incidentDirectionGrad != nullptr,
                        reinterpret_cast<float3*>(incidentDirectionGrad),
                        alpha,
                        &alphaGrad,
                        particleIdx,
                        {m_lidarExtendedFeaturesParameters.ptr, m_lidarExtendedFeaturesParameters.gradPtr, exclusiveGradient},
                        *reinterpret_cast<const TLidarExtendedFeaturesArr*>(features.data() + kOffset),
                        reinterpret_cast<TLidarExtendedFeaturesArr*>(integratedFeatures.data() + kOffset),
                        reinterpret_cast<TLidarExtendedFeaturesArr*>(integratedFeaturesGrad.data() + kOffset));
                } else {
                    particleLidarExtendedFeaturesIntegrateBwdToBuffer(
                        *reinterpret_cast<const float3*>(&incidentDirection),
                        incidentDirectionGrad != nullptr,
                        reinterpret_cast<float3*>(incidentDirectionGrad),
                        alpha,
                        &alphaGrad,
                        particleIdx,
                        {m_lidarExtendedFeaturesParameters.ptr, m_lidarExtendedFeaturesParameters.gradPtr, exclusiveGradient},
                        *reinterpret_cast<const TLidarExtendedFeaturesArr*>(features.data() + kOffset),
                        reinterpret_cast<TLidarExtendedFeaturesArr*>(integratedFeatures.data() + kOffset),
                        reinterpret_cast<TLidarExtendedFeaturesArr*>(integratedFeaturesGrad.data() + kOffset));
                }
            }
        }
    }

    template <bool EnableExtendedFeatures,
              bool EnableCameraExtendedFeatures,
              bool EnableLidarExtendedFeatures,
              typename TEnabledExtendedFeaturesVec>
    static __forceinline__ __device__ void extendedFeaturesIntegrateBwdToVec(float alpha,
                                                                             float& alphaGrad,
                                                                             const TEnabledExtendedFeaturesVec& features,
                                                                             TEnabledExtendedFeaturesVec& integratedFeatures,
                                                                             TEnabledExtendedFeaturesVec& integratedFeaturesGrad,
                                                                             TEnabledExtendedFeaturesVec& featuresGrad) {
        if constexpr (TDifferentiable) {

            if constexpr (HasExtendedFeatures && EnableExtendedFeatures) {
                particleExtendedFeaturesIntegrateBwd(alpha,
                                                     &alphaGrad,
                                                     *reinterpret_cast<const TExtendedFeaturesArr*>(features.data()),
                                                     reinterpret_cast<TExtendedFeaturesArr*>(integratedFeatures.data()),
                                                     reinterpret_cast<TExtendedFeaturesArr*>(integratedFeaturesGrad.data()),
                                                     reinterpret_cast<TExtendedFeaturesArr*>(featuresGrad.data()));
            }
            if constexpr (HasCameraExtendedFeatures && EnableCameraExtendedFeatures) {
                constexpr int kOffset = EnableExtendedFeatures ? ExtendedFeaturesDim : 0;
                particleCameraExtendedFeaturesIntegrateBwd(alpha,
                                                           &alphaGrad,
                                                           *reinterpret_cast<const TCameraExtendedFeaturesArr*>(features.data() + kOffset),
                                                           reinterpret_cast<TCameraExtendedFeaturesArr*>(integratedFeatures.data() + kOffset),
                                                           reinterpret_cast<TCameraExtendedFeaturesArr*>(integratedFeaturesGrad.data() + kOffset),
                                                           reinterpret_cast<TCameraExtendedFeaturesArr*>(featuresGrad.data() + kOffset));
            }
            if constexpr (HasLidarExtendedFeatures && EnableLidarExtendedFeatures) {
                constexpr int kOffset = (EnableExtendedFeatures ? ExtendedFeaturesDim : 0) +
                                        (EnableCameraExtendedFeatures ? CameraExtendedFeaturesDim : 0);
                particleLidarExtendedFeaturesIntegrateBwd(alpha,
                                                          &alphaGrad,
                                                          *reinterpret_cast<const TLidarExtendedFeaturesArr*>(features.data() + kOffset),
                                                          reinterpret_cast<TLidarExtendedFeaturesArr*>(integratedFeatures.data() + kOffset),
                                                          reinterpret_cast<TLidarExtendedFeaturesArr*>(integratedFeaturesGrad.data() + kOffset),
                                                          reinterpret_cast<TLidarExtendedFeaturesArr*>(featuresGrad.data() + kOffset));
            }
        }
    }

private:
    NREShRadiativeGaussianParticlesBuffer<TDensityRawParameters, TDifferentiable> m_densityRawParameters;

    int m_featureActiveShDegree = 0;
    NREShRadiativeGaussianParticlesBuffer<TFeaturesType, TDifferentiable> m_featureRawParameters;

    NREShRadiativeGaussianParticlesOptionalBuffer<TFeaturesType, TDifferentiable, HasExtendedFeatures>
        m_extendedFeaturesParameters;
    NREShRadiativeGaussianParticlesOptionalBuffer<TFeaturesType, TDifferentiable, HasCameraExtendedFeatures>
        m_cameraExtendedFeaturesParameters;
    NREShRadiativeGaussianParticlesOptionalBuffer<TFeaturesType, TDifferentiable, HasLidarExtendedFeatures>
        m_lidarExtendedFeaturesParameters;
};
