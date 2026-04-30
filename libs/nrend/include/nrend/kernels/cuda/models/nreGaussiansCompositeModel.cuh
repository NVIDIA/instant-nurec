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

#include <nrend/kernels/cuda/common/nreSimpleTuple.cuh>
#include <nrend/kernels/cuda/common/nreStdUtils.cuh>
#include <nrend/kernels/cuda/models/nreAppearanceEmbedding.cuh>
#include <nrend/renderer/renderParameters.h>
#include <nrend/utils/nrePose.h>

struct NREGaussiansCompositeParamsDummy {
    static constexpr int RenderingCumulativeNumInstancesBufferIndex = 0;
    static constexpr int RenderingActiveInstancesBufferIndex        = 0;
    static constexpr int ParticleDensityBufferIndex                 = 0;
    static constexpr int ParticleFeaturesBufferIndex                = 0;
    static constexpr int ExtendedFeaturesDim                        = 0;
    static constexpr int ParticleExtendedFeaturesBufferIndex        = 0;
    static constexpr int CameraExtendedFeaturesDim                  = 0;
    static constexpr int ParticleCameraExtendedFeaturesBufferIndex  = 0;
    static constexpr int LidarExtendedFeaturesDim                   = 0;
    static constexpr int ParticleLidarExtendedFeaturesBufferIndex   = 0;
    static constexpr int NumStaticParticles                         = 0;
    static constexpr int PrimitiveIdOffset                          = 0;
    static constexpr bool WarpParallelSearch                        = false;
    static constexpr bool EnableBackground                          = true;
    static constexpr bool EnablePostProcessings                     = true;
    static constexpr bool SaturateRadiance                          = true;
};
template <typename TParticles,
          typename TAppearanceEmbedding,
          typename TBackground,
          typename TPostProcessings,
          typename TParams,
          typename... TPrimitives>
struct NREGaussiansComposite : private TParams {

    using Particles                           = TParticles;
    using DensityRawParameters                = typename Particles::DensityRawParameters;
    using FeaturesRawParameters               = typename Particles::FeaturesRawParameters;
    using ExtendedFeaturesRawParameters       = typename Particles::ExtendedFeaturesRawParameters;
    using CameraExtendedFeaturesRawParameters = typename Particles::CameraExtendedFeaturesRawParameters;
    using LidarExtendedFeaturesRawParameters  = typename Particles::LidarExtendedFeaturesRawParameters;

    static inline __device__ uint32_t warpParallelLinearSearch(const uint32_t* __restrict__ cumNumParticlesInstances,
                                                               uint32_t numInstances,
                                                               uint32_t particleId) {
        constexpr uint32_t kWarpMask = 0xFFFFFFFF;
        const uint32_t laneId        = (threadIdx.y * blockDim.x + threadIdx.x) & (warpSize - 1);
        for (uint32_t i = 1; i < numInstances; i += warpSize) {
            const uint32_t instanceId      = min(i + laneId, numInstances - 1);
            const uint32_t cumNumParticles = cumNumParticlesInstances[instanceId];
            const uint32_t foundMask       = __ballot_sync(kWarpMask, particleId < cumNumParticles);
            if (foundMask != 0) {
                const uint32_t findingLaneId = __ffs(foundMask) - 1;
                return __shfl_sync(kWarpMask, instanceId, findingLaneId) - 1;
            }
        }
        return numInstances - 1;
    }

    static inline __device__ uint32_t linearSearch(const uint32_t* __restrict__ cumNumParticlesInstancesPtr,
                                                   uint32_t numInstances,
                                                   uint32_t particleId) {
        for (uint32_t i = 1; i < numInstances; ++i) {
            const uint32_t cumNumParticles = cumNumParticlesInstancesPtr[i];
            if (particleId < cumNumParticles) {
                return i - 1;
            }
        }
        return numInstances - 1;
    }

    struct RenderingActiveInstance {
        static constexpr uint32_t InvalidPrimitiveId         = 0xFFFFFFFF;
        static constexpr uint32_t InvalidPrimitiveInstanceId = 0xFFFFFFFF;

        uint32_t numParticles        = 0;
        uint32_t cumNumParticles     = 0;
        uint32_t primitiveId         = InvalidPrimitiveId;
        uint32_t primitiveInstanceId = InvalidPrimitiveInstanceId;
        uint32_t particlesOffset     = 0;
    };

    static inline __device__ void applyRigidTransform(DensityRawParameters& densityParams,
                                                      const nrend::TTrackInstancePose& pose) {
        const tcnn::mat3 rotation        = tcnn::to_mat3(tcnn::quat{pose[6], pose[3], pose[4], pose[5]});
        densityParams.position           = rotation * densityParams.position + pose.slice<0, 3>();
        const tcnn::quat transformedQuat = tcnn::quat(
            rotation * tcnn::to_mat3(tcnn::quat{densityParams.quaternion[0], densityParams.quaternion[1], densityParams.quaternion[2], densityParams.quaternion[3]}));
        densityParams.quaternion = tcnn::vec4{transformedQuat.w, transformedQuat.x, transformedQuat.y, transformedQuat.z};
    }

    template <int PrimitiveId,
              typename TFirstPrimitive,
              typename... TRemainingPrimitives>
    static inline __device__ DensityRawParameters fetchAndDeformPrimitiveParticleDensity(uint32_t primitiveId,
                                                                                         uint32_t primitiveInstanceId,
                                                                                         uint32_t primitiveParticleId,
                                                                                         nrend::TTimestamp timestamp,
                                                                                         const nrend::TTrackInstancePose& pose,
                                                                                         nrend::MemoryHandles parameters) {
        if (PrimitiveId == primitiveId) {
            typename TFirstPrimitive::Particles particle;
            particle.initializeDensity(parameters);
            auto densityParams = particle.fetchDensityRawParameters(primitiveParticleId);
            TFirstPrimitive::deform(timestamp, primitiveInstanceId, parameters, densityParams);
            DensityRawParameters deformedDensityParams = {densityParams.position, densityParams.density, densityParams.quaternion, densityParams.scale, 0.0f};
            applyRigidTransform(deformedDensityParams, pose);
            return deformedDensityParams;
        } else if constexpr (sizeof...(TRemainingPrimitives) > 0) {
            return fetchAndDeformPrimitiveParticleDensity<PrimitiveId + 1, TRemainingPrimitives...>(
                primitiveId, primitiveInstanceId, primitiveParticleId, timestamp, pose, parameters);
        } else {
            __builtin_unreachable();
            // if the primitive id is not found, return a default density parameters
            return DensityRawParameters{
                tcnn::vec3::zero(),
                0.0f, // zero density ensure rendering will not be visible
                tcnn::vec4::zero(),
                tcnn::vec3::zero(),
                0.0f};
        }
    }

    template <int PrimitiveId,
              typename TFirstPrimitive,
              typename... TRemainingPrimitives>
    static inline __device__ FeaturesRawParameters fetchPrimitiveParticleRawFeatures(uint32_t primitiveId,
                                                                                     uint32_t primitiveInstanceId,
                                                                                     uint32_t primitiveParticleId,
                                                                                     nrend::MemoryHandles parameters) {
        if (PrimitiveId == primitiveId) {
            typename TFirstPrimitive::Particles particle;
            particle.initializeFeatures(parameters);
            return particle.featuresRawParametersFromBuffer(primitiveParticleId);
        } else if constexpr (sizeof...(TRemainingPrimitives) > 0) {
            return fetchPrimitiveParticleRawFeatures<PrimitiveId + 1, TRemainingPrimitives...>(
                primitiveId, primitiveInstanceId, primitiveParticleId, parameters);
        } else {
            // if the primitive id is not found, return a default density parameters
            return FeaturesRawParameters::zero();
        }
    }

    template <int PrimitiveId,
              typename TFirstPrimitive,
              typename... TRemainingPrimitives>
    static inline __device__ void fetchPrimitiveParticleExtendedRawFeatures(uint32_t primitiveId,
                                                                            uint32_t primitiveInstanceId,
                                                                            uint32_t primitiveParticleId,
                                                                            nrend::MemoryHandles parameters,
                                                                            ExtendedFeaturesRawParameters& extendedFeaturesRawParameters) {
        if (PrimitiveId == primitiveId) {
            typename TFirstPrimitive::Particles particle;
            particle.template initializeExtendedFeatures<true, false, false>(parameters);
            extendedFeaturesRawParameters = particle.extendedFeaturesRawParametersFromBuffer(primitiveParticleId);
        } else if constexpr (sizeof...(TRemainingPrimitives) > 0) {
            fetchPrimitiveParticleExtendedRawFeatures<PrimitiveId + 1, TRemainingPrimitives...>(
                primitiveId, primitiveInstanceId, primitiveParticleId, parameters, extendedFeaturesRawParameters);
        } else {
            // if the primitive id is not found, return a default density parameters
            extendedFeaturesRawParameters = ExtendedFeaturesRawParameters::zero();
        }
    }

    template <int PrimitiveId,
              typename TFirstPrimitive,
              typename... TRemainingPrimitives>
    static inline __device__ void fetchPrimitiveParticleCameraExtendedRawFeatures(uint32_t primitiveId,
                                                                                  uint32_t primitiveInstanceId,
                                                                                  uint32_t primitiveParticleId,
                                                                                  nrend::MemoryHandles parameters,
                                                                                  CameraExtendedFeaturesRawParameters& extendedFeaturesRawParameters) {
        if (PrimitiveId == primitiveId) {
            typename TFirstPrimitive::Particles particle;
            particle.template initializeExtendedFeatures<false, true, false>(parameters);
            extendedFeaturesRawParameters = particle.cameraExtendedFeaturesRawParametersFromBuffer(primitiveParticleId);
        } else if constexpr (sizeof...(TRemainingPrimitives) > 0) {
            fetchPrimitiveParticleCameraExtendedRawFeatures<PrimitiveId + 1, TRemainingPrimitives...>(
                primitiveId, primitiveInstanceId, primitiveParticleId, parameters, extendedFeaturesRawParameters);
        } else {
            // if the primitive id is not found, return a default density parameters
            extendedFeaturesRawParameters = CameraExtendedFeaturesRawParameters::zero();
        }
    }

    template <int PrimitiveId,
              typename TFirstPrimitive,
              typename... TRemainingPrimitives>
    static inline __device__ void fetchPrimitiveParticleLidarExtendedRawFeatures(uint32_t primitiveId,
                                                                                 uint32_t primitiveInstanceId,
                                                                                 uint32_t primitiveParticleId,
                                                                                 nrend::MemoryHandles parameters,
                                                                                 LidarExtendedFeaturesRawParameters& extendedFeaturesRawParameters) {
        if (PrimitiveId == primitiveId) {
            typename TFirstPrimitive::Particles particle;
            particle.template initializeExtendedFeatures<false, false, true>(parameters);
            extendedFeaturesRawParameters = particle.lidarExtendedFeaturesRawParametersFromBuffer(primitiveParticleId);
        } else if constexpr (sizeof...(TRemainingPrimitives) > 0) {
            fetchPrimitiveParticleLidarExtendedRawFeatures<PrimitiveId + 1, TRemainingPrimitives...>(
                primitiveId, primitiveInstanceId, primitiveParticleId, parameters, extendedFeaturesRawParameters);
        } else {
            // if the primitive id is not found, return a default density parameters
            extendedFeaturesRawParameters = LidarExtendedFeaturesRawParameters::zero();
        }
    }

    // TODO : first interpolate the pose outside of the kernel and compute the matrix
    static inline __device__ void preprocess(uint32_t numParticles,
                                             uint32_t numActiveTrackInstances,
                                             const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                                             nrend::TTimestamp timestamp, // median timestamp
                                             const nrend::TTrackInstancePose* activeTrackInstancesStartPoseCudaPtr,
                                             const nrend::TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                                             nrend::MemoryHandles parameterMemoryHandles) {

        static constexpr uint32_t kWarpMask = 0xFFFFFFFF;

        const uint32_t particleId = blockIdx.x * blockDim.x + threadIdx.x;
        // discard invalid warps
        if (__all_sync(kWarpMask, particleId >= numParticles)) {
            return;
        }

        // warp-level binary search to find the instance ids for the particle in the first lane
        const uint32_t* cumNumParticlesInstancesPtr = parameterMemoryHandles.bufferPtr<uint32_t>(TParams::RenderingCumulativeNumInstancesBufferIndex);
        // search for the active instance id containing the first lane particle
        uint32_t activeInstanceId;
        if constexpr (TParams::WarpParallelSearch) {
            activeInstanceId = warpParallelLinearSearch(cumNumParticlesInstancesPtr,
                                                        numActiveTrackInstances,
                                                        __shfl_sync(kWarpMask, particleId, 0));
        } else {
            activeInstanceId = linearSearch(cumNumParticlesInstancesPtr,
                                            numActiveTrackInstances,
                                            particleId);
        }
        if (activeInstanceId >= numActiveTrackInstances) {
            __builtin_unreachable();
            return;
        }
        if (!__all_sync(kWarpMask, activeInstanceId == __shfl_sync(kWarpMask, activeInstanceId, 0))) {
            __builtin_unreachable();
            return;
        }

        // get the instance id data
        const RenderingActiveInstance activeInstance =
            parameterMemoryHandles.bufferPtr<RenderingActiveInstance>(TParams::RenderingActiveInstancesBufferIndex)[activeInstanceId];
        if (activeInstance.numParticles == 0) {
            return;
        }

        // instance particle id (ensure all threads are valid)
        const uint32_t instanceParticleId  = particleId - cumNumParticlesInstancesPtr[activeInstanceId];
        const uint32_t primitiveParticleId = activeInstance.particlesOffset + min(activeInstance.numParticles - 1, instanceParticleId);

        // fetch and deform the particles density
        // NOTE : this has to be done for every threads in the warp (deformation is using Tensor Cores)

        nrend::TTrackInstancePose pose;
        if (activeTrackInstancesStartPoseCudaPtr && activeTrackInstancesEndPoseCudaPtr) {
            // TODO : maybe more efficient to interpolate the pose outside of the kernel ? (per primitive instead of per particles)
            pose = nrend::interpolatedPose(activeTrackInstancesStartPoseCudaPtr[activeInstanceId],
                                           activeTrackInstancesEndPoseCudaPtr[activeInstanceId],
                                           0.5f);
        } else {
            // clang-format off
            pose = activeTrackInstancesStartPoseCudaPtr ? activeTrackInstancesStartPoseCudaPtr[activeInstanceId] : 
                   activeTrackInstancesEndPoseCudaPtr   ? activeTrackInstancesEndPoseCudaPtr[activeInstanceId] : 
                   nrend::TTrackInstancePose{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f}; // identity pose
            // clang-format on
        }

        const DensityRawParameters deformedParticleDensity = fetchAndDeformPrimitiveParticleDensity<0, TPrimitives...>(
            activeInstance.primitiveId + TParams::PrimitiveIdOffset,
            activeInstance.primitiveInstanceId,
            primitiveParticleId,
            timestamp,
            pose,
            parameterMemoryHandles);

        if (instanceParticleId < activeInstance.numParticles) {

            const uint32_t compositeParticleId = instanceParticleId + activeInstance.cumNumParticles;

            // write the deformed particle density to the output buffer
            parameterMemoryHandles.bufferPtr<DensityRawParameters>(TParams::ParticleDensityBufferIndex)[compositeParticleId] = deformedParticleDensity;

            // fetch and write the particle features
            parameterMemoryHandles.bufferPtr<FeaturesRawParameters>(TParams::ParticleFeaturesBufferIndex)[compositeParticleId] =
                fetchPrimitiveParticleRawFeatures<0, TPrimitives...>(
                    activeInstance.primitiveId + TParams::PrimitiveIdOffset,
                    activeInstance.primitiveInstanceId,
                    primitiveParticleId,
                    parameterMemoryHandles);

            // fetch and write the extended features
            if constexpr (TParams::ExtendedFeaturesDim) {
                fetchPrimitiveParticleExtendedRawFeatures<0, TPrimitives...>(
                    activeInstance.primitiveId + TParams::PrimitiveIdOffset,
                    activeInstance.primitiveInstanceId,
                    primitiveParticleId,
                    parameterMemoryHandles,
                    parameterMemoryHandles.bufferPtr<ExtendedFeaturesRawParameters>(TParams::ParticleExtendedFeaturesBufferIndex)[compositeParticleId]);
            }
            if constexpr (TParams::CameraExtendedFeaturesDim) {
                fetchPrimitiveParticleCameraExtendedRawFeatures<0, TPrimitives...>(
                    activeInstance.primitiveId + TParams::PrimitiveIdOffset,
                    activeInstance.primitiveInstanceId,
                    primitiveParticleId,
                    parameterMemoryHandles,
                    parameterMemoryHandles.bufferPtr<CameraExtendedFeaturesRawParameters>(TParams::ParticleCameraExtendedFeaturesBufferIndex)[compositeParticleId]);
            }
            if constexpr (TParams::LidarExtendedFeaturesDim) {
                fetchPrimitiveParticleLidarExtendedRawFeatures<0, TPrimitives...>(
                    activeInstance.primitiveId + TParams::PrimitiveIdOffset,
                    activeInstance.primitiveInstanceId,
                    primitiveParticleId,
                    parameterMemoryHandles,
                    parameterMemoryHandles.bufferPtr<LidarExtendedFeaturesRawParameters>(TParams::ParticleLidarExtendedFeaturesBufferIndex)[compositeParticleId]);
            }
        }
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
        if constexpr (TRay::HasBaseFeatures && TBackground::Enabled && TParams::EnableBackground) {
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
        if constexpr (TRay::HasBaseFeatures && TParams::EnablePostProcessings) {
            if (ray.isValid()) {
                TPostProcessings::eval(ray, params, parameters, sensorsIdsPtr);
            }
        }

        if constexpr (TRay::HasBaseFeatures && TParams::SaturateRadiance) {
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
