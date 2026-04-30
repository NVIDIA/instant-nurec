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

#include <nrend/kernels/cuda/sensors/sensorsProjection.cuh>
#include <nrend/renderer/gutRendererParameters.h>
#include <nrend/renderer/renderParameters.h>

struct GUTProjectorDummyParams {
    static constexpr float CovarianceDilation          = 0.f;
    static constexpr float AlphaThreshold              = 0.f;
    static constexpr bool TightOpacityBounding         = false;
    static constexpr bool RectBounding                 = false;
    static constexpr bool TileCulling                  = false;
    static constexpr bool PerRayParticleFeatures       = false;
    static constexpr float NearClipDistance            = 0.f;
    static constexpr float FarClipDistance             = 0.f;
    static constexpr bool NearFarZCulling              = false;
    static constexpr float MaxDepthValue               = 0.f;
    static constexpr bool BackwardProjection           = false;
    static constexpr bool EnableLinearProjection       = false;
    static constexpr uint32_t SceneDataDim             = 0;
    static constexpr int32_t SceneDataVisibilityOffset = -1;
};

struct GUTProjectionDummyParams {
    static constexpr int NRollingShutterIterations = 10;
    static constexpr int D                         = 3;
    // See Gustafsson and Hendeby 2012 for sigma point parameterization - this default parameter
    // choice is based on "The unscented Kalman filter for nonlinear estimation" - Wan and van der Merwe 2000
    static constexpr float Alpha = 0.1;
    static constexpr float Beta  = 2.f;
    static constexpr float Kappa = 0.f;
    static constexpr float Delta = 1.0f; ///< sqrt(Alpha*Alpha*(D+Kappa))
    // 10% out of bounds margin is acceptable for "valid" projection state
    static constexpr float ImageMarginFactor = 0.1f;
    // true: all sigma points must be valid to mark a projection as "valid"
    // false: a single valid sigma point is sufficient to mark a projection as "valid"
    static constexpr bool RequireAllSigmaPoints = true;
};

template <typename Particles,
          typename Params                   = GUTProjectorDummyParams,
          typename UTParams                 = GUTProjectionDummyParams,
          typename TilingParams             = nrend::GUTParameters::DefaultTiling,
          bool EnableFeatures               = true,
          bool EnableExtendedFeatures       = true,
          bool EnableCameraExtendedFeatures = true,
          bool EnableLidarExtendedFeatures  = false>
struct GUTProjector : Params, UTParams {

    static constexpr int EnabledFeaturesDim = EnableFeatures ? Particles::FeaturesDim : 0;
    static constexpr int EnabledExtendedFeaturesDim =
        (EnableExtendedFeatures ? Particles::ExtendedFeaturesDim : 0) +
        (EnableCameraExtendedFeatures ? Particles::CameraExtendedFeaturesDim : 0) +
        (EnableLidarExtendedFeatures ? Particles::LidarExtendedFeaturesDim : 0);

    using TPrecomputedFeaturesVec = tcnn::vec<EnabledFeaturesDim + EnabledExtendedFeaturesDim + (Params::BackwardProjection ? 1 : 0)>;

    static inline __device__ uint64_t concatTileDepthKeys(uint32_t tileKey, uint32_t depthKey) {
        return (static_cast<uint64_t>(tileKey) << 32) | depthKey;
    }

    static inline __device__ float tileMinParticlePowerResponse(const tcnn::vec2& tileCoords,
                                                                const tcnn::vec4& conicOpacity,
                                                                const tcnn::vec2& meanPosition) {

        const tcnn::vec2 tileSize = tcnn::vec2(TilingParams::BlockX, TilingParams::BlockY);
        const tcnn::vec2 tileMin  = tileSize * tileCoords;
        const tcnn::vec2 tileMax  = tileSize + tileMin;

        const tcnn::vec2 minOffset  = tileMin - meanPosition;
        const tcnn::vec2 leftAbove  = tcnn::vec2(minOffset.x > 0.0f, minOffset.y > 0.0f);
        const tcnn::vec2 notInRange = tcnn::vec2(leftAbove.x + (meanPosition.x > tileMax.x),
                                                 leftAbove.y + (meanPosition.y > tileMax.y));

        if ((notInRange.x + notInRange.y) > 0.0f) {
            const tcnn::vec2 p    = tcnn::mix(tileMax, tileMin, leftAbove);
            const tcnn::vec2 dxy  = tcnn::copysign(tileSize, minOffset);
            const tcnn::vec2 diff = meanPosition - p;
            const tcnn::vec2 rcp  = tcnn::vec2(__frcp_rn(tileSize.x * tileSize.x * conicOpacity.x),
                                               __frcp_rn(tileSize.y * tileSize.y * conicOpacity.z));

            const float tx = notInRange.y * __saturatef((dxy.x * conicOpacity.x * diff.x + dxy.x * conicOpacity.y * diff.y) * rcp.x);
            const float ty = notInRange.x * __saturatef((dxy.y * conicOpacity.y * diff.x + dxy.y * conicOpacity.z * diff.y) * rcp.y);

            const tcnn::vec2 minPosDiff = meanPosition - tcnn::vec2(p.x + tx * dxy.x, p.y + ty * dxy.y);

            return 0.5f * (conicOpacity.x * minPosDiff.x * minPosDiff.x + conicOpacity.z * minPosDiff.y * minPosDiff.y) + conicOpacity.y * minPosDiff.x * minPosDiff.y;
        }
        // mean position is within the tile
        return 0.f;
    }

    /// Convert a projected particle to its conic/opacity representation
    static inline __device__ bool computeProjectedExtentConicOpacity(tcnn::vec3 covariance,
                                                                     float covarianceDilationOffset,
                                                                     float opacity,
                                                                     tcnn::ivec2& extent,
                                                                     tcnn::vec4& conicOpacity,
                                                                     float& maxConicOpacityPower) {
        const tcnn::vec3 dilatedCovariance = tcnn::vec3{covariance.x + covarianceDilationOffset, covariance.y, covariance.z + covarianceDilationOffset};
        const float dilatedCovDet          = dilatedCovariance.x * dilatedCovariance.z - dilatedCovariance.y * dilatedCovariance.y;
        if (dilatedCovDet == 0.0f) {
            return false;
        }
        conicOpacity.slice<0, 3>() = tcnn::vec3{dilatedCovariance.z, -dilatedCovariance.y, dilatedCovariance.x} / dilatedCovDet;

        // see Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting" https://github.com/autonomousvision/mip-splatting
        const float covDet            = covariance.x * covariance.z - covariance.y * covariance.y;
        const float convolutionFactor = sqrtf(fmaxf(0.000025f, covDet / dilatedCovDet));
        conicOpacity.w                = opacity * convolutionFactor;

        if (conicOpacity.w < Params::AlphaThreshold) {
            return false;
        }

        maxConicOpacityPower     = logf(conicOpacity.w / Params::AlphaThreshold);
        const float extentFactor = Params::TightOpacityBounding ? fminf(3.33f, sqrtf(2.0f * maxConicOpacityPower)) : 3.33f;
        const float minLambda    = 0.01f;
        const float mid          = 0.5f * (dilatedCovariance.x + dilatedCovariance.z);
        const float lambda       = mid + sqrtf(fmaxf(minLambda, mid * mid - dilatedCovDet));
        const float radius       = extentFactor * sqrtf(lambda);
        const tcnn::vec2 extentf = Params::RectBounding ? min(extentFactor * sqrt(tcnn::vec2{dilatedCovariance.x, dilatedCovariance.z}), tcnn::vec2{radius}) : tcnn::vec2{radius};

        // Cast to integer, see reference:
        // https://github.com/nerfstudio-project/gsplat/blob/10586bbd17445de0c07db3b0c07c73aab49a0c29/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu#L179-L182
        extent.x = static_cast<int>(ceilf(extentf.x));
        extent.y = static_cast<int>(ceilf(extentf.y));

        return (extent.x > 0) && (extent.y > 0);
    }

    static inline __device__ bool linearParticleProjection(
        const tcnn::vec2& resolution,
        const nrend::PerspectiveProjectionParameters& sensorModel,
        const tcnn::vec3& sensorPosition,
        const tcnn::mat4x3& toSensorMatrix,
        const Particles& particles,
        const typename Particles::DensityParameters& particleParameters,
        tcnn::vec3& particleSensorRay,
        float& particleProjOpacity,
        tcnn::vec2& particleProjCenter,
        tcnn::vec3& particleProjCovariance) {

        // TODO : take principal point into account
        return particles.densityPerspectiveConicProjection(particleParameters,
                                                           resolution,
                                                           tcnn::vec2{Params::NearClipDistance, Params::FarClipDistance},
                                                           sensorModel.focalLength,
                                                           sensorModel.principalPoint,
                                                           toSensorMatrix,
                                                           sensorPosition,
                                                           particleSensorRay,
                                                           particleProjCenter,
                                                           particleProjCovariance,
                                                           particleProjOpacity);
    }

    static inline __device__ bool unscentedParticleProjection(
        const tcnn::vec2& resolution,
        const tcnn::mat4x3& toWorldMatrix,
        const nrend::TSensorModel& sensorModel,
        const tcnn::vec3& sensorPosition,
        const tcnn::mat4x3& toSensorMatrix,
        const nrend::TSensorState& sensorShutterState,
        const Particles& particles,
        const typename Particles::DensityParameters& particleParameters,
        tcnn::vec3& particleSensorRay,
        float& particleProjOpacity,
        tcnn::vec2& particleProjCenter,
        tcnn::vec3& particleProjCovariance) {

        particleProjOpacity = particles.opacity(particleParameters);
        if (particleProjOpacity < Params::AlphaThreshold) {
            return false;
        }

        const tcnn::vec3& particleMean = particles.position(particleParameters);
        if (nrend::sensorCullPosition<Params::NearFarZCulling>(sensorModel,
                                                               toSensorMatrix,
                                                               tcnn::vec2{Params::NearClipDistance, Params::FarClipDistance},
                                                               particleMean)) {
            return false;
        }

        const tcnn::vec3& particleScale   = particles.scale(particleParameters);
        const tcnn::mat3 particleRotation = particles.rotation(particleParameters);

        particleSensorRay = particleMean - sensorPosition;

        int numValidPoints = 0;
        tcnn::vec2 projectedSigmaPoints[2 * UTParams::D + 1];

        constexpr float Lambda = UTParams::Alpha * UTParams::Alpha * (UTParams::D + UTParams::Kappa) - UTParams::D;

        if (nrend::projectPointWithShutter<UTParams::NRollingShutterIterations>(toWorldMatrix * tcnn::vec4(particleMean, 1.f),
                                                                                resolution,
                                                                                sensorModel,
                                                                                sensorShutterState,
                                                                                UTParams::ImageMarginFactor,
                                                                                projectedSigmaPoints[0])) {
            numValidPoints++;
        }
        particleProjCenter = projectedSigmaPoints[0] * (Lambda / (UTParams::D + Lambda));

        constexpr float weightI = 1.f / (2.f * (UTParams::D + Lambda));
#pragma unroll
        for (int i = 0; i < UTParams::D; ++i) {
            const tcnn::vec3 delta = UTParams::Delta * particleScale[i] * particleRotation[i];

            if (nrend::projectPointWithShutter<UTParams::NRollingShutterIterations>(toWorldMatrix * tcnn::vec4(particleMean + delta, 1.f),
                                                                                    resolution,
                                                                                    sensorModel,
                                                                                    sensorShutterState,
                                                                                    UTParams::ImageMarginFactor,
                                                                                    projectedSigmaPoints[i + 1])) {
                numValidPoints++;
            }
            particleProjCenter += weightI * projectedSigmaPoints[i + 1];

            if (nrend::projectPointWithShutter<UTParams::NRollingShutterIterations>(toWorldMatrix * tcnn::vec4(particleMean - delta, 1.f),
                                                                                    resolution,
                                                                                    sensorModel,
                                                                                    sensorShutterState,
                                                                                    UTParams::ImageMarginFactor,
                                                                                    projectedSigmaPoints[i + 1 + UTParams::D])) {
                numValidPoints++;
            }
            particleProjCenter += weightI * projectedSigmaPoints[i + 1 + UTParams::D];
        }

        if constexpr (UTParams::RequireAllSigmaPoints) {
            if (numValidPoints < (2 * UTParams::D + 1)) {
                return false;
            }
        } else if (numValidPoints == 0) {
            return false;
        }

        {
            const tcnn::vec2 centeredPoint = projectedSigmaPoints[0] - particleProjCenter;
            constexpr float weight0        = Lambda / (UTParams::D + Lambda) + (1.f - UTParams::Alpha * UTParams::Alpha + UTParams::Beta);
            particleProjCovariance         = weight0 * tcnn::vec3(centeredPoint.x * centeredPoint.x,
                                                                  centeredPoint.x * centeredPoint.y,
                                                                  centeredPoint.y * centeredPoint.y);
        }
#pragma unroll
        for (int i = 0; i < 2 * UTParams::D; ++i) {
            const tcnn::vec2 centeredPoint = projectedSigmaPoints[i + 1] - particleProjCenter;
            particleProjCovariance += weightI * tcnn::vec3(centeredPoint.x * centeredPoint.x,
                                                           centeredPoint.x * centeredPoint.y,
                                                           centeredPoint.y * centeredPoint.y);
        }

        return true;
    }

    static inline __device__ void eval(tcnn::ivec2 tileGrid,
                                       uint32_t numParticles,
                                       tcnn::vec2 resolution,
                                       tcnn::vec2 offset,
                                       tcnn::mat4x3 objectToWorldMatrix,
                                       nrend::TSensorModel sensorModel,
                                       tcnn::vec3 sensorPosition,
                                       tcnn::mat4x3 sensorViewMatrix,
                                       nrend::TSensorState sensorShutterState,
                                       uint32_t* __restrict__ particlesTilesCountPtr,
                                       tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                       tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                       tcnn::ivec2* __restrict__ particlesProjectedExtentPtr,
                                       float* __restrict__ particlesGlobalDepthPtr,
                                       TPrecomputedFeaturesVec* __restrict__ particlesPrecomputedFeaturesPtr,
                                       float* __restrict__ sceneDataPtr,
                                       nrend::MemoryHandles parameters) {

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx >= numParticles) {
            return;
        }

        Particles particles;
        particles.initializeDensity(parameters);
        const auto particleParameters = particles.fetchDensityParameters(particleIdx);

        tcnn::vec2 particleProjCenter;
        float particleProjOpacity;
        tcnn::vec3 particleSensorRay;
        tcnn::vec3 particleProjCovariance;
        bool validProjection = false;
        if (Params::EnableLinearProjection && (sensorModel.modelType == nrend::TSensorModel::PerspectiveModel) && (sensorModel.shutterType == nrend::TSensorModel::GlobalShutter)) {
            validProjection = linearParticleProjection(
                resolution,
                sensorModel.perspectiveParams,
                sensorPosition,
                sensorViewMatrix,
                particles,
                particleParameters,
                particleSensorRay,
                particleProjOpacity,
                particleProjCenter,
                particleProjCovariance);
        } else {
            validProjection = unscentedParticleProjection(
                resolution,
                objectToWorldMatrix,
                sensorModel,
                sensorPosition,
                sensorViewMatrix,
                // FIXME : work directly in sensor space to avoid all intermediate transforms
                sensorShutterState,
                particles,
                particleParameters,
                particleSensorRay,
                particleProjOpacity,
                particleProjCenter,
                particleProjCovariance);
        }

        tcnn::ivec2 particleProjExtent;
        tcnn::vec4 particleProjConicOpacity;
        float particleMaxConicOpacityPower;
        if (validProjection) {
            // transform from frame UV space to frame-tile UV space
            particleProjCenter -= offset;

            validProjection = computeProjectedExtentConicOpacity(particleProjCovariance,
                                                                 Params::MinProjectedRayRadiusSq * nrend::projectionDilationFactor(sensorModel),
                                                                 particleProjOpacity,
                                                                 particleProjExtent,
                                                                 particleProjConicOpacity,
                                                                 particleMaxConicOpacityPower);
        }

        uint32_t numValidParticleProjections = 0;
        if (validProjection) {
            assert((particleProjExtent.x > 0) && (particleProjExtent.y > 0));

            // Compute the tile extent of the projected particle. This computation will be repeated in the expand kernel
            tcnn::ivec2 minTileExtent, maxTileExtent;
            tcnn::ivec2 minDenseTileExtent, maxDenseTileExtent;
            nrend::projectionTileExtent<TilingParams::BlockX, TilingParams::BlockY>(
                sensorModel, tileGrid, particleProjCenter, particleProjExtent,
                minTileExtent, maxTileExtent, minDenseTileExtent, maxDenseTileExtent);

            if (!TilingParams::EnableRayBasedCulling || minTileExtent.y < maxTileExtent.y) {
                if (Params::TileCulling && sensorSupportsTileCulling(sensorModel)) {
                    for (int y = minTileExtent.y; y < maxTileExtent.y; ++y) {
                        const float cy = sensorWrapAzimuthTileIfLidar(sensorModel, y, tileGrid.y);
                        for (int x = minTileExtent.x; x < maxTileExtent.x; ++x) {
                            const float cx = x;
                            assert(0 <= cx && cx < tileGrid.x);
                            if (tileMinParticlePowerResponse({cx, cy}, particleProjConicOpacity, particleProjCenter) < particleMaxConicOpacityPower) {
                                numValidParticleProjections++;
                            }
                        }
                    }
                } else {
                    numValidParticleProjections = (maxTileExtent.x - minTileExtent.x) * (maxTileExtent.y - minTileExtent.y);
                }
            }
        }

        particlesTilesCountPtr[particleIdx] = numValidParticleProjections;

        // Write visibility mask for particles that passed projection validation
        if constexpr (Params::SceneDataVisibilityOffset >= 0) {
            if (sceneDataPtr && numValidParticleProjections > 0) {
                sceneDataPtr[particleIdx * Params::SceneDataDim + Params::SceneDataVisibilityOffset] = 1.0f;
            }
        }

        if (numValidParticleProjections == 0) {
            particlesProjectedPositionPtr[particleIdx]     = tcnn::vec2::zero();
            particlesProjectedConicOpacityPtr[particleIdx] = tcnn::vec4::zero();
            particlesProjectedExtentPtr[particleIdx]       = tcnn::ivec2::zero();
            particlesGlobalDepthPtr[particleIdx]           = 0.f;
            return;
        }

        const float particleSensorDistance = length(particleSensorRay);

        if constexpr (!Params::PerRayParticleFeatures) {
            const tcnn::vec3 incidentDirection = particleSensorRay / particleSensorDistance;
            if constexpr (EnabledFeaturesDim) {
                particles.initializeFeatures(parameters);
                nrend::sliceVec<0, EnabledFeaturesDim>(particlesPrecomputedFeaturesPtr[particleIdx]) =
                    particles.featuresFromBuffer(particleIdx, incidentDirection);
            }
            if constexpr (EnabledExtendedFeaturesDim) {
                particles.template initializeExtendedFeatures<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(parameters);
                nrend::sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(particlesPrecomputedFeaturesPtr[particleIdx]) =
                    particles.template extendedFeaturesFromBuffer<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(particleIdx, incidentDirection);
            }
        }

        particlesProjectedPositionPtr[particleIdx]     = particleProjCenter;
        particlesProjectedConicOpacityPtr[particleIdx] = particleProjConicOpacity;
        particlesProjectedExtentPtr[particleIdx]       = particleProjExtent;
        if constexpr (Params::GlobalZOrder) {
            const tcnn::vec3& particleMean       = particles.position(particleParameters);
            particlesGlobalDepthPtr[particleIdx] = (particleMean.x * sensorViewMatrix[0][2] + particleMean.y * sensorViewMatrix[1][2] +
                                                    particleMean.z * sensorViewMatrix[2][2] + sensorViewMatrix[3][2]);
        } else {
            particlesGlobalDepthPtr[particleIdx] = particleSensorDistance;
        }
        if constexpr (Params::BackwardProjection) {
            static_assert(TPrecomputedFeaturesVec::size() > EnabledFeaturesDim + EnabledExtendedFeaturesDim, "TPrecomputedFeaturesVec wrong size.");
            particlesPrecomputedFeaturesPtr[particleIdx][EnabledFeaturesDim + EnabledExtendedFeaturesDim] = particleSensorDistance;
        }
    }

    static inline __device__ void prepare(tcnn::ivec2 tileGrid,
                                          uint32_t numParticles,
                                          tcnn::vec2 resolution,
                                          tcnn::vec2 offset,
                                          tcnn::mat4x3 objectToWorldMatrix,
                                          nrend::TSensorModel sensorModel,
                                          tcnn::vec3 sensorPosition,
                                          tcnn::mat4x3 sensorViewMatrix,
                                          nrend::TSensorState sensorShutterState,
                                          uint32_t* __restrict__ particlesCountIdxPtr,
                                          float* __restrict__ particlesDensityPtr,
                                          float* __restrict__ particlesFeaturesPtr,
                                          float* __restrict__ particlesExtendedFeaturesPtr,
                                          float* __restrict__ particlesSensorExtendedFeaturesPtr,
                                          float* __restrict__ sceneDataPtr,
                                          nrend::MemoryHandles parameters) {

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;

        bool validParticle = (particleIdx < numParticles);

        Particles particles;
        tcnn::vec3 particleSensorRay;

        if (validParticle) {
            particles.initializeDensity(parameters);
            const auto particleParameters = particles.fetchDensityParameters(particleIdx);

            tcnn::vec2 particleProjCenter;
            float particleProjOpacity;
            tcnn::vec3 particleProjCovariance;
            bool validProjection = false;
            if (Params::EnableLinearProjection && (sensorModel.modelType == nrend::TSensorModel::PerspectiveModel) && (sensorModel.shutterType == nrend::TSensorModel::GlobalShutter)) {
                validProjection = linearParticleProjection(
                    resolution,
                    sensorModel.perspectiveParams,
                    sensorPosition,
                    sensorViewMatrix,
                    particles,
                    particleParameters,
                    particleSensorRay,
                    particleProjOpacity,
                    particleProjCenter,
                    particleProjCovariance);
            } else {
                validProjection = unscentedParticleProjection(
                    resolution,
                    objectToWorldMatrix,
                    sensorModel,
                    sensorPosition,
                    sensorViewMatrix,
                    // FIXME : work directly in sensor space to avoid all intermediate transforms
                    sensorShutterState,
                    particles,
                    particleParameters,
                    particleSensorRay,
                    particleProjOpacity,
                    particleProjCenter,
                    particleProjCovariance);
            }

            tcnn::ivec2 particleProjExtent;
            tcnn::vec4 particleProjConicOpacity;
            float particleMaxConicOpacityPower;
            if (validProjection) {
                // transform from frame UV space to frame-tile UV space
                particleProjCenter -= offset;

                validProjection = computeProjectedExtentConicOpacity(particleProjCovariance,
                                                                     Params::MinProjectedRayRadiusSq * nrend::projectionDilationFactor(sensorModel),
                                                                     particleProjOpacity,
                                                                     particleProjExtent,
                                                                     particleProjConicOpacity,
                                                                     particleMaxConicOpacityPower);
            }

            uint32_t numValidParticleProjections = 0;
            if (validProjection) {
                assert((particleProjExtent.x > 0) && (particleProjExtent.y > 0));

                // Compute the tile extent of the projected particle. This computation will be repeated in the expand kernel
                tcnn::ivec2 minTileExtent, maxTileExtent;
                tcnn::ivec2 minDenseTileExtent, maxDenseTileExtent;
                nrend::projectionTileExtent<TilingParams::BlockX, TilingParams::BlockY>(
                    sensorModel, tileGrid, particleProjCenter, particleProjExtent,
                    minTileExtent, maxTileExtent, minDenseTileExtent, maxDenseTileExtent);

                if (!TilingParams::EnableRayBasedCulling || minTileExtent.y < maxTileExtent.y) {
                    if (Params::TileCulling && sensorSupportsTileCulling(sensorModel)) {
                        for (int y = minTileExtent.y; y < maxTileExtent.y; ++y) {
                            const float cy = sensorWrapAzimuthTileIfLidar(sensorModel, y, tileGrid.y);
                            assert(0 <= cy && cy < tileGrid.y);
                            for (int x = minTileExtent.x; x < maxTileExtent.x; ++x) {
                                const float cx = x;
                                assert(0 <= cx && cx < tileGrid.x);
                                if (tileMinParticlePowerResponse({cx, cy}, particleProjConicOpacity, particleProjCenter) < particleMaxConicOpacityPower) {
                                    numValidParticleProjections++;
                                    break;
                                }
                            }
                        }
                    } else {
                        numValidParticleProjections = (maxTileExtent.x - minTileExtent.x) * (maxTileExtent.y - minTileExtent.y);
                    }
                }

                validParticle = (numValidParticleProjections > 0);
            }
        }

        // Write visibility mask for particles that passed projection validation
        if constexpr (Params::SceneDataVisibilityOffset >= 0) {
            if (sceneDataPtr && validParticle) {
                sceneDataPtr[particleIdx * Params::SceneDataDim + Params::SceneDataVisibilityOffset] = 1.0f;
            }
        }

        uint32_t warpOffset   = validParticle ? 1 : 0;
        uint32_t globalOffset = 0;
        // warp-based valid cumsum of numValidParticleProjections
        if constexpr (true) {
            constexpr uint32_t kWarpMask = 0xFFFFFFFF;
            const uint32_t laneId        = threadIdx.x & (warpSize - 1);
            // Inclusive prefix scan using up-sweep pattern
#pragma unroll
            for (int offset = 1; offset < warpSize; offset <<= 1) {
                const uint32_t n = __shfl_up_sync(kWarpMask, warpOffset, offset);
                if (laneId >= offset) {
                    warpOffset += n;
                }
            }

            // Last lane performs atomic add with total warp count
            if (laneId == warpSize - 1) {
                globalOffset = atomicAdd(&particlesCountIdxPtr[0], warpOffset);
            }
            globalOffset = __shfl_sync(kWarpMask, globalOffset, warpSize - 1);
        } else if (validParticle) {
            globalOffset = atomicAdd(&particlesCountIdxPtr[0], warpOffset);
        }

        if (validParticle) {
            // -1 because the cumsum is inclusive
            const uint32_t idx = globalOffset + warpOffset - 1;
            // + 1 because the first element is the number of valid particles
            particlesCountIdxPtr[idx + 1]                                                         = particleIdx;
            reinterpret_cast<typename Particles::DensityRawParameters*>(particlesDensityPtr)[idx] = particles.fetchDensityRawParameters(particleIdx);
            static_assert(!Params::PerRayParticleFeatures, "Eval scene with PerRayParticleFeatures not supported");
            const tcnn::vec3 incidentDirection = tcnn::normalize(particleSensorRay);
            if constexpr (EnabledFeaturesDim) {
                particles.initializeFeatures(parameters);
                reinterpret_cast<tcnn::vec<EnabledFeaturesDim>*>(particlesFeaturesPtr)[idx] =
                    particles.featuresFromBuffer(particleIdx, incidentDirection);
            }
            if constexpr (EnabledExtendedFeaturesDim) {
                particles.template initializeExtendedFeatures<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(parameters);
            }
            if constexpr (EnableExtendedFeatures) {
                reinterpret_cast<tcnn::vec<Particles::ExtendedFeaturesDim>*>(particlesExtendedFeaturesPtr)[idx] =
                    particles.template extendedFeaturesFromBuffer<true, false, false>(particleIdx, incidentDirection);
            }
            static_assert(!(EnableCameraExtendedFeatures && EnableLidarExtendedFeatures),
                          "EnableCameraExtendedFeatures and EnableLidarExtendedFeatures cannot be enabled at the same time in eval scene.");
            if constexpr (EnableCameraExtendedFeatures) {
                reinterpret_cast<tcnn::vec<Particles::CameraExtendedFeaturesDim>*>(particlesSensorExtendedFeaturesPtr)[idx] =
                    particles.template extendedFeaturesFromBuffer<false, true, false>(particleIdx, incidentDirection);
            }
            if constexpr (EnableLidarExtendedFeatures) {
                reinterpret_cast<tcnn::vec<Particles::LidarExtendedFeaturesDim>*>(particlesSensorExtendedFeaturesPtr)[idx] =
                    particles.template extendedFeaturesFromBuffer<false, false, true>(particleIdx, incidentDirection);
            }
        }
    }

    static inline __device__ void fillInvalidTileIndices(uint64_t* __restrict__ unsortedTileDepthKeysPtr,
                                                         uint32_t* __restrict__ unsortedTileParticleIdxPtr,
                                                         uint32_t& tileOffset,
                                                         const uint32_t maxTileOffset) {
        for (; tileOffset < maxTileOffset; ++tileOffset) {
            unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(nrend::GUTParameters::InvalidTileIdx,
                                                                         __float_as_uint(Params::MaxDepthValue));
            unsortedTileParticleIdxPtr[tileOffset] = nrend::GUTParameters::InvalidParticleIdx;
        }
    }

    static inline __device__ void expand(tcnn::ivec2 tileGrid,
                                         int numParticles,
                                         nrend::TSensorModel sensorModel,
                                         nrend::TSensorState /*sensorState*/,
                                         const uint32_t* __restrict__ particlesTilesOffsetPtr,
                                         const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                         const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                         const tcnn::ivec2* __restrict__ particlesProjectedExtentPtr,
                                         const float* __restrict__ particlesGlobalDepthPtr,
                                         nrend::MemoryHandles parameters,
                                         uint64_t* __restrict__ unsortedTileDepthKeysPtr,
                                         uint32_t* __restrict__ unsortedTileParticleIdxPtr) {

        const int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (particleIdx >= numParticles) {
            return;
        }

        const uint32_t minTileOffset = (particleIdx == 0) ? 0 : particlesTilesOffsetPtr[particleIdx - 1];
        const uint32_t maxTileOffset = particlesTilesOffsetPtr[particleIdx];

        if (minTileOffset == maxTileOffset) {
            return;
        }

        const uint32_t depthKey = *reinterpret_cast<const uint32_t*>(&particlesGlobalDepthPtr[particleIdx]);

        const tcnn::vec2 particleProjCenter  = particlesProjectedPositionPtr[particleIdx];
        const tcnn::ivec2 particleProjExtent = particlesProjectedExtentPtr[particleIdx];
        assert((particleProjExtent.x > 0) && (particleProjExtent.y > 0));

        tcnn::ivec2 minTileExtent, maxTileExtent;
        tcnn::ivec2 minDenseTileExtent, maxDenseTileExtent;
        nrend::projectionTileExtent<TilingParams::BlockX, TilingParams::BlockY>(
            sensorModel, tileGrid, particleProjCenter, particleProjExtent,
            minTileExtent, maxTileExtent, minDenseTileExtent, maxDenseTileExtent);

        if (!TilingParams::EnableRayBasedCulling || minTileExtent.y < maxTileExtent.y) {
            uint32_t tileOffset = minTileOffset;

            if (Params::TileCulling && sensorSupportsTileCulling(sensorModel)) {
                const tcnn::vec4 conicOpacity    = particlesProjectedConicOpacityPtr[particleIdx];
                const float maxConicOpacityPower = logf(conicOpacity.w / Params::AlphaThreshold);
                for (int y = minTileExtent.y; y < maxTileExtent.y; ++y) {
                    const float cy = sensorWrapAzimuthTileIfLidar(sensorModel, y, tileGrid.y);
                    assert(0 <= cy && cy < tileGrid.y);
                    for (int x = minTileExtent.x; x < maxTileExtent.x; ++x) {
                        const float cx = x;
                        assert(0 <= cx && cx < tileGrid.x);
                        if (tileMinParticlePowerResponse(tcnn::vec2(cx, cy), conicOpacity, particleProjCenter) < maxConicOpacityPower) {
                            unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(cy * tileGrid.x + cx, depthKey);
                            unsortedTileParticleIdxPtr[tileOffset] = particleIdx;
                            tileOffset++;
                        }
                    }
                }
                // Fill the rest of the tile with invalid tile indices
                fillInvalidTileIndices(unsortedTileDepthKeysPtr, unsortedTileParticleIdxPtr, tileOffset, maxTileOffset);
            } else {
                for (int y = minTileExtent.y; y < maxTileExtent.y; ++y) {
                    const int cy = sensorWrapAzimuthTileIfLidar(sensorModel, y, tileGrid.y);
                    assert(0 <= cy && cy < tileGrid.y);
                    for (int x = minTileExtent.x; x < maxTileExtent.x; ++x) {
                        const int cx = x;
                        assert(0 <= cx && cx < tileGrid.x);
                        unsortedTileDepthKeysPtr[tileOffset]   = concatTileDepthKeys(cy * tileGrid.x + cx, depthKey);
                        unsortedTileParticleIdxPtr[tileOffset] = particleIdx;
                        tileOffset++;
                    }
                }
            }

#ifndef NDEBUG
            assert(tileOffset == maxTileOffset);
#else
            // FIXME : this will trigger for ~5% of time, need to understand why. For now, we fill the rest of the tile
            // with invalid tile indices to avoid a crash.
            if (tileOffset != maxTileOffset) {
                fillInvalidTileIndices(unsortedTileDepthKeysPtr, unsortedTileParticleIdxPtr, tileOffset, maxTileOffset);
            }
#endif
        }
    }

    static inline __device__ void
    evalBackward(uint32_t numParticles,
                 tcnn::vec2 resolution,
                 nrend::TSensorModel sensorModel,
                 tcnn::vec3 sensorPosition,
                 tcnn::mat4x3 sensorViewMatrix,
                 const uint32_t* __restrict__ particlesTilesCountPtr,
                 nrend::MemoryHandles parameters,
                 const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                 const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                 const TPrecomputedFeaturesVec* __restrict__ particlesPrecomputedFeaturesPtr,
                 const TPrecomputedFeaturesVec* __restrict__ particlesPrecomputedFeaturesGradPtr,
                 nrend::MemoryHandles parametersGradient) {
        if constexpr (Params::PerRayParticleFeatures) {
            return;
        }

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx >= numParticles) {
            return;
        }
        if (particlesTilesCountPtr[particleIdx] == 0) {
            return;
        }

        Particles particles;
        particles.initializeDensity(parameters);
        particles.initializeDensityGradient(parametersGradient);
        const tcnn::vec3 incidentDirection = tcnn::normalize(particles.fetchPosition(particleIdx) - sensorPosition);
        tcnn::vec3 incidentDirectionGrad   = tcnn::vec3::zero();

        if constexpr (EnabledFeaturesDim) {
            particles.initializeFeatures(parameters);
            particles.initializeFeaturesGradient(parametersGradient);

            particles.template featuresBwdToBuffer<true>(particleIdx,
                                                         nrend::sliceVec<0, EnabledFeaturesDim>(particlesPrecomputedFeaturesPtr[particleIdx]),
                                                         nrend::sliceVec<0, EnabledFeaturesDim>(particlesPrecomputedFeaturesGradPtr[particleIdx]),
                                                         incidentDirection,
                                                         incidentDirectionGrad);
        }

        if constexpr (EnabledExtendedFeaturesDim) {
            particles.template initializeExtendedFeatures<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(parameters);
            particles.template initializeExtendedFeaturesGradient<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(parametersGradient);
            particles.template extendedFeaturesBwdToBuffer<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures, true>(
                particleIdx,
                nrend::sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(particlesPrecomputedFeaturesPtr[particleIdx]),
                nrend::sliceVec<EnabledFeaturesDim, EnabledExtendedFeaturesDim>(particlesPrecomputedFeaturesGradPtr[particleIdx]),
                incidentDirection,
                incidentDirectionGrad);
        }

        particles.template densityIncidentDirectionBwdToBuffer<true>(particleIdx, sensorPosition, incidentDirectionGrad);

        if constexpr (Params::BackwardProjection) {
            static_assert(TPrecomputedFeaturesVec::size() > EnabledFeaturesDim + EnabledExtendedFeaturesDim, "TPrecomputedFeaturesVec wrong size.");
            particles.template densityPerspectiveConicProjectionBwdToBuffer<true>(particleIdx,
                                                                                  resolution,
                                                                                  tcnn::vec2{Params::NearClipDistance, Params::FarClipDistance},
                                                                                  sensorModel.perspectiveParams.focalLength,
                                                                                  sensorModel.perspectiveParams.principalPoint,
                                                                                  sensorViewMatrix,
                                                                                  sensorPosition,
                                                                                  Params::MinProjectedRayRadiusSq * nrend::projectionDilationFactor(sensorModel),
                                                                                  particlesProjectedPositionGradPtr[particleIdx],
                                                                                  particlesProjectedConicOpacityGradPtr[particleIdx],
                                                                                  particlesPrecomputedFeaturesGradPtr[particleIdx][EnabledFeaturesDim + EnabledExtendedFeaturesDim] /*hitDistance*/);
        }
    }

    static inline __device__ void
    prepareBackward(uint32_t numValidParticles,
                    tcnn::vec2 resolution,
                    nrend::TSensorModel sensorModel,
                    tcnn::vec3 sensorPosition,
                    tcnn::mat4x3 sensorViewMatrix,
                    uint32_t* __restrict__ particlesCountIdxPtr,
                    const void* __restrict__ particlesFeaturesPtr,
                    const void* __restrict__ particlesExtendedFeaturesPtr,
                    const void* __restrict__ particlesSensorExtendedFeaturesPtr,
                    nrend::MemoryHandles parameters,
                    const void* __restrict__ particlesDensityGradPtr,
                    const void* __restrict__ particlesFeaturesGradPtr,
                    const void* __restrict__ particlesExtendedFeaturesGradPtr,
                    const void* __restrict__ particlesSensorExtendedFeaturesGradPtr,
                    nrend::MemoryHandles parametersGradient) {
        if constexpr (Params::PerRayParticleFeatures) {
            return;
        }

        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numValidParticles) {
            return;
        }

        // + 1 because the first element is the number of valid particles
        const uint32_t particleIdx = particlesCountIdxPtr[idx + 1];

        Particles particles;
        particles.initializeDensity(parameters);
        particles.initializeDensityGradient(parametersGradient);
        // accumulate density gradient
        particles.fetchDensityRawParametersBwd<true>(
            particleIdx,
            reinterpret_cast<const typename Particles::DensityRawParameters*>(particlesDensityGradPtr)[idx]);

        const tcnn::vec3 incidentDirection = tcnn::normalize(particles.fetchPosition(particleIdx) - sensorPosition);
        tcnn::vec3 incidentDirectionGrad   = tcnn::vec3::zero();

        if constexpr (EnabledFeaturesDim) {
            particles.initializeFeatures(parameters);
            particles.initializeFeaturesGradient(parametersGradient);
            particles.template featuresBwdToBuffer<true>(particleIdx,
                                                         reinterpret_cast<const tcnn::vec<EnabledFeaturesDim>*>(particlesFeaturesPtr)[idx],
                                                         reinterpret_cast<const tcnn::vec<EnabledFeaturesDim>*>(particlesFeaturesGradPtr)[idx],
                                                         incidentDirection,
                                                         incidentDirectionGrad);
        }

        if constexpr (EnabledExtendedFeaturesDim) {
            particles.template initializeExtendedFeatures<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(parameters);
            particles.template initializeExtendedFeaturesGradient<EnableExtendedFeatures, EnableCameraExtendedFeatures, EnableLidarExtendedFeatures>(parametersGradient);

            if constexpr (EnableExtendedFeatures) {
                particles.template extendedFeaturesBwdToBuffer<true, false, false, true>(
                    particleIdx,
                    reinterpret_cast<const tcnn::vec<Particles::ExtendedFeaturesDim>*>(particlesExtendedFeaturesPtr)[idx],
                    reinterpret_cast<const tcnn::vec<Particles::ExtendedFeaturesDim>*>(particlesExtendedFeaturesGradPtr)[idx],
                    incidentDirection,
                    incidentDirectionGrad);
            }

            static_assert(!(EnableCameraExtendedFeatures && EnableLidarExtendedFeatures),
                          "EnableCameraExtendedFeatures and EnableLidarExtendedFeatures cannot be enabled at the same time in eval scene.");

            if constexpr (EnableCameraExtendedFeatures) {
                particles.template extendedFeaturesBwdToBuffer<false, true, false, true>(
                    particleIdx,
                    reinterpret_cast<const tcnn::vec<Particles::CameraExtendedFeaturesDim>*>(particlesSensorExtendedFeaturesPtr)[idx],
                    reinterpret_cast<const tcnn::vec<Particles::CameraExtendedFeaturesDim>*>(particlesSensorExtendedFeaturesGradPtr)[idx],
                    incidentDirection,
                    incidentDirectionGrad);
            }

            if constexpr (EnableLidarExtendedFeatures) {
                particles.template extendedFeaturesBwdToBuffer<false, false, true, true>(
                    particleIdx,
                    reinterpret_cast<const tcnn::vec<Particles::LidarExtendedFeaturesDim>*>(particlesSensorExtendedFeaturesPtr)[idx],
                    reinterpret_cast<const tcnn::vec<Particles::LidarExtendedFeaturesDim>*>(particlesSensorExtendedFeaturesGradPtr)[idx],
                    incidentDirection,
                    incidentDirectionGrad);
            }
        }

        particles.template densityIncidentDirectionBwdToBuffer<true>(particleIdx, sensorPosition, incidentDirectionGrad);
    }
};
