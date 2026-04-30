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

#include <nrend/sensors/sensors.h>

#ifndef __FLT_EPSILON__
#define __FLT_EPSILON__ 1.19209290E-07f
#endif

namespace nrend {

// Computes 2-norm of a [x,y] vector in a numerically stable way
static inline __device__ float stableNorm2(const tcnn::vec2& vec) {
    const float absX = fabsf(vec.x);
    const float absY = fabsf(vec.y);
    const float min  = fminf(absX, absY);
    const float max  = fmaxf(absX, absY);
    if (max <= 0.f) {
        return 0.f;
    }
    const float minMaxRatio = min / max;
    return max * sqrtf(1.f + minMaxRatio * minMaxRatio);
}

// Wraps the azimuth angle to the interval (-π, π]
static inline __device__ float normalizeAngle(float angle) {
    const float k3Pi = 3.f * tcnn::PI(); // 3π constant
    const float k2Pi = 2.f * tcnn::PI(); // 2π constant
    // branch-less execution of azimuth wrapping
    // NOTE(qi): avoid using fmod() if possible for numerical stability
    if (-k3Pi < angle && angle <= k3Pi) {
        angle = (angle > tcnn::PI()) ? angle - k2Pi : angle;
        angle = (angle <= -tcnn::PI()) ? angle + k2Pi : angle;
        return angle;
    } else {
        angle = fmod(angle + tcnn::PI(), k2Pi);
        angle = (angle <= 0) ? (angle + k2Pi) : angle;
        return angle - tcnn::PI();
    }
}

template <int N>
static inline __device__ float evalPolyHorner(const tcnn::vec<N>& coeffs, float x) {
    // Evaluates a N-1 degree polynomial y=f(x) using numerically stable Horner scheme.
    // With :
    // f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    float y = coeffs[N - 1];
#pragma unroll
    for (int i = N - 2; i >= 0; --i) {
        y = x * y + coeffs[i];
    }
    return y;
}

static inline __device__ float evalPolyHornerRange(const float* poly, float x, int32_t idxStart, int32_t idxEnd) {
    float result = 0.;
    for (int32_t idx = idxEnd - 1; idx >= idxStart; idx--) {
        result = result * x + poly[idx];
    }
    return result;
}

static inline __device__ float polyEval2d(const float* poly2d, int32_t order, float x, float y) {
    auto outerCoeffs = tcnn::vec<BivariateWindshieldDistortionParameters::MaxPolyOrder + 1>{};
    int32_t startIdx = 0;
    for (int32_t innerOrder = order; innerOrder >= 0; innerOrder--) {
        outerCoeffs[order - innerOrder] = evalPolyHornerRange(poly2d, x, startIdx, startIdx + innerOrder + 1);
        startIdx += innerOrder + 1;
    }
    return evalPolyHornerRange(outerCoeffs.data(), y, 0, order + 1);
}

static inline __device__ tcnn::vec3 bivariateWindshieldDistortion(
    const BivariateWindshieldDistortionParameters& params, const tcnn::vec3& ray) {
    // Evaluates windshield distortion model
    const float rayLength = tcnn::length(ray);
    const float phi       = asinf(ray.x / rayLength);
    const float theta     = asinf(ray.y / rayLength);
    const float x         = sinf(polyEval2d(params.horizontalPoly, params.horizontalPolyOrder, phi, theta));
    const float y         = sinf(polyEval2d(params.verticalPoly, params.verticalPolyOrder, phi, theta));
    const float z         = sqrtf(1.f - fminf(x * x + y * y, 1.f)) * (ray.z < 0.f ? -1.f : 1.f);
    return tcnn::vec3{x, y, z};
}

static inline __device__ tcnn::vec3 correctExternalDistortion(const SensorProjectionModel& sensorModel, const tcnn::vec3& ray) {
    switch (sensorModel.externalDistortionType) {
    case SensorProjectionModel::BivariateWindshieldDistortion: {
        const BivariateWindshieldDistortionParameters& externalDistortionParameters = sensorModel.bivariateWindshieldDistortionParameters;
        return bivariateWindshieldDistortion(externalDistortionParameters, ray);
    }
    default:
        return ray;
    }
    return ray;
}

static inline __device__ float relativeClockRotation(float begin, float end, SpinningDirection direction) {
    return (direction == SpinningDirection::CLOCK_WISE)
               ? (begin - end)
               : (end - begin);
}

template <bool PIXEL_SPACE = false>
static inline __device__ float relativeAngle(float angleStart, float angleEnd, SpinningDirection direction) {
    constexpr float kToPixel = static_cast<float>(RowOffsetStructuredSpinningLidarProjectionParameters::ANGLE_TO_PIXEL_SCALING_FACTOR);
    constexpr float k2Pi     = PIXEL_SPACE ? (2.f * tcnn::PI() * kToPixel) : (2.f * tcnn::PI()); // 2π constant
    // Compute the relative angle between two angles in radians
    //    angleStart: reference angle in radians
    //    angleEnd: angle to compute the relative angle to
    //    direction: spinning direction of the lidar
    const float relativeAngle = fmod(relativeClockRotation(angleStart, angleEnd, direction), k2Pi);
    // output range [0, 2π)
    return (relativeAngle < 0) ? (relativeAngle + k2Pi) : relativeAngle;
}

static inline __device__ float relativeShutterTime(const RowOffsetStructuredSpinningLidarProjectionParameters& sensorParams,
                                                   const tcnn::vec2& /*resolution*/,
                                                   const tcnn::vec2& position) {

    constexpr float kToAngle = 1.f / static_cast<float>(RowOffsetStructuredSpinningLidarProjectionParameters::ANGLE_TO_PIXEL_SCALING_FACTOR);

    // position = sensor_angles: N x 2 array of elevation and azimuth angles in radians
    // NOTE: all sensor angles are assumed to be in the vertical fov of the sensor
    const float elevation = position.x * kToAngle;
    const float azimuth   = position.y * kToAngle;

    // Back to angle space
    const float& elevationStart = sensorParams.fovStart.y;
    const float& azimuthStart   = sensorParams.fovStart.x;

    const float relativeElevation = relativeClockRotation(elevationStart, elevation, SpinningDirection::CLOCK_WISE);
    const float relativeAzimuth   = relativeAngle(azimuthStart, azimuth, sensorParams.spin);

    const int nPtsHoriz = sensorParams.nColumns * sensorParams.angleToColumnMapResolutionFactor;
    const int nPtsVert  = sensorParams.nRows * sensorParams.angleToColumnMapResolutionFactor;

    // Compute the relative frame times by dividing the angles with the resolution of the map
    const int horizontalIdx = tcnn::clamp(static_cast<int>(relativeAzimuth / sensorParams.mapResolution.x + 0.5f), 0, nPtsHoriz - 1);
    const int verticalIdx   = tcnn::clamp(static_cast<int>(relativeElevation / sensorParams.mapResolution.y + 0.5f), 0, nPtsVert - 1);

    const float columnIdx = static_cast<float>(sensorParams.angleToColumnMap[verticalIdx * nPtsHoriz + horizontalIdx]);
    return columnIdx / (sensorParams.nColumns - 1);
}

static inline __device__ float relativeShutterTime(const SensorProjectionModel& sensorModel,
                                                   const tcnn::vec2& resolution,
                                                   const tcnn::vec2& position) {

    if (sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
        return relativeShutterTime(sensorModel.nreHesaiP128LidarParams, resolution, position);
    }

    switch (sensorModel.shutterType) {
    case SensorProjectionModel::RollingTopToBottomShutter:
        return floorf(position.y) / (resolution.y - 1.f);
    case SensorProjectionModel::RollingLeftToRightShutter:
        return floorf(position.x) / (resolution.x - 1.f);
    case SensorProjectionModel::RollingBottomToTopShutter:
        return (resolution.y - ceilf(position.y)) / (resolution.y - 1.f);
    case SensorProjectionModel::RollingRightToLeftShutter:
        return (resolution.x - ceilf(position.x)) / (resolution.x - 1.f);
    default:
        return 0.5f;
    }
};

static __forceinline__ __device__ bool withinResolution(const tcnn::vec2& resolution, float tolerance, const tcnn::vec2& p) {
    const tcnn::vec2 tolMargin = resolution * tolerance;
    return (p.x > -tolMargin.x) && (p.y > -tolMargin.y) && (p.x < resolution.x + tolMargin.x) && (p.y < resolution.y + tolMargin.y);
}

static inline __device__ bool projectPoint(const OrthographicProjectionParameters& sensorParams,
                                           const tcnn::vec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    projected = 2.f * tcnn::vec2{position.x / resolution.x, position.y / resolution.y} + 1.0f;

    return withinResolution(resolution, tolerance, projected);
}

static inline __device__ bool projectPoint(const PerspectiveProjectionParameters& sensorParams,
                                           const tcnn::vec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    if (position.z <= 0.f) {
        projected = tcnn::vec2::zero();
        return false;
    }

    // Project using ideal pinhole model (position assumed to be clipped)
    projected = (position.xy() / position.z) * sensorParams.focalLength + sensorParams.principalPoint;

    return withinResolution(resolution, tolerance, projected);
}

static inline __device__ void applyRenderingResolution(const tcnn::vec2& nominalResolution,
                                                       const tcnn::vec2& renderingResolution,
                                                       tcnn::vec2& projected) {
    if (renderingResolution != nominalResolution) {
        const float rescalingFactor = renderingResolution.x / nominalResolution.x;
        // apply vertical offset for different aspect ratios
        projected.y += 0.5f * (renderingResolution.y / rescalingFactor - nominalResolution.y);
        projected *= rescalingFactor;
    }
}

static inline __device__ bool projectPoint(const OpenCVPinholeProjectionParameters& sensorParams,
                                           const tcnn::vec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    if (position.z <= 0.f) {
        projected = tcnn::vec2::zero();
        return false;
    }

    const tcnn::vec2 uvNormalized = position.xy() / position.z;

    // computeDistortion
    const tcnn::vec2 uvSquared = uvNormalized * uvNormalized;
    const float r2             = uvSquared.x + uvSquared.y;
    const float a1             = 2.f * uvNormalized.x * uvNormalized.y;
    const float a2             = r2 + 2.f * uvSquared.x;
    const float a3             = r2 + 2.f * uvSquared.y;

    const float icD_numerator   = 1.f + r2 * (sensorParams.radialCoeffs[0] + r2 * (sensorParams.radialCoeffs[1] + r2 * sensorParams.radialCoeffs[2]));
    const float icD_denominator = 1.f + r2 * (sensorParams.radialCoeffs[3] + r2 * (sensorParams.radialCoeffs[4] + r2 * sensorParams.radialCoeffs[5]));
    const float icD             = icD_numerator / icD_denominator;

    const tcnn::vec2 delta = tcnn::vec2{
        sensorParams.tangentialCoeffs[0] * a1 + sensorParams.tangentialCoeffs[1] * a2 + r2 * (sensorParams.thinPrismCoeffs[0] + r2 * sensorParams.thinPrismCoeffs[1]),
        sensorParams.tangentialCoeffs[0] * a3 + sensorParams.tangentialCoeffs[1] * a1 + r2 * (sensorParams.thinPrismCoeffs[2] + r2 * sensorParams.thinPrismCoeffs[3])};

    // Project using ideal pinhole model (apply radial / tangential / thin-prism distortions)
    // in case radial distortion is within limits
    const tcnn::vec2 uvND = icD * uvNormalized + delta;

    constexpr float kMinRadialDist = 0.8f, kMaxRadialDist = 1.2f;
    const bool validRadial = (icD > kMinRadialDist) && (icD < kMaxRadialDist);
    if (validRadial) {
        projected = uvND * sensorParams.focalLength;
    } else {
        // If the radial distortion is out-of-limits, the computed coordinates will be unreasonable
        // (might even flip signs) - check on which side of the image we overshoot, and set the coordinates
        // out of the image bounds accordingly. The coordinates will be clipped to
        // viable range and direction but the exact values cannot be trusted / are still invalid
        const float roiClippingRadius = hypotf(sensorParams.nominalResolution.x, sensorParams.nominalResolution.y);
        projected                     = (roiClippingRadius * rsqrtf(r2)) * uvNormalized;
    }

    projected += sensorParams.principalPoint;
    applyRenderingResolution(sensorParams.nominalResolution, resolution, projected);

    return validRadial && withinResolution(resolution, tolerance, projected);
}

static inline __device__ bool projectPoint(const OpenCVFisheyeProjectionParameters& sensorParams,
                                           const tcnn::vec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    float rho = stableNorm2(position);
    if (rho <= 0.f) {
        rho = __FLT_EPSILON__;
    }

    const float thetaFull = atan2f(rho, position.z);
    // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
    // In particular for omnidirectional cameras, this prevents points outside the FOV to be
    // wrongly projected to in-image-domain points because of badly constrained polynomials outside
    // the effective FOV (which is different to the image boundaries).
    //
    // These FOV-clamped projections will be marked as *invalid*
    const float theta = fminf(thetaFull, sensorParams.maxAngle);
    // Evaluate forward polynomial
    // (radial distances to the principal point in the normalized image domain (up to focal length scales))
    const float theta2 = theta * theta;
    const float delta =
        (theta * (evalPolyHorner<4>(sensorParams.radialCoeffs, theta2) * theta2 + 1.0f)) / rho;
    projected = sensorParams.focalLength * position.xy() * delta;

    projected += sensorParams.principalPoint;
    applyRenderingResolution(sensorParams.nominalResolution, resolution, projected);

    return (theta < sensorParams.maxAngle) && withinResolution(resolution, tolerance, projected);
}

template <int NumNewtonIterations = 20>
static inline __device__ bool projectPoint(const FThetaProjectionParameters& sensorParams,
                                           const tcnn::vec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {

    float rho = stableNorm2(position);
    if (rho <= 0.f) {
        rho = __FLT_EPSILON__;
    }

    const float thetaFull = atan2f(rho, position.z);
    // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
    // In particular for omnidirectional cameras, this prevents points outside the FOV to be
    // wrongly projected to in-image-domain points because of badly constrained polynomials outside
    // the effective FOV (which is different to the image boundaries).
    //
    // These FOV-clamped projections will be marked as *invalid*
    const float theta = fminf(thetaFull, sensorParams.maxAngle);

    // Evaluate forward polynomial (depending on reference direction), giving delta = f(theta) factor
    float delta = 0.f;
    if (sensorParams.referencePoly == FThetaProjectionParameters::PolynomialType::ANGLE_TO_PIXELDIST) {
        // angle-to-pixeldist polynomial (fw) is reference, evaluate it directly
        delta = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree>(sensorParams.angleToPixeldistPoly, theta);
    } else {
        if (sensorParams.referencePoly == FThetaProjectionParameters::PolynomialType::PIXELDIST_TO_ANGLE) {
            // pixeldist-to-angle polynomial (bw) is reference, evaluate its accurate inverse via Newton-based inversion
            delta = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree>(sensorParams.angleToPixeldistPoly, theta);
        } else {
            // start with a single Regula Falsi iteration to close in on the root in the valid range
            const float x1 = tcnn::length(sensorParams.nominalResolution) * .5f;
            const float p0 = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree>(sensorParams.pixeldistToAnglePoly, delta) - theta;
            const float p1 = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree>(sensorParams.pixeldistToAnglePoly, x1) - theta;
            delta          = x1 * p0 / (p0 - p1);
        }
        tcnn::vec<FThetaProjectionParameters::PolynomialDegree - 1> dPixeldistToAnglePoly;
#pragma unroll
        for (auto i = 1; i < FThetaProjectionParameters::PolynomialDegree; ++i) {
            dPixeldistToAnglePoly[i - 1] = i * sensorParams.pixeldistToAnglePoly[i];
        }
        for (auto i = 0; i < NumNewtonIterations; ++i) {
            const float dfdx     = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree - 1>(dPixeldistToAnglePoly, delta);
            const float residual = evalPolyHorner<FThetaProjectionParameters::PolynomialDegree>(sensorParams.pixeldistToAnglePoly, delta) - theta;
            const float grad     = residual / dfdx;
            delta -= grad;
            // Early stopping if the gradient is small
            if (fabs(grad) < 1e-6f) {
                break;
            }
        }
    }

    // Apply linear term A=[c,d;e,1] to f(theta)-weighted normalized 2d vectors, relative to principal point
    projected = (delta / rho) * tcnn::vec2{sensorParams.linear_cde[0] * position.x + sensorParams.linear_cde[1] * position.y,
                                           sensorParams.linear_cde[2] * position.x + position.y};
    projected += sensorParams.principalPoint;
    applyRenderingResolution(sensorParams.nominalResolution, resolution, projected);

    // FThetaCameraModelParameters are defined such that the image coordinate origin corresponds to
    // the center of the first pixel. To conform to the NCore CameraModel specification (having the image
    // coordinate origin aligned with the top-left corner of the first pixel) we therefore need to
    // offset the principal point by half a pixel. Please see NCore documentation for more information.
    projected += tcnn::vec2{0.5f};

    return (theta < sensorParams.maxAngle) && withinResolution(resolution, tolerance, projected);
}

template <bool Normalized = false>
static inline __device__ bool projectPoint(const RowOffsetStructuredSpinningLidarProjectionParameters& sensorParams,
                                           const tcnn::vec2& /*resolution*/,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {

    const tcnn::vec3 ray   = Normalized ? position : tcnn::normalize(position);
    const tcnn::vec2 angle = {asinf(ray.z), atan2f(ray.y, ray.x)}; // X is elevation, Y is azimuth

    projected = angle * static_cast<float>(RowOffsetStructuredSpinningLidarProjectionParameters::ANGLE_TO_PIXEL_SCALING_FACTOR);

#if 1
    // For readability
    const float& elevation = angle.x;
    const float& azimuth   = angle.y;

    // For fovStart and fovSpan, x is azimuth, y is elevation, different from the angle!!!!!!
    const float& azimuthStart   = sensorParams.fovStart.x;
    const float& azimuthSpan    = sensorParams.fovSpan.x;
    const float& elevationStart = sensorParams.fovStart.y;
    const float& elevationSpan  = sensorParams.fovSpan.y;

    const float relativeElevation = relativeClockRotation(elevationStart, elevation, SpinningDirection::CLOCK_WISE);
    const float relativeAzimuth   = relativeAngle(azimuthStart, azimuth, sensorParams.spin);

    const float toleranceElevation = tolerance * elevationSpan;
    const float toleranceAzimuth   = tolerance * azimuthSpan;
    return (relativeElevation <= elevationSpan + toleranceElevation) &&
           (relativeAzimuth <= azimuthSpan + toleranceAzimuth) &&
           (relativeElevation >= -toleranceElevation) &&
           (relativeAzimuth >= -toleranceAzimuth);
#else
    return true;
#endif
}

static inline __device__ bool projectPoint(const SensorProjectionModel& sensorModel,
                                           const tcnn::vec2& resolution,
                                           const tcnn::vec3& position,
                                           float tolerance,
                                           tcnn::vec2& projected) {
    tcnn::vec3 camRay = correctExternalDistortion(sensorModel, position);

    switch (sensorModel.modelType) {
    case SensorProjectionModel::OrthographicModel:
        return projectPoint(sensorModel.orthographicParams, resolution, camRay, tolerance, projected);
    case SensorProjectionModel::PerspectiveModel:
        return projectPoint(sensorModel.perspectiveParams, resolution, camRay, tolerance, projected);
    case SensorProjectionModel::OpenCVPinholeModel:
        return projectPoint(sensorModel.ocvPinholeParams, resolution, camRay, tolerance, projected);
    case SensorProjectionModel::OpenCVFisheyeModel:
        return projectPoint(sensorModel.ocvFisheyeParams, resolution, camRay, tolerance, projected);
    case SensorProjectionModel::FThetaModel:
        return projectPoint(sensorModel.fthetaParams, resolution, camRay, tolerance, projected);
    case SensorProjectionModel::RowOffsetStructuredSpinningLidarModel:
        return projectPoint(sensorModel.nreHesaiP128LidarParams, resolution, camRay, tolerance, projected);
    default:
        projected = tcnn::vec2::zero();
        return false;
    }
}

template <int NRollingShutterIterations>
static inline __device__ bool projectPointWithShutter(const tcnn::vec3& position,
                                                      const tcnn::vec2& resolution,
                                                      const SensorProjectionModel& sensorModel,
                                                      const nrend::TSensorState& sensorState,
                                                      float tolerance,
                                                      tcnn::vec2& projectedPosition) {

    const tcnn::vec3 tStart = sensorState.startPose.slice<0, 3>();
    const tcnn::quat qStart = tcnn::quat{sensorState.startPose[6], sensorState.startPose[3], sensorState.startPose[4], sensorState.startPose[5]};

    bool validProjection = projectPoint(sensorModel, resolution, tcnn::to_mat3(qStart) * position + tStart, tolerance, projectedPosition);
    if (sensorModel.shutterType == SensorProjectionModel::GlobalShutter) {
        return validProjection;
    }

    const tcnn::vec3 tEnd = sensorState.endPose.slice<0, 3>();
    const tcnn::quat qEnd = tcnn::quat{sensorState.endPose[6], sensorState.endPose[3], sensorState.endPose[4], sensorState.endPose[5]};

    if (!validProjection) {
        validProjection = projectPoint(sensorModel, resolution, tcnn::to_mat3(qEnd) * position + tEnd, tolerance, projectedPosition);
        if (!validProjection) {
            return false;
        }
    }

    // Compute the new timestamp and project again
#pragma unroll
    for (int i = 0; i < NRollingShutterIterations; ++i) {
        const float alpha = relativeShutterTime(sensorModel, resolution, projectedPosition);
        validProjection   = projectPoint(sensorModel,
                                         resolution,
                                         tcnn::to_mat3(tcnn::slerp(qStart, qEnd, alpha)) * position + tcnn::mix(tStart, tEnd, alpha),
                                         tolerance,
                                         projectedPosition);
    }

    return validProjection;
}

struct LidarRoundFloor {
    static inline __device__ float apply(float x) { return floorf(x); }
};
struct LidarRoundCeil {
    static inline __device__ float apply(float x) { return ceilf(x); }
};

template <typename RoundFn>
static inline __device__ int sampleDenseAz(float pixAz, float fovSpanPixAz, int cdfResolution) {
    int idx = static_cast<int>(RoundFn::apply(__fdiv_rn(pixAz, fovSpanPixAz) * cdfResolution));
    assert(0 <= idx);
    assert(idx <= cdfResolution);
    return idx;
}

template <typename RoundFn>
static inline __device__ int sampleDenseEl(float pixEl, float fovSpanPixEl, int cdfResolution) {
    int idx = static_cast<int>(RoundFn::apply(__fdiv_rn(pixEl, fovSpanPixEl) * cdfResolution));
    assert(0 <= idx);
    assert(idx <= cdfResolution);
    return idx;
}

template <typename RoundFn>
static inline __device__ int sampleTileAz(float pixAz, float fovSpanPixAz, int nBins) {
    int idx = static_cast<int>(RoundFn::apply(__fdiv_rn(pixAz, fovSpanPixAz) * nBins));
    assert(0 <= idx);
    assert(idx <= nBins);
    return idx;
}

static inline __device__ int sampleTileEl(int denseEl, const int* elevationCDFTable, int elevationCDFResolution) {
    assert(0 <= denseEl);
    assert(denseEl <= elevationCDFResolution);
    return elevationCDFTable[denseEl];
}

static inline __device__ int compute2DCumulativeSumTableRegionSize(int az0p, int el0p, int az1p, int el1p, int azimuthSpanPixel, int elevationSpanPixel, const int* __restrict__ outIntegral) {
    assert(0 <= az0p && az0p < azimuthSpanPixel + 1);
    assert(0 <= az1p && az1p < azimuthSpanPixel + 1);
    assert(0 <= el0p && el0p < elevationSpanPixel + 1);
    assert(0 <= el1p && el1p < elevationSpanPixel + 1);
    const int stride = elevationSpanPixel + 1;
    const int A      = outIntegral[az1p * stride + el1p];
    const int B      = outIntegral[az0p * stride + el1p];
    const int C      = outIntegral[az1p * stride + el0p];
    const int D      = outIntegral[az0p * stride + el0p];
    return A - B - C + D;
}

static inline __device__ bool hasAnyRaysInTile(const RowOffsetStructuredSpinningLidarProjectionParameters& sensorParams,
                                               const tcnn::ivec2& minDenseTileExtent,
                                               const tcnn::ivec2& maxDenseTileExtent) {
    const int az0 = minDenseTileExtent.y;
    const int az1 = maxDenseTileExtent.y;

    if (az0 >= az1) {
        return false;
    }

    const int raycdfSizeAz = sensorParams.azimuthCDFResolution;
    const int raycdfSizeEl = sensorParams.elevationCDFResolution;

    assert(0 <= az0 && az0 <= raycdfSizeAz);
    assert(0 <= az1 && az1 <= raycdfSizeAz);

    if (az0 <= 0 && az1 >= raycdfSizeAz) {
        return true;
    }

    const int numRays = compute2DCumulativeSumTableRegionSize(
        az0, minDenseTileExtent.x, az1, maxDenseTileExtent.x,
        raycdfSizeAz, raycdfSizeEl, sensorParams.denseRayMaskCDFTable);
    return numRays > 0;
}

static inline __device__ void projectionTileExtent(const RowOffsetStructuredSpinningLidarProjectionParameters& sensorParams,
                                                   const tcnn::ivec2& tileGrid,
                                                   const tcnn::vec2& position,
                                                   const tcnn::ivec2& extent,
                                                   tcnn::ivec2& minTileExtent,
                                                   tcnn::ivec2& maxTileExtent,
                                                   tcnn::ivec2& minDenseTileExtent,
                                                   tcnn::ivec2& maxDenseTileExtent) {
    minTileExtent      = {0, 0};
    maxTileExtent      = {0, 0};
    minDenseTileExtent = {0, 0};
    maxDenseTileExtent = {0, 0};

    constexpr float kToPixel = static_cast<float>(RowOffsetStructuredSpinningLidarProjectionParameters::ANGLE_TO_PIXEL_SCALING_FACTOR);

    const int raycdfSizeEl = sensorParams.elevationCDFResolution;
    const int raycdfSizeAz = sensorParams.azimuthCDFResolution;

    const float azimuthStartPixel   = kToPixel * sensorParams.fovStart.x;
    const float elevationStartPixel = kToPixel * sensorParams.fovStart.y;

    const float fovSpanPixEl  = kToPixel * sensorParams.fovSpan.y;
    const float fovSpanPixAz  = kToPixel * sensorParams.fovSpan.x;
    const float fullCirclePix = 2.f * tcnn::PI() * kToPixel;

    // X is elevation, Y is azimuth
    const float elevationPixel = position.x;
    const float azimuthPixel   = position.y;

    const float meanRelEl = relativeClockRotation(elevationStartPixel, elevationPixel, SpinningDirection::CLOCK_WISE);
    const float meanRelAz = relativeAngle</*PIXEL_SPACE=*/true>(azimuthStartPixel, azimuthPixel, sensorParams.spin);

    // Gaussian range (un-clamped in azimuth; may extend past [0, fovSpan])
    const float begAz    = meanRelAz - extent.y;
    const float endAzRaw = meanRelAz + extent.y;
    const float begEl    = fminf(fmaxf(meanRelEl - extent.x, 0.f), fovSpanPixEl);
    const float endEl    = fminf(fmaxf(meanRelEl + extent.x, 0.f), fovSpanPixEl);

    // Sample elevation
    const int minDenseEl = sampleDenseEl<LidarRoundFloor>(begEl, fovSpanPixEl, raycdfSizeEl);
    const int maxDenseEl = sampleDenseEl<LidarRoundCeil>(endEl, fovSpanPixEl, raycdfSizeEl);

    if (minDenseEl >= maxDenseEl) {
        return;
    }

    // Check full_cover before capping to 2*pi, since the cap can push end
    // below fov_span for wide FOVs even though the gaussian covers the full FOV.
    const bool fullCover = (begAz <= 0.f) && (endAzRaw >= fovSpanPixAz);
    const float endAz    = fminf(endAzRaw, begAz + fullCirclePix);

    const int nBinsAz = sensorParams.azimuthNBins;

    const bool underflows = (begAz < 0.f) && !fullCover;
    const bool overflows  = (endAz > fullCirclePix) && !fullCover;

    // Compute region A's pixel range (matching gsplat).
    // full_cover:  A = [0, fov_span)
    // underflows:  A = [0, end)
    // overflows:   A = [beg, fc)
    // inside:      A = [beg, end)
    float begApixAz, endApixAz;
    if (fullCover) {
        begApixAz = 0.f;
        endApixAz = fovSpanPixAz;
    } else if (underflows) {
        begApixAz = 0.f;
        endApixAz = endAz;
    } else if (overflows) {
        begApixAz = begAz;
        endApixAz = fullCirclePix;
    } else {
        begApixAz = begAz;
        endApixAz = endAz;
    }

    // Clamp pixel regions to [0, fovSpan].
    begApixAz = fminf(fmaxf(begApixAz, 0.f), fovSpanPixAz);
    endApixAz = fminf(fmaxf(endApixAz, 0.f), fovSpanPixAz);

    // Sample A azimuth.
    const int begAdense = sampleDenseAz<LidarRoundFloor>(begApixAz, fovSpanPixAz, raycdfSizeAz);
    const int endAdense = sampleDenseAz<LidarRoundCeil>(endApixAz, fovSpanPixAz, raycdfSizeAz);
    const bool hasRaysA = hasAnyRaysInTile(sensorParams, {minDenseEl, begAdense}, {maxDenseEl, endAdense});

    int begAtile = 0, endAtile = 0;
    if (hasRaysA) {
        begAtile = sampleTileAz<LidarRoundFloor>(begApixAz, fovSpanPixAz, nBinsAz);
        endAtile = sampleTileAz<LidarRoundCeil>(endApixAz, fovSpanPixAz, nBinsAz);
    }

    // Sample B azimuth -- only exists for underflows or overflows.
    // underflows: B = [beg+fc, fc),  overflows: B = [0, end-fc)
    bool hasRaysB = false;
    int begBtile = 0, endBtile = 0;
    if (underflows || overflows) {
        const float begBpixAz = fminf(fmaxf(underflows ? (begAz + fullCirclePix) : 0.f, 0.f), fovSpanPixAz);
        const float endBpixAz = fminf(fmaxf(underflows ? fullCirclePix : (endAz - fullCirclePix), 0.f), fovSpanPixAz);

        const int begBdense = sampleDenseAz<LidarRoundFloor>(begBpixAz, fovSpanPixAz, raycdfSizeAz);
        const int endBdense = sampleDenseAz<LidarRoundCeil>(endBpixAz, fovSpanPixAz, raycdfSizeAz);
        hasRaysB            = hasAnyRaysInTile(sensorParams, {minDenseEl, begBdense}, {maxDenseEl, endBdense});
        if (hasRaysB) {
            begBtile = sampleTileAz<LidarRoundFloor>(begBpixAz, fovSpanPixAz, nBinsAz);
            endBtile = sampleTileAz<LidarRoundCeil>(endBpixAz, fovSpanPixAz, nBinsAz);
        }
    }

    if (!hasRaysA && !hasRaysB) {
        return;
    }

    const bool periodicAz = fovSpanPixAz >= fullCirclePix;

    int tileMinAz = hasRaysA ? begAtile : 0;
    int tileMaxAz = hasRaysA ? endAtile : 0;

    if (periodicAz) {
        // Merge B into A by extending A across the 0/nBins seam.
        if (hasRaysB && underflows) {
            tileMinAz = begBtile - nBinsAz;
        }
        if (hasRaysB && overflows) {
            tileMaxAz = endBtile + nBinsAz;
        }
        // Cap to at most nBins wide to prevent double-counting at the seam.
        tileMinAz = max(tileMinAz, tileMaxAz - nBinsAz);
        tileMaxAz = min(tileMaxAz, tileMinAz + nBinsAz);
    }
    // non-periodic
    else {
        if (hasRaysA && hasRaysB && begBtile < endAtile && begAtile < endBtile) {
            // The two regions are adjacent. In non-periodic azimuth fovs,
            // this can only happen if the gaussian span == 2pi, including the whole fov.
            tileMinAz = 0;
            tileMaxAz = nBinsAz;
        } else {
            // TODO: Due to the fact that hfov <= 180 and gaussian az span <= 180,
            // There's no way that that the 2 regions are disjoint. But if we need
            // to support wider hfovs, this function will have to return 2 regions to be
            // rasterized.
            assert(!hasRaysA || !hasRaysB);
        }
    }

    assert(tileMaxAz - tileMinAz <= nBinsAz);
    assert(tileMinAz <= tileMaxAz);

    // [minDenseEl,maxDenseEl) is a half-open range.
    // Make sure that [tileMinEl,tileMaxEl) also is.
    // - If minDenseEl==maxDenseEl -> tileMinEl==tileMaxEl (no tiles)
    // - If minDenseEl<maxDenseEl -> tileMinEl<tileMaxEl. (at least one tile)
    const int tileMinEl = sampleTileEl(minDenseEl, sensorParams.elevationCDFTable, raycdfSizeEl);
    assert(maxDenseEl >= 1);
    const int tileMaxEl = min(sampleTileEl(maxDenseEl - 1, sensorParams.elevationCDFTable, raycdfSizeEl) + 1,
                              sensorParams.elevationNBins);

    assert(tileMinEl <= tileMaxEl);

    minTileExtent      = {tileMinEl, tileMinAz};
    maxTileExtent      = {tileMaxEl, tileMaxAz};
    minDenseTileExtent = {minDenseEl, begAdense};
    maxDenseTileExtent = {maxDenseEl, endAdense};
}

template <int DefaultTileWidth, int DefaultTileHeight>
static inline __device__ void projectionTileExtent(const SensorProjectionModel& sensorModel,
                                                   const tcnn::ivec2& tileGrid,
                                                   const tcnn::vec2& position,
                                                   const tcnn::ivec2& extent,
                                                   tcnn::ivec2& minTileExtent,
                                                   tcnn::ivec2& maxTileExtent,
                                                   tcnn::ivec2& minDenseTileExtent,
                                                   tcnn::ivec2& maxDenseTileExtent) {
    switch (sensorModel.modelType) {
    case SensorProjectionModel::RowOffsetStructuredSpinningLidarModel:
        projectionTileExtent(sensorModel.nreHesaiP128LidarParams, tileGrid, position, extent, minTileExtent, maxTileExtent, minDenseTileExtent, maxDenseTileExtent);
        break;
    default:
        minTileExtent = {
            min(tileGrid.x, max(0, static_cast<int>(floorf((position.x - 0.5f - extent.x) / DefaultTileWidth)))),
            min(tileGrid.y, max(0, static_cast<int>(floorf((position.y - 0.5f - extent.y) / DefaultTileHeight)))),
        };
        maxTileExtent = {
            min(tileGrid.x, max(0, static_cast<int>(ceilf((position.x - 0.5f + extent.x) / DefaultTileWidth)))),
            min(tileGrid.y, max(0, static_cast<int>(ceilf((position.y - 0.5f + extent.y) / DefaultTileHeight)))),
        };
    }
}

static inline __device__ uint32_t projectionDilationFactor(const SensorProjectionModel& sensorModel) {
    // Lidar projection is angular (TODO define a meaningfull factor)
    return (sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) ? 0.f : 1.f;
}

template <bool zCulling>
static inline __device__ bool nearFarClipPosition(const tcnn::vec2& nearFarClipDistances,
                                                  const tcnn::mat4x3& toSensorMatrix,
                                                  const tcnn::vec3& position) {
    if constexpr (zCulling) {
        const float zPosition = (position.x * toSensorMatrix[0][2] + position.y * toSensorMatrix[1][2] + position.z * toSensorMatrix[2][2] + toSensorMatrix[3][2]);
        return (nearFarClipDistances.x > zPosition) || (nearFarClipDistances.y < zPosition);
    } else {
        const float distance = tcnn::length(toSensorMatrix * vec4(position, 1.f));
        return (nearFarClipDistances.x > distance) || (nearFarClipDistances.y < distance);
    }
}

template <bool zCulling>
static inline __device__ bool sensorCullPosition(const SensorProjectionModel& sensorModel,
                                                 const tcnn::mat4x3& toSensorMatrix,
                                                 const tcnn::vec2& nearFarClipDistances,
                                                 const tcnn::vec3& position) {
    if constexpr (zCulling) {
        return sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel ? false : nearFarClipPosition<zCulling>(nearFarClipDistances, toSensorMatrix, position);
    } else {
        return nearFarClipPosition<zCulling>(nearFarClipDistances, toSensorMatrix, position);
    }
}

static inline __device__ bool sensorSupportsTileCulling(const SensorProjectionModel& sensorModel) {
    return sensorModel.modelType != SensorProjectionModel::RowOffsetStructuredSpinningLidarModel;
}

static inline __device__ int sensorWrapAzimuthTileIfLidar(const nrend::TSensorModel& sensorModel, int azTile, int azTileGridSize) {
    if (sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
        azTile %= azTileGridSize;
        if (azTile < 0) {
            azTile += azTileGridSize;
        }
    }
    assert(0 <= azTile && azTile < azTileGridSize);

    return azTile;
}

} // namespace nrend
