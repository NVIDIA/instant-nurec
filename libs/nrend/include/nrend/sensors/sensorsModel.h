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

#include <tiny-cuda-nn/vec.h>

namespace nrend {

/**
 * @brief Orthographic projection parameters
 * @note : in viewport resolution
 */
struct OrthographicProjectionParameters {
};

/**
 * @brief Perspective projection parameters
 * @note : in viewport resolution
 */
struct PerspectiveProjectionParameters {
    tcnn::vec2 principalPoint;
    tcnn::vec2 focalLength;
};

/**
 * @brief OpenCV pinhole with radial distortion projection parameters
 * @note : in nominal resolution
 */
struct OpenCVPinholeProjectionParameters {
    tcnn::vec2 nominalResolution;
    tcnn::vec2 principalPoint;
    tcnn::vec2 focalLength;
    tcnn::vec<6> radialCoeffs;
    tcnn::vec2 tangentialCoeffs;
    tcnn::vec4 thinPrismCoeffs;
};

/**
 * @brief OpenCV fisheye projection parameters
 * @note : in nominal resolution
 */
struct OpenCVFisheyeProjectionParameters {
    tcnn::vec2 nominalResolution;
    tcnn::vec2 principalPoint;
    tcnn::vec2 focalLength;
    tcnn::vec4 radialCoeffs;
    float maxAngle;
};

/**
 * @brief FTheta projection parameters
 * @note : in nominal resolution
 */
struct FThetaProjectionParameters {
    tcnn::vec2 nominalResolution;
    tcnn::vec2 principalPoint;
    enum PolynomialType {
        PIXELDIST_TO_ANGLE,
        ANGLE_TO_PIXELDIST,
        ///< start with Regula Falsi instead of angleToPixeldistPoly
        PIXELDIST_TO_ANGLE_RF
    } referencePoly;
    static constexpr size_t PolynomialDegree = 6;
    tcnn::vec<PolynomialDegree> pixeldistToAnglePoly; // backward polynomial
    tcnn::vec<PolynomialDegree> angleToPixeldistPoly; // forward polynomial
    float maxAngle;
    tcnn::vec<3> linear_cde; // Coefficients of the constrained linear term :math:`\begin{bmatrix} c & d \\ e & 1 \end{bmatrix}` transforming between sensor coordinates (in mm) to image coordinates (in px) (float32, [3,])
};

/**
 * @brief Bivariate windshield distortion parameters
 */
struct BivariateWindshieldDistortionParameters {
    static constexpr int32_t MaxPolyOrder = 5; // max poly order should be increased as requires
    static constexpr int32_t MaxPolySize  = (MaxPolyOrder + 2) * (MaxPolyOrder + 1) / 2;
    const float* horizontalPoly;
    const float* verticalPoly;
    int32_t horizontalPolyOrder;
    int32_t verticalPolyOrder;
};

/**
 * @brief Generalized projection parameters
 * @note : NDC resolution
 */
struct GeneralizedProjectionParameters {
    const tcnn::vec3* projectionMap;
    const tcnn::vec2* octahedralUnprojectMap;
    // TODO : add nominal width/heigh, optical center, fov
};

enum SpinningDirection {
    CLOCK_WISE,
    COUNTER_CLOCK_WISE
};

/**
 * @brief Row offset structured spinning lidar projection parameters
 * @note : fixed resolution
 */
struct RowOffsetStructuredSpinningLidarProjectionParameters {
    SpinningDirection spin;
    int32_t nRows;
    int32_t nColumns;
    tcnn::vec2 fovStart; ///< Start of (azimuth, elevation) field-of-view range (radians)
    tcnn::vec2 fovSpan;  ///< Span of (azimuth, elevation) field-of-view range (radians)
    int32_t azimuthNBins;
    int32_t elevationNBins;
    int32_t maxPtsPerTile;
    const tcnn::ivec2* tilesPackInfo;      ///< Device data [elevationNBins x azimuthNBins]
    const tcnn::ivec2* tilesToElementsMap; ///< Device data [nRows x nColumns]
    int32_t elevationCDFResolution;
    int32_t azimuthCDFResolution;
    const int* elevationCDFTable;    ///< Device data of the elevation CDF table [elevationCDFResolution + 1]
    const int* denseRayMaskCDFTable; ///< Device buffer of size (azimuthCDFResolution + 1), (elevationCDFResolution + 1)
    int angleToColumnMapResolutionFactor;
    tcnn::vec2 mapResolution;    ///< size of the map pixel in radians (x = horizontal, y = vertical)
    const int* angleToColumnMap; ///< Device LUT from angle to column [nRows x nColumns x angleToColumnMapResolutionFactor^2]

    static constexpr int32_t ANGLE_TO_PIXEL_SCALING_FACTOR = 1024;
};

/**
 * @brief Sensor projection model variants
 */
struct SensorProjectionModel {

    enum ShutterType {
        RollingTopToBottomShutter,
        RollingLeftToRightShutter,
        RollingBottomToTopShutter,
        RollingRightToLeftShutter,
        GlobalShutter
    } shutterType = GlobalShutter;

    enum ModelType {
        OrthographicModel,
        PerspectiveModel,
        OpenCVPinholeModel,
        OpenCVFisheyeModel,
        FThetaModel,
        RowOffsetStructuredSpinningLidarModel,
        GeneralizedModel,
        EmptyModel,
        Unsupported
    } modelType = EmptyModel;

    union {
        OrthographicProjectionParameters orthographicParams;
        PerspectiveProjectionParameters perspectiveParams;
        OpenCVPinholeProjectionParameters ocvPinholeParams;
        OpenCVFisheyeProjectionParameters ocvFisheyeParams;
        FThetaProjectionParameters fthetaParams;
        RowOffsetStructuredSpinningLidarProjectionParameters nreHesaiP128LidarParams;
        GeneralizedProjectionParameters generalizedParams;
    };

    enum ExternalDistortionType {
        EmptyExternalDistortionModel,
        BivariateWindshieldDistortion,
        UnsupportedExternalDistortion
    } externalDistortionType = EmptyExternalDistortionModel;

    union {
        BivariateWindshieldDistortionParameters bivariateWindshieldDistortionParameters;
    };
};

} // namespace nrend
