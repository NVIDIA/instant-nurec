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

#include "rolling_shutter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

// ---------------------------------------------------------------------------------------------
// Camera-specific types (camera model parameters and returns)

enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};

enum class ReferencePolynomial {
    FORWARD,
    BACKWARD
};

struct BivariateWindshieldModelParameters {
    ReferencePolynomial reference_poly;
    std::vector<float> horizontal_poly;
    std::vector<float> vertical_poly;
    std::vector<float> horizontal_poly_inverse;
    std::vector<float> vertical_poly_inverse;
    const float* horizontal_poly_buffer = nullptr; // device buffer of size horizontal_poly.size()
    const float* vertical_poly_buffer   = nullptr; // device buffer of size horizontal_poly.size()
};

using ExternalDistortionParametersVariant = std::variant<BivariateWindshieldModelParameters, std::monostate>;

struct CameraModelParameters {
    std::array<uint64_t, 2> resolution;
    ShutterType shutter_type;
    ExternalDistortionParametersVariant external_distortion_parameters = std::monostate{};
};

struct OpenCVPinholeCameraModelParameters : CameraModelParameters {
    std::array<float, 2> principal_point;
    std::array<float, 2> focal_length;
    std::array<float, 6> radial_coeffs;
    std::array<float, 2> tangential_coeffs;
    std::array<float, 4> thin_prism_coeffs;

    auto is_perfect_pinhole() const -> bool {
        auto const is_all_zero = [](auto const& arr) {
            return std::all_of(arr.begin(), arr.end(), [](auto const& value) {
                return std::abs(value) < std::numeric_limits<float>::epsilon();
            });
        };

        return is_all_zero(radial_coeffs) &&
               is_all_zero(tangential_coeffs) &&
               is_all_zero(thin_prism_coeffs);
    }
};

struct OpenCVFisheyeCameraModelParameters : CameraModelParameters {
    std::array<float, 2> principal_point;
    std::array<float, 2> focal_length;
    std::array<float, 4> radial_coeffs;
    float max_angle;
};

struct FThetaCameraModelParameters : CameraModelParameters {
    enum class PolynomialType {
        PIXELDIST_TO_ANGLE,
        ANGLE_TO_PIXELDIST,
    };
    std::array<float, 2> principal_point;
    PolynomialType reference_poly;
    static constexpr size_t PolynomialDegree = 6;
    std::array<float, PolynomialDegree> pixeldist_to_angle_poly; // backward polynomial
    std::array<float, PolynomialDegree> angle_to_pixeldist_poly; // forward polynomial
    float max_angle;
    std::array<float, 3> linear_cde;
};

using CameraModelParametersVariant = std::variant<OpenCVPinholeCameraModelParameters,
                                                  OpenCVFisheyeCameraModelParameters,
                                                  FThetaCameraModelParameters>;
