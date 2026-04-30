// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "cameras.h"

#include "rolling_shutter.cuh"
#include <ku/helper_math.cuh>

#include "overload_visitor.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

// ---------------------------------------------------------------------------------------------

// Math helpers (polynomial evaluation / stable norms)

inline __device__ float numerically_stable_norm2(float x, float y) {
    // Computes 2-norm of a [x,y] vector in a numerically stable way
    auto const abs_x = fabsf(x);
    auto const abs_y = fabsf(y);
    auto const min   = fminf(abs_x, abs_y);
    auto const max   = fmaxf(abs_x, abs_y);

    if (max <= 0.f)
        return 0.f;

    auto const min_max_ratio = min / max;
    return max * sqrtf(1.f + min_max_ratio * min_max_ratio);
}

template <size_t N_COEFFS>
inline __device__ float eval_poly_horner(std::array<float, N_COEFFS> const& poly, float x) {
    // Evaluates a polynomial y=f(x) with
    //
    // f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable Horner scheme.
    //
    // The degree of the polynomial is N_COEFFS - 1

    auto y = float{0};
    for (auto cit = poly.rbegin(); cit != poly.rend(); ++cit)
        y = x * y + (*cit);
    return y;
}

template <size_t N_COEFFS>
inline __device__ float eval_poly_odd_horner(std::array<float, N_COEFFS> const& poly_odd, float x) {
    // Evaluates an odd-only polynomial y=f(x) with
    //
    // f(x) = c_0*x^1 + c_1*x^3 + c_2*x^5 + c_3*x^7 + c_4*x^9 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable Horner scheme.
    //
    // The degree of the polynomial is 2*N_COEFFS - 1

    return x * eval_poly_horner(poly_odd, x * x); // evaluate x^2-based "regular" polynomial after facting out one x term
}

template <size_t N_COEFFS>
inline __device__ float eval_poly_even_horner(std::array<float, N_COEFFS> const& poly_even, float x) {
    // Evaluates an even-only polynomial y=f(x) with
    //
    // f(x) = c_0 + c_1*x^2 + c_2*x^4 + c_3*x^6 + c_4*x^8 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable Horner scheme.
    //
    // The degree of the polynomial is 2*(N_COEFFS - 1)

    return eval_poly_horner(poly_even, x * x); // evaluate x^2-substituted "regular" polynomial
}

// Enum to represent the type of polynomial
enum class PolynomialType {
    FULL, // Represents a full polynomial with all terms
    EVEN, // Represents an even-only polynomial
    ODD   // Represents an odd-only polynomial
};

template <PolynomialType POLYNOMIAL_TYPE, size_t N_COEFFS>
struct PolynomialProxy {
    std::array<float, N_COEFFS> const& coeffs;

    // Evaluate the polynomial using Horner's method based on the polynomial type
    inline __device__ float eval_horner(float x) const {
        if constexpr (POLYNOMIAL_TYPE == PolynomialType::FULL) {
            // Evaluate a full polynomial
            return eval_poly_horner(coeffs, x);
        } else if constexpr (POLYNOMIAL_TYPE == PolynomialType::EVEN) {
            // Evaluate an even-only polynomial
            return eval_poly_even_horner(coeffs, x);
        } else if constexpr (POLYNOMIAL_TYPE == PolynomialType::ODD) {
            // Evaluate an odd-only polynomial
            return eval_poly_odd_horner(coeffs, x);
        }
    }
};

template <size_t N_NEWTON_ITERATIONS, class PolyProxy, class DPolyProxy, class TInvPolyApproxProxy>
inline __device__ float eval_poly_inverse_horner_newton(PolyProxy const& poly,
                                                        DPolyProxy const& dpoly,
                                                        TInvPolyApproxProxy const& inv_poly_approx,
                                                        float y) {
    // Evaluates the inverse x = f^{-1}(y) of a reference polynomial y=f(x) (given by poly_coefficients) at points y
    // using numerically stable Horner scheme and Newton iterations starting from an approximate solution \\hat{x} = \\hat{f}^{-1}(y)
    // (given by inv_poly_approx) and the polynomials derivative df/dx (given by poly_derivative_coefficients)

    static_assert(N_NEWTON_ITERATIONS >= 0, "Require at least a single Newton iteration");

    // approximation / starting points - also returned for zero iterations
    auto x = inv_poly_approx.eval_horner(y);

#pragma unroll
    for (auto j = 0; j < N_NEWTON_ITERATIONS; ++j) {
        auto const dfdx     = dpoly.eval_horner(x);
        auto const residual = poly.eval_horner(x) - y;
        x -= residual / dfdx;
    }

    return x;
}

/**
 * @brief Checks if a given image point is within the image bounds considering a margin.
 *
 * This function determines whether a specified point in image coordinates lies within the bounds
 * of the image, taking into account a margin factor. The margin is calculated as a fraction of the
 * image resolution.
 *
 * @param image_point The point in image coordinates to check.
 * @param resolution The resolution of the image as an array where resolution[0] is the width and resolution[1] is the height.
 * @param margin_factor The factor by which the margin is calculated. The margin is computed as margin_factor * resolution.
 * @return true if the image point is within the image bounds considering the margin, false otherwise.
 */
__forceinline__ __device__ __host__ bool
image_point_in_image_bounds_margin(float2 const& image_point, std::array<uint64_t, 2> const& resolution, float margin_factor) {
    const float MARGIN_X = resolution[0] * margin_factor;
    const float MARGIN_Y = resolution[1] * margin_factor;
    bool valid           = true;
    valid &= (-MARGIN_X) <= image_point.x && image_point.x < (resolution[0] + MARGIN_X);
    valid &= (-MARGIN_Y) <= image_point.y && image_point.y < (resolution[1] + MARGIN_Y);
    return valid;
}

// ---------------------------------------------------------------------------------------------

// External Distortion Models

template <class DerivedExternalDistortion>
struct ExternalDistortion {
    // CRTP base class for all external distortion types
    inline __device__ float3 distort_camera_rays(float3 ray_in) const {
        return static_cast<DerivedExternalDistortion const*>(this)->distort_camera_rays_impl(ray_in);
    }
    inline __device__ float3 undistort_camera_rays(float3 cam_ray) const {
        return static_cast<DerivedExternalDistortion const*>(this)->undistort_camera_rays_impl(cam_ray);
    }
};

struct BivariateWindshieldModel : ExternalDistortion<BivariateWindshieldModel> {
    BivariateWindshieldModel(BivariateWindshieldModelParameters const& params) {
        horizontal_poly_order = compute_poly_order(params.horizontal_poly.size(), params.horizontal_poly);
        vertical_poly_order   = compute_poly_order(params.vertical_poly.size(), params.vertical_poly);
        if (horizontal_poly_order > N_MAX_POLY_ORDER || vertical_poly_order > N_MAX_POLY_ORDER) {
            throw std::runtime_error("A BivariateWindhsieldModel contains polynomials of larger degree than is permitted. Expected maximum number of terms to be " +
                                     std::to_string(N_MAX_POLY_SIZE) + ". Please increase compile-time limit");
        }
        std::copy(params.horizontal_poly.begin(), params.horizontal_poly.end(), horizontal_poly.begin());
        std::copy(params.vertical_poly.begin(), params.vertical_poly.end(), vertical_poly.begin());
        std::copy(params.horizontal_poly_inverse.begin(), params.horizontal_poly_inverse.end(), horizontal_poly_inverse.begin());
        std::copy(params.vertical_poly_inverse.begin(), params.vertical_poly_inverse.end(), vertical_poly_inverse.begin());
    }

    static constexpr auto N_MAX_POLY_ORDER = 5; // max poly order should be increased as requires
    static constexpr auto N_MAX_POLY_SIZE  = (N_MAX_POLY_ORDER + 2) * (N_MAX_POLY_ORDER + 1) / 2;
    std::array<float, N_MAX_POLY_SIZE> horizontal_poly;
    std::array<float, N_MAX_POLY_SIZE> vertical_poly;
    std::array<float, N_MAX_POLY_SIZE> horizontal_poly_inverse;
    std::array<float, N_MAX_POLY_SIZE> vertical_poly_inverse;
    int32_t horizontal_poly_order;
    int32_t vertical_poly_order;

    static int32_t compute_poly_order(int32_t max_degree, std::vector<float> const& poly) {
        auto poly_deg = 0;
        for (auto deg = 0; deg < max_degree; deg++) {
            auto const num_terms = (deg + 1) * (deg + 2) / 2;
            if (num_terms == poly.size()) {
                poly_deg = deg;
                break;
            } else if (num_terms > poly.size()) {
                throw std::runtime_error("windshield polynomial has invalid number of coefficients");
            }
        }
        return poly_deg;
    };

    static inline __device__ auto poly_eval_2d(const std::array<float, N_MAX_POLY_SIZE>& poly_2d, int32_t order, float x, float y) {
        auto const horner_range = [](const auto& poly, auto x, auto idx_start, auto idx_end) {
            auto result = 0.f;
            for (auto idx = idx_end - 1; idx >= idx_start; idx--) {
                result = result * x + poly[idx];
            }
            return result;
        };

        auto outer_coeffs = std::array<float, N_MAX_POLY_ORDER>{};
        auto start_idx    = 0;
        for (auto inner_order = order; inner_order >= 0; inner_order--) {
            outer_coeffs[order - inner_order] = horner_range(poly_2d, x, start_idx, start_idx + inner_order + 1);
            start_idx += inner_order + 1;
        }
        return horner_range(outer_coeffs, y, 0, order + 1);
    }

    static inline __device__ auto distort_ray(
        std::array<float, N_MAX_POLY_SIZE> const& poly_phi,
        std::array<float, N_MAX_POLY_SIZE> const& poly_theta,
        int32_t order_phi,
        int32_t order_theta,
        float3 ray_in) {
        // Evaluates windshield distortion model
        auto const ray_length = length(ray_in);
        auto const phi        = std::asin(ray_in.x / ray_length);
        auto const theta      = std::asin(ray_in.y / ray_length);
        auto const x          = std::sin(poly_eval_2d(poly_phi, order_phi, phi, theta));
        auto const y          = std::sin(poly_eval_2d(poly_theta, order_theta, phi, theta));
        auto const z          = std::sqrt(1.f - std::clamp(x * x + y * y, 0.f, 1.f)) * (ray_in.z < 0.f ? -1.f : 1.f);
        return make_float3(x, y, z);
    }

    inline __device__ float3 distort_camera_rays_impl(float3 ray_in) const {
        // Applies distortion to camera rays in forward direction, from external to internal
        return distort_ray(horizontal_poly, vertical_poly, horizontal_poly_order, vertical_poly_order, ray_in);
    }

    inline __device__ float3 undistort_camera_rays_impl(float3 cam_ray) const {
        // Applies distortion to camera rays in backward direction, from external to internal
        return distort_ray(horizontal_poly_inverse, vertical_poly_inverse, horizontal_poly_order, vertical_poly_order, cam_ray);
    }
};

using ExternalDistortionVariant = std::variant<BivariateWindshieldModel, std::monostate>;

// ---------------------------------------------------------------------------------------------

// Camera models

template <class DerivedCameraModel>
struct BaseCameraModel {
    // CRTP base class for all camera model types
protected:
    BaseCameraModel(CameraModelParameters const& camera_model_parameters) {
        // Register optional external distortion model
        std::visit(
            OverloadVisitor{
                [&](std::monostate const& params) { external_distortion_variant = std::monostate{}; },
                [&](BivariateWindshieldModelParameters const& params) { external_distortion_variant = BivariateWindshieldModel(params); },
            },
            camera_model_parameters.external_distortion_parameters);
    }

public:
    struct CameraRayToImagePointReturn {
        float2 image_point;
        bool valid_flag;
    };

    struct WorldPointToImagePointReturn {
        float2 image_point;
        bool valid_flag;
        int64_t timestamp_us;
        Pose3 T_world_sensor;
    };

    struct ImagePointToWorldRayReturn {
        Ray3 world_ray;
        int64_t timestamp_us;
        Pose3 T_sensor_world;
    };

    // Function computes the image point from a camera ray. If external distortion is present, it is corrected for.
    inline __device__ CameraRayToImagePointReturn
    camera_ray_to_image_point(float3 cam_ray, float margin_factor = 0.0) const {
        auto derived = static_cast<DerivedCameraModel const*>(this);
        if (!std::holds_alternative<std::monostate>(external_distortion_variant)) {
            const auto& external_distortion = std::get<BivariateWindshieldModel>(external_distortion_variant);
            cam_ray                         = external_distortion.distort_camera_rays(cam_ray);
        }
        return derived->camera_ray_to_image_point_impl(cam_ray, margin_factor);
    }

    // Function computes a camera ray given an image point. If external distortion is present, it is corrected for.
    inline __device__ float3
    image_point_to_camera_ray(float2 image_point) const {
        auto derived = static_cast<DerivedCameraModel const*>(this);
        auto cam_ray = derived->image_point_to_camera_ray_impl(image_point);
        if (!std::holds_alternative<std::monostate>(external_distortion_variant)) {
            const auto& external_distortion = std::get<BivariateWindshieldModel>(external_distortion_variant);
            cam_ray                         = external_distortion.undistort_camera_rays(cam_ray);
        }
        return cam_ray;
    }

    // Function to compute the relative frame time for a given image point based on the shutter type
    inline __device__ float
    shutter_relative_frame_time(float2 const& image_point) const {
        auto derived = static_cast<DerivedCameraModel const*>(this);

        auto relative_frame_time = 0.f;

        auto const& resolution = derived->parameters.resolution;
        switch (derived->parameters.shutter_type) {
        case ShutterType::ROLLING_TOP_TO_BOTTOM:
            relative_frame_time = std::floor(image_point.y) / (resolution[1] - 1);
            break;

        case ShutterType::ROLLING_LEFT_TO_RIGHT:
            relative_frame_time = std::floor(image_point.x) / (resolution[0] - 1);
            break;

        case ShutterType::ROLLING_BOTTOM_TO_TOP:
            relative_frame_time = (resolution[1] - std::ceil(image_point.y)) / (resolution[1] - 1);
            break;

        case ShutterType::ROLLING_RIGHT_TO_LEFT:
            relative_frame_time = (resolution[0] - std::ceil(image_point.x)) / (resolution[0] - 1);
            break;
        }

        return relative_frame_time;
    };

    inline __device__ Ray3
    image_point_to_world_ray_shutter_pose(float2 const& image_point, RollingShutter const& rolling_shutter) const {
        // Unproject ray and transform to world using shutter pose
        auto const derived       = static_cast<DerivedCameraModel const*>(this);
        auto const camera_ray    = derived->image_point_to_camera_ray(image_point);
        auto const relative_time = shutter_relative_frame_time(image_point);
        return rolling_shutter.sensor_ray_to_world_ray(relative_time, camera_ray);
    };

    inline __device__ ImagePointToWorldRayReturn
    image_point_to_world_ray_shutter_pose(float2 const& image_point,
                                          Pose3 const& T_sensor_world_start,
                                          Pose3 const& T_sensor_world_end,
                                          std::array<uint64_t, 2> const& timestamps_us) const {
        // Unproject ray and transform to world using shutter pose
        auto derived                   = static_cast<DerivedCameraModel const*>(this);
        auto const camera_ray          = derived->image_point_to_camera_ray(image_point);
        auto const relative_frame_time = shutter_relative_frame_time(image_point);
        // Interpolate the poses and timestamps
        auto const T_sensor_world = interpolate_pose(relative_frame_time, T_sensor_world_start, T_sensor_world_end);
        auto const timestamp_us   = interpolate_timestamp_us(relative_frame_time, timestamps_us);
        // Transform the camera ray to world ray
        auto const world_ray = T_sensor_world.transform_local_ray(camera_ray);
        return {world_ray, timestamp_us, T_sensor_world};
    }

    template <size_t N_ROLLING_SHUTTER_ITERATIONS>
    inline __device__ WorldPointToImagePointReturn
    world_point_to_image_point_shutter_pose(float3 const& world_point, RollingShutterParameters const& rolling_shutter_parameters, float margin_factor) const {
        // Perform rolling-shutter-based world point to image point projection / optimization
        auto derived = static_cast<DerivedCameraModel const*>(this);

        auto const& frame_T_world_sensors = rolling_shutter_parameters.T_world_sensors;

        auto const t_start = make_float3(frame_T_world_sensors[0],
                                         frame_T_world_sensors[1],
                                         frame_T_world_sensors[2]);
        auto const q_start = make_float4(frame_T_world_sensors[3],
                                         frame_T_world_sensors[4],
                                         frame_T_world_sensors[5],
                                         frame_T_world_sensors[6]);
        auto const t_end   = make_float3(frame_T_world_sensors[7],
                                         frame_T_world_sensors[8],
                                         frame_T_world_sensors[9]);
        auto const q_end   = make_float4(frame_T_world_sensors[10],
                                         frame_T_world_sensors[11],
                                         frame_T_world_sensors[12],
                                         frame_T_world_sensors[13]);

        // Always perform transformation using start pose
        auto const [image_point_start, valid_start] = derived->camera_ray_to_image_point(apply_quaternion(q_start, world_point) + t_start, margin_factor);

        if (derived->parameters.shutter_type == ShutterType::GLOBAL) {
            // Exit early if we have a global shutter sensor
            return {
                image_point_start,
                valid_start,
                rolling_shutter_parameters.timestamps_us[0],
                {
                    t_start,
                    q_start,
                },
            };
        }

        // Do initial transformations using both start and end poses to determine all candidate
        // points and take union of valid projections as iteration starting points
        auto const [image_point_end, valid_end] = derived->camera_ray_to_image_point(apply_quaternion(q_end, world_point) + t_end, margin_factor);

        // This selection prefers points at the start-of-frame pose over end-of-frame points
        // - the optimization will determine the final timestamp for each point
        auto init_image_point = float2{};
        if (valid_start) {
            init_image_point = image_point_start;
        } else if (valid_end) {
            init_image_point = image_point_end;
        } else {
            // No valid projection at start or finish -> mark point as invalid. Still
            // return projection result at end of frame to be consistent with ncore, as
            // this will be condensed at the python interface level
            return {
                image_point_end,
                false,
                rolling_shutter_parameters.timestamps_us[1],
                {
                    t_end,
                    q_end,
                },
            };
        }

        // Compute the new timestamp and project again
        auto image_points_rs_prev = init_image_point;
        auto valid_rs_prev        = true;
        auto relative_frame_time  = float{};
        auto t_rs                 = float3{};
        auto q_rs                 = float4{};

#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            relative_frame_time = shutter_relative_frame_time(image_points_rs_prev);

            t_rs = (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
            q_rs = unitquat_slerp(q_start, q_end, relative_frame_time);

            auto const [image_point_rs, valid_rs] = derived->camera_ray_to_image_point(apply_quaternion(q_rs, world_point) + t_rs, margin_factor);

            image_points_rs_prev = image_point_rs;
            valid_rs_prev        = valid_rs;
        }

        const auto timestamp_us_rs = rolling_shutter_parameters.timestamps_us[0] + int64_t(relative_frame_time * (rolling_shutter_parameters.timestamps_us[1] - rolling_shutter_parameters.timestamps_us[0]));

        return {
            image_points_rs_prev,
            valid_rs_prev,
            timestamp_us_rs,
            {
                t_rs,
                q_rs,
            },
        };
    }

    // Default external distortion to empty (monostate)
    ExternalDistortionVariant external_distortion_variant = std::monostate{};
};

struct PerfectPinholeCameraModel : BaseCameraModel<PerfectPinholeCameraModel> {
    // OpenCV-like pinhole camera model without any distortion (NCore conventions)
public:
    using Base = BaseCameraModel<PerfectPinholeCameraModel>;

    struct Parameters : CameraModelParameters {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;
    };

    Parameters parameters;

    PerfectPinholeCameraModel(Parameters const& parameters)
        : Base(parameters)
        , parameters(parameters) {}

    inline __device__ auto camera_ray_to_image_point_impl(float3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        auto image_point = make_float2(0.f, 0.f);

        // Treat all the points behind the camera plane to invalid / projecting to origin (NCore convention)
        if (cam_ray.z <= 0.f)
            return {image_point, false};

        // Project using ideal pinhole model
        image_point = (make_float2(cam_ray.x, cam_ray.y) / cam_ray.z) *
                          make_float2(parameters.focal_length[0], parameters.focal_length[1]) +
                      make_float2(parameters.principal_point[0], parameters.principal_point[1]);

        // Check if the image points fall within the image, set points that have too large distortion or fall outside the image sensor to invalid
        auto valid = true;
        valid &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);

        return {image_point, valid};
    }

    inline __device__ float3 image_point_to_camera_ray_impl(float2 image_point) const {
        // Transform the image point to uv coordinate
        auto const uv = (image_point - make_float2(parameters.principal_point[0], parameters.principal_point[1])) /
                        make_float2(parameters.focal_length[0], parameters.focal_length[1]);

        // Unproject the image point to camera ray
        auto const camera_ray = make_float3(uv.x, uv.y, 1.f);

        // Make sure ray is normalized
        return camera_ray / length(camera_ray);
    }
};

template <size_t N_MAX_UNDISTORTION_ITERATIONS = 5 /* half the number of maximum iterations as in NCore reference model (currently using 10)*/>
struct OpenCVPinholeCameraModel : BaseCameraModel<OpenCVPinholeCameraModel<N_MAX_UNDISTORTION_ITERATIONS>> {
    // OpenCV-compatible pinhole camera model (NCore conventions)
public:
    using Base = BaseCameraModel<OpenCVPinholeCameraModel<N_MAX_UNDISTORTION_ITERATIONS>>;

    OpenCVPinholeCameraModel(OpenCVPinholeCameraModelParameters const& parameters, float stop_undistortion_square_error_px2 = 1e-12)
        : Base(parameters)
        , parameters(parameters)
        , undistortion_stop_square_error_px2(stop_undistortion_square_error_px2) {}

    OpenCVPinholeCameraModelParameters parameters;
    float undistortion_stop_square_error_px2;

    struct DistortionReturn {
        float icD;
        float2 delta;
        float r2;
    };

    inline __device__ auto compute_distortion(float2 const& uv) const -> DistortionReturn {
        // Computes the radial, tangential, and thin-prism distortion given the camera ray
        auto const uv_squared = make_float2(uv.x * uv.x, uv.y * uv.y);
        auto const r2         = uv_squared.x + uv_squared.y;
        auto const a1         = 2.f * uv.x * uv.y;
        auto const a2         = r2 + 2.f * uv_squared.x;
        auto const a3         = r2 + 2.f * uv_squared.y;

        auto const icD_numerator   = 1.f + r2 * (parameters.radial_coeffs[0] + r2 * (parameters.radial_coeffs[1] + r2 * parameters.radial_coeffs[2]));
        auto const icD_denominator = 1.f + r2 * (parameters.radial_coeffs[3] + r2 * (parameters.radial_coeffs[4] + r2 * parameters.radial_coeffs[5]));
        auto const icD             = icD_numerator / icD_denominator;

        auto const delta_x = parameters.tangential_coeffs[0] * a1 + parameters.tangential_coeffs[1] * a2 + r2 * (parameters.thin_prism_coeffs[0] + r2 * parameters.thin_prism_coeffs[1]);
        auto const delta_y = parameters.tangential_coeffs[0] * a3 + parameters.tangential_coeffs[1] * a1 + r2 * (parameters.thin_prism_coeffs[2] + r2 * parameters.thin_prism_coeffs[3]);

        return {icD, make_float2(delta_x, delta_y), r2};
    }

    inline __device__ auto camera_ray_to_image_point_impl(float3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        auto image_point = make_float2(0.f, 0.f);

        // Treat all the points behind the camera plane to invalid / projecting to origin (NCore convention)
        if (cam_ray.z <= 0.f)
            return {image_point, false};

        // Evalutate distortion
        auto const uv_normalized    = make_float2(cam_ray.x, cam_ray.y) / cam_ray.z;
        auto const [icD, delta, r2] = compute_distortion(uv_normalized);

        auto constexpr k_min_radial_dist = 0.8f, k_max_radial_dist = 1.2f;
        auto const valid_radial = (icD > k_min_radial_dist) && (icD < k_max_radial_dist);

        // Project using ideal pinhole model (apply radial / tangential / thin-prism distortions)
        // in case radial distortion is within limits
        auto const uvND = icD * uv_normalized + delta;

        if (valid_radial) {
            image_point = uvND * make_float2(parameters.focal_length[0],
                                             parameters.focal_length[1]) +
                          make_float2(parameters.principal_point[0],
                                      parameters.principal_point[1]);
        } else {
            // If the radial distortion is out-of-limits, the computed coordinates will be unreasonable
            // (might even flip signs) - check on which side of the image we overshoot, and set the coordinates
            // out of the image bounds accordingly. The coordinates will be clipped to
            // viable range and direction but the exact values cannot be trusted / are still invalid
            auto const roi_clipping_radius = std::hypotf(parameters.resolution[0], parameters.resolution[1]);
            image_point                    = (roi_clipping_radius / std::sqrt(r2)) * uv_normalized +
                          make_float2(parameters.principal_point[0],
                                      parameters.principal_point[1]);
        }

        // Check if the image points fall within the image, set points that have too large distortion or fall outside the image sensor to invalid
        auto valid = valid_radial;
        valid &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);

        return {image_point, valid};
    }

    inline __device__ float2 compute_undistortion_iterative(float2 const& image_point) const {
        // Iteratively undistorts the image point using the inverse distortion model

        // Initial guess for the undistorted point
        auto const uv_0 = (image_point - make_float2(parameters.principal_point[0], parameters.principal_point[1])) /
                          make_float2(parameters.focal_length[0], parameters.focal_length[1]);

        auto uv = uv_0;
        for (auto j = 0; j < N_MAX_UNDISTORTION_ITERATIONS; ++j) {
            // Compute the distortion for the current estimate
            auto const [icD, delta, r2] = compute_distortion(uv);

            // Update the estimate using the inverse distortion model
            auto const uv_next = (uv_0 - delta) / icD;

            // Check for convergence
            if (auto const residual_vec = uv - uv_next; dot(residual_vec, residual_vec) < undistortion_stop_square_error_px2)
                break;

            uv = uv_next;
        }

        return uv;
    }

    inline __device__ float3 image_point_to_camera_ray_impl(float2 image_point) const {
        // Undistort the image point to uv coordinate
        auto const uv = compute_undistortion_iterative(image_point);

        // Unproject the undistorted image point to camera ray
        auto const camera_ray = make_float3(uv.x, uv.y, 1.f);

        // Make sure ray is normalized
        return camera_ray / length(camera_ray);
    }
};

template <size_t N_NEWTON_ITERATIONS = 3 /* fixed number of Netwon iteration for polynomial inversion - same as in NCore */>
struct OpenCVFisheyeCameraModel : BaseCameraModel<OpenCVFisheyeCameraModel<N_NEWTON_ITERATIONS>> {
    // OpenCV-compatible fisheye camera model (NCore conventions)
public:
    using Base = BaseCameraModel<OpenCVFisheyeCameraModel<N_NEWTON_ITERATIONS>>;

    OpenCVFisheyeCameraModel(OpenCVFisheyeCameraModelParameters const& parameters, float min_2d_norm = 1e-6f)
        : Base(parameters)
        , parameters(parameters)
        , min_2d_norm(min_2d_norm) {

        // initialize ninth-degree odd-only forward polynomial (mapping angles to normalized distances) theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
        auto const& [k1, k2, k3, k4] = parameters.radial_coeffs;
        forward_poly_odd             = {1.f, k1, k2, k3, k4};

        // eighth-degree differential of forward polynomial 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^8 + 9*k4*theta^8
        dforward_poly_even = {1, 3 * k1, 5 * k2, 7 * k3, 9 * k4};

        // approximate backward poly (mapping normalized distances to angles) *very crudely* by linear interpolation / equidistant angle model (also assuming image-centered principal point)
        auto const max_normalized_dist = fmaxf(parameters.resolution[0] / 2.f / parameters.focal_length[0],
                                               parameters.resolution[1] / 2.f / parameters.focal_length[1]);
        approx_backward_poly           = {0.f, parameters.max_angle / max_normalized_dist};
    }

    OpenCVFisheyeCameraModelParameters parameters;
    float min_2d_norm;
    std::array<float, 5> forward_poly_odd;
    std::array<float, 5> dforward_poly_even;
    std::array<float, 2> approx_backward_poly;

    inline __device__ auto camera_ray_to_image_point_impl(float3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        // Make sure norm is non-vanishing (norm vanishes for points along the principal-axis)
        auto cam_ray_xy_norm = numerically_stable_norm2(cam_ray.x, cam_ray.y);
        if (cam_ray_xy_norm <= 0.f)
            cam_ray_xy_norm = std::numeric_limits<float>::epsilon();

        auto const theta_full = atan2f(cam_ray_xy_norm, cam_ray.z);

        // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        // In particular for omnidirectional cameras, this prevents points outside the FOV to be
        // wrongly projected to in-image-domain points because of badly constrained polynomials outside
        // the effective FOV (which is different to the image boundaries).
        //
        // These FOV-clamped projections will be marked as *invalid*
        auto const theta = theta_full < parameters.max_angle ? theta_full : parameters.max_angle;

        // Evaluate forward polynomial (correspond to the radial distances to the principal point in the normalized image domain (up to focal length scales))
        auto const delta = eval_poly_odd_horner(forward_poly_odd, theta) / cam_ray_xy_norm;

        auto const image_point = float2{parameters.focal_length[0] * delta * cam_ray.x + parameters.principal_point[0],
                                        parameters.focal_length[1] * delta * cam_ray.y + parameters.principal_point[1]};

        auto valid = true;
        valid &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);
        valid &= theta < parameters.max_angle; // explicitly check for strictly smaller angles to classify FOV-clamped points as invalid

        return {image_point, valid};
    }

    inline __device__ float3 image_point_to_camera_ray_impl(float2 image_point) const {
        // Normalize the image point coordinates
        auto const uv = (image_point - float2{parameters.principal_point[0], parameters.principal_point[1]}) / float2{parameters.focal_length[0], parameters.focal_length[1]};

        // Compute the radial distance from the principal point
        auto const delta = length(uv);

        // Evaluate the inverse polynomial to find the angle theta
        auto const theta = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(PolynomialProxy<PolynomialType::ODD, 5>{forward_poly_odd},
                                                                                PolynomialProxy<PolynomialType::EVEN, 5>{dforward_poly_even},
                                                                                PolynomialProxy<PolynomialType::FULL, 2>{approx_backward_poly},
                                                                                delta);

        // Compute the camera ray and set the ones at the image center to [0,0,1]
        if (delta >= min_2d_norm) {
            // Scale the uv coordinates by the sine of the angle theta
            auto const scale_factor = sinf(theta) / delta;
            return make_float3(scale_factor * uv.x, scale_factor * uv.y, cosf(theta));
        } else {
            // For points at the image center, return a ray pointing straight ahead
            return make_float3(0.f, 0.f, 1.f);
        }
    }
};

template <size_t N_NEWTON_ITERATIONS = 3 /* fixed number of Netwon iteration for polynomial inversion - same as in NCore */>
struct FThetaCameraModel : BaseCameraModel<FThetaCameraModel<N_NEWTON_ITERATIONS>> {
    // NV-compatible FTheta camera model (NCore conventions)
public:
    using Base = BaseCameraModel<FThetaCameraModel<N_NEWTON_ITERATIONS>>;

    FThetaCameraModel(FThetaCameraModelParameters const& parameters,
                      float min_2d_norm = 1e-6f)
        : Base(parameters)
        , parameters(parameters)
        , min_2d_norm(min_2d_norm)
        , dreference_poly{} {

        if (parameters.reference_poly == FThetaCameraModelParameters::PolynomialType::PIXELDIST_TO_ANGLE)
            // compute first derivative of the backwards polynomial
            dreference_poly = {1.f * parameters.pixeldist_to_angle_poly.at(1), 2.f * parameters.pixeldist_to_angle_poly.at(2), 3.f * parameters.pixeldist_to_angle_poly.at(3), 4.f * parameters.pixeldist_to_angle_poly.at(4), 5.f * parameters.pixeldist_to_angle_poly.at(5)};
        else
            // compute first derivative of the forward polynomial
            dreference_poly = {1.f * parameters.angle_to_pixeldist_poly.at(1), 2.f * parameters.angle_to_pixeldist_poly.at(2), 3.f * parameters.angle_to_pixeldist_poly.at(3), 4.f * parameters.angle_to_pixeldist_poly.at(4), 5.f * parameters.angle_to_pixeldist_poly.at(5)};

        // FThetaCameraModelParameters are defined such that the image coordinate origin corresponds to
        // the center of the first pixel. To conform to the NCore CameraModel specification (having the image
        // coordinate origin aligned with the top-left corner of the first pixel) we therefore need to
        // offset the principal point by half a pixel. Please see NCore documentation for more information.
        this->parameters.principal_point[0] += .5f;
        this->parameters.principal_point[1] += .5f;
    }

    FThetaCameraModelParameters parameters;
    float min_2d_norm;
    std::array<float, 5> dreference_poly; // coefficient of first derivative of the reference polynomial

    inline __device__ auto camera_ray_to_image_point_impl(float3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        // Make sure norm is non-vanishing (norm vanishes for points along the principal-axis)
        auto cam_ray_xy_norm = numerically_stable_norm2(cam_ray.x, cam_ray.y);
        if (cam_ray_xy_norm <= 0.f)
            cam_ray_xy_norm = std::numeric_limits<float>::epsilon();

        auto const theta_full = atan2f(cam_ray_xy_norm, cam_ray.z);

        // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        // In particular for omnidirectional cameras, this prevents points outside the FOV to be
        // wrongly projected to in-image-domain points because of badly constrained polynomials outside
        // the effective FOV (which is different to the image boundaries).

        // These FOV-clamped projections will be marked as *invalid*
        auto const theta = theta_full < parameters.max_angle ? theta_full : parameters.max_angle;

        // Evaluate forward polynomial, giving delta = f(theta) factors
        auto const delta =
            (parameters.reference_poly == FThetaCameraModelParameters::PolynomialType::PIXELDIST_TO_ANGLE)
                ? eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>( // bw poly is reference, evaluate its inverse via Newton-based inversion
                      PolynomialProxy<PolynomialType::FULL, 6>{parameters.pixeldist_to_angle_poly},
                      PolynomialProxy<PolynomialType::FULL, 5>{dreference_poly},
                      PolynomialProxy<PolynomialType::FULL, 6>{parameters.angle_to_pixeldist_poly},
                      theta)
                : eval_poly_horner(parameters.angle_to_pixeldist_poly, theta); // fw is reference, evaluate it directly

        // Apply linear term A=[c,d;e,1] to f(theta)-weighted normalized 2d vectors, relative to principal point
        auto const& [c, d, e] = parameters.linear_cde;
        auto image_point      = delta * (make_float2(cam_ray.x, cam_ray.y) / cam_ray_xy_norm);
        image_point           = make_float2(c * image_point.x + d * image_point.y, e * image_point.x + image_point.y) +
                      make_float2(parameters.principal_point[0], parameters.principal_point[1]);

        auto valid = true;
        valid &= 0.0f <= image_point.x && image_point.x < parameters.resolution[0];
        valid &= 0.0f <= image_point.y && image_point.y < parameters.resolution[1];
        valid &= theta < parameters.max_angle;

        return {image_point, valid};
    }

    inline __device__ float3 image_point_to_camera_ray_impl(float2 image_point) const {
        // Get f(theta)-weighted normalized 2d vectors around principal point,
        // undoing linear term A = [c,d;e;1] via A^-1 = [1,-d;-e,c] / (c-e*d)
        auto const& [c, d, e] = parameters.linear_cde;
        image_point -= make_float2(parameters.principal_point[0], parameters.principal_point[1]);
        auto const image_point_dist = make_float2(image_point.x - d * image_point.y, -e * image_point.x + c * image_point.y) / (c - e * d);

        auto const rdist = length(image_point_dist);

        // Evaluate backward polynomial to get theta = f^-1(rdist) factor
        auto const theta =
            (parameters.reference_poly == FThetaCameraModelParameters::PolynomialType::PIXELDIST_TO_ANGLE)
                ? eval_poly_horner(parameters.pixeldist_to_angle_poly, rdist) // bw is reference, evaluate it directly
                : eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(       // fw is reference, evaluate its inverse via Newton-based inversion
                      PolynomialProxy<PolynomialType::FULL, 6>{parameters.angle_to_pixeldist_poly},
                      PolynomialProxy<PolynomialType::FULL, 5>{dreference_poly},
                      PolynomialProxy<PolynomialType::FULL, 6>{parameters.pixeldist_to_angle_poly},
                      rdist);

        // Compute the camera ray and set the ones at the image center to [0,0,1]
        if (rdist >= min_2d_norm) {
            auto const scale_factor = std::sin(theta) / rdist;
            return make_float3(scale_factor * image_point_dist.x, scale_factor * image_point_dist.y, std::cos(theta));
        } else {
            return make_float3(0.f, 0.f, 1.f);
        }
    }
};

template <size_t N_NEWTON_ITERATIONS = 3 /* fixed number of Netwon iteration for polynomial inversion - same as in NCore */>
struct BackwardsFThetaCameraModel : BaseCameraModel<BackwardsFThetaCameraModel<N_NEWTON_ITERATIONS>> {
    // NV-compatible FTheta camera model (NCore conventions)
public:
    using Base = BaseCameraModel<BackwardsFThetaCameraModel<N_NEWTON_ITERATIONS>>;

    BackwardsFThetaCameraModel(FThetaCameraModelParameters const& parameters,
                               float min_2d_norm = 1e-6f)
        : Base(parameters), parameters(parameters), min_2d_norm(min_2d_norm), dpixeldist_to_angle_poly{} {
        if (parameters.reference_poly != FThetaCameraModelParameters::PolynomialType::PIXELDIST_TO_ANGLE)
            throw std::runtime_error("Only supporting backwards reference polynomials");

        // FThetaCameraModelParameters are defined such that the image coordinate origin corresponds to
        // the center of the first pixel. To conform to the NCore CameraModel specification (having the image
        // coordinate origin aligned with the top-left corner of the first pixel) we therefore need to
        // offset the principal point by half a pixel. Please see NCore documentation for more information.
        this->parameters.principal_point[0] += .5f;
        this->parameters.principal_point[1] += .5f;

        // compute first derivative of the backwards polynomial

#pragma unroll
        for (auto j = 0; j < std::size(dpixeldist_to_angle_poly); ++j)
            dpixeldist_to_angle_poly[j] = (j + 1) * parameters.pixeldist_to_angle_poly.at(j + 1);
    }

    FThetaCameraModelParameters parameters;
    float min_2d_norm;
    std::array<float, FThetaCameraModelParameters::PolynomialDegree - 1> dpixeldist_to_angle_poly; // coefficient of first derivative of the backwards polynomial

    inline __device__ auto camera_ray_to_image_point(float3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        // Make sure norm is non-vanishing (norm vanishes for points along the principal-axis)
        auto cam_ray_xy_norm = numerically_stable_norm2(cam_ray.x, cam_ray.y);
        if (cam_ray_xy_norm <= 0.f)
            cam_ray_xy_norm = std::numeric_limits<float>::epsilon();

        auto const alpha_full = atan2f(cam_ray_xy_norm, cam_ray.z);

        // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        // In particular for omnidirectional cameras, this prevents points outside the FOV to be
        // wrongly projected to in-image-domain points because of badly constrained polynomials outside
        // the effective FOV (which is different to the image boundaries).
        //
        // These FOV-clamped projections will be marked as *invalid*
        auto const alpha = alpha_full < parameters.max_angle ? alpha_full : parameters.max_angle;

        auto const delta = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(
            PolynomialProxy<PolynomialType::FULL, 6>{parameters.pixeldist_to_angle_poly},
            PolynomialProxy<PolynomialType::FULL, 5>{dpixeldist_to_angle_poly},
            PolynomialProxy<PolynomialType::FULL, 6>{parameters.angle_to_pixeldist_poly},
            alpha);

        auto const theta       = delta / cam_ray_xy_norm;
        auto const image_point = make_float2(
            theta * cam_ray.x + parameters.principal_point[0],
            theta * cam_ray.y + parameters.principal_point[1]);

        auto valid = true;
        valid &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);
        valid &= alpha < parameters.max_angle;

        return {image_point, valid};
    }

    inline __device__ float3 image_point_to_camera_ray(float2 image_point) const {
        auto const image_point_dist = image_point - make_float2(parameters.principal_point[0], parameters.principal_point[1]);

        auto const rdist = length(image_point_dist);

        // Evaluate backward polynomial
        auto const alpha = eval_poly_horner(parameters.pixeldist_to_angle_poly, rdist);

        // Compute the camera ray and set the ones at the image center to [0,0,1]
        if (rdist >= min_2d_norm) {
            auto const scale_factor = sinf(alpha) / rdist;
            return make_float3(scale_factor * image_point_dist.x, scale_factor * image_point_dist.y, cosf(alpha));
        } else {
            return make_float3(0.f, 0.f, 1.f);
        }
    }
};
