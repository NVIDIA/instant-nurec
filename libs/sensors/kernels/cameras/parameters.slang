// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Camera parameter structures implementing projection and distortion interfaces

#include <sensors/kernels/cameras/interface.slang>

// ============================================================================
// Polynomial Utility Functions
// ============================================================================

// Maximum polynomial terms (degree 0-5, so 6 terms total)
static const uint MAX_POLYNOMIAL_TERMS = 6;

// Numerical constants
static const float EPSILON = 1e-6;  // General numerical guard for avoiding division by near-zero
static const float NEWTON_EPSILON = 1e-10;  // Tighter tolerance for Newton iteration convergence

// Shared polynomial evaluation utilities to avoid code duplication across camera models.
struct PolynomialUtils {
    // Evaluate polynomial: c[0] + c[1]*x + c[2]*x² + ... + c[degree]*x^degree
    [Differentiable]
    __generic<let MaxPolyNomialTerms: uint>
    static float eval_poly(float x, float c[MaxPolyNomialTerms], uint degree = MaxPolyNomialTerms - 1) {
        degree = min(degree, MaxPolyNomialTerms - 1);
        float result = 0.0;
        float x_pow = 1.0;

        [MaxIters(MaxPolyNomialTerms)]
        for (uint i = 0; i <= degree; i++) {
            result += c[i] * x_pow;
            x_pow *= x;
        }
        return result;
    }

    // Evaluate derivative of x*poly(x): d/dx[x*(c[0] + c[1]*x + ...)]
    // = c[0] + 2*c[1]*x + 3*c[2]*x² + ... (used for Newton iteration on odd polynomials)
    [Differentiable]
    static float eval_xpoly_derivative(float x, float c[MAX_POLYNOMIAL_TERMS], uint degree) {
        degree = min(degree, MAX_POLYNOMIAL_TERMS - 1);
        float result = 0.0;
        float x_pow = 1.0;

        [MaxIters(MAX_POLYNOMIAL_TERMS)]
        for (uint i = 0; i <= degree; i++) {
            result += float(i + 1) * c[i] * x_pow;
            x_pow *= x;
        }
        return result;
    }

    // Evaluate derivative of poly(x): d/dx[c[0] + c[1]*x + c[2]*x² + ...]
    // = c[1] + 2*c[2]*x + 3*c[3]*x² + ...
    [Differentiable]
    static float eval_poly_derivative(float x, float c[MAX_POLYNOMIAL_TERMS], uint degree) {
        degree = min(degree, MAX_POLYNOMIAL_TERMS - 1);
        if (degree == 0) return 0.0;

        float result = 0.0;
        float x_pow = 1.0;

        [MaxIters(MAX_POLYNOMIAL_TERMS)]
        for (uint i = 1; i <= degree; i++) {
            result += float(i) * c[i] * x_pow;
            x_pow *= x;
        }
        return result;
    }

    // Partial derivative dp/dx of the 2D triangular polynomial, evaluated analytically.
    // d(y_coeff_j)/dx uses the derivative of each x-Horner chain, then combined via y-Horner.
    __generic<let MaxPolyNomialTerms: uint>
    static float _eval_poly_2d_dfdx(float x, float y, float c[MaxPolyNomialTerms], uint order) {
        // REQUIRES: num_triangular_terms(order) <= MaxPolyNomialTerms
        // ORDER 1 -> 3 terms, ORDER 2 -> 6 terms, ORDER 3 -> 10 terms, ORDER 4 -> 15 terms

        int _1 = min(MaxPolyNomialTerms - 1, 1);
        int _2 = min(MaxPolyNomialTerms - 1, 2);
        int _3 = min(MaxPolyNomialTerms - 1, 3);
        int _4 = min(MaxPolyNomialTerms - 1, 4);
        int _5 = min(MaxPolyNomialTerms - 1, 5);
        int _6 = min(MaxPolyNomialTerms - 1, 6);
        int _7 = min(MaxPolyNomialTerms - 1, 7);
        int _8 = min(MaxPolyNomialTerms - 1, 8);
        int _10 = min(MaxPolyNomialTerms - 1, 10);
        int _11 = min(MaxPolyNomialTerms - 1, 11);
        int _13 = min(MaxPolyNomialTerms - 1, 13);

        if (order == 0) {
            return 0.0;
        } else if (order == 1) {
            return c[_1];
        } else if (order == 2) {
            float dc0 = c[_1] + x * (2.0 * c[_2]);
            float dc1 = c[_4];
            return dc0 + y * dc1;
        } else if (order == 3) {
            float dc0 = c[_1] + x * (2.0 * c[_2] + x * (3.0 * c[_3]));
            float dc1 = c[_5] + x * (2.0 * c[_6]);
            float dc2 = c[_8];
            return dc0 + y * (dc1 + y * dc2);
        } else {
            float dc0 = c[_1] + x * (2.0 * c[_2] + x * (3.0 * c[_3] + x * (4.0 * c[_4])));
            float dc1 = c[_6] + x * (2.0 * c[_7] + x * (3.0 * c[_8]));
            float dc2 = c[_10] + x * (2.0 * c[_11]);
            float dc3 = c[_13];
            return dc0 + y * (dc1 + y * (dc2 + y * dc3));
        }
    }

    // Partial derivative dp/dy of the 2D triangular polynomial, evaluated analytically.
    // Differentiates the outer y-Horner chain: dp/dy = y_coeff1 + y*(2*y_coeff2 + y*(3*y_coeff3 + ...))
    __generic<let MaxPolyNomialTerms: uint>
    static float _eval_poly_2d_dfdy(float x, float y, float c[MaxPolyNomialTerms], uint order) {
        // REQUIRES: num_triangular_terms(order) <= MaxPolyNomialTerms
        // ORDER 1 -> 3 terms, ORDER 2 -> 6 terms, ORDER 3 -> 10 terms, ORDER 4 -> 15 terms
        int _2 = min(MaxPolyNomialTerms - 1, 2);
        int _3 = min(MaxPolyNomialTerms - 1, 3);
        int _4 = min(MaxPolyNomialTerms - 1, 4);
        int _5 = min(MaxPolyNomialTerms - 1, 5);
        int _6 = min(MaxPolyNomialTerms - 1, 6);
        int _7 = min(MaxPolyNomialTerms - 1, 7);
        int _8 = min(MaxPolyNomialTerms - 1, 8);
        int _9 = min(MaxPolyNomialTerms - 1, 9);
        int _10 = min(MaxPolyNomialTerms - 1, 10);
        int _11 = min(MaxPolyNomialTerms - 1, 11);
        int _12 = min(MaxPolyNomialTerms - 1, 12);
        int _13 = min(MaxPolyNomialTerms - 1, 13);
        int _14 = min(MaxPolyNomialTerms - 1, 14);

        if (order == 0) {
            return 0.0;
        } else if (order == 1) {
            return c[_2];
        } else if (order == 2) {
            float yc1 = c[_3] + x * c[_4];
            return yc1 + y * (2.0 * c[_5]);
        } else if (order == 3) {
            float yc1 = c[_4] + x * (c[_5] + x * c[_6]);
            float yc2 = c[_7] + x * c[_8];
            return yc1 + y * (2.0 * yc2 + y * (3.0 * c[_9]));
        } else {
            float yc1 = c[_5] + x * (c[_6] + x * (c[_7] + x * c[_8]));
            float yc2 = c[_9] + x * (c[_10] + x * c[_11]);
            float yc3 = c[_12] + x * c[_13];
            return yc1 + y * (2.0 * yc2 + y * (3.0 * yc3 + y * (4.0 * c[_14])));
        }
    }

    // Evaluate 2D polynomial: c[0] + c[1]*x + c[2]*x² + ... using nested Horner's method.
    // The custom backward (eval_poly_2d_bwd) computes gradients analytically, eliminating
    // all autodiff intermediate storage from this function's forward pass.
    // TODO: Write a generic version but this made slangc crash => follow-up
    [Differentiable]
    __generic<let MaxPolyNomialTerms: uint>
    static float eval_poly_2d(float x, float y, float c[MaxPolyNomialTerms], uint order) {
        // REQUIRES: num_triangular_terms(order) <= MaxPolyNomialTerms
        // ORDER 1 -> 3 terms, ORDER 2 -> 6 terms, ORDER 3 -> 10 terms, ORDER 4 -> 15 terms

        int _0 = 0;
        int _1 = min(MaxPolyNomialTerms - 1, 1);
        int _2 = min(MaxPolyNomialTerms - 1, 2);
        int _3 = min(MaxPolyNomialTerms - 1, 3);
        int _4 = min(MaxPolyNomialTerms - 1, 4);
        int _5 = min(MaxPolyNomialTerms - 1, 5);
        int _6 = min(MaxPolyNomialTerms - 1, 6);
        int _7 = min(MaxPolyNomialTerms - 1, 7);
        int _8 = min(MaxPolyNomialTerms - 1, 8);
        int _9 = min(MaxPolyNomialTerms - 1, 9);
        int _10 = min(MaxPolyNomialTerms - 1, 10);
        int _11 = min(MaxPolyNomialTerms - 1, 11);
        int _12 = min(MaxPolyNomialTerms - 1, 12);
        int _13 = min(MaxPolyNomialTerms - 1, 13);
        int _14 = min(MaxPolyNomialTerms - 1, 14);


        if (order == 0) {

            return c[_0];

        } else if (order == 1) {

            float y_coeff0 = c[_0] + x * c[_1];
            float y_coeff1 = c[_2];
            return y_coeff0 + y * y_coeff1;

        } else if (order == 2) {

            float y_coeff0 = c[_0] + x * (c[_1] + x * c[_2]);
            float y_coeff1 = c[_3] + x * c[_4];
            float y_coeff2 = c[_5];
            return y_coeff0 + y * (y_coeff1 + y * y_coeff2);

        } else if (order == 3) {

            float y_coeff0 = c[_0] + x * (c[_1] + x * (c[_2] + x * c[_3]));
            float y_coeff1 = c[_4] + x * (c[_5] + x * c[_6]);
            float y_coeff2 = c[_7] + x * c[_8];
            float y_coeff3 = c[_9];

            return y_coeff0 + y * (y_coeff1 + y * (y_coeff2 + y * y_coeff3));

        } else { // if (order == 4)

            float y_coeff0 = c[_0] + x * (c[_1] + x * (c[_2] + x * (c[_3] + x * c[_4])));
            float y_coeff1 = c[_5] + x * (c[_6] + x * (c[_7] + x * c[_8]));
            float y_coeff2 = c[_9] + x * (c[_10] + x * c[_11]);
            float y_coeff3 = c[_12] + x * c[_13];
            float y_coeff4 = c[_14];

            return y_coeff0 + y * (y_coeff1 + y * (y_coeff2 + y * (y_coeff3 + y * y_coeff4)));

        }
    }

    // Custom backward for eval_poly_2d. Computes all gradients analytically from
    // primal inputs, so the forward pass stores zero autodiff intermediates.
    //   d_x  = d_result * dp/dx   (via _eval_poly_2d_dfdx)
    //   d_y  = d_result * dp/dy   (via _eval_poly_2d_dfdy)
    //   d_c[k] = d_result * x^i * y^j   (monomial basis)
    [BackwardDerivativeOf(eval_poly_2d)]
    __generic<let MaxPolyNomialTerms: uint>
    static void eval_poly_2d_bwd(
        inout DifferentialPair<float> x,
        inout DifferentialPair<float> y,
        inout DifferentialPair<float[MaxPolyNomialTerms]> c,
        uint order,
        float d_result
    ) {
        // REQUIRES: num_triangular_terms(order) <= MaxPolyNomialTerms
        // ORDER 1 -> 3 terms, ORDER 2 -> 6 terms, ORDER 3 -> 10 terms, ORDER 4 -> 15 terms
        float px = x.p;
        float py = y.p;
        float[MaxPolyNomialTerms] pc = c.p;

        // Accumulate x, y gradients from analytical partial derivatives
        float dp_dx = _eval_poly_2d_dfdx(px, py, pc, order);
        float dp_dy = _eval_poly_2d_dfdy(px, py, pc, order);
        x = diffPair(px, x.d + dp_dx * d_result);
        y = diffPair(py, y.d + dp_dy * d_result);

        // Accumulate coefficient gradients: d_c[k] = d_result * x^i(k) * y^j(k)
        float[MaxPolyNomialTerms] dc = c.d;

        int _0 = 0;
        int _1 = min(MaxPolyNomialTerms - 1, 1);
        int _2 = min(MaxPolyNomialTerms - 1, 2);
        int _3 = min(MaxPolyNomialTerms - 1, 3);
        int _4 = min(MaxPolyNomialTerms - 1, 4);
        int _5 = min(MaxPolyNomialTerms - 1, 5);
        int _6 = min(MaxPolyNomialTerms - 1, 6);
        int _7 = min(MaxPolyNomialTerms - 1, 7);
        int _8 = min(MaxPolyNomialTerms - 1, 8);
        int _9 = min(MaxPolyNomialTerms - 1, 9);
        int _10 = min(MaxPolyNomialTerms - 1, 10);
        int _11 = min(MaxPolyNomialTerms - 1, 11);
        int _12 = min(MaxPolyNomialTerms - 1, 12);
        int _13 = min(MaxPolyNomialTerms - 1, 13);
        int _14 = min(MaxPolyNomialTerms - 1, 14);

        float px2 = px * px;
        float py2 = py * py;

        if (order == 0) {

            dc[_0] += d_result;

        } else if (order == 1) {

            dc[_0] += d_result;
            dc[_1] += d_result * px;
            dc[_2] += d_result * py;

        } else if (order == 2) {

            dc[_0] += d_result;
            dc[_1] += d_result * px;
            dc[_2] += d_result * px2;
            dc[_3] += d_result * py;
            dc[_4] += d_result * px * py;
            dc[_5] += d_result * py2;

        } else if (order == 3) {

            float px3 = px2 * px;
            float py3 = py2 * py;
            dc[_0] += d_result;
            dc[_1] += d_result * px;
            dc[_2] += d_result * px2;
            dc[_3] += d_result * px3;
            dc[_4] += d_result * py;
            dc[_5] += d_result * px * py;
            dc[_6] += d_result * px2 * py;
            dc[_7] += d_result * py2;
            dc[_8] += d_result * px * py2;
            dc[_9] += d_result * py3;

        } else { // order == 4

            float px3 = px2 * px;
            float px4 = px3 * px;
            float py3 = py2 * py;
            float py4 = py3 * py;
            dc[_0]  += d_result;
            dc[_1]  += d_result * px;
            dc[_2]  += d_result * px2;
            dc[_3]  += d_result * px3;
            dc[_4]  += d_result * px4;
            dc[_5]  += d_result * py;
            dc[_6]  += d_result * px * py;
            dc[_7]  += d_result * px2 * py;
            dc[_8]  += d_result * px3 * py;
            dc[_9]  += d_result * py2;
            dc[_10] += d_result * px * py2;
            dc[_11] += d_result * px2 * py2;
            dc[_12] += d_result * py3;
            dc[_13] += d_result * px * py3;
            dc[_14] += d_result * py4;

        }

        c = diffPair(pc, dc);
    }
};

[Differentiable]
static float numerically_stable_xy_norm(float3 camera_ray) {
    // def _numerically_stable_xy_norm(cam_rays: torch.Tensor) -> torch.Tensor:
    // """Evaluate the norm in a numerically stable manner"""

    // xy_norms = torch.zeros_like(cam_rays[:, 0]).unsqueeze(1)  # Zero rays stay with zero norm

    // abs_pts = torch.abs(cam_rays[:, :2])
    // min_pts = torch.min(abs_pts, dim=1, keepdim=True).values
    // max_pts = torch.max(abs_pts, dim=1, keepdim=True).values

    // # Output the norm of non-zero rays only
    // non_zero_norms = max_pts > 0
    // min_max_ratio = min_pts[non_zero_norms] / max_pts[non_zero_norms]
    // xy_norms[non_zero_norms, None] = max_pts[non_zero_norms, None] * torch.sqrt(
    //     1 + torch.pow(min_max_ratio[:, None], 2)
    // )

    // return xy_norms

    float xy_norms = 0.0;
    float2 abs_pts = abs(camera_ray.xy);
    float min_pts = min(abs_pts.x, abs_pts.y);
    float max_pts = max(abs_pts.x, abs_pts.y);
    if (max_pts > 0.0) {
        float min_max_ratio = min_pts / max_pts;
        xy_norms = max_pts * sqrt(1.0 + min_max_ratio * min_max_ratio);
    }
    return xy_norms;
}

// ============================================================================
// External Distortion Implementations
// ============================================================================

// No external distortion - identity transformation
// IDifferentiable inherited from IExternalDistortion - has no differentiable fields
struct NoExternalDistortion : IExternalDistortion {
    [Differentiable]
    float3 distort(float3 camera_ray) {
        return camera_ray;
    }

    [Differentiable]
    float3 undistort(float3 distorted_ray) {
        return distorted_ray;
    }
};

// Bivariate windshield distortion model
// IDifferentiable inherited from IExternalDistortion - allows gradient flow through struct members
//
// Carries only ONE polynomial pair (h_poly + v_poly = 21 floats). The loader
// (load_bivariate_windshield_distort_params / load_bivariate_windshield_undistort_params)
// resolves which pair to load based on the kernel direction and reference_polynomial.
// This halves DiffPair storage from 84 to 42 floats.
struct BivariateWindshieldModelParameters : IExternalDistortion {
    static const uint MAX_H_POLYNOMIAL_TERMS = 6;
    static const uint MAX_V_POLYNOMIAL_TERMS = 15;

    no_diff uint h_poly_degree;
    no_diff uint v_poly_degree;
    float h_poly[MAX_H_POLYNOMIAL_TERMS];
    float v_poly[MAX_V_POLYNOMIAL_TERMS];

    [Differentiable]
    float3 apply_distortion(float3 camera_ray) {
        float3 ray_norm = normalize(camera_ray);
        float x = ray_norm.x;
        float y = ray_norm.y;
        float z = ray_norm.z;

        float phi = asin(x);
        float theta = asin(y);

        float adj_phi = PolynomialUtils.eval_poly_2d(phi, theta, h_poly, h_poly_degree);
        float adj_theta = PolynomialUtils.eval_poly_2d(phi, theta, v_poly, v_poly_degree);

        x = sin(adj_phi);
        y = sin(adj_theta);
        float xy_norm = x * x + y * y;
        z = sqrt(clamp(1 - xy_norm, 0.0, 1.0)) * (z > 0.0 ? 1.0 : -1.0);

        return float3(x, y, z);
    }

    // Note: This is the same as apply_distortion in both, what actually happens is based on the loaded params.
    // The loader (load_bivariate_windshield_distort_params / load_bivariate_windshield_undistort_params)
    [Differentiable]
    float3 distort(float3 camera_ray) {
        return apply_distortion(camera_ray);
    }

    [Differentiable]
    float3 undistort(float3 distorted_ray) {
        return apply_distortion(distorted_ray);
    }
};

// ============================================================================
// Camera Projection Implementations
// ============================================================================

// OpenCV Pinhole camera model with radial, tangential, and thin prism distortion
// IDifferentiable inherited from ICameraProjection - allows gradient flow through struct members
struct OpenCVPinholeCameraModelParameters : ICameraProjection {
    // Number of iterations for iterative undistortion
    static const int UNDISTORT_ITERATIONS = 10;

    float2 focal_length;        // (fx, fy)
    float2 principal_point;     // (cx, cy)
    float radial_coeffs[6];     // [k1, k2, k3, k4, k5, k6]
    float tangential_coeffs[2]; // [p1, p2]
    float thin_prism_coeffs[4]; // [s1, s2, s3, s4]
    float2 resolution;            // [width, height] resolution in pixels

    // Returns: icD, delta_x, delta_y, r_2
    [Differentiable]
    float4 compute_distortion(float2 xy) {
        // def __compute_distortion(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        //     """Computes the radial, tangential, and thin-prism distortion given the camera rays"""

        //     # Compute the helper variables
        //     xy_squared = torch.square(xy)
        //     r_2 = torch.sum(xy_squared, dim=1)
        //     xy_prod = xy[:, 0] * xy[:, 1]
        //     a1 = 2 * xy_prod
        //     a2 = r_2 + 2 * xy_squared[:, 0]
        //     a3 = r_2 + 2 * xy_squared[:, 1]

        //     icD_numerator = 1.0 + r_2 * (
        //         self.radial_coeffs[0] + r_2 * (self.radial_coeffs[1] + r_2 * self.radial_coeffs[2])
        //     )
        //     icD_denominator = 1.0 + r_2 * (
        //         self.radial_coeffs[3] + r_2 * (self.radial_coeffs[4] + r_2 * self.radial_coeffs[5])
        //     )
        //     icD = icD_numerator / icD_denominator

        //     delta_x = (
        //         self.tangential_coeffs[0] * a1
        //         + self.tangential_coeffs[1] * a2
        //         + r_2 * (self.thin_prism_coeffs[0] + r_2 * self.thin_prism_coeffs[1])
        //     )
        //     delta_y = (
        //         self.tangential_coeffs[0] * a3
        //         + self.tangential_coeffs[1] * a1
        //         + r_2 * (self.thin_prism_coeffs[2] + r_2 * self.thin_prism_coeffs[3])
        //     )

        float2 xy_squared = xy * xy;
        float r_2 = xy_squared.x + xy_squared.y;
        float xy_prod = xy.x * xy.y;

        float a1 = 2 * xy_prod;
        float a2 = r_2 + 2 * xy_squared.x;
        float a3 = r_2 + 2 * xy_squared.y;

        float icD_numerator = 1.0 + r_2 * (radial_coeffs[0] + r_2 * (radial_coeffs[1] + r_2 * radial_coeffs[2]));
        float icD_denominator = 1.0 + r_2 * (radial_coeffs[3] + r_2 * (radial_coeffs[4] + r_2 * radial_coeffs[5]));
        float icD = icD_numerator / icD_denominator;

        float delta_x = tangential_coeffs[0] * a1 + tangential_coeffs[1] * a2 + r_2 * (thin_prism_coeffs[0] + r_2 * thin_prism_coeffs[1]);
        float delta_y = tangential_coeffs[0] * a3 + tangential_coeffs[1] * a1 + r_2 * (thin_prism_coeffs[2] + r_2 * thin_prism_coeffs[3]);

        return float4(icD, delta_x, delta_y, r_2);
    }

    [Differentiable]
    CameraProjectResult camera_rays_to_image_points(float3 camera_ray) {
        CameraProjectResult result;

        // Check if ray points forward
        if (camera_ray.z <= 0.0) {
            result.valid = false;
            result.image_point = float2(0.0, 0.0);
            return result;
        }

        // Normalize to plane z=1
        float2 uv_normalized = float2(camera_ray.x, camera_ray.y) / camera_ray.z;

        // distortion = icD, delta_x, delta_y, r_2
        float4 distortion = compute_distortion(uv_normalized);

        const float k_min_radial_dist = 0.8;
        const float k_max_radial_dist = 1.2;

        if (distortion.x < k_min_radial_dist || distortion.x > k_max_radial_dist) {
            float roi_clipping_radius = length(resolution);
            result.image_point = uv_normalized * roi_clipping_radius / sqrt(distortion.w) + principal_point;
        } else {
            float2 uvND = uv_normalized * distortion.x + float2(distortion.y, distortion.z);
            result.image_point = uvND * focal_length + principal_point;
        }

        result.valid = (result.image_point.x >= 0.0 && result.image_point.x < resolution.x && result.image_point.y >= 0.0 && result.image_point.y < resolution.y);
        return result;
    }

    [Differentiable]
    float3 image_points_to_camera_rays(float2 image_point) {
        // Remove intrinsics to get distorted normalized coordinates
        float2 xy0 = (image_point - principal_point) / focal_length;

        // Iterative undistortion using fixed-point iteration
        float2 xy = xy0;

        [MaxIters(UNDISTORT_ITERATIONS)]
        for (int iter = 0; iter < UNDISTORT_ITERATIONS; iter++) {

            //     return icD, delta_x, delta_y, r_2
            float2 xy_squared = xy * xy;
            float r_2 = xy_squared.x + xy_squared.y;
            float xy_prod = xy.x * xy.y;

            float a1 = 2 * xy_prod;
            float a2 = r_2 + 2 * xy_squared.x;
            float a3 = r_2 + 2 * xy_squared.y;

            float icD_numerator = 1.0 + r_2 * (radial_coeffs[0] + r_2 * (radial_coeffs[1] + r_2 * radial_coeffs[2]));
            float icD_denominator = 1.0 + r_2 * (radial_coeffs[3] + r_2 * (radial_coeffs[4] + r_2 * radial_coeffs[5]));
            float icD = icD_numerator / icD_denominator;

            // distortion = icD, delta_x, delta_y, r_2
            float4 distortion = compute_distortion(xy);

            xy = (xy0 - distortion.yz) / distortion.x;
        }

        return normalize(float3(xy.x, xy.y, 1.0));
    }
};

// OpenCV Fisheye camera model
// Matches OpenCV's fisheye model: theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
// Uses Newton iteration for exact undistortion to match OpenCV behavior
// IDifferentiable inherited from ICameraProjection - allows gradient flow through struct members
struct OpenCVFisheyeCameraModelParameters : ICameraProjection {
    // Maximum Newton iterations for polynomial inversion (compile-time bound for autodiff)
    // TODO: evaluate if this can be lowered to 10 iterations like Ftheta.
    static const int MAX_NEWTON_ITERATIONS = 20;

    float2 principal_point;     // (cx, cy)
    float2 focal_length;        // (fx, fy)
    float forward_poly[4];      // [k1, k2, k3, k4] distortion coefficients
    float approx_backward_factor; // Factor for approximate backward polynomial for initial guess
    float2 resolution;            // [width, height] resolution in pixels
    no_diff float max_angle;            // Maximum ray angle in radians - configuration threshold
    no_diff int newton_iterations;      // Number of Newton iterations - discrete value
    no_diff float min_2d_norm;          // Minimum 2D norm threshold - configuration threshold

    // Evaluate forward polynomial
    // ninth-degree forward polynomial (mapping angles to normalized distances) theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
    [Differentiable]
    float eval_forward_poly(float theta) {
        float k1 = forward_poly[0], k2 = forward_poly[1], k3 = forward_poly[2], k4 = forward_poly[3];
        float theta2 = theta * theta;
        return theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))));
    }

    // eighth-degree differential of forward polynomial 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^8 + 9*k4*theta^8
    [Differentiable]
    float eval_dforward_poly(float theta) {
        float k1 = forward_poly[0], k2 = forward_poly[1], k3 = forward_poly[2], k4 = forward_poly[3];
        float theta2 = theta * theta;
        return 1.0 + theta2 * (3.0 * k1 + theta2 * (5.0 * k2 + theta2 * (7.0 * k3 + theta2 * 9.0 * k4)));
    }

    [Differentiable]
    float eval_approx_backward_poly(float x) {
        return approx_backward_factor * x;
    }

    // Invert forward polynomial using Newton iteration
    [Differentiable]
    float invert_forward_poly_newton(float y) {
        float x = eval_approx_backward_poly(y);
        [MaxIters(MAX_NEWTON_ITERATIONS)]
        for (int i = 0; i < newton_iterations; i++) {
            float df_x = eval_dforward_poly(x);
            float residual = eval_forward_poly(x) - y;
            x = x - residual / df_x;
        }
        return x;
    }

    [Differentiable]
    CameraProjectResult camera_rays_to_image_points(float3 camera_ray) {
        CameraProjectResult result;

        // Check forward direction
        if (camera_ray.z <= 0.0) {
            result.valid = false;
            result.image_point = float2(0.0, 0.0);
            return result;
        }

        // Compute ray angle
        float eps = 1e-6;
        float ray_xy_norm = max(numerically_stable_xy_norm(camera_ray), eps);

        float theta = min(atan2(ray_xy_norm, camera_ray.z), max_angle);

        float3 ray_norm = normalize(camera_ray);
        float delta = eval_forward_poly(theta);

        float2 image_point = focal_length * delta / ray_xy_norm * camera_ray.xy + principal_point;

        if (image_point.x < 0.0 || image_point.x >= resolution.x || image_point.y < 0.0 || image_point.y >= resolution.y) {
            result.valid = false;
            result.image_point = float2(0.0, 0.0);
        } else {
            result.valid = true;
            result.image_point = image_point;
        }

        return result;
    }

    [Differentiable]
    float3 image_points_to_camera_rays(float2 image_point) {
        // Remove intrinsics
        float x = (image_point.x - principal_point.x) / focal_length.x;
        float y = (image_point.y - principal_point.y) / focal_length.y;

        float delta = length(float2(x, y));
        if (delta < min_2d_norm) {
            return float3(0.0, 0.0, 1.0);
        }

        // Use Newton iteration to invert forward polynomial
        float theta = invert_forward_poly_newton(delta);

        // Compute ray direction
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);

        float scale = sin_theta / max(delta, min_2d_norm);
        return normalize(float3(x * scale, y * scale, cos_theta));
    }
};

// F-Theta camera model
// Uses Newton iteration to exactly invert polynomials, matching the reference implementation
// IDifferentiable inherited from ICameraProjection - allows gradient flow through struct members
struct FThetaCameraModelParameters : ICameraProjection {
    // Maximum Newton iterations for polynomial inversion (compile-time bound for autodiff)
    static const int MAX_NEWTON_ITERATIONS = 10;

    enum FThetaPolynomialType : int {
        FORWARD = 0,  // Forward polynomial (angle → radius) is the reference
        BACKWARD = 1  // Backward polynomial (radius → angle) is the reference
    };

    no_diff FThetaPolynomialType reference_poly;  // Discrete choice - not differentiable
    float2 principal_point;     // (cx, cy)
    float fw_poly[MAX_POLYNOMIAL_TERMS];          // Forward polynomial coefficients (angle → radius)
    float bw_poly[MAX_POLYNOMIAL_TERMS];          // Backward polynomial coefficients (radius → angle)
    float2x2 A;                 // Forward transformation matrix (2x2 linear term)
    float2x2 Ainv;              // Inverse transformation matrix (2x2 linear term)
    no_diff uint fw_poly_degree;        // Discrete value - not differentiable
    no_diff uint bw_poly_degree;        // Discrete value - not differentiable
    no_diff float max_angle;            // Configuration threshold - not differentiable
    no_diff int newton_iterations;      // Discrete value - not differentiable
    no_diff float min_2d_norm;          // Configuration threshold - not differentiable

    // Non-differentiable Newton solver for inverting bw_poly.
    // Runs to convergence without storing per-iteration autodiff context.
    float _solve_bw_newton(float theta) {
        float r = PolynomialUtils.eval_poly(theta, fw_poly, fw_poly_degree);
        for (int i = 0; i < newton_iterations; i++) {
            float df_r = PolynomialUtils.eval_poly_derivative(r, bw_poly, bw_poly_degree);
            if (abs(df_r) < NEWTON_EPSILON) break;
            float f_r = PolynomialUtils.eval_poly(r, bw_poly, bw_poly_degree);
            float delta = (f_r - theta) / df_r;
            r = r - delta;
            if (abs(delta) < NEWTON_EPSILON) break;
        }
        return max(r, 0.0);
    }

    // Invert backward polynomial using implicit differentiation.
    // Given theta, find r such that bw_poly(r) = theta.
    // Runs Newton to convergence non-differentiably, then applies one differentiable
    // correction step so autodiff gets exact gradients via the implicit function theorem.
    [Differentiable]
    float invert_bw_poly_newton(float theta) {
        float r_star = no_diff _solve_bw_newton(theta);
        float f_val = PolynomialUtils.eval_poly(r_star, bw_poly, bw_poly_degree);
        float df_val = PolynomialUtils.eval_poly_derivative(r_star, bw_poly, bw_poly_degree);
        float safe_df = abs(df_val) > NEWTON_EPSILON ? df_val : 1.0;
        return r_star - (f_val - theta) / safe_df;
    }



    // Non-differentiable Newton solver for inverting fw_poly.
    // Runs to convergence without storing per-iteration autodiff context.
    float _solve_fw_newton(float r) {
        float theta = PolynomialUtils.eval_poly(r, bw_poly, bw_poly_degree);
        for (int i = 0; i < newton_iterations; i++) {
            float df_theta = PolynomialUtils.eval_poly_derivative(theta, fw_poly, fw_poly_degree);
            if (abs(df_theta) < NEWTON_EPSILON) break;
            float f_theta = PolynomialUtils.eval_poly(theta, fw_poly, fw_poly_degree);
            float delta = (f_theta - r) / df_theta;
            theta = theta - delta;
            if (abs(delta) < NEWTON_EPSILON) break;
        }
        return max(theta, 0.0);
    }

    // Invert forward polynomial using implicit differentiation.
    // Runs Newton to convergence non-differentiably, then applies one differentiable
    // correction step so autodiff gets exact gradients via the implicit function theorem
    // without storing O(iterations) context.
    [Differentiable]
    float invert_fw_poly_newton(float r) {
        float theta_star = no_diff _solve_fw_newton(r);
        float f_val = PolynomialUtils.eval_poly(theta_star, fw_poly, fw_poly_degree);
        float df_val = PolynomialUtils.eval_poly_derivative(theta_star, fw_poly, fw_poly_degree);
        float safe_df = abs(df_val) > NEWTON_EPSILON ? df_val : 1.0;
        return theta_star - (f_val - r) / safe_df;
    }

    [Differentiable]
    CameraProjectResult camera_rays_to_image_points(float3 camera_ray) {
        CameraProjectResult result;
        result.valid = true;

        if (camera_ray.z <= 0.0) {
            result.valid = false;
            result.image_point = float2(0.0, 0.0);
            return result;
        }

        // Normalize the ray (no A transformation here - A is applied to 2D result)
        float3 ray_norm = normalize(camera_ray);

        // Compute theta from the raw normalized ray
        float theta = acos(clamp(ray_norm.z, -1.0, 1.0));
        if (theta > max_angle) {
            result.valid = false;
            result.image_point = float2(0.0, 0.0);
            return result;
        }

        // Evaluate polynomial to get radial distance
        // Forward polynomial: r = fw_poly(theta), backward polynomial requires Newton to invert
        float r;
        if (reference_poly == FThetaPolynomialType.FORWARD) {
            // FORWARD polynomial directly gives r from theta: r = fw_poly(theta)
            r = PolynomialUtils.eval_poly(theta, fw_poly, fw_poly_degree);
        } else {
            // BACKWARD polynomial requires Newton iteration to invert
            r = invert_bw_poly_newton(theta);
        }

        // Compute 2D offset from ray's XY components
        float xy_norm = sqrt(ray_norm.x * ray_norm.x + ray_norm.y * ray_norm.y);
        if (xy_norm < min_2d_norm) {
            result.image_point = principal_point;
            return result;
        }

        // Compute the 2D offset (before applying linear term A)
        float scale = r / xy_norm;
        float2 offset_2d = float2(ray_norm.x * scale, ray_norm.y * scale);

        // Apply linear term A to the 2D offset
        // A is stored as 3x3 with the 2x2 linear term in the upper-left
        float2 transformed_offset = float2(
            A[0][0] * offset_2d.x + A[0][1] * offset_2d.y,
            A[1][0] * offset_2d.x + A[1][1] * offset_2d.y
        );

        result.image_point = transformed_offset + principal_point;
        return result;
    }

    [Differentiable]
    float3 image_points_to_camera_rays(float2 image_point) {
        // Get 2D offset from principal point
        float2 offset_2d = float2(image_point.x - principal_point.x, image_point.y - principal_point.y);

        // Apply Ainv to the 2D offset FIRST
        // Ainv is stored as 3x3 with the 2x2 linear term in the upper-left
        float2 transformed_offset = float2(
            Ainv[0][0] * offset_2d.x + Ainv[0][1] * offset_2d.y,
            Ainv[1][0] * offset_2d.x + Ainv[1][1] * offset_2d.y
        );

        // Use the norm of the TRANSFORMED offset for polynomial evaluation
        float rdist = sqrt(transformed_offset.x * transformed_offset.x + transformed_offset.y * transformed_offset.y);
        if (rdist < min_2d_norm) {
            return float3(0.0, 0.0, 1.0);
        }

        // Evaluate polynomial using the transformed distance
        float theta;
        if (reference_poly == FThetaPolynomialType.BACKWARD) {
            // BACKWARD polynomial directly gives theta from rdist: theta = bw_poly(rdist)
            theta = PolynomialUtils.eval_poly(rdist, bw_poly, bw_poly_degree);
        } else {
            // FORWARD polynomial requires Newton iteration to invert
            theta = invert_fw_poly_newton(rdist);
        }

        // Compute camera ray from the TRANSFORMED offset
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);
        float scale = sin_theta / rdist;

        // Ray is computed directly from transformed offset - no additional Ainv multiplication
        float3 ray = float3(transformed_offset.x * scale, transformed_offset.y * scale, cos_theta);
        return normalize(ray);
    }
};


