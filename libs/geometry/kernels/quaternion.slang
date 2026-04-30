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

// Common quaternion operations library for Slang shaders
// Quaternion format: float4 in xyzw order (x, y, z, w)

module quaternion;

namespace quaternion {

// Constants
static const float QUAT_EPSILON = 1e-7f;
static const double QUAT_EPSILON_D = 1e-12;
static const float SMALL_ANGLE_THRESHOLD = 1e-6f;

// @brief Normalize a quaternion with safety checks
// @param quat The quaternion to normalize
// @return The normalized quaternion
[Differentiable]
float4 normalizeSafe(float4 quat) {
    const float norm_sq = dot(quat, quat);
    // Handle near-zero quaternions to avoid division by zero
    if (abs(norm_sq) < QUAT_EPSILON) {
        return identity();  // Return identity quaternion for degenerate case
    }
    // Since we are in a safe version, don't use rsqrt() and use the full sqrt.
    const float rnorm = 1.0f / sqrt(norm_sq);
    return quat * rnorm;
}

// @brief Normalize a quaternion with fast implementation
// @param quat The quaternion to normalize
// @return The normalized quaternion
[Differentiable]
float4 normalizeFast(float4 quat) {
    const float norm_sq = dot(quat, quat);
    // Check if already normalized to avoid unnecessary computation
    if (abs(norm_sq - 1.0f) < QUAT_EPSILON) {
        return quat;
    }
    const float rnorm = rsqrt(norm_sq);
    return quat * rnorm;
}

// @brief Quaternion conjugate
// @param q The quaternion
// @return The conjugate quaternion
[Differentiable]
float4 conjugate(float4 q) {
    return float4(-q.xyz, q.w);
}

// @brief Quaternion inverse (fast version, assumes normalized quaternion)
// @param q The quaternion to invert (must be normalized)
// @return The inverse quaternion
[Differentiable]
float4 inverse(float4 q) {
    // For unit quaternions, inverse is just the conjugate
    return conjugate(q);
}

// @brief Quaternion inverse with safety checks
// @param q The quaternion to invert
// @return The inverse quaternion
[Differentiable]
float4 inverseSafe(float4 q) {
    float norm_sq = dot(q, q);
    // Handle near-zero quaternions to avoid division by zero
    if (abs(norm_sq) < QUAT_EPSILON) {
        return identity();  // Return identity quaternion for degenerate case
    }
    // Check if already normalized to avoid unnecessary computation
    if (abs(norm_sq - 1.0f) < QUAT_EPSILON) {
        return conjugate(q);
    }
    // Use rsqrt for efficiency on GPU hardware: inv_norm_sq = 1/norm_sq = (1/sqrt(norm_sq))^2
    float inv_norm = rsqrt(norm_sq);
    return conjugate(q) * (inv_norm * inv_norm);
}

// @brief Multiply two quaternions using Hamilton product
// @param q1 The first quaternion
// @param q2 The second quaternion
// @return The product q1 * q2
[Differentiable]
float4 multiply(float4 q1, float4 q2) {
    // Optimized quaternion multiplication with fewer operations
    float3 q1_xyz = q1.xyz;
    float3 q2_xyz = q2.xyz;
    float3 cross_term = cross(q1_xyz, q2_xyz);
    float3 xyz = q1.w * q2_xyz + q2.w * q1_xyz + cross_term;
    float w = q1.w * q2.w - dot(q1_xyz, q2_xyz);
    return float4(xyz, w);
}

// @brief Convert normalized quaternion to 3x3 rotation matrix (fast version)
// @param normalized_quat A normalized quaternion
// @return The rotation matrix
[Differentiable]
float3x3 convertNormalizedQuatToMatrix(float4 normalized_quat) {
    float4 q = normalized_quat;
    float4 q2 = q * q;
    
    float xy = q.x * q.y, xz = q.x * q.z, xw = q.x * q.w;
    float yz = q.y * q.z, yw = q.y * q.w, zw = q.z * q.w;
    
    return float3x3(
        1.0f - 2.0f * (q2.y + q2.z), 2.0f * (xy - zw), 2.0f * (xz + yw),
        2.0f * (xy + zw), 1.0f - 2.0f * (q2.x + q2.z), 2.0f * (yz - xw),
        2.0f * (xz - yw), 2.0f * (yz + xw), 1.0f - 2.0f * (q2.x + q2.y)
    );
}

// @brief Convert 3x3 rotation matrix to quaternion using Shepperd's method
// Shepperd's method chooses the quaternion component with largest magnitude
// to avoid numerical instability near singularities.
// @param R The 3x3 rotation matrix (row-major)
// @return Quaternion in xyzw format (normalized)
[Differentiable]
float4 fromMatrix(float3x3 R) {
    // Compute decision values: diagonal elements and trace
    float r00 = R[0][0];
    float r11 = R[1][1];
    float r22 = R[2][2];
    float trace = r00 + r11 + r22;
    
    float4 quat;
    
    // Choose the largest to avoid division by small numbers
    if (trace > r00 && trace > r11 && trace > r22) {
        // Case 3: trace is max - compute w first
        float s = sqrt(1.0f + trace) * 2.0f;  // s = 4 * w
        quat = float4(
            (R[2][1] - R[1][2]) / s,   // x
            (R[0][2] - R[2][0]) / s,   // y
            (R[1][0] - R[0][1]) / s,   // z
            0.25f * s                  // w
        );
    } else if (r00 > r11 && r00 > r22) {
        // Case 0: r00 is max - compute x first
        float s = sqrt(1.0f + r00 - r11 - r22) * 2.0f;  // s = 4 * x
        quat = float4(
            0.25f * s,                 // x
            (R[0][1] + R[1][0]) / s,   // y
            (R[0][2] + R[2][0]) / s,   // z
            (R[2][1] - R[1][2]) / s    // w
        );
    } else if (r11 > r22) {
        // Case 1: r11 is max - compute y first
        float s = sqrt(1.0f + r11 - r00 - r22) * 2.0f;  // s = 4 * y
        quat = float4(
            (R[0][1] + R[1][0]) / s,   // x
            0.25f * s,                 // y
            (R[1][2] + R[2][1]) / s,   // z
            (R[0][2] - R[2][0]) / s    // w
        );
    } else {
        // Case 2: r22 is max - compute z first
        float s = sqrt(1.0f + r22 - r00 - r11) * 2.0f;  // s = 4 * z
        quat = float4(
            (R[0][2] + R[2][0]) / s,   // x
            (R[1][2] + R[2][1]) / s,   // y
            0.25f * s,                 // z
            (R[1][0] - R[0][1]) / s    // w
        );
    }
    
    // Normalize to ensure unit quaternion
    return normalizeSafe(quat);
}

// @brief Convert quaternion to 3x3 rotation matrix (with normalization)
// @param quat The quaternion (will be normalized)
// @return The rotation matrix
[Differentiable]
float3x3 toMatrix(float4 quat) {
    return convertNormalizedQuatToMatrix(normalizeSafe(quat));
}

// @brief Create quaternion from axis and angle
// @param axis The rotation axis (should be normalized)
// @param angle The rotation angle in radians
// @return The quaternion
[Differentiable]
float4 convertAngleToQuat(float3 axis, float angle) {
    float half_angle = angle * 0.5f;
    float s, c;
    sincos(half_angle, s, c);
    return float4(axis * s, c);
}

// @brief Extract axis and angle from normalized quaternion (fast version)
// @param normalized_quat A normalized quaternion
// @param axis Output: The rotation axis
// @param angle Output: The rotation angle
[Differentiable]
void convertQuatToAngle(float4 normalized_quat, out float3 axis, out float angle) {
    float4 q = normalized_quat;
    float w = clamp(q.w, -1.0f, 1.0f);
    angle = 2.0f * acos(w);
    
    float s = sqrt(1.0f - w * w);
    if (s < QUAT_EPSILON) {
        axis = float3(1.0f, 0.0f, 0.0f);
    } else {
        axis = q.xyz / s;
    }
}

// @brief Extract axis and angle from quaternion (with normalization)
// @param quat The quaternion (will be normalized)
// @param axis Output: The rotation axis
// @param angle Output: The rotation angle
[Differentiable]
void convertQuatToAngleSafe(float4 quat, out float3 axis, out float angle) {
    convertQuatToAngle(normalizeSafe(quat), axis, angle);
}

// @brief SO(3) exponential map with improved precision
// @param omega Angular velocity vector
// @return Quaternion representation
[Differentiable]
float4 so3Exp(float3 omega) {
    double theta_sq_d = double(dot(omega, omega));
    
    // Use double precision for small angle case to match PyTorch lietorch
    if (theta_sq_d < double(SMALL_ANGLE_THRESHOLD)) {
        double3 omega_d = double3(omega);
        double theta_sq_over_4 = theta_sq_d * 0.25;
        double4 result_d = double4(
            omega_d * (0.5 - theta_sq_d / 48.0 + theta_sq_d * theta_sq_d / 3840.0),
            1.0 - theta_sq_over_4 * (0.5 - theta_sq_d / 96.0 + theta_sq_d * theta_sq_d / 11520.0)
        );
        return float4(result_d);
    }
    
    // Use double precision for larger angles too
    double theta = sqrt(theta_sq_d);
    double half_theta = theta * 0.5;
    double sin_half_theta, cos_half_theta;
    sincos(half_theta, sin_half_theta, cos_half_theta);
    double inv_theta = 1.0 / theta;
    
    return float4(float3(double3(omega) * (sin_half_theta * inv_theta)), float(cos_half_theta));
}

// @brief SO(3) logarithm map with improved precision
// @param quat Quaternion
// @return Angular velocity vector
[Differentiable]
float3 so3Log(float4 quat) {
    // Use double precision for normalization
    double4 d = double4(quat);
    double norm_sq = dot(d, d);
    if (abs(norm_sq - 1.0) > QUAT_EPSILON_D) {
        double inv_norm = rsqrt(norm_sq);
        d *= inv_norm;
    }
    
    double w = abs(d.w);
    double3 v = d.xyz;
    if (d.w < 0) v = -v;
    
    double v_norm_sq = dot(v, v);
    
    // Use double precision Taylor expansion for small rotations
    if (v_norm_sq < QUAT_EPSILON_D) {
        return float3(v * (2.0 - v_norm_sq / 3.0 + 2.0 * v_norm_sq * v_norm_sq / 45.0));
    }
    
    double v_norm = sqrt(v_norm_sq);
    double theta = 2.0 * atan2(v_norm, w);
    return float3(v * (theta / v_norm));
}

// @brief Linear interpolation between two quaternions (LERP)
// @param q1 Start quaternion
// @param q2 End quaternion
// @param t Interpolation parameter [0, 1]
// @return Interpolated quaternion
[Differentiable]
float4 lerp(float4 q1, float4 q2, float t) {
    // Check if quaternions are on opposite hemispheres
    float dot_product = dot(q1, q2);
    if (dot_product < 0.0f) {
        q2 = -q2;
    }

    // TODO: We could use the builtin lerp to do this. For now, we need
    // to use our custom interpolation via: q1 + t * (q2 - q1).    
    float4 result = q1 + t * (q2 - q1);    
    return normalize(result);
}

// @brief Spherical linear interpolation between two quaternions (SLERP)
// @param q1 Start quaternion
// @param q2 End quaternion
// @param t Interpolation parameter [0, 1]
// @return Interpolated quaternion
[Differentiable]
float4 slerp(float4 q1, float4 q2, float t) {
    float dot_product = dot(q1, q2);
    
    // Handle quaternions on opposite hemispheres
    if (dot_product < 0.0f) {
        q2 = -q2;
        dot_product = -dot_product;
    }
    
    // Clamp dot product to avoid numerical issues
    dot_product = clamp(dot_product, -1.0f, 1.0f);
    
    // Use LERP for very similar quaternions
    if (dot_product > 0.9995f) {
        return lerp(q1, q2, t);
    }
    
    float theta = acos(dot_product);
    float sin_theta = sin(theta);
    float w1 = sin((1.0f - t) * theta) / sin_theta;
    float w2 = sin(t * theta) / sin_theta;
    
    return w1 * q1 + w2 * q2;
}

// @brief Manifold interpolation using SO(3) operations
// @param q1 Start quaternion
// @param q2 End quaternion
// @param t Interpolation parameter [0, 1]
// @return Interpolated quaternion
[Differentiable]
float4 manifoldInterp(float4 q1, float4 q2, float t) {
    float4 q1_inv = inverse(q1);
    float4 q_relative = multiply(q1_inv, q2);
    float3 omega = so3Log(q_relative);
    float4 q_interp = so3Exp(t * omega);
    return multiply(q1, q_interp);
}

// @brief Apply quaternion rotation to a vector
// @param q The quaternion
// @param v The vector to rotate
// @return The rotated vector
[Differentiable]
float3 rotateVector(float4 q, float3 v) {
    // Efficient rotation using quaternion formula: q * v * q^-1
    float3 qvec = q.xyz;
    float3 uv = cross(qvec, v);
    float3 uuv = cross(qvec, uv);
    return v + 2.0f * (q.w * uv + uuv);
}

// @brief Compute angular distance between two quaternions
// @param q1 First quaternion
// @param q2 Second quaternion
// @return Angular distance in radians
[Differentiable]
float angular_distance(float4 q1, float4 q2) {
    float dot_product = abs(dot(normalize(q1), normalize(q2)));
    dot_product = clamp(dot_product, 0.0f, 1.0f);
    return 2.0f * acos(dot_product);
}

// @brief Create identity quaternion
// @return Identity quaternion (no rotation)
[Differentiable]
float4 identity() {
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}

// @brief Convert quaternion from wxyz to xyzw format if needed
// @param quat Input quaternion
// @param wxyz_format If true, input is (w,x,y,z) and will be converted to (x,y,z,w)
// @return Quaternion in xyzw format
[Differentiable]
float4 wxyzToXyzw(float4 quat, bool wxyz_format) {
    if (wxyz_format) {
        // wxyz -> xyzw: (w, x, y, z) -> (x, y, z, w)
        return float4(quat.y, quat.z, quat.w, quat.x);
    }
    return quat;
}

// @brief Compute covariance matrix from quaternion and scale
// Covariance = R * S * S^T * R^T = (R * S) * (R * S)^T
// where R is the rotation matrix from quaternion and S is the diagonal scale matrix
// @param quat Quaternion in xyzw format (must be normalized)
// @param scale Scale vector (3D)
// @return 3x3 covariance matrix
[Differentiable]
float3x3 quatScaleToCovariance(float4 quat, float3 scale) {
    // Convert quaternion to rotation matrix
    float3x3 R = convertNormalizedQuatToMatrix(quat);

    // M = R * S where S is diagonal scale matrix
    // This scales column j of R by scale[j]
    // In row-major storage: M[row][col] = R[row][col] * scale[col]
    float3x3 M;
    M[0] = float3(R[0][0] * scale.x, R[0][1] * scale.y, R[0][2] * scale.z);
    M[1] = float3(R[1][0] * scale.x, R[1][1] * scale.y, R[1][2] * scale.z);
    M[2] = float3(R[2][0] * scale.x, R[2][1] * scale.y, R[2][2] * scale.z);

    // Covariance = M * M^T
    // Since Slang uses row-major matrices, we compute M * transpose(M)
    float3x3 covar;
    [ForceUnroll]
    for (int i = 0; i < 3; i++) {
        [ForceUnroll]
        for (int j = 0; j < 3; j++) {
            covar[i][j] = M[i][0] * M[j][0] + M[i][1] * M[j][1] + M[i][2] * M[j][2];
        }
    }

    return covar;
}

/// Quaternion struct with member functions for rotation operations
/// Format: (x, y, z, w) where w is the scalar component
struct Quat : IDifferentiable {
    float4 q;  // Quaternion components (x, y, z, w)

    // Constructors
    [Differentiable]
    __init(float4 components) {
        q = components;
    }

    [Differentiable]
    __init(float x, float y, float z, float w) {
        q = float4(x, y, z, w);
    }

    // Static factory methods
    [Differentiable]
    static Quat identity() {
        return Quat(0.0f, 0.0f, 0.0f, 1.0f);
    }

    [Differentiable]
    static Quat fromAxisAngle(float3 axis, float angle) {
        return Quat(quaternion::convertAngleToQuat(axis, angle));
    }

    // Member functions - delegate to standalone functions
    [Differentiable]
    Quat normalizeSafe() {
        return Quat(quaternion::normalizeSafe(this.q));
    }

    [Differentiable]
    Quat normalizeFast() {
        return Quat(quaternion::normalizeFast(this.q));
    }

    [Differentiable]
    Quat conjugate() {
        return Quat(quaternion::conjugate(this.q));
    }

    [Differentiable]
    Quat inverse() {
        return Quat(quaternion::inverse(this.q));
    }

    [Differentiable]
    Quat inverseSafe() {
        return Quat(quaternion::inverseSafe(this.q));
    }

    [Differentiable]
    Quat multiply(Quat other) {
        return Quat(quaternion::multiply(this.q, other.q));
    }

    [Differentiable]
    float3 rotateVector(float3 vec) {
        return quaternion::rotateVector(this.q, vec);
    }

    [Differentiable]
    float3x3 toMatrix() {
        return quaternion::toMatrix(this.q);
    }

    [Differentiable]
    static Quat slerp(Quat q1, Quat q2, float t) {
        return Quat(quaternion::slerp(q1.q, q2.q, t));
    }

    [Differentiable]
    static Quat lerp(Quat q1, Quat q2, float t) {
        return Quat(quaternion::lerp(q1.q, q2.q, t));
    }

    [Differentiable]
    float angularDistance(Quat other) {
        return quaternion::angular_distance(this.q, other.q);
    }
}

// Operator overloads for Quat
[Differentiable]
Quat operator*(Quat a, Quat b) {
    return a.multiply(b);
}

} // namespace quaternion