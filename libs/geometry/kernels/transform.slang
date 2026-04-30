// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

#include <slang_utils/constants.slang>
#include <geometry/kernels/quaternion.slang>

// This file contains the implementation of 3D transformation.
//
// It is heavily based on LieTorch, since it's meant to be used as
// its replacement.
//
// LieTorch implements specific gradient backward propagation instead
// of relying on naive autodiff approaches.  By propagating the
// gradient in tangent space, it claims to provide better numerical
// stability amongst other things.
//
// See "Tangent Space Backpropagation for 3D Transformation Groups"
// (https://arxiv.org/abs/2103.12032) for more details.
//
// Note: Only the forward pass is currently done, the backward pass
// in tangent space has not been implemented.  Still, we do the
// from_vec() / to_vec() approach (to convert from Euclidean embedding
// to one that keeps the gradient in tangent space) to prepare for the
// implementation which will do the backward pass differently.  Slang
// has mechanisms to support this by definiting custom differentiable
// types and custom associated derivate type.
//
// For the type being, from_vec() / to_vec() are no-op, but when tangent
// space propagation is implemented, they would convert the gradient
// to / from tangent space.

namespace transform {

static const float EPS = 1e-6;

struct Point : IDifferentiable
{
    float3 p;
};

struct SO3 : IDifferentiable
{
    quaternion::Quat unit_quaternion;

    // To avoid redundant normalization, it is the caller's responsibility to
    // provide a normalized quaternion.
    [Differentiable]
    static SO3 identity()
    {
        return SO3.from_vec(quaternion::identity());
    }

    [Differentiable]
    static SO3 from_vec(float4 quat)
    {
        return SO3(quaternion::Quat(quat));
    }

    [Differentiable]
    float4 to_vec()
    {
        return unit_quaternion.q;
    }

    typedef float3 Tangent;

    [Differentiable]
    SO3 inv()
    {
        return SO3(unit_quaternion.conjugate());
    }

    [Differentiable]
    Tangent log()
    {
        // Implementation taken from lietorch
        const float squared_n = dot(unit_quaternion.q.xyz, unit_quaternion.q.xyz);
        const float w = unit_quaternion.q.w;

        float two_atan_nbyw_by_n;

        /// Atan-based log thanks to
        ///
        /// C. Hertzberg et al.:
        /// "Integrating Generic Sensor Fusion Algorithms with Sound State
        /// Representation through Encapsulation of Manifolds"
        /// Information Fusion, 2011

        if (squared_n < EPS * EPS) {
            // If quaternion is normalized and n=0, then w should be 1;
            // w=0 should never happen here!
            const float squared_w = w * w;
            two_atan_nbyw_by_n =
                2 / w - 2.0/3.0 * squared_n / (w * squared_w);
        } else {
            let n = sqrt(squared_n);
            if (abs(w) < EPS) {
                if (w > 0) {
                    two_atan_nbyw_by_n = constants::pi<float>() / n;
                } else {
                    two_atan_nbyw_by_n = -constants::pi<float>() / n;
                }
            } else {
                two_atan_nbyw_by_n = 2 * atan(n / w) / n;
            }
        }

        return two_atan_nbyw_by_n * unit_quaternion.q.xyz;
    }

    [Differentiable]
    static SO3 exp(Tangent phi)
    {
        // Implementation taken from lietorch
        const float theta2 = dot(phi, phi);
        const float theta = sqrt(theta2);
        float imag_factor;
        float real_factor;

        if (theta < EPS) {
            const float theta4 = theta2 * theta2;
            imag_factor = 0.5 - (1.0/48.0) * theta2 + (1.0/3840.0) * theta4;
            real_factor = 1.0 - (1.0/8.0) * theta2 + (1.0/384.0) * theta4;
        } else {
            imag_factor = sin(0.5 * theta) / theta;
            real_factor = cos(0.5 * theta);
        }

        const float4 q = float4(imag_factor*phi.x, imag_factor*phi.y, imag_factor*phi.z, real_factor);
        return SO3(quaternion::Quat(q));
    }
};

struct SE3 : IDifferentiable
{
    SO3 so3;
    Point trans;

    // To avoid redundant normalization, it is the caller's responsibility to
    // provide a normalized quaternion.
    [Differentiable]
    static SE3 identity()
    {
        return SE3.from_vec(float3(0, 0, 0), quaternion::identity());
    }

    [Differentiable]
    static SE3 from_vec(float3 trans, float4 quat)
    {
        return SE3(SO3(quaternion::Quat(quat)), Point(trans));
    }

    [Differentiable]
    Tuple<float3, float4> to_vec()
    {
        return makeTuple(trans.p, so3.unit_quaternion.q);
    }

    [Differentiable]
    Point translation()
    {
        return trans;
    }
};

}; // Namespace transform


//
// Operators have to be defined outside the namespace to be accessible.
//

[Differentiable]
transform::Point operator+(transform::Point a, transform::Point b)
{
    return transform::Point(a.p + b.p);
}

[Differentiable]
transform::Point operator-(transform::Point a, transform::Point b)
{
    return transform::Point(a.p - b.p);
}

[Differentiable]
transform::Point operator*(float a, transform::Point b)
{
    return transform::Point(a * b.p);
}

[Differentiable]
transform::Point operator*(transform::Point a, float b)
{
    return transform::Point(a.p * b);
}

[Differentiable]
transform::Point operator*(quaternion::Quat a, transform::Point b)
{
    return transform::Point(a.rotateVector(b.p));
}

[Differentiable]
transform::SO3 operator*(transform::SO3 a, transform::SO3 b)
{
    transform::SO3 output;
    output.unit_quaternion = a.unit_quaternion.multiply(b.unit_quaternion);
    return output;
}

[Differentiable]
transform::Point operator*(transform::SO3 a, transform::Point b)
{
    return a.unit_quaternion * b;
}

[Differentiable]
transform::SE3 operator*(transform::SE3 a, transform::SE3 b)
{
    return transform::SE3(a.so3 * b.so3, a.trans + a.so3 * b.trans);
}

[Differentiable]
transform::Point operator*(transform::SE3 a, transform::Point b)
{
    return a.so3 * b + a.trans;
}


//
// Interpolation functions.
//

namespace transform {

[Differentiable]
Point interpolate(Point a, Point b, float alpha) {
    return (1 - alpha) * a + alpha * b;
}

[Differentiable]
SO3 interpolate(SO3 a, SO3 b, float alpha) {
    return a * SO3.exp(alpha * (a.inv() * b).log());
}

[Differentiable]
SE3 interpolate(SE3 a, SE3 b, float alpha) {
    return SE3(
        interpolate(a.so3, b.so3, alpha),
        interpolate(a.trans, b.trans, alpha),
     );
}

}; // Namespace transform

//
// 4x4 Matrix <=> SE3 conversion functions.
//
namespace transform {

// @brief Create SE3 from 4x4 transformation matrix (row-major)
// Extracts translation from 4th column, rotation from upper-left 3x3
// @param mat The 4x4 transformation matrix (row-major: mat[row][col])
// @return SE3 transformation
[Differentiable]
SE3 convertMatrixToSE3(float4x4 mat)
{
    // Extract translation from 4th column: mat[0][3], mat[1][3], mat[2][3]
    float3 trans = float3(mat[0][3], mat[1][3], mat[2][3]);
    
    // Extract upper-left 3x3 rotation matrix
    float3x3 R = float3x3(
        mat[0].xyz,
        mat[1].xyz,
        mat[2].xyz
    );
    
    // Convert rotation matrix to quaternion using Shepperd's method
    float4 quat = quaternion::fromMatrix(R);
    
    return SE3(SO3(quaternion::Quat(quat)), Point(trans));
}

// @brief Convert SE3 to 4x4 transformation matrix (row-major)
// Standard format: rotation in upper-left 3x3, translation in 4th column
[Differentiable]
float4x4 convertSE3ToMatrix(SE3 se3)
{
    let so3 = se3.so3;
    let trans = se3.trans;
    float3x3 R = quaternion::toMatrix(so3.unit_quaternion.q);
    return float4x4(
        float4(R[0], trans.p.x),
        float4(R[1], trans.p.y),
        float4(R[2], trans.p.z),
        float4(0.0, 0.0, 0.0, 1.0)
    );
}

}; // Namespace transform