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

#include <limits>

#define MAX_INT32 0x7FFFFFFF

using float6   = float[6];
using double44 = double[4][4];
using float44  = float[4][4];
// row-major float4 matrix
struct float4MatRM {
    float4 r0;
    float4 r1;
    float4 r2;
    float4 r3;
};

// camera FTheta distortion parameters
struct FThetaParams {
    // 1.0 / (FThetaWidth, FThetaHeight)
    float2 invSize;
    // length( (FThetaWidth, FThetaHeight) ) * .5f;
    float scale;
    // normalized distortion center
    float2 ncenter;
    // (FThetaPoly_a, FThetaPoly_b, FThetaPoly_c, FThetaPoly_d, FThetaPoly_e)
    float polyc[5];
};

// ===============================================================
// see https://developer.nvidia.com/blog/lerp-faster-cuda/
template <typename T>
inline __device__ T lerp(T x, T y, T a) {
    // return (1.0 - a) * x + a * y;
    return fma(a, y, fma(-a, x, x));
}

// ===============================================================
// Evaluate an order 2 polynomial
inline __host__ __device__ float poly2Eval(const float c[3], const float x) {
    float y = c[1] + c[2] * x;
    y       = c[0] + x * y;
    return y;
}

// ===============================================================
// Evaluate an order 3 polynomial
inline __host__ __device__ float poly3Eval(const float c[4], const float x) {
    float y = c[2] + c[3] * x;
    y       = c[1] + x * y;
    y       = c[0] + x * y;
    return y;
}

// ===============================================================
// Evaluate an order 4 polynomial
inline __host__ __device__ float poly4Eval(const float c[5], const float x) {
    float y = c[3] + c[4] * x;
    y       = c[2] + x * y;
    y       = c[1] + x * y;
    y       = c[0] + x * y;
    return y;
}

// ===============================================================
// Length of a vector
template <typename T>
inline __host__ __device__ float length(T x) {
    return sqrtf(dot(x, x));
}

// ===============================================================
// Normalize a vector
template <typename T>
inline __host__ __device__ T normalized(T x) {
    return x * rsqrtf(dot(x, x));
}

// ===============================================================
// Convert by reducing the dimension
template <typename T>
inline __host__ __device__ float2 make_float2(T x) {
    return make_float2(x.x, x.y);
}

template <typename T>
inline __host__ __device__ float3 make_float3(T x) {
    return make_float3(x.x, x.y, x.z);
}

// ===============================================================
// Perform smoothstep
template <typename T>
inline __host__ __device__ T smoothstep(T edge0, T edge1, T x) {
    const T t = clamp((x - edge0) / (edge1 - edge0), (T)0.0, (T)1.0);
    return t * t * (3.0 - 2.0 * t);
}

// ===============================================================
// Return the dot product of the 3 first component of the float4
inline __host__ __device__ float dot3(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ===============================================================
// Return the dot product of two float4s
inline __host__ __device__ float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// ===============================================================
// Return the elementwise sum of two float4s
inline __host__ __device__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// ===============================================================
// Return the vector product of two float4s
inline __host__ __device__ float4 operator*(const float4& a, const float& s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

// ===============================================================
// Return the product of a float4 vector and a 4x4 float matrix
inline __host__ __device__ float4 operator*(const float4& p, const float44& m) {
    float4 result;
    result.x = p.x * m[0][0] + p.y * m[1][0] + p.z * m[2][0] + p.w * m[3][0];
    result.y = p.x * m[0][1] + p.y * m[1][1] + p.z * m[2][1] + p.w * m[3][1];
    result.z = p.x * m[0][2] + p.y * m[1][2] + p.z * m[2][2] + p.w * m[3][2];
    result.w = p.x * m[0][3] + p.y * m[1][3] + p.z * m[2][3] + p.w * m[3][3];
    return result;
}

// ===============================================================
// Return the product of a float4 vector and a 4x4 float matrix
// represented as float16 (for using as kernel argument)
inline __host__ __device__ float4 operator*(const float4& p, const float4MatRM& m) {
    float4 result;
    result.x = p.x * m.r0.x + p.y * m.r1.x + p.z * m.r2.x + p.w * m.r3.x;
    result.y = p.x * m.r0.y + p.y * m.r1.y + p.z * m.r2.y + p.w * m.r3.y;
    result.z = p.x * m.r0.z + p.y * m.r1.z + p.z * m.r2.z + p.w * m.r3.z;
    result.w = p.x * m.r0.w + p.y * m.r1.w + p.z * m.r2.w + p.w * m.r3.w;
    return result;
}

// ===============================================================
// Return a float4 with it's w-scale applied and set to 1.0
inline __host__ __device__ float4 homogenize(const float4& p) {
    float inv = (abs(p.w)) > 1e-6f ? 1.0f / p.w : 1.0f;
    return p * inv;
}

// ===============================================================
// Return the dot product of two float3s
inline __host__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ===============================================================
// Return the elementwise product of two float3s
inline __host__ __device__ float3 hadamard_product(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// ===============================================================
// Return the cross product of two float3s
inline __host__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// ===============================================================
// Return the elementwise sum of two float3s
inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// ===============================================================
// Return the elementwise difference between two float3s
inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// ===============================================================
// Return the product of a float3 and a constant
inline __host__ __device__ float3 operator*(const float3& a, const float& s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// ===============================================================
// Return the elementwise sum of two float2s
inline __host__ __device__ float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// ===============================================================
// Return the elementwise difference between two float2s
inline __host__ __device__ float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// ===============================================================
// Return the product of a float2 and a constant
inline __host__ __device__ float2 operator*(const float2& a, const float& s) {
    return make_float2(a.x * s, a.y * s);
}

// ===============================================================
// Return the element-wise product between two float2s
inline __host__ __device__ float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

// ===============================================================
// Return the division between two float2s
inline __host__ __device__ float2 operator/(const float2& a, const float2& b) {
    return make_float2(a.x / b.x, a.y / b.y);
}

// ===============================================================
// Return the dot product of two float2s
inline __host__ __device__ float dot(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}

// ===============================================================
// Return the jet-map color mapping of the input value
inline __device__ float4 jet_map(float v) {
    const float vs = __saturatef(v);
    return make_float4(__saturatef(4.0f * (vs - 0.375f)) * __saturatef(-4.0f * (vs - 1.125f)),
                       __saturatef(4.0f * (vs - 0.125f)) * __saturatef(-4.0f * (vs - 0.875f)),
                       __saturatef(4.0f * vs + 0.5f) * __saturatef(-4.0f * (vs - 0.625f)), 1.0f);
}

// ===============================================================
// Convert log rgb to linear
inline __device__ float srgbToLinear(float srgb) {
    return (srgb <= 0.04045f) ? srgb / 12.92f : powf((srgb + 0.055f) / 1.055f, 2.4f);
}

// ===============================================================
// Convert linear rgb to log
inline __device__ float linearToSrgb(float linear) {
    return (linear < 0.0031308f) ? 12.92f * linear : 1.055f * powf(linear, 0.41666f) - 0.055f;
}

// ===============================================================
// Implementation of the atomicMinFloat using ordered int
__forceinline__ __device__ void atomicMinFloat(float* addr, float value) {
    if (value >= 0)
        __int_as_float(atomicMin((int*)addr, __float_as_int(value)));
    else
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
}

// ===============================================================
// Implementation of the atomicMaxFloat using ordered int
__forceinline__ __device__ void atomicMaxFloat(float* addr, float value) {
    if (value >= 0)
        __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
    else
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}
