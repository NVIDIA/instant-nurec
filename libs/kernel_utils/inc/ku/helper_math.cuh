/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#pragma once

#include <array> // used for Mat3

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b) {
    return a < b ? a : b;
}

inline float fmaxf(float a, float b) {
    return a > b ? a : b;
}

inline int max(int a, int b) {
    return a > b ? a : b;
}

inline int min(int a, int b) {
    return a < b ? a : b;
}

inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s) {
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a) {
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float3 make_float3(float s) {
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a) {
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s) {
    return make_float3(a.x, a.y, s);
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a) {
    return make_float2(-a.x, -a.y);
}

inline __host__ __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float4 operator-(float4 a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2& a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b) {
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
}
inline __host__ __device__ float2 operator+(float b, float2 a) {
    return make_float2(a.x + b, a.y + b);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}
inline __host__ __device__ float3 operator+(float b, float3 a) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4& a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}
inline __host__ __device__ float4 operator+(float b, float4 a) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2& a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b) {
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a) {
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2& a, float b) {
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3& a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a) {
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float4& a, float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ float4 operator-(float b, float4 a) {
    return make_float4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(float4& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2& a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a) {
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3& a, float3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float4& a, float4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b) {
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2& a, float2 b) {
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2& a, float b) {
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a) {
    return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3& a, float3 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a) {
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(float4& a, float4 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(float4& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a) {
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// compare less than
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ bool operator<(float2 a, float2 b) {
    return (a.x < b.x) && (a.y < b.y);
}

inline __host__ __device__ bool operator<(float3 a, float3 b) {
    return (a.x < b.x) && (a.y < b.y) && (a.z < b.z);
}

inline __host__ __device__ bool operator<(float4 a, float4 b) {
    return (a.x < b.x) && (a.y < b.y) && (a.z < b.z) && (a.w < b.w);
}

////////////////////////////////////////////////////////////////////////////////
// compare greater than
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ bool operator>(float2 a, float2 b) {
    return (a.x > b.x) && (a.y > b.y);
}

inline __host__ __device__ bool operator>(float3 a, float3 b) {
    return (a.x > b.x) && (a.y > b.y) && (a.z > b.z);
}

inline __host__ __device__ bool operator>(float4 a, float4 b) {
    return (a.x > b.x) && (a.y > b.y) && (a.z > b.z) && (a.w > b.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 fminf(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 fmaxf(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b) {
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b) {
    return max(a, min(f, b));
}

inline __device__ __host__ float3 clamp(float3 v, float a, float b) {
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b) {
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float length(float3 v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float length(float4 v) {
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float3 normalize(float3 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

inline __host__ __device__ float4 normalize(float4 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n) {
    return i - 2.0f * n * dot(n, i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x) {
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y * y * (make_float3(3.0f) - (make_float3(2.0f) * y)));
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : 3x3 matrix
////////////////////////////////////////////////////////////////////////////////
using Mat3 = std::array<float3, 3>; // each element is one row of the matrix

////////////////////////////////////////////////////////////////////////////////
// Mat3 : identity
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 identity() {
    return Mat3{make_float3(1.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), make_float3(0.f, 0.f, 1.f)};
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : skew symmetric
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 skew_symmetric(float3 const& t) {
    return Mat3{
        make_float3(0.f, -t.z, t.y),
        make_float3(t.z, 0.f, -t.x),
        make_float3(-t.y, t.x, 0.f)};
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : transpose
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 transpose_matrix(Mat3 const& M) {
    auto ret = Mat3{};

    ret[0] = make_float3(M[0].x, M[1].x, M[2].x);
    ret[1] = make_float3(M[0].y, M[1].y, M[2].y);
    ret[2] = make_float3(M[0].z, M[1].z, M[2].z);

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : add
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 operator+(Mat3 const& A, Mat3 const& B) {
    return Mat3{A[0] + B[0], A[1] + B[1], A[2] + B[2]};
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : subtract
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 operator-(Mat3 const& A, Mat3 const& B) {
    return Mat3{A[0] - B[0], A[1] - B[1], A[2] - B[2]};
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : scalar multiply
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 operator*(float const& a, Mat3 const& A) {
    return Mat3{a * A[0], a * A[1], a * A[2]};
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : multiply
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 operator*(Mat3 const& A, Mat3 const& B) {
    const auto bcol0 = make_float3(B[0].x, B[1].x, B[2].x);
    const auto bcol1 = make_float3(B[0].y, B[1].y, B[2].y);
    const auto bcol2 = make_float3(B[0].z, B[1].z, B[2].z);
    return Mat3{
        make_float3(dot(A[0], bcol0), dot(A[0], bcol1), dot(A[0], bcol2)),
        make_float3(dot(A[1], bcol0), dot(A[1], bcol1), dot(A[1], bcol2)),
        make_float3(dot(A[2], bcol0), dot(A[2], bcol1), dot(A[2], bcol2))};
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : vector multiply (inner product)
////////////////////////////////////////////////////////////////////////////////

inline __device__ float3 apply_matrix(Mat3 const& M, float3 const& x) {
    return make_float3(dot(M[0], x), dot(M[1], x), dot(M[2], x));
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : vector multiply (outer product)
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 outer_product(float3 const& t) {
    return Mat3{
        make_float3(t.x * t.x, t.x * t.y, t.x * t.z),
        make_float3(t.y * t.x, t.y * t.y, t.y * t.z),
        make_float3(t.z * t.x, t.z * t.y, t.z * t.z)};
}

////////////////////////////////////////////////////////////////////////////////
// Mat3 : rotation matrix from quaternion
////////////////////////////////////////////////////////////////////////////////

inline __device__ Mat3 unitquat_rotmatrix(float4 const& q) {
    auto ret = Mat3{};

    auto const x = q.x;
    auto const y = q.y;
    auto const z = q.z;
    auto const w = q.w;

    auto const xx = x * x;
    auto const yy = y * y;
    auto const zz = z * z;
    auto const ww = w * w;

    ret[0] = make_float3(xx - yy - zz + ww, 2 * (x * y - z * w), 2 * (x * z + y * w));
    ret[1] = make_float3(2 * (x * y + z * w), -xx + yy - zz + w * w, 2 * (y * z - x * w));
    ret[2] = make_float3(2 * (x * z - y * w), 2 * (y * z + x * w), -xx - yy + zz + ww);

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// quaternion : slerp interpolation
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float4 unitquat_slerp(float4 const& q_start, float4 q_end, float t) {
    // omega is the 'angle' between both quaternions
    auto cos_omega = dot(q_start, q_end);

    // flip quaternions with negative angle to perform shortest arc interpolation
    if (cos_omega < 0.0f) {
        cos_omega *= -1.f;
        q_end *= -1.f;
    }

    // true if q_start and q_end are close
    auto const nearby_quaternions = cos_omega > (1.0f - 1e-3);

    // General approach (use linear interpolation for nearby quaternions)
    auto const omega = acos(cos_omega);
    auto const alpha = nearby_quaternions ? (1.f - t) : sin((1.f - t) * omega);
    auto const beta  = nearby_quaternions ? t : sin(t * omega);

    // Interpolate
    auto const ret = normalize(alpha * q_start + beta * q_end);

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// quaternion : log(q0^-1 * q1) for unit quaternions q0, q1
////////////////////////////////////////////////////////////////////////////////

inline __device__ float3 log_q0invq1(float4 const& q0, float4 const& q1) {
    // compute log(q0^-1 * q1) for unit quaternions q0, q1
    auto q0invq1 = make_float4(
        q0.w * q1.x + q0.z * q1.y - q0.y * q1.z - q0.x * q1.w,
        -q0.z * q1.x + q0.w * q1.y + q0.x * q1.z - q0.y * q1.w,
        q0.y * q1.x - q0.x * q1.y + q0.w * q1.z - q0.z * q1.w,
        q0.x * q1.x + q0.y * q1.y + q0.z * q1.z + q0.w * q1.w);

    // https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf
    // Eq(16a) through Eq(16c) -- numerically more stable!

    if (q0invq1.w < 0.f) {
        // flip sign of q0invq1 to ensure it is the shortest path
        q0invq1 = -q0invq1;
    }

    auto const qr      = q0invq1.w;
    auto const qv      = make_float3(q0invq1.x, q0invq1.y, q0invq1.z);
    auto const qv_norm = length(qv);

    if (qv_norm < 1e-6) {
        auto const coeff = 2.0f / qr - (2.0f / 3.0f) * (qv_norm * qv_norm) / (qr * qr * qr);
        return coeff * qv;
    } else {
        auto const at = qv_norm / (qr + sqrtf(qr * qr + qv_norm * qv_norm));
        return 4.0f * atanf(at) * qv / qv_norm;
    }
}

////////////////////////////////////////////////////////////////////////////////
// quaternion : apply to point
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float3 apply_quaternion(float4 const& q, float3 const& p) {
    // Quaternion rotation matrix coefficients
    auto const xx = q.x * q.x;
    auto const yy = q.y * q.y;
    auto const zz = q.z * q.z;
    auto const xy = q.x * q.y;
    auto const xz = q.x * q.z;
    auto const yz = q.y * q.z;
    auto const wx = q.w * q.x;
    auto const wy = q.w * q.y;
    auto const wz = q.w * q.z;

    // Apply quaternion rotation to point
    auto const &p_x = p.x, p_y = p.y, p_z = p.z;
    auto const x = p_x * (1 - 2 * yy - 2 * zz) + p_y * (2 * xy - 2 * wz) + p_z * (2 * xz + 2 * wy);
    auto const y = p_x * (2 * xy + 2 * wz) + p_y * (1 - 2 * xx - 2 * zz) + p_z * (2 * yz - 2 * wx);
    auto const z = p_x * (2 * xz - 2 * wy) + p_y * (2 * yz + 2 * wx) + p_z * (1 - 2 * xx - 2 * yy);

    return make_float3(x, y, z);
}

inline __device__ __host__ float4 conjugate_quaternion(float4 const& q) {
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

////////////////////////////////////////////////////////////////////////////////
// covariance matrix : convert quaternion and scale to covariance matrix
////////////////////////////////////////////////////////////////////////////////
/**
 * Quaternion convention:
 *   - The input quaternion q is expected in (x, y, z, w) format,
 *
 * Args:
 *   q: float4 quaternion (x, y, z, w)
 *   s: float3 scale vector (sx, sy, sz)
 */
inline __device__ Mat3 quat_scale_to_covar(float4 const& q, float3 const& s) {
    // Convert quaternion to rotation matrix
    Mat3 R = unitquat_rotmatrix(q);

    // Construct diagonal scale matrix S
    Mat3 S = {
        make_float3(s.x, 0.f, 0.f),
        make_float3(0.f, s.y, 0.f),
        make_float3(0.f, 0.f, s.z)};

    Mat3 M = R * S;
    return M * transpose_matrix(M);
}