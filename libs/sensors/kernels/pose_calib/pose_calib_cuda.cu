// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// Hand-written CUDA translation of pose_calib.slang.
//
// Provides fused pose calibration and rolling shutter interpolation:
//   1. Load start/end 4x4 pose matrices
//   2. Convert to SE3 (translation + quaternion)
//   3. Optional: apply calibration delta (6D rotation -> matrix -> quaternion, right-multiply)
//   4. Optional: rolling shutter interpolation via slerp + lerp
//   5. Convert SE3 back to 4x4 matrix, store outputs + timestamps

#include "pose_calib_cuda.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

static constexpr uint32_t BLOCK_THREADS     = 256;
static constexpr uint32_t BLOCK_THREADS_BWD = 128;
static constexpr float QUAT_EPSILON         = 1e-7f;

// ============================================================================
// Device helpers
// ============================================================================

// Normalize quaternion (xyzw) with safety check for near-zero norm.
__device__ __forceinline__ float4 quat_normalize_safe(float4 q) {
    float norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (fabsf(norm_sq) < QUAT_EPSILON) {
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float inv_norm = 1.0f / sqrtf(norm_sq);
    return make_float4(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm, q.w * inv_norm);
}

// Hamilton product: q1 * q2 (quaternion multiply, xyzw format).
__device__ __forceinline__ float4 quat_multiply(float4 q1, float4 q2) {
    float3 v1 = make_float3(q1.x, q1.y, q1.z);
    float3 v2 = make_float3(q2.x, q2.y, q2.z);
    // cross(v1, v2)
    float3 cr = make_float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
    float3 xyz = make_float3(
        q1.w * v2.x + q2.w * v1.x + cr.x,
        q1.w * v2.y + q2.w * v1.y + cr.y,
        q1.w * v2.z + q2.w * v1.z + cr.z);
    float w = q1.w * q2.w - (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}

// Rotate vector by quaternion: q * v * q^-1 (assumes unit quaternion).
__device__ __forceinline__ float3 quat_rotate_vector(float4 q, float3 v) {
    float3 qv = make_float3(q.x, q.y, q.z);
    // uv = cross(qv, v)
    float3 uv = make_float3(
        qv.y * v.z - qv.z * v.y,
        qv.z * v.x - qv.x * v.z,
        qv.x * v.y - qv.y * v.x);
    // uuv = cross(qv, uv)
    float3 uuv = make_float3(
        qv.y * uv.z - qv.z * uv.y,
        qv.z * uv.x - qv.x * uv.z,
        qv.x * uv.y - qv.y * uv.x);
    return make_float3(
        v.x + 2.0f * (q.w * uv.x + uuv.x),
        v.y + 2.0f * (q.w * uv.y + uuv.y),
        v.z + 2.0f * (q.w * uv.z + uuv.z));
}

// Convert normalized quaternion (xyzw) to 3x3 rotation matrix (row-major, stored in 3 float3s).
__device__ __forceinline__ void quat_to_matrix(
    float4 q,
    float3& row0, float3& row1, float3& row2) {
    float x2 = q.x * q.x, y2 = q.y * q.y, z2 = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, xw = q.x * q.w;
    float yz = q.y * q.z, yw = q.y * q.w, zw = q.z * q.w;
    row0 = make_float3(1.0f - 2.0f * (y2 + z2), 2.0f * (xy - zw), 2.0f * (xz + yw));
    row1 = make_float3(2.0f * (xy + zw), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz - xw));
    row2 = make_float3(2.0f * (xz - yw), 2.0f * (yz + xw), 1.0f - 2.0f * (x2 + y2));
}

// Convert 3x3 rotation matrix to quaternion (xyzw) using Shepperd's method.
// R is row-major: R[i] is row i, R[i].{x,y,z} are columns 0,1,2.
__device__ __forceinline__ float4 matrix_to_quat(float3 R0, float3 R1, float3 R2) {
    float r00 = R0.x, r11 = R1.y, r22 = R2.z;
    float trace = r00 + r11 + r22;
    float4 quat;

    if (trace > r00 && trace > r11 && trace > r22) {
        float s = sqrtf(1.0f + trace) * 2.0f;
        quat    = make_float4(
            (R2.y - R1.z) / s,
            (R0.z - R2.x) / s,
            (R1.x - R0.y) / s,
            0.25f * s);
    } else if (r00 > r11 && r00 > r22) {
        float s = sqrtf(1.0f + r00 - r11 - r22) * 2.0f;
        quat    = make_float4(
            0.25f * s,
            (R0.y + R1.x) / s,
            (R0.z + R2.x) / s,
            (R2.y - R1.z) / s);
    } else if (r11 > r22) {
        float s = sqrtf(1.0f + r11 - r00 - r22) * 2.0f;
        quat    = make_float4(
            (R0.y + R1.x) / s,
            0.25f * s,
            (R1.z + R2.y) / s,
            (R0.z - R2.x) / s);
    } else {
        float s = sqrtf(1.0f + r22 - r00 - r11) * 2.0f;
        quat    = make_float4(
            (R0.z + R2.x) / s,
            (R1.z + R2.y) / s,
            0.25f * s,
            (R1.x - R0.y) / s);
    }
    return quat_normalize_safe(quat);
}

// SLERP between two quaternions (xyzw), handles hemisphere and near-parallel fallback.
// Does NOT normalize the result — store_se3_as_matrix handles final normalization,
// matching Slang's single normalizeSafe in convertSE3ToMatrix.
__device__ __forceinline__ float4 quat_slerp(float4 q1, float4 q2, float t) {
    float dp = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
    if (dp < 0.0f) {
        q2 = make_float4(-q2.x, -q2.y, -q2.z, -q2.w);
        dp = -dp;
    }
    dp = fminf(fmaxf(dp, -1.0f), 1.0f);

    if (dp > 0.9995f) {
        // Near-parallel: fall back to LERP (unnormalized). store_se3_as_matrix
        // calls quat_normalize_safe on the result anyway, so normalizing here
        // was double work. Matches the SLERP branch's "no normalize" policy.
        return make_float4(
            q1.x + t * (q2.x - q1.x),
            q1.y + t * (q2.y - q1.y),
            q1.z + t * (q2.z - q1.z),
            q1.w + t * (q2.w - q1.w));
    }

    float theta     = acosf(dp);
    float sin_theta = sinf(theta);
    float w1        = sinf((1.0f - t) * theta) / sin_theta;
    float w2        = sinf(t * theta) / sin_theta;
    return make_float4(
        w1 * q1.x + w2 * q2.x,
        w1 * q1.y + w2 * q2.y,
        w1 * q1.z + w2 * q2.z,
        w1 * q1.w + w2 * q2.w);
}

// Convert 6D rotation representation to 3x3 matrix via Gram-Schmidt.
// col1_offset, col2_offset are offsets from identity columns [1,0,0] and [0,1,0].
// Returns rows of the rotation matrix.
__device__ __forceinline__ void rotation6d_to_matrix(
    float3 col1_offset, float3 col2_offset,
    float3& row0, float3& row1, float3& row2) {
    float3 a1 = make_float3(1.0f + col1_offset.x, col1_offset.y, col1_offset.z);
    float3 a2 = make_float3(col2_offset.x, 1.0f + col2_offset.y, col2_offset.z);

    // b1 = normalize(a1)
    float len_sq_a1  = a1.x * a1.x + a1.y * a1.y + a1.z * a1.z;
    float inv_len_a1 = len_sq_a1 > QUAT_EPSILON ? rsqrtf(len_sq_a1) : 0.0f;
    float3 b1        = make_float3(a1.x * inv_len_a1, a1.y * inv_len_a1, a1.z * inv_len_a1);

    // b2 = normalize(a2 - dot(a2, b1) * b1)
    float a2_dot_b1 = a2.x * b1.x + a2.y * b1.y + a2.z * b1.z;
    float3 a2_orth  = make_float3(
        a2.x - a2_dot_b1 * b1.x,
        a2.y - a2_dot_b1 * b1.y,
        a2.z - a2_dot_b1 * b1.z);
    float len_sq_a2o  = a2_orth.x * a2_orth.x + a2_orth.y * a2_orth.y + a2_orth.z * a2_orth.z;
    float inv_len_a2o = len_sq_a2o > QUAT_EPSILON ? rsqrtf(len_sq_a2o) : 0.0f;
    float3 b2         = make_float3(a2_orth.x * inv_len_a2o, a2_orth.y * inv_len_a2o, a2_orth.z * inv_len_a2o);

    // b3 = cross(b1, b2)
    float3 b3 = make_float3(
        b1.y * b2.z - b1.z * b2.y,
        b1.z * b2.x - b1.x * b2.z,
        b1.x * b2.y - b1.y * b2.x);

    // Rotation matrix with b1, b2, b3 as rows (Slang row-major convention)
    row0 = b1;
    row1 = b2;
    row2 = b3;
}

// Fused 6D→quat: builds the 3x3 matrix and converts to quat in a single
// expression tree so nvcc can retire the 9-float row temporaries within
// matrix_to_quat's 4-branch trace selection instead of keeping them live
// across the named locals in the caller. Matches Slang's single-expression
// quaternion_fromMatrix(rotation6dToMatrix(...)) codegen.
__device__ __forceinline__ float4 rotation6d_to_quat(float3 col1_offset, float3 col2_offset) {
    float3 R0, R1, R2;
    rotation6d_to_matrix(col1_offset, col2_offset, R0, R1, R2);
    return matrix_to_quat(R0, R1, R2);
}

// Compute relative frame time [0,1] based on shutter type and image coordinates.
// Not differentiable (floor/ceil are discontinuous).
__device__ __forceinline__ float compute_relative_frame_time(
    float2 image_point, uint2 res, ShutterType shutter_type) {
    float safe_w = float(max(res.x, 2u) - 1u);
    float safe_h = float(max(res.y, 2u) - 1u);

    if (shutter_type == ShutterType::GLOBAL) {
        return 0.0f;
    } else if (shutter_type == ShutterType::ROLLING_TOP_TO_BOTTOM) {
        return floorf(image_point.y) / safe_h;
    } else if (shutter_type == ShutterType::ROLLING_BOTTOM_TO_TOP) {
        return (float(res.y) - ceilf(image_point.y)) / safe_h;
    } else if (shutter_type == ShutterType::ROLLING_LEFT_TO_RIGHT) {
        return floorf(image_point.x) / safe_w;
    } else {
        return (float(res.x) - ceilf(image_point.x)) / safe_w;
    }
}

// SE3 representation: translation (float3) + unit quaternion (float4 xyzw).
struct SE3 {
    float3 trans;
    float4 quat;
};

// SE3 right-multiply: a * b = SE3(a.so3 * b.so3, a.trans + a.so3 * b.trans)
__device__ __forceinline__ SE3 se3_multiply(SE3 a, SE3 b) {
    SE3 result;
    result.quat       = quat_multiply(a.quat, b.quat);
    float3 rotated_bt = quat_rotate_vector(a.quat, b.trans);
    result.trans      = make_float3(
        a.trans.x + rotated_bt.x,
        a.trans.y + rotated_bt.y,
        a.trans.z + rotated_bt.z);
    return result;
}

// Load a 4x4 row-major matrix from tensor[fidx, startend_idx, :, :].
// tensor is [V, 2, 4, 4] float32, contiguous row-major.
__device__ __forceinline__ SE3 load_pose_as_se3(
    const float* __restrict__ poses, int fidx, int startend_idx) {
    // Stride: [fidx * 2*4*4 + startend_idx * 4*4 + row * 4 + col]
    const float* base = poses + (fidx * 2 + startend_idx) * 16;
    float3 R0         = make_float3(base[0], base[1], base[2]);
    float3 R1         = make_float3(base[4], base[5], base[6]);
    float3 R2         = make_float3(base[8], base[9], base[10]);

    SE3 se3;
    se3.trans = make_float3(base[3], base[7], base[11]);
    se3.quat  = matrix_to_quat(R0, R1, R2);
    return se3;
}

// Convert SE3 to 4x4 matrix and store at T_out[tid, startend_idx, :, :].
// Normalizes the quaternion before building the rotation matrix, matching the
// Slang convertSE3ToMatrix which calls quaternion::toMatrix -> normalizeSafe.
// This ensures the CUDA forward and PyTorch backward operate on the same function.
__device__ __forceinline__ void store_se3_as_matrix(
    float* __restrict__ T_out, int tid, int startend_idx, SE3 se3) {
    float3 R0, R1, R2;
    float4 q_norm = quat_normalize_safe(se3.quat);
    quat_to_matrix(q_norm, R0, R1, R2);

    float* base = T_out + (tid * 2 + startend_idx) * 16;
    base[0]     = R0.x;
    base[1]     = R0.y;
    base[2]     = R0.z;
    base[3]     = se3.trans.x;
    base[4]     = R1.x;
    base[5]     = R1.y;
    base[6]     = R1.z;
    base[7]     = se3.trans.y;
    base[8]     = R2.x;
    base[9]     = R2.y;
    base[10]    = R2.z;
    base[11]    = se3.trans.z;
    base[12]    = 0.0f;
    base[13]    = 0.0f;
    base[14]    = 0.0f;
    base[15]    = 1.0f;
}

// Interpolate two SE3 poses: slerp for rotation, lerp for translation.
__device__ __forceinline__ SE3 interpolate_pose(SE3 a, SE3 b, float alpha) {
    SE3 result;
    result.quat  = quat_slerp(a.quat, b.quat, alpha);
    result.trans = make_float3(
        a.trans.x + alpha * (b.trans.x - a.trans.x),
        a.trans.y + alpha * (b.trans.y - a.trans.y),
        a.trans.z + alpha * (b.trans.z - a.trans.z));
    return result;
}

// ============================================================================
// Forward kernel (templated on ENABLE_CALIB and HAS_SUBSAMPLING)
// ============================================================================

// scratch layout when non-null (and ENABLE_CALIB): per-thread 5 float4 =
//   [0] q_calib_start, [1] q_calib_end,
//   [2] q_out_start,   [3] q_out_end,
//   [4] (t0, t1, 0, 0).
// Each row is written as soon as its value is first produced (q_calib right
// after se3_multiply, q_out + t0/t1 after slerp) so no register is held live
// across later stages just to be saved — keeps forward reg count at baseline.
// Backward reads them to skip Stages 1 & 2 recompute.
//
// q_base_start/end are NOT saved: the two matrix_to_quat calls they would
// replace in backward_stage_se3_quat cost ~0.5 us, but holding 2 float4s live
// in backward adds ~10 regs which hurts occupancy more than it saves compute.
template <bool ENABLE_CALIB, bool HAS_SUBSAMPLING>
__global__ void pose_calib_forward_kernel(
    int batch_size,
    const float* __restrict__ T_startend_allviews,   // [V, 2, 4, 4]
    const float* __restrict__ embed_weights,         // [V, 9]
    const int32_t* __restrict__ frame_idx,           // [N]
    const float* __restrict__ rect_points_lb,        // [N, 2, 2]
    const float* __restrict__ resolution,            // [N, 2]
    const int64_t* __restrict__ timestamps_startend, // [V, 2]
    ShutterType shutter_type,
    float* __restrict__ T_out,            // [N, 2, 4, 4]
    int64_t* __restrict__ timestamps_out, // [N, 2]
    float4* __restrict__ scratch) {       // [N, 5] or nullptr
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    int fidx = frame_idx[tid];

    // Load start/end poses as SE3
    SE3 T_start = load_pose_as_se3(T_startend_allviews, fidx, 0);
    SE3 T_end   = load_pose_as_se3(T_startend_allviews, fidx, 1);

    // Load timestamps
    int64_t ts_start = timestamps_startend[fidx * 2 + 0];
    int64_t ts_end   = timestamps_startend[fidx * 2 + 1];

    // Apply calibration delta
    if constexpr (ENABLE_CALIB) {
        const float* ew    = embed_weights + fidx * 9;
        float3 dx          = make_float3(ew[0], ew[1], ew[2]);
        float3 col1_offset = make_float3(ew[3], ew[4], ew[5]);
        float3 col2_offset = make_float3(ew[6], ew[7], ew[8]);

        float4 dq = rotation6d_to_quat(col1_offset, col2_offset);

        SE3 pose_delta;
        pose_delta.trans = dx;
        pose_delta.quat  = dq;

        T_start = se3_multiply(T_start, pose_delta);
        T_end   = se3_multiply(T_end, pose_delta);

        // Save q_calib right after se3_multiply so the values don't need to
        // stay live across the slerp block. Frees ~8 regs in forward.
        if (scratch != nullptr) {
            float4* row = scratch + tid * 5;
            row[0]      = T_start.quat;
            row[1]      = T_end.quat;
        }
    }

    // Apply rolling shutter interpolation
    float t0_saved = 0.0f, t1_saved = 0.0f;
    if constexpr (HAS_SUBSAMPLING) {
        const float* rp   = rect_points_lb + tid * 4;
        float2 rect_start = make_float2(rp[0], rp[1]);
        float2 rect_end   = make_float2(rp[2], rp[3]);

        const float* res = resolution + tid * 2;
        uint2 res_uint   = make_uint2(uint(res[0]), uint(res[1]));

        float t0 = compute_relative_frame_time(rect_start, res_uint, shutter_type);
        float t1 = compute_relative_frame_time(rect_end, res_uint, shutter_type);
        if (t0 > t1) {
            float tmp = t0;
            t0        = t1;
            t1        = tmp;
        }

        SE3 T_start_orig = T_start;
        SE3 T_end_orig   = T_end;
        T_start          = interpolate_pose(T_start_orig, T_end_orig, t0);
        T_end            = interpolate_pose(T_start_orig, T_end_orig, t1);

        int64_t ts_start_orig = ts_start;
        int64_t ts_end_orig   = ts_end;
        ts_start              = ts_start_orig + int64_t(float(ts_end_orig - ts_start_orig) * t0);
        ts_end                = ts_start_orig + int64_t(float(ts_end_orig - ts_start_orig) * t1);

        t0_saved = t0;
        t1_saved = t1;
    }

    // Save q_out (post-slerp, pre-normalize) and t0/t1. For non-sub path,
    // q_out == q_calib so row[2..3] mirror row[0..1].
    if constexpr (ENABLE_CALIB) {
        if (scratch != nullptr) {
            float4* row = scratch + tid * 5;
            row[2]      = T_start.quat;
            row[3]      = T_end.quat;
            row[4]      = make_float4(t0_saved, t1_saved, 0.0f, 0.0f);
        }
    }

    // Convert back to 4x4 and store
    store_se3_as_matrix(T_out, tid, 0, T_start);
    store_se3_as_matrix(T_out, tid, 1, T_end);

    timestamps_out[tid * 2 + 0] = ts_start;
    timestamps_out[tid * 2 + 1] = ts_end;
}

// ============================================================================
// Backward helpers
// ============================================================================

// Backward through quat_to_matrix: given grad_R (3x3 row-major), compute grad_q (xyzw).
// R[i][j] is quadratic in {x,y,z,w}; each partial dR[i][j]/dq_k is linear.
__device__ __forceinline__ float4 quat_to_matrix_backward(
    float4 q,
    float3 gR0, float3 gR1, float3 gR2) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    // grad_x = sum over (i,j) of gR[i][j] * dR[i][j]/dx
    float gx = 2.0f * (
                          /* R01 */ gR0.y * y + /* R02 */ gR0.z * z +
                          /* R10 */ gR1.x * y + /* R11 */ gR1.y * (-2.0f * x) + /* R12 */ gR1.z * (-w) +
                          /* R20 */ gR2.x * z + /* R21 */ gR2.y * w + /* R22 */ gR2.z * (-2.0f * x));
    float gy = 2.0f * (
                          /* R00 */ gR0.x * (-2.0f * y) + /* R01 */ gR0.y * x + /* R02 */ gR0.z * w +
                          /* R10 */ gR1.x * x + /* R12 */ gR1.z * z +
                          /* R20 */ gR2.x * (-w) + /* R21 */ gR2.y * z + /* R22 */ gR2.z * (-2.0f * y));
    float gz = 2.0f * (
                          /* R00 */ gR0.x * (-2.0f * z) + /* R01 */ gR0.y * (-w) + /* R02 */ gR0.z * x +
                          /* R10 */ gR1.x * w + /* R11 */ gR1.y * (-2.0f * z) + /* R12 */ gR1.z * y +
                          /* R20 */ gR2.x * x + /* R21 */ gR2.y * y);
    float gw = 2.0f * (
                          /* R01 */ gR0.y * (-z) + /* R02 */ gR0.z * y +
                          /* R10 */ gR1.x * z + /* R12 */ gR1.z * (-x) +
                          /* R20 */ gR2.x * (-y) + /* R21 */ gR2.y * x);
    return make_float4(gx, gy, gz, gw);
}

// Backward through quat_normalize_safe: given grad on normalized output,
// compute grad on unnormalized input.
// d(q/||q||)/dq = (I - q_hat * q_hat^T) / ||q||
// => grad_in = (grad_out - q_hat * dot(q_hat, grad_out)) / ||q||
__device__ __forceinline__ float4 quat_normalize_safe_backward(
    float4 q_in, float4 grad_out) {
    float norm_sq = q_in.x * q_in.x + q_in.y * q_in.y + q_in.z * q_in.z + q_in.w * q_in.w;
    if (fabsf(norm_sq) < QUAT_EPSILON) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float inv_norm = 1.0f / sqrtf(norm_sq);
    float4 q_hat   = make_float4(q_in.x * inv_norm, q_in.y * inv_norm,
                                 q_in.z * inv_norm, q_in.w * inv_norm);
    float dp       = q_hat.x * grad_out.x + q_hat.y * grad_out.y +
               q_hat.z * grad_out.z + q_hat.w * grad_out.w;
    return make_float4(
        (grad_out.x - q_hat.x * dp) * inv_norm,
        (grad_out.y - q_hat.y * dp) * inv_norm,
        (grad_out.z - q_hat.z * dp) * inv_norm,
        (grad_out.w - q_hat.w * dp) * inv_norm);
}

// Backward through store_se3_as_matrix: extract grad_R (3x3) and grad_trans (3)
// from grad_T_out at [tid, startend_idx, :, :].
// The forward: rows 0-2 contain [R | t], row 3 = [0 0 0 1].
// So grad_R = grad_T_out[:3, :3], grad_t = grad_T_out[:3, 3].
__device__ __forceinline__ void load_grad_T_out(
    const float* __restrict__ grad_T_out, int tid, int startend_idx,
    float3& gR0, float3& gR1, float3& gR2, float3& grad_trans) {
    const float* base = grad_T_out + (tid * 2 + startend_idx) * 16;
    gR0               = make_float3(base[0], base[1], base[2]);
    grad_trans.x      = base[3];
    gR1               = make_float3(base[4], base[5], base[6]);
    grad_trans.y      = base[7];
    gR2               = make_float3(base[8], base[9], base[10]);
    grad_trans.z      = base[11];
}

// Backward through quat_multiply: given grad on result = q1*q2,
// compute grad on q1 and q2.
//
// result.xyz = q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz)
// result.w   = q1.w * q2.w   - dot(q1.xyz, q2.xyz)
//
// d(result)/d(q2) (needed for delta quat grad):
//   d(res.x)/d(q2.x) = q1.w,  d(res.x)/d(q2.y) = -q1.z, d(res.x)/d(q2.z) = q1.y,  d(res.x)/d(q2.w) = q1.x
//   d(res.y)/d(q2.x) = q1.z,  d(res.y)/d(q2.y) = q1.w,  d(res.y)/d(q2.z) = -q1.x, d(res.y)/d(q2.w) = q1.y
//   d(res.z)/d(q2.x) = -q1.y, d(res.z)/d(q2.y) = q1.x,  d(res.z)/d(q2.z) = q1.w,  d(res.z)/d(q2.w) = q1.z
//   d(res.w)/d(q2.x) = -q1.x, d(res.w)/d(q2.y) = -q1.y, d(res.w)/d(q2.z) = -q1.z, d(res.w)/d(q2.w) = q1.w
__device__ __forceinline__ float4 quat_multiply_backward_q2(
    float4 q1, float4 grad_result) {
    float gx = grad_result.x, gy = grad_result.y, gz = grad_result.z, gw = grad_result.w;
    float x1 = q1.x, y1 = q1.y, z1 = q1.z, w1 = q1.w;
    return make_float4(
        w1 * gx + z1 * gy - y1 * gz - x1 * gw,
        -z1 * gx + w1 * gy + x1 * gz - y1 * gw,
        y1 * gx - x1 * gy + w1 * gz - z1 * gw,
        x1 * gx + y1 * gy + z1 * gz + w1 * gw);
}

// Backward through matrix_to_quat (polynomial Shepperd's method).
//
// Each branch's UNNORMALIZED quaternion is LINEAR in R entries (no sqrt, no
// division), so its Jacobian w.r.t. R is a constant {-1, 0, +1} sparse matrix.
// The forward (lines ~92-127) returns 4-branch divisive form, but the
// unnormalized polynomial vector is proportional to the unnormalized divisive
// vector by a positive scalar `s`, so after `normalize_safe` both produce the
// same normalized quaternion. Backprop through `normalize_safe` is identical
// to the divisive form; only the inner Jacobian of `quat_unnorm` w.r.t. R is
// replaced by the polynomial constant-coefficient form.
//
// Polynomial unnormalized quat per branch (matches `_matrix_to_quat` in
// libs/sensors/kernels/pose_calib/bindings.py):
//   k=0 (r00 max):   q = [1 - trace + 2*r00, r10 + r01,           r20 + r02,           r21 - r12]
//   k=1 (r11 max):   q = [r01 + r10,           1 - trace + 2*r11, r21 + r12,           r02 - r20]
//   k=2 (r22 max):   q = [r02 + r20,           r12 + r21,           1 - trace + 2*r22, r10 - r01]
//   k=3 (trace max): q = [r21 - r12,           r02 - r20,           r10 - r01,           1 + trace]
//
// Branch selection mirrors the forward kernel's strict-inequality chain so the
// backward differentiates the same branch the forward took.
__device__ __forceinline__ void matrix_to_quat_backward(
    float3 R0, float3 R1, float3 R2,
    float4 grad_quat_normalized,
    float3& grad_R0, float3& grad_R1, float3& grad_R2) {

    float r00 = R0.x, r01 = R0.y, r02 = R0.z;
    float r10 = R1.x, r11 = R1.y, r12 = R1.z;
    float r20 = R2.x, r21 = R2.y, r22 = R2.z;
    float trace = r00 + r11 + r22;

    // Branch selection: matches forward kernel's strict-inequality chain.
    // 0 = r00 dominant, 1 = r11 dominant, 2 = r22 dominant, 3 = trace dominant.
    int choice;
    if (trace > r00 && trace > r11 && trace > r22)
        choice = 3;
    else if (r00 > r11 && r00 > r22)
        choice = 0;
    else if (r11 > r22)
        choice = 1;
    else
        choice = 2;

    // Polynomial unnormalized quaternion (same direction as forward's divisive
    // form, scaled by positive `s` -- normalize_safe absorbs the difference).
    float4 quat_unnorm;
    switch (choice) {
    case 0: quat_unnorm = make_float4(1.0f - trace + 2.0f * r00, r10 + r01, r20 + r02, r21 - r12); break;
    case 1: quat_unnorm = make_float4(r01 + r10, 1.0f - trace + 2.0f * r11, r21 + r12, r02 - r20); break;
    case 2: quat_unnorm = make_float4(r02 + r20, r12 + r21, 1.0f - trace + 2.0f * r22, r10 - r01); break;
    default: quat_unnorm = make_float4(r21 - r12, r02 - r20, r10 - r01, 1.0f + trace); break;
    }

    // Backward through normalize_safe yields grad on unnormalized quaternion.
    float4 g = quat_normalize_safe_backward(quat_unnorm, grad_quat_normalized);

    // Polynomial Jacobian: each grad_R entry is +/- a single g component
    // (constant coefficients in {-1, 0, +1}; no sqrt, no division).
    // Coefficients derived from d(q_unnorm[i])/d(R[j,k]) for the chosen branch.
    switch (choice) {
    case 0: // r00 dominant: q.x = 1 - trace + 2*r00
        grad_R0 = make_float3(g.x, g.y, g.z);
        grad_R1 = make_float3(g.y, -g.x, -g.w);
        grad_R2 = make_float3(g.z, g.w, -g.x);
        break;
    case 1: // r11 dominant: q.y = 1 - trace + 2*r11
        grad_R0 = make_float3(-g.y, g.x, g.w);
        grad_R1 = make_float3(g.x, g.y, g.z);
        grad_R2 = make_float3(-g.w, g.z, -g.y);
        break;
    case 2: // r22 dominant: q.z = 1 - trace + 2*r22
        grad_R0 = make_float3(-g.z, -g.w, g.x);
        grad_R1 = make_float3(g.w, -g.z, g.y);
        grad_R2 = make_float3(g.x, g.y, g.z);
        break;
    default: // trace dominant: q.w = 1 + trace
        grad_R0 = make_float3(g.w, -g.z, g.y);
        grad_R1 = make_float3(g.z, g.w, -g.x);
        grad_R2 = make_float3(-g.y, g.x, g.w);
        break;
    }
}

// Backward through rotation6d_to_matrix (Gram-Schmidt).
// Forward:
//   a1 = [1+col1.x, col1.y, col1.z]
//   a2 = [col2.x, 1+col2.y, col2.z]
//   b1 = normalize(a1)
//   a2_orth = a2 - dot(a2, b1) * b1
//   b2 = normalize(a2_orth)
//   b3 = cross(b1, b2)
//   R = [b1; b2; b3]  (rows)
__device__ __forceinline__ void rotation6d_to_matrix_backward(
    float3 col1_offset, float3 col2_offset,
    float3 gR0, float3 gR1, float3 gR2,
    float3& grad_col1, float3& grad_col2) {

    // Re-run forward
    float3 a1 = make_float3(1.0f + col1_offset.x, col1_offset.y, col1_offset.z);
    float3 a2 = make_float3(col2_offset.x, 1.0f + col2_offset.y, col2_offset.z);

    float len_sq_a1  = a1.x * a1.x + a1.y * a1.y + a1.z * a1.z;
    float inv_len_a1 = len_sq_a1 > QUAT_EPSILON ? rsqrtf(len_sq_a1) : 0.0f;
    float3 b1        = make_float3(a1.x * inv_len_a1, a1.y * inv_len_a1, a1.z * inv_len_a1);

    float a2_dot_b1 = a2.x * b1.x + a2.y * b1.y + a2.z * b1.z;
    float3 a2_orth  = make_float3(
        a2.x - a2_dot_b1 * b1.x,
        a2.y - a2_dot_b1 * b1.y,
        a2.z - a2_dot_b1 * b1.z);
    float len_sq_a2o  = a2_orth.x * a2_orth.x + a2_orth.y * a2_orth.y + a2_orth.z * a2_orth.z;
    float inv_len_a2o = len_sq_a2o > QUAT_EPSILON ? rsqrtf(len_sq_a2o) : 0.0f;
    float3 b2         = make_float3(a2_orth.x * inv_len_a2o, a2_orth.y * inv_len_a2o, a2_orth.z * inv_len_a2o);

    // gR0 = grad w.r.t. row0 = b1
    // gR1 = grad w.r.t. row1 = b2
    // gR2 = grad w.r.t. row2 = b3 = cross(b1, b2)

    // Step 1: backward through b3 = cross(b1, b2)
    // grad_b1[k] = sum_i gR2[i] * d(cross(b1,b2)[i])/d(b1[k])
    //   d/d(b1.x): (0, -b2.z, b2.y)  =>  gR2.y*(-b2.z) + gR2.z*b2.y
    //   d/d(b1.y): (b2.z, 0, -b2.x)  =>  gR2.x*b2.z + gR2.z*(-b2.x)
    //   d/d(b1.z): (-b2.y, b2.x, 0)  =>  gR2.x*(-b2.y) + gR2.y*b2.x
    float3 grad_b1 = make_float3(
        gR0.x + (-b2.z * gR2.y + b2.y * gR2.z),
        gR0.y + (b2.z * gR2.x - b2.x * gR2.z),
        gR0.z + (-b2.y * gR2.x + b2.x * gR2.y));

    // grad_b2 from b3 = cross(b1, b2): d(cross)/d(b2) applied to gR2
    float3 grad_b2 = make_float3(
        gR1.x + (gR2.y * b1.z - gR2.z * b1.y),
        gR1.y + (gR2.z * b1.x - gR2.x * b1.z),
        gR1.z + (gR2.x * b1.y - gR2.y * b1.x));

    // Step 2: backward through b2 = normalize(a2_orth)
    // d(normalize(v))/dv = (I - v_hat * v_hat^T) / ||v||
    float dp_b2         = b2.x * grad_b2.x + b2.y * grad_b2.y + b2.z * grad_b2.z;
    float3 grad_a2_orth = make_float3(
        (grad_b2.x - b2.x * dp_b2) * inv_len_a2o,
        (grad_b2.y - b2.y * dp_b2) * inv_len_a2o,
        (grad_b2.z - b2.z * dp_b2) * inv_len_a2o);

    // Step 3: backward through a2_orth = a2 - dot(a2, b1) * b1
    // d(a2_orth)/d(a2) = I - b1*b1^T
    // d(a2_orth)/d(b1): grad_b1[k] += -a2[k]*dot(go,b1) - (a2.b1)*go[k]
    float dot_go_b1 = grad_a2_orth.x * b1.x + grad_a2_orth.y * b1.y + grad_a2_orth.z * b1.z;
    float3 grad_a2  = make_float3(
        grad_a2_orth.x - b1.x * dot_go_b1,
        grad_a2_orth.y - b1.y * dot_go_b1,
        grad_a2_orth.z - b1.z * dot_go_b1);

    // grad_b1 from a2_orth:
    // grad_b1[k] += sum_i grad_a2_orth[i] * (-a2[k]*b1[i] - (a2.b1)*delta_ik)
    //             = -a2[k] * dot(grad_a2_orth, b1) - (a2.b1) * grad_a2_orth[k]
    grad_b1.x += -a2.x * dot_go_b1 - a2_dot_b1 * grad_a2_orth.x;
    grad_b1.y += -a2.y * dot_go_b1 - a2_dot_b1 * grad_a2_orth.y;
    grad_b1.z += -a2.z * dot_go_b1 - a2_dot_b1 * grad_a2_orth.z;

    // Step 4: backward through b1 = normalize(a1)
    float dp_b1    = b1.x * grad_b1.x + b1.y * grad_b1.y + b1.z * grad_b1.z;
    float3 grad_a1 = make_float3(
        (grad_b1.x - b1.x * dp_b1) * inv_len_a1,
        (grad_b1.y - b1.y * dp_b1) * inv_len_a1,
        (grad_b1.z - b1.z * dp_b1) * inv_len_a1);

    // Step 5: a1 = [1+col1.x, col1.y, col1.z], a2 = [col2.x, 1+col2.y, col2.z]
    // grad_col1 = grad_a1 (identity Jacobian offset by constant)
    // grad_col2 = grad_a2
    grad_col1 = grad_a1;
    grad_col2 = grad_a2;
}

// Backward through quat_slerp using pre-computed shared state.
// The hemisphere flip, dp clamp, branch decision (use_lerp), and theta/sin_theta
// are computed ONCE by the caller (backward_stage_slerp) and reused across the
// two t0/t1 calls that share the same q1/q2 pair.
__device__ __forceinline__ void quat_slerp_backward_shared(
    float4 q1_in, float4 q2_adj, bool flipped, float t,
    float dp, float theta, float sin_theta, bool use_lerp,
    float4 grad_out,
    float4& grad_q1, float4& grad_q2) {

    if (use_lerp) {
        // LERP path (forward no longer normalizes — store_se3_as_matrix does).
        // result = (1-t)*q1 + t*q2_adj, so d(result)/d(q1) = (1-t)*I,
        // d(result)/d(q2_adj) = t*I. No normalize-chain step needed.
        grad_q1 = make_float4(
            (1.0f - t) * grad_out.x,
            (1.0f - t) * grad_out.y,
            (1.0f - t) * grad_out.z,
            (1.0f - t) * grad_out.w);
        float4 grad_q2_adj = make_float4(
            t * grad_out.x, t * grad_out.y, t * grad_out.z, t * grad_out.w);

        if (flipped) {
            grad_q2 = make_float4(-grad_q2_adj.x, -grad_q2_adj.y, -grad_q2_adj.z, -grad_q2_adj.w);
        } else {
            grad_q2 = grad_q2_adj;
        }
    } else {
        // SLERP path: result = w1*q1 + w2*q2_adj (no normalize — store_se3_as_matrix handles it)
        float w1 = sinf((1.0f - t) * theta) / sin_theta;
        float w2 = sinf(t * theta) / sin_theta;

        // d(out)/d(w1) = q1,  d(out)/d(w2) = q2_adj
        float grad_w1 = grad_out.x * q1_in.x + grad_out.y * q1_in.y +
                        grad_out.z * q1_in.z + grad_out.w * q1_in.w;
        float grad_w2 = grad_out.x * q2_adj.x + grad_out.y * q2_adj.y +
                        grad_out.z * q2_adj.z + grad_out.w * q2_adj.w;

        // dw1/dtheta and dw2/dtheta via simplified quotient rule.
        // Since w1 = sin((1-t)*theta)/sin(theta) and w2 = sin(t*theta)/sin(theta),
        // we reuse w1*sin_theta and w2*sin_theta instead of recomputing sinf:
        //   dw1/dtheta = ((1-t)*cos((1-t)*theta) - w1*cos_theta) / sin_theta
        //   dw2/dtheta = (t*cos(t*theta) - w2*cos_theta) / sin_theta
        float cos_theta     = dp;
        float inv_sin_theta = 1.0f / sin_theta;
        float cos_w1_arg    = cosf((1.0f - t) * theta);
        float cos_w2_arg    = cosf(t * theta);

        float dw1_dtheta = ((1.0f - t) * cos_w1_arg - w1 * cos_theta) * inv_sin_theta;
        float dw2_dtheta = (t * cos_w2_arg - w2 * cos_theta) * inv_sin_theta;

        // d(theta)/d(dp) = -1/sin(theta), chain with grad_w1/grad_w2
        float grad_dp = (grad_w1 * dw1_dtheta + grad_w2 * dw2_dtheta) * (-inv_sin_theta);

        grad_q1 = make_float4(
            w1 * grad_out.x + grad_dp * q2_adj.x,
            w1 * grad_out.y + grad_dp * q2_adj.y,
            w1 * grad_out.z + grad_dp * q2_adj.z,
            w1 * grad_out.w + grad_dp * q2_adj.w);
        float4 grad_q2_adj = make_float4(
            w2 * grad_out.x + grad_dp * q1_in.x,
            w2 * grad_out.y + grad_dp * q1_in.y,
            w2 * grad_out.z + grad_dp * q1_in.z,
            w2 * grad_out.w + grad_dp * q1_in.w);

        if (flipped) {
            grad_q2 = make_float4(-grad_q2_adj.x, -grad_q2_adj.y, -grad_q2_adj.z, -grad_q2_adj.w);
        } else {
            grad_q2 = grad_q2_adj;
        }
    }
}

// Load R^T * grad from a base pose directly from global memory.
// Avoids the quat->matrix recomputation in quat_rotate_vector_backward_v.
__device__ __forceinline__ float3 rotate_transpose_from_global(
    const float* __restrict__ T_startend_allviews,
    int fidx, int startend_idx,
    float3 grad) {
    const float* base = T_startend_allviews + (fidx * 2 + startend_idx) * 16;
    float3 R0         = make_float3(base[0], base[1], base[2]);
    float3 R1         = make_float3(base[4], base[5], base[6]);
    float3 R2         = make_float3(base[8], base[9], base[10]);
    return make_float3(
        R0.x * grad.x + R1.x * grad.y + R2.x * grad.z,
        R0.y * grad.x + R1.y * grad.y + R2.y * grad.z,
        R0.z * grad.x + R1.z * grad.y + R2.z * grad.z);
}

// ============================================================================
// Backward stage helpers
// ============================================================================
// Each stage is a __device__ __forceinline__ helper that scopes its temporaries
// to improve readability and encourage the compiler to reuse registers.

// Backward through store_se3_as_matrix for one startend slot.
// Loads grad_T_out, runs normalize_safe -> quat_to_matrix backward chain.
__device__ __forceinline__ void backward_stage_store_and_normalize(
    const float* __restrict__ grad_T_out, int tid, int startend_idx,
    float4 q_out,
    float4& grad_quat, float3& grad_trans) {
    float3 gR0, gR1, gR2;
    load_grad_T_out(grad_T_out, tid, startend_idx, gR0, gR1, gR2, grad_trans);
    float4 q_norm      = quat_normalize_safe(q_out);
    float4 grad_q_norm = quat_to_matrix_backward(q_norm, gR0, gR1, gR2);
    grad_quat          = quat_normalize_safe_backward(q_out, grad_q_norm);
}

// Backward through slerp interpolation for both start/end, accumulating
// gradients onto the calibrated quaternions and translations.
//
// The two slerp backward calls share q_calib_start/q_calib_end — only t differs.
// We compute hemisphere flip, dp clamp, branch decision, and theta/sin_theta
// ONCE here and reuse for both calls (saves 1 dot product, 1 acosf, 1 sinf,
// and the branch/clamp logic per thread).
__device__ __forceinline__ void backward_stage_slerp(
    float4 q_calib_start, float4 q_calib_end,
    float t0, float t1,
    float4 grad_quat_start, float4 grad_quat_end,
    float3 grad_trans_start, float3 grad_trans_end,
    float4& grad_q_calib_start, float4& grad_q_calib_end,
    float3& grad_t_calib_start, float3& grad_t_calib_end) {

    // Shared state: computed once, reused across both t0 and t1 backward calls.
    float dp_raw = q_calib_start.x * q_calib_end.x + q_calib_start.y * q_calib_end.y +
                   q_calib_start.z * q_calib_end.z + q_calib_start.w * q_calib_end.w;
    bool flipped  = dp_raw < 0.0f;
    float4 q2_adj = flipped
                        ? make_float4(-q_calib_end.x, -q_calib_end.y, -q_calib_end.z, -q_calib_end.w)
                        : q_calib_end;
    float dp      = fminf(fmaxf(flipped ? -dp_raw : dp_raw, -1.0f), 1.0f);
    bool use_lerp = (dp > 0.9995f);

    // theta/sin_theta only needed in SLERP branch; leave zero-init in NLERP.
    float theta = 0.0f, sin_theta = 0.0f;
    if (!use_lerp) {
        theta     = acosf(dp);
        sin_theta = sinf(theta);
    }

    {
        float4 gq1, gq2;
        quat_slerp_backward_shared(
            q_calib_start, q2_adj, flipped, t0,
            dp, theta, sin_theta, use_lerp,
            grad_quat_start, gq1, gq2);
        grad_q_calib_start = gq1;
        grad_q_calib_end   = gq2;
    }
    {
        float4 gq1, gq2;
        quat_slerp_backward_shared(
            q_calib_start, q2_adj, flipped, t1,
            dp, theta, sin_theta, use_lerp,
            grad_quat_end, gq1, gq2);
        grad_q_calib_start.x += gq1.x;
        grad_q_calib_start.y += gq1.y;
        grad_q_calib_start.z += gq1.z;
        grad_q_calib_start.w += gq1.w;
        grad_q_calib_end.x += gq2.x;
        grad_q_calib_end.y += gq2.y;
        grad_q_calib_end.z += gq2.z;
        grad_q_calib_end.w += gq2.w;
    }

    grad_t_calib_start = make_float3(
        (1.0f - t0) * grad_trans_start.x + (1.0f - t1) * grad_trans_end.x,
        (1.0f - t0) * grad_trans_start.y + (1.0f - t1) * grad_trans_end.y,
        (1.0f - t0) * grad_trans_start.z + (1.0f - t1) * grad_trans_end.z);
    grad_t_calib_end = make_float3(
        t0 * grad_trans_start.x + t1 * grad_trans_end.x,
        t0 * grad_trans_start.y + t1 * grad_trans_end.y,
        t0 * grad_trans_start.z + t1 * grad_trans_end.z);
}

// Backward through se3_multiply for the quaternion component.
// Reloads base quats from global, computes grad w.r.t. delta quat.
__device__ __forceinline__ float4 backward_stage_se3_quat(
    const float* __restrict__ T_startend_allviews, int fidx,
    float4 grad_q_calib_start, float4 grad_q_calib_end) {

    const float* base_s = T_startend_allviews + (fidx * 2 + 0) * 16;
    float4 q_base_start = matrix_to_quat(
        make_float3(base_s[0], base_s[1], base_s[2]),
        make_float3(base_s[4], base_s[5], base_s[6]),
        make_float3(base_s[8], base_s[9], base_s[10]));

    const float* base_e = T_startend_allviews + (fidx * 2 + 1) * 16;
    float4 q_base_end   = matrix_to_quat(
        make_float3(base_e[0], base_e[1], base_e[2]),
        make_float3(base_e[4], base_e[5], base_e[6]),
        make_float3(base_e[8], base_e[9], base_e[10]));

    float4 gd_s = quat_multiply_backward_q2(q_base_start, grad_q_calib_start);
    float4 gd_e = quat_multiply_backward_q2(q_base_end, grad_q_calib_end);
    return make_float4(
        gd_s.x + gd_e.x, gd_s.y + gd_e.y,
        gd_s.z + gd_e.z, gd_s.w + gd_e.w);
}

// Backward through se3_multiply for the translation component.
// Computes R_base^T * grad_trans for both start/end, sums results.
__device__ __forceinline__ float3 backward_stage_se3_trans(
    const float* __restrict__ T_startend_allviews, int fidx,
    float3 grad_t_calib_start, float3 grad_t_calib_end) {

    float3 gds = rotate_transpose_from_global(T_startend_allviews, fidx, 0, grad_t_calib_start);
    float3 gde = rotate_transpose_from_global(T_startend_allviews, fidx, 1, grad_t_calib_end);
    return make_float3(gds.x + gde.x, gds.y + gde.y, gds.z + gde.z);
}

// Backward through matrix_to_quat and rotation6d_to_matrix.
// Recomputes dR from col offsets, chains backward through both.
__device__ __forceinline__ void backward_stage_rot6d(
    float3 col1_offset, float3 col2_offset,
    float4 grad_dq,
    float3& grad_col1, float3& grad_col2) {

    float3 dR0, dR1, dR2;
    rotation6d_to_matrix(col1_offset, col2_offset, dR0, dR1, dR2);

    float3 grad_dR0, grad_dR1, grad_dR2;
    matrix_to_quat_backward(dR0, dR1, dR2, grad_dq, grad_dR0, grad_dR1, grad_dR2);

    rotation6d_to_matrix_backward(col1_offset, col2_offset, grad_dR0, grad_dR1, grad_dR2,
                                  grad_col1, grad_col2);
}

// ============================================================================
// Backward kernel
// ============================================================================

// Only ENABLE_CALIB=true variants produce gradients (embed_weights is the
// only differentiable input). When ENABLE_CALIB=false, no backward is needed.
//
// Each backward stage is extracted into a helper function so the compiler
// can reuse registers across stage boundaries. The SLERP backward uses a
// simplified quotient rule that reuses w1/w2 instead of recomputing sinf.
template <bool HAS_SUBSAMPLING>
__global__ void pose_calib_backward_kernel(
    int batch_size,
    const float* __restrict__ T_startend_allviews, // [V, 2, 4, 4]
    const float* __restrict__ embed_weights,       // [V, 9]
    const int32_t* __restrict__ frame_idx,         // [N]
    const float* __restrict__ rect_points_lb,      // [N, 2, 2]
    const float* __restrict__ resolution,          // [N, 2]
    ShutterType shutter_type,
    const float* __restrict__ grad_T_out,   // [N, 2, 4, 4]
    float* __restrict__ grad_embed_weights, // [V, 9]
    const float4* __restrict__ scratch) {   // [N, 5] saved by forward

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    int fidx = frame_idx[tid];

    // ====================================================================
    // Stages 1-2: Read saved intermediates from global memory (written by
    // forward kernel). Replaces full forward-chain recompute of q_calib and
    // q_out (via se3_multiply + slerp). q_base is NOT saved — reloading +
    // recomputing in Stage 5 costs less than the registers needed to hold
    // 2 extra float4s live through stages 3-4.
    // ====================================================================

    const float4* row    = scratch + tid * 5;
    float4 q_calib_start = row[0];
    float4 q_calib_end   = row[1];
    float4 q_out_start   = row[2];
    float4 q_out_end     = row[3];
    float t0 = 0.0f, t1 = 0.0f;
    if constexpr (HAS_SUBSAMPLING) {
        float4 ts = row[4];
        t0        = ts.x;
        t1        = ts.y;
    }

    // ====================================================================
    // Stage 3: Backward through store_se3_as_matrix (normalize + quat_to_matrix)
    // ====================================================================

    float3 grad_trans_start, grad_trans_end;
    float4 grad_quat_start, grad_quat_end;
    backward_stage_store_and_normalize(grad_T_out, tid, 0, q_out_start, grad_quat_start, grad_trans_start);
    backward_stage_store_and_normalize(grad_T_out, tid, 1, q_out_end, grad_quat_end, grad_trans_end);

    // ====================================================================
    // Stage 4: Backward through subsampling interpolation
    // ====================================================================

    float4 grad_q_calib_start, grad_q_calib_end;
    float3 grad_t_calib_start, grad_t_calib_end;

    if constexpr (HAS_SUBSAMPLING) {
        backward_stage_slerp(
            q_calib_start, q_calib_end, t0, t1,
            grad_quat_start, grad_quat_end,
            grad_trans_start, grad_trans_end,
            grad_q_calib_start, grad_q_calib_end,
            grad_t_calib_start, grad_t_calib_end);
    } else {
        grad_q_calib_start = grad_quat_start;
        grad_q_calib_end   = grad_quat_end;
        grad_t_calib_start = grad_trans_start;
        grad_t_calib_end   = grad_trans_end;
    }

    // ====================================================================
    // Stage 5: Backward through se3_multiply (reload base quats from global)
    // ====================================================================

    float4 grad_dq = backward_stage_se3_quat(
        T_startend_allviews, fidx, grad_q_calib_start, grad_q_calib_end);
    float3 grad_dx = backward_stage_se3_trans(
        T_startend_allviews, fidx, grad_t_calib_start, grad_t_calib_end);

    // ====================================================================
    // Stage 6: Backward through matrix_to_quat and rotation6d_to_matrix
    // Reload col offsets from embed_weights (deferred from Stage 1 to free
    // 6 registers across stages 2-5).
    // ====================================================================

    float3 grad_col1, grad_col2;
    {
        const float* ew    = embed_weights + fidx * 9;
        float3 col1_offset = make_float3(ew[3], ew[4], ew[5]);
        float3 col2_offset = make_float3(ew[6], ew[7], ew[8]);
        backward_stage_rot6d(col1_offset, col2_offset, grad_dq, grad_col1, grad_col2);
    }

    // ====================================================================
    // Scatter-accumulate gradients into grad_embed_weights[fidx]
    // ====================================================================
    float* out = grad_embed_weights + fidx * 9;
    atomicAdd(&out[0], grad_dx.x);
    atomicAdd(&out[1], grad_dx.y);
    atomicAdd(&out[2], grad_dx.z);
    atomicAdd(&out[3], grad_col1.x);
    atomicAdd(&out[4], grad_col1.y);
    atomicAdd(&out[5], grad_col1.z);
    atomicAdd(&out[6], grad_col2.x);
    atomicAdd(&out[7], grad_col2.y);
    atomicAdd(&out[8], grad_col2.z);
}

// ============================================================================
// Host wrappers
// ============================================================================

void pose_calib_forward_cuda(
    int batch_size,
    const torch::Tensor& T_startend_allviews,
    const torch::Tensor& embed_weights,
    const torch::Tensor& frame_idx,
    const torch::Tensor& rect_points_lb,
    const torch::Tensor& resolution,
    const torch::Tensor& timestamps_startend,
    int shutter_type,
    bool enable_calib,
    bool has_subsampling,
    const torch::Tensor& T_out,
    const torch::Tensor& timestamps_out,
    const torch::Tensor& scratch) {

    if (batch_size <= 0)
        return;

    CHECK_INPUT(T_startend_allviews);
    CHECK_INPUT(frame_idx);
    CHECK_INPUT(timestamps_startend);
    CHECK_INPUT(T_out);
    CHECK_INPUT(timestamps_out);

    if (enable_calib) {
        CHECK_INPUT(embed_weights);
    }
    if (has_subsampling) {
        CHECK_INPUT(rect_points_lb);
        CHECK_INPUT(resolution);
    }

    at::cuda::CUDAGuard device_guard(T_startend_allviews.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(BLOCK_THREADS);
    dim3 blocks(div_round_up(batch_size, BLOCK_THREADS));

    // Pointers (nullptr for unused optional tensors)
    auto poses_ptr   = T_startend_allviews.data_ptr<float>();
    auto fidx_ptr    = frame_idx.data_ptr<int32_t>();
    auto ts_ptr      = timestamps_startend.data_ptr<int64_t>();
    auto tout_ptr    = T_out.data_ptr<float>();
    auto tsout_ptr   = timestamps_out.data_ptr<int64_t>();
    auto ew_ptr      = enable_calib ? embed_weights.data_ptr<float>() : nullptr;
    auto rp_ptr      = has_subsampling ? rect_points_lb.data_ptr<float>() : nullptr;
    auto res_ptr     = has_subsampling ? resolution.data_ptr<float>() : nullptr;
    auto scratch_ptr = (enable_calib && scratch.numel() > 0)
                           ? reinterpret_cast<float4*>(scratch.data_ptr<float>())
                           : nullptr;

    auto st = static_cast<ShutterType>(shutter_type);

    // Dispatch to the right template specialization
    if (enable_calib && has_subsampling) {
        pose_calib_forward_kernel<true, true><<<blocks, threads, 0, stream>>>(
            batch_size, poses_ptr, ew_ptr, fidx_ptr, rp_ptr, res_ptr,
            ts_ptr, st, tout_ptr, tsout_ptr, scratch_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (enable_calib && !has_subsampling) {
        pose_calib_forward_kernel<true, false><<<blocks, threads, 0, stream>>>(
            batch_size, poses_ptr, ew_ptr, fidx_ptr, rp_ptr, res_ptr,
            ts_ptr, st, tout_ptr, tsout_ptr, scratch_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (!enable_calib && has_subsampling) {
        pose_calib_forward_kernel<false, true><<<blocks, threads, 0, stream>>>(
            batch_size, poses_ptr, ew_ptr, fidx_ptr, rp_ptr, res_ptr,
            ts_ptr, st, tout_ptr, tsout_ptr, nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        pose_calib_forward_kernel<false, false><<<blocks, threads, 0, stream>>>(
            batch_size, poses_ptr, ew_ptr, fidx_ptr, rp_ptr, res_ptr,
            ts_ptr, st, tout_ptr, tsout_ptr, nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void pose_calib_backward_cuda(
    int batch_size,
    const torch::Tensor& T_startend_allviews,
    const torch::Tensor& embed_weights,
    const torch::Tensor& frame_idx,
    const torch::Tensor& rect_points_lb,
    const torch::Tensor& resolution,
    int shutter_type,
    bool has_subsampling,
    const torch::Tensor& grad_T_out,
    const torch::Tensor& grad_embed_weights,
    const torch::Tensor& scratch) {

    if (batch_size <= 0)
        return;

    CHECK_INPUT(T_startend_allviews);
    CHECK_INPUT(embed_weights);
    CHECK_INPUT(frame_idx);
    CHECK_INPUT(grad_T_out);
    CHECK_INPUT(grad_embed_weights);
    // Precondition: callers only invoke backward when forward was run with
    // enable_calib=true, which always allocates a non-empty scratch tensor.
    TORCH_CHECK(scratch.numel() > 0, "scratch must be populated from forward (enable_calib=true)");
    CHECK_INPUT(scratch);

    if (has_subsampling) {
        CHECK_INPUT(rect_points_lb);
        CHECK_INPUT(resolution);
    }

    at::cuda::CUDAGuard device_guard(T_startend_allviews.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(BLOCK_THREADS_BWD);
    dim3 blocks(div_round_up(batch_size, BLOCK_THREADS_BWD));

    auto poses_ptr    = T_startend_allviews.data_ptr<float>();
    auto ew_ptr       = embed_weights.data_ptr<float>();
    auto fidx_ptr     = frame_idx.data_ptr<int32_t>();
    auto grad_out_ptr = grad_T_out.data_ptr<float>();
    auto grad_ew_ptr  = grad_embed_weights.data_ptr<float>();
    auto rp_ptr       = has_subsampling ? rect_points_lb.data_ptr<float>() : nullptr;
    auto res_ptr      = has_subsampling ? resolution.data_ptr<float>() : nullptr;
    auto scratch_ptr  = reinterpret_cast<const float4*>(scratch.data_ptr<float>());

    auto st = static_cast<ShutterType>(shutter_type);

    if (has_subsampling) {
        pose_calib_backward_kernel<true><<<blocks, threads, 0, stream>>>(
            batch_size, poses_ptr, ew_ptr, fidx_ptr, rp_ptr, res_ptr,
            st, grad_out_ptr, grad_ew_ptr, scratch_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        pose_calib_backward_kernel<false><<<blocks, threads, 0, stream>>>(
            batch_size, poses_ptr, ew_ptr, fidx_ptr, rp_ptr, res_ptr,
            st, grad_out_ptr, grad_ew_ptr, scratch_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
