// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "losses_cuda.h"

#include <array>
#include <chrono>
#include <functional>
#include <optional>

#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>

// DISPATCH 0: Road Gaussians Loss
// ============================================================================
// Merged from road_gaussians_cuda.cu — constrains height and rotation variance
// of gaussians within depth bins along the camera z-axis.

// TODO: Keep state between forward and backward pass
// TODO: use wxyz quaternions convention

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_std_dev(scalar_t const& sum, scalar_t const& sum2, scalar_t const& count) {
    if (count <= 1.0f)
        return 0.0f;
    auto const mean{sum / count};
    auto const variance{(sum2 / count) - (mean * mean)};
    auto const unbiased_variance{variance * count / (count - 1.0f)};
    return sqrtf(fmaxf(unbiased_variance, 0.0f));
}

enum {
    AccumY     = 0,
    AccumY2    = 1,
    AccumRoll  = 2,
    AccumRoll2 = 3,
    AccumPitch,
    AccumPitch2,
    Count,
    FieldsCount
};

template <typename scalar_t, size_t threads>
__global__ void road_gaussians_forward_reductor_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> stats,
    scalar_t* __restrict__ total_loss,
    int const n_biases,
    scalar_t const rotation_lambda) {

    __shared__ scalar_t total_loss_shared[threads];

    scalar_t loss_value = 0.F;

    unsigned int const bias_idx{blockIdx.x * blockDim.x + threadIdx.x};
    if (bias_idx < n_biases) {

        auto const count{stats[bias_idx][Count]};

        if (count > 1) {

            // Y variance
            auto const std_pos_y{compute_std_dev(stats[bias_idx][AccumY], stats[bias_idx][AccumY2], count)};

            auto const std_roll{compute_std_dev(stats[bias_idx][AccumRoll], stats[bias_idx][AccumRoll2], count)};
            auto const std_pitch{compute_std_dev(stats[bias_idx][AccumPitch], stats[bias_idx][AccumPitch2], count)};

            auto const bias_loss{(std_pos_y + rotation_lambda * (std_roll + std_pitch))};

            loss_value = bias_loss;
        }
    }

    total_loss_shared[threadIdx.x] = loss_value;

    __syncthreads();
    for (auto offset = threads >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            total_loss_shared[threadIdx.x] += total_loss_shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Apply division by n_biases at the final step
        auto const normalized_loss_shared{total_loss_shared[0] / n_biases};
        atomicAdd(total_loss, normalized_loss_shared);
    }
}

// ============================================================================
// Cuda implementation of geometric transformations
// ============================================================================

template <typename scalar_t>
__device__ __forceinline__ void quat_normalize(std::array<scalar_t, 4> const& q, std::array<scalar_t, 4>& q_normalized) {
    scalar_t const sq_norm{q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]};
    scalar_t const inv_norm{1.0f / sqrtf(sq_norm)};
    q_normalized[0] = q[0] * inv_norm;
    q_normalized[1] = q[1] * inv_norm;
    q_normalized[2] = q[2] * inv_norm;
    q_normalized[3] = q[3] * inv_norm;
}

template <typename scalar_t>
__device__ __forceinline__ void quat_normalize_backward(std::array<scalar_t, 4> const& q, std::array<scalar_t, 4> const& q_normalized, std::array<scalar_t, 4> const& grad_q_normalized, std::array<scalar_t, 4>& grad_q) {

    scalar_t const sq_norm{q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]};
    scalar_t const inv_norm{1.0f / sqrtf(sq_norm)};

#pragma unroll
    for (int i = 0; i < 4; i++) {
        grad_q[i] = 0.0f;

#pragma unroll
        for (int j = 0; j < 4; j++) {
            auto neg_contrib{q_normalized[i] * q_normalized[j]};

            if (i == j) {
                neg_contrib -= 1;
            }

            grad_q[i] -= grad_q_normalized[j] * neg_contrib * inv_norm;
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void quat_to_so3_matrix(std::array<scalar_t, 4> const& q, std::array<std::array<scalar_t, 3>, 3>& R) {

    std::array<scalar_t, 4> q_normalized;
    quat_normalize(q, q_normalized);

    // Convert quaternion to rotation matrix
    R[0][0] = 1.0f - 2.0f * (q_normalized[1] * q_normalized[1] + q_normalized[2] * q_normalized[2]);
    R[0][1] = 2.0f * (q_normalized[0] * q_normalized[1] - q_normalized[2] * q_normalized[3]);
    R[0][2] = 2.0f * (q_normalized[0] * q_normalized[2] + q_normalized[1] * q_normalized[3]);

    R[1][0] = 2.0f * (q_normalized[0] * q_normalized[1] + q_normalized[2] * q_normalized[3]);
    R[1][1] = 1.0f - 2.0f * (q_normalized[0] * q_normalized[0] + q_normalized[2] * q_normalized[2]);
    R[1][2] = 2.0f * (q_normalized[1] * q_normalized[2] - q_normalized[0] * q_normalized[3]);

    R[2][0] = 2.0f * (q_normalized[0] * q_normalized[2] - q_normalized[1] * q_normalized[3]);
    R[2][1] = 2.0f * (q_normalized[1] * q_normalized[2] + q_normalized[0] * q_normalized[3]);
    R[2][2] = 1.0f - 2.0f * (q_normalized[0] * q_normalized[0] + q_normalized[1] * q_normalized[1]);
}

template <typename scalar_t>
__device__ __forceinline__ void quat_to_so3_matrix_backward(std::array<scalar_t, 4> const& q, std::array<std::array<scalar_t, 3>, 3> const& R,
                                                            std::array<std::array<scalar_t, 3>, 3> const& grad_R, std::array<scalar_t, 4>& grad_q) {

    std::array<scalar_t, 4> grad_q_normalized;
    std::array<scalar_t, 4> q_normalized;

    quat_normalize(q, q_normalized);

    // Corrected gradient formulas for [x, y, z, w] quaternion convention
    // q_normalized[0] = x, q_normalized[1] = y, q_normalized[2] = z, q_normalized[3] = w
    grad_q_normalized[0] = 2 * q_normalized[1] * grad_R[0][1] + 2 * q_normalized[2] * grad_R[0][2] + 2 * q_normalized[1] * grad_R[1][0] - 4 * q_normalized[0] * grad_R[1][1] - 2 * q_normalized[3] * grad_R[1][2] + 2 * q_normalized[2] * grad_R[2][0] + 2 * q_normalized[3] * grad_R[2][1] - 4 * q_normalized[0] * grad_R[2][2];

    grad_q_normalized[1] = -4 * q_normalized[1] * grad_R[0][0] + 2 * q_normalized[0] * grad_R[0][1] + 2 * q_normalized[3] * grad_R[0][2] + 2 * q_normalized[0] * grad_R[1][0] + 2 * q_normalized[2] * grad_R[1][2] - 2 * q_normalized[3] * grad_R[2][0] + 2 * q_normalized[2] * grad_R[2][1] - 4 * q_normalized[1] * grad_R[2][2];

    grad_q_normalized[2] = -4 * q_normalized[2] * grad_R[0][0] - 2 * q_normalized[3] * grad_R[0][1] + 2 * q_normalized[0] * grad_R[0][2] + 2 * q_normalized[3] * grad_R[1][0] - 4 * q_normalized[2] * grad_R[1][1] + 2 * q_normalized[1] * grad_R[1][2] + 2 * q_normalized[0] * grad_R[2][0] + 2 * q_normalized[1] * grad_R[2][1];

    grad_q_normalized[3] = -2 * q_normalized[2] * grad_R[0][1] + 2 * q_normalized[1] * grad_R[0][2] + 2 * q_normalized[2] * grad_R[1][0] - 2 * q_normalized[0] * grad_R[1][2] - 2 * q_normalized[1] * grad_R[2][0] + 2 * q_normalized[0] * grad_R[2][1];

    quat_normalize_backward(q, q_normalized, grad_q_normalized, grad_q);
}

template <typename scalar_t>
__device__ __forceinline__ void tquat_to_se3_matrix(std::array<scalar_t, 7> const& tq, std::array<std::array<scalar_t, 4>, 4>& T) {
    // Extract translation (first 3 elements) and quaternion (last 4 elements)
    std::array<scalar_t, 3> const t{tq[0], tq[1], tq[2]};
    std::array<scalar_t, 4> const q{tq[3], tq[4], tq[5], tq[6]};

    // Get rotation matrix from quaternion
    std::array<std::array<scalar_t, 3>, 3> R;
    quat_to_so3_matrix(q, R);

    // Build SE3 matrix
    for (int i = 0; i < 3; i++) {
        for (int j{}; j < 3; j++) {
            T[i][j] = R[i][j];
        }
        T[i][3] = t[i];
    }
    T[3][0] = T[3][1] = T[3][2] = {};
    T[3][3]                     = 1.0f;
}

template <typename scalar_t>
__device__ __forceinline__ void se3_matrix_inverse(std::array<std::array<scalar_t, 4>, 4> const& T, std::array<std::array<scalar_t, 4>, 4>& T_inv) {
    // For SE3 matrix T = [R t; 0 1], inverse is [R^T -R^T*t; 0 1]

    // R^T (transpose of rotation part)
    for (int i = 0; i < 3; i++) {
        for (int j{}; j < 3; j++) {
            T_inv[i][j] = T[j][i];
        }
    }

    // -R^T * t
    for (int i = 0; i < 3; i++) {
        T_inv[i][3] = -(T_inv[i][0] * T[0][3] + T_inv[i][1] * T[1][3] + T_inv[i][2] * T[2][3]);
    }

    // Bottom row
    T_inv[3][0] = T_inv[3][1] = T_inv[3][2] = {};
    T_inv[3][3]                             = 1.0f;
}

template <typename scalar_t>
__device__ __forceinline__ void se3_transform_point(std::array<std::array<scalar_t, 4>, 4> const& T, std::array<scalar_t, 3> const& pos, std::array<scalar_t, 3>& transformed_pos) {
    transformed_pos[0] = T[0][0] * pos[0] + T[0][1] * pos[1] + T[0][2] * pos[2] + T[0][3];
    transformed_pos[1] = T[1][0] * pos[0] + T[1][1] * pos[1] + T[1][2] * pos[2] + T[1][3];
    transformed_pos[2] = T[2][0] * pos[0] + T[2][1] * pos[1] + T[2][2] * pos[2] + T[2][3];
}

template <typename scalar_t>
__device__ __forceinline__ void se3_transform_point_backward(std::array<std::array<scalar_t, 4>, 4> const& T, std::array<scalar_t, 3> const& pos, std::array<scalar_t, 3> const& transformed_pos, std::array<scalar_t, 3> const& grad_transformed_pos, std::array<scalar_t, 3>& grad_pos) {
    grad_pos[0] = T[0][0] * grad_transformed_pos[0] + T[1][0] * grad_transformed_pos[1] + T[2][0] * grad_transformed_pos[2];
    grad_pos[1] = T[0][1] * grad_transformed_pos[0] + T[1][1] * grad_transformed_pos[1] + T[2][1] * grad_transformed_pos[2];
    grad_pos[2] = T[0][2] * grad_transformed_pos[0] + T[1][2] * grad_transformed_pos[1] + T[2][2] * grad_transformed_pos[2];
}

// D can be 3 or 4
template <bool transpose_A, typename scalar_t, size_t DimA, size_t DimB>
__device__ __forceinline__ void so3_matrix_multiply(std::array<std::array<scalar_t, DimA>, DimA> const& A, std::array<std::array<scalar_t, DimB>, DimB> const& B, std::array<std::array<scalar_t, 3>, 3>& C) {
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            if constexpr (transpose_A) {
                C[i][j] = A[0][i] * B[0][j] + A[1][i] * B[1][j] + A[2][i] * B[2][j];
            } else {
                C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
            }
        }
    }
}

template <bool transpose_A, typename scalar_t, size_t DimA, size_t DimB>
__device__ __forceinline__ void so3_matrix_multiply_backward_to_A(std::array<std::array<scalar_t, DimA>, DimA> const& A, std::array<std::array<scalar_t, DimB>, DimB> const& B,
                                                                  std::array<std::array<scalar_t, 3>, 3> const& C, std::array<std::array<scalar_t, 3>, 3> const& grad_C, std::array<std::array<scalar_t, 3>, 3>& grad_A) {
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
        for (int i = 0; i < 3; i++) {
            if constexpr (transpose_A) {
                grad_A[k][i] = 0.0f;
            } else {
                grad_A[i][k] = 0.0f;
            }

#pragma unroll
            for (int j = 0; j < 3; j++) {
                if constexpr (transpose_A) {
                    grad_A[k][i] += B[k][j] * grad_C[i][j];
                } else {
                    grad_A[i][k] += B[k][j] * grad_C[i][j];
                }
            }
        }
    }
}

template <bool transpose_A, typename scalar_t, size_t DimA, size_t DimB>
__device__ __forceinline__ void so3_matrix_multiply_backward_to_B(std::array<std::array<scalar_t, DimA>, DimA> const& A, std::array<std::array<scalar_t, DimB>, DimB> const& B,
                                                                  std::array<std::array<scalar_t, 3>, 3> const& C, std::array<std::array<scalar_t, 3>, 3> const& grad_C, std::array<std::array<scalar_t, 3>, 3>& grad_B) {
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            grad_B[k][j] = 0.0f;

#pragma unroll
            for (int i = 0; i < 3; i++) {
                if constexpr (transpose_A) {
                    grad_B[k][j] += A[k][i] * grad_C[i][j];
                } else {
                    grad_B[k][j] += A[i][k] * grad_C[i][j];
                }
            }
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void so3_matrix_to_quat(std::array<std::array<scalar_t, 3>, 3> const& R, std::array<scalar_t, 4>& q_normalized) {

    auto const t0{R[0][0]};
    auto const t1{R[1][1]};
    auto const t2{R[2][2]};
    auto const trace{t0 + t1 + t2};
    // Create decision matrix and find largest element
    std::array<scalar_t, 4> decision_matrix;
    decision_matrix[0] = t0;
    decision_matrix[1] = t1;
    decision_matrix[2] = t2;
    decision_matrix[3] = trace;

    // Find index of largest element
    int choice       = 0;
    scalar_t max_val = decision_matrix[0];
    for (int i = 1; i < 4; i++) {
        if (decision_matrix[i] > max_val) {
            max_val = decision_matrix[i];
            choice  = i;
        }
    }

    std::array<scalar_t, 4> q;
    // Compute quaternion based on largest element
    if (choice != 3) { // Not trace
        int i = choice;
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;

        q[i] = 1.0f + R[i][i] - R[j][j] - R[k][k];
        q[j] = R[j][i] + R[i][j];
        q[k] = R[k][i] + R[i][k];
        q[3] = R[k][j] - R[j][k];
    } else { // Trace is largest
        q[0] = R[2][1] - R[1][2];
        q[1] = R[0][2] - R[2][0];
        q[2] = R[1][0] - R[0][1];
        q[3] = 1.0f + trace;
    }

    quat_normalize(q, q_normalized);
}

template <typename scalar_t>
__device__ __forceinline__ void so3_matrix_to_quat_backward(std::array<std::array<scalar_t, 3>, 3> const& R, std::array<scalar_t, 4> const& q_normalized,
                                                            std::array<scalar_t, 4> const& grad_q_normalized, std::array<std::array<scalar_t, 3>, 3>& grad_R) {

    // Initialize all gradient elements to zero
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            grad_R[i][j] = 0.0f;
        }
    }

    auto const t0{R[0][0]};
    auto const t1{R[1][1]};
    auto const t2{R[2][2]};
    auto const trace{t0 + t1 + t2};
    // Create decision matrix and find largest element
    std::array<scalar_t, 4> decision_matrix;
    decision_matrix[0] = t0;
    decision_matrix[1] = t1;
    decision_matrix[2] = t2;
    decision_matrix[3] = trace;

    // Find index of largest element
    int choice       = 0;
    scalar_t max_val = decision_matrix[0];
    for (int i = 1; i < 4; i++) {
        if (decision_matrix[i] > max_val) {
            max_val = decision_matrix[i];
            choice  = i;
        }
    }

    std::array<scalar_t, 4> q;
    // Compute quaternion based on largest element
    if (choice != 3) { // Not trace
        int i = choice;
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;

        q[i] = 1.0f + R[i][i] - R[j][j] - R[k][k];
        q[j] = R[j][i] + R[i][j];
        q[k] = R[k][i] + R[i][k];
        q[3] = R[k][j] - R[j][k];
    } else { // Trace is largest
        q[0] = R[2][1] - R[1][2];
        q[1] = R[0][2] - R[2][0];
        q[2] = R[1][0] - R[0][1];
        q[3] = 1.0f + trace;
    }

    // TODO: get q directly from forward pass
    std::array<scalar_t, 4> grad_q;
    quat_normalize_backward(q, q_normalized, grad_q_normalized, grad_q);

    if (choice != 3) { // Not trace
        // dq_i/dR_jk = delta_jk * (2 * delta_ij - 1)

        int i = choice;
        int j = (i + 1) % 3;
        int k = (j + 1) % 3;

        /*
        q[i] = 1.0f + R[i][i] - R[j][j] - R[k][k];
        q[j] = R[j][i] + R[i][j];
        q[k] = R[k][i] + R[i][k];
        q[3] = R[k][j] - R[j][k];
        */
        grad_R[i][i] = grad_q[i];
        grad_R[j][j] = -grad_q[i];
        grad_R[k][k] = -grad_q[i];
        grad_R[i][j] = grad_q[j];
        grad_R[j][i] = grad_q[j];
        grad_R[i][k] = grad_q[k];
        grad_R[k][i] = grad_q[k];
        grad_R[j][k] = -grad_q[3];
        grad_R[k][j] = grad_q[3];

    } else { // Trace is largest

        /*
        q[0] = R[2][1] - R[1][2];
        q[1] = R[0][2] - R[2][0];
        q[2] = R[1][0] - R[0][1];
        q[3] = 1.0f + R[0][0] + R[1][1] + R[2][2];
        */

        grad_R[0][0] = grad_q[3];
        grad_R[1][1] = grad_q[3];
        grad_R[2][2] = grad_q[3];
        grad_R[0][1] = -grad_q[2];
        grad_R[1][0] = grad_q[2];
        grad_R[0][2] = grad_q[1];
        grad_R[2][0] = -grad_q[1];
        grad_R[1][2] = -grad_q[0];
        grad_R[2][1] = grad_q[0];
    }
}

// Transform quaternion to camera space using direct matrix multiplication (matching Python)
// D can be 3 or 4
template <typename scalar_t, size_t D>
__device__ __forceinline__ void transform_quaternion_direct(std::array<std::array<scalar_t, D>, D> const& T_cam_world,
                                                            std::array<scalar_t, 4> const& q_world, std::array<scalar_t, 4>& q_cam) {

    std::array<std::array<scalar_t, 3>, 3> R_world_quat;
    std::array<std::array<scalar_t, 3>, 3> R_cam_quat_tmp;
    std::array<std::array<scalar_t, 3>, 3> R_cam_quat_tmp2;
    std::array<std::array<scalar_t, 3>, 3> R_cam_quat;

    quat_to_so3_matrix(q_world, R_world_quat);
    so3_matrix_multiply<true>(T_cam_world, R_world_quat, R_cam_quat_tmp);
    so3_matrix_multiply<false>(R_cam_quat_tmp, T_cam_world, R_cam_quat);

    so3_matrix_to_quat(R_cam_quat, q_cam);
}

template <typename scalar_t, size_t D>
__device__ __forceinline__ void transform_quaternion_direct_backward(std::array<std::array<scalar_t, D>, D> const& T_cam_world,
                                                                     std::array<scalar_t, 4> const& q_world,
                                                                     std::array<scalar_t, 4> const& q_cam, std::array<scalar_t, 4> const& grad_q_cam,
                                                                     std::array<scalar_t, 4>& grad_q_world) {

    std::array<std::array<scalar_t, 3>, 3> R_cam_quat_tmp;
    std::array<std::array<scalar_t, 3>, 3> R_cam_quat;
    std::array<std::array<scalar_t, 3>, 3> R_world_quat;

    quat_to_so3_matrix(q_world, R_world_quat);
    so3_matrix_multiply<true>(T_cam_world, R_world_quat, R_cam_quat_tmp);
    so3_matrix_multiply<false>(R_cam_quat_tmp, T_cam_world, R_cam_quat);

    // Convert result back to quaternion
    // so3_matrix_to_quat(R_cam_quat, q_cam);

    std::array<std::array<scalar_t, 3>, 3> grad_R_cam_quat;

    so3_matrix_to_quat_backward(R_cam_quat, q_cam, grad_q_cam, grad_R_cam_quat);

    std::array<std::array<scalar_t, 3>, 3> grad_R_cam_quat_tmp;
    std::array<std::array<scalar_t, 3>, 3> grad_R_world_quat;
    so3_matrix_multiply_backward_to_A<false>(R_cam_quat_tmp, T_cam_world, R_cam_quat, grad_R_cam_quat, grad_R_cam_quat_tmp);
    so3_matrix_multiply_backward_to_B<true>(T_cam_world, R_world_quat, R_cam_quat_tmp, grad_R_cam_quat_tmp, grad_R_world_quat);

    quat_to_so3_matrix_backward(q_world, R_world_quat, grad_R_world_quat, grad_q_world);
}

// Transform quaternion to angles (pitch, roll)
// x, y, z, w = quat.unbind(-1)  # Extract components from [..., 4], each component is of shape [...]

// # Rotation around Y-axis pointing down (yaw)
// siny = torch.sqrt(1.0 + 2.0 * (w * y - x * z))
// cosy = torch.sqrt(1.0 - 2.0 * (w * y - x * z))
// # siny and cosy are square roots (non-negative) => atan2(siny, cosy) is in [0, pi/2].
// yaw = 2.0 * torch.atan2(siny, cosy) - math.pi / 2

template <typename scalar_t>
__device__ __forceinline__ void quat_to_sincos(std::array<scalar_t, 4> const& q, std::array<scalar_t, 2>& sincos) {
    sincos[0] = 2.F * (q[3] * q[2] + q[0] * q[1]);
    sincos[1] = 1.F - 2.F * (q[1] * q[1] + q[2] * q[2]);
}

template <typename scalar_t>
__device__ __forceinline__ void quat_to_sincos_backward_accumulate(std::array<scalar_t, 4> const& q, std::array<scalar_t, 2> const& sincos, std::array<scalar_t, 2> const& grad_sincos, scalar_t& grad_q0, scalar_t& grad_q1, scalar_t& grad_q2, scalar_t& grad_q3) {

    auto const grad_sin_2{2.F * grad_sincos[0]};
    auto const grad_cos_4{4.F * grad_sincos[1]};

    // Gradient for q[0]
    grad_q0 += grad_sin_2 * q[1];

    // Gradient for q[1]
    grad_q1 += grad_sin_2 * q[0] - grad_cos_4 * q[1];

    // Gradient for q[2]
    grad_q2 += grad_sin_2 * q[3] - grad_cos_4 * q[2];

    // Gradient for q[3]
    grad_q3 += grad_sin_2 * q[2];
}

template <typename scalar_t>
__device__ __forceinline__ void atan2_backward(scalar_t const& sin, scalar_t const& cos, scalar_t const& grad_atan2, scalar_t& grad_sin, scalar_t& grad_cos) {
    auto const denom{std::max(std::numeric_limits<scalar_t>::epsilon(), sin * sin + cos * cos)};
    auto const grad_atan2_denom_inv{grad_atan2 / denom};
    grad_sin = grad_atan2_denom_inv * cos;
    grad_cos = -grad_atan2_denom_inv * sin;
}

// return torch.stack([roll, pitch, yaw], dim=-1)  # Stack the three angles into a single tensor of shape [..., 3]
template <typename scalar_t>
__device__ __forceinline__ void quat_to_roll_pitch(std::array<scalar_t, 4> const& q, std::array<scalar_t, 2>& roll_pitch) {

    auto const compute_xyzw = [w = q[3]](scalar_t const& x, scalar_t const& y, scalar_t const& z) {
        std::array<scalar_t, 2> sincos;
        quat_to_sincos(std::array<scalar_t, 4>{x, y, z, w}, sincos);
        return std::atan2(sincos[0], sincos[1]);
    };

    roll_pitch[0] = compute_xyzw(q[0], q[1], q[2]);
    roll_pitch[1] = compute_xyzw(q[2], q[1], q[0]);
}

template <typename scalar_t>
__device__ __forceinline__ void quat_to_roll_pitch_backward(std::array<scalar_t, 4> const& q, std::array<scalar_t, 2> const& roll_pitch, std::array<scalar_t, 2> const& grad_roll_pitch, std::array<scalar_t, 4>& grad_q) {

    auto const compute_xyzw_backward = [w = q[3], &grad_w = grad_q[3]](scalar_t const& x, scalar_t const& y, scalar_t const& z, scalar_t const& grad_roll_pitch, scalar_t& grad_x, scalar_t& grad_y, scalar_t& grad_z) {
        std::array<scalar_t, 2> sincos;
        std::array<scalar_t, 4> q{x, y, z, w};

        quat_to_sincos(q, sincos);

        std::array<scalar_t, 2> grad_sincos;
        atan2_backward(sincos[0], sincos[1], grad_roll_pitch, grad_sincos[0], grad_sincos[1]);

        quat_to_sincos_backward_accumulate(q, sincos, grad_sincos, grad_x, grad_y, grad_z, grad_w);
    };

#pragma unroll
    for (int i = 0; i < 4; i++) {
        grad_q[i] = 0.F;
    }

    compute_xyzw_backward(q[0], q[1], q[2], grad_roll_pitch[0], grad_q[0], grad_q[1], grad_q[2]);
    compute_xyzw_backward(q[2], q[1], q[0], grad_roll_pitch[1], grad_q[2], grad_q[1], grad_q[0]);
}

template <typename scalar_t>
class RoadGaussiansEstimator {

    std::array<std::array<scalar_t, 4>, 4> T_world_cam_;
    std::array<std::array<scalar_t, 4>, 4> T_cam_world_;
    std::array<scalar_t, 4> quat_world_;
    std::array<scalar_t, 4> quat_cam_;
    std::array<scalar_t, 2> roll_pitch_;
    std::array<scalar_t, 3> pos_world_;
    std::array<scalar_t, 3> pos_cam_;

    std::array<scalar_t, 7> camera_tquat_;

public:
    __device__ scalar_t get_pos_cam_y() const { return pos_cam_[1]; }
    __device__ scalar_t get_pos_cam_z() const { return pos_cam_[2]; }
    __device__ scalar_t get_roll() const { return roll_pitch_[0]; }
    __device__ scalar_t get_pitch() const { return roll_pitch_[1]; }

    __device__ auto get_T_cam_world() const { return T_cam_world_; }

    __device__ __forceinline__ RoadGaussiansEstimator(
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> tquat_cam_world,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> positions_world,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rotations_world,
        int const idx)
        : pos_world_({positions_world[idx][0], positions_world[idx][1], positions_world[idx][2]}), quat_world_({rotations_world[idx][0], rotations_world[idx][1], rotations_world[idx][2], rotations_world[idx][3]}) {

        tquat_to_se3_matrix(std::array<scalar_t, 7>{tquat_cam_world[0], tquat_cam_world[1], tquat_cam_world[2], tquat_cam_world[3], tquat_cam_world[4], tquat_cam_world[5], tquat_cam_world[6]}, T_cam_world_);
        se3_matrix_inverse(T_cam_world_, T_world_cam_);

        // Transform point to camera space
        se3_transform_point(T_world_cam_, pos_world_, pos_cam_);

        // Transform quaternion to camera space
        transform_quaternion_direct(T_cam_world_, quat_world_, quat_cam_);

        // Transform quaternion to angles
        quat_to_roll_pitch(quat_cam_, roll_pitch_);
    }

    // All work is now done in constructor, this method is no longer needed
    // but keeping it for API compatibility
    __device__ __forceinline__ void forward() {
        // No-op: all computation moved to constructor
    }

    __device__ __forceinline__ void backward(
        scalar_t const& grad_pos_cam_y,
        scalar_t const& grad_roll,
        scalar_t const& grad_pitch,
        std::array<scalar_t, 3>& grad_pos_world,
        std::array<scalar_t, 4>& grad_rotations_world) {

        std::array<scalar_t, 4> grad_quat_cam;
        quat_to_roll_pitch_backward(quat_cam_, roll_pitch_, std::array<scalar_t, 2>{grad_roll, grad_pitch}, grad_quat_cam);

        transform_quaternion_direct_backward(T_cam_world_, quat_world_, quat_cam_, grad_quat_cam, grad_rotations_world);

        std::array<scalar_t, 3> const grad_pos_cam{0.0f, grad_pos_cam_y, 0.0f};
        se3_transform_point_backward(T_world_cam_, pos_world_, pos_cam_, grad_pos_cam, grad_pos_world);
    }
};

template <typename scalar_t>
__device__ __forceinline__ void compute_std_dev_backward(scalar_t const& value, scalar_t const& sum,
                                                         scalar_t const& count, scalar_t const& std_dev,
                                                         scalar_t const& grad_std_dev, scalar_t& grad_value) {
    auto const mean{sum / count};
    auto const denom{(count - 1) * std_dev};
    grad_value = denom < std::numeric_limits<scalar_t>::epsilon() ? 0.0f : grad_std_dev * (1.0f / denom) * (value - mean);
}

template <typename scalar_t>
constexpr size_t road_gaussians_forward_shared_memory_size(int const n_biases) {
    return n_biases * FieldsCount * sizeof(scalar_t);
}

// CUDA forward kernel
template <typename scalar_t>
__global__ void road_gaussians_forward_estimator_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> positions_world,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rotations_world,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> tquat_cam_world,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> random_values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> stats,
    int const n_points,
    int const n_biases,
    float const min_bias,
    float const range_bias,
    float const grid_len) {

    auto const idx{blockIdx.x * blockDim.x + threadIdx.x};
    extern __shared__ scalar_t shared_sums_pool[]; // n_biases * FieldsCount;

    // Initialize shared memory with ALL threads
    auto const local_idx{threadIdx.x};
    for (uint offset{}; offset < n_biases * FieldsCount; offset += blockDim.x) {
        if (offset + local_idx < n_biases * FieldsCount) {
            shared_sums_pool[offset + local_idx] = 0;
        }
    }

    __syncthreads();

    if (idx < n_points) {
        RoadGaussiansEstimator<scalar_t> estimator(tquat_cam_world, positions_world, rotations_world, idx);

        auto const pos_cam_y{estimator.get_pos_cam_y()};
        auto const pos_cam_z{estimator.get_pos_cam_z()};
        auto const roll{estimator.get_roll()};
        auto const pitch{estimator.get_pitch()};

        // For each bias bin
        for (int bias_idx{}; bias_idx < n_biases; bias_idx++) {
            // Calculate bias range exactly like Python
            auto const bias_min{random_values[bias_idx] * range_bias + min_bias};
            auto const bias_max{bias_min + grid_len};

            auto const mask_at_bias{bias_min <= pos_cam_z && pos_cam_z < bias_max};

            if (mask_at_bias) {

                atomicAdd(&shared_sums_pool[bias_idx * FieldsCount + AccumY], pos_cam_y);
                atomicAdd(&shared_sums_pool[bias_idx * FieldsCount + AccumY2], pos_cam_y * pos_cam_y);
                atomicAdd(&shared_sums_pool[bias_idx * FieldsCount + AccumRoll], roll);
                atomicAdd(&shared_sums_pool[bias_idx * FieldsCount + AccumRoll2], roll * roll);
                atomicAdd(&shared_sums_pool[bias_idx * FieldsCount + AccumPitch], pitch);
                atomicAdd(&shared_sums_pool[bias_idx * FieldsCount + AccumPitch2], pitch * pitch);
                atomicAdd(&shared_sums_pool[bias_idx * FieldsCount + Count], 1.0f);
            }
        }
    }

    __syncthreads();

    // Accumulate shared memory results into global memory
    for (uint offset{}; offset < n_biases * FieldsCount; offset += blockDim.x) {
        if (offset + local_idx < n_biases * FieldsCount) {
            auto const bias_idx{offset + local_idx / FieldsCount};
            auto const field_idx{(offset + local_idx) % FieldsCount};
            auto const shared_value{shared_sums_pool[offset + local_idx]};
            if (fabsf(shared_value) > 0) {
                atomicAdd(&stats[bias_idx][field_idx], shared_value);
            }
        }
    }
}

// CUDA entry point function
void road_gaussians_forward_cuda(
    torch::Tensor const positions_world,
    torch::Tensor const rotations_world,
    torch::Tensor const tquat_cam_world,
    torch::Tensor const random_values,
    torch::Tensor const stats,
    torch::Tensor const total_loss,
    float min_bias,
    float range_bias,
    float grid_len,
    float rotation_lambda) {
#ifdef ROADGAUSSIANS_CHRONO
    auto const start_time{std::chrono::high_resolution_clock::now()};
#endif
    CHECK_INPUT(positions_world);
    CHECK_INPUT(rotations_world);
    CHECK_INPUT(tquat_cam_world);
    CHECK_INPUT(random_values);

    if (positions_world.size(-1) != 3 || rotations_world.size(-1) != 4) {
        throw std::invalid_argument("positions_world and rotations_world must have 3 and 4 dimensions respectively");
    }

    auto const n_points{positions_world.size(0)};
    auto const n_biases{random_values.size(0)};

    if (tquat_cam_world.numel() != 7) {
        throw std::invalid_argument("tquat_cam_world must have 7 elements, but has " + std::to_string(tquat_cam_world.numel()));
    }

    constexpr auto const threads{256};
    if (n_biases > threads) {
        throw std::invalid_argument("n_biases must be less than or equal to " + std::to_string(threads));
    }

    auto const blocks{(std::max(n_points, n_biases) + threads - 1) / threads};

    c10::cuda::OptionalCUDAGuard const device_guard(torch::device_of(positions_world));
    auto stream{c10::cuda::getCurrentCUDAStream()};

    using scalar_t = float;
    if (positions_world.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("positions_world must be a float tensor");
    }

    // Launch estimator kernel
    auto const shared_memory_size{road_gaussians_forward_shared_memory_size<scalar_t>(n_biases)};
    road_gaussians_forward_estimator_kernel<scalar_t><<<blocks, threads, shared_memory_size, stream>>>(
        positions_world.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        rotations_world.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        tquat_cam_world.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        random_values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        stats.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_points,
        n_biases,
        min_bias,
        range_bias,
        grid_len);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed");
    }

    auto const bias_blocks{div_round_up(n_biases, threads)};
    road_gaussians_forward_reductor_kernel<scalar_t, threads><<<bias_blocks, threads, 0, stream>>>(
        stats.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        total_loss.data_ptr<scalar_t>(),
        n_biases,
        rotation_lambda);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA reductor kernel launch failed");
    }
#ifdef ROADGAUSSIANS_CHRONO
    cudaDeviceSynchronize();
    auto const end_time{std::chrono::high_resolution_clock::now()};
    auto const duration{std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)};
    std::cout << "CUDA: RoadGaussiansEstimator kernel time: " << duration.count() / 1000.0 << " ms" << std::endl;
#endif
}
// CUDA backward kernel
template <typename scalar_t>
__global__ void road_gaussians_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> positions_world,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rotations_world,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> tquat_cam_world,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> random_values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> stats,
    scalar_t const* __restrict__ grad_total_loss,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_positions_world,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_rotations_world,
    int const n_points,
    int const n_biases,
    float const min_bias,
    float const range_bias,
    float const grid_len,
    float const rotation_lambda) {

    auto const idx{blockIdx.x * blockDim.x + threadIdx.x};
    if (idx >= n_points)
        return;

    auto const grad_output_value{grad_total_loss[0]}; // Scalar loss gradient

    RoadGaussiansEstimator<scalar_t> estimator(tquat_cam_world, positions_world, rotations_world, idx);

    auto const pos_cam_y{estimator.get_pos_cam_y()};
    auto const pos_cam_z{estimator.get_pos_cam_z()};
    auto const roll{estimator.get_roll()};
    auto const pitch{estimator.get_pitch()};

    // Initialize gradient accumulators
    std::array<scalar_t, 3> grad_pos_world_accum{};
    std::array<scalar_t, 4> grad_rotations_world_accum{};

    // For each bias bin that this point contributes to
    for (int bias_idx = 0; bias_idx < n_biases; bias_idx++) {
        auto const bias_min{random_values[bias_idx] * range_bias + min_bias};
        auto const bias_max{bias_min + grid_len};

        auto const mask_at_bias{bias_min <= pos_cam_z && pos_cam_z < bias_max};

        if (mask_at_bias) {

            auto const sum_y{stats[bias_idx][AccumY]};
            auto const sum_y2{stats[bias_idx][AccumY2]};
            auto const sum_roll{stats[bias_idx][AccumRoll]};
            auto const sum_roll2{stats[bias_idx][AccumRoll2]};
            auto const sum_pitch{stats[bias_idx][AccumPitch]};
            auto const sum_pitch2{stats[bias_idx][AccumPitch2]};
            auto const count{stats[bias_idx][Count]};

            if (count > 1.0f) {
                // Compute standard deviations
                auto const std_y{compute_std_dev(sum_y, sum_y2, count)};
                auto const std_roll{compute_std_dev(sum_roll, sum_roll2, count)};
                auto const std_pitch{compute_std_dev(sum_pitch, sum_pitch2, count)};

                // Compute gradients for this bias contribution
                scalar_t grad_pos_cam_y, grad_roll, grad_pitch;
                compute_std_dev_backward(pos_cam_y, sum_y, count, std_y, grad_output_value, grad_pos_cam_y);
                compute_std_dev_backward(roll, sum_roll, count, std_roll, grad_output_value, grad_roll);
                compute_std_dev_backward(pitch, sum_pitch, count, std_pitch, grad_output_value, grad_pitch);

                // Total gradient contribution from this bias
                std::array<scalar_t, 3> grad_pos_world;
                std::array<scalar_t, 4> grad_rotations_world;

                estimator.backward(grad_pos_cam_y / n_biases,
                                   rotation_lambda * grad_roll / n_biases,
                                   rotation_lambda * grad_pitch / n_biases,
                                   grad_pos_world, grad_rotations_world);

#pragma unroll
                for (int i = 0; i < 3; i++) {
                    grad_pos_world_accum[i] += grad_pos_world[i];
                }

#pragma unroll
                for (int i = 0; i < 4; i++) {
                    grad_rotations_world_accum[i] += grad_rotations_world[i];
                }
            }
        }
    }

    // Write gradients to output
    grad_positions_world[idx][0] = grad_pos_world_accum[0];
    grad_positions_world[idx][1] = grad_pos_world_accum[1];
    grad_positions_world[idx][2] = grad_pos_world_accum[2];
    grad_rotations_world[idx][0] = grad_rotations_world_accum[0];
    grad_rotations_world[idx][1] = grad_rotations_world_accum[1];
    grad_rotations_world[idx][2] = grad_rotations_world_accum[2];
    grad_rotations_world[idx][3] = grad_rotations_world_accum[3];
}

// CUDA backward entry point function
void road_gaussians_backward_cuda(
    torch::Tensor const positions_world,
    torch::Tensor const rotations_world,
    torch::Tensor const tquat_cam_world,
    torch::Tensor const random_values,
    torch::Tensor const stats,
    torch::Tensor const grad_total_loss,
    torch::Tensor const grad_positions_world,
    torch::Tensor const grad_rotations_world,
    float const min_bias,
    float const range_bias,
    float const grid_len,
    float const rotation_lambda) {

#ifdef ROADGAUSSIANS_CHRONO
    auto const start_time{std::chrono::high_resolution_clock::now()};
#endif

    CHECK_INPUT(positions_world);
    CHECK_INPUT(rotations_world);
    CHECK_INPUT(tquat_cam_world);
    CHECK_INPUT(random_values);
    CHECK_INPUT(grad_total_loss);

    if (positions_world.size(-1) != 3 || rotations_world.size(-1) != 4) {
        throw std::invalid_argument("positions_world and rotations_world must have 3 and 4 dimensions respectively");
    }

    auto const n_points{positions_world.size(0)};
    auto const n_biases{random_values.size(0)};

    if (tquat_cam_world.numel() != 7) {
        throw std::invalid_argument("tquat_cam_world must have 7 elements, but has " + std::to_string(tquat_cam_world.numel()));
    }

    if (positions_world.scalar_type() != torch::kFloat32) {
        throw std::invalid_argument("positions_world must be a float tensor");
    }

    using scalar_t = float;

    c10::cuda::OptionalCUDAGuard const device_guard(torch::device_of(positions_world));
    auto stream{c10::cuda::getCurrentCUDAStream()};

    auto const threads{256};
    auto const blocks{(n_points + threads - 1) / threads};

    // Launch backward kernel
    road_gaussians_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        positions_world.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        rotations_world.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        tquat_cam_world.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        random_values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        stats.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        grad_total_loss.data_ptr<scalar_t>(),
        grad_positions_world.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        grad_rotations_world.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_points,
        n_biases,
        min_bias,
        range_bias,
        grid_len,
        rotation_lambda);
}

// ============================================================================
