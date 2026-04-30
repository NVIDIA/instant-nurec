// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
//
// MCMC Strategy kernels ported to Slang
// This module provides Slang implementations of MCMC gaussian perturbation functions

#include <geometry/kernels/quaternion.slang>
#include <slang_gaussians/common/sigmoid.slang>
#include <slang_utils/extensions.slang>

// Fully fused kernel that accepts pre-activation values and applies activations internally
// This eliminates PyTorch normalize, exp, and sigmoid calls
//
// Input:
//   positions [N, 3] - current gaussian positions
//   quats [N, 4] - PRE-ACTIVATION quaternions (not normalized)
//   scales [N, 3] - PRE-ACTIVATION scales (before exp)
//   densities [N] - PRE-ACTIVATION densities (before sigmoid)
//   noise [N, 3] - random noise from torch.randn_like(positions)
//   current_lr - learning rate * noise_lr factor
//   wxyz_format - if true, quaternions are in wxyz format; if false, xyzw format
// Output:
//   positions [N, 3] - perturbed positions (in-place update)
[CUDAKernel]
[AutoPyBindCUDA]
void fused_perturb_gaussians_kernel(
    uint count,
    TensorView<float> positions,   // [N, 3] - positions to perturb (in-place)
    TensorView<float> quats,       // [N, 4] - PRE-ACTIVATION (not normalized)
    TensorView<float> scales,      // [N, 3] - PRE-ACTIVATION (before exp)
    TensorView<float> densities,   // [N] - PRE-ACTIVATION (before sigmoid)
    TensorView<float> noise,       // [N, 3] - random noise
    float current_lr,
    bool wxyz_format
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    // Load and normalize quaternion (fuses nn.functional.normalize)
    float4 raw_quat = quats.loadVec<4>(uint2(tid, 0));
    float4 quat = quaternion::normalizeSafe(quaternion::wxyzToXyzw(raw_quat, wxyz_format));

    // Load and activate scales (fuses torch.exp)
    float3 scale = exp(scales.loadVec<3>(uint2(tid, 0)));

    // Compute covariance matrix
    float3x3 covar = quaternion::quatScaleToCovariance(quat, scale);

    // Load and activate density (fuses torch.sigmoid)
    float density = sigmoid::standard(densities[tid]);

    // Load noise
    float3 n = noise.loadVec<3>(uint2(tid, 0));

    // Scale noise by sigmoid::steep(1 - density) * current_lr
    // sigmoid::steep gives higher weight to low opacity gaussians
    float weight = sigmoid::steep(1.0f - density) * current_lr;
    n = n * weight;

    // Multiply by covariance: noise = covar * noise
    float3 transformed_noise = float3(
        covar[0][0] * n.x + covar[0][1] * n.y + covar[0][2] * n.z,
        covar[1][0] * n.x + covar[1][1] * n.y + covar[1][2] * n.z,
        covar[2][0] * n.x + covar[2][1] * n.y + covar[2][2] * n.z
    );

    // Add noise to positions (in-place update)
    positions[uint2(tid, 0)] = positions[uint2(tid, 0)] + transformed_noise.x;
    positions[uint2(tid, 1)] = positions[uint2(tid, 1)] + transformed_noise.y;
    positions[uint2(tid, 2)] = positions[uint2(tid, 2)] + transformed_noise.z;
}

// Fully fused kernel for RigidGaussianModel with cuboid bounds constraint
// Accepts pre-activation values and applies activations internally
[CUDAKernel]
[AutoPyBindCUDA]
void fused_perturb_gaussians_rigid_kernel(
    uint count,
    TensorView<float> positions,      // [N, 3] - positions to perturb (in-place)
    TensorView<float> quats,          // [N, 4] - PRE-ACTIVATION (not normalized)
    TensorView<float> scales,         // [N, 3] - PRE-ACTIVATION (before exp)
    TensorView<float> densities,      // [N] - PRE-ACTIVATION (before sigmoid)
    TensorView<float> noise,          // [N, 3] - random noise
    TensorView<float> cuboid_dims,    // [N, 3] - cuboid dimensions per gaussian
    float current_lr,
    bool wxyz_format
) {
    uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) {
        return;
    }

    // Load and normalize quaternion (fuses nn.functional.normalize)
    float4 raw_quat = quats.loadVec<4>(uint2(tid, 0));
    float4 quat = quaternion::normalizeSafe(quaternion::wxyzToXyzw(raw_quat, wxyz_format));

    // Load and activate scales (fuses torch.exp)
    float3 scale = exp(scales.loadVec<3>(uint2(tid, 0)));

    // Compute covariance matrix
    float3x3 covar = quaternion::quatScaleToCovariance(quat, scale);

    // Load and activate density (fuses torch.sigmoid)
    float density = sigmoid::standard(densities[tid]);

    // Load noise
    float3 n = noise.loadVec<3>(uint2(tid, 0));

    // Scale noise by sigmoid::steep(1 - density) * current_lr
    float weight = sigmoid::steep(1.0f - density) * current_lr;
    n = n * weight;

    // Multiply by covariance
    float3 transformed_noise = float3(
        covar[0][0] * n.x + covar[0][1] * n.y + covar[0][2] * n.z,
        covar[1][0] * n.x + covar[1][1] * n.y + covar[1][2] * n.z,
        covar[2][0] * n.x + covar[2][1] * n.y + covar[2][2] * n.z
    );

    // Load current position
    float3 pos = positions.loadVec<3>(uint2(tid, 0));

    // Load cuboid dimensions and compute bounds
    float3 dims = cuboid_dims.loadVec<3>(uint2(tid, 0));
    float3 bounds = dims * 0.5f;

    // Compute distances from bounds
    float3 cur_pos_dist = abs(pos) - bounds;
    float3 pos_with_noise = pos + transformed_noise;
    float3 pos_with_noise_dist = abs(pos_with_noise) - bounds;

    // Apply cuboid constraint per-component:
    // - Don't move from inside to outside
    // - Don't move from outside to further outside
    [ForceUnroll]
    for (int i = 0; i < 3; i++) {
        bool inside_to_outside = (cur_pos_dist[i] < 0.0f) && (pos_with_noise_dist[i] >= 0.0f);
        bool outside_to_more_outside = (cur_pos_dist[i] >= 0.0f) && (pos_with_noise_dist[i] > cur_pos_dist[i]);
        if (inside_to_outside || outside_to_more_outside) {
            transformed_noise[i] = 0.0f;
        }
    }

    // Add noise to positions (in-place update)
    positions[uint2(tid, 0)] = pos.x + transformed_noise.x;
    positions[uint2(tid, 1)] = pos.y + transformed_noise.y;
    positions[uint2(tid, 2)] = pos.z + transformed_noise.z;
}
