// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
//
// GSplat Strategy kernels ported to Slang
// This module provides Slang implementations of GSplat densification strategy functions

#include <slang_utils/extensions.slang>

// ============================================================================
// update_gradient_buffers_kernel
// ============================================================================
// Fused kernel for updating gradient accumulation buffers during training.
// This computes the distance-scaled gradient norms for each Gaussian.
//
// Input:
//   count              - number of Gaussians
//   positions          - [N, 3] current gaussian positions
//   params_grad        - [N, 3] gradients of positions
//   ray_origin         - [3] camera/ray origin (single origin assumed)
//   grad_norm_accum    - [N, 1] accumulated gradient norms (in-place update)
//   grad_norm_denom    - [N, 1] accumulator denominator/count (in-place update)
//
[CUDAKernel]
[AutoPyBindCUDA]
void update_gradient_buffers_kernel(
    TensorView<float> positions,        // [N, 3]
    TensorView<float> params_grad,      // [N, 3]
    TensorView<float> ray_origin,       // [3]
    TensorView<float> grad_norm_accum,  // [N, 1] in-place update
    TensorView<int> grad_norm_denom     // [N, 1] in-place update
) {
    uint2 tid = uint2(cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x, 0);
    if (tid.x >= positions.size(0)) {
        return;
    }

    // Load gradient
    float3 grad = params_grad.loadVec<3>(tid);

    // Check if this Gaussian was hit (any non-zero gradient component)
    // This matches the PyTorch check: (params_grad != 0).max(dim=1)[0]
    if (grad.x == 0.0f && grad.y == 0.0f && grad.z == 0.0f) {
        return;
    }

    // Load position and ray origin
    float3 pos = positions.loadVec<3>(tid);
    float3 origin = ray_origin.loadVec<3>(0);

    // Compute distance to camera
    float3 diff = pos - origin;
    float distance = length(diff);

    // Scale gradient by distance and compute norm, divided by 2
    float3 scaled_grad = grad * distance;
    float grad_norm = length(scaled_grad) * 0.5f;

    // Note: We use simple addition since each Gaussian is processed by one thread
    grad_norm_accum[tid] = grad_norm_accum[tid] + grad_norm;
    grad_norm_denom[tid] = grad_norm_denom[tid] + 1;
}
