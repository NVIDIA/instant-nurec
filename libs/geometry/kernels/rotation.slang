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

// Converts 6D rotation representation to 3x3 rotation matrix.
// Uses Gram-Schmidt orthogonalization per Zhou et al.
// "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019)
// col1_offset: offset for first column from identity column (1, 0, 0)
// col2_offset: offset for second column from identity column (0, 1, 0)
// Output matches PyTorch's torch.stack((b1, b2, b3), dim=-2) where each bi is a row.
[Differentiable]
float3x3 rotation6dToMatrix(float3 col1_offset, float3 col2_offset) {
    // Reconstruct raw columns: identity columns + offsets
    float3 a1 = float3(1.0 + col1_offset.x, col1_offset.y, col1_offset.z);
    float3 a2 = float3(col2_offset.x, 1.0 + col2_offset.y, col2_offset.z);
    
    // Gram-Schmidt orthogonalization
    // Step 1: Normalize first vector
    float3 b1 = normalize(a1);
    
    // Step 2: Make second vector orthogonal to first, then normalize
    // b2 = normalize(a2 - proj_{b1}(a2)) = normalize(a2 - (a2·b1)*b1)
    float3 b2 = normalize(a2 - dot(a2, b1) * b1);
    
    // Step 3: Third vector is cross product of first two
    float3 b3 = cross(b1, b2);
    
    // Construct rotation matrix with b1, b2, b3 as rows (matching PyTorch convention)
    return float3x3(
        b1.x, b1.y, b1.z,  // row 0 = b1
        b2.x, b2.y, b2.z,  // row 1 = b2
        b3.x, b3.y, b3.z   // row 2 = b3
    );
}