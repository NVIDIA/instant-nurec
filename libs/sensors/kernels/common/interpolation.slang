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

// Pose interpolation utilities for sensor kernels.
// Used for rolling shutter compensation in both cameras and LiDARs.
//
// Note: This file requires <geometry/kernels/pose.slang> to be included
// by the consuming file before this one for SE3Pose and quaternion types.

// ============================================================================
// Pose Interpolation
// ============================================================================

// Interpolates a pose from two control poses at a given relative time.
// Uses linear interpolation for translation and SLERP for rotation.
//
// Parameters:
//   relative_time: Time in [0, 1]
//   control_trans0, control_trans1: Translation of control poses
//   control_rot0, control_rot1: Rotation quaternions of control poses
//   control_time0, control_time1: Times of control poses [0, 1]
//
// Returns:
//   Interpolated SE3Pose
//
[Differentiable]
SE3Pose interpolate_pose_2poses(
    float relative_time,
    float3 control_trans0,
    float3 control_trans1,
    float4 control_rot0,
    float4 control_rot1,
    float control_time0,
    float control_time1
) {
    float alpha = (control_time1 == control_time0) ? 0.0 : (relative_time - control_time0) / (control_time1 - control_time0);
    
    float3 trans = lerp(control_trans0, control_trans1, alpha);
    float4 rot = quaternion::slerp(control_rot0, control_rot1, alpha);
    
    return SE3Pose.from_translation_quaternion(trans, rot);
}

// Interpolates between two SE3 poses using a precomputed alpha.
// Uses SLERP for rotation and linear interpolation for translation.
//
// Parameters:
//   a: Start pose
//   b: End pose
//   alpha: Interpolation factor in [0, 1], where 0 returns a and 1 returns b
//
// Returns:
//   Interpolated transform::SE3
//
[Differentiable]
transform::SE3 interpolate_pose_2poses(transform::SE3 a, transform::SE3 b, float alpha) {
    float4 q_interp = quaternion::normalizeSafe(quaternion::slerp(
        a.so3.unit_quaternion.q,
        b.so3.unit_quaternion.q,
        alpha
    ));
    float3 t_interp = lerp(a.trans.p, b.trans.p, alpha);
    
    return transform::SE3.from_vec(t_interp, q_interp);
}