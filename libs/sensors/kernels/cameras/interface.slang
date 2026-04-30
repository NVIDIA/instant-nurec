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

// Camera projection interfaces and common types

// Shutter behavior enum
enum ShutterType : int {
    ROLLING_TOP_TO_BOTTOM = 1,
    ROLLING_LEFT_TO_RIGHT = 2,
    ROLLING_BOTTOM_TO_TOP = 3,
    ROLLING_RIGHT_TO_LEFT = 4,
    GLOBAL = 5
};

// Result of projecting a camera ray to image coordinates
struct CameraProjectResult : IDifferentiable {
    float2 image_point;
    no_diff bool valid;
};

// Camera projection interface: defines the core camera model operations
// All camera parameter types must implement this interface
// Extends IDifferentiable to enable gradient flow through generic functions
interface ICameraProjection : IDifferentiable {
    // Project camera ray to image point (normalized coordinates + intrinsics)
    // Returns the 2D image point and validity flag
    [Differentiable]
    CameraProjectResult camera_rays_to_image_points(float3 camera_ray);

    // Back project image point to camera ray (inverse of above)
    // Returns normalized 3D direction vector in camera frame
    [Differentiable]
    float3 image_points_to_camera_rays(float2 image_point);
};

// External distortion interface: defines 3D ray distortion operations
// External distortion is applied before the camera projection (earliest in pipeline)
// Extends IDifferentiable to enable gradient flow through generic functions
interface IExternalDistortion : IDifferentiable {
    // Apply external distortion to a camera ray
    [Differentiable]
    float3 distort(float3 camera_ray);

    // Remove external distortion from a camera ray
    [Differentiable]
    float3 undistort(float3 distorted_ray);
};

