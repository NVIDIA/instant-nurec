# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Camera kernels package - Layer 0 GPU operations for camera projection."""

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch

import libs.sensors.libcamera_slang_cc as camera_slang  # type: ignore # pycena: skip

from libs.sensors.kernels.cameras.bindings import (
    camera_rays_to_image_points,
    generate_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_shutter_pose,
    image_points_to_world_rays_static_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
)
from libs.sensors.kernels.cameras.parameters import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
)


__all__ = [
    # Slang module
    "camera_slang",
    # Enums
    "ShutterType",
    "ReferencePolynomial",
    "FThetaPolynomialType",
    # External distortion
    "ExternalDistortion",
    "NoExternalDistortion",
    "BivariateWindshieldDistortion",
    # Camera projections
    "CameraProjection",
    "OpenCVPinholeProjection",
    "OpenCVFisheyeProjection",
    "FThetaProjection",
    # Kernel functions
    "generate_image_points",
    "camera_rays_to_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
]
