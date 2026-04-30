# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Sensors module - Camera and LiDAR projection with GPU acceleration.

This module provides the sensors architecture with two layers:
- Layer 0 (kernels): GPU kernels implemented in Slang with Python bindings
- Layer 2 (models): Stateful nn.Module wrappers with learnable parameters

The module is organized into:
- kernels.cameras: Camera projection kernels and parameters
- kernels.lidars: LiDAR projection kernels and parameters
- models.cameras: Camera model classes (OpenCVPinhole, OpenCVFisheye, FTheta)
- models.lidars: LiDAR model classes
- models.frame: Frame structures for sensor observations

Example usage:
    # Layer 0: Direct kernel access
    from libs.sensors.kernels import cameras, lidars

    image_points, valid = cameras.camera_rays_to_image_points(
        camera_rays, projection, external_distortion
    )

    # Layer 2: High-level model API
    from libs.sensors import models

    camera = models.OpenCVPinholeCameraModel.from_ncore(ncore_camera_model)
    result = camera.world_points_to_image_points_shutter_pose(
        world_points, dynamic_pose
    )
"""

from libs.sensors import kernels


__all__ = [
    "kernels",
]
