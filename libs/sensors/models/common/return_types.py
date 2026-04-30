# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Return type dataclasses for Layer 2 sensor model operations.

These dataclasses provide structured return types for camera and LiDAR model methods,
enabling optional return of auxiliary information (poses, timestamps, validity flags).
"""

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class ImagePointsReturn:
    """Return type for camera_rays_to_image_points.

    Attributes:
        image_points: (N, 2) float - projected image coordinates
        valid_flag: (N,) bool - validity mask for projections
        jacobians: (N, 2, 3) float - Jacobians of the projection, or None
    """

    image_points: Tensor
    valid_flag: Tensor
    jacobians: Optional[Tensor] = None


@dataclass
class PixelsReturn:
    """Return type for camera_rays_to_pixels.

    Attributes:
        pixels: (N, 2) int - pixel indices
        valid_flag: (N,) bool - validity mask for projections
    """

    pixels: Tensor
    valid_flag: Tensor


@dataclass
class WorldPointsToImagePointsReturn:
    """Return type for world_points_to_image_points_* methods.

    Attributes:
        image_points: (N, 2) float - projected image coordinates (all points if return_all_projections=True,
                      otherwise only valid points when filtering is applied)
        T_world_sensors: (N, 4, 4) float - per-point transformation matrices, or None
        valid_flag: (N,) bool - validity mask for all input points, or None
        valid_indices: (M,) int64 - indices of valid projections relative to input points, or None
        timestamps_us: (N,) int64 - per-point timestamps, or None
    """

    image_points: Tensor
    T_world_sensors: Optional[Tensor] = None
    valid_flag: Optional[Tensor] = None
    valid_indices: Optional[Tensor] = None
    timestamps_us: Optional[Tensor] = None


@dataclass
class WorldPointsToPixelsReturn:
    """Return type for world_points_to_pixels_* methods.

    Attributes:
        pixels: (N, 2) int - pixel indices (all points if return_all_projections=True,
                otherwise only valid points when filtering is applied)
        T_world_sensors: (N, 4, 4) float - per-point transformation matrices, or None
        valid_flag: (N,) bool - validity mask for all input points, or None
        valid_indices: (M,) int64 - indices of valid projections relative to input points, or None
        timestamps_us: (N,) int64 - per-point timestamps, or None
    """

    pixels: Tensor
    T_world_sensors: Optional[Tensor] = None
    valid_flag: Optional[Tensor] = None
    valid_indices: Optional[Tensor] = None
    timestamps_us: Optional[Tensor] = None


@dataclass
class WorldRaysReturn:
    """Return type for back-projection methods (image_points_to_world_rays_*).

    Attributes:
        world_rays: (N, 6) float - [origin.xyz, direction.xyz] in world frame
        T_sensor_worlds: (N, 4, 4) float - per-ray transformation matrices, or None
        timestamps_us: (N,) int64 - per-ray timestamps, or None
    """

    world_rays: Tensor
    T_sensor_worlds: Optional[Tensor] = None
    timestamps_us: Optional[Tensor] = None


@dataclass
class SensorAnglesReturn:
    """Return type for sensor ray to angle conversions.

    Attributes:
        sensor_angles: (N, 2) float - [elevation_rad, azimuth_rad]
        valid_flag: (N,) bool - validity mask
    """

    sensor_angles: Tensor
    valid_flag: Optional[Tensor] = None


@dataclass
class SensorRayReturn:
    """Return type for sensor angle to ray conversions.

    Attributes:
        sensor_rays: (N, 3) float - normalized direction vectors
        valid_flag: (N,) bool - validity mask
    """

    sensor_rays: Tensor
    valid_flag: Optional[Tensor] = None


@dataclass
class WorldPointsToSensorAnglesReturn:
    """Return type for world points to sensor angles projection.

    Attributes:
        sensor_angles: (N, 2) float - [elevation_rad, azimuth_rad]
        T_world_sensors: (N, 4, 4) float - per-point transformation matrices, or None
        valid_flag: (N,) bool - validity mask, or None
        valid_indices: (M,) int64 - indices of valid projections relative to input points, or None
        timestamps_us: (N,) int64 - per-point timestamps, or None
    """

    sensor_angles: Tensor
    T_world_sensors: Optional[Tensor] = None
    valid_flag: Optional[Tensor] = None
    valid_indices: Optional[Tensor] = None
    timestamps_us: Optional[Tensor] = None


__all__ = [
    "ImagePointsReturn",
    "PixelsReturn",
    "SensorAnglesReturn",
    "SensorRayReturn",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldPointsToSensorAnglesReturn",
    "WorldRaysReturn",
]
