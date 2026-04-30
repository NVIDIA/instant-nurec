# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Row-offset structured spinning LiDAR model with properly-typed projection.

This module provides the Row-offset structured spinning LiDAR model with properly-typed projection.
"""

from libs.sensors.kernels.lidars import RowOffsetStructuredSpinningLidarProjection
from libs.sensors.models.lidars.lidar_model import LidarModel


class RowOffsetStructuredSpinningLidarModel(LidarModel):
    """Row-offset structured spinning LiDAR model with properly-typed projection.

    Extends LidarModel with RowOffsetStructuredSpinningLidarProjection for type-safe
    parameter access. Compatible with Hesai P128, Waymo, Pandar, and similar sensors.

    Example usage:
        # Create LiDAR model with projection parameters
        projection = RowOffsetStructuredSpinningLidarProjection(...)
        lidar = RowOffsetStructuredSpinningLidarModel(
            projection=projection,
            angles_to_columns_map_init=True,
        )

        # Access projection parameters with proper typing
        n_rows = lidar.projection.n_rows
        elevations = lidar.projection.row_elevations_rad
        offsets = lidar.projection.row_azimuth_offsets_rad  # Type-safe access

        # Generate rays with rolling shutter
        result = lidar.elements_to_world_rays_shutter_pose(elements, dynamic_pose)
    """

    _projection: RowOffsetStructuredSpinningLidarProjection

    def __init__(
        self,
        projection: RowOffsetStructuredSpinningLidarProjection,
        angles_to_columns_map_init: bool = False,
        fov_eps_factor: float = 4.0,
    ):
        """Initialize row-offset structured spinning LiDAR model.

        Args:
            projection: Row-offset structured spinning LiDAR projection parameters
            angles_to_columns_map_init: If True, eagerly builds the angles-to-columns map
                for efficient inverse projection. If False, the map will be built lazily
                on first use. Building the map upfront is recommended for production use.
            fov_eps_factor: Factor for FOV epsilon used in validity checks to account
                for accumulated numerical errors. Default is 4.0.
        """
        super().__init__(fov_eps_factor=fov_eps_factor)
        self._projection = projection

        if angles_to_columns_map_init:
            projection.ensure_angles_map()

    @property
    def projection(self) -> RowOffsetStructuredSpinningLidarProjection:
        """Get the row-offset structured spinning LiDAR projection parameters."""
        return self._projection


__all__ = [
    "RowOffsetStructuredSpinningLidarModel",
]
