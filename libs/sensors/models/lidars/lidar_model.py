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
Lidar model classes for Layer 2 - stateful nn.Module wrappers.

This module provides LidarModel class that wraps Layer 0 projection parameters
and kernel bindings with PyTorch nn.Module state management for training.
"""

from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor

from libs.geometry.kernels.pose import se3pose_to_matrix
from libs.sensors.kernels.common.pose import DynamicPose
from libs.sensors.kernels.lidars import (
    # Projection types
    LidarProjection,
    # Kernel functions
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
    sensor_angles_to_sensor_rays,
    sensor_rays_to_sensor_angles,
)
from libs.sensors.models.common.return_types import (
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from libs.sensors.models.common.utils import (
    TensorLike,
    batched_quat_slerp,
    poses_to_matrix,
    to_torch,
    valid_flags_to_indices,
)


# ============================================================================
# LiDAR Model Base Class
# ============================================================================


class LidarModel(nn.Module):
    """Abstract base class for all LiDAR models.

    This is an abstract base class. Use derived classes (RowOffsetStructuredSpinningLidarModel)
    which contain properly-typed projection members.

    The base class contains common sensor operations shared across all LiDAR types.
    Derived classes add projection-specific parameters.

    Wraps Layer 0 projection parameters as nn.Module state for gradient-based optimization.
    Calls Layer 0 kernel functions for projection operations.

    Note on Timestamps:
        - Static pose functions accept pose objects which may contain an optional `timestamp_us` field
        - If `return_timestamps=True`, the pose's timestamp (if present) will be returned in the results
        - Dynamic pose-based functions (shutter_pose) use normalized time [0, 1]
        - Timestamps are metadata only and do not affect the transformation computations

    Example usage:
        # Create LiDAR model with Layer 0 projection
        projection = RowOffsetStructuredSpinningLidarProjection(...)
        lidar = RowOffsetStructuredSpinningLidarModel(projection=projection)

        # Access projection parameters with proper typing
        n_rows = lidar.projection.n_rows
        elevations = lidar.projection.row_elevations_rad

        # Generate rays with rolling shutter using dynamic pose
        result = lidar.elements_to_world_rays_shutter_pose(
            elements, dynamic_pose
        )

    Attributes:
        projection: Layer 0 exposed type (contains all sensor parameters)
    """

    _fov_eps_rad: float

    def __init__(
        self,
        fov_eps_factor: float = 4.0,
    ):
        """Initialize LiDAR model base class.

        Args:
            fov_eps_factor: Factor for FOV epsilon used in validity checks to account
                for accumulated numerical errors. Default is 4.0.
        """
        super().__init__()
        self._fov_eps_rad = fov_eps_factor * torch.finfo(torch.float32).eps

    @property
    @abstractmethod
    def projection(self) -> LidarProjection:
        """Get the LiDAR projection parameters. Must be implemented by subclasses."""
        raise NotImplementedError

    # ============================================================================
    # Sensor Ray / Angle Conversions
    # ============================================================================

    def sensor_rays_to_sensor_angles(
        self,
        sensor_rays: TensorLike,
        normalized: bool = True,
        return_valid_flag: bool = False,
    ) -> SensorAnglesReturn:
        """Convert sensor rays to elevation/azimuth angles.

        Delegates to Layer 0 kernel.

        Args:
            sensor_rays: (N, 3) direction vectors in sensor frame (Tensor or numpy array)
            normalized: If True, assumes rays are already normalized. If False,
                normalizes rays before conversion.
            return_valid_flag: If True, return validity mask

        Returns:
            SensorAnglesReturn with sensor angles and optional validity mask
        """
        proj = self.projection
        device = proj.row_elevations_rad.device
        dtype = proj.row_elevations_rad.dtype

        # Convert numpy to torch if needed
        sensor_rays_t = to_torch(sensor_rays, device=device, dtype=dtype)
        assert isinstance(sensor_rays_t, Tensor)

        if not normalized:
            sensor_rays_t = sensor_rays_t / torch.norm(sensor_rays_t, dim=-1, keepdim=True)

        sensor_angles = sensor_rays_to_sensor_angles(
            projection=proj,
            sensor_rays=sensor_rays_t,
        )

        # Compute validity based on FOV if requested
        valid_flag = None
        if return_valid_flag:
            valid_flag = self._valid_sensor_angles(sensor_angles)

        return SensorAnglesReturn(
            sensor_angles=sensor_angles,
            valid_flag=valid_flag,
        )

    def sensor_angles_to_sensor_rays(
        self,
        sensor_angles: TensorLike,
        return_valid_flag: bool = False,
    ) -> SensorRayReturn:
        """Convert elevation/azimuth angles to sensor rays.

        Delegates to Layer 0 kernel.

        Args:
            sensor_angles: (N, 2) [elevation_rad, azimuth_rad] (Tensor or numpy array)
            return_valid_flag: If True, return validity mask

        Returns:
            SensorRayReturn with sensor rays and optional validity mask
        """
        proj = self.projection
        device = proj.row_elevations_rad.device
        dtype = proj.row_elevations_rad.dtype

        # Convert numpy to torch if needed
        sensor_angles = to_torch(sensor_angles, device=device, dtype=dtype)

        sensor_rays = sensor_angles_to_sensor_rays(
            projection=proj,
            sensor_angles=sensor_angles,
        )

        # Compute validity based on FOV if requested
        valid_flag = None
        if return_valid_flag:
            valid_flag = self._valid_sensor_angles(sensor_angles)

        return SensorRayReturn(
            sensor_rays=sensor_rays,
            valid_flag=valid_flag,
        )

    def elements_to_sensor_angles(
        self,
        elements: TensorLike,
        return_valid_flag: bool = False,
    ) -> SensorAnglesReturn:
        """Retrieves elevation and azimuth angles for elements.

        Delegates to Layer 0 kernel.

        Args:
            elements: (N, 2) int - [row, column] indices (Tensor or numpy array)
            return_valid_flag: If True, return validity mask for bounds checking

        Returns:
            SensorAnglesReturn with sensor angles and optional validity mask
        """
        proj = self.projection
        device = proj.row_elevations_rad.device

        # Convert numpy to torch if needed
        if isinstance(elements, np.ndarray):
            elements = torch.from_numpy(elements).to(device=device, dtype=torch.long)
        else:
            elements = elements.to(device=device, dtype=torch.long)

        sensor_angles, valid_flags = elements_to_sensor_angles(
            projection=proj,
            elements=elements,
            return_valid_flags=return_valid_flag,
        )

        return SensorAnglesReturn(
            sensor_angles=sensor_angles,
            valid_flag=valid_flags if return_valid_flag else None,
        )

    def elements_to_sensor_rays(
        self,
        elements: TensorLike,
    ) -> Tensor:
        """Convert elements to sensor rays.

        Combines elements_to_sensor_angles and sensor_angles_to_sensor_rays.

        Args:
            elements: (N, 2) int - [row, column] indices (Tensor or numpy array)

        Returns:
            sensor_rays: (N, 3) normalized direction vectors
        """
        angles_result = self.elements_to_sensor_angles(elements)
        rays_result = self.sensor_angles_to_sensor_rays(angles_result.sensor_angles)
        return rays_result.sensor_rays

    def elements_to_sensor_points(
        self,
        elements: TensorLike,
        element_distances: TensorLike,
    ) -> Tensor:
        """Convert elements and distances to 3D sensor points.

        Args:
            elements: (N, 2) int - [row, column] indices (Tensor or numpy array)
            element_distances: (N,) float - distances in meters (Tensor or numpy array)

        Returns:
            sensor_points: (N, 3) 3D points in sensor frame
        """
        proj = self.projection
        device = proj.row_elevations_rad.device
        dtype = proj.row_elevations_rad.dtype

        # Convert numpy to torch if needed
        element_distances = to_torch(element_distances, device=device, dtype=dtype)

        sensor_rays = self.elements_to_sensor_rays(elements)
        return sensor_rays * element_distances.unsqueeze(-1)

    # ============================================================================
    # World Ray Generation with Rolling Shutter
    # ============================================================================

    def elements_to_world_rays_shutter_pose(
        self,
        elements: TensorLike | None,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        sensor_rays: TensorLike | None = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-projects elements to world rays using rolling-shutter compensation.

        Args:
            elements: (N, 2) int - [row, column] indices, or None to generate all elements
                (Tensor or numpy array)
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            sensor_rays: (N, 3) optional pre-computed sensor rays for reuse. If provided,
                skips internal ray computation and uses these rays directly. Useful when
                the same elements are projected multiple times with different poses.
                (Tensor or numpy array)
            return_T_sensor_worlds: If True, return per-ray poses as (N, 4, 4) matrices
            return_timestamps: If True, return timestamps

        Returns:
            WorldRaysReturn with world rays and optional auxiliary data
        """
        # Convert numpy to torch if needed
        if elements is not None and isinstance(elements, np.ndarray):
            elements = torch.from_numpy(elements).to(dtype=torch.long)
        if sensor_rays is not None and isinstance(sensor_rays, np.ndarray):
            sensor_rays = torch.from_numpy(sensor_rays).to(dtype=torch.float32)

        # If sensor_rays provided, use optimized Python path (avoids recomputing rays)
        if sensor_rays is not None:
            return self._elements_to_world_rays_with_sensor_rays(
                elements=elements,
                dynamic_pose=dynamic_pose,
                sensor_rays=sensor_rays,
                start_timestamp_us=start_timestamp_us,
                end_timestamp_us=end_timestamp_us,
                return_T_sensor_worlds=return_T_sensor_worlds,
                return_timestamps=return_timestamps,
            )

        # Standard path: delegate to Layer 0 kernel
        world_rays, timestamps_us, poses_trans, poses_rot = generate_spinning_lidar_rays(
            projection=self.projection,
            elements=elements,
            dynamic_pose=dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_timestamps=return_timestamps,
            return_poses=return_T_sensor_worlds,
        )

        # Convert translations and rotations to 4x4 matrices if requested
        T_sensor_worlds = poses_to_matrix(poses_trans, poses_rot) if return_T_sensor_worlds else None

        return WorldRaysReturn(
            world_rays=world_rays,
            T_sensor_worlds=T_sensor_worlds,
            timestamps_us=timestamps_us if return_timestamps else None,
        )

    def _elements_to_world_rays_with_sensor_rays(
        self,
        elements: Tensor | None,
        dynamic_pose: DynamicPose,
        sensor_rays: Tensor,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Internal method for world ray computation with pre-computed sensor rays.

        Uses Python-based pose interpolation and transformation, matching ncore's
        implementation for sensor_rays reuse optimization.
        """
        if return_timestamps:
            assert start_timestamp_us is not None, "start_timestamp_us required when return_timestamps=True"
            assert end_timestamp_us is not None, "end_timestamp_us required when return_timestamps=True"

        proj = self.projection
        N = sensor_rays.shape[0]
        device = sensor_rays.device
        dtype = sensor_rays.dtype

        # Generate elements if not provided
        if elements is None:
            rows = torch.arange(proj.n_rows, device=device, dtype=torch.long)
            cols = torch.arange(proj.n_columns, device=device, dtype=torch.long)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
            elements = torch.stack([row_grid.flatten(), col_grid.flatten()], dim=-1)

        # elements is now guaranteed to be a Tensor (type narrowing for linter)
        assert elements is not None

        # Compute relative frame time based on column index (matches ncore)
        # Columns are measured in increasing time order irrespective of spin direction
        t = elements[:, 1].to(dtype) / (proj.n_columns - 1)

        # Get start and end poses
        pose_start = dynamic_pose.start_pose
        pose_end = dynamic_pose.end_pose

        # Interpolate translation (linear)
        world_position_rs = (1 - t).unsqueeze(-1) * pose_start.translation.unsqueeze(0) + t.unsqueeze(
            -1
        ) * pose_end.translation.unsqueeze(0)  # (N, 3)

        # Interpolate rotation (batched SLERP for per-element t)
        R_sensor_world_rs_quat = batched_quat_slerp(
            pose_start.rotation.unsqueeze(0).expand(N, -1),
            pose_end.rotation.unsqueeze(0).expand(N, -1),
            t,
        )  # (N, 4)

        # Convert quaternions to rotation matrices for transformation
        T_sensor_worlds_mat = se3pose_to_matrix(world_position_rs, R_sensor_world_rs_quat)  # (N, 4, 4)
        R_sensor_world_rs = T_sensor_worlds_mat[:, :3, :3]  # (N, 3, 3)

        # Transform sensor rays to world frame
        world_ray_directions = torch.bmm(R_sensor_world_rs, sensor_rays.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # Combine position and direction into world rays
        world_rays = torch.cat([world_position_rs, world_ray_directions], dim=-1)  # (N, 6)

        # Compute timestamps if requested
        timestamps_us = None
        if return_timestamps:
            # Assertions at top of method guarantee these are not None
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            timestamps_us = start_timestamp_us + (t * (end_timestamp_us - start_timestamp_us)).to(torch.int64)

        return WorldRaysReturn(
            world_rays=world_rays,
            T_sensor_worlds=T_sensor_worlds_mat if return_T_sensor_worlds else None,
            timestamps_us=timestamps_us,
        )

    # ============================================================================
    # World Points to Sensor Angles with Rolling Shutter
    # ============================================================================

    def world_points_to_sensor_angles_shutter_pose(
        self,
        world_points: TensorLike,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        max_iterations: int = 10,
        stop_mean_relative_time_error: float = 0.0001,
        stop_delta_mean_relative_time_error: float = 0.000001,
        lazy_init_angles_map: bool = False,
        return_T_world_sensors: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
    ) -> WorldPointsToSensorAnglesReturn:
        """Projects world points to sensor angles using rolling-shutter compensation.

        Args:
            world_points: (N, 3) world coordinates (Tensor or numpy array)
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            max_iterations: Max iterations for convergence
            stop_mean_relative_time_error: Stopping criterion for mean error
            stop_delta_mean_relative_time_error: Stopping criterion for error change
            lazy_init_angles_map: If True, lazily build angles-to-columns map
            return_T_world_sensors: If True, return per-point poses as (N, 4, 4) matrices
            return_valid_flag: If True, return validity mask
            return_valid_indices: If True, return indices of valid projections
            return_timestamps: If True, return timestamps

        Returns:
            WorldPointsToSensorAnglesReturn with sensor angles and optional auxiliary data
        """
        # Convert numpy to torch if needed
        if isinstance(world_points, np.ndarray):
            world_points = torch.from_numpy(world_points).to(dtype=torch.float32)

        # Always get valid flags if we need indices
        need_valid_flags = return_valid_flag or return_valid_indices

        sensor_angles, valid_flags, timestamps_us, poses_trans, poses_rot = inverse_project_spinning_lidar(
            projection=self.projection,
            world_points=world_points,
            dynamic_pose=dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            max_iterations=max_iterations,
            stop_mean_relative_time_error=stop_mean_relative_time_error,
            stop_delta_mean_relative_time_error=stop_delta_mean_relative_time_error,
            lazy_init_angles_map=lazy_init_angles_map,
            return_valid_flags=need_valid_flags,
            return_timestamps=return_timestamps,
            return_poses=return_T_world_sensors,
        )

        # Convert translations and rotations to 4x4 matrices if requested
        T_world_sensors = poses_to_matrix(poses_trans, poses_rot) if return_T_world_sensors else None

        # Compute valid indices if requested
        valid_indices = valid_flags_to_indices(valid_flags) if return_valid_indices else None

        return WorldPointsToSensorAnglesReturn(
            sensor_angles=sensor_angles,
            T_world_sensors=T_world_sensors,
            valid_flag=valid_flags if return_valid_flag else None,
            valid_indices=valid_indices,
            timestamps_us=timestamps_us if return_timestamps else None,
        )

    # ============================================================================
    # Utilities
    # ============================================================================

    def sensor_angles_relative_frame_times(
        self,
        sensor_angles: TensorLike,
    ) -> Tensor:
        """Get relative frame-times [0,1] for sensor angle coordinates.

        Uses the precomputed angles-to-columns map for proper handling of
        per-row azimuth offsets. This matches ncore's implementation.

        All sensor angles should be within the FOV of the sensor.

        Args:
            sensor_angles: (N, 2) [elevation_rad, azimuth_rad] (Tensor or numpy array)

        Returns:
            relative_times: (N,) float in [0, 1]
        """
        # Convert numpy to torch if needed
        if isinstance(sensor_angles, np.ndarray):
            sensor_angles = torch.from_numpy(sensor_angles).to(dtype=torch.float32)

        proj = self.projection
        device = sensor_angles.device
        dtype = sensor_angles.dtype

        # Ensure the angles-to-columns map is built
        proj.ensure_angles_map()

        angles_map = proj.angles_to_columns_map
        if angles_map is None:
            raise RuntimeError("angles_to_columns_map should be built after ensure_angles_map()")

        # Compute relative sensor angles (offset from FOV start)
        relative_sensor_angles = self._relative_sensor_angles(sensor_angles)
        relative_elevations_rad = relative_sensor_angles[:, 0]
        relative_azimuths_rad = relative_sensor_angles[:, 1]

        # Compute map resolution
        map_height = proj.angles_to_columns_map_resolution_factor * proj.n_rows
        map_width = proj.angles_to_columns_map_resolution_factor * proj.n_columns
        map_resolution_horiz_rad = proj.fov_horiz_span_rad / (map_width - 1)
        map_resolution_vert_rad = proj.fov_vert_span_rad / (map_height - 1)

        # Determine the location of the angle in the map (nearest neighbor lookup)
        # Formula: (angle + res/2) / res = angle/res + 0.5
        horizontal_nn_dist = relative_azimuths_rad / map_resolution_horiz_rad + 0.5
        horizontal_idxs = horizontal_nn_dist.to(torch.long).clamp(0, map_width - 1)

        vertical_nn_dist = relative_elevations_rad / map_resolution_vert_rad + 0.5
        vertical_idxs = vertical_nn_dist.to(torch.long).clamp(0, map_height - 1)

        # Grab the corresponding column index from the map
        column_indices = angles_map[vertical_idxs, horizontal_idxs]

        # Compute the relative frame time using the column-associated relative time
        return column_indices.to(dtype) / (proj.n_columns - 1)

    def _relative_sensor_angles(self, sensor_angles: Tensor) -> Tensor:
        """Compute relative sensor angles from FOV start.

        Args:
            sensor_angles: (N, 2) [elevation_rad, azimuth_rad]

        Returns:
            relative_angles: (N, 2) [relative_elevation, relative_azimuth] in radians
        """
        proj = self.projection
        elevations = sensor_angles[:, 0]
        azimuths = sensor_angles[:, 1]

        # Compute relative elevation (vertical direction is always cw from top)
        # fov_vert_start is at top, span goes downward
        rel_elevation = self._relative_angle(proj.fov_vert_start_rad, elevations, "cw")

        # Compute relative azimuth based on spinning direction
        rel_azimuth = self._relative_angle(proj.fov_horiz_start_rad, azimuths, proj.spinning_direction)

        return torch.stack([rel_elevation, rel_azimuth], dim=-1)

    def _relative_angle(self, start_rad: float, angles: Tensor, direction: str) -> Tensor:
        """Compute relative angle from start in given direction.

        Args:
            start_rad: Starting angle in radians
            angles: (N,) angles in radians
            direction: 'cw' for clockwise, 'ccw' for counterclockwise

        Returns:
            relative_angles: (N,) relative angles in radians (always positive)
        """
        if direction == "cw":
            # Clockwise: positive when angle < start (going backwards)
            rel = start_rad - angles
        else:  # ccw
            # Counterclockwise: positive when angle > start (going forwards)
            rel = angles - start_rad

        # Normalize to [0, 2π)
        rel = rel % (2 * torch.pi)
        return rel

    def _valid_relative_sensor_angles(self, relative_sensor_angles: Tensor) -> Tensor:
        """Check if relative sensor angles are within the FOV of the sensor.

        Relative angle inputs should be computed with _relative_sensor_angles
        to include epsilon corrections.

        Args:
            relative_sensor_angles: (N, 2) [relative_elevation, relative_azimuth] in radians

        Returns:
            valid: (N,) bool - True if angles are within FOV
        """
        proj = self.projection
        # Account for accumulated numerical errors via some epsilon in FOV check
        # (using 2x eps as 1x eps is "inherited" from the start of the FOV in the relative angles,
        # so effectively this checks 1x eps on the end of the FOV)
        return torch.logical_and(
            relative_sensor_angles[:, 0] <= proj.fov_vert_span_rad + 2 * self._fov_eps_rad,
            relative_sensor_angles[:, 1] <= proj.fov_horiz_span_rad + 2 * self._fov_eps_rad,
        )

    def _valid_sensor_angles(self, sensor_angles: Tensor) -> Tensor:
        """Check if sensor angles are within the FOV of the sensor.

        Args:
            sensor_angles: (N, 2) [elevation_rad, azimuth_rad]

        Returns:
            valid: (N,) bool - True if angles are within FOV
        """
        relative_sensor_angles = self._relative_sensor_angles(sensor_angles)
        return self._valid_relative_sensor_angles(relative_sensor_angles)

    def forward(self, *args, **kwargs):
        """Forward pass - not typically used directly.

        LiDAR models are typically used via their projection methods rather than
        as part of a forward pass. Override in subclasses if needed.
        """
        raise NotImplementedError(
            "LidarModel.forward() is not implemented. "
            "Use projection methods like elements_to_world_rays_shutter_pose directly."
        )

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def n_rows(self) -> int:
        """Get number of rows (vertical channels)."""
        return self.projection.n_rows

    @property
    def n_columns(self) -> int:
        """Get number of columns (horizontal samples)."""
        return self.projection.n_columns

    @property
    def n_elements(self) -> int:
        """Get total number of elements (n_rows * n_columns)."""
        return self.n_rows * self.n_columns

    @property
    def fov_vert(self) -> tuple[float, float]:
        """Get vertical FOV as (start_rad, span_rad)."""
        return (self.projection.fov_vert_start_rad, self.projection.fov_vert_span_rad)

    @property
    def fov_horiz(self) -> tuple[float, float]:
        """Get horizontal FOV as (start_rad, span_rad)."""
        return (self.projection.fov_horiz_start_rad, self.projection.fov_horiz_span_rad)

    @property
    def spinning_frequency_hz(self) -> float:
        """Get spinning frequency in Hz."""
        return self.projection.spinning_frequency_hz

    @property
    def spinning_direction(self) -> str:
        """Get spinning direction ('cw' or 'ccw')."""
        return self.projection.spinning_direction

    # ============================================================================
    # Public Validation Methods
    # ============================================================================

    def valid_sensor_angles(self, sensor_angles: TensorLike) -> Tensor:
        """Check if sensor angles are within the FOV of the sensor.

        Args:
            sensor_angles: (N, 2) [elevation_rad, azimuth_rad] (Tensor or numpy array)

        Returns:
            valid: (N,) bool - True if angles are within FOV
        """
        # Convert numpy to torch if needed
        if isinstance(sensor_angles, np.ndarray):
            sensor_angles = torch.from_numpy(sensor_angles).to(dtype=torch.float32)
        return self._valid_sensor_angles(sensor_angles)


__all__ = [
    "LidarModel",
]
