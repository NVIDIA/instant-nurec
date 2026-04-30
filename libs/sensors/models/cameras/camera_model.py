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
Base camera model class for all projection types.

This module provides the base camera model class for all projection types.
"""

from abc import abstractmethod

import torch
import torch.nn as nn

from torch import Tensor

from libs.geometry.kernels.pose import se3pose_to_matrix, se3pose_transform_direction
from libs.geometry.kernels.quaternion import quat_slerp
from libs.sensors.kernels.cameras import (
    CameraProjection,
    # External distortion types
    ExternalDistortion,
    # Enums and related types
    ShutterType,
    # Kernel functions
    camera_rays_to_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_shutter_pose,
    image_points_to_world_rays_static_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
)
from libs.sensors.kernels.common.pose import DynamicPose, Pose
from libs.sensors.models.common.return_types import (
    ImagePointsReturn,
    PixelsReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldRaysReturn,
)

# Import common helpers
from libs.sensors.models.common.utils import (
    filter_by_validity,
    poses_to_matrix,
    valid_flags_to_indices,
)


class CameraModel(nn.Module):
    """Base camera model class for all projection types.

    This is an abstract base class. Use derived classes (OpenCVPinholeCameraModel,
    OpenCVFisheyeCameraModel, FThetaCameraModel) which contain properly-typed projection members.

    The base class contains common sensor properties shared across all camera types.
    Derived classes add projection-specific parameters.

    Wraps Layer 0 projection parameters as nn.Module state for gradient-based optimization.
    Calls Layer 0 kernel functions for projection operations.

    Note on Timestamps:
        - Static pose functions accept pose objects which may contain an optional `timestamp_us` field
        - Dynamic pose-based functions (shutter_pose, mean_pose) use normalized time [0, 1]

    Example usage:
        # Create camera model with Layer 0 projection
        projection = OpenCVPinholeProjection.from_components(...)
        camera = OpenCVPinholeCameraModel(
            projection=projection,
            external_distortion=NoExternalDistortion(),
            resolution=(1920, 1080),
            shutter_type=ShutterType.GLOBAL,
        )

        # Access projection parameters with proper typing
        fx, fy = camera.projection.focal_length  # Type: Tensor (2,)

        # Project world points using static pose
        result = camera.world_points_to_image_points_static_pose(
            world_points, pose, return_timestamps=True
        )

    Attributes:
        external_distortion: Layer 0 external distortion parameters (NoExternal/Windshield)
        resolution: (width, height) in pixels
        shutter_type: Rolling or global shutter behavior
    """

    external_distortion: ExternalDistortion
    resolution: tuple[int, int]
    shutter_type: ShutterType

    def __init__(
        self,
        external_distortion: ExternalDistortion,
        resolution: tuple[int, int],
        shutter_type: ShutterType,
    ):
        """Initialize base camera model with common sensor properties.

        Args:
            external_distortion: Layer 0 external distortion parameters
            resolution: (width, height) in pixels
            shutter_type: Rolling or global shutter behavior
        """
        super().__init__()
        self.external_distortion = external_distortion
        self.resolution = resolution
        self.shutter_type = shutter_type

    @property
    @abstractmethod
    def projection(self) -> CameraProjection:
        """Get the camera projection parameters. Must be implemented by subclasses."""
        raise NotImplementedError

    # ============================================================================
    # Private Helpers for Building Return Types
    # ============================================================================

    def _build_image_points_return(
        self,
        image_points: Tensor,
        valid_flags: Tensor | None,
        timestamps_us_out: Tensor | None,
        poses_trans: Tensor | None,
        poses_rot: Tensor | None,
        return_T_world_sensors: bool,
        return_valid_flag: bool,
        return_valid_indices: bool,
        return_timestamps: bool,
        return_all_projections: bool,
    ) -> WorldPointsToImagePointsReturn:
        """Build WorldPointsToImagePointsReturn with common post-processing logic."""
        # Convert translations and rotations to 4x4 matrices if requested
        T_world_sensors = poses_to_matrix(poses_trans, poses_rot) if return_T_world_sensors else None

        # Compute valid indices if requested
        valid_indices = valid_flags_to_indices(valid_flags) if return_valid_indices else None

        # Filter to valid points only if not returning all projections
        if not return_all_projections and valid_flags is not None:
            image_points = filter_by_validity(image_points, valid_flags, return_all_projections)
            T_world_sensors = (
                filter_by_validity(T_world_sensors, valid_flags, return_all_projections)
                if T_world_sensors is not None
                else None
            )
            timestamps_us_out = (
                filter_by_validity(timestamps_us_out, valid_flags, return_all_projections)
                if timestamps_us_out is not None
                else None
            )

        return WorldPointsToImagePointsReturn(
            image_points=image_points,
            T_world_sensors=T_world_sensors,
            valid_flag=valid_flags if return_valid_flag else None,
            valid_indices=valid_indices,
            timestamps_us=timestamps_us_out if return_timestamps else None,
        )

    # ============================================================================
    # World Points to Image Points
    # ============================================================================

    def world_points_to_image_points_static_pose(
        self,
        world_points: Tensor,
        pose: Pose,
        timestamp_us: int | None = None,
        return_T_world_sensors: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        """Project world points using fixed sensor pose.

        Args:
            world_points: (N, 3) world coordinates
            pose: Static world → sensor pose (Pose object with translation and rotation)
            timestamp_us: Timestamp for the static pose
            return_T_world_sensors: If True, return per-point poses as (N, 4, 4) matrices
            return_valid_flag: If True, return validity mask
            return_valid_indices: If True, return indices of valid projections
            return_timestamps: If True, return timestamps
            return_all_projections: If True, return all projections (even invalid ones)

        Returns:
            WorldPointsToImagePointsReturn with projected image points and optional auxiliary data
        """
        # Create a dynamic pose from the static pose for the kernel
        dynamic_pose = DynamicPose.from_static_pose(pose)

        ts = timestamp_us if timestamp_us is not None else 0
        image_points, valid_flags, timestamps_us_out, poses_trans, poses_rot = project_world_points_mean_pose(
            world_points=world_points,
            projection=self.projection,
            external_distortion=self.external_distortion,
            dynamic_pose=dynamic_pose,
            resolution=self.resolution,
            start_timestamp_us=ts,
            end_timestamp_us=ts,
            return_valid_flags=True,
            return_timestamps=return_timestamps,
            return_poses=return_T_world_sensors,
        )

        return self._build_image_points_return(
            image_points=image_points,
            valid_flags=valid_flags,
            timestamps_us_out=timestamps_us_out,
            poses_trans=poses_trans,
            poses_rot=poses_rot,
            return_T_world_sensors=return_T_world_sensors,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

    def world_points_to_image_points_mean_pose(
        self,
        world_points: Tensor,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_world_sensors: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        """Project world points using mean pose (not compensating for sensor motion).

        Args:
            world_points: (N, 3) world coordinates
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            return_T_world_sensors: If True, return per-point poses as (N, 4, 4) matrices
            return_valid_flag: If True, return validity mask
            return_valid_indices: If True, return indices of valid projections
            return_timestamps: If True, return timestamps
            return_all_projections: If True, return all projections (even invalid ones)

        Returns:
            WorldPointsToImagePointsReturn with projected image points
        """
        image_points, valid_flags, timestamps_us_out, poses_trans, poses_rot = project_world_points_mean_pose(
            world_points=world_points,
            projection=self.projection,
            external_distortion=self.external_distortion,
            dynamic_pose=dynamic_pose,
            resolution=self.resolution,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_valid_flags=True,
            return_timestamps=return_timestamps,
            return_poses=return_T_world_sensors,
        )

        return self._build_image_points_return(
            image_points=image_points,
            valid_flags=valid_flags,
            timestamps_us_out=timestamps_us_out,
            poses_trans=poses_trans,
            poses_rot=poses_rot,
            return_T_world_sensors=return_T_world_sensors,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

    def world_points_to_image_points_shutter_pose(
        self,
        world_points: Tensor,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        max_iterations: int = 10,
        stop_mean_error_px: float = 0.001,
        stop_delta_mean_error_px: float = 0.00001,
        return_T_world_sensors: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        """Project world points using rolling-shutter compensation.

        Args:
            world_points: (N, 3) world coordinates
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            max_iterations: Max iterations for convergence
            stop_mean_error_px: Stopping criterion for mean error
            stop_delta_mean_error_px: Stopping criterion for error change
            return_T_world_sensors: If True, return per-point poses as (N, 4, 4) matrices
            return_valid_flag: If True, return validity mask
            return_valid_indices: If True, return indices of valid projections
            return_timestamps: If True, return timestamps
            return_all_projections: If True, return all projections (even invalid ones)

        Returns:
            WorldPointsToImagePointsReturn with projected image points
        """
        image_points, valid_flags, timestamps_us_out, poses_trans, poses_rot = project_world_points_shutter_pose(
            world_points=world_points,
            projection=self.projection,
            external_distortion=self.external_distortion,
            resolution=self.resolution,
            shutter_type=self.shutter_type,
            dynamic_pose=dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            max_iterations=max_iterations,
            stop_mean_error_px=stop_mean_error_px,
            stop_delta_mean_error_px=stop_delta_mean_error_px,
            return_valid_flags=True,
            return_timestamps=return_timestamps,
            return_poses=return_T_world_sensors,
        )

        return self._build_image_points_return(
            image_points=image_points,
            valid_flags=valid_flags,
            timestamps_us_out=timestamps_us_out,
            poses_trans=poses_trans,
            poses_rot=poses_rot,
            return_T_world_sensors=return_T_world_sensors,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

    # ============================================================================
    # World Points to Pixels
    # ============================================================================

    def world_points_to_pixels_static_pose(
        self,
        world_points: Tensor,
        pose: Pose,
        timestamp_us: int | None = None,
        return_T_world_sensors: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        """Project world points to pixel indices using fixed sensor pose.

        Args:
            world_points: (N, 3) world coordinates
            pose: Static world → sensor pose
            timestamp_us: Timestamp for the static pose
            return_T_world_sensors: If True, return per-point poses as (N, 4, 4) matrices
            return_valid_flag: If True, return validity mask
            return_valid_indices: If True, return indices of valid projections
            return_timestamps: If True, return timestamps
            return_all_projections: If True, return all projections (even invalid ones)

        Returns:
            WorldPointsToPixelsReturn with pixel indices
        """
        result = self.world_points_to_image_points_static_pose(
            world_points=world_points,
            pose=pose,
            timestamp_us=timestamp_us,
            return_T_world_sensors=return_T_world_sensors,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

        # Convert image points to pixel indices
        pixels = self.image_points_to_pixels(result.image_points)

        return WorldPointsToPixelsReturn(
            pixels=pixels,
            T_world_sensors=result.T_world_sensors,
            valid_flag=result.valid_flag,
            valid_indices=result.valid_indices,
            timestamps_us=result.timestamps_us,
        )

    def world_points_to_pixels_mean_pose(
        self,
        world_points: Tensor,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_world_sensors: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        """Project world points to pixel indices using mean pose.

        Args:
            world_points: (N, 3) world coordinates
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            return_T_world_sensors: If True, return per-point poses as (N, 4, 4) matrices
            return_valid_flag: If True, return validity mask
            return_valid_indices: If True, return indices of valid projections
            return_timestamps: If True, return timestamps
            return_all_projections: If True, return all projections (even invalid ones)

        Returns:
            WorldPointsToPixelsReturn with pixel indices
        """
        result = self.world_points_to_image_points_mean_pose(
            world_points=world_points,
            dynamic_pose=dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_world_sensors=return_T_world_sensors,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

        # Convert image points to pixel indices
        pixels = self.image_points_to_pixels(result.image_points)

        return WorldPointsToPixelsReturn(
            pixels=pixels,
            T_world_sensors=result.T_world_sensors,
            valid_flag=result.valid_flag,
            valid_indices=result.valid_indices,
            timestamps_us=result.timestamps_us,
        )

    def world_points_to_pixels_shutter_pose(
        self,
        world_points: Tensor,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        max_iterations: int = 10,
        stop_mean_error_px: float = 0.001,
        stop_delta_mean_error_px: float = 0.00001,
        return_T_world_sensors: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        """Project world points to pixel indices using rolling-shutter compensation.

        Args:
            world_points: (N, 3) world coordinates
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            max_iterations: Max iterations for convergence
            stop_mean_error_px: Stopping criterion for mean error
            stop_delta_mean_error_px: Stopping criterion for error change
            return_T_world_sensors: If True, return per-point poses as (N, 4, 4) matrices
            return_valid_flag: If True, return validity mask
            return_valid_indices: If True, return indices of valid projections
            return_timestamps: If True, return timestamps
            return_all_projections: If True, return all projections (even invalid ones)

        Returns:
            WorldPointsToPixelsReturn with pixel indices
        """
        result = self.world_points_to_image_points_shutter_pose(
            world_points=world_points,
            dynamic_pose=dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            max_iterations=max_iterations,
            stop_mean_error_px=stop_mean_error_px,
            stop_delta_mean_error_px=stop_delta_mean_error_px,
            return_T_world_sensors=return_T_world_sensors,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )

        # Convert image points to pixel indices
        pixels = self.image_points_to_pixels(result.image_points)

        return WorldPointsToPixelsReturn(
            pixels=pixels,
            T_world_sensors=result.T_world_sensors,
            valid_flag=result.valid_flag,
            valid_indices=result.valid_indices,
            timestamps_us=result.timestamps_us,
        )

    # ============================================================================
    # Image Points to World Rays
    # ============================================================================

    def image_points_to_world_rays_static_pose(
        self,
        image_points: Tensor,
        pose: Pose,
        timestamp_us: int | None = None,
        camera_rays: Tensor | None = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project image points to world rays using fixed sensor pose.

        Args:
            image_points: (N, 2) image coordinates
            pose: Static sensor → world pose
            timestamp_us: Timestamp for the static pose
            camera_rays: (N, 3) optional pre-computed camera rays for reuse
            return_T_sensor_worlds: If True, return per-ray poses as (N, 4, 4) matrices
            return_timestamps: If True, return timestamps

        Returns:
            WorldRaysReturn with world rays
        """
        # If camera_rays provided, use them directly instead of unprojecting
        if camera_rays is not None:
            # Transform camera rays to world rays using geometry kernels
            N = image_points.shape[0]
            device = image_points.device

            # Transform ray directions using SE3 pose (GPU-accelerated, differentiable)
            world_positions = pose.translation.unsqueeze(0).expand(N, 3)
            world_directions = se3pose_transform_direction(
                pose.translation.unsqueeze(0).expand(N, 3),
                pose.rotation.unsqueeze(0).expand(N, 4),
                camera_rays,
            )

            world_rays = torch.cat([world_positions, world_directions], dim=-1)

            T_sensor_worlds = None
            if return_T_sensor_worlds:
                T_sensor_world = se3pose_to_matrix(
                    pose.translation.unsqueeze(0),
                    pose.rotation.unsqueeze(0),
                ).squeeze(0)  # (4, 4)
                T_sensor_worlds = T_sensor_world.unsqueeze(0).expand(N, 4, 4).contiguous()

            timestamps_us_out = None
            if return_timestamps:
                assert timestamp_us is not None, "timestamp_us must be provided when return_timestamps=True"
                timestamps_us_out = torch.full((N,), timestamp_us, device=device, dtype=torch.int64)

            return WorldRaysReturn(
                world_rays=world_rays,
                T_sensor_worlds=T_sensor_worlds,
                timestamps_us=timestamps_us_out,
            )

        # Use kernel for standard path
        world_rays, timestamps_us_out, poses_trans, poses_rot = image_points_to_world_rays_static_pose(
            image_points=image_points,
            projection=self.projection,
            external_distortion=self.external_distortion,
            static_pose=pose,
            timestamp_us=timestamp_us,
            return_timestamps=return_timestamps,
            return_poses=return_T_sensor_worlds,
        )

        # Convert translations and rotations to 4x4 matrices if requested
        T_sensor_worlds = poses_to_matrix(poses_trans, poses_rot) if return_T_sensor_worlds else None

        return WorldRaysReturn(
            world_rays=world_rays,
            T_sensor_worlds=T_sensor_worlds,
            timestamps_us=timestamps_us_out if return_timestamps else None,
        )

    def image_points_to_world_rays_mean_pose(
        self,
        image_points: Tensor,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        camera_rays: Tensor | None = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project using mean pose (not compensating for sensor motion).

        Args:
            image_points: (N, 2) image coordinates
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            camera_rays: (N, 3) optional pre-computed camera rays for reuse
            return_T_sensor_worlds: If True, return per-ray poses as (N, 4, 4) matrices
            return_timestamps: If True, return timestamps

        Returns:
            WorldRaysReturn with world rays
        """
        # If camera_rays provided, compute mean pose and transform manually
        if camera_rays is not None:
            N = image_points.shape[0]
            device = image_points.device

            # Interpolate to get mean pose at t=0.5
            pose0 = dynamic_pose.start_pose
            pose1 = dynamic_pose.end_pose

            # Linear interpolation for translation
            mean_trans = 0.5 * pose0.translation + 0.5 * pose1.translation

            # SLERP for rotation (matching kernel behavior)
            mean_rot = quat_slerp(
                pose0.rotation.unsqueeze(0),
                pose1.rotation.unsqueeze(0),
                0.5,
            ).squeeze(0)

            mean_pose = Pose(translation=mean_trans, rotation=mean_rot)

            # Compute timestamp
            timestamp_us = None
            if return_timestamps:
                assert start_timestamp_us is not None and end_timestamp_us is not None
                timestamp_us = (start_timestamp_us + end_timestamp_us) // 2

            return self.image_points_to_world_rays_static_pose(
                image_points=image_points,
                pose=mean_pose,
                timestamp_us=timestamp_us,
                camera_rays=camera_rays,
                return_T_sensor_worlds=return_T_sensor_worlds,
                return_timestamps=return_timestamps,
            )

        # For mean pose, use the shutter pose kernel with global shutter type
        # which effectively uses the mean pose
        world_rays, timestamps_us_out, poses_trans, poses_rot = image_points_to_world_rays_shutter_pose(
            image_points=image_points,
            projection=self.projection,
            external_distortion=self.external_distortion,
            resolution=self.resolution,
            shutter_type=ShutterType.GLOBAL,
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
            timestamps_us=timestamps_us_out if return_timestamps else None,
        )

    def image_points_to_world_rays_shutter_pose(
        self,
        image_points: Tensor,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        camera_rays: Tensor | None = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project using rolling-shutter compensation.

        Args:
            image_points: (N, 2) image coordinates
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            camera_rays: (N, 3) optional pre-computed camera rays for reuse
            return_T_sensor_worlds: If True, return per-ray poses as (N, 4, 4) matrices
            return_timestamps: If True, return timestamps

        Returns:
            WorldRaysReturn with world rays
        """
        # Note: When camera_rays is provided, we still need to call the kernel
        # to get the per-pixel interpolated poses. The kernel handles the camera_rays
        # reuse optimization internally if we pass them.
        # For now, we don't support camera_rays reuse with shutter pose in the kernel,
        # so we just use the standard path.

        world_rays, timestamps_us_out, poses_trans, poses_rot = image_points_to_world_rays_shutter_pose(
            image_points=image_points,
            projection=self.projection,
            external_distortion=self.external_distortion,
            resolution=self.resolution,
            shutter_type=self.shutter_type,
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
            timestamps_us=timestamps_us_out if return_timestamps else None,
        )

    # ============================================================================
    # Pixels to World Rays
    # ============================================================================

    def pixels_to_world_rays_static_pose(
        self,
        pixel_idxs: Tensor,
        pose: Pose,
        timestamp_us: int | None = None,
        camera_rays: Tensor | None = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project pixel indices to world rays using fixed sensor pose.

        Args:
            pixel_idxs: (N, 2) int pixel indices
            pose: Static sensor → world pose
            timestamp_us: Timestamp for the static pose
            camera_rays: (N, 3) optional pre-computed camera rays for reuse
            return_T_sensor_worlds: If True, return per-ray poses as (N, 4, 4) matrices
            return_timestamps: If True, return timestamps

        Returns:
            WorldRaysReturn with world rays
        """
        image_points = self.pixels_to_image_points(pixel_idxs)
        return self.image_points_to_world_rays_static_pose(
            image_points=image_points,
            pose=pose,
            timestamp_us=timestamp_us,
            camera_rays=camera_rays,
            return_T_sensor_worlds=return_T_sensor_worlds,
            return_timestamps=return_timestamps,
        )

    def pixels_to_world_rays_mean_pose(
        self,
        pixel_idxs: Tensor,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        camera_rays: Tensor | None = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project pixel indices to world rays using mean pose.

        Args:
            pixel_idxs: (N, 2) int pixel indices
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            camera_rays: (N, 3) optional pre-computed camera rays for reuse
            return_T_sensor_worlds: If True, return per-ray poses as (N, 4, 4) matrices
            return_timestamps: If True, return timestamps

        Returns:
            WorldRaysReturn with world rays
        """
        image_points = self.pixels_to_image_points(pixel_idxs)
        return self.image_points_to_world_rays_mean_pose(
            image_points=image_points,
            dynamic_pose=dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            camera_rays=camera_rays,
            return_T_sensor_worlds=return_T_sensor_worlds,
            return_timestamps=return_timestamps,
        )

    def pixels_to_world_rays_shutter_pose(
        self,
        pixel_idxs: Tensor,
        dynamic_pose: DynamicPose,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        camera_rays: Tensor | None = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project pixel indices to world rays using rolling-shutter compensation.

        Args:
            pixel_idxs: (N, 2) int pixel indices
            dynamic_pose: Time-varying dynamic pose
            start_timestamp_us: Start timestamp for timestamp computation
            end_timestamp_us: End timestamp for timestamp computation
            camera_rays: (N, 3) optional pre-computed camera rays for reuse
            return_T_sensor_worlds: If True, return per-ray poses as (N, 4, 4) matrices
            return_timestamps: If True, return timestamps

        Returns:
            WorldRaysReturn with world rays
        """
        image_points = self.pixels_to_image_points(pixel_idxs)
        return self.image_points_to_world_rays_shutter_pose(
            image_points=image_points,
            dynamic_pose=dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            camera_rays=camera_rays,
            return_T_sensor_worlds=return_T_sensor_worlds,
            return_timestamps=return_timestamps,
        )

    # ============================================================================
    # Camera Ray / Image Point Conversions
    # ============================================================================

    def camera_rays_to_image_points(
        self,
        camera_rays: Tensor,
        return_jacobians: bool = False,
    ) -> ImagePointsReturn:
        """Convert camera rays to image points.

        Args:
            camera_rays: (N, 3) normalized rays in camera frame
            return_jacobians: If True, compute and return Jacobians (N, 2, 3)

        Returns:
            ImagePointsReturn with image points, validity mask, and optional Jacobians
        """
        # Track if we need to restore requires_grad state
        initial_requires_grad = camera_rays.requires_grad

        if return_jacobians:
            camera_rays = camera_rays.clone()
            camera_rays.requires_grad = True

        image_points, valid_flags = camera_rays_to_image_points(
            camera_rays=camera_rays,
            projection=self.projection,
            external_distortion=self.external_distortion,
        )

        jacobians = None
        if return_jacobians:
            # Compute Jacobians via autograd (matches ncore's approach)
            N = camera_rays.shape[0]
            jacobians = torch.empty((N, 2, 3), dtype=camera_rays.dtype, device=camera_rays.device)

            # Gradient for x-coordinate of image points
            initial_gradient = torch.ones(N, dtype=camera_rays.dtype, device=camera_rays.device)
            image_points[:, 0].backward(gradient=initial_gradient, retain_graph=True)

            assert camera_rays.grad is not None
            jacobians[:, 0] = camera_rays.grad.clone()

            camera_rays.grad.zero_()

            # Gradient for y-coordinate of image points
            image_points[:, 1].backward(gradient=initial_gradient)
            jacobians[:, 1] = camera_rays.grad.clone()

            # Cleanup
            camera_rays.grad.zero_()
            if not initial_requires_grad:
                camera_rays.requires_grad = False

        return ImagePointsReturn(image_points=image_points, valid_flag=valid_flags, jacobians=jacobians)

    def camera_rays_to_pixels(
        self,
        camera_rays: Tensor,
    ) -> PixelsReturn:
        """Convert camera rays to pixel indices.

        For each camera ray, computes the corresponding pixel index and a valid flag.

        Args:
            camera_rays: (N, 3) normalized rays in camera frame

        Returns:
            PixelsReturn with pixel indices and validity mask
        """
        image_points_result = self.camera_rays_to_image_points(camera_rays)

        return PixelsReturn(
            pixels=self.image_points_to_pixels(image_points_result.image_points),
            valid_flag=image_points_result.valid_flag,
        )

    def image_points_to_camera_rays(
        self,
        image_points: Tensor,
    ) -> Tensor:
        """Convert image points to camera rays.

        Args:
            image_points: (N, 2) image coordinates

        Returns:
            camera_rays: (N, 3) normalized directions in camera frame
        """
        return image_points_to_camera_rays(
            image_points=image_points,
            projection=self.projection,
            external_distortion=self.external_distortion,
        )

    def pixels_to_camera_rays(
        self,
        pixel_idxs: Tensor,
    ) -> Tensor:
        """Convert pixel indices to camera rays.

        Args:
            pixel_idxs: (N, 2) int pixel indices

        Returns:
            camera_rays: (N, 3) normalized directions in camera frame
        """
        image_points = self.pixels_to_image_points(pixel_idxs)
        return self.image_points_to_camera_rays(image_points)

    def pixels_to_image_points(
        self,
        pixel_idxs: Tensor,
    ) -> Tensor:
        """Convert pixel indices to continuous image point coordinates.

        Args:
            pixel_idxs: (N, 2) int pixel indices [x, y]

        Returns:
            image_points: (N, 2) float image coordinates (pixel center)
        """
        return pixel_idxs.float() + 0.5

    def image_points_to_pixels(
        self,
        image_points: Tensor,
    ) -> Tensor:
        """Convert continuous image points to pixel indices.

        Args:
            image_points: (N, 2) float image coordinates

        Returns:
            pixels: (N, 2) int pixel indices
        """
        return image_points.floor().to(torch.int32)

    # ============================================================================
    # Utilities
    # ============================================================================

    def image_points_relative_frame_times(
        self,
        image_points: Tensor,
    ) -> Tensor:
        """Get relative frame-times [0,1] based on image coordinates and rolling shutter.

        This matches ncore's implementation using floor/ceil and (resolution - 1) normalization.

        Args:
            image_points: (N, 2) image coordinates

        Returns:
            relative_times: (N,) float in [0, 1]
        """
        width, height = self.resolution

        if self.shutter_type == ShutterType.GLOBAL:
            return torch.zeros(image_points.shape[0], device=image_points.device, dtype=image_points.dtype)
        elif self.shutter_type == ShutterType.ROLLING_TOP_TO_BOTTOM:
            return torch.floor(image_points[:, 1]) / (height - 1)
        elif self.shutter_type == ShutterType.ROLLING_BOTTOM_TO_TOP:
            return (height - torch.ceil(image_points[:, 1])) / (height - 1)
        elif self.shutter_type == ShutterType.ROLLING_LEFT_TO_RIGHT:
            return torch.floor(image_points[:, 0]) / (width - 1)
        elif self.shutter_type == ShutterType.ROLLING_RIGHT_TO_LEFT:
            return (width - torch.ceil(image_points[:, 0])) / (width - 1)
        else:
            raise ValueError(f"Unknown shutter type: {self.shutter_type}")

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "CameraModel":
        """Apply image domain transformation to camera parameters.

        Creates a new camera model with transformed intrinsic parameters.
        Used when scaling/cropping images to maintain correct projections.

        Args:
            image_domain_scale: Isotropic (float) or anisotropic (tuple) scaling factor
            image_domain_offset: Offset in the scaled image domain (for cropping)
            new_resolution: Optional explicit new resolution (if None, computed from scale)

        Returns:
            New CameraModel with transformed parameters

        Note:
            This method must be implemented by subclasses since the projection type
            is model-specific.
        """
        raise NotImplementedError(
            "CameraModel.transform() must be implemented by subclasses. "
            "Use the concrete model class (OpenCVPinholeCameraModel, etc.)."
        )

    def forward(self, method: str, *args, **kwargs):
        """Forward pass - dispatches to the specified projection method.

        This allows flexible use of the camera model in different contexts (e.g.,
        forward projection for rendering, ray casting for ray tracing).

        Args:
            method: Name of the method to call (e.g., "world_points_to_image_points_static_pose",
                    "image_points_to_world_rays_static_pose")
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Result of the called method

        Example:
            # Forward projection (3D -> 2D)
            result = camera("world_points_to_image_points_static_pose", world_points, pose)

            # Ray casting (2D -> rays)
            result = camera("image_points_to_world_rays_static_pose", image_points, pose)
        """
        if not hasattr(self, method):
            raise AttributeError(f"CameraModel has no method '{method}'")
        fn = getattr(self, method)
        if not callable(fn):
            raise TypeError(f"'{method}' is not a callable method")
        return fn(*args, **kwargs)


__all__ = [
    "CameraModel",
]
