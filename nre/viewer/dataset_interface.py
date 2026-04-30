# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from ncore.sensors import CameraModel
from nre.datasets.tracks import CuboidTracks
from nre.utils.types import FrameConversion, HalfClosedInterval, PointCloud, RigTrajectories


@dataclass(kw_only=True)
class CameraTrajectoryId:
    """
    Uniquely identifies a camera on a rig in a specific trajectory
    """

    unique_camera_id: str
    sequence_id: str
    camera_id: str

    def camera_sequence_str(self) -> str:
        return f"{self.camera_id}_{self.sequence_id}"


@dataclass
class CameraTrajectoryData(metaclass=ABCMeta):
    """
    Specifies the abstract interface to provide the viewer with data about a specific camera trajectory.

    Args:
        - trajectory_id: identifier of the trajectory
        - start_timestamp_us: start of the trajectory
        - end_timestamp_us: end of the trajectory
        - camera_model: intrinsics of the camera
        - camera_unique_idx: unique index of the camera
        - average_exposure_time_us: average exposure time of the camera
    """

    trajectory_id: CameraTrajectoryId
    time_range_us: HalfClosedInterval
    camera_model: CameraModel
    camera_unique_idx: int
    average_exposure_time_us: int

    @abstractmethod
    def get_poses_world(self, timestamps_us: torch.Tensor) -> torch.Tensor:
        """
        Get poses (possibly interpolated) at given timestamps

        Args:
            - timestamp_us: torch.int64 timestamps to interpolate at (n_timestamps, )

        Returns:
            (n_timestamps, 4, 4) homogenous Rt matrices
        """
        ...

    @abstractmethod
    def get_closest_frame_image(self, timestamp_us: int) -> tuple[int, torch.Tensor]:
        """
        For a given timestamp, get the nearest ground truth camera frame and its timestamp

        Args:
            - timestamp_us: query unix timestamp

        Returns:
            - nearest_timestamp: a unix timestamp of the nearest frame
            - nearest_frame: a (h, w, 3) ground truth frame
        """
        ...

    @abstractmethod
    def get_frame_timestamps(self) -> torch.Tensor:
        """
        Get all frame timestamps that have corresponding images.

        Returns:
            - frame_timestamps: (n_frames, 2) tensor of (start, end) timestamps for each frame.
              Returns empty tensor if no frames are available.
        """
        ...


class ViewerDatasetInterface(metaclass=ABCMeta):
    """
    An interface layer between concrete dataset types (NCOREDataset ...) and the viewer
    """

    def __init__(self) -> None:
        self._time_interval_us: HalfClosedInterval | None = None

    @abstractmethod
    def get_trajectory_ids(self) -> list[CameraTrajectoryId]:
        """
        List available trajectories.
        """
        ...

    @abstractmethod
    def get_trajectory_data(self, trajectory_id: CameraTrajectoryId) -> CameraTrajectoryData:
        """
        Obtain specific data for a given trajectory.

        Args:
            - trajectory_id: query trajectory ID
        """
        ...

    @abstractmethod
    def distance_nre_to_distance_world(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Converts distances in NRE coordinate system to world space
        """
        ...

    @abstractmethod
    def get_point_cloud(
        self, color_type: Optional[Literal["camera-rgb", "semantics"]] = None, step_frame: int = 10
    ) -> PointCloud | None:
        """
        Get a representative point cloud in NRE space, if available. Useful for debugging
        coordinate frame errors through overlaying with 3D Gaussians renders.

        TODO: this could be moved to CameraTrajectoryData and made time-dependent (so the
        point cloud follows the current camera viewpoint).

        Returns:
            - point_cloud: (n_points, 3) point cloud
        """
        ...

    def get_initial_trajectory_data(self) -> CameraTrajectoryData:
        """
        Get an arbitrary trajectory to initialize the viewer.
        """
        return self.get_trajectory_data(self.get_trajectory_ids()[0])

    def get_time_converter_us(self) -> HalfClosedInterval:
        """
        Get an object for converting from unix epoch timestamps to [0, scene_duration).

        Note: this is logically a part of ViewerDatasetInterface initialization but since it
        relies on abstract methods which may not be functional at the start of __init__ we
        defer it to after initialization via a lazy getter.
        """
        if self._time_interval_us is None:
            self._time_interval_us = HalfClosedInterval.union(
                self.get_trajectory_data(id).time_range_us for id in self.get_trajectory_ids()
            )

        return self._time_interval_us

    def get_rig_trajectories(self) -> RigTrajectories | None:
        """
        Get the rig trajectories from the data-module
        """
        return None

    @property
    def is_empty(self) -> bool:
        """
        Return if this is a dummy empty interface. This is typicall true in e.g. NRM when
        the dataset is not yet available (the model hasn't finished its inference).
        """
        return False

    @abstractmethod
    def get_cuboid_tracks(self) -> Optional[CuboidTracks]:
        """
        Get the cuboid tracks from the data-module
        """
        ...

    @property
    @abstractmethod
    def world_to_nre(self) -> FrameConversion:
        """
        Get the world to frame transform from the datamodule
        """
        ...

    @abstractmethod
    def supports_get_closest_frame_image(self) -> bool:
        """
        Check if the dataset supports getting the closest frame image
        """
        ...
