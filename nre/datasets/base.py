# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Any, Generator, Optional, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data

from nre.datasets.tracks import CuboidTracks
from nre.utils.types import (
    AABB3D,
    BoundingBox,
    CameraFrustum,
    PointCloud,
    PointCloudColorType,
    RigTrajectories,
    TrackPointCloud,
)


class BaseDataset(torch.utils.data.Dataset, ABC, Sized):
    """Wraps the logic (such as train/val split) required to sample frames from an underlying BaseDataSource."""

    @abstractmethod
    def get_datasource(self) -> BaseDataSource: ...

    @abstractmethod
    def __len__(self) -> int:
        """Returns number of batches per epoch"""
        ...

    @abstractmethod
    def __getitem__(self, batch_idx) -> Any: ...

    @abstractmethod
    def get_max_num_rays_per_train_sample(self) -> int: ...

    def update_epoch(self, epoch: int, system, **kwargs) -> None: ...


class BaseDataSource(ABC):
    """
    Base class for all NRE data source variants. This class parses and loads the data from
    files and exposes methods relevant to the entire reconstruction, independently of train/val
    batching logic (covered by BaseDataset).
    """

    # Axis aligned bounding box in the NRE coordinate system (true to scale, centered at 0)
    aabb: AABB3D

    @abstractmethod
    def get_camera_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True) or logical (unique_sensors=False) camera sensor ids"""
        ...

    @abstractmethod
    def get_lidar_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True) or logical (unique_sensors=False) lidar sensor ids"""
        ...

    @abstractmethod
    def get_n_frames_per_camera(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of total frame numbers per unique (unique_sensors=True) or logical (unique_sensors=False) camera sensor instance"""
        ...

    @abstractmethod
    def get_n_frames_per_lidar(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of total frame numbers per unique (unique_sensors=True) or logical (unique_sensors=False) lidar sensor instance"""
        ...

    @abstractmethod
    def get_offset(self) -> npt.NDArray[np.float32]:
        """Returns the offset required for nrend export and rendering"""
        ...

    @abstractmethod
    def get_point_clouds(
        self,
        device: torch.device,
        lidar_ids: Optional[list[str]] = None,
        camera_ids: Optional[list[str]] = None,
        valid_points_only: bool = True,
        non_dynamic_points_only: bool = True,
        color_type: PointCloudColorType = None,
        step_frame: int = 1,
        visualize: bool = False,
        force: bool = True,
    ) -> Generator[PointCloud, None, None]:
        """Returns a generator for all point-clouds available for point-cloud sensor (lidar / camera), transformed into NGP frame.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Can be parameterized to only return valid (default), non-dynamic (default),
        and colored points colorized by one of the following strategies: "camera-rgb" (rgb scene colors),
        "semantics" (semantic colors obtain from shard meta data).
        """
        ...

    @abstractmethod
    def get_cuboid_tracks(
        self, dynamic_only: bool = False, world_frame: bool = False, include_generated: bool = False
    ) -> CuboidTracks:
        """Returns CuboidTracks: CuboidTracks

        Args:
            dynamic_only (bool, optional): if set return only tracks associated with dynamic objects. Defaults to False.
            world_frame (bool, optional): if set return tracks in world frame instead of NRE frame. Defaults to False.
            include_generated (bool, optional): if set return generated tracks. Defaults to False.

        Returns:
            CuboidTracks: CuboidTracks object
        """
        ...

    @abstractmethod
    def get_track_point_clouds(
        self,
        cuboid_tracks: CuboidTracks,
        cuboid_dim_scale_factor: float = 1.0,
        lidar_ids: Optional[list[str]] = None,
        camera_ids: Optional[list[str]] = None,
        return_color: bool = False,
        step_frame: int = 1,
        keep_all_track_poses: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> Generator[TrackPointCloud, None, None]:
        """Returns a generator for all object point-clouds available for point-cloud sensor (lidar / camera), transformed into NRE frame.
        The returned value is a (track_id, PointCloud) named tuple.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        The provided cuboid tracks should be in the original world coordinates.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Default point-cloud sensor: *first* logical lidar
        """
        ...

    @abstractmethod
    def get_camera_frusta(
        self,
        camera_id: Optional[str] = None,
        near_plane_depth: float = 0.1,
        far_plane_depth: float = 150.0,
        step_frame: int = 1,
    ) -> Generator[tuple[CameraFrustum, int], None, None]:
        """Returns a generator for all camera frusta for a given camera sensor, transformed into NGP frame.

        Camera sensor are specified by by either logical or unique sensor IDs.

        A single camera sensor needs to be specified - defaults to first camera sensor if not specified."""
        ...

    @abstractmethod
    def get_semantic_colormap(self, camera_semantics: bool, lidar_semantics: bool) -> Optional[npt.NDArray]:
        """Returns the semantic colormap for requested sensor types (camera and/or lidar), if available"""


@runtime_checkable
class RigTrajectoriesProvider(Protocol):
    """Marks a type to provide RigTrajectories via the get_rig_trajectories method."""

    def get_rig_trajectories(self) -> RigTrajectories:
        """Returns all rig trajectories associated with the instance."""
        ...
