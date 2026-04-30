# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Self

import dataclasses_json
import numpy as np
import numpy.typing as npt
import torch

from nre.datasets.base import BaseDataSource
from nre.datasets.ncore import NCOREDataSource
from nre.datasets.tracks import CuboidTracks
from nre.utils.fields import field_numpy_array
from nre.utils.types import (
    AABB3D,
    BoundingBox,
    CameraFrustum,
    PointCloud,
    PointCloudColorType,
    RigTrajectories,
    TrackPointCloud,
)


@dataclass
class DataSourceSummary:
    """
    A class which abstracts the datasource properties required during model init - think of config auto-generated
    based on the dataset.
    Part of it is serializable and used when loading checkpoints for inference, some is
    available only with the actual datasource available, at training start.
    """

    @dataclass(kw_only=True)
    class State(dataclasses_json.DataClassJsonMixin):
        """
        The serializable part of DatasourceSummary.
        """

        n_frames_per_camera: npt.NDArray[np.int32] = field_numpy_array(np.int32, shape=(-1,))
        n_frames_per_lidar: npt.NDArray[np.int32] = field_numpy_array(
            np.int32,
            shape=(-1,),
            default_factory=list,  # for backwards compatibility. This needs to be a list, not an array, so the converter can work.
        )
        # We are keeping these as dicts with sequence_id keys, even though there is only 1 sequence, for backwards compatibility.
        sequence_tracks_all: Optional[dict[str, CuboidTracks]]
        sequence_tracks_dynamic: Optional[dict[str, CuboidTracks]]
        rig_trajectories: Optional[RigTrajectories]
        xform_matrices: npt.NDArray[np.float32] = field_numpy_array(np.float32, shape=(-1, 4))
        aabb_blb: npt.NDArray[np.float32] = field_numpy_array(np.float32, shape=(1, 3))
        aabb_trf: npt.NDArray[np.float32] = field_numpy_array(np.float32, shape=(1, 3))

        def __post_init__(self):
            assert self.n_frames_per_camera.ndim == 1
            assert self.n_frames_per_lidar.ndim == 1
            assert self.aabb_blb.shape == (1, 3)
            assert self.aabb_blb.shape == self.aabb_trf.shape

    datasource: Optional[BaseDataSource]
    state: State  # the serializable part

    @classmethod
    def from_datasource(cls, datasource: BaseDataSource) -> Self:
        """
        Used to create the summary at training start, when creating the entire system
        with access to the datasource.
        """

        if isinstance(datasource, NCOREDataSource):
            xform_matrices = np.zeros((0, 4), dtype=np.float32)
            rig_trajectories = datasource.get_rig_trajectories()
            sequence_tracks_all = {
                datasource.sequence_id: datasource.get_cuboid_tracks(
                    dynamic_only=False, world_frame=False, include_generated=True
                )
            }
            sequence_tracks_dynamic = {
                datasource.sequence_id: datasource.get_cuboid_tracks(dynamic_only=True, world_frame=False)
            }
        else:
            raise TypeError(f"[DatasourceSummary]: unsupported datasource type {type(datasource).__name__}.")

        state = DataSourceSummary.State(
            n_frames_per_camera=datasource.get_n_frames_per_camera(unique_sensors=True),
            n_frames_per_lidar=datasource.get_n_frames_per_lidar(unique_sensors=True),
            sequence_tracks_all=sequence_tracks_all,
            sequence_tracks_dynamic=sequence_tracks_dynamic,
            xform_matrices=xform_matrices,
            rig_trajectories=rig_trajectories,
            aabb_blb=datasource.aabb.blb.cpu().numpy(),
            aabb_trf=datasource.aabb.trf.cpu().numpy(),
        )
        return cls(state=state, datasource=datasource)

    @classmethod
    def from_json(cls, json: str, infer_missing: bool = False) -> Self:
        """
        Used to create the summary from a serialized checkpoint, when running inference without
        access to the dataset (i.e. instantiating just the model, not the entire system).

        Args:
            json: (str) serialized DatasourceSummary
            infer_missing: (bool) if True, missing fields will be inferred from the datasource
                (possibly setting to None if not defaults are provided). A warning will be issued.

        Returns:
            DatasourceSummary
        """
        return cls(
            state=DataSourceSummary.State.from_json(json, infer_missing=infer_missing),
            datasource=None,
        )

    @classmethod
    def from_dict(cls, json: Dict[str, Any], infer_missing: bool = False) -> Self:
        """
        Used to create the summary from a dictionary, when running inference without
        access to the dataset (i.e. instantiating just the model, not the entire system).

        Args:
            json: (dict) serialized DatasourceSummary
            infer_missing: (bool) if True, missing fields will be inferred from the datasource
                (possibly setting to None if not defaults are provided). A warning will be issued.

        Returns:
            DatasourceSummary
        """
        return cls(
            state=DataSourceSummary.State.from_dict(json, infer_missing=infer_missing),
            datasource=None,
        )

    @classmethod
    def _clean_track_id_str(cls, track_id: str) -> str:
        """
        Clean track IDs str by removing "@<source>" suffixes.
        """
        return track_id.split("@")[0]

    @classmethod
    def _clean_track_ids_in_serialized_sequence_tracks_dict(cls, sequence_tracks: dict) -> None:
        """
        Clean track IDs in the serialized sequence tracks dict by removing "@<source>" suffixes in all cuboid tracks with corresponding "tracks_data" key.
        This modifies the dictionary in-place to avoid copying large data structures.

        Args:
            sequence_tracks: The serialized dictionary to clean. Will only clean if sequence_tracks.values()["tracks_data"]["tracks_id"] of type list exists.
        """
        for cuboid_tracks in sequence_tracks.values():
            if "tracks_data" in cuboid_tracks and "tracks_id" in cuboid_tracks["tracks_data"]:
                # Clean the track IDs in the serialized data
                original_track_ids = cuboid_tracks["tracks_data"]["tracks_id"]
                assert isinstance(original_track_ids, list), (
                    f"Expected tracks_id to be a list, got {type(original_track_ids)}"
                )
                cleaned_track_ids = [cls._clean_track_id_str(track_id) for track_id in original_track_ids]
                cuboid_tracks["tracks_data"]["tracks_id"] = cleaned_track_ids

    @classmethod
    def _clean_track_ids_in_serialized_dict(cls, data: dict) -> None:
        """
        Clean track IDs in the serialized data by removing "@<source>" suffixes.
        This modifies the dictionary in-place to avoid copying large data structures.

        Args:
            data: The serialized data dictionary to clean
        """
        # Clean track IDs in sequence_tracks_all
        if "sequence_tracks_all" in data and data["sequence_tracks_all"] is not None:
            cls._clean_track_ids_in_serialized_sequence_tracks_dict(data["sequence_tracks_all"])

        # Clean track IDs in sequence_tracks_dynamic
        if "sequence_tracks_dynamic" in data and data["sequence_tracks_dynamic"] is not None:
            cls._clean_track_ids_in_serialized_sequence_tracks_dict(data["sequence_tracks_dynamic"])

    def to_json(self) -> str:
        # Get the serialized data
        data = self.state.to_dict()
        # Clean track IDs in the serialized data
        self._clean_track_ids_in_serialized_dict(data)
        # Convert to JSON using the state's to_json method with cleaned data
        return self.state.__class__.from_dict(data).to_json(indent=4)

    # below are getters actually used by models to initialize

    def get_n_frames_per_camera(self) -> npt.NDArray[np.int32]:
        """Unique sensors not supported in DatasourceSummary"""
        return self.state.n_frames_per_camera

    def get_cuboid_tracks(self, dynamic_only: bool) -> Optional[CuboidTracks]:
        """Available for NCORE-like datasets (dynamic / all tracks, in NRE frame)"""
        if dynamic_only:
            if self.state.sequence_tracks_dynamic is not None:
                # Assumes a single sequence
                return next(iter(self.state.sequence_tracks_dynamic.values()))
        else:
            if self.state.sequence_tracks_all is not None:
                # Assumes a single sequence
                return next(iter(self.state.sequence_tracks_all.values()))
        return None

    def get_aabb(self) -> AABB3D:
        return AABB3D(blb=torch.from_numpy(self.state.aabb_blb), trf=torch.from_numpy(self.state.aabb_trf))

    def get_rig_trajectories(self) -> Optional[RigTrajectories]:
        """Available for NCORE-like datasets (camera poses interpolated from trajectories)"""
        return self.state.rig_trajectories

    def get_track_point_clouds(
        self,
        cuboid_tracks: CuboidTracks,
        cuboid_dim_scale_factor: float = 1.0,
        lidar_ids: Optional[list[str]] = None,
        camera_ids: Optional[list[str]] = None,
        return_color: bool = False,
        keep_all_track_poses: bool = False,
        step_frame: int = 1,
        device: torch.device = torch.device("cuda"),
    ) -> Optional[Generator[TrackPointCloud, None, None]]:
        """No point clouds when DatasourceSummary comes from deserialization"""
        if self.datasource is not None:
            return self.datasource.get_track_point_clouds(
                cuboid_tracks,
                cuboid_dim_scale_factor=cuboid_dim_scale_factor,
                lidar_ids=lidar_ids,
                camera_ids=camera_ids,
                return_color=return_color,
                keep_all_track_poses=keep_all_track_poses,
                step_frame=step_frame,
                device=device,
            )
        else:
            return None

    def get_point_clouds(
        self,
        device: torch.device,
        lidar_ids: Optional[list[str]] = None,
        camera_ids: Optional[list[str]] = None,
        color_type: PointCloudColorType = None,
        non_dynamic_points_only: bool = True,
        step_frame: int = 1,
    ) -> Optional[Generator[PointCloud, None, None]]:
        """No point clouds when DatasourceSummary comes from deserialization"""
        if self.datasource is not None:
            return self.datasource.get_point_clouds(
                device=device,
                lidar_ids=lidar_ids,
                camera_ids=camera_ids,
                non_dynamic_points_only=non_dynamic_points_only,
                color_type=color_type,
                step_frame=step_frame,
            )
        else:
            return None

    def get_camera_frusta(
        self, camera_id: str, far_plane_depth: float
    ) -> Optional[Generator[tuple[CameraFrustum, int], None, None]]:
        """No camera frusta or bboxes when DatasourceSummary comes from deserialization"""
        if self.datasource is not None:
            return self.datasource.get_camera_frusta(camera_id=camera_id, far_plane_depth=far_plane_depth)
        else:
            return None
