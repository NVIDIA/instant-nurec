# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import gc
import logging

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ncore.data import ConcreteCameraModelParametersUnion
from nre.artifact import Artifact
from nre.datasets.tracks import CuboidTracks
from nre.render.actors import ActorTracks
from nre.utils.types import RigTrajectories


log = logging.getLogger(__name__)


# Making it immutable so it is safe to use in dictionary keys (otherwise it is unhashable).
@dataclass(frozen=True)
class LogicalCameraId:
    logical_camera_id: str


class SceneInfo:
    """
    High-level API providing accessors to camera views, calibration parameters, trajectories and actor poses
    used to reconstruct a 3D scene and stored in a USDZ artifact.

    This can be used to query training view/pose data in order to render the model from the training viewpoints.
    """

    @dataclass
    class Camera:
        """Camera used to reconstruct the scene"""

        intrinsics: ConcreteCameraModelParametersUnion
        camera_to_rig: torch.Tensor
        logical_camera_id: str
        unique_sensor_idx: int

    _rig_trajectories: RigTrajectories  # Rig trajectories used to reconsrtuct the scene, includes world_to_nre
    _cameras: Dict[LogicalCameraId, Camera]  # Cameras used to reconstruct the scene
    _actor_tracks: ActorTracks  # Tracks of controllable cuboids to report to the user

    def __init__(self, artifact: Artifact):
        self._rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)
        self._cameras = SceneInfo._cameras_from_rig_trajectories(self._rig_trajectories)

        # Initialize cuboid tracks
        sequence_tracks = {k: CuboidTracks.from_dict(v) for k, v in artifact.sequence_tracks.items()}
        assert isinstance(sequence_tracks, dict)
        if len(sequence_tracks) > 0:
            artifact_cuboid_tracks = next(iter(sequence_tracks.values()))  # Assumes only one sequence
            self._actor_tracks = ActorTracks._from_cuboid_tracks(artifact_cuboid_tracks)
        else:
            self._actor_tracks = ActorTracks()

        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _cameras_from_rig_trajectories(
        rig_trajectories: RigTrajectories,
    ) -> Dict[LogicalCameraId, SceneInfo.Camera]:
        cameras: dict[LogicalCameraId, SceneInfo.Camera] = {}
        unique_camera_id_to_camera_id = {
            uci: rig_trajectories.camera_calibrations[uci].logical_sensor_name
            for uci in rig_trajectories.camera_calibrations.keys()
        }
        # Assume we only have a single trajectory
        assert len(rig_trajectories.rig_trajectories) == 1, (
            f"{SceneInfo.__name__}: expected a single rig_trajectory, got {len(rig_trajectories.rig_trajectories)}"
        )
        trajectory = rig_trajectories.rig_trajectories[0]
        for unique_camera_id in trajectory.cameras_frame_timestamps_us.keys():
            camera_calibration = rig_trajectories.camera_calibrations[unique_camera_id]
            logical_camera_id = unique_camera_id_to_camera_id[unique_camera_id]
            camera = SceneInfo.Camera(
                intrinsics=camera_calibration.camera_model_parameters,
                camera_to_rig=camera_calibration.T_sensor_rig,
                logical_camera_id=logical_camera_id,
                unique_sensor_idx=camera_calibration.unique_sensor_idx,
            )
            cameras[LogicalCameraId(logical_camera_id)] = camera
        return cameras

    def get_available_cameras(self) -> List[LogicalCameraId]:
        """Return all (logical_camera_id) available to query cameras via get_camera()"""
        return list(self._cameras.keys())

    def get_camera(self, camera_key: LogicalCameraId) -> SceneInfo.Camera:
        """Access calibration parameters and associated metadata of a selected camera on a selected rig trajectory
        that was used to reconstruct the scene.
        """
        return self._cameras[camera_key]

    def get_num_trajectories(self) -> int:
        """Get the number of camera rig trajectories used to reconstruct this scene."""
        return len(self._rig_trajectories.rig_trajectories)

    def get_trajectory(self, trajectory_idx: int = 0) -> RigTrajectories.RigTrajectory:
        """Get rig trajectory data for a given trajectory index between 0 and get_num_trajectories()."""
        return self._rig_trajectories.rig_trajectories[trajectory_idx]

    def get_trajectory_time_range(self, trajectory_idx: int = 0) -> Tuple[int, int]:
        """Get the full (start,end) timestamp range of a selected trajectory in microseconds.

        trajectory_index is between 0 and get_num_trajectories().
        """
        rig_timestamps_us = self._rig_trajectories.rig_trajectories[trajectory_idx].T_rig_world_timestamps_us
        start_time_us = rig_timestamps_us[0].item()
        end_time_us = rig_timestamps_us[-1].item()
        assert start_time_us >= 0
        assert end_time_us >= 0
        return (int(start_time_us), int(end_time_us))

    def get_actor_tracks(self) -> ActorTracks:
        """Get actor tracks loaded from the sequence_tracks.json in the USDZ artifact.

        Consider using get_actor_tracks() from the RenderableModel instead, as it may provide higher precision.
        """
        return self._actor_tracks

    # TODO: Add accessors to LiDAR data (see gRPC rendering service for inspiration).
