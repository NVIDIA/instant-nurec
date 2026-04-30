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

import logging
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch

from omegaconf import OmegaConf

import ncore.impl.common.transformations as ncore_transformations
import nre.utils.cli as cli

from ncore.sensors import CameraModel
from nre.artifact.artifact import Artifact
from nre.config.viewer import ViewerConfig
from nre.datasets.tracks import CuboidTracks
from nre.render import RenderableModel
from nre.utils.profiling import ScopedTimer
from nre.utils.types import FrameConversion, HalfClosedInterval, PointCloud, RigTrajectories
from nre.viewer.abstract_viewer import AbstractViewer
from nre.viewer.dataset_interface import CameraTrajectoryData, CameraTrajectoryId, ViewerDatasetInterface
from nre.viewer.lock import MockLock


logger = logging.getLogger(__name__)


@dataclass
class USDZLikeTrajectoryData(CameraTrajectoryData):
    """
    Part of the dataset-viewer compatibility layer for NCOREDataset.
    """

    parent: BaseUSDZLikeDatasetInterface

    @staticmethod
    def create(
        *,
        trajectory_id: CameraTrajectoryId,
        camera_model: CameraModel,
        camera_unique_idx: int,
        frame_timestamps_us: torch.Tensor,
        parent: BaseUSDZLikeDatasetInterface,
        rig_time_range_us: HalfClosedInterval,
    ) -> USDZLikeTrajectoryData:
        start, end = frame_timestamps_us.unbind(-1)

        if (start == -1).any().item():
            DEFAULT_FRAME_DURATION_US = 30_000  # 30ms
            logger.warning(
                f"Frame timestamps contain -1 (placeholder for missing data), filling assuming {DEFAULT_FRAME_DURATION_US=}"
            )
            start = end - DEFAULT_FRAME_DURATION_US

        average_exposure_time_us = int((end - start).float().mean().item())

        return USDZLikeTrajectoryData(
            trajectory_id=trajectory_id,
            camera_model=camera_model,
            camera_unique_idx=camera_unique_idx,
            time_range_us=rig_time_range_us,
            average_exposure_time_us=average_exposure_time_us,
            parent=parent,
        )

    def get_poses_world(self, timestamps_us: torch.Tensor) -> torch.Tensor:
        return self.parent._get_pose_world(self.trajectory_id, timestamps_us)

    def get_closest_frame_image(self, timestamp_us: int) -> tuple[int, torch.Tensor]:
        return self.parent._get_closest_frame_and_timestamp(self.trajectory_id, timestamp_us)

    def get_frame_timestamps(self) -> torch.Tensor:
        """
        Get all frame timestamps that have corresponding images.

        Returns:
            - frame_timestamps: (n_frames, 2) tensor of (start, end) timestamps for each frame.
              Returns empty tensor if no frames are available.
        """
        return self.parent._get_frame_timestamps(self.trajectory_id)


class BaseUSDZLikeDatasetInterface(ViewerDatasetInterface):
    """Part of the dataset interface shared with NCORE viewer (in online mode): derived from RigTrajectories and CuboidTracks"""

    def __init__(self, rig_trajectories: RigTrajectories, cuboid_tracks: CuboidTracks) -> None:
        super().__init__()
        self.rig_trajectories = rig_trajectories
        self.cuboid_tracks = cuboid_tracks

        self.trajectory_ids: list[CameraTrajectoryId] = []
        for unique_camera_id, camera_calibration in self.rig_trajectories.camera_calibrations.items():
            candidate_trajectories = [
                r for r in self.rig_trajectories.rig_trajectories if r.sequence_id == camera_calibration.sequence_id
            ]
            assert len(candidate_trajectories) == 1, (
                f"Expected exactly one trajectory for {unique_camera_id}, got {len(candidate_trajectories)}"
            )

            trajectory = candidate_trajectories[0]
            self.trajectory_ids.append(
                CameraTrajectoryId(
                    sequence_id=trajectory.sequence_id,
                    camera_id=camera_calibration.logical_sensor_name,
                    unique_camera_id=unique_camera_id,
                )
            )

    def get_trajectory_ids(self) -> list[CameraTrajectoryId]:
        return self.trajectory_ids

    def _get_closest_frame_and_timestamp(
        self, trajectory_id: CameraTrajectoryId, timestamp_us: int
    ) -> tuple[int, torch.Tensor]:
        raise NotImplementedError()

    def _select_rig_trajectory(self, trajectory_id: CameraTrajectoryId) -> RigTrajectories.RigTrajectory:
        """
        Selects the rig trajectory for a given trajectory id (using trajectory_id.sequence_id).
        There should be a single rig trajectory for NRE but there may be two for NRM (context and supervision).
        """
        for rig_trajectory in self.rig_trajectories.rig_trajectories:
            if rig_trajectory.sequence_id == trajectory_id.sequence_id:
                return rig_trajectory
        raise ValueError(f"Trajectory {trajectory_id} not found in rig trajectories")

    def _get_frame_timestamps(self, trajectory_id: CameraTrajectoryId) -> torch.Tensor:
        """
        Get all frame timestamps for a given trajectory.

        Returns:
            - frame_timestamps: (n_frames, 2) tensor of (start, end) timestamps for each frame.
              Returns empty tensor if no frames are available.
        """
        rig_trajectory = self._select_rig_trajectory(trajectory_id)
        if trajectory_id.unique_camera_id not in rig_trajectory.cameras_frame_timestamps_us:
            return torch.empty((0, 2), dtype=torch.int64)
        return rig_trajectory.cameras_frame_timestamps_us[trajectory_id.unique_camera_id]

    def supports_get_closest_frame_image(self) -> bool:
        return False

    def _get_trajectory_data_bounded(
        self, trajectory_id: CameraTrajectoryId, time_range_us: HalfClosedInterval | None
    ) -> CameraTrajectoryData:
        """
        Get the trajectory data bounded by given timestamps ranges.
        If time_range_us is None, the trajectory data will be bounded by the rig trajectory timestamps.
        """
        rig_trajectory = self._select_rig_trajectory(trajectory_id)
        camera_calibration = self.rig_trajectories.camera_calibrations[trajectory_id.unique_camera_id]
        camera_frame_timestamps_us = rig_trajectory.cameras_frame_timestamps_us[trajectory_id.unique_camera_id]

        if time_range_us is None:
            rig_timestamps_us = rig_trajectory.T_rig_world_timestamps_us
            rig_time_range_us = HalfClosedInterval(
                int(rig_timestamps_us[0]),
                int(rig_timestamps_us[-1]),
            )
        else:
            rig_time_range_us = time_range_us

        return USDZLikeTrajectoryData.create(
            trajectory_id=trajectory_id,
            camera_model=CameraModel.from_parameters(camera_calibration.camera_model_parameters, device="cuda"),
            camera_unique_idx=camera_calibration.unique_sensor_idx,
            frame_timestamps_us=camera_frame_timestamps_us,
            parent=self,
            rig_time_range_us=rig_time_range_us,
        )

    def get_trajectory_data(self, trajectory_id: CameraTrajectoryId) -> CameraTrajectoryData:
        return self._get_trajectory_data_bounded(trajectory_id, None)

    def get_cuboid_tracks(self) -> Optional[CuboidTracks]:
        return self.cuboid_tracks

    def distance_nre_to_distance_world(self, distance: torch.Tensor) -> torch.Tensor:
        return distance / self.rig_trajectories.world_to_nre.target_scale

    def _get_pose_world(
        self,
        trajectory_id: CameraTrajectoryId,
        timestamps_us: torch.Tensor,
    ) -> torch.Tensor:
        assert timestamps_us.ndim == 1, f"{timestamps_us.shape=}"

        rig_trajectory = self._select_rig_trajectory(trajectory_id)
        camera_calibration = self.rig_trajectories.camera_calibrations[trajectory_id.unique_camera_id]

        pose_timestamps_us = rig_trajectory.T_rig_world_timestamps_us
        T_rig_worlds = rig_trajectory.T_rig_worlds
        interpolator = ncore_transformations.PoseInterpolator(
            T_rig_worlds.cpu().numpy(), pose_timestamps_us.cpu().numpy()
        )
        T_rig_world = interpolator.interpolate_to_timestamps(timestamps_us)  # [t, 4, 4]
        T_camera_world = T_rig_world @ camera_calibration.T_sensor_rig.cpu().numpy()
        return torch.from_numpy(T_camera_world)

    @property
    def world_to_nre(self) -> FrameConversion:
        """
        Get the world to frame transform from the datamodule
        """
        return self.rig_trajectories.world_to_nre

    def get_rig_trajectories(self) -> RigTrajectories | None:
        return self.rig_trajectories


ERR_MSG_TEMPLATE = (
    "{artifact} does not contain `{value}`. Make sure you trained with "
    "checkpoint.artifact.rig_trajectories.enabled=true and "
    "checkpoint.artifact.sequence_tracks.enabled=true"
)


class ViewerUSDZInterface(BaseUSDZLikeDatasetInterface):
    """
    Specializations for pure USDZ artifacts: point clouds and nearest frames are not available.
    """

    def __init__(self, artifact: Artifact) -> None:
        try:
            rig_trajectories_raw = artifact.rig_trajectories
        except KeyError as e:
            raise ValueError(ERR_MSG_TEMPLATE.format(artifact=artifact, value="rig_trajectories")) from e
        rig_trajectories = RigTrajectories.from_dict(rig_trajectories_raw)

        try:
            sequence_tracks_raw = artifact.sequence_tracks
        except KeyError as e:
            raise ValueError(ERR_MSG_TEMPLATE.format(artifact=artifact, value="sequence_tracks")) from e

        sequence_tracks = [CuboidTracks.from_dict(v) for v in sequence_tracks_raw.values()]
        cuboid_tracks = sequence_tracks[0]  # Assumes a single sequence

        super().__init__(rig_trajectories=rig_trajectories, cuboid_tracks=cuboid_tracks)

    def get_point_cloud(
        self, color_type: Optional[Literal["camera-rgb", "semantics"]] = None, step_frame: int = 10
    ) -> PointCloud | None:
        return None

    def supports_get_closest_frame_image(self) -> bool:
        return False


class USDZViewer(AbstractViewer):
    def __init__(
        self,
        artifact: Artifact,
        host: str,
        port: int,
        enable_nrend: bool,
    ) -> None:
        super().__init__()

        self.artifact = artifact

        self.model = RenderableModel.load_from_artifact(artifact, enable_nrend=enable_nrend)
        self.config = ViewerConfig.model_validate(OmegaConf.create(artifact.parsed_config).viewer)
        self.host = host
        self.port = port
        self.system_lock = MockLock()  # no need to lock the system
        self._datasource_interface = ViewerUSDZInterface(self.artifact)

    def get_dataset_interface(self) -> ViewerUSDZInterface:
        return self._datasource_interface


@click.command("viewer")
@click.option(
    "--artifact-path",
    type=Path,
    help="Path to the usdz file.",
    required=True,
)
@click.option(
    "--host",
    type=str,
    help="Host to serve on ",
    default="127.0.0.1",
)
@click.option(
    "--port",
    type=int,
    help="Port to serve on ",
    default=8080,
)
@click.option(
    "--enable-nrend/--no-enable-nrend",
    help="Use nrend for faster rendering if supported by the model.",
    default=True,
)
@cli.scopedtimer_cli_options(print_func=logger.info)
def run_viewer(
    artifact_path: Path,
    host: str,
    port: int,
    enable_nrend: bool,
) -> None:
    artifact = Artifact(source=artifact_path)
    viewer = USDZViewer(artifact, host=host, port=port, enable_nrend=enable_nrend)
    viewer.start_server()

    try:
        while True:
            time.sleep(0.1)  # wait for ctrl+c to stop the server
    except KeyboardInterrupt:
        viewer.stop_server()

        time.sleep(2)  # give the threads time to shut down

    ScopedTimer.print_summary()
