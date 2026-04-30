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
import numpy as np
import torch
import torch.nn as nn

import nre.utils.cli as cli

from ncore.data import (
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.sensors import CameraModel
from nre.config.parse import parse_typed_config
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.models.gaussians.utils import PLYGaussianLoader, sh_degree_to_specular_dim
from nre.render import RenderableModel
from nre.utils.misc import to_torch
from nre.utils.profiling import ScopedTimer
from nre.utils.types import FrameConversion, HalfClosedInterval, PointCloud
from nre.viewer.abstract_viewer import AbstractViewer
from nre.viewer.dataset_interface import CameraTrajectoryData, CameraTrajectoryId, ViewerDatasetInterface
from nre.viewer.lock import MockLock
from nre.viewer.viewpoint import LookAtPose


logger = logging.getLogger(__name__)


def compute_camera_position(
    points: np.ndarray, up_vector: np.ndarray = np.array([0.0, 1.0, 0.0]), margin_factor: float = 1.2
) -> np.ndarray:
    """
    Similar to the earlier approach, but uses only a central percentile of points
    to ignore outliers when computing the bounding box.
    """
    # Get 5th and 95th percentile bounds directly
    bounds = np.percentile(points, [5.0, 95.0], axis=0)
    center = bounds.mean(axis=0)

    # Camera distance based on largest dimension
    camera_distance = margin_factor * (bounds[1] - bounds[0]).max()
    camera_position = center + np.array([0.0, 0.0, camera_distance])

    return LookAtPose(up=up_vector, look_at=center, position=camera_position).to_se3().as_matrix()


def create_default_camera_model() -> CameraModel:
    """Creates a default pinhole camera model."""
    return CameraModel.from_parameters(
        OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([935.12, 635.05], dtype=np.float32),
            focal_length=np.array([2059.0, 2059.0], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
    )


@dataclass
class PLYLikeTrajectoryData(CameraTrajectoryData):
    """
    Part of the dataset-viewer compatibility layer.
    """

    parent: ViewerPLYInterface

    @staticmethod
    def create(
        *,
        trajectory_id: CameraTrajectoryId,
        camera_model: CameraModel,
        camera_unique_idx: int,
        parent: ViewerPLYInterface,
        rig_time_range_us: HalfClosedInterval,
    ) -> PLYLikeTrajectoryData:
        return PLYLikeTrajectoryData(
            trajectory_id=trajectory_id,
            camera_model=camera_model,
            camera_unique_idx=camera_unique_idx,
            time_range_us=rig_time_range_us,
            average_exposure_time_us=0,
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


class ViewerPLYInterface(ViewerDatasetInterface):
    """Part of the dataset interface shared with NCORE viewer (in online mode): derived from RigTrajectories and CuboidTracks"""

    def __init__(self, camera_model: CameraModel, initial_camera_pose: torch.Tensor) -> None:
        super().__init__()

        self.camera_trajectory_id = CameraTrajectoryId(
            sequence_id="PLY reconstruction",
            camera_id="default_pinhole",
            unique_camera_id="default_pinhole",
        )

        # TODO: we can make this configurable from the viewer
        self.camera_model = camera_model
        self.initial_camera_pose = initial_camera_pose
        self.trajectory_ids = [self.camera_trajectory_id]

    def get_trajectory_ids(self) -> list[CameraTrajectoryId]:
        return self.trajectory_ids

    def _get_closest_frame_and_timestamp(
        self, trajectory_id: CameraTrajectoryId, timestamp_us: int
    ) -> tuple[int, torch.Tensor]:
        raise NotImplementedError()

    def _get_frame_timestamps(self, trajectory_id: CameraTrajectoryId) -> torch.Tensor:
        """
        Get all frame timestamps for a given trajectory.

        Returns:
            - frame_timestamps: (n_frames, 2) tensor of (start, end) timestamps for each frame.
              Returns empty tensor if no frames are available.
        """
        # PLY doesn't have frame timestamps, so return empty tensor
        return torch.empty((0, 2), dtype=torch.int64)

    def supports_get_closest_frame_image(self) -> bool:
        return False

    def get_trajectory_data(self, trajectory_id: CameraTrajectoryId) -> CameraTrajectoryData:
        return PLYLikeTrajectoryData.create(
            trajectory_id=self.camera_trajectory_id,
            camera_model=self.camera_model,
            camera_unique_idx=-1,  # -1 is treated as invalid sensor id in, e.g., bilateral grid
            parent=self,
            rig_time_range_us=HalfClosedInterval(0, 1),
        )

    def get_cuboid_tracks(self) -> Optional[CuboidTracks]:
        return None

    def distance_nre_to_distance_world(self, distance: torch.Tensor) -> torch.Tensor:
        return distance

    def _get_pose_world(
        self,
        trajectory_id: CameraTrajectoryId,
        timestamps_us: torch.Tensor,
    ) -> torch.Tensor:
        return self.initial_camera_pose

    @property
    def world_to_nre(self) -> FrameConversion:
        """
        Get the world to frame transform from the datamodule
        """
        return FrameConversion.from_origin_scale_axis(
            target_origin=np.zeros(3, dtype=np.float32),  # put the scene's center at the origin
            target_scale=1.0,
            target_axis=[0, 1, 2],
        )

    def get_point_cloud(
        self, color_type: Optional[Literal["camera-rgb", "semantics"]] = None, step_frame: int = 10
    ) -> PointCloud | None:  # type: ignore
        return None


class PLYViewer(AbstractViewer):
    def __init__(
        self,
        ply_file_path: Path,
        config: Path,
        host: str,
        port: int,
        enable_nrend: bool,
    ) -> None:
        super().__init__()

        self.host = host
        self.port = port
        self.system_lock = MockLock()  # no need to lock the system

        self.base_config = parse_typed_config(config_name=str(config), hydra_args=["dataset.path=", "out_dir="])
        self.config = self.base_config.viewer
        self.ply_data = PLYGaussianLoader(ply_file_path)

        camera_model = create_default_camera_model()
        # Initialize dummy datasource state
        datasource_state = DataSourceSummary.State(
            n_frames_per_camera=np.array([1], dtype=np.int32),
            sequence_tracks_all=None,
            sequence_tracks_dynamic=None,
            rig_trajectories=None,
            xform_matrices=np.zeros((0, 4), dtype=np.float32),
            aabb_blb=np.zeros((1, 3), dtype=np.float32),
            aabb_trf=np.ones((1, 3), dtype=np.float32),
        )
        datasource = DataSourceSummary(state=datasource_state, datasource=None)
        model = GaussiansComposite(
            self.base_config.model, self.base_config.trainer, datasource, init_from_datasource=False
        )

        initial_camera_pose = compute_camera_position(self.ply_data.positions.cpu().numpy())
        self._datasource_interface = ViewerPLYInterface(
            camera_model=camera_model, initial_camera_pose=to_torch(initial_camera_pose[None, ...], device="cpu")
        )

        # Load the paramaters from the ply file
        num_specular_dims = sh_degree_to_specular_dim(
            int(self.base_config.model.layers["gaussians"].progressive_training.max_n_features)
        )

        nodes = model.gaussians_nodes["gaussians"]
        device = self.ply_data.features_albedo.device

        # Set basic parameters
        for param_name in ["positions", "rotations", "scales", "densities", "features_albedo"]:
            setattr(nodes, param_name, nn.Parameter(getattr(self.ply_data, param_name)))

        # Handle specular features
        if self.ply_data.features_specular is not None:
            assert self.ply_data.features_specular.shape[1] == num_specular_dims, "Specular features dimension mismatch"
            nodes.features_specular = nn.Parameter(self.ply_data.features_specular)
            if hasattr(nodes.gaussians, "n_active_features"):
                nodes.gaussians.n_active_features = num_specular_dims
        else:
            nodes.features_specular = nn.Parameter(
                torch.zeros(self.ply_data.features_albedo.shape[0], num_specular_dims, device=device)
            )

        self.model = RenderableModel(model)
        assert isinstance(self.model._model, GaussiansComposite), "Model is not a GaussiansComposite model"
        self.model._model.to("cuda")
        self.model._model.eval()

        if enable_nrend:
            self.model._nrend.tracks = None
            self.model._model.setup_nrend()

    def get_dataset_interface(self) -> ViewerPLYInterface:
        return self._datasource_interface


@click.command("ply_viewer")
@click.option(
    "--ply-path",
    type=Path,
    help="Path to the ply file.",
    required=True,
)
@click.option(
    "--config",
    type=Path,
    help="Path to the ply file.",
    default="configs/experimental/3dgrt/exp_ply_viewer.yaml",
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
    default=False,
)
@cli.scopedtimer_cli_options(print_func=logger.info)
def run_ply_viewer(
    ply_path: Path,
    config: Path,
    host: str,
    port: int,
    enable_nrend: bool,
) -> None:
    ply_viewer = PLYViewer(ply_path, config=config, host=host, port=port, enable_nrend=enable_nrend)
    ply_viewer.start_server()

    try:
        while True:
            time.sleep(0.1)  # wait for ctrl+c to stop the server
    except KeyboardInterrupt:
        ply_viewer.stop_server()

        time.sleep(2)  # give the threads time to shut down

    ScopedTimer.print_summary()
