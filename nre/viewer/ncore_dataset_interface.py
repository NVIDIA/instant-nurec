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

from typing import Literal, Optional

import torch

from ncore.data import FrameTimepoint
from nre.datasets.ncore import NCOREDataSource
from nre.utils.types import PointCloud
from nre.viewer.dataset_interface import CameraTrajectoryId
from nre.viewer.usdz_viewer import BaseUSDZLikeDatasetInterface


class ViewerNCOREInterface(BaseUSDZLikeDatasetInterface):
    """
    Data available in the NCORE dataset and not in a USDZ file: point clouds and view frames.
    """

    def __init__(self, datasource: NCOREDataSource):
        cuboid_tracks = datasource.get_cuboid_tracks(dynamic_only=False, world_frame=True, include_generated=True)
        rig_trajectories = datasource.get_rig_trajectories()
        super().__init__(cuboid_tracks=cuboid_tracks, rig_trajectories=rig_trajectories)

        self.datasource = datasource

    def get_point_cloud(
        self, color_type: Optional[Literal["camera-rgb", "semantics"]] = None, step_frame: int = 10
    ) -> PointCloud | None:
        point_cloud: PointCloud | None
        try:
            point_cloud = PointCloud.collate_fn(
                [
                    pc
                    for pc in self.datasource.get_point_clouds(
                        device=torch.device("cpu"), step_frame=step_frame, color_type=color_type
                    )
                ],
                device=torch.device("cpu"),
            )
        except (StopIteration, ValueError):
            point_cloud = None  # no point clouds (perhaps too short?)

        return point_cloud

    def _get_closest_frame_and_timestamp(
        self, trajectory_id: CameraTrajectoryId, timestamp_us: int
    ) -> tuple[int, torch.Tensor]:
        camera_sensor = self.datasource.camera_sensors[trajectory_id.camera_id]
        frame_index = camera_sensor.get_closest_frame_index(timestamp_us=timestamp_us)
        frame_timestamp = camera_sensor.get_frame_timestamp_us(frame_index, frame_timepoint=FrameTimepoint.END)
        rgb_np = camera_sensor.get_frame_image_array(frame_index)
        return int(frame_timestamp), torch.from_numpy(rgb_np)

    def supports_get_closest_frame_image(self) -> bool:
        return True
