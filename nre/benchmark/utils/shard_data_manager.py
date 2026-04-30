# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shard Data Manager utility for managing shard loading and 3D label extraction."""

import glob
import logging

from typing import Any, Dict

import numpy as np

from scipy.spatial.transform import Rotation
from tqdm import tqdm

import ncore.impl.common.transformations as ncore_transformations

from ncore.sensors import CameraModel
from ncore_internal.data.v3 import ShardDataLoader


logger = logging.getLogger(__name__)


class ShardDataManager:
    """Manages shard loading, sensor initialization, and 3D label extraction.

    This class encapsulates all operations related to loading and processing
    data from shard files, including sensor setup and 3D bounding box extraction.

    Usage:
        # Load and process single shard
        manager = ShardDataManager('shard/*.zarr', 'camera_front')

        # Load and process multiple shards (reuse same instance)
        manager = ShardDataManager('shard1/*.zarr', 'camera_front')
        # ... process shard1 ...
        manager.load('shard2/*.zarr')
        # ... process shard2 ...
    """

    def __init__(self, shard_pattern: str, camera_id: str):
        """Initialize shard data manager and load shard data immediately.

        Args:
            shard_pattern: Glob pattern for shard files.
            camera_id: ID of the camera to use.
        """
        self.camera_id = camera_id
        self.shard_pattern = shard_pattern
        self._load(shard_pattern)

    def load(self, shard_pattern: str) -> None:
        """Load or reload shard data.

        Use this to switch to a different shard without recreating
        the manager instance. Useful for batch processing.

        Example:
            manager = ShardDataManager('shard1/*.zarr', 'camera_front')
            # ... process shard1 ...
            manager.load('shard2/*.zarr')
            # ... process shard2 ...

        Args:
            shard_pattern: Glob pattern for shard files.
        """
        self.shard_pattern = shard_pattern
        self._load(shard_pattern)

    def _load(self, shard_pattern: str) -> None:
        """Internal method to load shard data.

        Args:
            shard_pattern: Glob pattern for shard files.
        """
        logger.info("Loading shard data...")
        shard_files = glob.glob(shard_pattern)
        if not shard_files:
            raise ValueError(f"No shard files found matching pattern: {shard_pattern}")

        logger.info("  Found %d shard file(s)", len(shard_files))
        self.shard_loader: ShardDataLoader = ShardDataLoader(shard_files)

        # Get sensors
        camera_ids = self.shard_loader.get_camera_ids()
        logger.info("  Available cameras: %s", camera_ids)

        if self.camera_id not in camera_ids:
            raise ValueError(f"Camera {self.camera_id} not found in shard")

        self.camera_sensor: Any = self.shard_loader.get_camera_sensor(self.camera_id)
        lidar_ids = self.shard_loader.get_lidar_ids()
        if not lidar_ids:
            raise ValueError("No LIDAR sensors found in shard")
        self.lidar_sensor: Any = self.shard_loader.get_lidar_sensor(lidar_ids[0])

        # Get camera model
        camera_model_params = self.camera_sensor.get_camera_model_parameters()
        self.camera_model: CameraModel = CameraModel.from_parameters(camera_model_params, device="cpu")

        logger.info("  Successfully loaded shard data and camera model")

    @staticmethod
    def box2corners(bbox3) -> np.ndarray:
        """Convert 3D bounding box to 8 corner points.

        Args:
            bbox3: 3D bounding box with center, dimensions, and rotation.

        Returns:
            Array of 8 corner points [8, 3].
        """
        center = bbox3.to_array()[:3]
        dim = bbox3.to_array()[3:6]
        rot = bbox3.to_array()[6:]

        half_dim = dim / 2
        local_corners = np.array(
            [
                [half_dim[0], half_dim[1], half_dim[2]],
                [-half_dim[0], half_dim[1], half_dim[2]],
                [-half_dim[0], -half_dim[1], half_dim[2]],
                [half_dim[0], -half_dim[1], half_dim[2]],
                [half_dim[0], -half_dim[1], -half_dim[2]],
                [-half_dim[0], -half_dim[1], -half_dim[2]],
                [-half_dim[0], half_dim[1], -half_dim[2]],
                [half_dim[0], half_dim[1], -half_dim[2]],
            ]
        )

        rotation_matrix = Rotation.from_euler("XYZ", rot).as_matrix()
        corners = np.array([rotation_matrix @ corner + center for corner in local_corners])
        return corners

    def compile_3d_labels(self) -> Dict[int, Dict[str, dict]]:
        """Compile 3D bounding box labels from all LiDAR frames.

        Returns:
            Dictionary mapping frame_idx to track data:
            {frame_idx: {track_id: {"vertices": [8,3],
            "class": str, "bbox3": ...}}}
        """
        logger.info("Compiling labels for all frames...")
        box_dict = {}

        lidar_frame_range = self.lidar_sensor.get_frame_index_range()

        for frame_idx in tqdm(lidar_frame_range, desc="Loading labels"):
            frame_labels = self.lidar_sensor.get_frame_labels(frame_idx)

            # Get transformation from sensor to world
            t_sensor_world = self.lidar_sensor.get_frame_T_sensor_world(frame_idx)

            frame_boxes = {}
            for label in frame_labels:
                # Get 3D bounding box corners in SENSOR coordinates
                corners_sensor = self.box2corners(label.bbox3)

                # Transform from sensor coordinates to WORLD coordinates
                corners_world = ncore_transformations.transform_point_cloud(corners_sensor, t_sensor_world)

                # Store with FULL track ID
                track_id = str(label.track_id)
                frame_boxes[track_id] = {
                    "vertices": corners_world,
                    "class": str(label.label_class),
                    "bbox3": label.bbox3,
                }

            box_dict[frame_idx] = frame_boxes

        logger.info("Compiled labels for %d frames", len(box_dict))
        return box_dict
