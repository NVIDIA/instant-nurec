# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import abstractmethod
from pathlib import Path

import numpy as np

import nre.utils.ncore_utils as ncore_utils

from internal.scripts.ncore_vis.data_utils import Camera
from ncore_internal.data.v3 import ShardDataLoader


class NCoreLoader:
    """
    An NCore Dataset loader base class that controls the loading and packaging of the NCore data in a format
    that is accepted by the NCore Viewer. Implementations must extend off this class and implement the
    abstract methods.
    """

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        if len(Path(dataset_path).suffixes) == 0:
            dataset_path += ".zarr.itar"

        self.shard_files = ShardDataLoader.evaluate_shard_file_pattern(dataset_path)
        self.loader = ShardDataLoader(self.shard_files, open_consolidated=True)
        self.aux_loader = ncore_utils.AuxShardDataLoader.from_shard_data_loader(self.loader, open_consolidated=True)

    def get_lidar_semantic_segmentation_meta(self, lidar_id: str) -> dict | None:
        """
        Returns semantic seg dictionary if the lidar sensor with given [id] has
        semantic data in the NCORE dataset provided or None otherwise.

        Args:
            camera_id (str): id of the camera sensor

        Returns:
            dict | None: a dictionary containing semantic metadata if present in [lidar_id],
                         otherwise None
        """
        try:
            return self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)
        except Exception:
            return None

    def get_camera_semantic_segmentation_meta(self, camera_id: str) -> dict | None:
        """
        Returns semantic seg dictionary if the camera sensor with given [id] has
        semantic data in the NCORE dataset provided or None otherwise.

        Args:
            camera_id (str): id of the camera sensor

        Returns:
            dict | None: a dictionary containing semantic metadata if present in [camera_id],
                         otherwise None
        """
        # It is possible that some camera sensors do not have semantic segmentation data while
        # others do, this is to catch that edge case.
        try:
            return self.aux_loader.get_semantic_segmentation_meta(camera_id)
        except Exception:
            return None

    def get_camera_instance_segmentation_meta(self, camera_id: str) -> dict | None:
        """
        Returns instance seg meta dictionary if the camera sensor with given [id]
        has instance segmentation data in the NCORE dataset provided or None otherwise.

        Args:
            camera_id (str): id of the camera sensor

        Returns:
            dict | None: a dictionary containing instance metadata if present in [camera_id],
                         otherwise None
        """
        # It is possible that some camera sensors do not have instance segmentation data while
        # others do, this is to catch that edge case.
        try:
            return self.aux_loader.get_instance_segmentation_meta(camera_id)
        except Exception:
            return None

    def get_camera_depth_meta(self, camera_id: str) -> dict | None:
        """
        Returns depth meta dictionary if the camera sensor with given [id]
        has depth data in the NCORE dataset provided or None otherwise.
        """
        try:
            return self.aux_loader.get_depth_meta(camera_id)
        except Exception:
            return None

    def get_camera_normals_meta(self, camera_id: str) -> dict | None:
        """
        Returns normals meta dictionary if the camera sensor with given [id]
        has normals data in the NCORE dataset provided or None otherwise.
        """
        try:
            return self.aux_loader.get_normal_meta(camera_id)
        except Exception:
            return None

    def has_camera_semantic_segmentation(self, camera_id: str) -> bool:
        """
        Returns true if camera with [camera_id] had semantic segmentation
        data, otherwise None.

        Args:
            camera_id (str): id of the camera sensor

        Returns:
            bool: Whether the camera sensor has semantic segmentation data
                  or not.
        """
        return self.get_camera_semantic_segmentation_meta(camera_id) is not None

    def has_camera_instance_segmentation(self, camera_id: str) -> bool:
        """
        Returns true if camera with [camera_id] had instance segmentation
        data, otherwise None.

        Args:
            camera_id (str): id of the camera sensor

        Returns:
            bool: Whether the camera sensor has instance segmentation data
                  or not.
        """
        return self.get_camera_instance_segmentation_meta(camera_id) is not None

    def has_camera_depth(self, camera_id: str) -> bool:
        """
        Returns true if camera with [camera_id] had depth
        data, otherwise None.

        Args:
            camera_id (str): id of the camera sensor

        Returns:
            bool: Whether the camera sensor has depth data
                  or not.
        """
        return self.get_camera_depth_meta(camera_id) is not None

    def has_camera_normals(self, camera_id: str) -> bool:
        """
        Returns true if camera with [camera_id] had normals
        data, otherwise None.

        Args:
            camera_id (str): id of the camera sensor

        Returns:
            bool: Whether the camera sensor has normals data
                  or not.
        """
        return self.get_camera_normals_meta(camera_id) is not None

    def has_lidar_semantic_segmentation(self, lidar_id: str) -> bool:
        """
        Returns true if lidar with [lidar_id] had semantic segmentation
        data, otherwise None.

        Args:
            lidar_id (str): id of the lidar sensor

        Returns:
            bool: Whether the lidar sensor has semantic segmentation data
                  or not.
        """
        return self.get_lidar_semantic_segmentation_meta(lidar_id) is not None

    def get_lidar_ids(self) -> list[str]:
        """
        Returns a list of all lidar ids that were found in the provided
        dataset.

        Returns:
            list[str]: list of lidar sensor ids
        """
        return self.loader.get_lidar_ids()

    def get_camera_ids(self) -> list[str]:
        """
        Returns a list of all camera ids that were found in the provided
        dataset.

        Returns:
            list[str]: list of camera sensor ids
        """
        return self.loader.get_camera_ids()

    def get_sensor_indices(self, sensor_id: str) -> list[int]:
        """
        Given a [sensor_id], returns the frames associated with that camera.

        Args:
            camera_id (str): id of the sensor

        Returns:
            list[int]: list of frames available to the camera
        """
        sensor = self.loader.get_sensor(sensor_id)
        return list(sensor.get_frame_index_range(0, sensor.get_frames_count()))

    def get_closest_sensor_frame(self, frame: int, reference_sensor_id: str, target_sensor_id: str) -> int:
        """
        Given a [frame], the sensor id of the reference sensor the frame came from
        and the id of the target sensor, finds the closest frame within the target
        sensor to the timestamp

        Args:
            frame (int): frame from the reference sensor
            reference_sensor_id (str): id of reference sensor
            target_sensor_id (str): id of the target sensor

        Returns:
            int: frame from the target sensor closest to the frame provided
        """
        ref_sensor = self.loader.get_sensor(reference_sensor_id)
        timestamp = ref_sensor.get_frame_timestamp_us(frame)
        target_sensor = self.loader.get_sensor(target_sensor_id)
        return target_sensor.get_closest_frame_index(timestamp)

    @abstractmethod
    def get_trajectories(self) -> np.ndarray:
        """Returns the rig trajectories of the ncore data.

        Returns:
            np.ndarray: rig trajectories
        """
        ...

    @abstractmethod
    def get_camera(self, camera_id: str) -> Camera:
        """
        Returns information about the camera sensor associated with [camera_id].

        Args:
            camera_id (str): id of the camera sensor

        Returns:
            Camera: a Camera object associated with the sensor with id: [camera_id]
                    that holds information about the camera
        """
        ...

    @abstractmethod
    def get_camera_image(self, camera_id: str, frame: int) -> np.ndarray:
        """
        Given a [camera_id] and [frame], returns the image array associated
        with the camera at the given frame.

        Args:
            camera_id (str): id of the camera sensor
            frame (int): frame within range of the camera data

        Returns:
            np.ndarray: rbg image array
        """
        ...

    @abstractmethod
    def get_camera_semantic_image(self, camera_id: str, frame: int) -> np.ndarray | None:
        """
        Given a [camera_id] and [frame], returns the image array associated
        with the camera at the given frame but the color is based on semantic
        segmentation rather than rgb data at the point. If camera sensor does
        not have semantic segmentation data, then returns None.

        Args:
            camera_id (str): id of the camera sensor
            frame (int): frame within range of the camera data

        Returns:
            np.ndarray | None: a semantic camera image array if camera has semantics,
                               otherwise None
        """
        ...

    @abstractmethod
    def get_camera_semantic_overlay_image(self, camera_id: str, frame: int) -> np.ndarray | None:
        """
        Given a [camera_id] and [frame], returns the image array associated
        with the camera at the given frame but the color is the original rgb
        data overlayed with the semantic data to create a masked image. If
        the camera sensor does not have semantic segmentation data, then
        returns None.

        Args:
            camera_id (str): id of the camera sensor
            frame (int): frame within range of the camera data

        Returns:
            np.ndarray | None: A masked rgb image with semantic overlay if
                               camera has semantics, otherwise None
        """
        ...

    @abstractmethod
    def get_point_cloud(self, lidar_id: str, frame: int) -> np.ndarray:
        """
        Given a [lidar_id] and [frame], returns the point cloud associated
        with the lidar at the given frame.

        Args:
            lidar_id (str): id of the lidar sensor
            frame (int): frame within range of the lidar data

        Returns:
            np.ndarray: Array of shape (N, 3) with xyz-points
        """
        ...

    @abstractmethod
    def get_point_cloud_color(self, lidar_id: str, frame: int, type: str = "Intensity") -> np.ndarray:
        """
        Given a [lidar_id] and [frame], returns the color for each point
        associated with the lidar point cloud at the given [frame] and
        color coded based on [type].

        Args:
            lidar_id (str): id of the lidar sensor
            frame (int): frame within range of the lidar data
            type (str): controls what kind of data that is to be captured
                        within the color (e.g. Semantic, Intensity)

        Returns:
            np.ndarray : Array of shape (N, 3) with rgb values if [type] data
                         exists for [lidar_id], otherwise defaults to a color
        """
        ...

    @abstractmethod
    def get_fused_point_cloud(self, lidar_id: str, start_frame: int, end_frame: int, frame_step: int) -> np.ndarray:
        """
        Given a [lidar_id], returns the point cloud associated with the lidar
        for all frames specified within the range of [start_frame, end_frame]
        and only taking every [frame_step] frame.

        Args:
            lidar_id (str): id of the lidar sensor
            start_frame (int): start range of the fused point cloud
            end_frame: (int): end_range of the fused point cloud
            frame_step (int): how many frames to skip per frame selection

        Returns:
            np.ndarray: Array of shape (N, 3) with xyz-points
        """
        ...

    @abstractmethod
    def get_fused_point_cloud_color(
        self, lidar_id: str, start_frame: int, end_frame: int, frame_step: int, type: str = "Intensity"
    ) -> np.ndarray:
        """
        Given a [lidar_id] and [type], returns the color for each point
        associated with the lidar for all frames specified within the
        range of [start_frame, end_frame] and only taking every
        [frame_step] frame.

        Args:
            lidar_id (str): id of the lidar sensor
            start_frame (int): start range of the fused point cloud
            end_frame: (int): end_range of the fused point cloud
            frame_step (int): how many frames to skip per frame selection
            type (str): controls what kind of data that is to be captured
                        within the color (e.g. Semantic, Intensity)

        Returns:
            np.ndarray : Array of shape (N, 3) with rgb values if [type] data
                         exists for [lidar_id], otherwise defaults to a color
        """
        ...

    @abstractmethod
    def get_cuboid_class_color(self, cuboid_class: str) -> np.ndarray:
        """
        Returns an rgb color associated with [cuboid_class].

        Args:
            cuboid_class (str): object class

        Returns:
            np.ndarray: A unique RGB color for each cuboid class
        """
        ...

    @abstractmethod
    def get_cuboid_data(self, lidar_id: str, frame: int, cuboid_source: str) -> np.ndarray | None:
        """
        Given a [lidar_id] and [frame], returns the cuboid data associated
        with the lidar at the given frame.

        Args:
            lidar_id (str): id of the lidar sensor
            frame (int): frame within range of the lidar data
            cuboid_source (str): source of the cuboid labels (e.g. "autolabels")

        Returns:
            np.ndarray | None: An array of bounding boxes if cuboid data
                               exists for [lidar_id], otherwise None
        """
        ...

    @abstractmethod
    def get_fused_cuboid_data(
        self, lidar_id: str, start_frame: int, end_frame: int, frame_step: int, cuboid_source: str
    ) -> np.ndarray | None:
        """
        Given a [lidar_id], returns the cuboid data associated with the lidar
        for all frames specified within the range of [start_frame, end_frame]
        and only taking every [frame_step] frame.

        Args:
            lidar_id (str): id of the lidar sensor
            start_frame (int): start range of the fused cuboid data
            end_frame: (int): end_range of the fused cuboid data
            frame_step (int): how many frames to skip per frame selection
            cuboid_source (str): source of the cuboid labels (e.g. "autolabels")

        Returns:
            np.ndarray | None: Array of bounding boxes if cudoid data exists
                               for [lidar_id], otherwise None
        """
        ...
