# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np

import ncore.data

from internal.scripts.ncore_vis.data_utils import Camera
from internal.scripts.ncore_vis.loader import NCoreLoader
from ncore.impl.common.transformations import bbox_pose, transform_bbox, transform_point_cloud
from nre.utils.visualize import scalar2img


class NCoreLazyLoader(NCoreLoader):
    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path)
        self._default_point_cloud_color = np.array([255, 0, 0], dtype=np.uint8)
        self._cuboid_class_colors: dict[str, np.ndarray] = {}
        self._cuboid_dtype = [("bbox", "O"), ("pose", "O"), ("class", "O"), ("confidence", "O"), ("track_id", "O")]

    def get_trajectories(self) -> np.ndarray:
        sequence_poses = self.loader.get_poses()
        return sequence_poses.T_rig_worlds

    def get_camera(self, camera_id: str) -> Camera:
        camera = self.loader.get_camera_sensor(camera_id)
        return Camera(camera)

    def get_camera_image(self, camera_id: str, frame: int) -> np.ndarray:
        camera = self.loader.get_camera_sensor(camera_id)
        return camera.get_frame_image_array(frame)

    def get_camera_semantic_image(self, camera_id: str, frame: int) -> np.ndarray | None:
        semantic_meta = self.get_camera_semantic_segmentation_meta(camera_id)
        if semantic_meta is None:
            return None

        color_map = np.array(semantic_meta["stuff_colors"], dtype=np.uint8)
        camera = self.loader.get_camera_sensor(camera_id)
        timestamp = camera.get_frame_timestamp_us(frame)
        segmentation = np.asarray(self.aux_loader.get_semantic_segmentation(camera_id, timestamp))
        return color_map[segmentation]

    def get_camera_depth_image(self, camera_id: str, frame: int) -> np.ndarray | None:
        depth_meta = self.get_camera_depth_meta(camera_id)
        if depth_meta is None:
            return None

        camera = self.loader.get_camera_sensor(camera_id)
        timestamp = camera.get_frame_timestamp_us(frame)
        depth = self.aux_loader.get_depth(camera_id, timestamp)
        depth_image = scalar2img(
            depth,
            vmin=0.0,
            vmax=75,  # TODO: make configurable
        )

        return depth_image

    def get_camera_normals_image(self, camera_id: str, frame: int) -> np.ndarray | None:
        if self.get_camera_normals_meta(camera_id) is None:
            return None

        camera = self.loader.get_camera_sensor(camera_id)
        timestamp = camera.get_frame_timestamp_us(frame)
        normals = self.aux_loader.get_normal(camera_id, timestamp)
        colorized_normals = np.clip((normals + 1.0) * 127.0, 0, 255).astype(np.uint8)
        return colorized_normals

    def get_camera_semantic_overlay_image(self, camera_id: str, frame: int) -> np.ndarray | None:
        semantic_meta = self.get_camera_semantic_segmentation_meta(camera_id)
        if semantic_meta is None:
            return None

        color_map = np.array(semantic_meta["stuff_colors"], dtype=np.uint8)
        camera = self.loader.get_camera_sensor(camera_id)

        camera_image = camera.get_frame_image_array(frame)
        timestamp = camera.get_frame_timestamp_us(frame)
        segmentation = np.asarray(self.aux_loader.get_semantic_segmentation(camera_id, timestamp))
        semantic = color_map[segmentation]

        alpha = 0.5
        overlayed_image = (camera_image * (1 - alpha) + semantic * alpha).astype(np.uint8)
        return overlayed_image

    def get_point_cloud(self, lidar_id: str, frame: int) -> np.ndarray:
        sensor = self.loader.get_lidar_sensor(lidar_id)
        T_sensor_target = sensor.get_frame_T_sensor_world(frame)
        point_cloud = transform_point_cloud(sensor.get_frame_data(frame, "xyz_e"), T_sensor_target)
        return point_cloud

    def get_point_cloud_color(self, lidar_id: str, frame: int, type: str = "Intensity") -> np.ndarray:
        sensor = self.loader.get_lidar_sensor(lidar_id)
        point_count = sensor.get_frame_data(frame, "xyz_e").shape[0]
        match type:
            case "Semantic":
                semantic_meta = self.get_lidar_semantic_segmentation_meta(lidar_id)
                if not semantic_meta:
                    return np.full((point_count, 3), self._default_point_cloud_color, dtype=np.uint)
                color_map = np.array(semantic_meta["stuff_colors"], dtype=np.uint8)
                ignore_label = semantic_meta["ignore_label"]
                semantics = self.aux_loader.get_lidar_semantic_segmentation(
                    lidar_id, sensor.get_frame_timestamp_us(frame)
                )
                # Set invalid semantics (ignore label) to default color
                valid_labels = semantics != ignore_label
                labelmap = np.empty((semantics.shape[0], color_map.shape[1]), dtype=np.uint8)
                labelmap[valid_labels] = color_map[semantics[valid_labels]]
                labelmap[~valid_labels] = np.zeros((3,), dtype=np.uint8)
                return labelmap
            case "Intensity" | "Intensity γ=1/4":
                try:
                    intensity = sensor.get_frame_data(frame, "intensity")
                except Exception:
                    return np.full((point_count, 3), self._default_point_cloud_color, dtype=np.uint)
                intensity_colormap = np.repeat(intensity[:, np.newaxis], 3, axis=1) * 255
                intensity_colormap = intensity_colormap.astype(np.uint8)
                return (
                    intensity_colormap
                    if type == "Intensity"
                    else np.clip((intensity_colormap + 1) * 4, 0, 255).astype(np.uint8)
                )
            case "RGB":
                try:
                    rgb = sensor.get_frame_generic_data(frame, "rgb")
                    assert isinstance(rgb, np.ndarray) and rgb.shape == (point_count, 3) and rgb.dtype == np.uint8
                except Exception:
                    return np.full((point_count, 3), self._default_point_cloud_color, dtype=np.uint)
                return rgb
            case _:
                return self._default_point_cloud_color

    def get_fused_point_cloud(self, lidar_id: str, start_frame: int, end_frame: int, frame_step: int = 1) -> np.ndarray:
        fused_point_cloud = []
        for frame in range(start_frame, end_frame + 1, frame_step):
            fused_point_cloud.append(self.get_point_cloud(lidar_id, frame))
        return np.concatenate(fused_point_cloud)

    def get_fused_point_cloud_color(
        self, lidar_id: str, start_frame: int, end_frame: int, frame_step: int = 1, type: str = "Intensity"
    ) -> np.ndarray:
        fused_point_cloud_colors = []
        for frame in range(start_frame, end_frame + 1, frame_step):
            fused_point_cloud_colors.append(self.get_point_cloud_color(lidar_id, frame, type))
        return np.concatenate(fused_point_cloud_colors)

    def get_cuboid_class_color(self, cuboid_class: str) -> np.ndarray:
        if cuboid_class in self._cuboid_class_colors:
            return self._cuboid_class_colors[cuboid_class]
        self._cuboid_class_colors[cuboid_class] = np.random.randint(0, 255, (3), dtype=np.uint8)
        return self._cuboid_class_colors[cuboid_class]

    def get_cuboid_data(self, lidar_id: str, frame: int, cuboid_source: str) -> np.ndarray | None:
        sensor = self.loader.get_lidar_sensor(lidar_id)
        frame_labels = sensor.get_frame_labels(frame)
        if len(frame_labels) == 0:
            return None

        cuboid_data_list = []
        T_sensor_target = sensor.get_frame_T_sensor_world(frame)
        for label in frame_labels:
            # skip label if not from the enabled label source
            if label.source.name != cuboid_source:
                continue

            bbox = transform_bbox(label.bbox3.to_array(), T_sensor_target)
            bbox_pos = bbox_pose(bbox)
            cuboid_data = np.array(
                [(bbox, bbox_pos, label.label_class, label.confidence, label.track_id)],
                dtype=self._cuboid_dtype,
            )
            cuboid_data_list.append(cuboid_data)

        return np.vstack(cuboid_data_list) if len(cuboid_data_list) else None

    def get_fused_cuboid_data(
        self,
        lidar_id: str,
        start_frame: int,
        end_frame: int,
        frame_step: int = 1,
        cuboid_source: str = ncore.data.LabelSource._member_names_[0],
    ) -> np.ndarray | None:
        fused_cuboid_data = []
        for frame in range(start_frame, end_frame + 1, frame_step):
            cuboid_data = self.get_cuboid_data(lidar_id, frame, cuboid_source)
            if cuboid_data is None:
                continue
            fused_cuboid_data.append(cuboid_data)

        if len(fused_cuboid_data) == 1:
            return fused_cuboid_data[0]

        return np.concatenate(fused_cuboid_data) if len(fused_cuboid_data) > 1 else None
