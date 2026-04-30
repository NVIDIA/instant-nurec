# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import ast
import json
import os

from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import OpenEXR
import torch

from PIL import Image
from tqdm import tqdm

import ncore.sensors

from internal.scripts.ds_to_ncore.ds_to_ncore_utils import (
    calculate_bbox,
    create_ncore_camera_model,
)
from ncore.data import ConcreteLidarModelParametersUnion, LabelSource
from ncore.impl.common.transformations import PoseInterpolator, is_within_3d_bboxes, se3_inverse, time_bounds
from ncore_internal.data.v3 import FrameLabel3, Poses, ShardDataWriter
from ncore_internal.impl.common.util import uniform_subdivide_range
from nre.utils.geometry import PoseLinearVelocityInterpolator
from nre.utils.ncore_utils import AuxDataWriter


# Limit the valid bbox classes for performance reasons
ALLOWED_BBOX_CLASSES = [
    "car",
    "bicycle",
    "cycle",
    "person",
    "automobile",
    "truck",
]

SEMANTIC_STUFF_COLORS = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [255, 255, 255],
]

SEMANTIC_STUFF_CLASSES = [
    "unknown",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "egocar",
]

DS_TO_STUFF_MAPPING = {
    "BACKGROUND": "sky",
    "UNLABELLED": "unknown",
    "vegetation_vertical": "vegetation",
    "automobile,prop_general": "car",
    "automobile": "car",
    "barrier": "fence",
    "prop_pole": "pole",
    "prop_pole_vertical": "pole",
    "road_mark_lane": "road",
    "prop_general": "unknown",
    "drivable": "road",
}


class DS2NCoreWriter:
    def __init__(
        self,
        input_dir: str,
        camera_ids: list,
        lidar_ids: list,
        timestamp_offset: int = 0,
        max_frames: int = -1,
        n_shards: int = 1,
    ):
        """Initialize the offline writer class.

        Args:
            input_dir (str): The path to the folder created by drivesim.
            camera_ids (list): The names of the cameras (can be a subset of the cameras in the drivesim data)
            lidar_ids (list): The names of the lidar sensors (can be a subset of the sensors in the drivesim data)
            timestamp_offset (int): The microsecond timestamp offset between the frame-by-frame timestamps
                (color images, poses, etc) and the pointcloud timestamps. Typically this offset is 66ms (66666us).
            max_frames (int): The maximum frames to copy into the shard. Default (-1) is no maximum.
            n_shards (int): The number of ncore shards to create
        """
        self._input_dir = input_dir
        self._camera_ids = camera_ids
        self._lidar_ids = lidar_ids
        self._timestamp_offset = timestamp_offset
        self._max_frames = max_frames
        self._shard_count = n_shards
        # car size in meters
        self._car_size = 3
        # threshold for dynamic objects
        self._track_min_speed_ms = 0.01
        # the maximum timestamp taken from
        self._max_camera_timestamp = 0
        self._tmp_dir = os.path.join("/tmp/ncore", input_dir)

        # Used for creating rig poses
        self._unique_rig_timestamps: List[float] = []
        self._unique_rig_transforms: np.ndarray = np.empty((1, 4, 4))

        self._cameras = {}
        self._camera_depth_factor: dict[str, np.ndarray] = {}
        for camera_id in camera_ids:
            camera_dir = os.path.join(self._input_dir, camera_id)

            # verify camera paths with some backwards compatibility
            if not os.path.exists(os.path.join(camera_dir, f"{camera_id}.json")):
                camera_dir = os.path.join(input_dir, camera_id)

            if not os.path.isdir(camera_dir):
                print(f"Error: Camera directory missing: {camera_dir}")
                raise FileNotFoundError

            camera_json_path = os.path.join(camera_dir, f"{camera_id}.json")
            if not os.path.exists(camera_json_path):
                print(f"Error: Camera json file missing: {camera_json_path}")
                raise FileNotFoundError

            self._cameras[camera_id] = {
                "container_dir": camera_dir,
                "json_path": camera_json_path,
            }

        self._cameras = {
            camera_id: {
                "container_dir": os.path.join(self._input_dir, camera_id),
                "json_path": os.path.join(self._input_dir, camera_id, f"{camera_id}.json"),
            }
            for camera_id in camera_ids
        }
        self._camera_masks: dict[str, Image.Image | None] = {camera_id: None for camera_id in camera_ids}
        for camera_id in camera_ids:
            if not os.path.exists(self._cameras[camera_id]["json_path"]):
                raise FileNotFoundError(f"Missing camera json file. Did you use the correct camera id?")
        self._lidars = {
            lidar_id: {
                "container_dir": os.path.join(self._input_dir, lidar_id),
                "json_path": os.path.join(self._input_dir, lidar_id, f"{lidar_id}.json"),
            }
            for lidar_id in lidar_ids
        }
        for lidar_id in lidar_ids:
            if not os.path.exists(self._lidars[lidar_id]["json_path"]):
                raise FileNotFoundError(f"Missing lidar json file. Did you use the correct lidar id?")
        self.bboxes: dict[str, dict] = {}

    def get_prim_velocity(self, frame_index: int, prim_path: str) -> float:
        """Get the prim velocity at a frame index

        Args:
            frame_index (int): Frame index
            prim_path (str): Prim path

        Returns:
            float: Velocity according to the vehicle dynamics.
        """
        for i, path in enumerate(self._vehicle_dynamics[frame_index]["primPaths"]):
            if prim_path == path:
                # temporary hack because primVelocities is always zero
                return float(np.linalg.norm(np.array(self._vehicle_dynamics[frame_index]["primAccelerations"][i])))
        return 0.0

    def read_sensor_data(self):
        """Read camera and lidar data from the drivesim output files and build a list of unique timestamps.

        To account for possible gaps in the data, we build a list of unique timestamps and their corresponding
        rig transforms.
        """
        self._camera_data = {}
        for camera_id in self._camera_ids:
            with open(self._cameras[camera_id]["json_path"]) as file:
                data = json.load(file)
                camera_params = data["camera_params"]
                # rig_worlds is from "egoTransform"
                t_rig_worlds = np.array(data["T_rig_worlds"][: self._max_frames], dtype=np.float32)
                # world_cameras is from "cameraViewTransform"
                t_world_cameras = np.array(data["T_world_cameras"][: self._max_frames], dtype=np.float32)
                # Timestamps from "reference_time"
                timestamps_us = np.array(data["timestamps"][: self._max_frames], dtype=np.uint64)

                self._camera_data[camera_id] = {
                    "camera_params": camera_params,
                    "t_rig_worlds": t_rig_worlds,
                    "t_world_cameras": t_world_cameras,
                    "timestamps_us": timestamps_us,
                }

            # add unique rig poses/timestamps
            if len(self._unique_rig_timestamps) == 0:
                self._unique_rig_timestamps = timestamps_us.tolist()
                self._unique_rig_transforms = t_rig_worlds.astype(np.float64)
            else:
                unique_rig_timestamps, indices = np.unique(
                    np.concatenate([self._unique_rig_timestamps, timestamps_us]), return_index=True
                )
                self._unique_rig_timestamps = unique_rig_timestamps.tolist()
                self._unique_rig_transforms = np.concatenate(
                    [self._unique_rig_transforms, t_rig_worlds.astype(np.float64)]
                )[indices].tolist()
            self._max_camera_timestamp = max(self._max_camera_timestamp, int(np.max(timestamps_us)))

        self._lidar_data = {}
        for lidar_id in self._lidar_ids:
            with open(self._lidars[lidar_id]["json_path"]) as file:
                # Poses and timestamps are all end-of-frame
                data = json.load(file)
                # Timestamps from "reference_time" and are offset from the per-point data.
                lidar_timestamps = np.array(data["timestamps"], dtype=np.uint64)
                if self._max_camera_timestamp > 0:
                    max_lidar_frames = int(np.sum(lidar_timestamps <= self._max_camera_timestamp))
                    if max_lidar_frames > len(lidar_timestamps):
                        print(f"Warning: Dropping {max_lidar_frames - len(lidar_timestamps)} lidar frames")
                else:
                    max_lidar_frames = self._max_frames
                lidar_timestamps = lidar_timestamps[:max_lidar_frames]
                # rig_worlds is from egoTransform
                t_rig_worlds = np.array(data["T_rig_worlds"][:max_lidar_frames], dtype=np.float32)
                # world_lidars is from cameraViewTransform
                t_world_lidars = np.array(data["T_world_lidars"][:max_lidar_frames], dtype=np.float32)
                self._lidar_data[lidar_id] = {
                    "t_rig_worlds": t_rig_worlds,
                    "t_world_lidars": t_world_lidars,
                    "timestamps_us": lidar_timestamps,
                    "profile_name": data["profile_name"] if "profile_name" in data else "",
                    "motion_compensated": data["motion_compensated"] if "motion_compensated" in data else False,
                }

                # add unique rig poses/timestamps
                if self._unique_rig_timestamps is None:
                    self._unique_rig_timestamps = lidar_timestamps.tolist()
                    self._unique_rig_transforms = t_rig_worlds.astype(np.float64)
                else:
                    unique_rig_timestamps, indices = np.unique(
                        np.concatenate([self._unique_rig_timestamps, lidar_timestamps]), return_index=True
                    )
                    self._unique_rig_timestamps = unique_rig_timestamps.tolist()
                    self._unique_rig_transforms = np.concatenate(
                        [self._unique_rig_transforms, t_rig_worlds.astype(np.float64)]
                    )[indices]

        assert isinstance(self._unique_rig_timestamps, list)
        # Duplicate the first pose for t=0 if there is no pose at t=0
        if np.min(np.array(self._unique_rig_timestamps)) > 0:
            self._unique_rig_timestamps = [0] + self._unique_rig_timestamps
            self._unique_rig_transforms = np.concatenate(
                [[self._unique_rig_transforms[0]], self._unique_rig_transforms], axis=0
            )

        # sort timestamps and poses
        indices = np.argsort(self._unique_rig_timestamps)
        self._unique_rig_timestamps = np.array(self._unique_rig_timestamps)[indices].tolist()
        self._unique_rig_transforms = self._unique_rig_transforms[indices]

    def write_itar(self, output_dir: str, run_id: str, mask_prim_path: Optional[str] = None):
        """Write the zarr.itar file containing camera and lidar data.

        The ego mask is generated with the assumption of a specific prim path for the ego car. If this was not correct,
        the writer will have failed to create a mask and it must be done with the semantic segmentation image and the
        prim path.

        Args:
            output_dir (str): Output directory
            run_id (str): Run ID used for the container name
            mask_prim_path (str, optional): Name of the ego prim for masking the ego car. Defaults to None.
        """
        self.read_sensor_data()

        vehicle_dynamics_path = os.path.join(self._input_dir, "vehicle_dynamics.json")
        with open(vehicle_dynamics_path) as file:
            self._vehicle_dynamics = json.load(file)

        global_T_rig_world_timestamps_us = np.array(self._unique_rig_timestamps, dtype=np.uint64)

        # Select reference base pose and convert all poses relative to this reference.
        T_rig_world_base = self._unique_rig_transforms[0]
        global_T_rig_worlds = np.linalg.inv(T_rig_world_base) @ self._unique_rig_transforms

        # make world transforms relative to base
        for camera_id in self._camera_ids:
            self._camera_data[camera_id]["t_rig_worlds"] = (
                np.linalg.inv(T_rig_world_base) @ self._camera_data[camera_id]["t_rig_worlds"]
            )
            self._camera_data[camera_id]["t_world_cameras"] = (
                self._camera_data[camera_id]["t_world_cameras"] @ T_rig_world_base
            )
        for lidar_id in self._lidar_ids:
            self._lidar_data[lidar_id]["t_rig_worlds"] = (
                np.linalg.inv(T_rig_world_base) @ self._lidar_data[lidar_id]["t_rig_worlds"]
            )
            self._lidar_data[lidar_id]["t_world_lidars"] = (
                self._lidar_data[lidar_id]["t_world_lidars"] @ T_rig_world_base
            )

        global_target_start_timestamps_us, global_target_end_timestamps_us = time_bounds(
            global_T_rig_world_timestamps_us.tolist(), 0, self._unique_rig_timestamps[-1]
        )
        global_range_start = int(np.argmax(global_T_rig_world_timestamps_us >= global_target_start_timestamps_us))
        global_range_stop = int(
            np.argmin(global_T_rig_world_timestamps_us < global_target_end_timestamps_us)
            if global_target_end_timestamps_us < global_T_rig_world_timestamps_us[-1]
            else len(global_T_rig_world_timestamps_us)
        )
        global_T_rig_worlds = global_T_rig_worlds[global_range_start:global_range_stop]
        global_T_rig_world_timestamps_us = global_T_rig_world_timestamps_us[global_range_start:global_range_stop]
        global_start_timestamp_us = global_T_rig_world_timestamps_us[0]
        global_end_timestamp_us = global_T_rig_world_timestamps_us[-1]

        assert global_start_timestamp_us >= global_T_rig_world_timestamps_us[0]
        assert global_end_timestamp_us <= global_T_rig_world_timestamps_us[-1]

        self.read_bbox_data()

        for shard_id in range(self._shard_count):
            aux_writer = AuxDataWriter(
                Path(output_dir),
                store_base_name=f"{run_id}_{shard_id}-{self._shard_count}",
                sequence_id=run_id,
                shard_id=shard_id,
                shard_count=self._shard_count,
                store_type="itar",
            )

            # Apply uniform subdivision for current shard to get local pose range with non-inclusive *single* pose **overlap**.
            # This guarantees that all frames can be associated with a unique shard (needs to be un-done when loading multi-shard sequences)
            #
            # Example
            # shard0_pose_timestamps = [0,1,2,3], valid pose-range-timestamps to select frame data [0,3) -> 0 <= t < 3
            # shard1_pose_timestamps = [3,4,5,6], valid pose-range-timestamps to select frame data [3,6) -> 3 <= t < 6
            local_range, _ = uniform_subdivide_range(
                shard_id,
                self._shard_count,
                # *skip* first global pose unconditionally from all local-ranges to be sure we can interpolate within initial frames,
                # so first shard's pose range in the example above will really be [1,3) -> 1 <= t < 3
                1,
                len(global_T_rig_world_timestamps_us),
            )

            # extend local range by single non-inclusive pose to keep in local shard
            local_range_start = local_range[0]
            local_range_stop = min(
                # non-extended local range end
                (local_range[-1] + 1)
                # extend by single additional non-inclusive pose
                + 1,
                len(global_T_rig_world_timestamps_us),
            )

            local_T_rig_worlds = global_T_rig_worlds[local_range_start:local_range_stop]
            local_T_rig_world_timestamps_us = global_T_rig_world_timestamps_us[local_range_start:local_range_stop]
            local_start_timestamp_us = local_T_rig_world_timestamps_us[0]
            local_end_timestamp_us = local_T_rig_world_timestamps_us[-1]

            print(
                f"shard {shard_id + 1}/{self._shard_count} | local_range_start {local_range_start} / local_range_stop {local_range_stop} | "
                f"{local_start_timestamp_us} <= t < {local_end_timestamp_us}"
            )

            assert local_start_timestamp_us >= global_start_timestamp_us
            assert local_end_timestamp_us <= global_end_timestamp_us  # note: global bounds are inclusive

            data_writer = ShardDataWriter(
                output_dir_path=Path(output_dir),
                container_name=f"{run_id}_{shard_id}-{self._shard_count}",
                camera_ids=self._camera_ids,
                lidar_ids=self._lidar_ids,
                radar_ids=[],
                calibration_type="ds-calibration",
                egomotion_type="ds-egomotion",
                sequence_id=run_id,
                generic_meta_data={},
                shard_id=shard_id,
                shard_count=self._shard_count,
                store_shard_meta=False,
            )

            for camera_id in self._camera_ids:
                self.write_camera_data(
                    data_writer,
                    camera_id,
                    local_start_timestamp_us,
                    local_end_timestamp_us,
                    mask_prim_path=mask_prim_path,
                )
                self.write_camera_aux_data(
                    aux_writer,
                    camera_id,
                    local_start_timestamp_us,
                    local_end_timestamp_us,
                )

            for lidar_id in self._lidar_ids:
                self.write_lidar_data(data_writer, lidar_id, local_start_timestamp_us, local_end_timestamp_us)

            # Save the poses
            data_writer.store_poses(
                Poses(
                    T_rig_world_base=T_rig_world_base,
                    T_rig_worlds=local_T_rig_worlds,
                    T_rig_world_timestamps_us=local_T_rig_world_timestamps_us,
                )
            )
            print(f"Saving shard {shard_id}.")
            data_writer.finalize()

            aux_writer.finalize()

    def write_camera_aux_data(
        self,
        aux_writer: AuxDataWriter,
        camera_id: str,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ):
        camera_model_parameters = create_ncore_camera_model(self._camera_data[camera_id]["camera_params"][0])
        resolution = camera_model_parameters.resolution.tolist()
        # DS Cameras are currently global shutter
        end_of_frame_timestamps_us = self._camera_data[camera_id]["timestamps_us"]
        camera_width = resolution[0]
        camera_height = resolution[1]

        # Semantic Segmentation
        semantic_segmentation_path = os.path.join(self._cameras[camera_id]["container_dir"], "semantic_segmentation")
        if os.path.isdir(semantic_segmentation_path):
            # Map the ncore class index to the ds color
            ds2ncore_mapping: dict[tuple, int] = {}
            for i in range(len(end_of_frame_timestamps_us)):
                # idToLabels has json data mapping colors to classes like so:
                # "(0, 0, 0, 0)": {
                #   "class": "BACKGROUND"
                # },
                # "(25, 255, 82, 255)": {
                #   "class": "automobile,prop_general"
                # },
                map_path = os.path.join(semantic_segmentation_path, f"idToLabels_{i:06}.json")
                with open(map_path) as file:
                    ids_to_labels = json.load(file)
                    for color_str, class_pair in ids_to_labels.items():
                        sem_class = DS_TO_STUFF_MAPPING.get(class_pair["class"], "unknown")
                        rgba = tuple(map(int, color_str.strip("()").split(",")))
                        class_idx = SEMANTIC_STUFF_CLASSES.index(sem_class)
                        if rgba not in ds2ncore_mapping:
                            ds2ncore_mapping[rgba] = class_idx
                        else:
                            assert ds2ncore_mapping[rgba] == class_idx

            aux_writer.store_semantic_meta(
                camera_id=camera_id,
                resolution=resolution,
                dataset_name="ds",
                method="ds-gt",
                pretrained_checkpoint="",
                stuff_classes=SEMANTIC_STUFF_CLASSES,
                stuff_colors=SEMANTIC_STUFF_COLORS,
            )

            for i in tqdm(range(len(end_of_frame_timestamps_us)), desc="Semantic Segmentation"):
                if (
                    end_of_frame_timestamps_us[i] < start_timestamp_us
                    or end_of_frame_timestamps_us[i] >= end_timestamp_us
                ):
                    continue
                img_path = os.path.join(semantic_segmentation_path, f"{i:06}.png")
                assert os.path.exists(img_path)
                semantic_seg_np = np.array(Image.open(img_path))
                index_img = np.zeros(semantic_seg_np.shape[:2], dtype=np.uint8)

                for rgba, class_idx in ds2ncore_mapping.items():
                    index_img[(semantic_seg_np == rgba).all(axis=-1)] = class_idx

                if (mask := self._camera_masks[camera_id]) is not None:
                    index_img[np.array(mask) == 255] = SEMANTIC_STUFF_CLASSES.index("egocar")

                semantic_seg = Image.fromarray(index_img, mode="P")
                semantic_seg.putpalette(np.linspace(0, 255, 256, dtype=np.uint8).tolist())

                aux_writer.store_semantic_segmentation(
                    camera_id=camera_id,
                    frame_timestamps_us=end_of_frame_timestamps_us[i],
                    semantic_seg=semantic_seg,
                    image_file_format="png",
                )

        # Depth from the writer may contain distance_to_image_plane (depth) or distance_to_camera
        depth_data_path = os.path.join(self._cameras[camera_id]["container_dir"], "distance_to_image_plane")
        distance_data_path = os.path.join(self._cameras[camera_id]["container_dir"], "distance_to_camera")
        convert_to_depth = False
        if not os.path.isdir(depth_data_path) and os.path.isdir(distance_data_path):
            depth_data_path = distance_data_path
            convert_to_depth = True

        if os.path.isdir(depth_data_path):
            # Limit the depth values from DS to something more reasonable.
            max_depth = 250.0
            aux_writer.store_depth_meta(
                camera_id=camera_id,
                resolution=resolution,
                store_depth_as_png=False,
                max_depth_m=max_depth,
                method="ds-gt",
            )

            if convert_to_depth and camera_id not in self._camera_depth_factor:
                # Precompute the factor that converts from distance to depth.
                camera_model = ncore.sensors.CameraModel.from_parameters(
                    camera_model_parameters, device="cuda", dtype=torch.float32
                )
                camera_pixels_x, camera_pixels_y = np.meshgrid(
                    np.arange(camera_width, dtype=np.int16), np.arange(camera_height, dtype=np.int16)
                )  # [0, w-1] x [0, h-1]
                camera_all_pixels = np.stack([camera_pixels_x.flatten(), camera_pixels_y.flatten()], axis=1)
                rays_camera = camera_model.pixels_to_camera_rays(camera_all_pixels)
                depth_factor = rays_camera[..., 2].cpu().numpy().reshape(camera_height, camera_width)
                # cache depth factor for future shards
                self._camera_depth_factor[camera_id] = depth_factor

            for i in tqdm(range(len(end_of_frame_timestamps_us)), desc="Depth Images"):
                if (
                    end_of_frame_timestamps_us[i] < start_timestamp_us
                    or end_of_frame_timestamps_us[i] >= end_timestamp_us
                ):
                    continue

                img_path = os.path.join(depth_data_path, f"{i:06}.exr")
                assert os.path.exists(img_path)

                with open(img_path, "rb") as f:
                    exr_file = OpenEXR.InputFile(f)
                    exr_dw = exr_file.header()["dataWindow"]
                    depth_map_array = np.frombuffer(exr_file.channel("Y"), np.float16).reshape(
                        exr_dw.max.y - exr_dw.min.y + 1, exr_dw.max.x - exr_dw.min.x + 1
                    )
                    depth_data = depth_map_array.astype(np.float32)
                    if convert_to_depth:
                        depth_data *= self._camera_depth_factor[camera_id]

                    depth_data[depth_data > max_depth] = np.inf
                    aux_writer.store_depth(camera_id, end_of_frame_timestamps_us[i], depth_data)

        # Normals
        normals_path = os.path.join(self._cameras[camera_id]["container_dir"], "normals")
        if os.path.isdir(normals_path):
            aux_writer.store_normal_meta(camera_id=camera_id, resolution=resolution, method="ds-gt")
            t_world_cameras = self._camera_data[camera_id]["t_world_cameras"]

            for i in tqdm(range(len(end_of_frame_timestamps_us)), desc="Normals"):
                if (
                    end_of_frame_timestamps_us[i] < start_timestamp_us
                    or end_of_frame_timestamps_us[i] >= end_timestamp_us
                ):
                    continue

                # There is no motion during an image capture, so all rotations are the same for each ray
                rotation_world_to_sensor = np.tile(t_world_cameras[i], (camera_width * camera_height, 1, 1))[:, :3, :3]

                img_path = os.path.join(normals_path, f"{i:06}.png")
                assert os.path.exists(img_path)

                # AuxWriter expects normals in the range of [-1, 1]. Convert from [0, 255].
                colorized_normals_image = Image.open(img_path)
                normals_world_image_array = np.array(colorized_normals_image)[:, :, :3] / 127.5 - 1.0

                # Rotate and normalize
                normals_world = normals_world_image_array.reshape((-1, 3))
                normals_sensor = np.matmul(rotation_world_to_sensor, normals_world[:, :, np.newaxis]).squeeze(-1)
                norms = np.linalg.norm(normals_sensor, axis=1)
                norms[norms < 0.1] = 1
                normalized = normals_sensor / norms[:, np.newaxis]
                normals_sensor_image_array = normalized.reshape((camera_height, camera_width, 3))

                aux_writer.store_normal(camera_id, end_of_frame_timestamps_us[i], normals_sensor_image_array)

    def write_camera_data(
        self,
        data_writer: ShardDataWriter,
        camera_id: str,
        local_start_timestamp_us: int,
        local_end_timestamp_us: int,
        img_format: str = "jpeg",
        mask_prim_path: Optional[str] = None,
    ):
        """Writes the camera data (images, poses, timestamps) to the itar

        Args:
            data_writer (ShardDataWriter): The dataset writer
            camera_id (str): Camera ID
            local_start_timestamp_us: The start timestamp of this shard
            local_end_timestamp_us: The end timestamp of this shard
            img_format (str, optional): Color image format. Defaults to "jpeg".
            mask_prim_path (str, optional): Prim path for ego car, if necessary. Defaults to None.

        Raises:
            ValueError: If the ego mask and mask prim path are missing.
            ValueError: If an ego mask prim was provided but there is not exactly 1 match.
        """

        t_rig_worlds = self._camera_data[camera_id]["t_rig_worlds"]
        t_world_cameras = self._camera_data[camera_id]["t_world_cameras"]
        timestamps_us = self._camera_data[camera_id]["timestamps_us"]

        t_cam_rig = se3_inverse(t_rig_worlds[0]) @ se3_inverse(t_world_cameras[0])
        # Convert to NCore camera coordinate system (x-right, y-down, z-out) from Drivesim (y-up, z-in)
        rot = t_cam_rig[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        t_cam_rig[:3, :3] = rot

        continuous_local_frame_index = 0
        for i in tqdm(range(len(timestamps_us)), desc="Color Images"):
            if timestamps_us[i] < local_start_timestamp_us or timestamps_us[i] >= local_end_timestamp_us:
                continue
            img_path = os.path.join(self._cameras[camera_id]["container_dir"], f"{i:06}.{img_format}")
            assert os.path.exists(img_path)

            with open(img_path, "rb") as file:
                img_data = file.read()

            data_writer.store_camera_frame(
                camera_id=camera_id,
                continuous_frame_index=continuous_local_frame_index,
                image_file_binary_data=img_data,
                image_file_format=img_format,
                T_rig_worlds=np.array([t_rig_worlds[i], t_rig_worlds[i]], dtype=np.float32),
                timestamps_us=np.array([timestamps_us[i], timestamps_us[i]], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )
            continuous_local_frame_index += 1

        mask_path = os.path.join(self._cameras[camera_id]["container_dir"], "ego_mask.png")
        if os.path.exists(mask_path):
            self._camera_masks[camera_id] = Image.open(mask_path)
        elif mask_prim_path is not None:
            # There is no mask image, but we can extract the mask from instance segmentation
            instance_segmentation_path = os.path.join(
                self._cameras[camera_id]["container_dir"], "instance_segmentation"
            )
            img_path = os.path.join(instance_segmentation_path, "000000.png")
            id_map_path = os.path.join(instance_segmentation_path, "idToLabels_000000.json")

            instance_seg_image = Image.open(img_path)
            instance_seg_data = np.array(instance_seg_image)

            with open(id_map_path) as file:
                ids_to_labels = json.load(file)

            ego_list = [k for k in ids_to_labels.keys() if ids_to_labels[k] == mask_prim_path]
            if len(ego_list) == 1:
                ego_color_str = ego_list[0]
                ego_color = np.array([int(v) for v in ego_color_str.strip("()").split(",")], dtype=np.uint8)
                self._camera_masks[camera_id] = Image.fromarray(np.all(instance_seg_data == ego_color, axis=-1))
            else:
                raise ValueError("Multiple ego prim paths found for ego mask")
        else:
            print(
                f"Warning: Missing mask image or mask prim path: {camera_id}. Ignore this warning if the ego is not visible from the camera."
            )
            self._camera_masks[camera_id] = None

        timestamps_us = timestamps_us[
            np.logical_and(timestamps_us >= local_start_timestamp_us, timestamps_us < local_end_timestamp_us)
        ]
        camera_model = create_ncore_camera_model(self._camera_data[camera_id]["camera_params"][0])

        print(f"Saving {len(timestamps_us)} camera frames for {camera_id}")
        data_writer.store_camera_meta(
            camera_id=camera_id,
            frame_timestamps_us=timestamps_us,
            T_sensor_rig=t_cam_rig.astype(np.float32),
            camera_model_parameters=camera_model,
            mask_image=self._camera_masks[camera_id],
            generic_meta_data={},
        )

    def read_bbox_data(self):
        T_world_base_rig = np.linalg.inv(self._unique_rig_transforms[0])

        # bboxes are collected at the same timestamps as camera images
        bbox_timestamps_us = self._camera_data[self._camera_ids[0]]["timestamps_us"]

        # cache bad prims to reduce warnings
        bad_prims = []

        for frame_index in tqdm(range(0, len(bbox_timestamps_us)), desc="Bounding Boxes"):
            # Use camera timestamps for bboxes
            bbox_timestamp = bbox_timestamps_us[frame_index]
            bbox_ids_path = os.path.join(self._input_dir, f"bbox/all_bbox_ids_{frame_index:06}.npy")
            bbox_data_path = os.path.join(self._input_dir, f"bbox/all_bbox_data_{frame_index:06}.npy")
            ids_to_labels_path = os.path.join(self._input_dir, f"bbox/ids_to_labels_{frame_index:06}.npy")
            prims_list_path = os.path.join(self._input_dir, f"bbox/prim_paths_{frame_index:06}.json")

            bbox_ids = np.load(bbox_ids_path)
            bbox_data = np.load(bbox_data_path)
            ids_to_labels = np.load(ids_to_labels_path, allow_pickle=True)
            ids_to_labels = ast.literal_eval(str(ids_to_labels))

            # prim paths of bboxes for checking velocities
            with open(prims_list_path) as file:
                prim_paths = json.load(file)

            for bbox_index in range(bbox_data.shape[0]):
                # bbox, from drivesim, is a numpy.void type with these fields:
                # class_id, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, T_BBox_World, -1
                # where p1 and p2 are the two untransformed corners of the bounding box
                bbox = bbox_data[bbox_index]
                bbox_class = bbox[0]
                bbox_transform = bbox[7]
                track_id = bbox_ids[bbox_index]
                # used for debugging bad bboxes
                prim_path = prim_paths[bbox_index]

                class_label = None
                if "class" in ids_to_labels[bbox_class]:
                    # if we have multiple valid labels, pick the first one
                    if "," in ids_to_labels[bbox_class]["class"]:
                        labels = ids_to_labels[bbox_class]["class"].split(",")
                        for label in labels:
                            if label in ALLOWED_BBOX_CLASSES:
                                class_label = label
                                break
                    elif ids_to_labels[bbox_class]["class"] in ALLOWED_BBOX_CLASSES:
                        class_label = ids_to_labels[bbox_class]["class"]

                if class_label is None:
                    continue

                # skip bad bboxes from drivesim. Only warn once.
                if np.any(np.isinf([bbox[i] for i in range(7)])):
                    if track_id not in bad_prims:
                        print(f"Warning: {prim_path} bbox contains inf")
                        bad_prims.append(track_id)
                    continue

                # Make world frame relative to starting rig transform (consistent with other sensors)
                t_bbox_world = T_world_base_rig @ np.reshape(bbox_transform, (4, 4)).transpose()
                if invalid_rotation := np.linalg.det(t_bbox_world) < 0.9:
                    if track_id not in bad_prims:
                        print(f"Warning: {prim_path} transform has bad rotation {invalid_rotation}")
                        bad_prims.append(track_id)
                    continue

                if not track_id in self.bboxes:
                    self.bboxes[track_id] = {
                        "class_label": class_label,
                        "timestamps_us": [],
                        "t_bbox_world": [],
                        "bbox": [],
                        "dynamic": False,
                    }
                self.bboxes[track_id]["bbox"].append(bbox)
                self.bboxes[track_id]["timestamps_us"].append(bbox_timestamp)
                self.bboxes[track_id]["t_bbox_world"].append(t_bbox_world)

        for track_id in self.bboxes.keys():
            poses = self.bboxes[track_id]["t_bbox_world"]
            timestamps_us = self.bboxes[track_id]["timestamps_us"]
            track_speeds_m_s = PoseLinearVelocityInterpolator(np.stack(poses), np.array(timestamps_us)).get_speeds_m_s(
                timestamps_us
            )
            self.bboxes[track_id]["dynamic"] = np.median(track_speeds_m_s) > self._track_min_speed_ms

    def read_lidar_model_parameters(self, lidar_id: str) -> ConcreteLidarModelParametersUnion:
        # Currently this supports only the Hesai_P128_V4P5_HR10.json
        # Number of raw points is 921600 = 3600 columns x 128 channels x 2 returns per point
        # We filter out one of the two returns by using the one with the higher intensity

        # The lidar parameters are parsed from the lidar profile json file in drivesim
        lidar_profile = os.path.join(
            self._lidars[lidar_id]["container_dir"], self._lidar_data[lidar_id]["profile_name"] + ".json"
        )
        if not os.path.exists(lidar_profile):
            assert FileExistsError("Lidar Profile json missing. Copy from DS build directory.")

        with open(lidar_profile, "r") as file:
            profile = json.load(file)["profile"]

        # TODO: Extract n_columns from parameters
        n_columns = 3600

        # TODO: Add support for Example_Rotary? It does not have "channels"
        n_rows = profile["numberOfChannels"]
        spinning_frequency_hz = int(profile["scanRateBaseHz"])

        spinning_direction = profile["rotationDirection"].lower()
        assert spinning_direction in ["cw", "ccw"]

        # Each state contains a subset of the azimuths/elevations, but combined we get the whole picture
        state_count = profile["emitterStateCount"]
        states = profile["emitterStates"]
        azimuth_elevation_rad = np.zeros((n_rows, 2), dtype=np.float32)
        for s in range(state_count):
            azimuth = np.array(states[s]["azimuthDeg"]) * np.pi / 180.0
            elevation = np.array(states[s]["elevationDeg"]) * np.pi / 180.0
            # channelId is 1 indexed
            channel_ids = np.array(states[s]["channelId"]) - 1
            azimuth_elevation_rad[channel_ids, 0] = -azimuth
            azimuth_elevation_rad[channel_ids, 1] = elevation

        # sort in descending order
        sorted_indices = np.argsort(-azimuth_elevation_rad[:, 1])
        azimuth_elevation_rad[:, 0] = azimuth_elevation_rad[sorted_indices, 0]
        azimuth_elevation_rad[:, 1] = azimuth_elevation_rad[sorted_indices, 1]

        if spinning_direction == "cw":
            column_azimuths_rad = np.linspace(np.pi, -np.pi, n_columns + 1, dtype=np.float32)[:n_columns]
        else:
            column_azimuths_rad = np.linspace(-np.pi, np.pi, n_columns + 1, dtype=np.float32)[:n_columns]

        return ConcreteLidarModelParametersUnion(
            spinning_direction=spinning_direction,
            row_elevations_rad=azimuth_elevation_rad[:, 1],
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=azimuth_elevation_rad[:, 0],
            spinning_frequency_hz=spinning_frequency_hz,
            n_rows=n_rows,
            n_columns=n_columns,
        )

    def write_lidar_data(
        self,
        data_writer: ShardDataWriter,
        lidar_id: str,
        local_start_timestamp_us: int,
        local_end_timestamp_us: int,
    ):
        """Write the Lidar data to the itar

        Notes/Issues:
        - The writer often gives one extra frame of data which appears at the beginning, so here we skip the first frame.
        - Timestamps for the lidar poses are offset from the per-point timestamps in the pointcloud data by a given offset
        (defaulting to 66666us, or the time of 2 camera frames).
        - Pointcloud data from drivesim is in the frame of reference of the lidar as it was scanned. Here we motion compensate
        and transform the pointcloud data to the lidar's reference frame at the end of the frame.
        - Bounding boxes must also be motion compensated in a similar way.
        - Because points do not have object IDs, we must rely on the bounding boxes to determine if a point is dynamic or not.
        This is costly (many point-box intersection checks) if there are too many boxes.
        - Bounding boxes also might not contain all the points from a dynamic object, especially if the object is moving quickly
        and appears near the start/end of the Lidar scan boundary.

        Args:
            data_writer (ShardDataWriter): The dataset writer
            lidar_id (str): Lidar ID
            local_start_timestamp_us: The start timestamp of this shard
            local_end_timestamp_us: The end timestamp of this shard
        """
        lidar_end_times = []
        t_rig_worlds = self._lidar_data[lidar_id]["t_rig_worlds"]
        t_world_lidar_0 = self._lidar_data[lidar_id]["t_world_lidars"][0]
        lidar_timestamps_us = self._lidar_data[lidar_id]["timestamps_us"]

        # t_world_lidars is actually not correct and needs to be rotated
        fix = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        t_lidar_rig = se3_inverse(t_rig_worlds[0]) @ se3_inverse(t_world_lidar_0) @ fix

        # assume that the ego has not moved during the first frame
        t_rig_world_interpolator = PoseInterpolator(
            np.concatenate([[t_rig_worlds[0]], t_rig_worlds]), np.concatenate([[0], lidar_timestamps_us])
        )

        local_range_start = np.argmax(lidar_timestamps_us >= local_start_timestamp_us)
        local_range_stop = (
            np.argmin(lidar_timestamps_us < local_end_timestamp_us)
            if lidar_timestamps_us[-1] > local_end_timestamp_us
            else len(lidar_timestamps_us)
        )
        local_frame_numbers = list(range(local_range_start, local_range_stop))

        print(
            f"Lidar frames for shard: | local_range_start {local_range_start} / local_range_stop {local_range_stop} | "
            f"{local_start_timestamp_us} <= t < {local_end_timestamp_us}"
        )

        continuous_local_frame_index = 0
        lidar_model_parameters = self.read_lidar_model_parameters(lidar_id)

        for frame_index in tqdm(local_frame_numbers, desc="Lidar Data"):
            frame_end_timestamp_us = lidar_timestamps_us[frame_index]

            data_path = os.path.join(self._lidars[lidar_id]["container_dir"], f"{lidar_id}_{frame_index:06}.npy")
            if not os.path.exists(data_path):
                # Sometimes the first or last lidar frame might be missing, largely due to timestamp offsets or incomplete frames
                continue

            # offline pc_data looks like this:
            # np.column_stack((xyz_e, intensity, per_point_timestamps_us, offsets, valid, channel_id))
            pc_data = np.load(data_path)
            assert pc_data.shape[1] == 8

            # The lidar model supports a single return per beam. If we get 2 (as we do from drivesim), choose the beam
            # with the higher intensity.
            if lidar_model_parameters is not None:
                points_per_frame = lidar_model_parameters.n_rows * lidar_model_parameters.n_columns
                # cut off extra data
                pc_data = pc_data[: points_per_frame * 2, :]
                intensity = pc_data[:, 3].astype(np.float32)
                if pc_data.shape[0] == points_per_frame * 2:
                    selection = np.full(intensity.shape, True)
                    selection[::2] = intensity[::2] > intensity[1::2]
                    selection[1::2] = np.logical_not(selection[::2])
                    intensity = intensity[selection]
                    pc_data = pc_data[selection, :]
            else:
                intensity = pc_data[:, 3].astype(np.float32)

            # The lidar points as they are detected (not motion compensated)
            xyz_lidars = np.concatenate(
                [pc_data[:, :3].astype(np.float32), np.ones([pc_data.shape[0], 1]).astype(np.float32)], axis=-1
            )
            per_point_timestamps_us = pc_data[:, 4].astype(np.uint64)
            valid = pc_data[:, 6]

            valid_indices = np.logical_and(
                valid,
                np.logical_and(
                    per_point_timestamps_us >= self._timestamp_offset,
                    per_point_timestamps_us <= np.max(lidar_timestamps_us) + self._timestamp_offset,
                ),
            )

            if np.sum(valid_indices) == 0:
                print(f"Warning: no valid point times for lidar frame {frame_index}")
                continue

            per_point_timestamps_us = per_point_timestamps_us[valid_indices] - self._timestamp_offset
            xyz_lidars = xyz_lidars[valid_indices]
            intensity = intensity[valid_indices]

            lidar_frame_time_us = 100000
            # Skip frames that start before 0
            if frame_end_timestamp_us < lidar_frame_time_us:
                print(f"Skipping frame {frame_index} which starts before 0 and ends at {frame_end_timestamp_us}")
                continue

            frame_start_timestamp_us = frame_end_timestamp_us - lidar_frame_time_us
            lidar_end_times.append(frame_end_timestamp_us)

            # Compute the lidar-to-world transform for each point
            t_rig_world_per_point = t_rig_world_interpolator.interpolate_to_timestamps(per_point_timestamps_us)
            t_lidar_world_per_point = t_rig_world_per_point @ t_lidar_rig

            # Compute the rig-to-world at start & end-of-frame transforms
            t_rig_world_start = t_rig_world_interpolator.interpolate_to_timestamps(
                np.array([frame_start_timestamp_us])
            )[0]
            t_rig_world_end = t_rig_world_interpolator.interpolate_to_timestamps(np.array([frame_end_timestamp_us]))[0]

            # Compute the world-to-lidar transform at end-of-frame
            t_world_lidar_end = se3_inverse(t_lidar_rig) @ se3_inverse(t_rig_world_end)

            # Motion compensated in world frame
            xyz_world = (t_lidar_world_per_point @ xyz_lidars[:, :, None]).squeeze(-1)

            # Motion compensated in sensor frame (at end-of-frame)
            xyz_e = (t_world_lidar_end @ xyz_world.T).T[:, :3].astype(np.float32)
            xyz_s = (t_world_lidar_end @ t_lidar_world_per_point @ np.array([0.0, 0.0, 0.0, 1.0]))[:, :3].astype(
                np.float32
            )

            # Filter out points in the ego car
            non_ego_points = np.linalg.norm(xyz_s - xyz_e, axis=1) > self._car_size
            valid_indices = np.argwhere(valid_indices).flatten()[non_ego_points]
            xyz_e = xyz_e[non_ego_points]
            xyz_s = xyz_s[non_ego_points]
            intensity = intensity[non_ego_points]
            per_point_timestamps_us = per_point_timestamps_us[non_ego_points]

            # Use bounding boxes to calculate dynamic flags
            dynamic_flag = np.zeros(per_point_timestamps_us.shape, dtype=np.int8)

            frame_labels = []

            # # Object Ids are all zeros for Hesai Lidar
            # object_id_path = os.path.join(self._input_dir, f"{lidar_id}/{lidar_id}_objects_{i-start_frame:06}.npy")
            # object_ids = np.load(object_id_path)

            for track_id, bbox_data in self.bboxes.items():
                class_label = bbox_data["class_label"]
                bbox_interp = PoseLinearVelocityInterpolator(
                    np.array(bbox_data["t_bbox_world"]),
                    np.array(bbox_data["timestamps_us"]),
                )
                # skip bboxes without timestamps during this lidar frame
                if (
                    bbox_data["timestamps_us"][0] > frame_end_timestamp_us
                    or bbox_data["timestamps_us"][-1] < frame_start_timestamp_us
                ):
                    continue

                # Find nearest bbox in case this changes over time
                bbox = bbox_data["bbox"][-1]
                for nearest_index in range(0, len(bbox_data["timestamps_us"])):
                    if bbox_data["timestamps_us"][nearest_index] >= frame_end_timestamp_us:
                        bbox = bbox_data["bbox"][nearest_index]
                        break

                t_bbox_world = bbox_interp.interpolate_to_timestamps([frame_end_timestamp_us])[0]
                t_lidar_bbox = t_world_lidar_end @ t_bbox_world
                frame_bbox, bbox_tensor = calculate_bbox(bbox, t_lidar_bbox)

                # Skip car bbox
                if np.linalg.norm(frame_bbox.centroid) < self._car_size:
                    continue

                # Re-interpolate the bbox based on the lidar points, so the box will nicely match the lidar points
                pc_intersection_indices = is_within_3d_bboxes(xyz_e, bbox_tensor.reshape(1, -1)).squeeze(-1)
                valid_timestamps = per_point_timestamps_us[pc_intersection_indices]
                if valid_timestamps.shape[0] == 0:
                    continue

                bbox_mean_timestamp_us = int(np.mean(valid_timestamps))
                t_bbox_world = bbox_interp.interpolate_to_timestamps(np.array([bbox_mean_timestamp_us]))[0]
                t_lidar_bbox = t_world_lidar_end @ t_bbox_world
                frame_bbox, bbox_tensor = calculate_bbox(bbox, t_lidar_bbox)

                # get the velocity of this bbox in m/s
                bbox_velocity_ms = float(np.linalg.norm(bbox_interp.get_velocities_m_s([bbox_mean_timestamp_us])[0]))
                dynamic_flag[pc_intersection_indices] = bbox_data["dynamic"]

                frame_labels.append(
                    FrameLabel3(
                        label_id=f"label_{frame_index}-{track_id}",
                        track_id=f"track_{track_id}",
                        label_class=class_label,
                        bbox3=frame_bbox,
                        global_speed=bbox_velocity_ms,
                        confidence=None,
                        timestamp_us=bbox_mean_timestamp_us,
                        source=LabelSource.GT_SYNTHETIC,
                    )
                )

            model_element = None
            if lidar_model_parameters is not None:
                model_element = np.stack(
                    np.meshgrid(
                        np.arange(lidar_model_parameters.n_rows, dtype=np.uint16),
                        np.arange(lidar_model_parameters.n_columns, dtype=np.uint16),
                        indexing="xy",
                    ),
                    axis=-1,
                )
                model_element = model_element.reshape(-1, 2)
                model_element = model_element[valid_indices, :]

            data_writer.store_lidar_frame(
                lidar_id=lidar_id,
                continuous_frame_index=continuous_local_frame_index,
                xyz_s=xyz_s,
                xyz_e=xyz_e,
                intensity=intensity,
                timestamp_us=per_point_timestamps_us,
                frame_labels=frame_labels,
                T_rig_worlds=np.array([t_rig_world_start, t_rig_world_end], dtype=np.float32),
                timestamps_us=np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64),
                model_element=model_element,
                generic_data={
                    "dynamic_flag": dynamic_flag,
                },
                generic_meta_data={},
            )
            continuous_local_frame_index += 1

        print(f"Saving {len(lidar_end_times)} lidar frames for {lidar_id}")
        data_writer.store_lidar_meta(
            lidar_id=lidar_id,
            frame_timestamps_us=np.array(lidar_end_times, dtype=np.uint64),
            T_sensor_rig=t_lidar_rig.astype(np.float32),
            lidar_model_parameters=lidar_model_parameters,
            generic_meta_data={},
        )


@click.command("ds_to_ncore")
@click.option(
    "--input-dir",
    type=str,
    help="Drivesim data output by the NCoreWriter",
    required=True,
)
@click.option(
    "--run-id",
    type=str,
    help="Name of the run, to be used when naming the .zarr.itar file",
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Parent directory where the output will be saved",
    required=True,
)
@click.option(
    "--camera-ids",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be used (multiple value option, separate with commas)",
    default=None,
)
@click.option(
    "--lidar-ids",
    "lidar_ids",
    multiple=True,
    type=str,
    help="Lidars to be used (multiple value option, separate with commas)",
    default=None,
)
@click.option(
    "--n-shards",
    type=int,
    help="Number of shards to create.",
    default=1,
    required=False,
)
@click.option(
    "--mask-prim-path",
    type=str,
    help="The prim path to use as the ego car mask. Often /Entities/Ego",
    default=None,
    required=False,
)
@click.option(
    "--lidar-timestamp-offset",
    type=int,
    # This value is 2 frames = 2x33ms. It is currently hardcoded in drivesim and not exposed to the writers.
    # If the framerate changes, this offset must also be updated!
    help="The offset between lidar frame timestamps and point-wise timestamps (default 66666us = 2x33ms)",
    default=66666,
    required=False,
)
@click.option(
    "--max-frames",
    type=int,
    help="Max number of frames to save.",
    default=10000,
    required=False,
)
def ds_to_ncore(
    input_dir: str,
    run_id: str,
    output_dir: str,
    camera_ids: list[str],
    lidar_ids: list[str],
    n_shards: int,
    mask_prim_path: str,
    lidar_timestamp_offset: int,
    max_frames: int,
) -> None:
    # This enables us to pass comma separated values for camera and lidar ids.
    if len(camera_ids) == 1 and "," in camera_ids[0]:
        camera_ids = camera_ids[0].split(",")
    if len(lidar_ids) == 1 and "," in lidar_ids[0]:
        lidar_ids = lidar_ids[0].split(",")

    offline_writer = DS2NCoreWriter(
        input_dir,
        camera_ids=list(camera_ids),
        lidar_ids=list(lidar_ids),
        timestamp_offset=lidar_timestamp_offset,
        max_frames=max_frames,
        n_shards=n_shards,
    )

    offline_writer.write_itar(output_dir, run_id=run_id, mask_prim_path=mask_prim_path)


if __name__ == "__main__":
    ds_to_ncore()
