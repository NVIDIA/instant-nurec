# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import concurrent.futures

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto, unique
from types import EllipsisType
from typing import Dict, Generator, List, Literal, NamedTuple, Optional, TypeVar, cast, overload
from weakref import WeakValueDictionary

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import PIL.Image as PILImage
import simplejpeg
import torch
import torch.utils.data
import tqdm

from cuml.cluster import DBSCAN
from scipy import ndimage
from upath import UPath

import ncore
import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import ncore.sensors
import ncore_internal.impl.common.transformations as ncore_internal_transformations
import ncore_internal.impl.nvidia.rig as ncore_internal_rig
import nre.utils.ncore_utils as ncore_utils

from libs.vren.interface import (  # type: ignore
    image_points_to_world_rays_shutter_pose,
    world_points_to_image_points_shutter_pose,
)
from nre.config.dataset import NCoreDatasetConfig
from nre.config.nre import NREConfig
from nre.config.sensor import LidarModelsConfig
from nre.datasets.base import BaseDataset, BaseDataSource, RigTrajectoriesProvider
from nre.datasets.registry import register as register_dataset
from nre.datasets.samplers.base import (
    BaseBatchSampler,
    BaseCameraPixelSampler,
    BaseFrameSampler,
    BaseLidarPointSampler,
    BaseSensorSampler,
    SkipCameraPixelSampler,
    SkipFrameSampler,
    SkipLidarPointSampler,
    SkipSensorSampler,
)
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.datasets.utils import (
    PackedMask,
    classify_dynamic_cuboid_tracks,
    color_pc_semantics,
    compute_camera_visible_intervals_cuboid_tracks,
    compute_cameras_valid_pixels_frame_mask,
    compute_cuboid_tracks,
    compute_points_outside_tracks,
    compute_valid_lidarpoints_all,
    compute_valid_lidarpoints_trafficlight_cameravisible,
    compute_valid_pixels_ego,
    compute_valid_pixels_sceneflow,
    compute_valid_pixels_trafficlight,
    get_indices_of_points_visible_in_image,
    lidar_frame_dynamic_flag,
    transform_cuboid_track_observations,
    visualise_point_cloud,
)
from nre.utils.batch import (
    CameraFrameLabels,
    CameraFreePoseViewGeometry,
    ConcreteCameraModelsUnion,
    ConcreteLidarModelsUnion,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    LidarFrameLabels,
    RectSubsampled,
    compute_pixel_footprint,
    generate_grid_2d_indices,
)
from nre.utils.geometry import (
    se3_matrix_to_tquat,
    tquat_to_se3_matrix,
)
from nre.utils.lidar_model import LidarModelBundle, get_lidar_model_parameters_with_fallbacks
from nre.utils.misc import (
    assert_default_device_on_local_rank,
    assert_device_on_local_rank,
    compute_process_local_rng_seed,
    map_optional,
    set_default_device,
    to_torch,
    unpack_optional,
)
from nre.utils.profiling import ScopedTimer, TimingTag
from nre.utils.types import (
    AABB3D,
    CameraFrustum,
    FrameConversion,
    HalfClosedInterval,
    NovelViewOverrides,
    PointCloud,
    PointCloudColorType,
    RayFlags,
    RigTrajectories,
    TrackPointCloud,
)
from nre.visualdebugger import get_visualdebugger


class UniqueSensorId(NamedTuple):
    """Represents a unique sensor ID along with its unique index for a given sensor type"""

    id: str
    idx: int


# Stores already instantiated NCOREDataSources to allow sharing between val/train samplers.
# We use WeakValueDictionary so that when there are no references to a value, it is evicted
# from a dictionary. This means that as long as an NCORE{Train,Sequential}Dataset exists and holds
# a reference to an NCOREDataSource, it remains discoverable for reuse via its config hash.
# Once all samplers are dropped, the data source will be evicted from this cache, avoiding
# potential memory leaks.
# This somewhat hacky caching mechanism is to be replaced with caching on LightningDataModule
# instance later.
DATASOURCE_CACHE: WeakValueDictionary[NCoreDatasetConfig, NCOREDataSource] = WeakValueDictionary()


@register_dataset("ncore")
def ncore_dataset_factory(config: NREConfig, split: str = "train") -> NCOREDataset:
    """
    A factory method which creates the appropriate sampler depending on split parameter while caching
    the underlying data source.
    """
    dataset_config = NCoreDatasetConfig.model_validate(config.dataset)

    if dataset_config not in DATASOURCE_CACHE:
        source = NCOREDataSource(
            dataset_config, config.sensor.lidar_models
        )  # bind to variable to avoid drop between creation and use
        DATASOURCE_CACHE[dataset_config] = source
    else:
        source = DATASOURCE_CACHE[dataset_config]

    match split:
        case "train":
            return NCORETrainDataset(source, dataset_config, split)
        case "val":
            return NCORESequentialDataset(source, dataset_config, split)
        case "train-sequential":
            return NCORESequentialDataset(source, dataset_config, split)
        case _:
            raise ValueError(f"Unexpected {split=}.")


class NCOREDataset(BaseDataset):
    """Abstract base class for functionality common to train and val splits"""

    def __init__(self, datasource: NCOREDataSource, config: NCoreDatasetConfig, split: str):
        self.datasource = datasource
        self.datasource._maybe_init_worker()
        self.camera_ids = datasource.train_camera_ids if "train" in split else datasource.val_camera_ids
        self.lidar_ids = datasource.train_lidar_ids if "train" in split else datasource.val_lidar_ids
        self.use_real_lidar_rays = config.use_real_lidar_rays

        # We do not create rays in the dataset, so we don't need to
        # create the rig modules unless we are using depth data or normals
        has_depth = (aux_loader := datasource.aux_loader) is not None and aux_loader.has_depth()
        has_normal = aux_loader is not None and aux_loader.has_normal()
        if has_depth or has_normal:
            rig_trajectories = self.datasource.get_rig_trajectories()
            self.camera_rig_module = CameraFreePoseViewGeometry.from_rig_trajectories(
                rig_trajectories, interp_with_rig=config.interp_with_rig
            )

    def get_datasource(self) -> NCOREDataSource:
        self.datasource._maybe_init_worker()
        return self.datasource

    def get_camera_data_batch(
        self,
        sensor_id: str,
        sensor_frame_idx: int,
        sampled_pixels: RectSubsampled | None = None,
    ) -> DataBatch.Camera:
        """
        This method returns a `DataBatch` based on the query criteria:

        - sensor_id: The sensor id.
        - sensor_frame_idx: The underlying "raw" index of the ncore sensor.
        - sampled_pixels: The crop/resize information.

        Note this function is designed with the assumption there is only a single sequence & a single chunk in the dataset.
        """
        datasource = self.datasource
        datasource._maybe_init_worker()

        camera_sensor = datasource.camera_sensors[sensor_id]
        frame_timestamps_startend_us = (
            camera_sensor.get_frame_timestamp_us(sensor_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START),
            camera_sensor.get_frame_timestamp_us(sensor_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END),
        )

        # fetch the camera labels
        with ScopedTimer("get_frame_image/NCOREDataset.get_camera_data_batch", TimingTag.DATALOADER):
            # decode image data on CPU
            frame_image_array = datasource.decode_image_cpu(
                camera_sensor.get_frame_handle(sensor_frame_idx),
                # perform subsampling while decoding, if required
                subsample_factor=map_optional(sampled_pixels, lambda x: x.subsample_factor),
            )

        # Note: In the case of a non-None subsampled rect we also need to
        # - adjust the camera intrinsics (see `RectSubsampled.apply_to_model`)
        # - adjust the rolling shutter effects (see `RectSubsampled.apply_to_camera_rolling_shutter`)
        # But we won't do them here to keep the batch structure simple and clean. Instead we will carry the
        # subsample information in the camera_meta, and handle these when they are needed.
        if sampled_pixels is not None:
            # we only need to crop the decoded image array, as it was subsampled during decoding already
            frame_image_array = sampled_pixels.crop_array(frame_image_array)

        height, width, _ = frame_image_array.shape
        unique_frame_idx = (
            datasource.camera_linear_start_frame_indices[sensor_id]
            + sensor_frame_idx
            - datasource.camera_frame_ranges[sensor_id].start
        )
        unique_sensor_idx = datasource.camera_unique_ids[sensor_id].idx

        # save the info to the batch
        batch = DataBatch.Camera(
            meta=[
                FrameMeta(
                    unique_sensor_idx=unique_sensor_idx,
                    unique_frame_idx=unique_frame_idx,
                    subsample=sampled_pixels,
                )
            ],
            labels=CameraFrameLabels(
                rgb=to_torch(frame_image_array.astype(np.float32) / 255.0, device="cpu").unsqueeze(0),
            ),
        )

        # handling flags
        # TODO(ruilong): The name of "RayFlags" should be updated since we move away from the ray-based batch format.
        flags = torch.full(
            (height, width, 1),
            RayFlags.RGB_LABEL.value,
            dtype=torch.int32,
            device="cpu",
        )

        # store invalid flag of pixels (usually only required in validation mode)
        valid_mask = datasource.cameras_frame_valid_pixels_masks[sensor_id][sensor_frame_idx].unpacked()[
            ..., np.newaxis
        ]
        if sampled_pixels is not None:
            valid_mask = valid_mask.astype(np.float32)
            valid_mask = sampled_pixels.apply_to_array(valid_mask, interpolation=cv2.INTER_LINEAR)
            valid_mask = valid_mask > 0.5
        flags[~valid_mask] |= RayFlags.INVALID

        # handing aux data (and update flags)
        aux_loader = datasource.aux_loader
        if aux_loader is not None:
            # load semantic segmentation image
            semantics = np.asarray(aux_loader.get_semantic_segmentation(sensor_id, frame_timestamps_startend_us[1]))[
                ..., np.newaxis
            ]
            if sampled_pixels is not None:
                semantics = sampled_pixels.apply_to_array(semantics, interpolation=cv2.INTER_NEAREST)

            # classify sampled rays for sky and road
            sky_mask = semantics == datasource.sensor_sky_class_ids[sensor_id]
            road_mask = semantics == datasource.sensor_road_class_ids[sensor_id]
            vehicle_mask = np.logical_or.reduce(
                [semantics == class_id for class_id in datasource.sensor_vehicle_classes_ids[sensor_id]]
            )

            # update ray types
            flags |= RayFlags.VALID_SEMANTIC.value
            flags[sky_mask] |= RayFlags.SKY_SEMANTIC.value
            flags[road_mask] |= RayFlags.ROAD_SEMANTIC.value
            flags[vehicle_mask] |= RayFlags.VEHICLE_SEMANTIC.value

            # Note(ruilong): We probably don't need it anymore since we have set the flags.
            batch.labels.semantic = to_torch(semantics, device="cpu").unsqueeze(0)

            # TODO: Add a config option that specifies that we need depth and/or normals
            if aux_loader.has_depth() or aux_loader.has_normal():
                # Get full resolution
                image_width, image_height = datasource.camera_model_parameters[sensor_id].resolution

                pixels = unpack_optional(sampled_pixels)
                meta = batch.meta[0]
                sensor_model = cast(ConcreteCameraModelsUnion, self.camera_rig_module.get_sensor_model(meta))

                # Only compute these once. They are need by both normals and depth
                elements, sensor_rays = self.camera_rig_module._compute_elements_and_sensor_rays(
                    sensor_model, torch.device("cpu")
                )

                if aux_loader.has_depth():
                    # Load dense distance supervision from aux-data
                    depth = aux_loader.get_depth(
                        sensor_id, frame_timestamps_startend_us[1], (image_width, image_height)
                    )[..., np.newaxis]
                    depth = pixels.apply_to_array(depth, interpolation=cv2.INTER_NEAREST)

                    distance = torch.abs(
                        to_torch(depth[..., 0], device="cpu") / sensor_rays[..., 2]
                    )  # convert z-axis depth into distance along rays
                    distance = torch.reshape(distance, (height, width, 1))

                    match aux_loader.get_depth_meta(sensor_id)["method"]:
                        case "DepthAnythingV2":
                            # TODO: This is a relative depth estimate, but we can use Lidar to scale it to metric space
                            batch.labels.relative_distance = (
                                datasource.world_to_nre.target_scale * distance
                            ).unsqueeze(0)
                        case "MetricDepthAnythingV2" | "ds-gt" | "sauron-z-depth":
                            batch.labels.metric_distance = (datasource.world_to_nre.target_scale * distance).unsqueeze(
                                0
                            )
                        case _:
                            raise ValueError(
                                "Depth estimation method must be one of [DepthAnythingV2, MetricDepthAnythingV2]"
                            )

                if aux_loader.has_normal():
                    # Load normal supervision from aux-data
                    normal = aux_loader.get_normal(
                        sensor_id, frame_timestamps_startend_us[1], (image_width, image_height)
                    )  # shape: [image_height, image_width, 3]
                    normal = np.reshape(pixels.apply_to_array(normal, interpolation=cv2.INTER_LINEAR), (-1, 3, 1))

                    pose_and_timestamps_startend_return = self.camera_rig_module.get_poses_and_timestamps_startend(meta)
                    T_sensor_world_startend = pose_and_timestamps_startend_return.T_sensor_world_startend
                    timestamps_startend_us = pose_and_timestamps_startend_return.timestamps_startend_us

                    world_rays_return = sensor_model.pixels_to_world_rays_shutter_pose(
                        elements.reshape(-1, 2),
                        T_sensor_world_startend[0],
                        T_sensor_world_startend[1],
                        start_timestamp_us=cast(int, timestamps_startend_us[0].item()),
                        end_timestamp_us=cast(int, timestamps_startend_us[1].item()),
                        camera_rays=sensor_rays.reshape(-1, 3),
                        return_timestamps=False,
                        return_T_sensor_worlds=True,
                    )

                    # T_sensor_worlds transforms from sensor to world, normals are transformed with this
                    # transformation to convert from sensor to world space
                    rotation_sensor_to_world = unpack_optional(world_rays_return.T_sensor_worlds).numpy()[:, :3, :3]
                    normal_world = np.matmul(rotation_sensor_to_world, normal).squeeze(-1)

                    # normalize the normals for valid pixels
                    norms = np.linalg.norm(normal_world, axis=1)

                    valid_normal_mask = norms > 0.1
                    norms[~valid_normal_mask] = 1
                    batch.labels.normals = to_torch(normal_world / norms[:, np.newaxis], device="cpu").reshape(
                        (1, height, width, 3)
                    )

                    flags[np.reshape(valid_normal_mask, (height, width, 1))] |= RayFlags.VALID_NORMAL.value

        batch.labels.flags = flags.unsqueeze(0)
        _ = batch.labels.n_valid_rgb  # force computation of n_valid_rgb at data batch loading time
        _ = batch.labels.n_valid  # force computation of n_valid at data batch loading time
        _ = batch.labels.n_difixed  # force computation of n_difixed at data batch loading time
        _ = batch.labels.n_valid_bg  # force computation of n_valid_bg at data batch loading time
        return batch

    def get_lidar_data_batch(
        self,
        sensor_id: str,
        sensor_frame_idx: int,
        sampled_pixels: RectSubsampled | np.ndarray | EllipsisType = Ellipsis,
        return_index: int = 0,
    ) -> DataBatch.Lidar:
        """
        This method returns a `DataBatch` based on the query criteria:

        - sensor_id: The sensor id.
        - sensor_frame_idx: The underlying "raw" index of the ncore sensor.
        - pixel_samples: The crop/resize information. (Not supported for now)
        - return_index: The per-frame return index of the lidar sensor (for multi-return lidars).

        Note this function is designed with the assumption there is only a single sequence & a single chunk in the dataset.
        """
        if isinstance(sampled_pixels, RectSubsampled):
            raise NotImplementedError("RectSubsampled subsampling for lidar is not supported for now")

        datasource = self.datasource
        datasource._maybe_init_worker()

        lidar_sensor = datasource.lidar_sensors[sensor_id]
        lidar_model = unpack_optional(
            datasource.lidar_models[sensor_id],
            msg=f"NCOREDataSource: lidar model parameters are mandatory for this datasource but not available for '{sensor_id}'",
        )

        height = lidar_model.n_rows
        width = lidar_model.n_columns
        unique_frame_idx = (
            datasource.lidar_linear_start_frame_indices[sensor_id]
            + sensor_frame_idx
            - datasource.lidar_frame_ranges[sensor_id].start
        )

        model_elements: npt.NDArray[np.uint16] = unpack_optional(
            lidar_sensor.get_frame_ray_bundle_model_element(sensor_frame_idx),
            msg="Lidar sensor must provide model elements",
        )
        _intensity = lidar_sensor.get_frame_ray_bundle_return_intensity(sensor_frame_idx, return_index=return_index)[
            ..., np.newaxis
        ]  # [N,1]

        if not self.use_real_lidar_rays:
            # default fast case: directly infer ray distances without heavy transformations (fast for both V3 and V4 data)
            _distance_ncore = lidar_sensor.get_frame_ray_bundle_return_distance_m(
                frame_index=sensor_frame_idx, return_index=return_index
            )[..., np.newaxis]  # [N,1]

            # scale measured distances from ncore -> nre units if necessary
            if (target_scale := datasource.world_to_nre.target_scale) != 1.0:
                _distance = target_scale * _distance_ncore
            else:
                _distance = _distance_ncore

            # empty stubs for this code path [for type-checking only]
            _rays = np.empty((0, 6), dtype=np.float32)
            _rays_timestamps_us = np.empty((0,), dtype=np.uint64)
        else:
            # Fallback: compute the rays from the lidar points
            # (note: this is slow for V4 data as motion-compensation needs to be applied)
            T_sensor_world_end = datasource.world_to_nre.transform_poses(
                lidar_sensor.get_frames_T_sensor_target(
                    "world",
                    sensor_frame_idx,
                    frame_timepoint=ncore.data.FrameTimepoint.END,
                )
            )
            pc = lidar_sensor.get_frame_point_cloud(
                frame_index=sensor_frame_idx,
                motion_compensation=True,
                with_start_points=True,
                return_index=return_index,
            )
            _xyz_s = ncore_transformations.transform_point_cloud(unpack_optional(pc.xyz_m_start), T_sensor_world_end)
            _xyz_e = ncore_transformations.transform_point_cloud(pc.xyz_m_end, T_sensor_world_end)

            _distance = np.linalg.norm(_xyz_e - _xyz_s, axis=1, keepdims=True)

            _xyz_v = _xyz_e - _xyz_s  # vector between start -> end points
            _rays = np.concatenate([_xyz_s, _xyz_v / _distance.clip(min=1e-6)], axis=-1)
            _rays_timestamps_us = lidar_sensor.get_frame_ray_bundle_timestamp_us(sensor_frame_idx)

        if isinstance(sampled_pixels, np.ndarray):
            _intensity = _intensity[sampled_pixels]
            _distance = _distance[sampled_pixels]
            model_elements = model_elements[sampled_pixels]
            if self.use_real_lidar_rays:
                _rays = _rays[sampled_pixels]
                _rays_timestamps_us = _rays_timestamps_us[sampled_pixels]

        intensity = np.full((height, width, 1), fill_value=np.nan, dtype=np.float32)
        intensity[rows := model_elements[:, 0], cols := model_elements[:, 1], :] = _intensity
        distance = np.full((height, width, 1), fill_value=np.nan, dtype=np.float32)
        distance[rows, cols, :] = _distance
        raydrop = np.ones((height, width, 1), dtype=np.float32)
        raydrop[rows, cols, :] = 0.0  # set 0 for non-dropped rays

        sparse_rays = None
        sparse_timestamps = None
        sparse_elements = None
        if self.use_real_lidar_rays:
            sparse_rays = _rays
            sparse_timestamps = _rays_timestamps_us[:, np.newaxis]
            sparse_elements = model_elements

        unique_sensor_idx = datasource.lidar_unique_ids[sensor_id].idx

        # save the info to the batch
        batch = DataBatch.Lidar(
            meta=[
                FrameMeta(
                    unique_sensor_idx=unique_sensor_idx,
                    unique_frame_idx=unique_frame_idx,
                    subsample=None,  # Note(ruilong): This is not supported for lidar for now.
                )
            ],
            labels=LidarFrameLabels(
                distance=to_torch(distance, device="cpu").unsqueeze(0),
                intensity=to_torch(intensity, device="cpu").unsqueeze(0),
                raydrop=to_torch(raydrop, device="cpu").unsqueeze(0),
                sparse_rays=map_optional(sparse_rays, lambda x: to_torch(x, device="cpu").unsqueeze(0)),
                sparse_timestamps=map_optional(
                    sparse_timestamps, lambda x: to_torch(x, device="cpu", dtype=torch.int64).unsqueeze(0)
                ),
                sparse_elements=map_optional(
                    sparse_elements, lambda x: to_torch(x, device="cpu", dtype=torch.int64).unsqueeze(0)
                ),
            ),
        )

        # handling flags (for all the valid elements)
        _flags = torch.zeros((len(model_elements),), dtype=torch.int32, device="cpu")
        # store invalid flag of rays if per-frame mask exists (usually only required in validation mode)
        valid_mask = torch.from_numpy(
            datasource.lidars_frame_valid_points_masks[sensor_id][sensor_frame_idx].unpacked()
        )
        if isinstance(sampled_pixels, np.ndarray):
            valid_mask = valid_mask[sampled_pixels]
        _flags[~valid_mask] |= RayFlags.INVALID

        # load semantic labels, if available, and update flags
        aux_loader = datasource.aux_loader
        if aux_loader is not None:
            if aux_loader.has_lidar_semantic_segmentation():
                # load semantic segmentation image
                semantics = aux_loader.get_lidar_semantic_segmentation(
                    sensor_id, lidar_sensor.get_frame_timestamp_us(sensor_frame_idx)
                )[sampled_pixels]

                _flags |= RayFlags.VALID_SEMANTIC.value
                sky_mask = semantics == datasource.sensor_sky_class_ids[sensor_id]
                _flags[sky_mask] |= RayFlags.SKY_SEMANTIC.value
                road_mask = semantics == datasource.sensor_road_class_ids[sensor_id]
                _flags[road_mask] |= RayFlags.ROAD_SEMANTIC.value
                vehicle_mask = np.logical_or.reduce(
                    [semantics == class_id for class_id in datasource.sensor_vehicle_classes_ids[sensor_id]]
                )
                _flags[vehicle_mask] |= RayFlags.VEHICLE_SEMANTIC.value

        # full resolution flags, initialized with dropped rays flags.
        flags = torch.full(
            (height, width, 1),
            RayFlags.DROPPED.value,
            dtype=torch.int32,
            device="cpu",
        )
        flags[rows, cols, :] = _flags[..., None]  # set flags for non-dropped rays
        # Optionally ignore some rows of the lidar points by setting them to invalid.
        if self.datasource.lidar_ignore_rows:
            flags[self.datasource.lidar_ignore_rows, :, :] |= RayFlags.INVALID.value
        batch.labels.flags = flags.unsqueeze(0)
        _ = batch.labels.n_valid_lidar  # force computation of n_valid_lidar at data batch loading time
        return batch


class NCORESequentialDataset(NCOREDataset):
    """Encapsulates the logic for sampling frames sequentially"""

    def __init__(self, datasource: NCOREDataSource, config: NCoreDatasetConfig, split: str):
        super().__init__(datasource, config, split)
        assert config.name == "ncore"

        # store relevant parameters from config
        self.n_image_subsample: int

        camera_frame_step: int | None
        camera_exclude_frame_step: int | None
        self.camera_frames: dict[str, list[int]] = {}

        lidar_frame_step: int | None
        lidar_exclude_frame_step: int | None
        self.lidar_frames: dict[str, list[int]] = {}
        self.return_lidar: bool
        self.return_camera: bool

        if split == "val":
            self.n_image_subsample = config.n_val_image_subsample
            camera_frame_start = config.val_camera_frame_start
            camera_frame_step = config.val_camera_frame_step
            camera_exclude_frame_start = config.val_camera_exclude_frame_start
            camera_exclude_frame_step = config.val_camera_exclude_frame_step
            lidar_frame_start = config.val_lidar_frame_start
            lidar_frame_step = config.val_lidar_frame_step
            lidar_exclude_frame_start = config.val_lidar_exclude_frame_start
            lidar_exclude_frame_step = config.val_lidar_exclude_frame_step
            self.return_lidar = config.val_lidar
            self.return_camera = config.val_camera
        elif split == "train-sequential":
            self.n_image_subsample = config.n_train_sequential_image_subsample
            camera_frame_start = config.train_sequential_camera_frame_start
            camera_frame_step = config.train_sequential_camera_frame_step
            camera_exclude_frame_start = config.train_sequential_camera_exclude_frame_start
            camera_exclude_frame_step = config.train_sequential_camera_exclude_frame_step
            lidar_frame_start = config.train_sequential_lidar_frame_start
            lidar_frame_step = config.train_sequential_lidar_frame_step
            lidar_exclude_frame_start = config.train_sequential_lidar_exclude_frame_start
            lidar_exclude_frame_step = config.train_sequential_lidar_exclude_frame_step
            self.return_lidar = config.train_sequential_lidar
            self.return_camera = config.train_sequential_camera
        else:
            raise ValueError(f"Unexpected {split=}.")

        assert (camera_frame_step or camera_exclude_frame_step) and (
            (not camera_frame_step) or (not camera_exclude_frame_step)
        ), f"{self.__class__.__name__} Exactly one of camera_frame_step or camera_exclude_frame_step must be specified"

        if camera_frame_step is not None:
            assert camera_frame_start is not None, "camera_frame_start must be specified"
            for camera_id in self.camera_ids:
                self.camera_frames[camera_id] = list(
                    datasource.camera_frame_ranges[camera_id][camera_frame_start::camera_frame_step]
                )
        else:
            assert camera_exclude_frame_start is not None, "camera_exclude_frame_start must be specified"
            assert camera_exclude_frame_step is not None, "camera_exclude_frame_step must be specified"
            for camera_id in self.camera_ids:
                excluded_frames = set(
                    datasource.camera_frame_ranges[camera_id][camera_exclude_frame_start::camera_exclude_frame_step]
                )
                self.camera_frames[camera_id] = [
                    i for i in datasource.camera_frame_ranges[camera_id] if i not in excluded_frames
                ]

        assert (lidar_frame_step or lidar_exclude_frame_step) and (
            (not lidar_frame_step) or (not lidar_exclude_frame_step)
        ), f"{self.__class__.__name__} Exactly one of lidar_frame_step or lidar_exclude_frame_step must be specified"

        if lidar_frame_step is not None:
            assert lidar_frame_start is not None, "lidar_frame_start must be specified"
            for lidar_id in self.lidar_ids:
                self.lidar_frames[lidar_id] = list(
                    datasource.lidar_frame_ranges[lidar_id][lidar_frame_start::lidar_frame_step]
                )
        else:
            assert lidar_exclude_frame_start is not None, "lidar_exclude_frame_start must be specified"
            assert lidar_exclude_frame_step is not None, "lidar_exclude_frame_step must be specified"
            for lidar_id in self.lidar_ids:
                excluded_frames = set(
                    datasource.lidar_frame_ranges[lidar_id][lidar_exclude_frame_start::lidar_exclude_frame_step]
                )
                self.lidar_frames[lidar_id] = [
                    i for i in datasource.lidar_frame_ranges[lidar_id] if i not in excluded_frames
                ]

        # initialize optional extrinsic overwrites
        self.sensor_transl_delta_m: None | npt.NDArray = None
        self.T_camera_delta_rot: None | npt.NDArray = None
        self.T_lidar_delta_rot: None | npt.NDArray = None

        sensor_transl_delta_m_array: None | npt.NDArray = None
        sensor_rot_delta_deg_array: None | npt.NDArray = None
        self.val_sensor_transl_delta_m = config.val_sensor_transl_delta_m
        self.val_sensor_rot_delta_deg = config.val_sensor_rot_delta_deg
        if split == "val":
            if self.val_sensor_transl_delta_m is not None:
                sensor_transl_delta_m_array = np.array(self.val_sensor_transl_delta_m, dtype=np.float32)
            if self.val_sensor_rot_delta_deg is not None:
                sensor_rot_delta_deg_array = np.array(self.val_sensor_rot_delta_deg, dtype=np.float32)

        self.sensor_transl_delta_m, self.T_camera_delta_rot, self.T_lidar_delta_rot = (
            self.parse_sensor_transl_rot_delta(sensor_transl_delta_m_array, sensor_rot_delta_deg_array)
        )

    def parse_sensor_transl_rot_delta(
        self, sensor_transl_delta_m: npt.NDArray | None, sensor_rot_delta_deg: npt.NDArray | None
    ) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None]:
        parsed_sensor_transl_delta_m: npt.NDArray | None = None
        parsed_T_camera_delta_rot: npt.NDArray | None = None
        parsed_T_lidar_delta_rot: npt.NDArray | None = None
        if sensor_transl_delta_m is not None:
            assert sensor_transl_delta_m.shape == (3,), (
                f"{self.__class__.__name__}: Expecting 3d translation offset vector in 'val_sensor_transl_delta_m'"
            )
            parsed_sensor_transl_delta_m = np.array(sensor_transl_delta_m, dtype=np.float32)
        if sensor_rot_delta_deg is not None:
            sensor_rot_delta_deg_array = np.array(sensor_rot_delta_deg, dtype=np.float32)
            assert sensor_rot_delta_deg_array.shape == (3,), (
                f"{self.__class__.__name__}: Expecting three rotation Euler angles vector in 'val_sensor_rot_delta_deg'"
            )
            # precompute validation extrinsic delta rotation once (these are specified relative to the
            # rig frame so that they are always consistent for differently posed cameras)
            parsed_T_camera_delta_rot = (
                self.datasource.T_CAMERA_RIG.transpose()
                @ (
                    T_sensor_delta_rot := ncore_internal_transformations.euler_2_so3(
                        sensor_rot_delta_deg_array, degrees=True, seq="xyz"
                    )
                )
                @ self.datasource.T_CAMERA_RIG
            )
            parsed_T_lidar_delta_rot = T_sensor_delta_rot
        return parsed_sensor_transl_delta_m, parsed_T_camera_delta_rot, parsed_T_lidar_delta_rot

    def get_max_num_rays_per_train_sample(self) -> int:
        raise ValueError(
            f"{self.__class__.__name__}: get_max_num_rays_per_train_sample should only be called during training"
        )

    def __len__(self) -> int:
        """Returns the total number of samples provided by the dataset (depending on parametrization)"""

        self.datasource._maybe_init_worker()

        # sum of all per-camera / per-lidar frame-ranges
        size = 0
        if self.return_camera:
            size += sum(len(self.camera_frames[camera_id]) for camera_id in self.camera_ids)
        if self.return_lidar:
            size += sum(len(self.lidar_frames[lidar_id]) for lidar_id in self.lidar_ids)

        return size

    def get_item_novel_view_overrides(
        self, tuple_idx: tuple[int, Optional[NovelViewOverrides]]
    ) -> DataAndRenderingBatch:
        """
        Returns a specific sample of the dataset (depending on split type and parametrization)

        Args:
            tuple_idx: A tuple of (batch_idx, novel_view_overrides).

        Note that we use a single argument as this is supposed to be used with a pytorch DataLoader/Sampler.
        """

        self.datasource._maybe_init_worker()
        sequence_id = self.datasource.sequence_id

        batch_idx, novel_view_overrides = tuple_idx

        # parse optional extrinsic overwrites
        if novel_view_overrides is not None:
            transl_delta_m, T_camera_delta_rot, T_lidar_delta_rot = self.parse_sensor_transl_rot_delta(
                novel_view_overrides.transl_delta_m, novel_view_overrides.rot_delta_deg
            )
        else:
            transl_delta_m = self.sensor_transl_delta_m
            T_camera_delta_rot = self.T_camera_delta_rot
            T_lidar_delta_rot = self.T_lidar_delta_rot

        run_frames = 0
        if self.return_camera:
            for camera_id in self.camera_ids:
                n_camera_frames = len(self.camera_frames[camera_id])
                if batch_idx >= run_frames + n_camera_frames:
                    # current camera depleted, check next one
                    run_frames += n_camera_frames
                    continue

                # determine frame of current camera
                camera_frame_index = self.camera_frames[camera_id][batch_idx - run_frames]

                camera_model = self.datasource.camera_models[camera_id]
                camera_width, camera_height = (
                    int(camera_model.resolution[0].item()),
                    int(camera_model.resolution[1].item()),
                )

                T_offset = np.eye(4, dtype=np.float32)
                # apply optional rotation deltas (specified relative to rig frame)
                if T_camera_delta_rot is not None:
                    T_offset[:3, :3] = T_camera_delta_rot

                # translation deltas relative to the rig are simple linear offsets
                if transl_delta_m is not None:
                    T_offset[:3, 3] = transl_delta_m

                T_rig_nre = self.datasource.world_to_nre.transform_poses(
                    self.datasource.camera_sensors[camera_id].get_frames_T_source_target(
                        source_node="rig",
                        target_node="world",
                        frame_indices=camera_frame_index,
                        frame_timepoint=None,  # evaluates both start/end timepoints
                    )  # 2x4x4
                )

                T_offset_nre_startend = np.stack(
                    [
                        T_rig_nre[0] @ T_offset @ np.linalg.inv(T_rig_nre[0]),
                        T_rig_nre[1] @ T_offset @ np.linalg.inv(T_rig_nre[1]),
                    ]
                )
                del T_rig_nre, T_offset

                data_camera_batch = self.get_camera_data_batch(
                    camera_id,
                    camera_frame_index,
                    RectSubsampled(
                        i=0,
                        j=0,
                        height=camera_height // self.n_image_subsample,
                        width=camera_width // self.n_image_subsample,
                        subsample_factor=float(self.n_image_subsample),
                        original_width=camera_width,
                        original_height=camera_height,
                    ),
                )

                if T_camera_delta_rot is not None or transl_delta_m is not None:
                    # Pose will be modified, so set all labels to INVALID flag
                    if data_camera_batch.labels.flags is not None:
                        data_camera_batch.labels.flags |= RayFlags.INVALID.value

                # Add the extrinsic override information to the frame meta
                # [TODO]: Move the computation of the extrinsic overrides to torch once v1 is retired
                data_camera_batch.meta[0].T_offset_nre_startend = to_torch(
                    T_offset_nre_startend, device="cpu", dtype=torch.float32
                )

                data_batch = DataBatch(
                    idx=batch_idx,
                    worker_id=None,
                    sequence_id=[sequence_id],
                    lidar=None,
                    camera=data_camera_batch,
                )

                # Only includes the DataBatch as rendering batch will be initialized in the calibration module on GPU
                return DataAndRenderingBatch(data=data_batch, rendering=None)

        if self.return_lidar != "off":
            for lidar_id in self.lidar_ids:
                n_lidar_frames = len(self.lidar_frames[lidar_id])
                if batch_idx >= run_frames + n_lidar_frames:
                    # current lidar depleted, check next one
                    run_frames += n_lidar_frames
                    continue

                # determine frame of current camera
                lidar_frame_index = self.lidar_frames[lidar_id][batch_idx - run_frames]

                data_lidar_batch = self.get_lidar_data_batch(
                    sensor_id=lidar_id,
                    sensor_frame_idx=lidar_frame_index,
                )

                if T_lidar_delta_rot is not None or transl_delta_m is not None:
                    # Pose will be modified, so set all labels to INVALID flag
                    if data_lidar_batch.labels.flags is not None:
                        data_lidar_batch.labels.flags |= RayFlags.INVALID.value

                T_offset = np.eye(4, dtype=np.float32)
                # apply optional rotation deltas (specified relative to rig frame)
                if T_lidar_delta_rot is not None:
                    T_offset[:3, :3] = T_lidar_delta_rot

                # translation deltas relative to the rig are simple linear offsets
                if transl_delta_m is not None:
                    T_offset[:3, 3] = transl_delta_m

                T_rig_nre = self.datasource.world_to_nre.transform_poses(
                    self.datasource.lidar_sensors[lidar_id].get_frames_T_source_target(
                        source_node="rig",
                        target_node="world",
                        frame_indices=lidar_frame_index,
                        frame_timepoint=None,  # evaluates both start/end timepoints
                    )  # 2x4x4
                )

                T_offset_nre_startend = np.stack(
                    [
                        T_rig_nre[0] @ T_offset @ np.linalg.inv(T_rig_nre[0]),
                        T_rig_nre[1] @ T_offset @ np.linalg.inv(T_rig_nre[1]),
                    ]
                )
                del T_rig_nre, T_offset

                data_batch = DataBatch(
                    idx=batch_idx,
                    worker_id=None,
                    sequence_id=[sequence_id],
                    lidar=data_lidar_batch,
                    camera=None,
                )

                data_lidar_batch.meta[0].T_offset_nre_startend = to_torch(
                    T_offset_nre_startend, device="cpu", dtype=torch.float32
                )

                return DataAndRenderingBatch(data=data_batch, rendering=None)

        raise IndexError(f"Out of range sequential sample {batch_idx}")

    @overload
    def __getitem__(self, idx: int) -> DataAndRenderingBatch: ...

    @overload
    def __getitem__(self, idx: tuple[int, Optional[NovelViewOverrides]]) -> DataAndRenderingBatch: ...

    def __getitem__(self, idx: tuple[int, Optional[NovelViewOverrides]] | int) -> DataAndRenderingBatch:
        """
        Returns a specific sample of the dataset (depending on split type and parametrization)

        Args:
            idx: In the normal case this is the batch_idx [int].
                 Also supports novel view sampling when used in combination with a sampler that supports it.
                 Then this is a tuple of (batch_idx, novel_view_overrides) [tuple[int, NovelViewOverrides]].
        """
        if isinstance(idx, int):
            batch = self.get_item_novel_view_overrides((idx, None))
            batch_idx = batch.data.idx if isinstance(batch, DataAndRenderingBatch) else batch.idx
            assert batch_idx == idx
        else:
            batch = self.get_item_novel_view_overrides(idx)
            batch_idx = batch.data.idx if isinstance(batch, DataAndRenderingBatch) else batch.idx
            assert batch_idx == idx[0]

        return batch


class NCORETrainDataset(NCOREDataset):
    """Encapsulates the logic for sampling frames during training"""

    def __init__(
        self,
        datasource: NCOREDataSource,
        config: NCoreDatasetConfig,
        split: str,
    ):
        super().__init__(datasource, config, split)

        # store relevant parameters from config
        self.n_samples_per_epoch: int = config.n_samples_per_epoch
        self.n_train_sample_camera_rays: int = config.n_train_sample_camera_rays
        self.n_train_sample_lidar_rays: int = config.n_train_sample_lidar_rays

        self._worker_id: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None

        self.batch_samplers: list[BaseBatchSampler] = []

        for batch_sampler_config in config.samplers.values():
            self.batch_samplers.append(
                BaseBatchSampler.batch_sampler_factory(batch_sampler_config.name, batch_sampler_config, dataset=self)
            )

    @property
    def rng(self) -> np.random.Generator:
        self._maybe_reinit_rng()
        return unpack_optional(self._rng)

    def _maybe_reinit_rng(self) -> None:
        """
        Sets the rng according to which worker is executing.
        """
        # Determine current worker / process and whether re-initialization is necessary
        match torch.utils.data.get_worker_info():
            case None:
                # main process case
                if self._rng and self._worker_id is None:
                    # skip re-initialization
                    return
                # in case we move back from a worker to the main process we should also re-init
                self._worker_id = None
                seed = 0  # deterministic sampling

            case torch.utils.data._utils.worker.WorkerInfo(id=worker_id, seed=worker_seed):  # type:ignore
                # worker process case
                if self._rng and self._worker_id is worker_id:
                    # skip re-initialization
                    return
                self._worker_id = worker_id
                seed = worker_seed  # non-deterministic sampling

        seed = compute_process_local_rng_seed(seed)

        self._rng = np.random.default_rng(seed=seed)

    @ScopedTimer("NCORETrainDataset.__getitem__", TimingTag.DATALOADER)
    def __getitem__(self, batch_idx: int) -> DataAndRenderingBatch:
        # make sure worker is initialized
        self.datasource._maybe_init_worker()
        with ScopedTimer("NCORETrainDataset.__getitem__/get_batch", TimingTag.DATALOADER):
            batches = [batch_sampler.get_batch(batch_idx, self) for batch_sampler in self.batch_samplers]
        assert len(batches) == 1, "NCOREDataset: multiple batches per sample is not supported for now"
        return batches[0]

    def __len__(self) -> int:
        """Returns the total number of samples provided by the dataset (depending on split type and parametrization)"""

        return self.n_samples_per_epoch

    def update_epoch(self, epoch: int, system, **kwargs) -> None:
        for batch_sampler in self.batch_samplers:
            batch_sampler.update_epoch(epoch, system, **kwargs)

    def get_max_num_rays_per_train_sample(self):
        return sum([batch_sampler.get_max_rays_num() for batch_sampler in self.batch_samplers])

    @ScopedTimer("NCORETrainDataset.get_train_batch", TimingTag.DATALOADER)
    def get_train_batch(
        self,
        batch_idx: int,
        n_train_sample_camera_rays: int,
        camera_sensor_sampler: BaseSensorSampler,
        camera_frame_sampler: BaseFrameSampler,
        camera_pixel_sampler: BaseCameraPixelSampler,
        n_train_sample_lidar_rays: int,
        lidar_sensor_sampler: BaseSensorSampler,
        lidar_frame_sampler: BaseFrameSampler,
        lidar_point_sampler: BaseLidarPointSampler,
    ) -> DataAndRenderingBatch:
        """
        This method returns a batch from the training split, applying random sampling internally.
        Lidar / camera samplers need to be provided if requesting sensor-specific rays.
        """

        # make sure worker is initialized
        self.datasource._maybe_init_worker()

        # randomly sample chunk according to per-chunk probability ~ induces underlying sequence to sample from
        sequence_id = self.datasource.sequence_id

        data_camera_batch: DataBatch.Camera | None = None
        if n_train_sample_camera_rays:
            assert not isinstance(camera_sensor_sampler, SkipSensorSampler), (
                f"{self.__class__.__name__}: require camera sensor sampler if requesting camera rays in training batch"
            )
            assert not isinstance(camera_frame_sampler, SkipFrameSampler), (
                f"{self.__class__.__name__}: require camera frame sampler if requesting camera rays in training batch"
            )
            assert not isinstance(camera_pixel_sampler, SkipCameraPixelSampler), (
                f"{self.__class__.__name__}: require camera pixel sampler if requesting camera rays in training batch"
            )

            # sample a camera sensor using the SensorSampler
            sampled_camera_ids: list[str] = camera_sensor_sampler.sample_sensor(
                self.rng, batch_idx, self.camera_ids
            ).sampled_sensor_ids
            assert len(sampled_camera_ids) == 1, (
                "NCOREDataset: multiple camera sensors per batch is not supported for now"
            )
            camera_id = sampled_camera_ids[0]
            camera_frame_range = self.datasource.camera_frame_ranges[camera_id]

            # sample camera frame using FrameSampler
            camera_frame_idx = camera_frame_sampler.sample_frame(
                self.rng,
                batch_idx,
                frame_range=camera_frame_range,
                unique_sensor_id=(unique_camera_id := self.datasource.camera_unique_ids[camera_id].id),
            ).sampled_frame_idx

            # determine valid pixels of frame
            frame_valid_pixels_mask = self.datasource.cameras_frame_valid_pixels_masks[camera_id][
                camera_frame_idx
            ].unpacked()

            # sample pixels of frame using CameraPixelSampler
            pixel_samples = camera_pixel_sampler.sample_camera_pixels(
                rng=self.rng,
                batch_idx=batch_idx,
                frame_all_pixels=self.datasource.cameras_all_pixels[camera_id],
                frame_valid_pixels_mask=frame_valid_pixels_mask,
                n_frame_pixel_samples=n_train_sample_camera_rays,
                unique_camera_id=unique_camera_id,
                camera_frame_idx=camera_frame_idx,
                frame_range=camera_frame_range,
            )

            assert isinstance(pixel_samples.sampled_pixels, RectSubsampled)
            data_camera_batch = self.get_camera_data_batch(
                sensor_id=camera_id,
                sensor_frame_idx=camera_frame_idx,
                sampled_pixels=pixel_samples.sampled_pixels,
            )

        data_lidar_batch: DataBatch.Lidar | None = None
        if n_train_sample_lidar_rays:
            assert not isinstance(lidar_sensor_sampler, SkipSensorSampler), (
                f"{self.__class__.__name__}: require lidar sensor sampler if requesting lidar rays in training batch"
            )
            assert not isinstance(lidar_frame_sampler, SkipFrameSampler), (
                f"{self.__class__.__name__}: require lidar frame sampler if requesting lidar rays in training batch"
            )
            assert not isinstance(lidar_frame_sampler, SkipLidarPointSampler), (
                f"{self.__class__.__name__}: require lidar point sampler if requesting lidar rays in training batch"
            )

            # sample a lidar sensor using the SensorSampler
            sampled_lidar_ids = lidar_sensor_sampler.sample_sensor(
                self.rng, batch_idx, self.lidar_ids
            ).sampled_sensor_ids
            assert len(sampled_lidar_ids) == 1, (
                "NCOREDataset: multiple lidar sensors per batch is not supported for now"
            )
            lidar_id = sampled_lidar_ids[0]
            lidar_frame_range = self.datasource.lidar_frame_ranges[lidar_id]

            # sample a lidar frame using FrameSampler
            lidar_frame_idx = lidar_frame_sampler.sample_frame(
                self.rng,
                batch_idx,
                lidar_frame_range,
                unique_sensor_id=(unique_lidar_id := self.datasource.lidar_unique_ids[lidar_id].id),
            ).sampled_frame_idx

            # determine valid points of frame
            valid_points_mask = self.datasource.lidars_frame_valid_points_masks[lidar_id][lidar_frame_idx].unpacked()
            # sample points of frame using LidarPointSampler
            point_samples = lidar_point_sampler.sample_lidar_points(
                self.rng,
                batch_idx,
                lidar_frame_range,
                n_train_sample_lidar_rays,
                valid_points_mask,
                unique_lidar_id,
                lidar_frame_idx,
            )

            data_lidar_batch = self.get_lidar_data_batch(
                sensor_id=lidar_id,
                sensor_frame_idx=lidar_frame_idx,
                sampled_pixels=point_samples.sampled_point_idxs,
            )

        data_batch = DataBatch(
            idx=batch_idx,
            worker_id=[self._worker_id] if self._worker_id is not None else None,
            sequence_id=[sequence_id],
            lidar=data_lidar_batch,
            camera=data_camera_batch,
        )

        return DataAndRenderingBatch(data=data_batch, rendering=None)


class NCOREDataSource(BaseDataSource, RigTrajectoriesProvider):
    UNCONDITIONALLY_DYNAMIC_CLASSES: set[str] = set(
        [
            "pedestrian",
            "stroller",
            "person",
            "person_group",
            "rider",
            "bicycle_with_rider",
            "bicycle",
            "CYCLIST",
            "motorcycle",
            "motorcycle_with_rider",
            "cycle",
        ]
    )

    # Frame convention conversion from camera frame convention (x right, y down, z forward)
    # to rig frame convention (x forward, y left, z up)
    T_CAMERA_RIG = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)

    @unique
    class ValidMeasurementsMethod(IntEnum):
        """Different method variants to determine valid pixels/lidarpoints"""

        EGO = auto()  # ego camera masks only
        EGO_CUBOIDTRACKS = (
            auto()
        )  # ego camera masks and frame-projected dynamic objects from cuboid tracks; dynamic_flag lidar points
        EGO_SCENEFLOW = auto()  # ego camera masks and scene flow masks
        EGO_CUBOIDTRACKS_SCENEFLOW = auto()  # ego camera masks, frame-projected dynamic objects from cuboid tracks, and scene flow masks; dynamic_flag lidar points
        EGO_CUBOIDTRACKS_TRAFFICLIGHT = auto()  # ego camera masks, frame-projected dynamic objects from cuboid tracks, and traffic light masks; dynamic_flag, non-traffic light, camera-visible lidar points
        EGO_FRAMEMASKS = auto()  # ego camera masks and per-frame camera masks

    def __init__(self, config: NCoreDatasetConfig, lidar_model_config: Optional[LidarModelsConfig]) -> None:
        """This method loads **metadata** and defers loading actual data to _load_data"""
        super().__init__()

        # Init parameters from config
        self.camera_mask_sources: List[Literal["dataset", "aux"]] = config.camera_mask_sources
        self.n_camera_mask_dilation_iterations: int = config.n_camera_mask_dilation_iterations
        self.camera_mask_overrides: Dict[str, str] = config.camera_mask_overrides
        self.camera_mask_border: Optional[Dict[str, list[int]]] = config.camera_mask_border

        self.camera_max_fov_deg: float = config.camera_max_fov_deg
        self.lidar_model_parameter_cwccw_fallback: bool = config.lidar_model_parameter_cwccw_fallback
        self.lidar_model_parameter_nominal_values: dict[str, Literal["HESAI-Pandar128", "HESAI-AT128"]] = (
            config.lidar_model_parameter_nominal_values
        )
        self.cuboid_tracks_params = config.cuboid_tracks_params
        self.lidar_dynamic_points_params = config.lidar_dynamic_points

        # load valid measurements (pixels/lidarpoints) method and method-dependent parameters
        self.valid_measurements_method = NCOREDataSource.ValidMeasurementsMethod[config.valid_measurements_method]
        self.valid_pixels_cuboid_tracks_params = config.valid_pixels_cuboid_track_params
        self.valid_pixels_pixels_scene_flow_params = config.valid_pixels_scene_flow_params
        self.valid_pixels_traffic_light_params = config.valid_pixels_traffic_light_params
        self.valid_pixels_frame_mask_params = config.valid_pixels_frame_mask_params
        self.valid_points_cuboid_track_params = config.valid_lidarpoints_cuboid_track_params

        # V3 sequence loader parameters
        self.cuboid_loading_max_workers: Optional[int] = config.cuboid_loading_max_workers

        # V4 sequence loader parameters
        self.poses_component_group: str = config.poses_component_group
        self.intrinsics_component_group: str = config.intrinsics_component_group
        self.masks_component_group: str = config.masks_component_group
        self.cuboids_component_group: str = config.cuboids_component_group

        self.camera_ids: list[str] = config.camera_ids
        self.lidar_ids: list[str] = config.lidar_ids
        self.train_camera_ids: list[str] = config.train_camera_ids
        self.train_lidar_ids: list[str] = config.train_lidar_ids
        self.lidar_ignore_rows: list[int] = config.lidar_ignore_rows
        self.val_camera_ids: list[str] = (
            config.val_camera_ids
        )  # validation cameras are the ones not in train_camera_ids
        self.val_lidar_ids: list[str] = config.val_lidar_ids

        self.all_camera_ids = sorted(set(self.camera_ids + self.train_camera_ids))
        self.all_lidar_ids = sorted(set(self.lidar_ids + self.train_lidar_ids))
        self.camera_point_cloud_ignore_classes = set(config.camera_point_cloud_ignore_classes)
        self.camera_point_cloud_dynamic_classes = set(config.camera_point_cloud_dynamic_classes)

        self.lidar_model_config: Optional[LidarModelsConfig] = lidar_model_config

        # Maximum distance away from the ego pose that will still be covered by the bounding box and the feature volume
        self.max_dist_m: float = config.max_dist_m

        self.open_consolidated: bool = config.open_consolidated

        self.jpeg_backend_cpu: Literal["PIL", "simplejpeg"] = config.jpeg_backend_cpu
        self.simplejpeg_fastdct: bool = config.simplejpeg_fastdct
        self.simplejpeg_fastupsample: bool = config.simplejpeg_fastupsample

        self.frame_generic_data_pose_overwrite: bool = config.frame_generic_data_pose_overwrite

        self.aux_data: bool = config.aux_data

        self.tqdm_disabled = not config.show_progress_bars

        ## Determine V3 / V4 NCore data format to load and branch accordingly
        self.path = UPath(config.path)
        (
            self.data_format,
            self.sequence_id,
            self.time_range_us,
            # contains either V3 zarr.itar shards, or V4 zarr.itar archives / zarr directories
            self.dataset_paths,
        ) = ncore_utils.parse_sequence_meta_file(self.path)

        def update_time_range_bounds(time_range_us: HalfClosedInterval) -> HalfClosedInterval:
            """Restricts time range according to config seek_offset_sec and duration_sec"""

            SEC_TO_USEC = 1_000_000
            if seek_offset_sec := config.seek_offset_sec:
                time_range_us.start += int(seek_offset_sec * SEC_TO_USEC)
                if time_range_us.end <= time_range_us.start:
                    raise AssertionError(f"{config.seek_offset_sec=} leads to an empty interval.")

            if duration_sec := config.duration_sec:
                time_range_us.end = min(time_range_us.start + int(duration_sec * SEC_TO_USEC), time_range_us.end)
                if time_range_us.end <= time_range_us.start:
                    raise AssertionError(f"{config.duration_sec=} leads to an empty interval.")

            return time_range_us

        self.time_range_us = update_time_range_bounds(self.time_range_us)

        # sensor-id to range of sensor frame indices
        self.camera_frame_ranges: dict[str, range] = {}
        self.lidar_frame_ranges: dict[str, range] = {}
        # sensor-id to global linear start frame index (across all cameras/lidars)
        self.camera_linear_start_frame_indices: dict[str, int] = {}
        self.lidar_linear_start_frame_indices: dict[str, int] = {}

        # initialize shards once fully
        self.sequence_loader: Optional[ncore.data.SequenceLoaderProtocol] = None
        self.worker_id: Optional[int] = None

        # initialize params to generate traffic light cuboid tracks
        self.generate_static_rigid = config.generate_static_rigid_cuboid_tracks.enabled
        self.visibility_check_camera_ids = config.generate_static_rigid_cuboid_tracks.visibility_check_camera_ids
        self.max_visible_distance = config.generate_static_rigid_cuboid_tracks.max_distance_m
        self.generated_static_rigid_cuboidtracks: Optional[CuboidTracks] = None
        self.static_rigid_classes = config.generate_static_rigid_cuboid_tracks.rigid_classes
        self.static_rigid_point_cloud_path = config.generate_static_rigid_cuboid_tracks.point_cloud_path
        # make sure worker is initialized
        self._maybe_init_worker()

    def get_path(self) -> UPath:
        """
        The path to the .json file defining this datasource.
        """
        return self.path

    def get_offset(self) -> npt.NDArray[np.float32]:
        nre_to_world = self.world_to_nre.inverse()
        return nre_to_world.target_origin

    def _maybe_init_worker(self) -> None:
        """
        Detects if worker ID has changed (indicating multiprocessing fork) and either
            * returns early if no changes
            * performs full data initialization if this is the first use of this instance
            * recreates file descriptors to avoid accidental sharing if fork has happened
        """

        # Determine current worker / process and whether re-initialization is necessary.
        # Terminate if not necessary.
        match torch.utils.data.get_worker_info():
            case None:
                # main process case
                if self.sequence_loader is not None and self.worker_id is None:
                    # skip re-initialization
                    return
                # in case we move back from a worker to the main process we should also re-init
                self.worker_id = None
            case torch.utils.data._utils.worker.WorkerInfo(id=worker_id):  # type:ignore
                # worker process case
                if self.sequence_loader is not None and self.worker_id is worker_id:
                    # skip re-initialization
                    return
                self.worker_id = worker_id

        # Reload case: check if reloading is sufficient for initialization, terminate if yes
        if self.sequence_loader is not None:
            # only reload data loaders
            self.sequence_loader.reload_resources()
            if self.aux_loader is not None:
                self.aux_loader.reload_store_resources()
            return

        # No termination means we need a full initialization
        self._load_data()

    @staticmethod
    def _check_if_present_sensor_ids(
        sensor_ids_requested: List[str], sensor_ids_shard: List[str], sensor_type: str
    ) -> None:
        """Check if requested sensor data in the shard."""
        if sensor_ids_not_in_shard := [
            sensor_id for sensor_id in sensor_ids_requested if sensor_id not in sensor_ids_shard
        ]:
            raise RuntimeError(f"Requested {sensor_type} not present in the data: {', '.join(sensor_ids_not_in_shard)}")

    def _load_data(self) -> None:
        """Initializes the dataset from scratch (start-up case)"""
        # Full initial load case
        self.camera_sensors: dict[str, ncore.data.CameraSensorProtocol] = {}
        self.camera_models: dict[str, ncore.sensors.CameraModel] = {}
        self.camera_model_parameters: dict[str, ncore.data.ConcreteCameraModelParametersUnion] = {}
        self.camera_unique_ids: dict[str, UniqueSensorId] = {}
        self.cameras_all_pixels: dict[str, np.ndarray] = {}
        self.cameras_all_rays: dict[str, np.ndarray] = {}
        self.cameras_all_rays_halfpixoffset: dict[str, np.ndarray] = {}
        self.cameras_all_footprints: dict[str, np.ndarray] = {}
        self.cameras_all_footprints_halfpixoffset: dict[str, np.ndarray] = {}
        self.cameras_valid_pixels_ego_masks: dict[str, PackedMask] = {}
        self.cameras_frame_valid_pixels_masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)
        self.cameras_frame_track_idxs: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
        self.lidars_frame_valid_points_masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)
        self.lidars_frame_non_dynamic_points_masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)
        self.lidar_sensors: dict[str, ncore.data.LidarSensorProtocol] = {}
        self.lidar_models: dict[str, ConcreteLidarModelsUnion | None] = {}
        self.lidar_model_parameters: dict[str, Optional[ncore.data.ConcreteLidarModelParametersUnion]] = {}
        self.lidar_unique_ids: dict[str, UniqueSensorId] = {}

        self.cuboidtracks_all: CuboidTracks | None = None
        self.cuboidtracks_dynamic: CuboidTracks | None = None
        self.cuboids_df: pd.DataFrame | None = None

        self.aux_loader: Optional[ncore_utils.AuxShardDataLoader] = None
        self.sensor_sky_class_ids: dict[str, int] = {}
        self.sensor_road_class_ids: dict[str, int] = {}
        self.sensor_vehicle_classes_ids: dict[str, list[int]] = {}
        self.sensor_trafficlight_class_ids: dict[str, int] = {}
        self.sensor_dinov2_windows: dict[str, np.ndarray] = {}

        # init sequence loader
        sequence_loader = self.sequence_loader = ncore_utils.create_sequence_loader(
            data_format=self.data_format,
            dataset_paths=self.dataset_paths,
            open_consolidated=self.open_consolidated,
            v3_cuboid_loading_max_workers=self.cuboid_loading_max_workers,
            v4_poses_component_group=self.poses_component_group,
            v4_intrinsics_component_group=self.intrinsics_component_group,
            v4_masks_component_group=self.masks_component_group,
            v4_cuboids_component_group=self.cuboids_component_group,
        )

        # init aux loader if enabled
        if self.aux_data:
            self.aux_loader = ncore_utils.AuxShardDataLoader(
                sequence_id=sequence_loader.sequence_id,
                dataset_paths=self.dataset_paths,
                open_consolidated=self.open_consolidated,
            )

        ## Load sensor data and

        # conditionally applied per-sensor overwrites
        S = TypeVar("S", bound=ncore.data.SensorProtocol)

        def patch_sensor(sensor: S) -> S:
            """Patch sensor behaviour conditionally based on config options"""

            if self.frame_generic_data_pose_overwrite:
                # Create a new pose graph associated with this sensor
                # instance that makes use of overwritten T_sensor_world poses
                # instead of the regular rig-consistent T_rig_world poses

                # Collect all per-frame start/end T_sensor_world poses from generic data
                T_sensor_worlds_list = []

                for frame_idx in range(sensor.frames_count):
                    if sensor.has_frame_generic_data(frame_idx, "T_sensor_worlds"):
                        T_sensor_worlds_list.append(
                            sensor.get_frame_generic_data(frame_idx, "T_sensor_worlds")
                        )  # (2,4,4)
                    else:
                        raise RuntimeError(
                            f"Sensor {sensor.sensor_id} frame {frame_idx} missing generic data field 'T_sensor_worlds' required for pose overwrite"
                        )
                T_sensor_worlds = np.concatenate(T_sensor_worlds_list, axis=0)  # (2*N_frames,4,4)
                T_sensor_worlds_timestamps_us = sensor.frames_timestamps_us.flatten()

                # Filter out unique timestamps / poses
                # (for instance, global shutter sensors / virtual sensors tend to have identical start/end poses)
                T_sensor_worlds_timestamps_us, unique_indices = np.unique(
                    T_sensor_worlds_timestamps_us, return_index=True
                )
                T_sensor_worlds = T_sensor_worlds[unique_indices]

                # Grab the existing pose-graph edges except for the rig-world edge, and add our new sensor-world edge
                edges = [
                    edge
                    for nodes, edge in sensor.pose_graph.normalized_edge_map.items()
                    if nodes != ncore_transformations.PoseGraphInterpolator.normalize_node_pair("rig", "world")
                ] + [
                    ncore_transformations.PoseGraphInterpolator.Edge(
                        sensor.sensor_id,
                        "world",
                        T_sensor_worlds,
                        T_sensor_worlds_timestamps_us,
                    )
                ]

                # Instantiate and assign overwritten pose graph to sensor
                sensor.set_pose_graph(ncore_transformations.PoseGraphInterpolator(edges))

            return sensor

        # load camera sensors
        self._check_if_present_sensor_ids(self.all_camera_ids, sequence_loader.camera_ids, "cameras")
        self.camera_sensors = {
            camera_id: patch_sensor(sequence_loader.get_camera_sensor(camera_id)) for camera_id in self.all_camera_ids
        }

        # load camera model parameters and models
        self.camera_model_parameters = {
            camera_id: camera_sensor.model_parameters for camera_id, camera_sensor in self.camera_sensors.items()
        }

        # restrict effective FOV of omnidirectional cameras for cuboid projections to prevent issues with *invalid* points projected
        # back into the valid image domain FOV - this way they get properly classified as invalid
        for camera_model_parameter in self.camera_model_parameters.values():
            if not isinstance(
                camera_model_parameter,
                (ncore.data.FThetaCameraModelParameters, ncore.data.OpenCVFisheyeCameraModelParameters),
            ):
                continue
            camera_model_parameter.max_angle = min(
                np.deg2rad(self.camera_max_fov_deg) / 2.0, camera_model_parameter.max_angle
            )

        camera_models = self.camera_models = {
            camera_id: ncore.sensors.CameraModel.from_parameters(
                camera_model_parameter, device="cpu", dtype=torch.float32
            )
            for camera_id, camera_model_parameter in self.camera_model_parameters.items()
        }

        # construct unique camera instance ids (<camera_id>@<sequence_id>) and instance indices
        self.camera_unique_ids = {
            camera_id: UniqueSensorId("@".join((camera_id, self.sequence_id)), camera_instance_idx)
            for camera_instance_idx, camera_id in enumerate(self.all_camera_ids)
        }

        # determine all pixels and rays per camera
        for camera_id in self.all_camera_ids:
            camera_sensor = self.camera_sensors[camera_id]
            camera_model = camera_models[camera_id]

            # sample pixel ranges
            w = int(camera_model.resolution[0].item())
            h = int(camera_model.resolution[1].item())

            # all pixels
            self.cameras_all_pixels |= {camera_id: generate_grid_2d_indices((w, h)).numpy()}

            # dataset associated mask or mask overwrite (if present)
            dataset_camera_mask_array = ncore_utils.get_mask_image(
                # use 'ego' camera mask as static image mask if available
                camera_sensor.get_mask_images().get("ego"),
                tuple(camera_sensor.model_parameters.resolution),
                # load image from override, if available
                self.camera_mask_overrides.get(camera_id),
            )

            # aux camera mask (if enabled and present)
            aux_camera_mask_array: Optional[np.ndarray] = None
            if self.aux_loader is not None:
                if self.aux_loader.has_egomask(camera_id):
                    aux_camera_mask_array = self.aux_loader.get_egomask(camera_id, 0)  # get the aggregated ego-mask

            # determine final mask to use based on priority of sources defined in config
            camera_mask_array: Optional[np.ndarray] = None
            for camera_mask_source in self.camera_mask_sources:
                if camera_mask_source == "dataset":
                    camera_mask_array = dataset_camera_mask_array
                elif camera_mask_source == "aux":
                    camera_mask_array = aux_camera_mask_array
                else:
                    raise ValueError(f"Invalid camera mask source {camera_mask_source} in config")

                if camera_mask_array is not None:
                    break  # found a valid mask from this source

            if camera_mask_array is not None:
                # Dilate mask boundary
                camera_mask_array = ndimage.binary_dilation(
                    camera_mask_array, iterations=self.n_camera_mask_dilation_iterations
                )

                # Subsample valid pixels relative to mask (True for parts that we want to keep)
                camera_valid_pixels_ego_mask = np.logical_not(cast(np.ndarray, camera_mask_array))
            else:
                # No mask / consider all pixels as valid
                camera_valid_pixels_ego_mask = np.ones(
                    (int(camera_model.resolution[1].item()), int(camera_model.resolution[0].item())), dtype=bool
                )

            if self.camera_mask_border is not None and camera_id in self.camera_mask_border:
                # top, right, bottom, left
                camera_valid_pixels_ego_mask[: self.camera_mask_border[camera_id][0]] = False
                camera_valid_pixels_ego_mask[:, : self.camera_mask_border[camera_id][1]] = False
                if self.camera_mask_border[camera_id][2] != 0:
                    camera_valid_pixels_ego_mask[-self.camera_mask_border[camera_id][2] :] = False

                if self.camera_mask_border[camera_id][3] != 0:
                    camera_valid_pixels_ego_mask[:, -self.camera_mask_border[camera_id][3] :] = False

            self.cameras_valid_pixels_ego_masks |= {camera_id: PackedMask(camera_valid_pixels_ego_mask)}

            # precompute all rays at pixel centers and at half-pixel offsets
            self.cameras_all_rays |= {
                camera_id: camera_model.pixels_to_camera_rays(self.cameras_all_pixels[camera_id])
                .reshape(h, w, 3)
                .numpy()
            }
            self.cameras_all_rays_halfpixoffset |= {
                camera_id: camera_model.image_points_to_camera_rays(
                    camera_model.pixels_to_image_points(self.cameras_all_pixels[camera_id]) + 0.5
                )
                .reshape(h, w, 3)
                .numpy()
            }

            # Compute the footprint of the pixels at unit length
            self.cameras_all_footprints |= {camera_id: compute_pixel_footprint(self.cameras_all_rays[camera_id])}
            self.cameras_all_footprints_halfpixoffset |= {
                camera_id: compute_pixel_footprint(self.cameras_all_rays_halfpixoffset[camera_id])
            }

        # load lidar sensors and lidar model parameters
        self._check_if_present_sensor_ids(self.all_lidar_ids, sequence_loader.lidar_ids, "lidars")
        self.lidar_sensors = {
            lidar_id: patch_sensor(sequence_loader.get_lidar_sensor(lidar_id)) for lidar_id in self.all_lidar_ids
        }

        def get_lidar_model_parameters(
            lidar_id: str,
            lidar_sensor: ncore.data.LidarSensorProtocol,
            nominal_values: dict[str, Literal["HESAI-Pandar128", "HESAI-AT128"]],
        ) -> Optional[ncore.data.ConcreteLidarModelParametersUnion]:
            """Wrapper to load lidar model parameters from the dataset, applying optional fallbacks if enabled"""
            if lidar_id in nominal_values:
                lidar_models: LidarModelsConfig = unpack_optional(self.lidar_model_config)
                if nominal_values[lidar_id] == "HESAI-AT128":
                    return LidarModelBundle.load_from_config(lidar_models.HESAI_AT128).lidar_parameters
                elif nominal_values[lidar_id] == "HESAI-Pandar128":
                    return LidarModelBundle.load_from_config(lidar_models.HESAI_Pandar128).lidar_parameters
                else:
                    raise ValueError(
                        f"{self.__class__.__name__} lidar model type {nominal_values[lidar_id]} not supported"
                    )
            lidar_model_parameters = get_lidar_model_parameters_with_fallbacks(
                lidar_sensor, self.lidar_model_parameter_cwccw_fallback
            )
            return lidar_model_parameters

        self.lidar_model_parameters = {
            lidar_id: get_lidar_model_parameters(lidar_id, lidar_sensor, self.lidar_model_parameter_nominal_values)
            for lidar_id, lidar_sensor in self.lidar_sensors.items()
        }

        self.lidar_models = {
            lidar_id: cast(
                ConcreteLidarModelsUnion | None,
                ncore.sensors.LidarModel.maybe_from_parameters(
                    lidar_model_parameters, device="cpu", dtype=torch.float32
                ),
            )
            for lidar_id, lidar_model_parameters in self.lidar_model_parameters.items()
        }

        # construct unique lidar instance ids (<lidar_id>@<sequence_id>) and instance indices
        self.lidar_unique_ids = {
            lidar_id: UniqueSensorId("@".join((lidar_id, self.sequence_id)), lidar_instance_idx)
            for lidar_instance_idx, lidar_id in enumerate(self.all_lidar_ids)
        }

        # load aux data shards if enabled
        if self.aux_loader is not None:
            # get index of 'sky' class for each sensor
            self.sensor_sky_class_ids = {
                camera_id: self.aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"].index("sky")
                for camera_id in self.all_camera_ids
            }
            if self.aux_loader.has_lidar_semantic_segmentation():
                self.sensor_sky_class_ids.update(
                    {
                        lidar_id: self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_classes"].index(
                            "sky"
                        )
                        for lidar_id in self.all_lidar_ids
                    }
                )
            # get index of 'road' class for each sensor
            self.sensor_road_class_ids = {
                camera_id: self.aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"].index("road")
                for camera_id in self.all_camera_ids
            }
            if self.aux_loader.has_lidar_semantic_segmentation():
                self.sensor_road_class_ids.update(
                    {
                        lidar_id: self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_classes"].index(
                            "road"
                        )
                        for lidar_id in self.all_lidar_ids
                    }
                )
            # get index of vehicle classes (including "car", "truck", "bus", "train") for each sensor
            VEHICLE_CLASSES = ["car", "truck", "bus", "train"]
            self.sensor_vehicle_classes_ids = {
                camera_id: [
                    stuff_classes.index(vehicle_class)
                    for vehicle_class in VEHICLE_CLASSES
                    if vehicle_class
                    in (stuff_classes := self.aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"])
                ]
                for camera_id in self.all_camera_ids
            }

            if self.aux_loader.has_lidar_semantic_segmentation():
                self.sensor_vehicle_classes_ids.update(
                    {
                        lidar_id: [
                            self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_classes"].index(
                                vehicle_class
                            )
                            for vehicle_class in ["car", "truck", "bus", "train"]
                        ]
                        for lidar_id in self.all_lidar_ids
                    }
                )

            # get DINOv2 related metadata
            if self.aux_loader.has_dinov2():
                self.sensor_dinov2_windows = {}
                for camera_id in self.all_camera_ids:
                    dinov2_meta_info, _, _ = self.aux_loader.get_dinov2_meta(camera_id)
                    self.sensor_dinov2_windows[camera_id] = np.asarray(
                        [
                            dinov2_meta_info["window_left"],
                            dinov2_meta_info["window_top"],
                            dinov2_meta_info["window_width"],
                            dinov2_meta_info["window_height"],
                        ]
                    )

            match self.valid_measurements_method:
                case NCOREDataSource.ValidMeasurementsMethod.EGO_CUBOIDTRACKS_TRAFFICLIGHT:
                    self.sensor_trafficlight_class_ids = {
                        camera_id: self.aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"].index(
                            "traffic light"
                        )
                        for camera_id in self.all_camera_ids
                    }
                    self.sensor_trafficlight_class_ids.update(
                        {
                            lidar_id: self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)[
                                "stuff_classes"
                            ].index("traffic light")
                            for lidar_id in self.all_lidar_ids
                        }
                    )

        cumulative_camera_start_frame_index: int = 0
        cumulative_lidar_start_frame_index: int = 0

        # load all rig->world and world->world-global poses

        # TODO: frame-pose only data might fail here as there are no rig poses and might require refined logic
        rig_world_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
            sequence_loader.pose_graph.get_edge("rig", "world"),
            msg="Rig-to-world poses are currently required to determine scene extend",
        )

        T_rig_world = rig_world_edge.T_source_target
        T_rig_world_timestamps_us: np.ndarray = unpack_optional(
            rig_world_edge.timestamps_us, msg="Rig-to-world pose requires to be dynamic"
        )

        # NCore V4: world→world_global is optional (global CRS, e.g. ECEF). Absent edge ⇒ identity float64, matching
        # world_world_global_edge.T_source_target dtype when the edge exists and RigTrajectories.T_world_base.
        world_world_global_edge = sequence_loader.pose_graph.get_edge("world", "world_global")
        self.T_world_world_global = (
            world_world_global_edge.T_source_target
            if world_world_global_edge is not None
            else np.eye(4, dtype=np.float64)
        )

        # restrict the time_range_us to ensure all rig-world poses are known in the full interval
        # (crucial when using seek_offset_sec/duration_sec, because the next available rig-world
        #  pose may be sampled after the time range starts, rendering the very beginning un-posed)
        self.time_range_us = self.time_range_us.restricted(T_rig_world_timestamps_us)

        # determine linear per-sensor-frame index ranges depending on dataset time restrictions,
        # making sure *both* frame start and end-times are fully covered
        def get_sensor_frame_range(frames_timestamps_us: np.ndarray) -> range:
            # make sure end-of-frame times are are covered by the time range
            cover_range = self.time_range_us.cover_range(frames_timestamps_us[:, ncore.data.FrameTimepoint.END])
            # make sure that the first frame's start-of-frame time is also covered - skip frames as required
            # (could be more than a single frame if frame ranges are not exclusively partitioning the time range)
            while len(cover_range) and (
                int(frames_timestamps_us[cover_range.start, ncore.data.FrameTimepoint.START]) not in self.time_range_us
            ):
                cover_range = cover_range[1:]

            return cover_range

        self.camera_frame_ranges = {
            camera_id: get_sensor_frame_range(self.camera_sensors[camera_id].frames_timestamps_us)
            for camera_id in self.all_camera_ids
        }
        self.lidar_frame_ranges = {
            lidar_id: get_sensor_frame_range(self.lidar_sensors[lidar_id].frames_timestamps_us)
            for lidar_id in self.all_lidar_ids
        }

        # determine per-sensor starting index to return unique linear frame indices for each sample
        for camera_id, camera_frame_range in self.camera_frame_ranges.items():
            self.camera_linear_start_frame_indices[camera_id] = cumulative_camera_start_frame_index
            cumulative_camera_start_frame_index += len(camera_frame_range)

        for lidar_id, lidar_frame_range in self.lidar_frame_ranges.items():
            self.lidar_linear_start_frame_indices[lidar_id] = cumulative_lidar_start_frame_index
            cumulative_lidar_start_frame_index += len(lidar_frame_range)

        # compute average position and extent (largest axis of the scene's AABB relative to the world frame)
        # to put the scene's center at the origin
        rig_world_positions = T_rig_world[self.time_range_us.cover_range(T_rig_world_timestamps_us)][:, :3, 3]
        mean_rig_world_position_m = rig_world_positions.mean(axis=0).astype(np.float32)

        # Center the scenes at the mean rig position
        self.world_to_nre = FrameConversion.from_origin_scale_axis(
            target_origin=mean_rig_world_position_m,  # put the scene's center at the origin
            target_scale=1.0,
            target_axis=[0, 1, 2],
        )

        # Compute per-frame valid measurements (pixels/lidarpoints), e.g., excluding dynamic objects, ego-car, ...
        match self.valid_measurements_method:
            case NCOREDataSource.ValidMeasurementsMethod.EGO:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                # lidar point masks (all lidar points are considered valid in this measurement method)
                self.lidars_frame_valid_points_masks = compute_valid_lidarpoints_all(
                    self.lidar_frame_ranges,
                    self.lidar_sensors,
                )
            case NCOREDataSource.ValidMeasurementsMethod.EGO_CUBOIDTRACKS:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                self._compute_valid_pixels_cuboidtracks()
                # lidar point masks
                self._compute_valid_lidarpoints_dynamic()
            case NCOREDataSource.ValidMeasurementsMethod.EGO_SCENEFLOW:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                assert self.aux_loader is not None
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_sceneflow(
                    aux_loader=self.aux_loader,
                    camera_sensors=self.camera_sensors,
                    camera_models=self.camera_models,
                    camera_frame_ranges=self.camera_frame_ranges,
                    valid_pixels_scene_flow_config=self.valid_pixels_pixels_scene_flow_params,
                    cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
                )
                # lidar point masks (all lidar points are considered valid in this measurement method)
                self.lidars_frame_valid_points_masks = compute_valid_lidarpoints_all(
                    self.lidar_frame_ranges,
                    self.lidar_sensors,
                )
            case NCOREDataSource.ValidMeasurementsMethod.EGO_CUBOIDTRACKS_SCENEFLOW:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                self._compute_valid_pixels_cuboidtracks()
                assert self.aux_loader is not None
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_sceneflow(
                    aux_loader=self.aux_loader,
                    camera_sensors=self.camera_sensors,
                    camera_models=self.camera_models,
                    camera_frame_ranges=self.camera_frame_ranges,
                    valid_pixels_scene_flow_config=self.valid_pixels_pixels_scene_flow_params,
                    cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
                )
                # lidar point masks
                self._compute_valid_lidarpoints_dynamic()
            case NCOREDataSource.ValidMeasurementsMethod.EGO_CUBOIDTRACKS_TRAFFICLIGHT:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                self._compute_valid_pixels_cuboidtracks()

                assert self.aux_loader is not None
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_trafficlight(
                    aux_loader=self.aux_loader,
                    camera_sensors=self.camera_sensors,
                    camera_frame_ranges=self.camera_frame_ranges,
                    valid_pixels_traffic_light_params=self.valid_pixels_traffic_light_params,
                    sensor_trafficlight_class_ids=self.sensor_trafficlight_class_ids,
                    cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
                    tqdm_disabled=self.tqdm_disabled,
                )
                # lidar point masks
                self.lidars_frame_valid_points_masks = compute_valid_lidarpoints_trafficlight_cameravisible(
                    aux_loader=self.aux_loader,
                    lidar_sensors=self.lidar_sensors,
                    all_camera_ids=self.all_camera_ids,
                    time_range_us=self.time_range_us,
                    sensor_trafficlight_class_ids=self.sensor_trafficlight_class_ids,
                    lidars_frame_valid_points_masks=self.lidars_frame_valid_points_masks,
                    tqdm_disabled=self.tqdm_disabled,
                )
                self._compute_valid_lidarpoints_dynamic()
            case NCOREDataSource.ValidMeasurementsMethod.EGO_FRAMEMASKS:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                self.cameras_frame_valid_pixels_masks = compute_cameras_valid_pixels_frame_mask(
                    camera_sensors=self.camera_sensors,
                    camera_frame_ranges=self.camera_frame_ranges,
                    valid_pixels_frame_mask_params=self.valid_pixels_frame_mask_params,
                    cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
                    tqdm_disabled=self.tqdm_disabled,
                )
                # lidar point masks (all lidar points are considered valid in this measurement method)
                self.lidars_frame_valid_points_masks = compute_valid_lidarpoints_all(
                    self.lidar_frame_ranges,
                    self.lidar_sensors,
                )
            case _:
                raise ValueError(
                    f"{self.__class__.__name__}: unsupported valid measurements method {self.valid_measurements_method}"
                )

        # Compute scene extent from point-clouds, if available, otherwise fall back to trajectory-inferred extent
        extent = torch.tensor(0.0, dtype=torch.float32)
        if len(
            point_clouds := list(
                self.get_point_clouds(
                    device=torch.device("cuda"), step_frame=10, non_dynamic_points_only=True, force=False
                )
            )
        ):
            pc = PointCloud.collate_fn(point_clouds, device=torch.device("cuda"))
            if pc.n_points > 0:
                extent = (pc.xyz_end.max(0).values - pc.xyz_end.min(0).values).max()

        if extent == 0:
            # make sure that the max distance at the boundary is included when scaling the scene extent to the target AABB scale
            world_max_extent_m = np.max(np.max(rig_world_positions, axis=0) - np.min(rig_world_positions, axis=0))
            extent = torch.tensor(world_max_extent_m + 2 * self.max_dist_m, dtype=torch.float32)

        # TODO: Cube for now, change to cuboid
        extent = torch.tensor([extent] * 3, dtype=torch.float32)

        self.aabb = AABB3D.from_center_extent(center=torch.zeros(3, dtype=torch.float32), extent=extent)

    def _compute_cuboid_tracks(self) -> None:
        """Computes cuboid tracks for all dynamic objects in the dataset"""
        # Only compute tracks if not initialized before
        if (self.cuboidtracks_all is None) or (self.cuboidtracks_dynamic is None) or (self.cuboids_df is None):
            cuboidtracks_all, self.cuboids_df = compute_cuboid_tracks(
                sequence_loader=unpack_optional(self.sequence_loader),
                time_range_us=self.time_range_us,
                cuboid_tracks_params=self.cuboid_tracks_params,
                tqdm_disabled=self.tqdm_disabled,
            )

            # Optional: compute per-track camera-visible time intervals to restrict dynamic classification to those windows
            cuboidtracks_visible_intervals = None
            if self.cuboid_tracks_params.camera_visibility:
                cuboidtracks_visible_intervals = compute_camera_visible_intervals_cuboid_tracks(
                    cuboidtracks=cuboidtracks_all,
                    camera_sensors=self.camera_sensors,
                    camera_models=self.camera_models,
                    camera_frame_ranges=self.camera_frame_ranges,
                    cameras_all_pixels=self.cameras_all_pixels,
                    cameras_all_rays=self.cameras_all_rays,
                    subsample=4,
                    tqdm_disabled=self.tqdm_disabled,
                )

            self.cuboidtracks_all, self.cuboidtracks_dynamic = classify_dynamic_cuboid_tracks(
                cuboidtracks=cuboidtracks_all,
                cuboid_tracks_params=self.cuboid_tracks_params,
                unconditionally_dynamic_classes=self.UNCONDITIONALLY_DYNAMIC_CLASSES,
                tqdm_disabled=self.tqdm_disabled,
                cuboidtracks_visible_intervals=cuboidtracks_visible_intervals,
            )

    def _compute_valid_pixels_cuboidtracks(self) -> None:
        """Computes per-frame projections of cuboid track dynamic objects as valid-pixels masks"""

        # tracks are required
        self._compute_cuboid_tracks()

        assert self.cuboidtracks_dynamic is not None
        compute_camera_visible_intervals_cuboid_tracks(
            cuboidtracks=self.cuboidtracks_dynamic,
            camera_sensors=self.camera_sensors,
            camera_models=self.camera_models,
            camera_frame_ranges=self.camera_frame_ranges,
            cameras_all_pixels=self.cameras_all_pixels,
            cameras_all_rays=self.cameras_all_rays,
            cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
            cameras_frame_track_idxs=self.cameras_frame_track_idxs
            if self.valid_pixels_cuboid_tracks_params.frame_track_idxs
            else None,
            valid_pixels_cuboid_tracks_params=self.valid_pixels_cuboid_tracks_params,
            max_intersections_per_ray=1,
            tqdm_disabled=self.tqdm_disabled,
        )

    def _compute_lidarpoints_dynamic(self, visualize=False) -> None:
        """
        Computes per-frame lidar dynamic point masks, either directly loading dynamic flag that comes
        with the data, or computing the flag by overlap with dynamic cuboid tracks.
        """

        def visualize_dynamic_points(points_all: np.ndarray, segmentation_mask: np.ndarray, tracks: dict) -> None:
            col = np.repeat(np.array([[0, 0, 255]], dtype=np.float32), points_all.shape[0], axis=0) / 255
            col[segmentation_mask, :] = np.array([[255, 0, 0]], dtype=np.float32)

            visualdebugger = get_visualdebugger()
            visualdebugger.clear()
            visualdebugger.add_point_cloud(
                "Lidar points", points_all, colors_quantities={"All points": col}, enabled=True, radius=0.0001
            )

            for track_id, (pose, half_dim) in tracks.items():
                cuboid_v = np.array(
                    [
                        [-0.5, -0.5, -0.5],
                        [0.5, -0.5, -0.5],
                        [0.5, 0.5, -0.5],
                        [-0.5, 0.5, -0.5],
                        [-0.5, -0.5, 0.5],
                        [0.5, -0.5, 0.5],
                        [0.5, 0.5, 0.5],
                        [-0.5, 0.5, 0.5],
                    ]
                )
                cuboid_v *= half_dim * 2
                cuboid_v = np.dot(cuboid_v, pose[:3, :3].T) + pose[:3, 3]

                cuboid_f = np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
                )

                cuboid_color = (235 / 255.0, 240 / 255.0, 11 / 255.0)

                visualdebugger.add_surface_mesh(
                    f"t-{track_id}",
                    cuboid_v,
                    cuboid_f,
                    color=cuboid_color,
                    edge_width=2,
                    edge_color=(0.0, 0.0, 0.0),
                    transparency=0.3,
                )
            visualdebugger.show()

        if len(self.lidars_frame_non_dynamic_points_masks):
            # masks already computed / exit early
            return

        if self.lidar_dynamic_points_params.method == "dynamic_tracks":
            # tracks are required for "dynamic_tracks" method
            self._compute_cuboid_tracks()

        for lidar_id in self.lidar_sensors.keys():
            lidar_frame_valid_points_masks = self.lidars_frame_valid_points_masks[lidar_id]
            lidar_frame_non_dynamic_points_masks = self.lidars_frame_non_dynamic_points_masks[lidar_id]

            lidar_sensor = self.lidar_sensors[lidar_id]

            for lidar_frame_idx in tqdm.tqdm(
                self.lidar_frame_ranges[lidar_id],
                desc=f"Dynamic Lidar Points Masks [lidars->frames] via '{self.lidar_dynamic_points_params.method}'",
                disable=self.tqdm_disabled,
            ):
                xyz_m_end: np.ndarray | None = None
                frame_dyn_tracks = {}  # only filled by for "dynamic_tracks" method / used for visualization
                non_dynamic_point_mask: np.ndarray
                match self.lidar_dynamic_points_params.method:
                    case "dynamic_flag":
                        # load dynamic flag from the source data, mark non-dynamic points as *valid* only
                        if (dynamic_flag := lidar_frame_dynamic_flag(lidar_sensor, lidar_frame_idx)) is None:
                            # dynamic flag data is not available - error out
                            raise ValueError(
                                f"{self.__class__.__name__}: dynamic_flag data missing for lidar {lidar_id} at frame {lidar_frame_idx}"
                            )

                        non_dynamic_point_mask = (
                            dynamic_flag != 1
                        )  # 1 ~ Point is classified to be dynamic [deprecated value of DynamicFlagState.DYNAMIC]

                    case "dynamic_tracks":
                        assert self.cuboidtracks_dynamic is not None and self.cuboids_df is not None

                        # load dynamic track observations in current frame time
                        frame_time_start = lidar_sensor.get_frame_timestamp_us(
                            lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                        )
                        frame_time_end = lidar_sensor.get_frame_timestamp_us(
                            lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                        )
                        frame_cuboids_df = self.cuboids_df[
                            (self.cuboids_df["timestamp_us"] >= frame_time_start)
                            & (self.cuboids_df["timestamp_us"] <= frame_time_end)
                        ]

                        dynamic_frame_cuboids_df = frame_cuboids_df[
                            frame_cuboids_df["track_id"].isin(self.cuboidtracks_dynamic.tracks_id)
                        ]

                        dynamic_cuboid_observations = [
                            ncore.data.CuboidTrackObservation.from_dict(row.to_dict())
                            for _, row in dynamic_frame_cuboids_df.iterrows()
                        ]

                        # Default annotation - all points are non-dynamic
                        non_dynamic_point_mask = np.full(lidar_sensor.get_frame_ray_bundle_count(lidar_frame_idx), True)

                        # If there are any dynamic tracks in this frame use them to labels the point cloud
                        if len(dynamic_cuboid_observations):
                            # Grab the current point-cloud to annotate it (motion-compensated to end-of-spin frame)
                            pc = lidar_sensor.get_frame_point_cloud(
                                lidar_frame_idx,
                                motion_compensation=True,
                                with_start_points=False,
                                return_index=0,  # closest returns only
                            )

                            non_dynamic_point_mask_tensor, cuboid_transforms, track_dim = compute_points_outside_tracks(
                                pc.xyz_m_end,
                                # make sure observations are relative to end-of-spin frame of the lidar
                                transform_cuboid_track_observations(
                                    observations=dynamic_cuboid_observations,
                                    pose_graph=lidar_sensor.pose_graph,
                                    target_frame_id=lidar_sensor.sensor_id,
                                    target_frame_timestamp_us=lidar_sensor.get_frame_timestamp_us(
                                        lidar_frame_idx,
                                        ncore.data.FrameTimepoint.END,
                                    ),
                                    tqdm_disabled=True,  # don't log individual transformations
                                ),
                                self.valid_points_cuboid_track_params.track_padding_m,
                            )
                            non_dynamic_point_mask = non_dynamic_point_mask_tensor.cpu().numpy()

                            # Visualize the dynamic tracks
                            if visualize:
                                for cuboid_observation_idx, cuboid_observation in enumerate(
                                    dynamic_cuboid_observations
                                ):
                                    frame_dyn_tracks[cuboid_observation.track_id] = (
                                        cuboid_transforms[cuboid_observation_idx].cpu().numpy(),
                                        track_dim[cuboid_observation_idx].cpu().numpy() / 2,
                                    )
                                    cuboid_observation_idx += 1

                    case _:
                        raise ValueError(
                            f"{self.__class__.__name__} unknown lidar_dynamic_point_method {self.lidar_dynamic_points_params.method}"
                        )

                # store per-point static / non-dynamic point flag
                lidar_frame_non_dynamic_points_masks[lidar_frame_idx] = PackedMask(non_dynamic_point_mask)

                if visualize and (xyz_m_end is not None):
                    visualize_dynamic_points(
                        xyz_m_end, lidar_frame_valid_points_masks[lidar_frame_idx].unpacked(), frame_dyn_tracks
                    )

    def _compute_valid_lidarpoints_dynamic(self, visualize=False) -> None:
        """
        Initializes or extends valid lidar point masks based on per-frame lidar dynamic point masks
        (see '_compute_lidarpoints_dynamic()'). Points classified as dynamic will be marked as invalid.
        """

        if not len(self.lidars_frame_non_dynamic_points_masks):
            # compute dynamic point masks explicitly if not present yet
            self._compute_lidarpoints_dynamic()

        for lidar_id in self.lidar_sensors.keys():
            lidar_frame_valid_points_masks = self.lidars_frame_valid_points_masks[lidar_id]
            lidar_frame_non_dynamic_points_masks = self.lidars_frame_non_dynamic_points_masks[lidar_id]

            lidar_sensor = self.lidar_sensors[lidar_id]

            for lidar_frame_idx in tqdm.tqdm(
                self.lidar_frame_ranges[lidar_id],
                desc=f"Dynamic Lidar Points Masks [lidars->frames] via '{self.lidar_dynamic_points_params.method}'",
                disable=self.tqdm_disabled,
            ):
                # load per-point static / non-dynamic point flag
                non_dynamic_point_mask = unpack_optional(
                    lidar_frame_non_dynamic_points_masks[lidar_frame_idx]
                ).unpacked()

                if (frame_valid_points_mask := lidar_frame_valid_points_masks.get(lidar_frame_idx)) is not None:
                    # extend existing mask
                    lidar_frame_valid_points_masks[lidar_frame_idx] = PackedMask(
                        frame_valid_points_mask.unpacked() & non_dynamic_point_mask
                    )
                else:
                    # initialize new mask
                    lidar_frame_valid_points_masks[lidar_frame_idx] = PackedMask(non_dynamic_point_mask)

    @staticmethod
    def _get_sorted_unique_sensor_ids_list(sensor_unique_ids: dict[str, UniqueSensorId]) -> list[str]:
        """Returns a list of unique sensor ids sorted by their unique index"""
        # sort unique sensor IDs by their unique index
        unique_ids_idx_sorted = sorted(sensor_unique_ids.values(), key=lambda u: u.idx)

        # drop indices, only return string IDs
        return [unique_id.id for unique_id in unique_ids_idx_sorted]

    def get_camera_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True, sorted by unique camera sensor idx) or logical (unique_sensors=False) camera sensor ids"""
        if unique_sensors:
            return self._get_sorted_unique_sensor_ids_list(self.camera_unique_ids)
        else:
            return self.all_camera_ids

    def get_lidar_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True, sorted by unique lidar sensor idx) or logical (unique_sensors=False) lidar sensor ids"""
        if unique_sensors:
            return self._get_sorted_unique_sensor_ids_list(self.lidar_unique_ids)
        else:
            return self.all_lidar_ids

    def get_n_frames_per_camera(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of total frame numbers per unique (unique_sensors=True) or logical (unique_sensors=False) camera sensor instance"""

        # make sure worker is initialized
        self._maybe_init_worker()

        if unique_sensors:
            n_frames_per_camera = np.zeros((len(self.all_camera_ids),), np.int32)

            for camera_id in self.all_camera_ids:
                camera_unique_idx = self.camera_unique_ids[camera_id].idx
                n_frames_per_camera[camera_unique_idx] += len(self.camera_frame_ranges[camera_id])

            return n_frames_per_camera
        else:
            return np.array(
                [len(self.camera_frame_ranges[camera_id]) for camera_id in self.all_camera_ids],
                dtype=np.int32,
            )

    def get_n_frames_per_lidar(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of total frame numbers per unique (unique_sensors=True) or logical (unique_sensors=False) lidar sensor instance"""

        # make sure worker is initialized
        self._maybe_init_worker()

        if unique_sensors:
            n_frames_per_lidar = np.zeros((len(self.all_lidar_ids),), np.int32)

            for lidar_id in self.all_lidar_ids:
                lidar_unique_idx = self.lidar_unique_ids[lidar_id].idx
                n_frames_per_lidar[lidar_unique_idx] += len(self.lidar_frame_ranges[lidar_id])

            return n_frames_per_lidar
        else:
            return np.array(
                [len(self.lidar_frame_ranges[lidar_id]) for lidar_id in self.all_lidar_ids],
                dtype=np.int32,
            )

    def _sensor_ids_to_unique_ids(self, input_sensor_ids: list[str], sensor_type: str) -> Generator[str, None, None]:
        """Converts logical or unique sensor ids to the corresponding set of unique ids for a certain sensor type.
        Valid unique ids are provided as is, logical sensor ids are expanded to the corresponding unique ids.
        Raises KeyError if input sensor was not found"""

        match sensor_type:
            case "camera":
                sensor_unique_ids = self.camera_unique_ids
            case "lidar":
                sensor_unique_ids = self.lidar_unique_ids
            case _:
                raise ValueError(f"{self.__class__.__name__} unknown sensor type {sensor_type}")

        for input_sensor_id in input_sensor_ids:
            sensor_found = False
            if input_sensor_id in sensor_unique_ids:
                # input sensor id is a logical sensor name
                yield sensor_unique_ids[input_sensor_id].id
                sensor_found = True
            else:
                # input sensor id might be a unique name - look for it and return if found
                for unique_sensor_id in sensor_unique_ids.values():
                    if unique_sensor_id.id == input_sensor_id:
                        yield input_sensor_id
                        sensor_found = True
            if not sensor_found:
                raise KeyError(f"{self.__class__.__name__} unknown sensor id {input_sensor_id}")

    def get_dynamic_semantic_class_ids(
        self, sensor_type: list[str], dynamic_class_names: list[str]
    ) -> Optional[list[int]]:
        """Returns semantic class IDs for given dynamic class names and sensor type.

        Args:
            sensor_type: List of sensor types (e.g., ["camera"], ["lidar"], ["camera", "lidar"])
            dynamic_class_names: List of semantic class names to consider dynamic
                                 (e.g., ["person", "car", "bicycle"])

        Returns:
            List of semantic class IDs for dynamic classes, or None if semantic data unavailable
        """
        self._maybe_init_worker()

        semantic_classes_map = self.get_semantic_classes_map(
            camera_semantics="camera" in sensor_type,
            lidar_semantics="lidar" in sensor_type,
        )

        if semantic_classes_map is None:
            return None

        dynamic_class_ids = []
        for class_name in dynamic_class_names:
            if class_name in semantic_classes_map:
                dynamic_class_ids.append(semantic_classes_map[class_name])

        return dynamic_class_ids if dynamic_class_ids else None

    def get_cuboid_tracks(
        self, dynamic_only: bool = False, world_frame: bool = False, include_generated: bool = False
    ) -> CuboidTracks:
        """Returns the cuboid tracks with optional filtering and transformation to world frame.

        Args:
            dynamic_only (bool, optional): if set return only tracks associated with dynamic objects. Defaults to False.
            world_frame (bool, optional): if set return tracks in world frame instead of NRE frame. Defaults to False.
            include_generated (bool, optional): if set return generated tracks. Defaults to False.

        Returns:
            CuboidTracks: cuboid tracks after optional filtering and transformation
        """

        # make sure we are initialized
        self._maybe_init_worker()

        # tracks are required
        self._compute_cuboid_tracks()

        # select all or dynamic only subset to return
        cuboidtracks = self.cuboidtracks_dynamic if dynamic_only else self.cuboidtracks_all
        assert cuboidtracks is not None, "NCoreDataSource: CuboidTracks not initialized"

        cuboidtracks = CuboidTracks.Ops.transform_with_frame_conversion(
            cuboidtracks,
            self.world_to_nre if not world_frame else None,
            np.identity(4, dtype=np.float32),
        )

        if include_generated:
            if self.generate_static_rigid:
                if self.generated_static_rigid_cuboidtracks is None:
                    self.generated_static_rigid_cuboidtracks = self._get_static_rigid_cuboid_tracks(
                        visibility_check_camera_ids=self.visibility_check_camera_ids,
                        max_visible_distance=self.max_visible_distance,
                        rigid_cls=self.static_rigid_classes,
                        point_cloud_path=self.static_rigid_point_cloud_path,
                    )
                # The static rigid cuboid tracks are generated in NRE frame
                static_rigid_cuboidtracks = CuboidTracks.Ops.transform_with_frame_conversion(
                    self.generated_static_rigid_cuboidtracks,
                    self.world_to_nre.inverse() if world_frame else None,
                    np.identity(4, dtype=np.float32),
                )
                cuboidtracks = CuboidTracks.Ops.concatenate(
                    [cuboidtracks, static_rigid_cuboidtracks],
                )

        return cuboidtracks

    def _get_static_rigid_cuboid_tracks(
        self,
        visibility_check_camera_ids: list[str],
        max_visible_distance: float,
        rigid_cls: list[str],
        point_cloud_path: Optional[str] = None,
    ) -> CuboidTracks:
        """Generate cuboid tracks for static rigid objects. The tracks will be in the NRE frame.

        Supports two modes:
        - Lidar mode: uses lidar point clouds with semantic labels (original behavior)
        - Lidar-free mode: uses accumulated point cloud PLY with segment_id when lidar
          semantic segmentation is unavailable. Requires point_cloud_path to be set.
        """
        # make sure we are initialized
        self._maybe_init_worker()

        assert self.aux_loader is not None, "aux data was not loaded"
        use_lidar = self.aux_loader.has_lidar_semantic_segmentation()

        if use_lidar:
            stuff_classes = unpack_optional(self.get_semantic_classes_map(False, True))
        else:
            assert point_cloud_path is not None, (
                "point_cloud_path must be set in generate_static_rigid_cuboid_tracks config for lidar-free pipelines"
            )
            stuff_classes = unpack_optional(self.get_semantic_classes_map(True, False))
        assert stuff_classes is not None, "Semantic class mapping not available"
        label_idx_list = [stuff_classes[cls] for cls in rigid_cls]
        device = torch.device("cuda")

        # Get the camera frusta
        camera_frusta = [
            (camera_frustum.to(torch.device("cuda")), timestamp_us)
            for camera_id in visibility_check_camera_ids
            for camera_frustum, timestamp_us in unpack_optional(
                self.get_camera_frusta(camera_id=camera_id, far_plane_depth=max_visible_distance)
            )
        ]

        # For lidar-free: load the accumulated point cloud once and transform to NRE frame
        ply_xyz_nre = None
        ply_segment_id = None
        if not use_lidar:
            import point_cloud_utils as pcu

            ply_data = pcu.load_triangle_mesh(point_cloud_path).vertex_data
            ply_xyz_world = ply_data.positions
            ply_segment_id = ply_data.custom_attributes.get("segment_id")
            assert ply_segment_id is not None, f"PLY file {point_cloud_path} does not contain segment_id attribute"
            assert ply_segment_id.min() >= 0 and ply_segment_id.max() <= 255, (
                f"ply_segment_id values out of uint8 range: min={ply_segment_id.min()}, max={ply_segment_id.max()}"
            )
            ply_segment_id = ply_segment_id.astype(np.uint8)
            ply_xyz_nre = self.world_to_nre.transform_points(ply_xyz_world)

        all_tracks_ids = []
        all_tracks_poses = []
        all_tracks_timestamps_us = []
        all_tracks_label_class = []
        all_tracks_flags = []
        all_cuboid_dims = []

        for idx, label_idx in enumerate(label_idx_list):
            rigid_cls_name = rigid_cls[idx]

            if use_lidar:
                # Original lidar path
                pcs = [
                    pc
                    for pc in unpack_optional(
                        self.get_lidar_point_clouds(
                            lidar_ids=self.all_lidar_ids,
                            valid_points_only=True,
                            non_dynamic_points_only=True,
                            color_type=None,
                            step_frame=1,
                            device=device,
                            visualize=False,
                            semantic_colormap=None,
                            lidar_point_cloud_selected_classes=[label_idx],
                        )
                    )
                ]
                if len(pcs) == 0:
                    continue
                point_cloud_xyz = PointCloud.collate_fn(pcs, device=device).xyz_end
            else:
                # Lidar-free path: filter accumulated point cloud by segment_id
                assert ply_xyz_nre is not None and ply_segment_id is not None
                class_mask = ply_segment_id == label_idx
                if not np.any(class_mask):
                    continue
                point_cloud_xyz = torch.from_numpy(ply_xyz_nre[class_mask].astype(np.float32)).to(device)

            if point_cloud_xyz.numel() == 0:
                continue

            # Use DBSCAN to cluster the points
            pc_pos = point_cloud_xyz.cpu().numpy()

            # Use cuML DBSCAN (gpu acceleration)
            dbscan = DBSCAN(eps=0.5, min_samples=50, metric="euclidean")
            dtype = "int64" if pc_pos.shape[0] >= 46340 else "int32"
            cluster_labels = dbscan.fit_predict(pc_pos, out_dtype=dtype)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            assert_default_device_on_local_rank()

            for cluster_idx in range(n_clusters):
                track_id = f"generated_{rigid_cls_name}_{cluster_idx}"
                cluster_mask = cluster_labels == cluster_idx
                cluster_points = point_cloud_xyz[cluster_mask]
                # Check if the point is visible from any of the cameras
                timestamp_list_us = []
                for camera_frustum, timestamp_us in camera_frusta:
                    # if 20% of the points are in the frustum, then consider it visible
                    if camera_frustum.points_in_frustum(cluster_points).sum() > 0.2 * cluster_points.shape[0]:
                        timestamp_list_us.append(timestamp_us)

                if len(timestamp_list_us) == 0:
                    continue
                # Determine the AABB track cuboid in NRE frame and start and end timestamp to the traffic light
                start_timestamp_us = min(timestamp_list_us)
                end_timestamp_us = max(timestamp_list_us)
                track_centroid = cluster_points.mean(dim=0).cpu().numpy()
                track_pose = np.identity(4)
                track_pose[:3, 3] = track_centroid
                track_dims = cluster_points.max(dim=0).values - cluster_points.min(dim=0).values
                all_tracks_ids.append(track_id)
                all_tracks_poses.append(np.array([track_pose, track_pose], dtype=np.float32))
                all_tracks_timestamps_us.append(np.array([start_timestamp_us, end_timestamp_us], dtype=np.int64))
                all_tracks_label_class.append(f"generated {rigid_cls_name}")
                all_tracks_flags.append(TrackFlags.NONE | TrackFlags.DYNAMIC)
                all_cuboid_dims.append(track_dims.cpu().numpy())

        return CuboidTracks.Factory.from_numpy(
            all_tracks_ids,
            all_tracks_poses,
            all_tracks_timestamps_us,
            all_tracks_label_class,
            all_tracks_flags,
            cuboids_dims=all_cuboid_dims,
        )

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
        The returned value is a TrackPointCloud.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        The provided cuboid tracks should be in the original world coordinates.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Default point-cloud sensor: *first* logical lidar
        """

        if camera_ids is not None:
            lidar_ids = [] if lidar_ids is None else list(lidar_ids)
        elif lidar_ids is not None:
            camera_ids = []
        else:
            # default to available lidar sensor if not provided explicitly, or to the first available camera sensor
            # if lidar is not available
            if self.all_lidar_ids:
                lidar_ids = [self.all_lidar_ids[0]]
                camera_ids = []
            else:
                lidar_ids = []
                camera_ids = [self.all_camera_ids[0]]

        # we can produce point clouds for cameras
        for pc in self.get_camera_track_point_clouds(
            camera_ids,
            cuboid_tracks,
            cuboid_dim_scale_factor,
            return_color,
            step_frame,
            keep_all_track_poses,
            device,
        ):
            yield pc

        # we can produce point clouds for lidars
        for pc in self.get_lidar_track_point_clouds(
            lidar_ids,
            cuboid_tracks,
            cuboid_dim_scale_factor,
            return_color,
            step_frame,
            keep_all_track_poses,
            device,
        ):
            yield pc

    def get_lidar_track_point_clouds(
        self,
        lidar_ids: list[str],
        cuboid_tracks: CuboidTracks,
        cuboid_dim_scale_factor: float = 1.0,
        return_color: bool = False,
        step_frame: int = 1,
        keep_all_track_poses: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> List[TrackPointCloud]:
        """Returns a generator for all object point-clouds available for lidar sensor, transformed into NRE frame.
        The returned value is a TrackPointCloud.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        The provided cuboid tracks should be in the original world coordinates.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Default point-cloud sensor: *first* logical lidar
        """

        # make sure we are initialized
        self._maybe_init_worker()

        if str(device) != "cpu":
            assert self.worker_id is None, "Loading the lidar point clouds on the GPU must be done on the main process"
            # verify that the current GPU is correct
            assert_device_on_local_rank(device)

        track_point_cloud_list: list[TrackPointCloud] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for unique_lidar_id in self._sensor_ids_to_unique_ids(lidar_ids, "lidar"):
                for lidar_id in self.lidar_sensors.keys():
                    if self.lidar_unique_ids[lidar_id].id != unique_lidar_id:
                        continue

                    lidar_frame_range = self.lidar_frame_ranges[lidar_id][::step_frame]
                    track_point_cloud_results: list[None | List[TrackPointCloud]] = [None] * len(lidar_frame_range)

                    def load_lidar_track_point_cloud(result_index: int, lidar_frame_index: int, lidar_id: str) -> None:
                        set_default_device()

                        lidar_sensor = self.lidar_sensors[lidar_id]

                        # First check if any of the cuboids exist in the current frame's timestamp range
                        xyz_time_start = lidar_sensor.get_frame_timestamp_us(
                            lidar_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                        )
                        xyz_time_end = lidar_sensor.get_frame_timestamp_us(
                            lidar_frame_index, frame_timepoint=ncore.data.FrameTimepoint.END
                        )

                        if not keep_all_track_poses:
                            current_cuboid_idxs = (
                                torch.logical_and(
                                    cuboid_tracks.tracks_timestamps_us >= xyz_time_start,
                                    cuboid_tracks.tracks_timestamps_us <= xyz_time_end,
                                )
                                .nonzero()
                                .squeeze()
                            )
                        else:
                            current_cuboid_idxs = (cuboid_tracks.tracks_timestamps_us >= 0).nonzero().squeeze()

                        if current_cuboid_idxs.dim() == 0:
                            current_cuboid_idxs = current_cuboid_idxs.unsqueeze(0).unsqueeze(1)

                        if current_cuboid_idxs.numel() == 0:
                            return

                        # Get existed cuboid poses and dimensions in the current frame
                        current_cuboid_jidx = (
                            torch.searchsorted(cuboid_tracks.tracks_packinfo[:, 0], current_cuboid_idxs, right=True) - 1
                        )
                        cuboid_poses = cuboid_tracks.tracks_poses[current_cuboid_idxs].data
                        cuboid_poses = tquat_to_se3_matrix(cuboid_poses, unbatch=False)
                        cuboid_dim = cuboid_tracks.cuboids_dims[current_cuboid_jidx] * cuboid_dim_scale_factor

                        frame_T_sensor_world = lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index)

                        # load motion-compensated point clouds (represented in sensor-frame)
                        pc = lidar_sensor.get_frame_point_cloud(
                            frame_index=lidar_frame_index,
                            motion_compensation=True,
                            with_start_points=True,
                            return_index=0,  # closest returns only
                        )
                        xyz_s = to_torch(unpack_optional(pc.xyz_m_start), device=device, non_blocking=True)
                        xyz_e = to_torch(pc.xyz_m_end, device=device, non_blocking=True)

                        T_sensor_nre = to_torch(
                            self.world_to_nre.transform_poses(frame_T_sensor_world), device=device, non_blocking=True
                        )

                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        # color the point clouds
                        if return_color:
                            color, color_mask, scale = self._color_pc_rgb(
                                xyz_sensor=xyz_e,
                                frame_T_sensor_world=frame_T_sensor_world,
                                lidar_frame_timestamp=xyz_time_end,
                                device=device,
                            )

                            color = color[color_mask]
                            scale = scale[color_mask]
                            xyz_s = xyz_s[color_mask]
                            xyz_e = xyz_e[color_mask]

                        xyz_s = (
                            (self.world_to_nre.target_scale * T_sensor_nre[:3, :3]) @ xyz_s.T + T_sensor_nre[:3, 3:4]
                        ).T
                        xyz_e = (
                            (self.world_to_nre.target_scale * T_sensor_nre[:3, :3]) @ xyz_e.T + T_sensor_nre[:3, 3:4]
                        ).T

                        # Spawn the points within each cuboid (in local coordinates)
                        ret = []
                        for cuboid_idx in range(cuboid_poses.shape[0]):
                            cuboid_T = cuboid_poses[cuboid_idx, :, :].to(device=device)
                            cuboid_extent = cuboid_dim[cuboid_idx, :3].to(device=device)

                            xyz_e_local = (cuboid_T[:3, :3].T @ (xyz_e.T - cuboid_T[:3, 3:4])).T
                            xyz_s_local = (cuboid_T[:3, :3].T @ (xyz_s.T - cuboid_T[:3, 3:4])).T
                            xyz_mask = torch.all(torch.abs(xyz_e_local) < cuboid_extent / 2, dim=1)

                            ret.append(
                                TrackPointCloud(
                                    track_id=cuboid_tracks.tracks_id[current_cuboid_jidx[cuboid_idx]],
                                    point_cloud=PointCloud(
                                        xyz_start=xyz_s_local[xyz_mask],
                                        xyz_end=xyz_e_local[xyz_mask],
                                        flags=None,
                                        color=color[xyz_mask] if return_color else None,
                                        semantic_class_id=None,
                                        camera_footprint_scale=scale[xyz_mask] if return_color else None,
                                    ),
                                )
                            )

                        track_point_cloud_results[result_index] = ret

                    for future in tqdm.tqdm(
                        concurrent.futures.as_completed(
                            [
                                executor.submit(load_lidar_track_point_cloud, index, lidar_frame_range[index], lidar_id)
                                for index in range(len(lidar_frame_range))
                            ]
                        ),
                        desc="Get Lidar Track Point Clouds",
                        total=len(lidar_frame_range),
                        disable=self.tqdm_disabled,
                    ):
                        try:
                            future.result()
                        except BaseException as exc:
                            raise exc  # Exceptions in threads need to be re-raised to be visible.

                    for track_point_cloud in track_point_cloud_results:
                        if track_point_cloud is not None and len(track_point_cloud) > 0:
                            track_point_cloud_list.extend(track_point_cloud)  # type: ignore

        return track_point_cloud_list

    def get_camera_track_point_clouds(
        self,
        camera_ids: list[str],
        cuboid_tracks: CuboidTracks,
        cuboid_dim_scale_factor: float = 1.0,
        return_color: bool = True,
        step_frame: int = 1,
        keep_all_track_poses: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> Generator[TrackPointCloud, None, None]:
        """Returns a generator for all object point-clouds available for camera sensor, transformed into NRE frame.
        The returned value is a TrackPointCloud.

        Use camera depth information to replace lidar to capture point clouds in each cuboid

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        The provided cuboid tracks should be in the original world coordinates.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Default point-cloud sensor: *first* logical camera

        Args:
            cuboid_tracks: Cuboid tracks in world coordinates
            cuboid_dim_scale_factor: Cuboid dimension scale factor
            camera_ids: List of camera IDs to use
            step_frame: Frame sampling step
            device: Computation device

        Returns:
            Generator for TrackPointCloud, containing point clouds in cuboid local coordinates
        """
        # make sure we are initialized
        self._maybe_init_worker()

        # provided samples are ordered by sensors, then chunks
        for unique_camera_id in self._sensor_ids_to_unique_ids(camera_ids, "camera"):
            assert (aux_loader := self.aux_loader) is not None, f"{self.__class__.__name__}: aux data was not loaded"

            assert aux_loader.has_depth(), f"{self.__class__.__name__}: Camera point clouds require depth information"

            for camera_id in self.camera_sensors.keys():
                if self.camera_unique_ids[camera_id].id != unique_camera_id:
                    continue

                camera_sensor = self.camera_sensors[camera_id]
                camera_params = camera_sensor.model_parameters
                frame_width, frame_height = camera_params.resolution

                all_pixels = to_torch(self.cameras_all_pixels[camera_id], device=device)
                all_camera_rays = to_torch(
                    self.cameras_all_rays[camera_id].reshape((-1, 3)),
                    device=device,
                )
                all_camera_footprints = to_torch(
                    self.cameras_all_footprints[camera_id].reshape(-1),
                    device=device,
                )

                for camera_frame_index in self.camera_frame_ranges[camera_id][::step_frame]:
                    # First check if any of the cuboids exist in the current frame's timestamp range
                    camera_frame_timestamp_start, camera_frame_timestamp_end = camera_sensor.frames_timestamps_us[
                        camera_frame_index
                    ]

                    if not keep_all_track_poses:
                        current_cuboid_idxs = (
                            torch.logical_and(
                                cuboid_tracks.tracks_timestamps_us >= camera_frame_timestamp_start,
                                cuboid_tracks.tracks_timestamps_us <= camera_frame_timestamp_end,
                            )
                            .nonzero()
                            .squeeze()
                        )
                    else:
                        current_cuboid_idxs = (cuboid_tracks.tracks_timestamps_us >= 0).nonzero().squeeze()

                    if current_cuboid_idxs.dim() == 0:
                        current_cuboid_idxs = current_cuboid_idxs.unsqueeze(0).unsqueeze(1)

                    if current_cuboid_idxs.numel() == 0:
                        continue

                    # Get existed cuboid poses and dimensions in the current frame
                    current_cuboid_jidx = (
                        torch.searchsorted(cuboid_tracks.tracks_packinfo[:, 0], current_cuboid_idxs, right=True) - 1
                    )
                    cuboid_poses = cuboid_tracks.tracks_poses[current_cuboid_idxs].data
                    cuboid_poses = tquat_to_se3_matrix(cuboid_poses, unbatch=False).to(device)
                    cuboid_dim = cuboid_tracks.cuboids_dims[current_cuboid_jidx] * cuboid_dim_scale_factor

                    depth = to_torch(
                        aux_loader.get_depth(
                            camera_id, camera_frame_timestamp_end, (frame_width, frame_height)
                        ).reshape(-1)
                        * self.world_to_nre.target_scale,
                        device=device,
                    )

                    valid_depth_mask = depth > 0
                    if not valid_depth_mask.any():
                        continue

                    depth = depth[valid_depth_mask]
                    pixels = all_pixels[valid_depth_mask]
                    camera_rays = all_camera_rays[valid_depth_mask]
                    footprints = all_camera_footprints[valid_depth_mask]

                    if return_color:
                        color = to_torch(
                            camera_sensor.get_frame_image_array(camera_frame_index).reshape((-1, 3)), device=device
                        )
                        color = color[valid_depth_mask]

                    assert aux_loader.get_depth_meta(camera_id)["method"] in (
                        "MetricDepthAnythingV2",
                        "sauron-z-depth",
                    ), f"{self.__class__.__name__}: assuming z-axis metric depth"

                    distance = torch.abs(depth / camera_rays[:, 2])
                    scale = footprints * distance

                    T_sensor_nre_startend = self.world_to_nre.transform_poses(
                        camera_sensor.get_frames_T_sensor_target(
                            target_node="world",
                            frame_indices=camera_frame_index,
                            frame_timepoint=None,  # evaluates both start/end timepoints
                        )  # 2x4x4
                    )

                    if str(device) == "cpu":
                        camera_model = self.camera_models[camera_id]
                        world_rays = camera_model.pixels_to_world_rays_shutter_pose(
                            pixels,
                            T_sensor_nre_startend[0],
                            T_sensor_nre_startend[1],
                        ).world_rays
                    else:
                        camera_model = ncore.sensors.CameraModel.from_parameters(camera_params, device=str(device))
                        world_rays = image_points_to_world_rays_shutter_pose(
                            camera_params,
                            camera_model.pixels_to_image_points(pixels),
                            se3_matrix_to_tquat(T_sensor_nre_startend),
                            torch.LongTensor([camera_frame_timestamp_start, camera_frame_timestamp_end]),
                        ).world_rays

                    # NRE space
                    xyz_s = world_rays[:, :3]
                    xyz_e = world_rays[:, :3] + world_rays[:, 3:] * distance.unsqueeze(-1)

                    # Spawn the points within each cuboid (in local coordinates)
                    for cuboid_idx in range(cuboid_poses.shape[0]):
                        cuboid_T = cuboid_poses[cuboid_idx, :, :]
                        cuboid_extent = cuboid_dim[cuboid_idx, :3]

                        xyz_e_local = (cuboid_T[:3, :3].T @ (xyz_e.T - cuboid_T[:3, 3:4])).T
                        xyz_s_local = (cuboid_T[:3, :3].T @ (xyz_s.T - cuboid_T[:3, 3:4])).T
                        xyz_mask = torch.all(torch.abs(xyz_e_local) < cuboid_extent / 2, dim=1)

                        yield TrackPointCloud(
                            track_id=cuboid_tracks.tracks_id[current_cuboid_jidx[cuboid_idx]],
                            point_cloud=PointCloud(
                                xyz_start=xyz_s_local[xyz_mask],
                                xyz_end=xyz_e_local[xyz_mask],
                                flags=None,
                                color=color[xyz_mask] if return_color else None,
                                semantic_class_id=None,
                                camera_footprint_scale=scale[xyz_mask],
                            ),
                        )

    def decode_image_cpu(
        self, image_handle: ncore.data.EncodedImageHandle, subsample_factor: float | None = None
    ) -> np.ndarray:
        """Decode encoded image data on CPU using enabled decoding backend and optionally subsample
        the image by the given factor (if possible while decoding)"""

        # load the encoded image data
        image_data = image_handle.get_data()

        # use simplejpeg to decode the image data if possible (wraps libjpeg-turbo)
        if self.jpeg_backend_cpu == "simplejpeg" and image_data.get_encoded_image_format().lower() in (
            "jpeg",
            "jpg",
        ):
            # decode jpeg header to obtain source resolution
            height, width, _, _ = simplejpeg.decode_jpeg_header(encoded_image := image_data.get_encoded_image_data())

            if subsample_factor is not None:
                # resize if requested and required
                assert (height := int(height)) % subsample_factor == 0 and (
                    width := int(width)
                ) % subsample_factor == 0, (
                    f"Image dimensions ({height}x{width}) must be divisible by subsample_factor ({subsample_factor})"
                )
                height = height // subsample_factor
                width = width // subsample_factor

            image = simplejpeg.decode_jpeg(
                encoded_image,
                fastdct=self.simplejpeg_fastdct,
                fastupsample=self.simplejpeg_fastupsample,
                min_width=width,
                min_height=height,
            )
        else:
            # fall back to ncore-internal (PIL-based) decoding and optional subsampling
            pil_image = image_data.get_decoded_image()
            if subsample_factor is not None and subsample_factor != 1.0:
                # resize if requested and required
                new_size = (
                    int(pil_image.width // subsample_factor),
                    int(pil_image.height // subsample_factor),
                )
                pil_image = pil_image.resize(new_size, resample=PILImage.Resampling.LANCZOS)
            image = np.asarray(pil_image)

        return image

    def _color_pc_rgb(
        self,
        xyz_sensor: torch.Tensor,
        frame_T_sensor_world: npt.NDArray,
        lidar_frame_timestamp: int,
        device: torch.device = torch.device("cuda"),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return colors, the observation scale, and color validity mask for the point cloud corresponding to a specific lidar frame.
        Points in regions of camera overlap we will be assigned a color and observation scale of a single camera only."""

        # Transform the point cloud to the world coordinate frame
        xyz_world = ncore_transformations.transform_point_cloud(
            xyz_sensor.unsqueeze(0), to_torch(frame_T_sensor_world, device=device).unsqueeze(0)
        ).squeeze(0)
        assert len(xyz_world.shape) == 2 and xyz_world.shape[1] == 3

        color = torch.zeros_like(xyz_world, dtype=torch.uint8)
        scale = torch.zeros_like(xyz_world[:, 0])
        color_mask = torch.zeros_like(xyz_world[:, 0], dtype=torch.bool)

        if not self.camera_sensors:
            raise ValueError(
                "[NCOREDataSource] At least a single camera needs to be available for point-cloud coloring"
            )

        for camera_id, camera_sensor in self.camera_sensors.items():
            # Find the closest camera frame
            cam_frame_index = camera_sensor.get_closest_frame_index(lidar_frame_timestamp)

            # Decode image
            img_frame = to_torch(self.decode_image_cpu(camera_sensor.get_frame_handle(cam_frame_index)), device="cpu")

            T_world_sensor = camera_sensor.get_frames_T_source_sensor(
                source_node="world",
                frame_indices=cam_frame_index,
                frame_timepoint=None,  # evaluates both start/end timepoints
            )  # 2x4x4

            T_sensor_world_end = ncore_transformations.se3_inverse(T_world_sensor[1])

            # Project pc to image points
            if str(device) == "cpu":
                camera_model = self.camera_models[camera_id]
                world_point_projections = camera_model.world_points_to_image_points_shutter_pose(
                    xyz_world,
                    T_world_sensor[0],
                    T_world_sensor[1],
                    return_valid_indices=True,
                )
            else:
                camera_params = self.camera_model_parameters[camera_id]
                world_point_projections = world_points_to_image_points_shutter_pose(
                    camera_params,
                    xyz_world,
                    se3_matrix_to_tquat(T_world_sensor),
                    torch.LongTensor(camera_sensor.frames_timestamps_us[cam_frame_index, :]),
                )

            pixels = world_point_projections.image_points.floor().int().cpu()
            valid_idx = unpack_optional(world_point_projections.valid_indices).to(device)  # torch

            point_colors = img_frame[pixels[:, 1], pixels[:, 0]]

            # note that for regions of overlap between cameras we will overwrite the colors
            color[valid_idx] = point_colors.to(device)
            color_mask[valid_idx] = True

            # We could use the T_world_sensors that camera_model.world_points_to_image_points_shutter_pose to compensate
            # for rolling shutter instead of using T_sensor_world_end but it will be slower and that level of accuracy is not important here.
            footprints = to_torch(self.cameras_all_footprints[camera_id], device="cpu")
            offset = (xyz_world[valid_idx] - torch.FloatTensor(T_sensor_world_end[:3, 3]).to(device)).norm(dim=-1)
            scale[valid_idx] = footprints[pixels[:, 1], pixels[:, 0]].to(device) * offset

        return color, color_mask, scale

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
        """Returns a generator for all point-clouds available for point-cloud sensor (lidar / camera with aux-depth), transformed into NRE frame.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Can be parameterized to only return valid (default), non-dynamic (default),
        and colored points colorized by one of the following strategies: "camera-rgb" (rgb scene colors),
        "semantics" (semantic colors obtained from shard meta data).

        If forced, at least a single point-cloud providing sensor needs to be available (otherwise result can be empty).

        Default point-cloud sensor: *first* logical lidar
        """

        # make sure we are initialized
        self._maybe_init_worker()

        if camera_ids is not None:
            lidar_ids = [] if lidar_ids is None else lidar_ids
        elif lidar_ids is not None:
            camera_ids = []
        else:
            # default to available lidar sensor if not provided explicitly, or to the first available camera sensor
            # if lidar is not available
            if self.all_lidar_ids:
                lidar_ids = [self.all_lidar_ids[0]]
                camera_ids = []
            else:
                lidar_ids = []
                camera_ids = [self.all_camera_ids[0]]

        if force and (not len(lidar_ids) and not len(camera_ids)):
            # require at least a single point-cloud providing sensor in forced mode
            raise ValueError(
                f"{self.__class__.__name__}: At least a single sensor required for forced point cloud generation"
            )

        if non_dynamic_points_only and len(lidar_ids):
            # compute dynamic point masks required to filter for dynamic points
            self._compute_lidarpoints_dynamic()

        semantic_colormap = (
            self.get_semantic_colormap(
                camera_semantics=len(camera_ids) > 0,
                lidar_semantics=len(lidar_ids) > 0,
            )
            if color_type == "semantics"
            else None
        )

        # aux depth-data is required for camera-based point-clouds - skip if not available
        if len(camera_ids):
            if self.aux_loader and self.aux_loader.has_depth():
                # we can produce point clouds for cameras
                for pc in self.get_camera_point_clouds(
                    camera_ids,
                    valid_points_only,
                    non_dynamic_points_only,
                    color_type,
                    step_frame,
                    device,
                    visualize,
                    semantic_colormap,
                ):
                    yield pc
            elif force:
                # insufficient aux-data for camera-based point clouds - error out if forced
                raise ValueError(f"{self.__class__.__name__}: aux depth data is required for camera-based point clouds")

        for pc in self.get_lidar_point_clouds(
            lidar_ids,
            valid_points_only,
            non_dynamic_points_only,
            color_type,
            step_frame,
            device,
            visualize,
            semantic_colormap,
        ):
            yield pc

    def get_camera_point_clouds(
        self,
        camera_ids: list[str],
        valid_points_only: bool,
        non_dynamic_points_only: bool,
        color_type: PointCloudColorType,
        step_frame: int,
        device: torch.device,
        visualize: bool,
        semantic_colormap: Optional[np.ndarray],
        filter_previous_predictions: bool = True,
    ) -> Generator[PointCloud, None, None]:
        """Returns a generator for all point-clouds available for the specified camera ids, transformed into NRE frame.

        To avoid redundancy and since camera depth predictions are imperfect (especially in far-away regions), we filter out
        points that are visible in the next camera frame by default.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        Can be parameterized to only return valid (default), non-dynamic (default),
        and colored points colorized by one of the following strategies: "camera-rgb" (rgb scene colors),
        "semantics" (semantic colors obtained from shard meta data).
        Args:
            camera_ids: camera ids to generate point clouds for [list[str]]
            valid_points_only: whether to filter out invalid points (typically points in the camera mask or that are semantically labeled as egovehicle) [bool]
            non_dynamic_points_only: whether to filter out dynamic points (as determined by their semantic label) [bool]
            color_type: color type to use for the point cloud (rgb or semantic) [PointCloudColorType]
            step_frame: frame step to for generating point clouds across frames [int]
            device: device to run the operation on [torch.device]
            visualize: whether to visualize the point cloud [bool]
            semantic_colormap: semantic colormap to use for the point cloud [np.ndarray]
            filter_previous_predictions: whether to filter out points that are visible in the previous camera frame [bool]
        """
        # make sure we are initialized
        self._maybe_init_worker()

        # TODO: Add multithreading as in get_lidar_point_clouds
        for unique_camera_id in self._sensor_ids_to_unique_ids(camera_ids, "camera"):
            aux_loader = self.aux_loader
            assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"
            assert aux_loader.has_depth(), (
                f"{self.__class__.__name__}: Depth must be available when projecting camera point clouds"
            )

            if non_dynamic_points_only or color_type == "semantics":
                assert aux_loader.has_semantic_segmentation(), (
                    f"{self.__class__.__name__}: Camera semantics must be available"
                )

            for camera_id in self.camera_sensors.keys():
                if self.camera_unique_ids[camera_id].id != unique_camera_id:
                    continue

                assert aux_loader.get_depth_meta(camera_id)["method"] in ("MetricDepthAnythingV2", "sauron-z-depth"), (
                    f"{self.__class__.__name__}: Metric depth must be available when projecting camera point clouds"
                )

                camera_sensor = self.camera_sensors[camera_id]
                exclude_classes = []
                for class_index, class_name in enumerate(
                    aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"]
                ):
                    if (valid_points_only and class_name in self.camera_point_cloud_ignore_classes) or (
                        non_dynamic_points_only and class_name in self.camera_point_cloud_dynamic_classes
                    ):
                        exclude_classes.append(class_index)

                all_pixels = to_torch(self.cameras_all_pixels[camera_id], device=device)
                all_camera_rays = to_torch(self.cameras_all_rays[camera_id].reshape((-1, 3)), device=device)
                all_camera_footprints = to_torch(self.cameras_all_footprints[camera_id].reshape(-1), device=device)
                camera_params = self.camera_model_parameters[camera_id]
                frame_width, frame_height = camera_params.resolution

                assert step_frame > 0, "step_frame must be positive"
                filter_step_frame = step_frame

                if filter_previous_predictions:
                    # check if the camera is facing in the direction of motion
                    # to do so we check if the position of the first frame is visible in the second frame

                    range_idxs = self.camera_frame_ranges[camera_id]
                    first_camera_frame_index = range_idxs[0]
                    first_frame_position = camera_sensor.get_frames_T_sensor_target(
                        "world", first_camera_frame_index, ncore.data.FrameTimepoint.START
                    )[:3, 3]

                    second_camera_frame_index = range_idxs[1]
                    second_frame_T_sensor_startend = self.world_to_nre.transform_poses(
                        np.stack(
                            [
                                camera_sensor.get_frames_T_sensor_target(
                                    "world", second_camera_frame_index, ncore.data.FrameTimepoint.START
                                ),
                                camera_sensor.get_frames_T_sensor_target(
                                    "world", second_camera_frame_index, ncore.data.FrameTimepoint.END
                                ),
                            ]
                        )
                    )
                    second_frame_nre_to_sensor_startend = ncore_transformations.se3_inverse(
                        second_frame_T_sensor_startend
                    )
                    second_frame_camera_timestamp_startend = camera_sensor.frames_timestamps_us[
                        second_camera_frame_index
                    ]

                    valid_indices = get_indices_of_points_visible_in_image(
                        first_frame_position[np.newaxis, :],
                        second_frame_nre_to_sensor_startend,
                        camera_params,
                        device,
                        second_frame_camera_timestamp_startend[0],
                        second_frame_camera_timestamp_startend[1],
                    )

                    facing_in_direction_of_motion = True if len(valid_indices) > 0 else False
                    if not facing_in_direction_of_motion:
                        filter_step_frame = -filter_step_frame

                previous_frame_prediction: Optional[PointCloud] = None
                for camera_frame_index in self.camera_frame_ranges[camera_id][::filter_step_frame]:
                    camera_frame_timestamp_start = camera_sensor.get_frame_timestamp_us(
                        camera_frame_index, ncore.data.FrameTimepoint.START
                    )
                    camera_frame_timestamp_end = camera_sensor.get_frame_timestamp_us(
                        camera_frame_index, ncore.data.FrameTimepoint.END
                    )
                    if aux_loader.has_semantic_segmentation():
                        semantic_class_id = np.asarray(
                            aux_loader.get_semantic_segmentation(camera_id, camera_frame_timestamp_end)
                        ).reshape(-1)
                    else:
                        semantic_class_id = None

                    color: Optional[np.ndarray]
                    match color_type:
                        case "camera-rgb":
                            color = camera_sensor.get_frame_image_array(camera_frame_index).reshape((-1, 3))
                        case "semantics":
                            color = color_pc_semantics(
                                to_torch(unpack_optional(semantic_class_id), device="cpu"),
                                to_torch(unpack_optional(semantic_colormap), device="cpu"),
                                -1,
                            ).numpy()
                        case None:
                            color = None
                        case _:
                            raise ValueError(f"Unexpected color type {color_type}.")

                    pixels = all_pixels
                    camera_rays = all_camera_rays
                    depth = (
                        aux_loader.get_depth(
                            camera_id, camera_frame_timestamp_end, (frame_width, frame_height)
                        ).reshape(-1)
                        * self.world_to_nre.target_scale
                    )

                    footprints = all_camera_footprints

                    valid_pixels_mask = self.cameras_frame_valid_pixels_masks[camera_id][camera_frame_index].unpacked()
                    if valid_points_only:
                        point_filter = valid_pixels_mask.reshape(-1)
                    else:
                        point_filter = np.ones(len(depth), dtype=bool)

                    for exclude_class in exclude_classes:
                        point_filter = np.logical_and(point_filter, semantic_class_id != exclude_class)

                    point_filter = np.logical_and(depth > 0, point_filter)
                    if not np.any(point_filter):
                        continue

                    depth = depth[point_filter]
                    pixels = pixels[point_filter]
                    camera_rays = camera_rays[point_filter]
                    footprints = footprints[point_filter]
                    if color is not None:
                        color = color[point_filter]
                    if semantic_class_id is not None:
                        semantic_class_id = semantic_class_id[point_filter]

                    distance = to_torch(depth, device=device) / camera_rays[:, 2]
                    scale = footprints * distance

                    T_sensor_startend = self.world_to_nre.transform_poses(
                        np.stack(
                            [
                                camera_sensor.get_frames_T_sensor_target(
                                    "world", camera_frame_index, ncore.data.FrameTimepoint.START
                                ),
                                camera_sensor.get_frames_T_sensor_target(
                                    "world", camera_frame_index, ncore.data.FrameTimepoint.END
                                ),
                            ]
                        )
                    )

                    # Project pc to image points
                    if str(device) == "cpu":
                        camera_model = self.camera_models[camera_id]
                        world_rays = camera_model.pixels_to_world_rays_shutter_pose(
                            pixels,
                            T_sensor_startend[0],
                            T_sensor_startend[1],
                        ).world_rays
                    else:
                        camera_model = ncore.sensors.CameraModel.from_parameters(camera_params, device=str(device))
                        world_rays = image_points_to_world_rays_shutter_pose(
                            camera_params,
                            camera_model.pixels_to_image_points(pixels),
                            se3_matrix_to_tquat(T_sensor_startend),
                            torch.LongTensor([camera_frame_timestamp_start, camera_frame_timestamp_end]),
                        ).world_rays

                    xyz = world_rays[:, :3] + world_rays[:, 3:] * distance.unsqueeze(-1)
                    flags = torch.zeros((len(xyz),), dtype=torch.int32, device=device)

                    # Load semantic labels, if available, and update flags
                    if semantic_class_id is not None:
                        flags |= RayFlags.VALID_SEMANTIC.value
                        flags[semantic_class_id == self.sensor_road_class_ids[camera_id]] |= RayFlags.ROAD_SEMANTIC
                        flags[semantic_class_id == self.sensor_sky_class_ids[camera_id]] |= RayFlags.SKY_SEMANTIC
                        for vehicle_class_id in self.sensor_vehicle_classes_ids[camera_id]:
                            flags[semantic_class_id == vehicle_class_id] |= RayFlags.VEHICLE_SEMANTIC

                    if color is not None and visualize:
                        visualise_point_cloud(xyz.cpu().numpy(), color / 255)

                    point_cloud = PointCloud(
                        xyz_start=xyz,
                        xyz_end=xyz,
                        flags=flags,
                        color=to_torch(color, device=device) if color is not None else None,
                        camera_footprint_scale=scale,
                        semantic_class_id=(
                            to_torch(semantic_class_id, device=device) if semantic_class_id is not None else None
                        ),
                        sensor_type=["camera"],
                    )

                    if filter_previous_predictions:
                        if previous_frame_prediction is not None:
                            nre_to_sensor_startend = ncore_transformations.se3_inverse(T_sensor_startend)
                            keep_mask = torch.ones(
                                previous_frame_prediction.xyz_start.shape[0], dtype=torch.bool, device=device
                            )
                            valid_indices = get_indices_of_points_visible_in_image(
                                previous_frame_prediction.xyz_end,
                                nre_to_sensor_startend,
                                camera_params,
                                device,
                                camera_frame_timestamp_start,
                                camera_frame_timestamp_end,
                                to_torch(valid_pixels_mask, device=device),
                            )
                            keep_mask[valid_indices] = False
                            yield previous_frame_prediction[keep_mask]

                        previous_frame_prediction = point_cloud
                    else:
                        yield point_cloud

                if previous_frame_prediction is not None:
                    yield previous_frame_prediction

    def get_lidar_point_clouds(
        self,
        lidar_ids: list[str],
        valid_points_only: bool,
        non_dynamic_points_only: bool,
        color_type: PointCloudColorType,
        step_frame: int,
        device: torch.device,
        visualize: bool,
        semantic_colormap: Optional[np.ndarray],
        lidar_point_cloud_selected_classes: Optional[list[int]] = None,
    ) -> list[PointCloud]:
        """Returns a list of all point-clouds available for the specified lidar ids, transformed into NRE frame.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        Can be parameterized to only return valid (default), non-dynamic (default),
        and colored points colorized by one of the following strategies: "camera-rgb" (rgb scene colors),
        "semantics" (semantic colors obtained from shard meta data).
        """
        # make sure we are initialized
        self._maybe_init_worker()

        if str(device) != "cpu":
            assert self.worker_id is None, "Loading the lidar point clouds on the GPU must be done on the main process"
            # verify that the current GPU is correct
            assert_device_on_local_rank(device)

        point_cloud_list: list[PointCloud] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for unique_lidar_id in self._sensor_ids_to_unique_ids(lidar_ids, "lidar"):
                lidar_aux_loader = self.aux_loader

                for lidar_id in self.lidar_sensors.keys():
                    if self.lidar_unique_ids[lidar_id].id != unique_lidar_id:
                        continue

                    lidar_sensor = self.lidar_sensors[lidar_id]

                    if has_lidar_semantics := (
                        lidar_aux_loader is not None and lidar_aux_loader.has_lidar_semantic_segmentation()
                    ):
                        ignore_label_class = unpack_optional(lidar_aux_loader).get_lidar_semantic_segmentation_meta(
                            lidar_id
                        )["ignore_label"]
                    else:
                        ignore_label_class = None

                    if not has_lidar_semantics and color_type == "semantics":
                        raise ValueError(
                            f'"semantics" coloring selected, but no semantic aux data available for lidar "{lidar_id}".'
                        )

                    lidar_frame_range = self.lidar_frame_ranges[lidar_id][::step_frame]
                    point_cloud_results: list[None | PointCloud] = [None] * len(lidar_frame_range)

                    def load_lidar_point_cloud(result_index: int, lidar_frame_index: int):
                        set_default_device()

                        point_filter: slice | torch.Tensor = slice(None)
                        if non_dynamic_points_only and valid_points_only:
                            point_filter = to_torch(
                                np.logical_and(
                                    self.lidars_frame_valid_points_masks[lidar_id][lidar_frame_index].unpacked(),
                                    self.lidars_frame_non_dynamic_points_masks[lidar_id][lidar_frame_index].unpacked(),
                                ),
                                device=device,
                            )
                        elif non_dynamic_points_only:
                            point_filter = to_torch(
                                self.lidars_frame_non_dynamic_points_masks[lidar_id][lidar_frame_index].unpacked(),
                                device=device,
                            )
                        elif valid_points_only:
                            point_filter = to_torch(
                                self.lidars_frame_valid_points_masks[lidar_id][lidar_frame_index].unpacked(),
                                device=device,
                            )

                        # load the point cloud data (represented in sensor-frame) and move to target device
                        pc = lidar_sensor.get_frame_point_cloud(
                            lidar_frame_index,
                            motion_compensation=True,
                            with_start_points=True,
                            return_index=0,  # closest returns only
                        )
                        xyz_s = to_torch(
                            unpack_optional(pc.xyz_m_start),
                            device=device,
                            non_blocking=True,
                        )[point_filter]
                        xyz_e = to_torch(
                            pc.xyz_m_end,
                            device=device,
                            non_blocking=True,
                        )[point_filter]
                        intensity = to_torch(
                            lidar_sensor.get_frame_ray_bundle_return_intensity(lidar_frame_index),
                            device=device,
                            non_blocking=True,
                        )[point_filter]

                        T_sensor_nre = to_torch(
                            self.world_to_nre.transform_poses(
                                lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index)
                            ),
                            device=device,
                            non_blocking=True,
                        )

                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        color = None
                        scale = None
                        color_mask = None

                        # extract color from camera sensors, if requested
                        if color_type == "camera-rgb":
                            color, color_mask, scale = self._color_pc_rgb(
                                xyz_e,
                                lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index),
                                lidar_sensor.get_frame_timestamp_us(lidar_frame_index),
                                device,
                            )
                            # mask out points that could not be colored
                            xyz_s = xyz_s[color_mask]
                            xyz_e = xyz_e[color_mask]
                            intensity = intensity[color_mask]
                            color = color[color_mask]
                            scale = scale[color_mask]
                        elif color_type == "generic-data":
                            color = to_torch(
                                lidar_sensor.get_frame_generic_data(lidar_frame_index, "rgb"), device=device
                            )[point_filter]

                        # transform points from sensor to NRE frame
                        xyz_s = (
                            (self.world_to_nre.target_scale * T_sensor_nre[:3, :3]) @ xyz_s.T + T_sensor_nre[:3, 3:4]
                        ).T
                        xyz_e = (
                            (self.world_to_nre.target_scale * T_sensor_nre[:3, :3]) @ xyz_e.T + T_sensor_nre[:3, 3:4]
                        ).T

                        lidar_timestamp = lidar_sensor.get_frame_timestamp_us(lidar_frame_index)
                        flags = torch.zeros((xyz_s.shape[0],), dtype=torch.int32, device=device)

                        # Load semantic labels, if available, and update flags
                        semantic_class_id = None
                        if has_lidar_semantics:
                            semantic_class_id = to_torch(
                                unpack_optional(lidar_aux_loader).get_lidar_semantic_segmentation(
                                    lidar_id, lidar_timestamp
                                ),
                                device=device,
                            )[point_filter]

                            if color_mask is not None:
                                semantic_class_id = semantic_class_id[color_mask]

                            if valid_points_only:
                                valid_mask = semantic_class_id != ignore_label_class
                                xyz_s = xyz_s[valid_mask]
                                xyz_e = xyz_e[valid_mask]
                                flags = flags[valid_mask]
                                semantic_class_id = unpack_optional(semantic_class_id)[valid_mask]
                                intensity = intensity[valid_mask]

                                if color is not None:
                                    color = color[valid_mask]
                                    scale = unpack_optional(scale)[valid_mask]
                            else:
                                # Set invalid flags
                                flags[semantic_class_id == ignore_label_class] |= RayFlags.INVALID

                            # Filter by lidar_point_cloud_selected_classes if specified
                            assert semantic_class_id is not None  # Always set inside has_lidar_semantics block
                            if lidar_point_cloud_selected_classes is not None:
                                selected_mask = torch.isin(
                                    semantic_class_id, torch.tensor(lidar_point_cloud_selected_classes, device=device)
                                )
                                xyz_s = xyz_s[selected_mask]
                                xyz_e = xyz_e[selected_mask]
                                flags = flags[selected_mask]
                                semantic_class_id = semantic_class_id[selected_mask]
                                intensity = intensity[selected_mask]

                                if color is not None:
                                    color = color[selected_mask]
                                    scale = unpack_optional(scale)[selected_mask] if scale is not None else None

                            # Set road semantic flags
                            flags |= RayFlags.VALID_SEMANTIC.value
                            flags[semantic_class_id == self.sensor_road_class_ids[lidar_id]] |= RayFlags.ROAD_SEMANTIC
                            flags[semantic_class_id == self.sensor_sky_class_ids[lidar_id]] |= RayFlags.SKY_SEMANTIC
                            for vehicle_class_id in self.sensor_vehicle_classes_ids[lidar_id]:
                                flags[semantic_class_id == vehicle_class_id] |= RayFlags.VEHICLE_SEMANTIC

                            if color_type == "semantics":
                                assert ignore_label_class is not None
                                color = color_pc_semantics(
                                    unpack_optional(semantic_class_id),
                                    to_torch(unpack_optional(semantic_colormap), device=device),
                                    ignore_label_class,
                                )

                        if color is not None and visualize:
                            visualise_point_cloud(xyz_e.cpu().numpy(), color.cpu().numpy() / 255)

                        point_cloud_results[result_index] = PointCloud(
                            xyz_start=xyz_s,
                            xyz_end=xyz_e,
                            flags=flags,
                            color=color if color is not None else None,
                            semantic_class_id=(semantic_class_id if semantic_class_id is not None else None),
                            sensor_type=["lidar"],
                            intensity=intensity,
                            camera_footprint_scale=scale if scale is not None else None,
                        )

                    for future in tqdm.tqdm(
                        concurrent.futures.as_completed(
                            [
                                executor.submit(load_lidar_point_cloud, index, lidar_frame_range[index])
                                for index in range(len(lidar_frame_range))
                            ]
                        ),
                        desc="Get Lidar Point Clouds",
                        total=len(lidar_frame_range),
                        disable=self.tqdm_disabled,
                    ):
                        try:
                            future.result()
                        except BaseException as exc:
                            raise exc  # Exceptions in threads need to be re-raised to be visible

                    assert None not in point_cloud_results
                    point_cloud_list.extend(point_cloud_results)  # type: ignore

        return point_cloud_list

    def get_camera_frusta(
        self,
        camera_id: Optional[str] = None,
        near_plane_depth: float = 0.1,
        far_plane_depth: float = 150.0,
        step_frame: int = 1,
    ) -> Generator[tuple[CameraFrustum, int], None, None]:
        """Returns a generator for all camera frusta for a given camera sensor, transformed into NRE frame.

        Camera sensor are specified by either logical or unique sensor IDs.

        A single camera sensor needs to be specified - defaults to first camera sensor if not specified."""

        assert near_plane_depth < far_plane_depth, (
            "[NCOREDataSource] Near plane depth of camera frustum is larger than far plane depth"
        )

        # make sure we are initialized
        self._maybe_init_worker()

        # default to first camera if not provided explicitly
        assert len(self.all_camera_ids), "[NCOREDataSource] no camera sensors loaded"
        camera_id = self.all_camera_ids[0] if camera_id is None else camera_id

        # provided samples are ordered by unique sensors
        for _ in self._sensor_ids_to_unique_ids([camera_id], "camera"):
            camera_sensor = self.camera_sensors[camera_id]
            camera_model = self.camera_models[camera_id]

            w = int(camera_model.resolution[0].item())
            h = int(camera_model.resolution[1].item())

            # Initialize the corner uv values
            corner_pixels = torch.tensor([[0, h], [0, 0], [w, 0], [w, h]], dtype=torch.int32)

            # Extract the rays in the camera coordinate system and compute the distance along the ray
            camera_rays = camera_model.pixels_to_camera_rays(corner_pixels)

            # Camera rays are already normalized
            near_plane_dists = near_plane_depth / camera_rays[:, 2:3]
            far_plane_dists = far_plane_depth / camera_rays[:, 2:3]

            for camera_frame_index in self.camera_frame_ranges[camera_id][::step_frame]:
                T_sensor_startend_nre = self.world_to_nre.transform_poses(
                    camera_sensor.get_frames_T_sensor_target(
                        target_node="world",
                        frame_indices=camera_frame_index,
                        frame_timepoint=None,  # evaluates both start/end timepoints
                    )  # 2x4x4
                )
                rays = camera_model.pixels_to_world_rays_mean_pose(
                    corner_pixels,
                    T_sensor_startend_nre[0],
                    T_sensor_startend_nre[1],
                    camera_rays=camera_rays,
                ).world_rays

                # For origin we take the mean value of the ray origins
                corners = torch.cat(
                    [
                        rays[:, :3] + rays[:, 3:6] * near_plane_dists * self.world_to_nre.target_scale,
                        rays[:, :3] + rays[:, 3:6] * far_plane_dists * self.world_to_nre.target_scale,
                    ],
                    dim=0,
                )

                yield (
                    CameraFrustum(corners=corners),
                    camera_sensor.get_frame_timestamp_us(camera_frame_index),
                )

    def get_semantic_classes_frame_masks(
        self, class_names: list[str], camera_semantics: bool, lidar_semantics: bool
    ) -> dict[str, dict[int, PackedMask]]:  # indexed as [unique-sensor-id][sensor-frame-index] -> PackedMask
        """Returns the boolean (packed) 1d masks for each frame of each camera / lidar, where the flag is True if
        the pixel/point belongs to one of the classes selected in class_names and false otherwise"""

        # make sure we are initialized
        self._maybe_init_worker()

        # aux data is required for semantic classes
        aux_loader = self.aux_loader
        assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"

        ret: dict[str, dict[int, PackedMask]] = defaultdict(dict)

        # check for availability of camera semantics
        semantic_class_id_map = self.get_semantic_classes_map(camera_semantics=True, lidar_semantics=False)
        assert semantic_class_id_map, f"{self.__class__.__name__}: No aux semantic classes map available"

        # iterate over each frame, collecting masks
        if camera_semantics:
            assert aux_loader.has_semantic_segmentation(), (
                f"{self.__class__.__name__}: No camera semantic segmentation auxiliary data available"
            )

            for camera_id, camera_frame_range in tqdm.tqdm(
                self.camera_frame_ranges.items(), desc="Semantic Classes Masks [cameras]", disable=self.tqdm_disabled
            ):
                camera_sensor = self.camera_sensors[camera_id]
                unique_camera_id = self.camera_unique_ids[camera_id].id
                for camera_frame_idx in tqdm.tqdm(
                    camera_frame_range, desc="Semantic Classes Masks [cameras->frames]", disable=self.tqdm_disabled
                ):
                    # load semantic annotation for current frame
                    semantic_mask = np.asarray(
                        aux_loader.get_semantic_segmentation(
                            camera_id,
                            int(
                                camera_sensor.get_frame_timestamp_us(
                                    camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                                )
                            ),
                        )
                    ).flatten()

                    # construct mask value
                    valid_semantic_mask = np.zeros_like(semantic_mask, dtype=bool)
                    for class_name in class_names:
                        valid_semantic_mask |= semantic_mask == semantic_class_id_map[class_name]

                    # store as memory efficient packed-mask
                    ret[unique_camera_id] |= {camera_frame_idx: PackedMask(valid_semantic_mask)}

        if lidar_semantics:
            assert aux_loader.has_lidar_semantic_segmentation(), (
                f"{self.__class__.__name__}: No lidar semantic segmentation auxiliary data available"
            )

            for lidar_id, lidar_frame_range in tqdm.tqdm(
                self.lidar_frame_ranges.items(), desc="Semantic Classes Masks [lidars]", disable=self.tqdm_disabled
            ):
                lidar_sensor = self.lidar_sensors[lidar_id]
                unique_lidar_id = self.lidar_unique_ids[lidar_id].id
                for lidar_frame_idx in tqdm.tqdm(
                    lidar_frame_range, desc="Semantic Classes Masks [lidars->frames]", disable=self.tqdm_disabled
                ):
                    # load semantic annotation for current frame
                    semantic_mask = aux_loader.get_lidar_semantic_segmentation(
                        lidar_id,
                        int(
                            lidar_sensor.get_frame_timestamp_us(
                                lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                            )
                        ),
                    )

                    # construct mask value
                    valid_semantic_mask = np.zeros_like(semantic_mask, dtype=bool)
                    for class_name in class_names:
                        valid_semantic_mask |= semantic_mask == semantic_class_id_map[class_name]

                    # store as memory efficient packed-mask
                    ret[unique_lidar_id] |= {lidar_frame_idx: PackedMask(valid_semantic_mask)}

        return ret

    # Data export functionality
    @dataclass(slots=True)
    class CameraFrameExport:
        """Represents one exported camera-associated frame"""

        sequence_id: str
        frame_idx: int
        timestamp_start_us: int
        timestamp_end_us: int
        T_sensor_to_world_start: npt.NDArray[np.float64]
        T_sensor_to_world_end: npt.NDArray[np.float64]
        camera_model_parameters: ncore.data.ConcreteCameraModelParametersUnion
        image_data: ncore.data.EncodedImageData  # TODO(janickm): should be exported publicly in NCore API
        valid_pixels_mask: npt.NDArray[np.bool_]

        frame_track_idxs: Optional[npt.NDArray[np.int32]]

        # Aux data - exported if aux is enabled
        sem_seg_meta: Optional[dict]
        sem_seg_image: Optional[PILImage.Image]

    def export_camera_frames(self, camera_id: str) -> Generator[NCOREDataSource.CameraFrameExport, None, None]:
        """Returns a generator for all camera frame data of a given sensor"""

        # make sure we are initialized
        self._maybe_init_worker()

        camera_sensor = self.camera_sensors[camera_id]
        camera_model_parameters = self.camera_model_parameters[camera_id]

        sem_seg_meta: dict | None = None
        if (aux_loader := self.aux_loader) is not None:
            sem_seg_meta = aux_loader.get_semantic_segmentation_meta(camera_id)

        for camera_frame_idx in self.camera_frame_ranges[camera_id]:
            sem_seg_image: PILImage.Image | None = None
            if aux_loader is not None:
                sem_seg_image = aux_loader.get_semantic_segmentation(
                    camera_id, int(camera_sensor.get_frame_timestamp_us(camera_frame_idx))
                )

            frame_track_idxs = self.cameras_frame_track_idxs[camera_id].get(camera_frame_idx)

            yield NCOREDataSource.CameraFrameExport(
                sequence_id=self.sequence_id,
                frame_idx=camera_frame_idx,
                timestamp_start_us=int(
                    camera_sensor.get_frame_timestamp_us(
                        camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    )
                ),
                timestamp_end_us=int(
                    camera_sensor.get_frame_timestamp_us(
                        camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                    )
                ),
                T_sensor_to_world_start=camera_sensor.get_frames_T_sensor_target(
                    "world", camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                ),
                T_sensor_to_world_end=camera_sensor.get_frames_T_sensor_target(
                    "world", camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                ),
                camera_model_parameters=camera_model_parameters,
                image_data=camera_sensor.get_frame_data(camera_frame_idx),
                valid_pixels_mask=self.cameras_frame_valid_pixels_masks[camera_id][camera_frame_idx].unpacked(),
                frame_track_idxs=frame_track_idxs,
                sem_seg_meta=sem_seg_meta,
                sem_seg_image=sem_seg_image,
            )

    @dataclass(slots=True)
    class LidarFrameExport:
        """Represents one exported lidar-associated frame"""

        sequence_id: str
        frame_idx: int
        timestamp_start_us: int
        timestamp_end_us: int
        T_sensor_to_world_start: npt.NDArray[np.float64]
        T_sensor_to_world_end: npt.NDArray[np.float64]
        xyz_s: npt.NDArray[np.float32]  # in sensor frame at end time
        xyz_e: npt.NDArray[np.float32]  # in sensor frame at end time
        intensity: npt.NDArray[np.float32]
        ray_drop: npt.NDArray[np.float32]
        dynamic_flag: npt.NDArray[np.int8] | None

    def export_lidar_frames(
        self, lidar_id: str, return_index: int = 0
    ) -> Generator[NCOREDataSource.LidarFrameExport, None, None]:
        """Returns a generator for all lidar frame data of a given sensor"""

        # make sure we are initialized
        self._maybe_init_worker()

        lidar_sensor = self.lidar_sensors[lidar_id]

        for lidar_frame_idx in self.lidar_frame_ranges[lidar_id]:
            pc = lidar_sensor.get_frame_point_cloud(
                frame_index=lidar_frame_idx,
                motion_compensation=True,
                with_start_points=True,
                return_index=0,  # closest returns only
            )
            xyz_s: np.ndarray = unpack_optional(pc.xyz_m_start)
            xyz_e = pc.xyz_m_end
            intensity = lidar_sensor.get_frame_ray_bundle_return_intensity(lidar_frame_idx, return_index=return_index)
            raydrop = np.zeros_like(intensity)
            dynamic_flag = lidar_frame_dynamic_flag(lidar_sensor, lidar_frame_idx)

            yield NCOREDataSource.LidarFrameExport(
                sequence_id=self.sequence_id,
                frame_idx=lidar_frame_idx,
                timestamp_start_us=int(
                    lidar_sensor.get_frame_timestamp_us(
                        lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    )
                ),
                timestamp_end_us=int(
                    lidar_sensor.get_frame_timestamp_us(lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END)
                ),
                T_sensor_to_world_start=lidar_sensor.get_frames_T_sensor_target(
                    "world", lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                ),
                T_sensor_to_world_end=lidar_sensor.get_frames_T_sensor_target(
                    "world", lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                ),
                xyz_s=xyz_s,
                xyz_e=xyz_e,
                intensity=intensity,
                ray_drop=raydrop,
                dynamic_flag=dynamic_flag,
            )

    def get_rig_trajectories(self) -> RigTrajectories:
        # implements the protocol: RigTrajectoriesProvider

        # make sure we are initialized
        self._maybe_init_worker()

        def rig_trajectories_generator() -> Generator[RigTrajectories.RigTrajectory, None, None]:
            """Produces individual rig trajectories"""

            ## collect rig poses
            loader = self.sequence_loader
            assert loader is not None, f"NCOREDataSource: sequence_loader is not initialized"

            # TODO: frame-pose only data might fail here as there are no rig poses and might require refined logic
            rig_world_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
                loader.pose_graph.get_edge("rig", "world"),
                msg="Rig-to-world poses required for rig-trajectories",
            )

            # all rig poses
            T_rig_world: np.ndarray = rig_world_edge.T_source_target
            T_rig_world_timestamps_us: np.ndarray = unpack_optional(
                rig_world_edge.timestamps_us, msg="Rig-to-world pose requires to be dynamic"
            )

            # rig poses restricted to the time range of interest
            T_rig_world = T_rig_world[poses_range := self.time_range_us.cover_range(T_rig_world_timestamps_us)]
            T_rig_world_timestamps_us = T_rig_world_timestamps_us[poses_range]

            # TODO: this pose resolution will differ for the current free-pose hacks that use different per-frame poses
            #       relative to the rig in V3 data - consider a to replicate a special case / config here as a fallback
            cameras_frame_T_rig_worlds = {
                camera_id: to_torch(
                    self.camera_sensors[camera_id].get_frames_T_source_target(
                        source_node="rig",
                        target_node="world",
                        frame_indices=np.array(camera_frame_range),
                        frame_timepoint=None,  # evaluates both start/end timepoints
                    ),  # Nx2x4x4
                    device="cpu",
                    dtype=torch.float64,
                )
                for camera_id, camera_frame_range in self.camera_frame_ranges.items()
            }

            def get_sensor_frame_timestamps_us(sensor: ncore.data.SensorProtocol, frame_range: range) -> torch.Tensor:
                return torch.tensor(
                    sensor.frames_timestamps_us[frame_range],
                    # cast np.uint64 -> torch.int64 (torch doesn't support unsigned integers - mind potential overflows in the future)
                    dtype=torch.int64,
                    device="cpu",
                ).reshape(
                    len(frame_range), 2
                )  # only a sanity check to be sure that sensor.frames_timestamps_us is Nx2 givin per-frame start/end timestamps

            def get_rig_bbox(
                loader: ncore.data.SequenceLoaderProtocol,
            ) -> Optional[ncore.data.BBox3]:
                """Attempts to parse a dataset-specific ego bounding box centered around the rig frame, if available

                Note: this is currently a non-specified property of NCore and might be abscent in most datasets
                """

                # try to load ego bbox from NV rig meta-data (available, e.g., for clipgt data)
                if nv_rig := loader.generic_meta_data.get("nv-rig", None):
                    # parse a NV rig file for it's body-associated bbox
                    return ncore.data.BBox3.from_array(ncore_internal_rig.vehicle_bbox(cast(dict, nv_rig)))
                elif vehicle_bbox := cast(dict | None, loader.generic_meta_data.get("vehicle-bbox", None)):
                    # load bbox from vehicle-bbox field in meta data (available for some datasets, e.g., PAI)
                    return ncore.data.BBox3(
                        centroid=tuple(vehicle_bbox["centroid"]),
                        dim=tuple(vehicle_bbox["dim"]),
                        rot=tuple(vehicle_bbox["rot"]),
                    )

                # no further ego bbox candidates available
                return None

            yield RigTrajectories.RigTrajectory(
                sequence_id=self.sequence_id,
                rig_bbox=get_rig_bbox(loader),
                T_rig_worlds=to_torch(
                    T_rig_world,
                    device="cpu",
                    # forcing to float64 here to maintain data-structure contract,
                    # altough world poses are really only float32 precision (global poses have higher precision)
                    dtype=torch.float64,
                ),
                T_rig_world_timestamps_us=to_torch(
                    T_rig_world_timestamps_us
                    # cast np.uint64 -> torch.int64 (torch doesn't support unsigned integers - mind potential overflows in the future)
                    .astype(np.int64),
                    device="cpu",
                ),
                cameras_frame_timestamps_us={
                    self.camera_unique_ids[camera_id].id: get_sensor_frame_timestamps_us(
                        self.camera_sensors[camera_id],
                        self.camera_frame_ranges[camera_id],
                    )
                    for camera_id in self.all_camera_ids
                },
                lidars_frame_timestamps_us={
                    self.lidar_unique_ids[lidar_id].id: get_sensor_frame_timestamps_us(
                        self.lidar_sensors[lidar_id], self.lidar_frame_ranges[lidar_id]
                    )
                    for lidar_id in self.all_lidar_ids
                },
                cameras_linear_start_frame_indices={
                    self.camera_unique_ids[camera_id].id: self.camera_linear_start_frame_indices[camera_id]
                    for camera_id in self.all_camera_ids
                },
                lidars_linear_start_frame_indices={
                    self.lidar_unique_ids[lidar_id].id: self.lidar_linear_start_frame_indices[lidar_id]
                    for lidar_id in self.all_lidar_ids
                },
                cameras_frame_T_rig_worlds={
                    self.camera_unique_ids[camera_id].id: cameras_frame_T_rig_worlds[camera_id]
                    for camera_id in self.all_camera_ids
                },
            )

        def camera_calibrations_generator() -> Generator[tuple[str, RigTrajectories.CameraCalibration], None, None]:
            """Produces individual camera calibrations"""

            for camera_id, camera_sensor in self.camera_sensors.items():
                unique_id = self.camera_unique_ids[camera_id]
                yield (
                    unique_id.id,  # unique sensor id
                    RigTrajectories.CameraCalibration(
                        sequence_id=self.sequence_id,
                        logical_sensor_name=camera_id,  # logical sensor name
                        unique_sensor_idx=unique_id.idx,  # unique sensor index
                        T_sensor_rig=to_torch(
                            unpack_optional(
                                camera_sensor.T_sensor_rig, msg=f"Missing T_sensor_rig for camera {camera_id}"
                            ),
                            device="cpu",
                        ),
                        camera_model_parameters=self.camera_model_parameters[camera_id],
                    ),
                )

        def lidar_calibrations_generator() -> Generator[tuple[str, RigTrajectories.LidarCalibration], None, None]:
            """Produces individual lidar calibrations"""

            for lidar_id, lidar_sensor in self.lidar_sensors.items():
                unique_id = self.lidar_unique_ids[lidar_id]
                yield (
                    unique_id.id,  # unique sensor id
                    RigTrajectories.LidarCalibration(
                        sequence_id=self.sequence_id,
                        logical_sensor_name=lidar_id,  # logical sensor name
                        unique_sensor_idx=unique_id.idx,  # unique sensor index
                        T_sensor_rig=to_torch(
                            unpack_optional(
                                lidar_sensor.T_sensor_rig, msg=f"Missing T_sensor_rig for lidar {lidar_id}"
                            ),
                            device="cpu",
                        ),
                        lidar_model_parameters=self.lidar_model_parameters[lidar_id],
                    ),
                )

        return RigTrajectories(
            T_world_base=to_torch(self.T_world_world_global, device="cpu"),
            world_to_nre=self.world_to_nre,
            rig_trajectories=list(rig_trajectories_generator()),
            camera_calibrations=OrderedDict(list(camera_calibrations_generator())),
            lidar_calibrations=OrderedDict(list(lidar_calibrations_generator())),
        )

    def get_semantic_colormap(self, camera_semantics: bool, lidar_semantics: bool) -> Optional[npt.NDArray[np.uint8]]:
        """Returns the semantic colormap for requested sensor types (camera and/or lidar), if available

        Raise an error if the mapping is not consistent across all sequences and enabled sensors
        """

        # make sure worker is initialized
        self._maybe_init_worker()

        semantic_colormap: Optional[npt.NDArray[np.uint8]] = None

        def collect_semantic_colormap(
            semantic_colormap: Optional[npt.NDArray[np.uint8]], semantic_colormap_current: npt.NDArray[np.uint8]
        ) -> npt.NDArray:
            if semantic_colormap is None:
                semantic_colormap = semantic_colormap_current
            elif (semantic_colormap != semantic_colormap_current).any():
                raise ValueError(
                    f"{self.__class__.__name__} semantic color map is not consistent across all sequences and "
                    f"requested sensors (camera={camera_semantics}, lidar={lidar_semantics})"
                )
            return semantic_colormap

        aux_loader = self.aux_loader
        assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"

        if camera_semantics and aux_loader.has_semantic_segmentation():
            for camera_id in self.all_camera_ids:
                semantic_colormap_current = np.array(
                    aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_colors"], dtype=np.uint8
                )

                semantic_colormap = collect_semantic_colormap(semantic_colormap, semantic_colormap_current)

        if lidar_semantics and aux_loader.has_lidar_semantic_segmentation():
            for lidar_id in self.all_lidar_ids:
                semantic_colormap_current = np.array(
                    aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_colors"],
                    dtype=np.uint8,
                )

                semantic_colormap = collect_semantic_colormap(semantic_colormap, semantic_colormap_current)

        return semantic_colormap

    def get_semantic_classes_map(self, camera_semantics: bool, lidar_semantics: bool) -> Optional[dict[str, int]]:
        """
        Returns a dictionary mapping semantic class names to their corresponding class indices
        for requested sensor types (camera and/or lidar).

        Raise an error if the mapping is not consistent across all sequences and enabled sensors
        """

        # make sure worker is initialized
        self._maybe_init_worker()

        semantic_mapping: Optional[dict[str, int]] = None

        def collect_semantic_mapping(
            semantic_mapping: Optional[dict[str, int]], semantic_mapping_current: dict[str, int]
        ) -> dict[str, int]:
            if semantic_mapping is None:
                semantic_mapping = semantic_mapping_current
            elif semantic_mapping != semantic_mapping_current:
                raise ValueError(
                    f"{self.__class__.__name__} semantic class mapping is not consistent across all sequences and "
                    f"requested sensors (camera={camera_semantics}, lidar={lidar_semantics})"
                )
            return semantic_mapping

        aux_loader = self.aux_loader
        assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"

        if camera_semantics and aux_loader.has_semantic_segmentation():
            for camera_id in self.all_camera_ids:
                semantic_mapping_current = {
                    class_name: class_idx
                    for class_idx, class_name in enumerate(
                        aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"]
                    )
                }

                semantic_mapping = collect_semantic_mapping(semantic_mapping, semantic_mapping_current)

        if lidar_semantics and aux_loader.has_lidar_semantic_segmentation():
            for lidar_id in self.all_lidar_ids:
                semantic_mapping_current = {
                    class_name: class_idx
                    for class_idx, class_name in enumerate(
                        aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_classes"]
                    )
                }

                semantic_mapping = collect_semantic_mapping(semantic_mapping, semantic_mapping_current)

        return semantic_mapping
