# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import dataclasses
import json
import logging

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import numpy as np
import numpy.typing as npt
import torch

from scipy import ndimage
from upath import UPath

import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import nre.utils.ncore_utils as ncore_utils

from nre.datasets.tracks import CuboidTracks, CuboidTracksDataPack, TrackFlags
from nre.datasets.utils import compute_cuboid_df, consolidate_cuboid_tracks
from nre.nrm.config.dataset import BaseNCoreNRMDatasetConfig
from nre.nrm.datasets.nrm_base import BaseNRMIndexableDataset, CameraSubsampler, NRMDataError
from nre.nrm.datasets.registry import register as register_dataset
from nre.nrm.datasets.samplers import (
    AdaptiveSequentialFrameBatchSampler,
    FrameBatchSamplerReturn,
    sample_lidar_frame_batch,
)
from nre.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    LidarFrameLabels,
    NRMDataBatch,
)
from nre.utils.files import parse_universal_path
from nre.utils.geometry import se3_matrix_inverse
from nre.utils.lidar_model import get_lidar_model_parameters_with_fallbacks
from nre.utils.misc import to_torch, unpack_optional
from nre.utils.types import FrameConversion, HalfClosedInterval, RayFlags, RigTrajectories


logger = logging.getLogger(__name__)


def subranges_to_intervals(timestamps_us: np.ndarray, subranges: list[tuple[float, float]]) -> list[HalfClosedInterval]:
    intervals: list[HalfClosedInterval] = []
    min_timestamps_us: int = int(timestamps_us.min())
    max_timestamps_us: int = int(timestamps_us.max())
    inner_timestamps_us = max_timestamps_us - min_timestamps_us
    for start, end in subranges:
        intervals.append(
            HalfClosedInterval(
                min_timestamps_us + int(start * inner_timestamps_us), min_timestamps_us + int(end * inner_timestamps_us)
            )
        )
    return intervals


def interval_list_intersect(
    intervals: list[HalfClosedInterval], other_interval: HalfClosedInterval
) -> list[HalfClosedInterval]:
    """
    Returns a list of intervals that are the intersection of the given intervals with the other_interval.
    """
    intersected_intervals: list[HalfClosedInterval] = []
    for interval in intervals:
        intersection = interval.intersection(other_interval)
        if intersection is not None:
            intersected_intervals.append(intersection)
    return intersected_intervals


def get_lidar_sensor_from_sequence_loader(
    sequence_loader: ncore.data.SequenceLoaderProtocol, lidar_id_candidates: list[str]
) -> ncore.data.LidarSensorProtocol:
    """
    Get the lidar sensor from the sequence loader by the lidar id, the function will return the first available lidar sensor.
    """
    for lidar_id in lidar_id_candidates:
        if lidar_id in sequence_loader.lidar_ids:
            return sequence_loader.get_lidar_sensor(lidar_id)
    raise ValueError(
        f"Lidar sensor with id candidates {lidar_id_candidates} not found in the sequence loader, available lidar ids: {sequence_loader.lidar_ids}"
    )


@register_dataset("nrm-ncore")
class NCoreNRMDataset(BaseNRMIndexableDataset):
    """
    The native ncore dataset loader
    """

    UNCONDITIONALLY_DYNAMIC_LABELS: set[str] = set(
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

    @dataclass(kw_only=True, frozen=True)
    class UniqueFrameId:
        sensor_id: str
        frame_idx: int

    @dataclass(frozen=True)
    class ExtendedCameraId:
        """Extended camera id that includes potential external ncore data sources"""

        camera_id: str
        unique_sensor_idx: int
        external_ncore_path: str | None = None
        sample_ratio: float = 1.0

        @staticmethod
        def from_config(
            config: str | BaseNCoreNRMDatasetConfig.ExternalSupervisionCameraIdConfig, unique_sensor_idx: int = -1
        ):
            if isinstance(config, str):
                return NCoreNRMDataset.ExtendedCameraId(
                    camera_id=config, unique_sensor_idx=unique_sensor_idx, external_ncore_path=None, sample_ratio=1.0
                )
            else:
                return NCoreNRMDataset.ExtendedCameraId(
                    camera_id=config.camera_id,
                    unique_sensor_idx=config.unique_sensor_idx,
                    external_ncore_path=config.ncore_path,
                    sample_ratio=config.sample_ratio,
                )

        def __str__(self) -> str:
            if self.external_ncore_path is None:
                return self.camera_id
            # Replace slashes since these names will be used to save e.g. vis files into filesystem.
            return f"{self.camera_id}-({self.external_ncore_path.replace('/', '_')})"

        @property
        def loader_key(self) -> str:
            return self.main_loader_key() if self.external_ncore_path is None else self.external_ncore_path

        @property
        def canonical_order(self) -> str:
            """Canonical order to be in rig trajectory and the data batch."""
            seq = f"{self.unique_sensor_idx:03d}"
            if self.external_ncore_path is not None:
                seq += f"-{self.external_ncore_path}"
            return seq

        @staticmethod
        def main_loader_key() -> str:
            return "main"

    @dataclass
    class LoadersAndSensorsResult:
        """Result of loading sequence loaders, aux loaders, and camera/lidar sensors for an ncore sequence."""

        T_rig_worlds_with_timestamps_us: dict[str, tuple[np.ndarray, np.ndarray]]
        sequence_loaders: dict[str, ncore.data.SequenceLoaderProtocol]
        aux_loaders: dict[str, ncore_utils.AuxShardDataLoader]
        camera_sensors: dict["NCoreNRMDataset.ExtendedCameraId", ncore.data.CameraSensorProtocol]
        lidar_sensors: dict[str, ncore.data.LidarSensorProtocol]

    def __init__(self, config: BaseNCoreNRMDatasetConfig, split: str = "train"):
        self.ncore_json_list_path = parse_universal_path(
            unpack_optional(config.ncore_json_list_path), s3_block_size_mb=config.s3_block_size_mb
        )
        self.ncore_json_base_path = (
            parse_universal_path(
                config.ncore_json_base_path,
                s3_block_size_mb=config.s3_block_size_mb,
                s3_cache_type=config.s3_cache_type,
            )
            if config.ncore_json_base_path is not None
            else None
        )
        self.open_consolidated = config.open_consolidated
        self.camera_max_fov_deg = config.camera_max_fov_deg
        self.n_camera_mask_dilation_iterations = config.n_camera_mask_dilation_iterations
        self.split = split

        self.all_supervision_camera_ids: list[NCoreNRMDataset.ExtendedCameraId] = []
        for camera_idx, camera_id_config in enumerate(config.supervision_camera_ids):
            # For string-based camera ids, we directly use their sequence as in the config.
            # For external supervision cameras, the unique sensor index is specified directly in the config.
            self.all_supervision_camera_ids.append(
                NCoreNRMDataset.ExtendedCameraId.from_config(camera_id_config, camera_idx)
            )
        self.all_context_camera_ids: list[NCoreNRMDataset.ExtendedCameraId] = []
        for camera_id_config in config.context_camera_ids:
            try:
                camera_id_idx = [str(c) for c in self.all_supervision_camera_ids].index(str(camera_id_config))
            except ValueError as e:
                raise ValueError(
                    f"Context camera {camera_id_config} not found in supervision cameras {self.all_supervision_camera_ids}"
                ) from e
            self.all_context_camera_ids.append(self.all_supervision_camera_ids[camera_id_idx])

        self.aux_data_params = config.aux_data
        self.cuboid_tracks_params = config.cuboid_tracks_params
        self.lidar_frame_batch_params = config.lidar_frame_batch

        # V3 sequence loader parameters
        self.cuboid_loading_max_workers: Optional[int] = config.cuboid_loading_max_workers

        # V4 sequence loader parameters
        self.poses_component_group: str = config.poses_component_group
        self.intrinsics_component_group: str = config.intrinsics_component_group
        self.masks_component_group: str = config.masks_component_group
        self.cuboids_component_group: str = config.cuboids_component_group

        # camera_ids and lidar_ids determine the unique sensor indices within this dataset.
        # [Note that when co-training on multiple datasets, it is currently fine to share index space since it's only used for distinguishing sensors.]
        self.lidar_ids: list[str] = []
        if self.cuboid_tracks_params.lidar_id:
            self.lidar_ids.append(self.cuboid_tracks_params.lidar_id)
        self.lidar_ids = sorted(list(set(self.lidar_ids)))
        assert len(self.lidar_ids) < 2, "Only one LiDAR sensor is supported for now"

        self.ncore_json_paths: list[UPath] = []
        with self.ncore_json_list_path.open("r") as f:
            for line in f.readlines():
                line = line.strip()
                if self.ncore_json_base_path is not None:
                    ncore_path = self.ncore_json_base_path / line
                else:
                    ncore_path = parse_universal_path(
                        line, s3_block_size_mb=config.s3_block_size_mb, s3_cache_type=config.s3_cache_type
                    )
                self.ncore_json_paths.append(ncore_path)
        n_ncore_json_files = len(self.ncore_json_paths)
        self.sequence_subranges: dict[str, list[tuple[float, float]]] = {}

        if (subrange_json_path := config.subrange_json_path) is not None:
            with Path(subrange_json_path).open("r") as f:
                subrange_data = json.load(f)
            for sequence_id, subranges in subrange_data.items():
                self.sequence_subranges[sequence_id] = [(s[0], s[1]) for s in subranges]
            # Filter ncore_json_paths to be only including those with subranges
            self.ncore_json_paths = [path for path in self.ncore_json_paths if path.stem in self.sequence_subranges]
        else:
            self.sequence_subranges = {path.stem: [(0.0, 1.0)] for path in self.ncore_json_paths}

        logger.info(
            f"Loaded {len(self.ncore_json_paths)}/{n_ncore_json_files} samples from {self.ncore_json_list_path}"
        )

        self.num_samples_per_sequence: int = config.frame_batch_sampler.n_samples_per_sequence

        # Whether to (or not) consolidate rendering batch to save memory
        self.compute_rendering_data = config.compute_rendering_data

        # Cache for _get_loaders_and_sensors: at most one entry, keyed by ncore_json_path (only used when not in a worker)
        self.cache_loaders_and_sensors = config.cache_loaders_and_sensors
        self._loaders_sensors_cache: dict[str, NCoreNRMDataset.LoadersAndSensorsResult] = {}

        # Camera and lidar id mappings
        self.camera_id_mapping = config.camera_id_mapping
        self.lidar_id_mapping = config.lidar_id_mapping

        # Current config might not be concrete (i.e. augmentations might not be applied yet)
        self.non_concrete_config = config

    def __len__(self) -> int:
        return len(self.ncore_json_paths) * self.num_samples_per_sequence

    def _compute_cuboid_tracks(
        self,
        context_frame_batch: FrameBatchSamplerReturn,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        camera_sensors: dict[ExtendedCameraId, ncore.data.CameraSensorProtocol],
        lidar_sensors: dict[str, ncore.data.LidarSensorProtocol],
        T_world_ref: np.ndarray,
    ) -> CuboidTracksDataPack:
        if self.cuboid_tracks_params.lidar_id is None:
            cuboid_tracks = CuboidTracks.Factory.empty(device=torch.device("cpu"))
            return CuboidTracksDataPack(
                tracks_data=cuboid_tracks.tracks_data,
                cuboidtracks_data=cuboid_tracks.cuboidtracks_data,
            )

        assert self.cuboid_tracks_params.lidar_id in lidar_sensors, "Required LiDAR sensor not found in the dataset"

        frame_batch_min_timestamps_us: int = int(1e16)
        frame_batch_max_timestamps_us: int = 0
        for sensor_id, frame_idxs in context_frame_batch.sampled_sensor_frame_idxs.items():
            sensor = [v for k, v in (camera_sensors | lidar_sensors).items() if str(k) == sensor_id][0]
            sensor_min_timestamp_us = sensor.get_frame_timestamp_us(min(frame_idxs), ncore.data.FrameTimepoint.START)
            sensor_max_timestamp_us = sensor.get_frame_timestamp_us(max(frame_idxs), ncore.data.FrameTimepoint.END)
            frame_batch_min_timestamps_us = min(frame_batch_min_timestamps_us, sensor_min_timestamp_us)
            frame_batch_max_timestamps_us = max(frame_batch_max_timestamps_us, sensor_max_timestamp_us)

        time_range_us = HalfClosedInterval(
            frame_batch_min_timestamps_us - self.cuboid_tracks_params.track_extrapolate_timestamps_us,
            frame_batch_max_timestamps_us + self.cuboid_tracks_params.track_extrapolate_timestamps_us,
        )
        cuboids_df = compute_cuboid_df(sequence_loader, time_range_us, serialize_observation=False)

        # First associate all tracks within the batch
        all_batch_tracks = consolidate_cuboid_tracks(
            cuboids_df=cuboids_df,
            sequence_loader=sequence_loader,
            track_label_sources=[self.cuboid_tracks_params.track_label_source],
            track_min_centroid_rig_dist_m=self.cuboid_tracks_params.track_min_centroid_rig_dist_m,
            T_world_world_base=T_world_ref,
            tqdm_disabled=True,
        )

        all_track_ids = []
        all_tracks_poses = []
        all_tracks_timestamps_us = []
        all_tracks_label_class = []
        all_tracks_flags = []
        all_cuboid_dims = []

        for track_id, track in all_batch_tracks.items():
            if len(track["timestamps_us"]) <= 1:
                continue

            # initialize track-associated pose-interpolator
            poses_list: list[np.ndarray] = track["poses"]
            timestamps_us_list: list[int] = track["timestamps_us"]
            track_flags = TrackFlags.NONE

            # Perform extrapolation just in case this chunk hits the clip boundary.
            # Note: track-pose extrapolation is intentionally unconditional. The former
            # `track_extrapolate: bool` off-switch was removed because every production
            # config relied on it — do not reintroduce the switch.
            # extrapolate first pose to the past
            poses_list.insert(
                0,
                # extrapolate into pre-time P = (P_1 @ P_0^-1)^-1 @ P_0 = (P_0 @ P_1^-1) @ P_0
                (poses_list[0] @ ncore_transformations.se3_inverse(poses_list[1])) @ poses_list[0],
            )
            timestamps_us_list.insert(0, timestamps_us_list[0] - (timestamps_us_list[1] - timestamps_us_list[0]))

            # extrapolate last pose to the future
            poses_list.append(
                # extrapolate into post-time P = (P_N @ P_{N-1}^-1) @ P_N
                (poses_list[-1] @ ncore_transformations.se3_inverse(poses_list[-2])) @ poses_list[-1],
            )
            timestamps_us_list.append(timestamps_us_list[-1] + (timestamps_us_list[-1] - timestamps_us_list[-2]))

            poses = np.stack(poses_list, dtype=np.float32)
            timestamps_us = np.stack(timestamps_us_list)

            track_travel_distance_m: float = np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]).item()
            # Scale travel distance by actual sensor timestamp differences
            if (timestamps_diff_us := (timestamps_us.max() - timestamps_us.min())) > 0:
                track_travel_distance_m *= float(frame_batch_max_timestamps_us - frame_batch_min_timestamps_us) / float(
                    timestamps_diff_us
                )

            track_is_dynamic: bool = (
                track["label_class"] in self.UNCONDITIONALLY_DYNAMIC_LABELS
                or track_travel_distance_m > self.cuboid_tracks_params.track_min_travel_distance_m
            )
            if track_is_dynamic:
                track_flags |= TrackFlags.DYNAMIC

            # store all tracks unconditionally
            all_track_ids.append(track_id)
            all_tracks_poses.append(poses)
            all_tracks_timestamps_us.append(timestamps_us)
            all_tracks_label_class.append(track["label_class"])
            all_tracks_flags.append(track_flags)
            all_cuboid_dims.append(track["dimension"])

        # Map to member structs
        cuboid_tracks = CuboidTracks.Factory.from_numpy(
            all_track_ids,
            all_tracks_poses,
            all_tracks_timestamps_us,
            all_tracks_label_class,
            all_tracks_flags,
            cuboids_dims=all_cuboid_dims,
            device=torch.device("cpu"),
        )
        return CuboidTracksDataPack(
            tracks_data=cuboid_tracks.tracks_data,
            cuboidtracks_data=cuboid_tracks.cuboidtracks_data,
        )

    def _load_data_batch(
        self,
        frame_batch: FrameBatchSamplerReturn,
        camera_idx_mapping: dict[UniqueFrameId, int],
        camera_sensors: dict[ExtendedCameraId, ncore.data.CameraSensorProtocol],
        lidar_idx_mapping: dict[UniqueFrameId, int],
        lidar_sensors: dict[str, ncore.data.LidarSensorProtocol],
        aux_loaders: dict[str, ncore_utils.AuxShardDataLoader],
        camera_subsampler: CameraSubsampler,
    ) -> DataBatch:
        """
        Load actual data batch given the sampled frame batch. idx_mapping is used to determine the unique frame index for the frame meta.
        """
        # Used to spawn a warning if both depth aux and lidar is loaded.
        depth_aux_loaded: bool = False

        ## Load cameras

        # This determines the ordering of images in the actual batch.
        # As long as network is equivariant to the order of images, this is not important.
        frame_batch_camera_ids = [
            matched_camera_ids[0]
            for camera_id_name in frame_batch.sampled_sensor_frame_idxs.keys()
            if len(matched_camera_ids := [c for c in camera_sensors.keys() if str(c) == camera_id_name]) > 0
        ]
        frame_batch_camera_ids = sorted(frame_batch_camera_ids, key=lambda x: x.canonical_order)

        # Read Camera-based data
        camera_batch_list: list[DataBatch.Camera] = []
        for camera_id in frame_batch_camera_ids:
            frame_idxs = frame_batch.sampled_sensor_frame_idxs[str(camera_id)]
            if camera_id not in camera_sensors:
                continue
            camera_sensor = camera_sensors[camera_id]
            camera_width = camera_subsampler.frame_width
            camera_height = camera_subsampler.frame_height

            # Statically unmasked pixels
            if (camera_mask_array := ncore_utils.get_camera_sensor_mask(camera_sensor)) is not None:
                camera_mask_array = camera_subsampler.apply_frame_data(camera_mask_array)
                camera_mask_array = ndimage.binary_dilation(
                    camera_mask_array, iterations=self.n_camera_mask_dilation_iterations
                )
                invalid_ego_mask = cast(np.ndarray, camera_mask_array)
            else:
                # No mask / consider all pixels as valid
                invalid_ego_mask = np.zeros((camera_height, camera_width), dtype=bool)

            # Determine unique sensor index mapping
            unique_sensor_idx = camera_id.unique_sensor_idx
            for frame_idx in frame_idxs:
                # Obtain this timestamp to index aux data
                frame_end_timestamp_us = int(
                    camera_sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.END)
                )

                # initialize ray flags (potentially updated below by additional label-derived flags)
                flags = torch.full(
                    (camera_height, camera_width),
                    RayFlags.RGB_LABEL.value,
                    dtype=torch.int32,
                    device="cpu",
                )

                # For non-main loaders, we assume they are synthetic
                if camera_id.loader_key != camera_id.main_loader_key():
                    flags |= RayFlags.SYNTHETIC.value

                # Collect labels data
                labels = CameraFrameLabels()
                frame_image_array = camera_sensor.get_frame_image_array(frame_idx).astype(np.float32) / 255.0
                frame_image_array = camera_subsampler.apply_frame_data(frame_image_array)
                labels.rgb = to_torch(frame_image_array, device="cpu").unsqueeze(0)

                # Load auxiliary information
                if camera_id.loader_key in aux_loaders:
                    aux_loader = aux_loaders[camera_id.loader_key]
                    data_camera_id = self.camera_id_mapping.get(camera_id.camera_id, camera_id.camera_id)
                    sky_mask: np.ndarray | bool = False
                    if self.aux_data_params.semantic_segmentation and aux_loader.has_semantic_segmentation(
                        data_camera_id
                    ):
                        semantics = np.asarray(
                            aux_loader.get_semantic_segmentation(data_camera_id, frame_end_timestamp_us)
                        )
                        stuff_classes = aux_loader.get_semantic_segmentation_meta(data_camera_id)["stuff_classes"]
                        sky_mask = semantics == stuff_classes.index("sky")

                        # classify sampled rays for sky and road
                        flags |= RayFlags.VALID_SEMANTIC.value
                        semantics = camera_subsampler.apply_frame_data(semantics)
                        flags[semantics == stuff_classes.index("sky")] |= RayFlags.SKY_SEMANTIC.value
                        flags[semantics == stuff_classes.index("road")] |= RayFlags.ROAD_SEMANTIC.value
                        for vehicle_class in ["car", "truck", "bus", "train", "motorcycle", "bicycle"]:
                            flags[semantics == stuff_classes.index(vehicle_class)] |= RayFlags.VEHICLE_SEMANTIC.value
                        # Some egocar regions are detected in semantic segmentations.
                        if "egocar" in stuff_classes:
                            invalid_ego_mask |= semantics == stuff_classes.index("egocar")
                        # (H, W) -> (1, H, W, 1)
                        # We rely fully on flags to store semantic information, so no need to pass on labels.semantic
                        # labels.semantic = to_torch(semantics, device="cpu")[None, ..., None]

                    if self.aux_data_params.depth and aux_loader.has_depth(data_camera_id):
                        # Distance map already processed by the data pre-processing stage
                        depth = aux_loader.get_depth(data_camera_id, frame_end_timestamp_us)
                        # For DS-based data, depth is nan/inf for sky regions, correct them to be 0.
                        # Applied before subsampling to avoid enlarging nan regions.
                        depth[~np.isfinite(depth) | sky_mask] = 0.0
                        depth = camera_subsampler.apply_depth_data(depth, mode="nearest-min")
                        # (H, W) -> (1, H, W, 1)
                        labels.metric_distance = to_torch(depth, device="cpu")[None, ..., None]
                        depth_aux_loaded = True

                    if self.aux_data_params.egomask and aux_loader.has_egomask(data_camera_id):
                        egomask = aux_loader.get_egomask(data_camera_id, 0)
                        egomask = camera_subsampler.apply_frame_data(egomask)
                        egomask = ndimage.binary_dilation(egomask, iterations=self.n_camera_mask_dilation_iterations)
                        invalid_ego_mask |= cast(np.ndarray, egomask)

                # store invalid flag of pixels (usually only required in validation mode)
                flags[invalid_ego_mask] |= RayFlags.INVALID
                flags[invalid_ego_mask] |= RayFlags.EGO_SEMANTIC
                labels.flags = flags[None, ..., None]  # (H, W) -> (1, H, W, 1)

                camera_batch_list.append(
                    DataBatch.Camera(
                        meta=[
                            FrameMeta(
                                unique_sensor_idx=unique_sensor_idx,
                                unique_frame_idx=camera_idx_mapping[
                                    self.UniqueFrameId(sensor_id=str(camera_id), frame_idx=frame_idx)
                                ],
                                # We don't have to pass in subsample here as we already store subsampled intrinsics
                                # in the rig trajectory.
                                subsample=None,
                            )
                        ],
                        labels=labels,
                    )
                )

        ## Load lidars
        lidar_batch_list: list[DataBatch.Lidar] = []
        for unique_sensor_idx, lidar_id in enumerate(self.lidar_ids):
            if lidar_id not in frame_batch.sampled_sensor_frame_idxs.keys():
                continue
            if lidar_id not in lidar_sensors:
                continue

            lidar_sensor = lidar_sensors[lidar_id]

            lidar_model_parameters = get_lidar_model_parameters_with_fallbacks(lidar_sensor, True)
            if lidar_model_parameters is None:
                continue

            height, width = lidar_model_parameters.n_rows, lidar_model_parameters.n_columns

            for frame_idx in frame_batch.sampled_sensor_frame_idxs[lidar_id]:
                pc = lidar_sensor.get_frame_point_cloud(
                    frame_index=frame_idx,
                    motion_compensation=True,
                    with_start_points=True,
                    return_index=0,  # closest returns only
                )
                _xyz_s: np.ndarray = unpack_optional(pc.xyz_m_start)
                _xyz_e = pc.xyz_m_end
                if len(_xyz_s) == 0 or len(_xyz_e) == 0:
                    continue

                model_elements: npt.NDArray[np.uint16] = unpack_optional(
                    lidar_sensor.get_frame_ray_bundle_model_element(frame_idx),
                    msg="Lidar sensor must provide model elements",
                )
                if depth_aux_loaded:
                    logger.warning(
                        f"Both depth aux data and lidar data from {lidar_id} are loaded, which might cause inefficiencies in data loading."
                    )
                rows, cols = model_elements[:, 0], model_elements[:, 1]

                _distance = np.linalg.norm(_xyz_e - _xyz_s, axis=1, keepdims=True)
                distance = np.full((height, width, 1), fill_value=np.nan, dtype=np.float32)
                distance[rows, cols, :] = _distance

                flags = torch.full((height, width, 1), RayFlags.DROPPED.value, dtype=torch.int32, device="cpu")
                flags[rows, cols, :] = 0  # set flags for non-dropped rays

                lidar_batch_list.append(
                    DataBatch.Lidar(
                        meta=[
                            FrameMeta(
                                unique_sensor_idx=unique_sensor_idx,
                                unique_frame_idx=lidar_idx_mapping[
                                    self.UniqueFrameId(sensor_id=lidar_id, frame_idx=frame_idx)
                                ],
                            )
                        ],
                        labels=LidarFrameLabels(
                            flags=flags.unsqueeze(0),
                            distance=to_torch(distance, device="cpu").unsqueeze(0),
                        ),
                    )
                )

        return DataBatch(
            camera=DataBatch.Camera.collate_fn(camera_batch_list),
            lidar=DataBatch.Lidar.collate_fn(lidar_batch_list) if len(lidar_batch_list) > 0 else None,
        )

    def _get_rig_trajectory(
        self,
        sequence_id_prefix: str,
        frame_batch: FrameBatchSamplerReturn,
        camera_sensors: dict[ExtendedCameraId, ncore.data.CameraSensorProtocol],
        lidar_sensors: dict[str, ncore.data.LidarSensorProtocol],
        T_world_ref: np.ndarray,
        T_rig_worlds_with_timestamps_us: dict[str, tuple[np.ndarray, np.ndarray]],
        camera_subsampler: CameraSubsampler,
    ) -> tuple[RigTrajectories, dict[UniqueFrameId, int], dict[UniqueFrameId, int]]:
        """
        Obtain rig-trajectory based on the sampled sensors.
        The rig trajectory will contain the full rig poses, frame_batch-sampled and subsampled cameras/lidars.

        This will additionally return a UniqueFrameId to index mapping, which matches the logic of Camera/LidarFreePoseViewGeometry
        so we can properly query a frame via its unique frame idx.
        """
        ## Load cameras

        # loader_key -> camera_id_name -> timestamps_us
        frame_timestamps_us_list: list[tuple[int, int]] = []
        camera_frame_timestamps_us: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        all_camera_model_parameters: dict[
            NCoreNRMDataset.ExtendedCameraId, ncore.data.ConcreteCameraModelParametersUnion
        ] = {}
        camera_idx_mapping: dict[NCoreNRMDataset.UniqueFrameId, int] = {}

        # Find the matching ExtendedCameraId given the string name from the sampler.
        frame_batch_camera_ids = [
            matched_camera_ids[0]
            for camera_id_name in frame_batch.sampled_sensor_frame_idxs.keys()
            if len(matched_camera_ids := [c for c in camera_sensors.keys() if str(c) == camera_id_name]) > 0
        ]
        # This determines the OrderedDict ordering of cameras in the rig trajectory.
        frame_batch_camera_ids = sorted(frame_batch_camera_ids, key=lambda x: x.canonical_order)

        current_unique_frame_idx: int = 0
        for camera_id in frame_batch_camera_ids:
            camera_sensor = camera_sensors[camera_id]
            camera_model_parameters_copy = dataclasses.replace(camera_sensor.model_parameters)

            # Some camera models have bad linear_cde values, manually fix them without overwriting originals
            if isinstance(camera_model_parameters_copy, ncore.data.FThetaCameraModelParameters) and np.all(
                camera_model_parameters_copy.linear_cde == 0.0
            ):
                camera_model_parameters_copy.linear_cde = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            if isinstance(
                camera_model_parameters_copy,
                (ncore.data.FThetaCameraModelParameters, ncore.data.OpenCVFisheyeCameraModelParameters),
            ):
                # (This would make boundary pixels of omnidirectional cameras to be classified as invalid)
                camera_model_parameters_copy.max_angle = min(
                    np.deg2rad(self.camera_max_fov_deg) / 2.0, camera_model_parameters_copy.max_angle
                )

            camera_model_parameters = camera_subsampler.apply_camera_parameters(camera_model_parameters_copy)
            all_camera_model_parameters[camera_id] = camera_model_parameters
            frame_timestamps_us_list = []
            for frame_idx in frame_batch.sampled_sensor_frame_idxs[str(camera_id)]:
                frame_start_timestamp_us = int(
                    camera_sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.START)
                )
                frame_end_timestamp_us = int(
                    camera_sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.END)
                )
                frame_timestamps_us_list.append((frame_start_timestamp_us, frame_end_timestamp_us))

                # NB [JH]: We must ensure that the sequence of iteration matches the logic of CameraFreePoseViewGeometry
                camera_idx_mapping[NCoreNRMDataset.UniqueFrameId(sensor_id=str(camera_id), frame_idx=frame_idx)] = (
                    current_unique_frame_idx
                )
                current_unique_frame_idx += 1

            camera_frame_timestamps_us[camera_id.loader_key][str(camera_id)] = torch.tensor(
                frame_timestamps_us_list, dtype=torch.int64, device="cpu"
            )

        ## Load lidars (always assume loader key is "main")
        lidar_loader_key = NCoreNRMDataset.ExtendedCameraId.main_loader_key()
        lidar_frame_timestamps_us: dict[str, torch.Tensor] = {}
        all_lidar_model_parameters: dict[str, ncore.data.ConcreteLidarModelParametersUnion | None] = {}
        lidar_idx_mapping: dict[NCoreNRMDataset.UniqueFrameId, int] = {}
        frame_batch_lidar_ids = [l for l in self.lidar_ids if l in frame_batch.sampled_sensor_frame_idxs.keys()]

        current_unique_frame_idx = 0
        for lidar_id in frame_batch_lidar_ids:
            lidar_sensor = lidar_sensors[lidar_id]
            all_lidar_model_parameters[lidar_id] = get_lidar_model_parameters_with_fallbacks(lidar_sensor, True)
            frame_timestamps_us_list = []
            for frame_idx in frame_batch.sampled_sensor_frame_idxs[lidar_id]:
                frame_start_timestamp_us = int(
                    lidar_sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.START)
                )
                frame_end_timestamp_us = int(
                    lidar_sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.END)
                )
                frame_timestamps_us_list.append((frame_start_timestamp_us, frame_end_timestamp_us))
                lidar_idx_mapping[NCoreNRMDataset.UniqueFrameId(sensor_id=lidar_id, frame_idx=frame_idx)] = (
                    current_unique_frame_idx
                )
                current_unique_frame_idx += 1

            lidar_frame_timestamps_us[lidar_id] = torch.tensor(
                frame_timestamps_us_list, dtype=torch.int64, device="cpu"
            )

        rig_trajectores: list[RigTrajectories.RigTrajectory] = []
        for loader_key, (T_rig_worlds, T_rig_world_timestamps_us) in T_rig_worlds_with_timestamps_us.items():
            if loader_key not in camera_frame_timestamps_us.keys():
                continue

            # In the new batch design the sensor poses can only obtained by interpolating rig poses.
            # In cases where rig timestamps do not fully cover the sensor timestamps, we extend the rig using constant padding.
            # This can happen, e.g., in Gen3C setting where rig timestamps are end-of-frame ones, so start-of-frame of the 1st frame
            # is not covered.
            sensor_min_timestamp_us = (
                int(min((v.min().item() for v in camera_frame_timestamps_us[loader_key].values()))) - 1
            )
            sensor_max_timestamp_us = (
                int(max((v.max().item() for v in camera_frame_timestamps_us[loader_key].values()))) + 1
            )
            if sensor_min_timestamp_us < int(T_rig_world_timestamps_us[0].item()):
                T_rig_worlds = np.concatenate([T_rig_worlds[:1], T_rig_worlds], axis=0)
                T_rig_world_timestamps_us = np.concatenate(
                    [[sensor_min_timestamp_us], T_rig_world_timestamps_us], axis=0
                )
            if sensor_max_timestamp_us > int(T_rig_world_timestamps_us[-1].item()):
                T_rig_worlds = np.concatenate([T_rig_worlds, T_rig_worlds[-1:]], axis=0)
                T_rig_world_timestamps_us = np.concatenate(
                    [T_rig_world_timestamps_us, [sensor_max_timestamp_us]], axis=0
                )

            # Convert to proper torch tensors
            rig_trajectores.append(
                RigTrajectories.RigTrajectory(
                    sequence_id=sequence_id_prefix + loader_key,
                    rig_bbox=None,
                    cameras_frame_timestamps_us=camera_frame_timestamps_us[loader_key],
                    lidars_frame_timestamps_us=lidar_frame_timestamps_us if loader_key == lidar_loader_key else {},
                    T_rig_worlds=to_torch(T_world_ref @ T_rig_worlds, device="cpu", dtype=torch.float64),
                    T_rig_world_timestamps_us=to_torch(T_rig_world_timestamps_us, device="cpu", dtype=torch.int64),
                )
            )

        camera_calibrations = OrderedDict(
            [
                (
                    str(camera_id),
                    RigTrajectories.CameraCalibration(
                        sequence_id=sequence_id_prefix + camera_id.loader_key,
                        logical_sensor_name=str(camera_id),
                        unique_sensor_idx=camera_id.unique_sensor_idx,
                        T_sensor_rig=to_torch(unpack_optional(camera_sensors[camera_id].T_sensor_rig), device="cpu"),
                        camera_model_parameters=all_camera_model_parameters[camera_id],
                    ),
                )
                for camera_id in frame_batch_camera_ids
            ]
        )

        lidar_calibrations = OrderedDict(
            [
                (
                    str(lidar_id),
                    RigTrajectories.LidarCalibration(
                        sequence_id=sequence_id_prefix + lidar_loader_key,
                        logical_sensor_name=str(lidar_id),
                        unique_sensor_idx=self.lidar_ids.index(lidar_id),
                        T_sensor_rig=to_torch(unpack_optional(lidar_sensors[lidar_id].T_sensor_rig), device="cpu"),
                        lidar_model_parameters=all_lidar_model_parameters[lidar_id],
                    ),
                )
                for lidar_id in frame_batch_lidar_ids
            ]
        )

        return (
            RigTrajectories(
                # Since world coordinates are already transformed to NRM space,
                # to record the ncore world coordinates, we leverage T_world_base here.
                # This would not affect rays or transforms, just for book-keeping for primitive merging.
                T_world_base=se3_matrix_inverse(to_torch(T_world_ref, device="cpu", dtype=torch.float64)),
                world_to_nre=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
                rig_trajectories=rig_trajectores,
                camera_calibrations=camera_calibrations,
                lidar_calibrations=lidar_calibrations,
            ),
            camera_idx_mapping,
            lidar_idx_mapping,
        )

    def _get_loaders_and_sensors(
        self,
        ncore_json_path: UPath,
        all_camera_ids: "list[NCoreNRMDataset.ExtendedCameraId]",
    ) -> "NCoreNRMDataset.LoadersAndSensorsResult":
        """
        Load sequence loaders, aux loaders, camera/lidar sensors, and rig poses for the given
        ncore sequence meta path. Returns a LoadersAndSensorsResult dataclass.
        When cache_loaders_and_sensors is True and not in a DataLoader worker, result is cached (one entry keyed by ncore_json_path).
        """
        use_cache = False
        if self.cache_loaders_and_sensors:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                logger.warning(
                    "cache_loaders_and_sensors is enabled but running inside a DataLoader worker (worker_id=%s); "
                    "caching is disabled to avoid per-worker caches. Use num_workers=0 to enable caching.",
                    worker_info.id,
                )
            else:
                use_cache = True

        if use_cache:
            cache_key = str(ncore_json_path)
            cached = self._loaders_sensors_cache.get(cache_key)
            if cached is not None:
                return cached

        # Clear cache to release memory
        self._loaders_sensors_cache.clear()

        (
            data_format,
            _,  # sequence_id
            _,  # time_range_us
            # contains either V3 zarr.itar shards, or V4 zarr.itar archives / zarr directories
            dataset_paths,
        ) = ncore_utils.parse_sequence_meta_file(ncore_json_path)

        # Here poses and loaders are indexed by the sensor_id.loader_key (ncore file name), while sensors are indexed by sensor_id.
        # NB [JH]: We should be very careful about the poses' timestamps_us -- it can be a large superset of sensor timestamps (e.g. 36s vs 10s)
        # Poses dict contains T_rig_worlds and T_rig_worlds_timestamps_us
        T_rig_worlds_with_timestamps_us: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        sequence_loaders: dict[str, ncore.data.SequenceLoaderProtocol] = {}
        aux_loaders: dict[str, ncore_utils.AuxShardDataLoader] = {}
        camera_sensors: dict[NCoreNRMDataset.ExtendedCameraId, ncore.data.CameraSensorProtocol] = {}
        lidar_sensors: dict[str, ncore.data.LidarSensorProtocol] = {}

        # ShardDataLoader is logging using root. Let's suppress this information.
        (root_logger := logging.getLogger()).setLevel(logging.WARNING)
        for camera_id in all_camera_ids:
            # Determine the ncore file to load
            if camera_id.external_ncore_path is not None:
                current_dataset_paths = [ncore_json_path.parent / camera_id.external_ncore_path]
            else:
                current_dataset_paths = dataset_paths

            # Load the ncore files
            if (sequence_loader := sequence_loaders.get(camera_id.loader_key)) is None:
                try:
                    sequence_loader = sequence_loaders[camera_id.loader_key] = ncore_utils.create_sequence_loader(
                        data_format=data_format,
                        dataset_paths=current_dataset_paths,
                        open_consolidated=self.open_consolidated,
                        v3_cuboid_loading_max_workers=self.cuboid_loading_max_workers,
                        v4_poses_component_group=self.poses_component_group,
                        v4_intrinsics_component_group=self.intrinsics_component_group,
                        v4_masks_component_group=self.masks_component_group,
                        v4_cuboids_component_group=self.cuboids_component_group,
                    )
                except FileNotFoundError as e:
                    raise NRMDataError(
                        f"Ncore files not found for the given dataset_paths {current_dataset_paths}."
                    ) from e

                # TODO: frame-pose only data might fail here as there are no rig poses and might require refined logic
                rig_world_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
                    sequence_loader.pose_graph.get_edge("rig", "world"),
                    msg="Rig-to-world poses required for rig-trajectories",
                )

                # all rig poses with timestamps
                T_rig_worlds_with_timestamps_us[camera_id.loader_key] = (
                    rig_world_edge.T_source_target,
                    unpack_optional(rig_world_edge.timestamps_us, msg="Rig-to-world pose requires to be dynamic"),
                )

                try:
                    if self.aux_data_params.enabled:
                        signal_override_paths: dict[str, UPath] = {}
                        if isinstance(self.aux_data_params.depth, str):
                            depth_override_path = parse_universal_path(
                                self.aux_data_params.depth.replace("{{clip_id}}", sequence_loader.sequence_id),
                                s3_block_size_mb=self.non_concrete_config.s3_block_size_mb,
                                s3_cache_type=self.non_concrete_config.s3_cache_type,
                            )
                            if depth_override_path.exists():
                                signal_override_paths["depth"] = depth_override_path
                        aux_loaders[camera_id.loader_key] = ncore_utils.AuxShardDataLoader(
                            sequence_id=sequence_loader.sequence_id,
                            dataset_paths=current_dataset_paths,
                            open_consolidated=self.open_consolidated,
                            signal_override_paths=signal_override_paths,
                        )
                except ValueError as e:
                    raise NRMDataError(f"Failed to load auxiliary data for sequence {ncore_json_path.stem}.") from e

            # Load the camera sensors
            camera_sensors[camera_id] = sequence_loader.get_camera_sensor(
                self.camera_id_mapping.get(camera_id.camera_id, camera_id.camera_id)
            )

            # Load all the lidar sensors for the first time meeting a main loader.
            # Allow multiple possibilities of lidar ids separated by semicolon.
            if len(lidar_sensors) == 0 and camera_id.external_ncore_path is None:
                lidar_sensors = {
                    lidar_id: get_lidar_sensor_from_sequence_loader(
                        sequence_loader,
                        lidar_id_candidates=[self.lidar_id_mapping.get(p, p) for p in lidar_id.split(";")],
                    )
                    for lidar_id in self.lidar_ids
                }

        root_logger.setLevel(logging.INFO)
        result = NCoreNRMDataset.LoadersAndSensorsResult(
            T_rig_worlds_with_timestamps_us=T_rig_worlds_with_timestamps_us,
            sequence_loaders=sequence_loaders,
            aux_loaders=aux_loaders,
            camera_sensors=camera_sensors,
            lidar_sensors=lidar_sensors,
        )
        if use_cache:
            # Cache only 1 entry by overwriting the existing entry.
            self._loaders_sensors_cache = {str(ncore_json_path): result}
        return result

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> NRMDataBatch:
        # Disable fsspect INFO logs to not spam the logs.
        logging.getLogger("fsspec").setLevel(logging.WARNING)

        sequence_idx: int = batch_idx // self.num_samples_per_sequence
        sample_idx: int = batch_idx % self.num_samples_per_sequence

        concrete_config = self.non_concrete_config.concretize(self.epoch, rng)

        frame_batch_sampler = AdaptiveSequentialFrameBatchSampler(concrete_config.frame_batch_sampler)
        assert sample_idx < frame_batch_sampler.n_samples_per_sequence, "Sample index out of bounds"

        context_id_lookup = {str(c): c for c in self.all_context_camera_ids}
        supervision_id_lookup = {str(c): c for c in self.all_supervision_camera_ids}

        context_camera_ids: list[NCoreNRMDataset.ExtendedCameraId] = [
            context_id_lookup[str(NCoreNRMDataset.ExtendedCameraId.from_config(camera_id))]
            for camera_id in concrete_config.context_camera_ids
        ]
        supervision_camera_ids: list[NCoreNRMDataset.ExtendedCameraId] = [
            supervision_id_lookup[str(NCoreNRMDataset.ExtendedCameraId.from_config(camera_id))]
            for camera_id in concrete_config.supervision_camera_ids
        ]
        assert set(map(str, context_camera_ids)) <= set(map(str, supervision_camera_ids)), (
            f"context_camera_ids must be a subset of supervision_camera_ids; "
            f"context={sorted(map(str, context_camera_ids))} "
            f"supervision={sorted(map(str, supervision_camera_ids))}"
        )

        ncore_json_path: UPath = self.ncore_json_paths[sequence_idx]
        if not ncore_json_path.exists():
            raise NRMDataError(f"{ncore_json_path} does not exist.")

        loaders_sensors = self._get_loaders_and_sensors(ncore_json_path, supervision_camera_ids)
        T_rig_worlds_with_timestamps_us = loaders_sensors.T_rig_worlds_with_timestamps_us
        sequence_loaders = loaders_sensors.sequence_loaders
        aux_loaders = loaders_sensors.aux_loaders
        camera_sensors = loaders_sensors.camera_sensors
        lidar_sensors = loaders_sensors.lidar_sensors

        # Determine the timestamps interval to select frames from.
        main_loader_key: str = NCoreNRMDataset.ExtendedCameraId.main_loader_key()
        main_sequence_loader = sequence_loaders[main_loader_key]
        context_camera_frame_timestamps_us: dict[str, np.ndarray] = {}

        select_intervals = subranges_to_intervals(
            T_rig_worlds_with_timestamps_us[main_loader_key][1],  # T_rig_world_timestamps_us
            self.sequence_subranges[ncore_json_path.stem],
        )
        for loader_key, (_, T_rig_world_timestamps_us) in T_rig_worlds_with_timestamps_us.items():
            if loader_key != main_loader_key:
                select_intervals = interval_list_intersect(
                    select_intervals,
                    HalfClosedInterval(T_rig_world_timestamps_us.min(), T_rig_world_timestamps_us.max()),
                )
        # Intersect also with sensor timestamps (with +/- 0.1s tolerance)
        for camera_id in context_camera_ids:
            timestamps_us = camera_sensors[camera_id].get_frames_timestamps_us(ncore.data.FrameTimepoint.END)
            select_intervals = interval_list_intersect(
                select_intervals,
                HalfClosedInterval(int(timestamps_us.min()) - 100000, int(timestamps_us.max()) + 100000),
            )
            context_camera_frame_timestamps_us[str(camera_id)] = timestamps_us

        lidar_frame_timestamps_us: dict[str, np.ndarray] = {
            lidar_id: lidar_sensors[lidar_id].get_frames_timestamps_us(ncore.data.FrameTimepoint.END)
            for lidar_id in self.lidar_ids
        }

        context_frame_batch = frame_batch_sampler.sample_frame_batch(
            sample_idx,
            context_camera_frame_timestamps_us,
            select_intervals,
        )
        if len(context_frame_batch.sampled_sensor_frame_idxs) == 0:
            # If nothing is sampled (e.g. out of bounds), return 0-sized batch to be concatenated with other batches.
            return NRMDataBatch(context=[], cuboid_tracks=[], context_rig=[], meta=[])

        if len(self.lidar_ids) > 0:
            context_frame_batch = sample_lidar_frame_batch(
                config=self.lidar_frame_batch_params,
                frame_batch=context_frame_batch,
                sensor_frame_timestamps_us=context_camera_frame_timestamps_us | lidar_frame_timestamps_us,
                lidar_id=self.lidar_ids[0],
            )

        # Determine a good reference coordinates (first camera first frame - non-rig)
        ref_camera_id = context_camera_ids[0]
        T_world_ref = camera_sensors[ref_camera_id].get_frames_T_source_sensor(
            source_node="world",
            frame_indices=min(context_frame_batch.sampled_sensor_frame_idxs[str(ref_camera_id)]),
            frame_timepoint=ncore.data.FrameTimepoint.END,
        )

        # Load context frames.
        context_camera_subsampler = CameraSubsampler(concrete_config.camera_subsampler)
        context_rig_trajectory, context_camera_mapping, context_lidar_mapping = self._get_rig_trajectory(
            "context-",
            context_frame_batch,
            camera_sensors,
            lidar_sensors,
            T_world_ref,
            T_rig_worlds_with_timestamps_us,
            context_camera_subsampler,
        )
        context = DataAndRenderingBatch(
            data=self._load_data_batch(
                context_frame_batch,
                context_camera_mapping,
                camera_sensors,
                context_lidar_mapping,
                lidar_sensors,
                aux_loaders if self.aux_data_params.enabled_context else {},
                context_camera_subsampler,
            )
        )

        # Predict pins supervision_frame_batch.n_frames_per_sample=0; supervision is disabled.
        supervision = None
        supervision_rig_trajectory = None

        cuboid_tracks = self._compute_cuboid_tracks(
            context_frame_batch,
            main_sequence_loader,
            camera_sensors,
            lidar_sensors,
            T_world_ref,
        )

        # Compute meta data
        meta = {
            "ncore_json_path": ncore_json_path,
            "sequence_id": main_sequence_loader.sequence_id,
        }
        if self.cache_loaders_and_sensors:
            meta["sequence_loader"] = main_sequence_loader

        nrm_data_batch = NRMDataBatch(
            context=[context],
            context_rig=[context_rig_trajectory],
            supervision=[supervision] if supervision is not None else None,
            supervision_rig=[supervision_rig_trajectory] if supervision_rig_trajectory is not None else None,
            cuboid_tracks=[cuboid_tracks],
            meta=[meta],
        )

        if self.compute_rendering_data:
            nrm_data_batch.maybe_compute_rendering_data(device=torch.device("cpu"))

        return nrm_data_batch
