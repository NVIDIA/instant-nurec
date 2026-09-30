# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import hashlib
import logging
import math
import os

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from scipy import ndimage
from upath import UPath

import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import instant_nurec.utils.ncore_utils as ncore_utils

from instant_nurec.datasets.tracks import CuboidTracks, CuboidTracksDataPack, TrackFlags
from instant_nurec.datasets.utils import compute_cuboid_df, consolidate_cuboid_tracks
from instant_nurec.config_schema.dataset import ExternalSupervisionCameraIdConfig, NCoreInstantNuRecDatasetConfig
from instant_nurec.datasets.instantnurec_base import (
    BaseInstantNuRecIndexableDataset,
    CameraSubsampler,
    InstantNuRecDataError,
)
from instant_nurec.datasets.samplers import (
    AdaptiveSequentialFrameBatchSampler,
    SampledSensorFrameIdxs,
    UniformFrameBatchSampler,
    get_closest_frame_index,
)
from instant_nurec.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    InstantNuRecDataBatch,
)
from instant_nurec.utils.files import parse_universal_path
from instant_nurec.utils.geometry import se3_matrix_inverse
from instant_nurec.utils.misc import to_torch, unpack_optional
from instant_nurec.utils.types import FrameConversion, HalfClosedInterval, RayFlags, RigTrajectories


logger = logging.getLogger(__name__)


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


class NCoreInstantNuRecDataset(BaseInstantNuRecIndexableDataset[InstantNuRecDataBatch]):
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
            # Waymo v18.7 emits lowercase `cyclist`. The reference set has
            # only uppercase `CYCLIST`; accepting both fixes that upstream
            # integration bug while preserving existing labels.
            "cyclist",
        ]
    )

    @dataclass(kw_only=True, frozen=True)
    class UniqueFrameId:
        sensor_id: str
        frame_idx: int

    @dataclass(frozen=True)
    class ExtendedCameraId:
        """Camera id with optional sequence-relative external NCore source."""

        camera_id: str
        unique_sensor_idx: int
        external_ncore_path: str | None = None
        sample_ratio: float = 1.0

        @staticmethod
        def from_config(
            camera_id: str | ExternalSupervisionCameraIdConfig,
            unique_sensor_idx: int = -1,
        ) -> "NCoreInstantNuRecDataset.ExtendedCameraId":
            if isinstance(camera_id, str):
                return NCoreInstantNuRecDataset.ExtendedCameraId(
                    camera_id=camera_id,
                    unique_sensor_idx=unique_sensor_idx,
                )
            return NCoreInstantNuRecDataset.ExtendedCameraId(
                camera_id=camera_id.camera_id,
                unique_sensor_idx=camera_id.unique_sensor_idx,
                external_ncore_path=camera_id.ncore_path,
                sample_ratio=camera_id.sample_ratio,
            )

        def __str__(self) -> str:
            if self.external_ncore_path is None:
                return self.camera_id
            return f"{self.camera_id}-({self.external_ncore_path.replace('/', '_')})"

        @property
        def loader_key(self) -> str:
            return self.main_loader_key() if self.external_ncore_path is None else self.external_ncore_path

        @staticmethod
        def main_loader_key() -> str:
            return "main"

        @property
        def canonical_order(self) -> str:
            """Canonical order to be in rig trajectory and the data batch."""
            order = f"{self.unique_sensor_idx:03d}"
            if self.external_ncore_path is not None:
                order += f"-{self.external_ncore_path}"
            return order

    @dataclass
    class LoadersAndSensorsResult:
        """Result of loading sequence, optional labels, and camera sensors."""

        T_rig_worlds_with_timestamps_us: dict[str, tuple[np.ndarray, np.ndarray]]
        sequence_loaders: dict[str, ncore.data.SequenceLoaderProtocol]
        aux_loaders: dict[str, ncore_utils.AuxShardDataLoader]
        camera_sensors: dict["NCoreInstantNuRecDataset.ExtendedCameraId", ncore.data.CameraSensorProtocol]

    def __init__(
        self,
        config: NCoreInstantNuRecDatasetConfig,
        frame_width: int,
        frame_height: int,
        n_frames_per_sample: int,
        global_seed: int | None = None,
        retry_on_error: bool = False,
    ):
        # ``frame_width`` / ``frame_height`` / ``n_frames_per_sample`` are
        # passed in by the caller (typically ``instant_nurec.model.make``),
        # not via the dataset config.
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._n_frames_per_sample = n_frames_per_sample
        self._global_seed = global_seed
        self._retry_on_error = retry_on_error

        self.open_consolidated = config.open_consolidated
        self.camera_max_fov_deg = config.camera_max_fov_deg
        self.n_camera_mask_dilation_iterations = config.n_camera_mask_dilation_iterations

        self.all_supervision_camera_ids: list[NCoreInstantNuRecDataset.ExtendedCameraId] = []
        for camera_idx, camera_id_config in enumerate(config.supervision_camera_ids):
            # For string-based camera ids, we directly use their sequence as in the config.
            # For external supervision cameras, the unique sensor index is specified directly in the config.
            self.all_supervision_camera_ids.append(
                NCoreInstantNuRecDataset.ExtendedCameraId.from_config(camera_id_config, camera_idx)
            )
        self.all_context_camera_ids: list[NCoreInstantNuRecDataset.ExtendedCameraId] = []
        for camera_id_config in config.context_camera_ids:
            try:
                camera_id_idx = [str(c) for c in self.all_supervision_camera_ids].index(str(camera_id_config))
            except ValueError as e:
                raise ValueError(
                    f"Context camera {camera_id_config} not found in supervision cameras {self.all_supervision_camera_ids}"
                ) from e
            self.all_context_camera_ids.append(self.all_supervision_camera_ids[camera_id_idx])

        self.cuboid_tracks_params = config.cuboid_tracks_params

        self.ncore_json_paths = self._resolve_ncore_json_paths(config)
        logger.info("Loaded %d sequence(s).", len(self.ncore_json_paths))

        self.num_samples_per_sequence: int = config.frame_batch_sampler.n_samples_per_sequence
        self.config = config
        # Match the reference dataset contract: train sampling is deterministic for
        # (epoch, item index, global seed), while validation keeps rng_epoch=-1
        # so its samples do not drift from epoch to epoch.
        self._rng_epoch = -1
        self._epoch = -1

    @staticmethod
    def _resolve_ncore_json_paths(config: NCoreInstantNuRecDatasetConfig) -> list[UPath]:
        """Resolve explicit paths or an official-style newline manifest."""

        if config.ncore_json_paths:
            if config.ncore_json_list_path is not None:
                logger.warning(
                    "Both ncore_json_paths and ncore_json_list_path are set; "
                    "using the explicit ncore_json_paths override."
                )
            return [parse_universal_path(path) for path in config.ncore_json_paths]

        manifest = parse_universal_path(unpack_optional(config.ncore_json_list_path))
        base = parse_universal_path(config.ncore_json_base_path) if config.ncore_json_base_path else None
        paths: list[UPath] = []
        with manifest.open("r") as stream:
            for raw_line in stream:
                entry = raw_line.strip()
                if not entry or entry.startswith("#"):
                    continue
                if base is not None and "://" not in entry and not entry.startswith("/"):
                    paths.append(base / entry)
                else:
                    paths.append(parse_universal_path(entry))
        if not paths:
            raise ValueError(f"NCore manifest contains no sequence paths: {manifest}")
        return paths

    def _build_frame_batch_sampler(self) -> AdaptiveSequentialFrameBatchSampler | UniformFrameBatchSampler:
        if self.config.frame_batch_sampler.name == "uniform":
            return UniformFrameBatchSampler(
                self.config.frame_batch_sampler,
                n_frames_per_sample=self._n_frames_per_sample,
            )
        return AdaptiveSequentialFrameBatchSampler(
            self.config.frame_batch_sampler,
            n_frames_per_sample=self._n_frames_per_sample,
        )

    def set_rng_epoch(self, rng_epoch: int) -> None:
        self._rng_epoch = rng_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def _get_rng(self, batch_idx: int) -> np.random.Generator:
        """Return the official per-item SHA256-derived NumPy generator."""

        global_seed = getattr(self, "_global_seed", None)
        if global_seed is None:
            if "PL_GLOBAL_SEED" not in os.environ:
                raise RuntimeError(
                    "No dataset global seed was supplied and PL_GLOBAL_SEED is unset; "
                    "pass global_seed or call pytorch_lightning.seed_everything() first"
                )
            global_seed = int(os.environ["PL_GLOBAL_SEED"])
        digest = hashlib.sha256(f"{self._rng_epoch}_{batch_idx}_{global_seed}".encode()).digest()
        return np.random.default_rng(seed=int.from_bytes(digest[:8], "big"))

    def _build_camera_subsampler(self):
        from instant_nurec.datasets.instantnurec_base import CameraSubsampler

        return CameraSubsampler(frame_width=self._frame_width, frame_height=self._frame_height)

    def _subsample_camera_model_parameters(
        self,
        camera_sensor: ncore.data.CameraSensorProtocol,
        camera_subsampler: CameraSubsampler,
    ) -> ncore.data.ConcreteCameraModelParametersUnion:
        """Copy and resize/crop a source camera calibration for inference."""

        camera_model_parameters = dataclasses.replace(camera_sensor.model_parameters)

        # Some camera models have bad linear_cde values. Fix only the copy so
        # the source NCore calibration remains untouched.
        if isinstance(camera_model_parameters, ncore.data.FThetaCameraModelParameters) and np.all(
            camera_model_parameters.linear_cde == 0.0
        ):
            camera_model_parameters.linear_cde = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        if isinstance(camera_model_parameters, ncore.data.FThetaCameraModelParameters):
            camera_model_parameters.max_angle = min(
                np.deg2rad(self.camera_max_fov_deg) / 2.0,
                camera_model_parameters.max_angle,
            )

        return camera_subsampler.apply_camera_parameters(camera_model_parameters)

    def __len__(self) -> int:
        return len(self.ncore_json_paths) * self.num_samples_per_sequence

    def _compute_cuboid_tracks(
        self,
        context_frame_batch: SampledSensorFrameIdxs,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        camera_sensors: dict[ExtendedCameraId, ncore.data.CameraSensorProtocol],
        T_world_ref: np.ndarray,
    ) -> CuboidTracksDataPack:
        frame_batch_min_timestamps_us: int = int(1e16)
        frame_batch_max_timestamps_us: int = 0
        for sensor_id, frame_idxs in context_frame_batch.items():
            sensor = [v for k, v in camera_sensors.items() if str(k) == sensor_id][0]
            sensor_min_timestamp_us = sensor.get_frame_timestamp_us(min(frame_idxs), ncore.data.FrameTimepoint.START)
            sensor_max_timestamp_us = sensor.get_frame_timestamp_us(max(frame_idxs), ncore.data.FrameTimepoint.END)
            frame_batch_min_timestamps_us = min(frame_batch_min_timestamps_us, sensor_min_timestamp_us)
            frame_batch_max_timestamps_us = max(frame_batch_max_timestamps_us, sensor_max_timestamp_us)

        time_range_us = HalfClosedInterval(
            frame_batch_min_timestamps_us - self.cuboid_tracks_params.track_extrapolate_timestamps_us,
            frame_batch_max_timestamps_us + self.cuboid_tracks_params.track_extrapolate_timestamps_us,
        )
        cuboids_df = compute_cuboid_df(sequence_loader, time_range_us)

        # First associate all tracks within the batch
        all_batch_tracks = consolidate_cuboid_tracks(
            cuboids_df=cuboids_df,
            sequence_loader=sequence_loader,
            track_label_sources=[self.cuboid_tracks_params.track_label_source],
            track_min_centroid_rig_dist_m=self.cuboid_tracks_params.track_min_centroid_rig_dist_m,
            T_world_world_base=T_world_ref,
        )

        all_track_ids = []
        all_tracks_poses = []
        all_tracks_timestamps_us = []
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
            all_tracks_flags.append(track_flags)
            all_cuboid_dims.append(track["dimension"])

        # Map to member structs
        cuboid_tracks = CuboidTracks.Factory.from_numpy(
            all_track_ids,
            all_tracks_poses,
            all_tracks_timestamps_us,
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
        frame_batch: SampledSensorFrameIdxs,
        camera_idx_mapping: dict[UniqueFrameId, int],
        camera_sensors: dict[ExtendedCameraId, ncore.data.CameraSensorProtocol],
        camera_subsampler: CameraSubsampler,
        aux_loaders: dict[str, ncore_utils.AuxShardDataLoader],
    ) -> DataBatch:
        """
        Load actual data batch given the sampled frame batch. idx_mapping is used to determine the unique frame index for the frame meta.
        """
        ## Load cameras

        # This determines the ordering of images in the actual batch.
        # As long as network is equivariant to the order of images, this is not important.
        frame_batch_camera_ids = [
            matched_camera_ids[0]
            for camera_id_name in frame_batch.keys()
            if len(matched_camera_ids := [c for c in camera_sensors.keys() if str(c) == camera_id_name]) > 0
        ]
        frame_batch_camera_ids = sorted(frame_batch_camera_ids, key=lambda x: x.canonical_order)

        # Read Camera-based data
        camera_batch_list: list[DataBatch.Camera] = []
        for camera_id in frame_batch_camera_ids:
            frame_idxs = frame_batch[str(camera_id)]
            if camera_id not in camera_sensors:
                continue
            camera_sensor = camera_sensors[camera_id]
            aux_loader = aux_loaders.get(camera_id.loader_key)
            frame_height = camera_subsampler.frame_height
            frame_width = camera_subsampler.frame_width

            static_mask = ncore_utils.get_camera_sensor_mask(camera_sensor)
            if static_mask is None:
                static_invalid_mask = np.zeros((frame_height, frame_width), dtype=bool)
            else:
                static_mask = camera_subsampler.apply_frame_data(static_mask)
                static_invalid_mask = cast(
                    np.ndarray,
                    ndimage.binary_dilation(
                        static_mask,
                        iterations=self.n_camera_mask_dilation_iterations,
                    ),
                )

            # Determine unique sensor index mapping
            unique_sensor_idx = camera_id.unique_sensor_idx
            for frame_idx in frame_idxs:
                frame_end_timestamp_us = int(
                    camera_sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.END)
                )
                # Collect labels data
                labels = CameraFrameLabels()
                frame_image_array = camera_sensor.get_frame_image_array(frame_idx).astype(np.float32) / 255.0
                frame_image_array = camera_subsampler.apply_frame_data(frame_image_array)
                labels.rgb = to_torch(frame_image_array, device="cpu").unsqueeze(0)
                flags = torch.full(
                    (frame_height, frame_width),
                    int(RayFlags.RGB_LABEL),
                    dtype=torch.int32,
                )
                if camera_id.external_ncore_path is not None:
                    flags |= int(RayFlags.SYNTHETIC)
                invalid_ego_mask = static_invalid_mask.copy()

                if aux_loader is not None:
                    data_camera_id = camera_id.camera_id
                    sky_mask: np.ndarray | bool = False
                    if self.config.aux_data.semantic_segmentation and aux_loader.has_semantic_segmentation(
                        data_camera_id
                    ):
                        semantics = np.asarray(
                            aux_loader.get_semantic_segmentation(data_camera_id, frame_end_timestamp_us)
                        )
                        classes = aux_loader.get_semantic_segmentation_meta(data_camera_id)["stuff_classes"]
                        sky_mask = semantics == classes.index("sky") if "sky" in classes else False
                        semantics = camera_subsampler.apply_frame_data(semantics)
                        flags |= int(RayFlags.VALID_SEMANTIC)
                        if "sky" in classes:
                            flags[semantics == classes.index("sky")] |= int(RayFlags.SKY_SEMANTIC)
                        if "road" in classes:
                            flags[semantics == classes.index("road")] |= int(RayFlags.ROAD_SEMANTIC)
                        for vehicle_class in ("car", "truck", "bus", "train", "motorcycle", "bicycle"):
                            if vehicle_class in classes:
                                flags[semantics == classes.index(vehicle_class)] |= int(RayFlags.VEHICLE_SEMANTIC)
                        if "egocar" in classes:
                            invalid_ego_mask |= semantics == classes.index("egocar")

                    if self.config.aux_data.depth and aux_loader.has_depth(data_camera_id):
                        metric_distance = aux_loader.get_depth(data_camera_id, frame_end_timestamp_us)
                        metric_distance[~np.isfinite(metric_distance) | sky_mask] = 0.0
                        metric_distance = camera_subsampler.apply_depth_data(metric_distance)
                        labels.metric_distance = to_torch(metric_distance, device="cpu")[None, ..., None]

                    if self.config.aux_data.egomask and aux_loader.has_egomask(data_camera_id):
                        ego_mask = aux_loader.get_egomask(data_camera_id)
                        ego_mask = camera_subsampler.apply_frame_data(ego_mask)
                        invalid_ego_mask |= cast(
                            np.ndarray,
                            ndimage.binary_dilation(
                                ego_mask,
                                iterations=self.n_camera_mask_dilation_iterations,
                            ),
                        )

                invalid_ego_mask_t = torch.from_numpy(invalid_ego_mask)
                flags[invalid_ego_mask_t] |= int(RayFlags.INVALID | RayFlags.EGO_SEMANTIC)
                labels.flags = flags[None, ..., None]

                camera_batch_list.append(
                    DataBatch.Camera(
                        meta=[
                            FrameMeta(
                                unique_sensor_idx=unique_sensor_idx,
                                unique_frame_idx=camera_idx_mapping[
                                    self.UniqueFrameId(sensor_id=str(camera_id), frame_idx=frame_idx)
                                ],
                            )
                        ],
                        labels=labels,
                    )
                )

        return DataBatch(camera=DataBatch.Camera.collate_fn(camera_batch_list))

    def _get_rig_trajectory(
        self,
        sequence_id_prefix: str,
        frame_batch: SampledSensorFrameIdxs,
        camera_sensors: dict[ExtendedCameraId, ncore.data.CameraSensorProtocol],
        T_world_ref: np.ndarray,
        T_rig_worlds_with_timestamps_us: dict[str, tuple[np.ndarray, np.ndarray]],
        camera_subsampler: CameraSubsampler,
    ) -> tuple[RigTrajectories, dict[UniqueFrameId, int]]:
        """
        Obtain rig-trajectory based on the sampled sensors.
        The rig trajectory will contain the full rig poses and frame_batch-sampled cameras.

        This will additionally return a UniqueFrameId to index mapping, which matches the logic of CameraFreePoseViewGeometry
        so we can properly query a frame via its unique frame idx.
        """
        ## Load cameras

        # loader_key -> camera_id_name -> timestamps_us
        frame_timestamps_us_list: list[tuple[int, int]] = []
        camera_frame_timestamps_us: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        all_camera_model_parameters: dict[
            NCoreInstantNuRecDataset.ExtendedCameraId, ncore.data.ConcreteCameraModelParametersUnion
        ] = {}
        camera_idx_mapping: dict[NCoreInstantNuRecDataset.UniqueFrameId, int] = {}

        # Find the matching ExtendedCameraId given the string name from the sampler.
        frame_batch_camera_ids = [
            matched_camera_ids[0]
            for camera_id_name in frame_batch.keys()
            if len(matched_camera_ids := [c for c in camera_sensors.keys() if str(c) == camera_id_name]) > 0
        ]
        # This determines the OrderedDict ordering of cameras in the rig trajectory.
        frame_batch_camera_ids = sorted(frame_batch_camera_ids, key=lambda x: x.canonical_order)

        current_unique_frame_idx: int = 0
        for camera_id in frame_batch_camera_ids:
            camera_sensor = camera_sensors[camera_id]
            camera_model_parameters = self._subsample_camera_model_parameters(camera_sensor, camera_subsampler)
            all_camera_model_parameters[camera_id] = camera_model_parameters
            frame_timestamps_us_list = []
            for frame_idx in frame_batch[str(camera_id)]:
                frame_start_timestamp_us = int(
                    camera_sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.START)
                )
                frame_end_timestamp_us = int(
                    camera_sensor.get_frame_timestamp_us(frame_idx, ncore.data.FrameTimepoint.END)
                )
                frame_timestamps_us_list.append((frame_start_timestamp_us, frame_end_timestamp_us))

                # NB [JH]: We must ensure that the sequence of iteration matches the logic of CameraFreePoseViewGeometry
                camera_idx_mapping[NCoreInstantNuRecDataset.UniqueFrameId(sensor_id=str(camera_id), frame_idx=frame_idx)] = (
                    current_unique_frame_idx
                )
                current_unique_frame_idx += 1

            camera_frame_timestamps_us[camera_id.loader_key][str(camera_id)] = torch.tensor(
                frame_timestamps_us_list, dtype=torch.int64, device="cpu"
            )

        rig_trajectores: list[RigTrajectories.RigTrajectory] = []
        for loader_key, (T_rig_worlds, T_rig_world_timestamps_us) in T_rig_worlds_with_timestamps_us.items():
            loader_camera_timestamps = camera_frame_timestamps_us.get(loader_key)
            if not loader_camera_timestamps:
                continue

            # Interpolation must cover each exposure boundary. Constant endpoint
            # padding is the official behavior for both main and external rigs.
            sensor_min_timestamp_us = int(min(v.min().item() for v in loader_camera_timestamps.values())) - 1
            sensor_max_timestamp_us = int(max(v.max().item() for v in loader_camera_timestamps.values())) + 1
            if sensor_min_timestamp_us < int(T_rig_world_timestamps_us[0]):
                T_rig_worlds = np.concatenate([T_rig_worlds[:1], T_rig_worlds], axis=0)
                T_rig_world_timestamps_us = np.concatenate(
                    [[sensor_min_timestamp_us], T_rig_world_timestamps_us], axis=0
                )
            if sensor_max_timestamp_us > int(T_rig_world_timestamps_us[-1]):
                T_rig_worlds = np.concatenate([T_rig_worlds, T_rig_worlds[-1:]], axis=0)
                T_rig_world_timestamps_us = np.concatenate(
                    [T_rig_world_timestamps_us, [sensor_max_timestamp_us]], axis=0
                )

            rig_trajectores.append(
                RigTrajectories.RigTrajectory(
                    sequence_id=sequence_id_prefix + loader_key,
                    cameras_frame_timestamps_us=loader_camera_timestamps,
                    T_rig_worlds=to_torch(T_world_ref @ T_rig_worlds, device="cpu", dtype=torch.float64),
                    T_rig_world_timestamps_us=to_torch(
                        T_rig_world_timestamps_us,
                        device="cpu",
                        dtype=torch.int64,
                    ),
                )
            )

        camera_calibrations = OrderedDict(
            [
                (
                    str(camera_id),
                    RigTrajectories.CameraCalibration(
                        sequence_id=sequence_id_prefix + camera_id.loader_key,
                        unique_sensor_idx=camera_id.unique_sensor_idx,
                        T_sensor_rig=to_torch(unpack_optional(camera_sensors[camera_id].T_sensor_rig), device="cpu"),
                        camera_model_parameters=all_camera_model_parameters[camera_id],
                    ),
                )
                for camera_id in frame_batch_camera_ids
            ]
        )

        return (
            RigTrajectories(
                # Since world coordinates are already transformed to scene space,
                # to record the ncore world coordinates, we leverage T_world_base here.
                # This would not affect rays or transforms, just for book-keeping for primitive merging.
                T_world_base=se3_matrix_inverse(to_torch(T_world_ref, device="cpu", dtype=torch.float64)),
                world_to_scene=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
                rig_trajectories=rig_trajectores,
                camera_calibrations=camera_calibrations,
            ),
            camera_idx_mapping,
        )

    def _get_loaders_and_sensors(
        self,
        ncore_json_path: UPath,
        all_camera_ids: "list[NCoreInstantNuRecDataset.ExtendedCameraId]",
    ) -> "NCoreInstantNuRecDataset.LoadersAndSensorsResult":
        """
        Load sequence loaders, camera sensors, and rig poses for the
        given ncore sequence meta path. Returns a LoadersAndSensorsResult dataclass.
        """
        (
            _,  # sequence_id
            _,  # time_range_us
            # V4 zarr.itar archives / zarr directories
            dataset_paths,
        ) = ncore_utils.parse_sequence_meta_file(ncore_json_path)

        # Load each source archive once. External supervision archives carry
        # their own calibrated cameras and rig trajectories but share the main
        # clip's world coordinate frame.
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.WARNING)
        T_rig_worlds_with_timestamps_us: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        sequence_loaders: dict[str, ncore.data.SequenceLoaderProtocol] = {}
        aux_loaders: dict[str, ncore_utils.AuxShardDataLoader] = {}
        camera_sensors: dict[NCoreInstantNuRecDataset.ExtendedCameraId, ncore.data.CameraSensorProtocol] = {}
        try:
            for camera_id in all_camera_ids:
                current_dataset_paths = (
                    dataset_paths
                    if camera_id.external_ncore_path is None
                    else [ncore_json_path.parent / camera_id.external_ncore_path]
                )
                loader_key = camera_id.loader_key
                sequence_loader = sequence_loaders.get(loader_key)
                if sequence_loader is None:
                    try:
                        sequence_loader = ncore_utils.create_sequence_loader(
                            dataset_paths=current_dataset_paths,
                            open_consolidated=self.open_consolidated,
                            v4_poses_component_group="default",
                            v4_intrinsics_component_group="default",
                            v4_masks_component_group="default",
                            v4_cuboids_component_group="default",
                        )
                    except FileNotFoundError as exc:
                        raise InstantNuRecDataError(
                            f"Ncore files not found for dataset_paths {current_dataset_paths}."
                        ) from exc
                    sequence_loaders[loader_key] = sequence_loader
                    rig_world_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
                        sequence_loader.pose_graph.get_edge("rig", "world"),
                        msg="Rig-to-world poses required for rig-trajectories",
                    )
                    T_rig_worlds_with_timestamps_us[loader_key] = (
                        rig_world_edge.T_source_target,
                        unpack_optional(
                            rig_world_edge.timestamps_us,
                            msg="Rig-to-world pose requires to be dynamic",
                        ),
                    )
                    if self.config.aux_data.enabled:
                        try:
                            signal_override_paths: dict[str, UPath] = {}
                            if isinstance(self.config.aux_data.depth, str):
                                depth_override_path = parse_universal_path(
                                    self.config.aux_data.depth.replace(
                                        "{{clip_id}}", sequence_loader.sequence_id
                                    )
                                )
                                if depth_override_path.exists():
                                    signal_override_paths["depth"] = depth_override_path
                            aux_loaders[loader_key] = ncore_utils.AuxShardDataLoader(
                                sequence_id=sequence_loader.sequence_id,
                                dataset_paths=current_dataset_paths,
                                open_consolidated=self.open_consolidated,
                                signal_override_paths=signal_override_paths,
                            )
                        except ValueError as exc:
                            raise InstantNuRecDataError(
                                f"Failed to load auxiliary labels for sequence {ncore_json_path.stem}."
                            ) from exc
                camera_sensors[camera_id] = sequence_loader.get_camera_sensor(camera_id.camera_id)
        finally:
            root_logger.setLevel(previous_level)

        return NCoreInstantNuRecDataset.LoadersAndSensorsResult(
            T_rig_worlds_with_timestamps_us=T_rig_worlds_with_timestamps_us,
            sequence_loaders=sequence_loaders,
            aux_loaders=aux_loaders,
            camera_sensors=camera_sensors,
        )

    def load_full_camera_rig(
        self,
        ncore_json_path: str | UPath,
        reference_rig: RigTrajectories,
        camera_id: str | None = None,
    ) -> RigTrajectories:
        """Load every source exposure for one context camera as a lightweight rig.

        The returned rig reuses the supplied reference rig's aligned world
        trajectory and coordinate conversion, but rebuilds the selected camera
        calibration from the source NCore sequence using the exact resize/crop
        transform used for model inference. Only calibration, poses, and frame
        start/end timestamps are materialized; images and per-pixel rays remain
        lazy so callers can render one frame at a time.
        """

        context_camera_lookup = {str(context_camera): context_camera for context_camera in self.all_context_camera_ids}
        if camera_id is None:
            if len(context_camera_lookup) != 1:
                raise ValueError(
                    "camera_id is required when more than one context camera is configured; "
                    f"available cameras: {sorted(context_camera_lookup)}"
                )
            selected_camera = next(iter(context_camera_lookup.values()))
        else:
            try:
                selected_camera = context_camera_lookup[camera_id]
            except KeyError as exc:
                raise ValueError(
                    f"Camera {camera_id!r} is not a configured context camera; "
                    f"available cameras: {sorted(context_camera_lookup)}"
                ) from exc

        selected_camera_id = str(selected_camera)
        if selected_camera_id not in reference_rig.camera_calibrations:
            raise ValueError(f"Reference rig has no calibration for context camera {selected_camera_id!r}")
        reference_calibration = reference_rig.camera_calibrations[selected_camera_id]
        reference_trajectories = [
            trajectory
            for trajectory in reference_rig.rig_trajectories
            if trajectory.sequence_id == reference_calibration.sequence_id
        ]
        if len(reference_trajectories) != 1:
            raise ValueError(
                "Expected exactly one reference trajectory for camera "
                f"{selected_camera_id!r}, found {len(reference_trajectories)}"
            )
        reference_trajectory = reference_trajectories[0]

        source_path = ncore_json_path if isinstance(ncore_json_path, UPath) else parse_universal_path(ncore_json_path)
        if not source_path.exists():
            raise InstantNuRecDataError(f"{source_path} does not exist.")
        loaders_sensors = self._get_loaders_and_sensors(source_path, self.all_context_camera_ids)
        camera_sensor = loaders_sensors.camera_sensors[selected_camera]

        frame_start_timestamps_us = np.asarray(
            camera_sensor.get_frames_timestamps_us(ncore.data.FrameTimepoint.START),
            dtype=np.int64,
        )
        frame_end_timestamps_us = np.asarray(
            camera_sensor.get_frames_timestamps_us(ncore.data.FrameTimepoint.END),
            dtype=np.int64,
        )
        if frame_start_timestamps_us.ndim != 1 or frame_end_timestamps_us.ndim != 1:
            raise ValueError("Source camera frame timestamps must be one-dimensional")
        if frame_start_timestamps_us.shape != frame_end_timestamps_us.shape:
            raise ValueError(
                "Source camera START/END timestamp counts differ: "
                f"{len(frame_start_timestamps_us)} != {len(frame_end_timestamps_us)}"
            )
        if len(frame_start_timestamps_us) == 0:
            raise ValueError(f"Source camera {selected_camera_id!r} has no frames")
        if np.any(frame_end_timestamps_us < frame_start_timestamps_us):
            raise ValueError("Source camera frame END timestamps must not precede START timestamps")

        # Full-video output must not outrun the scene reconstruction. Mirror the
        # sampler's interval/chunk calculation across every configured context
        # camera and fail explicitly when --max-chunks would truncate the clip.
        rig_timestamps_us = np.asarray(
            loaders_sensors.T_rig_worlds_with_timestamps_us[
                NCoreInstantNuRecDataset.ExtendedCameraId.main_loader_key()
            ][1]
        )
        sequence_start_timestamp_us = int(rig_timestamps_us.min())
        sequence_end_timestamp_us = int(rig_timestamps_us.max())
        for context_camera in self.all_context_camera_ids:
            if context_camera == selected_camera:
                context_end_timestamps_us = frame_end_timestamps_us
            else:
                context_end_timestamps_us = np.asarray(
                    loaders_sensors.camera_sensors[context_camera].get_frames_timestamps_us(
                        ncore.data.FrameTimepoint.END
                    ),
                    dtype=np.int64,
                )
            if context_end_timestamps_us.ndim != 1 or len(context_end_timestamps_us) == 0:
                raise ValueError(f"Context camera {context_camera!s} has no one-dimensional END timestamps")
            sequence_start_timestamp_us = max(
                sequence_start_timestamp_us,
                int(context_end_timestamps_us.min()) - 100_000,
            )
            sequence_end_timestamp_us = min(
                sequence_end_timestamp_us,
                int(context_end_timestamps_us.max()) + 100_000,
            )
        if sequence_end_timestamp_us < sequence_start_timestamp_us:
            raise ValueError("Context-camera and rig-pose timestamp ranges do not overlap")
        gap_us = (
            self.config.frame_batch_sampler.frame_gap_timestamp_us
            if self.config.frame_batch_sampler.name == "uniform"
            else self.config.frame_batch_sampler.max_frame_gap_timestamp_us
        )
        max_chunk_timespan_us = gap_us * self._n_frames_per_sample
        required_chunks = max(
            1,
            math.ceil(
                (sequence_end_timestamp_us - sequence_start_timestamp_us) / max_chunk_timespan_us
            ),
        )
        if required_chunks > self.num_samples_per_sequence:
            raise ValueError(
                "Full calibrated video requires a complete reconstructed scene, but "
                f"--max-chunks={self.num_samples_per_sequence} covers only the first part of this clip. "
                f"Rerun with --max-chunks {required_chunks}."
            )

        frame_timestamps_us = torch.from_numpy(
            np.stack([frame_start_timestamps_us, frame_end_timestamps_us], axis=1)
        ).to(device=reference_trajectory.T_rig_world_timestamps_us.device)

        # Keep interpolation valid at every exposure boundary. Some source
        # sequences start/end just outside the nominal rig-pose range; constant
        # endpoint padding matches the inference loader's trajectory policy.
        full_T_rig_worlds = reference_trajectory.T_rig_worlds
        full_T_rig_world_timestamps_us = reference_trajectory.T_rig_world_timestamps_us
        sensor_min_timestamp_us = int(frame_start_timestamps_us.min()) - 1
        sensor_max_timestamp_us = int(frame_end_timestamps_us.max()) + 1
        if sensor_min_timestamp_us < int(full_T_rig_world_timestamps_us[0].item()):
            full_T_rig_worlds = torch.cat([full_T_rig_worlds[:1], full_T_rig_worlds], dim=0)
            full_T_rig_world_timestamps_us = torch.cat(
                [
                    full_T_rig_world_timestamps_us.new_tensor([sensor_min_timestamp_us]),
                    full_T_rig_world_timestamps_us,
                ],
                dim=0,
            )
        if sensor_max_timestamp_us > int(full_T_rig_world_timestamps_us[-1].item()):
            full_T_rig_worlds = torch.cat([full_T_rig_worlds, full_T_rig_worlds[-1:]], dim=0)
            full_T_rig_world_timestamps_us = torch.cat(
                [
                    full_T_rig_world_timestamps_us,
                    full_T_rig_world_timestamps_us.new_tensor([sensor_max_timestamp_us]),
                ],
                dim=0,
            )

        camera_subsampler = self._build_camera_subsampler()
        camera_model_parameters = self._subsample_camera_model_parameters(camera_sensor, camera_subsampler)
        source_T_sensor_rig = np.asarray(unpack_optional(camera_sensor.T_sensor_rig))
        camera_calibration = RigTrajectories.CameraCalibration(
            sequence_id=reference_calibration.sequence_id,
            unique_sensor_idx=reference_calibration.unique_sensor_idx,
            T_sensor_rig=to_torch(
                source_T_sensor_rig,
                device=reference_calibration.T_sensor_rig.device,
                dtype=reference_calibration.T_sensor_rig.dtype,
            ),
            camera_model_parameters=camera_model_parameters,
        )
        full_trajectory = RigTrajectories.RigTrajectory(
            sequence_id=reference_trajectory.sequence_id,
            cameras_frame_timestamps_us={selected_camera_id: frame_timestamps_us},
            T_rig_worlds=full_T_rig_worlds,
            T_rig_world_timestamps_us=full_T_rig_world_timestamps_us,
        )
        logger.info(
            "Loaded %d source frames for render camera %s without images or rays.",
            len(frame_timestamps_us),
            selected_camera_id,
        )
        return RigTrajectories(
            T_world_base=reference_rig.T_world_base,
            world_to_scene=reference_rig.world_to_scene,
            rig_trajectories=[full_trajectory],
            camera_calibrations=OrderedDict([(selected_camera_id, camera_calibration)]),
        )

    def getitem_allow_exceptions(
        self,
        batch_idx: int,
        rng: np.random.Generator,
    ) -> InstantNuRecDataBatch:
        # Disable fsspect INFO logs to not spam the logs.
        logging.getLogger("fsspec").setLevel(logging.WARNING)

        sequence_idx: int = batch_idx // self.num_samples_per_sequence
        sample_idx: int = batch_idx % self.num_samples_per_sequence

        frame_batch_sampler = self._build_frame_batch_sampler()
        assert sample_idx < frame_batch_sampler.n_samples_per_sequence, "Sample index out of bounds"

        context_id_lookup = {str(c): c for c in self.all_context_camera_ids}
        supervision_id_lookup = {str(c): c for c in self.all_supervision_camera_ids}

        context_camera_ids: list[NCoreInstantNuRecDataset.ExtendedCameraId] = [
            context_id_lookup[str(NCoreInstantNuRecDataset.ExtendedCameraId.from_config(camera_id))]
            for camera_id in self.config.context_camera_ids
        ]
        supervision_camera_ids: list[NCoreInstantNuRecDataset.ExtendedCameraId] = [
            supervision_id_lookup[str(NCoreInstantNuRecDataset.ExtendedCameraId.from_config(camera_id))]
            for camera_id in self.config.supervision_camera_ids
        ]
        assert set(map(str, context_camera_ids)) <= set(map(str, supervision_camera_ids)), (
            f"context_camera_ids must be a subset of supervision_camera_ids; "
            f"context={sorted(map(str, context_camera_ids))} "
            f"supervision={sorted(map(str, supervision_camera_ids))}"
        )

        ncore_json_path: UPath = self.ncore_json_paths[sequence_idx]
        if not ncore_json_path.exists():
            raise InstantNuRecDataError(f"{ncore_json_path} does not exist.")

        loaders_sensors = self._get_loaders_and_sensors(ncore_json_path, supervision_camera_ids)
        T_rig_worlds_with_timestamps_us = loaders_sensors.T_rig_worlds_with_timestamps_us
        sequence_loaders = loaders_sensors.sequence_loaders
        aux_loaders = loaders_sensors.aux_loaders
        camera_sensors = loaders_sensors.camera_sensors
        main_loader_key = NCoreInstantNuRecDataset.ExtendedCameraId.main_loader_key()
        sequence_loader = sequence_loaders[main_loader_key]

        # Determine the timestamps interval to select frames from.
        context_camera_frame_timestamps_us: dict[str, np.ndarray] = {}

        # Standalone predict always selects the full sequence range; subranges
        # were a training-time control that the predict YAML never carried.
        main_timestamps = T_rig_worlds_with_timestamps_us[main_loader_key][1]
        select_intervals = [HalfClosedInterval(int(main_timestamps.min()), int(main_timestamps.max()))]
        for loader_key, (_, pose_timestamps_us) in T_rig_worlds_with_timestamps_us.items():
            if loader_key != main_loader_key:
                select_intervals = interval_list_intersect(
                    select_intervals,
                    HalfClosedInterval(int(pose_timestamps_us.min()), int(pose_timestamps_us.max())),
                )
        # Intersect also with sensor timestamps (with +/- 0.1s tolerance)
        for camera_id in context_camera_ids:
            timestamps_us = camera_sensors[camera_id].get_frames_timestamps_us(ncore.data.FrameTimepoint.END)
            select_intervals = interval_list_intersect(
                select_intervals,
                HalfClosedInterval(int(timestamps_us.min()) - 100000, int(timestamps_us.max()) + 100000),
            )
            context_camera_frame_timestamps_us[str(camera_id)] = timestamps_us

        context_frame_batch = frame_batch_sampler.sample_frame_batch(
            sample_idx,
            context_camera_frame_timestamps_us,
            select_intervals,
            rng=rng,
        )
        if len(context_frame_batch) == 0:
            # If nothing is sampled (e.g. out of bounds), return 0-sized batch to be concatenated with other batches.
            return InstantNuRecDataBatch(context=[], cuboid_tracks=[], context_rig=[], meta=[])

        # Determine a good reference coordinates (first camera first frame - non-rig)
        ref_camera_id = context_camera_ids[0]
        T_world_ref = camera_sensors[ref_camera_id].get_frames_T_source_sensor(
            source_node="world",
            frame_indices=min(context_frame_batch[str(ref_camera_id)]),
            frame_timepoint=ncore.data.FrameTimepoint.END,
        )

        # Load context frames.
        context_camera_subsampler = self._build_camera_subsampler()
        context_rig_trajectory, context_camera_mapping = self._get_rig_trajectory(
            "context-",
            context_frame_batch,
            camera_sensors,
            T_world_ref,
            T_rig_worlds_with_timestamps_us,
            context_camera_subsampler,
        )
        context = DataAndRenderingBatch(
            data=self._load_data_batch(
                context_frame_batch,
                context_camera_mapping,
                camera_sensors,
                context_camera_subsampler,
                aux_loaders if self.config.aux_data.enabled_context else {},
            )
        )

        # Training samples novel supervision independently inside the temporal
        # extent selected for this context chunk.  Predict profiles leave
        # n_frames_per_camera at zero and retain the original behavior.
        supervision = None
        supervision_rig_trajectory = None
        n_supervision_frames = self.config.supervision_frame_batch.n_frames_per_camera
        if n_supervision_frames > 0:
            context_times: list[int] = []
            for camera_id in context_camera_ids:
                frame_indices = context_frame_batch[str(camera_id)]
                camera_times = camera_sensors[camera_id].get_frames_timestamps_us(ncore.data.FrameTimepoint.END)
                context_times.extend(int(camera_times[index]) for index in frame_indices)
            start_us, end_us = min(context_times), max(context_times)
            supervision_frame_batch: SampledSensorFrameIdxs = {}
            for camera_id in supervision_camera_ids:
                camera_times = camera_sensors[camera_id].get_frames_timestamps_us(ncore.data.FrameTimepoint.END)
                min_index = max(
                    get_closest_frame_index(
                        camera_times,
                        start_us - self.config.supervision_frame_batch.prepend_timestamps_us,
                    ),
                    0,
                )
                max_index = min(
                    get_closest_frame_index(
                        camera_times,
                        end_us + self.config.supervision_frame_batch.append_timestamps_us,
                    ),
                    len(camera_times) - 1,
                )
                candidates = np.arange(min_index, max_index + 1)
                if self.config.supervision_frame_batch.sample_strategy == "random":
                    indices = np.sort(
                        rng.choice(candidates, size=n_supervision_frames, replace=True)
                    ).tolist()
                else:
                    indices = []
                    bins = np.linspace(0, len(candidates), n_supervision_frames + 1, dtype=int)
                    for bin_start, bin_end in zip(bins[:-1], bins[1:]):
                        if bin_start == bin_end:
                            indices.append(int(candidates[bin_start]))
                        else:
                            indices.append(int(rng.choice(candidates[bin_start:bin_end])))
                if camera_id.sample_ratio != 1.0:
                    sampled_size = round(len(indices) * camera_id.sample_ratio)
                    indices = np.sort(rng.choice(indices, size=sampled_size, replace=False)).tolist()
                if self.config.supervision_frame_batch.include_context_frames and str(camera_id) in context_frame_batch:
                    indices = sorted(set(indices) | set(context_frame_batch[str(camera_id)]))
                if indices:
                    supervision_frame_batch[str(camera_id)] = indices

            supervision_subsampler_cfg = self.config.supervision_frame_batch.camera_subsampler
            supervision_camera_subsampler = CameraSubsampler(
                frame_width=supervision_subsampler_cfg.frame_width,
                frame_height=supervision_subsampler_cfg.frame_height,
            )
            supervision_rig_trajectory, supervision_camera_mapping = self._get_rig_trajectory(
                "supervision-",
                supervision_frame_batch,
                camera_sensors,
                T_world_ref,
                T_rig_worlds_with_timestamps_us,
                supervision_camera_subsampler,
            )
            supervision = DataAndRenderingBatch(
                data=self._load_data_batch(
                    supervision_frame_batch,
                    supervision_camera_mapping,
                    camera_sensors,
                    supervision_camera_subsampler,
                    aux_loaders,
                )
            )

        cuboid_tracks = self._compute_cuboid_tracks(
            context_frame_batch,
            sequence_loader,
            camera_sensors,
            T_world_ref,
        )

        meta = {
            "ncore_json_path": ncore_json_path,
            "sequence_id": sequence_loader.sequence_id,
        }

        instantnurec_data_batch = InstantNuRecDataBatch(
            context=[context],
            supervision=[supervision] if supervision is not None else None,
            context_rig=[context_rig_trajectory],
            supervision_rig=[supervision_rig_trajectory] if supervision_rig_trajectory is not None else None,
            cuboid_tracks=[cuboid_tracks],
            meta=[meta],
        )

        return instantnurec_data_batch
