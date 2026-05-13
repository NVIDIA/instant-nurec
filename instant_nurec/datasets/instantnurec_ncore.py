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
import logging

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from upath import UPath

import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import instant_nurec.utils.ncore_utils as ncore_utils

from instant_nurec.datasets.tracks import CuboidTracks, CuboidTracksDataPack, TrackFlags
from instant_nurec.datasets.utils import compute_cuboid_df, consolidate_cuboid_tracks
from instant_nurec.config_schema.dataset import NCoreInstantNuRecDatasetConfig
from instant_nurec.datasets.instantnurec_base import CameraSubsampler, InstantNuRecDataError
from instant_nurec.datasets.samplers import (
    AdaptiveSequentialFrameBatchSampler,
    SampledSensorFrameIdxs,
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
from instant_nurec.utils.types import FrameConversion, HalfClosedInterval, RigTrajectories


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


class NCoreInstantNuRecDataset(torch.utils.data.Dataset[InstantNuRecDataBatch]):
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
        """Camera id with a unique sensor index."""

        camera_id: str
        unique_sensor_idx: int

        @staticmethod
        def from_config(camera_id: str, unique_sensor_idx: int = -1) -> "NCoreInstantNuRecDataset.ExtendedCameraId":
            return NCoreInstantNuRecDataset.ExtendedCameraId(
                camera_id=camera_id, unique_sensor_idx=unique_sensor_idx
            )

        def __str__(self) -> str:
            return self.camera_id

        @property
        def canonical_order(self) -> str:
            """Canonical order to be in rig trajectory and the data batch."""
            return f"{self.unique_sensor_idx:03d}"

    @dataclass
    class LoadersAndSensorsResult:
        """Result of loading sequence loader and camera sensors for an ncore sequence."""

        T_rig_worlds_with_timestamps_us: tuple[np.ndarray, np.ndarray]
        sequence_loader: ncore.data.SequenceLoaderProtocol
        camera_sensors: dict["NCoreInstantNuRecDataset.ExtendedCameraId", ncore.data.CameraSensorProtocol]

    def __init__(
        self,
        config: NCoreInstantNuRecDatasetConfig,
        frame_width: int,
        frame_height: int,
        n_frames_per_sample: int,
    ):
        # ``frame_width`` / ``frame_height`` / ``n_frames_per_sample`` are
        # passed in by the caller (typically ``instant_nurec.model.make``),
        # not via the dataset config.
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._n_frames_per_sample = n_frames_per_sample

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

        self.ncore_json_paths: list[UPath] = [parse_universal_path(p) for p in config.ncore_json_paths]
        logger.info(f"Loaded {len(self.ncore_json_paths)} sequence(s).")

        self.num_samples_per_sequence: int = config.frame_batch_sampler.n_samples_per_sequence
        self.config = config

    def _build_frame_batch_sampler(self) -> AdaptiveSequentialFrameBatchSampler:
        return AdaptiveSequentialFrameBatchSampler(
            self.config.frame_batch_sampler,
            n_frames_per_sample=self._n_frames_per_sample,
        )

    def _build_camera_subsampler(self):
        from instant_nurec.datasets.instantnurec_base import CameraSubsampler

        return CameraSubsampler(frame_width=self._frame_width, frame_height=self._frame_height)

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

            # Determine unique sensor index mapping
            unique_sensor_idx = camera_id.unique_sensor_idx
            for frame_idx in frame_idxs:
                # Collect labels data
                labels = CameraFrameLabels()
                frame_image_array = camera_sensor.get_frame_image_array(frame_idx).astype(np.float32) / 255.0
                frame_image_array = camera_subsampler.apply_frame_data(frame_image_array)
                labels.rgb = to_torch(frame_image_array, device="cpu").unsqueeze(0)

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
        T_rig_worlds_with_timestamps_us: tuple[np.ndarray, np.ndarray],
        camera_subsampler: CameraSubsampler,
    ) -> tuple[RigTrajectories, dict[UniqueFrameId, int]]:
        """
        Obtain rig-trajectory based on the sampled sensors.
        The rig trajectory will contain the full rig poses and frame_batch-sampled cameras.

        This will additionally return a UniqueFrameId to index mapping, which matches the logic of CameraFreePoseViewGeometry
        so we can properly query a frame via its unique frame idx.
        """
        ## Load cameras

        # camera_id_name -> timestamps_us
        frame_timestamps_us_list: list[tuple[int, int]] = []
        camera_frame_timestamps_us: dict[str, torch.Tensor] = {}
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
            camera_model_parameters_copy = dataclasses.replace(camera_sensor.model_parameters)

            # Some camera models have bad linear_cde values, manually fix them without overwriting originals
            if isinstance(camera_model_parameters_copy, ncore.data.FThetaCameraModelParameters) and np.all(
                camera_model_parameters_copy.linear_cde == 0.0
            ):
                camera_model_parameters_copy.linear_cde = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            if isinstance(camera_model_parameters_copy, ncore.data.FThetaCameraModelParameters):
                # (This would make boundary pixels of omnidirectional cameras to be classified as invalid)
                camera_model_parameters_copy.max_angle = min(
                    np.deg2rad(self.camera_max_fov_deg) / 2.0, camera_model_parameters_copy.max_angle
                )

            camera_model_parameters = camera_subsampler.apply_camera_parameters(camera_model_parameters_copy)
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

            camera_frame_timestamps_us[str(camera_id)] = torch.tensor(
                frame_timestamps_us_list, dtype=torch.int64, device="cpu"
            )

        # Standalone predict has a single loader keyed `"main"` (no external archives).
        T_rig_worlds, T_rig_world_timestamps_us = T_rig_worlds_with_timestamps_us

        # In the new batch design the sensor poses can only obtained by interpolating rig poses.
        # In cases where rig timestamps do not fully cover the sensor timestamps, we extend the rig using constant padding.
        # This can happen, e.g., in Gen3C setting where rig timestamps are end-of-frame ones, so start-of-frame of the 1st frame
        # is not covered.
        sensor_min_timestamp_us = int(min((v.min().item() for v in camera_frame_timestamps_us.values()))) - 1
        sensor_max_timestamp_us = int(max((v.max().item() for v in camera_frame_timestamps_us.values()))) + 1
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

        rig_trajectores: list[RigTrajectories.RigTrajectory] = [
            RigTrajectories.RigTrajectory(
                sequence_id=sequence_id_prefix + "main",
                cameras_frame_timestamps_us=camera_frame_timestamps_us,
                T_rig_worlds=to_torch(T_world_ref @ T_rig_worlds, device="cpu", dtype=torch.float64),
                T_rig_world_timestamps_us=to_torch(T_rig_world_timestamps_us, device="cpu", dtype=torch.int64),
            )
        ]

        camera_calibrations = OrderedDict(
            [
                (
                    str(camera_id),
                    RigTrajectories.CameraCalibration(
                        sequence_id=sequence_id_prefix + "main",
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

        # ShardDataLoader is logging using root. Let's suppress this information.
        (root_logger := logging.getLogger()).setLevel(logging.WARNING)
        try:
            sequence_loader = ncore_utils.create_sequence_loader(
                dataset_paths=dataset_paths,
                open_consolidated=self.open_consolidated,
                v4_poses_component_group="default",
                v4_intrinsics_component_group="default",
                v4_masks_component_group="default",
                v4_cuboids_component_group="default",
            )
        except FileNotFoundError as e:
            raise InstantNuRecDataError(f"Ncore files not found for dataset_paths {dataset_paths}.") from e

        # NB [JH]: We should be very careful about the poses' timestamps_us -- it can be a large
        # superset of sensor timestamps (e.g. 36s vs 10s).
        # TODO: frame-pose-only data might fail here as there are no rig poses and might require refined logic.
        rig_world_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
            sequence_loader.pose_graph.get_edge("rig", "world"),
            msg="Rig-to-world poses required for rig-trajectories",
        )
        T_rig_worlds_with_timestamps_us = (
            rig_world_edge.T_source_target,
            unpack_optional(rig_world_edge.timestamps_us, msg="Rig-to-world pose requires to be dynamic"),
        )

        camera_sensors = {
            camera_id: sequence_loader.get_camera_sensor(camera_id.camera_id) for camera_id in all_camera_ids
        }

        root_logger.setLevel(logging.INFO)
        return NCoreInstantNuRecDataset.LoadersAndSensorsResult(
            T_rig_worlds_with_timestamps_us=T_rig_worlds_with_timestamps_us,
            sequence_loader=sequence_loader,
            camera_sensors=camera_sensors,
        )

    def __getitem__(self, batch_idx: int) -> InstantNuRecDataBatch:
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
        sequence_loader = loaders_sensors.sequence_loader
        camera_sensors = loaders_sensors.camera_sensors

        # Determine the timestamps interval to select frames from.
        context_camera_frame_timestamps_us: dict[str, np.ndarray] = {}

        # Standalone predict always selects the full sequence range; subranges
        # were a training-time control that the predict YAML never carried.
        main_timestamps = T_rig_worlds_with_timestamps_us[1]
        select_intervals = [HalfClosedInterval(int(main_timestamps.min()), int(main_timestamps.max()))]
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
            context_rig=[context_rig_trajectory],
            cuboid_tracks=[cuboid_tracks],
            meta=[meta],
        )

        return instantnurec_data_batch
