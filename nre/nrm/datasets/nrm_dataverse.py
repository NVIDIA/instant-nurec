# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

from dataverse.datasets import DataField
from dataverse.utils import dataset_from_config
from einops import rearrange, repeat
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional as F

from ncore.data import ConcreteCameraModelParametersUnion, OpenCVPinholeCameraModelParameters, ShutterType
from nre.nrm.config.dataset import DataverseNRMDatasetConfig
from nre.nrm.datasets.nrm_base import BaseNRMIndexableDataset, CameraSubsampler, NRMDataError
from nre.nrm.datasets.registry import register as register_dataset
from nre.nrm.datasets.samplers import BaseFrameBatchSampler, sample_supervision_frame_batch
from nre.nrm.datasets.samplers import make as make_frame_batch_sampler
from nre.utils.batch import (
    CameraFrameLabels,
    CameraFreePoseViewGeometry,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    NRMDataBatch,
    RenderingBatch,
)
from nre.utils.misc import unpack_optional
from nre.utils.types import FrameConversion, HalfClosedInterval, RayFlags, RigTrajectories


@register_dataset("nrm-dataverse")
class DataverseNRMDataset(BaseNRMIndexableDataset):
    """
    Represents a single dataset from dataverse (e.g. Re10K, MVImgNet, DL3DV)
    """

    def __init__(
        self,
        config: DataverseNRMDatasetConfig,
        split: str = "train",
    ) -> None:
        self.config = config
        self.dataverse_dataset = dataset_from_config(self._config_to_dictconfig(config))
        self._num_videos = self.dataverse_dataset.num_videos()  # cache in case it's expensive
        self.required_fields = frozenset(
            [
                DataField.IMAGE_RGB,
                DataField.CAMERA_C2W_TRANSFORM,
                DataField.CAMERA_INTRINSICS,
            ]
        )
        available_fields = frozenset(self.dataverse_dataset.available_data_fields())
        if missing := self.required_fields - available_fields:
            raise ValueError(f"Required fields {missing} not available in the dataset")

        self.camera_subsampler = CameraSubsampler(config.camera_subsampler)

        self._frame_counts: dict[tuple[int, int], int] = {}  # key is (video_idx, view_idx)

        # Multiple frame samplers will be concatenated in the final dataset.
        self.frame_batch_samplers: list[BaseFrameBatchSampler] = []
        for sampler_config in config.frame_batch_samplers.values():
            self.frame_batch_samplers.append(make_frame_batch_sampler(sampler_config.name, sampler_config))

    def __repr__(self) -> str:
        return f"DataverseNRMDataset(config={self.config})"  # helpful for debugging

    @property
    def num_samples_per_sequence(self) -> int:
        return sum([frame_batch_sampler.num_samples_per_sequence for frame_batch_sampler in self.frame_batch_samplers])

    def __len__(self) -> int:
        """
        Returns the length of the dataset = num_videos * num_samples_per_sequence.
        """
        return self._num_videos * self.num_samples_per_sequence

    @staticmethod
    def _config_to_dictconfig(config: DataverseNRMDatasetConfig) -> DictConfig:
        """
        A utility function needed to instantiate datasets from dataverse.
        """
        params = {"root_path": config.subset_spec.root_path}
        if isinstance(config.subset_spec, DataverseNRMDatasetConfig.Re10kParams):
            params["annotation_json"] = config.subset_spec.annotation_json

        return DictConfig(
            content={
                "target": config.subset_spec.target,
                "params": params,
            }
        )

    def _get_camera_model_parameters(
        self, src_wh: tuple[int, int], camera_intrinsics: Tensor
    ) -> ConcreteCameraModelParametersUnion:
        """
        Construct CameraModelParameters from camera intrinsics. Returns the parameters for the target (downsampled) resolution.
        """
        # assert camera_intrinsics are the same
        intrinsics = camera_intrinsics[0]  # take the first camera's intrinsics
        assert torch.all(intrinsics == camera_intrinsics), "Camera intrinsics are not the same for all cameras"

        camera_model_parameters = OpenCVPinholeCameraModelParameters(
            resolution=np.array(src_wh, dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            focal_length=intrinsics[0:2].numpy().astype(np.float32),
            principal_point=intrinsics[2:4].numpy().astype(np.float32),
            # TODO: see if it can be populated from the dataset
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
        return self.camera_subsampler.apply_camera_parameters(camera_model_parameters)

    def _count_frames(self, video_idx: int, view_idx: int) -> int:
        """
        Count the number of frames in a video and view.
        Cached because some datasets implement this rather inefficiently.
        """
        key = (video_idx, view_idx)
        if key not in self._frame_counts:
            self._frame_counts[key] = self.dataverse_dataset.num_frames(video_idx, view_idx)
        return self._frame_counts[key]

    @staticmethod
    def _compute_mean_cam_ref(c2ws: Tensor) -> Tensor:
        """Compute a reference frame from the mean of camera poses using Gram-Schmidt orthogonalization.

        Returns:
            T_world_ref: (1, 4, 4) world-to-reference transform (inverse of the mean camera pose).
        """
        position_avg = c2ws[:, :3, 3].mean(0)
        forward_avg = c2ws[:, :3, 2].mean(0)
        down_avg = c2ws[:, :3, 1].mean(0)

        forward_avg = F.normalize(forward_avg, dim=0)
        down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
        right_avg = torch.cross(down_avg, forward_avg)

        mean_c2w = torch.eye(4, device=c2ws.device, dtype=c2ws.dtype)
        mean_c2w[:3, :] = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1)
        return mean_c2w.unsqueeze(0).inverse()

    def _get_rig_trajectory_and_data_batch(
        self, sequence_id: str, video_idx: int, view_idx: int, frame_idxs: list[int], T_world_ref: Optional[Tensor]
    ) -> tuple[RigTrajectories, DataBatch, Tensor]:
        """
        Construct RigTrajectories and DataBatch from the dataverse dataset.
        Note that the T_rig_worlds in RigTrajectories is not the full trajectory, but only for the camera poses.
        This also does not support different camera parameters for different frames of a view.

        Args:
            sequence_id: unique identifier for the sequence
            video_idx: index of the video
            view_idx: index of the view
            frame_idxs: list of frame indices
            T_world_ref: optional world-to-reference transform (if None, determined by config.camera_ref_normalization)

        Returns:
            rig_trajectory: RigTrajectories for the video and view
            data_batch: DataBatch containing the camera data for the video and view
            T_world_ref: world-to-reference transform
        """
        try:
            data = self.dataverse_dataset.read(video_idx, frame_idxs, view_idxs=[view_idx] * len(frame_idxs))
        except Exception as e:
            raise NRMDataError(  # TODO: add something like "DataCorruptionError" to dataverse and only catch that
                f"Error reading data from dataverse, dataset: {self}, video_idx: {video_idx}, "
                f"view_idx: {view_idx}, frame_idxs: {frame_idxs}"
            ) from e

        assert all(field in data for field in self.required_fields)

        images_src_res_rgb_bchw: Tensor = data[DataField.IMAGE_RGB]
        camera_c2w_transform: Tensor = data[DataField.CAMERA_C2W_TRANSFORM]
        camera_intrinsics_src_res: Tensor = data[DataField.CAMERA_INTRINSICS]

        if T_world_ref is None:
            if self.config.camera_ref_normalization == "first_frame":
                T_world_ref = camera_c2w_transform[0:1].inverse()
            else:
                T_world_ref = self._compute_mean_cam_ref(camera_c2w_transform)
        camera_c2w_transform = T_world_ref @ camera_c2w_transform

        src_height, src_width = images_src_res_rgb_bchw.shape[-2:]
        camera_model_parameters = self._get_camera_model_parameters((src_width, src_height), camera_intrinsics_src_res)

        camera_width = self.camera_subsampler.frame_width
        camera_height = self.camera_subsampler.frame_height

        flags = torch.full(
            (len(frame_idxs), camera_height, camera_width),
            RayFlags.RGB_LABEL.value,
            dtype=torch.int32,
            device="cpu",
        )
        labels = CameraFrameLabels()
        # Convert RGB to float [0, 1] so apply_frame_data uses bilinear interpolation (uint8 is for discrete maps)
        images_np = rearrange(images_src_res_rgb_bchw, "B C H W -> B H W C").numpy()
        if images_np.dtype.kind != "f":
            images_np = images_np.astype(np.float32) / 255.0
        labels.rgb = torch.stack(
            [torch.from_numpy(self.camera_subsampler.apply_frame_data(images_np[i])) for i in range(len(images_np))],
            dim=0,
        )
        labels.flags = flags[..., None]

        data_batch = DataBatch(
            camera=DataBatch.Camera(
                meta=[
                    FrameMeta(
                        unique_sensor_idx=view_idx,
                        # For T_rig_worlds of dataverse, we don't load the full trajectory, but only keep the camera poses.
                        # This is different from the ncore dataset where all Poses are loaded into the rig.
                        # Hence unique_frame_idx has to be a range
                        unique_frame_idx=unique_frame_idx,
                        subsample=None,
                    )
                    for unique_frame_idx in range(len(frame_idxs))
                ],
                labels=labels,
            )
        )

        frame_timestamps_us = (
            torch.tensor(frame_idxs, dtype=torch.int64).unsqueeze(1).repeat(1, 2)
        )  # frame_start == frame_end

        # Deduplicate timestamps for the rig pose since samplers might provide duplicated frames.
        T_rig_world_timestamps_us, timestamps_us_idx = torch.unique(
            frame_timestamps_us[..., 1],  # end of frame timestamps,
            return_inverse=True,
        )
        T_rig_worlds = torch.zeros_like(camera_c2w_transform)[: len(T_rig_world_timestamps_us)]
        T_rig_worlds[timestamps_us_idx] = camera_c2w_transform

        # If after dedup only one timestamp remains, duplicate it.
        if len(T_rig_world_timestamps_us) == 1:
            T_rig_worlds = torch.cat([T_rig_worlds, T_rig_worlds])
            T_rig_world_timestamps_us = torch.cat([T_rig_world_timestamps_us, T_rig_world_timestamps_us + 1])

        rig_trajectory = RigTrajectories(
            rig_trajectories=[
                RigTrajectories.RigTrajectory(
                    sequence_id=sequence_id,
                    cameras_frame_timestamps_us={str(view_idx): frame_timestamps_us},
                    lidars_frame_timestamps_us={},
                    T_rig_worlds=T_rig_worlds,
                    T_rig_world_timestamps_us=T_rig_world_timestamps_us,
                    rig_bbox=None,
                )
            ],
            T_world_base=torch.linalg.inv(T_world_ref).to(torch.float64),
            world_to_nre=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            camera_calibrations=OrderedDict(
                [
                    (
                        str(view_idx),
                        RigTrajectories.CameraCalibration(
                            sequence_id=sequence_id,
                            logical_sensor_name=str(view_idx),
                            unique_sensor_idx=view_idx,
                            T_sensor_rig=torch.eye(4, dtype=torch.float32, device="cpu"),
                            camera_model_parameters=camera_model_parameters,
                        ),
                    )
                ]
            ),
            lidar_calibrations=OrderedDict({}),
        )

        return rig_trajectory, data_batch, T_world_ref

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> NRMDataBatch:
        """
        Returns a NRMDataBatch of batch size 1 (to be later collated).

        Args:
            batch_idx: index of the (video, sample) pair in the dataset
        """
        video_idx: int = batch_idx // self.num_samples_per_sequence
        sample_idx: int = batch_idx % self.num_samples_per_sequence

        for frame_batch_sampler in self.frame_batch_samplers:
            if sample_idx < frame_batch_sampler.num_samples_per_sequence:
                break
            sample_idx -= frame_batch_sampler.num_samples_per_sequence

        view_idx = 0
        n_frames = self._count_frames(video_idx, view_idx)

        context_frame_batch = frame_batch_sampler.sample_frame_batch(
            rng,
            sample_idx,
            {str(view_idx): np.arange(n_frames, dtype=np.int64)},
            [HalfClosedInterval(0, n_frames)],
            None,
        )
        supervision_frame_batch = sample_supervision_frame_batch(
            config=self.config.supervision_frame_batch,
            rng=rng,
            context_frame_batch=context_frame_batch,
            sensor_frame_timestamps_us={str(view_idx): np.arange(n_frames, dtype=np.int64)},
            supervision_sensor_ids=[str(view_idx)],
            sensor_sample_ratios=[1.0],
        )

        # Load context frames.
        context_rig_trajectory, context_data_batch, T_world_ref = self._get_rig_trajectory_and_data_batch(
            "context",
            video_idx,
            view_idx,
            context_frame_batch.sampled_sensor_frame_idxs[str(view_idx)],
            T_world_ref=None,  # T_world_ref will be set to the first camera's transform
        )
        context_rendering_data = (
            CameraFreePoseViewGeometry.from_rig_trajectories(context_rig_trajectory)
            .cuda()
            .to_rendering_data(unpack_optional(context_data_batch.camera).to("cuda"), cache_sensor_params=True)
        )

        # Load supervision frames.
        supervision_rig_trajectory, supervision_data_batch, _ = self._get_rig_trajectory_and_data_batch(
            "supervision",
            video_idx,
            view_idx,
            supervision_frame_batch.sampled_sensor_frame_idxs[str(view_idx)],
            T_world_ref=T_world_ref,  # Use the same T_world_ref as for context
        )
        supervision_rendering_data = (
            CameraFreePoseViewGeometry.from_rig_trajectories(supervision_rig_trajectory)
            .cuda()
            .to_rendering_data(unpack_optional(supervision_data_batch.camera).to("cuda"), cache_sensor_params=True)
        )

        return NRMDataBatch(
            context=[
                DataAndRenderingBatch(data=context_data_batch, rendering=RenderingBatch(camera=context_rendering_data))
            ],
            supervision=[
                DataAndRenderingBatch(
                    data=supervision_data_batch, rendering=RenderingBatch(camera=supervision_rendering_data)
                )
            ],
            context_rig=[context_rig_trajectory],
            supervision_rig=[supervision_rig_trajectory],
            meta=None,
        )
