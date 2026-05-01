# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Most of the code here is copied from NCore in ncore/impl/sensors/ and adapted to use only torch (and not numpy).

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Self, TypeAlias, Union, cast

import torch

from libs.sensors.kernels.pose_calib import compute_poses_and_timestamps
from ncore.data import ShutterType
from ncore.sensors import (
    CameraModel,
    FThetaCameraModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
)


ConcreteCameraModelsUnion: TypeAlias = Union[FThetaCameraModel, OpenCVFisheyeCameraModel, OpenCVPinholeCameraModel]


@dataclass(kw_only=True, slots=True)
class RectSubsampledBase:
    """
    Subsampled rectangular pixel region with offset i/j and dimension height/width.

    Note that the offset i/j and dimension height/width are relative to the scaled pixel domain. I.e., subsampling is applied first, then cropping.

    The fields are:
    - original_width: The original width of the sensor. [int]
    - original_height: The original height of the sensor. [int]
    - width: The width of the pixel region. [int]
    - height: The height of the pixel region. [int]
    - i: Optional. The offset in the x-direction. Default is 0. [int]
    - j: Optional. The offset in the y-direction. Default is 0. [int]
    - subsample_factor: Optional. The amount of isotropic subsampling. The larger the value is, the smaller the sampled region will be. 1 means no subsampling. Default is 1. [float]
    """

    original_width: int
    original_height: int
    width: int
    height: int
    i: int = 0
    j: int = 0
    subsample_factor: float = 1.0

    def __post_init__(self):
        assert self.subsample_factor > 0.0, "Invalid subsample_factor value"
        assert self.i >= 0, "Invalid i value"
        assert self.j >= 0, "Invalid j value"
        assert self.width > 0, "Invalid width value"
        assert self.height > 0, "Invalid height value"


@dataclass(kw_only=True, slots=True)
class RectSubsampledSensor(RectSubsampledBase):
    # Subsample-related attributes to be computed after initialization.
    rect_points_lb: torch.Tensor = field(init=False, repr=False)
    resolution: torch.Tensor = field(init=False, repr=False)

    # CPU copies of the rect_points_lb and resolution tensors
    rect_points_lb_cpu: torch.Tensor = field(init=False, repr=False)
    resolution_cpu: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        RectSubsampledBase.__post_init__(self)
        self.rect_points_lb = 0.5 + self.subsample_factor * torch.tensor(
            [
                [self.i, self.j],
                [
                    self.i + (self.width - 1.0 / self.subsample_factor),
                    self.j + (self.height - 1.0 / self.subsample_factor),
                ],
            ],
            dtype=torch.float32,
        )
        self.resolution = torch.tensor([self.original_width, self.original_height], dtype=torch.float32)

        # Keep CPU copies of the rect_points_lb and resolution tensors
        self.rect_points_lb_cpu = self.rect_points_lb.clone().cpu()
        self.resolution_cpu = self.resolution.clone().cpu()

    def to(self, *args, **kwargs) -> Self:
        obj = replace(self)
        obj.rect_points_lb = obj.rect_points_lb.to(*args, **kwargs)
        obj.resolution = obj.resolution.to(*args, **kwargs)
        assert obj.rect_points_lb.shape == (2, 2) and obj.rect_points_lb.dtype == torch.float32
        assert obj.resolution.shape == (2,) and obj.resolution.dtype == torch.float32
        return obj


class SensorModelComputations:
    @dataclass
    class PosesAndTimestampsStartendReturn:
        T_sensor_world_startend: torch.Tensor
        timestamps_startend_us: torch.Tensor  # (2,)
        timestamps_startend_us_gpu: torch.Tensor  # (1, 2)
        timestamps_startend_us_cpu: torch.Tensor  # (1, 2)

    @staticmethod
    def _get_poses_and_timestamps_startend_slang(
        subsample_rect_points_lb: Optional[torch.Tensor],
        subsample_resolution: Optional[torch.Tensor],
        subsample_rect_points_lb_cpu: Optional[torch.Tensor],
        subsample_resolution_cpu: Optional[torch.Tensor],
        T_sensor_world_startend_allviews: torch.Tensor,
        timestamps_startend_us_allviews: torch.Tensor,
        timestamps_startend_us_allviews_cpu: torch.Tensor,
        shutter_type: ShutterType,
        unique_frame_idx: int,
        unique_frame_idx_tensor: Optional[torch.Tensor],
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """
        GPU implementation using Slang kernel for rolling shutter interpolation.
        """
        device = T_sensor_world_startend_allviews.device

        # Prepare frame_idx tensor
        if unique_frame_idx_tensor is None:
            unique_frame_idx_tensor = torch.tensor([unique_frame_idx], dtype=torch.int64, device=device)

        # Call Slang kernel
        T_sensor_world_startend_batch, timestamps_startend_us_batch = compute_poses_and_timestamps(
            T_sensor_world_startend_allviews,
            None,  # embed_weights: standalone predict has no per-frame calibration deltas
            unique_frame_idx_tensor,
            subsample_rect_points_lb,
            subsample_resolution,
            timestamps_startend_us_allviews,
            shutter_type.value,
            False,  # enable_calib: dead in standalone predict (no calibration learning)
        )

        # Squeeze batch dimension (batch_size=1) to get single frame result
        T_sensor_world_startend = T_sensor_world_startend_batch.squeeze(0)  # (2, 4, 4)
        timestamps_startend_us = timestamps_startend_us_batch.squeeze(0)  # (2,)

        timestamps_startend_us_gpu = timestamps_startend_us.unsqueeze(0)

        # Standalone predict pins `FrameMeta.subsample = None`, so the NRE-side
        # `interpolate_rect_timestamps_cpu` per-pixel-rect path was unreachable.
        timestamps_startend_us_cpu = timestamps_startend_us_allviews_cpu[unique_frame_idx].unsqueeze(0)

        return SensorModelComputations.PosesAndTimestampsStartendReturn(
            T_sensor_world_startend=T_sensor_world_startend,
            timestamps_startend_us=timestamps_startend_us,
            timestamps_startend_us_gpu=timestamps_startend_us_gpu,
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
        )

    @staticmethod
    def get_poses_and_timestamps_startend(
        subsample: Optional[RectSubsampledSensor],
        T_sensor_world_startend_allviews: torch.Tensor,
        timestamps_startend_us_allviews: torch.Tensor,
        timestamps_startend_us_allviews_cpu: torch.Tensor,
        sensor_models: torch.nn.ModuleDict,
        unique_frame_idx: int,
        unique_frame_idx_tensor: Optional[torch.Tensor],
        unique_sensor_idx_str: str,
    ):
        # Standalone predict requires CUDA tensors; the compiled CPU fallback
        # was dropped in Phase 1 step 4.3.
        assert T_sensor_world_startend_allviews.is_cuda, (
            "get_poses_and_timestamps_startend requires CUDA tensors in the standalone predict pipeline."
        )
        shutter_type = cast(ShutterType, sensor_models[unique_sensor_idx_str].shutter_type)
        return SensorModelComputations._get_poses_and_timestamps_startend_slang(
            subsample.rect_points_lb.unsqueeze(0) if subsample is not None else None,
            subsample.resolution.unsqueeze(0) if subsample is not None else None,
            subsample.rect_points_lb_cpu.unsqueeze(0) if subsample is not None else None,
            subsample.resolution_cpu.unsqueeze(0) if subsample is not None else None,
            T_sensor_world_startend_allviews,
            timestamps_startend_us_allviews,
            timestamps_startend_us_allviews_cpu,
            shutter_type,
            unique_frame_idx,
            unique_frame_idx_tensor,
        )


