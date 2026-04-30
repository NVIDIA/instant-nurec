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
from typing import Optional, Self, Tuple, TypeAlias, Union, cast

import torch

from libs.sensors.kernels.pose_calib import compute_poses_and_timestamps
from ncore.data import ShutterType
from ncore.sensors import (
    CameraModel,
    FThetaCameraModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
    RowOffsetStructuredSpinningLidarModel,
)
from nre.utils.geometry import (
    interpolate_se3_poses,
    quat_slerp,
    quat_to_so3_matrix,
    rotation_6d_to_matrix,
    so3_matrix_to_quat,
)
from nre.utils.misc import unpack_optional
from nre.utils.torch_compile import TorchCompile


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
    @staticmethod
    @TorchCompile.conditional(fullgraph=True, dynamic=True)
    def compute_poses_calib(T_sensor_world_startend: torch.Tensor, pose_deltas: torch.Tensor) -> torch.Tensor:
        dx, drot = torch.split(pose_deltas, [3, 6], dim=-1)
        rot = rotation_6d_to_matrix(drot)  # (* 3, 3)

        # The single per-frame pose delta needs to be broadcasted to both start and end poses.
        # If unique_frame_idx is not provided, T_sensor_world_startend is (N, 2, 4, 4)) and rot is (N, 3, 3),
        # so this requires adding a dimension for start-end pairs to broadcast rot and dx to start/end poses.
        if T_sensor_world_startend.ndim == 4:
            rot = rot.unsqueeze(1)  # (N, 3, 3) -> (N, 1, 3, 3), to be broadcasted to (N, 2, 3, 3) below
            dx = dx.unsqueeze(1)  # (N, 3) -> (N, 1, 3), to be broadcasted to (N, 2, 3) below

        transform = torch.broadcast_to(
            torch.eye(4, device=pose_deltas.device, dtype=pose_deltas.dtype), T_sensor_world_startend.shape
        ).clone()  # (*, 4, 4)
        transform[..., :3, :3] = rot  # rot is broadcasted (N,1,3,3) -> (N,2,3,3) when 4-dimensional
        transform[..., :3, 3] = dx  # dx is broadcasted (N,1,3) -> (N,2,3) when 4-dimensional

        # There is only one delta transformation per frame, applied to both the frame start and end poses here.
        # The order T_sensor_world_startend @ transform implies that the transformation is applied in camera space.
        return torch.matmul(T_sensor_world_startend, transform)  # (*, 4, 4)

    @staticmethod
    def get_poses_calib(
        embeds: Optional[torch.nn.Embedding],
        T_sensor_world_startend_allviews: torch.Tensor,
        unique_frame_idx: Optional[int] = None,
        unique_frame_idx_tensor: Optional[torch.Tensor] = None,
        enable_calib: bool = True,
        enable_torch_compile: bool = False,
    ) -> torch.Tensor:
        """
        Get the calibrated pose for a given frame index.

        If enable_calib is False, return the raw pose.

        If unique_frame_idx is None, return all poses.
        """
        device = T_sensor_world_startend_allviews.device
        has_unique_frame_idx = unique_frame_idx is not None and unique_frame_idx != -1

        T_sensor_world_startend = (
            T_sensor_world_startend_allviews[unique_frame_idx]
            if has_unique_frame_idx
            else T_sensor_world_startend_allviews
        )

        if enable_calib:
            assert embeds is not None
            frame_idx = (
                unique_frame_idx_tensor
                if has_unique_frame_idx
                else torch.arange(len(T_sensor_world_startend_allviews), device=device)
            )
            poses_deltas = embeds(frame_idx)  # (N, 9)
            T_sensor_world_startend = SensorModelComputations.compute_poses_calib(
                T_sensor_world_startend, poses_deltas, enable_torch_compile=enable_torch_compile
            )
        else:
            # To make torch autograd happy, we still hook the embeds into the autograd graph
            if embeds is not None:
                zero = embeds(torch.tensor(0, device=device)).sum() * 0.0
                T_sensor_world_startend = T_sensor_world_startend + zero

        return T_sensor_world_startend

    @dataclass
    class PosesAndTimestampsStartendReturn:
        T_sensor_world_startend: torch.Tensor
        timestamps_startend_us: torch.Tensor  # (2,)
        timestamps_startend_us_gpu: torch.Tensor  # (1, 2)
        timestamps_startend_us_cpu: torch.Tensor  # (1, 2)

    @staticmethod
    # Torch compiled function at this level is suspected to provoke errors in multi-gpu
    # @TorchCompile.conditional(fullgraph=True, dynamic=True)
    def _get_poses_and_timestamps_startend_compiled(
        subsample_rect_points_lb: Optional[torch.Tensor],
        subsample_resolution: Optional[torch.Tensor],
        embeds: Optional[torch.nn.Embedding],
        T_offset_nre_startend: Optional[torch.Tensor],
        T_sensor_world_startend_allviews: torch.Tensor,
        timestamps_startend_us_allviews: torch.Tensor,
        sensor_model_shutter_type_if_not_lidar: Optional[ShutterType],
        unique_frame_idx: int,
        unique_frame_idx_tensor: Optional[torch.Tensor],
        enable_calib: bool = True,
        is_lidar: bool = False,
        enable_torch_compile: bool = False,
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """
        Getter to request startend sensor poses and timestamps for a given frame index and sensor index.
        """
        T_sensor_world_startend = SensorModelComputations.get_poses_calib(
            embeds,
            T_sensor_world_startend_allviews,
            unique_frame_idx,
            unique_frame_idx_tensor,
            enable_calib,
            enable_torch_compile=enable_torch_compile,
        )
        timestamps_startend_us = timestamps_startend_us_allviews[unique_frame_idx]
        if subsample_rect_points_lb is not None and subsample_resolution is not None:
            if is_lidar:
                raise NotImplementedError("subsample on poses and timestamps is not supported for Lidar")
            else:
                sensor_model_shutter_type = unpack_optional(sensor_model_shutter_type_if_not_lidar)
                T_sensor_world_startend, timestamps_startend_us = (
                    CameraModelComputations.apply_rect_subsampled_to_camera_rolling_shutter(
                        subsample_rect_points_lb,
                        subsample_resolution,
                        sensor_model_shutter_type,
                        T_sensor_world_startend,
                        timestamps_startend_us,
                        enable_torch_compile=enable_torch_compile,
                    )
                )

        timestamps_startend_us_cpu = timestamps_startend_us.unsqueeze(0).clone().cpu()

        return SensorModelComputations.PosesAndTimestampsStartendReturn(
            T_sensor_world_startend=T_sensor_world_startend
            if T_offset_nre_startend is None
            else T_offset_nre_startend @ T_sensor_world_startend,
            timestamps_startend_us=timestamps_startend_us,
            timestamps_startend_us_gpu=timestamps_startend_us.unsqueeze(0),
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
        )

    @staticmethod
    def _get_poses_and_timestamps_startend_slang(
        subsample_rect_points_lb: Optional[torch.Tensor],
        subsample_resolution: Optional[torch.Tensor],
        subsample_rect_points_lb_cpu: Optional[torch.Tensor],
        subsample_resolution_cpu: Optional[torch.Tensor],
        embeds: Optional[torch.nn.Embedding],  # None if enable_calib is False
        T_offset_nre_startend: Optional[torch.Tensor],
        T_sensor_world_startend_allviews: torch.Tensor,
        timestamps_startend_us_allviews: torch.Tensor,
        timestamps_startend_us_allviews_cpu: torch.Tensor,
        shutter_type: ShutterType,
        unique_frame_idx: int,
        unique_frame_idx_tensor: Optional[torch.Tensor],
        enable_calib: bool = True,
        is_lidar: bool = False,
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """
        GPU implementation using Slang kernel for pose calibration and rolling shutter interpolation.
        """
        device = T_sensor_world_startend_allviews.device

        # Prepare frame_idx tensor
        if unique_frame_idx_tensor is None:
            unique_frame_idx_tensor = torch.tensor([unique_frame_idx], dtype=torch.int64, device=device)

        # Lidar doesn't support subsampling
        if is_lidar and subsample_rect_points_lb is not None:
            raise NotImplementedError("subsample on poses and timestamps is not supported for Lidar")

        # Call Slang kernel
        T_sensor_world_startend_batch, timestamps_startend_us_batch = compute_poses_and_timestamps(
            T_sensor_world_startend_allviews,
            embeds.weight if embeds is not None else None,
            unique_frame_idx_tensor,
            subsample_rect_points_lb,
            subsample_resolution,
            timestamps_startend_us_allviews,
            shutter_type.value,
            enable_calib,
        )

        # Squeeze batch dimension (batch_size=1) to get single frame result
        T_sensor_world_startend = T_sensor_world_startend_batch.squeeze(0)  # (2, 4, 4)
        timestamps_startend_us = timestamps_startend_us_batch.squeeze(0)  # (2,)

        # Apply T_offset if present
        if T_offset_nre_startend is not None:
            T_sensor_world_startend = T_offset_nre_startend @ T_sensor_world_startend

        timestamps_startend_us_gpu = timestamps_startend_us.unsqueeze(0)

        # Compute timestamps entirely on CPU - no GPU sync needed
        if subsample_rect_points_lb_cpu is not None and subsample_resolution_cpu is not None:
            # Interpolate timestamps on CPU using the helper
            timestamps_startend_us_cpu = CameraModelComputations.interpolate_rect_timestamps_cpu(
                subsample_rect_points_lb_cpu.squeeze(0),  # (1, 2, 2) -> (2, 2)
                subsample_resolution_cpu.squeeze(0),  # (1, 2) -> (2,)
                shutter_type,
                timestamps_startend_us_allviews_cpu[unique_frame_idx],  # (2,)
            )
        else:
            timestamps_startend_us_cpu = timestamps_startend_us_allviews_cpu[unique_frame_idx]

        timestamps_startend_us_cpu = timestamps_startend_us_cpu.unsqueeze(0)

        return SensorModelComputations.PosesAndTimestampsStartendReturn(
            T_sensor_world_startend=T_sensor_world_startend,
            timestamps_startend_us=timestamps_startend_us,
            timestamps_startend_us_gpu=timestamps_startend_us_gpu,
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
        )

    @staticmethod
    def get_poses_and_timestamps_startend(
        subsample: Optional[RectSubsampledSensor],
        embeds: Optional[torch.nn.Embedding],
        T_offset_nre_startend: Optional[torch.Tensor],
        T_sensor_world_startend_allviews: torch.Tensor,
        timestamps_startend_us_allviews: torch.Tensor,
        timestamps_startend_us_allviews_cpu: torch.Tensor,
        sensor_models: torch.nn.ModuleDict,
        unique_frame_idx: int,
        unique_frame_idx_tensor: Optional[torch.Tensor],
        unique_sensor_idx_str: str,
        enable_calib: bool = True,
        is_lidar: bool = False,
        enable_torch_compile: bool = False,
    ):
        shutter_type = (
            ShutterType.GLOBAL if is_lidar else cast(ShutterType, sensor_models[unique_sensor_idx_str].shutter_type)
        )

        # GPU path: use Slang kernel for CUDA tensors
        if T_sensor_world_startend_allviews.is_cuda:
            return SensorModelComputations._get_poses_and_timestamps_startend_slang(
                subsample.rect_points_lb.unsqueeze(0) if subsample is not None else None,
                subsample.resolution.unsqueeze(0) if subsample is not None else None,
                subsample.rect_points_lb_cpu.unsqueeze(0) if subsample is not None else None,
                subsample.resolution_cpu.unsqueeze(0) if subsample is not None else None,
                embeds,
                T_offset_nre_startend,
                T_sensor_world_startend_allviews,
                timestamps_startend_us_allviews,
                timestamps_startend_us_allviews_cpu,
                shutter_type,
                unique_frame_idx,
                unique_frame_idx_tensor,
                enable_calib,
                is_lidar,
            )

        # CPU fallback: use PyTorch reference implementation
        return SensorModelComputations._get_poses_and_timestamps_startend_compiled(
            subsample.rect_points_lb if subsample is not None else None,
            subsample.resolution if subsample is not None else None,
            embeds,
            T_offset_nre_startend,
            T_sensor_world_startend_allviews,
            timestamps_startend_us_allviews,
            shutter_type if not is_lidar else None,
            unique_frame_idx,
            unique_frame_idx_tensor,
            enable_calib,
            is_lidar,
            enable_torch_compile=enable_torch_compile,
        )


