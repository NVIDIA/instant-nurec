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


class CameraModelComputations:
    @staticmethod
    def pixels_to_image_points(camera_model: CameraModel, pixel_idxs: torch.Tensor) -> torch.Tensor:
        """Given integer-based pixels indices, computes corresponding continuous image point coordinates representing the *center* of each pixel."""

        # Convert to torch
        assert not pixel_idxs.is_floating_point(), "[CameraModel]: Pixel indices must be integers"

        # Compute the image point coordinates representing the center of each pixel (shift from top left corner to the center)
        return pixel_idxs.to(camera_model.dtype) + 0.5

    @staticmethod
    def image_points_relative_frame_times_kernel(
        image_points: torch.Tensor, resolution: torch.Tensor, shutter_type: ShutterType
    ) -> torch.Tensor:
        """Get relative frame-times based on the image point coordinates and rolling shutter type"""

        # Floor/Ceil the continuous image points to the row / column index following the image coordinate
        # convention that index defines the top left corner of each pixel, e.g., the first pixels
        # u/v-range is [0.0, 1.0]
        if shutter_type == ShutterType.ROLLING_TOP_TO_BOTTOM:
            t = torch.floor(image_points[:, 1]) / (resolution[1] - 1)
        elif shutter_type == ShutterType.ROLLING_LEFT_TO_RIGHT:
            t = torch.floor(image_points[:, 0]) / (resolution[0] - 1)
        elif shutter_type == ShutterType.ROLLING_BOTTOM_TO_TOP:
            t = (resolution[1] - torch.ceil(image_points[:, 1])) / (resolution[1] - 1)
        elif shutter_type == ShutterType.ROLLING_RIGHT_TO_LEFT:
            t = (resolution[0] - torch.ceil(image_points[:, 0])) / (resolution[0] - 1)
        elif shutter_type == ShutterType.GLOBAL:
            t = torch.zeros_like(image_points[:, 0])
        else:
            raise TypeError(f"unsupported shutter-type {shutter_type.name} for timestamp interpolation")

        return t

    @staticmethod
    def image_points_relative_frame_times(camera_model: CameraModel, image_points: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for image_points_relative_frame_times_kernel with the camera's resolution and shutter type + tensor conversion"""
        return CameraModelComputations.image_points_relative_frame_times_kernel(
            image_points.to(device=image_points.device, dtype=image_points.dtype),
            camera_model.resolution,
            camera_model.shutter_type,
        )

    @staticmethod
    @torch.compile
    def interpolate_rect_timestamps_cpu(
        rect_points_lt_rb: torch.Tensor,
        resolution: torch.Tensor,
        shutter_type: ShutterType,
        frame_timestamps_startend_us: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate timestamps for a rectangular pixel region based on rolling shutter timing.

        Args:
            rect_points_lt_rb: The left-top and right-bottom points of the rectangular region. (2, 2)
            resolution: The full frame resolution (width, height).
            shutter_type: The sensor shutter type to evaluate the timestamps for.
            frame_timestamps_startend_us: The start and end timestamps of the full sensor frame. (2,)

        Returns:
            The start and end timestamps of the pixel region. (2,)
        """
        assert rect_points_lt_rb.device.type == "cpu"
        assert resolution.device.type == "cpu"
        assert frame_timestamps_startend_us.device.type == "cpu"
        # compute relative frame times of the cropped rectangular region
        rect_relative_frame_times = CameraModelComputations.image_points_relative_frame_times_kernel(
            rect_points_lt_rb, resolution, shutter_type
        )

        # make sure to order relative frame times according to increasing start and end of frame
        # timepoints in case we have "inverse" shutter directions (like right-to-left, down-to-up)
        rect_relative_frame_times = torch.sort(rect_relative_frame_times, dim=0).values

        # evaluate the start/end timestamps associated with the cropped rectangular region
        camera_frame_duration_us = frame_timestamps_startend_us[1] - frame_timestamps_startend_us[0]
        pixelrect_timestamps_startend_us = (
            frame_timestamps_startend_us[0] + (rect_relative_frame_times * camera_frame_duration_us).long()
        )

        return pixelrect_timestamps_startend_us

    @staticmethod
    @TorchCompile.conditional(fullgraph=True, dynamic=True)
    def apply_rect_subsampled_to_camera_rolling_shutter(
        rect_points_lt_rb: torch.Tensor,
        resolution: torch.Tensor,
        shutter_type: ShutterType,
        frame_poses_startend: torch.Tensor,
        frame_timestamps_startend_us: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the rectangular pixel region subsampling to the full frame to obtain poses
           and timestamps corresponding to the crop.

        Args:
            shutter_type: The sensor shutter type to evaluate the timestamps for.
            frame_poses_startend: The start and end poses of the full sensor frame. (2, 4, 4)
            frame_timestamps_startend_us: The start and end timestamps of the full sensor frame. (2,)

        Returns:
            The start and end poses, and timestamps of the pixel region.
        """
        # compute relative frame times of the cropped rectangular region
        rect_relative_frame_times = CameraModelComputations.image_points_relative_frame_times_kernel(
            rect_points_lt_rb, resolution, shutter_type
        )

        # make sure to order relative frame times according to increasing start and end of frame
        # timepoints in case we have "inverse" shutter directions (like right-to-left, down-to-up)
        rect_relative_frame_times = torch.sort(rect_relative_frame_times, dim=0).values

        # evaluate the start/end timestamps associated with the cropped rectangular region
        camera_frame_duration_us = frame_timestamps_startend_us[1] - frame_timestamps_startend_us[0]
        pixelrect_timestamps_startend_us = (
            frame_timestamps_startend_us[0] + (rect_relative_frame_times * camera_frame_duration_us).long()
        )

        # interpolate the poses at the start/end times of the cropped rectangular region
        pixelrect_poses_start, pixelrect_poses_end = interpolate_se3_poses(
            frame_poses_startend[0], frame_poses_startend[1], rect_relative_frame_times
        )
        pixelrect_poses_startend = torch.stack([pixelrect_poses_start, pixelrect_poses_end], dim=0)

        return pixelrect_poses_startend, pixelrect_timestamps_startend_us


class LidarModelComputations:
    @staticmethod
    @TorchCompile.conditional(fullgraph=True, dynamic=True)
    def elements_to_world_rays_shutter_pose(
        lidar_model: RowOffsetStructuredSpinningLidarModel,
        elements: torch.Tensor,
        T_sensor_world_start: torch.Tensor,
        T_sensor_world_end: torch.Tensor,
        start_timestamp_us: Optional[torch.Tensor],
        end_timestamp_us: Optional[torch.Tensor],
        sensor_rays: torch.Tensor,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> RowOffsetStructuredSpinningLidarModel.WorldRaysReturn:
        """Unprojects elements to world rays using *rolling-shutter compensation* of sensor motion.

        Can optionally re-use known sensor rays associated with elements.

        For each element returns 3d world rays [point, direction], represented by 3d start of ray points and 3d ray directions in the world frame
        """
        device = T_sensor_world_start.device

        # Check if the variables are numpy, convert them to torch and send them to correct device
        assert T_sensor_world_start.shape == (4, 4)
        assert T_sensor_world_end.shape == (4, 4)
        assert len(elements.shape) == 2
        assert elements.shape[1] == 2
        # assert elements.dtype == torch.int64
        assert T_sensor_world_start.dtype == lidar_model.dtype
        assert T_sensor_world_end.dtype == lidar_model.dtype

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[LidarModel]: End timestamp must be larger or equal to the start timestamp"
            )

            # Make sure timestamps have correct type (might be, e.g., np.uint64, which torch doesn't like)
            start_timestamp_us = start_timestamp_us.to(torch.int64)
            end_timestamp_us = end_timestamp_us.to(torch.int64)

        # Convert the start and end rotation matrix to quaternions
        R_sensor_world_s_quat = so3_matrix_to_quat(T_sensor_world_start[None, :3, :3])  # [1, 4]
        R_sensor_world_e_quat = so3_matrix_to_quat(T_sensor_world_end[None, :3, :3])  # [1, 4]

        # Reuse provided sensor rays
        assert len(sensor_rays.shape) == 2
        assert len(sensor_rays) == len(elements)
        assert sensor_rays.shape[1] == 3
        assert sensor_rays.dtype == lidar_model.dtype

        # Get relative frame-times based on the elements column index relative to the total number of columns
        # (columns are measured in increasing time order irrespective of spin-direction)
        t = elements[:, 1].to(lidar_model.dtype) / (lidar_model.n_columns - 1)

        world_position_rs = (1 - t)[..., None] * T_sensor_world_start[:3, 3:4].transpose(0, 1).repeat(
            t.shape[0], 1
        ) + t[..., None] * T_sensor_world_end[:3, 3:4].transpose(0, 1).repeat(t.shape[0], 1)  # [n_elements, 3]

        R_sensor_world_rs = quat_to_so3_matrix(
            quat_slerp(R_sensor_world_s_quat.repeat(t.shape[0], 1), R_sensor_world_e_quat.repeat(t.shape[0], 1), t)
        )  # [n_elements, 3, 3]

        world_ray_directions_rs = torch.bmm(R_sensor_world_rs, sensor_rays[:, :, None]).squeeze(-1)  # [n_elements, 3]

        # Copy the values in the output variable
        return_var = RowOffsetStructuredSpinningLidarModel.WorldRaysReturn(
            world_rays=torch.cat(tensors=(world_position_rs, world_ray_directions_rs), dim=1)  # [n_elements, 6])
        )

        if return_T_sensor_worlds:
            return_var.T_sensor_worlds = torch.zeros((len(sensor_rays), 4, 4), dtype=lidar_model.dtype, device=device)
            return_var.T_sensor_worlds[:, :3, :3] = R_sensor_world_rs
            return_var.T_sensor_worlds[:, :3, 3] = world_position_rs
            return_var.T_sensor_worlds[:, 3, 3] = 1

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            return_var.timestamps_us = (
                start_timestamp_us + (t[..., None] * (end_timestamp_us - start_timestamp_us)).to(torch.int64)
            ).squeeze(-1)  # [n_elements]

        return return_var
