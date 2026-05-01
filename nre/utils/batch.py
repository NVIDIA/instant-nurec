# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import queue

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List, Literal, Optional, Self, Sequence, Tuple, TypeAlias, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from libs.geometry.kernels.pose import se3pose_from_matrix
from libs.sensors.kernels.cameras import image_points_to_world_rays_shutter_pose
from ncore.data import (
    ConcreteCameraModelParametersUnion,
    ConcreteLidarModelParametersUnion,
)
from ncore.impl.common.transformations import PoseInterpolator
from ncore.sensors import (
    CameraModel,
    FThetaCameraModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
)
from nre.utils.geometry import tquat_to_se3_matrix
from nre.utils.misc import assert_same_type, collate_fn, to_torch, unpack_optional
from nre.utils.prober import get_global_prober
from nre.utils.profiling import ScopedTimer
from nre.utils.sensors import (
    RectSubsampledSensor,
    SensorModelComputations,
)
from nre.utils.sensors.ncore_sensors_converters import (
    CameraModelConverter,
    DynamicPose,
    Pose,
)
from nre.utils.types import (
    CuboidTracksDataPack,
    RayFlags,
    RigTrajectories,
)


ConcreteCameraModelsUnion: TypeAlias = FThetaCameraModel | OpenCVFisheyeCameraModel | OpenCVPinholeCameraModel
ConcreteSensorModelParametersUnion: TypeAlias = ConcreteCameraModelParametersUnion | ConcreteLidarModelParametersUnion


@dataclass(kw_only=True, slots=True)
class RectSubsampled(RectSubsampledSensor):
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

    def to_json(self) -> dict:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "width": self.width,
            "height": self.height,
            "i": self.i,
            "j": self.j,
            "subsample_factor": self.subsample_factor,
        }

    def __hash__(self) -> int:
        return hash(
            (self.original_width, self.original_height, self.width, self.height, self.i, self.j, self.subsample_factor)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RectSubsampled):
            return False
        return self.to_json() == other.to_json()


@torch.autocast(device_type="cuda", enabled=False)
def compute_pixel_footprint(camera_rays: torch.Tensor | np.ndarray, pixel_offset: int = 10):
    """Computes the cone footprint of a pixel at unit length (diameter of a cone
    carved by the pixel). Predict-only standalone keeps just the "cone" mode;
    "plane_intersect" had no callers.
    """

    assert pixel_offset != 0, "pixel_offset must not be zero to avoid division by zero in footprint calculation"

    is_input_numpy = isinstance(camera_rays, np.ndarray)
    if is_input_numpy:
        camera_rays = to_torch(cast(npt.NDArray, camera_rays), device="cpu", dtype=torch.float32)
    assert isinstance(camera_rays, torch.Tensor)

    # Convert to float32 on CUDA unconditionally to avoid mixed precision issues
    if camera_rays.device.type == "cuda":
        camera_rays = camera_rays.float()

    # Normalize camera rays to unit length
    camera_rays = F.normalize(camera_rays, dim=-1)

    # Bottom right corner
    dot_product = torch.einsum(
        "ijk, ijk -> ij", camera_rays[pixel_offset:, pixel_offset:], camera_rays[:-pixel_offset, :-pixel_offset]
    )
    # Clamp between [-1, 1] to avoid NaN in arccos.
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    solid_angle_bottom_right = torch.arccos(
        F.pad(dot_product.unsqueeze(0), (0, pixel_offset, 0, pixel_offset), mode="replicate").squeeze(0)
    )

    # Top right corner
    dot_product = torch.einsum(
        "ijk, ijk -> ij",
        camera_rays[pixel_offset:, :-pixel_offset],
        camera_rays[:-pixel_offset, pixel_offset:],
    )
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    solid_angle_top_right = torch.arccos(
        F.pad(dot_product.unsqueeze(0), (0, pixel_offset, pixel_offset, 0), mode="replicate").squeeze(0)
    )

    solid_angle = torch.mean(
        torch.cat([solid_angle_bottom_right[:, :, None], solid_angle_top_right[:, :, None]], dim=-1), dim=-1
    )

    # Some angles can be zero due to precision. Replace with a small value to
    # avoid NaNs through the exp/log activation chain.
    solid_angle.masked_fill_(solid_angle == 0, 1e-12)

    footprint = 2 * torch.tan(0.5 * solid_angle) / pixel_offset

    return footprint.numpy() if is_input_numpy else footprint


def generate_grid_2d_indices(
    resolution: Tuple[int, int], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Computes (x, y) pixel coordinates for all pixels in the sensor frame.

    Args:
        resolution: (w, h) sensor width and height.

    Returns:
        torch.Tensor: A (N, 2) tensor with N = width * height, zero-based
            indices x in [0, w-1], y in [0, h-1]. Predict only ever needs
            "xy" order; the "yx" branch was dropped.
    """
    w, h = resolution
    sensor_pixels_x, sensor_pixels_y = torch.meshgrid(
        torch.arange(w, dtype=torch.int16, device=device),
        torch.arange(h, dtype=torch.int16, device=device),
        indexing="xy",
    )
    return torch.stack([sensor_pixels_x.flatten(), sensor_pixels_y.flatten()], dim=1)


@dataclass(kw_only=True, slots=True)
class RenderingData:
    """Data for rendering. Shared between camera and lidar.

    Note: The `rays`, `poses_tquat_startend` and the underlying scene representation (e.g. 3D Gaussians)
    should be in the same coordinate system. For example, the most common case is that it carries `rays`
    in the NRE space, the `poses_tquat_startend` is the transform from sensor to NRE space, and the underlying
    scene representation (e.g. 3D Gaussians) is also in the NRE space.

    The fields are:

    - rays: Ray origins and directions for the camera or lidar. [Tensor[float32]]. (B, height, width, 6).
    - sensor_model_parameters: List of model parameters for the camera or lidar. List of length B.
    - poses_tquat_startend: Start and end poses of the frame. [Tensor[float32]]. (B, 2, 7)
    - timestamps_startend_us: Start and end timestamps of the frame in microseconds. [Tensor[int64]]. (B, 2)
    """

    rays: torch.Tensor
    sensor_model_parameters: list[ConcreteSensorModelParametersUnion]
    poses_tquat_startend: torch.Tensor  # (B, 2, 7)
    timestamps_startend_us: torch.Tensor  # (B, 2) - kept on GPU for GPU operations
    rays_timestamps_us: torch.Tensor | None = None  # (B, height, width, 1)
    _rays_footprints: torch.Tensor | None = None  # (B, height, width, 1)
    timestamps_startend_us_cpu: torch.Tensor  # (B, 2) - cpu copy to avoid .item() calls
    _distance_to_depth_scale: torch.Tensor | None = None  # [Tensor[float32]] (B, height, width, 1)
    _rays_is_sky: torch.Tensor | None = (
        None  # [Tensor[bool]] (B, height, width, 1) — sky rays mask for gradient detachment
    )

    def __post_init__(self):
        B = self.rays.shape[0]
        assert self.rays.ndim == 4 and self.rays.shape[3] == 6, "Rays must be a 4D tensor (B, height, width, 6)"
        assert len(self.sensor_model_parameters) == B, "Model parameters must be a list of length B"
        assert self.poses_tquat_startend.shape == (B, 2, 7), "Poses must be a 3D tensor (B, 2, 7)"
        assert self.timestamps_startend_us.shape == (B, 2), "Timestamps must be a 2D tensor (B, 2)"
        assert self.timestamps_startend_us_cpu.shape == (B, 2), "CPU timestamps must be a 2D tensor (B, 2)"
        if self.rays_timestamps_us is not None:
            assert self.rays_timestamps_us.ndim == 4 and self.rays_timestamps_us.shape[3] == 1, (
                f"Rays timestamps must be a 4D tensor (B, height, width, 1), but got {self.rays_timestamps_us.shape}"
            )
        if self._rays_footprints is not None:
            assert self._rays_footprints.ndim == 4 and self._rays_footprints.shape[3] == 1, (
                f"Rays footprints must be a 4D tensor (B, height, width, 1), but got {self._rays_footprints.shape}"
            )
        if self._distance_to_depth_scale is not None:
            assert (
                self._distance_to_depth_scale.ndim == 4
                and self._distance_to_depth_scale.shape[:3] == self.rays.shape[:3]
            ), (
                f"Depth to distance scale must be a 4D tensor (B, height, width, 1) and match the rays shape (B, height, width, 6), but got {self._distance_to_depth_scale.shape}"
            )

    @property
    def b(self) -> int:
        return self.rays.shape[0]

    @property
    def h(self) -> int:
        return self.rays.shape[1]

    @property
    def w(self) -> int:
        return self.rays.shape[2]

    @property
    @torch.autocast(device_type="cuda", enabled=False)
    def distance_to_depth_scale(self) -> torch.Tensor:
        """Compute the multiplication factor to convert depth to distance.
        Shape: (B, height, width, 1)"""
        if self._distance_to_depth_scale is None:
            scales: list[torch.Tensor] = []
            for bidx in range(self.b):
                sensor_model = CameraModel.from_parameters(
                    cast(ConcreteCameraModelParametersUnion, self.sensor_model_parameters[bidx]),
                    device=self.rays.device,
                )
                width, height = sensor_model.resolution.tolist()
                elements = generate_grid_2d_indices((width, height), device=self.rays.device)
                sensor_rays = sensor_model.pixels_to_camera_rays(elements).reshape(height, width, 3)
                scales.append(sensor_rays[..., 2].reshape(height, width))
            self._distance_to_depth_scale = torch.stack(scales, dim=0).unsqueeze(-1)
        return self._distance_to_depth_scale

    @property
    def ray_footprints(self) -> torch.Tensor:
        assert self._rays_footprints is not None, "Rays footprints must be set"
        return self._rays_footprints

    @property
    @torch.autocast(device_type="cuda", enabled=False)
    def uv_directions_frame_end(self) -> torch.Tensor:
        """Compute the UV directions for the pixels in the frame.
        Shape: (B, 2, 3), here the 2nd dimension is U and V directions, normalized"""
        frame_end_se3 = tquat_to_se3_matrix(self.poses_tquat_startend[:, 1], unbatch=False)  # (B, 4, 4)
        return frame_end_se3[:, :3, :2].transpose(1, 2)

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        if any(item.rays_timestamps_us is None for item in seq):
            rays_timestamps_us = None
        else:
            rays_timestamps_us = collate_fn([item.rays_timestamps_us for item in seq], device)
        if any(item._rays_footprints is None for item in seq):
            _rays_footprints = None
        else:
            _rays_footprints = collate_fn([item._rays_footprints for item in seq], device)
        # Keep GPU version on target device, CPU copy on CPU
        timestamps_startend_us = collate_fn([item.timestamps_startend_us for item in seq], device)
        timestamps_startend_us_cpu = collate_fn([item.timestamps_startend_us_cpu for item in seq], torch.device("cpu"))
        if any(item._distance_to_depth_scale is None for item in seq):
            _distance_to_depth_scale = None
        else:
            _distance_to_depth_scale = collate_fn([item._distance_to_depth_scale for item in seq], device)
        if any(item._rays_is_sky is None for item in seq):
            _rays_is_sky = None
        else:
            _rays_is_sky = collate_fn([item._rays_is_sky for item in seq], device)
        return cls(
            rays=collate_fn([item.rays for item in seq], device),
            sensor_model_parameters=[p for item in seq for p in item.sensor_model_parameters],
            poses_tquat_startend=collate_fn([item.poses_tquat_startend for item in seq], device),
            timestamps_startend_us=timestamps_startend_us,
            rays_timestamps_us=rays_timestamps_us,
            _rays_footprints=_rays_footprints,
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
            _distance_to_depth_scale=_distance_to_depth_scale,
            _rays_is_sky=_rays_is_sky,
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            rays=self.rays.to(*args, **kwargs),
            sensor_model_parameters=self.sensor_model_parameters,
            poses_tquat_startend=self.poses_tquat_startend.to(*args, **kwargs),
            timestamps_startend_us=self.timestamps_startend_us.to(*args, **kwargs),
            rays_timestamps_us=self.rays_timestamps_us.to(*args, **kwargs)
            if self.rays_timestamps_us is not None
            else None,
            _rays_footprints=self._rays_footprints.to(*args, **kwargs) if self._rays_footprints is not None else None,
            # CPU copy stays on CPU - don't move it
            timestamps_startend_us_cpu=self.timestamps_startend_us_cpu,
            _distance_to_depth_scale=self._distance_to_depth_scale.to(*args, **kwargs)
            if self._distance_to_depth_scale is not None
            else None,
            _rays_is_sky=self._rays_is_sky.to(*args, **kwargs) if self._rays_is_sky is not None else None,
        )

    def __getitem__(self, item: Union[int, slice]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            rays=self.rays[item],
            sensor_model_parameters=self.sensor_model_parameters[item],
            poses_tquat_startend=self.poses_tquat_startend[item],
            timestamps_startend_us=self.timestamps_startend_us[item],
            rays_timestamps_us=self.rays_timestamps_us[item] if self.rays_timestamps_us is not None else None,
            _rays_footprints=self._rays_footprints[item] if self._rays_footprints is not None else None,
            timestamps_startend_us_cpu=self.timestamps_startend_us_cpu[item],
            _distance_to_depth_scale=self._distance_to_depth_scale[item]
            if self._distance_to_depth_scale is not None
            else None,
            _rays_is_sky=self._rays_is_sky[item] if self._rays_is_sky is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class FrameMeta:
    """Metadata for a frame. Shared between camera and lidar.

    The fields are:
    - unique_sensor_idx: Index of the sensor that captured the frame. int32
    - unique_frame_idx: Unique index (among all sensors) for the frame. int32
    - subsample: Subsampled pixel region (crop and resize info). Default to None.
    """

    unique_sensor_idx: int
    unique_frame_idx: int
    subsample: RectSubsampled | None = None
    # [TODO]: temporary solution to align with the current design, should be improved in the future
    T_offset_nre_startend: torch.Tensor | None = None  # (2, 4, 4) offset in the NRE frame or None

    # Tensor version that will be used for cuda, populated automatically
    unique_frame_idx_tensor: torch.Tensor | None = None

    # Stringified version of the unique sensor index, used to index sensor_models as a nn.ModuleDict.
    unique_sensor_idx_str: str | None = None

    def __post_init__(self):
        if self.unique_frame_idx_tensor is None:
            self.unique_frame_idx_tensor = (
                torch.tensor([self.unique_frame_idx], dtype=torch.int32) if self.unique_frame_idx != -1 else None
            )
        if self.unique_sensor_idx_str is None:
            self.unique_sensor_idx_str = str(self.unique_sensor_idx)
        if self.T_offset_nre_startend is not None:
            assert self.T_offset_nre_startend.shape == (2, 4, 4), "T_offset_nre_startend must be a 3D tensor (2, 4, 4)"
            assert self.T_offset_nre_startend.dtype == torch.float32, "T_offset_nre_startend must be a float32 tensor"

    @property
    def h(self) -> int | None:
        if self.subsample is None:
            return None
        else:
            return self.subsample.height

    @property
    def w(self) -> int | None:
        if self.subsample is None:
            return None
        else:
            return self.subsample.width

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> List[Self]:
        return [s.to(device) for s in seq]

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            unique_sensor_idx=self.unique_sensor_idx,
            unique_frame_idx=self.unique_frame_idx,
            subsample=self.subsample.to(*args, **kwargs) if self.subsample is not None else None,
            T_offset_nre_startend=self.T_offset_nre_startend.to(*args, **kwargs)
            if self.T_offset_nre_startend is not None
            else None,
            unique_frame_idx_tensor=self.unique_frame_idx_tensor.to(*args, **kwargs)
            if self.unique_frame_idx_tensor is not None
            else None,
            unique_sensor_idx_str=self.unique_sensor_idx_str,
        )


@dataclass(kw_only=True, slots=True)
class CameraFrameLabels:
    """Labels for a camera frame.

    The fields are:
    - flags: Optional. Bitmask integer value (see RayFlags). Default is None. [Tensor[int32]]. (B, height, width, 1).
    - rgb: Optional. RGB value within [0, 1]. Default is None. [Tensor[float32]]. (B, height, width, 3).
    - distance: Optional. Metric ray-depth in NRE scale (not z-depth). Default is None. [Tensor[float32]]. (B, height, width, 1).
    - metric_distance: Optional. Non-metric ray depth from a depth estimation model. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - relative_distance: Optional. Non-metric ray depth from a depth estimation model. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - alpha: Optional. Transparency of the pixel valued in [0, 1]. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - semantic: Optional. Semantic class of the pixel. Default is None. [Tensor[uint8]]. (B, height, width, 1).
    - normals: Optional. Per-pixel normal vectors relative to the world frame. Default is None. [Tensor[float32]]. (B, height, width, 3).
    - velocity: Optional. Velocity vector of the pixel (unit is m/s). Default is None. [Tensor[float32]]. (B, height, width, 3/6).
    - _n_valid_rgb: Optional. Number of valid RGB channels as cache for n_valid_rgb property. Default is None. int32.
    - _n_valid: Optional. Number of valid pixels as cache for n_valid property. Default is None. int32.
    - _n_difixed: Optional. Number of Difixed as cache for n_difixed property. Default is None. int32.
    - _n_valid_bg: Optional. Number of valid pixels for background loss (excludes INVALID, DIFIXED, SYNTHETIC). Default is None. int32.
    """

    flags: torch.Tensor | None = None
    rgb: torch.Tensor | None = None
    metric_distance: torch.Tensor | None = None
    relative_distance: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    semantic: torch.Tensor | None = None
    normals: torch.Tensor | None = None
    velocity: torch.Tensor | None = None
    _n_valid_rgb: int | None = None
    _n_valid: int | None = None
    _n_difixed: int | None = None
    _n_valid_bg: int | None = None
    # TODO: add other labels from ExtraSignals

    def __post_init__(self):
        if self.flags is not None:
            assert self.flags.ndim == 4 and self.flags.shape[3] == 1, "Flags must be a 4D tensor (B, height, width, 1)"
            assert self.flags.dtype == torch.int32, "Flags must be a int32 tensor"
        if self.rgb is not None:
            assert self.rgb.ndim == 4 and self.rgb.shape[3] == 3, "RGB must be a 4D tensor (B, height, width, 3)"
            assert self.rgb.dtype == torch.float32, "RGB must be a float32 tensor"
        if self.metric_distance is not None:
            assert self.metric_distance.ndim == 4 and self.metric_distance.shape[3] == 1, (
                "Metric distance must be a 4D tensor (B, height, width, 1)"
            )
            assert self.metric_distance.dtype == torch.float32, "Metric distance must be a float32 tensor"
        if self.relative_distance is not None:
            assert self.relative_distance.ndim == 4 and self.relative_distance.shape[3] == 1, (
                "Relative distance must be a 4D tensor (B, height, width, 1)"
            )
            assert self.relative_distance.dtype == torch.float32, "Relative distance must be a float32 tensor"
        if self.alpha is not None:
            assert self.alpha.ndim == 4 and self.alpha.shape[3] == 1, "Alpha must be a 4D tensor (B, height, width, 1)"
            assert self.alpha.dtype == torch.float32, "Alpha must be a float32 tensor"
        if self.semantic is not None:
            assert self.semantic.ndim == 4 and self.semantic.shape[3] == 1, (
                "Semantic must be a 4D tensor (B, height, width, 1)"
            )
            assert self.semantic.dtype == torch.uint8, "Semantic must be a uint8 tensor"
        if self.normals is not None:
            assert self.normals.ndim == 4 and self.normals.shape[3] == 3, (
                "Normals must be a 4D tensor (B, height, width, 3)"
            )
            assert self.normals.dtype == torch.float32, "Normals must be a float32 tensor"
        if self.velocity is not None:
            assert self.velocity.ndim == 4 and self.velocity.shape[3] in [
                3,
                6,
            ], "Velocity must be a 4D tensor (B, height, width, 3/6)"
            assert self.velocity.dtype == torch.float32, "Velocity must be a float32 tensor"
        if self._n_valid_rgb is not None:
            assert isinstance(self._n_valid_rgb, int), "N valid RGB must be an int"
        if self._n_valid is not None:
            assert isinstance(self._n_valid, int), "N valid must be an int"
        if self._n_difixed is not None:
            assert isinstance(self._n_difixed, int), "N Difixed must be an int"
        if self._n_valid_bg is not None:
            assert isinstance(self._n_valid_bg, int), "N valid bg must be an int"

    @property
    def n_valid_rgb(self) -> int:
        if self._n_valid_rgb is None:
            rgb_valid_mask = self.get_mask_flags_all(RayFlags.RGB_LABEL) & self.get_mask_flags_none(RayFlags.INVALID)
            self._n_valid_rgb = int(rgb_valid_mask.sum().item()) * 3
        return self._n_valid_rgb

    @property
    def n_valid(self) -> int:
        if self._n_valid is None:
            valid_mask = self.get_mask_flags_none(RayFlags.INVALID)
            self._n_valid = int(valid_mask.sum().item())
        return self._n_valid

    @property
    def n_difixed(self) -> int:
        if self._n_difixed is None:
            difixed_mask = self.get_mask_flags_all(RayFlags.DIFIXED)
            self._n_difixed = int(difixed_mask.sum().item())
        return self._n_difixed

    @property
    def n_valid_bg(self) -> int:
        """Number of valid pixels for background loss (excludes INVALID, DIFIXED, SYNTHETIC)."""
        if self._n_valid_bg is None:
            valid_mask = (
                self.get_mask_flags_none(RayFlags.INVALID)
                & self.get_mask_flags_none(RayFlags.DIFIXED)
                & self.get_mask_flags_none(RayFlags.SYNTHETIC)
            )
            self._n_valid_bg = int(valid_mask.sum().item())
        return self._n_valid_bg

    @property
    def b(self) -> int | None:
        batch_sizes = [
            getattr(self, attr).shape[0]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(batch_sizes) == 0:
            return None
        else:
            assert all(b == batch_sizes[0] for b in batch_sizes), "All batch sizes must be the same"
            return batch_sizes[0]

    @property
    def h(self) -> int | None:
        heights = [
            getattr(self, attr).shape[1]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(heights) == 0:
            return None
        else:
            assert all(h == heights[0] for h in heights), "All heights must be the same"
            return heights[0]

    @property
    def w(self) -> int | None:
        widths = [
            getattr(self, attr).shape[2]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(widths) == 0:
            return None
        else:
            assert all(w == widths[0] for w in widths), "All widths must be the same"
            return widths[0]

    def get_mask_flags_all(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *all* flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).eq(flags.value)

    def get_mask_flags_none(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *none* of the flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).eq(0)

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        # For metric distance, if any is not None, we set the others to zeros
        metric_distance_seq = [item.metric_distance for item in seq]
        if (
            first_not_none_distance := next(
                (distance for distance in metric_distance_seq if distance is not None), None
            )
        ) is not None:
            metric_distance_seq = [
                torch.zeros_like(first_not_none_distance) if distance is None else distance
                for distance in metric_distance_seq
            ]

        return cls(
            flags=collate_fn([item.flags for item in seq], device),
            rgb=collate_fn([item.rgb for item in seq], device),
            metric_distance=collate_fn(metric_distance_seq, device),
            relative_distance=collate_fn([item.relative_distance for item in seq], device),
            alpha=collate_fn([item.alpha for item in seq], device),
            semantic=collate_fn([item.semantic for item in seq], device),
            normals=collate_fn([item.normals for item in seq], device),
            velocity=collate_fn([item.velocity for item in seq], device),
            _n_valid_rgb=sum([item.n_valid_rgb for item in seq]),
            _n_valid=sum([item.n_valid for item in seq]),
            _n_difixed=sum([item.n_difixed for item in seq]),
            _n_valid_bg=sum([item.n_valid_bg for item in seq]),
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            flags=self.flags.to(*args, **kwargs) if self.flags is not None else None,
            rgb=self.rgb.to(*args, **kwargs) if self.rgb is not None else None,
            metric_distance=self.metric_distance.to(*args, **kwargs) if self.metric_distance is not None else None,
            relative_distance=self.relative_distance.to(*args, **kwargs)
            if self.relative_distance is not None
            else None,
            alpha=self.alpha.to(*args, **kwargs) if self.alpha is not None else None,
            semantic=self.semantic.to(*args, **kwargs) if self.semantic is not None else None,
            normals=self.normals.to(*args, **kwargs) if self.normals is not None else None,
            velocity=self.velocity.to(*args, **kwargs) if self.velocity is not None else None,
            _n_valid_rgb=self._n_valid_rgb,
            _n_valid=self._n_valid,
            _n_difixed=self._n_difixed,
            _n_valid_bg=self._n_valid_bg,
        )

    def __getitem__(self, item: Union[int, slice, torch.Tensor]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            flags=self.flags[item] if self.flags is not None else None,
            rgb=self.rgb[item] if self.rgb is not None else None,
            metric_distance=self.metric_distance[item] if self.metric_distance is not None else None,
            relative_distance=self.relative_distance[item] if self.relative_distance is not None else None,
            alpha=self.alpha[item] if self.alpha is not None else None,
            semantic=self.semantic[item] if self.semantic is not None else None,
            normals=self.normals[item] if self.normals is not None else None,
            velocity=self.velocity[item] if self.velocity is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class LidarFrameLabels:
    """Labels for a lidar frame.

    The fields are:
    - flags: Optional. Bitmask integer value (see RayFlags). Default is None. [Tensor[int32]]. (B, height, width, 1).
    - distance: Optional. Metric ray-depth in NRE scale (not z-depth). Default is None. [Tensor[float32]]. (B, height, width, 1).
    - intensity: Optional. Intensity of the lidar response. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - raydrop: Optional. The possiblity that the ray should be dropped. Default is None. [Tensor[float32]]. (B, height, width, 1).
    - _n_valid_lidar: Optional. The number of valid lidar rays as cache for n_valid_lidar property. Default is None. int32.
    - sparse_rays: Optional. The rays from the lidar points. Default is None. [Tensor[float32]]. (B, n_sparse_rays, 6).
    - sparse_timestamps: Optional. The timestamps of the lidar points. Default is None. [Tensor[int64]]. (B, n_sparse_rays, 1).
    - sparse_elements: Optional. The elements (row, col) of the lidar points. Default is None. [Tensor[int64]]. (B, n_sparse_rays, 2).
    """

    flags: torch.Tensor | None = None
    distance: torch.Tensor | None = None
    intensity: torch.Tensor | None = None
    raydrop: torch.Tensor | None = None
    _n_valid_lidar: int | None = None
    # TODO: add other labels from ExtraSignals

    # To support cases where rays are computed from the lidar points.
    sparse_rays: torch.Tensor | None = None
    sparse_timestamps: torch.Tensor | None = None
    sparse_elements: torch.Tensor | None = None

    def __post_init__(self):
        if self.flags is not None:
            assert self.flags.ndim == 4 and self.flags.shape[3] == 1, "Flags must be a 4D tensor (B, height, width, 1)"
            assert self.flags.dtype == torch.int32, "Flags must be a int32 tensor"
        if self.distance is not None:
            assert self.distance.ndim == 4 and self.distance.shape[3] == 1, (
                "Distance must be a 4D tensor (B, height, width, 1)"
            )
            assert self.distance.dtype == torch.float32, "Distance must be a float32 tensor"
        if self.intensity is not None:
            assert self.intensity.ndim == 4 and self.intensity.shape[3] == 1, (
                "Intensity must be a 4D tensor (B, height, width, 1)"
            )
            assert self.intensity.dtype == torch.float32, "Intensity must be a float32 tensor"
        if self.raydrop is not None:
            assert self.raydrop.ndim == 4 and self.raydrop.shape[3] == 1, (
                "Raydrop must be a 4D tensor (B, height, width, 1)"
            )
            assert self.raydrop.dtype == torch.float32, "Raydrop must be a float32 tensor"
        if self._n_valid_lidar is not None:
            assert isinstance(self._n_valid_lidar, int), "N valid must be an int"
        if self.sparse_rays is not None:
            assert self.sparse_rays.ndim == 3 and self.sparse_rays.shape[2] == 6, (
                "Rays must be a 4D tensor (B, n_sparse_rays, 6)"
            )
            assert self.sparse_rays.dtype == torch.float32, "Rays must be a float32 tensor"
        if self.sparse_timestamps is not None:
            assert self.sparse_timestamps.ndim == 3 and self.sparse_timestamps.shape[2] == 1, (
                "Timestamps must be a 4D tensor (B, n_sparse_rays, 1)"
            )
            assert self.sparse_timestamps.dtype == torch.int64, "Timestamps must be a int64 tensor"
        if self.sparse_elements is not None:
            assert self.sparse_elements.ndim == 3 and self.sparse_elements.shape[2] == 2, (
                "Elements must be a 4D tensor (B, n_sparse_rays, 2)"
            )
            assert self.sparse_elements.dtype == torch.int64, "Elements must be a int64 tensor"

    @property
    def n_valid_lidar(self) -> int:
        if self._n_valid_lidar is None:
            valid_lidar_mask = self.get_mask_flags_none(RayFlags.INVALID) & self.get_mask_flags_none(RayFlags.DROPPED)
            self._n_valid_lidar = int(valid_lidar_mask.sum().item())
        return self._n_valid_lidar

    @property
    def b(self) -> int | None:
        batch_sizes = [
            getattr(self, attr).shape[0]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(batch_sizes) == 0:
            return None
        else:
            assert all(b == batch_sizes[0] for b in batch_sizes), "All batch sizes must be the same"
            return batch_sizes[0]

    @property
    def h(self) -> int | None:
        heights = [
            getattr(self, attr).shape[1]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(heights) == 0:
            return None
        else:
            assert all(h == heights[0] for h in heights), "All heights must be the same"
            return heights[0]

    @property
    def w(self) -> int | None:
        widths = [
            getattr(self, attr).shape[2]
            for attr in self.__dataclass_fields__
            if isinstance(getattr(self, attr), torch.Tensor)
        ]
        if len(widths) == 0:
            return None
        else:
            assert all(w == widths[0] for w in widths), "All widths must be the same"
            return widths[0]

    def get_mask_flags_all(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *all* flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).eq(flags.value)

    def get_mask_flags_none(self, flags: RayFlags) -> torch.Tensor:
        """Mask indicating the rays that have *none* of the flag bits of 'flags' set"""
        assert self.flags is not None, "flags are required"
        return torch.bitwise_and(self.flags, flags.value).eq(0)

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        return cls(
            flags=collate_fn([item.flags for item in seq], device),
            distance=collate_fn([item.distance for item in seq], device),
            intensity=collate_fn([item.intensity for item in seq], device),
            raydrop=collate_fn([item.raydrop for item in seq], device),
            _n_valid_lidar=sum([item.n_valid_lidar for item in seq]),
            sparse_rays=collate_fn([item.sparse_rays for item in seq], device),
            sparse_timestamps=collate_fn([item.sparse_timestamps for item in seq], device),
            sparse_elements=collate_fn([item.sparse_elements for item in seq], device),
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            flags=self.flags.to(*args, **kwargs) if self.flags is not None else None,
            distance=self.distance.to(*args, **kwargs) if self.distance is not None else None,
            intensity=self.intensity.to(*args, **kwargs) if self.intensity is not None else None,
            raydrop=self.raydrop.to(*args, **kwargs) if self.raydrop is not None else None,
            _n_valid_lidar=self._n_valid_lidar,
            sparse_rays=self.sparse_rays.to(*args, **kwargs) if self.sparse_rays is not None else None,
            sparse_timestamps=self.sparse_timestamps.to(*args, **kwargs)
            if self.sparse_timestamps is not None
            else None,
            sparse_elements=self.sparse_elements.to(*args, **kwargs) if self.sparse_elements is not None else None,
        )

    def __getitem__(self, item: Union[int, slice, torch.Tensor]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            flags=self.flags[item] if self.flags is not None else None,
            distance=self.distance[item] if self.distance is not None else None,
            intensity=self.intensity[item] if self.intensity is not None else None,
            raydrop=self.raydrop[item] if self.raydrop is not None else None,
            sparse_rays=self.sparse_rays[item] if self.sparse_rays is not None else None,
            sparse_timestamps=self.sparse_timestamps[item] if self.sparse_timestamps is not None else None,
            sparse_elements=self.sparse_elements[item] if self.sparse_elements is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class DataBatch:
    """Data for both camera and lidar frames."""

    @dataclass(kw_only=True, slots=True)
    class Camera:
        """Data for a camera frame. Includes the frame meta and labels."""

        meta: List[FrameMeta]
        labels: CameraFrameLabels

        @property
        def b(self) -> int:
            return len(self.meta)

        @property
        def h(self) -> int | None:
            heights = [meta.h for meta in self.meta] + [self.labels.h]
            heights = [h for h in heights if h is not None]
            if len(heights) == 0:
                return None
            else:
                assert all(h == heights[0] for h in heights), "All heights must be the same"
                return heights[0]

        @property
        def w(self) -> int | None:
            widths = [meta.w for meta in self.meta] + [self.labels.w]
            widths = [w for w in widths if w is not None]
            if len(widths) == 0:
                return None
            else:
                assert all(w == widths[0] for w in widths), "All widths must be the same"
                return widths[0]

        @classmethod
        def collate_fn(
            cls,
            seq: List[DataBatch.Camera],
            device: torch.device = torch.device("cpu"),
        ) -> DataBatch.Camera:
            return cls(
                meta=FrameMeta.collate_fn([meta for item in seq for meta in item.meta], device),
                labels=CameraFrameLabels.collate_fn([item.labels for item in seq], device),
            )

        def to(self, *args, **kwargs) -> Self:
            return self.__class__(
                meta=[meta.to(*args, **kwargs) for meta in self.meta],
                labels=self.labels.to(*args, **kwargs),
            )

        def __getitem__(self, item: Union[int, slice]) -> Self:
            """Allows indexing into the dataclass to get a subset of the data."""
            if isinstance(item, int):
                item = slice(item, item + 1)

            return self.__class__(meta=self.meta[item], labels=self.labels[item])

    @dataclass(kw_only=True, slots=True)
    class Lidar:
        """Data for a lidar frame. Includes the frame meta and labels."""

        meta: List[FrameMeta]
        labels: LidarFrameLabels

        @property
        def b(self) -> int:
            return len(self.meta)

        @property
        def h(self) -> int | None:
            heights = [meta.h for meta in self.meta] + [self.labels.h]
            heights = [h for h in heights if h is not None]
            if len(heights) == 0:
                return None
            else:
                assert all(h == heights[0] for h in heights), "All heights must be the same"
                return heights[0]

        @property
        def w(self) -> int | None:
            widths = [meta.w for meta in self.meta] + [self.labels.w]
            widths = [w for w in widths if w is not None]
            if len(widths) == 0:
                return None
            else:
                assert all(w == widths[0] for w in widths), "All widths must be the same"
                return widths[0]

        @classmethod
        def collate_fn(
            cls,
            seq: List[DataBatch.Lidar],
            device: torch.device = torch.device("cpu"),
        ) -> DataBatch.Lidar:
            return cls(
                meta=FrameMeta.collate_fn([meta for item in seq for meta in item.meta], device),
                labels=LidarFrameLabels.collate_fn([item.labels for item in seq], device),
            )

        def to(self, *args, **kwargs) -> Self:
            return self.__class__(
                meta=[meta.to(*args, **kwargs) for meta in self.meta],
                labels=self.labels.to(*args, **kwargs),
            )

        def __getitem__(self, item: Union[int, slice]) -> Self:
            """Allows indexing into the dataclass to get a subset of the data."""
            if isinstance(item, int):
                item = slice(item, item + 1)

            return self.__class__(
                meta=self.meta[item],
                labels=self.labels[item],
            )

    idx: int | None = None
    worker_id: List[int] | None = None
    sequence_id: List[str] | None = None

    camera: Camera | None = None
    lidar: Lidar | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[DataBatch],
        device: torch.device = torch.device("cpu"),
    ) -> DataBatch:
        assert all(item.idx == seq[0].idx for item in seq), "All items must have the same idx"

        if any(item.worker_id is None for item in seq):
            worker_id = None
        else:
            worker_id = [worker_id for item in seq for worker_id in unpack_optional(item.worker_id)]

        if any(item.sequence_id is None for item in seq):
            sequence_id = None
        else:
            sequence_id = [sequence_id for item in seq for sequence_id in unpack_optional(item.sequence_id)]

        if any(item.camera is None for item in seq):
            camera = None
        else:
            camera = DataBatch.Camera.collate_fn([unpack_optional(item.camera) for item in seq], device)

        if any(item.lidar is None for item in seq):
            lidar = None
        else:
            lidar = DataBatch.Lidar.collate_fn([unpack_optional(item.lidar) for item in seq], device)

        return cls(
            idx=seq[0].idx,
            worker_id=worker_id,
            sequence_id=sequence_id,
            camera=camera,
            lidar=lidar,
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            idx=self.idx,
            worker_id=self.worker_id,
            sequence_id=self.sequence_id,
            camera=self.camera.to(*args, **kwargs) if self.camera is not None else None,
            lidar=self.lidar.to(*args, **kwargs) if self.lidar is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class RenderingBatch:
    """
    A RenderingBatch is a collection of RenderingData for camera and lidar.
    """

    camera: RenderingData | None = None
    lidar: RenderingData | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[RenderingBatch],
        device: torch.device = torch.device("cpu"),
    ) -> RenderingBatch:
        if any(item.camera is None for item in seq):
            camera = None
        else:
            camera = RenderingData.collate_fn([unpack_optional(item.camera) for item in seq], device)

        if any(item.lidar is None for item in seq):
            lidar = None
        else:
            lidar = RenderingData.collate_fn([unpack_optional(item.lidar) for item in seq], device)

        return cls(camera=camera, lidar=lidar)

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            camera=self.camera.to(*args, **kwargs) if self.camera is not None else None,
            lidar=self.lidar.to(*args, **kwargs) if self.lidar is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class DataAndRenderingBatch:
    """
    A DataAndRenderingBatch is a compounding type used for training and validation.

    The fields are:
    - data: The information from the dataset that are required for training and validation. [DataBatch]
    - rendering: The information that will be fed into the renderer. [RenderingBatch]

    Note the design of the workflow for training and validation is:

    1. None -> [dataloader] -> DataAndRenderingBatch(data: DataBatch, rendering: RenderingBatch | None)
    2. DataAndRenderingBatch -> [pre-processing] -> DataAndRenderingBatch(data: DataBatch, rendering: RenderingBatch)
    3. RenderingBatch  -> [renderer] -> RenderingOutput
    4. (DataAndRenderingBatch, RenderingOutput) -> [loss] -> loss

    For inference, the workflow is:

    rendering: RenderingBatch -> [renderer] -> RenderingOutput
    """

    # DataBatch is always provided by the dataloader, whereas the RenderingBatch is optional.
    # In some cases we would like to compute the rendering data in the dataloader.
    # For example, for the Difix sampler, or to hide the latency when not performing camera pose optimization.
    data: DataBatch
    rendering: RenderingBatch | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[DataAndRenderingBatch],
        device: torch.device = torch.device("cpu"),
    ) -> DataAndRenderingBatch:
        if all(seq_isnone := [item.rendering is None for item in seq]):
            rendering = None
        elif not any(seq_isnone):
            rendering = RenderingBatch.collate_fn(cast(list[RenderingBatch], [item.rendering for item in seq]), device)
        else:
            raise ValueError("DataAndRenderingBatch.collate_fn: All items must have either a rendering or no rendering")

        return cls(
            data=DataBatch.collate_fn([item.data for item in seq], device),
            rendering=rendering,
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            data=self.data.to(*args, **kwargs),
            rendering=self.rendering.to(*args, **kwargs) if self.rendering is not None else None,
        )

    def pin_memory(self):
        """
        Enable pinned memory for async data transfer in PyTorch.

        When using a DataLoader with `pin_memory=True`, PyTorch calls
        `pin_memory()` on each object it retrieves from the dataset.
        Implementing this method ensures that the returned object is moved
        into pinned (page-locked) memory.
        """
        q = queue.Queue()
        q.put(self)
        while not q.empty():
            dataclass_obj = q.get()
            for field in fields(dataclass_obj):
                f = getattr(dataclass_obj, field.name)
                pin_memory_attr = getattr(f, "pin_memory", None)
                if callable(pin_memory_attr):
                    setattr(dataclass_obj, field.name, pin_memory_attr())
                elif is_dataclass(f):
                    q.put(f)
        return self


# Note(ruilong): temporarily place this class here because NCORETrainDataset depends on it.
# Should move it to nre.models.view_geometry after migration.
class CameraFreePoseViewGeometry(torch.nn.Module):
    """
    FreePoseViewGeometry for camera sensors. It stores raw (un-subsampled) camera extrinsics and intrinsics for all frames & views. [Exist for the new batch format]
    """

    def __init__(
        self,
        T_sensor_world_startend_allviews: torch.Tensor,  # (n_frames, 2, 4, 4)
        timestamps_startend_us_allviews: torch.Tensor,  # (n_frames, 2)
        sensor_models: dict[str, CameraModel],  # mapping from unique_sensor_idx to CameraModel
        # Maps a sensor id to a range of unique frame indices that can be used to recover the slices of
        # T_sensor_world_startend_allviews and timestamps_startend_us_allviews belonging to a specific sensor.
        sensor_ids_to_frame_range: dict[str, range],
        enable_calib: bool = False,
    ):
        super().__init__()
        assert T_sensor_world_startend_allviews.shape[0] == timestamps_startend_us_allviews.shape[0], (
            "T_sensor_world_startend_allviews and timestamps_startend_us_allviews must have the same number of frames"
        )
        self.T_sensor_world_startend_allviews = torch.nn.Buffer(T_sensor_world_startend_allviews, persistent=False)
        self.timestamps_startend_us_allviews = torch.nn.Buffer(timestamps_startend_us_allviews, persistent=False)
        self.timestamps_startend_us_allviews_cpu = timestamps_startend_us_allviews.clone().cpu()
        self.sensor_models = torch.nn.ModuleDict(sensor_models)
        self.sensor_ids_to_frame_range = sensor_ids_to_frame_range

        # When `cache_sensor_params` is True, cache per-sensor subsampled data for `to_rendering_data`:
        # footprints, ncore `parameters`, and `sensorlib_parameters` from `CameraModelConverter`
        # (world rays use these with `image_points_to_world_rays_shutter_pose`).
        # `rect_subsampled` is stored to invalidate the cache when subsampling changes.
        self.cached_sensor_params: dict[str, dict] = {
            k: {
                "rect_subsampled": None,
                "footprints": None,
                "parameters": None,
                "sensorlib_parameters": None,
            }
            for k in sensor_models.keys()
        }

        self.enable_calib = enable_calib
        self.embeds: torch.nn.Embedding | None = None
        if enable_calib:
            # Delta positions (3D) + Delta rotations (6D) = 9-D embedding space.
            # Setup a single delta transformation per camera frame to be optimized in 9-D space.
            # Any 9-D vector in the embedding space corresponds to an SE(3) transformation.
            # Each SE(3) transformation will be applied to both the frame start and end poses of the same frame.
            self.embeds = torch.nn.Embedding(T_sensor_world_startend_allviews.shape[0], 9)
            torch.nn.init.zeros_(self.embeds.weight)

        self.cached_sensor_subsample: dict[tuple[Optional[RectSubsampled], int], ConcreteCameraModelsUnion] = {}

    @staticmethod
    def from_rig_trajectories(
        rig_trajectories: RigTrajectories, enable_calib: bool = False, interp_with_rig: bool = True
    ) -> CameraFreePoseViewGeometry:
        """
        Initialize a `CameraFreePoseViewGeometry` from a `NCOREDataSource`.
        """
        camera_calibrations = rig_trajectories.camera_calibrations
        world_to_nre = rig_trajectories.world_to_nre

        # collect all extrinsics and intrinsics
        T_sensor_world_startend_allviews = []
        timestamps_startend_us_allviews = []
        sensor_models: dict[str, CameraModel] = {}
        sensor_ids_to_frame_range: dict[str, range] = {}  # camera_id -> range of unique frame indices
        unique_frame_start_index = 0

        for sensor_id, camera_calibration in camera_calibrations.items():
            unique_sensor_idx = camera_calibration.unique_sensor_idx if camera_calibration.unique_sensor_idx >= 0 else 0

            sensor_model = CameraModel.from_parameters(
                camera_calibration.camera_model_parameters, device=torch.device("cpu"), dtype=torch.float32
            )
            sensor_models[str(unique_sensor_idx)] = sensor_model

            candidate_trajectories = [
                r for r in rig_trajectories.rig_trajectories if r.sequence_id == camera_calibration.sequence_id
            ]
            assert len(candidate_trajectories) == 1, (
                f"Expected exactly one rig trajectory to match the sequence with name {camera_calibration.sequence_id}"
            )
            rig_trajectory = candidate_trajectories[0]

            if interp_with_rig:
                pose_interpolator = PoseInterpolator(
                    rig_trajectory.T_rig_worlds.cpu(), rig_trajectory.T_rig_world_timestamps_us.cpu()
                )
            elif rig_trajectory.cameras_frame_T_rig_worlds is not None:
                poses = rig_trajectory.cameras_frame_T_rig_worlds[sensor_id]
            else:
                raise ValueError("Expected cameras_frame_T_rig_worlds to be available when interp_with_rig is False")

            timestamps_us = rig_trajectory.cameras_frame_timestamps_us[sensor_id]
            assert timestamps_us.ndim == 2 and timestamps_us.shape[1] == 2, (
                "timestamps_us is expected to be a 2D tensor with shape (n_frames, 2)"
            )
            timestamps_startend_us_allviews.append(timestamps_us)

            T_sensor_rig_np = camera_calibration.T_sensor_rig.cpu().numpy()
            for frame_idx, timestamp_us in enumerate(timestamps_us):
                if interp_with_rig:
                    T_rig_world_startend = pose_interpolator.interpolate_to_timestamps(timestamp_us.cpu())
                else:
                    # directly fetch the per-frame pose
                    T_rig_world_startend = poses[frame_idx].numpy()
                T_sensor_world_startend = world_to_nre.transform_poses(T_rig_world_startend @ T_sensor_rig_np)
                T_sensor_world_startend_allviews.append(torch.from_numpy(T_sensor_world_startend).to(torch.float32))

            # Build map from sensor id to a range of unique frame indices that can be used to recover the slices of
            # T_sensor_world_startend_allviews and timestamps_startend_us_allviews belonging to a specific sensor.
            num_frames = timestamps_us.shape[0]
            sensor_ids_to_frame_range[sensor_id] = range(
                unique_frame_start_index, unique_frame_start_index + num_frames
            )
            unique_frame_start_index += num_frames

        return CameraFreePoseViewGeometry(
            T_sensor_world_startend_allviews=torch.stack(T_sensor_world_startend_allviews, dim=0),
            timestamps_startend_us_allviews=torch.cat(timestamps_startend_us_allviews, dim=0),
            sensor_models=sensor_models,
            sensor_ids_to_frame_range=sensor_ids_to_frame_range,
            enable_calib=enable_calib,
        )

    def get_timestamps(self, unique_frame_idx: int | None = None) -> torch.Tensor:
        """Get frame start/end timestamps of either a selected training frame or all training frames from all cameras.

        Args:
            unique_frame_idx: The index of a specfic camera frame or -1 or None to return all timestamps.

        Returns:
            A (2,) tensor containing the start and end timestamps of a frame if unique_frame_idx is provided,
            otherwise a tensor of shape (n_frames, 2) containing the start and end timestamp of each camera frame.
        """
        if unique_frame_idx is None or unique_frame_idx == -1:
            return self.timestamps_startend_us_allviews  # (n_frames, 2)
        else:
            return self.timestamps_startend_us_allviews[unique_frame_idx]  # (2,)

    def get_sensor_model(self, frame_meta: FrameMeta) -> CameraModel:
        """
        Getter to request sensor model for a given sensor index.

        Args:
            frame_meta: The frame metadata. See `FrameMeta` for more details.

        Returns:
            The sensor model in the subsampled domain.
        """
        subsample = frame_meta.subsample
        unique_sensor_idx = frame_meta.unique_sensor_idx if frame_meta.unique_sensor_idx >= 0 else 0
        sensor_model = cast(ConcreteCameraModelsUnion, self.sensor_models[str(unique_sensor_idx)])
        if subsample is not None:
            return subsample.apply_to_camera_model(sensor_model)
        else:
            return sensor_model

    def _compute_elements_and_sensor_rays(
        self, sensor_model: CameraModel, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the camera rays for a given frame.
        """
        width, height = sensor_model.resolution.tolist()
        elements = generate_grid_2d_indices((width, height), device=device)
        sensor_rays = sensor_model.pixels_to_camera_rays(elements)
        elements = elements.reshape(height, width, 2)
        sensor_rays = sensor_rays.reshape(height, width, 3)
        return elements, sensor_rays

    def get_poses_and_timestamps_startend(
        self,
        meta: FrameMeta,
        enable_calib: Optional[bool] = None,
        enable_torch_compile: bool = False,
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """
        Get the poses and timestamps for a given frame.
        """
        return SensorModelComputations.get_poses_and_timestamps_startend(
            subsample=meta.subsample,
            embeds=self.embeds,
            T_offset_nre_startend=meta.T_offset_nre_startend,
            T_sensor_world_startend_allviews=self.T_sensor_world_startend_allviews,
            timestamps_startend_us_allviews=self.timestamps_startend_us_allviews,
            timestamps_startend_us_allviews_cpu=self.timestamps_startend_us_allviews_cpu,
            sensor_models=self.sensor_models,
            unique_frame_idx=meta.unique_frame_idx,
            unique_frame_idx_tensor=meta.unique_frame_idx_tensor,
            unique_sensor_idx_str=unpack_optional(meta.unique_sensor_idx_str),
            enable_calib=enable_calib if enable_calib is not None else self.enable_calib,
            is_lidar=False,
            enable_torch_compile=enable_torch_compile,
        )

    @ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data")
    def to_rendering_data(
        self,
        data_batch: DataBatch.Camera,
        cache_sensor_params: bool = False,
        skip_calib: bool = False,
        global_step_for_prober: Optional[int] = None,
        enable_torch_compile: bool = False,
    ) -> RenderingData:
        """
        Convert a `DataBatch.Camera` to a `RenderingData`.

        This function will compute the followings and store them in the `RenderingData` object:
        * rays in world space via sensorlib (`sensorlib_parameters` from `CameraModelConverter` and `image_points_to_world_rays_shutter_pose`).
        * startend poses and timestamps for the given frame in the subsampled domain.
        * sensor model in the subsampled domain.
        """
        if data_batch.b == 1:
            return self._to_rendering_data_single_batch(
                data_batch,
                cache_sensor_params=cache_sensor_params,
                skip_calib=skip_calib,
                global_step_for_prober=global_step_for_prober,
                enable_torch_compile=enable_torch_compile,
            )
        else:
            rendering_data_list: list[RenderingData] = []
            for bidx in range(data_batch.b):
                rendering_data_list.append(
                    self._to_rendering_data_single_batch(
                        data_batch[bidx],
                        cache_sensor_params=cache_sensor_params,
                        skip_calib=skip_calib,
                        global_step_for_prober=global_step_for_prober,
                        enable_torch_compile=enable_torch_compile,
                    )
                )
            return RenderingData.collate_fn(rendering_data_list, device=rendering_data_list[0].rays.device)

    def _to_rendering_data_single_batch(
        self,
        data_batch: DataBatch.Camera,
        cache_sensor_params: bool = False,
        skip_calib: bool = False,
        global_step_for_prober: Optional[int] = None,
        enable_torch_compile: bool = False,
    ) -> RenderingData:
        """
        Internal single batch version of `to_rendering_data`.
        """
        assert data_batch.b == 1, "Only one frame is supported"
        meta = data_batch.meta[0]

        with ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data/get_poses_and_timestamps_startend"):
            enable_calib = self.enable_calib and not skip_calib

            # Mimic querying data stored in the rig module.
            pose_and_timestamps_startend_return = self.get_poses_and_timestamps_startend(
                meta, enable_calib, enable_torch_compile=enable_torch_compile
            )

            T_sensor_world_startend = pose_and_timestamps_startend_return.T_sensor_world_startend
            timestamps_startend_us = pose_and_timestamps_startend_return.timestamps_startend_us
            timestamps_startend_us_gpu = pose_and_timestamps_startend_return.timestamps_startend_us_gpu
            timestamps_startend_us_cpu = pose_and_timestamps_startend_return.timestamps_startend_us_cpu

        with ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data/cache_sensor_params"):
            subsample_unique_sensor_idx = (meta.subsample, meta.unique_sensor_idx)
            if subsample_unique_sensor_idx in self.cached_sensor_subsample:
                sensor_model = self.cached_sensor_subsample[subsample_unique_sensor_idx]
            else:
                sensor_model = cast(ConcreteCameraModelsUnion, self.get_sensor_model(meta))
                self.cached_sensor_subsample[subsample_unique_sensor_idx] = sensor_model

            if not cache_sensor_params:
                _, sensor_rays = self._compute_elements_and_sensor_rays(sensor_model, T_sensor_world_startend.device)
                # Note(ruilong): footprints are only required when AAA is on. Should be optional but here we always compute
                # it to be aligned with the old batch format.
                footprints = compute_pixel_footprint(sensor_rays)[..., None]  # (height, width, 1)
                sensor_model_parameters = sensor_model.get_parameters()
                sensorlib_parameters = CameraModelConverter.convert(sensor_model, device=T_sensor_world_startend.device)
            else:
                subsample = meta.subsample
                unique_sensor_idx = meta.unique_sensor_idx if meta.unique_sensor_idx >= 0 else 0
                cached_sensor_params = self.cached_sensor_params[str(unique_sensor_idx)]
                if (cached_sensor_params["rect_subsampled"] == subsample) and (
                    cached_sensor_params["sensorlib_parameters"] is not None
                ):
                    # Cache hit: subsampled footprints, ncore parameters, and sensorlib parameters.

                    footprints = cached_sensor_params["footprints"]
                    sensor_model_parameters = cached_sensor_params["parameters"]
                    sensorlib_parameters = cached_sensor_params["sensorlib_parameters"]
                else:
                    # Subsample changed or cache empty: recompute footprints and sensorlib parameters.
                    _, sensor_rays = self._compute_elements_and_sensor_rays(
                        sensor_model, T_sensor_world_startend.device
                    )
                    footprints = compute_pixel_footprint(sensor_rays)[..., None]  # (height, width, 1)
                    sensor_model_parameters = sensor_model.get_parameters()
                    sensorlib_parameters = CameraModelConverter.convert(
                        sensor_model, device=T_sensor_world_startend.device
                    )
                    self.cached_sensor_params[str(unique_sensor_idx)] = {
                        "rect_subsampled": subsample,
                        "footprints": footprints,
                        "parameters": sensor_model_parameters,
                        "sensorlib_parameters": sensorlib_parameters,
                    }

        with ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data/se3pose_from_matrix"):
            translations, rotations = se3pose_from_matrix(T_sensor_world_startend)
            poses_tquat_startend = torch.cat([translations, rotations], dim=1)
            poses_tquat_startend = poses_tquat_startend.unsqueeze(0)

        with ScopedTimer("CameraFreePoseViewGeometry/to_rendering_data/pixels_to_world_rays_shutter_pose"):
            timestamps_cpu = timestamps_startend_us_cpu.flatten()

            (world_rays, timestamps_us, _, _) = image_points_to_world_rays_shutter_pose(
                image_points=None,
                projection=sensorlib_parameters.projection,
                external_distortion=sensorlib_parameters.external_distortion,
                resolution=sensorlib_parameters.resolution,
                shutter_type=sensorlib_parameters.shutter_type,
                dynamic_pose=DynamicPose(
                    start_pose=Pose(translation=translations[0], rotation=rotations[0]),
                    end_pose=Pose(translation=translations[1], rotation=rotations[1]),
                ),
                start_timestamp_us=int(timestamps_cpu[0].item()),
                end_timestamp_us=int(timestamps_cpu[1].item()),
                return_timestamps=True,
            )

        if global_step_for_prober is not None:
            if (
                prober_result := get_global_prober()(
                    global_step_for_prober,
                    "sensor_to_world_rays_shutter_pose",
                    T_sensor_world_startend=T_sensor_world_startend,
                    timestamps_startend_us=timestamps_startend_us,
                    world_rays_grad=world_rays,
                )
            ) is not None:
                # Connect the gradient probing to the rest of the computation graph
                (world_rays,) = prober_result

        rays = unpack_optional(world_rays)  # camera rays are in nre space
        rays = rays.reshape(sensorlib_parameters.resolution[1], sensorlib_parameters.resolution[0], 6)
        timestamps = unpack_optional(timestamps_us)
        timestamps = timestamps.reshape(sensorlib_parameters.resolution[1], sensorlib_parameters.resolution[0], 1)

        return RenderingData(
            rays=rays.unsqueeze(0),
            # FIXME: This will be on cpu in numpy array.
            sensor_model_parameters=[sensor_model_parameters],
            poses_tquat_startend=poses_tquat_startend,
            timestamps_startend_us=timestamps_startend_us_gpu,
            rays_timestamps_us=timestamps.unsqueeze(0),
            _rays_footprints=footprints.unsqueeze(0),
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
        )


# Note(ruilong): temporarily place this class here because NCORETrainDataset depends on it.
# Should move it to nre.models.view_geometry after migration.

# Collate all attributes, *except* batch.idx.
# The input batch.idx's must be either be None or be the same index.
# The resulting batch.idx is this unique index, or None if all input indices are None.
def batch_collate_fn(
    item_or_seq: Union[DataBatch, DataAndRenderingBatch, Sequence[DataBatch], Sequence[DataAndRenderingBatch]],
    device: torch.device = torch.device("cpu"),
) -> DataBatch | DataAndRenderingBatch:
    # If it's a single batch, just return it
    if isinstance(item_or_seq, DataBatch | DataAndRenderingBatch):
        return item_or_seq

    if len(item_or_seq) == 1:
        return item_or_seq[0]

    # Short cuts for DataBatch and DataAndRenderingBatch
    if isinstance(item_or_seq[0], DataBatch):
        return DataBatch.collate_fn(cast(list[DataBatch], item_or_seq), device)
    elif isinstance(item_or_seq[0], DataAndRenderingBatch):
        return DataAndRenderingBatch.collate_fn(cast(list[DataAndRenderingBatch], item_or_seq), device)
    else:
        raise ValueError("Unsupported batch type")


@dataclass(slots=False, kw_only=True)
class NRMDataBatch:
    """
    A batch that contains (B,) groups of context and supervision DataBatch(es).
    We also precompute RenderingBatch altogether to hide latency for data preprocessing.

    Contains
        - context: list of context images (A DataAndRenderingBatch)
        - supervision: list of supervision images (A DataAndRenderingBatch)
        - cuboid_tracks: list of cuboid tracks
        - context_rig: list of rig trajectories for context (intrinsics subsampled already to context images)
        - supervision_rig: list of rig trajectories for supervision (intrinsics subsampled already to supervision images)
        - meta: list of dictionaries of metadata for each batch, e.g. sequence_id, ncore_json_path, etc.
    """

    context: list[DataAndRenderingBatch]
    supervision: list[DataAndRenderingBatch] | None = None
    cuboid_tracks: list[CuboidTracksDataPack] | None = None
    context_rig: list[RigTrajectories] | None = None
    supervision_rig: list[RigTrajectories] | None = None
    meta: list[dict[str, Any]] | None = None

    def __post_init__(self):
        if self.supervision is not None:
            assert len(self.context) == len(self.supervision), "Number of context and supervision batches must match"
        if self.cuboid_tracks is not None:
            assert len(self.context) == len(self.cuboid_tracks), "Number of context and cuboid tracks must match"
        if self.context_rig is not None:
            assert len(self.context) == len(self.context_rig), "Number of context and context_rig must match"
        if self.supervision_rig is not None:
            assert len(self.context) == len(self.supervision_rig), (
                "Number of supervision and supervision_rig must match"
            )

    def __getitem__(self, item: Union[int, slice]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            context=self.context[item],
            supervision=self.supervision[item] if self.supervision is not None else None,
            cuboid_tracks=self.cuboid_tracks[item] if self.cuboid_tracks is not None else None,
            context_rig=self.context_rig[item] if self.context_rig is not None else None,
            supervision_rig=self.supervision_rig[item] if self.supervision_rig is not None else None,
            meta=self.meta[item] if self.meta is not None else None,
        )

    def __len__(self) -> int:
        return len(self.context)

    def to(self, device: torch.device, **kwargs) -> Self:
        """Move all dataclass-aware fields to ``device``. Self-invented: NRE
        relies on Lightning's auto batch transfer; the standalone predict loop
        must move tensors explicitly."""

        def _move_list(items, attr: str | None = None):
            if items is None:
                return None
            out = []
            for item in items:
                if hasattr(item, "to_device"):
                    out.append(item.to_device(device))
                elif hasattr(item, "to"):
                    out.append(item.to(device, **kwargs))
                else:
                    out.append(item)
            return out

        return self.__class__(
            context=_move_list(self.context),
            supervision=_move_list(self.supervision),
            cuboid_tracks=_move_list(self.cuboid_tracks),
            context_rig=_move_list(self.context_rig),
            supervision_rig=_move_list(self.supervision_rig),
            meta=self.meta,
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def maybe_compute_rendering_data(self, device: torch.device):
        """Populates self.{context,supervision}[...].data.rendering unless already present"""

        for batch, rigs in ((self.context, self.context_rig), (self.supervision, self.supervision_rig)):
            if rigs is None or batch is None:
                continue
            for data, rig in zip(batch, rigs):
                # Do not re-compute if rendering data already exists.
                if data.rendering is not None:
                    continue
                # Note here that data.data.{camera,lidar} are supposed to be fp32 in their post-init checks.
                camera_rendering_data = (
                    CameraFreePoseViewGeometry.from_rig_trajectories(rig)
                    .to(device=device)
                    .to_rendering_data(data.data.camera.to(device), cache_sensor_params=True, enable_torch_compile=True)
                    if data.data.camera is not None
                    else None
                )
                lidar_rendering_data = (
                    LidarFreePoseViewGeometry.from_rig_trajectories(rig)
                    .to(device=device)
                    .to_rendering_data(data.data.lidar.to(device), cache_sensor_params=True, enable_torch_compile=True)
                    if data.data.lidar is not None
                    else None
                )
                data.rendering = RenderingBatch(camera=camera_rendering_data, lidar=lidar_rendering_data)

    @classmethod
    def collate_fn(
        cls,
        seq: Sequence[NRMDataBatch],
        device: torch.device = torch.device("cpu"),
        unsqueeze_if_zero_dim: bool = True,
    ) -> Self:
        T = TypeVar("T")

        def _collate_vals(*vals: list[T] | None) -> list[T] | None:
            assert len(vals) > 0, "At least one value must be provided"
            if any(val is None for val in vals):
                return None

            vals_arr = sum([unpack_optional(val) for val in vals], [])
            assert_same_type(vals_arr)

            for vi, v in enumerate(vals_arr):
                if isinstance(v, DataBatch):
                    vals_arr[vi] = cast(T, v.to(device))

                elif hasattr(v, "to_device"):
                    vals_arr[vi] = cast(T, v.to_device(device))

            return vals_arr

        return cls(
            context=unpack_optional(_collate_vals(*[batch.context for batch in seq])),
            supervision=_collate_vals(*[batch.supervision for batch in seq]),
            cuboid_tracks=_collate_vals(*[batch.cuboid_tracks for batch in seq]),
            context_rig=_collate_vals(*[batch.context_rig for batch in seq]),
            supervision_rig=_collate_vals(*[batch.supervision_rig for batch in seq]),
            meta=_collate_vals(*[batch.meta for batch in seq]),
        )
