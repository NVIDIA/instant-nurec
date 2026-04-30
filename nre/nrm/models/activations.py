# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

from dataclasses import dataclass
from typing import Self

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor

from nre.models.nn_extensions import module_call_type
from nre.nrm.config.models import GaussiansActivationConfig
from nre.utils.misc import unpack_optional


class OpacityActivation(nn.Module):
    """Activation function for opacity values using sigmoid with configurable shift."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        self.opacity_shift = config.opacity_shift

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation with shift to opacity values."""
        return torch.sigmoid(x + self.opacity_shift)


class ScaleActivation(nn.Module):
    """Activation function for scale values with exponential activation and clamping."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        self.scale_shift_log_ratio = config.scale_shift_log_ratio
        self.scale_max = config.scale_max
        self.scale_min = config.scale_min

        # Apply a shift so that the scale is scale_max * exp(scale_shift_log_ratio) when x is 0
        # This is adapted from GS-LRM where the scale_shift is 0.23 and the scale_max is 0.3
        # GS-LRM uses 0.23 because exp(0.23) = 0.1, i.e., scale_max / 3
        # When scale_shift_log_ratio = -1, the scale is scale_max / e when x is 0
        self._scale_shift = math.log(self.scale_max) + self.scale_shift_log_ratio

    def forward(self, x: Tensor, scene_rescale: float = 1.0, pixel_scale: Tensor | None = None) -> Tensor:
        """
        Apply exponential activation to scale values with clamping.
        Uses exponential activation to ensure positive scales, with maximum clamping
        to prevent numerical instability in 3D Gaussian splatting backpropagation.
        """
        scale = torch.exp(x + self._scale_shift)
        scale = scale.clamp(max=self.scale_max)
        # Default scale_min=0 is a no-op for exp(); use 0.01 with 3DGUT renderer to avoid NaN gradients.
        scale = scale.clamp(min=self.scale_min)
        scale = scale / scene_rescale
        return scale


class PixelScaleActivation(nn.Module):
    """Activation function for pixel scale values using exponential activation and clamping."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        self.pixel_scale_min = config.scale_min
        self.pixel_scale_max = config.scale_max

    def forward(self, x: Tensor, scene_rescale: float = 1.0, pixel_scale: Tensor | None = None) -> Tensor:
        scale = self.pixel_scale_min + (self.pixel_scale_max - self.pixel_scale_min) * x.sigmoid()
        assert pixel_scale is not None, "Pixel scale must be provided"
        return scale * pixel_scale


class RotationActivation(nn.Module):
    """Activation function for rotation quaternions using L2 normalization."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Normalize rotation quaternions to unit length."""
        return torch.nn.functional.normalize(x, dim=-1)


class DistanceActivation(nn.Module):
    """Activation function for distance values using sigmoid with min/max bounds."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        assert config.distance_type == "sigmoid", "Distance activation must be sigmoid"
        self.distance_min = config.distance_min
        self.distance_max = config.distance_max
        self.distance_shift = config.distance_shift

    def forward(self, x: Tensor, scene_rescale: float = 1.0) -> Tensor:
        """
        Apply sigmoid activation to map distance values to [distance_min, distance_max] range.
        The sigmoid ensures smooth mapping from unbounded input to bounded distance range.
        """
        # Apply a shift so that w is sigmoid(distance_shift) when x is 0
        # This makes training on object-centric data easier.
        # Default distance_shift = -1.65 is legacy from BTimer.
        # When distance_shift = -1.65, the initial depth is 0.16 * distance_max.
        w = torch.sigmoid(x + self.distance_shift)
        depth = self.distance_min + (self.distance_max - self.distance_min) * w
        depth = depth / scene_rescale
        return depth


class XyzActivation(nn.Module):
    """Activation function for direct XYZ coordinates using signed exponential with offset."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        assert config.xyz_type == "exp", "XYZ activation must be exp"
        self.z_offset = config.z_offset

        # Add a fixed offset to the XYZ coordinates
        # This is because the reference camera is at (0, 0, 0) facing +z
        # so we need a offset on z-axis to make the initial gs visible.
        self._z_offset_vec = nn.Buffer(torch.tensor([0.0, 0.0, self.z_offset]))

    def forward(self, x: Tensor, scene_rescale: float = 1.0) -> Tensor:
        """
        Apply signed exponential activation to XYZ coordinates.
        Uses sign(x) * (exp(|x|) - 1) to preserve sign while providing exponential growth,
        then adds a fixed offset and rescales for the scene coordinate system.
        """
        xyz = torch.sign(x) * (torch.expm1(torch.abs(x)))
        xyz = xyz + self._z_offset_vec
        xyz = xyz / scene_rescale
        return xyz


class RgbActivation(nn.Module):
    """Activation function for RGB color values using sigmoid."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation to map RGB values to [0, 1] range."""
        return torch.sigmoid(2 * x)


class SkyMaskActivation(nn.Module):
    """Activation function for sky mask values using clamped sigmoid."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        self.clamp_min = config.sky_mask_clamp_min
        self.clamp_max = config.sky_mask_clamp_max

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation with input clamping for numerical stability."""
        return torch.sigmoid(x.clamp(self.clamp_min, self.clamp_max))


class ForwardFlowActivation(nn.Module):
    """Activation function for forward flow values using linear scaling."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        self.scale = config.forward_flow_scale

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear scaling to forward flow values."""
        return x * self.scale


class FalloffSigmaActivation(nn.Module):
    """Activation function for falloff sigma values using sigmoid with min/max bounds."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        self.sigma_min = config.falloff_sigma_min
        self.sigma_max = config.falloff_sigma_max
        self.clamp_min = config.falloff_sigma_clamp_min
        self.clamp_max = config.falloff_sigma_clamp_max

    def activate(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x.clamp(self.clamp_min, self.clamp_max))

    def scale(
        self, unscaled_sigma: Tensor, time_span_s: float, frame_gap_timestamps_us: Tensor | None = None
    ) -> Tensor:
        assert len(unscaled_sigma.shape) == 4, "Unscaled sigma must not be flattened"
        time_min_s: float | torch.Tensor = self.sigma_min * time_span_s
        time_max_s: float = self.sigma_max * time_span_s
        if self.sigma_min < 0.0:
            # Infer from frame gap timestamps
            time_min_s = torch.max(unpack_optional(frame_gap_timestamps_us), dim=-1).values / 1e6
            # Divide by 2.0 to be extremely conservative
            # (so if exp factor is inf then gaussian will fall-off at middle of frame gap)
            # Now we divide by 1.0 to make the transition less noticeable.
            time_min_s = rearrange(time_min_s, "B -> B 1 1 1")
        # Linear scaling for now
        sigma = time_min_s + (time_max_s - time_min_s) * unscaled_sigma
        return sigma.flatten(0, 2)

    def forward(self, x: Tensor, time_span_s: float, frame_gap_timestamps_us: Tensor | None = None) -> Tensor:
        """
        Apply sigmoid activation to map falloff sigma values to [sigma_min, sigma_max] range.
        Input is clamped for numerical stability before sigmoid activation.
        """
        return self.scale(self.activate(x), time_span_s, frame_gap_timestamps_us)


@dataclass(kw_only=True, slots=True)
class GaussianParams:
    """
    Parameters for 3D Gaussian primitives. It could be either activated or not.
    All the gaussian attributes should have the same prefix shape (if not None),
    and the rest dimension should be matching the corresponding attributes.
    - rgb: (*, 3)
    - scale: (*, 3)
    - rotation: (*, 4)
    - opacity: (*, 1)
    These two are mutually exclusive:
        - xyz: (*, 3)
        - distance: (*, 1)
    """

    rgb: Tensor
    scale: Tensor
    rotation: Tensor
    opacity: Tensor
    xyz: Tensor | None = None
    distance: Tensor | None = None

    # Indicates whether this set of parameters has been already activated.
    activated: bool = False

    def __getitem__(self, key: torch.Tensor | slice | int) -> Self:
        return type(self)(
            rgb=self.rgb[key],
            scale=self.scale[key],
            rotation=self.rotation[key],
            opacity=self.opacity[key],
            xyz=self.xyz[key] if self.xyz is not None else None,
            distance=self.distance[key] if self.distance is not None else None,
            activated=self.activated,
        )

    def rearrange(self, pattern: str, **axes_lengths: int) -> Self:
        return type(self)(
            rgb=rearrange(self.rgb, pattern, **axes_lengths),
            scale=rearrange(self.scale, pattern, **axes_lengths),
            rotation=rearrange(self.rotation, pattern, **axes_lengths),
            opacity=rearrange(self.opacity, pattern, **axes_lengths),
            xyz=rearrange(self.xyz, pattern, **axes_lengths) if self.xyz is not None else None,
            distance=rearrange(self.distance, pattern, **axes_lengths) if self.distance is not None else None,
            activated=self.activated,
        )

    def flatten(self) -> Self:
        return type(self)(
            rgb=self.rgb.reshape(-1, 3),
            scale=self.scale.reshape(-1, 3),
            rotation=self.rotation.reshape(-1, 4),
            opacity=self.opacity.reshape(-1, 1),
            xyz=self.xyz.reshape(-1, 3) if self.xyz is not None else None,
            distance=self.distance.reshape(-1, 1) if self.distance is not None else None,
            activated=self.activated,
        )

    def __post_init__(self):
        prefix_shape = self.scale.shape[:-1]
        assert self.rgb.shape[:-1] == prefix_shape, "RGB shape must match prefix shape"
        assert self.rotation.shape[:-1] == prefix_shape, "Rotation shape must match prefix shape"
        assert self.opacity.shape[:-1] == prefix_shape, "Opacity shape must match prefix shape"
        if self.xyz is not None:
            assert self.xyz.shape[:-1] == prefix_shape, "XYZ shape must match prefix shape"
        if self.distance is not None:
            assert self.distance.shape[:-1] == prefix_shape, "Distance shape must match prefix shape"
        assert int(self.xyz is not None) + int(self.distance is not None) == 1, (
            "Exactly one of xyz or distance must be provided"
        )
        if self.activated:
            assert self.xyz is not None, "XYZ must be provided if Gaussian parameters are already activated"

    @property
    def prefix_shape(self) -> tuple[int, ...]:
        return self.rgb.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.rgb.device


class GaussianActivations(nn.Module):
    """Combined activation functions for Gaussian parameters."""

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        self.rgb = RgbActivation(config)
        self.scale = ScaleActivation(config) if config.scale_type == "world" else PixelScaleActivation(config)
        self.rotation = RotationActivation(config)
        self.opacity = OpacityActivation(config)
        self.xyz = XyzActivation(config) if config.xyz_type != "none" else nn.Identity()
        self.distance = DistanceActivation(config) if config.distance_type != "none" else nn.Identity()

    def forward(
        self,
        gs_params: GaussianParams,
        rays_o: Tensor | None = None,
        rays_d: Tensor | None = None,
        scene_rescale: float = 1.0,
    ) -> GaussianParams:
        assert not gs_params.activated, "Gaussian parameters must not be already activated"

        rgb = self.rgb(gs_params.rgb)
        scales = self.scale(gs_params.scale, scene_rescale)
        rotations = self.rotation(gs_params.rotation)
        opacity = self.opacity(gs_params.opacity)

        xyz: Tensor | None = None
        if gs_params.xyz is not None:
            xyz = self.xyz(gs_params.xyz, scene_rescale)

        elif gs_params.distance is not None:
            assert rays_o is not None and rays_d is not None, "Rays must be provided for distance-based prediction"
            distance = self.distance(gs_params.distance, scene_rescale)
            xyz = rays_o + rays_d * distance  # Compute XYZ from ray and distance

        return GaussianParams(rgb=rgb, scale=scales, rotation=rotations, opacity=opacity, xyz=xyz, activated=True)

    __call__ = module_call_type(forward)
