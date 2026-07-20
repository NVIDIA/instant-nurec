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

import math

from dataclasses import dataclass
from typing import Self

import torch
import torch.nn as nn

from torch import Tensor

from instant_nurec.config_schema.models import GaussiansActivationConfig


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

    def forward(self, x: Tensor, scene_rescale: float = 1.0) -> Tensor:
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


class RotationActivation(nn.Module):
    """Activation function for rotation quaternions using L2 normalization."""

    def forward(self, x: Tensor) -> Tensor:
        """Normalize rotation quaternions to unit length."""
        return torch.nn.functional.normalize(x, dim=-1)


class RgbActivation(nn.Module):
    """Activation function for RGB color values using sigmoid."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation to map RGB values to [0, 1] range."""
        return torch.sigmoid(2 * x)


@dataclass(kw_only=True, slots=True)
class GaussianParams:
    """
    Parameters for 3D Gaussian primitives. The decoder always emits
    activated parameters in the predict pipeline.
    All the gaussian attributes should have the same prefix shape (if not None),
    and the rest dimension should be matching the corresponding attributes.
    - rgb: (*, 3)
    - scale: (*, 3)
    - rotation: (*, 4)
    - opacity: (*, 1)
    - xyz: (*, 3)
    """

    rgb: Tensor
    scale: Tensor
    rotation: Tensor
    opacity: Tensor
    xyz: Tensor

    def __getitem__(self, key: torch.Tensor | slice | int) -> Self:
        return type(self)(
            rgb=self.rgb[key],
            scale=self.scale[key],
            rotation=self.rotation[key],
            opacity=self.opacity[key],
            xyz=self.xyz[key],
        )

    def flatten(self) -> Self:
        return type(self)(
            rgb=self.rgb.reshape(-1, 3),
            scale=self.scale.reshape(-1, 3),
            rotation=self.rotation.reshape(-1, 4),
            opacity=self.opacity.reshape(-1, 1),
            xyz=self.xyz.reshape(-1, 3),
        )

    def __post_init__(self):
        prefix_shape = self.scale.shape[:-1]
        assert self.rgb.shape[:-1] == prefix_shape, "RGB shape must match prefix shape"
        assert self.rotation.shape[:-1] == prefix_shape, "Rotation shape must match prefix shape"
        assert self.opacity.shape[:-1] == prefix_shape, "Opacity shape must match prefix shape"
        assert self.xyz.shape[:-1] == prefix_shape, "XYZ shape must match prefix shape"


class GaussianActivations(nn.Module):
    """Combined activation functions for Gaussian parameters.

    Predict-only calls the per-attribute submodules directly
    (`self.gaussian_activations.{rgb,scale,opacity,rotation}` from the
    decoder); `forward` was unused, so the `xyz` / `distance` activation
    classes that fed it are gone
    """

    def __init__(self, config: GaussiansActivationConfig):
        super().__init__()
        self.rgb = RgbActivation()
        self.scale = ScaleActivation(config)
        self.rotation = RotationActivation()
        self.opacity = OpacityActivation(config)
