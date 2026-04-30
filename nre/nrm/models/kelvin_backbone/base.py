# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import torch

from torch import nn

from nre.nrm.models.base import BaseNRMSupervisionPack


@dataclass(kw_only=True)
class KelvinMotionSupervision:
    """
    Motion supervision targets vs predicted flow.
    Fields:
    - source_timestamps_us: (..., 1)
    - target_timestamps_us: (..., 1)
    - context_flow: (..., 3)
    - reference_flow: (..., 3) | None
    """

    source_timestamps_us: torch.Tensor
    target_timestamps_us: torch.Tensor
    context_flow: torch.Tensor
    reference_flow: torch.Tensor | None = None


@dataclass(kw_only=True)
class KelvinNRMSupervisionPack(BaseNRMSupervisionPack):
    """
    Supervision pack for the Kelvin model.

    The fields are:
    - context_rgb: (B, H, W, 3) if pixel-aligned
    - context_depth: (B, H, W, 1) if pixel-aligned
    - context_depth_conf: (B, H, W, 1) if pixel-aligned
    - context_semantic_logits: (B, H, W, C) if pixel-aligned
    - context_xyz: (B, H, W, 3) if pixel-aligned
    - context_world_normal: (B, H, W, 3) if pixel-aligned (unit-norm, world space)
    - predicted_sky_cubemap: (6, S, S, 3)
    - reference_sky_cubemap: (6, S, S, 3)
    - reference_sky_cubemap_mask: (6, S, S, 1)
    - motion_supervisions: list[KelvinMotionSupervision] (... = B, H, W)
    """

    context_rgb: torch.Tensor | None = None
    context_depth: torch.Tensor | None = None
    context_depth_conf: torch.Tensor | None = None
    context_semantic_logits: torch.Tensor | None = None
    context_xyz: torch.Tensor | None = None
    context_world_normal: torch.Tensor | None = None
    predicted_sky_cubemap: torch.Tensor | None = None
    reference_sky_cubemap: torch.Tensor | None = None
    reference_sky_cubemap_mask: torch.Tensor | None = None
    motion_supervisions: list[KelvinMotionSupervision] = field(default_factory=list)


def _tokengs_init_weights(m: nn.Module):
    """Initialize weights of the transformer backbone."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@dataclass(kw_only=True, slots=True)
class KelvinLatent(ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        Get the batch size of the latent.
        """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get the device of the latent.
        """

    @property
    @abstractmethod
    def deepest(self) -> torch.Tensor:
        """
        Get the deepest feature of the latent.
        Note this is not necessarily normalized via layer norm.
        Size will be (B, V, h, w, C)
        """


@dataclass(kw_only=True, slots=True)
class KelvinFeatureLatent(KelvinLatent):
    feature: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.feature.shape[0]

    @property
    def device(self) -> torch.device:
        return self.feature.device

    @property
    def deepest(self) -> torch.Tensor:
        return self.feature


@dataclass(kw_only=True, slots=True)
class KelvinMultiscaleFeaturesLatent(KelvinLatent):
    """
    Features means transformed queries (i.e. output of the attention block).
    """

    # (B, V, h, w, C)
    features: list[torch.Tensor]

    # (B, V, n_cls_tokens, C)
    cls_tokens: list[torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        return self.features[0].shape[0]

    @property
    def device(self) -> torch.device:
        return self.features[0].device

    @property
    def deepest(self) -> torch.Tensor:
        return self.features[-1]
