# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from abc import abstractmethod
from typing import Generic

import torch
import torch.nn as nn

from nre.datasets.tracks import CuboidTracks
from nre.nrm.config.models import KelvinModelConfig
from nre.nrm.primitives.base import NRMPrimitiveType
from nre.utils.batch import DataAndRenderingBatch


class BaseNRM(nn.Module, Generic[NRMPrimitiveType]):
    def __init__(self, config: KelvinModelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[NRMPrimitiveType]:
        """Perform a reconstruction step on the provided images."""

    @abstractmethod
    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[DataAndRenderingBatch]:
        """Prepare context data before reconstruction."""
