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
from typing import Any, Generic, Iterator

import torch
import torch.nn as nn

from nre.datasets.tracks import CuboidTracks
from nre.nrm.config.models import BaseModelConfig
from nre.nrm.primitives.base import NRMPrimitiveType
from nre.utils.batch import DataAndRenderingBatch


class BaseNRM(nn.Module, Generic[NRMPrimitiveType]):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.config = config

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        """
        Hook function invoked at system.on_train_batch_start(), i.e. the very start of each train step.
        """
        return {}

    def on_train_from_scratch_start(self, system, **kwargs) -> None:
        """
        Hook function for system, invoked at system.on_train_start() only when system.resume is False.
        Useful for e.g. setting scalers, shape initialization process for sdf models, etc.
        """
        pass

    @torch.no_grad()
    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        """
        Temporary hack to properly initialize nrend because nrend expects a renderable primitive during initialization.
        However, the renderable primitive is not available until the feed-forward pass is called.
        Hence this function will then be used to serialize the future-generated primitive to a JSON dict.
        TODO [JH]: Use lazy initialization for BaseGaussianRenderer.factory
        """
        return {}

    @abstractmethod
    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[NRMPrimitiveType]:
        """
        Perform a "reconstruction" step on the provided images.
        """
        pass



    @abstractmethod
    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[DataAndRenderingBatch]:
        """
        Hook function to prepare context data for the model.
        e.g. for the dynamic celsius model we have to compute the velocity for the context.
        """
        pass

    def get_potential_unused_parameters(self) -> Iterator[nn.Parameter]:
        """
        Return an iterator of potential unused parameters that can be used to sink the parameters during training.
        This is useful for avoiding unused parameters in DDP setting.
        """
        return iter([])
