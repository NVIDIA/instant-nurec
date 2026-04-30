# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import math

from abc import ABC, abstractmethod
from typing import Any, Type

import torch
import torch.nn as nn
import torch_scatter

from omegaconf import DictConfig

from nre.config.trainer import TrainerConfig
from nre.datasets.tracks import Tracks
from nre.models.base import BaseModel
from nre.models.custom_modules import Embedding
from nre.models.nn_extensions import module_call_type
from nre.utils.misc import unpack_optional
from nre.utils.trainer import adjust_step_for_world_size
from nre.utils.types import ModelInput


log = logging.getLogger(__name__)


def beta_cosine_schedule(beta_config: DictConfig, global_step: int) -> float:
    """
    Cosine annealing schedule for beta, so that we decrease the time transitioning over time.
    """
    ratio = (global_step - beta_config.start_global_step) / (
        beta_config.end_global_step - beta_config.start_global_step
    )
    ratio = max(0.0, min(1.0, ratio))
    beta = beta_config.start_timestamps_us + 0.5 * (beta_config.end_timestamps_us - beta_config.start_timestamps_us) * (
        1 - math.cos(math.pi * ratio)
    )
    return beta


class BaseInputEmbedding(BaseModel, ABC):
    @staticmethod
    def factory(
        name: str,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ) -> BaseInputEmbedding:
        assert (timestamps_us is not None) or (tracks is not None), "Either timestamps_us or tracks must be provided"

        variants: dict[str, Type[BaseInputEmbedding]] = {
            "skip-input-embedding": SkipInputEmbedding,
            "holistic-remap-time-input-embedding": HolisticRemapTimeInputEmbedding,
            "holistic-step-time-input-embedding": HolisticStepTimeInputEmbedding,
            "individual-remap-time-input-embedding": IndividualRemapTimeInputEmbedding,
            "individual-step-time-input-embedding": IndividualStepTimeInputEmbedding,
            "weighted-instance-input-embedding": WeightedInstanceInputEmbedding,
        }
        variant = variants[name]
        return variant(config, trainer_config=trainer_config, timestamps_us=timestamps_us, tracks=tracks)

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ):
        super().__init__(config)
        self.trainer_config = trainer_config

    @abstractmethod
    def forward(self, inputs: ModelInput) -> ModelInput: ...

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @property
    def requires_grad(self) -> bool:
        raise NotImplementedError

    @torch.no_grad()
    def uniform_sample(self, n_points: int, device: torch.device | None = None) -> torch.Tensor:
        raise NotImplementedError

    __call__ = module_call_type(forward)


class SkipInputEmbedding(BaseInputEmbedding):
    """An empty class that does nothing"""

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ):
        super().__init__(config, trainer_config, timestamps_us, tracks)

    def forward(self, inputs: ModelInput) -> ModelInput:
        return inputs

    @property
    def embedding_dim(self) -> int:
        return 0

    @property
    def requires_grad(self) -> bool:
        return False


class WeightedInstanceInputEmbedding(BaseInputEmbedding):
    """Embed instance indices into a pre-determined set of weights"""

    embedding: Embedding

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ) -> None:
        super().__init__(config, trainer_config, timestamps_us, tracks)

        assert tracks is not None, f"{self.__class__.__name__} requires tracks to be provided"

        self.embedding = Embedding(
            tracks.n_tracks,
            self.config.embedding_dim,
            device=self.device,
            weight_init_config=self.config.weight_init_config,
            requires_grad=self.config.requires_grad,
        )

    def forward(self, inputs: ModelInput) -> ModelInput:
        inputs.instance_emb = self.embedding(unpack_optional(inputs.instance_idx))
        return inputs

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def requires_grad(self) -> bool:
        return self.embedding.weight.requires_grad

    @torch.no_grad()
    def uniform_sample(self, n_points: int, device: torch.device | None = None) -> torch.Tensor:
        ret = torch.empty(
            [n_points, self.embedding_dim], dtype=self.embedding.weight.dtype, device=device or self.embedding.device
        )
        match (init_config := self.embedding.weight_init_config).method:
            case "random_normal" | "randn":
                ret.normal_(init_config.mean, init_config.std)
            case "random_uniform" | "linspace" | "meshgrid":
                ret.uniform_(-init_config.from_, init_config.to_)
            case "random_bernoulli":
                ret.bernoulli_(init_config.p)
            case _:
                mean = self.embedding.weight.mean().item()
                std = self.embedding.weight.std().item()
                ret.normal_(mean, std)
        return ret


class HolisticTimeInputEmbedding(BaseInputEmbedding):
    """
    Base class for time embeddings that first determine the range of input timestamps_us for further computation.
    For background (when timestamps_us is provided), we use the min and max of the provided timestamps.
    For object (when tracks is provided), we use the min and max of the timestamps of all tracks. Hence the name 'Holistic'.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ):
        super().__init__(config, trainer_config, timestamps_us, tracks)

        if timestamps_us is not None:
            self.timestamps_us_min = timestamps_us.min().item()
            self.timestamps_us_max = timestamps_us.max().item()

        elif tracks is not None:
            self.timestamps_us_min = tracks.tracks_timestamps_us.min().item()
            self.timestamps_us_max = tracks.tracks_timestamps_us.max().item()

    def get_extra_state(self) -> Any:
        return {
            "timestamps_us_min": self.timestamps_us_min,
            "timestamps_us_max": self.timestamps_us_max,
        }

    def set_extra_state(self, state: Any) -> None:
        self.timestamps_us_min = state["timestamps_us_min"]
        self.timestamps_us_max = state["timestamps_us_max"]


class HolisticRemapTimeInputEmbedding(HolisticTimeInputEmbedding):
    """
    Time embedding that remaps timestamps_us to a fixed range (remap_min to remap_max),
    with the 'Holistic' approach derived from base class.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ):
        super().__init__(config, trainer_config, timestamps_us, tracks)
        self.remap_min = self.config.remap_min
        self.remap_max = self.config.remap_max

    def forward(self, inputs: ModelInput) -> ModelInput:
        timestamps_us = unpack_optional(inputs.timestamps_us)
        ratio = (timestamps_us - self.timestamps_us_min) / (self.timestamps_us_max - self.timestamps_us_min)
        inputs.time_emb = (ratio * (self.remap_max - self.remap_min) + self.remap_min).unsqueeze(-1)
        return inputs

    @property
    def embedding_dim(self) -> int:
        return 1

    @property
    def requires_grad(self) -> bool:
        return False

    @torch.no_grad()
    def uniform_sample(self, n_points: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.empty([n_points, self.embedding_dim], device=device).uniform_(self.remap_min, self.remap_max)


class HolisticStepTimeInputEmbedding(HolisticTimeInputEmbedding):
    """
    Time embedding that uses a step function to embed timestamps_us, with the 'Holistic' approach derived from base class.
    The step function is differentiable and is taken from NSC (https://arxiv.org/abs/2306.07970)
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ):
        super().__init__(config, trainer_config, timestamps_us, tracks)

        self.config.beta.start_global_step = adjust_step_for_world_size(
            trainer_config, self.config.beta.start_global_step
        )
        self.config.beta.end_global_step = adjust_step_for_world_size(trainer_config, self.config.beta.end_global_step)
        log.info(
            f"HolisticStepTimeInputEmbedding/beta: start_global_step={self.config.beta.start_global_step} end_global_step={self.config.beta.end_global_step}"
        )

        n_steps, n_dims = self.config.n_steps, self.config.n_dims
        self.u = nn.Parameter(
            torch.linspace(0.0, 1.0, n_steps * n_dims + 2)[1:-1]
            .reshape(1, n_steps, n_dims)
            .transpose(1, 2)
            .contiguous(),
            requires_grad=self.config.requires_grad,
        )
        self.beta = nn.Parameter(
            torch.tensor(self.config.beta.start_timestamps_us).float(),
            requires_grad=False,
        )

    def forward(self, inputs: ModelInput) -> ModelInput:
        timestamps_us = unpack_optional(inputs.timestamps_us)
        ratio = (timestamps_us - self.timestamps_us_min) / (self.timestamps_us_max - self.timestamps_us_min)
        # Note that sigmoid(-5.0 ~ 5.0) = 0.01 ~ 0.99
        inv_beta = (self.timestamps_us_max - self.timestamps_us_min) / self.beta * 10.0
        u = torch.sigmoid(self.u * 10 - 5)
        # We don't apply clamping to u to allow for more flexibility. (T x n_dims x n_steps)
        inputs.time_emb = torch.sigmoid(inv_beta * (ratio[:, None, None] - u)).mean(dim=-1)
        return inputs

    @property
    def embedding_dim(self) -> int:
        return self.u.size(1)

    @property
    def requires_grad(self) -> bool:
        return self.u.requires_grad

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict:
        self.beta.fill_(beta_cosine_schedule(self.config.beta, global_step))
        return {}

    @torch.no_grad()
    def uniform_sample(self, n_points: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.empty([n_points, self.embedding_dim], device=device).uniform_(0, 1)


class IndividualTimeInputEmbedding(BaseInputEmbedding):
    """
    Base class for time embeddings that first determine the range of input timestamps_us for further computation.
    For object (when tracks is provided), we use the min and max of the timestamps of each track individually. (Hence the name 'Individual')
    We do not support timestamps_us argument (background) for this class.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ):
        super().__init__(config, trainer_config, timestamps_us, tracks)

        assert tracks is not None, f"{self.__class__.__name__} requires tracks to be provided"

        tracks_indptr = tracks.tracks_packinfo[:, 1].cumsum(0)
        tracks_indptr = torch.cat([torch.tensor([0], device=tracks_indptr.device), tracks_indptr])
        min_timestamps_us = torch_scatter.segment_min_csr(tracks.tracks_timestamps_us, tracks_indptr)[0]
        max_timestamps_us = torch_scatter.segment_max_csr(tracks.tracks_timestamps_us, tracks_indptr)[0]
        self.timestamps_us_ranges = nn.Buffer(torch.stack([min_timestamps_us, max_timestamps_us], dim=1))


class IndividualRemapTimeInputEmbedding(IndividualTimeInputEmbedding):
    """
    Time embedding that remaps timestamps_us to a fixed range (remap_min to remap_max),
    with the 'Individual' approach derived from base class.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ):
        super().__init__(config, trainer_config, timestamps_us, tracks)
        self.remap_min = self.config.remap_min
        self.remap_max = self.config.remap_max

    def forward(self, inputs: ModelInput) -> ModelInput:
        timestamps_us = unpack_optional(inputs.timestamps_us)
        timestamps_us_ranges = self.timestamps_us_ranges[unpack_optional(inputs.instance_idx)]
        ratio = (timestamps_us - timestamps_us_ranges[:, 0]) / (timestamps_us_ranges[:, 1] - timestamps_us_ranges[:, 0])
        inputs.time_emb = (ratio.clamp(0, 1) * (self.remap_max - self.remap_min) + self.remap_min).unsqueeze(-1)
        return inputs

    @property
    def embedding_dim(self) -> int:
        return 1

    @property
    def requires_grad(self) -> bool:
        return False

    @torch.no_grad()
    def uniform_sample(self, n_points: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.empty([n_points, self.embedding_dim], device=device).uniform_(self.remap_min, self.remap_max)


class IndividualStepTimeInputEmbedding(IndividualTimeInputEmbedding):
    """
    Time embedding that uses a step function to embed timestamps_us, with the 'Individual' approach derived from base class.
    The step function is differentiable and is taken from NSC (https://arxiv.org/abs/2306.07970)
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        timestamps_us: torch.Tensor | None = None,
        tracks: Tracks | None = None,
    ):
        super().__init__(config, trainer_config, timestamps_us, tracks)
        self._trainer_config = trainer_config
        self.update_stepness = self.config.get("update_stepness", True)

        if self.update_stepness:
            self.config.beta.start_global_step = adjust_step_for_world_size(
                self.trainer_config, self.config.beta.start_global_step
            )
            self.config.beta.end_global_step = adjust_step_for_world_size(
                self.trainer_config, self.config.beta.end_global_step
            )
            log.info(
                f"IndividualStepTimeInputEmbedding/beta: start_global_step={self.config.beta.start_global_step} end_global_step={self.config.beta.end_global_step}"
            )

        n_steps, n_dims = self.config.n_steps, self.config.n_dims
        self.u = nn.Parameter(
            torch.linspace(0.0, 1.0, n_steps * n_dims + 2)[1:-1]
            .reshape(1, n_steps, n_dims)
            .transpose(1, 2)
            .repeat(unpack_optional(tracks).n_tracks, 1, 1)
            .contiguous(),
            requires_grad=self.config.requires_grad,
        )
        self.n_steps = n_steps
        self.n_dims = n_dims
        self.beta = nn.Parameter(
            torch.tensor(self.config.beta.start_timestamps_us).float(),
            requires_grad=False,
        )

    def forward(self, inputs: ModelInput) -> ModelInput:
        instance_idx = unpack_optional(inputs.instance_idx)
        timestamps_us = unpack_optional(inputs.timestamps_us)
        timestamps_us_ranges = self.timestamps_us_ranges[instance_idx]
        ratio = (timestamps_us - timestamps_us_ranges[:, 0]) / (timestamps_us_ranges[:, 1] - timestamps_us_ranges[:, 0])
        beta = (self.beta / (timestamps_us_ranges[:, 1] - timestamps_us_ranges[:, 0]))[:, None, None].repeat(
            1, 1, self.n_steps
        )
        output = ratio[:, None, None] - unpack_optional(self.u)[instance_idx]
        msk = output <= 0.0
        output[msk] = 0.5 * torch.exp(output[msk] / torch.clamp_min(torch.abs(beta[msk]), 1e-3))
        output[~msk] = 1 - 0.5 * torch.exp(-output[~msk] / torch.clamp_min(torch.abs(beta[~msk]), 1e-3))
        inputs.time_emb = output.mean(dim=-1)

        return inputs

    @property
    def embedding_dim(self) -> int:
        return self.u.size(1)

    @property
    def requires_grad(self) -> bool:
        return self.u.requires_grad

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict:
        if self.update_stepness:
            self.beta.fill_(beta_cosine_schedule(self.config.beta, global_step))
        return {}

    @torch.no_grad()
    def uniform_sample(self, n_points: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.empty([n_points, self.embedding_dim], device=device).uniform_(0, 1)
