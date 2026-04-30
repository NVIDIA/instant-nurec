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

import contextlib

from abc import ABC
from typing import Any, Optional, Type

import torch

from omegaconf import DictConfig

from nre.config.base_schema import config_to_primitive
from nre.datasets.tracks import CuboidTracks
from nre.models.feature_volume import BaseFeatureVolume, TcnnMultiLevelEncoding
from nre.models.input_embedding import BaseInputEmbedding
from nre.models.nn_extensions import module_call_type
from nre.utils.misc import unpack_optional
from nre.utils.trainer import TrainerConfig


class BaseObjectFeatureVolume(BaseFeatureVolume, ABC):
    @staticmethod
    def obj_factory(
        name: str,
        config: DictConfig,
        trainer_config: TrainerConfig,
        precision: str | int,
        cuboid_tracks: CuboidTracks,
    ) -> BaseObjectFeatureVolume:
        variants: dict[str, Type[BaseObjectFeatureVolume]] = {
            "hash-grid-object": HashGridObjectFeatureVolume,
        }
        return variants[name](config, trainer_config, precision, cuboid_tracks)

    time_input_embedding: BaseInputEmbedding
    instance_input_embedding: BaseInputEmbedding

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        precision: str | int,
        cuboid_tracks: CuboidTracks,
    ) -> None:
        super().__init__(config, precision)

    @property
    def time_dependent(self) -> bool:
        raise NotImplementedError


class HashGridObjectFeatureVolume(BaseObjectFeatureVolume):
    """
    Fused Object Hash Grid that takes x,y,z object local coordinates as input, as well as an object ID.
    It outputs a feature vector (containing geometry + appearance features) for each input.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        precision: str | int,
        # Local object-wise aabbs [N, 6]
        # First three coordinates are the lower-back-left corner and the other three are top-right-front corner
        cuboid_tracks: CuboidTracks,
    ):
        super().__init__(config, trainer_config, precision, cuboid_tracks)

        self.n_output_dims = self.config.n_output_dims

        # Application-specific setup

        # Object input embeddings
        self.time_input_embedding = BaseInputEmbedding.factory(
            config.time_input_embedding.name, config.time_input_embedding, trainer_config, tracks=cuboid_tracks
        )
        self.instance_input_embedding = BaseInputEmbedding.factory(
            config.instance_input_embedding.name, config.instance_input_embedding, trainer_config, tracks=cuboid_tracks
        )

        # initialize encoding
        self.n_input_dims = 3 + self.instance_input_embedding.embedding_dim + self.time_input_embedding.embedding_dim
        if self.config.hash_grid_encoding_config.otype == "Permuto" and self.instance_input_embedding.requires_grad:
            # If the input instance embedding is learnable (requires_grad is True), we should also enable its gradient
            self.config.hash_grid_encoding_config.max_input_grad_dims = 3 + self.instance_input_embedding.embedding_dim

        assert self.config.mlp_network_config is not None, (
            f"[{self.__class__.__name__}] expects non-empty `mlp_network_config` to build fused hash-grid-with-mlp"
        )

        self.encoding = TcnnMultiLevelEncoding(
            n_input_dims=self.n_input_dims,  # [xyzi] or [xyzit]
            n_pos_dims=3,  # [xyz] spatial location
            n_output_dims=self.n_output_dims,
            encoding_config=config_to_primitive(self.config.hash_grid_encoding_config),
            trainer_config=trainer_config,
            encoding_interp_config=config_to_primitive(self.config.encoding_interp_config),
            encoding_prog_config=config_to_primitive(self.config.encoding_prog_config),
            encoding_lod_config=config_to_primitive(self.config.encoding_lod_config),
            mlp_network_config=config_to_primitive(self.config.mlp_network_config),
            mlp_network_input_include_xyz=self.config.include_xyz,
            dtype=self.dtype,
            seed=self.config.seed,
            enable_jit_if_supported=self.config.enable_jit_if_supported,
            double_backward_skip_input_grad=self.config.double_backward_skip_input_grad,
        )

    @property
    def encoding_config(self) -> dict:
        return unpack_optional(self.encoding.encoding_config)

    @encoding_config.setter
    def encoding_config(self, val):
        raise RuntimeError(
            f"[{self.__class__.__name__}] `encoding_config` can not be changed at runtime. Please re-instantiate."
        )

    @property
    def mlp_network_config(self) -> dict:
        return unpack_optional(self.encoding.mlp_network_config)

    @mlp_network_config.setter
    def mlp_network_config(self, val):
        raise RuntimeError(
            f"[{self.__class__.__name__}] `mlp_network_config` can not be changed at runtime. Please re-instantiate."
        )

    @contextlib.contextmanager
    def backward_input_only(self):
        yield self.encoding.backward_input_only()

    @property
    def time_dependent(self) -> bool:
        return self.time_input_embedding.embedding_dim > 0

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-computed concatenated embeddings.

        Args:
            embeddings: Concatenated embeddings [N, 3 + instance_dim + time_dim]
                        Order: [xyzs_unit_cube, instance_emb, time_emb]

        Returns:
            Feature vectors [N, n_output_dims]
        """
        if embeddings.size(0) == 0:
            return torch.empty((0, self.encoding.n_output_dims), dtype=torch.float, device=self.device)

        return self.encoding(embeddings).float()

    __call__ = module_call_type(forward)
