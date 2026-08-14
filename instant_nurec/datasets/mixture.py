# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import logging
import os

from dataclasses import dataclass, field

import numpy as np
import torch

from instant_nurec.config_schema.dataset import NCoreMixtureDatasetConfig
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.utils.batch import InstantNuRecDataBatch


logger = logging.getLogger(__name__)


class NCoreMixtureDataset(torch.utils.data.Dataset[InstantNuRecDataBatch]):
    """Mixture sampling with the same length and resampling contract as Bazel NRE."""

    @dataclass
    class SubDataset:
        name: str
        dataset: NCoreInstantNuRecDataset
        sample_ratio: float
        full_length: int = field(init=False)
        sampled_length: int = field(init=False)

        def __post_init__(self) -> None:
            self.full_length = len(self.dataset)
            if self.full_length <= 0:
                raise ValueError(f"Dataset {self.name!r} is empty")
            self.sampled_length = int(self.full_length * self.sample_ratio)
            logger.info(
                "Mixture component %s has %d samples; selecting %d (ratio %.4g).",
                self.name,
                self.full_length,
                self.sampled_length,
                self.sample_ratio,
            )

        def sample_at(self, index: int, rng: np.random.Generator) -> InstantNuRecDataBatch:
            if not 0 <= index < self.sampled_length:
                raise IndexError(f"Sub-index {index} is outside component {self.name!r}")
            if self.sampled_length >= self.full_length and index < self.full_length:
                return self.dataset[index]
            return self.dataset[int(rng.integers(self.full_length))]

    def __init__(self, config: NCoreMixtureDatasetConfig, *, global_seed: int | None = None) -> None:
        self.datasets: list[NCoreMixtureDataset.SubDataset] = []
        self._global_seed = global_seed
        self._rng_epoch = -1
        self._epoch = -1
        for name, component in config.mixture.items():
            if component.sample_ratio == 0:
                continue
            dataset_config = component.config
            dataset = NCoreInstantNuRecDataset(
                dataset_config,
                frame_width=dataset_config.camera_subsampler.frame_width,
                frame_height=dataset_config.camera_subsampler.frame_height,
                n_frames_per_sample=dataset_config.frame_batch_sampler.n_frames_per_sample,
                global_seed=global_seed,
            )
            self.datasets.append(self.SubDataset(name, dataset, component.sample_ratio))
        if not self.datasets:
            raise ValueError("NCore mixture has no enabled components")

    def __len__(self) -> int:
        return sum(component.sampled_length for component in self.datasets)

    def set_rng_epoch(self, rng_epoch: int) -> None:
        self._rng_epoch = rng_epoch
        for component in self.datasets:
            component.dataset.set_rng_epoch(rng_epoch)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        for component in self.datasets:
            component.dataset.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    def _get_rng(self, batch_idx: int) -> np.random.Generator:
        global_seed = self._global_seed
        if global_seed is None:
            if "PL_GLOBAL_SEED" not in os.environ:
                raise RuntimeError(
                    "No mixture global seed was supplied and PL_GLOBAL_SEED is unset; "
                    "pass global_seed or call pytorch_lightning.seed_everything() first"
                )
            global_seed = int(os.environ["PL_GLOBAL_SEED"])
        digest = hashlib.sha256(f"{self._rng_epoch}_{batch_idx}_{global_seed}".encode()).digest()
        return np.random.default_rng(seed=int.from_bytes(digest[:8], "big"))

    def __getitem__(self, batch_idx: int) -> InstantNuRecDataBatch:
        if not 0 <= batch_idx < len(self):
            raise IndexError(f"Mixture index {batch_idx} is outside [0, {len(self)})")
        rng = self._get_rng(batch_idx)
        component_index = batch_idx
        for component in self.datasets:
            if component_index < component.sampled_length:
                return component.sample_at(component_index, rng)
            component_index -= component.sampled_length
        raise RuntimeError("Mixture index routing fell through")
