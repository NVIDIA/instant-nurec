# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from instant_nurec.config_schema.dataset import NCoreInstantNuRecDatasetConfig
from instant_nurec.config_schema.train import KelvinTrainConfig
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.datasets.mixture import NCoreMixtureDataset
from instant_nurec.utils.batch import InstantNuRecDataBatch


class KelvinTrainingDataModule(LightningDataModule):
    def __init__(self, config: KelvinTrainConfig):
        super().__init__()
        self.config = config
        self.train_dataset: NCoreInstantNuRecDataset | NCoreMixtureDataset | None = None
        self.val_dataset: NCoreInstantNuRecDataset | NCoreMixtureDataset | None = None

    def _make_dataset(self, dataset_config):
        if not isinstance(dataset_config, NCoreInstantNuRecDatasetConfig):
            return NCoreMixtureDataset(dataset_config, global_seed=self.config.seed)
        return NCoreInstantNuRecDataset(
            dataset_config,
            frame_width=dataset_config.camera_subsampler.frame_width,
            frame_height=dataset_config.camera_subsampler.frame_height,
            n_frames_per_sample=dataset_config.frame_batch_sampler.n_frames_per_sample,
            global_seed=self.config.seed,
        )

    def train_dataloader(self) -> DataLoader:
        dataset_config = self.config.dataset.train
        assert dataset_config is not None
        if self.train_dataset is None:
            self.train_dataset = self._make_dataset(dataset_config)
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            self.train_dataset.set_rng_epoch(trainer.current_epoch)
            self.train_dataset.set_epoch(trainer.current_epoch)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.system.train_batch_size,
            num_workers=self.config.system.train_num_workers,
            shuffle=True,
            pin_memory=True,
            # Dataloaders are recreated each epoch to propagate rng_epoch into
            # workers, exactly as in the Bazel Lightning implementation.
            persistent_workers=False,
            collate_fn=InstantNuRecDataBatch.collate_fn,
        )

    def val_dataloader(self) -> DataLoader | list:
        dataset_config = self.config.dataset.val
        if dataset_config is None:
            # Lightning DataModule hooks must return an iterable collection,
            # not None, when a phase intentionally has no validation split.
            return []
        if self.val_dataset is None:
            self.val_dataset = self._make_dataset(dataset_config)
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            # Validation samples are deliberately epoch-independent.  Only the
            # augmentation epoch changes; rng_epoch stays at its default -1.
            self.val_dataset.set_epoch(trainer.current_epoch)
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.system.val_batch_size,
            num_workers=self.config.system.val_num_workers,
            shuffle=False,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=InstantNuRecDataBatch.collate_fn,
        )
