# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import weakref

from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler

from instant_nurec.config_schema.dataset import NCoreInstantNuRecDatasetConfig
from instant_nurec.config_schema.train import TrainingConfig
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.datasets.mixture import NCoreMixtureDataset
from instant_nurec.utils.batch import InstantNuRecDataBatch


logger = logging.getLogger(__name__)


class SkipBatchSampler(BatchSampler):
    """Skip completed batches while preserving the original epoch length.

    ``sampler`` deliberately remains the first constructor argument. Lightning
    reconstructs custom batch samplers with an injected ``DistributedSampler``
    for DDP and preserves ``first_batch_idx`` from the captured constructor
    arguments.
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
        first_batch_idx: int = 0,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        if first_batch_idx < 0:
            raise ValueError(f"first_batch_idx must be non-negative, got {first_batch_idx}")
        self.first_batch_idx = first_batch_idx

    def __iter__(self):
        for batch_idx, batch in enumerate(super().__iter__()):
            if batch_idx >= self.first_batch_idx:
                yield batch

    # Do not subtract ``first_batch_idx`` here. Lightning restores its loop
    # progress from the checkpoint and needs the original number of batches to
    # continue emitting epoch-global batch indices (and scheduler progress).


class TrainingDataModule(LightningDataModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.train_dataset: NCoreInstantNuRecDataset | NCoreMixtureDataset | None = None
        self.val_dataset: NCoreInstantNuRecDataset | NCoreMixtureDataset | None = None
        self._next_train_batch_idx = 0

    def state_dict(self) -> dict[str, int]:
        """Persist the first not-yet-completed train batch for mid-epoch resume."""

        return {"next_train_batch_idx": self._next_train_batch_idx}

    def load_state_dict(self, state_dict: dict) -> None:
        self._next_train_batch_idx = int(state_dict.get("next_train_batch_idx", 0))

    def _make_dataset(self, dataset_config):
        if not isinstance(dataset_config, NCoreInstantNuRecDatasetConfig):
            return NCoreMixtureDataset(
                dataset_config,
                global_seed=self.config.seed,
                retry_on_error=True,
            )
        return NCoreInstantNuRecDataset(
            dataset_config,
            frame_width=dataset_config.camera_subsampler.frame_width,
            frame_height=dataset_config.camera_subsampler.frame_height,
            n_frames_per_sample=dataset_config.frame_batch_sampler.n_frames_per_sample,
            global_seed=self.config.seed,
            retry_on_error=True,
        )

    def train_dataloader(self) -> DataLoader:
        dataset_config = self.config.dataset.train
        assert dataset_config is not None
        if self.train_dataset is None:
            self.train_dataset = self._make_dataset(dataset_config)
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            self.train_dataset.set_rng_epoch(trainer.current_epoch)
            self.train_dataset.set_epoch(trainer.current_epoch)

        if self._next_train_batch_idx:
            logger.info("Skipping %d completed train batches after resume.", self._next_train_batch_idx)

        # Seed the single-process RandomSampler exactly like the distributed
        # sampler (global seed + epoch). This makes the underlying permutation
        # reconstructible after a restart. In distributed runs Lightning sees
        # RandomSampler and injects its DistributedSampler into
        # SkipBatchSampler, retaining first_batch_idx.
        epoch = trainer.current_epoch if trainer is not None else 0
        generator = torch.Generator().manual_seed(self.config.seed + epoch)
        batch_sampler = SkipBatchSampler(
            sampler=RandomSampler(self.train_dataset, generator=generator),
            batch_size=self.config.system.train_batch_size,
            first_batch_idx=self._next_train_batch_idx,
        )
        return DataLoader(
            self.train_dataset,
            num_workers=self.config.system.train_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            # Dataloaders are recreated each epoch to propagate rng_epoch into
            # workers, exactly as in the reference Lightning implementation.
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
        trainer = getattr(self, "trainer", None)
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


class ResumableDataModuleCallback(Callback):
    """Advance the data-module cursor before any batch checkpoint is saved."""

    def __init__(self, datamodule: TrainingDataModule) -> None:
        super().__init__()
        self._datamodule = weakref.ref(datamodule)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        del trainer, pl_module, outputs, batch
        if (datamodule := self._datamodule()) is not None:
            # Lightning resumes with an epoch-global batch_idx, including after
            # repeated mid-epoch interruptions.
            datamodule._next_train_batch_idx = batch_idx + 1

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        del trainer, pl_module
        if (datamodule := self._datamodule()) is not None:
            datamodule._next_train_batch_idx = 0
