# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import weakref

from typing import Sized, cast

from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.callback import Callback
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler

from nre.nrm.config.dataset import NRMEpochSplitConfig, NRMSplitConfig, NRMSplitsConfig
from nre.nrm.config.nrm import NRMConfig
from nre.nrm.datasets.nrm_base import BaseNRMDataset
from nre.nrm.datasets.registry import make as make_dataset
from nre.utils.batch import NRMDataBatch
from nre.utils.misc import unpack_optional


log = logging.getLogger(__name__)


class SkipBatchSampler(BatchSampler):
    """
    A BatchSampler subclass that ensures to skip the first N seen batches when resuming mid-epoch.
    PL injects distributed sampler by re-creating this class with the replaced 'sampler' argument.
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool = False,
        first_batch_idx: int = 0,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.first_batch_idx = first_batch_idx

    def __iter__(self):
        for i, batch in enumerate(super().__iter__()):
            if i >= self.first_batch_idx:
                yield batch

    # Although we skip the first few batches, PL still depends on the length of the dataloader
    # to determine the total number of expected batches in this epoch.
    # This is why we can simply inherit __len__ here.


class NRMDataModule(LightningDataModule):
    nrm_config: NRMConfig
    dataset_config: NRMSplitsConfig

    # Keep a reference of the current datasets, so no need to rebuild for every epoch.
    train_dataset: BaseNRMDataset | None = None
    val_dataset: BaseNRMDataset | None = None
    test_dataset: BaseNRMDataset | None = None
    predict_dataset: BaseNRMDataset | None = None

    def __init__(self, nrm_config: NRMConfig) -> None:
        super().__init__()
        self.nrm_config = nrm_config
        assert isinstance(nrm_config.dataset, NRMSplitsConfig)
        self.dataset_config = nrm_config.dataset

        # Persistent state to track the next batch idx to be loaded, useful for mid-epoch resume.
        # This should be synchronized across all ranks.
        self._next_train_batch_idx: int = 0

    def state_dict(self) -> dict:
        return {
            "next_train_batch_idx": self._next_train_batch_idx,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._next_train_batch_idx = int(state_dict.get("next_train_batch_idx", 0))

    def _get_rebuild_config(
        self, current_dataset: BaseNRMDataset | None, split_config: NRMSplitConfig | NRMEpochSplitConfig
    ) -> NRMSplitConfig | None:
        """
        Check if the dataset needs to be rebuilt. If so, return the configuration to rebuild it.
        """
        if not isinstance(split_config, NRMEpochSplitConfig):
            # for datasets which are not per-epoch, build only if no dataset exists currently
            return split_config if current_dataset is None else None

        # for datasets which are per-epoch, we need to look up the right config in the milestones
        milestones = split_config.milestones()

        if self.trainer is None:
            # test/validation, use the last dataset config (likely highest resolution, etc)
            return milestones[max(milestones.keys())]

        current_epoch = self.trainer.current_epoch

        if current_epoch in milestones:
            # if current_epoch is in milestones, we need to rebuild the dataset
            return milestones[current_epoch]
        if current_dataset is None:
            # if current_dataset is None, we pick the last milestone before or equal to current epoch
            previous_milestone_idx = max(k for k in milestones.keys() if k <= current_epoch)
            return milestones[previous_milestone_idx]

        # if current_epoch is not in milestones and we already have a dataset, we don't need to rebuild it
        return None

    def train_dataloader(self) -> DataLoader:
        """
        Returns a training dataloader
        """
        if (dataset_config := self._get_rebuild_config(self.train_dataset, self.dataset_config.train)) is not None:
            self.train_dataset = cast(BaseNRMDataset, make_dataset(dataset_config.name, dataset_config, "train"))

        assert self.train_dataset is not None, "train_dataset should be built by now"
        assert isinstance(self.train_dataset, Sized), "train_dataset should be a BaseNRMIndexableDataset"

        if self.trainer is not None:
            self.train_dataset.set_rng_epoch(self.trainer.current_epoch)
            self.train_dataset.set_epoch(self.trainer.current_epoch)

        if self._next_train_batch_idx != 0:
            log.info(f"Skipping {self._next_train_batch_idx} trained batches from resuming.")

        batch_sampler = SkipBatchSampler(
            # Here the sampler is set to mimic the behavior of setting shuffle=True & generator=None
            # Within PL, it will automatically inject into the batch sampler and switch the sampler to
            # a DistributedSampler if needed (It detects if sampler is RandomSampler or SequentialSampler and set the shuffle flag).
            # This also handles complicated scheme such as ModelParallel, where PL properly pass in the
            # data_parallel_group to the DistributedSampler.
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.nrm_config.system.train_batch_size,
            first_batch_idx=self._next_train_batch_idx,
        )

        prefetch_factor = (
            self.nrm_config.system.train_prefetch_factor if self.nrm_config.system.train_num_workers > 0 else None
        )
        return DataLoader(
            self.train_dataset,
            num_workers=self.nrm_config.system.train_num_workers,
            # We reload the dataset at every epoch, so no need to persist workers, which might
            # lead to additional RAM consumption in trainval mode
            persistent_workers=False,
            batch_sampler=batch_sampler,
            collate_fn=NRMDataBatch.collate_fn,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a validation dataloader
        """
        if (dataset_config := self._get_rebuild_config(self.val_dataset, self.dataset_config.val)) is not None:
            self.val_dataset = cast(BaseNRMDataset, make_dataset(dataset_config.name, dataset_config, "val"))

        assert self.val_dataset is not None, "val_dataset should be built by now"
        assert isinstance(self.val_dataset, Sized), "val_dataset should be a BaseNRMIndexableDataset"

        if self.trainer is not None:
            # Not calling set_rng_epoch here, as validation RNGs should not depend on epoch.
            self.val_dataset.set_epoch(self.trainer.current_epoch)

        prefetch_factor = (
            self.nrm_config.system.val_prefetch_factor if self.nrm_config.system.val_num_workers > 0 else None
        )
        return DataLoader(
            self.val_dataset,
            num_workers=self.nrm_config.system.val_num_workers,
            persistent_workers=False,
            batch_size=self.nrm_config.system.val_batch_size,
            collate_fn=NRMDataBatch.collate_fn,
            # Unfortunately PL's logic forces us to keep training workers alive during validation,
            # even with persistent_workers=False. In cases where we OOM we need some workarounds for this.
            pin_memory=False,
            prefetch_factor=prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a test dataloader
        """
        assert self.dataset_config.test is not None, (
            "dataset.test has to be specified in the config to use the test mode"
        )
        if (dataset_config := self._get_rebuild_config(self.test_dataset, self.dataset_config.test)) is not None:
            self.test_dataset = cast(BaseNRMDataset, make_dataset(dataset_config.name, dataset_config, "test"))

        if self.trainer is not None and self.test_dataset is not None:
            # Not calling set_rng_epoch here, as testing RNGs should not depend on epoch.
            self.test_dataset.set_epoch(self.trainer.current_epoch)

        prefetch_factor = (
            self.nrm_config.system.test_prefetch_factor if self.nrm_config.system.test_num_workers > 0 else None
        )
        return DataLoader(
            unpack_optional(self.test_dataset),
            num_workers=self.nrm_config.system.test_num_workers,
            persistent_workers=False,
            batch_size=self.nrm_config.system.test_batch_size,
            pin_memory=True,
            collate_fn=NRMDataBatch.collate_fn,
            prefetch_factor=prefetch_factor,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Returns a predict dataloader
        """
        dataset_config = self.dataset_config.predict
        assert dataset_config is not None, "dataset.predict has to be specified in the config to use the predict mode"

        self.predict_dataset = cast(BaseNRMDataset, make_dataset(dataset_config.name, dataset_config, "predict"))

        if self.trainer is not None and self.predict_dataset is not None:
            # Not calling set_rng_epoch here, as predict RNGs should not depend on epoch.
            self.predict_dataset.set_epoch(self.trainer.current_epoch)

        # Dataset is guaranteed to be loaded in the main process sequentially, typically with batch size 1.
        return DataLoader(
            unpack_optional(self.predict_dataset),
            num_workers=self.nrm_config.system.predict_num_workers,
            persistent_workers=False,
            batch_size=self.nrm_config.system.predict_batch_size,
            pin_memory=True,
            collate_fn=NRMDataBatch.collate_fn,
        )


