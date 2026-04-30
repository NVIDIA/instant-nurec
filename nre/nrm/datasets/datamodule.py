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

from typing import cast

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from nre.nrm.config.dataset import NRMSplitsConfig
from nre.nrm.config.nrm import NRMConfig
from nre.nrm.datasets.nrm_base import BaseNRMDataset
from nre.nrm.datasets.registry import make as make_dataset
from nre.utils.batch import NRMDataBatch
from nre.utils.misc import unpack_optional


log = logging.getLogger(__name__)


class NRMDataModule(LightningDataModule):
    nrm_config: NRMConfig
    dataset_config: NRMSplitsConfig

    predict_dataset: BaseNRMDataset | None = None

    def __init__(self, nrm_config: NRMConfig) -> None:
        super().__init__()
        self.nrm_config = nrm_config
        assert isinstance(nrm_config.dataset, NRMSplitsConfig)
        self.dataset_config = nrm_config.dataset

    def predict_dataloader(self) -> DataLoader:
        """Returns a predict dataloader."""
        dataset_config = self.dataset_config.predict
        assert dataset_config is not None, "dataset.predict has to be specified in the config to use the predict mode"

        self.predict_dataset = cast(BaseNRMDataset, make_dataset(dataset_config.name, dataset_config, "predict"))

        if self.trainer is not None and self.predict_dataset is not None:
            self.predict_dataset.set_epoch(self.trainer.current_epoch)

        return DataLoader(
            unpack_optional(self.predict_dataset),
            num_workers=self.nrm_config.system.predict_num_workers,
            persistent_workers=False,
            batch_size=self.nrm_config.system.predict_batch_size,
            pin_memory=True,
            collate_fn=NRMDataBatch.collate_fn,
        )


