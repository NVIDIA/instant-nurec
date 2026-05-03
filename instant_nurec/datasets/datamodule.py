# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from torch.utils.data import DataLoader

from instant_nurec.config_schema.nrm import NRMConfig
from instant_nurec.datasets.nrm_ncore import NCoreNRMDataset
from instant_nurec.utils.batch import NRMDataBatch


class NRMDataModule:
    def __init__(self, nrm_config: NRMConfig) -> None:
        self.nrm_config = nrm_config
        self.predict_dataset: NCoreNRMDataset | None = None

    def predict_dataloader(self) -> DataLoader:
        dataset_config = self.nrm_config.dataset.predict
        assert dataset_config is not None, "dataset.predict has to be specified in the config to use the predict mode"

        self.predict_dataset = NCoreNRMDataset(dataset_config)
        return DataLoader(
            self.predict_dataset,
            num_workers=self.nrm_config.system.predict_num_workers,
            persistent_workers=False,
            batch_size=self.nrm_config.system.predict_batch_size,
            pin_memory=True,
            collate_fn=NRMDataBatch.collate_fn,
        )
