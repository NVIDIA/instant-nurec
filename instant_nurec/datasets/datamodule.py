# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
