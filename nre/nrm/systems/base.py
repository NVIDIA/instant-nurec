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

from abc import ABC

import torch

from torch import nn

from nre.nrm.config.nrm import BaseNRMSystemConfig, NRMConfig
from nre.nrm.datasets.datamodule import NRMDataModule
from nre.nrm.models.base import BaseNRM
from nre.utils.batch import NRMDataBatch
from nre.utils.types import Checkpoint


class BaseNRMSystem(nn.Module, ABC):
    """Predict-only system. Self-invented: NRE inherits LightningModule for the
    Trainer.fit/validate/test surfaces; we keep just nn.Module since the
    predict driver invokes hooks directly."""

    config: BaseNRMSystemConfig
    model: BaseNRM
    datamodule: NRMDataModule

    def __init__(self, config: NRMConfig) -> None:
        super().__init__()

        self.cached_config = config
        self.out_dir = config.out_dir
        self.run_id = config.run_id
        self.config = config.system
        self.predict_config = config.predict

        self.datamodule = NRMDataModule(config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def on_predict_batch_start(self, batch: NRMDataBatch, batch_local_idx: int, dataloader_idx: int = 0) -> None:
        self.model.update_step_train_batch_start(0, 0, self)

    def on_load_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.load_state_dict(checkpoint["state_dict"], assign=True)
