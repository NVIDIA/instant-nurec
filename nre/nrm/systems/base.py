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

from pytorch_lightning import LightningModule

from nre.config.base_schema import config_to_primitive
from nre.nrm.config.nrm import BaseNRMSystemConfig, NRMConfig
from nre.nrm.datasets.datamodule import NRMDataModule
from nre.nrm.models.base import BaseNRM
from nre.utils.batch import NRMDataBatch
from nre.utils.types import Checkpoint


class BaseNRMSystem(LightningModule, ABC):
    config: BaseNRMSystemConfig
    model: BaseNRM
    datamodule: NRMDataModule

    device: torch.device  # mypy type fix for obtaininig system's device

    def __init__(self, config: NRMConfig) -> None:
        super().__init__()

        self.save_hyperparameters(config_to_primitive(config.to_dictconfig()))

        # save what we need from the config
        self.cached_config = config
        self.out_dir = config.out_dir
        self.run_id = config.run_id
        self.config = config.system
        self.predict_config = config.predict

        self.datamodule = NRMDataModule(config)

    # ---- Test loop methods ----


    def on_predict_batch_start(self, batch: NRMDataBatch, batch_local_idx: int, dataloader_idx: int = 0) -> None:
        self.model.update_step_train_batch_start(self.current_epoch, self.global_step, self)


    # ---- Checkpoint-related methods ----

    def on_load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        The issue here is that the buffers can change in size so we cannot initialize them in advance and load them directly from the checkpoint
        """
        if self._trainer is not None:
            # Manually load all the loops as otherwise global_step and current epoch are not set correctly in the validation mode
            # see: https://github.com/Lightning-AI/lightning/issues/17127
            self.trainer.predict_loop.load_state_dict(checkpoint["loops"]["predict_loop"])

        self.load_state_dict(checkpoint["state_dict"], assign=True)
