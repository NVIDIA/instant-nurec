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

import logging
import os

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, TypeVar, cast

import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import do_nothing_closure
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig as PL_OptimizerLRSchedulerConfig

from libs.losses.orchestration.config import LossAggregatorBatchReturn
from libs.losses.orchestration.loss_aggregator import LossAggregator
from nre.config.base_schema import config_to_primitive
from nre.nrm.config.nrm import BaseNRMSystemConfig, NRMConfig
from nre.nrm.datasets.datamodule import NRMDataModule
from nre.nrm.models.base import BaseNRM
from nre.utils.batch import NRMDataBatch
from nre.utils.log import BatchMediaLogger
from nre.utils.misc import unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.types import Checkpoint


logger = logging.getLogger(__name__)


class BaseNRMSystem(LightningModule, ABC):
    config: BaseNRMSystemConfig
    model: BaseNRM
    last_loss_return: LossAggregatorBatchReturn | None
    datamodule: NRMDataModule

    device: torch.device  # mypy type fix for obtaininig system's device

    def __init__(self, config: NRMConfig) -> None:
        super().__init__()

        # handle the training loop iteration ourselves for the sake of flexibility
        self.automatic_optimization = False

        self.save_hyperparameters(config_to_primitive(config.to_dictconfig()))

        # save what we need from the config
        self.cached_config = config
        self.mode = config.mode
        self.max_epochs = config.system.max_epochs
        self.out_dir = config.out_dir
        self.run_id = config.run_id
        self.resume = config.resume
        self.config = config.system
        self.predict_config = config.predict

        self.datamodule = NRMDataModule(config)
        assert config.loss is not None

        # Slang could not properly dispatch disabled loss functions, giving empty tensors.
        self.loss = LossAggregator(config.loss, config.system.trainer, force_disable_cuda=True)

        self.media_logger = BatchMediaLogger(self, self.config)
        self.last_loss_return = None



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
            self.trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
            self.trainer.validate_loop.load_state_dict(checkpoint["loops"]["validate_loop"])
            self.trainer.test_loop.load_state_dict(checkpoint["loops"]["test_loop"])
            self.trainer.predict_loop.load_state_dict(checkpoint["loops"]["predict_loop"])

        # Load the remainder of the state_dict
        self.load_state_dict(checkpoint["state_dict"], assign=True)
        if "train" in self.mode:
            assert self.config.world_size == checkpoint["trainer.world_size"]

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        # For sanity check that training environment matches.
        checkpoint["trainer.world_size"] = self.config.world_size
