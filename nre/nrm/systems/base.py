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
from nre.nrm.utils.optim import ProgressBasedLRScheduler, configure_optimizers
from nre.utils.batch import NRMDataBatch
from nre.utils.log import BatchMediaLogger
from nre.utils.misc import unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.trainer import BroadcastExceptions
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

    def configure_optimizers(self) -> list[PL_OptimizerLRSchedulerConfig]:
        """
        Returns all system-associated as well as model-associated optimizers
        """
        system_optims = configure_optimizers(self.config.to_dictconfig(), self.model)
        return [cast(PL_OptimizerLRSchedulerConfig, x) for x in system_optims]

    # ---- Training loop methods ----

    def on_train_epoch_start(self) -> None:
        # Setup pytorch seed for different ranks (mainly for model runtime behaviours)
        if torch.distributed.is_initialized():
            # This is the global seed set by pl.seed_everything:
            # https://lightning.ai/docs/fabric/2.5.1/api/utilities.html
            base_seed = int(os.environ.get("PL_GLOBAL_SEED", 0))
            rank = torch.distributed.get_rank()
            torch.manual_seed(base_seed + self.current_epoch * torch.distributed.get_world_size() + rank)

    def on_train_batch_start(self, batch: NRMDataBatch, batch_local_idx: int):
        additional_parameters: dict[str, torch.Tensor] = self.model.update_step_train_batch_start(
            self.current_epoch, self.global_step, self
        )
        additional_parameters |= self.loss.update_step_train_batch_start(self.current_epoch, self.global_step, self)
        """ We iterate over the additional parameters, if the same parameter group already exist we
            append the parameter to the group, else we initialize a new parameter group. """
        for key, value in additional_parameters.items():
            for param_group in self.trainer.optimizers[0].param_groups:
                if param_group["name"] == key:
                    param_group["params"].append(value)
                    break
            else:
                self.trainer.optimizers[0].add_param_group({"params": value, "name": key})

    def training_step(self, batch: NRMDataBatch, batch_local_idx: int):
        """Evaluates single iteration of the external training loop (can be overwritten in derived classes to modify)"""

        # Due to a barrier inside manual_backward below,
        # we need to ensure that either all process or none enters it.
        # If an exception is raised, it'll be broadcasted to all processes
        # and training will abort instead of hang.
        with BroadcastExceptions(self.trainer):
            # 0. Construct list of optimizers (also for single optimizer case)
            pl_optimizers = self.optimizers(
                # PL optimizers will automatically handle precision and profiling
                use_pl_optimizer=True
            )
            pl_optimizers = pl_optimizers if isinstance(pl_optimizers, list) else [pl_optimizers]

            # 1. Invalidate gradients in all optimizers
            for pl_optimizer in pl_optimizers:
                # note: this assumes to set None (not zero-values) to '.grad' properties of parameter tensors, but we can't
                # request this explicitly here as the "default" 'set_to_none' keyword property is not supported by all
                # optimizers, e.g., 'apex.FusedAdam.zero_grad()' is doesn't respect a 'set_to_none' keyword parameter
                pl_optimizer.optimizer.zero_grad()

            # 2. Evaluate *system-level* training losses for given batch
            with ScopedTimer("BaseNRMSystem/training_step/full_forward"):
                loss_return = self.training_losses(batch, batch_local_idx)
            self.last_loss_return = loss_return

        # Broadcast exceptions raised during optimization.
        # Note: this is adding synchronization overhead, which might be significant if the
        #       workload is uneven across processes. Ideally one would only broadcast exceptions
        #       when the processes are idle, e.g., right after backward pass
        #       (where the gradients are synchronized).
        #       I couldn't implement this idea. We'd have to call self.manual_backward() with a possibly
        #       non-existing loss_return. I've tried to create dummy total_value Tensor:
        #           1. empty tensor: resulted in error (no gradient found in element 0)
        #           2. 0.0 value, with gradient: assertion error
        with BroadcastExceptions(self.trainer):
            # 3. Backprop gradients of *system-level* training losses to parameters
            # Note 1: there's a inter-process barrier here to gather the gradients of
            #         all subprocesses.  All processes or none must call this function
            #         or else training will hang indefinitely.
            # Note 2: we're assuming that no exceptions occur in the code
            #         leading to the barrier inside manual_backward. If they do, process will hang.
            #         I don't think this is likely to happen, though (famous last words).
            with ScopedTimer("BaseNRMSystem/training_step/manual_backward"):
                self.manual_backward(loss_return.total_value)

            # 4. Step optimizers to update parameters
            # Manually maintain by ourselves the trainer's optimizer step progress, to temporarily fix https://github.com/Lightning-AI/pytorch-lightning/issues/17958
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_ready()
            self.trainer.profiler.start("optimizer_step")  # type: ignore[attr-defined]

            for pl_optimizer in pl_optimizers:
                # skip step if none of the parameters has associated gradients after backpropagation (e.g., because all parameters are frozen)
                # circumventing gradient scaler shortcomings not able to handle this setting
                if all([p.grad is None for group in pl_optimizer.optimizer.param_groups for p in group["params"]]):
                    continue

                if pl_optimizer._strategy is not None:
                    pl_optimizer._strategy.optimizer_step(pl_optimizer._optimizer, do_nothing_closure)

            self.trainer.profiler.stop("optimizer_step")  # type: ignore[attr-defined]
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.increment_completed()

            # 5. Handle lr-schedulers
            for lr_scheduler_config in self.trainer.strategy.lr_scheduler_configs:
                # check if we need to step this lr-scheduler / exit early if not
                match interval := lr_scheduler_config.interval:
                    case "epoch":
                        if not (
                            self.trainer.is_last_batch
                            and (self.trainer.current_epoch + 1) % lr_scheduler_config.frequency == 0
                        ):
                            continue  # step not required
                    case "step":
                        if not ((batch_local_idx + 1) % lr_scheduler_config.frequency == 0):
                            continue  # step not required
                    case _:
                        raise RuntimeError(f"Invalid scheduler interval '{interval}'")

                # step lr-scheduler depending on type
                if not isinstance(
                    scheduler := lr_scheduler_config.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if isinstance(scheduler, ProgressBasedLRScheduler):
                        assert interval == "step"
                        scheduler.set_progress(
                            self.trainer.current_epoch,
                            unpack_optional(self.trainer.max_epochs),
                            batch_local_idx,
                            int(self.trainer.num_training_batches),
                        )

                    # step a regular scheduler
                    scheduler.step()
                else:
                    # incorporate the monitored metric for ReduceLROnPlateau
                    assert (monitor := lr_scheduler_config.monitor) is not None, (
                        "Missing metric to monitor for ReduceLROnPlateau scheduler"
                    )

                    if (monitor_metric := self.trainer.callback_metrics.get(monitor)) is not None:
                        # step the scheduler with available metric
                        scheduler.step(monitor_metric)
                    else:
                        # monitored metric is not available - warn / error out depending on strict flag
                        if lr_scheduler_config.strict:
                            raise RuntimeError(f"Missing monitored metric {monitor} for ReduceLROnPlateau scheduler")
                        else:
                            logger.warning(
                                f"Missing monitored metric {monitor} for ReduceLROnPlateau scheduler - skipping scheduler update"
                            )

    @abstractmethod
    def training_losses(self, batch: NRMDataBatch, batch_local_idx: int) -> LossAggregatorBatchReturn:
        """Evaluate system-specific losses for a single training batch"""
        ...

    def on_train_batch_end(
        self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: NRMDataBatch, batch_local_idx: int
    ):
        super().on_train_batch_end(outputs, batch, batch_local_idx)
        # Compute a unique index for easy comparing across different runs.
        self.media_logger.flush_logged_media("train", media_step=self.current_epoch * 10000 + batch_local_idx)
        self.last_loss_return = None

    # ---- Validation loop methods ----

    def on_validation_start(self) -> None:
        self.val_dir = os.path.join(self.out_dir, self.run_id, "val")

    def on_validation_batch_start(self, batch: NRMDataBatch, batch_local_idx: int, dataloader_idx: int = 0) -> None:
        self.model.update_step_train_batch_start(self.current_epoch, self.global_step, self)
        torch.cuda.empty_cache()

    def on_validation_batch_end(
        self,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: NRMDataBatch,
        batch_local_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_validation_batch_end(outputs, batch, batch_local_idx, dataloader_idx)
        self.media_logger.save_logged_videos(
            os.path.join(self.out_dir, self.run_id, "val", f"epoch-{self.current_epoch:03d}"),
            f"r{self.global_rank:02d}-b{batch_local_idx:04d}",
        )
        # Overwrite previous epoch results.
        self.media_logger.save_logged_ply_point_clouds(
            os.path.join(self.out_dir, self.run_id, "val", "ply-point-clouds"),
            f"r{self.global_rank:02d}-b{batch_local_idx:04d}",
        )
        # Compute a unique index for easy comparing across different runs.
        self.media_logger.flush_logged_media("val", media_step=self.current_epoch * 10000 + batch_local_idx)

    # ---- Test loop methods ----

    def on_test_start(self) -> None:
        self.test_dir = os.path.join(self.out_dir, self.run_id, "test")

    def on_test_batch_start(self, batch: NRMDataBatch, batch_local_idx: int, dataloader_idx: int = 0) -> None:
        self.model.update_step_train_batch_start(self.current_epoch, self.global_step, self)

    def on_predict_batch_start(self, batch: NRMDataBatch, batch_local_idx: int, dataloader_idx: int = 0) -> None:
        self.model.update_step_train_batch_start(self.current_epoch, self.global_step, self)

    def on_test_batch_end(
        self,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: NRMDataBatch,
        batch_local_idx: int,
        dataloader_idx: int = 0,
    ):
        super().on_test_batch_end(outputs, batch, batch_local_idx)
        self.media_logger.save_logged_videos(
            os.path.join(self.out_dir, self.run_id, "test", f"epoch-{self.current_epoch:03d}"),
            f"r{self.global_rank:02d}-b{batch_local_idx:04d}",
        )
        self.media_logger.flush_logged_media("test", media_step=batch_local_idx)

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
