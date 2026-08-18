# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import signal
import threading

from pathlib import Path
from types import FrameType
from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


class PreemptionInterrupt(RuntimeError):
    """Raised after a requested preemption checkpoint is safely published."""


class AtomicCheckpointAndExitOnSignalCallback(Callback):
    """Checkpoint after the current batch and exit when a signal is received.

    The signal handler only sets a flag. All checkpoint work happens at a
    Lightning batch boundary, collectively across ranks. The checkpoint is
    written to ``<path>.tmp`` and atomically renamed on rank zero, allowing an
    outer host/Slurm script to requeue the job without requiring ``scontrol``
    inside the training container.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        signal_type: int | None = signal.SIGUSR1,
        check_every_n_batches: int = 1,
    ) -> None:
        super().__init__()
        if check_every_n_batches < 1:
            raise ValueError("check_every_n_batches must be at least 1")
        self.checkpoint_path = Path(checkpoint_path)
        self.signal_type = signal_type
        self.check_every_n_batches = check_every_n_batches
        self.preempting = False
        self._previous_handler: Any = None
        self._signal_handler = self._mark_preempting
        self._install_signal_handler()

    def _install_signal_handler(self) -> None:
        if self.signal_type is None or threading.current_thread() is not threading.main_thread():
            return
        if self._previous_handler is None:
            self._previous_handler = signal.getsignal(self.signal_type)
        signal.signal(self.signal_type, self._signal_handler)

    def _mark_preempting(self, signum: int, frame: FrameType | None) -> None:
        del signum, frame
        # Keep the Python signal handler side-effect-free apart from the flag;
        # logging and I/O happen at the next synchronized batch boundary.
        self.preempting = True

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        del trainer, pl_module, stage
        if (
            self.signal_type is not None
            and self._previous_handler is not None
            and threading.current_thread() is threading.main_thread()
            and signal.getsignal(self.signal_type) == self._signal_handler
        ):
            signal.signal(self.signal_type, self._previous_handler)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch
        if batch_idx % self.check_every_n_batches == 0:
            self._check_preempting(trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module, outputs, batch, dataloader_idx
        if batch_idx % self.check_every_n_batches == 0:
            self._check_preempting(trainer)

    def _check_preempting(self, trainer: Trainer) -> None:
        # A host launcher normally signals rank zero, but reduce across all
        # ranks as well so forwarding the signal to any worker is sufficient.
        self.preempting = trainer.strategy.reduce_boolean_decision(self.preempting, all=False)
        if not self.preempting:
            # Some JIT extensions replace process signal handlers. Reinstalling
            # it at batch boundaries keeps preemption reliable.
            self._install_signal_handler()
            return

        logger.warning("Saving atomic preemption checkpoint to %s.", self.checkpoint_path)

        # on_train_batch_end runs after ``processed`` but before ``completed``
        # advances. Make the checkpoint represent a completed batch so
        # Lightning does not replay it. Validation is deliberately restarted
        # from its beginning because epoch metrics cannot be resumed exactly.
        train_progress = trainer.fit_loop.epoch_loop.batch_progress
        train_progress.total.completed = train_progress.total.processed
        train_progress.current.completed = train_progress.current.processed
        trainer.fit_loop.epoch_loop.val_loop.batch_progress.reset()

        if trainer.is_global_zero:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.strategy.barrier("preemption_checkpoint_directory")

        temporary_path = Path(f"{self.checkpoint_path}.tmp")
        trainer.save_checkpoint(temporary_path)
        if trainer.is_global_zero:
            os.replace(temporary_path, self.checkpoint_path)
        trainer.strategy.barrier("preemption_checkpoint_publish")

        if trainer.is_global_zero:
            experiment = getattr(trainer.logger, "experiment", None)
            mark_preempting = getattr(experiment, "mark_preempting", None)
            if callable(mark_preempting):
                mark_preempting()

        raise PreemptionInterrupt(f"checkpoint saved to {self.checkpoint_path}")
