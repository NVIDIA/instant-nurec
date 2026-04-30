# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import glob
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

from types import FrameType
from typing import Any, Dict, Optional

import wandb

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.loggers.logger import DummyExperiment, DummyLogger, Logger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer

from nre.config.base_schema import config_to_primitive
from nre.config.logger import DummyLoggerConfig, LoggerConfigType, TensorboardLoggerConfig, WandbLoggerConfig
from nre.config.nre import NREConfig
from nre.systems.base import BaseSystem, BaseSystemSO
from nre.utils.misc import rank_zero_only, unpack_optional
from nre.utils.trainer import BroadcastExceptions


logger = logging.getLogger(__name__)


def get_latest_version(folder):
    versions = [int(pathlib.PurePath(path).name.split("_")[-1]) for path in glob.glob(f"{folder}/version_*/")]

    if len(versions) == 0:
        return -1

    versions.sort()
    return versions[-1]


def get_latest_checkpoint(out_dir, version):
    chkpts = []
    while version > -1:
        folder = os.path.join(out_dir, f"version_{version}", "checkpoints")

        latest = f"{folder}/last.ckpt"
        if os.path.exists(latest):
            return latest, version

        chkpts = glob.glob(f"{folder}/epoch=*.ckpt")

        if len(chkpts) > 0:
            break

        version -= 1

    if len(chkpts) == 0:
        return None, None

    latest = max(chkpts, key=os.path.getctime)

    return latest, version


class TimingLogger(Callback):
    """
    Simple callback to log the elapsed time for training and validation.
    Specifically, we measure the total time spent in `train_epoch`s and the total time spent in `validation_epoch`s.

    Note: we can't `self.log()` inside `on_train/validation_end` so we use `self.logger.experiment.log()`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_elapsed_time_s: float = 0.0
        self.validation_elapsed_time_s: float = 0.0

        self.train_epoch_start_end_timer_s: Optional[float] = None
        self.validation_epoch_start_end_timer_s: Optional[float] = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.train_epoch_start_end_timer_s = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        now = time.time()
        assert self.train_epoch_start_end_timer_s is not None, (
            f"{self.__class__.__name__}: contract error, timer is None"
        )
        assert now > self.train_epoch_start_end_timer_s, (
            f"{self.__class__.__name__}: contract error, time went backwards"
        )
        self.train_elapsed_time_s += now - self.train_epoch_start_end_timer_s

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.validation_epoch_start_end_timer_s = time.time()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        now = time.time()
        assert self.validation_epoch_start_end_timer_s is not None, (
            f"{self.__class__.__name__}: contract error, timer is None"
        )
        assert now > self.validation_epoch_start_end_timer_s, (
            f"{self.__class__.__name__}: contract error, time went backwards"
        )
        self.validation_elapsed_time_s += now - self.validation_epoch_start_end_timer_s

    @staticmethod
    def log_scalar(name: str, value: float | int, pl_module: LightningModule, step: Optional[int] = None) -> None:
        logger = pl_module.logger
        match logger:
            case WandbLogger():
                logger.experiment.log({name: value}, step=step)
            case TensorBoardLogger():
                logger.experiment.add_scalar(name, value, global_step=step)
            case DummyLogger():
                pass
            case _:
                raise ValueError(f"Unsupported logger type {type(logger)=}")

        if isinstance(pl_module, BaseSystemSO):
            pl_module.collect_metric(name, value)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_end(trainer, pl_module)
        TimingLogger.log_scalar("train/elapsed_time_s", self.train_elapsed_time_s, pl_module)
        logger.info(f"Train elapsed time: {self.train_elapsed_time_s:.2f} seconds")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)
        TimingLogger.log_scalar("test/elapsed_time_s", self.validation_elapsed_time_s, pl_module)
        logger.info(f"Validation elapsed time: {self.validation_elapsed_time_s:.2f} seconds")


class SmoothedProgressBar(TQDMProgressBar):
    """TQDMProgressBar that uses tqdm's EMA smoothing for it/s display.

    This will more accurately reflect the actual training rate, since that changes
    as the model converges and strategy changes.

    PyTorch Lightning's default TQDMProgressBar bypasses tqdm.update() by
    directly setting bar.n and calling bar.refresh(). This means tqdm's EMA
    (controlled by the smoothing parameter) is never fed, and the displayed
    rate always falls back to the global average (n / elapsed).

    This subclass overrides on_train_batch_end to call bar.update() instead,
    which properly feeds the EMA and gives a responsive it/s reading.
    """

    SMOOTHING_WINDOW_99P = 20  # ~N iterations contribute 99% of the it/s

    def init_train_tqdm(self) -> Tqdm:
        smoothing_alpha = 1 - 0.01 ** (1 / self.SMOOTHING_WINDOW_99P)

        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=smoothing_alpha,
            bar_format=self.BAR_FORMAT,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            # Use update() instead of directly setting .n so tqdm's EMA is fed
            delta = n - self.train_progress_bar.n
            if delta > 0:
                self.train_progress_bar.update(delta)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            # Use update() instead of directly setting .n so tqdm's EMA is fed
            delta = n - self.val_progress_bar.n
            if delta > 0:
                self.val_progress_bar.update(delta)
            self.val_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))


class _EpochEndModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that ensures last.ckpt captures post-batch-end model state.

    PL calls callback on_train_batch_end BEFORE LightningModule on_train_batch_end.
    When every_n_train_steps coincides with the last training step, the checkpoint
    saved during the callback misses model updates (densification, SH degree
    increment) that run in the LightningModule hook.  The epoch-end save
    (save_on_train_epoch_end=True) should fix this, but PL suppresses it via
    _last_global_step_saved deduplication.  We clear that flag so the epoch-end
    save always proceeds with the fully-updated model state.
    """

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if hasattr(self, "_last_global_step_saved"):
            self._last_global_step_saved = -1
        super().on_train_epoch_end(trainer, pl_module)


def make_callbacks(config: NREConfig) -> list[Callback]:
    callbacks = [
        _EpochEndModelCheckpoint(
            dirpath=config.ckpt_dir,
            filename="{step:06d}",
            save_on_train_epoch_end=config.checkpoint.save_on_train_epoch_end,
            every_n_train_steps=config.checkpoint.every_n_train_steps,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_top_k=config.checkpoint.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        SmoothedProgressBar(refresh_rate=1),
        TimingLogger(),
    ]

    if os.environ.get("PID_PATH", ""):
        callbacks.append(
            CheckpointAndExitOnSlurmTimeoutCallback(
                config.ckpt_dir if config.checkpoint.save_on_preemption else None, os.environ["PID_PATH"]
            )
        )

    if config.mode == "trainval" and config.force_validate:
        callbacks.append(ForceValidateCallback())

    return callbacks


class ForceValidateCallback(Callback):
    """
    Forces validation to happen at least once at the end of training if the mode is "trainval". This is useful for resuming
    preemptable runs since otherwise resuming a checkpoint interrupted after training but before validation finished
    will not re-run validation.
    """

    def __init__(self):
        self.has_validated = False

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.has_validated = True

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.getLogger(__name__).info(f"Validation has not yet run, forcing validation")

        if not self.has_validated:
            assert isinstance(trainer.lightning_module, BaseSystem)
            trainer.validate(trainer.lightning_module, datamodule=trainer.lightning_module.datamodule)


class PreemptionInterruptException(Exception):
    """
    Exception to be raised when the training process is preempted (e.g. by a signal).
    This will be raised from the CheckpointAndExitOnSignalCallback and its subclasses.
    """

    def __init__(self, message: str = "Preempted"):
        super().__init__(message)
        self.message = message


class CheckpointAndExitOnSignalCallback(Callback):
    """
    A callback to automatically save a checkpoint and exit the training process when a signal is received.
    """

    def __init__(
        self,
        ckpt_dir_path: str | None,
        signal_type: int,
        check_signal_every_n_batches: int = 5,
    ):
        self.preempting = False
        self.ckpt_dir_path = ckpt_dir_path
        self.check_signal_every_n_batches = check_signal_every_n_batches
        self.signal_type = signal_type
        signal.signal(self.signal_type, self._mark_preempting)

    def _mark_preempting(self, sig: int, frame: Optional[FrameType]) -> None:
        logger.info(f"Received signal {sig}, marking run as preempting")
        self.preempting = True

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if batch_idx % self.check_signal_every_n_batches == 0:
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
        if batch_idx % self.check_signal_every_n_batches == 0:
            self._check_preempting(trainer)

    def _check_preempting(self, trainer: Trainer) -> None:
        # Broadcast the preempting state:
        self.preempting = trainer.strategy.broadcast(self.preempting, src=0)

        if self.preempting:
            logger.info("Saving checkpoint and exiting in response to the preempting signal")

            # Set both the train and val loop to a clean state:
            # - For the train loop, mock completion of the batch by advancing "completed" to "processed".
            #   This is mainly to fix a bug in PL2.4 where the current batch might be processed again.
            #   Fixed in PL2.6 via https://github.com/Lightning-AI/pytorch-lightning/pull/20379
            # - For the val loop, simply reset so it could be re-run.
            # Here we're free to alter PL's internal state since we're exiting anyway.
            train_progress = trainer.fit_loop.epoch_loop.batch_progress
            train_progress.total.completed = train_progress.total.processed  # type: ignore[attr-defined]
            train_progress.current.completed = train_progress.current.processed  # type: ignore[attr-defined]
            val_progress = trainer.fit_loop.epoch_loop.val_loop.batch_progress
            val_progress.reset()

            if self.ckpt_dir_path is not None:
                # First write to a temporary checkpoint, then rename it to the final checkpoint.
                # This is to ensure that the checkpoint is not corrupted in case of an unexpected exit
                # (e.g. job gets killed while saving -- note this is less likely to happen since we have 300s but just in case).
                trainer.save_checkpoint(f"{self.ckpt_dir_path}/last.ckpt.tmp")
                if trainer.is_global_zero:
                    os.rename(f"{self.ckpt_dir_path}/last.ckpt.tmp", f"{self.ckpt_dir_path}/last.ckpt")
                logger.info(f"Checkpoint saved to {self.ckpt_dir_path}/last.ckpt")

            # Finish wandb **after** checkpoint is saved since wandb could hang while uploading files
            # causing jobs to timeout.
            if trainer.is_global_zero and wandb.run is not None:
                wandb.run.mark_preempting()
                wandb.finish(exit_code=1)

            # We cannot use trainer.should_stop = True because it will trigger additional validation runs
            raise PreemptionInterruptException()

        # Due to unknown reason, when using the mamba backbone, the signal cannot be detected properly.
        # This might be due to mamba JIT's complicated implementation destroys the signal handler.
        # We find that setting this up again regularly can fix the issue.
        signal.signal(self.signal_type, self._mark_preempting)


class CheckpointAndExitOnSlurmTimeoutCallback(CheckpointAndExitOnSignalCallback):
    """
    A callback to automatically save a checkpoint and exit the training process on Slurm cluster.
    This is supposed to be used in junction with the cluster toolbox, so that the job could be properly requeued.
    The process is done by writing the pid of the training process to the specified path, which is used by the slurm script
    to send a SIGUSR1 signal to the training process before the job times out. The training process then marks the
    wandb run as preempting, which will cause the wandb agent to requeue the run, and terminate with a non-zero exit code
    (which is needed for wandb to requeue the run).

    The entire process follows the following execution flow:

    --------------> X                      X
    (TRAINING)      (300s before 4h limit) (SIGUSR1 is passed through bash to training processes)

    ----------------------------> X
    (After current batch ends)    (This callback marks preempting=True, saves the checkpoint, and marks trainer to stop)

    ------------------------------------> X ---------------(around 10s poll)------------> X
    (Trainer finishes cleaning up)        (Process does not exist anymore)                (Call scontrol requeue)

    X
    (requeue happens and current slurm job is terminated)

    This has been tested to work with multi-node training.

    Known Limitations:
    - Even when we set check_signal_every_n_batches=1, in cases where the dataloader hangs (e.g. due to slow S3 downloading) and
    we cannot get to on_**_batch_end step, the checkpoints will still not be saved. Fortunately such hanging usually happens just
    due to latency at the beginning of each epoch, so no work will be lost.
    """

    def __init__(
        self,
        ckpt_dir_path: str | None,
        pid_path: str,
        check_signal_every_n_batches: int = 5,
    ):
        # Note that you might see the following in the logs:
        #   SLURM auto-requeueing enabled. Setting signal handlers.
        # This message originates from PL's SlurmClusterEnvironment plugin, which gets automatically set-up.
        # That plugin is however not use-able since 'scontrol requeue' is not available in the docker container that we run in.
        # Here we make it disfunctional by completely **rewriting** the signal handler for SIGUSR1.
        logger.info(
            f"Registering signal handler for SIGUSR1, replacing existing handler {signal.getsignal(signal.SIGUSR1)}"
        )

        super().__init__(ckpt_dir_path, signal.SIGUSR1, check_signal_every_n_batches)

        # Write the pid of the training process so the external driving script can send SIGUSR1 to it.
        # Signal is only sent to the process with global rank 0 (hence the PID writing); state is broadcast by the DDP strategy.
        if int(os.environ["SLURM_PROCID"]) == 0:
            pid = str(os.getpid())
            logger.info(f"[Main Rank] Writing pid {pid} to {pid_path}")
            with open(pid_path, "w") as f:
                f.write(pid)


class VersionedCallback(Callback):
    def __init__(self, save_root, version=None, use_version=True):
        self.save_root = save_root
        self._version = version
        self.use_version = use_version

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        existing_versions = []
        if os.path.isdir(self.save_root):
            for f in os.listdir(self.save_root):
                bn = os.path.basename(f)
                if bn.startswith("version_"):
                    dir_ver = os.path.splitext(bn)[0].split("_")[1].replace("/", "")
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1

    @property
    def savedir(self):
        if not self.use_version:
            return self.save_root
        return os.path.join(
            self.save_root, self.version if isinstance(self.version, str) else f"version_{self.version}"
        )


class CodeSnapshotCallback(VersionedCallback):
    def __init__(self, save_root: str, version=None, use_version=True):
        super().__init__(save_root, version, use_version)

    def get_file_list(self):
        return [
            b.decode()
            for b in set(subprocess.check_output("git ls-files", shell=True).splitlines())
            | set(subprocess.check_output("git ls-files --others --exclude-standard", shell=True).splitlines())
        ]

    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        try:
            self.save_code_snapshot()
        except:
            logger.warning(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


def _make_wandb_logger(config: WandbLoggerConfig) -> WandbLogger:
    logger = WandbLogger(
        name=config.run_name,
        save_dir=config.save_dir,
        offline=config.offline,
        id=config.run_id,
        project=config.project,
        anonymous=config.anonymous,
        log_model=config.log_model,
        group=config.group,
        tags=config.tags,
        job_type=config.job_type,
        entity=config.entity,
    )

    if not isinstance(logger.experiment, DummyExperiment):
        config.run_name = logger.experiment.name

        wandb_config: Dict = config_to_primitive(config.to_dictconfig())

        # Allow user to quickly associate and find the original maglev workflows that created this run.
        # MagLev environment variables are documented here:
        # https://maglev.nvda.ai/docs/components/workflows/defining-workflows/setting-up-environment-variables.md
        if os.environ.get("WORKFLOW_NAME", ""):
            base_url = "https://maglev.nvda.ai/ide/workflows"
            wf = os.environ.get("WORKFLOW_NAME", "")
            run = os.environ.get("WORKFLOW_RUN_ID", "")
            task = os.environ.get("WORKFLOW_TASK_NAME", "")
            job = os.environ.get("WORKFLOW_JOB_ID", "")
            wandb_config["maglev_url"] = f"{base_url}/workflow/{wf}/run/{run}/task/{task}/job/{job}"

        logger.experiment.config.update(wandb_config, allow_val_change=True)

    return logger


def _make_tensorboard_logger(config: TensorboardLoggerConfig) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=config.save_dir,
        name=config.run_name,
        version=config.run_id,
        log_graph=config.log_graph,
        default_hp_metric=config.default_hp_metric,
        prefix=config.prefix,
    )


def make_logger(config: LoggerConfigType) -> Logger:
    if isinstance(config, DummyLoggerConfig):
        return DummyLogger()

    # Wandb expects the folder to exist already
    # However, for non-rank-zero workers the logger is already replaced by DummyLogger so no need to create dir.
    rank_zero_only(os.makedirs)(config.save_dir, exist_ok=True)

    if isinstance(config, WandbLoggerConfig):
        return _make_wandb_logger(config)
    elif isinstance(config, TensorboardLoggerConfig):
        return _make_tensorboard_logger(config)
    else:
        raise TypeError(f"Unknown config type {type(config)=}.")
