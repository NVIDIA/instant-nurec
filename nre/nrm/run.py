# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import signal
import sys

import click
import click_default_group
import pytorch_lightning as pl
import torch
import yaml

from lightning_utilities.core.rank_zero import rank_zero_info
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.strategies import DDPStrategy, Strategy

# Hack: Pure import datasets to ensure dataset registry (e.g. nrm-ncore) is populated for pycena
import nre.nrm.datasets  # noqa: F401
import nre.nrm.systems

from nre.config.parse import assert_no_out_dir_override_in_resume, dump_config
from nre.config.version import get_version
from nre.nrm.config.nrm import NRMConfig, parse_typed_nrm_config
from nre.nrm.datasets.datamodule import NRMDataModule, ResumableDataModuleCallback
from nre.nrm.systems.base import BaseNRMSystem
from nre.utils.callbacks import (
    CheckpointAndExitOnSignalCallback,
    CheckpointAndExitOnSlurmTimeoutCallback,
    ForceValidateCallback,
    LearningRateMonitor,
    ModelCheckpoint,
    PreemptionInterruptException,
    TimingLogger,
    TQDMProgressBar,
    make_logger,
)
from nre.utils.misc import is_env_true, rank_zero_only, unpack_optional
from nre.utils.profiling import ProfilerBackend, ScopedTimer, ScopedTimerConfig, VerbosityLevel
from nre.viewer.lightning_viewer import maybe_create_lightning_viewer


class ScopedTimerCallback(Callback):
    """
    Callback to initialize and print the summary of the scoped timer.
    """

    def __init__(self):
        ScopedTimer.set_global_config(
            ScopedTimerConfig(
                enabled=True, verbosity=VerbosityLevel.SUMMARY, synchronize=True, profiling_backend=ProfilerBackend.NVTX
            )
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        ScopedTimer.set_step(trainer.global_step)

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        ScopedTimer.print_summary()

    def on_fit_end(self, trainer, pl_module) -> None:
        ScopedTimer.print_summary()


class MemoryProfilingCallback(Callback):
    """
    Callback to benchmark memory usage via torch.cuda.memory module
    TODO: Move this to a separate file.
    """

    def __init__(self, profiling_path: str):
        self.profiling_path = profiling_path
        # Start memory profiling immediately
        torch.cuda.memory._record_memory_history()

    def on_fit_end(self, trainer, pl_module) -> None:
        os.makedirs(self.profiling_path, exist_ok=True)
        snapshot_path = os.path.join(self.profiling_path, "memory_history.pickle")
        rank_zero_info(f"Dumping memory snapshot to {snapshot_path}")
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)


def setup_environment_and_logger(config: NRMConfig) -> logging.Logger:
    if (local_rank := os.environ.get("LOCAL_RANK")) is not None:
        torch.cuda.set_device(int(local_rank))
        logging.getLogger(__name__).info(
            f"[info] Distributed training detected. LOCAL_RANK = {local_rank}, device = {torch.cuda.current_device()}"
        )

    if is_env_true("CUDA_SYNC_DEBUG", False):
        torch.cuda.set_sync_debug_mode("warn")
        logging.getLogger(__name__).info("CUDA synchronization debug mode enabled (CUDA_SYNC_DEBUG=1)")

    logger = logging.getLogger(__name__)
    if config.verbose:
        logger.setLevel(logging.DEBUG)

    # Use the same seed for model initialization across all processes
    pl.seed_everything(config.seed, workers=True)

    return logger


def make_callbacks(config: NRMConfig, datamodule: NRMDataModule) -> list[Callback]:
    is_slurm_cluster = "PID_PATH" in os.environ

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=config.ckpt_dir,
        filename=f"epoch={{epoch:d}}-metric={{{config.checkpoint.monitor}:.2f}}",
        every_n_epochs=config.system.check_val_every_n_epoch,
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k,
        # Save an additional "last.ckpt" for convenient resuming.
        # We cannot set to "link" here since it might be wrongly linked to the TopK ckpt, not the last one.
        save_last=True,
        auto_insert_metric_name=False,
        # By default this callback will detect if the current last.ckpt is generated by itself, and if so it
        # will save last-v1.ckpt instead. Since we also have CheckpointAndExitOnSlurmTimeoutCallback, we
        # want to disable this behavior.
        enable_version_counter=False,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    callbacks = [
        model_checkpoint_callback,
        lr_monitor_callback,
        TQDMProgressBar(refresh_rate=1),
        TimingLogger(),
        # Print out the actual NRM parameters for inspection
        ModelSummary(max_depth=2),
        ResumableDataModuleCallback(datamodule),
    ]

    if config.profiling.enabled:
        if config.profiling.disable_checkpoint:
            callbacks.remove(model_checkpoint_callback)

        if config.profiling.scopedtimer:
            callbacks.append(ScopedTimerCallback())

        if config.profiling.record_memory_history:
            callbacks.append(
                MemoryProfilingCallback(
                    profiling_path=os.path.join(config.out_dir, config.run_id, "profiling"),
                )
            )

    if config.mode == "trainval" and config.force_validate:
        callbacks.append(ForceValidateCallback())

    if is_slurm_cluster:
        callbacks.append(CheckpointAndExitOnSlurmTimeoutCallback(config.ckpt_dir, os.environ["PID_PATH"]))

    if config.preempt_on_interrupt:
        callbacks.append(
            CheckpointAndExitOnSignalCallback(config.ckpt_dir, signal.SIGINT, check_signal_every_n_batches=1)
        )

    return callbacks


def launch_trainer_loop(config: NRMConfig, system: BaseNRMSystem, logger: logging.Logger) -> None:
    pl_logger = make_logger(config.logger)

    # Make the pytorch lightning callback functions
    callbacks = make_callbacks(config, system.datamodule)

    # Make a visualizer callback if enabled
    if (viewer_callback := maybe_create_lightning_viewer(system, config.viewer)) is not None:
        callbacks.append(viewer_callback)

    # Optionally, skip autocast and only use grad_scaler when using half precision
    precision = config.system.precision
    limit_train_batches: int | None = None
    limit_val_batches: int | float = 1.0
    if config.profiling.enabled:
        limit_train_batches = config.profiling.limit_train_val_batches
        limit_val_batches = config.profiling.limit_train_val_batches

    strategy: Strategy | str = config.system.strategy
    if strategy == "ddp_find_unused_parameters":
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        devices=unpack_optional(config.system.device_count),
        callbacks=callbacks,
        logger=pl_logger,
        strategy=strategy,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches if "val" in config.mode else 0.0,
        reload_dataloaders_every_n_epochs=1,  # NB: This has to be set to 1 to ensure set_rng_epoch in dataset works.
        precision=precision,
        max_epochs=config.system.max_epochs,
        check_val_every_n_epoch=config.system.check_val_every_n_epoch,
        log_every_n_steps=config.system.log_every_n_steps,
        enable_progress_bar=True,
        num_sanity_val_steps=config.system.num_sanity_val_steps,
        num_nodes=config.system.num_nodes,
    )

    ckpt_path = config.resume if (config.resume and not config.resume_weights_only) else None

    # For val/test/predict without a checkpoint but with init weights, we call the hook here (it otherwise runs from
    # system.on_train_start() during trainer.fit() only). Set call_train_from_scratch_hook_for_validation=False to
    # skip calling it for eval, e.g. if the hook has train-only side effects.
    has_full_init = bool(getattr(config.model, "init_weights_path", None))
    init_weights_paths = getattr(config.model, "init_weights_paths", None)
    if not has_full_init and init_weights_paths is not None:
        # Must stay in lockstep with KelvinModel.on_train_from_scratch_start's full-model check.
        if {"full", "tokengs"} & init_weights_paths.keys():
            has_full_init = True
    if (
        config.call_train_from_scratch_hook_for_validation
        and ckpt_path is None
        and has_full_init
        and config.mode in ("val", "test", "predict")
    ):
        system.model.on_train_from_scratch_start(system)

    if "predict" not in config.mode:
        raise ValueError(f"Only predict mode is supported in this standalone; got mode={config.mode}.")
    # Set return_predictions to False since we return primitives which is memory-consuming.
    trainer.predict(system, datamodule=system.datamodule, ckpt_path=ckpt_path, return_predictions=False)


@click.command("main")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.argument("hydra-args", nargs=-1)
def main(config_name: str, hydra_args: list[str]) -> None:
    """Main entry point for NRE-NRM training, validation, testing and prediction"""

    assert_no_out_dir_override_in_resume(hydra_args)
    config = parse_typed_nrm_config(config_name=config_name, hydra_args=hydra_args)

    # Save the parsed config at early stage
    rank_zero_only(os.makedirs)(config.config_dir, exist_ok=True)
    dump_config(os.path.join(config.config_dir, "parsed.yaml"), config)

    logger = setup_environment_and_logger(config)
    #   Other rank might have different run_ids, but this does not seem to be a problem since wandb is lazy-init and
    # will hence only take effect on rank_0.
    #   [Otherwise it might be hard to all_gather run_id since NCCL is not fully set up until this point]
    rank_zero_info("NRM RUN 🆔: %s", config.run_id)

    # Make the system
    checkpoint = None if (not config.resume_weights_only or not config.resume) else config.resume
    system = nre.nrm.systems.make(
        config.system.name,
        config,
        load_from_checkpoint=checkpoint,
    )

    try:
        launch_trainer_loop(config, system, logger)
    except PreemptionInterruptException as e:
        rank_zero_info(f"Preemption detected: {e}. Exiting with code 1.")
        sys.exit(1)


@click.command("parse_config")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.argument("hydra-args", nargs=-1)
def parse_config(config_name: str, hydra_args: list[str]) -> None:
    """Parse and print the NRM configuration to surface any configuration errors"""

    assert_no_out_dir_override_in_resume(hydra_args)
    config = parse_typed_nrm_config(config_name=config_name, hydra_args=hydra_args)

    # Convert config to a dictionary and print as YAML
    config_dict = OmegaConf.to_container(config.to_dictconfig(), resolve=True)
    print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
    print(f"\n✓ Configuration parsed successfully!")


@click.group(cls=click_default_group.DefaultGroup, default="main", default_if_no_args=True)
@click.version_option(version=str(unpack_optional(get_version(), default="version-not-available")))
def cli():
    pass


cli.add_command(main)
cli.add_command(parse_config)

if __name__ == "__main__":
    cli(show_default=True)
