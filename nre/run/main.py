# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

import click
import pytorch_lightning as pl
import torch

import nre.systems

from nre import __version__
from nre.config.nre import NREConfig
from nre.config.parse import assert_no_out_dir_override_in_resume, dump_config, parse_typed_config
from nre.models.utils import MixedPrecisionNoCastPlugin
from nre.systems.base import BaseSystem, DDPStrategySO
from nre.utils.callbacks import PreemptionInterruptException, make_callbacks, make_logger
from nre.utils.misc import is_env_true, rank_zero_info, rank_zero_only, unpack_optional
from nre.utils.prober import get_global_prober
from nre.utils.profiling import ScopedTimer, create_profiler
from nre.utils.trainer import adjust_step_for_world_size
from nre.viewer.lightning_viewer import maybe_create_lightning_viewer


def setup_environment_and_logger(config: NREConfig) -> logging.Logger:
    # Configure PyTorch matmul precision for better performance
    torch.set_float32_matmul_precision(config.matmul_precision)

    if (local_rank := os.environ.get("LOCAL_RANK")) is not None:
        torch.cuda.set_device(int(local_rank))
        logging.getLogger(__name__).info(
            f"[info] Distributed training detected. LOCAL_RANK = {local_rank}, device = {torch.cuda.current_device()}"
        )

    if is_env_true("CUDA_SYNC_DEBUG", False):
        torch.cuda.set_sync_debug_mode("warn")
        logging.getLogger(__name__).info("CUDA synchronization debug mode enabled (CUDA_SYNC_DEBUG=1)")

    # If set, configure torch's default allocation block size - appending to any existing key:value configs.
    if (split_size := config.system.max_split_size_mb) is not None:
        new_entry = f"max_split_size_mb:{split_size}"
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_entry if conf is None else ",".join([conf, new_entry])

    logger = logging.getLogger(__name__)
    if config.verbose:
        logger.setLevel(logging.DEBUG)

    if config.version != __version__:
        logger.warning(f"Using config for different NRE version ({config.version=}) than current {__version__=}.")

    # Use the same seed for model initialization across all processes
    pl.seed_everything(config.seed, workers=True)

    logger.info("Setting up TensorProber with config")
    get_global_prober().configure(config.prober)

    if config.scopedtimer is not None:
        ScopedTimer.set_global_config(config.scopedtimer)
        if config.scopedtimer.logfile is not None and config.scopedtimer.enabled:
            logfilename = (
                config.scopedtimer.logfile
                if os.path.isabs(config.scopedtimer.logfile)
                else os.path.join(config.out_dir, config.run_id, config.scopedtimer.logfile)
            )
            ScopedTimer.set_global_logfile(logfilename)
        else:
            ScopedTimer.set_global_print_func(logger.info)

    return logger


def launch_trainer_loop(config: NREConfig, system: BaseSystem, logger: logging.Logger) -> None:
    pl_logger = make_logger(config.logger)

    # Make the pytorch lightning callback functions
    callbacks = make_callbacks(config)

    # Make a visualizer callback if enabled
    if (viewer_callback := maybe_create_lightning_viewer(system, config.viewer)) is not None:
        callbacks.append(viewer_callback)

    # Optionally, skip autocast and only use grad_scaler when using half precision
    precision = config.trainer.precision
    plugins = config.trainer.plugins

    # Optionally, skip autocast and only use grad_scaler when using half precision
    if config.trainer.precision == "16-no-cast":
        precision = "16-mixed"  # Change it back to `16` or `16-mixed`, otherwise PL complains.
        if plugins is None:
            plugins = []
        plugins.append(MixedPrecisionNoCastPlugin("16-mixed", "cuda"))

    profiler = create_profiler(config.to_dictconfig()) if config.profiling.enabled else None

    if config.trainer.world_size > 1:
        # Customize DDP strategy for Scene Optimization
        # TODO: We should not find unused parameters here. Check later if there is a way around it. Maybe by doing some dummy operation or setting LR to 0.
        import datetime

        nccl_timeout = datetime.timedelta(minutes=config.trainer.nccl_timeout_minutes)
        strategy = DDPStrategySO(find_unused_parameters=True, timeout=nccl_timeout)  # type: ignore
    else:
        strategy = "auto"  # type: ignore

    trainer = pl.Trainer(
        devices=unpack_optional(config.trainer.device_count),
        callbacks=callbacks,
        logger=pl_logger,
        strategy=strategy,
        limit_val_batches=1.0 if "val" in config.mode else 0.0,
        profiler=profiler,
        reload_dataloaders_every_n_epochs=system.update_n_epochs,
        precision=precision,
        plugins=plugins,
        max_epochs=config.trainer.max_epochs,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        val_check_interval=config.trainer.val_check_interval,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_progress_bar=config.trainer.enable_progress_bar,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        num_nodes=config.trainer.num_nodes,
        detect_anomaly=config.trainer.detect_anomaly,
    )

    if config.resume is not None and not os.path.exists(config.resume):
        raise FileNotFoundError(f"Checkpoint {config.resume=} to be resumed does not exist")

    if "train" in config.mode or "trainval" in config.mode:
        if config.resume and not config.resume_weights_only:
            trainer.fit(system, datamodule=system.datamodule, ckpt_path=config.resume)
        else:
            trainer.fit(system, datamodule=system.datamodule)
    elif "val" in config.mode:
        trainer.validate(system, datamodule=system.datamodule, ckpt_path=config.resume)
    elif "test" in config.mode:
        trainer.test(system, datamodule=system.datamodule, ckpt_path=config.resume)


@click.command("main")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.argument("hydra-args", nargs=-1)
def main(
    config_name: str,
    hydra_args: list[str],
) -> None:
    """Main entry point for Neural Reconstruction Engine training, validation and testing"""

    assert_no_out_dir_override_in_resume(hydra_args)
    config = parse_typed_config(config_name=config_name, hydra_args=hydra_args)

    # Save the parsed config now in case we need to verify it.
    rank_zero_only(os.makedirs)(config.config_dir, exist_ok=True)
    dump_config(os.path.join(config.config_dir, "parsed.yaml"), config)
    logger = setup_environment_and_logger(config)

    # Other rank might have different run_ids, but this does not seem to be a problem since wandb is lazy-init and
    # will hence only take effect on rank_0. Otherwise it might be hard to all_gather run_id since NCCL is not
    # fully set up until this point.
    rank_zero_info("RUN 🆔: %s", config.run_id)

    # By default Pytorch Lightning stores the checkpoints at the end of validation. If no validation is performed
    # the checkpoints are never stored. Checkpoint callback has a parameter 'save_on_train_epoch_end' that is used
    # to circumvent this and we set it to true. However the parameter is ignored if 'check_val_every_n_epoch' is not 1.
    # This ugly hack sets 'check_val_every_n_epoch' to 1 when we are not doing validation (the parameter is not used anyways)
    if "val" not in config.mode:
        config.trainer.check_val_every_n_epoch = 1
        logger.warning(
            "The parameter 'check_val_every_n_epoch' overridden to 1 to enable storing checkpoints. "
            "This has no effect on the training process and can be safely ignored."
        )

    # If checkpointing is enabled, check that the parameters are compatible. If 'every_n_epochs == None', PL saves a checkpoint
    # at the end of every epoch (equivalent to 'every_n_epochs == 1').
    if config.checkpoint.every_n_train_steps is not None and config.checkpoint.every_n_train_steps > 0:
        # Adjust the checkpointing interval to ensure that the last checkpoint is saved
        # Estimate the number of checkpoints to save
        n_global_steps = config.trainer.max_epochs * config.dataset.n_samples_per_epoch
        n_checkpoints_to_save = (
            n_global_steps + config.checkpoint.every_n_train_steps - 1
        ) // config.checkpoint.every_n_train_steps
        # Recompute the checkpoint interval to ensure that the last checkpoint is saved
        n_actual_steps = config.trainer.max_epochs * adjust_step_for_world_size(
            config.trainer, config.dataset.n_samples_per_epoch
        )
        config.checkpoint.every_n_train_steps = (n_actual_steps + n_checkpoints_to_save - 1) // n_checkpoints_to_save
        logger.info(f"Number of checkpoints to save: {n_checkpoints_to_save}")
        logger.info(f"Number of training steps: {n_actual_steps}")
        logger.info(f"Checkpoint interval: {config.checkpoint.every_n_train_steps}")

        if n_actual_steps % config.checkpoint.every_n_train_steps != 0:
            raise ValueError(
                f"The product of 'max_epochs' and 'n_samples_per_epoch' ({n_actual_steps}) is not divisible by "
                f"'checkpoint.every_n_train_steps' ({config.checkpoint.every_n_train_steps}). This means no "
                f"checkpoint will be saved at the end of training."
            )
        if config.checkpoint.every_n_train_steps > n_actual_steps:
            raise ValueError(
                f"The parameter 'checkpoint.every_n_train_steps' ({config.checkpoint.every_n_train_steps}) is larger "
                f"than the product of 'max_epochs' ({config.trainer.max_epochs}) and 'n_samples_per_epoch' "
                f"({config.dataset.n_samples_per_epoch}). This means no checkpoint will be saved at all."
            )

    # Sanity check if multi-GPU configuration is correct. Do this before system is created to avoid loading data unnecessarily.
    if config.trainer.world_size < 0:
        raise ValueError(
            f"The parameter 'trainer.world_size' ({config.trainer.world_size}) is negative. This indicates that "
            f"'trainer.world_size' is configured incorrectly. Please set 'trainer.world_size' correctly to the number "
            f"of total GPUs across all nodes to use for training."
        )
    if config.trainer.num_nodes < 0:
        raise ValueError(
            f"The parameter 'trainer.num_nodes' ({config.trainer.num_nodes}) is negative. This indicates that "
            f"'trainer.num_nodes' is configured incorrectly. Please set 'trainer.num_nodes' correctly to the number "
            f"of nodes to use for training."
        )

    if unpack_optional(config.trainer.device_count) > torch.cuda.device_count():
        raise ValueError(
            f"The parameter 'trainer.device_count' ({config.trainer.device_count}) is larger than the number of "
            f"available GPUs ({torch.cuda.device_count()}). If multi-node training is intended, please set "
            f"'+trainer.device_count' to the number of GPUs per node and 'trainer.num_nodes' to the number of nodes "
            f"in the command line."
        )

    # Make the system
    checkpoint = None if (not config.resume_weights_only or not config.resume) else config.resume
    system = nre.systems.make(
        config.system.name,
        config,
        load_from_checkpoint=checkpoint,
    )

    try:
        launch_trainer_loop(config, system, logger)
    except PreemptionInterruptException as e:
        rank_zero_info(f"Preemption detected: {e}. Exiting with code 1.")
        sys.exit(1)

    ScopedTimer.print_summary()
