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
import sys

import click
import click_default_group
import pytorch_lightning as pl
import torch

from lightning_utilities.core.rank_zero import rank_zero_info
from pytorch_lightning.callbacks.callback import Callback

import nre.nrm.datasets  # noqa: F401  (populates dataset registry)
import nre.nrm.systems

from nre.config.parse import assert_no_out_dir_override_in_resume, dump_config
from nre.config.version import get_version
from nre.nrm.config.nrm import NRMConfig, parse_typed_nrm_config
from nre.nrm.datasets.datamodule import NRMDataModule
from nre.nrm.systems.base import BaseNRMSystem
from nre.utils.callbacks import (
    PreemptionInterruptException,
    TQDMProgressBar,
    make_logger,
)
from nre.utils.misc import is_env_true, rank_zero_only, unpack_optional


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

    pl.seed_everything(config.seed, workers=True)

    return logger


def make_callbacks(config: NRMConfig, datamodule: NRMDataModule) -> list[Callback]:
    # Predict-only standalone: only the progress bar matters at this stage.
    # Training callbacks (ModelCheckpoint / LearningRateMonitor / TimingLogger /
    # ResumableDataModuleCallback / SLURM-timeout / preempt-on-interrupt /
    # ScopedTimer / MemoryProfiling / ForceValidate / lightning-viewer) were
    # removed in Phase 1 step 4.3.
    del config, datamodule
    return [TQDMProgressBar(refresh_rate=1)]


def launch_trainer_loop(config: NRMConfig, system: BaseNRMSystem, logger: logging.Logger) -> None:
    pl_logger = make_logger(config.logger)

    callbacks = make_callbacks(config, system.datamodule)

    trainer = pl.Trainer(
        devices=unpack_optional(config.system.device_count),
        callbacks=callbacks,
        logger=pl_logger,
        precision=config.system.precision,
        enable_progress_bar=True,
        num_nodes=config.system.num_nodes,
    )

    ckpt_path = config.resume if (config.resume and not config.resume_weights_only) else None

    # For predict without a checkpoint but with init weights, run the
    # train-from-scratch hook (kelvin's path requires it for proper weight
    # loading from the pretrained ngc artifact).
    has_full_init = bool(getattr(config.model, "init_weights_path", None))
    init_weights_paths = getattr(config.model, "init_weights_paths", None)
    if not has_full_init and init_weights_paths is not None:
        if {"full", "tokengs"} & init_weights_paths.keys():
            has_full_init = True
    if config.call_train_from_scratch_hook_for_validation and ckpt_path is None and has_full_init:
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
    """Main entry point for the standalone Kelvin predict pipeline."""

    assert_no_out_dir_override_in_resume(hydra_args)
    config = parse_typed_nrm_config(config_name=config_name, hydra_args=hydra_args)

    # Save the parsed config at early stage
    rank_zero_only(os.makedirs)(config.config_dir, exist_ok=True)
    dump_config(os.path.join(config.config_dir, "parsed.yaml"), config)

    logger = setup_environment_and_logger(config)
    rank_zero_info("NRM RUN 🆔: %s", config.run_id)

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


@click.group(cls=click_default_group.DefaultGroup, default="main", default_if_no_args=True)
@click.version_option(version=str(unpack_optional(get_version(), default="version-not-available")))
def cli():
    pass


cli.add_command(main)

if __name__ == "__main__":
    cli(show_default=True)
