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

import click
import click_default_group
import pytorch_lightning as pl

from lightning_utilities.core.rank_zero import rank_zero_info

import nre.nrm.datasets  # noqa: F401  (populates dataset registry)
import nre.nrm.systems

from nre.config.parse import dump_config
from nre.config.version import get_version
from nre.nrm.config.nrm import NRMConfig, parse_typed_nrm_config
from nre.nrm.systems.base import BaseNRMSystem
from nre.utils.callbacks import TQDMProgressBar, make_logger
from nre.utils.misc import rank_zero_only, unpack_optional


def setup_environment_and_logger(config: NRMConfig) -> logging.Logger:
    logger = logging.getLogger(__name__)
    if config.verbose:
        logger.setLevel(logging.DEBUG)
    pl.seed_everything(config.seed, workers=True)
    return logger


def launch_trainer_loop(config: NRMConfig, system: BaseNRMSystem) -> None:
    trainer = pl.Trainer(
        devices=unpack_optional(config.system.device_count),
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        logger=make_logger(config.logger),
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

    config = parse_typed_nrm_config(config_name=config_name, hydra_args=hydra_args)

    rank_zero_only(os.makedirs)(config.config_dir, exist_ok=True)
    dump_config(os.path.join(config.config_dir, "parsed.yaml"), config)

    setup_environment_and_logger(config)
    rank_zero_info("NRM RUN 🆔: %s", config.run_id)

    checkpoint = None if (not config.resume_weights_only or not config.resume) else config.resume
    system = nre.nrm.systems.make(config.system.name, config, load_from_checkpoint=checkpoint)
    launch_trainer_loop(config, system)


@click.group(cls=click_default_group.DefaultGroup, default="main", default_if_no_args=True)
@click.version_option(version=str(unpack_optional(get_version(), default="version-not-available")))
def cli():
    pass


cli.add_command(main)

if __name__ == "__main__":
    cli(show_default=True)
