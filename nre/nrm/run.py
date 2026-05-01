# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Predict-only entrypoint. Self-invented: NRE drives the predict loop with
# pl.Trainer.predict; we strip pytorch_lightning and call the system hooks
# directly so the standalone has no Lightning/Trainer dependency.

import logging
import os
import random

import numpy as np
import torch

import nre.nrm.datasets  # noqa: F401  (populates dataset registry)
import nre.nrm.systems

from nre.config.parse import dump_config
from nre.nrm.config.nrm import NRMConfig, parse_typed_nrm_config
from nre.nrm.systems.base import BaseNRMSystem


logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_environment_and_logger(config: NRMConfig) -> logging.Logger:
    log = logging.getLogger(__name__)
    if config.verbose:
        log.setLevel(logging.DEBUG)
    _seed_everything(config.seed)
    return log


def launch_predict_loop(config: NRMConfig, system: BaseNRMSystem) -> None:
    device = _select_device()
    system.to(device)
    system.eval()

    ckpt_path = config.resume if (config.resume and not config.resume_weights_only) else None
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        system.on_load_checkpoint(ckpt)
        system.to(device)

    has_full_init = bool(getattr(config.model, "init_weights_path", None))
    init_weights_paths = getattr(config.model, "init_weights_paths", None)
    if not has_full_init and init_weights_paths is not None:
        if {"full", "tokengs"} & init_weights_paths.keys():
            has_full_init = True
    if config.call_train_from_scratch_hook_for_validation and ckpt_path is None and has_full_init:
        system.model.on_train_from_scratch_start(system)
        system.to(device)

    if "predict" not in config.mode:
        raise ValueError(f"Only predict mode is supported in this standalone; got mode={config.mode}.")

    dataloader = system.datamodule.predict_dataloader()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            system.on_predict_batch_start(batch, batch_idx)
            outputs = system.predict_step(batch, batch_idx)
            system.on_predict_batch_end(outputs, batch, batch_idx)


def main(config_name: str, hydra_args: list[str] | tuple[str, ...]) -> None:
    """Main entry point for the standalone Kelvin predict pipeline."""

    config = parse_typed_nrm_config(config_name=config_name, hydra_args=hydra_args)

    os.makedirs(config.config_dir, exist_ok=True)
    dump_config(os.path.join(config.config_dir, "parsed.yaml"), config)

    setup_environment_and_logger(config)
    logger.info("NRM RUN 🆔: %s", config.run_id)

    checkpoint = None if (not config.resume_weights_only or not config.resume) else config.resume
    system = nre.nrm.systems.make(config.system.name, config, load_from_checkpoint=checkpoint)
    launch_predict_loop(config, system)
