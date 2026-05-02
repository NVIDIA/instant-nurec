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
import yaml

import instant_nurec._pkg.nrm.datasets  # noqa: F401  (populates dataset registry)
import instant_nurec._pkg.nrm.systems as nrm_systems

from instant_nurec._pkg.nrm.config.nrm import NRMConfig


logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_predict(config: NRMConfig) -> None:
    """Run the standalone Kelvin predict pipeline against an already-typed config."""
    os.makedirs(config.config_dir, exist_ok=True)
    with open(os.path.join(config.config_dir, "parsed.yaml"), "w") as fp:
        yaml.safe_dump(config.model_dump(mode="json"), fp, sort_keys=False)

    _seed_everything(config.seed)
    logger.info("NRM RUN \U0001f194: %s", config.run_id)

    assert config.resume, "Standalone predict requires a checkpoint path (config.resume)."
    system = nrm_systems.make(config, load_from_checkpoint=config.resume)
    device = torch.device("cuda")
    system.to(device).eval()

    dataloader = system.datamodule.predict_dataloader()
    with torch.inference_mode():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = system.predict_step(batch)
            system.on_predict_batch_end(outputs, batch)
