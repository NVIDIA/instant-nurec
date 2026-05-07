# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random

import numpy as np
import torch

import instant_nurec.datasets  # noqa: F401  (populates dataset registry)
import instant_nurec.model as instantnurec_systems

from instant_nurec.config_schema.instantnurec import InstantNuRecConfig


logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_predict(config: InstantNuRecConfig) -> None:
    """Run the Kelvin predict pipeline against an already-typed config."""
    _seed_everything(config.seed)
    logger.info("InstantNuRec RUN \U0001f194: %s", config.run_id)

    system = instantnurec_systems.make(config)
    device = torch.device("cuda")
    system.to(device).eval()

    dataloader = system.datamodule.predict_dataloader()
    with torch.inference_mode():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = system.predict_step(batch)
            system.on_predict_batch_end(outputs, batch)
