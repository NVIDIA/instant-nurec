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

from __future__ import annotations

import os

import shortuuid

from instant_nurec.config_schema.base_schema import BaseConfigSchema, Field
from instant_nurec.config_schema.dataset import NRMSplitsConfig
from instant_nurec.config_schema.models import KelvinModelConfig
from instant_nurec.config_schema.predict import PredictConfig


SENTINEL = "<sentinel>"


class GaussiansNRMSystemConfig(BaseConfigSchema):
    """Predict-only system config; just dataloader knobs."""

    predict_num_workers: int = Field(default=0, description="Number of workers for the predict dataloader per-node.")
    predict_batch_size: int = Field(default=1, description="Batch size for the predict dataloader. Typically set to 1.")


class NRMConfig(BaseConfigSchema):
    """Top-level NRM predict configuration."""

    seed: int = Field(default=38, description="Random seed.")

    out_dir: str

    system: GaussiansNRMSystemConfig
    dataset: NRMSplitsConfig
    model: KelvinModelConfig

    predict: PredictConfig = Field(
        default_factory=PredictConfig,
        description="Configuration for predict-time-only functionality such as primitive merging",
    )

    config_dir: str = Field(
        default=SENTINEL,
        description="Directory where parsed.yaml is dumped. Auto-derived as out_dir/run_id/config.",
    )
    run_id: str = Field(
        default_factory=shortuuid.uuid,
        description=(
            "Unique identifier of this run; auto-generated as a shortuuid unless "
            "overridden via the NRE_ENV_RUN_ID environment variable."
        ),
    )

    def model_post_init(self, __context) -> None:
        if (env_run_id := os.environ.get("NRE_ENV_RUN_ID")) is not None:
            self.run_id = env_run_id
        self.config_dir = os.path.join(self.out_dir, self.run_id, "config")
