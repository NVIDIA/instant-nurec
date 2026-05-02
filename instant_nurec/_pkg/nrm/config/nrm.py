# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import os

import shortuuid

from instant_nurec._pkg.config.base_schema import BaseConfigSchema, Field
from instant_nurec._pkg.nrm.config.dataset import NRMSplitsConfig
from instant_nurec._pkg.nrm.config.models import KelvinModelConfig
from instant_nurec._pkg.nrm.config.predict import PredictConfig


SENTINEL = "<sentinel>"


class GaussiansNRMSystemConfig(BaseConfigSchema):
    """Predict-only system config; just dataloader knobs."""

    predict_num_workers: int = Field(default=0, description="Number of workers for the predict dataloader per-node.")
    predict_batch_size: int = Field(default=1, description="Batch size for the predict dataloader. Typically set to 1.")


class NRMConfig(BaseConfigSchema):
    """Top-level NRM predict configuration."""

    seed: int = Field(default=38, description="Random seed.")
    resume: str | None

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
        if self.resume is not None:
            if not self.resume.endswith(".ckpt"):
                self.resume += ".ckpt"
            if not os.path.exists(self.resume):
                raise FileNotFoundError(f"Checkpoint {self.resume!r} does not exist")
        if (env_run_id := os.environ.get("NRE_ENV_RUN_ID")) is not None:
            self.run_id = env_run_id
        self.config_dir = os.path.join(self.out_dir, self.run_id, "config")
