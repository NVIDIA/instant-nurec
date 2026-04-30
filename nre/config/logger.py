# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal

import shortuuid

from nre.config.base_schema import BaseConfigSchema, Field


GENERATE = "<generate>"


class BaseLoggerConfig(BaseConfigSchema):
    enabled: bool | None = None
    run_id: str = GENERATE
    save_dir: str | None = None

    def model_post_init(self, __context) -> None:
        if self.run_id == GENERATE:
            if (run_id := os.environ.get("NRE_ENV_RUN_ID")) is not None:
                self.run_id = run_id
            else:
                self.run_id = shortuuid.uuid()


class DummyLoggerConfig(BaseLoggerConfig):
    name: Literal["dummy"]

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.enabled is None:
            self.enabled = False


LoggerConfigType = DummyLoggerConfig


class BatchMediaLoggerConfigMixin(BaseConfigSchema):
    log_media_every_n_steps: int = Field(
        description="Number of steps between logging media (images, videos, etc.) during training."
    )
    log_media_every_n_steps_val: int = Field(
        description="Number of steps between logging media (images, videos, etc.) during validation."
    )
    log_media_subsample: int = Field(
        default=1,
        description=("Stride-subsample each tile before compositing logged image grids."),
    )
