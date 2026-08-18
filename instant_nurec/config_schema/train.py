# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from pathlib import Path
from typing import Literal

import shortuuid

from instant_nurec.config_schema.base_schema import BaseConfigSchema, Field
from instant_nurec.config_schema.dataset import InstantNuRecSplitsConfig
from instant_nurec.config_schema.models import KelvinModelConfig


class KelvinOptimizerConfig(BaseConfigSchema):
    lr: float = Field(default=1.0e-4, gt=0)
    eps: float = Field(default=1.0e-15, gt=0)
    betas: tuple[float, float] = Field(default=(0.9, 0.99))
    weight_decay: float = Field(default=0.0, ge=0)
    implementation: Literal["auto", "apex-fused-adam", "torch-adam"] = "auto"


class KelvinSchedulerConfig(BaseConfigSchema):
    warmup_factor: float = Field(default=0.01, gt=0, le=1)
    warmup_steps: int = Field(default=100, ge=1)
    cosine_factor: float = Field(default=0.0333, gt=0, le=1)
    cosine_factor_progress: float = Field(default=1.0, gt=0, le=1)


class KelvinLoggerConfig(BaseConfigSchema):
    """Experiment logger settings for standalone Kelvin training."""

    name: Literal["csv", "wandb"] = "csv"
    project: str = "NRE"
    entity: str = ""
    run_name: str | None = None
    group: str = ""
    tags: list[str] = Field(default_factory=list)
    job_type: str = ""
    offline: bool = False
    log_model: bool = False


class KelvinSystemTrainConfig(BaseConfigSchema):
    max_epochs: int = Field(default=40, ge=1)
    train_batch_size: int = Field(default=2, ge=1)
    val_batch_size: int = Field(default=2, ge=1)
    train_num_workers: int = Field(default=6, ge=0)
    val_num_workers: int = Field(default=4, ge=0)
    precision: Literal["bf16-mixed", "32-true"] = "bf16-mixed"
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | str = "auto"
    num_nodes: int = Field(default=1, ge=1)
    strategy: str = "auto"
    deterministic: bool | Literal["warn"] = Field(
        default="warn",
        description=(
            "Lightning deterministic policy. 'warn' preserves deterministic dataset sampling "
            "while permitting CUDA bicubic backward, which has no deterministic implementation."
        ),
    )
    enable_render_global_step: int = Field(default=1_000_000, ge=0)
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    log_every_n_steps: int = Field(default=10, ge=1)
    save_every_n_train_steps: int | None = Field(default=None, ge=1)
    save_on_preemption: bool = Field(
        default=True,
        description="Atomically save checkpoints/last.ckpt and exit when SIGUSR1 is received.",
    )
    checkpoint_monitor: str = "val/psnr"
    checkpoint_mode: Literal["min", "max"] = "max"
    save_top_k: int = Field(default=2, ge=0)
    optimizer: KelvinOptimizerConfig = Field(default_factory=KelvinOptimizerConfig)
    scheduler: KelvinSchedulerConfig = Field(default_factory=KelvinSchedulerConfig)


class KelvinLossConfig(BaseConfigSchema):
    primitive_sky_cubemap: float = Field(default=1.0, ge=0)
    primitive_rgb: float = Field(default=0.1, ge=0)
    primitive_distance: float = Field(default=1.0, ge=0)
    primitive_distance_gradient: float = Field(default=1.0, ge=0)
    primitive_semantics: float = Field(default=0.01, ge=0)
    primitive_normal: float = Field(default=0.2, ge=0)
    primitive_velocity: float = Field(default=1.0, ge=0)
    rgb: float = Field(default=0.0, ge=0)
    lpips: float = Field(default=0.0, ge=0)
    distance: float = Field(default=0.0, ge=0)
    background: float = Field(default=0.0, ge=0)
    distance_min_m: float = Field(default=0.1, gt=0)
    primitive_distance_max_m: float = Field(default=300.0, gt=0)
    primitive_distance_gradient_max_m: float = Field(default=50.0, gt=0)
    render_distance_max_m: float = Field(default=50.0, gt=0)

    @classmethod
    def context_phase(cls) -> "KelvinLossConfig":
        return cls()

    @classmethod
    def render_phase(cls) -> "KelvinLossConfig":
        return cls(
            primitive_sky_cubemap=0.01,
            primitive_rgb=0.01,
            primitive_distance=1.0,
            primitive_distance_gradient=0.01,
            primitive_semantics=0.01,
            primitive_normal=0.2,
            primitive_velocity=1.0,
            rgb=1.0,
            lpips=0.2,
            distance=0.1,
            background=2.0,
        )


class KelvinTrainConfig(BaseConfigSchema):
    """Standalone training configuration aligned to Bazel NRE Kelvin."""

    seed: int = 38
    out_dir: Path
    run_id: str = Field(default_factory=shortuuid.uuid)
    phase: Literal["context", "render"] = "context"
    bazel_reference: str = "kelvin-pa-front-2026-08-13"
    dataset: InstantNuRecSplitsConfig
    model: KelvinModelConfig = Field(default_factory=KelvinModelConfig)
    system: KelvinSystemTrainConfig = Field(default_factory=KelvinSystemTrainConfig)
    logger: KelvinLoggerConfig = Field(default_factory=KelvinLoggerConfig)
    loss: KelvinLossConfig | None = None
    resume_from_checkpoint: Path | None = None

    def model_post_init(self, __context) -> None:
        if (environment_run_id := os.environ.get("NRE_ENV_RUN_ID")) is not None:
            self.run_id = environment_run_id
        if self.dataset.train is None:
            raise ValueError("dataset.train is required")
        if self.phase == "render":
            self.system.enable_render_global_step = 0
            if self.loss is None:
                self.loss = KelvinLossConfig.render_phase()
            if self.system.strategy == "auto":
                self.system.strategy = "ddp_find_unused_parameters_true"
        elif self.loss is None:
            self.loss = KelvinLossConfig.context_phase()
