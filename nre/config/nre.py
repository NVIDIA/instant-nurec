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

import logging
import os

from functools import reduce
from math import gcd
from typing import Any, Literal

from libs.losses.orchestration.config import LossConfig
from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.checkpoint import CheckpointConfig
from nre.config.datamodule import SODatamoduleConfig
from nre.config.dataset import (
    NCoreDatasetConfig,
)
from nre.config.difix import DifixConfig
from nre.config.logger import LoggerConfigType
from nre.config.model import ModelConfig
from nre.config.prober import ProberConfig
from nre.config.profiling import ProfilingConfig
from nre.config.scopedtimer import ScopedTimerConfig
from nre.config.sensor import SensorConfig
from nre.config.systems import (
    GaussiansSystemConfig,
    NRendTestGaussiansSystemConfig,
)
from nre.config.trainer import TrainerConfig
from nre.config.version import Version, get_version
from nre.config.viewer import ViewerConfig


SENTINEL = "<sentinel>"

current_version = get_version()

cmd_logger = logging.getLogger(__name__)


class NREConfig(BaseConfigSchema):
    seed: int
    mode: Literal["train", "val", "test", "trainval"]

    resume: str | None
    resume_weights_only: bool
    verbose: bool
    force_validate: bool = Field(default=False, description="Force validation of the config.")

    out_dir: str
    logger: LoggerConfigType = Field(discriminator="name")

    checkpoint: CheckpointConfig
    difix: DifixConfig

    viewer: ViewerConfig = Field(default_factory=ViewerConfig, description="Web-based viewer configuration.")
    prober: ProberConfig = Field(
        default_factory=ProberConfig, description="Tensor prober configuration for debugging and testing."
    )
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig, description="Profiling configuration.")
    scopedtimer: ScopedTimerConfig = Field(default_factory=ScopedTimerConfig, description="Scoped timer configuration.")

    trainer: TrainerConfig
    dataset: NCoreDatasetConfig = Field(discriminator="name")
    datamodule: SODatamoduleConfig
    model: ModelConfig
    system: NRendTestGaussiansSystemConfig | GaussiansSystemConfig = Field(discriminator="name")
    sensor: SensorConfig
    loss: LossConfig | None

    version: Version | None = Field(
        default=current_version,
        description="Not to be set by the user. Used to detect NRE version mismatch when loading old configs. Not available in sandboxed test executions",
    )

    save_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where images are saved during validation phase. If left unchanged, defaults to `out_dir/save`"
        ),
    )
    ckpt_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where model checkpoints are saved during training. If left "
            "unchanged, defaults to `out_dir/checkpoints`"
        ),
    )
    config_dir: str = Field(
        default=SENTINEL,
        description=(
            "Directory where the config (with all the auto-generated and default "
            "fields) will be stored. If left unchanged, defaults to `out_dir/config`"
        ),
    )
    run_id: str = Field(
        default=SENTINEL,
        description=(
            "A unique identifier of the training run. If left unchanged, will be auto-generated. "
            "If resuming training from a checkpoint, the previous run_id will be restored."
        ),
    )
    train_config_name: str = Field(
        default=SENTINEL,
        description=(
            "The name of the training config file. If empty, the config name will be the one provided in the command line. "
        ),
    )
    matmul_precision: Literal["highest", "high", "medium"] = Field(
        default="highest",
        description=(
            "PyTorch matmul precision setting. Controls the internal precision of float32 matrix multiplications. "
            "'highest' uses float32 datatype (default), 'high' uses TensorFloat32 or bfloat16-based algorithm, "
            "'medium' uses bfloat16 datatype for better performance with some precision trade-off."
        ),
    )

    def _setup_resume_path(self) -> None:
        if self.resume is None:
            return

        # When user use resume=auto, we will try to resume from the last checkpoint with the same run_id.
        # If the checkpoint does not exist, we will train from scratch.
        # This is useful for long-running training jobs on clusters where the job is constantly interrupted & restarted.
        if self.resume == "auto":
            is_auto_resume = True
            self.resume = "last"
        else:
            is_auto_resume = False

        # Add .ckpt if config.resume doesn't include it. Useful to run resume=last or resume=best
        if not self.resume.endswith(".ckpt"):
            self.resume += ".ckpt"
        if not os.path.exists(self.resume):
            if self.ckpt_dir == SENTINEL:
                if is_auto_resume:
                    # Automatically find out the checkpoint directory
                    self.ckpt_dir = os.path.join(self.out_dir, self.logger.run_id, "checkpoints")
                    cmd_logger.info(f"Automatically resuming from {self.ckpt_dir}")
                else:
                    raise FileNotFoundError("config.ckpt_dir is not provided, can't guess the resume path.")

            self.resume = os.path.join(self.ckpt_dir, self.resume)
            if not os.path.exists(self.resume):
                if is_auto_resume:
                    # If checkpoint not found, training from scratch!
                    self.resume = None
                    cmd_logger.info("Checkpoint not found, training from scratch.")
                # Do not check for checkpoint existence here.
                # It's only used by the trainer, and can be safely ignored when it's not being used.
                # This check will eventually be done on training.

    def _setup_run_id(self) -> None:
        """propagate run_id set in self.logger to other fields"""
        run_id = self.logger.run_id
        self.run_id = run_id
        self.save_dir = os.path.join(self.out_dir, run_id, "save")
        self.ckpt_dir = os.path.join(self.out_dir, run_id, "checkpoints")
        self.config_dir = os.path.join(self.out_dir, run_id, "config")

    def _setup_update_n_epochs(self) -> None:
        # Check if any of the samplers set the update_n_epochs parameter otherwise use the global value
        if isinstance(self.dataset, NCoreDatasetConfig) and self.dataset.samplers:
            update_n_epochs = []
            for batch_sampler, sampler_config in self.dataset.samplers.items():
                if not hasattr(sampler_config, "update_n_epochs"):
                    raise ValueError(f"Required key 'update_n_epochs' is missing from {batch_sampler} config.")
                elif type(update_interval := sampler_config.get("update_n_epochs")) is not int or (update_interval < 0):
                    raise TypeError(f"Key 'update_n_epochs' in {batch_sampler} must be a non-negative integer.")

                if update_interval > 0:
                    update_n_epochs.append(update_interval)

            if update_n_epochs:
                gcd_all = reduce(gcd, update_n_epochs)
                if gcd_all != min(update_n_epochs):
                    raise ValueError(
                        "Update intervals update_n_epochs of individual samplers must be multiples of each other."
                    )
                self.datamodule.update_n_epochs = gcd_all

    def model_post_init(self, __context) -> None:
        self._setup_update_n_epochs()
        self._setup_resume_path()
        self._setup_run_id()  # modifies self.ckpt_dir so needs to be called AFTER _setup_resume_path

        # Handle train_config_name setup
        if self.train_config_name == SENTINEL and __context is not None and "config_name" in __context:
            self.train_config_name = __context.get("config_name")
