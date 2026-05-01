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

from typing import Literal

from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.logger import LoggerConfigType
from nre.nrm.config.dataset import NRMSplitsConfig
from nre.nrm.config.models import KelvinModelConfig
from nre.nrm.config.predict import PredictConfig


SENTINEL = "<sentinel>"


class BaseNRMSystemConfig(BaseConfigSchema):
    """
    Predict-only system config. NRE merged in trainer/datamodule config too;
    the standalone keeps just the dataloader knobs.
    """

    predict_num_workers: int = Field(default=0, description="Number of workers for the predict dataloader per-node.")
    predict_batch_size: int = Field(default=1, description="Batch size for the predict dataloader. Typically set to 1.")


class GaussiansNRMSystemConfig(BaseNRMSystemConfig):
    """
    System config for the Gaussians NRM system.
    """

    name: Literal["base-nrm-system"]


class NRMConfig(BaseConfigSchema):
    """
    Top-level configuration for NRM training/validation/testing.
    """

    seed: int = Field(default=38, description="Random seed.")
    mode: Literal["train", "val", "test", "trainval", "predict"]

    resume: str | None
    resume_weights_only: bool
    call_train_from_scratch_hook_for_validation: bool = Field(
        default=False,
        description=(
            "When True, the model's on_train_from_scratch_start hook is also called when running val/test/predict "
            "without a checkpoint but with init weights. Set to False if the hook contains train-only logic that "
            "must not run during eval (e.g. writing training state)."
        ),
    )
    verbose: bool = Field(default=False, description="Verbose mode.")

    out_dir: str
    logger: LoggerConfigType = Field(discriminator="name")

    system: GaussiansNRMSystemConfig = Field(discriminator="name")
    dataset: NRMSplitsConfig = Field(discriminator="name")

    model: KelvinModelConfig = Field(discriminator="name")

    # Predict configuration
    predict: PredictConfig = Field(
        default_factory=PredictConfig,
        description="Configuration for predict-time-only functionality such as primitive merging",
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

    def _setup_resume_path(self) -> None:
        """Predict-only standalone: resume is set by load_predict_config to the
        absolute path of the downloaded NGC checkpoint; just validate it exists."""
        if self.resume is None:
            return
        if not self.resume.endswith(".ckpt"):
            self.resume += ".ckpt"
        if not os.path.exists(self.resume):
            raise FileNotFoundError(f"Checkpoint {self.resume!r} does not exist")

    def _setup_run_id(self) -> None:
        """propagate run_id set in self.logger to other fields"""
        run_id = self.logger.run_id
        self.run_id = run_id
        self.save_dir = os.path.join(self.out_dir, run_id, "save")
        self.ckpt_dir = os.path.join(self.out_dir, run_id, "checkpoints")
        self.config_dir = os.path.join(self.out_dir, run_id, "config")

    def model_post_init(self, __context) -> None:
        self._setup_resume_path()
        self._setup_run_id()  # modifies self.ckpt_dir so needs to be called AFTER _setup_resume_path
