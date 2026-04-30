# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Literal

from pydantic import ConfigDict

from nre.config.base_schema import BaseConfigSchema, Field


class NovelViewPoseConfig(BaseConfigSchema):
    """Configuration for a single novel view pose with translation and rotation deltas."""

    translation: list[float] | None = None
    rotation: list[float] | None = None


class PSchedulerConfig(BaseConfigSchema):
    """Probability scheduler configuration for training-time difix."""

    p_init: float = 0.5
    p_final: float | None = None
    milestones: list[int] = Field(default_factory=list)
    update_interval: Literal["step", "epoch"] = "step"
    gamma: float = 0.5


class TrainingDifixConfig(BaseConfigSchema):
    """Configuration for training-time difix."""

    enabled: bool = False
    p_scheduler: PSchedulerConfig = Field(default_factory=PSchedulerConfig)
    start_step: int = 0
    use_color_transfer: bool = True
    num_workers: int = 0
    novel_view_poses: list[NovelViewPoseConfig] = Field(default_factory=list)
    shuffle_novel_views: bool = False
    log_images: bool = False
    log_every_n_steps: int = 100
    log_buffer_images: bool = False
    debug_dir: str | None = None


class InferenceDifixConfig(BaseConfigSchema):
    """Configuration for inference-time difix."""

    enabled: bool = False
    use_color_transfer: bool = True


class DifixModelConfig(BaseConfigSchema):
    """Configuration for the Difix model."""

    name: str = "cosmos-difix"
    cache_dir: str = "~/.cache/nre/difix"
    # Set model config to avoid pydantic warnings on protected namespaces, i.e. below class fields starting with model_
    model_config = ConfigDict(protected_namespaces=())
    model_url: str = ""
    model_resolution: tuple[int, int] = (576, 1024)
    model_filename: str = ""


class DifixConfig(DifixModelConfig):
    """Main difix configuration with model parameters and training/inference settings."""

    training: TrainingDifixConfig = Field(default_factory=TrainingDifixConfig)
    inference: InferenceDifixConfig = Field(default_factory=InferenceDifixConfig)
