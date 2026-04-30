# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""General-purpose schedulers for training.

This module contains scheduler classes that can be used across different parts of the codebase:
- MultiStepScheduler: Piecewise-constant scheduler with gamma decay at milestones
"""

from __future__ import annotations

from typing import Literal

import omegaconf

from nre.config.trainer import TrainerConfig
from nre.utils.trainer import adjust_step_for_world_size


class MultiStepScheduler:
    """Piecewise-constant scheduler: multiply by gamma at each milestone.

    Mirrors PyTorch's MultiStepLR semantics with support for step/epoch intervals.
    Optimized for fast per-step evaluation.

    Config keys:
      - update_interval: "step" | "epoch" - whether to use steps or epochs
      - milestones: list[int] - training steps/epochs where value changes
      - gamma: float (default 0.1) - multiplier applied at each milestone
    """

    def __init__(self, config: omegaconf.DictConfig, trainer_config: TrainerConfig, value_init: float) -> None:
        """Initialize the multi-step scheduler.

        Args:
            config: Configuration with update_interval, milestones, and gamma
            trainer_config: Trainer configuration for step adjustments
            value_init: Initial value (before any milestones)
        """
        self.value_init = value_init
        self.update_interval: Literal["epoch", "step"] = getattr(config, "update_interval", "step")
        self.gamma = float(getattr(config, "gamma", 0.1))
        raw_milestones = list(getattr(config, "milestones", []))
        self.value_final = getattr(config, "p_final", None)

        # Adjust milestones for distributed training if using steps
        if self.update_interval == "step":
            self.milestones = [adjust_step_for_world_size(trainer_config, m) for m in raw_milestones]
        else:
            self.milestones = raw_milestones
        self.milestones.sort()

        # If p_final is provided, derive gamma to reach p_final at the last milestone count
        if self.value_final is not None and self.milestones:
            num_segments = len(self.milestones)
            # Avoid division by zero; if value_init is 0, fallback to given gamma
            if value_init > 0.0 and self.value_final > 0.0:
                self.gamma = (self.value_final / value_init) ** (1.0 / num_segments)

        # Precompute values at each milestone for O(1) lookup after counting (piecewise-constant)
        self._milestone_values = [value_init * (self.gamma**i) for i in range(len(self.milestones) + 1)]

    def __call__(self, epoch: int, global_step: int) -> float:
        """Compute the scheduled value based on milestones.

        Args:
            epoch: Current training epoch
            global_step: Current global training step

        Returns:
            value_init * (gamma ** count), where count is the number of passed milestones
        """
        progress = global_step if self.update_interval == "step" else epoch

        # Fast path for common case: before first milestone
        if not self.milestones or progress < self.milestones[0]:
            return self._milestone_values[0]

        # Count passed milestones (milestones are sorted)
        count = 0
        for milestone in self.milestones:
            if progress >= milestone:
                count += 1
            else:
                break

        return self._milestone_values[count]
