# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math

import torch

from instant_nurec.config_schema.train import TrainingOptimizerConfig, TrainingSchedulerConfig


logger = logging.getLogger(__name__)


class CosineWithWarmupPBScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Progress-based schedule used by the reference training recipe."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: TrainingSchedulerConfig, last_epoch: int = -1):
        self.warmup_factor = config.warmup_factor
        self.warmup_steps = config.warmup_steps
        self.cosine_factor = config.cosine_factor
        self.cosine_factor_progress = config.cosine_factor_progress
        self._current_epoch = 0
        self._total_epochs = 0
        self._current_local_step = 0
        self._total_local_steps = 0
        super().__init__(optimizer, last_epoch)

    def set_progress(self, epoch: int, total_epochs: int, local_step: int, total_local_steps: int) -> None:
        self._current_epoch = epoch
        self._total_epochs = total_epochs
        self._current_local_step = local_step
        self._total_local_steps = total_local_steps

    def get_lr(self) -> list[float]:
        if self._step_count <= 1:
            return [float(group["initial_lr"]) * 1.0e-6 for group in self.optimizer.param_groups]
        if self._total_local_steps <= 0 or self._total_epochs <= 0:
            raise RuntimeError("set_progress() must be called before stepping the training scheduler")
        current_global_step = self._current_epoch * self._total_local_steps + self._current_local_step
        if current_global_step < self.warmup_steps:
            denominator = max(self.warmup_steps - 1, 1)
            warmup_progress = current_global_step / denominator
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * warmup_progress
        else:
            denominator = max(self._total_epochs * self._total_local_steps - self.warmup_steps, 1)
            progress = (current_global_step - self.warmup_steps) / denominator
            progress = min(progress / self.cosine_factor_progress, 1.0)
            factor = self.cosine_factor + (1.0 - self.cosine_factor) * (1.0 + math.cos(math.pi * progress)) / 2.0
        return [float(group["initial_lr"]) * factor for group in self.optimizer.param_groups]


def make_optimizer(
    parameters,
    config: TrainingOptimizerConfig,
) -> tuple[torch.optim.Optimizer, str]:
    kwargs = {
        "lr": config.lr,
        "eps": config.eps,
        "betas": config.betas,
        "weight_decay": config.weight_decay,
    }
    if config.implementation in ("auto", "apex-fused-adam"):
        try:
            from apex.optimizers import FusedAdam

            return FusedAdam(parameters, **kwargs), "apex-fused-adam"
        except (ImportError, RuntimeError) as exc:
            if config.implementation == "apex-fused-adam":
                raise RuntimeError(
                    "Exact optimizer-kernel matching requires apex.optimizers.FusedAdam; install NVIDIA Apex "
                    "or select implementation=torch-adam for a numerically non-bitwise fallback."
                ) from exc
            logger.warning("Apex FusedAdam is unavailable; using torch.optim.Adam (not bitwise equivalent).")
    return torch.optim.Adam(parameters, **kwargs), "torch-adam"
