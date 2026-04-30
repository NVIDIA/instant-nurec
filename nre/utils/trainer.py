# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import os

from typing import Optional

import torch.distributed

from pytorch_lightning import Trainer

from nre.config.trainer import TrainerConfig
from nre.utils.misc import rank_zero_only  # type: ignore


def _div_and_round_up(numerator: int, denominator: int | float) -> int:
    return int(math.ceil(numerator / denominator))


def adjust_gaussian_count_for_world_size(trainer_config: TrainerConfig, count: int) -> int:
    """
    Adjusts a global count (e.g., number of Gaussian particles) to account for world size.

    Args:
        trainer_config: The trainer configuration containing world size and relative schedule settings.
        count: The global count to adjust.

    Returns:
        int: The adjusted global count.
    """
    assert count >= 0, f"Expect count to be non-negative, got {count}"
    if trainer_config.world_size == 1:
        return count
    local_count = count // trainer_config.world_size
    assert local_count * trainer_config.world_size <= count, (
        f"local_count * world_size = {local_count * trainer_config.world_size} > count = {count}"
    )
    # Make sure that the sum of the local counts is equal to the global count
    missing = count - local_count * trainer_config.world_size
    if rank_zero_only.rank < missing:
        local_count += 1
    return local_count


def adjust_step_for_world_size(trainer_config: TrainerConfig, step: int) -> int:
    """
    Adjusts a global step to account for world size.

    Args:
        trainer_config: The trainer configuration containing world size and relative schedule settings.
        step: The global step to adjust.

    Returns:
        int: The adjusted global step.
    """
    if trainer_config.relative_schedule and step >= 0:
        return _div_and_round_up(step, trainer_config.world_size * trainer_config.training_step_scaling_factor)
    else:
        return step


def adjust_gamma_for_world_size(trainer_config: TrainerConfig, gamma: float) -> float:
    """
    Adjusts a Exponential Learnig Rate Scheduler gamma for world size.
    """
    return pow(gamma, trainer_config.world_size * trainer_config.training_step_scaling_factor)


def adjust_learning_rate_for_batch_size(trainer_config: TrainerConfig, learning_rate: float) -> float:
    """
    Adjusts a learning rate to account for world size.

    Args:
        trainer_config: The trainer configuration containing world size and relative learning rate settings.
        learning_rate: The learning rate to adjust.

    Returns:
        float: The adjusted learning rate.
    """
    BS = (
        trainer_config.world_size * trainer_config.batch_size_scaling_factor
    )  # FIXME: should also incorporate batch size
    if trainer_config.relative_lr:
        # Multiplying by the world size as recommended by this article:
        # Goyal, Priya et al - Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
        # return learning_rate * trainer_config.world_size
        #
        # Following gsplat: https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py#L292-L296
        # This gives better quality
        return learning_rate * math.sqrt(BS)
    else:
        return learning_rate


def adjust_betas_for_batch_size(trainer_config: TrainerConfig, betas: list[float]) -> list[float]:
    """
    Adjusts a list of betas to account for world size.
    """
    BS = (
        trainer_config.world_size * trainer_config.batch_size_scaling_factor
    )  # FIXME: should also incorporate batch size
    # - gsplat formula: https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py#L295-L296
    # return [1 - BS * (1 - b) for b in betas]
    # - Follow Grendal paper: https://arxiv.org/abs/2406.18533
    return [math.pow(b, BS) for b in betas]


def adjust_num_workers_for_world_size(trainer_config: TrainerConfig, num_workers: int) -> int:
    """
    Adjusts the number of workers, taking into account the world size.

    Args:
        trainer_config: The trainer configuration containing world size and relative num workers settings.
        num_workers: The number of workers to adjust.

    Returns:
        int: The adjusted number of workers
    """
    if trainer_config.relative_num_workers:
        if num_workers > 0:
            return _div_and_round_up(num_workers, trainer_config.world_size)
        else:
            return num_workers
    else:
        return num_workers


class BroadcastExceptions:
    """Context manager for coordinated error handling in distributed training.

    Ensures that if any process encounters an error during distributed training, all processes
    are notified and shut down together. This prevents training processes from
    becoming desynchronized or deadlocked when errors occur in a subset of processes.

    Note: This effectively syncs all processes upon exit,
          which might result in unwanted bubbles specially
          when workload in each GPU is uneven.
    """

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Broadcast exception from one process to all others in DDP training."""

        some_process_raised_exception = self.trainer.strategy.reduce_boolean_decision(exc_val is not None, all=False)

        # If some process raised an exception, abort training
        if some_process_raised_exception:
            self.trainer.should_stop = True

            # If this process raised an exception,
            if exc_val is not None:
                return False  # re-raise it
            raise RuntimeError(f"[rank{self.trainer.global_rank}] Aborting training due to exception in another rank")
