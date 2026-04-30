# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from typing import Literal

import omegaconf

from libs.losses.models.lambda_schedulers.registry import register as register_lambda_scheduler
from nre.config.trainer import TrainerConfig
from nre.utils.trainer import adjust_step_for_world_size


class BaseLambdaScheduler(ABC):
    lambda_init: float

    def __init__(
        self, config: omegaconf.dictconfig.DictConfig, trainer_config: TrainerConfig, lambda_init: float, **kwargs
    ) -> None:
        self.lambda_init = lambda_init
        self.trainer_config = trainer_config

    @abstractmethod
    def __call__(self, epoch: int, global_step: int, **kwargs) -> float:
        raise NotImplementedError("")


@register_lambda_scheduler("linear_lambda")
class LinearLambdaScheduler(BaseLambdaScheduler):
    start: int
    end: int
    update_interval: Literal["epoch", "step"] = "step"
    update_frequency: int = 1
    lambda_end: float = 0

    def __init__(
        self, config: omegaconf.dictconfig.DictConfig, trainer_config: TrainerConfig, lambda_init: float, **kwargs
    ) -> None:
        super().__init__(config, trainer_config, lambda_init, **kwargs)
        self.lambda_end = config.lambda_end
        self.update_interval = config.update_interval
        if self.update_interval == "step":
            self.update_frequency = adjust_step_for_world_size(trainer_config, config.update_frequency)
            self.start = adjust_step_for_world_size(trainer_config, config.start)
            self.end = adjust_step_for_world_size(trainer_config, config.end)
        else:
            self.update_frequency = config.update_frequency
            self.start = config.start
            self.end = config.end

        self.total_stages = (self.end - self.start) // self.update_frequency

    def __call__(self, epoch: int, global_step: int, **kwargs) -> float:
        cur_stage = ((global_step if self.update_interval == "step" else epoch) - self.start) // self.update_frequency
        ratio = min(1.0, max(0.0, cur_stage / self.total_stages))
        return (1 - ratio) * self.lambda_init + ratio * self.lambda_end
