# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:  # Needed to avoid a circular dependency due to type annotations
    from libs.losses.models.lambda_schedulers.lambda_schedulers import BaseLambdaScheduler
    from nre.config.trainer import TrainerConfig

lambda_schedulers: dict[str, Callable[..., "BaseLambdaScheduler"]] = {}


def register(name: str):
    def decorator(cls):
        lambda_schedulers[name] = cls
        return cls

    return decorator


def make(name, config, trainer_config: "TrainerConfig", lambda_init, **kwargs) -> "BaseLambdaScheduler":
    lambda_scheduler_class = lambda_schedulers[name]
    return lambda_scheduler_class(config, trainer_config, lambda_init, **kwargs)
