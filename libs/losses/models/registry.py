# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Registry for loss functions and loss classes."""

from __future__ import annotations

import functools
import inspect

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn


# Safeguard to avoid a circular dependency due to type annotations
# in case a system inside .base is registered by using the decorator @register.
if TYPE_CHECKING:
    from libs.losses.models.base_losses import BaseLoss

losses: dict[str, BaseLoss] = {}


def make_loss(name: str, config, trainer_config):
    return losses[name](config=config, trainer_config=trainer_config)


def register_loss(name: str):
    def decorator(cls):
        losses[name] = cls
        return cls

    return decorator


def allow_extra_kwargs(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    Allow extra kwargs to be passed to the loss function.
    This is useful when some loss_fns require extra kwargs but some don't, under which case
    we annotate the latter with this decorator.
    """
    allowed_kwargs = set(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **{k: v for k, v in kwargs.items() if k in allowed_kwargs})

    return wrapper


loss_fns: dict[str, Callable[..., torch.Tensor]] = {
    "huber": nn.HuberLoss(reduction="none"),
    "mse": nn.MSELoss(reduction="none"),
    "l1": allow_extra_kwargs(nn.L1Loss(reduction="none").forward),
    "bce": nn.BCELoss(reduction="none"),
    "bce_with_logits": nn.BCEWithLogitsLoss(reduction="none"),
    "cross_entropy": nn.CrossEntropyLoss(reduction="none"),
    "smooth_l1": nn.SmoothL1Loss(reduction="none"),
    "square": torch.square,
    "abs": torch.abs,
}


def make_loss_fn(name: str):
    return loss_fns[name]


def register_loss_fn(name: str):
    def decorator(fn: Callable[..., torch.Tensor]):
        loss_fns[name] = fn
        return fn

    return decorator
