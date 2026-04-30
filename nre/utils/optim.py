# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone keeps a stub. The real file
# carried optimizer/scheduler factories + StepFun* schedulers — all
# training-only. Predict never instantiates an optimizer.

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from torch.optim.lr_scheduler import LRScheduler


class LRSchedulerConfigType(TypedDict, total=False):
    scheduler: LRScheduler
    interval: NotRequired[str]
    frequency: NotRequired[int]
    monitor: NotRequired[str]
    strict: NotRequired[bool]
    name: NotRequired[str]


class OptimizerLRSchedulerConfig(TypedDict):
    optimizer: Any
    lr_scheduler: NotRequired[LRSchedulerConfigType]


LRSchedulerTypeUnion = Any


class StepFunLRSchedulerMixin(LRScheduler):
    pass


def get_model_parameters(*_args: Any, **_kwargs: Any) -> list:
    return []


optim_fns: dict[str, Any] = {}
