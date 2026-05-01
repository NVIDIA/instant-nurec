# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from omegaconf import DictConfig


class BaseModel(nn.Module):
    post_processings: Iterable[nn.Module]

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        # In PyTorch, any registered buffer in an nn.Module will be automatically moved to the correct device when `to(device)` is
        # called on the module. Thus, we can create a transient marker to track the device of the model. Because it's a registered
        # buffer, it is moved alongside the rest of the model. Thus, we can reliably query self._device_indicator.device to determine
        # which device the model currently resides on.
        self._device_indicator = nn.Buffer(torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_indicator.device

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        return {}

    def on_train_from_scratch_start(self, system, **kwargs) -> None:
        pass
