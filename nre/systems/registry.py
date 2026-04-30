# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch


# Safeguard to avoid a circular dependency due to type annotations
# in case a system inside .base is registered by using the decorator @register.
if TYPE_CHECKING:
    from nre.config.nre import NREConfig
    from nre.systems.base import BaseSystem


systems: dict[str, BaseSystem] = {}


def make(name: str, config: NREConfig, load_from_checkpoint: Optional[str] = None) -> BaseSystem:
    if load_from_checkpoint is not None:
        # Load the model and force it to be on the GPU.
        # Usually checkpoints are saved with the model on the CPU,
        # so we need to move it to the GPU, as that's what callers expect.
        # Without it, we might run into issues where tensors in different devices are
        # used in the same torch operation.
        return systems[name].load_from_checkpoint(
            load_from_checkpoint, strict=False, config=config, map_location=torch.device("cuda")
        )

    return systems[name](config)


def register(name: str):
    def decorator(cls):
        systems[name] = cls
        return cls

    return decorator
