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
    from libs.losses.orchestration.config import RayReduceFn

reduce_fns: dict[str, Callable[..., "RayReduceFn"]] = {}


def make(name: str, config):
    return reduce_fns[name](config)


def register(name: str):
    def decorator(fn: Callable[..., "RayReduceFn"]):
        reduce_fns[name] = fn
        return fn

    return decorator
