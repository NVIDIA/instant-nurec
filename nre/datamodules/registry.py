# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    from nre.datamodules.base import BaseDataModule

datamodules: dict[str, Callable[..., "BaseDataModule"]] = {}


def register(name):
    def decorator(cls):
        datamodules[name] = cls
        return cls

    return decorator


def make(name, config, **kwargs) -> "BaseDataModule":
    datamodule = datamodules[name](config, **kwargs)
    return datamodule
