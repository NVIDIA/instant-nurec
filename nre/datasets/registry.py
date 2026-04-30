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
    from nre.datasets.base import BaseDataset

datasets: dict[str, Callable[..., "BaseDataset"]] = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls

    return decorator


def make(name, config, split) -> "BaseDataset":
    # ncore dataset need some values from other parts of the config.
    # In the long run we should move towards this not being the case.
    if name == "ncore":
        dataset = datasets[name](config, split)
    else:
        dataset = datasets[name](config.dataset, split)

    return dataset
