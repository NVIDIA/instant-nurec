# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import warnings

from typing import Any, Literal, Optional

from pydantic import field_validator

from nre.config.base_schema import BaseConfigSchema


class BaseDatamoduleConfig(BaseConfigSchema):
    update_n_epochs: int
    train_num_workers: int
    train_batch_size: int
    val_num_workers: int
    test_num_workers: int

    mock_dataloader: bool


class PrefetchConfig(BaseConfigSchema):
    enabled: bool
    queue_size: int


class SODatamoduleConfig(BaseDatamoduleConfig):
    name: Literal["so"]

    rolling_buffer: Optional[Any] = None

    @field_validator("rolling_buffer")
    def warn_if_rolling_buffer(cls, value: Optional[Any]) -> None:
        if value is not None:
            warnings.warn("Rolling buffer is deprecated and ignored")
        return None

    prefetch: PrefetchConfig
