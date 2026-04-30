# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omegaconf
import torch

from libs.losses.models.reduce_functions.registry import register as register_reduce_fn
from libs.losses.orchestration.config import RayReduceFn, ReduceConfig


@register_reduce_fn("mean")
class MeanReduceFn(RayReduceFn):
    """Standard mean reduction"""

    def __init__(self, config: omegaconf.dictconfig.DictConfig, **kwargs) -> None:
        pass

    def __call__(self, value: torch.Tensor, reduce_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if reduce_mask is not None:
            return (value * reduce_mask).sum() / reduce_mask.sum().clamp(min=1)
        return value.mean()


@register_reduce_fn("quantile")
class QuantileMeanReduceFn(RayReduceFn):
    """Filter out top quantile before applying mean"""

    def __init__(self, config: ReduceConfig, **kwargs) -> None:
        self.quantile = config.quantile
        if self.quantile is not None:
            assert self.quantile > 0 and self.quantile <= 1, "quantile needs to be in ]0,1]"

    def __call__(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        flattened_value = value.flatten()
        n_values = len(value.flatten())
        assert self.quantile is not None
        filtered_idxs = flattened_value.argsort()[: int(n_values * self.quantile)]
        if filtered_idxs.nelement() == 0:
            return torch.tensor(0.0, device=value.device)
        return value[filtered_idxs].mean()


@register_reduce_fn("sum")
class SumReduceFn(RayReduceFn):
    """Standard sum reduction"""

    def __init__(self, config: ReduceConfig, **kwargs) -> None:
        pass

    def __call__(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return value.sum()
