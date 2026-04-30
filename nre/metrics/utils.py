# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from enum import Enum, auto

import torch


class AggregationMethod(Enum):
    MEAN = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    WEIGHTED_MEAN = auto()


def aggregate_tensors(
    tensors: list[torch.Tensor],
    weights: list[torch.Tensor] | None = None,
    method: AggregationMethod = AggregationMethod.MEAN,
) -> torch.Tensor:
    """Aggregate a list of tensors using the specified method.

    Args:
        tensors: List of tensors to aggregate.
        weights: List of weights to use for weighted mean.
        method: Aggregation method to use.

    Returns:
        Aggregated Torch.tensor.
    """
    if method == AggregationMethod.MEAN:
        return torch.stack(tensors).mean(dim=0)
    elif method == AggregationMethod.SUM:
        return torch.stack(tensors).sum(dim=0)
    elif method == AggregationMethod.MIN:
        return torch.stack(tensors).min(dim=0)[0]  # min returns (values, indices)
    elif method == AggregationMethod.MAX:
        return torch.stack(tensors).max(dim=0)[0]  # max returns (values, indices)
    elif method == AggregationMethod.WEIGHTED_MEAN:
        if weights is None:
            raise ValueError("Weights must be provided for WEIGHTED_MEAN")
        return aggregate_weighted_mean(tensors, weights)
    raise ValueError(f"Unsupported aggregation method: {method}")


def aggregate_weighted_mean(values: list[torch.Tensor], weights: list[torch.Tensor]) -> torch.Tensor:
    """Aggregate tensors using weighted mean."""
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length")
    if len(values) == 0:
        raise ValueError("Cannot aggregate empty list")

    stacked_values = torch.stack(values)
    stacked_weights = torch.stack(weights)

    # Avoid division by zero
    total_weight = stacked_weights.sum(dim=0)
    # If total weight is zero, return NaN for that category
    result = torch.where(
        total_weight > 0,
        (stacked_values * stacked_weights).sum(dim=0) / total_weight,
        torch.full_like(total_weight, float("nan")),
    )

    return result
