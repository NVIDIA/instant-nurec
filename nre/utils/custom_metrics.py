# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any

import torch

from torchmetrics import Metric


class Delta1Accuracy(Metric):
    """
    Computes the δ₁ (delta-1) accuracy for depth estimation.

    Delta-1 accuracy measures the percentage of predicted depth values that are within a threshold
    δ of the ground truth, based on the ratio between prediction and target. It is defined as:

        δ₁ = (1/N) ∑ₙ [max(ŷₙ / yₙ, yₙ / ŷₙ) < δ]

    where:
        - ŷₙ: predicted depth at pixel n
        - yₙ: ground truth depth at pixel n
        - δ: typically set to 1.25 for δ₁ accuracy

    This metric is commonly used in monocular depth estimation tasks.

    Args:
        delta (float): Threshold value to determine correct predictions. Default is 1.25.
    """

    total: torch.Tensor
    correct: torch.Tensor

    def __init__(self, delta: float = 1.25, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delta = delta
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Ensure preds and target are of the same shape
        assert preds.shape == target.shape, "Shape mismatch between preds and target"

        # Avoid division by zero
        mask = (target > 0) & (preds > 0)
        preds = preds[mask]
        target = target[mask]

        # Compute max ratio
        ratio = torch.max(preds / target, target / preds)
        correct = (ratio < self.delta).sum()
        total = ratio.numel()

        self.correct += correct
        self.total += total

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total


class AbsRelError(Metric):
    """
    Computes the Absolute Relative Error (AbsRel) for depth estimation.

    The AbsRel metric measures the average of the absolute difference between predicted and ground truth
    depth, normalized by the ground truth:

        AbsRel = (1/N) ∑ₙ |ŷₙ - yₙ| / yₙ

    where:
        - ŷₙ: predicted depth at pixel n
        - yₙ: ground truth depth at pixel n

    It is a widely used metric to evaluate the accuracy of monocular depth predictions,
    particularly for its sensitivity to under- or over-estimations across a range of depths.

    Args:
        max_rel_error: If set, only pixels with |ŷ - y| / y <= max_rel_error contribute to the mean.
            Pixels with larger absolute relative error are excluded from the sum and count.

    Note:
        When ``max_rel_error`` is set and no valid pixel falls below the outlier threshold
        (i.e. every pixel is an outlier, or the input had no valid pixels to begin with),
        ``compute()`` returns a sentinel value exactly equal to ``max_rel_error``. Consumers
        of this metric (W&B plots, regression tests, etc.) should therefore treat a value
        of exactly ``max_rel_error`` as "frame was fully saturated / had no usable pixels",
        not as a real measurement.
    """

    sum_absrel: torch.Tensor
    total: torch.Tensor

    def __init__(self, max_rel_error: float | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_rel_error = max_rel_error
        self.add_state("sum_absrel", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.shape == target.shape, "Shape mismatch between preds and target"

        mask = target > 0  # avoid divide-by-zero
        preds = preds[mask]
        target = target[mask]

        absrel = torch.abs(preds - target) / target
        if self.max_rel_error is not None:
            absrel = absrel[absrel <= self.max_rel_error]
        self.sum_absrel += absrel.sum()
        self.total += absrel.numel()

    def compute(self) -> torch.Tensor:
        if self.total.item() == 0 and self.max_rel_error is not None:
            return torch.tensor(self.max_rel_error, device=self.sum_absrel.device, dtype=self.sum_absrel.dtype)
        return self.sum_absrel / self.total
