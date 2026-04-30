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

import math

from typing import Any

import torch

from torch._prims_common import DeviceLikeType
from torchmetrics.image import PeakSignalNoiseRatio

from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


class PSNRMetric(BaseMetric):
    """Peak Signal-to-Noise Ratio metric for images."""

    _NAME = MetricType.PSNR.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
        **kwargs,
    ):
        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for PSNR metric.")
        # Set data_range from kwargs or use default
        self.data_range = kwargs.pop("data_range", 1.0)
        # Finite upper bound for PSNR when images are identical (MSE=0 would give infinity)
        self.max_psnr = 10 * math.log10((self.data_range**2) / (1e-10))
        self._psnr_criterion = PeakSignalNoiseRatio(data_range=self.data_range, **kwargs)
        if device:
            self.to(device)

    def validate_inputs(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Validate that inputs are valid image tensors.

        Args:
            pred: The predicted image tensor of shape [..., h, w].
            target: The target image tensor of shape [..., h, w].
            mask: Optional mask tensor of shape [h, w].
        """
        # Validate input types
        for i, img in enumerate([pred, target]):
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Input {i} must be a torch.Tensor, got {type(img)}")
            if img.dim() < 2:
                raise ValueError(f"Input {i} must have at least 2 dimensions, but got {img.dim()}")

        # Validate shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Predicted and target shapes must match: {pred.shape} vs {target.shape}")

        # Validate mask if provided
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"Mask must be a torch.Tensor, got {type(mask)}")
            if mask.dtype != torch.bool:
                raise ValueError(f"Mask must be boolean tensor, got {mask.dtype}")
            if mask.shape != pred.shape[-2:]:
                raise ValueError(f"Mask shape {mask.shape} must match image spatial dimensions {pred.shape[-2:]}")

    @torch.no_grad()
    def _compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> MetricResult:
        """Compute PSNR between predicted and target images.

        Args:
            pred: The predicted image tensor of shape [..., h, w].
            target: The target image tensor of shape [..., h, w].
            mask: The mask to apply to the images of shape [h, w]. Defaults to None.

        Returns:
            MetricResult: The PSNR value and metadata. The PSNR value is a tensor of shape [batch_size] if there is a batch dimension, otherwise it is a scalar.
        """
        # Store original shapes for metadata
        original_shape = pred.shape

        if mask is not None:
            # Broadcast mask to match the leading dimensions of pred/target
            # If pred is [..., h, w] and mask is [h, w], we need to expand mask to [..., h, w]
            mask_expanded = mask.expand(pred.shape)
            pred = pred[mask_expanded]
            target = target[mask_expanded]

        psnr_value = self._psnr_criterion(pred, target)

        # Handle the case where PSNR is inf (identical frames)
        psnr_value[torch.isinf(psnr_value)] = self.max_psnr

        # Calculate masked pixels correctly
        if mask is not None:
            # Count pixels where mask is True (valid pixels)
            num_masked_pixels = torch.sum(mask).item()
        else:
            # If no mask, all pixels are valid
            num_masked_pixels = original_shape[-2] * original_shape[-1]

        # Create metadata with input information
        metadata = {
            "data_range": self.data_range,
            "input_shape": list(original_shape),
            "masked_pixels": num_masked_pixels,
        }

        return MetricResult(values={self._NAME: psnr_value}, metadata=metadata)

    def aggregate(self) -> dict[AggregationMethod, MetricResult]:
        """Aggregate stored values using the specified method."""
        aggregated_metrics: dict[AggregationMethod, MetricResult] = {}
        if len(self._values) > 0:
            for method in self._aggregation_methods:
                aggregates = aggregate_tensors([value.values[self._NAME] for value in self._values], method=method)
                aggregated_metrics[method] = MetricResult(values={self._NAME: aggregates})
        return aggregated_metrics

    def type(self) -> MetricType:
        """Return the type of the metric."""
        return MetricType.PSNR

    def metadata(self) -> dict[str, Any]:
        """Return the metadata for the metric."""
        return {"data_range": self.data_range}

    def reset(self) -> None:
        """Reset the PSNR criterion."""
        self._psnr_criterion.reset()

    def to(self, device: DeviceLikeType) -> PSNRMetric:
        """Move the metric to the specified device.

        Args:
            device: The device to move the metric to.

        Returns:
            PSNRMetric: The metric instance with the device set.
        """
        super().to(device)
        self._psnr_criterion.to(device)
        return self
