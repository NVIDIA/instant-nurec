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

from typing import Any

import torch
import torch.nn.functional as F

from torch._prims_common import DeviceLikeType

from nre.metrics.impl.utils.feature_extractor import FeatureExtractorFactory
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


class PerceptualMetric(BaseMetric):
    """Generic perceptual quality metric for images using configurable feature extractors."""

    _NAME = MetricType.PERCEPTUAL.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
        extractor_type: str = "segformer",
        pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
        cache_dir: str | None = None,
        feature_batch_size: int | None = 32,
    ):
        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for perceptual metric.")

        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractorFactory.create_extractor(
            extractor_type=extractor_type, pretrained_path=pretrained_path, cache_dir=cache_dir, device=device
        )

        # Store configuration
        self.extractor_type = extractor_type
        self.pretrained_path = pretrained_path
        self.cache_dir = cache_dir
        self.feature_batch_size = feature_batch_size

        if device:
            self.to(device)

    def validate_inputs(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Validate that inputs are valid image tensors.

        Args:
            pred: The predicted image tensor of shape [..., h, w].
            target: The target image tensor of shape [..., h, w].
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

    def _compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> MetricResult:
        """Compute perceptual distance between predicted and target images.

        Args:
            pred: The predicted image tensor of shape [..., h, w].
            target: The target image tensor of shape [..., h, w].

        Returns:
            MetricResult: The perceptual distance value and metadata.
        """
        # Store original shapes for metadata
        original_shape = pred.shape

        # Ensure we have batch dimension
        if pred.dim() == 3:  # (C, H, W)
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # Extract features
        features_pred = self.feature_extractor.extract_features_batch(
            pred, batch_size=self.feature_batch_size, return_numpy=False
        )
        features_target = self.feature_extractor.extract_features_batch(
            target, batch_size=self.feature_batch_size, return_numpy=False
        )

        # Ensure features are tensors (required for F.mse_loss)
        if not isinstance(features_pred, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for features_pred, got {type(features_pred)}")
        if not isinstance(features_target, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for features_target, got {type(features_target)}")

        # Compute perceptual distance (L2 distance in feature space)
        perceptual_distance = F.mse_loss(features_pred, features_target)

        # Create metadata with input information
        metadata = {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "input_shape": list(original_shape),
            "feature_dim": self.feature_extractor.feature_dim,
        }

        return MetricResult(values={self._NAME: perceptual_distance}, metadata=metadata)

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
        return MetricType.PERCEPTUAL

    def metadata(self) -> dict[str, Any]:
        """Return the metadata for the metric."""
        return {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "feature_dim": self.feature_extractor.feature_dim,
        }

    def reset(self) -> None:
        """Reset the metric state."""
        pass
