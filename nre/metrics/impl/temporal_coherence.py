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

import numpy as np
import scipy.signal as scipy_signal
import torch

from torch._prims_common import DeviceLikeType

from nre.metrics.impl.utils.feature_extractor import FeatureExtractorFactory
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


class TemporalCoherenceMetric(BaseMetric):
    """Generic temporal coherence metric for video sequences using configurable
    feature extractors."""

    _NAME = MetricType.TEMPORAL_COHERENCE.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: (list[AggregationMethod] | AggregationMethod) = AggregationMethod.MEAN,
        extractor_type: str = "segformer",
        pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
        cache_dir: str | None = None,
        window_size: int = 5,
        feature_batch_size: int | None = 32,
    ):
        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for temporal coherence metric.")

        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractorFactory.create_extractor(
            extractor_type=extractor_type, pretrained_path=pretrained_path, cache_dir=cache_dir, device=device
        )

        # Store configuration
        self.extractor_type = extractor_type
        self.pretrained_path = pretrained_path
        self.cache_dir = cache_dir
        self.window_size = window_size
        self.feature_batch_size = feature_batch_size

        if device:
            self.to(device)

    def validate_inputs(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Validate that inputs are valid sequence tensors.

        Args:
            pred: The predicted sequence tensor of shape [T, C, H, W].
            target: The target sequence tensor of shape [T, C, H, W].
        """
        # Validate input types
        for i, seq in enumerate([pred, target]):
            if not isinstance(seq, torch.Tensor):
                raise TypeError(f"Input {i} must be a torch.Tensor, got {type(seq)}")
            if seq.dim() != 4:
                raise ValueError(f"Input {i} must have 4 dimensions [T, C, H, W], but got {seq.dim()}")

        # Validate shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Predicted and target shapes must match: {pred.shape} vs {target.shape}")

        # Validate sequence length
        if pred.shape[0] < self.window_size:
            raise ValueError(f"Sequence length {pred.shape[0]} must be at least window_size {self.window_size}")

    def _compute_temporal_coherence(self, features: torch.Tensor) -> float:
        """Compute temporal coherence by measuring smoothness of changes.

        Args:
            features: Feature tensor of shape (N_frames, N_dims).

        Returns:
            A float coherence score in (0, 1], higher indicates smoother
            changes.
        """

        if len(features) < 2:
            return 1.0

        # Compute frame-to-frame distances using PyTorch
        diffs = torch.norm(features[1:] - features[:-1], dim=1)

        # Convert to numpy for scipy signal processing (unavoidable for savgol_filter)
        diffs_np = diffs.cpu().numpy()

        # Smooth the differences to get expected transitions
        if len(diffs_np) > self.window_size:
            # Ensure polyorder is less than window_size
            polyorder = min(3, self.window_size - 1)
            smoothed = scipy_signal.savgol_filter(diffs_np, self.window_size, polyorder)
            # Compute deviation from smooth transitions
            deviation = np.abs(diffs_np - smoothed)
            coherence = 1.0 / (1.0 + np.mean(deviation))
        else:
            # Too few frames for smoothing
            coherence = 1.0 / (1.0 + np.std(diffs_np))

        return float(coherence)

    def _compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> MetricResult:
        """Compute temporal coherence between predicted and target sequences.

        Args:
            pred: The predicted sequence tensor of shape [T, C, H, W].
            target: The target sequence tensor of shape [T, C, H, W].

        Returns:
            MetricResult: The temporal coherence metrics and metadata.
        """
        # Store original shapes and device for metadata and device preservation
        original_shape = pred.shape
        original_device = pred.device

        # Extract features
        features_pred = self.feature_extractor.extract_features_batch(
            pred, return_numpy=False, batch_size=self.feature_batch_size
        )
        features_target = self.feature_extractor.extract_features_batch(
            target, return_numpy=False, batch_size=self.feature_batch_size
        )

        # Ensure features are tensors (required for tensor operations)
        assert isinstance(features_pred, torch.Tensor)
        assert isinstance(features_target, torch.Tensor)

        # Compute temporal coherence for both sequences
        pred_coherence = self._compute_temporal_coherence(features_pred)
        target_coherence = self._compute_temporal_coherence(features_target)

        # Compute coherence ratio (pred/target with numerical stability)
        coherence_ratio = pred_coherence / (target_coherence + 1e-8)
        coherence_tensor = torch.tensor(coherence_ratio, device=original_device)

        # Create metadata with input information
        metadata = {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "input_shape": list(original_shape),
            "feature_dim": self.feature_extractor.feature_dim,
            "window_size": self.window_size,
            "pred_coherence": pred_coherence,
            "target_coherence": target_coherence,
        }

        return MetricResult(values={self._NAME: coherence_tensor}, metadata=metadata)

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
        return MetricType.TEMPORAL_COHERENCE

    def metadata(self) -> dict[str, Any]:
        """Return the metadata for the metric."""
        return {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "feature_dim": self.feature_extractor.feature_dim,
            "window_size": self.window_size,
        }

    def reset(self) -> None:
        """Reset the metric state."""
        pass
