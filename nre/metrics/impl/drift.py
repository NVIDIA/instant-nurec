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

from typing import Any, Tuple

import numpy as np
import torch

from torch._prims_common import DeviceLikeType

from nre.metrics.impl.utils.feature_extractor import FeatureExtractorFactory
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


class FeatureDriftMetric(BaseMetric):
    """Generic feature drift metric for comparing sequences."""

    _NAME = MetricType.FEATURE_DRIFT.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: (list[AggregationMethod] | AggregationMethod) = AggregationMethod.MEAN,
        extractor_type: str = "segformer",
        pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
        cache_dir: str | None = None,
        window_size: int = 10,
        feature_batch_size: int | None = 32,
    ):
        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for feature drift metric.")

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

    def _compute_drift_statistics(
        self, gt_features: torch.Tensor, gen_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Measure progressive deviation from ground truth over time.

        Args:
            gt_features: Ground-truth features of shape (N_frames, N_dims).
            gen_features: Generated features of shape (N_frames, N_dims).

        Returns:
            A tuple of (avg_drift, drift_values, drift_correlation).
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(gt_features, np.ndarray):
            gt_features = torch.from_numpy(gt_features).to(self.device)
        if isinstance(gen_features, np.ndarray):
            gen_features = torch.from_numpy(gen_features).to(self.device)

        device = gt_features.device
        n_frames = gt_features.shape[0]

        # Pre-allocate drift values tensor
        drift_values = torch.zeros(n_frames, device=device)

        for i in range(n_frames):
            # Compute windowed average distance
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(n_frames, i + self.window_size // 2 + 1)

            # Compute distances for window frames
            window_gt = gt_features[start_idx:end_idx]
            window_gen = gen_features[start_idx:end_idx]

            # Compute L2 distances for each frame in window
            window_dists = torch.norm(window_gt - window_gen, dim=1)

            # Average distance in window
            drift_values[i] = torch.mean(window_dists)

        # Check if drift increases over time (quality degradation)
        time_indices = torch.arange(n_frames, device=device, dtype=torch.float32)

        # Handle edge case where drift_values are constant (std = 0)
        drift_std = torch.std(drift_values)
        if drift_std == 0:
            correlation = torch.tensor(0.0, device=device)
        else:
            # Compute correlation coefficient manually
            drift_mean = torch.mean(drift_values)
            time_mean = torch.mean(time_indices)

            drift_centered = drift_values - drift_mean
            time_centered = time_indices - time_mean

            numerator = torch.sum(drift_centered * time_centered)
            denominator = torch.sqrt(torch.sum(drift_centered**2) * torch.sum(time_centered**2))

            if denominator == 0:
                correlation = torch.tensor(0.0, device=device)
            else:
                correlation = numerator / denominator
                # Handle NaN case
                if torch.isnan(correlation):
                    correlation = torch.tensor(0.0, device=device)

        avg_drift = torch.mean(drift_values)

        return avg_drift, drift_values, correlation

    def _compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> MetricResult:
        """Compute feature drift between predicted and target sequences.

        Args:
            pred: The predicted sequence tensor of shape [T, C, H, W].
            target: The target sequence tensor of shape [T, C, H, W].

        Returns:
            MetricResult: The drift metrics and metadata.
        """
        # Store original shapes for metadata
        original_shape = pred.shape

        # Extract features for both sequences (keep as tensors)
        features_pred = self.feature_extractor.extract_features_batch(
            pred, return_numpy=False, batch_size=self.feature_batch_size
        )
        features_target = self.feature_extractor.extract_features_batch(
            target, return_numpy=False, batch_size=self.feature_batch_size
        )

        # Ensure features are tensors on the same device
        if isinstance(features_pred, np.ndarray):
            features_pred = torch.from_numpy(features_pred).to(self.device)
        if isinstance(features_target, np.ndarray):
            features_target = torch.from_numpy(features_target).to(self.device)

        # Compute drift statistics (pred vs target comparison)
        avg_drift, drift_values, drift_correlation = self._compute_drift_statistics(features_target, features_pred)

        # Create metadata (convert tensors to Python types for metadata)
        metadata = {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "input_shape": list(original_shape),
            "feature_dim": self.feature_extractor.feature_dim,
            "window_size": self.window_size,
            "avg_drift": float(avg_drift.item()),
            "drift_correlation": float(drift_correlation.item()),
            "drift_values_shape": list(drift_values.shape),
        }

        return MetricResult(
            values={
                self._NAME: avg_drift,
                f"{self._NAME}_correlation": drift_correlation,
            },
            metadata=metadata,
        )

    def aggregate(self) -> dict[AggregationMethod, MetricResult]:
        """Aggregate stored values using the specified method."""
        aggregated_metrics: dict[AggregationMethod, MetricResult] = {}
        if len(self._values) > 0:
            for method in self._aggregation_methods:
                # Aggregate main drift metric
                aggregates = aggregate_tensors([value.values[self._NAME] for value in self._values], method=method)
                # Aggregate correlation metric
                corr_aggregates = aggregate_tensors(
                    [value.values[f"{self._NAME}_correlation"] for value in self._values], method=method
                )
                aggregated_metrics[method] = MetricResult(
                    values={
                        self._NAME: aggregates,
                        f"{self._NAME}_correlation": corr_aggregates,
                    }
                )
        return aggregated_metrics

    def type(self) -> MetricType:
        """Return the type of the metric."""
        return MetricType.FEATURE_DRIFT

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
