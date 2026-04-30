# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Higher-order moments metric implementation (D-Skew and D-Kurtosis).

This module implements metrics for comparing higher-order statistical moments
between ground truth and generated feature distributions. It supports both
third-order moments (skewness) and fourth-order moments (kurtosis).
"""

from enum import Enum
from typing import Any, Dict, List, Union

import torch

from torch._prims_common import DeviceLikeType

from nre.metrics.impl.utils.feature_extractor import FeatureExtractorFactory
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod


class MomentType(Enum):
    """Enum for supported moment types."""

    SKEWNESS = "skewness"  # Third-order moment
    KURTOSIS = "kurtosis"  # Fourth-order moment


class HigherOrderMomentsMetric(BaseMetric):
    """Higher-order moments metric for comparing feature distributions.

    This metric computes differences in higher-order statistical moments
    (skewness or kurtosis) between ground truth and generated features.

    Attributes:
        moment_type: Type of moment to compute (SKEWNESS or KURTOSIS).
        extractor_type: Type of feature extractor to use.
        pretrained_path: Path to pretrained model weights.
        feature_batch_size: Batch size for feature extraction.
        device: Device to run computations on.
    """

    def __init__(
        self,
        moment_type: Union[MomentType, str] = MomentType.SKEWNESS,
        device: DeviceLikeType | None = None,
        extractor_type: str = "segformer",
        pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
        cache_dir: str | None = None,
        feature_batch_size: int = 32,
        aggregation_methods: Union[AggregationMethod, List[AggregationMethod]] = AggregationMethod.MEAN,
    ) -> None:
        """Initialize the HigherOrderMomentsMetric.

        Args:
            moment_type: Type of moment to compute (SKEWNESS or KURTOSIS).
            device: Device to run computations on.
            extractor_type: Type of feature extractor.
            pretrained_path: Path to pretrained model.
            cache_dir: Directory to cache pretrained models.
            feature_batch_size: Batch size for feature extraction.
            aggregation_methods: Methods for aggregating results.

        Raises:
            ValueError: If weighted mean aggregation is requested or if
                moment_type is invalid.
        """

        super().__init__(device=device, aggregation_methods=aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean aggregation is not supported for higher-order moments metrics")

        # Convert string to enum if needed
        if isinstance(moment_type, str):
            moment_type = MomentType(moment_type.lower())

        self.moment_type = moment_type
        self.device = device
        self.extractor_type = extractor_type
        self.pretrained_path = pretrained_path
        self.cache_dir = cache_dir
        self.feature_batch_size = feature_batch_size

        # Initialize feature extractor
        self.extractor = FeatureExtractorFactory.create_extractor(
            extractor_type=extractor_type,
            pretrained_path=pretrained_path,
            cache_dir=cache_dir,
            device=device,
        )

    def validate_inputs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Validate input tensors.

        Args:
            pred: The predicted tensor of shape [..., H, W] or [T, C, H, W].
            target: The target tensor of shape [..., H, W] or [T, C, H, W].

        Raises:
            TypeError: If inputs are not torch.Tensors.
            ValueError: If tensor shapes don't match or have wrong dimensions.
        """

        # Check types
        for i, tensor in enumerate([pred, target]):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Input {i} must be a torch.Tensor")

        # Check shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Input shapes must match. Got pred: {pred.shape}, target: {target.shape}")

        # Check dimensions (3D or 4D)
        if pred.ndim not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D tensors, got {pred.ndim}D")

    def compute_moments(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute statistical moments for features.

        Args:
            features: Feature tensor of shape (n_samples, n_features).

        Returns:
            Dictionary containing computed moments and statistics.

        Raises:
            ValueError: If features tensor is empty or has wrong shape.
        """

        if features.numel() == 0:
            raise ValueError("Features tensor cannot be empty")

        if features.ndim != 2:
            raise ValueError(f"Features must be 2D tensor, got {features.ndim}D")

        n_samples, _ = features.shape

        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples for moment computation, got {n_samples}")

        # Compute basic statistics
        mean = torch.mean(features, dim=0)
        centered = features - mean

        # Compute variance for normalization (using Bessel's correction)
        variance = torch.var(features, dim=0, unbiased=True)
        std = torch.sqrt(variance)

        # Avoid division by zero
        std_safe = torch.where(std > 1e-8, std, torch.ones_like(std))

        moments = {
            "mean": mean,
            "centered": centered,
            "variance": variance,
            "std": std,
        }

        if self.moment_type == MomentType.SKEWNESS:
            # Third-order moment (skewness)
            skewness = torch.mean((centered / std_safe) ** 3, dim=0)
            moments["skewness"] = skewness

        elif self.moment_type == MomentType.KURTOSIS:
            # Fourth-order moment (kurtosis)
            kurtosis = torch.mean((centered / std_safe) ** 4, dim=0)
            normalized_kurtosis = kurtosis
            excess_kurtosis = kurtosis - 3.0  # Excess kurtosis

            moments.update(
                {
                    "kurtosis": kurtosis,
                    "normalized_kurtosis": normalized_kurtosis,
                    "excess_kurtosis": excess_kurtosis,
                }
            )

        return moments

    def _compute_moment_metrics(self, gt_features: torch.Tensor, gen_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute moment-based metrics between GT and generated features.

        Args:
            gt_features: Ground truth features (n_samples, n_features).
            gen_features: Generated features (n_samples, n_features).

        Returns:
            Dictionary containing computed metrics as tensors.
        """

        # Compute moments for both distributions
        gt_moments = self.compute_moments(gt_features)
        gen_moments = self.compute_moments(gen_features)

        metrics = {}

        if self.moment_type == MomentType.SKEWNESS:
            # Skewness-based metrics
            gt_skew = gt_moments["skewness"]
            gen_skew = gen_moments["skewness"]

            skew_diff = torch.abs(gt_skew - gen_skew)

            metrics.update(
                {
                    "d_skew": torch.mean(skew_diff),
                    "gt_skewness_mean": torch.mean(gt_skew),
                    "gen_skewness_mean": torch.mean(gen_skew),
                    "gt_skewness_std": torch.std(gt_skew),
                    "gen_skewness_std": torch.std(gen_skew),
                    "max_skewness_diff": torch.max(skew_diff),
                    "mean_skewness_diff": torch.mean(skew_diff),
                    "skewness_diff_ratio": torch.mean(skew_diff) / (torch.mean(torch.abs(gt_skew)) + 1e-8),
                }
            )

        elif self.moment_type == MomentType.KURTOSIS:
            # Kurtosis-based metrics
            gt_kurt = gt_moments["kurtosis"]
            gen_kurt = gen_moments["kurtosis"]
            gt_excess = gt_moments["excess_kurtosis"]
            gen_excess = gen_moments["excess_kurtosis"]

            kurt_diff = torch.abs(gt_kurt - gen_kurt)
            excess_diff = torch.abs(gt_excess - gen_excess)

            metrics.update(
                {
                    "d_kurt": torch.mean(kurt_diff),
                    "d_kurt_normalized": torch.mean(kurt_diff),
                    "d_kurt_excess": torch.mean(excess_diff),
                    "gt_kurtosis_mean": torch.mean(gt_kurt),
                    "gen_kurtosis_mean": torch.mean(gen_kurt),
                    "gt_kurtosis_std": torch.std(gt_kurt),
                    "gen_kurtosis_std": torch.std(gen_kurt),
                    "max_kurtosis_diff": torch.max(kurt_diff),
                    "mean_kurtosis_diff": torch.mean(kurt_diff),
                    "kurtosis_diff_ratio": torch.mean(kurt_diff) / (torch.mean(torch.abs(gt_kurt)) + 1e-8),
                    "gt_excess_kurtosis_mean": torch.mean(gt_excess),
                    "gen_excess_kurtosis_mean": torch.mean(gen_excess),
                    "max_excess_kurtosis_diff": torch.max(excess_diff),
                    "mean_excess_kurtosis_diff": torch.mean(excess_diff),
                }
            )

        # Common metrics (convert to tensors for type consistency)
        metrics.update(
            {
                "gt_samples": torch.tensor(gt_features.shape[0]),
                "gen_samples": torch.tensor(gen_features.shape[0]),
                "feature_dim": torch.tensor(gt_features.shape[1]),
            }
        )

        return metrics

    def _compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> MetricResult:
        """Compute the higher-order moments metric.

        Args:
            pred: The predicted tensor of shape [..., H, W] or [T, C, H, W].
            target: The target tensor of shape [..., H, W] or [T, C, H, W].

        Returns:
            MetricResult containing computed metrics and metadata.
        """

        # Store original shape for metadata
        original_shape = pred.shape

        # Handle 3D input by adding batch dimension
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # Extract features
        pred_features = self.extractor.extract_features_batch(pred, return_numpy=False)
        target_features = self.extractor.extract_features_batch(target, return_numpy=False)

        # Ensure features are tensors (for mypy)
        assert isinstance(pred_features, torch.Tensor)
        assert isinstance(target_features, torch.Tensor)

        # Compute moment metrics
        tensor_values = self._compute_moment_metrics(target_features, pred_features)

        # Create metadata
        metadata = {
            "original_shape": original_shape,
            "moment_type": self.moment_type.value,
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "feature_batch_size": self.feature_batch_size,
        }

        return MetricResult(values=tensor_values, metadata=metadata)

    def type(self) -> MetricType:
        """Return the metric type based on moment type.

        Returns:
            MetricType.D_SKEW for skewness, MetricType.D_KURT for kurtosis.
        """
        if self.moment_type == MomentType.SKEWNESS:
            return MetricType.D_SKEW
        elif self.moment_type == MomentType.KURTOSIS:
            return MetricType.D_KURT
        else:
            raise ValueError(f"Unknown moment type: {self.moment_type}")

    def aggregate(self) -> Dict[AggregationMethod, MetricResult]:
        """Aggregate stored values using the specified method."""
        from nre.metrics.utils import aggregate_tensors

        aggregated_metrics: Dict[AggregationMethod, MetricResult] = {}
        if len(self._values) > 0:
            for method in self._aggregation_methods:
                # Determine primary metric key based on moment type
                primary_key = "d_skew" if self.moment_type == MomentType.SKEWNESS else "d_kurt"

                # Aggregate primary metric
                aggregates = aggregate_tensors(
                    [value.values[primary_key] for value in self._values],
                    method=method,
                )

                # Create aggregated result
                aggregated_metrics[method] = MetricResult(
                    values={primary_key: aggregates},
                    metadata=self._values[0].metadata,
                )

        return aggregated_metrics

    def metadata(self) -> Dict[str, Any]:
        """Return metric metadata.

        Returns:
            Dictionary containing metric configuration.
        """
        return {
            "moment_type": self.moment_type.value,
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "feature_batch_size": self.feature_batch_size,
        }

    def reset(self) -> None:
        """Reset the metric state."""
        pass


# Convenience factory functions
def create_d_skew_metric(
    device: DeviceLikeType | None = None,
    extractor_type: str = "segformer",
    pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
    feature_batch_size: int = 32,
    aggregation_methods: Union[AggregationMethod, List[AggregationMethod]] = AggregationMethod.MEAN,
) -> HigherOrderMomentsMetric:
    """Create a D-Skew metric instance.

    Args:
        device: Device to run computations on.
        extractor_type: Type of feature extractor.
        pretrained_path: Path to pretrained model.
        feature_batch_size: Batch size for feature extraction.
        aggregation_methods: Methods for aggregating results.

    Returns:
        HigherOrderMomentsMetric configured for skewness computation.
    """
    return HigherOrderMomentsMetric(
        moment_type=MomentType.SKEWNESS,
        device=device,
        extractor_type=extractor_type,
        pretrained_path=pretrained_path,
        feature_batch_size=feature_batch_size,
        aggregation_methods=aggregation_methods,
    )


def create_d_kurtosis_metric(
    device: DeviceLikeType | None = None,
    extractor_type: str = "segformer",
    pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
    feature_batch_size: int = 32,
    aggregation_methods: Union[AggregationMethod, List[AggregationMethod]] = AggregationMethod.MEAN,
) -> HigherOrderMomentsMetric:
    """Create a D-Kurtosis metric instance.

    Args:
        device: Device to run computations on.
        extractor_type: Type of feature extractor.
        pretrained_path: Path to pretrained model.
        feature_batch_size: Batch size for feature extraction.
        aggregation_methods: Methods for aggregating results.

    Returns:
        HigherOrderMomentsMetric configured for kurtosis computation.
    """
    return HigherOrderMomentsMetric(
        moment_type=MomentType.KURTOSIS,
        device=device,
        extractor_type=extractor_type,
        pretrained_path=pretrained_path,
        feature_batch_size=feature_batch_size,
        aggregation_methods=aggregation_methods,
    )
