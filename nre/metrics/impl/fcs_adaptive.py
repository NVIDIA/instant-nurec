# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""FCS Adaptive metric implementation.

This module implements the Feature Coverage Score (FCS) Adaptive metric which
measures how well the generated features cover the ground truth feature space
using adaptive thresholds based on local neighborhood structures.

The FCS Adaptive metric computes coverage in both directions (GT→Gen and
Gen→GT) and provides a symmetric measure of feature space coverage.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors
from torch._prims_common import DeviceLikeType

from nre.metrics.impl.utils.feature_extractor import FeatureExtractorFactory
from nre.metrics.metric import AggregationMethod, BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import aggregate_tensors


class FCSAdaptiveMetric(BaseMetric):
    """Feature Coverage Score (FCS) Adaptive metric for measuring feature
    space coverage.

    This metric computes how well the generated features cover the ground truth
    feature space using adaptive thresholds based on k-nearest neighbor
    distances. It provides both directional coverage scores (GT→Gen and
    Gen→GT) and a symmetric overall coverage score.

    The adaptive nature means that the coverage threshold for each sample is
    determined by its local neighborhood structure in the feature space.
    """

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: (List[AggregationMethod] | AggregationMethod) = AggregationMethod.MEAN,
        extractor_type: str = "segformer",
        pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
        cache_dir: str | None = None,
        n_neighbors: int = 15,
        feature_batch_size: int | None = 32,
    ):
        """Initialize the FCS Adaptive metric.

        Args:
            device: Device to run computations on.
            aggregation_methods: Methods for aggregating results across
                multiple computations.
            extractor_type: Type of feature extractor to use.
            pretrained_path: Path to pretrained model weights.
            cache_dir: Directory to cache pretrained models.
            n_neighbors: Number of neighbors for adaptive threshold
                computation.
            feature_batch_size: Batch size for feature extraction.
        """

        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for FCS Adaptive metric.")

        self.extractor_type = extractor_type
        self.pretrained_path = pretrained_path
        self.cache_dir = cache_dir
        self.n_neighbors = n_neighbors
        self.feature_batch_size = feature_batch_size

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractorFactory.create_extractor(
            extractor_type=extractor_type,
            pretrained_path=pretrained_path,
            cache_dir=cache_dir,
            device=device,
        )

        if device:
            self.to(device)

    def validate_inputs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Validate that inputs are valid tensors for FCS Adaptive computation.

        Args:
            pred: The predicted tensor of shape [..., H, W] or [T, C, H, W].
            target: The target tensor of shape [..., H, W] or [T, C, H, W].
        """
        # Validate input types
        for i, tensor in enumerate([pred, target]):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Input {i} must be a torch.Tensor, got {type(tensor)}")
            if tensor.dim() < 3:
                raise ValueError(f"Input {i} must have at least 3 dimensions, but got {tensor.dim()}")

        # Validate shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Predicted and target shapes must match: {pred.shape} vs {target.shape}")

        # Validate minimum number of samples for k-NN
        batch_size = pred.shape[0]
        if batch_size < self.n_neighbors + 1:
            raise ValueError(
                f"Need at least {self.n_neighbors + 1} samples for k-NN with k={self.n_neighbors}, but got {batch_size}"
            )

    def compute_adaptive_fcs(
        self,
        gt_features: np.ndarray,
        gen_features: np.ndarray,
        k: Optional[int] = None,
    ) -> Tuple[float, float, float, np.ndarray]:
        """Compute adaptive Feature Coverage Score (FCS).

        Args:
            gt_features: Ground truth features of shape (n_samples, n_dims).
            gen_features: Generated features of shape (n_samples, n_dims).
            k: Number of neighbors for threshold computation. If None, uses
                self.n_neighbors.

        Returns:
            Tuple containing:
            - fcs_adaptive: Symmetric FCS score (average of both directions)
            - fcs_gt_to_gen: Coverage from GT to Generated
            - fcs_gen_to_gt: Coverage from Generated to GT
            - covered_mask: Boolean mask indicating covered GT samples
        """
        if k is None:
            k = self.n_neighbors

        # 1) Compute GT→GT pairwise distances
        nbrs_gt = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(gt_features)
        distances_gt, _ = nbrs_gt.kneighbors(gt_features)

        # Remove self-distances (first column) and get k-th neighbor distance
        gt_thresholds = distances_gt[:, k]  # k-th nearest neighbor distance

        # 2) Compute GT→Gen cross distances
        nbrs_gen = NearestNeighbors(n_neighbors=1, metric="cosine").fit(gen_features)
        cross_distances, _ = nbrs_gen.kneighbors(gt_features)
        cross_distances = cross_distances.flatten()

        # 3) Compute coverage mask for GT→Gen direction
        covered_gt = cross_distances <= gt_thresholds
        fcs_gt_to_gen = float(np.mean(covered_gt))

        # 4) Compute symmetric direction: Gen→GT
        nbrs_gen_self = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(gen_features)
        distances_gen, _ = nbrs_gen_self.kneighbors(gen_features)
        gen_thresholds = distances_gen[:, k]

        # Gen→GT cross distances
        nbrs_gt_for_gen = NearestNeighbors(n_neighbors=1, metric="cosine").fit(gt_features)
        cross_distances_gen, _ = nbrs_gt_for_gen.kneighbors(gen_features)
        cross_distances_gen = cross_distances_gen.flatten()

        # Coverage mask for Gen→GT direction
        covered_gen = cross_distances_gen <= gen_thresholds
        fcs_gen_to_gt = float(np.mean(covered_gen))

        # 5) Symmetric FCS score
        fcs_adaptive = 0.5 * (fcs_gt_to_gen + fcs_gen_to_gt)

        return fcs_adaptive, fcs_gt_to_gen, fcs_gen_to_gt, covered_gt

    def _compute_fcs_adaptive_metrics(self, gt_features: np.ndarray, gen_features: np.ndarray) -> Dict[str, Any]:
        """Compute FCS Adaptive metrics between ground truth and generated
        features.

        Args:
            gt_features: Ground truth features of shape (n_samples, n_dims).
            gen_features: Generated features of shape (n_samples, n_dims).

        Returns:
            Dictionary containing FCS Adaptive metrics and related statistics.
        """
        # Compute adaptive FCS
        (
            fcs_adaptive,
            fcs_gt_to_gen,
            fcs_gen_to_gt,
            covered_mask,
        ) = self.compute_adaptive_fcs(gt_features, gen_features)

        # Additional coverage statistics
        coverage_ratio = float(np.sum(covered_mask)) / len(covered_mask)
        uncovered_samples = int(np.sum(~covered_mask))

        # Compute coverage quality metrics
        if np.any(covered_mask):
            # Average distance for covered samples
            nbrs = NearestNeighbors(n_neighbors=1, metric="cosine").fit(gen_features)
            distances, _ = nbrs.kneighbors(gt_features)
            covered_distances = distances[covered_mask].flatten()
            avg_covered_distance = float(np.mean(covered_distances))
            std_covered_distance = float(np.std(covered_distances))
        else:
            avg_covered_distance = float("inf")
            std_covered_distance = 0.0

        # Compute threshold statistics
        nbrs_gt = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric="cosine").fit(gt_features)
        distances_gt, _ = nbrs_gt.kneighbors(gt_features)
        thresholds = distances_gt[:, self.n_neighbors]
        avg_threshold = float(np.mean(thresholds))
        std_threshold = float(np.std(thresholds))

        results = {
            "fcs_adaptive": fcs_adaptive,
            "fcs_gt_to_gen": fcs_gt_to_gen,
            "fcs_gen_to_gt": fcs_gen_to_gt,
            "coverage_ratio": coverage_ratio,
            "uncovered_samples": uncovered_samples,
            "avg_covered_distance": avg_covered_distance,
            "std_covered_distance": std_covered_distance,
            "avg_threshold": avg_threshold,
            "std_threshold": std_threshold,
            "n_neighbors": self.n_neighbors,
        }

        # Add sample and feature dimension info
        results["gt_samples"] = gt_features.shape[0]
        results["gen_samples"] = gen_features.shape[0]
        results["feature_dim"] = gt_features.shape[1]

        return results

    def _compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> MetricResult:
        """Compute FCS Adaptive between predicted and target tensors.

        Args:
            pred: The predicted tensor of shape [..., H, W] or [T, C, H, W].
            target: The target tensor of shape [..., H, W] or [T, C, H, W].

        Returns:
            MetricResult: The FCS Adaptive metrics and metadata.
        """
        # Store original shapes for metadata
        original_shape = pred.shape

        # Ensure we have batch dimension for feature extraction
        if pred.dim() == 3:  # (C, H, W)
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # Extract features using the feature extractor
        gt_features = self.feature_extractor.extract_features_batch(target, batch_size=self.feature_batch_size)
        gen_features = self.feature_extractor.extract_features_batch(pred, batch_size=self.feature_batch_size)

        # Preserve device information before numpy conversion
        original_device = None
        if isinstance(gt_features, torch.Tensor):
            original_device = gt_features.device
            gt_features = gt_features.cpu().numpy()
        elif isinstance(gen_features, torch.Tensor):
            original_device = gen_features.device
            gen_features = gen_features.cpu().numpy()
        else:
            # Fallback: use self.device if both are numpy arrays
            if self.device is not None:
                original_device = torch.device(self.device)
            else:
                original_device = torch.device("cpu")

        # Ensure both are numpy arrays
        if isinstance(gen_features, torch.Tensor):
            gen_features = gen_features.cpu().numpy()

        # Compute FCS Adaptive metrics
        metrics = self._compute_fcs_adaptive_metrics(gt_features, gen_features)

        # Create metadata
        metadata = {
            "original_shape": original_shape,
            "extractor_type": self.extractor_type,
            "n_neighbors": self.n_neighbors,
            "feature_batch_size": self.feature_batch_size,
        }

        # Convert result back to tensor on original device
        fcs_tensor = torch.tensor(metrics["fcs_adaptive"])
        if original_device is not None:
            fcs_tensor = fcs_tensor.to(original_device)

        return MetricResult(
            values={"fcs_adaptive": fcs_tensor},
            metadata=metadata,
        )

    def aggregate(self) -> Dict[AggregationMethod, MetricResult]:
        """Aggregate stored values using the specified method."""
        aggregated_metrics: Dict[AggregationMethod, MetricResult] = {}
        if len(self._values) > 0:
            for method in self._aggregation_methods:
                # Aggregate main FCS Adaptive metric
                aggregates = aggregate_tensors(
                    [value.values["fcs_adaptive"] for value in self._values],
                    method=method,
                )
                aggregated_metrics[method] = MetricResult(
                    values={"fcs_adaptive": aggregates},
                    metadata=self._values[0].metadata,
                )
        return aggregated_metrics

    def type(self) -> MetricType:
        """Return the metric type."""
        return MetricType.FCS_ADAPTIVE

    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the metric configuration."""
        return {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "n_neighbors": self.n_neighbors,
            "feature_batch_size": self.feature_batch_size,
        }

    def reset(self) -> None:
        """Reset the metric state."""
        pass
