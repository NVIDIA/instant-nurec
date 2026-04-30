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

from typing import Any, Dict

import numpy as np
import sklearn.neighbors as sk_neighbors
import torch

from torch._prims_common import DeviceLikeType

from nre.metrics.impl.utils.feature_extractor import FeatureExtractorFactory
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


class NTDMetric(BaseMetric):
    """Neural Topological Divergence (NTD) metric for comparing feature
    distributions using graph spectral analysis."""

    _NAME = MetricType.NTD.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: (list[AggregationMethod] | AggregationMethod) = AggregationMethod.MEAN,
        extractor_type: str = "segformer",
        pretrained_path: str = ("nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
        cache_dir: str | None = None,
        n_neighbors: int = 12,
        symmetrize: bool = True,
        feature_batch_size: int | None = 32,
    ):
        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for NTD metric.")

        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractorFactory.create_extractor(
            extractor_type=extractor_type,
            pretrained_path=pretrained_path,
            cache_dir=cache_dir,
            device=device,
        )

        # Store configuration
        self.extractor_type = extractor_type
        self.pretrained_path = pretrained_path
        self.cache_dir = cache_dir
        self.n_neighbors = n_neighbors
        self.symmetrize = symmetrize
        self.feature_batch_size = feature_batch_size

        if device:
            self.to(device)

    def validate_inputs(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Validate that inputs are valid tensors for NTD computation.

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

        # Validate minimum number of samples for k-NN graph
        batch_size = pred.shape[0] if pred.dim() == 4 else 1
        if batch_size < self.n_neighbors + 1:
            raise ValueError(
                f"Need at least {self.n_neighbors + 1} samples for k-NN graph "
                f"with k={self.n_neighbors}, but got {batch_size}"
            )

    def compute_graph_spectrum(
        self,
        features: np.ndarray,
        n_neighbors: int | None = None,
        symmetrize: bool | None = None,
    ) -> np.ndarray:
        """
        Compute the non-negative Laplacian spectrum of a k-NN graph.

        Args:
            features: Feature matrix of shape (N_samples, N_dims).
            n_neighbors: Number of neighbors for k-NN graph construction.
                If None, uses self.n_neighbors.
            symmetrize: If True, symmetrize adjacency to make graph undirected.
                If None, uses self.symmetrize.

        Returns:
            Sorted non-negative Laplacian eigenvalues as a 1-D numpy array.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if symmetrize is None:
            symmetrize = self.symmetrize

        # Validate inputs
        if features.shape[0] < n_neighbors + 1:
            raise ValueError(
                f"Need at least {n_neighbors + 1} samples for k-NN graph "
                f"with k={n_neighbors}, but got {features.shape[0]}"
            )

        # Step-1: directed k-NN (no self-loops)
        adjacency = sk_neighbors.kneighbors_graph(
            features,
            n_neighbors=n_neighbors,
            mode="connectivity",
            include_self=False,  # zero diagonal – no self-loops
        )
        adjacency = adjacency.toarray().astype(np.float64)

        # Step-2: symmetrize → undirected graph
        if symmetrize:
            adjacency = np.maximum(adjacency, adjacency.T)

        # Step-3: combinatorial Laplacian L = D − A
        degrees = adjacency.sum(axis=1)
        laplacian = np.diag(degrees) - adjacency

        # Step-4: eigen-spectrum (L is symmetric PSD)
        eigvals = np.linalg.eigvalsh(laplacian)

        # Numerical safety – remove tiny negative values
        eigvals[eigvals < 0] = 0.0
        return np.sort(eigvals)

    def _compute_ntd_metrics(self, gt_features: np.ndarray, gen_features: np.ndarray) -> Dict[str, Any]:
        """Compute Neural Topological Divergence (NTD) metrics.

        Args:
            gt_features: Ground truth features of shape (N_samples, N_dims).
            gen_features: Generated features of shape (N_samples, N_dims).

        Returns:
            Dictionary containing NTD distance and spectral information.
        """

        results: Dict[str, Any] = {}

        # Compute spectral graphs for both feature sets
        spectrum_gt = self.compute_graph_spectrum(gt_features)
        spectrum_gen = self.compute_graph_spectrum(gen_features)

        # Store spectra for analysis
        results["spectrum_gt"] = spectrum_gt
        results["spectrum_gen"] = spectrum_gen

        # Compute NTD as difference in spectral areas (total eigenvalue sums)
        spectral_area_gt = float(np.sum(spectrum_gt))
        spectral_area_gen = float(np.sum(spectrum_gen))
        ntd_distance = float(np.abs(spectral_area_gt - spectral_area_gen))

        results["ntd_distance"] = ntd_distance
        results["spectral_area_gt"] = spectral_area_gt
        results["spectral_area_gen"] = spectral_area_gen

        # Additional spectral metrics
        # Spectral energy (sum of squared eigenvalues)
        spectral_energy_gt = float(np.sum(spectrum_gt**2))
        spectral_energy_gen = float(np.sum(spectrum_gen**2))
        energy_difference = float(np.abs(spectral_energy_gt - spectral_energy_gen))

        results["spectral_energy_gt"] = spectral_energy_gt
        results["spectral_energy_gen"] = spectral_energy_gen
        results["energy_difference"] = energy_difference

        # Spectral shape comparison (normalized spectra)
        if spectral_area_gt > 0 and spectral_area_gen > 0:
            normalized_gt = spectrum_gt / spectral_area_gt
            normalized_gen = spectrum_gen / spectral_area_gen

            # Align spectra lengths for comparison
            min_len = min(len(normalized_gt), len(normalized_gen))
            shape_difference = float(np.sum(np.abs(normalized_gt[:min_len] - normalized_gen[:min_len])))
            results["spectral_shape_difference"] = shape_difference
        else:
            results["spectral_shape_difference"] = 0.0

        # Connectivity statistics
        results["n_neighbors"] = self.n_neighbors
        results["symmetrize"] = self.symmetrize
        results["gt_samples"] = gt_features.shape[0]
        results["gen_samples"] = gen_features.shape[0]
        results["feature_dim"] = gt_features.shape[1]

        return results

    def _compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> MetricResult:
        """Compute NTD between predicted and target tensors.

        Args:
            pred: The predicted tensor of shape [..., H, W] or [T, C, H, W].
            target: The target tensor of shape [..., H, W] or [T, C, H, W].

        Returns:
            MetricResult: The NTD metrics and metadata.
        """
        # Store original shapes for metadata
        original_shape = pred.shape

        # Ensure we have batch dimension for feature extraction
        if pred.dim() == 3:  # (C, H, W)
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # Store original device for result conversion
        original_device = pred.device

        # Extract features for both tensors with optional batching
        features_pred = self.feature_extractor.extract_features_batch(
            pred, return_numpy=True, batch_size=self.feature_batch_size
        )
        features_target = self.feature_extractor.extract_features_batch(
            target, return_numpy=True, batch_size=self.feature_batch_size
        )

        # Ensure features are numpy arrays
        if isinstance(features_pred, torch.Tensor):
            features_pred = features_pred.cpu().numpy()
        if isinstance(features_target, torch.Tensor):
            features_target = features_target.cpu().numpy()

        # Compute NTD metrics (target as ground truth, pred as generated)
        ntd_results = self._compute_ntd_metrics(features_target, features_pred)

        # Convert main metric results back to tensors on original device
        tensor_values = {
            "ntd_distance": torch.tensor(ntd_results["ntd_distance"], device=original_device),
            "spectral_area_gt": torch.tensor(ntd_results["spectral_area_gt"], device=original_device),
            "spectral_area_gen": torch.tensor(ntd_results["spectral_area_gen"], device=original_device),
            "energy_difference": torch.tensor(ntd_results["energy_difference"], device=original_device),
            "spectral_shape_difference": torch.tensor(ntd_results["spectral_shape_difference"], device=original_device),
        }

        # Create metadata with input information (keep as float values for metadata)
        metadata = {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "input_shape": list(original_shape),
            "feature_dim": self.feature_extractor.feature_dim,
            "n_neighbors": self.n_neighbors,
            "symmetrize": self.symmetrize,
            "ntd_distance": ntd_results["ntd_distance"],
            "spectral_area_gt": ntd_results["spectral_area_gt"],
            "spectral_area_gen": ntd_results["spectral_area_gen"],
            "energy_difference": ntd_results["energy_difference"],
            "spectral_shape_difference": ntd_results["spectral_shape_difference"],
        }

        return MetricResult(
            values={
                self._NAME: tensor_values["ntd_distance"],
                f"{self._NAME}_energy_diff": tensor_values["energy_difference"],
                f"{self._NAME}_shape_diff": tensor_values["spectral_shape_difference"],
            },
            metadata=metadata,
        )

    def aggregate(self) -> dict[AggregationMethod, MetricResult]:
        """Aggregate stored values using the specified method."""
        aggregated_metrics: dict[AggregationMethod, MetricResult] = {}
        if len(self._values) > 0:
            for method in self._aggregation_methods:
                # Aggregate main NTD metric
                aggregates = aggregate_tensors(
                    [value.values[self._NAME] for value in self._values],
                    method=method,
                )
                # Aggregate energy difference metric
                energy_aggregates = aggregate_tensors(
                    [value.values[f"{self._NAME}_energy_diff"] for value in self._values],
                    method=method,
                )
                # Aggregate shape difference metric
                shape_aggregates = aggregate_tensors(
                    [value.values[f"{self._NAME}_shape_diff"] for value in self._values],
                    method=method,
                )
                aggregated_metrics[method] = MetricResult(
                    values={
                        self._NAME: aggregates,
                        f"{self._NAME}_energy_diff": energy_aggregates,
                        f"{self._NAME}_shape_diff": shape_aggregates,
                    }
                )
        return aggregated_metrics

    def type(self) -> MetricType:
        """Return the type of the metric."""
        return MetricType.NTD

    def metadata(self) -> dict[str, Any]:
        """Return the metadata for the metric."""
        return {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "feature_dim": self.feature_extractor.feature_dim,
            "n_neighbors": self.n_neighbors,
            "symmetrize": self.symmetrize,
        }

    def reset(self) -> None:
        """Reset the metric state."""
        pass
