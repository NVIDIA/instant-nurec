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

import logging

from typing import Any

import numpy as np
import scipy.linalg as scipy_linalg
import torch

from torch._prims_common import DeviceLikeType

from nre.metrics.impl.utils.feature_extractor import FeatureExtractorFactory
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


logger = logging.getLogger(__name__)


class FIDMetric(BaseMetric):
    """Generic FID metric for images using configurable feature extractors."""

    _NAME = MetricType.FID.name.lower()

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
            raise ValueError("Weighted mean is not supported for FID metric.")

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

    def _compute_fid(self, features_pred: np.ndarray, features_target: np.ndarray) -> float:
        """Compute Frechet Inception Distance (FID) under Gaussian assumption.

        Args:
            features_pred: Predicted features of shape (N, D).
            features_target: Target features of shape (M, D).

        Returns:
            FID score as float (non-negative).
        """
        # Check for sufficient samples
        if features_pred.shape[0] < 2 or features_target.shape[0] < 2:
            logger.warning("Insufficient samples for FID computation")
            return 0.0

        # Compute means
        mu_pred = np.mean(features_pred, axis=0)
        mu_target = np.mean(features_target, axis=0)

        # Compute covariance matrices with proactive regularization
        eps = 1e-6
        sigma_pred = np.cov(features_pred, rowvar=False) + eps * np.eye(features_pred.shape[1])
        sigma_target = np.cov(features_target, rowvar=False) + eps * np.eye(features_target.shape[1])

        # Check for NaN values
        if np.any(np.isnan(mu_pred)) or np.any(np.isnan(mu_target)):
            logger.warning("NaN in feature means")
            return 0.0

        if np.any(np.isnan(sigma_pred)) or np.any(np.isnan(sigma_target)):
            logger.warning("NaN in covariance matrices")
            return 0.0

        # Compute mean difference
        diff = mu_pred - mu_target
        diff_sq = diff.dot(diff)

        # Compute matrix square root with error handling
        try:
            product = sigma_pred.dot(sigma_target)
            covmean, info = scipy_linalg.sqrtm(product, disp=False)

            if info > 0:
                logger.warning(f"sqrtm convergence issue, info={info}")

            # Handle complex numbers
            if np.iscomplexobj(covmean):
                if not np.allclose(covmean.imag, 0, atol=1e-3):
                    max_imag = np.max(np.abs(covmean.imag))
                    logger.warning(f"Large imaginary component in sqrtm: {max_imag:.6f}")
                covmean = covmean.real

            # Compute FID
            fid = diff_sq + np.trace(sigma_pred) + np.trace(sigma_target) - 2 * np.trace(covmean)

            # Ensure non-negative (handle numerical precision issues)
            if fid < 0:
                if abs(fid) < 1e-6:
                    logger.warning(f"Small negative FID ({fid:.8f}) due to numerical precision, setting to 0")
                    fid = 0.0
                else:
                    logger.error(f"Large negative FID ({fid:.6f}) - computation issue")
                    return 0.0

            # Clamp FID to reasonable maximum value
            MAX_FID_VALUE = 10000.0
            if fid > MAX_FID_VALUE:
                logger.warning(f"FID value ({fid:.2f}) exceeds maximum, clamping to {MAX_FID_VALUE}")
                fid = MAX_FID_VALUE

            return float(fid)

        except (ValueError, np.linalg.LinAlgError) as e:
            logger.error(f"Error in FID computation: {e}")
            return 0.0

    def _compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> MetricResult:
        """Compute FID between predicted and target images.

        Args:
            pred: The predicted image tensor of shape [..., h, w].
            target: The target image tensor of shape [..., h, w].

        Returns:
            MetricResult: The FID value and metadata.
        """
        # Store original shapes for metadata
        original_shape = pred.shape

        # Ensure we have batch dimension
        if pred.dim() == 3:  # (C, H, W)
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # Preserve original device
        original_device = pred.device

        # Extract features with optional batching
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

        # Compute FID
        fid_value = self._compute_fid(features_pred, features_target)

        # Create metadata with input information
        metadata = {
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "input_shape": list(original_shape),
            "feature_dim": self.feature_extractor.feature_dim,
        }

        # Convert result back to tensor on original device
        fid_tensor = torch.tensor(fid_value, device=original_device)

        return MetricResult(values={self._NAME: fid_tensor}, metadata=metadata)

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
        return MetricType.FID

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
