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

from torch._prims_common import DeviceLikeType
from torchmetrics.image import StructuralSimilarityIndexMeasure

from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


class SSIMMetric(BaseMetric):
    """Structural Similarity Index Measure (SSIM) metric to compare two images.

    SSIM quantifies the perceptual similarity between two images, with values ranging from -1 to 1.
    - A value of 1.0 indicates perfect structural similarity (identical images).
    - A value of 0 indicates no structural correlation.
    - Values lower than 0 indicate anti-correlation in structure.

    Higher SSIM values indicate greater perceptual similarity between images, making it a commonly
    used metric for evaluating image quality and fidelity after transformation (e.g., compression,
    restoration, generation).

    Input requirements:
        - Shape: [B, C, H, W] or [C, H, W] (batch of images or single image)
        - Dtype: float16, float32, or float64
        - Range: [0, data_range] (default [0, 1.0])
    """

    _NAME = MetricType.SSIM.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
        data_range: float = 1.0,
        kernel_size: int = 11,
        **kwargs,
    ):
        """Initialize the SSIM metric.

        Args:
            device: The device to run the metric on.
            aggregation_methods: The aggregation methods to use for combining multiple results.
            data_range: The data range of the input images (default 1.0 for [0, 1] normalized images).
            kernel_size: The size of the Gaussian kernel used for SSIM computation.
            **kwargs: Additional arguments passed to torchmetrics.StructuralSimilarityIndexMeasure.
                Supported kwargs include: sigma (float), k1 (float), k2 (float), reduction (str).
        """
        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for SSIM metric.")
        self.data_range = data_range
        self.kernel_size = kernel_size
        self._ssim_criterion = StructuralSimilarityIndexMeasure(
            data_range=self.data_range, kernel_size=self.kernel_size, **kwargs
        )
        if device:
            self.to(device)

    def validate_inputs(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Validate that inputs are valid image tensors.

        Args:
            pred: The predicted image tensor of shape [B, C, H, W] or [C, H, W].
            target: The target image tensor of shape [B, C, H, W] or [C, H, W].
            mask: Optional boolean mask tensor of shape [H, W].
        """
        # Validate input types and dimensions
        for i, img in enumerate([pred, target]):
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Input {i} must be a torch.Tensor, got {type(img)}")
            if img.dim() not in (3, 4):
                raise ValueError(f"Input {i} must be 3D (C, H, W) or 4D (B, C, H, W), but got {img.dim()}D")

        # Validate shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Predicted and target shapes must match: {pred.shape} vs {target.shape}")

        # Validate dtype - SSIM requires floating-point tensors
        valid_dtypes = (torch.float16, torch.float32, torch.float64)
        for i, img in enumerate([pred, target]):
            if img.dtype not in valid_dtypes:
                raise TypeError(
                    f"Input {i} must be a floating-point tensor (float16, float32, or float64), "
                    f"got {img.dtype}. If using uint8 images, convert with: tensor.float() / 255.0"
                )

        # Validate input range
        for i, img in enumerate([pred, target]):
            if img.min() < 0.0 or img.max() > self.data_range:
                raise ValueError(
                    f"Input {i} must be in [0, {self.data_range}] range, "
                    f"got min={img.min().item():.4f}, max={img.max().item():.4f}"
                )

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
        """Compute SSIM between predicted and target images.

        Args:
            pred: The predicted image tensor of shape [C, H, W] or [B, C, H, W].
                Should be a floating-point tensor (float16, float32, or float64).
                Expected range is [0, data_range] (default [0, 1.0]).
            target: The target image tensor of shape [C, H, W] or [B, C, H, W].
                Should be a floating-point tensor (float16, float32, or float64).
                Expected range is [0, data_range] (default [0, 1.0]).
            mask: Optional boolean mask tensor of shape [H, W]. Defaults to None.
                True indicates valid pixels to include in computation.
                False indicates pixels to exclude (set to neutral value).

        Note:
            Masked regions are set to 0 in both pred and target images. This may
            bias SSIM upward when masks cover large areas, as those regions become
            identical in both images.

        Returns:
            MetricResult: The SSIM value and metadata. The SSIM value is a tensor of shape [batch_size]
                if there is a batch dimension, otherwise it is a scalar.
        """
        # Store original shapes for metadata
        original_shape = pred.shape

        # Prevent accumulation in the underlying torchmetrics metric so each
        # _compute() call is independent.
        self._ssim_criterion.reset()

        # Ensure batch dimension exists (torchmetrics SSIM requires [B, C, H, W])
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        if mask is not None:
            # Move mask to the same device as input tensors
            mask = mask.to(pred.device)
            # For SSIM with mask, we apply the mask by setting masked regions to a neutral value
            # Expand mask to [B, C, H, W] shape
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
            # Clone tensors to avoid modifying originals
            pred = pred.clone()
            target = target.clone()
            # Set masked regions to 0 (neutral value)
            pred[~mask_expanded] = 0.0
            target[~mask_expanded] = 0.0

        ssim_value = self._ssim_criterion(pred, target)

        # Calculate masked pixels correctly
        if mask is not None:
            # Count pixels where mask is True (valid pixels)
            num_masked_pixels = torch.sum(mask).item()
        else:
            # If no mask, all pixels are valid
            num_masked_pixels = original_shape[-2] * original_shape[-1]

        # Create metadata by extending base metric metadata with runtime info
        metadata = self.metadata()
        metadata["input_shape"] = list(original_shape)
        metadata["masked_pixels"] = num_masked_pixels

        return MetricResult(values={self._NAME: ssim_value}, metadata=metadata)

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
        return MetricType.SSIM

    def metadata(self) -> dict[str, Any]:
        """Return the metadata for the metric."""
        return {"data_range": self.data_range, "kernel_size": self.kernel_size}

    def reset(self) -> None:
        """Reset SSIM metric state and underlying criterion."""
        self.clear()
        self._ssim_criterion.reset()

    def to(self, device: DeviceLikeType) -> SSIMMetric:
        """Move the metric to the specified device.

        Args:
            device: The device to move the metric to.

        Returns:
            SSIMMetric: The metric instance with the device set.
        """
        super().to(device)
        self._ssim_criterion.to(device)
        return self
