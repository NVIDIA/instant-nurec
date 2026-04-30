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

from typing import Any, Literal

import torch

from torch._prims_common import DeviceLikeType
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


class LPIPSMetric(BaseMetric):
    """Learned Perceptual Image Patch Similarity (LPIPS) metric to compare two images.

    LPIPS quantifies perceptual similarity between images using deep network features.
    Output range: [0, +∞). Lower values indicate higher perceptual similarity, with 0.0 meaning
    perceptually identical images. The maximum achievable LPIPS value is data-dependent and
    varies by network backbone ("alex", "vgg", or "squeeze").

    Input requirements:
        - Shape: [N, 3, H, W] or [3, H, W] (3-channel RGB images)
        - Dtype: float16, float32, or float64
        - Range: [0, 1] when normalize=True, or [-1, 1] when normalize=False
        - Minimum H, W: Depends on the backbone (net_type). Typically >=64 for safety.

    Note:
        Unlike PSNR and SSIM, LPIPS does not support an arbitrary `data_range` parameter.
        Use `normalize=True` for [0, 1] inputs or `normalize=False` for [-1, 1] inputs.
        For other ranges (e.g., [0, 255]), normalize your inputs before calling this metric.

    Output interpretation:
        - 0.0: Images are perceptually identical according to the chosen backbone.
        - Higher values: Increasing perceptual dissimilarity.
    """

    _NAME = MetricType.LPIPS.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
        net_type: Literal["alex", "vgg", "squeeze"] = "alex",
        normalize: bool = True,
        **kwargs,
    ):
        """Initialize the LPIPS metric.

        Args:
            device: The device to run the metric on.
            aggregation_methods: The aggregation methods to use.
            net_type: The network backbone for feature extraction. One of "alex", "vgg", or "squeeze".
                The backbone determines the minimum required H, W dimensions for input images.
            normalize: Whether to normalize the input images from [0, 1] to [-1, 1].
                If False, inputs are expected to be in [-1, 1] range.
            **kwargs: Additional arguments passed to LearnedPerceptualImagePatchSimilarity.
        """
        super().__init__(device, aggregation_methods)
        if AggregationMethod.WEIGHTED_MEAN in self._aggregation_methods:
            raise ValueError("Weighted mean is not supported for LPIPS metric.")

        self.net_type = net_type
        self.normalize = normalize
        self._lpips_criterion = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=normalize, **kwargs)
        if device:
            self.to(device)

    def validate_inputs(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Validate that inputs are valid image tensors.

        Args:
            pred: The predicted image tensor of shape [N, 3, H, W] or [3, H, W].
            target: The target image tensor of shape [N, 3, H, W] or [3, H, W].
            mask: Optional boolean mask tensor of shape [H, W].
                True indicates valid pixels to include in computation.
                False indicates pixels to exclude (set to neutral value).
        """
        # Validate input types
        for i, img in enumerate([pred, target]):
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Input {i} must be a torch.Tensor, got {type(img)}")
            if img.dim() not in (3, 4):
                raise ValueError(f"Input {i} must be 3D (C, H, W) or 4D (N, C, H, W), but got {img.dim()}D")
            # LPIPS requires exactly 3 channels (index -3 is channel dim for both 3D and 4D)
            if img.shape[-3] != 3:
                raise ValueError(f"Input {i} must have exactly 3 channels for LPIPS, got {img.shape[-3]}")

        # Validate shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Predicted and target shapes must match: {pred.shape} vs {target.shape}")

        # Validate dtype - LPIPS requires floating-point tensors
        valid_dtypes = (torch.float16, torch.float32, torch.float64)
        for i, img in enumerate([pred, target]):
            if img.dtype not in valid_dtypes:
                raise TypeError(
                    f"Input {i} must be a floating-point tensor (float16, float32, or float64), "
                    f"got {img.dtype}. If using uint8 images, convert with: tensor.float() / 255.0"
                )

        # Validate input range based on normalize setting
        min_value = 0.0 if self.normalize else -1.0
        for i, img in enumerate([pred, target]):
            if img.min() < min_value or img.max() > 1.0:
                raise ValueError(
                    f"Input {i} must be in [{min_value}, 1] range when normalize={self.normalize}, "
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
        """Compute LPIPS between predicted and target images.

        Args:
            pred: The predicted image tensor of shape [N, 3, H, W] or [3, H, W].
                Must be a floating-point tensor (float16, float32, or float64).
                Expected range is [0, 1] when normalize=True, or [-1, 1] when normalize=False.
                Minimum H, W depends on the backbone (net_type); typically >=64 for safety.
            target: The target image tensor of shape [N, 3, H, W] or [3, H, W].
                Must be a floating-point tensor (float16, float32, or float64).
                Expected range is [0, 1] when normalize=True, or [-1, 1] when normalize=False.
                Minimum H, W depends on the backbone (net_type); typically >=64 for safety.
            mask: Optional boolean mask tensor of shape [H, W]. Defaults to None.
                True indicates valid pixels to include in computation.
                False indicates pixels to exclude (set to neutral value).

        Note:
            Masked regions are set to 0 in both pred and target images. This may
            bias LPIPS downward (indicating higher similarity) when masks cover
            large areas, as those regions become identical in both images.

        Returns:
            MetricResult: The LPIPS value and metadata.
        """
        # Store original shapes for metadata
        original_shape = pred.shape

        # Prevent accumulation in the underlying torchmetrics metric so each
        # _compute() call is independent.
        self._lpips_criterion.reset()

        # Ensure batch dimension exists (LPIPS requires [N, C, H, W])
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        if mask is not None:
            # Move mask to the same device as input tensors
            mask = mask.to(pred.device)
            # For LPIPS, we apply the mask by setting masked regions to a neutral value
            # Expand mask to [N, C, H, W] shape
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
            # Clone tensors to avoid modifying originals
            pred = pred.clone()
            target = target.clone()
            # Set masked regions to 0 (neutral value)
            pred[~mask_expanded] = 0.0
            target[~mask_expanded] = 0.0

        lpips_value = self._lpips_criterion(pred, target)

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

        return MetricResult(values={self._NAME: lpips_value}, metadata=metadata)

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
        return MetricType.LPIPS

    def metadata(self) -> dict[str, Any]:
        """Return the metadata for the metric."""
        return {"net_type": self.net_type, "normalize": self.normalize}

    def reset(self) -> None:
        """Reset LPIPS metric state and underlying criterion."""
        self.clear()
        self._lpips_criterion.reset()

    def to(self, device: DeviceLikeType) -> LPIPSMetric:
        """Move the metric to the specified device.

        Args:
            device: The device to move the metric to.

        Returns:
            LPIPSMetric: The metric instance with the device set.
        """
        super().to(device)
        self._lpips_criterion.to(device)
        return self
