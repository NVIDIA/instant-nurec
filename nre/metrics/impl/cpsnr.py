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
from nre.metrics.utils import AggregationMethod, aggregate_tensors, aggregate_weighted_mean


class CPSNRMetric(BaseMetric):
    """Categorical Peak Signal-to-Noise Ratio metric for images."""

    def __init__(
        self,
        data_range: float = 1.0,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.WEIGHTED_MEAN,
    ):
        super().__init__(device, aggregation_methods)
        self.data_range = data_range
        # Finite upper bound for PSNR when images are identical (MSE=0 would give infinity)
        self.max_psnr = 10 * math.log10((self.data_range**2) / (1e-10))
        # Initialize PSNR metric with no reduction to get per-pixel values
        self._psnr_metric = PeakSignalNoiseRatio(data_range=self.data_range, reduction="none", dim=2)
        if device:
            self._psnr_metric.to(device)

    def validate_inputs(
        self,
        eval_frame: torch.Tensor,
        gt_frame: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        segmentation_frame: torch.Tensor | None = None,
        color_dict: dict[str, tuple[int, int, int]] | dict[str, int] | None = None,
        include_categories: set[str] | None = None,
        overall_psnr: bool = False,
    ) -> None:
        """Validate that inputs are valid image tensors."""
        # Validate input types
        for i, img in enumerate([eval_frame, gt_frame]):
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Input {i} must be a torch.Tensor, got {type(img)}")
            if img.dim() < 2:
                raise ValueError(f"Input {i} must have at least 2 dimensions")

        # Validate shapes match
        if eval_frame.shape != gt_frame.shape:
            raise ValueError(f"Evaluated and ground truth shapes must match: {eval_frame.shape} vs {gt_frame.shape}")

        # Validate mask if provided
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError(f"Valid mask must be a torch.Tensor, got {type(valid_mask)}")
            if valid_mask.dtype != torch.bool:
                raise ValueError(f"Valid mask must be boolean tensor, got {valid_mask.dtype}")
            if valid_mask.shape != eval_frame.shape[:2]:
                raise ValueError(
                    f"Valid mask shape {valid_mask.shape} must match image spatial dimensions {eval_frame.shape[:2]}"
                )

    def _compute(
        self,
        eval_frame: torch.FloatTensor,
        gt_frame: torch.FloatTensor,
        valid_mask: torch.Tensor | None = None,
        segmentation_frame: torch.Tensor | None = None,
        color_dict: dict[str, tuple[int, int, int]] | dict[str, int] | None = None,
        include_categories: set[str] | None = None,
        overall_psnr: bool = False,
    ) -> MetricResult:
        """
        Computes the categorical Peak Signal-to-Noise Ratio (CPSNR) between an evaluated frame and a ground truth frame.

        Args:
            eval_frame: The evaluated frame tensor (H x W x C).
            gt_frame: The ground truth frame tensor (H x W x C).
            valid_mask: A mask tensor specifying valid regions for PSNR computation (H x W).
            segmentation_frame: Segmentation mask frame tensor (H x W x 3) with pixel colors representing classes, or (H x W) with class ids.
            color_dict: A dictionary mapping class names to (R, G, B) color tuples or class ids.
            include_categories: Set of categories to include. If it's not provided, all categories are included.
            overall_psnr: Whether to compute overall PSNR. Defaults to False.

        Returns:
            MetricResult: Contains:
                - values: Dictionary with PSNR results per category and overall
                - metadata: Contains psnr_map_tensor, valid_mask, and other metadata
        """
        if include_categories is None and color_dict is not None:
            include_categories = set(color_dict.keys())

        if include_categories is not None and valid_mask is None:
            valid_mask = torch.ones((gt_frame.shape[0], gt_frame.shape[1]), dtype=torch.bool, device=gt_frame.device)

        h, w = eval_frame.shape[:2]

        with torch.no_grad():
            psnr_map_tensor = self._psnr_metric(eval_frame, gt_frame).reshape(h, w).clip(max=self.max_psnr)

        # generate psnr results per category
        psnr_results = {}
        pixel_counts = {}
        if segmentation_frame is not None and color_dict is not None:
            for class_name, color in color_dict.items():
                if isinstance(color, tuple):
                    color_tensor = torch.tensor(color, dtype=segmentation_frame.dtype, device=segmentation_frame.device)
                    class_mask = torch.all(segmentation_frame == color_tensor, dim=2)
                else:
                    class_mask = segmentation_frame == color

                if valid_mask is not None:
                    class_mask = torch.logical_and(class_mask, valid_mask)

                # exclude non-included categories from valid pixels
                if include_categories is not None and class_name not in include_categories:
                    if valid_mask is not None:
                        valid_mask = torch.logical_and(valid_mask, torch.logical_not(class_mask))
                    continue

                pixel_count = torch.sum(class_mask)

                # Only include classes that have at least one pixel
                if pixel_count > 0:
                    psnr_results[class_name] = torch.mean(psnr_map_tensor[class_mask])
                    pixel_counts[class_name] = pixel_count

        # compute overall psnr results
        if valid_mask is not None:
            valid_psnr_values = psnr_map_tensor[valid_mask]
            pixel_count = torch.sum(valid_mask)
        else:
            valid_psnr_values = psnr_map_tensor
            pixel_count = torch.tensor(gt_frame.shape[0] * gt_frame.shape[1], device=gt_frame.device)
            # Create a default mask (all pixels valid) for metadata
            valid_mask = torch.ones((h, w), dtype=torch.bool, device=eval_frame.device)

        if overall_psnr:
            # Only include overall PSNR if there are valid pixels
            if pixel_count > 0:
                psnr_results["overall"] = torch.mean(valid_psnr_values)
                pixel_counts["overall"] = pixel_count

        # Prepare metadata
        metadata = {
            "psnr_map_tensor": psnr_map_tensor,
            "valid_mask": valid_mask,
            "pixel_counts": pixel_counts,
            "categories_computed": list(psnr_results.keys()),
        }

        return MetricResult(values=psnr_results, metadata=metadata)

    def aggregate(self) -> dict[AggregationMethod, MetricResult]:
        if len(self._values) == 0:
            return {}

        # Find all categories that appear in any of the stored results
        all_categories: set[str] = set()
        for value in self._values:
            all_categories.update(value.values.keys())

        aggregated_metrics: dict[AggregationMethod, MetricResult] = {}

        for method in self._aggregation_methods:
            aggregated_psnr: dict[str, torch.Tensor] = {}  # category name -> aggregated psnr
            aggregated_pixel_counts: dict[str, torch.Tensor] = {}  # category name -> aggregated pixel counts
            for category in all_categories:
                # Collect values for this category from all frames that have it
                psnr_values = []
                pixel_count_values = []

                for value in self._values:
                    if category in value.values:
                        psnr_value = value.values[category]
                        pixel_count = value.metadata["pixel_counts"][category].float()
                        # Only include values that are finite and have positive pixel count
                        if torch.isfinite(psnr_value) and pixel_count > 0:
                            psnr_values.append(psnr_value)
                            pixel_count_values.append(pixel_count)

                if len(psnr_values) > 0:
                    # Check if this category has any valid data across all frames
                    total_pixels = sum(pixel_count_values)
                    if total_pixels > 0:
                        weights = pixel_count_values if method == AggregationMethod.WEIGHTED_MEAN else None
                        pixel_count_method = (
                            AggregationMethod.SUM if method == AggregationMethod.WEIGHTED_MEAN else method
                        )
                        # Use weighted mean for PSNR values, weighted by pixel counts if method is WEIGHTED_MEAN
                        aggregated_psnr[category] = aggregate_tensors(psnr_values, weights=weights, method=method)
                        aggregated_pixel_counts[category] = aggregate_tensors(
                            pixel_count_values, method=pixel_count_method
                        )

            metadata = {
                "aggregated_categories": list(all_categories),
                "num_results": len(self._values),
                "pixel_counts": aggregated_pixel_counts,
            }

            aggregated_metrics[method] = MetricResult(values=aggregated_psnr, metadata=metadata)

        return aggregated_metrics

    def type(self) -> MetricType:
        """Return the type of the metric."""
        return MetricType.CPSNR

    def metadata(self) -> dict[str, Any]:
        """Return the metadata for the metric."""
        return {"data_range": self.data_range}

    def reset(self) -> None:
        """Reset the CPSNR criterion."""
        self._psnr_metric.reset()

    def to(self, device: DeviceLikeType) -> CPSNRMetric:
        """Move metric to specified device."""
        super().to(device)
        self._psnr_metric.to(device)
        return self
