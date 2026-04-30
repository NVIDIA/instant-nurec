# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Object-Level Perceptual Quality Metric.

This metric computes perceptual quality metrics between pre-cropped object images.
It accepts pre-cropped images as input and computes quality metrics from provided crops.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torch._prims_common import DeviceLikeType

from nre.metrics.impl.object_level_semantic import ObjectMetadata
from nre.metrics.impl.utils.perceptual_utils import (
    compute_artifact_score,
    compute_blur_penalty,
    compute_cem_score,
    compute_channel_coherence_score,
    compute_chroma_hf_score,
    compute_color_histogram_similarity,
    compute_edge_similarity,
    compute_gradient_similarity,
    compute_hue_variance_score,
    compute_multi_scale_ssim,
    compute_y_chroma_ratio_score,
)
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


# Default weights for perceptual metric computation
DEFAULT_PERCEPTUAL_WEIGHTS = {
    "edge_similarity": 0.25,
    "gradient_similarity": 0.25,
    "blur_score": 0.22,
    "artifact_score": 0.22,
    "ssim": 0.04,
    "color_hist_similarity": 0.02,
}


class ObjectLevelPerceptualMetric(BaseMetric):
    """Object-level perceptual quality metric for pre-cropped images.

    This metric computes perceptual quality metrics between objects using:
    - Edge similarity
    - Gradient similarity
    - Blur score
    - Artifact score
    - SSIM
    - Color histogram similarity
    - CEM score
    - Hue variance score
    - Chroma HF score
    - Channel coherence score
    - Y chroma ratio score

    Usage:
        metric = ObjectLevelPerceptualMetric(device="cuda")
        result = metric(crops_pred, crops_target, track_ids, class_names)
    """

    _NAME = MetricType.OBJECT_LEVEL_PERCEPTUAL.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
    ) -> None:
        """Initialize Object-Level Perceptual Simple metric.

        Args:
            device: Device to run computation on.
            aggregation_methods: Aggregation methods to use.
        """
        super().__init__(device, aggregation_methods)

        self.weights = DEFAULT_PERCEPTUAL_WEIGHTS

        if device:
            self.to(device)

    def validate_inputs(
        self,
        pred: torch.Tensor | np.ndarray | list,
        target: torch.Tensor | np.ndarray | list,
        track_ids: list[str] | None = None,
        class_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate inputs.

        Args:
            pred: Predicted crops [N, H, W, C], single crop [H, W, C], or list of crops.
            target: Target crops [N, H, W, C], single crop [H, W, C], or list of crops.
            track_ids: Optional track IDs for each object.
            class_names: Optional class names for each object.
        """
        # List mode validation
        if isinstance(pred, list) and isinstance(target, list):
            if len(pred) != len(target):
                raise ValueError(f"pred and target lists must have same length: {len(pred)} vs {len(target)}")
            if len(pred) == 0:
                raise ValueError("Empty crop lists provided")
            # Validate list elements are valid array types
            for i, (p, t) in enumerate(zip(pred, target)):
                if not isinstance(p, (torch.Tensor, np.ndarray)):
                    raise TypeError(f"pred[{i}] must be Tensor or ndarray, got {type(p)}")
                if not isinstance(t, (torch.Tensor, np.ndarray)):
                    raise TypeError(f"target[{i}] must be Tensor or ndarray, got {type(t)}")
            return

        # Validate input types
        if not isinstance(pred, (torch.Tensor, np.ndarray)):
            raise TypeError(f"pred must be torch.Tensor or np.ndarray, got {type(pred)}")
        if not isinstance(target, (torch.Tensor, np.ndarray)):
            raise TypeError(f"target must be torch.Tensor or np.ndarray, got {type(target)}")

        # Validate pred value range
        if isinstance(pred, torch.Tensor) and pred.dtype in [torch.float32, torch.float64, torch.float16]:
            if pred.min() < 0.0 or pred.max() > 1.0:
                raise ValueError(f"pred values must be in [0, 1], got [{pred.min():.3f}, {pred.max():.3f}]")
        elif isinstance(pred, np.ndarray) and pred.dtype in [np.float32, np.float64, np.float16]:
            if pred.min() < 0.0 or pred.max() > 1.0:
                raise ValueError(f"pred values must be in [0, 1], got [{pred.min():.3f}, {pred.max():.3f}]")

        # Validate target value range
        if isinstance(target, torch.Tensor) and target.dtype in [torch.float32, torch.float64, torch.float16]:
            if target.min() < 0.0 or target.max() > 1.0:
                raise ValueError(f"target values must be in [0, 1], got [{target.min():.3f}, {target.max():.3f}]")
        elif isinstance(target, np.ndarray) and target.dtype in [np.float32, np.float64, np.float16]:
            if target.min() < 0.0 or target.max() > 1.0:
                raise ValueError(f"target values must be in [0, 1], got [{target.min():.3f}, {target.max():.3f}]")

        # Validate shapes match
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shapes must match: {pred.shape} vs {target.shape}")

        # Validate dimensions
        if pred.ndim not in [3, 4]:
            raise ValueError(f"pred must be 3D [H,W,C] or 4D [N,H,W,C], got {pred.ndim}D")

        # Validate batch consistency
        batch_size = pred.shape[0] if pred.ndim == 4 else 1
        if track_ids is not None and len(track_ids) != batch_size:
            raise ValueError(f"track_ids length {len(track_ids)} must match batch size {batch_size}")
        if class_names is not None and len(class_names) != batch_size:
            raise ValueError(f"class_names length {len(class_names)} must match batch size {batch_size}")

    def _compute_list(
        self,
        pred_crops: list,
        target_crops: list,
        obj_metadata: ObjectMetadata | None = None,
    ) -> MetricResult:
        """Compute perceptual metrics for list of variable-sized crops.

        Args:
            pred_crops: List of predicted crops (can be different sizes).
            target_crops: List of target crops (can be different sizes).
            obj_metadata: ObjectMetadata containing track/frame/bbox info.

        Returns:
            MetricResult with perceptual scores and metadata.
        """
        batch_size = len(pred_crops)
        all_metrics: dict[str, list[float]] = {
            "edge_similarity": [],
            "gradient_similarity": [],
            "blur_score": [],
            "artifact_score": [],
            "ssim": [],
            "color_hist_similarity": [],
            "cem_score": [],
            "hue_variance_score": [],
            "chroma_hf_score": [],
            "channel_coherence_score": [],
            "y_chroma_ratio_score": [],
            "perceptual_score": [],
        }

        # Process each crop individually
        for i in range(batch_size):
            pred_crop = pred_crops[i]
            target_crop = target_crops[i]

            # Convert to numpy if needed
            if isinstance(pred_crop, torch.Tensor):
                pred_crop = pred_crop.permute(1, 2, 0).cpu().numpy()
                pred_crop = (pred_crop * 255).astype(np.uint8)
            if isinstance(target_crop, torch.Tensor):
                target_crop = target_crop.permute(1, 2, 0).cpu().numpy()
                target_crop = (target_crop * 255).astype(np.uint8)

            metrics = self._compute_single_perceptual_metrics(pred_crop, target_crop)
            for key, value in metrics.items():
                all_metrics[key].append(value)

        perceptual_scores = torch.tensor(all_metrics["perceptual_score"], dtype=torch.float32)

        # Build detailed metadata using helper method
        result_metadata = self._build_detailed_metadata(batch_size, all_metrics, obj_metadata)

        # Move to device
        perceptual_scores = perceptual_scores.to(self.device)

        return MetricResult(values={self._NAME: perceptual_scores}, metadata=result_metadata)

    def _build_detailed_metadata(
        self,
        batch_size: int,
        all_metrics: dict[str, list],
        obj_metadata: ObjectMetadata | None,
    ) -> dict[str, Any]:
        """Build detailed per-object data for aggregation and visualization.

        This creates a list of per-object dictionaries containing all metric
        values, track IDs, class names, frame indices, and bounding boxes.
        This data is later used by aggregate() to compute per-track and
        per-class statistics, and by visualization to display crops and metrics.

        Args:
            batch_size: Number of objects in batch.
            all_metrics: Dictionary of metric lists.
            obj_metadata: ObjectMetadata containing track/frame/bbox info.

        Returns:
            Metadata dictionary containing only 'detailed_data' with per-object info.
        """
        detailed_data: list[dict[str, Any]] = []

        # Extract values from obj_metadata (with defaults)
        meta = obj_metadata or ObjectMetadata()
        track_ids = meta.track_ids
        class_names = meta.class_names
        frame_idx = meta.frame_idx
        gt_frame_idx = meta.gt_frame_idx
        bboxes_gt = meta.bboxes_gt
        bboxes_rendered = meta.bboxes_rendered
        rendered_timestamp = meta.rendered_timestamp

        for i in range(batch_size):
            obj_data: dict[str, Any] = {
                "track_id": track_ids[i] if track_ids and i < len(track_ids) else None,
                "class_name": class_names[i] if class_names and i < len(class_names) else None,
                "frame_idx": frame_idx,
                "gt_frame_idx": gt_frame_idx,
                "rendered_timestamp": rendered_timestamp,
            }

            # Add bbox data if provided (used by visualization to extract crops)
            if bboxes_gt and i < len(bboxes_gt):
                bbox = bboxes_gt[i]
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    obj_data["bbox_gt"] = {
                        "x": int(bbox[0]),
                        "y": int(bbox[1]),
                        "width": int(bbox[2]),
                        "height": int(bbox[3]),
                    }
                else:
                    obj_data["bbox_gt"] = bbox

            if bboxes_rendered and i < len(bboxes_rendered):
                bbox = bboxes_rendered[i]
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    obj_data["bbox_rendered"] = {
                        "x": int(bbox[0]),
                        "y": int(bbox[1]),
                        "width": int(bbox[2]),
                        "height": int(bbox[3]),
                    }
                else:
                    obj_data["bbox_rendered"] = bbox

            # Add all individual metric scores
            for metric_key, values_list in all_metrics.items():
                if i < len(values_list):
                    obj_data[metric_key] = values_list[i]

            detailed_data.append(obj_data)

        return {
            "detailed_data": detailed_data,
        }

    def _compute_single_perceptual_metrics(
        self,
        crop_pred: np.ndarray,
        crop_target: np.ndarray,
    ) -> dict[str, float]:
        """Compute all perceptual metrics for a single pair of crops.

        Args:
            crop_pred: Predicted crop [H, W, C] (numpy uint8 or float).
            crop_target: Target crop [H, W, C] (numpy uint8 or float).

        Returns:
            Dictionary with all perceptual metrics.
        """
        # Ensure uint8 format
        if crop_pred.dtype in [np.float32, np.float64]:
            crop_pred = (crop_pred * 255).astype(np.uint8)
        if crop_target.dtype in [np.float32, np.float64]:
            crop_target = (crop_target * 255).astype(np.uint8)

        # Compute all individual metrics using corrected utility functions
        edge_similarity = compute_edge_similarity(crop_target, crop_pred)
        gradient_similarity = compute_gradient_similarity(crop_target, crop_pred)
        blur_score = compute_blur_penalty(crop_target, crop_pred)
        artifact_score = compute_artifact_score(crop_target, crop_pred)
        ssim = compute_multi_scale_ssim(crop_target, crop_pred)
        color_hist_similarity = compute_color_histogram_similarity(crop_target, crop_pred)
        cem_score = compute_cem_score(crop_target, crop_pred)
        hue_variance_score = compute_hue_variance_score(crop_target, crop_pred)
        chroma_hf_score = compute_chroma_hf_score(crop_target, crop_pred)

        # Channel coherence (despite the function name, it returns coherence, 1 = good)
        channel_coherence_score = compute_channel_coherence_score(crop_target, crop_pred)

        y_chroma_ratio_score = compute_y_chroma_ratio_score(crop_target, crop_pred)

        # Compute combined perceptual score
        perceptual_score = (
            self.weights["edge_similarity"] * edge_similarity
            + self.weights["gradient_similarity"] * gradient_similarity
            + self.weights["blur_score"] * blur_score
            + self.weights["artifact_score"] * artifact_score
            + self.weights["ssim"] * ssim
            + self.weights["color_hist_similarity"] * color_hist_similarity
        )

        return {
            "edge_similarity": edge_similarity,
            "gradient_similarity": gradient_similarity,
            "blur_score": blur_score,
            "artifact_score": artifact_score,
            "ssim": ssim,
            "color_hist_similarity": color_hist_similarity,
            "cem_score": cem_score,
            "hue_variance_score": hue_variance_score,
            "chroma_hf_score": chroma_hf_score,
            "channel_coherence_score": channel_coherence_score,
            "y_chroma_ratio_score": y_chroma_ratio_score,
            "perceptual_score": perceptual_score,
        }

    def _compute(
        self,
        pred: torch.Tensor | np.ndarray | list,
        target: torch.Tensor | np.ndarray | list,
        obj_metadata: ObjectMetadata | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute perceptual metrics between cropped objects.

        Args:
            pred: Predicted crops [N, H, W, C], single crop [H, W, C], or list of crops.
            target: Target crops [N, H, W, C], single crop [H, W, C], or list of crops.
            obj_metadata: ObjectMetadata containing track/frame/bbox info.
            **kwargs: Additional arguments (ignored).

        Returns:
            MetricResult with perceptual scores and metadata.
        """
        # Handle list mode (variable-sized crops)
        if isinstance(pred, list) and isinstance(target, list):
            return self._compute_list(pred, target, obj_metadata)

        # Convert to numpy if needed
        if isinstance(pred, torch.Tensor):
            # Convert from [C, H, W] or [N, C, H, W] to [H, W, C] or [N, H, W, C]
            if pred.ndim == 3:  # [C, H, W]
                pred = pred.permute(1, 2, 0).cpu().numpy()
            else:  # [N, C, H, W]
                pred = pred.permute(0, 2, 3, 1).cpu().numpy()
            # Scale from [0, 1] to [0, 255] uint8
            pred = (pred * 255).astype(np.uint8)
        if isinstance(target, torch.Tensor):
            # Convert from [C, H, W] or [N, C, H, W] to [H, W, C] or [N, H, W, C]
            if target.ndim == 3:  # [C, H, W]
                target = target.permute(1, 2, 0).cpu().numpy()
            else:  # [N, C, H, W]
                target = target.permute(0, 2, 3, 1).cpu().numpy()
            # Scale from [0, 1] to [0, 255] uint8
            target = (target * 255).astype(np.uint8)

        # Type narrowing for mypy: after conversion, pred and target are numpy arrays
        assert isinstance(pred, np.ndarray), "pred should be np.ndarray after conversion"
        assert isinstance(target, np.ndarray), "target should be np.ndarray after conversion"

        # Handle single crop or batch
        if pred.ndim == 3:
            # Single crop
            metrics = self._compute_single_perceptual_metrics(pred, target)
            perceptual_scores = torch.tensor([metrics["perceptual_score"]], dtype=torch.float32)
            all_metrics = {k: [v] for k, v in metrics.items()}
            batch_size = 1
        else:
            # Batch of crops
            batch_size = pred.shape[0]
            all_metrics = {
                "edge_similarity": [],
                "gradient_similarity": [],
                "blur_score": [],
                "artifact_score": [],
                "ssim": [],
                "color_hist_similarity": [],
                "cem_score": [],
                "hue_variance_score": [],
                "chroma_hf_score": [],
                "channel_coherence_score": [],
                "y_chroma_ratio_score": [],
                "perceptual_score": [],
            }

            for i in range(batch_size):
                metrics = self._compute_single_perceptual_metrics(pred[i], target[i])
                for key, value in metrics.items():
                    all_metrics[key].append(value)

            perceptual_scores = torch.tensor(all_metrics["perceptual_score"], dtype=torch.float32)

        # Build detailed metadata using helper method
        result_metadata = self._build_detailed_metadata(batch_size, all_metrics, obj_metadata)

        # Move to device
        perceptual_scores = perceptual_scores.to(self.device)

        # Return as MetricResult (values must be a dict with metric name as key)
        return MetricResult(values={self._NAME: perceptual_scores}, metadata=result_metadata)

    def aggregate(self) -> dict[AggregationMethod, MetricResult]:
        """Aggregate stored values using the specified method."""
        aggregated_metrics: dict[AggregationMethod, MetricResult] = {}

        if len(self._values) == 0:
            return aggregated_metrics

        # Collect ALL detailed data across all frames
        all_detailed_data = []
        for value in self._values:
            if "detailed_data" in value.metadata:
                all_detailed_data.extend(value.metadata["detailed_data"])

        # Organize by track_id for per-track aggregation
        detailed_by_track: dict[str, list[dict]] = {}
        for obj_data in all_detailed_data:
            track_id = obj_data.get("track_id")
            if track_id:
                if track_id not in detailed_by_track:
                    detailed_by_track[track_id] = []
                detailed_by_track[track_id].append(obj_data)

        # Organize by class_name for per-class aggregation
        detailed_by_class: dict[str, list[dict]] = {}
        for obj_data in all_detailed_data:
            class_name = obj_data.get("class_name")
            if class_name:
                if class_name not in detailed_by_class:
                    detailed_by_class[class_name] = []
                detailed_by_class[class_name].append(obj_data)

        # Compute per-track statistics
        per_track_result = {}
        for track_id, track_frames in detailed_by_track.items():
            track_metrics: dict[str, Any] = {}
            track_metrics["num_frames"] = int(len(track_frames))
            first_class = track_frames[0].get("class_name", "unknown") if track_frames else "unknown"
            if first_class != "unknown":
                track_metrics["class_name"] = str(first_class)

            sub_metrics = self._aggregate_sub_metrics(track_frames)
            track_metrics.update(sub_metrics)

            per_track_result[track_id] = track_metrics

        # Compute per-class statistics
        per_class_result = {}
        for class_name, class_data in detailed_by_class.items():
            class_metrics: dict[str, Any] = {}
            class_metrics["num_objects"] = int(len(class_data))

            unique_tracks = {obj.get("track_id") for obj in class_data if obj.get("track_id")}
            class_metrics["num_tracks"] = int(len(unique_tracks))

            sub_metrics = self._aggregate_sub_metrics(class_data)
            class_metrics.update(sub_metrics)

            per_class_result[class_name] = class_metrics

        # Flatten all scores to individual [1] tensors for aggregate_tensors compatibility
        all_values = [value.values[self._NAME] for value in self._values]
        flat_tensors = [score.unsqueeze(0) for value in all_values for score in value]

        for method in self._aggregation_methods:
            if method in [AggregationMethod.MEAN, AggregationMethod.SUM, AggregationMethod.MIN, AggregationMethod.MAX]:
                aggregate_value = aggregate_tensors(flat_tensors, method=method).squeeze()

                # For MEAN, include per-track and per-class data in metadata
                metadata: dict[str, Any] = {}
                if method == AggregationMethod.MEAN:
                    metadata["per_track"] = per_track_result
                    metadata["per_class"] = per_class_result

                aggregated_metrics[method] = MetricResult(values={self._NAME: aggregate_value}, metadata=metadata)

        return aggregated_metrics

    def _aggregate_sub_metrics(self, data_list: list[dict]) -> dict[str, float]:
        """Helper to aggregate all sub-metrics for a group of objects.

        Args:
            data_list: List of per-object dictionaries with metric values.

        Returns:
            Dictionary with aggregated statistics (mean, std, min, max) for each metric.
            Ordered: perceptual_score first, then sub-metrics alphabetically.
        """
        if not data_list:
            return {}

        aggregated: dict[str, float] = {}

        # All metrics
        metric_keys = [
            "perceptual_score",
            "artifact_score",
            "blur_score",
            "cem_score",
            "channel_coherence_score",
            "chroma_hf_score",
            "color_hist_similarity",
            "edge_similarity",
            "gradient_similarity",
            "hue_variance_score",
            "ssim",
            "y_chroma_ratio_score",
        ]

        for key in metric_keys:
            values = [obj[key] for obj in data_list if key in obj and isinstance(obj[key], (int, float))]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))

        return aggregated

    def type(self) -> MetricType:
        """Return the type of the metric."""
        return MetricType.OBJECT_LEVEL_PERCEPTUAL

    def metadata(self) -> dict[str, Any]:
        """Return configuration metadata for the metric.

        Returns:
            Dictionary containing metric configuration including weights
            for each perceptual sub-metric.
        """
        return {
            "metric_name": self._NAME,
            "weights": self.weights,
            "sub_metrics": list(self.weights.keys()),
        }

    def reset(self) -> None:
        """Reset the metric state."""
        pass
