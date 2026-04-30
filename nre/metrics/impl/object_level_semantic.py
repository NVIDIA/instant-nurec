# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Object-level semantic similarity metric using DNN features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from torch._prims_common import DeviceLikeType
from torch.nn.functional import cosine_similarity

from nre.metrics.impl.utils.feature_extractor import BaseFeatureExtractor, FeatureExtractorFactory
from nre.metrics.metric import BaseMetric, MetricResult
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod, aggregate_tensors


@dataclass
class ObjectMetadata:
    track_ids: list[str] | None = None
    class_names: list[str] | None = None
    frame_idx: int | None = None
    gt_frame_idx: int | None = None
    bboxes_gt: list[dict] | None = None
    bboxes_rendered: list[dict] | None = None
    rendered_timestamp: int | None = None


# Semantic score adjustment constants
# SEMANTIC_BASELINE: Empirical floor (cosine sim for unrelated objects).
#   Meaningful range is [0.7, 1.0], only 0.3 units wide.
# SEMANTIC_BELOW_BASELINE_SCALE: Penalizes below-baseline scores to [0, ~0.21],
#   reserving most of [0, 1] for quality discrimination above baseline.
SEMANTIC_BASELINE = 0.7
SEMANTIC_BELOW_BASELINE_SCALE = 0.3


class ObjectLevelSemanticMetric(BaseMetric):
    """Computes semantic similarity between object crops using DNN features.

    Accepts either:
        - Image crops (features are extracted automatically), or
        - Precomputed features of the cropped object image

    Usage:
        # With cropped images
        metric = ObjectLevelSemanticMetric(device="cuda")
        result = metric(crops_pred, crops_target, track_ids, class_names)

        # With pre-extracted features
        metric = ObjectLevelSemanticMetric(device="cuda", precomputed_features_only=True)
        result = metric(feats_pred, feats_target, track_ids, class_names)
    """

    _NAME = MetricType.OBJECT_LEVEL_SEMANTIC.name.lower()

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
        extractor_type: str = "dinov2",
        pretrained_path: str = "facebook/dinov2-base",
        cache_dir: str | None = None,
        feature_batch_size: int | None = 32,
        feature_layers: list[int] | None = None,
        min_size: int = 224,
        precomputed_features_only: bool = False,
    ) -> None:
        """Initialize Object-Level Semantic metric.

        Args:
            device: Device to run computation on.
            aggregation_methods: Aggregation methods to use.
            extractor_type: Type of feature extractor (default: "dinov2").
            pretrained_path: Path to pretrained model.
            cache_dir: Cache directory for models.
            feature_batch_size: Batch size for feature extraction.
            feature_layers: Layer indices to extract features from.
            min_size: Minimum image size for feature extraction (default: 224).
            precomputed_features_only: If True, skip loading feature extractor.
        """
        super().__init__(device, aggregation_methods)

        # Configuration
        self.extractor_type = extractor_type
        self.pretrained_path = pretrained_path
        self.cache_dir = cache_dir
        self.feature_batch_size = feature_batch_size
        self.min_size = min_size
        self.precomputed_features_only = precomputed_features_only

        # Feature extractor (Optional when precomputed_features_only=True)
        self.feature_extractor: Optional[BaseFeatureExtractor]
        if not precomputed_features_only:
            # Use only final layer (12) instead of multiple layers [6, 9, 12]
            # Note: hidden_states[12] = output of final transformer layer (layer 11)
            _feature_layers = feature_layers if feature_layers is not None else [12]
            self.feature_extractor = FeatureExtractorFactory.create_extractor(
                extractor_type=extractor_type,
                pretrained_path=pretrained_path,
                cache_dir=cache_dir,
                device=device,
                feature_layers=_feature_layers,
            )
        else:
            self.feature_extractor = None

        if device:
            self.to(device)

    def validate_inputs(
        self,
        pred: torch.Tensor | np.ndarray | list,
        target: torch.Tensor | np.ndarray | list,
        **kwargs: Any,
    ) -> None:
        """Validate inputs (single, batch, multi-scale, or batch-multi-scale).

        Args:
            pred: Single crop, batch, list of crops, or batch of lists.
                Single crop/feature formats:
                - Single crop (numpy): [H, W, C] uint8 RGB [0, 255]
                - Batch crops (numpy): [N, H, W, C] uint8 RGB [0, 255]
                - Single crop (torch): [C, H, W] float [0, 1]
                - Batch crops (torch): [N, C, H, W] float [0, 1]
                - Features: [N, D] or [D] for precomputed features
                Multi-scale formats (one object):
                - Multi-scale crops: List[np.ndarray], each [H, W, C]
                  Example: [crop_0.8x, crop_1.0x, crop_1.2x, crop_1.5x]
                Batch multi-scale formats (multiple objects):
                - Batch multi-scale: List[List[np.ndarray]]
                  Example: [[obj1_0.8x, obj1_1.0x, ...],
                           [obj2_0.8x, obj2_1.0x, ...], ...]
            target: Same format/shape as pred.
        """
        # List mode (multi-scale crops - single or batch)
        if isinstance(pred, list) and isinstance(target, list):
            # Validate list lengths
            if len(pred) != len(target):
                raise ValueError(f"pred and target lists must have same length: {len(pred)} vs {len(target)}")
            if len(pred) == 0:
                raise ValueError("Empty crop lists provided")

            # Check if batch multi-scale (list of lists)
            if isinstance(pred[0], list) and isinstance(target[0], list):
                # Validate batch multi-scale mode
                for obj_idx, (pred_obj, tgt_obj) in enumerate(zip(pred, target)):
                    if not isinstance(pred_obj, list) or not isinstance(tgt_obj, list):
                        raise TypeError(f"Object {obj_idx}: Expected lists, got {type(pred_obj)} and {type(tgt_obj)}")

                    if len(pred_obj) != len(tgt_obj):
                        raise ValueError(
                            f"Object {obj_idx}: pred and target scale lists must match: "
                            f"{len(pred_obj)} vs {len(tgt_obj)}"
                        )

                    # Validate each crop
                    for scale_idx, (p, t) in enumerate(zip(pred_obj, tgt_obj)):
                        if not isinstance(p, np.ndarray) or not isinstance(t, np.ndarray):
                            raise TypeError(
                                f"Object {obj_idx}, scale {scale_idx}: Expected np.ndarray, got {type(p)} and {type(t)}"
                            )
            else:
                # Single object multi-scale mode
                for i, (p, t) in enumerate(zip(pred, target)):
                    if not isinstance(p, np.ndarray) or not isinstance(t, np.ndarray):
                        raise TypeError(f"Crop {i}: pred and target must be np.ndarray, got {type(p)} and {type(t)}")
            return

        # Single crop/features mode
        if not isinstance(pred, (torch.Tensor, np.ndarray)):
            raise TypeError(f"pred must be torch.Tensor, np.ndarray, or list, got {type(pred)}")
        if not isinstance(target, (torch.Tensor, np.ndarray)):
            raise TypeError(f"target must be torch.Tensor, np.ndarray, or list, got {type(target)}")

        # Validate dimensions match
        pred_ndim = pred.ndim
        target_ndim = target.ndim
        if pred_ndim != target_ndim:
            raise ValueError(f"pred and target must have same dimensions: {pred_ndim} vs {target_ndim}")

        # Validate shape compatibility
        # For tensor [C,H,W] or [N,C,H,W], channels must match
        # For numpy [H,W,C] or [N,H,W,C], channels must match
        if isinstance(pred, torch.Tensor):
            # Tensor format: channels first
            if pred_ndim == 3 and pred.shape[0] != target.shape[0]:
                raise ValueError(f"Channel count must match: {pred.shape[0]} vs {target.shape[0]}")
            if pred_ndim == 4:
                if pred.shape[0] != target.shape[0]:
                    raise ValueError(f"Batch size must match: {pred.shape[0]} vs {target.shape[0]}")
                if pred.shape[1] != target.shape[1]:
                    raise ValueError(f"Channel count must match: {pred.shape[1]} vs {target.shape[1]}")

            # Validate value range for torch IMAGE tensors only (not features)
            # Features (ndim=2) can have any range, but images (ndim=3 or 4) should be [0, 1]
            if pred.dtype in [torch.float32, torch.float64, torch.float16]:
                if pred_ndim in [3, 4]:  # Only validate images, not feature vectors
                    if pred.min() < 0.0 or pred.max() > 1.0:
                        raise ValueError(
                            f"Torch image tensor values must be in [0, 1], got range [{pred.min():.3f}, {pred.max():.3f}]"
                        )
                    if target.min() < 0.0 or target.max() > 1.0:
                        raise ValueError(
                            f"Torch image tensor values must be in [0, 1], got range [{target.min():.3f}, {target.max():.3f}]"
                        )
        else:
            # Numpy format: channels last
            if pred_ndim == 3 and pred.shape[2] != target.shape[2]:
                raise ValueError(f"Channel count must match: {pred.shape[2]} vs {target.shape[2]}")
            if pred_ndim == 4:
                if pred.shape[0] != target.shape[0]:
                    raise ValueError(f"Batch size must match: {pred.shape[0]} vs {target.shape[0]}")
                if pred.shape[3] != target.shape[3]:
                    raise ValueError(f"Channel count must match: {pred.shape[3]} vs {target.shape[3]}")

            # Validate value range for numpy IMAGE arrays only (not features)
            # Features (ndim=2) can have any range, but images (ndim=3 or 4) should be [0, 1] or [0, 255]
            if pred.dtype == np.uint8:
                # uint8 should be [0, 255] - no explicit check needed (dtype enforces it)
                pass
            elif pred.dtype in [np.float32, np.float64, np.float16]:
                if pred_ndim in [3, 4]:  # Only validate images, not feature vectors
                    # float images should be [0, 1]
                    if pred.min() < 0.0 or pred.max() > 1.0:
                        raise ValueError(
                            f"Float numpy image array values must be in [0, 1], got range [{pred.min():.3f}, {pred.max():.3f}]"
                        )
                    if target.min() < 0.0 or target.max() > 1.0:
                        raise ValueError(
                            f"Float numpy image array values must be in [0, 1], got range [{target.min():.3f}, {target.max():.3f}]"
                        )

    def _extract_features_from_numpy(self, crops: np.ndarray) -> torch.Tensor:
        """Extract features from numpy using feature extractor.

        Args:
            crops: Images [N,H,W,C] or [H,W,C], uint8 RGB [0, 255].

        Returns:
            Features [N, D] or single feature [D].
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not initialized. Set precomputed_features_only=False.")

        # Convert to torch [N, C, H, W] in [0, 1]
        if crops.ndim == 3:
            # Single crop [H, W, C] → [1, C, H, W]
            crops_tensor = torch.from_numpy(crops).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        else:
            # Batch [N, H, W, C] → [N, C, H, W]
            crops_tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).float().to(self.device) / 255.0

        # Use feature_extractor's batch extraction
        features = self.feature_extractor.extract_features_batch(
            crops_tensor,
            return_numpy=False,
            batch_size=self.feature_batch_size,
        )

        # Ensure features is a Tensor (return_numpy=False guarantees this)
        assert isinstance(features, torch.Tensor), "Expected torch.Tensor"

        # Remove batch dimension if single crop
        if crops.ndim == 3:
            return features.squeeze(0)
        return features

    def _extract_features(self, crops: torch.Tensor) -> torch.Tensor:
        """Extract features from crops using feature extractor.

        Args:
            crops: Images [N,C,H,W] or [C,H,W], float [0, 1].

        Returns:
            Features [N, D] or single feature [D].
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not initialized. Set precomputed_features_only=False.")

        # Add batch dimension if single crop
        if crops.ndim == 3:
            # Single crop [C, H, W] → [1, C, H, W]
            crops_batch = crops.unsqueeze(0)
        else:
            crops_batch = crops

        # Use feature_extractor's batch extraction
        features = self.feature_extractor.extract_features_batch(
            crops_batch, return_numpy=False, batch_size=self.feature_batch_size
        )

        # Ensure features is a Tensor (return_numpy=False guarantees this)
        assert isinstance(features, torch.Tensor), "Expected torch.Tensor"

        # Remove batch dimension if single crop
        if crops.ndim == 3:
            return features.squeeze(0)
        return features

    def _compute_similarity(self, feats_pred: torch.Tensor, feats_target: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between features.

        Args:
            feats_pred: Predicted features [N, D] or [D].
            feats_target: Target features [N, D] or [D].

        Returns:
            Similarity scores [N] or scalar.
        """
        # Ensure batch dimension for computation
        if feats_pred.ndim == 1:
            feats_pred = feats_pred.unsqueeze(0)
            feats_target = feats_target.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute cosine similarity (returns values in [-1, 1])
        similarities = cosine_similarity(feats_pred, feats_target, dim=1)

        # Convert from [-1, 1] to [0, 1] range
        similarities = (similarities + 1.0) / 2.0

        # Remove batch dimension if input was single feature
        if squeeze_output:
            similarities = similarities.squeeze(0)

        return similarities

    def _compute(
        self,
        pred: torch.Tensor | np.ndarray | list,
        target: torch.Tensor | np.ndarray | list,
        obj_metadata: ObjectMetadata | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute semantic similarity (single, batch, or multi-scale).

        Args:
            pred: Single crop, batch, list of crops, or batch of lists.
                Single crop/feature formats:
                - Single crop (numpy): [H, W, C] uint8 RGB [0, 255]
                - Batch crops (numpy): [N, H, W, C] uint8 RGB [0, 255]
                - Single crop (torch): [C, H, W] float [0, 1]
                - Batch crops (torch): [N, C, H, W] float [0, 1]
                - Features: [N, D] or [D] for precomputed features
                Multi-scale formats:
                - Single object: List[np.ndarray] each [H, W, C]
                - Batch objects: List[List[np.ndarray]]
            target: Same format/shape as pred.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            MetricResult with similarity scores and metadata.
        """
        # Multi-scale mode (single or batch)
        if isinstance(pred, list) and isinstance(target, list):
            # Check if batch multi-scale (list of lists)
            if len(pred) > 0 and isinstance(pred[0], list):
                return self._compute_batch_multiscale(pred, target, obj_metadata)
            # Single object multi-scale
            return self._compute_multiscale(pred, target, obj_metadata)

        # Single crop or precomputed features mode
        if not self.precomputed_features_only:
            if isinstance(pred, np.ndarray):
                feats_pred = self._extract_features_from_numpy(pred)
            else:
                # Type narrowing: pred must be torch.Tensor here
                assert isinstance(pred, torch.Tensor), "Expected torch.Tensor after validation"
                feats_pred = self._extract_features(pred)

            if isinstance(target, np.ndarray):
                feats_target = self._extract_features_from_numpy(target)
            else:
                # Type narrowing: target must be torch.Tensor here
                assert isinstance(target, torch.Tensor), "Expected torch.Tensor after validation"
                feats_target = self._extract_features(target)
        else:
            if isinstance(pred, np.ndarray):
                feats_pred = torch.from_numpy(pred).to(self.device)
            else:
                # Type narrowing: pred must be torch.Tensor here
                assert isinstance(pred, torch.Tensor), "Expected torch.Tensor for precomputed features"
                feats_pred = pred
            if isinstance(target, np.ndarray):
                feats_target = torch.from_numpy(target).to(self.device)
            else:
                # Type narrowing: target must be torch.Tensor here
                assert isinstance(target, torch.Tensor), "Expected torch.Tensor for precomputed features"
                feats_target = target

        # Compute similarity
        similarities = self._compute_similarity(feats_pred, feats_target)

        # Build per-object detailed data using helper method
        similarities_list = similarities.cpu().tolist() if similarities.ndim > 0 else [similarities.item()]

        result_metadata = self._build_detailed_metadata(similarities_list, obj_metadata)

        return MetricResult(values={self._NAME: similarities}, metadata=result_metadata)

    def _build_detailed_metadata(
        self,
        similarities_list: list[float],
        obj_metadata: ObjectMetadata | None,
    ) -> dict[str, Any]:
        """Build detailed per-object data for aggregation and visualization.

        This creates a list of per-object dictionaries containing semantic
        scores (raw and adjusted), track IDs, class names, frame indices,
        and bounding boxes. This data is later used by aggregate() to compute
        per-track and per-class statistics, and by visualization to display
        crops and metrics.

        Args:
            similarities_list: List of semantic similarity scores (one per object).
            obj_metadata: ObjectMetadata containing track/frame/bbox info.

        Returns:
            Metadata dictionary containing only 'detailed_data' with per-object info.
        """
        detailed_data: list[dict[str, Any]] = []
        batch_size = len(similarities_list)

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
            semantic_raw = similarities_list[i]

            # Compute semantic_adjusted using baseline adjustment
            # Scores below baseline are penalized, above baseline are normalized
            if semantic_raw < SEMANTIC_BASELINE:
                semantic_adjusted = semantic_raw * SEMANTIC_BELOW_BASELINE_SCALE
            else:
                normalized = (semantic_raw - SEMANTIC_BASELINE) / (1.0 - SEMANTIC_BASELINE)
                semantic_adjusted = normalized**2

            obj_data: dict[str, Any] = {
                "track_id": track_ids[i] if track_ids and i < len(track_ids) else None,
                "class_name": class_names[i] if class_names and i < len(class_names) else None,
                "frame_idx": frame_idx,
                "gt_frame_idx": gt_frame_idx,
                "rendered_timestamp": rendered_timestamp,
                "semantic_raw": semantic_raw,
                "semantic_adjusted": semantic_adjusted,
            }

            # Add bbox data if provided (convert to dict format for viz)
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

            detailed_data.append(obj_data)

        return {"detailed_data": detailed_data}

    def _compute_multiscale(
        self,
        pred_crops: list[np.ndarray],
        target_crops: list[np.ndarray],
        obj_metadata: ObjectMetadata | None = None,
    ) -> MetricResult:
        """Compute similarity with multi-scale crops.

        Args:
            pred_crops: List of crops at different scales.
                Each element is [H, W, C] uint8 RGB [0, 255].
                Typical scales: [0.8x, 1.0x, 1.2x, 1.5x] = 4 crops.
            target_crops: List of crops at same scales as pred_crops.
                Same shape requirements as pred_crops.

        Returns:
            MetricResult with computed similarities (features averaged
            across scales before computing similarity).
        """
        # Extract features from multi-scale crops
        all_pred_features = self._extract_features_from_crop_list(pred_crops)
        all_target_features = self._extract_features_from_crop_list(target_crops)

        # Average across scales
        pred_features = all_pred_features.mean(dim=0)
        target_features = all_target_features.mean(dim=0)

        # Compute similarity
        similarities = self._compute_similarity(pred_features, target_features)

        # Build detailed data using helper method (single object case)
        semantic_raw = similarities.item() if similarities.ndim == 0 else similarities[0].item()

        result_metadata = self._build_detailed_metadata([semantic_raw], obj_metadata)

        return MetricResult(values={self._NAME: similarities}, metadata=result_metadata)

    def _compute_batch_multiscale(
        self,
        pred_objects: list[list[np.ndarray]],
        target_objects: list[list[np.ndarray]],
        obj_metadata: ObjectMetadata | None = None,
    ) -> MetricResult:
        """Compute similarity for multiple objects with multi-scale crops.

        Optimized for full GPU batch processing: flattens all crops from
        all objects, processes in single batch, then reshapes per object.

        Args:
            pred_objects: List of objects, each with multi-scale crops.
                Shape: List[List[np.ndarray]]
                Example: [[obj1_0.8x, obj1_1.0x, obj1_1.2x, obj1_1.5x],
                         [obj2_0.8x, obj2_1.0x, obj2_1.2x, obj2_1.5x],
                         ...]
                Each crop is [H, W, C] uint8 RGB [0, 255].
            target_objects: Same structure as pred_objects.
            obj_metadata: ObjectMetadata containing track/frame/bbox info.

        Returns:
            MetricResult with computed similarities.
            Output shape: [N_objects] where each value is the similarity
            for that object (averaged across scales).
        """
        num_objects = len(pred_objects)
        num_scales = len(pred_objects[0]) if pred_objects else 0

        # Flatten all crops from all objects
        all_pred_crops = [crop for obj_crops in pred_objects for crop in obj_crops]
        all_target_crops = [crop for obj_crops in target_objects for crop in obj_crops]

        # Extract features for all crops (N_objects × N_scales)
        all_pred_feats = self._extract_features_from_crop_list(all_pred_crops)
        all_target_feats = self._extract_features_from_crop_list(all_target_crops)

        # Reshape: [N_objects × N_scales, D] -> [N_objects, N_scales, D]
        all_pred_feats = all_pred_feats.view(num_objects, num_scales, -1)
        all_target_feats = all_target_feats.view(num_objects, num_scales, -1)

        # Average across scales: [N_objects, N_scales, D] -> [N_objects, D]
        pred_features = all_pred_feats.mean(dim=1)
        target_features = all_target_feats.mean(dim=1)

        # Compute similarities for all objects: [N_objects, D] -> [N_objects]
        similarities = self._compute_similarity(pred_features, target_features)

        # Build per-object detailed data using helper method
        similarities_list = similarities.cpu().tolist()
        result_metadata = self._build_detailed_metadata(similarities_list, obj_metadata)

        return MetricResult(values={self._NAME: similarities}, metadata=result_metadata)

    def _preprocess_crop_to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        """Preprocess a single crop to fixed size tensor using pure torch.

        Args:
            crop: Single crop [H, W, C] uint8 RGB [0, 255].

        Returns:
            Preprocessed tensor [C, min_size, min_size] float32 in [0, 1].
        """
        h, w = crop.shape[:2]
        max_dim = max(h, w)

        # Convert to tensor [C, H, W] and normalize to [0, 1]
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

        # Compute padding to make square (centered)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        pad_h_end = max_dim - h - pad_h
        pad_w_end = max_dim - w - pad_w

        # Pad to square: F.pad uses (left, right, top, bottom) order
        padded = F.pad(tensor, (pad_w, pad_w_end, pad_h, pad_h_end), mode="constant", value=0)

        # Resize to min_size x min_size using bilinear interpolation
        # interpolate expects [N, C, H, W], so add batch dim
        resized = F.interpolate(
            padded.unsqueeze(0),
            size=(self.min_size, self.min_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return resized

    def _extract_features_from_crop_list(self, crops: list[np.ndarray]) -> torch.Tensor:
        """Extract features from list of crops (batched processing).

        This method preprocesses crops to a fixed size and extracts features
        in batches for efficiency. Handles crops of different sizes by padding
        and resizing them to a uniform size before batching.

        Args:
            crops: List of crops, each [H, W, C] uint8 RGB [0, 255].
                Crops can have different sizes (different bounding boxes).

        Returns:
            Features [N, D] where N = len(crops).
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not initialized.")

        # Preprocess all crops to fixed size tensors
        crop_tensors = [self._preprocess_crop_to_tensor(crop) for crop in crops]

        # Stack into batch [N, C, min_size, min_size] and move to device
        crop_batch = torch.stack(crop_tensors).to(self.device)

        # Extract features using batch processing
        features = self.feature_extractor.extract_features_batch(
            crop_batch,
            return_numpy=False,
            batch_size=self.feature_batch_size,
        )

        assert isinstance(features, torch.Tensor), "Expected Tensor with return_numpy=False"

        return features

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

        # Organize by track_id and class_name
        detailed_by_track: dict[str, list[dict]] = {}
        detailed_by_class: dict[str, list[dict]] = {}
        for obj_data in all_detailed_data:
            track_id = obj_data.get("track_id")
            class_name = obj_data.get("class_name")

            if track_id:
                if track_id not in detailed_by_track:
                    detailed_by_track[track_id] = []
                detailed_by_track[track_id].append(obj_data)

            if class_name:
                if class_name not in detailed_by_class:
                    detailed_by_class[class_name] = []
                detailed_by_class[class_name].append(obj_data)

        # Compute per-track statistics
        per_track_result = {}
        for track_id, track_frames in detailed_by_track.items():
            stats = self._compute_semantic_stats(track_frames)
            if stats:
                track_metrics: dict[str, Any] = {}
                track_metrics["num_frames"] = int(len(track_frames))
                first_class = track_frames[0].get("class_name", "unknown") if track_frames else "unknown"
                if first_class != "unknown":
                    track_metrics["class_name"] = str(first_class)
                track_metrics.update(stats)
                per_track_result[track_id] = track_metrics

        # Compute per-class statistics
        per_class_result = {}
        for class_name, class_data in detailed_by_class.items():
            stats = self._compute_semantic_stats(class_data)
            if stats:
                class_metrics: dict[str, Any] = {}
                class_metrics["num_objects"] = int(len(class_data))
                unique_tracks = {obj.get("track_id") for obj in class_data if obj.get("track_id")}
                class_metrics["num_tracks"] = int(len(unique_tracks))
                class_metrics.update(stats)
                per_class_result[class_name] = class_metrics

        # Flatten all scores to individual [1] tensors for aggregate_tensors compatibility
        all_values = [value.values[self._NAME] for value in self._values]
        flat_tensors = [score.unsqueeze(0) for value in all_values for score in value]

        for method in self._aggregation_methods:
            if method in [AggregationMethod.MEAN, AggregationMethod.SUM, AggregationMethod.MIN, AggregationMethod.MAX]:
                aggregate_value = aggregate_tensors(flat_tensors, method=method).squeeze()

                # For MEAN, include per-track and per-class data in metadata
                metadata = (
                    {"per_track": per_track_result, "per_class": per_class_result}
                    if method == AggregationMethod.MEAN
                    else {}
                )

                aggregated_metrics[method] = MetricResult(values={self._NAME: aggregate_value}, metadata=metadata)

        return aggregated_metrics

    def _compute_semantic_stats(self, data_list: list[dict]) -> dict[str, float]:
        """Helper to compute semantic statistics for a group of objects.

        Args:
            data_list: List of per-object dictionaries with metric values.

        Returns:
            Dictionary with aggregated statistics (mean, std, min, max) for semantic metrics.
            Returns empty dict if no valid scores found.
        """
        scores_raw = [obj["semantic_raw"] for obj in data_list if "semantic_raw" in obj]
        scores_adjusted = [obj["semantic_adjusted"] for obj in data_list if "semantic_adjusted" in obj]

        if not scores_raw:
            return {}

        stats: dict[str, float] = {}

        # Semantic adjusted stats (if available)
        if scores_adjusted:
            stats["semantic_adjusted_mean"] = float(np.mean(scores_adjusted))
            stats["semantic_adjusted_std"] = float(np.std(scores_adjusted))
            stats["semantic_adjusted_min"] = float(np.min(scores_adjusted))
            stats["semantic_adjusted_max"] = float(np.max(scores_adjusted))

        # Semantic raw stats
        stats["semantic_raw_mean"] = float(np.mean(scores_raw))
        stats["semantic_raw_std"] = float(np.std(scores_raw))
        stats["semantic_raw_min"] = float(np.min(scores_raw))
        stats["semantic_raw_max"] = float(np.max(scores_raw))

        return stats

    def type(self) -> MetricType:
        """Return the type of the metric."""
        return MetricType.OBJECT_LEVEL_SEMANTIC

    def metadata(self) -> dict[str, Any]:
        """Return configuration metadata for the metric.

        Returns:
            Dict with extractor type, model path, and settings.
        """
        metadata = {
            "metric_name": self._NAME,
            "extractor_type": self.extractor_type,
            "pretrained_path": self.pretrained_path,
            "feature_batch_size": self.feature_batch_size,
            "precomputed_features_only": self.precomputed_features_only,
        }

        # Add feature extractor info if available
        if self.feature_extractor is not None and hasattr(self.feature_extractor, "feature_dim"):
            metadata["feature_dim"] = self.feature_extractor.feature_dim

        return metadata

    def reset(self) -> None:
        """Reset the metric state."""
        pass
