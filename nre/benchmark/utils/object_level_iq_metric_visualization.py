# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Visualization functions for object-level image quality metrics.

This module provides visualization utilities for analyzing object-level
metrics computed from tracked objects in rendered videos.
"""

import gc
import json
import logging
import os
import random

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from nre.benchmark.utils.shard_data_manager import ShardDataManager
from nre.metrics.utils import AggregationMethod


log = logging.getLogger(__name__)


class CachedShardFrameLoader:
    """On-demand shard frame loader with caching."""

    def __init__(
        self,
        shard_mgr: ShardDataManager,
        num_frames: int,
    ) -> None:
        """Initialize loader.

        Args:
            shard_mgr: ShardDataManager for loading frames.
            num_frames: Total frames available.
        """
        self._shard_mgr = shard_mgr
        self._num_frames = num_frames
        self._cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        """Return total frames available."""
        return self._num_frames

    def __getitem__(self, idx: int) -> Optional[np.ndarray]:
        """Load frame, using cache if available.

        Args:
            idx: Frame index.

        Returns:
            Frame array, or None if out of bounds.
        """
        if idx < 0 or idx >= self._num_frames:
            return None

        if idx in self._cache:
            return self._cache[idx]

        try:
            frame = self._shard_mgr.camera_sensor.get_frame_image_array(idx)
            self._cache[idx] = frame
            return frame
        except (OSError, IndexError, ValueError) as e:
            log.warning("Failed to load frame %d: %s", idx, e)
            return None

    def clear(self) -> None:
        """Clear cached frames to free memory."""
        self._cache.clear()


class CachedVideoFrameLoader:
    """On-demand video frame loader with caching."""

    def __init__(self, video_path: str) -> None:
        """Initialize loader.

        Args:
            video_path: Path to video file.
        """
        self._video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        self._num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        """Return total frames in video."""
        return self._num_frames

    def __getitem__(self, idx: int) -> Optional[np.ndarray]:
        """Load frame, using cache if available.

        Args:
            idx: Frame index.

        Returns:
            Frame array (RGB), or None if out of bounds or read fails.
        """
        if idx < 0 or idx >= self._num_frames:
            return None

        if idx in self._cache:
            return self._cache[idx]

        try:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._cache[idx] = rgb_frame
                return rgb_frame
            return None
        except (OSError, cv2.error) as e:
            log.warning("Failed to load video frame %d: %s", idx, e)
            return None

    def clear(self) -> None:
        """Clear cached frames to free memory."""
        self._cache.clear()

    def release(self) -> None:
        """Release video capture and clear cache."""
        self._cache.clear()
        if self._cap.isOpened():
            self._cap.release()


def _extract_crop_from_frame(
    frames: Optional[Union[List[Any], CachedShardFrameLoader, CachedVideoFrameLoader, Dict[int, Any]]],
    frame_idx: int,
    bbox: Dict[str, int],
) -> Optional[np.ndarray]:
    """Extract crop from frame using bounding box.

    Args:
        frames: List, CachedShardFrameLoader, CachedVideoFrameLoader, or dict.
        frame_idx: Index of the frame to extract from.
        bbox: Bounding box dict with keys 'x', 'y', 'width', 'height'.

    Returns:
        Cropped region as numpy array, or None if extraction fails.
    """
    if frames is None or frame_idx < 0:
        return None

    # Handle dict (subsample mode), list, or CachedShardFrameLoader
    if isinstance(frames, dict):
        frame = frames.get(frame_idx)
    elif frame_idx >= len(frames):
        return None
    else:
        frame = frames[frame_idx]
    if frame is None:
        return None

    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

    # Clamp coordinates to frame boundaries
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(frame.shape[1], x + w)
    y_end = min(frame.shape[0], y + h)

    crop = frame[y_start:y_end, x_start:x_end]
    return crop if crop.size > 0 else None


def _extract_viz_data_from_manager(
    metric_manager: Any,
    metadata: Optional[Dict[str, Any]],
    aggregated_results: Optional[Dict[str, Dict[Any, Any]]] = None,
) -> Dict[str, Any]:
    """Extract and prepare visualization data from MetricManager.

    Args:
        metric_manager: MetricManager instance with computed metrics.
        metadata: Metadata dict with video/shard paths (can be None).
        aggregated_results: Pre-computed aggregated results (optional).

    Returns:
        Prepared data structure for visualization.
    """
    if metadata is None:
        metadata = {}

    # Use provided aggregated results or compute them
    if aggregated_results is None:
        aggregated_results = metric_manager.aggregate()

    # Build detailed_by_track from raw metric values
    detailed_by_track: Dict[str, List[Dict[str, Any]]] = {}

    # Extract from semantic metric using public API
    if metric_manager.has_metric("semantic"):
        semantic_metric = metric_manager.get_metric("semantic")
        for value in semantic_metric.values():
            detailed_data = value.metadata.get("detailed_data", [])
            for obj_data in detailed_data:
                track_id = obj_data.get("track_id")
                if track_id:
                    if track_id not in detailed_by_track:
                        detailed_by_track[track_id] = []
                    detailed_by_track[track_id].append(obj_data)

    # Merge perceptual data using public API
    if metric_manager.has_metric("perceptual"):
        perceptual_metric = metric_manager.get_metric("perceptual")
        for value in perceptual_metric.values():
            detailed_data = value.metadata.get("detailed_data", [])
            for obj_data in detailed_data:
                track_id = obj_data.get("track_id")
                if track_id and track_id in detailed_by_track:
                    frame_idx = obj_data.get("frame_idx")
                    for existing_obj in detailed_by_track[track_id]:
                        if existing_obj.get("frame_idx") == frame_idx:
                            existing_obj.update(obj_data)
                            break

    # Merge per-track metrics from aggregated results (semantic + perceptual)
    per_track_merged: Dict[str, Dict[str, Any]] = {}

    if aggregated_results and "semantic" in aggregated_results:
        if AggregationMethod.MEAN in aggregated_results["semantic"]:
            per_track_semantic = aggregated_results["semantic"][AggregationMethod.MEAN].metadata.get("per_track", {})
            per_track_merged = {tid: dict(metrics) for tid, metrics in per_track_semantic.items()}

    if aggregated_results and "perceptual" in aggregated_results:
        if AggregationMethod.MEAN in aggregated_results["perceptual"]:
            per_track_perceptual = aggregated_results["perceptual"][AggregationMethod.MEAN].metadata.get(
                "per_track", {}
            )
            for track_id, perc_metrics in per_track_perceptual.items():
                if track_id in per_track_merged:
                    per_track_merged[track_id].update(perc_metrics)
                else:
                    per_track_merged[track_id] = perc_metrics

    # Build visualization data structure
    return {
        "metadata": metadata,
        "metrics": {
            "semantic": {
                "aggregated_results": [
                    {"method": "mean", "result": {"metadata": {"detailed_by_track": detailed_by_track}}},
                    {"method": "per_track", "result": {"metadata": {"per_track": per_track_merged}}},
                ]
            }
        },
    }


def visualize_tracked_objects(
    output_dir: str,
    max_tracks: Optional[int] = 10,
    samples_per_track: int = 3,
    figsize_per_row: Tuple[int, int] = (16, 4),
    metric_manager: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    aggregated_results: Optional[Dict[str, Dict[Any, Any]]] = None,
    results_path: Optional[str] = None,
) -> None:
    """Visualize ground truth vs rendered crops for tracked objects with metrics.

    Creates side-by-side visualizations showing ground truth and rendered
    object crops along with their quality metrics.

    Args:
        output_dir: Directory to save visualization plots.
        max_tracks: Maximum number of tracks to visualize (None = all).
        samples_per_track: Number of frame samples to show per track.
        figsize_per_row: Figure size for each row (width, height).
        metric_manager: MetricManager instance with computed metrics (preferred).
        metadata: Metadata dict with video/shard paths (required with metric_manager).
        aggregated_results: Pre-computed aggregated results (optional, for efficiency).
        results_path: Path to results YAML file (alternative, for standalone use).

    Note:
        Provide either (metric_manager + metadata) OR results_path.
        Using metric_manager avoids file I/O and is faster.

    Example:
        # With MetricManager (fast, no I/O)
        visualize_tracked_objects(
            output_dir='results/visualizations',
            metric_manager=metric_manager,
            metadata=metadata,
            max_tracks=5,
            samples_per_track=3
        )

        # With file path (standalone use)
        visualize_tracked_objects(
            output_dir='results/visualizations',
            results_path='results/camera_front/metrics.yaml',
            max_tracks=5,
            samples_per_track=3
        )
    """
    # Load/prepare data from either MetricManager or file
    if metric_manager is not None:
        # Extract data from MetricManager
        log.info("Extracting visualization data from MetricManager")
        results = _extract_viz_data_from_manager(metric_manager, metadata, aggregated_results)
        log.info("Using in-memory results data for visualization")
    elif results_path is not None:
        # Load from file (for standalone visualization)
        log.info("Loading results from: %s", results_path)
        if results_path.endswith((".yaml", ".yml")):
            with open(results_path, "r", encoding="utf-8") as f:
                results = yaml.safe_load(f)
        else:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
    else:
        raise ValueError("Must provide either (metric_manager + metadata) or results_path. ")

    # Extract data from standard MetricManager YAML format
    detailed_metrics: Dict[str, List[Dict[str, Any]]] = {}
    per_track_metrics: Dict[str, Dict[str, Any]] = {}

    if "metrics" in results:
        # Read from standard MetricManager section
        for metric_name in ["semantic", "perceptual"]:
            if metric_name not in results["metrics"]:
                continue

            for agg_result in results["metrics"][metric_name].get("aggregated_results", []):
                # Extract detailed_by_track from MEAN aggregation
                if agg_result["method"] == "mean":
                    result_metadata = agg_result["result"].get("metadata", {})
                    if "detailed_by_track" in result_metadata and not detailed_metrics:
                        # Only set once (from first metric that has it)
                        detailed_metrics = result_metadata["detailed_by_track"]

                # Extract per_track statistics
                elif agg_result["method"] == "per_track":
                    per_track_data = agg_result["result"].get("metadata", {}).get("per_track", {})
                    # Merge into per_track_metrics
                    for track_id, track_metrics in per_track_data.items():
                        if track_id not in per_track_metrics:
                            per_track_metrics[track_id] = {}
                        per_track_metrics[track_id].update(track_metrics)

    # Validate we found the required data
    if not detailed_metrics:
        log.warning("No detailed metrics found - visualization requires per-frame data")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have bbox data for actual crop extraction
    first_track = list(detailed_metrics.keys())[0]
    first_frame = detailed_metrics[first_track][0] if detailed_metrics[first_track] else {}
    has_bbox_data = "bbox_gt" in first_frame and "bbox_rendered" in first_frame

    # Select tracks FIRST (before loading frames) to know which frames we need
    track_ids = [tid for tid in detailed_metrics.keys() if tid in per_track_metrics]
    if max_tracks is not None and len(track_ids) > max_tracks:
        track_scores = []
        for tid in track_ids:
            track_info = per_track_metrics.get(tid, {})
            score = track_info.get(
                "semantic_adjusted_mean",
                track_info.get("semantic_raw_mean", 0.0),
            )
            track_scores.append((tid, score))
        track_scores.sort(key=lambda x: x[1], reverse=True)
        track_ids = [tid for tid, _ in track_scores[:max_tracks]]

    log.info("Selected %d tracks for visualization", len(track_ids))

    # Check if timestamp matching was used
    uses_timestamp_matching = "gt_frame_idx" in first_frame

    # Compute which frame indices will be needed for visualization
    # Include ALL frames from selected tracks (for histograms which sample randomly)
    needed_rendered_indices: set = set()
    needed_gt_indices: set = set()
    for track_id in track_ids:
        track_data = detailed_metrics[track_id]
        # Include all frames from selected tracks
        for frame_data in track_data:
            # Rendered video always uses frame_idx
            rendered_idx = frame_data.get("frame_idx")
            if rendered_idx is not None:
                needed_rendered_indices.add(rendered_idx)
            # GT frames use gt_frame_idx when timestamp matching, else frame_idx
            if uses_timestamp_matching:
                gt_idx = frame_data.get("gt_frame_idx")
                if gt_idx is not None:
                    needed_gt_indices.add(gt_idx)
            else:
                gt_idx = frame_data.get("frame_idx")
                if gt_idx is not None:
                    needed_gt_indices.add(gt_idx)

    log.info(
        "Pre-computed %d rendered, %d GT frame indices",
        len(needed_rendered_indices),
        len(needed_gt_indices),
    )

    video_frames: Optional[Union[List[Any], Dict[int, Any], CachedVideoFrameLoader]] = None
    gt_frames: Optional[Union[List[Any], CachedShardFrameLoader, Dict[int, Any]]] = None

    if has_bbox_data:
        log.info("BBox data found - will display actual object crops")

        # Get video path and shard pattern from metadata
        rendered_video_path = results["metadata"].get("rendered_video", "")

        # Use lazy loader for rendered video frames (loads on-demand)
        if rendered_video_path and os.path.exists(rendered_video_path):
            log.info("Initializing lazy video loader: %s", rendered_video_path)
            video_frames = CachedVideoFrameLoader(rendered_video_path)
            log.info("Video has %d frames (loaded on-demand)", len(video_frames))

        # Try to load GT frames from shard (if shard path available)
        shard_pattern = results["metadata"].get("shard_pattern", "")
        camera_id = results["metadata"].get("camera_id", "")

        if shard_pattern and camera_id:
            try:
                log.info("Loading GT frames from shard...")
                shard_mgr = ShardDataManager(shard_pattern, camera_id)
                num_camera_frames = len(shard_mgr.camera_sensor.get_frame_index_range())

                if uses_timestamp_matching:
                    # Timestamp mode: use cached loader (on-demand loading)
                    log.info("Using cached GT frame loader (timestamp mode)")
                    gt_frames = CachedShardFrameLoader(shard_mgr, num_camera_frames)
                    log.info(
                        "Initialized cached loader for %d GT frames",
                        len(gt_frames),
                    )
                else:
                    # Subsample mode: load GT frames keyed by rendered video index
                    gt_subsample = results["metadata"].get("gt_subsample_frames", 1)
                    gt_frames_dict: Dict[int, Any] = {}
                    for rendered_idx in needed_rendered_indices:
                        gt_camera_idx = rendered_idx * gt_subsample
                        if gt_camera_idx < num_camera_frames:
                            gt_frame = shard_mgr.camera_sensor.get_frame_image_array(gt_camera_idx)
                            gt_frames_dict[rendered_idx] = gt_frame
                    gt_frames = gt_frames_dict
                    log.info(
                        "Loaded %d GT frames (subsample=%d, needed only)",
                        len(gt_frames),
                        gt_subsample,
                    )
            except (OSError, IOError) as e:
                log.warning("Could not load GT frames due to I/O error: %s", e)
                gt_frames = None
            except (KeyError, IndexError) as e:
                log.warning("Could not load GT frames due to data format issue: %s", e)
                gt_frames = None
    else:
        log.info("No BBox data - will show placeholder text")

    # Visualize each track
    for track_id in track_ids:
        track_data = detailed_metrics[track_id]
        track_info = per_track_metrics.get(track_id, {})
        # Handle both "class_name" and "class" keys
        class_name = track_info.get("class_name", track_info.get("class", "unknown"))

        # Sample frames from this track
        num_frames = len(track_data)
        if num_frames <= samples_per_track:
            sample_indices = list(range(num_frames))
        else:
            # Evenly space samples
            step = num_frames // samples_per_track
            sample_indices = [i * step for i in range(samples_per_track)]

        # Create figure with 3 columns: GT | Metrics | Rendered
        num_rows = len(sample_indices)
        # Very compact width for tight layout
        fig_width = 11
        fig_height = figsize_per_row[1] * num_rows
        fig, axes = plt.subplots(
            num_rows,
            3,
            figsize=(fig_width, fig_height),
            gridspec_kw={"width_ratios": [1, 0.55, 1]},
        )
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"Track: {track_id} | Class: {class_name} | "
            f"Semantic (adj): {track_info.get('semantic_adjusted_mean', 0.0):.3f} ± "
            f"{track_info.get('semantic_adjusted_std', 0.0):.3f}",
            fontsize=14,
            fontweight="bold",
        )

        for row_idx, sample_idx in enumerate(sample_indices):
            frame_data = track_data[sample_idx]
            frame_idx = frame_data["frame_idx"]

            # Get all metrics
            semantic_raw = frame_data.get("semantic_raw", 0.0)
            semantic_adjusted = frame_data.get("semantic_adjusted", 0.0)

            # Perceptual metrics
            perceptual_score = frame_data.get("perceptual_score", 0.0)
            ssim = frame_data.get("ssim", 0.0)
            edge_sim = frame_data.get("edge_similarity", 0.0)
            gradient_sim = frame_data.get("gradient_similarity", 0.0)
            blur_score = frame_data.get("blur_score_scaled", frame_data.get("blur_score", 0.0))
            artifact_score = frame_data.get("artifact_score_scaled", frame_data.get("artifact_score", 0.0))
            color_sim = frame_data.get("color_similarity", frame_data.get("color_hist_similarity", 0.0))
            cem_score = frame_data.get("cem_score", 0.0)
            hue_var_score = frame_data.get("hue_var_score", frame_data.get("hue_variance_score", 0.0))
            chroma_hf = frame_data.get("chroma_hf_score", 0.0)
            channel_coherence = frame_data.get("channel_coherence_score", 0.0)
            y_chroma_ratio = frame_data.get("y_chroma_ratio_score", 0.0)

            # Extract actual crops if bbox data available
            gt_crop = None
            rendered_crop = None

            if has_bbox_data and frame_data.get("bbox_gt") and frame_data.get("bbox_rendered"):
                # Use gt_frame_idx for GT when timestamp matching, else frame_idx
                gt_idx = frame_data.get("gt_frame_idx", frame_idx)
                gt_crop = _extract_crop_from_frame(gt_frames, gt_idx, frame_data["bbox_gt"])
                rendered_crop = _extract_crop_from_frame(video_frames, frame_idx, frame_data["bbox_rendered"])

            # Display GT (actual crop or placeholder)
            if gt_crop is not None and gt_crop.size > 0:
                axes[row_idx, 0].imshow(gt_crop)
                axes[row_idx, 0].set_title("Ground Truth", fontsize=10, fontweight="bold")
                axes[row_idx, 0].axis("off")
            else:
                # Fallback to placeholder
                axes[row_idx, 0].text(
                    0.5,
                    0.5,
                    f"GT\nFrame {frame_idx}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="darkgreen",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
                axes[row_idx, 0].set_title("Ground Truth", fontsize=10, fontweight="bold")
                axes[row_idx, 0].axis("off")

            # Middle column: Metrics text
            axes[row_idx, 1].axis("off")
            metrics_text = (
                f"OVERALL SCORES\n"
                f"─────────────────────\n"
                f"Semantic (adj):   {semantic_adjusted:.4f}\n"
                f"Perceptual Score: {perceptual_score:.4f}\n"
                f"\n"
                f"SEMANTIC SIMILARITY\n"
                f"─────────────────────\n"
                f"Raw (Cosine):     {semantic_raw:.4f}\n"
                f"Adjusted:         {semantic_adjusted:.4f}\n"
                f"\n"
                f"PERCEPTUAL METRICS\n"
                f"─────────────────────\n"
                f"SSIM:             {ssim:.4f}\n"
                f"Edge Sim:         {edge_sim:.4f}\n"
                f"Gradient Sim:     {gradient_sim:.4f}\n"
                f"Color Sim:        {color_sim:.4f}\n"
                f"Blur Score:       {blur_score:.4f}\n"
                f"Artifact Score:   {artifact_score:.4f}\n"
                f"CEM Score:        {cem_score:.4f}\n"
                f"Hue Variance:     {hue_var_score:.4f}\n"
                f"Chroma HF:        {chroma_hf:.4f}\n"
                f"Channel Coher.:   {channel_coherence:.4f}\n"
                f"Y/Chroma Ratio:   {y_chroma_ratio:.4f}"
            )

            axes[row_idx, 1].text(
                0.5,
                0.5,
                metrics_text,
                ha="center",
                va="center",
                transform=axes[row_idx, 1].transAxes,
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", edgecolor="gray", alpha=0.95),
            )

            # Right column: Rendered view
            if rendered_crop is not None and rendered_crop.size > 0:
                axes[row_idx, 2].imshow(rendered_crop)
                axes[row_idx, 2].set_title(
                    f"Rendered View (Frame {frame_idx})",
                    fontsize=10,
                    fontweight="bold",
                )
                axes[row_idx, 2].axis("off")
            else:
                # Fallback to placeholder
                axes[row_idx, 2].text(
                    0.5,
                    0.5,
                    f"Rendered\nFrame {frame_idx}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="darkblue",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                )
                axes[row_idx, 2].set_title(
                    f"Rendered View (Frame {frame_idx})",
                    fontsize=10,
                    fontweight="bold",
                )
                axes[row_idx, 2].axis("off")

        # Adjust layout with minimal spacing to keep rows aligned
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
        plt.subplots_adjust(hspace=0.15, wspace=0.1, top=0.96)  # Tight spacing

        # Save figure
        output_path = os.path.join(output_dir, f"track_{track_id}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        gc.collect()
        log.info("Saved visualization: %s", output_path)

        # Create time-series visualization showing all frames
        _create_timeseries_visualization(
            track_id=track_id,
            track_data=track_data,
            track_info=track_info,
            class_name=class_name,
            gt_frames=gt_frames,
            video_frames=video_frames,
            has_bbox_data=has_bbox_data,
            output_dir=output_dir,
            samples_per_track=samples_per_track,
        )

        # Clear frame caches after each track to free memory
        if isinstance(gt_frames, CachedShardFrameLoader):
            gt_frames.clear()
        if isinstance(video_frames, CachedVideoFrameLoader):
            video_frames.clear()

    # Filter metrics to selected tracks only (avoids loading frames for all tracks)
    selected_detailed = {tid: detailed_metrics[tid] for tid in track_ids}
    selected_per_track = {tid: per_track_metrics[tid] for tid in track_ids if tid in per_track_metrics}

    # Create score histogram with example crops for semantic raw
    _create_score_histogram_visualization(
        detailed_metrics=selected_detailed,
        gt_frames=gt_frames,
        video_frames=video_frames,
        has_bbox_data=has_bbox_data,
        output_dir=output_dir,
        score_key="semantic_raw",
        title="Semantic Raw Score (Cosine Similarity)",
        filename="semantic_raw_histogram_with_examples.png",
    )
    if isinstance(gt_frames, CachedShardFrameLoader):
        gt_frames.clear()
    if isinstance(video_frames, CachedVideoFrameLoader):
        video_frames.clear()

    # Create histograms for each perceptual metric
    perceptual_metrics = [
        ("ssim", "SSIM Score"),
        ("edge_similarity", "Edge Similarity"),
        ("gradient_similarity", "Gradient Similarity"),
        ("color_similarity", "Color Similarity"),
        ("blur_score_scaled", "Blur Score (Scaled)"),
        ("artifact_score_scaled", "Artifact Score (Scaled)"),
        ("cem_score", "CEM Score"),
        ("hue_var_score", "Hue Variance Score"),
        ("channel_coherence_score", "Channel Coherence Score"),
        ("perceptual_score", "Overall Perceptual Score"),
    ]

    for score_key, title in perceptual_metrics:
        _create_score_histogram_visualization(
            detailed_metrics=selected_detailed,
            gt_frames=gt_frames,
            video_frames=video_frames,
            has_bbox_data=has_bbox_data,
            output_dir=output_dir,
            score_key=score_key,
            title=title,
            filename=f"{score_key}_histogram_with_examples.png",
        )
        if isinstance(gt_frames, CachedShardFrameLoader):
            gt_frames.clear()
        if isinstance(video_frames, CachedVideoFrameLoader):
            video_frames.clear()

    # Create track summary visualization with frame samples
    _create_track_summary_visualization(
        all_track_data=selected_per_track,
        detailed_metrics=selected_detailed,
        gt_frames=gt_frames,
        video_frames=video_frames,
        has_bbox_data=has_bbox_data,
        output_dir=output_dir,
    )
    if isinstance(gt_frames, CachedShardFrameLoader):
        gt_frames.clear()
    if isinstance(video_frames, CachedVideoFrameLoader):
        video_frames.clear()

    # Cleanup to free memory
    if isinstance(video_frames, CachedVideoFrameLoader):
        video_frames.release()
    plt.close("all")
    log.info("Visualization complete! Saved to: %s", output_dir)


def _create_score_histogram_visualization(
    detailed_metrics: Dict[str, List[Dict[str, Any]]],
    gt_frames: Optional[Union[List[Any], CachedShardFrameLoader, Dict[int, Any]]],
    video_frames: Optional[Union[List[Any], Dict[int, Any], CachedVideoFrameLoader]],
    has_bbox_data: bool,
    output_dir: str,
    score_key: str,
    title: str,
    filename: str,
    num_bins: int = 10,
    examples_per_bin: int = 6,
) -> None:
    """Create histogram of all scores with example crops from each bin.

    Args:
        detailed_metrics: Frame-by-frame metrics for all tracks.
        gt_frames: List of ground truth frames.
        video_frames: List of rendered video frames.
        has_bbox_data: Whether bbox data is available.
        output_dir: Directory to save visualizations.
        score_key: Key for the score to visualize (e.g., 'semantic_raw', 'ssim').
        title: Title for the histogram plot.
        filename: Output filename for the visualization.
        num_bins: Number of histogram bins.
        examples_per_bin: Number of example crops to show per bin.
    """
    log.info("Creating histogram visualization for %s...", score_key)

    # Collect all scores and frame data
    all_data = []
    for track_id, track_data in detailed_metrics.items():
        for frame_data in track_data:
            score_value = frame_data.get(score_key, frame_data.get(score_key.replace("_scaled", ""), 0.0))
            all_data.append(
                {
                    "track_id": track_id,
                    "frame_data": frame_data,
                    "score": score_value,
                }
            )

    if len(all_data) == 0:
        log.warning("No data available for histogram visualization of %s", score_key)
        return

    # Extract scores for histogram
    scores = [d["score"] for d in all_data]

    # Create bins based on actual data range
    min_score = min(scores)
    max_score = max(scores)
    # Add small margin to include edge values
    score_range = max_score - min_score
    margin = score_range * 0.05 if score_range > 0 else 0.05
    bin_edges = np.linspace(min_score - margin, max_score + margin, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each data point to a bin
    for data_point in all_data:
        score = data_point["score"]
        bin_idx = np.digitize(score, bin_edges) - 1
        bin_idx = max(0, min(num_bins - 1, bin_idx))  # Clamp to valid range
        data_point["bin_idx"] = bin_idx

    # Sample examples from each bin
    bin_samples: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_bins)}
    for data_point in all_data:
        bin_idx = data_point["bin_idx"]
        bin_samples[bin_idx].append(data_point)

    # Randomly sample examples from each bin
    for bin_idx in bin_samples:
        if len(bin_samples[bin_idx]) > examples_per_bin:
            bin_samples[bin_idx] = random.sample(bin_samples[bin_idx], examples_per_bin)

    # Create figure with more space for separation
    fig_width = 20
    fig_height = 5 + examples_per_bin * 3  # More height for histogram + example rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1 + examples_per_bin,
        num_bins,
        height_ratios=[2.0] + [1] * examples_per_bin,  # Larger histogram row
        hspace=0.5,
        wspace=0.1,  # Increased hspace for clear separation
    )

    # Top row: Histogram spanning all columns
    ax_hist = fig.add_subplot(gs[0, :])
    counts, _, patches = ax_hist.hist(scores, bins=bin_edges, edgecolor="black", alpha=0.7, linewidth=1.5)
    ax_hist.set_xlabel(title, fontsize=12, fontweight="bold")
    ax_hist.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax_hist.set_title(
        f"Distribution of {title} Across All Objects and Frames (Total: {len(all_data)})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax_hist.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Color histogram bars
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.RdYlGn(bin_centers[i]))

    # Add count labels on bars
    for i, count in enumerate(counts):
        if count > 0:
            ax_hist.text(
                bin_centers[i], count, f"{int(count)}", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    # Bottom rows: Example crops for each bin
    for bin_idx in range(num_bins):
        samples = bin_samples[bin_idx]

        for example_idx in range(examples_per_bin):
            row_idx = 1 + example_idx
            ax = fig.add_subplot(gs[row_idx, bin_idx])

            if example_idx < len(samples):
                sample = samples[example_idx]
                frame_data = sample["frame_data"]
                frame_idx = frame_data["frame_idx"]

                # Create side-by-side GT and Rendered
                gt_crop = None
                rendered_crop = None

                if has_bbox_data and frame_data.get("bbox_gt") and frame_data.get("bbox_rendered"):
                    # Use gt_frame_idx for GT when timestamp matching, else frame_idx
                    gt_idx = frame_data.get("gt_frame_idx", frame_idx)
                    gt_crop = _extract_crop_from_frame(gt_frames, gt_idx, frame_data["bbox_gt"])
                    rendered_crop = _extract_crop_from_frame(video_frames, frame_idx, frame_data["bbox_rendered"])

                    # Combine GT and Rendered side-by-side
                    if (
                        gt_crop is not None
                        and rendered_crop is not None
                        and gt_crop.size > 0
                        and rendered_crop.size > 0
                    ):
                        # Resize to same height
                        h_target = min(gt_crop.shape[0], rendered_crop.shape[0], 150)
                        gt_resized = cv2.resize(
                            gt_crop, (int(gt_crop.shape[1] * h_target / gt_crop.shape[0]), h_target)
                        )
                        rnd_resized = cv2.resize(
                            rendered_crop, (int(rendered_crop.shape[1] * h_target / rendered_crop.shape[0]), h_target)
                        )

                        # Concatenate horizontally
                        combined = np.hstack([gt_resized, np.ones((h_target, 5, 3), dtype=np.uint8) * 255, rnd_resized])
                        ax.imshow(combined)
                        ax.set_title(f"Score: {sample['score']:.3f}\nTrack: {sample['track_id'][:15]}", fontsize=7)
                    else:
                        ax.text(0.5, 0.5, "No crop", ha="center", va="center", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No bbox", ha="center", va="center", fontsize=8)
            else:
                # Empty cell
                ax.text(0.5, 0.5, "", ha="center", va="center")

            ax.axis("off")

    plt.suptitle(f"{title} Distribution with Example Crops (GT | Rendered)", fontsize=16, fontweight="bold", y=0.995)

    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    log.info("Saved %s histogram visualization: %s", score_key, output_path)


def _create_timeseries_visualization(
    track_id: str,
    track_data: List[Dict[str, Any]],
    track_info: Dict[str, Any],
    class_name: str,
    gt_frames: Optional[Union[List[Any], CachedShardFrameLoader, Dict[int, Any]]],
    video_frames: Optional[Union[List[Any], Dict[int, Any], CachedVideoFrameLoader]],
    has_bbox_data: bool,
    output_dir: str,
    samples_per_track: int = 20,
) -> None:
    """Create time-series visualization with metrics plot and frame thumbnails.

    Args:
        track_id: Track identifier.
        track_data: List of per-frame metrics for this track.
        track_info: Aggregated statistics for this track.
        class_name: Object class name.
        gt_frames: List of ground truth frames.
        video_frames: List of rendered video frames.
        has_bbox_data: Whether bbox data is available.
        output_dir: Directory to save visualizations.
        samples_per_track: Maximum frames to display in visualization.
    """
    num_frames = len(track_data)
    if num_frames == 0:
        return

    # Limit frames for visualization - sample evenly if needed
    max_frames_display = min(samples_per_track, num_frames)
    if num_frames > max_frames_display:
        # Sample evenly spaced frames
        step = num_frames / max_frames_display
        sampled_indices = [int(i * step) for i in range(max_frames_display)]
        track_data_sampled = [track_data[i] for i in sampled_indices]
    else:
        track_data_sampled = track_data
        sampled_indices = list(range(num_frames))

    num_display_frames = len(track_data_sampled)

    # Extract metrics for sampled frames
    frame_indices = [d["frame_idx"] for d in track_data_sampled]
    semantic_raw = [d.get("semantic_raw", 0.0) for d in track_data_sampled]
    semantic_adjusted = [d.get("semantic_adjusted", 0.0) for d in track_data_sampled]

    # Perceptual metrics
    perceptual_scores = [d.get("perceptual_score", 0.0) for d in track_data_sampled]
    ssim_scores = [d.get("ssim", 0.0) for d in track_data_sampled]
    edge_sim = [d.get("edge_similarity", 0.0) for d in track_data_sampled]
    gradient_sim = [d.get("gradient_similarity", 0.0) for d in track_data_sampled]
    color_sim = [d.get("color_similarity", d.get("color_hist_similarity", 0.0)) for d in track_data_sampled]
    blur_score = [d.get("blur_score_scaled", d.get("blur_score", 0.0)) for d in track_data_sampled]
    artifact_score = [d.get("artifact_score_scaled", d.get("artifact_score", 0.0)) for d in track_data_sampled]
    cem_score = [d.get("cem_score", 0.0) for d in track_data_sampled]
    hue_var_score = [d.get("hue_var_score", d.get("hue_variance_score", 0.0)) for d in track_data_sampled]
    channel_coherence = [d.get("channel_coherence_score", 0.0) for d in track_data_sampled]

    # Create figure: metrics plot on top, frame thumbnails below
    # Make it very wide to accommodate displayed frames
    fig_width = max(20, num_display_frames * 1.5)  # Scale width with number of frames
    fig_height = 12  # Increased height for two plot rows

    # Create gridspec: two plot rows on top, bottom rows for images
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(4, num_display_frames, height_ratios=[1.2, 1.2, 1, 1], hspace=0.4, wspace=0.05)

    # First plot row: Overall scores
    ax_overall = fig.add_subplot(gs[0, :])
    ax_overall.plot(range(num_display_frames), semantic_raw, "o-", label="Semantic Raw", linewidth=2, markersize=5)
    ax_overall.plot(
        range(num_display_frames),
        semantic_adjusted,
        "s-",
        label="Semantic Adj",
        linewidth=2.5,
        markersize=6,
        color="red",
    )
    ax_overall.plot(range(num_display_frames), perceptual_scores, "^-", label="Perceptual", linewidth=2, markersize=5)

    ax_overall.set_ylabel("Overall Scores", fontsize=10, fontweight="bold")
    title_text = (
        f"Track {track_id} | Class: {class_name} | "
        f"Semantic (adj) Mean: {track_info.get('semantic_adjusted_mean', 0.0):.3f} ± "
        f"{track_info.get('semantic_adjusted_std', 0.0):.3f}"
    )
    if num_frames > max_frames_display:
        title_text += f" | Showing {num_display_frames}/{num_frames} frames"
    ax_overall.set_title(title_text, fontsize=13, fontweight="bold")
    ax_overall.legend(loc="upper right", fontsize=8, ncol=4)
    ax_overall.grid(True, alpha=0.3, linestyle="--")
    ax_overall.set_ylim(-0.05, 1.05)
    ax_overall.set_xticklabels([])  # No x-labels on first plot

    # Add value annotations for semantic_raw only
    for i, (x, y) in enumerate(zip(range(num_display_frames), semantic_raw)):
        ax_overall.text(
            x,
            y + 0.03,  # Position slightly above the point
            f"{y:.3f}",
            fontsize=7,
            ha="center",
            va="bottom",
            color="tab:blue",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Second plot row: Detailed perceptual metrics
    ax_detail = fig.add_subplot(gs[1, :])
    ax_detail.plot(range(num_display_frames), ssim_scores, "o-", label="SSIM", linewidth=1.5, markersize=4)
    ax_detail.plot(range(num_display_frames), edge_sim, "s-", label="Edge Sim", linewidth=1.5, markersize=4)
    ax_detail.plot(range(num_display_frames), gradient_sim, "^-", label="Gradient", linewidth=1.5, markersize=4)
    ax_detail.plot(range(num_display_frames), color_sim, "v-", label="Color", linewidth=1.5, markersize=4)
    ax_detail.plot(range(num_display_frames), blur_score, "D-", label="Blur", linewidth=1.5, markersize=4)
    ax_detail.plot(range(num_display_frames), artifact_score, "p-", label="Artifact", linewidth=1.5, markersize=4)
    ax_detail.plot(range(num_display_frames), cem_score, "h-", label="CEM", linewidth=1.5, markersize=4)
    ax_detail.plot(range(num_display_frames), hue_var_score, "*-", label="Hue Var", linewidth=1.5, markersize=5)
    ax_detail.plot(range(num_display_frames), channel_coherence, "X-", label="Ch Coher", linewidth=1.5, markersize=4)

    ax_detail.set_xlabel("Frame Index", fontsize=10, fontweight="bold")
    ax_detail.set_ylabel("Perceptual Metrics", fontsize=10, fontweight="bold")
    ax_detail.legend(loc="upper right", fontsize=7, ncol=5)
    ax_detail.grid(True, alpha=0.3, linestyle="--")
    ax_detail.set_ylim(-0.05, 1.05)
    ax_detail.set_xticks(range(num_display_frames))
    ax_detail.set_xticklabels([str(idx) for idx in frame_indices], rotation=45, ha="right", fontsize=8)

    # Bottom rows: Ground truth and rendered frames
    for col_idx, frame_data in enumerate(track_data_sampled):
        frame_idx = frame_data["frame_idx"]

        # Extract crops if available
        gt_crop = None
        rendered_crop = None

        if has_bbox_data and frame_data.get("bbox_gt") and frame_data.get("bbox_rendered"):
            # Use gt_frame_idx for GT crops if available, otherwise fall back to frame_idx
            gt_idx = frame_data.get("gt_frame_idx", frame_idx)
            gt_crop = _extract_crop_from_frame(gt_frames, gt_idx, frame_data["bbox_gt"])
            rendered_crop = _extract_crop_from_frame(video_frames, frame_idx, frame_data["bbox_rendered"])

        # Ground truth thumbnail
        ax_gt = fig.add_subplot(gs[2, col_idx])
        if gt_crop is not None and gt_crop.size > 0:
            ax_gt.imshow(gt_crop)
        else:
            ax_gt.text(0.5, 0.5, f"GT\n{frame_idx}", ha="center", va="center", fontsize=8)
        ax_gt.axis("off")
        if col_idx == 0:
            ax_gt.set_ylabel("GT", fontsize=10, fontweight="bold", rotation=0, labelpad=20)

        # Rendered thumbnail
        ax_rendered = fig.add_subplot(gs[3, col_idx])
        if rendered_crop is not None and rendered_crop.size > 0:
            ax_rendered.imshow(rendered_crop)
        else:
            ax_rendered.text(0.5, 0.5, f"Rnd\n{frame_idx}", ha="center", va="center", fontsize=8)
        ax_rendered.axis("off")
        if col_idx == 0:
            ax_rendered.set_ylabel("Rendered", fontsize=10, fontweight="bold", rotation=0, labelpad=20)

    # Save time-series figure
    output_path_ts = os.path.join(output_dir, f"track_{track_id}_timeseries.png")
    plt.savefig(output_path_ts, dpi=120, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    log.info("Saved time-series visualization: %s", output_path_ts)


def _create_track_summary_visualization(
    all_track_data: Dict[str, Dict[str, Any]],
    detailed_metrics: Dict[str, List[Dict[str, Any]]],
    gt_frames: Optional[Union[List[Any], CachedShardFrameLoader, Dict[int, Any]]],
    video_frames: Optional[Union[List[Any], Dict[int, Any], CachedVideoFrameLoader]],
    has_bbox_data: bool,
    output_dir: str,
) -> None:
    """Create summary visualization showing mean semantic scores across all tracks.

    Args:
        all_track_data: Dictionary mapping track_id to track statistics.
        detailed_metrics: Dictionary mapping track_id to per-frame metrics.
        gt_frames: List of ground truth frames.
        video_frames: List of rendered video frames.
        has_bbox_data: Whether bbox data is available.
        output_dir: Directory to save visualizations.
    """
    if not all_track_data:
        return

    # Extract track IDs and mean semantic raw scores
    track_ids = []
    mean_semantic_raw = []
    track_frame_data = []

    for track_id, track_info in all_track_data.items():
        if track_id not in detailed_metrics or not detailed_metrics[track_id]:
            continue

        track_ids.append(track_id)
        mean_semantic_raw.append(track_info.get("semantic_raw_mean", 0.0))
        track_frame_data.append(detailed_metrics[track_id])

    if not track_ids:
        return

    num_tracks = len(track_ids)
    num_frames_to_show = 7  # Show 7-8 frames per track

    # Create figure with enough height for frames below x-axis
    fig_width = max(20, num_tracks * 2.5)
    fig_height = 12

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create gridspec: top for bar chart, bottom for frames
    gs = fig.add_gridspec(
        num_frames_to_show + 1, num_tracks, height_ratios=[2.0] + [1.0] * num_frames_to_show, hspace=0.15, wspace=0.1
    )

    # Top subplot: Bar chart of mean semantic scores
    ax_bar = fig.add_subplot(gs[0, :])

    x_positions = np.arange(num_tracks)
    # Set bar width to 0.8 to make bars align with columns below
    bars = ax_bar.bar(x_positions, mean_semantic_raw, width=0.8, color="steelblue", alpha=0.7)

    # Color bars based on score (red=low, green=high)
    for i, bar in enumerate(bars):
        score = mean_semantic_raw[i]
        if score < 0.5:
            bar.set_color("red")
        elif score < 0.7:
            bar.set_color("orange")
        else:
            bar.set_color("green")

    ax_bar.set_ylabel("Mean Semantic Raw Score", fontsize=12, fontweight="bold")
    ax_bar.set_title("Mean Semantic Similarity Per Track (with Frame Samples)", fontsize=14, fontweight="bold")
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(track_ids, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylim(0, 1.05)
    # Set x-axis limits to align with frame columns
    ax_bar.set_xlim(-0.5, num_tracks - 0.5)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (x, y) in enumerate(zip(x_positions, mean_semantic_raw)):
        ax_bar.text(x, y + 0.02, f"{y:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Below: Show sampled frames for each track
    for track_idx, track_id in enumerate(track_ids):
        frame_data = track_frame_data[track_idx]
        num_frames = len(frame_data)

        # Sample frames evenly from 0 to 100 (or total frames if less)
        max_frame_idx = min(100, num_frames - 1)
        if max_frame_idx > 0:
            frame_indices = np.linspace(0, max_frame_idx, num_frames_to_show, dtype=int)
        else:
            frame_indices = np.array([0] * num_frames_to_show, dtype=int)

        for row_idx, frame_idx in enumerate(frame_indices):
            if frame_idx >= num_frames:
                frame_idx = num_frames - 1

            frame_info = frame_data[frame_idx]
            actual_frame_idx = frame_info.get("frame_idx", frame_idx)

            # Extract actual crops if bbox data available
            gt_crop = None
            rendered_crop = None

            if has_bbox_data and frame_info.get("bbox_gt") and frame_info.get("bbox_rendered"):
                # Use gt_frame_idx for GT when timestamp matching, else actual_frame_idx
                gt_idx = frame_info.get("gt_frame_idx", actual_frame_idx)
                gt_crop = _extract_crop_from_frame(gt_frames, gt_idx, frame_info["bbox_gt"])
                rendered_crop = _extract_crop_from_frame(video_frames, actual_frame_idx, frame_info["bbox_rendered"])

            # Create combined image (GT | Rendered)
            ax = fig.add_subplot(gs[row_idx + 1, track_idx])

            if gt_crop is not None and gt_crop.size > 0 and rendered_crop is not None and rendered_crop.size > 0:
                # Crops are already numpy arrays in RGB format [H, W, 3]
                # Resize both to same height for concatenation
                target_height = 100  # Fixed height for thumbnails

                # Resize GT crop
                gt_h, gt_w = gt_crop.shape[:2]
                gt_new_w = int(gt_w * target_height / gt_h)
                gt_resized = cv2.resize(gt_crop, (gt_new_w, target_height), interpolation=cv2.INTER_LINEAR)

                # Resize rendered crop
                rend_h, rend_w = rendered_crop.shape[:2]
                rend_new_w = int(rend_w * target_height / rend_h)
                rend_resized = cv2.resize(rendered_crop, (rend_new_w, target_height), interpolation=cv2.INTER_LINEAR)

                # Concatenate horizontally (GT | Rendered)
                combined = np.concatenate([gt_resized, rend_resized], axis=1)
                ax.imshow(combined)

                # Add frame number label
                if row_idx == 0:
                    ax.set_title(f"F{actual_frame_idx}", fontsize=7, pad=2)
                else:
                    ax.text(
                        0.5,
                        0.95,
                        f"F{actual_frame_idx}",
                        transform=ax.transAxes,
                        fontsize=6,
                        ha="center",
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                    )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)

            ax.axis("off")

    # Save figure
    output_path = os.path.join(output_dir, "track_summary_with_frames.png")
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    log.info("Saved track summary visualization: %s", output_path)
