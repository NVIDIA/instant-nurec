# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import csv
import json
import logging
import os

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from nre.benchmark.utils import ObjectDataLoader, load_camera_offset_json, visualize_tracked_objects
from nre.benchmark.utils.detailed_metrics_writer import write_detailed_metrics
from nre.metrics import AggregationMethod, MetricFactory, MetricManager, MetricType, ObjectMetadata


# Mapping of CLI metric names to MetricType enum values
IMAGE_COMPARISON_METRICS = {
    "psnr": MetricType.PSNR,
    "ssim": MetricType.SSIM,
    "lpips": MetricType.LPIPS,
}


log = logging.getLogger(__name__)

DEFAULT_TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_timestamps_json(json_path: str) -> Dict[str, int]:
    """Load timestamps.json exported by the render CLI

    Args:
        json_path: Path to the JSON file

    Returns:
        Dictionary that maps image file names (without extension) to frame-end timestamps in microseconds
    """

    with open(json_path, "r", encoding="utf-8") as f:
        frame_list = json.load(f)

    timestamps: Dict[str, int] = {}
    for item in frame_list:
        file_name = item["file_name"]  # File name with extension
        base_name, _ = os.path.splitext(file_name)
        timestamp = int(item["frame_end_timestamp_us"])
        timestamps[base_name] = timestamp

    return timestamps


def find_image(file_path_noext: str, extensions: Optional[List[str]] = None) -> str:
    """Check if an image with any supported extension exists at the given file path"""
    if extensions is None:
        extensions = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]
    for ext in extensions:
        candidate_path = file_path_noext + ext
        if os.path.isfile(candidate_path):
            return candidate_path
    raise ValueError(f"No image found {file_path_noext} with any supported extension")


def load_image_to_tensor(image_path: str, device: torch.device = DEFAULT_TORCH_DEVICE) -> torch.Tensor:
    """Load an RGB image into a float32 torch.Tensor of shape (C, H, W) with values within [0, 1.0]"""
    image: Image.Image = Image.open(image_path)
    image = image.convert("RGB")
    return torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1).float() / 255.0


def load_mask_to_tensor(
    image_path: str, device: torch.device = DEFAULT_TORCH_DEVICE, invert: bool = False
) -> torch.Tensor:
    """Load an 2D image mask to a 2D boolean torch.Tensor with optional mask inversion"""
    image: Image.Image = Image.open(image_path)
    image = image.convert("L")
    mask_tensor = torch.from_numpy(np.array(image)).to(device).bool()
    return mask_tensor if not invert else ~mask_tensor


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Overlays a mask on an image by making pixels in the mask region brighter but leaving other pixels unchanged."""

    assert mask.ndim == 2, "mask must be a 2D array"
    assert image.ndim == 3, "image must be a 3D array"
    assert mask.shape == image.shape[:2], "image and mask must have the same resolution"
    assert mask.dtype == bool, "mask must be a boolean array"

    overlay = image.copy()
    overlay[mask] = ((1 - alpha) * image[mask].astype(np.float32) + alpha * 255).astype(np.uint8)
    return overlay


def create_diff_image(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    factor: Optional[float] = 1.0,
    invert: bool = False,
) -> Tuple[np.ndarray, float]:
    """Creates an RGB abs-diff image between two images with optional masking and inversion

    Inversion makes small differences more visible.

    If factor is None, it is automatically set to 255.0 / np.max(diff), which has little effect if max is near 255.
    We recommend using a fixed factor to highlight errors at the scale of interest, consistently throughout a video..
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"
    assert pred.ndim == 3, "pred must be a 3D array"
    assert target.ndim == 3, "target must be a 3D array"
    assert pred.dtype == np.uint8, "pred must be a uint8 array"
    assert target.dtype == np.uint8, "target must be a uint8 array"
    assert mask is None or (mask.ndim == 2 and mask.dtype == bool), "mask must be a 2D boolean array when specified"
    assert factor is None or factor > 0.0, "factor must be positive"

    diff = np.abs(pred.astype(np.float32) - target.astype(np.float32))  # Prevents overflow

    if factor is None:
        max_value = np.max(diff)
        factor = 255.0 / max_value if max_value > 0 else 1.0  # Avoid division by zero in case of identical images.

    diff = np.clip(diff * factor, 0, 255).astype(np.uint8)
    if mask is not None:
        diff[mask > 0] = 0  # Suppress masked pixels
    return 255 - diff if invert else diff, factor


class TextAlignment(Enum):
    """Enum for text alignment options when drawing text on images"""

    TOP_LEFT = auto()  # Anchor text at top left
    TOP_CENTER = auto()  # Anchor text at top center
    TOP_RIGHT = auto()  # Anchor text at top right
    BOTTOM_LEFT = auto()  # Anchor text at bottom left
    BOTTOM_CENTER = auto()  # Anchor text at bottom center
    BOTTOM_RIGHT = auto()  # Anchor text at bottom right


def draw_text(
    image: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    alignment: TextAlignment = TextAlignment.TOP_CENTER,
    margin: int = 5,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] | None = (0, 0, 0),
) -> None:
    """Draws text into an image in place with the selected alignment and with background

    The text size can be controlled via the font parameter.
    """

    draw = ImageDraw.Draw(image)

    im_width, im_height = image.size

    # Dictionary to map TextAlignment to (x,y,anchor)
    # Anchor's 1st letter: l: left, m: middle, r: right
    # Anchor's 2nd letter: a: ascender (~top), m: middle, d: descender (~bottom)
    ALIGNMENTS: Dict[TextAlignment, Tuple[int, int, str]] = {
        TextAlignment.TOP_LEFT: (margin, margin, "la"),
        TextAlignment.TOP_CENTER: (im_width // 2, margin, "ma"),
        TextAlignment.TOP_RIGHT: (im_width - margin, margin, "ra"),
        TextAlignment.BOTTOM_LEFT: (margin, im_height - margin, "ld"),
        TextAlignment.BOTTOM_CENTER: (im_width // 2, im_height - margin, "md"),
        TextAlignment.BOTTOM_RIGHT: (im_width - margin, im_height - margin, "rd"),
    }
    # Resolve text position and anchor.
    x, y, anchor = ALIGNMENTS[alignment]

    # Draw text background if bg_color is provided.
    if bg_color is not None:
        left, top, right, bottom = font.getbbox(text, anchor=anchor)
        draw.rectangle(
            [(left + x - margin, top + y - margin), (right + x + margin, bottom + y + margin)], fill=bg_color
        )

    draw.text(xy=(x, y), text=text, fill=font_color, anchor=anchor, font=font)


def create_comparison_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    metrics: Optional[Dict[str, float]] = None,
    diff_factor: Optional[float] = 5.0,
) -> Image.Image:
    """Creates a 2x2 comparison image between predicted and target image with titles and metrics overlays

    Args:
         pred: Predicted float32 image tensor of shape (C, H, W) with values in [0, 1]
         target: Target float32 image tensor of shape (C, H, W) with values in [0, 1]
         mask: Optional boolean mask tensor of shape (H, W), with nonzeros at invalid pixels
         metrics: Dictionary of metrics, to overlay selected metrics on the comparison image
         diff_factor: Factor by which the difference image is multiplied to make small differences more visible.
                If None, it is automatically set to 1.0 / np.max(diff). Note however that this auto setting relies
                on only a single value in the image, does not enhance the image at all if the max value is high,
                and changes per compared image pair causing video flickering.

    Returns:
         PIL Image: A 2x2 stacked image for visualization purposes, containing
             (a) the predicted image, (b) the target image, (c) the masked difference image,
             and (d) an enhanced version of the difference image.
             Masked out pixels are highlighted on (a) and (b), and they are suppressed in (c) and (d).
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"
    assert pred.ndim == 3, "pred must be a 3D tensor"
    assert pred.shape[1:] == target.shape[1:], "pred and target must have the same shape"
    assert target.ndim == 3, "target must be a 3D tensor"
    assert pred.dtype == torch.float32, "pred must be a float32 tensor"
    assert target.dtype == torch.float32, "target must be a float32 tensor"
    assert mask is None or (mask.ndim == 2 and mask.dtype == torch.bool), (
        "mask must be a 2D boolean tensor when specified"
    )
    assert mask is None or (mask.shape == pred.shape[1:]), "mask must have the same resolution as as pred"

    to_pil = transforms.ToPILImage()

    # converts a (C, H, W) torch.Tensor to a shape (H, W, C) numpy array
    pred_array = np.array(to_pil(pred))
    target_array = np.array(to_pil(target))
    mask_array = np.array(to_pil(mask.float())) > 0 if mask is not None else None

    width = pred_array.shape[1]
    height = pred_array.shape[0]

    pred_mask_overlay = overlay_mask(pred_array, mask_array) if mask_array is not None else pred_array
    target_mask_overlay = overlay_mask(target_array, mask_array) if mask_array is not None else target_array
    diff_image, _ = create_diff_image(pred_array, target_array, mask=mask_array)
    enhanced_diff_image, diff_factor = create_diff_image(pred_array, target_array, mask=mask_array, factor=diff_factor)

    pred_mask_overlay_pil = Image.fromarray(pred_mask_overlay)
    target_mask_overlay_pil = Image.fromarray(target_mask_overlay)
    diff_image_pil = Image.fromarray(diff_image)
    enhanced_diff_image_pil = Image.fromarray(enhanced_diff_image)

    # Overlay image titles
    font = ImageFont.load_default(size=20)
    draw_text(pred_mask_overlay_pil, "Rendered", font=font)
    draw_text(target_mask_overlay_pil, "Input", font=font)
    draw_text(diff_image_pil, "Difference", font=font)
    draw_text(enhanced_diff_image_pil, f"Difference x{diff_factor:.1f}", font=font)

    # Overlay some cherry-picked metrics with manual setting of units and shown decimal places.
    if metrics is None:
        metrics = {}
    text = " | ".join([f"{metric_name.upper()}: {metric_value:.2f}" for metric_name, metric_value in metrics.items()])
    if len(text) > 0:
        draw_text(diff_image_pil, text, font=font, alignment=TextAlignment.BOTTOM_CENTER)

    # Create a 2x2 stacked image
    stacked_image = Image.new("RGB", (width * 2, height * 2))
    stacked_image.paste(pred_mask_overlay_pil, (0, 0))  # Top left
    stacked_image.paste(target_mask_overlay_pil, (width, 0))  # Top right
    stacked_image.paste(diff_image_pil, (0, height))  # Bottom left
    stacked_image.paste(enhanced_diff_image_pil, (width, height))  # Bottom right
    return stacked_image


def create_image_comparison_metric_manager(
    metrics_to_compute: List[str],
    device: str = "cuda",
    lpips_net_type: str = "alex",
) -> MetricManager:
    """Create and configure a MetricManager for metrics that compare two images.

    Args:
        metrics_to_compute: List of metric names to compute (e.g., ["psnr", "ssim", "lpips"])
        device: Device to run metrics on ("cuda" or "cpu")
        lpips_net_type: Network type for LPIPS metric ("alex", "vgg", or "squeeze")

    Returns:
        Configured MetricManager with registered metrics
    """
    manager = MetricManager(
        train_config_name="image_comparison_eval",
        mode="val",
        device=device,
    )

    for metric_name in metrics_to_compute:
        if metric_name not in IMAGE_COMPARISON_METRICS:
            raise ValueError(
                f"Unknown metric '{metric_name}'. Supported metrics: {list(IMAGE_COMPARISON_METRICS.keys())}"
            )

        metric_type = IMAGE_COMPARISON_METRICS[metric_name]

        # Create metric with appropriate parameters
        if metric_type == MetricType.PSNR:
            metric = MetricFactory[metric_type](
                data_range=1.0,
                device=device,
                aggregation_methods=[AggregationMethod.MEAN],
            )
        elif metric_type == MetricType.SSIM:
            metric = MetricFactory[metric_type](
                data_range=1.0,
                device=device,
                aggregation_methods=[AggregationMethod.MEAN],
            )
        elif metric_type == MetricType.LPIPS:
            metric = MetricFactory[metric_type](
                device=device,
                net_type=lpips_net_type,
                normalize=True,
                aggregation_methods=[AggregationMethod.MEAN],
            )
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")

        manager.register_metric(metric_name, metric)

    return manager


def compute_image_metrics(
    metric_manager: MetricManager,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute metrics that compare a single predicted image to a target image using MetricManager.

    Args:
        metric_manager: Configured MetricManager with registered metrics
        pred: Predicted float32 image tensor of shape (C, H, W) with values in [0, 1]
        target: Target float32 image tensor of shape (C, H, W) with values in [0, 1]
        mask: Optional boolean mask tensor of shape (H, W), True at pixels to include

    Returns:
        Dictionary of metric names to scalar values
    """
    # Add batch dimension for metrics framework: (C, H, W) -> (1, C, H, W)
    pred_batched = pred.unsqueeze(0)
    target_batched = target.unsqueeze(0)

    metrics_results: Dict[str, float] = {}

    for metric_name in metric_manager.list_metrics():
        metric_manager.compute(metric_name, pred_batched, target_batched, mask)

        last_result = metric_manager.get_last(metric_name)
        if last_result is not None:
            primary_value = last_result.values.get(metric_name, None)
            if primary_value is not None:
                metrics_results[metric_name] = float(primary_value.item())

    return metrics_results


class MetricsStore:
    """Stores a list of metric (KPI) values for multiple metrics and multiple image sequences"""

    @dataclass
    class Item:
        """A metric value entry for a specific image frame and category"""

        sequence_id: str  # Unique identifier of the image sequence within the clip
        frame_id: str  # Unique frame identifier within the image sequence (e.g. image file name)
        metric_name: str  # Name of the metric
        metric_value: float  # Value of the metric calculated for the frame
        semantic_class: Optional[str] = None  # Label of the image region used to calculate the metric value
        frame_timestamp_us: Optional[int] = None  # Timestamp of the frame in microseconds (typ. frame-end timestamp)

    def __init__(self) -> None:
        self._metrics: List[MetricsStore.Item] = []

    def add_metric(
        self,
        sequence_id: str,
        frame_id: str,
        metric_name: str,
        metric_value: float,
        semantic_class: Optional[str] = None,
        frame_timestamp_us: Optional[int] = None,
    ) -> None:
        """
        Add a metric value calculated for a specific image frame of a specific sequence of frames.

        Args:
            sequence_id: The ID of the image sequence (same as logical camera ID if available)
            frame_id: The unique frame identifier within a sequence
            metric_name: The name of the metric (e.g., "psnr")
            metric_value: The float value of the metric
            semantic_class: Label of the image region used to calculate the metric value (mask or semantic class name)
            frame_timestamp_us: The timestamp of the image frame in microseconds (typically frame-end timestamp)
        """
        self._metrics.append(
            MetricsStore.Item(
                sequence_id=sequence_id,
                frame_id=frame_id,
                frame_timestamp_us=frame_timestamp_us,
                metric_name=metric_name.lower().strip(),
                metric_value=metric_value,
                semantic_class=semantic_class,
            )
        )

    def to_columns(self) -> Dict[str, List[Any]]:
        """Convert the stored data to a columnar table format, column_name -> list of values per row.

        Metric values are listed by metric name in separate columns.
        Rows are not returned in any particular order.
        """
        # Initialize columns. Metric names to be returned in separate columns.
        # The hereby explicitly named fields are therefore shared by multiple metric values in the same row.
        metric_names = sorted(list(set([item.metric_name for item in self._metrics])))
        column_names = ["image_sequence_id", "frame_id", "frame_timestamp", "semantic_class"] + metric_names
        columns: Dict[str, List[Any]] = {column_name: [] for column_name in column_names}

        # Not every metric value is in a separate row, so collect metric values that should be in the same row.
        # Every unique tuple (sequence_id, frame_id, frame_timestamp_us, semantic_class) defines a row in the table.
        rows_dict: Dict[Tuple[str, str, Optional[int], Optional[str]], Dict[str, float]] = {}
        for item in self._metrics:
            row_id = (item.sequence_id, item.frame_id, item.frame_timestamp_us, item.semantic_class)
            rows_dict.setdefault(row_id, {})[item.metric_name] = item.metric_value

        # Cycle through rows and add a value to every column per row.
        for row_id, metrics in rows_dict.items():
            sequence_id, frame_id, frame_timestamp_us, semantic_class = row_id
            # Data shared by multiple metric values in the same row.
            columns["image_sequence_id"].append(sequence_id)
            columns["frame_id"].append(frame_id)
            columns["frame_timestamp"].append(frame_timestamp_us)
            columns["semantic_class"].append(semantic_class)
            # Visit every metric name (column) consistently, to fill in metric values for all metrics
            # (even if they are not present in the data for this row).
            for metric_name in metric_names:
                columns[metric_name].append(metrics.get(metric_name, None))

        return columns

    MetricsTree = Dict[Optional[str], Dict[str, Dict[str, List[float]]]]

    def to_tree(self) -> MetricsTree:
        """Organize metric values into a tree structure by semantic class, metric name, image sequence.

        Allows easy aggregation of values over frames first, metrics second and semantic classes last.

        Returns:
            Dictionary: semantic_class: Optional[str] -> sequence_id: str -> metric_name: str -> list of metric values: List[float]
        """

        metric_tree: MetricsStore.MetricsTree = {}

        for item in self._metrics:
            sequences = metric_tree.setdefault(item.semantic_class, {})
            metrics = sequences.setdefault(item.sequence_id, {})
            metrics.setdefault(item.metric_name, [])
            metrics[item.metric_name].append(item.metric_value)

        return metric_tree

    # Type for a dictionary of statistics calculated over a list of values (statistic name: str -> value: float).
    Statistics = Dict[str, float | int]

    # Type for a list of metrics aggregated over frames.
    # semantic class: Optional[str] -> sequence_id: str -> metric_name: str -> statistics: Statistics
    MetricStatisticsOverFrames = Dict[Optional[str], Dict[str, Dict[str, Statistics]]]

    @staticmethod
    def calculate_aggregated_metrics(metrics: MetricsTree) -> MetricStatisticsOverFrames:
        """Compute various metrics statistics over image frames (per semantic class and per metric)

        Args:
            metrics: Dictionary of per-frame metric values
              metric_name: str -> sequence_id: str -> list of metric values: List[float]

        Returns:
            Dictionary: semantic class: Optional[str] -> sequence_id: str -> metric_name: str -> statistic name: str -> value:float
        """

        stats_per_sequence: MetricsStore.MetricStatisticsOverFrames = {}

        for semantic_class, sequences in metrics.items():
            for sequence_id, metrics_by_name in sequences.items():
                for metric_name, metric_values in metrics_by_name.items():
                    stats = (
                        stats_per_sequence.setdefault(semantic_class, {})
                        .setdefault(sequence_id, {})
                        .setdefault(metric_name, {})
                    )
                    stats["avg"] = np.mean(np.array(metric_values)).item()
                    stats["min"] = np.min(np.array(metric_values)).item()
                    stats["max"] = np.max(np.array(metric_values)).item()
                    stats["std"] = np.std(np.array(metric_values)).item()
                    stats["cnt"] = len(metric_values)

        return stats_per_sequence

    def print(self) -> None:
        for item in self._metrics:
            print(
                f"{item.sequence_id} {item.frame_id} {item.frame_timestamp_us} {item.metric_name} {item.metric_value}"
            )

    def print_summary(self) -> None:
        """Log aggregated metrics to the terminal"""
        metrics_tree = self.to_tree()
        per_sequence_metrics = MetricsStore.calculate_aggregated_metrics(metrics_tree)

        for semantic_class, sequences in per_sequence_metrics.items():
            log.info(f"Aggregated metrics for '{semantic_class}':")
            for sequence_id, metrics in sequences.items():
                log.info(f"  {sequence_id}:")
                for metric_name, metric_statistics in metrics.items():
                    metric_value = metric_statistics["avg"]
                    log.info(f"    {metric_name}: {metric_value:.4f}")

    def save_yaml(self, file_path: str, statistics_only: bool = True) -> None:
        """Save the metrics to a YAML file"""

        per_frame_metrics = self.to_tree()
        per_sequence_metrics = MetricsStore.calculate_aggregated_metrics(per_frame_metrics)

        data_to_save: Dict[str, Any] = {}
        data_to_save["per_sequence_metrics"] = per_sequence_metrics
        if not statistics_only:
            data_to_save["per_frame_metrics"] = per_frame_metrics

        with open(file_path, "w") as f:
            yaml.dump(data_to_save, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Keep CSV writer for diagnostics even if parquet format is normally used for efficiency.
    def save_csv(self, file_path: str) -> None:
        """Save the metrics in a tabular format to a CSV file"""

        pd.DataFrame(self.to_columns()).to_csv(file_path, index=False)

    def save_parquet(self, file_path: str) -> None:
        """Save the metrics in a tabular format to a parquet file"""

        pd.DataFrame(self.to_columns()).to_parquet(file_path, index=False)


def compute_object_level_metrics(
    rendered_video_path: str,
    shard_pattern: str,
    camera_id: str,
    output_dir: str,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    max_frames: Optional[int] = None,
    device: str = "cuda",
    feature_batch_size: int = 32,
    crop_padding_ratio: float = 0.0,
    min_bbox_size: int = 50,
    gt_subsample_frames: int = 1,
    rendered_timestamps_file: Optional[str] = None,
    visualize: bool = False,
    viz_max_tracks: Optional[int] = None,
    viz_samples_per_track: int = 3,
    write_detailed: bool = True,
) -> None:
    """Compute object-level metrics using ObjectDataLoader and MetricManager.

    - ObjectDataLoader: handles all data loading (shard, video, GT frames, crops)
    - MetricManager: handles metric registration, computation, and aggregation

    Args:
        rendered_video_path: Path to rendered video file.
        shard_pattern: Glob pattern for shard files.
        camera_id: Camera ID to process.
        output_dir: Output directory for results.
        x_offset: Camera X offset in meters.
        y_offset: Camera Y offset in meters.
        z_offset: Camera Z offset in meters.
        max_frames: Maximum frames to process (None = all).
        device: Compute device ('cuda' or 'cpu').
        feature_batch_size: Batch size for feature extraction.
        crop_padding_ratio: Padding ratio around bounding boxes.
        min_bbox_size: Minimum bbox size in pixels.
        gt_subsample_frames: GT frame stride for alignment. Ignored if
            rendered_timestamps_file is provided.
        rendered_timestamps_file: Optional path to rendered video timestamps file.
        visualize: Whether to generate visualizations.
        viz_max_tracks: Maximum number of tracks to visualize.
        viz_samples_per_track: Number of frame samples per track.

    Returns:
        None. All metrics are saved via MetricManager.
    """
    log.info("=" * 80)
    log.info("Computing object-level metrics with MetricManager")
    log.info("  Video: %s", rendered_video_path)
    log.info("  Camera: %s", camera_id)
    log.info("  Device: %s", device)

    # Phase 1: Data Loading
    log.info("Phase 1: Loading data...")
    data_loader = ObjectDataLoader(
        shard_pattern=shard_pattern,
        camera_id=camera_id,
        device=device,
        crop_padding_ratio=crop_padding_ratio,
        min_bbox_size=min_bbox_size,
    )

    data_loader.load_video(
        rendered_video_path=rendered_video_path,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        max_frames=max_frames,
        gt_subsample_frames=gt_subsample_frames,
        rendered_timestamps_file=rendered_timestamps_file,
    )

    # Phase 2: MetricManager Setup
    log.info("Phase 2: Setting up MetricManager...")
    metric_manager = MetricManager(
        train_config_name="object_level_rendering_eval",
        mode="val",
        run_id=f"eval_{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        device=device,
    )

    # Register metrics using MetricFactory
    metric_manager.register_metric(
        "semantic",
        MetricFactory[MetricType.OBJECT_LEVEL_SEMANTIC](
            device=device,
            feature_batch_size=feature_batch_size,
            aggregation_methods=[
                AggregationMethod.MEAN,
            ],
        ),
    )
    metric_manager.register_metric(
        "perceptual",
        MetricFactory[MetricType.OBJECT_LEVEL_PERCEPTUAL](
            device=device,
            aggregation_methods=[
                AggregationMethod.MEAN,
            ],
        ),
    )

    log.info("  Registered metrics: %s", metric_manager.list_metrics())

    # Phase 3: Metric Computation
    log.info("Phase 3: Computing metrics...")

    # Counters for reporting
    num_frames_processed = 0
    num_crops_processed = 0

    for rendered_video_idx in tqdm.tqdm(range(data_loader.num_frames), desc="Processing frames"):
        # Get crops for this frame
        crops_data = data_loader.get_frame_crops_with_metadata(rendered_video_idx)

        if crops_data is None:
            log.debug("Frame %d: No crops data (likely no objects or frame not in range)", rendered_video_idx)
            continue

        if not crops_data["pred_crops"]:
            log.debug("Frame %d: No valid crops extracted", rendered_video_idx)
            continue

        num_frames_processed += 1
        num_crops_in_frame = len(crops_data["pred_crops"])
        num_crops_processed += num_crops_in_frame

        # Compute metrics for all crops in this frame (batch processing)
        pred_crops = crops_data["pred_crops"]  # List[np.ndarray], each [H, W, C]
        gt_crops = crops_data["gt_crops"]  # List[np.ndarray], each [H, W, C]

        # Format crops for batch semantic computation: List[List[np.ndarray]]
        # Each object has 1 crop (single-scale), wrapped in a list: [[crop1], [crop2], ...]
        # This format was created to support the semantic metric's multi-scale mode.
        pred_crops_multiscale = [[crop] for crop in pred_crops]
        gt_crops_multiscale = [[crop] for crop in gt_crops]

        # Compute semantic metric
        semantic_metric = metric_manager.get_metric("semantic")
        obj_metadata = ObjectMetadata(
            track_ids=crops_data["track_ids"],
            class_names=crops_data["class_names"],
            frame_idx=rendered_video_idx,
            gt_frame_idx=crops_data["camera_frame_idx"],
            bboxes_gt=crops_data["bboxes_gt"],
            bboxes_rendered=crops_data["bboxes_rendered"],
            rendered_timestamp=crops_data["rendered_timestamp"],
        )
        semantic_result = semantic_metric.compute(
            pred_crops_multiscale,
            gt_crops_multiscale,
            obj_metadata=obj_metadata,
        )
        semantic_metric.append(semantic_result)

        # Prepare crops: resize rendered crops to match GT sizes
        pred_crops_resized = []
        for pred_crop, gt_crop in zip(pred_crops, gt_crops):
            if pred_crop.shape != gt_crop.shape:
                pred_crop_resized = cv2.resize(
                    pred_crop, (gt_crop.shape[1], gt_crop.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            else:
                pred_crop_resized = pred_crop
            pred_crops_resized.append(pred_crop_resized)

        # Compute perceptual metric
        perceptual_metric = metric_manager.get_metric("perceptual")
        perceptual_result = perceptual_metric.compute(
            pred_crops_resized,  # List of crops (variable sizes)
            gt_crops,  # List of crops (variable sizes)
            obj_metadata=obj_metadata,
        )
        perceptual_metric.append(perceptual_result)

    log.info("  Processed %d frames, %d crops total", num_frames_processed, num_crops_processed)

    # Phase 4: Aggregation and Export
    log.info("Phase 4: Aggregating and saving metrics...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    aggregated_results = metric_manager.aggregate()

    try:
        metric_manager.write_metrics(output_dir, aggregate_metrics=True, ext="yaml")
        log.info("  Saved metrics: %s/metrics.yaml", output_dir)
    except RuntimeError:
        log.exception("Failed to write metrics YAML")

    # Save detailed metrics (per-frame, per-track, per-class)
    if write_detailed:
        try:
            write_detailed_metrics(output_dir, metric_manager, aggregated_results)
        except (OSError, TypeError, KeyError, ValueError) as e:
            log.warning("Failed to write detailed metrics YAML: %s", e)
    else:
        log.info("  Skipping detailed metrics (--no-detailed-metrics)")

    # Extract minimal values for console logging only
    semantic_mean = 0.0
    perceptual_mean = 0.0
    num_tracks = 0

    if "semantic" in aggregated_results:
        if AggregationMethod.MEAN in aggregated_results["semantic"]:
            semantic_mean = float(
                aggregated_results["semantic"][AggregationMethod.MEAN].values.get("object_level_semantic", 0.0)
            )
            # Per-track data
            per_track_data = aggregated_results["semantic"][AggregationMethod.MEAN].metadata.get("per_track", {})
            num_tracks = len(per_track_data)

    if "perceptual" in aggregated_results:
        if AggregationMethod.MEAN in aggregated_results["perceptual"]:
            perceptual_mean = float(
                aggregated_results["perceptual"][AggregationMethod.MEAN].values.get("object_level_perceptual", 0.0)
            )

    # Phase 5: Visualization (Optional)
    if visualize:
        log.info("Phase 5: Generating visualizations...")
        viz_output_dir = os.path.join(output_dir, "visualizations")

        # Create minimal metadata needed for visualization (video/shard paths)
        viz_metadata = {
            "rendered_video": rendered_video_path,
            "shard_pattern": shard_pattern,
            "camera_id": camera_id,
            "gt_subsample_frames": gt_subsample_frames,
        }

        # Visualization extracts all metric data internally from MetricManager
        visualize_tracked_objects(
            output_dir=viz_output_dir,
            metric_manager=metric_manager,
            metadata=viz_metadata,
            aggregated_results=aggregated_results,
            max_tracks=viz_max_tracks,
            samples_per_track=viz_samples_per_track,
        )

    # Cleanup resources
    data_loader.close()

    # Summary logging
    log.info("=" * 80)
    log.info("Object-level metrics computation complete")
    log.info("  Summary:")
    log.info("    Semantic similarity (mean): %.4f", semantic_mean)
    log.info("    Perceptual score (mean): %.4f", perceptual_mean)
    log.info("    Total tracks: %d", num_tracks)
    log.info("    Total crops: %d", num_crops_processed)
    log.info("  Results saved to: %s", output_dir)


@click.command("eval-rendering-metrics")
@click.option(
    "--render-dir",
    type=str,
    help="Path to the root directory of rendered images or videos. "
    "For image-level metrics: <render-dir>/<image-seq>/<frame-timestamp>.<ext>. "
    "For object-level metrics: <render-dir>/<image-seq>/video.mp4 (or .avi, .mov). "
    "Where <image-seq> is typically the camera name, e.g. camera_front_wide_120fov",
    required=True,
)
@click.option(
    "--gt-dir",
    type=str,
    help="Path to the directory containing the ground-truth data required for evaluating rendering. "
    "Image sequences are expected as <gt-dir>/camera_images/<image-seq>/<frame-timestamp>.<ext>, "
    "per-image-sequence masks as <gt-dir>/camera_ego_masks/<image-seq>.png. "
    "The input masks are expected to be zero where pixels are valid. "
    "Required for image-level metrics, ignored when using --object-level-metrics, but this argument must be always provided. "
    "If you are using object-level metrics, you can provide a dummy directory here.",
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to a (preferably new) output directory to export benchmark results to",
    required=True,
)
@click.option(
    "--rendered-image-extension",
    type=click.Choice(["png", "PNG", "jpg", "JPG", "jpeg", "JPEG"]),
    help="Image extension used for the rendered images",
    default="png",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Export a frame illustrating the difference between each pair of rendered and ground-truth image",
    default=False,
)
@click.option(
    "--visualization-multiplier",
    type=float,
    help="Factor by which the difference image is multiplied to make small differences more visible consistently"
    "over video frames (only used if --visualize is set)",
    default=5.0,
)
@click.option(
    "--visualization-normalized",
    is_flag=True,
    help="Discard --visualization-multiplier and normalize the difference image by the maximum difference per frame "
    "instead (only used if --visualize is set). The use of this feature is not recommended because it varies from "
    "frame to frame and has little to no enhancement effect when a few sporadic pixels have a high error (likely).",
    default=False,
)
@click.option(
    "--visualization-jpeg-quality",
    type=int,
    help="JPEG quality parameter for the output frames generated for error visualization/video.",
    default=95,
)
@click.option(
    "--max-frames",
    type=int,
    help="Maximum number of frames per image sequence to process, e.g. for quick-testing. No limit if not specified.",
    default=None,
)
@click.option(
    "--metrics",
    type=click.Choice(["psnr", "ssim", "lpips"], case_sensitive=False),
    multiple=True,
    help="Metrics to compute for image-level evaluation. Can be specified multiple times. "
    "Supported metrics: psnr, ssim, lpips. Default: psnr",
    default=("psnr",),
)
@click.option(
    "--lpips-net-type",
    type=click.Choice(["alex", "vgg", "squeeze"]),
    help="Network type for LPIPS metric. 'alex' is fastest, 'vgg' is most accurate.",
    default="alex",
)
@click.option(
    "--object-level-metrics",
    is_flag=True,
    help="Compute object-level metrics (requires shard data)",
    default=False,
)
@click.option(
    "--shard-pattern",
    type=str,
    help="Glob pattern for shard files (required if --object-level-metrics is set)",
    default=None,
)
@click.option(
    "--camera-offset",
    type=float,
    nargs=3,
    help="Camera offset in meters [x, y, z] for object-level metrics",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--camera-offset-json",
    type=click.Path(exists=True),
    help="NDAS offset JSON file with 6-DOF offsets (tx_m, ty_m, tz_m). Overrides --camera-offset if provided.",
    default=None,
)
@click.option(
    "--object-feature-batch-size",
    type=int,
    help="Batch size for object-level feature extraction",
    default=32,
)
@click.option(
    "--object-crop-padding",
    type=float,
    help="Padding ratio around bounding boxes for object crops",
    default=0.0,
)
@click.option(
    "--object-min-bbox-size",
    type=int,
    help="Minimum bounding box size in pixels for object-level metrics",
    default=50,
)
@click.option(
    "--gt-subsample-frames",
    type=int,
    help="GT frame stride for alignment with downsampled rendered video. Ignored if --rendered-timestamps is provided.",
    default=1,
)
@click.option(
    "--rendered-timestamps",
    type=click.Path(exists=True),
    help="Timestamps file for rendered video. Matches rendered frames to GT by timestamp instead of --gt-subsample-frames.",
    default=None,
)
@click.option(
    "--visualize-objects",
    is_flag=True,
    help="Generate visualizations for object-level metrics",
    default=False,
)
@click.option(
    "--viz-max-tracks",
    type=int,
    help="Maximum number of tracks to visualize (None = all)",
    default=None,
)
@click.option(
    "--viz-samples-per-track",
    type=int,
    help="Number of frame samples to show per track in visualizations",
    default=3,
)
@click.option(
    "--no-detailed-metrics",
    is_flag=True,
    help="Skip saving metrics_detailed.yaml to reduce storage requirements.",
    default=False,
)
@click.option(
    "--save-parquet/--no-save-parquet",
    help="Save per-frame rendering metrics to parquet format",
    default=False,
)
@click.option(
    "--save-csv/--no-save-csv",
    help="Save per-frame rendering metrics to CSV format",
    default=False,
)
@click.option(
    "--save-yaml/--no-save-yaml",
    help="Save rendering metrics to YAML format (aggregate metrics only)",
    default=False,
)
def eval_rendering_metrics(
    render_dir: str,
    gt_dir: str,
    output_dir: str,
    rendered_image_extension: str,
    visualize: bool,
    visualization_multiplier: float,
    visualization_normalized: bool,
    visualization_jpeg_quality: int,
    max_frames: Optional[int],
    metrics: Tuple[str, ...],
    lpips_net_type: str,
    object_level_metrics: bool,
    shard_pattern: Optional[str],
    camera_offset: Tuple[float, float, float],
    camera_offset_json: Optional[str],
    object_feature_batch_size: int,
    object_crop_padding: float,
    object_min_bbox_size: int,
    gt_subsample_frames: int,
    rendered_timestamps: Optional[str],
    visualize_objects: bool,
    viz_max_tracks: Optional[int],
    viz_samples_per_track: int,
    no_detailed_metrics: bool,
    save_parquet: bool,
    save_csv: bool,
    save_yaml: bool,
) -> None:
    """Evaluates rendering metrics on a directory of rendered images/w.r.t. to a directory of ground-truth images
    for multiple image sequences (typically one image sequence per camera).

    Rendered images and corresponding ground-truth images are assumed to have the same filenames.
    Absolute timestamps should be used for the input file names, because sensors are not necessarily synchronized,
    and matching rendered and ground-truth image pairs must be perfectly synchronous.

    Ground-truth images are on-the-fly resized to match the resolution of the rendered images.
    GT images can therefore be exported at their original resolution once, and sequences rendered at
    different resolutions can be evaluated without having to re-export the GT at each resolution.

    When --object-level-metrics is enabled, evaluates per-object semantic
    and perceptual quality using 3D bounding boxes from shard data and
    rendered video frames, bypassing GT image directory requirements.
    """

    if not os.path.isdir(render_dir):
        raise ValueError(f"Missing directory {render_dir}")

    if visualization_multiplier is not None and visualization_multiplier <= 0.0:
        raise ValueError("--visualization-multiplier must be positive or None")

    # Validate gt_dir only if doing image-level metrics
    if not object_level_metrics and not os.path.isdir(gt_dir):
        raise ValueError(f"Missing directory {gt_dir}")

    rendered_seq_ids = [d for d in os.listdir(render_dir) if os.path.isdir(os.path.join(render_dir, d))]
    log.info("Found %d rendered sequences: %s", len(rendered_seq_ids), rendered_seq_ids)

    metrics_store = MetricsStore()

    # TODO(DESIGN): Metric computation strategy
    # Current: Metrics with similar inputs can be computed together (e.g. PSNR; SSIM); metrics with different
    #          input types require command-line flags to select which type to compute (e.g. object-level metrics).
    # Future: Auto-detect and compute all metrics supported by the provided inputs
    #         (e.g., if both gt_dir or rendered_dir (containing video/images) and shard_pattern (containing shard files)
    #         can be provided, compute both image-level and object-level metrics automatically).

    # Two separate paths: image-level OR object-level metrics
    if not object_level_metrics:
        # Image-level metrics only (default, backward compatible)
        # Create MetricManager for image-level metrics
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_metric_manager: MetricManager = create_image_comparison_metric_manager(
            metrics_to_compute=list(metrics),
            device=device,
            lpips_net_type=lpips_net_type,
        )
        log.info("Computing image comparison metrics: %s", " ".join(image_metric_manager.list_metrics()))

        for rendered_seq_id in rendered_seq_ids:
            rendered_seq_dir = os.path.join(render_dir, rendered_seq_id)
            rendered_image_files = [
                f
                for f in os.listdir(rendered_seq_dir)
                if os.path.isfile(os.path.join(rendered_seq_dir, f)) and f.endswith("." + rendered_image_extension)
            ]
            rendered_image_files.sort()

            if len(rendered_image_files) == 0:
                raise ValueError(f"No rendered images found in {rendered_seq_dir}")

            # Load the mapping from image names to frame timestamps from JSON when available.
            timestamps_json = os.path.join(rendered_seq_dir, "timestamps.json")
            frame_timestamps: Optional[Dict[str, int]] = None
            if os.path.isfile(timestamps_json):
                log.info("Loading timestamps from %s", timestamps_json)
                frame_timestamps = load_timestamps_json(timestamps_json)

            if max_frames is not None and max_frames < len(rendered_image_files):
                log.warning(
                    "Only evaluating %d of %d frames due to --max-frames", max_frames, len(rendered_image_files)
                )
                rendered_image_files = rendered_image_files[:max_frames]

            log.info("Creating directory %s", output_dir)
            seq_output_dir = os.path.join(output_dir, rendered_seq_id)
            os.makedirs(seq_output_dir, exist_ok=True)

            # Load the camera mask. The stored mask is expected to have zeros where pixels are valid.
            camera_mask_path = os.path.join(gt_dir, "camera_ego_masks", rendered_seq_id + ".png")
            camera_mask: Optional[torch.Tensor] = None
            if os.path.isfile(camera_mask_path):
                log.info("Loading camera mask %s", camera_mask_path)
                camera_mask = load_mask_to_tensor(camera_mask_path)
            else:
                raise ValueError(f"Missing camera mask file {camera_mask_path}")

            gt_image_dir = os.path.join(gt_dir, "camera_images", rendered_seq_id)
            if os.path.isdir(gt_image_dir):
                log.info("Found ground-truth image directory %s", gt_image_dir)
            else:
                raise ValueError(f"Missing ground-truth image directory {gt_image_dir}")

            # Resize operations are cached to avoid re-creating them for each image.
            resize_image: Optional[transforms.Resize] = None
            resize_mask: Optional[transforms.Resize] = None

            for rendered_image_file in tqdm.tqdm(
                rendered_image_files, desc=f"Evaluating rendered images from {rendered_seq_id}"
            ):
                rendered_image_path = os.path.join(rendered_seq_dir, rendered_image_file)
                rendered_image = load_image_to_tensor(rendered_image_path)
                _, rendered_height, rendered_width = rendered_image.size()

                image_name = os.path.splitext(rendered_image_file)[0]

                gt_image_path = find_image(os.path.join(gt_image_dir, image_name))
                gt_image = load_image_to_tensor(gt_image_path)
                if gt_image.size() != rendered_image.size():
                    # Ground-truth images are on-the-fly resized to match the resolution of the rendered images.
                    # GT images can therefore be exported at their original resolution once, and sequences rendered at
                    # different resolutions can be evaluated without having to re-export the GT at each resolution.
                    if resize_image is None:
                        resize_image = transforms.Resize(
                            (rendered_height, rendered_width),
                            transforms.InterpolationMode.BILINEAR,
                            max_size=None,
                            antialias=True,
                        )
                    assert resize_image is not None  # Type narrowing for static analyzer
                    gt_image = resize_image(gt_image)

                if camera_mask is not None and (camera_mask.size() != rendered_image.size()[1:]):
                    # Resize the camera mask to match the resolution of the rendered images.
                    # Camera masks can therefore be exported at their original resolution once, and sequences rendered at
                    # different resolutions can be evaluated without having to re-export the masks at each resolution.
                    if resize_mask is None:
                        resize_mask = transforms.Resize(
                            (rendered_height, rendered_width),
                            transforms.InterpolationMode.NEAREST,
                            max_size=None,
                            antialias=True,
                        )
                    assert resize_mask is not None  # Type narrowing for static analyzer
                    camera_mask = resize_mask(camera_mask.unsqueeze(0)).squeeze(0)
                    # transform.Resize returns bool when fed with bool if interpolation method is NEAREST.
                    # mypy complains that camera mask may be None, even though it may not at this point.
                    assert camera_mask is not None and camera_mask.dtype == torch.bool

                # Invert the mask because per-pixel evaluation requires the camera mask to
                # have zeros where pixels are invalid (non-zeros select pixels to be evaluated).
                inv_camera_mask = ~camera_mask if camera_mask is not None else None

                # Use MetricManager-based computation
                computed_metrics = compute_image_metrics(
                    metric_manager=image_metric_manager,
                    pred=rendered_image,
                    target=gt_image,
                    mask=inv_camera_mask,
                )

                # Use frame timestamps loaded from file, if available,
                # otherwise assume the image name is a timestamp.
                frame_timestamp_us: Optional[int] = None
                if frame_timestamps is not None:
                    try:
                        frame_timestamp_us = frame_timestamps[image_name]
                    except KeyError:
                        raise KeyError(f"Image {image_name} missing from {rendered_seq_id}/timestamps.json") from None
                else:
                    frame_timestamp_us = int(image_name)

                # Add each computed metric to the metrics store
                for metric_name, metric_value in computed_metrics.items():
                    metrics_store.add_metric(
                        sequence_id=rendered_seq_id,
                        frame_id=image_name,
                        frame_timestamp_us=frame_timestamp_us,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        semantic_class="static_camera_mask",  # No semantic classes yet, only camera mask.
                    )

                if visualize:
                    comp_image = create_comparison_image(
                        rendered_image,
                        gt_image,
                        camera_mask,
                        metrics=computed_metrics,
                        diff_factor=visualization_multiplier if not visualization_normalized else None,
                    )
                    comp_image.save(
                        os.path.join(seq_output_dir, image_name + ".jpg"), quality=visualization_jpeg_quality
                    )

    # Object-level metrics computation
    if object_level_metrics:
        if shard_pattern is None:
            raise ValueError("--shard-pattern is required when --object-level-metrics is set")

        log.info("=" * 80)
        log.info("Computing object-level metrics with MetricManager...")

        for rendered_seq_id in rendered_seq_ids:
            log.info("Processing object-level metrics for sequence: %s", rendered_seq_id)

            # Construct paths
            rendered_seq_dir = os.path.join(render_dir, rendered_seq_id)

            # Find first video file in directory
            video_files = [f for f in os.listdir(rendered_seq_dir) if f.endswith((".mp4", ".avi", ".mov"))]

            if len(video_files) > 0:
                # Use existing video file
                rendered_video_path = os.path.join(rendered_seq_dir, video_files[0])
                log.info("Using video file: %s", rendered_video_path)
            else:
                # Note: For image sequences, users should convert to video first
                log.warning(
                    "No video file found in %s. Object-level metrics require video input. Skipping.", rendered_seq_dir
                )
                continue

            try:
                # Get camera offset from JSON file or CLI argument
                if camera_offset_json:
                    x_offset, y_offset, z_offset = load_camera_offset_json(camera_offset_json)
                    log.info("Using camera offset from JSON: x=%.3f, y=%.3f, z=%.3f", x_offset, y_offset, z_offset)
                else:
                    x_offset, y_offset, z_offset = camera_offset

                # Compute metrics using new MetricManager-based implementation
                compute_object_level_metrics(
                    rendered_video_path=rendered_video_path,
                    shard_pattern=shard_pattern,
                    camera_id=rendered_seq_id,  # Assume seq_id matches camera_id
                    output_dir=os.path.join(output_dir, rendered_seq_id),
                    x_offset=x_offset,
                    y_offset=y_offset,
                    z_offset=z_offset,
                    max_frames=max_frames,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    feature_batch_size=object_feature_batch_size,
                    crop_padding_ratio=object_crop_padding,
                    min_bbox_size=object_min_bbox_size,
                    gt_subsample_frames=gt_subsample_frames,
                    rendered_timestamps_file=rendered_timestamps,
                    visualize=visualize_objects,
                    viz_max_tracks=viz_max_tracks,
                    viz_samples_per_track=viz_samples_per_track,
                    write_detailed=not no_detailed_metrics,
                )

            except (RuntimeError, ValueError, IndexError) as e:
                log.error("Failed to compute object-level metrics for %s: %s", rendered_seq_id, e)
                continue

        log.info("Object-level metrics computation complete")

    # Print summary of image-level metrics (PSNR/SSIM/LPIPS)
    # Note: Object-level metrics are saved per-video via MetricManager
    metrics_store.print_summary()

    if save_yaml:
        # Save image-level rendering metrics to YAML
        os.makedirs(output_dir, exist_ok=True)
        metrics_yaml_path = os.path.join(output_dir, "rendering_metrics.yaml")
        log.info("Saving metrics to %s", metrics_yaml_path)
        metrics_store.save_yaml(metrics_yaml_path)

    if save_csv:
        # Save image-level rendering metrics to CSV (useful for Kratos upload)
        os.makedirs(output_dir, exist_ok=True)
        metrics_csv_path = os.path.join(output_dir, "rendering_metrics.csv")
        log.info("Saving metrics to %s", metrics_csv_path)
        metrics_store.save_csv(metrics_csv_path)

    if save_parquet:
        # Save image-level rendering metrics to parquet tabular format, recommended for efficient use in the cluster
        os.makedirs(output_dir, exist_ok=True)
        metrics_parquet_path = os.path.join(output_dir, "rendering_metrics.parquet")
        log.info("Saving metrics to %s", metrics_parquet_path)
        metrics_store.save_parquet(metrics_parquet_path)
