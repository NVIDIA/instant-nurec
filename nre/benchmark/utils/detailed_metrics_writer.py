# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility for saving detailed per-frame metrics to YAML."""

import logging
import os

from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from nre.metrics.metric import MetricManager
from nre.metrics.utils import AggregationMethod


log = logging.getLogger(__name__)


def _convert_to_python_type(value: Any) -> Any:
    """Convert numpy/torch types to Python native types for YAML serialization.

    Args:
        value: Value to convert (can be numpy, torch, dict, list, or native).

    Returns:
        Python native type suitable for YAML serialization.
    """
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().tolist()
    if isinstance(value, dict):
        return {k: _convert_to_python_type(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_to_python_type(v) for v in value]
    return value


def write_detailed_metrics(
    output_dir: str,
    metric_manager: MetricManager,
    aggregated_results: Dict[str, Dict],
    filename: str = "metrics_detailed.yaml",
) -> None:
    """Save detailed per-frame, per-track metrics to YAML file.

    Extracts and saves the same data used for visualization, enabling
    offline analysis without regenerating visualizations.

    Args:
        output_dir: Directory to save the YAML file.
        metric_manager: MetricManager with computed metrics.
        aggregated_results: Pre-computed aggregated results from metric_manager.
        filename: Output filename (default: metrics_detailed.yaml).

    Output YAML structure:
        detailed_by_track:
            <track_id>:
                - frame_idx: 0
                  semantic_raw: 0.85
                  semantic_adjusted: 0.72
                  perceptual_score: 0.91
                  ssim: 0.88
                  ...
                - frame_idx: 1
                  ...
        per_track_summary:
            <track_id>:
                semantic_mean: 0.83
                semantic_std: 0.05
                perceptual_mean: 0.89
                ...
        per_class_summary:
            Vehicle:
                semantic_mean: 0.81
                num_tracks: 15
                ...
    """
    # Build detailed_by_track from raw metric values
    detailed_by_track: Dict[str, List[Dict[str, Any]]] = {}

    # Extract semantic data
    if metric_manager.has_metric("semantic"):
        semantic_metric = metric_manager.get_metric("semantic")
        for value in semantic_metric.values():
            for obj_data in value.metadata.get("detailed_data", []):
                track_id = obj_data.get("track_id")
                if track_id:
                    if track_id not in detailed_by_track:
                        detailed_by_track[track_id] = []
                    clean_data = {k: _convert_to_python_type(v) for k, v in obj_data.items()}
                    detailed_by_track[track_id].append(clean_data)

    # Merge perceptual data
    if metric_manager.has_metric("perceptual"):
        perceptual_metric = metric_manager.get_metric("perceptual")
        for value in perceptual_metric.values():
            for obj_data in value.metadata.get("detailed_data", []):
                track_id = obj_data.get("track_id")
                if track_id and track_id in detailed_by_track:
                    frame_idx = obj_data.get("frame_idx")
                    for existing in detailed_by_track[track_id]:
                        if existing.get("frame_idx") == frame_idx:
                            for k, v in obj_data.items():
                                if k not in existing:
                                    existing[k] = _convert_to_python_type(v)
                            break

    # Extract per-track and per-class summaries from aggregated results
    per_track_summary: Dict[str, Dict[str, Any]] = {}
    per_class_summary: Dict[str, Dict[str, Any]] = {}

    for metric_name in ["semantic", "perceptual"]:
        if metric_name not in aggregated_results:
            continue
        if AggregationMethod.MEAN not in aggregated_results[metric_name]:
            continue

        agg_meta = aggregated_results[metric_name][AggregationMethod.MEAN].metadata

        # Per-track
        for tid, metrics in agg_meta.get("per_track", {}).items():
            if tid not in per_track_summary:
                per_track_summary[tid] = {}
            per_track_summary[tid].update({k: _convert_to_python_type(v) for k, v in metrics.items()})

        # Per-class
        for cls, metrics in agg_meta.get("per_class", {}).items():
            if cls not in per_class_summary:
                per_class_summary[cls] = {}
            per_class_summary[cls].update({k: _convert_to_python_type(v) for k, v in metrics.items()})

    detailed_output = {
        "detailed_by_track": detailed_by_track,
        "per_track_summary": per_track_summary,
        "per_class_summary": per_class_summary,
    }

    detailed_path = os.path.join(output_dir, filename)
    with open(detailed_path, "w", encoding="utf-8") as f:
        yaml.dump(
            detailed_output,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    log.info("Saved detailed metrics: %s", detailed_path)
