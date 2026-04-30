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

import json
import logging

from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
import torch

from PIL import Image

from nre.metrics import MetricFactory, MetricType


def load_image_to_tensor(image_path: str) -> torch.Tensor:
    """Load an image to a tensor."""
    try:
        img: Image.Image = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return tensor
    except (IOError, FileNotFoundError) as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")


def load_video_to_tensor(
    video_path: str,
    max_frames: int | None = None,
    frame_skip: int = 1,
    target_width: int = 224,
    target_height: int = 224,
    device: str = "cpu",
) -> torch.Tensor:
    """Load frames from a video file and convert to tensor.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None = all)
        frame_skip: Skip every N frames (1 = process all frames)
        target_width: Target width for output frames
        target_height: Target height for output frames
        device: Device to place tensor on

    Returns:
        Tensor of frames with shape [N, C, H, W]
    """
    logging.info(f"Loading frames from: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logging.info(
        f"  Video info: {total_frames} frames @ {fps:.1f} FPS, "
        f"{original_width}x{original_height} -> {target_width}x{target_height}"
    )

    frames = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on frame_skip
        if frame_count % frame_skip == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to target dimensions
            if (target_width, target_height) != (original_width, original_height):
                frame_rgb = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # Convert to tensor directly
            processed_frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            frames.append(processed_frame)

            extracted_count += 1

            # Check max_frames limit
            if max_frames is not None and extracted_count >= max_frames:
                break

        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    logging.info(f"  Extracted {extracted_count} frames from {total_frames} total frames")

    # Stack frames and move to device
    video_tensor = torch.stack(frames).to(device)
    return video_tensor


@click.command("compute-metrics")
@click.option("--pred-path", type=str, required=True, help="Path to prediction video/image")
@click.option("--target-path", type=str, required=True, help="Path to target/ground truth video/image")
@click.option("--output-path", type=str, required=True, help="Path to output JSON file")
@click.option(
    "--metric",
    type=str,
    default="psnr",
    help="Metric to compute (psnr, ssim, lpips, cpsnr, fid, drift, fcs_adaptive, ntd, d_skew, d_kurt, perceptual, temporal_coherence)",
)
@click.option("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
@click.option("--cache-dir", type=str, default=None, help="Directory to cache pretrained models")
@click.option("--max-frames", type=int, default=None, help="Maximum frames to process (None = all)")
@click.option("--frame-skip", type=int, default=1, help="Skip every N frames (1 = process all)")
@click.option("--feature-batch-size", type=int, default=4, help="Batch size for feature extraction")
@click.option(
    "--lpips-net-type",
    type=click.Choice(["alex", "vgg", "squeeze"]),
    default="alex",
    help="Network type for LPIPS metric (alex, vgg, squeeze)",
)
def compute_metrics(
    pred_path: str,
    target_path: str,
    output_path: str,
    metric: str,
    device: str,
    cache_dir: str | None,
    max_frames: int | None,
    frame_skip: int,
    feature_batch_size: int,
    lpips_net_type: str,
) -> None:
    """Compute metrics for a prediction and ground truth."""

    output_json: dict[str, Any] = {}

    # Create output directory if it doesn't exist
    output_path_file = Path(output_path)
    output_path_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Computing metrics for {pred_path} and {target_path} and saving to {output_path_file}")

    # Handle different metrics
    match metric:
        case "psnr":
            metric_instance = MetricFactory[MetricType.PSNR](data_range=1.0)

            # Load the prediction and ground truth which are expected to be images
            prediction = load_image_to_tensor(pred_path)
            ground_truth = load_image_to_tensor(target_path)

            result = metric_instance.compute(prediction, ground_truth)

            output_json["prediction_path"] = pred_path
            output_json["ground_truth_path"] = target_path

            output_json["metric"] = "psnr"
            output_json["value"] = result.get_value("psnr").item()
            output_json["metadata"] = result.metadata

        case "ssim":
            metric_instance = MetricFactory[MetricType.SSIM](data_range=1.0)

            # Load the prediction and ground truth which are expected to be images
            prediction = load_image_to_tensor(pred_path)
            ground_truth = load_image_to_tensor(target_path)

            result = metric_instance.compute(prediction, ground_truth)

            output_json["prediction_path"] = pred_path
            output_json["ground_truth_path"] = target_path

            output_json["metric"] = "ssim"
            output_json["value"] = result.get_value("ssim").item()
            output_json["metadata"] = result.metadata

        case "lpips":
            # Set up device for LPIPS (uses neural network)
            if device == "cuda" and not torch.cuda.is_available():
                logging.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            metric_instance = MetricFactory[MetricType.LPIPS](device=device, net_type=lpips_net_type, normalize=True)

            # Load the prediction and ground truth which are expected to be images
            prediction = load_image_to_tensor(pred_path).to(device)
            ground_truth = load_image_to_tensor(target_path).to(device)

            result = metric_instance.compute(prediction, ground_truth)

            output_json["prediction_path"] = pred_path
            output_json["ground_truth_path"] = target_path

            output_json["metric"] = "lpips"
            output_json["value"] = result.get_value("lpips").item()
            output_json["metadata"] = result.metadata

        case "cpsnr":
            metric_instance = MetricFactory[MetricType.CPSNR](data_range=1.0)

            # TODO: Add CPSNR computation
            raise NotImplementedError("CPSNR computation is not implemented yet")

        case "fid" | "drift" | "fcs_adaptive" | "ntd" | "d_skew" | "d_kurt" | "perceptual" | "temporal_coherence":
            # Set up device
            if device == "cuda" and not torch.cuda.is_available():
                logging.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            logging.info(f"Using device: {device}")

            # Define metric configurations
            METRIC_CONFIGS = {
                "fid": (MetricType.FID, "fid"),
                "drift": (MetricType.FEATURE_DRIFT, "feature_drift"),
                "fcs_adaptive": (MetricType.FCS_ADAPTIVE, "fcs_adaptive"),
                "ntd": (MetricType.NTD, "ntd_distance"),
                "d_skew": (MetricType.D_SKEW, "d_skew"),
                "d_kurt": (MetricType.D_KURT, "d_kurt"),
                "perceptual": (MetricType.PERCEPTUAL, "perceptual_distance"),
                "temporal_coherence": (MetricType.TEMPORAL_COHERENCE, "temporal_coherence_ratio"),
            }

            if metric in METRIC_CONFIGS:
                metric_type, metric_key = METRIC_CONFIGS[metric]
                metric_instance = MetricFactory[metric_type](
                    device=device,
                    extractor_type="segformer",
                    pretrained_path="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                    cache_dir=cache_dir,
                    feature_batch_size=feature_batch_size,
                )
            else:
                raise ValueError(f"Unexpected metric type: {metric}")

            # Load video frames
            logging.info("Loading video frames...")
            prediction = load_video_to_tensor(pred_path, max_frames=max_frames, frame_skip=frame_skip, device=device)
            ground_truth = load_video_to_tensor(
                target_path, max_frames=max_frames, frame_skip=frame_skip, device=device
            )

            # Match frame counts
            min_frames = min(len(prediction), len(ground_truth))
            if len(prediction) != len(ground_truth):
                logging.info(f"Matching frame counts: Pred={len(prediction)}, GT={len(ground_truth)} -> {min_frames}")
                prediction = prediction[:min_frames]
                ground_truth = ground_truth[:min_frames]

            logging.info(f"Processing {min_frames} matched frames")

            # Compute metric
            result = metric_instance.compute(prediction, ground_truth)

            # Extract all metric values
            metric_values = {}
            for key, value in result.values.items():
                if isinstance(value, torch.Tensor):
                    metric_values[key] = value.item()
                else:
                    metric_values[key] = float(value)

            output_json["prediction_path"] = pred_path
            output_json["ground_truth_path"] = target_path
            output_json["metric"] = metric
            output_json["primary_value"] = metric_values.get(metric_key, list(metric_values.values())[0])
            output_json["all_values"] = metric_values
            output_json["metadata"] = result.metadata
            output_json["processing_info"] = {
                "device": device,
                "frames_processed": min_frames,
                "max_frames": max_frames,
                "frame_skip": frame_skip,
                "feature_batch_size": feature_batch_size,
            }

        case _:
            raise ValueError(f"Invalid metric: {metric}")

    # Output the JSON to the output path
    with open(output_path_file, "w") as f:
        json.dump(output_json, f, indent=4)

    logging.info(f"Metrics computed and saved to {output_path_file}")
