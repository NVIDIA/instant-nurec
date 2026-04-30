# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import os

from pathlib import Path

import numpy as np
import pytest

from click.testing import CliRunner
from PIL import Image

from apps.aux_gen.ncore_aux_data import cli


def calculate_iou(mask1, mask2):
    """
    Calculate IoU between two binary masks
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 1.0
    return iou


def compare_segmentation_outputs(trt_dir, torch_dir):
    """
    Compare segmentation outputs from TensorRT and PyTorch models
    """
    # Get all segmentation images from both directories
    trt_files = sorted(glob.glob(os.path.join(trt_dir, "*.jpg")))
    torch_files = sorted(glob.glob(os.path.join(torch_dir, "*.jpg")))

    if len(trt_files) == 0 or len(torch_files) == 0:
        print(f"No images found. TRT files: {len(trt_files)}, PyTorch files: {len(torch_files)}")
        return None, None

    # Verify we have the same number of files
    if len(trt_files) != len(torch_files):
        print(f"Warning: Different number of files. TRT: {len(trt_files)}, PyTorch: {len(torch_files)}")

    # Compare matching files
    iou_scores = []
    max_diffs = []

    for trt_file, torch_file in zip(trt_files, torch_files):
        # Extract just the filename for comparison
        trt_name = os.path.basename(trt_file)
        torch_name = os.path.basename(torch_file)

        # Load images
        trt_img = np.array(Image.open(trt_file))
        torch_img = np.array(Image.open(torch_file))

        # Calculate maximum difference
        max_diff = np.abs(trt_img.astype(np.float32) - torch_img.astype(np.float32)).max()
        max_diffs.append(max_diff)

        # Convert to binary masks for IoU calculation (for each class)
        if len(trt_img.shape) == 3 and trt_img.shape[2] == 3:  # RGB image
            # Convert to grayscale
            trt_gray = np.mean(trt_img, axis=2).astype(np.uint8)
            torch_gray = np.mean(torch_img, axis=2).astype(np.uint8)

            # Get unique classes
            unique_classes = np.unique(np.concatenate([trt_gray.flatten(), torch_gray.flatten()]))

            # Calculate IoU for each class
            class_ious = []
            for cls in unique_classes:
                if cls == 0:  # Skip background
                    continue
                trt_mask = trt_gray == cls
                torch_mask = torch_gray == cls
                iou = calculate_iou(trt_mask, torch_mask)
                class_ious.append(iou)

            if class_ious:
                iou_scores.append(np.mean(class_ious))
        else:
            # Binary comparison
            trt_mask = trt_img > 0
            torch_mask = torch_img > 0
            iou = calculate_iou(trt_mask, torch_mask)
            iou_scores.append(iou)

    # Print results
    if iou_scores:
        mean_iou = np.mean(iou_scores)
        mean_max_diff = np.mean(max_diffs)
        print("\nResults:")
        print(f"Average IoU: {mean_iou:.4f}")
        print(f"Average Max Difference: {mean_max_diff:.4f}")
        return mean_iou, mean_max_diff
    else:
        print("No valid comparisons made.")
        return None, None


@pytest.fixture
def small_dataset_path() -> Path:
    path = Path(
        "..",
        "test_data_ncore",
        "cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.zarr.itar",
    )
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )
    return path


def test_trt_torch_model_comparison(small_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Integration test to compare TensorRT and PyTorch model outputs"""

    output_root = tmp_path_factory.mktemp("trt_torch_comparison_test")
    trt_output = output_root / "trt_output"
    torch_output = output_root / "torch_output"
    os.makedirs(trt_output, exist_ok=True)
    os.makedirs(torch_output, exist_ok=True)

    camera_id = "camera_front_wide_120fov"  # Use the same camera as in aux_gen_test.py

    print("\nRunning with TensorRT model...")
    trt_result = CliRunner().invoke(
        cli,
        [
            f"--shard-file-pattern={small_dataset_path}",
            f"--camera-id={camera_id}",
            f"--output-dir={trt_output}",
            "--visualize",
            "--store-meta",
            "offset",
            "--shard-duration-sec=0.4",  # restrict to few frames only to terminate more quickly
        ],
        catch_exceptions=False,
    )
    assert trt_result.exit_code == 0, f"TensorRT run failed with exit code {trt_result.exit_code}"

    print("\nRunning with PyTorch...")
    torch_result = CliRunner().invoke(
        cli,
        [
            f"--shard-file-pattern={small_dataset_path}",
            f"--camera-id={camera_id}",
            f"--output-dir={torch_output}",
            "--visualize",
            "--disable-trt",
            "--store-meta",
            "offset",
            "--shard-duration-sec=0.4",  # restrict to few frames only to terminate more quickly
        ],
        catch_exceptions=False,
    )
    assert torch_result.exit_code == 0, f"PyTorch run failed with exit code {torch_result.exit_code}"

    mean_iou, mean_max_diff = compare_segmentation_outputs(torch_output, trt_output)

    assert mean_iou > 0.5, "IoU is too low"
    assert mean_max_diff < 10, "Max difference is too high"
