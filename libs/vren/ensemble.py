# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch

from numba import jit, prange

from libs.vren.interface import vren  # type: ignore


@jit(nopython=True, parallel=True, cache=True)
def ensemble_numba(points: np.ndarray, ignore_label=255) -> np.ndarray:
    res = np.full(len(points), ignore_label, dtype=np.uint8)

    for idx in prange(len(points)):
        label = points[idx]
        valid_label = label[label != ignore_label]
        if len(valid_label) == 0:
            continue

        # First find labels with maximum counts
        counts = np.bincount(valid_label)
        max_count = counts.max()

        if np.sum(counts == max_count) == 1:
            res[idx] = np.argmax(counts)
        else:
            # Multiple labels have same max count, use the first one in original array
            max_labels = np.where(counts == max_count)[0]
            for l in valid_label:
                if l in max_labels:
                    res[idx] = l
                    break

    return res


def ensemble_cuda(points: np.ndarray | torch.Tensor, device: torch.device, ignore_label: int = 255) -> torch.Tensor:
    """
    Use CUDA implementation of ensemble function

    Args:
        points: numpy array or torch.Tensor, shape (num_points, num_cameras)
        device: torch.device to use for the CUDA kernel
        ignore_label: value to ignore

    Returns:
        torch.Tensor[uint8] @ cuda, shape (num_points,)
    """

    assert device.type == "cuda", "CUDA device is required"

    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points)  # type: ignore

    assert isinstance(points, torch.Tensor), "Input points must be a numpy array or torch.Tensor"

    # Ensure input is contiguous and correct type
    points = points.to(dtype=torch.uint8, device=device)  # zero-copy if already correct
    if not points.is_contiguous():
        points = points.contiguous()

    # Input validation
    if points.size(0) == 0:
        return torch.empty(0, dtype=points.dtype, device=device)

    if torch.all(points == ignore_label):
        return torch.full((points.shape[0],), ignore_label, device=device, dtype=points.dtype)

    max_classes = int(torch.max(points[points != ignore_label]))
    assert max_classes < 256, (
        "Maximum class number must be less than 256, because the CUDA implementation uses unsigned char to store the class number"
    )

    return vren.lidar_seg_ensemble(
        points, torch.full((len(points),), ignore_label, dtype=points.dtype, device=device), ignore_label
    )
