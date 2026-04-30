# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Common utility functions for camera and LiDAR models.
"""

from typing import Union

import numpy as np
import torch

from torch import Tensor

from libs.geometry.kernels.pose import se3pose_to_matrix
from libs.geometry.kernels.quaternion import quat_slerp_batched


# Type alias for inputs that can be either Tensor or numpy array
TensorLike = Union[Tensor, np.ndarray]


def to_torch(
    data: TensorLike,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Convert numpy array or tensor to torch tensor on specified device/dtype.

    Args:
        data: Input data (numpy array or torch tensor)
        device: Target device
        dtype: Target dtype

    Returns:
        Tensor on the specified device with the specified dtype
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device=device, dtype=dtype)
    return data.to(device=device, dtype=dtype)


def poses_to_matrix(translations: Tensor | None, rotations: Tensor | None) -> Tensor | None:
    """Convert translations and rotations to 4x4 transformation matrices.

    Args:
        translations: (N, 3) translation vectors, or None
        rotations: (N, 4) quaternions in xyzw format, or None

    Returns:
        (N, 4, 4) transformation matrices, or None if inputs are None
    """
    if translations is None or rotations is None:
        return None
    return se3pose_to_matrix(translations, rotations)


def valid_flags_to_indices(valid_flags: Tensor | None) -> Tensor | None:
    """Convert boolean validity mask to indices of valid elements.

    Args:
        valid_flags: (N,) boolean mask, or None

    Returns:
        (M,) int64 indices of valid elements, or None if input is None
    """
    if valid_flags is None:
        return None
    return torch.where(valid_flags)[0]


def batched_quat_slerp(q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
    """Batched spherical linear interpolation between quaternions.

    Unlike quat_slerp which only supports scalar t, this supports per-element t.
    Delegates to the GPU-accelerated, differentiable kernel in geometry/kernels.

    Args:
        q1: (N, 4) start quaternions in xyzw format
        q2: (N, 4) end quaternions in xyzw format
        t: (N,) interpolation parameters in [0, 1]

    Returns:
        (N, 4) interpolated quaternions
    """
    return quat_slerp_batched(q1, q2, t)


def compute_scaled_resolution(
    original_resolution: tuple[int, int],
    scale: float | tuple[float, float],
    new_resolution: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Compute new resolution after scaling.

    Args:
        original_resolution: (width, height) in pixels
        scale: Isotropic (float) or anisotropic (tuple) scaling factor
        new_resolution: Optional explicit new resolution (overrides scale computation)

    Returns:
        New (width, height) tuple
    """
    if new_resolution is not None:
        return new_resolution

    if isinstance(scale, tuple):
        scale_x, scale_y = scale
    else:
        scale_x, scale_y = scale, scale

    return (
        int(original_resolution[0] * scale_x),
        int(original_resolution[1] * scale_y),
    )


def filter_by_validity(
    data: Tensor,
    valid_flags: Tensor | None,
    return_all: bool,
) -> Tensor:
    """Filter data by validity flags.

    Args:
        data: (N, ...) tensor to filter
        valid_flags: (N,) boolean mask, or None
        return_all: If True, return all data without filtering

    Returns:
        Filtered or original data
    """
    if return_all or valid_flags is None:
        return data
    return data[valid_flags]


__all__ = [
    "TensorLike",
    "to_torch",
    "poses_to_matrix",
    "valid_flags_to_indices",
    "batched_quat_slerp",
    "compute_scaled_resolution",
    "filter_by_validity",
]
