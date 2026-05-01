# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions dealing e.g. with generic geometric sampling or transformations."""

import logging

import numpy as np
import torch


log = logging.getLogger(__name__)


def se3_matrix_inverse(se3: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """Compute the inverse of rigid transformations given as SE3 matrices

    Args:
        se3: single / batch of SE3 transformation matrices [bs, 4, 4] or [4,4]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single / batch of SE3 matrices [bs, 4, 4] or [4,4]
    """

    # Convert numpy array to torch tensor
    if isinstance(se3, np.ndarray):
        se3 = torch.from_numpy(se3)

    # batch dimensions unconditionally
    se3 = se3.reshape((-1, 4, 4))  # (N,4,4)

    ret = torch.eye(4, dtype=se3.dtype, device=se3.device).reshape(1, 4, 4).repeat((len(se3), 1, 1))
    ret[:, :3, :3] = (Rt := se3[:, :3, :3].transpose(1, 2))
    ret[:, :3, 3:] = -Rt @ se3[:, :3, 3:]

    # unbatch dimensions conditionally
    if unbatch:
        ret = ret.squeeze()

    return ret  # (N,4,4) or (4,4)


def se3_matrix_to_tquat(se3: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """
    Converts a single / batch of SE3 matrices (4x4) into a single / batch [t,q]
    7d transformation representations consisting of [translation, normalized_quaternion] parts

    Args:
        se3: single / batch of SE3 transformation matrices [bs, 4, 4] or [4,4]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single/ batch of 7D quaternion representation [translation, unit_quaternion]  [bs, 7] or [7]
    """

    # Convert numpy array to torch tensor
    if isinstance(se3, np.ndarray):
        se3 = torch.from_numpy(se3)

    # batch dimensions unconditionally
    se3 = se3.reshape((-1, 4, 4))  # (N,4,4)

    ret = torch.empty((len(se3), 7), dtype=se3.dtype, device=se3.device)
    if len(se3):
        ret[:, :3] = se3[:, :3, 3]
        ret[:, 3:] = so3_matrix_to_quat(se3[:, :3, :3], unbatch=False)

    if unbatch:  # unbatch dimensions conditionally
        ret = ret.squeeze()

    return ret  # (N,7) or (7,)


def so3_matrix_to_quat(R: torch.Tensor | np.ndarray, unbatch: bool = True, normalize: bool = True) -> torch.Tensor:
    """
    Converts a single / batch of SO3 rotation matrices (3x3) to unit quaternion representation.
    Version that is compatible with torch.compile.

    Args:
        R: single / batch of SO3 rotation matrices [bs, 3, 3] or [3,3]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single / batch of unit quaternions (XYZW convention)  [bs, 4] or [4]
    """

    # Convert numpy array to torch tensor
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)

    R = R.reshape((-1, 3, 3))  # batch dimensions unconditionally
    num_rotations, D1, D2 = R.shape
    assert (D1, D2) == (3, 3), "so3_matrix_to_quat: Input has to be a Bx3x3 tensor."

    # Build decision matrix: [r00, r11, r22, trace]
    decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
    decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
    choices = decision_matrix.argmax(dim=1)

    # Compute quaternions for all 4 cases
    # Case 0: i=0, j=1, k=2 (r00 is max)
    q0 = torch.stack(
        [
            1 - decision_matrix[:, -1] + 2 * R[:, 0, 0],  # qx
            R[:, 1, 0] + R[:, 0, 1],  # qy
            R[:, 2, 0] + R[:, 0, 2],  # qz
            R[:, 2, 1] - R[:, 1, 2],  # qw
        ],
        dim=1,
    )

    # Case 1: i=1, j=2, k=0 (r11 is max)
    q1 = torch.stack(
        [
            R[:, 0, 1] + R[:, 1, 0],  # qx
            1 - decision_matrix[:, -1] + 2 * R[:, 1, 1],  # qy
            R[:, 2, 1] + R[:, 1, 2],  # qz
            R[:, 0, 2] - R[:, 2, 0],  # qw
        ],
        dim=1,
    )

    # Case 2: i=2, j=0, k=1 (r22 is max)
    q2 = torch.stack(
        [
            R[:, 0, 2] + R[:, 2, 0],  # qx
            R[:, 1, 2] + R[:, 2, 1],  # qy
            1 - decision_matrix[:, -1] + 2 * R[:, 2, 2],  # qz
            R[:, 1, 0] - R[:, 0, 1],  # qw
        ],
        dim=1,
    )

    # Case 3: trace is max
    q3 = torch.stack(
        [
            R[:, 2, 1] - R[:, 1, 2],  # qx
            R[:, 0, 2] - R[:, 2, 0],  # qy
            R[:, 1, 0] - R[:, 0, 1],  # qz
            1 + decision_matrix[:, -1],  # qw
        ],
        dim=1,
    )

    # Select the appropriate quaternion based on choices
    qcands = torch.stack((q0, q1, q2, q3), dim=1)
    oh = torch.nn.functional.one_hot(choices.to(torch.long), num_classes=4).to(dtype=R.dtype)
    quat = (qcands * oh.unsqueeze(-1)).sum(dim=1)

    if normalize:
        quat = torch.nn.functional.normalize(quat, dim=1)

    if unbatch:  # unbatch dimensions conditionally
        quat = quat.squeeze()

    return quat  # (N,4) or (4,)


def tquat_to_se3_matrix(tquat: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """
    Converts a single / batch of [t,q] 7d transformation representations consisting of
    [translation, normalized_quaternion] parts into a single / batch of N SE3 matrices (4x4)

    Args:
        quat: single/ batch of 7D quaternion representation (XYZW convention) [translation, unit_quaternion]  [bs, 7] or [7]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single / batch of SE3 matrices [bs, 4, 4] or [4,4]
    """

    # Convert numpy array to torch tensor
    if isinstance(tquat, np.ndarray):
        tquat = torch.from_numpy(tquat)

    # batch dimensions unconditionally
    tquat = tquat.reshape((-1, 7))  # (N,7)

    ret = torch.eye(4, dtype=tquat.dtype, device=tquat.device).reshape(1, 4, 4).repeat((len(tquat), 1, 1))
    ret[:, :3, :3] = quat_to_so3_matrix(tquat[:, 3:], unbatch=False)
    ret[:, :3, 3] = tquat[:, :3]

    # unbatch dimensions conditionally
    if unbatch:
        ret = ret.squeeze()

    return ret  # (N,4,4) or (4,4)


def quat_mult_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiplies two quaternions.
    """

    # batch dimensions unconditionally
    batch_dims = q1.shape[:-1]  # batch dimensions BS0, BS1, ..., BSN (potentially empty)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack([x, y, z, w], dim=-1).reshape(batch_dims + (4,))
