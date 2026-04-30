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
import math

from dataclasses import dataclass
from typing import List, Tuple

import lietorch as lt
import numpy as np
import scipy
import torch

from numpy.typing import ArrayLike, NDArray

import ncore_internal.impl.common.transformations as ncore_internal_transformations

from ncore.impl.common.transformations import PoseInterpolator
from nre.utils.knn import knn_points


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


def quat_to_so3_matrix(quat: torch.Tensor | np.ndarray, unbatch: bool = True, normalize: bool = True) -> torch.Tensor:
    """
    Converts a single / batch of quaternions (4) to SO3 representation.

    Args:
        quat: single / batch of quaternions (XYZW convention) [bs, 4] or [4]]
        unbatch: if the single example should be unbatched (first dimension removed) or not

    Returns:
        single / batch of SO3 matrices [bs, 3, 3] or [3,3]
    """

    # Convert numpy array to torch tensor
    quat_torch: torch.Tensor = torch.from_numpy(quat) if isinstance(quat, np.ndarray) else quat
    quat_torch = quat_torch.reshape((-1, 4))  # batch dimensions unconditionally

    # Normalize the quaternions
    if normalize:
        quat_torch = quat_torch / torch.norm(quat_torch, dim=1, keepdim=True)

    num_quats, _ = quat_torch.shape

    x, y, z, w = torch.unbind(quat_torch, -1)
    x_2 = x * x
    y_2 = y * y
    z_2 = z * z
    xy = x * y
    xz = x * z
    xw = x * w
    yz = y * z
    yw = y * w
    zw = z * w

    R = torch.stack(
        (
            1 - 2 * (y_2 + z_2),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (x_2 + z_2),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (x_2 + y_2),
        ),
        -1,
    ).reshape(num_quats, 3, 3)

    if unbatch:  # unbatch dimensions conditionally
        R = R.squeeze()

    return R  # (N,3,3) or (3,3)


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert (a batch of) quaternions to Euler angles. Convention is the following:
    - pitch: rotation around X-axis pointing to the right. Range: [-pi, pi]
    - yaw: rotation around Y-axis pointing down. Range: [-pi/2, pi/2]
    - roll: rotation around Z-axis pointing forward. Range: [-pi, pi]

    Args:
        quaternions (torch.Tensor): A batch of quaternions with shape [..., 4] in (x, y, z, w) format.

    Returns:
        torch.Tensor: Euler angles of shape [..., 3] (roll, pitch, yaw) in radians.
    """
    assert quat.shape[-1] == 4, "Each quaternion must have 4 components (x, y, z, w)."

    x, y, z, w = quat.unbind(-1)  # Extract components from [..., 4], each component is of shape [...]

    # Rotation around X-axis pointing to the right (pitch)
    sinp_cosy = 2.0 * (w * x + y * z)
    cosp_cosy = 1.0 - 2.0 * (x * x + y * y)
    pitch = torch.atan2(sinp_cosy, cosp_cosy)

    # Rotation around Y-axis pointing down (yaw)
    # Use asin for numerical stability, avoiding square roots that can cause gradient issues
    # The expression 2*(w*y - x*z) is clamped to [-1 + e, 1 - e] to ensure asin is well-defined
    yaw_sin = torch.clamp(2.0 * (w * y - x * z), -1.0 + 1e-5, 1.0 - 1e-5)
    yaw = torch.asin(yaw_sin)

    # Rotation around Z-axis pointing forward (roll)
    sinr_cosy = 2.0 * (w * z + x * y)
    cosr_cosy = 1.0 - 2.0 * (y * y + z * z)
    roll = torch.atan2(sinr_cosy, cosr_cosy)

    return torch.stack([roll, pitch, yaw], dim=-1)  # Stack the three angles into a single tensor of shape [..., 3]


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


def se3_matrix_to_se3(T: torch.Tensor | np.ndarray, unbatch=True, reduced=False) -> lt.SE3:
    """Converts a single / batch of rigid transformations represented as 4x4 / 3x4 (reduced) matrices

    ⎡ R  t ⎤
    ⎣ 0  1 ⎦

    to SE3 Lie group elements (unbatches conditionally)

    Args:
        T: single / batch of SE3 transformation matrices [BS0, BS1, ..., BSN, D, 4]
        unbatch: if the single example should be unbatched (first dimension removed) or not
        reduced: D = 3 if True ("reduced"), D = 4 if False ("not reduced")

    Returns:
        single / batch of SE3 Lie group elements [] / [BS0, BS1, ..., BSN]
    """

    # Convert numpy array to torch tensor
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T)

    # batch dimensions unconditionally
    batch_dims = T.shape[:-2]  # batch dimensions BS0, BS1, ..., BSN (potentially empty)
    T = T.reshape((-1, 4, 4)) if not reduced else T.reshape((-1, 3, 4))

    vec = torch.hstack((T[:, :3, 3], so3_matrix_to_quat(T[:, :3, :3], unbatch=False))).reshape(batch_dims + (7,))

    if unbatch:  # unbatch dimensions conditionally
        vec = vec.squeeze()

    return lt.SE3.InitFromVec(vec)


def rotation_6d_to_matrix(d6_id: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6), around identity element

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    assert d6_id.shape[-1] == 6, "6d rotations need to have six parameters"

    a1 = torch.stack((d6_id[..., 0] + 1.0, d6_id[..., 1], d6_id[..., 2]), dim=-1)
    a2 = torch.stack((d6_id[..., 3], d6_id[..., 4] + 1.0, d6_id[..., 5]), dim=-1)
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6), around identity element

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    assert matrix.shape[-2:] == (3, 3), "6d rotation representations can only be computed for 3x3 rotation matrices"

    batch_dim = matrix.shape[:-2]

    d6 = matrix[..., :2, :].reshape(batch_dim + (6,))

    d6_id = d6 - torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=matrix.device, dtype=matrix.dtype).expand(
        batch_dim + (6,)
    )

    return d6_id


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


class PoseLinearVelocityInterpolator(PoseInterpolator):
    """
    Linearly interpolates poses and positions as well as differential position velocities
    to desired timestamps.

    Extrapolation requests result in errors.

    Args:
        poses (np.array): metric poses at given timestamps in a se3 representation [n,4,4]
        timestamps (np.array | list): timestamps of the positions [n]
    """

    def __init__(self, poses: np.ndarray, timestamps_us: np.ndarray | list):
        super().__init__(poses, timestamps_us)
        if poses.ndim == 3:
            # extract from SE3
            positions = poses[:, :3, 3]
        else:
            raise ValueError(f"[PoseLinearVelocityInterpolator]: invalid dimension {poses.ndim} of 'poses'")

        # create splines
        self.position_spline_x = scipy.interpolate.UnivariateSpline(
            timestamps_us,
            positions[:, 0],
            k=1,  # linear spline
            ext="raise",  # disallow extrapolation
            check_finite=True,
        )
        self.position_spline_y = scipy.interpolate.UnivariateSpline(
            timestamps_us,
            positions[:, 1],
            k=1,  # linear spline
            ext="raise",  # disallow extrapolation
            check_finite=True,
        )
        self.position_spline_z = scipy.interpolate.UnivariateSpline(
            timestamps_us,
            positions[:, 2],
            k=1,  # linear spline
            ext="raise",  # disallow extrapolation
            check_finite=True,
        )
        self.velocity_spline_x = self.position_spline_x.derivative()
        self.velocity_spline_y = self.position_spline_y.derivative()
        self.velocity_spline_z = self.position_spline_z.derivative()

    def get_positions(self, timestamps_us: np.ndarray | list) -> np.ndarray:
        """Interpolates metric positions at provided timestamps"""
        return np.column_stack(
            [
                self.position_spline_x(timestamps_us),
                self.position_spline_y(timestamps_us),
                self.position_spline_z(timestamps_us),
            ]
        )

    def get_velocities_m_us(self, timestamps_us: np.ndarray | list) -> np.ndarray:
        """Interpolates velocity vectors (in m/us) at provided timestamps"""
        return np.column_stack(
            [
                self.velocity_spline_x(timestamps_us),
                self.velocity_spline_y(timestamps_us),
                self.velocity_spline_z(timestamps_us),
            ]
        )

    def get_velocities_m_s(self, timestamps_us: np.ndarray | list) -> np.ndarray:
        """Interpolates velocity vectors (in m/s) at provided timestamps"""
        return self.get_velocities_m_us(timestamps_us) * 1e6  # convert from m/us to m/s

    def get_velocities_km_h(self, timestamps_us: np.ndarray | list) -> np.ndarray:
        """Interpolates velocity vectors (in km/h) at provided timestamps"""
        return self.get_velocities_m_s(timestamps_us) * 3.6  # convert from m/s to km/h

    def get_speeds_m_s(self, timestamps_us: np.ndarray | list) -> np.ndarray:
        """Interpolates absolute speeds (in m/h) at provided timestamps"""
        return np.linalg.norm(self.get_velocities_m_s(timestamps_us), axis=1)

    def get_speeds_km_h(self, timestamps_us: np.ndarray | list) -> np.ndarray:
        """Interpolates absolute speeds (in km/h) at provided timestamps"""
        return np.linalg.norm(self.get_velocities_km_h(timestamps_us), axis=1)

    def get_distance_m(self, timestamps_us: np.ndarray | list) -> float:
        """
        Calculates the total distance traversed across the given poses.

        This method computes the cumulative distance along the trajectory by summing
        the Euclidean distances between consecutive positions.

        Args:
            timestamps_us: Time range to compute distance for.

        Returns:
            float: Total distance traversed in meters
        """
        if len(timestamps_us) < 2:
            return 0.0

        # Use get_positions for the provided timestamps
        positions = self.get_positions(timestamps_us)

        # Calculate distances between consecutive positions
        position_diffs = np.diff(positions, axis=0)
        segment_distances = np.linalg.norm(position_diffs, axis=1)

        # Sum all segment distances
        distance = np.sum(segment_distances)

        return float(distance)

    def get_displacement_m(self, timestamps_us: np.ndarray | list) -> float:
        """
        Calculates the straight-line displacement between start and end poses for the given time range.

        This method computes the Euclidean distance between the first and last pose positions
        within the specified time range.

        Args:
            timestamps_us: Time range to compute displacement for.

        Returns:
            float: Straight-line displacement between start and end poses in meters
        """
        if len(timestamps_us) < 2:
            return 0.0

        # Use get_positions for the provided timestamps
        positions = self.get_positions(timestamps_us)

        # Calculate straight-line displacement between first and last positions
        displacement = np.linalg.norm(positions[-1] - positions[0])

        return float(displacement)


def interpolate_se3_poses(pose_s: torch.Tensor, pose_e: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """Interpolate/extrapolate pose components linearly between two poses using
    linear interpolation for positions / SLERP interpolation for orientations
    given an interpolation point t in [0,1]

    Args:
        pose_s: Start pose. [Tensor[float32]]. (4, 4)
        pose_e: End pose. [Tensor[float32]]. (4, 4)
        ts: Interpolation points. [Tensor[float32]]. (N)

    Returns:
        Interpolated poses. [Tensor[float32]]. (N, 4, 4)
    """
    assert pose_s.shape == (4, 4)
    assert pose_e.shape == (4, 4)
    assert ts.ndim == 1
    N = ts.shape[0]
    device = pose_s.device
    dtype = pose_s.dtype

    # Convert the start and end rotation matrix to quaternions
    pose_s_quat = so3_matrix_to_quat(pose_s[None, :3, :3]).expand(N, -1)  # [N, 4]
    pose_e_quat = so3_matrix_to_quat(pose_e[None, :3, :3]).expand(N, -1)  # [N, 4]

    # Evaluate orientation interpolation at t
    interp_rot = quat_to_so3_matrix(quat_slerp(pose_s_quat, pose_e_quat, ts))  # [N, 3, 3]

    # Evaluate translation interpolation at t
    interp_transl = (1 - ts)[:, None] * pose_s[None, :3, 3] + ts[:, None] * pose_e[None, :3, 3]  # [N, 3]

    interp_pose = torch.eye(4, 4, device=device, dtype=dtype).repeat(N, 1, 1)
    interp_pose[:, :3, :3] = interp_rot
    interp_pose[:, :3, 3] = interp_transl
    return interp_pose


def quat_slerp(quat_s: torch.Tensor, quat_e: torch.Tensor, t: torch.Tensor, shortest_arc=True) -> torch.Tensor:
    """
    Batch-wise implementation of SLERP (spherical linear interpolation)

    Args:
        quat_s: batch of unit quaternions denoting the start rotation [bs, 4]
        quat_e: batch of unit quaternions denoting the end rotation  [bs, 4]
        t: interpolation steps within 0.0 and 1.0, 0.0 corresponding to q0 and 1.0 to q1 [bs]
        shortest_arc: if True, interpolation will be performed along the shortest arc on SO(3)
    Returns:
        batch of interpolated quaternions [bs, 4]
    """

    assert quat_s.shape == quat_e.shape, "Input quaternions must be of the same shape."

    if len(quat_s.shape) == 1:
        quat_s = torch.unsqueeze(quat_s, 0)
        quat_e = torch.unsqueeze(quat_e, 0)

    assert t.ndim == 1 and t.shape[0] == quat_e.shape[0], "t is expected to have shape [bs]."

    # omega is the 'angle' between both quaternions
    cos_omega = torch.sum(quat_s * quat_e, dim=-1)

    if shortest_arc:
        # Flip quaternions with negative angle to perform shortest arc interpolation.
        quat_e = torch.where((cos_omega < 0).unsqueeze(-1), -quat_e, quat_e)
        cos_omega = torch.abs(cos_omega)

    # True when q0 and q1 are close.
    nearby_quaternions = cos_omega > (1.0 - 1e-3)

    # Clamp to avoid numerical issues in acos at backward pass, as the derivative of
    # acos is undefined at 1 and -1.
    cos_omega = torch.clamp(cos_omega, -1.0 + 1e-6, 1.0 - 1e-6)

    # General approach
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1 - t) * omega)

    beta = torch.sin(t * omega)
    # Use linear interpolation for nearby quaternions
    alpha = torch.where(nearby_quaternions, (1 - t), alpha)
    beta = torch.where(nearby_quaternions, t, beta)

    # Interpolation
    quat = alpha.reshape(-1, 1) * quat_s + beta.reshape(-1, 1) * quat_e
    quat = torch.nn.functional.normalize(quat, dim=-1)

    return quat


def pose_offsets_to_se3(
    rig_translation_offset: Tuple[float, float, float],
    rig_rotation_offset: Tuple[float, float, float],
    rotation_first: bool = False,
) -> np.ndarray:
    """Calculates the 4x4 SE3 matrix from translation and rotation offsets to transform all sensors in the rig frame.

    Args:
        rig_translation_offset: Translation offsets in meters in rig space to be applied to the rig.
            Matches the config parameter val_sensor_transl_delta_m of a validation (mode=val) run.
        rig_rotation_offset: Rotation offsets (yaw, -roll, -pitch) in degrees in rig space to be applied to the rig.
            Matches the config parameter val_sensor_rot_delta_deg of a validation (mode=val) run.
            The rig frame axes are permuted (x,y,z)->(z,-x,-y) before transformation in mode=val,
            this is why the effect is (yaw, -roll, -pitch) instead of the originally intended (roll, yaw, pitch).
        rotation_first: If True, the rotation offsets are applied first, then the translation offsets.
            Otherwise (by default), the translation offsets are applied first, then the rotation offsets.

    Returns:
        rig_offset_se3: 4x4 float32 ndarray, SE3 matrix to transform all sensors in the rig frame by
            left-multiplying the sensor-to-rig poses.

    """
    assert len(rig_translation_offset) == 3, "Translation offset must be a 3-tuple"
    assert len(rig_rotation_offset) == 3, "Rotation offset must be a 3-tuple"

    # Convert rotation offsets to a 3x3 rotation matrix (SO3).
    delta_rot = ncore_internal_transformations.euler_2_so3(
        np.array(rig_rotation_offset, dtype=np.float32), degrees=True, seq="xyz"
    )

    # This is a hack in mode=val but we aim to match the behavior of mode=val, so we replicate the hack for now.
    # This is equivalent to an axis shuffle (x,y,z) -> (z,-x,-y) in the rig frame, applying roll(x)-pitch(y)-yaw(z)
    # rotation, and shuffling back the coordinates, so the rotation is around (z,-x,-y), not (x,y,z) rig coordinates.
    # Utlimately, input rotation offsets are (yaw, -pitch, -roll) in the rig frame.
    # T_ROT_AXIS_PERMUTATION is called T_CAMERA_RIG in mode=val.
    T_ROT_AXIS_PERMUTATION = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    delta_rot = T_ROT_AXIS_PERMUTATION.transpose() @ delta_rot @ T_ROT_AXIS_PERMUTATION

    # Assemble the 4x4 transformation matrix.
    rig_offset_se3 = np.eye(4, dtype=np.float32)
    rig_offset_se3[:3, :3] = delta_rot
    rig_offset_se3[:3, 3] = rig_translation_offset

    if rotation_first:
        rig_offset_se3[:3, 3] = delta_rot @ rig_offset_se3[:3, 3]

    return rig_offset_se3
