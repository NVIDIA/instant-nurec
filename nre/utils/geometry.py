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


@dataclass
class Plane:
    """Class to represent a plane in 3D space"""

    normal: NDArray[np.float32]  # Expected shape: (3,)
    offset: float

    def __post_init__(self):
        assert len(self.normal.shape) == 1
        assert self.normal.size == 3
        self.normal = self.normal.flatten()

    def distance_from_points(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate distance between the plane and provided points"""
        assert len(points.shape) == 2
        assert points.shape[1] == 3
        return np.abs(points @ self.normal + self.offset) / np.linalg.norm(self.normal)


def hann(x: ArrayLike, threshold: float) -> np.ndarray:
    """Elevated cosine (Hann) function applied to a scalar or array

    The function is maximal 1 at 0 and smoothly drops to 0 inside [-threshold, threshold].
    Returns 1.0 at 0, 0.75 at 33%, 0.5 at 50%, 0.25 at 66% and 0.0 beyond 100% of threshold and -threshold.
    """
    assert threshold > 0.0
    return 0.5 + 0.5 * np.cos(math.pi * np.clip(np.asarray(x) / threshold, a_min=-1.0, a_max=1.0))


class PlaneDetectorRansac:
    """Algorithm for detecting a global dominant plane in a point cloud"""

    # Setting defaults might seem arbitrary at this point but simplifies tests that are agnostic to these parameters.
    def __init__(self, distance_threshold: float = 1.0, angle_threshold_deg: float = 30.0):
        assert distance_threshold > 0.0
        assert angle_threshold_deg > 0.0
        assert angle_threshold_deg <= 90.0
        self.distance_threshold = distance_threshold
        self.angle_threshold_deg = angle_threshold_deg

    def generate_planes_from_single_points(
        self, points: NDArray[np.float32], normals: NDArray[np.float32], num_planes: int
    ) -> Tuple[List[Plane], NDArray[np.int32]]:
        """Generate a given number of plane hypotheses, each from a single oriented point (assuming accurate normals)

        Requires point normals of decent quality, such that the plane with that normal passing through the point
        aligns well with (other points on) the sought plane in the scene.

        Needs less plane hypotheses to achieve the same chance of including at least one uncontaminated hypothesis
        than from a point triplet because the chance of sampling 1 inlier is higher than sampling 3 inliers.

        Hypotheses from single points also make it possible to prefilter generator (or seed) points based on geometric
        constraints on their generated plane, i.e. avoid full testing of plane hypotheses that do not satisfy certain
        constraints, such as passing close to a known ground control point. This saves useless computations.
        """
        assert points.shape[1] == 3
        assert points.shape == normals.shape
        assert len(points) >= num_planes  # otherwise np.random.choice() raises an error
        assert np.allclose(np.sum(normals**2, axis=1), 1.0)  # Point normals expected to be unit-norm

        sample_indices = np.random.choice(len(points), size=(num_planes,), replace=False)
        sample_points = points[sample_indices]
        plane_normals = normals[sample_indices]
        plane_offsets = -np.sum(plane_normals * sample_points, axis=1)
        planes = [Plane(plane_normals[i], plane_offsets[i]) for i in range(num_planes)]

        return planes, sample_indices

    def generate_planes_from_point_triplets(
        self, points: NDArray[np.float32], normals: NDArray[np.float32], num_planes: int
    ) -> Tuple[List[Plane], NDArray[np.int32]]:
        """Generate a given number of plane hypotheses from random samples of oriented point triplets

        Requires point normals but they do not necessarily need to be of high quality.
        It can return more precise uncontaminated planes than from individual oriented points because it is unlikely
        to sample multiple points of a triplet from nearby, but it requires more hypotheses to achieve the same chance
        of an uncontaminated hypothesis because the chance of sampling 3 inliers is lower than sampling 1 inlier.
        """
        # The requirement of input point normals is easy to remove if needed.
        # Hypotheses from point triplets are the only solution when no point normals are available.

        assert points.shape[1] == 3
        assert points.shape == normals.shape
        assert np.allclose(np.sum(normals**2, axis=1), 1.0)  # Point normals expected to be unit-norm

        num_points = len(points)
        triplet_indices = np.random.choice(num_points, size=(num_planes, 3), replace=False)

        # Collect the sampled point triplets to generate planes.
        # A point triplet is formed by the respective rows of the three matrices below, each of shape (num_planes,3).
        seed_points_a = points[triplet_indices[:, 0]]
        seed_points_b = points[triplet_indices[:, 1]]
        seed_points_c = points[triplet_indices[:, 2]]

        # Compute unit-length normals of each plane (per point triplet).
        plane_normals = np.cross(seed_points_c - seed_points_a, seed_points_b - seed_points_a)
        plane_normals /= np.linalg.norm(plane_normals, axis=1, keepdims=True)

        # Compute the plane offset such that the first point fits the plane.
        plane_offsets = -np.sum(plane_normals * seed_points_a, axis=1)

        # Ensure consistent plane orientation with its seed points:
        # Flip the plane such that the angle between its normal and at least 2 of the 3 seed normals < 90 degrees.
        flip_planes = np.any(
            np.stack(
                [
                    np.sum(normals[triplet_indices[:, 0]] * plane_normals, axis=1) < 0.0,
                    np.sum(normals[triplet_indices[:, 1]] * plane_normals, axis=1) < 0.0,
                    np.sum(normals[triplet_indices[:, 2]] * plane_normals, axis=1) < 0.0,
                ],
                axis=1,
            ),
            axis=1,
        )
        plane_normals[flip_planes] = -plane_normals[flip_planes]
        plane_offsets[flip_planes] = -plane_offsets[flip_planes]
        planes = [Plane(plane_normals[i], plane_offsets[i]) for i in range(num_planes)]
        return planes, triplet_indices

    def evaluate_plane_hypothesis(
        self, plane: Plane, points: NDArray[np.float32], normals: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Score the consistency between a plane hypothesis and a number of orinted points

        Soft scoring: Only points exactly on the plane and with a normal consistent with the plane normal
        contribute with a max score of (1.0), the contribution drops with distance and normal angle deviations,
        and vanishes at the distance and angle thresholds specified.
        This is in contrast to classic hard count of 1 for points within thresholds and 0 for outliers.
        """
        assert points.shape[1] == 3
        assert points.shape == normals.shape
        assert np.allclose(np.sum(normals**2, axis=1), 1.0)  # Point normals expected to be unit-norm
        normal_length = np.linalg.norm(plane.normal)
        assert normal_length > 0.0

        distances = plane.distance_from_points(points)
        distance_weights = hann(distances, self.distance_threshold)
        # Clipping is important, otherwise numerical errors can result in values outside of [-1,1]
        # in which case np.arccos() to returns nans.
        dot_product = np.clip(normals @ (plane.normal / normal_length), a_min=-1.0, a_max=1.0)
        angles_deg = np.degrees(np.arccos(dot_product))
        angle_weights = hann(angles_deg, self.angle_threshold_deg)
        plane_score = np.sum(distance_weights * angle_weights)
        assert not np.isnan(plane_score)
        return plane_score

    def find_inliers(
        self, plane: Plane, points: NDArray[np.float32], normals: NDArray[np.float32]
    ) -> NDArray[np.bool_]:
        """Label points as inlier (True) or outlier (False) to a given plane based on their positions and normals"""
        assert points.shape[1] == 3
        assert points.shape == normals.shape
        assert np.allclose(np.sum(normals**2, axis=1), 1.0)  # Point normals expected to be unit-norm
        normal_length = np.linalg.norm(plane.normal)
        assert normal_length > 0.0

        distances = plane.distance_from_points(points)
        angles_deg = np.degrees(np.arccos(normals @ (plane.normal / normal_length)))
        return np.logical_and(distances <= self.distance_threshold, angles_deg <= self.angle_threshold_deg)


def estimate_ground_plane(
    trajectory: np.ndarray, scale: float, order: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 2d plane on `trajectory`.

    Args:
        trajectory: 3d positions from the pose matrices describing the path in FLU [x, y, z] coordinates, shape (N, 3).
        scale: length of the fitted plane's side e.g. 2 gives a plane centered over the span [-1, +1].
        order: linear or quadratic plane fitting, via values 1 or 2 respectively.

    Returns:
        plane: A (3, 3) matrix, where rows 1 and 2 describe vectors lying on the plane, with row the normal vector.
        init_points: The ground plane points that are fit to the trajectory using second order least squares approximation.
        optimised_points: The initial ground plane points used to initialise the optimisation process.
    """
    if order not in (1, 2):
        raise ValueError(f"Order of the fit must be linear (=1) or quadratic (=2), got {order=}")

    # TODO: add an optimisation constraint e.g. on the normal of the plane to resolve ill-posed problems, such as a
    #       perfectly straight line, a single point, or a tight cluster of points when there is no real ego motion.
    match (zero_grad := np.all(np.gradient(trajectory, axis=0) == 0.0, axis=0)).sum():
        case 0:
            pass
        case _:
            zero_grad_dims = dict(zip(list("xyz"), zero_grad))
            log.warning("The input trajectory is has zero gradient in the following axes: %s", zero_grad_dims)

    # Create points to uniformly cover plane in XY plane (Z is height) over the AABB.
    grid_side = np.linspace(-scale / 2, scale / 2, 20)
    _xs, _ys = np.meshgrid(grid_side, grid_side)
    plane_xs, plane_ys = _xs.flatten(), _ys.flatten()
    plane_zs = np.ones_like(plane_xs) * trajectory[:, 1].mean()

    # Fitting a 2d ground plane to a (theoretically) possible 1d line trajectory is ill-posed; with infinite solutions.
    # Add extra constraint in the form of lateral noise to the trajectory. Mew points deviate along the 2d plane's axes.
    # This introduces more inductive bias, making use of the fact we know we drive on a locally flat 2d plane.
    new_points = trajectory.copy()[::2]  # sample every other trajectory point.
    # Add scaled, low magnitude noise in x and y directions for each new point.
    new_points[:, (0, 1)] += (2 * np.random.random((new_points.shape[0], 2)) - 1) * 0.02 * scale
    # Simply append the new "noisy" trajectory points to include them in the least squares fit.
    trajectory = np.r_[trajectory, new_points]

    # Pick out the x, y and z coordinates.
    trajectory_xs = trajectory[:, 0]
    trajectory_ys = trajectory[:, 1]
    trajectory_zs = trajectory[:, 2]

    # Build linear descriptors from trajectory.
    descriptors = [np.ones(len(trajectory_xs)), trajectory_xs, trajectory_ys]

    if order == 2:
        # Add quadratic descriptors from trajectory.
        sq_descriptors = [
            trajectory_xs * trajectory_ys,
            trajectory_xs**2,
            trajectory_ys**2,
        ]
        descriptors.extend(sq_descriptors)

    descriptors = np.c_[descriptors].T
    # Fit a simple plane via least squares.
    coefficients, residuals, _, _ = scipy.linalg.lstsq(descriptors, trajectory_zs)
    if residuals and (residuals.max() > 1e-3 * scale):
        log.warning("warning: trajectory plane fit may be inaccurate: %.4f", residuals.max())
        log.warning("This may happen with large curvature in height e.g. a trajectory going up/down a hill")

    # Evaluate fit on the points of the plane.
    plane_xys = np.c_[np.ones(plane_xs.shape), plane_xs, plane_ys]
    if order == 2:
        plane_xys = np.c_[plane_xys, plane_xs * plane_ys, plane_xs**2, plane_ys**2]

    pred_plane_zs = np.dot(plane_xys, coefficients).reshape(plane_xs.shape).flatten()
    optimised_points = np.c_[plane_xs, plane_ys, pred_plane_zs].astype(np.float32)
    centered = optimised_points - optimised_points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    assert vh.shape == (3, 3), "expected three 3-vectors describing the 2d plane"

    vh = vh.astype(np.float32)

    init_points = np.c_[plane_xs, plane_ys, plane_zs].astype(np.float32)
    assert init_points.shape == optimised_points.shape
    return vh, init_points, optimised_points


def uniformly_sample_aabb(mins: torch.Tensor, maxes: torch.Tensor, spacing: float) -> torch.Tensor:
    """Return uniformly spaced 3d point coordinates within the bounding cube, defined by its min/max coords.

    Args:
        mins: the (x, y, z) coordinates corresponding to the minimums of the box.
        maxes: the (x, y, z) coordinates corresponding to the maximums of the box.
        spacing: the equal spacing of sampled points, used for all three axes.

    Returns:
        3d points uniformly filling the bounding cube, shape (num_points, 3[x, y, z]).
            Where: num_points = (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / (spacing ** 3)
    """
    xmin, ymin, zmin = mins.tolist()
    xmax, ymax, zmax = maxes.tolist()
    x_steps = (xmax - xmin) / spacing
    y_steps = (ymax - ymin) / spacing
    z_steps = (zmax - zmin) / spacing
    xs = torch.linspace(xmin, xmax, int(x_steps))
    ys = torch.linspace(ymin, ymax, int(y_steps))
    zs = torch.linspace(zmin, zmax, int(z_steps))
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="xy"), dim=0)
    assert isinstance(grid, torch.Tensor), "Assertion for mypy"
    return grid.T.reshape(-1, 3).to(dtype=torch.float32)


def compute_aabbs_and_trajs(
    visible_ranges: np.ndarray, ego2worlds: np.ndarray, volume_limit: float
) -> Tuple[List, List]:
    """Return a minimal number of axis-aligned bounding boxes (AABBs) that covers all the
    space visible by the robot while ensuring each AABB volume not exceeding `volume_limit`.
    """

    def get_world_corners(transform, visible_ranges):
        corners = [
            [visible_ranges[0][0], visible_ranges[1][0], visible_ranges[2][0], 1],
            [visible_ranges[0][0], visible_ranges[1][0], visible_ranges[2][1], 1],
            [visible_ranges[0][0], visible_ranges[1][1], visible_ranges[2][0], 1],
            [visible_ranges[0][0], visible_ranges[1][1], visible_ranges[2][1], 1],
            [visible_ranges[0][1], visible_ranges[1][0], visible_ranges[2][0], 1],
            [visible_ranges[0][1], visible_ranges[1][0], visible_ranges[2][1], 1],
            [visible_ranges[0][1], visible_ranges[1][1], visible_ranges[2][0], 1],
            [visible_ranges[0][1], visible_ranges[1][1], visible_ranges[2][1], 1],
        ]
        return [np.dot(transform, np.array(corner).T)[:3] for corner in corners]

    aabbs = []
    trajs = []
    curr_box = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]])
    prev_box = None
    traj = []
    for tform in ego2worlds:
        corners = np.array(get_world_corners(tform, visible_ranges))
        assert corners.shape == (8, 3)
        max_corner, min_corner = np.max(corners, axis=0), np.min(corners, axis=0)
        curr_box = np.stack(
            [
                np.min(np.stack([curr_box[:, 0], min_corner]), axis=0),
                np.max(np.stack([curr_box[:, 1], max_corner]), axis=0),
            ]
        ).T
        traj.append(tform)
        curr_volume = np.prod(curr_box[:, 1] - curr_box[:, 0])
        if curr_volume > volume_limit:
            if prev_box is not None:
                aabbs.append(prev_box)
                trajs.append(np.stack(traj[:-1]))
                traj = traj[-1:]
            curr_box = np.stack([min_corner, max_corner]).T
        prev_box = curr_box
    if all(curr_box[dim][0] != float("inf") for dim in range(3)):
        aabbs.append(curr_box)
        trajs.append(np.stack(traj))
    return aabbs, trajs


def cartesian_to_spherical(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transforms a tensor of cartesian coordinates to their spherical representation.

    Args:
        data: tensor of cartesian coordinates (*, 3)

    Returns:
        phi: (*, )
        theta: (*, )
        r: : (*, )
    """
    assert data.shape[-1] == 3, "data must have 3 dimensions"

    r = torch.norm(data, dim=-1)
    phi = torch.acos(torch.clamp(data[..., 2] / r, min=-1.0, max=1.0))
    theta = torch.atan2(data[..., 0], data[..., 1])

    return phi, theta, r


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


def vector_align_rotation(v1: torch.Tensor | np.ndarray, v2: torch.Tensor | np.ndarray, unbatch=True) -> torch.Tensor:
    """
    Returns a single / batch of rotation matrices that rotate each vector in v1 to align with the corresponding vector in v2.
    Args:
        v1: single/ batch of non-zero vectors, shape [bs, 3] or [3]
        v2: single/ batch of non-zero vectors, shape [bs, 3] or [3]

    Returns:
        single / batch of rotation matrices, shape [bs, 3, 3] or [3, 3]
    """

    if isinstance(v1, np.ndarray):
        v1 = torch.from_numpy(v1)

    if isinstance(v2, np.ndarray):
        v2 = torch.from_numpy(v2)

    # batch dimensions unconditionally
    v1 = v1.reshape((-1, 3))  # (N,3)
    v2 = v2.reshape((-1, 3))  # (N,3)

    N = v1.shape[0]

    u = torch.nn.functional.normalize(v1, dim=-1)
    Ru = torch.nn.functional.normalize(v2, dim=-1)
    I = torch.eye(3, 3, device=v1.device).unsqueeze(0).repeat(N, 1, 1)

    # the cos angle between the vectors
    c = torch.bmm(u.view(N, 1, 3), Ru.view(N, 3, 1)).squeeze(-1)

    eps = 1.0e-10
    # the cross product matrix of a vector to rotate around
    K = torch.bmm(Ru.unsqueeze(2), u.unsqueeze(1)) - torch.bmm(u.unsqueeze(2), Ru.unsqueeze(1))
    # Rodrigues' formula
    ret = I + K + (K @ K) / (1 + c)[..., None]
    same_direction_mask = torch.abs(c - 1.0) < eps
    same_direction_mask = same_direction_mask.squeeze(-1)
    opposite_direction_mask = torch.abs(c + 1.0) < eps
    opposite_direction_mask = opposite_direction_mask.squeeze(-1)
    ret[same_direction_mask] = torch.eye(3, dtype=v1.dtype, device=v1.device)
    ret[opposite_direction_mask] = -torch.eye(3, dtype=v1.dtype, device=v1.device)

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


@torch.no_grad()
def chamfer_distance_pytorch(x: torch.Tensor, y: torch.Tensor, norm: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the chamfer distance for two point clouds x, y.
    If norm==2, the returned distances are the squared values"""

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    _x = x[None] if len(x.shape) == 2 else x
    _y = y[None] if len(y.shape) == 2 else y

    if _y.shape[0] != _x.shape[0] or _y.shape[2] != _x.shape[2]:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(_x, _y, norm=norm, K=1)
    y_nn = knn_points(_y, _x, norm=norm, K=1)

    cham_x_to_y = x_nn.dists[..., 0]  # (N, P1)
    cham_y_to_x = y_nn.dists[..., 0]  # (N, P2)
    cham_x_to_y = cham_x_to_y[0] if len(x.shape) == 2 else cham_x_to_y
    cham_y_to_x = cham_y_to_x[0] if len(y.shape) == 2 else cham_y_to_x
    return cham_x_to_y, cham_y_to_x


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


def box_filter_points(points: torch.Tensor, box_size: float, max_count: int) -> torch.Tensor:
    """
    Box filter the input point cloud to remove points that are too dense in a given box size.
    Args:
        points: Input point cloud of shape (N, 3)
        box_size: Size of the box to filter points
        max_count: Maximum number of points allowed in each box
    Returns:
        Filtered point cloud of shape (M, 3) where M <= N
    """
    box_inds = torch.floor(points / box_size).long()
    _, box_inds_inverse, counts = torch.unique(box_inds, return_inverse=True, return_counts=True, dim=0)
    filter_mask = counts[box_inds_inverse] > max_count
    return points[filter_mask]


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
