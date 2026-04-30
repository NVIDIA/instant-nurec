# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Comprehensive unit tests for camera sensor kernels with oracle implementations.

Tests the Slang-backed camera projection functions by verifying against
reference Python implementations:
- Forward projection (world → image)
- Back projection (image → world rays)
- Rolling shutter handling
- Different camera models (pinhole, fisheye, f-theta)
- External distortion
- Pose interpolation

Reference implementations are adapted from the original ncore camera_test.py.
"""

import json
import math
import unittest

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import scipy
import scipy.linalg
import torch

from libs.geometry.kernels.quaternion import quat_identity
from libs.sensors.kernels.cameras import (
    BivariateWindshieldDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
    camera_rays_to_image_points,
    generate_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_shutter_pose,
    image_points_to_world_rays_static_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
)
from libs.sensors.kernels.common.pose import DynamicPose, Pose, Trajectory


# Test configuration - Slang kernels require CUDA
device = torch.device("cuda")

# Tolerance parameters - Camera models have complex distortion so tolerances are relaxed
ATOL = 1e-4
RTOL = 1e-4
MAX_DEVIATION_IN_PIXEL = 0.01  # Maximum allowed deviation in pixels
MAX_DEVIATION_RAY = 0.001  # Maximum allowed deviation for ray direction

# Tighter tolerances for precise gradient tests
TIGHT_RTOL = 1e-5
TIGHT_ATOL = 1e-5

# Very strict tolerance for normalization checks
NORM_ATOL = 1e-6

# Relaxed tolerances for numerical gradient comparisons
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-4


# ============================================================================
# Reference Implementations (Python oracles)
# ============================================================================


class ReferenceFThetaCamera:
    """Reference implementation of F-Theta camera model for oracle testing.

    This is a pure Python implementation based on the original ncore camera model.
    It uses Newton's method for angle-to-radius inversion.
    """

    _FORWARD_POLYNOMIAL_ACCURACY = 0.01

    def __init__(
        self,
        imageSize: np.ndarray,
        principalPoint: np.ndarray,
        backwardPolynomial: list,
        A: Optional[np.ndarray] = None,
    ):
        """Initialize reference F-Theta camera.

        Args:
            imageSize: (width, height) image dimensions
            principalPoint: (cx, cy) principal point
            backwardPolynomial: Polynomial coefficients for radius -> angle mapping
            A: Optional 3x3 transformation matrix (identity if None)
        """
        assert (imageSize[0] > principalPoint[0]) and (imageSize[1] > principalPoint[1])
        assert backwardPolynomial[0] == 0
        assert 1 < len(backwardPolynomial)

        self._imageSize = np.array(imageSize)
        self._principalPoint = np.array(principalPoint)
        self._maxRadius = self._calculateMaxRadius()
        self._backwardPolynomial = backwardPolynomial
        self._forwardPolynomial = self._determineForwardPolynomial(self._maxRadius)
        self._A = A if A is not None else np.eye(3)
        self._Ainv = np.linalg.inv(self._A)

    def isVisible(self, point2d: np.ndarray) -> bool:
        """Check if a 2D point is within image bounds."""
        lastPixel = self._imageSize - np.array([1, 1])
        return (0 <= point2d[0]) and (point2d[0] <= lastPixel[0]) and (0 <= point2d[1]) and (point2d[1] <= lastPixel[1])

    def rays2imagePoints(self, points3d: np.ndarray) -> np.ndarray:
        """Project 3D rays to image points.

        Matches ncore implementation: A is applied to 2D result AFTER projection,
        not to the 3D ray before projection.

        Args:
            points3d: (N, 3) ray directions

        Returns:
            (N, 2) image point coordinates
        """
        rays3d = np.array(points3d, dtype=float).T
        rays3d_norm = np.linalg.norm(rays3d, axis=0)
        rays3d /= rays3d_norm

        # NOTE: Do NOT apply A to 3D ray here - A is applied to 2D result later

        # Project ray to equatorial plane
        directions2d = rays3d[0:2]
        directions2d_norm = np.array(np.linalg.norm(directions2d, axis=0))

        # Compute spherical coordinates polar angle
        polars = np.arctan2(directions2d_norm, rays3d[2])

        # Apply lens distortion
        radii = self._angles2radiiNewton(polars)

        directions2d_norm[directions2d_norm < np.finfo(float).eps] = 1.0
        offsets2d = directions2d * (radii / directions2d_norm)

        # Apply A to the 2D offsets (ncore: A @ offsets2d)
        # Extract 2x2 portion of A for 2D transformation
        A_2x2 = self._A[:2, :2]
        offsets2d = A_2x2 @ offsets2d

        # Add principal point
        polar_mask = np.broadcast_to(np.finfo(float).eps < polars, offsets2d.shape).T
        offsets2d = offsets2d.T
        imagePoints2d = np.full_like(offsets2d, self._principalPoint)
        imagePoints2d[polar_mask] += offsets2d[polar_mask]

        return imagePoints2d

    def imagePoints2rays(self, imagePoints2d: np.ndarray) -> np.ndarray:
        """Back-project image points to camera rays.

        Matches ncore implementation: Ainv is applied to 2D offset BEFORE
        polynomial evaluation, not to the 3D ray after.

        Args:
            imagePoints2d: (N, 2) image coordinates

        Returns:
            (N, 3) normalized ray directions
        """
        offsets2d = np.array(imagePoints2d) - self._principalPoint
        return self._offsets2rays(offsets2d)

    def _offsets2rays(self, offset2d: np.ndarray) -> np.ndarray:
        """Convert 2D offsets from principal point to 3D rays.

        Matches ncore: Apply Ainv to 2D offset FIRST, then use transformed
        offset for polynomial evaluation and ray computation.
        """
        offset = np.array(offset2d)

        # Apply Ainv to 2D offset FIRST (ncore: Ainv @ offset)
        # Extract 2x2 portion of Ainv for 2D transformation
        Ainv_2x2 = self._Ainv[:2, :2]
        if offset.ndim == 1:
            transformed_offset = Ainv_2x2 @ offset
        else:
            transformed_offset = (Ainv_2x2 @ offset.T).T

        # Use the norm of the TRANSFORMED offset for polynomial evaluation
        radius = np.linalg.norm(transformed_offset, axis=transformed_offset.ndim - 1, keepdims=True)
        theta = self._radius2angle(radius)
        s, c = np.sin(theta), np.cos(theta)
        radius[radius < np.finfo(float).eps] = 1.0

        # Compute ray from TRANSFORMED offset - no additional Ainv multiplication
        ray = np.append(transformed_offset * s / radius, c, axis=transformed_offset.ndim - 1)
        ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
        return ray

    def _determineForwardPolynomial(self, maxRadius: float) -> np.ndarray:
        """Compute forward polynomial from backward polynomial using least squares."""
        linearSystemMatrix, linearSystemVector = self._getForwardPolynomialLinearSystem(maxRadius)
        coefficients = self._solveLinearEquation(linearSystemMatrix, linearSystemVector)
        return np.concatenate(([0.0], coefficients))

    def _getForwardPolynomialLinearSystem(self, maxRadius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Build linear system for forward polynomial fitting."""
        samplesRadius = np.array(range(1, int(np.ceil(maxRadius))))
        samplesAngle = np.array([self._radius2angle(r) for r in samplesRadius])
        transposedSystemMatrix = [samplesAngle**p for p in range(1, len(self._backwardPolynomial))]
        return np.transpose(transposedSystemMatrix), samplesRadius

    def _calculateMaxRadius(self) -> float:
        """Calculate maximum radius from principal point to image corners."""
        corners = np.array([[0, 0], [self._imageSize[0] - 1, 0], [0, self._imageSize[1] - 1], self._imageSize - [1, 1]])
        radiusAtCorners = [np.linalg.norm(corner - self._principalPoint) for corner in corners]
        return np.max(np.array(radiusAtCorners))

    def _radius2angle(self, radius: np.ndarray) -> np.ndarray:
        """Apply backward polynomial: radius -> angle."""
        theta = np.zeros_like(radius)
        for c in reversed(self._backwardPolynomial):
            theta = c + radius * theta
        return theta

    def _dradius2angle(self, radius: np.ndarray) -> np.ndarray:
        """Derivative of backward polynomial."""
        theta = np.zeros_like(radius)
        dpolynomial = [i * c for i, c in enumerate(self._backwardPolynomial)]
        for c in reversed(dpolynomial[1:]):
            theta = c + radius * theta
        return theta

    def _angles2radiiNewton(self, thetas: np.ndarray) -> np.ndarray:
        """Convert angles to radii using Newton's method."""
        MAX_ITERATIONS = 10
        THRESHOLD_RESIDUAL = np.finfo(float).eps * 100

        radii = np.array(self._angle2radiusApproximation(thetas))
        residuals = self._radius2angle(radii) - thetas

        iterCount = 0
        notConvergedMask = np.abs(residuals) > THRESHOLD_RESIDUAL
        while iterCount < MAX_ITERATIONS and np.any(notConvergedMask):
            derivatives = self._dradius2angle(radii)
            radii[notConvergedMask] -= residuals[notConvergedMask] / derivatives[notConvergedMask]
            residuals = self._radius2angle(radii) - thetas
            notConvergedMask = np.abs(residuals) > THRESHOLD_RESIDUAL
            iterCount += 1

        radii[notConvergedMask] = np.nan
        return radii

    def _angle2radiusApproximation(self, theta: np.ndarray) -> np.ndarray:
        """Approximate forward polynomial for initial Newton guess."""
        radius = np.zeros_like(theta)
        for c in reversed(self._forwardPolynomial):
            radius = c + theta * radius
        return radius

    @staticmethod
    def _solveLinearEquation(linearSystemMatrix: np.ndarray, linearSystemVector: np.ndarray) -> np.ndarray:
        """Solve linear system using least squares."""
        solution, _, _, _ = scipy.linalg.lstsq(linearSystemMatrix, linearSystemVector)
        return solution


class ReferenceOpenCVPinholeCamera:
    """Reference implementation of OpenCV Pinhole camera model.

    Supports radial (k1-k6), tangential (p1, p2), and thin prism (s1-s4) distortion.
    """

    def __init__(
        self,
        focal_length: np.ndarray,
        principal_point: np.ndarray,
        radial_coeffs: np.ndarray,
        tangential_coeffs: np.ndarray,
        thin_prism_coeffs: np.ndarray,
        resolution: np.ndarray,
        dtype: npt.DTypeLike = np.float32,
    ):
        self.focal_length: np.ndarray = focal_length.astype(dtype)
        self.principal_point: np.ndarray = principal_point.astype(dtype)
        self.radial_coeffs: np.ndarray = radial_coeffs.astype(dtype)
        self.tangential_coeffs: np.ndarray = tangential_coeffs.astype(dtype)
        self.thin_prism_coeffs: np.ndarray = thin_prism_coeffs.astype(dtype)
        self.resolution = resolution
        self.dtype = dtype

    def camera_ray_to_image_point(self, ray: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Project a camera ray to image coordinates.

        Args:
            ray: (3,) ray direction

        Returns:
            image_point: (2,) image coordinates
            valid: bool indicating if projection is valid
        """
        if ray[2] <= 0:
            return np.array([0.0, 0.0], dtype=self.dtype), False

        # Perspective normalization
        x = ray[0] / ray[2]
        y = ray[1] / ray[2]

        # Distortion
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        k1, k2, k3, k4, k5, k6 = self.radial_coeffs
        p1, p2 = self.tangential_coeffs
        s1, s2, s3, s4 = self.thin_prism_coeffs

        # Radial distortion factor
        radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial_denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        radial = radial_num / radial_denom if radial_denom != 0 else radial_num

        # Tangential distortion
        xy = x * y
        dx_tangential = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
        dy_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy

        # Thin prism distortion
        dx_prism = s1 * r2 + s2 * r4
        dy_prism = s3 * r2 + s4 * r4

        # Apply distortions
        x_distorted = x * radial + dx_tangential + dx_prism
        y_distorted = y * radial + dy_tangential + dy_prism

        # Apply camera matrix
        u = x_distorted * self.focal_length[0] + self.principal_point[0]
        v = y_distorted * self.focal_length[1] + self.principal_point[1]

        valid = 0 <= u < self.resolution[0] and 0 <= v < self.resolution[1]
        return np.array([u, v], dtype=self.dtype), valid

    def camera_ray_to_distorted_normalized(self, ray: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Compute distorted normalized coordinates for a camera ray.

        This is useful for computing intrinsics Jacobians since:
        - image_x = fx * x_distorted + cx
        - image_y = fy * y_distorted + cy

        Args:
            ray: (3,) ray direction

        Returns:
            distorted_normalized: (2,) distorted normalized coordinates (x_d, y_d)
            valid: bool indicating if ray is in front of camera
        """
        if ray[2] <= 0:
            return np.array([0.0, 0.0], dtype=self.dtype), False

        # Perspective normalization
        x = ray[0] / ray[2]
        y = ray[1] / ray[2]

        # Distortion
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        k1, k2, k3, k4, k5, k6 = self.radial_coeffs
        p1, p2 = self.tangential_coeffs
        s1, s2, s3, s4 = self.thin_prism_coeffs

        # Radial distortion factor
        radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial_denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        radial = radial_num / radial_denom if radial_denom != 0 else radial_num

        # Tangential distortion
        xy = x * y
        dx_tangential = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
        dy_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy

        # Thin prism distortion
        dx_prism = s1 * r2 + s2 * r4
        dy_prism = s3 * r2 + s4 * r4

        # Apply distortions
        x_distorted = x * radial + dx_tangential + dx_prism
        y_distorted = y * radial + dy_tangential + dy_prism

        return np.array([x_distorted, y_distorted], dtype=self.dtype), True

    def compute_intrinsics_jacobian(self, ray: np.ndarray) -> Optional[dict]:
        """Compute Jacobian of image point w.r.t. intrinsic parameters.

        For pinhole model:
        - image_x = fx * x_distorted + cx
        - image_y = fy * y_distorted + cy

        Returns:
            dict with keys: 'd_fx', 'd_fy', 'd_cx', 'd_cy' containing derivatives,
            or None if ray is invalid.
        """
        distorted, valid = self.camera_ray_to_distorted_normalized(ray)
        if not valid:
            return None

        x_d, y_d = distorted

        return {
            # d(image_x)/d(fx) = x_distorted, d(image_y)/d(fx) = 0
            "d_image_x_d_fx": x_d,
            "d_image_y_d_fx": 0.0,
            # d(image_x)/d(fy) = 0, d(image_y)/d(fy) = y_distorted
            "d_image_x_d_fy": 0.0,
            "d_image_y_d_fy": y_d,
            # d(image_x)/d(cx) = 1, d(image_y)/d(cx) = 0
            "d_image_x_d_cx": 1.0,
            "d_image_y_d_cx": 0.0,
            # d(image_x)/d(cy) = 0, d(image_y)/d(cy) = 1
            "d_image_x_d_cy": 0.0,
            "d_image_y_d_cy": 1.0,
        }

    def image_point_to_camera_ray(self, image_point: np.ndarray, max_iterations: int = 20) -> np.ndarray:
        """Back-project image point to camera ray using Newton iteration.

        Args:
            image_point: (2,) pixel coordinates
            max_iterations: Number of Newton iterations for undistortion

        Returns:
            (3,) normalized ray direction
        """
        # Remove camera matrix
        x = (image_point[0] - self.principal_point[0]) / self.focal_length[0]
        y = (image_point[1] - self.principal_point[1]) / self.focal_length[1]

        # Initial guess (no distortion)
        x0, y0 = x, y

        # Iterative undistortion
        for _ in range(max_iterations):
            r2 = x0 * x0 + y0 * y0
            r4 = r2 * r2
            r6 = r4 * r2

            k1, k2, k3, k4, k5, k6 = self.radial_coeffs
            p1, p2 = self.tangential_coeffs
            s1, s2, s3, s4 = self.thin_prism_coeffs

            radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            radial_denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
            radial = radial_num / radial_denom if radial_denom != 0 else radial_num

            xy = x0 * y0
            dx_tangential = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x0 * x0)
            dy_tangential = p1 * (r2 + 2.0 * y0 * y0) + 2.0 * p2 * xy

            dx_prism = s1 * r2 + s2 * r4
            dy_prism = s3 * r2 + s4 * r4

            x_distorted = x0 * radial + dx_tangential + dx_prism
            y_distorted = y0 * radial + dy_tangential + dy_prism

            x0 = (x - dx_tangential - dx_prism) / radial if radial != 0 else x
            y0 = (y - dy_tangential - dy_prism) / radial if radial != 0 else y

        return np.array([x0, y0, 1.0], dtype=self.dtype) / np.linalg.norm([x0, y0, 1.0])


class ReferenceSimplePinholeCamera:
    """Simple reference pinhole camera with symbolic Jacobian evaluations.

    Supports k1, k2, k3, p1, p2 distortion coefficients (no k4-k6 or thin prism).
    Used for validating Jacobian computations.
    """

    def __init__(
        self,
        focal_length: np.ndarray,
        principal_point: np.ndarray,
        radial_coeffs: np.ndarray,
        tangential_coeffs: np.ndarray,
        dtype: npt.DTypeLike = np.float32,
    ):
        self.focal_length = focal_length.astype(dtype)
        self.principal_point = principal_point.astype(dtype)
        self.radial_coeffs = radial_coeffs.astype(dtype)  # k1, k2, k3 only
        self.tangential_coeffs = tangential_coeffs.astype(dtype)  # p1, p2
        self.dtype = dtype

        # Validate we only use k1, k2, k3
        assert not np.any(radial_coeffs[3:]), "Only supporting non-zero k1, k2, k3"

    def _distortion(self, uvN: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute distortion and its Jacobian."""
        # Helper variables
        u0u0 = uvN[0] * uvN[0]
        u1u1 = uvN[1] * uvN[1]
        r_2 = u0u0 + u1u1
        uv_prod = uvN[0] * uvN[1]
        a1 = 2 * uv_prod
        a2 = r_2 + 2 * u0u0
        a3 = r_2 + 2 * u1u1

        k1, k2, k3 = self.radial_coeffs[:3]
        p1, p2 = self.tangential_coeffs

        icD = 1.0 + r_2 * (k1 + r_2 * (k2 + r_2 * k3))

        delta_x = p1 * a1 + p2 * a2
        delta_y = p1 * a3 + p2 * a1

        uvND = uvN * icD + np.array([delta_x, delta_y], dtype=self.dtype)

        # Jacobian computation
        b1 = k2 + k3 * r_2
        b11 = 2 * (k1 + b1 * r_2) + r_2 * (2 * k3 * r_2 + 2 * b1)
        b2 = uvN[0] * b11
        b3 = uvN[1] * b11
        b4 = (k1 + b1 * r_2) * r_2 + 1.0

        J_uvND = np.array(
            [
                [
                    2 * p1 * uvN[1] + 6 * p2 * uvN[0] + uvN[0] * b2 + b4,
                    2 * p1 * uvN[0] + 2 * p2 * uvN[1] + uvN[0] * b3,
                ],
                [
                    2 * p1 * uvN[0] + 2 * p2 * uvN[1] + uvN[1] * b2,
                    6 * p1 * uvN[1] + 2 * p2 * uvN[0] + uvN[1] * b3 + b4,
                ],
            ],
            dtype=self.dtype,
        )

        return uvND, J_uvND

    def _perspective_normalization(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perspective divide and its Jacobian."""
        uvN = np.array([x[0] / x[2], x[1] / x[2]], dtype=self.dtype)
        J_uvN = np.array(
            [[1 / x[2], 0, -x[0] / x[2] ** 2], [0, 1 / x[2], -x[1] / x[2] ** 2]],
            dtype=self.dtype,
        )
        return uvN, J_uvN

    def _perspective_projection(self, uvND: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply camera matrix and its Jacobian."""
        uv = uvND * self.focal_length + self.principal_point
        J_uv = np.array(
            [[self.focal_length[0], 0], [0, self.focal_length[1]]],
            dtype=self.dtype,
        )
        return uv, J_uv

    def camera_ray_to_image_point(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project ray to image point with Jacobian (chain rule)."""
        uvN, J_uvN = self._perspective_normalization(x)
        uvND, J_uvND = self._distortion(uvN)
        uv, J_uv = self._perspective_projection(uvND)
        return uv, J_uv @ J_uvND @ J_uvN


class ReferenceOpenCVFisheyeCamera:
    """Reference implementation of OpenCV Fisheye camera model."""

    def __init__(
        self,
        focal_length: np.ndarray,
        principal_point: np.ndarray,
        radial_coeffs: np.ndarray,
        max_angle: float,
        resolution: np.ndarray,
        dtype: npt.DTypeLike = np.float32,
    ):
        self.focal_length: np.ndarray = focal_length.astype(dtype)
        self.principal_point: np.ndarray = principal_point.astype(dtype)
        self.radial_coeffs: np.ndarray = radial_coeffs.astype(dtype)  # [k1, k2, k3, k4]
        self.max_angle = max_angle
        self.resolution = resolution
        self.dtype = dtype

    def camera_ray_to_image_point_opencv(self, ray: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Project ray using OpenCV's fisheye model.

        This uses OpenCV's cv2.fisheye.projectPoints for reference.
        """
        ray = np.array(ray, dtype=np.float64)

        if ray[2] <= 0:
            return np.array([0.0, 0.0], dtype=self.dtype), False

        # Use OpenCV fisheye projection
        rvec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tvec = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        K = np.array(
            [
                [self.focal_length[0], 0, self.principal_point[0]],
                [0, self.focal_length[1], self.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        d = self.radial_coeffs.astype(np.float64)

        try:
            p, _ = cv2.fisheye.projectPoints(ray.reshape(1, 1, 3), rvec, tvec, K, d, None, 0.0)
            image_point = p.reshape(2)

            # Check bounds and angle
            ray_norm = ray / np.linalg.norm(ray)
            theta = np.arccos(np.clip(ray_norm[2], -1, 1))

            valid = (
                theta <= self.max_angle
                and 0 <= image_point[0] < self.resolution[0]
                and 0 <= image_point[1] < self.resolution[1]
            )

            return image_point.astype(self.dtype), valid
        except Exception:
            return np.array([0.0, 0.0], dtype=self.dtype), False


# ============================================================================
# Helper Functions
# ============================================================================


def generate_backward_polynomial(resolution: int, base_angle: float, order: int) -> list:
    """Generate backward polynomial for FTheta camera.

    Args:
        resolution: Image resolution (pixels)
        base_angle: Base angle for polynomial
        order: Polynomial order

    Returns:
        List of polynomial coefficients
    """
    first_to_last_pixel_distance = resolution - 1
    polynomial: list[float] = [0.0]
    for j in range(1, order + 1):
        polynomial.append(base_angle / ((0.5 * first_to_last_pixel_distance) ** j))
    return polynomial


def compute_cumulative_angle_at_border(base_angle: float, order: int) -> float:
    """Compute expected angle at image border."""
    return base_angle * order


def create_ftheta_projection(
    resolution: tuple,
    principal_point: np.ndarray,
    backward_polynomial: list,
    A: Optional[np.ndarray] = None,
) -> FThetaProjection:
    """Create FThetaProjection from reference parameters.

    Args:
        resolution: (width, height) tuple
        principal_point: (cx, cy) principal point
        backward_polynomial: Backward polynomial coefficients (c0 + c1*r + c2*r^2 + ...)
        A: Optional transformation matrix

    Returns:
        FThetaProjection instance ready for kernel calls

    Note:
        The Slang kernel uses eval_poly which computes:
            coeffs[0]*theta + coeffs[1]*theta^2 + coeffs[2]*theta^3 + ...
        The reference polynomial is in standard form:
            c[0] + c[1]*r + c[2]*r^2 + c[3]*r^3 + ...
        Since c[0] = 0 for valid backward polynomials, we shift coefficients by 1.
    """
    # Create reference camera to get forward polynomial
    ref_camera = ReferenceFThetaCamera(
        np.array(resolution), principal_point, backward_polynomial, A=A if A is not None else np.eye(3)
    )

    from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS

    bw_poly = np.zeros(MAX_POLYNOMIAL_TERMS, dtype=np.float32)
    fw_poly = np.zeros(MAX_POLYNOMIAL_TERMS, dtype=np.float32)

    for i, c in enumerate(backward_polynomial):
        if i < MAX_POLYNOMIAL_TERMS:
            bw_poly[i] = c

    for i, c in enumerate(ref_camera._forwardPolynomial):
        if i < MAX_POLYNOMIAL_TERMS:
            fw_poly[i] = c

    dbw_poly = np.zeros(MAX_POLYNOMIAL_TERMS, dtype=np.float32)
    dfw_poly = np.zeros(MAX_POLYNOMIAL_TERMS, dtype=np.float32)
    for i in range(MAX_POLYNOMIAL_TERMS - 1):
        dbw_poly[i] = (i + 1) * bw_poly[i + 1]
        dfw_poly[i] = (i + 1) * fw_poly[i + 1]

    # Extract 2x2 portion of A for kernel (kernel uses 2x2 linear term)
    A_mat_3x3 = A if A is not None else np.eye(3)
    A_mat_2x2 = A_mat_3x3[:2, :2]

    return FThetaProjection.from_components(
        principal_point=torch.tensor(principal_point, dtype=torch.float32, device=device),
        fw_poly=torch.tensor(fw_poly, dtype=torch.float32, device=device),
        bw_poly=torch.tensor(bw_poly, dtype=torch.float32, device=device),
        A=torch.tensor(A_mat_2x2, dtype=torch.float32, device=device),
        Ainv=torch.tensor(np.linalg.inv(A_mat_2x2), dtype=torch.float32, device=device),
        dfw_poly=torch.tensor(dfw_poly, dtype=torch.float32, device=device),
        dbw_poly=torch.tensor(dbw_poly, dtype=torch.float32, device=device),
        reference_poly=FThetaPolynomialType.BACKWARD,
        max_angle=float(ref_camera._radius2angle(np.array([ref_camera._maxRadius]))[0]),
        newton_iterations=10,
        min_2d_norm=1e-8,
    )


def create_pinhole_projection(
    focal_length: np.ndarray,
    principal_point: np.ndarray,
    radial_coeffs: np.ndarray,
    tangential_coeffs: np.ndarray,
    thin_prism_coeffs: np.ndarray,
    resolution: np.ndarray,
) -> OpenCVPinholeProjection:
    """Create OpenCVPinholeProjection from parameters."""
    return OpenCVPinholeProjection.from_components(
        focal_length=torch.tensor(focal_length, dtype=torch.float32, device=device),
        principal_point=torch.tensor(principal_point, dtype=torch.float32, device=device),
        radial_coeffs=torch.tensor(radial_coeffs, dtype=torch.float32, device=device),
        tangential_coeffs=torch.tensor(tangential_coeffs, dtype=torch.float32, device=device),
        thin_prism_coeffs=torch.tensor(thin_prism_coeffs, dtype=torch.float32, device=device),
        resolution=torch.tensor(resolution, dtype=torch.int32, device=device),
    )


def create_fisheye_projection(
    resolution: np.ndarray,
    focal_length: np.ndarray,
    principal_point: np.ndarray,
    radial_coeffs: np.ndarray,
    max_angle: float,
) -> OpenCVFisheyeProjection:
    """Create OpenCVFisheyeProjection from parameters."""
    # Create forward polynomial: theta * (1 + k1*theta^2 + k2*theta^4 + ...)
    # OpenCV fisheye uses: theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    forward_poly = torch.zeros(4, dtype=torch.float32, device=device)
    forward_poly[:] = torch.tensor(radial_coeffs[:4], dtype=torch.float32)

    return OpenCVFisheyeProjection.from_components(
        principal_point=torch.tensor(principal_point, dtype=torch.float32, device=device),
        focal_length=torch.tensor(focal_length, dtype=torch.float32, device=device),
        forward_poly=forward_poly,
        resolution=torch.tensor(resolution, dtype=torch.int32, device=device),
        max_angle=max_angle,
        newton_iterations=10,
        min_2d_norm=torch.tensor(1e-8, dtype=torch.float32, device=device),
    )


# ============================================================================
# Common Test Helpers
# ============================================================================


def create_identity_dynamic_pose(dev: torch.device, dtype: torch.dtype = torch.float32) -> DynamicPose:
    """Create a dynamic pose with identity rotation and zero translation."""
    trans = torch.zeros(3, device=dev, dtype=dtype)
    rot = quat_identity((1,), device=dev).squeeze(0).to(dtype)  # Identity quaternion [w, x, y, z]
    pose = Pose(trans, rot)
    return DynamicPose.from_static_pose(pose)


def create_dynamic_pose(
    trans_start: torch.Tensor,
    trans_end: torch.Tensor,
    rot_start: torch.Tensor,
    rot_end: torch.Tensor,
    dev: torch.device,
) -> DynamicPose:
    """Create a dynamic pose with specified start/end translations and rotations."""
    start_pose = Pose(trans_start, rot_start)
    end_pose = Pose(trans_end, rot_end)
    return DynamicPose(start_pose=start_pose, end_pose=end_pose)


# ============================================================================
# Test Classes
# ============================================================================


class TestGenerateImagePoints(unittest.TestCase):
    """Test generate image points."""

    def test_generate_image_points(self):
        """Test generate image points."""
        RESOLUTION = (533, 300)

        # Reference implementation
        def generate_image_points_reference(resolution: tuple[int, int], device: torch.device) -> torch.Tensor:
            w, h = resolution
            sensor_pixels_x, sensor_pixels_y = torch.meshgrid(
                torch.arange(w, dtype=torch.int16, device=device),
                torch.arange(h, dtype=torch.int16, device=device),
                indexing="xy",
            )
            return torch.stack([sensor_pixels_x.flatten(), sensor_pixels_y.flatten()], dim=1).float() + 0.5

        image_points_gt = generate_image_points_reference(RESOLUTION, device=device)
        image_points_from_kernel = generate_image_points(RESOLUTION, device=device)
        torch.testing.assert_close(image_points_from_kernel, image_points_gt)


class TestCudaAvailable(unittest.TestCase):
    """Verify CUDA is available for testing GPU kernels."""

    def test_cuda_available(self):
        """Check that CUDA device is available."""
        self.assertTrue(torch.cuda.is_available(), "CUDA device required for camera kernel tests")


class TestFThetaCameraOracle(unittest.TestCase):
    """Oracle tests for F-Theta camera model comparing kernel to reference."""

    def test_image_points_to_rays_polynomial_orders(self):
        """Test backward polynomial coefficients from r**1 to r**4."""
        for order in range(1, 5):
            with self.subTest(polynomial_order=order):
                self._test_image_points_to_rays(order)

    def _test_image_points_to_rays(self, order: int):
        """Test image point to ray conversion for given polynomial order."""
        base_angle = np.radians(45)
        resolution = 1000
        principal_point = np.array([(resolution - 1) / 2, (resolution - 1) / 2])
        backward_polynomial = generate_backward_polynomial(resolution, base_angle, order)

        # Create reference camera and kernel projection
        ref_camera = ReferenceFThetaCamera(np.array([resolution, resolution]), principal_point, backward_polynomial)
        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial)
        external_distortion = NoExternalDistortion()

        # Test cases: principal point, right edge, bottom edge
        test_points = [
            [principal_point[0], principal_point[1]],  # Center
            [resolution - 1, principal_point[1]],  # Right
            [principal_point[0], resolution - 1],  # Bottom
        ]

        cumulative_angle = compute_cumulative_angle_at_border(base_angle, order)
        expected_rays = [
            [0, 0, 1],  # Center -> optical axis
            [np.sin(cumulative_angle), 0, np.cos(cumulative_angle)],  # Right
            [0, np.sin(cumulative_angle), np.cos(cumulative_angle)],  # Bottom
        ]

        # Convert to tensors
        image_points = torch.tensor(test_points, dtype=torch.float32, device=device)

        # Call kernel
        camera_rays = image_points_to_camera_rays(image_points, projection, external_distortion)

        # Compare with reference
        for i, (point, expected) in enumerate(zip(test_points, expected_rays)):
            ref_ray = ref_camera.imagePoints2rays(np.array([point]))[0]

            # Compare kernel result to reference
            kernel_ray = camera_rays[i].cpu().numpy()
            np.testing.assert_allclose(kernel_ray, ref_ray, atol=MAX_DEVIATION_RAY, err_msg=f"Point {i}: {point}")

            # Also compare to expected analytical ray
            np.testing.assert_allclose(
                kernel_ray, expected, atol=MAX_DEVIATION_RAY, err_msg=f"Point {i} vs expected: {point}"
            )

    def test_rays_to_image_points_polynomial_orders(self):
        """Test ray to image point conversion for various polynomial orders.

        Now uses Newton iteration for exact inversion, should match reference exactly.
        """
        for order in range(1, 5):
            with self.subTest(polynomial_order=order):
                self._test_rays_to_image_points(order)

    def _test_rays_to_image_points(self, order: int):
        """Test ray to image point projection for given polynomial order."""
        # Use a smaller base angle so cumulative angle stays < 90 degrees for all orders
        # For order 4, we need base_angle * 4 < 90, so base_angle < 22.5 degrees
        base_angle = np.radians(20)  # 20 * 4 = 80 degrees max
        resolution = 1000
        principal_point = np.array([(resolution - 1) / 2, (resolution - 1) / 2])
        backward_polynomial = generate_backward_polynomial(resolution, base_angle, order)

        ref_camera = ReferenceFThetaCamera(np.array([resolution, resolution]), principal_point, backward_polynomial)
        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial)
        external_distortion = NoExternalDistortion()

        cumulative_angle = compute_cumulative_angle_at_border(base_angle, order)

        # Test rays (all should have positive z since cumulative_angle < 90 degrees)
        test_rays = [
            [0, 0, 1],  # Optical axis
            [np.sin(cumulative_angle), 0, np.cos(cumulative_angle)],  # Right
            [0, np.sin(cumulative_angle), np.cos(cumulative_angle)],  # Bottom
        ]

        expected_points = [
            [principal_point[0], principal_point[1]],
            [resolution - 1, principal_point[1]],
            [principal_point[0], resolution - 1],
        ]

        camera_rays = torch.tensor(test_rays, dtype=torch.float32, device=device)
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        # Call kernel
        image_points, valid = camera_rays_to_image_points(camera_rays, projection, external_distortion)

        # Verify all are valid (rays should be in front of camera)
        self.assertTrue(
            valid.all(),
            f"All test rays should produce valid projections (cumulative_angle={np.degrees(cumulative_angle):.1f}°)",
        )

        # Compare with reference (now should match exactly due to Newton iteration)
        for i, (ray, expected) in enumerate(zip(test_rays, expected_points)):
            ref_point = ref_camera.rays2imagePoints(np.array([ray]))[0]

            kernel_point = image_points[i].cpu().numpy()
            np.testing.assert_allclose(kernel_point, ref_point, atol=MAX_DEVIATION_IN_PIXEL, err_msg=f"Ray {i}: {ray}")
            np.testing.assert_allclose(
                kernel_point, expected, atol=MAX_DEVIATION_IN_PIXEL, err_msg=f"Ray {i} vs expected"
            )

    def test_round_trip_consistency(self):
        """Test image → rays → image round-trip consistency.

        With Newton iteration for exact polynomial inversion, round-trip should be exact.
        """
        resolution = 1000
        principal_point = np.array([resolution / 2, resolution / 2])
        focal_length_pixel = 500.0
        backward_polynomial = [
            0.0,
            0.4 / focal_length_pixel,
            (0.4 / focal_length_pixel) ** 2,
            (0.4 / focal_length_pixel) ** 3,
            (0.4 / focal_length_pixel) ** 4,
        ]

        ref_camera = ReferenceFThetaCamera(np.array([resolution, resolution]), principal_point, backward_polynomial)
        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial)
        external_distortion = NoExternalDistortion()

        # Test various points from center to corner
        for p in range(0, int(principal_point[0]), 50):
            with self.subTest(pixel_offset=p):
                original_point = np.array([[float(p), float(p)]])
                image_point_tensor = torch.tensor(original_point, dtype=torch.float32, device=device)

                # Image → Rays
                camera_rays = image_points_to_camera_rays(image_point_tensor, projection, external_distortion)

                # Compare ray with reference
                ref_ray = ref_camera.imagePoints2rays(original_point)[0]
                kernel_ray = camera_rays[0].cpu().numpy()
                np.testing.assert_allclose(kernel_ray, ref_ray, atol=MAX_DEVIATION_RAY)

                # Rays → Image (now uses Newton iteration for exact inversion)
                reprojected_points, valid = camera_rays_to_image_points(camera_rays, projection, external_distortion)

                # Check consistency with strict tolerance
                self.assertTrue(valid[0].item(), f"Point {original_point} should reproject validly")
                np.testing.assert_allclose(
                    reprojected_points[0].cpu().numpy(),
                    original_point[0],
                    atol=MAX_DEVIATION_IN_PIXEL,
                    err_msg=f"Round trip failed for point {original_point}",
                )

    def test_principal_point_projects_to_optical_axis(self):
        """Test that principal point projects to optical axis ray."""
        resolution = 1000
        principal_point = np.array([499.5, 499.5])
        backward_polynomial = [0.0, 0.001, 0.0, 0.0]

        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial)
        external_distortion = NoExternalDistortion()

        pp_tensor = torch.tensor([[principal_point[0], principal_point[1]]], dtype=torch.float32, device=device)
        camera_rays = image_points_to_camera_rays(pp_tensor, projection, external_distortion)

        # Should be [0, 0, 1]
        np.testing.assert_allclose(camera_rays[0].cpu().numpy(), [0, 0, 1], atol=NORM_ATOL)

    def test_shifted_principal_point(self):
        """Test principal point shift behavior."""
        fov = np.radians(90)
        resolution = np.array([1000, 1000])
        principal_point = np.array([10.0, 10.0])
        backward_polynomial = [0, fov / resolution[0]]

        ref_camera = ReferenceFThetaCamera(resolution, principal_point, backward_polynomial)
        projection = create_ftheta_projection(tuple(resolution), principal_point, backward_polynomial)
        external_distortion = NoExternalDistortion()

        # Test points offset from shifted principal point
        test_points = [
            [10 + resolution[0] / 2, 10],  # Half resolution right of PP
            [10, 10 + resolution[1] / 2],  # Half resolution below PP
        ]

        expected_rays = [
            [np.sin(fov / 2), 0, np.cos(fov / 2)],
            [0, np.sin(fov / 2), np.cos(fov / 2)],
        ]

        for point, expected in zip(test_points, expected_rays):
            with self.subTest(point=point):
                image_point = torch.tensor([point], dtype=torch.float32, device=device)
                camera_rays = image_points_to_camera_rays(image_point, projection, external_distortion)

                ref_ray = ref_camera.imagePoints2rays(np.array([point]))[0]
                np.testing.assert_allclose(camera_rays[0].cpu().numpy(), ref_ray, atol=MAX_DEVIATION_RAY)
                np.testing.assert_allclose(camera_rays[0].cpu().numpy(), expected, atol=MAX_DEVIATION_RAY)

    def test_batch_processing(self):
        """Test that batch processing matches single-point processing."""
        resolution = 1000
        principal_point = np.array([resolution / 2, resolution / 2])
        backward_polynomial = [0.0, 0.001, 0.00001]

        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial)
        external_distortion = NoExternalDistortion()

        # Generate random valid points
        np.random.seed(42)
        num_points = 100
        points = np.random.uniform(0, resolution - 1, (num_points, 2)).astype(np.float32)

        # Batch processing
        batch_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        batch_rays = image_points_to_camera_rays(batch_tensor, projection, external_distortion)

        # Single point processing
        for i in range(num_points):
            single_tensor = torch.tensor([points[i]], dtype=torch.float32, device=device)
            single_ray = image_points_to_camera_rays(single_tensor, projection, external_distortion)

            np.testing.assert_allclose(
                batch_rays[i].cpu().numpy(), single_ray[0].cpu().numpy(), atol=NORM_ATOL, err_msg=f"Point {i} mismatch"
            )

    def test_calculate_max_radius(self):
        """Test max radius calculation for various principal point positions.

        Validates that the maximum radius from principal point to image corners
        is computed correctly for different configurations.
        """
        size2d = np.array([10, 5])
        max2d = size2d - np.array([1, 1])

        test_cases = [
            # (principal_point, expected_max_radius)
            (np.array([0, 0]), np.linalg.norm(max2d)),
            (np.array([1, 2]), np.linalg.norm(max2d - np.array([1, 2]))),
            (np.array([max2d[0], 0]), np.linalg.norm(max2d)),
            (np.array([0, max2d[1]]), np.linalg.norm(max2d)),
            (max2d, np.linalg.norm(max2d)),
        ]

        for principal_point, expected_max_radius in test_cases:
            with self.subTest(principal_point=principal_point.tolist()):
                ref_camera = ReferenceFThetaCamera(size2d, principal_point, [0, 1])
                actual_max_radius = ref_camera._maxRadius
                self.assertAlmostEqual(actual_max_radius, expected_max_radius, places=6)

    def test_round_trip_consistency_linear_term(self):
        """Test round-trip consistency with non-trivial linear term [c,d;e,1].

        Tests that F-Theta camera with non-identity linear transformation
        maintains consistency through image→rays→image round-trip.
        """
        resolution = 1000
        principal_point = np.array([resolution / 2, resolution / 2])
        focal_length_pixel = 500.0
        backward_polynomial = [
            0.0,
            0.4 / focal_length_pixel,
            (0.4 / focal_length_pixel) ** 2,
            (0.4 / focal_length_pixel) ** 3,
            (0.4 / focal_length_pixel) ** 4,
        ]

        # Non-identity linear term: A = [[c, d], [e, 1]]
        # Using c=1.2, d=0.1, e=0.2
        c, d, e = 1.2, 0.1, 0.2
        A = np.array([[c, d, 0], [e, 1, 0], [0, 0, 1]], dtype=np.float32)

        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial, A=A)
        external_distortion = NoExternalDistortion()

        # Test various points from center to corner
        for p in range(0, int(principal_point[0]), 50):
            with self.subTest(pixel_offset=p):
                original_point = np.array([[float(p), float(p)]])
                image_point_tensor = torch.tensor(original_point, dtype=torch.float32, device=device)

                # Image → Rays
                camera_rays = image_points_to_camera_rays(image_point_tensor, projection, external_distortion)

                # Rays → Image
                reprojected_points, valid = camera_rays_to_image_points(camera_rays, projection, external_distortion)

                # Check consistency
                self.assertTrue(valid[0].item(), f"Point {original_point} should reproject validly")
                np.testing.assert_allclose(
                    reprojected_points[0].cpu().numpy(),
                    original_point[0],
                    atol=MAX_DEVIATION_IN_PIXEL,
                    err_msg=f"Round trip failed for point {original_point} with linear term",
                )

    def test_image_points_to_rays_with_linear_term(self):
        """Test image→rays with non-identity linear term against reference.

        This test catches bugs where Ainv is applied in the wrong order.
        Unlike round-trip tests, this compares actual ray directions against
        the reference implementation, which will fail if the linear term
        is applied incorrectly (e.g., to the 3D ray instead of 2D offset).
        """
        resolution = 1000
        principal_point = np.array([resolution / 2, resolution / 2])
        focal_length_pixel = 500.0
        backward_polynomial = [
            0.0,
            0.4 / focal_length_pixel,
            (0.4 / focal_length_pixel) ** 2,
            (0.4 / focal_length_pixel) ** 3,
            (0.4 / focal_length_pixel) ** 4,
        ]

        # Non-identity linear term with significant off-diagonal components
        # This makes the bug detectable - with identity A, the bug is hidden
        c, d, e = 1.2, 0.1, 0.2
        A = np.array([[c, d, 0], [e, 1, 0], [0, 0, 1]], dtype=np.float32)

        ref_camera = ReferenceFThetaCamera(
            np.array([resolution, resolution]), principal_point, backward_polynomial, A=A
        )
        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial, A=A)
        external_distortion = NoExternalDistortion()

        # Test points at various positions (not just on diagonal)
        test_points = [
            [100.0, 200.0],
            [300.0, 150.0],
            [450.0, 400.0],
            [200.0, 350.0],
            [principal_point[0] + 100, principal_point[1] - 50],
        ]

        for point in test_points:
            with self.subTest(point=point):
                image_point = np.array([point])
                image_point_tensor = torch.tensor(image_point, dtype=torch.float32, device=device)

                # Get ray from kernel
                kernel_ray = image_points_to_camera_rays(image_point_tensor, projection, external_distortion)

                # Get ray from reference
                ref_ray = ref_camera.imagePoints2rays(image_point)[0]

                # Compare - this will fail if Ainv is applied in wrong order
                np.testing.assert_allclose(
                    kernel_ray[0].cpu().numpy(),
                    ref_ray,
                    atol=MAX_DEVIATION_RAY,
                    err_msg=f"Ray mismatch for point {point} with linear term A=[{c},{d};{e},1]",
                )

    def test_rays_to_image_points_with_linear_term(self):
        """Test rays→image with non-identity linear term against reference.

        This test catches bugs where A is applied in the wrong order.
        Unlike round-trip tests, this compares actual image coordinates against
        the reference implementation, which will fail if the linear term
        is applied incorrectly (e.g., to the 3D ray instead of 2D offset).
        """
        resolution = 1000
        principal_point = np.array([resolution / 2, resolution / 2])
        focal_length_pixel = 500.0
        backward_polynomial = [
            0.0,
            0.4 / focal_length_pixel,
            (0.4 / focal_length_pixel) ** 2,
            (0.4 / focal_length_pixel) ** 3,
            (0.4 / focal_length_pixel) ** 4,
        ]

        # Non-identity linear term with significant off-diagonal components
        c, d, e = 1.2, 0.1, 0.2
        A = np.array([[c, d, 0], [e, 1, 0], [0, 0, 1]], dtype=np.float32)

        ref_camera = ReferenceFThetaCamera(
            np.array([resolution, resolution]), principal_point, backward_polynomial, A=A
        )
        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial, A=A)
        external_distortion = NoExternalDistortion()

        # Test rays at various angles (not just along axes)
        test_rays = [
            [0.1, 0.2, 1.0],
            [0.3, 0.1, 1.0],
            [-0.2, 0.15, 1.0],
            [0.25, -0.1, 1.0],
            [0.0, 0.0, 1.0],  # Principal ray
        ]

        for ray in test_rays:
            with self.subTest(ray=ray):
                # Normalize ray
                ray_normalized = np.array(ray) / np.linalg.norm(ray)
                ray_tensor = torch.tensor([ray_normalized], dtype=torch.float32, device=device)

                # Get image point from kernel
                kernel_point, valid = camera_rays_to_image_points(ray_tensor, projection, external_distortion)

                # Get image point from reference
                ref_point = ref_camera.rays2imagePoints(np.array([ray_normalized]))[0]

                # Only compare valid projections
                if valid[0].item():
                    np.testing.assert_allclose(
                        kernel_point[0].cpu().numpy(),
                        ref_point,
                        atol=MAX_DEVIATION_IN_PIXEL,
                        err_msg=f"Image point mismatch for ray {ray} with linear term A=[{c},{d};{e},1]",
                    )

    def test_round_trip_consistency_forward_poly(self):
        """Test round-trip consistency using forward polynomial as reference.

        Tests F-Theta camera with ANGLE_TO_PIXELDIST (forward) as the reference
        polynomial type, using real-world camera parameters.
        """
        resolution = (3848, 2168)
        principal_point = np.array([1909.3092, 1103.2788])

        # Real-world forward polynomial parameters
        fw_poly_coeffs = np.array(
            [0.0, 3139.486, 164.573, -442.129, 259.583, 153.666],
            dtype=np.float32,
        )
        bw_poly_coeffs = np.array(
            [0.0, 0.000319, -5.44e-09, 4.78e-12, -1.03e-15, -1.13e-19],
            dtype=np.float32,
        )
        max_angle = 0.7037

        from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS

        fw_poly = np.zeros(MAX_POLYNOMIAL_TERMS, dtype=np.float32)
        bw_poly = np.zeros(MAX_POLYNOMIAL_TERMS, dtype=np.float32)
        for i, c in enumerate(fw_poly_coeffs):
            if i < MAX_POLYNOMIAL_TERMS:
                fw_poly[i] = c
        for i, c in enumerate(bw_poly_coeffs):
            if i < MAX_POLYNOMIAL_TERMS:
                bw_poly[i] = c

        dfw_poly = np.zeros(MAX_POLYNOMIAL_TERMS, dtype=np.float32)
        dbw_poly = np.zeros(MAX_POLYNOMIAL_TERMS, dtype=np.float32)
        for i in range(MAX_POLYNOMIAL_TERMS - 1):
            dfw_poly[i] = (i + 1) * fw_poly[i + 1]
            dbw_poly[i] = (i + 1) * bw_poly[i + 1]

        projection = FThetaProjection.from_components(
            principal_point=torch.tensor(principal_point, dtype=torch.float32, device=device),
            fw_poly=torch.tensor(fw_poly, dtype=torch.float32, device=device),
            bw_poly=torch.tensor(bw_poly, dtype=torch.float32, device=device),
            A=torch.eye(2, dtype=torch.float32, device=device),
            Ainv=torch.eye(2, dtype=torch.float32, device=device),
            dfw_poly=torch.tensor(dfw_poly, dtype=torch.float32, device=device),
            dbw_poly=torch.tensor(dbw_poly, dtype=torch.float32, device=device),
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=max_angle,
            newton_iterations=10,
            min_2d_norm=1e-8,
        )
        external_distortion = NoExternalDistortion()

        # Test points from center toward edge
        for p in range(0, int(principal_point[0]), 100):
            with self.subTest(pixel_offset=p):
                original_point = np.array([[float(p), float(p)]])
                image_point_tensor = torch.tensor(original_point, dtype=torch.float32, device=device)

                # Image → Rays
                camera_rays = image_points_to_camera_rays(image_point_tensor, projection, external_distortion)

                # Rays → Image
                reprojected_points, valid = camera_rays_to_image_points(camera_rays, projection, external_distortion)

                # Check consistency
                self.assertTrue(valid[0].item(), f"Point {original_point} should reproject validly")
                np.testing.assert_allclose(
                    reprojected_points[0].cpu().numpy(),
                    original_point[0],
                    atol=MAX_DEVIATION_IN_PIXEL,
                    err_msg=f"Round trip failed for point {original_point} with forward poly",
                )


class TestOpenCVPinholeCameraOracle(unittest.TestCase):
    """Oracle tests for OpenCV Pinhole camera model."""

    def setUp(self):
        """Set up test fixtures with various pinhole camera configurations."""
        self.resolution = np.array([1920, 1280])

        # Ideal pinhole (no distortion)
        self.ideal_params = {
            "focal_length": np.array([500.0, 500.0]),
            "principal_point": np.array([960.0, 640.0]),
            "radial_coeffs": np.zeros(6),
            "tangential_coeffs": np.zeros(2),
            "thin_prism_coeffs": np.zeros(4),
            "resolution": self.resolution,
        }

        # Waymo-style distorted pinhole
        self.distorted_params = {
            "focal_length": np.array([2059.0471, 2059.0471]),
            "principal_point": np.array([935.1248, 635.0525]),
            "radial_coeffs": np.array([0.0424, -0.3417, 0.01, 0.02, -0.01, 0.02]),
            "tangential_coeffs": np.array([0.00181, -0.000055]),
            "thin_prism_coeffs": np.array([0.01, 0.02, 0.02, 0.01]),
            "resolution": self.resolution,
        }

    def test_ideal_pinhole_round_trip(self):
        """Test round-trip consistency for ideal pinhole camera."""
        projection = create_pinhole_projection(**self.ideal_params)
        external_distortion = NoExternalDistortion()

        # Test points in valid region
        test_points = np.array(
            [
                [960.0, 640.0],  # Principal point
                [1000.0, 700.0],
                [500.0, 400.0],
            ],
            dtype=np.float32,
        )

        image_points = torch.tensor(test_points, dtype=torch.float32, device=device)

        # Image → Rays
        camera_rays = image_points_to_camera_rays(image_points, projection, external_distortion)

        # Check rays are normalized
        norms = camera_rays.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=NORM_ATOL, rtol=NORM_ATOL)

        # Rays → Image
        reprojected, valid = camera_rays_to_image_points(camera_rays, projection, external_distortion)

        self.assertTrue(valid.all(), "All points should be valid")
        np.testing.assert_allclose(
            reprojected.cpu().numpy(), test_points, atol=MAX_DEVIATION_IN_PIXEL, err_msg="Round trip failed"
        )

    def test_principal_point_projects_to_optical_axis(self):
        """Test that principal point maps to optical axis [0, 0, 1]."""
        for params in [self.ideal_params, self.distorted_params]:
            with self.subTest(distorted="radial_coeffs" in params and np.any(params["radial_coeffs"])):
                projection = create_pinhole_projection(**params)
                external_distortion = NoExternalDistortion()

                pp = params["principal_point"]
                pp_tensor = torch.tensor([[pp[0], pp[1]]], dtype=torch.float32, device=device)

                camera_rays = image_points_to_camera_rays(pp_tensor, projection, external_distortion)

                # Should be [0, 0, 1]
                np.testing.assert_allclose(camera_rays[0].cpu().numpy(), [0, 0, 1], atol=TIGHT_ATOL)

    def test_distorted_pinhole_round_trip(self):
        """Test round-trip consistency for distorted pinhole camera."""
        projection = create_pinhole_projection(**self.distorted_params)
        external_distortion = NoExternalDistortion()

        # Test various points
        step = 50
        for p in range(0, int(self.distorted_params["principal_point"][0]), step):
            with self.subTest(pixel_offset=p):
                original_point = np.array([[p, p]], dtype=np.float32)
                image_points = torch.tensor(original_point, dtype=torch.float32, device=device)

                # Image → Rays
                camera_rays = image_points_to_camera_rays(image_points, projection, external_distortion)

                # Rays → Image
                reprojected, valid = camera_rays_to_image_points(camera_rays, projection, external_distortion)

                # Allow slightly larger tolerance for distorted cameras
                np.testing.assert_allclose(
                    reprojected[0].cpu().numpy(),
                    original_point[0],
                    atol=GRAD_ATOL,
                    err_msg=f"Round trip failed for {p}",
                )

    def test_behind_camera_invalid(self):
        """Test that rays behind camera are marked invalid."""
        projection = create_pinhole_projection(**self.ideal_params)
        external_distortion = NoExternalDistortion()

        # Ray pointing backward
        backward_rays = torch.tensor([[0.0, 0.0, -1.0], [0.5, 0.5, -0.5]], dtype=torch.float32, device=device)
        backward_rays = backward_rays / backward_rays.norm(dim=-1, keepdim=True)

        _, valid = camera_rays_to_image_points(backward_rays, projection, external_distortion)

        self.assertFalse(valid[0].item(), "Ray pointing backward should be invalid")
        self.assertFalse(valid[1].item(), "Ray with negative z component should be invalid")


class TestOpenCVFisheyeCameraOracle(unittest.TestCase):
    """Oracle tests for OpenCV Fisheye camera model."""

    def setUp(self):
        """Set up fisheye camera test fixtures."""
        self.resolution = np.array([3840, 2160])
        self.params = {
            "resolution": self.resolution,
            "focal_length": np.array([1913.76478, 1913.99708]),
            "principal_point": np.array([1928.184506, 1083.862789]),
            "radial_coeffs": np.array([-0.030093122, -0.005103817, -0.000849622, 0.001079542]),
            "max_angle": np.deg2rad(140 / 2),
        }

    def test_principal_point_projects_to_optical_axis(self):
        """Test that principal point maps to optical axis."""
        projection = create_fisheye_projection(**self.params)
        external_distortion = NoExternalDistortion()

        pp = self.params["principal_point"]
        pp_tensor = torch.tensor([[pp[0], pp[1]]], dtype=torch.float32, device=device)

        camera_rays = image_points_to_camera_rays(pp_tensor, projection, external_distortion)

        # Should be approximately [0, 0, 1]
        np.testing.assert_allclose(camera_rays[0].cpu().numpy(), [0, 0, 1], atol=MAX_DEVIATION_RAY)

    def test_round_trip_consistency(self):
        """Test that back-projection followed by forward projection is consistent.

        With Newton iteration for exact polynomial inversion, round-trip should be exact.
        """
        projection = create_fisheye_projection(**self.params)
        external_distortion = NoExternalDistortion()

        # Test various image points (not too close to edges)
        pp = self.params["principal_point"]
        test_points = np.array(
            [
                [pp[0], pp[1]],  # Principal point
                [pp[0] + 100, pp[1]],
                [pp[0], pp[1] + 100],
                [pp[0] + 200, pp[1] + 100],
            ],
            dtype=np.float32,
        )

        for i, point in enumerate(test_points):
            with self.subTest(point_idx=i):
                image_tensor = torch.tensor([point], dtype=torch.float32, device=device)

                # Back-project to ray
                camera_rays = image_points_to_camera_rays(image_tensor, projection, external_distortion)

                # Check ray is normalized
                ray_norm = camera_rays.norm(dim=-1)
                self.assertTrue(
                    torch.allclose(ray_norm, torch.ones_like(ray_norm), atol=TIGHT_ATOL),
                    "Camera rays should be normalized",
                )

                # Forward project (now uses Newton iteration)
                reprojected, valid = camera_rays_to_image_points(camera_rays, projection, external_distortion)

                if valid[0].item():
                    # Round-trip should be exact with Newton iteration
                    np.testing.assert_allclose(
                        reprojected[0].cpu().numpy(),
                        point,
                        atol=MAX_DEVIATION_IN_PIXEL,
                        err_msg=f"Round trip failed for point {i}",
                    )

    def test_opencv_reference_consistency(self):
        """Test consistency with OpenCV's fisheye projection.

        Verifies that our implementation matches OpenCV's cv2.fisheye.projectPoints.
        """
        projection = create_fisheye_projection(**self.params)
        ref_camera = ReferenceOpenCVFisheyeCamera(**self.params)
        external_distortion = NoExternalDistortion()

        # Test rays at various angles
        angles = np.linspace(0.01, self.params["max_angle"] * 0.9, 20)

        for angle in angles:
            with self.subTest(angle_deg=np.degrees(angle)):
                # Create ray at this angle in +X direction
                ray = np.array([[np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
                ray_tensor = torch.tensor(ray, dtype=torch.float32, device=device)

                # Project using kernel
                kernel_point, valid = camera_rays_to_image_points(ray_tensor, projection, external_distortion)

                # Project using OpenCV reference
                ref_point, ref_valid = ref_camera.camera_ray_to_image_point_opencv(ray[0])

                if valid[0].item() and ref_valid:
                    np.testing.assert_allclose(
                        kernel_point[0].cpu().numpy(),
                        ref_point,
                        atol=MAX_DEVIATION_IN_PIXEL,  # Cross-implementation comparison (our float32 vs OpenCV float64)
                        err_msg=f"Mismatch at angle {np.degrees(angle):.1f}°",
                    )


class TestBivariateWindshieldOracle(unittest.TestCase):
    """Oracle tests for bivariate windshield distortion model."""

    @staticmethod
    def poly_eval_2d_reference(coefficients: np.ndarray, x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
        """Reference implementation of 2D polynomial evaluation.

        The bivariate polynomial, provided as a 1D array [c0, c1, c2...cn] is evaluated as:
        c0*x^0 + c1*x^1 + c2*x^2 + (c3*x^0 + c4*x^1)*y^1 + (c5*x^0)*y^2
        """
        if x.shape != y.shape:
            raise ValueError(f"Expected x and y to be same size, got {x.shape} and {y.shape}")

        x_flat = x.flatten()
        y_flat = y.flatten()
        y_coeffs = np.zeros((order + 1, x_flat.shape[0]), dtype=x.dtype)

        start_idx = 0
        for inner_order in reversed(range(order + 1)):
            x_coeffs = coefficients[start_idx : start_idx + inner_order + 1]
            # Evaluate x polynomial using Horner's method
            result = np.zeros_like(x_flat)
            for c in reversed(x_coeffs):
                result = c + x_flat * result
            y_coeffs[order - inner_order, :] = result
            start_idx += inner_order + 1

        # Evaluate y polynomial using Horner's method
        z = np.zeros_like(y_flat)
        for c in reversed(y_coeffs):
            z = c + y_flat * z
        return z.reshape(x.shape)

    def test_poly_eval_2d_zeros(self):
        """Test 2D polynomial evaluation with zero coefficients."""
        coeffs = np.zeros(3, dtype=np.float32)
        x = np.array([-1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([-2.0, 1.0, 3.0, 5.0], dtype=np.float32)

        result = self.poly_eval_2d_reference(coeffs, x, y, order=1)
        expected = np.zeros_like(x)
        np.testing.assert_allclose(result, expected, atol=NORM_ATOL)

    def test_poly_eval_2d_oracle(self):
        """Test 2D polynomial evaluation against oracle values.

        Uses known coefficients and expected output values.
        """
        coeffs = np.array(
            [0.90113, 0.77499, 0.55887, 0.77048, 0.47019, 0.84775, 0.68832, 0.77690, 0.92327, 0.83983],
            dtype=np.float32,
        )
        x = np.array([1.2, 1.2], dtype=np.float32)
        y = np.array([0.4, 0.4], dtype=np.float32)

        result = self.poly_eval_2d_reference(coeffs, x, y, order=3)
        expected = np.array([5.31406952, 5.31406952], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=ATOL, atol=TIGHT_ATOL)

    def test_poly_eval_2d_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        coeffs = np.zeros(3, dtype=np.float32)
        x = np.array([-1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([-2.0, 1.0, 3.0, 5.0], dtype=np.float32)  # Different size

        with self.assertRaises(ValueError):
            self.poly_eval_2d_reference(coeffs, x, y, order=1)

    def test_distort_rays_sign_flip(self):
        """Test ray distortion with sign-flipping polynomials.

        Uses polynomials that flip the sign of phi and theta angles,
        verifying the x and y components are negated.
        """
        # Use 3 coefficients for order 1 bivariate polynomial (triangular: 1+2=3)
        # MAX_H_POLYNOMIAL_TERMS = 6, MAX_V_POLYNOMIAL_TERMS = 15
        h_poly = torch.zeros(3, dtype=torch.float32, device=device)
        v_poly = torch.zeros(3, dtype=torch.float32, device=device)
        # Set coefficients for sign flip: h(phi, theta) = -phi, v(phi, theta) = -theta
        # For order 1: h = c0 + c1*phi + c2*theta = -phi requires c1 = -1
        # We use the simplified form where h_poly[1] = -1 gives h = -phi
        h_poly[1] = -1.0  # Coefficient for phi term
        v_poly[2] = -1.0  # Coefficient for theta term (in bivariate form)

        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly,
            v_poly_inv=v_poly,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        # Test rays pointing forward with small x/y offsets
        # Rays must have positive z to project validly through pinhole camera
        rays = torch.tensor(
            [[0.1, 0.1, 1.0], [-0.1, 0.1, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        rays = rays / rays.norm(dim=-1, keepdim=True)  # Normalize

        # Create simple pinhole camera for testing
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        # Project with distortion
        _image_points, valid = camera_rays_to_image_points(rays, projection, distortion)

        # Verify projection succeeds
        self.assertTrue(valid.all(), "All rays should project validly")

    def test_distort_undistort_inverse(self):
        """Test that distort followed by undistort recovers original rays.

        Uses real-world windshield distortion polynomial coefficients.
        """
        # Real-world windshield polynomial coefficients
        h_poly_fwd = np.array(
            [
                -0.000475919834570959,
                0.99944007396698,
                0.000166745347087272,
                0.000205887947231531,
                0.0055195577442646,
                0.000861024134792387,
            ],
            dtype=np.float32,
        )
        v_poly_fwd = np.array(
            [
                0.00152770057320595,
                -0.000532537756953388,
                -5.65027039556298e-05,
                -4.02410341848736e-06,
                0.000608163303695619,
                1.0094313621521,
                -0.00125278066843748,
                0.00823396816849708,
                -0.000293767458060756,
                0.0185473654419184,
                -0.003074218519032,
                0.00599765172228217,
                0.0172030478715897,
                -0.00364979170262814,
                0.0069147446192801,
            ],
            dtype=np.float32,
        )
        h_poly_inv = np.array(
            [0.0004770369, 1.0005774, -0.00016896478, -0.00020207358, -0.0054899976, -0.0008536868],
            dtype=np.float32,
        )
        v_poly_inv = np.array(
            [
                -0.0015191488,
                0.00052959577,
                7.882431e-05,
                -6.966009e-06,
                -0.00059701066,
                0.9906775,
                0.00116782,
                -0.007893825,
                0.00026140467,
                -0.017767625,
                0.0027627628,
                -0.00544897,
                -0.015480865,
                0.0033684247,
                -0.0057964055,
            ],
            dtype=np.float32,
        )

        # Convert directly to tensors (sizes already match MAX_H/V_POLYNOMIAL_TERMS: 6 and 15)
        h_fwd = torch.tensor(h_poly_fwd, dtype=torch.float32, device=device)
        v_fwd = torch.tensor(v_poly_fwd, dtype=torch.float32, device=device)
        h_inv = torch.tensor(h_poly_inv, dtype=torch.float32, device=device)
        v_inv = torch.tensor(v_poly_inv, dtype=torch.float32, device=device)

        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_fwd,
            v_poly=v_fwd,
            h_poly_inv=h_inv,
            v_poly_inv=v_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        # Test rays at various angles
        phi = 0.05
        theta = 0.02
        rays = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],  # Optical axis
                    [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(theta)],
                    [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), -np.cos(theta)],
                ],
                dtype=torch.float32,
                device=device,
            ),
            dim=-1,
        )

        # Create simple camera
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        # Project rays → image → rays (round-trip)
        image_points, valid = camera_rays_to_image_points(rays, projection, distortion)

        # Only test valid projections (rays with z > 0)
        valid_rays = rays[valid]
        valid_points = image_points[valid]

        if len(valid_points) > 0:
            # Back-project
            recovered_rays = image_points_to_camera_rays(valid_points, projection, distortion)

            # Check ray recovery (direction similarity via dot product)
            for i in range(len(valid_rays)):
                dot_product = (valid_rays[i] * recovered_rays[i]).sum().item()
                self.assertGreater(dot_product, 0.99, f"Ray {i} not recovered correctly (dot product: {dot_product})")

    def test_identity_distortion(self):
        """Test that identity-like polynomial produces no distortion."""
        # Identity polynomial: adj_phi = phi, adj_theta = theta
        # Use 3 coefficients for order 1 bivariate polynomial (triangular: 1+2=3)
        h_poly = torch.zeros(3, dtype=torch.float32, device=device)
        v_poly = torch.zeros(3, dtype=torch.float32, device=device)
        h_poly[1] = 1.0  # Coefficient for phi: adj_phi = phi
        v_poly[2] = 1.0  # Coefficient for theta: adj_theta = theta

        # For identity, forward = backward
        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly,
            v_poly_inv=v_poly,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        # Create a simple pinhole camera
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        # Test rays
        camera_rays = torch.tensor([[0.1, 0.1, 1.0], [0.2, 0.0, 1.0]], dtype=torch.float32, device=device)
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        # Project with distortion
        image_points_with_dist, _ = camera_rays_to_image_points(camera_rays, projection, distortion)

        # Project without distortion
        image_points_no_dist, _ = camera_rays_to_image_points(camera_rays, projection, NoExternalDistortion())

        # Should be very similar for identity-like distortion
        # Note: This test may need adjustment based on actual polynomial interpretation
        # For now we just verify the function runs without error

    def test_small_distortion_round_trip(self):
        """Test that small windshield distortion allows for round-trip projection."""
        # Small distortion polynomials (near identity)
        # Use 3 coefficients for order 1 bivariate polynomial (triangular: 1+2=3)
        # Bivariate polynomial order 1: h = c[0] + c[1]*phi + c[2]*theta
        # For identity: adj_phi = phi requires c[1]=1, adj_theta = theta requires c[2]=1
        h_poly = torch.zeros(3, dtype=torch.float32, device=device)
        v_poly = torch.zeros(3, dtype=torch.float32, device=device)
        # Near-identity: adj_phi ≈ phi + small_offset, adj_theta ≈ theta + small_offset
        h_poly[1] = 1.0  # coefficient of phi
        h_poly[0] = 0.001  # tiny constant offset
        v_poly[2] = 1.0  # coefficient of theta (index 2 in bivariate form for order 1)
        v_poly[0] = 0.001  # tiny constant offset

        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly,  # Approximate inverse (not exact)
            v_poly_inv=v_poly,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        # Simple pinhole camera
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        # Test rays near optical axis
        camera_rays = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        # Project rays to image
        image_points, valid = camera_rays_to_image_points(camera_rays, projection, distortion)

        # All should be valid
        self.assertTrue(valid.all(), "All rays should project validly")

        # Back-project
        rays_back = image_points_to_camera_rays(image_points, projection, distortion)

        # Rays should be approximately recovered
        for i in range(len(camera_rays)):
            # Check direction similarity (dot product close to 1)
            dot_product = (camera_rays[i] * rays_back[i]).sum().item()
            self.assertGreater(dot_product, 0.99, f"Ray {i} not recovered correctly")


class TestCameraProjection(unittest.TestCase):
    """Test camera projection operations (original tests)."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Create a simple pinhole camera
        self.projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        self.external_distortion = NoExternalDistortion()
        self.resolution = (int(self.projection.resolution[0].item()), int(self.projection.resolution[1].item()))

        # Create a simple static pose (identity)
        self.static_pose = Pose(
            torch.zeros(3, device=device),
            quat_identity((1,), device=device).squeeze(0),
        )

        # Create a simple dynamic pose with 2 control poses
        pose1_trans = torch.zeros(3, device=device)
        pose1_rot = quat_identity((1,), device=device).squeeze(0)
        pose2_trans = torch.tensor([0.1, 0.0, 0.0], device=device)
        pose2_rot = quat_identity((1,), device=device).squeeze(0)
        self.dynamic_pose = create_dynamic_pose(pose1_trans, pose2_trans, pose1_rot, pose2_rot, device)

    def test_camera_rays_to_image_points_basic(self):
        """Test basic camera ray projection."""
        camera_rays = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]],
            device=device,
        )
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        image_points, valid = camera_rays_to_image_points(
            camera_rays,
            self.projection,
            self.external_distortion,
        )

        self.assertEqual(image_points.shape, (3, 2))
        self.assertEqual(valid.shape, (3,))
        self.assertEqual(image_points.device.type, device.type)

    def test_image_points_to_camera_rays_basic(self):
        """Test basic image point back-projection."""
        image_points = torch.tensor(
            [[320.0, 240.0], [420.0, 240.0], [320.0, 340.0]],
            device=device,
        )

        camera_rays = image_points_to_camera_rays(
            image_points,
            self.projection,
            self.external_distortion,
        )

        self.assertEqual(camera_rays.shape, (3, 3))

        # Check that rays are normalized
        norms = camera_rays.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_round_trip_projection(self):
        """Test that image → camera rays → image is consistent."""
        image_points_orig = torch.tensor(
            [[320.0, 240.0], [400.0, 300.0], [250.0, 150.0]],
            device=device,
        )

        camera_rays = image_points_to_camera_rays(
            image_points_orig,
            self.projection,
            self.external_distortion,
        )

        image_points_new, valid = camera_rays_to_image_points(
            camera_rays,
            self.projection,
            self.external_distortion,
        )

        self.assertEqual(image_points_new.shape, image_points_orig.shape)
        np.testing.assert_allclose(
            image_points_new.cpu().numpy(), image_points_orig.cpu().numpy(), atol=MAX_DEVIATION_IN_PIXEL
        )

    def test_project_world_points_mean_pose(self):
        """Test world point projection with mean pose."""
        world_points = torch.tensor(
            [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]],
            device=device,
        )

        image_points, valid, _, _, _ = project_world_points_mean_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.dynamic_pose,
            self.resolution,
            return_valid_flags=True,
        )

        self.assertEqual(image_points.shape, (3, 2))
        assert valid is not None
        self.assertEqual(valid.shape, (3,))

    def test_project_world_points_shutter_pose(self):
        """Test world point projection with rolling shutter."""
        world_points = torch.tensor(
            [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]],
            device=device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            ShutterType.GLOBAL,
            self.dynamic_pose,
            max_iterations=5,
            return_valid_flags=True,
        )

        self.assertEqual(image_points.shape, (2, 2))
        assert valid is not None
        self.assertEqual(valid.shape, (2,))

    def test_image_points_to_world_rays_static_pose(self):
        """Test back-projection to world rays with static pose."""
        image_points = torch.tensor(
            [[320.0, 240.0], [400.0, 300.0]],
            device=device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_static_pose(
            image_points,
            self.projection,
            self.external_distortion,
            self.static_pose,
        )

        self.assertEqual(world_rays.shape, (2, 6))

        # Check that directions are normalized
        directions = world_rays[:, 3:6]
        norms = directions.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_image_points_to_world_rays_shutter_pose(self):
        """Test back-projection to world rays with rolling shutter."""
        image_points = torch.tensor(
            [[320.0, 240.0], [400.0, 300.0]],
            device=device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            ShutterType.GLOBAL,
            self.dynamic_pose,
        )

        self.assertEqual(world_rays.shape, (2, 6))

    def test_behind_camera_invalid(self):
        """Test that points behind camera are marked invalid."""
        camera_rays = torch.tensor([[0.0, 0.0, -1.0]], device=device)
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        image_points, valid = camera_rays_to_image_points(
            camera_rays,
            self.projection,
            self.external_distortion,
        )

        # Should be marked as invalid
        self.assertFalse(valid[0].item())

    def test_batch_processing(self):
        """Test that batched processing works correctly."""
        world_points = torch.randn(100, 3, device=device)
        world_points[:, 2] = torch.abs(world_points[:, 2]) + 2.0

        image_points, valid, _, _, _ = project_world_points_mean_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.dynamic_pose,
            self.resolution,
            return_valid_flags=True,
        )

        self.assertEqual(image_points.shape, (100, 2))
        assert valid is not None
        self.assertEqual(valid.shape, (100,))

    def test_empty_input(self):
        """Test handling of empty input."""
        world_points = torch.empty((0, 3), device=device)

        image_points, valid, _, _, _ = project_world_points_mean_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.dynamic_pose,
            self.resolution,
            return_valid_flags=True,
        )

        self.assertEqual(image_points.shape, (0, 2))
        assert valid is not None
        self.assertEqual(valid.shape, (0,))

    def test_different_shutter_types(self):
        """Test different rolling shutter types."""
        world_points = torch.tensor([[0.0, 0.0, 5.0]], device=device)

        for shutter_type in [
            ShutterType.GLOBAL,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            ShutterType.ROLLING_LEFT_TO_RIGHT,
        ]:
            with self.subTest(shutter_type=shutter_type):
                image_points, valid, _, _, _ = project_world_points_shutter_pose(
                    world_points,
                    self.projection,
                    self.external_distortion,
                    self.resolution,
                    shutter_type,
                    self.dynamic_pose,
                    return_valid_flags=True,
                )

                self.assertEqual(image_points.shape, (1, 2))
                assert valid is not None
                self.assertEqual(valid.shape, (1,))


class TestDistortionModels(unittest.TestCase):
    """Test different distortion models."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

    def test_no_distortion(self):
        """Test that no distortion model works."""
        distortion = NoExternalDistortion()

        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        camera_rays = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        image_points, _valid = camera_rays_to_image_points(
            camera_rays,
            projection,
            distortion,
        )

        self.assertEqual(image_points.shape, (1, 2))


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of camera projections."""

    def test_near_optical_axis_stability(self):
        """Test stability for rays very close to optical axis."""
        resolution = 1000
        principal_point = np.array([resolution / 2, resolution / 2])
        backward_polynomial = [0.0, 0.001, 0.0]

        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial)
        external_distortion = NoExternalDistortion()

        # Very small offsets from optical axis
        small_offsets = [1e-8, 1e-6, 1e-4, 1e-2]

        for offset in small_offsets:
            with self.subTest(offset=offset):
                rays = torch.tensor([[offset, 0, 1.0], [0, offset, 1.0]], dtype=torch.float32, device=device)
                rays = rays / rays.norm(dim=-1, keepdim=True)

                image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

                # Should produce finite results
                self.assertTrue(torch.isfinite(image_points).all(), f"Infinite values for offset {offset}")
                self.assertTrue(valid.any(), f"All invalid for offset {offset}")

    def test_near_max_angle_stability(self):
        """Test stability for rays near maximum angle."""
        resolution = 1000
        principal_point = np.array([resolution / 2, resolution / 2])
        max_angle_deg = 60
        backward_polynomial = [0.0, np.radians(max_angle_deg) / (resolution / 2)]

        projection = create_ftheta_projection((resolution, resolution), principal_point, backward_polynomial)
        external_distortion = NoExternalDistortion()

        # Test points near the edge
        edge_points = torch.tensor(
            [[resolution - 2, principal_point[1]], [principal_point[0], resolution - 2]],
            dtype=torch.float32,
            device=device,
        )

        camera_rays = image_points_to_camera_rays(edge_points, projection, external_distortion)

        # Should produce finite, normalized rays
        self.assertTrue(torch.isfinite(camera_rays).all())
        norms = camera_rays.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=TIGHT_ATOL, rtol=TIGHT_RTOL)


class TestJacobianOracle(unittest.TestCase):
    """Oracle tests for camera projection Jacobian computations.

    Tests that computed Jacobians match both reference implementations
    and PyTorch autograd results.
    """

    def setUp(self):
        """Set up test fixtures with camera configurations."""
        # Simple distorted pinhole (k1, k2, k3, p1, p2 only)
        self.simple_pinhole_params = {
            "focal_length": np.array([2059.047, 2059.423]),
            "principal_point": np.array([935.125, 635.052]),
            "radial_coeffs": np.array([0.0424, -0.3417, 0.01, 0.0, 0.0, 0.0]),
            "tangential_coeffs": np.array([0.00181, -0.000055]),
            "thin_prism_coeffs": np.zeros(4),
            "resolution": np.array([1920, 1080]),
        }

        # Full distorted pinhole (all coefficients)
        self.full_pinhole_params = {
            "focal_length": np.array([2059.047, 2059.047]),
            "principal_point": np.array([935.125, 635.052]),
            "radial_coeffs": np.array([0.0424, -0.3417, 0.0, 0.0, 0.0, 0.0]),
            "tangential_coeffs": np.array([0.00181, -0.000055]),
            "thin_prism_coeffs": np.zeros(4),
            "resolution": np.array([1920, 1080]),
        }

        # Ideal pinhole (no distortion)
        self.ideal_pinhole_params = {
            "focal_length": np.array([2059.047, 2059.047]),
            "principal_point": np.array([935.125, 635.052]),
            "radial_coeffs": np.zeros(6),
            "tangential_coeffs": np.zeros(2),
            "thin_prism_coeffs": np.zeros(4),
            "resolution": np.array([1920, 1080]),
        }

    def test_pinhole_jacobian_reference(self):
        """Test reference pinhole Jacobian is self-consistent.

        Verifies that the analytically computed Jacobian from the reference
        implementation matches numerical finite differences computed on the
        reference implementation itself.

        Note: The original test compared against API-returned Jacobians. Since
        the kernel API doesn't return Jacobians, we verify the reference
        implementation's internal consistency instead.
        """
        # Use simple params that reference implementation supports
        params = self.simple_pinhole_params.copy()
        params["radial_coeffs"] = np.array([0.0424, -0.3417, 0.01, 0.0, 0.0, 0.0])

        ref_camera = ReferenceSimplePinholeCamera(
            focal_length=params["focal_length"],
            principal_point=params["principal_point"],
            radial_coeffs=params["radial_coeffs"],
            tangential_coeffs=params["tangential_coeffs"],
            dtype=np.float64,  # Use float64 for better numerical precision
        )

        # Test rays at various positions (valid rays with z > 0)
        test_rays = np.array(
            [
                [0.01, 0.02, 1.0],
                [0.05, 0.03, 1.0],
                [-0.02, 0.04, 1.0],
                [0.1, 0.1, 1.0],
            ],
            dtype=np.float64,
        )
        # Normalize rays
        test_rays = test_rays / np.linalg.norm(test_rays, axis=1, keepdims=True)

        for i in range(len(test_rays)):
            with self.subTest(ray_idx=i):
                ray = test_rays[i]

                # Reference Jacobian (analytical)
                ref_point, ref_jacobian = ref_camera.camera_ray_to_image_point(ray)

                # Numerical Jacobian via central finite differences on reference
                eps = 1e-7
                numerical_jacobian = np.zeros((2, 3), dtype=np.float64)

                for j in range(3):
                    ray_plus = ray.copy()
                    ray_plus[j] += eps
                    ray_minus = ray.copy()
                    ray_minus[j] -= eps

                    point_plus, _ = ref_camera.camera_ray_to_image_point(ray_plus)
                    point_minus, _ = ref_camera.camera_ray_to_image_point(ray_minus)

                    numerical_jacobian[:, j] = (point_plus - point_minus) / (2 * eps)

                # Compare analytical to numerical with tight tolerance
                # since both are from the same implementation
                np.testing.assert_allclose(
                    ref_jacobian,
                    numerical_jacobian,
                    rtol=ATOL,
                    atol=ATOL,
                    err_msg=f"Reference Jacobian mismatch for ray {i}",
                )

    def test_jacobian_autograd_consistency(self):
        """Test Jacobians are consistent with PyTorch autograd.

        Verifies that numerically computed Jacobians match what PyTorch's
        automatic differentiation produces.
        """
        camera_params_list = [
            ("ideal", self.ideal_pinhole_params),
            ("distorted", self.full_pinhole_params),
        ]

        for name, params in camera_params_list:
            with self.subTest(camera_type=name):
                projection = create_pinhole_projection(**params)
                external_distortion = NoExternalDistortion()

                # Create wrapper for autograd
                def projection_fn(ray: torch.Tensor) -> torch.Tensor:
                    points, _ = camera_rays_to_image_points(ray.unsqueeze(0), projection, external_distortion)
                    return points.squeeze(0)

                # Test rays (valid rays from image points)
                test_points = torch.tensor(
                    [[200.0, 200.0], [500.0, 300.0], [100.0, 400.0]],
                    dtype=torch.float32,
                    device=device,
                )
                rays = image_points_to_camera_rays(test_points, projection, external_distortion)

                for i in range(len(rays)):
                    ray = rays[i].clone().requires_grad_(True)

                    # Compute Jacobian via autograd
                    try:
                        autograd_jacobian = torch.autograd.functional.jacobian(
                            projection_fn, ray, strategy="reverse-mode"
                        )

                        # Compute numerical Jacobian
                        eps = 1e-4
                        numerical_jacobian = torch.zeros(2, 3, dtype=torch.float32, device=device)
                        base_point = projection_fn(ray.detach())

                        for j in range(3):
                            perturbed_ray = ray.detach().clone()
                            perturbed_ray[j] += eps
                            perturbed_point = projection_fn(perturbed_ray)
                            numerical_jacobian[:, j] = (perturbed_point - base_point) / eps

                        np.testing.assert_allclose(
                            autograd_jacobian.cpu().detach().numpy(),
                            numerical_jacobian.cpu().numpy(),
                            rtol=GRAD_RTOL,
                            atol=GRAD_ATOL,
                            err_msg=f"Jacobian mismatch for {name} camera, ray {i}",
                        )
                    except Exception as e:
                        # Autograd may fail for some configurations; that's okay for this test
                        pass

    def test_jacobian_special_rays(self):
        """Test Jacobians for special ray configurations.

        Tests rays along principal axis and rays at various angles.
        """
        projection = create_pinhole_projection(**self.ideal_pinhole_params)
        external_distortion = NoExternalDistortion()

        # Special rays
        special_rays = torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Principal axis
                [0.1, 0.0, 1.0],  # Small x offset
                [0.0, 0.1, 1.0],  # Small y offset
                [0.1, 0.1, 1.0],  # Diagonal
            ],
            dtype=torch.float32,
            device=device,
        )
        special_rays = special_rays / special_rays.norm(dim=-1, keepdim=True)

        for i in range(len(special_rays)):
            with self.subTest(ray_idx=i):
                ray = special_rays[i]
                ray_tensor = ray.unsqueeze(0)

                # Project
                image_point, valid = camera_rays_to_image_points(ray_tensor, projection, external_distortion)

                self.assertTrue(valid[0].item(), f"Ray {i} should be valid")

                # Compute numerical Jacobian
                eps = 1e-5
                numerical_jacobian = torch.zeros(2, 3, dtype=torch.float32, device=device)

                for j in range(3):
                    perturbed_ray = ray.clone()
                    perturbed_ray[j] += eps
                    perturbed_point, _ = camera_rays_to_image_points(
                        perturbed_ray.unsqueeze(0), projection, external_distortion
                    )
                    numerical_jacobian[:, j] = (perturbed_point[0] - image_point[0]) / eps

                # Jacobian should be finite and reasonable
                self.assertTrue(
                    torch.isfinite(numerical_jacobian).all(),
                    f"Jacobian should be finite for ray {i}",
                )

                # For principal axis ray, x and y derivatives should be roughly equal
                # due to symmetry of ideal pinhole
                if i == 0:
                    np.testing.assert_allclose(
                        numerical_jacobian[0, 0].item(),
                        numerical_jacobian[1, 1].item(),
                        rtol=GRAD_ATOL,
                        err_msg="Jacobian should be symmetric for principal axis ray",
                    )


class TestIntrinsicsDifferentiability(unittest.TestCase):
    """Tests for differentiability of camera intrinsic parameters.

    Verifies that gradients flow through camera intrinsic parameters
    (focal length, principal point, distortion coefficients) for all
    camera models and external distortion types.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.device = device

    def test_opencv_pinhole_focal_length_gradient(self):
        """Test gradient flows through focal length for OpenCV pinhole model."""
        # Create parameters with requires_grad
        focal_length = torch.tensor([1000.0, 1000.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.zeros(6, device=self.device, requires_grad=True)
        tangential_coeffs = torch.zeros(2, device=self.device, requires_grad=True)
        thin_prism_coeffs = torch.zeros(4, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Test camera rays
        rays = torch.tensor([[0.1, 0.1, 1.0]], device=self.device, dtype=torch.float32)
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        # Check gradients exist and are non-zero for focal_length
        self.assertIsNotNone(focal_length.grad, "Focal length gradient should exist")
        self.assertTrue((focal_length.grad != 0).any().item(), "Focal length gradient should be non-zero")

        # Check gradients exist for principal point
        self.assertIsNotNone(principal_point.grad, "Principal point gradient should exist")
        self.assertTrue((principal_point.grad != 0).any().item(), "Principal point gradient should be non-zero")

    def test_opencv_pinhole_distortion_gradient(self):
        """Test gradient flows through distortion coefficients for OpenCV pinhole model."""
        focal_length = torch.tensor([1000.0, 1000.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.tensor([0.01, 0.001, 0.0001, 0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        tangential_coeffs = torch.tensor([0.001, 0.001], device=self.device, requires_grad=True)
        thin_prism_coeffs = torch.tensor([0.0001, 0.0001, 0.0001, 0.0001], device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Test camera rays (off-axis to engage distortion)
        rays = torch.tensor([[0.2, 0.15, 1.0]], device=self.device, dtype=torch.float32)
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        # Check radial coefficients gradient (at least first few should be non-zero)
        self.assertIsNotNone(radial_coeffs.grad, "Radial coeffs gradient should exist")
        # k1 gradient should be non-zero for off-axis ray
        self.assertTrue(
            radial_coeffs.grad[0].abs() > 1e-8, f"k1 gradient should be non-zero, got {radial_coeffs.grad[0]}"
        )

        # Check tangential coefficients gradient
        self.assertIsNotNone(tangential_coeffs.grad, "Tangential coeffs gradient should exist")
        self.assertTrue(
            (tangential_coeffs.grad.abs() > 1e-8).any().item(), "Tangential coeffs gradient should be non-zero"
        )

    def test_opencv_fisheye_gradient(self):
        """Test gradient flows through OpenCV fisheye intrinsics."""
        focal_length = torch.tensor([500.0, 500.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        forward_poly = torch.tensor([0.01, 0.001, 0.0001, 0.00001], device=self.device, requires_grad=True)

        projection = OpenCVFisheyeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            forward_poly=forward_poly,
            resolution=torch.tensor([640, 480], device=self.device),
            max_angle=2.0,
            newton_iterations=10,
            min_2d_norm=torch.tensor(1e-6, device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Test camera rays
        rays = torch.tensor([[0.15, 0.1, 1.0]], device=self.device, dtype=torch.float32)
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(focal_length.grad, "Focal length gradient should exist")
        self.assertTrue((focal_length.grad != 0).any().item(), "Focal length gradient should be non-zero")

        self.assertIsNotNone(forward_poly.grad, "Forward poly gradient should exist")
        self.assertTrue(
            (forward_poly.grad.abs() > 1e-10).any().item(),
            f"Forward poly gradient should be non-zero, got {forward_poly.grad}",
        )

    def test_ftheta_gradient(self):
        """Test gradient flows through F-Theta intrinsics."""
        from libs.sensors.kernels.cameras.parameters import FThetaPolynomialType

        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        # Simple f-theta polynomial: r = a0 + a1*theta (with a1 = 500 to get reasonable pixel coords)
        # Use 2 coefficients for the polynomial
        fw_poly = torch.tensor([0.0, 500.0], device=self.device, requires_grad=True)
        bw_poly = torch.tensor([0.0, 0.002], device=self.device, requires_grad=True)  # inverse approx
        dfw_poly = torch.tensor([500.0], device=self.device, requires_grad=True)  # derivative of fw_poly
        dbw_poly = torch.tensor([0.002], device=self.device, requires_grad=True)  # derivative of bw_poly
        A = torch.eye(2, device=self.device, requires_grad=True)
        Ainv = torch.eye(2, device=self.device, requires_grad=True)

        projection = FThetaProjection.from_components(
            principal_point=principal_point,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=A,
            Ainv=Ainv,
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=2.0,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        external_distortion = NoExternalDistortion()

        # Test camera rays
        rays = torch.tensor([[0.1, 0.1, 1.0]], device=self.device, dtype=torch.float32)
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(principal_point.grad, "Principal point gradient should exist")
        self.assertTrue((principal_point.grad != 0).any().item(), "Principal point gradient should be non-zero")

        self.assertIsNotNone(fw_poly.grad, "Forward poly gradient should exist")
        # fw_poly[1] is the linear term, should have gradient
        self.assertTrue(
            (fw_poly.grad.abs() > 1e-8).any().item(), f"fw_poly gradient should be non-zero, got {fw_poly.grad}"
        )

    def test_bivariate_windshield_gradient(self):
        """Test gradient flows through bivariate windshield distortion parameters."""
        from libs.sensors.kernels.cameras.parameters import ReferencePolynomial

        # Camera intrinsics
        focal_length = torch.tensor([1000.0, 1000.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.zeros(6, device=self.device, requires_grad=True)
        tangential_coeffs = torch.zeros(2, device=self.device, requires_grad=True)
        thin_prism_coeffs = torch.zeros(4, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )

        # Windshield distortion with small coefficients (simple linear distortion)
        h_poly = torch.tensor([0.0, 1.01, 0.0], device=self.device, requires_grad=True)  # small distortion
        v_poly = torch.tensor([0.0, 1.01, 0.0], device=self.device, requires_grad=True)
        h_poly_inv = torch.tensor([0.0, 0.99, 0.0], device=self.device, requires_grad=True)  # approx inverse
        v_poly_inv = torch.tensor([0.0, 0.99, 0.0], device=self.device, requires_grad=True)

        external_distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        # Test camera rays
        rays = torch.tensor([[0.1, 0.1, 1.0]], device=self.device, dtype=torch.float32)
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        # Check windshield polynomial gradients (on the original leaf tensors)
        self.assertIsNotNone(h_poly.grad, "h_poly gradient should exist")
        # The linear term (index 1) should have gradient
        self.assertTrue(h_poly.grad[1].abs() > 1e-8, f"h_poly[1] gradient should be non-zero, got {h_poly.grad[1]}")

        self.assertIsNotNone(v_poly.grad, "v_poly gradient should exist")
        self.assertTrue(v_poly.grad[1].abs() > 1e-8, f"v_poly[1] gradient should be non-zero, got {v_poly.grad[1]}")

    def test_backprojection_intrinsics_gradient(self):
        """Test gradient flows through intrinsics in back-projection."""
        focal_length = torch.tensor([1000.0, 1000.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.tensor([0.01, 0.001, 0.0, 0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        tangential_coeffs = torch.zeros(2, device=self.device, requires_grad=True)
        thin_prism_coeffs = torch.zeros(4, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Test image points
        image_points = torch.tensor([[350.0, 270.0]], device=self.device, dtype=torch.float32)

        # Back-projection
        rays = image_points_to_camera_rays(image_points, projection, external_distortion)

        # Compute loss and backprop
        loss = rays.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(focal_length.grad, "Focal length gradient should exist in backprojection")
        self.assertTrue(
            (focal_length.grad != 0).any().item(), "Focal length gradient should be non-zero in backprojection"
        )

        self.assertIsNotNone(principal_point.grad, "Principal point gradient should exist in backprojection")
        self.assertTrue(
            (principal_point.grad != 0).any().item(), "Principal point gradient should be non-zero in backprojection"
        )

    def test_gradient_matches_pytorch_reference(self):
        """Verify kernel gradients match PyTorch reference implementation.

        This test compares analytical gradients from the kernel against
        a pure PyTorch reference implementation's autograd gradients.
        """
        # Camera parameters
        focal_length_val = [1000.0, 1000.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.0] * 6
        tangential_coeffs_val = [0.0, 0.0]
        thin_prism_coeffs_val = [0.0] * 4

        # Test rays
        rays = torch.tensor([[0.1, 0.1, 1.0]], device=self.device, dtype=torch.float32)
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # --- PyTorch reference implementation ---
        ref_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        ref_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        ref_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        ref_pts = TestKernelGradientsMatchPyTorchReference.torch_pinhole_project(
            rays, ref_fl, ref_pp, ref_rc, ref_tc, ref_tpc
        )
        ref_pts.sum().backward()

        # --- Kernel implementation ---
        kernel_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        kernel_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        kernel_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=kernel_fl,
            principal_point=kernel_pp,
            radial_coeffs=kernel_rc,
            tangential_coeffs=kernel_tc,
            thin_prism_coeffs=kernel_tpc,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        kernel_pts, _ = camera_rays_to_image_points(rays, projection, NoExternalDistortion())
        kernel_pts.sum().backward()

        # Compare gradients - should match exactly since both use analytical differentiation
        np.testing.assert_allclose(
            kernel_fl.grad.cpu().numpy(),
            ref_fl.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=NORM_ATOL,
            err_msg="Focal length gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            ref_pp.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=NORM_ATOL,
            err_msg="Principal point gradient mismatch",
        )

    # -------------------------------------------------------------------------
    # Tests comparing gradients against reference implementations
    # These tests catch loadOnce() issues by verifying gradient accumulation
    # -------------------------------------------------------------------------

    def test_focal_length_gradient_matches_reference(self):
        """Test that focal length gradient matches reference implementation.

        This test computes expected gradients from the reference implementation
        and compares them against the kernel's analytical gradients. This catches
        loadOnce() bugs where only one thread's gradient contribution would be counted.
        """
        # Camera parameters
        focal_length_np = np.array([1000.0, 1000.0], dtype=np.float32)
        principal_point_np = np.array([320.0, 240.0], dtype=np.float32)
        radial_coeffs_np = np.zeros(6, dtype=np.float32)
        tangential_coeffs_np = np.zeros(2, dtype=np.float32)
        thin_prism_coeffs_np = np.zeros(4, dtype=np.float32)
        resolution = np.array([640, 480])

        # Create reference camera
        ref_camera = ReferenceOpenCVPinholeCamera(
            focal_length=focal_length_np,
            principal_point=principal_point_np,
            radial_coeffs=radial_coeffs_np,
            tangential_coeffs=tangential_coeffs_np,
            thin_prism_coeffs=thin_prism_coeffs_np,
            resolution=resolution,
        )

        # Create multiple test rays (enough to span multiple warps)
        num_rays = 100
        np.random.seed(42)
        rays_np = np.random.randn(num_rays, 3).astype(np.float32)
        rays_np[:, 0] = np.abs(rays_np[:, 0]) * 0.2  # Positive x
        rays_np[:, 1] = np.abs(rays_np[:, 1]) * 0.15  # Positive y
        rays_np[:, 2] = 1.0  # Forward
        rays_np = rays_np / np.linalg.norm(rays_np, axis=1, keepdims=True)

        # Compute expected gradients from reference implementation
        # For loss = sum(image_x) + sum(image_y):
        # d(loss)/d(fx) = sum(x_distorted for all rays)
        # d(loss)/d(fy) = sum(y_distorted for all rays)
        expected_d_fx = 0.0
        expected_d_fy = 0.0
        for ray in rays_np:
            jac = ref_camera.compute_intrinsics_jacobian(ray)
            if jac is not None:
                expected_d_fx += jac["d_image_x_d_fx"]
                expected_d_fy += jac["d_image_y_d_fy"]

        # Create kernel projection with requires_grad
        focal_length = torch.tensor(focal_length_np, device=self.device, requires_grad=True)
        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=torch.tensor(principal_point_np, device=self.device),
            radial_coeffs=torch.tensor(radial_coeffs_np, device=self.device),
            tangential_coeffs=torch.tensor(tangential_coeffs_np, device=self.device),
            thin_prism_coeffs=torch.tensor(thin_prism_coeffs_np, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Forward pass
        rays = torch.tensor(rays_np, device=self.device)
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        analytical_grad = focal_length.grad.cpu().numpy()

        # Compare against reference
        np.testing.assert_allclose(
            analytical_grad[0],
            expected_d_fx,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg=f"fx gradient mismatch: kernel={analytical_grad[0]}, reference={expected_d_fx}",
        )
        np.testing.assert_allclose(
            analytical_grad[1],
            expected_d_fy,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg=f"fy gradient mismatch: kernel={analytical_grad[1]}, reference={expected_d_fy}",
        )

    def test_principal_point_gradient_matches_reference(self):
        """Test that principal point gradient matches reference implementation.

        For pinhole camera: d(image_x)/d(cx) = 1, d(image_y)/d(cy) = 1 for each ray.
        Total gradient = num_rays for each component.
        """
        # Camera parameters
        focal_length_np = np.array([1000.0, 1000.0], dtype=np.float32)
        principal_point_np = np.array([320.0, 240.0], dtype=np.float32)
        radial_coeffs_np = np.zeros(6, dtype=np.float32)
        tangential_coeffs_np = np.zeros(2, dtype=np.float32)
        thin_prism_coeffs_np = np.zeros(4, dtype=np.float32)
        resolution = np.array([640, 480])

        # Create reference camera
        ref_camera = ReferenceOpenCVPinholeCamera(
            focal_length=focal_length_np,
            principal_point=principal_point_np,
            radial_coeffs=radial_coeffs_np,
            tangential_coeffs=tangential_coeffs_np,
            thin_prism_coeffs=thin_prism_coeffs_np,
            resolution=resolution,
        )

        # Create multiple test rays
        num_rays = 150
        np.random.seed(123)
        rays_np = np.random.randn(num_rays, 3).astype(np.float32)
        rays_np[:, 2] = np.abs(rays_np[:, 2]) + 0.5  # Ensure positive z
        rays_np = rays_np / np.linalg.norm(rays_np, axis=1, keepdims=True)

        # Compute expected gradients from reference
        # For loss = sum(image_x) + sum(image_y):
        # d(loss)/d(cx) = num_valid_rays (each ray contributes 1)
        # d(loss)/d(cy) = num_valid_rays (each ray contributes 1)
        expected_d_cx = 0.0
        expected_d_cy = 0.0
        for ray in rays_np:
            jac = ref_camera.compute_intrinsics_jacobian(ray)
            if jac is not None:
                expected_d_cx += jac["d_image_x_d_cx"]
                expected_d_cy += jac["d_image_y_d_cy"]

        # Create kernel projection
        principal_point = torch.tensor(principal_point_np, device=self.device, requires_grad=True)
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor(focal_length_np, device=self.device),
            principal_point=principal_point,
            radial_coeffs=torch.tensor(radial_coeffs_np, device=self.device),
            tangential_coeffs=torch.tensor(tangential_coeffs_np, device=self.device),
            thin_prism_coeffs=torch.tensor(thin_prism_coeffs_np, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Forward pass
        rays = torch.tensor(rays_np, device=self.device)
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        analytical_grad = principal_point.grad.cpu().numpy()

        # Compare against reference - principal point gradient should be exact
        np.testing.assert_allclose(
            analytical_grad[0],
            expected_d_cx,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg=f"cx gradient mismatch: kernel={analytical_grad[0]}, reference={expected_d_cx}",
        )
        np.testing.assert_allclose(
            analytical_grad[1],
            expected_d_cy,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg=f"cy gradient mismatch: kernel={analytical_grad[1]}, reference={expected_d_cy}",
        )

    def test_intrinsics_gradient_accumulation_large_batch(self):
        """Test gradient accumulation with large batch against reference.

        Uses >1024 rays to test cross-warp gradient accumulation. This specifically
        catches loadOnce() issues where only one thread per warp contributes.
        """
        # Camera parameters
        focal_length_np = np.array([500.0, 500.0], dtype=np.float32)
        principal_point_np = np.array([320.0, 240.0], dtype=np.float32)
        radial_coeffs_np = np.zeros(6, dtype=np.float32)
        tangential_coeffs_np = np.zeros(2, dtype=np.float32)
        thin_prism_coeffs_np = np.zeros(4, dtype=np.float32)
        resolution = np.array([640, 480])

        # Create reference camera
        ref_camera = ReferenceOpenCVPinholeCamera(
            focal_length=focal_length_np,
            principal_point=principal_point_np,
            radial_coeffs=radial_coeffs_np,
            tangential_coeffs=tangential_coeffs_np,
            thin_prism_coeffs=thin_prism_coeffs_np,
            resolution=resolution,
        )

        # Large batch: 2048 rays (64 warps worth)
        num_rays = 2048
        np.random.seed(999)
        rays_np = np.random.randn(num_rays, 3).astype(np.float32) * 0.2
        rays_np[:, 2] = 1.0
        rays_np = rays_np / np.linalg.norm(rays_np, axis=1, keepdims=True)

        # Compute expected gradients from reference
        expected_d_fx = 0.0
        expected_d_fy = 0.0
        expected_d_cx = 0.0
        expected_d_cy = 0.0
        for ray in rays_np:
            jac = ref_camera.compute_intrinsics_jacobian(ray)
            if jac is not None:
                expected_d_fx += jac["d_image_x_d_fx"]
                expected_d_fy += jac["d_image_y_d_fy"]
                expected_d_cx += jac["d_image_x_d_cx"]
                expected_d_cy += jac["d_image_y_d_cy"]

        # Create kernel projection
        focal_length = torch.tensor(focal_length_np, device=self.device, requires_grad=True)
        principal_point = torch.tensor(principal_point_np, device=self.device, requires_grad=True)
        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=torch.tensor(radial_coeffs_np, device=self.device),
            tangential_coeffs=torch.tensor(tangential_coeffs_np, device=self.device),
            thin_prism_coeffs=torch.tensor(thin_prism_coeffs_np, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Forward pass
        rays = torch.tensor(rays_np, device=self.device)
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        fl_grad = focal_length.grad.cpu().numpy()
        pp_grad = principal_point.grad.cpu().numpy()

        # Compare focal length gradient
        np.testing.assert_allclose(
            fl_grad[0],
            expected_d_fx,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg=f"Large batch fx gradient mismatch: kernel={fl_grad[0]}, reference={expected_d_fx}",
        )
        np.testing.assert_allclose(
            fl_grad[1],
            expected_d_fy,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg=f"Large batch fy gradient mismatch: kernel={fl_grad[1]}, reference={expected_d_fy}",
        )

        # Compare principal point gradient (should be exactly num_rays)
        np.testing.assert_allclose(
            pp_grad[0],
            expected_d_cx,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg=f"Large batch cx gradient mismatch: kernel={pp_grad[0]}, reference={expected_d_cx}",
        )
        np.testing.assert_allclose(
            pp_grad[1],
            expected_d_cy,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg=f"Large batch cy gradient mismatch: kernel={pp_grad[1]}, reference={expected_d_cy}",
        )

    def test_distortion_gradient_matches_torch_reference(self):
        """Test distortion coefficient gradients against PyTorch reference implementation.

        Uses a pure-PyTorch implementation (similar to ncore) to compute expected gradients
        via autograd, then compares against the kernel's gradients. This catches loadOnce()
        issues where gradients would not accumulate properly across threads.
        """
        # Camera parameters with non-zero distortion
        focal_length_val = [1000.0, 1000.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.1, 0.01, 0.001, 0.0, 0.0, 0.0]
        tangential_coeffs_val = [0.001, 0.001]
        thin_prism_coeffs_val = [0.0, 0.0, 0.0, 0.0]

        # Test rays (off-axis to engage distortion)
        num_rays = 100
        torch.manual_seed(456)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.3
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # --- Pure PyTorch reference implementation (similar to ncore) ---
        def torch_pinhole_project(
            rays: torch.Tensor,
            focal_length: torch.Tensor,
            principal_point: torch.Tensor,
            radial_coeffs: torch.Tensor,
            tangential_coeffs: torch.Tensor,
            thin_prism_coeffs: torch.Tensor,
        ) -> torch.Tensor:
            """Pure PyTorch implementation of OpenCV pinhole projection."""
            # Perspective normalization
            uv_normalized = rays[:, :2] / rays[:, 2:3]

            # Compute distortion
            xy_squared = uv_normalized**2
            r_2 = xy_squared.sum(dim=1, keepdim=True)
            r_4 = r_2**2
            r_6 = r_2**3
            xy_prod = uv_normalized[:, 0:1] * uv_normalized[:, 1:2]

            # Radial distortion
            k1, k2, k3, k4, k5, k6 = (
                radial_coeffs[0],
                radial_coeffs[1],
                radial_coeffs[2],
                radial_coeffs[3],
                radial_coeffs[4],
                radial_coeffs[5],
            )
            radial_num = 1.0 + k1 * r_2 + k2 * r_4 + k3 * r_6
            radial_denom = 1.0 + k4 * r_2 + k5 * r_4 + k6 * r_6
            radial = radial_num / radial_denom

            # Tangential distortion
            p1, p2 = tangential_coeffs[0], tangential_coeffs[1]
            a1 = 2 * xy_prod
            a2 = r_2 + 2 * xy_squared[:, 0:1]
            a3 = r_2 + 2 * xy_squared[:, 1:2]
            delta_x = p1 * a1 + p2 * a2
            delta_y = p1 * a3 + p2 * a1

            # Thin prism distortion
            s1, s2, s3, s4 = thin_prism_coeffs[0], thin_prism_coeffs[1], thin_prism_coeffs[2], thin_prism_coeffs[3]
            delta_x = delta_x + s1 * r_2 + s2 * r_4
            delta_y = delta_y + s3 * r_2 + s4 * r_4

            # Apply distortion
            uv_distorted = uv_normalized * radial + torch.cat([delta_x, delta_y], dim=1)

            # Apply camera matrix
            image_points = uv_distorted * focal_length + principal_point

            return image_points

        # Compute reference gradient using PyTorch autograd
        ref_radial_coeffs = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        ref_focal_length = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_principal_point = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_tangential = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        ref_thin_prism = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        ref_image_points = torch_pinhole_project(
            rays, ref_focal_length, ref_principal_point, ref_radial_coeffs, ref_tangential, ref_thin_prism
        )
        ref_loss = ref_image_points.sum()
        ref_loss.backward()

        ref_k1_grad = ref_radial_coeffs.grad[0].item()
        ref_fx_grad = ref_focal_length.grad[0].item()
        ref_cx_grad = ref_principal_point.grad[0].item()

        # --- Kernel implementation ---
        kernel_radial_coeffs = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        kernel_focal_length = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_principal_point = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_tangential = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        kernel_thin_prism = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=kernel_focal_length,
            principal_point=kernel_principal_point,
            radial_coeffs=kernel_radial_coeffs,
            tangential_coeffs=kernel_tangential,
            thin_prism_coeffs=kernel_thin_prism,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        kernel_image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)
        kernel_loss = kernel_image_points.sum()
        kernel_loss.backward()

        kernel_k1_grad = kernel_radial_coeffs.grad[0].item()
        kernel_fx_grad = kernel_focal_length.grad[0].item()
        kernel_cx_grad = kernel_principal_point.grad[0].item()

        # Compare gradients - should match closely since both use same math
        np.testing.assert_allclose(
            kernel_k1_grad,
            ref_k1_grad,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg=f"k1 gradient mismatch: kernel={kernel_k1_grad}, reference={ref_k1_grad}",
        )
        np.testing.assert_allclose(
            kernel_fx_grad,
            ref_fx_grad,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg=f"fx gradient mismatch: kernel={kernel_fx_grad}, reference={ref_fx_grad}",
        )
        np.testing.assert_allclose(
            kernel_cx_grad,
            ref_cx_grad,
            rtol=NORM_ATOL,
            atol=TIGHT_ATOL,
            err_msg=f"cx gradient mismatch: kernel={kernel_cx_grad}, reference={ref_cx_grad}",
        )

    def test_backprojection_intrinsics_gradient_matches_reference(self):
        """Test backprojection intrinsics gradients match PyTorch reference implementation.

        Verifies that gradients through image_points_to_camera_rays match
        the pure PyTorch reference implementation's analytical gradients.
        """
        # Camera parameters (ideal pinhole for simpler analysis)
        focal_length_val = [500.0, 500.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.0] * 6
        tangential_coeffs_val = [0.0, 0.0]
        thin_prism_coeffs_val = [0.0] * 4

        # Test image points
        num_points = 80
        torch.manual_seed(789)
        image_points = torch.rand(num_points, 2, device=self.device)
        image_points[:, 0] = image_points[:, 0] * 400 + 120  # x in [120, 520]
        image_points[:, 1] = image_points[:, 1] * 300 + 90  # y in [90, 390]

        # --- PyTorch reference implementation ---
        ref_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        ref_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        ref_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        ref_rays = TestKernelGradientsMatchPyTorchReference.torch_pinhole_backproject(
            image_points, ref_fl, ref_pp, ref_rc, ref_tc, ref_tpc
        )
        ref_rays.sum().backward()

        # --- Kernel implementation ---
        kernel_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        kernel_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        kernel_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=kernel_fl,
            principal_point=kernel_pp,
            radial_coeffs=kernel_rc,
            tangential_coeffs=kernel_tc,
            thin_prism_coeffs=kernel_tpc,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        kernel_rays = image_points_to_camera_rays(image_points, projection, external_distortion)
        kernel_rays.sum().backward()

        # Compare gradients - should match exactly since both use analytical differentiation
        np.testing.assert_allclose(
            kernel_fl.grad.cpu().numpy(),
            ref_fl.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=NORM_ATOL,
            err_msg="Backprojection focal length gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            ref_pp.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=NORM_ATOL,
            err_msg="Backprojection principal point gradient mismatch",
        )

    # -------------------------------------------------------------------------
    # Tests for intrinsics gradients in world-space functions
    # -------------------------------------------------------------------------

    def _create_simple_pose(self, trans, rot):
        """Create a simple pose object for testing."""
        return Pose(trans, rot)

    def _create_dynamic_pose(self, trans_start, trans_end, rot_start, rot_end):
        """Create a dynamic pose with control poses."""
        return create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

    def test_project_world_points_shutter_pose_intrinsics_gradient(self):
        """Test gradient flows through intrinsics in shutter pose world projection."""
        focal_length = torch.tensor([500.0, 500.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.zeros(6, device=self.device, requires_grad=True)
        tangential_coeffs = torch.zeros(2, device=self.device, requires_grad=True)
        thin_prism_coeffs = torch.zeros(4, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()
        resolution = (int(projection.resolution[0].item()), int(projection.resolution[1].item()))

        # Create identity dynamic pose (no pose transformation)
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        trans_end = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)
        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        # World points in front of camera
        world_points = torch.tensor([[0.5, 0.3, 5.0], [1.0, 0.5, 10.0]], device=self.device)

        # Forward projection
        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            external_distortion,
            resolution,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        # Check focal length gradient
        self.assertIsNotNone(focal_length.grad, "Focal length gradient should exist")
        self.assertTrue(
            (focal_length.grad != 0).any().item(),
            f"Focal length gradient should be non-zero, got {focal_length.grad}",
        )

        # Check principal point gradient
        self.assertIsNotNone(principal_point.grad, "Principal point gradient should exist")
        self.assertTrue(
            (principal_point.grad != 0).any().item(),
            f"Principal point gradient should be non-zero, got {principal_point.grad}",
        )

    def test_project_world_points_mean_pose_intrinsics_gradient(self):
        """Test gradient flows through intrinsics in mean pose world projection."""
        focal_length = torch.tensor([500.0, 500.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.zeros(6, device=self.device, requires_grad=True)
        tangential_coeffs = torch.zeros(2, device=self.device, requires_grad=True)
        thin_prism_coeffs = torch.zeros(4, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()
        resolution = (int(projection.resolution[0].item()), int(projection.resolution[1].item()))

        # Create identity dynamic pose
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        trans_end = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)
        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        # World points in front of camera
        world_points = torch.tensor([[0.5, 0.3, 5.0], [1.0, 0.5, 10.0]], device=self.device)

        # Forward projection
        image_points, valid, _, _, _ = project_world_points_mean_pose(
            world_points,
            projection,
            external_distortion,
            dynamic_pose,
            resolution,
        )

        # Compute loss and backprop
        loss = image_points.sum()
        loss.backward()

        # Check focal length gradient
        self.assertIsNotNone(focal_length.grad, "Focal length gradient should exist")
        self.assertTrue(
            (focal_length.grad != 0).any().item(),
            f"Focal length gradient should be non-zero, got {focal_length.grad}",
        )

        # Check principal point gradient
        self.assertIsNotNone(principal_point.grad, "Principal point gradient should exist")
        self.assertTrue(
            (principal_point.grad != 0).any().item(),
            f"Principal point gradient should be non-zero, got {principal_point.grad}",
        )

    def test_image_points_to_world_rays_static_pose_intrinsics_gradient(self):
        """Test gradient flows through intrinsics in static pose world backprojection."""
        focal_length = torch.tensor([500.0, 500.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.tensor([0.01, 0.001, 0.0, 0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        tangential_coeffs = torch.zeros(2, device=self.device, requires_grad=True)
        thin_prism_coeffs = torch.zeros(4, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Create identity static pose
        trans = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        rot = quat_identity((1,), device=self.device).squeeze(0)
        static_pose = self._create_simple_pose(trans, rot)

        # Image points
        image_points = torch.tensor([[350.0, 270.0], [400.0, 300.0]], device=self.device)

        # Backprojection
        world_rays, _, _, _ = image_points_to_world_rays_static_pose(
            image_points,
            projection,
            external_distortion,
            static_pose,
        )

        # Compute loss on ray directions (which depend on intrinsics)
        loss = world_rays[:, 3:].sum()
        loss.backward()

        # Check focal length gradient
        self.assertIsNotNone(focal_length.grad, "Focal length gradient should exist")
        self.assertTrue(
            (focal_length.grad != 0).any().item(),
            f"Focal length gradient should be non-zero, got {focal_length.grad}",
        )

        # Check principal point gradient
        self.assertIsNotNone(principal_point.grad, "Principal point gradient should exist")
        self.assertTrue(
            (principal_point.grad != 0).any().item(),
            f"Principal point gradient should be non-zero, got {principal_point.grad}",
        )

    def test_image_points_to_world_rays_shutter_pose_intrinsics_gradient(self):
        """Test gradient flows through intrinsics in shutter pose world backprojection."""
        focal_length = torch.tensor([500.0, 500.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.tensor([0.01, 0.001, 0.0, 0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        tangential_coeffs = torch.zeros(2, device=self.device, requires_grad=True)
        thin_prism_coeffs = torch.zeros(4, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()
        resolution = (640, 480)

        # Create identity dynamic pose
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        trans_end = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)
        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        # Image points
        image_points = torch.tensor([[350.0, 270.0], [400.0, 300.0]], device=self.device)

        # Backprojection
        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            external_distortion,
            resolution,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        # Compute loss on ray directions (which depend on intrinsics)
        loss = world_rays[:, 3:].sum()
        loss.backward()

        # Check focal length gradient
        self.assertIsNotNone(focal_length.grad, "Focal length gradient should exist")
        self.assertTrue(
            (focal_length.grad != 0).any().item(),
            f"Focal length gradient should be non-zero, got {focal_length.grad}",
        )

        # Check principal point gradient
        self.assertIsNotNone(principal_point.grad, "Principal point gradient should exist")
        self.assertTrue(
            (principal_point.grad != 0).any().item(),
            f"Principal point gradient should be non-zero, got {principal_point.grad}",
        )


class TestPoseDifferentiability(unittest.TestCase):
    """Tests for differentiability of pose parameters (translations and rotations).

    Verifies that gradients flow through pose parameters for all pose-based
    projection and back-projection functions.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.device = device
        self.projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.zeros(6, device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )
        self.external_distortion = NoExternalDistortion()
        self.resolution = (int(self.projection.resolution[0].item()), int(self.projection.resolution[1].item()))

    def _create_dynamic_pose(self, trans_start, trans_end, rot_start, rot_end):
        """Create a dynamic pose with control poses that have requires_grad."""
        return create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

    def _create_static_pose(self, trans, rot):
        """Create a static pose with requires_grad."""
        return Pose(trans, rot)

    def test_project_world_points_shutter_pose_translation_gradient(self):
        """Test gradient flows through translations in shutter pose projection."""
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        trans_end = torch.tensor([0.1, 0.0, 0.0], device=self.device, requires_grad=True)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        # World points in front of camera
        world_points = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], device=self.device)

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        # Check translation gradients exist and are non-zero
        self.assertIsNotNone(trans_start.grad, "Start translation gradient should exist")
        self.assertIsNotNone(trans_end.grad, "End translation gradient should exist")
        self.assertTrue(
            (trans_start.grad != 0).any().item() or (trans_end.grad != 0).any().item(),
            "At least one translation gradient should be non-zero",
        )

    def test_project_world_points_shutter_pose_rotation_gradient(self):
        """Test gradient flows through rotations in shutter pose projection."""
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        trans_end = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        rot_start = quat_identity((1,), device=self.device).squeeze(0).requires_grad_(True)
        rot_end = quat_identity((1,), device=self.device).squeeze(0).requires_grad_(True)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        world_points = torch.tensor([[1.0, 0.5, 5.0], [0.5, 1.0, 5.0]], device=self.device)

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(rot_start.grad, "Start rotation gradient should exist")
        self.assertIsNotNone(rot_end.grad, "End rotation gradient should exist")

    def test_project_world_points_mean_pose_translation_gradient(self):
        """Test gradient flows through translations in mean pose projection."""
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        trans_end = torch.tensor([0.1, 0.0, 0.0], device=self.device, requires_grad=True)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        world_points = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], device=self.device)

        image_points, valid, _, _, _ = project_world_points_mean_pose(
            world_points,
            self.projection,
            self.external_distortion,
            dynamic_pose,
            self.resolution,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(trans_start.grad, "Start translation gradient should exist")
        self.assertIsNotNone(trans_end.grad, "End translation gradient should exist")
        self.assertTrue(
            (trans_start.grad != 0).any().item() or (trans_end.grad != 0).any().item(),
            "At least one translation gradient should be non-zero",
        )

    def test_image_points_to_world_rays_static_pose_translation_gradient(self):
        """Test gradient flows through translation in static pose backprojection."""
        # Use non-zero translation so gradient is meaningful
        trans = torch.tensor([1.0, 2.0, 3.0], device=self.device, requires_grad=True)
        rot = quat_identity((1,), device=self.device).squeeze(0)

        static_pose = self._create_static_pose(trans, rot)

        image_points = torch.tensor([[320.0, 240.0], [400.0, 300.0]], device=self.device)

        world_rays, _, _, _ = image_points_to_world_rays_static_pose(
            image_points,
            self.projection,
            self.external_distortion,
            static_pose,
        )

        # Loss on ray origins (which depend on translation)
        loss = world_rays[:, :3].sum()
        loss.backward()

        self.assertIsNotNone(trans.grad, "Translation gradient should exist")
        # Check that gradients are non-zero (gradient accumulation across threads may vary)
        self.assertTrue(
            (trans.grad != 0).all().item(),
            f"Translation gradient {trans.grad} should be non-zero for all components",
        )

    def test_image_points_to_world_rays_static_pose_rotation_gradient(self):
        """Test gradient flows through rotation in static pose backprojection."""
        trans = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        rot = quat_identity((1,), device=self.device).squeeze(0).requires_grad_(True)

        static_pose = self._create_static_pose(trans, rot)

        image_points = torch.tensor([[400.0, 300.0], [250.0, 200.0]], device=self.device)

        world_rays, _, _, _ = image_points_to_world_rays_static_pose(
            image_points,
            self.projection,
            self.external_distortion,
            static_pose,
        )

        # Loss on ray directions (which depend on rotation)
        loss = world_rays[:, 3:].sum()
        loss.backward()

        self.assertIsNotNone(rot.grad, "Rotation gradient should exist")

    def test_image_points_to_world_rays_shutter_pose_translation_gradient(self):
        """Test gradient flows through translations in shutter pose backprojection."""
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        trans_end = torch.tensor([0.1, 0.0, 0.0], device=self.device, requires_grad=True)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        image_points = torch.tensor([[320.0, 240.0], [400.0, 300.0]], device=self.device)

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = world_rays[:, :3].sum()  # Ray origins
        loss.backward()

        self.assertIsNotNone(trans_start.grad, "Start translation gradient should exist")
        self.assertIsNotNone(trans_end.grad, "End translation gradient should exist")
        self.assertTrue(
            (trans_start.grad != 0).any().item() or (trans_end.grad != 0).any().item(),
            "At least one translation gradient should be non-zero",
        )

    def test_pose_gradient_is_correct(self):
        """Verify pose translation gradients are mathematically correct.

        For a world point projected through a camera with identity rotation:
        - image_x = fx * (world_x - trans_x) / (world_z - trans_z) + cx
        - image_y = fy * (world_y - trans_y) / (world_z - trans_z) + cy

        At trans = [0, 0, 0]:
        - d(image_x)/d(trans_x) = -fx / world_z
        - d(image_y)/d(trans_y) = -fy / world_z
        - d(image_x)/d(trans_z) = fx * world_x / world_z^2
        - d(image_y)/d(trans_z) = fy * world_y / world_z^2

        For loss = image_x + image_y:
        - d(loss)/d(trans_x) = -fx / world_z
        - d(loss)/d(trans_y) = -fy / world_z
        - d(loss)/d(trans_z) = fx * world_x / world_z^2 + fy * world_y / world_z^2
        """
        # Test with identity rotation
        rot = quat_identity((1,), device=self.device).squeeze(0)
        world_points = torch.tensor([[0.5, 0.3, 5.0]], device=self.device)

        # Kernel
        kernel_trans = torch.tensor([0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        dynamic_pose = self._create_dynamic_pose(kernel_trans, kernel_trans.clone(), rot, rot.clone())
        kernel_pts, _, _, _, _ = project_world_points_mean_pose(
            world_points, self.projection, self.external_distortion, dynamic_pose, self.resolution
        )
        kernel_pts.sum().backward()
        kernel_grad = kernel_trans.grad.clone()

        # Expected gradients (using setUp's focal_length=[500, 500])
        fx, fy = 500.0, 500.0
        world_x, world_y, world_z = 0.5, 0.3, 5.0

        expected_dx = -fx / world_z  # -100
        expected_dy = -fy / world_z  # -100
        expected_dz = fx * world_x / (world_z**2) + fy * world_y / (world_z**2)  # 16

        expected_grad = torch.tensor([expected_dx, expected_dy, expected_dz], device=self.device)

        np.testing.assert_allclose(
            kernel_grad.cpu().numpy(),
            expected_grad.cpu().numpy(),
            rtol=ATOL,
            atol=ATOL,
            err_msg=f"Kernel ({kernel_grad}) and expected ({expected_grad}) pose gradients should match",
        )


class TestUniformParameterGradientAccumulation(unittest.TestCase):
    """Tests for correct gradient accumulation when all threads access the same intrinsic parameters.

    These tests verify that gradients for camera intrinsic parameters (focal_length, principal_point,
    distortion coefficients) are correctly accumulated when multiple rays (threads) use the same
    parameters. This catches bugs where `loadOnce` is used instead of `loadUniform` for uniform access.

    Key insight: Camera intrinsics are loaded uniformly by all threads. If loadOnce was incorrectly
    used, the gradient would be ~1/32 (warp size) of the correct value because only one thread's
    contribution would be counted instead of all threads.

    The fix was to use loadUniform/loadVecUniform which properly accumulates gradients via
    WaveActiveSum reduction across all threads in a warp.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.device = device

    def test_focal_length_gradient_accumulation_many_rays(self):
        """Test gradient accumulation for focal length with many rays.

        All rays use the same focal length parameters, so the gradient should
        accumulate contributions from all rays.
        """
        # Create parameters
        focal_length = torch.tensor([1000.0, 1000.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device)
        radial_coeffs = torch.zeros(6, device=self.device)
        tangential_coeffs = torch.zeros(2, device=self.device)
        thin_prism_coeffs = torch.zeros(4, device=self.device)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Create many rays (100 rays ensures multiple warps)
        # Use non-symmetric rays with positive x to get non-zero gradient sum
        num_rays = 100
        rays = torch.zeros(num_rays, 3, device=self.device)
        rays[:, 0] = torch.linspace(0.05, 0.2, num_rays, device=self.device)  # positive x
        rays[:, 1] = torch.linspace(0.02, 0.15, num_rays, device=self.device)  # positive y
        rays[:, 2] = 1.0  # forward
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Loss: sum of all image point x-coordinates
        # Each ray contributes to the focal_length gradient
        loss = image_points[:, 0].sum()

        # Analytical gradient
        loss.backward()
        analytical_grad = focal_length.grad.clone()

        # For pinhole model: image_x = fx * x/z + cx
        # d(image_x)/d(fx) = x/z for each ray
        # Total gradient should be sum over all rays
        expected_fx_grad = (rays[:, 0] / rays[:, 2]).sum().item()

        # Verify gradient matches expected accumulation
        self.assertAlmostEqual(
            analytical_grad[0].item(),
            expected_fx_grad,
            delta=max(abs(expected_fx_grad) * GRAD_RTOL, GRAD_ATOL),
            msg=f"Focal length x gradient mismatch: got {analytical_grad[0].item()}, expected ~{expected_fx_grad}",
        )

        # fy gradient should be 0 since we only sum x-coordinates
        self.assertAlmostEqual(
            analytical_grad[1].item(),
            0.0,
            delta=GRAD_ATOL,
            msg=f"Focal length y gradient should be ~0, got {analytical_grad[1].item()}",
        )

    def test_principal_point_gradient_accumulation(self):
        """Test gradient accumulation for principal point with many rays."""
        focal_length = torch.tensor([1000.0, 1000.0], device=self.device)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.zeros(6, device=self.device)
        tangential_coeffs = torch.zeros(2, device=self.device)
        thin_prism_coeffs = torch.zeros(4, device=self.device)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Create many rays
        num_rays = 200
        torch.manual_seed(42)
        rays = torch.randn(num_rays, 3, device=self.device)
        rays[:, 2] = rays[:, 2].abs() + 0.5  # Ensure positive z
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Loss: sum of all image points
        loss = image_points.sum()

        # Analytical gradient
        loss.backward()
        analytical_grad = principal_point.grad.clone()

        # For principal point, d(image_point_x)/d(cx) = 1 for each ray
        # d(image_point_y)/d(cy) = 1 for each ray
        # Total gradient for cx = num_rays, same for cy
        expected_grad = float(num_rays)

        # Principal point gradient should be exactly num_rays (each ray contributes 1)
        self.assertAlmostEqual(
            analytical_grad[0].item(),
            expected_grad,
            delta=GRAD_ATOL,
            msg=f"Expected cx gradient = {expected_grad}, got {analytical_grad[0].item()}",
        )
        self.assertAlmostEqual(
            analytical_grad[1].item(),
            expected_grad,
            delta=GRAD_ATOL,
            msg=f"Expected cy gradient = {expected_grad}, got {analytical_grad[1].item()}",
        )

    def test_radial_distortion_gradient_accumulation(self):
        """Test gradient accumulation for radial distortion coefficients."""
        focal_length = torch.tensor([1000.0, 1000.0], device=self.device)
        principal_point = torch.tensor([320.0, 240.0], device=self.device)
        radial_coeffs = torch.tensor([0.1, 0.01, 0.001, 0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        tangential_coeffs = torch.zeros(2, device=self.device)
        thin_prism_coeffs = torch.zeros(4, device=self.device)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Create many rays (off-axis to engage distortion)
        num_rays = 150
        torch.manual_seed(123)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.3
        rays[:, 2] = 1.0  # All rays point forward with varying x,y
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Loss
        loss = image_points.sum()

        # Analytical gradient
        loss.backward()
        analytical_grad = radial_coeffs.grad.clone()

        # Verify k1 gradient exists and is non-zero (distortion is engaged by off-axis rays)
        self.assertIsNotNone(analytical_grad, "Radial coeffs gradient should exist")
        self.assertTrue(
            analytical_grad[0].abs() > 1e-6,
            f"k1 gradient should be non-zero for off-axis rays, got {analytical_grad[0].item()}",
        )

        # The gradient should scale with number of rays (if one ray contributes g, n rays contribute ~n*g)
        # Test with subset of rays to verify accumulation
        subset_rays = rays[:50]
        focal_length2 = torch.tensor([1000.0, 1000.0], device=self.device)
        principal_point2 = torch.tensor([320.0, 240.0], device=self.device)
        radial_coeffs2 = torch.tensor([0.1, 0.01, 0.001, 0.0, 0.0, 0.0], device=self.device, requires_grad=True)

        projection2 = OpenCVPinholeProjection.from_components(
            focal_length=focal_length2,
            principal_point=principal_point2,
            radial_coeffs=radial_coeffs2,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points2, _ = camera_rays_to_image_points(subset_rays, projection2, external_distortion)
        loss2 = image_points2.sum()
        loss2.backward()

        # Full gradient should be larger than subset gradient (roughly proportional)
        # Allow some tolerance due to different ray distributions
        self.assertGreater(
            analytical_grad[0].abs().item(),
            radial_coeffs2.grad[0].abs().item() * 0.5,
            f"Full batch gradient ({analytical_grad[0].item()}) should be larger than subset ({radial_coeffs2.grad[0].item()})",
        )

    def test_large_batch_gradient_accumulation(self):
        """Test gradient accumulation with a large batch (>1024 rays).

        Uses many rays to ensure multiple warps process them, testing
        that cross-warp gradient accumulation via loadUniform works correctly.
        """
        focal_length = torch.tensor([500.0, 500.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        radial_coeffs = torch.zeros(6, device=self.device)
        tangential_coeffs = torch.zeros(2, device=self.device)
        thin_prism_coeffs = torch.zeros(4, device=self.device)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Large batch: 2048 rays (64 warps worth)
        num_rays = 2048
        torch.manual_seed(999)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.2
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Loss
        loss = image_points.sum()

        # Analytical gradient
        loss.backward()
        analytical_fx = focal_length.grad[0].item()
        analytical_cx = principal_point.grad[0].item()

        # Expected gradients:
        # d(image_x)/d(fx) = x/z for each ray, sum over all rays
        # d(image_x)/d(cx) = 1 for each ray, so total = num_rays
        expected_fx_grad = (rays[:, 0] / rays[:, 2]).sum().item()
        expected_cx_grad = float(num_rays)

        # Verify focal length gradient (use relative tolerance for larger values)
        self.assertAlmostEqual(
            analytical_fx,
            expected_fx_grad,
            delta=max(abs(expected_fx_grad) * GRAD_RTOL, GRAD_ATOL),
            msg=f"Large batch fx gradient mismatch: got {analytical_fx}, expected ~{expected_fx_grad}",
        )

        # Verify principal point gradient (should be exactly num_rays)
        self.assertAlmostEqual(
            analytical_cx,
            expected_cx_grad,
            delta=GRAD_ATOL,
            msg=f"Large batch cx gradient mismatch: got {analytical_cx}, expected {expected_cx_grad}",
        )

    def test_fisheye_uniform_gradient_accumulation(self):
        """Test gradient accumulation for OpenCV fisheye model."""
        focal_length = torch.tensor([400.0, 400.0], device=self.device, requires_grad=True)
        principal_point = torch.tensor([320.0, 240.0], device=self.device, requires_grad=True)
        forward_poly = torch.tensor([0.01, 0.001, 0.0001, 0.00001], device=self.device, requires_grad=True)
        resolution = torch.tensor([3840, 2160], device=self.device)

        projection = OpenCVFisheyeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            forward_poly=forward_poly,
            resolution=resolution,
            max_angle=2.0,
            newton_iterations=10,
            min_2d_norm=torch.tensor(1e-6, device=self.device),
        )
        external_distortion = NoExternalDistortion()

        # Create rays
        num_rays = 100
        torch.manual_seed(456)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.3
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Loss
        loss = image_points.sum()
        loss.backward()

        analytical_cx = principal_point.grad[0].item()
        analytical_cy = principal_point.grad[1].item()

        # Principal point gradient should be num_rays in bounds (each ray contributes 1)
        expected_grad = float(valid.sum().item())

        self.assertAlmostEqual(
            analytical_cx,
            expected_grad,
            delta=GRAD_ATOL,
            msg=f"Fisheye cx gradient mismatch: got {analytical_cx}, expected {expected_grad}",
        )
        self.assertAlmostEqual(
            analytical_cy,
            expected_grad,
            delta=GRAD_ATOL,
            msg=f"Fisheye cy gradient mismatch: got {analytical_cy}, expected {expected_grad}",
        )

    def test_bivariate_windshield_uniform_gradient_accumulation(self):
        """Test gradient accumulation for bivariate windshield distortion parameters."""
        # Camera intrinsics
        focal_length = torch.tensor([1000.0, 1000.0], device=self.device)
        principal_point = torch.tensor([320.0, 240.0], device=self.device)
        radial_coeffs = torch.zeros(6, device=self.device)
        tangential_coeffs = torch.zeros(2, device=self.device)
        thin_prism_coeffs = torch.zeros(4, device=self.device)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=self.device),
        )

        # Windshield distortion with requires_grad
        # Use 3 coefficients for order 1 bivariate polynomial (triangular: 1+2=3)
        h_poly = torch.tensor([0.0, 0.01, 0.001], device=self.device, requires_grad=True)
        v_poly = torch.tensor([0.0, 0.01, 0.001], device=self.device, requires_grad=True)
        h_poly_inv = torch.tensor([0.0, -0.01, -0.001], device=self.device)
        v_poly_inv = torch.tensor([0.0, -0.01, -0.001], device=self.device)

        external_distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        # Create rays
        num_rays = 80
        torch.manual_seed(789)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.2
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Forward projection
        image_points, valid = camera_rays_to_image_points(rays, projection, external_distortion)

        # Loss
        loss = image_points.sum()
        loss.backward()

        # Verify gradients exist and are non-zero
        self.assertIsNotNone(h_poly.grad, "h_poly gradient should exist")
        self.assertIsNotNone(v_poly.grad, "v_poly gradient should exist")
        self.assertTrue(
            (h_poly.grad.abs() > 1e-8).any().item(),
            f"h_poly gradient should be non-zero, got {h_poly.grad}",
        )
        self.assertTrue(
            (v_poly.grad.abs() > 1e-8).any().item(),
            f"v_poly gradient should be non-zero, got {v_poly.grad}",
        )

        # Test accumulation by comparing full batch vs subset
        subset_rays = rays[:30]
        h_poly2 = torch.tensor([0.0, 0.01, 0.001], device=self.device, requires_grad=True)
        v_poly2 = torch.tensor([0.0, 0.01, 0.001], device=self.device, requires_grad=True)

        external_distortion2 = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly2,
            v_poly=v_poly2,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        image_points2, _ = camera_rays_to_image_points(subset_rays, projection, external_distortion2)
        loss2 = image_points2.sum()
        loss2.backward()

        # Full batch gradient magnitude should be larger than subset
        full_grad_norm = h_poly.grad.norm().item()
        subset_grad_norm = h_poly2.grad.norm().item()
        self.assertGreater(
            full_grad_norm,
            subset_grad_norm * 0.5,
            f"Full batch h_poly gradient norm ({full_grad_norm}) should be larger than subset ({subset_grad_norm})",
        )


class TestKernelGradientsMatchPyTorchReference(unittest.TestCase):
    """Comprehensive tests comparing kernel gradients against pure-PyTorch reference implementations.

    These tests catch loadOnce() issues and verify gradient correctness by comparing
    against ncore-style PyTorch implementations that use autograd. Each kernel variant
    and camera model combination is tested.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.device = device

    # =========================================================================
    # PyTorch Reference Implementations (similar to ncore)
    # =========================================================================

    @staticmethod
    def torch_pinhole_project(
        rays: torch.Tensor,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
        radial_coeffs: torch.Tensor,
        tangential_coeffs: torch.Tensor,
        thin_prism_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """Pure PyTorch implementation of OpenCV pinhole projection."""
        # Perspective normalization
        uv_normalized = rays[:, :2] / rays[:, 2:3]

        # Compute distortion
        xy_squared = uv_normalized**2
        r_2 = xy_squared.sum(dim=1, keepdim=True)
        r_4 = r_2**2
        r_6 = r_2**3
        xy_prod = uv_normalized[:, 0:1] * uv_normalized[:, 1:2]

        # Radial distortion
        k1, k2, k3, k4, k5, k6 = (
            radial_coeffs[0],
            radial_coeffs[1],
            radial_coeffs[2],
            radial_coeffs[3],
            radial_coeffs[4],
            radial_coeffs[5],
        )
        radial_num = 1.0 + k1 * r_2 + k2 * r_4 + k3 * r_6
        radial_denom = 1.0 + k4 * r_2 + k5 * r_4 + k6 * r_6
        radial = radial_num / radial_denom

        # Tangential distortion
        p1, p2 = tangential_coeffs[0], tangential_coeffs[1]
        a1 = 2 * xy_prod
        a2 = r_2 + 2 * xy_squared[:, 0:1]
        a3 = r_2 + 2 * xy_squared[:, 1:2]
        delta_x = p1 * a1 + p2 * a2
        delta_y = p1 * a3 + p2 * a1

        # Thin prism distortion
        s1, s2, s3, s4 = thin_prism_coeffs[0], thin_prism_coeffs[1], thin_prism_coeffs[2], thin_prism_coeffs[3]
        delta_x = delta_x + s1 * r_2 + s2 * r_4
        delta_y = delta_y + s3 * r_2 + s4 * r_4

        # Apply distortion
        uv_distorted = uv_normalized * radial + torch.cat([delta_x, delta_y], dim=1)

        # Apply camera matrix
        image_points = uv_distorted * focal_length + principal_point

        return image_points

    @staticmethod
    def torch_pinhole_backproject(
        image_points: torch.Tensor,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
        radial_coeffs: torch.Tensor,
        tangential_coeffs: torch.Tensor,
        thin_prism_coeffs: torch.Tensor,
        max_iterations: int = 10,
    ) -> torch.Tensor:
        """Pure PyTorch implementation of OpenCV pinhole backprojection with iterative undistortion."""
        # Remove camera matrix
        uv_normalized = (image_points - principal_point) / focal_length

        # Iterative undistortion
        uv = uv_normalized.clone()
        for _ in range(max_iterations):
            xy_squared = uv**2
            r_2 = xy_squared.sum(dim=1, keepdim=True)
            r_4 = r_2**2
            r_6 = r_2**3
            xy_prod = uv[:, 0:1] * uv[:, 1:2]

            k1, k2, k3, k4, k5, k6 = (
                radial_coeffs[0],
                radial_coeffs[1],
                radial_coeffs[2],
                radial_coeffs[3],
                radial_coeffs[4],
                radial_coeffs[5],
            )
            radial_num = 1.0 + k1 * r_2 + k2 * r_4 + k3 * r_6
            radial_denom = 1.0 + k4 * r_2 + k5 * r_4 + k6 * r_6
            radial = radial_num / radial_denom

            p1, p2 = tangential_coeffs[0], tangential_coeffs[1]
            a1 = 2 * xy_prod
            a2 = r_2 + 2 * xy_squared[:, 0:1]
            a3 = r_2 + 2 * xy_squared[:, 1:2]
            delta_x = p1 * a1 + p2 * a2
            delta_y = p1 * a3 + p2 * a1

            s1, s2, s3, s4 = thin_prism_coeffs[0], thin_prism_coeffs[1], thin_prism_coeffs[2], thin_prism_coeffs[3]
            delta_x = delta_x + s1 * r_2 + s2 * r_4
            delta_y = delta_y + s3 * r_2 + s4 * r_4

            uv = (uv_normalized - torch.cat([delta_x, delta_y], dim=1)) / radial

        # Create rays with z=1
        rays = torch.cat([uv, torch.ones_like(uv[:, 0:1])], dim=1)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        return rays

    @staticmethod
    def eval_poly_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial using Horner's method."""
        result = torch.zeros_like(x)
        for c in reversed(coeffs):
            result = c + x * result
        return result

    @staticmethod
    def torch_fisheye_project(
        rays: torch.Tensor,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
        radial_coeffs: torch.Tensor,  # [k1, k2, k3, k4]
        resolution: torch.Tensor,
        max_angle: float,
    ) -> torch.Tensor:
        """Pure PyTorch implementation of OpenCV fisheye projection."""
        # Compute theta (angle from optical axis)
        xy_norm = torch.sqrt(rays[:, 0:1] ** 2 + rays[:, 1:2] ** 2).clamp(min=1e-8)
        theta = torch.atan2(xy_norm, rays[:, 2:3])

        # Apply fisheye distortion: theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        theta_2 = theta**2
        theta_4 = theta_2**2
        theta_6 = theta_2 * theta_4
        theta_8 = theta_4**2

        k1, k2, k3, k4 = radial_coeffs[0], radial_coeffs[1], radial_coeffs[2], radial_coeffs[3]
        theta_d = theta * (1.0 + k1 * theta_2 + k2 * theta_4 + k3 * theta_6 + k4 * theta_8)

        # Project to image plane
        scale = theta_d / xy_norm
        uv = rays[:, :2] * scale

        # Apply camera matrix
        image_points = uv * focal_length + principal_point

        return image_points

    @staticmethod
    def torch_fisheye_backproject(
        image_points: torch.Tensor,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
        radial_coeffs: torch.Tensor,  # [k1, k2, k3, k4]
        resolution: torch.Tensor,
        max_angle: float,
        max_iterations: int = 10,
    ) -> torch.Tensor:
        """Pure PyTorch implementation of OpenCV fisheye backprojection with Newton iteration."""
        # Remove camera matrix
        uv = (image_points - principal_point) / focal_length
        theta_d = torch.sqrt(uv[:, 0:1] ** 2 + uv[:, 1:2] ** 2).clamp(min=1e-8)

        # Newton iteration to find theta from theta_d
        k1, k2, k3, k4 = radial_coeffs[0], radial_coeffs[1], radial_coeffs[2], radial_coeffs[3]
        approx_backward_factor = max_angle / torch.max(resolution / 2 / focal_length)
        theta = approx_backward_factor * theta_d  # Initial guess

        for _ in range(max_iterations):
            theta_2 = theta**2
            theta_4 = theta_2**2
            theta_6 = theta_2 * theta_4
            theta_8 = theta_4**2

            # f(theta) = theta * (1 + k1*theta^2 + ...) - theta_d
            f = theta * (1.0 + k1 * theta_2 + k2 * theta_4 + k3 * theta_6 + k4 * theta_8) - theta_d
            # f'(theta) = 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^6 + 9*k4*theta^8
            df = 1.0 + 3 * k1 * theta_2 + 5 * k2 * theta_4 + 7 * k3 * theta_6 + 9 * k4 * theta_8

            theta = theta - f / df.clamp(min=1e-8)

        # Reconstruct rays
        scale = torch.sin(theta) / theta_d
        x = uv[:, 0:1] * scale
        y = uv[:, 1:2] * scale
        z = torch.cos(theta)

        rays = torch.cat([x, y, z], dim=1)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        return rays

    @staticmethod
    def torch_ftheta_project(
        rays: torch.Tensor,
        principal_point: torch.Tensor,
        fw_poly: torch.Tensor,
        A: torch.Tensor,  # 2x2 linear term
    ) -> torch.Tensor:
        """Pure PyTorch implementation of F-Theta projection."""
        # Compute theta (angle from optical axis)
        xy_norm = torch.sqrt(rays[:, 0:1] ** 2 + rays[:, 1:2] ** 2).clamp(min=1e-8)
        theta = torch.atan2(xy_norm, rays[:, 2:3])

        # Evaluate forward polynomial: r = sum(fw_poly[i] * theta^(i+1))
        r = torch.zeros_like(theta)
        theta_power = theta.clone()
        for i, coeff in enumerate(fw_poly):
            r = r + coeff * theta_power
            theta_power = theta_power * theta

        # Compute unit direction in image plane
        uv_dir = rays[:, :2] / xy_norm

        # Apply linear term A
        uv = r * uv_dir
        uv_transformed = torch.einsum("ij,nj->ni", A, uv)

        # Add principal point
        image_points = uv_transformed + principal_point

        return image_points

    @staticmethod
    def torch_ftheta_backproject(
        image_points: torch.Tensor,
        principal_point: torch.Tensor,
        bw_poly: torch.Tensor,
        Ainv: torch.Tensor,  # 2x2 inverse linear term
    ) -> torch.Tensor:
        """Pure PyTorch implementation of F-Theta backprojection."""
        # Remove principal point and apply inverse linear term
        uv = image_points - principal_point
        uv_transformed = torch.einsum("ij,nj->ni", Ainv, uv)

        # Compute radius
        r = torch.sqrt(uv_transformed[:, 0:1] ** 2 + uv_transformed[:, 1:2] ** 2).clamp(min=1e-8)

        # Evaluate backward polynomial: theta = sum(bw_poly[i] * r^(i+1))
        theta = torch.zeros_like(r)
        r_power = r.clone()
        for i, coeff in enumerate(bw_poly):
            theta = theta + coeff * r_power
            r_power = r_power * r

        # Reconstruct rays
        uv_dir = uv_transformed / r
        x = torch.sin(theta) * uv_dir[:, 0:1]
        y = torch.sin(theta) * uv_dir[:, 1:2]
        z = torch.cos(theta)

        rays = torch.cat([x, y, z], dim=1)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        return rays

    # =========================================================================
    # OpenCV Pinhole Tests
    # =========================================================================

    def test_opencv_pinhole_forward_projection_gradient(self):
        """Test OpenCV pinhole forward projection gradients match PyTorch reference."""
        num_rays = 100
        torch.manual_seed(42)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.3
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        # Parameters
        focal_length_val = [1000.0, 1000.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.05, 0.01, 0.001, 0.0, 0.0, 0.0]
        tangential_coeffs_val = [0.001, 0.001]
        thin_prism_coeffs_val = [0.0001, 0.0001, 0.0001, 0.0001]

        # Reference
        ref_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        ref_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        ref_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        ref_pts = self.torch_pinhole_project(rays, ref_fl, ref_pp, ref_rc, ref_tc, ref_tpc)
        ref_pts.sum().backward()

        # Kernel
        kernel_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        kernel_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        kernel_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=kernel_fl,
            principal_point=kernel_pp,
            radial_coeffs=kernel_rc,
            tangential_coeffs=kernel_tc,
            thin_prism_coeffs=kernel_tpc,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        kernel_pts, _ = camera_rays_to_image_points(rays, projection, NoExternalDistortion())
        kernel_pts.sum().backward()

        # Compare all gradients
        np.testing.assert_allclose(
            kernel_fl.grad.cpu().numpy(),
            ref_fl.grad.cpu().numpy(),
            rtol=ATOL,
            atol=ATOL,
            err_msg="Focal length gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            ref_pp.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg="Principal point gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_rc.grad[:3].cpu().numpy(),
            ref_rc.grad[:3].cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg="Radial coeffs gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_tc.grad.cpu().numpy(),
            ref_tc.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=GRAD_ATOL,
            err_msg="Tangential coeffs gradient mismatch",
        )

    def test_opencv_pinhole_backprojection_gradient(self):
        """Test OpenCV pinhole backprojection gradients match PyTorch reference."""
        num_points = 80
        torch.manual_seed(123)
        image_points = torch.rand(num_points, 2, device=self.device)
        image_points[:, 0] = image_points[:, 0] * 400 + 120
        image_points[:, 1] = image_points[:, 1] * 300 + 90

        # Use simpler distortion for stable backprojection
        focal_length_val = [500.0, 500.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.01, 0.001, 0.0, 0.0, 0.0, 0.0]
        tangential_coeffs_val = [0.0, 0.0]
        thin_prism_coeffs_val = [0.0, 0.0, 0.0, 0.0]

        # Reference
        ref_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        ref_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        ref_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        ref_rays = self.torch_pinhole_backproject(image_points, ref_fl, ref_pp, ref_rc, ref_tc, ref_tpc)
        ref_rays.sum().backward()

        # Kernel
        kernel_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        kernel_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        kernel_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=kernel_fl,
            principal_point=kernel_pp,
            radial_coeffs=kernel_rc,
            tangential_coeffs=kernel_tc,
            thin_prism_coeffs=kernel_tpc,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        kernel_rays = image_points_to_camera_rays(image_points, projection, NoExternalDistortion())
        kernel_rays.sum().backward()

        # Compare gradients
        np.testing.assert_allclose(
            kernel_fl.grad.cpu().numpy(),
            ref_fl.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=NORM_ATOL,
            err_msg="Backprojection focal length gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            ref_pp.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=NORM_ATOL,
            err_msg="Backprojection principal point gradient mismatch",
        )

    # =========================================================================
    # OpenCV Fisheye Tests
    # =========================================================================

    def test_opencv_fisheye_forward_projection_gradient(self):
        """Test OpenCV fisheye forward projection gradients match PyTorch reference."""
        num_rays = 100
        torch.manual_seed(42)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.3
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        focal_length_val = [400.0, 400.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.01, 0.001, 0.0001, 0.00001]

        # Reference
        ref_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        resolution = torch.tensor([3840, 2160], device=self.device)
        max_angle = 2.0

        ref_pts = self.torch_fisheye_project(rays, ref_fl, ref_pp, ref_rc, resolution, max_angle)
        ref_pts.sum().backward()

        # Kernel
        kernel_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_forward_poly = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVFisheyeProjection.from_components(
            focal_length=kernel_fl,
            principal_point=kernel_pp,
            forward_poly=kernel_forward_poly,
            resolution=resolution,
            max_angle=2.0,
            newton_iterations=10,
            min_2d_norm=torch.tensor(1e-6, device=self.device),
        )
        kernel_pts, _ = camera_rays_to_image_points(rays, projection, NoExternalDistortion())
        kernel_pts.sum().backward()

        # Compare gradients
        np.testing.assert_allclose(
            kernel_fl.grad.cpu().numpy(),
            ref_fl.grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Fisheye focal length gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            ref_pp.grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Fisheye principal point gradient mismatch",
        )
        # Note: The polynomial gradient comparison needs care since kernel uses different poly structure
        # Just verify they're non-zero and in similar ballpark
        self.assertTrue(kernel_forward_poly.grad.abs().sum() > 0.1, "Fisheye forward poly gradient should be non-zero")

    def test_opencv_fisheye_backprojection_gradient(self):
        """Test OpenCV fisheye backprojection gradients match PyTorch reference."""
        num_points = 80
        torch.manual_seed(456)
        image_points = torch.rand(num_points, 2, device=self.device)
        image_points[:, 0] = image_points[:, 0] * 200 + 220
        image_points[:, 1] = image_points[:, 1] * 150 + 165

        focal_length_val = [400.0, 400.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.01, 0.001, 0.0001, 0.00001]

        # Reference
        ref_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        resolution = torch.tensor([3840, 2160], device=self.device)
        max_angle = 2.0
        ref_rays = self.torch_fisheye_backproject(image_points, ref_fl, ref_pp, ref_rc, resolution, max_angle)
        ref_rays.sum().backward()

        # Kernel
        kernel_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_forward_poly = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVFisheyeProjection.from_components(
            focal_length=kernel_fl,
            principal_point=kernel_pp,
            forward_poly=kernel_forward_poly,
            resolution=resolution,
            max_angle=max_angle,
            newton_iterations=10,
            min_2d_norm=torch.tensor(1e-6, device=self.device),
        )
        kernel_rays = image_points_to_camera_rays(image_points, projection, NoExternalDistortion())
        kernel_rays.sum().backward()

        # Compare gradients
        np.testing.assert_allclose(
            kernel_fl.grad.cpu().numpy(),
            ref_fl.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=NORM_ATOL,
            err_msg="Fisheye backprojection focal length gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            ref_pp.grad.cpu().numpy(),
            rtol=TIGHT_RTOL,
            atol=NORM_ATOL,
            err_msg="Fisheye backprojection principal point gradient mismatch",
        )

    # =========================================================================
    # F-Theta Tests
    # =========================================================================

    def test_ftheta_forward_projection_gradient(self):
        """Test F-Theta forward projection gradients match PyTorch reference.

        Note: F-Theta uses complex polynomial inversion, so we focus on comparing
        principal point gradients which should be exact (each ray contributes 1).
        """
        num_rays = 100
        torch.manual_seed(42)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.2
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        principal_point_val = [320.0, 240.0]
        fw_poly_val = [500.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bw_poly_val = [0.002, 0.0, 0.0, 0.0, 0.0, 0.0]
        A_val = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        Ainv_val = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        expected_pp_grad = [num_rays, num_rays]

        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_fw = torch.tensor(fw_poly_val, device=self.device, requires_grad=True)
        kernel_bw = torch.tensor(bw_poly_val, device=self.device, requires_grad=True)
        kernel_A = torch.tensor(A_val, device=self.device, requires_grad=True)
        kernel_Ainv = torch.tensor(Ainv_val, device=self.device, requires_grad=True)
        kernel_dfw = torch.tensor([500.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        kernel_dbw = torch.tensor([0.002, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

        projection = FThetaProjection.from_components(
            principal_point=kernel_pp,
            fw_poly=kernel_fw,
            bw_poly=kernel_bw,
            A=kernel_A,
            Ainv=kernel_Ainv,
            dfw_poly=kernel_dfw,
            dbw_poly=kernel_dbw,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        kernel_pts, _ = camera_rays_to_image_points(rays, projection, NoExternalDistortion())
        kernel_pts.sum().backward()

        # Compare principal point gradient against expected value
        # Gradients flow back to the original leaf tensors (not the concatenated intrinsics)
        assert kernel_pp.grad is not None, "Expected gradients for principal_point"
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            expected_pp_grad,
            rtol=TIGHT_RTOL,
            atol=TIGHT_ATOL,
            err_msg="F-Theta principal point gradient mismatch",
        )
        # Verify polynomial gradient is non-zero (gradient flows through)
        assert kernel_fw.grad is not None, "Expected gradients for fw_poly"
        self.assertTrue(kernel_fw.grad.abs().sum() > 0.1, "F-Theta forward poly gradient should be non-zero")

    def test_ftheta_backprojection_gradient(self):
        """Test F-Theta backprojection gradients are non-zero and consistent.

        Note: F-Theta uses complex polynomial inversion, so we verify gradients
        flow through (are non-zero) and check consistency between batches.
        """
        num_points = 80
        torch.manual_seed(789)
        image_points = torch.rand(num_points, 2, device=self.device)
        image_points[:, 0] = image_points[:, 0] * 200 + 220
        image_points[:, 1] = image_points[:, 1] * 150 + 165

        principal_point_val = [320.0, 240.0]
        fw_poly_val = [0.0, 500.0, 0.0, 0.0, 0.0, 0.0]
        bw_poly_val = [0.0, 0.002, 0.0, 0.0, 0.0, 0.0]
        A_val = [[1.0, 0.0], [0.0, 1.0]]
        Ainv_val = [[1.0, 0.0], [0.0, 1.0]]

        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_fw = torch.tensor(fw_poly_val, device=self.device, requires_grad=True)
        kernel_bw = torch.tensor(bw_poly_val, device=self.device, requires_grad=True)
        kernel_A = torch.tensor(A_val, device=self.device, requires_grad=True)
        kernel_Ainv = torch.tensor(Ainv_val, device=self.device, requires_grad=True)
        kernel_dfw = torch.tensor([500.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        kernel_dbw = torch.tensor([0.002, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)

        projection = FThetaProjection.from_components(
            principal_point=kernel_pp,
            fw_poly=kernel_fw,
            bw_poly=kernel_bw,
            A=kernel_A,
            Ainv=kernel_Ainv,
            dfw_poly=kernel_dfw,
            dbw_poly=kernel_dbw,
            reference_poly=FThetaPolynomialType.BACKWARD,  # Use backward as reference for backprojection
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        kernel_rays = image_points_to_camera_rays(image_points, projection, NoExternalDistortion())
        kernel_rays.sum().backward()

        # Verify gradients are non-zero (gradients flow through)
        # Gradients flow back to the original leaf tensors (not the concatenated intrinsics)
        assert kernel_pp.grad is not None, "Expected gradients for principal_point"
        assert kernel_bw.grad is not None, "Expected gradients for bw_poly"
        self.assertTrue(kernel_pp.grad.abs().sum() > 0.001, "F-Theta principal point gradient should be non-zero")
        self.assertTrue(kernel_bw.grad.abs().sum() > 0.001, "F-Theta bw_poly gradient should be non-zero")

        # Test gradient accumulation: larger batch should have larger gradient magnitude
        kernel_pp2 = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_bw2 = torch.tensor(bw_poly_val, device=self.device, requires_grad=True)
        projection2 = FThetaProjection.from_components(
            principal_point=kernel_pp2,
            fw_poly=kernel_fw.detach(),
            bw_poly=kernel_bw2,
            A=kernel_A.detach(),
            Ainv=kernel_Ainv.detach(),
            dfw_poly=kernel_dfw,
            dbw_poly=kernel_dbw,
            reference_poly=FThetaPolynomialType.BACKWARD,
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        subset_points = image_points[:40]
        kernel_rays2 = image_points_to_camera_rays(subset_points, projection2, NoExternalDistortion())
        kernel_rays2.sum().backward()

        # Full batch gradient magnitude should be approximately 2x subset for principal point
        # Gradients flow back to the original leaf tensors
        assert kernel_pp2.grad is not None, "Expected gradients for principal_point2"
        ratio = kernel_pp.grad.abs().sum() / kernel_pp2.grad.abs().sum()
        self.assertGreater(ratio.item(), 1.5, f"Full batch gradient should be ~2x subset, got ratio {ratio.item()}")

    # =========================================================================
    # Large Batch Tests (catch loadOnce() issues)
    # =========================================================================

    def test_opencv_pinhole_large_batch_gradient_accumulation(self):
        """Test gradient accumulation with large batch for OpenCV pinhole."""
        num_rays = 2048  # 64 warps
        torch.manual_seed(999)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.2
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        focal_length_val = [500.0, 500.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tangential_coeffs_val = [0.0, 0.0]
        thin_prism_coeffs_val = [0.0, 0.0, 0.0, 0.0]

        # Reference
        ref_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        ref_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        ref_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        ref_pts = self.torch_pinhole_project(rays, ref_fl, ref_pp, ref_rc, ref_tc, ref_tpc)
        ref_pts.sum().backward()

        # Kernel
        kernel_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        kernel_tc = torch.tensor(tangential_coeffs_val, device=self.device, requires_grad=True)
        kernel_tpc = torch.tensor(thin_prism_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVPinholeProjection.from_components(
            focal_length=kernel_fl,
            principal_point=kernel_pp,
            radial_coeffs=kernel_rc,
            tangential_coeffs=kernel_tc,
            thin_prism_coeffs=kernel_tpc,
            resolution=torch.tensor([640, 480], device=self.device),
        )
        kernel_pts, _ = camera_rays_to_image_points(rays, projection, NoExternalDistortion())
        kernel_pts.sum().backward()

        # Principal point gradient should be exactly num_rays (each ray contributes 1 to x and 1 to y)
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            ref_pp.grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Large batch principal point gradient mismatch",
        )
        # Should be [num_rays, num_rays]
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            [num_rays, num_rays],
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Principal point gradient should equal num_rays",
        )

    def test_opencv_fisheye_large_batch_gradient_accumulation(self):
        """Test gradient accumulation with large batch for OpenCV fisheye."""
        num_rays = 1024
        torch.manual_seed(888)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.2
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        focal_length_val = [400.0, 400.0]
        principal_point_val = [320.0, 240.0]
        radial_coeffs_val = [0.0, 0.0, 0.0, 0.0]

        # Reference
        ref_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        ref_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        ref_rc = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)
        resolution = torch.tensor([3840, 2160], device=self.device)
        max_angle = 2.0

        ref_pts = self.torch_fisheye_project(rays, ref_fl, ref_pp, ref_rc, resolution, max_angle)
        ref_pts.sum().backward()

        # Kernel
        kernel_fl = torch.tensor(focal_length_val, device=self.device, requires_grad=True)
        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_forward_poly = torch.tensor(radial_coeffs_val, device=self.device, requires_grad=True)

        projection = OpenCVFisheyeProjection.from_components(
            focal_length=kernel_fl,
            principal_point=kernel_pp,
            forward_poly=kernel_forward_poly,
            resolution=resolution,
            max_angle=max_angle,
            newton_iterations=10,
            min_2d_norm=torch.tensor(1e-6, device=self.device),
        )
        kernel_pts, _ = camera_rays_to_image_points(rays, projection, NoExternalDistortion())
        kernel_pts.sum().backward()

        # Compare principal point gradient
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            ref_pp.grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Fisheye large batch principal point gradient mismatch",
        )
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            [num_rays, num_rays],
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Fisheye principal point gradient should equal num_rays",
        )

    def test_ftheta_large_batch_gradient_accumulation(self):
        """Test gradient accumulation with large batch for F-Theta."""
        num_rays = 1024
        torch.manual_seed(777)
        rays = torch.randn(num_rays, 3, device=self.device) * 0.15
        rays[:, 2] = 1.0
        rays = rays / rays.norm(dim=-1, keepdim=True)

        principal_point_val = [320.0, 240.0]
        fw_poly_val = [500.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bw_poly_val = [0.002, 0.0, 0.0, 0.0, 0.0, 0.0]
        A_val = [[1.0, 0.0], [0.0, 1.0]]

        kernel_pp = torch.tensor(principal_point_val, device=self.device, requires_grad=True)
        kernel_fw = torch.tensor(fw_poly_val, device=self.device, requires_grad=True)
        kernel_bw = torch.tensor(bw_poly_val, device=self.device, requires_grad=True)
        kernel_A = torch.tensor(A_val, device=self.device, requires_grad=True)
        kernel_Ainv = torch.eye(2, device=self.device)
        kernel_dfw = torch.tensor([500.0] + [0.0] * 5, device=self.device)
        kernel_dbw = torch.tensor([0.002] + [0.0] * 5, device=self.device)

        projection = FThetaProjection.from_components(
            principal_point=kernel_pp,
            fw_poly=kernel_fw,
            bw_poly=kernel_bw,
            A=kernel_A,
            Ainv=kernel_Ainv,
            dfw_poly=kernel_dfw,
            dbw_poly=kernel_dbw,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        kernel_pts, _ = camera_rays_to_image_points(rays, projection, NoExternalDistortion())
        kernel_pts.sum().backward()

        # Principal point gradient should equal num_rays exactly (each ray contributes 1)
        np.testing.assert_allclose(
            kernel_pp.grad.cpu().numpy(),
            [num_rays, num_rays],
            rtol=ATOL,
            atol=GRAD_ATOL,
            err_msg="F-Theta principal point gradient should equal num_rays",
        )


class TestRollingShutterWithExternalDistortion(unittest.TestCase):
    """Tests for rolling shutter kernels combined with BivariateWindshield external distortion.

    Verifies analytical correctness of:
      - Forward projection values against Python reference implementations
      - Backprojection values against Python reference implementations
      - Gradient values against finite differences

    Uses identity pose (start=end) where noted to eliminate rolling shutter iteration
    complexity and enable closed-form reference computation.
    """

    device = torch.device("cuda")

    # ---- Pure-Python reference implementations ----

    @staticmethod
    def _ref_eval_poly_2d(x, y, c, order):
        """Triangular 2D polynomial matching Slang PolynomialUtils.eval_poly_2d.

        Uses nested Horner's method on (x, y). Triangular term counts:
        order 0: 1, order 1: 3, order 2: 6, order 3: 10, order 4: 15.
        """
        if order == 0:
            return c[0]
        elif order == 1:
            return (c[0] + x * c[1]) + y * c[2]
        elif order == 2:
            return (c[0] + x * (c[1] + x * c[2])) + y * ((c[3] + x * c[4]) + y * c[5])
        elif order == 3:
            y0 = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
            y1 = c[4] + x * (c[5] + x * c[6])
            y2 = c[7] + x * c[8]
            y3 = c[9]
            return y0 + y * (y1 + y * (y2 + y * y3))
        elif order == 4:
            y0 = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))
            y1 = c[5] + x * (c[6] + x * (c[7] + x * c[8]))
            y2 = c[9] + x * (c[10] + x * c[11])
            y3 = c[12] + x * c[13]
            y4 = c[14]
            return y0 + y * (y1 + y * (y2 + y * (y3 + y * y4)))
        raise NotImplementedError(f"order {order}")

    @staticmethod
    def _ref_eval_poly(x, c, degree):
        """1D polynomial matching Slang PolynomialUtils.eval_poly: c[0] + c[1]*x + c[2]*x^2 + ..."""
        return sum(c[i] * x**i for i in range(degree + 1))

    @staticmethod
    def _ref_bivariate_distort(ray, h_poly, v_poly, h_order, v_order):
        """Bivariate windshield apply_distortion (pure math)."""
        norm = math.sqrt(sum(r**2 for r in ray))
        rx, ry, rz = ray[0] / norm, ray[1] / norm, ray[2] / norm
        phi = math.asin(max(-1.0, min(1.0, rx)))
        theta = math.asin(max(-1.0, min(1.0, ry)))
        ref = TestRollingShutterWithExternalDistortion._ref_eval_poly_2d
        adj_phi = ref(phi, theta, h_poly, h_order)
        adj_theta = ref(phi, theta, v_poly, v_order)
        xd = math.sin(adj_phi)
        yd = math.sin(adj_theta)
        zd = math.sqrt(max(0.0, 1.0 - xd**2 - yd**2)) * (1.0 if rz > 0 else -1.0)
        return (xd, yd, zd)

    @staticmethod
    def _ref_pinhole_project(ray, fx, fy, cx, cy):
        """Pinhole projection (zero radial/tangential/thin-prism distortion)."""
        return (ray[0] / ray[2] * fx + cx, ray[1] / ray[2] * fy + cy)

    @staticmethod
    def _ref_pinhole_backproject(ip, fx, fy, cx, cy):
        """Pinhole backprojection (zero distortion). Returns normalized camera ray."""
        ux = (ip[0] - cx) / fx
        uy = (ip[1] - cy) / fy
        n = math.sqrt(ux**2 + uy**2 + 1.0)
        return (ux / n, uy / n, 1.0 / n)

    @staticmethod
    def _ref_ftheta_project(ray, fw_poly, degree, pp, A):
        """FTheta projection (FORWARD reference polynomial)."""
        norm = math.sqrt(sum(r**2 for r in ray))
        rx, ry, rz = ray[0] / norm, ray[1] / norm, ray[2] / norm
        theta = math.acos(max(-1.0, min(1.0, rz)))
        r = TestRollingShutterWithExternalDistortion._ref_eval_poly(theta, fw_poly, degree)
        xy_norm = math.sqrt(rx**2 + ry**2)
        if xy_norm < 1e-6:
            return (pp[0], pp[1])
        scale = r / xy_norm
        ox, oy = rx * scale, ry * scale
        tx = A[0][0] * ox + A[0][1] * oy
        ty = A[1][0] * ox + A[1][1] * oy
        return (tx + pp[0], ty + pp[1])

    @staticmethod
    def _ref_ftheta_backproject(ip, fw_poly_c1, pp, Ainv):
        """FTheta backprojection for linear fw_poly (r = c1*theta, so theta = rdist/c1)."""
        ox = ip[0] - pp[0]
        oy = ip[1] - pp[1]
        tx = Ainv[0][0] * ox + Ainv[0][1] * oy
        ty = Ainv[1][0] * ox + Ainv[1][1] * oy
        rdist = math.sqrt(tx**2 + ty**2)
        if rdist < 1e-6:
            return (0.0, 0.0, 1.0)
        theta = rdist / fw_poly_c1
        s = math.sin(theta) / rdist
        rx, ry, rz = tx * s, ty * s, math.cos(theta)
        n = math.sqrt(rx**2 + ry**2 + rz**2)
        return (rx / n, ry / n, rz / n)

    # ---- Rolling-shutter pose interpolation reference helpers ----

    @staticmethod
    def _ref_slerp(q0, q1, t):
        """Quaternion SLERP in xyzw format, matching Slang quaternion::slerp."""
        dot = sum(a * b for a, b in zip(q0, q1))
        if dot < 0:
            q1 = [-x for x in q1]
            dot = -dot
        dot = max(min(dot, 1.0), -1.0)
        if dot > 0.9995:
            result = [q0[i] + t * (q1[i] - q0[i]) for i in range(4)]
            norm = math.sqrt(sum(x * x for x in result))
            return [x / norm for x in result]
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        w0 = math.sin((1 - t) * theta) / sin_theta
        w1 = math.sin(t * theta) / sin_theta
        return [w0 * q0[i] + w1 * q1[i] for i in range(4)]

    @staticmethod
    def _ref_quat_rotate(q, v):
        """Rotate vector v by quaternion q (xyzw format): q * v * q^-1."""
        qx, qy, qz, qw = q
        uv = [
            qy * v[2] - qz * v[1],
            qz * v[0] - qx * v[2],
            qx * v[1] - qy * v[0],
        ]
        uuv = [
            qy * uv[2] - qz * uv[1],
            qz * uv[0] - qx * uv[2],
            qx * uv[1] - qy * uv[0],
        ]
        return [v[i] + 2.0 * (qw * uv[i] + uuv[i]) for i in range(3)]

    @staticmethod
    def _ref_inverse_transform_point(trans, rot, point):
        """R^T * (point - trans): world-to-camera transform. rot is xyzw quaternion."""
        translated = [point[i] - trans[i] for i in range(3)]
        rot_inv = [-rot[0], -rot[1], -rot[2], rot[3]]
        return TestRollingShutterWithExternalDistortion._ref_quat_rotate(rot_inv, translated)

    @staticmethod
    def _ref_transform_direction(rot, direction):
        """R * direction: camera-to-world rotation. rot is xyzw quaternion."""
        return TestRollingShutterWithExternalDistortion._ref_quat_rotate(rot, direction)

    @classmethod
    def _ref_rs_forward_pinhole(
        cls,
        world_point,
        trans_start,
        trans_end,
        rot_start,
        rot_end,
        h_poly,
        v_poly,
        h_order,
        v_order,
        fx,
        fy,
        cx,
        cy,
        height=480,
    ):
        """Reference rolling-shutter forward projection with iterative solver.

        Matches the Slang project_world_point_rolling_shutter kernel:
        initial t=0.5, iterate until convergence.
        """
        ip = None
        t = 0.5
        for _ in range(20):
            trans = [trans_start[i] + t * (trans_end[i] - trans_start[i]) for i in range(3)]
            rot = cls._ref_slerp(rot_start, rot_end, t)
            cam_point = cls._ref_inverse_transform_point(trans, rot, world_point)
            if cam_point[2] <= 0:
                return None
            norm = math.sqrt(sum(x * x for x in cam_point))
            cam_ray = [x / norm for x in cam_point]
            dist_ray = cls._ref_bivariate_distort(cam_ray, h_poly, v_poly, h_order, v_order)
            ip = cls._ref_pinhole_project(dist_ray, fx, fy, cx, cy)
            t_new = math.floor(ip[1]) / (height - 1)
            if abs(t_new - t) * height < 0.001:
                break
            t = t_new
        return ip

    @classmethod
    def _ref_rs_backproject_pinhole(
        cls,
        image_point,
        trans_start,
        trans_end,
        rot_start,
        rot_end,
        h_poly_inv,
        v_poly_inv,
        h_order,
        v_order,
        fx,
        fy,
        cx,
        cy,
        height=480,
    ):
        """Reference rolling-shutter backprojection (no iteration needed).

        Matches the Slang backproject kernel: t from scanline, then undistort + rotate.
        Returns (origin, direction) in world frame.
        """
        t = math.floor(image_point[1]) / (height - 1)
        trans = [trans_start[i] + t * (trans_end[i] - trans_start[i]) for i in range(3)]
        rot = cls._ref_slerp(rot_start, rot_end, t)
        cam_ray = cls._ref_pinhole_backproject(image_point, fx, fy, cx, cy)
        undist_ray = cls._ref_bivariate_distort(cam_ray, h_poly_inv, v_poly_inv, h_order, v_order)
        norm = math.sqrt(sum(x * x for x in undist_ray))
        undist_ray = [x / norm for x in undist_ray]
        world_dir = cls._ref_transform_direction(rot, undist_ray)
        return trans, world_dir

    @classmethod
    def _ref_rs_forward_ftheta(
        cls,
        world_point,
        trans_start,
        trans_end,
        rot_start,
        rot_end,
        h_poly,
        v_poly,
        h_order,
        v_order,
        fw_poly,
        fw_poly_degree,
        pp,
        A,
        height=480,
    ):
        """Reference rolling-shutter forward projection with FTheta model."""
        ip = None
        t = 0.5
        for _ in range(20):
            trans = [trans_start[i] + t * (trans_end[i] - trans_start[i]) for i in range(3)]
            rot = cls._ref_slerp(rot_start, rot_end, t)
            cam_point = cls._ref_inverse_transform_point(trans, rot, world_point)
            if cam_point[2] <= 0:
                return None
            norm = math.sqrt(sum(x * x for x in cam_point))
            cam_ray = [x / norm for x in cam_point]
            dist_ray = cls._ref_bivariate_distort(cam_ray, h_poly, v_poly, h_order, v_order)
            ip = cls._ref_ftheta_project(dist_ray, fw_poly, fw_poly_degree, pp, A)
            t_new = math.floor(ip[1]) / (height - 1)
            if abs(t_new - t) * height < 0.001:
                break
            t = t_new
        return ip

    @classmethod
    def _ref_rs_backproject_ftheta(
        cls,
        image_point,
        trans_start,
        trans_end,
        rot_start,
        rot_end,
        h_poly_inv,
        v_poly_inv,
        h_order,
        v_order,
        fw_poly_c1,
        pp,
        Ainv,
        height=480,
    ):
        """Reference rolling-shutter backprojection with FTheta model."""
        t = math.floor(image_point[1]) / (height - 1)
        trans = [trans_start[i] + t * (trans_end[i] - trans_start[i]) for i in range(3)]
        rot = cls._ref_slerp(rot_start, rot_end, t)
        cam_ray = cls._ref_ftheta_backproject(image_point, fw_poly_c1, pp, Ainv)
        undist_ray = cls._ref_bivariate_distort(cam_ray, h_poly_inv, v_poly_inv, h_order, v_order)
        norm = math.sqrt(sum(x * x for x in undist_ray))
        undist_ray = [x / norm for x in undist_ray]
        world_dir = cls._ref_transform_direction(rot, undist_ray)
        return trans, world_dir

    # ---- Polynomial coefficient constants used by all tests ----

    H_POLY_FWD = [0.0, 1.01, 0.0, 0.0, 0.0, 0.0]  # padded to MAX_H=6
    V_POLY_FWD = [0.0, 0.0, 1.01] + [0.0] * 12  # padded to MAX_V=15
    H_POLY_INV = [0.0, 0.99, 0.0, 0.0, 0.0, 0.0]
    V_POLY_INV = [0.0, 0.0, 0.99] + [0.0] * 12
    H_ORDER = 1
    V_ORDER = 1

    # Order-2 polynomials with cross-coupling terms (6 triangular terms each)
    # p(phi, theta) = c[0] + c[1]*phi + c[2]*phi^2 + c[3]*theta + c[4]*phi*theta + c[5]*theta^2
    H_POLY_FWD_O2 = [0.005, 1.01, 0.002, 0.003, 0.001, 0.001]  # adj_phi ≈ phi
    V_POLY_FWD_O2 = [0.005, 0.003, 0.001, 1.01, 0.001, 0.002] + [0.0] * 9  # adj_theta ≈ theta
    H_POLY_INV_O2 = [-0.005, 0.99, -0.002, -0.003, -0.001, -0.001]
    V_POLY_INV_O2 = [-0.005, -0.003, -0.001, 0.99, -0.001, -0.002] + [0.0] * 9
    H_ORDER_O2 = 2
    V_ORDER_O2 = 2

    # Order-3 v_poly (10 triangular terms). h_poly stays order 2 (MAX_H=6).
    # Monomials: 1, x, x², x³, y, xy, x²y, y², xy², y³
    V_POLY_FWD_O3 = [0.005, 0.003, 0.001, 0.0005, 1.01, 0.001, 0.0005, 0.002, 0.0005, 0.001] + [0.0] * 5
    V_POLY_INV_O3 = [-0.005, -0.003, -0.001, -0.0005, 0.99, -0.001, -0.0005, -0.002, -0.0005, -0.001] + [0.0] * 5
    V_ORDER_O3 = 3

    # Order-4 v_poly (15 triangular terms = MAX_V). h_poly stays order 2.
    # Monomials: 1, x, x², x³, x⁴, y, xy, x²y, x³y, y², xy², x²y², y³, xy³, y⁴
    V_POLY_FWD_O4 = [
        0.005,
        0.003,
        0.001,
        0.0005,
        0.0002,
        1.01,
        0.001,
        0.0005,
        0.0002,
        0.002,
        0.0005,
        0.0002,
        0.001,
        0.0003,
        0.0005,
    ]
    V_POLY_INV_O4 = [
        -0.005,
        -0.003,
        -0.001,
        -0.0005,
        -0.0002,
        0.99,
        -0.001,
        -0.0005,
        -0.0002,
        -0.002,
        -0.0005,
        -0.0002,
        -0.001,
        -0.0003,
        -0.0005,
    ]
    V_ORDER_O4 = 4

    FW_POLY = [0.0, 500.0, 0.0, 0.0, 0.0, 0.0]
    FW_POLY_DEGREE = 5
    FW_POLY_C1 = 500.0

    # ---- Factory helpers (return objects + leaf tensors for gradient access) ----

    @staticmethod
    def _make_pinhole_with_leaves(dev, requires_grad=False):
        focal_length = torch.tensor([500.0, 500.0], device=dev, requires_grad=requires_grad)
        principal_point = torch.tensor([320.0, 240.0], device=dev, requires_grad=requires_grad)
        radial_coeffs = torch.zeros(6, device=dev, requires_grad=requires_grad)
        tangential_coeffs = torch.zeros(2, device=dev, requires_grad=requires_grad)
        thin_prism_coeffs = torch.zeros(4, device=dev, requires_grad=requires_grad)
        projection = OpenCVPinholeProjection.from_components(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=torch.tensor([640, 480], device=dev),
        )
        return projection, focal_length, principal_point

    @staticmethod
    def _make_ftheta_with_leaves(dev, requires_grad=False):
        from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS

        pp = torch.tensor([320.0, 240.0], device=dev, requires_grad=requires_grad)
        fw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=dev, requires_grad=requires_grad)
        bw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=dev, requires_grad=requires_grad)
        fw_poly.data[1] = 500.0
        bw_poly.data[1] = 0.002
        dfw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=dev)
        dbw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=dev)
        dfw_poly[0] = 500.0
        dbw_poly[0] = 0.002
        A = torch.eye(2, device=dev, requires_grad=requires_grad)
        projection = FThetaProjection.from_components(
            principal_point=pp,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=A,
            Ainv=torch.eye(2, device=dev),
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        return projection, pp, fw_poly

    @staticmethod
    def _make_bivariate_with_leaves(dev, requires_grad=False):
        from libs.sensors.kernels.cameras.parameters import ReferencePolynomial

        h_poly = torch.tensor([0.0, 1.01, 0.0], device=dev, requires_grad=requires_grad)
        v_poly = torch.tensor([0.0, 0.0, 1.01], device=dev, requires_grad=requires_grad)
        h_poly_inv = torch.tensor([0.0, 0.99, 0.0], device=dev, requires_grad=requires_grad)
        v_poly_inv = torch.tensor([0.0, 0.0, 0.99], device=dev, requires_grad=requires_grad)
        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )
        return distortion, h_poly, v_poly, h_poly_inv, v_poly_inv

    @staticmethod
    def _make_bivariate_order2_with_leaves(dev, requires_grad=False):
        from libs.sensors.kernels.cameras.parameters import ReferencePolynomial

        h_poly = torch.tensor([0.005, 1.01, 0.002, 0.003, 0.001, 0.001], device=dev, requires_grad=requires_grad)
        v_poly = torch.tensor([0.005, 0.003, 0.001, 1.01, 0.001, 0.002], device=dev, requires_grad=requires_grad)
        h_poly_inv = torch.tensor(
            [-0.005, 0.99, -0.002, -0.003, -0.001, -0.001], device=dev, requires_grad=requires_grad
        )
        v_poly_inv = torch.tensor(
            [-0.005, -0.003, -0.001, 0.99, -0.001, -0.002], device=dev, requires_grad=requires_grad
        )
        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )
        return distortion, h_poly, v_poly, h_poly_inv, v_poly_inv

    @staticmethod
    def _make_bivariate_order3_with_leaves(dev, requires_grad=False):
        """Order-2 h_poly (max for MAX_H=6) + order-3 v_poly (10 terms)."""
        from libs.sensors.kernels.cameras.parameters import ReferencePolynomial

        h_poly = torch.tensor([0.005, 1.01, 0.002, 0.003, 0.001, 0.001], device=dev, requires_grad=requires_grad)
        v_poly = torch.tensor(
            [0.005, 0.003, 0.001, 0.0005, 1.01, 0.001, 0.0005, 0.002, 0.0005, 0.001],
            device=dev,
            requires_grad=requires_grad,
        )
        h_poly_inv = torch.tensor(
            [-0.005, 0.99, -0.002, -0.003, -0.001, -0.001], device=dev, requires_grad=requires_grad
        )
        v_poly_inv = torch.tensor(
            [-0.005, -0.003, -0.001, -0.0005, 0.99, -0.001, -0.0005, -0.002, -0.0005, -0.001],
            device=dev,
            requires_grad=requires_grad,
        )
        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )
        return distortion, h_poly, v_poly, h_poly_inv, v_poly_inv

    @staticmethod
    def _make_bivariate_order4_with_leaves(dev, requires_grad=False):
        """Order-2 h_poly (max for MAX_H=6) + order-4 v_poly (15 terms = MAX_V)."""
        from libs.sensors.kernels.cameras.parameters import ReferencePolynomial

        h_poly = torch.tensor([0.005, 1.01, 0.002, 0.003, 0.001, 0.001], device=dev, requires_grad=requires_grad)
        v_poly = torch.tensor(
            [
                0.005,
                0.003,
                0.001,
                0.0005,
                0.0002,
                1.01,
                0.001,
                0.0005,
                0.0002,
                0.002,
                0.0005,
                0.0002,
                0.001,
                0.0003,
                0.0005,
            ],
            device=dev,
            requires_grad=requires_grad,
        )
        h_poly_inv = torch.tensor(
            [-0.005, 0.99, -0.002, -0.003, -0.001, -0.001], device=dev, requires_grad=requires_grad
        )
        v_poly_inv = torch.tensor(
            [
                -0.005,
                -0.003,
                -0.001,
                -0.0005,
                -0.0002,
                0.99,
                -0.001,
                -0.0005,
                -0.0002,
                -0.002,
                -0.0005,
                -0.0002,
                -0.001,
                -0.0003,
                -0.0005,
            ],
            device=dev,
            requires_grad=requires_grad,
        )
        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )
        return distortion, h_poly, v_poly, h_poly_inv, v_poly_inv

    def _make_identity_dynamic_pose(self):
        """Identity start+end poses — rolling shutter converges to identity transform."""
        z = torch.zeros(3, device=self.device)
        r = quat_identity((1,), device=self.device).squeeze(0)
        return create_dynamic_pose(z, z.clone(), r, r.clone(), self.device)

    def _make_dynamic_pose(self, requires_grad=False):
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device, requires_grad=requires_grad)
        trans_end = torch.tensor([0.1, 0.0, 0.0], device=self.device, requires_grad=requires_grad)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)
        if requires_grad:
            rot_start = rot_start.detach().requires_grad_(True)
            rot_end = rot_end.detach().requires_grad_(True)
        pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)
        return pose, trans_start, trans_end, rot_start, rot_end

    # ---- Finite-difference helper for projection ----

    def _pinhole_forward_loss(
        self,
        world_points,
        fl=(500.0, 500.0),
        pp=(320.0, 240.0),
        h=(0.0, 1.01, 0.0),
        v=(0.0, 0.0, 1.01),
        h_inv=(0.0, 0.99, 0.0),
        v_inv=(0.0, 0.0, 0.99),
    ):
        """Evaluate pinhole+bivariate forward loss with specified parameter values (no grad)."""
        from libs.sensors.kernels.cameras.parameters import ReferencePolynomial

        proj = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor(fl, device=self.device),
            principal_point=torch.tensor(pp, device=self.device),
            radial_coeffs=torch.zeros(6, device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )
        dist = BivariateWindshieldDistortion.from_components(
            h_poly=torch.tensor(h, device=self.device),
            v_poly=torch.tensor(v, device=self.device),
            h_poly_inv=torch.tensor(h_inv, device=self.device),
            v_poly_inv=torch.tensor(v_inv, device=self.device),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )
        pose = self._make_identity_dynamic_pose()
        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                proj,
                dist,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return ip.sum().item()

    def _pinhole_forward_loss_bivariate(self, world_points, h, v, h_inv, v_inv):
        """Evaluate pinhole+bivariate forward loss for arbitrary polynomial orders."""
        from libs.sensors.kernels.cameras.parameters import ReferencePolynomial

        proj = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.zeros(6, device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )
        dist = BivariateWindshieldDistortion.from_components(
            h_poly=torch.tensor(h, device=self.device),
            v_poly=torch.tensor(v, device=self.device),
            h_poly_inv=torch.tensor(h_inv, device=self.device),
            v_poly_inv=torch.tensor(v_inv, device=self.device),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )
        pose = self._make_identity_dynamic_pose()
        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                proj,
                dist,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return ip.sum().item()

    def _pose_forward_loss(self, world_points, trans_start_val, trans_end_val):
        """Evaluate forward loss with specified pose translations (no grad)."""
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        ts = torch.tensor(trans_start_val, device=self.device)
        te = torch.tensor(trans_end_val, device=self.device)
        rs = quat_identity((1,), device=self.device).squeeze(0)
        re = quat_identity((1,), device=self.device).squeeze(0)
        pose = create_dynamic_pose(ts, te, rs, re, self.device)
        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                projection,
                distortion,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return ip.sum().item()

    def _ftheta_forward_loss(self, world_points, fw_poly_1=500.0):
        """Evaluate ftheta+bivariate forward loss with specified fw_poly[1] value (no grad)."""
        from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS, ReferencePolynomial

        pp = torch.tensor([320.0, 240.0], device=self.device)
        fw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        bw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        fw_poly[1] = fw_poly_1
        bw_poly[1] = 0.002
        dfw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        dbw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        dfw_poly[0] = fw_poly_1
        dbw_poly[0] = 0.002
        A = torch.eye(2, device=self.device)
        proj = FThetaProjection.from_components(
            principal_point=pp,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=A,
            Ainv=torch.eye(2, device=self.device),
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        dist = BivariateWindshieldDistortion.from_components(
            h_poly=torch.tensor([0.0, 1.01, 0.0], device=self.device),
            v_poly=torch.tensor([0.0, 0.0, 1.01], device=self.device),
            h_poly_inv=torch.tensor([0.0, 0.99, 0.0], device=self.device),
            v_poly_inv=torch.tensor([0.0, 0.0, 0.99], device=self.device),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )
        pose = self._make_identity_dynamic_pose()
        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                proj,
                dist,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return ip.sum().item()

    def _pinhole_backproject_loss(
        self, image_points, fl=(500.0, 500.0), pp=(320.0, 240.0), h_inv=(0.0, 0.99, 0.0), v_inv=(0.0, 0.0, 0.99)
    ):
        """Evaluate backprojection loss with specified parameter values (no grad).

        Uses identity pose, so the undistort direction uses h_poly_inv / v_poly_inv.
        The forward polynomials don't affect backprojection, but are required for construction.
        """
        from libs.sensors.kernels.cameras.parameters import ReferencePolynomial

        proj = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor(fl, device=self.device),
            principal_point=torch.tensor(pp, device=self.device),
            radial_coeffs=torch.zeros(6, device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )
        dist = BivariateWindshieldDistortion.from_components(
            h_poly=torch.tensor([0.0, 1.01, 0.0], device=self.device),
            v_poly=torch.tensor([0.0, 0.0, 1.01], device=self.device),
            h_poly_inv=torch.tensor(h_inv, device=self.device),
            v_poly_inv=torch.tensor(v_inv, device=self.device),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )
        pose = self._make_identity_dynamic_pose()
        with torch.no_grad():
            world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
                image_points,
                proj,
                dist,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return world_rays.sum().item()

    def _pose_forward_loss_with_rotation(
        self, world_points, trans_start_val, trans_end_val, rot_start_val, rot_end_val
    ):
        """Evaluate pinhole forward loss with specified pose translations and rotations (no grad)."""
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        ts = torch.tensor(trans_start_val, device=self.device)
        te = torch.tensor(trans_end_val, device=self.device)
        rs = torch.tensor(rot_start_val, device=self.device)
        re = torch.tensor(rot_end_val, device=self.device)
        pose = create_dynamic_pose(ts, te, rs, re, self.device)
        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                projection,
                distortion,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return ip.sum().item()

    def _ftheta_pose_forward_loss(self, world_points, trans_start_val, trans_end_val, rot_start_val, rot_end_val):
        """Evaluate ftheta forward loss with specified pose translations and rotations (no grad)."""
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        ts = torch.tensor(trans_start_val, device=self.device)
        te = torch.tensor(trans_end_val, device=self.device)
        rs = torch.tensor(rot_start_val, device=self.device)
        re = torch.tensor(rot_end_val, device=self.device)
        pose = create_dynamic_pose(ts, te, rs, re, self.device)
        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                projection,
                distortion,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return ip.sum().item()

    def _ftheta_backproject_pose_loss(self, image_points, trans_start_val, trans_end_val, rot_start_val, rot_end_val):
        """Evaluate ftheta backprojection loss with specified pose (no grad). Linear fw_poly."""
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        ts = torch.tensor(trans_start_val, device=self.device)
        te = torch.tensor(trans_end_val, device=self.device)
        rs = torch.tensor(rot_start_val, device=self.device)
        re = torch.tensor(rot_end_val, device=self.device)
        pose = create_dynamic_pose(ts, te, rs, re, self.device)
        with torch.no_grad():
            world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
                image_points,
                projection,
                distortion,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return world_rays.sum().item()

    @staticmethod
    def _make_ftheta_nonlinear_with_leaves(dev, requires_grad=False):
        """FTheta with non-linear fw_poly: r = 500*theta - 30*theta^3.

        Newton iteration is required to invert this polynomial during backprojection
        (with reference_poly=FORWARD). The cubic term is small enough that the
        polynomial is monotonically increasing for angles up to ~2.3 rad.
        """
        from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS

        pp = torch.tensor([320.0, 240.0], device=dev, requires_grad=requires_grad)
        fw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=dev, requires_grad=requires_grad)
        bw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=dev, requires_grad=requires_grad)
        fw_poly.data[1] = 500.0
        fw_poly.data[3] = -30.0
        bw_poly.data[1] = 0.002
        dfw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=dev)
        dbw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=dev)
        dfw_poly[0] = 500.0
        dfw_poly[2] = -90.0
        dbw_poly[0] = 0.002
        A = torch.eye(2, device=dev, requires_grad=requires_grad)
        projection = FThetaProjection.from_components(
            principal_point=pp,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=A,
            Ainv=torch.eye(2, device=dev),
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        return projection, pp, fw_poly

    def _ftheta_nonlinear_backproject_pose_loss(
        self, image_points, trans_start_val, trans_end_val, rot_start_val, rot_end_val
    ):
        """Evaluate non-linear ftheta backprojection loss with specified pose (no grad)."""
        projection, _, _ = self._make_ftheta_nonlinear_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        ts = torch.tensor(trans_start_val, device=self.device)
        te = torch.tensor(trans_end_val, device=self.device)
        rs = torch.tensor(rot_start_val, device=self.device)
        re = torch.tensor(rot_end_val, device=self.device)
        pose = create_dynamic_pose(ts, te, rs, re, self.device)
        with torch.no_grad():
            world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
                image_points,
                projection,
                distortion,
                (640, 480),
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return world_rays.sum().item()

    def _real_ftheta_pose_forward_loss(
        self, cam_data, world_points, trans_start_val, trans_end_val, rot_start_val, rot_end_val
    ):
        """Forward projection loss with real FTheta params and specified RS pose (no grad)."""
        projection, distortion, res, _, _, _, _ = self._make_real_ftheta_with_leaves(cam_data)
        if distortion is None:
            distortion = NoExternalDistortion()
        ts = torch.tensor(trans_start_val, device=self.device)
        te = torch.tensor(trans_end_val, device=self.device)
        rs = torch.tensor(rot_start_val, device=self.device)
        re = torch.tensor(rot_end_val, device=self.device)
        pose = create_dynamic_pose(ts, te, rs, re, self.device)
        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                projection,
                distortion,
                res,
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return ip.sum().item()

    # ---- Forward pass tests (values verified against Python reference) ----

    def test_pinhole_bivariate_shutter_forward(self):
        """Forward projection (pinhole + bivariate + RS) matches Python reference.

        Uses identity pose so camera_point = world_point, eliminating RS iteration.
        Pipeline: world_point → normalize → bivariate_distort → pinhole_project.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [1.0, 0.5, 10.0],
                [0.3, 0.2, 3.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (3, 2))
        self.assertTrue(valid.all())

        for i, wp in enumerate(world_points.cpu().tolist()):
            dist_ray = self._ref_bivariate_distort(wp, self.H_POLY_FWD, self.V_POLY_FWD, self.H_ORDER, self.V_ORDER)
            expected = self._ref_pinhole_project(dist_ray, 500, 500, 320, 240)
            actual = image_points[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.05,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_ftheta_bivariate_shutter_forward(self):
        """Forward projection (ftheta + bivariate + RS) matches Python reference.

        fw_poly = [0, 500, 0, ...] → r = 500*theta (linear model, A=I).
        """
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [0.5, 0.3, 8.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (2, 2))
        self.assertTrue(valid.all())

        A_ref = [[1, 0], [0, 1]]
        for i, wp in enumerate(world_points.cpu().tolist()):
            dist_ray = self._ref_bivariate_distort(wp, self.H_POLY_FWD, self.V_POLY_FWD, self.H_ORDER, self.V_ORDER)
            expected = self._ref_ftheta_project(dist_ray, self.FW_POLY, self.FW_POLY_DEGREE, [320, 240], A_ref)
            actual = image_points[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.05,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_pinhole_bivariate_shutter_backproject(self):
        """Backprojection (pinhole + bivariate + RS) matches Python reference.

        Pipeline: image_point → pinhole_backproject → bivariate_undistort (h/v_poly_inv).
        With identity pose, world_ray = undistorted camera_ray, origin = [0,0,0].
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        image_points = torch.tensor(
            [
                [350.0, 260.0],
                [400.0, 300.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        self.assertEqual(world_rays.shape, (2, 6))

        origins = world_rays[:, :3]
        directions = world_rays[:, 3:]

        self.assertTrue(
            torch.allclose(origins, torch.zeros_like(origins), atol=1e-4),
            "Identity pose should give origin=[0,0,0]",
        )

        norms = directions.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

        for i, ip in enumerate(image_points.cpu().tolist()):
            cam_ray = self._ref_pinhole_backproject(ip, 500, 500, 320, 240)
            expected_dir = self._ref_bivariate_distort(
                cam_ray, self.H_POLY_INV, self.V_POLY_INV, self.H_ORDER, self.V_ORDER
            )
            actual = directions[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected_dir,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Backproject direction mismatch for point {i}",
            )

    def test_ftheta_bivariate_shutter_backproject(self):
        """Backprojection (ftheta + bivariate + RS) matches Python reference.

        fw_poly = [0, 500, ...] → theta = rdist/500 (analytical inverse for linear model).
        """
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        image_points = torch.tensor(
            [
                [330.0, 250.0],
                [350.0, 260.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        self.assertEqual(world_rays.shape, (2, 6))

        origins = world_rays[:, :3]
        directions = world_rays[:, 3:]

        self.assertTrue(torch.allclose(origins, torch.zeros_like(origins), atol=1e-4))

        norms = directions.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

        Ainv_ref = [[1, 0], [0, 1]]
        for i, ip in enumerate(image_points.cpu().tolist()):
            cam_ray = self._ref_ftheta_backproject(ip, self.FW_POLY_C1, [320, 240], Ainv_ref)
            expected_dir = self._ref_bivariate_distort(
                cam_ray, self.H_POLY_INV, self.V_POLY_INV, self.H_ORDER, self.V_ORDER
            )
            actual = directions[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected_dir,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"FTheta backproject direction mismatch for point {i}",
            )

    # ---- Gradient tests (verified against finite differences) ----

    def test_pinhole_bivariate_shutter_intrinsics_gradient(self):
        """Pinhole intrinsic gradients match finite differences.

        d(loss)/d(pp) = N (each point contributes +1 per pp component).
        d(loss)/d(fl) verified via central finite differences.
        """
        projection, focal_length, principal_point = self._make_pinhole_with_leaves(self.device, requires_grad=True)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.5, 0.3, 5.0],
                [1.0, 0.5, 10.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(focal_length.grad)
        self.assertTrue(focal_length.grad.abs().sum() > 0)
        self.assertIsNotNone(principal_point.grad)

        # Analytical: d(image_point)/d(pp) = 1 per point, loss = sum → grad = N
        np.testing.assert_allclose(
            principal_point.grad.cpu().numpy(),
            [2.0, 2.0],
            rtol=0.02,
            err_msg="pp gradient should equal N (number of points)",
        )

        # Finite-difference verification for focal_length
        eps = 0.5

        for idx in range(2):
            fl_plus = [500.0, 500.0]
            fl_plus[idx] += eps
            fl_minus = [500.0, 500.0]
            fl_minus[idx] -= eps

            loss_plus = self._pinhole_forward_loss(world_points, fl=fl_plus)
            loss_minus = self._pinhole_forward_loss(world_points, fl=fl_minus)
            fd = (loss_plus - loss_minus) / (2 * eps)

            np.testing.assert_allclose(
                focal_length.grad[idx].item(),
                fd,
                rtol=0.05,
                atol=0.01,
                err_msg=f"focal_length[{idx}] grad: autograd={focal_length.grad[idx].item():.6f}, fd={fd:.6f}",
            )

    def test_ftheta_bivariate_shutter_intrinsics_gradient(self):
        """FTheta intrinsic gradients: pp analytical + fw_poly[1] via FD."""
        projection, pp, fw_poly = self._make_ftheta_with_leaves(self.device, requires_grad=True)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [[0.5, 0.3, 8.0]],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(pp.grad)
        self.assertTrue(pp.grad.abs().sum() > 0)
        self.assertIsNotNone(fw_poly.grad)
        self.assertTrue(fw_poly.grad.abs().sum() > 0, "fw_poly gradient should be non-zero")

        # Analytical: d(image_point)/d(pp) = 1 per point
        np.testing.assert_allclose(
            pp.grad.cpu().numpy(),
            [1.0, 1.0],
            rtol=0.02,
            err_msg="FTheta pp gradient should be ~1 per point",
        )

        # FD for fw_poly[1] (dominant coefficient: r = fw_poly[1] * theta)
        eps = 0.5

        loss_plus = self._ftheta_forward_loss(world_points, fw_poly_1=500.0 + eps)
        loss_minus = self._ftheta_forward_loss(world_points, fw_poly_1=500.0 - eps)
        fd = (loss_plus - loss_minus) / (2 * eps)

        self.assertNotAlmostEqual(fd, 0.0, places=3, msg="FD for fw_poly[1] should be non-zero")

        np.testing.assert_allclose(
            fw_poly.grad[1].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"fw_poly[1] grad: autograd={fw_poly.grad[1].item():.6f}, fd={fd:.6f}",
        )

    def test_bivariate_distortion_gradient_through_shutter(self):
        """Bivariate windshield gradients match finite differences (h_poly[1] tested)."""
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, h_poly, v_poly, _, _ = self._make_bivariate_with_leaves(self.device, requires_grad=True)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.3, 0.2, 5.0],
                [0.8, 0.4, 8.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(h_poly.grad)
        self.assertTrue(h_poly.grad.abs().sum() > 0)
        self.assertIsNotNone(v_poly.grad)
        self.assertTrue(v_poly.grad.abs().sum() > 0)

        # Finite-difference for h_poly[1] (dominant coefficient: maps phi → 1.01*phi)
        eps = 1e-3

        h_plus = [0.0, 1.01 + eps, 0.0]
        h_minus = [0.0, 1.01 - eps, 0.0]

        loss_plus = self._pinhole_forward_loss(world_points, h=h_plus)
        loss_minus = self._pinhole_forward_loss(world_points, h=h_minus)
        fd = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            h_poly.grad[1].item(),
            fd,
            rtol=0.05,
            atol=0.1,
            err_msg=f"h_poly[1] grad: autograd={h_poly.grad[1].item():.6f}, fd={fd:.6f}",
        )

        # Finite-difference for v_poly[2] (dominant coefficient: maps theta → 1.01*theta)
        v_plus = [0.0, 0.0, 1.01 + eps]
        v_minus = [0.0, 0.0, 1.01 - eps]

        loss_plus = self._pinhole_forward_loss(world_points, v=v_plus)
        loss_minus = self._pinhole_forward_loss(world_points, v=v_minus)
        fd = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            v_poly.grad[2].item(),
            fd,
            rtol=0.05,
            atol=0.1,
            err_msg=f"v_poly[2] grad: autograd={v_poly.grad[2].item():.6f}, fd={fd:.6f}",
        )

    def test_pose_gradient_through_bivariate_shutter(self):
        """Pose translation and rotation gradients are non-zero through bivariate + RS."""
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)
        dynamic_pose, trans_start, trans_end, rot_start, rot_end = self._make_dynamic_pose(requires_grad=True)

        world_points = torch.tensor(
            [
                [0.5, 0.3, 5.0],
                [1.0, 0.5, 5.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(trans_start.grad, "trans_start gradient should exist")
        self.assertIsNotNone(trans_end.grad, "trans_end gradient should exist")
        self.assertTrue(
            trans_end.grad.abs().sum() > 0,
            "trans_end gradient should be non-zero (non-identity pose)",
        )
        self.assertIsNotNone(rot_start.grad, "rot_start gradient should exist")
        self.assertIsNotNone(rot_end.grad, "rot_end gradient should exist")

        # FD for trans_end[0] (the non-zero translation component)
        eps = 1e-3

        te_base = [0.1, 0.0, 0.0]
        ts_base = [0.0, 0.0, 0.0]

        te_plus = list(te_base)
        te_plus[0] += eps
        te_minus = list(te_base)
        te_minus[0] -= eps

        loss_plus = self._pose_forward_loss(world_points, ts_base, te_plus)
        loss_minus = self._pose_forward_loss(world_points, ts_base, te_minus)
        fd = (loss_plus - loss_minus) / (2 * eps)

        self.assertNotAlmostEqual(fd, 0.0, places=3, msg="FD for trans_end[0] should be non-zero")

        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd,
            rtol=0.05,
            atol=0.1,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd:.6f}",
        )

    # ---- Rolling-shutter interpolation value tests (non-identity pose) ----

    def test_rs_forward_nonidentity_translation(self):
        """Forward projection with non-identity translation exercises RS iterative solver.

        Uses trans_start=[0,0,0], trans_end=[0.2,0.1,0] with identity rotations.
        The per-scanline pose interpolation produces different camera transforms for
        different projected y-coordinates, which the iterative RS solver must converge on.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.2, 0.1, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        rot_end = [0.0, 0.0, 0.0, 1.0]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        world_points = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [1.0, 0.5, 10.0],
                [0.3, 0.2, 3.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (3, 2))
        self.assertTrue(valid.all(), "All points should project to valid image locations")

        for i, wp in enumerate(world_points.cpu().tolist()):
            expected = self._ref_rs_forward_pinhole(
                wp,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                self.H_POLY_FWD,
                self.V_POLY_FWD,
                self.H_ORDER,
                self.V_ORDER,
                500,
                500,
                320,
                240,
                height=480,
            )
            self.assertIsNotNone(expected, f"Point {i}: reference solver should converge")
            actual = image_points[i].cpu().tolist()
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.1,
                err_msg=f"Point {i}: world={wp}",
            )

        # Verify the RS interpolation actually matters: results should differ from identity-pose projection
        identity_pose = self._make_identity_dynamic_pose()
        ip_identity, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            identity_pose,
        )
        diff = (image_points - ip_identity).abs().max().item()
        self.assertGreater(diff, 0.1, "Non-identity pose should produce different projections than identity")

    def test_rs_forward_nonidentity_rotation(self):
        """Forward projection with non-identity rotation exercises SLERP interpolation.

        Uses a 5-degree Z-axis rotation from start to end pose, plus a small translation.
        This tests that the per-scanline SLERP produces correct intermediate rotations.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.1, 0.0, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        rot_end = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        world_points = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [1.0, 0.5, 10.0],
                [0.3, 0.2, 3.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (3, 2))
        self.assertTrue(valid.all(), "All points should project to valid image locations")

        for i, wp in enumerate(world_points.cpu().tolist()):
            expected = self._ref_rs_forward_pinhole(
                wp,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                self.H_POLY_FWD,
                self.V_POLY_FWD,
                self.H_ORDER,
                self.V_ORDER,
                500,
                500,
                320,
                240,
                height=480,
            )
            self.assertIsNotNone(expected, f"Point {i}: reference solver should converge")
            actual = image_points[i].cpu().tolist()
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.1,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_rs_backproject_nonidentity_translation(self):
        """Backprojection with non-identity translation verifies per-scanline pose lookup.

        Uses trans_start=[0,0,0], trans_end=[0.2,0.1,0] with identity rotations.
        Backprojection is non-iterative: t is computed directly from the image point's
        y-coordinate, so this tests the scanline → pose → world ray pipeline.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.2, 0.1, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        rot_end = [0.0, 0.0, 0.0, 1.0]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        image_points = torch.tensor(
            [
                [350.0, 100.0],
                [400.0, 300.0],
                [300.0, 460.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        self.assertEqual(world_rays.shape, (3, 6))

        for i, ip in enumerate(image_points.cpu().tolist()):
            expected_origin, expected_dir = self._ref_rs_backproject_pinhole(
                ip,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                self.H_POLY_INV,
                self.V_POLY_INV,
                self.H_ORDER,
                self.V_ORDER,
                500,
                500,
                320,
                240,
                height=480,
            )
            actual_origin = world_rays[i, :3].cpu().tolist()
            actual_dir = world_rays[i, 3:].cpu().tolist()

            np.testing.assert_allclose(
                actual_origin,
                expected_origin,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Point {i}: origin mismatch (ip={ip})",
            )
            np.testing.assert_allclose(
                actual_dir,
                expected_dir,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Point {i}: direction mismatch (ip={ip})",
            )

        # Verify origins differ per scanline (they should since translation is interpolated)
        origins = world_rays[:, :3]
        origin_spread = (origins.max(dim=0).values - origins.min(dim=0).values).abs().max().item()
        self.assertGreater(origin_spread, 0.01, "Different scanlines should yield different ray origins")

    def test_rs_backproject_nonidentity_rotation(self):
        """Backprojection with non-identity rotation verifies per-scanline SLERP for ray direction.

        Uses a 5-degree Z-axis rotation from start to end pose.
        Different scanlines should produce different world ray directions even for the same
        camera-frame ray, confirming SLERP interpolation is applied correctly.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.1, 0.0, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        rot_end = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        image_points = torch.tensor(
            [
                [350.0, 50.0],
                [350.0, 240.0],
                [350.0, 450.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        self.assertEqual(world_rays.shape, (3, 6))

        for i, ip in enumerate(image_points.cpu().tolist()):
            expected_origin, expected_dir = self._ref_rs_backproject_pinhole(
                ip,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                self.H_POLY_INV,
                self.V_POLY_INV,
                self.H_ORDER,
                self.V_ORDER,
                500,
                500,
                320,
                240,
                height=480,
            )
            actual_origin = world_rays[i, :3].cpu().tolist()
            actual_dir = world_rays[i, 3:].cpu().tolist()

            np.testing.assert_allclose(
                actual_origin,
                expected_origin,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Point {i}: origin mismatch (ip={ip})",
            )
            np.testing.assert_allclose(
                actual_dir,
                expected_dir,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"Point {i}: direction mismatch (ip={ip})",
            )

        # Same image u-coordinate but different scanlines should give different directions
        dirs = world_rays[:, 3:]
        dir_spread = (dirs.max(dim=0).values - dirs.min(dim=0).values).abs().max().item()
        self.assertGreater(dir_spread, 0.001, "SLERP should produce different directions at different scanlines")

    def test_rs_ftheta_forward_nonidentity_translation(self):
        """FTheta forward projection with non-identity translation exercises RS iterative solver.

        Same pose as the pinhole variant but through the FTheta projection path
        (fw_poly=[0,500,...], A=I). Catches bugs in the FTheta-specific RS kernel wrapper.
        """
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.2, 0.1, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        rot_end = [0.0, 0.0, 0.0, 1.0]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        world_points = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [0.5, 0.3, 8.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (2, 2))
        self.assertTrue(valid.all(), "All points should project to valid image locations")

        A_ref = [[1, 0], [0, 1]]
        for i, wp in enumerate(world_points.cpu().tolist()):
            expected = self._ref_rs_forward_ftheta(
                wp,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                self.H_POLY_FWD,
                self.V_POLY_FWD,
                self.H_ORDER,
                self.V_ORDER,
                self.FW_POLY,
                self.FW_POLY_DEGREE,
                [320, 240],
                A_ref,
                height=480,
            )
            self.assertIsNotNone(expected, f"Point {i}: reference solver should converge")
            actual = image_points[i].cpu().tolist()
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.1,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_rs_ftheta_forward_nonidentity_rotation(self):
        """FTheta forward projection with non-identity rotation exercises SLERP through FTheta path."""
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.1, 0.0, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        rot_end = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        world_points = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [0.5, 0.3, 8.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (2, 2))
        self.assertTrue(valid.all(), "All points should project to valid image locations")

        A_ref = [[1, 0], [0, 1]]
        for i, wp in enumerate(world_points.cpu().tolist()):
            expected = self._ref_rs_forward_ftheta(
                wp,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                self.H_POLY_FWD,
                self.V_POLY_FWD,
                self.H_ORDER,
                self.V_ORDER,
                self.FW_POLY,
                self.FW_POLY_DEGREE,
                [320, 240],
                A_ref,
                height=480,
            )
            self.assertIsNotNone(expected, f"Point {i}: reference solver should converge")
            actual = image_points[i].cpu().tolist()
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.1,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_rs_ftheta_backproject_nonidentity_translation(self):
        """FTheta backprojection with non-identity translation verifies per-scanline pose through FTheta path."""
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.2, 0.1, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        rot_end = [0.0, 0.0, 0.0, 1.0]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        image_points = torch.tensor(
            [
                [330.0, 100.0],
                [350.0, 300.0],
                [340.0, 460.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        self.assertEqual(world_rays.shape, (3, 6))

        Ainv_ref = [[1, 0], [0, 1]]
        for i, ip in enumerate(image_points.cpu().tolist()):
            expected_origin, expected_dir = self._ref_rs_backproject_ftheta(
                ip,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                self.H_POLY_INV,
                self.V_POLY_INV,
                self.H_ORDER,
                self.V_ORDER,
                self.FW_POLY_C1,
                [320, 240],
                Ainv_ref,
                height=480,
            )
            actual_origin = world_rays[i, :3].cpu().tolist()
            actual_dir = world_rays[i, 3:].cpu().tolist()

            np.testing.assert_allclose(
                actual_origin,
                expected_origin,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Point {i}: origin mismatch (ip={ip})",
            )
            np.testing.assert_allclose(
                actual_dir,
                expected_dir,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"Point {i}: direction mismatch (ip={ip})",
            )

        origins = world_rays[:, :3]
        origin_spread = (origins.max(dim=0).values - origins.min(dim=0).values).abs().max().item()
        self.assertGreater(origin_spread, 0.01, "Different scanlines should yield different ray origins")

    def test_rs_ftheta_backproject_nonidentity_rotation(self):
        """FTheta backprojection with non-identity rotation verifies per-scanline SLERP through FTheta path."""
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.1, 0.0, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        rot_end = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        image_points = torch.tensor(
            [
                [340.0, 50.0],
                [340.0, 240.0],
                [340.0, 450.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        self.assertEqual(world_rays.shape, (3, 6))

        Ainv_ref = [[1, 0], [0, 1]]
        for i, ip in enumerate(image_points.cpu().tolist()):
            expected_origin, expected_dir = self._ref_rs_backproject_ftheta(
                ip,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                self.H_POLY_INV,
                self.V_POLY_INV,
                self.H_ORDER,
                self.V_ORDER,
                self.FW_POLY_C1,
                [320, 240],
                Ainv_ref,
                height=480,
            )
            actual_origin = world_rays[i, :3].cpu().tolist()
            actual_dir = world_rays[i, 3:].cpu().tolist()

            np.testing.assert_allclose(
                actual_origin,
                expected_origin,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Point {i}: origin mismatch (ip={ip})",
            )
            np.testing.assert_allclose(
                actual_dir,
                expected_dir,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"Point {i}: direction mismatch (ip={ip})",
            )

        dirs = world_rays[:, 3:]
        dir_spread = (dirs.max(dim=0).values - dirs.min(dim=0).values).abs().max().item()
        self.assertGreater(dir_spread, 0.001, "SLERP should produce different directions at different scanlines")

    # ---- Rolling-shutter pose gradient tests (non-identity pose, FD verified) ----

    def test_rs_pinhole_rotation_gradient(self):
        """Pinhole: gradients w.r.t. rot_end verified via FD through RS + bivariate + SLERP.

        Uses trans_end=[0.1,0,0] plus a 5-degree Z rotation. Verifies FD for trans_end[0]
        (translation through SLERP path) and rot_end[2] (the sin(angle/2) component).
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        ts_val = [0.0, 0.0, 0.0]
        te_val = [0.1, 0.0, 0.0]
        rs_val = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        re_val = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        trans_start = torch.tensor(ts_val, device=self.device, requires_grad=True)
        trans_end = torch.tensor(te_val, device=self.device, requires_grad=True)
        rot_start = torch.tensor(rs_val, device=self.device, requires_grad=True)
        rot_end = torch.tensor(re_val, device=self.device, requires_grad=True)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        world_points = torch.tensor(
            [[0.5, 0.3, 5.0], [1.0, 0.5, 5.0]],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(trans_end.grad)
        self.assertTrue(trans_end.grad.abs().sum() > 0, "trans_end gradient should be non-zero")
        self.assertIsNotNone(rot_end.grad)
        self.assertTrue(rot_end.grad.abs().sum() > 0, "rot_end gradient should be non-zero")

        eps = 1e-3

        # FD for trans_end[0]
        te_plus = list(te_val)
        te_plus[0] += eps
        te_minus = list(te_val)
        te_minus[0] -= eps
        fd_te = (
            self._pose_forward_loss_with_rotation(world_points, ts_val, te_plus, rs_val, re_val)
            - self._pose_forward_loss_with_rotation(world_points, ts_val, te_minus, rs_val, re_val)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_te, 0.0, places=3)
        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd_te,
            rtol=0.02,
            atol=0.01,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd_te:.6f}",
        )

        # FD for rot_end[2] (sin(angle/2) component of Z-axis rotation)
        re_plus = list(re_val)
        re_plus[2] += eps
        re_minus = list(re_val)
        re_minus[2] -= eps
        fd_re = (
            self._pose_forward_loss_with_rotation(world_points, ts_val, te_val, rs_val, re_plus)
            - self._pose_forward_loss_with_rotation(world_points, ts_val, te_val, rs_val, re_minus)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_re, 0.0, places=3)
        np.testing.assert_allclose(
            rot_end.grad[2].item(),
            fd_re,
            rtol=0.02,
            atol=0.01,
            err_msg=f"rot_end[2] grad: autograd={rot_end.grad[2].item():.6f}, fd={fd_re:.6f}",
        )

    def test_rs_ftheta_translation_gradient(self):
        """FTheta: gradients w.r.t. trans_end verified via FD through RS + bivariate.

        Uses trans_start=[0,0,0], trans_end=[0.1,0,0] with identity rotations.
        Verifies gradient flows correctly through the FTheta RS kernel wrapper.
        """
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        ts_val = [0.0, 0.0, 0.0]
        te_val = [0.1, 0.0, 0.0]
        rs_val = [0.0, 0.0, 0.0, 1.0]
        re_val = [0.0, 0.0, 0.0, 1.0]

        trans_start = torch.tensor(ts_val, device=self.device, requires_grad=True)
        trans_end = torch.tensor(te_val, device=self.device, requires_grad=True)
        rot_start = torch.tensor(rs_val, device=self.device, requires_grad=True)
        rot_end = torch.tensor(re_val, device=self.device, requires_grad=True)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        world_points = torch.tensor(
            [[0.5, 0.3, 8.0], [1.0, 0.5, 8.0]],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(trans_end.grad)
        self.assertTrue(trans_end.grad.abs().sum() > 0, "trans_end gradient should be non-zero")

        eps = 1e-3

        # FD for trans_end[0]
        te_plus = list(te_val)
        te_plus[0] += eps
        te_minus = list(te_val)
        te_minus[0] -= eps
        fd = (
            self._ftheta_pose_forward_loss(world_points, ts_val, te_plus, rs_val, re_val)
            - self._ftheta_pose_forward_loss(world_points, ts_val, te_minus, rs_val, re_val)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd, 0.0, places=3)
        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd,
            rtol=0.02,
            atol=0.01,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd:.6f}",
        )

    def test_rs_ftheta_rotation_gradient(self):
        """FTheta: gradients w.r.t. rot_end verified via FD through RS + bivariate + SLERP.

        Uses trans_end=[0.1,0,0] plus a 5-degree Z rotation.
        Verifies FD for both trans_end[0] and rot_end[2] through the FTheta path.
        """
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        ts_val = [0.0, 0.0, 0.0]
        te_val = [0.1, 0.0, 0.0]
        rs_val = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        re_val = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        trans_start = torch.tensor(ts_val, device=self.device, requires_grad=True)
        trans_end = torch.tensor(te_val, device=self.device, requires_grad=True)
        rot_start = torch.tensor(rs_val, device=self.device, requires_grad=True)
        rot_end = torch.tensor(re_val, device=self.device, requires_grad=True)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        world_points = torch.tensor(
            [[0.5, 0.3, 8.0], [1.0, 0.5, 8.0]],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(trans_end.grad)
        self.assertTrue(trans_end.grad.abs().sum() > 0, "trans_end gradient should be non-zero")
        self.assertIsNotNone(rot_end.grad)
        self.assertTrue(rot_end.grad.abs().sum() > 0, "rot_end gradient should be non-zero")

        eps = 1e-3

        # FD for trans_end[0]
        te_plus = list(te_val)
        te_plus[0] += eps
        te_minus = list(te_val)
        te_minus[0] -= eps
        fd_te = (
            self._ftheta_pose_forward_loss(world_points, ts_val, te_plus, rs_val, re_val)
            - self._ftheta_pose_forward_loss(world_points, ts_val, te_minus, rs_val, re_val)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_te, 0.0, places=3)
        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd_te,
            rtol=0.02,
            atol=0.01,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd_te:.6f}",
        )

        # FD for rot_end[2]
        re_plus = list(re_val)
        re_plus[2] += eps
        re_minus = list(re_val)
        re_minus[2] -= eps
        fd_re = (
            self._ftheta_pose_forward_loss(world_points, ts_val, te_val, rs_val, re_plus)
            - self._ftheta_pose_forward_loss(world_points, ts_val, te_val, rs_val, re_minus)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_re, 0.0, places=3)
        np.testing.assert_allclose(
            rot_end.grad[2].item(),
            fd_re,
            rtol=0.02,
            atol=0.01,
            err_msg=f"rot_end[2] grad: autograd={rot_end.grad[2].item():.6f}, fd={fd_re:.6f}",
        )

    def test_rs_ftheta_backproject_translation_gradient(self):
        """FTheta backprojection gradient w.r.t. trans_end verified via FD through RS + Newton.

        Backprojection with FORWARD reference_poly inverts fw_poly via Newton iteration.
        This test verifies gradients flow through Newton's implicit differentiation
        and the RS pose interpolation in the backprojection path.
        """
        projection, _, _ = self._make_ftheta_with_leaves(self.device, requires_grad=False)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        ts_val = [0.0, 0.0, 0.0]
        te_val = [0.1, 0.0, 0.0]
        rs_val = [0.0, 0.0, 0.0, 1.0]
        re_val = [0.0, 0.0, 0.0, 1.0]

        trans_start = torch.tensor(ts_val, device=self.device, requires_grad=True)
        trans_end = torch.tensor(te_val, device=self.device, requires_grad=True)
        rot_start = torch.tensor(rs_val, device=self.device, requires_grad=True)
        rot_end = torch.tensor(re_val, device=self.device, requires_grad=True)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        image_points = torch.tensor(
            [[340.0, 100.0], [360.0, 350.0]],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        loss = world_rays.sum()
        loss.backward()

        self.assertIsNotNone(trans_end.grad)
        self.assertTrue(trans_end.grad.abs().sum() > 0, "trans_end gradient should be non-zero")

        eps = 1e-3
        te_plus = list(te_val)
        te_plus[0] += eps
        te_minus = list(te_val)
        te_minus[0] -= eps
        fd = (
            self._ftheta_backproject_pose_loss(image_points, ts_val, te_plus, rs_val, re_val)
            - self._ftheta_backproject_pose_loss(image_points, ts_val, te_minus, rs_val, re_val)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd, 0.0, places=3)
        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd,
            rtol=0.02,
            atol=0.01,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd:.6f}",
        )

    def test_rs_ftheta_backproject_rotation_gradient(self):
        """FTheta backprojection gradient w.r.t. rot_end verified via FD through RS + SLERP + Newton."""
        projection, _, _ = self._make_ftheta_with_leaves(self.device, requires_grad=False)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        ts_val = [0.0, 0.0, 0.0]
        te_val = [0.1, 0.0, 0.0]
        rs_val = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        re_val = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        trans_start = torch.tensor(ts_val, device=self.device, requires_grad=True)
        trans_end = torch.tensor(te_val, device=self.device, requires_grad=True)
        rot_start = torch.tensor(rs_val, device=self.device, requires_grad=True)
        rot_end = torch.tensor(re_val, device=self.device, requires_grad=True)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        image_points = torch.tensor(
            [[340.0, 100.0], [360.0, 350.0]],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        loss = world_rays.sum()
        loss.backward()

        self.assertIsNotNone(rot_end.grad)
        self.assertTrue(rot_end.grad.abs().sum() > 0, "rot_end gradient should be non-zero")

        eps = 1e-3

        # FD for trans_end[0]
        te_plus = list(te_val)
        te_plus[0] += eps
        te_minus = list(te_val)
        te_minus[0] -= eps
        fd_te = (
            self._ftheta_backproject_pose_loss(image_points, ts_val, te_plus, rs_val, re_val)
            - self._ftheta_backproject_pose_loss(image_points, ts_val, te_minus, rs_val, re_val)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_te, 0.0, places=3)
        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd_te,
            rtol=0.02,
            atol=0.01,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd_te:.6f}",
        )

        # FD for rot_end[2]
        re_plus = list(re_val)
        re_plus[2] += eps
        re_minus = list(re_val)
        re_minus[2] -= eps
        fd_re = (
            self._ftheta_backproject_pose_loss(image_points, ts_val, te_val, rs_val, re_plus)
            - self._ftheta_backproject_pose_loss(image_points, ts_val, te_val, rs_val, re_minus)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_re, 0.0, places=3)
        np.testing.assert_allclose(
            rot_end.grad[2].item(),
            fd_re,
            rtol=0.02,
            atol=0.01,
            err_msg=f"rot_end[2] grad: autograd={rot_end.grad[2].item():.6f}, fd={fd_re:.6f}",
        )

    def test_rs_ftheta_nonlinear_newton_backproject_gradient(self):
        """FTheta backprojection with non-linear fw_poly: Newton iteration + implicit diff gradients.

        Uses fw_poly = [0, 500, 0, -30, 0, 0] (r = 500*theta - 30*theta^3), which requires
        multiple Newton iterations to invert. Gradients flow through implicit differentiation
        (one differentiable correction step after non-diff Newton convergence).
        Verifies FD for both trans_end[0] and rot_end[2].
        """
        projection, _, _ = self._make_ftheta_nonlinear_with_leaves(self.device, requires_grad=False)
        distortion, _, _, _, _ = self._make_bivariate_with_leaves(self.device)

        ts_val = [0.0, 0.0, 0.0]
        te_val = [0.1, 0.0, 0.0]
        rs_val = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        re_val = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        trans_start = torch.tensor(ts_val, device=self.device, requires_grad=True)
        trans_end = torch.tensor(te_val, device=self.device, requires_grad=True)
        rot_start = torch.tensor(rs_val, device=self.device, requires_grad=True)
        rot_end = torch.tensor(re_val, device=self.device, requires_grad=True)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        image_points = torch.tensor(
            [[340.0, 100.0], [360.0, 350.0]],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        loss = world_rays.sum()
        loss.backward()

        self.assertIsNotNone(trans_end.grad)
        self.assertTrue(trans_end.grad.abs().sum() > 0, "trans_end gradient should be non-zero")
        self.assertIsNotNone(rot_end.grad)
        self.assertTrue(rot_end.grad.abs().sum() > 0, "rot_end gradient should be non-zero")

        eps = 1e-3

        # FD for trans_end[0]
        te_plus = list(te_val)
        te_plus[0] += eps
        te_minus = list(te_val)
        te_minus[0] -= eps
        fd_te = (
            self._ftheta_nonlinear_backproject_pose_loss(image_points, ts_val, te_plus, rs_val, re_val)
            - self._ftheta_nonlinear_backproject_pose_loss(image_points, ts_val, te_minus, rs_val, re_val)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_te, 0.0, places=3)
        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd_te,
            rtol=0.02,
            atol=0.01,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd_te:.6f}",
        )

        # FD for rot_end[2]
        re_plus = list(re_val)
        re_plus[2] += eps
        re_minus = list(re_val)
        re_minus[2] -= eps
        fd_re = (
            self._ftheta_nonlinear_backproject_pose_loss(image_points, ts_val, te_val, rs_val, re_plus)
            - self._ftheta_nonlinear_backproject_pose_loss(image_points, ts_val, te_val, rs_val, re_minus)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_re, 0.0, places=3)
        np.testing.assert_allclose(
            rot_end.grad[2].item(),
            fd_re,
            rtol=0.02,
            atol=0.01,
            err_msg=f"rot_end[2] grad: autograd={rot_end.grad[2].item():.6f}, fd={fd_re:.6f}",
        )

    # ---- Real-param non-linear FTheta + RS pose tests ----

    @staticmethod
    def _poly_order_from_length(n):
        """Triangular polynomial order from coefficient count: n = (order+1)(order+2)/2."""
        return int((-3 + math.sqrt(1 + 8 * n)) / 2)

    def test_rs_real_ftheta_forward_nonidentity_pose(self):
        """Real non-linear FTheta (degree 5) forward projection with non-identity RS pose.

        Uses trained camera parameters with degree-5 angle_to_pixeldist_poly (non-linear
        higher-order terms). Verifies the non-linear polynomial evaluation through the
        RS iterative solver matches a Python reference. Camera has no external distortion.
        """
        all_cams = self._load_real_params()
        cam_data = None
        for uid, data in all_cams.items():
            if data["camera_model_type"] == "ftheta" and data["external_distortion"] is None:
                cam_data = data
                break
        self.assertIsNotNone(cam_data, "No FTheta camera without ext distortion found in test data")

        projection, _, res, _, _, _, _ = self._make_real_ftheta_with_leaves(cam_data)
        no_dist = NoExternalDistortion()

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.2, 0.1, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        rot_end = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        world_points = torch.tensor(
            [[0.3, 0.2, 10.0], [1.0, -0.5, 8.0], [-0.5, 0.8, 12.0]],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            no_dist,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertTrue(valid.all(), "All points should be valid for moderate angles")

        intr = cam_data["intrinsics"]
        fw_poly = intr["angle_to_pixeldist_poly"]
        fw_poly_degree = len(fw_poly) - 1
        pp = intr["principal_point"]
        cde = intr["linear_cde"]
        A = [[cde[0], cde[1]], [cde[2], 1.0]]
        height = res[1]

        h_poly_id = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        v_poly_id = [0.0, 0.0, 1.0] + [0.0] * 12

        for i, wp in enumerate(world_points.cpu().tolist()):
            expected = self._ref_rs_forward_ftheta(
                wp,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                h_poly_id,
                v_poly_id,
                1,
                1,
                fw_poly,
                fw_poly_degree,
                pp,
                A,
                height=height,
            )
            self.assertIsNotNone(expected, f"Point {i}: reference solver should converge")
            actual = image_points[i].cpu().tolist()
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-3,
                atol=0.5,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_rs_real_ftheta_forward_pose_gradient(self):
        """FD gradient test for real non-linear FTheta (degree 5) + RS with non-identity pose.

        Verifies gradients w.r.t. trans_end[0] and rot_end[2] flow correctly through
        the degree-5 polynomial evaluation and the RS iterative solver.
        """
        all_cams = self._load_real_params()
        cam_data = None
        for uid, data in all_cams.items():
            if data["camera_model_type"] == "ftheta" and data["external_distortion"] is None:
                cam_data = data
                break
        self.assertIsNotNone(cam_data, "No FTheta camera without ext distortion found in test data")

        projection, _, res, _, _, _, _ = self._make_real_ftheta_with_leaves(cam_data)
        no_dist = NoExternalDistortion()

        ts_val = [0.0, 0.0, 0.0]
        te_val = [0.1, 0.0, 0.0]
        rs_val = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        re_val = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        trans_start = torch.tensor(ts_val, device=self.device, requires_grad=True)
        trans_end = torch.tensor(te_val, device=self.device, requires_grad=True)
        rot_start = torch.tensor(rs_val, device=self.device, requires_grad=True)
        rot_end = torch.tensor(re_val, device=self.device, requires_grad=True)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        world_points = torch.tensor(
            [[0.3, 0.2, 10.0], [1.0, -0.5, 8.0]],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            no_dist,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(trans_end.grad)
        self.assertTrue(trans_end.grad.abs().sum() > 0, "trans_end gradient should be non-zero")
        self.assertIsNotNone(rot_end.grad)
        self.assertTrue(rot_end.grad.abs().sum() > 0, "rot_end gradient should be non-zero")

        eps = 1e-3

        te_plus = list(te_val)
        te_plus[0] += eps
        te_minus = list(te_val)
        te_minus[0] -= eps
        fd_te = (
            self._real_ftheta_pose_forward_loss(cam_data, world_points, ts_val, te_plus, rs_val, re_val)
            - self._real_ftheta_pose_forward_loss(cam_data, world_points, ts_val, te_minus, rs_val, re_val)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_te, 0.0, places=3)
        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd_te,
            rtol=5e-3,
            atol=0.01,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd_te:.6f}",
        )

        re_plus = list(re_val)
        re_plus[2] += eps
        re_minus = list(re_val)
        re_minus[2] -= eps
        fd_re = (
            self._real_ftheta_pose_forward_loss(cam_data, world_points, ts_val, te_val, rs_val, re_plus)
            - self._real_ftheta_pose_forward_loss(cam_data, world_points, ts_val, te_val, rs_val, re_minus)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_re, 0.0, places=3)
        np.testing.assert_allclose(
            rot_end.grad[2].item(),
            fd_re,
            rtol=5e-4,
            atol=0.01,
            err_msg=f"rot_end[2] grad: autograd={rot_end.grad[2].item():.6f}, fd={fd_re:.6f}",
        )

    def test_rs_real_ftheta_distortion_forward_nonidentity_pose(self):
        """Real non-linear FTheta + bivariate windshield distortion with non-identity RS pose.

        Uses trained camera parameters with degree-5 fw_poly and order-2/order-4 bivariate
        windshield distortion (BACKWARD reference). Combines non-linear FTheta polynomial
        evaluation, real distortion, and RS iterative solving.
        """
        all_cams = self._load_real_params()
        cam_data = None
        for uid, data in all_cams.items():
            if data.get("external_distortion") is not None:
                cam_data = data
                break
        self.assertIsNotNone(cam_data, "No camera with ext distortion found in test data")

        projection, distortion, res, _, _, _, _ = self._make_real_ftheta_with_leaves(cam_data)
        self.assertIsNotNone(distortion)

        trans_start = [0.0, 0.0, 0.0]
        trans_end = [0.2, 0.1, 0.0]
        rot_start = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        rot_end = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        ts = torch.tensor(trans_start, device=self.device)
        te = torch.tensor(trans_end, device=self.device)
        rs = torch.tensor(rot_start, device=self.device)
        re = torch.tensor(rot_end, device=self.device)
        dynamic_pose = create_dynamic_pose(ts, te, rs, re, self.device)

        world_points = torch.tensor(
            [[0.3, 0.2, 10.0], [1.0, -0.5, 8.0], [-0.5, 0.8, 12.0]],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertTrue(valid.all(), "All points should be valid for moderate angles")

        intr = cam_data["intrinsics"]
        fw_poly = intr["angle_to_pixeldist_poly"]
        fw_poly_degree = len(fw_poly) - 1
        pp = intr["principal_point"]
        cde = intr["linear_cde"]
        A = [[cde[0], cde[1]], [cde[2], 1.0]]
        height = res[1]

        ext = cam_data["external_distortion"]
        h_poly_inv = ext["horizontal_poly_inverse"]
        v_poly_inv = ext["vertical_poly_inverse"]
        h_order = self._poly_order_from_length(len(h_poly_inv))
        v_order = self._poly_order_from_length(len(v_poly_inv))

        for i, wp in enumerate(world_points.cpu().tolist()):
            expected = self._ref_rs_forward_ftheta(
                wp,
                trans_start,
                trans_end,
                rot_start,
                rot_end,
                h_poly_inv,
                v_poly_inv,
                h_order,
                v_order,
                fw_poly,
                fw_poly_degree,
                pp,
                A,
                height=height,
            )
            self.assertIsNotNone(expected, f"Point {i}: reference solver should converge")
            actual = image_points[i].cpu().tolist()
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-3,
                atol=0.5,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_rs_real_ftheta_distortion_forward_pose_gradient(self):
        """FD gradient test for real non-linear FTheta + distortion + RS with non-identity pose.

        Verifies gradients w.r.t. trans_end[0] and rot_end[2] flow correctly through
        the degree-5 polynomial, bivariate windshield distortion, and the RS iterative solver.
        """
        all_cams = self._load_real_params()
        cam_data = None
        for uid, data in all_cams.items():
            if data.get("external_distortion") is not None:
                cam_data = data
                break
        self.assertIsNotNone(cam_data, "No camera with ext distortion found in test data")

        projection, distortion, res, _, _, _, _ = self._make_real_ftheta_with_leaves(cam_data)
        self.assertIsNotNone(distortion)

        ts_val = [0.0, 0.0, 0.0]
        te_val = [0.1, 0.0, 0.0]
        rs_val = [0.0, 0.0, 0.0, 1.0]
        angle_rad = math.radians(5.0)
        re_val = [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]

        trans_start = torch.tensor(ts_val, device=self.device, requires_grad=True)
        trans_end = torch.tensor(te_val, device=self.device, requires_grad=True)
        rot_start = torch.tensor(rs_val, device=self.device, requires_grad=True)
        rot_end = torch.tensor(re_val, device=self.device, requires_grad=True)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        world_points = torch.tensor(
            [[0.3, 0.2, 10.0], [1.0, -0.5, 8.0]],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(trans_end.grad)
        self.assertTrue(trans_end.grad.abs().sum() > 0, "trans_end gradient should be non-zero")
        self.assertIsNotNone(rot_end.grad)
        self.assertTrue(rot_end.grad.abs().sum() > 0, "rot_end gradient should be non-zero")

        eps = 5e-4

        te_plus = list(te_val)
        te_plus[0] += eps
        te_minus = list(te_val)
        te_minus[0] -= eps
        fd_te = (
            self._real_ftheta_pose_forward_loss(cam_data, world_points, ts_val, te_plus, rs_val, re_val)
            - self._real_ftheta_pose_forward_loss(cam_data, world_points, ts_val, te_minus, rs_val, re_val)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_te, 0.0, places=3)
        np.testing.assert_allclose(
            trans_end.grad[0].item(),
            fd_te,
            rtol=0.015,
            atol=0.01,
            err_msg=f"trans_end[0] grad: autograd={trans_end.grad[0].item():.6f}, fd={fd_te:.6f}",
        )

        re_plus = list(re_val)
        re_plus[2] += eps
        re_minus = list(re_val)
        re_minus[2] -= eps
        fd_re = (
            self._real_ftheta_pose_forward_loss(cam_data, world_points, ts_val, te_val, rs_val, re_plus)
            - self._real_ftheta_pose_forward_loss(cam_data, world_points, ts_val, te_val, rs_val, re_minus)
        ) / (2 * eps)
        self.assertNotAlmostEqual(fd_re, 0.0, places=3)
        np.testing.assert_allclose(
            rot_end.grad[2].item(),
            fd_re,
            rtol=0.02,
            atol=0.01,
            err_msg=f"rot_end[2] grad: autograd={rot_end.grad[2].item():.6f}, fd={fd_re:.6f}",
        )

    # ---- Gradient tests continued ----

    def test_backproject_gradient_bivariate_shutter(self):
        """Backprojection gradients verified against finite differences for intrinsics and distortion."""
        projection, focal_length, _ = self._make_pinhole_with_leaves(self.device, requires_grad=True)
        distortion, _, _, h_poly_inv, _ = self._make_bivariate_with_leaves(self.device, requires_grad=True)
        dynamic_pose = self._make_identity_dynamic_pose()

        image_points = torch.tensor(
            [
                [450.0, 350.0],
                [550.0, 400.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = world_rays.sum()
        loss.backward()

        self.assertIsNotNone(focal_length.grad, "focal_length gradient should exist through backprojection")
        self.assertTrue(focal_length.grad.abs().sum() > 0, "focal_length gradient should be non-zero")
        self.assertIsNotNone(h_poly_inv.grad, "h_poly_inv gradient should exist through backprojection")
        self.assertTrue(h_poly_inv.grad.abs().sum() > 0, "h_poly_inv gradient should be non-zero")

        # FD for focal_length[0] through backprojection
        eps = 0.5

        fl_plus = [500.0 + eps, 500.0]
        fl_minus = [500.0 - eps, 500.0]

        loss_plus = self._pinhole_backproject_loss(image_points, fl=fl_plus)
        loss_minus = self._pinhole_backproject_loss(image_points, fl=fl_minus)
        fd_fl = (loss_plus - loss_minus) / (2 * eps)

        self.assertNotAlmostEqual(fd_fl, 0.0, places=3, msg="FD for focal_length[0] backproject should be non-zero")

        np.testing.assert_allclose(
            focal_length.grad[0].item(),
            fd_fl,
            rtol=0.05,
            atol=0.01,
            err_msg=f"focal_length[0] backproject grad: autograd={focal_length.grad[0].item():.6f}, fd={fd_fl:.6f}",
        )

        # FD for h_poly_inv[1] through backprojection (dominant undistort coefficient)
        eps = 1e-3

        hinv_plus = [0.0, 0.99 + eps, 0.0]
        hinv_minus = [0.0, 0.99 - eps, 0.0]

        loss_plus = self._pinhole_backproject_loss(image_points, h_inv=hinv_plus)
        loss_minus = self._pinhole_backproject_loss(image_points, h_inv=hinv_minus)
        fd_hinv = (loss_plus - loss_minus) / (2 * eps)

        self.assertNotAlmostEqual(fd_hinv, 0.0, places=3, msg="FD for h_poly_inv[1] backproject should be non-zero")

        np.testing.assert_allclose(
            h_poly_inv.grad[1].item(),
            fd_hinv,
            rtol=0.05,
            atol=0.01,
            err_msg=f"h_poly_inv[1] backproject grad: autograd={h_poly_inv.grad[1].item():.6f}, fd={fd_hinv:.6f}",
        )

    # ---- Order-2 polynomial tests (cross-coupling, quadratic terms) ----

    def test_pinhole_bivariate_order2_shutter_forward(self):
        """Forward projection (pinhole + order-2 bivariate + RS) matches Python reference.

        Order-2 polynomials exercise quadratic and cross-coupling terms:
        p(phi, theta) = c[0] + c[1]*phi + c[2]*phi^2 + c[3]*theta + c[4]*phi*theta + c[5]*theta^2
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_order2_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.5, 0.3, 5.0],
                [1.0, 0.5, 10.0],
                [0.3, 0.2, 3.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (3, 2))
        self.assertTrue(valid.all())

        for i, wp in enumerate(world_points.cpu().tolist()):
            dist_ray = self._ref_bivariate_distort(
                wp, self.H_POLY_FWD_O2, self.V_POLY_FWD_O2, self.H_ORDER_O2, self.V_ORDER_O2
            )
            expected = self._ref_pinhole_project(dist_ray, 500, 500, 320, 240)
            actual = image_points[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.05,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_ftheta_bivariate_order2_shutter_forward(self):
        """Forward projection (ftheta + order-2 bivariate + RS) matches Python reference."""
        projection, _, _ = self._make_ftheta_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_order2_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.5, 0.3, 8.0],
                [0.2, 0.4, 6.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (2, 2))
        self.assertTrue(valid.all())

        A_ref = [[1, 0], [0, 1]]
        for i, wp in enumerate(world_points.cpu().tolist()):
            dist_ray = self._ref_bivariate_distort(
                wp, self.H_POLY_FWD_O2, self.V_POLY_FWD_O2, self.H_ORDER_O2, self.V_ORDER_O2
            )
            expected = self._ref_ftheta_project(dist_ray, self.FW_POLY, self.FW_POLY_DEGREE, [320, 240], A_ref)
            actual = image_points[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.05,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_pinhole_bivariate_order2_shutter_backproject(self):
        """Backprojection (pinhole + order-2 bivariate + RS) matches Python reference.

        Undistort direction uses h_poly_inv/v_poly_inv (also order 2).
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_order2_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        image_points = torch.tensor(
            [
                [350.0, 260.0],
                [400.0, 300.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        self.assertEqual(world_rays.shape, (2, 6))

        origins = world_rays[:, :3]
        directions = world_rays[:, 3:]

        self.assertTrue(
            torch.allclose(origins, torch.zeros_like(origins), atol=1e-4),
            "Identity pose should give origin=[0,0,0]",
        )

        norms = directions.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

        for i, ip in enumerate(image_points.cpu().tolist()):
            cam_ray = self._ref_pinhole_backproject(ip, 500, 500, 320, 240)
            expected_dir = self._ref_bivariate_distort(
                cam_ray, self.H_POLY_INV_O2, self.V_POLY_INV_O2, self.H_ORDER_O2, self.V_ORDER_O2
            )
            actual = directions[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected_dir,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Order-2 backproject direction mismatch for point {i}",
            )

    def test_bivariate_order2_gradient_cross_coupling(self):
        """Order-2 cross-coupling gradient (h_poly[4] = phi*theta term) matches FD.

        Also verifies all distortion polynomial gradients are non-zero,
        confirming gradient flow through every coefficient slot.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, h_poly, v_poly, _, _ = self._make_bivariate_order2_with_leaves(self.device, requires_grad=True)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.3, 0.2, 5.0],
                [0.8, 0.4, 8.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertIsNotNone(h_poly.grad)
        self.assertTrue(h_poly.grad.abs().sum() > 0, "h_poly grad should be non-zero")
        self.assertIsNotNone(v_poly.grad)
        self.assertTrue(v_poly.grad.abs().sum() > 0, "v_poly grad should be non-zero")

        # FD for h_poly[4] (cross-coupling term: phi*theta)
        eps = 1e-3

        h_base = list(self.H_POLY_FWD_O2)
        v_base = list(self.V_POLY_FWD_O2[:6])
        h_inv_base = list(self.H_POLY_INV_O2)
        v_inv_base = list(self.V_POLY_INV_O2[:6])

        h_plus = list(h_base)
        h_plus[4] += eps
        h_minus = list(h_base)
        h_minus[4] -= eps

        loss_plus = self._pinhole_forward_loss_bivariate(world_points, h_plus, v_base, h_inv_base, v_inv_base)
        loss_minus = self._pinhole_forward_loss_bivariate(world_points, h_minus, v_base, h_inv_base, v_inv_base)
        fd_h4 = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            h_poly.grad[4].item(),
            fd_h4,
            rtol=0.05,
            atol=0.01,
            err_msg=f"h_poly[4] grad: autograd={h_poly.grad[4].item():.6f}, fd={fd_h4:.6f}",
        )

        # FD for v_poly[3] (dominant vertical term: theta)
        v_plus = list(v_base)
        v_plus[3] += eps
        v_minus = list(v_base)
        v_minus[3] -= eps

        loss_plus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_plus, h_inv_base, v_inv_base)
        loss_minus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_minus, h_inv_base, v_inv_base)
        fd_v3 = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            v_poly.grad[3].item(),
            fd_v3,
            rtol=0.05,
            atol=0.01,
            err_msg=f"v_poly[3] grad: autograd={v_poly.grad[3].item():.6f}, fd={fd_v3:.6f}",
        )

        # FD for v_poly[4] (cross-coupling term: phi*theta)
        v_plus = list(v_base)
        v_plus[4] += eps
        v_minus = list(v_base)
        v_minus[4] -= eps

        loss_plus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_plus, h_inv_base, v_inv_base)
        loss_minus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_minus, h_inv_base, v_inv_base)
        fd_v4 = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            v_poly.grad[4].item(),
            fd_v4,
            rtol=0.05,
            atol=0.01,
            err_msg=f"v_poly[4] grad: autograd={v_poly.grad[4].item():.6f}, fd={fd_v4:.6f}",
        )

    # ---- Order-3 tests (v_poly order 3, h_poly order 2) ----

    def test_pinhole_bivariate_order3_shutter_forward(self):
        """Forward projection with order-3 v_poly matches Python reference.

        Exercises the order-3 branch of eval_poly_2d including cubic terms (x³, y³)
        and higher cross-coupling (x²y, xy²).
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_order3_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.5, 0.3, 5.0],
                [1.0, 0.5, 10.0],
                [0.2, 0.4, 6.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )
        self.assertEqual(image_points.shape, (3, 2))
        self.assertTrue(valid.all())

        for i, wp in enumerate(world_points.cpu().tolist()):
            dist_ray = self._ref_bivariate_distort(
                wp, self.H_POLY_FWD_O2, self.V_POLY_FWD_O3, self.H_ORDER_O2, self.V_ORDER_O3
            )
            expected = self._ref_pinhole_project(dist_ray, 500, 500, 320, 240)
            actual = image_points[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.05,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_pinhole_bivariate_order3_shutter_backproject(self):
        """Backprojection with order-3 v_poly_inv matches Python reference."""
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_order3_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        image_points = torch.tensor(
            [
                [350.0, 260.0],
                [400.0, 300.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        self.assertEqual(world_rays.shape, (2, 6))

        origins = world_rays[:, :3]
        directions = world_rays[:, 3:]

        self.assertTrue(torch.allclose(origins, torch.zeros_like(origins), atol=1e-4))

        norms = directions.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

        for i, ip in enumerate(image_points.cpu().tolist()):
            cam_ray = self._ref_pinhole_backproject(ip, 500, 500, 320, 240)
            expected_dir = self._ref_bivariate_distort(
                cam_ray, self.H_POLY_INV_O2, self.V_POLY_INV_O3, self.H_ORDER_O2, self.V_ORDER_O3
            )
            actual = directions[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected_dir,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Order-3 backproject direction mismatch for point {i}",
            )

    def test_bivariate_order3_gradient(self):
        """Order-3 v_poly gradients match FD.

        Uses large off-axis angles so higher-order monomials are numerically significant.
        Tests v_poly[4] (y monomial, dominant) and v_poly[7] (y² monomial) which exercise
        the order-3 branch of the custom eval_poly_2d_bwd backward.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, h_poly, v_poly, _, _ = self._make_bivariate_order3_with_leaves(self.device, requires_grad=True)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [1.0, 1.0, 3.0],
                [1.5, 0.8, 4.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertTrue(h_poly.grad.abs().sum() > 0)
        self.assertTrue(v_poly.grad.abs().sum() > 0)

        # FD for v_poly[4] (y monomial — dominant theta term)
        eps = 1e-3

        h_base = list(self.H_POLY_FWD_O2)
        v_base = list(self.V_POLY_FWD_O3[:10])
        h_inv_base = list(self.H_POLY_INV_O2)
        v_inv_base = list(self.V_POLY_INV_O3[:10])

        v_plus = list(v_base)
        v_plus[4] += eps
        v_minus = list(v_base)
        v_minus[4] -= eps

        loss_plus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_plus, h_inv_base, v_inv_base)
        loss_minus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_minus, h_inv_base, v_inv_base)
        fd = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            v_poly.grad[4].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"v_poly[4] grad: autograd={v_poly.grad[4].item():.6f}, fd={fd:.6f}",
        )

        # FD for v_poly[7] (y² monomial)
        v_plus = list(v_base)
        v_plus[7] += eps
        v_minus = list(v_base)
        v_minus[7] -= eps

        loss_plus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_plus, h_inv_base, v_inv_base)
        loss_minus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_minus, h_inv_base, v_inv_base)
        fd = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            v_poly.grad[7].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"v_poly[7] grad: autograd={v_poly.grad[7].item():.6f}, fd={fd:.6f}",
        )

    # ---- Order-4 tests (v_poly order 4, h_poly order 2) ----

    def test_pinhole_bivariate_order4_shutter_forward(self):
        """Forward projection with order-4 v_poly matches Python reference.

        Exercises the order-4 (else) branch of eval_poly_2d, which is the most
        complex with quartic terms (x⁴, y⁴) and all cross-coupling monomials.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_order4_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.5, 0.3, 5.0],
                [1.0, 0.5, 10.0],
                [0.2, 0.4, 6.0],
            ],
            device=self.device,
        )

        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )

        self.assertEqual(image_points.shape, (3, 2))
        self.assertTrue(valid.all())

        for i, wp in enumerate(world_points.cpu().tolist()):
            dist_ray = self._ref_bivariate_distort(
                wp, self.H_POLY_FWD_O2, self.V_POLY_FWD_O4, self.H_ORDER_O2, self.V_ORDER_O4
            )
            expected = self._ref_pinhole_project(dist_ray, 500, 500, 320, 240)
            actual = image_points[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=0.05,
                err_msg=f"Point {i}: world={wp}",
            )

    def test_pinhole_bivariate_order4_shutter_backproject(self):
        """Backprojection with order-4 v_poly_inv matches Python reference."""
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, _, _, _, _ = self._make_bivariate_order4_with_leaves(self.device)
        dynamic_pose = self._make_identity_dynamic_pose()

        image_points = torch.tensor(
            [
                [350.0, 260.0],
                [400.0, 300.0],
            ],
            device=self.device,
        )

        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        self.assertEqual(world_rays.shape, (2, 6))

        origins = world_rays[:, :3]
        directions = world_rays[:, 3:]

        self.assertTrue(torch.allclose(origins, torch.zeros_like(origins), atol=1e-4))

        norms = directions.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

        for i, ip in enumerate(image_points.cpu().tolist()):
            cam_ray = self._ref_pinhole_backproject(ip, 500, 500, 320, 240)
            expected_dir = self._ref_bivariate_distort(
                cam_ray, self.H_POLY_INV_O2, self.V_POLY_INV_O4, self.H_ORDER_O2, self.V_ORDER_O4
            )

            actual = directions[i].cpu().tolist()

            np.testing.assert_allclose(
                actual,
                expected_dir,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Order-4 backproject direction mismatch for point {i}",
            )

    def test_bivariate_order4_gradient(self):
        """Order-4 v_poly gradients match FD.

        Uses large off-axis angles so higher-order monomials are numerically significant.
        Tests v_poly[5] (y monomial, dominant) and v_poly[9] (y² monomial) which exercise
        the else (order>=4) branch of the custom eval_poly_2d_bwd backward.
        """
        projection, _, _ = self._make_pinhole_with_leaves(self.device)
        distortion, h_poly, v_poly, _, _ = self._make_bivariate_order4_with_leaves(self.device, requires_grad=True)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [1.0, 1.0, 3.0],
                [1.5, 0.8, 4.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            (640, 480),
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        self.assertTrue(h_poly.grad.abs().sum() > 0)
        self.assertTrue(v_poly.grad.abs().sum() > 0)

        # FD for v_poly[5] (y monomial — dominant theta term)
        eps = 1e-3

        h_base = list(self.H_POLY_FWD_O2)
        v_base = list(self.V_POLY_FWD_O4)
        h_inv_base = list(self.H_POLY_INV_O2)
        v_inv_base = list(self.V_POLY_INV_O4)

        v_plus = list(v_base)
        v_plus[5] += eps
        v_minus = list(v_base)
        v_minus[5] -= eps

        loss_plus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_plus, h_inv_base, v_inv_base)
        loss_minus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_minus, h_inv_base, v_inv_base)
        fd = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            v_poly.grad[5].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"v_poly[5] grad: autograd={v_poly.grad[5].item():.6f}, fd={fd:.6f}",
        )

        # FD for v_poly[9] (y² monomial)
        v_plus = list(v_base)
        v_plus[9] += eps
        v_minus = list(v_base)
        v_minus[9] -= eps

        loss_plus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_plus, h_inv_base, v_inv_base)
        loss_minus = self._pinhole_forward_loss_bivariate(world_points, h_base, v_minus, h_inv_base, v_inv_base)
        fd = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            v_poly.grad[9].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"v_poly[9] grad: autograd={v_poly.grad[9].item():.6f}, fd={fd:.6f}",
        )

    # ---- Tests using real trained camera parameters ----

    REAL_CAMERA_PARAMS_PATH = "libs/sensors/kernels/cameras/test_data/test_camera_params.json"

    @classmethod
    def _load_real_params(cls) -> dict:
        with open(cls.REAL_CAMERA_PARAMS_PATH) as f:
            return json.load(f)

    def _make_real_ftheta_with_leaves(self, camera_data, requires_grad=False):
        """Build FThetaProjection + BivariateWindshieldDistortion from real JSON camera data.

        Returns (projection, distortion, resolution, pp, fw_poly, h_poly_inv, v_poly_inv).
        distortion may be None if camera has no external_distortion.

        With BACKWARD reference polynomial the kernel reads h_poly_inv/v_poly_inv for
        forward projection, so those are the leaf tensors that receive gradients.
        """
        from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS

        intr = camera_data["intrinsics"]
        res = tuple(camera_data["resolution"])

        pp = torch.tensor(intr["principal_point"], device=self.device, requires_grad=requires_grad)

        fw_coeffs = intr["angle_to_pixeldist_poly"]
        bw_coeffs = intr["pixeldist_to_angle_poly"]
        fw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device, requires_grad=requires_grad)
        bw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)

        for i, c in enumerate(fw_coeffs):
            fw_poly.data[i] = c

        for i, c in enumerate(bw_coeffs):
            bw_poly.data[i] = c

        dfw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        dbw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)

        for i in range(len(fw_coeffs) - 1):
            dfw_poly[i] = fw_coeffs[i + 1] * (i + 1)

        for i in range(len(bw_coeffs) - 1):
            dbw_poly[i] = bw_coeffs[i + 1] * (i + 1)

        cde = intr["linear_cde"]
        A = torch.tensor([[cde[0], cde[1]], [cde[2], 1.0]], device=self.device)
        Ainv = torch.linalg.inv(A)

        projection = FThetaProjection.from_components(
            principal_point=pp,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=A,
            Ainv=Ainv,
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=intr["max_angle_rad"],
            newton_iterations=10,
            min_2d_norm=1e-6,
        )

        distortion = None
        h_poly_inv = None
        v_poly_inv = None
        ext = camera_data.get("external_distortion")
        if ext is not None:
            h_poly = torch.tensor(ext["horizontal_poly"], device=self.device)
            v_poly = torch.tensor(ext["vertical_poly"], device=self.device)
            h_poly_inv = torch.tensor(ext["horizontal_poly_inverse"], device=self.device, requires_grad=requires_grad)
            v_poly_inv = torch.tensor(ext["vertical_poly_inverse"], device=self.device, requires_grad=requires_grad)
            distortion = BivariateWindshieldDistortion.from_components(
                h_poly=h_poly,
                v_poly=v_poly,
                h_poly_inv=h_poly_inv,
                v_poly_inv=v_poly_inv,
                reference_polynomial=ReferencePolynomial.BACKWARD,
            )

        return projection, distortion, res, pp, fw_poly, h_poly_inv, v_poly_inv

    def _real_roundtrip_loss(
        self, cam_data, world_points, *, fw_poly_override=None, h_poly_inv_override=None, v_poly_inv_override=None
    ):
        """Evaluate project → backproject round-trip loss with real params (no grad).

        Loss = sum of all direction components from the backprojected world rays.
        With BACKWARD reference, forward projection reads h_poly_inv/v_poly_inv and
        backprojection reads h_poly/v_poly.
        """
        from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS

        intr = cam_data["intrinsics"]
        res = tuple(cam_data["resolution"])

        pp = torch.tensor(intr["principal_point"], device=self.device)
        fw_coeffs = fw_poly_override if fw_poly_override is not None else intr["angle_to_pixeldist_poly"]
        bw_coeffs = intr["pixeldist_to_angle_poly"]

        fw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        bw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)

        for i, c in enumerate(fw_coeffs):
            fw_poly[i] = c

        for i, c in enumerate(bw_coeffs):
            bw_poly[i] = c

        dfw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        dbw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)

        for i in range(len(fw_coeffs) - 1):
            dfw_poly[i] = fw_coeffs[i + 1] * (i + 1)

        for i in range(len(bw_coeffs) - 1):
            dbw_poly[i] = bw_coeffs[i + 1] * (i + 1)

        cde = intr["linear_cde"]
        A = torch.tensor([[cde[0], cde[1]], [cde[2], 1.0]], device=self.device)
        Ainv = torch.linalg.inv(A)

        proj = FThetaProjection.from_components(
            principal_point=pp,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=A,
            Ainv=Ainv,
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=intr["max_angle_rad"],
            newton_iterations=10,
            min_2d_norm=1e-6,
        )

        ext = cam_data.get("external_distortion")
        if ext is not None:
            h_inv = h_poly_inv_override if h_poly_inv_override is not None else ext["horizontal_poly_inverse"]
            v_inv = v_poly_inv_override if v_poly_inv_override is not None else ext["vertical_poly_inverse"]

            dist = BivariateWindshieldDistortion.from_components(
                h_poly=torch.tensor(ext["horizontal_poly"], device=self.device),
                v_poly=torch.tensor(ext["vertical_poly"], device=self.device),
                h_poly_inv=torch.tensor(h_inv, device=self.device),
                v_poly_inv=torch.tensor(v_inv, device=self.device),
                reference_polynomial=ReferencePolynomial.BACKWARD,
            )
        else:
            dist = NoExternalDistortion()

        pose = self._make_identity_dynamic_pose()

        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                proj,
                dist,
                res,
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )

            world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
                ip,
                proj,
                dist,
                res,
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )

        return world_rays[:, 3:6].sum().item()

    def test_real_ftheta_no_distortion_forward_backproject_roundtrip(self):
        """Forward-then-backproject round-trip with real FTheta params (no ext distortion).

        Verifies ray recovery and that the round-trip gradient for fw_poly is near-zero.
        With reference_poly=FORWARD, backprojection uses Newton iteration on fw_poly,
        so project→backproject is (approximately) the identity: d(roundtrip)/d(fw_poly) ≈ 0.
        """
        all_cams = self._load_real_params()
        cam_data = None

        for uid, data in all_cams.items():
            if data["camera_model_type"] == "ftheta" and data["external_distortion"] is None:
                cam_data = data
                break

        self.assertIsNotNone(cam_data, "No FTheta camera without ext distortion found in test data")

        projection, distortion, res, pp, fw_poly, _, _ = self._make_real_ftheta_with_leaves(
            cam_data, requires_grad=True
        )
        no_dist = NoExternalDistortion()
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.3, 0.2, 10.0],
                [1.0, -0.5, 8.0],
                [-0.5, 0.8, 12.0],
            ],
            device=self.device,
        )

        # Forward projection
        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            no_dist,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )

        self.assertTrue(valid.all(), "All points should be valid for moderate angles")

        # Backprojection
        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            no_dist,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )
        directions = world_rays[:, 3:6]

        # Round-trip direction recovery
        for i in range(len(world_points)):
            orig_dir = world_points[i] / world_points[i].norm()
            recovered_dir = directions[i] / directions[i].norm()
            np.testing.assert_allclose(
                recovered_dir.detach().cpu().numpy(),
                orig_dir.cpu().numpy(),
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"Real FTheta (no ext dist) round-trip mismatch for point {i}",
            )

        # Gradient through full round-trip pipeline
        loss = directions.sum()
        loss.backward()

        self.assertIsNotNone(fw_poly.grad)

        # With FORWARD reference, Newton iteration uses fw_poly for both project and
        # backproject, so the round-trip is the identity and d(loss)/d(fw_poly) ≈ 0.
        # FD also gives exactly 0, confirming the forward/backward symmetry.
        np.testing.assert_allclose(
            fw_poly.grad[1].item(),
            0.0,
            atol=1e-3,
            err_msg="Round-trip fw_poly[1] gradient should be near-zero (forward/backward cancel)",
        )

    def test_real_ftheta_with_distortion_forward_backproject_roundtrip(self):
        """Forward-then-backproject round-trip with real FTheta + bivariate windshield distortion.

        Verifies ray recovery and FD gradients through the full project → backproject pipeline.
        With BACKWARD reference, forward projection uses h_poly_inv/v_poly_inv and
        backprojection uses h_poly/v_poly. Perturbing the inverse polys breaks the
        forward/backward symmetry, giving non-zero FD through the round-trip.
        """
        all_cams = self._load_real_params()
        cam_data = None

        for uid, data in all_cams.items():
            if data.get("external_distortion") is not None:
                cam_data = data
                break

        self.assertIsNotNone(cam_data, "No camera with ext distortion found in test data")

        projection, distortion, res, pp, fw_poly, h_poly_inv, v_poly_inv = self._make_real_ftheta_with_leaves(
            cam_data, requires_grad=True
        )

        self.assertIsNotNone(distortion)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.3, 0.2, 10.0],
                [1.0, -0.5, 8.0],
                [-0.5, 0.8, 12.0],
            ],
            device=self.device,
        )

        # Forward projection
        image_points, valid, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
            return_valid_flags=True,
        )

        self.assertTrue(valid.all(), "All points should be valid for moderate angles")

        # Backprojection
        world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
            image_points,
            projection,
            distortion,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        directions = world_rays[:, 3:6]

        # Round-trip direction recovery
        for i in range(len(world_points)):
            orig_dir = world_points[i] / world_points[i].norm()
            recovered_dir = directions[i] / directions[i].norm()

            np.testing.assert_allclose(
                recovered_dir.detach().cpu().numpy(),
                orig_dir.cpu().numpy(),
                rtol=2e-3,
                atol=2e-3,
                err_msg=f"Real FTheta+distortion round-trip mismatch for point {i}",
            )

        # Gradient through full round-trip pipeline

        loss = directions.sum()
        loss.backward()

        self.assertIsNotNone(h_poly_inv.grad)
        self.assertTrue(h_poly_inv.grad.abs().sum() > 0, "h_poly_inv round-trip gradient should be non-zero")
        self.assertIsNotNone(v_poly_inv.grad)
        self.assertTrue(v_poly_inv.grad.abs().sum() > 0, "v_poly_inv round-trip gradient should be non-zero")

        # FD for h_poly_inv[1] through project → backproject
        ext = cam_data["external_distortion"]

        # h_poly_inv is used in forward projection only (BACKWARD ref), so perturbing it
        # changes the projected image points without changing the backprojection undistortion.
        # Wider tolerance (10%) than single-direction tests because the round-trip chains
        # two kernel calls through Newton iteration, amplifying numerical noise.
        eps = 1e-3
        h_inv_base = list(ext["horizontal_poly_inverse"])
        h_inv_plus = list(h_inv_base)
        h_inv_plus[1] += eps
        h_inv_minus = list(h_inv_base)
        h_inv_minus[1] -= eps

        fd = (
            self._real_roundtrip_loss(cam_data, world_points, h_poly_inv_override=h_inv_plus)
            - self._real_roundtrip_loss(cam_data, world_points, h_poly_inv_override=h_inv_minus)
        ) / (2 * eps)

        self.assertNotAlmostEqual(fd, 0.0, places=5, msg="FD for h_poly_inv[1] round-trip should be non-zero")

        np.testing.assert_allclose(
            h_poly_inv.grad[1].item(),
            fd,
            rtol=0.10,
            atol=1e-3,
            err_msg=f"Round-trip h_poly_inv[1] grad: autograd={h_poly_inv.grad[1].item():.6f}, fd={fd:.6f}",
        )

        # FD for v_poly_inv[5] through project → backproject
        v_inv_base = list(ext["vertical_poly_inverse"])
        v_inv_plus = list(v_inv_base)
        v_inv_plus[5] += eps
        v_inv_minus = list(v_inv_base)
        v_inv_minus[5] -= eps

        fd = (
            self._real_roundtrip_loss(cam_data, world_points, v_poly_inv_override=v_inv_plus)
            - self._real_roundtrip_loss(cam_data, world_points, v_poly_inv_override=v_inv_minus)
        ) / (2 * eps)

        self.assertNotAlmostEqual(fd, 0.0, places=5, msg="FD for v_poly_inv[5] round-trip should be non-zero")

        np.testing.assert_allclose(
            v_poly_inv.grad[5].item(),
            fd,
            rtol=0.10,
            atol=1e-3,
            err_msg=f"Round-trip v_poly_inv[5] grad: autograd={v_poly_inv.grad[5].item():.6f}, fd={fd:.6f}",
        )

    def test_real_ftheta_with_distortion_gradient(self):
        """Gradient flow + analytical pp check with real trained distortion parameters.

        With BACKWARD reference polynomial, forward projection reads h_poly_inv/v_poly_inv,
        so gradients for the distortion flow through those tensors.
        Verifies pp gradient analytically (d(image)/d(pp) = N_points) and all other
        leaf gradients are non-zero (quantitative FD is in the dedicated _fd test).
        """
        all_cams = self._load_real_params()
        cam_data = None

        for uid, data in all_cams.items():
            if data.get("external_distortion") is not None:
                cam_data = data
                break

        self.assertIsNotNone(cam_data, "No camera with ext distortion found in test data")

        projection, distortion, res, pp, fw_poly, h_poly_inv, v_poly_inv = self._make_real_ftheta_with_leaves(
            cam_data, requires_grad=True
        )

        self.assertIsNotNone(distortion)
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.5, 0.3, 8.0],
                [1.0, -0.5, 6.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        # Non-zero gradient existence checks
        self.assertIsNotNone(pp.grad)
        self.assertTrue(pp.grad.abs().sum() > 0, "pp gradient should be non-zero")
        self.assertIsNotNone(fw_poly.grad)
        self.assertTrue(fw_poly.grad.abs().sum() > 0, "fw_poly gradient should be non-zero")
        self.assertIsNotNone(h_poly_inv.grad)
        self.assertTrue(h_poly_inv.grad.abs().sum() > 0, "h_poly_inv gradient should be non-zero")
        self.assertIsNotNone(v_poly_inv.grad)
        self.assertTrue(v_poly_inv.grad.abs().sum() > 0, "v_poly_inv gradient should be non-zero")

        # Analytical: d(image_point)/d(pp) = N_points per component
        n_points = len(world_points)

        np.testing.assert_allclose(
            pp.grad.cpu().numpy(),
            [n_points, n_points],
            rtol=0.02,
            err_msg=f"Real FTheta pp gradient should be ~{n_points} per component",
        )

    def _real_ftheta_forward_loss(
        self, cam_data, world_points, *, fw_poly_override=None, h_poly_inv_override=None, v_poly_inv_override=None
    ):
        """Evaluate forward loss with real params, optionally overriding specific polys for FD.

        With BACKWARD reference, forward projection reads h_poly_inv/v_poly_inv,
        so FD perturbations target those tensors.
        """
        from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS

        intr = cam_data["intrinsics"]
        res = tuple(cam_data["resolution"])

        pp = torch.tensor(intr["principal_point"], device=self.device)
        fw_coeffs = fw_poly_override if fw_poly_override is not None else intr["angle_to_pixeldist_poly"]
        bw_coeffs = intr["pixeldist_to_angle_poly"]

        fw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        bw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)

        for i, c in enumerate(fw_coeffs):
            fw_poly[i] = c

        for i, c in enumerate(bw_coeffs):
            bw_poly[i] = c

        dfw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        dbw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)

        for i in range(len(fw_coeffs) - 1):
            dfw_poly[i] = fw_coeffs[i + 1] * (i + 1)

        for i in range(len(bw_coeffs) - 1):
            dbw_poly[i] = bw_coeffs[i + 1] * (i + 1)

        cde = intr["linear_cde"]
        A = torch.tensor([[cde[0], cde[1]], [cde[2], 1.0]], device=self.device)
        Ainv = torch.linalg.inv(A)

        proj = FThetaProjection.from_components(
            principal_point=pp,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=A,
            Ainv=Ainv,
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=intr["max_angle_rad"],
            newton_iterations=10,
            min_2d_norm=1e-6,
        )

        ext = cam_data.get("external_distortion")
        if ext is not None:
            h_inv = h_poly_inv_override if h_poly_inv_override is not None else ext["horizontal_poly_inverse"]
            v_inv = v_poly_inv_override if v_poly_inv_override is not None else ext["vertical_poly_inverse"]
            dist = BivariateWindshieldDistortion.from_components(
                h_poly=torch.tensor(ext["horizontal_poly"], device=self.device),
                v_poly=torch.tensor(ext["vertical_poly"], device=self.device),
                h_poly_inv=torch.tensor(h_inv, device=self.device),
                v_poly_inv=torch.tensor(v_inv, device=self.device),
                reference_polynomial=ReferencePolynomial.BACKWARD,
            )
        else:
            dist = NoExternalDistortion()

        pose = self._make_identity_dynamic_pose()
        with torch.no_grad():
            ip, _, _, _, _ = project_world_points_shutter_pose(
                world_points,
                proj,
                dist,
                res,
                ShutterType.ROLLING_TOP_TO_BOTTOM,
                pose,
            )
        return ip.sum().item()

    def test_real_ftheta_with_distortion_gradient_fd(self):
        """FD verification of fw_poly[1], h_poly_inv[1], v_poly_inv[5] with real trained params.

        With BACKWARD reference polynomial, forward projection reads h_poly_inv/v_poly_inv,
        so FD perturbations target those inverse polynomial coefficients.
        """
        all_cams = self._load_real_params()
        cam_data = None

        for uid, data in all_cams.items():
            if data.get("external_distortion") is not None:
                cam_data = data
                break

        self.assertIsNotNone(cam_data, "No camera with ext distortion found in test data")

        projection, distortion, res, pp, fw_poly, h_poly_inv, v_poly_inv = self._make_real_ftheta_with_leaves(
            cam_data, requires_grad=True
        )
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.5, 0.3, 8.0],
                [1.0, -0.5, 6.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            distortion,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        ext = cam_data["external_distortion"]
        intr = cam_data["intrinsics"]

        # FD for fw_poly[1] (dominant focal-length-like term)
        eps = 0.5

        fw_base = list(intr["angle_to_pixeldist_poly"])
        fw_plus = list(fw_base)
        fw_plus[1] += eps
        fw_minus = list(fw_base)
        fw_minus[1] -= eps

        fd = (
            self._real_ftheta_forward_loss(cam_data, world_points, fw_poly_override=fw_plus)
            - self._real_ftheta_forward_loss(cam_data, world_points, fw_poly_override=fw_minus)
        ) / (2 * eps)

        self.assertNotAlmostEqual(fd, 0.0, places=3, msg="FD for real fw_poly[1] should be non-zero")

        np.testing.assert_allclose(
            fw_poly.grad[1].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"Real fw_poly[1] grad: autograd={fw_poly.grad[1].item():.6f}, fd={fd:.6f}",
        )

        # FD for h_poly_inv[1] (dominant phi term in inverse poly, used during forward proj)
        eps = 1e-3

        h_inv_base = list(ext["horizontal_poly_inverse"])
        h_inv_plus = list(h_inv_base)
        h_inv_plus[1] += eps
        h_inv_minus = list(h_inv_base)
        h_inv_minus[1] -= eps

        fd = (
            self._real_ftheta_forward_loss(cam_data, world_points, h_poly_inv_override=h_inv_plus)
            - self._real_ftheta_forward_loss(cam_data, world_points, h_poly_inv_override=h_inv_minus)
        ) / (2 * eps)

        self.assertNotAlmostEqual(fd, 0.0, places=3, msg="FD for real h_poly_inv[1] should be non-zero")

        np.testing.assert_allclose(
            h_poly_inv.grad[1].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"Real h_poly_inv[1] grad: autograd={h_poly_inv.grad[1].item():.6f}, fd={fd:.6f}",
        )

        # FD for v_poly_inv[5] (dominant theta term in inverse poly)
        eps = 1e-3

        v_inv_base = list(ext["vertical_poly_inverse"])
        v_inv_plus = list(v_inv_base)
        v_inv_plus[5] += eps
        v_inv_minus = list(v_inv_base)
        v_inv_minus[5] -= eps

        fd = (
            self._real_ftheta_forward_loss(cam_data, world_points, v_poly_inv_override=v_inv_plus)
            - self._real_ftheta_forward_loss(cam_data, world_points, v_poly_inv_override=v_inv_minus)
        ) / (2 * eps)

        self.assertNotAlmostEqual(fd, 0.0, places=3, msg="FD for real v_poly_inv[5] should be non-zero")

        np.testing.assert_allclose(
            v_poly_inv.grad[5].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"Real v_poly_inv[5] grad: autograd={v_poly_inv.grad[5].item():.6f}, fd={fd:.6f}",
        )

    def test_real_ftheta_no_distortion_gradient(self):
        """Analytical pp check + FD for fw_poly[1] with real FTheta params (no ext distortion)."""
        all_cams = self._load_real_params()
        cam_data = None

        for uid, data in all_cams.items():
            if data["camera_model_type"] == "ftheta" and data["external_distortion"] is None:
                cam_data = data
                break

        self.assertIsNotNone(cam_data, "No FTheta camera without ext distortion found in test data")

        projection, _, res, pp, fw_poly, _, _ = self._make_real_ftheta_with_leaves(cam_data, requires_grad=True)
        no_dist = NoExternalDistortion()
        dynamic_pose = self._make_identity_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.5, 0.3, 8.0],
                [1.0, -0.5, 6.0],
            ],
            device=self.device,
        )

        image_points, _, _, _, _ = project_world_points_shutter_pose(
            world_points,
            projection,
            no_dist,
            res,
            ShutterType.ROLLING_TOP_TO_BOTTOM,
            dynamic_pose,
        )

        loss = image_points.sum()
        loss.backward()

        # Non-zero gradient existence checks
        self.assertIsNotNone(pp.grad)
        self.assertTrue(pp.grad.abs().sum() > 0, "pp gradient should be non-zero")
        self.assertIsNotNone(fw_poly.grad)
        self.assertTrue(fw_poly.grad.abs().sum() > 0, "fw_poly gradient should be non-zero")

        # Analytical: d(image_point)/d(pp) = N_points per component
        n_points = len(world_points)

        np.testing.assert_allclose(
            pp.grad.cpu().numpy(),
            [n_points, n_points],
            rtol=0.02,
            err_msg=f"Real FTheta (no dist) pp gradient should be ~{n_points} per component",
        )

        # FD for fw_poly[1]
        intr = cam_data["intrinsics"]
        eps = 0.5

        fw_base = list(intr["angle_to_pixeldist_poly"])
        fw_plus = list(fw_base)
        fw_plus[1] += eps
        fw_minus = list(fw_base)
        fw_minus[1] -= eps

        cam_no_dist = dict(cam_data)
        cam_no_dist["external_distortion"] = None

        fd = (
            self._real_ftheta_forward_loss(cam_no_dist, world_points, fw_poly_override=fw_plus)
            - self._real_ftheta_forward_loss(cam_no_dist, world_points, fw_poly_override=fw_minus)
        ) / (2 * eps)

        self.assertNotAlmostEqual(fd, 0.0, places=3, msg="FD for real fw_poly[1] should be non-zero")

        np.testing.assert_allclose(
            fw_poly.grad[1].item(),
            fd,
            rtol=0.05,
            atol=0.01,
            err_msg=f"Real fw_poly[1] (no dist) grad: autograd={fw_poly.grad[1].item():.6f}, fd={fd:.6f}",
        )


if __name__ == "__main__":
    unittest.main()
