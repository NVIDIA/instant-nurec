# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataclasses + enums consumed by the in-tree torch ray-gen
(``ray_gen.py``).

FTheta-only by design. OpenCVPinhole and OpenCVFisheye distortion
models are intentionally not supported on the input side; the other
distortion families are explicitly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class ShutterType(IntEnum):
    """Camera shutter behavior."""

    ROLLING_TOP_TO_BOTTOM = 1
    ROLLING_LEFT_TO_RIGHT = 2
    ROLLING_BOTTOM_TO_TOP = 3
    ROLLING_RIGHT_TO_LEFT = 4
    GLOBAL = 5


class FThetaPolynomialType(IntEnum):
    """Reference polynomial type for F-Theta camera model."""

    FORWARD = 0
    BACKWARD = 1


class ReferencePolynomial(IntEnum):
    """Reference polynomial type for bivariate windshield distortion."""

    FORWARD = 0
    BACKWARD = 1


@dataclass
class CameraProjection:
    """Base class for camera projection parameters."""


@dataclass
class ExternalDistortion:
    """Base class for external distortion parameters."""


@dataclass
class NoExternalDistortion(ExternalDistortion):
    """No external distortion - identity transformation."""


@dataclass
class BivariateWindshieldDistortion(ExternalDistortion):
    """Bivariate windshield distortion for FTheta camera rays."""

    h_poly: torch.Tensor
    v_poly: torch.Tensor
    h_poly_inv: torch.Tensor
    v_poly_inv: torch.Tensor
    reference_polynomial: ReferencePolynomial
    h_poly_degree: int = -1
    v_poly_degree: int = -1

    def __post_init__(self) -> None:
        self.reference_polynomial = self._coerce_reference_polynomial(self.reference_polynomial)
        h_poly_degree = self._compute_poly_order(self.h_poly, "h_poly")
        v_poly_degree = self._compute_poly_order(self.v_poly, "v_poly")
        h_poly_inv_degree = self._compute_poly_order(self.h_poly_inv, "h_poly_inv")
        v_poly_inv_degree = self._compute_poly_order(self.v_poly_inv, "v_poly_inv")

        if h_poly_inv_degree != h_poly_degree:
            raise ValueError("h_poly_inv must have the same bivariate order as h_poly.")
        if v_poly_inv_degree != v_poly_degree:
            raise ValueError("v_poly_inv must have the same bivariate order as v_poly.")
        if self.h_poly_degree >= 0 and self.h_poly_degree != h_poly_degree:
            raise ValueError("h_poly_degree does not match h_poly coefficient count.")
        if self.v_poly_degree >= 0 and self.v_poly_degree != v_poly_degree:
            raise ValueError("v_poly_degree does not match v_poly coefficient count.")
        self.h_poly_degree = h_poly_degree
        self.v_poly_degree = v_poly_degree

    @staticmethod
    def _compute_poly_order(poly_coeffs: torch.Tensor, name: str) -> int:
        """Return the order for triangular bivariate coefficient storage."""
        if poly_coeffs.dim() != 1:
            raise ValueError(f"{name} must be 1D, got shape {tuple(poly_coeffs.shape)}")
        n_coeffs = int(torch.numel(poly_coeffs))
        if n_coeffs == 0:
            raise ValueError(f"{name} must contain at least one coefficient.")
        num_terms = 0
        for order_candidate in range(n_coeffs):
            num_terms += order_candidate + 1
            if num_terms == n_coeffs:
                return order_candidate
            if num_terms > n_coeffs:
                break
        raise ValueError(
            f"{name} coefficient count is not consistent with triangular bivariate polynomial storage."
        )

    @staticmethod
    def _coerce_reference_polynomial(value: object) -> ReferencePolynomial:
        if isinstance(value, ReferencePolynomial):
            return value
        original = value
        if hasattr(value, "name"):
            value = getattr(value, "name")
        elif hasattr(value, "value") and not isinstance(value, str):
            value = getattr(value, "value")
        if isinstance(value, str):
            normalized = value.strip().upper().rsplit(".", maxsplit=1)[-1]
            if normalized in ReferencePolynomial.__members__:
                return ReferencePolynomial[normalized]
            try:
                return ReferencePolynomial(int(normalized))
            except ValueError as e:
                raise ValueError(
                    f"unrecognized reference_polynomial value: {original!r}"
                ) from e
        try:
            return ReferencePolynomial(int(value))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"unrecognized reference_polynomial value: {original!r}"
            ) from e

    @classmethod
    def from_components(
        cls,
        h_poly: torch.Tensor,
        v_poly: torch.Tensor,
        h_poly_inv: torch.Tensor,
        v_poly_inv: torch.Tensor,
        reference_polynomial: ReferencePolynomial,
    ) -> "BivariateWindshieldDistortion":
        return cls(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=reference_polynomial,
        )

    @staticmethod
    def _eval_poly_2d(
        poly: torch.Tensor, order: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate triangular bivariate coefficients with NRE's Horner order."""
        start_idx = 0
        y_coeffs = []
        for inner_order in range(order, -1, -1):
            coeffs = poly[start_idx : start_idx + inner_order + 1]
            result = torch.zeros_like(x)
            for coeff in reversed(coeffs):
                result = result * x + coeff
            y_coeffs.append(result)
            start_idx += inner_order + 1

        result = torch.zeros_like(y)
        for coeff in reversed(y_coeffs):
            result = result * y + coeff
        return result

    def _active_polynomials(self, *, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        use_inverse = inverse
        if self.reference_polynomial == ReferencePolynomial.BACKWARD:
            use_inverse = not use_inverse
        if use_inverse:
            return self.h_poly_inv, self.v_poly_inv
        return self.h_poly, self.v_poly

    def _apply_distortion(self, camera_rays: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        if camera_rays.shape[-1] != 3:
            raise ValueError(f"camera_rays must end with dimension 3, got shape {tuple(camera_rays.shape)}")

        h_poly, v_poly = self._active_polynomials(inverse=inverse)
        h_poly = h_poly.to(device=camera_rays.device, dtype=camera_rays.dtype)
        v_poly = v_poly.to(device=camera_rays.device, dtype=camera_rays.dtype)

        ray_norm = torch.nn.functional.normalize(camera_rays, dim=-1)
        phi = torch.asin(torch.clamp(ray_norm[..., 0], -1.0, 1.0))
        theta = torch.asin(torch.clamp(ray_norm[..., 1], -1.0, 1.0))
        x = torch.sin(self._eval_poly_2d(h_poly, self.h_poly_degree, phi, theta))
        y = torch.sin(self._eval_poly_2d(v_poly, self.v_poly_degree, phi, theta))
        z_square = torch.clamp(1.0 - x * x - y * y, min=0.0, max=1.0)
        z_sign = torch.where(ray_norm[..., 2] < 0.0, -1.0, 1.0)
        z = torch.sqrt(z_square) * z_sign
        return torch.stack([x, y, z], dim=-1)

    def distort_camera_rays(self, camera_rays: torch.Tensor) -> torch.Tensor:
        return self._apply_distortion(camera_rays, inverse=False)

    def undistort_camera_rays(self, camera_rays: torch.Tensor) -> torch.Tensor:
        return self._apply_distortion(camera_rays, inverse=True)


@dataclass
class FThetaProjection(CameraProjection):
    """F-Theta camera projection — pure-torch fields (no slang packing).

    The slang version packs everything into a single ``intrinsics`` tensor
    for efficient GPU transfer; we keep the unpacked form because the
    torch impl reads individual properties directly.
    """

    principal_point: torch.Tensor  # (2,)
    fw_poly: torch.Tensor  # (degree+1,) coefficients in ascending order
    bw_poly: torch.Tensor  # (degree+1,)
    A: torch.Tensor  # (2, 2)
    Ainv: torch.Tensor  # (2, 2)
    dfw_poly: torch.Tensor  # (degree,) — derivative of fw_poly
    dbw_poly: torch.Tensor  # (degree,) — derivative of bw_poly
    reference_poly: FThetaPolynomialType
    max_angle: float
    newton_iterations: int
    min_2d_norm: float

    @classmethod
    def from_components(
        cls,
        principal_point: torch.Tensor,
        fw_poly: torch.Tensor,
        bw_poly: torch.Tensor,
        A: torch.Tensor,
        Ainv: torch.Tensor,
        dfw_poly: torch.Tensor,
        dbw_poly: torch.Tensor,
        reference_poly: FThetaPolynomialType,
        max_angle: float,
        newton_iterations: int,
        min_2d_norm: float,
    ) -> "FThetaProjection":
        return cls(
            principal_point=principal_point,
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=A,
            Ainv=Ainv,
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=reference_poly,
            max_angle=max_angle,
            newton_iterations=newton_iterations,
            min_2d_norm=min_2d_norm,
        )


@dataclass
class Pose:
    """Static SE(3) pose."""

    translation: torch.Tensor  # (3,)
    rotation: torch.Tensor  # (4,) XYZW quaternion


@dataclass
class DynamicPose:
    """Time-varying pose with two control points."""

    start_pose: Pose
    end_pose: Pose


__all__ = [
    "BivariateWindshieldDistortion",
    "CameraProjection",
    "DynamicPose",
    "ExternalDistortion",
    "FThetaPolynomialType",
    "FThetaProjection",
    "NoExternalDistortion",
    "Pose",
    "ReferencePolynomial",
    "ShutterType",
]
