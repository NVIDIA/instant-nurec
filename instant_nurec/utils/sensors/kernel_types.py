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
models are intentionally not supported on the input side (Kelvin
predict was only ever exercised with FTheta cameras; per-
@jiahuihuang the other distortion families are explicitly dropped).
The ``BivariateWindshieldDistortion`` remains because it pairs with
FTheta on real datasets.
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
    """Bivariate windshield distortion — passive value container.

    Math lives in the slang kernel; torch ray-gen rejects this type until
    the windshield branch is wired (no current predict dataset uses it).
    """

    h_poly: torch.Tensor
    v_poly: torch.Tensor
    h_poly_inv: torch.Tensor
    v_poly_inv: torch.Tensor
    reference_polynomial: ReferencePolynomial

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
