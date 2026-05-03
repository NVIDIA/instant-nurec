"""In-tree replacements for the dataclasses + enums in
``libs.sensors.kernels.cameras.parameters`` and
``libs.sensors.kernels.common``.

After Phase A.6 the standalone consumes these dataclasses through pure-torch
ray-gen (``_image_points_to_world_rays_torch.py``); the slang kernel and its
``isinstance`` checks against the libs-side classes are gone from the
predict path. Field names mirror the libs version so the existing converter
in ``ncore_sensors_converters.py`` keeps working.

FTheta-only for now (the standalone's predict baseline uses
``camera_front_wide_120fov`` which is FTheta). OpenCVPinhole / OpenCVFisheye
+ BivariateWindshieldDistortion stubs raise on use; add them when a
non-FTheta dataset comes through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import torch


class ShutterType(IntEnum):
    """Camera shutter behavior (mirrors libs.sensors.kernels.cameras.parameters.ShutterType)."""

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
class OpenCVPinholeProjection(CameraProjection):
    """OpenCVPinhole projection — passive value container.

    The math (forward + inverse projection) lives in the slang kernel that
    Phase A.6 replaces; the torch ray-gen rejects this type until the
    pinhole branch is wired (no current dataset uses it).
    """

    focal_length: torch.Tensor
    principal_point: torch.Tensor
    radial_coeffs: torch.Tensor
    tangential_coeffs: torch.Tensor
    thin_prism_coeffs: torch.Tensor
    resolution: torch.Tensor

    @classmethod
    def from_components(
        cls,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
        radial_coeffs: torch.Tensor,
        tangential_coeffs: torch.Tensor,
        thin_prism_coeffs: torch.Tensor,
        resolution: torch.Tensor,
    ) -> "OpenCVPinholeProjection":
        return cls(
            focal_length=focal_length,
            principal_point=principal_point,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            resolution=resolution,
        )


@dataclass
class OpenCVFisheyeProjection(CameraProjection):
    """OpenCVFisheye projection — passive value container (see Pinhole note)."""

    focal_length: torch.Tensor
    principal_point: torch.Tensor
    forward_poly: torch.Tensor
    resolution: torch.Tensor
    max_angle: float
    newton_iterations: int
    min_2d_norm: torch.Tensor

    @classmethod
    def from_components(
        cls,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
        forward_poly: torch.Tensor,
        resolution: torch.Tensor,
        max_angle: float,
        newton_iterations: int,
        min_2d_norm: torch.Tensor,
    ) -> "OpenCVFisheyeProjection":
        return cls(
            focal_length=focal_length,
            principal_point=principal_point,
            forward_poly=forward_poly,
            resolution=resolution,
            max_angle=max_angle,
            newton_iterations=newton_iterations,
            min_2d_norm=min_2d_norm,
        )


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
    """Static SE(3) pose — replaces libs.sensors.kernels.common.Pose."""

    translation: torch.Tensor  # (3,)
    rotation: torch.Tensor  # (4,) XYZW quaternion


@dataclass
class DynamicPose:
    """Time-varying pose with two control points — replaces
    libs.sensors.kernels.common.DynamicPose."""

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
    "OpenCVFisheyeProjection",
    "OpenCVPinholeProjection",
    "Pose",
    "ReferencePolynomial",
    "ShutterType",
]
