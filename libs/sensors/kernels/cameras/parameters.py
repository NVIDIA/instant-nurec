# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Camera parameter dataclasses for Layer 0 kernel operations.

These dataclasses mirror the Slang structs and contain working parameters for GPU execution.
They are used by Layer 0 kernels and created by Layer 2 models.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch

from torch import Tensor


MAX_H_POLYNOMIAL_TERMS = 6
MAX_V_POLYNOMIAL_TERMS = 15
MAX_POLYNOMIAL_TERMS = 6

# FTheta intrinsics tensor layout offsets (derived from MAX_POLYNOMIAL_TERMS)
# Layout: [pp(2), fw_poly(N), bw_poly(N), A(4), Ainv(4), dfw_poly(N), dbw_poly(N)]
_N = MAX_POLYNOMIAL_TERMS
_FTHETA_PP_OFFSET = 0
_FTHETA_FW_OFFSET = _FTHETA_PP_OFFSET + 2  # 2
_FTHETA_BW_OFFSET = _FTHETA_FW_OFFSET + _N  # 8
_FTHETA_A_OFFSET = _FTHETA_BW_OFFSET + _N  # 14
_FTHETA_AINV_OFFSET = _FTHETA_A_OFFSET + 4  # 18
_FTHETA_DFW_OFFSET = _FTHETA_AINV_OFFSET + 4  # 22
_FTHETA_DBW_OFFSET = _FTHETA_DFW_OFFSET + _N  # 28
_FTHETA_INTRINSICS_SIZE = _FTHETA_DBW_OFFSET + _N  # 34


# Enums matching Slang definitions
class ShutterType(IntEnum):
    """Camera shutter behavior types."""

    ROLLING_TOP_TO_BOTTOM = 1
    ROLLING_LEFT_TO_RIGHT = 2
    ROLLING_BOTTOM_TO_TOP = 3
    ROLLING_RIGHT_TO_LEFT = 4
    GLOBAL = 5


class ReferencePolynomial(IntEnum):
    """Reference polynomial type for bivariate windshield distortion."""

    FORWARD = 0
    BACKWARD = 1


class FThetaPolynomialType(IntEnum):
    """Reference polynomial type for F-Theta camera model."""

    FORWARD = 0
    BACKWARD = 1


# External distortion parameter structures
@dataclass
class ExternalDistortion:
    """Base class for external distortion parameters."""

    pass


@dataclass
class NoExternalDistortion(ExternalDistortion):
    """No external distortion - identity transformation."""

    pass


def _validate_poly(poly: Tensor, max_terms: int, name: str) -> None:
    """Validate polynomial tensor shape and size.

    This prevents silent truncation that would cause degree metadata to be inconsistent
    with the actual stored coefficients, which could lead to OOB accesses in Slang kernels.

    Args:
        poly: Tensor of polynomial coefficients
        max_terms: Maximum number of terms supported by the Slang kernel
        name: Name of the polynomial for error messages

    Raises:
        ValueError: If poly is not 1D or has more coefficients than max_terms
    """
    if poly.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(poly.shape)}")
    if poly.shape[0] > max_terms:
        raise ValueError(f"{name} has {poly.shape[0]} coefficients, but at most {max_terms} are supported.")


def _pad_poly_to_max_terms(poly: Tensor, max_terms: int, name: str = "Polynomial") -> Tensor:
    """Validate and pad 1D polynomial coefficients to fixed size for Slang kernel.

    Args:
        poly: 1D tensor of polynomial coefficients
        max_terms: Maximum number of terms supported by the Slang kernel
        name: Name of the polynomial for error messages

    Returns:
        Padded tensor of shape (max_terms,)

    Raises:
        ValueError: If poly is not 1D or has more coefficients than max_terms
    """
    _validate_poly(poly, max_terms, name)
    if poly.shape[0] == max_terms:
        return poly
    return torch.cat([poly, torch.zeros(max_terms - poly.shape[0], device=poly.device, dtype=poly.dtype)])


@dataclass
class BivariateWindshieldDistortion(ExternalDistortion):
    """Bivariate windshield distortion parameters (working parameters for GPU).

    Stores all 40 differentiable parameters in a single tensor for efficient GPU transfer
    (avoiding overhead of multiple DiffTensorView objects). Individual parameters are
    accessible via properties. Non-differentiable config values are stored separately.

    Note:
        Polynomials are limited to at most 10 coefficients (degree ≤ 9) due to
        fixed-size arrays in the Slang kernel.

    Attributes:
        distortion_coeffs: (2*(MAX_H_POLYNOMIAL_TERMS+MAX_V_POLYNOMIAL_TERMS),) packed tensor containing all differentiable parameters:
            [h_poly(MAX_H_POLYNOMIAL_TERMS),
            v_poly(MAX_V_POLYNOMIAL_TERMS),
            h_poly_inv(MAX_H_POLYNOMIAL_TERMS),
            v_poly_inv(MAX_V_POLYNOMIAL_TERMS)]
        reference_polynomial: Which polynomial is the reference (forward or backward)
        h_poly_degree: Degree of horizontal polynomial (coefficients beyond this are zero)
        v_poly_degree: Degree of vertical polynomial (coefficients beyond this are zero)

    Properties:
        h_poly: (MAX_H_POLYNOMIAL_TERMS,) horizontal polynomial coefficients (padded to size MAX_H_POLYNOMIAL_TERMS)
        v_poly: (MAX_V_POLYNOMIAL_TERMS,) vertical polynomial coefficients (padded to size MAX_V_POLYNOMIAL_TERMS)
        h_poly_inv: (MAX_H_POLYNOMIAL_TERMS,) horizontal inverse polynomial coefficients (padded to size MAX_H_POLYNOMIAL_TERMS)
        v_poly_inv: (MAX_V_POLYNOMIAL_TERMS,) vertical inverse polynomial coefficients (padded to size MAX_V_POLYNOMIAL_TERMS)
    """

    distortion_coeffs: Tensor  # (40,) packed tensor
    reference_polynomial: ReferencePolynomial
    h_poly_degree: int
    v_poly_degree: int

    @staticmethod
    def _compute_poly_order(poly_coeffs: torch.Tensor):
        """Computes the order of a bivariate polynomial give it's array of coefficients"""
        order = 0
        num_terms = 0
        for order_candidate in range(torch.numel(poly_coeffs)):
            num_terms += order_candidate + 1
            if num_terms == torch.numel(poly_coeffs):
                order = order_candidate
                break
            elif num_terms > torch.numel(poly_coeffs):
                raise ValueError(
                    "The input length of the windshield distortion coefficients is not consistent with the assumed polynomial form."
                )
        return order

    @classmethod
    def from_components(
        cls,
        h_poly: Tensor,
        v_poly: Tensor,
        h_poly_inv: Tensor,
        v_poly_inv: Tensor,
        reference_polynomial: ReferencePolynomial,
    ) -> "BivariateWindshieldDistortion":
        """Create distortion from individual component tensors.

        This factory method provides a convenient API for creating distortion
        from individual parameters while internally packing them into a single
        tensor for efficient GPU transfer.

        Args:
            h_poly: Horizontal polynomial coefficients (triangular count ≤ MAX_H_POLYNOMIAL_TERMS)
            v_poly: Vertical polynomial coefficients (triangular count ≤ MAX_V_POLYNOMIAL_TERMS)
            h_poly_inv: Horizontal inverse polynomial coefficients
            v_poly_inv: Vertical inverse polynomial coefficients
            reference_polynomial: Which polynomial is the reference (forward or backward)

        Returns:
            BivariateWindshieldDistortion with packed distortion_coeffs tensor

        Raises:
            ValueError: If any polynomial exceeds the maximum supported size or has
                invalid coefficient count for bivariate form.
        """
        # Validate polynomial sizes via _compute_poly_order (checks bivariate triangular form)
        # and _pad_poly_to_max_terms (checks 1D shape and max size)
        h_poly_degree = BivariateWindshieldDistortion._compute_poly_order(h_poly)
        v_poly_degree = BivariateWindshieldDistortion._compute_poly_order(v_poly)

        assert h_poly_degree <= MAX_H_POLYNOMIAL_TERMS, (
            "h_poly_degree must be less than or equal to MAX_H_POLYNOMIAL_TERMS"
        )
        assert v_poly_degree <= MAX_V_POLYNOMIAL_TERMS, (
            "v_poly_degree must be less than or equal to MAX_V_POLYNOMIAL_TERMS"
        )

        distortion_coeffs = torch.cat(
            [
                _pad_poly_to_max_terms(h_poly, MAX_H_POLYNOMIAL_TERMS, "h_poly"),
                _pad_poly_to_max_terms(v_poly, MAX_V_POLYNOMIAL_TERMS, "v_poly"),
                _pad_poly_to_max_terms(h_poly_inv, MAX_H_POLYNOMIAL_TERMS, "h_poly_inv"),
                _pad_poly_to_max_terms(v_poly_inv, MAX_V_POLYNOMIAL_TERMS, "v_poly_inv"),
            ]
        )
        return cls(
            distortion_coeffs=distortion_coeffs,
            reference_polynomial=reference_polynomial,
            h_poly_degree=h_poly_degree,
            v_poly_degree=v_poly_degree,
        )

# Camera projection parameter structures
@dataclass
class CameraProjection:
    """Base class for camera projection parameters."""

    pass


@dataclass
class OpenCVPinholeProjection(CameraProjection):
    """OpenCV Pinhole camera projection (working parameters for GPU).

    Standard pinhole camera model with radial, tangential, and thin prism distortion.

    Stores all 18 parameters in a single tensor for efficient GPU transfer (avoiding
    overhead of multiple DiffTensorView objects). Individual parameters are accessible
    via properties.

    Attributes:
        intrinsics: (18,) packed tensor containing all parameters in order:
            [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4, resolution_x, resolution_y]

    Properties:
        focal_length: (2,) [fx, fy] focal lengths in pixels
        principal_point: (2,) [cx, cy] principal point in pixels
        radial_coeffs: (6,) [k1, k2, k3, k4, k5, k6] radial distortion coefficients
        tangential_coeffs: (2,) [p1, p2] tangential distortion coefficients
        thin_prism_coeffs: (4,) [s1, s2, s3, s4] thin prism distortion coefficients
        resolution: (2,) [width, height] resolution in pixels
    """

    intrinsics: (
        Tensor  # (18,) [fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4, resolution_x, resolution_y]
    )

    @classmethod
    def from_components(
        cls,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        resolution: Tensor,
    ) -> "OpenCVPinholeProjection":
        """Create projection from individual component tensors.

        This factory method provides a convenient API for creating projections
        from individual parameters while internally packing them into a single
        tensor for efficient GPU transfer.

        Args:
            focal_length: (2,) [fx, fy] focal lengths in pixels
            principal_point: (2,) [cx, cy] principal point in pixels
            radial_coeffs: (6,) [k1, k2, k3, k4, k5, k6] radial distortion coefficients
            tangential_coeffs: (2,) [p1, p2] tangential distortion coefficients
            thin_prism_coeffs: (4,) [s1, s2, s3, s4] thin prism distortion coefficients
            resolution: (2,) [width, height] resolution in pixels

        Returns:
            OpenCVPinholeProjection with packed intrinsics tensor
        """
        intrinsics = torch.cat(
            [
                focal_length,
                principal_point,
                radial_coeffs,
                tangential_coeffs,
                thin_prism_coeffs,
                resolution.to(torch.float32),
            ]
        )
        if intrinsics.shape[0] != 18:
            raise ValueError(f"Intrinsics tensor shape is not 18, got {intrinsics.shape}")
        return cls(intrinsics=intrinsics)

    @property
    def focal_length(self) -> Tensor:
        """(2,) [fx, fy] focal lengths in pixels."""
        return self.intrinsics[0:2]

    @property
    def principal_point(self) -> Tensor:
        """(2,) [cx, cy] principal point in pixels."""
        return self.intrinsics[2:4]

    @property
    def radial_coeffs(self) -> Tensor:
        """(6,) [k1, k2, k3, k4, k5, k6] radial distortion coefficients."""
        return self.intrinsics[4:10]

    @property
    def tangential_coeffs(self) -> Tensor:
        """(2,) [p1, p2] tangential distortion coefficients."""
        return self.intrinsics[10:12]

    @property
    def thin_prism_coeffs(self) -> Tensor:
        """(4,) [s1, s2, s3, s4] thin prism distortion coefficients."""
        return self.intrinsics[12:16]

    @property
    def resolution(self) -> Tensor:
        """(2,) [width, height] resolution in pixels."""
        return self.intrinsics[16:18]

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "OpenCVPinholeProjection":
        """Apply image domain transformation to camera parameters.

        Used when scaling/cropping images to maintain correct projections.

        Args:
            image_domain_scale: Isotropic (float) or anisotropic (tuple) scaling factor
            image_domain_offset: Offset in the scaled image domain (for cropping)
            new_resolution: Optional explicit new resolution (if None, computed from scale)

        Returns:
            New OpenCVPinholeProjection with transformed parameters
        """
        device = self.intrinsics.device
        dtype = self.intrinsics.dtype

        # Get scale factors
        if isinstance(image_domain_scale, tuple):
            scale = torch.tensor(image_domain_scale, device=device, dtype=dtype)
        else:
            scale = torch.tensor([image_domain_scale, image_domain_scale], device=device, dtype=dtype)

        offset = torch.tensor(image_domain_offset, device=device, dtype=dtype)

        # Compute new resolution
        if new_resolution is not None:
            new_res = torch.tensor(new_resolution, device=device, dtype=dtype)
        else:
            new_res = self.resolution * scale

        # Transform parameters
        new_principal_point = self.principal_point * scale - offset
        new_focal_length = self.focal_length * scale

        return OpenCVPinholeProjection.from_components(
            focal_length=new_focal_length,
            principal_point=new_principal_point,
            radial_coeffs=self.radial_coeffs.clone(),
            tangential_coeffs=self.tangential_coeffs.clone(),
            thin_prism_coeffs=self.thin_prism_coeffs.clone(),
            resolution=new_res,
        )


@dataclass
class OpenCVFisheyeProjection(CameraProjection):
    """OpenCV Fisheye camera projection (working parameters for GPU).

    Wide-angle fisheye camera model using equidistant projection.

    Stores all 16 differentiable parameters in a single tensor for efficient GPU transfer
    (avoiding overhead of multiple DiffTensorView objects). Individual parameters are
    accessible via properties. Non-differentiable config values are stored separately.

    Attributes:
        intrinsics: (11,) packed tensor containing all differentiable parameters in order:
            [cx, cy, fx, fy, k1, k2, k3, k4, ab, resolution_x, resolution_y]
        max_angle: Maximum ray angle in radians (non-differentiable)
        newton_iterations: Number of Newton iterations for undistortion (non-differentiable)
        min_2d_norm: Minimum 2D norm threshold (non-differentiable)

    Properties:
        principal_point: (2,) [cx, cy] principal point in pixels
        focal_length: (2,) [fx, fy] focal lengths in pixels
        forward_poly: (4,) [k1, k2, k3, k4] forward distortion coefficients
        approx_backward_factor: (1,) approximate backward polynomial for back projection
        resolution: (2,) [width, height] resolution in pixels
    """

    intrinsics: Tensor  # (11,) [cx, cy, fx, fy, k1, k2, k3, k4, ab, resolution_x, resolution_y]
    max_angle: float
    newton_iterations: int
    min_2d_norm: float

    @classmethod
    def from_components(
        cls,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        resolution: Tensor,
        max_angle: float,
        newton_iterations: int,
        min_2d_norm: Tensor,
    ) -> "OpenCVFisheyeProjection":
        """Create projection from individual component tensors.

        This factory method provides a convenient API for creating projections
        from individual parameters while internally packing them into a single
        tensor for efficient GPU transfer.

        Args:
            principal_point: (2,) [cx, cy] principal point in pixels
            focal_length: (2,) [fx, fy] focal lengths in pixels
            forward_poly: (4,) [k1, k2, k3, k4] forward distortion coefficients
            resolution: (2,) [width, height] resolution in pixels
            max_angle: Maximum ray angle in radians
            newton_iterations: Number of Newton iterations for undistortion
            min_2d_norm: Minimum 2D norm threshold (scalar tensor)

        Returns:
            OpenCVFisheyeProjection with packed intrinsics tensor
        """
        dist = resolution / 2 / focal_length
        approx_backward_factor = max_angle / torch.max(dist[0], dist[1])
        intrinsics = torch.cat(
            [
                principal_point,
                focal_length,
                forward_poly,
                approx_backward_factor.unsqueeze(0),
                resolution.to(torch.float32),
            ]
        )
        if intrinsics.shape[0] != 11:
            raise ValueError(f"Intrinsics tensor shape is not 11, got {intrinsics.shape}")
        return cls(
            intrinsics=intrinsics,
            max_angle=max_angle,
            newton_iterations=newton_iterations,
            min_2d_norm=float(min_2d_norm.item()) if isinstance(min_2d_norm, Tensor) else min_2d_norm,
        )

    @property
    def principal_point(self) -> Tensor:
        """(2,) [cx, cy] principal point in pixels."""
        return self.intrinsics[0:2]

    @property
    def focal_length(self) -> Tensor:
        """(2,) [fx, fy] focal lengths in pixels."""
        return self.intrinsics[2:4]

    @property
    def forward_poly(self) -> Tensor:
        """(4,) [k1, k2, k3, k4] forward distortion coefficients."""
        return self.intrinsics[4:8]

    @property
    def resolution(self) -> Tensor:
        """(2,) [width, height] resolution in pixels."""
        return self.intrinsics[9:11]

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "OpenCVFisheyeProjection":
        """Apply image domain transformation to camera parameters.

        Used when scaling/cropping images to maintain correct projections.

        Args:
            image_domain_scale: Isotropic (float) or anisotropic (tuple) scaling factor
            image_domain_offset: Offset in the scaled image domain (for cropping)
            new_resolution: Optional explicit new resolution (if None, computed from scale)

        Returns:
            New OpenCVFisheyeProjection with transformed parameters
        """
        device = self.intrinsics.device
        dtype = self.intrinsics.dtype

        # Get scale factors
        if isinstance(image_domain_scale, tuple):
            scale = torch.tensor(image_domain_scale, device=device, dtype=dtype)
        else:
            scale = torch.tensor([image_domain_scale, image_domain_scale], device=device, dtype=dtype)

        offset = torch.tensor(image_domain_offset, device=device, dtype=dtype)

        # Compute new resolution
        if new_resolution is not None:
            new_res = torch.tensor(new_resolution, device=device, dtype=dtype)
        else:
            new_res = self.resolution * scale

        # Transform parameters
        new_principal_point = self.principal_point * scale - offset
        new_focal_length = self.focal_length * scale

        return OpenCVFisheyeProjection.from_components(
            principal_point=new_principal_point,
            focal_length=new_focal_length,
            forward_poly=self.forward_poly.clone(),
            resolution=new_res,
            max_angle=self.max_angle,
            newton_iterations=self.newton_iterations,
            min_2d_norm=torch.tensor(self.min_2d_norm, device=device, dtype=dtype),
        )


@dataclass
class FThetaProjection(CameraProjection):
    """F-Theta camera projection (working parameters for GPU).

    F-theta lens model with polynomial distortion and linear transformations.

    Stores all differentiable parameters in a single packed tensor for efficient GPU transfer
    (avoiding overhead of multiple DiffTensorView objects). Individual parameters are
    accessible via properties. Non-differentiable config values are stored separately.

    Attributes:
        intrinsics: Packed tensor of shape (_FTHETA_INTRINSICS_SIZE,) containing:
            [principal_point(2), fw_poly(N), bw_poly(N), A(4), Ainv(4), dfw_poly(N), dbw_poly(N)]
            where N = MAX_POLYNOMIAL_TERMS.
        reference_poly: Which polynomial is the reference (forward or backward)
        fw_poly_degree: Degree of forward polynomial (coefficients beyond this are zero)
        bw_poly_degree: Degree of backward polynomial (coefficients beyond this are zero)
        max_angle: Maximum ray angle in radians
        newton_iterations: Number of Newton iterations for undistortion
        min_2d_norm: Minimum 2D norm threshold
    """

    intrinsics: Tensor
    reference_poly: FThetaPolynomialType
    fw_poly_degree: int
    bw_poly_degree: int
    max_angle: float
    newton_iterations: int
    min_2d_norm: float

    @classmethod
    def from_components(
        cls,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        Ainv: Tensor,
        dfw_poly: Tensor,
        dbw_poly: Tensor,
        reference_poly: FThetaPolynomialType,
        max_angle: float,
        newton_iterations: int,
        min_2d_norm: float,
    ) -> "FThetaProjection":
        """Create projection from individual component tensors.

        This factory method provides a convenient API for creating projections
        from individual parameters while internally packing them into a single
        tensor for efficient GPU transfer.

        Args:
            principal_point: (2,) [cx, cy] principal point in pixels
            fw_poly: (degree,) forward polynomial coefficients, degree ≤ MAX_POLYNOMIAL_TERMS
            bw_poly: (degree,) backward polynomial coefficients, degree ≤ MAX_POLYNOMIAL_TERMS
            A: (2, 2) forward transformation matrix (linear term)
            Ainv: (2, 2) inverse transformation matrix (linear term)
            dfw_poly: (degree,) derivative of forward polynomial, degree ≤ MAX_POLYNOMIAL_TERMS
            dbw_poly: (degree,) derivative of backward polynomial, degree ≤ MAX_POLYNOMIAL_TERMS
            reference_poly: Which polynomial is the reference (forward or backward)
            max_angle: Maximum ray angle in radians
            newton_iterations: Number of Newton iterations for undistortion
            min_2d_norm: Minimum 2D norm threshold

        Returns:
            FThetaProjection with packed intrinsics tensor

        Raises:
            ValueError: If any polynomial exceeds MAX_POLYNOMIAL_TERMS or is not 1D.
        """
        # Validation is handled by _pad_poly_to_max_terms which checks 1D shape and max size
        fw_poly_degree = fw_poly.shape[0] - 1
        bw_poly_degree = bw_poly.shape[0] - 1
        intrinsics = torch.cat(
            [
                principal_point,
                _pad_poly_to_max_terms(fw_poly, MAX_POLYNOMIAL_TERMS, "fw_poly"),
                _pad_poly_to_max_terms(bw_poly, MAX_POLYNOMIAL_TERMS, "bw_poly"),
                A.flatten(),
                Ainv.flatten(),
                _pad_poly_to_max_terms(dfw_poly, MAX_POLYNOMIAL_TERMS, "dfw_poly"),
                _pad_poly_to_max_terms(dbw_poly, MAX_POLYNOMIAL_TERMS, "dbw_poly"),
            ]
        )
        return cls(
            intrinsics=intrinsics,
            reference_poly=reference_poly,
            fw_poly_degree=fw_poly_degree,
            bw_poly_degree=bw_poly_degree,
            max_angle=max_angle,
            newton_iterations=newton_iterations,
            min_2d_norm=min_2d_norm,
        )

    @property
    def principal_point(self) -> Tensor:
        """(2,) [cx, cy] principal point in pixels."""
        return self.intrinsics[_FTHETA_PP_OFFSET : _FTHETA_PP_OFFSET + 2]

    @property
    def fw_poly(self) -> Tensor:
        """(N,) forward polynomial coefficients (padded to MAX_POLYNOMIAL_TERMS)."""
        return self.intrinsics[_FTHETA_FW_OFFSET:_FTHETA_BW_OFFSET]

    @property
    def bw_poly(self) -> Tensor:
        """(N,) backward polynomial coefficients (padded to MAX_POLYNOMIAL_TERMS)."""
        return self.intrinsics[_FTHETA_BW_OFFSET:_FTHETA_A_OFFSET]

    @property
    def dfw_poly(self) -> Tensor:
        """(N,) derivative of forward polynomial (padded to MAX_POLYNOMIAL_TERMS)."""
        return self.intrinsics[_FTHETA_DFW_OFFSET:_FTHETA_DBW_OFFSET]

    @property
    def dbw_poly(self) -> Tensor:
        """(N,) derivative of backward polynomial (padded to MAX_POLYNOMIAL_TERMS)."""
        return self.intrinsics[_FTHETA_DBW_OFFSET:_FTHETA_INTRINSICS_SIZE]

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "FThetaProjection":
        """Apply image domain transformation to camera parameters.

        Used when scaling/cropping images to maintain correct projections.
        Follows ncore's FTheta transform conventions including the 0.5px offset.

        Args:
            image_domain_scale: Isotropic (float) or anisotropic (tuple) scaling factor
            image_domain_offset: Offset in the scaled image domain (for cropping)
            new_resolution: Optional explicit new resolution (if None, computed from scale)

        Returns:
            New FThetaProjection with transformed parameters
        """
        device = self.intrinsics.device
        dtype = self.intrinsics.dtype

        # Get scale factors
        if isinstance(image_domain_scale, tuple):
            scale_u, scale_v = image_domain_scale
        else:
            scale_u = scale_v = image_domain_scale

        scale = torch.tensor([scale_u, scale_v], device=device, dtype=dtype)
        offset = torch.tensor(image_domain_offset, device=device, dtype=dtype)

        # FTheta uses 0.5px offset convention for principal point
        # Transform: (pp + 0.5) * scale - 0.5 - offset
        new_principal_point = (self.principal_point + 0.5) * scale - 0.5 - offset

        # Get original polynomial coefficients (only use active degrees)
        fw_poly_orig = self.fw_poly[: self.fw_poly_degree + 1]
        bw_poly_orig = self.bw_poly[: self.bw_poly_degree + 1]
        dfw_poly_orig = self.dfw_poly[: self.fw_poly_degree]
        dbw_poly_orig = self.dbw_poly[: self.bw_poly_degree]

        # Scale forward polynomial by v-scale (linear scaling of coefficients)
        # fw_poly maps angles -> pixel distances, so scale the output
        new_fw_poly = fw_poly_orig * scale_v

        # Scale backward polynomial by polynomial composition with 1/v-scale
        # bw_poly maps pixel distances -> angles, so scale the input
        # P(x/s) = sum(c_i * (x/s)^i) = sum(c_i / s^i * x^i)
        powers = torch.arange(len(bw_poly_orig), device=device, dtype=dtype)
        scale_factors = torch.pow(torch.tensor(1.0 / scale_v, device=device, dtype=dtype), powers)
        new_bw_poly = bw_poly_orig * scale_factors

        # Scale derivative polynomials correspondingly
        new_dfw_poly = dfw_poly_orig * scale_v
        if len(dbw_poly_orig) > 0:
            d_powers = torch.arange(len(dbw_poly_orig), device=device, dtype=dtype)
            d_scale_factors = torch.pow(torch.tensor(1.0 / scale_v, device=device, dtype=dtype), d_powers + 1)
            new_dbw_poly = dbw_poly_orig * d_scale_factors
        else:
            new_dbw_poly = dbw_poly_orig

        # Incorporate anisotropic ratio into linear term A
        # A = [c, d; e, 1] -> scale c and d by (scale_u / scale_v)
        scale_ratio = scale_u / scale_v
        A_flat = self.intrinsics[_FTHETA_A_OFFSET:_FTHETA_AINV_OFFSET].clone()
        A_flat[0] *= scale_ratio  # c
        A_flat[1] *= scale_ratio  # d
        new_A = A_flat.reshape(2, 2)

        # Ainv = 1/(c-ed) * [1, -d; -e, c] -> recompute
        c, d, e = A_flat[0], A_flat[1], A_flat[2]
        det = c - e * d
        new_Ainv = torch.tensor([[1, -d], [-e, c]], device=device, dtype=dtype) / det

        return FThetaProjection.from_components(
            principal_point=new_principal_point,
            fw_poly=new_fw_poly,
            bw_poly=new_bw_poly,
            A=new_A,
            Ainv=new_Ainv,
            dfw_poly=new_dfw_poly,
            dbw_poly=new_dbw_poly,
            reference_poly=self.reference_poly,
            max_angle=self.max_angle,
            newton_iterations=self.newton_iterations,
            min_2d_norm=self.min_2d_norm,
        )


__all__ = [
    "BivariateWindshieldDistortion",
    "CameraProjection",
    "ExternalDistortion",
    "FThetaPolynomialType",
    "FThetaProjection",
    "NoExternalDistortion",
    "OpenCVFisheyeProjection",
    "OpenCVPinholeProjection",
    "ReferencePolynomial",
    "ShutterType",
]
