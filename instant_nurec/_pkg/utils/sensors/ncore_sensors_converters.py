# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Converters from ncore sensor models to kernel-compatible parameter types.

This module provides converter classes that transform ncore camera and lidar models
into the working parameter dataclasses used by Layer 0 GPU kernels.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from instant_nurec._pkg.utils.sensors._kernel_types import (
    BivariateWindshieldDistortion,
    CameraProjection,
    DynamicPose,
    ExternalDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    Pose,
    ReferencePolynomial,
    ShutterType,
)
from ncore.data import FThetaCameraModelParameters
from ncore.data import ReferencePolynomial as NcoreReferencePolynomial
from ncore.sensors import (
    BivariateWindshieldModel,
    CameraModel,
    FThetaCameraModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
)


@dataclass
class CameraModelConverterResult:
    projection: CameraProjection
    external_distortion: ExternalDistortion
    resolution: tuple[int, int]
    shutter_type: ShutterType


class CameraModelConverter:
    """Converts ncore CameraModel to kernel-compatible CameraProjection and ExternalDistortion."""

    @staticmethod
    def convert(
        camera_model: CameraModel,
        device: Optional[torch.device] = None,
    ) -> CameraModelConverterResult:
        """Convert ncore CameraModel to kernel-compatible parameter types.

        Args:
            camera_model: ncore camera model to convert
            device: Target device for tensor parameters (defaults to CPU)
            dtype: Target dtype for tensor parameters

        Returns:
            CameraModelConverterResult containing projection, external distortion, resolution, and shutter type

        Raises:
            TypeError: If camera model type is not supported
        """
        if device is None:
            device = torch.device("cpu")

        # Convert projection based on type
        projection: CameraProjection
        if isinstance(camera_model, OpenCVPinholeCameraModel):
            projection = CameraModelConverter._convert_opencv_pinhole(camera_model, device)
        elif isinstance(camera_model, OpenCVFisheyeCameraModel):
            projection = CameraModelConverter._convert_opencv_fisheye(camera_model, device)
        elif isinstance(camera_model, FThetaCameraModel):
            projection = CameraModelConverter._convert_ftheta(camera_model, device)
        else:
            raise TypeError(f"Unsupported camera model type: {type(camera_model).__name__}")

        # Convert external distortion
        external_distortion = CameraModelConverter._convert_external_distortion(camera_model, device)

        return CameraModelConverterResult(
            projection=projection,
            external_distortion=external_distortion,
            resolution=tuple(camera_model.resolution.tolist()),
            shutter_type=ShutterType(camera_model.shutter_type.value),
        )

    @staticmethod
    def _convert_opencv_pinhole(
        camera_model: OpenCVPinholeCameraModel,
        device: torch.device,
    ) -> OpenCVPinholeProjection:
        """Convert OpenCVPinholeCameraModel to OpenCVPinholeProjection."""

        return OpenCVPinholeProjection.from_components(
            focal_length=camera_model.focal_length.to(device),
            principal_point=camera_model.principal_point.to(device),
            radial_coeffs=camera_model.radial_coeffs.to(device),
            tangential_coeffs=camera_model.tangential_coeffs.to(device),
            thin_prism_coeffs=camera_model.thin_prism_coeffs.to(device),
            resolution=camera_model.resolution.to(device),
        )

    @staticmethod
    def _convert_opencv_fisheye(
        camera_model: OpenCVFisheyeCameraModel,
        device: torch.device,
    ) -> OpenCVFisheyeProjection:
        """Convert OpenCVFisheyeCameraModel to OpenCVFisheyeProjection."""

        return OpenCVFisheyeProjection.from_components(
            focal_length=camera_model.focal_length.to(device),
            principal_point=camera_model.principal_point.to(device),
            forward_poly=camera_model.forward_poly[[3, 5, 7, 9]].to(device),
            resolution=camera_model.resolution.to(device),
            max_angle=camera_model.max_angle,
            newton_iterations=camera_model.newton_iterations,
            min_2d_norm=torch.tensor(1e-6, device=device),
        )

    @staticmethod
    def _convert_ftheta(
        camera_model: FThetaCameraModel,
        device: torch.device,
    ) -> FThetaProjection:
        """Convert FThetaCameraModel to FThetaProjection."""

        return FThetaProjection.from_components(
            principal_point=camera_model.principal_point.to(device),
            fw_poly=camera_model.fw_poly.to(device),
            bw_poly=camera_model.bw_poly.to(device),
            A=camera_model.A.to(device),
            Ainv=camera_model.Ainv.to(device),
            dfw_poly=camera_model.dfw_poly.to(device),
            dbw_poly=camera_model.dbw_poly.to(device),
            reference_poly=FThetaPolynomialType.FORWARD
            if camera_model.reference_poly == FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST
            else FThetaPolynomialType.BACKWARD,
            max_angle=camera_model.max_angle,
            newton_iterations=camera_model.newton_iterations,
            min_2d_norm=1e-6,
        )

    @staticmethod
    def _convert_external_distortion(
        camera_model: CameraModel,
        device: torch.device,
    ) -> ExternalDistortion:
        """Convert external distortion parameters from ncore model.

        Args:
            camera_model: ncore camera model
            device: Target device for tensors
            dtype: Target dtype for tensors

        Returns:
            ExternalDistortion subclass instance
        """
        # Check if external distortion is present
        if camera_model.external_distortion is None:
            return NoExternalDistortion()

        # Check the type of external distortion
        # ncore uses BivariateWindshieldModelParameters
        if isinstance(camera_model.external_distortion, BivariateWindshieldModel):
            # Convert polynomial coefficients
            return BivariateWindshieldDistortion.from_components(
                h_poly=camera_model.external_distortion.horizontal_poly.to(device),
                v_poly=camera_model.external_distortion.vertical_poly.to(device),
                h_poly_inv=camera_model.external_distortion.horizontal_poly_inverse.to(device),
                v_poly_inv=camera_model.external_distortion.vertical_poly_inverse.to(device),
                reference_polynomial=ReferencePolynomial.FORWARD
                if camera_model.external_distortion.reference_poly == NcoreReferencePolynomial.FORWARD
                else ReferencePolynomial.BACKWARD,
            )

        # Default to no external distortion
        return NoExternalDistortion()


__all__ = [
    "CameraModelConverter",
    "CameraModelConverterResult",
    "Pose",
    "DynamicPose",
]
