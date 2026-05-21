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

"""Converters from ncore camera models to kernel-compatible parameter types."""

from dataclasses import dataclass
from typing import Optional

import torch

from instant_nurec.utils.sensors.kernel_types import (
    BivariateWindshieldDistortion,
    CameraProjection,
    DynamicPose,
    ExternalDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    Pose,
    ShutterType,
)
from ncore.data import FThetaCameraModelParameters
from ncore.sensors import (
    BivariateWindshieldModel,
    CameraModel,
    FThetaCameraModel,
)


def _looks_like_bivariate_windshield(external_distortion: object) -> bool:
    return all(
        hasattr(external_distortion, name)
        for name in (
            "horizontal_poly",
            "vertical_poly",
            "horizontal_poly_inverse",
            "vertical_poly_inverse",
        )
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

        # Convert projection based on type. FTheta is the only supported
        # input projection — Pinhole/Fisheye were intentionally dropped
        # because Kelvin predict was never exercised with them.
        projection: CameraProjection
        if isinstance(camera_model, FThetaCameraModel):
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

        external_distortion = camera_model.external_distortion

        # ncore exposes this as BivariateWindshieldModel on constructed
        # cameras, but some loaders hand us the parameter object directly.
        if isinstance(external_distortion, BivariateWindshieldModel) or _looks_like_bivariate_windshield(
            external_distortion
        ):
            def tensor(name: str) -> torch.Tensor:
                return torch.as_tensor(
                    getattr(external_distortion, name),
                    device=device,
                ).flatten()

            reference_polynomial = getattr(external_distortion, "reference_poly", None)
            if reference_polynomial is None:
                reference_polynomial = getattr(
                    external_distortion, "reference_polynomial", None
                )
            return BivariateWindshieldDistortion.from_components(
                h_poly=tensor("horizontal_poly"),
                v_poly=tensor("vertical_poly"),
                h_poly_inv=tensor("horizontal_poly_inverse"),
                v_poly_inv=tensor("vertical_poly_inverse"),
                reference_polynomial=reference_polynomial,
            )

        # Default to no external distortion
        return NoExternalDistortion()


__all__ = [
    "CameraModelConverter",
    "CameraModelConverterResult",
    "Pose",
    "DynamicPose",
]
