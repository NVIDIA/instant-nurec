# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
F-Theta camera model with properly-typed projection.

This module provides the F-Theta camera model with properly-typed projection.
"""

from libs.sensors.kernels.cameras import (
    # External distortion types
    ExternalDistortion,
    FThetaProjection,
    # Enums and related types
    ShutterType,
)
from libs.sensors.models.cameras.camera_model import CameraModel

# Import common helpers
from libs.sensors.models.common import compute_scaled_resolution


class FThetaCameraModel(CameraModel):
    """F-Theta camera model with properly-typed projection.

    Extends CameraModel with FThetaProjection for type-safe parameter access.

    Example usage:
        # Create camera model with projection parameters
        projection = FThetaProjection.from_components(
            principal_point=torch.tensor([cx, cy]),
            fw_poly=torch.tensor([...]),
            bw_poly=torch.tensor([...]),
            A=torch.tensor([[...], [...]]),
            Ainv=torch.tensor([[...], [...]]),
            dfw_poly=torch.tensor([...]),
            dbw_poly=torch.tensor([...]),
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=max_angle,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )
        camera = FThetaCameraModel(
            projection=projection,
            external_distortion=NoExternalDistortion(),
            resolution=(width, height),
            shutter_type=ShutterType.GLOBAL,
        )

        # Access projection parameters with proper typing
        cx, cy = camera.projection.principal_point  # Type: Tensor (2,)
        fw_poly = camera.projection.fw_poly  # Forward polynomial coefficients
    """

    _projection: FThetaProjection

    def __init__(
        self,
        projection: FThetaProjection,
        external_distortion: ExternalDistortion,
        resolution: tuple[int, int],
        shutter_type: ShutterType,
    ):
        """Initialize F-Theta camera model.

        Args:
            projection: F-Theta projection parameters
            external_distortion: Layer 0 external distortion parameters
            resolution: (width, height) in pixels
            shutter_type: Rolling or global shutter behavior
        """
        super().__init__(external_distortion, resolution, shutter_type)
        self._projection = projection

    @property
    def projection(self) -> FThetaProjection:
        """Get the F-Theta projection parameters."""
        return self._projection

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "FThetaCameraModel":
        """Apply image domain transformation to camera parameters.

        Args:
            image_domain_scale: Isotropic (float) or anisotropic (tuple) scaling factor
            image_domain_offset: Offset in the scaled image domain (for cropping)
            new_resolution: Optional explicit new resolution (if None, computed from scale)

        Returns:
            New FThetaCameraModel with transformed parameters
        """
        new_projection = self._projection.transform(
            image_domain_scale=image_domain_scale,
            image_domain_offset=image_domain_offset,
            new_resolution=new_resolution,
        )

        model_resolution = compute_scaled_resolution(self.resolution, image_domain_scale, new_resolution)

        return FThetaCameraModel(
            projection=new_projection,
            external_distortion=self.external_distortion,
            resolution=model_resolution,
            shutter_type=self.shutter_type,
        )


__all__ = [
    "FThetaCameraModel",
]
