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
OpenCV Pinhole camera model with properly-typed projection.

This module provides the OpenCV Pinhole camera model with properly-typed projection.
"""

from libs.sensors.kernels.cameras import (
    # External distortion types
    ExternalDistortion,
    OpenCVPinholeProjection,
    # Enums and related types
    ShutterType,
)
from libs.sensors.models.cameras.camera_model import CameraModel
from libs.sensors.models.common import compute_scaled_resolution


class OpenCVPinholeCameraModel(CameraModel):
    """OpenCV Pinhole camera model with properly-typed projection.

    Extends CameraModel with OpenCVPinholeProjection for type-safe parameter access.

    Example usage:
        # Create camera model with projection parameters
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([fx, fy]),
            principal_point=torch.tensor([cx, cy]),
            radial_coeffs=torch.tensor([k1, k2, k3, k4, k5, k6]),
            tangential_coeffs=torch.tensor([p1, p2]),
            thin_prism_coeffs=torch.tensor([s1, s2, s3, s4]),
            resolution=torch.tensor([width, height]),
        )
        camera = OpenCVPinholeCameraModel(
            projection=projection,
            external_distortion=NoExternalDistortion(),
            resolution=(width, height),
            shutter_type=ShutterType.GLOBAL,
        )

        # Access projection parameters with proper typing
        fx, fy = camera.projection.focal_length  # Type: Tensor (2,)
        k1, k2, k3 = camera.projection.radial_coeffs[:3]  # Radial distortion
    """

    _projection: OpenCVPinholeProjection

    def __init__(
        self,
        projection: OpenCVPinholeProjection,
        external_distortion: ExternalDistortion,
        resolution: tuple[int, int],
        shutter_type: ShutterType,
    ):
        """Initialize OpenCV Pinhole camera model.

        Args:
            projection: OpenCV Pinhole projection parameters
            external_distortion: Layer 0 external distortion parameters
            resolution: (width, height) in pixels
            shutter_type: Rolling or global shutter behavior
        """
        super().__init__(external_distortion, resolution, shutter_type)
        self._projection = projection

    @property
    def projection(self) -> OpenCVPinholeProjection:
        """Get the OpenCV Pinhole projection parameters."""
        return self._projection

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "OpenCVPinholeCameraModel":
        """Apply image domain transformation to camera parameters.

        Args:
            image_domain_scale: Isotropic (float) or anisotropic (tuple) scaling factor
            image_domain_offset: Offset in the scaled image domain (for cropping)
            new_resolution: Optional explicit new resolution (if None, computed from scale)

        Returns:
            New OpenCVPinholeCameraModel with transformed parameters
        """
        new_projection = self._projection.transform(
            image_domain_scale=image_domain_scale,
            image_domain_offset=image_domain_offset,
            new_resolution=new_resolution,
        )

        model_resolution = compute_scaled_resolution(self.resolution, image_domain_scale, new_resolution)

        return OpenCVPinholeCameraModel(
            projection=new_projection,
            external_distortion=self.external_distortion,
            resolution=model_resolution,
            shutter_type=self.shutter_type,
        )


__all__ = [
    "OpenCVPinholeCameraModel",
]
