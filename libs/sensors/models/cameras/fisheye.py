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
OpenCV Fisheye camera model with properly-typed projection.

This module provides the OpenCV Fisheye camera model with properly-typed projection.
"""

from libs.sensors.kernels.cameras import (
    # External distortion types
    ExternalDistortion,
    OpenCVFisheyeProjection,
    # Enums and related types
    ShutterType,
)
from libs.sensors.models.cameras.camera_model import CameraModel

# Import common helpers
from libs.sensors.models.common.utils import compute_scaled_resolution


class OpenCVFisheyeCameraModel(CameraModel):
    """OpenCV Fisheye camera model with properly-typed projection.

    Extends CameraModel with OpenCVFisheyeProjection for type-safe parameter access.

    Example usage:
        # Create camera model with projection parameters
        projection = OpenCVFisheyeProjection.from_components(
            focal_length=torch.tensor([fx, fy]),
            principal_point=torch.tensor([cx, cy]),
            forward_poly=torch.tensor([k1, k2, k3, k4]),
            resolution=torch.tensor([width, height]),
            max_angle=max_angle,
            newton_iterations=10,
            min_2d_norm=torch.tensor(1e-6),
        )
        camera = OpenCVFisheyeCameraModel(
            projection=projection,
            external_distortion=NoExternalDistortion(),
            resolution=(width, height),
            shutter_type=ShutterType.GLOBAL,
        )

        # Access projection parameters with proper typing
        fx, fy = camera.projection.focal_length  # Type: Tensor (2,)
        distortion = camera.projection.forward_poly  # Fisheye distortion (k1-k4)
    """

    _projection: OpenCVFisheyeProjection

    def __init__(
        self,
        projection: OpenCVFisheyeProjection,
        external_distortion: ExternalDistortion,
        resolution: tuple[int, int],
        shutter_type: ShutterType,
    ):
        """Initialize OpenCV Fisheye camera model.

        Args:
            projection: OpenCV Fisheye projection parameters
            external_distortion: Layer 0 external distortion parameters
            resolution: (width, height) in pixels
            shutter_type: Rolling or global shutter behavior
        """
        super().__init__(external_distortion, resolution, shutter_type)
        self._projection = projection

    @property
    def projection(self) -> OpenCVFisheyeProjection:
        """Get the OpenCV Fisheye projection parameters."""
        return self._projection

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "OpenCVFisheyeCameraModel":
        """Apply image domain transformation to camera parameters.

        Args:
            image_domain_scale: Isotropic (float) or anisotropic (tuple) scaling factor
            image_domain_offset: Offset in the scaled image domain (for cropping)
            new_resolution: Optional explicit new resolution (if None, computed from scale)

        Returns:
            New OpenCVFisheyeCameraModel with transformed parameters
        """
        new_projection = self._projection.transform(
            image_domain_scale=image_domain_scale,
            image_domain_offset=image_domain_offset,
            new_resolution=new_resolution,
        )

        model_resolution = compute_scaled_resolution(self.resolution, image_domain_scale, new_resolution)

        return OpenCVFisheyeCameraModel(
            projection=new_projection,
            external_distortion=self.external_distortion,
            resolution=model_resolution,
            shutter_type=self.shutter_type,
        )


__all__ = [
    "OpenCVFisheyeCameraModel",
]
