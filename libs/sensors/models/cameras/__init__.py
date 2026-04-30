# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Camera models package - Layer 2 stateful camera models.

This module provides:
- CameraModel: Abstract base class for all camera models
- OpenCVPinholeCameraModel: Standard pinhole with radial/tangential distortion
- OpenCVFisheyeCameraModel: Wide-angle fisheye with equidistant projection
- FThetaCameraModel: F-theta lens model with polynomial distortion
- ImageFrame: Frame class for camera observations
"""

from libs.sensors.kernels.cameras import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
)
from libs.sensors.models.cameras.camera_model import CameraModel
from libs.sensors.models.cameras.fisheye import OpenCVFisheyeCameraModel
from libs.sensors.models.cameras.ftheta import FThetaCameraModel
from libs.sensors.models.cameras.image_frame import ImageFrame, ImageFrameGroup
from libs.sensors.models.cameras.pinhole import OpenCVPinholeCameraModel


__all__ = [
    # Camera models
    "CameraModel",
    "OpenCVPinholeCameraModel",
    "OpenCVFisheyeCameraModel",
    "FThetaCameraModel",
    # Frame types
    "ImageFrame",
    "ImageFrameGroup",
    # Re-exported Layer 0 projection types
    "CameraProjection",
    "OpenCVPinholeProjection",
    "OpenCVFisheyeProjection",
    "FThetaProjection",
    # Re-exported Layer 0 distortion types
    "ExternalDistortion",
    "NoExternalDistortion",
    "BivariateWindshieldDistortion",
    # Re-exported Layer 0 enums
    "ShutterType",
    "FThetaPolynomialType",
    "ReferencePolynomial",
]
