# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Sensors models package - Layer 2 stateful sensor models.

This module provides the model layer (Layer 2) of the sensors architecture:
- Stateful sensor models (nn.Module) with learnable parameters
- Frame structures for organizing sensor observations with poses
- Camera and LiDAR models wrapping Layer 0 kernels

The module is organized into:
- models.cameras: Camera model classes (OpenCVPinhole, OpenCVFisheye, FTheta)
- models.lidars: LiDAR model classes
- models.common: Base Frame class, return types, and utilities

Example usage:
    from libs.sensors.models import (
        OpenCVPinholeCameraModel,
        OpenCVPinholeProjection,
        NoExternalDistortion,
        ShutterType,
        Pose,
        DynamicPose,
        ImageFrame,
    )

    # Create camera model with Layer 0 projection
    projection = OpenCVPinholeProjection.from_components(...)
    camera = OpenCVPinholeCameraModel(
        projection=projection,
        external_distortion=NoExternalDistortion(),
        resolution=(1920, 1080),
        shutter_type=ShutterType.GLOBAL,
    )

    # Create a pose and dynamic pose
    pose = Pose(translation=translation, rotation=rotation)
    dynamic_pose = DynamicPose.from_static_pose(pose)

    # Create image frame with camera model
    frame = ImageFrame(
        id=0,
        camera_model=camera,
        pose=pose,
        timestamp_start_us=1000000,
        timestamp_end_us=1000000,
        image=image_tensor,
    )
"""

from libs.sensors.kernels.common.pose import DynamicPose, Pose, Trajectory
from libs.sensors.models.cameras import (
    # Re-exported Layer 0 types
    BivariateWindshieldDistortion,
    # Camera models
    CameraModel,
    CameraProjection,
    ExternalDistortion,
    FThetaCameraModel,
    FThetaPolynomialType,
    FThetaProjection,
    # Frame types
    ImageFrame,
    ImageFrameGroup,
    NoExternalDistortion,
    OpenCVFisheyeCameraModel,
    OpenCVFisheyeProjection,
    OpenCVPinholeCameraModel,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
)
from libs.sensors.models.common import (
    Frame,
    ImagePointsReturn,
    PixelsReturn,
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from libs.sensors.models.lidars import (
    LidarFrame,
    LidarFrameSet,
    LidarModel,
    # Re-exported Layer 0 types
    LidarProjection,
    RowOffsetStructuredSpinningLidarModel,
    RowOffsetStructuredSpinningLidarProjection,
)


__all__ = [
    # Pose types (from libs.geometry)
    "Pose",
    "Trajectory",
    "DynamicPose",
    # Camera models
    "CameraModel",
    "OpenCVPinholeCameraModel",
    "OpenCVFisheyeCameraModel",
    "FThetaCameraModel",
    # Camera projection types (re-exported from Layer 0)
    "CameraProjection",
    "OpenCVPinholeProjection",
    "OpenCVFisheyeProjection",
    "FThetaProjection",
    # External distortion types (re-exported from Layer 0)
    "ExternalDistortion",
    "NoExternalDistortion",
    "BivariateWindshieldDistortion",
    # Camera enums (re-exported from Layer 0)
    "ShutterType",
    "FThetaPolynomialType",
    "ReferencePolynomial",
    # LiDAR models
    "LidarModel",
    "RowOffsetStructuredSpinningLidarModel",
    # LiDAR projection types (re-exported from Layer 0)
    "LidarProjection",
    "RowOffsetStructuredSpinningLidarProjection",
    # Frame types
    "Frame",
    "ImageFrame",
    "ImageFrameGroup",
    "LidarFrame",
    "LidarFrameSet",
    # Return types
    "ImagePointsReturn",
    "PixelsReturn",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldRaysReturn",
    "SensorAnglesReturn",
    "SensorRayReturn",
    "WorldPointsToSensorAnglesReturn",
]
