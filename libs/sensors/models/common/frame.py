# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Base Frame class for trainable sensor frames with learnable pose.

This module provides the abstract base class for all sensor frame types,
containing common properties shared by cameras and LiDARs.
"""

from abc import abstractmethod
from typing import Any, Union

import torch.nn as nn

from libs.sensors.kernels.common.pose import DynamicPose, Pose


class Frame(nn.Module):
    """Base class for trainable sensor frames with learnable pose.

    Common frame properties shared by all sensor types (cameras, LiDARs).
    Subclasses add sensor-specific models and observation data.

    Use Cases:
    - Static pose: Fixed sensor position, global shutter, or when motion is negligible
    - Dynamic pose: Rolling shutter sensors, moving sensors, or temporal pose variation

    Attributes:
        id: Unique identifier for this frame (numeric preferred, string supported).
            The type of identifier is up to the user. This identifier is tied to the frame
            for the entire runtime, enabling quick lookup and access via FrameGroup types
            (e.g., ImageFrameGroup, LidarFrameSet). The id must be unique within a FrameGroup.
        pose: Learnable pose or dynamic pose (T_sensor_world or T_world_sensor)
              - Pose for static transformations
              - DynamicPose for time-varying transformations (rolling shutter)
        timestamp_start_us: Frame start timestamp in microseconds (int64)
        timestamp_end_us: Frame end timestamp in microseconds (int64)
                         For global shutter: start == end
                         For rolling shutter: start < end (capture duration)
        metadata: Flexible metadata dictionary

    Example:
        # Subclasses (ImageFrame, LidarFrame) add sensor-specific fields
        frame = ImageFrame(
            id=0,
            pose=learnable_pose,
            timestamp_start_us=1000000,
            timestamp_end_us=1033333,  # 30fps rolling shutter
            camera_model=camera,
            image=image_tensor,
        )
    """

    def __init__(
        self,
        id: Union[int, str],
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize base frame with common properties.

        Args:
            id: Unique identifier for this frame
            pose: Learnable pose (Pose) or dynamic pose (DynamicPose) object
            timestamp_start_us: Frame start timestamp in microseconds
            timestamp_end_us: Frame end timestamp in microseconds
            metadata: Optional metadata dictionary
        """
        super().__init__()
        self.id = id
        self.pose = pose
        self.timestamp_start_us = timestamp_start_us
        self.timestamp_end_us = timestamp_end_us
        self.metadata = metadata or {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - behavior depends on use case (e.g., projection, rendering).

        Must be implemented by subclasses or wrappers for specific use cases.
        """
        raise NotImplementedError("Frame.forward() must be implemented by subclasses")

    @property
    def is_rolling_shutter(self) -> bool:
        """Check if frame has rolling shutter (start != end timestamp)."""
        return self.timestamp_start_us != self.timestamp_end_us

    @property
    def frame_duration_us(self) -> int:
        """Get frame duration in microseconds."""
        return self.timestamp_end_us - self.timestamp_start_us


__all__ = [
    "Frame",
]
