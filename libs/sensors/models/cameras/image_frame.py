# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""ImageFrame class for trainable camera frames with learnable pose.

This module provides the ImageFrame class that extends Frame with camera-specific
model and image observation data.
"""

from typing import TYPE_CHECKING, Any, Union

from torch import Tensor

from libs.sensors.kernels.common.pose import DynamicPose, Pose
from libs.sensors.models.common.frame import Frame


if TYPE_CHECKING:
    from libs.sensors.models.cameras.camera_model import CameraModel


class ImageFrame(Frame):
    """A trainable camera frame extending Frame.

    Adds camera-specific model and observation data.
    Inherits common properties from Frame (id, pose, timestamps, metadata).

    Attributes (in addition to Frame attributes):
        camera_model: The camera model used to capture this frame (Layer 2)
                     Can be any derived CameraModel type (OpenCVPinholeCameraModel,
                     OpenCVFisheyeCameraModel, FThetaCameraModel)
        image: Image tensor (H, W, C) float32 [0, 1]

    Example:
        # Create image frame with camera model and image
        frame = ImageFrame(
            id=0,
            camera_model=camera,
            pose=learnable_pose,
            timestamp_start_us=1000000,
            timestamp_end_us=1033333,  # rolling shutter
            image=image_tensor,  # (H, W, 3)
        )

        # Access properties
        print(f"Frame {frame.id} captured by {type(frame.camera_model).__name__}")
        print(f"Image shape: {frame.image.shape}")
    """

    # Type hint for registered buffer (set via register_buffer in __init__)
    image: Tensor

    def __init__(
        self,
        id: Union[int, str],
        camera_model: "CameraModel",  # Base type accepts all derived camera models
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        image: Tensor,  # (H, W, C)
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize image frame with camera model and observation.

        Args:
            id: Unique identifier for this frame
            camera_model: The camera model used to capture this frame
            pose: Learnable pose (Pose) or dynamic pose (DynamicPose) object
            timestamp_start_us: Frame start timestamp in microseconds
            timestamp_end_us: Frame end timestamp in microseconds
            image: Image tensor (H, W, C) float32 [0, 1]
            metadata: Optional metadata dictionary
        """
        super().__init__(
            id=id,
            pose=pose,
            timestamp_start_us=timestamp_start_us,
            timestamp_end_us=timestamp_end_us,
            metadata=metadata,
        )
        self.camera_model = camera_model
        self.register_buffer("image", image)

    def forward(self, *args, **kwargs):
        """Forward pass - behavior depends on use case (e.g., projection, rendering).

        This is a placeholder that should be implemented by subclasses or wrappers
        based on specific use cases like rendering or pose optimization.
        """
        raise NotImplementedError(
            "ImageFrame.forward() should be implemented by subclasses or wrappers "
            "for specific use cases (e.g., rendering, pose optimization)"
        )

    @property
    def height(self) -> int:
        """Get image height in pixels."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Get image width in pixels."""
        return self.image.shape[1]

    @property
    def channels(self) -> int:
        """Get number of image channels."""
        return self.image.shape[2] if self.image.dim() > 2 else 1


# Type alias for a collection of image frames indexed by ID
ImageFrameGroup = dict[Union[int, str], ImageFrame]


__all__ = [
    "ImageFrame",
    "ImageFrameGroup",
]
