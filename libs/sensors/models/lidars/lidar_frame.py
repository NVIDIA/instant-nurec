# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LidarFrame class for trainable LiDAR frames with learnable pose.

This module provides the LidarFrame class that extends Frame with LiDAR-specific
model and observation data.
"""

from typing import TYPE_CHECKING, Any, Union

from torch import Tensor

from libs.sensors.kernels.common.pose import DynamicPose, Pose
from libs.sensors.models.common.frame import Frame


if TYPE_CHECKING:
    from libs.sensors.models.lidars.lidar_model import LidarModel


class LidarFrame(Frame):
    """A trainable LiDAR frame extending Frame.

    Adds LiDAR-specific model and observation data.
    Inherits common properties from Frame (id, pose, timestamps, metadata).

    Data Format:
    - Dense: distance_m shape (H, W, R) - full range image with R returns per ray,
             model_element indices are implicit grid indices
    - Sparse: distance_m shape (N, R) - filtered measurements with explicit
              model_element (N, 2) indices
    - R: Maximum number of returns per ray (typically 1-4, common values: 1, 2, or 3)
    - Invalid returns marked with NaN (for distance_m) or 0.0 (for intensity)

    Frame-level timestamps vs Point-level timestamps:
    - timestamp_start_us/timestamp_end_us (from Frame): Frame capture interval
    - timestamp_us (LiDAR-specific): Per-point timestamps within frame interval
      Used for spinning LiDARs where each point has a unique capture time
      Note: Singular name (timestamp_us) following NCore convention for per-point values

    Attributes (in addition to Frame attributes):
        lidar_model: The LiDAR model used to capture this frame (Layer 2)
        distance_m: Distance measurements in meters float32
                   - Dense format: (H, W, R) - full range image with R returns per ray
                   - Sparse format: (N, R) - filtered measurements with R returns per ray
                   - R = max returns per ray (typically 1-4)
                   - Invalid/missing returns marked with NaN
        intensity: Intensity values float32 [0, 1]
                  - Dense format: (H, W, R)
                  - Sparse format: (N, R)
                  - Invalid/missing returns marked with 0.0
        model_element: Model element indices (N, 2) int32 [row, col] - only for sparse format
                      Specifies which (row, col) in the sensor model each measurement corresponds to
        timestamp_us: Per-point timestamps int64 or None
                     - Dense format: (H, W) or (H, W, R) - same timestamp for all returns or per-return
                     - Sparse format: (N,) or (N, R) - same timestamp for all returns or per-return
                     - For spinning LiDARs with rolling shutter
                     - Note: Singular name following NCore convention
        optional_properties: Dict of optional per-ray properties (e.g., elongation)

    Optional Properties (stored as buffers if present):
        elongation: Per-ray elongation/pulse width float32
                   - Dense format: (H, W, R)
                   - Sparse format: (N, R)
                   - Invalid/missing returns marked with NaN
        semantic_class: Semantic classification labels int32
                       - Dense format: (H, W, R) or (H, W) - per-return or per-ray
                       - Sparse format: (N, R) or (N,) - per-return or per-ray

    Example:
        # Create dense LiDAR frame
        frame = LidarFrame(
            id=0,
            lidar_model=lidar,
            pose=learnable_pose,
            timestamp_start_us=1000000,
            timestamp_end_us=1100000,  # 100ms scan
            distance_m=distance_tensor,  # (128, 2048, 2) for dual return
            intensity=intensity_tensor,  # (128, 2048, 2)
            timestamp_us=per_point_timestamps,  # (128, 2048)
        )
    """

    # Type hints for registered buffers (set via register_buffer in __init__)
    distance_m: Tensor
    intensity: Tensor
    model_element: Tensor | None
    timestamp_us: Tensor | None

    def __init__(
        self,
        id: Union[int, str],
        lidar_model: "LidarModel",
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        distance_m: Tensor,  # (H, W, R) or (N, R) float32 - distances in meters, R = max returns
        intensity: Tensor,  # (H, W, R) or (N, R) float32 [0, 1]
        model_element: Tensor | None = None,  # (N, 2) int32 - for sparse format only
        timestamp_us: Tensor | None = None,  # (H, W) or (N,) or (H, W, R) or (N, R) int64
        optional_properties: dict[str, Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize LiDAR frame with model and observation data.

        Args:
            id: Unique identifier for this frame
            lidar_model: The LiDAR model used to capture this frame
            pose: Learnable pose (Pose) or dynamic pose (DynamicPose) object
            timestamp_start_us: Frame start timestamp in microseconds
            timestamp_end_us: Frame end timestamp in microseconds
            distance_m: Distance measurements tensor
            intensity: Intensity values tensor
            model_element: Element indices for sparse format (optional)
            timestamp_us: Per-point timestamps (optional)
            optional_properties: Dict of optional per-ray properties
            metadata: Optional metadata dictionary
        """
        super().__init__(
            id=id,
            pose=pose,
            timestamp_start_us=timestamp_start_us,
            timestamp_end_us=timestamp_end_us,
            metadata=metadata,
        )
        self.lidar_model = lidar_model

        # Core properties (always present)
        self.register_buffer("distance_m", distance_m)
        self.register_buffer("intensity", intensity)

        # Sparse format support
        if model_element is not None:
            self.register_buffer("model_element", model_element)
        else:
            self.model_element = None

        # Per-point timestamps (for spinning LiDARs)
        if timestamp_us is not None:
            self.register_buffer("timestamp_us", timestamp_us)
        else:
            self.timestamp_us = None

        # Optional properties
        self._optional_property_names: list[str] = []
        if optional_properties:
            for key, value in optional_properties.items():
                self.register_buffer(key, value)
                self._optional_property_names.append(key)

    def forward(self, *args, **kwargs):
        """Forward pass - behavior depends on use case (e.g., projection, ray generation).

        This is a placeholder that should be implemented by subclasses or wrappers
        based on specific use cases like point cloud generation or pose optimization.
        """
        raise NotImplementedError(
            "LidarFrame.forward() should be implemented by subclasses or wrappers "
            "for specific use cases (e.g., point cloud generation, pose optimization)"
        )

    @property
    def is_sparse(self) -> bool:
        """Check if frame uses sparse format (has explicit model_element indices)."""
        return self.model_element is not None

    @property
    def is_dense(self) -> bool:
        """Check if frame uses dense format (H, W, R) tensors."""
        return self.model_element is None

    @property
    def n_points(self) -> int:
        """Get number of measurement points."""
        if self.is_sparse:
            return self.distance_m.shape[0]
        else:
            # Dense format: H * W
            return self.distance_m.shape[0] * self.distance_m.shape[1]

    @property
    def max_returns(self) -> int:
        """Get maximum number of returns per ray."""
        return self.distance_m.shape[-1]

    @property
    def optional_properties(self) -> dict[str, Tensor]:
        """Get dict of optional properties."""
        return {name: getattr(self, name) for name in self._optional_property_names}


# Type alias for a collection of LiDAR frames indexed by ID
LidarFrameSet = dict[Union[int, str], LidarFrame]


__all__ = [
    "LidarFrame",
    "LidarFrameSet",
]
