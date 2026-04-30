# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Common utilities for sensor models.

This module provides:
- Frame: Base class for sensor frames
- Return type dataclasses for model operations
- Utility functions shared between camera and LiDAR models
"""

from libs.sensors.models.common.frame import Frame
from libs.sensors.models.common.return_types import (
    ImagePointsReturn,
    PixelsReturn,
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from libs.sensors.models.common.utils import (
    TensorLike,
    batched_quat_slerp,
    compute_scaled_resolution,
    filter_by_validity,
    poses_to_matrix,
    to_torch,
    valid_flags_to_indices,
)


__all__ = [
    # Frame base class
    "Frame",
    # Return types
    "ImagePointsReturn",
    "PixelsReturn",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldRaysReturn",
    "SensorAnglesReturn",
    "SensorRayReturn",
    "WorldPointsToSensorAnglesReturn",
    # Utils
    "TensorLike",
    "to_torch",
    "poses_to_matrix",
    "valid_flags_to_indices",
    "batched_quat_slerp",
    "compute_scaled_resolution",
    "filter_by_validity",
]
