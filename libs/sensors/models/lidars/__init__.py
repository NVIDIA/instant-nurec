# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LiDAR models package - Layer 2 stateful LiDAR models.

This module provides:
- LidarModel: Abstract base class for all LiDAR models
- RowOffsetStructuredSpinningLidarModel: Row-offset structured spinning LiDAR model
- LidarFrame: Frame class for LiDAR observations
"""

from libs.sensors.kernels.lidars import (
    LidarProjection,
    RowOffsetStructuredSpinningLidarProjection,
)
from libs.sensors.models.lidars.lidar_frame import LidarFrame, LidarFrameSet
from libs.sensors.models.lidars.lidar_model import LidarModel
from libs.sensors.models.lidars.row_offset_structured_spinning import RowOffsetStructuredSpinningLidarModel


__all__ = [
    # LiDAR models
    "LidarModel",
    "RowOffsetStructuredSpinningLidarModel",
    # Frame types
    "LidarFrame",
    "LidarFrameSet",
    # Re-exported Layer 0 projection types
    "LidarProjection",
    "RowOffsetStructuredSpinningLidarProjection",
]
