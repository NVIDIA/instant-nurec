# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LiDAR kernels package - Layer 0 GPU operations for LiDAR projection."""

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch

import libs.sensors.liblidar_slang_cc as lidar_slang  # type: ignore # pycena: skip

from libs.sensors.kernels.lidars.bindings import (
    SpinningDirection,
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
    sensor_angles_to_sensor_rays,
    sensor_rays_to_sensor_angles,
)
from libs.sensors.kernels.lidars.parameters import (
    LidarProjection,
    RowOffsetStructuredSpinningLidarProjection,
)


__all__ = [
    # Slang module
    "lidar_slang",
    # Enums
    "SpinningDirection",
    # LiDAR projections
    "RowOffsetStructuredSpinningLidarProjection",
    "LidarProjection",
    # Kernel functions
    "generate_spinning_lidar_rays",
    "elements_to_sensor_angles",
    "inverse_project_spinning_lidar",
    "sensor_rays_to_sensor_angles",
    "sensor_angles_to_sensor_rays",
]
