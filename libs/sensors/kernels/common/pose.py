# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Python dataclasses for pose and trajectory types.

These dataclasses mirror the Slang struct definitions in pose.slang and trajectory.slang,
and provide the Python-side representation for pose data passed to GPU kernels.

The types defined here are:
- Pose: Static SE3 pose with translation and quaternion rotation
- Trajectory: Piecewise trajectory with control poses and normalized times
- DynamicPose: Time-varying pose over normalized time [0, 1]

These types are used by:
- libs.sensors.kernels.cameras: As inputs to camera kernel bindings
- libs.sensors.kernels.lidars: As inputs to lidar kernel bindings
- libs.sensors.models: For Layer 2 sensor model operations
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from torch import Tensor


@dataclass
class Pose:
    """SE3 pose representation with translation and quaternion rotation.

    Mirrors the SE3Pose struct in pose.slang.

    Attributes:
        translation: (3,) translation vector [x, y, z]
        rotation: (4,) quaternion in wxyz format [qw, qx, qy, qz]
    """

    translation: Tensor
    rotation: Tensor


@dataclass
class Trajectory:
    """Piecewise trajectory with control poses and normalized times.

    Mirrors the PiecewiseTrajectory struct in trajectory.slang.
    Used for interpolating poses over time.

    Attributes:
        control_poses: Sequence of Pose objects defining the trajectory keyframes
        control_count: Number of control poses (must match len(control_poses))
        control_times: (N,) tensor of normalized times [0, 1] for each control pose
    """

    control_poses: tuple[Pose, ...]
    control_count: int
    control_times: Tensor


@dataclass
class DynamicPose:
    """Time-varying pose over normalized time [0, 1].

    Represents pose interpolation between exactly two poses: start (t=0) and end (t=1).
    Used for rolling shutter compensation and motion during exposure.

    The normalized time convention:
    - t=0.0 corresponds to the start of the exposure/scan
    - t=1.0 corresponds to the end of the exposure/scan

    Attributes:
        start_pose: Pose at t=0 (start of exposure/scan)
        end_pose: Pose at t=1 (end of exposure/scan)
    """

    # Cache for control_times tensor to avoid CUDA sync on repeated tensor creation
    _control_times_cache: ClassVar[dict[torch.device, torch.Tensor]] = {}

    start_pose: Pose
    end_pose: Pose

    def to_trajectory(self) -> Trajectory:
        """Convert to Trajectory for kernel consumption.

        Returns:
            Trajectory with two control poses at t=0 and t=1
        """
        device = self.start_pose.translation.device
        if device not in DynamicPose._control_times_cache:
            DynamicPose._control_times_cache[device] = torch.tensor([0.0, 1.0], device=device)

        return Trajectory(
            control_poses=(self.start_pose, self.end_pose),
            control_count=2,
            control_times=DynamicPose._control_times_cache[device],
        )


__all__ = [
    "Pose",
    "Trajectory",
    "DynamicPose",
]
