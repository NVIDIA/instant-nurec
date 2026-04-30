# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import viser.transforms as vtf

from ncore_internal.data.v3 import CameraSensor


class Camera:
    def __init__(self, camera_sensor: CameraSensor) -> None:
        width, height = camera_sensor.get_camera_model_parameters().resolution

        self.sensor = camera_sensor
        self.fov: float = 1.2
        self.aspect = width / height
        self.scale = 0.15

    def position_at_frame(self, frame: int) -> np.ndarray:
        """
        Returns the camera's 3D world position at given [frame]

        Args:
            frame (int): frame within range of the camera

        Returns:
            np.ndarray: a 3x3 world transform matrix
        """
        T_world_camera = np.linalg.inv(self.sensor.get_frame_T_world_sensor(frame))
        return T_world_camera[:3, 3]

    def wxyz_at_frame(self, frame: int) -> np.ndarray:
        """
        Returns the xyzw world coordinates at given [frame]

        Args:
            frame (int): frame witin range of the camera

        Returns:
            np.ndarray: a 4-value array corresponding the wxyz coordinates
        """
        T_world_camera = np.linalg.inv(self.sensor.get_frame_T_world_sensor(frame))
        return vtf.SO3.from_matrix(T_world_camera[:3, :3]).wxyz

    def get_fov(self) -> float:
        """
        Returns a field-of-view (full angle in radians) for viewing the camera data.
        This is harded since a viser client's camera has a singular fov and using the
        camera's intrinsic fov will cause issues when viewing within the viewer.

        Returns:
            float: field-of-view (full angle in radians) to view the camera in
        """
        return self.fov
