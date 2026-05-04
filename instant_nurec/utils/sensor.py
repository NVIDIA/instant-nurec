# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from ncore.impl.data.types import ConcreteCameraModelParametersUnion, OpenCVPinholeCameraModelParameters, ShutterType
from ncore.impl.sensors.camera import CameraModel


def to_simple_pinhole_model_parameters(
    camera_model_parameters: ConcreteCameraModelParametersUnion,
) -> OpenCVPinholeCameraModelParameters:
    """Convert any camera model parameters to simple pinhole model parameters
    (equal focal lengths, principal point at image center).

    Predict-only always uses the `method="horizontal"`, `reduce="min"`,
    `percentile=1.0` configuration that the encoder calls with; the
    `corner`/`vertical` methods, the `max`/`mean` reductions, and the
    sub-1.0 percentile path were all dead.

    Computes camera rays at the horizontal image edges, takes the smallest
    angle off the optical axis, and uses that to set the focal length.
    """
    camera_model = CameraModel.from_parameters(camera_model_parameters, device="cpu")
    original_resolution = camera_model_parameters.resolution.astype(np.int64)
    original_principal_point = camera_model_parameters.principal_point
    pinhole_principal_point = camera_model_parameters.resolution.astype(np.float32) / 2.0

    image_points = torch.tensor(
        [
            [0, original_principal_point[1]],
            [original_resolution[0], original_principal_point[1]],
        ]
    )
    pinhole_pixel_distance = pinhole_principal_point[0].item()

    camera_rays = camera_model.image_points_to_camera_rays(image_points.float())
    camera_rays = torch.nn.functional.normalize(camera_rays, dim=-1)
    angles = torch.arccos(camera_rays[:, 2])
    fov = torch.min(angles).item()

    focal = pinhole_pixel_distance / np.tan(fov)
    return OpenCVPinholeCameraModelParameters(
        resolution=np.copy(camera_model_parameters.resolution),
        shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=pinhole_principal_point,
        focal_length=np.array([focal, focal], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )
