# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch

from ncore.impl.data.types import ConcreteCameraModelParametersUnion, OpenCVPinholeCameraModelParameters, ShutterType
from ncore.impl.sensors.camera import CameraModel


def to_simple_pinhole_model_parameters(
    camera_model_parameters: ConcreteCameraModelParametersUnion,
) -> OpenCVPinholeCameraModelParameters:
    """Convert any camera model parameters to simple pinhole model parameters
    (equal focal lengths, principal point at image center).

    Predict-only standalone always uses the `method="horizontal"`, `reduce="min"`,
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
