# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from instant_nurec.utils.sensors.kernel_types import (
    DynamicPose,
    OpenCVPinholeProjection,
    Pose,
)
from instant_nurec.utils.sensors.ncore_sensors_converters import CameraModelConverter
from instant_nurec.utils.sensors.ray_gen import (
    camera_rays_to_image_points,
    image_points_to_world_rays_shutter_pose,
)
from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from ncore.sensors import CameraModel


def _parameters() -> OpenCVPinholeCameraModelParameters:
    return OpenCVPinholeCameraModelParameters(
        resolution=np.array([1920, 1280], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=np.array([960.0, 640.0], dtype=np.float32),
        focal_length=np.array([1100.0, 1090.0], dtype=np.float32),
        radial_coeffs=np.array([0.01, -0.005, 0.001, 0.002, -0.001, 0.0002], dtype=np.float32),
        tangential_coeffs=np.array([0.0003, -0.0002], dtype=np.float32),
        thin_prism_coeffs=np.array([0.0001, -0.00005, 0.00007, -0.00003], dtype=np.float32),
    )


def _identity_dynamic_pose() -> DynamicPose:
    pose = Pose(
        translation=torch.zeros(3),
        rotation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )
    return DynamicPose(start_pose=pose, end_pose=pose)


def test_opencv_pinhole_inverse_projection_matches_ncore() -> None:
    model = CameraModel.from_parameters(_parameters(), device="cpu")
    converted = CameraModelConverter.convert(model)
    image_points = torch.tensor(
        [
            [0.5, 0.5],
            [960.5, 640.5],
            [1919.5, 1279.5],
            [400.25, 700.75],
        ],
        dtype=torch.float32,
    )

    world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
        image_points=image_points,
        projection=converted.projection,
        external_distortion=converted.external_distortion,
        resolution=converted.resolution,
        shutter_type=converted.shutter_type,
        dynamic_pose=_identity_dynamic_pose(),
    )
    expected = model.image_points_to_camera_rays(image_points)

    torch.testing.assert_close(world_rays[:, :3], torch.zeros_like(expected))
    torch.testing.assert_close(world_rays[:, 3:], expected, rtol=1e-6, atol=1e-7)


def test_opencv_pinhole_forward_projection_and_validity_match_ncore() -> None:
    parameters = _parameters()
    model = CameraModel.from_parameters(parameters, device="cpu")
    camera_rays = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.2, -0.1, 1.0],
            [-0.6, 0.4, 1.0],
            [0.0, 0.0, -1.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    actual = camera_rays_to_image_points(parameters, camera_rays)
    expected = model.camera_rays_to_image_points(camera_rays)

    torch.testing.assert_close(actual.image_points, expected.image_points, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.valid_flag, expected.valid_flag)


def test_opencv_pinhole_projection_transform_preserves_distortion() -> None:
    model = CameraModel.from_parameters(_parameters(), device="cpu")
    projection = CameraModelConverter.convert(model).projection
    assert isinstance(projection, OpenCVPinholeProjection)

    transformed = projection.transform(
        image_domain_scale=(0.5, 0.25),
        image_domain_offset=(10.0, 20.0),
        new_resolution=(950, 300),
    )

    torch.testing.assert_close(transformed.focal_length, torch.tensor([550.0, 272.5]))
    torch.testing.assert_close(transformed.principal_point, torch.tensor([470.0, 140.0]))
    torch.testing.assert_close(transformed.radial_coeffs, projection.radial_coeffs)
    torch.testing.assert_close(transformed.tangential_coeffs, projection.tangential_coeffs)
    torch.testing.assert_close(transformed.thin_prism_coeffs, projection.thin_prism_coeffs)
    torch.testing.assert_close(transformed.resolution, torch.tensor([950.0, 300.0]))
