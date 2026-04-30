# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for FThetaCameraModel.

Tests cover:
- Model initialization
- Transform method
- Projection/back-projection consistency
- World point projection
- World ray generation
"""

import unittest

import torch

from libs.sensors.kernels.cameras import (
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    ShutterType,
)
from libs.sensors.kernels.common.pose import DynamicPose, Pose
from libs.sensors.models.cameras import CameraModel, FThetaCameraModel


class TestFThetaCameraModel(unittest.TestCase):
    """Tests for FThetaCameraModel."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.resolution = (640, 480)

        # Create an F-theta projection with simple linear mapping
        # For a simple test, use identity-like polynomials
        self.projection = FThetaProjection.from_components(
            principal_point=torch.tensor([320.0, 240.0], device=self.device, dtype=self.dtype),
            fw_poly=torch.tensor([0.0, 300.0], device=self.device, dtype=self.dtype),  # r = 300 * theta
            bw_poly=torch.tensor([0.0, 1.0 / 300.0], device=self.device, dtype=self.dtype),  # theta = r / 300
            A=torch.eye(2, device=self.device, dtype=self.dtype),
            Ainv=torch.eye(2, device=self.device, dtype=self.dtype),
            dfw_poly=torch.tensor([300.0], device=self.device, dtype=self.dtype),
            dbw_poly=torch.tensor([1.0 / 300.0], device=self.device, dtype=self.dtype),
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )

        self.camera = FThetaCameraModel(
            projection=self.projection,
            external_distortion=NoExternalDistortion(),
            resolution=self.resolution,
            shutter_type=ShutterType.GLOBAL,
        )

        # Create a static pose (identity)
        self.static_pose = Pose(
            translation=torch.zeros(3, device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
        )

        # Create a dynamic pose
        pose_start = Pose(
            translation=torch.zeros(3, device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
        )
        pose_end = Pose(
            translation=torch.tensor([0.1, 0.0, 0.0], device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
        )
        self.dynamic_pose = DynamicPose(start_pose=pose_start, end_pose=pose_end)

    def test_ftheta_camera_initialization(self):
        """Test F-theta camera model initialization."""
        self.assertEqual(self.camera.resolution, self.resolution)
        self.assertIsInstance(self.camera.projection, FThetaProjection)
        self.assertEqual(self.camera.shutter_type, ShutterType.GLOBAL)

    def test_ftheta_is_camera_model(self):
        """Test that FThetaCameraModel is a CameraModel."""
        self.assertIsInstance(self.camera, CameraModel)

    def test_ftheta_transform(self):
        """Test F-theta camera transform."""
        scaled_camera = self.camera.transform(image_domain_scale=0.5, new_resolution=(320, 240))

        self.assertEqual(scaled_camera.resolution, (320, 240))
        self.assertIsInstance(scaled_camera.projection, FThetaProjection)

    def test_ftheta_transform_with_offset(self):
        """Test F-theta camera transform with offset."""
        scaled_camera = self.camera.transform(
            image_domain_scale=1.0,
            image_domain_offset=(50.0, 50.0),
            new_resolution=(540, 380),
        )

        self.assertEqual(scaled_camera.resolution, (540, 380))

    def test_ftheta_projection_roundtrip(self):
        """Test F-theta projection and back-projection consistency."""
        # Image points close to principal point (within FOV)
        image_points = torch.tensor(
            [[320.0, 240.0], [350.0, 260.0], [290.0, 220.0]], device=self.device, dtype=self.dtype
        )

        # Convert to camera rays
        camera_rays = self.camera.image_points_to_camera_rays(image_points)

        # Project back
        result = self.camera.camera_rays_to_image_points(camera_rays)

        # Should be approximately equal
        torch.testing.assert_close(result.image_points, image_points, atol=1e-2, rtol=1e-2)

    def test_world_points_to_image_points_static_pose(self):
        """Test world point projection with static pose."""
        world_points = torch.tensor(
            [[0.0, 0.0, 5.0], [0.5, 0.0, 5.0], [-0.5, 0.5, 10.0]], device=self.device, dtype=self.dtype
        )

        result = self.camera.world_points_to_image_points_static_pose(
            world_points=world_points,
            pose=self.static_pose,
            return_valid_flag=True,
            return_T_world_sensors=True,
        )

        self.assertEqual(result.image_points.shape[1], 2)
        self.assertIsNotNone(result.valid_flag)
        self.assertIsNotNone(result.T_world_sensors)

    def test_world_points_to_pixels_static_pose(self):
        """Test world point to pixel projection."""
        world_points = torch.tensor([[0.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)

        result = self.camera.world_points_to_pixels_static_pose(
            world_points=world_points,
            pose=self.static_pose,
            return_valid_flag=True,
        )

        self.assertEqual(result.pixels.dtype, torch.int32)
        self.assertEqual(result.pixels.shape, (1, 2))

    def test_image_points_to_world_rays_static_pose(self):
        """Test back-projection to world rays with static pose."""
        image_points = torch.tensor([[320.0, 240.0], [350.0, 260.0]], device=self.device, dtype=self.dtype)

        result = self.camera.image_points_to_world_rays_static_pose(
            image_points=image_points,
            pose=self.static_pose,
            return_T_sensor_worlds=True,
        )

        self.assertEqual(result.world_rays.shape, (2, 6))
        self.assertIsNotNone(result.T_sensor_worlds)

    def test_image_points_to_world_rays_shutter_pose(self):
        """Test back-projection with rolling shutter pose."""
        image_points = torch.tensor([[320.0, 240.0], [320.0, 300.0]], device=self.device, dtype=self.dtype)

        result = self.camera.image_points_to_world_rays_shutter_pose(
            image_points=image_points,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=1000,
            return_timestamps=True,
        )

        self.assertEqual(result.world_rays.shape, (2, 6))
        self.assertIsNotNone(result.timestamps_us)

    def test_pixels_to_image_points(self):
        """Test pixel to image point conversion."""
        pixels = torch.tensor([[100, 200], [300, 400]], device=self.device, dtype=torch.int32)

        image_points = self.camera.pixels_to_image_points(pixels)

        expected = torch.tensor([[100.5, 200.5], [300.5, 400.5]], device=self.device, dtype=self.dtype)
        torch.testing.assert_close(image_points, expected)

    def test_image_points_to_pixels(self):
        """Test image point to pixel conversion."""
        image_points = torch.tensor([[100.7, 200.3], [300.9, 400.1]], device=self.device, dtype=self.dtype)

        pixels = self.camera.image_points_to_pixels(image_points)

        expected = torch.tensor([[100, 200], [300, 400]], device=self.device, dtype=torch.int32)
        torch.testing.assert_close(pixels, expected)

    def test_camera_rays_normalized(self):
        """Test that camera rays from image points are normalized."""
        image_points = torch.tensor(
            [[320.0, 240.0], [350.0, 260.0], [310.0, 230.0]], device=self.device, dtype=self.dtype
        )

        camera_rays = self.camera.image_points_to_camera_rays(image_points)

        norms = torch.norm(camera_rays, dim=1)
        torch.testing.assert_close(norms, torch.ones(3, device=self.device), atol=1e-5, rtol=1e-5)

    def test_image_points_relative_frame_times_global_shutter(self):
        """Test relative frame times for global shutter."""
        image_points = torch.tensor([[100.0, 200.0], [300.0, 400.0]], device=self.device, dtype=self.dtype)

        times = self.camera.image_points_relative_frame_times(image_points)

        expected = torch.zeros(2, device=self.device, dtype=self.dtype)
        torch.testing.assert_close(times, expected)


if __name__ == "__main__":
    unittest.main()
