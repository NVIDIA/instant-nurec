# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for OpenCVPinholeCameraModel.

Tests cover:
- Model initialization
- Projection/back-projection consistency
- Rolling shutter handling
- Transform method
- Return type structure
"""

import unittest

import torch

from libs.sensors.kernels.cameras import (
    NoExternalDistortion,
    OpenCVPinholeProjection,
    ShutterType,
)
from libs.sensors.kernels.common.pose import DynamicPose, Pose
from libs.sensors.models.cameras import OpenCVPinholeCameraModel
from libs.sensors.models.common import WorldPointsToImagePointsReturn


class TestOpenCVPinholeCameraModel(unittest.TestCase):
    """Tests for OpenCVPinholeCameraModel."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.resolution = (640, 480)

        # Create a simple pinhole projection (no distortion)
        self.projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device, dtype=self.dtype),
            principal_point=torch.tensor([320.0, 240.0], device=self.device, dtype=self.dtype),
            radial_coeffs=torch.zeros(6, device=self.device, dtype=self.dtype),
            tangential_coeffs=torch.zeros(2, device=self.device, dtype=self.dtype),
            thin_prism_coeffs=torch.zeros(4, device=self.device, dtype=self.dtype),
            resolution=torch.tensor([640.0, 480.0], device=self.device, dtype=self.dtype),
        )

        self.camera = OpenCVPinholeCameraModel(
            projection=self.projection,
            external_distortion=NoExternalDistortion(),
            resolution=self.resolution,
            shutter_type=ShutterType.GLOBAL,
        )

        # Create a static pose (identity)
        self.static_pose = Pose(
            translation=torch.zeros(3, device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),  # identity quat
        )

        # Create a dynamic pose (start = identity, end = small translation)
        pose_start = Pose(
            translation=torch.zeros(3, device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
        )
        pose_end = Pose(
            translation=torch.tensor([0.1, 0.0, 0.0], device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
        )
        self.dynamic_pose = DynamicPose(start_pose=pose_start, end_pose=pose_end)

    def test_camera_model_initialization(self):
        """Test that camera model initializes correctly."""
        self.assertEqual(self.camera.resolution, self.resolution)
        self.assertEqual(self.camera.shutter_type, ShutterType.GLOBAL)
        self.assertIsInstance(self.camera.projection, OpenCVPinholeProjection)

    def test_camera_rays_to_image_points_basic(self):
        """Test basic camera ray to image point projection."""
        # Ray pointing straight ahead (should project to principal point)
        camera_rays = torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=self.dtype)

        result = self.camera.camera_rays_to_image_points(camera_rays)

        self.assertEqual(result.image_points.shape, (1, 2))
        self.assertIsNotNone(result.valid_flag)
        # Should project near principal point
        self.assertAlmostEqual(result.image_points[0, 0].item(), 320.0, delta=1.0)
        self.assertAlmostEqual(result.image_points[0, 1].item(), 240.0, delta=1.0)

    def test_camera_rays_to_image_points_with_jacobians(self):
        """Test camera ray projection with Jacobian computation."""
        camera_rays = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]], device=self.device, dtype=self.dtype)

        result = self.camera.camera_rays_to_image_points(camera_rays, return_jacobians=True)

        self.assertIsNotNone(result.jacobians)
        assert result.jacobians is not None
        self.assertEqual(result.jacobians.shape, (2, 2, 3))  # (N, 2, 3)

    def test_image_points_to_camera_rays_roundtrip(self):
        """Test that projecting and back-projecting gives consistent results."""
        # Start with some image points
        image_points = torch.tensor(
            [[320.0, 240.0], [400.0, 300.0], [200.0, 100.0]], device=self.device, dtype=self.dtype
        )

        # Convert to camera rays
        camera_rays = self.camera.image_points_to_camera_rays(image_points)

        # Project back to image points
        result = self.camera.camera_rays_to_image_points(camera_rays)

        # Should get back approximately the same image points
        torch.testing.assert_close(result.image_points, image_points, atol=1e-3, rtol=1e-3)

    def test_world_points_to_image_points_static_pose(self):
        """Test world point projection with static pose."""
        # World points in front of camera
        world_points = torch.tensor(
            [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [-1.0, 1.0, 10.0]], device=self.device, dtype=self.dtype
        )

        result = self.camera.world_points_to_image_points_static_pose(
            world_points=world_points,
            pose=self.static_pose,
            return_valid_flag=True,
            return_T_world_sensors=True,
        )

        self.assertEqual(result.image_points.shape[1], 2)
        self.assertIsNotNone(result.valid_flag)
        assert result.T_world_sensors is not None
        self.assertIsNotNone(result.T_world_sensors)
        assert result.T_world_sensors is not None
        self.assertEqual(result.T_world_sensors.shape[-2:], (4, 4))

    def test_world_points_to_image_points_return_all_projections(self):
        """Test return_all_projections parameter."""
        # Some points in front, some behind camera
        world_points = torch.tensor(
            [[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [0.0, 0.0, 3.0]], device=self.device, dtype=self.dtype
        )

        result_filtered = self.camera.world_points_to_image_points_static_pose(
            world_points=world_points,
            pose=self.static_pose,
            return_valid_flag=True,
            return_all_projections=False,
        )

        result_all = self.camera.world_points_to_image_points_static_pose(
            world_points=world_points,
            pose=self.static_pose,
            return_valid_flag=True,
            return_all_projections=True,
        )

        # All projections should return same number as input
        self.assertEqual(result_all.image_points.shape[0], 3)
        # Filtered should return fewer (only valid)
        self.assertLessEqual(result_filtered.image_points.shape[0], 3)

    def test_world_points_to_image_points_return_valid_indices(self):
        """Test return_valid_indices parameter."""
        world_points = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 3.0]], device=self.device, dtype=self.dtype)

        result = self.camera.world_points_to_image_points_static_pose(
            world_points=world_points,
            pose=self.static_pose,
            return_valid_indices=True,
            return_all_projections=True,
        )

        self.assertIsNotNone(result.valid_indices)
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.dtype, torch.int64)

    def test_world_points_to_pixels_static_pose(self):
        """Test world point to pixel projection."""
        world_points = torch.tensor([[0.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)

        result = self.camera.world_points_to_pixels_static_pose(
            world_points=world_points, pose=self.static_pose, return_valid_flag=True
        )

        self.assertEqual(result.pixels.dtype, torch.int32)
        self.assertEqual(result.pixels.shape, (1, 2))

    def test_image_points_to_world_rays_static_pose(self):
        """Test back-projection to world rays with static pose."""
        image_points = torch.tensor([[320.0, 240.0], [400.0, 300.0]], device=self.device, dtype=self.dtype)

        result = self.camera.image_points_to_world_rays_static_pose(
            image_points=image_points,
            pose=self.static_pose,
            return_T_sensor_worlds=True,
        )

        self.assertEqual(result.world_rays.shape, (2, 6))  # (N, 6) = [origin, direction]
        assert result.T_sensor_worlds is not None
        self.assertIsNotNone(result.T_sensor_worlds)
        assert result.T_sensor_worlds is not None
        self.assertEqual(result.T_sensor_worlds.shape, (2, 4, 4))

    def test_image_points_to_world_rays_with_camera_rays_reuse(self):
        """Test world ray generation with pre-computed camera rays."""
        image_points = torch.tensor([[320.0, 240.0], [400.0, 300.0]], device=self.device, dtype=self.dtype)

        # Pre-compute camera rays
        camera_rays = self.camera.image_points_to_camera_rays(image_points)

        # Use standard path
        result_standard = self.camera.image_points_to_world_rays_static_pose(
            image_points=image_points, pose=self.static_pose
        )

        # Use camera_rays reuse path
        result_reuse = self.camera.image_points_to_world_rays_static_pose(
            image_points=image_points, pose=self.static_pose, camera_rays=camera_rays
        )

        # Results should be the same
        torch.testing.assert_close(result_standard.world_rays, result_reuse.world_rays, atol=1e-5, rtol=1e-5)

    def test_image_points_to_world_rays_mean_pose(self):
        """Test back-projection with mean pose."""
        image_points = torch.tensor([[320.0, 240.0]], device=self.device, dtype=self.dtype)

        result = self.camera.image_points_to_world_rays_mean_pose(
            image_points=image_points,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=1000,
            return_timestamps=True,
        )

        self.assertEqual(result.world_rays.shape, (1, 6))
        self.assertIsNotNone(result.timestamps_us)

    def test_image_points_to_world_rays_shutter_pose(self):
        """Test back-projection with rolling shutter pose."""
        image_points = torch.tensor([[320.0, 240.0], [320.0, 400.0]], device=self.device, dtype=self.dtype)

        result = self.camera.image_points_to_world_rays_shutter_pose(
            image_points=image_points,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=1000,
            return_T_sensor_worlds=True,
            return_timestamps=True,
        )

        self.assertEqual(result.world_rays.shape, (2, 6))
        self.assertIsNotNone(result.T_sensor_worlds)
        self.assertIsNotNone(result.timestamps_us)

    def test_pixels_to_image_points(self):
        """Test pixel to image point conversion."""
        pixels = torch.tensor([[100, 200], [300, 400]], device=self.device, dtype=torch.int32)

        image_points = self.camera.pixels_to_image_points(pixels)

        # Pixel center is at pixel + 0.5
        expected = torch.tensor([[100.5, 200.5], [300.5, 400.5]], device=self.device, dtype=self.dtype)
        torch.testing.assert_close(image_points, expected)

    def test_image_points_to_pixels(self):
        """Test image point to pixel conversion."""
        image_points = torch.tensor([[100.7, 200.3], [300.9, 400.1]], device=self.device, dtype=self.dtype)

        pixels = self.camera.image_points_to_pixels(image_points)

        # Floor operation
        expected = torch.tensor([[100, 200], [300, 400]], device=self.device, dtype=torch.int32)
        torch.testing.assert_close(pixels, expected)

    def test_image_points_relative_frame_times_global_shutter(self):
        """Test relative frame times for global shutter."""
        image_points = torch.tensor([[100.0, 200.0], [300.0, 400.0]], device=self.device, dtype=self.dtype)

        times = self.camera.image_points_relative_frame_times(image_points)

        # Global shutter should return 0 for all points
        expected = torch.zeros(2, device=self.device, dtype=self.dtype)
        torch.testing.assert_close(times, expected)

    def test_image_points_relative_frame_times_rolling_shutter(self):
        """Test relative frame times for rolling shutter."""
        # Create a camera with rolling shutter
        camera_rolling = OpenCVPinholeCameraModel(
            projection=self.projection,
            external_distortion=NoExternalDistortion(),
            resolution=self.resolution,
            shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        )

        image_points = torch.tensor(
            [[100.0, 0.0], [100.0, 239.5], [100.0, 479.0]], device=self.device, dtype=self.dtype
        )

        times = camera_rolling.image_points_relative_frame_times(image_points)

        # Times should be in [0, 1] and increase with y
        self.assertTrue(torch.all(times >= 0.0))
        self.assertTrue(torch.all(times <= 1.0))
        self.assertTrue(times[0] < times[1] < times[2])

    def test_transform_scale(self):
        """Test camera transform with scaling."""
        scaled_camera = self.camera.transform(image_domain_scale=0.5)

        # Resolution should be halved
        self.assertEqual(scaled_camera.resolution, (320, 240))

        # Focal length should be halved
        torch.testing.assert_close(
            scaled_camera.projection.focal_length, self.camera.projection.focal_length * 0.5, atol=1e-5, rtol=1e-5
        )

        # Principal point should be halved
        torch.testing.assert_close(
            scaled_camera.projection.principal_point, self.camera.projection.principal_point * 0.5, atol=1e-5, rtol=1e-5
        )

    def test_transform_with_offset(self):
        """Test camera transform with offset (cropping)."""
        scaled_camera = self.camera.transform(
            image_domain_scale=1.0, image_domain_offset=(100.0, 50.0), new_resolution=(440, 380)
        )

        # Resolution should be updated
        self.assertEqual(scaled_camera.resolution, (440, 380))

        # Principal point should be offset
        expected_pp = self.camera.projection.principal_point - torch.tensor(
            [100.0, 50.0], device=self.device, dtype=self.dtype
        )
        torch.testing.assert_close(scaled_camera.projection.principal_point, expected_pp, atol=1e-5, rtol=1e-5)

    def test_transform_anisotropic_scale(self):
        """Test camera transform with anisotropic scaling."""
        scaled_camera = self.camera.transform(image_domain_scale=(0.5, 0.25))

        # Resolution should be scaled anisotropically
        self.assertEqual(scaled_camera.resolution, (320, 120))

    def test_world_points_to_image_points_mean_pose(self):
        """Test world point projection with mean pose."""
        world_points = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)

        result = self.camera.world_points_to_image_points_mean_pose(
            world_points=world_points,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=1000,
            return_valid_flag=True,
            return_timestamps=True,
        )

        self.assertEqual(result.image_points.shape[1], 2)
        self.assertIsNotNone(result.valid_flag)
        self.assertIsNotNone(result.timestamps_us)

    def test_world_points_to_image_points_shutter_pose(self):
        """Test world point projection with rolling shutter pose."""
        world_points = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_points=world_points,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=1000,
            return_valid_flag=True,
            return_T_world_sensors=True,
            return_timestamps=True,
        )

        self.assertEqual(result.image_points.shape[1], 2)
        self.assertIsNotNone(result.valid_flag)
        self.assertIsNotNone(result.T_world_sensors)
        self.assertIsNotNone(result.timestamps_us)

    def test_world_points_to_pixels_mean_pose(self):
        """Test world point to pixel projection with mean pose."""
        world_points = torch.tensor([[0.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)

        result = self.camera.world_points_to_pixels_mean_pose(
            world_points=world_points,
            dynamic_pose=self.dynamic_pose,
            return_valid_flag=True,
        )

        self.assertEqual(result.pixels.dtype, torch.int32)
        self.assertEqual(result.pixels.shape, (1, 2))

    def test_world_points_to_pixels_shutter_pose(self):
        """Test world point to pixel projection with shutter pose."""
        world_points = torch.tensor([[0.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)

        result = self.camera.world_points_to_pixels_shutter_pose(
            world_points=world_points,
            dynamic_pose=self.dynamic_pose,
            return_valid_flag=True,
        )

        self.assertEqual(result.pixels.dtype, torch.int32)
        self.assertEqual(result.pixels.shape, (1, 2))

    def test_pixels_to_world_rays_static_pose(self):
        """Test pixel to world ray conversion with static pose."""
        pixels = torch.tensor([[320, 240], [400, 300]], device=self.device, dtype=torch.int32)

        result = self.camera.pixels_to_world_rays_static_pose(
            pixel_idxs=pixels,
            pose=self.static_pose,
            return_T_sensor_worlds=True,
        )

        self.assertEqual(result.world_rays.shape, (2, 6))
        self.assertIsNotNone(result.T_sensor_worlds)

    def test_pixels_to_world_rays_mean_pose(self):
        """Test pixel to world ray conversion with mean pose."""
        pixels = torch.tensor([[320, 240]], device=self.device, dtype=torch.int32)

        result = self.camera.pixels_to_world_rays_mean_pose(
            pixel_idxs=pixels,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=1000,
            return_timestamps=True,
        )

        self.assertEqual(result.world_rays.shape, (1, 6))
        self.assertIsNotNone(result.timestamps_us)

    def test_pixels_to_world_rays_shutter_pose(self):
        """Test pixel to world ray conversion with shutter pose."""
        pixels = torch.tensor([[320, 240], [320, 400]], device=self.device, dtype=torch.int32)

        result = self.camera.pixels_to_world_rays_shutter_pose(
            pixel_idxs=pixels,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=1000,
            return_timestamps=True,
        )

        self.assertEqual(result.world_rays.shape, (2, 6))
        self.assertIsNotNone(result.timestamps_us)

    def test_pixels_to_camera_rays(self):
        """Test pixel to camera ray conversion."""
        pixels = torch.tensor([[320, 240], [400, 300]], device=self.device, dtype=torch.int32)

        camera_rays = self.camera.pixels_to_camera_rays(pixels)

        self.assertEqual(camera_rays.shape, (2, 3))
        # Rays should be normalized
        norms = torch.norm(camera_rays, dim=1)
        torch.testing.assert_close(norms, torch.ones(2, device=self.device), atol=1e-5, rtol=1e-5)

    def test_camera_rays_to_pixels(self):
        """Test camera ray to pixel conversion."""
        # Ray pointing straight ahead
        camera_rays = torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=self.dtype)

        result = self.camera.camera_rays_to_pixels(camera_rays)

        self.assertEqual(result.pixels.shape, (1, 2))
        self.assertEqual(result.pixels.dtype, torch.int32)
        self.assertIsNotNone(result.valid_flag)

    def test_image_points_relative_frame_times_bottom_to_top(self):
        """Test relative frame times for bottom-to-top rolling shutter."""
        camera_rolling = OpenCVPinholeCameraModel(
            projection=self.projection,
            external_distortion=NoExternalDistortion(),
            resolution=self.resolution,
            shutter_type=ShutterType.ROLLING_BOTTOM_TO_TOP,
        )

        # Use points that stay within the valid y range for BOTTOM_TO_TOP
        # The formula uses ceil, so we need to be careful with edge cases
        image_points = torch.tensor(
            [[100.0, 1.0], [100.0, 240.0], [100.0, 478.0]], device=self.device, dtype=self.dtype
        )

        times = camera_rolling.image_points_relative_frame_times(image_points)

        # Times should be in approximately [0, 1] and decrease with y (bottom to top)
        self.assertTrue(torch.all(times >= 0.0))
        self.assertTrue(torch.all(times <= 1.01))  # Allow small numerical tolerance
        self.assertTrue(times[0] > times[1] > times[2])

    def test_image_points_relative_frame_times_left_to_right(self):
        """Test relative frame times for left-to-right rolling shutter."""
        camera_rolling = OpenCVPinholeCameraModel(
            projection=self.projection,
            external_distortion=NoExternalDistortion(),
            resolution=self.resolution,
            shutter_type=ShutterType.ROLLING_LEFT_TO_RIGHT,
        )

        image_points = torch.tensor(
            [[0.0, 100.0], [319.5, 100.0], [639.0, 100.0]], device=self.device, dtype=self.dtype
        )

        times = camera_rolling.image_points_relative_frame_times(image_points)

        # Times should increase with x
        self.assertTrue(torch.all(times >= 0.0))
        self.assertTrue(torch.all(times <= 1.0))
        self.assertTrue(times[0] < times[1] < times[2])

    def test_batch_projection(self):
        """Test projection with larger batch of points."""
        N = 100
        world_points = torch.rand(N, 3, device=self.device, dtype=self.dtype) * 10
        world_points[:, 2] += 2.0  # Ensure points are in front of camera

        result = self.camera.world_points_to_image_points_static_pose(
            world_points=world_points,
            pose=self.static_pose,
            return_valid_flag=True,
            return_all_projections=True,
        )

        self.assertEqual(result.image_points.shape[0], N)


if __name__ == "__main__":
    unittest.main()
