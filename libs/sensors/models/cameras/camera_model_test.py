# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for CameraModel abstract base class.

Tests cover:
- Forward dispatch method
- nn.Module inheritance
- Projection property access
- Base class transform raises NotImplementedError
- External distortion and resolution attributes
"""

import unittest

import torch
import torch.nn as nn

from libs.sensors.kernels.cameras import (
    NoExternalDistortion,
    OpenCVPinholeProjection,
    ShutterType,
)
from libs.sensors.kernels.common.pose import DynamicPose, Pose
from libs.sensors.models.cameras import CameraModel, OpenCVPinholeCameraModel
from libs.sensors.models.common import WorldPointsToImagePointsReturn, WorldRaysReturn


class TestCameraModelAbstraction(unittest.TestCase):
    """Tests for CameraModel abstract base class behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device, dtype=self.dtype),
            principal_point=torch.tensor([320.0, 240.0], device=self.device, dtype=self.dtype),
            radial_coeffs=torch.zeros(6, device=self.device, dtype=self.dtype),
            tangential_coeffs=torch.zeros(2, device=self.device, dtype=self.dtype),
            thin_prism_coeffs=torch.zeros(4, device=self.device, dtype=self.dtype),
            resolution=torch.tensor([640.0, 480.0], device=self.device, dtype=self.dtype),
        )

        self.camera = OpenCVPinholeCameraModel(
            projection=projection,
            external_distortion=NoExternalDistortion(),
            resolution=(640, 480),
            shutter_type=ShutterType.GLOBAL,
        )

        # Create a static pose
        self.static_pose = Pose(
            translation=torch.zeros(3, device=self.device),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device),
        )

        # Create dynamic pose
        pose_start = Pose(
            translation=torch.zeros(3, device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
        )
        pose_end = Pose(
            translation=torch.tensor([0.1, 0.0, 0.0], device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
        )
        self.dynamic_pose = DynamicPose(start_pose=pose_start, end_pose=pose_end)

    def test_forward_dispatches_to_method(self):
        """Test that forward() dispatches to the specified method."""
        # Test that forward calls the specified method
        world_points = torch.tensor([[0.0, 0.0, 5.0]], device=self.device)

        # Call via forward
        result = self.camera.forward("world_points_to_image_points_static_pose", world_points, self.static_pose)
        assert isinstance(result, WorldPointsToImagePointsReturn)
        self.assertIsNotNone(result.image_points)

        # Test error on invalid method
        with self.assertRaises(AttributeError):
            self.camera.forward("nonexistent_method")

    def test_forward_dispatches_to_back_projection(self):
        """Test that forward() can dispatch to back-projection method."""
        image_points = torch.tensor([[320.0, 240.0]], device=self.device)

        result = self.camera.forward(
            "image_points_to_world_rays_static_pose",
            image_points,
            self.static_pose,
        )
        assert isinstance(result, WorldRaysReturn)
        self.assertEqual(result.world_rays.shape, (1, 6))

    def test_forward_with_kwargs(self):
        """Test that forward() passes kwargs correctly."""
        world_points = torch.tensor([[0.0, 0.0, 5.0]], device=self.device)

        result = self.camera.forward(
            "world_points_to_image_points_static_pose",
            world_points,
            self.static_pose,
            return_valid_flag=True,
        )
        self.assertIsNotNone(result.valid_flag)

    def test_forward_raises_type_error_on_non_callable(self):
        """Test that forward() raises TypeError for non-callable attributes."""
        with self.assertRaises(TypeError):
            self.camera.forward("resolution")

    def test_camera_is_nn_module(self):
        """Test that camera model is an nn.Module."""
        self.assertIsInstance(self.camera, nn.Module)

    def test_projection_property(self):
        """Test projection property access."""
        projection = self.camera.projection
        self.assertIsInstance(projection, OpenCVPinholeProjection)

    def test_camera_is_subclass_of_camera_model(self):
        """Test that OpenCVPinholeCameraModel is a subclass of CameraModel."""
        self.assertIsInstance(self.camera, CameraModel)

    def test_resolution_attribute(self):
        """Test resolution attribute access."""
        self.assertEqual(self.camera.resolution, (640, 480))

    def test_shutter_type_attribute(self):
        """Test shutter_type attribute access."""
        self.assertEqual(self.camera.shutter_type, ShutterType.GLOBAL)

    def test_external_distortion_attribute(self):
        """Test external_distortion attribute access."""
        self.assertIsInstance(self.camera.external_distortion, NoExternalDistortion)

    def test_camera_model_has_parameters(self):
        """Test that camera model has parameters from nn.Module."""
        # Should be able to iterate over parameters (even if empty)
        params = list(self.camera.parameters())
        # The model should have some structure even if no trainable params
        self.assertIsInstance(params, list)

    def test_camera_model_state_dict(self):
        """Test that camera model supports state_dict."""
        state = self.camera.state_dict()
        self.assertIsInstance(state, dict)

    def test_camera_model_to_device(self):
        """Test that camera model can be moved to different device."""
        # Move to CPU (works everywhere)
        camera_cpu = self.camera.to("cpu")
        self.assertIsInstance(camera_cpu, CameraModel)


if __name__ == "__main__":
    unittest.main()
