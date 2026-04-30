# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for Layer 2 return type dataclasses.

Tests cover:
- ImagePointsReturn
- WorldPointsToImagePointsReturn
- WorldPointsToPixelsReturn
- WorldRaysReturn
- SensorAnglesReturn
- SensorRayReturn
- WorldPointsToSensorAnglesReturn
"""

import unittest

import torch

from libs.sensors.models.common.return_types import (
    ImagePointsReturn,
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)


class TestImagePointsReturn(unittest.TestCase):
    """Tests for ImagePointsReturn dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def test_basic_creation(self):
        """Test basic ImagePointsReturn creation."""
        image_points = torch.tensor([[320.0, 240.0], [400.0, 300.0]], device=self.device, dtype=self.dtype)
        valid_flag = torch.tensor([True, True], device=self.device, dtype=torch.bool)

        result = ImagePointsReturn(image_points=image_points, valid_flag=valid_flag)

        self.assertEqual(result.image_points.shape, (2, 2))
        self.assertEqual(result.valid_flag.shape, (2,))
        self.assertIsNone(result.jacobians)

    def test_with_jacobians(self):
        """Test ImagePointsReturn with Jacobians."""
        image_points = torch.tensor([[320.0, 240.0]], device=self.device, dtype=self.dtype)
        valid_flag = torch.tensor([True], device=self.device, dtype=torch.bool)
        jacobians = torch.randn(1, 2, 3, device=self.device, dtype=self.dtype)

        result = ImagePointsReturn(image_points=image_points, valid_flag=valid_flag, jacobians=jacobians)

        self.assertIsNotNone(result.jacobians)
        assert result.jacobians is not None
        self.assertEqual(result.jacobians.shape, (1, 2, 3))


class TestWorldPointsToImagePointsReturn(unittest.TestCase):
    """Tests for WorldPointsToImagePointsReturn dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def test_basic_creation(self):
        """Test basic WorldPointsToImagePointsReturn creation."""
        image_points = torch.tensor([[320.0, 240.0]], device=self.device, dtype=self.dtype)

        result = WorldPointsToImagePointsReturn(image_points=image_points)

        self.assertEqual(result.image_points.shape, (1, 2))
        self.assertIsNone(result.T_world_sensors)
        self.assertIsNone(result.valid_flag)
        self.assertIsNone(result.valid_indices)
        self.assertIsNone(result.timestamps_us)

    def test_full_creation(self):
        """Test WorldPointsToImagePointsReturn with all fields."""
        N = 3
        image_points = torch.randn(N, 2, device=self.device, dtype=self.dtype)
        T_world_sensors = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).expand(N, -1, -1)
        valid_flag = torch.tensor([True, True, False], device=self.device, dtype=torch.bool)
        valid_indices = torch.tensor([0, 1], device=self.device, dtype=torch.int64)
        timestamps_us = torch.tensor([0, 500, 1000], device=self.device, dtype=torch.int64)

        result = WorldPointsToImagePointsReturn(
            image_points=image_points,
            T_world_sensors=T_world_sensors,
            valid_flag=valid_flag,
            valid_indices=valid_indices,
            timestamps_us=timestamps_us,
        )

        self.assertEqual(result.image_points.shape, (N, 2))
        assert result.T_world_sensors is not None
        self.assertEqual(result.T_world_sensors.shape, (N, 4, 4))
        assert result.valid_flag is not None
        self.assertEqual(result.valid_flag.shape, (N,))
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.shape, (2,))
        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape, (N,))


class TestWorldPointsToPixelsReturn(unittest.TestCase):
    """Tests for WorldPointsToPixelsReturn dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_basic_creation(self):
        """Test basic WorldPointsToPixelsReturn creation."""
        pixels = torch.tensor([[320, 240], [400, 300]], device=self.device, dtype=torch.int32)

        result = WorldPointsToPixelsReturn(pixels=pixels)

        self.assertEqual(result.pixels.shape, (2, 2))
        self.assertEqual(result.pixels.dtype, torch.int32)

    def test_full_creation(self):
        """Test WorldPointsToPixelsReturn with all fields."""
        N = 2
        pixels = torch.tensor([[320, 240], [400, 300]], device=self.device, dtype=torch.int32)
        T_world_sensors = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)
        valid_flag = torch.tensor([True, True], device=self.device, dtype=torch.bool)

        result = WorldPointsToPixelsReturn(
            pixels=pixels,
            T_world_sensors=T_world_sensors,
            valid_flag=valid_flag,
        )

        self.assertEqual(result.pixels.shape, (N, 2))
        assert result.T_world_sensors is not None
        self.assertEqual(result.T_world_sensors.shape, (N, 4, 4))
        assert result.valid_flag is not None
        self.assertEqual(result.valid_flag.shape, (N,))


class TestWorldRaysReturn(unittest.TestCase):
    """Tests for WorldRaysReturn dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def test_basic_creation(self):
        """Test basic WorldRaysReturn creation."""
        world_rays = torch.randn(5, 6, device=self.device, dtype=self.dtype)

        result = WorldRaysReturn(world_rays=world_rays)

        self.assertEqual(result.world_rays.shape, (5, 6))
        self.assertIsNone(result.T_sensor_worlds)
        self.assertIsNone(result.timestamps_us)

    def test_full_creation(self):
        """Test WorldRaysReturn with all fields."""
        N = 5
        world_rays = torch.randn(N, 6, device=self.device, dtype=self.dtype)
        T_sensor_worlds = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).expand(N, -1, -1)
        timestamps_us = torch.arange(N, device=self.device, dtype=torch.int64) * 100

        result = WorldRaysReturn(
            world_rays=world_rays,
            T_sensor_worlds=T_sensor_worlds,
            timestamps_us=timestamps_us,
        )

        self.assertEqual(result.world_rays.shape, (N, 6))
        assert result.T_sensor_worlds is not None
        self.assertEqual(result.T_sensor_worlds.shape, (N, 4, 4))
        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape, (N,))


class TestSensorAnglesReturn(unittest.TestCase):
    """Tests for SensorAnglesReturn dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def test_basic_creation(self):
        """Test basic SensorAnglesReturn creation."""
        sensor_angles = torch.tensor([[0.1, 0.5], [-0.1, -0.5]], device=self.device, dtype=self.dtype)

        result = SensorAnglesReturn(sensor_angles=sensor_angles)

        self.assertEqual(result.sensor_angles.shape, (2, 2))
        self.assertIsNone(result.valid_flag)

    def test_with_valid_flag(self):
        """Test SensorAnglesReturn with valid flag."""
        sensor_angles = torch.tensor([[0.1, 0.5]], device=self.device, dtype=self.dtype)
        valid_flag = torch.tensor([True], device=self.device, dtype=torch.bool)

        result = SensorAnglesReturn(sensor_angles=sensor_angles, valid_flag=valid_flag)

        self.assertIsNotNone(result.valid_flag)
        assert result.valid_flag is not None
        self.assertEqual(result.valid_flag.shape, (1,))


class TestSensorRayReturn(unittest.TestCase):
    """Tests for SensorRayReturn dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def test_basic_creation(self):
        """Test basic SensorRayReturn creation."""
        sensor_rays = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=self.device, dtype=self.dtype)

        result = SensorRayReturn(sensor_rays=sensor_rays)

        self.assertEqual(result.sensor_rays.shape, (2, 3))
        self.assertIsNone(result.valid_flag)

    def test_with_valid_flag(self):
        """Test SensorRayReturn with valid flag."""
        sensor_rays = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=self.dtype)
        valid_flag = torch.tensor([True], device=self.device, dtype=torch.bool)

        result = SensorRayReturn(sensor_rays=sensor_rays, valid_flag=valid_flag)

        self.assertIsNotNone(result.valid_flag)


class TestWorldPointsToSensorAnglesReturn(unittest.TestCase):
    """Tests for WorldPointsToSensorAnglesReturn dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def test_basic_creation(self):
        """Test basic WorldPointsToSensorAnglesReturn creation."""
        sensor_angles = torch.tensor([[0.1, 0.5]], device=self.device, dtype=self.dtype)

        result = WorldPointsToSensorAnglesReturn(sensor_angles=sensor_angles)

        self.assertEqual(result.sensor_angles.shape, (1, 2))
        self.assertIsNone(result.T_world_sensors)
        self.assertIsNone(result.valid_flag)
        self.assertIsNone(result.valid_indices)
        self.assertIsNone(result.timestamps_us)

    def test_full_creation(self):
        """Test WorldPointsToSensorAnglesReturn with all fields."""
        N = 3
        sensor_angles = torch.randn(N, 2, device=self.device, dtype=self.dtype)
        T_world_sensors = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).expand(N, -1, -1)
        valid_flag = torch.tensor([True, True, False], device=self.device, dtype=torch.bool)
        valid_indices = torch.tensor([0, 1], device=self.device, dtype=torch.int64)
        timestamps_us = torch.tensor([0, 500, 1000], device=self.device, dtype=torch.int64)

        result = WorldPointsToSensorAnglesReturn(
            sensor_angles=sensor_angles,
            T_world_sensors=T_world_sensors,
            valid_flag=valid_flag,
            valid_indices=valid_indices,
            timestamps_us=timestamps_us,
        )

        self.assertEqual(result.sensor_angles.shape, (N, 2))
        assert result.T_world_sensors is not None
        self.assertEqual(result.T_world_sensors.shape, (N, 4, 4))
        assert result.valid_flag is not None
        self.assertEqual(result.valid_flag.shape, (N,))
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.shape, (2,))
        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape, (N,))


if __name__ == "__main__":
    unittest.main()
