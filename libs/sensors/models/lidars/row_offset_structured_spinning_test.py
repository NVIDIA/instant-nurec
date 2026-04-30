# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for RowOffsetStructuredSpinningLidarModel.

Tests cover:
- Model initialization and properties
- Sensor ray/angle conversions
- Element to ray/angle mappings
- World ray generation with rolling shutter
- Inverse projection with rolling shutter
- FOV validation
- numpy input support
- angles_to_columns_map handling
"""

import unittest

import numpy as np
import torch

from libs.sensors.kernels.common.pose import DynamicPose, Pose
from libs.sensors.kernels.lidars import RowOffsetStructuredSpinningLidarProjection
from libs.sensors.models.lidars import RowOffsetStructuredSpinningLidarModel


class TestRowOffsetStructuredSpinningLidarModelInitialization(unittest.TestCase):
    """Tests for RowOffsetStructuredSpinningLidarModel initialization and properties."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        # Create elevation angles (typical for spinning LiDAR)
        self.row_elevations_rad = torch.linspace(
            0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype
        )  # ~15 deg to -15 deg

        # Create azimuth angles (full 360 deg rotation)
        self.column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )

        # Small azimuth offsets per row
        self.row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.row_elevations_rad,
            column_azimuths_rad=self.column_azimuths_rad,
            row_azimuth_offsets_rad=self.row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
        )

    def test_basic_initialization(self):
        """Test basic RowOffsetStructuredSpinningLidarModel initialization."""
        lidar = RowOffsetStructuredSpinningLidarModel(projection=self.projection)

        self.assertEqual(lidar.n_rows, self.n_rows)
        self.assertEqual(lidar.n_columns, self.n_columns)
        self.assertEqual(lidar.n_elements, self.n_rows * self.n_columns)

    def test_initialization_with_map_init(self):
        """Test initialization with eager angles-to-columns map building."""
        lidar = RowOffsetStructuredSpinningLidarModel(projection=self.projection, angles_to_columns_map_init=True)

        # Map should be built
        self.assertIsNotNone(lidar.projection.angles_to_columns_map)

    def test_fov_properties(self):
        """Test FOV property accessors."""
        lidar = RowOffsetStructuredSpinningLidarModel(projection=self.projection)

        fov_vert = lidar.fov_vert
        fov_horiz = lidar.fov_horiz

        self.assertEqual(fov_vert, (0.26, 0.52))
        self.assertAlmostEqual(fov_horiz[0], -torch.pi, places=5)
        self.assertAlmostEqual(fov_horiz[1], 2 * torch.pi, places=5)

    def test_spinning_properties(self):
        """Test spinning property accessors."""
        lidar = RowOffsetStructuredSpinningLidarModel(projection=self.projection)

        self.assertEqual(lidar.spinning_frequency_hz, 10.0)
        self.assertEqual(lidar.spinning_direction, "cw")

    def test_projection_property(self):
        """Test projection property access."""
        lidar = RowOffsetStructuredSpinningLidarModel(projection=self.projection)

        self.assertIsInstance(lidar.projection, RowOffsetStructuredSpinningLidarProjection)


class TestRowOffsetStructuredSpinningLidarModelSensorConversions(unittest.TestCase):
    """Tests for sensor ray/angle conversions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )
        row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
        )

        self.lidar = RowOffsetStructuredSpinningLidarModel(projection=projection)

    def test_sensor_rays_to_sensor_angles_basic(self):
        """Test basic sensor ray to angle conversion."""
        # Ray pointing forward
        sensor_rays = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=self.dtype)

        result = self.lidar.sensor_rays_to_sensor_angles(sensor_rays)

        self.assertEqual(result.sensor_angles.shape, (1, 2))
        # Forward ray should have elevation ~0 and azimuth ~0
        self.assertAlmostEqual(result.sensor_angles[0, 0].item(), 0.0, places=3)  # elevation
        self.assertAlmostEqual(result.sensor_angles[0, 1].item(), 0.0, places=3)  # azimuth

    def test_sensor_rays_to_sensor_angles_normalized_false(self):
        """Test sensor ray conversion with normalization."""
        # Non-unit ray
        sensor_rays = torch.tensor([[2.0, 0.0, 0.0]], device=self.device, dtype=self.dtype)

        result = self.lidar.sensor_rays_to_sensor_angles(sensor_rays, normalized=False)

        self.assertEqual(result.sensor_angles.shape, (1, 2))
        # Should still get correct angles after normalization
        self.assertAlmostEqual(result.sensor_angles[0, 0].item(), 0.0, places=3)

    def test_sensor_rays_to_sensor_angles_with_valid_flag(self):
        """Test sensor ray conversion with validity flag."""
        sensor_rays = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=self.device, dtype=self.dtype)

        result = self.lidar.sensor_rays_to_sensor_angles(sensor_rays, return_valid_flag=True)

        self.assertIsNotNone(result.valid_flag)
        assert result.valid_flag is not None
        self.assertEqual(result.valid_flag.shape, (2,))

    def test_sensor_angles_to_sensor_rays_basic(self):
        """Test basic sensor angle to ray conversion."""
        # Angles: elevation=0, azimuth=0 -> forward ray
        sensor_angles = torch.tensor([[0.0, 0.0]], device=self.device, dtype=self.dtype)

        result = self.lidar.sensor_angles_to_sensor_rays(sensor_angles)

        self.assertEqual(result.sensor_rays.shape, (1, 3))
        # Should point along x-axis
        torch.testing.assert_close(
            result.sensor_rays[0], torch.tensor([1.0, 0.0, 0.0], device=self.device), atol=1e-5, rtol=1e-5
        )

    def test_sensor_rays_angles_roundtrip(self):
        """Test roundtrip conversion between rays and angles."""
        # Start with some angles
        sensor_angles = torch.tensor(
            [[0.1, 0.5], [-0.1, -0.5], [0.0, torch.pi / 2]], device=self.device, dtype=self.dtype
        )

        # Convert to rays
        rays_result = self.lidar.sensor_angles_to_sensor_rays(sensor_angles)

        # Convert back to angles
        angles_result = self.lidar.sensor_rays_to_sensor_angles(rays_result.sensor_rays)

        # Should get back original angles
        torch.testing.assert_close(angles_result.sensor_angles, sensor_angles, atol=1e-5, rtol=1e-5)

    def test_elements_to_sensor_angles(self):
        """Test element to sensor angle conversion."""
        elements = torch.tensor([[0, 0], [8, 180], [15, 359]], device=self.device, dtype=torch.long)

        result = self.lidar.elements_to_sensor_angles(elements)

        self.assertEqual(result.sensor_angles.shape, (3, 2))

    def test_elements_to_sensor_angles_with_valid_flag(self):
        """Test element conversion with bounds checking."""
        # Some valid, some invalid elements
        elements = torch.tensor([[0, 0], [100, 0], [0, 1000]], device=self.device, dtype=torch.long)

        result = self.lidar.elements_to_sensor_angles(elements, return_valid_flag=True)

        self.assertIsNotNone(result.valid_flag)
        assert result.valid_flag is not None
        self.assertTrue(result.valid_flag[0].item())  # Valid
        self.assertFalse(result.valid_flag[1].item())  # Invalid row
        self.assertFalse(result.valid_flag[2].item())  # Invalid column

    def test_elements_to_sensor_rays(self):
        """Test element to sensor ray conversion."""
        elements = torch.tensor([[0, 0], [8, 180]], device=self.device, dtype=torch.long)

        sensor_rays = self.lidar.elements_to_sensor_rays(elements)

        self.assertEqual(sensor_rays.shape, (2, 3))
        # Rays should be normalized
        norms = torch.norm(sensor_rays, dim=1)
        torch.testing.assert_close(norms, torch.ones(2, device=self.device), atol=1e-5, rtol=1e-5)

    def test_elements_to_sensor_points(self):
        """Test element to sensor point conversion."""
        elements = torch.tensor([[0, 0], [8, 180]], device=self.device, dtype=torch.long)
        distances = torch.tensor([5.0, 10.0], device=self.device, dtype=self.dtype)

        points = self.lidar.elements_to_sensor_points(elements, distances)

        self.assertEqual(points.shape, (2, 3))
        # Points should be at specified distances
        point_distances = torch.norm(points, dim=1)
        torch.testing.assert_close(point_distances, distances, atol=1e-5, rtol=1e-5)


class TestRowOffsetStructuredSpinningLidarModelNumpySupport(unittest.TestCase):
    """Tests for numpy input support."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )
        row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
        )

        self.lidar = RowOffsetStructuredSpinningLidarModel(projection=projection)

    def test_sensor_rays_to_sensor_angles_numpy(self):
        """Test sensor ray conversion with numpy input."""
        sensor_rays_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        result = self.lidar.sensor_rays_to_sensor_angles(sensor_rays_np)

        self.assertIsInstance(result.sensor_angles, torch.Tensor)
        self.assertEqual(result.sensor_angles.shape, (2, 2))

    def test_sensor_angles_to_sensor_rays_numpy(self):
        """Test sensor angle conversion with numpy input."""
        sensor_angles_np = np.array([[0.0, 0.0], [0.1, 0.5]], dtype=np.float32)

        result = self.lidar.sensor_angles_to_sensor_rays(sensor_angles_np)

        self.assertIsInstance(result.sensor_rays, torch.Tensor)
        self.assertEqual(result.sensor_rays.shape, (2, 3))

    def test_elements_to_sensor_angles_numpy(self):
        """Test element conversion with numpy input."""
        elements_np = np.array([[0, 0], [8, 180]], dtype=np.int64)

        result = self.lidar.elements_to_sensor_angles(elements_np)

        self.assertIsInstance(result.sensor_angles, torch.Tensor)
        self.assertEqual(result.sensor_angles.shape, (2, 2))

    def test_elements_to_sensor_points_numpy(self):
        """Test element to point conversion with numpy inputs."""
        elements_np = np.array([[0, 0], [8, 180]], dtype=np.int64)
        distances_np = np.array([5.0, 10.0], dtype=np.float32)

        points = self.lidar.elements_to_sensor_points(elements_np, distances_np)

        self.assertIsInstance(points, torch.Tensor)
        self.assertEqual(points.shape, (2, 3))


class TestRowOffsetStructuredSpinningLidarModelWorldOperations(unittest.TestCase):
    """Tests for world ray generation and inverse projection."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )
        row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
        )

        self.lidar = RowOffsetStructuredSpinningLidarModel(projection=projection)

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

    def test_elements_to_world_rays_shutter_pose_basic(self):
        """Test basic world ray generation with rolling shutter."""
        elements = torch.tensor([[0, 0], [8, 180], [15, 359]], device=self.device, dtype=torch.long)

        result = self.lidar.elements_to_world_rays_shutter_pose(
            elements=elements,
            dynamic_pose=self.dynamic_pose,
        )

        self.assertEqual(result.world_rays.shape, (3, 6))  # (N, 6) = [origin, direction]

    def test_elements_to_world_rays_generate_all(self):
        """Test world ray generation for all elements."""
        result = self.lidar.elements_to_world_rays_shutter_pose(
            elements=None,  # Generate all
            dynamic_pose=self.dynamic_pose,
        )

        expected_n = self.n_rows * self.n_columns
        self.assertEqual(result.world_rays.shape, (expected_n, 6))

    def test_elements_to_world_rays_with_poses(self):
        """Test world ray generation with pose return."""
        elements = torch.tensor([[0, 0], [8, 180]], device=self.device, dtype=torch.long)

        result = self.lidar.elements_to_world_rays_shutter_pose(
            elements=elements,
            dynamic_pose=self.dynamic_pose,
            return_T_sensor_worlds=True,
        )

        self.assertIsNotNone(result.T_sensor_worlds)
        assert result.T_sensor_worlds is not None
        self.assertEqual(result.T_sensor_worlds.shape, (2, 4, 4))

    def test_elements_to_world_rays_with_timestamps(self):
        """Test world ray generation with timestamp return."""
        elements = torch.tensor([[0, 0], [8, 180]], device=self.device, dtype=torch.long)

        result = self.lidar.elements_to_world_rays_shutter_pose(
            elements=elements,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=100000,
            return_timestamps=True,
        )

        self.assertIsNotNone(result.timestamps_us)
        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape, (2,))
        # Timestamps should be in range [0, 100000]
        self.assertTrue(torch.all(result.timestamps_us >= 0))
        self.assertTrue(torch.all(result.timestamps_us <= 100000))

    def test_elements_to_world_rays_with_sensor_rays_reuse(self):
        """Test world ray generation with pre-computed sensor rays."""
        elements = torch.tensor([[0, 0], [8, 180]], device=self.device, dtype=torch.long)

        # Pre-compute sensor rays
        sensor_rays = self.lidar.elements_to_sensor_rays(elements)

        # Generate world rays with standard path
        result_standard = self.lidar.elements_to_world_rays_shutter_pose(
            elements=elements,
            dynamic_pose=self.dynamic_pose,
        )

        # Generate world rays with sensor_rays reuse
        result_reuse = self.lidar.elements_to_world_rays_shutter_pose(
            elements=elements,
            dynamic_pose=self.dynamic_pose,
            sensor_rays=sensor_rays,
        )

        # Results should be approximately equal
        torch.testing.assert_close(result_standard.world_rays, result_reuse.world_rays, atol=1e-4, rtol=1e-4)

    def test_world_points_to_sensor_angles_shutter_pose_basic(self):
        """Test basic inverse projection with rolling shutter."""
        # World points in front of sensor
        world_points = torch.tensor(
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [3.0, 3.0, 0.1]], device=self.device, dtype=self.dtype
        )

        result = self.lidar.world_points_to_sensor_angles_shutter_pose(
            world_points=world_points,
            dynamic_pose=self.dynamic_pose,
            lazy_init_angles_map=True,
            return_valid_flag=True,
        )

        self.assertEqual(result.sensor_angles.shape, (3, 2))
        self.assertIsNotNone(result.valid_flag)

    def test_world_points_to_sensor_angles_with_poses(self):
        """Test inverse projection with pose return."""
        world_points = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], device=self.device, dtype=self.dtype)

        result = self.lidar.world_points_to_sensor_angles_shutter_pose(
            world_points=world_points,
            dynamic_pose=self.dynamic_pose,
            lazy_init_angles_map=True,
            return_T_world_sensors=True,
        )

        self.assertIsNotNone(result.T_world_sensors)
        assert result.T_world_sensors is not None
        self.assertEqual(result.T_world_sensors.shape, (2, 4, 4))

    def test_world_points_to_sensor_angles_with_valid_indices(self):
        """Test inverse projection with valid indices return."""
        world_points = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], device=self.device, dtype=self.dtype)

        result = self.lidar.world_points_to_sensor_angles_shutter_pose(
            world_points=world_points,
            dynamic_pose=self.dynamic_pose,
            lazy_init_angles_map=True,
            return_valid_indices=True,
        )

        self.assertIsNotNone(result.valid_indices)
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.dtype, torch.int64)


class TestRowOffsetStructuredSpinningLidarModelFOVValidation(unittest.TestCase):
    """Tests for FOV validation methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )
        row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
        )

        self.lidar = RowOffsetStructuredSpinningLidarModel(projection=projection)

    def test_valid_sensor_angles_within_fov(self):
        """Test validation for angles within FOV."""
        # Angles within FOV
        sensor_angles = torch.tensor([[0.1, 0.0], [-0.1, 1.0]], device=self.device, dtype=self.dtype)

        valid = self.lidar.valid_sensor_angles(sensor_angles)

        self.assertEqual(valid.shape, (2,))
        self.assertTrue(torch.all(valid))

    def test_valid_sensor_angles_outside_fov(self):
        """Test validation for angles outside FOV."""
        # Angles outside FOV (elevation too high)
        sensor_angles = torch.tensor([[1.0, 0.0], [0.0, 0.0]], device=self.device, dtype=self.dtype)

        valid = self.lidar.valid_sensor_angles(sensor_angles)

        self.assertFalse(valid[0].item())  # Out of FOV
        self.assertTrue(valid[1].item())  # Within FOV

    def test_valid_sensor_angles_numpy(self):
        """Test validation with numpy input."""
        sensor_angles_np = np.array([[0.1, 0.0], [-0.1, 1.0]], dtype=np.float32)

        valid = self.lidar.valid_sensor_angles(sensor_angles_np)

        self.assertIsInstance(valid, torch.Tensor)
        self.assertEqual(valid.shape, (2,))


class TestRowOffsetStructuredSpinningLidarModelFrameTimes(unittest.TestCase):
    """Tests for sensor_angles_relative_frame_times."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )
        row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
        )

        self.lidar = RowOffsetStructuredSpinningLidarModel(projection=projection, angles_to_columns_map_init=True)

    def test_sensor_angles_relative_frame_times_range(self):
        """Test that relative frame times are in [0, 1]."""
        # Angles within FOV
        sensor_angles = torch.tensor(
            [[0.1, -torch.pi + 0.1], [0.0, 0.0], [-0.1, torch.pi - 0.1]], device=self.device, dtype=self.dtype
        )

        times = self.lidar.sensor_angles_relative_frame_times(sensor_angles)

        self.assertEqual(times.shape, (3,))
        self.assertTrue(torch.all(times >= 0.0))
        self.assertTrue(torch.all(times <= 1.0))

    def test_sensor_angles_relative_frame_times_numpy(self):
        """Test relative frame times with numpy input."""
        sensor_angles_np = np.array([[0.1, 0.0], [0.0, 1.0]], dtype=np.float32)

        times = self.lidar.sensor_angles_relative_frame_times(sensor_angles_np)

        self.assertIsInstance(times, torch.Tensor)
        self.assertEqual(times.shape, (2,))


class TestRowOffsetStructuredSpinningLidarModelEdgeCases(unittest.TestCase):
    """Tests for edge cases and additional coverage."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )
        row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
        )

        self.lidar = RowOffsetStructuredSpinningLidarModel(projection=projection)

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

    def test_batch_elements_to_sensor_angles(self):
        """Test batch processing of elements to sensor angles."""
        N = 100
        rows = torch.randint(0, self.n_rows, (N,), device=self.device)
        cols = torch.randint(0, self.n_columns, (N,), device=self.device)
        elements = torch.stack([rows, cols], dim=-1)

        result = self.lidar.elements_to_sensor_angles(elements)

        self.assertEqual(result.sensor_angles.shape, (N, 2))

    def test_batch_sensor_rays_to_angles(self):
        """Test batch processing of sensor rays."""
        N = 100
        sensor_rays = torch.randn(N, 3, device=self.device, dtype=self.dtype)
        sensor_rays = sensor_rays / torch.norm(sensor_rays, dim=-1, keepdim=True)

        result = self.lidar.sensor_rays_to_sensor_angles(sensor_rays)

        self.assertEqual(result.sensor_angles.shape, (N, 2))

    def test_batch_sensor_angles_to_rays(self):
        """Test batch processing of sensor angles to rays."""
        N = 100
        sensor_angles = torch.zeros(N, 2, device=self.device, dtype=self.dtype)
        sensor_angles[:, 0] = torch.linspace(-0.2, 0.2, N, device=self.device)  # elevations
        sensor_angles[:, 1] = torch.linspace(-torch.pi, torch.pi, N, device=self.device)  # azimuths

        result = self.lidar.sensor_angles_to_sensor_rays(sensor_angles)

        self.assertEqual(result.sensor_rays.shape, (N, 3))

    def test_world_points_to_sensor_angles_with_timestamps(self):
        """Test inverse projection with timestamps."""
        world_points = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], device=self.device, dtype=self.dtype)

        result = self.lidar.world_points_to_sensor_angles_shutter_pose(
            world_points=world_points,
            dynamic_pose=self.dynamic_pose,
            start_timestamp_us=0,
            end_timestamp_us=100000,
            lazy_init_angles_map=True,
            return_timestamps=True,
        )

        self.assertIsNotNone(result.timestamps_us)
        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape, (2,))

    def test_elements_to_world_rays_with_rotation(self):
        """Test world ray generation with non-identity rotation."""
        import math

        # Create a dynamic pose with rotation
        pose_start = Pose(
            translation=torch.zeros(3, device=self.device, dtype=self.dtype),
            rotation=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype),
        )
        # Small rotation around z-axis
        angle = 0.1
        pose_end = Pose(
            translation=torch.tensor([0.1, 0.0, 0.0], device=self.device, dtype=self.dtype),
            rotation=torch.tensor(
                [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)], device=self.device, dtype=self.dtype
            ),
        )
        dynamic_pose = DynamicPose(start_pose=pose_start, end_pose=pose_end)

        elements = torch.tensor([[0, 0], [8, 180]], device=self.device, dtype=torch.long)

        result = self.lidar.elements_to_world_rays_shutter_pose(
            elements=elements,
            dynamic_pose=dynamic_pose,
        )

        self.assertEqual(result.world_rays.shape, (2, 6))

    def test_sensor_rays_direction_normalization(self):
        """Test that sensor rays are normalized."""
        elements = torch.tensor([[0, 0], [8, 180], [15, 270]], device=self.device, dtype=torch.long)

        sensor_rays = self.lidar.elements_to_sensor_rays(elements)

        norms = torch.norm(sensor_rays, dim=1)
        torch.testing.assert_close(norms, torch.ones(3, device=self.device), atol=1e-5, rtol=1e-5)

    def test_custom_fov_eps_factor(self):
        """Test initialization with custom FOV epsilon factor."""
        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )
        row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
        )

        lidar = RowOffsetStructuredSpinningLidarModel(projection=projection, fov_eps_factor=8.0)

        self.assertIsInstance(lidar, RowOffsetStructuredSpinningLidarModel)


class TestRowOffsetStructuredSpinningLidarModelCounterClockwise(unittest.TestCase):
    """Tests for counter-clockwise spinning direction."""

    def setUp(self):
        """Set up test fixtures with CCW spinning direction."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )
        row_azimuth_offsets_rad = torch.zeros(self.n_rows, device=self.device, dtype=self.dtype)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            fov_horiz_start_rad=-torch.pi,
            fov_horiz_span_rad=2 * torch.pi,
            fov_vert_start_rad=0.26,
            fov_vert_span_rad=0.52,
            spinning_frequency_hz=10.0,
            spinning_direction="ccw",  # Counter-clockwise
        )

        self.lidar = RowOffsetStructuredSpinningLidarModel(projection=projection)

    def test_ccw_spinning_direction(self):
        """Test CCW spinning direction property."""
        self.assertEqual(self.lidar.spinning_direction, "ccw")

    def test_ccw_sensor_rays_conversion(self):
        """Test sensor ray conversion with CCW direction."""
        sensor_rays = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=self.dtype)

        result = self.lidar.sensor_rays_to_sensor_angles(sensor_rays)

        self.assertEqual(result.sensor_angles.shape, (1, 2))

    def test_ccw_elements_to_sensor_angles(self):
        """Test element to angle conversion with CCW direction."""
        elements = torch.tensor([[0, 0], [8, 180]], device=self.device, dtype=torch.long)

        result = self.lidar.elements_to_sensor_angles(elements)

        self.assertEqual(result.sensor_angles.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
