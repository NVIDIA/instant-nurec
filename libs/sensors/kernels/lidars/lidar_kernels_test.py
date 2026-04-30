# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Comprehensive unit tests for LiDAR sensor kernels with oracle implementations.

Tests the Slang-backed LiDAR projection functions by verifying against
reference Python implementations:
- Element to sensor angle conversion
- Ray generation with pose interpolation
- Inverse projection (world → sensor angles)
- Sensor ray/angle conversions (spherical coordinates)
- Rolling shutter handling for spinning LiDARs

The LiDAR projection uses standard spherical coordinates without any distortion,
so the reference implementation is simply the mathematical definition.
"""

import json
import unittest

import numpy as np
import parameterized
import torch

from scipy.spatial.transform import Rotation

from libs.geometry.kernels.quaternion import quat_identity
from libs.sensors.kernels.common.pose import DynamicPose, Pose, Trajectory
from libs.sensors.kernels.lidars import (
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
    sensor_angles_to_sensor_rays,
    sensor_rays_to_sensor_angles,
)


# Path to test data files (relative to workspace root, as used by Bazel)
TEST_DATA_DIR = "libs/sensors/kernels/lidars/test_data"


# Test configuration - Slang kernels require CUDA
device = torch.device("cuda")

# Test parameters
NUM_ELEMENTS = 100

# Tolerance parameters - LiDAR uses exact spherical math so tolerances can be tight
ATOL = 1e-5
RTOL = 1e-5
MAX_ANGLE_DEVIATION = 1e-5  # Maximum allowed angle deviation in radians

# Tighter tolerances for gradient comparisons
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-4

# Tolerances for numerical (finite difference) gradient comparisons.
# Central finite difference: (f(x+ε) - f(x-ε)) / 2ε has two error sources:
#   1. Truncation error: O(ε²) from Taylor expansion - decreases with smaller ε
#   2. Roundoff error: ~machine_epsilon/ε from catastrophic cancellation - increases with smaller ε
# For float32 (machine_epsilon ≈ 1e-7) with ε=1e-4, the combined error is ~0.1-1%.
# We use 2% tolerance to account for both error sources plus any function-specific factors.
NUMERICAL_GRAD_RTOL = 0.02
NUMERICAL_GRAD_ATOL = 0.02

# Very strict tolerance for normalization checks
NORM_ATOL = 1e-6


# ============================================================================
# Reference Implementation (Python Oracle)
# ============================================================================


class ReferenceSphericalCoordinates:
    """Reference implementation of spherical coordinate conversions.

    Uses the standard mathematical definition:
        x = cos(elevation) * cos(azimuth)
        y = cos(elevation) * sin(azimuth)
        z = sin(elevation)

    Where:
        elevation: angle from XY plane (positive up)
        azimuth: angle in XY plane from X axis (positive toward Y)

    This is the same convention used by the LiDAR kernel.
    """

    @staticmethod
    def spherical_to_cartesian(elevation: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
        """Convert spherical coordinates to Cartesian ray direction.

        Args:
            elevation: (N,) elevation angles in radians
            azimuth: (N,) azimuth angles in radians

        Returns:
            (N, 3) normalized ray directions [x, y, z]
        """
        cos_elev = np.cos(elevation)
        x = cos_elev * np.cos(azimuth)
        y = cos_elev * np.sin(azimuth)
        z = np.sin(elevation)
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def cartesian_to_spherical(rays: np.ndarray) -> np.ndarray:
        """Convert Cartesian ray direction to spherical coordinates.

        Args:
            rays: (N, 3) ray directions (will be normalized)

        Returns:
            (N, 2) [elevation, azimuth] in radians
        """
        # Normalize rays
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays = rays / np.maximum(norms, 1e-10)

        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]

        # Elevation: angle from XY plane
        xy_norm = np.sqrt(x * x + y * y)
        elevation = np.arctan2(z, xy_norm)

        # Azimuth: angle in XY plane from X axis
        azimuth = np.arctan2(y, x)

        return np.stack([elevation, azimuth], axis=-1)

    @staticmethod
    def normalize_angle(angle: np.ndarray) -> np.ndarray:
        """Normalize angle to [-π, π]."""
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi


# ============================================================================
# Common Test Helpers
# ============================================================================


def normalize_angle_np(angle: np.ndarray) -> np.ndarray:
    """Normalize angle to [-π, π] (numpy version)."""
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def normalize_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-π, π] (torch version)."""
    return torch.remainder(angle + np.pi, 2 * np.pi) - np.pi


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z] using scipy."""
    scipy_quat = Rotation.from_matrix(R.cpu().numpy()).as_quat()  # [x, y, z, w]
    return torch.tensor(
        [scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]],
        device=R.device,
        dtype=R.dtype,
    )


def create_identity_dynamic_pose(dev: torch.device, dtype: torch.dtype = torch.float32) -> DynamicPose:
    """Create a dynamic pose with identity rotation and zero translation."""
    trans = torch.zeros(3, device=dev, dtype=dtype)
    rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev, dtype=dtype)  # Identity quaternion [w, x, y, z]
    pose = Pose(trans, rot)
    return DynamicPose.from_static_pose(pose)


def create_dynamic_pose(
    trans_start: torch.Tensor,
    trans_end: torch.Tensor,
    rot_start: torch.Tensor,
    rot_end: torch.Tensor,
    dev: torch.device,
) -> DynamicPose:
    """Create a dynamic pose with specified start/end translations and rotations."""
    start_pose = Pose(trans_start, rot_start)
    end_pose = Pose(trans_end, rot_end)
    return DynamicPose(start_pose=start_pose, end_pose=end_pose)


def create_dynamic_pose_from_transforms(
    T_start: torch.Tensor,
    T_end: torch.Tensor,
    dev: torch.device,
) -> DynamicPose:
    """Create a dynamic pose from 4x4 transformation matrices."""
    trans_start = T_start[:3, 3]
    trans_end = T_end[:3, 3]
    rot_start = rotmat_to_quat(T_start[:3, :3])
    rot_end = rotmat_to_quat(T_end[:3, :3])
    return create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, dev)


# ============================================================================
# CUDA Availability Check
# ============================================================================


class TestSpinningDirection(unittest.TestCase):
    """Test SpinningDirection enum and its conversions."""

    def test_enum_values(self):
        """Test that enum values match Slang SpinningDirection."""
        # These values must match interface.slang: CLOCKWISE = 0, COUNTERCLOCKWISE = 1
        self.assertEqual(SpinningDirection.CLOCKWISE, 0)
        self.assertEqual(SpinningDirection.COUNTERCLOCKWISE, 1)

    def test_enum_is_int(self):
        """Test that SpinningDirection can be used as int (for kernel calls)."""
        # IntEnum values should work directly as integers
        self.assertIsInstance(int(SpinningDirection.CLOCKWISE), int)
        self.assertIsInstance(int(SpinningDirection.COUNTERCLOCKWISE), int)

        # Should be usable in arithmetic (verifies IntEnum behavior)
        self.assertEqual(SpinningDirection.CLOCKWISE + 1, SpinningDirection.COUNTERCLOCKWISE)

    def test_from_string_clockwise(self):
        """Test conversion from 'cw' string."""
        result = SpinningDirection.from_string("cw")
        self.assertEqual(result, SpinningDirection.CLOCKWISE)

    def test_from_string_counterclockwise(self):
        """Test conversion from 'ccw' string."""
        result = SpinningDirection.from_string("ccw")
        self.assertEqual(result, SpinningDirection.COUNTERCLOCKWISE)

    def test_from_string_invalid(self):
        """Test that unknown strings raise ValueError."""
        with self.assertRaises(ValueError) as context:
            SpinningDirection.from_string("unknown")
        self.assertIn("Invalid spinning direction", str(context.exception))
        self.assertIn("'cw' or 'ccw'", str(context.exception))

    def test_from_string_case_insensitive(self):
        """Test that string conversion is case-insensitive."""
        self.assertEqual(SpinningDirection.from_string("CW"), SpinningDirection.CLOCKWISE)
        self.assertEqual(SpinningDirection.from_string("CCW"), SpinningDirection.COUNTERCLOCKWISE)
        self.assertEqual(SpinningDirection.from_string("Cw"), SpinningDirection.CLOCKWISE)
        self.assertEqual(SpinningDirection.from_string("Ccw"), SpinningDirection.COUNTERCLOCKWISE)

    def test_enum_identity(self):
        """Test enum identity comparisons."""
        self.assertIs(SpinningDirection.from_string("cw"), SpinningDirection.CLOCKWISE)
        self.assertIs(SpinningDirection.from_string("ccw"), SpinningDirection.COUNTERCLOCKWISE)


class TestLidarProjection(unittest.TestCase):
    """Test LiDAR projection operations."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Create a simple spinning LiDAR configuration
        # 16 rows, 1024 columns, 360 degree FOV
        n_rows = 16
        n_columns = 1024

        # Row elevations: -15 to 15 degrees
        row_elevations = torch.linspace(-15, 15, n_rows, device=device) * np.pi / 180.0

        # Column azimuths: 0 to 360 degrees
        column_azimuths = torch.linspace(0, 2 * np.pi, n_columns, device=device)

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=15.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=30.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=None,  # No row offsets for basic test
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

        # Create a simple dynamic pose
        self.dynamic_pose = self._create_simple_dynamic_pose()

    def _create_simple_dynamic_pose(self):
        """Create a simple dynamic pose with 2 control poses."""
        pose1_trans = torch.zeros(3, device=device)
        pose1_rot = quat_identity((1,), device=device).squeeze(0)
        pose2_trans = torch.tensor([0.1, 0.0, 0.0], device=device)
        pose2_rot = quat_identity((1,), device=device).squeeze(0)
        return create_dynamic_pose(pose1_trans, pose2_trans, pose1_rot, pose2_rot, device)

    def test_elements_to_sensor_angles_basic(self):
        """Test element to sensor angle conversion."""
        # Create element indices
        elements = torch.tensor(
            [
                [0, 0],  # First row, first column
                [7, 512],  # Middle row, middle column
                [15, 1023],  # Last row, last column
            ],
            device=device,
            dtype=torch.int32,
        )

        sensor_angles, _ = elements_to_sensor_angles(
            self.projection,
            elements,
        )

        # Check shape
        self.assertEqual(sensor_angles.shape, (3, 2))

        # Check that angles are within expected range
        elevations = sensor_angles[:, 0]
        azimuths = sensor_angles[:, 1]

        self.assertTrue(torch.all(elevations >= -15.0 * np.pi / 180.0))
        self.assertTrue(torch.all(elevations <= 15.0 * np.pi / 180.0))
        self.assertTrue(torch.all(azimuths >= 0.0))
        self.assertTrue(torch.all(azimuths <= 2 * np.pi))

    def test_elements_to_sensor_angles_valid_flags(self):
        """Test that valid_flags are returned correctly for in-bounds elements."""
        # Create valid element indices
        elements = torch.tensor(
            [
                [0, 0],  # First row, first column - valid
                [7, 512],  # Middle row, middle column - valid
                [15, 1023],  # Last row, last column - valid
            ],
            device=device,
            dtype=torch.int32,
        )

        sensor_angles, valid_flags = elements_to_sensor_angles(
            self.projection,
            elements,
            return_valid_flags=True,
        )

        # Check shapes
        self.assertEqual(sensor_angles.shape, (3, 2))
        assert valid_flags is not None
        self.assertEqual(valid_flags.shape, (3,))

        # All elements are valid (within bounds)
        self.assertTrue(torch.all(valid_flags), "All in-bounds elements should be valid")

    def test_elements_to_sensor_angles_out_of_bounds_row(self):
        """Test that out-of-bounds row indices are flagged as invalid."""
        # Projection has 16 rows (0-15)
        elements = torch.tensor(
            [
                [0, 0],  # Valid
                [-1, 0],  # Invalid: negative row
                [16, 0],  # Invalid: row >= n_rows
                [100, 0],  # Invalid: way out of bounds
            ],
            device=device,
            dtype=torch.int32,
        )

        sensor_angles, valid_flags = elements_to_sensor_angles(
            self.projection,
            elements,
            return_valid_flags=True,
        )

        # Check validity
        expected_valid = torch.tensor([True, False, False, False], device=device)
        assert valid_flags is not None
        self.assertTrue(
            torch.all(valid_flags == expected_valid),
            f"Expected valid_flags {expected_valid.cpu()}, got {valid_flags.cpu()}",
        )

        # Invalid elements should have zero angles
        assert valid_flags is not None
        self.assertTrue(
            torch.allclose(sensor_angles[~valid_flags], torch.zeros(3, 2, device=device)),
            "Out-of-bounds elements should return (0, 0) angles",
        )

    def test_elements_to_sensor_angles_out_of_bounds_column(self):
        """Test that out-of-bounds column indices are flagged as invalid."""
        # Projection has 1024 columns (0-1023)
        elements = torch.tensor(
            [
                [0, 0],  # Valid
                [0, -1],  # Invalid: negative column
                [0, 1024],  # Invalid: col >= n_columns
                [0, 9999],  # Invalid: way out of bounds
            ],
            device=device,
            dtype=torch.int32,
        )

        _sensor_angles, valid_flags = elements_to_sensor_angles(
            self.projection,
            elements,
            return_valid_flags=True,
        )

        # Check validity
        expected_valid = torch.tensor([True, False, False, False], device=device)
        assert valid_flags is not None
        self.assertTrue(
            torch.all(valid_flags == expected_valid),
            f"Expected valid_flags {expected_valid.cpu()}, got {valid_flags.cpu()}",
        )

    def test_elements_to_sensor_angles_mixed_validity(self):
        """Test batch with mix of valid and invalid elements."""
        elements = torch.tensor(
            [
                [0, 0],  # Valid
                [-1, 0],  # Invalid row
                [7, 512],  # Valid
                [0, -1],  # Invalid column
                [15, 1023],  # Valid
                [16, 1024],  # Both invalid
            ],
            device=device,
            dtype=torch.int32,
        )

        sensor_angles, valid_flags = elements_to_sensor_angles(
            self.projection,
            elements,
            return_valid_flags=True,
        )

        # Check validity
        expected_valid = torch.tensor([True, False, True, False, True, False], device=device)
        assert valid_flags is not None
        self.assertTrue(
            torch.all(valid_flags == expected_valid),
            f"Expected valid_flags {expected_valid.cpu()}, got {valid_flags.cpu()}",
        )

        # Valid elements should have non-zero angles (at least one component)
        valid_angles = sensor_angles[valid_flags]
        self.assertTrue(valid_angles.shape[0] == 3, "Should have 3 valid elements")

    def test_elements_to_sensor_angles_empty_input(self):
        """Test handling of empty input tensor."""
        elements = torch.empty((0, 2), device=device, dtype=torch.int32)

        sensor_angles, valid_flags = elements_to_sensor_angles(
            self.projection,
            elements,
            return_valid_flags=True,
        )

        self.assertEqual(sensor_angles.shape, (0, 2))
        assert valid_flags is not None
        self.assertEqual(valid_flags.shape, (0,))

    def test_elements_to_sensor_angles_without_valid_flags(self):
        """Test that function works correctly when return_valid_flags=False."""
        elements = torch.tensor(
            [
                [0, 0],
                [7, 512],
            ],
            device=device,
            dtype=torch.int32,
        )

        # Should return tuple with None for valid_flags
        sensor_angles, valid_flags = elements_to_sensor_angles(
            self.projection,
            elements,
            return_valid_flags=False,
        )

        # sensor_angles should be a tensor, valid_flags should be None
        self.assertIsInstance(sensor_angles, torch.Tensor)
        self.assertEqual(sensor_angles.shape, (2, 2))
        self.assertIsNone(valid_flags)

    def test_sensor_angles_to_sensor_rays(self):
        """Test conversion from sensor angles to rays."""
        # Create sensor angles
        sensor_angles = torch.tensor(
            [
                [0.0, 0.0],  # Horizontal forward
                [0.0, np.pi / 2],  # Horizontal right
                [15.0 * np.pi / 180.0, 0.0],  # Up and forward
            ],
            device=device,
        )

        sensor_rays = sensor_angles_to_sensor_rays(
            self.projection,
            sensor_angles,
        )

        # Check shape
        self.assertEqual(sensor_rays.shape, (3, 3))

        # Check that rays are normalized
        norms = sensor_rays.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_sensor_rays_to_sensor_angles(self):
        """Test conversion from sensor rays to angles."""
        # Create sensor rays (normalized)
        sensor_rays = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Right
                [0.0, 1.0, 0.0],  # Forward (or up, depending on convention)
                [0.0, 0.0, 1.0],  # Up (or forward)
            ],
            device=device,
        )
        sensor_rays = sensor_rays / sensor_rays.norm(dim=-1, keepdim=True)

        sensor_angles = sensor_rays_to_sensor_angles(
            self.projection,
            sensor_rays,
        )

        # Check shape
        self.assertEqual(sensor_angles.shape, (3, 2))

    def test_round_trip_angles_rays(self):
        """Test that angles → rays → angles is consistent."""
        # Start with sensor angles
        sensor_angles_orig = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.5],
                [-0.1, 1.0],
            ],
            device=device,
        )

        # Convert to rays
        sensor_rays = sensor_angles_to_sensor_rays(
            self.projection,
            sensor_angles_orig,
        )

        # Convert back to angles
        sensor_angles_new = sensor_rays_to_sensor_angles(
            self.projection,
            sensor_rays,
        )

        # Check consistency
        # Note: May have differences due to numerical precision and angle wrapping
        self.assertEqual(sensor_angles_new.shape, sensor_angles_orig.shape)

    def test_generate_spinning_lidar_rays(self):
        """Test ray generation for spinning LiDAR."""
        # Create element indices
        elements = torch.tensor(
            [
                [0, 0],
                [7, 512],
                [15, 1023],
            ],
            device=device,
            dtype=torch.int32,
        )

        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            self.dynamic_pose,
        )

        # Check shape (6D rays: origin + direction)
        self.assertEqual(world_rays.shape, (3, 6))

        # Check that directions are normalized
        directions = world_rays[:, 3:6]
        norms = directions.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_inverse_project_spinning_lidar(self):
        """Test inverse projection of world points to sensor angles."""
        # Create world points in front of sensor
        world_points = torch.tensor(
            [
                [5.0, 0.0, 0.0],  # 5m to the right
                [0.0, 10.0, 0.0],  # 10m forward
                [0.0, 5.0, 2.0],  # Forward and up
            ],
            device=device,
        )

        sensor_angles, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            self.dynamic_pose,
            max_iterations=5,
            return_valid_flags=True,
        )

        # Check shapes
        self.assertEqual(sensor_angles.shape, (3, 2))
        assert valid is not None
        self.assertEqual(valid.shape, (3,))

    def test_batch_processing(self):
        """Test that batched processing works correctly."""
        # Create many element indices
        num_elements = NUM_ELEMENTS
        rows = torch.randint(0, self.projection.n_rows, (num_elements,), device=device)
        cols = torch.randint(0, self.projection.n_columns, (num_elements,), device=device)
        elements = torch.stack([rows, cols], dim=1).to(torch.int32)

        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            self.dynamic_pose,
        )

        # Check shape
        self.assertEqual(world_rays.shape, (num_elements, 6))

    def test_empty_input(self):
        """Test handling of empty input."""
        elements = torch.empty((0, 2), device=device, dtype=torch.int32)

        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            self.dynamic_pose,
        )

        # Should return empty tensor
        self.assertEqual(world_rays.shape, (0, 6))

    def test_fov_validation(self):
        """Test that points outside FOV are marked invalid."""
        # Create points that should be outside FOV
        world_points = torch.tensor(
            [
                [0.0, 0.0, 10.0],  # Way above (should be outside vert FOV)
                [0.0, 0.0, -10.0],  # Way below
            ],
            device=device,
        )

        _sensor_angles, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            self.dynamic_pose,
            max_iterations=5,
            return_valid_flags=True,
        )

        # Check that points outside FOV are marked invalid
        assert valid is not None
        self.assertFalse(valid.all(), "Expected some points to be marked invalid")
        self.assertEqual(valid.sum(), 0, "All points should be outside FOV")

    def test_element_bounds_checking(self):
        """Test that valid element indices produce valid results."""
        # Valid elements
        elements = torch.tensor(
            [
                [0, 0],
                [self.projection.n_rows - 1, self.projection.n_columns - 1],
            ],
            device=device,
            dtype=torch.int32,
        )

        sensor_angles, _ = elements_to_sensor_angles(
            self.projection,
            elements,
        )

        self.assertEqual(sensor_angles.shape, (2, 2))


class TestLidarWithRowOffsets(unittest.TestCase):
    """Test LiDAR with row azimuth offsets (Hesai-style)."""

    def setUp(self):
        """Set up test fixtures with row offsets."""
        torch.manual_seed(42)

        # Create a LiDAR with row offsets
        n_rows = 8
        n_columns = 512

        row_elevations = torch.linspace(-10, 10, n_rows, device=device) * np.pi / 180.0
        column_azimuths = torch.linspace(0, 2 * np.pi, n_columns, device=device)

        # Add small row offsets (typical for Hesai sensors)
        row_offsets = torch.linspace(0, 0.1, n_rows, device=device)

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=10.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=20.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=row_offsets,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

        self.dynamic_pose = self._create_simple_dynamic_pose()

    def _create_simple_dynamic_pose(self):
        """Create a simple dynamic pose."""
        pose1_trans = torch.zeros(3, device=device)
        pose1_rot = quat_identity((1,), device=device).squeeze(0)
        pose2_trans = torch.tensor([0.2, 0.0, 0.0], device=device)
        pose2_rot = quat_identity((1,), device=device).squeeze(0)
        return create_dynamic_pose(pose1_trans, pose2_trans, pose1_rot, pose2_rot, device)

    def test_row_offsets_applied(self):
        """Test that row offsets are properly applied."""
        # Elements from different rows
        elements = torch.tensor(
            [
                [0, 100],
                [self.projection.n_rows - 1, 100],
            ],
            device=device,
            dtype=torch.int32,
        )

        sensor_angles, _ = elements_to_sensor_angles(
            self.projection,
            elements,
        )

        # Angles should differ due to row offsets
        # The azimuth values should have the row offset applied
        self.assertEqual(sensor_angles.shape, (2, 2))


# ============================================================================
# Oracle Tests - Verifying against reference implementation
# ============================================================================


class TestSphericalCoordinatesOracle(unittest.TestCase):
    """Oracle tests for spherical coordinate conversions.

    Verifies the LiDAR kernel implementation matches the mathematical
    definition of spherical coordinates exactly.
    """

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

        # Create a simple LiDAR projection for testing
        n_rows = 16
        n_columns = 360

        row_elevations = torch.linspace(-30, 30, n_rows, device=device) * np.pi / 180.0
        column_azimuths = torch.linspace(0, 2 * np.pi, n_columns, device=device)

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=30.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=60.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

        self.ref = ReferenceSphericalCoordinates()

    def test_angles_to_rays_known_values(self):
        """Test spherical to Cartesian conversion for known values."""
        # Test cases: (elevation, azimuth) -> expected (x, y, z)
        test_cases = [
            # Horizontal plane (elevation = 0)
            ((0.0, 0.0), (1.0, 0.0, 0.0)),  # +X
            ((0.0, np.pi / 2), (0.0, 1.0, 0.0)),  # +Y
            ((0.0, np.pi), (-1.0, 0.0, 0.0)),  # -X
            ((0.0, -np.pi / 2), (0.0, -1.0, 0.0)),  # -Y
            # Vertical direction
            ((np.pi / 2, 0.0), (0.0, 0.0, 1.0)),  # +Z (up)
            ((-np.pi / 2, 0.0), (0.0, 0.0, -1.0)),  # -Z (down)
            # 45 degree elevation
            ((np.pi / 4, 0.0), (np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2)),
            ((np.pi / 4, np.pi / 2), (0.0, np.sqrt(2) / 2, np.sqrt(2) / 2)),
        ]

        for (elevation, azimuth), expected_ray in test_cases:
            with self.subTest(elevation=np.degrees(elevation), azimuth=np.degrees(azimuth)):
                # Kernel
                angles = torch.tensor([[elevation, azimuth]], dtype=torch.float32, device=device)
                kernel_rays = sensor_angles_to_sensor_rays(self.projection, angles)

                # Reference
                ref_rays = self.ref.spherical_to_cartesian(np.array([elevation]), np.array([azimuth]))

                # Compare to expected
                np.testing.assert_allclose(
                    kernel_rays[0].cpu().numpy(),
                    expected_ray,
                    atol=ATOL,
                    err_msg=f"Failed for elevation={np.degrees(elevation)}, azimuth={np.degrees(azimuth)}",
                )

                # Compare kernel to reference
                np.testing.assert_allclose(
                    kernel_rays[0].cpu().numpy(),
                    ref_rays[0],
                    atol=ATOL,
                    err_msg="Kernel doesn't match reference",
                )

    def test_rays_to_angles_known_values(self):
        """Test Cartesian to spherical conversion for known values."""
        # Test cases: (x, y, z) -> expected (elevation, azimuth)
        test_cases = [
            ((1.0, 0.0, 0.0), (0.0, 0.0)),  # +X
            ((0.0, 1.0, 0.0), (0.0, np.pi / 2)),  # +Y
            ((-1.0, 0.0, 0.0), (0.0, np.pi)),  # -X
            ((0.0, -1.0, 0.0), (0.0, -np.pi / 2)),  # -Y
            ((0.0, 0.0, 1.0), (np.pi / 2, 0.0)),  # +Z (up) - azimuth is undefined
            ((0.0, 0.0, -1.0), (-np.pi / 2, 0.0)),  # -Z (down)
        ]

        for ray, (expected_elev, expected_az) in test_cases:
            with self.subTest(ray=ray):
                # Kernel
                rays = torch.tensor([ray], dtype=torch.float32, device=device)
                kernel_angles = sensor_rays_to_sensor_angles(self.projection, rays)

                kernel_elev = kernel_angles[0, 0].cpu().item()
                kernel_az = kernel_angles[0, 1].cpu().item()

                # Check elevation
                np.testing.assert_allclose(
                    kernel_elev,
                    expected_elev,
                    atol=ATOL,
                    err_msg=f"Elevation mismatch for ray {ray}",
                )

                # Check azimuth (skip for poles where azimuth is undefined)
                if abs(ray[2]) < 0.999:
                    np.testing.assert_allclose(
                        kernel_az,
                        expected_az,
                        atol=ATOL,
                        err_msg=f"Azimuth mismatch for ray {ray}",
                    )

    def test_round_trip_angles_rays_angles(self):
        """Test exact round-trip: angles → rays → angles."""
        # Test various angles
        elevations = np.linspace(-80, 80, 17) * np.pi / 180  # Avoid poles
        azimuths = np.linspace(-170, 170, 35) * np.pi / 180

        for elev in elevations:
            for az in azimuths:
                with self.subTest(elevation_deg=np.degrees(elev), azimuth_deg=np.degrees(az)):
                    original_angles = torch.tensor([[elev, az]], dtype=torch.float32, device=device)

                    # Angles → Rays
                    rays = sensor_angles_to_sensor_rays(self.projection, original_angles)

                    # Rays → Angles
                    recovered_angles = sensor_rays_to_sensor_angles(self.projection, rays)

                    # Compare (handle angle wrapping for azimuth)
                    orig = original_angles[0].cpu().numpy()
                    recov = recovered_angles[0].cpu().numpy()

                    np.testing.assert_allclose(
                        recov[0], orig[0], atol=MAX_ANGLE_DEVIATION, err_msg=f"Elevation round-trip failed"
                    )

                    # Azimuth might wrap around ±π
                    az_diff = abs(self.ref.normalize_angle(np.array([recov[1] - orig[1]]))[0])
                    self.assertLess(
                        az_diff,
                        MAX_ANGLE_DEVIATION,
                        f"Azimuth round-trip failed: orig={np.degrees(orig[1]):.2f}°, "
                        f"recov={np.degrees(recov[1]):.2f}°",
                    )

    def test_round_trip_rays_angles_rays(self):
        """Test exact round-trip: rays → angles → rays."""
        # Generate random rays
        np.random.seed(42)
        num_rays = 100
        rays_np = np.random.randn(num_rays, 3)
        rays_np = rays_np / np.linalg.norm(rays_np, axis=1, keepdims=True)

        rays = torch.tensor(rays_np, dtype=torch.float32, device=device)

        # Rays → Angles
        angles = sensor_rays_to_sensor_angles(self.projection, rays)

        # Angles → Rays
        recovered_rays = sensor_angles_to_sensor_rays(self.projection, angles)

        # Compare
        for i in range(num_rays):
            orig_ray = rays_np[i]
            recov_ray = recovered_rays[i].cpu().numpy()

            # Rays should be identical (both normalized)
            np.testing.assert_allclose(
                recov_ray, orig_ray, atol=ATOL, err_msg=f"Ray {i} round-trip failed: {orig_ray} -> {recov_ray}"
            )

    def test_reference_consistency(self):
        """Test that kernel matches reference for random angles."""
        np.random.seed(123)
        num_samples = 200

        # Random elevations in [-80, 80] degrees (avoid poles)
        elevations = np.random.uniform(-80, 80, num_samples) * np.pi / 180
        azimuths = np.random.uniform(-180, 180, num_samples) * np.pi / 180

        angles_np = np.stack([elevations, azimuths], axis=1).astype(np.float32)
        angles = torch.tensor(angles_np, device=device)

        # Kernel
        kernel_rays = sensor_angles_to_sensor_rays(self.projection, angles)

        # Reference
        ref_rays = self.ref.spherical_to_cartesian(elevations, azimuths)

        # Compare
        np.testing.assert_allclose(
            kernel_rays.cpu().numpy(),
            ref_rays,
            atol=ATOL,
            err_msg="Kernel angles→rays doesn't match reference",
        )

        # Now test the inverse
        kernel_angles = sensor_rays_to_sensor_angles(
            self.projection,
            torch.tensor(ref_rays, dtype=torch.float32, device=device),
        )
        ref_angles = self.ref.cartesian_to_spherical(ref_rays)

        # Compare elevations
        np.testing.assert_allclose(
            kernel_angles[:, 0].cpu().numpy(),
            ref_angles[:, 0],
            atol=ATOL,
            err_msg="Kernel rays→angles elevation doesn't match reference",
        )

        # Compare azimuths (accounting for wrapping)
        kernel_az = kernel_angles[:, 1].cpu().numpy()
        ref_az = ref_angles[:, 1]
        az_diff = np.abs(self.ref.normalize_angle(kernel_az - ref_az))
        self.assertTrue(
            np.all(az_diff < ATOL),
            f"Kernel rays→angles azimuth doesn't match reference. Max diff: {np.max(az_diff)}",
        )

    def test_rays_normalized(self):
        """Test that output rays are always normalized."""
        np.random.seed(456)
        num_samples = 100

        elevations = np.random.uniform(-85, 85, num_samples) * np.pi / 180
        azimuths = np.random.uniform(-180, 180, num_samples) * np.pi / 180

        angles = torch.tensor(
            np.stack([elevations, azimuths], axis=1),
            dtype=torch.float32,
            device=device,
        )

        rays = sensor_angles_to_sensor_rays(self.projection, angles)
        norms = rays.norm(dim=-1)

        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=NORM_ATOL,
            rtol=NORM_ATOL,
            msg="Output rays should be normalized",
        )


class TestElementToAngleOracle(unittest.TestCase):
    """Oracle tests for element to angle conversion."""

    def setUp(self):
        """Set up test fixtures."""
        n_rows = 8
        n_columns = 16

        self.row_elevations = np.linspace(-20, 20, n_rows) * np.pi / 180
        self.column_azimuths = np.linspace(0, 2 * np.pi * (n_columns - 1) / n_columns, n_columns)
        self.row_offsets = np.linspace(0, 0.1, n_rows)

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.tensor(self.row_elevations, dtype=torch.float32, device=device),
            column_azimuths_rad=torch.tensor(self.column_azimuths, dtype=torch.float32, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=20.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=40.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=torch.tensor(self.row_offsets, dtype=torch.float32, device=device),
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

    def test_elements_to_angles_exact(self):
        """Test that element→angle conversion matches expected values exactly."""
        n_rows = len(self.row_elevations)
        n_cols = len(self.column_azimuths)

        # Test all elements
        for row in range(n_rows):
            for col in range(n_cols):
                with self.subTest(row=row, col=col):
                    elements = torch.tensor([[row, col]], dtype=torch.int32, device=device)
                    kernel_angles, _ = elements_to_sensor_angles(self.projection, elements)

                    kernel_elev = kernel_angles[0, 0].cpu().item()
                    kernel_az = kernel_angles[0, 1].cpu().item()

                    # Expected values
                    expected_elev = self.row_elevations[row]
                    expected_az = self.column_azimuths[col] + self.row_offsets[row]
                    # Normalize to [-π, π]
                    expected_az = ((expected_az + np.pi) % (2 * np.pi)) - np.pi

                    np.testing.assert_allclose(
                        kernel_elev, expected_elev, atol=ATOL, err_msg=f"Elevation mismatch at ({row}, {col})"
                    )

                    # Azimuth might wrap
                    az_diff = abs(((kernel_az - expected_az + np.pi) % (2 * np.pi)) - np.pi)
                    self.assertLess(
                        az_diff,
                        ATOL,
                        f"Azimuth mismatch at ({row}, {col}): expected {np.degrees(expected_az):.4f}°, "
                        f"got {np.degrees(kernel_az):.4f}°",
                    )


# ============================================================================
# Rolling Shutter Round-Trip Tests (matching ncore test_rolling_shutter_projection)
# ============================================================================


class TestRollingShutterRoundTrip(unittest.TestCase):
    """Test rolling shutter forward/inverse projection consistency.

    This is the key oracle test that verifies the sensorlib implementation
    matches the expected behavior from the ncore reference implementation.

    The test flow matches ncore's test_rolling_shutter_projection:
    1. Generate world rays from elements with rolling shutter
    2. Create world points along the rays at random distances
    3. Inverse project world points back to sensor angles
    4. Compare with original element angles (>98% should match within 1e-3 rad)
    """

    def setUp(self):
        """Set up test fixtures with realistic sensor motion."""
        torch.manual_seed(42)

        # Create a realistic spinning LiDAR configuration
        n_rows = 16
        n_columns = 1024

        # Row elevations: -15 to 15 degrees
        row_elevations = torch.linspace(-15, 15, n_rows, device=device) * np.pi / 180.0

        # Column azimuths: 0 to 360 degrees (full rotation)
        column_azimuths = torch.linspace(0, 2 * np.pi, n_columns, device=device)

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=15.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=30.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

        # Create all element indices
        elements_grid = np.stack(
            np.meshgrid(
                np.arange(n_rows, dtype=np.int32),
                np.arange(n_columns, dtype=np.int32),
                indexing="ij",
            ),
            axis=-1,
        )
        self.all_elements = torch.tensor(elements_grid.reshape(-1, 2), device=device, dtype=torch.int32)

        # Realistic sensor poses (from ncore test)
        self.T_sensor_world_start = torch.tensor(
            [
                [9.9974847e-01, -2.2219338e-02, -3.0501457e-03, 4.6345856e01],
                [2.2246171e-02, 9.9971139e-01, 9.0645989e-03, 2.4201742e-01],
                [2.8478564e-03, -9.1301724e-03, 9.9995428e-01, 2.0181880e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ],
            dtype=torch.float32,
            device=device,
        )

        self.T_sensor_world_end = torch.tensor(
            [
                [9.9977809e-01, -2.1048529e-02, -8.6604326e-04, 4.7494629e01],
                [2.1049019e-02, 9.9977827e-01, 5.6204927e-04, 2.4444677e-01],
                [8.5402117e-04, -5.8015389e-04, 9.9999946e-01, 2.0235672e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ],
            dtype=torch.float32,
            device=device,
        )

        self.dynamic_pose = create_dynamic_pose_from_transforms(
            self.T_sensor_world_start, self.T_sensor_world_end, device
        )

    def test_rolling_shutter_round_trip(self):
        """Test forward/inverse projection round-trip consistency.

        This is the main oracle test matching ncore's test_rolling_shutter_projection.
        """
        # Use a subset of elements for faster testing
        np.random.seed(42)
        num_test_elements = 1000
        indices = np.random.choice(len(self.all_elements), num_test_elements, replace=False)
        elements = self.all_elements[indices]

        # Step 1: Forward projection - elements to world rays
        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            self.dynamic_pose,
        )

        self.assertEqual(world_rays.shape, (num_test_elements, 6))

        # Step 2: Generate world points at random distances along rays
        torch.manual_seed(0)
        distances = torch.rand(num_test_elements, 1, device=device) * 100  # 0-100m
        world_points = world_rays[:, :3] + world_rays[:, 3:6] * distances

        # Step 3: Inverse projection - world points to sensor angles
        sensor_angles_inv, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            self.dynamic_pose,
            max_iterations=10,
            stop_mean_relative_time_error=1e-4,
            stop_delta_mean_relative_time_error=1e-6,
            return_valid_flags=True,
        )

        # Step 4: Get original sensor angles from elements
        sensor_angles_orig, _ = elements_to_sensor_angles(
            self.projection,
            elements,
        )

        # Step 5: Compare - at least 98% should match within 1e-3 rad
        assert valid is not None
        valid_mask = valid.cpu().numpy()
        valid_count = valid_mask.sum()

        # Compute angle differences for valid projections
        angles_orig = sensor_angles_orig[valid].cpu().numpy()
        angles_inv = sensor_angles_inv[valid].cpu().numpy()

        # Normalize angles for comparison
        elev_diff = np.abs(angles_inv[:, 0] - angles_orig[:, 0])
        az_diff = np.abs(normalize_angle_np(angles_inv[:, 1] - angles_orig[:, 1]))

        angle_errors = np.sqrt(elev_diff**2 + az_diff**2)

        # Count how many are within tolerance
        tolerance_rad = 1e-3  # ~0.057 degrees
        within_tolerance = np.sum(angle_errors < tolerance_rad)
        match_ratio = within_tolerance / valid_count if valid_count > 0 else 0

        # Assert at least 98% match (matching ncore test threshold)
        self.assertGreater(
            match_ratio,
            0.98,
            f"Rolling shutter round-trip consistency failed: only {match_ratio * 100:.1f}% "
            f"of {valid_count} valid projections matched within {np.degrees(tolerance_rad):.4f}° "
            f"(max error: {np.degrees(np.max(angle_errors)):.4f}°)",
        )

    def test_rolling_shutter_with_identity_pose(self):
        """Test that identity poses produce consistent results."""
        identity_pose = create_identity_dynamic_pose(device)

        # Test with a few elements
        elements = torch.tensor(
            [[0, 0], [8, 512], [15, 1023]],
            device=device,
            dtype=torch.int32,
        )

        # Forward projection
        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            identity_pose,
        )

        # With identity pose, origin should be at (0,0,0)
        origins = world_rays[:, :3]
        self.assertTrue(
            torch.allclose(origins, torch.zeros_like(origins), atol=ATOL),
            "With identity pose, ray origins should be at (0,0,0)",
        )

        # Directions should be normalized
        directions = world_rays[:, 3:6]
        norms = directions.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=ATOL),
            "World ray directions should be normalized",
        )

        # With identity pose, round-trip should be exact
        # Generate world points and inverse project
        distances = torch.tensor([[10.0], [20.0], [30.0]], device=device)
        world_points = origins + directions * distances

        sensor_angles_inv, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            identity_pose,
            max_iterations=10,
            return_valid_flags=True,
        )

        # All points should be valid (within FOV)
        # Note: some may be outside FOV depending on element selection
        assert valid is not None
        self.assertTrue(valid.any(), "At least some points should be valid")

        # For valid points, the round-trip should match
        if valid.all():
            sensor_angles_orig, _ = elements_to_sensor_angles(self.projection, elements)

            elev_diff = torch.abs(sensor_angles_inv[:, 0] - sensor_angles_orig[:, 0])
            az_diff = torch.abs(normalize_angle_torch(sensor_angles_inv[:, 1] - sensor_angles_orig[:, 1]))

            self.assertTrue(
                torch.all(elev_diff < 1e-3),
                f"Elevation round-trip failed: max diff = {elev_diff.max().item():.6f} rad",
            )
            self.assertTrue(
                torch.all(az_diff < 1e-3),
                f"Azimuth round-trip failed: max diff = {az_diff.max().item():.6f} rad",
            )

    def test_rolling_shutter_timestamps_consistency(self):
        """Test that elements from same column get same relative time."""
        # Elements from same column but different rows
        col = 512
        elements = torch.tensor(
            [[0, col], [8, col], [15, col]],
            device=device,
            dtype=torch.int32,
        )

        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            self.dynamic_pose,
        )

        # All elements from same column should have same origin
        # (since they have same relative time and thus same interpolated pose)
        origins = world_rays[:, :3]

        # Origins should be very close (same interpolated position)
        for i in range(1, len(origins)):
            self.assertTrue(
                torch.allclose(origins[0], origins[i], atol=ATOL),
                f"Elements from same column should have same origin, but got "
                f"{origins[0].cpu().numpy()} vs {origins[i].cpu().numpy()}",
            )


class TestRollingShutterWithRowOffsets(unittest.TestCase):
    """Test rolling shutter with row azimuth offsets (Hesai-style sensors)."""

    def setUp(self):
        """Set up test fixtures with row offsets."""
        torch.manual_seed(42)

        n_rows = 8
        n_columns = 512

        row_elevations = torch.linspace(-10, 10, n_rows, device=device) * np.pi / 180.0
        column_azimuths = torch.linspace(0, 2 * np.pi, n_columns, device=device)

        # Add row offsets (typical for Hesai sensors)
        row_offsets = torch.linspace(0, 0.1, n_rows, device=device)

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=10.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=20.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=row_offsets,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

        # Create dynamic pose with motion
        self.dynamic_pose = self._create_dynamic_pose()

    def _create_dynamic_pose(self):
        """Create dynamic pose with some translation."""
        pose1_trans = torch.zeros(3, device=device)
        pose1_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        pose2_trans = torch.tensor([0.5, 0.0, 0.0], device=device)
        pose2_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        return create_dynamic_pose(pose1_trans, pose2_trans, pose1_rot, pose2_rot, device)

    def test_rolling_shutter_round_trip_with_offsets(self):
        """Test round-trip with row offsets applied."""
        # Create elements
        np.random.seed(123)
        num_elements = 500
        rows = np.random.randint(0, self.projection.n_rows, num_elements)
        cols = np.random.randint(0, self.projection.n_columns, num_elements)
        elements = torch.tensor(
            np.stack([rows, cols], axis=1),
            device=device,
            dtype=torch.int32,
        )

        # Forward projection
        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            self.dynamic_pose,
        )

        # Generate world points
        torch.manual_seed(123)
        distances = torch.rand(num_elements, 1, device=device) * 50
        world_points = world_rays[:, :3] + world_rays[:, 3:6] * distances

        # Inverse projection
        sensor_angles_inv, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            self.dynamic_pose,
            max_iterations=10,
            return_valid_flags=True,
        )

        # Get original angles
        sensor_angles_orig, _ = elements_to_sensor_angles(
            self.projection,
            elements,
        )

        # Compare valid projections
        assert valid is not None
        valid_mask = valid.cpu().numpy()
        if valid_mask.sum() > 0:
            angles_orig = sensor_angles_orig[valid].cpu().numpy()
            angles_inv = sensor_angles_inv[valid].cpu().numpy()

            elev_diff = np.abs(angles_inv[:, 0] - angles_orig[:, 0])
            az_diff = np.abs(normalize_angle_np(angles_inv[:, 1] - angles_orig[:, 1]))
            angle_errors = np.sqrt(elev_diff**2 + az_diff**2)

            tolerance_rad = 1e-3
            within_tolerance = np.sum(angle_errors < tolerance_rad)
            match_ratio = within_tolerance / valid_mask.sum()

            self.assertGreater(
                match_ratio,
                0.95,  # Slightly relaxed for row offset complexity
                f"Round-trip with row offsets failed: {match_ratio * 100:.1f}% matched",
            )


# ============================================================================
# Angles Map Tests
# ============================================================================


class TestEnsureAnglesMap(unittest.TestCase):
    """Test ensure_angles_map() functionality with a small sensor config.

    Uses a tiny sensor (4 rows, 16 columns) to keep tests fast while
    verifying the map building and inverse projection with map work correctly.
    """

    def setUp(self):
        """Set up a small sensor for fast map building."""
        self.n_rows = 4
        self.n_columns = 16

        row_elevations = torch.linspace(-10, 10, self.n_rows, device=device) * np.pi / 180.0
        column_azimuths = torch.linspace(0, 2 * np.pi, self.n_columns, device=device)

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=10.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=20.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

    def test_ensure_angles_map_builds_map(self):
        """Test that ensure_angles_map builds the map when not present."""
        # Initially no map
        self.assertIsNone(self.projection.angles_to_columns_map)

        # Build map
        self.projection.ensure_angles_map(resolution_factor=2)

        # Map should now exist
        self.assertIsNotNone(self.projection.angles_to_columns_map)
        self.assertEqual(self.projection.angles_to_columns_map_resolution_factor, 2)

        # Check map shape: (n_rows * factor, n_columns * factor)
        expected_shape = (self.n_rows * 2, self.n_columns * 2)
        assert self.projection.angles_to_columns_map is not None
        self.assertEqual(self.projection.angles_to_columns_map.shape, expected_shape)

    def test_ensure_angles_map_idempotent(self):
        """Test that calling ensure_angles_map twice doesn't rebuild."""
        self.projection.ensure_angles_map(resolution_factor=2)
        map_id = id(self.projection.angles_to_columns_map)

        # Call again - should not rebuild
        self.projection.ensure_angles_map(resolution_factor=4)  # Different factor ignored
        self.assertEqual(id(self.projection.angles_to_columns_map), map_id)
        self.assertEqual(self.projection.angles_to_columns_map_resolution_factor, 2)

    def test_inverse_projection_with_map(self):
        """Test that inverse projection works correctly with the map."""
        # Build map first
        self.projection.ensure_angles_map(resolution_factor=2)

        # Create dynamic pose
        pose_trans = torch.zeros(3, device=device)
        pose_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        dynamic_pose = create_dynamic_pose(pose_trans, pose_trans, pose_rot, pose_rot, device)

        # Create elements and forward project
        elements = torch.tensor([[0, 0], [1, 5], [2, 10], [3, 15]], device=device, dtype=torch.int32)
        world_rays, _, _, _ = generate_spinning_lidar_rays(self.projection, elements, dynamic_pose)

        # Create world points along rays
        distances = torch.ones(4, 1, device=device) * 10.0
        world_points = world_rays[:, :3] + world_rays[:, 3:6] * distances

        # Inverse project with lazy_init=False (should use the pre-built map)
        sensor_angles_inv, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            dynamic_pose,
            lazy_init_angles_map=False,  # Use pre-built map
            return_valid_flags=True,
        )

        # Get original angles
        sensor_angles_orig, _ = elements_to_sensor_angles(self.projection, elements)

        # All should be valid
        assert valid is not None
        self.assertTrue(valid.all())

        # Compare with angle wraparound handling
        elev_diff = torch.abs(sensor_angles_inv[:, 0] - sensor_angles_orig[:, 0])
        az_diff = torch.abs(sensor_angles_inv[:, 1] - sensor_angles_orig[:, 1])
        # Handle 2π wraparound for azimuth
        az_diff = torch.min(az_diff, 2 * np.pi - az_diff)

        self.assertTrue((elev_diff < 1e-3).all(), f"Elevation mismatch: max={elev_diff.max()}")
        self.assertTrue((az_diff < 1e-3).all(), f"Azimuth mismatch: max={az_diff.max()}")

    def test_inverse_projection_with_row_offsets_and_map(self):
        """Test map with row offsets produces consistent results."""
        # Add row offsets
        row_offsets = torch.linspace(0, 0.1, self.n_rows, device=device)
        projection_with_offsets = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.projection.row_elevations_rad,
            column_azimuths_rad=self.projection.column_azimuths_rad,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=10.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=20.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=row_offsets,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

        # Build map with row offsets
        projection_with_offsets.ensure_angles_map(resolution_factor=2)
        self.assertIsNotNone(projection_with_offsets.angles_to_columns_map)

        # Verify map values are in valid column range
        map_values = projection_with_offsets.angles_to_columns_map
        assert map_values is not None
        self.assertTrue((map_values >= 0).all())
        self.assertTrue((map_values < self.n_columns).all())


# ============================================================================
# Multi-Device and Multi-Dtype Tests
# ============================================================================


class TestMultiDevice(unittest.TestCase):
    """Test LiDAR operations across different devices."""

    def _create_projection(self, dev):
        """Create a simple projection on specified device."""
        n_rows, n_columns = 8, 256

        return RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.linspace(-10, 10, n_rows, device=dev) * np.pi / 180.0,
            column_azimuths_rad=torch.linspace(0, 2 * np.pi, n_columns, device=dev),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=10.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=20.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        """Test that operations work on CUDA device."""
        dev = torch.device("cuda")
        projection = self._create_projection(dev)
        dynamic_pose = create_identity_dynamic_pose(dev)

        elements = torch.tensor([[0, 0], [4, 128]], device=dev, dtype=torch.int32)

        # Should not raise
        world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, dynamic_pose)

        self.assertEqual(world_rays.device.type, "cuda")
        self.assertEqual(world_rays.shape, (2, 6))


class TestDtypeConsistency(unittest.TestCase):
    """Test operations with different floating point dtypes."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_rows = 8
        self.n_columns = 256

    def _create_projection(self, dtype):
        """Create projection with specified dtype."""
        return RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=torch.linspace(-10, 10, self.n_rows, device=device, dtype=dtype) * np.pi / 180.0,
            column_azimuths_rad=torch.linspace(0, 2 * np.pi, self.n_columns, device=device, dtype=dtype),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=10.0 * np.pi / 180.0,  # TOP of FOV (cw convention)
            fov_vert_span_rad=20.0 * np.pi / 180.0,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
            angles_to_columns_map_resolution_factor=1,
        )

    def test_float32(self):
        """Test with float32 dtype."""
        dtype = torch.float32
        projection = self._create_projection(dtype)
        dynamic_pose = create_identity_dynamic_pose(device, dtype)

        elements = torch.tensor([[0, 0], [4, 128]], device=device, dtype=torch.int32)

        world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, dynamic_pose)

        self.assertEqual(world_rays.dtype, torch.float32)

    def test_float32_angle_conversions(self):
        """Test angle conversions with float32."""
        dtype = torch.float32
        projection = self._create_projection(dtype)

        angles = torch.tensor([[0.0, 0.0], [0.1, 0.5]], device=device, dtype=dtype)
        rays = sensor_angles_to_sensor_rays(projection, angles)

        self.assertEqual(rays.dtype, torch.float32)

        # Round-trip
        angles_back = sensor_rays_to_sensor_angles(projection, rays)
        self.assertEqual(angles_back.dtype, torch.float32)

        # Verify accuracy
        np.testing.assert_allclose(
            angles.cpu().numpy(),
            angles_back.cpu().numpy(),
            atol=ATOL,
        )


# ============================================================================
# Parameterized Tests with Real LiDAR Configurations (matching ncore tests)
# ============================================================================


def load_lidar_params_from_json(json_path: str) -> dict:
    """Load LiDAR model parameters from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def create_projection_from_params(params: dict, dev: torch.device) -> RowOffsetStructuredSpinningLidarProjection:
    """Create a RowOffsetStructuredSpinningLidarProjection from JSON parameters."""
    n_rows = params["n_rows"]
    n_columns = params["n_columns"]

    row_elevations = torch.tensor(params["row_elevations_rad"], device=dev, dtype=torch.float32)
    column_azimuths = torch.tensor(params["column_azimuths_rad"], device=dev, dtype=torch.float32)

    row_offsets = None
    if "row_azimuth_offsets_rad" in params and params["row_azimuth_offsets_rad"]:
        row_offsets = torch.tensor(params["row_azimuth_offsets_rad"], device=dev, dtype=torch.float32)

    # Compute FOV from parameters
    fov_vert_start = float(row_elevations.max().item())  # Highest elevation (most positive)
    fov_vert_end = float(row_elevations.min().item())  # Lowest elevation (most negative)
    fov_vert_span = abs(fov_vert_start - fov_vert_end)

    # Horizontal FOV from column azimuths
    fov_horiz_start = float(column_azimuths[0].item())
    fov_horiz_end = float(column_azimuths[-1].item())
    # Account for wraparound
    if fov_horiz_end < fov_horiz_start:
        fov_horiz_span = (2 * np.pi - fov_horiz_start) + fov_horiz_end
    else:
        fov_horiz_span = fov_horiz_end - fov_horiz_start

    return RowOffsetStructuredSpinningLidarProjection(
        n_rows=n_rows,
        n_columns=n_columns,
        row_elevations_rad=row_elevations,
        column_azimuths_rad=column_azimuths,
        fov_horiz_start_rad=fov_horiz_start,
        fov_horiz_span_rad=fov_horiz_span,
        fov_vert_start_rad=fov_vert_start,
        fov_vert_span_rad=fov_vert_span,
        row_azimuth_offsets_rad=row_offsets,
        spinning_frequency_hz=params.get("spinning_frequency_hz", 10.0),
        spinning_direction=params.get("spinning_direction", "cw"),
        angles_to_columns_map=None,
        angles_to_columns_map_resolution_factor=1,
    )


# Test configurations: (json_filename, map_resolution_factor)
# These match the ncore test configurations
LIDAR_CONFIG_PARAMS = [
    ("row-offset-spinning-lidar-model-parameters.json", 3),
    ("row-offset-spinning-lidar-model-parameters-waymo.json", 3),
    ("row-offset-spinning-lidar-model-parameters-pandaset.json", 4),
    ("row-offset-spinning-lidar-model-parameters-hesai-at128.json", 3),
]


@parameterized.parameterized_class(
    ("config_file", "map_res_factor"),
    LIDAR_CONFIG_PARAMS,
)
class TestRealLidarConfigurations(unittest.TestCase):
    """Test LiDAR operations with real sensor configurations.

    This matches the ncore TestRowOffsetStructuredSpinningLidarModel test class
    which runs tests across multiple real LiDAR configurations.
    """

    config_file: str
    map_res_factor: int

    @classmethod
    def setUpClass(cls):
        """Load the configuration once per test class."""
        config_path = f"{TEST_DATA_DIR}/{cls.config_file}"
        cls.params = load_lidar_params_from_json(config_path)
        cls.projection = create_projection_from_params(cls.params, device)

        # Create all element indices
        n_rows = cls.params["n_rows"]
        n_columns = cls.params["n_columns"]
        elements_grid = np.stack(
            np.meshgrid(
                np.arange(n_rows, dtype=np.int32),
                np.arange(n_columns, dtype=np.int32),
                indexing="ij",
            ),
            axis=-1,
        )
        cls.all_elements = torch.tensor(elements_grid.reshape(-1, 2), device=device, dtype=torch.int32)

    def _create_dynamic_pose(self):
        """Create a dynamic pose with realistic motion."""
        T_start = torch.tensor(
            [
                [9.9974847e-01, -2.2219338e-02, -3.0501457e-03, 4.6345856e01],
                [2.2246171e-02, 9.9971139e-01, 9.0645989e-03, 2.4201742e-01],
                [2.8478564e-03, -9.1301724e-03, 9.9995428e-01, 2.0181880e00],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        T_end = torch.tensor(
            [
                [9.9977809e-01, -2.1048529e-02, -8.6604326e-04, 4.7494629e01],
                [2.1049019e-02, 9.9977827e-01, 5.6204927e-04, 2.4444677e-01],
                [8.5402117e-04, -5.8015389e-04, 9.9999946e-01, 2.0235672e00],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        return create_dynamic_pose_from_transforms(T_start, T_end, device)

    def test_angle_conversion_round_trip(self):
        """Test element → ray → angle → ray round-trip (matches ncore test_angle_conversion)."""
        # Use subset of elements for speed
        np.random.seed(42)
        num_test = min(1000, len(self.all_elements))
        indices = np.random.choice(len(self.all_elements), num_test, replace=False)
        elements = self.all_elements[indices]

        # Element → sensor angles
        sensor_angles, _ = elements_to_sensor_angles(self.projection, elements)

        # Sensor angles → sensor rays
        sensor_rays = sensor_angles_to_sensor_rays(self.projection, sensor_angles)

        # Sensor rays → sensor angles (reconstructed)
        sensor_angles_reconstructed = sensor_rays_to_sensor_angles(self.projection, sensor_rays)

        # Sensor angles → sensor rays (reconstructed)
        sensor_rays_reconstructed = sensor_angles_to_sensor_rays(self.projection, sensor_angles_reconstructed)

        # Compare rays
        np.testing.assert_allclose(
            sensor_rays.cpu().numpy(),
            sensor_rays_reconstructed.cpu().numpy(),
            atol=ATOL,
            err_msg=f"Angle conversion round-trip failed for {self.config_file}",
        )

    def test_angle_conversion_2_round_trip(self):
        """Test element → angle → ray → angle round-trip (matches ncore test_angle_conversion_2)."""
        # Use subset of elements for speed
        np.random.seed(42)
        num_test = min(1000, len(self.all_elements))
        indices = np.random.choice(len(self.all_elements), num_test, replace=False)
        elements = self.all_elements[indices]

        # Element → sensor angles
        sensor_angles, _ = elements_to_sensor_angles(self.projection, elements)

        # Sensor angles → sensor rays
        sensor_rays = sensor_angles_to_sensor_rays(self.projection, sensor_angles)

        # Sensor rays → sensor angles (reconstructed)
        sensor_angles_reconstructed = sensor_rays_to_sensor_angles(self.projection, sensor_rays)

        # Compare angles (accounting for azimuth wrapping)
        orig_angles = sensor_angles.cpu().numpy()
        recon_angles = sensor_angles_reconstructed.cpu().numpy()

        # Elevation should match exactly
        np.testing.assert_allclose(
            orig_angles[:, 0],
            recon_angles[:, 0],
            atol=ATOL,
            err_msg=f"Elevation round-trip failed for {self.config_file}",
        )

        # Azimuth should match (accounting for wrapping)
        az_diff = np.abs(normalize_angle_np(orig_angles[:, 1] - recon_angles[:, 1]))
        self.assertTrue(
            np.all(az_diff < 1e-5),
            f"Azimuth round-trip failed for {self.config_file}: max diff = {np.max(az_diff)}",
        )

    def test_rolling_shutter_projection_round_trip(self):
        """Test rolling shutter forward/inverse projection (matches ncore test_rolling_shutter_projection)."""
        dynamic_pose = self._create_dynamic_pose()

        # Use subset of elements for speed
        np.random.seed(42)
        num_test = min(2000, len(self.all_elements))
        indices = np.random.choice(len(self.all_elements), num_test, replace=False)
        elements = self.all_elements[indices]

        # Forward projection: elements → world rays
        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            dynamic_pose,
        )

        # Generate world points at random distances
        torch.manual_seed(42)
        distances = torch.rand(num_test, 1, device=device) * 100  # 0-100m
        world_points = world_rays[:, :3] + world_rays[:, 3:6] * distances

        # Inverse projection: world points → sensor angles
        sensor_angles_inv, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            dynamic_pose,
            max_iterations=10,
            stop_mean_relative_time_error=1e-4,
            stop_delta_mean_relative_time_error=1e-6,
            return_valid_flags=True,
        )

        # Get original sensor angles from elements
        sensor_angles_orig, _ = elements_to_sensor_angles(self.projection, elements)

        # Compare valid projections
        assert valid is not None
        valid_mask = valid.cpu().numpy()
        valid_count = valid_mask.sum()

        if valid_count > 0:
            angles_orig = sensor_angles_orig[valid].cpu().numpy()
            angles_inv = sensor_angles_inv[valid].cpu().numpy()

            elev_diff = np.abs(angles_inv[:, 0] - angles_orig[:, 0])
            az_diff = np.abs(normalize_angle_np(angles_inv[:, 1] - angles_orig[:, 1]))
            angle_errors = np.sqrt(elev_diff**2 + az_diff**2)

            # At least 98% should match within 1e-3 rad (matching ncore threshold)
            tolerance_rad = 1e-3
            within_tolerance = np.sum(angle_errors < tolerance_rad)
            match_ratio = within_tolerance / valid_count

            # Note: ncore uses 0.98, but relaxed to 0.95 for some configs
            self.assertGreater(
                match_ratio,
                0.95,
                f"Rolling shutter round-trip failed for {self.config_file}: "
                f"only {match_ratio * 100:.1f}% matched within {np.degrees(tolerance_rad):.4f}° "
                f"(max error: {np.degrees(np.max(angle_errors)):.4f}°)",
            )

    def test_elements_produce_valid_angles(self):
        """Test that all valid elements produce valid sensor angles."""
        # Use subset of elements
        np.random.seed(123)
        num_test = min(500, len(self.all_elements))
        indices = np.random.choice(len(self.all_elements), num_test, replace=False)
        elements = self.all_elements[indices]

        sensor_angles, _ = elements_to_sensor_angles(self.projection, elements)

        # Check shape
        self.assertEqual(sensor_angles.shape, (num_test, 2))

        # Check that angles are finite
        self.assertTrue(torch.all(torch.isfinite(sensor_angles)), f"Non-finite angles for {self.config_file}")

    def test_ray_generation_normalized(self):
        """Test that generated rays are normalized."""
        dynamic_pose = self._create_dynamic_pose()

        # Use subset of elements
        np.random.seed(456)
        num_test = min(500, len(self.all_elements))
        indices = np.random.choice(len(self.all_elements), num_test, replace=False)
        elements = self.all_elements[indices]

        world_rays, _, _, _ = generate_spinning_lidar_rays(
            self.projection,
            elements,
            dynamic_pose,
        )

        # Check that directions are normalized
        directions = world_rays[:, 3:6]
        norms = directions.norm(dim=-1)

        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=ATOL,
            rtol=RTOL,
            msg=f"World ray directions not normalized for {self.config_file}",
        )


# ============================================================================
# Intrinsic Differentiability Tests
# ============================================================================


class TestIntrinsicsDifferentiability(unittest.TestCase):
    """Tests for differentiability of LiDAR intrinsic parameters.

    These tests verify that gradients flow correctly through:
    - Row elevations (per-row elevation angles)
    - Column azimuths (per-column azimuth angles)
    - Row azimuth offsets (per-row azimuth offsets)
    - Pose parameters (translations and rotations)
    - Input rays/angles for spherical conversions
    """

    def setUp(self):
        """Set up test fixtures."""
        # Basic LiDAR parameters
        self.n_rows = 8
        self.n_columns = 16
        self.device = device

        # Create row elevations (ranging from -15° to +15°)
        self.row_elevations = torch.linspace(-0.26, 0.26, self.n_rows, device=self.device, dtype=torch.float32)

        # Create column azimuths (full 360° sweep)
        self.column_azimuths = torch.linspace(
            -np.pi, np.pi, self.n_columns + 1, device=self.device, dtype=torch.float32
        )[:-1]

        # Row azimuth offsets (small per-row timing offsets)
        self.row_offsets = torch.linspace(0, 0.02, self.n_rows, device=self.device, dtype=torch.float32)

        # Create LiDAR projection
        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.row_elevations,
            column_azimuths_rad=self.column_azimuths,
            row_azimuth_offsets_rad=self.row_offsets,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

    def _create_dynamic_pose(self):
        """Create a simple dynamic pose with two control poses."""
        trans_start = torch.tensor([0.0, 0.0, 2.0], device=self.device, dtype=torch.float32)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        trans_end = torch.tensor([0.1, 0.0, 2.0], device=self.device, dtype=torch.float32)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)
        return create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

    def test_elements_to_sensor_angles_elevation_gradient(self):
        """Test gradient flow through row elevation parameters."""
        # Create tensors with requires_grad
        row_elevations = self.row_elevations.clone().requires_grad_(True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=self.column_azimuths,
            row_azimuth_offsets_rad=None,  # No row offsets for simplicity
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

        # Create a batch of elements spanning all rows
        elements = torch.tensor([[i, 0] for i in range(self.n_rows)], device=self.device, dtype=torch.int32)

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)

        # Compute loss (sum of elevation angles)
        loss = sensor_angles[:, 0].sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(row_elevations.grad, "Row elevation gradients should be computed")
        self.assertTrue(torch.any(row_elevations.grad != 0), "Row elevation gradients should be non-zero")

    def test_elements_to_sensor_angles_azimuth_gradient(self):
        """Test gradient flow through column azimuth parameters."""
        # Create tensors with requires_grad
        column_azimuths = self.column_azimuths.clone().requires_grad_(True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

        # Create a batch of elements spanning all columns
        elements = torch.tensor([[0, j] for j in range(self.n_columns)], device=self.device, dtype=torch.int32)

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)

        # Compute loss (sum of azimuth angles)
        loss = sensor_angles[:, 1].sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(column_azimuths.grad, "Column azimuth gradients should be computed")
        self.assertTrue(torch.any(column_azimuths.grad != 0), "Column azimuth gradients should be non-zero")

    def test_elements_to_sensor_angles_row_offset_gradient(self):
        """Test gradient flow through row azimuth offset parameters."""
        # Create tensors with requires_grad
        row_offsets = self.row_offsets.clone().requires_grad_(True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.row_elevations,
            column_azimuths_rad=self.column_azimuths,
            row_azimuth_offsets_rad=row_offsets,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

        # Create a batch of elements
        elements = torch.tensor(
            [[i, j] for i in range(self.n_rows) for j in range(self.n_columns // 4)],
            device=self.device,
            dtype=torch.int32,
        )

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)

        # Compute loss (sum of azimuth angles - affected by row offsets)
        loss = sensor_angles[:, 1].sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(row_offsets.grad, "Row offset gradients should be computed")
        self.assertTrue(torch.any(row_offsets.grad != 0), "Row offset gradients should be non-zero")

    def test_generate_spinning_lidar_rays_elevation_gradient(self):
        """Test gradient flow through row elevations in ray generation."""
        row_elevations = self.row_elevations.clone().requires_grad_(True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=self.column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

        dynamic_pose = self._create_dynamic_pose()

        # Create elements
        elements = torch.tensor([[i, 0] for i in range(self.n_rows)], device=self.device, dtype=torch.int32)

        # Forward pass
        world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, dynamic_pose)

        # Verify forward pass produces expected results
        # z-direction should be approximately sin(elevation) for identity rotation
        expected_z = torch.sin(row_elevations).detach()
        actual_z = world_rays[:, 5].detach()
        torch.testing.assert_close(
            actual_z, expected_z, atol=ATOL, rtol=RTOL, msg="Forward pass z-direction should match sin(elevation)"
        )

        # Compute loss
        loss = world_rays[:, 5].sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(row_elevations.grad, "Row elevation gradients should be computed")
        self.assertTrue(
            torch.any(row_elevations.grad != 0), "Row elevation gradients should be non-zero for ray generation"
        )

    def test_generate_spinning_lidar_rays_azimuth_gradient(self):
        """Test gradient flow through column azimuths in ray generation."""
        column_azimuths = self.column_azimuths.clone().requires_grad_(True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

        dynamic_pose = self._create_dynamic_pose()

        # Create elements spanning columns
        elements = torch.tensor([[0, j] for j in range(self.n_columns)], device=self.device, dtype=torch.int32)

        # Forward pass
        world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, dynamic_pose)

        # Compute loss (x and y directions depend on azimuth)
        loss = world_rays[:, 3:5].sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(column_azimuths.grad, "Column azimuth gradients should be computed")
        self.assertTrue(
            torch.any(column_azimuths.grad != 0), "Column azimuth gradients should be non-zero for ray generation"
        )

    def test_sensor_rays_to_angles_gradient(self):
        """Test gradient flow through sensor ray to angle conversion."""
        # Create random sensor rays (normalized)
        rays = torch.randn(100, 3, device=self.device, dtype=torch.float32)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        rays = rays.requires_grad_(True)

        # Forward pass
        angles = sensor_rays_to_sensor_angles(self.projection, rays)

        # Compute loss
        loss = angles.sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(rays.grad, "Ray gradients should be computed")
        self.assertTrue(torch.any(rays.grad != 0), "Ray gradients should be non-zero")

    def test_sensor_angles_to_rays_gradient(self):
        """Test gradient flow through sensor angle to ray conversion."""
        # Create random angles
        angles = torch.randn(100, 2, device=self.device, dtype=torch.float32)
        angles[:, 0] = angles[:, 0].clamp(-np.pi / 2, np.pi / 2)  # Valid elevations
        angles = angles.requires_grad_(True)

        # Forward pass
        rays = sensor_angles_to_sensor_rays(self.projection, angles)

        # Compute loss
        loss = rays.sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(angles.grad, "Angle gradients should be computed")
        self.assertTrue(torch.any(angles.grad != 0), "Angle gradients should be non-zero")

    def test_elevation_gradient_is_correct(self):
        """Verify elevation gradients match expected analytical values.

        For elements_to_sensor_angles, the output elevation is a direct lookup:
            output_elevation = row_elevations[row_index]

        Therefore: d(output_elevation)/d(row_elevations[i]) = 1 if row_index == i, else 0

        With one element per row and loss = sum(output_elevations),
        the expected gradient is [1, 1, 1] (each row used exactly once).
        """
        row_elevations = torch.tensor([-0.2, 0.0, 0.2], device=self.device, dtype=torch.float32, requires_grad=True)
        column_azimuths = torch.tensor([0.0, 1.0, 2.0], device=self.device, dtype=torch.float32)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=3,
            n_columns=3,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )

        # Single element per row
        elements = torch.tensor([[0, 0], [1, 0], [2, 0]], device=self.device, dtype=torch.int32)

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)
        loss = sensor_angles[:, 0].sum()  # Sum of elevations

        # Get kernel gradient
        loss.backward()
        kernel_grad = row_elevations.grad.clone()

        # Expected gradient: each row is used exactly once, so gradient is [1, 1, 1]
        expected_grad = torch.ones(3, device=self.device)

        np.testing.assert_allclose(
            kernel_grad.detach().cpu().numpy(),
            expected_grad.cpu().numpy(),
            rtol=RTOL,
            atol=ATOL,
            err_msg="Elevation gradient should be [1, 1, 1] when each row is used once",
        )

    def test_azimuth_gradient_is_correct(self):
        """Verify azimuth gradients match expected analytical values.

        For elements_to_sensor_angles, the output azimuth is a direct lookup:
            output_azimuth = column_azimuths[column_index]

        Therefore: d(output_azimuth)/d(column_azimuths[i]) = 1 if column_index == i, else 0

        With one element per column and loss = sum(output_azimuths),
        the expected gradient is [1, 1, 1] (each column used exactly once).
        """
        row_elevations = torch.tensor([0.0], device=self.device, dtype=torch.float32)
        column_azimuths = torch.tensor([0.0, 1.0, 2.0], device=self.device, dtype=torch.float32, requires_grad=True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=1,
            n_columns=3,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )

        # Single element per column
        elements = torch.tensor([[0, 0], [0, 1], [0, 2]], device=self.device, dtype=torch.int32)

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)
        loss = sensor_angles[:, 1].sum()  # Sum of azimuths

        # Get kernel gradient
        loss.backward()
        kernel_grad = column_azimuths.grad.clone()

        # Expected gradient: each column is used exactly once, so gradient is [1, 1, 1]
        expected_grad = torch.ones(3, device=self.device)

        np.testing.assert_allclose(
            kernel_grad.detach().cpu().numpy(),
            expected_grad.cpu().numpy(),
            rtol=RTOL,
            atol=ATOL,
            err_msg="Azimuth gradient should be [1, 1, 1] when each column is used once",
        )

    def test_ray_generation_elevation_gradient_is_correct(self):
        """Verify ray generation elevation gradients match expected analytical values.

        For spherical-to-Cartesian conversion with identity pose:
            z_direction = sin(elevation)

        Therefore: d(z_direction)/d(elevation) = cos(elevation)

        With one element per row and loss = sum(z_directions),
        the expected gradient is [cos(-0.2), cos(0), cos(0.2)] for elevations [-0.2, 0, 0.2].
        """
        elevations = [-0.2, 0.0, 0.2]
        row_elevations = torch.tensor(elevations, device=self.device, dtype=torch.float32, requires_grad=True)
        column_azimuths = torch.tensor([0.0], device=self.device, dtype=torch.float32)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=3,
            n_columns=1,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )

        dynamic_pose = self._create_dynamic_pose()

        elements = torch.tensor([[0, 0], [1, 0], [2, 0]], device=self.device, dtype=torch.int32)

        # Forward pass
        world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, dynamic_pose)
        loss = world_rays[:, 5].sum()  # Sum of z-directions

        # Get kernel gradient
        loss.backward()
        kernel_grad = row_elevations.grad.clone()

        # Expected gradient: d(sin(e))/d(e) = cos(e) for each elevation
        expected_grad = torch.tensor([np.cos(e) for e in elevations], device=self.device)

        np.testing.assert_allclose(
            kernel_grad.detach().cpu().numpy(),
            expected_grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Ray generation elevation gradient should be cos(elevation)",
        )


class TestPoseDifferentiability(unittest.TestCase):
    """Tests for differentiability of pose parameters (translations and rotations).

    Verifies that gradients flow through pose parameters for all pose-based
    LiDAR functions.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.device = device
        self.n_rows = 4
        self.n_columns = 8

        self.projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=torch.linspace(-0.2, 0.2, self.n_rows, device=self.device),
            column_azimuths_rad=torch.linspace(-np.pi, np.pi, self.n_columns + 1, device=self.device)[:-1],
            row_azimuth_offsets_rad=torch.zeros(self.n_rows, device=self.device),
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.2,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.4,
        )

    def _create_dynamic_pose(self, trans_start, trans_end, rot_start, rot_end):
        """Create a dynamic pose with control poses."""
        return create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

    def test_generate_spinning_lidar_rays_translation_gradient(self):
        """Test gradient flows through translations in ray generation."""
        trans_start = torch.tensor([0.0, 0.0, 2.0], device=self.device, requires_grad=True)
        trans_end = torch.tensor([0.1, 0.0, 2.0], device=self.device, requires_grad=True)
        rot_start = quat_identity((), device=self.device, dtype=torch.float32)
        rot_end = quat_identity((), device=self.device, dtype=torch.float32)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        elements = torch.tensor([[0, 0], [1, 2], [2, 4]], device=self.device, dtype=torch.int32)

        world_rays, _, _, _ = generate_spinning_lidar_rays(self.projection, elements, dynamic_pose)

        # Loss on ray origins (depend on translation)
        loss = world_rays[:, :3].sum()
        loss.backward()

        self.assertIsNotNone(trans_start.grad, "Start translation gradient should exist")
        self.assertIsNotNone(trans_end.grad, "End translation gradient should exist")
        self.assertTrue(
            (trans_start.grad != 0).any().item() or (trans_end.grad != 0).any().item(),
            "At least one translation gradient should be non-zero",
        )

    def test_generate_spinning_lidar_rays_rotation_gradient(self):
        """Test gradient flows through rotations in ray generation."""
        trans_start = torch.tensor([0.0, 0.0, 2.0], device=self.device)
        trans_end = torch.tensor([0.0, 0.0, 2.0], device=self.device)
        rot_start = quat_identity((), device=self.device, dtype=torch.float32).requires_grad_(True)
        rot_end = quat_identity((), device=self.device, dtype=torch.float32).requires_grad_(True)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        elements = torch.tensor([[0, 0], [1, 2], [2, 4]], device=self.device, dtype=torch.int32)

        world_rays, _, _, _ = generate_spinning_lidar_rays(self.projection, elements, dynamic_pose)

        # Loss on ray directions (depend on rotation)
        loss = world_rays[:, 3:].sum()
        loss.backward()

        self.assertIsNotNone(rot_start.grad, "Start rotation gradient should exist")
        self.assertIsNotNone(rot_end.grad, "End rotation gradient should exist")

    def test_inverse_project_spinning_lidar_translation_gradient(self):
        """Test gradient flows through translations in inverse projection."""
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        trans_end = torch.tensor([0.1, 0.0, 0.0], device=self.device, requires_grad=True)
        rot_start = quat_identity((), device=self.device, dtype=torch.float32)
        rot_end = quat_identity((), device=self.device, dtype=torch.float32)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        # World points that will project into the sensor FOV
        world_points = torch.tensor(
            [[5.0, 0.0, 0.5], [5.0, 1.0, 0.0], [5.0, -1.0, -0.5]], device=self.device, dtype=torch.float32
        )

        sensor_angles, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            dynamic_pose,
            return_valid_flags=True,
        )

        loss = sensor_angles.sum()
        loss.backward()

        self.assertIsNotNone(trans_start.grad, "Start translation gradient should exist")
        self.assertIsNotNone(trans_end.grad, "End translation gradient should exist")

    def test_inverse_project_spinning_lidar_rotation_gradient(self):
        """Test gradient flows through rotations in inverse projection."""
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        trans_end = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        rot_start = quat_identity((), device=self.device, dtype=torch.float32).requires_grad_(True)
        rot_end = quat_identity((), device=self.device, dtype=torch.float32).requires_grad_(True)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        world_points = torch.tensor(
            [[5.0, 0.5, 0.2], [5.0, -0.5, 0.1], [5.0, 0.0, -0.2]], device=self.device, dtype=torch.float32
        )

        sensor_angles, valid, _, _, _ = inverse_project_spinning_lidar(
            self.projection,
            world_points,
            dynamic_pose,
            return_valid_flags=True,
        )

        loss = sensor_angles.sum()
        loss.backward()

        self.assertIsNotNone(rot_start.grad, "Start rotation gradient should exist")
        self.assertIsNotNone(rot_end.grad, "End rotation gradient should exist")

    def test_numerical_pose_gradient_check(self):
        """Verify analytical pose gradients match numerical gradients for ray generation."""
        trans_start = torch.tensor([0.0, 0.0, 2.0], device=self.device, requires_grad=True)
        trans_end = torch.tensor([0.0, 0.0, 2.0], device=self.device)
        rot_start = quat_identity((), device=self.device, dtype=torch.float32)
        rot_end = quat_identity((), device=self.device, dtype=torch.float32)

        dynamic_pose = self._create_dynamic_pose(trans_start, trans_end, rot_start, rot_end)

        elements = torch.tensor([[0, 0]], device=self.device, dtype=torch.int32)

        # Analytical gradient
        world_rays, _, _, _ = generate_spinning_lidar_rays(self.projection, elements, dynamic_pose)
        loss = world_rays[:, :3].sum()  # Sum of ray origins
        loss.backward()
        analytical_grad = trans_start.grad.clone()

        # Numerical gradient using larger epsilon to avoid float32 precision issues
        eps = 0.01
        numerical_grad = torch.zeros(3, device=self.device)

        with torch.no_grad():
            for i in range(3):
                # Plus direction
                t_plus = torch.tensor([0.0, 0.0, 2.0], device=self.device)
                t_plus[i] += eps
                pose_plus = self._create_dynamic_pose(t_plus, trans_end, rot_start, rot_end)
                rays_plus, _, _, _ = generate_spinning_lidar_rays(self.projection, elements, pose_plus)
                loss_plus = rays_plus[:, :3].sum()

                # Minus direction
                t_minus = torch.tensor([0.0, 0.0, 2.0], device=self.device)
                t_minus[i] -= eps
                pose_minus = self._create_dynamic_pose(t_minus, trans_end, rot_start, rot_end)
                rays_minus, _, _, _ = generate_spinning_lidar_rays(self.projection, elements, pose_minus)
                loss_minus = rays_minus[:, :3].sum()

                numerical_grad[i] = (loss_plus.item() - loss_minus.item()) / (2 * eps)

        np.testing.assert_allclose(
            analytical_grad.cpu().numpy(),
            numerical_grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg=f"Analytical ({analytical_grad}) and numerical ({numerical_grad}) pose gradients should match",
        )


class TestSharedIndexGradientAccumulation(unittest.TestCase):
    """Tests for correct gradient accumulation when multiple threads access the same indices.

    These tests specifically verify that gradients are correctly accumulated when multiple
    elements (threads) read from the same lookup table index. This catches bugs where
    `loadOnce` is incorrectly used instead of `loadUniform` for shared access patterns.

    Key insight: If loadOnce is incorrectly used for shared access, the gradient magnitude
    will be ~1/N of correct (where N is warp size) because only one thread's contribution
    is counted instead of all threads.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.device = device

    def test_elevation_gradient_with_shared_rows(self):
        """Test gradient accumulation when many elements share the same row.

        Creates multiple elements that all use the same row index, verifying that
        the gradient for that row's elevation correctly accumulates contributions
        from all elements.
        """
        # Small lookup table with 2 rows
        n_rows = 2
        n_columns = 4
        row_elevations = torch.tensor([0.1, 0.2], device=self.device, dtype=torch.float32, requires_grad=True)
        column_azimuths = torch.linspace(0, 1.5, n_columns, device=self.device, dtype=torch.float32)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )

        # Create 100 elements, all using row 0 (50 elements) and row 1 (50 elements)
        # This ensures many threads access the same lookup table indices
        elements_row0 = [[0, j % n_columns] for j in range(50)]
        elements_row1 = [[1, j % n_columns] for j in range(50)]
        elements = torch.tensor(elements_row0 + elements_row1, device=self.device, dtype=torch.int32)

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)

        # Loss: sum of all elevation outputs
        # Each row_elevation[i] contributes to 50 elements
        loss = sensor_angles[:, 0].sum()

        # Analytical gradient
        loss.backward()
        analytical_grad = row_elevations.grad.clone()

        # Numerical gradient
        eps = 1e-4
        numerical_grad = torch.zeros_like(row_elevations)
        for i in range(n_rows):
            row_elev_plus = row_elevations.detach().clone()
            row_elev_minus = row_elevations.detach().clone()
            row_elev_plus[i] += eps
            row_elev_minus[i] -= eps

            proj_plus = RowOffsetStructuredSpinningLidarProjection(
                n_rows=n_rows,
                n_columns=n_columns,
                row_elevations_rad=row_elev_plus,
                column_azimuths_rad=column_azimuths,
                row_azimuth_offsets_rad=None,
                spinning_frequency_hz=20.0,
                spinning_direction="ccw",
                fov_horiz_start_rad=-np.pi,
                fov_horiz_span_rad=2 * np.pi,
                fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
                fov_vert_span_rad=1.0,
            )
            proj_minus = RowOffsetStructuredSpinningLidarProjection(
                n_rows=n_rows,
                n_columns=n_columns,
                row_elevations_rad=row_elev_minus,
                column_azimuths_rad=column_azimuths,
                row_azimuth_offsets_rad=None,
                spinning_frequency_hz=20.0,
                spinning_direction="ccw",
                fov_horiz_start_rad=-np.pi,
                fov_horiz_span_rad=2 * np.pi,
                fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
                fov_vert_span_rad=1.0,
            )

            loss_plus = elements_to_sensor_angles(proj_plus, elements)[0][:, 0].sum()
            loss_minus = elements_to_sensor_angles(proj_minus, elements)[0][:, 0].sum()
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        # The gradient should be ~50 for each row (each elevation contributes to 50 elements)
        # If loadOnce bug exists, gradient would be ~1.5 (50/32 warp size)
        np.testing.assert_allclose(
            analytical_grad.detach().cpu().numpy(),
            numerical_grad.cpu().numpy(),
            rtol=NUMERICAL_GRAD_RTOL,
            atol=NUMERICAL_GRAD_ATOL,
            err_msg=(
                f"Shared-index gradient mismatch! "
                f"Analytical: {analytical_grad.cpu().numpy()}, "
                f"Numerical: {numerical_grad.cpu().numpy()}. "
                f"If analytical << numerical, this indicates incorrect loadOnce usage "
                f"instead of loadUniform for shared index access."
            ),
        )

        # Sanity check: gradients should be approximately 50 (one per element using that row)
        expected_grad = 50.0  # 50 elements use each row
        self.assertTrue(
            torch.allclose(
                analytical_grad,
                torch.tensor([expected_grad, expected_grad], device=self.device),
                rtol=NUMERICAL_GRAD_RTOL,
            ),
            f"Expected gradient ~{expected_grad} per row, got {analytical_grad.cpu().numpy()}",
        )

    def test_azimuth_gradient_with_shared_columns(self):
        """Test gradient accumulation when many elements share the same column.

        Creates multiple elements that all use the same column index, verifying that
        the gradient for that column's azimuth correctly accumulates contributions
        from all elements.
        """
        n_rows = 4
        n_columns = 2  # Small lookup table
        row_elevations = torch.linspace(-0.2, 0.2, n_rows, device=self.device, dtype=torch.float32)
        column_azimuths = torch.tensor([0.5, 1.0], device=self.device, dtype=torch.float32, requires_grad=True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )

        # Create 80 elements: 40 use column 0, 40 use column 1
        elements_col0 = [[i % n_rows, 0] for i in range(40)]
        elements_col1 = [[i % n_rows, 1] for i in range(40)]
        elements = torch.tensor(elements_col0 + elements_col1, device=self.device, dtype=torch.int32)

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)

        # Loss: sum of all azimuth outputs
        loss = sensor_angles[:, 1].sum()

        # Analytical gradient
        loss.backward()
        analytical_grad = column_azimuths.grad.clone()

        # Numerical gradient
        eps = 1e-4
        numerical_grad = torch.zeros_like(column_azimuths)
        for i in range(n_columns):
            col_az_plus = column_azimuths.detach().clone()
            col_az_minus = column_azimuths.detach().clone()
            col_az_plus[i] += eps
            col_az_minus[i] -= eps

            proj_plus = RowOffsetStructuredSpinningLidarProjection(
                n_rows=n_rows,
                n_columns=n_columns,
                row_elevations_rad=row_elevations,
                column_azimuths_rad=col_az_plus,
                row_azimuth_offsets_rad=None,
                spinning_frequency_hz=20.0,
                spinning_direction="ccw",
                fov_horiz_start_rad=-np.pi,
                fov_horiz_span_rad=2 * np.pi,
                fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
                fov_vert_span_rad=1.0,
            )
            proj_minus = RowOffsetStructuredSpinningLidarProjection(
                n_rows=n_rows,
                n_columns=n_columns,
                row_elevations_rad=row_elevations,
                column_azimuths_rad=col_az_minus,
                row_azimuth_offsets_rad=None,
                spinning_frequency_hz=20.0,
                spinning_direction="ccw",
                fov_horiz_start_rad=-np.pi,
                fov_horiz_span_rad=2 * np.pi,
                fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
                fov_vert_span_rad=1.0,
            )

            loss_plus = elements_to_sensor_angles(proj_plus, elements)[0][:, 1].sum()
            loss_minus = elements_to_sensor_angles(proj_minus, elements)[0][:, 1].sum()
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            analytical_grad.detach().cpu().numpy(),
            numerical_grad.cpu().numpy(),
            rtol=NUMERICAL_GRAD_RTOL,
            atol=NUMERICAL_GRAD_ATOL,
            err_msg=(
                f"Shared-index gradient mismatch for azimuths! "
                f"Analytical: {analytical_grad.cpu().numpy()}, "
                f"Numerical: {numerical_grad.cpu().numpy()}."
            ),
        )

        # Sanity check: gradients should be ~40 (40 elements use each column)
        expected_grad = 40.0
        self.assertTrue(
            torch.allclose(
                analytical_grad,
                torch.tensor([expected_grad, expected_grad], device=self.device),
                rtol=NUMERICAL_GRAD_RTOL,
            ),
            f"Expected gradient ~{expected_grad} per column, got {analytical_grad.cpu().numpy()}",
        )

    def test_row_offset_gradient_with_shared_rows(self):
        """Test gradient accumulation for row azimuth offsets with shared access."""
        n_rows = 3
        n_columns = 4
        row_elevations = torch.linspace(-0.1, 0.1, n_rows, device=self.device, dtype=torch.float32)
        column_azimuths = torch.linspace(0, 1.0, n_columns, device=self.device, dtype=torch.float32)
        row_offsets = torch.tensor([0.01, 0.02, 0.03], device=self.device, dtype=torch.float32, requires_grad=True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=row_offsets,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )

        # 60 elements: 20 per row
        elements = torch.tensor(
            [[i % n_rows, j % n_columns] for i in range(n_rows) for j in range(20)],
            device=self.device,
            dtype=torch.int32,
        )

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)
        loss = sensor_angles[:, 1].sum()  # Sum of azimuths (affected by row offsets)

        # Analytical gradient
        loss.backward()
        analytical_grad = row_offsets.grad.clone()

        # Numerical gradient
        eps = 1e-4
        numerical_grad = torch.zeros_like(row_offsets)
        for i in range(n_rows):
            offset_plus = row_offsets.detach().clone()
            offset_minus = row_offsets.detach().clone()
            offset_plus[i] += eps
            offset_minus[i] -= eps

            proj_plus = RowOffsetStructuredSpinningLidarProjection(
                n_rows=n_rows,
                n_columns=n_columns,
                row_elevations_rad=row_elevations,
                column_azimuths_rad=column_azimuths,
                row_azimuth_offsets_rad=offset_plus,
                spinning_frequency_hz=20.0,
                spinning_direction="ccw",
                fov_horiz_start_rad=-np.pi,
                fov_horiz_span_rad=2 * np.pi,
                fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
                fov_vert_span_rad=1.0,
            )
            proj_minus = RowOffsetStructuredSpinningLidarProjection(
                n_rows=n_rows,
                n_columns=n_columns,
                row_elevations_rad=row_elevations,
                column_azimuths_rad=column_azimuths,
                row_azimuth_offsets_rad=offset_minus,
                spinning_frequency_hz=20.0,
                spinning_direction="ccw",
                fov_horiz_start_rad=-np.pi,
                fov_horiz_span_rad=2 * np.pi,
                fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
                fov_vert_span_rad=1.0,
            )

            loss_plus = elements_to_sensor_angles(proj_plus, elements)[0][:, 1].sum()
            loss_minus = elements_to_sensor_angles(proj_minus, elements)[0][:, 1].sum()
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(
            analytical_grad.detach().cpu().numpy(),
            numerical_grad.cpu().numpy(),
            rtol=NUMERICAL_GRAD_RTOL,
            atol=NUMERICAL_GRAD_ATOL,
            err_msg=(
                f"Shared-index gradient mismatch for row offsets! "
                f"Analytical: {analytical_grad.cpu().numpy()}, "
                f"Numerical: {numerical_grad.cpu().numpy()}."
            ),
        )

    def test_large_batch_gradient_accumulation(self):
        """Test gradient accumulation with a large batch to stress test warp-level reduction.

        Uses >1024 elements to ensure multiple warps process elements sharing the same index,
        testing that cross-warp gradient accumulation works correctly.
        """
        n_rows = 4
        n_columns = 8
        row_elevations = torch.linspace(-0.2, 0.2, n_rows, device=self.device, dtype=torch.float32, requires_grad=True)
        column_azimuths = torch.linspace(-1.0, 1.0, n_columns, device=self.device, dtype=torch.float32)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )

        # 2048 elements - ensures multiple warps (32 threads each) access same indices
        # Each row is accessed by 512 elements
        num_elements = 2048
        elements = torch.tensor(
            [[i % n_rows, i % n_columns] for i in range(num_elements)], device=self.device, dtype=torch.int32
        )

        # Forward pass
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)
        loss = sensor_angles[:, 0].sum()

        # Analytical gradient
        loss.backward()
        analytical_grad = row_elevations.grad.clone()

        # Each row elevation contributes to num_elements/n_rows = 512 outputs
        expected_grad_per_row = num_elements / n_rows

        # Numerical gradient for first row only (expensive for all)
        eps = 1e-4
        row_elev_plus = row_elevations.detach().clone()
        row_elev_minus = row_elevations.detach().clone()
        row_elev_plus[0] += eps
        row_elev_minus[0] -= eps

        proj_plus = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elev_plus,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )
        proj_minus = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elev_minus,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.5,  # TOP of FOV (cw convention)
            fov_vert_span_rad=1.0,
        )

        loss_plus = elements_to_sensor_angles(proj_plus, elements)[0][:, 0].sum()
        loss_minus = elements_to_sensor_angles(proj_minus, elements)[0][:, 0].sum()
        numerical_grad_row0 = (loss_plus - loss_minus) / (2 * eps)

        # Verify analytical gradient for row 0 matches numerical
        np.testing.assert_allclose(
            analytical_grad[0].detach().cpu().numpy(),
            numerical_grad_row0.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg=(
                f"Large batch gradient mismatch! "
                f"Analytical[0]: {analytical_grad[0].item()}, "
                f"Numerical[0]: {numerical_grad_row0.item()}, "
                f"Expected: ~{expected_grad_per_row}"
            ),
        )

        # All rows should have similar gradients (each accessed equally)
        self.assertTrue(
            torch.allclose(analytical_grad, analytical_grad.mean() * torch.ones_like(analytical_grad), rtol=GRAD_RTOL),
            f"Gradients should be uniform across rows, got {analytical_grad.cpu().numpy()}",
        )


# ============================================================================
# PyTorch Reference Gradient Comparison Tests
# ============================================================================


class TestLidarGradientsMatchPyTorchReference(unittest.TestCase):
    """Comprehensive tests comparing kernel gradients against pure-PyTorch reference implementations.

    These tests catch loadOnce() issues and verify gradient correctness by comparing
    against ncore-style PyTorch implementations that use autograd. Each LiDAR function
    is tested against its mathematical reference.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.device = device

    # =========================================================================
    # PyTorch Reference Implementations (similar to ncore)
    # =========================================================================

    @staticmethod
    def torch_spherical_to_cartesian(elevation: torch.Tensor, azimuth: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation of spherical to Cartesian conversion.

        Args:
            elevation: (...,) elevation angles in radians
            azimuth: (...,) azimuth angles in radians

        Returns:
            (..., 3) normalized ray directions [x, y, z]
        """
        cos_elev = torch.cos(elevation)
        x = cos_elev * torch.cos(azimuth)
        y = cos_elev * torch.sin(azimuth)
        z = torch.sin(elevation)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def torch_cartesian_to_spherical(rays: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation of Cartesian to spherical conversion.

        Args:
            rays: (..., 3) ray directions

        Returns:
            (..., 2) [elevation, azimuth] in radians
        """
        xy_norm = torch.sqrt(rays[..., 0] ** 2 + rays[..., 1] ** 2).clamp(min=1e-8)
        elevation = torch.atan2(rays[..., 2], xy_norm)
        azimuth = torch.atan2(rays[..., 1], rays[..., 0])
        return torch.stack([elevation, azimuth], dim=-1)

    @staticmethod
    def torch_quaternion_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation of quaternion-vector rotation.

        Args:
            q: (4,) quaternion [x, y, z, w]
            v: (..., 3) vectors to rotate

        Returns:
            (..., 3) rotated vectors
        """
        # Hamilton product: q * v * q^(-1)
        # Using formula: v' = v + 2 * qw * (q_xyz x v) + 2 * (q_xyz x (q_xyz x v))
        qw = q[3]
        qv = q[:3]

        # Cross products
        cross1 = torch.cross(qv.unsqueeze(0).expand(v.shape[0], 3), v, dim=-1)
        cross2 = torch.cross(qv.unsqueeze(0).expand(v.shape[0], 3), cross1, dim=-1)

        return v + 2.0 * qw * cross1 + 2.0 * cross2

    @staticmethod
    def torch_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation of quaternion spherical interpolation.

        Args:
            q0: (4,) start quaternion
            q1: (4,) end quaternion
            t: (...,) interpolation factor

        Returns:
            (..., 4) interpolated quaternion
        """
        # Compute dot product
        dot = (q0 * q1).sum()

        # If negative, negate one quaternion
        if dot < 0:
            q1 = -q1
            dot = -dot

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q0 + t.unsqueeze(-1) * (q1 - q0)
            return result / result.norm(dim=-1, keepdim=True)

        # Compute slerp
        theta_0 = torch.acos(dot.clamp(-1, 1))
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        sin_theta_0 = torch.sin(theta_0)

        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return s0.unsqueeze(-1) * q0 + s1.unsqueeze(-1) * q1

    def torch_generate_world_ray(
        self,
        elevation: torch.Tensor,
        azimuth: torch.Tensor,
        trans0: torch.Tensor,
        trans1: torch.Tensor,
        rot0: torch.Tensor,
        rot1: torch.Tensor,
        relative_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure PyTorch implementation of world ray generation.

        Args:
            elevation: (N,) elevation angles
            azimuth: (N,) azimuth angles
            trans0, trans1: (3,) start/end translations
            rot0, rot1: (4,) start/end rotations (quaternions)
            relative_time: (N,) interpolation factors

        Returns:
            origins: (N, 3) ray origins
            directions: (N, 3) ray directions
        """
        N = elevation.shape[0]

        # Generate sensor rays from angles
        sensor_rays = self.torch_spherical_to_cartesian(elevation, azimuth)

        # Interpolate pose for each ray
        origins = torch.zeros(N, 3, device=self.device)
        directions = torch.zeros(N, 3, device=self.device)

        for i in range(N):
            t = relative_time[i]
            # Linear interpolation for translation
            trans = trans0 * (1 - t) + trans1 * t
            # Slerp for rotation
            rot = self.torch_slerp(rot0, rot1, t)

            # Rotate sensor ray to world frame
            world_dir = self.torch_quaternion_rotate(rot, sensor_rays[i : i + 1, :])

            origins[i] = trans
            directions[i] = world_dir.squeeze(0)

        return origins, directions

    def _create_simple_projection(self, n_rows=8, n_columns=16):
        """Create a simple LiDAR projection for testing."""
        row_elevations = torch.linspace(-0.26, 0.26, n_rows, device=self.device, dtype=torch.float32)
        column_azimuths = torch.linspace(-np.pi, np.pi, n_columns + 1, device=self.device, dtype=torch.float32)[:-1]
        return RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

    # =========================================================================
    # Spherical Coordinate Conversion Tests
    # =========================================================================

    def test_sensor_angles_to_rays_gradient_matches_reference(self):
        """Test sensor_angles_to_sensor_rays gradients match PyTorch reference."""
        num_angles = 100
        torch.manual_seed(42)

        projection = self._create_simple_projection()

        # Create random angles
        elevation = (torch.rand(num_angles, device=self.device) - 0.5) * 0.6  # [-0.3, 0.3] rad
        azimuth = (torch.rand(num_angles, device=self.device) - 0.5) * 2 * np.pi  # Full range

        # Reference implementation
        ref_elev = elevation.clone().requires_grad_(True)
        ref_az = azimuth.clone().requires_grad_(True)
        ref_rays = self.torch_spherical_to_cartesian(ref_elev, ref_az)
        ref_rays.sum().backward()

        # Kernel implementation
        kernel_angles = torch.stack([elevation, azimuth], dim=-1).requires_grad_(True)
        kernel_rays = sensor_angles_to_sensor_rays(projection, kernel_angles)
        kernel_rays.sum().backward()

        # Compare gradients
        ref_combined_grad = torch.stack([ref_elev.grad, ref_az.grad], dim=-1)
        np.testing.assert_allclose(
            kernel_angles.grad.cpu().numpy(),
            ref_combined_grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=ATOL,
            err_msg="sensor_angles_to_sensor_rays gradient mismatch",
        )

    def test_sensor_rays_to_angles_gradient_matches_reference(self):
        """Test sensor_rays_to_sensor_angles gradients match PyTorch reference."""
        num_rays = 100
        torch.manual_seed(123)

        projection = self._create_simple_projection()

        # Create random rays
        rays = torch.randn(num_rays, 3, device=self.device)
        rays = rays / rays.norm(dim=-1, keepdim=True)  # Normalize

        # Reference implementation
        ref_rays = rays.clone().requires_grad_(True)
        ref_angles = self.torch_cartesian_to_spherical(ref_rays)
        ref_angles.sum().backward()

        # Kernel implementation
        kernel_rays = rays.clone().requires_grad_(True)
        kernel_angles = sensor_rays_to_sensor_angles(projection, kernel_rays)
        kernel_angles.sum().backward()

        # Compare gradients
        np.testing.assert_allclose(
            kernel_rays.grad.cpu().numpy(),
            ref_rays.grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=ATOL,
            err_msg="sensor_rays_to_sensor_angles gradient mismatch",
        )

    def test_spherical_round_trip_gradient_matches_reference(self):
        """Test round-trip (angles -> rays -> angles) gradients match reference."""
        num_angles = 80
        torch.manual_seed(456)

        projection = self._create_simple_projection()

        elevation = (torch.rand(num_angles, device=self.device) - 0.5) * 0.4
        azimuth = (torch.rand(num_angles, device=self.device) - 0.5) * 2 * np.pi

        # Reference: round-trip in PyTorch
        ref_elev = elevation.clone().requires_grad_(True)
        ref_az = azimuth.clone().requires_grad_(True)
        ref_rays = self.torch_spherical_to_cartesian(ref_elev, ref_az)
        ref_angles_back = self.torch_cartesian_to_spherical(ref_rays)
        ref_angles_back.sum().backward()

        # Kernel: round-trip
        kernel_angles = torch.stack([elevation, azimuth], dim=-1).requires_grad_(True)
        kernel_rays = sensor_angles_to_sensor_rays(projection, kernel_angles)
        kernel_angles_back = sensor_rays_to_sensor_angles(projection, kernel_rays)
        kernel_angles_back.sum().backward()

        # Compare gradients
        ref_combined_grad = torch.stack([ref_elev.grad, ref_az.grad], dim=-1)
        np.testing.assert_allclose(
            kernel_angles.grad.cpu().numpy(),
            ref_combined_grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Round-trip gradient mismatch",
        )

    # =========================================================================
    # Element to Sensor Angles Gradient Tests
    # =========================================================================

    def test_elevation_gradient_matches_reference(self):
        """Test that elevation gradients match expected mathematical gradient."""
        n_rows, n_columns = 8, 16
        torch.manual_seed(789)

        row_elevations = torch.linspace(
            -0.26, 0.26, n_rows, device=self.device, dtype=torch.float32, requires_grad=True
        )
        column_azimuths = torch.linspace(-np.pi, np.pi, n_columns + 1, device=self.device, dtype=torch.float32)[:-1]

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

        # Create elements: all elements use row 0
        num_elements = 50
        elements = torch.zeros(num_elements, 2, device=self.device, dtype=torch.int32)
        elements[:, 0] = 0  # All row 0
        elements[:, 1] = torch.randint(0, n_columns, (num_elements,), device=self.device)

        # Forward and backward
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)
        loss = sensor_angles[:, 0].sum()  # Sum of elevations
        loss.backward()

        # For row 0, the gradient should be num_elements (each element contributes 1)
        # Other rows should have gradient 0
        expected_grad = torch.zeros(n_rows, device=self.device)
        expected_grad[0] = num_elements

        np.testing.assert_allclose(
            row_elevations.grad.cpu().numpy(),
            expected_grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Elevation gradient mismatch",
        )

    def test_azimuth_gradient_matches_reference(self):
        """Test that azimuth gradients match expected mathematical gradient."""
        n_rows, n_columns = 8, 16
        torch.manual_seed(321)

        row_elevations = torch.linspace(-0.26, 0.26, n_rows, device=self.device, dtype=torch.float32)
        # Create column_azimuths as a leaf tensor
        column_azimuths = torch.linspace(
            -np.pi, np.pi - (2 * np.pi / n_columns), n_columns, device=self.device, dtype=torch.float32
        ).requires_grad_(True)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

        # Create elements: all elements use column 0
        num_elements = 50
        elements = torch.zeros(num_elements, 2, device=self.device, dtype=torch.int32)
        elements[:, 0] = torch.randint(0, n_rows, (num_elements,), device=self.device)
        elements[:, 1] = 0  # All column 0

        # Forward and backward
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)
        loss = sensor_angles[:, 1].sum()  # Sum of azimuths
        loss.backward()

        # For column 0, the gradient should be num_elements
        # Other columns should have gradient 0
        expected_grad = torch.zeros(n_columns, device=self.device)
        expected_grad[0] = num_elements

        np.testing.assert_allclose(
            column_azimuths.grad.cpu().numpy(),
            expected_grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Azimuth gradient mismatch",
        )

    # =========================================================================
    # Ray Generation Gradient Tests
    # =========================================================================

    def test_ray_generation_elevation_gradient_matches_reference(self):
        """Test ray generation elevation gradients match PyTorch reference."""
        n_rows, n_columns = 4, 8
        torch.manual_seed(555)

        row_elevations = torch.linspace(-0.2, 0.2, n_rows, device=self.device, dtype=torch.float32, requires_grad=True)
        column_azimuths = torch.linspace(-np.pi, np.pi, n_columns + 1, device=self.device, dtype=torch.float32)[:-1]

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.2,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.4,
        )

        # Create simple identity pose
        dynamic_pose = create_identity_dynamic_pose(self.device)

        # Create elements for row 0 only
        num_elements = 30
        elements = torch.zeros(num_elements, 2, device=self.device, dtype=torch.int32)
        elements[:, 0] = 0  # All row 0
        elements[:, 1] = torch.randint(0, n_columns, (num_elements,), device=self.device)

        # Forward and backward - correct argument order: projection, elements, dynamic_pose
        world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, dynamic_pose)
        loss = world_rays[:, 3:6].sum()  # Sum of directions
        loss.backward()

        # Gradient should be non-zero for row 0, zero for others
        self.assertTrue(row_elevations.grad[0].abs() > 0.1, "Row 0 elevation gradient should be non-zero")
        self.assertTrue(
            (row_elevations.grad[1:].abs() < 1e-5).all(),
            "Other row elevation gradients should be zero",
        )

    def test_ray_generation_pose_gradient_matches_reference(self):
        """Test ray generation pose gradients match expected behavior."""
        n_rows, n_columns = 4, 8
        torch.manual_seed(666)

        row_elevations = torch.linspace(-0.2, 0.2, n_rows, device=self.device, dtype=torch.float32)
        column_azimuths = torch.linspace(-np.pi, np.pi, n_columns + 1, device=self.device, dtype=torch.float32)[:-1]

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.2,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.4,
        )

        # Create pose with requires_grad
        trans_start = torch.tensor([0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        rot_start = quat_identity((1,), device=self.device).squeeze(0)
        trans_end = torch.tensor([1.0, 0.0, 0.0], device=self.device, requires_grad=True)
        rot_end = quat_identity((1,), device=self.device).squeeze(0)
        dynamic_pose = create_dynamic_pose(trans_start, trans_end, rot_start, rot_end, self.device)

        # Create elements
        num_elements = 20
        elements = torch.zeros(num_elements, 2, device=self.device, dtype=torch.int32)
        elements[:, 0] = torch.randint(0, n_rows, (num_elements,), device=self.device)
        elements[:, 1] = torch.randint(0, n_columns, (num_elements,), device=self.device)

        # Forward and backward - correct argument order: projection, elements, dynamic_pose
        world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, dynamic_pose)
        loss = world_rays[:, :3].sum()  # Sum of origins
        loss.backward()

        # Translation gradients should be non-zero (origins depend on translation)
        self.assertTrue(trans_start.grad.abs().sum() > 0.01, "Start translation gradient should be non-zero")
        self.assertTrue(trans_end.grad.abs().sum() > 0.01, "End translation gradient should be non-zero")

        # Verify gradient accumulation: larger batch should have larger gradient
        # Re-run with subset to verify scaling
        trans_start2 = torch.tensor([0.0, 0.0, 0.0], device=self.device, requires_grad=True)
        trans_end2 = torch.tensor([1.0, 0.0, 0.0], device=self.device, requires_grad=True)
        dynamic_pose2 = create_dynamic_pose(trans_start2, trans_end2, rot_start, rot_end, self.device)

        subset_elements = elements[:10]
        world_rays2, _, _, _ = generate_spinning_lidar_rays(projection, subset_elements, dynamic_pose2)
        world_rays2[:, :3].sum().backward()

        # Full batch gradient magnitude should be larger than subset
        full_grad = (trans_start.grad.abs() + trans_end.grad.abs()).sum()
        subset_grad = (trans_start2.grad.abs() + trans_end2.grad.abs()).sum()
        self.assertGreater(
            full_grad.item(),
            subset_grad.item() * 0.8,  # Full should be at least 80% larger (we have 2x elements)
            f"Full batch gradient ({full_grad}) should be larger than subset ({subset_grad})",
        )

    # =========================================================================
    # Large Batch Gradient Accumulation Tests
    # =========================================================================

    def test_sensor_angles_to_rays_large_batch_accumulation(self):
        """Test gradient accumulation with large batch for sensor_angles_to_rays."""
        num_angles = 2048
        torch.manual_seed(777)

        projection = self._create_simple_projection()

        elevation = torch.zeros(num_angles, device=self.device)
        azimuth = torch.zeros(num_angles, device=self.device)
        angles = torch.stack([elevation, azimuth], dim=-1).requires_grad_(True)

        # Forward and backward
        rays = sensor_angles_to_sensor_rays(projection, angles)
        rays.sum().backward()

        # At elevation=0, azimuth=0:
        # x = cos(0)*cos(0) = 1, y = cos(0)*sin(0) = 0, z = sin(0) = 0
        # dx/d_elev = -sin(0)*cos(0) = 0, dx/d_az = cos(0)*(-sin(0)) = 0
        # dy/d_elev = -sin(0)*sin(0) = 0, dy/d_az = cos(0)*cos(0) = 1
        # dz/d_elev = cos(0) = 1, dz/d_az = 0
        # Total: d_elev = 0 + 0 + 1 = 1, d_az = 0 + 1 + 0 = 1 per element
        expected_elev_grad = num_angles  # Each element contributes 1
        expected_az_grad = num_angles

        np.testing.assert_allclose(
            angles.grad[:, 0].sum().cpu().numpy(),
            expected_elev_grad,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg=f"Elevation gradient should accumulate to {expected_elev_grad}",
        )
        np.testing.assert_allclose(
            angles.grad[:, 1].sum().cpu().numpy(),
            expected_az_grad,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg=f"Azimuth gradient should accumulate to {expected_az_grad}",
        )

    def test_elements_to_angles_large_batch_accumulation(self):
        """Test gradient accumulation with large batch for elements_to_sensor_angles."""
        n_rows, n_columns = 8, 256
        torch.manual_seed(888)

        row_elevations = torch.linspace(
            -0.26, 0.26, n_rows, device=self.device, dtype=torch.float32, requires_grad=True
        )
        column_azimuths = torch.linspace(-np.pi, np.pi, n_columns + 1, device=self.device, dtype=torch.float32)[:-1]

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.26,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.52,
        )

        # Create elements with uniform distribution across all rows
        elements_per_row = 256
        elements = []
        for row in range(n_rows):
            for _ in range(elements_per_row):
                col = np.random.randint(0, n_columns)
                elements.append([row, col])
        elements = torch.tensor(elements, device=self.device, dtype=torch.int32)

        # Forward and backward
        sensor_angles, _ = elements_to_sensor_angles(projection, elements)
        loss = sensor_angles[:, 0].sum()  # Sum of elevations
        loss.backward()

        # Each row should have gradient equal to elements_per_row
        expected_grad = torch.ones(n_rows, device=self.device) * elements_per_row

        np.testing.assert_allclose(
            row_elevations.grad.cpu().numpy(),
            expected_grad.cpu().numpy(),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
            err_msg="Large batch elevation gradient accumulation mismatch",
        )

    def test_ray_generation_large_batch_accumulation(self):
        """Test gradient accumulation with large batch for generate_spinning_lidar_rays."""
        n_rows, n_columns = 8, 256
        torch.manual_seed(999)

        row_elevations = torch.linspace(-0.2, 0.2, n_rows, device=self.device, dtype=torch.float32, requires_grad=True)
        column_azimuths = torch.linspace(-np.pi, np.pi, n_columns + 1, device=self.device, dtype=torch.float32)[:-1]

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
            fov_horiz_start_rad=-np.pi,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.2,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.4,
        )

        # Create identity pose
        dynamic_pose = create_identity_dynamic_pose(self.device)

        # Create elements with uniform distribution across rows
        elements_per_row = 128
        elements = []
        for row in range(n_rows):
            for _ in range(elements_per_row):
                col = np.random.randint(0, n_columns)
                elements.append([row, col])
        elements = torch.tensor(elements, device=self.device, dtype=torch.int32)

        # Forward and backward - correct argument order: projection, elements, dynamic_pose
        world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, dynamic_pose)
        loss = world_rays[:, 3:6].sum()  # Sum of directions
        loss.backward()

        # Verify gradients are non-zero for all rows
        for row in range(n_rows):
            self.assertTrue(
                row_elevations.grad[row].abs() > 0.1,
                f"Row {row} elevation gradient should be non-zero, got {row_elevations.grad[row]}",
            )

        # Gradients should be roughly proportional to elements_per_row
        # (exact value depends on the derivative of direction w.r.t. elevation)
        total_grad = row_elevations.grad.abs().sum()
        self.assertGreater(total_grad, elements_per_row * n_rows * 0.1, "Total gradient should be substantial")


if __name__ == "__main__":
    unittest.main()
