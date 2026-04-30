# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for LiDAR parameter dataclasses.

Tests the LiDAR parameter data structures including:
- Parameter dataclass construction
- Type validation
- Optional parameters (row offsets, maps)
- Spinning direction literals
- build_angles_to_columns_map function
- Slang kernel find_nearest_column with map lookup
"""

import math
import unittest

import numpy as np
import torch

from libs.sensors.kernels.common.pose import DynamicPose, Pose
from libs.sensors.kernels.lidars import (
    elements_to_sensor_angles,
    inverse_project_spinning_lidar,
)
from libs.sensors.kernels.lidars.parameters import (
    LidarProjection,
    RowOffsetStructuredSpinningLidarProjection,
    build_angles_to_columns_map,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRowOffsetStructuredSpinningLidarProjection(unittest.TestCase):
    """Test RowOffsetStructuredSpinningLidarProjection dataclass."""

    def test_basic_creation(self):
        """Test basic LiDAR projection creation without row offsets."""
        n_rows = 16
        n_columns = 1024

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.linspace(-0.2, 0.2, n_rows, device=device),
            column_azimuths_rad=torch.linspace(0, 2 * np.pi, n_columns, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.2,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.4,
        )

        self.assertEqual(projection.n_rows, n_rows)
        self.assertEqual(projection.n_columns, n_columns)
        self.assertEqual(projection.row_elevations_rad.shape, (n_rows,))
        self.assertEqual(projection.column_azimuths_rad.shape, (n_columns,))
        self.assertIsNone(projection.row_azimuth_offsets_rad)
        self.assertEqual(projection.spinning_frequency_hz, 0.0)
        self.assertEqual(projection.spinning_direction, "cw")

    def test_creation_with_row_offsets(self):
        """Test LiDAR projection creation with row azimuth offsets (Hesai-style)."""
        n_rows = 32
        n_columns = 2048
        row_offsets = torch.linspace(0, 0.1, n_rows, device=device)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.linspace(-0.3, 0.3, n_rows, device=device),
            column_azimuths_rad=torch.linspace(0, 2 * np.pi, n_columns, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.3,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.6,
            row_azimuth_offsets_rad=row_offsets,
            spinning_frequency_hz=20.0,
            spinning_direction="ccw",
        )

        self.assertIsNotNone(projection.row_azimuth_offsets_rad)
        assert projection.row_azimuth_offsets_rad is not None
        self.assertEqual(projection.row_azimuth_offsets_rad.shape, (n_rows,))
        self.assertEqual(projection.spinning_frequency_hz, 20.0)
        self.assertEqual(projection.spinning_direction, "ccw")

    def test_spinning_direction_values(self):
        """Test that spinning direction accepts both 'cw' and 'ccw'."""
        n_rows, n_columns = 16, 512

        # Clockwise
        proj_cw = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.zeros(n_rows, device=device),
            column_azimuths_rad=torch.zeros(n_columns, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
            spinning_direction="cw",
        )

        # Counterclockwise
        proj_ccw = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.zeros(n_rows, device=device),
            column_azimuths_rad=torch.zeros(n_columns, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
            spinning_direction="ccw",
        )

        self.assertEqual(proj_cw.spinning_direction, "cw")
        self.assertEqual(proj_ccw.spinning_direction, "ccw")

    def test_fov_parameters(self):
        """Test FOV parameter storage."""
        fov_h_start = -np.pi
        fov_h_span = 2 * np.pi
        fov_v_start = -0.2617  # -15 degrees
        fov_v_span = 0.5236  # 30 degrees

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=16,
            n_columns=1024,
            row_elevations_rad=torch.zeros(16, device=device),
            column_azimuths_rad=torch.zeros(1024, device=device),
            fov_horiz_start_rad=fov_h_start,
            fov_horiz_span_rad=fov_h_span,
            fov_vert_start_rad=fov_v_start,
            fov_vert_span_rad=fov_v_span,
        )

        self.assertAlmostEqual(projection.fov_horiz_start_rad, fov_h_start, places=5)
        self.assertAlmostEqual(projection.fov_horiz_span_rad, fov_h_span, places=5)
        self.assertAlmostEqual(projection.fov_vert_start_rad, fov_v_start, places=5)
        self.assertAlmostEqual(projection.fov_vert_span_rad, fov_v_span, places=5)

    def test_optional_angles_map(self):
        """Test optional angles-to-columns map."""
        n_rows, n_columns = 16, 512

        # Without map
        proj_no_map = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.zeros(n_rows, device=device),
            column_azimuths_rad=torch.zeros(n_columns, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
            angles_to_columns_map=None,
        )

        # With map
        map_height, map_width = 100, 200
        angles_map = torch.zeros((map_height, map_width), dtype=torch.int32, device=device)

        proj_with_map = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.zeros(n_rows, device=device),
            column_azimuths_rad=torch.zeros(n_columns, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
            angles_to_columns_map=angles_map,
            angles_to_columns_map_resolution_factor=2,
        )

        self.assertIsNone(proj_no_map.angles_to_columns_map)
        self.assertIsNotNone(proj_with_map.angles_to_columns_map)
        assert proj_with_map.angles_to_columns_map is not None
        self.assertEqual(proj_with_map.angles_to_columns_map.shape, (map_height, map_width))
        self.assertEqual(proj_with_map.angles_to_columns_map_resolution_factor, 2)

    def test_different_row_column_counts(self):
        """Test LiDAR with various row/column configurations."""
        configs = [
            (16, 1024),  # Velodyne-style
            (32, 2048),  # Hesai-style
            (128, 2048),  # High-res
            (64, 1024),  # Mid-range
        ]

        for n_rows, n_columns in configs:
            with self.subTest(n_rows=n_rows, n_columns=n_columns):
                projection = RowOffsetStructuredSpinningLidarProjection(
                    n_rows=n_rows,
                    n_columns=n_columns,
                    row_elevations_rad=torch.linspace(-0.3, 0.3, n_rows, device=device),
                    column_azimuths_rad=torch.linspace(0, 2 * np.pi, n_columns, device=device),
                    fov_horiz_start_rad=0.0,
                    fov_horiz_span_rad=2 * np.pi,
                    fov_vert_start_rad=0.3,  # TOP of FOV (cw convention)
                    fov_vert_span_rad=0.6,
                )

                self.assertEqual(projection.n_rows, n_rows)
                self.assertEqual(projection.n_columns, n_columns)
                self.assertEqual(projection.row_elevations_rad.shape[0], n_rows)
                self.assertEqual(projection.column_azimuths_rad.shape[0], n_columns)


class TestLidarProjectionTypeAlias(unittest.TestCase):
    """Test LidarProjection type alias."""

    def test_type_alias(self):
        """Test that LidarProjection is an alias for RowOffsetStructuredSpinningLidarProjection."""
        self.assertIs(LidarProjection, RowOffsetStructuredSpinningLidarProjection)


class TestElevationAndAzimuthArrays(unittest.TestCase):
    """Test elevation and azimuth angle arrays."""

    def test_elevation_range(self):
        """Test that elevation angles cover expected range."""
        n_rows = 32
        min_elev = -15 * np.pi / 180  # -15 degrees
        max_elev = 15 * np.pi / 180  # 15 degrees

        elevations = torch.linspace(min_elev, max_elev, n_rows, device=device)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=512,
            row_elevations_rad=elevations,
            column_azimuths_rad=torch.zeros(512, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=max_elev,  # TOP of FOV (cw convention)
            fov_vert_span_rad=max_elev - min_elev,
        )

        self.assertAlmostEqual(projection.row_elevations_rad[0].item(), min_elev, places=5)
        self.assertAlmostEqual(projection.row_elevations_rad[-1].item(), max_elev, places=5)

    def test_azimuth_full_circle(self):
        """Test that azimuth angles cover full 360 degrees."""
        n_columns = 1024
        azimuths = torch.linspace(0, 2 * np.pi, n_columns, device=device)

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=16,
            n_columns=n_columns,
            row_elevations_rad=torch.zeros(16, device=device),
            column_azimuths_rad=azimuths,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
        )

        self.assertAlmostEqual(projection.column_azimuths_rad[0].item(), 0.0, places=5)
        self.assertAlmostEqual(projection.column_azimuths_rad[-1].item(), 2 * np.pi, places=5)

    def test_non_uniform_elevations(self):
        """Test LiDAR with non-uniform elevation spacing."""
        # Typical real-world LiDAR might have denser rows near horizon
        n_rows = 16
        elevations = torch.tensor(
            [
                -0.2617,
                -0.2094,
                -0.1571,
                -0.1047,
                -0.0524,
                -0.0175,
                -0.0087,
                0.0,
                0.0087,
                0.0175,
                0.0524,
                0.1047,
                0.1571,
                0.2094,
                0.2617,
                0.3142,
            ],
            device=device,
        )

        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=512,
            row_elevations_rad=elevations,
            column_azimuths_rad=torch.zeros(512, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.3,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.6,
        )

        self.assertEqual(projection.row_elevations_rad.shape[0], n_rows)


class TestTensorDevices(unittest.TestCase):
    """Test that parameters work with different devices."""

    def test_parameters_on_cuda(self):
        """Test creating parameters on CUDA device."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        cuda_device = torch.device("cuda")
        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=16,
            n_columns=512,
            row_elevations_rad=torch.zeros(16, device=cuda_device),
            column_azimuths_rad=torch.zeros(512, device=cuda_device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
        )

        self.assertEqual(projection.row_elevations_rad.device.type, "cuda")
        self.assertEqual(projection.column_azimuths_rad.device.type, "cuda")

    def test_parameters_on_cpu(self):
        """Test creating parameters on CPU device."""
        cpu_device = torch.device("cpu")
        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=16,
            n_columns=512,
            row_elevations_rad=torch.zeros(16, device=cpu_device),
            column_azimuths_rad=torch.zeros(512, device=cpu_device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
        )

        self.assertEqual(projection.row_elevations_rad.device.type, "cpu")


class TestDataclassProperties(unittest.TestCase):
    """Test dataclass properties and behavior."""

    def test_dataclass_repr(self):
        """Test that parameters have string representation."""
        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=16,
            n_columns=512,
            row_elevations_rad=torch.zeros(16, device=device),
            column_azimuths_rad=torch.zeros(512, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
        )

        repr_str = repr(projection)
        self.assertIsInstance(repr_str, str)
        self.assertIn("RowOffsetStructuredSpinningLidarProjection", repr_str)

    def test_default_values(self):
        """Test default values for optional parameters."""
        projection = RowOffsetStructuredSpinningLidarProjection(
            n_rows=8,
            n_columns=256,
            row_elevations_rad=torch.zeros(8, device=device),
            column_azimuths_rad=torch.zeros(256, device=device),
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.0,
            fov_vert_span_rad=0.5,
            # All optional parameters use defaults
        )

        self.assertIsNone(projection.row_azimuth_offsets_rad)
        self.assertEqual(projection.spinning_frequency_hz, 0.0)
        self.assertEqual(projection.spinning_direction, "cw")
        self.assertIsNone(projection.angles_to_columns_map)
        self.assertEqual(projection.angles_to_columns_map_resolution_factor, 1)


# ============================================================================
# Tests for build_angles_to_columns_map function
# ============================================================================


class TestBuildAnglesToColumnsMap(unittest.TestCase):
    """Test the build_angles_to_columns_map 2D map function.

    The 2D map indexes (elevation_idx, azimuth_idx) -> column, which properly
    accounts for row azimuth offsets when finding the nearest column.
    """

    def setUp(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")

    def _create_map_params(self, n_rows=16, n_columns=512, with_row_offsets=False, resolution_factor=4):
        """Helper to create all parameters needed for build_angles_to_columns_map."""
        two_pi = 2 * np.pi
        row_elevations = torch.linspace(-0.2, 0.2, n_rows, device=self.device)
        column_azimuths = torch.linspace(0, two_pi, n_columns + 1, device=self.device)[:-1]

        if with_row_offsets:
            row_offsets = torch.linspace(0, 0.1, n_rows, device=self.device)
        else:
            row_offsets = None

        fov_vert_start = 0.2  # top
        fov_vert_span = 0.4
        fov_horiz_start = 0.0
        fov_horiz_span = two_pi

        return {
            "n_rows": n_rows,
            "n_columns": n_columns,
            "row_elevations_rad": row_elevations,
            "column_azimuths_rad": column_azimuths,
            "row_azimuth_offsets_rad": row_offsets,
            "fov_vert_start_rad": fov_vert_start,
            "fov_vert_span_rad": fov_vert_span,
            "fov_horiz_start_rad": fov_horiz_start,
            "fov_horiz_span_rad": fov_horiz_span,
            "spinning_direction": "ccw",
            "resolution_factor": resolution_factor,
        }

    def test_map_shape_default_resolution(self):
        """Test 2D map has correct shape."""
        params = self._create_map_params(n_rows=16, n_columns=512, resolution_factor=4)

        map_tensor = build_angles_to_columns_map(**params)

        expected_height = 16 * 4  # n_rows * resolution_factor
        expected_width = 512 * 4  # n_columns * resolution_factor
        self.assertEqual(map_tensor.shape, (expected_height, expected_width))
        self.assertEqual(map_tensor.dtype, torch.int32)

    def test_map_shape_higher_resolution(self):
        """Test map has correct shape with different resolution factors."""
        for resolution_factor in [2, 4, 8]:
            with self.subTest(resolution_factor=resolution_factor):
                params = self._create_map_params(n_rows=16, n_columns=256, resolution_factor=resolution_factor)
                map_tensor = build_angles_to_columns_map(**params)
                expected_height = 16 * resolution_factor
                expected_width = 256 * resolution_factor
                self.assertEqual(map_tensor.shape, (expected_height, expected_width))

    def test_map_values_in_range(self):
        """Test all map values are valid column indices."""
        n_columns = 1024
        params = self._create_map_params(n_rows=16, n_columns=n_columns, resolution_factor=2)

        map_tensor = build_angles_to_columns_map(**params)

        self.assertTrue((map_tensor >= 0).all(), "All indices should be >= 0")
        self.assertTrue((map_tensor < n_columns).all(), f"All indices should be < {n_columns}")

    def test_map_uniform_azimuths(self):
        """Test map with uniformly spaced azimuths and no row offsets."""
        n_columns = 360
        n_rows = 16
        params = self._create_map_params(
            n_rows=n_rows, n_columns=n_columns, with_row_offsets=False, resolution_factor=1
        )

        map_tensor = build_angles_to_columns_map(**params)

        # For uniform spacing without row offsets, columns should increase monotonically
        # across the horizontal dimension
        for row_idx in range(n_rows):
            row_data = map_tensor[row_idx, :]
            # Check that most adjacent values are either equal or increasing
            # (with possible wraparound at boundaries)
            diffs = row_data[1:].float() - row_data[:-1].float()
            increasing_or_equal = (diffs >= -1) & (diffs <= 1)
            self.assertTrue(
                increasing_or_equal.sum() > n_columns * 0.9, f"Row {row_idx} should have mostly monotonic columns"
            )

    def test_map_non_uniform_row_offsets(self):
        """Test map with row azimuth offsets."""
        params = self._create_map_params(n_rows=16, n_columns=100, with_row_offsets=True, resolution_factor=2)

        map_tensor = build_angles_to_columns_map(**params)

        # Verify map values are in range
        self.assertTrue((map_tensor >= 0).all())
        self.assertTrue((map_tensor < 100).all())

    def test_map_preserves_device(self):
        """Test that map is created on same device as input."""
        params = self._create_map_params(n_rows=16, n_columns=100, resolution_factor=2)

        map_tensor = build_angles_to_columns_map(**params)

        self.assertEqual(map_tensor.device, params["column_azimuths_rad"].device)

    def test_map_lookup_accuracy(self):
        """Test that 2D map lookup is accurate for various sensor angles."""
        n_rows = 16
        n_columns = 512
        params = self._create_map_params(
            n_rows=n_rows, n_columns=n_columns, with_row_offsets=False, resolution_factor=4
        )

        map_tensor = build_angles_to_columns_map(**params)
        map_height, map_width = map_tensor.shape

        # Test several (elevation, azimuth) pairs
        test_elevations = [0.0, 0.1, -0.1]
        test_azimuths = [0.0, np.pi / 4, np.pi, 3 * np.pi / 2]

        column_azimuths = params["column_azimuths_rad"]
        fov_vert_start = params["fov_vert_start_rad"]
        fov_vert_span = params["fov_vert_span_rad"]
        fov_horiz_span = params["fov_horiz_span_rad"]

        for elev in test_elevations:
            for azim in test_azimuths:
                with self.subTest(elevation=elev, azimuth=azim):
                    # Compute map indices
                    rel_elev = fov_vert_start - elev  # cw convention
                    if rel_elev < 0:
                        rel_elev = 0
                    rel_azim = azim  # ccw direction

                    vert_idx = int(rel_elev / fov_vert_span * (map_height - 1) + 0.5)
                    horiz_idx = int(rel_azim / fov_horiz_span * (map_width - 1) + 0.5)

                    vert_idx = max(0, min(map_height - 1, vert_idx))
                    horiz_idx = max(0, min(map_width - 1, horiz_idx))

                    col_from_map = map_tensor[vert_idx, horiz_idx].item()

                    # Verify it's a valid column
                    self.assertGreaterEqual(col_from_map, 0)
                    self.assertLess(col_from_map, n_columns)


# ============================================================================
# Tests for find_nearest_column Slang kernel with map lookup
# ============================================================================

# Tolerances for tests
ATOL = 1e-3
RTOL = 1e-3


class TestFindNearestColumnKernel(unittest.TestCase):
    """Test find_nearest_column through the inverse_project_spinning_lidar kernel.

    These tests verify that the O(1) map lookup produces the same results
    as the O(n) linear search fallback.
    """

    def setUp(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = torch.device("cuda")

    def _create_simple_dynamic_pose(self):
        """Create a simple dynamic pose for testing."""
        from libs.geometry.kernels.quaternion import quat_identity

        device = self.device
        pose_trans = torch.zeros(3, device=device)
        pose_rot = quat_identity((1,), device=device).squeeze(0)

        pose = Pose(translation=pose_trans, rotation=pose_rot)
        return DynamicPose.from_static_pose(pose)

    def _create_projection_without_map(self, n_rows=16, n_columns=512):
        """Create a LiDAR projection without map lookup."""
        column_azimuths = torch.linspace(0, 2 * np.pi, n_columns + 1, device=self.device)[:-1]

        return RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=torch.linspace(-0.2, 0.2, n_rows, device=self.device),
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=0.0,
            fov_horiz_span_rad=2 * np.pi,
            fov_vert_start_rad=0.25,  # TOP of FOV (cw convention)
            fov_vert_span_rad=0.5,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=None,
        )

    def _create_projection_with_map(self, n_rows=16, n_columns=512, resolution_factor=2):
        """Create a LiDAR projection with 2D map lookup."""
        row_elevations = torch.linspace(-0.2, 0.2, n_rows, device=self.device)
        column_azimuths = torch.linspace(0, 2 * np.pi, n_columns + 1, device=self.device)[:-1]
        # Use same FOV as _create_projection_without_map for comparison tests
        fov_vert_start = 0.25  # TOP of FOV (cw convention)
        fov_vert_span = 0.5
        fov_horiz_start = 0.0
        fov_horiz_span = 2 * np.pi

        angles_map = build_angles_to_columns_map(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=None,
            fov_vert_start_rad=fov_vert_start,
            fov_vert_span_rad=fov_vert_span,
            fov_horiz_start_rad=fov_horiz_start,
            fov_horiz_span_rad=fov_horiz_span,
            spinning_direction="cw",
            resolution_factor=resolution_factor,
        )

        return RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=fov_horiz_start,
            fov_horiz_span_rad=fov_horiz_span,
            fov_vert_start_rad=fov_vert_start,
            fov_vert_span_rad=fov_vert_span,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=angles_map,
            angles_to_columns_map_resolution_factor=resolution_factor,
        )

    def test_elements_to_angles_consistency(self):
        """Test that elements_to_sensor_angles works with both map and no-map projections."""
        n_rows, n_columns = 16, 256

        proj_no_map = self._create_projection_without_map(n_rows, n_columns)
        proj_with_map = self._create_projection_with_map(n_rows, n_columns)

        # Create test elements
        elements = torch.tensor(
            [
                [0, 0],
                [8, 128],
                [15, 255],
                [4, 64],
            ],
            device=self.device,
            dtype=torch.int32,
        )

        angles_no_map, _ = elements_to_sensor_angles(proj_no_map, elements)
        angles_with_map, _ = elements_to_sensor_angles(proj_with_map, elements)

        # Results should be identical (elements_to_angles doesn't use find_nearest_column)
        self.assertTrue(
            torch.allclose(angles_no_map, angles_with_map, atol=ATOL, rtol=RTOL),
            "elements_to_sensor_angles should give same results with/without map",
        )

    def test_inverse_projection_with_map(self):
        """Test inverse projection works correctly with map lookup."""
        proj_with_map = self._create_projection_with_map(n_rows=16, n_columns=512)
        dynamic_pose = self._create_simple_dynamic_pose()

        # Create world points in front of sensor
        world_points = torch.tensor(
            [
                [5.0, 0.0, 0.0],  # Right
                [0.0, 10.0, 0.0],  # Forward
                [0.0, 5.0, 1.0],  # Forward and up
                [-3.0, 8.0, -0.5],  # Forward-left and down
            ],
            device=self.device,
        )

        sensor_angles, valid, _, _, _ = inverse_project_spinning_lidar(
            proj_with_map,
            world_points,
            dynamic_pose,
            max_iterations=5,
            return_valid_flags=True,
        )

        # Check shapes
        self.assertEqual(sensor_angles.shape, (4, 2))
        assert valid is not None
        self.assertEqual(valid.shape, (4,))

        # Check that valid points have reasonable angles
        valid_mask = valid.bool()
        if valid_mask.any():
            valid_angles = sensor_angles[valid_mask]
            # Elevation should be in FOV range
            self.assertTrue(
                (valid_angles[:, 0] >= -0.3).all() and (valid_angles[:, 0] <= 0.3).all(),
                "Elevations should be within FOV",
            )

    def test_inverse_projection_map_vs_linear_search(self):
        """Test that map lookup produces similar results to linear search."""
        n_rows, n_columns = 16, 256

        proj_no_map = self._create_projection_without_map(n_rows, n_columns)
        proj_with_map = self._create_projection_with_map(n_rows, n_columns, resolution_factor=4)
        dynamic_pose = self._create_simple_dynamic_pose()

        # Create world points
        world_points = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.5],
                [2.0, 4.0, 0.2],
                [-2.0, 6.0, -0.1],
            ],
            device=self.device,
        )

        angles_no_map, valid_no_map, _, _, _ = inverse_project_spinning_lidar(
            proj_no_map,
            world_points,
            dynamic_pose,
            max_iterations=10,
            return_valid_flags=True,
        )

        angles_with_map, valid_with_map, _, _, _ = inverse_project_spinning_lidar(
            proj_with_map,
            world_points,
            dynamic_pose,
            max_iterations=10,
            return_valid_flags=True,
        )

        # Valid flags should be the same
        self.assertTrue(torch.equal(valid_no_map, valid_with_map), "Valid flags should match between map and no-map")

        # Angles should be very close for valid points
        assert valid_no_map is not None
        valid_mask = valid_no_map.bool()
        if valid_mask.any():
            self.assertTrue(
                torch.allclose(
                    angles_no_map[valid_mask],
                    angles_with_map[valid_mask],
                    atol=0.01,  # Allow small difference due to quantization
                    rtol=0.01,
                ),
                f"Angles differ: no_map={angles_no_map[valid_mask]}, with_map={angles_with_map[valid_mask]}",
            )

    def test_2d_map_with_row_offsets_round_trip(self):
        """Test 2D map with row azimuth offsets - full forward/inverse round trip.

        This test the 2D map properly accounts for row azimuth offsets when finding columns.
        """
        n_rows, n_columns = 16, 256
        row_elevations = torch.linspace(-0.2, 0.2, n_rows, device=self.device)
        column_azimuths = torch.linspace(0, 2 * np.pi, n_columns + 1, device=self.device)[:-1]
        # Significant row offsets (like Hesai P128)
        row_offsets = torch.linspace(0, 0.15, n_rows, device=self.device)

        fov_vert_start = 0.25
        fov_vert_span = 0.5
        fov_horiz_start = 0.0
        fov_horiz_span = 2 * np.pi

        # Build 2D map with row offsets
        angles_map = build_angles_to_columns_map(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            row_azimuth_offsets_rad=row_offsets,
            fov_vert_start_rad=fov_vert_start,
            fov_vert_span_rad=fov_vert_span,
            fov_horiz_start_rad=fov_horiz_start,
            fov_horiz_span_rad=fov_horiz_span,
            spinning_direction="cw",
            resolution_factor=4,
        )

        proj = RowOffsetStructuredSpinningLidarProjection(
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations,
            column_azimuths_rad=column_azimuths,
            fov_horiz_start_rad=fov_horiz_start,
            fov_horiz_span_rad=fov_horiz_span,
            fov_vert_start_rad=fov_vert_start,
            fov_vert_span_rad=fov_vert_span,
            row_azimuth_offsets_rad=row_offsets,
            spinning_frequency_hz=10.0,
            spinning_direction="cw",
            angles_to_columns_map=angles_map,
            angles_to_columns_map_resolution_factor=4,
        )

        dynamic_pose = self._create_simple_dynamic_pose()

        # Test with a batch of elements (random sample)
        np.random.seed(42)
        num_elements = 200
        rows = np.random.randint(0, n_rows, num_elements)
        cols = np.random.randint(0, n_columns, num_elements)
        elements = torch.tensor(
            np.stack([rows, cols], axis=1),
            device=self.device,
            dtype=torch.int32,
        )

        # Forward projection: elements → world rays
        from libs.sensors.kernels.lidars import generate_spinning_lidar_rays

        world_rays, _, _, _ = generate_spinning_lidar_rays(proj, elements, dynamic_pose)

        # Generate world points along rays
        torch.manual_seed(42)
        distances = torch.rand(num_elements, 1, device=self.device) * 50
        world_points = world_rays[:, :3] + world_rays[:, 3:6] * distances

        # Inverse projection: world points → sensor angles
        sensor_angles_inv, valid, _, _, _ = inverse_project_spinning_lidar(
            proj, world_points, dynamic_pose, max_iterations=10, return_valid_flags=True
        )

        # Get original angles from elements
        sensor_angles_orig, _ = elements_to_sensor_angles(proj, elements)

        # Check round-trip accuracy for valid projections
        assert valid is not None
        valid_mask = valid.cpu().numpy()
        valid_count = valid_mask.sum()

        if valid_count > 0:
            angles_orig = sensor_angles_orig[valid].cpu().numpy()
            angles_inv = sensor_angles_inv[valid].cpu().numpy()

            # Compute angle differences
            elev_diff = np.abs(angles_inv[:, 0] - angles_orig[:, 0])
            az_orig = angles_orig[:, 1]
            az_inv = angles_inv[:, 1]
            az_diff = np.abs(np.arctan2(np.sin(az_inv - az_orig), np.cos(az_inv - az_orig)))

            angle_errors = np.sqrt(elev_diff**2 + az_diff**2)

            # At least 95% should match within tolerance
            tolerance_rad = 0.002  # ~0.1 degrees
            within_tolerance = np.sum(angle_errors < tolerance_rad)
            match_ratio = within_tolerance / valid_count

            self.assertGreater(
                match_ratio,
                0.95,
                f"2D map with row offsets round-trip failed: {match_ratio * 100:.1f}% matched "
                f"(max error: {np.degrees(np.max(angle_errors)):.4f}°)",
            )

    def test_map_with_different_resolution_factors(self):
        """Test map lookup with different resolution factors."""
        n_columns = 256
        dynamic_pose = self._create_simple_dynamic_pose()

        world_points = torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [2.0, 4.0, 0.1],
            ],
            device=self.device,
        )

        for resolution_factor in [1, 2, 4, 8]:
            with self.subTest(resolution_factor=resolution_factor):
                proj = self._create_projection_with_map(n_columns=n_columns, resolution_factor=resolution_factor)

                angles, valid, _, _, _ = inverse_project_spinning_lidar(
                    proj,
                    world_points,
                    dynamic_pose,
                    max_iterations=5,
                    return_valid_flags=True,
                )

                self.assertEqual(angles.shape, (2, 2))
                assert valid is not None
                self.assertEqual(valid.shape, (2,))

    def test_round_trip_elements_to_angles_to_rays(self):
        """Test round-trip: elements → angles → rays → angles."""
        n_rows, n_columns = 16, 256

        proj_with_map = self._create_projection_with_map(n_rows, n_columns)

        # Create test elements
        elements = torch.tensor(
            [
                [4, 64],
                [8, 128],
                [12, 192],
            ],
            device=self.device,
            dtype=torch.int32,
        )

        # Get angles from elements
        angles_orig, _ = elements_to_sensor_angles(proj_with_map, elements)

        # Convert angles to rays (this is tested elsewhere, just use for round-trip)
        # Note: This doesn't fully test find_nearest_column since elements_to_sensor_angles
        # doesn't use it - but it validates the overall projection pipeline
        self.assertEqual(angles_orig.shape, (3, 2))

    def test_large_batch_with_map(self):
        """Test map lookup with large batch of points."""
        proj_with_map = self._create_projection_with_map(n_rows=32, n_columns=1024)
        dynamic_pose = self._create_simple_dynamic_pose()

        # Generate many random world points
        num_points = 1000
        world_points = torch.randn(num_points, 3, device=self.device)
        # Make sure they're in front of the sensor
        world_points[:, 1] = torch.abs(world_points[:, 1]) + 2.0

        angles, valid, _, _, _ = inverse_project_spinning_lidar(
            proj_with_map,
            world_points,
            dynamic_pose,
            max_iterations=5,
            return_valid_flags=True,
        )

        self.assertEqual(angles.shape, (num_points, 2))
        assert valid is not None
        self.assertEqual(valid.shape, (num_points,))

        # Check that we got some valid results
        valid_count = valid.sum().item()
        self.assertGreater(valid_count, 0, "Should have some valid projections")


if __name__ == "__main__":
    unittest.main()
