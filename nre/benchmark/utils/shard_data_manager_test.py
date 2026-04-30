# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for Shard Data Manager utility."""

import unittest

from unittest.mock import MagicMock, Mock, patch

import numpy as np

from nre.benchmark.utils.shard_data_manager import ShardDataManager


class MockBBox3:
    """Mock 3D bounding box for testing."""

    def __init__(self, center, dim, rot):
        """Initialize mock bbox.

        Args:
            center: Center coordinates [x, y, z]
            dim: Dimensions [length, width, height]
            rot: Rotation [roll, pitch, yaw] in radians
        """
        self._data = np.concatenate([center, dim, rot])

    def to_array(self):
        """Return bbox as array."""
        return self._data


class TestShardDataManager(unittest.TestCase):
    """Test cases for ShardDataManager class."""

    def test_box2corners_simple_box(self) -> None:
        """Test converting 3D bbox to corner points."""
        # Create a simple box: 2x2x2 centered at origin, no rotation
        center = np.array([0.0, 0.0, 0.0])
        dim = np.array([2.0, 2.0, 2.0])
        rot = np.array([0.0, 0.0, 0.0])  # No rotation

        bbox3 = MockBBox3(center, dim, rot)
        corners = ShardDataManager.box2corners(bbox3)

        # Should have 8 corners
        self.assertEqual(corners.shape, (8, 3))

        # Check that corners are at expected positions
        # For a 2x2x2 box at origin, corners should be at ±1 in each dim
        expected_corners = np.array(
            [
                [1, 1, 1],  # front-top-right
                [-1, 1, 1],  # front-top-left
                [-1, -1, 1],  # front-bottom-left
                [1, -1, 1],  # front-bottom-right
                [1, -1, -1],  # back-bottom-right
                [-1, -1, -1],  # back-bottom-left
                [-1, 1, -1],  # back-top-left
                [1, 1, -1],  # back-top-right
            ]
        )

        np.testing.assert_array_almost_equal(corners, expected_corners)

    def test_box2corners_translated_box(self) -> None:
        """Test bbox to corners with translation."""
        # 2x2x2 box centered at (10, 5, 3)
        center = np.array([10.0, 5.0, 3.0])
        dim = np.array([2.0, 2.0, 2.0])
        rot = np.array([0.0, 0.0, 0.0])

        bbox3 = MockBBox3(center, dim, rot)
        corners = ShardDataManager.box2corners(bbox3)

        # Center of all corners should be the center of the box
        computed_center = corners.mean(axis=0)
        np.testing.assert_array_almost_equal(computed_center, center)

    def test_box2corners_different_dimensions(self) -> None:
        """Test bbox to corners with non-uniform dimensions."""
        # 4x2x1 box
        center = np.array([0.0, 0.0, 0.0])
        dim = np.array([4.0, 2.0, 1.0])  # length, width, height
        rot = np.array([0.0, 0.0, 0.0])

        bbox3 = MockBBox3(center, dim, rot)
        corners = ShardDataManager.box2corners(bbox3)

        # Check dimensions span correct range
        x_range = corners[:, 0].max() - corners[:, 0].min()
        y_range = corners[:, 1].max() - corners[:, 1].min()
        z_range = corners[:, 2].max() - corners[:, 2].min()

        self.assertAlmostEqual(x_range, 4.0, places=5)
        self.assertAlmostEqual(y_range, 2.0, places=5)
        self.assertAlmostEqual(z_range, 1.0, places=5)

    def test_box2corners_with_rotation(self) -> None:
        """Test bbox to corners with rotation."""
        # 2x2x2 box with 45 degree rotation around Z-axis
        center = np.array([0.0, 0.0, 0.0])
        dim = np.array([2.0, 2.0, 2.0])
        rot = np.array([0.0, 0.0, np.pi / 4])  # 45 degrees around Z

        bbox3 = MockBBox3(center, dim, rot)
        corners = ShardDataManager.box2corners(bbox3)

        # Should still have 8 corners
        self.assertEqual(corners.shape, (8, 3))

        # Center should still be at origin
        computed_center = corners.mean(axis=0)
        np.testing.assert_array_almost_equal(computed_center, center, decimal=5)

        # Check that corners are rotated (not axis-aligned)
        # At least one corner should not be at ±1 in x or y
        has_non_axis_aligned = False
        for corner in corners:
            if not np.allclose(np.abs(corner[:2]), 1.0, atol=0.1):
                has_non_axis_aligned = True
                break
        self.assertTrue(has_non_axis_aligned)

    @patch("glob.glob")
    @patch("nre.benchmark.utils.shard_data_manager.ShardDataLoader")
    def test_initialization(self, mock_shard_loader_cls: Mock, mock_glob: Mock) -> None:
        """Test ShardDataManager initialization."""
        # Mock glob to return shard files
        mock_glob.return_value = ["shard1.zarr", "shard2.zarr"]

        # Mock ShardDataLoader
        mock_loader = MagicMock()
        mock_loader.get_camera_ids.return_value = ["camera_front"]
        mock_loader.get_lidar_ids.return_value = ["lidar_top"]

        mock_camera_sensor = MagicMock()
        mock_camera_sensor.get_camera_model_parameters.return_value = {}
        mock_loader.get_camera_sensor.return_value = mock_camera_sensor

        mock_lidar_sensor = MagicMock()
        mock_loader.get_lidar_sensor.return_value = mock_lidar_sensor

        mock_shard_loader_cls.return_value = mock_loader

        # Mock CameraModel.from_parameters
        with patch("nre.benchmark.utils.shard_data_manager.CameraModel") as mock_camera_model:
            mock_camera_model.from_parameters.return_value = MagicMock()

            manager = ShardDataManager("shard*.zarr", "camera_front")

            # Verify initialization
            self.assertEqual(manager.camera_id, "camera_front")
            self.assertEqual(manager.shard_pattern, "shard*.zarr")
            mock_glob.assert_called_once_with("shard*.zarr")
            mock_shard_loader_cls.assert_called_once()

    @patch("nre.benchmark.utils.shard_data_manager.ShardDataLoader")
    @patch("glob.glob")
    def test_initialization_no_shard_files(self, mock_glob: Mock, mock_shard_loader_cls: Mock) -> None:
        """Test initialization fails with no shard files."""
        mock_glob.return_value = []

        with self.assertRaises(ValueError) as context:
            ShardDataManager("nonexistent*.zarr", "camera_front")

        self.assertIn("No shard files found", str(context.exception))
        # Should fail before trying to load shard
        mock_shard_loader_cls.assert_not_called()

    @patch("nre.benchmark.utils.shard_data_manager.CameraModel")
    @patch("glob.glob")
    @patch("nre.benchmark.utils.shard_data_manager.ShardDataLoader")
    def test_initialization_invalid_camera(
        self,
        mock_shard_loader_cls: Mock,
        mock_glob: Mock,
        mock_camera_model: Mock,
    ) -> None:
        """Test initialization fails with invalid camera ID."""
        mock_glob.return_value = ["shard1.zarr"]

        mock_loader = MagicMock()
        mock_loader.get_camera_ids.return_value = [
            "camera_left",
            "camera_right",
        ]
        mock_loader.get_lidar_ids.return_value = ["lidar_top"]
        mock_shard_loader_cls.return_value = mock_loader

        with self.assertRaises(ValueError) as context:
            ShardDataManager("shard*.zarr", "camera_front")

        self.assertIn("not found in shard", str(context.exception))
        # Should fail before loading camera model
        mock_camera_model.from_parameters.assert_not_called()

    @patch("glob.glob")
    @patch("nre.benchmark.utils.shard_data_manager.ShardDataLoader")
    def test_load_new_shard(self, mock_shard_loader_cls: Mock, mock_glob: Mock) -> None:
        """Test loading a new shard pattern."""
        # Initial setup
        mock_glob.return_value = ["shard1.zarr"]
        mock_loader = self._create_mock_loader()
        mock_shard_loader_cls.return_value = mock_loader

        with patch("nre.benchmark.utils.shard_data_manager.CameraModel") as mock_camera_model:
            mock_camera_model.from_parameters.return_value = MagicMock()

            manager = ShardDataManager("shard1*.zarr", "camera_front")
            self.assertEqual(manager.shard_pattern, "shard1*.zarr")

            # Load new shard
            mock_glob.return_value = ["shard2.zarr"]
            manager.load("shard2*.zarr")

            self.assertEqual(manager.shard_pattern, "shard2*.zarr")
            # Should have called glob twice (init + load)
            self.assertEqual(mock_glob.call_count, 2)

    @patch("glob.glob")
    @patch("nre.benchmark.utils.shard_data_manager.ShardDataLoader")
    def test_reload_alias(self, mock_shard_loader_cls: Mock, mock_glob: Mock) -> None:
        """Test reload() method (alias for load())."""
        mock_glob.return_value = ["shard1.zarr"]
        mock_loader = self._create_mock_loader()
        mock_shard_loader_cls.return_value = mock_loader

        with patch("nre.benchmark.utils.shard_data_manager.CameraModel") as mock_camera_model:
            mock_camera_model.from_parameters.return_value = MagicMock()

            manager = ShardDataManager("shard1*.zarr", "camera_front")

            # Load with new pattern
            mock_glob.return_value = ["shard2.zarr"]
            manager.load("shard2*.zarr")

            self.assertEqual(manager.shard_pattern, "shard2*.zarr")

    def _create_mock_loader(self) -> MagicMock:
        """Create a properly configured mock shard loader."""
        mock_loader = MagicMock()
        mock_loader.get_camera_ids.return_value = ["camera_front"]
        mock_loader.get_lidar_ids.return_value = ["lidar_top"]

        mock_camera_sensor = MagicMock()
        mock_camera_sensor.get_camera_model_parameters.return_value = {}
        mock_loader.get_camera_sensor.return_value = mock_camera_sensor

        mock_lidar_sensor = MagicMock()
        mock_loader.get_lidar_sensor.return_value = mock_lidar_sensor

        return mock_loader


if __name__ == "__main__":
    unittest.main()
