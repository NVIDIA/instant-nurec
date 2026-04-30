# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for LidarModel abstract base class.

Tests cover:
- nn.Module inheritance
- Forward raises NotImplementedError
- Subclass relationship
- Property accessors
- State dict and device handling
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from libs.sensors.kernels.lidars import RowOffsetStructuredSpinningLidarProjection
from libs.sensors.models.lidars import LidarModel, RowOffsetStructuredSpinningLidarModel


class TestLidarModelAbstraction(unittest.TestCase):
    """Tests for LidarModel abstract base class behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.n_rows = 16
        self.n_columns = 360

        # Create elevation angles (typical for spinning LiDAR)
        row_elevations_rad = torch.linspace(0.26, -0.26, self.n_rows, device=self.device, dtype=self.dtype)

        # Create azimuth angles (full 360 deg rotation)
        column_azimuths_rad = torch.linspace(
            -torch.pi, torch.pi - (2 * torch.pi / self.n_columns), self.n_columns, device=self.device, dtype=self.dtype
        )

        # Small azimuth offsets per row
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

    def test_is_nn_module(self):
        """Test that LidarModel is an nn.Module."""
        self.assertIsInstance(self.lidar, nn.Module)

    def test_forward_raises_not_implemented(self):
        """Test that forward() raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.lidar.forward()

    def test_is_subclass_of_lidar_model(self):
        """Test that RowOffsetStructuredSpinningLidarModel is a subclass of LidarModel."""
        self.assertIsInstance(self.lidar, LidarModel)

    def test_n_rows_property(self):
        """Test n_rows property accessor."""
        self.assertEqual(self.lidar.n_rows, self.n_rows)

    def test_n_columns_property(self):
        """Test n_columns property accessor."""
        self.assertEqual(self.lidar.n_columns, self.n_columns)

    def test_n_elements_property(self):
        """Test n_elements property accessor."""
        self.assertEqual(self.lidar.n_elements, self.n_rows * self.n_columns)

    def test_fov_vert_property(self):
        """Test fov_vert property accessor."""
        fov_vert = self.lidar.fov_vert
        self.assertEqual(len(fov_vert), 2)
        self.assertAlmostEqual(fov_vert[0], 0.26, places=5)
        self.assertAlmostEqual(fov_vert[1], 0.52, places=5)

    def test_fov_horiz_property(self):
        """Test fov_horiz property accessor."""
        fov_horiz = self.lidar.fov_horiz
        self.assertEqual(len(fov_horiz), 2)
        self.assertAlmostEqual(fov_horiz[0], -torch.pi, places=5)
        self.assertAlmostEqual(fov_horiz[1], 2 * torch.pi, places=5)

    def test_spinning_frequency_hz_property(self):
        """Test spinning_frequency_hz property accessor."""
        self.assertEqual(self.lidar.spinning_frequency_hz, 10.0)

    def test_spinning_direction_property(self):
        """Test spinning_direction property accessor."""
        self.assertEqual(self.lidar.spinning_direction, "cw")

    def test_lidar_model_state_dict(self):
        """Test that lidar model supports state_dict."""
        state = self.lidar.state_dict()
        self.assertIsInstance(state, dict)

    def test_lidar_model_to_device(self):
        """Test that lidar model can be moved to different device."""
        lidar_cpu = self.lidar.to("cpu")
        self.assertIsInstance(lidar_cpu, LidarModel)

    def test_lidar_model_has_parameters(self):
        """Test that lidar model has parameters from nn.Module."""
        params = list(self.lidar.parameters())
        self.assertIsInstance(params, list)

    def test_valid_sensor_angles_public_method(self):
        """Test public valid_sensor_angles method."""
        # Angles within FOV
        sensor_angles = torch.tensor([[0.1, 0.0], [-0.1, 1.0]], device=self.device, dtype=self.dtype)

        valid = self.lidar.valid_sensor_angles(sensor_angles)

        self.assertEqual(valid.shape, (2,))
        self.assertEqual(valid.dtype, torch.bool)

    def test_valid_sensor_angles_with_numpy(self):
        """Test valid_sensor_angles with numpy input."""
        sensor_angles_np = np.array([[0.1, 0.0], [-0.1, 1.0]], dtype=np.float32)

        valid = self.lidar.valid_sensor_angles(sensor_angles_np)

        self.assertIsInstance(valid, torch.Tensor)
        self.assertEqual(valid.shape, (2,))


if __name__ == "__main__":
    unittest.main()
