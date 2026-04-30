# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for BBox Projector utility."""

import unittest

from unittest.mock import MagicMock

import numpy as np

from nre.benchmark.utils.bbox_projector import BBoxProjector


class TestBBoxProjector(unittest.TestCase):
    """Test cases for BBoxProjector class."""

    def test_initialization(self) -> None:
        """Test BBoxProjector initialization."""
        mock_camera_model = MagicMock()
        mock_camera_sensor = MagicMock()

        projector = BBoxProjector(mock_camera_model, mock_camera_sensor)

        self.assertEqual(projector.camera_model, mock_camera_model)
        self.assertEqual(projector.camera_sensor, mock_camera_sensor)

    def test_update_sensors(self) -> None:
        """Test updating camera model and sensor."""
        mock_camera_model = MagicMock()
        mock_camera_sensor = MagicMock()
        projector = BBoxProjector(mock_camera_model, mock_camera_sensor)

        new_model = MagicMock()
        new_sensor = MagicMock()
        projector.update_sensors(new_model, new_sensor)

        self.assertEqual(projector.camera_model, new_model)
        self.assertEqual(projector.camera_sensor, new_sensor)

    def test_get_2d_bbox_from_projection(self) -> None:
        """Test converting 2D points to bounding box."""
        # Create test points forming a box
        image_points = np.array(
            [
                [10.5, 20.3],
                [50.7, 20.1],
                [50.9, 80.6],
                [10.2, 80.8],
            ]
        )

        bbox = BBoxProjector.get_2d_bbox_from_projection(image_points)
        x, y, width, height = bbox

        # Check bbox encompasses all points
        self.assertEqual(x, 10)  # floor(10.2)
        self.assertEqual(y, 20)  # floor(20.1)
        self.assertEqual(width, 41)  # ceil(50.9) - floor(10.2)
        self.assertEqual(height, 61)  # ceil(80.8) - floor(20.1)

    def test_get_2d_bbox_from_projection_with_batch(self) -> None:
        """Test bbox extraction with batch dimension."""
        # Add batch dimension
        image_points = np.array(
            [
                [
                    [10.5, 20.3],
                    [50.7, 20.1],
                    [50.9, 80.6],
                    [10.2, 80.8],
                ]
            ]
        )

        bbox = BBoxProjector.get_2d_bbox_from_projection(image_points)
        x, y, width, height = bbox

        self.assertEqual(x, 10)
        self.assertEqual(y, 20)
        self.assertEqual(width, 41)
        self.assertEqual(height, 61)

    def test_compute_scaled_min_size(self) -> None:
        """Test computing scaled minimum size."""
        # Same resolution - no scaling
        ref_shape = (1080, 1920)
        target_shape = (1080, 1920)
        scaled = BBoxProjector.compute_scaled_min_size(ref_shape, target_shape, reference_min_size=20)
        self.assertEqual(scaled, 20)

        # 2x resolution increase
        ref_shape = (540, 960)
        target_shape = (1080, 1920)
        scaled = BBoxProjector.compute_scaled_min_size(ref_shape, target_shape, reference_min_size=20)
        self.assertEqual(scaled, 40)

        # 0.5x resolution decrease
        ref_shape = (1080, 1920)
        target_shape = (540, 960)
        scaled = BBoxProjector.compute_scaled_min_size(ref_shape, target_shape, reference_min_size=20)
        self.assertEqual(scaled, 10)

        # Minimum is 1 pixel
        ref_shape = (1080, 1920)
        target_shape = (10, 20)
        scaled = BBoxProjector.compute_scaled_min_size(ref_shape, target_shape, reference_min_size=20)
        self.assertGreaterEqual(scaled, 1)

    def test_scale_bbox_around_center(self) -> None:
        """Test scaling bbox around center."""
        # Original bbox: 100x100 centered at (150, 150)
        bbox = (100, 100, 100, 100)
        image_shape = (400, 400)

        # Scale 2x (expand)
        scaled_bbox = BBoxProjector.scale_bbox_around_center(bbox, scale=2.0, image_shape=image_shape)
        x, y, width, height = scaled_bbox
        self.assertEqual(width, 200)
        self.assertEqual(height, 200)
        # Center should remain at (150, 150)
        center_x = x + width // 2
        center_y = y + height // 2
        self.assertEqual(center_x, 150)
        self.assertEqual(center_y, 150)

        # Scale 0.5x (shrink)
        scaled_bbox = BBoxProjector.scale_bbox_around_center(bbox, scale=0.5, image_shape=image_shape)
        x, y, width, height = scaled_bbox
        self.assertEqual(width, 50)
        self.assertEqual(height, 50)

    def test_scale_bbox_clamped_to_image_bounds(self) -> None:
        """Test bbox scaling is clamped to image boundaries."""
        # Bbox near edge
        bbox = (0, 0, 50, 50)
        image_shape = (100, 100)

        # Scale 4x - should be clamped to image
        scaled_bbox = BBoxProjector.scale_bbox_around_center(bbox, scale=4.0, image_shape=image_shape)
        x, y, width, height = scaled_bbox

        # Should be clamped to image boundaries
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 100)
        self.assertLessEqual(y + height, 100)

    def test_is_bbox_valid_within_bounds(self) -> None:
        """Test bbox validation for bbox within image bounds."""
        bbox = (10, 10, 50, 50)
        image_shape = (100, 100)

        self.assertTrue(BBoxProjector.is_bbox_valid(bbox, image_shape, min_size=20))

    def test_is_bbox_valid_out_of_bounds(self) -> None:
        """Test bbox validation for out-of-bounds bbox."""
        # Negative coordinates
        bbox = (-10, 10, 50, 50)
        image_shape = (100, 100)
        self.assertFalse(BBoxProjector.is_bbox_valid(bbox, image_shape, min_size=20))

        # Extends beyond image
        bbox = (80, 80, 50, 50)
        image_shape = (100, 100)
        self.assertFalse(BBoxProjector.is_bbox_valid(bbox, image_shape, min_size=20))

    def test_is_bbox_valid_too_small(self) -> None:
        """Test bbox validation for too small bbox."""
        bbox = (10, 10, 15, 15)  # 15x15 bbox
        image_shape = (100, 100)

        self.assertFalse(BBoxProjector.is_bbox_valid(bbox, image_shape, min_size=20))

    def test_crop_object_valid(self) -> None:
        """Test cropping object from image."""
        # Create test image
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = (50, 50, 60, 60)

        crop = BBoxProjector.crop_object(image, bbox, padding_ratio=0.0, min_size=50)

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.shape[0], 60)  # height
        self.assertEqual(crop.shape[1], 60)  # width
        self.assertEqual(crop.shape[2], 3)  # RGB

    def test_crop_object_with_padding(self) -> None:
        """Test cropping object with padding."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = (50, 50, 60, 60)

        crop = BBoxProjector.crop_object(image, bbox, padding_ratio=0.2, min_size=50)

        self.assertIsNotNone(crop)
        assert crop is not None
        # Crop should be larger than bbox due to padding
        # max(60, 60) * 0.2 = 12 pixels padding
        expected_size = 60 + 2 * 12  # 84
        self.assertEqual(crop.shape[0], expected_size)
        self.assertEqual(crop.shape[1], expected_size)

    def test_crop_object_clamped_to_image(self) -> None:
        """Test crop is clamped to image boundaries."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        # Bbox near edge
        bbox = (0, 0, 60, 60)

        crop = BBoxProjector.crop_object(image, bbox, padding_ratio=0.5, min_size=50)

        self.assertIsNotNone(crop)
        # Should be clamped, not extend beyond image
        assert crop is not None
        self.assertLessEqual(crop.shape[0], 200)
        self.assertLessEqual(crop.shape[1], 200)

    def test_crop_object_invalid_bbox(self) -> None:
        """Test cropping with invalid bbox returns None."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # Too small bbox
        bbox = (50, 50, 10, 10)
        crop = BBoxProjector.crop_object(image, bbox, padding_ratio=0.0, min_size=50)
        self.assertIsNone(crop)

        # Out of bounds bbox
        bbox = (250, 250, 60, 60)
        crop = BBoxProjector.crop_object(image, bbox, padding_ratio=0.0, min_size=50)
        self.assertIsNone(crop)


if __name__ == "__main__":
    unittest.main()
