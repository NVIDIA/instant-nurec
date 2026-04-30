# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for Object-Level Perceptual Quality Metric."""

import unittest

import numpy as np
import torch

from nre.metrics.impl.object_level_perceptual import ObjectLevelPerceptualMetric
from nre.metrics.impl.object_level_semantic import ObjectMetadata
from nre.metrics.utils import AggregationMethod


class TestObjectLevelPerceptualMetric(unittest.TestCase):
    """Test cases for ObjectLevelPerceptualMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        np.random.seed(42)  # Fixed seed for reproducible tests
        self.device = "cpu"

    def test_initialization(self) -> None:
        """Test ObjectLevelPerceptualMetric initialization."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        self.assertIsNotNone(metric.weights)
        self.assertIn("edge_similarity", metric.weights)
        self.assertIn("gradient_similarity", metric.weights)
        self.assertIn("blur_score", metric.weights)

    def test_validate_inputs_valid_numpy(self) -> None:
        """Test input validation with valid numpy arrays."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        # Valid numpy inputs should not raise
        pred = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        metric.validate_inputs(pred, target)

    def test_validate_inputs_valid_torch(self) -> None:
        """Test input validation with valid torch tensors."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        # Valid torch inputs (will be converted to numpy)
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        metric.validate_inputs(pred, target)

    def test_validate_inputs_invalid_type(self) -> None:
        """Test input validation with invalid input types."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        with self.assertRaises(TypeError):
            metric.validate_inputs("not_tensor", np.zeros((2, 64, 64, 3)))  # type: ignore[arg-type]

    def test_validate_inputs_mismatched_dimensions(self) -> None:
        """Test input validation with mismatched dimensions."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        with self.assertRaises(ValueError):
            pred = np.random.rand(2, 64, 64, 3)
            target = np.random.rand(3, 64, 64, 3)
            metric.validate_inputs(pred, target)

    def test_validate_inputs_wrong_number_of_channels(self) -> None:
        """Test input validation with wrong number of channels."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        with self.assertRaises(ValueError):
            pred = np.random.rand(2, 64, 64, 1)  # Grayscale
            target = np.random.rand(2, 64, 64, 3)
            metric.validate_inputs(pred, target)

    def test_validate_inputs_invalid_list_elements(self) -> None:
        """Test input validation rejects lists with invalid element types."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        valid_crop = np.random.rand(64, 64, 3).astype(np.float32)

        with self.assertRaises(TypeError):
            metric.validate_inputs([valid_crop, valid_crop], [valid_crop, "not_an_array"])

    def test_compute_with_identical_crops(self) -> None:
        """Test compute with identical crops should give high scores."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        # Identical crops should have high perceptual quality
        crops = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        pred = crops.copy()
        target = crops.copy()

        result = metric.compute(pred, target)

        self.assertIn("object_level_perceptual", result.values.keys())
        scores = result.get_value("object_level_perceptual")

        # Identical images should have high quality scores
        self.assertTrue(torch.all(scores > 0.8))
        self.assertEqual(scores.shape[0], 2)

    def test_compute_returns_all_sub_metrics(self) -> None:
        """Test that compute returns detailed_data with all sub-metrics."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        pred = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)

        result = metric.compute(pred, target)

        # After refactoring, compute() only returns detailed_data in metadata
        self.assertIn("detailed_data", result.metadata)
        detailed_data = result.metadata["detailed_data"]
        self.assertEqual(len(detailed_data), 1)

        # Verify each object has all expected sub-metrics
        obj_data = detailed_data[0]
        expected_metrics = [
            "edge_similarity",
            "gradient_similarity",
            "blur_score",
            "artifact_score",
            "ssim",
            "color_hist_similarity",
            "cem_score",
            "hue_variance_score",
            "chroma_hf_score",
            "channel_coherence_score",
            "y_chroma_ratio_score",
            "perceptual_score",
        ]

        for metric_name in expected_metrics:
            self.assertIn(metric_name, obj_data, f"Missing metric: {metric_name}")

    def test_compute_scores_in_valid_range(self) -> None:
        """Test that all computed scores are in valid range [0, 1]."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        pred = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)

        result = metric.compute(pred, target)

        # Check main score is in valid range
        scores = result.get_value("object_level_perceptual")
        self.assertTrue(torch.all(scores >= 0.0), f"object_level_perceptual has values < 0: {scores.min().item()}")
        self.assertTrue(torch.all(scores <= 1.0), f"object_level_perceptual has values > 1: {scores.max().item()}")

    def test_compute_with_different_sizes(self) -> None:
        """Test compute with crops of different sizes."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        # Different sized crops (should handle resizing internally)
        pred = np.random.randint(0, 255, (2, 32, 48, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (2, 32, 48, 3), dtype=np.uint8)

        result = metric.compute(pred, target)

        self.assertIsNotNone(result)
        self.assertIn("object_level_perceptual", result.values.keys())

    def test_compute_single_crop(self) -> None:
        """Test compute with a single crop."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        pred = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)

        result = metric.compute(pred, target)

        scores = result.get_value("object_level_perceptual")
        self.assertEqual(scores.shape[0], 1)

    def test_compute_batch_of_crops(self) -> None:
        """Test compute with a batch of crops."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        batch_size = 5
        pred = np.random.randint(0, 255, (batch_size, 64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (batch_size, 64, 64, 3), dtype=np.uint8)

        result = metric.compute(pred, target)

        scores = result.get_value("object_level_perceptual")
        self.assertEqual(scores.shape[0], batch_size)

    def test_aggregate_mean(self) -> None:
        """Test aggregation with mean method returns scalar."""
        metric = ObjectLevelPerceptualMetric(
            device=self.device,
            aggregation_methods=AggregationMethod.MEAN,
        )

        # Compute multiple times
        for _ in range(3):
            pred = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            target = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            result = metric.compute(pred, target)
            metric.append(result)

        results = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, results)
        mean_result = results[AggregationMethod.MEAN]
        self.assertIn("object_level_perceptual", mean_result.values.keys())

        # Verify MEAN returns a scalar, not a list
        mean_value = mean_result.values["object_level_perceptual"]
        self.assertTrue(torch.is_tensor(mean_value))
        self.assertEqual(mean_value.ndim, 0, "MEAN should return scalar tensor")
        self.assertGreaterEqual(mean_value.item(), 0.0)
        self.assertLessEqual(mean_value.item(), 1.0)

    def test_aggregate_multiple_methods(self) -> None:
        """Test aggregation with multiple methods."""
        metric = ObjectLevelPerceptualMetric(
            device=self.device,
            aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.MAX],
        )

        # Compute multiple times
        for _ in range(3):
            pred = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            target = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            result = metric.compute(pred, target)
            metric.append(result)

        results = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, results)
        self.assertIn(AggregationMethod.MAX, results)

    def test_reset(self) -> None:
        """Test reset method."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        # Compute some values
        pred = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        metric.compute(pred, target)

        # Reset should pass (doesn't clear for object-level metrics)
        metric.reset()
        # This is expected behavior for object-level metrics

    def test_compute_with_float_inputs(self) -> None:
        """Test compute with float inputs in [0, 1] range."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        # Float inputs in [0, 1]
        pred = np.random.rand(2, 64, 64, 3).astype(np.float32)
        target = np.random.rand(2, 64, 64, 3).astype(np.float32)

        result = metric.compute(pred, target)

        self.assertIsNotNone(result)
        self.assertIn("object_level_perceptual", result.values.keys())

    def test_compute_with_torch_inputs(self) -> None:
        """Test compute with torch tensor inputs."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        # Torch tensors in NCHW format
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        result = metric.compute(pred, target)

        self.assertIsNotNone(result)
        self.assertIn("object_level_perceptual", result.values.keys())

    def test_perceptual_score_is_weighted_combination(self) -> None:
        """Test that perceptual_score is properly weighted."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        pred = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)

        result = metric.compute(pred, target)

        # After refactoring, individual metrics are in detailed_data
        self.assertIn("detailed_data", result.metadata)
        obj_data = result.metadata["detailed_data"][0]

        edge_sim = obj_data["edge_similarity"]
        gradient_sim = obj_data["gradient_similarity"]
        perceptual_score = obj_data["perceptual_score"]

        # Perceptual score should be a combination (not equal to individual metrics)
        self.assertIsNotNone(perceptual_score)
        # Score should be different from individual metrics (weighted combination)
        self.assertNotEqual(perceptual_score, edge_sim)
        self.assertNotEqual(perceptual_score, gradient_sim)

        # Verify scores are in valid range
        self.assertGreaterEqual(perceptual_score, 0.0)
        self.assertLessEqual(perceptual_score, 1.0)

    def test_aggregate_sum_returns_scalar(self) -> None:
        """Test that SUM aggregation returns scalar."""
        metric = ObjectLevelPerceptualMetric(
            device=self.device,
            aggregation_methods=AggregationMethod.SUM,
        )

        # Compute multiple times with 2 objects each
        for _ in range(3):
            pred = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            target = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            result = metric.compute(pred, target)
            metric.append(result)

        results = metric.aggregate()
        sum_result = results[AggregationMethod.SUM]
        sum_value = sum_result.values["object_level_perceptual"]

        # Verify SUM returns scalar
        self.assertTrue(torch.is_tensor(sum_value))
        self.assertEqual(sum_value.ndim, 0, "SUM should return scalar tensor")

    def test_aggregate_min_max_return_scalars(self) -> None:
        """Test that MIN and MAX aggregations return scalars."""
        metric = ObjectLevelPerceptualMetric(
            device=self.device,
            aggregation_methods=[AggregationMethod.MIN, AggregationMethod.MAX],
        )

        # Compute multiple times
        for _ in range(3):
            pred = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            target = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
            result = metric.compute(pred, target)
            metric.append(result)

        results = metric.aggregate()

        # Verify MIN returns scalar
        min_result = results[AggregationMethod.MIN]
        min_value = min_result.values["object_level_perceptual"]
        self.assertTrue(torch.is_tensor(min_value))
        self.assertEqual(min_value.ndim, 0, "MIN should return scalar tensor")

        # Verify MAX returns scalar
        max_result = results[AggregationMethod.MAX]
        max_value = max_result.values["object_level_perceptual"]
        self.assertTrue(torch.is_tensor(max_value))
        self.assertEqual(max_value.ndim, 0, "MAX should return scalar tensor")

        # MIN should be <= MAX
        self.assertLessEqual(min_value.item(), max_value.item())

    def test_aggregate_per_track(self) -> None:
        """Test MEAN aggregation includes per-track data in metadata."""
        metric = ObjectLevelPerceptualMetric(
            device=self.device,
            aggregation_methods=AggregationMethod.MEAN,
        )

        # Compute with track IDs and class names
        track_ids = ["track_1", "track_1", "track_2"]
        class_names = ["car", "car", "truck"]

        for _ in range(2):
            pred = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
            target = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
            result = metric.compute(
                pred, target, obj_metadata=ObjectMetadata(track_ids=track_ids, class_names=class_names)
            )
            metric.append(result)

        results = metric.aggregate()
        mean_result = results[AggregationMethod.MEAN]

        # Verify per-track data is in MEAN metadata
        self.assertIn("per_track", mean_result.metadata)
        per_track_data = mean_result.metadata["per_track"]

        # Should have 2 tracks
        self.assertEqual(len(per_track_data), 2)
        self.assertIn("track_1", per_track_data)
        self.assertIn("track_2", per_track_data)

        # Verify track_1 has correct statistics
        track_1_metrics = per_track_data["track_1"]
        self.assertIn("perceptual_score_mean", track_1_metrics)
        self.assertIn("perceptual_score_std", track_1_metrics)
        self.assertIn("perceptual_score_min", track_1_metrics)
        self.assertIn("perceptual_score_max", track_1_metrics)
        self.assertIn("num_frames", track_1_metrics)
        self.assertIn("class_name", track_1_metrics)

    def test_aggregate_per_class(self) -> None:
        """Test MEAN aggregation includes per-class data in metadata."""
        metric = ObjectLevelPerceptualMetric(
            device=self.device,
            aggregation_methods=AggregationMethod.MEAN,
        )

        # Compute with track IDs and class names
        track_ids = ["track_1", "track_2", "track_3"]
        class_names = ["car", "car", "truck"]

        for _ in range(2):
            pred = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
            target = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
            result = metric.compute(
                pred, target, obj_metadata=ObjectMetadata(track_ids=track_ids, class_names=class_names)
            )
            metric.append(result)

        results = metric.aggregate()
        mean_result = results[AggregationMethod.MEAN]

        # Verify per-class data is in MEAN metadata
        self.assertIn("per_class", mean_result.metadata)
        per_class_data = mean_result.metadata["per_class"]

        # Should have 2 classes
        self.assertEqual(len(per_class_data), 2)
        self.assertIn("car", per_class_data)
        self.assertIn("truck", per_class_data)

        # Verify car class has correct statistics
        car_metrics = per_class_data["car"]
        self.assertIn("perceptual_score_mean", car_metrics)
        self.assertIn("perceptual_score_std", car_metrics)
        self.assertIn("num_objects", car_metrics)
        self.assertIn("num_tracks", car_metrics)

        # Car class should have 2 tracks and 4 objects (2 per frame * 2 frames)
        self.assertEqual(car_metrics["num_tracks"], 2)
        self.assertEqual(car_metrics["num_objects"], 4)

    def test_validate_inputs_invalid_float_range(self) -> None:
        """Test input validation rejects float values outside [0, 1]."""
        metric = ObjectLevelPerceptualMetric(device=self.device)

        # Float numpy inputs outside [0, 1] should raise ValueError
        pred = np.random.rand(2, 64, 64, 3).astype(np.float32) * 2.0  # [0, 2]
        target = np.random.rand(2, 64, 64, 3).astype(np.float32)

        with self.assertRaises(ValueError):
            metric.validate_inputs(pred, target)

        # Float torch inputs outside [0, 1] should also raise ValueError
        pred_torch = torch.rand(3, 64, 64) * 2.0  # [0, 2]
        target_torch = torch.rand(3, 64, 64)

        with self.assertRaises(ValueError):
            metric.validate_inputs(pred_torch, target_torch)


if __name__ == "__main__":
    unittest.main()
