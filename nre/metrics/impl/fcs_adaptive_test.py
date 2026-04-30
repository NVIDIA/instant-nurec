# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for FCS Adaptive metric implementation."""

import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nre.metrics.impl.fcs_adaptive import FCSAdaptiveMetric
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod


class TestFCSAdaptiveMetric(unittest.TestCase):
    """Test cases for FCSAdaptiveMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = "cpu"
        self.extractor_type = "segformer"
        self.pretrained_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.n_neighbors = 5
        self.feature_batch_size = 32

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization(self, mock_create_extractor: MagicMock) -> None:
        """Test FCSAdaptiveMetric initialization."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            n_neighbors=self.n_neighbors,
            feature_batch_size=self.feature_batch_size,
        )

        self.assertEqual(metric.extractor_type, self.extractor_type)
        self.assertEqual(metric.pretrained_path, self.pretrained_path)
        self.assertEqual(metric.n_neighbors, self.n_neighbors)
        self.assertEqual(metric.feature_batch_size, self.feature_batch_size)
        mock_create_extractor.assert_called_once_with(
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            cache_dir=None,
            device=self.device,
        )

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_weighted_mean_not_supported(self, mock_create_extractor: MagicMock) -> None:
        """Test that weighted mean aggregation raises ValueError."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        with self.assertRaises(ValueError):
            FCSAdaptiveMetric(
                device=self.device,
                aggregation_methods=AggregationMethod.WEIGHTED_MEAN,
            )

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_correct(self, mock_create_extractor: MagicMock) -> None:
        """Test validate_inputs with correct inputs."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=self.n_neighbors)

        # Valid 4D tensors with enough samples
        pred = torch.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 64, 64)

        # Should not raise any exception
        metric.validate_inputs(pred, target)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_insufficient_samples(self, mock_create_extractor: MagicMock) -> None:
        """Test validate_inputs with insufficient samples for k-NN."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=5)

        # Too few samples for k-NN (need at least k+1 samples)
        pred = torch.randn(3, 3, 64, 64)  # Only 3 samples, need 6
        target = torch.randn(3, 3, 64, 64)

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("Need at least 6 samples", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_shape_mismatch(self, mock_create_extractor: MagicMock) -> None:
        """Test validate_inputs with shape mismatch."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device)

        pred = torch.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 32, 32)  # Different spatial dimensions

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("shapes must match", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_wrong_type(self, mock_create_extractor: MagicMock) -> None:
        """Test validate_inputs with wrong input type."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device)

        # numpy array instead of tensor
        pred = np.random.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 64, 64)

        with self.assertRaises(TypeError) as context:
            metric.validate_inputs(pred, target)  # type: ignore[arg-type]

        self.assertIn("Input 0 must be a torch.Tensor", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_adaptive_fcs_basic(self, mock_create_extractor: MagicMock) -> None:
        """Test compute_adaptive_fcs with basic functionality."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=3)

        # Create test features with known structure
        np.random.seed(42)
        gt_features = np.random.randn(20, 10)
        gen_features = np.random.randn(20, 10)

        (
            fcs_adaptive,
            fcs_gt_to_gen,
            fcs_gen_to_gt,
            covered_mask,
        ) = metric.compute_adaptive_fcs(gt_features, gen_features, k=3)

        # Check return types and ranges
        self.assertIsInstance(fcs_adaptive, float)
        self.assertIsInstance(fcs_gt_to_gen, float)
        self.assertIsInstance(fcs_gen_to_gt, float)
        self.assertIsInstance(covered_mask, np.ndarray)

        # FCS scores should be in [0, 1] range
        self.assertGreaterEqual(fcs_adaptive, 0.0)
        self.assertLessEqual(fcs_adaptive, 1.0)
        self.assertGreaterEqual(fcs_gt_to_gen, 0.0)
        self.assertLessEqual(fcs_gt_to_gen, 1.0)
        self.assertGreaterEqual(fcs_gen_to_gt, 0.0)
        self.assertLessEqual(fcs_gen_to_gt, 1.0)

        # Adaptive FCS should be average of directional scores
        expected_adaptive = 0.5 * (fcs_gt_to_gen + fcs_gen_to_gt)
        self.assertAlmostEqual(fcs_adaptive, expected_adaptive, places=10)

        # Covered mask should have correct shape and type
        self.assertEqual(covered_mask.shape, (20,))
        self.assertEqual(covered_mask.dtype, bool)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_adaptive_fcs_identical_features(self, mock_create_extractor: MagicMock) -> None:
        """Test compute_adaptive_fcs with identical features."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=3)

        # Create identical features
        features = np.random.randn(15, 8)
        gt_features = features.copy()
        gen_features = features.copy()

        (
            fcs_adaptive,
            fcs_gt_to_gen,
            fcs_gen_to_gt,
            covered_mask,
        ) = metric.compute_adaptive_fcs(gt_features, gen_features, k=3)

        # With identical features, coverage should be perfect
        self.assertAlmostEqual(fcs_adaptive, 1.0, places=10)
        self.assertAlmostEqual(fcs_gt_to_gen, 1.0, places=10)
        self.assertAlmostEqual(fcs_gen_to_gt, 1.0, places=10)

        # All samples should be covered
        self.assertTrue(np.all(covered_mask))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_adaptive_fcs_empty_features(self, mock_create_extractor: MagicMock) -> None:
        """Test compute_adaptive_fcs with empty features."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=3)

        # Empty features arrays
        gt_features = np.empty((0, 10))
        gen_features = np.empty((0, 10))

        with self.assertRaises((ValueError, IndexError)):
            metric.compute_adaptive_fcs(gt_features, gen_features, k=3)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_fcs_adaptive_metrics_basic(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute_fcs_adaptive_metrics with basic functionality."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=5)

        # Create test features
        np.random.seed(42)
        gt_features = np.random.randn(50, 128)
        gen_features = np.random.randn(50, 128)

        metrics = metric._compute_fcs_adaptive_metrics(gt_features, gen_features)

        # Check that all expected metrics are present
        expected_keys = [
            "fcs_adaptive",
            "fcs_gt_to_gen",
            "fcs_gen_to_gt",
            "coverage_ratio",
            "uncovered_samples",
            "avg_covered_distance",
            "std_covered_distance",
            "avg_threshold",
            "std_threshold",
            "n_neighbors",
            "gt_samples",
            "gen_samples",
            "feature_dim",
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check value ranges and types
        self.assertGreaterEqual(metrics["fcs_adaptive"], 0.0)
        self.assertLessEqual(metrics["fcs_adaptive"], 1.0)
        self.assertGreaterEqual(metrics["coverage_ratio"], 0.0)
        self.assertLessEqual(metrics["coverage_ratio"], 1.0)
        self.assertGreaterEqual(metrics["uncovered_samples"], 0)
        self.assertLessEqual(metrics["uncovered_samples"], 50)

        # Check sample and dimension counts
        self.assertEqual(metrics["gt_samples"], 50)
        self.assertEqual(metrics["gen_samples"], 50)
        self.assertEqual(metrics["feature_dim"], 128)
        self.assertEqual(metrics["n_neighbors"], 5)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_basic(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute method with basic functionality."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = np.random.randn(10, 512)
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=3)

        pred = torch.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 64, 64)

        result = metric._compute(pred, target)

        # Check result structure
        self.assertIn("fcs_adaptive", result.values)
        self.assertIsInstance(result.values["fcs_adaptive"], torch.Tensor)

        # Check metadata
        self.assertIn("original_shape", result.metadata)
        self.assertIn("extractor_type", result.metadata)
        self.assertIn("n_neighbors", result.metadata)
        self.assertIn("feature_batch_size", result.metadata)

        # Verify feature extractor was called twice
        self.assertEqual(mock_extractor.extract_features_batch.call_count, 2)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_3d_input(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute method with 3D input (single image)."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = np.random.randn(10, 512)
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=5)

        # 3D input (single image) - should work with proper mocking
        pred = torch.randn(3, 64, 64)
        target = torch.randn(3, 64, 64)

        # Should handle 3D input by adding batch dimension
        result = metric._compute(pred, target)
        self.assertIn("fcs_adaptive", result.values)
        self.assertIsInstance(result.values["fcs_adaptive"], torch.Tensor)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_type(self, mock_create_extractor: MagicMock) -> None:
        """Test type method returns correct MetricType."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device)

        self.assertEqual(metric.type(), MetricType.FCS_ADAPTIVE)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_metadata(self, mock_create_extractor: MagicMock) -> None:
        """Test metadata method returns correct information."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            n_neighbors=self.n_neighbors,
            feature_batch_size=self.feature_batch_size,
        )

        metadata = metric.metadata()

        expected_keys = [
            "extractor_type",
            "pretrained_path",
            "n_neighbors",
            "feature_batch_size",
        ]
        for key in expected_keys:
            self.assertIn(key, metadata)

        self.assertEqual(metadata["extractor_type"], self.extractor_type)
        self.assertEqual(metadata["pretrained_path"], self.pretrained_path)
        self.assertEqual(metadata["n_neighbors"], self.n_neighbors)
        self.assertEqual(metadata["feature_batch_size"], self.feature_batch_size)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_aggregation(self, mock_create_extractor: MagicMock) -> None:
        """Test aggregation functionality."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, aggregation_methods=AggregationMethod.MEAN)

        # Add some test results to internal state
        from nre.metrics.metric import MetricResult

        metric._values = [
            MetricResult(
                values={"fcs_adaptive": torch.tensor(0.7)},
                metadata={"test": "data"},
            ),
            MetricResult(
                values={"fcs_adaptive": torch.tensor(0.8)},
                metadata={"test": "data"},
            ),
            MetricResult(
                values={"fcs_adaptive": torch.tensor(0.9)},
                metadata={"test": "data"},
            ),
        ]

        # Test aggregation
        aggregated = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, aggregated)
        result = aggregated[AggregationMethod.MEAN]
        self.assertIn("fcs_adaptive", result.values)

        # Mean of [0.7, 0.8, 0.9] should be 0.8
        self.assertAlmostEqual(result.values["fcs_adaptive"].item(), 0.8, places=5)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_different_k_values(self, mock_create_extractor: MagicMock) -> None:
        """Test compute_adaptive_fcs with different k values."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=5)

        # Create test features
        np.random.seed(42)
        gt_features = np.random.randn(30, 10)
        gen_features = np.random.randn(30, 10)

        # Test with different k values
        for k in [1, 3, 5, 10]:
            if k < gt_features.shape[0]:
                (
                    fcs_adaptive,
                    _,  # fcs_gt_to_gen (unused)
                    _,  # fcs_gen_to_gt (unused)
                    covered_mask,
                ) = metric.compute_adaptive_fcs(gt_features, gen_features, k=k)

                # All scores should be valid
                self.assertGreaterEqual(fcs_adaptive, 0.0)
                self.assertLessEqual(fcs_adaptive, 1.0)
                self.assertEqual(len(covered_mask), gt_features.shape[0])

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_coverage_statistics(self, mock_create_extractor: MagicMock) -> None:
        """Test that coverage statistics are computed correctly."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FCSAdaptiveMetric(device=self.device, n_neighbors=3)

        # Create features where some GT samples are far from Gen
        # samples
        gt_features = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11]])
        gen_features = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [0, 1], [1, 2]])

        metrics = metric._compute_fcs_adaptive_metrics(gt_features, gen_features)

        # Check that coverage statistics make sense
        self.assertGreaterEqual(metrics["coverage_ratio"], 0.0)
        self.assertLessEqual(metrics["coverage_ratio"], 1.0)
        self.assertGreaterEqual(metrics["uncovered_samples"], 0)
        self.assertLessEqual(metrics["uncovered_samples"], 5)

        # Coverage ratio should match uncovered samples
        expected_coverage = 1.0 - (metrics["uncovered_samples"] / 5.0)
        self.assertAlmostEqual(metrics["coverage_ratio"], expected_coverage, places=10)


if __name__ == "__main__":
    unittest.main()
