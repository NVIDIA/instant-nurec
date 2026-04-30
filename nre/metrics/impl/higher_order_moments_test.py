# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for Higher-Order Moments metric implementation."""

import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nre.metrics.impl.higher_order_moments import (
    HigherOrderMomentsMetric,
    MomentType,
    create_d_kurtosis_metric,
    create_d_skew_metric,
)
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod


class TestHigherOrderMomentsMetric(unittest.TestCase):
    """Test cases for HigherOrderMomentsMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = "cpu"
        self.extractor_type = "segformer"
        self.pretrained_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.feature_batch_size = 32

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization_skewness(self, mock_create_extractor: MagicMock) -> None:
        """Test HigherOrderMomentsMetric initialization for skewness."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(
            moment_type=MomentType.SKEWNESS,
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            feature_batch_size=self.feature_batch_size,
        )

        self.assertEqual(metric.moment_type, MomentType.SKEWNESS)
        self.assertEqual(metric.extractor_type, self.extractor_type)
        self.assertEqual(metric.pretrained_path, self.pretrained_path)
        self.assertEqual(metric.feature_batch_size, self.feature_batch_size)
        mock_create_extractor.assert_called_once_with(
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            cache_dir=None,
            device=self.device,
        )

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization_kurtosis(self, mock_create_extractor: MagicMock) -> None:
        """Test HigherOrderMomentsMetric initialization for kurtosis."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(
            moment_type=MomentType.KURTOSIS,
            device=self.device,
        )

        self.assertEqual(metric.moment_type, MomentType.KURTOSIS)
        mock_create_extractor.assert_called_once()

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization_string_moment_type(self, mock_create_extractor: MagicMock) -> None:
        """Test initialization with string moment type."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(
            moment_type="skewness",
            device=self.device,
        )

        self.assertEqual(metric.moment_type, MomentType.SKEWNESS)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_weighted_mean_not_supported(self, mock_create_extractor: MagicMock) -> None:
        """Test that weighted mean aggregation raises ValueError."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        with self.assertRaises(ValueError):
            HigherOrderMomentsMetric(
                device=self.device,
                aggregation_methods=AggregationMethod.WEIGHTED_MEAN,
            )

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_correct(self, mock_create_extractor: MagicMock) -> None:
        """Test validate_inputs with correct inputs."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(device=self.device)

        # Valid 4D tensors
        pred = torch.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 64, 64)

        # Should not raise any exception
        metric.validate_inputs(pred, target)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_shape_mismatch(self, mock_create_extractor: MagicMock) -> None:
        """Test validate_inputs with shape mismatch."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(device=self.device)

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

        metric = HigherOrderMomentsMetric(device=self.device)

        # numpy array instead of tensor
        pred = np.random.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 64, 64, device=self.device)

        with self.assertRaises(TypeError) as context:
            metric.validate_inputs(pred, target)  # type: ignore[arg-type]

        self.assertIn("Input 0 must be a torch.Tensor", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_moments_skewness(self, mock_create_extractor: MagicMock) -> None:
        """Test compute_moments for skewness."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.SKEWNESS, device=self.device)

        # Create test features with known properties
        features = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=self.device)

        moments = metric.compute_moments(features)

        # Check that expected keys are present for skewness
        expected_keys = ["mean", "centered", "variance", "std", "skewness"]
        for key in expected_keys:
            self.assertIn(key, moments)

        # Check shapes
        self.assertEqual(moments["mean"].shape, (3,))
        self.assertEqual(moments["centered"].shape, (3, 3))
        self.assertEqual(moments["skewness"].shape, (3,))

        # Check that mean is computed correctly
        expected_mean = torch.tensor([4.0, 5.0, 6.0], device=self.device)
        torch.testing.assert_close(moments["mean"], expected_mean)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_moments_kurtosis(self, mock_create_extractor: MagicMock) -> None:
        """Test compute_moments for kurtosis."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.KURTOSIS, device=self.device)

        # Create test features
        features = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=self.device)

        moments = metric.compute_moments(features)

        # Check that expected keys are present for kurtosis
        expected_keys = [
            "mean",
            "centered",
            "variance",
            "std",
            "kurtosis",
            "normalized_kurtosis",
            "excess_kurtosis",
        ]
        for key in expected_keys:
            self.assertIn(key, moments)

        # Check shapes
        self.assertEqual(moments["kurtosis"].shape, (3,))
        self.assertEqual(moments["excess_kurtosis"].shape, (3,))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_moments_empty_features(self, mock_create_extractor: MagicMock) -> None:
        """Test compute_moments with empty features."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(device=self.device)

        # Empty features tensor
        features = torch.empty((0, 3), device=self.device)

        with self.assertRaises(ValueError):
            metric.compute_moments(features)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_moment_metrics_skewness(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute_moment_metrics for skewness."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.SKEWNESS, device=self.device)

        # Create test features
        gt_features = torch.randn(100, 512, device=self.device)
        gen_features = torch.randn(100, 512, device=self.device)

        metrics = metric._compute_moment_metrics(gt_features, gen_features)

        # Check that all expected skewness metrics are present
        expected_keys = [
            "d_skew",
            "gt_skewness_mean",
            "gen_skewness_mean",
            "gt_skewness_std",
            "gen_skewness_std",
            "max_skewness_diff",
            "mean_skewness_diff",
            "skewness_diff_ratio",
            "gt_samples",
            "gen_samples",
            "feature_dim",
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check that d_skew is non-negative
        self.assertGreaterEqual(metrics["d_skew"].item(), 0.0)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_moment_metrics_kurtosis(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute_moment_metrics for kurtosis."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.KURTOSIS, device=self.device)

        # Create test features
        gt_features = torch.randn(100, 512, device=self.device)
        gen_features = torch.randn(100, 512, device=self.device)

        metrics = metric._compute_moment_metrics(gt_features, gen_features)

        # Check that all expected kurtosis metrics are present
        expected_keys = [
            "d_kurt",
            "d_kurt_normalized",
            "d_kurt_excess",
            "gt_kurtosis_mean",
            "gen_kurtosis_mean",
            "gt_kurtosis_std",
            "gen_kurtosis_std",
            "max_kurtosis_diff",
            "mean_kurtosis_diff",
            "kurtosis_diff_ratio",
            "gt_excess_kurtosis_mean",
            "gen_excess_kurtosis_mean",
            "gt_samples",
            "gen_samples",
            "feature_dim",
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

        # Check that d_kurt values are non-negative
        self.assertGreaterEqual(metrics["d_kurt"].item(), 0.0)
        self.assertGreaterEqual(metrics["d_kurt_normalized"].item(), 0.0)
        self.assertGreaterEqual(metrics["d_kurt_excess"].item(), 0.0)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_identical_features_skewness(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute_moment_metrics with identical features for
        skewness."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.SKEWNESS, device=self.device)

        # Create identical features
        features = torch.randn(50, 256, device=self.device)
        gt_features = features.clone()
        gen_features = features.clone()

        metrics = metric._compute_moment_metrics(gt_features, gen_features)

        # D-Skew should be zero for identical features
        self.assertAlmostEqual(metrics["d_skew"].item(), 0.0, places=10)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_identical_features_kurtosis(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute_moment_metrics with identical features for
        kurtosis."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.KURTOSIS, device=self.device)

        # Create identical features
        features = torch.randn(50, 256, device=self.device)
        gt_features = features.clone()
        gen_features = features.clone()

        metrics = metric._compute_moment_metrics(gt_features, gen_features)

        # D-Kurtosis should be zero for identical features
        self.assertAlmostEqual(metrics["d_kurt"].item(), 0.0, places=10)
        self.assertAlmostEqual(metrics["d_kurt_excess"].item(), 0.0, places=10)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_basic_skewness(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute method for skewness."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = torch.randn(10, 512, device=self.device)
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.SKEWNESS, device=self.device)

        pred = torch.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 64, 64)

        result = metric._compute(pred, target)

        # Check result structure for skewness
        self.assertIn("d_skew", result.values)
        self.assertIsInstance(result.values["d_skew"], torch.Tensor)

        # Check metadata
        self.assertIn("moment_type", result.metadata)
        self.assertEqual(result.metadata["moment_type"], "skewness")

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_basic_kurtosis(self, mock_create_extractor: MagicMock) -> None:
        """Test _compute method for kurtosis."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = torch.randn(10, 512, device=self.device)
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.KURTOSIS, device=self.device)

        pred = torch.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 64, 64)

        result = metric._compute(pred, target)

        # Check result structure for kurtosis
        self.assertIn("d_kurt", result.values)
        self.assertIsInstance(result.values["d_kurt"], torch.Tensor)

        # Check metadata
        self.assertIn("moment_type", result.metadata)
        self.assertEqual(result.metadata["moment_type"], "kurtosis")

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_type_property_skewness(self, mock_create_extractor: MagicMock) -> None:
        """Test type property returns correct MetricType for skewness."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.SKEWNESS, device=self.device)

        self.assertEqual(metric.type(), MetricType.D_SKEW)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_type_property_kurtosis(self, mock_create_extractor: MagicMock) -> None:
        """Test type property returns correct MetricType for kurtosis."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(moment_type=MomentType.KURTOSIS, device=self.device)

        self.assertEqual(metric.type(), MetricType.D_KURT)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_metadata(self, mock_create_extractor: MagicMock) -> None:
        """Test metadata method returns correct information."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(
            moment_type=MomentType.SKEWNESS,
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            feature_batch_size=self.feature_batch_size,
        )

        metadata = metric.metadata()

        expected_keys = [
            "moment_type",
            "extractor_type",
            "pretrained_path",
            "feature_batch_size",
        ]
        for key in expected_keys:
            self.assertIn(key, metadata)

        self.assertEqual(metadata["moment_type"], "skewness")
        self.assertEqual(metadata["extractor_type"], self.extractor_type)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_factory_functions(self, mock_create_extractor: MagicMock) -> None:
        """Test convenience factory functions."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        # Test D-Skew factory
        skew_metric = create_d_skew_metric(device=self.device)
        self.assertEqual(skew_metric.moment_type, MomentType.SKEWNESS)
        self.assertEqual(skew_metric.type(), MetricType.D_SKEW)

        # Test D-Kurtosis factory
        kurt_metric = create_d_kurtosis_metric(device=self.device)
        self.assertEqual(kurt_metric.moment_type, MomentType.KURTOSIS)
        self.assertEqual(kurt_metric.type(), MetricType.D_KURT)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_aggregation(self, mock_create_extractor: MagicMock) -> None:
        """Test aggregation functionality."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = HigherOrderMomentsMetric(
            moment_type=MomentType.SKEWNESS,
            device=self.device,
            aggregation_methods=AggregationMethod.MEAN,
        )

        # Add some test results to internal state
        from nre.metrics.metric import MetricResult

        metric._values = [
            MetricResult(
                values={"d_skew": torch.tensor(1.0)},
                metadata={"test": "data"},
            ),
            MetricResult(
                values={"d_skew": torch.tensor(2.0)},
                metadata={"test": "data"},
            ),
            MetricResult(
                values={"d_skew": torch.tensor(3.0)},
                metadata={"test": "data"},
            ),
        ]

        # Test aggregation
        aggregated = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, aggregated)
        result = aggregated[AggregationMethod.MEAN]
        self.assertIn("d_skew", result.values)

        # Mean of [1.0, 2.0, 3.0] should be 2.0
        self.assertAlmostEqual(result.values["d_skew"].item(), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
