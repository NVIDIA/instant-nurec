# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for generic feature drift metric implementation."""

import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nre.metrics.impl.drift import FeatureDriftMetric
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod


class TestFeatureDriftMetric(unittest.TestCase):
    """Test cases for FeatureDriftMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = "cpu"
        self.extractor_type = "segformer"
        self.pretrained_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.window_size = 5

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization(self, mock_create_extractor: MagicMock) -> None:
        """Test FeatureDriftMetric initialization."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            window_size=self.window_size,
        )

        self.assertEqual(metric.extractor_type, self.extractor_type)
        self.assertEqual(metric.pretrained_path, self.pretrained_path)
        self.assertEqual(metric.window_size, self.window_size)
        mock_create_extractor.assert_called_once_with(
            extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, cache_dir=None, device=self.device
        )

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_weighted_mean_not_supported(self, mock_create_extractor: MagicMock) -> None:
        """Test that weighted mean aggregation raises ValueError."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        with self.assertRaises(ValueError) as context:
            FeatureDriftMetric(
                device=self.device,
                aggregation_methods=AggregationMethod.WEIGHTED_MEAN,
                extractor_type=self.extractor_type,
                pretrained_path=self.pretrained_path,
            )

        self.assertIn("Weighted mean is not supported", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_valid(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with valid inputs."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            window_size=self.window_size,
        )

        # Valid inputs should not raise
        pred = torch.rand(10, 3, 64, 64)  # Sequence of 10 frames
        target = torch.rand(10, 3, 64, 64)
        metric.validate_inputs(pred, target)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_invalid_dimensions(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with invalid dimensions."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            window_size=self.window_size,
        )

        # Should have 4 dimensions [T, C, H, W]
        pred = torch.rand(3, 64, 64)  # Missing time dimension
        target = torch.rand(3, 64, 64)

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("4 dimensions", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_short_sequence(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with sequence shorter than window size."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=10
        )

        # Sequence too short
        pred = torch.rand(5, 3, 64, 64)  # Only 5 frames, need 10
        target = torch.rand(5, 3, 64, 64)

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("at least window_size", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_drift_statistics(self, mock_create_extractor: MagicMock) -> None:
        """Test drift statistics computation."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=3
        )

        # Create test features (GT and generated)
        gt_features = torch.randn(10, 512, device=self.device)
        gen_features = torch.randn(10, 512, device=self.device)

        avg_drift, drift_values, drift_correlation = metric._compute_drift_statistics(gt_features, gen_features)

        self.assertIsInstance(avg_drift, torch.Tensor)
        self.assertIsInstance(drift_values, torch.Tensor)
        self.assertIsInstance(drift_correlation, torch.Tensor)
        self.assertGreaterEqual(float(avg_drift.item()), 0.0)
        self.assertEqual(len(drift_values), 10)  # One value per frame

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the _compute method."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = np.random.randn(10, 512)
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=5
        )

        pred = torch.rand(10, 3, 64, 64)
        target = torch.rand(10, 3, 64, 64)

        result = metric._compute(pred, target)

        self.assertIn(metric._NAME, result.values)
        self.assertIn(f"{metric._NAME}_correlation", result.values)
        self.assertIsInstance(result.values[metric._NAME], torch.Tensor)
        self.assertIsInstance(result.values[f"{metric._NAME}_correlation"], torch.Tensor)
        self.assertIn("extractor_type", result.metadata)
        self.assertIn("window_size", result.metadata)
        self.assertIn("avg_drift", result.metadata)
        self.assertIn("drift_correlation", result.metadata)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_type_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the type method."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        self.assertEqual(metric.type(), MetricType.FEATURE_DRIFT)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_metadata_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the metadata method."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            window_size=self.window_size,
        )

        metadata = metric.metadata()

        self.assertEqual(metadata["extractor_type"], self.extractor_type)
        self.assertEqual(metadata["pretrained_path"], self.pretrained_path)
        self.assertEqual(metadata["feature_dim"], 512)
        self.assertEqual(metadata["window_size"], self.window_size)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_aggregate_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the aggregate method."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        # Add some test values
        from nre.metrics.metric import MetricResult

        test_result = MetricResult(
            values={
                metric._NAME: torch.tensor(0.5),
                f"{metric._NAME}_correlation": torch.tensor(0.3),
            },
            metadata={},
        )
        metric._values.append(test_result)
        metric._values.append(test_result)

        aggregated = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, aggregated)
        self.assertIn(metric._NAME, aggregated[AggregationMethod.MEAN].values)
        self.assertIn(f"{metric._NAME}_correlation", aggregated[AggregationMethod.MEAN].values)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_drift_statistics_identical_sequences(self, mock_create_extractor: MagicMock) -> None:
        """Test drift computation with identical sequences (zero drift)."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=3
        )

        # Identical features should produce zero drift
        features = torch.randn(5, 512, device=self.device)
        gt_features = features.clone()
        gen_features = features.clone()

        avg_drift, drift_values, correlation = metric._compute_drift_statistics(gt_features, gen_features)

        # Should have zero or near-zero drift
        self.assertAlmostEqual(float(avg_drift.item()), 0.0, places=10)
        # All drift values should be zero
        torch.testing.assert_close(drift_values, torch.zeros(5), atol=1e-10, rtol=1e-10)
        # Correlation should be 0 (constant values)
        self.assertEqual(float(correlation.item()), 0.0)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_drift_statistics_progressive_drift(self, mock_create_extractor: MagicMock) -> None:
        """Test drift with progressively increasing differences."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=3
        )

        # Create features with increasing drift over time
        torch.manual_seed(42)  # For reproducible results
        gt_features = torch.randn(10, 512, device=self.device)
        gen_features = gt_features.clone()

        # Add progressively larger noise
        for i in range(10):
            noise_scale = i * 0.5  # Increasing noise
            torch.manual_seed(42 + i)  # Different seed for each frame
            gen_features[i] += torch.randn(512, device=self.device) * noise_scale

        _, drift_values, correlation = metric._compute_drift_statistics(gt_features, gen_features)

        # Should have positive correlation (drift increases over time)
        self.assertGreater(float(correlation.item()), 0.0)
        # Later drift values should generally be larger than earlier ones
        self.assertGreater(float(torch.mean(drift_values[-3:]).item()), float(torch.mean(drift_values[:3]).item()))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_drift_statistics_decreasing_drift(self, mock_create_extractor: MagicMock) -> None:
        """Test drift with progressively decreasing differences."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=3
        )

        # Create features with decreasing drift over time
        torch.manual_seed(42)  # For reproducible results
        gt_features = torch.randn(10, 512, device=self.device)
        gen_features = gt_features.clone()

        # Add progressively smaller noise
        for i in range(10):
            noise_scale = (10 - i) * 0.5  # Decreasing noise
            torch.manual_seed(42 + i)  # Different seed for each frame
            gen_features[i] += torch.randn(512, device=self.device) * noise_scale

        _, _, correlation = metric._compute_drift_statistics(gt_features, gen_features)

        # Should have negative correlation (drift decreases over time)
        self.assertLess(float(correlation.item()), 0.0)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_drift_statistics_constant_values(self, mock_create_extractor: MagicMock) -> None:
        """Test drift computation when all drift values are identical."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FeatureDriftMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=3
        )

        # Create features with constant distance between gt and gen
        gt_features = torch.zeros(5, 512, device=self.device)
        gen_features = torch.ones(5, 512, device=self.device)  # Constant distance

        avg_drift, drift_values, correlation = metric._compute_drift_statistics(gt_features, gen_features)

        # All drift values should be identical
        self.assertTrue(torch.allclose(drift_values, drift_values[0]))
        # Correlation should be 0.0 (handled by std == 0 case)
        self.assertEqual(float(correlation.item()), 0.0)
        # Average should equal individual values
        self.assertAlmostEqual(float(avg_drift.item()), float(drift_values[0].item()))


if __name__ == "__main__":
    unittest.main()
