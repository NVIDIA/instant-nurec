# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for generic temporal coherence metric implementation."""

import unittest

from unittest.mock import MagicMock, patch

import torch

from nre.metrics.impl.temporal_coherence import TemporalCoherenceMetric
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod


class TestTemporalCoherenceMetric(unittest.TestCase):
    """Test cases for TemporalCoherenceMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = "cpu"
        self.extractor_type = "segformer"
        self.pretrained_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.window_size = 5

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization(self, mock_create_extractor: MagicMock) -> None:
        """Test TemporalCoherenceMetric initialization."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = TemporalCoherenceMetric(
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
            TemporalCoherenceMetric(
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

        metric = TemporalCoherenceMetric(
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

        metric = TemporalCoherenceMetric(
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

        metric = TemporalCoherenceMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=10
        )

        # Sequence too short
        pred = torch.rand(5, 3, 64, 64)  # Only 5 frames, need 10
        target = torch.rand(5, 3, 64, 64)

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("at least window_size", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_temporal_coherence(self, mock_create_extractor: MagicMock) -> None:
        """Test temporal coherence computation."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = TemporalCoherenceMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=3
        )

        # Create test features
        features = torch.randn(10, 512, device=self.device)

        coherence = metric._compute_temporal_coherence(features)

        self.assertIsInstance(coherence, float)
        self.assertGreater(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the _compute method."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = torch.randn(10, 512, device=self.device)
        mock_create_extractor.return_value = mock_extractor

        metric = TemporalCoherenceMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, window_size=5
        )

        pred = torch.rand(10, 3, 64, 64)
        target = torch.rand(10, 3, 64, 64)

        result = metric._compute(pred, target)

        self.assertIn(metric._NAME, result.values)
        self.assertIsInstance(result.values[metric._NAME], torch.Tensor)
        self.assertIn("extractor_type", result.metadata)
        self.assertIn("window_size", result.metadata)
        self.assertIn("pred_coherence", result.metadata)
        self.assertIn("target_coherence", result.metadata)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_type_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the type method."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = TemporalCoherenceMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        self.assertEqual(metric.type(), MetricType.TEMPORAL_COHERENCE)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_metadata_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the metadata method."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = TemporalCoherenceMetric(
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

        metric = TemporalCoherenceMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        # Add some test values
        from nre.metrics.metric import MetricResult

        test_result = MetricResult(
            values={metric._NAME: torch.tensor(0.95)},  # Ratio value
            metadata={},
        )
        metric._values.append(test_result)
        metric._values.append(test_result)

        aggregated = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, aggregated)
        self.assertIn(metric._NAME, aggregated[AggregationMethod.MEAN].values)


if __name__ == "__main__":
    unittest.main()
