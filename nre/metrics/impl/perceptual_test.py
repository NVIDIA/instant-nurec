# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for generic perceptual metric implementation."""

import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional as F

from nre.metrics.impl.perceptual import PerceptualMetric
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod


class TestPerceptualMetric(unittest.TestCase):
    """Test cases for PerceptualMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = "cpu"
        self.extractor_type = "segformer"
        self.pretrained_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization(self, mock_create_extractor: MagicMock) -> None:
        """Test PerceptualMetric initialization."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        self.assertEqual(metric.extractor_type, self.extractor_type)
        self.assertEqual(metric.pretrained_path, self.pretrained_path)
        mock_create_extractor.assert_called_once_with(
            extractor_type=self.extractor_type, pretrained_path=self.pretrained_path, cache_dir=None, device=self.device
        )

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_weighted_mean_not_supported(self, mock_create_extractor: MagicMock) -> None:
        """Test that weighted mean aggregation raises ValueError."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        with self.assertRaises(ValueError) as context:
            PerceptualMetric(
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

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        # Valid inputs should not raise
        pred = torch.rand(3, 64, 64)
        target = torch.rand(3, 64, 64)
        metric.validate_inputs(pred, target)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_invalid_type(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with invalid input types."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        with self.assertRaises(TypeError):
            metric.validate_inputs("not_tensor", torch.rand(3, 64, 64))  # type: ignore[arg-type]

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_shape_mismatch(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with mismatched shapes."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        pred = torch.rand(3, 64, 64)
        target = torch.rand(3, 32, 32)

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("shapes must match", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the _compute method."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = torch.randn(1, 512)
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        pred = torch.rand(3, 64, 64)
        target = torch.rand(3, 64, 64)

        result = metric._compute(pred, target)

        self.assertIn(metric._NAME, result.values)
        self.assertIsInstance(result.values[metric._NAME], torch.Tensor)
        self.assertIn("extractor_type", result.metadata)
        self.assertIn("pretrained_path", result.metadata)
        self.assertIn("feature_dim", result.metadata)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_type_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the type method."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        self.assertEqual(metric.type(), MetricType.PERCEPTUAL)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_metadata_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the metadata method."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        metadata = metric.metadata()

        self.assertEqual(metadata["extractor_type"], self.extractor_type)
        self.assertEqual(metadata["pretrained_path"], self.pretrained_path)
        self.assertEqual(metadata["feature_dim"], 512)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_aggregate_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the aggregate method."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        # Add some test values
        from nre.metrics.metric import MetricResult

        test_result = MetricResult(values={metric._NAME: torch.tensor(0.5)}, metadata={})
        metric._values.append(test_result)
        metric._values.append(test_result)

        aggregated = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, aggregated)
        self.assertIn(metric._NAME, aggregated[AggregationMethod.MEAN].values)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_perceptual_compute_mse_calculation(self, mock_create_extractor: MagicMock) -> None:
        """Test MSE loss is calculated correctly."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 3
        # Known features for predictable MSE
        mock_extractor.extract_features_batch.side_effect = [
            torch.tensor([[1.0, 2.0, 3.0]], device=self.device),  # pred: [1, 2, 3]
            torch.tensor([[2.0, 3.0, 4.0]], device=self.device),  # target: [2, 3, 4]
        ]
        # Expected MSE: ((1-2)² + (2-3)² + (3-4)²) / 3 = 3/3 = 1.0
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        pred = torch.randn(3, 64, 64)
        target = torch.randn(3, 64, 64)

        result = metric._compute(pred, target)

        # Check MSE calculation
        perceptual_value = float(result.values["perceptual"])
        self.assertAlmostEqual(perceptual_value, 1.0, places=5)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_perceptual_compute_metric_result_structure(self, mock_create_extractor: MagicMock) -> None:
        """Test MetricResult has correct structure."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = torch.randn(1, 512, device=self.device)
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        pred = torch.randn(3, 64, 64)
        target = torch.randn(3, 64, 64)

        result = metric._compute(pred, target)

        # Check MetricResult structure
        self.assertIn(metric._NAME, result.values)
        self.assertIsInstance(result.values[metric._NAME], torch.Tensor)
        self.assertIsInstance(result.metadata, dict)

        # Check required metadata keys
        required_keys = ["extractor_type", "pretrained_path", "input_shape", "feature_dim"]
        for key in required_keys:
            self.assertIn(key, result.metadata)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_perceptual_compute_zero_distance_identical(self, mock_create_extractor: MagicMock) -> None:
        """Test zero perceptual distance for identical features."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        # Return identical features
        identical_features = torch.randn(1, 512, device=self.device)
        mock_extractor.extract_features_batch.return_value = identical_features
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        pred = torch.randn(3, 64, 64)
        target = torch.randn(3, 64, 64)

        result = metric._compute(pred, target)

        # Should be zero distance
        perceptual_value = float(result.values["perceptual"])
        self.assertAlmostEqual(perceptual_value, 0.0, places=10)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_perceptual_single_image(self, mock_create_extractor: MagicMock) -> None:
        """Test Perceptual computation with single image input (1, C, H, W)."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        # Create different features for pred and target
        pred_features = torch.randn(1, 512)
        target_features = torch.randn(1, 512)
        mock_extractor.extract_features_batch.side_effect = [pred_features, target_features]
        mock_create_extractor.return_value = mock_extractor

        metric = PerceptualMetric(
            device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path
        )

        # Create single image inputs with batch dimension (1, C, H, W)
        pred_image = torch.randn(1, 3, 256, 256)
        target_image = torch.randn(1, 3, 256, 256)

        # Compute perceptual distance for single images
        metric_result = metric.compute(pred_image, target_image)

        # Validate results
        self.assertIn("perceptual", metric_result.values)
        perceptual_value = float(metric_result.values["perceptual"])

        # Perceptual distance should be a valid number
        self.assertIsInstance(perceptual_value, float)
        self.assertGreaterEqual(perceptual_value, 0.0)
        self.assertFalse(np.isnan(perceptual_value))
        self.assertFalse(np.isinf(perceptual_value))

        # Verify metadata
        self.assertIn("extractor_type", metric_result.metadata)
        self.assertIn("pretrained_path", metric_result.metadata)
        self.assertIn("input_shape", metric_result.metadata)
        self.assertEqual(metric_result.metadata["input_shape"], [1, 3, 256, 256])

        # Verify the expected MSE calculation
        expected_mse = F.mse_loss(pred_features, target_features)
        self.assertAlmostEqual(perceptual_value, float(expected_mse), places=6)


if __name__ == "__main__":
    unittest.main()
