# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for generic FID metric implementation."""

import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nre.metrics.impl.fid import FIDMetric
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod


class TestFIDMetric(unittest.TestCase):
    """Test cases for FIDMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = "cpu"
        self.extractor_type = "segformer"
        self.pretrained_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization(self, mock_create_extractor: MagicMock) -> None:
        """Test FIDMetric initialization."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

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
            FIDMetric(
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

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        # Valid inputs should not raise
        pred = torch.rand(3, 64, 64)
        target = torch.rand(3, 64, 64)
        metric.validate_inputs(pred, target)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_invalid_type(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with invalid input types."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        with self.assertRaises(TypeError):
            metric.validate_inputs("not_tensor", torch.rand(3, 64, 64))  # type: ignore[arg-type]

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_shape_mismatch(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with mismatched shapes."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        pred = torch.rand(3, 64, 64)
        target = torch.rand(3, 32, 32)

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("shapes must match", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_fid(self, mock_create_extractor: MagicMock) -> None:
        """Test FID computation."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        # Create test features
        features_pred = torch.randn(10, 512, device=self.device)
        features_target = torch.randn(10, 512, device=self.device)

        # Convert to numpy arrays for _compute_fid
        features_pred_np = features_pred.cpu().numpy()
        features_target_np = features_target.cpu().numpy()

        fid_score = metric._compute_fid(features_pred_np, features_target_np)

        self.assertIsInstance(fid_score, float)
        self.assertGreaterEqual(fid_score, 0.0)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the _compute method."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_extractor.extract_features_batch.return_value = torch.randn(1, 512, device=self.device)
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

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

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        self.assertEqual(metric.type(), MetricType.FID)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_metadata_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the metadata method."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        metadata = metric.metadata()

        self.assertEqual(metadata["extractor_type"], self.extractor_type)
        self.assertEqual(metadata["pretrained_path"], self.pretrained_path)
        self.assertEqual(metadata["feature_dim"], 512)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_aggregate_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the aggregate method."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        # Add some test values
        from nre.metrics.metric import MetricResult

        test_result = MetricResult(values={metric._NAME: torch.tensor(1.5)}, metadata={})
        metric._values.append(test_result)
        metric._values.append(test_result)

        aggregated = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, aggregated)
        self.assertIn(metric._NAME, aggregated[AggregationMethod.MEAN].values)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_fid_video_consistency(self, mock_create_extractor: MagicMock) -> None:
        """Test FID with identical video sequences (should be very small)."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        # Create identical features for both pred and target to get low FID
        identical_features = torch.randn(10, 512, device=self.device)
        mock_extractor.extract_features_batch.return_value = identical_features
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        # Create identical video sequences
        video_frames = torch.randn(10, 3, 256, 256)
        identical_frames = video_frames.clone()

        # Compute FID between identical sequences
        metric_result = metric.compute(video_frames, identical_frames)

        # FID should be very small (close to 0) for identical sequences
        fid_value = float(metric_result.values["fid"])
        self.assertIsInstance(fid_value, float)
        self.assertGreaterEqual(fid_value, 0.0)  # FID should never be negative
        self.assertLess(fid_value, 2e-6)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_fid_non_negative(self, mock_create_extractor: MagicMock) -> None:
        """Test that FID is never negative."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        # Test with multiple different feature combinations
        test_cases = [
            # Case 1: Different random features
            (torch.randn(5, 512, device=self.device), torch.randn(5, 512, device=self.device)),
            # Case 2: One zero, one random
            (torch.zeros(5, 512, device=self.device), torch.randn(5, 512, device=self.device)),
            # Case 3: Both with different scales
            (torch.randn(5, 512, device=self.device) * 10, torch.randn(5, 512, device=self.device) * 0.1),
            # Case 4: Large batch
            (torch.randn(50, 512, device=self.device), torch.randn(50, 512, device=self.device)),
        ]

        for i, (pred_features, target_features) in enumerate(test_cases):
            with self.subTest(case=i):
                # Mock different features for pred and target
                mock_extractor.extract_features_batch.side_effect = [pred_features, target_features]

                # Create test data
                pred_images = torch.randn(len(pred_features), 3, 256, 256)
                target_images = torch.randn(len(target_features), 3, 256, 256)

                # Compute FID
                metric_result = metric.compute(pred_images, target_images)
                fid_value = float(metric_result.values["fid"])

                # FID must never be negative
                self.assertIsInstance(fid_value, float)
                self.assertGreaterEqual(fid_value, 0.0, f"FID should never be negative, got {fid_value} for case {i}")
                self.assertFalse(np.isnan(fid_value), f"FID should not be NaN, got {fid_value} for case {i}")
                self.assertFalse(np.isinf(fid_value), f"FID should not be infinite, got {fid_value} for case {i}")

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_fid_numerical_instability(self, mock_create_extractor: MagicMock) -> None:
        """Test FID computation under numerical instability conditions."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        # Test cases that could cause numerical instability
        instability_cases = [
            # Case 1: Very small values (near machine epsilon)
            {
                "name": "very_small_values",
                "pred": torch.randn(10, 512, device=self.device) * 1e-10,
                "target": torch.randn(10, 512, device=self.device) * 1e-10,
            },
            # Case 2: Very large values (but still reasonable)
            {
                "name": "very_large_values",
                "pred": torch.randn(10, 512, device=self.device) * 1e3,
                "target": torch.randn(10, 512, device=self.device) * 1e3,
            },
            # Case 3: Mixed scales (one tiny, one large)
            {
                "name": "mixed_extreme_scales",
                "pred": torch.randn(10, 512, device=self.device) * 1e-6,
                "target": torch.randn(10, 512, device=self.device) * 1e3,
            },
            # Case 4: Near-singular covariance (highly correlated features)
            {
                "name": "near_singular_covariance",
                "pred": self._create_near_singular_features(10, 512),
                "target": self._create_near_singular_features(10, 512),
            },
            # Case 5: Constant features (zero variance)
            {
                "name": "constant_features",
                "pred": torch.ones(10, 512, device=self.device) * 5.0,
                "target": torch.ones(10, 512, device=self.device) * 3.0,
            },
            # Case 6: Features with extreme outliers
            {
                "name": "extreme_outliers",
                "pred": self._create_features_with_outliers(10, 512),
                "target": self._create_features_with_outliers(10, 512),
            },
            # Case 7: Minimum sample size (could cause covariance issues)
            {
                "name": "minimum_samples",
                "pred": torch.randn(2, 512, device=self.device),
                "target": torch.randn(2, 512, device=self.device),
            },
        ]

        for case in instability_cases:
            with self.subTest(case=case["name"]):
                # Mock different features for pred and target
                mock_extractor.extract_features_batch.side_effect = [case["pred"], case["target"]]

                # Create test data
                pred_features = case["pred"]
                target_features = case["target"]
                pred_images = torch.randn(len(pred_features), 3, 256, 256)  # type: ignore[arg-type]
                target_images = torch.randn(len(target_features), 3, 256, 256)  # type: ignore[arg-type]

                # Compute FID - should handle numerical instability gracefully
                try:
                    metric_result = metric.compute(pred_images, target_images)
                    fid_value = float(metric_result.values["fid"])

                    # Validate numerical stability
                    self.assertIsInstance(fid_value, float)
                    self.assertFalse(np.isnan(fid_value), f"FID should not be NaN for {case['name']}, got {fid_value}")
                    self.assertFalse(
                        np.isinf(fid_value), f"FID should not be infinite for {case['name']}, got {fid_value}"
                    )
                    self.assertGreaterEqual(
                        fid_value, 0.0, f"FID should be non-negative for {case['name']}, got {fid_value}"
                    )
                    # FID should be reasonable (not extremely large)
                    self.assertLess(fid_value, 1e6, f"FID should be reasonable for {case['name']}, got {fid_value}")

                except Exception as e:
                    self.fail(f"FID computation failed for {case['name']} with error: {e}")

    def _create_near_singular_features(self, n_samples: int, n_features: int) -> torch.Tensor:
        """Create features with near-singular covariance matrix."""
        # Create base features
        base = torch.randn(n_samples, 1, device=self.device)
        # Make most features highly correlated with small noise
        features = base.repeat(1, n_features)
        # Add tiny random noise to avoid exact singularity
        noise = torch.randn(n_samples, n_features, device=self.device) * 1e-6
        return features + noise

    def _create_features_with_outliers(self, n_samples: int, n_features: int) -> torch.Tensor:
        """Create features with extreme outliers."""
        features = torch.randn(n_samples, n_features, device=self.device)
        # Add extreme outliers to some samples
        outlier_indices = torch.randperm(n_samples, device=self.device)[: max(1, n_samples // 5)]
        features[outlier_indices] *= 1000  # Make extreme outliers
        return features

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_fid_single_image(self, mock_create_extractor: MagicMock) -> None:
        """Test FID computation with single image input (1, C, H, W)."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        # Create different features for pred and target to get meaningful FID
        pred_features = torch.randn(1, 512, device=self.device)
        target_features = torch.randn(1, 512, device=self.device) + 2.0
        mock_extractor.extract_features_batch.side_effect = [
            pred_features,  # pred features
            target_features,  # target features (shifted)
        ]
        mock_create_extractor.return_value = mock_extractor

        metric = FIDMetric(device=self.device, extractor_type=self.extractor_type, pretrained_path=self.pretrained_path)

        # Create single image inputs with batch dimension (1, C, H, W)
        pred_image = torch.randn(1, 3, 256, 256)
        target_image = torch.randn(1, 3, 256, 256)

        # Compute FID for single images
        metric_result = metric.compute(pred_image, target_image)

        # Validate results
        self.assertIn("fid", metric_result.values)
        fid_value = float(metric_result.values["fid"])

        # FID should be a valid number
        self.assertIsInstance(fid_value, float)
        self.assertGreaterEqual(fid_value, 0.0)
        self.assertFalse(np.isnan(fid_value))
        self.assertFalse(np.isinf(fid_value))

        # Verify metadata
        self.assertIn("extractor_type", metric_result.metadata)
        self.assertIn("pretrained_path", metric_result.metadata)
        self.assertIn("input_shape", metric_result.metadata)
        self.assertEqual(metric_result.metadata["input_shape"], [1, 3, 256, 256])


if __name__ == "__main__":
    unittest.main()
