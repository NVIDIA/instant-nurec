# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for Neural Topological Divergence (NTD) metric implementation."""

import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nre.metrics.impl.ntd import NTDMetric
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod


class TestNTDMetric(unittest.TestCase):
    """Test cases for NTDMetric class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = "cpu"
        self.extractor_type = "segformer"
        self.pretrained_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.n_neighbors = 12
        self.symmetrize = True

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_initialization(self, mock_create_extractor: MagicMock) -> None:
        """Test NTDMetric initialization."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = NTDMetric(
            device=self.device,
            extractor_type=self.extractor_type,
            pretrained_path=self.pretrained_path,
            n_neighbors=self.n_neighbors,
            symmetrize=self.symmetrize,
        )

        self.assertEqual(metric.extractor_type, self.extractor_type)
        self.assertEqual(metric.pretrained_path, self.pretrained_path)
        self.assertEqual(metric.n_neighbors, self.n_neighbors)
        self.assertEqual(metric.symmetrize, self.symmetrize)
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
        mock_create_extractor.return_value = mock_extractor

        with self.assertRaises(ValueError) as context:
            NTDMetric(aggregation_methods=AggregationMethod.WEIGHTED_MEAN)

        self.assertIn(
            "Weighted mean is not supported for NTD metric",
            str(context.exception),
        )

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_valid(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with valid tensors."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = NTDMetric(n_neighbors=5)

        # Valid 4D tensors (batch of images)
        pred = torch.randn(10, 3, 64, 64)  # 10 samples > n_neighbors + 1
        target = torch.randn(10, 3, 64, 64)

        # Should not raise any exception
        metric.validate_inputs(pred, target)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_insufficient_samples(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with insufficient samples for k-NN graph."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = NTDMetric(n_neighbors=10)

        # Too few samples (5 < n_neighbors + 1 = 11)
        pred = torch.randn(5, 3, 64, 64)
        target = torch.randn(5, 3, 64, 64)

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("Need at least 11 samples for k-NN graph", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_shape_mismatch(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with mismatched shapes."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = NTDMetric(n_neighbors=5)

        pred = torch.randn(10, 3, 64, 64)
        target = torch.randn(10, 3, 32, 32)  # Different spatial dimensions

        with self.assertRaises(ValueError) as context:
            metric.validate_inputs(pred, target)

        self.assertIn("Predicted and target shapes must match", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_validate_inputs_wrong_type(self, mock_create_extractor: MagicMock) -> None:
        """Test input validation with wrong tensor types."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = NTDMetric(n_neighbors=5)

        pred = np.random.randn(10, 3, 64, 64)  # numpy array instead of tensor
        target = torch.randn(10, 3, 64, 64)

        with self.assertRaises(TypeError) as context:
            metric.validate_inputs(pred, target)  # type: ignore[arg-type]

        self.assertIn("Input 0 must be a torch.Tensor", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_graph_spectrum_basic(self, mock_create_extractor: MagicMock) -> None:
        """Test basic graph spectrum computation."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        # Create a simple 2D dataset where we can predict the spectrum
        np.random.seed(42)
        features = np.random.randn(20, 5)

        metric = NTDMetric(n_neighbors=5, symmetrize=True)
        spectrum = metric.compute_graph_spectrum(features)

        # Basic checks
        self.assertIsInstance(spectrum, np.ndarray)
        self.assertEqual(len(spectrum), 20)  # Should have N eigenvalues for N samples
        self.assertTrue(np.all(spectrum >= 0))  # All eigenvalues should be non-negative
        self.assertTrue(np.all(spectrum[:-1] <= spectrum[1:]))  # Should be sorted

        # First eigenvalue should be 0 (or very close) for connected graph
        self.assertLess(spectrum[0], 1e-10)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_graph_spectrum_insufficient_samples(self, mock_create_extractor: MagicMock) -> None:
        """Test graph spectrum computation with insufficient samples."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        features = np.random.randn(5, 3)  # Only 5 samples
        metric = NTDMetric(n_neighbors=10)  # Need 11 samples

        with self.assertRaises(ValueError) as context:
            metric.compute_graph_spectrum(features)

        self.assertIn("Need at least 11 samples for k-NN graph", str(context.exception))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_graph_spectrum_symmetrize_options(self, mock_create_extractor: MagicMock) -> None:
        """Test graph spectrum with different symmetrize options."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        np.random.seed(42)
        features = np.random.randn(15, 4)

        metric = NTDMetric(n_neighbors=5)

        # Test with symmetrize=True
        spectrum_sym = metric.compute_graph_spectrum(features, symmetrize=True)

        # Test with symmetrize=False
        spectrum_asym = metric.compute_graph_spectrum(features, symmetrize=False)

        # Both should be valid spectra but potentially different
        self.assertTrue(np.all(spectrum_sym >= 0))
        self.assertTrue(np.all(spectrum_asym >= 0))
        self.assertEqual(len(spectrum_sym), len(spectrum_asym))

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_ntd_metrics_basic(self, mock_create_extractor: MagicMock) -> None:
        """Test basic NTD metrics computation."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        np.random.seed(42)
        gt_features = np.random.randn(20, 10)
        gen_features = np.random.randn(20, 10)

        metric = NTDMetric(n_neighbors=8)
        results = metric._compute_ntd_metrics(gt_features, gen_features)

        # Check required keys
        required_keys = [
            "ntd_distance",
            "spectrum_gt",
            "spectrum_gen",
            "spectral_area_gt",
            "spectral_area_gen",
            "spectral_energy_gt",
            "spectral_energy_gen",
            "energy_difference",
            "spectral_shape_difference",
        ]

        for key in required_keys:
            self.assertIn(key, results)

        # Check types and values (internal method returns floats, not tensors)
        self.assertIsInstance(results["ntd_distance"], float)
        self.assertIsInstance(results["spectrum_gt"], np.ndarray)
        self.assertIsInstance(results["spectrum_gen"], np.ndarray)
        self.assertGreaterEqual(results["ntd_distance"], 0)
        self.assertGreaterEqual(results["energy_difference"], 0)
        self.assertGreaterEqual(results["spectral_shape_difference"], 0)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_ntd_metrics_identical_features(self, mock_create_extractor: MagicMock) -> None:
        """Test NTD metrics with identical features (should give zero distance)."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        np.random.seed(42)
        features = np.random.randn(15, 8)

        metric = NTDMetric(n_neighbors=6)
        results = metric._compute_ntd_metrics(features, features.copy())

        # NTD distance should be very small (numerical precision)
        self.assertLess(results["ntd_distance"], 1e-10)
        self.assertLess(results["energy_difference"], 1e-10)
        self.assertLess(results["spectral_shape_difference"], 1e-10)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_full_pipeline(self, mock_create_extractor: MagicMock) -> None:
        """Test the full computation pipeline."""
        # Mock feature extractor
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        # Mock extracted features
        torch.manual_seed(42)
        mock_features_pred = torch.randn(15, 512, device=self.device)
        mock_features_target = torch.randn(15, 512, device=self.device)

        mock_extractor.extract_features_batch.side_effect = [
            mock_features_target,
            mock_features_pred,
        ]

        metric = NTDMetric(n_neighbors=8, device=self.device)

        # Create input tensors
        pred = torch.randn(15, 3, 32, 32)
        target = torch.randn(15, 3, 32, 32)

        # Compute metric
        result = metric._compute(pred, target)

        # Check result structure
        self.assertIn("ntd", result.values)
        self.assertIn("ntd_energy_diff", result.values)
        self.assertIn("ntd_shape_diff", result.values)

        # Check metadata
        self.assertIn("ntd_distance", result.metadata)
        self.assertIn("spectral_area_gt", result.metadata)
        self.assertIn("spectral_area_gen", result.metadata)
        self.assertIn("n_neighbors", result.metadata)

        # Verify feature extractor was called correctly
        self.assertEqual(mock_extractor.extract_features_batch.call_count, 2)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_compute_3d_input(self, mock_create_extractor: MagicMock) -> None:
        """Test computation with 3D input (single image)."""
        # Mock feature extractor
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        # Mock extracted features (single sample expanded to batch)
        torch.manual_seed(42)
        mock_features = torch.randn(1, 512, device=self.device)
        mock_extractor.extract_features_batch.return_value = mock_features

        metric = NTDMetric(n_neighbors=0, device=self.device)  # n_neighbors=0 for single sample

        # Create 3D input tensors (single images)
        pred = torch.randn(3, 32, 32)
        target = torch.randn(3, 32, 32)

        # This should work (input gets expanded to batch dimension)
        # But will fail validation due to insufficient samples for k-NN
        with self.assertRaises(ValueError):
            metric._compute(pred, target)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_type_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the type method returns correct MetricType."""
        mock_extractor = MagicMock()
        mock_create_extractor.return_value = mock_extractor

        metric = NTDMetric()
        self.assertEqual(metric.type(), MetricType.NTD)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_metadata_method(self, mock_create_extractor: MagicMock) -> None:
        """Test the metadata method returns correct information."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = NTDMetric(
            extractor_type="test_extractor",
            pretrained_path="test_path",
            n_neighbors=15,
            symmetrize=False,
        )

        metadata = metric.metadata()

        expected_keys = [
            "extractor_type",
            "pretrained_path",
            "feature_dim",
            "n_neighbors",
            "symmetrize",
        ]
        for key in expected_keys:
            self.assertIn(key, metadata)

        self.assertEqual(metadata["extractor_type"], "test_extractor")
        self.assertEqual(metadata["pretrained_path"], "test_path")
        self.assertEqual(metadata["feature_dim"], 512)
        self.assertEqual(metadata["n_neighbors"], 15)
        self.assertEqual(metadata["symmetrize"], False)

    @patch("nre.metrics.impl.utils.feature_extractor.FeatureExtractorFactory.create_extractor")
    def test_aggregation(self, mock_create_extractor: MagicMock) -> None:
        """Test metric aggregation functionality."""
        mock_extractor = MagicMock()
        mock_extractor.feature_dim = 512
        mock_create_extractor.return_value = mock_extractor

        metric = NTDMetric(
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.MAX,
            ]
        )

        # Create some mock results
        from nre.metrics.metric import MetricResult

        result1 = MetricResult(
            values={
                "ntd": torch.tensor(1.0),
                "ntd_energy_diff": torch.tensor(0.5),
                "ntd_shape_diff": torch.tensor(0.3),
            },
            metadata={},
        )
        result2 = MetricResult(
            values={
                "ntd": torch.tensor(2.0),
                "ntd_energy_diff": torch.tensor(1.0),
                "ntd_shape_diff": torch.tensor(0.7),
            },
            metadata={},
        )

        # Add results to metric
        metric._values = [result1, result2]

        # Test aggregation
        aggregated = metric.aggregate()

        self.assertIn(AggregationMethod.MEAN, aggregated)
        self.assertIn(AggregationMethod.MAX, aggregated)

        # Check that all metric values are aggregated
        mean_result = aggregated[AggregationMethod.MEAN]
        self.assertIn("ntd", mean_result.values)
        self.assertIn("ntd_energy_diff", mean_result.values)
        self.assertIn("ntd_shape_diff", mean_result.values)


if __name__ == "__main__":
    unittest.main()
