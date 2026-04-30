# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for generic feature extractor system."""

import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nre.metrics.impl.utils.feature_extractor import (
    BaseFeatureExtractor,
    FeatureExtractorFactory,
    SegformerFeatureExtractor,
)


class TestBaseFeatureExtractor(unittest.TestCase):
    """Test cases for BaseFeatureExtractor abstract class."""

    def test_abstract_methods(self) -> None:
        """Test that BaseFeatureExtractor cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseFeatureExtractor("test_path")  # type: ignore


class TestSegformerFeatureExtractor(unittest.TestCase):
    """Test cases for SegformerFeatureExtractor class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.pretrained_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.device = "cpu"

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    def test_initialization(self, mock_model: MagicMock, mock_processor: MagicMock) -> None:
        """Test SegformerFeatureExtractor initialization."""
        # Mock the model and processor
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        extractor = SegformerFeatureExtractor(pretrained_path=self.pretrained_path, device=self.device)

        self.assertEqual(extractor.pretrained_path, self.pretrained_path)
        self.assertEqual(extractor.device, torch.device(self.device))
        self.assertEqual(extractor.feature_dim, 512)
        # The implementation validates and uses the provided pretrained path
        mock_model.assert_called_once_with(self.pretrained_path, cache_dir=None)
        mock_processor.assert_called_once_with(self.pretrained_path, cache_dir=None)

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    def test_device_autodetection(self, mock_model: MagicMock, mock_processor: MagicMock) -> None:
        """Test device autodetection when device is None."""
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        with patch("torch.cuda.is_available", return_value=False):
            extractor = SegformerFeatureExtractor(pretrained_path=self.pretrained_path, device=None)
            self.assertEqual(extractor.device, torch.device("cpu"))

        with patch("torch.cuda.is_available", return_value=True):
            extractor = SegformerFeatureExtractor(pretrained_path=self.pretrained_path, device=None)
            self.assertEqual(extractor.device, torch.device("cuda"))

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    def test_convert_tensor_to_pil(self, mock_model: MagicMock, mock_processor: MagicMock) -> None:
        """Test tensor to PIL conversion."""
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        extractor = SegformerFeatureExtractor(pretrained_path=self.pretrained_path, device=self.device)

        # Create test tensor
        test_tensor = torch.rand(2, 3, 64, 64)
        pil_images = extractor._convert_tensor_to_pil(test_tensor)  # type: ignore

        self.assertEqual(len(pil_images), 2)
        # Check that all images are PIL Images
        for img in pil_images:
            self.assertEqual(img.size, (64, 64))

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    @patch("nre.metrics.impl.utils.feature_extractor.SegformerFeatureExtractor._extract_features_common")
    def test_extract_features_batch(
        self,
        mock_extract_common: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
    ) -> None:
        """Test batch feature extraction."""
        # Mock the model and processor
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        # Mock the feature extraction to return expected results
        mock_extract_common.return_value = torch.rand(2, 512)

        extractor = SegformerFeatureExtractor(pretrained_path=self.pretrained_path, device=self.device)

        # Test feature extraction
        test_images = torch.rand(2, 3, 64, 64)
        features = extractor.extract_features_batch(test_images)

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape, (2, 512))
        mock_extract_common.assert_called_once()

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    @patch("nre.metrics.impl.utils.feature_extractor.SegformerFeatureExtractor._extract_features_common")
    def test_extract_features_batch_numpy(
        self,
        mock_extract_common: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
    ) -> None:
        """Test batch feature extraction with numpy return."""
        # Mock the model and processor
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        # Mock the feature extraction to return expected results
        mock_extract_common.return_value = np.random.rand(3, 512)

        extractor = SegformerFeatureExtractor(pretrained_path=self.pretrained_path, device=self.device)

        # Test sequence feature extraction (using return_numpy=True)
        test_sequence = torch.rand(3, 3, 64, 64)
        features = extractor.extract_features_batch(test_sequence, return_numpy=True)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (3, 512))
        mock_extract_common.assert_called_once()

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    @patch("nre.metrics.impl.utils.feature_extractor.SegformerFeatureExtractor._extract_features_common")
    def test_extract_features_batch_with_batching(
        self,
        mock_extract_common: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
    ) -> None:
        """Test batch feature extraction with batch size limit."""
        # Mock the model and processor
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        # Mock the feature extraction to return different results for each batch
        mock_extract_common.side_effect = [
            torch.rand(2, 512),  # First batch (2 samples)
            torch.rand(2, 512),  # Second batch (2 samples)
            torch.rand(1, 512),  # Third batch (1 sample)
        ]

        extractor = SegformerFeatureExtractor(pretrained_path=self.pretrained_path, device=self.device)

        # Test with batch size smaller than total samples
        test_images = torch.rand(5, 3, 64, 64)
        features = extractor.extract_features_batch(test_images, batch_size=2, return_numpy=True)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (5, 512))
        # Should have called _extract_features_common 3 times (batches of 2, 2, 1)
        self.assertEqual(mock_extract_common.call_count, 3)

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    @patch("nre.metrics.impl.utils.feature_extractor.SegformerFeatureExtractor._extract_features_common")
    def test_extract_features_batch_size_larger_than_data(
        self,
        mock_extract_common: MagicMock,
        mock_model: MagicMock,
        mock_processor: MagicMock,
    ) -> None:
        """Test batch processing when batch_size > number of images."""
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        # Mock the feature extraction to return expected results
        mock_extract_common.return_value = np.random.rand(3, 512)

        extractor = SegformerFeatureExtractor(pretrained_path=self.pretrained_path, device=self.device)

        # Test with batch_size larger than data size
        test_images = torch.rand(3, 3, 64, 64)
        features = extractor.extract_features_batch(test_images, batch_size=10, return_numpy=True)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (3, 512))
        # Should process all at once (1 call)
        self.assertEqual(mock_extract_common.call_count, 1)


class TestFeatureExtractorFactory(unittest.TestCase):
    """Test cases for FeatureExtractorFactory class."""

    def test_get_available_extractors(self) -> None:
        """Test getting available extractor types."""
        available = FeatureExtractorFactory.get_available_extractors()
        self.assertIn("segformer", available)
        self.assertIsInstance(available, list)

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    def test_create_segformer_extractor(self, mock_model: MagicMock, mock_processor: MagicMock) -> None:
        """Test creating SegformerFeatureExtractor via factory."""
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        valid_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        extractor = FeatureExtractorFactory.create_extractor(
            extractor_type="segformer", pretrained_path=valid_path, device="cpu"
        )

        self.assertIsInstance(extractor, SegformerFeatureExtractor)
        self.assertEqual(extractor.pretrained_path, valid_path)
        self.assertEqual(extractor.device, torch.device("cpu"))

    def test_create_unsupported_extractor(self) -> None:
        """Test creating unsupported extractor type."""
        valid_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        with self.assertRaises(ValueError) as context:
            FeatureExtractorFactory.create_extractor(extractor_type="unsupported", pretrained_path=valid_path)

        self.assertIn("Unsupported extractor type", str(context.exception))
        self.assertIn("unsupported", str(context.exception))

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    def test_invalid_pretrained_path(self, mock_model: MagicMock, mock_processor: MagicMock) -> None:
        """Test that invalid pretrained paths raise ValueError."""
        # Mock the model and processor
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        invalid_path = "invalid/model-path"

        with self.assertRaises(ValueError) as context:
            SegformerFeatureExtractor(pretrained_path=invalid_path, device="cpu")

        self.assertIn("Invalid pretrained_path", str(context.exception))
        self.assertIn(invalid_path, str(context.exception))
        self.assertIn("Must be one of", str(context.exception))

        # Verify that from_pretrained was NOT called due to early validation failure
        mock_model.assert_not_called()
        mock_processor.assert_not_called()

    @patch("transformers.SegformerImageProcessor.from_pretrained")
    @patch("transformers.SegformerForSemanticSegmentation.from_pretrained")
    def test_valid_pretrained_paths(self, mock_model: MagicMock, mock_processor: MagicMock) -> None:
        """Test that all valid pretrained paths are accepted."""
        # Mock the model and processor
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor.return_value = mock_processor_instance

        for valid_path in SegformerFeatureExtractor.VALID_MODELS:
            with self.subTest(path=valid_path):
                # This should not raise a ValueError for invalid path
                extractor = SegformerFeatureExtractor(pretrained_path=valid_path, device="cpu")
                self.assertEqual(extractor.pretrained_path, valid_path)

        # Verify that from_pretrained was called for each valid path
        self.assertEqual(mock_model.call_count, len(SegformerFeatureExtractor.VALID_MODELS))
        self.assertEqual(mock_processor.call_count, len(SegformerFeatureExtractor.VALID_MODELS))


if __name__ == "__main__":
    unittest.main()
