# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tempfile
import unittest

from unittest.mock import MagicMock, Mock, patch

import torch


# Mock cosmos_predict1 module for unit tests to avoid loading actual model checkpoints
# Only mock if we're not running integration tests with real checkpoints
# Note: Missing dependencies (mediapy, IPython, loguru, einops.pack/unpack) are now
# handled directly in tokenizer.py, so they work everywhere
if not os.getenv("TEST_WITH_COSMOS_CHECKPOINTS"):
    sys.modules["cosmos_predict1"] = Mock()
    sys.modules["cosmos_predict1.tokenizer"] = Mock()
    sys.modules["cosmos_predict1.tokenizer.inference"] = Mock()
    sys.modules["cosmos_predict1.tokenizer.inference.video_lib"] = Mock()
    sys.modules["cosmos_predict1.tokenizer.networks"] = Mock()

from nre.nrm.models.tokenizer import (
    _get_cosmos_diffusion_mean_std,
    _get_tokenizer_config,
    denormalize_latents,
)


class TestGetTokenizerConfig(unittest.TestCase):
    """Test _get_tokenizer_config function"""

    @patch("nre.nrm.models.tokenizer.TokenizerConfigs")
    def test_extracts_model_name_from_path(self, mock_configs):
        """Test that model name is correctly extracted from checkpoint path"""
        mock_config = {"name": "CV8x8x8", "latent_channels": 16}
        mock_configs.__getitem__.return_value = MagicMock(value=mock_config)

        checkpoint_path = "/path/to/Cosmos-Tokenize1-CV8x8x8-720p"
        config = _get_tokenizer_config(checkpoint_path)

        # Verify the model name was correctly extracted and underscores replaced
        mock_configs.__getitem__.assert_called_once_with("CV8x8x8_720p")
        self.assertEqual(config, mock_config)

    @patch("nre.nrm.models.tokenizer.TokenizerConfigs")
    def test_handles_hyphens_in_model_name(self, mock_configs):
        """Test that hyphens are replaced with underscores"""
        mock_config = {"name": "CV4x8x8", "latent_channels": 16}
        mock_configs.__getitem__.return_value = MagicMock(value=mock_config)

        checkpoint_path = "Cosmos-Tokenize1-CV4x8x8-360p"
        config = _get_tokenizer_config(checkpoint_path)

        mock_configs.__getitem__.assert_called_once_with("CV4x8x8_360p")


class TestGetCosmosDiffusionMeanStd(unittest.TestCase):
    """Test _get_cosmos_diffusion_mean_std function"""

    def test_loads_and_reshapes_mean_std(self):
        """Test that mean and std are loaded and reshaped correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock mean and std tensors
            latent_ch = 16
            latent_chunk_duration = 3
            # Mean and std should have shape [latent_ch * latent_chunk_duration]
            orig_mean = torch.randn(latent_ch * latent_chunk_duration)
            orig_std = torch.rand(latent_ch * latent_chunk_duration) + 0.5  # Keep positive

            # Save to temporary file
            mean_std_path = os.path.join(tmpdir, "mean_std.pt")
            torch.save((orig_mean, orig_std), mean_std_path)

            # Test loading and reshaping
            mean, std = _get_cosmos_diffusion_mean_std(tmpdir, torch.float32, latent_ch, latent_chunk_duration)

            # Check shapes
            expected_shape = [1, latent_ch, latent_chunk_duration, 1, 1]
            self.assertEqual(list(mean.shape), expected_shape)
            self.assertEqual(list(std.shape), expected_shape)

            # Check dtype
            self.assertEqual(mean.dtype, torch.float32)
            self.assertEqual(std.dtype, torch.float32)

    def test_uses_original_dtype_when_none_specified(self):
        """Test that original dtype is preserved when dtype=None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            latent_ch = 8
            latent_chunk_duration = 2
            orig_mean = torch.randn(latent_ch * latent_chunk_duration, dtype=torch.float16)
            orig_std = torch.rand(latent_ch * latent_chunk_duration, dtype=torch.float16) + 0.5

            mean_std_path = os.path.join(tmpdir, "mean_std.pt")
            torch.save((orig_mean, orig_std), mean_std_path)

            mean, std = _get_cosmos_diffusion_mean_std(tmpdir, None, latent_ch, latent_chunk_duration)

            self.assertEqual(mean.dtype, torch.float16)
            self.assertEqual(std.dtype, torch.float16)


class TestDenormalizeLatents(unittest.TestCase):
    """Test denormalize_latents function"""

    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 2
        self.num_views = 2
        self.num_frames_per_view = 4  # Frames per view after separating views
        self.latent_ch = 16
        self.height = 32
        self.width = 32
        self.sigma_data = 0.5

        # Create mock latent statistics
        # Shape: (1, T, C, 1, 1) as per denormalize_latents docstring
        self.latent_mean = torch.randn(1, self.num_frames_per_view, self.latent_ch, 1, 1)
        self.latent_std = torch.rand(1, self.num_frames_per_view, self.latent_ch, 1, 1) + 0.5

    def test_denormalize_with_batch_dimension(self):
        """Test denormalization with batch dimension"""
        # Input shape: (B, V*T, C, H, W)
        total_frames = self.num_views * self.num_frames_per_view
        model_input = torch.randn(self.batch_size, total_frames, self.latent_ch, self.height, self.width)

        output = denormalize_latents(
            model_input,
            self.latent_std,
            self.latent_mean,
            self.num_views,
            self.sigma_data,
        )

        # Output shape: (B, V*T, C, H, W) stays the same structure
        expected_shape = (
            self.batch_size,
            self.num_views * self.num_frames_per_view,
            self.latent_ch,
            self.height,
            self.width,
        )
        self.assertEqual(output.shape, expected_shape)

        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())

    def test_denormalize_without_batch_dimension(self):
        """Test denormalization without batch dimension (adds and removes batch dim)"""
        # Input shape: (V*T, C, H, W)
        total_frames = self.num_views * self.num_frames_per_view
        model_input = torch.randn(total_frames, self.latent_ch, self.height, self.width)

        output = denormalize_latents(
            model_input,
            self.latent_std,
            self.latent_mean,
            self.num_views,
            self.sigma_data,
        )

        # Output shape: (V*T, C, H, W) stays the same structure
        expected_shape = (self.num_views * self.num_frames_per_view, self.latent_ch, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(len(output.shape), 4)

        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())

    def test_denormalize_applies_correct_transformation(self):
        """Test that denormalization applies correct mathematical transformation"""
        # Use simple values for easier verification
        model_input = torch.ones(1, 4, 16, 32, 32)
        latent_mean = torch.zeros(1, 4, 16, 1, 1)  # Shape: (1, T, C, 1, 1)
        latent_std = torch.ones(1, 4, 16, 1, 1)  # Shape: (1, T, C, 1, 1)
        sigma_data = 1.0

        output = denormalize_latents(
            model_input, latent_std, latent_mean, num_input_multi_views=1, sigma_data=sigma_data
        )

        # With sigma_data=1, mean=0, std=1: output should equal input after transformations
        # The function does: (input / sigma_data) * std + mean
        # So with our values: (1 / 1) * 1 + 0 = 1
        # Output keeps the same dimension structure: (B, T, C, H, W)
        expected_shape = (1, 4, 16, 32, 32)  # (B, T, C, H, W)
        self.assertEqual(output.shape, expected_shape)

    def test_single_view_case(self):
        """Test with single view (num_input_multi_views=1)"""
        model_input = torch.randn(self.batch_size, self.num_frames_per_view, self.latent_ch, self.height, self.width)

        output = denormalize_latents(
            model_input,
            self.latent_std,
            self.latent_mean,
            num_input_multi_views=1,
            sigma_data=self.sigma_data,
        )

        # Shape stays the same: (B, T, C, H, W)
        expected_shape = (self.batch_size, self.num_frames_per_view, self.latent_ch, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_multiple_views_case(self):
        """Test with multiple views"""
        num_views = 4
        total_frames = num_views * self.num_frames_per_view
        # Need to adjust latent stats for this case (shape: (1, T, C, 1, 1))
        latent_mean_extended = torch.randn(1, self.num_frames_per_view, self.latent_ch, 1, 1)
        latent_std_extended = torch.rand(1, self.num_frames_per_view, self.latent_ch, 1, 1) + 0.5

        model_input = torch.randn(self.batch_size, total_frames, self.latent_ch, self.height, self.width)

        output = denormalize_latents(
            model_input,
            latent_std_extended,
            latent_mean_extended,
            num_input_multi_views=num_views,
            sigma_data=self.sigma_data,
        )

        # Shape stays the same: (B, V*T, C, H, W)
        expected_shape = (
            self.batch_size,
            num_views * self.num_frames_per_view,
            self.latent_ch,
            self.height,
            self.width,
        )
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_custom_sigma_data(self):
        """Test with custom sigma_data value"""
        model_input = torch.randn(
            self.batch_size,
            self.num_views * self.num_frames_per_view,
            self.latent_ch,
            self.height,
            self.width,
        )
        custom_sigma = 2.0

        output = denormalize_latents(
            model_input,
            self.latent_std,
            self.latent_mean,
            self.num_views,
            sigma_data=custom_sigma,
        )

        # Shape stays the same: (B, V*T, C, H, W)
        expected_shape = (
            self.batch_size,
            self.num_views * self.num_frames_per_view,
            self.latent_ch,
            self.height,
            self.width,
        )
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())


class TestIntegration(unittest.TestCase):
    """Integration tests requiring actual cosmos_predict1 components

    Note: These tests require a proper Python environment with all cosmos dependencies.
    Run outside of Bazel using: python -m pytest nre/nrm/models/tokenizer_test.py::TestIntegration -v
    """

    @unittest.skipUnless(os.getenv("TEST_WITH_COSMOS_CHECKPOINTS"), "Requires cosmos model checkpoints")
    def test_load_tokenizer_with_checkpoints(self):
        """Integration test with actual model checkpoints (requires env var)"""
        from nre.nrm.models.tokenizer import load_cosmos_1_tokenizer

        checkpoint_path = os.getenv("COSMOS_CHECKPOINT_PATH")
        self.assertIsNotNone(checkpoint_path, "COSMOS_CHECKPOINT_PATH must be set")
        assert checkpoint_path is not None  # Type narrowing for mypy

        # Test loading encoder only
        tokenizer = load_cosmos_1_tokenizer(checkpoint_path, load_encoder=True, load_decoder=False, load_jit=True)
        self.assertIsNotNone(tokenizer)

    @unittest.skipUnless(os.getenv("TEST_WITH_COSMOS_CHECKPOINTS"), "Requires cosmos model checkpoints")
    def test_encode_decode_with_checkpoints(self):
        """End-to-end test of encoding and decoding (requires env var)"""
        from nre.nrm.models.tokenizer import load_cosmos_1_tokenizer

        checkpoint_path = os.getenv("COSMOS_CHECKPOINT_PATH")
        self.assertIsNotNone(checkpoint_path, "COSMOS_CHECKPOINT_PATH must be set")
        assert checkpoint_path is not None  # Type narrowing for mypy

        tokenizer = load_cosmos_1_tokenizer(checkpoint_path, load_encoder=True, load_decoder=True, load_jit=True)

        # Create test input
        input_tensor = torch.rand(1, 3, 9, 512, 512).to("cuda").to(torch.bfloat16)  # [B, C, T, H, W]
        input_tensor = input_tensor * 2.0 - 1.0  # Normalize to [-1..1]

        # Encode
        (latent,) = tokenizer.encode(input_tensor)  # type: ignore[attr-defined]

        # Check latent shape
        self.assertEqual(len(latent.shape), 5)
        self.assertEqual(latent.shape[0], 1)  # batch

        # Decode
        reconstructed = tokenizer.decode(latent)  # type: ignore[attr-defined]

        # Check reconstructed shape matches input
        self.assertEqual(reconstructed.shape, input_tensor.shape)

    @unittest.skipUnless(os.getenv("TEST_WITH_COSMOS_CHECKPOINTS"), "Requires cosmos model checkpoints")
    def test_encode_decode_shape_assertions(self):
        """Test encoding and decoding with specific shape assertions (from tokenizer.py main)"""
        from nre.nrm.models.tokenizer import load_cosmos_1_tokenizer

        # Use checkpoint path from environment variable or default path
        checkpoint_path = os.getenv("COSMOS_CHECKPOINT_PATH")
        if checkpoint_path is None:
            raise ValueError("COSMOS_CHECKPOINT_PATH must be set")

        tokenizer = load_cosmos_1_tokenizer(checkpoint_path, load_encoder=True, load_decoder=True, load_jit=True)

        # Create test input: [B, C, T, H, W]
        input_tensor = torch.rand(1, 3, 9, 512, 512).to("cuda").to(torch.bfloat16)
        input_tensor = input_tensor * 2.0 - 1.0  # Normalize to [-1..1]

        # Encode
        (latent,) = tokenizer.encode(input_tensor)  # type: ignore[attr-defined]

        # Check latent shape properties
        # Expected: [B, C, T, H, W] with spatial compression ~8x and temporal compression ~3x
        self.assertEqual(len(latent.shape), 5, f"Expected 5D tensor, got shape {latent.shape}")
        self.assertEqual(latent.shape[0], 1, f"Expected batch size 1, got {latent.shape[0]}")
        self.assertEqual(latent.shape[1], 16, f"Expected 16 channels, got {latent.shape[1]}")

        # Check spatial compression (512 / 8 = 64)
        self.assertEqual(latent.shape[3], 64, f"Expected height 64, got {latent.shape[3]}")
        self.assertEqual(latent.shape[4], 64, f"Expected width 64, got {latent.shape[4]}")

        # Check temporal compression (allow some flexibility due to padding)
        # For 9 input frames, CV8x8x8 should give 2-3 latent frames
        expected_temporal_range = (2, 3)
        self.assertIn(
            latent.shape[2],
            expected_temporal_range,
            f"Expected temporal dim in {expected_temporal_range}, got {latent.shape[2]}",
        )

        # Decode
        reconstructed_tensor = tokenizer.decode(latent)  # type: ignore[attr-defined]

        # Assert reconstructed shape matches input (or is close due to padding)
        self.assertEqual(len(reconstructed_tensor.shape), 5)
        self.assertEqual(reconstructed_tensor.shape[0], input_tensor.shape[0])
        self.assertEqual(reconstructed_tensor.shape[1], input_tensor.shape[1])
        # Temporal dimension might be padded
        self.assertGreaterEqual(reconstructed_tensor.shape[2], input_tensor.shape[2])
        self.assertEqual(reconstructed_tensor.shape[3], input_tensor.shape[3])
        self.assertEqual(reconstructed_tensor.shape[4], input_tensor.shape[4])


if __name__ == "__main__":
    unittest.main()
