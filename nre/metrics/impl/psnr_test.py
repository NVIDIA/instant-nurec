# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

import torch

from nre.metrics.impl.psnr import PSNRMetric
from nre.metrics.utils import AggregationMethod


class TestPSNRMetric(unittest.TestCase):
    def setUp(self):
        self.data_range = 1.0
        self.psnr = PSNRMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Create test frames as tensors
        self.frame_height = 100
        self.frame_width = 120
        self.channels = 3

        # Perfect match (zero noise) - normalized to 0-1 range
        self.gt_frame = torch.ones((self.channels, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        self.eval_frame_perfect = self.gt_frame.clone()

        # Low noise frame - normalized to 0-1 range
        self.eval_frame_low_noise = self.gt_frame.clone().float()
        torch.manual_seed(42)
        noise = torch.randint(-5, 6, self.eval_frame_low_noise.shape, dtype=torch.float32) / 255.0
        self.eval_frame_low_noise += noise
        self.eval_frame_low_noise = torch.clamp(self.eval_frame_low_noise, 0, 1)

        # High noise frame - normalized to 0-1 range
        self.eval_frame_high_noise = self.gt_frame.clone().float()
        torch.manual_seed(42)  # Reset seed for reproducibility
        noise = torch.randint(-50, 51, self.eval_frame_high_noise.shape, dtype=torch.float32) / 255.0
        self.eval_frame_high_noise += noise
        self.eval_frame_high_noise = torch.clamp(self.eval_frame_high_noise, 0, 1)

        # Worst match (maximum difference) - normalized to 0-1 range
        self.gt_frame_worst = (
            torch.ones((self.channels, self.frame_height, self.frame_width), dtype=torch.float32) * 1.0
        )
        self.eval_frame_worst = torch.zeros((self.channels, self.frame_height, self.frame_width), dtype=torch.float32)

        # Create test mask
        self.valid_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        self.valid_mask[20:40, 30:60] = False  # Create a region to mask out

    def test_psnr_perfect_match(self):
        """Test PSNR computation with identical frames"""
        metric_result = self.psnr.compute(pred=self.eval_frame_perfect, target=self.gt_frame)

        # For identical frames, PSNR should be very high
        self.assertAlmostEqual(float(metric_result["psnr"]), self.psnr.max_psnr, places=1)

        # Test metadata
        self.assertIn("data_range", metric_result.metadata)
        self.assertIn("input_shape", metric_result.metadata)
        self.assertIn("masked_pixels", metric_result.metadata)
        self.assertEqual(metric_result.metadata["data_range"], self.data_range)
        self.assertEqual(metric_result.metadata["input_shape"], [self.channels, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_psnr_with_noise(self):
        """Test PSNR computation with noisy frames"""
        metric_result_low = self.psnr.compute(self.eval_frame_low_noise, self.gt_frame)
        metric_result_high = self.psnr.compute(self.eval_frame_high_noise, self.gt_frame)

        # Test PSNR values
        self.assertGreater(float(metric_result_low["psnr"]), float(metric_result_high["psnr"]))
        self.assertLess(float(metric_result_low["psnr"]), self.psnr.max_psnr)
        self.assertLess(float(metric_result_high["psnr"]), self.psnr.max_psnr)

        # Test metadata consistency
        self.assertEqual(metric_result_low.metadata["data_range"], metric_result_high.metadata["data_range"])
        self.assertEqual(metric_result_low.metadata["input_shape"], metric_result_high.metadata["input_shape"])

    def test_psnr_with_mask(self):
        """Test PSNR computation with boolean mask"""
        metric_result = self.psnr.compute(self.eval_frame_perfect, self.gt_frame, mask=self.valid_mask)

        # PSNR should be very high because frames are identical
        self.assertAlmostEqual(float(metric_result["psnr"]), self.psnr.max_psnr, places=1)

        # Test metadata with mask which counts valid pixels (where mask is True)
        excluded_area = torch.sum(torch.logical_not(self.valid_mask)).item()
        expected_masked_pixels = self.frame_height * self.frame_width - excluded_area
        self.assertEqual(metric_result.metadata["masked_pixels"], expected_masked_pixels)

    def test_psnr_worst_match(self):
        """Test PSNR computation with worst case (maximum difference) frames"""
        metric_result = self.psnr.compute(self.eval_frame_worst, self.gt_frame_worst)

        # For data_range=1 with max difference (1 vs 0), PSNR should be very low
        # The MSE would be 1, so PSNR = 10*log10(1^2/1) = 0 dB
        self.assertAlmostEqual(float(metric_result["psnr"]), 0.0, places=1)

        # Test metadata
        self.assertEqual(metric_result.metadata["data_range"], self.data_range)
        self.assertEqual(metric_result.metadata["input_shape"], [self.channels, self.frame_height, self.frame_width])

    def test_psnr_metadata_structure(self):
        """Test that metadata has the correct structure and types"""
        metric_result = self.psnr.compute(self.eval_frame_perfect, self.gt_frame)

        # Test metadata keys exist
        required_keys = ["data_range", "input_shape", "masked_pixels"]
        for key in required_keys:
            self.assertIn(key, metric_result.metadata)

        # Test metadata types
        self.assertIsInstance(metric_result.metadata["data_range"], (int, float))
        self.assertIsInstance(metric_result.metadata["input_shape"], list)
        self.assertIsInstance(metric_result.metadata["masked_pixels"], int)

        # Test metadata values
        self.assertEqual(metric_result.metadata["data_range"], 1.0)
        self.assertEqual(len(metric_result.metadata["input_shape"]), 3)
        self.assertEqual(metric_result.metadata["input_shape"][0], self.channels)
        self.assertEqual(metric_result.metadata["input_shape"][1], self.frame_height)
        self.assertEqual(metric_result.metadata["input_shape"][2], self.frame_width)

    def test_psnr_metric_result_interface(self):
        """Test MetricResult interface methods"""
        metric_result = self.psnr.compute(self.eval_frame_perfect, self.gt_frame)

        # Test get_value method
        psnr_value = metric_result.get_value("psnr")
        self.assertIsInstance(psnr_value, torch.Tensor)

        # Test get_available_values method
        available_values = metric_result.get_available_values()
        self.assertIn("psnr", available_values)
        self.assertEqual(len(available_values), 1)

        # Test to_dict method
        result_dict = metric_result.to_dict()
        self.assertIn("psnr", result_dict)
        self.assertIsInstance(result_dict["psnr"], torch.Tensor)

        # Test dictionary-like access
        self.assertTrue("psnr" in metric_result)
        self.assertEqual(metric_result["psnr"], psnr_value)

    def test_psnr_reset_functionality(self):
        """Test that reset functionality works correctly"""
        # Compute PSNR first
        metric_result1 = self.psnr.compute(self.eval_frame_perfect, self.gt_frame)

        # Reset the metric
        self.psnr.reset()

        # Compute PSNR again - should work the same
        metric_result2 = self.psnr.compute(self.eval_frame_perfect, self.gt_frame)

        # Results should be identical
        self.assertAlmostEqual(float(metric_result1["psnr"]), float(metric_result2["psnr"]), places=6)

    def test_psnr_device_handling(self):
        """Test PSNR metric device handling"""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            # Move metric to device
            self.psnr.to(device)

            # Move test data to device
            gt_frame_device = self.gt_frame.to(device)
            eval_frame_device = self.eval_frame_perfect.to(device)

            # Compute PSNR on device
            metric_result = self.psnr.compute(eval_frame_device, gt_frame_device)

            # Should work without errors
            self.assertIsInstance(metric_result["psnr"], torch.Tensor)
            # Check device type matches (ignore index)
            self.assertEqual(metric_result["psnr"].device.type, device.type)

    def test_psnr_input_validation(self):
        """Test PSNR input validation"""
        # Test with invalid input types
        with self.assertRaises(TypeError):
            self.psnr.compute("invalid", self.gt_frame)

        with self.assertRaises(TypeError):
            self.psnr.compute(self.eval_frame_perfect, "invalid")

        # Test with insufficient dimensions
        invalid_tensor = torch.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.psnr.compute(invalid_tensor, self.gt_frame)

    def test_psnr_rgb_channels_first_format(self):
        """Test PSNR computation with RGB channels first format [3, h, w]"""
        # Create test data in [3, h, w] format - normalized to 0-1 range
        gt_rgb = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_rgb = gt_rgb.clone()

        # Add some noise to prediction
        torch.manual_seed(42)
        noise = torch.randint(-10, 11, pred_rgb.shape, dtype=torch.float32) / 255.0
        pred_rgb += noise
        pred_rgb = torch.clamp(pred_rgb, 0, 1)

        metric_result = self.psnr.compute(pred=pred_rgb, target=gt_rgb)

        # Should compute PSNR successfully
        self.assertIsInstance(metric_result["psnr"], torch.Tensor)
        self.assertGreater(float(metric_result["psnr"]), 0.0)
        self.assertLess(float(metric_result["psnr"]), self.psnr.max_psnr)

        # Test metadata reflects correct shape
        self.assertEqual(metric_result.metadata["input_shape"], [3, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_psnr_batch_rgb_format(self):
        """Test PSNR computation with batch RGB format [10, 3, h, w]"""
        batch_size = 10

        # Create test data in [10, 3, h, w] format - normalized to 0-1 range
        gt_batch = torch.ones((batch_size, 3, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_batch = gt_batch.clone()

        # Add different noise to each batch item
        torch.manual_seed(42)
        for i in range(batch_size):
            noise = torch.randint(-15, 16, (3, self.frame_height, self.frame_width), dtype=torch.float32) / 255.0
            pred_batch[i] += noise
        pred_batch = torch.clamp(pred_batch, 0, 1)

        metric_result = self.psnr.compute(pred=pred_batch, target=gt_batch)

        # Should compute PSNR successfully
        self.assertIsInstance(metric_result["psnr"], torch.Tensor)
        self.assertGreater(float(metric_result["psnr"]), 0.0)
        self.assertLess(float(metric_result["psnr"]), self.psnr.max_psnr)

        # Test metadata reflects correct shape
        self.assertEqual(metric_result.metadata["input_shape"], [batch_size, 3, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_psnr_grayscale_format(self):
        """Test PSNR computation with grayscale format [h, w]"""
        # Create test data in [h, w] format (grayscale) - normalized to 0-1 range
        gt_gray = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_gray = gt_gray.clone()

        # Add some noise
        torch.manual_seed(42)
        noise = torch.randint(-8, 9, pred_gray.shape, dtype=torch.float32) / 255.0
        pred_gray += noise
        pred_gray = torch.clamp(pred_gray, 0, 1)

        metric_result = self.psnr.compute(pred=pred_gray, target=gt_gray)

        # Should compute PSNR successfully
        self.assertIsInstance(metric_result["psnr"], torch.Tensor)
        self.assertGreater(float(metric_result["psnr"]), 0.0)
        self.assertLess(float(metric_result["psnr"]), self.psnr.max_psnr)

        # Test metadata reflects correct shape
        self.assertEqual(metric_result.metadata["input_shape"], [self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_psnr_single_channel_format(self):
        """Test PSNR computation with single channel format [1, h, w]"""
        # Create test data in [1, h, w] format - normalized to 0-1 range
        gt_single = torch.ones((1, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_single = gt_single.clone()

        # Add some noise
        torch.manual_seed(42)
        noise = torch.randint(-12, 13, pred_single.shape, dtype=torch.float32) / 255.0
        pred_single += noise
        pred_single = torch.clamp(pred_single, 0, 1)

        metric_result = self.psnr.compute(pred=pred_single, target=gt_single)

        # Should compute PSNR successfully
        self.assertIsInstance(metric_result["psnr"], torch.Tensor)
        self.assertGreater(float(metric_result["psnr"]), 0.0)
        self.assertLess(float(metric_result["psnr"]), self.psnr.max_psnr)

        # Test metadata reflects correct shape
        self.assertEqual(metric_result.metadata["input_shape"], [1, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_psnr_batch_grayscale_format(self):
        """Test PSNR computation with batch grayscale format [5, h, w]"""
        batch_size = 5

        # Create test data in [5, h, w] format - normalized to 0-1 range
        gt_batch_gray = torch.ones((batch_size, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_batch_gray = gt_batch_gray.clone()

        # Add different noise to each batch item
        torch.manual_seed(42)
        for i in range(batch_size):
            noise = torch.randint(-20, 21, (self.frame_height, self.frame_width), dtype=torch.float32) / 255.0
            pred_batch_gray[i] += noise
        pred_batch_gray = torch.clamp(pred_batch_gray, 0, 1)

        metric_result = self.psnr.compute(pred=pred_batch_gray, target=gt_batch_gray)

        # Should compute PSNR successfully
        self.assertIsInstance(metric_result["psnr"], torch.Tensor)
        self.assertGreater(float(metric_result["psnr"]), 0.0)
        self.assertLess(float(metric_result["psnr"]), self.psnr.max_psnr)

        # Test metadata reflects correct shape
        self.assertEqual(metric_result.metadata["input_shape"], [batch_size, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_psnr_complex_batch_format(self):
        """Test PSNR computation with complex batch format [2, 4, 3, h, w]"""
        batch_size = 2
        sequence_length = 4

        # Create test data in [2, 4, 3, h, w] format - normalized to 0-1 range
        gt_complex = (
            torch.ones((batch_size, sequence_length, 3, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        )
        pred_complex = gt_complex.clone()

        # Add noise
        torch.manual_seed(42)
        noise = torch.randint(-25, 26, pred_complex.shape, dtype=torch.float32) / 255.0
        pred_complex += noise
        pred_complex = torch.clamp(pred_complex, 0, 1)

        metric_result = self.psnr.compute(pred=pred_complex, target=gt_complex)

        # Should compute PSNR successfully
        self.assertIsInstance(metric_result["psnr"], torch.Tensor)
        self.assertGreater(float(metric_result["psnr"]), 0.0)
        self.assertLess(float(metric_result["psnr"]), self.psnr.max_psnr)

        # Test metadata reflects correct shape
        expected_shape = [batch_size, sequence_length, 3, self.frame_height, self.frame_width]
        self.assertEqual(metric_result.metadata["input_shape"], expected_shape)
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_psnr_mask_with_different_formats(self):
        """Test PSNR computation with masks for different tensor formats"""
        # Test with RGB format [3, h, w] - normalized to 0-1 range
        gt_rgb = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_rgb = gt_rgb.clone()

        # Create mask for RGB format
        mask_rgb = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        mask_rgb[10:30, 20:50] = False  # Mask out a region

        metric_result = self.psnr.compute(pred=pred_rgb, target=gt_rgb, mask=mask_rgb)

        # Should work correctly
        self.assertAlmostEqual(float(metric_result["psnr"]), self.psnr.max_psnr, places=1)

        # Test masked pixels calculation
        excluded_area = torch.sum(torch.logical_not(mask_rgb)).item()
        expected_masked_pixels = self.frame_height * self.frame_width - excluded_area
        self.assertEqual(metric_result.metadata["masked_pixels"], expected_masked_pixels)

    def test_psnr_shape_validation(self):
        """Test PSNR shape validation for various invalid cases"""
        # Test with 1D tensor (insufficient dimensions)
        invalid_1d = torch.tensor([1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            self.psnr.compute(invalid_1d, invalid_1d)

        # Test with mismatched shapes
        gt_mismatch = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_mismatch = torch.ones((3, self.frame_height + 10, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_mismatch, gt_mismatch)

        # Test with mask shape mismatch
        gt_valid = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_valid = gt_valid.clone()
        invalid_mask = torch.ones((self.frame_height + 5, self.frame_width), dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_valid, gt_valid, mask=invalid_mask)

        # Test with non-boolean mask
        invalid_mask_type = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_valid, gt_valid, mask=invalid_mask_type)

    def test_psnr_mask_shape_validation(self):
        """Test that mask shape validation works correctly with pred.shape[-2:] approach"""
        # Test with different tensor formats to ensure mask validation works

        # Format 1: [c, h, w] - mask should be [h, w]
        gt_chw = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_chw = gt_chw.clone()
        mask_correct = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        mask_wrong = torch.ones((self.frame_height + 5, self.frame_width), dtype=torch.bool)

        # Should work with correct mask shape
        result = self.psnr.compute(pred_chw, gt_chw, mask=mask_correct)
        self.assertIsInstance(result["psnr"], torch.Tensor)

        # Should fail with wrong mask shape
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_chw, gt_chw, mask=mask_wrong)

        # Format 2: [b, c, h, w] - mask should still be [h, w]
        batch_size = 4
        gt_bchw = torch.ones((batch_size, 3, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_bchw = gt_bchw.clone()

        # Should work with correct mask shape (same as before)
        result = self.psnr.compute(pred_bchw, gt_bchw, mask=mask_correct)
        self.assertIsInstance(result["psnr"], torch.Tensor)

        # Should fail with wrong mask shape
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_bchw, gt_bchw, mask=mask_wrong)

        # Format 3: [h, w] (grayscale) - mask should be [h, w]
        gt_gray = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)
        pred_gray = gt_gray.clone()

        # Should work with correct mask shape
        result = self.psnr.compute(pred_gray, gt_gray, mask=mask_correct)
        self.assertIsInstance(result["psnr"], torch.Tensor)

        # Should fail with wrong mask shape
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_gray, gt_gray, mask=mask_wrong)

    def test_psnr_consistency_across_formats(self):
        """Test that PSNR gives consistent results across different tensor formats for same data"""
        # Create identical data in different formats
        base_value = 128.0

        # Format 1: [c, h, w] (channels first format - our standard)
        gt_chw = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * base_value
        pred_chw = gt_chw.clone()

        # Format 2: [h, w, c] (channels last format)
        gt_hwc = gt_chw.permute(1, 2, 0)  # [c, h, w] -> [h, w, c]
        pred_hwc = pred_chw.permute(1, 2, 0)

        # Format 3: [1, c, h, w] (batch of 1)
        gt_bchw = gt_chw.unsqueeze(0)  # [c, h, w] -> [1, c, h, w]
        pred_bchw = pred_chw.unsqueeze(0)

        # Compute PSNR for all formats
        result_chw = self.psnr.compute(pred_chw, gt_chw)
        self.psnr.reset()
        result_hwc = self.psnr.compute(pred_hwc, gt_hwc)
        self.psnr.reset()
        result_bchw = self.psnr.compute(pred_bchw, gt_bchw)

        # All should give identical PSNR values (perfect match)
        self.assertAlmostEqual(float(result_chw["psnr"]), self.psnr.max_psnr, places=1)
        self.assertAlmostEqual(float(result_hwc["psnr"]), self.psnr.max_psnr, places=1)
        self.assertAlmostEqual(float(result_bchw["psnr"]), self.psnr.max_psnr, places=1)

    def test_psnr_mask_broadcasting(self):
        """Test PSNR computation with mask broadcasting for different tensor shapes"""
        # Test case 1: [c, h, w] with [h, w] mask
        gt_chw = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 128
        pred_chw = gt_chw.clone()
        mask_hw = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        mask_hw[20:40, 30:60] = False  # Mask out a region

        result = self.psnr.compute(pred_chw, gt_chw, mask=mask_hw)
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=1)

        # Test case 2: [b, c, h, w] with [h, w] mask (your example)
        batch_size = 10
        gt_bchw = torch.ones((batch_size, 3, self.frame_height, self.frame_width), dtype=torch.float32) * 128
        pred_bchw = gt_bchw.clone()

        result = self.psnr.compute(pred_bchw, gt_bchw, mask=mask_hw)
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=1)

        # Test case 3: [h, w] (grayscale) with [h, w] mask
        gt_gray = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32) * 128
        pred_gray = gt_gray.clone()

        result = self.psnr.compute(pred_gray, gt_gray, mask=mask_hw)
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=1)

    def test_psnr_inf_handling(self):
        """Test that PSNR handles infinite values correctly"""
        # Create identical tensors (should give inf PSNR)
        gt_identical = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 128
        pred_identical = gt_identical.clone()

        # Test without mask
        result = self.psnr.compute(pred_identical, gt_identical)
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=1)

        # Test with mask
        mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        mask[20:40, 30:60] = False

        result = self.psnr.compute(pred_identical, gt_identical, mask=mask)
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=1)

        # Test with batch dimension
        gt_batch = torch.ones((5, 3, self.frame_height, self.frame_width), dtype=torch.float32) * 128
        pred_batch = gt_batch.clone()

        result = self.psnr.compute(pred_batch, gt_batch, mask=mask)
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=1)

        # Verify that the result is finite and equals max_psnr
        self.assertTrue(torch.isfinite(result["psnr"]))
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=4)

        # Test edge case: create a scenario where PSNR would be inf
        # Use very small data_range to make PSNR calculation more likely to produce inf
        psnr_small_range = PSNRMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)
        gt_small = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_small = gt_small.clone()

        result_small = psnr_small_range.compute(pred_small, gt_small)
        # Should handle inf correctly and return max_psnr
        self.assertTrue(torch.isfinite(result_small["psnr"]))
        self.assertAlmostEqual(float(result_small["psnr"]), psnr_small_range.max_psnr, places=5)

    def test_psnr_mask_broadcasting_with_noise(self):
        """Test PSNR mask broadcasting with noisy data"""
        # Create test data with noise - normalized to 0-1 range
        gt_noisy = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_noisy = gt_noisy.clone()

        # Add noise to prediction
        torch.manual_seed(42)
        noise = torch.randint(-10, 11, pred_noisy.shape, dtype=torch.float32) / 255.0
        pred_noisy += noise
        pred_noisy = torch.clamp(pred_noisy, 0, 1)

        # Create mask that excludes noisy regions
        mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        mask[20:40, 30:60] = False  # Mask out a region

        # Test with different tensor shapes
        # Case 1: [c, h, w]
        result_chw = self.psnr.compute(pred_noisy, gt_noisy, mask=mask)

        # Case 2: [b, c, h, w]
        batch_size = 4
        gt_batch = gt_noisy.unsqueeze(0).expand(batch_size, -1, -1, -1)
        pred_batch = pred_noisy.unsqueeze(0).expand(batch_size, -1, -1, -1)

        result_batch = self.psnr.compute(pred_batch, gt_batch, mask=mask)

        # Both should give similar PSNR values (same underlying data)
        self.assertAlmostEqual(float(result_chw["psnr"]), float(result_batch["psnr"]), places=1)

        # Both should be less than max_psnr (due to noise)
        self.assertLess(float(result_chw["psnr"]), self.psnr.max_psnr)
        self.assertLess(float(result_batch["psnr"]), self.psnr.max_psnr)

        # Test that masked pixels are correctly counted
        expected_masked_pixels = torch.sum(mask).item()
        self.assertEqual(result_chw.metadata["masked_pixels"], expected_masked_pixels)
        self.assertEqual(result_batch.metadata["masked_pixels"], expected_masked_pixels)

    def test_psnr_mask_validation(self):
        """Test that mask validation correctly enforces [h, w] shape"""
        # Test with valid [h, w] mask
        gt_valid = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_valid = gt_valid.clone()
        valid_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)

        # This should work
        result = self.psnr.compute(pred_valid, gt_valid, mask=valid_mask)
        self.assertIsInstance(result["psnr"], torch.Tensor)

        # Test with wrong spatial dimensions
        invalid_mask = torch.ones((self.frame_height + 5, self.frame_width), dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_valid, gt_valid, mask=invalid_mask)

        # Test with [c, h, w] mask (should fail - mask must be [h, w])
        mask_chw = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_valid, gt_valid, mask=mask_chw)

        # Test with [1, h, w] mask (should fail - mask must be [h, w])
        mask_3d = torch.ones((1, self.frame_height, self.frame_width), dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_valid, gt_valid, mask=mask_3d)

        # Test with [b, c, h, w] mask (should fail - mask must be [h, w])
        mask_bchw = torch.ones((2, 3, self.frame_height, self.frame_width), dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.psnr.compute(pred_valid, gt_valid, mask=mask_bchw)

        # Test with batch dimension input but [h, w] mask (should work)
        gt_batch = torch.ones((2, 3, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_batch = gt_batch.clone()

        result = self.psnr.compute(pred_batch, pred_batch, mask=valid_mask)
        self.assertIsInstance(result["psnr"], torch.Tensor)

        # Test that mask broadcasting works correctly with different input shapes
        # The mask should be applied to each batch/channel slice

        # Test with partial mask (only some pixels valid)
        partial_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        partial_mask[20:40, 30:60] = False  # Mask out a region

        # Test with [c, h, w] input
        result_chw = self.psnr.compute(pred_valid, gt_valid, mask=partial_mask)
        expected_masked_pixels = torch.sum(partial_mask).item()
        self.assertEqual(result_chw.metadata["masked_pixels"], expected_masked_pixels)

        # Test with [b, c, h, w] input
        result_bchw = self.psnr.compute(pred_batch, gt_batch, mask=partial_mask)
        self.assertEqual(result_bchw.metadata["masked_pixels"], expected_masked_pixels)

        # Both should give same PSNR since they have same underlying data
        self.assertAlmostEqual(float(result_chw["psnr"]), float(result_bchw["psnr"]), places=5)

    def test_psnr_no_batch_dimension(self):
        """Test that PSNR works correctly without batch dimensions"""
        # Test with [c, h, w] shape - normalized to 0-1 range
        gt_chw = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_chw = gt_chw.clone()

        result = self.psnr.compute(pred_chw, gt_chw)
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=4)

        # Test with [h, w] shape - normalized to 0-1 range
        gt_hw = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        pred_hw = gt_hw.clone()

        result = self.psnr.compute(pred_hw, gt_hw)
        self.assertAlmostEqual(float(result["psnr"]), self.psnr.max_psnr, places=4)

    def test_psnr_aggregate_single_value(self):
        """Test PSNR aggregation with a single stored value"""
        # Create PSNR metric with multiple aggregation methods
        psnr_multi = PSNRMetric(
            data_range=self.data_range,
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.SUM,
                AggregationMethod.MIN,
                AggregationMethod.MAX,
            ],
        )

        # Compute and store a single value
        result = psnr_multi.compute(pred=self.eval_frame_perfect, target=self.gt_frame)
        psnr_multi.append(result)

        # Aggregate the single value
        aggregated = psnr_multi.aggregate()

        # Should have results for all aggregation methods
        self.assertEqual(len(aggregated), 4)
        self.assertIn(AggregationMethod.MEAN, aggregated)
        self.assertIn(AggregationMethod.SUM, aggregated)
        self.assertIn(AggregationMethod.MIN, aggregated)
        self.assertIn(AggregationMethod.MAX, aggregated)

        # For a single value, all aggregation methods should give the same result
        single_value = float(result["psnr"])
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MEAN]["psnr"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.SUM]["psnr"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MIN]["psnr"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MAX]["psnr"]), single_value, places=6)

    def test_psnr_aggregate_multiple_values(self):
        """Test PSNR aggregation with multiple stored values"""
        # Create PSNR metric with multiple aggregation methods
        psnr_multi = PSNRMetric(
            data_range=self.data_range,
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.SUM,
                AggregationMethod.MIN,
                AggregationMethod.MAX,
            ],
        )

        # Create different test frames with varying noise levels
        test_frames = []
        for i in range(5):
            # Create frame with different noise levels
            gt_frame = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
            pred_frame = gt_frame.clone()

            # Add different noise
            torch.manual_seed(42 + i)
            noise = torch.randint(-20, 21, pred_frame.shape, dtype=torch.float32) / 255.0
            pred_frame += noise
            pred_frame = torch.clamp(pred_frame, 0, 1)

            test_frames.append((pred_frame, gt_frame))

        # Compute and store multiple values
        stored_values = []
        for pred, target in test_frames:
            result = psnr_multi.compute(pred=pred, target=target)
            psnr_multi.append(result)
            stored_values.append(float(result["psnr"]))

        # Aggregate the values
        aggregated = psnr_multi.aggregate()

        # Test that aggregation methods work correctly
        self.assertAlmostEqual(
            float(aggregated[AggregationMethod.MEAN]["psnr"]), sum(stored_values) / len(stored_values), places=5
        )
        self.assertAlmostEqual(float(aggregated[AggregationMethod.SUM]["psnr"]), sum(stored_values), places=5)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MIN]["psnr"]), min(stored_values), places=5)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MAX]["psnr"]), max(stored_values), places=5)

        # Verify that MIN <= MEAN <= MAX
        min_val = float(aggregated[AggregationMethod.MIN]["psnr"])
        mean_val = float(aggregated[AggregationMethod.MEAN]["psnr"])
        max_val = float(aggregated[AggregationMethod.MAX]["psnr"])

        self.assertLessEqual(min_val, mean_val)
        self.assertLessEqual(mean_val, max_val)

    def test_psnr_aggregate_with_perfect_and_noisy_frames(self):
        """Test PSNR aggregation with mix of perfect and noisy frames"""
        # Create PSNR metric with mean aggregation
        psnr_mean = PSNRMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Store perfect match (should give max PSNR)
        perfect_result = psnr_mean.compute(pred=self.eval_frame_perfect, target=self.gt_frame)
        psnr_mean.append(perfect_result)

        # Store low noise frame
        low_noise_result = psnr_mean.compute(pred=self.eval_frame_low_noise, target=self.gt_frame)
        psnr_mean.append(low_noise_result)

        # Store high noise frame
        high_noise_result = psnr_mean.compute(pred=self.eval_frame_high_noise, target=self.gt_frame)
        psnr_mean.append(high_noise_result)

        # Aggregate
        aggregated = psnr_mean.aggregate()

        # Should have mean aggregation result
        self.assertIn(AggregationMethod.MEAN, aggregated)

        # Mean should be between min and max PSNR values
        perfect_psnr = float(perfect_result["psnr"])
        low_noise_psnr = float(low_noise_result["psnr"])
        high_noise_psnr = float(high_noise_result["psnr"])
        mean_psnr = float(aggregated[AggregationMethod.MEAN]["psnr"])

        min_psnr = min(perfect_psnr, low_noise_psnr, high_noise_psnr)
        max_psnr = max(perfect_psnr, low_noise_psnr, high_noise_psnr)

        self.assertGreaterEqual(mean_psnr, min_psnr)
        self.assertLessEqual(mean_psnr, max_psnr)

        # Mean should be approximately the arithmetic mean
        expected_mean = (perfect_psnr + low_noise_psnr + high_noise_psnr) / 3
        self.assertAlmostEqual(mean_psnr, expected_mean, places=5)

    def test_psnr_aggregate_empty_values(self):
        """Test PSNR aggregation with no stored values"""
        # Create PSNR metric with multiple aggregation methods
        psnr_empty = PSNRMetric(
            data_range=self.data_range,
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.SUM,
                AggregationMethod.MIN,
                AggregationMethod.MAX,
            ],
        )

        # Try to aggregate with no values stored
        aggregated = psnr_empty.aggregate()

        # Should return empty dictionary when no values are stored
        self.assertEqual(len(aggregated), 0)

    def test_psnr_aggregate_single_method(self):
        """Test PSNR aggregation with only one aggregation method"""
        # Create PSNR metric with only SUM aggregation
        psnr_sum = PSNRMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.SUM)

        # Store multiple values
        for i in range(3):
            gt_frame = torch.ones((3, self.frame_height, self.frame_width), dtype=torch.float32) * 128
            pred_frame = gt_frame.clone()

            # Add different noise
            torch.manual_seed(42 + i)
            noise = torch.randint(-20, 21, pred_frame.shape, dtype=torch.float32) / 255.0
            pred_frame += noise
            pred_frame = torch.clamp(pred_frame, 0, 1)

            result = psnr_sum.compute(pred=pred_frame, target=gt_frame)
            psnr_sum.append(result)

        # Aggregate
        aggregated = psnr_sum.aggregate()

        # Should only have SUM aggregation
        self.assertEqual(len(aggregated), 1)
        self.assertIn(AggregationMethod.SUM, aggregated)
        self.assertNotIn(AggregationMethod.MEAN, aggregated)
        self.assertNotIn(AggregationMethod.MIN, aggregated)
        self.assertNotIn(AggregationMethod.MAX, aggregated)

    def test_psnr_aggregate_with_masks(self):
        """Test PSNR aggregation with masked computations"""
        # Create PSNR metric with mean aggregation
        psnr_masked = PSNRMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Create different masks
        mask1 = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        mask1[10:30, 20:40] = False  # Mask out a region

        mask2 = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        mask2[50:70, 60:80] = False  # Mask out a different region

        # Store results with different masks
        result1 = psnr_masked.compute(pred=self.eval_frame_perfect, target=self.gt_frame, mask=mask1)
        psnr_masked.append(result1)

        result2 = psnr_masked.compute(pred=self.eval_frame_perfect, target=self.gt_frame, mask=mask2)
        psnr_masked.append(result2)

        # Aggregate
        aggregated = psnr_masked.aggregate()

        # Should work correctly with masked computations
        self.assertIn(AggregationMethod.MEAN, aggregated)
        mean_psnr = float(aggregated[AggregationMethod.MEAN]["psnr"])

        # Both should be perfect PSNR since frames are identical
        self.assertAlmostEqual(float(result1["psnr"]), self.psnr.max_psnr, places=1)
        self.assertAlmostEqual(float(result2["psnr"]), self.psnr.max_psnr, places=1)
        self.assertAlmostEqual(mean_psnr, self.psnr.max_psnr, places=1)

    def test_psnr_aggregate_clear_functionality(self):
        """Test that clear functionality works with aggregation"""
        # Create PSNR metric
        psnr_clear = PSNRMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Store some values
        for i in range(3):
            result = psnr_clear.compute(pred=self.eval_frame_perfect, target=self.gt_frame)
            psnr_clear.append(result)

        # Verify values are stored
        self.assertEqual(len(psnr_clear), 3)

        # Clear the values
        psnr_clear.clear()

        # Verify values are cleared
        self.assertEqual(len(psnr_clear), 0)

        # Aggregate should return empty dict
        aggregated = psnr_clear.aggregate()
        self.assertEqual(len(aggregated), 0)

    def test_psnr_aggregate_device_handling(self):
        """Test PSNR aggregation with device handling"""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            # Create PSNR metric and move to device
            psnr_device = PSNRMetric(
                data_range=self.data_range, aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.SUM]
            ).to(device)

            # Move test data to device
            gt_device = self.gt_frame.to(device)
            pred_device = self.eval_frame_perfect.to(device)

            # Store multiple values on device
            for i in range(3):
                result = psnr_device.compute(pred=pred_device, target=gt_device)
                psnr_device.append(result)

            # Aggregate on device
            aggregated = psnr_device.aggregate()

            # Should work without errors
            self.assertIn(AggregationMethod.MEAN, aggregated)
            self.assertIn(AggregationMethod.SUM, aggregated)

            # Check that results are on the correct device
            mean_psnr = aggregated[AggregationMethod.MEAN]["psnr"]
            sum_psnr = aggregated[AggregationMethod.SUM]["psnr"]

            self.assertEqual(mean_psnr.device.type, device.type)
            self.assertEqual(sum_psnr.device.type, device.type)

    def test_psnr_aggregate_metric_result_structure(self):
        """Test that aggregated results maintain proper MetricResult structure"""
        # Create PSNR metric with multiple aggregation methods
        psnr_structure = PSNRMetric(
            data_range=self.data_range,
            aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.MIN, AggregationMethod.MAX],
        )

        # Store some values
        for i in range(2):
            result = psnr_structure.compute(pred=self.eval_frame_perfect, target=self.gt_frame)
            psnr_structure.append(result)

        # Aggregate
        aggregated = psnr_structure.aggregate()

        # Test structure for each aggregation method
        for method, result in aggregated.items():
            # Should be MetricResult instance
            self.assertIsInstance(result, type(psnr_structure.compute(self.eval_frame_perfect, self.gt_frame)))

            # Should have 'psnr' key in values
            self.assertIn("psnr", result.values)

            # Should be a tensor
            self.assertIsInstance(result.values["psnr"], torch.Tensor)

            # Should be a scalar tensor
            self.assertEqual(result.values["psnr"].dim(), 0)

    def test_psnr_aggregate_consistency_with_compute(self):
        """Test that aggregation is consistent with individual compute results"""
        # Create PSNR metric
        psnr_consistency = PSNRMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Create test frames with known PSNR values
        test_cases = [
            (self.eval_frame_perfect, self.gt_frame),  # Perfect match
            (self.eval_frame_low_noise, self.gt_frame),  # Low noise
            (self.eval_frame_high_noise, self.gt_frame),  # High noise
        ]

        # Store results
        individual_results = []
        for pred, target in test_cases:
            result = psnr_consistency.compute(pred=pred, target=target)
            psnr_consistency.append(result)
            individual_results.append(float(result["psnr"]))

        # Aggregate
        aggregated = psnr_consistency.aggregate()
        mean_psnr = float(aggregated[AggregationMethod.MEAN]["psnr"])

        # Should match manual calculation
        expected_mean = sum(individual_results) / len(individual_results)
        self.assertAlmostEqual(mean_psnr, expected_mean, places=5)

        # Verify the relationship between individual values
        self.assertGreater(individual_results[0], individual_results[1])  # Perfect > Low noise
        self.assertGreater(individual_results[1], individual_results[2])  # Low noise > High noise

    def test_psnr_metadata(self):
        """Test that PSNR metric has metadata."""
        psnr_metric = PSNRMetric(data_range=self.data_range)
        self.assertEqual(psnr_metric.metadata(), {"data_range": 1.0})


if __name__ == "__main__":
    unittest.main()
