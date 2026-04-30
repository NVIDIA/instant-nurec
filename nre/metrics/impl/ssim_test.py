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

from nre.metrics.impl.ssim import SSIMMetric
from nre.metrics.utils import AggregationMethod


class TestSSIMMetric(unittest.TestCase):
    def setUp(self):
        self.data_range = 1.0
        self.ssim = SSIMMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Create test frames as tensors
        self.frame_height = 64
        self.frame_width = 64
        self.channels = 3

        # Gray image (0.5 intensity) - normalized to 0-1 range
        self.gray_image = torch.ones((self.channels, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5

        # White image (all 1's) and black image (all 0's) for max difference tests
        self.white_image = torch.ones((self.channels, self.frame_height, self.frame_width), dtype=torch.float32)
        self.black_image = torch.zeros((self.channels, self.frame_height, self.frame_width), dtype=torch.float32)

        # Random image - normalized to 0-1 range
        torch.manual_seed(42)
        self.random_image = torch.rand((self.channels, self.frame_height, self.frame_width), dtype=torch.float32)

        # Clone of random image for perfect match tests
        self.random_image_clone = self.random_image.clone()

    def test_ssim_perfect_match(self):
        """Test SSIM computation with identical frames"""
        metric_result = self.ssim.compute(pred=self.random_image_clone, target=self.random_image)

        # For identical frames, SSIM should be 1.0
        self.assertAlmostEqual(float(metric_result["ssim"]), 1.0, places=4)

        # Test metadata
        self.assertIn("data_range", metric_result.metadata)
        self.assertIn("input_shape", metric_result.metadata)
        self.assertIn("masked_pixels", metric_result.metadata)
        self.assertIn("kernel_size", metric_result.metadata)
        self.assertEqual(metric_result.metadata["data_range"], self.data_range)
        self.assertEqual(metric_result.metadata["input_shape"], [self.channels, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_ssim_monotonicity(self):
        """Test SSIM decreases with increasing difference"""
        metric_result_identical = self.ssim.compute(self.random_image_clone, self.random_image)
        metric_result_random = self.ssim.compute(self.random_image, self.gray_image)
        metric_result_max_diff = self.ssim.compute(self.black_image, self.white_image)

        # Test SSIM monotonicity: identical > random > max_difference
        ssim_identical = float(metric_result_identical["ssim"])
        ssim_random = float(metric_result_random["ssim"])
        ssim_max_diff = float(metric_result_max_diff["ssim"])

        self.assertGreater(ssim_identical, ssim_random)
        self.assertGreater(ssim_random, ssim_max_diff)

        # Identical should be 1.0
        self.assertAlmostEqual(ssim_identical, 1.0, places=4)

        # Max difference should be very low
        self.assertLess(ssim_max_diff, 0.1)

    def test_ssim_with_mask(self):
        """Test SSIM computation with boolean mask affects the result"""
        # Create test mask
        valid_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        valid_mask[20:40, 30:60] = False  # Create a region to mask out

        # Use random vs white images so we can verify mask actually changes the result
        # Without mask: comparing random vs white gives low SSIM
        result_no_mask = self.ssim.compute(self.random_image, self.white_image)
        ssim_no_mask = float(result_no_mask["ssim"])

        # With mask: masked regions become identical (both set to 0), increasing similarity
        result_with_mask = self.ssim.compute(self.random_image, self.white_image, mask=valid_mask)
        ssim_with_mask = float(result_with_mask["ssim"])

        # Masked result should be higher than unmasked (more regions are identical)
        self.assertGreater(ssim_with_mask, ssim_no_mask)

        # Both should be less than 1.0 (not testing identical images)
        self.assertLess(ssim_no_mask, 1.0)
        self.assertLess(ssim_with_mask, 1.0)

        # Test metadata with mask which counts valid pixels (where mask is True)
        excluded_area = torch.sum(torch.logical_not(valid_mask)).item()
        expected_masked_pixels = self.frame_height * self.frame_width - excluded_area
        self.assertEqual(result_with_mask.metadata["masked_pixels"], expected_masked_pixels)

    def test_ssim_max_difference(self):
        """Test SSIM computation with maximum difference (all 0's vs all 1's)"""
        metric_result = self.ssim.compute(self.black_image, self.white_image)

        # For data_range=1 with max difference (1 vs 0), SSIM should be very low
        self.assertLess(float(metric_result["ssim"]), 0.1)

        # Test metadata
        self.assertEqual(metric_result.metadata["data_range"], self.data_range)
        self.assertEqual(metric_result.metadata["input_shape"], [self.channels, self.frame_height, self.frame_width])

    def test_ssim_metadata_structure(self):
        """Test that metadata has the correct structure and types"""
        metric_result = self.ssim.compute(self.random_image_clone, self.random_image)

        # Test metadata keys exist
        required_keys = ["data_range", "kernel_size", "input_shape", "masked_pixels"]
        for key in required_keys:
            self.assertIn(key, metric_result.metadata)

        # Test metadata types
        self.assertIsInstance(metric_result.metadata["data_range"], (int, float))
        self.assertIsInstance(metric_result.metadata["kernel_size"], int)
        self.assertIsInstance(metric_result.metadata["input_shape"], list)
        self.assertIsInstance(metric_result.metadata["masked_pixels"], int)

        # Test metadata values
        self.assertEqual(metric_result.metadata["data_range"], 1.0)
        self.assertEqual(len(metric_result.metadata["input_shape"]), 3)
        self.assertEqual(metric_result.metadata["input_shape"][0], self.channels)
        self.assertEqual(metric_result.metadata["input_shape"][1], self.frame_height)
        self.assertEqual(metric_result.metadata["input_shape"][2], self.frame_width)

    def test_ssim_metric_result_interface(self):
        """Test MetricResult interface methods"""
        metric_result = self.ssim.compute(self.random_image_clone, self.random_image)

        # Test get_value method
        ssim_value = metric_result.get_value("ssim")
        self.assertIsInstance(ssim_value, torch.Tensor)

        # Test get_available_values method
        available_values = metric_result.get_available_values()
        self.assertIn("ssim", available_values)
        self.assertEqual(len(available_values), 1)

        # Test to_dict method
        result_dict = metric_result.to_dict()
        self.assertIn("ssim", result_dict)
        self.assertIsInstance(result_dict["ssim"], torch.Tensor)

        # Test dictionary-like access
        self.assertTrue("ssim" in metric_result)
        self.assertEqual(metric_result["ssim"], ssim_value)

    def test_ssim_reset_functionality(self):
        """Test that reset functionality works correctly"""
        # Compute SSIM first
        metric_result1 = self.ssim.compute(self.random_image_clone, self.random_image)

        # Reset the metric
        self.ssim.reset()

        # Compute SSIM again - should work the same
        metric_result2 = self.ssim.compute(self.random_image_clone, self.random_image)

        # Results should be identical
        self.assertAlmostEqual(float(metric_result1["ssim"]), float(metric_result2["ssim"]), places=6)

    def test_ssim_device_handling(self):
        """Test SSIM metric device handling"""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            # Move metric to device
            self.ssim.to(device)

            # Move test data to device
            random_image_device = self.random_image.to(device)
            random_image_clone_device = self.random_image_clone.to(device)

            # Compute SSIM on device
            metric_result = self.ssim.compute(random_image_clone_device, random_image_device)

            # Should work without errors
            self.assertIsInstance(metric_result["ssim"], torch.Tensor)
            # Check device type matches (ignore index)
            self.assertEqual(metric_result["ssim"].device.type, device.type)

    def test_ssim_input_validation(self):
        """Test SSIM input validation"""
        # Test with invalid input types
        with self.assertRaises(TypeError):
            self.ssim.compute("invalid", self.random_image)

        with self.assertRaises(TypeError):
            self.ssim.compute(self.random_image, "invalid")

    def test_ssim_rgb_channels_first_format(self):
        """Test SSIM computation with RGB channels first format [3, h, w]"""
        # Use existing fixtures which are already in [3, h, w] format
        metric_result = self.ssim.compute(pred=self.random_image, target=self.gray_image)

        # Should compute SSIM successfully
        self.assertIsInstance(metric_result["ssim"], torch.Tensor)
        self.assertGreater(float(metric_result["ssim"]), 0.0)
        self.assertLessEqual(float(metric_result["ssim"]), 1.0)

        # Test metadata reflects correct shape
        self.assertEqual(metric_result.metadata["input_shape"], [self.channels, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_ssim_batch_rgb_format(self):
        """Test SSIM computation with batch RGB format [N, 3, h, w]"""
        batch_size = 2

        # Stack existing fixtures to create batch format
        pred_batch = self.random_image.unsqueeze(0).expand(batch_size, -1, -1, -1)
        target_batch = self.gray_image.unsqueeze(0).expand(batch_size, -1, -1, -1)

        metric_result = self.ssim.compute(pred=pred_batch, target=target_batch)

        # Should compute SSIM successfully
        self.assertIsInstance(metric_result["ssim"], torch.Tensor)
        self.assertGreater(float(metric_result["ssim"]), 0.0)
        self.assertLessEqual(float(metric_result["ssim"]), 1.0)

        # Test metadata reflects correct shape
        self.assertEqual(
            metric_result.metadata["input_shape"], [batch_size, self.channels, self.frame_height, self.frame_width]
        )
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_ssim_grayscale_format(self):
        """Test SSIM computation with grayscale format [1, h, w]"""
        # Create simple grayscale test data in [1, h, w] format
        gray_single = torch.ones((1, self.frame_height, self.frame_width), dtype=torch.float32) * 0.5
        white_single = torch.ones((1, self.frame_height, self.frame_width), dtype=torch.float32)

        metric_result = self.ssim.compute(pred=gray_single, target=white_single)

        # Should compute SSIM successfully
        self.assertIsInstance(metric_result["ssim"], torch.Tensor)
        self.assertGreater(float(metric_result["ssim"]), 0.0)
        self.assertLessEqual(float(metric_result["ssim"]), 1.0)

        # Test metadata reflects correct shape
        self.assertEqual(metric_result.metadata["input_shape"], [1, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_ssim_shape_validation(self):
        """Test SSIM shape validation for various invalid cases"""
        # Test with 2D tensor (missing channel dimension)
        invalid_2d = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.ssim.compute(invalid_2d, invalid_2d)

        # Test with 5D tensor (too many dimensions)
        invalid_5d = torch.ones((1, 1, self.channels, self.frame_height, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.ssim.compute(invalid_5d, invalid_5d)

        # Test with mismatched shapes
        mismatch = torch.ones((self.channels, self.frame_height + 10, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.ssim.compute(self.white_image, mismatch)

        # Test with mask shape mismatch
        invalid_mask = torch.ones((self.frame_height + 5, self.frame_width), dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.ssim.compute(self.random_image, self.random_image_clone, mask=invalid_mask)

        # Test with non-boolean mask
        invalid_mask_type = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.ssim.compute(self.random_image, self.random_image_clone, mask=invalid_mask_type)

    def test_ssim_aggregate_single_value(self):
        """Test SSIM aggregation with a single stored value"""
        # Create SSIM metric with multiple aggregation methods
        ssim_multi = SSIMMetric(
            data_range=self.data_range,
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.SUM,
                AggregationMethod.MIN,
                AggregationMethod.MAX,
            ],
        )

        # Compute and store a single value
        result = ssim_multi.compute(pred=self.random_image_clone, target=self.random_image)
        ssim_multi.append(result)

        # Aggregate the single value
        aggregated = ssim_multi.aggregate()

        # Should have results for all aggregation methods
        self.assertEqual(len(aggregated), 4)
        self.assertIn(AggregationMethod.MEAN, aggregated)
        self.assertIn(AggregationMethod.SUM, aggregated)
        self.assertIn(AggregationMethod.MIN, aggregated)
        self.assertIn(AggregationMethod.MAX, aggregated)

        # For a single value, all aggregation methods should give the same result
        single_value = float(result["ssim"])
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MEAN]["ssim"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.SUM]["ssim"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MIN]["ssim"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MAX]["ssim"]), single_value, places=6)

    def test_ssim_aggregate_multiple_values(self):
        """Test SSIM aggregation with multiple stored values"""
        # Create SSIM metric with multiple aggregation methods
        ssim_multi = SSIMMetric(
            data_range=self.data_range,
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.SUM,
                AggregationMethod.MIN,
                AggregationMethod.MAX,
            ],
        )

        # Use existing fixtures to create different comparison pairs
        test_frames = [
            (self.random_image_clone, self.random_image),  # Identical
            (self.random_image, self.gray_image),  # Different
            (self.black_image, self.white_image),  # Max difference
        ]

        # Compute and store multiple values
        stored_values = []
        for pred, target in test_frames:
            result = ssim_multi.compute(pred=pred, target=target)
            ssim_multi.append(result)
            stored_values.append(float(result["ssim"]))

        # Aggregate the values
        aggregated = ssim_multi.aggregate()

        # Test that aggregation methods work correctly
        self.assertAlmostEqual(
            float(aggregated[AggregationMethod.MEAN]["ssim"]), sum(stored_values) / len(stored_values), places=5
        )
        self.assertAlmostEqual(float(aggregated[AggregationMethod.SUM]["ssim"]), sum(stored_values), places=5)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MIN]["ssim"]), min(stored_values), places=5)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MAX]["ssim"]), max(stored_values), places=5)

        # Verify that MIN <= MEAN <= MAX
        min_val = float(aggregated[AggregationMethod.MIN]["ssim"])
        mean_val = float(aggregated[AggregationMethod.MEAN]["ssim"])
        max_val = float(aggregated[AggregationMethod.MAX]["ssim"])

        self.assertLessEqual(min_val, mean_val)
        self.assertLessEqual(mean_val, max_val)

    def test_ssim_aggregate_with_varied_frames(self):
        """Test SSIM aggregation with mix of identical, random, and max-diff frames"""
        # Create SSIM metric with mean aggregation
        ssim_mean = SSIMMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Store identical match (should give SSIM close to 1.0)
        identical_result = ssim_mean.compute(pred=self.random_image_clone, target=self.random_image)
        ssim_mean.append(identical_result)

        # Store random vs gray comparison
        random_result = ssim_mean.compute(pred=self.random_image, target=self.gray_image)
        ssim_mean.append(random_result)

        # Store max difference frame
        max_diff_result = ssim_mean.compute(pred=self.black_image, target=self.white_image)
        ssim_mean.append(max_diff_result)

        # Aggregate
        aggregated = ssim_mean.aggregate()

        # Should have mean aggregation result
        self.assertIn(AggregationMethod.MEAN, aggregated)

        # Mean should be between min and max SSIM values
        identical_ssim = float(identical_result["ssim"])
        random_ssim = float(random_result["ssim"])
        max_diff_ssim = float(max_diff_result["ssim"])
        mean_ssim = float(aggregated[AggregationMethod.MEAN]["ssim"])

        min_ssim = min(identical_ssim, random_ssim, max_diff_ssim)
        max_ssim = max(identical_ssim, random_ssim, max_diff_ssim)

        self.assertGreaterEqual(mean_ssim, min_ssim)
        self.assertLessEqual(mean_ssim, max_ssim)

        # Mean should be approximately the arithmetic mean
        expected_mean = (identical_ssim + random_ssim + max_diff_ssim) / 3
        self.assertAlmostEqual(mean_ssim, expected_mean, places=5)

    def test_ssim_aggregate_empty_values(self):
        """Test SSIM aggregation with no stored values"""
        # Create SSIM metric with multiple aggregation methods
        ssim_empty = SSIMMetric(
            data_range=self.data_range,
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.SUM,
                AggregationMethod.MIN,
                AggregationMethod.MAX,
            ],
        )

        # Try to aggregate with no values stored
        aggregated = ssim_empty.aggregate()

        # Should return empty dictionary when no values are stored
        self.assertEqual(len(aggregated), 0)

    def test_ssim_aggregate_single_method(self):
        """Test SSIM aggregation with only one aggregation method"""
        # Create SSIM metric with only SUM aggregation
        ssim_sum = SSIMMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.SUM)

        # Store multiple values using existing fixtures
        test_frames = [
            (self.random_image_clone, self.random_image),
            (self.random_image, self.gray_image),
            (self.black_image, self.white_image),
        ]
        for pred, target in test_frames:
            result = ssim_sum.compute(pred=pred, target=target)
            ssim_sum.append(result)

        # Aggregate
        aggregated = ssim_sum.aggregate()

        # Should only have SUM aggregation
        self.assertEqual(len(aggregated), 1)
        self.assertIn(AggregationMethod.SUM, aggregated)
        self.assertNotIn(AggregationMethod.MEAN, aggregated)
        self.assertNotIn(AggregationMethod.MIN, aggregated)
        self.assertNotIn(AggregationMethod.MAX, aggregated)

    def test_ssim_aggregate_clear_functionality(self):
        """Test that clear functionality works with aggregation"""
        # Create SSIM metric
        ssim_clear = SSIMMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Store some values
        for i in range(3):
            result = ssim_clear.compute(pred=self.random_image_clone, target=self.random_image)
            ssim_clear.append(result)

        # Verify values are stored
        self.assertEqual(len(ssim_clear), 3)

        # Clear the values
        ssim_clear.clear()

        # Verify values are cleared
        self.assertEqual(len(ssim_clear), 0)

        # Aggregate should return empty dict
        aggregated = ssim_clear.aggregate()
        self.assertEqual(len(aggregated), 0)

    def test_ssim_aggregate_device_handling(self):
        """Test SSIM aggregation with device handling"""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            # Create SSIM metric and move to device
            ssim_device = SSIMMetric(
                data_range=self.data_range, aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.SUM]
            ).to(device)

            # Move test data to device
            random_image_device = self.random_image.to(device)
            random_image_clone_device = self.random_image_clone.to(device)

            # Store multiple values on device
            for i in range(3):
                result = ssim_device.compute(pred=random_image_clone_device, target=random_image_device)
                ssim_device.append(result)

            # Aggregate on device
            aggregated = ssim_device.aggregate()

            # Should work without errors
            self.assertIn(AggregationMethod.MEAN, aggregated)
            self.assertIn(AggregationMethod.SUM, aggregated)

            # Check that results are on the correct device
            mean_ssim = aggregated[AggregationMethod.MEAN]["ssim"]
            sum_ssim = aggregated[AggregationMethod.SUM]["ssim"]

            self.assertEqual(mean_ssim.device.type, device.type)
            self.assertEqual(sum_ssim.device.type, device.type)

    def test_ssim_aggregate_metric_result_structure(self):
        """Test that aggregated results maintain proper MetricResult structure"""
        # Create SSIM metric with multiple aggregation methods
        ssim_structure = SSIMMetric(
            data_range=self.data_range,
            aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.MIN, AggregationMethod.MAX],
        )

        # Store some values
        for i in range(2):
            result = ssim_structure.compute(pred=self.random_image_clone, target=self.random_image)
            ssim_structure.append(result)

        # Aggregate
        aggregated = ssim_structure.aggregate()

        # Test structure for each aggregation method
        for method, result in aggregated.items():
            # Should be MetricResult instance
            self.assertIsInstance(result, type(ssim_structure.compute(self.random_image_clone, self.random_image)))

            # Should have 'ssim' key in values
            self.assertIn("ssim", result.values)

            # Should be a tensor
            self.assertIsInstance(result.values["ssim"], torch.Tensor)

            # Should be a scalar tensor
            self.assertEqual(result.values["ssim"].dim(), 0)

    def test_ssim_aggregate_consistency_with_compute(self):
        """Test that aggregation is consistent with individual compute results"""
        # Create SSIM metric
        ssim_consistency = SSIMMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Create test frames with known SSIM ordering
        test_cases = [
            (self.random_image_clone, self.random_image),  # Identical
            (self.random_image, self.gray_image),  # Random vs gray
            (self.black_image, self.white_image),  # Max difference
        ]

        # Store results
        individual_results = []
        for pred, target in test_cases:
            result = ssim_consistency.compute(pred=pred, target=target)
            ssim_consistency.append(result)
            individual_results.append(float(result["ssim"]))

        # Aggregate
        aggregated = ssim_consistency.aggregate()
        mean_ssim = float(aggregated[AggregationMethod.MEAN]["ssim"])

        # Should match manual calculation
        expected_mean = sum(individual_results) / len(individual_results)
        self.assertAlmostEqual(mean_ssim, expected_mean, places=5)

        # Verify the relationship between individual values (higher SSIM = more similar)
        self.assertGreater(individual_results[0], individual_results[1])  # Identical > Random
        self.assertGreater(individual_results[1], individual_results[2])  # Random > Max diff

    def test_ssim_metadata(self):
        """Test that SSIM metric has metadata."""
        ssim_metric = SSIMMetric(data_range=self.data_range)
        expected_metadata = {"data_range": 1.0, "kernel_size": 11}
        self.assertEqual(ssim_metric.metadata(), expected_metadata)

    def test_ssim_custom_kernel_size(self):
        """Test SSIM with custom kernel size"""
        ssim_custom = SSIMMetric(data_range=self.data_range, kernel_size=7)

        metric_result = ssim_custom.compute(self.random_image_clone, self.random_image)

        # Should work with custom kernel size
        self.assertAlmostEqual(float(metric_result["ssim"]), 1.0, places=4)
        self.assertEqual(metric_result.metadata["kernel_size"], 7)

    def test_ssim_weighted_mean_not_supported(self):
        """Test that weighted mean aggregation raises an error"""
        with self.assertRaises(ValueError):
            SSIMMetric(data_range=self.data_range, aggregation_methods=AggregationMethod.WEIGHTED_MEAN)


if __name__ == "__main__":
    unittest.main()
