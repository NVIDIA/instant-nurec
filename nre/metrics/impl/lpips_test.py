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

from nre.metrics.impl.lpips import LPIPSMetric
from nre.metrics.utils import AggregationMethod


class TestLPIPSMetric(unittest.TestCase):
    def setUp(self):
        self.lpips = LPIPSMetric(net_type="alex", normalize=True, aggregation_methods=AggregationMethod.MEAN)

        # Create test frames as tensors (LPIPS requires 3 channels)
        self.frame_height = 64  # Smaller for faster tests
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

    def test_lpips_perfect_match(self):
        """Test LPIPS computation with identical frames"""
        metric_result = self.lpips.compute(pred=self.random_image_clone, target=self.random_image)

        # For identical frames, LPIPS should be close to 0 (perfect similarity)
        self.assertAlmostEqual(float(metric_result["lpips"]), 0.0, places=4)

        # Test metadata
        self.assertIn("net_type", metric_result.metadata)
        self.assertIn("normalize", metric_result.metadata)
        self.assertIn("input_shape", metric_result.metadata)
        self.assertIn("masked_pixels", metric_result.metadata)
        self.assertEqual(metric_result.metadata["net_type"], "alex")
        self.assertEqual(metric_result.metadata["normalize"], True)
        self.assertEqual(metric_result.metadata["input_shape"], [self.channels, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_lpips_monotonicity(self):
        """Test LPIPS values for different image comparisons"""
        metric_result_identical = self.lpips.compute(self.random_image_clone, self.random_image)
        metric_result_random = self.lpips.compute(self.random_image, self.gray_image)
        metric_result_max_diff = self.lpips.compute(self.black_image, self.white_image)

        lpips_identical = float(metric_result_identical["lpips"])
        lpips_random = float(metric_result_random["lpips"])
        lpips_max_diff = float(metric_result_max_diff["lpips"])

        # Identical frames should have near-zero LPIPS
        self.assertAlmostEqual(lpips_identical, 0.0, places=4)

        # Different images should have higher LPIPS than identical
        self.assertLess(lpips_identical, lpips_random)
        self.assertLess(lpips_identical, lpips_max_diff)

        # Both non-identical comparisons should have non-trivial LPIPS
        self.assertGreater(lpips_random, 0.1)
        self.assertGreater(lpips_max_diff, 0.1)

        # Note: LPIPS measures perceptual (feature-based) difference, not pixel difference.
        # Two flat images (black vs white) may have lower LPIPS than textured vs flat,
        # because deep network features for uniform images are similar. We don't assert
        # a specific ordering between lpips_random and lpips_max_diff.

    def test_lpips_with_mask(self):
        """Test LPIPS computation with boolean mask affects the result"""
        # Create test mask
        valid_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        valid_mask[20:40, 30:60] = False  # Create a region to mask out

        # Use random vs white images so we can verify mask actually changes the result
        # Without mask: comparing random vs white gives some LPIPS value
        result_no_mask = self.lpips.compute(self.random_image, self.white_image)
        lpips_no_mask = float(result_no_mask["lpips"])

        # With mask: masked regions become identical (both set to 0), reducing perceptual difference
        result_with_mask = self.lpips.compute(self.random_image, self.white_image, mask=valid_mask)
        lpips_with_mask = float(result_with_mask["lpips"])

        # Masked result should be lower than unmasked (more regions are identical)
        self.assertLess(lpips_with_mask, lpips_no_mask)

        # Both should be non-zero (not testing identical images)
        self.assertGreater(lpips_no_mask, 0.1)
        self.assertGreater(lpips_with_mask, 0.1)

        # Test metadata with mask which counts valid pixels (where mask is True)
        excluded_area = torch.sum(torch.logical_not(valid_mask)).item()
        expected_masked_pixels = self.frame_height * self.frame_width - excluded_area
        self.assertEqual(result_with_mask.metadata["masked_pixels"], expected_masked_pixels)

    def test_lpips_max_difference(self):
        """Test LPIPS computation with maximum difference (all 0's vs all 1's)"""

        # For black vs white images
        # alex : 0.8139544129371643
        # vgg: 0.45383718609809875
        # squeeze: 0.642092764377594
        metric_result = self.lpips.compute(self.black_image, self.white_image)
        self.assertGreater(float(metric_result["lpips"]), 0.8)

        # For random vs white images
        # alex : 1.3531250953674316
        # vgg: 0.8114886283874512
        # squeeze: 0.8950120806694031
        metric_result = self.lpips.compute(self.random_image, self.white_image)
        self.assertGreater(float(metric_result["lpips"]), 1.3)

        # Test metadata
        self.assertEqual(metric_result.metadata["net_type"], "alex")
        self.assertEqual(metric_result.metadata["input_shape"], [self.channels, self.frame_height, self.frame_width])

    def test_lpips_metadata_structure(self):
        """Test that metadata has the correct structure and types"""
        metric_result = self.lpips.compute(self.random_image_clone, self.random_image)

        # Test metadata keys exist
        required_keys = ["net_type", "normalize", "input_shape", "masked_pixels"]
        for key in required_keys:
            self.assertIn(key, metric_result.metadata)

        # Test metadata types
        self.assertIsInstance(metric_result.metadata["net_type"], str)
        self.assertIsInstance(metric_result.metadata["normalize"], bool)
        self.assertIsInstance(metric_result.metadata["input_shape"], list)
        self.assertIsInstance(metric_result.metadata["masked_pixels"], int)

        # Test metadata values
        self.assertEqual(metric_result.metadata["net_type"], "alex")
        self.assertEqual(metric_result.metadata["normalize"], True)
        self.assertEqual(len(metric_result.metadata["input_shape"]), 3)
        self.assertEqual(metric_result.metadata["input_shape"][0], self.channels)
        self.assertEqual(metric_result.metadata["input_shape"][1], self.frame_height)
        self.assertEqual(metric_result.metadata["input_shape"][2], self.frame_width)

    def test_lpips_metric_result_interface(self):
        """Test MetricResult interface methods"""
        metric_result = self.lpips.compute(self.random_image_clone, self.random_image)

        # Test get_value method
        lpips_value = metric_result.get_value("lpips")
        self.assertIsInstance(lpips_value, torch.Tensor)

        # Test get_available_values method
        available_values = metric_result.get_available_values()
        self.assertIn("lpips", available_values)
        self.assertEqual(len(available_values), 1)

        # Test to_dict method
        result_dict = metric_result.to_dict()
        self.assertIn("lpips", result_dict)
        self.assertIsInstance(result_dict["lpips"], torch.Tensor)

        # Test dictionary-like access
        self.assertTrue("lpips" in metric_result)
        self.assertEqual(metric_result["lpips"], lpips_value)

    def test_lpips_reset_functionality(self):
        """Test that reset functionality works correctly"""
        # Compute LPIPS first
        metric_result1 = self.lpips.compute(self.random_image_clone, self.random_image)

        # Reset the metric
        self.lpips.reset()

        # Compute LPIPS again - should work the same
        metric_result2 = self.lpips.compute(self.random_image_clone, self.random_image)

        # Results should be identical
        self.assertAlmostEqual(float(metric_result1["lpips"]), float(metric_result2["lpips"]), places=6)

    def test_lpips_device_handling(self):
        """Test LPIPS metric device handling"""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            # Move metric to device
            self.lpips.to(device)

            # Move test data to device
            random_image_device = self.random_image.to(device)
            random_image_clone_device = self.random_image_clone.to(device)

            # Compute LPIPS on device
            metric_result = self.lpips.compute(random_image_clone_device, random_image_device)

            # Should work without errors
            self.assertIsInstance(metric_result["lpips"], torch.Tensor)
            # Check device type matches (ignore index)
            self.assertEqual(metric_result["lpips"].device.type, device.type)

    def test_lpips_input_validation(self):
        """Test LPIPS input validation"""
        # Test with invalid input types
        with self.assertRaises(TypeError):
            self.lpips.compute("invalid", self.random_image)

        with self.assertRaises(TypeError):
            self.lpips.compute(self.random_image_clone, "invalid")

    def test_lpips_requires_three_channels(self):
        """Test that LPIPS requires exactly 3 channels"""
        # Test with 1 channel (grayscale)
        gt_gray = torch.ones((1, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_gray = gt_gray.clone()
        with self.assertRaises(ValueError):
            self.lpips.compute(pred_gray, gt_gray)

        # Test with 2 channels
        gt_2ch = torch.ones((2, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_2ch = gt_2ch.clone()
        with self.assertRaises(ValueError):
            self.lpips.compute(pred_2ch, gt_2ch)

        # Test with 4 channels (RGBA)
        gt_rgba = torch.ones((4, self.frame_height, self.frame_width), dtype=torch.float32)
        pred_rgba = gt_rgba.clone()
        with self.assertRaises(ValueError):
            self.lpips.compute(pred_rgba, gt_rgba)

    def test_lpips_rgb_channels_first_format(self):
        """Test LPIPS computation with RGB channels first format [3, h, w]"""
        # Use existing fixtures which are already in [3, h, w] format
        metric_result = self.lpips.compute(pred=self.random_image, target=self.gray_image)

        # Should compute LPIPS successfully
        self.assertIsInstance(metric_result["lpips"], torch.Tensor)
        self.assertGreaterEqual(float(metric_result["lpips"]), 0.0)

        # Test metadata reflects correct shape
        self.assertEqual(metric_result.metadata["input_shape"], [self.channels, self.frame_height, self.frame_width])
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_lpips_batch_rgb_format(self):
        """Test LPIPS computation with batch RGB format [N, 3, h, w]"""
        batch_size = 2

        # Stack existing fixtures to create batch format
        pred_batch = self.random_image.unsqueeze(0).expand(batch_size, -1, -1, -1)
        target_batch = self.gray_image.unsqueeze(0).expand(batch_size, -1, -1, -1)

        metric_result = self.lpips.compute(pred=pred_batch, target=target_batch)

        # Should compute LPIPS successfully
        self.assertIsInstance(metric_result["lpips"], torch.Tensor)
        self.assertGreaterEqual(float(metric_result["lpips"]), 0.0)

        # Test metadata reflects correct shape
        self.assertEqual(
            metric_result.metadata["input_shape"], [batch_size, self.channels, self.frame_height, self.frame_width]
        )
        self.assertEqual(metric_result.metadata["masked_pixels"], self.frame_height * self.frame_width)

    def test_lpips_shape_validation(self):
        """Test LPIPS shape validation for various invalid cases"""
        # Test with 2D tensor (insufficient dimensions)
        invalid_2d = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.lpips.compute(invalid_2d, invalid_2d)

        # Test with 5D tensor (too many dimensions)
        invalid_5d = torch.ones((1, 1, self.channels, self.frame_height, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.lpips.compute(invalid_5d, invalid_5d)

        # Test with mismatched shapes
        mismatch_image = torch.ones((self.channels, self.frame_height + 8, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.lpips.compute(self.white_image, mismatch_image)

        # Test with mask shape mismatch
        invalid_mask = torch.ones((self.frame_height + 8, self.frame_width), dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.lpips.compute(self.random_image, self.random_image_clone, mask=invalid_mask)

        # Test with non-boolean mask
        invalid_mask_type = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.lpips.compute(self.random_image, self.random_image_clone, mask=invalid_mask_type)

    def test_lpips_aggregate_single_value(self):
        """Test LPIPS aggregation with a single stored value"""
        # Create LPIPS metric with multiple aggregation methods
        lpips_multi = LPIPSMetric(
            net_type="alex",
            normalize=True,
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.SUM,
                AggregationMethod.MIN,
                AggregationMethod.MAX,
            ],
        )

        # Compute and store a single value
        result = lpips_multi.compute(pred=self.random_image_clone, target=self.random_image)
        lpips_multi.append(result)

        # Aggregate the single value
        aggregated = lpips_multi.aggregate()

        # Should have results for all aggregation methods
        self.assertEqual(len(aggregated), 4)
        self.assertIn(AggregationMethod.MEAN, aggregated)
        self.assertIn(AggregationMethod.SUM, aggregated)
        self.assertIn(AggregationMethod.MIN, aggregated)
        self.assertIn(AggregationMethod.MAX, aggregated)

        # For a single value, all aggregation methods should give the same result
        single_value = float(result["lpips"])
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MEAN]["lpips"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.SUM]["lpips"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MIN]["lpips"]), single_value, places=6)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MAX]["lpips"]), single_value, places=6)

    def test_lpips_aggregate_multiple_values(self):
        """Test LPIPS aggregation with multiple stored values"""
        # Create LPIPS metric with multiple aggregation methods
        lpips_multi = LPIPSMetric(
            net_type="alex",
            normalize=True,
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
            result = lpips_multi.compute(pred=pred, target=target)
            lpips_multi.append(result)
            stored_values.append(float(result["lpips"]))

        # Aggregate the values
        aggregated = lpips_multi.aggregate()

        # Test that aggregation methods work correctly
        self.assertAlmostEqual(
            float(aggregated[AggregationMethod.MEAN]["lpips"]), sum(stored_values) / len(stored_values), places=5
        )
        self.assertAlmostEqual(float(aggregated[AggregationMethod.SUM]["lpips"]), sum(stored_values), places=5)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MIN]["lpips"]), min(stored_values), places=5)
        self.assertAlmostEqual(float(aggregated[AggregationMethod.MAX]["lpips"]), max(stored_values), places=5)

        # Verify that MIN <= MEAN <= MAX
        min_val = float(aggregated[AggregationMethod.MIN]["lpips"])
        mean_val = float(aggregated[AggregationMethod.MEAN]["lpips"])
        max_val = float(aggregated[AggregationMethod.MAX]["lpips"])

        self.assertLessEqual(min_val, mean_val)
        self.assertLessEqual(mean_val, max_val)

    def test_lpips_aggregate_with_varied_frames(self):
        """Test LPIPS aggregation with mix of identical, random, and max-diff frames"""
        # Create LPIPS metric with mean aggregation
        lpips_mean = LPIPSMetric(net_type="alex", normalize=True, aggregation_methods=AggregationMethod.MEAN)

        # Store identical match (should give LPIPS close to 0)
        identical_result = lpips_mean.compute(pred=self.random_image_clone, target=self.random_image)
        lpips_mean.append(identical_result)

        # Store random vs gray comparison
        random_result = lpips_mean.compute(pred=self.random_image, target=self.gray_image)
        lpips_mean.append(random_result)

        # Store max difference frame
        max_diff_result = lpips_mean.compute(pred=self.black_image, target=self.white_image)
        lpips_mean.append(max_diff_result)

        # Aggregate
        aggregated = lpips_mean.aggregate()

        # Should have mean aggregation result
        self.assertIn(AggregationMethod.MEAN, aggregated)

        # Mean should be between min and max LPIPS values
        identical_lpips = float(identical_result["lpips"])
        random_lpips = float(random_result["lpips"])
        max_diff_lpips = float(max_diff_result["lpips"])
        mean_lpips = float(aggregated[AggregationMethod.MEAN]["lpips"])

        min_lpips = min(identical_lpips, random_lpips, max_diff_lpips)
        max_lpips = max(identical_lpips, random_lpips, max_diff_lpips)

        self.assertGreaterEqual(mean_lpips, min_lpips)
        self.assertLessEqual(mean_lpips, max_lpips)

        # Mean should be approximately the arithmetic mean
        expected_mean = (identical_lpips + random_lpips + max_diff_lpips) / 3
        self.assertAlmostEqual(mean_lpips, expected_mean, places=5)

    def test_lpips_aggregate_empty_values(self):
        """Test LPIPS aggregation with no stored values"""
        # Create LPIPS metric with multiple aggregation methods
        lpips_empty = LPIPSMetric(
            net_type="alex",
            normalize=True,
            aggregation_methods=[
                AggregationMethod.MEAN,
                AggregationMethod.SUM,
                AggregationMethod.MIN,
                AggregationMethod.MAX,
            ],
        )

        # Try to aggregate with no values stored
        aggregated = lpips_empty.aggregate()

        # Should return empty dictionary when no values are stored
        self.assertEqual(len(aggregated), 0)

    def test_lpips_aggregate_single_method(self):
        """Test LPIPS aggregation with only one aggregation method"""
        # Create LPIPS metric with only SUM aggregation
        lpips_sum = LPIPSMetric(net_type="alex", normalize=True, aggregation_methods=AggregationMethod.SUM)

        # Store multiple values using existing fixtures
        test_frames = [
            (self.random_image_clone, self.random_image),
            (self.random_image, self.gray_image),
            (self.black_image, self.white_image),
        ]
        for pred, target in test_frames:
            result = lpips_sum.compute(pred=pred, target=target)
            lpips_sum.append(result)

        # Aggregate
        aggregated = lpips_sum.aggregate()

        # Should only have SUM aggregation
        self.assertEqual(len(aggregated), 1)
        self.assertIn(AggregationMethod.SUM, aggregated)
        self.assertNotIn(AggregationMethod.MEAN, aggregated)
        self.assertNotIn(AggregationMethod.MIN, aggregated)
        self.assertNotIn(AggregationMethod.MAX, aggregated)

    def test_lpips_aggregate_clear_functionality(self):
        """Test that clear functionality works with aggregation"""
        # Create LPIPS metric
        lpips_clear = LPIPSMetric(net_type="alex", normalize=True, aggregation_methods=AggregationMethod.MEAN)

        # Store some values
        for i in range(3):
            result = lpips_clear.compute(pred=self.random_image_clone, target=self.random_image)
            lpips_clear.append(result)

        # Verify values are stored
        self.assertEqual(len(lpips_clear), 3)

        # Clear the values
        lpips_clear.clear()

        # Verify values are cleared
        self.assertEqual(len(lpips_clear), 0)

        # Aggregate should return empty dict
        aggregated = lpips_clear.aggregate()
        self.assertEqual(len(aggregated), 0)

    def test_lpips_aggregate_device_handling(self):
        """Test LPIPS aggregation with device handling"""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            # Create LPIPS metric and move to device
            lpips_device = LPIPSMetric(
                net_type="alex",
                normalize=True,
                aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.SUM],
            ).to(device)

            # Move test data to device
            random_image_device = self.random_image.to(device)
            random_image_clone_device = self.random_image_clone.to(device)

            # Store multiple values on device
            for i in range(3):
                result = lpips_device.compute(pred=random_image_clone_device, target=random_image_device)
                lpips_device.append(result)

            # Aggregate on device
            aggregated = lpips_device.aggregate()

            # Should work without errors
            self.assertIn(AggregationMethod.MEAN, aggregated)
            self.assertIn(AggregationMethod.SUM, aggregated)

            # Check that results are on the correct device
            mean_lpips = aggregated[AggregationMethod.MEAN]["lpips"]
            sum_lpips = aggregated[AggregationMethod.SUM]["lpips"]

            self.assertEqual(mean_lpips.device.type, device.type)
            self.assertEqual(sum_lpips.device.type, device.type)

    def test_lpips_aggregate_metric_result_structure(self):
        """Test that aggregated results maintain proper MetricResult structure"""
        # Create LPIPS metric with multiple aggregation methods
        lpips_structure = LPIPSMetric(
            net_type="alex",
            normalize=True,
            aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.MIN, AggregationMethod.MAX],
        )

        # Store some values
        for i in range(2):
            result = lpips_structure.compute(pred=self.random_image_clone, target=self.random_image)
            lpips_structure.append(result)

        # Aggregate
        aggregated = lpips_structure.aggregate()

        # Test structure for each aggregation method
        for method, result in aggregated.items():
            # Should be MetricResult instance
            self.assertIsInstance(result, type(lpips_structure.compute(self.random_image_clone, self.random_image)))

            # Should have 'lpips' key in values
            self.assertIn("lpips", result.values)

            # Should be a tensor
            self.assertIsInstance(result.values["lpips"], torch.Tensor)

            # Should be a scalar tensor
            self.assertEqual(result.values["lpips"].dim(), 0)

    def test_lpips_aggregate_consistency_with_compute(self):
        """Test that aggregation is consistent with individual compute results"""
        # Create LPIPS metric
        lpips_consistency = LPIPSMetric(net_type="alex", normalize=True, aggregation_methods=AggregationMethod.MEAN)

        # Create test frames with different LPIPS values
        test_cases = [
            (self.random_image_clone, self.random_image),  # Identical
            (self.random_image, self.gray_image),  # Random vs gray
            (self.black_image, self.white_image),  # Black vs white
        ]

        # Store results
        individual_results = []
        for pred, target in test_cases:
            result = lpips_consistency.compute(pred=pred, target=target)
            lpips_consistency.append(result)
            individual_results.append(float(result["lpips"]))

        # Aggregate
        aggregated = lpips_consistency.aggregate()
        mean_lpips = float(aggregated[AggregationMethod.MEAN]["lpips"])

        # Should match manual calculation
        expected_mean = sum(individual_results) / len(individual_results)
        self.assertAlmostEqual(mean_lpips, expected_mean, places=5)

        # Verify that identical frames have lowest LPIPS
        self.assertLess(individual_results[0], individual_results[1])  # Identical < Random
        self.assertLess(individual_results[0], individual_results[2])  # Identical < Black vs white

    def test_lpips_metadata(self):
        """Test that LPIPS metric has metadata."""
        lpips_metric = LPIPSMetric(net_type="alex", normalize=True)
        expected_metadata = {"net_type": "alex", "normalize": True}
        self.assertEqual(lpips_metric.metadata(), expected_metadata)

    def test_lpips_different_net_types(self):
        """Test LPIPS with different network types"""
        # Test with VGG network
        lpips_vgg = LPIPSMetric(net_type="vgg", normalize=True)
        metric_result_vgg = lpips_vgg.compute(self.random_image_clone, self.random_image)

        self.assertIsInstance(metric_result_vgg["lpips"], torch.Tensor)
        self.assertEqual(metric_result_vgg.metadata["net_type"], "vgg")

        # Test with squeeze network
        lpips_squeeze = LPIPSMetric(net_type="squeeze", normalize=True)
        metric_result_squeeze = lpips_squeeze.compute(self.random_image_clone, self.random_image)

        self.assertIsInstance(metric_result_squeeze["lpips"], torch.Tensor)
        self.assertEqual(metric_result_squeeze.metadata["net_type"], "squeeze")

    def test_lpips_normalize_option(self):
        """Test LPIPS with different normalize options"""
        # Test with normalize=True (expects [0, 1] input)
        lpips_norm = LPIPSMetric(net_type="alex", normalize=True)
        metric_result_norm = lpips_norm.compute(self.random_image_clone, self.random_image)
        self.assertEqual(metric_result_norm.metadata["normalize"], True)

        # Test with normalize=False (expects [-1, 1] input)
        lpips_no_norm = LPIPSMetric(net_type="alex", normalize=False)

        # Create data in [-1, 1] range
        random_image_neg = self.random_image * 2 - 1  # Convert [0, 1] to [-1, 1]
        random_image_clone_neg = self.random_image_clone * 2 - 1

        metric_result_no_norm = lpips_no_norm.compute(random_image_clone_neg, random_image_neg)
        self.assertEqual(metric_result_no_norm.metadata["normalize"], False)

    def test_lpips_weighted_mean_not_supported(self):
        """Test that weighted mean aggregation raises an error"""
        with self.assertRaises(ValueError):
            LPIPSMetric(net_type="alex", normalize=True, aggregation_methods=AggregationMethod.WEIGHTED_MEAN)


if __name__ == "__main__":
    unittest.main()
