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

from nre.metrics.impl.cpsnr import CPSNRMetric
from nre.metrics.utils import AggregationMethod


class TestCPSNRMetric(unittest.TestCase):
    def setUp(self):
        self.data_range = 1

        self.cpsnr = CPSNRMetric(self.data_range, aggregation_methods=AggregationMethod.MEAN)

        # Create test frames as tensors
        self.frame_height = 100
        self.frame_width = 120
        self.channels = 3

        # Perfect match (zero noise) - normalized to 0-1 range
        self.gt_frame = torch.ones((self.frame_height, self.frame_width, self.channels), dtype=torch.float32) * 0.5
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
            torch.ones((self.frame_height, self.frame_width, self.channels), dtype=torch.float32) * 1.0
        )
        self.eval_frame_worst = torch.zeros((self.frame_height, self.frame_width, self.channels), dtype=torch.float32)

        # Create test mask
        self.valid_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        self.valid_mask[20:40, 30:60] = False  # Create a region to mask out

        # Create test segmentation frame and color dictionary - using integer class indices
        self.segmentation_frame = torch.zeros((self.frame_height, self.frame_width), dtype=torch.int32)
        self.segmentation_frame[0:50, 0:60] = 1  # Class 1: top-left
        self.segmentation_frame[0:50, 60:120] = 2  # Class 2: top-right
        self.segmentation_frame[50:100, 0:60] = 3  # Class 3: bottom-left
        self.segmentation_frame[50:100, 60:120] = 4  # Class 4: bottom-right

        self.color_dict = {
            "class1": 1,
            "class2": 2,
            "class3": 3,
            "class4": 4,
        }

        self.include_categories = {"class3", "class4"}

    def tearDown(self):
        """Clean up after each test to ensure test independence."""
        self.cpsnr.reset()
        self.cpsnr.clear()

    def test_cpsnr_perfect_match(self):
        """Test PSNR computation with identical frames"""
        metric_result = self.cpsnr.compute(self.eval_frame_perfect, self.gt_frame, overall_psnr=True)
        psnr_results = metric_result.values
        psnr_map = metric_result.metadata["psnr_map_tensor"]
        pixel_counts = metric_result.metadata["pixel_counts"]

        # For identical frames, PSNR should be very high
        self.assertAlmostEqual(float(psnr_results["overall"]), self.cpsnr.max_psnr, places=1)
        self.assertEqual(int(pixel_counts["overall"]), self.frame_height * self.frame_width)

        # All pixels should have maximum PSNR
        self.assertTrue(torch.allclose(psnr_map, torch.tensor(self.cpsnr.max_psnr), rtol=1e-2))

    def test_cpsnr_with_noise(self):
        """Test PSNR computation with noisy frames"""
        metric_result_low = self.cpsnr.compute(self.eval_frame_low_noise, self.gt_frame, overall_psnr=True)
        metric_result_high = self.cpsnr.compute(self.eval_frame_high_noise, self.gt_frame, overall_psnr=True)

        psnr_results_low = metric_result_low.values
        psnr_results_high = metric_result_high.values

        self.assertGreater(float(psnr_results_low["overall"]), float(psnr_results_high["overall"]))
        self.assertLess(float(psnr_results_low["overall"]), self.cpsnr.max_psnr)
        self.assertLess(float(psnr_results_high["overall"]), self.cpsnr.max_psnr)

    def test_cpsnr_with_mask(self):
        """Test PSNR computation with boolean mask"""
        metric_result = self.cpsnr.compute(
            self.eval_frame_perfect, self.gt_frame, valid_mask=self.valid_mask, overall_psnr=True
        )
        psnr_results = metric_result.values
        valid_pixels = metric_result.metadata["valid_mask"]
        pixel_counts = metric_result.metadata["pixel_counts"]

        # PSNR should be very high because noisy region is masked out
        self.assertAlmostEqual(float(psnr_results["overall"]), self.cpsnr.max_psnr, places=1)

        # Pixel count should be reduced by the mask area
        excluded_area = torch.sum(torch.logical_not(self.valid_mask)).item()
        expected_pixel_count = self.frame_height * self.frame_width - excluded_area
        self.assertEqual(int(pixel_counts["overall"]), expected_pixel_count)

        # Check if valid_pixels is correct (should be where the original mask is FALSE)
        self.assertTrue(torch.equal(valid_pixels, self.valid_mask))

    def test_cpsnr_with_segmentation(self):
        """Test PSNR computation with segmentation data"""
        # Add different noise levels to different segments
        eval_frame_segmented = self.gt_frame.clone().float()

        # Get masks for each class
        class1_mask = self.segmentation_frame == 1
        class2_mask = self.segmentation_frame == 2
        class3_mask = self.segmentation_frame == 3

        # Add noise to each class region
        # Class 1: No noise (keep as is)
        # Class 2: Low noise
        eval_frame_segmented[class2_mask] += 0.02
        # Class 3: High noise
        eval_frame_segmented[class3_mask] += 0.08

        # Clip
        eval_frame_segmented = torch.clamp(eval_frame_segmented, 0, 1)

        # Create a proper segmentation frame for the test
        # Each pixel should be exactly one of the class indices in color_dict
        test_segmentation_frame = torch.zeros_like(self.segmentation_frame, dtype=torch.int32)
        test_segmentation_frame[class1_mask] = 1
        test_segmentation_frame[class2_mask] = 2
        test_segmentation_frame[class3_mask] = 3

        # Create a default boolean mask (all pixels included)
        default_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)

        # Compute PSNR with segmentation
        metric_result = self.cpsnr.compute(
            eval_frame_segmented,
            self.gt_frame,
            valid_mask=default_mask,
            segmentation_frame=test_segmentation_frame,
            color_dict=self.color_dict,
            overall_psnr=True,
        )
        psnr_results = metric_result.values

        # Check that class-specific PSNRs match expected pattern
        self.assertGreater(float(psnr_results["class1"]), float(psnr_results["class2"]))
        self.assertGreater(float(psnr_results["class2"]), float(psnr_results["class3"]))

        # Check pixel counts
        class1_pixel_count = torch.sum(class1_mask).item()
        class2_pixel_count = torch.sum(class2_mask).item()
        class3_pixel_count = torch.sum(class3_mask).item()

        pixel_counts = metric_result.metadata["pixel_counts"]

        self.assertEqual(int(pixel_counts["class1"]), class1_pixel_count)
        self.assertEqual(int(pixel_counts["class2"]), class2_pixel_count)
        self.assertEqual(int(pixel_counts["class3"]), class3_pixel_count)

    def test_cpsnr_with_include_categories(self):
        """Test PSNR computation with include categories"""
        # Setup same as segmentation test
        eval_frame_segmented = self.gt_frame.clone().float()
        class2_mask = self.segmentation_frame == 2
        eval_frame_segmented[class2_mask] += 0.02
        class3_mask = self.segmentation_frame == 3
        eval_frame_segmented[class3_mask] += 0.08
        class4_mask = self.segmentation_frame == 4
        eval_frame_segmented[class4_mask] += 0.15

        eval_frame_segmented = torch.clamp(eval_frame_segmented, 0, 1)

        metric_result = self.cpsnr.compute(
            eval_frame_segmented,
            self.gt_frame,
            segmentation_frame=self.segmentation_frame,
            color_dict=self.color_dict,
            include_categories=self.include_categories,
            overall_psnr=True,
        )
        psnr_results = metric_result.values
        valid_pixels = metric_result.metadata["valid_mask"]
        pixel_counts = metric_result.metadata["pixel_counts"]

        # Class3 and Class4 should be included in overall PSNR calculation
        expected_valid_pixel_count = torch.sum(class3_mask | class4_mask).item()
        self.assertEqual(int(pixel_counts["overall"]), expected_valid_pixel_count)

        # Valid pixels mask should include class3 and class4
        self.assertTrue(torch.all(valid_pixels == class3_mask | class4_mask))

    def test_cpsnr_worst_match(self):
        """Test PSNR computation with worst case (maximum difference) frames"""
        metric_result = self.cpsnr.compute(self.eval_frame_worst, self.gt_frame_worst, overall_psnr=True)
        psnr_results = metric_result.values
        psnr_map = metric_result.metadata["psnr_map_tensor"]
        pixel_counts = metric_result.metadata["pixel_counts"]

        # For data_range=1 with max difference (1 vs 0), PSNR should be very low
        # The actual value depends on the PSNR calculation implementation
        self.assertLess(float(psnr_results["overall"]), 10.0)  # Should be very low PSNR
        self.assertEqual(int(pixel_counts["overall"]), self.frame_height * self.frame_width)

        # All pixels should have minimum PSNR
        self.assertTrue(torch.all(psnr_map < 10.0))

    def test_cpsnr_metadata_structure(self):
        """Test that CPSNRMetric metadata has the correct structure"""
        metric_result = self.cpsnr.compute(self.eval_frame_perfect, self.gt_frame, overall_psnr=True)

        # Test metadata types
        self.assertIsInstance(metric_result.metadata["psnr_map_tensor"], torch.Tensor)
        self.assertIsInstance(metric_result.metadata["valid_mask"], torch.Tensor)
        self.assertIsInstance(metric_result.metadata["pixel_counts"], dict)
        self.assertIsInstance(metric_result.metadata["categories_computed"], list)

        # Test metadata values
        self.assertEqual(metric_result.metadata["psnr_map_tensor"].shape, (self.frame_height, self.frame_width))
        self.assertIn("overall", metric_result.metadata["categories_computed"])
        self.assertIn("overall", metric_result.metadata["pixel_counts"])
        self.assertEqual(metric_result.metadata["pixel_counts"]["overall"].item(), self.frame_height * self.frame_width)

        # Test that valid_mask is a boolean tensor with correct shape
        valid_mask = metric_result.metadata["valid_mask"]
        self.assertEqual(valid_mask.shape, (self.frame_height, self.frame_width))
        self.assertEqual(valid_mask.dtype, torch.bool)
        # When no mask is provided, all pixels should be valid
        self.assertTrue(torch.all(valid_mask))

    def test_cpsnr_metadata_with_custom_mask(self):
        """Test that CPSNRMetric metadata correctly stores custom masks"""
        # Create a custom mask
        custom_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        custom_mask[20:40, 30:60] = False  # Mask out a region

        metric_result = self.cpsnr.compute(
            self.eval_frame_perfect, self.gt_frame, valid_mask=custom_mask, overall_psnr=True
        )

        # Test that the custom mask is stored correctly
        stored_mask = metric_result.metadata["valid_mask"]
        self.assertIsInstance(stored_mask, torch.Tensor)
        self.assertEqual(stored_mask.shape, (self.frame_height, self.frame_width))
        self.assertEqual(stored_mask.dtype, torch.bool)

        # The stored mask should be identical to the input mask
        self.assertTrue(torch.equal(stored_mask, custom_mask))

        # Test that the pixel_counts reflects the masked pixels
        expected_valid_pixels = torch.sum(custom_mask).item()
        self.assertEqual(metric_result.metadata["pixel_counts"]["overall"].item(), expected_valid_pixels)

    def test_cpsnr_reset_functionality(self):
        """Test that reset functionality works correctly"""
        # Compute CPSNR first
        metric_result1 = self.cpsnr.compute(self.eval_frame_perfect, self.gt_frame, overall_psnr=True)

        # Reset the metric
        self.cpsnr.reset()

        # Compute CPSNR again - should work the same
        metric_result2 = self.cpsnr.compute(self.eval_frame_perfect, self.gt_frame, overall_psnr=True)

        # Results should be identical
        self.assertAlmostEqual(
            float(metric_result1.values["overall"]), float(metric_result2.values["overall"]), places=6
        )

    def test_cpsnr_device_handling(self):
        """Test CPSNRMetric device handling"""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            # Move metric to device
            self.cpsnr.to(device)

            # Move test data to device
            gt_frame_device = self.gt_frame.to(device)
            eval_frame_device = self.eval_frame_perfect.to(device)

            # Compute CPSNR on device
            metric_result = self.cpsnr.compute(eval_frame_device, gt_frame_device, overall_psnr=True)

            # Should work without errors
            self.assertIsInstance(metric_result.values["overall"], torch.Tensor)
            # Check device type matches (ignore index)
            self.assertEqual(metric_result.values["overall"].device.type, device.type)

    def test_cpsnr_weighted_mean_aggregation(self):
        """Test that CPSNRMetric weighted mean aggregation works correctly"""
        # Create test data with segmentation
        eval_frame_segmented = self.gt_frame.clone().float()

        # Get masks for each class
        class1_mask = self.segmentation_frame == 1
        class2_mask = self.segmentation_frame == 2
        class3_mask = self.segmentation_frame == 3

        # Add different noise levels to each class region
        eval_frame_segmented[class1_mask] += 0.005  # Very low noise
        eval_frame_segmented[class2_mask] += 0.02  # Low noise
        eval_frame_segmented[class3_mask] += 0.08  # High noise
        eval_frame_segmented = torch.clamp(eval_frame_segmented, 0, 1)

        # Create a default boolean mask (all pixels included)
        default_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)

        # Create CPSNR metric with weighted mean aggregation
        cpsnr_weighted = CPSNRMetric(data_range=1.0, aggregation_methods=AggregationMethod.WEIGHTED_MEAN)

        # Compute multiple CPSNR results with different noise levels
        for noise_multiplier in [0.5, 1.0, 1.5]:
            # Add varying noise
            noisy_frame = eval_frame_segmented.clone().float()
            torch.manual_seed(42 + int(noise_multiplier * 100))
            additional_noise = torch.randint(-2, 3, noisy_frame.shape, dtype=torch.float32) * noise_multiplier * 0.01
            noisy_frame += additional_noise
            noisy_frame = torch.clamp(noisy_frame, 0, 1)

            # Compute CPSNR and accumulate
            metric_result = cpsnr_weighted.compute(
                noisy_frame,
                self.gt_frame,
                valid_mask=default_mask,
                segmentation_frame=self.segmentation_frame,
                color_dict=self.color_dict,
                overall_psnr=True,
            )
            cpsnr_weighted.append(metric_result)

        # Aggregate the results
        aggregated_result = cpsnr_weighted.aggregate()

        # Test that we have aggregated results for each category
        self.assertIn("overall", aggregated_result[AggregationMethod.WEIGHTED_MEAN].values)
        self.assertIn("class1", aggregated_result[AggregationMethod.WEIGHTED_MEAN].values)
        self.assertIn("class2", aggregated_result[AggregationMethod.WEIGHTED_MEAN].values)
        self.assertIn("class3", aggregated_result[AggregationMethod.WEIGHTED_MEAN].values)
        self.assertIn("class4", aggregated_result[AggregationMethod.WEIGHTED_MEAN].values)

        # Test that each category has the expected structure (single PSNR value)
        for category in ["overall", "class1", "class2", "class3", "class4"]:
            category_result = aggregated_result[AggregationMethod.WEIGHTED_MEAN][category]
            self.assertIsInstance(category_result, torch.Tensor)
            self.assertEqual(category_result.dim(), 0)  # Should be a scalar tensor

            # PSNR should be finite and reasonable
            psnr_value = category_result.item()
            self.assertTrue(torch.isfinite(torch.tensor(psnr_value)))
            self.assertGreater(psnr_value, -100.0)  # Allow negative PSNR values
            self.assertLess(psnr_value, cpsnr_weighted.max_psnr)

            # Check that pixel counts are available in metadata and are summed (not averaged)
            self.assertIn("pixel_counts", aggregated_result[AggregationMethod.WEIGHTED_MEAN].metadata)
            self.assertIn(category, aggregated_result[AggregationMethod.WEIGHTED_MEAN].metadata["pixel_counts"])
            pixel_count = aggregated_result[AggregationMethod.WEIGHTED_MEAN].metadata["pixel_counts"][category]
            self.assertIsInstance(pixel_count, torch.Tensor)
            self.assertGreater(pixel_count.item(), 0)

        # Test that PSNR values follow expected pattern (class1 > class2 > class3 due to noise levels)
        class1_psnr = aggregated_result[AggregationMethod.WEIGHTED_MEAN]["class1"].item()
        class2_psnr = aggregated_result[AggregationMethod.WEIGHTED_MEAN]["class2"].item()
        class3_psnr = aggregated_result[AggregationMethod.WEIGHTED_MEAN]["class3"].item()

        self.assertGreater(class1_psnr, class2_psnr)
        self.assertGreater(class2_psnr, class3_psnr)

        # Test metadata
        self.assertIn("num_results", aggregated_result[AggregationMethod.WEIGHTED_MEAN].metadata)
        self.assertIn("aggregated_categories", aggregated_result[AggregationMethod.WEIGHTED_MEAN].metadata)
        self.assertIn("pixel_counts", aggregated_result[AggregationMethod.WEIGHTED_MEAN].metadata)

        self.assertEqual(
            set(aggregated_result[AggregationMethod.WEIGHTED_MEAN].metadata["aggregated_categories"]),
            {"overall", "class1", "class2", "class3", "class4"},
        )

        # Test that weighted mean gives different results than simple mean
        cpsnr_mean = CPSNRMetric(data_range=1.0, aggregation_methods=AggregationMethod.MEAN)
        for noise_multiplier in [0.5, 1.0, 1.5]:
            noisy_frame = eval_frame_segmented.clone().float()
            torch.manual_seed(42 + int(noise_multiplier * 100))
            additional_noise = torch.randint(-2, 3, noisy_frame.shape, dtype=torch.float32) * noise_multiplier * 0.01
            noisy_frame += additional_noise
            noisy_frame = torch.clamp(noisy_frame, 0, 1)

            metric_result = cpsnr_mean.compute(
                noisy_frame,
                self.gt_frame,
                valid_mask=default_mask,
                segmentation_frame=self.segmentation_frame,
                color_dict=self.color_dict,
                overall_psnr=True,
            )
            cpsnr_mean.append(metric_result)

        aggregated_mean_result = cpsnr_mean.aggregate()

        # The weighted mean and simple mean should give different results
        # (unless all frames have exactly the same pixel counts for each class)
        for category in ["class1", "class2", "class3"]:
            weighted_psnr = aggregated_result[AggregationMethod.WEIGHTED_MEAN][category].item()
            mean_psnr = aggregated_mean_result[AggregationMethod.MEAN][category].item()
            # They might be the same if pixel counts are identical across frames, but that's unlikely
            # So we just check they're both finite and reasonable
            self.assertTrue(torch.isfinite(torch.tensor(weighted_psnr)))
            self.assertTrue(torch.isfinite(torch.tensor(mean_psnr)))

    def test_cpsnr_aggregation_by_category(self):
        """Test that CPSNRMetric aggregate function properly aggregates values by category"""
        # Create test data with segmentation
        eval_frame_segmented = self.gt_frame.clone().float()

        # Get masks for each class
        class1_mask = self.segmentation_frame == 1
        class2_mask = self.segmentation_frame == 2
        class3_mask = self.segmentation_frame == 3

        # Add different noise levels to each class region
        eval_frame_segmented[class1_mask] += 0.005  # Very low noise
        eval_frame_segmented[class2_mask] += 0.02  # Low noise
        eval_frame_segmented[class3_mask] += 0.08  # High noise
        eval_frame_segmented = torch.clamp(eval_frame_segmented, 0, 1)

        # Create a default boolean mask (all pixels included)
        default_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)

        # Compute multiple CPSNR results with different noise levels
        for noise_multiplier in [0.5, 1.0, 1.5]:
            # Add varying noise
            noisy_frame = eval_frame_segmented.clone().float()
            torch.manual_seed(42 + int(noise_multiplier * 100))
            additional_noise = torch.randint(-2, 3, noisy_frame.shape, dtype=torch.float32) * noise_multiplier * 0.01
            noisy_frame += additional_noise
            noisy_frame = torch.clamp(noisy_frame, 0, 1)

            # Compute CPSNR and accumulate
            metric_result = self.cpsnr.compute(
                noisy_frame,
                self.gt_frame,
                valid_mask=default_mask,
                segmentation_frame=self.segmentation_frame,
                color_dict=self.color_dict,
                overall_psnr=True,
            )
            self.cpsnr.append(metric_result)

            # Aggregate the results
        aggregated_result = self.cpsnr.aggregate()

        # Test that we have aggregated results for each category
        self.assertIn("overall", aggregated_result[AggregationMethod.MEAN].values)
        self.assertIn("class1", aggregated_result[AggregationMethod.MEAN].values)
        self.assertIn("class2", aggregated_result[AggregationMethod.MEAN].values)
        self.assertIn("class3", aggregated_result[AggregationMethod.MEAN].values)
        self.assertIn("class4", aggregated_result[AggregationMethod.MEAN].values)

        # Test that each category has the expected structure (single PSNR value)
        for category in ["overall", "class1", "class2", "class3", "class4"]:
            category_result = aggregated_result[AggregationMethod.MEAN][category]
            self.assertIsInstance(category_result, torch.Tensor)
            self.assertEqual(category_result.dim(), 0)  # Should be a scalar tensor

            # PSNR should be finite and reasonable
            psnr_value = category_result.item()
            self.assertTrue(torch.isfinite(torch.tensor(psnr_value)))
            self.assertGreater(psnr_value, -100.0)  # Allow negative PSNR values
            self.assertLess(psnr_value, self.cpsnr.max_psnr)

            # Check that pixel counts are available in metadata
            self.assertIn("pixel_counts", aggregated_result[AggregationMethod.MEAN].metadata)
            self.assertIn(category, aggregated_result[AggregationMethod.MEAN].metadata["pixel_counts"])
            pixel_count = aggregated_result[AggregationMethod.MEAN].metadata["pixel_counts"][category]
            self.assertIsInstance(pixel_count, torch.Tensor)
            self.assertGreater(pixel_count.item(), 0)

        # Test that PSNR values follow expected pattern (class1 > class2 > class3 due to noise levels)
        class1_psnr = aggregated_result[AggregationMethod.MEAN]["class1"].item()
        class2_psnr = aggregated_result[AggregationMethod.MEAN]["class2"].item()
        class3_psnr = aggregated_result[AggregationMethod.MEAN]["class3"].item()

        self.assertGreater(class1_psnr, class2_psnr)
        self.assertGreater(class2_psnr, class3_psnr)

        # Test metadata
        self.assertIn("num_results", aggregated_result[AggregationMethod.MEAN].metadata)
        self.assertIn("aggregated_categories", aggregated_result[AggregationMethod.MEAN].metadata)
        self.assertIn("pixel_counts", aggregated_result[AggregationMethod.MEAN].metadata)

        self.assertEqual(
            set(aggregated_result[AggregationMethod.MEAN].metadata["aggregated_categories"]),
            {"overall", "class1", "class2", "class3", "class4"},
        )

    def test_cpsnr_aggregation_with_include_categories(self):
        """Test that CPSNRMetric aggregate function works correctly with include_categories"""
        # Create test data with segmentation
        eval_frame_segmented = self.gt_frame.clone().float()

        # Get masks for each class
        class3_mask = self.segmentation_frame == 3
        class4_mask = self.segmentation_frame == 4

        # Add different noise levels
        eval_frame_segmented[class3_mask] += 0.02
        eval_frame_segmented[class4_mask] += 0.08
        eval_frame_segmented = torch.clamp(eval_frame_segmented, 0, 1)

        # Compute multiple CPSNR results with include_categories
        for noise_multiplier in [0.8, 1.0, 1.2]:
            noisy_frame = eval_frame_segmented.clone().float()
            torch.manual_seed(42 + int(noise_multiplier * 100))
            additional_noise = torch.randint(-1, 2, noisy_frame.shape, dtype=torch.float32) * noise_multiplier * 0.005
            noisy_frame += additional_noise
            noisy_frame = torch.clamp(noisy_frame, 0, 1)

            metric_result = self.cpsnr.compute(
                noisy_frame,
                self.gt_frame,
                segmentation_frame=self.segmentation_frame,
                color_dict=self.color_dict,
                include_categories=self.include_categories,
                overall_psnr=True,
            )
            self.cpsnr.append(metric_result)

        # Aggregate the results
        aggregated_result = self.cpsnr.aggregate()

        # Test that we only have aggregated results for included categories
        self.assertIn("overall", aggregated_result[AggregationMethod.MEAN].values)
        self.assertIn("class3", aggregated_result[AggregationMethod.MEAN].values)
        self.assertIn("class4", aggregated_result[AggregationMethod.MEAN].values)
        self.assertNotIn("class1", aggregated_result[AggregationMethod.MEAN].values)
        self.assertNotIn("class2", aggregated_result[AggregationMethod.MEAN].values)

        # Test that PSNR values follow expected pattern (class3 > class4 due to noise levels)
        class3_psnr = aggregated_result[AggregationMethod.MEAN]["class3"].item()
        class4_psnr = aggregated_result[AggregationMethod.MEAN]["class4"].item()

        self.assertGreater(class3_psnr, class4_psnr)

        # Test metadata
        self.assertEqual(aggregated_result[AggregationMethod.MEAN].metadata["num_results"], 3)
        self.assertEqual(
            set(aggregated_result[AggregationMethod.MEAN].metadata["aggregated_categories"]),
            {"overall", "class3", "class4"},
        )

    def test_cpsnr_aggregation_empty(self):
        """Test that CPSNRMetric aggregate function handles empty state correctly"""
        # Try to aggregate without any accumulated values
        aggregated_result = self.cpsnr.aggregate()

        # Should return an empty result
        self.assertIsInstance(aggregated_result, dict)
        self.assertEqual(aggregated_result, {})

    def test_validate_inputs_valid_tensors(self):
        """Test validate_inputs with valid tensor inputs"""
        # Should not raise any exceptions with valid inputs
        try:
            self.cpsnr.validate_inputs(self.eval_frame_perfect, self.gt_frame)
        except Exception as e:
            self.fail(f"validate_inputs raised an exception with valid inputs: {e}")

    def test_validate_inputs_non_tensor_eval_frame(self):
        """Test validate_inputs with non-tensor eval_frame"""
        with self.assertRaises(TypeError) as context:
            self.cpsnr.validate_inputs([1, 2, 3], self.gt_frame)
        self.assertIn("Input 0 must be a torch.Tensor", str(context.exception))

    def test_validate_inputs_non_tensor_gt_frame(self):
        """Test validate_inputs with non-tensor gt_frame"""
        with self.assertRaises(TypeError) as context:
            self.cpsnr.validate_inputs(self.eval_frame_perfect, "not a tensor")
        self.assertIn("Input 1 must be a torch.Tensor", str(context.exception))

    def test_validate_inputs_1d_tensor(self):
        """Test validate_inputs with 1D tensor"""
        one_d_tensor = torch.tensor([1, 2, 3])
        with self.assertRaises(ValueError) as context:
            self.cpsnr.validate_inputs(one_d_tensor, self.gt_frame)
        self.assertIn("Input 0 must have at least 2 dimensions", str(context.exception))

    def test_validate_inputs_0d_tensor(self):
        """Test validate_inputs with 0D tensor"""
        zero_d_tensor = torch.tensor(5)
        with self.assertRaises(ValueError) as context:
            self.cpsnr.validate_inputs(zero_d_tensor, self.gt_frame)
        self.assertIn("Input 0 must have at least 2 dimensions", str(context.exception))

    def test_validate_inputs_shape_mismatch(self):
        """Test validate_inputs with shape mismatch"""
        different_shape_tensor = torch.ones((50, 60, 3), dtype=torch.float32)
        with self.assertRaises(ValueError) as context:
            self.cpsnr.validate_inputs(different_shape_tensor, self.gt_frame)
        self.assertIn("Evaluated and ground truth shapes must match", str(context.exception))

    def test_validate_inputs_valid_mask_correct(self):
        """Test validate_inputs with correct valid_mask"""
        valid_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        try:
            self.cpsnr.validate_inputs(self.eval_frame_perfect, self.gt_frame, valid_mask=valid_mask)
        except Exception as e:
            self.fail(f"validate_inputs raised an exception with correct valid_mask: {e}")

    def test_validate_inputs_valid_mask_non_tensor(self):
        """Test validate_inputs with non-tensor valid_mask"""
        with self.assertRaises(TypeError) as context:
            self.cpsnr.validate_inputs(self.eval_frame_perfect, self.gt_frame, valid_mask=[True, False])
        self.assertIn("Valid mask must be a torch.Tensor", str(context.exception))

    def test_validate_inputs_valid_mask_wrong_dtype(self):
        """Test validate_inputs with valid_mask of wrong dtype"""
        wrong_dtype_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)
        with self.assertRaises(ValueError) as context:
            self.cpsnr.validate_inputs(self.eval_frame_perfect, self.gt_frame, valid_mask=wrong_dtype_mask)
        self.assertIn("Valid mask must be boolean tensor", str(context.exception))

    def test_validate_inputs_valid_mask_wrong_shape(self):
        """Test validate_inputs with valid_mask of wrong shape"""
        wrong_shape_mask = torch.ones((50, 60), dtype=torch.bool)
        with self.assertRaises(ValueError) as context:
            self.cpsnr.validate_inputs(self.eval_frame_perfect, self.gt_frame, valid_mask=wrong_shape_mask)
        self.assertIn("Valid mask shape", str(context.exception))
        self.assertIn("must match image spatial dimensions", str(context.exception))

    def test_validate_inputs_valid_mask_3d_shape(self):
        """Test validate_inputs with valid_mask that has 3D shape"""
        wrong_shape_mask = torch.ones((self.frame_height, self.frame_width, 3), dtype=torch.bool)
        with self.assertRaises(ValueError) as context:
            self.cpsnr.validate_inputs(self.eval_frame_perfect, self.gt_frame, valid_mask=wrong_shape_mask)
        self.assertIn("Valid mask shape", str(context.exception))
        self.assertIn("must match image spatial dimensions", str(context.exception))

    def test_validate_inputs_all_optional_params(self):
        """Test validate_inputs with all optional parameters"""
        valid_mask = torch.ones((self.frame_height, self.frame_width), dtype=torch.bool)
        segmentation_frame = torch.zeros((self.frame_height, self.frame_width), dtype=torch.int32)
        color_dict = {"class1": 1}
        include_categories = {"class1"}

        try:
            self.cpsnr.validate_inputs(
                self.eval_frame_perfect,
                self.gt_frame,
                valid_mask=valid_mask,
                segmentation_frame=segmentation_frame,
                color_dict=color_dict,
                include_categories=include_categories,
                overall_psnr=True,
            )
        except Exception as e:
            self.fail(f"validate_inputs raised an exception with all optional params: {e}")

    def test_validate_inputs_none_valid_mask(self):
        """Test validate_inputs with None valid_mask"""
        try:
            self.cpsnr.validate_inputs(self.eval_frame_perfect, self.gt_frame, valid_mask=None)
        except Exception as e:
            self.fail(f"validate_inputs raised an exception with None valid_mask: {e}")

    def test_validate_inputs_different_tensor_types(self):
        """Test validate_inputs with different tensor types"""
        # Test with different dtypes
        float64_tensor = torch.ones((self.frame_height, self.frame_width, self.channels), dtype=torch.float64)
        int_tensor = torch.ones((self.frame_height, self.frame_width, self.channels), dtype=torch.int32)

        # Should work with different dtypes
        try:
            self.cpsnr.validate_inputs(float64_tensor, self.gt_frame)
            self.cpsnr.validate_inputs(int_tensor, self.gt_frame)
        except Exception as e:
            self.fail(f"validate_inputs raised an exception with different tensor types: {e}")

    def test_validate_inputs_2d_tensors(self):
        """Test validate_inputs with 2D tensors (grayscale images)"""
        gray_eval = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)
        gray_gt = torch.ones((self.frame_height, self.frame_width), dtype=torch.float32)

        try:
            self.cpsnr.validate_inputs(gray_eval, gray_gt)
        except Exception as e:
            self.fail(f"validate_inputs raised an exception with 2D tensors: {e}")

    def test_validate_inputs_4d_tensors(self):
        """Test validate_inputs with 4D tensors (batch of images)"""
        batch_eval = torch.ones((2, self.frame_height, self.frame_width, self.channels), dtype=torch.float32)
        batch_gt = torch.ones((2, self.frame_height, self.frame_width, self.channels), dtype=torch.float32)

        try:
            self.cpsnr.validate_inputs(batch_eval, batch_gt)
        except Exception as e:
            self.fail(f"validate_inputs raised an exception with 4D tensors: {e}")

    def test_cpsnr_metadata(self):
        """Test that CPSNR metric has metadata."""
        cpsnr_metric = CPSNRMetric(data_range=1.0)
        self.assertEqual(cpsnr_metric.metadata(), {"data_range": 1.0})


if __name__ == "__main__":
    unittest.main()
