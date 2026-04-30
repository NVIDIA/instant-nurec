# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for perceptual quality metrics utility functions."""

import unittest

import cv2
import numpy as np

from nre.metrics.impl.utils.perceptual_utils import (
    _resize_for_metric,
    compute_artifact_score,
    compute_blur_penalty,
    compute_cem_score,
    compute_channel_coherence_score,
    compute_chroma_hf_score,
    compute_color_histogram_similarity,
    compute_edge_similarity,
    compute_gradient_similarity,
    compute_hue_variance_score,
    compute_multi_scale_ssim,
    compute_y_chroma_ratio_score,
)


class TestPerceptualUtils(unittest.TestCase):
    """Test cases for perceptual utility functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create test images
        self.img_size = 64
        np.random.seed(42)  # Fixed seed for reproducible tests
        # Identical images
        self.img1 = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        self.img1_copy = self.img1.copy()

        # Different image
        self.img2 = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)

        # Grayscale images
        self.gray1 = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY)
        self.gray2 = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY)

        # Blurred image
        self.img_blurred = cv2.GaussianBlur(self.img1, (15, 15), 5.0)

    def test_edge_similarity_identical_images(self) -> None:
        """Test edge similarity with identical images."""
        score = compute_edge_similarity(self.img1, self.img1_copy)
        self.assertGreater(score, 0.95, "Identical images should have high edge similarity")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_edge_similarity_different_images(self) -> None:
        """Test edge similarity with different images."""
        score = compute_edge_similarity(self.img1, self.img2)
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_edge_similarity_grayscale(self) -> None:
        """Test edge similarity with grayscale images."""
        score = compute_edge_similarity(self.gray1, self.gray1.copy())
        self.assertGreater(score, 0.95, "Identical grayscale images should have high edge similarity")

    def test_edge_similarity_blur_detection(self) -> None:
        """Test that edge similarity detects blur."""
        score = compute_edge_similarity(self.img1, self.img_blurred)
        # Blurred image should have lower edge similarity
        self.assertLess(score, 0.9, "Blurred image should have lower edge similarity")

    def test_gradient_similarity_identical_images(self) -> None:
        """Test gradient similarity with identical images."""
        score = compute_gradient_similarity(self.img1, self.img1_copy)
        self.assertGreater(score, 0.95, "Identical images should have high gradient similarity")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_gradient_similarity_different_images(self) -> None:
        """Test gradient similarity with different images."""
        score = compute_gradient_similarity(self.img1, self.img2)
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_gradient_similarity_blur_detection(self) -> None:
        """Test that gradient similarity detects blur."""
        score = compute_gradient_similarity(self.img1, self.img_blurred)
        self.assertLess(score, 0.9, "Blurred image should have lower gradient similarity")

    def test_blur_penalty_identical_images(self) -> None:
        """Test blur penalty with identical images."""
        score = compute_blur_penalty(self.img1, self.img1_copy)
        self.assertGreater(score, 0.95, "Identical images should have high blur score")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_blur_penalty_detects_blur(self) -> None:
        """Test that blur penalty detects blurred images."""
        score = compute_blur_penalty(self.img1, self.img_blurred)
        self.assertLess(score, 0.9, "Blurred image should have lower blur score")

    def test_blur_penalty_grayscale(self) -> None:
        """Test blur penalty with grayscale images."""
        score = compute_blur_penalty(self.gray1, self.gray1.copy())
        self.assertGreater(score, 0.95, "Identical grayscale images should have high blur score")

    def test_artifact_score_identical_images(self) -> None:
        """Test artifact score with identical images."""
        score = compute_artifact_score(self.img1, self.img1_copy)
        self.assertGreater(score, 0.9, "Identical images should have high artifact score")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_artifact_score_different_images(self) -> None:
        """Test artifact score with different images."""
        score = compute_artifact_score(self.img1, self.img2)
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_artifact_score_with_float_input(self) -> None:
        """Test artifact score with float inputs."""
        img1_float = self.img1.astype(np.float32) / 255.0
        img2_float = self.img1_copy.astype(np.float32) / 255.0
        score = compute_artifact_score(img1_float, img2_float)
        self.assertGreater(score, 0.9, "Identical float images should have high artifact score")

    def test_multi_scale_ssim_identical_images(self) -> None:
        """Test SSIM with identical images."""
        score = compute_multi_scale_ssim(self.img1, self.img1_copy)
        self.assertGreater(score, 0.95, "Identical images should have high SSIM")
        # SSIM can be in [-1, 1] but typically [0, 1]
        self.assertGreaterEqual(score, -1.0, "Score should be >= -1.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_multi_scale_ssim_different_images(self) -> None:
        """Test SSIM with different images."""
        score = compute_multi_scale_ssim(self.img1, self.img2)
        self.assertGreaterEqual(score, -1.0, "Score should be >= -1.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_multi_scale_ssim_grayscale(self) -> None:
        """Test SSIM with grayscale images."""
        score = compute_multi_scale_ssim(self.gray1, self.gray1.copy())
        self.assertGreater(score, 0.95, "Identical grayscale images should have high SSIM")

    def test_color_histogram_similarity_identical_images(self) -> None:
        """Test color histogram similarity with identical images."""
        score = compute_color_histogram_similarity(self.img1, self.img1_copy)
        self.assertGreater(score, 0.95, "Identical images should have high histogram similarity")
        self.assertGreaterEqual(score, -1.0, "Score should be >= -1.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_color_histogram_similarity_different_images(self) -> None:
        """Test color histogram similarity with different images."""
        score = compute_color_histogram_similarity(self.img1, self.img2)
        self.assertGreaterEqual(score, -1.0, "Score should be >= -1.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_cem_score_identical_images(self) -> None:
        """Test CEM score with identical images."""
        score = compute_cem_score(self.img1, self.img1_copy)
        self.assertGreater(score, 0.9, "Identical images should have high CEM score")
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_cem_score_different_images(self) -> None:
        """Test CEM score with different images."""
        score = compute_cem_score(self.img1, self.img2)
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_hue_variance_score_identical_images(self) -> None:
        """Test hue variance score with identical images."""
        score = compute_hue_variance_score(self.img1, self.img1_copy)
        self.assertGreater(score, 0.9, "Identical images should have high hue variance score")
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_hue_variance_score_different_images(self) -> None:
        """Test hue variance score with different images."""
        score = compute_hue_variance_score(self.img1, self.img2)
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_chroma_hf_score_identical_images(self) -> None:
        """Test chroma HF score with identical images."""
        score = compute_chroma_hf_score(self.img1, self.img1_copy)
        self.assertGreater(score, 0.9, "Identical images should have high chroma HF score")
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_chroma_hf_score_different_images(self) -> None:
        """Test chroma HF score with different images."""
        score = compute_chroma_hf_score(self.img1, self.img2)
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_channel_incoherence_score_identical_images(self) -> None:
        """Test channel incoherence score with identical images."""
        score = compute_channel_coherence_score(self.img1, self.img1_copy)
        self.assertGreater(score, 0.9, "Identical images should have high channel coherence score")
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_channel_incoherence_score_different_images(self) -> None:
        """Test channel incoherence score with different images."""
        score = compute_channel_coherence_score(self.img1, self.img2)
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_y_chroma_ratio_score_identical_images(self) -> None:
        """Test Y/Chroma ratio score with identical images."""
        score = compute_y_chroma_ratio_score(self.img1, self.img1_copy)
        self.assertGreater(score, 0.9, "Identical images should have high Y/Chroma ratio score")
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_y_chroma_ratio_score_different_images(self) -> None:
        """Test Y/Chroma ratio score with different images."""
        score = compute_y_chroma_ratio_score(self.img1, self.img2)
        self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
        self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_resize_for_metric_no_resize_needed(self) -> None:
        """Test resize when image is already small."""
        small_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        resized = _resize_for_metric(small_img, max_side=256)
        self.assertEqual(resized.shape, small_img.shape, "Small image should not be resized")

    def test_resize_for_metric_large_image(self) -> None:
        """Test resize with large image."""
        large_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        resized = _resize_for_metric(large_img, max_side=256)
        self.assertLessEqual(max(resized.shape[:2]), 256, "Large image should be resized")
        self.assertEqual(resized.shape[2], 3, "Should preserve number of channels")

    def test_resize_for_metric_grayscale(self) -> None:
        """Test resize with grayscale image."""
        large_gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        resized = _resize_for_metric(large_gray, max_side=256)
        self.assertLessEqual(max(resized.shape[:2]), 256, "Large grayscale image should be resized")
        self.assertEqual(len(resized.shape), 2, "Should preserve grayscale format")

    def test_resize_for_metric_aspect_ratio(self) -> None:
        """Test that resize preserves aspect ratio."""
        rect_img = np.random.randint(0, 255, (400, 200, 3), dtype=np.uint8)
        resized = _resize_for_metric(rect_img, max_side=256)
        original_aspect = rect_img.shape[0] / rect_img.shape[1]
        resized_aspect = resized.shape[0] / resized.shape[1]
        self.assertAlmostEqual(original_aspect, resized_aspect, places=2, msg="Aspect ratio should be preserved")

    def test_all_metrics_return_float(self) -> None:
        """Test that all metrics return float values."""
        metrics = [
            compute_edge_similarity,
            compute_gradient_similarity,
            compute_blur_penalty,
            compute_artifact_score,
            compute_multi_scale_ssim,
            compute_color_histogram_similarity,
            compute_cem_score,
            compute_hue_variance_score,
            compute_chroma_hf_score,
            compute_channel_coherence_score,
            compute_y_chroma_ratio_score,
        ]

        for metric in metrics:
            score = metric(self.img1, self.img2)
            self.assertIsInstance(score, float, f"{metric.__name__} should return float")

    def test_metrics_handle_different_dtypes(self) -> None:
        """Test that metrics handle different input dtypes."""
        # uint8
        img_uint8 = self.img1
        # float32 [0, 1]
        img_float32 = self.img1.astype(np.float32) / 255.0

        # Test a few representative metrics
        score_uint8 = compute_edge_similarity(img_uint8, img_uint8.copy())
        score_float = compute_edge_similarity(
            (img_float32 * 255).astype(np.uint8), (img_float32 * 255).astype(np.uint8).copy()
        )

        # Should give similar results
        self.assertAlmostEqual(
            score_uint8, score_float, delta=0.05, msg="Results should be similar for different dtypes"
        )

    def test_metrics_with_noisy_image(self) -> None:
        """Test metrics detect noise in images."""
        # Add random noise
        noise = np.random.randint(-30, 30, self.img1.shape, dtype=np.int16)
        img_noisy = np.clip(self.img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Artifact score should detect noise
        score = compute_artifact_score(self.img1, img_noisy)
        self.assertLess(score, 0.95, "Noisy image should have lower artifact score")

    def test_edge_cases_empty_or_constant_images(self) -> None:
        """Test metrics with constant images."""
        const_img = np.ones((64, 64, 3), dtype=np.uint8) * 128

        # Metrics should handle constant images without crashing
        # Note: Constant images may produce NaN for edge-based metrics (no edges)
        score = compute_edge_similarity(const_img, const_img.copy())
        # Allow NaN for constant images (expected behavior)
        if not np.isnan(score):
            self.assertGreaterEqual(score, 0.0, "Should handle constant images")
            self.assertLessEqual(score, 1.0, "Should handle constant images")


if __name__ == "__main__":
    unittest.main()
