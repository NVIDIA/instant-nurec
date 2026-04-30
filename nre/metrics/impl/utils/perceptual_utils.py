# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Perceptual quality metrics for image similarity assessment.

This module provides functional implementations of perceptual
quality metrics for evaluating image similarity and quality. Metrics include:

Core Metrics:
- Edge similarity: Sobel edge detection with strength and shape comparison
- Gradient similarity: Gradient magnitude similarity (GMS)
- Blur detection: Laplacian variance-based sharpness measurement
- Artifact detection: LAB color variance and channel correlation analysis
- Structural similarity (SSIM): Multi-scale SSIM or approximation
- Color histogram similarity: RGB histogram correlation

Chromatic Aberration Detection:
- CEM (Chroma Edge Misalignment): Gradient orientation coherence between luma/chroma
- Hue variance: Local hue variance for rainbow artifact detection
- Chroma HF: High-frequency chroma noise using Laplacian on UV channels
- Channel incoherence: Gradient channel correlation between color and luminance
- Y/Chroma ratio: High-frequency energy ratio between luminance and chroma

"""

from __future__ import annotations

import logging

import cv2
import numpy as np


try:
    from scipy.ndimage import sobel  # type: ignore[import]

    SCIPY_AVAILABLE = True
except ImportError:
    sobel = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available. Edge detection will use OpenCV instead.")

try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore[import]

    SKIMAGE_AVAILABLE = True
except ImportError:
    ssim = None  # type: ignore[assignment]
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image not available. SSIM metric will be approximated.")

logger = logging.getLogger(__name__)


# Metric Combination Weights and Constants

# Edge similarity: balance between edge strength and shape
EDGE_STRENGTH_WEIGHT = 0.6  # Weight for edge strength ratio
EDGE_SHAPE_WEIGHT = 0.4  # Weight for edge shape similarity

# Gradient similarity: balance between gradient strength and GMS
GRADIENT_STRENGTH_WEIGHT = 0.7  # Weight for gradient strength ratio
GRADIENT_GMS_WEIGHT = 0.3  # Weight for gradient magnitude similarity

# Blur score: balance between relative blur and absolute sharpness
BLUR_RELATIVE_WEIGHT = 0.7  # Weight for relative blur ratio
BLUR_ABSOLUTE_WEIGHT = 0.3  # Weight for absolute sharpness

# Artifact score: weights for different artifact components
ARTIFACT_COLOR_NOISE_WEIGHT = 0.40  # High-freq color noise (primary)
ARTIFACT_CHANNEL_CORR_WEIGHT = 0.30  # Channel correlation consistency
ARTIFACT_TOTAL_VAR_WEIGHT = 0.20  # Total variation
ARTIFACT_SATURATION_WEIGHT = 0.10  # Over-saturation penalty

# Chromatic aberration detection: exponential penalty factors
# Higher values = more sensitive to aberrations (steeper penalty curve)
CEM_PENALTY_FACTOR = 4.5  # Chroma Edge Misalignment sensitivity
HUE_VARIANCE_PENALTY_FACTOR = 3.5  # Hue variance (rainbow artifacts)
CHROMA_HF_PENALTY_FACTOR = 2.5  # High-frequency chroma noise
CHANNEL_COHERENCE_PENALTY_FACTOR = 4.0  # Channel gradient coherence
Y_CHROMA_RATIO_PENALTY_FACTOR = 3.0  # Y/Chroma energy ratio

# SSIM constants: (k * L)² where k1=0.01, k2=0.03, L=1.0 (normalized image range)
SSIM_C1 = 0.01**2  # Luminance stabilization constant
SSIM_C2 = 0.03**2  # Contrast stabilization constant

# GMS (Gradient Magnitude Similarity) constant - smaller = more sensitive
GMS_C = 10.0  # Stabilization constant for gradient comparison


def compute_edge_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute edge-based similarity using Sobel edge detection.

    Combines edge strength ratio and shape similarity for blur detection.

    Args:
        img1: Reference image [H, W, C] or [H, W].
        img2: Comparison image [H, W, C] or [H, W].

    Returns:
        Edge similarity score in [0, 1].
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = img1, img2

    # Compute edge maps
    if SCIPY_AVAILABLE and sobel is not None:
        edges1_x = sobel(gray1, axis=0)
        edges1_y = sobel(gray1, axis=1)
        edges1 = np.sqrt(edges1_x**2 + edges1_y**2)

        edges2_x = sobel(gray2, axis=0)
        edges2_y = sobel(gray2, axis=1)
        edges2 = np.sqrt(edges2_x**2 + edges2_y**2)
    else:
        edges1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        edges1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        edges1 = np.sqrt(edges1_x**2 + edges1_y**2)

        edges2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        edges2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        edges2 = np.sqrt(edges2_x**2 + edges2_y**2)

    # Compare edge strength BEFORE normalization (captures blur)
    mean_strength1 = np.mean(edges1)
    mean_strength2 = np.mean(edges2)
    strength_ratio = min(mean_strength2, mean_strength1) / (max(mean_strength2, mean_strength1) + 1e-8)

    # Normalize for shape comparison
    edges1_norm = edges1 / (np.max(edges1) + 1e-8)
    edges2_norm = edges2 / (np.max(edges2) + 1e-8)

    # Compute shape similarity
    edge_diff = np.abs(edges1_norm - edges2_norm)
    shape_similarity = 1.0 - np.mean(edge_diff)

    # Combine strength and shape (emphasize strength for blur detection)
    edge_similarity = EDGE_STRENGTH_WEIGHT * strength_ratio + EDGE_SHAPE_WEIGHT * shape_similarity

    return float(edge_similarity)


def compute_gradient_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute gradient magnitude similarity (GMS) for blur detection.

    Combines gradient strength ratio with GMS metric.

    Args:
        img1: Reference image [H, W, C] or [H, W].
        img2: Comparison image [H, W, C] or [H, W].

    Returns:
        Gradient similarity score in [0, 1].
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray1, gray2 = img1.astype(np.float32), img2.astype(np.float32)

    # Compute gradients
    grad1_x = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    grad1_y = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag1 = np.sqrt(grad1_x**2 + grad1_y**2)

    grad2_x = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
    grad2_y = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag2 = np.sqrt(grad2_x**2 + grad2_y**2)

    # Compare gradient strength (blur detection)
    mean_grad1 = np.mean(grad_mag1)
    mean_grad2 = np.mean(grad_mag2)
    grad_strength_ratio = min(mean_grad2, mean_grad1) / (max(mean_grad2, mean_grad1) + 1e-8)

    # Compute GMS (Gradient Magnitude Similarity)
    gms = (2 * grad_mag1 * grad_mag2 + GMS_C) / (grad_mag1**2 + grad_mag2**2 + GMS_C)
    gms_score = np.mean(gms)

    # Combine strength ratio and GMS (emphasize strength)
    gradient_similarity = GRADIENT_STRENGTH_WEIGHT * grad_strength_ratio + GRADIENT_GMS_WEIGHT * gms_score

    return float(gradient_similarity)


def compute_blur_penalty(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute blur score using Laplacian variance sharpness measurement.

    Args:
        img1: Reference image [H, W, C] or [H, W].
        img2: Comparison image [H, W, C] or [H, W].

    Returns:
        Blur score in [0, 1], where 1 indicates sharp image.
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = img1, img2

    # Compute Laplacian variance (measure of sharpness)
    lap1 = cv2.Laplacian(gray1, cv2.CV_64F)
    lap2 = cv2.Laplacian(gray2, cv2.CV_64F)

    var1 = lap1.var()
    var2 = lap2.var()

    # Measure relative blur AND absolute sharpness
    blur_ratio = min(var2, var1) / (max(var2, var1) + 1e-8)

    # Absolute sharpness component
    img_size = gray2.shape[0] * gray2.shape[1]
    normalized_var2 = var2 / img_size * 10000
    absolute_sharpness = min(1.0, normalized_var2 / 100)

    # Combine relative and absolute blur measures
    blur_score = BLUR_RELATIVE_WEIGHT * blur_ratio + BLUR_ABSOLUTE_WEIGHT * absolute_sharpness

    return float(blur_score)


def compute_artifact_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Detect rendering artifacts using multi-component analysis.

    Analyzes HF color noise, channel correlation, over-saturation, and
    spatial color smoothness to detect rendering anomalies.

    Args:
        img1: Reference image [H, W, C] in RGB format.
        img2: Comparison image [H, W, C] in RGB format.

    Returns:
        Artifact score in [0, 1], where 1 indicates no artifacts.
    """
    # Convert to uint8 if needed
    if img1.dtype != np.uint8:
        img1_uint8 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
        img2_uint8 = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
    else:
        img1_uint8 = img1
        img2_uint8 = img2

    # 1. LAB color space high-frequency content (detects color artifacts)
    lab1 = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

    # a and b channels capture color information
    a1, b1 = lab1[:, :, 1], lab1[:, :, 2]
    a2, b2 = lab2[:, :, 1], lab2[:, :, 2]

    # Compute local color variance (artifacts have high local variation)
    a1_smooth = cv2.GaussianBlur(a1, (5, 5), 1.5)
    b1_smooth = cv2.GaussianBlur(b1, (5, 5), 1.5)
    a2_smooth = cv2.GaussianBlur(a2, (5, 5), 1.5)
    b2_smooth = cv2.GaussianBlur(b2, (5, 5), 1.5)

    # High-frequency color content (artifacts)
    hf_a1 = np.std(a1 - a1_smooth)
    hf_b1 = np.std(b1 - b1_smooth)
    hf_a2 = np.std(a2 - a2_smooth)
    hf_b2 = np.std(b2 - b2_smooth)

    hf_color1 = hf_a1 + hf_b1
    hf_color2 = hf_a2 + hf_b2

    # img2 with more HF color = artifacts
    # Use more aggressive penalty
    hf_ratio = hf_color1 / (hf_color2 + 1e-8)
    if hf_ratio > 1.0:  # img2 has less noise (better)
        color_noise_score = min(1.0, hf_ratio)
    else:  # img2 has more noise (worse)
        color_noise_score = hf_ratio**2  # Quadratic penalty

    # 2. RGB channel correlation (artifacts show as channel desync)
    gray1 = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # For each color channel, check how well it correlates with luminance
    def channel_correlations(img: np.ndarray, gray: np.ndarray) -> list[float]:
        """Compute channel-luminance correlations.

        Args:
            img: RGB image [H, W, C].
            gray: Grayscale luminance image [H, W].

        Returns:
            List of absolute correlation values for each RGB channel.
        """
        corrs = []
        for c in range(3):
            ch = img[:, :, c].astype(np.float32)
            corr = np.corrcoef(ch.flatten(), gray.flatten())[0, 1]
            corrs.append(abs(corr))  # Use absolute correlation
        return corrs

    corrs1 = channel_correlations(img1_uint8, gray1)
    corrs2 = channel_correlations(img2_uint8, gray2)

    # Compute consistency score
    corr_diff = np.mean([abs(c1 - c2) for c1, c2 in zip(corrs1, corrs2)])
    channel_score = np.exp(-5 * corr_diff)  # Exponential penalty for differences

    # 3. Over-saturation detection (common rendering artifact)
    hsv1 = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2HSV)

    # Count oversaturated pixels (S > 250 or V > 250)
    oversat1 = (np.sum(hsv1[:, :, 1] > 250) + np.sum(hsv1[:, :, 2] > 250)) / (2 * hsv1[:, :, 1].size)
    oversat2 = (np.sum(hsv2[:, :, 1] > 250) + np.sum(hsv2[:, :, 2] > 250)) / (2 * hsv2[:, :, 1].size)

    if oversat2 > oversat1 * 2.0:
        saturation_score = 0.3  # Heavy penalty
    elif oversat2 > oversat1 * 1.3:
        saturation_score = 0.7
    else:
        saturation_score = 1.0

    # 4. Spatial color smoothness (artifacts have discontinuities)
    # Compute total variation in color channels
    def total_variation(img: np.ndarray) -> float:
        """Compute total variation across color channels.

        Args:
            img: RGB image [H, W, C].

        Returns:
            Total variation score summed across all RGB channels.
        """
        tv = 0.0
        for c in range(3):
            ch = img[:, :, c].astype(np.float32)
            tv += np.sum(np.abs(ch[1:, :] - ch[:-1, :])) + np.sum(np.abs(ch[:, 1:] - ch[:, :-1]))
        return tv

    tv1 = total_variation(img1_uint8)
    tv2 = total_variation(img2_uint8)

    # High TV in img2 relative to img1 = artifacts / noise
    tv_ratio = tv1 / (tv2 + 1e-8)
    if tv_ratio > 1.0:  # img2 has less variation (smoother/better)
        tv_score = min(1.0, tv_ratio)
    else:  # img2 has more variation (artifacts/noise)
        tv_score = tv_ratio**2  # Quadratic penalty

    # Combine all components with weights
    artifact_score = (
        ARTIFACT_COLOR_NOISE_WEIGHT * color_noise_score
        + ARTIFACT_CHANNEL_CORR_WEIGHT * channel_score
        + ARTIFACT_TOTAL_VAR_WEIGHT * tv_score
        + ARTIFACT_SATURATION_WEIGHT * saturation_score
    )

    return float(artifact_score)


def compute_multi_scale_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity index (SSIM).

    Uses scikit-image implementation if available, otherwise approximates.

    Args:
        img1: Reference image [H, W, C] or [H, W].
        img2: Comparison image [H, W, C] or [H, W].

    Returns:
        SSIM score in [-1, 1], typically in [0, 1].
    """
    # Ensure float32
    if img1.dtype != np.float32:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if SKIMAGE_AVAILABLE and ssim is not None:
        if len(img1.shape) == 3:
            ssim_vals = []
            for c in range(img1.shape[2]):
                ssim_val = ssim(img1[:, :, c], img2[:, :, c], data_range=1.0, gaussian_weights=True)
                ssim_vals.append(ssim_val)
            return float(np.mean(ssim_vals))
        else:
            return float(ssim(img1, img2, data_range=1.0))
    else:
        # SSIM approximation
        if len(img1.shape) == 3:
            ssim_approx = []
            for c in range(img1.shape[2]):
                ch1, ch2 = img1[:, :, c], img2[:, :, c]
                mu1, mu2 = ch1.mean(), ch2.mean()
                std1, std2 = ch1.std() + 1e-8, ch2.std() + 1e-8
                cov = np.mean((ch1 - mu1) * (ch2 - mu2))
                corr = cov / (std1 * std2)

                luminance = (2 * mu1 * mu2 + SSIM_C1) / (mu1**2 + mu2**2 + SSIM_C1)
                contrast = (2 * std1 * std2 + SSIM_C2) / (std1**2 + std2**2 + SSIM_C2)

                ssim_approx.append(luminance * contrast * corr)
            return float(np.mean(ssim_approx))
        else:
            mu1, mu2 = img1.mean(), img2.mean()
            std1, std2 = img1.std() + 1e-8, img2.std() + 1e-8
            cov = np.mean((img1 - mu1) * (img2 - mu2))
            corr = cov / (std1 * std2)

            luminance = (2 * mu1 * mu2 + SSIM_C1) / (mu1**2 + mu2**2 + SSIM_C1)
            contrast = (2 * std1 * std2 + SSIM_C2) / (std1**2 + std2**2 + SSIM_C2)

            return float(luminance * contrast * corr)


def compute_color_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute RGB histogram correlation.

    Args:
        img1: Reference image [H, W, C].
        img2: Comparison image [H, W, C].

    Returns:
        Histogram similarity score in [-1, 1].
    """
    hist_similarity = []

    for c in range(min(img1.shape[2], 3)):
        hist1 = cv2.calcHist([img1], [c], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [c], None, [256], [0, 256])

        hist1 = hist1.flatten() / (hist1.sum() + 1e-8)
        hist2 = hist2.flatten() / (hist2.sum() + 1e-8)

        correlation = np.corrcoef(hist1, hist2)[0, 1]
        hist_similarity.append(correlation)

    return float(np.mean(hist_similarity))


def _resize_for_metric(img: np.ndarray, max_side: int = 256) -> np.ndarray:
    """Downscale image while preserving aspect ratio for efficient computation.

    Args:
        img: Input image [H, W] or [H, W, C].
        max_side: Maximum dimension size.

    Returns:
        Resized image.
    """
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def compute_cem_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Chroma Edge Misalignment (CEM) score.

    Detects chromatic aberration via luma/chroma gradient orientation coherence.

    Args:
        img1: Reference image [H, W, C] in RGB format.
        img2: Comparison image [H, W, C] in RGB format.

    Returns:
        CEM score in [0, 1], where 1 indicates no aberration.
    """
    # Downscale for robustness/speed
    i1 = _resize_for_metric(img1)
    i2 = _resize_for_metric(img2)

    # Convert to YUV
    yuv1 = cv2.cvtColor(i1, cv2.COLOR_RGB2YUV).astype(np.float32)
    yuv2 = cv2.cvtColor(i2, cv2.COLOR_RGB2YUV).astype(np.float32)
    y1, u1, v1 = cv2.split(yuv1)
    y2, u2, v2 = cv2.split(yuv2)

    def grad_and_angle(imgf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradient magnitude and orientation.

        Args:
            imgf: Single channel image [H, W] as float32.

        Returns:
            Tuple of (magnitude, angle) arrays:
                - magnitude: Gradient magnitude [H, W].
                - angle: Gradient orientation in radians [0, 2π) [H, W].
        """
        gx = cv2.Sobel(imgf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(imgf, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = cv2.phase(gx, gy)  # [0, 2pi)
        return mag, ang

    ymag1, yang1 = grad_and_angle(y1)
    ymag2, yang2 = grad_and_angle(y2)
    umag1, uang1 = grad_and_angle(u1)
    vmag1, vang1 = grad_and_angle(v1)
    umag2, uang2 = grad_and_angle(u2)
    vmag2, vang2 = grad_and_angle(v2)

    # Orientation disagreement (weighted by chroma magnitude)
    def misalign(
        ymag: np.ndarray, yang: np.ndarray, umag: np.ndarray, uang: np.ndarray, vmag: np.ndarray, vang: np.ndarray
    ) -> float:
        """Compute misalignment score between luma and chroma gradients.

        Args:
            ymag: Luma gradient magnitude [H, W].
            yang: Luma gradient orientation in radians [H, W].
            umag: U channel gradient magnitude [H, W].
            uang: U channel gradient orientation in radians [H, W].
            vmag: V channel gradient magnitude [H, W].
            vang: V channel gradient orientation in radians [H, W].

        Returns:
            Misalignment score (higher = more chromatic aberration).
        """
        # Angular distance in [0, pi]
        du = np.abs(np.pi - np.abs(np.pi - np.abs(uang - yang)))
        dv = np.abs(np.pi - np.abs(np.pi - np.abs(vang - yang)))
        # Weight by chroma edge strength, focus away from strong luma edges
        ymask = (ymag > np.percentile(ymag, 60)).astype(np.float32)
        inv_y = 1.0 - cv2.dilate(ymask, np.ones((3, 3), np.uint8), iterations=1)
        numerator = np.mean(((du * umag) + (dv * vmag)) * inv_y)
        denominator = np.mean((umag + vmag) * inv_y) + 1e-8
        score = numerator / denominator
        return float(score)

    m1 = misalign(ymag1, yang1, umag1, uang1, vmag1, vang1)
    m2 = misalign(ymag2, yang2, umag2, uang2, vmag2, vang2)
    delta = max(0.0, m2 - m1)
    score = float(np.exp(-CEM_PENALTY_FACTOR * delta))
    return score


def compute_hue_variance_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute local hue variance score for rainbow artifact detection.

    Args:
        img1: Reference image [H, W, C] in RGB format.
        img2: Comparison image [H, W, C] in RGB format.

    Returns:
        Hue variance score in [0, 1], where 1 indicates no artifacts.
    """
    i1 = _resize_for_metric(img1)
    i2 = _resize_for_metric(img2)

    hsv1 = cv2.cvtColor(i1, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv2 = cv2.cvtColor(i2, cv2.COLOR_RGB2HSV).astype(np.float32)
    h1 = hsv1[:, :, 0] / 180.0  # Normalize hue to [0,1]
    h2 = hsv2[:, :, 0] / 180.0

    # Local variance via mean filter
    k = 7
    m1 = cv2.blur(h1, (k, k))
    m2 = cv2.blur(h2, (k, k))
    v1 = cv2.blur((h1 - m1) ** 2, (k, k))
    v2 = cv2.blur((h2 - m2) ** 2, (k, k))

    # Focus on non-luminance edge regions to avoid natural edges
    y1 = cv2.cvtColor(i1, cv2.COLOR_RGB2YUV)[:, :, 0].astype(np.float32)
    gy = cv2.Sobel(y1, cv2.CV_32F, 1, 0, ksize=3)
    gx = cv2.Sobel(y1, cv2.CV_32F, 0, 1, ksize=3)
    edge_mask = (cv2.magnitude(gx, gy) / (np.max(y1) + 1e-8) > 0.2).astype(np.float32)
    non_edge = 1.0 - cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)

    hv1 = float(np.mean(v1 * non_edge))
    hv2 = float(np.mean(v2 * non_edge))
    increase = max(0.0, (hv2 - hv1) / (hv1 + 1e-8))
    score = float(np.exp(-HUE_VARIANCE_PENALTY_FACTOR * increase))
    return score


def compute_chroma_hf_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute high-frequency chroma noise score via UV channel Laplacian.

    Args:
        img1: Reference image [H, W, C] in RGB format.
        img2: Comparison image [H, W, C] in RGB format.

    Returns:
        Chroma HF score in [0, 1], where 1 indicates no noise.
    """
    i1 = _resize_for_metric(img1)
    i2 = _resize_for_metric(img2)

    yuv1 = cv2.cvtColor(i1, cv2.COLOR_RGB2YUV).astype(np.float32)
    yuv2 = cv2.cvtColor(i2, cv2.COLOR_RGB2YUV).astype(np.float32)
    u1, v1 = yuv1[:, :, 1], yuv1[:, :, 2]
    u2, v2 = yuv2[:, :, 1], yuv2[:, :, 2]

    hf1 = np.mean(np.abs(cv2.Laplacian(u1, cv2.CV_32F))) + np.mean(np.abs(cv2.Laplacian(v1, cv2.CV_32F)))
    hf2 = np.mean(np.abs(cv2.Laplacian(u2, cv2.CV_32F))) + np.mean(np.abs(cv2.Laplacian(v2, cv2.CV_32F)))

    inc = max(0.0, (hf2 - hf1) / (hf1 + 1e-8))
    return float(np.exp(-CHROMA_HF_PENALTY_FACTOR * inc))


def compute_channel_coherence_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute gradient channel coherence score.

    Detects chromatic aberration via correlation between color and luminance
    gradient fields.

    Args:
        img1: Reference image [H, W, C] in RGB format.
        img2: Comparison image [H, W, C] in RGB format.

    Returns:
        Channel coherence score in [0, 1], where 1 indicates good coherence.
    """
    i1 = _resize_for_metric(img1)
    i2 = _resize_for_metric(img2)

    y2 = cv2.cvtColor(i2, cv2.COLOR_RGB2YUV)[:, :, 0].astype(np.float32)
    r2, g2, b2 = cv2.split(i2.astype(np.float32))

    # Gradients
    def grad_mag(imgf: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude.

        Args:
            imgf: Single channel image [H, W] as float32.

        Returns:
            Gradient magnitude [H, W].
        """
        return cv2.magnitude(cv2.Sobel(imgf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(imgf, cv2.CV_32F, 0, 1, ksize=3))

    gy = grad_mag(y2)
    gr = grad_mag(r2)
    gg = grad_mag(g2)
    gb = grad_mag(b2)
    color_mag2 = (gr + gg + gb) / 3.0

    # Local correlation between color and luminance gradients
    def local_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Compute local correlation between two gradient fields.

        Args:
            a: First gradient field [H, W].
            b: Second gradient field [H, W].

        Returns:
            Mean local correlation score between the two fields.
        """
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        ma = cv2.blur(a, (7, 7))
        mb = cv2.blur(b, (7, 7))
        sa = cv2.blur((a - ma) ** 2, (7, 7)) + 1e-8
        sb = cv2.blur((b - mb) ** 2, (7, 7)) + 1e-8
        cov = cv2.blur((a - ma) * (b - mb), (7, 7))
        corr = cov / np.sqrt(sa * sb)
        return float(np.mean(corr))

    # Reference coherence
    y1 = cv2.cvtColor(i1, cv2.COLOR_RGB2YUV)[:, :, 0].astype(np.float32)
    r1, g1, b1 = cv2.split(i1.astype(np.float32))
    color_mag1 = (grad_mag(r1) + grad_mag(g1) + grad_mag(b1)) / 3.0
    coh1 = local_corr(color_mag1, grad_mag(y1))
    coh2 = local_corr(color_mag2, gy)
    drop = max(0.0, coh1 - coh2)
    return float(np.exp(-CHANNEL_COHERENCE_PENALTY_FACTOR * drop))


def compute_y_chroma_ratio_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Y/Chroma high-frequency energy ratio score.

    Detects rainbow artifacts by measuring unnatural increases in chroma HF
    energy relative to luminance HF energy.

    Args:
        img1: Reference image [H, W, C] in RGB format.
        img2: Comparison image [H, W, C] in RGB format.

    Returns:
        Y/Chroma ratio score in [0, 1], where 1 indicates natural ratio.
    """
    i1 = _resize_for_metric(img1)
    i2 = _resize_for_metric(img2)

    yuv1 = cv2.cvtColor(i1, cv2.COLOR_RGB2YUV).astype(np.float32)
    yuv2 = cv2.cvtColor(i2, cv2.COLOR_RGB2YUV).astype(np.float32)
    y1 = yuv1[:, :, 0]
    y2 = yuv2[:, :, 0]
    u1, v1 = yuv1[:, :, 1], yuv1[:, :, 2]
    u2, v2 = yuv2[:, :, 1], yuv2[:, :, 2]

    def hf_energy(imgf: np.ndarray) -> float:
        """Compute high-frequency energy using Laplacian.

        Args:
            imgf: Single channel image [H, W] as float32.

        Returns:
            Mean absolute Laplacian value (high-frequency energy).
        """
        return float(np.mean(np.abs(cv2.Laplacian(imgf, cv2.CV_32F))))

    r1 = hf_energy(y1) / (hf_energy(u1) + hf_energy(v1) + 1e-8)
    r2 = hf_energy(y2) / (hf_energy(u2) + hf_energy(v2) + 1e-8)
    drop = max(0.0, (r1 - r2) / (r1 + 1e-8))
    return float(np.exp(-Y_CHROMA_RATIO_PENALTY_FACTOR * drop))
