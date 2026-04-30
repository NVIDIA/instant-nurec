# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging

from pathlib import Path
from typing import Any, Dict, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from omegaconf import OmegaConf

import nre.systems

from nre.config.nre import NREConfig
from nre.models.post_processing import PPISPPostProcessing
from nre.models.post_processings.ppisp import (
    CRF,
    PPISP,
    BasePPISP,
    ColorCorrection,
    ExposureOffset,
    PiecewisePowerFunction,
    RadialFalloff,
    Vignetting,
)


logger = logging.getLogger(__name__)


def create_grayscale_bar_gradient(size: int, num_bars: int = 16) -> np.ndarray:
    """Create a grayscale gradient image with vertical bars.

    Args:
        size: Width and height of the output image in pixels
        num_bars: Number of vertical bars in the gradient

    Returns:
        np.ndarray: Grayscale gradient image with shape (size, size, 3)
    """
    bar_width = size // num_bars
    viz_image = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(num_bars):
        intensity = np.power(i / (num_bars - 1), 2.2)  # Values from 0 to 1
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < num_bars - 1 else size
        viz_image[:, x_start:x_end] = intensity
    return viz_image


def plot_exposure_offsets(
    gs: plt.GridSpec,
    ppisp_model: BasePPISP,
    camera_idx: int,
) -> None:
    """Plot exposure offsets on the given gridspec."""
    gs_sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = plt.gcf().add_subplot(gs_sub[0])
    ax_viz = plt.gcf().add_subplot(gs_sub[1])

    # Get packed exposure parameters
    all_exposure_params = ppisp_model.packed_exposure_params

    # Extract exposure offsets for this camera only
    start_idx = sum(ppisp_model.n_frames_per_camera[:camera_idx])
    end_idx = start_idx + ppisp_model.n_frames_per_camera[camera_idx]
    camera_exposure_offsets = all_exposure_params[start_idx:end_idx].detach().cpu().numpy()

    # Calculate mean offset for this camera
    mean_offset = torch.mean(all_exposure_params[start_idx:end_idx])

    # Calculate number of frames for this camera
    num_frames = end_idx - start_idx

    # Plot exposure offsets
    ax_plot.plot(range(num_frames), camera_exposure_offsets, "b-", label="Exposure Offset")
    ax_plot.axhline(y=mean_offset.item(), color="b", linestyle="--", alpha=0.5, label="Mean Offset")
    ax_plot.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_plot.set_title("Exposure Offset Over Time")
    ax_plot.set_xlabel("Frame Index")
    ax_plot.set_ylabel("Exposure Offset [EV]")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend()

    # Create visualization image
    size = 256
    comparison_image = create_grayscale_bar_gradient(size)

    # Apply exposure offset to bottom half (2^mean_offset)
    comparison_image[size // 2 :, :] = (
        ExposureOffset.apply_exposure_offset(torch.from_numpy(comparison_image[size // 2 :, :]).to("cuda"), mean_offset)
        .cpu()
        .numpy()
    )

    # Plot the visualization
    ax_viz.imshow(np.clip(np.power(comparison_image, 1.0 / 2.2), 0.0, 1.0))
    ax_viz.set_title("Mean Exposure Offset Visualization")
    ax_viz.axis("off")
    ax_viz.text(
        size // 2, 20, "Original", ha="center", va="top", color="white", bbox=dict(facecolor="black", alpha=0.5)
    )
    ax_viz.text(
        size // 2,
        size - 20,
        f"{'+' if mean_offset.item() > 0.0 else '-'}{abs(mean_offset.item()):.2f} EV",
        ha="center",
        va="bottom",
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )


def plot_vignetting_curves(
    gs: plt.GridSpec,
    ppisp_model: BasePPISP,
    camera_idx: int,
) -> None:
    """Plot vignetting curves for R, G, B channels and show a visualization."""
    gs_sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_curves = plt.gcf().add_subplot(gs_sub[0])
    ax_viz = plt.gcf().add_subplot(gs_sub[1])

    # Get packed vignetting parameters once
    packed_vignetting_params = ppisp_model.packed_vignetting_params

    # Plot vignetting curves
    r = torch.linspace(0, np.sqrt(2) / 2.0, 100, device="cuda")
    r2 = r * r  # squared radial distances

    colors = [
        (1, 0, 0, 0.6),  # red with alpha
        (0, 1, 0, 0.6),  # green with alpha
        (0, 0, 1, 0.6),  # blue with alpha
    ]

    # Use RadialFalloff.apply_radial_falloff for each channel with direct 2D indexing
    for channel_idx, color in enumerate(colors):
        with torch.no_grad():
            # Direct 2D indexing - no intermediate PackedParams construction needed
            optical_center = packed_vignetting_params.optical_center[camera_idx, channel_idx]  # Shape: (2,)
            alphas = packed_vignetting_params.alphas[camera_idx, channel_idx]  # Shape: (num_alpha_terms,)

            # Create dummy coordinates at distance r from optical center
            coords_xy = torch.stack([r + optical_center[0], torch.full_like(r, optical_center[1].item())], dim=1)

            # Create dummy values of 1.0 to get the falloff factors
            dummy_values = torch.ones_like(r)

            # Apply radial falloff using the static method
            vig = RadialFalloff.apply_radial_falloff(dummy_values, coords_xy, optical_center, alphas)

            ax_curves.plot(r.cpu(), vig.cpu(), color=color, label=["Red", "Green", "Blue"][channel_idx], linewidth=2.0)

    ax_curves.set_xlabel("Radial Distance")
    ax_curves.set_ylabel("Light Transmission")
    ax_curves.set_title("Vignetting Curves (R,G,B)")
    ax_curves.grid(True, alpha=0.3)
    ax_curves.legend()
    ax_curves.set_ylim(bottom=0)

    # Create visualization image
    size = 256
    gray_value = 0.75
    rgb = torch.full((size * size, 3), gray_value, device="cuda")
    y, x = torch.meshgrid(
        torch.linspace(0, 1, size, device="cuda"),
        torch.linspace(0, 1, size, device="cuda"),
        indexing="ij",
    )
    coords_xy = torch.stack([x.flatten(), y.flatten()], dim=1)

    # Apply vignetting using the static method for each RGB channel with direct 2D indexing
    processed_rgb = rgb.clone()
    for channel_idx in range(3):
        # Direct 2D indexing - no intermediate PackedParams construction needed
        optical_center = packed_vignetting_params.optical_center[camera_idx, channel_idx]  # Shape: (2,)
        alphas = packed_vignetting_params.alphas[camera_idx, channel_idx]  # Shape: (num_alpha_terms,)

        # Apply vignetting to this channel using the static method
        processed_rgb[:, channel_idx] = RadialFalloff.apply_radial_falloff(
            rgb[:, channel_idx], coords_xy, optical_center, alphas
        )

    processed_image = processed_rgb.reshape(size, size, 3).cpu().numpy()

    # Plot the processed image
    ax_viz.set_title("Vignetting Effect Visualization")
    ax_viz.axis("off")
    ax_viz.imshow(np.power(processed_image, 1.0 / 2.2))

    # Add gray dashed lines at image center (0.5, 0.5)
    center_x = size // 2
    center_y = size // 2
    ax_viz.axhline(y=center_y, color="gray", linestyle="--", alpha=0.5)
    ax_viz.axvline(x=center_x, color="gray", linestyle="--", alpha=0.5)

    # Add black cross at optical center for each RGB channel with direct 2D indexing
    cross_size = 10
    cross_width = 2
    for channel_idx, color in enumerate(colors):
        # Direct 2D indexing - no intermediate PackedParams construction needed
        optical_center_np = (
            packed_vignetting_params.optical_center[camera_idx, channel_idx].detach().cpu().numpy()
        )  # Shape: (2,)
        optical_center_x = float(optical_center_np[0]) * size
        optical_center_y = float(optical_center_np[1]) * size
        ax_viz.plot(
            [optical_center_x - cross_size, optical_center_x + cross_size],
            [optical_center_y, optical_center_y],
            color=color,
            linewidth=cross_width,
        )
        ax_viz.plot(
            [optical_center_x, optical_center_x],
            [optical_center_y - cross_size, optical_center_y + cross_size],
            color=color,
            linewidth=cross_width,
        )


class ChromCoordTransform:
    """
    Transform between window coordinates and barycentric RG chromaticity coordinates.

    The class defines an isosceles triangle in the viewport where:
    - The top vertex is at the center horizontally, margin pixels from the top (Blue)
    - The bottom left vertex is margin pixels from the bottom left corner (Red)
    - The bottom right vertex is margin pixels from the bottom right corner (Green)
    - Coordinates inside this triangle map to valid RG chromaticity values
    """

    def __init__(self, width: int, height: int, margin: int = 20):
        """
        Initialize the coordinate transform.

        Args:
            width: Width of the viewport in pixels
            height: Height of the viewport in pixels
            margin: Margin from the edges in pixels
        """
        self.width = width
        self.height = height
        self.margin = margin

        # Define triangle vertices in window coordinates
        self.top = (width // 2, margin)  # Blue at top
        self.bottom_left = (margin, height - margin)  # Red at bottom left
        self.bottom_right = (width - margin, height - margin)  # Green at bottom right

        # Calculate effective width and height for scaling
        self.eff_width = width - 2 * margin
        self.eff_height = height - 2 * margin

    def window_to_barycentric(self, x: float, y: float) -> tuple:
        """
        Convert window coordinates to barycentric (RG) coordinates.

        Args:
            x: X coordinate in window space
            y: Y coordinate in window space

        Returns:
            Tuple of (r, g) chromaticity coordinates
        """
        # Calculate vectors from top vertex to point and bottom vertices
        v0x = self.bottom_left[0] - self.top[0]
        v0y = self.bottom_left[1] - self.top[1]

        v1x = self.bottom_right[0] - self.top[0]
        v1y = self.bottom_right[1] - self.top[1]

        v2x = x - self.top[0]
        v2y = y - self.top[1]

        # Calculate dot products
        d00 = v0x * v0x + v0y * v0y
        d01 = v0x * v1x + v0y * v1y
        d11 = v1x * v1x + v1y * v1y
        d20 = v2x * v0x + v2y * v0y
        d21 = v2x * v1x + v2y * v1y

        # Calculate barycentric coordinates
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        # Map barycentric coordinates to RG chromaticity
        # Top vertex is pure blue (0,0), bottom left is pure red (1,0), bottom right is pure green (0,1)
        r = v
        g = w

        return (r, g)

    def barycentric_to_window(self, r: float, g: float) -> tuple:
        """
        Convert barycentric (RG) coordinates to window coordinates.

        Args:
            r: Red chromaticity coordinate
            g: Green chromaticity coordinate

        Returns:
            Tuple of (x, y) window coordinates
        """
        # Calculate barycentric coordinates
        # r corresponds to bottom left, g to bottom right, b (1-r-g) to top
        b = 1.0 - r - g

        # Weighted sum of vertices - return floating point values, not integers
        x = r * self.bottom_left[0] + g * self.bottom_right[0] + b * self.top[0]
        y = r * self.bottom_left[1] + g * self.bottom_right[1] + b * self.top[1]

        return (x, y)


def create_chromaticity_triangle(transform: ChromCoordTransform) -> np.ndarray:
    """Create RG chromaticity triangle with colors using coordinate transform"""
    chrom_image = np.ones((transform.height, transform.width, 3), dtype=np.float32)

    for y in range(transform.height):
        for x in range(transform.width):
            r, g = transform.window_to_barycentric(x, y)
            b = 1 - r - g

            # Calculate RGB color
            rgb = np.array([r, g, b])
            # Normalize by maximum component for full saturation
            rgb_max = rgb.max()
            if rgb_max > 0:  # Avoid division by zero
                rgb = rgb / rgb_max

            # Calculate blend factor using clamped linear functions
            slope = transform.height / 2
            r_blend = np.clip(slope * r + 0.5, 0.0, 1.0)
            g_blend = np.clip(slope * g + 0.5, 0.0, 1.0)
            b_blend = np.clip(slope * b + 0.5, 0.0, 1.0)
            blend = r_blend * g_blend * b_blend

            # Blend between RGB color and white
            chrom_image[y, x] = blend * rgb + (1 - blend) * np.ones(3)

    return chrom_image


def plot_color_correction_curves(
    gs_top: plt.GridSpec,
    gs_bottom: plt.GridSpec,
    ppisp_model: BasePPISP,
    camera_idx: int,
) -> None:
    """
    Plot color correction curves for the four target chromaticities (R, G, B, neutral).

    Args:
        gs_top: GridSpec for the top row
        gs_bottom: GridSpec for the bottom row
        ppisp_model: The PPISP model instance
        camera_idx: Index of the camera
    """
    # Derive frame indices from camera_idx and model data
    frame_start_idx = sum(ppisp_model.n_frames_per_camera[:camera_idx])
    num_frames = ppisp_model.n_frames_per_camera[camera_idx]

    # 1. Extract trained data from the model using packed parameters for this camera only
    source_chroms = ColorCorrection.get_default_source_chroms("cuda")
    packed_color_params = ppisp_model.packed_color_params
    # Filter to only this camera's frames
    camera_color_params = packed_color_params[frame_start_idx : frame_start_idx + num_frames]
    h = ColorCorrection.params_to_homography(camera_color_params)
    target_chroms = ColorCorrection.apply_color_correction_rg(source_chroms, h)
    shifts = target_chroms - source_chroms.unsqueeze(0)  # Shape: [num_frames_for_camera, 4, 2]

    # If data contains NaNs, return early
    if torch.isnan(shifts).any():
        logger.warning("Color correction data contains NaNs, skipping plot")
        return

    # Extract shifts for each dimension
    red_cyan_shifts = shifts[:, :, 0].cpu().numpy()
    green_magenta_shifts = shifts[:, :, 1].cpu().numpy()

    # Define target names and colors (Blue, Red, Green, Neutral)
    target_names = ["Blue", "Red", "Green", "Neutral"]
    target_colors = ["blue", "red", "green", "gray"]

    # 2. Set up plots
    gs_top_sub = gs_top.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    gs_bottom_sub = gs_bottom.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)

    ax_red_cyan_time = plt.gcf().add_subplot(gs_top_sub[0])
    ax_rg_diagram = plt.gcf().add_subplot(gs_top_sub[1])
    ax_green_magenta_time = plt.gcf().add_subplot(gs_bottom_sub[0])
    ax_viz = plt.gcf().add_subplot(gs_bottom_sub[1])

    # 3. Create data for plots
    # Exaggerate shifts for better visibility
    scale = 5.0
    chroms_exaggerated_shifts = source_chroms.unsqueeze(0) + shifts * scale

    frame_indices = np.arange(num_frames)

    # Create RG chromaticity diagram
    size = 256
    transform = ChromCoordTransform(width=size, height=int(size * np.sqrt(3.0) / 2.0), margin=0)
    rg_diagram = create_chromaticity_triangle(transform)
    bar_width = size // 4
    rgb_colors = [
        [0, 0, 0.9],  # Blue
        [0.9, 0, 0],  # Red
        [0, 0.9, 0],  # Green
        [0.5, 0.5, 0.5],  # Neutral gray
    ]

    comparison_image = np.full((size, size, 3), 0.5, dtype=np.float32)
    for i, color in enumerate(rgb_colors):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < 3 else size
        comparison_image[:, x_start:x_end] = color

    # Get color correction matrix from average target chromaticities
    mean_target_chroms = chroms_exaggerated_shifts.mean(dim=0)
    h_matrix = ColorCorrection.get_h_from_chrom_pairs(source_chroms, mean_target_chroms)

    # Apply color correction only to bottom half
    bottom_half = torch.from_numpy(comparison_image[size // 2 :].reshape(-1, 3)).to("cuda")
    h_batch = h_matrix.unsqueeze(0).expand(bottom_half.shape[0], -1, -1)
    comparison_image[size // 2 :] = (
        ColorCorrection.apply_color_correction_rgb(bottom_half, h_batch).reshape(-1, size, 3).cpu().numpy()
    )
    comparison_image = np.clip(comparison_image, 0.0, 1.0)

    # 4. Plot everything
    # Plot red-cyan shifts over time
    for i, (name, color_str) in enumerate(zip(target_names, target_colors)):
        ax_red_cyan_time.plot(frame_indices, red_cyan_shifts[:, i], color=color_str, label=name, alpha=0.7)

    ax_red_cyan_time.set_title("Red-Cyan Shift Over Time")
    ax_red_cyan_time.set_xlabel("Frame Index")
    ax_red_cyan_time.set_ylabel("Red-Cyan Shift")
    ax_red_cyan_time.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax_red_cyan_time.grid(True, alpha=0.3)
    ax_red_cyan_time.legend()

    # Plot green-magenta shifts over time
    for i, (name, color_str) in enumerate(zip(target_names, target_colors)):
        ax_green_magenta_time.plot(frame_indices, green_magenta_shifts[:, i], color=color_str, label=name, alpha=0.7)

    ax_green_magenta_time.set_title("Green-Magenta Shift Over Time")
    ax_green_magenta_time.set_xlabel("Frame Index")
    ax_green_magenta_time.set_ylabel("Green-Magenta Shift")
    ax_green_magenta_time.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax_green_magenta_time.grid(True, alpha=0.3)
    ax_green_magenta_time.legend()

    # Plot chromaticity diagram
    ax_rg_diagram.imshow(rg_diagram)
    ax_rg_diagram.axis("off")
    ax_rg_diagram.set_title(f"Chromaticity Shifts Over Time, Scaled {scale:.1f}x")

    # Plot chromaticity trajectories
    cross_size = 7
    cross_width = 2
    for i in range(len(target_names)):
        # Transform chromaticity coordinates to diagram coordinates
        points = np.array(
            [
                transform.barycentric_to_window(r, g)
                for r, g in zip(
                    chroms_exaggerated_shifts[:, i, 0].cpu().numpy(), chroms_exaggerated_shifts[:, i, 1].cpu().numpy()
                )
            ]
        )

        # Plot trajectory
        ax_rg_diagram.plot(points[:, 0], points[:, 1], "-", color="black", linewidth=1.0, alpha=0.7)

        # Add cross marker at final position
        final_x, final_y = points[-1]
        ax_rg_diagram.plot(
            [final_x - cross_size, final_x + cross_size], [final_y, final_y], "-", color="black", linewidth=cross_width
        )
        ax_rg_diagram.plot(
            [final_x, final_x], [final_y - cross_size, final_y + cross_size], "-", color="black", linewidth=cross_width
        )

        # Add cross marker at source position
        source_r, source_g = source_chroms[i, 0].item(), source_chroms[i, 1].item()
        source_x, source_y = transform.barycentric_to_window(source_r, source_g)
        ax_rg_diagram.plot(
            [source_x - cross_size * 0.75, source_x + cross_size * 0.75],
            [source_y, source_y],
            "-",
            color="black",
            linewidth=cross_width / 2,
            alpha=0.5,
        )
        ax_rg_diagram.plot(
            [source_x, source_x],
            [source_y - cross_size * 0.75, source_y + cross_size * 0.75],
            "-",
            color="black",
            linewidth=cross_width / 2,
            alpha=0.5,
        )

    # Display before/after visualization
    ax_viz.imshow(np.power(comparison_image, 1.0 / 2.2))
    ax_viz.axis("off")
    ax_viz.set_title("Mean Color Correction Visualization")
    ax_viz.text(
        size // 2, 20, "Original", ha="center", va="top", color="white", bbox=dict(facecolor="black", alpha=0.5)
    )
    ax_viz.text(
        size // 2,
        size - 20,
        f"Color Corrected, Scaled {scale:.1f}x",
        ha="center",
        va="bottom",
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )


def plot_crf_curves(
    gs: plt.GridSpec,
    camera_curve_points: "PiecewisePowerFunction.CurvePoints",
    camera_idx: int,
) -> None:
    """Plot camera response function (CRF) curves for R, G, B channels and show a visualization."""
    gs_sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_curves = plt.gcf().add_subplot(gs_sub[0])
    ax_viz = plt.gcf().add_subplot(gs_sub[1])

    colors = [
        (1, 0, 0, 0.6),  # red with alpha
        (0, 1, 0, 0.6),  # green with alpha
        (0, 0, 1, 0.6),  # blue with alpha
    ]

    # Find the dynamic range (maximum x value) for plotting using vectorized operations
    # camera_curve_points has shape (3,) for all attributes, representing RGB channels
    max_x = 0.0
    with torch.no_grad():
        # Create ones tensor with shape (3,) for all channels
        ones_tensor = torch.ones(3, device="cuda")
        inverse_ones = PiecewisePowerFunction.inverse(camera_curve_points, ones_tensor)
        max_x = inverse_ones.max().item()

    x_values = torch.linspace(0, max_x, 200, device="cuda")

    # Plot CRF curves for each channel separately
    with torch.no_grad():
        for channel_idx, color in enumerate(colors):
            # Create curve points for a single channel
            single_channel_curve_points = PiecewisePowerFunction.CurvePoints(
                x0=camera_curve_points.x0[channel_idx],
                y0=camera_curve_points.y0[channel_idx],
                slope_p0=camera_curve_points.slope_p0[channel_idx],
                y0_pre_gamma=camera_curve_points.y0_pre_gamma[channel_idx],
                slope_line=camera_curve_points.slope_line[channel_idx],
                gamma=camera_curve_points.gamma[channel_idx],
                x1=camera_curve_points.x1[channel_idx],
                y1=camera_curve_points.y1[channel_idx],
                slope_p1=camera_curve_points.slope_p1[channel_idx],
                shoulder_x=camera_curve_points.shoulder_x[channel_idx],
                shoulder_y=camera_curve_points.shoulder_y[channel_idx],
            )

            # Apply PPF to this channel's x_values
            y_values = PiecewisePowerFunction.apply_ppf(single_channel_curve_points, x_values)

            ax_curves.plot(
                x_values.cpu(),
                y_values.cpu(),
                color=color,
                label=["Red", "Green", "Blue"][channel_idx],
                linewidth=2.0,
            )

    # Add vertical line at x=1.0
    ax_curves.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)

    ax_curves.set_xlabel("Linear Input Intensity")
    ax_curves.set_ylabel("Output Intensity")
    ax_curves.set_title("Camera Response Function and Color Space Conversion (R, G, B)")
    ax_curves.grid(True, alpha=0.3)
    ax_curves.legend()

    # Create visualization image
    size = 256
    comparison_image = create_grayscale_bar_gradient(size, num_bars=16)
    rgb_flat = torch.from_numpy(comparison_image[size // 2 :, :].reshape(-1, 3)).to("cuda")

    # Apply CRF to each channel separately
    processed_rgb = torch.zeros_like(rgb_flat)
    for channel_idx in range(3):
        # Create curve points for a single channel
        single_channel_curve_points = PiecewisePowerFunction.CurvePoints(
            x0=camera_curve_points.x0[channel_idx],
            y0=camera_curve_points.y0[channel_idx],
            slope_p0=camera_curve_points.slope_p0[channel_idx],
            y0_pre_gamma=camera_curve_points.y0_pre_gamma[channel_idx],
            slope_line=camera_curve_points.slope_line[channel_idx],
            gamma=camera_curve_points.gamma[channel_idx],
            x1=camera_curve_points.x1[channel_idx],
            y1=camera_curve_points.y1[channel_idx],
            slope_p1=camera_curve_points.slope_p1[channel_idx],
            shoulder_x=camera_curve_points.shoulder_x[channel_idx],
            shoulder_y=camera_curve_points.shoulder_y[channel_idx],
        )

        # Apply PPF to this channel's values
        processed_rgb[:, channel_idx] = PiecewisePowerFunction.apply_ppf(
            single_channel_curve_points, rgb_flat[:, channel_idx]
        )

    comparison_image[size // 2 :, :] = processed_rgb.reshape(size // 2, size, 3).cpu().numpy()

    # Plot the visualization
    ax_viz.imshow(comparison_image)
    ax_viz.set_title("Tone Mapping Visualization")
    ax_viz.axis("off")
    ax_viz.text(size // 2, 20, "Linear", ha="center", va="top", color="white", bbox=dict(facecolor="black", alpha=0.5))
    ax_viz.text(
        size // 2,
        size - 20,
        "Tone Mapped",
        ha="center",
        va="bottom",
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )


@torch.no_grad()
def extract_ppisp_params(
    ppisp_model: BasePPISP,
) -> Dict[str, Any]:
    """
    Extract all PPISP parameters into a single JSON-compatible dictionary.

    Args:
        ppisp_model: The PPISP model instance

    Returns:
        Dictionary containing all extracted parameters
    """
    # Create top-level structure
    params = {
        "num_cameras": ppisp_model.num_cameras,
        "num_frames": sum(ppisp_model.n_frames_per_camera),
        "n_frames_per_camera": ppisp_model.n_frames_per_camera,
    }

    # 1. Exposure offsets using official accessor
    packed_exposure_params = ppisp_model.packed_exposure_params
    params["exposure_params"] = packed_exposure_params.detach().cpu().numpy().flatten().tolist()

    # 2. Vignetting using official accessor
    packed_vignetting_params = ppisp_model.packed_vignetting_params
    vignetting_params = []
    for camera_idx in range(ppisp_model.num_cameras):
        # List of parameters for each color channel
        camera_params = []
        for channel_idx in range(3):
            color_params = {}
            color_params["optical_center"] = (
                packed_vignetting_params.optical_center[camera_idx, channel_idx].detach().cpu().numpy().tolist()
            )
            color_params["alpha"] = (
                packed_vignetting_params.alphas[camera_idx, channel_idx].detach().cpu().numpy().tolist()
            )
            camera_params.append(color_params)
        vignetting_params.append(camera_params)
    params["vignetting"] = vignetting_params

    # 3. Color correction homography parameters using official accessor
    packed_color_params = ppisp_model.packed_color_params
    params["color_params"] = packed_color_params.detach().cpu().numpy().tolist()

    # 4. CRF effective parameters only (curve points are already effective parameters)
    crf_curve_points = ppisp_model.crf_curve_points  # CurvePoints object with tensors shaped [num_cameras, 3]
    crf_effective_values = []
    for camera_idx in range(ppisp_model.num_cameras):
        camera_effective_params = []
        for channel_idx in range(3):
            # Access individual curve point parameters and index into them
            effective_channel_params = {
                "x0": crf_curve_points.x0[camera_idx, channel_idx].detach().cpu().item(),
                "y0": crf_curve_points.y0[camera_idx, channel_idx].detach().cpu().item(),
                "slope_p0": crf_curve_points.slope_p0[camera_idx, channel_idx].detach().cpu().item(),
                "y0_pre_gamma": crf_curve_points.y0_pre_gamma[camera_idx, channel_idx].detach().cpu().item(),
                "slope_line": crf_curve_points.slope_line[camera_idx, channel_idx].detach().cpu().item(),
                "gamma": crf_curve_points.gamma[camera_idx, channel_idx].detach().cpu().item(),
                "x1": crf_curve_points.x1[camera_idx, channel_idx].detach().cpu().item(),
                "y1": crf_curve_points.y1[camera_idx, channel_idx].detach().cpu().item(),
                "slope_p1": crf_curve_points.slope_p1[camera_idx, channel_idx].detach().cpu().item(),
                "shoulder_x": crf_curve_points.shoulder_x[camera_idx, channel_idx].detach().cpu().item(),
                "shoulder_y": crf_curve_points.shoulder_y[camera_idx, channel_idx].detach().cpu().item(),
            }
            camera_effective_params.append(effective_channel_params)
        crf_effective_values.append(camera_effective_params)
    params["crf"] = {"effective_values": crf_effective_values}

    return params


@torch.no_grad()
def generate_camera_report(
    ppisp_model: BasePPISP,
    camera_idx: int,
    output_path: Path,
) -> None:
    """
    Generate PDF report for a single camera's PPISP parameters.

    Args:
        ppisp_model: The PPISP model instance
        camera_idx: Index of the camera
        output_path: Full path where to save the generated report
    """
    logger.info(f"Generating report: {output_path}")

    num_rows = 5  # Exposure offsets, vignetting, 2 rows for color correction, and CRF
    fig_width = 20
    row_height = 5
    fig = plt.figure(figsize=(fig_width, row_height * num_rows))
    gs_main = fig.add_gridspec(num_rows, 1, height_ratios=[1] * num_rows)

    # Plot exposure offsets
    plot_exposure_offsets(gs_main[0], ppisp_model, camera_idx)

    # Plot vignetting curves
    plot_vignetting_curves(gs_main[1], ppisp_model, camera_idx)

    # Plot color correction curves (passing two separate rows)
    plot_color_correction_curves(gs_main[2], gs_main[3], ppisp_model, camera_idx)

    # Get curve points for this specific camera and create a CurvePoints object
    all_curve_points = ppisp_model.crf_curve_points
    camera_curve_points = PiecewisePowerFunction.CurvePoints(
        x0=all_curve_points.x0[camera_idx],  # Shape: (3,)
        y0=all_curve_points.y0[camera_idx],
        slope_p0=all_curve_points.slope_p0[camera_idx],
        y0_pre_gamma=all_curve_points.y0_pre_gamma[camera_idx],
        slope_line=all_curve_points.slope_line[camera_idx],
        gamma=all_curve_points.gamma[camera_idx],
        x1=all_curve_points.x1[camera_idx],
        y1=all_curve_points.y1[camera_idx],
        slope_p1=all_curve_points.slope_p1[camera_idx],
        shoulder_x=all_curve_points.shoulder_x[camera_idx],
        shoulder_y=all_curve_points.shoulder_y[camera_idx],
    )

    # Plot camera response function curves
    plot_crf_curves(gs_main[4], camera_curve_points, camera_idx)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_all_ppisp_reports(
    checkpoint_path: str,
    config_path: Optional[str],
    output_dir: str,
) -> None:
    """
    Generate PDF reports analyzing the PPISP model parameters for all cameras.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        config_path: Optional path to model config (will be loaded from checkpoint if not provided)
        output_dir: Directory where to save the generated reports
    """
    # Load checkpoint and create system
    if not config_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = OmegaConf.create(ckpt["hyper_parameters"])
    else:
        config = OmegaConf.load(config_path)

    # This prevents the model from initializing with the current datasource
    if config.resume is None:
        config.resume = str(checkpoint_path)

    typed_config = NREConfig.model_validate(config)
    system = nre.systems.make(config.system.name, typed_config, load_from_checkpoint=str(checkpoint_path))

    # Find PPISP post-processing module in the system
    ppisp_module = None
    for post_proc in system.model.post_processings:
        if isinstance(post_proc, PPISPPostProcessing):
            ppisp_module = post_proc
            break

    if ppisp_module is None:
        raise ValueError("No PPISP post-processing module found in system!")

    # Get camera IDs from config
    if not config.dataset.camera_ids:
        raise ValueError("No camera IDs found in config!")
    camera_ids = config.dataset.camera_ids
    if len(camera_ids) != ppisp_module.ppisp.num_cameras:
        raise ValueError(
            f"Number of camera IDs in config ({len(camera_ids)}) "
            f"doesn't match model's number of cameras ({ppisp_module.ppisp.num_cameras})"
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate report for each camera
    for camera_idx, camera_id in enumerate(camera_ids):
        camera_output_path = output_path / f"{camera_id}_ppisp_report.pdf"
        generate_camera_report(
            ppisp_model=ppisp_module.ppisp,
            camera_idx=camera_idx,
            output_path=camera_output_path,
        )

    # Extract and save all parameters to a single JSON file
    all_params = extract_ppisp_params(ppisp_module.ppisp)
    json_path = output_path / "ppisp_parameters.json"
    with open(json_path, "w") as f:
        json.dump(all_params, f, indent=2)

    logger.info(f"Saved all parameters to: {json_path}")


@click.command()
@click.option("--checkpoint", type=str, required=True, help="Path to model checkpoint")
@click.option("--config", type=str, required=True, help="Path to model config")
@click.option("--output-dir", required=True, type=str, help="Output directory for reports")
def main(
    checkpoint: str,
    config: Optional[str],
    output_dir: str,
) -> None:
    """Generate analysis reports for PPISP model parameters."""
    logging.basicConfig(level=logging.INFO)
    generate_all_ppisp_reports(checkpoint, config, output_dir)


if __name__ == "__main__":
    main()
