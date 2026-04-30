# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

from types import SimpleNamespace
from typing import cast

import omegaconf
import torch
import torch.nn as nn

from libs.losses.functional.cuda_losses_function import CudaLossesFunction
from libs.losses.kernel.constants import GRID_NUM_CHANNELS, GRID_NUM_COLS, GRID_NUM_ROWS
from libs.losses.models.loss_fns import total_variation_spatial
from libs.losses.orchestration.config import LossConfig
from libs.losses.orchestration.loss_aggregator import LossAggregator
from nre.config.model import SkyEnvMapBackgroundConfig
from nre.config.trainer import TrainerConfig
from nre.datasets.tracks import CuboidTracks
from nre.models.background import SkyEnvMapBackground
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.models.gaussians.gaussians_model import BaseGaussianModel, RigidGaussianModel
from nre.models.nn_extensions import TypedModuleDict, TypedModuleList
from nre.models.post_processing import BasePostProcessing, BilateralGridPerCamera, BilateralGridPerFrame
from nre.utils.batch import CameraFrameLabels, DataAndRenderingBatch, DataBatch, FrameMeta, LidarFrameLabels
from nre.utils.types import (
    ExtraSignal,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    RayFlags,
)


class MockTrainerConfig(TrainerConfig):
    def __init__(self):
        super().__init__(
            max_epochs=1,
            check_val_every_n_epoch=1,
            precision="32",
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )


class TestCudaLosses(unittest.TestCase):
    def test_cuda_simple_gradient(self):
        """Simple gradient test with minimal complexity to debug gradient flow."""

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        torch.manual_seed(42)

        # Simple 2x2 RGB test
        B, H, W, C = 1, 2, 2, 3
        rgb_pred = torch.randn(B, H, W, C, device=device, requires_grad=True)
        rgb_gt = torch.randn(B, H, W, C, device=device)
        rgb_flags = torch.full((B, H, W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32, device=device)

        # Simple 2x2 Lidar test
        lidar_pred = torch.randn(B, H, W, 1, device=device, requires_grad=True)
        lidar_gt = torch.randn(B, H, W, 1, device=device)
        lidar_flags = torch.zeros((B, H, W, 1), dtype=torch.int32, device=device)  # all valid

        # Simple 2x2 Background
        n_rays = B * H * W
        rgb_flags_flat = rgb_flags.view(-1)
        rgb_flags_flat[0] |= RayFlags.SKY_SEMANTIC.value
        sky_mask = (rgb_flags_flat & RayFlags.SKY_SEMANTIC.value) != 0
        bg_gt = torch.ones(n_rays, device=device)
        bg_gt[sky_mask] = 0.0  # Ground truth mask: 1 for non-sky (foreground), 0 for sky
        bg_pred = torch.rand(n_rays, device=device, requires_grad=True)

        # Simple 2x2 Background Lidar
        lidar_flags_flat = lidar_flags.view(-1)
        lidar_flags_flat[0] |= RayFlags.SKY_SEMANTIC.value
        sky_mask_lidar = (lidar_flags_flat & RayFlags.SKY_SEMANTIC.value) != 0
        bg_lidar_gt = torch.ones(n_rays, device=device)
        bg_lidar_gt[sky_mask_lidar] = 0.0  # Ground truth mask: 1 for non-sky (foreground), 0 for sky
        bg_lidar_pred = torch.rand(n_rays, device=device, requires_grad=True)

        # Compute valid counts manually
        rgb_valid = ((rgb_flags & RayFlags.RGB_LABEL.value) != 0) & ((rgb_flags & RayFlags.INVALID.value) == 0)
        n_valid_rgb = rgb_valid.sum().item()
        rgb_factor = 1.0 / n_valid_rgb if n_valid_rgb > 0 else 0.0

        lidar_valid = ((lidar_flags & RayFlags.INVALID.value) == 0) & ((lidar_flags & RayFlags.DROPPED.value) == 0)
        n_valid_lidar = lidar_valid.sum().item()
        lidar_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        bg_valid = (
            ((rgb_flags_flat & RayFlags.INVALID.value) == 0)
            & ((rgb_flags_flat & RayFlags.DIFIXED.value) == 0)
            & ((rgb_flags_flat & RayFlags.SYNTHETIC.value) == 0)
        )
        n_valid_bg_count = bg_valid.sum().item()
        bg_factor = 1.0 / n_valid_bg_count if n_valid_bg_count > 0 else 0.0

        bg_lidar_valid = ((lidar_flags_flat & RayFlags.INVALID.value) == 0) & (
            (lidar_flags_flat & RayFlags.DROPPED.value) == 0
        )
        n_valid_bg_lidar = bg_lidar_valid.sum().item()
        bg_lidar_factor = 1.0 / n_valid_bg_lidar if n_valid_bg_lidar > 0 else 0.0

        # Simple bilateral grid
        grid_per_camera = torch.randn(
            (1, GRID_NUM_CHANNELS, 2, 2, 2), dtype=torch.float32, device=device, requires_grad=True
        )
        grid_per_frame = torch.randn(
            (3, GRID_NUM_CHANNELS, 2, 2, 2), dtype=torch.float32, device=device, requires_grad=True
        )
        num_cam = (
            grid_per_camera.shape[0] * grid_per_camera.shape[2] * grid_per_camera.shape[3] * grid_per_camera.shape[4]
        )
        num_frame = (
            grid_per_frame.shape[0] * grid_per_frame.shape[2] * grid_per_frame.shape[3] * grid_per_frame.shape[4]
        )
        grid_per_camera_factor = 1.0 / num_cam if num_cam > 0 else 0.0
        grid_per_frame_factor = 1.0 / num_frame if num_frame > 0 else 0.0

        grid_camera_spatial_tv_factor = 1.0 / num_cam if num_cam > 0 else 0.0
        grid_frame_spatial_tv_factor = 1.0 / num_frame if num_frame > 0 else 0.0

        # Prepare flattened grids for CUDALossesFunction
        grids_cam = grid_per_camera.view(
            grid_per_camera.shape[0] * grid_per_camera.shape[1],
            grid_per_camera.shape[2],
            grid_per_camera.shape[3],
            grid_per_camera.shape[4],
        )
        grids_frame = grid_per_frame.view(
            grid_per_frame.shape[0] * grid_per_frame.shape[1],
            grid_per_frame.shape[2],
            grid_per_frame.shape[3],
            grid_per_frame.shape[4],
        )

        # Simple gaussian scales test (pre-activation values for exp() activation)
        N_scales = 10
        # Use randn() - 3 to keep scales small after exp()
        gaussian_scales = (torch.randn(N_scales, 3, device=device) - 3.0).requires_grad_(True)
        scale_factor = 1.0 / gaussian_scales.numel()

        # Sky-env-map cubemap texture with spatial variation (B=1, D=6, H=4, W=4, C=3)
        # D=6 for cubemap (6 faces), create random non-uniform texture to get non-zero TV loss and gradients
        bg_tex = torch.randn((1, 6, 4, 4, 3), device=device, dtype=torch.float32, requires_grad=True)
        bg_tex_factor = 1.0

        # Simple gaussian densities test
        N_densities = 15
        gaussian_densities = torch.randn(N_densities, device=device, requires_grad=True)
        density_factor = 1.0 / gaussian_densities.numel()

        # Prepare out_of_bound positions and cuboid dimensions
        out_of_bound_positions = torch.tensor([[0.2, -0.1, 0.25], [0.1, 0.2, 0.3]], device=device, requires_grad=True)
        out_of_bound_cuboid_dims = torch.tensor([[0.3, 0.25, 0.4], [0.2, 0.15, 0.3]], device=device)
        out_of_bound_factor = 1.0 / (out_of_bound_positions.shape[0] * out_of_bound_positions.shape[1])

        # Simple gaussian z-scales test (pre-activation values for exp() activation)
        N_z_scales = 8192
        # Use randn() - 3 to keep z-scales small after exp()
        # Pass full [N, 3] tensor, CUDA kernel will extract z-component
        gaussian_z_scales = (torch.randn(N_z_scales, 3, device=device) - 3.0).requires_grad_(True)
        z_scale_threshold = 0.01  # Threshold for ReLU (applied to exp(z_scale))
        z_scale_factor = 1.0 / N_z_scales  # Mean over N elements (not N*3)

        # Simple intensity test (enabled, uses same B,H,W as lidar)
        intensity_pred = torch.randn(B, H, W, 1, device=device, requires_grad=True)
        intensity_gt = torch.randn(B, H, W, 1, device=device)
        intensity_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Simple raydrop test (enabled, uses same B,H,W as lidar)
        raydrop_pred = torch.randn(B, H, W, 1, device=device, requires_grad=True)
        raydrop_gt = torch.randn(B, H, W, 1, device=device)
        raydrop_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Forward pass
        (
            rgb_loss,
            lidar_loss,
            bg_loss,
            bg_lidar_loss,
            grid_drift_loss,
            grid_camera_spatial_tv_loss,
            grid_frame_spatial_tv_loss,
            scale_loss,
            bg_tex_loss,
            density_loss,
            out_of_bound_loss,
            z_scale_loss,
            intensity_loss,
            raydrop_loss,
        ) = CudaLossesFunction.apply(
            rgb_flags,
            rgb_pred,
            rgb_gt,
            rgb_factor,
            lidar_flags,
            lidar_pred,
            lidar_gt,
            lidar_factor,
            intensity_pred,
            intensity_gt,
            intensity_factor,
            raydrop_pred,
            raydrop_gt,
            raydrop_factor,
            bg_pred,
            bg_factor,
            bg_lidar_pred,
            bg_lidar_factor,
            grids_cam,
            grids_frame,
            grid_per_camera_factor,
            grid_per_frame_factor,
            grid_camera_spatial_tv_factor,
            grid_frame_spatial_tv_factor,
            gaussian_scales,
            scale_factor,
            bg_tex,
            bg_tex_factor,
            gaussian_densities,
            density_factor,
            torch.ones(max(N_scales, N_densities), device=device, dtype=torch.float32),  # gaussian_visibility
            out_of_bound_positions,
            out_of_bound_cuboid_dims,
            out_of_bound_factor,
            gaussian_z_scales,
            z_scale_threshold,
            z_scale_factor,
        )

        # Backward pass - sum all losses to get scalar
        total_loss = (
            rgb_loss.sum()
            + lidar_loss.sum()
            + bg_loss.sum()
            + bg_lidar_loss.sum()
            + grid_drift_loss.sum()
            + grid_camera_spatial_tv_loss.sum()
            + grid_frame_spatial_tv_loss.sum()
            + scale_loss.sum()
            + bg_tex_loss.sum()
            + density_loss.sum()
            + out_of_bound_loss.sum()
            + z_scale_loss.sum()
            + intensity_loss.sum()
            + raydrop_loss.sum()
        )

        self.assertTrue(total_loss.requires_grad, "Total loss should require gradients")
        self.assertGreater(total_loss.item(), 0, "Total loss should be positive")

        total_loss.backward()

        # Check gradients
        self.assertIsNotNone(rgb_pred.grad, "RGB pred should have gradients")
        self.assertIsNotNone(lidar_pred.grad, "Lidar pred should have gradients")
        self.assertIsNotNone(intensity_pred.grad, "Intensity pred should have gradients")
        self.assertIsNotNone(raydrop_pred.grad, "Raydrop pred should have gradients")
        self.assertIsNotNone(bg_pred.grad, "Background pred should have gradients")
        self.assertIsNotNone(bg_lidar_pred.grad, "Background Lidar pred should have gradients")
        self.assertIsNotNone(grid_per_camera.grad, "grid_per_camera should have gradients")
        self.assertIsNotNone(grid_per_frame.grad, "grid_per_frame should have gradients")
        self.assertIsNotNone(gaussian_scales.grad, "gaussian_scales should have gradients")
        self.assertIsNotNone(bg_tex.grad, "bg_tex should have gradients")
        self.assertIsNotNone(gaussian_densities.grad, "gaussian_densities should have gradients")
        self.assertIsNotNone(out_of_bound_positions.grad, "out_of_bound positions should have gradients")
        self.assertIsNotNone(gaussian_z_scales.grad, "gaussian_z_scales should have gradients")

        # Gradients should be non-zero
        self.assertGreater(rgb_pred.grad.abs().sum().item(), 1e-8, "RGB gradients should be non-zero")
        self.assertGreater(lidar_pred.grad.abs().sum().item(), 1e-8, "Lidar gradients should be non-zero")
        self.assertGreater(bg_pred.grad.abs().sum().item(), 1e-8, "Background gradients should be non-zero")
        self.assertGreater(bg_lidar_pred.grad.abs().sum().item(), 1e-8, "Background Lidar gradients should be non-zero")
        self.assertGreater(
            grid_per_camera.grad.abs().sum().item(), 1e-8, "grid_per_camera gradients should be non-zero"
        )
        self.assertGreater(grid_per_frame.grad.abs().sum().item(), 1e-8, "grid_per_frame gradients should be non-zero")
        self.assertGreater(
            gaussian_scales.grad.abs().sum().item(), 1e-8, "gaussian_scales gradients should be non-zero"
        )
        self.assertGreater(bg_tex_loss.sum().item(), 0.0, "bg_tex_loss should be non-zero for non-uniform texture")
        self.assertGreater(
            gaussian_densities.grad.abs().sum().item(), 1e-8, "gaussian_densities gradients should be non-zero"
        )
        self.assertGreater(
            out_of_bound_positions.grad.abs().sum().item(), 1e-8, "out_of_bound gradients should be non-zero"
        )
        self.assertGreater(
            gaussian_z_scales.grad.abs().sum().item(), 1e-8, "gaussian_z_scales gradients should be non-zero"
        )
        self.assertGreater(intensity_pred.grad.abs().sum().item(), 1e-8, "Intensity gradients should be non-zero")
        self.assertGreater(raydrop_pred.grad.abs().sum().item(), 1e-8, "Raydrop gradients should be non-zero")

        # Manual gradient check for L1 loss: d/dx |x - gt| = sign(x - gt) * factor
        # For valid pixels only
        expected_rgb_grad = torch.zeros_like(rgb_pred)
        diff_rgb = rgb_pred - rgb_gt
        expected_rgb_grad[rgb_valid.expand_as(rgb_pred)] = (
            torch.sign(diff_rgb[rgb_valid.expand_as(rgb_pred)]) * rgb_factor
        )

        expected_lidar_grad = torch.zeros_like(lidar_pred)
        diff_lidar = lidar_pred - lidar_gt
        expected_lidar_grad[lidar_valid.expand_as(lidar_pred)] = (
            torch.sign(diff_lidar[lidar_valid.expand_as(lidar_pred)]) * lidar_factor
        )

        # Manual gradient check for MSE background: d/dx (p - g)^2 = 2*(p - g)*bg_factor
        expected_bg_grad = torch.zeros_like(bg_pred)
        diff_bg = bg_pred - bg_gt
        expected_bg_grad[bg_valid] = 2 * diff_bg[bg_valid] * bg_factor

        # Manual gradient check for MSE background Lidar: d/dx (p - g)^2 = 2*(p - g)*bg_lidar_factor
        expected_bg_lidar_grad = torch.zeros_like(bg_lidar_pred)
        diff_bg_lidar = bg_lidar_pred - bg_lidar_gt
        expected_bg_lidar_grad[bg_lidar_valid] = 2 * diff_bg_lidar[bg_lidar_valid] * bg_lidar_factor

        # Manual gradient check for bilateral grid drift loss

        # Camera grid expected gradients
        reshaped_cam = grid_per_camera.view(
            grid_per_camera.shape[0], GRID_NUM_ROWS, GRID_NUM_COLS, *grid_per_camera.shape[2:]
        )
        identity = torch.eye(GRID_NUM_ROWS, GRID_NUM_COLS, device=device).view(
            1, GRID_NUM_ROWS, GRID_NUM_COLS, *([1] * (reshaped_cam.dim() - 3))
        )
        diff_cam = reshaped_cam - identity
        norm_cam = torch.norm(diff_cam, p="fro", dim=(1, 2))
        norm_cam_exp = norm_cam.view(norm_cam.shape[0], 1, 1, norm_cam.shape[1], norm_cam.shape[2], norm_cam.shape[3])
        expected_cam_grad = (diff_cam / norm_cam_exp) * grid_per_camera_factor
        expected_cam_grad = expected_cam_grad.view_as(grid_per_camera)

        # Frame grid expected gradients
        reshaped_frame = grid_per_frame.view(
            grid_per_frame.shape[0], GRID_NUM_ROWS, GRID_NUM_COLS, *grid_per_frame.shape[2:]
        )
        # Reuse identity with frame dims: reshape identity to match reshaped_frame
        identity_frame = torch.eye(GRID_NUM_ROWS, GRID_NUM_COLS, device=device).view(
            1, GRID_NUM_ROWS, GRID_NUM_COLS, *([1] * (reshaped_frame.dim() - 3))
        )
        diff_frame = reshaped_frame - identity_frame
        norm_frame = torch.norm(diff_frame, p="fro", dim=(1, 2))
        norm_frame_exp = norm_frame.view(
            norm_frame.shape[0], 1, 1, norm_frame.shape[1], norm_frame.shape[2], norm_frame.shape[3]
        )
        expected_frame_grad = (diff_frame / norm_frame_exp) * grid_per_frame_factor
        expected_frame_grad = expected_frame_grad.view_as(grid_per_frame)

        # Manual gradient check for grid spatial tv loss
        def grid_tv_spatial_grad(x: torch.Tensor) -> torch.Tensor | None:
            x = x.detach().requires_grad_(True)
            x.grad = None
            tv_loss_python = total_variation_spatial(x)
            tv_loss_python.mean().backward()
            return x.grad

        expected_cam_grad += grid_tv_spatial_grad(grid_per_camera)
        expected_frame_grad += grid_tv_spatial_grad(grid_per_frame)

        # Manual gradient check for gaussian scale loss with exp() activation
        # Loss: mean(exp(x) + exp(y) + exp(z)) where mean is via scale_factor pre-multiplication
        # Since exp() is always positive, abs(exp(x)) = exp(x)
        # Gradient: d/dx_preact = exp(x_preact) * scale_factor
        expected_scale_grad = torch.exp(gaussian_scales) * scale_factor

        # Manual gradient check for out_of_bound loss
        expected_out_of_bound_grad = torch.zeros_like(out_of_bound_positions)
        detach_out_of_bound_positions = out_of_bound_positions.detach()
        abs_pos = detach_out_of_bound_positions.abs()
        losses = abs_pos - out_of_bound_cuboid_dims / 2
        mask = losses > 0  # Coordinates where ReLU(loss) is active
        # ReLU derivative is +/-1 for exceeded coords; scale by out_of_bound_factor
        expected_out_of_bound_grad[mask] = torch.sign(detach_out_of_bound_positions)[mask] * out_of_bound_factor

        # Manual gradient check for sky_env_map: permute to (B, C, D, H, W) and use total_variation_spatial
        bg_tex_permuted = bg_tex.detach().permute(0, 4, 1, 2, 3).requires_grad_(True)
        tv_loss_python = total_variation_spatial(bg_tex_permuted)
        (tv_loss_python.sum() * bg_tex_factor).backward()
        expected_bg_tex_grad = bg_tex_permuted.grad.permute(0, 2, 3, 4, 1)

        # Manual gradient check for gaussian density loss: d/dx |x| * factor = sign(x) * factor
        expected_density_grad = torch.sign(gaussian_densities) * density_factor

        # Compare gradients
        self.assertTrue(
            torch.allclose(rgb_pred.grad, expected_rgb_grad, atol=1e-6, rtol=1e-5),
            f"RGB gradients don't match expected. Max diff: {(rgb_pred.grad - expected_rgb_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(lidar_pred.grad, expected_lidar_grad, atol=1e-6, rtol=1e-5),
            f"Lidar gradients don't match expected. Max diff: {(lidar_pred.grad - expected_lidar_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(bg_pred.grad, expected_bg_grad, atol=1e-6, rtol=1e-5),
            f"Background gradients don't match expected. Max diff: {(bg_pred.grad - expected_bg_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(bg_lidar_pred.grad, expected_bg_lidar_grad, atol=1e-6, rtol=1e-5),
            f"Background Lidar gradients don't match expected. Max diff: {(bg_lidar_pred.grad - expected_bg_lidar_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(grid_per_frame.grad, expected_frame_grad, atol=1e-6, rtol=1e-5),
            f"Grid per frame gradients don't match expected. Max diff: {(grid_per_frame.grad - expected_frame_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(grid_per_camera.grad, expected_cam_grad, atol=1e-6, rtol=1e-5),
            f"Grid per camera gradients don't match expected. Max diff: {(grid_per_camera.grad - expected_cam_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(gaussian_scales.grad, expected_scale_grad, atol=1e-6, rtol=1e-5),
            f"Gaussian scales gradients don't match expected. Max diff: {(gaussian_scales.grad - expected_scale_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(bg_tex.grad, expected_bg_tex_grad, atol=1e-6, rtol=1e-5),
            f"bg_tex gradients don't match expected. Max diff: {(bg_tex.grad - expected_bg_tex_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(gaussian_densities.grad, expected_density_grad, atol=1e-6, rtol=1e-5),
            f"Gaussian densities gradients don't match expected. Max diff: {(gaussian_densities.grad - expected_density_grad).abs().max().item():.6e}",
        )
        self.assertTrue(
            torch.allclose(out_of_bound_positions.grad, expected_out_of_bound_grad, atol=1e-6, rtol=1e-5),
            f"out_of_bound gradients don't match expected. Max diff: {(out_of_bound_positions.grad - expected_out_of_bound_grad).abs().max().item():.6e}",
        )

    def test_cuda_vs_python_loss_forward_and_backward(self):
        # Random RGB and lidar test data with some invalid pixels
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        torch.manual_seed(42)

        # Use different dimensions for RGB and lidar to test flexibility
        B_rgb, H_rgb, W_rgb, C = 2, 120, 160, 3
        B_lidar, H_lidar, W_lidar = 3, 64, 180

        # ===== RGB data =====
        rgb_gt = torch.randn(B_rgb, H_rgb, W_rgb, C, device=device)
        rgb_pred_cuda = torch.randn(B_rgb, H_rgb, W_rgb, C, device=device, requires_grad=True)
        rgb_pred_py = rgb_pred_cuda.clone().detach().requires_grad_(True)

        # RGB flags: start valid, some pixels randomly marked INVALID and SKY_SEMANTIC
        rgb_flags = torch.full((B_rgb, H_rgb, W_rgb, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32, device=device)
        rgb_invalid_mask = torch.rand(B_rgb, H_rgb, W_rgb, 1, device=device) < 0.1
        rgb_sky_mask = torch.rand(B_rgb, H_rgb, W_rgb, 1, device=device) < 0.2
        rgb_flags[rgb_invalid_mask] |= RayFlags.INVALID.value
        rgb_flags[rgb_sky_mask] |= RayFlags.SKY_SEMANTIC.value

        # ===== Lidar data =====
        lidar_gt = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) * 100.0  # Random distances 0-100m
        lidar_pred_cuda = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) * 100.0
        lidar_pred_cuda.requires_grad_(True)
        lidar_pred_py = lidar_pred_cuda.clone().detach().requires_grad_(True)

        # Lidar flags: start with no flags (valid), some rays randomly marked INVALID or DROPPED or SKY_SEMANTIC
        lidar_flags = torch.zeros((B_lidar, H_lidar, W_lidar, 1), dtype=torch.int32, device=device)
        lidar_invalid_mask = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) < 0.05
        lidar_dropped_mask = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) < 0.05
        lidar_sky_mask = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device) < 0.2
        lidar_flags[lidar_invalid_mask] |= RayFlags.INVALID.value
        lidar_flags[lidar_dropped_mask] |= RayFlags.DROPPED.value
        lidar_flags[lidar_sky_mask] |= RayFlags.SKY_SEMANTIC.value

        # ===== Intensity data =====
        intensity_gt = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device)
        intensity_pred_cuda = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device).requires_grad_(True)
        intensity_pred_py = intensity_pred_cuda.clone().detach().requires_grad_(True)

        # ===== Raydrop data =====
        raydrop_gt = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device)
        raydrop_pred_cuda = torch.rand(B_lidar, H_lidar, W_lidar, 1, device=device).requires_grad_(True)
        raydrop_pred_py = raydrop_pred_cuda.clone().detach().requires_grad_(True)

        # Define loss config and trainer
        config = LossConfig.model_validate(
            {
                "rgb": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "lidar": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "background": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "background_lidar": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "sky_env_map_background": {"fn": "total_variation_spatial", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "bilateral_grid_drift": {"fn": "identity_distance", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "bilateral_grid_per_camera_tv": {
                    "fn": "total_variation_spatial",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                },
                "bilateral_grid_per_frame_spatial_tv": {
                    "fn": "total_variation_spatial",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                },
                "gaussian_scale": {
                    "fn": "abs",
                    "lambda_": 1.0,
                    "reduce": {"name": "mean"},
                    "layer_lambdas": {"road": 1.0},
                    "visibility_filter": True,
                },
                "gaussian_density": {
                    "fn": "abs",
                    "lambda_": 0.02,
                    "reduce": {"name": "mean"},
                    "layer_lambdas": {},
                    "visibility_filter": True,
                },
                "out_of_bound": {"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "intensity": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
                "raydrop": {"fn": "mse", "lambda_": 1.0, "reduce": {"name": "mean"}},
            }
        )
        trainer_config = MockTrainerConfig()

        # Mock GaussiansComposite to avoid actual model initialization
        class MockGaussiansComposite(GaussiansComposite):
            def __init__(
                self,
                *,
                gaussians_nodes: dict[str, BaseGaussianModel] | None = None,
                post_processings: list[BasePostProcessing] | None = None,
                **attrs,
            ):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.gaussians_nodes = TypedModuleDict[BaseGaussianModel](gaussians_nodes or {})
                self.post_processings = TypedModuleList[BasePostProcessing](post_processings or [])
                for key, value in attrs.items():
                    setattr(self, key, value)

            def get_gaussians_node_ids(self, non_empty_only: bool = False) -> list[str]:
                return list(self.gaussians_nodes.keys())

        # Mock BilateralGridPerCamera and BilateralGridPerFrame to avoid actual initialization
        class MockBilateralGridPerCamera(BilateralGridPerCamera):
            def __init__(self, **attrs):
                # Initialize nn.Module to allow setting module attributes
                nn.Module.__init__(self)
                for k, v in attrs.items():
                    setattr(self, k, v)

        class MockBilateralGridPerFrame(BilateralGridPerFrame):
            def __init__(self, **attrs):
                # Initialize nn.Module to allow setting module attributes
                nn.Module.__init__(self)
                for k, v in attrs.items():
                    setattr(self, k, v)

        # Mock Gaussian node with scales
        # Aligns with nre/models/gaussians/gaussians_model.py where preactivated values are stored
        class MockGaussianNode(BaseGaussianModel):
            def __init__(self, scales, densities):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                # Store pre-activation (log-space) values, aligning with gaussians_model.py
                self.scales = scales
                self.densities = densities

            def get_scales(self, preactivation=False) -> torch.Tensor:
                if preactivation:
                    # Return log-space values for CUDA (which will apply exp() in kernel)
                    return self.scales
                else:
                    # Return activated values for Python loss
                    return torch.exp(self.scales)

            def get_densities(self, preactivation=False):
                return self.densities

        class MockRigidGaussianModel(RigidGaussianModel):
            def __init__(self, positions: torch.Tensor, cuboid_dims: torch.Tensor, cuboid_ids: torch.Tensor):
                # intentionally skip super().__init__()
                torch.nn.Module.__init__(self)
                self.positions = torch.nn.Parameter(positions)
                self.cuboid_tracks = cast(CuboidTracks, SimpleNamespace(cuboids_dims=cuboid_dims))
                self.gaussian_cuboid_ids = nn.Buffer(cuboid_ids)

            def get_positions(self) -> torch.nn.Parameter:
                return cast(torch.nn.Parameter, self.positions)

            def get_num_gaussians(self) -> int:
                return self.positions.shape[0]

            def _get_zeros(self, *extra_dims: int) -> torch.Tensor:
                return torch.zeros(
                    self.positions.shape[0],
                    *extra_dims,
                    device=self.positions.device,
                    dtype=self.positions.dtype,
                )

            def get_scales(self, preactivation=False) -> torch.Tensor:
                return self._get_zeros(3)

            def get_densities(self, preactivation=False):
                return self._get_zeros()

        # ===== Rigid Gaussian data for out of bound loss =====
        positions_out_base = torch.tensor(
            [[0.4, -0.2, 0.15], [-0.35, 0.5, -0.25], [0.1, 0.05, -0.45]],
            dtype=torch.float32,
            device=device,
        )
        cuboid_dims_out = torch.tensor(
            [[0.6, 0.5, 0.4], [0.7, 0.9, 0.8], [0.2, 0.3, 0.9]],
            dtype=torch.float32,
            device=device,
        )
        gaussian_cuboid_ids = torch.tensor([2, 0, 1], dtype=torch.long, device=device)

        positions_out_cuda = positions_out_base.clone().detach().requires_grad_(True)
        positions_out_py = positions_out_base.clone().detach().requires_grad_(True)
        node_cuda = MockRigidGaussianModel(positions_out_cuda, cuboid_dims_out, gaussian_cuboid_ids)
        node_py = MockRigidGaussianModel(positions_out_py, cuboid_dims_out.clone().detach(), gaussian_cuboid_ids)

        # ===== Test CUDA implementation =====
        N_gaussians = 500
        n_rays_rgb = B_rgb * H_rgb * W_rgb
        n_rays_lidar = B_lidar * H_lidar * W_lidar
        gaussian_vis_mask = (torch.rand(N_gaussians + 3, device=device) > 0.3).float()
        # Make opacity require grad for background loss
        opacity_cuda = torch.ones(n_rays_rgb, device=device, requires_grad=True)
        rendered_cam_cuda = GaussiansRenderReturn(
            rgb=rgb_pred_cuda.reshape(-1, C),
            opacity=opacity_cuda,
            distance=torch.zeros(n_rays_rgb, device=device),
            visibility=gaussian_vis_mask,
        )
        # Make opacity require grad for background lidar loss
        opacity_lidar_cuda = torch.ones(n_rays_lidar, device=device, requires_grad=True)
        rendered_lidar_cuda = GaussiansRenderReturn(
            rgb=torch.zeros(n_rays_lidar, 3, device=device),
            opacity=opacity_lidar_cuda,
            distance=lidar_pred_cuda.reshape(-1, 1),
            extra_ray_signals=ExtraSignal(
                intensity=intensity_pred_cuda.reshape(-1, 1),
                raydrop=raydrop_pred_cuda.reshape(-1, 1),
            ),
        )
        results_cuda = GaussiansCompositeReturn(rendered_cam=rendered_cam_cuda, rendered_lidar=rendered_lidar_cuda)

        camera_labels_cuda = CameraFrameLabels(flags=rgb_flags, rgb=rgb_gt)
        camera_cuda = DataBatch.Camera(
            meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)], labels=camera_labels_cuda
        )

        lidar_labels_cuda = LidarFrameLabels(
            flags=lidar_flags, distance=lidar_gt, intensity=intensity_gt, raydrop=raydrop_gt
        )
        lidar_cuda = DataBatch.Lidar(
            meta=[FrameMeta(unique_sensor_idx=1, unique_frame_idx=0)], labels=lidar_labels_cuda
        )

        data_batch_cuda = DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], camera=camera_cuda, lidar=lidar_cuda)
        batch_cuda = DataAndRenderingBatch(data=data_batch_cuda)

        bilateral_grid_per_camera_cuda = torch.randn(
            (1, GRID_NUM_CHANNELS, 6, 8, 8), dtype=torch.float32, device=device, requires_grad=True
        )
        bilateral_grid_per_frame_cuda = torch.randn(
            (29, GRID_NUM_CHANNELS, 8, 16, 16), dtype=torch.float32, device=device, requires_grad=True
        )

        # Mock Gaussian scales for gaussian_scale loss
        gaussian_scales_cuda = torch.randn(N_gaussians, 3, device=device) * 0.1
        gaussian_scales_cuda = gaussian_scales_cuda.detach().requires_grad_(True)

        # Mock Gaussian densities for gaussian_density loss
        gaussian_densities_cuda = torch.randn(N_gaussians, device=device)
        gaussian_densities_cuda = gaussian_densities_cuda.detach().requires_grad_(True)

        bg_cfg = SkyEnvMapBackgroundConfig.model_validate(
            {
                "name": "sky-env-map",
                "envmap_type": "equirectangular",
                "height": H_rgb,
                "width": W_rgb,
                "saturate_radiance": False,
                "should_inpaint": False,
                "inpaint_threshold": 0.0,
                "inpaint_kernel_size": 5,
                "min_grad_updates": 0,
                "composite_in_linear_space": False,
            }
        )

        model_cuda = MockGaussiansComposite(
            gaussians_nodes={
                "road": MockGaussianNode(gaussian_scales_cuda, gaussian_densities_cuda),
                "rigid": node_cuda,
            },
            post_processings=[
                MockBilateralGridPerCamera(bilateral_grid=SimpleNamespace(grid=bilateral_grid_per_camera_cuda)),
                MockBilateralGridPerFrame(bilateral_grid=SimpleNamespace(grid=bilateral_grid_per_frame_cuda)),
            ],
            background=SkyEnvMapBackground(bg_cfg, trainer_config),
        )

        model_cuda = model_cuda.to(device)

        # Compute forward with CUDA losses enabled
        assert config is not None
        aggregator_cuda = LossAggregator(config, trainer_config)
        agg_ret_cuda = aggregator_cuda(step=0, model=model_cuda, results=results_cuda, target=batch_cuda)
        loss_cuda = agg_ret_cuda.total_value

        # Ensure loss requires grad
        self.assertTrue(loss_cuda.requires_grad, "CUDAloss should require gradients")
        self.assertGreater(loss_cuda.item(), 0, "CUDAloss should be positive")

        # Compute backward for CUDA
        loss_cuda.backward()
        rgb_grad_cuda = rgb_pred_cuda.grad.clone()
        lidar_grad_cuda = lidar_pred_cuda.grad.clone()
        bg_grad_cuda = opacity_cuda.grad.clone()
        bg_lidar_grad_cuda = opacity_lidar_cuda.grad.clone()
        per_camera_grad_cuda = bilateral_grid_per_camera_cuda.grad.clone()
        per_frame_grad_cuda = bilateral_grid_per_frame_cuda.grad.clone()
        gaussian_scales_grad_cuda = gaussian_scales_cuda.grad.clone()
        gaussian_densities_grad_cuda = gaussian_densities_cuda.grad.clone()
        out_of_bound_grad_cuda = node_cuda.positions.grad.clone()
        intensity_grad_cuda = intensity_pred_cuda.grad.clone()
        raydrop_grad_cuda = raydrop_pred_cuda.grad.clone()

        # Capture individual loss returns for forward comparisons
        out_of_bound_key = "out_of_bound_l1_mean"
        self.assertIn(out_of_bound_key, agg_ret_cuda.loss_returns)
        out_of_bound_loss_cuda = agg_ret_cuda.loss_returns[out_of_bound_key].reduced_value.detach()

        intensity_key = "intensity_mse_mean"
        self.assertIn(intensity_key, agg_ret_cuda.loss_returns)
        intensity_loss_cuda = agg_ret_cuda.loss_returns[intensity_key].reduced_value.detach()

        raydrop_key = "raydrop_mse_mean"
        self.assertIn(raydrop_key, agg_ret_cuda.loss_returns)
        raydrop_loss_cuda = agg_ret_cuda.loss_returns[raydrop_key].reduced_value.detach()

        # ===== Test Python implementation =====
        # Make opacity require grad for background loss
        opacity_py = torch.ones(n_rays_rgb, device=device, requires_grad=True)
        rendered_cam_py = GaussiansRenderReturn(
            rgb=rgb_pred_py.reshape(-1, C),
            opacity=opacity_py,
            distance=torch.zeros(n_rays_rgb, device=device),
            visibility=gaussian_vis_mask,
        )
        # Make opacity require grad for background lidar loss
        opacity_lidar_py = torch.ones(n_rays_lidar, device=device, requires_grad=True)
        rendered_lidar_py = GaussiansRenderReturn(
            rgb=torch.zeros(n_rays_lidar, 3, device=device),
            opacity=opacity_lidar_py,
            distance=lidar_pred_py.reshape(-1, 1),
            extra_ray_signals=ExtraSignal(
                intensity=intensity_pred_py.reshape(-1, 1),
                raydrop=raydrop_pred_py.reshape(-1, 1),
            ),
        )
        results_py = GaussiansCompositeReturn(rendered_cam=rendered_cam_py, rendered_lidar=rendered_lidar_py)

        camera_labels_py = CameraFrameLabels(flags=rgb_flags, rgb=rgb_gt)
        camera_py = DataBatch.Camera(meta=[FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)], labels=camera_labels_py)

        lidar_labels_py = LidarFrameLabels(
            flags=lidar_flags, distance=lidar_gt, intensity=intensity_gt, raydrop=raydrop_gt
        )
        lidar_py = DataBatch.Lidar(meta=[FrameMeta(unique_sensor_idx=1, unique_frame_idx=0)], labels=lidar_labels_py)

        data_batch_py = DataBatch(idx=0, worker_id=None, sequence_id=["dummy"], camera=camera_py, lidar=lidar_py)
        batch_py = DataAndRenderingBatch(data=data_batch_py)

        bilateral_grid_per_camera_py = bilateral_grid_per_camera_cuda.clone().detach().requires_grad_(True)
        bilateral_grid_per_frame_py = bilateral_grid_per_frame_cuda.clone().detach().requires_grad_(True)

        # Mock Gaussian scales for Python (clone from CUDA to ensure same input)
        gaussian_scales_py = gaussian_scales_cuda.clone().detach().requires_grad_(True)

        # Mock Gaussian densities for Python (clone from CUDA to ensure same input)
        gaussian_densities_py = gaussian_densities_cuda.clone().detach().requires_grad_(True)

        model_py = MockGaussiansComposite(
            gaussians_nodes={"road": MockGaussianNode(gaussian_scales_py, gaussian_densities_py), "rigid": node_py},
            post_processings=[
                MockBilateralGridPerCamera(bilateral_grid=SimpleNamespace(grid=bilateral_grid_per_camera_py)),
                MockBilateralGridPerFrame(bilateral_grid=SimpleNamespace(grid=bilateral_grid_per_frame_py)),
            ],
            background=SkyEnvMapBackground(bg_cfg, trainer_config),
        )
        model_py = model_py.to(device)

        # Instantiate with force_disable_cuda to bypass CUDA fused losses
        assert config is not None
        aggregator_py = LossAggregator(config, trainer_config, force_disable_cuda=True)
        agg_ret_py = aggregator_py(step=0, model=model_py, results=results_py, target=batch_py)
        loss_py = agg_ret_py.total_value

        # Ensure loss requires grad
        self.assertTrue(loss_py.requires_grad, "Python loss should require gradients")
        self.assertGreater(loss_py.item(), 0, "Python loss should be positive")

        # Compute backward for Python
        loss_py.backward()
        rgb_grad_py = rgb_pred_py.grad.clone()
        lidar_grad_py = lidar_pred_py.grad.clone()
        bg_grad_py = opacity_py.grad.clone()
        bg_lidar_grad_py = opacity_lidar_py.grad.clone()
        per_camera_grad_py = bilateral_grid_per_camera_py.grad.clone()
        per_frame_grad_py = bilateral_grid_per_frame_py.grad.clone()
        gaussian_scales_grad_py = gaussian_scales_py.grad.clone()
        gaussian_densities_grad_py = gaussian_densities_py.grad.clone()
        out_of_bound_grad_py = node_py.positions.grad.clone()
        intensity_grad_py = intensity_pred_py.grad.clone()
        raydrop_grad_py = raydrop_pred_py.grad.clone()

        self.assertIn(out_of_bound_key, agg_ret_py.loss_returns)
        out_of_bound_loss_py = agg_ret_py.loss_returns[out_of_bound_key].reduced_value.detach()

        self.assertIn(intensity_key, agg_ret_py.loss_returns)
        intensity_loss_py = agg_ret_py.loss_returns[intensity_key].reduced_value.detach()

        self.assertIn(raydrop_key, agg_ret_py.loss_returns)
        raydrop_loss_py = agg_ret_py.loss_returns[raydrop_key].reduced_value.detach()

        # ===== Compare forward results =====
        print(f"Forward: CUDA={loss_cuda.item():.6f}, Python={loss_py.item():.6f}")
        forward_rel_diff = abs(loss_cuda.item() - loss_py.item()) / (abs(loss_py.item()) + 1e-8)
        self.assertTrue(
            torch.allclose(loss_cuda, loss_py, atol=1e-4, rtol=5e-3),
            f"Forward pass mismatch: CUDA={loss_cuda.item()}, Python={loss_py.item()}, rel_diff={forward_rel_diff:.6e}",
        )

        # ===== Compare RGB backward results =====
        rgb_cuda_nonzero = rgb_grad_cuda.abs().gt(1e-8).sum().item()
        rgb_py_nonzero = rgb_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(rgb_cuda_nonzero, 0, "CUDARGB gradients should be non-zero")
        self.assertGreater(rgb_py_nonzero, 0, "Python RGB gradients should be non-zero")

        rgb_grad_max_abs_diff = (rgb_grad_cuda - rgb_grad_py).abs().max().item()
        print(f"Backward RGB: nonzero={rgb_cuda_nonzero}/{rgb_grad_cuda.numel()}, max_diff={rgb_grad_max_abs_diff:.6e}")
        self.assertTrue(
            torch.allclose(rgb_grad_cuda, rgb_grad_py, atol=1e-6, rtol=1e-8),
            f"RGB Backward pass mismatch: max_abs_diff={rgb_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Lidar backward results =====
        lidar_cuda_nonzero = lidar_grad_cuda.abs().gt(1e-8).sum().item()
        lidar_py_nonzero = lidar_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(lidar_cuda_nonzero, 0, "CUDALidar gradients should be non-zero")
        self.assertGreater(lidar_py_nonzero, 0, "Python Lidar gradients should be non-zero")

        lidar_grad_max_abs_diff = (lidar_grad_cuda - lidar_grad_py).abs().max().item()
        print(
            f"Backward Lidar: nonzero={lidar_cuda_nonzero}/{lidar_grad_cuda.numel()}, max_diff={lidar_grad_max_abs_diff:.6e}"
        )
        # Allow larger tolerance for Lidar backward differences
        self.assertTrue(
            torch.allclose(lidar_grad_cuda, lidar_grad_py, atol=1e-4, rtol=1e-2),
            f"Lidar Backward pass mismatch: max_abs_diff={lidar_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Background backward results =====
        bg_cuda_nonzero = bg_grad_cuda.abs().gt(1e-8).sum().item()
        bg_py_nonzero = bg_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(bg_cuda_nonzero, 0, "CUDABackground gradients should be non-zero")
        self.assertGreater(bg_py_nonzero, 0, "Python Background gradients should be non-zero")

        bg_grad_max_abs_diff = (bg_grad_cuda - bg_grad_py).abs().max().item()
        print(
            f"Backward Background: nonzero={bg_cuda_nonzero}/{bg_grad_cuda.numel()}, max_diff={bg_grad_max_abs_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(bg_grad_cuda, bg_grad_py, atol=1e-6, rtol=1e-8),
            f"Background Backward pass mismatch: max_abs_diff={bg_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Background Lidar backward results =====
        bg_lidar_cuda_nonzero = bg_lidar_grad_cuda.abs().gt(1e-8).sum().item()
        bg_lidar_py_nonzero = bg_lidar_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(bg_lidar_cuda_nonzero, 0, "CUDABackground Lidar gradients should be non-zero")
        self.assertGreater(bg_lidar_py_nonzero, 0, "Python Background Lidar gradients should be non-zero")

        bg_lidar_grad_max_abs_diff = (bg_lidar_grad_cuda - bg_lidar_grad_py).abs().max().item()
        print(
            f"Backward Background Lidar: nonzero={bg_lidar_cuda_nonzero}/{bg_lidar_grad_cuda.numel()}, max_diff={bg_lidar_grad_max_abs_diff:.6e}"
        )
        # Allow larger tolerance for Background Lidar backward differences
        self.assertTrue(
            torch.allclose(bg_lidar_grad_cuda, bg_lidar_grad_py, atol=1e-4, rtol=1e-2),
            f"Background Lidar Backward pass mismatch: max_abs_diff={bg_lidar_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Bilateral Grid backward results (Potentially contributed by Grid Dift & Spatial TV) =====
        per_camera_grad_max_abs_diff = (per_camera_grad_cuda - per_camera_grad_py).abs().max().item()
        per_frame_grad_max_abs_diff = (per_frame_grad_cuda - per_frame_grad_py).abs().max().item()
        # Allow some tolerance for grid drift gradient differences
        self.assertTrue(
            torch.allclose(per_camera_grad_cuda, per_camera_grad_py, atol=1e-4, rtol=1e-2),
            f"Bilateral Grid Drift Per Camera backward pass mismatch: max_abs_diff={per_camera_grad_max_abs_diff:.6e}",
        )
        self.assertTrue(
            torch.allclose(per_frame_grad_cuda, per_frame_grad_py, atol=1e-4, rtol=1e-2),
            f"Bilateral Grid Drift Per Frame backward pass mismatch: max_abs_diff={per_frame_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Gaussian Scale backward results =====
        gaussian_scales_cuda_nonzero = gaussian_scales_grad_cuda.abs().gt(1e-8).sum().item()
        gaussian_scales_py_nonzero = gaussian_scales_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(gaussian_scales_cuda_nonzero, 0, "CUDAGaussian scales gradients should be non-zero")
        self.assertGreater(gaussian_scales_py_nonzero, 0, "Python Gaussian scales gradients should be non-zero")

        gaussian_scales_grad_max_abs_diff = (gaussian_scales_grad_cuda - gaussian_scales_grad_py).abs().max().item()
        self.assertTrue(
            torch.allclose(gaussian_scales_grad_cuda, gaussian_scales_grad_py, atol=1e-6, rtol=1e-8),
            f"Gaussian Scales Backward pass mismatch: max_abs_diff={gaussian_scales_grad_max_abs_diff:.6e}",
        )

        # Compare sky-env-map spatial TV loss between CUDA and Python
        bg_tex_cuda = agg_ret_cuda.loss_returns["sky_env_map_background_total_variation_spatial_mean"].reduced_value
        bg_tex_py = agg_ret_py.loss_returns["sky_env_map_background_total_variation_spatial_mean"].reduced_value
        self.assertTrue(
            torch.allclose(bg_tex_cuda, bg_tex_py, atol=1e-6, rtol=1e-6),
            f"sky_env_map_background TV loss mismatch: CUDA={bg_tex_cuda.item()}, Python={bg_tex_py.item()}",
        )

        # ===== Compare Gaussian Density backward results =====
        gaussian_densities_cuda_nonzero = gaussian_densities_grad_cuda.abs().gt(1e-8).sum().item()
        gaussian_densities_py_nonzero = gaussian_densities_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(gaussian_densities_cuda_nonzero, 0, "CUDAGaussian densities gradients should be non-zero")
        self.assertGreater(gaussian_densities_py_nonzero, 0, "Python Gaussian densities gradients should be non-zero")

        gaussian_densities_grad_max_abs_diff = (
            (gaussian_densities_grad_cuda - gaussian_densities_grad_py).abs().max().item()
        )
        print(
            f"Backward Gaussian Densities: nonzero={gaussian_densities_cuda_nonzero}/{gaussian_densities_grad_cuda.numel()}, max_diff={gaussian_densities_grad_max_abs_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(gaussian_densities_grad_cuda, gaussian_densities_grad_py, atol=1e-6, rtol=1e-8),
            f"Gaussian Densities Backward pass mismatch: max_abs_diff={gaussian_densities_grad_max_abs_diff:.6e}",
        )

        # ===== Compare out_of_bound forward and backward results =====
        print(
            f"Forward out_of_bound: CUDA={out_of_bound_loss_cuda.item():.6f}, Python={out_of_bound_loss_py.item():.6f}"
        )
        self.assertTrue(
            torch.allclose(out_of_bound_loss_cuda, out_of_bound_loss_py, atol=1e-4, rtol=5e-3),
            f"out_of_bound forward mismatch: CUDA={out_of_bound_loss_cuda.item():.6f}, Python={out_of_bound_loss_py.item():.6f}",
        )

        out_of_bound_cuda_nonzero = out_of_bound_grad_cuda.abs().gt(1e-8).sum().item()
        out_of_bound_py_nonzero = out_of_bound_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(out_of_bound_cuda_nonzero, 0, "CUDAout_of_bound gradients should be non-zero")
        self.assertGreater(out_of_bound_py_nonzero, 0, "Python out_of_bound gradients should be non-zero")

        out_of_bound_grad_max_abs_diff = (out_of_bound_grad_cuda - out_of_bound_grad_py).abs().max().item()
        print(f"Backward out_of_bound: CUDA={out_of_bound_grad_cuda}, Python={out_of_bound_grad_py}")
        print(
            f"Backward out_of_bound: nonzero={out_of_bound_cuda_nonzero}/{out_of_bound_grad_cuda.numel()}, max_diff={out_of_bound_grad_max_abs_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(out_of_bound_grad_cuda, out_of_bound_grad_py, atol=1e-6, rtol=1e-8),
            f"out_of_bound backward mismatch: max_abs_diff={out_of_bound_grad_max_abs_diff:.6e}",
        )

        # ===== Compare Intensity forward results =====
        print(f"Forward intensity: CUDA={intensity_loss_cuda.item():.6f}, Python={intensity_loss_py.item():.6f}")
        self.assertTrue(
            torch.allclose(intensity_loss_cuda, intensity_loss_py, atol=1e-4, rtol=5e-3),
            f"Intensity forward mismatch: CUDA={intensity_loss_cuda.item():.6f}, Python={intensity_loss_py.item():.6f}",
        )

        # ===== Compare Raydrop forward results =====
        print(f"Forward raydrop: CUDA={raydrop_loss_cuda.item():.6f}, Python={raydrop_loss_py.item():.6f}")
        self.assertTrue(
            torch.allclose(raydrop_loss_cuda, raydrop_loss_py, atol=1e-4, rtol=5e-3),
            f"Raydrop forward mismatch: CUDA={raydrop_loss_cuda.item():.6f}, Python={raydrop_loss_py.item():.6f}",
        )

        # ===== Compare Intensity backward results =====
        intensity_cuda_nonzero = intensity_grad_cuda.abs().gt(1e-8).sum().item()
        intensity_py_nonzero = intensity_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(intensity_cuda_nonzero, 0, "CUDAIntensity gradients should be non-zero")
        self.assertGreater(intensity_py_nonzero, 0, "Python Intensity gradients should be non-zero")

        intensity_grad_max_diff = (intensity_grad_cuda - intensity_grad_py).abs().max().item()
        print(
            f"Backward Intensity: nonzero={intensity_cuda_nonzero}/{intensity_grad_cuda.numel()}, max_diff={intensity_grad_max_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(intensity_grad_cuda, intensity_grad_py, atol=1e-4, rtol=1e-2),
            f"Intensity Backward mismatch: max_abs_diff={intensity_grad_max_diff:.6e}",
        )

        # ===== Compare Raydrop backward results =====
        raydrop_cuda_nonzero = raydrop_grad_cuda.abs().gt(1e-8).sum().item()
        raydrop_py_nonzero = raydrop_grad_py.abs().gt(1e-8).sum().item()
        self.assertGreater(raydrop_cuda_nonzero, 0, "CUDARaydrop gradients should be non-zero")
        self.assertGreater(raydrop_py_nonzero, 0, "Python Raydrop gradients should be non-zero")

        raydrop_grad_max_diff = (raydrop_grad_cuda - raydrop_grad_py).abs().max().item()
        print(
            f"Backward Raydrop: nonzero={raydrop_cuda_nonzero}/{raydrop_grad_cuda.numel()}, max_diff={raydrop_grad_max_diff:.6e}"
        )
        self.assertTrue(
            torch.allclose(raydrop_grad_cuda, raydrop_grad_py, atol=1e-4, rtol=1e-2),
            f"Raydrop Backward mismatch: max_abs_diff={raydrop_grad_max_diff:.6e}",
        )
