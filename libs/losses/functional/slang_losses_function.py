# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Layer 1: Slang losses autograd function

This module contains autograd functions that wrap Slang loss kernels with PyTorch's autograd system.
"""

import torch

from libs.losses.kernel import slang_losses
from libs.losses.kernel.constants import BLOCK_THREADS, GRID_NUM_CHANNELS
from libs.slang_utils.utils import div_up


class SlangLossesFunction(torch.autograd.Function):
    """
    Stateless class for all Slang losses custom autograd function

    Layer 1 as explained in the docs/architecture/modules-losses.md file.
    """

    @staticmethod
    def forward(
        ctx,
        rgb_flags: torch.Tensor,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        rgb_factor: float,
        lidar_flags: torch.Tensor,
        lidar_pred: torch.Tensor,
        lidar_gt: torch.Tensor,
        lidar_factor: float,
        intensity_pred: torch.Tensor,
        intensity_gt: torch.Tensor,
        intensity_factor: float,
        raydrop_pred: torch.Tensor,
        raydrop_gt: torch.Tensor,
        raydrop_factor: float,
        bg_pred: torch.Tensor,
        bg_factor: float,
        bg_lidar_pred: torch.Tensor,
        bg_lidar_factor: float,
        grids_per_camera: torch.Tensor,
        grids_per_frame: torch.Tensor,
        grid_drift_per_camera_factor: float,
        grid_drift_per_frame_factor: float,
        grid_camera_spatial_tv_factor: float,
        grid_frame_spatial_tv_factor: float,
        gaussian_scales: torch.Tensor,
        scale_factor: float,
        bg_tex: torch.Tensor,
        bg_tex_factor: float,
        gaussian_densities: torch.Tensor,
        density_factor: float,
        gaussian_visibility: torch.Tensor,
        out_of_bound_positions: torch.Tensor,
        out_of_bound_cuboid_dims: torch.Tensor,
        out_of_bound_factor: float,
        gaussian_z_scales: torch.Tensor,
        z_scale_threshold: float,
        z_scale_factor: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Assert input shapes and types
        # Cf. rgb_flags, rgb_pred, rgb_gt in nre/utils/batch.py flags, rgb in CameraFrameLabels, DataAndRenderingBatch
        #     lidar_flags, lidar_pred, lidar_gt in nre/utils/batch.py flags, distance in LidarFrameLabels, GaussiansRenderReturn
        #     bg_pred, bg_lidar_pred in nre/utils/types.py opacity in GaussiansRenderReturn, VolumeRenderingReturn
        assert rgb_flags.ndim == 4 and rgb_flags.shape[3] == 1 and rgb_flags.dtype == torch.int32, (
            f"rgb_flags must be int32 Tensor of shape [B,H,W,1], got shape={rgb_flags.shape}, dtype={rgb_flags.dtype}"
        )
        assert rgb_pred.ndim == 4 and rgb_pred.shape[3] == 3 and rgb_pred.dtype == torch.float32, (
            f"rgb_pred must be float32 Tensor of shape [B,H,W,3], got shape={rgb_pred.shape}, dtype={rgb_pred.dtype}"
        )
        assert rgb_gt.ndim == 4 and rgb_gt.shape[3] == 3 and rgb_gt.dtype == torch.float32, (
            f"rgb_gt must be float32 Tensor of shape [B,H,W,3], got shape={rgb_gt.shape}, dtype={rgb_gt.dtype}"
        )
        assert lidar_flags.ndim == 4 and lidar_flags.shape[3] == 1 and lidar_flags.dtype == torch.int32, (
            f"lidar_flags must be int32 Tensor of shape [B,H,W,1], got shape={lidar_flags.shape}, dtype={lidar_flags.dtype}"
        )
        assert lidar_pred.ndim == 4 and lidar_pred.shape[3] == 1 and lidar_pred.dtype == torch.float32, (
            f"lidar_pred must be float32 Tensor of shape [B,H,W,1], got shape={lidar_pred.shape}, dtype={lidar_pred.dtype}"
        )
        assert lidar_gt.ndim == 4 and lidar_gt.shape[3] == 1 and lidar_gt.dtype == torch.float32, (
            f"lidar_gt must be float32 Tensor of shape [B,H,W,1], got shape={lidar_gt.shape}, dtype={lidar_gt.dtype}"
        )
        assert intensity_pred.ndim == 4 and intensity_pred.shape[3] == 1 and intensity_pred.dtype == torch.float32, (
            f"intensity_pred must be float32 Tensor of shape [B,H,W,1], got shape={intensity_pred.shape}, dtype={intensity_pred.dtype}"
        )
        assert intensity_gt.ndim == 4 and intensity_gt.shape[3] == 1 and intensity_gt.dtype == torch.float32, (
            f"intensity_gt must be float32 Tensor of shape [B,H,W,1], got shape={intensity_gt.shape}, dtype={intensity_gt.dtype}"
        )
        assert raydrop_pred.ndim == 4 and raydrop_pred.shape[3] == 1 and raydrop_pred.dtype == torch.float32, (
            f"raydrop_pred must be float32 Tensor of shape [B,H,W,1], got shape={raydrop_pred.shape}, dtype={raydrop_pred.dtype}"
        )
        assert raydrop_gt.ndim == 4 and raydrop_gt.shape[3] == 1 and raydrop_gt.dtype == torch.float32, (
            f"raydrop_gt must be float32 Tensor of shape [B,H,W,1], got shape={raydrop_gt.shape}, dtype={raydrop_gt.dtype}"
        )
        assert bg_pred.ndim == 1 and bg_pred.dtype == torch.float32, (
            f"bg_pred must be float32 Tensor of shape [N_rays], got shape={bg_pred.shape}, dtype={bg_pred.dtype}"
        )
        assert bg_lidar_pred.ndim == 1 and bg_lidar_pred.dtype == torch.float32, (
            f"bg_lidar_pred must be float32 Tensor of shape [N_rays_lidar], got shape={bg_lidar_pred.shape}, dtype={bg_lidar_pred.dtype}"
        )
        assert (
            grids_per_camera.ndim == 4
            and grids_per_camera.dtype == torch.float32
            and grids_per_camera.shape[0] % GRID_NUM_CHANNELS == 0
        ), (
            f"grids_per_camera must be float32 Tensor [B*C,D,H,W] with C={GRID_NUM_CHANNELS}, got shape={grids_per_camera.shape}, dtype={grids_per_camera.dtype}"
        )
        assert (
            grids_per_frame.ndim == 4
            and grids_per_frame.dtype == torch.float32
            and grids_per_frame.shape[0] % GRID_NUM_CHANNELS == 0
        ), (
            f"grids_per_frame must be float32 Tensor [B*C,D,H,W] with C={GRID_NUM_CHANNELS}, got shape={grids_per_frame.shape}, dtype={grids_per_frame.dtype}"
        )
        assert gaussian_scales.ndim == 2 and gaussian_scales.shape[1] == 3 and gaussian_scales.dtype == torch.float32, (
            f"gaussian_scales must be float32 Tensor of shape [N, 3], got shape={gaussian_scales.shape}, dtype={gaussian_scales.dtype}"
        )
        assert (
            out_of_bound_positions.ndim == 2
            and out_of_bound_positions.shape[1] == 3
            and out_of_bound_positions.dtype == torch.float32
        ), (
            f"out_of_bound_positions must be float32 Tensor of shape [N,3], got shape={out_of_bound_positions.shape}, dtype={out_of_bound_positions.dtype}"
        )
        assert (
            out_of_bound_cuboid_dims.shape == out_of_bound_positions.shape
            and out_of_bound_cuboid_dims.dtype == torch.float32
        ), (
            f"out_of_bound_cuboid_dims must be float32 Tensor of shape [N,3], got shape={out_of_bound_cuboid_dims.shape}, dtype={out_of_bound_cuboid_dims.dtype}"
        )
        assert (bg_tex.ndim == 4 or bg_tex.ndim == 5) and bg_tex.dtype == torch.float32, (
            f"bg_tex must be float32 4D or 5D Tensor, got shape={bg_tex.shape}, dtype={bg_tex.dtype}"
        )
        assert gaussian_densities.ndim == 1 and gaussian_densities.dtype == torch.float32, (
            f"gaussian_densities must be float32 Tensor of shape [N], got shape={gaussian_densities.shape}, dtype={gaussian_densities.dtype}"
        )
        assert gaussian_visibility.ndim == 1 and gaussian_visibility.dtype == torch.float32, (
            f"gaussian_visibility must be float32 Tensor of shape [N], got shape={gaussian_visibility.shape}, dtype={gaussian_visibility.dtype}"
        )
        assert (
            gaussian_z_scales.ndim == 2 and gaussian_z_scales.shape[1] == 3 and gaussian_z_scales.dtype == torch.float32
        ), (
            f"gaussian_z_scales must be float32 Tensor of shape [N, 3], got shape={gaussian_z_scales.shape}, dtype={gaussian_z_scales.dtype}"
        )

        # Get dimensions for each input tensor
        B_rgb, H_rgb, W_rgb, _ = rgb_flags.shape
        B_lidar, H_lidar, W_lidar, _ = lidar_flags.shape
        N_rays_rgb = bg_pred.shape[0]
        N_rays_lidar = bg_lidar_pred.shape[0]
        BC_gc, D_gc, H_gc, W_gc = grids_per_camera.shape
        B_gc = BC_gc // GRID_NUM_CHANNELS
        BC_gf, D_gf, H_gf, W_gf = grids_per_frame.shape
        B_gf = BC_gf // GRID_NUM_CHANNELS
        numel_grids_per_camera = B_gc * D_gc * H_gc * W_gc
        numel_grids_per_frame = B_gf * D_gf * H_gf * W_gf
        N_scales = gaussian_scales.shape[0]
        if bg_tex.ndim == 4:
            B_tex, H_tex, W_tex, C_tex = bg_tex.shape
            D_tex = 1
        else:
            B_tex, D_tex, H_tex, W_tex, C_tex = bg_tex.shape
            assert D_tex == 6, "SkyEnvMapBackground texture (B,D,H,W,C) must have D==6 for cubemap"
        numel_bg_tex = B_tex * D_tex * H_tex * W_tex * C_tex
        N_densities = gaussian_densities.shape[0]
        num_out_of_bound_gaussians = out_of_bound_positions.shape[0]
        N_z_scales = gaussian_z_scales.shape[0]

        # Extra assert input shape dependency across tensor inputs
        if bg_factor >= 0:  # bg_factor is -1 for dummy tensors
            # N_rays in Background is the same as the number of RGB pixels, cf. BackgroundLoss in nre/losses/losses.py
            N_rays_rgb_flags = B_rgb * H_rgb * W_rgb
            assert N_rays_rgb == N_rays_rgb_flags, (
                f"bg_pred shape must be [{N_rays_rgb_flags}], got shape={bg_pred.shape}"
            )
        if bg_lidar_factor >= 0:  # bg_lidar_factor is -1 for dummy tensors
            # N_rays_lidar in Background Lidar is the same as the number of Lidar pixels
            N_rays_lidar_flags = B_lidar * H_lidar * W_lidar
            assert N_rays_lidar == N_rays_lidar_flags, (
                f"bg_lidar_pred shape must be [{N_rays_lidar_flags}], got shape={bg_lidar_pred.shape}"
            )

        # Ensure inputs are contiguous: Slang CUDA kernels require contiguous tensors for correct memory access
        rgb_flags = rgb_flags.contiguous()
        lidar_flags = lidar_flags.contiguous()
        rgb_pred = rgb_pred.contiguous()
        lidar_pred = lidar_pred.contiguous()
        intensity_pred = intensity_pred.contiguous()
        intensity_gt = intensity_gt.contiguous()
        raydrop_pred = raydrop_pred.contiguous()
        raydrop_gt = raydrop_gt.contiguous()
        bg_pred = bg_pred.contiguous()
        bg_lidar_pred = bg_lidar_pred.contiguous()
        rgb_gt = rgb_gt.contiguous()
        lidar_gt = lidar_gt.contiguous()
        grids_per_camera = grids_per_camera.contiguous()
        grids_per_frame = grids_per_frame.contiguous()
        bg_tex = bg_tex.contiguous()
        gaussian_scales = gaussian_scales.contiguous()
        gaussian_densities = gaussian_densities.contiguous()
        gaussian_visibility = gaussian_visibility.contiguous()
        gaussian_z_scales = gaussian_z_scales.contiguous()

        if bg_tex.ndim == 5:  # View bg_tex as 4D as Slang accepts up to 4D tensors (reverted at end of backward)
            bg_tex = bg_tex.view(B_tex * D_tex, H_tex, W_tex, C_tex)

        out_of_bound_positions = out_of_bound_positions.contiguous()
        out_of_bound_cuboid_dims = out_of_bound_cuboid_dims.contiguous()

        # Output tensors sized for each output tensor (these must be contiguous for backward pass)
        rgb_loss = torch.empty(
            (B_rgb, H_rgb, W_rgb), dtype=torch.float32, device=rgb_pred.device, memory_format=torch.contiguous_format
        )
        lidar_loss = torch.empty(
            (B_lidar, H_lidar, W_lidar),
            dtype=torch.float32,
            device=lidar_pred.device,
            memory_format=torch.contiguous_format,
        )
        intensity_loss = torch.empty(
            (B_lidar, H_lidar, W_lidar),
            dtype=torch.float32,
            device=intensity_pred.device,
            memory_format=torch.contiguous_format,
        )
        raydrop_loss = torch.empty(
            (B_lidar, H_lidar, W_lidar),
            dtype=torch.float32,
            device=raydrop_pred.device,
            memory_format=torch.contiguous_format,
        )
        bg_loss = torch.empty(
            N_rays_rgb,
            dtype=torch.float32,
            device=bg_pred.device,
            memory_format=torch.contiguous_format,
        )
        bg_lidar_loss = torch.empty(
            N_rays_lidar,
            dtype=torch.float32,
            device=bg_lidar_pred.device,
            memory_format=torch.contiguous_format,
        )
        grids_drift_loss = torch.empty(
            (numel_grids_per_camera + numel_grids_per_frame),
            dtype=torch.float32,
            device=grids_per_camera.device,
            memory_format=torch.contiguous_format,
        )
        grid_camera_spatial_tv_loss = torch.empty(
            numel_grids_per_camera,
            dtype=torch.float32,
            device=grids_per_camera.device,
            memory_format=torch.contiguous_format,
        )
        grid_frame_spatial_tv_loss = torch.empty(
            numel_grids_per_frame,
            dtype=torch.float32,
            device=grids_per_frame.device,
            memory_format=torch.contiguous_format,
        )
        scale_loss = torch.empty(
            N_scales,
            dtype=torch.float32,
            device=gaussian_scales.device,
        )
        bg_tex_loss = torch.empty(
            numel_bg_tex,
            dtype=torch.float32,
            device=bg_tex.device,
            memory_format=torch.contiguous_format,
        )
        density_loss = torch.empty(
            N_densities,
            dtype=torch.float32,
            device=gaussian_densities.device,
            memory_format=torch.contiguous_format,
        )
        out_of_bound_loss = torch.empty(
            num_out_of_bound_gaussians,
            dtype=torch.float32,
            device=out_of_bound_positions.device,
            memory_format=torch.contiguous_format,
        )
        z_scale_loss = torch.empty(
            N_z_scales,
            dtype=torch.float32,
            device=gaussian_z_scales.device,
            memory_format=torch.contiguous_format,
        )

        # ========================================================================
        # Launch 4 separate dispatches for different loss categories
        # This replaces the monolithic kernel with optimized per-category kernels
        # ========================================================================

        # DISPATCH 1: Camera rendering losses (RGB + background)
        # Skip dispatch if all losses are disabled (factor == -1)
        if rgb_factor >= 0 or bg_factor >= 0:
            blocks_camera = max(
                div_up(B_rgb * H_rgb * W_rgb, BLOCK_THREADS),
                div_up(N_rays_rgb, BLOCK_THREADS),
            )
            slang_losses.camera_losses_kernel(
                (BLOCK_THREADS, 1, 1),  # blockSize
                (blocks_camera, 1, 1),  # gridSize (sized for camera losses)
                # Dimensions
                B_rgb,
                H_rgb,
                W_rgb,
                # Factors
                rgb_factor,
                bg_factor,
                # Inputs
                rgb_flags,
                rgb_gt,
                (rgb_pred, (rgb_pred,)),
                (bg_pred, (bg_pred,)),
                # Outputs
                (rgb_loss, (rgb_loss,)),
                (bg_loss, (bg_loss,)),
            )

        # DISPATCH 2: LiDAR rendering losses (LiDAR + background_lidar + intensity)
        # Skip dispatch if all losses are disabled (factor == -1)
        if lidar_factor >= 0 or bg_lidar_factor >= 0 or intensity_factor >= 0 or raydrop_factor >= 0:
            blocks_lidar = max(
                div_up(B_lidar * H_lidar * W_lidar, BLOCK_THREADS),
                div_up(N_rays_lidar, BLOCK_THREADS),
            )
            slang_losses.lidar_losses_kernel(
                (BLOCK_THREADS, 1, 1),
                (blocks_lidar, 1, 1),  # gridSize (sized for lidar losses)
                # Dimensions
                B_lidar,
                H_lidar,
                W_lidar,
                # Factors
                lidar_factor,
                bg_lidar_factor,
                intensity_factor,
                raydrop_factor,
                # Inputs
                lidar_flags,
                lidar_gt,
                intensity_gt,
                raydrop_gt,
                (lidar_pred, (lidar_pred,)),
                (bg_lidar_pred, (bg_lidar_pred,)),
                (intensity_pred, (intensity_pred,)),
                (raydrop_pred, (raydrop_pred,)),
                # Outputs
                (lidar_loss, (lidar_loss,)),
                (bg_lidar_loss, (bg_lidar_loss,)),
                (intensity_loss, (intensity_loss,)),
                (raydrop_loss, (raydrop_loss,)),
            )

        # DISPATCH 3: Gaussian regularization losses (scale + density + z_scale + out_of_bound)
        # Skip dispatch if all losses are disabled (factor == -1)
        if scale_factor >= 0 or density_factor >= 0 or z_scale_factor >= 0 or out_of_bound_factor >= 0:
            blocks_gaussian = max(
                div_up(N_scales, BLOCK_THREADS),
                div_up(N_densities, BLOCK_THREADS),
                div_up(N_z_scales, BLOCK_THREADS),
                div_up(num_out_of_bound_gaussians, BLOCK_THREADS),
            )
            slang_losses.gaussian_losses_kernel(
                (BLOCK_THREADS, 1, 1),
                (blocks_gaussian, 1, 1),  # gridSize (sized for gaussian losses)
                # Dimensions
                N_scales,
                N_densities,
                N_z_scales,
                num_out_of_bound_gaussians,
                # Factors
                scale_factor,
                density_factor,
                z_scale_factor,
                z_scale_threshold,
                out_of_bound_factor,
                # Visibility
                gaussian_visibility,
                # Inputs
                (gaussian_scales, (gaussian_scales,)),
                (gaussian_densities, (gaussian_densities,)),
                (gaussian_z_scales, (gaussian_z_scales,)),
                (out_of_bound_positions, (out_of_bound_positions,)),
                out_of_bound_cuboid_dims,
                # Outputs
                (scale_loss, (scale_loss,)),
                (density_loss, (density_loss,)),
                (z_scale_loss, (z_scale_loss,)),
                (out_of_bound_loss, (out_of_bound_loss,)),
            )

        # DISPATCH 4: Background texture & grid regularization losses
        # Skip dispatch if all losses are disabled (factor == -1)
        if (
            bg_tex_factor >= 0
            or grid_drift_per_camera_factor >= 0
            or grid_drift_per_frame_factor >= 0
            or grid_camera_spatial_tv_factor >= 0
            or grid_frame_spatial_tv_factor >= 0
        ):
            blocks_background_grid = max(
                div_up(numel_bg_tex, BLOCK_THREADS),
                div_up(numel_grids_per_camera, BLOCK_THREADS),
                div_up(numel_grids_per_frame, BLOCK_THREADS),
            )
            slang_losses.background_grid_losses_kernel(
                (BLOCK_THREADS, 1, 1),
                (blocks_background_grid, 1, 1),  # gridSize (sized for bg_tex)
                # Background texture dimensions
                B_tex,
                D_tex,
                H_tex,
                W_tex,
                C_tex,
                # Grid dimensions
                B_gc,
                D_gc,
                H_gc,
                W_gc,
                B_gf,
                D_gf,
                H_gf,
                W_gf,
                # Factors
                bg_tex_factor,
                grid_drift_per_camera_factor,
                grid_drift_per_frame_factor,
                grid_camera_spatial_tv_factor,
                grid_frame_spatial_tv_factor,
                # Inputs
                (bg_tex, (bg_tex,)),
                (grids_per_camera, (grids_per_camera,)),
                (grids_per_frame, (grids_per_frame,)),
                # Outputs
                (bg_tex_loss, (bg_tex_loss,)),
                (grids_drift_loss, (grids_drift_loss,)),
                (grid_camera_spatial_tv_loss, (grid_camera_spatial_tv_loss,)),
                (grid_frame_spatial_tv_loss, (grid_frame_spatial_tv_loss,)),
            )

        # Save for backward pass
        ctx.save_for_backward(
            rgb_flags,
            rgb_pred,
            rgb_gt,
            lidar_flags,
            lidar_pred,
            lidar_gt,
            intensity_pred,
            intensity_gt,
            raydrop_pred,
            raydrop_gt,
            bg_pred,
            bg_lidar_pred,
            rgb_loss,
            lidar_loss,
            intensity_loss,
            raydrop_loss,
            bg_loss,
            bg_lidar_loss,
            grids_per_camera,
            grids_per_frame,
            grids_drift_loss,
            grid_camera_spatial_tv_loss,
            grid_frame_spatial_tv_loss,
            gaussian_scales,
            scale_loss,
            bg_tex,
            bg_tex_loss,
            gaussian_densities,
            density_loss,
            out_of_bound_positions,
            out_of_bound_cuboid_dims,
            out_of_bound_loss,
            gaussian_visibility,
            gaussian_z_scales,
            z_scale_loss,
        )
        ctx.rgb_factor = rgb_factor
        ctx.lidar_factor = lidar_factor
        ctx.intensity_factor = intensity_factor
        ctx.raydrop_factor = raydrop_factor
        ctx.bg_factor = bg_factor
        ctx.bg_lidar_factor = bg_lidar_factor
        ctx.grid_drift_per_camera_factor = grid_drift_per_camera_factor
        ctx.grid_drift_per_frame_factor = grid_drift_per_frame_factor
        ctx.grid_camera_spatial_tv_factor = grid_camera_spatial_tv_factor
        ctx.grid_frame_spatial_tv_factor = grid_frame_spatial_tv_factor
        ctx.scale_factor = scale_factor
        ctx.bg_tex_factor = bg_tex_factor
        ctx.D_tex = D_tex
        ctx.density_factor = density_factor
        ctx.out_of_bound_factor = out_of_bound_factor
        ctx.z_scale_threshold = z_scale_threshold
        ctx.z_scale_factor = z_scale_factor
        return (
            rgb_loss,
            lidar_loss,
            bg_loss,
            bg_lidar_loss,
            grids_drift_loss,
            grid_camera_spatial_tv_loss,
            grid_frame_spatial_tv_loss,
            scale_loss,
            bg_tex_loss,
            density_loss,
            out_of_bound_loss,
            z_scale_loss,
            intensity_loss,
            raydrop_loss,
        )

    @staticmethod
    def backward(
        ctx,
        grad_rgb_loss,
        grad_lidar_loss,
        grad_bg_loss,
        grad_bg_lidar_loss,
        grad_grids_drift_loss,
        grad_grid_camera_spatial_tv_loss,
        grad_grid_frame_spatial_tv_loss,
        grad_scale_loss,
        grad_bg_tex_loss,
        grad_density_loss,
        grad_out_of_bound_loss,
        grad_z_scale_loss,
        grad_intensity_loss,
        grad_raydrop_loss,
    ):
        (
            rgb_flags,
            rgb_pred,
            rgb_gt,
            lidar_flags,
            lidar_pred,
            lidar_gt,
            intensity_pred,
            intensity_gt,
            raydrop_pred,
            raydrop_gt,
            bg_pred,
            bg_lidar_pred,
            rgb_loss,
            lidar_loss,
            intensity_loss,
            raydrop_loss,
            bg_loss,
            bg_lidar_loss,
            grids_per_camera,
            grids_per_frame,
            grids_drift_loss,
            grid_camera_spatial_tv_loss,
            grid_frame_spatial_tv_loss,
            gaussian_scales,
            scale_loss,
            bg_tex,
            bg_tex_loss,
            gaussian_densities,
            density_loss,
            out_of_bound_positions,
            out_of_bound_cuboid_dims,
            out_of_bound_loss,
            gaussian_visibility,
            gaussian_z_scales,
            z_scale_loss,
        ) = ctx.saved_tensors
        rgb_factor = ctx.rgb_factor
        lidar_factor = ctx.lidar_factor
        intensity_factor = ctx.intensity_factor
        raydrop_factor = ctx.raydrop_factor
        bg_factor = ctx.bg_factor
        bg_lidar_factor = ctx.bg_lidar_factor
        grid_drift_per_camera_factor = ctx.grid_drift_per_camera_factor
        grid_drift_per_frame_factor = ctx.grid_drift_per_frame_factor
        grid_camera_spatial_tv_factor = ctx.grid_camera_spatial_tv_factor
        grid_frame_spatial_tv_factor = ctx.grid_frame_spatial_tv_factor
        scale_factor = ctx.scale_factor
        bg_tex_factor = ctx.bg_tex_factor
        D_tex = ctx.D_tex
        density_factor = ctx.density_factor
        out_of_bound_factor = ctx.out_of_bound_factor
        z_scale_threshold = ctx.z_scale_threshold
        z_scale_factor = ctx.z_scale_factor

        # Get dimensions for each input tensor
        B_rgb, H_rgb, W_rgb, _ = rgb_flags.shape
        B_lidar, H_lidar, W_lidar, _ = lidar_flags.shape
        N_rays_rgb = bg_pred.shape[0]
        N_rays_lidar = bg_lidar_pred.shape[0]
        BC_gc, D_gc, H_gc, W_gc = grids_per_camera.shape
        B_gc = BC_gc // GRID_NUM_CHANNELS
        BC_gf, D_gf, H_gf, W_gf = grids_per_frame.shape
        B_gf = BC_gf // GRID_NUM_CHANNELS
        numel_grids_per_camera = B_gc * D_gc * H_gc * W_gc
        numel_grids_per_frame = B_gf * D_gf * H_gf * W_gf
        N_scales = gaussian_scales.shape[0]
        BD_tex, H_tex, W_tex, C_tex = bg_tex.shape
        B_tex = BD_tex // D_tex
        numel_bg_tex = B_tex * D_tex * H_tex * W_tex * C_tex
        N_densities = gaussian_densities.shape[0]
        num_out_of_bound_gaussians = out_of_bound_positions.shape[0]
        N_z_scales = gaussian_z_scales.shape[0]

        # Only create gradient tensors for pred tensors (the only differentiable inputs) and loss tensors
        # Output loss tensors need gradients (initialized with zeros) so the backward pass can flow through them
        grad_rgb_pred = torch.zeros_like(rgb_pred, memory_format=torch.contiguous_format)
        grad_lidar_pred = torch.zeros_like(lidar_pred, memory_format=torch.contiguous_format)
        grad_intensity_pred = torch.zeros_like(intensity_pred, memory_format=torch.contiguous_format)
        grad_raydrop_pred = torch.zeros_like(raydrop_pred, memory_format=torch.contiguous_format)
        grad_bg_pred = torch.zeros_like(bg_pred, memory_format=torch.contiguous_format)
        grad_bg_lidar_pred = torch.zeros_like(bg_lidar_pred, memory_format=torch.contiguous_format)
        grad_grids_per_camera = torch.zeros_like(grids_per_camera, memory_format=torch.contiguous_format)
        grad_grids_per_frame = torch.zeros_like(grids_per_frame, memory_format=torch.contiguous_format)
        grad_gaussian_scales = torch.zeros_like(gaussian_scales, memory_format=torch.contiguous_format)
        grad_bg_tex = torch.zeros_like(bg_tex, memory_format=torch.contiguous_format)
        grad_gaussian_densities = torch.zeros_like(gaussian_densities, memory_format=torch.contiguous_format)
        grad_out_of_bound_positions = torch.zeros_like(out_of_bound_positions, memory_format=torch.contiguous_format)
        grad_gaussian_z_scales = torch.zeros_like(gaussian_z_scales, memory_format=torch.contiguous_format)

        # Ensure gradients are contiguous: Slang CUDA kernels require contiguous tensors for correct memory access
        # When a loss is disabled (factor is -1), its gradient has empty shape (it is a scalar) and cannot be made
        # contiguous() (it is marked as broadcasted), thus we create a tensor with the same shape as its loss
        if rgb_factor >= 0:
            grad_rgb_loss = grad_rgb_loss.contiguous()
        else:
            grad_rgb_loss = torch.zeros_like(rgb_loss, memory_format=torch.contiguous_format)
        if lidar_factor >= 0:
            grad_lidar_loss = grad_lidar_loss.contiguous()
        else:
            grad_lidar_loss = torch.zeros_like(lidar_loss, memory_format=torch.contiguous_format)
        if intensity_factor >= 0:
            grad_intensity_loss = grad_intensity_loss.contiguous()
        else:
            grad_intensity_loss = torch.zeros_like(intensity_loss, memory_format=torch.contiguous_format)
        if raydrop_factor >= 0:
            grad_raydrop_loss = grad_raydrop_loss.contiguous()
        else:
            grad_raydrop_loss = torch.zeros_like(raydrop_loss, memory_format=torch.contiguous_format)
        if bg_factor >= 0:
            grad_bg_loss = grad_bg_loss.contiguous()
        else:
            grad_bg_loss = torch.zeros_like(bg_loss, memory_format=torch.contiguous_format)
        if bg_lidar_factor >= 0:
            grad_bg_lidar_loss = grad_bg_lidar_loss.contiguous()
        else:
            grad_bg_lidar_loss = torch.zeros_like(bg_lidar_loss, memory_format=torch.contiguous_format)
        if grid_drift_per_camera_factor >= 0 or grid_drift_per_frame_factor >= 0:
            grad_grids_drift_loss = grad_grids_drift_loss.contiguous()
        else:
            grad_grids_drift_loss = torch.zeros_like(grids_drift_loss, memory_format=torch.contiguous_format)
        if grid_camera_spatial_tv_factor >= 0:
            grad_grid_camera_spatial_tv_loss = grad_grid_camera_spatial_tv_loss.contiguous()
        else:
            grad_grid_camera_spatial_tv_loss = torch.zeros_like(
                grid_camera_spatial_tv_loss, memory_format=torch.contiguous_format
            )
        if grid_frame_spatial_tv_factor >= 0:
            grad_grid_frame_spatial_tv_loss = grad_grid_frame_spatial_tv_loss.contiguous()
        else:
            grad_grid_frame_spatial_tv_loss = torch.zeros_like(
                grid_frame_spatial_tv_loss, memory_format=torch.contiguous_format
            )

        if scale_factor >= 0:
            grad_scale_loss = grad_scale_loss.contiguous()
        else:
            grad_scale_loss = torch.zeros_like(scale_loss, memory_format=torch.contiguous_format)
        if bg_tex_factor >= 0:
            grad_bg_tex_loss = grad_bg_tex_loss.contiguous()
        else:
            grad_bg_tex_loss = torch.zeros_like(bg_tex_loss, memory_format=torch.contiguous_format)
        if density_factor >= 0:
            grad_density_loss = grad_density_loss.contiguous()
        else:
            grad_density_loss = torch.zeros_like(density_loss, memory_format=torch.contiguous_format)
        if out_of_bound_factor >= 0:
            grad_out_of_bound_loss = grad_out_of_bound_loss.contiguous()
        else:
            grad_out_of_bound_loss = torch.zeros_like(out_of_bound_loss, memory_format=torch.contiguous_format)
        if z_scale_factor >= 0:
            grad_z_scale_loss = grad_z_scale_loss.contiguous()
        else:
            grad_z_scale_loss = torch.zeros_like(z_scale_loss, memory_format=torch.contiguous_format)

        # ========================================================================
        # Launch 4 separate backward dispatches (matches forward dispatch structure)
        # ========================================================================

        # BACKWARD DISPATCH 1: Camera rendering losses
        # Skip dispatch if all losses are disabled (factor == -1)
        if rgb_factor >= 0 or bg_factor >= 0:
            blocks_camera = max(
                div_up(B_rgb * H_rgb * W_rgb, BLOCK_THREADS),
                div_up(N_rays_rgb, BLOCK_THREADS),
            )
            slang_losses.camera_losses_kernel_bwd_diff(
                (BLOCK_THREADS, 1, 1),
                (blocks_camera, 1, 1),
                # Dimensions
                B_rgb,
                H_rgb,
                W_rgb,
                # Factors
                rgb_factor,
                bg_factor,
                # Inputs
                rgb_flags,
                rgb_gt,
                (rgb_pred, (grad_rgb_pred,)),
                (bg_pred, (grad_bg_pred,)),
                # Outputs
                (rgb_loss, (grad_rgb_loss,)),
                (bg_loss, (grad_bg_loss,)),
            )

        # BACKWARD DISPATCH 2: LiDAR rendering losses
        # Skip dispatch if all losses are disabled (factor == -1)
        if lidar_factor >= 0 or bg_lidar_factor >= 0 or intensity_factor >= 0 or raydrop_factor >= 0:
            blocks_lidar = max(
                div_up(B_lidar * H_lidar * W_lidar, BLOCK_THREADS),
                div_up(N_rays_lidar, BLOCK_THREADS),
            )
            slang_losses.lidar_losses_kernel_bwd_diff(
                (BLOCK_THREADS, 1, 1),
                (blocks_lidar, 1, 1),
                # Dimensions
                B_lidar,
                H_lidar,
                W_lidar,
                # Factors
                lidar_factor,
                bg_lidar_factor,
                intensity_factor,
                raydrop_factor,
                # Inputs
                lidar_flags,
                lidar_gt,
                intensity_gt,
                raydrop_gt,
                (lidar_pred, (grad_lidar_pred,)),
                (bg_lidar_pred, (grad_bg_lidar_pred,)),
                (intensity_pred, (grad_intensity_pred,)),
                (raydrop_pred, (grad_raydrop_pred,)),
                # Outputs
                (lidar_loss, (grad_lidar_loss,)),
                (bg_lidar_loss, (grad_bg_lidar_loss,)),
                (intensity_loss, (grad_intensity_loss,)),
                (raydrop_loss, (grad_raydrop_loss,)),
            )

        # BACKWARD DISPATCH 3: Gaussian regularization losses
        # Skip dispatch if all losses are disabled (factor == -1)
        if scale_factor >= 0 or density_factor >= 0 or z_scale_factor >= 0 or out_of_bound_factor >= 0:
            blocks_gaussian = max(
                div_up(N_scales, BLOCK_THREADS),
                div_up(N_densities, BLOCK_THREADS),
                div_up(N_z_scales, BLOCK_THREADS),
                div_up(num_out_of_bound_gaussians, BLOCK_THREADS),
            )
            slang_losses.gaussian_losses_kernel_bwd_diff(
                (BLOCK_THREADS, 1, 1),
                (blocks_gaussian, 1, 1),
                # Dimensions
                N_scales,
                N_densities,
                N_z_scales,
                num_out_of_bound_gaussians,
                # Factors
                scale_factor,
                density_factor,
                z_scale_factor,
                z_scale_threshold,
                out_of_bound_factor,
                # Visibility
                gaussian_visibility,
                # Inputs
                (gaussian_scales, (grad_gaussian_scales,)),
                (gaussian_densities, (grad_gaussian_densities,)),
                (gaussian_z_scales, (grad_gaussian_z_scales,)),
                (out_of_bound_positions, (grad_out_of_bound_positions,)),
                out_of_bound_cuboid_dims,
                # Outputs
                (scale_loss, (grad_scale_loss,)),
                (density_loss, (grad_density_loss,)),
                (z_scale_loss, (grad_z_scale_loss,)),
                (out_of_bound_loss, (grad_out_of_bound_loss,)),
            )

        # BACKWARD DISPATCH 4: Background texture & grid regularization losses
        # Skip dispatch if all losses are disabled (factor == -1)
        if (
            bg_tex_factor >= 0
            or grid_drift_per_camera_factor >= 0
            or grid_drift_per_frame_factor >= 0
            or grid_camera_spatial_tv_factor >= 0
            or grid_frame_spatial_tv_factor >= 0
        ):
            blocks_background_grid = max(
                div_up(numel_bg_tex, BLOCK_THREADS),
                div_up(numel_grids_per_camera, BLOCK_THREADS),
                div_up(numel_grids_per_frame, BLOCK_THREADS),
            )
            slang_losses.background_grid_losses_kernel_bwd_diff(
                (BLOCK_THREADS, 1, 1),
                (blocks_background_grid, 1, 1),
                # Background texture dimensions
                B_tex,
                D_tex,
                H_tex,
                W_tex,
                C_tex,
                # Grid dimensions
                B_gc,
                D_gc,
                H_gc,
                W_gc,
                B_gf,
                D_gf,
                H_gf,
                W_gf,
                # Factors
                bg_tex_factor,
                grid_drift_per_camera_factor,
                grid_drift_per_frame_factor,
                grid_camera_spatial_tv_factor,
                grid_frame_spatial_tv_factor,
                # Inputs
                (bg_tex, (grad_bg_tex,)),
                (grids_per_camera, (grad_grids_per_camera,)),
                (grids_per_frame, (grad_grids_per_frame,)),
                # Outputs
                (bg_tex_loss, (grad_bg_tex_loss,)),
                (grids_drift_loss, (grad_grids_drift_loss,)),
                (grid_camera_spatial_tv_loss, (grad_grid_camera_spatial_tv_loss,)),
                (grid_frame_spatial_tv_loss, (grad_grid_frame_spatial_tv_loss,)),
            )

        if D_tex > 1:  # View grad_bg_tex back as 5D if it was originally 5D (D_tex > 1 CUBEMAP)
            grad_bg_tex = grad_bg_tex.view(B_tex, D_tex, H_tex, W_tex, C_tex)

        return (
            None,  # rgb_flags
            grad_rgb_pred,  # rgb_pred
            None,  # rgb_gt (ground truth, no gradient needed)
            None,  # rgb_factor
            None,  # lidar_flags
            grad_lidar_pred,  # lidar_pred
            None,  # lidar_gt (ground truth, no gradient needed)
            None,  # lidar_factor
            grad_intensity_pred,  # intensity_pred
            None,  # intensity_gt (ground truth, no gradient needed)
            None,  # intensity_factor
            grad_raydrop_pred,  # raydrop_pred
            None,  # raydrop_gt (ground truth, no gradient needed)
            None,  # raydrop_factor
            grad_bg_pred,  # bg_pred
            None,  # bg_factor
            grad_bg_lidar_pred,  # bg_lidar_pred
            None,  # bg_lidar_factor
            grad_grids_per_camera,
            grad_grids_per_frame,
            None,  # grid_drift_per_camera_factor
            None,  # grid_drift_per_frame_factor
            None,  # grid_camera_spatial_tv_factor
            None,  # grid_frame_spatial_tv_factor
            grad_gaussian_scales,  # gaussian_scales
            None,  # scale_factor
            grad_bg_tex,  # bg_tex
            None,  # bg_tex_factor
            grad_gaussian_densities,  # gaussian_densities
            None,  # density_factor
            None,  # gaussian_visibility
            grad_out_of_bound_positions,
            None,  # out_of_bound_cuboid_dims
            None,  # out_of_bound_factor
            grad_gaussian_z_scales,  # gaussian_z_scales
            None,  # z_scale_threshold
            None,  # z_scale_factor
        )
