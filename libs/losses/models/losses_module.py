# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Layer 2: Losses model layer - Neural network modules for fused CUDA losses."""

import math

from typing import Any

import omegaconf
import torch

from libs.losses.functional.cuda_losses_function import CudaLossesFunction
from libs.losses.kernel.constants import GRID_NUM_CHANNELS
from libs.losses.models.reduce_functions import SumReduceFn
from libs.losses.models.utils import (
    _get_bilateral_grids,
    _get_out_of_bound_gaussian_nodes,
    _maybe_update_mcmc_visibility_counters,
    get_rendered_visibility_mask,
)
from libs.losses.orchestration.config import LossReturn
from nre.models.background import SkyEnvMapBackground
from nre.models.base import BaseModel
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.models.gaussians.gaussians_model import RigidGaussianModel
from nre.models.post_processing import (
    BilateralGridPerCamera,
    BilateralGridPerFrame,
)
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.misc import unpack_optional
from nre.utils.types import GaussiansCompositeReturn, RayFlags


class ModuleLosses(torch.nn.Module):
    """
    Stateful class for all fused CUDA losses model layer

    Layer 2 as explained in the docs/architecture/modules-losses.md file.
    """

    available: list[str]
    losses: list[Any]
    sum_reduce_fn: SumReduceFn
    dummy_flags: torch.Tensor
    dummy_rgb: torch.Tensor
    dummy_lidar: torch.Tensor
    dummy_out: torch.Tensor
    dummy_bg: torch.Tensor
    dummy_gc: torch.Tensor
    dummy_gf: torch.Tensor
    dummy_tex: torch.Tensor

    def __init__(self):
        super().__init__()
        # Mean reduction is converted to pre-multiplication and sum reduction in Slang losses
        self.available = [
            "rgb_l1_mean",
            "lidar_l1_mean",
            "background_mse_mean",
            "background_lidar_mse_mean",
            "bilateral_grid_drift_identity_distance_mean",
            "bilateral_grid_per_camera_tv_total_variation_spatial_mean",
            "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean",
            "gaussian_scale_abs_mean",
            "sky_env_map_background_total_variation_spatial_mean",
            "gaussian_density_abs_mean",
            "out_of_bound_l1_mean",
            "gaussian_z_scale_abs_mean",
            "intensity_mse_mean",
            "raydrop_mse_mean",
        ]
        self.losses = []
        self.sum_reduce_fn = SumReduceFn(omegaconf.DictConfig({}))
        dummy_device = torch.device("cuda")
        self.dummy_flags = torch.empty((1, 1, 1, 1), dtype=torch.int32, device=dummy_device)
        self.dummy_rgb = torch.empty((1, 1, 1, 3), dtype=torch.float32, device=dummy_device)
        self.dummy_lidar = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_out = torch.empty((1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_bg = torch.empty((1,), dtype=torch.float32, device=dummy_device)
        self.dummy_gc = torch.empty((1 * GRID_NUM_CHANNELS, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_gf = torch.empty((1 * GRID_NUM_CHANNELS, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_scales = torch.empty((1, 3), dtype=torch.float32, device=dummy_device)
        self.dummy_tex = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_densities = torch.empty((1,), dtype=torch.float32, device=dummy_device)
        self.dummy_gaussian_positions = torch.empty((1, 3), dtype=torch.float32, device=torch.device("cuda"))
        self.dummy_gaussian_cuboid_dims = torch.empty((1, 3), dtype=torch.float32, device=torch.device("cuda"))
        self.dummy_z_scales = torch.empty((1, 3), dtype=torch.float32, device=dummy_device)
        self.dummy_intensity = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_raydrop = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_visibility = torch.ones((1,), dtype=torch.float32, device=dummy_device)

    def is_enabled(
        self,
        loss_name: str,
        model: BaseModel,
        camera_grids: list[BilateralGridPerCamera],
        frame_grids: list[BilateralGridPerFrame],
        target: DataAndRenderingBatch,
        out_of_bound_gaussian_nodes: list[RigidGaussianModel],
    ) -> bool:
        if loss_name == "rgb_l1_mean":
            if target.data.camera is not None:
                assert target.data.camera.labels.flags is not None and target.data.camera.labels.rgb is not None, (
                    "Slang losses requires camera labels flags and rgb!"
                )
                return True
        elif loss_name == "lidar_l1_mean":
            if target.data.lidar is not None:
                assert target.data.lidar.labels.flags is not None and target.data.lidar.labels.distance is not None, (
                    "Slang losses requires lidar labels flags and distance!"
                )
                return True
        elif loss_name == "background_mse_mean":
            if target.data.camera is not None:
                assert target.data.camera.labels.flags is not None, "Slang losses requires camera labels flags!"
                return True
        elif loss_name == "background_lidar_mse_mean":
            if target.data.lidar is not None:
                assert target.data.lidar.labels.flags is not None, "Slang losses requires lidar labels flags!"
                return True
        elif loss_name == "bilateral_grid_drift_identity_distance_mean":
            if len(camera_grids) > 0 or len(frame_grids) > 0:
                return True
        elif loss_name == "bilateral_grid_per_camera_tv_total_variation_spatial_mean":
            if len(camera_grids) > 0:
                return True
        elif loss_name == "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean":
            if len(frame_grids) > 0:
                return True
        elif loss_name == "gaussian_scale_abs_mean":
            return True
        elif loss_name == "sky_env_map_background_total_variation_spatial_mean":
            return True
        elif loss_name == "gaussian_density_abs_mean":
            return True
        elif loss_name == "out_of_bound_l1_mean":
            assert isinstance(model, GaussiansComposite), (
                "Slang losses requires GaussiansComposite model for out_of_bound!"
            )

            return len(out_of_bound_gaussian_nodes) > 0
        elif loss_name == "gaussian_z_scale_abs_mean":
            return True
        elif loss_name == "intensity_mse_mean":
            if target.data.lidar is not None:
                return True
        elif loss_name == "raydrop_mse_mean":
            if target.data.lidar is not None:
                return True
        return False

    def forward(
        self,
        step: int,
        model: BaseModel,  # noqa: ARG002, pylint: disable=unused-argument
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
    ) -> dict[str, LossReturn]:
        assert results is not None and target is not None, "Slang losses requires results and target!"

        camera_grids = _get_bilateral_grids(model, BilateralGridPerCamera)
        frame_grids = _get_bilateral_grids(model, BilateralGridPerFrame)
        out_of_bound_gaussian_nodes = _get_out_of_bound_gaussian_nodes(model)

        run_losses = []
        for loss in self.losses:
            if loss.should_run_fn(step) and self.is_enabled(
                loss.name, model, camera_grids, frame_grids, target, out_of_bound_gaussian_nodes
            ):
                run_losses.append(loss)

        # Check which loss types are being run
        run_rgb = any("rgb" == loss.name.split("_")[0] for loss in run_losses)
        run_lidar = any("lidar" == loss.name.split("_")[0] for loss in run_losses)
        run_bg = any(
            "background" == loss.name.split("_")[0] and "lidar" != loss.name.split("_")[1] for loss in run_losses
        )
        run_bg_lidar = any(
            "background" == loss.name.split("_")[0] and "lidar" == loss.name.split("_")[1] for loss in run_losses
        )
        run_bilateral_grid_drift = any("bilateral_grid_drift" in loss.name for loss in run_losses)
        run_per_camera_spatial_tv = any("per_camera_tv" in loss.name for loss in run_losses)
        run_per_frame_spatial_tv = any("per_frame_spatial_tv" in loss.name for loss in run_losses)
        run_gaussian_scale = any("gaussian_scale" in loss.name for loss in run_losses)
        run_sky_env_map = any(loss.name == "sky_env_map_background_total_variation_spatial_mean" for loss in run_losses)
        run_gaussian_density = any("gaussian_density" in loss.name for loss in run_losses)
        run_out_of_bound = any(loss.name == "out_of_bound_l1_mean" for loss in run_losses)
        run_gaussian_z_scale = next((loss for loss in run_losses if "gaussian_z_scale" in loss.name), None)
        run_intensity = any("intensity" == loss.name.split("_")[0] for loss in run_losses)
        run_raydrop = any("raydrop" == loss.name.split("_")[0] for loss in run_losses)

        # Load RGB data or use dummy tensors
        rgb_flags: torch.Tensor | None = self.dummy_flags
        rgb_gt: torch.Tensor | None = self.dummy_rgb
        rgb_pred: torch.Tensor | None = self.dummy_rgb
        rgb_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_rgb and results.rendered_cam is not None:
            assert target.data.camera is not None, "Slang losses requires camera data for RGB loss!"
            rgb_flags = target.data.camera.labels.flags  # [B_rgb,H_rgb,W_rgb,1]
            rgb_gt = target.data.camera.labels.rgb  # [B_rgb,H_rgb,W_rgb,3]
            rgb_pred = unpack_optional(results.rendered_cam.rgb)  # [n_rays_rgb, 3]
            assert rgb_gt is not None, "rgb_gt should not be None"
            rgb_pred = rgb_pred.reshape_as(rgb_gt)  # [B_rgb,H_rgb,W_rgb,3]
            n_valid_rgb = target.data.camera.labels.n_valid_rgb
            rgb_factor = 1.0 / n_valid_rgb if n_valid_rgb > 0 else 0.0

        # Load Lidar data or use dummy tensors
        lidar_flags: torch.Tensor | None = self.dummy_flags
        lidar_gt: torch.Tensor | None = self.dummy_lidar
        lidar_pred: torch.Tensor | None = self.dummy_lidar
        lidar_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_lidar:
            assert target.data.lidar is not None, "Slang losses requires lidar data for Lidar loss!"
            lidar_flags = target.data.lidar.labels.flags  # [B_lidar,H_lidar,W_lidar,1]
            lidar_gt = target.data.lidar.labels.distance  # [B_lidar,H_lidar,W_lidar,1]
            lidar_pred = unpack_optional(results.rendered_lidar).distance  # [n_rays_lidar, 1]
            assert lidar_gt is not None, "lidar_gt should not be None"
            lidar_pred = lidar_pred.reshape_as(lidar_gt)  # [B_lidar,H_lidar,W_lidar,1]
            n_valid_lidar = target.data.lidar.labels.n_valid_lidar
            lidar_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Load Intensity data or use dummy tensors
        intensity_pred: torch.Tensor = self.dummy_intensity
        intensity_gt: torch.Tensor = self.dummy_intensity
        intensity_factor: float = -1.0

        if run_intensity:
            assert target.data.lidar is not None, "Slang losses requires lidar data for Intensity loss!"
            if not run_lidar:
                lidar_flags = target.data.lidar.labels.flags
            intensity_gt = unpack_optional(target.data.lidar.labels.intensity)  # [B,H,W,1]
            intensity_pred = unpack_optional(
                unpack_optional(unpack_optional(results.rendered_lidar).extra_ray_signals).intensity
            )
            intensity_pred = intensity_pred.reshape_as(intensity_gt)  # [B,H,W,1]
            n_valid = target.data.lidar.labels.n_valid_lidar
            intensity_factor = 1.0 / n_valid if n_valid > 0 else 0.0

        # Load Raydrop data or use dummy tensors
        raydrop_pred: torch.Tensor = self.dummy_raydrop
        raydrop_gt: torch.Tensor = self.dummy_raydrop
        raydrop_factor: float = -1.0

        if run_raydrop:
            assert target.data.lidar is not None, "Slang losses requires lidar data for Raydrop loss!"
            if not run_lidar and not run_intensity:
                lidar_flags = target.data.lidar.labels.flags
            raydrop_gt = unpack_optional(target.data.lidar.labels.raydrop)  # [B,H,W,1]
            raydrop_pred = unpack_optional(
                unpack_optional(unpack_optional(results.rendered_lidar).extra_ray_signals).raydrop
            )
            raydrop_pred = raydrop_pred.reshape_as(raydrop_gt)  # [B,H,W,1]
            # Raydrop kernel checks INVALID only (no DROPPED), so use INVALID-only count for normalization
            raydrop_valid_mask = target.data.lidar.labels.get_mask_flags_none(RayFlags.INVALID)
            n_valid_raydrop = int(raydrop_valid_mask.sum().item())
            raydrop_factor = 1.0 / n_valid_raydrop if n_valid_raydrop > 0 else 0.0

        # Load Background data or use dummy tensors
        bg_pred: torch.Tensor | None = self.dummy_bg
        bg_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_bg and results.rendered_cam is not None:
            assert target.data.camera is not None, "Slang losses requires camera data for Background loss!"
            if not run_rgb:  # rgb_flags was not loaded and it is needed for background
                rgb_flags = target.data.camera.labels.flags  # [B_rgb,H_rgb,W_rgb,1]
            bg_pred = results.rendered_cam.opacity  # [n_rays]
            n_valid_bg = target.data.camera.labels.n_valid_bg
            bg_factor = 1.0 / n_valid_bg if n_valid_bg > 0 else 0.0

        # Load Background Lidar data or use dummy tensors
        bg_lidar_pred: torch.Tensor | None = self.dummy_bg
        bg_lidar_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_bg_lidar:
            assert target.data.lidar is not None, "Slang losses requires lidar data for Background Lidar loss!"
            if not run_lidar:  # lidar_flags was not loaded and it is needed for background_lidar
                lidar_flags = target.data.lidar.labels.flags  # [B_lidar,H_lidar,W_lidar,1]
            bg_lidar_pred = unpack_optional(results.rendered_lidar).opacity  # [n_rays_lidar]
            n_valid_lidar = target.data.lidar.labels.n_valid_lidar
            bg_lidar_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Load Bilateral Grid data or use dummy tensors
        grids_per_camera: torch.Tensor = self.dummy_gc
        grids_per_frame: torch.Tensor = self.dummy_gf

        numel_grids_per_camera = 0
        numel_grids_per_frame = 0
        if len(camera_grids) > 0:
            assert len(camera_grids) == 1, "Slang losses requires exactly one BilateralGridPerCamera"
            grids_per_camera = camera_grids[0].bilateral_grid.grid
            B_gc, C_gc, D_gc, H_gc, W_gc = grids_per_camera.shape
            assert C_gc == GRID_NUM_CHANNELS, f"BilateralGridPerCamera must have {GRID_NUM_CHANNELS} channels"
            grids_per_camera = grids_per_camera.view(B_gc * C_gc, D_gc, H_gc, W_gc)
            numel_grids_per_camera = B_gc * D_gc * H_gc * W_gc
        if len(frame_grids) > 0:
            assert len(frame_grids) == 1, "Slang losses requires exactly one BilateralGridPerFrame"
            grids_per_frame = frame_grids[0].bilateral_grid.grid
            B_gf, C_gf, D_gf, H_gf, W_gf = grids_per_frame.shape
            assert C_gf == GRID_NUM_CHANNELS, f"BilateralGridPerFrame must have {GRID_NUM_CHANNELS} channels"
            grids_per_frame = grids_per_frame.view(B_gf * C_gf, D_gf, H_gf, W_gf)
            numel_grids_per_frame = B_gf * D_gf * H_gf * W_gf

        grid_drift_per_camera_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        grid_drift_per_frame_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        if run_bilateral_grid_drift:
            if numel_grids_per_camera > 0 or numel_grids_per_frame > 0:
                grid_drift_per_camera_factor = (
                    1.0 / (numel_grids_per_camera + numel_grids_per_frame)
                    if numel_grids_per_camera + numel_grids_per_frame > 0
                    else 0.0
                )
                grid_drift_per_frame_factor = (
                    1.0 / (numel_grids_per_camera + numel_grids_per_frame)
                    if numel_grids_per_camera + numel_grids_per_frame > 0
                    else 0.0
                )

        per_camera_spatial_tv_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        per_frame_spatial_tv_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        if run_per_camera_spatial_tv:
            if numel_grids_per_camera > 0:
                per_camera_spatial_tv_factor = 1.0 / numel_grids_per_camera if numel_grids_per_camera > 0 else 0.0
        if run_per_frame_spatial_tv:
            if numel_grids_per_frame > 0:
                per_frame_spatial_tv_factor = 1.0 / numel_grids_per_frame if numel_grids_per_frame > 0 else 0.0

        # Load Gaussian scale data or use dummy tensors
        gaussian_scales: torch.Tensor = self.dummy_scales
        scale_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_gaussian_scale:
            # Type narrow: gaussian_scale loss requires GaussiansComposite
            assert isinstance(model, GaussiansComposite), "gaussian_scale loss requires model to be GaussiansComposite"

            # Find the gaussian_scale loss config to get layer_lambdas
            gaussian_scale_loss = next((loss for loss in self.losses if "gaussian_scale" in loss.name), None)

            if gaussian_scale_loss is not None:
                # Get layer_lambdas from config (might not exist)
                layer_lambdas_dict = getattr(gaussian_scale_loss, "layer_lambdas", {})

                # Collect pre-activation scales with layer-specific weights applied in log-space
                # This allows exp() to be fused in the Slang kernel
                scales_list = []
                for node_id in model.get_gaussians_node_ids():
                    layer_lambda = layer_lambdas_dict.get(node_id, 1.0)

                    # Get pre-activation (log-space) scales to move exp() to Slang kernel
                    node_scales_preact = model.gaussians_nodes[node_id].get_scales(preactivation=True)

                    # Handle λ <= 0 specially to avoid math.log(0) ValueError
                    # When λ=0, the Python impl produces zeros: get_scales() * 0 = 0
                    # We match this by passing -inf in log-space: exp(-inf) = 0
                    if layer_lambda <= 0.0:
                        weighted_scales_preact = torch.full_like(node_scales_preact, float("-inf"))
                    else:
                        # Apply layer weighting in log-space: exp(x + log(λ)) = exp(x) * λ
                        # When λ=1.0, log(1.0)=0.0 and adding 0.0 is a no-op
                        weighted_scales_preact = node_scales_preact + math.log(layer_lambda)

                    scales_list.append(weighted_scales_preact)

                if scales_list:  # Only concatenate if we have scales
                    gaussian_scales = torch.cat(scales_list, dim=0)
                    scale_factor = 1.0 / gaussian_scales.numel()

        # Load Sky-Env-Map textures or use dummy tensors
        bg_tex: torch.Tensor | None = self.dummy_tex
        bg_tex_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_sky_env_map:
            assert isinstance(model, GaussiansComposite), "SkyEnvMap loss requires GaussiansComposite model"
            assert isinstance(model.background, SkyEnvMapBackground), "SkyEnvMap loss requires SkyEnvMapBackground"
            bg_tex = model.background.textures
            assert bg_tex is not None, "SkyEnvMap loss requires background textures"
            # SkyEnvMap factor is -1 (deactivated), 0 (active but invalid texture) or 1 (active and valid texture)
            # It has 3 pre-multiplication factors (for D, H, W) that will be computed in the Slang kernel
            bg_tex_factor = 1.0 if bg_tex.numel() > 0 else 0.0

        # Load Gaussian density data or use dummy tensors
        gaussian_densities: torch.Tensor = self.dummy_densities
        density_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_gaussian_density:
            # Type narrow: gaussian_density loss requires GaussiansComposite
            assert isinstance(model, GaussiansComposite), (
                "gaussian_density loss requires model to be GaussiansComposite"
            )

            # Find the gaussian_density loss config to get layer_lambdas
            gaussian_density_loss = next((loss for loss in self.losses if "gaussian_density" in loss.name), None)

            if gaussian_density_loss is not None:
                # Get layer_lambdas from config (might not exist)
                layer_lambdas_dict = getattr(gaussian_density_loss, "layer_lambdas", {})

                # Collect densities with layer-specific weights
                densities_list = []
                for node_id in model.get_gaussians_node_ids():
                    node_densities = model.gaussians_nodes[node_id].get_densities()
                    layer_lambda = layer_lambdas_dict.get(node_id, 1.0)
                    weighted_densities = node_densities * layer_lambda
                    # Flatten to 1D tensor (densities might be [N, 1] but we need [N])
                    densities_list.append(weighted_densities.flatten())

                if densities_list:  # Only concatenate if we have densities
                    gaussian_densities = torch.cat(densities_list, dim=0)
                    density_factor = 1.0 / gaussian_densities.numel()

        # Visibility mask for gaussian_scale / gaussian_density (same logic as PyTorch path)
        visibility_gaussian: torch.Tensor | None = None
        if run_gaussian_scale or run_gaussian_density:
            scale_loss_cfg = next((loss for loss in self.losses if "gaussian_scale" in loss.name), None)
            density_loss_cfg = next((loss for loss in self.losses if "gaussian_density" in loss.name), None)
            use_visibility_scale = run_gaussian_scale and getattr(scale_loss_cfg, "visibility_filter", False)
            use_visibility_density = run_gaussian_density and getattr(density_loss_cfg, "visibility_filter", False)

            if use_visibility_scale != use_visibility_density and run_gaussian_scale and run_gaussian_density:
                raise ValueError(
                    "gaussian_scale and gaussian_density must have matching visibility_filter when both are active. "
                    f"Got scale={use_visibility_scale}, density={use_visibility_density}. "
                    "The Slang kernel applies a single shared visibility mask to both."
                )

            if use_visibility_scale or use_visibility_density:
                element_count = max(gaussian_scales.shape[0], gaussian_densities.shape[0])
                ref_tensor = gaussian_scales if (run_gaussian_scale and scale_factor >= 0) else gaussian_densities
                loss_cfg = scale_loss_cfg if scale_loss_cfg else density_loss_cfg
                occlusion_aware = bool(getattr(loss_cfg, "occlusion_aware", False))
                visibility_gaussian = get_rendered_visibility_mask(
                    results=results,
                    element_count=element_count,
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                    visibility_filter=True,
                    occlusion_aware=occlusion_aware,
                )
                _maybe_update_mcmc_visibility_counters(model=model, results=results, visibility=visibility_gaussian)

        n_gaussian = max(gaussian_scales.shape[0], gaussian_densities.shape[0])
        ref_tensor_vis = gaussian_scales if (run_gaussian_scale and scale_factor >= 0) else gaussian_densities
        gaussian_visibility = (
            visibility_gaussian
            if visibility_gaussian is not None
            else torch.ones(n_gaussian, device=ref_tensor_vis.device, dtype=ref_tensor_vis.dtype)
        )

        # Load out_of_bound data or use dummy tensors
        out_of_bound_positions: torch.Tensor = self.dummy_gaussian_positions
        out_of_bound_cuboid_dims: torch.Tensor = self.dummy_gaussian_cuboid_dims
        out_of_bound_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_out_of_bound:
            positions_list: list[torch.Tensor] = []
            dims_list: list[torch.Tensor] = []
            for node in out_of_bound_gaussian_nodes:
                positions = node.get_positions()
                cuboid_dims = node.cuboid_tracks.cuboids_dims
                gaussian_cuboid_ids = node.gaussian_cuboid_ids
                dims = cuboid_dims[gaussian_cuboid_ids]
                positions_list.append(positions)
                dims_list.append(dims)
            if positions_list:
                out_of_bound_positions = torch.cat(positions_list, dim=0)
                out_of_bound_cuboid_dims = torch.cat(dims_list, dim=0)
                num_elements = out_of_bound_positions.shape[0] * out_of_bound_positions.shape[1]
                out_of_bound_factor = 1.0 / num_elements if num_elements > 0 else 0.0

        # Load Gaussian Z-scale data or use dummy tensors
        # Note: gaussian_z_scales is loaded separately from gaussian_scales because:
        # 1. gaussian_scales = ALL layers concatenated, with layer_lambda weighting applied
        # 2. gaussian_z_scales = ONE specific layer (e.g. "road"), WITHOUT weighting
        # Even if the same layer appears in both, the data differs due to weighting.
        gaussian_z_scales: torch.Tensor = self.dummy_z_scales
        z_scale_threshold: float = -1.0  # use -1 to indicate dummy/disabled
        z_scale_factor: float = -1.0

        if run_gaussian_z_scale:
            # Type narrow: gaussian_z_scale loss requires GaussiansComposite
            assert isinstance(model, GaussiansComposite), (
                "gaussian_z_scale loss requires model to be GaussiansComposite"
            )

            # Get layer_name and threshold from config
            layer_name = run_gaussian_z_scale.layer_name
            z_scale_threshold = run_gaussian_z_scale.road_z_scale

            # Get pre-activation Z-scales (full [N, 3] tensor, kernel will extract z-component)
            layer_scales = model.gaussians_nodes[layer_name].get_scales(preactivation=True)
            gaussian_z_scales = layer_scales  # Pass full [N, 3] tensor, let Slang extract [:, 2]

            z_scale_factor = 1.0 / layer_scales.shape[0]  # N elements (not N*3)

        losses = CudaLossesFunction.apply(
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
            grids_per_camera,
            grids_per_frame,
            grid_drift_per_camera_factor,
            grid_drift_per_frame_factor,
            per_camera_spatial_tv_factor,
            per_frame_spatial_tv_factor,
            gaussian_scales,
            scale_factor,
            bg_tex,
            bg_tex_factor,
            gaussian_densities,
            density_factor,
            gaussian_visibility,
            out_of_bound_positions,
            out_of_bound_cuboid_dims,
            out_of_bound_factor,
            gaussian_z_scales,
            z_scale_threshold,
            z_scale_factor,
        )

        ret: dict[str, LossReturn] = {}
        for loss in run_losses:
            loss_idx = self.available.index(loss.name)
            ret[loss.name] = LossReturn(
                name=loss.name,
                lambda_=loss.lambda_,
                value=losses[loss_idx],
                reduce_fn=self.sum_reduce_fn,
            )

        return ret
