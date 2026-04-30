# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import math

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from einops import rearrange, repeat
from safetensors.torch import load_file

from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.difix.model import DifixModel, DifixModelFactory
from nre.models.gaussians.renderers import BaseGaussianRenderer
from nre.nrm.config.models import CelsiusModelConfig
from nre.nrm.models.activations import (
    FalloffSigmaActivation,
    ForwardFlowActivation,
    GaussianActivations,
    GaussianParams,
    SkyMaskActivation,
)
from nre.nrm.models.base import BaseNRM, BaseNRMSupervisionPack
from nre.nrm.models.blocks import (
    AttentionBlock,
    ContinuousTimeEmbed,
    FeedForwardMLP,
    FeedForwardMLPConv,
    Mamba2Block,
    PatchEmbed,
    UnpatchProgressiveConv,
)
from nre.nrm.models.post_processing import PerCameraAffinePostProcessing
from nre.nrm.primitives.celsius_primitive import CelsiusNRMPrimitive, ModulatedLinearLayer
from nre.nrm.utils.motion import TimeRemapping
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.files import local_temp_file, parse_universal_path
from nre.utils.geometry import box_filter_points
from nre.utils.log import BatchMediaLogger
from nre.utils.misc import set_zero, unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.types import RayFlags
from nre.utils.visualize import flow2img, make_image_grid, scalar2img


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CelsiusNRMSupervisionPack(BaseNRMSupervisionPack):
    """
    Supervision pack for the Celsius model.

    The fields are:
    - context_distance: (B, H, W, 1)
    - context_velocity: (B, H, W, 3/6)
    - context_unscaled_falloff_sigma: (B, H, W, 1)
    """

    context_distance: torch.Tensor | None = None
    context_velocity: torch.Tensor | None = None
    context_unscaled_falloff_sigma: torch.Tensor | None = None


class CelsiusNRM(BaseNRM[CelsiusNRMPrimitive, CelsiusNRMSupervisionPack]):
    """
    Neural reconstruction model that predicts pixel-aligned Gaussians, with a shorter context and ViT backbone.
    Reference:
        [a] Yang et al. STORM: Spatio-Temporal Reconstruction Model for Large-Scale Outdoor Scenes.
        [b] Liang et al. Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos.
    """

    config: CelsiusModelConfig

    class SkyModule(nn.Module):
        def __init__(self, config: CelsiusModelConfig):
            super().__init__()
            self.embed_dim: int = config.encoder.embed_dim
            self.init_token_scale: float = config.init_token_scale
            self.patch_shape = config.patch_shape

            self.sky_token = nn.Parameter(torch.randn(self.embed_dim) * self.init_token_scale)
            self.sky_head = ModulatedLinearLayer(input_dim=3, hidden_dim=32, condition_dim=self.embed_dim, out_dim=3)
            self.sky_deconv = nn.ConvTranspose2d(self.embed_dim, 1, self.patch_shape, stride=self.patch_shape)

    class MotionModule(nn.Module):
        """
        Motion related module for PixelSTORM
        """

        motion_tokens: nn.Parameter | None

        def __init__(self, config: CelsiusModelConfig):
            super().__init__()
            self.n_motion_tokens: int = config.motion_module.n_motion_tokens
            self.motion_qkv_dim: int = config.motion_module.motion_qkv_dim
            self.init_token_scale: float = config.init_token_scale
            self.embed_dim: int = config.encoder.embed_dim
            self.unpatch_dim: int = config.motion_module.unpatch_dim
            self.motion_falloff: bool = config.motion_module.falloff
            self.patch_shape = config.patch_shape
            self.bidirectional_flow: bool = config.motion_module.bidirectional_flow

            self.velocity_dim = motion_dim = 6 if self.bidirectional_flow else 3
            if self.motion_falloff:
                motion_dim += 1

            if self.n_motion_tokens > 0:
                self.motion_tokens = nn.Parameter(
                    torch.randn(self.n_motion_tokens, self.embed_dim) * self.init_token_scale
                )
                self.motion_value_head = FeedForwardMLP(self.embed_dim, 256, motion_dim)
            else:
                self.motion_tokens = None
                self.motion_value_head = FeedForwardMLP(self.motion_qkv_dim, self.unpatch_dim * 2, motion_dim)
            self.unpatch_motion_query = nn.Sequential(
                UnpatchProgressiveConv(self.patch_shape, self.embed_dim, self.unpatch_dim),
                FeedForwardMLPConv(self.unpatch_dim, self.unpatch_dim * 2, self.motion_qkv_dim),
            )
            self.motion_key_heads = nn.ModuleList(
                [
                    FeedForwardMLP(input_dim=self.embed_dim, hidden_dim=self.embed_dim, output_dim=self.motion_qkv_dim)
                    for _ in range(self.n_motion_tokens)
                ]
            )
            self.motion_tau = 0.5

        def forward(
            self, x: torch.Tensor, motion_tokens: torch.Tensor | None, B: int, V: int, H: int, W: int
        ) -> torch.Tensor:
            motion_queries = self.unpatch_motion_query(x)
            if motion_tokens is not None:
                motion_values = self.motion_value_head(motion_tokens)  # (B, n_motion_tokens, C)
                motion_keys = torch.stack(
                    [self.motion_key_heads[i](motion_tokens[:, i]) for i in range(self.n_motion_tokens)], dim=1
                )  # (B, n_motion_tokens, C)
                motion_params = torch.nn.functional.scaled_dot_product_attention(
                    rearrange(motion_queries, "(B V) C H W -> B 1 (V H W) C", B=B, V=V),
                    motion_keys[:, None],
                    motion_values[:, None],
                    scale=1.0 / self.motion_tau,
                    dropout_p=0.0,
                )
                motion_params = rearrange(motion_params, "B 1 (V H W) C -> B V H W C", V=V, H=H, W=W)
            else:
                # Treat motion tokens as queries
                motion_params = self.motion_value_head(rearrange(motion_queries, "(B V) C H W -> B V H W C", B=B, V=V))
            return motion_params

    sky_module: SkyModule | None
    affine_module: PerCameraAffinePostProcessing | None
    motion_module: MotionModule | None

    def __init__(self, config: CelsiusModelConfig):
        super().__init__(config)

        # Padding for determining visibility mask for the dynamic objects
        self.cuboids_dims_padding = nn.Buffer(torch.tensor(self.config.track_padding_m, dtype=torch.float32))

        # Network: preprocessor
        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        self.mask_normalize = transforms.Normalize(mean=[0.5], std=[0.5], inplace=False)

        # Network: embeddings
        embed_dim = (encoder_config := self.config.encoder).embed_dim
        patch_shape = self.config.patch_shape
        self.patch_embed_img = PatchEmbed(
            patch_shape=patch_shape,
            input_dim=3 if not self.config.legacy_mask_input else 4,
            embed_dim=embed_dim,
            norm=self.config.use_patch_embed_norm,
        )
        self.patch_embed_ray = PatchEmbed(
            patch_shape=patch_shape, input_dim=6, embed_dim=embed_dim, norm=self.config.use_patch_embed_norm
        )
        self.patch_embed_time = ContinuousTimeEmbed(
            patch_shape=patch_shape, embed_dim=embed_dim, frequency_embedding_dim=256
        )

        # Network: activations
        activation_config = config.activations

        # Combined activations for Gaussian parameters
        self.gaussian_activations = GaussianActivations(activation_config)

        # Individual activations for other parameters
        self.sky_mask_activation = SkyMaskActivation(activation_config)
        self.forward_flow_activation = ForwardFlowActivation(activation_config)
        self.falloff_sigma_activation = FalloffSigmaActivation(activation_config)

        # Network: encoder
        self.encoder_blocks = nn.ModuleList()
        for block_type in encoder_config.block_pattern:
            if block_type == "T":
                self.encoder_blocks.append(
                    AttentionBlock(
                        input_dim=embed_dim,
                        n_heads=encoder_config.n_heads,
                        mlp_ratio=encoder_config.mlp_ratio,
                        qkv_bias=True,
                        layer_norm_eps=1e-6,
                        layer_scale_init_values=encoder_config.layer_scale_init_values,
                        qk_norm=encoder_config.qk_norm,
                    )
                )
            elif block_type == "M":
                self.encoder_blocks.append(Mamba2Block(embed_dim))
            else:
                raise ValueError(f"Invalid block type: {block_type}")
        assert len(self.encoder_blocks) == encoder_config.depth, (
            f"Number of encoder blocks must match depth. Got {len(self.encoder_blocks)} blocks, expected {encoder_config.depth}."
        )

        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6) if self.config.use_encoder_norm else nn.Identity()
        # The order of the output dimensions is: xyz or depth, rgb, scale, rotation, opacity
        output_dim = (3 if self.config.centroid_prediction == "xyz" else 1) + 3 + 3 + 4 + 1
        # Network: gaussian decoder (static attributes)
        self.deconv = nn.ConvTranspose2d(embed_dim, output_dim, patch_shape, stride=patch_shape)
        # Network: sky module
        self.sky_module = self.SkyModule(config) if self.config.sky_module.enabled else None
        # Network: affine module
        self.affine_module = None
        if self.config.affine_module.enabled:
            self.affine_module = PerCameraAffinePostProcessing(
                embed_dim=embed_dim,
                init_token_scale=config.init_token_scale,
                cross_attend=config.affine_module.cross_attend,
                n_affine_tokens=config.affine_module.n_affine_tokens,
                kv_norm=False,
            )
        # Network: motion module
        self.motion_module = self.MotionModule(config) if self.config.motion_module.enabled else None
        self.replace_velocity_with_context: bool = False

        # Prepare gaussian renderer
        self.gaussians_renderer = BaseGaussianRenderer.factory(
            self.config.renderer.name,
            self.config.renderer,
            self,  # type: ignore
        )

        # Prepare Difix model (use private attribute to avoid being included in model parameters)
        # Initialize the model once to avoid repeated construction overhead.
        self._difix_model: DifixModel | None = None
        if self.config.difix is not None:
            self._difix_model = DifixModelFactory.get(
                self.config.difix.model_url,
                self.config.difix.cache_dir,
                self.config.difix.model_filename,
                self.config.difix.model_resolution,
            )

    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        # Hack the nrend initialization logic.
        return CelsiusNRMPrimitive.from_positions_and_fixed_scale(
            positions=torch.rand(1, 3, device=torch.device("cpu")),
            scale=1.0,
            velocity_dim=self.motion_module.velocity_dim if self.motion_module is not None else 0,
        ).serialize_to_json_dict(with_state_dict=with_state_dict)

    @staticmethod
    def _convert_btimer_state_dict_to_celsius(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply key transformations to address the key name mismatch between Btimer and Celsius models.
        """
        key_mappings = {
            "enc_blocks.": "encoder_blocks.",
            "enc_norm.": "encoder_norm.",
            "patch_embed.": "patch_embed_img.",
            "patch_plucker_embed.": "patch_embed_ray.",
        }
        new_state_dict = {}
        for k, v in state_dict.items():
            for old_key, new_key in key_mappings.items():
                k = k.replace(old_key, new_key)
            new_state_dict[k] = v
        return new_state_dict

    def on_train_from_scratch_start(self, system, **kwargs):
        # Recursively weight initialization
        self.apply(self._init_weights)

        if (pretrained_path := self.config.init_weights_path) is not None:
            with local_temp_file(parse_universal_path(pretrained_path, s3_block_size_mb=256)) as local_model_path:
                if local_model_path.suffix == ".safetensors":
                    init_state_dict = load_file(local_model_path)
                else:
                    init_state_dict = torch.load(local_model_path, weights_only=False)["state_dict"]
            # Remove the model. prefix
            init_state_dict = {k.replace("model.", ""): v for k, v in init_state_dict.items()}
            # Apply key transformations to address the key name mismatch between Btimer and Celsius
            init_state_dict = self._convert_btimer_state_dict_to_celsius(init_state_dict)

            # zero-init happens only when loading pretrained and pretrained does not contain this key.
            self.patch_embed_time.zero_init()
            if self.affine_module is not None:
                self.affine_module.zero_init()

            # Concatenate state dict from other modules so we can load strictly
            for prefix, state_dict in [
                ("patch_embed_time", self.patch_embed_time.state_dict()),
                ("sky_module", self.sky_module.state_dict() if self.sky_module is not None else {}),
                ("affine_module", self.affine_module.state_dict() if self.affine_module is not None else {}),
                ("motion_module", self.motion_module.state_dict() if self.motion_module is not None else {}),
                ("gaussian_activations", self.gaussian_activations.state_dict()),
            ]:
                for k, v in state_dict.items():
                    if (state_key := f"{prefix}.{k}") not in init_state_dict:
                        init_state_dict[state_key] = v

            # Forcely use the padding from config
            init_state_dict["cuboids_dims_padding"] = self.cuboids_dims_padding.data
            self.load_state_dict(init_state_dict, strict=True)

    @classmethod
    def _init_weights(cls, m: nn.Module):
        """
        Initialize weights of the model.
        (Note that this is slightly different from PixelSTORM where only Linear layers are initialized.)
        """

        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = (m.kernel_size[0] * m.kernel_size[1] * m.out_channels) // m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0, 0.002)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        """
        Mainly to control whether we need to detach the linear layer of the affine module.
        """
        if self.affine_module is not None:
            self.affine_module.set_detach_linear_grad(
                global_step < self.config.affine_module.optimization_start_global_step
            )

        if self.motion_module is not None:
            start_step = self.config.motion_module.context_replace_start_global_step
            end_step = self.config.motion_module.context_replace_end_global_step
            use_pd_prob: float = (global_step - start_step) / (end_step - start_step)
            self.replace_velocity_with_context = torch.rand(1).item() > use_pd_prob

        return {}

    def _mask_from_cuboid_tracks(
        self, rays: torch.Tensor, rays_timestamps_us: torch.Tensor, cuboid_tracks: CuboidTracks | None
    ) -> torch.Tensor:
        """
        Args:
            rays: shape [..., 6]
            rays_timestamps_us: shape [..., 1] or [...]
        Returns:
            mask: shape [...]
        """
        if cuboid_tracks is None:
            return torch.zeros_like(rays[..., 0], dtype=torch.bool)

        dynamic_cuboids = CuboidTracks.Ops.subset_from_mask(
            cuboid_tracks, cuboid_tracks.tracks_flags & TrackFlags.DYNAMIC != 0
        )
        # Do ray-bbox intersection to obtain the mask
        rays_o, rays_d = rays[..., :3], rays[..., 3:]
        hit_cnt = dynamic_cuboids.ray_intersection(
            rays_o.reshape(-1, 3).contiguous(),
            rays_d.reshape(-1, 3).contiguous(),
            rays_timestamps_us.reshape(-1).contiguous(),
            self.cuboids_dims_padding,
            max_intersections_per_ray=1,
            with_intersections_ts=False,
        ).intersections_cnt
        return (hit_cnt > 0).reshape(rays.shape[:-1])

    def _compute_data_and_rendering_batch_velocity(
        self, batch: DataAndRenderingBatch, cuboid_track: CuboidTracks
    ) -> torch.Tensor | None:
        """
        Compute the velocity from the cuboid tracks.
        return None if signal not available or not computable.
        """
        # These two conditions are important even though we have lidar already.
        if (batch_data_camera := batch.data.camera) is None:
            return None

        if batch.rendering is None or (batch_rendering_camera := batch.rendering.camera) is None:
            return None

        # Context distance is important for velocity computation.
        # First check if already precomputed.
        if (distance := batch_data_camera.labels.metric_distance) is None:
            # If not, check if we can compute online.
            lidar_config = self.config.velocity_from_lidar
            if (
                lidar_config.enabled
                and lidar_config.gap_from_image_us > 0
                and (batch_data_lidar := batch.data.lidar) is not None
                and (batch_rendering_lidar := batch.rendering.lidar) is not None
            ):
                lidar_end_timestamps_us = batch_rendering_lidar.timestamps_startend_us[:, 1]

                # For each image frame, accumulate lidar points and render distance map.
                img_distances: list[torch.Tensor] = []
                for camera_frame_idx in range(batch_rendering_camera.b):
                    camera_start_timestamp_us, camera_end_timestamp_us = batch_rendering_camera.timestamps_startend_us[
                        camera_frame_idx
                    ]
                    lidar_batch_indices = torch.where(
                        torch.abs(lidar_end_timestamps_us - camera_end_timestamp_us.item())
                        <= lidar_config.gap_from_image_us
                    )[0]

                    # Obtain the lidar points belong to lidar_batch_indices.
                    lidar_labels = batch_data_lidar.labels[lidar_batch_indices]
                    lidar_rays_mask = unpack_optional(lidar_labels.flags)[..., 0] == 0
                    b, h, w = lidar_rays_mask.shape
                    lidar_distance = unpack_optional(lidar_labels.distance)
                    # Remove lidar points that are too close to the camera.
                    lidar_rays_mask = torch.logical_and(
                        lidar_rays_mask, lidar_distance[..., 0] > lidar_config.near_mask_threshold_m
                    )

                    lidar_rays = batch_rendering_lidar.rays[lidar_batch_indices][lidar_rays_mask]
                    lidar_timestamps_us = unpack_optional(batch_rendering_lidar.rays_timestamps_us)[
                        lidar_batch_indices
                    ][lidar_rays_mask]
                    lidar_distance = lidar_distance[lidar_rays_mask]
                    lidar_xyz_e = lidar_rays[..., :3] + lidar_rays[..., 3:] * lidar_distance

                    lidar_xyz_e = cuboid_track.warp_world_points_to_timestamps(
                        lidar_xyz_e,
                        lidar_timestamps_us.squeeze(-1),
                        ((camera_start_timestamp_us + camera_end_timestamp_us) // 2)[None].repeat(lidar_xyz_e.shape[0]),
                        self.cuboids_dims_padding,
                    )

                    # Rasterize the lidar points using our Gaussian renderer.
                    lidar_xyz_e = box_filter_points(
                        lidar_xyz_e, lidar_config.box_filter_voxel_size_m, max_count=lidar_config.box_filter_max_count
                    )
                    primitive = CelsiusNRMPrimitive.from_positions_and_fixed_scale(
                        lidar_xyz_e,
                        scale=lidar_config.gaussians_scale,
                        velocity_dim=self.motion_module.velocity_dim if self.motion_module is not None else 0,
                    )
                    primitive.gaussians_renderer = self.gaussians_renderer
                    rendered_return = primitive.forward(
                        batch_rendering_camera[camera_frame_idx],
                        [batch_data_camera.meta[camera_frame_idx]],
                    )
                    img_h, img_w = unpack_optional(batch_data_camera.h), unpack_optional(batch_data_camera.w)
                    img_distances.append(
                        unpack_optional(rendered_return.rendered_cam).distance.reshape(1, img_h, img_w, -1)
                    )

                distance = torch.cat(img_distances, dim=0)

            # If not, return None.
            else:
                return None

        # Sub-select dynamic tracks only
        cuboid_track = CuboidTracks.Ops.subset_from_mask(
            cuboid_track, cuboid_track.tracks_flags & TrackFlags.DYNAMIC != 0
        )

        batch_rays = batch_rendering_camera.rays
        batch_rays_timestamps_us = unpack_optional(batch_rendering_camera.rays_timestamps_us)

        world_points = batch_rays[..., :3] + distance * batch_rays[..., 3:]
        frame_gap_timestamps_us = TimeRemapping.from_timestamps_startend_us(
            batch_rendering_camera.timestamps_startend_us,
            torch.tensor(
                [m.unique_sensor_idx for m in batch_data_camera.meta], dtype=torch.int64, device=batch_rays.device
            ),
        ).frame_gap_timestamps_us

        try:
            world_points_forward = cuboid_track.warp_world_points_to_timestamps(
                world_points,
                batch_rays_timestamps_us,
                batch_rays_timestamps_us + frame_gap_timestamps_us[:, None, None, 1:2],
                self.cuboids_dims_padding,
            )
            world_points_backward = cuboid_track.warp_world_points_to_timestamps(
                world_points,
                batch_rays_timestamps_us,
                batch_rays_timestamps_us - frame_gap_timestamps_us[:, None, None, 0:1],
                self.cuboids_dims_padding,
            )
        except (IndexError, RuntimeError):
            # In case of malformed cuboid tracks (e.g. too short/empty) or runtime errors, we skip computation.
            return None

        if self.config.motion_module.bidirectional_flow:
            return torch.cat(
                [
                    (world_points_forward - world_points) / (frame_gap_timestamps_us[:, None, None, 1:2] * 1e-6),
                    (world_points - world_points_backward) / (frame_gap_timestamps_us[:, None, None, 0:1] * 1e-6),
                ],
                dim=-1,
            )
        else:
            return (world_points_forward - world_points_backward) / (
                frame_gap_timestamps_us[:, None, None, :].sum(dim=-1, keepdim=True) * 1e-6
            )

    @torch.no_grad()
    def _log_context_media(
        self,
        media_logger: BatchMediaLogger,
        gaussian_params: GaussianParams,
        sky_mask: torch.Tensor | None,
        speed_falloff: tuple[torch.Tensor | None, torch.Tensor | None],
        VHW: tuple[int, int, int],
        grid_width: int,
    ):
        """
        Log media for the context.
        """
        V, H, W = VHW
        forward_speed_mps, unscaled_falloff_sigma = speed_falloff
        im_context_rgbs = (
            rearrange(gaussian_params.rgb, "(V H W) C -> V H W C", V=V, H=H, W=W, C=3).float().cpu().numpy()
        )
        im_context_rgbs = (im_context_rgbs * 255).astype(np.uint8)
        media_logger.log_image("RGB (direct)", make_image_grid([t for t in im_context_rgbs], grid_width=grid_width))
        im_context_opacities = (
            rearrange(gaussian_params.opacity, "(V H W) 1 -> V H W", V=V, H=H, W=W).float().cpu().numpy()
        )
        im_context_opacities = (im_context_opacities * 255).astype(np.uint8)
        media_logger.log_image(
            "Opacity (Context)", make_image_grid([t for t in im_context_opacities], grid_width=grid_width)
        )
        if sky_mask is not None:
            im_context_sky_mask = rearrange(sky_mask, "(V H W) 1 -> V H W", V=V, H=H, W=W).float().cpu().numpy()
            im_context_sky_mask = (im_context_sky_mask * 255).astype(np.uint8)
            media_logger.log_image(
                "Sky Mask (Context)",
                make_image_grid([t for t in im_context_sky_mask], grid_width=grid_width),
            )
        if forward_speed_mps is not None:
            im_context_velocity = (
                rearrange(forward_speed_mps, "(V H W) C -> V H W C", V=V, H=H, W=W).float().cpu().numpy()[..., [0, 2]]
            )
            im_context_motion = flow2img(
                make_image_grid([t for t in im_context_velocity], grid_width=grid_width), rad_max=None
            )
            if unscaled_falloff_sigma is not None:
                # Falloff is already normalized by the span, let's re-map it to 0-1.
                im_context_falloff = unscaled_falloff_sigma.reshape(V, H, W).float().cpu().numpy()
                im_context_falloff = make_image_grid(
                    [scalar2img(t, vmin=0.0, vmax=1.0) for t in im_context_falloff], grid_width=grid_width
                )
                im_context_motion = np.concatenate([im_context_motion, im_context_falloff], axis=0)
            media_logger.log_image("Motion (Context)", im_context_motion)

    @ScopedTimer("CelsiusModel.prepare_supervision")
    def prepare_supervision(
        self,
        context: list[DataAndRenderingBatch],
        supervision: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        supervision_packs: list[CelsiusNRMSupervisionPack],
        media_logger: BatchMediaLogger | None,
    ) -> tuple[list[DataAndRenderingBatch], list[CelsiusNRMSupervisionPack]]:
        """
        Remove the bounding box region from supervision by marking them INVALID.
        """
        prepared_supervision: list[DataAndRenderingBatch] = []

        for bidx, (context_batch, batch) in enumerate(zip(context, supervision)):
            if (camera_data := batch.data.camera) is not None and cuboid_tracks is not None:
                assert camera_data.labels.flags is not None, "Camera labels must have flags"
                assert batch.rendering is not None and batch.rendering.camera is not None, (
                    "Rendering camera must be provided"
                )
                assert context_batch.rendering is not None and context_batch.rendering.camera is not None, (
                    "Rendering camera must be provided"
                )

                new_flags = camera_data.labels.flags
                new_velocity = camera_data.labels.velocity

                if self.motion_module is not None:
                    # For dynamic scenes, compute a proxy velocity for reference.
                    new_velocity = self._compute_data_and_rendering_batch_velocity(batch, cuboid_tracks[bidx])

                else:
                    # For static scenes, we have to make sure that supervision is masked at different timestamps.
                    new_flags = camera_data.labels.flags.clone()
                    cuboid_mask = self._mask_from_cuboid_tracks(
                        batch.rendering.camera.rays,
                        unpack_optional(batch.rendering.camera.rays_timestamps_us),
                        cuboid_tracks[bidx],
                    )

                    # If the supervision is close to a context, we can safely supervise with it.
                    s_t = batch.rendering.camera.timestamps_startend_us
                    s_t = s_t[:, 0] + (s_t[:, 1] - s_t[:, 0]) // 2
                    c_t = context_batch.rendering.camera.timestamps_startend_us
                    c_t = c_t[:, 0] + (c_t[:, 1] - c_t[:, 0]) // 2
                    in_context_mask = (s_t[:, None] - c_t).abs().min(
                        dim=-1
                    ).values < CelsiusNRMPrimitive.STATIC_CLIP_TIME_DIFF_S * 1e6
                    cuboid_mask[in_context_mask] = False
                    new_flags[cuboid_mask] |= RayFlags.INVALID

                batch = replace(
                    batch,
                    data=replace(
                        batch.data,
                        camera=replace(
                            camera_data, labels=replace(camera_data.labels, flags=new_flags, velocity=new_velocity)
                        ),
                    ),
                )

            if (
                (lidar_data := batch.data.lidar) is not None
                and cuboid_tracks is not None
                and self.motion_module is None
            ):
                assert lidar_data.labels.flags is not None, "Lidar labels must have flags"
                assert batch.rendering is not None and batch.rendering.lidar is not None, (
                    "Rendering lidar must be provided"
                )
                assert context_batch.rendering is not None and context_batch.rendering.lidar is not None, (
                    "Rendering lidar must be provided"
                )
                new_flags = lidar_data.labels.flags.clone()
                cuboid_mask = self._mask_from_cuboid_tracks(
                    batch.rendering.lidar.rays,
                    unpack_optional(batch.rendering.lidar.rays_timestamps_us),
                    cuboid_tracks[bidx],
                )

                # If the supervision is close to a context, we can safely supervise with it.
                s_t = batch.rendering.lidar.timestamps_startend_us
                s_t = s_t[:, 0] + (s_t[:, 1] - s_t[:, 0]) // 2
                c_t = context_batch.rendering.lidar.timestamps_startend_us
                c_t = c_t[:, 0] + (c_t[:, 1] - c_t[:, 0]) // 2
                in_context_mask = (s_t[:, None] - c_t).abs().min(
                    dim=-1
                ).values < CelsiusNRMPrimitive.STATIC_CLIP_TIME_DIFF_S * 1e6
                cuboid_mask[in_context_mask] = False

                new_flags[cuboid_mask] |= RayFlags.INVALID
                batch = replace(
                    batch,
                    data=replace(
                        batch.data, lidar=replace(lidar_data, labels=replace(lidar_data.labels, flags=new_flags))
                    ),
                )

            prepared_supervision.append(batch)

        return prepared_supervision, supervision_packs

    @ScopedTimer("CelsiusModel.prepare_context")
    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[DataAndRenderingBatch]:
        """
        For dynamic scenes, compute the velocity for the context if possible.
        """
        if cuboid_tracks is None or self.motion_module is None:
            return context

        prepared_context: list[DataAndRenderingBatch] = []

        for bidx, context_batch in enumerate(context):
            if (camera_data := context_batch.data.camera) is not None:
                new_context_velocity = camera_data.labels.velocity
                if self.motion_module is not None:
                    # For dynamic scenes, compute a proxy velocity for reference.
                    new_context_velocity = self._compute_data_and_rendering_batch_velocity(
                        context_batch, cuboid_tracks[bidx]
                    )
                context_batch = replace(
                    context_batch,
                    data=replace(
                        context_batch.data,
                        camera=replace(
                            camera_data,
                            labels=replace(camera_data.labels, velocity=new_context_velocity),
                        ),
                    ),
                )

            prepared_context.append(context_batch)

        return prepared_context

    @ScopedTimer("CelsiusModel.reconstruct")
    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        media_logger: BatchMediaLogger | None,
        compute_supervision_pack: bool = False,
    ) -> tuple[list[CelsiusNRMPrimitive], list[CelsiusNRMSupervisionPack] | None]:
        # Prepare the input context images
        batch_rgbs: list[torch.Tensor] = []
        batch_pluckers: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        batch_continuous_times: list[torch.Tensor] = []
        batch_time_remappings: list[TimeRemapping] = []
        batch_camera_idxs: list[torch.Tensor] = []

        for bidx, batch in enumerate(context):
            assert batch.rendering is not None, "Rendering must be provided"
            assert batch.rendering.camera is not None, "Rendering camera must be provided"

            rays = batch.rendering.camera.rays
            rays_cam_o, rays_cam_d = rays[..., :3], rays[..., 3:]
            num_imgs, img_height, img_width = rays.shape[:3]
            rgb = unpack_optional(unpack_optional(unpack_optional(batch.data).camera).labels.rgb)

            # Compute plucker embedding (dxo, d)
            plucker = torch.cat(
                [torch.cross(rays_cam_o * self.config.scene_rescale, rays_cam_d, dim=-1), rays_cam_d], dim=-1
            )
            assert plucker.shape == (
                num_imgs,
                img_height,
                img_width,
                6,
            ), f"Plucker shape must be (num_imgs, img_height, img_width, 6), but got {plucker.shape}"

            rays_timestamps_us = unpack_optional(batch.rendering.camera.rays_timestamps_us).squeeze(-1)
            assert rays_timestamps_us.shape == (
                num_imgs,
                img_height,
                img_width,
            ), f"Rays timestamps shape must be (num_imgs, img_height, img_width), but got {rays_timestamps_us.shape}"

            # Pool sensor idx into a (V,) tensor since we assume the sensor idx from all pixels are the same.
            assert batch.data.camera is not None, "Camera data must be provided"
            unique_sensor_idx = torch.tensor(
                [meta.unique_sensor_idx for meta in batch.data.camera.meta], dtype=torch.int64, device=rays.device
            )

            if bidx == 0 and (media_logger is not None and media_logger.should_log_media):
                input_img_rgbs: list[np.ndarray] = []
                for rgb_i in rgb:
                    rgb_np = rgb_i.cpu().numpy()
                    input_img_rgbs.append((rgb_np * 255).astype(np.uint8))
                num_views = len(unique_sensor_idx.unique())
                media_logger.log_image("Input RGB", make_image_grid(input_img_rgbs, grid_width=num_imgs // num_views))

            time_remapping = TimeRemapping.from_timestamps_startend_us(
                batch.rendering.camera.timestamps_startend_us, unique_sensor_idx
            )

            batch_rgbs.append(rgb)
            batch_pluckers.append(plucker)
            batch_continuous_times.append(time_remapping.timestamps_us_to_continuous_times(rays_timestamps_us))
            batch_time_remappings.append(time_remapping)
            batch_camera_idxs.append(unique_sensor_idx)

            if self.config.legacy_mask_input:
                valid_flag = ~self._mask_from_cuboid_tracks(
                    rays,
                    rays_timestamps_us,
                    cuboid_tracks[bidx] if (cuboid_tracks is not None and self.motion_module is None) else None,
                )
                assert valid_flag.shape == (
                    num_imgs,
                    img_height,
                    img_width,
                ), f"Valid flag shape must be (num_imgs, img_height, img_width), but got {valid_flag.shape}"
                batch_masks.append(valid_flag.float())

        # Assertions about shapes should come with the stack function
        rgbs_in = torch.stack(batch_rgbs, dim=0)
        B, V, H, W, _ = rgbs_in.shape

        rgbs_in = self.rgb_normalize(rearrange(rgbs_in, "B V H W C -> (B V) C H W"))
        pluckers_in = rearrange(torch.stack(batch_pluckers, dim=0), "B V H W C -> (B V) C H W")
        times_in = rearrange(torch.stack(batch_continuous_times, dim=0), "B V H W -> (B V) H W")

        # Embedding:
        if self.config.legacy_mask_input:
            masks_in = self.mask_normalize(rearrange(torch.stack(batch_masks, dim=0), "B V H W -> (B V) 1 H W"))
            rgbs_in = torch.cat([rgbs_in, masks_in], dim=1)
        x = self.patch_embed_img(rgbs_in)
        x = x + self.patch_embed_ray(pluckers_in)
        x = x + self.patch_embed_time(times_in)
        _, h, w, _ = x.shape  # h and w is the number of patches

        # Flatten to tokens and prepend sky/motion token
        x = rearrange(x, "(B V) h w C -> B (V h w) C", B=B, V=V)

        n_affine_tokens: int = 0
        if self.affine_module is not None:
            x, affine_token = self.affine_module.transform_tokens(x, torch.stack(batch_camera_idxs, dim=0))
            x = torch.cat([affine_token, x], dim=1)
            n_affine_tokens = affine_token.shape[1]

        n_motion_tokens: int = 0
        if self.motion_module is not None and self.motion_module.n_motion_tokens > 0:
            assert self.motion_module.motion_tokens is not None, "Motion tokens must not be None"
            x = torch.cat([self.motion_module.motion_tokens[None].repeat(B, 1, 1), x], dim=1)
            n_motion_tokens = self.motion_module.n_motion_tokens

        if self.sky_module is not None:
            x = torch.cat([self.sky_module.sky_token[None, None].repeat(B, 1, 1), x], dim=1)

        # Encoder:
        with ScopedTimer("CelsiusModel.backbone"):
            for block in self.encoder_blocks:
                if self.config.activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        # Optional layer normalization after the encoder
        x = self.encoder_norm(x)

        # Split back to images and other tokens
        if self.sky_module is not None:
            sky_token, x = x[:, :1], x[:, 1:]  # (B, 1, C), (B, V*h*w, C)
        else:
            sky_token = None

        if self.motion_module is not None and n_motion_tokens > 0:
            motion_tokens, x = x[:, :n_motion_tokens], x[:, n_motion_tokens:]
        else:
            motion_tokens = None

        if self.affine_module is not None and n_affine_tokens > 0:
            affine_token, x = (x[:, :n_affine_tokens], x[:, n_affine_tokens:])  # (B, 1, C), (B, V*h*w, C)
            affine_matrix, affine_bias = self.affine_module.decode_affine(affine_token)
        else:
            affine_matrix, affine_bias = None, None

        # Decoder:
        x = rearrange(x, "B (V h w) C -> (B V) C h w", B=B, V=V, h=h, w=w)
        gs_params = rearrange(self.deconv(x), "(B V) C H W -> B V H W C", B=B, V=V)
        if self.sky_module is not None:
            sky_params = rearrange(self.sky_module.sky_deconv(x), "(B V) C H W -> B V H W C", B=B, V=V)
        else:
            sky_params = None

        if self.motion_module is not None:
            if self.config.activation_checkpointing:
                motion_params = torch.utils.checkpoint.checkpoint(
                    self.motion_module, x, motion_tokens, B, V, H, W, use_reentrant=False
                )
            else:
                motion_params = self.motion_module(x, motion_tokens, B, V, H, W)
        else:
            motion_params = None

        # Build the reconstructed Gaussian splatting scene.
        primitives: list[CelsiusNRMPrimitive] = []
        supervision_packs: list[CelsiusNRMSupervisionPack] = []
        for bidx, batch in enumerate(context):
            supervision_pack = CelsiusNRMSupervisionPack()

            assert batch.rendering is not None, "Rendering must be provided"
            assert batch.rendering.camera is not None, "Rendering camera must be provided"

            rays = batch.rendering.camera.rays.reshape(-1, 6)  # [B, H, W, 6]
            rays_cam_o, rays_cam_d = rays[..., :3], rays[..., 3:]
            timestamps_us = unpack_optional(batch.rendering.camera.rays_timestamps_us).reshape(-1, 1)  # [B, H, W, 1]

            # Extract the parameters and apply activations
            gs_xyzd, gs_rgb, gs_scale, gs_rotation, gs_opacity = (
                gs_params[bidx]
                .flatten(0, 2)
                .split([3 if self.config.centroid_prediction == "xyz" else 1, 3, 3, 4, 1], dim=-1)
            )
            gaussian_params: GaussianParams = self.gaussian_activations(
                GaussianParams(
                    rgb=gs_rgb,
                    scale=gs_scale,
                    rotation=gs_rotation,
                    opacity=gs_opacity,
                    xyz=gs_xyzd if self.config.centroid_prediction == "xyz" else None,
                    distance=gs_xyzd if self.config.centroid_prediction == "distance" else None,
                ),
                rays_cam_o,
                rays_cam_d,
                scene_rescale=self.config.scene_rescale,
            )
            if self.config.centroid_prediction == "distance":
                gaussian_params.distance = self.gaussian_activations.distance(
                    gs_xyzd, scene_rescale=self.config.scene_rescale
                )
                if compute_supervision_pack:
                    supervision_pack.context_distance = unpack_optional(gaussian_params.distance).reshape(-1, H, W, 1)

            # road_mask is only populated during the primitive merging phase, under the circumstances that road-semantic
            # masks of context views are available. This attribute is mainly used in NRM-ply-init for SO.
            road_mask = None

            # While in theory it's better to not activate here but use the fused BCE with logits,
            # we still do it to avoid code duplication and accelerate the primitive rendering.
            sky_mask = None
            if sky_params is not None:
                sky_mask = sky_params[bidx].flatten(0, 2)
                sky_mask = self.sky_mask_activation(sky_mask)

            # Convert from continuous time (within range 0-1) speed/falloff to real-world speed/falloff.
            forward_speed_mps, falloff_sigma, unscaled_falloff_sigma = None, None, None
            if motion_params is not None and self.motion_module is not None:
                flow_dim = self.motion_module.velocity_dim
                forward_flow = motion_params[bidx][..., :flow_dim].flatten(0, 2)
                forward_speed_mps = (
                    self.forward_flow_activation(forward_flow) / batch_time_remappings[bidx].time_span_s()
                )

                # Assign the forward speed before replacing it with something else.
                # So we can still supervise it in the loss.
                if compute_supervision_pack:
                    supervision_pack.context_velocity = forward_speed_mps.reshape(-1, H, W, flow_dim)

                # Zero out speed for sky regions (simple heuristic)
                if sky_mask is not None:
                    forward_speed_mps = forward_speed_mps * (1 - sky_mask.detach().view(-1, 1))

                # Process sigma-related values
                if self.motion_module.motion_falloff:
                    raw_falloff_sigma = motion_params[bidx][..., flow_dim : flow_dim + 1]
                    unscaled_falloff_sigma = self.falloff_sigma_activation.activate(raw_falloff_sigma)
                    if compute_supervision_pack:
                        # unscaled_falloff_sigma is not flattened yet until activation.scale()
                        supervision_pack.context_unscaled_falloff_sigma = unscaled_falloff_sigma

                # At the beginning of training, we swap the velocity and sigma with gt.
                if batch.data.camera is not None and self.replace_velocity_with_context:
                    if (context_velocity := batch.data.camera.labels.velocity) is not None:
                        forward_speed_mps = set_zero(forward_speed_mps) + context_velocity.reshape(-1, flow_dim)
                        if unscaled_falloff_sigma is not None:
                            unscaled_falloff_sigma = set_zero(unscaled_falloff_sigma) + torch.where(
                                torch.any(context_velocity > 0.1, dim=-1, keepdim=True),
                                torch.zeros_like(unscaled_falloff_sigma),
                                torch.ones_like(unscaled_falloff_sigma),
                            )

                # Finally, scale the sigma to obtain the final value.
                if unscaled_falloff_sigma is not None:
                    falloff_sigma = self.falloff_sigma_activation.scale(
                        unscaled_falloff_sigma,
                        batch_time_remappings[bidx].time_span_s(),
                        batch_time_remappings[bidx].frame_gap_timestamps_us,
                    )

            # If dynamic content is not modeled, we explicitly remove those by setting opacity to 0
            # to avoid confusion in supervision.
            dynamic_bbox_mask = None
            if self.motion_module is None:
                dynamic_bbox_mask = self._mask_from_cuboid_tracks(
                    rays,
                    unpack_optional(batch.rendering.camera.rays_timestamps_us),
                    cuboid_tracks[bidx] if cuboid_tracks is not None else None,
                )
                dynamic_bbox_mask = dynamic_bbox_mask.view(-1, 1)

            # Log output from model context.
            if bidx == 0 and (media_logger is not None and media_logger.should_log_media):
                num_views = len(set([meta.unique_sensor_idx for meta in unpack_optional(batch.data.camera).meta]))
                self._log_context_media(
                    media_logger,
                    gaussian_params,
                    sky_mask,
                    (forward_speed_mps, unscaled_falloff_sigma),
                    (V, H, W),
                    grid_width=V // num_views,
                )

            primitive = CelsiusNRMPrimitive(
                positions=unpack_optional(gaussian_params.xyz),
                rotations=gaussian_params.rotation,
                scales=gaussian_params.scale,
                densities=gaussian_params.opacity,
                rgb=gaussian_params.rgb,
                timestamps_us=timestamps_us,
                forward_speed_mps=forward_speed_mps,
                falloff_sigma=falloff_sigma,
                dynamic_bbox_mask=dynamic_bbox_mask,
                sky_token=sky_token[bidx] if sky_token is not None else None,
                sky_head=self.sky_module.sky_head if self.sky_module is not None else None,
                sky_rotation=None,
                road_mask=road_mask,
                sky_mask=sky_mask,
                affine_matrix=affine_matrix[bidx] if affine_matrix is not None else None,
                affine_bias=affine_bias[bidx] if affine_bias is not None else None,
                gaussians_renderer=self.gaussians_renderer,
                checkpointing="render" if self.config.use_deferred_bp else "none",
                difix_model=self._difix_model,
            )

            if self.training:
                # Previously we add random bg color augment training to encourage Gaussian coverage and in turn depth stability
                # But we found it's not very stable, so we use black for now.
                solid_bg_color = torch.zeros(3, device=gaussian_params.opacity.device)
                primitive.set_solid_background_color(solid_bg_color)

                # Randomly replace sky Gaussians with sky MLPs to have it learned.
                if sky_mask is not None and torch.rand(1).item() > 0.8:
                    primitive.set_sky_mask_enabled(True)
                    primitive.set_solid_background_color(None)

            primitives.append(primitive)
            supervision_packs.append(supervision_pack)

        return primitives, supervision_packs if compute_supervision_pack else None
