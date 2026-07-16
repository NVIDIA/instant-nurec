# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

from dataclasses import dataclass

import torch
import torch.utils.checkpoint

from einops import rearrange
from torch import nn

from instant_nurec.datasets.tracks import CuboidTracks, TrackFlags
from instant_nurec.config_schema.models import (
    KelvinDAv3EncoderConfig,
    KelvinDPTDecoderConfig,
    KelvinModelConfig,
)
from instant_nurec.model.activations import GaussianActivations, GaussianParams
from instant_nurec.model.blocks.aa_vit import AlternateAttentionVisionTransformer
from instant_nurec.model.blocks.dpt import DPTFullHead
from instant_nurec.model.blocks.embeds import ContinuousTimeEmbed
from instant_nurec.model.backbone.base import (
    KelvinLatent,
    KelvinMultiscaleFeaturesLatent,
)
from instant_nurec.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinSemanticClass,
    KelvinStaticLayer,
)
from instant_nurec.utils.motion import TimeRemapping, warp_points_with_cuboid_tracks
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.misc import unpack_optional


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class KelvinDecoderReturn:
    # Allowing all dynamic layers
    static_layer: KelvinStaticLayer | None
    dynamic_layers: list[KelvinDynamicLayer]


class KelvinDPTDecoder(nn.Module):
    """
    DPT Head (compared to corresponding encoder this is ~5-10% of parameters & FLOPS)
    """

    class TimeModulatedMotionHead(nn.Module):
        """
        Takes in image tokens, source time, and target time, output motion offset from each frame to the target time.
        """

        def __init__(self, config: KelvinDPTDecoderConfig, model_config: KelvinModelConfig):
            super().__init__()
            self.embed_dim = model_config.encoder.embed_dim // 2
            self.vit = AlternateAttentionVisionTransformer(
                depth=config.motion_depth,
                embed_dim=self.embed_dim,  # Match encoder ViT
                n_heads=model_config.encoder.n_heads,
                mlp_ratio=4.0,
                aa_start_block_idx=0,
                img_pos_embed_shape=518 // model_config.patch_shape[0],
                n_cls_tokens=0,
                with_default_global_cls_tokens=False,
                rope_frequency=100.0,
                checkpointing="all" if config.checkpointing else "none",
                n_cls_tokens_aa=2,  # [CLS + SRC-Time]
                use_modulated_attention=True,
            )
            self.source_time_embed = ContinuousTimeEmbed(
                patch_shape=(1, 1),
                embed_dim=self.embed_dim,
                frequency_embedding_dim=config.time_encoding_dim,
                max_period=500.0,
            )
            self.source_time_norm = nn.LayerNorm(self.embed_dim)
            self.target_time_embed = ContinuousTimeEmbed(
                patch_shape=(1, 1),
                embed_dim=self.embed_dim // 2,
                frequency_embedding_dim=config.time_encoding_dim,
                max_period=500.0,
            )
            self.final_motion_head = DPTFullHead(
                input_dim=self.embed_dim * 2,
                reassemble_hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
                reassemble_dim=config.dpt_dim,
                output_dim=3 + 3,
                n_blocks=len(config.dpt_reassemble_hidden_dims),
                head_before_conv="1-layer",
                head_after_conv="2-layers",
                head_after_conv_dim=32,
                pos_embed_strength=0.1,
                checkpointing=config.checkpointing,
            )
            self.cls_token_norm = nn.LayerNorm(self.embed_dim)

        def _encode_timestamps_us(
            self, time_remappings: list[TimeRemapping], timestamps_us: torch.Tensor, embed_block: ContinuousTimeEmbed
        ) -> torch.Tensor:
            """
            Encode timestamps into continuous time embeddings.
            Input:
                time_remappings: (B, )
                timestamps_us: (B, V, H, W, 1)
                embed_block: ContinuousTimeEmbed
            Output:
                t_embed: (B, V, C)
            """
            B, V, H, W, _ = timestamps_us.shape
            frame_timestamp_us = timestamps_us[:, :, H // 2, W // 2, 0]
            t_float = torch.stack(
                [
                    time_remappings[bidx].timestamps_us_to_continuous_times(frame_timestamp_us[bidx])
                    for bidx in range(B)
                ],
                dim=0,
            )  # (B, V)
            t_embed = embed_block(rearrange(t_float, "B V -> (B V) 1 1"))
            t_embed = rearrange(t_embed, "(B V) 1 1 C -> B V C", B=B, V=V)
            return t_embed

        def forward(
            self,
            encoded_latent: KelvinMultiscaleFeaturesLatent,
            output_shape: tuple[int, int],
            fusion_features: torch.Tensor | None,
            chunk_size: int,
            *,  # Force keyword-only for timing to be clear
            time_remappings: list[TimeRemapping],
            source_timestamps_us: torch.Tensor,
            prev_target_timestamps_us: torch.Tensor,
            next_target_timestamps_us: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Input:
                encoded_latent: (B, V, h, w, C) & (B, V, n_cls_tokens, C)
                time_remappings: (B, )
                source_timestamps_us: (B, V, H, W, 1)
                prev_target_timestamps_us: (B, V, H, W, 1)
                next_target_timestamps_us: (B, V, H, W, 1)
            Output:
                Flow to be added on XYZ to reach prev timestamp: (B, V, H, W, 3)
                Flow to be added on XYZ to reach next timestamp: (B, V, H, W, 3)
            """
            H, W = output_shape
            B, V, _, _, _ = prev_target_timestamps_us.shape
            assert prev_target_timestamps_us.shape == (B, V, H, W, 1), (
                f"Expected (B, V, H, W, 1), got {prev_target_timestamps_us.shape}"
            )
            assert prev_target_timestamps_us.shape == next_target_timestamps_us.shape

            prev_target_t_embed = self._encode_timestamps_us(
                time_remappings, prev_target_timestamps_us, self.target_time_embed
            )
            next_target_t_embed = self._encode_timestamps_us(
                time_remappings, next_target_timestamps_us, self.target_time_embed
            )
            source_t_embed = self._encode_timestamps_us(time_remappings, source_timestamps_us, self.source_time_embed)
            source_t_embed = self.source_time_norm(source_t_embed)

            multiscale_features: list[torch.Tensor] = []
            for feat, src_cls_token in zip(encoded_latent.features, unpack_optional(encoded_latent.cls_tokens)):
                with torch.autocast("cuda", enabled=True):
                    src_cls_token = self.cls_token_norm(src_cls_token[..., self.embed_dim :])
                    img_feat, _ = self.vit.get_intermediate_features(
                        # Last the last half (x) and remove local_x part.
                        img_tokens=feat[..., self.embed_dim :],
                        block_indices=[len(self.vit.blocks) - 1],
                        global_cls_token=torch.cat([src_cls_token, source_t_embed.unsqueeze(-2)], dim=-2),
                        modulation_cond=torch.cat([prev_target_t_embed, next_target_t_embed], dim=-1),
                    )
                multiscale_features.append(rearrange(img_feat[-1], "B V h w C -> (B V) h w C"))

            x = self.final_motion_head(
                multiscale_features, output_shape=output_shape, fusion_features=fusion_features, chunk_size=chunk_size
            )
            x = rearrange(x, "(B V) C H W -> B V H W C", B=B, V=V)
            flow_prev, flow_next = x.split([3, 3], dim=-1)
            return flow_prev, flow_next

    def __init__(self, config: KelvinDPTDecoderConfig, model_config: KelvinModelConfig):
        super().__init__()
        self.config = config
        embed_dim = model_config.encoder.embed_dim
        self.n_blocks = 1
        if isinstance(model_config.encoder, KelvinDAv3EncoderConfig):
            self.n_blocks = len(model_config.encoder.take_block_indices)
        assert self.n_blocks == len(config.dpt_reassemble_hidden_dims), "Number of blocks must match"

        # Pre-training heads.
        # Output depth and depth confidence
        # or alternatively, world-points and its confidence (need the 5-layers before-conv setting)
        self.depth_head = DPTFullHead(
            input_dim=embed_dim,
            reassemble_hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
            reassemble_dim=config.dpt_dim,
            output_dim=1 + 1,
            n_blocks=self.n_blocks,
            head_before_conv="1-layer",
            head_after_conv="2-layers",
            head_after_conv_dim=32,
            pos_embed_strength=0.1,
            checkpointing=config.checkpointing,
        )

        # Up-scale RGB features for fusion (helps with HD output).
        rgb_fusion_dim = config.dpt_dim // 2
        self.rgb_fusion = nn.Sequential(
            nn.Conv2d(3, rgb_fusion_dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(rgb_fusion_dim // 4, rgb_fusion_dim // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(rgb_fusion_dim // 2, rgb_fusion_dim, 3, 1, 1),
            nn.GELU(),
        )

        # Context-training heads (RGB, world-normal, semantic-Logits).
        self.n_semantic_classes = len(KelvinSemanticClass)
        self.context_head = DPTFullHead(
            input_dim=embed_dim,
            reassemble_hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
            reassemble_dim=config.dpt_dim,
            output_dim=3 + 3 + self.n_semantic_classes,
            n_blocks=self.n_blocks,
            head_before_conv="1-layer",
            head_after_conv="2-layers",
            head_after_conv_dim=32,
            pos_embed_strength=0.1,
            checkpointing=config.checkpointing,
        )
        # Time-conditioned motion-offset head.
        self.context_motion_head = self.TimeModulatedMotionHead(config, model_config)

        # GS-training heads: predict (scale[3], world-quaternion[4], opacity[1]).
        gs_output_dim = 3 + 4 + 1
        self.gaussians_head = DPTFullHead(
            input_dim=embed_dim,
            reassemble_hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
            reassemble_dim=config.dpt_dim,
            output_dim=gs_output_dim,
            n_blocks=self.n_blocks,
            head_before_conv="1-layer",
            head_after_conv="2-layers",
            head_after_conv_dim=32,
            pos_embed_strength=0.1,
            checkpointing=config.checkpointing,
        )

        self.cuboids_dims_padding = nn.Buffer(torch.tensor(model_config.track_padding_m, dtype=torch.float32))
        self.gaussian_activations = GaussianActivations(model_config.activations)

    @torch.autocast("cuda", enabled=False)
    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
    ) -> list[KelvinDecoderReturn]:
        """
        The returned GaussianParams will have shape (B, V, H, W, C)
        """
        assert isinstance(encoded_latent, KelvinMultiscaleFeaturesLatent), (
            "Encoded latent must be a KelvinMultiscaleFeaturesLatent"
        )
        renderings = [unpack_optional(unpack_optional(batch.rendering).camera) for batch in batches]
        data = [unpack_optional(batch.data.camera) for batch in batches]

        img_rgb = torch.stack([unpack_optional(d.labels.rgb) for d in data], dim=0)
        B, V, H, W, _ = img_rgb.shape
        img_feats = [rearrange(feat, "B V h w C -> (B V) h w C") for feat in encoded_latent.features]

        # Forward and activate depth
        depth_and_dconf = self.depth_head(img_feats, output_shape=(H, W), chunk_size=self.config.dpt_chunk_size)
        depth_and_dconf = rearrange(depth_and_dconf, "(B V) C H W -> B V C H W", B=B, V=V)
        pred_depth = torch.exp(depth_and_dconf[:, :, 0].unsqueeze(-1) - math.log(scene_rescale))  # (B, V, H, W, 1)

        # Forward and activate context
        img_rgb = rearrange(img_rgb, "B V H W C -> (B V) C H W")
        if self.config.checkpointing:
            rgb_fusion_features = torch.utils.checkpoint.checkpoint(self.rgb_fusion, img_rgb, use_reentrant=False)
        else:
            rgb_fusion_features = self.rgb_fusion(img_rgb)
        context_features_tensor = self.context_head(
            img_feats, output_shape=(H, W), fusion_features=rgb_fusion_features, chunk_size=self.config.dpt_chunk_size
        )
        context_features_tensor = rearrange(context_features_tensor, "(B V) C H W -> B V H W C", B=B, V=V)
        (
            context_rgb,
            context_world_normal,
            context_semantic_logits,
        ) = context_features_tensor.split(
            [3, 3, self.n_semantic_classes],
            dim=-1,
        )
        context_rgb = self.gaussian_activations.rgb(context_rgb)
        context_world_normal = torch.nn.functional.normalize(context_world_normal, dim=-1)

        # For motion, determine the gap based on the time remappings for now
        source_timestamps_us = torch.stack(
            [unpack_optional(renderings[bidx].rays_timestamps_us) for bidx in range(B)], dim=0
        )
        frame_gap_timestamps_us = torch.stack(
            [time_remappings[bidx].frame_gap_timestamps_us for bidx in range(B)], dim=0
        ).to(img_rgb.device)
        prev_target_timestamps_us = source_timestamps_us - frame_gap_timestamps_us[..., 0][..., None, None, None]
        next_target_timestamps_us = source_timestamps_us + frame_gap_timestamps_us[..., 1][..., None, None, None]
        # This typically gives sharp motion boundary.
        context_dynamic_mask = torch.argmax(context_semantic_logits, dim=-1) == KelvinSemanticClass.MOVABLE.value

        context_prev_flow, context_next_flow = self.context_motion_head.forward(
            encoded_latent,
            output_shape=(H, W),
            fusion_features=None,
            chunk_size=self.config.dpt_chunk_size,
            time_remappings=time_remappings,
            source_timestamps_us=source_timestamps_us,
            prev_target_timestamps_us=prev_target_timestamps_us,
            next_target_timestamps_us=next_target_timestamps_us,
        )
        context_prev_flow = context_prev_flow / scene_rescale
        context_next_flow = context_next_flow / scene_rescale

        # If cuboid tracks are provided, use them instead.
        if cuboid_tracks is not None:
            # No need to re-scale points here since both cuboids and pred_depth are already scaled.
            context_prev_flow_list: list[torch.Tensor] = []
            context_next_flow_list: list[torch.Tensor] = []
            context_dynamic_mask_list: list[torch.Tensor] = []
            for bidx in range(B):
                dynamic_track = CuboidTracks.Ops.subset_from_mask(
                    cuboid_tracks[bidx], cuboid_tracks[bidx].tracks_flags & TrackFlags.DYNAMIC != 0
                )
                context_xyz = (
                    pred_depth[bidx].detach()
                    / renderings[bidx].distance_to_depth_scale
                    * renderings[bidx].rays[..., 3:]
                    + renderings[bidx].rays[..., :3]
                )
                # Auxiliary association via car-ray-cuboid intersection on movable rays. This serves
                # as a fallback when point-cuboid intersection misses (e.g. due to inaccurate depth).
                # Rays with multiple intersections are deemed ambiguous (-1).
                movable_mask = context_dynamic_mask[bidx]
                aux_ray_intersection_result = dynamic_track.ray_intersection(
                    renderings[bidx].rays[..., :3][movable_mask],
                    renderings[bidx].rays[..., 3:][movable_mask],
                    source_timestamps_us[bidx, ..., 0][movable_mask],
                    max_intersections_per_ray=2,
                )
                aux_movable_tracks_idx = aux_ray_intersection_result.intersections_tracks_idx[..., 0]
                aux_movable_tracks_idx[aux_ray_intersection_result.intersections_cnt != 1] = -1
                aux_tracks_idx = torch.full_like(movable_mask, -1, dtype=aux_movable_tracks_idx.dtype)
                aux_tracks_idx[movable_mask] = aux_movable_tracks_idx

                dynamic_mask, (prev_world_points, next_world_points) = warp_points_with_cuboid_tracks(
                    points=context_xyz,
                    source_timestamps_us=source_timestamps_us[bidx],
                    target_timestamps_us_list=[prev_target_timestamps_us[bidx], next_target_timestamps_us[bidx]],
                    dynamic_tracks=dynamic_track,
                    aux_tracks_idx=aux_tracks_idx,
                    cuboids_dims_padding=self.cuboids_dims_padding,
                )
                context_prev_flow_list.append(prev_world_points - context_xyz)
                context_next_flow_list.append(next_world_points - context_xyz)
                context_dynamic_mask_list.append(dynamic_mask)

            # Replace with ones from gt cuboids.
            context_prev_flow = torch.stack(context_prev_flow_list, dim=0)
            context_next_flow = torch.stack(context_next_flow_list, dim=0)
            context_dynamic_mask = torch.stack(context_dynamic_mask_list, dim=0)

        # Forward and activate gaussian parameters
        gs_params_tensor = self.gaussians_head(
            img_feats,
            output_shape=(H, W),
            fusion_features=None,
            chunk_size=self.config.dpt_chunk_size,
        )
        gs_params_tensor = rearrange(gs_params_tensor, "(B V) C H W -> B V H W C", B=B, V=V)
        gs_scale, gs_world_quaternion, gs_opacity = gs_params_tensor.split([3, 4, 1], dim=-1)
        gs_distance = torch.stack([pred_depth[bidx] / renderings[bidx].distance_to_depth_scale for bidx in range(B)])

        gs_scale = self.gaussian_activations.scale(gs_scale, scene_rescale=scene_rescale)
        gs_valid_mask = KelvinSemanticClass.opacity_mask_from_semantic_probs(
            torch.softmax(context_semantic_logits, dim=-1)
        )  # (B, V, H, W, 1)
        gs_opacity = self.gaussian_activations.opacity(gs_opacity) * (gs_valid_mask > 0.5).float().detach()
        gs_world_quaternion = self.gaussian_activations.rotation(gs_world_quaternion)
        gs_xyz = torch.stack(
            [renderings[bidx].rays[..., :3] + renderings[bidx].rays[..., 3:] * gs_distance[bidx] for bidx in range(B)]
        )

        gs_params = GaussianParams(
            rgb=context_rgb,
            scale=gs_scale,
            rotation=gs_world_quaternion,
            opacity=gs_opacity,
            xyz=gs_xyz,
        )

        # Build up the primitive
        return_values: list[KelvinDecoderReturn] = []
        for bidx in range(B):
            gs_bidx = gs_params[bidx].flatten()
            world_points = unpack_optional(gs_bidx.xyz)
            prev_world_points = world_points + context_prev_flow[bidx].reshape(-1, 3)
            next_world_points = world_points + context_next_flow[bidx].reshape(-1, 3)

            dynamic_mask = context_dynamic_mask[bidx].reshape(-1)
            static_mask = torch.where(~dynamic_mask)[0]
            dynamic_mask = torch.where(dynamic_mask)[0]

            # Derive per-gaussian semantic class from logits (argmax) for the static layer
            sem_class = torch.argmax(context_semantic_logits[bidx], dim=-1).reshape(-1)  # (V*H*W,)
            semantic_class_static = sem_class[static_mask].unsqueeze(-1).to(torch.uint8)  # (n_static, 1)
            normals_static = context_world_normal[bidx].reshape(-1, 3)[static_mask]  # (n_static, 3)

            static_layer = KelvinStaticLayer(
                positions=world_points[static_mask],
                rotations=gs_bidx.rotation[static_mask],
                scales=gs_bidx.scale[static_mask],
                densities=gs_bidx.opacity[static_mask],
                rgb=gs_bidx.rgb[static_mask],
                semantic_class=semantic_class_static,
                normals=normals_static,
            )

            dynamic_layer = KelvinDynamicLayer(
                keyframe_positions=torch.stack(
                    [
                        prev_world_points[dynamic_mask],
                        world_points[dynamic_mask],
                        next_world_points[dynamic_mask],
                    ],
                    dim=1,
                ),
                keyframe_timestamps_us=torch.stack(
                    [
                        prev_target_timestamps_us[bidx].reshape(-1)[dynamic_mask],
                        source_timestamps_us[bidx].reshape(-1)[dynamic_mask],
                        next_target_timestamps_us[bidx].reshape(-1)[dynamic_mask],
                    ],
                    dim=1,
                ),
                max_densities=gs_bidx.opacity[dynamic_mask],
                rotations=gs_bidx.rotation[dynamic_mask],
                scales=gs_bidx.scale[dynamic_mask],
                rgb=gs_bidx.rgb[dynamic_mask],
            )
            # Dynamic layers have a typical smaller timespan so their presence should be guaranteed.
            dynamic_layer = dynamic_layer.ensure_minimum_density(0.75)
            return_values.append(
                KelvinDecoderReturn(
                    static_layer=static_layer,
                    dynamic_layers=[dynamic_layer],
                )
            )

        return return_values
