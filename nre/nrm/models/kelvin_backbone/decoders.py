# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import math
import pickle

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.utils.checkpoint

from einops import einsum, rearrange, repeat
from torch import nn

from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.models.gaussians.utils import sh_degree_to_specular_dim
from nre.models.nn_extensions import TypedModuleList
from nre.nrm.config.models import (
    KelvinDAv3EncoderConfig,
    KelvinDPTDecoderConfig,
    KelvinModelConfig,
    KelvinPointQueryCADecoderConfig,
    KelvinTokenGSDecoderConfig,
)
from nre.nrm.models.activations import GaussianActivations, GaussianParams
from nre.nrm.models.blocks.aa_vit import AlternateAttentionVisionTransformer
from nre.nrm.models.blocks.attention import CrossAttentionBlock, KVProjector
from nre.nrm.models.blocks.dav3 import convert_dav3_state_dict_to_nrm
from nre.nrm.models.blocks.dpt import DPTFullHead
from nre.nrm.models.blocks.embeds import ContinuousTimeEmbed
from nre.nrm.models.kelvin_backbone.base import (
    KelvinLatent,
    KelvinMotionSupervision,
    KelvinMultiscaleFeaturesLatent,
    KelvinNRMSupervisionPack,
    _tokengs_init_weights,
)
from nre.nrm.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinSemanticClass,
    KelvinStaticLayer,
)
from nre.nrm.utils.motion import TimeRemapping, warp_points_with_cuboid_tracks
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.log import BatchMediaLogger
from nre.utils.misc import unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.visualize import make_image_grid, scalar2img


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class KelvinDecoderReturn:
    # Allowing all dynamic layers
    static_layer: KelvinStaticLayer | None
    dynamic_layers: list[KelvinDynamicLayer]
    supervision_pack: KelvinNRMSupervisionPack


class KelvinDecoderBase(nn.Module, ABC):
    @abstractmethod
    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        """
        Initialize the weights of the model from the loaded state dicts.
        """

    @abstractmethod
    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> list[KelvinDecoderReturn]:
        """
        Decode from the encoded latent into Gaussian parameters.
        Batches are used to pass in useful information about the raw context (e.g. timestamps, etc.)
        """

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please call decode() method directly.")

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs):
        # Do nothing by default
        pass

    def get_potential_unused_parameters(self) -> Iterator[nn.Parameter]:
        return iter([])


class KelvinTokenGSDecoder(KelvinDecoderBase):
    def __init__(self, config: KelvinTokenGSDecoderConfig, model_config: KelvinModelConfig):
        super().__init__()

        dim = model_config.encoder.embed_dim
        num_latent_heads = model_config.encoder.n_heads
        self.feature_norm = nn.LayerNorm(dim)
        self.kv_projector = KVProjector(
            dim=dim,
            n_heads=num_latent_heads,
            kv_bias=True,
            k_norm=config.use_qk_norm,
        )
        self.blocks = TypedModuleList(
            [
                CrossAttentionBlock(
                    dim,
                    num_latent_heads,
                    qkv_bias=True,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    qk_norm=config.use_qk_norm,
                    mlp_ratio=4.0,
                    dropout=0.0,
                    layer_scale_init_values=config.layer_scale_init_values,
                    kv_projector=None,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(dim) if config.use_decoder_norm else nn.Identity()
        self.gaussian_tokens = nn.Parameter(
            torch.randn(config.num_gaussian_tokens, dim) * config.gaussian_token_init_std
        )
        self.n_gaussians_per_token = model_config.patch_shape[0] * model_config.patch_shape[1]  # for now
        n_gaussian_params = 3 + 3 + 3 + 4 + 1
        self.token_to_gs_linear = nn.Linear(dim, n_gaussian_params * self.n_gaussians_per_token)
        self.gaussian_activations = GaussianActivations(model_config.activations)

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        self.apply(_tokengs_init_weights)
        torch.nn.init.trunc_normal_(self.token_to_gs_linear.weight, std=0.002)

    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> list[KelvinDecoderReturn]:
        """
        The returned GaussianParams will have shape (B, N, C)
        """
        batch_size = encoded_latent.batch_size
        deepest = encoded_latent.deepest
        encoder_feature = rearrange(deepest, "B V h w C -> B (V h w) C")
        encoder_feature = self.feature_norm(encoder_feature)

        k, v = self.kv_projector(k=encoder_feature, v=encoder_feature)
        gaussian_tokens = repeat(self.gaussian_tokens, "N C -> B N C", B=batch_size)
        for block in self.blocks:
            gaussian_tokens = block(gaussian_tokens, k, v)

        gaussian_tokens = self.norm(gaussian_tokens)

        gs_params_pre_activation = rearrange(
            self.token_to_gs_linear(gaussian_tokens),
            "B N (gs_per_token gs_params) -> B (N gs_per_token) gs_params",
            gs_per_token=self.n_gaussians_per_token,
        )
        gs_xyz, gs_rgb, gs_scale, gs_rotation, gs_opacity = gs_params_pre_activation.split([3, 3, 3, 4, 1], dim=-1)
        gs_params = GaussianParams(
            rgb=gs_rgb, scale=gs_scale, rotation=gs_rotation, opacity=gs_opacity, xyz=gs_xyz, activated=False
        )
        gs_params = self.gaussian_activations.forward(gs_params, scene_rescale=scene_rescale)

        return_values: list[KelvinDecoderReturn] = []
        for bidx in range(batch_size):
            gs_params_bidx = gs_params[bidx].flatten()
            return_values.append(
                KelvinDecoderReturn(
                    static_layer=KelvinStaticLayer(
                        rotations=gs_params_bidx.rotation,
                        scales=gs_params_bidx.scale,
                        rgb=gs_params_bidx.rgb,
                        positions=unpack_optional(gs_params_bidx.xyz),
                        densities=gs_params_bidx.opacity,
                    ),
                    dynamic_layers=[],
                    supervision_pack=KelvinNRMSupervisionPack(),
                )
            )
        return return_values


class KelvinDPTDecoder(KelvinDecoderBase):
    """
    DPT Head (compared to corresponding encoder this is ~5-10% of parameters & FLOPS)
    TODO: Compare SDT head from https://aigeeksgroup.github.io/AnyDepth/

    See schematic plot here:
    https://excalidraw.com/#json=e8F-fbXBIoMoihwIZxxwe,U5OH-X-P6QjCUh6i8xb5ag
    """

    class TimeModulatedMotionHead(nn.Module):
        """
        Takes in image tokens, source time, and target time, output motion offset from each frame to the target time.
        Currently the API jointly predicts forward & backward motion, as well as dynamic probability.
        - Pros: save compute by predicting two motion offsets at once.
        - Cons: somewhat introduces unncessary dependencies between the two motion offsets which should be independent.
            However, the original V-DPM style modulation does not work well if we de-couple them, where forward & backward
            motion offsets cancel each other out (i.e. the network does not seem to take src-time nor tgt-time into account).
            Two solutions might be possible: (1) Predict XYZ instead of offset, matching the exact V-DPM setting,
            (2) Adding attention mask (as in StreetForward / 4RC) may help, but haven't tried it yet.
        """

        def __init__(self, config: KelvinDPTDecoderConfig, model_config: KelvinModelConfig):
            super().__init__()
            ffn_type = (
                "mlp"
                if not isinstance(model_config.encoder, KelvinDAv3EncoderConfig)
                else model_config.encoder.ffn_type
            )
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
                ffn_type=ffn_type,
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

        @ScopedTimer("KelvinDPTDecoder.motion_head")
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
        self.context_head_init_values = [0.0] * 3 + [0.0, -1.0, 0.0] + [float("nan")] * self.n_semantic_classes
        # Time-conditioned training heads (motion-offset for now)
        self.context_motion_head: KelvinDPTDecoder.TimeModulatedMotionHead | nn.Identity = nn.Identity()
        if config.motion_depth > 0:
            self.context_motion_head = self.TimeModulatedMotionHead(config, model_config)

        # GS-training heads (scale, world-quaternion, opacity, higher-order SHs, depth-offset, uv-offset)
        self.sh_degree = 0
        self.depth_offset_dim = 1 if config.depth_offset else 0
        self.uv_offset_dim = 2 if config.uv_offset else 0
        gs_output_dim = (
            3 + 4 + 1 + sh_degree_to_specular_dim(self.sh_degree) + self.depth_offset_dim + self.uv_offset_dim
        )
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
        self.gaussians_head_init_values = (
            [float("nan")] * 3
            + [float("nan")] * 4
            + [float("nan")] * (1 + sh_degree_to_specular_dim(self.sh_degree))
            + [0.0] * (self.depth_offset_dim + self.uv_offset_dim)
        )

        self.cuboids_dims_padding = nn.Buffer(torch.tensor(model_config.track_padding_m, dtype=torch.float32))
        self.gaussian_activations = GaussianActivations(model_config.activations)

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        if "dav3" not in loaded_state_dicts:
            logger.warning("[KelvinDPTDecoder] No dav3 state dict found, skipping weight initialization.")
            return

        state_dict = convert_dav3_state_dict_to_nrm(
            loaded_state_dicts["dav3"],
            patch_embed_name=None,
            backbone_name=None,
            dpt_reassemble_name="depth_head.reassemble",
            dpt_depth_head_name="depth_head.fusion_head",
            dpt_rays_head_name=None,
            camera_encoder_name=None,
        )
        self.context_head.zero_init(init_values=self.context_head_init_values)
        self.gaussians_head.zero_init(init_values=self.gaussians_head_init_values)
        for prefix, module_state_dict in (
            ("rgb_fusion", self.rgb_fusion.state_dict()),
            ("context_head", self.context_head.state_dict()),
            ("context_motion_head", self.context_motion_head.state_dict()),
            ("gaussians_head", self.gaussians_head.state_dict()),
        ):
            state_dict |= {f"{prefix}.{k}": v for k, v in module_state_dict.items()}

        state_dict["cuboids_dims_padding"] = self.cuboids_dims_padding.data
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def inverse_log_transform(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.expm1(torch.abs(x)))

    def _log_gaussian_statistics(
        self,
        media_logger: BatchMediaLogger,
        gaussian_params: GaussianParams,
        supervision_pack: KelvinNRMSupervisionPack,
        context: DataAndRenderingBatch,
        grid_width: int,
    ) -> None:
        """
        Log statistics of the gaussian primitives.
        """
        if supervision_pack.context_rgb is not None:
            media_logger.log_image(
                "Context RGB",
                make_image_grid(
                    [t for t in (supervision_pack.context_rgb.detach().cpu().numpy() * 255).astype(np.uint8)],
                    grid_width=grid_width,
                ),
            )
        if (
            supervision_pack.context_depth is not None
            and supervision_pack.context_xyz is not None
            and context.rendering is not None
        ):
            rays = unpack_optional(unpack_optional(context.rendering.camera).rays)
            ref_xyz = rays[..., :3] + rays[..., 3:] * supervision_pack.context_depth
            diff_norm = torch.linalg.norm(ref_xyz - supervision_pack.context_xyz, dim=-1).detach().cpu().numpy()
            diff_norm_img = make_image_grid(
                [scalar2img(t, vmin=0.0, vmax=1.0) for t in diff_norm],
                grid_width=grid_width,
            )
            media_logger.log_image("Context XYZ Diff", diff_norm_img)
        if supervision_pack.context_world_normal is not None:
            pred_normal_img = ((supervision_pack.context_world_normal.detach().cpu().numpy() + 1.0) * 0.5 * 255).astype(
                np.uint8
            )
            normal_imgs = [t for t in pred_normal_img]
            if context.data.camera is not None and (gt_normal := context.data.camera.labels.normals) is not None:
                normal_imgs += [t for t in ((gt_normal.detach().cpu().numpy() + 1.0) * 0.5 * 255).astype(np.uint8)]
            media_logger.log_image(
                "Context World Normal",
                make_image_grid(normal_imgs, grid_width=grid_width),
            )
        if supervision_pack.context_semantic_logits is not None and context.data.camera is not None:
            pred_sem_img = KelvinSemanticClass.semantics_to_rgb(
                torch.argmax(supervision_pack.context_semantic_logits.detach(), dim=-1, keepdim=True).to(torch.uint8)
            )
            gt_sem_img = KelvinSemanticClass.semantics_to_rgb(
                KelvinSemanticClass.get_target_from_frame_labels(context.data.camera.labels)
            )
            media_logger.log_image(
                "Context Semantics",
                make_image_grid(
                    [p.cpu().numpy() for p in pred_sem_img] + [g.cpu().numpy() for g in gt_sem_img],
                    grid_width=grid_width,
                ),
            )

        opacity = gaussian_params.opacity.detach()[..., 0].cpu().numpy()
        media_logger.log_image(
            "Context Opacity",
            make_image_grid(
                [t for t in (opacity * 255).astype(np.uint8)],
                grid_width=grid_width,
            ),
        )

    def _dump_meshing_data(
        self,
        gaussian_params: GaussianParams,
        supervision_pack: KelvinNRMSupervisionPack,
        context: DataAndRenderingBatch,
    ):
        """
        Dump useful data for meshing using meshing tool.
        """
        rendering = unpack_optional(unpack_optional(context.rendering).camera)
        save_path = "/tmp/meshing_data.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "xyz": unpack_optional(gaussian_params.xyz).detach().cpu().numpy(),  # [V, H, W, 3]
                    "depth": unpack_optional(supervision_pack.context_depth).detach().cpu().numpy(),  # [V, H, W, 1]
                    "rgb": unpack_optional(supervision_pack.context_rgb).detach().cpu().numpy(),  # [V, H, W, 3]
                    "rays": unpack_optional(rendering.rays).detach().cpu().numpy(),  # [V, H, W, 6]
                    "intrinsics": [i.to_dict() for i in rendering.sensor_model_parameters],
                },
                f,
            )
        logger.info(f"Dumped meshing information to {save_path}...")

    def get_potential_unused_parameters(self) -> Iterator[torch.nn.Parameter]:
        # Note -- if we shard model parameters need to reconsider this design.
        return self.gaussians_head.parameters()

    @ScopedTimer("KelvinDPTDecoder.decode")
    @torch.autocast("cuda", enabled=False)
    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
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
        pred_depth_conf = torch.exp(depth_and_dconf[:, :, 1].unsqueeze(-1)) + 1.0  # (B, V, H, W, 1)

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

        context_prev_flow: torch.Tensor | None = None
        context_next_flow: torch.Tensor | None = None
        motion_supervisions: list[list[KelvinMotionSupervision]] = [[] for _ in range(B)]
        if isinstance(self.context_motion_head, self.TimeModulatedMotionHead):
            context_prev_flow, context_next_flow = self.context_motion_head.forward(
                encoded_latent,
                output_shape=(H, W),
                fusion_features=rgb_fusion_features if self.config.fusion_for_gs_motion else None,
                chunk_size=self.config.dpt_chunk_size,
                time_remappings=time_remappings,
                source_timestamps_us=source_timestamps_us,
                prev_target_timestamps_us=prev_target_timestamps_us,
                next_target_timestamps_us=next_target_timestamps_us,
            )
            context_prev_flow = context_prev_flow / scene_rescale
            context_next_flow = context_next_flow / scene_rescale
            motion_supervisions = [
                [
                    KelvinMotionSupervision(
                        source_timestamps_us=source_timestamps_us[bidx],
                        target_timestamps_us=prev_target_timestamps_us[bidx],
                        context_flow=context_prev_flow[bidx],
                    ),
                    KelvinMotionSupervision(
                        source_timestamps_us=source_timestamps_us[bidx],
                        target_timestamps_us=next_target_timestamps_us[bidx],
                        context_flow=context_next_flow[bidx],
                    ),
                ]
                for bidx in range(B)
            ]

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
                    None,
                    max_intersections_per_ray=2,
                    with_intersections_ts=False,
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

        if context_prev_flow is None or context_next_flow is None:
            raise RuntimeError(
                "No motion head found in the model, and cuboid tracks are not provided. Dynamic actors cannot be inferred."
            )

        # Forward and activate gaussian parameters
        gs_params_tensor = self.gaussians_head(
            img_feats,
            output_shape=(H, W),
            fusion_features=rgb_fusion_features if self.config.fusion_for_gs_motion else None,
            chunk_size=self.config.dpt_chunk_size,
        )
        gs_params_tensor = rearrange(gs_params_tensor, "(B V) C H W -> B V H W C", B=B, V=V)
        (
            gs_scale,
            gs_world_quaternion,
            gs_opacity,
            gs_specular,  # TODO: wire gs_specular through to GaussianParams when sh_degree > 0
            gs_depth_offset,
            gs_uv_offset,
        ) = gs_params_tensor.split(
            [3, 4, 1, sh_degree_to_specular_dim(self.sh_degree), self.depth_offset_dim, self.uv_offset_dim],
            dim=-1,
        )
        gs_depth = pred_depth
        if self.config.depth_offset:
            gs_depth_offset = gs_depth_offset / scene_rescale
            if media_logger is not None:
                media_logger.log("gs_stats/depth_offset", gs_depth_offset.detach().mean().item())
            # Linear activation on depth offset
            gs_depth = gs_depth + gs_depth_offset
        gs_distance = torch.stack([gs_depth[bidx] / renderings[bidx].distance_to_depth_scale for bidx in range(B)])

        # If scales and the potential UV offsets are predicted in the pixel unit (so that they're resolution-agnostic).
        # We need to scale them by the pixel scale, defined by footprint(cone) * distance.
        # Note that if rendered via 2D evaluation, the most accurate definition should be footprint(plane) * depth.
        # Here we simply assume that 3DGUT is always used.
        pixel_scale = torch.stack([renderings[bidx].ray_footprints for bidx in range(B)]) * gs_distance
        gs_scale = self.gaussian_activations.scale(gs_scale, scene_rescale=scene_rescale, pixel_scale=pixel_scale)
        gs_valid_mask = KelvinSemanticClass.opacity_mask_from_semantic_probs(
            torch.softmax(context_semantic_logits, dim=-1)
        )  # (B, V, H, W, 1)
        gs_opacity = self.gaussian_activations.opacity(gs_opacity) * (gs_valid_mask > 0.5).float().detach()
        gs_world_quaternion = self.gaussian_activations.rotation(gs_world_quaternion)
        gs_xyz = torch.stack(
            [renderings[bidx].rays[..., :3] + renderings[bidx].rays[..., 3:] * gs_distance[bidx] for bidx in range(B)]
        )
        if self.config.uv_offset:
            if media_logger is not None:
                media_logger.log("gs_stats/uv_offset", gs_uv_offset.detach().mean().item())
            gs_uv_offset = einsum(
                torch.stack(
                    [renderings[bidx].uv_directions_frame_end for bidx in range(B)],
                ),
                gs_uv_offset * pixel_scale,
                "B V UV D, B V H W UV -> B V H W D",
            )
            gs_xyz = gs_xyz + gs_uv_offset

        gs_params = GaussianParams(
            rgb=context_rgb,
            scale=gs_scale,
            rotation=gs_world_quaternion,
            opacity=gs_opacity,
            xyz=gs_xyz,
            activated=True,
        )

        supervision_packs = [
            KelvinNRMSupervisionPack(
                context_rgb=context_rgb[bidx],
                context_depth=pred_depth[bidx],
                context_depth_conf=pred_depth_conf[bidx],
                context_semantic_logits=context_semantic_logits[bidx],
                context_world_normal=context_world_normal[bidx],
                motion_supervisions=motion_supervisions[bidx],
            )
            for bidx in range(B)
        ]

        # Log Gaussian statistics and supervision packs information
        if media_logger is not None and media_logger.should_log_media:
            num_views = len(set([meta.unique_sensor_idx for meta in data[0].meta]))
            self._log_gaussian_statistics(
                media_logger,
                gs_params[0],
                supervision_packs[0],
                batches[0],
                grid_width=V // num_views,
            )
            # Log unprojected DPT point cloud (validation only; PLY overwritten each time; disabled for now)
            # media_logger.log_ply_point_cloud(
            #     "gspcd",
            #     gs_xyz[0].detach().reshape(-1, 3).cpu().numpy(),
            #     color=context_rgb[0].detach().reshape(-1, 3).cpu().numpy(),
            #     other_attributes={
            #         "confidence": pred_depth_conf[0].detach().reshape(-1).cpu().numpy(),
            #         "view_index": np.repeat(np.arange(V, dtype=np.uint8), H * W),
            #     },
            # )

        # Optionally dump meshing data
        # self._dump_meshing_data(gs_params[0], supervision_packs[0], batches[0])

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
                    supervision_pack=supervision_packs[bidx],
                )
            )

        return return_values


class KelvinPointQueryCADecoder(KelvinDPTDecoder):
    """Replaces the DPT gaussian head with a cross-attention head that uses
    depth-derived xyz positions (2D grid tokenization) as queries."""

    def __init__(self, config: KelvinPointQueryCADecoderConfig, model_config: KelvinModelConfig):
        dpt_config = KelvinDPTDecoderConfig(
            name="dpt-decoder",
            dpt_dim=config.dpt_dim,
            dpt_reassemble_hidden_dims=config.dpt_reassemble_hidden_dims,
            checkpointing=config.checkpointing,
            dpt_chunk_size=config.dpt_chunk_size,
            time_encoding_dim=config.time_encoding_dim,
            motion_depth=config.motion_depth,
        )
        super().__init__(dpt_config, model_config)
        self.config = config  # type: ignore[assignment]

        del self.gaussians_head
        del self.gaussians_head_init_values
        del self.sh_degree
        del self.depth_offset_dim
        del self.uv_offset_dim

        embed_dim = model_config.encoder.embed_dim
        num_heads = model_config.encoder.n_heads

        gs_per_token = config.grid_center_stride**2
        n_gs_params = 3 + 4 + 1  # scale, rotation, opacity

        self.xyz_embed = nn.Sequential(
            nn.Linear(gs_per_token * 3, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.ca_feature_norm = nn.LayerNorm(embed_dim)
        self.ca_kv_projector = KVProjector(
            dim=embed_dim,
            n_heads=num_heads,
            kv_bias=True,
            k_norm=config.use_qk_norm,
        )
        self.ca_blocks = TypedModuleList(
            [
                CrossAttentionBlock(
                    embed_dim,
                    num_heads,
                    qkv_bias=True,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    qk_norm=config.use_qk_norm,
                    mlp_ratio=4.0,
                    dropout=0.0,
                    layer_scale_init_values=config.layer_scale_init_values,
                    kv_projector=None,
                )
                for _ in range(config.ca_depth)
            ]
        )

        assert isinstance(model_config.encoder, KelvinDAv3EncoderConfig), (
            "KelvinPointQueryCADecoder requires a KelvinDAv3EncoderConfig encoder"
        )
        n_scales = len(model_config.encoder.take_block_indices)
        self.ca_kv_proj = nn.Linear(n_scales * embed_dim, embed_dim, bias=False)

        self.gs_linear = nn.Linear(embed_dim, gs_per_token * n_gs_params)

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        if "dav3" not in loaded_state_dicts:
            logger.warning("[KelvinPointQueryCADecoder] No dav3 state dict found, skipping weight initialization.")
            return

        state_dict = convert_dav3_state_dict_to_nrm(
            loaded_state_dicts["dav3"],
            patch_embed_name=None,
            backbone_name=None,
            dpt_reassemble_name="depth_head.reassemble",
            dpt_depth_head_name="depth_head.fusion_head",
            dpt_rays_head_name=None,
            camera_encoder_name=None,
        )
        self.context_head.zero_init(init_values=self.context_head_init_values)

        for m in [self.xyz_embed, self.ca_feature_norm, self.ca_kv_projector, self.ca_blocks]:
            m.apply(_tokengs_init_weights)
        torch.nn.init.trunc_normal_(self.gs_linear.weight, std=0.002)
        if self.gs_linear.bias is not None:
            torch.nn.init.constant_(self.gs_linear.bias, 0)
        for prefix, mod in (
            ("rgb_fusion", self.rgb_fusion),
            ("context_head", self.context_head),
            ("context_motion_head", self.context_motion_head),
            ("xyz_embed", self.xyz_embed),
            ("ca_feature_norm", self.ca_feature_norm),
            ("ca_kv_projector", self.ca_kv_projector),
            ("ca_blocks", self.ca_blocks),
            ("ca_kv_proj", self.ca_kv_proj),
            ("gs_linear", self.gs_linear),
        ):
            state_dict |= {f"{prefix}.{k}": v for k, v in mod.state_dict().items()}

        state_dict["cuboids_dims_padding"] = self.cuboids_dims_padding.data
        self.load_state_dict(state_dict, strict=True)

    def get_potential_unused_parameters(self) -> Iterator[nn.Parameter]:
        return self.context_motion_head.parameters()

    @ScopedTimer("KelvinPointQueryCADecoder.decode")
    @torch.autocast("cuda", enabled=False)
    def decode(
        self,
        encoded_latent: KelvinLatent,
        batches: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> list[KelvinDecoderReturn]:
        assert isinstance(encoded_latent, KelvinMultiscaleFeaturesLatent), (
            "Encoded latent must be a KelvinMultiscaleFeaturesLatent"
        )
        assert isinstance(self.config, KelvinPointQueryCADecoderConfig)
        renderings = [unpack_optional(unpack_optional(batch.rendering).camera) for batch in batches]
        data = [unpack_optional(batch.data.camera) for batch in batches]

        img_rgb = torch.stack([unpack_optional(d.labels.rgb) for d in data], dim=0)
        B, V, H, W, _ = img_rgb.shape
        img_feats = [rearrange(feat, "B V h w C -> (B V) h w C") for feat in encoded_latent.features]

        # --- Timestamps for motion ---
        source_timestamps_us = torch.stack(
            [unpack_optional(renderings[bidx].rays_timestamps_us) for bidx in range(B)], dim=0
        )  # (B, V, H, W, 1)
        frame_gap_timestamps_us = torch.stack(
            [time_remappings[bidx].frame_gap_timestamps_us for bidx in range(B)], dim=0
        ).to(img_rgb.device)  # (B, 2)
        prev_target_timestamps_us = source_timestamps_us - frame_gap_timestamps_us[..., 0][..., None, None, None]
        next_target_timestamps_us = source_timestamps_us + frame_gap_timestamps_us[..., 1][..., None, None, None]

        # --- Depth (shared with DPT) ---
        depth_and_dconf = self.depth_head(img_feats, output_shape=(H, W), chunk_size=self.config.dpt_chunk_size)
        depth_and_dconf = rearrange(depth_and_dconf, "(B V) C H W -> B V C H W", B=B, V=V)
        pred_depth = torch.exp(depth_and_dconf[:, :, 0].unsqueeze(-1) - math.log(scene_rescale))
        pred_depth_conf = torch.exp(depth_and_dconf[:, :, 1].unsqueeze(-1)) + 1.0

        # --- Context (shared with DPT, used for supervision + gaussian RGB) ---
        img_rgb_nchw = rearrange(img_rgb, "B V H W C -> (B V) C H W")
        if self.config.checkpointing:
            rgb_fusion_features = torch.utils.checkpoint.checkpoint(self.rgb_fusion, img_rgb_nchw, use_reentrant=False)
        else:
            rgb_fusion_features = self.rgb_fusion(img_rgb_nchw)
        context_features_tensor = self.context_head(
            img_feats, output_shape=(H, W), fusion_features=rgb_fusion_features, chunk_size=self.config.dpt_chunk_size
        )
        context_features_tensor = rearrange(context_features_tensor, "(B V) C H W -> B V H W C", B=B, V=V)
        context_rgb, context_world_normal, context_semantic_logits = context_features_tensor.split(
            [3, 3, self.n_semantic_classes],
            dim=-1,
        )
        context_rgb = self.gaussian_activations.rgb(context_rgb)
        context_world_normal = torch.nn.functional.normalize(context_world_normal, dim=-1)

        # --- XYZ from depth ---
        gs_distance = torch.stack([pred_depth[bidx] / renderings[bidx].distance_to_depth_scale for bidx in range(B)])
        full_xyz = torch.stack(
            [renderings[bidx].rays[..., :3] + renderings[bidx].rays[..., 3:] * gs_distance[bidx] for bidx in range(B)]
        )

        # --- Full-resolution flow + dynamic mask (mirrors KelvinDPTDecoder) ---
        if cuboid_tracks is not None:
            context_prev_flow_list: list[torch.Tensor] = []
            context_next_flow_list: list[torch.Tensor] = []
            context_dynamic_mask_list: list[torch.Tensor] = []
            for bidx in range(B):
                dynamic_track = CuboidTracks.Ops.subset_from_mask(
                    cuboid_tracks[bidx], cuboid_tracks[bidx].tracks_flags & TrackFlags.DYNAMIC != 0
                )
                context_xyz = full_xyz[bidx]
                context_prev_flow_list.append(
                    dynamic_track.warp_world_points_to_timestamps(
                        context_xyz,
                        source_timestamps_us[bidx],
                        prev_target_timestamps_us[bidx],
                        self.cuboids_dims_padding,
                    )
                    - context_xyz
                )
                context_next_flow_list.append(
                    dynamic_track.warp_world_points_to_timestamps(
                        context_xyz,
                        source_timestamps_us[bidx],
                        next_target_timestamps_us[bidx],
                        self.cuboids_dims_padding,
                    )
                    - context_xyz
                )
                context_dynamic_mask_list.append(
                    dynamic_track.point_intersection(
                        context_xyz,
                        source_timestamps_us[bidx],
                        self.cuboids_dims_padding,
                        return_dense_mask=False,
                    )
                )
            context_prev_flow = torch.stack(context_prev_flow_list, dim=0)
            context_next_flow = torch.stack(context_next_flow_list, dim=0)
            context_dynamic_mask = torch.stack(context_dynamic_mask_list, dim=0).squeeze(-1)
        else:
            context_prev_flow, context_next_flow = self.context_motion_head(
                encoded_latent,
                output_shape=(H, W),
                fusion_features=rgb_fusion_features,
                chunk_size=self.config.dpt_chunk_size,
                time_remappings=time_remappings,
                source_timestamps_us=source_timestamps_us,
                prev_target_timestamps_us=prev_target_timestamps_us,
                next_target_timestamps_us=next_target_timestamps_us,
            )
            context_prev_flow = context_prev_flow / scene_rescale
            context_next_flow = context_next_flow / scene_rescale
            context_dynamic_mask = torch.argmax(context_semantic_logits, dim=-1) == KelvinSemanticClass.MOVABLE.value

        # --- Downsample depth map to base point set ---
        s = self.config.xyz_downsample_stride
        xyz_ds = full_xyz[:, :, ::s, ::s, :]  # (B, V, H', W', 3)

        flow_prev_ds = context_prev_flow[:, :, ::s, ::s, :]
        flow_next_ds = context_next_flow[:, :, ::s, ::s, :]
        dyn_mask_ds = context_dynamic_mask[:, :, ::s, ::s]
        ts_src_ds = source_timestamps_us[:, :, ::s, ::s, :]
        ts_prev_ds = prev_target_timestamps_us[:, :, ::s, ::s, :]
        ts_next_ds = next_target_timestamps_us[:, :, ::s, ::s, :]

        footprints_ds = torch.stack([renderings[bidx].ray_footprints for bidx in range(B)])[:, :, ::s, ::s, :]
        distance_ds = gs_distance[:, :, ::s, ::s, :].detach()
        pscale_ds = footprints_ds * distance_ds * s

        # Sky/ego mask
        if self.config.use_gt_semantic_mask:
            gt_semantic = torch.stack(
                [KelvinSemanticClass.get_target_from_frame_labels(data[bidx].labels) for bidx in range(B)],
                dim=0,
            )
            gt_semantic_ds = gt_semantic[:, :, ::s, ::s, :]
            opacity_mask_ds = (
                (gt_semantic_ds != KelvinSemanticClass.SKY) & (gt_semantic_ds != KelvinSemanticClass.EGO)
            ).float()
        else:
            semantic_probs_ds = torch.softmax(context_semantic_logits, dim=-1)[:, :, ::s, ::s, :]
            opacity_mask_ds = KelvinSemanticClass.opacity_mask_from_semantic_probs(semantic_probs_ds)

        # --- 2D spatial grid tokenization ---
        cs = self.config.grid_center_stride
        K = cs * cs
        _, _, Hds, Wds, _ = xyz_ds.shape

        Hds_t = (Hds // cs) * cs
        Wds_t = (Wds // cs) * cs
        xyz_ds_t = xyz_ds[:, :, :Hds_t, :Wds_t, :]
        pscale_ds_t = pscale_ds[:, :, :Hds_t, :Wds_t, :]
        mask_ds_t = opacity_mask_ds[:, :, :Hds_t, :Wds_t, :]

        xyz_grp = rearrange(xyz_ds_t, "B V (h cs1) (w cs2) C -> B (V h w) (cs1 cs2) C", cs1=cs, cs2=cs)
        pscale_grp = rearrange(pscale_ds_t, "B V (h cs1) (w cs2) 1 -> B (V h w) (cs1 cs2) 1", cs1=cs, cs2=cs)
        mask_grp = rearrange(mask_ds_t, "B V (h cs1) (w cs2) 1 -> B (V h w) (cs1 cs2) 1", cs1=cs, cs2=cs)

        query_xyz = xyz_grp  # (B, M, K, 3)
        gs_xyz = rearrange(xyz_grp, "B M K C -> B (M K) C")
        pixel_scale = rearrange(pscale_grp, "B M K 1 -> B (M K) 1")
        point_valid_mask = rearrange(mask_grp, "B M K 1 -> B (M K) 1")

        ctx_rgb_ds_t = context_rgb[:, :, ::s, ::s, :][:, :, :Hds_t, :Wds_t, :]
        gs_ctx_rgb = rearrange(ctx_rgb_ds_t, "B V (h cs1) (w cs2) C -> B (V h w cs1 cs2) C", cs1=cs, cs2=cs)

        ctx_sem_ds_t = context_semantic_logits[:, :, ::s, ::s, :][:, :, :Hds_t, :Wds_t, :]
        gs_sem_class = rearrange(
            ctx_sem_ds_t.argmax(dim=-1), "B V (h cs1) (w cs2) -> B (V h w cs1 cs2)", cs1=cs, cs2=cs
        )

        ctx_normal_ds_t = context_world_normal[:, :, ::s, ::s, :][:, :, :Hds_t, :Wds_t, :]
        gs_normals = rearrange(ctx_normal_ds_t, "B V (h cs1) (w cs2) C -> B (V h w cs1 cs2) C", cs1=cs, cs2=cs)

        gs_prev_flow = rearrange(
            flow_prev_ds[:, :, :Hds_t, :Wds_t, :], "B V (h cs1) (w cs2) C -> B (V h w cs1 cs2) C", cs1=cs, cs2=cs
        )
        gs_next_flow = rearrange(
            flow_next_ds[:, :, :Hds_t, :Wds_t, :], "B V (h cs1) (w cs2) C -> B (V h w cs1 cs2) C", cs1=cs, cs2=cs
        )
        gs_dyn_mask = rearrange(
            dyn_mask_ds[:, :, :Hds_t, :Wds_t], "B V (h cs1) (w cs2) -> B (V h w cs1 cs2)", cs1=cs, cs2=cs
        )
        gs_ts_src = rearrange(
            ts_src_ds[:, :, :Hds_t, :Wds_t, :], "B V (h cs1) (w cs2) 1 -> B (V h w cs1 cs2)", cs1=cs, cs2=cs
        )
        gs_ts_prev = rearrange(
            ts_prev_ds[:, :, :Hds_t, :Wds_t, :], "B V (h cs1) (w cs2) 1 -> B (V h w cs1 cs2)", cs1=cs, cs2=cs
        )
        gs_ts_next = rearrange(
            ts_next_ds[:, :, :Hds_t, :Wds_t, :], "B V (h cs1) (w cs2) 1 -> B (V h w cs1 cs2)", cs1=cs, cs2=cs
        )

        # --- Cross-attention: xyz queries attend to multiscale backbone features ---
        queries = self.xyz_embed(query_xyz.flatten(-2, -1))  # (B, M, K*3) -> (B, M, C)

        kv_features = self.ca_kv_proj(
            torch.cat([rearrange(f, "B V h w C -> B (V h w) C") for f in encoded_latent.features], dim=-1)
        )
        kv_features = self.ca_feature_norm(kv_features)
        k_proj, v_proj = self.ca_kv_projector(k=kv_features, v=kv_features)

        for block in self.ca_blocks:
            queries = block(queries, k_proj, v_proj)

        # --- Predict & activate gaussian attributes ---
        gs_attrs = rearrange(self.gs_linear(queries), "B M (K P) -> B (M K) P", K=K)

        gs_scale, gs_rotation, gs_opacity = gs_attrs.split([3, 4, 1], dim=-1)
        gs_rgb = gs_ctx_rgb
        gs_scale = self.gaussian_activations.scale(gs_scale, scene_rescale=scene_rescale, pixel_scale=pixel_scale)
        gs_rotation = self.gaussian_activations.rotation(gs_rotation)
        gs_opacity = self.gaussian_activations.opacity(gs_opacity) * point_valid_mask.detach()

        gs_params = GaussianParams(
            rgb=gs_rgb,
            scale=gs_scale,
            rotation=gs_rotation,
            opacity=gs_opacity,
            xyz=gs_xyz,
            activated=True,
        )

        # --- Supervision packs (full-resolution flow for primitive_velocity loss) ---
        supervision_packs = [
            KelvinNRMSupervisionPack(
                context_rgb=context_rgb[bidx],
                context_depth=pred_depth[bidx],
                context_depth_conf=pred_depth_conf[bidx],
                context_semantic_logits=context_semantic_logits[bidx],
                context_world_normal=context_world_normal[bidx],
                motion_supervisions=[
                    KelvinMotionSupervision(
                        source_timestamps_us=source_timestamps_us[bidx],
                        target_timestamps_us=prev_target_timestamps_us[bidx],
                        context_flow=context_prev_flow[bidx],
                    ),
                    KelvinMotionSupervision(
                        source_timestamps_us=source_timestamps_us[bidx],
                        target_timestamps_us=next_target_timestamps_us[bidx],
                        context_flow=context_next_flow[bidx],
                    ),
                ],
            )
            for bidx in range(B)
        ]

        # --- Build return (static + dynamic layers) ---
        return_values: list[KelvinDecoderReturn] = []
        for bidx in range(B):
            gs_bidx = gs_params[bidx].flatten()
            world_points = unpack_optional(gs_bidx.xyz)
            prev_world_points = world_points + gs_prev_flow[bidx]
            next_world_points = world_points + gs_next_flow[bidx]

            dynamic_mask_bool = gs_dyn_mask[bidx]
            static_indices = torch.where(~dynamic_mask_bool)[0]
            dynamic_indices = torch.where(dynamic_mask_bool)[0]

            semantic_class_static = gs_sem_class[bidx][static_indices].unsqueeze(-1).to(torch.uint8)
            normals_static = gs_normals[bidx][static_indices]
            static_layer = KelvinStaticLayer(
                positions=world_points[static_indices],
                rotations=gs_bidx.rotation[static_indices],
                scales=gs_bidx.scale[static_indices],
                densities=gs_bidx.opacity[static_indices],
                rgb=gs_bidx.rgb[static_indices],
                semantic_class=semantic_class_static,
                normals=normals_static,
            )
            dynamic_layer = KelvinDynamicLayer(
                keyframe_positions=torch.stack(
                    [
                        prev_world_points[dynamic_indices],
                        world_points[dynamic_indices],
                        next_world_points[dynamic_indices],
                    ],
                    dim=1,
                ),
                keyframe_timestamps_us=torch.stack(
                    [
                        gs_ts_prev[bidx][dynamic_indices],
                        gs_ts_src[bidx][dynamic_indices],
                        gs_ts_next[bidx][dynamic_indices],
                    ],
                    dim=1,
                ),
                max_densities=gs_bidx.opacity[dynamic_indices],
                rotations=gs_bidx.rotation[dynamic_indices],
                scales=gs_bidx.scale[dynamic_indices],
                rgb=gs_bidx.rgb[dynamic_indices],
            )
            # Match KelvinDPTDecoder: dynamic layers span a short time range, so cap min density for visibility.
            dynamic_layer = dynamic_layer.ensure_minimum_density(0.75)
            return_values.append(
                KelvinDecoderReturn(
                    static_layer=static_layer,
                    dynamic_layers=[dynamic_layer],
                    supervision_pack=supervision_packs[bidx],
                )
            )
        return return_values


def make_decoder(config: KelvinModelConfig) -> KelvinDecoderBase:
    if isinstance(config.decoder, KelvinTokenGSDecoderConfig):
        return KelvinTokenGSDecoder(config.decoder, config)
    elif isinstance(config.decoder, KelvinPointQueryCADecoderConfig):
        return KelvinPointQueryCADecoder(config.decoder, config)
    elif isinstance(config.decoder, KelvinDPTDecoderConfig):
        return KelvinDPTDecoder(config.decoder, config)
    else:
        raise ValueError(f"Unsupported decoder config: {config.decoder}")
