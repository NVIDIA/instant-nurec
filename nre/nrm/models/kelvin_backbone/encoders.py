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

from abc import ABC, abstractmethod
from typing import Iterator, cast

import numpy as np
import torch
import torchvision.transforms as transforms

from einops import rearrange
from torch import nn

from ncore.data import ConcreteCameraModelParametersUnion, OpenCVPinholeCameraModelParameters
from nre.models.nn_extensions import TypedModuleList
from nre.nrm.config.models import (
    KelvinDAv3EncoderConfig,
    KelvinModelConfig,
    KelvinTokenGSEncoderConfig,
)
from nre.nrm.models.blocks.aa_vit import AlternateAttentionVisionTransformer
from nre.nrm.models.blocks.attention import AttentionBlock
from nre.nrm.models.blocks.dav3 import CameraEncoder, convert_dav3_state_dict_to_nrm
from nre.nrm.models.blocks.embeds import PatchEmbed
from nre.nrm.models.kelvin_backbone.base import (
    KelvinFeatureLatent,
    KelvinLatent,
    KelvinMultiscaleFeaturesLatent,
    _tokengs_init_weights,
)
from nre.nrm.utils.motion import TimeRemapping
from nre.nrm.utils.sensor import to_simple_pinhole_model_parameters
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.geometry import tquat_to_se3_matrix
from nre.utils.log import BatchMediaLogger
from nre.utils.misc import unpack_optional
from nre.utils.profiling import ScopedTimer


logger = logging.getLogger(__name__)


class KelvinEncoderBase(nn.Module, ABC):
    @abstractmethod
    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        """
        Initialize the weights of the model from the loaded state dicts.
        """

    @abstractmethod
    def encode(
        self,
        batches: list[DataAndRenderingBatch],
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> KelvinLatent:
        """
        Encode the input batch into a latent representation.
        """

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please call encode() method directly.")

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs):
        # Do nothing by default
        pass

    def get_potential_unused_parameters(self) -> Iterator[nn.Parameter]:
        return iter([])


class KelvinTokenGSEncoder(KelvinEncoderBase):
    def __init__(self, config: KelvinTokenGSEncoderConfig, model_config: KelvinModelConfig):
        super().__init__()
        self.num_latent_heads = config.n_heads
        dim = config.embed_dim

        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        self.patch_embed_img = PatchEmbed(
            patch_shape=model_config.patch_shape,
            input_dim=3,
            embed_dim=dim,
            norm=True,
        )
        self.patch_embed_ray = PatchEmbed(patch_shape=model_config.patch_shape, input_dim=6, embed_dim=dim, norm=False)

        self.blocks = TypedModuleList(
            [
                AttentionBlock(
                    dim,
                    config.n_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    layer_scale_init_values=config.layer_scale_init_values,
                    qk_norm=config.use_qk_norm,
                )
                for _ in range(config.depth)
            ]
        )

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        self.apply(_tokengs_init_weights)

    def encode(
        self,
        batches: list[DataAndRenderingBatch],
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> KelvinLatent:
        batch_rgbs: list[torch.Tensor] = []
        batch_pluckers: list[torch.Tensor] = []

        for batch in batches:
            data = unpack_optional(batch.data.camera)
            rendering = unpack_optional(unpack_optional(batch.rendering).camera)

            rays = rendering.rays
            rays_cam_o, rays_cam_d = rays[..., :3], rays[..., 3:]
            num_imgs, img_height, img_width = rays.shape[:3]

            rgb = unpack_optional(data.labels.rgb)
            batch_rgbs.append(rgb)

            # Compute plucker embedding (dxo, d)
            plucker = torch.cat([torch.cross(rays_cam_o * scene_rescale, rays_cam_d, dim=-1), rays_cam_d], dim=-1)
            assert plucker.shape == (
                num_imgs,
                img_height,
                img_width,
                6,
            ), f"Plucker shape must be (num_imgs, img_height, img_width, 6), but got {plucker.shape}"
            batch_pluckers.append(plucker)

        rgbs_in = rearrange(torch.stack(batch_rgbs, dim=0), "B V H W C -> (B V) C H W")
        pluckers_in = rearrange(torch.stack(batch_pluckers, dim=0), "B V H W C -> (B V) C H W")

        # Patch embed: proj_img(rgb) + proj_ray(plucker), then a single LayerNorm on the sum.
        emb_img = self.patch_embed_img.proj(self.rgb_normalize(rgbs_in))
        emb_ray = self.patch_embed_ray.proj(pluckers_in)
        combined = emb_img + emb_ray
        combined = rearrange(combined, "B C h w -> B h w C")
        image_tokens = self.patch_embed_img.norm(combined)

        B = len(batches)
        batch_v, patch_h, patch_w, _ = image_tokens.shape
        num_views = batch_v // B
        image_tokens = rearrange(image_tokens, "(B V) h w C -> B (V h w) C", B=B, V=num_views)

        for block in self.blocks:
            image_tokens = block(image_tokens)

        # Reshape to image
        feature = rearrange(image_tokens, "B (V h w) C -> B V h w C", V=num_views, h=patch_h, w=patch_w)
        return KelvinFeatureLatent(feature=feature)


class KelvinDAv3Encoder(KelvinEncoderBase):
    def __init__(self, config: KelvinDAv3EncoderConfig, model_config: KelvinModelConfig):
        super().__init__()
        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)

        embed_dim = config.embed_dim // 2
        patch_shape = model_config.patch_shape

        self.patch_embed_img = PatchEmbed(
            patch_shape=patch_shape,
            input_dim=3,
            embed_dim=embed_dim,
            norm=False,
        )
        self.embed_camera = CameraEncoder(input_dim=9, output_dim=embed_dim)

        self.vit = AlternateAttentionVisionTransformer(
            depth=config.depth,
            embed_dim=embed_dim,
            n_heads=config.n_heads,
            mlp_ratio=4.0,
            aa_start_block_idx=config.aa_start_block_idx,
            img_pos_embed_shape=518 // patch_shape[0],
            n_cls_tokens=1,
            with_default_global_cls_tokens=False,
            rope_frequency=100.0,
            ffn_type=config.ffn_type,
            checkpointing=config.checkpointing,
        )
        self.take_block_indices = config.take_block_indices

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        if "dav3" not in loaded_state_dicts:
            logger.warning("[KelvinDAv3Encoder] No dav3 state dict found, skipping weight initialization.")
            return

        state_dict = convert_dav3_state_dict_to_nrm(
            loaded_state_dicts["dav3"],
            patch_embed_name="patch_embed_img",
            backbone_name="vit",
            dpt_reassemble_name=None,
            dpt_depth_head_name=None,
            dpt_rays_head_name=None,
            camera_encoder_name="embed_camera",
        )
        del state_dict["vit.default_global_cls_tokens"]
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _fov_wh_from_pinhole(pinhole_parameters: OpenCVPinholeCameraModelParameters) -> torch.Tensor:
        """
        Computes the fov and width/height from the pinhole parameters.
        """
        fov_w = 2 * np.arctan2(pinhole_parameters.resolution[0] / 2, pinhole_parameters.focal_length[0])
        fov_h = 2 * np.arctan2(pinhole_parameters.resolution[1] / 2, pinhole_parameters.focal_length[1])
        return torch.tensor([fov_w, fov_h]).float()

    @ScopedTimer("KelvinDAv3Encoder.encode")
    @torch.autocast("cuda", enabled=False)
    def encode(
        self,
        batches: list[DataAndRenderingBatch],
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
        media_logger: BatchMediaLogger | None = None,
    ) -> KelvinLatent:
        batch_rgbs: list[torch.Tensor] = []
        batch_c2ws: list[torch.Tensor] = []
        batch_fovs: list[torch.Tensor] = []

        for batch in batches:
            data = unpack_optional(batch.data.camera)
            rendering = unpack_optional(unpack_optional(batch.rendering).camera)

            rgb = unpack_optional(data.labels.rgb)
            num_imgs, _, _ = rgb.shape[:3]
            batch_rgbs.append(rgb)

            # Use end of frame pose for c2w approximation
            c2w_frame_end = tquat_to_se3_matrix(rendering.poses_tquat_startend[:, 1, :], unbatch=False)
            c2w_frame_end[:, :3, 3] *= scene_rescale
            batch_c2ws.append(c2w_frame_end)

            # Use simple pinhole model for fov approximation (TODO: Investigate?)
            # Since prediction is depth so we should probably hint rays as accurate as possible.
            pinhole_parameters = [
                to_simple_pinhole_model_parameters(
                    cast(ConcreteCameraModelParametersUnion, rendering.sensor_model_parameters[vidx]),
                    method="horizontal",
                    reduce="min",
                    percentile=1.0,
                )
                for vidx in range(num_imgs)
            ]
            fov_wh = torch.stack([self._fov_wh_from_pinhole(pinhole_parameters[vidx]) for vidx in range(num_imgs)]).to(
                rgb.device
            )
            batch_fovs.append(fov_wh)

        # Assertions about shapes should come with the stack function
        rgbs_in = torch.stack(batch_rgbs, dim=0)
        B, V, H, W, _ = rgbs_in.shape

        x = self.patch_embed_img(self.rgb_normalize(rearrange(rgbs_in, "B V H W C -> (B V) C H W")))
        _, h, w, _ = x.shape  # h and w is the number of patches
        x = rearrange(x, "(B V) h w C -> B V h w C", B=B, V=V)

        # Compute camera encoding
        c2w_in, fov_in = torch.stack(batch_c2ws, dim=0), torch.stack(batch_fovs, dim=0)
        camera_encodings = self.embed_camera.forward(c2w_in, fov_in)

        with torch.autocast("cuda", enabled=True):
            img_feats, cls_tokens = self.vit.get_intermediate_features(
                x, block_indices=self.take_block_indices, global_cls_token=camera_encodings.unsqueeze(2)
            )

        return KelvinMultiscaleFeaturesLatent(features=img_feats, cls_tokens=cls_tokens)


def make_encoder(config: KelvinModelConfig) -> KelvinEncoderBase:
    if isinstance(config.encoder, KelvinTokenGSEncoderConfig):
        return KelvinTokenGSEncoder(config.encoder, config)
    elif isinstance(config.encoder, KelvinDAv3EncoderConfig):
        return KelvinDAv3Encoder(config.encoder, config)
    else:
        raise ValueError(f"Unsupported encoder config: {config.encoder}")
