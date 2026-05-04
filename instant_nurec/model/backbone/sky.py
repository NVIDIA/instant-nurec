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

import torch
import torch.nn as _nn

from einops import rearrange, repeat
from torch import nn

from instant_nurec.utils.nn_extensions import TypedModuleList
from instant_nurec.config_schema.models import (
    KelvinModelConfig,
    KelvinSkyCubemapDecoderConfig,
)
from instant_nurec.model.blocks.attention import CrossAttentionBlock, KVProjector
from instant_nurec.model.blocks.dpt import DPTFusionHead, DPTReassembleBlock
from instant_nurec.model.blocks.embeds import PatchEmbed, PositionalEmbed
from instant_nurec.model.backbone.base import KelvinLatent
from instant_nurec.utils.cubemap import cubemap_ray_directions
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.misc import unpack_optional


class _RGBNormalize(_nn.Module):
    """Wraps ``torchvision.transforms.v2.Normalize`` for tensors of shape
    ``(..., C, H, W)``. The mean/std are stored as non-persistent buffers
    so they don't appear in ``state_dict()``.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("_mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        return TF.normalize(
            x,
            mean=self._mean.flatten().tolist(),
            std=self._std.flatten().tolist(),
        )


class CubemapDecoderSky(nn.Module):
    """
    Let's start with brute force decoding.
    """

    class DPTFusionUpsampler(nn.Module):
        """Upsampler from patches to RGB values."""

        def __init__(self, embed_dim: int, dpt_dim: int, sky_cubemap_size: int):
            super().__init__()
            self.reassemble = DPTReassembleBlock(
                input_dim=embed_dim,
                output_dim=dpt_dim,
                n_blocks=4,
                hidden_dims=(embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim),
                pos_embed_strength=0.1,
            )
            self.decode_head = DPTFusionHead(
                input_dim=dpt_dim,
                output_dim=3,
                n_blocks=4,
                before_conv="1-layer",
                after_conv="2-layers",
                after_conv_dim=32,
                pos_embed_strength=0.1,
            )
            self.sky_cubemap_size = sky_cubemap_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_list = self.reassemble([x] * self.reassemble.n_blocks)
            return self.decode_head(x_list, output_shape=(self.sky_cubemap_size, self.sky_cubemap_size))

    class RayPatchEmbed(nn.Module):
        def __init__(self, embed_dim: int, pe_dim: int, patch_shape: tuple[int, int]):
            super().__init__()
            assert pe_dim % 3 == 0, "Embedding dimension must be divisible by 3."
            self.pe_dim = pe_dim
            self.ray_embed = PatchEmbed(
                patch_shape=patch_shape,
                input_dim=pe_dim,
                embed_dim=embed_dim,
                norm=True,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rx, ry, rz = x[:, 0], x[:, 1], x[:, 2]
            rx = PositionalEmbed.get_1d_sincos_nerf_embed(rx, self.pe_dim // 3)
            ry = PositionalEmbed.get_1d_sincos_nerf_embed(ry, self.pe_dim // 3)
            rz = PositionalEmbed.get_1d_sincos_nerf_embed(rz, self.pe_dim // 3)
            x = torch.cat([rx, ry, rz], dim=-1)
            return self.ray_embed(rearrange(x, "B h w C -> B C h w"))

    def __init__(self, config: KelvinSkyCubemapDecoderConfig, model_config: KelvinModelConfig):
        super().__init__()
        self.config = config
        self.patch_shape = model_config.patch_shape
        self.sky_cubemap_size = config.cubemap_size
        assert self.sky_cubemap_size % self.patch_shape[0] == 0, "Sky cubemap size must be divisible by patch shape"

        self.num_latent_heads = model_config.encoder.n_heads
        self.embed_dim = config.embed_dim
        self.depth = config.depth

        # Down-project and normalize the backbone features
        self.feature_transform = nn.Sequential(
            nn.Linear(model_config.encoder.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        # Skip connections for RGB information
        self.rgb_normalize = _RGBNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.patch_embed_img = PatchEmbed(
            patch_shape=self.patch_shape,
            input_dim=3,
            embed_dim=self.embed_dim,
            norm=True,
        )

        # Ray directional embeddings to build img-cubemap connections
        self.patch_embed_ray = self.RayPatchEmbed(self.embed_dim, 3 * 16, self.patch_shape)

        # Query rays for cross-attention
        self.token = nn.Parameter(torch.randn(self.embed_dim) * 0.02)
        self.query_rays = nn.Buffer(cubemap_ray_directions(self.sky_cubemap_size, device=torch.device("cuda")))

        self.blocks = TypedModuleList(
            [
                CrossAttentionBlock(
                    self.embed_dim,
                    self.num_latent_heads,
                    qkv_bias=True,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    qk_norm=True,
                    mlp_ratio=4.0,
                    dropout=0.0,
                    layer_scale_init_values=1.0,
                    kv_projector=None,
                )
                for _ in range(self.depth)
            ]
        )
        self.kv_projector = KVProjector(
            dim=self.embed_dim,
            n_heads=self.num_latent_heads,
            kv_bias=True,
            k_norm=True,
        )

        self.upsample = self.DPTFusionUpsampler(self.embed_dim, 128, self.sky_cubemap_size)

    def decode(self, encoded_latent: KelvinLatent, batches: list[DataAndRenderingBatch]) -> torch.Tensor:
        """
        Args:
            encoded_latent: KelvinMultiscaleFeaturesLatent
            batches: list[DataAndRenderingBatch]

        Returns:
            torch.Tensor: (B, 6, S, S, 3)
        """
        batch_size = encoded_latent.batch_size

        batch_rgbs: list[torch.Tensor] = []
        batch_rays: list[torch.Tensor] = []
        for batch in batches:
            data = unpack_optional(batch.data.camera)
            rendering = unpack_optional(unpack_optional(batch.rendering).camera)
            rgb = unpack_optional(data.labels.rgb)
            batch_rgbs.append(rgb)
            batch_rays.append(rendering.rays[..., 3:])

        # KV = backbone + RGB + Ray
        rgbs_in = torch.stack(batch_rgbs, dim=0)
        rgbs_in = self.patch_embed_img(self.rgb_normalize(rearrange(rgbs_in, "B V H W C -> (B V) C H W")))
        rgbs_in = rearrange(rgbs_in, "(B V) h w C -> B (V h w) C", B=batch_size)
        rays_in = torch.stack(batch_rays, dim=0)
        rays_in = self.patch_embed_ray(rearrange(rays_in, "B V H W C -> (B V) C H W"))
        rays_in = rearrange(rays_in, "(B V) h w C -> B (V h w) C", B=batch_size)
        features_in = self.feature_transform(encoded_latent.deepest)  # This contains normalization after proj.
        features_in = features_in.reshape(batch_size, -1, self.embed_dim) + rgbs_in + rays_in
        k, v = self.kv_projector(k=features_in, v=features_in)

        # Q = Ray
        queries = self.patch_embed_ray(rearrange(self.query_rays, "F SH SW C -> F C SH SW"))
        _, s, _, _ = queries.shape
        queries = repeat(queries, "F h w C -> B (F h w) C", B=batch_size) + self.token

        for block in self.blocks:
            queries = block(queries, k, v)

        with torch.autocast("cuda", enabled=False):
            queries = rearrange(queries.float(), "B (F h w) C -> (B F) h w C", F=6, h=s, w=s)
            if self.config.checkpointing:
                queries = torch.utils.checkpoint.checkpoint(self.upsample, queries, use_reentrant=False)
            else:
                queries = self.upsample(queries)
            queries = rearrange(queries, "(B F) C H W -> B F H W C", F=6)
            queries = torch.clamp(queries, min=-1.0, max=1.0) * 0.5 + 0.5

        return queries
