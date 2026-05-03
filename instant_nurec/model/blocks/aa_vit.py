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

import functools

from typing import Callable, Literal, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint

from einops import rearrange, repeat

from instant_nurec.model.blocks.attention import AttentionBlock, ModulatedAttentionBlock
from instant_nurec.model.blocks.embeds import RotaryPositionEmbed2D


class AlternateAttentionVisionTransformer(nn.Module):
    """
    DINOv2 ViT network with multi-image inputs, based on Alternate attention (AA) blocks.
    RoPE is added only for the local per-image attention layers.

    Reference:
    - https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/model/dinov2/vision_transformer.py
    """

    IMG_POS_EMBED_INTERP_OFFSET: float = 0.1

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float,
        aa_start_block_idx: int,
        img_pos_embed_shape: int,
        n_cls_tokens: int,
        with_default_global_cls_tokens: bool,
        rope_frequency: float,
        checkpointing: Literal["all", "local", "none"] = "none",
        n_cls_tokens_aa: int | None = None,
        use_modulated_attention: bool = False,
    ):
        """
        Args:
            [Attention-block-related]
            depth: int, the number of blocks
            embed_dim: int, the dimension of the embedding
            n_heads: int, the number of heads
            mlp_ratio: float, the ratio of the MLP hidden dimension to the input dimension
            aa_start_block_idx: int, the index of the block to start the AA blocks
            checkpointing: whether to checkpoint all blocks, only-local blocks (since global blocks recomputation is compute-heavy), or none

            [Embeddings/CLS tokens-related]
            img_pos_embed_shape: int, the shape of the positional embedding for the image tokens, side length of the square image
            n_cls_tokens: int, the number of CLS tokens
            n_cls_tokens_aa: int | None, the number of CLS tokens for the AA blocks, if None, then n_cls_tokens is used
            with_default_global_cls_tokens: bool, whether to use the default global CLS tokens
            rope_frequency: float, the frequency of the RoPE

            use_modulated_attention: if True, use ModulatedAttentionBlock for every layer (pass ``modulation_cond`` in
                ``get_intermediate_features``; last dim must match ``embed_dim``).
        """
        super().__init__()
        self.cls_tokens = nn.Parameter(torch.zeros(n_cls_tokens, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(n_cls_tokens, embed_dim))
        self.n_cls_tokens = n_cls_tokens
        self.n_cls_tokens_aa = n_cls_tokens_aa if n_cls_tokens_aa is not None else n_cls_tokens

        self.img_pos_embed_shape = img_pos_embed_shape
        self.img_pos_embed = nn.Parameter(torch.zeros(img_pos_embed_shape, img_pos_embed_shape, embed_dim))

        # This is the query camera tokens when not provided by the user (ref view + src views)
        self.default_global_cls_tokens = (
            nn.Parameter(torch.randn(2, self.n_cls_tokens_aa, embed_dim)) if with_default_global_cls_tokens else None
        )

        self.aa_start_block_idx = aa_start_block_idx
        self.use_modulated_attention = use_modulated_attention
        self.rope = RotaryPositionEmbed2D(frequency=rope_frequency)
        self.norm_layer = nn.LayerNorm([embed_dim])

        block_cls = ModulatedAttentionBlock if use_modulated_attention else AttentionBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    input_dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    layer_norm_eps=1e-6,
                    layer_scale_init_values=1.0,
                    qk_norm=i >= aa_start_block_idx,
                    rope=self.rope if i >= aa_start_block_idx else None,
                )
                for i in range(depth)
            ]
        )
        self.checkpointing = checkpointing

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Please call get_intermediate_layers() instead to obtain intermediate features.")

    def get_interpolated_img_pos_embed(self, height: int, width: int):
        """
        Interpolate the positional embedding for the image tokens to the new height and width.
        Returns:
            (height, width, embed_dim) tensor
        """
        if (height, width) == (self.img_pos_embed_shape, self.img_pos_embed_shape):
            return self.img_pos_embed

        dtype = self.img_pos_embed.dtype
        # Historical kludge: add a small number to avoid floating point error in the
        # interpolation, see https://github.com/facebookresearch/dino/issues/8
        sh = float(height + self.IMG_POS_EMBED_INTERP_OFFSET) / self.img_pos_embed_shape
        sw = float(width + self.IMG_POS_EMBED_INTERP_OFFSET) / self.img_pos_embed_shape
        patch_pos_embed = nn.functional.interpolate(
            self.img_pos_embed.float().moveaxis(2, 0)[None],  # (H, W, embed_dim) -> (1, embed_dim, H, W)
            mode="bicubic",
            antialias=False,
            scale_factor=(sh, sw),
        )
        assert (height, width) == patch_pos_embed.shape[-2:]

        return patch_pos_embed[0].moveaxis(0, 2).to(dtype)

    def get_rope_positions(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the rope positions for the global and local attention.
        Returns:
            (n_CLS_aa + hw, 2) tensor, the rope positions for the global attention
            (n_CLS_aa + hw, 2) tensor, the rope positions for the local attention
        """
        rope_pos_local = torch.cartesian_prod(
            torch.arange(height, device=device), torch.arange(width, device=device)
        ).reshape(-1, 2)
        rope_pos_global = torch.zeros_like(rope_pos_local)

        # Arange for CLS tokens
        rope_pos_cls = repeat(
            torch.arange(self.n_cls_tokens_aa, device=device, dtype=rope_pos_local.dtype), "n -> n D", D=2
        )
        rope_pos_global = torch.cat([rope_pos_cls, rope_pos_global + self.n_cls_tokens_aa], dim=0)
        rope_pos_local = torch.cat([rope_pos_cls, rope_pos_local + self.n_cls_tokens_aa], dim=0)

        return rope_pos_global, rope_pos_local

    def _forward_attention_block(
        self,
        block_fn: Callable,
        block_type: Literal["global", "local"],
        x: torch.Tensor,
        rope_positions: torch.Tensor | None,
        modulation_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.checkpointing == "all" or (self.checkpointing == "local" and block_type == "local"):
            block_fn = functools.partial(torch.utils.checkpoint.checkpoint, block_fn, use_reentrant=False)

        if self.use_modulated_attention:
            assert modulation_cond is not None, "modulation_cond is required when use_modulated_attention=True"
            if block_type == "local":
                modulation_cond = rearrange(modulation_cond, "B V C -> (B V) 1 C")
            return cast(torch.Tensor, block_fn(x, modulation_cond, rope_positions=rope_positions))

        else:
            return cast(torch.Tensor, block_fn(x, rope_positions=rope_positions))

    def get_intermediate_features(
        self,
        img_tokens: torch.Tensor,
        block_indices: list[int],
        global_cls_token: torch.Tensor | None = None,
        modulation_cond: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Returns the features from the intermediate blocks specified by block_indices.
        Args:
            img_tokens: (B, V, h, w, C) tensor -- already embedded in token space (positional encodings yet to be added)
            block_indices: list of integers, the indices of the intermediate blocks to return
            global_cls_token: (B, V, n_CLS_aa, C) tensor, the global CLS token to use for the AA blocks
            modulation_cond: (B, V, C) per-view conditioning for ModulatedAttentionBlock; required if
                ``use_modulated_attention`` is True.
        Returns:
            img_features: list of (B, V, h, w, C*2) tensors, the image token features
            cls_features: list of (B, V, n_CLS_aa, C*2) tensors, the CLS token features
        """
        if (last_block_idx := len(self.blocks) - 1) not in block_indices:
            raise ValueError(
                f"Last block index {last_block_idx} is not in block_indices {block_indices}. Model is not fully forwarded."
            )
        if self.use_modulated_attention and modulation_cond is None:
            raise ValueError("modulation_cond must be provided when use_modulated_attention=True")

        B, V, h, w, C = img_tokens.shape

        # Prepend CLS and add positional embedding to img tokens -> (B, V, n_CLS + h * w, C)
        x = torch.cat(
            [
                repeat(self.cls_tokens + self.cls_pos_embed, "n C -> B V n C", B=B, V=V),
                rearrange(img_tokens + self.get_interpolated_img_pos_embed(h, w), "B V h w C -> B V (h w) C"),
            ],
            dim=-2,
        )

        # Prepare rope positions to be used
        global_rope_pos, local_rope_pos = self.get_rope_positions(h, w, device=x.device)
        global_rope_pos = repeat(global_rope_pos, "N D -> B (V N) D", B=B, V=V)
        local_rope_pos = repeat(local_rope_pos, "N D -> (B V) N D", B=B, V=V)

        # Forward pass
        output_img_features: list[torch.Tensor] = []
        output_cls_features: list[torch.Tensor] = []

        local_x: torch.Tensor | None = None  # Last x after local attention
        for block_idx, block in enumerate(self.blocks):
            # For early blocks, perform only local attention without rope
            if block_idx < self.aa_start_block_idx:
                x = rearrange(x, "B V N C -> (B V) N C")
                x = self._forward_attention_block(block.forward, "local", x, None, modulation_cond)
                x = rearrange(x, "(B V) N C -> B V N C", B=B, V=V)
                local_x = x

            # At the transition, swap out local CLS token into global CLS token
            if block_idx == self.aa_start_block_idx:
                # Fallback to default ones if not provided by the user
                if global_cls_token is None:
                    assert self.default_global_cls_tokens is not None, (
                        "Please enable with_default_global_cls_tokens for the model"
                    )
                    first_img_token = self.default_global_cls_tokens[0:1]  # (1, n_CLS_aa, C)
                    other_img_tokens = self.default_global_cls_tokens[1:2].repeat(V - 1, 1, 1)
                    global_cls_token = torch.cat([first_img_token, other_img_tokens], dim=0).repeat(
                        B, 1, 1, 1
                    )  # (B, V, n_CLS_aa, C)

                x = torch.cat([global_cls_token, x[:, :, self.n_cls_tokens :]], dim=2)

            # For later odd blocks, perform global attention with rope
            if (block_idx >= self.aa_start_block_idx) and (block_idx % 2 == 1):
                x = rearrange(x, "B V N C -> B (V N) C")
                x = self._forward_attention_block(block.forward, "global", x, global_rope_pos, modulation_cond)
                x = rearrange(x, "B (V N) C -> B V N C", V=V)

            # For later even blocks, perform local attention without rope
            if (block_idx >= self.aa_start_block_idx) and (block_idx % 2 == 0):
                x = rearrange(x, "B V N C -> (B V) N C")
                x = self._forward_attention_block(block.forward, "local", x, local_rope_pos, modulation_cond)
                x = rearrange(x, "(B V) N C -> B V N C", B=B, V=V)
                local_x = x

            # Append the output features
            if block_idx in block_indices:
                assert local_x is not None
                output_cls_features.append(
                    torch.cat(
                        [local_x[:, :, : self.n_cls_tokens_aa], x[:, :, : self.n_cls_tokens_aa]],
                        dim=-1,
                    )
                )
                output_img_features.append(
                    torch.cat(
                        [
                            local_x[:, :, self.n_cls_tokens_aa :],
                            self.norm_layer(x[:, :, self.n_cls_tokens_aa :]),
                        ],
                        dim=-1,
                    ).reshape(B, V, h, w, -1)
                )

        return output_img_features, output_cls_features
