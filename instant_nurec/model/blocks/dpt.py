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

import math

from typing import Literal, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint

from einops import rearrange, repeat

from instant_nurec.model.blocks.embeds import NormalizedPositionalEmbed
from instant_nurec.model.blocks.layers import LayerNorm2d
from instant_nurec.utils.misc import unpack_optional


class DPTReassembleBlock(nn.Module):
    """
    Obtain features extracted from intermediate blocks of the ViT backbone, and resize
    (re-assemble as in DPT https://arxiv.org/pdf/2103.13413 terminology) them to multiple
    scales for the later fusion stage.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_blocks: int,
        hidden_dims: tuple[int, ...],
        pos_embed_strength: float | None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_blocks = n_blocks
        self.hidden_dims = hidden_dims

        assert n_blocks >= 2, "At least 2 blocks are required to re-assemble features."
        assert len(hidden_dims) == n_blocks, "Number of output dimensions must match number of blocks."

        self.norm_layer = nn.LayerNorm([input_dim])
        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(input_dim, hd, kernel_size=1, stride=1, padding=0) for hd in hidden_dims]
        )

        self.resize_layers = nn.ModuleList()
        # For the first D-2 blocks, we use conv-transpose to expand their sizes to 4x, 2x...
        for i in range(self.n_blocks - 2):
            stride = 2 ** (self.n_blocks - 2 - i)
            self.resize_layers.append(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i], kernel_size=stride, stride=stride, padding=0)
            )
        self.resize_layers.extend(
            [
                # For the 2nd-last block, the size is kept unchanged.
                nn.Identity(),
                # For the last block, we downsample to 1/2x with 3x3 conv.
                nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1),
            ]
        )

        self.output_layers = nn.ModuleList(
            [nn.Conv2d(hd, output_dim, kernel_size=3, stride=1, padding=1, bias=False) for hd in hidden_dims]
        )

        self.pos_embed_strength = pos_embed_strength
        self.pos_embed = NormalizedPositionalEmbed(T=100.0) if pos_embed_strength is not None else None

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            x_list: list of (B, h, w, input_dim) tensors, the features extracted from the intermediate blocks of the ViT backbone.
            (from early layers to deeper layers)
        Returns:
            list of (B, C', H', W') tensors, the re-assembled features at multiple scales (H' and W' are different for each scale)
            (from fine to coarse scales)
        """
        x_out: list[torch.Tensor] = []

        for x, proj_layer, resize_layer, output_layer in zip(
            x_list, self.proj_layers, self.resize_layers, self.output_layers, strict=True
        ):
            x = self.norm_layer(x)
            x = proj_layer(rearrange(x, "B h w C -> B C h w"))
            if self.pos_embed is not None and self.pos_embed_strength is not None:
                pos_embed = self.pos_embed(rearrange(x, "B C h w -> B h w C")) * self.pos_embed_strength
                x = x + repeat(pos_embed, "1 h w C -> B C h w", B=x.shape[0])
            x = resize_layer(x)
            x = output_layer(x)
            x_out.append(x)

        return x_out


class DPTFusionBlock(nn.Module):
    """
    Building block for one level of the fusion head.
    """

    def __init__(self, input_dim: int, output_dim: int, with_residual: bool):
        super().__init__()
        self.input_dim = input_dim

        self.main_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
        )

        self.res_block: nn.Sequential | None = None
        if with_residual:
            self.res_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
            )

        self.out_conv = nn.Conv2d(input_dim, output_dim, 1, 1, 0, bias=True)

    def forward(
        self, x: torch.Tensor, x_res: torch.Tensor | None = None, resize: tuple[int, int] | None = None
    ) -> torch.Tensor:
        if self.res_block is not None:
            assert x_res is not None, "Residual connection requires a residual input."
            x_res = self.res_block(x_res) + x_res
            x = x + unpack_optional(x_res)

        x = self.main_block(x) + x

        if resize is not None:
            x = nn.functional.interpolate(x, size=resize, mode="bilinear", align_corners=True)

        x = self.out_conv(x)
        return x


class DPTFusionHead(nn.Module):
    """
    Fuses resized features from the re-assemble blocks into a final map required by downstream tasks.
    One can use multiple heads on top of the features for multiple different tasks.
    """

    before_conv: nn.Module
    after_conv: nn.Sequential

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_blocks: int,
        before_conv: Literal["1-layer", "5-layers"],
        after_conv: Literal["2-layers", "2-layers-w-norm"],
        after_conv_dim: int,
        pos_embed_strength: float | None,
    ):
        super().__init__()

        self.refinement_blocks = nn.ModuleList(
            [
                DPTFusionBlock(input_dim, input_dim, with_residual=block_idx < n_blocks - 1)
                for block_idx in range(n_blocks)
            ]
        )

        # Before conv is the layer before resizing to desired final shape.
        if before_conv == "1-layer":
            self.before_conv = nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=True)
        elif before_conv == "5-layers":
            self.before_conv = nn.Sequential(
                nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(input_dim // 2, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(input_dim // 2, input_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
            )
        else:
            raise ValueError(f"Invalid before_conv: {before_conv}")

        # After conv is after resizing.
        if after_conv == "2-layers":
            self.after_conv = nn.Sequential(
                nn.Conv2d(input_dim // 2, after_conv_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(after_conv_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=True),
            )
        elif after_conv == "2-layers-w-norm":
            self.after_conv = nn.Sequential(
                nn.Conv2d(input_dim // 2, after_conv_dim, kernel_size=3, stride=1, padding=1, bias=True),
                LayerNorm2d(after_conv_dim, eps=1e-5),  # eps to be aligned with nn.LayerNorm
                nn.ReLU(inplace=True),
                nn.Conv2d(after_conv_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=True),
            )
        else:
            raise ValueError(f"Invalid after_conv: {after_conv}")

        self.pos_embed_strength = pos_embed_strength
        self.pos_embed = NormalizedPositionalEmbed(T=100.0) if pos_embed_strength is not None else None

    def zero_init(self, init_values: list[float]):
        """Initialize so that output is zero regardless of input"""
        assert isinstance(last_conv := self.after_conv[-1], nn.Conv2d)
        assert last_conv.bias is not None, "Bias should not be None for zero_init"
        weight_data = last_conv.weight.data
        bias_data = last_conv.bias.data
        assert bias_data.shape[0] == len(init_values), "Bias shape must match init_values length"

        last_conv.reset_parameters()
        for i, value in enumerate(init_values):
            if not math.isnan(value):
                bias_data[i] = value
                weight_data[i] = 0.0

    def forward(
        self,
        x_list: list[torch.Tensor],
        output_shape: tuple[int, int] | None = None,
        fusion_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_list: list of (B, C', H, W) tensors, the resized features from the re-assemble blocks (H and W are different for each scale)
            (from early/high-res layers to deeper/low-res layers)
            fusion_features: (B, C' // 2, H, W) tensor, HD features desired to be fused into the final output.
        Returns:
            (B, C, H, W) tensor, the final fused feature map.
        """

        assert len(x_list) == len(self.refinement_blocks), "Number of features and refinement blocks must match."

        # Iterative refinement reversely
        x = x_list[-1]
        for block_idx in reversed(range(len(x_list))):
            # For blocks D-1,...,1, use next block's shape, for block 0, upsample to 2x.
            if block_idx > 0:
                resize = (x_list[block_idx - 1].shape[2], x_list[block_idx - 1].shape[3])
            else:
                resize = (x.shape[2] * 2, x.shape[3] * 2)
            # Residual x_input is fed in only for block D-2,...,0.
            x = self.refinement_blocks[block_idx](
                x, x_list[block_idx] if block_idx < len(x_list) - 1 else None, resize=resize
            )

        # Before conv
        x = self.before_conv(x)

        # Resize + After conv
        if output_shape is not None:
            x = nn.functional.interpolate(x, size=output_shape, mode="bilinear", align_corners=True)

        # Fusion of HD features if available
        if fusion_features is not None:
            x = x + fusion_features

        if self.pos_embed is not None and self.pos_embed_strength is not None:
            pos_embed = self.pos_embed(rearrange(x, "B C h w -> B h w C")) * self.pos_embed_strength
            x = x + repeat(pos_embed, "1 h w C -> B C h w", B=x.shape[0])

        x = self.after_conv(x)
        return x


class DPTFullHead(nn.Module):
    """
    A full head implementation that directly takes multi-scale features as input and outputs re-scaled final feature.
    It is a simple stack of Re-assemble + Fusion head.
    """

    def __init__(
        self,
        input_dim: int,
        reassemble_hidden_dims: tuple[int, ...],
        reassemble_dim: int,
        output_dim: int,
        n_blocks: int,
        head_before_conv: Literal["1-layer", "5-layers"],
        head_after_conv: Literal["2-layers", "2-layers-w-norm"],
        head_after_conv_dim: int,
        pos_embed_strength: float | None,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.reassemble = DPTReassembleBlock(
            input_dim=input_dim,
            output_dim=reassemble_dim,
            n_blocks=n_blocks,
            hidden_dims=reassemble_hidden_dims,
            pos_embed_strength=pos_embed_strength,
        )
        self.fusion_head = DPTFusionHead(
            input_dim=reassemble_dim,
            output_dim=output_dim,
            n_blocks=n_blocks,
            before_conv=head_before_conv,
            after_conv=head_after_conv,
            after_conv_dim=head_after_conv_dim,
            pos_embed_strength=pos_embed_strength,
        )
        self.checkpointing = checkpointing

    def zero_init(self, init_values: list[float] | None = None):
        """Initialize so that output is zero regardless of input"""
        if init_values is not None:
            self.fusion_head.zero_init(init_values)

    def _raw_forward(
        self,
        x_list: list[torch.Tensor],
        output_shape: tuple[int, int] | None = None,
        fusion_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.reassemble(x_list)
        x = self.fusion_head(x, output_shape=output_shape, fusion_features=fusion_features)
        return x

    def forward(
        self,
        x_list: list[torch.Tensor],
        output_shape: tuple[int, int] | None = None,
        fusion_features: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        batch_size: int = x_list[0].shape[0]
        chunk_size = min(chunk_size, batch_size) if chunk_size is not None and chunk_size > 0 else batch_size
        n_chunks = math.ceil(batch_size / chunk_size)
        output_list: list[torch.Tensor] = []
        for i in range(n_chunks):
            x_list_chunk = [x[i * chunk_size : (i + 1) * chunk_size] for x in x_list]
            # Also chunk the fusion features if provided.
            if fusion_features is not None:
                fusion_features_chunk = fusion_features[i * chunk_size : (i + 1) * chunk_size]
            else:
                fusion_features_chunk = None
            if self.checkpointing:
                x = cast(
                    torch.Tensor,
                    torch.utils.checkpoint.checkpoint(
                        self._raw_forward, x_list_chunk, output_shape, fusion_features_chunk, use_reentrant=False
                    ),
                )
            else:
                x = self._raw_forward(x_list_chunk, output_shape, fusion_features_chunk)
            output_list.append(x)
        return torch.cat(output_list, dim=0)
