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
import torch.nn as nn

from einops import rearrange, repeat

from instant_nurec.model.blocks.attention import CrossAttention, CrossAttentionWithKVProjector


class PerCameraAffinePostProcessing(nn.Module):
    """
    This post processing module is a special case of BilateralGrid with configuration:
        - num_grids = number of cameras
        - width = height = depth (sampled via luminance) = 1
    There are two main methods within this module:
        - transform_tokens: cross attention between affine tokens and x.
        - decode_affine: This will decode the affine tokens into a 3x3 matrix and a 3x1 bias vector.
    """

    affine_attention: CrossAttention

    def __init__(
        self,
        embed_dim: int,
        init_token_scale: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_token_scale = init_token_scale

        self.kv_norm = nn.LayerNorm(self.embed_dim)
        self.affine_linear = nn.Linear(self.embed_dim, 3 * 4)
        self.affine_attention = CrossAttentionWithKVProjector(
            dim=self.embed_dim,
            n_heads=16,
            bias=True,
            norm=False,
            allow_legacy_state_dict=True,
        )
        self.affine_token = nn.Parameter(torch.randn(self.embed_dim) * self.init_token_scale)

    def _transform_tokens_cross_attention(
        self, x: torch.Tensor, camera_idxs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            x: (B, v * t * hw, C)
            camera_idxs: (B, v * t)
        Output:
            embedded_x: (B, v * t * hw, C)
            affine_token: (B, n_affine_tokens, C)
        """
        # Permute affine tokens to apply cross attention
        # Cross attend tokens with input embeddings to inform affine_tokens with image info.
        inferred_v: int = torch.unique(camera_idxs).shape[0]
        original_camera_idxs = camera_idxs = rearrange(camera_idxs, "B (v t) -> B v t", v=inferred_v)
        camera_idxs = camera_idxs.median(2).values  # (B, v)
        assert torch.all(camera_idxs[..., None] == original_camera_idxs), (
            f"Camera idxs must be the same for each divided view. Got {original_camera_idxs}."
        )

        B, v = camera_idxs.shape
        kv = rearrange(x, "B (v thw) C -> (B v) (thw) C", v=v)
        affine_token = repeat(self.affine_token, "C -> (B v) 1 C", B=B, v=v)
        affine_token = self.affine_attention(affine_token, kv, kv).squeeze(1)
        affine_token = rearrange(affine_token, "(B v) C -> B v C", B=B, v=v)
        # original x is kept unchanged
        return x, affine_token

    def transform_tokens(self, x: torch.Tensor, camera_idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            x: (B, v * t * hw, C)
            camera_idxs: (B, v * t)
        Output:
            embedded_x: (B, v * t * hw, C)
            affine_token: (B, n_affine_tokens, C)
        """
        return self._transform_tokens_cross_attention(self.kv_norm(x), camera_idxs)

    @torch.autocast("cuda", enabled=False)
    def decode_affine(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            x: (B, n_affine_tokens, C)
        Output:
            affine_matrix: (B, n_affine_tokens, 3, 3)
            affine_bias: (B, n_affine_tokens, 3)
        """
        affine: torch.Tensor = self.affine_linear(x.float())  # (B, n_affine_tokens, 3 * 4)
        affine_matrix, affine_bias = affine.split([3 * 3, 3], dim=-1)
        affine_matrix = (
            rearrange(affine_matrix, "B n (a b) -> B n a b", a=3, b=3)
            + torch.eye(3, device=x.device, dtype=x.dtype)[None, None]
        )
        return affine_matrix, affine_bias

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please use transform_tokens or decode_affine instead.")
