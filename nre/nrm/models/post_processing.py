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

import torch
import torch.nn as nn

from einops import rearrange, repeat

from nre.nrm.models.blocks.attention import CrossAttention, CrossAttentionWithKVProjector
from nre.nrm.models.blocks.embeds import PositionalEmbed
from nre.utils.misc import stop_gradient


logger = logging.getLogger(__name__)


class PerCameraAffinePostProcessing(nn.Module):
    """
    This post processing module is a special case of BilateralGrid with configuration:
        - num_grids = number of cameras
        - width = height = depth (sampled via luminance) = 1
    There are two main methods within this module:
        - transform_tokens: If cross_attend is True, then this will perform cross attention between affine tokens and x.
          Otherwise, it will emebd x with view indices embedding.
        - decode_affine: This will decode the affine tokens into a 3x3 matrix and a 3x1 bias vector.
    """

    affine_attention: CrossAttention | None

    def __init__(
        self,
        embed_dim: int,
        init_token_scale: float,
        cross_attend: bool = True,
        n_affine_tokens: int = -1,
        kv_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_token_scale = init_token_scale
        self.n_affine_tokens = n_affine_tokens
        self.cross_attend = cross_attend

        # When ``kv_norm`` is False we use ``nn.Identity`` so that no
        # ``kv_norm.weight`` / ``kv_norm.bias`` entries appear in the state
        # dict; this preserves legacy checkpoint compatibility for call sites
        # (e.g. Celsius) that never trained with this LayerNorm.
        self.kv_norm = nn.LayerNorm(self.embed_dim) if kv_norm else nn.Identity()
        self.affine_linear = nn.Linear(self.embed_dim, 3 * 4)
        if self.cross_attend:
            self.affine_attention = CrossAttentionWithKVProjector(
                dim=self.embed_dim,
                n_heads=16,
                bias=True,
                norm=False,
                allow_legacy_state_dict=True,
            )
            self.affine_token = nn.Parameter(torch.randn(self.embed_dim) * self.init_token_scale)
        else:
            # Only used in older celsius models.
            self.affine_attention = None
            self.affine_token = nn.Parameter(torch.randn(self.n_affine_tokens, self.embed_dim) * self.init_token_scale)

        self._detach_linear_grad: bool = False

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
        if self.n_affine_tokens != -1 and v != self.n_affine_tokens:
            # No need to ensure camera assignments match training time, since cross attention is perm-equivariant.
            # Just need to ensure *number of cameras* matches.
            logger.warning(
                f"Module is configured to see {self.n_affine_tokens} views, but got {v} views. Most likely this is not an issue."
            )

        kv = rearrange(x, "B (v thw) C -> (B v) (thw) C", v=v)
        affine_token = repeat(self.affine_token, "C -> (B v) 1 C", B=B, v=v)
        assert self.affine_attention is not None
        affine_token = self.affine_attention(affine_token, kv, kv).squeeze(1)
        affine_token = rearrange(affine_token, "(B v) C -> B v C", B=B, v=v)
        # original x is kept unchanged
        return x, affine_token

    def _transform_tokens_view_embedding(
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
        # Since view indices are usually small in value, we use a small T value here to adapt.
        idx_emb = PositionalEmbed.get_1d_sincos_pos_embed(camera_idxs.float(), self.embed_dim, T=32.0)  # (B, v * t, C)
        x = rearrange(x, "B (vt hw) C -> B vt hw C", vt=idx_emb.shape[1]) + idx_emb[:, :, None]
        x = rearrange(x, "B vt hw C -> B (vt hw) C")  # reshape to original
        # Also embed affine tokens themselves
        affine_emb = PositionalEmbed.get_1d_sincos_pos_embed(
            torch.arange(self.n_affine_tokens, device=x.device).float(), self.embed_dim, T=32.0
        )
        affine_token = self.affine_token + affine_emb
        return x, repeat(affine_token, "a C -> B a C", B=x.shape[0], a=self.n_affine_tokens)

    def transform_tokens(self, x: torch.Tensor, camera_idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            x: (B, v * t * hw, C)
            camera_idxs: (B, v * t)
        Output:
            embedded_x: (B, v * t * hw, C)
            affine_token: (B, n_affine_tokens, C)
        """
        x = self.kv_norm(x)
        if self.cross_attend:
            return self._transform_tokens_cross_attention(x, camera_idxs)
        else:
            return self._transform_tokens_view_embedding(x, camera_idxs)

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
        if self._detach_linear_grad:
            affine = stop_gradient(affine)
        affine_matrix, affine_bias = affine.split([3 * 3, 3], dim=-1)
        affine_matrix = (
            rearrange(affine_matrix, "B n (a b) -> B n a b", a=3, b=3)
            + torch.eye(3, device=x.device, dtype=x.dtype)[None, None]
        )
        return affine_matrix, affine_bias

    def zero_init(self):
        self.affine_linear.weight.data.zero_()
        self.affine_linear.bias.data.zero_()

    def set_detach_linear_grad(self, detach: bool):
        self._detach_linear_grad = detach

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please use transform_tokens or decode_affine instead.")
