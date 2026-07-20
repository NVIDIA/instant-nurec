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

"""
Hierarchy of the modules in this file:
- [AttentionBlock] = [SelfAttention] + FFN/LN
  - SelfAttention = QKV-Projector + SDPA
- [CrossAttentionBlock] = [CrossAttention] + [SelfAttention] + FFN/LN
  - [CrossAttention] = [KVProjector] + Q-Projector + SDPA
"""

from typing import Optional

import torch
import torch.nn as nn

from einops import rearrange

from instant_nurec.utils.nn_extensions import module_call_type
from instant_nurec.model.blocks.embeds import RotaryPositionEmbed2D
from instant_nurec.model.blocks.layers import FeedForwardMLP, LayerScale


class SelfAttention(nn.Module):
    """
    A Self-attention module that takes tokens as input and performs SDPA operation.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        rope: RotaryPositionEmbed2D | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads

        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        head_dim = dim // n_heads

        self.dim = dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, rope_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Input:
            x: (B, N, dim) tensor
            rope_positions: (B, N, 2) tensor, the rope positions for the input tokens
        Output:
            (B, N, dim) tensor
        """
        B, N, C = x.shape
        assert C == self.dim, f"Input tensor has incorrect dimension. Expected {self.dim}, got {C}."

        # Obtain QKV and apply QK normalization
        qkv = rearrange(self.qkv(x), "B N (QKV H E) -> QKV B H N E", QKV=3, H=self.n_heads)
        q, k, v = torch.unbind(qkv, dim=0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to QK before computing attention
        if self.rope is not None:
            assert rope_positions is not None, "Rope positions must be provided"
            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        # Attention and projection
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop)
        x = rearrange(x, "B H N E -> B N (H E)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    __call__ = module_call_type(forward)


class KVProjector(nn.Module):
    """
    This module projects the keys and values to the point right before the SDPA operation.
    This is a sub-operation of both self-attention and cross-attention.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        kv_bias: bool = False,
        k_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads

        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"

        self.dim = dim
        self.head_dim = dim // n_heads

        self.to_k = nn.Linear(dim, dim, bias=kv_bias)
        self.to_v = nn.Linear(dim, dim, bias=kv_bias)
        self.k_norm = nn.LayerNorm(self.head_dim) if k_norm else nn.Identity()

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            k: (B, N, dim) tensor
            v: (B, N, dim) tensor
        Output:
            projected_k: (B, H, M, head_dim) tensor (pre-projected and rearranged)
            projected_v: (B, H, M, head_dim) tensor (pre-projected and rearranged)
            where head_dim = dim / H
        """
        k = self.to_k(k)
        k = rearrange(k, "B N (H C) -> B H N C", H=self.n_heads)
        k = self.k_norm(k)
        v = self.to_v(v)
        v = rearrange(v, "B N (H C) -> B H N C", H=self.n_heads)
        return k, v

    __call__ = module_call_type(forward)


class CrossAttention(nn.Module):
    """
    Standard cross-attention block which can optionally skip projecting the keys and values
    (assuming they already come in pre-projected) for TokenGS architecture.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        q_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        kv_projector: KVProjector | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.kv_projector = kv_projector

        # Consistency check
        if self.kv_projector is not None:
            assert self.kv_projector.n_heads == self.n_heads, (
                f"kv_projector must have the same number of heads as the cross-attention, got {self.kv_projector.n_heads} and {self.n_heads}"
            )
            assert self.kv_projector.dim == dim, (
                f"kv_projector must have the same dimension as the cross-attention, got {self.kv_projector.dim} and {dim}"
            )
            assert isinstance(self.kv_projector.k_norm, nn.Identity) != qk_norm, (
                "kv_projector must have the same QK normalization as the cross-attention"
            )

        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        head_dim = dim // n_heads

        self.dim = dim
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=q_bias)
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=q_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def _project_q(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # k and v should already be projected and rearranged, check
        head_c = self.dim // self.n_heads
        assert k.ndim == 4 and k.shape[1] == self.n_heads and k.shape[3] == head_c, (
            f"k must have 4 dimensions, [B, {self.n_heads}, M, {head_c}], got {k.shape}"
        )
        assert v.ndim == 4 and v.shape[1] == self.n_heads and v.shape[3] == head_c, (
            f"v must have 4 dimensions, [B, {self.n_heads}, M, {head_c}], got {v.shape}"
        )
        assert k.shape == v.shape, f"k and v must have the same shape, got {k.shape} and {v.shape}"

        # project q
        q = rearrange(self.to_q(q), "B N (H C) -> B H N C", H=self.n_heads)
        return self.q_norm(q)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Input:
            When kv_projector is not None:
                q: (B, N, dim) tensor
                k: (B, M, dim) tensor
                v: (B, M, dim) tensor
            When kv_projector is None:
                q: (B, N, dim) tensor
                k: (B, H, M, head_dim) tensor (pre-projected and rearranged)
                v: (B, H, M, head_dim) tensor (pre-projected and rearranged)
        Output:
            (B, N, dim) tensor
        """
        if self.kv_projector is not None:
            k, v = self.kv_projector(k, v)
        q = self._project_q(q, k, v)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop)
        x = rearrange(x, "B H N C -> B N (H C)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    __call__ = module_call_type(forward)


class CrossAttentionWithKVProjector(CrossAttention):
    """
    Cross-attention module with a built-in KVProjector.
    It also supports loading the state dict with the legacy format.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm: bool = False,
        allow_legacy_state_dict: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            q_bias=bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=norm,
            kv_projector=KVProjector(
                dim=dim,
                n_heads=n_heads,
                kv_bias=bias,
                k_norm=norm,
            ),
        )

        # Previously the CrossAttention module has kv_projector's attributes directly put in the module.
        # To support those we need to load the state dict with the legacy format.
        if allow_legacy_state_dict:
            self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    @staticmethod
    def _pre_load_state_dict_hook(
        module: "CrossAttention", state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs
    ) -> None:
        if module.kv_projector is not None:
            for key in list(state_dict.keys()):
                for attr in ["to_k.", "to_v.", "k_norm.", "v_norm."]:
                    if key.startswith(prefix + attr):
                        new_key = key.replace(prefix + attr, prefix + "kv_projector." + attr)
                        state_dict[new_key] = state_dict.pop(key)
                        break


def _maybe_layer_scale(dim: int, init_values: Optional[float]) -> LayerScale | nn.Identity:
    """Makes a LayerScale module if init_values is not None, otherwise returns an Identity."""
    return LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()


class AttentionBlock(nn.Module):
    """Standard Transformer block: one self-attention application + an MLP."""

    mlp: FeedForwardMLP

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        layer_norm_eps: float = 1e-5,
        layer_scale_init_values: Optional[float] = 1e-5,
        qk_norm: bool = False,
        rope: RotaryPositionEmbed2D | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.attn = SelfAttention(
            input_dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.ls1 = _maybe_layer_scale(input_dim, layer_scale_init_values)
        self.norm2 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.mlp = FeedForwardMLP(
            input_dim=input_dim,
            hidden_dim=int(input_dim * mlp_ratio),
            output_dim=input_dim,
            dropout=proj_drop,
        )
        self.ls2 = _maybe_layer_scale(input_dim, layer_scale_init_values)

    def forward(self, x: torch.Tensor, rope_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Input:
            x: (B, N, C) tensor
            rope_positions: (B, N, 2) tensor, the rope positions for the input tokens
        Output:
            (B, N, C) tensor
        """
        x = x + self.ls1(self.attn(self.norm1(x), rope_positions=rope_positions))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

    __call__ = module_call_type(forward)


class ModulatedAttentionBlock(AttentionBlock):
    """
    AttentionBlock with AdaLN-style conditioning on a separate cond vector (vdpm ConditionalBlock pattern):
    (1 + scale) * norm1(x) + shift, attention, then gate * attn_out. Uses affine-free norm1; FFN path unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        *args,
        modulation_cond_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, *args, **kwargs)

        # Re-create norm1 without elementwise_affine (it would otherwise cancel the effect of modulation).
        self.norm1 = nn.LayerNorm(input_dim, eps=self.norm1.eps, elementwise_affine=False)

        # Allow cond_dim to be specified (defaults to input_dim).
        cond_dim = input_dim if modulation_cond_dim is None else modulation_cond_dim
        self.modulation = nn.Linear(cond_dim, 3 * input_dim, bias=True)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Input:
            x: (B, N, C) tensor
            cond: (B, n, cond_dim), if n != 1 then N in x is supposed to be equally divided into n segments, and modulated respectively
            rope_positions: (B, N, 2) tensor, the rope positions for the input tokens
        Output:
            (B, N, C) tensor
        """
        _, N, _ = x.shape
        _, n_segments, _ = cond.shape
        assert N % n_segments == 0, f"N ({N}) must be divisible by n_segments ({n_segments})"
        segment_size = N // n_segments

        mod: torch.Tensor = self.modulation(torch.nn.functional.silu(cond))
        shift, scale, gate = mod.unsqueeze(-2).chunk(3, dim=-1)  # (B, n, 1, C) * 3

        def prepare_before_modulation(value: torch.Tensor) -> torch.Tensor:
            return rearrange(value, "B (n S) C -> B n S C", n=n_segments, S=segment_size)

        def restore_after_modulation(h: torch.Tensor) -> torch.Tensor:
            return rearrange(h, "B n S C -> B (n S) C")

        h = restore_after_modulation((1.0 + scale) * prepare_before_modulation(self.norm1(x)) + shift)
        h = self.attn(h, rope_positions=rope_positions)
        h = restore_after_modulation(gate * prepare_before_modulation(h))

        x = x + self.ls1(h)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

    __call__ = module_call_type(forward)  # type: ignore[assignment]


class CrossAttentionBlock(nn.Module):
    """
    Transformer decoder block with cross-attention, self-attention, and MLP.

    Can be configured to work with either pre-projected keys/values (kv_projection=False)
    or to project them internally (kv_projection=True).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init_values: Optional[float] = 1e-5,
        kv_projector: KVProjector | None = None,
    ):
        super().__init__()

        self.ca_norm = nn.LayerNorm(dim)
        self.ca = CrossAttention(
            dim=dim,
            n_heads=n_heads,
            q_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            kv_projector=kv_projector,
        )
        self.ls_ca = _maybe_layer_scale(dim, layer_scale_init_values)

        self.sa_norm = nn.LayerNorm(dim)
        self.sa = SelfAttention(dim, n_heads, qkv_bias, attn_drop, proj_drop, qk_norm)
        self.ls_sa = _maybe_layer_scale(dim, layer_scale_init_values)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = FeedForwardMLP(dim, int(dim * mlp_ratio), dim, bias=True, dropout=dropout)
        self.ls_mlp = _maybe_layer_scale(dim, layer_scale_init_values)

    def forward(self, q_tokens: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Input:
            When kv_projection=False:
                q_tokens: (B, N, dim) tensor
                k: (B, H, M, head_dim) tensor (pre-projected and rearranged)
                v: (B, H, M, head_dim) tensor (pre-projected and rearranged)
            When kv_projection=True:
                q_tokens: (B, N, dim) tensor
                k: (B, M, dim) tensor
                v: (B, M, dim) tensor
        Output:
            (B, N, dim) tensor
        """
        q_tokens = q_tokens + self.ls_ca(self.ca(self.ca_norm(q_tokens), k, v))
        q_tokens = q_tokens + self.ls_sa(self.sa(self.sa_norm(q_tokens)))
        q_tokens = q_tokens + self.ls_mlp(self.mlp(self.mlp_norm(q_tokens)))
        return q_tokens

    __call__ = module_call_type(forward)
