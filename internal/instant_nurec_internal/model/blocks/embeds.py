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

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PatchEmbed(nn.Module):
    """Split the input into patches, and use a 2D convolution layer to embed the patches."""

    def __init__(
        self,
        patch_shape: tuple[int, int],
        input_dim: int,
        embed_dim: int,
        norm: bool = True,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(self.input_dim, self.embed_dim, kernel_size=self.patch_shape, stride=self.patch_shape)
        self.norm = nn.LayerNorm(self.embed_dim) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, H, W) input image tensor
        Returns:
            (B, h, w, embed_dim) flattened image features
        """
        _, _, H, W = x.shape
        assert H % (ph := self.patch_shape[0]) == 0, f"Input image height ({H}) is not a multiple of patch size ({ph})."
        assert W % (pw := self.patch_shape[1]) == 0, f"Input image width ({W}) is not a multiple of patch size ({pw})."

        x = self.proj(x)
        x = rearrange(x, "B C h w -> B h w C")
        # Normalization applied to the last dimension (embed_dim)
        x = self.norm(x)
        return x


class PositionalEmbed(nn.Module):
    """Adds positional embeddings to the input tensor."""

    @staticmethod
    def get_2d_sincos_grid_embed(w: torch.Tensor, h: torch.Tensor, embed_dim: int, T: float) -> torch.Tensor:
        """
        Args:
            w: (width,) tensor
            h: (height,) tensor
            embed_dim: int
            T: float, controls the minimum frequency of the embeddings.
        Returns:
            (height, width, embed_dim) tensor
        """
        height, width = h.shape[0], w.shape[0]
        grid_w, grid_h = torch.meshgrid(w, h, indexing="xy")  # (H, W)
        emb_w = PositionalEmbed.get_1d_sincos_pos_embed(grid_w.float().flatten(), embed_dim // 2, T=T)  # (H*W, D//2)
        emb_h = PositionalEmbed.get_1d_sincos_pos_embed(grid_h.float().flatten(), embed_dim // 2, T=T)  # (H*W, D//2)
        return torch.cat([emb_w, emb_h], dim=-1).reshape(height, width, embed_dim)  # (H, W, D)

    @staticmethod
    def get_1d_sincos_pos_embed(x: torch.Tensor, embed_dim: int, T: float) -> torch.Tensor:
        """
        This is the same as the timestep embedding in openai model:
            Embedding(x, l) = [sin(x * T^(-l/L)), cos(x * T^(-l/L))], l=0..L-1
        Args:
            x: (...,) tensor
            embed_dim: int
            T: float, controls the minimum frequency of the embeddings.
                If x are input pixel locations, T is recommended to be 10000.0.
                If x are input view indices, T is recommended to be 32.0.
        Returns:
            (..., D) tensor
        """
        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        omega = torch.arange(embed_dim // 2, dtype=x.dtype, device=x.device) / (embed_dim // 2)
        omega = torch.exp(-math.log(T) * omega)
        x = x[..., None] * omega  # (..., D//2)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (..., D)
        return x

    @staticmethod
    def get_1d_sincos_nerf_embed(x: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """
        This is the same as the positional embedding in NeRF (x should be normalized to [-1, 1]).
            Embedding(x, l) = [sin(x * 2^l * pi), cos(x * 2^l * pi)], l=0..L-1
        Args:
            x: (..., D) tensor
            embed_dim: int
        Returns:
            (..., D) tensor
        """
        return PositionalEmbed.get_1d_sincos_pos_embed(x * math.pi, embed_dim, 2 ** (-embed_dim / 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute positional embedding for x (you have to add it back to x yourself!)
        """
        raise NotImplementedError("Please use child-classes instead.")



class NormalizedPositionalEmbed(PositionalEmbed):
    """
    Normalized positional embedding as used in MoGE DPT.
    It first normalizes the uv coordinates into 0-1 normalized space before embed them.

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.
    """

    def __init__(self, T: float):
        super().__init__()

        self.T = T

    @staticmethod
    def get_normalized_uv(
        w: int, h: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute normalized spans for X and Y
        aspect_ratio: float = w / h
        diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
        span_x = aspect_ratio / diag_factor
        span_y = 1.0 / diag_factor

        # Establish the linspace boundaries
        left_x = -span_x * (w - 1) / w
        right_x = span_x * (w - 1) / w
        top_y = -span_y * (h - 1) / h
        bottom_y = span_y * (h - 1) / h

        # Generate 1D coordinates
        x_coords = torch.linspace(left_x, right_x, steps=w, dtype=dtype, device=device)
        y_coords = torch.linspace(top_y, bottom_y, steps=h, dtype=dtype, device=device)
        return x_coords, y_coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, h, w, embed_dim) input image features
        Returns:
            (1, h, w, embed_dim) image features with positional embeddings
        """
        _, emb_height, emb_width, embed_dim = x.shape
        return self.forward_shape_only(emb_height, emb_width, embed_dim, x.device, x.dtype)

    def forward_shape_only(
        self,
        emb_height: int,
        emb_width: int,
        embed_dim: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Returns:
            (1, h, w, embed_dim) image features with positional embeddings
        """
        x_coords, y_coords = self.get_normalized_uv(emb_width, emb_height, device, dtype)
        embed = self.get_2d_sincos_grid_embed(x_coords, y_coords, embed_dim, self.T)
        return embed[None]


class ContinuousTimeEmbed(nn.Module):
    """
    Embeds scalar timesteps (preferrably within range 0-1) into vector representations.
    Reference: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """

    def __init__(
        self, patch_shape: tuple[int, int], embed_dim: int, frequency_embedding_dim: int, max_period: float = 10000.0
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, embed_dim, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim
        self.max_period = max_period

    @classmethod
    def timestep_embedding(cls, t: torch.Tensor, dim: int, max_period: float) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: (N,) 1-D Tensor of timestep, one per batch element. These may be fractional.
            dim: int, the dimension of the output.
            max_period: float, controls the minimum frequency of the embeddings.

        Returns:
            torch.Tensor: (N, D) Tensor of time embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def zero_init(self):
        """Initialize so that output is zero regardless of input"""
        if isinstance(last_linear := self.mlp[-1], nn.Linear):
            last_linear.weight.data.zero_()
            last_linear.bias.data.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, H, W) 3-D Tensor of continuous time.
        Returns:
            torch.Tensor: (B, h, w, embed_dim) Tensor of time embeddings.
        """
        B, H, W = t.shape
        assert H % (ph := self.patch_shape[0]) == 0, f"Input image height ({H}) is not a multiple of patch size ({ph})."
        assert W % (pw := self.patch_shape[1]) == 0, f"Input image width ({W}) is not a multiple of patch size ({pw})."

        # Compute mean of the patches
        t = nn.functional.avg_pool2d(t[:, None], kernel_size=self.patch_shape, stride=self.patch_shape)
        _, _, h, w = t.shape

        t_freq = self.timestep_embedding(t.view(-1), self.frequency_embedding_dim, max_period=self.max_period)
        t_emb = self.mlp(t_freq)
        return rearrange(t_emb, "(B h w) D -> B h w D", h=h, w=w, B=B)


class RotaryPositionEmbed2D(nn.Module):
    """
    2D Rotary Position Embedding implementation. This module applies rotary position
    embeddings to input tokens based on their 2D spatial positions.
    It handles the position-dependent rotation of features separately for vertical
    and horizontal dimensions.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """
        Args:
            frequency: Base frequency for the position embeddings.
            scaling_factor: Scaling factor for frequency computation.
        """
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        # Compute frequency bands and generate position-dependent frequencies
        exponents = torch.arange(0, dim, 2, device=device).float() / dim
        inv_freq = 1.0 / (self.base_frequency**exponents)
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, inv_freq)

        # Compute and cache frequency components
        angles = angles.to(dtype)
        angles = torch.cat((angles, angles), dim=-1)
        cos_components = angles.cos().to(dtype)
        sin_components = angles.sin().to(dtype)

        return cos_components, sin_components

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        # Helper function to rotate features
        def _rotate_features(x: torch.Tensor) -> torch.Tensor:
            feature_dim = x.shape[-1]
            x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        # Apply rotation
        return (tokens * cos) + (_rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (B, H, N, dim).
                   The dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (B, N, 2) containing
                      the y and x coordinates for each token.

        Returns:
            (B, H, N, dim) tensor, the tokens with applied 2D rotary position embeddings.
        """
        assert tokens.size(-1) % 4 == 0, "Feature dimension must be divisible by 4."
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (B, N, 2)."

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)
