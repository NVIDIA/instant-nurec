# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

import torch
import torch.nn as nn

from einops import rearrange


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class FeedForwardMLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.drop2 = nn.Dropout(dropout)

    def zero_init(self):
        """Initialize so that output is zero regardless of input"""
        self.fc2.weight.data.zero_()
        if self.fc2.bias is not None:
            self.fc2.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: (..., dim) tensor
        Output:
            (..., dim) tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FeedForwardSwiGLU(nn.Module):
    """FeedForward MLP with SwiGLU activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Since this class is typically used exchangeably with FeedForwardMLP, we keep the same arg name
        # with "hidden_dim". However in order to match the FLOPS & number of parameters, we reduce it by 1/3.
        # Further make it a multiple of 8 for efficient fusion.
        hidden_dim = (int(hidden_dim * 2 / 3) + 7) // 8 * 8

        self.w12 = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: (..., dim) tensor
        Output:
            (..., dim) tensor
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(x1) * x2
        return self.w3(hidden)


class LayerNorm2d(nn.Module):
    """
    Perform Layer Norm on 2D input tensor.
    """

    def __init__(self, n_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, C, H, W) tensor
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


