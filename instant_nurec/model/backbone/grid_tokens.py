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

"""Grid tokenization used by the selective point-query decoder.

The public inference path keeps only the tensors needed to emit and filter
static Gaussians. In particular, training supervision and motion-flow tensors
are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from einops import rearrange


@dataclass(kw_only=True, slots=True)
class GridTokens:
    """Grouped query positions and aligned per-Gaussian source values."""

    query_xyz: torch.Tensor
    gs_xyz: torch.Tensor
    point_valid_mask: torch.Tensor
    gs_ctx_rgb: torch.Tensor
    gs_sem_class: torch.Tensor
    gs_normals: torch.Tensor
    gs_source_indices: torch.Tensor


@dataclass(kw_only=True, slots=True)
class FullResInputs:
    """Full-resolution tensors consumed by :func:`build_grid_tokens`."""

    xyz: torch.Tensor
    valid_mask: torch.Tensor
    ctx_rgb: torch.Tensor
    sem_class: torch.Tensor
    normals: torch.Tensor
    source_indices: torch.Tensor


def concat_grid_tokens(tokens: list[GridTokens]) -> GridTokens:
    """Concatenate regular and selective token groups in input order."""
    if not tokens:
        raise ValueError("At least one GridTokens instance is required")
    if len(tokens) == 1:
        return tokens[0]
    return GridTokens(
        **{
            field.name: torch.cat([getattr(token, field.name) for token in tokens], dim=1)
            for field in fields(GridTokens)
        }
    )


def _flatten_token_points(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return rearrange(x, "B M K C -> B (M K) C")
    if x.ndim == 3:
        return rearrange(x, "B M K -> B (M K)")
    raise ValueError(f"Expected a grouped tensor with 3 or 4 dimensions, got {tuple(x.shape)}")


def _make_group_pipeline(
    *,
    stride: int,
    cell_size: int,
    cropped_height: int,
    cropped_width: int,
    selected_indices: torch.Tensor | None,
):
    def gather_tokens(x: torch.Tensor) -> torch.Tensor:
        assert selected_indices is not None
        index = selected_indices
        for _ in x.shape[3:]:
            index = index.unsqueeze(-1)
        return torch.gather(x, dim=2, index=index.expand(*selected_indices.shape, *x.shape[3:]))

    def run(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x[:, :, ::stride, ::stride, :]
            x = x[:, :, :cropped_height, :cropped_width, :]
            x = rearrange(
                x,
                "B V (h c1) (w c2) C -> B V (h w) (c1 c2) C",
                c1=cell_size,
                c2=cell_size,
            )
        elif x.ndim == 4:
            x = x[:, :, ::stride, ::stride]
            x = x[:, :, :cropped_height, :cropped_width]
            x = rearrange(
                x,
                "B V (h c1) (w c2) -> B V (h w) (c1 c2)",
                c1=cell_size,
                c2=cell_size,
            )
        else:
            raise ValueError(f"Expected a full-resolution tensor with 4 or 5 dimensions, got {tuple(x.shape)}")

        if selected_indices is not None:
            x = gather_tokens(x)

        if x.ndim == 5:
            return rearrange(x, "B V M K C -> B (V M) K C")
        return rearrange(x, "B V M K -> B (V M) K")

    return run


def build_grid_tokens(
    *,
    stride: int,
    cell_size: int,
    full: FullResInputs,
    keep_pixel_mask: torch.Tensor | None = None,
    keep_token_threshold: float = 0.0,
    max_tokens_per_view: int | None = None,
) -> GridTokens:
    """Downsample pixels, group them into tokens, and optionally cap tokens.

    When ``keep_pixel_mask`` is supplied, a token is eligible when at least
    ``keep_token_threshold`` of its sub-points are selected. Each view emits a
    fixed number of token slots; unused slots have a zero validity mask.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if cell_size < 1:
        raise ValueError("cell_size must be at least 1")
    if not 0.0 <= keep_token_threshold <= 1.0:
        raise ValueError("keep_token_threshold must be in [0, 1]")

    batch_size, n_views, height, width, xyz_dim = full.xyz.shape
    if xyz_dim != 3:
        raise ValueError(f"xyz must have three channels, got {tuple(full.xyz.shape)}")
    expected_prefix = (batch_size, n_views, height, width)
    for name in ("valid_mask", "ctx_rgb", "sem_class", "normals", "source_indices"):
        tensor = getattr(full, name)
        if tensor.shape[:4] != expected_prefix:
            raise ValueError(f"{name} must start with {expected_prefix}, got {tuple(tensor.shape)}")

    strided_height = (height + stride - 1) // stride
    strided_width = (width + stride - 1) // stride
    cropped_height = (strided_height // cell_size) * cell_size
    cropped_width = (strided_width // cell_size) * cell_size
    if cropped_height == 0 or cropped_width == 0:
        raise ValueError(f"Input {height}x{width} is too small for stride={stride} and cell_size={cell_size}")

    selected_indices: torch.Tensor | None = None
    token_valid: torch.Tensor | None = None
    if keep_pixel_mask is not None:
        if keep_pixel_mask.shape != expected_prefix:
            raise ValueError(f"keep_pixel_mask must have shape {expected_prefix}, got {tuple(keep_pixel_mask.shape)}")
        if max_tokens_per_view is None or max_tokens_per_view < 1:
            raise ValueError("max_tokens_per_view must be positive when keep_pixel_mask is provided")

        keep_pixel_mask = keep_pixel_mask[:, :, ::stride, ::stride]
        keep_pixel_mask = keep_pixel_mask[:, :, :cropped_height, :cropped_width]
        grouped_keep = rearrange(
            keep_pixel_mask,
            "B V (h c1) (w c2) -> B V (h w) (c1 c2)",
            c1=cell_size,
            c2=cell_size,
        ).float()
        keep_ratio = grouped_keep.mean(dim=-1)
        keep_token = keep_ratio >= keep_token_threshold
        n_tokens_per_view = keep_token.shape[-1]
        n_selected = min(max_tokens_per_view, n_tokens_per_view)
        random_scores = torch.rand_like(keep_ratio)
        random_scores = random_scores.masked_fill(~keep_token, float("-inf"))
        selected_scores, selected_indices = torch.topk(random_scores, k=n_selected, dim=-1)
        token_valid = torch.isfinite(selected_scores)

    run = _make_group_pipeline(
        stride=stride,
        cell_size=cell_size,
        cropped_height=cropped_height,
        cropped_width=cropped_width,
        selected_indices=selected_indices,
    )
    xyz = run(full.xyz)
    valid_mask = run(full.valid_mask)
    context_rgb = run(full.ctx_rgb)
    semantic_class = run(full.sem_class)
    normals = run(full.normals)
    source_indices = run(full.source_indices)

    if token_valid is not None:
        token_valid = rearrange(token_valid, "B V M -> B (V M)")
        valid_mask = valid_mask * token_valid[..., None, None].to(valid_mask.dtype)

    return GridTokens(
        query_xyz=xyz,
        gs_xyz=_flatten_token_points(xyz),
        point_valid_mask=_flatten_token_points(valid_mask),
        gs_ctx_rgb=_flatten_token_points(context_rgb),
        gs_sem_class=_flatten_token_points(semantic_class),
        gs_normals=_flatten_token_points(normals),
        gs_source_indices=_flatten_token_points(source_indices),
    )
