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

"""Branch-coverage tests for instant_nurec.model.blocks.aa_vit.

We avoid running ``get_intermediate_features`` (the full forward pass) which
requires real backbone weights and only matters at predict-runtime — the
end-to-end parity test covers it. This suite focuses on the cheap utility
methods that are easy to validate in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.model.blocks.aa_vit import AlternateAttentionVisionTransformer
from instant_nurec.model.blocks.attention import (
    AttentionBlock,
    ModulatedAttentionBlock,
)


def _make_aa_vit(**overrides):
    base = dict(
        depth=4,
        embed_dim=32,
        n_heads=4,
        mlp_ratio=4.0,
        aa_start_block_idx=2,
        img_pos_embed_shape=4,
        n_cls_tokens=1,
        with_default_global_cls_tokens=False,
        rope_frequency=100.0,
    )
    base.update(overrides)
    return AlternateAttentionVisionTransformer(**base)


# ---------------------------------------------------------------------------
# Constructor branches
# ---------------------------------------------------------------------------


def test_aa_vit_default_global_cls_tokens_off():
    m = _make_aa_vit(with_default_global_cls_tokens=False)
    assert m.default_global_cls_tokens is None


def test_aa_vit_default_global_cls_tokens_on():
    m = _make_aa_vit(with_default_global_cls_tokens=True)
    assert m.default_global_cls_tokens is not None
    # shape: (2, n_cls_tokens_aa, embed_dim)
    assert m.default_global_cls_tokens.shape == (2, 1, 32)


def test_aa_vit_n_cls_tokens_aa_default_matches_n_cls_tokens():
    m = _make_aa_vit(n_cls_tokens=3, n_cls_tokens_aa=None)
    assert m.n_cls_tokens_aa == 3


def test_aa_vit_n_cls_tokens_aa_explicit_overrides():
    m = _make_aa_vit(n_cls_tokens=3, n_cls_tokens_aa=7)
    assert m.n_cls_tokens_aa == 7


def test_aa_vit_use_modulated_attention_false_picks_attention_block():
    m = _make_aa_vit(use_modulated_attention=False, depth=2)
    assert all(isinstance(b, AttentionBlock) and not isinstance(b, ModulatedAttentionBlock) for b in m.blocks)


def test_aa_vit_use_modulated_attention_true_picks_modulated():
    m = _make_aa_vit(use_modulated_attention=True, depth=2)
    assert all(isinstance(b, ModulatedAttentionBlock) for b in m.blocks)


def test_aa_vit_aa_start_block_idx_controls_rope_assignment():
    """Blocks with idx >= aa_start_block_idx use rope; earlier blocks do not."""
    m = _make_aa_vit(depth=4, aa_start_block_idx=2)
    # blocks[0..1] should have rope=None, blocks[2..3] should have rope set
    assert m.blocks[0].attn.rope is None
    assert m.blocks[1].attn.rope is None
    assert m.blocks[2].attn.rope is m.rope
    assert m.blocks[3].attn.rope is m.rope


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_aa_vit_forward_raises():
    m = _make_aa_vit()
    with pytest.raises(NotImplementedError, match="get_intermediate_layers"):
        m.forward(torch.zeros(1))


# ---------------------------------------------------------------------------
# get_interpolated_img_pos_embed
# ---------------------------------------------------------------------------


def test_get_interpolated_img_pos_embed_size_match_shortcut():
    """When (h, w) == (img_pos_embed_shape, img_pos_embed_shape) we should
    get back the parameter directly without interpolation."""
    m = _make_aa_vit(img_pos_embed_shape=4)
    out = m.get_interpolated_img_pos_embed(4, 4)
    assert out is m.img_pos_embed


def test_get_interpolated_img_pos_embed_interpolates_when_shape_differs():
    m = _make_aa_vit(img_pos_embed_shape=4)
    out = m.get_interpolated_img_pos_embed(8, 8)
    assert out.shape == (8, 8, 32)
    # Output is not the param itself
    assert out is not m.img_pos_embed


def test_get_interpolated_img_pos_embed_preserves_dtype():
    m = _make_aa_vit(img_pos_embed_shape=4)
    # Cast the param to float64 to verify dtype propagation.
    with torch.no_grad():
        m.img_pos_embed.data = m.img_pos_embed.data.double()
    out = m.get_interpolated_img_pos_embed(8, 8)
    assert out.dtype == torch.float64


# ---------------------------------------------------------------------------
# get_rope_positions
# ---------------------------------------------------------------------------


def test_get_rope_positions_shapes():
    m = _make_aa_vit(n_cls_tokens=2)
    g, loc = m.get_rope_positions(height=4, width=4, device=torch.device("cpu"))
    # Both have (n_cls_tokens_aa + h*w, 2) shape
    assert g.shape == (2 + 16, 2)
    assert loc.shape == (2 + 16, 2)


def test_get_rope_positions_global_and_local_differ():
    """The global path zeros the spatial part; the local path is a cartesian
    product. They must differ for h*w >= 1 with h*w > 1 (any non-zero spatial coord)."""
    m = _make_aa_vit(n_cls_tokens=1)
    g, loc = m.get_rope_positions(height=4, width=4, device=torch.device("cpu"))
    assert not torch.equal(g, loc)


def test_get_rope_positions_n_cls_tokens_aa_branch_value():
    """The first n_cls_tokens_aa entries are the cls position arange,
    repeated across the trailing dim 2."""
    m = _make_aa_vit(n_cls_tokens=3)
    g, loc = m.get_rope_positions(height=2, width=2, device=torch.device("cpu"))
    expected_cls = torch.arange(3).reshape(3, 1).expand(3, 2)
    assert torch.equal(g[:3], expected_cls)
    assert torch.equal(loc[:3], expected_cls)


# ---------------------------------------------------------------------------
# get_intermediate_features
# ---------------------------------------------------------------------------


def test_get_intermediate_features_returns_correct_shapes():
    """For block_indices=[depth-1] and global_cls_token provided, the output
    is a single feat (B, V, h, w, 2*C) and a single cls (B, V, n_cls, 2*C)."""
    m = _make_aa_vit(depth=4, embed_dim=32, n_heads=4, aa_start_block_idx=2)
    m.eval()
    img_tokens = torch.randn(1, 2, 4, 4, 32)
    global_cls_token = torch.randn(1, 2, 1, 32)
    feats, cls = m.get_intermediate_features(
        img_tokens, block_indices=[3], global_cls_token=global_cls_token
    )
    assert len(feats) == 1
    assert feats[0].shape == (1, 2, 4, 4, 64)
    assert len(cls) == 1
    assert cls[0].shape == (1, 2, 1, 64)


def test_get_intermediate_features_multiple_block_indices():
    """Asking for multiple block indices yields multiple outputs in order."""
    m = _make_aa_vit(depth=4, embed_dim=32, n_heads=4, aa_start_block_idx=2)
    m.eval()
    img_tokens = torch.randn(1, 1, 4, 4, 32)
    global_cls_token = torch.randn(1, 1, 1, 32)
    feats, cls = m.get_intermediate_features(
        img_tokens, block_indices=[1, 2, 3], global_cls_token=global_cls_token
    )
    assert len(feats) == 3
    assert len(cls) == 3


def test_get_intermediate_features_missing_last_block_raises():
    m = _make_aa_vit(depth=4)
    img_tokens = torch.randn(1, 1, 4, 4, 32)
    with pytest.raises(ValueError, match="Last block index"):
        m.get_intermediate_features(img_tokens, block_indices=[0, 1, 2])


def test_get_intermediate_features_modulated_requires_cond():
    m = _make_aa_vit(depth=4, use_modulated_attention=True)
    img_tokens = torch.randn(1, 1, 4, 4, 32)
    with pytest.raises(ValueError, match="modulation_cond"):
        m.get_intermediate_features(img_tokens, block_indices=[3])


def test_get_intermediate_features_default_global_cls_tokens_branch():
    """global_cls_token=None falls back to default_global_cls_tokens when
    with_default_global_cls_tokens=True."""
    m = _make_aa_vit(
        depth=4, embed_dim=32, aa_start_block_idx=2, with_default_global_cls_tokens=True
    )
    m.eval()
    img_tokens = torch.randn(1, 2, 4, 4, 32)
    # global_cls_token=None → uses default_global_cls_tokens
    feats, cls = m.get_intermediate_features(
        img_tokens, block_indices=[3], global_cls_token=None
    )
    assert feats[0].shape == (1, 2, 4, 4, 64)


def test_get_intermediate_features_no_default_no_global_cls_raises():
    """global_cls_token=None and with_default_global_cls_tokens=False →
    AssertionError when transitioning to global blocks."""
    m = _make_aa_vit(
        depth=4, aa_start_block_idx=2, with_default_global_cls_tokens=False
    )
    img_tokens = torch.randn(1, 2, 4, 4, 32)
    with pytest.raises(AssertionError, match="with_default_global_cls_tokens"):
        m.get_intermediate_features(img_tokens, block_indices=[3], global_cls_token=None)


def test_get_intermediate_features_aa_start_zero_path():
    """When aa_start_block_idx=0, the early-block local-attention loop body
    is never entered for any block; the transition body fires immediately
    at block_idx=0."""
    m = _make_aa_vit(
        depth=2, embed_dim=32, n_heads=4, aa_start_block_idx=0,
        with_default_global_cls_tokens=True,
    )
    m.eval()
    img_tokens = torch.randn(1, 1, 4, 4, 32)
    feats, cls = m.get_intermediate_features(img_tokens, block_indices=[1])
    assert feats[0].shape == (1, 1, 4, 4, 64)


def test_get_intermediate_features_modulated_with_cond():
    """use_modulated_attention=True + modulation_cond provided → forward
    completes."""
    m = _make_aa_vit(
        depth=4, embed_dim=32, n_heads=4, aa_start_block_idx=2,
        use_modulated_attention=True, with_default_global_cls_tokens=True,
    )
    m.eval()
    B, V = 1, 2
    img_tokens = torch.randn(B, V, 4, 4, 32)
    modulation_cond = torch.randn(B, V, 32)
    feats, cls = m.get_intermediate_features(
        img_tokens, block_indices=[3], modulation_cond=modulation_cond
    )
    assert feats[0].shape == (B, V, 4, 4, 64)


# ---------------------------------------------------------------------------
# _forward_attention_block branches (private but worth covering directly)
# ---------------------------------------------------------------------------


def test_forward_attention_block_checkpointing_all_wraps():
    """checkpointing='all' wraps via torch.utils.checkpoint.checkpoint regardless of block_type."""
    m = _make_aa_vit(depth=2, checkpointing="all")
    m.eval()
    x = torch.randn(2, 5, 32)
    block = m.blocks[0]
    out = m._forward_attention_block(
        block.forward, "local", x, rope_positions=None, modulation_cond=None
    )
    assert out.shape == x.shape


def test_forward_attention_block_checkpointing_local_only_wraps_local():
    """checkpointing='local' wraps only block_type='local' calls; 'global' is unwrapped."""
    m = _make_aa_vit(depth=2, checkpointing="local")
    m.eval()
    x = torch.randn(2, 5, 32)
    block = m.blocks[0]
    out_local = m._forward_attention_block(
        block.forward, "local", x, rope_positions=None, modulation_cond=None
    )
    out_global = m._forward_attention_block(
        block.forward, "global", x, rope_positions=None, modulation_cond=None
    )
    assert out_local.shape == x.shape
    assert out_global.shape == x.shape


def test_forward_attention_block_checkpointing_none_no_wrap():
    """checkpointing='none' uses the block directly."""
    m = _make_aa_vit(depth=2, checkpointing="none")
    m.eval()
    x = torch.randn(2, 5, 32)
    block = m.blocks[0]
    out = m._forward_attention_block(
        block.forward, "local", x, rope_positions=None, modulation_cond=None
    )
    assert out.shape == x.shape


def test_forward_attention_block_modulated_requires_cond_assertion():
    """use_modulated_attention=True + modulation_cond=None inside
    _forward_attention_block → AssertionError."""
    m = _make_aa_vit(depth=2, use_modulated_attention=True)
    x = torch.randn(2, 5, 32)
    block = m.blocks[0]
    with pytest.raises(AssertionError, match="modulation_cond is required"):
        m._forward_attention_block(
            block.forward, "local", x, rope_positions=None, modulation_cond=None
        )
