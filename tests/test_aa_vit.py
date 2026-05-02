"""Branch-coverage tests for nre.nrm.models.blocks.aa_vit.

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

from nre.nrm.models.blocks.aa_vit import AlternateAttentionVisionTransformer
from nre.nrm.models.blocks.attention import (
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
    g, l = m.get_rope_positions(height=4, width=4, device=torch.device("cpu"))
    # Both have (n_cls_tokens_aa + h*w, 2) shape
    assert g.shape == (2 + 16, 2)
    assert l.shape == (2 + 16, 2)


def test_get_rope_positions_global_and_local_differ():
    """The global path zeros the spatial part; the local path is a cartesian
    product. They must differ for h*w >= 1 with h*w > 1 (any non-zero spatial coord)."""
    m = _make_aa_vit(n_cls_tokens=1)
    g, l = m.get_rope_positions(height=4, width=4, device=torch.device("cpu"))
    assert not torch.equal(g, l)


def test_get_rope_positions_n_cls_tokens_aa_branch_value():
    """The first n_cls_tokens_aa entries are the cls position arange,
    repeated across the trailing dim 2."""
    m = _make_aa_vit(n_cls_tokens=3)
    g, l = m.get_rope_positions(height=2, width=2, device=torch.device("cpu"))
    expected_cls = torch.arange(3).reshape(3, 1).expand(3, 2)
    assert torch.equal(g[:3], expected_cls)
    assert torch.equal(l[:3], expected_cls)
