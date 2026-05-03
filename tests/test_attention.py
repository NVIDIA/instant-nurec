"""Branch-coverage tests for nre.nrm.models.blocks.attention.

Pure torch.nn / functional / einops — no compiled-lib stubs needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.nrm.models.blocks.attention import (
    AttentionBlock,
    CrossAttention,
    CrossAttentionBlock,
    CrossAttentionWithKVProjector,
    KVProjector,
    ModulatedAttentionBlock,
    SelfAttention,
    _maybe_layer_scale,
)
from instant_nurec.nrm.models.blocks.embeds import RotaryPositionEmbed2D
from instant_nurec.nrm.models.blocks.layers import LayerScale


# ---------------------------------------------------------------------------
# _maybe_layer_scale
# ---------------------------------------------------------------------------


def test_maybe_layer_scale_none_returns_identity():
    out = _maybe_layer_scale(dim=8, init_values=None)
    assert isinstance(out, torch.nn.Identity)


def test_maybe_layer_scale_value_returns_layer_scale():
    out = _maybe_layer_scale(dim=8, init_values=0.1)
    assert isinstance(out, LayerScale)


# ---------------------------------------------------------------------------
# SelfAttention
# ---------------------------------------------------------------------------


def test_self_attention_dim_must_be_divisible_by_n_heads():
    with pytest.raises(AssertionError, match="must be divisible"):
        SelfAttention(dim=10, n_heads=3)


def test_self_attention_qk_norm_false_uses_identity():
    sa = SelfAttention(dim=16, n_heads=4, qk_norm=False)
    assert isinstance(sa.q_norm, torch.nn.Identity)
    assert isinstance(sa.k_norm, torch.nn.Identity)


def test_self_attention_qk_norm_true_uses_layernorm():
    sa = SelfAttention(dim=16, n_heads=4, qk_norm=True)
    assert isinstance(sa.q_norm, torch.nn.LayerNorm)
    assert isinstance(sa.k_norm, torch.nn.LayerNorm)


def test_self_attention_forward_shape_no_rope():
    sa = SelfAttention(dim=16, n_heads=4, rope=None)
    x = torch.randn(2, 5, 16)
    out = sa(x)
    assert out.shape == (2, 5, 16)


def test_self_attention_forward_with_rope_requires_positions():
    rope = RotaryPositionEmbed2D()
    sa = SelfAttention(dim=16, n_heads=4, rope=rope)
    x = torch.randn(1, 4, 16)
    with pytest.raises(AssertionError, match="Rope positions"):
        sa(x, rope_positions=None)


def test_self_attention_forward_with_rope_and_positions():
    rope = RotaryPositionEmbed2D()
    sa = SelfAttention(dim=16, n_heads=4, rope=rope)
    x = torch.randn(1, 4, 16)
    positions = torch.zeros(1, 4, 2, dtype=torch.long)
    out = sa(x, rope_positions=positions)
    assert out.shape == (1, 4, 16)


def test_self_attention_dim_mismatch_assert():
    sa = SelfAttention(dim=16, n_heads=4)
    x = torch.randn(1, 4, 8)  # last dim 8, not 16
    with pytest.raises(AssertionError, match="incorrect dimension"):
        sa(x)


# ---------------------------------------------------------------------------
# KVProjector
# ---------------------------------------------------------------------------


def test_kv_projector_dim_must_be_divisible_by_n_heads():
    with pytest.raises(AssertionError, match="must be divisible"):
        KVProjector(dim=10, n_heads=3)


def test_kv_projector_k_norm_false_uses_identity():
    p = KVProjector(dim=16, n_heads=4, k_norm=False)
    assert isinstance(p.k_norm, torch.nn.Identity)


def test_kv_projector_k_norm_true_uses_layernorm():
    p = KVProjector(dim=16, n_heads=4, k_norm=True)
    assert isinstance(p.k_norm, torch.nn.LayerNorm)


def test_kv_projector_forward_shape():
    p = KVProjector(dim=16, n_heads=4)
    k = torch.randn(2, 5, 16)
    v = torch.randn(2, 5, 16)
    out_k, out_v = p(k, v)
    assert out_k.shape == (2, 4, 5, 4)  # B, H, N, head_dim
    assert out_v.shape == (2, 4, 5, 4)


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------


def test_cross_attention_kv_projector_n_heads_mismatch():
    p = KVProjector(dim=16, n_heads=2)
    with pytest.raises(AssertionError, match="same number of heads"):
        CrossAttention(dim=16, n_heads=4, kv_projector=p)


def test_cross_attention_kv_projector_dim_mismatch():
    p = KVProjector(dim=8, n_heads=4)
    with pytest.raises(AssertionError, match="same dimension"):
        CrossAttention(dim=16, n_heads=4, kv_projector=p)


def test_cross_attention_kv_projector_qk_norm_mismatch():
    """kv_projector with k_norm=False but cross-attn with qk_norm=True must fail."""
    p = KVProjector(dim=16, n_heads=4, k_norm=False)
    with pytest.raises(AssertionError, match="same QK normalization"):
        CrossAttention(dim=16, n_heads=4, qk_norm=True, kv_projector=p)


def test_cross_attention_dim_must_be_divisible_by_n_heads():
    with pytest.raises(AssertionError, match="must be divisible"):
        CrossAttention(dim=10, n_heads=3)


def test_cross_attention_forward_with_kv_projector():
    p = KVProjector(dim=16, n_heads=4)
    ca = CrossAttention(dim=16, n_heads=4, kv_projector=p)
    q = torch.randn(2, 5, 16)
    k = torch.randn(2, 7, 16)  # not pre-projected
    v = torch.randn(2, 7, 16)
    out = ca(q, k, v)
    assert out.shape == (2, 5, 16)


def test_cross_attention_forward_without_kv_projector():
    """Without internal projection, kv must be pre-projected to (B, H, M, head_dim)."""
    ca = CrossAttention(dim=16, n_heads=4, kv_projector=None)
    q = torch.randn(1, 5, 16)
    k = torch.randn(1, 4, 7, 4)  # B, H, M, head_dim
    v = torch.randn(1, 4, 7, 4)
    out = ca(q, k, v)
    assert out.shape == (1, 5, 16)


def test_cross_attention_project_q_rejects_wrong_k_ndim():
    ca = CrossAttention(dim=16, n_heads=4, kv_projector=None)
    q = torch.randn(1, 5, 16)
    k_bad = torch.randn(1, 7, 16)  # 3-dim, not 4-dim
    v_bad = torch.randn(1, 4, 7, 4)
    with pytest.raises(AssertionError, match="k must have 4 dimensions"):
        ca(q, k_bad, v_bad)


def test_cross_attention_project_q_rejects_kv_shape_mismatch():
    ca = CrossAttention(dim=16, n_heads=4, kv_projector=None)
    q = torch.randn(1, 5, 16)
    k = torch.randn(1, 4, 7, 4)
    v = torch.randn(1, 4, 8, 4)  # different M
    with pytest.raises(AssertionError, match="k and v must have the same shape"):
        ca(q, k, v)


# ---------------------------------------------------------------------------
# CrossAttentionWithKVProjector
# ---------------------------------------------------------------------------


def test_cross_attention_with_kv_projector_constructs_kvprojector_internally():
    ca = CrossAttentionWithKVProjector(dim=16, n_heads=4)
    assert isinstance(ca.kv_projector, KVProjector)
    assert ca.kv_projector.dim == 16
    assert ca.kv_projector.n_heads == 4


def test_cross_attention_with_kv_projector_legacy_state_dict_hook_renames_keys():
    """allow_legacy_state_dict=True installs a pre-load hook that prefixes
    the legacy `to_k.` / `to_v.` keys with `kv_projector.`."""
    ca = CrossAttentionWithKVProjector(dim=16, n_heads=4, allow_legacy_state_dict=True)
    legacy_state = {
        "to_k.weight": torch.zeros(16, 16),
        "to_v.weight": torch.zeros(16, 16),
        "k_norm.weight": torch.zeros(16),
        "v_norm.weight": torch.zeros(16),
    }
    # Manually invoke the registered hook with empty prefix (matches normal load_state_dict behavior).
    CrossAttentionWithKVProjector._pre_load_state_dict_hook(ca, legacy_state, prefix="")
    assert "kv_projector.to_k.weight" in legacy_state
    assert "kv_projector.to_v.weight" in legacy_state
    assert "kv_projector.k_norm.weight" in legacy_state
    assert "to_k.weight" not in legacy_state  # rename, not copy
    assert "to_v.weight" not in legacy_state


def test_cross_attention_with_kv_projector_no_legacy_hook_by_default():
    ca = CrossAttentionWithKVProjector(dim=16, n_heads=4, allow_legacy_state_dict=False)
    # No public way to introspect hooks, but at least construction shouldn't fail.
    assert ca.kv_projector is not None


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------


def test_attention_block_forward_shape_no_rope():
    block = AttentionBlock(input_dim=16, n_heads=4)
    x = torch.randn(2, 5, 16)
    out = block(x)
    assert out.shape == (2, 5, 16)


def test_attention_block_layer_scale_none_uses_identity():
    block = AttentionBlock(input_dim=16, n_heads=4, layer_scale_init_values=None)
    assert isinstance(block.ls1, torch.nn.Identity)
    assert isinstance(block.ls2, torch.nn.Identity)


def test_attention_block_layer_scale_default_uses_layer_scale():
    block = AttentionBlock(input_dim=16, n_heads=4)  # default 1e-5
    assert isinstance(block.ls1, LayerScale)
    assert isinstance(block.ls2, LayerScale)


def test_attention_block_with_rope():
    rope = RotaryPositionEmbed2D()
    block = AttentionBlock(input_dim=16, n_heads=4, rope=rope)
    x = torch.randn(1, 4, 16)
    pos = torch.zeros(1, 4, 2, dtype=torch.long)
    out = block(x, rope_positions=pos)
    assert out.shape == (1, 4, 16)


# ---------------------------------------------------------------------------
# ModulatedAttentionBlock
# ---------------------------------------------------------------------------


def test_modulated_attention_block_norm1_is_affine_free():
    block = ModulatedAttentionBlock(input_dim=16, n_heads=4)
    assert block.norm1.elementwise_affine is False


def test_modulated_attention_block_default_cond_dim_matches_input_dim():
    block = ModulatedAttentionBlock(input_dim=16, n_heads=4)
    assert block.modulation.in_features == 16
    assert block.modulation.out_features == 3 * 16


def test_modulated_attention_block_custom_cond_dim():
    block = ModulatedAttentionBlock(input_dim=16, n_heads=4, modulation_cond_dim=32)
    assert block.modulation.in_features == 32
    assert block.modulation.out_features == 3 * 16


def test_modulated_attention_block_forward_shape():
    block = ModulatedAttentionBlock(input_dim=16, n_heads=4)
    x = torch.randn(2, 8, 16)
    cond = torch.randn(2, 4, 16)  # 4 segments → segment_size=2
    out = block(x, cond)
    assert out.shape == (2, 8, 16)


def test_modulated_attention_block_rejects_indivisible_n_segments():
    block = ModulatedAttentionBlock(input_dim=16, n_heads=4)
    x = torch.randn(1, 5, 16)  # N=5
    cond = torch.randn(1, 2, 16)  # 5 % 2 != 0
    with pytest.raises(AssertionError, match="divisible by n_segments"):
        block(x, cond)


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------


def test_cross_attention_block_forward_shape_with_kv_projector():
    p = KVProjector(dim=16, n_heads=4)
    block = CrossAttentionBlock(dim=16, n_heads=4, kv_projector=p)
    q = torch.randn(2, 5, 16)
    k = torch.randn(2, 7, 16)
    v = torch.randn(2, 7, 16)
    out = block(q, k, v)
    assert out.shape == (2, 5, 16)


def test_cross_attention_block_forward_shape_without_kv_projector():
    block = CrossAttentionBlock(dim=16, n_heads=4, kv_projector=None)
    q = torch.randn(1, 5, 16)
    k = torch.randn(1, 4, 7, 4)
    v = torch.randn(1, 4, 7, 4)
    out = block(q, k, v)
    assert out.shape == (1, 5, 16)


def test_cross_attention_block_layer_scale_none_uses_identity():
    block = CrossAttentionBlock(dim=16, n_heads=4, layer_scale_init_values=None)
    assert isinstance(block.ls_ca, torch.nn.Identity)
    assert isinstance(block.ls_sa, torch.nn.Identity)
    assert isinstance(block.ls_mlp, torch.nn.Identity)
