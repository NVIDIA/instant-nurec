"""Branch-coverage tests for nre.nrm.models.blocks.embeds.

Pure torch.nn / torch.nn.functional / einops — no compiled-lib stubs needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.nre.nrm.models.blocks.embeds import (
    ContinuousTimeEmbed,
    NormalizedPositionalEmbed,
    PatchEmbed,
    PositionalEmbed,
    RotaryPositionEmbed2D,
)


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------


def test_patch_embed_default_norm_is_layernorm():
    pe = PatchEmbed(patch_shape=(2, 2), input_dim=3, embed_dim=8)
    assert isinstance(pe.norm, torch.nn.LayerNorm)


def test_patch_embed_norm_false_uses_identity():
    pe = PatchEmbed(patch_shape=(2, 2), input_dim=3, embed_dim=8, norm=False)
    assert isinstance(pe.norm, torch.nn.Identity)


def test_patch_embed_forward_shape():
    pe = PatchEmbed(patch_shape=(2, 2), input_dim=3, embed_dim=8)
    x = torch.randn(2, 3, 8, 8)
    out = pe(x)
    # 8 / 2 = 4 patches per side
    assert out.shape == (2, 4, 4, 8)


def test_patch_embed_rejects_height_not_multiple_of_patch():
    pe = PatchEmbed(patch_shape=(2, 2), input_dim=3, embed_dim=8)
    x = torch.randn(1, 3, 7, 8)  # 7 not divisible by 2
    with pytest.raises(AssertionError, match="height"):
        pe(x)


def test_patch_embed_rejects_width_not_multiple_of_patch():
    pe = PatchEmbed(patch_shape=(2, 2), input_dim=3, embed_dim=8)
    x = torch.randn(1, 3, 8, 7)  # 7 not divisible by 2
    with pytest.raises(AssertionError, match="width"):
        pe(x)


# ---------------------------------------------------------------------------
# PositionalEmbed (static methods)
# ---------------------------------------------------------------------------


def test_get_1d_sincos_pos_embed_shape():
    x = torch.linspace(0, 1, steps=4)
    out = PositionalEmbed.get_1d_sincos_pos_embed(x, embed_dim=8, T=10000.0)
    assert out.shape == (4, 8)


def test_get_1d_sincos_pos_embed_first_half_sin_second_half_cos():
    """At x=0: sin(0)=0, cos(0)=1 — embed should be [0]*half + [1]*half."""
    x = torch.zeros(1)
    out = PositionalEmbed.get_1d_sincos_pos_embed(x, embed_dim=4, T=10000.0)
    half = 2
    assert torch.allclose(out[0, :half], torch.zeros(half))
    assert torch.allclose(out[0, half:], torch.ones(half))


def test_get_1d_sincos_pos_embed_rejects_odd_dim():
    x = torch.zeros(1)
    with pytest.raises(AssertionError, match="even"):
        PositionalEmbed.get_1d_sincos_pos_embed(x, embed_dim=3, T=10000.0)


def test_get_2d_sincos_grid_embed_shape():
    w = torch.linspace(0, 1, steps=4)
    h = torch.linspace(0, 1, steps=3)
    out = PositionalEmbed.get_2d_sincos_grid_embed(w, h, embed_dim=8, T=10000.0)
    assert out.shape == (3, 4, 8)


def test_get_1d_sincos_nerf_embed_uses_pi_scaling():
    """nerf_embed multiplies x by pi before delegating to get_1d_sincos_pos_embed."""
    x = torch.zeros(1)
    out = PositionalEmbed.get_1d_sincos_nerf_embed(x, embed_dim=4)
    # at x=0 → x*pi=0 → sin=0, cos=1
    assert torch.allclose(out[0, :2], torch.zeros(2))
    assert torch.allclose(out[0, 2:], torch.ones(2))


def test_positional_embed_base_forward_raises():
    pe = PositionalEmbed()
    with pytest.raises(NotImplementedError, match="child-classes"):
        pe.forward(torch.zeros(1))


# ---------------------------------------------------------------------------
# NormalizedPositionalEmbed
# ---------------------------------------------------------------------------


def test_normalized_positional_embed_get_normalized_uv_square():
    """Square aspect ratio → equal x/y span."""
    x_coords, y_coords = NormalizedPositionalEmbed.get_normalized_uv(
        w=4, h=4, device=torch.device("cpu"), dtype=torch.float32
    )
    assert x_coords.shape == (4,)
    assert y_coords.shape == (4,)
    # symmetry around 0
    assert torch.allclose(x_coords, -x_coords.flip(0))
    assert torch.allclose(y_coords, -y_coords.flip(0))


def test_normalized_positional_embed_get_normalized_uv_rectangular():
    """Wider-than-tall → x span > y span."""
    x_coords, y_coords = NormalizedPositionalEmbed.get_normalized_uv(
        w=8, h=2, device=torch.device("cpu"), dtype=torch.float32
    )
    assert x_coords.abs().max() > y_coords.abs().max()


def test_normalized_positional_embed_forward_shape():
    npe = NormalizedPositionalEmbed(T=10000.0)
    x = torch.randn(2, 4, 8, 16)
    out = npe(x)
    assert out.shape == (1, 4, 8, 16)  # always batch-1, broadcast


def test_normalized_positional_embed_forward_shape_only():
    npe = NormalizedPositionalEmbed(T=10000.0)
    out = npe.forward_shape_only(emb_height=3, emb_width=5, embed_dim=8)
    assert out.shape == (1, 3, 5, 8)


def test_normalized_positional_embed_forward_shape_only_default_args():
    """Default device/dtype kwargs path."""
    npe = NormalizedPositionalEmbed(T=10000.0)
    out = npe.forward_shape_only(emb_height=2, emb_width=2, embed_dim=4)
    assert out.shape == (1, 2, 2, 4)


# ---------------------------------------------------------------------------
# ContinuousTimeEmbed
# ---------------------------------------------------------------------------


def test_continuous_time_embed_timestep_embedding_even_dim():
    t = torch.tensor([0.0, 1.0])
    out = ContinuousTimeEmbed.timestep_embedding(t, dim=8, max_period=10000.0)
    assert out.shape == (2, 8)


def test_continuous_time_embed_timestep_embedding_odd_dim_zero_padded():
    """Odd `dim` triggers the zero-padding branch."""
    t = torch.tensor([0.0])
    out = ContinuousTimeEmbed.timestep_embedding(t, dim=5, max_period=10000.0)
    assert out.shape == (1, 5)
    # last column should be zero (padding)
    assert out[0, -1].item() == 0.0


def test_continuous_time_embed_forward_shape():
    cte = ContinuousTimeEmbed(patch_shape=(2, 2), embed_dim=8, frequency_embedding_dim=16)
    t = torch.randn(2, 4, 4)
    out = cte(t)
    assert out.shape == (2, 2, 2, 8)


def test_continuous_time_embed_rejects_height_not_multiple_of_patch():
    cte = ContinuousTimeEmbed(patch_shape=(2, 2), embed_dim=8, frequency_embedding_dim=16)
    t = torch.randn(1, 3, 4)
    with pytest.raises(AssertionError, match="height"):
        cte(t)


def test_continuous_time_embed_rejects_width_not_multiple_of_patch():
    cte = ContinuousTimeEmbed(patch_shape=(2, 2), embed_dim=8, frequency_embedding_dim=16)
    t = torch.randn(1, 4, 3)
    with pytest.raises(AssertionError, match="width"):
        cte(t)


def test_continuous_time_embed_zero_init_makes_output_zero():
    cte = ContinuousTimeEmbed(patch_shape=(2, 2), embed_dim=8, frequency_embedding_dim=16)
    cte.zero_init()
    t = torch.randn(2, 4, 4)
    out = cte(t)
    assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# RotaryPositionEmbed2D
# ---------------------------------------------------------------------------


def test_rotary_position_embed2d_shape_preserved():
    rope = RotaryPositionEmbed2D()
    tokens = torch.randn(1, 2, 5, 8)  # dim divisible by 4
    positions = torch.zeros(1, 5, 2, dtype=torch.long)
    out = rope(tokens, positions)
    assert out.shape == tokens.shape


def test_rotary_position_embed2d_rejects_non_divisible_4_feature_dim():
    rope = RotaryPositionEmbed2D()
    tokens = torch.randn(1, 2, 5, 6)  # 6 not divisible by 4
    positions = torch.zeros(1, 5, 2, dtype=torch.long)
    with pytest.raises(AssertionError, match="divisible by 4"):
        rope(tokens, positions)


def test_rotary_position_embed2d_rejects_wrong_positions_ndim():
    rope = RotaryPositionEmbed2D()
    tokens = torch.randn(1, 2, 5, 8)
    positions = torch.zeros(5, 2, dtype=torch.long)  # ndim=2, not 3
    with pytest.raises(AssertionError, match="\\(B, N, 2\\)"):
        rope(tokens, positions)


def test_rotary_position_embed2d_rejects_wrong_positions_last_dim():
    rope = RotaryPositionEmbed2D()
    tokens = torch.randn(1, 2, 5, 8)
    positions = torch.zeros(1, 5, 3, dtype=torch.long)  # last dim 3, not 2
    with pytest.raises(AssertionError, match="\\(B, N, 2\\)"):
        rope(tokens, positions)


def test_rotary_position_embed2d_zero_position_returns_input():
    """At position (0,0) the cos=1/sin=0 → output equals input."""
    rope = RotaryPositionEmbed2D()
    tokens = torch.randn(1, 2, 5, 8)
    positions = torch.zeros(1, 5, 2, dtype=torch.long)
    out = rope(tokens, positions)
    # cos(0)=1, sin(0)=0 → out == tokens
    assert torch.allclose(out, tokens, atol=1e-5)


def test_rotary_position_embed2d_nonzero_position_perturbs():
    rope = RotaryPositionEmbed2D()
    tokens = torch.randn(1, 1, 4, 8)
    positions_zero = torch.zeros(1, 4, 2, dtype=torch.long)
    positions_one = torch.ones(1, 4, 2, dtype=torch.long)
    out_zero = rope(tokens, positions_zero)
    out_one = rope(tokens, positions_one)
    assert not torch.allclose(out_zero, out_one)
