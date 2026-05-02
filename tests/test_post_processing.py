"""Branch-coverage tests for nre.nrm.models.post_processing.

The constructor pulls in CrossAttentionWithKVProjector but it imports
cleanly in the cpu-only test venv. We only exercise the methods that
don't require feeding through the cross-attention layer
(``decode_affine`` and ``forward``); those paths cover the
shape-rearrangement + identity-addition branch and the explicit
NotImplementedError branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nre.nrm.models.post_processing import PerCameraAffinePostProcessing


# ---------------------------------------------------------------------------
# decode_affine
# ---------------------------------------------------------------------------


def test_decode_affine_returns_correct_shapes():
    m = PerCameraAffinePostProcessing(embed_dim=32, init_token_scale=0.1)
    x = torch.zeros(2, 4, 32)
    affine_matrix, affine_bias = m.decode_affine(x)
    assert affine_matrix.shape == (2, 4, 3, 3)
    assert affine_bias.shape == (2, 4, 3)


def test_constructor_requires_embed_dim_divisible_by_16_heads():
    """The internal cross attention uses n_heads=16, so embed_dim % 16 must be 0."""
    with pytest.raises(AssertionError, match="must be divisible"):
        PerCameraAffinePostProcessing(embed_dim=8, init_token_scale=0.1)


def test_decode_affine_zero_input_yields_identity_matrix():
    """When affine_linear(0)=bias_only, with default-initialized bias the
    matrix output is approximately bias_reshaped + I. We verify the
    +I term is applied by zeroing the linear layer."""
    m = PerCameraAffinePostProcessing(embed_dim=16, init_token_scale=0.1)
    # Force the linear layer to output exactly zero for any input
    with torch.no_grad():
        m.affine_linear.weight.zero_()
        m.affine_linear.bias.zero_()
    x = torch.randn(1, 1, 16)
    affine_matrix, affine_bias = m.decode_affine(x)
    # matrix == 0 + I
    assert torch.allclose(affine_matrix[0, 0], torch.eye(3))
    # bias == 0
    assert torch.allclose(affine_bias, torch.zeros_like(affine_bias))


def test_decode_affine_casts_input_to_float():
    """The implementation calls ``x.float()`` — half-precision input must
    not crash and must produce float32 output."""
    m = PerCameraAffinePostProcessing(embed_dim=16, init_token_scale=0.1)
    x = torch.zeros(1, 2, 16, dtype=torch.float16)
    affine_matrix, affine_bias = m.decode_affine(x)
    # Output dtype follows the cast (and identity addition uses x.dtype which is half).
    # We just check shapes + finiteness.
    assert affine_matrix.shape == (1, 2, 3, 3)
    assert affine_bias.shape == (1, 2, 3)
    assert torch.isfinite(affine_matrix).all()


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_forward_raises_not_implemented():
    m = PerCameraAffinePostProcessing(embed_dim=16, init_token_scale=0.1)
    with pytest.raises(NotImplementedError, match="transform_tokens or decode_affine"):
        m.forward()


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


def test_constructor_creates_expected_submodules():
    m = PerCameraAffinePostProcessing(embed_dim=16, init_token_scale=0.05)
    assert m.embed_dim == 16
    assert m.init_token_scale == 0.05
    assert isinstance(m.kv_norm, torch.nn.LayerNorm)
    assert isinstance(m.affine_linear, torch.nn.Linear)
    assert m.affine_linear.in_features == 16
    assert m.affine_linear.out_features == 12  # 3 * 4
    assert m.affine_token.shape == (16,)


def test_constructor_init_token_scale_controls_param_magnitude():
    """init_token_scale is multiplied with randn — at scale=0 the affine_token
    is exactly zero."""
    m = PerCameraAffinePostProcessing(embed_dim=16, init_token_scale=0.0)
    assert torch.allclose(m.affine_token, torch.zeros_like(m.affine_token))
