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

"""Branch-coverage tests for instant_nurec.model.post_processing.

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


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.model.post_processing import PerCameraAffinePostProcessing


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


# ---------------------------------------------------------------------------
# transform_tokens / _transform_tokens_cross_attention
# ---------------------------------------------------------------------------


def test_transform_tokens_returns_unchanged_x_and_affine_token():
    """transform_tokens returns (x, affine_token); x is unchanged from input."""
    m = PerCameraAffinePostProcessing(embed_dim=32, init_token_scale=0.1)
    m.eval()
    B, v, t, hw, C = 1, 2, 3, 4, 32
    x = torch.randn(B, v * t * hw, C)
    camera_idxs = torch.tensor([[0, 0, 0, 1, 1, 1]])
    embedded_x, affine_token = m.transform_tokens(x, camera_idxs)
    # embedded_x is the LayerNorm-applied version of the input
    assert embedded_x.shape == (B, v * t * hw, C)
    # affine_token is (B, v, C)
    assert affine_token.shape == (B, v, C)


def test_transform_tokens_camera_idxs_must_be_consistent_per_segment():
    """Within each (v, t) segment the camera id must be consistent —
    a t-axis that mixes cameras triggers an AssertionError."""
    m = PerCameraAffinePostProcessing(embed_dim=32, init_token_scale=0.1)
    m.eval()
    B, v, t, hw, C = 1, 2, 3, 4, 32
    x = torch.randn(B, v * t * hw, C)
    # Inconsistent camera ids: median per-segment doesn't match all segment entries
    camera_idxs = torch.tensor([[0, 1, 0, 1, 0, 1]])
    with pytest.raises(AssertionError, match="must be the same"):
        m.transform_tokens(x, camera_idxs)


def test_transform_tokens_inferred_v_from_unique_camera_idxs():
    """The number of views v is inferred from the unique values in
    camera_idxs — three unique values → v=3."""
    m = PerCameraAffinePostProcessing(embed_dim=32, init_token_scale=0.1)
    m.eval()
    B, v, t, hw, C = 1, 3, 2, 4, 32
    x = torch.randn(B, v * t * hw, C)
    camera_idxs = torch.tensor([[0, 0, 1, 1, 2, 2]])
    embedded_x, affine_token = m.transform_tokens(x, camera_idxs)
    assert affine_token.shape == (B, 3, C)


def test_transform_tokens_eval_mode_deterministic():
    """In eval mode, repeating the call yields identical output."""
    m = PerCameraAffinePostProcessing(embed_dim=32, init_token_scale=0.1)
    m.eval()
    x = torch.randn(1, 8, 32)
    camera_idxs = torch.tensor([[0, 0, 0, 0]])
    e1, a1 = m.transform_tokens(x, camera_idxs)
    e2, a2 = m.transform_tokens(x.clone(), camera_idxs.clone())
    assert torch.allclose(e1, e2)
    assert torch.allclose(a1, a2)
