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

"""Branch-coverage tests for instant_nurec.model.blocks.layers.

Pure torch.nn — no compiled-lib stubs needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec_internal.model.blocks.layers import FeedForwardMLP, LayerNorm2d, LayerScale


# ---------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------


def test_layerscale_init_values_set_gamma_uniformly():
    ls = LayerScale(dim=4, init_values=0.5)
    assert torch.allclose(ls.gamma, torch.full((4,), 0.5))


def test_layerscale_default_init_values_close_to_zero():
    ls = LayerScale(dim=3)
    assert torch.allclose(ls.gamma, torch.full((3,), 1e-5))


def test_layerscale_forward_out_of_place_does_not_mutate_input():
    ls = LayerScale(dim=4, init_values=2.0, inplace=False)
    x = torch.ones(1, 4)
    x_before = x.clone()
    out = ls(x)
    # Input untouched
    assert torch.equal(x, x_before)
    # Output is x * gamma (= 2)
    assert torch.allclose(out, torch.full_like(x, 2.0))


def test_layerscale_forward_inplace_mutates_input():
    ls = LayerScale(dim=4, init_values=3.0, inplace=True)
    x = torch.ones(1, 4)
    out = ls(x)
    # Inplace: out aliases x and both are scaled
    assert out.data_ptr() == x.data_ptr()
    assert torch.allclose(x, torch.full_like(x, 3.0))


# ---------------------------------------------------------------------------
# FeedForwardMLP
# ---------------------------------------------------------------------------


def test_feedforward_mlp_shape_preservation():
    mlp = FeedForwardMLP(input_dim=8, hidden_dim=16, output_dim=8)
    x = torch.randn(3, 5, 8)
    out = mlp(x)
    assert out.shape == (3, 5, 8)


def test_feedforward_mlp_dimension_change():
    mlp = FeedForwardMLP(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(2, 8)
    out = mlp(x)
    assert out.shape == (2, 4)


def test_feedforward_mlp_bias_default_true():
    mlp = FeedForwardMLP(input_dim=8, hidden_dim=16, output_dim=4)
    assert mlp.fc1.bias is not None
    assert mlp.fc2.bias is not None


def test_feedforward_mlp_bias_false_disables_bias():
    mlp = FeedForwardMLP(input_dim=8, hidden_dim=16, output_dim=4, bias=False)
    assert mlp.fc1.bias is None
    assert mlp.fc2.bias is None


def test_feedforward_mlp_zero_init_with_bias():
    mlp = FeedForwardMLP(input_dim=8, hidden_dim=16, output_dim=4, bias=True)
    mlp.zero_init()
    assert torch.equal(mlp.fc2.weight, torch.zeros_like(mlp.fc2.weight))
    assert torch.equal(mlp.fc2.bias, torch.zeros_like(mlp.fc2.bias))
    # Output is zero regardless of input
    out = mlp(torch.randn(7, 8))
    assert torch.allclose(out, torch.zeros_like(out))


def test_feedforward_mlp_zero_init_without_bias():
    """zero_init must not crash when fc2.bias is None."""
    mlp = FeedForwardMLP(input_dim=8, hidden_dim=16, output_dim=4, bias=False)
    mlp.zero_init()  # should not raise
    out = mlp(torch.randn(7, 8))
    assert torch.allclose(out, torch.zeros_like(out))


def test_feedforward_mlp_eval_mode_disables_dropout():
    """Dropout(0.0) is a no-op anyway, but check that any randomness from
    nonzero dropout disappears in eval mode."""
    mlp = FeedForwardMLP(input_dim=8, hidden_dim=16, output_dim=8, dropout=0.5)
    mlp.eval()
    x = torch.randn(2, 8)
    out1 = mlp(x)
    out2 = mlp(x)
    assert torch.equal(out1, out2)  # deterministic in eval


# ---------------------------------------------------------------------------
# LayerNorm2d
# ---------------------------------------------------------------------------


def test_layernorm2d_shape_preserved():
    ln = LayerNorm2d(n_dim=3)
    x = torch.randn(2, 3, 4, 5)
    out = ln(x)
    assert out.shape == (2, 3, 4, 5)


def test_layernorm2d_default_weight_one_bias_zero():
    ln = LayerNorm2d(n_dim=4)
    assert torch.equal(ln.weight, torch.ones(4))
    assert torch.equal(ln.bias, torch.zeros(4))


def test_layernorm2d_zero_input_yields_zero_output():
    """Mean and var of all-zeros are zero — output is therefore weight*0 + bias = 0
    when bias=0."""
    ln = LayerNorm2d(n_dim=3)
    x = torch.zeros(1, 3, 4, 5)
    out = ln(x)
    # Stable to eps; should be exactly bias broadcasted (= 0).
    assert torch.allclose(out, torch.zeros_like(out))


def test_layernorm2d_normalizes_along_channel_dim():
    """For a batch where each (h, w) slice has channels [1,2,3], the result
    should have unit-ish stddev across channels (modulo eps)."""
    ln = LayerNorm2d(n_dim=3, eps=0.0)
    # Set weight=1, bias=0 (already default)
    x = torch.tensor(
        [[[[1.0]], [[2.0]], [[3.0]]]]  # (1, 3, 1, 1)
    )
    out = ln(x)
    # mean across channels of normalized output should be ~0
    assert torch.allclose(out.mean(1), torch.zeros_like(out.mean(1)), atol=1e-5)


def test_layernorm2d_weight_and_bias_applied():
    """Setting weight=2, bias=5 scales/shifts the normalized output."""
    ln = LayerNorm2d(n_dim=3)
    with torch.no_grad():
        ln.weight.fill_(2.0)
        ln.bias.fill_(5.0)
    x = torch.randn(1, 3, 2, 2)
    out = ln(x)
    # check: mean across channel ≈ 5 (because normalized has 0-mean → +bias)
    assert torch.allclose(out.mean(1), torch.full_like(out.mean(1), 5.0), atol=1e-5)


def test_layernorm2d_eps_prevents_div_by_zero():
    """With constant input, var=0; eps must keep division finite."""
    ln = LayerNorm2d(n_dim=3, eps=1e-6)
    x = torch.full((1, 3, 4, 4), 7.0)  # constant per channel and spatial
    out = ln(x)
    assert torch.isfinite(out).all()
