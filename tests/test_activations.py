"""Branch-coverage tests for nre.nrm.models.activations.

The module imports cleanly in the cpu-only test venv (only depends on
torch + the pydantic-style ``GaussiansActivationConfig`` dataclass).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nre.nrm.config.models import GaussiansActivationConfig
from nre.nrm.models.activations import (
    GaussianActivations,
    GaussianParams,
    OpacityActivation,
    RgbActivation,
    RotationActivation,
    ScaleActivation,
)


# ---------------------------------------------------------------------------
# OpacityActivation
# ---------------------------------------------------------------------------


def test_opacity_activation_zero_input_uses_shift():
    """opacity(0) = sigmoid(0 + shift). Default shift = -2.0."""
    cfg = GaussiansActivationConfig()
    op = OpacityActivation(cfg)
    out = op(torch.tensor([0.0]))
    expected = torch.sigmoid(torch.tensor([cfg.opacity_shift]))
    assert torch.allclose(out, expected)


def test_opacity_activation_custom_shift():
    cfg = GaussiansActivationConfig(opacity_shift=1.0)
    op = OpacityActivation(cfg)
    out = op(torch.tensor([0.0]))
    expected = torch.sigmoid(torch.tensor([1.0]))
    assert torch.allclose(out, expected)


def test_opacity_activation_preserves_shape():
    cfg = GaussiansActivationConfig()
    op = OpacityActivation(cfg)
    x = torch.zeros(3, 4, 5)
    assert op(x).shape == (3, 4, 5)


# ---------------------------------------------------------------------------
# ScaleActivation
# ---------------------------------------------------------------------------


def test_scale_activation_at_zero_uses_shift():
    """At x=0: scale = clamp(exp(log(scale_max) + ratio), scale_min, scale_max).
    With defaults (scale_max=0.3, ratio=-1), exp(log(0.3) - 1) = 0.3/e."""
    cfg = GaussiansActivationConfig()
    sc = ScaleActivation(cfg)
    out = sc(torch.tensor([0.0]))
    expected = torch.tensor([cfg.scale_max * math.exp(cfg.scale_shift_log_ratio)])
    assert torch.allclose(out, expected)


def test_scale_activation_clamps_above_max():
    """Large positive x → exp blows up, gets clamped to scale_max."""
    cfg = GaussiansActivationConfig()
    sc = ScaleActivation(cfg)
    out = sc(torch.tensor([100.0]))  # exp(100 + shift) >>> scale_max
    assert torch.all(out <= cfg.scale_max)
    assert out.item() == pytest.approx(cfg.scale_max)


def test_scale_activation_clamps_below_min_when_set():
    """When scale_min > 0, very-negative x is clamped from below."""
    cfg = GaussiansActivationConfig(scale_min=0.01)
    sc = ScaleActivation(cfg)
    out = sc(torch.tensor([-100.0]))
    assert out.item() == pytest.approx(0.01)


def test_scale_activation_scene_rescale_divides():
    """scene_rescale divides the post-clamp value."""
    cfg = GaussiansActivationConfig()
    sc = ScaleActivation(cfg)
    base = sc(torch.tensor([0.0]))
    scaled = sc(torch.tensor([0.0]), scene_rescale=2.0)
    assert torch.allclose(scaled, base / 2.0)


def test_scale_activation_default_scene_rescale_is_one():
    cfg = GaussiansActivationConfig()
    sc = ScaleActivation(cfg)
    out_default = sc(torch.tensor([0.5]))
    out_one = sc(torch.tensor([0.5]), scene_rescale=1.0)
    assert torch.allclose(out_default, out_one)


# ---------------------------------------------------------------------------
# RotationActivation
# ---------------------------------------------------------------------------


def test_rotation_activation_unit_quat_unchanged():
    rot = RotationActivation()
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(rot(q), q)


def test_rotation_activation_normalizes_to_unit_length():
    rot = RotationActivation()
    q = torch.tensor([2.0, 0.0, 0.0, 0.0])
    out = rot(q)
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_rotation_activation_batched():
    rot = RotationActivation()
    q = torch.randn(5, 4)
    out = rot(q)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-6)


# ---------------------------------------------------------------------------
# RgbActivation
# ---------------------------------------------------------------------------


def test_rgb_activation_at_zero_is_half():
    """sigmoid(2*0) = 0.5."""
    rgb = RgbActivation()
    out = rgb(torch.tensor([0.0]))
    assert out.item() == pytest.approx(0.5)


def test_rgb_activation_doubles_argument_under_sigmoid():
    """sigmoid(2*x), not sigmoid(x)."""
    rgb = RgbActivation()
    out = rgb(torch.tensor([1.0]))
    expected = torch.sigmoid(torch.tensor([2.0]))
    assert torch.allclose(out, expected)


def test_rgb_activation_output_in_unit_interval():
    rgb = RgbActivation()
    out = rgb(torch.tensor([-100.0, 0.0, 100.0]))
    assert torch.all((out >= 0.0) & (out <= 1.0))


# ---------------------------------------------------------------------------
# GaussianParams
# ---------------------------------------------------------------------------


def _make_valid_params(prefix=(2, 3)):
    return GaussianParams(
        rgb=torch.zeros(*prefix, 3),
        scale=torch.zeros(*prefix, 3),
        rotation=torch.zeros(*prefix, 4),
        opacity=torch.zeros(*prefix, 1),
        xyz=torch.zeros(*prefix, 3),
    )


def test_gaussian_params_post_init_accepts_consistent_shapes():
    p = _make_valid_params(prefix=(4,))
    assert p.scale.shape == (4, 3)


def test_gaussian_params_post_init_rejects_rgb_shape_mismatch():
    with pytest.raises(AssertionError, match="RGB"):
        GaussianParams(
            rgb=torch.zeros(5, 3),
            scale=torch.zeros(4, 3),
            rotation=torch.zeros(4, 4),
            opacity=torch.zeros(4, 1),
            xyz=torch.zeros(4, 3),
        )


def test_gaussian_params_post_init_rejects_rotation_shape_mismatch():
    with pytest.raises(AssertionError, match="Rotation"):
        GaussianParams(
            rgb=torch.zeros(4, 3),
            scale=torch.zeros(4, 3),
            rotation=torch.zeros(5, 4),
            opacity=torch.zeros(4, 1),
            xyz=torch.zeros(4, 3),
        )


def test_gaussian_params_post_init_rejects_opacity_shape_mismatch():
    with pytest.raises(AssertionError, match="Opacity"):
        GaussianParams(
            rgb=torch.zeros(4, 3),
            scale=torch.zeros(4, 3),
            rotation=torch.zeros(4, 4),
            opacity=torch.zeros(5, 1),
            xyz=torch.zeros(4, 3),
        )


def test_gaussian_params_post_init_rejects_xyz_shape_mismatch():
    with pytest.raises(AssertionError, match="XYZ"):
        GaussianParams(
            rgb=torch.zeros(4, 3),
            scale=torch.zeros(4, 3),
            rotation=torch.zeros(4, 4),
            opacity=torch.zeros(4, 1),
            xyz=torch.zeros(5, 3),
        )


def test_gaussian_params_getitem_with_int():
    p = _make_valid_params(prefix=(4,))
    out = p[0]
    assert out.scale.shape == (3,)
    assert out.rgb.shape == (3,)
    assert out.rotation.shape == (4,)
    assert out.opacity.shape == (1,)
    assert out.xyz.shape == (3,)


def test_gaussian_params_getitem_with_slice():
    p = _make_valid_params(prefix=(4,))
    out = p[1:3]
    assert out.scale.shape == (2, 3)


def test_gaussian_params_getitem_with_bool_mask():
    p = _make_valid_params(prefix=(4,))
    mask = torch.tensor([True, False, True, False])
    out = p[mask]
    assert out.scale.shape == (2, 3)


def test_gaussian_params_flatten_collapses_prefix():
    p = _make_valid_params(prefix=(2, 3))
    flat = p.flatten()
    assert flat.scale.shape == (6, 3)
    assert flat.rgb.shape == (6, 3)
    assert flat.rotation.shape == (6, 4)
    assert flat.opacity.shape == (6, 1)
    assert flat.xyz.shape == (6, 3)


# ---------------------------------------------------------------------------
# GaussianActivations
# ---------------------------------------------------------------------------


def test_gaussian_activations_holds_all_four_submodules():
    cfg = GaussiansActivationConfig()
    g = GaussianActivations(cfg)
    assert isinstance(g.rgb, RgbActivation)
    assert isinstance(g.scale, ScaleActivation)
    assert isinstance(g.rotation, RotationActivation)
    assert isinstance(g.opacity, OpacityActivation)


def test_gaussian_activations_propagates_config_to_scale_and_opacity():
    cfg = GaussiansActivationConfig(opacity_shift=0.7, scale_max=0.5)
    g = GaussianActivations(cfg)
    assert g.opacity.opacity_shift == 0.7
    assert g.scale.scale_max == 0.5
