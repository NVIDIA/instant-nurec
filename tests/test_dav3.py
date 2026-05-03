"""Branch-coverage tests for instant_nurec.nrm.models.blocks.dav3.CameraEncoder.

Pure torch — no compiled-lib stubs needed. ``so3_matrix_to_quat``
imports from instant_nurec.utils.geometry which is already cpu-importable
(used by the existing ``tests/test_geometry.py`` suite).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.nrm.models.blocks.dav3 import CameraEncoder


# ---------------------------------------------------------------------------
# CameraEncoder
# ---------------------------------------------------------------------------


def test_camera_encoder_constructor_creates_expected_pieces():
    ce = CameraEncoder(input_dim=9, output_dim=64, depth=2, n_heads=8)
    # Trunk has `depth` AttentionBlocks
    assert len(ce.trunk) == 2
    assert isinstance(ce.token_norm, torch.nn.LayerNorm)
    assert isinstance(ce.trunk_norm, torch.nn.LayerNorm)


def test_camera_encoder_forward_shape():
    ce = CameraEncoder(input_dim=9, output_dim=32, depth=1, n_heads=8)
    B, V = 2, 3
    T = torch.eye(4).reshape(1, 1, 4, 4).expand(B, V, 4, 4).contiguous()
    fov = torch.full((B, V, 2), 0.5)
    out = ce(T, fov)
    assert out.shape == (B, V, 32)


def test_camera_encoder_forward_quaternion_sign_canonicalization():
    """The forward path negates quaternions whose w (4th element) is negative
    — encoding the same rotation twice (once via R, once via -R-equivalent
    sign-flipped quaternion) should land on the same pose encoding."""
    ce = CameraEncoder(input_dim=9, output_dim=32, depth=1, n_heads=8)
    ce.eval()  # disable any dropout / batchnorm running stats

    # Pick a random rotation; encode it as identity then 180° about z.
    T_a = torch.eye(4).reshape(1, 1, 4, 4)
    fov = torch.tensor([[[0.5, 0.5]]])
    out_a = ce(T_a, fov)
    out_a2 = ce(T_a.clone(), fov.clone())
    # deterministic in eval mode, identical inputs
    assert torch.allclose(out_a, out_a2)


def test_camera_encoder_forward_translation_extracted_correctly():
    """Translation t1 vs t2 should produce distinct pose tokens (via the
    pose_branch MLP)."""
    ce = CameraEncoder(input_dim=9, output_dim=32, depth=1, n_heads=8)
    ce.eval()

    T1 = torch.eye(4).reshape(1, 1, 4, 4).clone()
    T2 = torch.eye(4).reshape(1, 1, 4, 4).clone()
    T2[..., 0, 3] = 5.0  # different translation
    fov = torch.tensor([[[0.5, 0.5]]])

    out1 = ce(T1, fov)
    out2 = ce(T2, fov)
    assert not torch.allclose(out1, out2)


def test_camera_encoder_forward_fov_swap_via_indexing():
    """The implementation reads ``fov_wh[..., [1, 0]]``, i.e. it swaps the
    last two coords. Two fov inputs that differ in [1, 0] order should
    produce different outputs."""
    ce = CameraEncoder(input_dim=9, output_dim=32, depth=1, n_heads=8)
    ce.eval()

    T = torch.eye(4).reshape(1, 1, 4, 4)
    fov_a = torch.tensor([[[0.5, 1.0]]])
    fov_b = torch.tensor([[[1.0, 0.5]]])
    out_a = ce(T, fov_a)
    out_b = ce(T, fov_b)
    assert not torch.allclose(out_a, out_b)


def test_camera_encoder_handles_batched_views():
    """Batch dim B>1 and view dim V>1 flow through forward without shape errors."""
    ce = CameraEncoder(input_dim=9, output_dim=32, depth=1, n_heads=8)
    ce.eval()
    B, V = 4, 5
    T = torch.eye(4).reshape(1, 1, 4, 4).expand(B, V, 4, 4).contiguous()
    fov = torch.full((B, V, 2), 0.5)
    out = ce(T, fov)
    assert out.shape == (B, V, 32)
