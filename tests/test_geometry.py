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

"""Branch-coverage tests for instant_nurec.utils.geometry.

Pure-math helpers; verified by:
  - identity / known-rotation roundtrips,
  - numpy ↔ torch input parity,
  - batched ↔ singular shape branches (unbatch=True/False),
  - quaternion XYZW convention sanity (so3 → quat → so3 round-trip),
  - quaternion multiplication identity + commutativity-with-conjugate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.utils.geometry import (
    quat_mult_xyzw,
    quat_to_so3_matrix,
    se3_matrix_inverse,
    se3_matrix_to_tquat,
    so3_matrix_to_quat,
    tquat_to_se3_matrix,
)


# ---------------------------------------------------------------------------
# se3_matrix_inverse
# ---------------------------------------------------------------------------


def _se3_from_R_t(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build a single 4x4 SE3 from a 3x3 R and 3-vector t."""
    M = torch.eye(4, dtype=R.dtype)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def test_se3_matrix_inverse_identity_is_identity():
    out = se3_matrix_inverse(torch.eye(4))
    assert torch.allclose(out, torch.eye(4))


def test_se3_matrix_inverse_inverts_translation():
    M = _se3_from_R_t(torch.eye(3), torch.tensor([1.0, 2.0, 3.0]))
    inv = se3_matrix_inverse(M)
    assert torch.allclose(inv @ M, torch.eye(4), atol=1e-6)
    assert torch.allclose(M @ inv, torch.eye(4), atol=1e-6)


def test_se3_matrix_inverse_inverts_rotation_and_translation():
    # 90° rotation about z + nontrivial translation
    R = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    M = _se3_from_R_t(R, torch.tensor([1.0, 2.0, 3.0]))
    inv = se3_matrix_inverse(M)
    assert torch.allclose(inv @ M, torch.eye(4), atol=1e-6)


def test_se3_matrix_inverse_accepts_numpy_input():
    M = np.eye(4, dtype=np.float32)
    M[0, 3] = 5.0
    inv = se3_matrix_inverse(M)
    assert isinstance(inv, torch.Tensor)
    assert pytest.approx(inv[0, 3].item(), abs=1e-6) == -5.0


def test_se3_matrix_inverse_unbatch_true_returns_2d():
    out = se3_matrix_inverse(torch.eye(4), unbatch=True)
    assert out.shape == (4, 4)


def test_se3_matrix_inverse_unbatch_false_returns_3d():
    out = se3_matrix_inverse(torch.eye(4), unbatch=False)
    assert out.shape == (1, 4, 4)


def test_se3_matrix_inverse_batched_input_keeps_batch_dim():
    batch = torch.eye(4).repeat(3, 1, 1)
    out = se3_matrix_inverse(batch, unbatch=False)
    assert out.shape == (3, 4, 4)


# ---------------------------------------------------------------------------
# se3_matrix_to_tquat / tquat_to_se3_matrix round-trips
# ---------------------------------------------------------------------------


def test_se3_to_tquat_extracts_translation():
    t = torch.tensor([1.5, -2.5, 3.5])
    M = _se3_from_R_t(torch.eye(3), t)
    tquat = se3_matrix_to_tquat(M)
    assert torch.allclose(tquat[:3], t)


def test_se3_to_tquat_identity_quat_is_unit_w():
    tquat = se3_matrix_to_tquat(torch.eye(4))
    # XYZW convention; identity rotation → (0,0,0,1)
    assert torch.allclose(tquat[3:], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


def test_se3_to_tquat_round_trip_preserves_pose():
    # 60° about a non-axis-aligned vector + translation
    angle = math.radians(60.0)
    axis = torch.tensor([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    s, c = math.sin(angle / 2), math.cos(angle / 2)
    quat = torch.tensor([axis[0] * s, axis[1] * s, axis[2] * s, c])
    R = quat_to_so3_matrix(quat)
    M = _se3_from_R_t(R, torch.tensor([0.5, 0.6, 0.7]))

    M2 = tquat_to_se3_matrix(se3_matrix_to_tquat(M))
    assert torch.allclose(M, M2, atol=1e-5)


def test_tquat_to_se3_accepts_numpy_input():
    tquat = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    M = tquat_to_se3_matrix(tquat)
    assert torch.allclose(M[:3, 3], torch.tensor([1.0, 2.0, 3.0]))


def test_se3_to_tquat_unbatch_branches():
    M = torch.eye(4)
    assert se3_matrix_to_tquat(M, unbatch=True).shape == (7,)
    assert se3_matrix_to_tquat(M, unbatch=False).shape == (1, 7)


def test_se3_to_tquat_accepts_numpy_input():
    """The implementation auto-converts np.ndarray to torch.Tensor."""
    M = np.eye(4, dtype=np.float32)
    tquat = se3_matrix_to_tquat(M, unbatch=True)
    assert tquat.shape == (7,)
    # Identity → translation [0,0,0] and quaternion [0,0,0,1] (xyzw last)
    assert torch.allclose(tquat[:3], torch.zeros(3))


def test_tquat_to_se3_unbatch_branches():
    tquat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert tquat_to_se3_matrix(tquat, unbatch=True).shape == (4, 4)
    assert tquat_to_se3_matrix(tquat, unbatch=False).shape == (1, 4, 4)


# ---------------------------------------------------------------------------
# so3_matrix_to_quat
# ---------------------------------------------------------------------------


def test_so3_to_quat_identity_is_w_unit():
    q = so3_matrix_to_quat(torch.eye(3))
    # The trace branch (Case 3) should fire for identity; |q| = 1, w ≈ 1
    assert pytest.approx(q[3].item(), abs=1e-6) == 1.0
    assert pytest.approx(q[:3].abs().sum().item(), abs=1e-6) == 0.0


def test_so3_to_quat_normalize_branch_returns_unit_quat():
    q = so3_matrix_to_quat(torch.eye(3), normalize=True)
    assert pytest.approx(q.norm().item(), abs=1e-6) == 1.0


def test_so3_to_quat_no_normalize_branch_skips_normalization():
    # Pick a rotation where the unnormalized intermediate has nontrivial norm.
    q = so3_matrix_to_quat(torch.eye(3), normalize=False)
    # For identity the unnormalized "case 3" branch returns (0,0,0,1+trace)=(0,0,0,4).
    assert pytest.approx(q[3].item(), abs=1e-6) == 4.0


def test_so3_to_quat_accepts_numpy_input():
    q = so3_matrix_to_quat(np.eye(3, dtype=np.float32))
    assert isinstance(q, torch.Tensor)


def test_so3_to_quat_rejects_non_3x3_input():
    """Either RuntimeError (from reshape) or AssertionError (from the
    explicit shape check) is acceptable; the function must reject."""
    with pytest.raises((AssertionError, RuntimeError)):
        so3_matrix_to_quat(torch.zeros(2, 2))


def test_so3_to_quat_round_trip_via_quat_to_so3_recovers_rotation():
    # 30° about z
    angle = math.radians(30.0)
    R = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R2 = quat_to_so3_matrix(so3_matrix_to_quat(R))
    assert torch.allclose(R, R2, atol=1e-6)


def test_so3_to_quat_decision_branches_via_axis_aligned_rotations():
    """Each of the 4 cases (r00 max, r11 max, r22 max, trace max) corresponds
    to a different rotation regime. Probe rotations that should hit each."""
    # 180° about x → r00 dominant
    Rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    qx = so3_matrix_to_quat(Rx)
    Rx2 = quat_to_so3_matrix(qx)
    assert torch.allclose(Rx, Rx2, atol=1e-5)

    # 180° about y → r11 dominant
    Ry = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    qy = so3_matrix_to_quat(Ry)
    Ry2 = quat_to_so3_matrix(qy)
    assert torch.allclose(Ry, Ry2, atol=1e-5)

    # 180° about z → r22 dominant
    Rz = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    qz = so3_matrix_to_quat(Rz)
    Rz2 = quat_to_so3_matrix(qz)
    assert torch.allclose(Rz, Rz2, atol=1e-5)

    # Identity → trace dominant
    qI = so3_matrix_to_quat(torch.eye(3))
    RI2 = quat_to_so3_matrix(qI)
    assert torch.allclose(torch.eye(3), RI2, atol=1e-6)


# ---------------------------------------------------------------------------
# quat_to_so3_matrix
# ---------------------------------------------------------------------------


def test_quat_to_so3_identity_quat_is_identity_matrix():
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])  # XYZW unit
    R = quat_to_so3_matrix(q)
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_quat_to_so3_normalize_branch():
    # Non-unit input quaternion; with normalize=True, R should still be a rotation.
    q = torch.tensor([0.0, 0.0, 0.0, 2.0])  # 2 * unit-w
    R = quat_to_so3_matrix(q, normalize=True)
    # Must be orthogonal with det=+1.
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
    assert pytest.approx(torch.linalg.det(R).item(), abs=1e-5) == 1.0


def test_quat_to_so3_no_normalize_branch_takes_input_as_is():
    """The normalize=False branch is the path under test — verify the
    function returns a (3, 3) tensor without raising."""
    q = torch.tensor([0.0, 0.0, 0.0, 2.0])
    R = quat_to_so3_matrix(q, normalize=False)
    assert R.shape == (3, 3)


def test_quat_to_so3_accepts_numpy_input():
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    R = quat_to_so3_matrix(q)
    assert isinstance(R, torch.Tensor)


def test_quat_to_so3_unbatch_branches():
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert quat_to_so3_matrix(q, unbatch=True).shape == (3, 3)
    assert quat_to_so3_matrix(q, unbatch=False).shape == (1, 3, 3)


# ---------------------------------------------------------------------------
# quat_mult_xyzw
# ---------------------------------------------------------------------------


def test_quat_mult_identity_left_is_noop():
    qI = torch.tensor([0.0, 0.0, 0.0, 1.0])
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])  # 120° about (1,1,1)/√3
    out = quat_mult_xyzw(qI, q)
    assert torch.allclose(out, q, atol=1e-6)


def test_quat_mult_identity_right_is_noop():
    qI = torch.tensor([0.0, 0.0, 0.0, 1.0])
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])
    out = quat_mult_xyzw(q, qI)
    assert torch.allclose(out, q, atol=1e-6)


def test_quat_mult_q_times_q_conjugate_is_identity():
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])  # unit quat
    q_conj = torch.tensor([-q[0], -q[1], -q[2], q[3]])
    out = quat_mult_xyzw(q, q_conj)
    qI = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(out, qI, atol=1e-6)


def test_quat_mult_batched_shape_preserved():
    q1 = torch.zeros(4, 5, 4)
    q1[..., 3] = 1.0
    q2 = torch.zeros(4, 5, 4)
    q2[..., 3] = 1.0
    out = quat_mult_xyzw(q1, q2)
    assert out.shape == (4, 5, 4)
