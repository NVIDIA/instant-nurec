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

"""Branch-coverage tests for ``instant_nurec.utils.geometry.se3pose_from_matrix``.


The helper does Shepperd's method in float64 internally and casts the
final quaternion to f32.

Bit-exact match with slang on GPU is provably impossible (different SASS
instruction sequences); the residual ~1-3 ULP drift is absorbed by
``tests/tolerance.json``'s ``_vertex_count_delta``. These unit tests
verify the math contract rather than slang bit-equality.
"""

from __future__ import annotations

import sys

from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec.utils.geometry import (  # noqa: E402
    se3pose_from_matrix,
    tquat_to_se3_matrix,
)


def _random_se3_batch(n: int, *, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.empty((n, 4, 4), dtype=dtype)
    for i in range(n):
        A = torch.randn(3, 3, generator=g, dtype=dtype)
        Q, R = torch.linalg.qr(A)
        d = torch.sign(torch.diag(R))
        d[d == 0] = 1
        Q = Q * d.unsqueeze(0)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        t = torch.randn(3, generator=g, dtype=dtype)
        M[i, :3, :3] = Q
        M[i, :3, 3] = t
        M[i, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    return M


def test_identity_returns_zero_translation_and_unit_quaternion():
    M = torch.eye(4).reshape(1, 4, 4)
    t, q = se3pose_from_matrix(M)
    assert torch.allclose(t, torch.zeros(1, 3))
    # XYZW identity quaternion is (0, 0, 0, 1).
    assert torch.allclose(q, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_translation_only():
    M = torch.eye(4).reshape(1, 4, 4).clone()
    M[0, :3, 3] = torch.tensor([1.0, 2.0, -3.0])
    t, q = se3pose_from_matrix(M)
    assert torch.allclose(t, torch.tensor([[1.0, 2.0, -3.0]]))
    assert torch.allclose(q, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_rotation_only_pi_around_z():
    """Pi-rotation around z: q = (0, 0, sin(pi/2), cos(pi/2)) = (0, 0, 1, 0)."""
    M = torch.eye(4).reshape(1, 4, 4).clone()
    M[0, :3, :3] = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    t, q = se3pose_from_matrix(M)
    assert torch.allclose(t, torch.zeros(1, 3))
    # q and -q encode the same rotation, so abs check is the contract.
    assert torch.allclose(q.abs(), torch.tensor([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6)


def test_returns_two_separate_tensors_not_concat():
    """Wrapper returns ``(t (N,3), q (N,4))`` matching the slang kernel."""
    M = _random_se3_batch(3)
    t, q = se3pose_from_matrix(M)
    assert t.shape == (3, 3)
    assert q.shape == (3, 4)


def test_quaternions_are_unit_norm():
    M = _random_se3_batch(8, seed=1)
    _, q = se3pose_from_matrix(M)
    norms = q.norm(dim=1)
    assert torch.allclose(norms, torch.ones(8, dtype=norms.dtype), atol=1e-6)


def test_round_trip_via_tquat_to_se3():
    """Reconstructing the matrix from (t, q) recovers the input."""
    M = _random_se3_batch(8, seed=3, dtype=torch.float64)
    t, q = se3pose_from_matrix(M)
    tquat = torch.cat([t, q.to(t.dtype)], dim=1)
    M_back = tquat_to_se3_matrix(tquat, unbatch=False)
    assert torch.allclose(M_back, M, atol=1e-6)


def test_accepts_flat_n16_input():
    """Kernel signature accepts (N, 16); wrapper must too."""
    M = _random_se3_batch(4, seed=4)
    M_flat = M.reshape(4, 16)
    t, q = se3pose_from_matrix(M_flat)
    t2, q2 = se3pose_from_matrix(M)
    assert torch.allclose(t, t2)
    assert torch.allclose(q, q2)


def test_two_pose_batch_used_at_call_site():
    """The actual call site (batch.py:773) passes (2, 4, 4) — start/end pose."""
    M = _random_se3_batch(2, seed=5)
    t, q = se3pose_from_matrix(M)
    assert t.shape == (2, 3)
    assert q.shape == (2, 4)
    # Per-row indexing must work (batch.py uses ``rotations[0]`` / ``[1]``).
    assert t[0].shape == (3,)
    assert q[1].shape == (4,)


def test_outputs_are_contiguous():
    """Downstream kernels assume contiguous tensors."""
    M = _random_se3_batch(5, seed=6)
    t, q = se3pose_from_matrix(M)
    assert t.is_contiguous()
    assert q.is_contiguous()


def test_dtype_preserved_float32():
    M = _random_se3_batch(3, seed=7).to(torch.float32)
    t, q = se3pose_from_matrix(M)
    assert t.dtype == torch.float32
    assert q.dtype == torch.float32


def test_dtype_preserved_float64():
    M = _random_se3_batch(3, seed=8).to(torch.float64)
    t, q = se3pose_from_matrix(M)
    assert t.dtype == torch.float64
    assert q.dtype == torch.float64


def test_device_preserved():
    M = _random_se3_batch(3, seed=9)
    t, q = se3pose_from_matrix(M)
    assert t.device == M.device
    assert q.device == M.device
