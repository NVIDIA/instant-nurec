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

"""Branch-coverage tests for instant_nurec.utils.merge_covariances.

Covers both helpers in the module:
  - ``_batched_eigh``: small / large CUDA chunking branches
  - ``merge_covariances_kl_optimal``: identity/single/multi-voxel paths,
    NaN-input guard, reflection fix, roundtrip math.
"""

import pytest
import torch

from instant_nurec.utils import merge_covariances as cov_mod
from instant_nurec.utils.merge_covariances import _batched_eigh, merge_covariances_kl_optimal
from instant_nurec.utils.geometry import quat_to_so3_matrix


def _make_identity_quat_wxyz(n: int, device: torch.device) -> torch.Tensor:
    """Create n identity quaternions in wxyz format."""
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0  # w=1, x=y=z=0
    return q


class TestBatchedEigh:
    """Branch coverage for instant_nurec.utils.merge_covariances._batched_eigh."""

    def test_small_cpu_no_chunk(self):
        """CPU input -> single-shot path; result matches torch.linalg.eigh."""
        matrices = torch.eye(3).unsqueeze(0).expand(4, 3, 3).contiguous()
        vals, vecs = _batched_eigh(matrices)
        ref_vals, ref_vecs = torch.linalg.eigh(matrices)
        torch.testing.assert_close(vals, ref_vals)
        torch.testing.assert_close(vecs, ref_vecs)

    def test_small_cuda_no_chunk(self):
        """Small CUDA input (≤ _EIGH_MAX_BATCH) takes the non-chunked branch."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        matrices = torch.eye(3, device="cuda").unsqueeze(0).expand(4, 3, 3).contiguous()
        vals, vecs = _batched_eigh(matrices)
        ref_vals, ref_vecs = torch.linalg.eigh(matrices)
        torch.testing.assert_close(vals, ref_vals)
        torch.testing.assert_close(vecs, ref_vecs)

    def test_large_cuda_chunked(self, monkeypatch):
        """CUDA + shape[0] > _EIGH_MAX_BATCH exercises the chunking branch."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        monkeypatch.setattr(cov_mod, "_EIGH_MAX_BATCH", 2)
        # 5 matrices, chunk=2 -> three chunks ([2, 2, 1]) cover boundary + remainder.
        torch.manual_seed(0)
        a = torch.randn(5, 3, 3, device="cuda")
        sym = (a + a.transpose(1, 2)) * 0.5
        vals, vecs = _batched_eigh(sym)
        ref_vals, ref_vecs = torch.linalg.eigh(sym)
        torch.testing.assert_close(vals, ref_vals)
        # Eigenvectors may differ by per-column sign across calls; reconstruct
        # the covariance and compare that.
        rebuilt = vecs * vals[:, None, :] @ vecs.transpose(1, 2)
        ref_rebuilt = ref_vecs * ref_vals[:, None, :] @ ref_vecs.transpose(1, 2)
        torch.testing.assert_close(rebuilt, ref_rebuilt, atol=1e-5, rtol=1e-5)

    def test_cpu_above_max_still_no_chunk(self, monkeypatch):
        """`is_cuda=False` always takes the single-shot branch regardless of size."""
        monkeypatch.setattr(cov_mod, "_EIGH_MAX_BATCH", 1)
        matrices = torch.eye(3).unsqueeze(0).expand(4, 3, 3).contiguous()
        vals, vecs = _batched_eigh(matrices)
        ref_vals, ref_vecs = torch.linalg.eigh(matrices)
        torch.testing.assert_close(vals, ref_vals)
        torch.testing.assert_close(vecs, ref_vecs)


class TestMergeCovariancesKLOptimal:
    """Branch coverage for merge_covariances_kl_optimal."""

    def test_identical_gaussians_same_position(self):
        """Two identical Gaussians at the same position -> output matches input."""
        device = torch.device("cpu")
        n = 2
        positions = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device=device)
        rotations_wxyz = _make_identity_quat_wxyz(n, device)
        scales = torch.tensor([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]], device=device)
        weights = torch.tensor([[0.5], [0.5]], device=device)
        inverse_indices = torch.tensor([0, 0], device=device)
        voxel_positions = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        _, scales_merged = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )

        # Covariance should be identical to input (identity rotation, same scales).
        # eigh returns eigenvalues in ascending order, so scales may be reordered.
        assert scales_merged.shape == (1, 3)
        sorted_input = torch.sort(scales[0])[0]
        sorted_output = torch.sort(scales_merged[0])[0]
        torch.testing.assert_close(sorted_output, sorted_input, atol=1e-5, rtol=1e-5)

    def test_different_positions_scale_grows(self):
        """Two Gaussians at different positions -> merged scale must be larger than input."""
        device = torch.device("cpu")
        n = 2
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
        rotations_wxyz = _make_identity_quat_wxyz(n, device)
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], device=device)
        weights = torch.tensor([[0.5], [0.5]], device=device)
        inverse_indices = torch.tensor([0, 0], device=device)
        voxel_positions = torch.tensor([[0.5, 0.0, 0.0]], device=device)

        _, scales_merged = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )

        assert scales_merged.max().item() > 0.1
        assert scales_merged.max().item() > 0.4

    def test_single_gaussian_per_voxel(self):
        """Single Gaussian per voxel -> output unchanged."""
        device = torch.device("cpu")
        positions = torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], device=device)
        rotations_wxyz = _make_identity_quat_wxyz(2, device)
        scales = torch.tensor([[0.3, 0.2, 0.1], [0.5, 0.4, 0.3]], device=device)
        weights = torch.tensor([[1.0], [1.0]], device=device)
        inverse_indices = torch.tensor([0, 1], device=device)
        voxel_positions = positions.clone()

        _, scales_merged = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )

        assert scales_merged.shape == (2, 3)
        for i in range(2):
            sorted_input = torch.sort(scales[i])[0]
            sorted_output = torch.sort(scales_merged[i])[0]
            torch.testing.assert_close(sorted_output, sorted_input, atol=1e-5, rtol=1e-5)

    def test_roundtrip_covariance(self):
        """Build covariance from (R, s), decompose back -> same covariance."""
        quat_xyzw = torch.nn.functional.normalize(torch.tensor([[0.1, 0.2, 0.3, 0.9]]), dim=1)
        R = quat_to_so3_matrix(quat_xyzw, unbatch=False)
        scales_in = torch.tensor([[0.5, 0.3, 0.1]])

        s_sq = scales_in * scales_in
        sigma = R * s_sq[:, None, :]
        sigma = sigma @ R.transpose(1, 2)

        eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
        scales_out = torch.sqrt(eigenvalues.clamp(min=1e-8))

        s_sq_out = scales_out * scales_out
        sigma_reconstructed = eigenvectors * s_sq_out[:, None, :]
        sigma_reconstructed = sigma_reconstructed @ eigenvectors.transpose(1, 2)

        torch.testing.assert_close(sigma_reconstructed, sigma, atol=1e-5, rtol=1e-5)

    def test_proper_rotation_output(self):
        """Output rotations are always proper (det=+1) — exercises the reflection-fix branch."""
        device = torch.device("cpu")
        # Seeded so the random rotations include cases that drive `eigh`'s
        # eigenvector basis into a left-handed orientation (det<0). The
        # reflection-mask branch then re-flips the first column.
        torch.manual_seed(42)
        n = 32
        positions = torch.randn(n, 3, device=device)
        rotations_wxyz = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=1)
        scales = torch.rand(n, 3, device=device) * 0.5 + 0.1
        weights = torch.ones(n, 1, device=device) / n
        inverse_indices = torch.zeros(n, dtype=torch.long, device=device)
        voxel_positions = positions.mean(dim=0, keepdim=True)

        rot_merged, _ = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )

        quat_xyzw = rot_merged[:, [1, 2, 3, 0]]
        R = quat_to_so3_matrix(quat_xyzw, unbatch=False)
        det = torch.linalg.det(R)
        torch.testing.assert_close(det, torch.ones_like(det), atol=1e-4, rtol=1e-4)

    def test_nan_inputs_dropped(self):
        """NaN/inf in any of (positions, scales, rotations) -> dropped; rest merge OK."""
        device = torch.device("cpu")
        n = 4
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]], device=device
        )
        rotations_wxyz = _make_identity_quat_wxyz(n, device)
        scales = torch.tensor(
            [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, float("inf"), 0.1], [0.1, 0.1, 0.1]], device=device
        )
        weights = torch.ones(n, 1, device=device) / n
        inverse_indices = torch.zeros(n, dtype=torch.long, device=device)
        voxel_positions = torch.tensor([[0.25, 0.0, 0.0]], device=device)

        rot_merged, scales_merged = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )
        assert torch.isfinite(scales_merged).all()
        assert torch.isfinite(rot_merged).all()

    def test_no_nan_path_no_warning(self, caplog):
        """All-finite input takes the fast path (no NaN-drop warning is emitted)."""
        import logging

        device = torch.device("cpu")
        n = 2
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
        rotations_wxyz = _make_identity_quat_wxyz(n, device)
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], device=device)
        weights = torch.tensor([[0.5], [0.5]], device=device)
        inverse_indices = torch.tensor([0, 0], device=device)
        voxel_positions = torch.tensor([[0.0, 0.0, 0.0]], device=device)

        with caplog.at_level(logging.WARNING, logger="instant_nurec.utils.merge_covariances"):
            merge_covariances_kl_optimal(
                positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
            )
        assert not any("NaN/inf" in r.message for r in caplog.records)

    def test_multi_voxel_assignment(self):
        """Inverse indices that span multiple voxels are scattered correctly per voxel."""
        device = torch.device("cpu")
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
            ],
            device=device,
        )
        rotations_wxyz = _make_identity_quat_wxyz(4, device)
        scales = torch.full((4, 3), 0.1, device=device)
        weights = torch.full((4, 1), 0.5, device=device)
        inverse_indices = torch.tensor([0, 0, 1, 1], device=device)
        voxel_positions = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], device=device)

        rot_merged, scales_merged = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )
        assert rot_merged.shape == (2, 4)
        assert scales_merged.shape == (2, 3)
        # Two identical pairs at the same voxel center -> per-voxel scales unchanged.
        for i in range(2):
            sorted_output = torch.sort(scales_merged[i])[0]
            torch.testing.assert_close(sorted_output, torch.full((3,), 0.1), atol=1e-5, rtol=1e-5)
