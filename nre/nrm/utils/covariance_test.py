# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from nre.nrm.utils.covariance import merge_covariances_kl_optimal
from nre.utils.geometry import quat_to_so3_matrix


def _make_identity_quat_wxyz(n: int, device: torch.device) -> torch.Tensor:
    """Create n identity quaternions in wxyz format."""
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0  # w=1, x=y=z=0
    return q


def _build_covariance(rotations_wxyz: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Build covariance matrices from rotations (wxyz) and scales."""
    quat_xyzw = rotations_wxyz[:, [1, 2, 3, 0]]
    R = quat_to_so3_matrix(quat_xyzw, unbatch=False)
    s_sq = scales * scales
    sigma = R * s_sq[:, None, :]
    sigma = sigma @ R.transpose(1, 2)
    return sigma


class TestMergeCovariancesKLOptimal:
    """Tests for merge_covariances_kl_optimal."""

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

        rot_merged, scales_merged = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )

        # Covariance should be identical to input (identity rotation, same scales)
        # eigh returns eigenvalues in ascending order, so scales may be reordered
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

        rot_merged, scales_merged = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )

        # The spread of positions (0.5m apart from center) adds to covariance,
        # so merged scales should be strictly larger than input (0.1)
        assert scales_merged.max().item() > 0.1
        # The max merged scale should capture the 0.5m spread
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

        rot_merged, scales_merged = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )

        assert scales_merged.shape == (2, 3)
        for i in range(2):
            sorted_input = torch.sort(scales[i])[0]
            sorted_output = torch.sort(scales_merged[i])[0]
            torch.testing.assert_close(sorted_output, sorted_input, atol=1e-5, rtol=1e-5)

    def test_roundtrip_covariance(self):
        """Build covariance from (R, s), decompose back -> same covariance."""
        # Create a rotation from a known quaternion
        quat_xyzw = torch.nn.functional.normalize(torch.tensor([[0.1, 0.2, 0.3, 0.9]]), dim=1)
        R = quat_to_so3_matrix(quat_xyzw, unbatch=False)
        scales_in = torch.tensor([[0.5, 0.3, 0.1]])

        # Build covariance
        s_sq = scales_in * scales_in
        sigma = R * s_sq[:, None, :]
        sigma = sigma @ R.transpose(1, 2)

        # Decompose
        eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
        scales_out = torch.sqrt(eigenvalues.clamp(min=1e-8))

        # Rebuild covariance from decomposed values
        s_sq_out = scales_out * scales_out
        sigma_reconstructed = eigenvectors * s_sq_out[:, None, :]
        sigma_reconstructed = sigma_reconstructed @ eigenvectors.transpose(1, 2)

        torch.testing.assert_close(sigma_reconstructed, sigma, atol=1e-5, rtol=1e-5)

    def test_proper_rotation_output(self):
        """Ensure output rotation matrices have det > 0 (proper rotation)."""
        device = torch.device("cpu")
        n = 4
        positions = torch.randn(n, 3, device=device)
        rotations_wxyz = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=1)
        scales = torch.rand(n, 3, device=device) * 0.5 + 0.1
        weights = torch.ones(n, 1, device=device) / n
        inverse_indices = torch.zeros(n, dtype=torch.long, device=device)
        voxel_positions = positions.mean(dim=0, keepdim=True)

        rot_merged, _ = merge_covariances_kl_optimal(
            positions, rotations_wxyz, scales, weights, inverse_indices, voxel_positions
        )

        # Convert back to rotation matrix and check determinant
        quat_xyzw = rot_merged[:, [1, 2, 3, 0]]
        R = quat_to_so3_matrix(quat_xyzw, unbatch=False)
        det = torch.linalg.det(R)
        torch.testing.assert_close(det, torch.ones_like(det), atol=1e-4, rtol=1e-4)

    def test_nan_inputs_do_not_crash(self):
        """NaN/inf in scales or positions are dropped; remaining Gaussians merge correctly."""
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
