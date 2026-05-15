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

from __future__ import annotations

import logging

import torch  # type: ignore
import torch_scatter  # type: ignore

from instant_nurec.utils.geometry import quat_to_so3_matrix, so3_matrix_to_quat


logger = logging.getLogger(__name__)

# cusolver's batched eigh can fail with CUSOLVER_STATUS_INVALID_VALUE for very
# large batches (observed at ~18M 3x3 matrices). Chunking avoids this.
_EIGH_MAX_BATCH = 1_000_000


def _batched_eigh(matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh with automatic chunking for large GPU batches."""
    if matrices.shape[0] <= _EIGH_MAX_BATCH or not matrices.is_cuda:
        return torch.linalg.eigh(matrices)
    eigenvalues_list: list[torch.Tensor] = []
    eigenvectors_list: list[torch.Tensor] = []
    for chunk in matrices.split(_EIGH_MAX_BATCH):
        vals, vecs = torch.linalg.eigh(chunk)
        eigenvalues_list.append(vals)
        eigenvectors_list.append(vecs)
    return torch.cat(eigenvalues_list), torch.cat(eigenvectors_list)


def merge_covariances_kl_optimal(
    positions: torch.Tensor,
    rotations_wxyz: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    inverse_indices: torch.Tensor,
    voxel_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    KL-optimal (moment-matched) merge of Gaussian spatial parameters within voxels.

    Computes the merged covariance as the weighted sum of per-Gaussian covariances plus
    the spread of their means, then decomposes back into rotation and scale.

    Args:
        positions: Per-Gaussian positions [N, 3]
        rotations_wxyz: Per-Gaussian rotations in wxyz format [N, 4]
        scales: Per-Gaussian scales [N, 3]
        weights: Per-Gaussian normalized weights [N, 1]
        inverse_indices: Voxel assignment per Gaussian [N]
        voxel_positions: Weighted-average voxel positions [M, 3]

    Returns:
        (rotations_wxyz_merged [M, 4], scales_merged [M, 3])
    """
    # Drop invalid Gaussians: zero out weights for any Gaussian with NaN/inf.
    valid = (
        torch.isfinite(positions).all(dim=1)
        & torch.isfinite(scales).all(dim=1)
        & torch.isfinite(rotations_wxyz).all(dim=1)
    )
    n_bad = (~valid).sum().item()
    if n_bad > 0:
        logger.warning(
            "[merge_covariances_kl_optimal] %d/%d Gaussians have NaN/inf; dropping from covariance merge.",
            n_bad,
            positions.shape[0],
        )
        positions = positions[valid]
        rotations_wxyz = rotations_wxyz[valid]
        scales = scales[valid]
        weights = weights[valid]
        inverse_indices = inverse_indices[valid]

    # Convert wxyz -> xyzw -> rotation matrices [N, 3, 3]
    quat_xyzw = rotations_wxyz[:, [1, 2, 3, 0]]
    R = quat_to_so3_matrix(quat_xyzw, unbatch=False)  # [N, 3, 3]

    # Per-Gaussian covariance: Sigma_i = R_i @ diag(s_i^2) @ R_i^T
    s_sq = scales * scales  # [N, 3]
    sigma = R * s_sq[:, None, :]  # broadcast: [N, 3, 3] (R_i * diag element-wise per column)
    sigma = sigma @ R.transpose(1, 2)  # [N, 3, 3]

    # Position residuals relative to voxel center
    delta = positions - voxel_positions[inverse_indices]  # [N, 3]
    # Spread term: delta_i @ delta_i^T
    spread = delta.unsqueeze(2) * delta.unsqueeze(1)  # [N, 3, 3]

    # Total per-Gaussian contribution: (Sigma_i + delta_i delta_i^T) * w_i
    sigma_total = (sigma + spread) * weights.unsqueeze(-1)  # [N, 3, 3]

    # Sum per voxel
    sigma_total_flat = sigma_total.reshape(-1, 9)  # [N, 9]
    num_voxels = voxel_positions.shape[0]
    sigma_merged = torch_scatter.scatter_add(sigma_total_flat, inverse_indices, dim=0, dim_size=num_voxels)  # [M, 9]
    sigma_merged = sigma_merged.reshape(-1, 3, 3)  # [M, 3, 3]

    # Symmetrize to avoid numerical issues
    sigma_merged = (sigma_merged + sigma_merged.transpose(1, 2)) * 0.5

    # Regularize: add eps * I to ensure positive semi-definiteness for eigh.
    eps = 1e-6
    eye3 = torch.eye(3, device=sigma_merged.device, dtype=sigma_merged.dtype).unsqueeze(0)
    sigma_merged = sigma_merged + eps * eye3

    # Replace any remaining non-finite entries (NaN/inf from numerical overflow) with eps * I.
    bad_mask = ~torch.isfinite(sigma_merged).all(dim=-1).all(dim=-1)  # [M]
    if bad_mask.any():
        sigma_merged[bad_mask] = eps * eye3

    # Eigendecompose — chunk to avoid cusolver batch-size limits on GPU.
    eigenvalues, eigenvectors = _batched_eigh(sigma_merged)

    # Extract scales: sqrt(clamp(eigenvalues))
    scales_merged = torch.sqrt(eigenvalues.clamp(min=eps))  # [M, 3]

    # Fix reflections: ensure det(eigenvectors) > 0 for proper rotation
    det = torch.linalg.det(eigenvectors)  # [M]
    reflection_mask = det < 0
    if reflection_mask.any():
        eigenvectors = eigenvectors.clone()
        eigenvectors[reflection_mask, :, 0] *= -1

    # Convert eigenvectors -> quaternion (xyzw) -> wxyz
    quat_xyzw_merged = so3_matrix_to_quat(eigenvectors, unbatch=False)  # [M, 4]
    rotations_wxyz_merged = quat_xyzw_merged[:, [3, 0, 1, 2]]  # -> wxyz

    return rotations_wxyz_merged, scales_merged
