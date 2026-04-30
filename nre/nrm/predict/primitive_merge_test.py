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

from nre.nrm.predict.primitive_merge import voxelize_with_fusion


def _make_identity_quat_wxyz(n: int, device: torch.device) -> torch.Tensor:
    """Create n identity quaternions in wxyz format."""
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0  # w=1, x=y=z=0
    return q


class TestVoxelizeWithFusion:
    """Tests for voxelize_with_fusion with fusion_mode parameter."""

    def test_average_mode_unchanged(self):
        """Average mode produces same results as before (backward compat)."""
        device = torch.device("cpu")
        pts = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], device=device)
        features = {
            "rotations": _make_identity_quat_wxyz(2, device),
            "scales": torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], device=device),
            "densities": torch.tensor([[1.0], [1.0]], device=device),
        }
        voxel_pts_avg, voxel_feats_avg = voxelize_with_fusion(pts, features, 0.1, fusion_mode="average")
        # Both points should fall in same voxel
        assert voxel_pts_avg.shape[0] == 1

    def test_kl_optimal_mode_basic(self):
        """KL-optimal mode runs without error and produces output."""
        device = torch.device("cpu")
        pts = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], device=device)
        features = {
            "rotations": _make_identity_quat_wxyz(2, device),
            "scales": torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], device=device),
            "densities": torch.tensor([[1.0], [1.0]], device=device),
        }
        voxel_pts, voxel_feats = voxelize_with_fusion(pts, features, 0.1, fusion_mode="kl_optimal")
        assert voxel_pts.shape[0] == 1
        assert "rotations" in voxel_feats
        assert "scales" in voxel_feats
        assert "densities" in voxel_feats

    def test_kl_optimal_scales_grow_with_spread(self):
        """When Gaussians are spread apart, KL-optimal scales are larger than average scales."""
        device = torch.device("cpu")
        pts = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], device=device)
        features = {
            "rotations": _make_identity_quat_wxyz(2, device),
            "scales": torch.tensor([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]], device=device),
            "densities": torch.tensor([[1.0], [1.0]], device=device),
        }
        _, feats_avg = voxelize_with_fusion(pts, dict(features), 0.1, fusion_mode="average")
        _, feats_kl = voxelize_with_fusion(pts, dict(features), 0.1, fusion_mode="kl_optimal")

        # KL-optimal max scale should be larger because it accounts for position spread
        assert feats_kl["scales"].max().item() > feats_avg["scales"].max().item()

    def test_default_fusion_mode_is_average(self):
        """Default fusion_mode is 'average'."""
        device = torch.device("cpu")
        pts = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        features = {"densities": torch.tensor([[1.0]], device=device)}
        # Should work without specifying fusion_mode
        voxel_pts, voxel_feats = voxelize_with_fusion(pts, features, 0.1)
        assert voxel_pts.shape == (1, 3)
