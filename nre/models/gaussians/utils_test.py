# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

import numpy as np
import torch

from nre.models.gaussians.utils import subsample_points, track_random_initialization
from nre.utils.types import PointCloud


class TestTrackRandomInitialization(unittest.TestCase):
    def test_track_random_initialization_shapes_and_values(self):
        device = torch.device("cuda")

        # Deterministic random for reproducibility
        torch.manual_seed(123)

        num_points = 64
        track_index = 7
        cuboid_extent = torch.tensor([2.0, 4.0, 6.0], device=device)
        default_scale = 0.3

        pc, gid, gscale = track_random_initialization(
            num_points=num_points,
            track_index=track_index,
            cuboid_extent=cuboid_extent,
            default_scale=default_scale,
            device=device,
        )

        # Basic types and shapes
        self.assertIsInstance(pc, PointCloud)
        self.assertEqual(pc.n_points, num_points)
        self.assertEqual(pc.xyz_start.shape, (num_points, 3))
        self.assertTrue(torch.equal(pc.xyz_start, pc.xyz_end))
        assert pc.color is not None
        self.assertEqual(pc.color.shape, (num_points, 3))
        self.assertEqual(pc.color.dtype, torch.uint8)
        assert pc.camera_footprint_scale is not None
        self.assertEqual(pc.camera_footprint_scale.shape, (num_points,))

        self.assertEqual(gid.shape, (num_points, 1))
        self.assertEqual(gid.dtype, torch.int32)
        self.assertTrue(torch.all(gid == track_index))

        self.assertEqual(gscale.shape, (num_points, 1))
        self.assertTrue(torch.allclose(gscale.squeeze(-1), pc.camera_footprint_scale))

        # Devices
        self.assertEqual(pc.xyz_start.device.type, device.type)
        self.assertEqual(pc.xyz_end.device.type, device.type)
        self.assertEqual(pc.color.device.type, device.type)
        self.assertEqual(pc.camera_footprint_scale.device.type, device.type)
        self.assertEqual(gid.device.type, device.type)
        self.assertEqual(gscale.device.type, device.type)

        # Values
        self.assertTrue(
            torch.allclose(pc.camera_footprint_scale, torch.full((num_points,), default_scale, device=device))
        )

        # Points should lie within [-extent/2, extent/2] per axis
        lower = -cuboid_extent / 2
        upper = cuboid_extent / 2
        self.assertTrue(torch.all(pc.xyz_end >= lower))
        self.assertTrue(torch.all(pc.xyz_end <= upper))


class TestSubsamplePoints(unittest.TestCase):
    def _make_pc(self, n: int, base: float, device: torch.device) -> PointCloud:
        # Construct deterministic, easily-checkable coordinates
        xyz = torch.stack(
            [
                torch.linspace(base, base + n - 1, n, device=device),
                torch.linspace(base + 1000, base + 1000 + n - 1, n, device=device),
                torch.linspace(base + 2000, base + 2000 + n - 1, n, device=device),
            ],
            dim=1,
        )

        color = (torch.arange(n, device=device)[:, None].repeat(1, 3) % 256).to(torch.uint8)
        scale = torch.linspace(0.1, 1.0, n, device=device)
        return PointCloud(
            xyz_start=xyz,
            xyz_end=xyz,
            color=color,
            camera_footprint_scale=scale,
        )

    def test_no_subsampling_when_below_threshold(self):
        device = torch.device("cuda")

        pc1 = self._make_pc(5, base=0.0, device=device)
        pc2 = self._make_pc(7, base=100.0, device=device)
        gid1 = torch.full((pc1.n_points, 1), 1, dtype=torch.int32, device=device)
        gid2 = torch.full((pc2.n_points, 1), 2, dtype=torch.int32, device=device)
        assert pc1.camera_footprint_scale is not None
        assert pc2.camera_footprint_scale is not None
        gsc1 = pc1.camera_footprint_scale.unsqueeze(-1)
        gsc2 = pc2.camera_footprint_scale.unsqueeze(-1)

        num_points_target = pc1.n_points + pc2.n_points  # equal -> no subsampling
        observed_counts = [pc1.n_points, pc2.n_points]

        pcs_out, gids_out, gscs_out, counts_out = subsample_points(
            num_points=num_points_target,
            point_clouds=[pc1, pc2],
            gaussian_cuboid_ids=[gid1, gid2],
            gaussian_scales=[gsc1, gsc2],
            observed_counts=observed_counts,
        )

        # Should be unchanged in sizes
        self.assertEqual(pcs_out[0].n_points, pc1.n_points)
        self.assertEqual(pcs_out[1].n_points, pc2.n_points)
        self.assertTrue(torch.equal(gids_out[0], gid1))
        self.assertTrue(torch.equal(gids_out[1], gid2))
        self.assertTrue(torch.allclose(gscs_out[0], gsc1))
        self.assertTrue(torch.allclose(gscs_out[1], gsc2))
        self.assertEqual(counts_out, observed_counts)

    def test_deterministic_subsampling(self):
        device = torch.device("cuda")

        pc1 = self._make_pc(10, base=0.0, device=device)
        pc2 = self._make_pc(6, base=100.0, device=device)
        gid1 = torch.full((pc1.n_points, 1), 11, dtype=torch.int32, device=device)
        gid2 = torch.full((pc2.n_points, 1), 22, dtype=torch.int32, device=device)
        assert pc1.camera_footprint_scale is not None
        assert pc2.camera_footprint_scale is not None
        gsc1 = pc1.camera_footprint_scale.unsqueeze(-1)
        gsc2 = pc2.camera_footprint_scale.unsqueeze(-1)

        total_points = pc1.n_points + pc2.n_points  # 16
        num_points_target = 8  # ratio = 0.5 -> expect 5 and 3
        ratio = float(num_points_target) / total_points
        n1 = int(pc1.n_points * ratio)
        n2 = int(pc2.n_points * ratio)
        self.assertEqual(n1, 5)
        self.assertEqual(n2, 3)

        # Prepare deterministic numpy RNG and compute expected indices
        seed = 0
        np.random.seed(seed)
        expected_idx1 = np.random.choice(pc1.n_points, n1, replace=False)
        expected_idx2 = np.random.choice(pc2.n_points, n2, replace=False)

        # Reset seed so function uses the same random draws
        np.random.seed(seed)
        observed_counts = [pc1.n_points, pc2.n_points]
        pcs_out, gids_out, gscs_out, counts_out = subsample_points(
            num_points=num_points_target,
            point_clouds=[pc1, pc2],
            gaussian_cuboid_ids=[gid1, gid2],
            gaussian_scales=[gsc1, gsc2],
            observed_counts=observed_counts,
        )

        # Validate sizes
        self.assertEqual(pcs_out[0].n_points, n1)
        self.assertEqual(pcs_out[1].n_points, n2)
        self.assertEqual(gids_out[0].shape, (n1, 1))
        self.assertEqual(gids_out[1].shape, (n2, 1))
        self.assertEqual(gscs_out[0].shape, (n1, 1))
        self.assertEqual(gscs_out[1].shape, (n2, 1))
        self.assertEqual(counts_out, observed_counts)

        # Check that subsampled elements match expected indices
        self.assertTrue(torch.allclose(pcs_out[0].xyz_end, pc1.xyz_end[torch.tensor(expected_idx1, device=device)]))
        self.assertTrue(torch.allclose(pcs_out[1].xyz_end, pc2.xyz_end[torch.tensor(expected_idx2, device=device)]))
        self.assertTrue(torch.equal(gids_out[0], gid1[torch.tensor(expected_idx1, device=device)]))
        self.assertTrue(torch.equal(gids_out[1], gid2[torch.tensor(expected_idx2, device=device)]))
        self.assertTrue(torch.allclose(gscs_out[0], gsc1[torch.tensor(expected_idx1, device=device)]))
        self.assertTrue(torch.allclose(gscs_out[1], gsc2[torch.tensor(expected_idx2, device=device)]))


if __name__ == "__main__":
    unittest.main()
