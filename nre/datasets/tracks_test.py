# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import lietorch as lt
import numpy as np
import pytest
import torch

from scipy.spatial.transform import Rotation

from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks, RayIntersectionTransformFilter, TrackFlags, Tracks
from nre.models.custom_modules import ray_samples_in_distranges_masks
from nre.utils.misc import get_pack_info_from_n
from nre.utils.tests import CommonTestCase


class SimpleCuboidTracksTest(CommonTestCase):
    def setUp(self) -> None:
        # Create simple track "scene"

        #           ┌──────┐     ┌──────┐
        #           │      │     │      │
        #           │   ───┼───► │      │
        #           │      │     │      │
        #           └──────┘     └──────┘
        #
        #
        #           ┌──────┐
        #           │      │
        #           │      │
        #           │      │
        #           └──────┘
        #
        #    ▲         ▲      ▲
        #    │   ▲     │      │
        #    │   │     │      │
        #    │   └──►  │      │

        tracks_id: list[str] = []
        tracks_poses: list[np.ndarray] = []
        tracks_timestamps_us: list[np.ndarray] = []
        tracks_label_class: list[str] = []
        tracks_flags: list[TrackFlags] = []
        cuboids_dims: list[np.ndarray] = []

        # 1m cube cuboid static track at (1, 1.5, 0)
        tracks_id.append("static")
        pose0 = np.eye(4, 4, dtype=np.float32)
        pose0[:3, 3] = [1, 1.5, 0]
        poses = np.stack([pose0, pose0])
        tracks_poses.append(poses)
        tracks_timestamps_us.append(np.array([0, 1000000], np.int64))  # at time 0sec -> 1sec
        tracks_label_class.append("automobile")
        tracks_flags.append(TrackFlags.NONE)
        cuboids_dims.append(np.array([1, 1, 1], dtype=np.float32))

        # 1m cube cuboid dynamic track between (1, 3, 0) and (3, 3, 0)
        tracks_id.append("dynamic")
        pose0 = np.eye(4, 4, dtype=np.float32)
        pose0[:3, 3] = [1, 3, 0]
        pose1 = np.eye(4, 4, dtype=np.float32)
        pose1[:3, 3] = [3, 3, 0]
        poses = np.stack([pose0, pose1])
        tracks_poses.append(poses)
        tracks_timestamps_us.append(np.array([0, 1000000], np.int64))  # at time 0sec -> 1sec

        tracks_label_class.append("pedestrian")
        tracks_flags.append(TrackFlags.DYNAMIC)
        cuboids_dims.append(np.array([1, 1, 1], dtype=np.float32))

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims=cuboids_dims
        )

        # Create three rays intersecting none, two, and one track
        self.rays_o = torch.tensor([[-1, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.float32, device=torch.device("cuda"))
        self.rays_d = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32, device=torch.device("cuda"))
        self.rays_timestamps_us = torch.tensor([0, 0, 1000000 // 2], dtype=torch.int64, device=torch.device("cuda"))

    def test_ray_intersection_transform_filter(self):
        ## no cuboid padding
        (
            intersection_rays_cuboid_o,
            intersection_rays_cuboid_d,
            intersection_rays_ts,
            intersection_idxs,
        ) = self.tracks.ray_intersection_transform_filter(
            self.rays_o,
            self.rays_d,
            self.rays_timestamps_us,
        )

        self._compareTensor(intersection_rays_cuboid_o, np.array([[0.0, -1.5, 0.0], [0.0, -3, 0.0], [0.0, -3, 0.0]]))
        self._compareTensor(intersection_rays_cuboid_d, np.array([[0.0, 1.0, 0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
        self._compareTensor(intersection_rays_ts, np.array([[1.0, 2.0], [2.5, 3.5], [2.5, 3.5]]))
        self._compareTensor(intersection_idxs, np.array([[1, 0], [1, 1], [2, 1]]))

        ## with cuboid padding
        cuboids_dims_padding = 0.25
        (
            intersection_rays_cuboid_o_padding,
            intersection_rays_cuboid_d_padding,
            intersection_rays_ts_padding,
            intersection_idxs_padding,
        ) = self.tracks.ray_intersection_transform_filter(
            self.rays_o, self.rays_d, self.rays_timestamps_us, cuboids_dims_padding=cuboids_dims_padding
        )

        # positive padding doesn't affect the transformed rays
        self._compareTensor(intersection_rays_cuboid_o_padding, intersection_rays_cuboid_o)
        self._compareTensor(intersection_rays_cuboid_d_padding, intersection_rays_cuboid_d)
        self._compareTensor(intersection_idxs_padding, intersection_idxs)
        self._compareTensor(
            intersection_rays_ts_padding,
            intersection_rays_ts
            # as we intersect along the x axis only, the intersection_rays_ts will change based on half of the padding
            + torch.tensor([-cuboids_dims_padding / 2, cuboids_dims_padding / 2], device=torch.device("cuda")),
        )

    def test_ray_samples_in_distranges_masks(self):
        # compute regular filtered intersections
        (
            _,
            _,
            intersection_rays_ts,
            intersection_idxs,
        ) = self.tracks.ray_intersection_transform_filter(
            self.rays_o,
            self.rays_d,
            self.rays_timestamps_us,
        )

        # create packed distance samples along surviving rays
        ray_idx, inv_ray_idx, n_intersections_of_ray = torch.unique_consecutive(
            intersection_idxs[:, 0], return_counts=True, return_inverse=True
        )

        N_rays = len(ray_idx)
        N_samples_per_ray = 40

        rays_samples_packinfo = get_pack_info_from_n(
            torch.tensor([N_samples_per_ray] * N_rays, device=torch.device("cuda"), dtype=torch.int32),
        )
        rays_samples_t = torch.linspace(
            0, 4, steps=N_samples_per_ray, dtype=torch.float32, device=torch.device("cuda")
        ).repeat(N_rays)

        # create packed distance ranges for each ray
        rays_distranges_packinfo = get_pack_info_from_n(n_intersections_of_ray.to(torch.int32))
        rays_distranges_ts = intersection_rays_ts[inv_ray_idx]

        rays_samples_distranges_cover = ray_samples_in_distranges_masks(
            rays_samples_packinfo, rays_samples_t, rays_distranges_packinfo, rays_distranges_ts
        )

        # we expect to have at least a single sample inside the query ranges (true for current test-scene)
        self.assertTrue(rays_samples_distranges_cover.any())

        # check validity of results explicitly
        for ray_idx in range(N_rays):
            sample_start_idx = rays_samples_packinfo[ray_idx, 0]
            distranges_start_idx = rays_distranges_packinfo[ray_idx, 0]
            for sample_idx in range(rays_samples_packinfo[ray_idx, 1]):
                sample_t = rays_samples_t[sample_start_idx + sample_idx]
                cover_ref = False
                for distrange_idx in range(rays_distranges_packinfo[ray_idx, 1]):
                    if (
                        rays_distranges_ts[distranges_start_idx + distrange_idx][0] <= sample_t
                        and sample_t <= rays_distranges_ts[distranges_start_idx + distrange_idx][1]
                    ):
                        cover_ref = True
                        break

                self.assertEqual(cover_ref, rays_samples_distranges_cover[sample_start_idx + sample_idx])


class RandomCuboidTracksTest(CommonTestCase):
    def setUp(self) -> None:
        # Create a random track "scene".
        # The tracks are randomly distributed within [-10, 10]^3, with random rotation.
        # Each track is of very large size 10x10x10 to maximize the chance of intersection.
        # The rays are originating from the origin (with a random delta), and pointing randomly.
        # Timestamp is from 0 to 10000.

        tracks_id: list[str] = []
        tracks_poses: list[np.ndarray] = []
        tracks_timestamps_us: list[np.ndarray] = []
        tracks_label_class: list[str] = []
        tracks_flags: list[TrackFlags] = []
        cuboids_dims: list[np.ndarray] = []

        rng = np.random.RandomState(0)

        for idx in range(100):
            tracks_id.append(f"track_{idx}")
            poses = np.repeat(np.eye(4, 4, dtype=np.float32)[None, :, :], 10, axis=0)
            poses[:, :3, :3] = Rotation.random(10, random_state=rng).as_matrix()
            poses[:, :3, 3] = rng.uniform(-10, 10, (10, 3))
            tracks_poses.append(poses)
            tracks_timestamps_us.append(np.linspace(0, 10000, 10, dtype=np.int64))
            tracks_label_class.append("unknown")
            track_flag = TrackFlags.NONE
            for value in TrackFlags.__members__:
                if rng.rand() < 0.5:
                    track_flag |= TrackFlags[value]
            tracks_flags.append(track_flag)
            cuboids_dims.append(np.array([10, 10, 10], dtype=np.float32))

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims=cuboids_dims
        )

        # Create random rays with origin close to the center of the scene and random direction
        rays_o = rng.uniform(-1, 1, (1000, 3)).astype(np.float32)
        rays_d = rng.randn(1000, 3).astype(np.float32)
        rays_d /= np.linalg.norm(rays_d, axis=1)[:, None]
        rays_timestamps_us = rng.randint(0, 10000, 1000).astype(np.int64)

        self.rays_o = torch.from_numpy(rays_o).cuda()
        self.rays_d = torch.from_numpy(rays_d).cuda()
        self.rays_timestamps_us = torch.from_numpy(rays_timestamps_us).cuda()

        # Create an empty track "scene".
        self.empty_tracks = CuboidTracks.Factory.empty()

    def test_ray_intersection_transform_filter_gradient(self):
        ## test gradient
        self.rays_o.requires_grad = True
        self.rays_d.requires_grad = True
        self.tracks.tracks_poses.data.requires_grad = True
        (
            intersection_rays_cuboid_o,
            intersection_rays_cuboid_d,
            intersection_rays_ts,
            intersection_idxs,
        ) = self.tracks.ray_intersection_transform_filter(
            self.rays_o,
            self.rays_d,
            self.rays_timestamps_us,
        )

        # Check that we have at least one valid intersection
        assert intersection_rays_cuboid_o.shape[0] > 0, "Test requires at least one intersection"

        grad_intersection_rays_cuboid_o = torch.randn_like(intersection_rays_cuboid_o)
        grad_intersection_rays_cuboid_d = torch.randn_like(intersection_rays_cuboid_d)

        intersection_rays_cuboid_o.backward(grad_intersection_rays_cuboid_o, retain_graph=True)
        intersection_rays_cuboid_d.backward(grad_intersection_rays_cuboid_d)

        grad_rays_o = torch.clone(self.rays_o.grad)
        grad_rays_d = torch.clone(self.rays_d.grad)
        grad_tracks_poses = torch.clone(self.tracks.tracks_poses.data.grad)
        self.rays_o.grad.zero_()
        self.rays_d.grad.zero_()
        self.tracks.tracks_poses.data.grad.zero_()

        # Compute the same thing with lietorch
        pose_indices_array = []
        pose_alpha_array = []
        for ray_idx, track_idx in intersection_idxs.cpu().numpy():
            pose_offset = self.tracks.tracks_packinfo[track_idx, 0]
            pose_count = self.tracks.tracks_packinfo[track_idx, 1]
            pose_time = self.rays_timestamps_us[ray_idx]
            candidate_times = self.tracks.tracks_timestamps_us[pose_offset : pose_offset + pose_count]
            pose_ind = torch.searchsorted(candidate_times, pose_time, right=True) - 1
            pose_indices_array.append(pose_offset + pose_ind.int())
            pose_alpha_array.append(
                (pose_time - candidate_times[pose_ind]) / (candidate_times[pose_ind + 1] - candidate_times[pose_ind])
            )

        pose_indices = torch.tensor(pose_indices_array, device=torch.device("cuda"))
        pose_alpha = torch.tensor(pose_alpha_array, device=torch.device("cuda"))

        pose_start = self.tracks.tracks_poses[pose_indices]
        pose_end = self.tracks.tracks_poses[pose_indices + 1]
        R_start = lt.SO3.InitFromVec(pose_start.vec()[:, 3:])
        R_end = lt.SO3.InitFromVec(pose_end.vec()[:, 3:])
        t_start, t_end = pose_start.translation(), pose_end.translation()

        R_alpha = R_start * lt.SO3.exp(pose_alpha[:, None] * (R_start.inv() * R_end).log())
        t_alpha = (1 - pose_alpha)[:, None] * t_start + pose_alpha[:, None] * t_end

        intersection_rays_cuboid_o_ref = R_alpha.inv().act(self.rays_o[intersection_idxs[:, 0]] - t_alpha[:, :3])
        intersection_rays_cuboid_d_ref = R_alpha.inv().act(self.rays_d[intersection_idxs[:, 0]])

        self._compareTensor(intersection_rays_cuboid_o.detach(), intersection_rays_cuboid_o_ref.detach(), decimal=4)
        self._compareTensor(intersection_rays_cuboid_d.detach(), intersection_rays_cuboid_d_ref.detach(), decimal=4)

        intersection_rays_cuboid_o_ref.backward(grad_intersection_rays_cuboid_o, retain_graph=True)
        intersection_rays_cuboid_d_ref.backward(grad_intersection_rays_cuboid_d)

        grad_rays_o_ref = torch.clone(self.rays_o.grad)
        grad_rays_d_ref = torch.clone(self.rays_d.grad)
        grad_tracks_poses_ref = torch.clone(self.tracks.tracks_poses.data.grad)

        self._compareTensor(grad_rays_o, grad_rays_o_ref, decimal=4)
        self._compareTensor(grad_rays_d, grad_rays_d_ref, decimal=4)
        self._compareTensor(grad_tracks_poses, grad_tracks_poses_ref, decimal=4)

    def test_serialization(self):
        def check(value: CuboidTracks | Tracks, ref: CuboidTracks | Tracks):
            self.assertEqual(value.tracks_id, ref.tracks_id)
            self._compareTensor(value.tracks_packinfo, ref.tracks_packinfo)
            self._compareTensor(value.tracks_poses.data, ref.tracks_poses.data)
            self._compareTensor(value.tracks_timestamps_us, ref.tracks_timestamps_us)
            self._compareTensor(value.tracks_flags, ref.tracks_flags)
            self.assertEqual(value.max_track_n_poses, ref.max_track_n_poses)
            self.assertEqual(value.tracks_label_class, ref.tracks_label_class)

            if isinstance(value, CuboidTracks) and isinstance(ref, CuboidTracks):
                self._compareTensor(value.cuboids_dims, ref.cuboids_dims)

        test_cases = [self.tracks, self.empty_tracks]
        for tracks in test_cases:
            with self.subTest(tracks=tracks):
                # Serialize and deserialize CuboidTracks
                cuboidtracks_from_dict = CuboidTracks.from_dict(tracks.to_dict())
                cuboidtracks_from_json = CuboidTracks.from_json(tracks.to_json())

                # Check that the data is the same
                check(cuboidtracks_from_dict, tracks)
                check(cuboidtracks_from_json, tracks)

                # Serialize and deserialize Tracks
                base_tracks = Tracks(tracks_data=tracks.tracks_data)
                tracks_from_dict = Tracks.from_dict(base_tracks.to_dict())
                tracks_from_json = Tracks.from_json(base_tracks.to_json())

                # Check that the data is the same
                check(tracks_from_dict, base_tracks)
                check(tracks_from_json, base_tracks)


class RayIntersectionTransformFilterTest(CommonTestCase):
    def setUp(self):
        self.rays_o = torch.tensor(
            [[0.3731, 0.8346, 0.3223], [0.4916, 0.1230, 0.5064]], dtype=torch.float32, device="cuda:0"
        )
        self.rays_d = torch.tensor(
            [[0.6650, 0.2934, 0.9937], [0.7220, 0.8061, 0.2895]], dtype=torch.float32, device="cuda:0"
        )
        self.rays_timestamps_us = torch.tensor([437924324, 437924324], dtype=torch.int64, device="cuda:0")
        self.tracks_packinfo = torch.tensor([[0, 2], [2, 2]], dtype=torch.int32, device="cuda:0")
        self.tracks_poses_data = torch.tensor(
            [
                [0.9650, 0.9710, 0.5485, 0.0000, 0.0000, 0.0000, 1.0000],
                [0.1416, 0.3529, 0.5123, 0.0000, 0.0000, 0.0000, 1.0000],
                [0.9650, 0.9710, 0.5485, 0.0000, 0.0000, 0.0000, 1.0000],
                [0.1416, 0.3529, 0.5123, 0.0000, 0.0000, 0.0000, 1.0000],
            ],
            dtype=torch.float32,
            device="cuda:0",
        )
        self.tracks_timestamps_us = torch.tensor(
            [409604592, 437924324, 409604592, 437924324], dtype=torch.int64, device="cuda:0"
        )
        self.cuboids_dims = torch.tensor(
            [[0.9260, 0.2189, 0.3379], [0.8446, 0.8691, 0.7651]], dtype=torch.float32, device="cuda:0"
        )
        self.max_track_n_poses = 2
        self.row_major_order = True

        # gold data
        self.intersection_rays_cuboid_o_gold = torch.tensor(
            [[0.3500, -0.2300, -0.0059], [0.3500, -0.2300, -0.0059]], dtype=torch.float32, device="cuda:0"
        )
        self.intersection_rays_cuboid_d_gold = torch.tensor(
            [[0.7220, 0.8061, 0.2895], [0.7220, 0.8061, 0.2895]], dtype=torch.float32, device="cuda:0"
        )
        self.intersection_rays_ts_gold = torch.tensor(
            [[0.1495, 0.1565], [-0.2538, 0.1001]], dtype=torch.float32, device="cuda:0"
        )
        self.intersection_idxs_gold = torch.tensor([[1, 0], [1, 1]], dtype=torch.int32, device="cuda:0")
        return

    def test_ray_intersection_transform_filter(self):
        (
            intersection_rays_cuboid_o,
            intersection_rays_cuboid_d,
            intersection_rays_ts,
            intersection_idxs,
        ) = RayIntersectionTransformFilter.apply(
            self.rays_o,
            self.rays_d,
            self.rays_timestamps_us,
            self.tracks_packinfo,
            self.tracks_poses_data,
            self.tracks_timestamps_us,
            self.cuboids_dims,
            self.max_track_n_poses,
            self.row_major_order,
        )
        self._compareTensor(
            intersection_rays_cuboid_o.detach(), self.intersection_rays_cuboid_o_gold.detach(), decimal=4
        )
        self._compareTensor(
            intersection_rays_cuboid_d.detach(), self.intersection_rays_cuboid_d_gold.detach(), decimal=4
        )
        self._compareTensor(intersection_rays_ts.detach(), self.intersection_rays_ts_gold.detach(), decimal=4)
        self._compareTensor(intersection_idxs.detach(), self.intersection_idxs_gold.detach(), decimal=4)


class BinarySearchIntersectionTest(CommonTestCase):
    def setUp(self):
        # Create a test track with known timestamps for testing binary search behavior
        tracks_id = ["test_track"]
        tracks_poses = [np.eye(4, dtype=np.float32)[None, :, :].repeat(4, axis=0)]
        tracks_timestamps_us = [np.array([100000, 200000, 300000, 400000], dtype=np.int64)]
        tracks_label_class = ["test"]
        tracks_flags = [TrackFlags.NONE]
        cuboids_dims = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims
        )

        # Create test rays that will intersect the cuboid at different timestamps
        self.rays_o = torch.tensor(
            [
                [0.0, -1.0, 0.0],  # Ray before first timestamp
                [0.0, -1.0, 0.0],  # Ray at first timestamp
                [0.0, -1.0, 0.0],  # Ray between timestamps
                [0.0, -1.0, 0.0],  # Ray at last timestamp
                [0.0, -1.0, 0.0],  # Ray after last timestamp
            ],
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        self.rays_d = torch.tensor([[0.0, 1.0, 0.0]], device=torch.device("cuda"), dtype=torch.float32).repeat(5, 1)

        self.rays_timestamps_us = torch.tensor(
            [
                50000,  # Before first timestamp
                100000,  # Exactly at first timestamp
                250000,  # Between timestamps
                400000,  # At last timestamp
                450000,  # After last timestamp
            ],
            device=torch.device("cuda"),
            dtype=torch.int64,
        )

    def test_ray_intersection_binary_search(self):
        # Test regular ray intersection
        intersections_cnt = self.tracks.ray_intersection(
            self.rays_o, self.rays_d, self.rays_timestamps_us, with_intersections_ts=False
        ).intersections_cnt

        # Verify intersection counts
        expected_counts = torch.tensor([0, 1, 1, 1, 0], device=torch.device("cuda"), dtype=torch.int32)
        self._compareTensor(intersections_cnt, expected_counts)

    def test_rolling_shutter_intersection_binary_search(self):
        # Test rolling shutter intersection
        pixel_idxs = torch.tensor([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], device="cuda:0", dtype=torch.int16)

        camera_rays = self.rays_d

        # Camera poses at start and end of frame
        camera_poses = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Start pose
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # End pose
            ],
            device="cuda:0",
            dtype=torch.float32,
        )

        # Frame timestamps spanning our test timestamps
        camera_timestamp_start_us = 50000
        camera_timestamp_end_us = 450000

        w, h = 5, 1  # Simple 5x1 image
        shutter_type = 2  # ROLLING_LEFT_TO_RIGHT

        intersections_cnt = self.tracks.ray_rolling_shutter_intersection(
            pixel_idxs,
            camera_rays,
            camera_poses,
            camera_timestamp_start_us,
            camera_timestamp_end_us,
            w,
            h,
            shutter_type,
        ).intersections_cnt

        # Verify intersection counts
        expected_counts = torch.tensor([0, 1, 1, 1, 0], device="cuda:0", dtype=torch.int32)
        self._compareTensor(intersections_cnt, expected_counts)

    def test_edge_cases(self):
        # Create an edge case track with only one timestamp
        with pytest.raises(AssertionError, match="Tracks: require at least two poses per track"):
            single_timestamp_track = CuboidTracks.Factory.from_numpy(
                tracks_id=["single"],
                tracks_poses=[np.eye(4, dtype=np.float32)[None, :, :]],
                tracks_timestamps_us=[np.array([200000], dtype=np.int64)],
                tracks_label_class=["test"],
                tracks_flags=[TrackFlags.NONE],
                cuboids_dims=[np.array([1.0, 1.0, 1.0], dtype=np.float32)],
            )

        # Test rays at different timestamps
        rays_o = torch.tensor([[0.0, -1.0, 0.0]], device="cuda:0", dtype=torch.float32).repeat(3, 1)
        rays_d = torch.tensor([[0.0, 1.0, 0.0]], device="cuda:0", dtype=torch.float32).repeat(3, 1)
        rays_timestamps_us = torch.tensor(
            [
                50000,  # Before timestamp
                400000,  # At last timestamp
                500000,  # After timestamp
            ],
            device="cuda:0",
            dtype=torch.int64,
        )

        # Test intersection
        intersections_cnt = self.tracks.ray_intersection(
            rays_o, rays_d, rays_timestamps_us, with_intersections_ts=False
        ).intersections_cnt

        # Should only intersect at exact timestamp
        expected_counts = torch.tensor([0, 1, 0], device="cuda:0", dtype=torch.int32)
        self._compareTensor(intersections_cnt, expected_counts)


class FramePosesInterpolationTest(CommonTestCase):
    def setUp(self):
        # Create a test track with known timestamps and poses
        tracks_id = ["test_track"]

        # Create poses with known translations for easy verification
        base_pose = np.eye(4, dtype=np.float32)
        poses = []
        for x in range(4):
            pose = base_pose.copy()
            pose[0, 3] = float(x)  # Moving along x-axis: 0->1->2->3
            poses.append(pose)
        tracks_poses = [np.stack(poses)]

        tracks_timestamps_us = [np.array([100000, 200000, 300000, 400000], dtype=np.int64)]
        tracks_label_class = ["test"]
        tracks_flags = [TrackFlags.NONE]
        cuboids_dims = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims
        )

    def test_frame_poses_interpolation_basic(self):
        test_cases = [
            # (start_ts, end_ts, expected_count, expected_start_x, expected_end_x)
            (50000, 150000, 0, None, None),  # Before first timestamp
            (50000, 400000, 0, None, None),  # At last timestamp
            (100000, 200000, 1, 0.0, 1.0),  # Exactly at timestamps
            (150000, 250000, 1, 0.5, 1.5),  # Between timestamps
            (150000, 400000, 1, 0.5, 3.0),  # At last timestamps
            (350000, 450000, 0, None, None),  # After last timestamp
            (200000, 300000, 1, 1.0, 2.0),  # Exactly at middle timestamps
            (100000, 400000, 1, 0.0, 3.0),  # Full range
        ]

        for start_ts, end_ts, expected_count, expected_start_x, expected_end_x in test_cases:
            frame_timestamps_us = torch.tensor([start_ts, end_ts], dtype=torch.int64, device=torch.device("cuda"))

            num_valid_tracks, track_ids, start_poses, end_poses = self.tracks.frame_poses_interpolation(
                frame_timestamps_us
            )

            # Check number of valid tracks
            self._compareTensor(
                num_valid_tracks, torch.tensor([expected_count], device=torch.device("cuda"), dtype=torch.int32)
            )

            if expected_count > 0:
                # Check track IDs
                self._compareTensor(track_ids, torch.tensor([[0, 0]], device=torch.device("cuda"), dtype=torch.int32))

                # Check interpolated poses
                # Start pose
                expected_start_pose = torch.tensor(
                    [expected_start_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda:0", dtype=torch.float32
                )
                self._compareTensor(start_poses[0], expected_start_pose)

                # End pose
                expected_end_pose = torch.tensor(
                    [expected_end_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda:0", dtype=torch.float32
                )
                self._compareTensor(end_poses[0], expected_end_pose)

    def test_frame_poses_interpolation_multiple_tracks(self):
        # Create tracks with multiple objects
        tracks_id = ["track1", "track2"]

        # First track moves along x-axis
        poses1 = []
        base_pose = np.eye(4, dtype=np.float32)
        for x in range(4):
            pose = base_pose.copy()
            pose[0, 3] = float(x)
            poses1.append(pose)

        # Second track moves along y-axis
        poses2 = []
        for y in range(4):
            pose = base_pose.copy()
            pose[1, 3] = float(y)
            poses2.append(pose)

        tracks_poses = [np.stack(poses1), np.stack(poses2)]
        tracks_timestamps_us = [
            np.array([100000, 200000, 300000, 400000], dtype=np.int64),
            np.array([100000, 200000, 300000, 400000], dtype=np.int64),
        ]
        tracks_label_class = ["test1", "test2"]
        tracks_flags = [TrackFlags.NONE, TrackFlags.NONE]
        cuboids_dims = [np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)]

        tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims
        )

        # Test interpolation at middle point
        frame_timestamps_us = torch.tensor([150000, 250000], dtype=torch.int64, device=torch.device("cuda"))

        num_valid_tracks, track_ids, start_poses, end_poses = tracks.frame_poses_interpolation(frame_timestamps_us)

        # Should have both tracks valid
        self._compareTensor(num_valid_tracks, torch.tensor([2], device=torch.device("cuda"), dtype=torch.int32))

        # Check track IDs
        expected_track_ids = torch.tensor([[0, 0], [1, 1]], device=torch.device("cuda"), dtype=torch.int32)
        self._compareTensor(track_ids, expected_track_ids)

        # Check interpolated poses
        # Track 1 (x-axis movement)
        expected_track1_start = torch.tensor(
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        expected_track1_end = torch.tensor(
            [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=torch.device("cuda"), dtype=torch.float32
        )

        # Track 2 (y-axis movement)
        expected_track2_start = torch.tensor(
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        expected_track2_end = torch.tensor(
            [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0], device=torch.device("cuda"), dtype=torch.float32
        )

        self._compareTensor(start_poses[0], expected_track1_start)
        self._compareTensor(end_poses[0], expected_track1_end)
        self._compareTensor(start_poses[1], expected_track2_start)
        self._compareTensor(end_poses[1], expected_track2_end)


class CleanTrackIdsTest(CommonTestCase):
    """Test suite for CuboidTracks.Ops.clean_track_ids functionality."""

    def setUp(self):
        # Create a track with IDs that have @suffixes (as seen in NCORE data)
        self.original_track_ids = [
            "15@scene:obstacles:autolabels:v2",
            "349@scene:obstacles:autolabels:v2",
            "42@scene:obstacles:autolabels:v2",
        ]
        self.expected_cleaned_ids = ["15", "349", "42"]

        tracks_poses: list[np.ndarray] = []
        tracks_timestamps_us: list[np.ndarray] = []
        tracks_label_class: list[str] = []
        tracks_flags: list[TrackFlags] = []
        cuboids_dims: list[np.ndarray] = []

        for i, track_id in enumerate(self.original_track_ids):
            # Create simple identity poses
            pose0 = np.eye(4, 4, dtype=np.float32)
            pose0[:3, 3] = [float(i), 0, 0]
            pose1 = pose0.copy()
            pose1[:3, 3] = [float(i) + 1, 0, 0]
            tracks_poses.append(np.stack([pose0, pose1]))
            tracks_timestamps_us.append(np.array([0, 1000000], np.int64))
            tracks_label_class.append("vehicle")
            tracks_flags.append(TrackFlags.DYNAMIC)
            cuboids_dims.append(np.array([2, 1, 1], dtype=np.float32))

        self.tracks = CuboidTracks.Factory.from_numpy(
            self.original_track_ids,
            tracks_poses,
            tracks_timestamps_us,
            tracks_label_class,
            tracks_flags,
            cuboids_dims=cuboids_dims,
        )

    def test_clean_track_ids_removes_suffixes(self):
        """Test that clean_track_ids removes @suffixes from track IDs."""
        cleaned_tracks = CuboidTracks.Ops.clean_track_ids(self.tracks, DataSourceSummary._clean_track_id_str)

        # Verify track IDs are cleaned
        self.assertEqual(cleaned_tracks.tracks_id, self.expected_cleaned_ids)

        # Verify original tracks are unchanged (immutability)
        self.assertEqual(self.tracks.tracks_id, self.original_track_ids)

    def test_clean_track_ids_preserves_other_data(self):
        """Test that clean_track_ids preserves all other track data."""
        cleaned_tracks = CuboidTracks.Ops.clean_track_ids(self.tracks, DataSourceSummary._clean_track_id_str)

        # Verify other data is preserved
        self._compareTensor(cleaned_tracks.tracks_packinfo, self.tracks.tracks_packinfo)
        self._compareTensor(cleaned_tracks.tracks_poses.data, self.tracks.tracks_poses.data)
        self._compareTensor(cleaned_tracks.tracks_timestamps_us, self.tracks.tracks_timestamps_us)
        self._compareTensor(cleaned_tracks.tracks_flags, self.tracks.tracks_flags)
        self.assertEqual(cleaned_tracks.max_track_n_poses, self.tracks.max_track_n_poses)
        self.assertEqual(cleaned_tracks.tracks_label_class, self.tracks.tracks_label_class)
        self._compareTensor(cleaned_tracks.cuboids_dims, self.tracks.cuboids_dims)

    def test_clean_track_ids_with_no_suffix(self):
        """Test that clean_track_ids works correctly when IDs have no suffixes."""
        # Create tracks with already clean IDs
        clean_ids = ["track_1", "track_2"]
        tracks_poses = [
            np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]),
            np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]),
        ]
        tracks_timestamps_us = [np.array([0, 1000], np.int64), np.array([0, 1000], np.int64)]
        tracks_label_class = ["car", "car"]
        tracks_flags = [TrackFlags.NONE, TrackFlags.NONE]
        cuboids_dims = [np.array([1, 1, 1], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32)]

        clean_tracks = CuboidTracks.Factory.from_numpy(
            clean_ids, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims=cuboids_dims
        )

        # Apply cleaning - should be a no-op
        result = CuboidTracks.Ops.clean_track_ids(clean_tracks, DataSourceSummary._clean_track_id_str)
        self.assertEqual(result.tracks_id, clean_ids)

    def test_clean_track_ids_with_empty_tracks(self):
        """Test that clean_track_ids works correctly with empty tracks."""
        empty_tracks = CuboidTracks.Factory.empty()
        result = CuboidTracks.Ops.clean_track_ids(empty_tracks, DataSourceSummary._clean_track_id_str)
        self.assertEqual(result.tracks_id, [])
        self.assertEqual(result.n_tracks, 0)

    def test_clean_track_ids_subset_lookup_after_cleaning(self):
        """Test that subset_from_tracks_id works correctly after cleaning track IDs.

        This tests the real-world scenario where cleaned track IDs are used to lookup
        tracks in a CuboidTracks that has also been cleaned.
        """
        cleaned_tracks = CuboidTracks.Ops.clean_track_ids(self.tracks, DataSourceSummary._clean_track_id_str)

        # This should work without raising ValueError
        subset = CuboidTracks.Ops.subset_from_tracks_id(cleaned_tracks, ["15", "42"])

        self.assertEqual(subset.tracks_id, ["15", "42"])
        self.assertEqual(subset.n_tracks, 2)


class InterpolateTracksPosesTest(CommonTestCase):
    def setUp(self):
        # Create two tracks: one with translation only, one with translation and rotation
        tracks_id = ["track_translation", "track_rotation"]

        # Track 1: translation along x-axis
        base_pose = np.eye(4, dtype=np.float32)
        poses1 = []
        for x in range(4):
            pose = base_pose.copy()
            pose[0, 3] = float(x)
            poses1.append(pose)
        poses1 = np.stack(poses1)

        # Track 2: rotation about z-axis (0, 90, 180, 270 degrees)
        poses2 = []
        for angle_deg in [0, 90, 180, 270]:
            pose = base_pose.copy()
            rot = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
            pose[:3, :3] = rot
            poses2.append(pose)
        poses2 = np.stack(poses2)

        tracks_poses = [poses1, poses2]
        tracks_timestamps_us = [
            np.array([100000, 200000, 300000, 400000], dtype=np.int64),
            np.array([100000, 200000, 300000, 400000], dtype=np.int64),
        ]
        tracks_label_class = ["translation", "rotation"]
        tracks_flags = [TrackFlags.NONE, TrackFlags.NONE]
        cuboids_dims = [np.array([1.0, 1.0, 1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)]

        self.tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims
        )

    def test_interpolate_tracks_poses_translation(self):
        # Test translation track at boundaries and interpolation
        track_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        timestamps = torch.tensor([100000, 250000, 400000], dtype=torch.int64, device="cuda")
        # 100000: at first pose, 250000: halfway (should be x=1.5), 400000: last pose
        expected_x = [0.0, 1.5, 3.0]
        poses = self.tracks.interpolate_tracks_poses(timestamps, track_idx.repeat(len(timestamps)))
        for i, x in enumerate(expected_x):
            self._compareTensor(
                poses[i].vec(), torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
            )

    def test_interpolate_tracks_poses_ex_translation(self):
        # Test translation track with out of bounds timestamps
        track_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        timestamps = torch.tensor([-100000, 100000, 250000, 400000, 10000000], dtype=torch.int64, device="cuda")
        # -100000 is out of bounds,: at first pose, 250000: halfway (should be x=1.5), 400000: last pose, 10000000: out of bounds
        expected_x = [0.0, 0.0, 1.5, 3.0, 0.0]
        expected_mask = [False, True, True, True, False]
        poses, mask = self.tracks.interpolate_tracks_poses_ex(
            timestamps,
            track_idx.repeat(len(timestamps)),
        )

        for i, x in enumerate(expected_x):
            if expected_mask[i]:
                self._compareTensor(
                    poses[i].vec(), torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
                )

            assert mask[i].item() == expected_mask[i]

    def test_interpolate_tracks_poses_rotation(self):
        # Test rotation track at boundaries and interpolation
        track_idx = torch.tensor([1], dtype=torch.int32, device="cuda")
        timestamps = torch.tensor([100000, 200000, 300000, 400000, 250000], dtype=torch.int64, device="cuda")
        # 100000: 0 deg, 200000: 90 deg, 300000: 180 deg, 400000: 270 deg, 250000: halfway (135 deg)

        expected_angles = [0, 90, 180, 270, 135]
        poses = self.tracks.interpolate_tracks_poses(timestamps, track_idx.repeat(len(timestamps)))
        for i, angle in enumerate(expected_angles):
            # Extract quaternion from pose
            quat = poses[i].vec()[3:].detach().cpu().numpy()
            # Convert to rotation matrix and get angle
            rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
            z_angle = rot.as_euler("zxy", degrees=True)[0]
            # Normalize angle to [0, 360)
            z_angle = (z_angle + 360) % 360
            expected = angle % 360
            # Allow a small tolerance due to quaternion interpolation
            self.assertTrue(np.isclose(z_angle, expected, atol=1.0), f"Expected {expected}, got {z_angle}")

    def test_interpolate_tracks_poses_multiple(self):
        # Test batch interpolation for both tracks at the same timestamp
        timestamps = torch.tensor([150000, 350000], dtype=torch.int64, device="cuda")
        tracks_idx = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        poses = self.tracks.interpolate_tracks_poses(timestamps, tracks_idx)
        # Track 0: x = 0.5, Track 1: rotation = 225 deg
        self._compareTensor(
            poses[0].vec(), torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
        )

        # For rotation, check angle
        quat = poses[1].vec()[3:].detach().cpu().numpy()
        rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
        z_angle = rot.as_euler("zxy", degrees=True)[0]
        z_angle = (z_angle + 360) % 360
        self.assertTrue(np.isclose(z_angle, 225, atol=1.0), f"Expected 225, got {z_angle}")

    def test_interpolate_tracks_poses_zero_gap(self):
        # Create a track with two poses at the same timestamp
        tracks_id = ["zero_gap_track"]
        base_pose = np.eye(4, dtype=np.float32)
        pose1 = base_pose.copy()
        pose1[0, 3] = 1.0
        pose2 = base_pose.copy()
        pose2[0, 3] = 2.0
        tracks_poses = [np.stack([pose1, pose2])]
        tracks_timestamps_us = [np.array([100000, 100000], dtype=np.int64)]  # zero gap
        tracks_label_class = ["zero_gap"]
        tracks_flags = [TrackFlags.NONE]
        cuboids_dims = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]

        tracks = CuboidTracks.Factory.from_numpy(
            tracks_id, tracks_poses, tracks_timestamps_us, tracks_label_class, tracks_flags, cuboids_dims
        )
        # Interpolate at the timestamp
        timestamps = torch.tensor([100000], dtype=torch.int64, device="cuda")
        track_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        poses = tracks.interpolate_tracks_poses(timestamps, track_idx)
        # Should return the first pose (alpha=0)
        self._compareTensor(
            poses[0].vec(), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
        )
