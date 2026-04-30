# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import random
import unittest

from collections import defaultdict
from unittest.mock import Mock

import numpy as np
import torch

from PIL import Image
from scipy import ndimage

from ncore.data import BBox3
from nre.config import ValidPixelsFrameMaskConfig, ValidPixelsSceneFlowConfig, ValidPixelsTrafficLightConfig
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.datasets.utils import (
    PackedMask,
    compute_cameras_valid_pixels_frame_mask,
    compute_point_cloud_inside_tracks_mask,
    compute_points_outside_tracks,
    compute_valid_lidarpoints_trafficlight_cameravisible,
    compute_valid_pixels_ego,
    compute_valid_pixels_sceneflow,
    compute_valid_pixels_trafficlight,
)
from nre.utils.types import HalfClosedInterval, PointCloud


class TestPackedMask(unittest.TestCase):
    def test_random(self):
        # make sure that random masks are packed / unpacked correctly
        NUM_RUNS = 20
        MAX_DIMS = 4
        MAX_SIZE = 5
        for n in range(NUM_RUNS):
            for d in range(MAX_DIMS):  # number of dimensions
                dims = [random.randrange(MAX_SIZE) for _ in range(d)]

                mask_ref = np.random.choice([True, False], size=dims)

                packed_mask = PackedMask(mask_ref)

                np.testing.assert_array_equal(mask_ref, packed_mask.unpacked())


class TestComputeFunctions(unittest.TestCase):
    def generate_mask(self):
        self.camera_name = "dummy_camera_name"
        self.lidar_name = "dummy_lidar_name"
        self.dims = (16, 20)
        self.num_lidar_points = 200
        self.mask_ref = np.random.choice([True, False], size=self.dims)
        self.packed_mask = PackedMask(self.mask_ref)
        self.num_frames = 2

    def test_compute_valid_pixels_ego(self):
        self.generate_mask()

        masks = compute_valid_pixels_ego(
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            cameras_valid_pixels_ego_masks={self.camera_name: self.packed_mask},
        )

        assert len(masks[self.camera_name]) == self.num_frames

        for i in range(self.num_frames):
            np.testing.assert_array_equal(self.mask_ref, masks[self.camera_name][i].unpacked())

    def test_compute_cameras_valid_pixels_frame_mask(self):
        self.generate_mask()

        base_mask = np.random.choice([True, False], size=self.dims)
        tree_mask = np.random.choice([True, False], size=self.dims)
        n_dilation_iterations = 2
        config = ValidPixelsFrameMaskConfig(
            n_dilation_iterations=n_dilation_iterations,
            classes=["road", "trees"],
        )

        def get_frame_generic_data(continuous_frame_index: int, name: str) -> np.ndarray:
            if name == "trees":
                return tree_mask
            return base_mask

        camera_sensor = Mock()
        camera_sensor.get_frame_generic_data = get_frame_generic_data

        masks = compute_valid_pixels_ego(
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            cameras_valid_pixels_ego_masks={self.camera_name: self.packed_mask},
        )

        masks = compute_cameras_valid_pixels_frame_mask(
            camera_sensors={self.camera_name: camera_sensor},
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            valid_pixels_frame_mask_params=config,
            cameras_frame_valid_pixels_masks=masks,
            tqdm_disabled=False,
        )

        # combine & dilate masks
        include_mask = np.logical_or(base_mask, tree_mask)
        exclude_mask = ndimage.binary_dilation(
            np.logical_not(include_mask),
            iterations=n_dilation_iterations,
        )
        include_mask = np.logical_not(exclude_mask)

        for i in range(self.num_frames):
            np.testing.assert_array_equal(self.mask_ref & include_mask, masks[self.camera_name][i].unpacked())

    def test_compute_valid_pixels_sceneflow(self):
        self.generate_mask()

        masks = compute_valid_pixels_ego(
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            cameras_valid_pixels_ego_masks={self.camera_name: self.packed_mask},
        )

        flow = np.zeros(self.dims, np.float32)
        flow[1:3, 2:5] = 5.0
        # This area will be dilated and undilated
        # 0 0 0 0 0 0 0 ...
        # 0 0 1 1 1 0 0 ...
        # 0 0 1 1 1 0 0 ...
        # 0 0 0 0 0 0 0 ...
        # After that it changes to this:
        mask_copy = self.mask_ref.copy()
        mask_copy[2:4, 2:6] = False
        # 0 0 0 0 0 0 0 ...
        # 0 0 0 0 0 0 0 ...
        # 0 0 1 1 1 1 0 ...
        # 0 0 1 1 1 1 0 ...
        # 0 0 0 0 0 0 0 ...

        aux_loader = Mock()
        aux_loader.get_scene_flow_magnitude = lambda camera_id, timestamp: flow
        camera_sensor = Mock()
        camera_sensor.get_frame_timestamp_us = lambda x: x * 33333
        camera_model = Mock()
        camera_model.resolution = torch.tensor([self.dims[1], self.dims[0]], dtype=torch.int32)
        frame_range = range(self.num_frames)
        valid_pixels_scene_flow_config = ValidPixelsSceneFlowConfig(
            flow_min_speed_ms=0.1,
            flow_dilate_radius=1,
            flow_downsample_scale=2,
        )

        masks = compute_valid_pixels_sceneflow(
            aux_loader,
            camera_sensors={self.camera_name: camera_sensor},
            camera_models={self.camera_name: camera_model},
            camera_frame_ranges={self.camera_name: frame_range},
            valid_pixels_scene_flow_config=valid_pixels_scene_flow_config,
            cameras_frame_valid_pixels_masks=masks,
        )

        for i in range(self.num_frames):
            np.testing.assert_array_equal(mask_copy, masks[self.camera_name][i].unpacked())

    def test_compute_valid_pixels_trafficlight(self):
        self.generate_mask()

        masks = compute_valid_pixels_ego(
            camera_frame_ranges={self.camera_name: range(self.num_frames)},
            cameras_valid_pixels_ego_masks={self.camera_name: self.packed_mask},
        )

        class_ids = [0, 16, 20]
        trafficlight_class_id = 16
        semantic_segmentation = np.random.choice(class_ids, size=self.dims)

        aux_loader = Mock()
        # Semantic Segmentation is a PILImage.Image of class ids
        aux_loader.get_semantic_segmentation = lambda c, t: Image.fromarray(semantic_segmentation.astype(np.uint8))

        camera_sensor = Mock()
        camera_sensor.get_frame_timestamp_us = lambda frame_idx: 33333 * frame_idx
        camera_frame_range = range(self.num_frames)
        valid_pixels_traffic_light_params = ValidPixelsTrafficLightConfig(seg_dilate_radius=1)

        masks = compute_valid_pixels_trafficlight(
            aux_loader,
            camera_sensors={self.camera_name: camera_sensor},
            camera_frame_ranges={self.camera_name: camera_frame_range},
            valid_pixels_traffic_light_params=valid_pixels_traffic_light_params,
            sensor_trafficlight_class_ids={self.camera_name: trafficlight_class_id},
            cameras_frame_valid_pixels_masks=masks,
            tqdm_disabled=False,
        )

        for i in range(self.num_frames):
            light_mask = semantic_segmentation == trafficlight_class_id
            assert np.all(np.logical_not(masks[self.camera_name][i].unpacked()[light_mask]))

    def test_compute_valid_lidarpoints_trafficlight_cameravisible(self):
        self.generate_mask()

        class_ids = [0, 16, 20]
        trafficlight_class_id = 16
        semantic_segmentation = np.random.choice(class_ids, size=(self.num_lidar_points,))
        visibility_mask = np.random.choice([True, False], size=(self.num_lidar_points,))

        aux_loader = Mock()
        aux_loader.get_lidar_semantic_segmentation = lambda lidar_id, frame_timestamps_us: semantic_segmentation
        aux_loader.get_lidar_camera_visibility = lambda lidar_id, frame_timestamps_us, camera_ids: {
            self.camera_name: visibility_mask
        }

        all_camera_ids = [self.camera_name]

        camera_sensor = Mock()
        camera_sensor.get_frame_timestamp_us = lambda frame_idx: 33333 * frame_idx

        lidar_sensor = Mock()
        lidar_sensor.get_frames_timestamps_us = lambda: np.array(
            [frame_idx * 100000 for frame_idx in range(self.num_frames)]
        )
        lidar_sensor.get_frame_timestamp = lambda frame_idx: 100000 * frame_idx

        time_range_us = HalfClosedInterval(0, 100000)

        masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)

        masks = compute_valid_lidarpoints_trafficlight_cameravisible(
            aux_loader,
            lidar_sensors={self.lidar_name: lidar_sensor},
            all_camera_ids=all_camera_ids,
            time_range_us=time_range_us,
            sensor_trafficlight_class_ids={self.lidar_name: trafficlight_class_id},
            lidars_frame_valid_points_masks=masks,
            tqdm_disabled=False,
        )

        assert len(masks[self.lidar_name]) == self.num_frames

        for i in range(self.num_frames):
            expected_result = np.logical_and(semantic_segmentation != trafficlight_class_id, visibility_mask)
            np.testing.assert_array_equal(expected_result, masks[self.lidar_name][i].unpacked())

    def test_compute_points_outside_tracks(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Case 1: all points outside (zero-sized box)
        tracks_all_outside = [BBox3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
        with self.subTest("all_outside"):
            outside_mask, cuboid_T, track_dim = compute_points_outside_tracks(
                points,
                tracks_all_outside,
                [0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(outside_mask.cpu().numpy(), np.array([True, True, True, True]))
            assert cuboid_T.shape == (1, 4, 4) or cuboid_T.shape == (4, 4)
            np.testing.assert_allclose(track_dim.cpu().numpy(), np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

        # Case 2: all points inside (large box)
        tracks_all_inside = [BBox3((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (0.0, 0.0, 0.0))]
        with self.subTest("all_inside"):
            outside_mask, cuboid_T, track_dim = compute_points_outside_tracks(
                points,
                tracks_all_inside,
                [0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(outside_mask.cpu().numpy(), np.array([False, False, False, False]))
            assert cuboid_T.shape == (1, 4, 4) or cuboid_T.shape == (4, 4)
            np.testing.assert_allclose(track_dim.cpu().numpy(), np.array([[10.0, 10.0, 10.0]], dtype=np.float32))

        # Case 3: some points outside (box size 2, half-extent 1)
        tracks_some_outside = [BBox3((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))]
        with self.subTest("some_outside"):
            outside_mask, cuboid_T, track_dim = compute_points_outside_tracks(
                points,
                tracks_some_outside,
                [0.0, 0.0, 0.0],
            )
            np.testing.assert_array_equal(outside_mask.cpu().numpy(), np.array([False, False, True, True]))
            assert cuboid_T.shape == (1, 4, 4) or cuboid_T.shape == (4, 4)
            np.testing.assert_allclose(track_dim.cpu().numpy(), np.array([[2.0, 2.0, 2.0]], dtype=np.float32))

        # Padding increases box; with extent 3 (half-extent 1.5), x=2 remains outside and x=3 remains outside
        with self.subTest("some_outside_with_padding"):
            outside_mask_padded, _, track_dim_padded = compute_points_outside_tracks(
                points,
                tracks_some_outside,
                [3.0, 3.0, 3.0],
            )
            np.testing.assert_array_equal(outside_mask_padded.cpu().numpy(), np.array([False, False, False, True]))
            np.testing.assert_allclose(track_dim_padded.cpu().numpy(), np.array([[5.0, 5.0, 5.0]], dtype=np.float32))

    def test_compute_point_cloud_inside_tracks_mask(self):
        # Build a simple point cloud on CUDA to match implementation device
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        pc = PointCloud(xyz_start=points.clone(), xyz_end=points.clone())

        time_range_us = HalfClosedInterval(0, 200000)

        def make_cuboid_tracks(
            center_xyz: tuple[float, float, float], dims_lwh: tuple[float, float, float]
        ) -> CuboidTracks:
            # Two identical poses to satisfy Tracks.Factory requirement (>=2 poses per track)
            T = np.eye(4, dtype=np.float32)
            T[:3, 3] = np.array(center_xyz, dtype=np.float32)
            poses = np.stack([T, T], axis=0).astype(np.float32)
            timestamps = np.array([0, 100000], dtype=np.int64)
            return CuboidTracks.Factory.from_numpy(
                tracks_id=["track_0"],
                tracks_poses=[poses],
                tracks_timestamps_us=[timestamps],
                tracks_label_class=["car"],
                tracks_flags=[TrackFlags.DYNAMIC],
                cuboids_dims=[np.array(dims_lwh, dtype=np.float32)],
            )

        # Case 1: all points outside tracks => all False
        with self.subTest("all_outside"):
            cuboidtracks_dynamic = make_cuboid_tracks(center_xyz=(10.0, 0.0, 0.0), dims_lwh=(1.0, 1.0, 1.0))
            mask = compute_point_cloud_inside_tracks_mask(
                pc,
                time_range_us=time_range_us,
                cuboidtracks_dynamic=cuboidtracks_dynamic,
                track_padding_m=[0.0, 0.0, 0.0],
                batch_size=1,
                tqdm_disabled=True,
            )
            torch.testing.assert_close(
                mask, torch.tensor([False, False, False, False], dtype=torch.bool, device="cuda")
            )

        # Case 2: all points inside tracks => all True
        with self.subTest("all_inside"):
            cuboidtracks_dynamic = make_cuboid_tracks(center_xyz=(1.5, 0.0, 0.0), dims_lwh=(10.0, 10.0, 10.0))
            mask = compute_point_cloud_inside_tracks_mask(
                pc,
                time_range_us=time_range_us,
                cuboidtracks_dynamic=cuboidtracks_dynamic,
                track_padding_m=[0.0, 0.0, 0.0],
                batch_size=1,
                tqdm_disabled=True,
            )
            torch.testing.assert_close(mask, torch.tensor([True, True, True, True], dtype=torch.bool, device="cuda"))

        # Case 3: mixed outside/inside => some False, some True
        with self.subTest("some_inside"):
            # Box of length 2 centered at 0.5 covers x in [-0.5, 1.5] -> points 0 and 1 inside
            cuboidtracks_dynamic = make_cuboid_tracks(center_xyz=(0.5, 0.0, 0.0), dims_lwh=(2.0, 2.0, 2.0))
            mask = compute_point_cloud_inside_tracks_mask(
                pc,
                time_range_us=time_range_us,
                cuboidtracks_dynamic=cuboidtracks_dynamic,
                track_padding_m=[0.0, 0.0, 0.0],
                batch_size=1,
                tqdm_disabled=True,
            )
            torch.testing.assert_close(mask, torch.tensor([True, True, False, False], dtype=torch.bool, device="cuda"))

        # Case 4: semantic class restriction - only test specified classes
        with self.subTest("semantic_class_restriction"):
            # Create point cloud with semantic classes: [0=dynamic, 1=static, 0=dynamic, 1=static]
            semantic_class_id = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
            pc_semantic = PointCloud(
                xyz_start=points.clone(), xyz_end=points.clone(), semantic_class_id=semantic_class_id
            )
            # All points are geometrically inside the track
            cuboidtracks_dynamic = make_cuboid_tracks(center_xyz=(1.5, 0.0, 0.0), dims_lwh=(10.0, 10.0, 10.0))
            mask = compute_point_cloud_inside_tracks_mask(
                pc_semantic,
                time_range_us=time_range_us,
                cuboidtracks_dynamic=cuboidtracks_dynamic,
                track_padding_m=[0.0, 0.0, 0.0],
                batch_size=1,
                tqdm_disabled=True,
                restrict_to_class_ids=[0],  # Only test class 0 (dynamic)
            )
            # Only points with class 0 are tested and marked as inside; class 1 always False
            torch.testing.assert_close(mask, torch.tensor([True, False, True, False], dtype=torch.bool, device="cuda"))
