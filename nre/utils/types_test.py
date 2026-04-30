# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
import json
import tempfile
import unittest

from typing import Optional, get_args

import hydra
import lietorch as lt
import numpy as np
import parameterized
import torch

from python.runfiles import runfiles
from scipy.spatial.transform import Rotation

import ncore.data

from ncore.data import ConcreteCameraModelParametersUnion
from nre.config.parse import NREConfig
from nre.datasets import make as make_dataset
from nre.datasets.ncore import NCOREDataSource
from nre.utils.tests import CommonTestCase
from nre.utils.types import (
    AABB3D,
    FrameConversion,
    HalfClosedInterval,
    RigTrajectories,
    SceneContractor,
    TracksData,
)


RUNFILES = runfiles.Create()


class TestSerialization(unittest.TestCase):
    """Test to re-serialization of different datatypes"""

    def setUp(self):
        # load ncore dataset from dedicated config

        # initialize config to load dataset from test-data
        with hydra.initialize(version_base=None, config_path="../../configs"):
            dataset_path = RUNFILES.Rlocation(
                "test_data_ncore/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.json"
            )

            # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
            dataset_path_str = '"{0}"'.format(str(dataset_path))

            cfg = hydra.compose(
                config_name="tests/ncore_ds",
                overrides=[
                    f"dataset.path={dataset_path_str}",
                    # faster initialization
                    "dataset.valid_measurements_method=EGO",
                ],
            )
            self.cfg = NREConfig.model_validate(cfg)

        # load the ncore data source
        self.datasource = make_dataset(self.cfg.dataset.name, self.cfg, split="train").get_datasource()

    @parameterized.parameterized.expand(itertools.product((False, True)))
    def test_rigtrajectories(self, add_rig_bbox: bool):
        """Test re-serialization of FrameConversion / RigTrajectories"""

        assert isinstance(self.datasource, NCOREDataSource)

        rig_trajectories_ref = self.datasource.get_rig_trajectories()

        if add_rig_bbox:
            # add a bbox to test (re-)serialization as the original data doesn't provide one
            rig_trajectories_ref.rig_trajectories[0].rig_bbox = ncore.data.BBox3.from_array(np.random.rand(9))

        # verify that sensor frame end timestamps are covered by T_rig_world_timestamps_us
        for rig_trajectory in rig_trajectories_ref.rig_trajectories:
            T_rig_world_time_range = HalfClosedInterval.from_series(rig_trajectory.T_rig_world_timestamps_us)

            for sensor_name, sensor_timestamps in {
                **rig_trajectory.cameras_frame_timestamps_us,
                **rig_trajectory.lidars_frame_timestamps_us,
            }.items():
                sensor_frame_ends = sensor_timestamps[:, 1]
                sensor_frame_end_time_range = HalfClosedInterval.from_series(sensor_frame_ends)

                self.assertTrue(
                    sensor_frame_end_time_range in T_rig_world_time_range,
                    msg=(
                        f"Rig-world calibration time range ({T_rig_world_time_range}) does not cover the sensor frame end "
                        f"timestamp range ({sensor_frame_end_time_range}) for sensor {sensor_name}."
                    ),
                )

        # check we actually loaded some non-trivial trajectory
        self.assertGreater(len(rig_trajectories_ref.rig_trajectories), 0)
        self.assertGreater(len(rig_trajectories_ref.camera_calibrations), 0)
        self.assertGreater(len(rig_trajectories_ref.lidar_calibrations), 0)

        # serialize to file
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            with open(tmp_file.name, "w") as outfile:
                json.dump(rig_trajectories_ref.to_dict(), outfile, indent=4, sort_keys=True)

            with open(tmp_file.name, "r") as infile:
                rig_trajectories_reload_dict = json.load(infile)

        # test de-serialization
        rig_trajectories_reload = RigTrajectories.from_dict(rig_trajectories_reload_dict)

        # make sure original / de-serialized data is the same
        self.assertDictEqual(rig_trajectories_ref.to_dict(), rig_trajectories_reload.to_dict())

        # make sure types are correctly deserialized
        self.assertTrue(
            type(next(iter(rig_trajectories_reload.camera_calibrations.values())).camera_model_parameters)
            in get_args(ConcreteCameraModelParametersUnion),
        )


class TestSeekOffset(TestSerialization):
    """Same as TestSerialization but with seek_offset_sec settings"""

    def setUp(self):
        # load ncore dataset from dedicated config

        # initialize config to load dataset from test-data
        with hydra.initialize(version_base=None, config_path="../../configs"):
            dataset_path = RUNFILES.Rlocation(
                "test_data_ncore/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.json"
            )

            # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
            dataset_path_str = '"{0}"'.format(str(dataset_path))

            cfg = hydra.compose(
                config_name="tests/ncore_ds",
                overrides=[
                    f"dataset.path={dataset_path_str}",
                    "dataset.seek_offset_sec=0.15",
                    # faster initialization
                    "dataset.valid_measurements_method=EGO",
                ],
            )
            self.cfg = NREConfig.model_validate(cfg)

        # load the ncore data source
        self.datasource = make_dataset(self.cfg.dataset.name, self.cfg, split="train").get_datasource()


class TestFrameConversion(unittest.TestCase):
    def setUp(self):
        rotation_vector = np.array([0.1, 0.2, 0.3])
        self.R = Rotation.from_rotvec(rotation_vector).as_matrix()
        self.o = np.array([1.0, 2.0, 4.0])
        self.s = 0.02

    def _make_fc(self, dtype):
        """Build a FrameConversion at self.R / self.o / self.s with the given matrix dtype."""
        matrix = np.eye(4, dtype=dtype)
        matrix[:3, :3] = self.R
        matrix[:3, 3] = -self.o
        matrix[3, 3] = 1 / self.s
        return FrameConversion(matrix=matrix)

    @parameterized.parameterized.expand([(np.float32,), (np.float64,)])
    def test_transform_points(self, dtype):
        fc = self._make_fc(dtype)
        # Test forward transform
        src_points = np.array([[1.3, 3.2, 1.2], [-0.0, 1.0, -3.0]], dtype=dtype)
        tgt_points = fc.transform_points(src_points)
        exp_tgt_points = (src_points @ self.R.T - self.o.T) * self.s
        np.testing.assert_array_almost_equal(tgt_points, exp_tgt_points, decimal=6)

        # Test inverse transform
        exp_src_points = fc.inverse().transform_points(tgt_points)
        np.testing.assert_array_almost_equal(src_points, exp_src_points, decimal=5)

    @parameterized.parameterized.expand([(np.float32,), (np.float64,)])
    def test_transform_poses(self, dtype):
        fc = self._make_fc(dtype)
        R = Rotation.from_rotvec(np.array([0.2, -0.2, 0.5])).as_matrix()
        src_pose = np.eye(4, dtype=dtype)
        src_pose[:3, :3] = R
        src_pose[:3, 3] = np.array([2.0, -10, 3])
        exp_src_pose = fc.inverse().transform_poses(fc.transform_poses(src_pose))
        np.testing.assert_array_almost_equal(src_pose, exp_src_pose, decimal=5)

    @parameterized.parameterized.expand([(np.float32,), (np.float64,)])
    def test_dtype_property(self, dtype):
        """FrameConversion.dtype mirrors FrameConversion.matrix.dtype."""
        fc = self._make_fc(dtype)
        self.assertEqual(fc.dtype, np.dtype(dtype))
        self.assertEqual(fc.matrix.dtype, fc.dtype)

    def test_rejects_non_floating_dtype(self):
        """__post_init__ rejects non-floating matrix dtypes with a TypeError."""
        with self.assertRaises(TypeError):
            FrameConversion(matrix=np.eye(4, dtype=np.int32))

    @parameterized.parameterized.expand([(np.float32,), (np.float64,)])
    def test_get_transformation_matrices_dtype(self, dtype):
        """T and S returned by get_transformation_matrices match fc.dtype for both configured dtypes."""
        fc = self._make_fc(dtype)
        T, S = fc.get_transformation_matrices()
        self.assertEqual(T.dtype, fc.dtype)
        self.assertEqual(S.dtype, fc.dtype)

    @parameterized.parameterized.expand([(np.float32,), (np.float64,)])
    def test_inverse_preserves_dtype(self, dtype):
        """inverse() returns a FrameConversion with the same dtype as self."""
        fc = self._make_fc(dtype)
        inv = fc.inverse()
        self.assertEqual(inv.dtype, fc.dtype)
        self.assertEqual(inv.matrix.dtype, fc.dtype)

    @parameterized.parameterized.expand(list(itertools.product([np.float32, np.float64], [np.float32, np.float64])))
    def test_transform_poses_dtype_roundtrip(self, fc_dtype, input_dtype):
        """transform_poses must always return fc.dtype regardless of input dtype.

        This is the class's declared output-dtype contract (see FrameConversion.matrix docstring).

        Regression test for a latent dtype bug: when a downstream NCore dataset stores poses as
        float64 (allowed by the NCore v4 writer), PoseGraphInterpolator.evaluate_poses returns
        float64 poses that flow into FrameConversion.transform_poses. Previously numpy silently
        promoted the result to float64, which crashed downstream torch matmuls against float32
        LiDAR tensors with 'expected mat1 and mat2 to have the same dtype, but got: double != float'.
        """
        fc = self._make_fc(fc_dtype)

        # Single pose.
        src_pose = np.eye(4, dtype=input_dtype)
        src_pose[:3, 3] = np.array([2.0, -10.0, 3.0])
        out = fc.transform_poses(src_pose)
        self.assertEqual(out.dtype, fc.dtype)
        self.assertEqual(out.shape, (4, 4))

        # Batched input.
        src_poses = np.stack([src_pose, src_pose @ src_pose], axis=0)
        out_batched = fc.transform_poses(src_poses)
        self.assertEqual(out_batched.dtype, fc.dtype)
        self.assertEqual(out_batched.shape, (2, 4, 4))

    @parameterized.parameterized.expand(list(itertools.product([np.float32, np.float64], [np.float32, np.float64])))
    def test_transform_points_dtype_roundtrip(self, fc_dtype, input_dtype):
        """transform_points must always return fc.dtype regardless of input dtype."""
        fc = self._make_fc(fc_dtype)
        src_points = np.array([[1.3, 3.2, 1.2], [-0.0, 1.0, -3.0]], dtype=input_dtype)

        out = fc.transform_points(src_points)
        self.assertEqual(out.dtype, fc.dtype)
        self.assertEqual(out.shape, (2, 3))

        # Singular (3,) input.
        out_single = fc.transform_points(src_points[0])
        self.assertEqual(out_single.dtype, fc.dtype)
        self.assertEqual(out_single.shape, (3,))

    def test_float64_preserves_ecef_scale_precision(self):
        """Constructing an f64 FrameConversion preserves sub-micrometer round-trip precision at ECEF scale.

        Justifies the dtype knob: an ECEF-like source point (~earth radius, 6.378e6 m) held as f64
        and round-tripped through transform_points -> inverse().transform_points loses decimeters
        under an f32 FrameConversion (because the input is silently truncated to f32 at the ULP of
        2^-1 = 0.5 m at this magnitude) but stays sub-micrometer under an f64 FrameConversion. The
        NCore v4 writer accepts f64 poses, so future converters may emit them and this knob lets
        NuRec consume them end-to-end without losing precision.
        """
        ecef_origin = np.array([6.378e6, 1.2e5, 0.0])
        # True source point in f64, offset by a sub-metre amount that is below the f32 ULP at this magnitude.
        src_point_f64 = np.array([[ecef_origin[0] + 1.234, ecef_origin[1] + 5.678, 9.012]])

        def build(dtype):
            matrix = np.eye(4, dtype=dtype)
            matrix[:3, 3] = -ecef_origin
            return FrameConversion(matrix=matrix)

        fc32 = build(np.float32)
        fc64 = build(np.float64)

        # Feed the same f64 source into both pipelines. f32 fc truncates the input to f32 on entry.
        roundtrip32 = fc32.inverse().transform_points(fc32.transform_points(src_point_f64))
        roundtrip64 = fc64.inverse().transform_points(fc64.transform_points(src_point_f64))

        # Dtype contract: output dtype always matches fc.dtype.
        self.assertEqual(roundtrip32.dtype, np.dtype(np.float32))
        self.assertEqual(roundtrip64.dtype, np.dtype(np.float64))

        # Compare against the f64 ground truth (promote f32 result to f64 for the comparison).
        err32 = float(np.abs(roundtrip32.astype(np.float64) - src_point_f64).max())
        err64 = float(np.abs(roundtrip64 - src_point_f64).max())

        # Guard the test's own validity: f32 must actually lose precision here (~0.25 m is the expected
        # ULP at 6.378e6), otherwise the test would be vacuously true.
        self.assertGreater(err32, 1e-2, "float32 round-trip at ECEF scale should lose decimetres")
        self.assertLess(err64, 1e-6, "float64 round-trip at ECEF scale should stay sub-micrometer")


class TestSceneContractor(CommonTestCase):
    def setUp(self):
        self.aabb = AABB3D(torch.FloatTensor([2, 4, 3]), torch.FloatTensor([6, 5, 8]))

    @parameterized.parameterized.expand(
        [
            (None, False),
            (2.0, False),
            (float("inf"), False),
            (float("inf"), True),
        ]
    )
    def test_round_trip(self, degree: Optional[float], is_merf: bool):
        scene_contractor = SceneContractor(degree, self.aabb, is_merf=is_merf)
        points = 20 * torch.rand(100, 3) - 10  # points in [-10, 10]
        contracted_points = scene_contractor.to_contracted_space(points)
        if degree is not None:
            self.assertLessEqual(contracted_points.abs().max().item(), 1)

        points_round_trip = scene_contractor.from_contracted_space(contracted_points)
        self._compareTensor(points, points_round_trip, decimal=4)
        self._compareTensor(contracted_points, scene_contractor.to_contracted_space(points_round_trip), decimal=4)


class TestTracksData(CommonTestCase):
    def test_empty_tracks_data(self):
        """Check if TracksData.empty() creates a valid object (i.e. __post_init__ does not raise an error)"""
        tracks_data = TracksData.empty(device=torch.device("cuda"))

    def test_tracks_data_post_init(self):
        """Check if TracksData.__post_init__ validates data"""
        # valid tracksdata, shouldn't raise
        _ = TracksData(
            tracks_id=["track1", "track2"],
            tracks_label_class=["car", "car"],
            max_track_n_poses=1,
            tracks_packinfo=torch.tensor([[0, 1], [1, 1]], dtype=torch.int32, device=torch.device("cuda")),
            tracks_poses=lt.SE3.Identity(2, device=torch.device("cuda")),
            tracks_timestamps_us=torch.tensor([1, 2], dtype=torch.int64, device=torch.device("cuda")),
            tracks_flags=torch.tensor([1, 2], dtype=torch.int32, device=torch.device("cuda")),
        )

        # invalid tracks_packinfo
        with self.assertRaises(ValueError):
            TracksData(
                tracks_id=["track1", "track2"],
                tracks_label_class=["car", "car"],
                max_track_n_poses=1,
                tracks_packinfo=torch.tensor([[0, 1]], dtype=torch.int32, device=torch.device("cuda")),
                tracks_poses=lt.SE3.Identity(2, device=torch.device("cuda")),
                tracks_timestamps_us=torch.tensor([1, 2], dtype=torch.int64, device=torch.device("cuda")),
                tracks_flags=torch.tensor([1, 2], dtype=torch.int32, device=torch.device("cuda")),
            )
