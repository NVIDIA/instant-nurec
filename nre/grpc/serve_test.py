# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import contextlib
import unittest

from concurrent import futures
from unittest import mock

import grpc
import numpy as np
import torch

import ncore.data

from nre.grpc.grpc_server_config import GrpcServerConfig
from nre.grpc.protos.common_pb2 import Empty, Pose, PoseAtTime
from nre.grpc.protos.common_pb2 import Trajectory as ProtoTrajectory
from nre.grpc.protos.sensorsim_pb2 import (
    BatchRGBRenderRequest,
    BatchRGBRenderRequestItem,
    CameraSpec,
    DynamicObjectTrack,
    FthetaCameraParam,
    LidarRenderFilter,
    LidarRenderRequest,
    LidarSpec,
    LinearCde,
    OpenCVFisheyeCameraParam,
    OpenCVPinholeCameraParam,
    RGBRenderRequest,
)
from nre.grpc.protos.sensorsim_pb2 import BivariateWindshieldModelParameters as ProtoWParams
from nre.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub, add_SensorsimServiceServicer_to_server
from nre.grpc.serve import (
    Backend,
    CameraBank,
    ModelIncompatibilityError,
    SceneDownloadInterceptor,
    SensorSimService,
    TimestampOutOfRangeError,
    _classify_render_exception,
    _is_local_scene_uri,
    actor_tracks_from_grpc,
    actor_tracks_to_grpc,
)


class TestCamera(unittest.TestCase):
    def test_camera_spec(self):
        camera_spec = CameraSpec()
        camera_spec.logical_id = "camera1"
        camera_spec.resolution_h = 567
        camera_spec.resolution_w = 320
        camera_spec.shutter_type = 1

        fthetaCameraParam = FthetaCameraParam()
        fthetaCameraParam.principal_point_x = 0
        fthetaCameraParam.principal_point_y = 1
        fthetaCameraParam.reference_poly = 1
        fthetaCameraParam.pixeldist_to_angle_poly.extend([1, 2, 3, 4, 5])
        fthetaCameraParam.angle_to_pixeldist_poly.extend([0, 0, 0, 0, 0])
        fthetaCameraParam.max_angle = 120
        fthetaCameraParam.linear_cde.linear_c = 1
        fthetaCameraParam.linear_cde.linear_d = 2
        fthetaCameraParam.linear_cde.linear_e = 3

        camera_spec.ftheta_param.CopyFrom(fthetaCameraParam)

        np.testing.assert_equal(camera_spec.logical_id, "camera1")
        np.testing.assert_equal(camera_spec.resolution_h, 567)
        np.testing.assert_equal(camera_spec.resolution_w, 320)
        np.testing.assert_equal(camera_spec.shutter_type, 1)
        np.testing.assert_equal(camera_spec.WhichOneof("camera_param"), "ftheta_param")
        np.testing.assert_equal(camera_spec.ftheta_param.principal_point_x, 0)
        np.testing.assert_equal(camera_spec.ftheta_param.principal_point_y, 1)
        np.testing.assert_equal(camera_spec.ftheta_param.reference_poly, 1)
        np.testing.assert_array_equal(camera_spec.ftheta_param.pixeldist_to_angle_poly, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(camera_spec.ftheta_param.angle_to_pixeldist_poly, [0, 0, 0, 0, 0])
        np.testing.assert_equal(camera_spec.ftheta_param.max_angle, 120)
        np.testing.assert_equal(camera_spec.ftheta_param.linear_cde.linear_c, 1)
        np.testing.assert_equal(camera_spec.ftheta_param.linear_cde.linear_d, 2)
        np.testing.assert_equal(camera_spec.ftheta_param.linear_cde.linear_e, 3)

        # Check if the field 'temporary_camera_spec' is marked as deprecated
        descriptor = CameraSpec.DESCRIPTOR
        field_descriptor = descriptor.fields_by_name.get("temporary_camera_spec")
        self.assertIsNotNone(field_descriptor, "Field 'temporary_camera_spec' does not exist in CameraSpec.")
        self.assertTrue(
            field_descriptor.GetOptions().deprecated, "Field 'temporary_camera_spec' is not marked as deprecated."
        )

        # Check if the field 'trajectory_idx' is marked as deprecated
        field_descriptor = descriptor.fields_by_name.get("trajectory_idx")
        self.assertIsNotNone(field_descriptor, "Field 'trajectory_idx' does not exist in CameraSpec.")
        self.assertTrue(field_descriptor.GetOptions().deprecated, "Field 'trajectory_idx' is not marked as deprecated.")


class TestFThetaCameraModelParameters(unittest.TestCase):
    def test_get_camera_model_parameters_populates_external_distortion(self):
        linear_cde = LinearCde(linear_c=1.0, linear_d=2.0, linear_e=3.0)
        intrinsics = CameraSpec(
            logical_id="camera1",
            resolution_h=567,
            resolution_w=320,
            shutter_type=1,
            ftheta_param=FthetaCameraParam(
                principal_point_x=1.0,
                principal_point_y=2.0,
                reference_poly=ProtoWParams.ReferencePolynomial.BACKWARD,
                pixeldist_to_angle_poly=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                angle_to_pixeldist_poly=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                max_angle=1.0,
                linear_cde=linear_cde,
            ),
            bivariate_windshield_model_param=ProtoWParams(
                reference_poly=1,
                horizontal_poly=[1.0, 2.0],
                vertical_poly=[3.0],
                horizontal_poly_inverse=[4.0],
                vertical_poly_inverse=[5.0, 6.0],
            ),
        )

        camera_parameters = CameraBank._get_camera_model_parameters(intrinsics)
        assert isinstance(camera_parameters, ncore.data.FThetaCameraModelParameters)
        np.testing.assert_array_equal(camera_parameters.resolution, np.array([320, 567], dtype=np.uint64))
        np.testing.assert_equal(camera_parameters.shutter_type, 1)
        np.testing.assert_array_equal(camera_parameters.principal_point, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_equal(camera_parameters.reference_poly, 1)
        np.testing.assert_array_equal(
            camera_parameters.pixeldist_to_angle_poly, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            camera_parameters.angle_to_pixeldist_poly, np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
        )
        np.testing.assert_equal(camera_parameters.max_angle, 1.0)
        np.testing.assert_array_equal(camera_parameters.linear_cde, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        ext = camera_parameters.external_distortion_parameters
        np.testing.assert_equal(ext.reference_poly, ncore.data.ReferencePolynomial.BACKWARD)
        np.testing.assert_array_equal(ext.horizontal_poly, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(ext.vertical_poly, np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(ext.horizontal_poly_inverse, np.array([4.0], dtype=np.float32))
        np.testing.assert_array_equal(ext.vertical_poly_inverse, np.array([5.0, 6.0], dtype=np.float32))


class TestOpencvFisheyeCameraModelParameters(unittest.TestCase):
    def test_get_camera_model_parameters_populates_external_distortion(self):
        intrinsics = CameraSpec(
            logical_id="camera1",
            resolution_h=567,
            resolution_w=320,
            shutter_type=1,
            opencv_fisheye_param=OpenCVFisheyeCameraParam(
                principal_point_x=1.0,
                principal_point_y=2.0,
                focal_length_x=3.0,
                focal_length_y=4.0,
                radial_coeffs=[5.0, 6.0, 7.0, 8.0],
                max_angle=1.0,
            ),
        )

        camera_parameters = CameraBank._get_camera_model_parameters(intrinsics)
        assert isinstance(camera_parameters, ncore.data.OpenCVFisheyeCameraModelParameters)
        np.testing.assert_array_equal(camera_parameters.resolution, np.array([320, 567], dtype=np.uint64))
        np.testing.assert_equal(camera_parameters.shutter_type, 1)
        np.testing.assert_array_equal(camera_parameters.principal_point, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(camera_parameters.focal_length, np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(camera_parameters.radial_coeffs, np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        np.testing.assert_equal(camera_parameters.max_angle, 1.0)

        assert not intrinsics.HasField("external_distortion")


class TestOpencvPinholeCameraModelParameters(unittest.TestCase):
    def test_get_camera_model_parameters_populates_external_distortion(self):
        intrinsics = CameraSpec(
            logical_id="camera1",
            resolution_h=567,
            resolution_w=320,
            shutter_type=1,
            opencv_pinhole_param=OpenCVPinholeCameraParam(
                principal_point_x=1.0,
                principal_point_y=2.0,
                focal_length_x=3.0,
                focal_length_y=4.0,
                radial_coeffs=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                tangential_coeffs=[11.0, 12.0],
                thin_prism_coeffs=[13.0, 14.0, 15.0, 16.0],
            ),
        )

        camera_parameters = CameraBank._get_camera_model_parameters(intrinsics)
        assert isinstance(camera_parameters, ncore.data.OpenCVPinholeCameraModelParameters)
        np.testing.assert_array_equal(camera_parameters.resolution, np.array([320, 567], dtype=np.uint64))
        np.testing.assert_equal(camera_parameters.shutter_type, 1)
        np.testing.assert_array_equal(camera_parameters.principal_point, np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(camera_parameters.focal_length, np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(
            camera_parameters.radial_coeffs, np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(camera_parameters.tangential_coeffs, np.array([11.0, 12.0], dtype=np.float32))
        np.testing.assert_array_equal(
            camera_parameters.thin_prism_coeffs, np.array([13.0, 14.0, 15.0, 16.0], dtype=np.float32)
        )

        assert not intrinsics.HasField("external_distortion")


class TestActorTracksConversion(unittest.TestCase):
    def test_actor_tracks_round_trip(self):
        track = DynamicObjectTrack()
        track.id = "1"
        track.semantic_class = "automobile"

        track.object_size.size_x = 4.5
        track.object_size.size_y = 2.0
        track.object_size.size_z = 1.8

        trajectory = ProtoTrajectory()

        timestamps = [0, 500000, 1000000, 1500000, 2000000]

        for i, timestamp_us in enumerate(timestamps):
            pose = Pose()
            pose.vec.x = i * 5.0
            pose.vec.y = 0.0
            pose.vec.z = 0.0
            pose.quat.w = 1.0
            pose.quat.x = 0.0
            pose.quat.y = 0.0
            pose.quat.z = 0.0

            pose_at_time = PoseAtTime()
            pose_at_time.pose.CopyFrom(pose)
            pose_at_time.timestamp_us = timestamp_us

            trajectory.poses.append(pose_at_time)

        track.trajectory.CopyFrom(trajectory)

        original_tracks = [track]

        actor_tracks = actor_tracks_from_grpc(original_tracks)
        converted_tracks = actor_tracks_to_grpc(actor_tracks)

        self.assertEqual(len(converted_tracks), 1)
        converted_track = converted_tracks[0]

        self.assertEqual(converted_track.id, "1")
        self.assertEqual(converted_track.semantic_class, "automobile")

        np.testing.assert_almost_equal(converted_track.object_size.size_x, 4.5, decimal=6)
        np.testing.assert_almost_equal(converted_track.object_size.size_y, 2.0, decimal=6)
        np.testing.assert_almost_equal(converted_track.object_size.size_z, 1.8, decimal=6)

        self.assertEqual(len(converted_track.trajectory.poses), 5)

        for i in range(5):
            original_pose_at_time = track.trajectory.poses[i]
            converted_pose_at_time = converted_track.trajectory.poses[i]

            self.assertEqual(converted_pose_at_time.timestamp_us, timestamps[i])

            np.testing.assert_equal(converted_pose_at_time.pose.vec.x, original_pose_at_time.pose.vec.x)
            np.testing.assert_equal(converted_pose_at_time.pose.vec.y, original_pose_at_time.pose.vec.y)
            np.testing.assert_equal(converted_pose_at_time.pose.vec.z, original_pose_at_time.pose.vec.z)

            np.testing.assert_equal(converted_pose_at_time.pose.quat.w, original_pose_at_time.pose.quat.w)
            np.testing.assert_equal(converted_pose_at_time.pose.quat.x, original_pose_at_time.pose.quat.x)
            np.testing.assert_equal(converted_pose_at_time.pose.quat.y, original_pose_at_time.pose.quat.y)
            np.testing.assert_equal(converted_pose_at_time.pose.quat.z, original_pose_at_time.pose.quat.z)


class TestLidarRenderFilter(unittest.TestCase):
    """Test LidarRenderFilter protobuf message functionality."""

    def test_lidar_render_filter_defaults_to_unset(self):
        """Test that filter fields are unset by default."""
        filter_msg = LidarRenderFilter()

        self.assertFalse(filter_msg.HasField("raydrop_threshold"))
        self.assertFalse(filter_msg.HasField("opacity_threshold"))
        self.assertFalse(filter_msg.HasField("enable_distance_filter"))
        self.assertFalse(filter_msg.HasField("distance_filter_threshold"))

    def test_lidar_render_filter_set_all_fields(self):
        """Test setting all filter fields."""
        filter_msg = LidarRenderFilter(
            raydrop_threshold=0.5,
            opacity_threshold=0.0,
            enable_distance_filter=True,
            distance_filter_threshold=0.02,
        )

        self.assertTrue(filter_msg.HasField("raydrop_threshold"))
        self.assertTrue(filter_msg.HasField("opacity_threshold"))
        self.assertTrue(filter_msg.HasField("enable_distance_filter"))
        self.assertTrue(filter_msg.HasField("distance_filter_threshold"))

        np.testing.assert_almost_equal(filter_msg.raydrop_threshold, 0.5)
        np.testing.assert_almost_equal(filter_msg.opacity_threshold, 0.0)
        self.assertTrue(filter_msg.enable_distance_filter)
        np.testing.assert_almost_equal(filter_msg.distance_filter_threshold, 0.02)

    def test_lidar_render_filter_partial_fields(self):
        """Test setting only some filter fields."""
        filter_msg = LidarRenderFilter(
            raydrop_threshold=0.3,
            enable_distance_filter=False,
        )

        self.assertTrue(filter_msg.HasField("raydrop_threshold"))
        self.assertFalse(filter_msg.HasField("opacity_threshold"))
        self.assertTrue(filter_msg.HasField("enable_distance_filter"))
        self.assertFalse(filter_msg.HasField("distance_filter_threshold"))

        np.testing.assert_almost_equal(filter_msg.raydrop_threshold, 0.3)
        self.assertFalse(filter_msg.enable_distance_filter)

    def test_lidar_render_filter_zero_values_are_set(self):
        """Test that zero values are correctly detected as set (not default)."""
        filter_msg = LidarRenderFilter(
            raydrop_threshold=0.0,
            opacity_threshold=0.0,
            distance_filter_threshold=0.0,
        )

        # Zero values should still be detected as "set"
        self.assertTrue(filter_msg.HasField("raydrop_threshold"))
        self.assertTrue(filter_msg.HasField("opacity_threshold"))
        self.assertTrue(filter_msg.HasField("distance_filter_threshold"))

        np.testing.assert_almost_equal(filter_msg.raydrop_threshold, 0.0)
        np.testing.assert_almost_equal(filter_msg.opacity_threshold, 0.0)
        np.testing.assert_almost_equal(filter_msg.distance_filter_threshold, 0.0)


class TestLidarRenderRequest(unittest.TestCase):
    """Test LidarRenderRequest with render_filter field."""

    def test_lidar_render_request_without_filter(self):
        """Test that LidarRenderRequest works without a filter."""
        request = LidarRenderRequest(
            scene_id="test_scene",
            frame_start_us=1000000,
            frame_end_us=1100000,
        )

        self.assertEqual(request.scene_id, "test_scene")
        self.assertEqual(request.frame_start_us, 1000000)
        self.assertEqual(request.frame_end_us, 1100000)
        self.assertFalse(request.HasField("render_filter"))

    def test_lidar_render_request_with_filter(self):
        """Test that LidarRenderRequest correctly includes render_filter."""
        render_filter = LidarRenderFilter(
            raydrop_threshold=0.5,
            opacity_threshold=0.0,
            enable_distance_filter=True,
            distance_filter_threshold=0.02,
        )

        request = LidarRenderRequest(
            scene_id="test_scene",
            frame_start_us=1000000,
            frame_end_us=1100000,
            render_filter=render_filter,
        )

        self.assertTrue(request.HasField("render_filter"))
        np.testing.assert_almost_equal(request.render_filter.raydrop_threshold, 0.5)
        np.testing.assert_almost_equal(request.render_filter.opacity_threshold, 0.0)
        self.assertTrue(request.render_filter.enable_distance_filter)
        np.testing.assert_almost_equal(request.render_filter.distance_filter_threshold, 0.02)

    def test_lidar_render_request_with_partial_filter(self):
        """Test LidarRenderRequest with partially set filter fields."""
        render_filter = LidarRenderFilter(
            raydrop_threshold=0.7,
        )

        request = LidarRenderRequest(
            scene_id="test_scene",
            frame_start_us=1000000,
            frame_end_us=1100000,
            render_filter=render_filter,
        )

        self.assertTrue(request.HasField("render_filter"))
        self.assertTrue(request.render_filter.HasField("raydrop_threshold"))
        self.assertFalse(request.render_filter.HasField("opacity_threshold"))
        self.assertFalse(request.render_filter.HasField("enable_distance_filter"))
        self.assertFalse(request.render_filter.HasField("distance_filter_threshold"))

        np.testing.assert_almost_equal(request.render_filter.raydrop_threshold, 0.7)

    def test_lidar_render_request_with_lidar_spec(self):
        """Test LidarRenderRequest with both lidar_config and render_filter."""
        lidar_spec = LidarSpec(
            lidar_type=1,  # SPINNING
        )

        render_filter = LidarRenderFilter(
            raydrop_threshold=0.5,
            enable_distance_filter=False,
        )

        request = LidarRenderRequest(
            scene_id="test_scene",
            lidar_config=lidar_spec,
            frame_start_us=1000000,
            frame_end_us=1100000,
            render_filter=render_filter,
        )

        self.assertEqual(request.scene_id, "test_scene")
        self.assertEqual(request.lidar_config.lidar_type, 1)
        self.assertTrue(request.HasField("render_filter"))
        np.testing.assert_almost_equal(request.render_filter.raydrop_threshold, 0.5)
        self.assertFalse(request.render_filter.enable_distance_filter)


class TestLidarFilterExtraction(unittest.TestCase):
    """Test the filter extraction logic used in serve.py."""

    def _extract_filter_params(self, request: LidarRenderRequest):
        """
        Mimics the filter extraction logic from serve.py render_lidar_request().
        Returns a dict of filter parameters, with None for unset values.
        """
        raydrop_threshold = None
        opacity_threshold = None
        enable_distance_filter = None
        distance_filter_threshold = None

        if request.HasField("render_filter"):
            render_filter = request.render_filter
            if render_filter.HasField("raydrop_threshold"):
                raydrop_threshold = render_filter.raydrop_threshold
            if render_filter.HasField("opacity_threshold"):
                opacity_threshold = render_filter.opacity_threshold
            if render_filter.HasField("enable_distance_filter"):
                enable_distance_filter = render_filter.enable_distance_filter
            if render_filter.HasField("distance_filter_threshold"):
                distance_filter_threshold = render_filter.distance_filter_threshold

        return {
            "raydrop_threshold": raydrop_threshold,
            "opacity_threshold": opacity_threshold,
            "enable_distance_filter": enable_distance_filter,
            "distance_filter_threshold": distance_filter_threshold,
        }

    def test_extract_filter_params_no_filter(self):
        """Test extraction when no filter is provided."""
        request = LidarRenderRequest(scene_id="test")
        params = self._extract_filter_params(request)

        self.assertIsNone(params["raydrop_threshold"])
        self.assertIsNone(params["opacity_threshold"])
        self.assertIsNone(params["enable_distance_filter"])
        self.assertIsNone(params["distance_filter_threshold"])

    def test_extract_filter_params_all_set(self):
        """Test extraction when all filter params are set."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                raydrop_threshold=0.5,
                opacity_threshold=0.0,
                enable_distance_filter=True,
                distance_filter_threshold=0.02,
            ),
        )
        params = self._extract_filter_params(request)

        np.testing.assert_almost_equal(params["raydrop_threshold"], 0.5)
        np.testing.assert_almost_equal(params["opacity_threshold"], 0.0)
        self.assertTrue(params["enable_distance_filter"])
        np.testing.assert_almost_equal(params["distance_filter_threshold"], 0.02)

    def test_extract_filter_params_partial_set(self):
        """Test extraction when only some filter params are set."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                raydrop_threshold=0.3,
            ),
        )
        params = self._extract_filter_params(request)

        np.testing.assert_almost_equal(params["raydrop_threshold"], 0.3)
        self.assertIsNone(params["opacity_threshold"])
        self.assertIsNone(params["enable_distance_filter"])
        self.assertIsNone(params["distance_filter_threshold"])

    def test_extract_filter_params_zero_values(self):
        """Test that zero values are extracted correctly (not as None)."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                raydrop_threshold=0.0,
                opacity_threshold=0.0,
            ),
        )
        params = self._extract_filter_params(request)

        # Zero should be extracted as 0.0, not None
        np.testing.assert_almost_equal(params["raydrop_threshold"], 0.0)
        np.testing.assert_almost_equal(params["opacity_threshold"], 0.0)
        self.assertIsNone(params["enable_distance_filter"])
        self.assertIsNone(params["distance_filter_threshold"])

    def test_extract_filter_params_boolean_false(self):
        """Test that boolean False is extracted correctly."""
        request = LidarRenderRequest(
            scene_id="test",
            render_filter=LidarRenderFilter(
                enable_distance_filter=False,
            ),
        )
        params = self._extract_filter_params(request)

        self.assertIsNone(params["raydrop_threshold"])
        self.assertIsNone(params["opacity_threshold"])
        self.assertFalse(params["enable_distance_filter"])
        self.assertIsNone(params["distance_filter_threshold"])


class TestRenderErrorClassification(unittest.TestCase):
    def test_timestamp_out_of_range_maps_to_out_of_range(self):
        status, message = _classify_render_exception(TimestampOutOfRangeError("timestamp outside range"))
        self.assertEqual(status, grpc.StatusCode.OUT_OF_RANGE)
        self.assertEqual(message, "timestamp outside range")

    def test_model_incompatibility_maps_to_failed_precondition(self):
        status, message = _classify_render_exception(ModelIncompatibilityError("model mismatch"))
        self.assertEqual(status, grpc.StatusCode.FAILED_PRECONDITION)
        self.assertEqual(message, "model mismatch")

    def test_value_error_maps_to_invalid_argument(self):
        status, message = _classify_render_exception(ValueError("bad argument"))
        self.assertEqual(status, grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(message, "bad argument")

    def test_key_error_maps_to_not_found(self):
        status, message = _classify_render_exception(KeyError("missing-scene"))
        self.assertEqual(status, grpc.StatusCode.NOT_FOUND)
        self.assertEqual(message, "missing-scene")

    def test_out_of_memory_maps_to_resource_exhausted(self):
        status, message = _classify_render_exception(torch.OutOfMemoryError("oom"))
        self.assertEqual(status, grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.assertIn("oom", message)
        self.assertIn("unconverged checkpoint", message)

    def test_unknown_exception_maps_to_unknown(self):
        status, message = _classify_render_exception(RuntimeError("unexpected"))
        self.assertEqual(status, grpc.StatusCode.UNKNOWN)
        self.assertEqual(message, "unexpected")


class TestBackendDynamicObjectValidation(unittest.TestCase):
    def test_dynamic_object_updates_require_server_flag(self):
        backend = Backend(
            renderable_model=mock.MagicMock(),
            camera_bank=mock.MagicMock(),
            world_to_nre=mock.MagicMock(),
            lidar_bank=mock.MagicMock(),
            asset_bank=mock.MagicMock(),
        )

        with self.assertRaises(ValueError) as exc:
            backend._validate_dynamic_object_updates_request(num_dynamic_objects=1, enable_editing_actors=False)
        self.assertIn("--enable-editing-actors", str(exc.exception))

    def test_dynamic_object_updates_require_model_support(self):
        renderable_model = mock.MagicMock()
        renderable_model.supports_edit_actors.return_value = False
        backend = Backend(
            renderable_model=renderable_model,
            camera_bank=mock.MagicMock(),
            world_to_nre=mock.MagicMock(),
            lidar_bank=mock.MagicMock(),
            asset_bank=mock.MagicMock(),
        )

        with self.assertRaises(ValueError) as exc:
            backend._validate_dynamic_object_updates_request(num_dynamic_objects=1, enable_editing_actors=True)
        self.assertIn("does not support actor updates", str(exc.exception))


class TestRenderGrpcStatusCodes(unittest.TestCase):
    def _start_service(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        server_config = GrpcServerConfig()
        service = SensorSimService(
            server,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=server_config,
        )
        add_SensorsimServiceServicer_to_server(service, server)
        port = server.add_insecure_port("localhost:0")
        server.start()

        channel = grpc.insecure_channel(f"localhost:{port}")
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = SensorsimServiceStub(channel)
        return server, channel, stub, service

    def test_render_rgb_returns_out_of_range(self):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise TimestampOutOfRangeError("timestamp outside range")
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_rgb(RGBRenderRequest(scene_id="scene"))
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.OUT_OF_RANGE)
                details = ctx.exception.details() or ""
                self.assertIn("timestamp outside range", details)
        finally:
            channel.close()
            server.stop(0)

    def test_render_rgb_returns_resource_exhausted(self):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise torch.OutOfMemoryError("oom")
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_rgb(RGBRenderRequest(scene_id="scene"))
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)
                details = ctx.exception.details() or ""
                self.assertIn("oom", details)
        finally:
            channel.close()
            server.stop(0)

    def test_batch_render_rgb_returns_failed_precondition(self):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise ModelIncompatibilityError("model mismatch")
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                request = BatchRGBRenderRequest(
                    items=[
                        BatchRGBRenderRequestItem(
                            camera_name="cam",
                            request=RGBRenderRequest(scene_id="scene"),
                        )
                    ]
                )
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.batch_render_rgb(request)
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)
                details = ctx.exception.details() or ""
                self.assertIn("model mismatch", details)
        finally:
            channel.close()
            server.stop(0)

    def test_render_lidar_returns_invalid_argument(self):
        @contextlib.contextmanager
        def _raise_backend(*args, **kwargs):
            raise ValueError("actor editing is disabled")
            yield  # pragma: no cover - unreachable

        server, channel, stub, service = self._start_service()
        try:
            with mock.patch.object(service, "get_backend", _raise_backend):
                with self.assertRaises(grpc.RpcError) as ctx:
                    stub.render_lidar(LidarRenderRequest(scene_id="scene"))
                self.assertEqual(ctx.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
                details = ctx.exception.details() or ""
                self.assertIn("actor editing is disabled", details)
        finally:
            channel.close()
            server.stop(0)


class TestServerConfig(unittest.TestCase):
    def setUp(self):
        self.server_config = GrpcServerConfig()
        self.expected_server_config_map = {
            "artifact_glob": "null",
            "host": "localhost",
            "port": "8080",
            "health_port": "null",
            "test_scenes_are_valid": "False",
            "renderer": "default",
            "enable_difix": "False",
            "difix_url": "https://api.ngc.nvidia.com/v2/org/nvidia/team/nre/models/nurec-fixer/versions/cosmos_3dgut/files/cosmos_3dgut.pt",
            "difix_cache": "~/.cache/nre/difix",
            "difix_model_filename": "cosmos_3dgut.pt",
            "difix_resolution": "(576, 1024)",
            "ray_chunk_size": "4611686018427387904",
            "egocar_hood_dir": "null",
            "download_cache_dir": "~/.cache/nre/downloaded_scenes",
            "download_cache_size": "5",
            "max_workers": "1",
            "enable_editing_actors": "False",
            "cache_size": "10",
            "metrics_output_dir": "null",
        }

    def test_server_config_map(self):
        server_config_map = SensorSimService._build_server_config_map(self.server_config)
        self.assertEqual(
            server_config_map,
            self.expected_server_config_map,
        )

    @unittest.skip("Skipping server client round trip test by default to avoid flakiness in CI.")
    def test_server_client_round_trip(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        server_config = GrpcServerConfig()
        service = SensorSimService(
            server,
            artifacts_glob=None,
            ray_chunk_size=1,
            egocar_hoods_dir=None,
            downloader=None,
            scene_cache=None,
            cache_size=1,
            metrics_output_dir=None,
            enable_editing_actors=False,
            server_config=server_config,
        )
        add_SensorsimServiceServicer_to_server(service, server)
        port = server.add_insecure_port("localhost:0")
        server.start()

        channel = grpc.insecure_channel(f"localhost:{port}")
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
            stub = SensorsimServiceStub(channel)
            response = stub.get_server_config(Empty())

            self.assertEqual(response.server_config, self.expected_server_config_map)
        finally:
            channel.close()
            server.stop(0)


def _make_handler_call_details(invocation_metadata=None, method="/nre.Sensorsim/RenderRgb"):
    """Fake HandlerCallDetails for interceptor tests"""
    details = unittest.mock.MagicMock()
    details.method = method
    details.invocation_metadata = list(invocation_metadata or [])
    return details


class TestIsLocalSceneUri(unittest.TestCase):
    """Unit tests for _is_local_scene_uri (dispatch: file:// or absolute path = local; else remote)."""

    def test_file_uri_returns_true(self):
        self.assertTrue(_is_local_scene_uri("file:///data/scene.usdz"))
        self.assertTrue(_is_local_scene_uri("FILE:///other/path.usdz"))

    def test_absolute_path_returns_true(self):
        self.assertTrue(_is_local_scene_uri("/data/scenes/foo.usdz"))

    def test_https_uri_returns_false(self):
        self.assertFalse(_is_local_scene_uri("https://example.com/scene.usdz"))

    def test_http_uri_returns_false(self):
        self.assertFalse(_is_local_scene_uri("http://example.com/scene.usdz"))

    def test_bare_hostname_or_relative_returns_false(self):
        """Scheme-less hostnames and relative paths are not treated as local (routed to download)."""
        self.assertFalse(_is_local_scene_uri("urm.nvidia.com/path/to/scene.usdz"))
        self.assertFalse(_is_local_scene_uri("scenes/foo.usdz"))


class TestSceneDownloadInterceptor(unittest.TestCase):
    """Unit tests for SceneDownloadInterceptor: x-nre-scene-url / x-nre-scene-id dispatch and errors."""

    def setUp(self):
        self.interceptor = SceneDownloadInterceptor()
        self.service = unittest.mock.MagicMock(spec=SensorSimService)
        self.interceptor.set_service(self.service)

    def test_no_x_nre_scene_url_calls_continuation(self):
        details = _make_handler_call_details(invocation_metadata=[])
        continuation = unittest.mock.Mock(return_value=unittest.mock.sentinel.handler)
        result = self.interceptor.intercept_service(continuation, details)
        continuation.assert_called_once_with(details)
        self.assertIs(result, unittest.mock.sentinel.handler)

    def test_file_uri_calls_register_local_scene(self):
        details = _make_handler_call_details(
            invocation_metadata=[
                ("x-nre-scene-url", "file:///data/scene.usdz"),
                ("x-nre-scene-id", "scene_1"),
            ]
        )
        continuation = unittest.mock.Mock(return_value=unittest.mock.sentinel.handler)
        result = self.interceptor.intercept_service(continuation, details)
        self.service._register_local_scene.assert_called_once_with("scene_1", "file:///data/scene.usdz")
        continuation.assert_called_once_with(details)
        self.assertIs(result, unittest.mock.sentinel.handler)

    def test_bare_path_calls_register_local_scene(self):
        details = _make_handler_call_details(
            invocation_metadata=[
                ("x-nre-scene-url", "/data/scenes/foo.usdz"),
                ("x-nre-scene-id", "scene_2"),
            ]
        )
        continuation = unittest.mock.Mock(return_value=unittest.mock.sentinel.handler)
        result = self.interceptor.intercept_service(continuation, details)
        self.service._register_local_scene.assert_called_once_with("scene_2", "/data/scenes/foo.usdz")
        continuation.assert_called_once_with(details)
        self.assertIs(result, unittest.mock.sentinel.handler)

    def test_https_uri_calls_download_scene(self):
        details = _make_handler_call_details(
            invocation_metadata=[
                ("x-nre-scene-url", "https://example.com/scene.usdz"),
                ("x-nre-scene-id", "scene_2"),
            ]
        )
        continuation = unittest.mock.Mock(return_value=unittest.mock.sentinel.handler)
        result = self.interceptor.intercept_service(continuation, details)
        self.service._download_scene.assert_called_once_with("https://example.com/scene.usdz", "scene_2")
        continuation.assert_called_once_with(details)
        self.assertIs(result, unittest.mock.sentinel.handler)

    def test_invalid_scene_id_aborts_with_invalid_argument(self):
        details = _make_handler_call_details(
            invocation_metadata=[
                ("x-nre-scene-url", "file:///data/scene.usdz"),
                ("x-nre-scene-id", "../invalid"),
            ]
        )
        continuation = unittest.mock.Mock(return_value=unittest.mock.sentinel.handler)
        result = self.interceptor.intercept_service(continuation, details)
        self.service._register_local_scene.assert_not_called()
        self.service._download_scene.assert_not_called()
        continuation.assert_not_called()
        self.assertIsNot(result, unittest.mock.sentinel.handler)
        # Returned handler should be a terminator that aborts with INVALID_ARGUMENT
        handler = result
        self.assertIsNotNone(handler.unary_unary)
        context = unittest.mock.Mock()
        handler.unary_unary(None, context)
        context.abort.assert_called_once()
        call_args = context.abort.call_args[0]
        self.assertEqual(call_args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("Invalid scene ID", call_args[1])

    def test_register_local_scene_raises_aborts_with_invalid_argument(self):
        self.service._register_local_scene.side_effect = ValueError("path does not exist")
        details = _make_handler_call_details(
            invocation_metadata=[
                ("x-nre-scene-url", "file:///nonexistent.usdz"),
                ("x-nre-scene-id", "scene_1"),
            ]
        )
        continuation = unittest.mock.Mock(return_value=unittest.mock.sentinel.handler)
        result = self.interceptor.intercept_service(continuation, details)
        continuation.assert_not_called()
        context = unittest.mock.Mock()
        result.unary_unary(None, context)
        context.abort.assert_called_once()
        call_args = context.abort.call_args[0]
        self.assertEqual(call_args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("Failed to register local scene", call_args[1])
        self.assertIn("path does not exist", call_args[1])

    def test_download_scene_raises_aborts_with_invalid_argument(self):
        self.service._download_scene.side_effect = RuntimeError("network error")
        details = _make_handler_call_details(
            invocation_metadata=[
                ("x-nre-scene-url", "https://example.com/scene.usdz"),
                ("x-nre-scene-id", "scene_1"),
            ]
        )
        continuation = unittest.mock.Mock(return_value=unittest.mock.sentinel.handler)
        result = self.interceptor.intercept_service(continuation, details)
        continuation.assert_not_called()
        context = unittest.mock.Mock()
        result.unary_unary(None, context)
        context.abort.assert_called_once()
        call_args = context.abort.call_args[0]
        self.assertEqual(call_args[0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("Failed to download scene", call_args[1])
        self.assertIn("network error", call_args[1])
