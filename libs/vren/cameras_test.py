# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import dataclasses
import unittest

from typing import cast

import numpy as np
import parameterized
import torch

from python.runfiles import runfiles

from libs.vren.interface import (  # type: ignore
    camera_rays_to_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_shutter_pose,
    world_points_to_image_points_shutter_pose,
)
from ncore.data import (
    BivariateWindshieldModelParameters,
    ConcreteCameraModelParametersUnion,
    FrameTimepoint,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ReferencePolynomial,
    ShutterType,
)
from ncore.sensors import CameraModel, FThetaCameraModel, OpenCVFisheyeCameraModel, OpenCVPinholeCameraModel
from ncore_internal.data.v3 import ShardDataLoader
from nre.utils.geometry import se3_matrix_to_tquat, tquat_to_se3_matrix
from nre.utils.misc import unpack_optional
from nre.utils.tests import CommonTestCase


RUNFILES = runfiles.Create()


class CamerasTest(CommonTestCase):
    def setUp(self) -> None:
        # load test FTheta camera sensor from test-data
        self.camera_sensor = ShardDataLoader(
            [
                RUNFILES.Rlocation(
                    "test_data_ncore/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.zarr.itar"
                ),
            ],
            open_consolidated=False,
        ).get_camera_sensor("camera_front_wide_120fov")

        # NV FTheta test parameterization (bw-based)
        self.ncore_ftheta_bw_camera_model_parameters = cast(
            FThetaCameraModelParameters, self.camera_sensor.get_camera_model_parameters()
        )
        assert (
            self.ncore_ftheta_bw_camera_model_parameters.reference_poly
            == FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE
        ), "Expected reference_poly to be PIXELDIST_TO_ANGLE / backward variant for this test's data"
        self.ncore_ftheta_bw_camera_model = FThetaCameraModel(self.ncore_ftheta_bw_camera_model_parameters)

        # FTheta (fw-based) camera model parameters with
        # - different shutter direction (left-to-right)
        # - significant linear_cde term
        self.ncore_ftheta_fw_camera_model_parameters = FThetaCameraModelParameters(
            resolution=np.array([3848, 2168], dtype=np.uint64),
            shutter_type=ShutterType.ROLLING_LEFT_TO_RIGHT,
            principal_point=np.array([1909.3092041015625, 1103.27880859375], dtype=np.float32),
            reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=np.array(
                [
                    0.0,
                    0.00031855489942245185,
                    -5.4367417234857385e-09,
                    4.775631279319015e-12,
                    -1.0283620548333567e-15,
                    -1.1274463994279525e-19,
                ],
                dtype=np.float32,
            ),
            angle_to_pixeldist_poly=np.array(
                [
                    0.0,
                    3139.48583984375,
                    164.5725860595703,
                    -442.12896728515625,
                    259.5827331542969,
                    153.66644287109375,
                ],
                dtype=np.float32,
            ),
            max_angle=0.7037167544041137,
            linear_cde=np.array([1.1, -0.1, 0.2], dtype=np.float32),  # updated from [1,0,0] to be more significant
        )
        self.ncore_ftheta_fw_camera_model = FThetaCameraModel(self.ncore_ftheta_fw_camera_model_parameters)

        # Define the ftheta camera model with windshield distortion
        horizontal_poly = np.array(
            [
                -0.000475919834570959,
                0.99944007396698,
                0.000166745347087272,
                0.000205887947231531,
                0.0055195577442646,
                0.000861024134792387,
            ],
            dtype=np.float32,
        )
        vertical_poly = np.array(
            [
                0.00152770057320595,
                -0.000532537756953388,
                -5.65027039556298e-05,
                -4.02410341848736e-06,
                0.000608163303695619,
                1.0094313621521,
                -0.00125278066843748,
                0.00823396816849709,
                -0.000293767458060756,
                0.0185473654419184,
                -0.003074218519032,
                0.00599765172228217,
                0.0172030478715897,
                -0.00364979170262814,
                0.0069147446192801,
            ],
            dtype=np.float32,
        )
        horizontal_poly_inverse = np.array(
            [0.0004770369, 1.0005774, -0.00016896478, -0.00020207358, -0.0054899976, -0.0008536868],
            dtype=np.float32,
        )
        vertical_poly_inverse = np.array(
            [
                -0.0015191488,
                0.00052959577,
                7.882431e-05,
                -6.966009e-06,
                -0.00059701066,
                0.9906775,
                0.00116782,
                -0.007893825,
                0.00026140467,
                -0.017767625,
                0.0027627628,
                -0.00544897,
                -0.015480865,
                0.0033684247,
                -0.0057964055,
            ],
            dtype=np.float32,
        )
        windshield_params = BivariateWindshieldModelParameters(
            reference_poly=ReferencePolynomial.FORWARD,
            horizontal_poly=horizontal_poly,
            vertical_poly=vertical_poly,
            horizontal_poly_inverse=horizontal_poly_inverse,
            vertical_poly_inverse=vertical_poly_inverse,
        )
        self.ncore_ftheta_fw_ws_camera_model_parameters = dataclasses.replace(
            self.ncore_ftheta_fw_camera_model_parameters, external_distortion_parameters=windshield_params
        )
        self.ncore_ftheta_fw_ws_camera_model = FThetaCameraModel(self.ncore_ftheta_fw_ws_camera_model_parameters)

        # Waymo OpenCVPinhole test parameterization
        self.ncore_opencvpinhole_camera_model_parameters = OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
            principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
            focal_length=np.array(
                [
                    2059.0471439559833,
                    2059.0471439559833,
                ],
                dtype=np.float32,
            ),
            radial_coeffs=np.array(
                [
                    0.04239636827428756,
                    -0.34165672675852826,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=np.float32,
            ),
            tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
            thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
        )

        # add additional arbitrary radial and thin-prism coeffs for this test only to guarantee code-coverage
        self.ncore_opencvpinhole_camera_model_parameters.radial_coeffs[2:] = [0.01, 0.02, -0.01, 0.02]
        self.ncore_opencvpinhole_camera_model_parameters.thin_prism_coeffs[:] = [0.01, 0.02, 0.02, 0.01]

        self.ncore_opencvpinhole_camera_model = OpenCVPinholeCameraModel(
            self.ncore_opencvpinhole_camera_model_parameters
        )
        self.ncore_opencvpinhole_ws_camera_model_parameters = dataclasses.replace(
            self.ncore_opencvpinhole_camera_model_parameters, external_distortion_parameters=windshield_params
        )
        self.ncore_opencvpinhole_ws_camera_model = OpenCVPinholeCameraModel(
            self.ncore_opencvpinhole_ws_camera_model_parameters
        )

        # LiAuto OpenCVFisheye test parameterization
        self.ncore_opencvfisheye_camera_model_parameters = OpenCVFisheyeCameraModelParameters(
            resolution=np.array([3840, 2160], dtype=np.uint64),
            shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
            principal_point=np.array([1928.184506, 1083.862789], dtype=np.float32),
            focal_length=np.array(
                [
                    1913.76478,
                    1913.99708,
                ],
                dtype=np.float32,
            ),
            radial_coeffs=np.array(
                [
                    -0.030093122,
                    -0.005103817,
                    -0.000849622,
                    0.001079542,
                ],
                dtype=np.float32,
            ),
            max_angle=np.deg2rad(140 / 2),
        )

        self.ncore_opencvfisheye_camera_model = OpenCVFisheyeCameraModel(
            self.ncore_opencvfisheye_camera_model_parameters
        )
        self.ncore_opencvfisheye_ws_camera_model_parameters = dataclasses.replace(
            self.ncore_opencvfisheye_camera_model_parameters, external_distortion_parameters=windshield_params
        )
        self.ncore_opencvfisheye_ws_camera_model = OpenCVFisheyeCameraModel(
            self.ncore_opencvfisheye_ws_camera_model_parameters
        )

        # create test data
        torch.manual_seed(0)  # make sure tests are reproducible
        self.camera_points = 2 * torch.rand(5000, 3).cuda() - 1
        self.world_points = 70 * (2 * torch.rand(3000, 3).cuda() - 1)

    def _generate_diag_edge_image_points(self, resolution) -> torch.Tensor:
        width, height = int(resolution[0]), int(resolution[1])
        # sample along diagonals of actual rectangle
        diag_steps = max(width, height)
        t_diag = torch.linspace(0.0, 1.0, steps=diag_steps)
        x_main = t_diag * (width - 1) + 0.5
        y_main = t_diag * (height - 1) + 0.5
        main_diag = torch.stack([x_main, y_main], dim=1)
        x_anti = (1.0 - t_diag) * (width - 1) + 0.5
        y_anti = t_diag * (height - 1) + 0.5
        anti_diag = torch.stack([x_anti, y_anti], dim=1)
        # image edges: top, bottom, left, right
        x_top = torch.arange(width).float() + 0.5
        top = torch.stack([x_top, torch.full((width,), 0.5)], dim=1)
        bottom = torch.stack([x_top, torch.full((width,), height - 0.5)], dim=1)
        y_left = torch.arange(height).float() + 0.5
        left = torch.stack([torch.full((height,), 0.5), y_left], dim=1)
        right = torch.stack([torch.full((height,), width - 0.5), y_left], dim=1)
        return torch.cat([main_diag, anti_diag, top, bottom, left, right], dim=0).cuda()

    def test_ncore_camera_rays_to_image_points_consistency(self) -> None:
        def check(ref_projection: CameraModel.ImagePointsReturn, projection: CameraModel.ImagePointsReturn):
            # check for consistency
            self._compareTensor(ref_projection.valid_flag, projection.valid_flag)
            self._compareTensor(
                ref_projection.image_points,
                projection.image_points,
                # sub 3 decimal pixel accuracy is sufficient
                decimal=3,
            )

        for ncore_camera_model, ncore_camera_model_parameters in [
            # FTheta backward
            (self.ncore_ftheta_bw_camera_model, self.ncore_ftheta_bw_camera_model_parameters),
            # FTheta forward
            (self.ncore_ftheta_fw_camera_model, self.ncore_ftheta_fw_camera_model_parameters),
            # FTheta forward with windshield distortion
            (self.ncore_ftheta_fw_ws_camera_model, self.ncore_ftheta_fw_ws_camera_model_parameters),
            # OpenCVPinhole
            (self.ncore_opencvpinhole_camera_model, self.ncore_opencvpinhole_camera_model_parameters),
            # OpenCVPinhole with windshield distortion
            (self.ncore_opencvpinhole_ws_camera_model, self.ncore_opencvpinhole_ws_camera_model_parameters),
            # OpenCVFisheye
            (self.ncore_opencvfisheye_camera_model, self.ncore_opencvfisheye_camera_model_parameters),
            # OpenCVFisheye with windshield distortion
            (self.ncore_opencvfisheye_ws_camera_model, self.ncore_opencvfisheye_ws_camera_model_parameters),
        ]:
            check(
                # evaluate reference ncore model
                ncore_camera_model.camera_rays_to_image_points(self.camera_points),
                # evaluate cuda model
                camera_rays_to_image_points(ncore_camera_model_parameters, self.camera_points),
            )

    def test_ncore_image_points_to_camera_rays_consistency(self) -> None:
        def check(
            ncore_camera_model: CameraModel,
            camera_model_parameters: ConcreteCameraModelParametersUnion,
        ):
            # generate all required test pixels
            image_points = self._generate_diag_edge_image_points(camera_model_parameters.resolution)

            # check for consistency
            self._compareTensor(
                ncore_camera_model.image_points_to_camera_rays(image_points),
                image_points_to_camera_rays(camera_model_parameters, image_points),
                # sub 3 decimal pixel accuracy is sufficient
                decimal=3,
            )

        for ncore_camera_model, ncore_camera_model_parameters in [
            # FTheta backward
            (self.ncore_ftheta_bw_camera_model, self.ncore_ftheta_bw_camera_model_parameters),
            # FTheta forward
            (self.ncore_ftheta_fw_camera_model, self.ncore_ftheta_fw_camera_model_parameters),
            # FTheta forward with windshield distortion
            (self.ncore_ftheta_fw_ws_camera_model, self.ncore_ftheta_fw_ws_camera_model_parameters),
            # OpenCVPinhole
            (self.ncore_opencvpinhole_camera_model, self.ncore_opencvpinhole_camera_model_parameters),
            # OpenCVPinhole with windshield distortion
            (self.ncore_opencvpinhole_ws_camera_model, self.ncore_opencvpinhole_ws_camera_model_parameters),
            # OpenCVFisheye
            (self.ncore_opencvfisheye_camera_model, self.ncore_opencvfisheye_camera_model_parameters),
            # OpenCVFisheye with windshield distortion
            (self.ncore_opencvfisheye_ws_camera_model, self.ncore_opencvfisheye_ws_camera_model_parameters),
        ]:
            check(
                ncore_camera_model,
                cast(ConcreteCameraModelParametersUnion, ncore_camera_model_parameters),
            )

    @parameterized.parameterized.expand((True, False))
    def test_ncore_world_rays_to_image_points_consistency(self, return_all_projections) -> None:
        # sample some timestamped world->sensor poses
        T_world_sensors = torch.stack(
            [
                torch.tensor(self.camera_sensor.get_frame_T_world_sensor(0, FrameTimepoint.START)),
                torch.tensor(self.camera_sensor.get_frame_T_world_sensor(0, FrameTimepoint.END)),
            ]
        )
        timestamps_us = torch.zeros((2,), dtype=torch.int64)
        timestamps_us[0] = self.camera_sensor.get_frame_timestamp_us(0, FrameTimepoint.START)
        timestamps_us[1] = self.camera_sensor.get_frame_timestamp_us(0, FrameTimepoint.END)

        def check(
            ref_projection: CameraModel.WorldPointsToImagePointsReturn,
            projection: CameraModel.WorldPointsToImagePointsReturn,
        ):
            # check for consistency
            self._compareTensor(
                unpack_optional(ref_projection.valid_indices), unpack_optional(projection.valid_indices)
            )

            ref_image_points = ref_projection.image_points
            image_points = projection.image_points

            # Only verify accuracy for valid points
            if return_all_projections:
                ref_image_points = ref_image_points[ref_projection.valid_indices]
                image_points = image_points[ref_projection.valid_indices]

            self._compareTensor(
                ref_image_points,
                image_points,
                # sub 3 decimal pixel accuracy is sufficient
                decimal=3,
            )
            self._compareTensor(
                unpack_optional(ref_projection.timestamps_us),
                unpack_optional(projection.timestamps_us),
            )
            self._compareTensor(
                unpack_optional(ref_projection.T_world_sensors),
                tquat_to_se3_matrix(unpack_optional(projection.T_world_sensors)),
            )

        for ncore_camera_model, ncore_camera_model_parameters in [
            # FTheta backward
            (self.ncore_ftheta_bw_camera_model, self.ncore_ftheta_bw_camera_model_parameters),
            # FTheta forward
            (self.ncore_ftheta_fw_camera_model, self.ncore_ftheta_fw_camera_model_parameters),
            # FTheta forward with windshield distortion
            (self.ncore_ftheta_fw_ws_camera_model, self.ncore_ftheta_fw_ws_camera_model_parameters),
            # OpenCVPinhole
            (self.ncore_opencvpinhole_camera_model, self.ncore_opencvpinhole_camera_model_parameters),
            # OpenCVPinhole with windshield distortion
            (self.ncore_opencvpinhole_ws_camera_model, self.ncore_opencvpinhole_ws_camera_model_parameters),
            # OpenCVFisheye
            (self.ncore_opencvfisheye_camera_model, self.ncore_opencvfisheye_camera_model_parameters),
            # OpenCVFisheye with windshield distortion
            (self.ncore_opencvfisheye_ws_camera_model, self.ncore_opencvfisheye_ws_camera_model_parameters),
        ]:
            check(
                # evaluate reference ncore model
                ncore_camera_model.world_points_to_image_points_shutter_pose(
                    self.world_points,
                    T_world_sensors[0],
                    T_world_sensors[1],
                    cast(int, timestamps_us[0].item()),
                    cast(int, timestamps_us[1].item()),
                    return_T_world_sensors=True,
                    return_timestamps=True,
                    return_valid_indices=True,
                    return_all_projections=return_all_projections,
                ),
                # evaluate cuda model
                world_points_to_image_points_shutter_pose(
                    ncore_camera_model_parameters,
                    self.world_points,
                    se3_matrix_to_tquat(T_world_sensors),
                    timestamps_us,
                    return_all_projections=return_all_projections,
                ),
            )

    def test_ncore_image_points_to_world_rays_consistency(self) -> None:
        # sample some timestamped world->sensor poses
        T_sensor_worlds = torch.stack(
            [
                torch.tensor(self.camera_sensor.get_frame_T_sensor_world(0, FrameTimepoint.START)),
                torch.tensor(self.camera_sensor.get_frame_T_sensor_world(0, FrameTimepoint.END)),
            ]
        )
        timestamps_us = torch.zeros((2,), dtype=torch.int64)
        timestamps_us[0] = self.camera_sensor.get_frame_timestamp_us(0, FrameTimepoint.START)
        timestamps_us[1] = self.camera_sensor.get_frame_timestamp_us(0, FrameTimepoint.END)

        def check(
            ncore_camera_model: CameraModel,
            camera_model_parameters: ConcreteCameraModelParametersUnion,
        ):
            # generate all required test pixels
            image_points = self._generate_diag_edge_image_points(camera_model_parameters.resolution)

            # evaluate reference ncore model
            ref_world_rays = ncore_camera_model.image_points_to_world_rays_shutter_pose(
                image_points,
                T_sensor_worlds[0],
                T_sensor_worlds[1],
                cast(int, timestamps_us[0].item()),
                cast(int, timestamps_us[1].item()),
                return_T_sensor_worlds=True,
                return_timestamps=True,
            )

            # evaluate cuda model
            world_rays = image_points_to_world_rays_shutter_pose(
                camera_model_parameters,
                image_points,
                se3_matrix_to_tquat(T_sensor_worlds),
                timestamps_us,
            )

            ref_world_ray_orgs = ref_world_rays.world_rays[:, :3]
            ref_world_ray_dirs = ref_world_rays.world_rays[:, 3:]

            world_ray_orgs = world_rays.world_rays[:, :3]
            world_ray_dirs = world_rays.world_rays[:, 3:]

            # check for consistency
            self._compareTensor(
                unpack_optional(ref_world_rays.timestamps_us),
                unpack_optional(world_rays.timestamps_us),
            )
            self._compareTensor(
                ref_world_ray_orgs,
                world_ray_orgs,
            )
            self._compareTensor(
                ref_world_ray_dirs,
                world_ray_dirs,
                # sub 3 decimal pixel accuracy is sufficient
                decimal=3,
            )
            self._compareTensor(
                unpack_optional(ref_world_rays.T_sensor_worlds),
                tquat_to_se3_matrix(unpack_optional(world_rays.T_sensor_worlds)),
            )

        for ncore_camera_model, ncore_camera_model_parameters in [
            # FTheta backward
            (self.ncore_ftheta_bw_camera_model, self.ncore_ftheta_bw_camera_model_parameters),
            # FTheta forward
            (self.ncore_ftheta_fw_camera_model, self.ncore_ftheta_fw_camera_model_parameters),
            # FTheta forward with windshield distortion
            (self.ncore_ftheta_fw_ws_camera_model, self.ncore_ftheta_fw_ws_camera_model_parameters),
            # OpenCVPinhole
            (self.ncore_opencvpinhole_camera_model, self.ncore_opencvpinhole_camera_model_parameters),
            # OpenCVPinhole with windshield distortion
            (self.ncore_opencvpinhole_ws_camera_model, self.ncore_opencvpinhole_ws_camera_model_parameters),
            # OpenCVFisheye
            (self.ncore_opencvfisheye_camera_model, self.ncore_opencvfisheye_camera_model_parameters),
            # OpenCVFisheye with windshield distortion
            (self.ncore_opencvfisheye_ws_camera_model, self.ncore_opencvfisheye_ws_camera_model_parameters),
        ]:
            check(
                ncore_camera_model,
                cast(ConcreteCameraModelParametersUnion, ncore_camera_model_parameters),
            )


if __name__ == "__main__":
    unittest.main()
