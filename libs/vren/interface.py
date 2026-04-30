# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import importlib

from typing import Any, TypeAlias

# import torch for tensor operations
import torch

from ncore.data import (
    BivariateWindshieldModelParameters,
    ConcreteCameraModelParametersUnion,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ReferencePolynomial,
    ShutterType,
)
from ncore.sensors import CameraModel


# Pycena is not able to find libvren_cc via regular import (also it does not like __init__.py).
# Also MyPy likes the vren module variable to be of Any type so it treats all its attributes as dynamic, avoiding
# name-defined errors.
vren: Any = importlib.import_module("libs.vren.libvren_cc")

# Camera model parameter pack type definitions
VrenConcreteCameraModelParametersUnion: TypeAlias = (
    vren.OpenCVPinholeCameraModelParameters | vren.OpenCVFisheyeCameraModelParameters | vren.FThetaCameraModelParameters
)
ExternalDistortionParametersUnion: TypeAlias = (
    BivariateWindshieldModelParameters | vren.BivariateWindshieldModelParameters
)
CameraModelParametersUnion: TypeAlias = ConcreteCameraModelParametersUnion | VrenConcreteCameraModelParametersUnion


def external_distortion_to_vren(
    external_distortion_parameters: ExternalDistortionParametersUnion | None,
) -> vren.BivariateWindshieldModelParameters | None:
    """Converts external distortion parameters to vren format"""

    REFERENCE_POLY_MAP = {
        ReferencePolynomial.FORWARD: vren.ReferencePolynomial.FORWARD,
        ReferencePolynomial.BACKWARD: vren.ReferencePolynomial.BACKWARD,
    }

    if external_distortion_parameters is not None:
        vren_external_distortion_parameters = vren.BivariateWindshieldModelParameters()
        vren_external_distortion_parameters.reference_poly = REFERENCE_POLY_MAP[
            external_distortion_parameters.reference_poly
        ]
        vren_external_distortion_parameters.horizontal_poly = external_distortion_parameters.horizontal_poly.tolist()
        vren_external_distortion_parameters.vertical_poly = external_distortion_parameters.vertical_poly.tolist()
        vren_external_distortion_parameters.horizontal_poly_inverse = (
            external_distortion_parameters.horizontal_poly_inverse.tolist()
        )
        vren_external_distortion_parameters.vertical_poly_inverse = (
            external_distortion_parameters.vertical_poly_inverse.tolist()
        )
        return vren_external_distortion_parameters
    else:
        return None


def to_vren(
    ncore_or_vren_camera_model_parameters: CameraModelParametersUnion,
) -> VrenConcreteCameraModelParametersUnion:
    """Provides vren camera model parameter (generic over camera model type) by either

        - initializing from corresponding NCore camera model parameter type
        - forwarding argument directly if of 'vren.<CameraModelType>Parameters' type

    Errors out if non-supported camera model parameter structure is provided"""

    SHUTTER_TYPE_MAP = {
        ShutterType.ROLLING_TOP_TO_BOTTOM: vren.ShutterType.ROLLING_TOP_TO_BOTTOM,
        ShutterType.ROLLING_LEFT_TO_RIGHT: vren.ShutterType.ROLLING_LEFT_TO_RIGHT,
        ShutterType.ROLLING_BOTTOM_TO_TOP: vren.ShutterType.ROLLING_BOTTOM_TO_TOP,
        ShutterType.ROLLING_RIGHT_TO_LEFT: vren.ShutterType.ROLLING_RIGHT_TO_LEFT,
        ShutterType.GLOBAL: vren.ShutterType.GLOBAL,
    }

    match ncore_or_vren_camera_model_parameters:
        case (
            vren.OpenCVPinholeCameraModelParameters()
            | vren.OpenCVFisheyeCameraModelParameters()
            | vren.FThetaCameraModelParameters()
        ):
            # identity transformation
            return ncore_or_vren_camera_model_parameters

        case OpenCVPinholeCameraModelParameters(
            resolution=resolution,
            shutter_type=shutter_type,
            external_distortion_parameters=external_distortion_parameters,
            principal_point=principal_point,
            focal_length=focal_length,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
        ):
            vren_params = vren.OpenCVPinholeCameraModelParameters()

            vren_params.resolution = resolution.tolist()
            vren_params.shutter_type = SHUTTER_TYPE_MAP[shutter_type]
            vren_params.external_distortion_parameters = external_distortion_to_vren(external_distortion_parameters)

            vren_params.principal_point = principal_point.tolist()
            vren_params.focal_length = focal_length.tolist()
            vren_params.radial_coeffs = radial_coeffs.tolist()
            vren_params.tangential_coeffs = tangential_coeffs.tolist()
            vren_params.thin_prism_coeffs = thin_prism_coeffs.tolist()

            return vren_params

        case OpenCVFisheyeCameraModelParameters(
            resolution=resolution,
            shutter_type=shutter_type,
            external_distortion_parameters=external_distortion_parameters,
            principal_point=principal_point,
            focal_length=focal_length,
            radial_coeffs=radial_coeffs,
            max_angle=max_angle,
        ):
            vren_params = vren.OpenCVFisheyeCameraModelParameters()

            vren_params.resolution = resolution.tolist()
            vren_params.shutter_type = SHUTTER_TYPE_MAP[shutter_type]
            vren_params.external_distortion_parameters = external_distortion_to_vren(external_distortion_parameters)

            vren_params.principal_point = principal_point.tolist()
            vren_params.focal_length = focal_length.tolist()
            vren_params.radial_coeffs = radial_coeffs.tolist()
            vren_params.max_angle = max_angle

            return vren_params

        case FThetaCameraModelParameters(
            resolution=resolution,
            shutter_type=shutter_type,
            external_distortion_parameters=external_distortion_parameters,
            principal_point=principal_point,
            reference_poly=reference_poly,
            pixeldist_to_angle_poly=pixeldist_to_angle_poly,
            angle_to_pixeldist_poly=angle_to_pixeldist_poly,
            max_angle=max_angle,
            linear_cde=linear_cde,
        ):
            vren_params = vren.FThetaCameraModelParameters()

            vren_params.resolution = resolution.tolist()
            vren_params.shutter_type = SHUTTER_TYPE_MAP[shutter_type]
            vren_params.external_distortion_parameters = external_distortion_to_vren(external_distortion_parameters)

            vren_params.principal_point = principal_point.tolist()
            match reference_poly:
                case FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE:
                    vren_params.reference_poly = vren.PolynomialType.PIXELDIST_TO_ANGLE
                case FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST:
                    vren_params.reference_poly = vren.PolynomialType.ANGLE_TO_PIXELDIST
            vren_params.pixeldist_to_angle_poly = pixeldist_to_angle_poly.tolist()
            vren_params.angle_to_pixeldist_poly = angle_to_pixeldist_poly.tolist()
            vren_params.max_angle = max_angle
            vren_params.linear_cde = linear_cde

            return vren_params

        case _:
            raise ValueError(f"Unsupported camera model parameters type {type(ncore_or_vren_camera_model_parameters)}")


def camera_rays_to_image_points(
    camera_model_parameters: CameraModelParametersUnion, cam_rays: torch.Tensor
) -> CameraModel.ImagePointsReturn:
    """Projects camera rays to image points using a cuda camera model"""

    assert cam_rays.dim() == 2 and cam_rays.shape[1] == 3 and cam_rays.dtype == torch.float32

    image_points, valid_flag = vren.camera_rays_to_image_points(
        to_vren(camera_model_parameters), cam_rays.cuda().contiguous()
    )

    return CameraModel.ImagePointsReturn(image_points=image_points, valid_flag=valid_flag)


def image_points_to_camera_rays(
    camera_model_parameters: CameraModelParametersUnion, image_points: torch.Tensor
) -> torch.Tensor:
    """Unprojects image points to camera rays using a cuda camera model"""

    assert image_points.dim() == 2 and image_points.shape[1] == 2 and image_points.dtype == torch.float32

    cam_rays = vren.image_points_to_camera_rays(to_vren(camera_model_parameters), image_points.cuda().contiguous())

    return cam_rays


def world_points_to_image_points_shutter_pose(
    camera_model_parameters: CameraModelParametersUnion,
    world_points: torch.Tensor,
    T_world_sensors: torch.Tensor,
    timestamps_us: torch.Tensor,
    return_all_projections=False,
) -> CameraModel.WorldPointsToImagePointsReturn:
    """Projects world points to image points using cuda camera model and rolling-shutter poses"""

    assert world_points.dim() == 2 and world_points.shape[1] == 3 and world_points.dtype == torch.float32
    assert T_world_sensors.shape == (2, 7) and T_world_sensors.dtype == torch.float32
    assert timestamps_us.shape == (2,) and timestamps_us.dtype == torch.int64

    # make sure parameters are living on the CPU, as they are passed as constant kernel invocation arguments
    # along with the camera model parameters (memory footprint is small - 2*7 float32 + 2*int64)
    rolling_shutter_parameters = vren.RollingShutterParameters()
    rolling_shutter_parameters.T_world_sensors = T_world_sensors.cpu().numpy().flatten().tolist()
    rolling_shutter_parameters.timestamps_us = timestamps_us.cpu().numpy().tolist()

    (
        image_points,
        valid_flag,
        point_timestamps_us,
        T_world_sensors,
    ) = vren.world_points_to_image_points_shutter_pose(
        to_vren(camera_model_parameters), rolling_shutter_parameters, world_points.cuda().contiguous()
    )

    valid_indices = torch.nonzero(valid_flag).squeeze(1)

    return CameraModel.WorldPointsToImagePointsReturn(
        image_points=image_points if return_all_projections else image_points[valid_indices, :],
        valid_indices=valid_indices,
        timestamps_us=point_timestamps_us[valid_indices],
        T_world_sensors=T_world_sensors[valid_indices],
    )


def image_points_to_world_rays_shutter_pose(
    camera_model_parameters: CameraModelParametersUnion,
    image_points: torch.Tensor,
    T_sensor_worlds: torch.Tensor,
    timestamps_us: torch.Tensor,
) -> CameraModel.WorldRaysReturn:
    assert image_points.dim() == 2 and image_points.shape[1] == 2 and image_points.dtype == torch.float32
    assert T_sensor_worlds.shape == (2, 7) and T_sensor_worlds.dtype == torch.float32
    assert timestamps_us.shape == (2,) and timestamps_us.dtype == torch.int64

    # Use tensor-based version that avoids GPU->CPU transfers
    # Ensure tensors are on GPU and contiguous (only transfer if needed)
    image_points_gpu = image_points.cuda().contiguous()
    device = image_points_gpu.device
    T_sensor_worlds_gpu = T_sensor_worlds.to(device=device, non_blocking=True).contiguous()
    timestamps_us_gpu = timestamps_us.to(device=device, non_blocking=True).contiguous()

    (
        world_rays,
        timestamps_us_out,
        T_sensor_worlds_out,
    ) = vren.image_points_to_world_points_shutter_pose(
        to_vren(camera_model_parameters), T_sensor_worlds_gpu, timestamps_us_gpu, image_points_gpu
    )

    return CameraModel.WorldRaysReturn(
        world_rays=world_rays,
        timestamps_us=timestamps_us_out,
        T_sensor_worlds=T_sensor_worlds_out,
    )
