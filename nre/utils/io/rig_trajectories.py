# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import warnings

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from ncore.data import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,  # TODO : add rolling shutter support
)
from nre.utils.io.utils import initialize_usd_stage, nre_tf_to_usd_tf
from nre.utils.types import ArtifactContents, HalfClosedInterval, NamedSerialized, NamedUSDStage, RigTrajectories


def add_default_opencv_pinhole_camera(stage: Usd.Stage, path: str, camera_name: str, extrinsics: Gf.Matrix4d):
    """
    Add a default fisheye camera to the stage from the following NCore test model :
    OpenCVPinholeCameraModelParameters(
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
    """
    camera_prim = stage.DefinePrim(f"{path}/{camera_name}", "Camera")
    camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set(Vt.Token("pinholeOpenCV"))
    camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f([0.001, 10000000]))
    #
    # intrinsics
    #
    # nominal resolution
    camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(1920)
    camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(1280)
    # principal point
    camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(935.1248081874216)
    camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(635.052474560227)
    # focal lenght
    camera_prim.CreateAttribute("openCVFx", Sdf.ValueTypeNames.Float).Set(2059.0471439559833)
    camera_prim.CreateAttribute("openCVFy", Sdf.ValueTypeNames.Float).Set(2059.0471439559833)
    # radial coeffs
    camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(0.04239636827428756)
    camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(-0.34165672675852826)
    camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(0)
    camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(0)
    camera_prim.CreateAttribute("fthetaPolyE", Sdf.ValueTypeNames.Float).Set(0)
    camera_prim.CreateAttribute("fthetaPolyF", Sdf.ValueTypeNames.Float).Set(0)
    # tangential coeffs
    camera_prim.CreateAttribute("p0", Sdf.ValueTypeNames.Float).Set(0.001805535524580487)
    camera_prim.CreateAttribute("p1", Sdf.ValueTypeNames.Float).Set(-0.00005530628187935031)
    # thin prism coeffs
    camera_prim.CreateAttribute("s0", Sdf.ValueTypeNames.Float).Set(0)
    camera_prim.CreateAttribute("s1", Sdf.ValueTypeNames.Float).Set(0)
    camera_prim.CreateAttribute("s2", Sdf.ValueTypeNames.Float).Set(0)
    camera_prim.CreateAttribute("s3", Sdf.ValueTypeNames.Float).Set(0)
    #
    # extrinsics
    #
    camera_transform = UsdGeom.Xformable(camera_prim).AddTransformOp()
    camera_transform.Set(extrinsics)
    #
    # visibility
    #
    UsdGeom.Imageable(camera_prim).CreateVisibilityAttr().Set("invisible")


def add_default_opencv_fisheye_camera(stage: Usd.Stage, path: str, camera_name: str, extrinsics: Gf.Matrix4d):
    """
    Add a default fisheye camera to the stage from the following NCore test model :
    OpenCVFisheyeCameraModelParameters(
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
    """
    camera_prim = stage.DefinePrim(f"{path}/{camera_name}", "Camera")
    camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set(Vt.Token("fisheyeOpenCV"))
    camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f([0.001, 10000000]))
    #
    # intrinsics
    #
    # nominal resolution
    camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(3840)
    camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(2160)
    # principal point
    camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(1928.184506)
    camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(1083.862789)
    # focal lenght
    camera_prim.CreateAttribute("openCVFx", Sdf.ValueTypeNames.Float).Set(1913.76478)
    camera_prim.CreateAttribute("openCVFy", Sdf.ValueTypeNames.Float).Set(1913.99708)
    # radial coeffs
    camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(-0.030093122)
    camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(-0.005103817)
    camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(-0.000849622)
    camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(0.001079542)
    # max FoV
    camera_prim.CreateAttribute("fthetaMaxFov", Sdf.ValueTypeNames.Float).Set(140)
    #
    # extrinsics
    #
    camera_transform = UsdGeom.Xformable(camera_prim).AddTransformOp()
    camera_transform.Set(extrinsics)
    #
    # visibility
    #
    UsdGeom.Imageable(camera_prim).CreateVisibilityAttr().Set("invisible")


def rig_trajectories_time_range(rig_trajectories: RigTrajectories) -> HalfClosedInterval:
    """Returns time range as a half close interval, containing every frame of the rig trajectories associated with the instance."""
    rigs_time_ranges = [
        HalfClosedInterval.from_series(rig.T_rig_world_timestamps_us) for rig in rig_trajectories.rig_trajectories
    ]
    total_time_range = (
        HalfClosedInterval(
            min([range.start for range in rigs_time_ranges]), max([range.end for range in rigs_time_ranges])
        )
        if rigs_time_ranges
        else HalfClosedInterval(0, 0)
    )
    # 53 bits of precision, a bit less than 16 digits (15 digits => ~ 32 years in microseconds)
    double_floating_precision_limit = 1e15
    if total_time_range.length > double_floating_precision_limit:
        warnings.warn(
            f"RigTrajectories span a time range which may not be representable in double precision floating point : {total_time_range.length * 1e-06} seconds"
        )
    return total_time_range


def _add_neutral_exposure_attributes(camera_prim: Usd.Prim) -> None:
    """Add exposure attributes that result in exposure scale of 1.0.

    Using the Iray formula: exposureScale = (responsivity * iso * cameraShutter) / (100 * fStop^2)
    With these values: (1.0 * 100 * 1.0) / (100 * 1.0^2) = 1.0
    """
    camera_prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(0.0)
    camera_prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(1.0)
    camera_prim.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(100.0)
    camera_prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(1.0)
    camera_prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(1.0)


def serialize_rig_trajectories_usd(
    rig_trajectories: RigTrajectories,
    usd_timestamp_offset: int = 0,
    add_default_cameras: bool = True,
    force_no_exposure: bool = False,
) -> Usd.Stage:
    # Get the rig transforms and camera sensor parameters
    rig_transforms: dict[str, dict[float, Gf.Matrix4d]] = {}
    camera_params: dict[str, dict] = {}
    camera_usd = nre_tf_to_usd_tf([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # invert Y/Z axis
    camera_calibrations = rig_trajectories.camera_calibrations
    for i, rig_trajectory in enumerate(rig_trajectories.rig_trajectories):
        rig_name = f"sensor_rig_{i}"
        rig_transforms[rig_name] = {}
        T_rig_timestamps_us = rig_trajectory.T_rig_world_timestamps_us.numpy()  # timestamp in microseconds
        T_rig_transform = rig_trajectory.T_rig_worlds.numpy().astype(np.double)  # rig to world transform
        assert T_rig_timestamps_us.shape[0] == T_rig_transform.shape[0]
        for t in range(T_rig_timestamps_us.shape[0]):
            rig_transforms[rig_name][T_rig_timestamps_us[t]] = nre_tf_to_usd_tf(T_rig_transform[t])

        for camera_unique_name in rig_trajectory.cameras_frame_timestamps_us:
            camera_data = camera_calibrations[camera_unique_name]
            camera_params[camera_data.logical_sensor_name] = {
                "rig_name": rig_name,
                "extrinsics": camera_usd * nre_tf_to_usd_tf(camera_data.T_sensor_rig.numpy().astype(np.double)),
                "intrinsics": camera_data.camera_model_parameters,
            }

    stage = initialize_usd_stage()

    # Define an xform containing the rig transforms
    usd_time_code_per_second = stage.GetTimeCodesPerSecond()
    usd_timestamp_scale = usd_time_code_per_second * 1e-06
    usd_start_time_code = np.inf
    usd_end_time_code = 0.0
    for rig_name, rig_transform in rig_transforms.items():
        rig_prim = stage.DefinePrim(f"/World/{rig_name}", "Xform")
        rig_xform = UsdGeom.Xformable(rig_prim)
        rig_transform_op = rig_xform.AddTransformOp()
        for timestamps_us, transform in rig_transform.items():
            usd_time_code = usd_timestamp_scale * (timestamps_us - usd_timestamp_offset)
            rig_transform_op.Set(transform, usd_time_code)
            usd_start_time_code = min(usd_start_time_code, usd_time_code)
            usd_end_time_code = max(usd_end_time_code, usd_time_code)

    # time related metadata
    if usd_start_time_code <= usd_end_time_code:
        stage.SetMetadata("startTimeCode", usd_start_time_code)
        stage.SetMetadata("endTimeCode", usd_end_time_code)
    stage.SetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec", usd_timestamp_offset)

    # Define camera prims
    for name, params in camera_params.items():
        camera_model = params["intrinsics"]
        camera_prim = stage.DefinePrim(f"/World/{params['rig_name']}/{name}", "Camera")
        camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f([0.001, 10000000]))
        match camera_model:
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
                camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set(
                    Vt.Token("pinholeOpenCV")
                )
                # nominal resolution
                resolution_list = resolution.tolist()
                camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(resolution_list[0])
                camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(resolution_list[1])
                # principal point
                principal_point_list = principal_point.tolist()
                camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(principal_point_list[0])
                camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(principal_point_list[1])
                # focal length
                focal_length_list = focal_length.tolist()
                camera_prim.CreateAttribute("openCVFx", Sdf.ValueTypeNames.Float).Set(focal_length_list[0])
                camera_prim.CreateAttribute("openCVFy", Sdf.ValueTypeNames.Float).Set(focal_length_list[1])
                # radial coeffs
                radial_coeffs_list = radial_coeffs.tolist()
                camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[0])
                camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[1])
                camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[2])
                camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[3])
                camera_prim.CreateAttribute("fthetaPolyE", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[4])
                camera_prim.CreateAttribute("fthetaPolyF", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[5])
                # tangential coeffs
                tangential_coeffs_list = tangential_coeffs.tolist()
                camera_prim.CreateAttribute("p0", Sdf.ValueTypeNames.Float).Set(tangential_coeffs_list[0])
                camera_prim.CreateAttribute("p1", Sdf.ValueTypeNames.Float).Set(tangential_coeffs_list[1])
                # thin prism coeffs
                thin_prism_coeffs_list = thin_prism_coeffs.tolist()
                camera_prim.CreateAttribute("s0", Sdf.ValueTypeNames.Float).Set(thin_prism_coeffs_list[0])
                camera_prim.CreateAttribute("s1", Sdf.ValueTypeNames.Float).Set(thin_prism_coeffs_list[1])
                camera_prim.CreateAttribute("s2", Sdf.ValueTypeNames.Float).Set(thin_prism_coeffs_list[2])

            case OpenCVFisheyeCameraModelParameters(
                resolution=resolution,
                shutter_type=shutter_type,
                external_distortion_parameters=external_distortion_parameters,
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=radial_coeffs,
                max_angle=max_angle,
            ):
                camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set(
                    Vt.Token("fisheyeOpenCV")
                )
                # nominal resolution
                resolution_list = resolution.tolist()
                camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(resolution_list[0])
                camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(resolution_list[1])
                # principal point
                principal_point_list = principal_point.tolist()
                camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(principal_point_list[0])
                camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(principal_point_list[1])
                # focal lenght
                focal_length_list = focal_length.tolist()
                camera_prim.CreateAttribute("openCVFx", Sdf.ValueTypeNames.Float).Set(focal_length_list[0])
                camera_prim.CreateAttribute("openCVFy", Sdf.ValueTypeNames.Float).Set(focal_length_list[1])
                # radial coeffs
                radial_coeffs_list = radial_coeffs.tolist()
                camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[0])
                camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[1])
                camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[2])
                camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(radial_coeffs_list[3])
                # max FoV
                camera_prim.CreateAttribute("fthetaMaxFov", Sdf.ValueTypeNames.Float).Set(2.0 * np.rad2deg(max_angle))

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
                camera_prim.CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set(
                    Vt.Token("fisheyePolynomial")
                )
                # nominal resolution
                resolution_list = resolution.tolist()
                camera_prim.CreateAttribute("fthetaWidth", Sdf.ValueTypeNames.Float).Set(resolution_list[0])
                camera_prim.CreateAttribute("fthetaHeight", Sdf.ValueTypeNames.Float).Set(resolution_list[1])
                # principal point
                principal_point_list = principal_point.tolist()
                camera_prim.CreateAttribute("fthetaCx", Sdf.ValueTypeNames.Float).Set(principal_point_list[0])
                camera_prim.CreateAttribute("fthetaCy", Sdf.ValueTypeNames.Float).Set(principal_point_list[1])
                # pixeldist to angle poly
                pixeldist_to_angle_poly_list = pixeldist_to_angle_poly.tolist()
                camera_prim.CreateAttribute("fthetaPolyA", Sdf.ValueTypeNames.Float).Set(
                    pixeldist_to_angle_poly_list[0]
                )
                camera_prim.CreateAttribute("fthetaPolyB", Sdf.ValueTypeNames.Float).Set(
                    pixeldist_to_angle_poly_list[1]
                )
                camera_prim.CreateAttribute("fthetaPolyC", Sdf.ValueTypeNames.Float).Set(
                    pixeldist_to_angle_poly_list[2]
                )
                camera_prim.CreateAttribute("fthetaPolyD", Sdf.ValueTypeNames.Float).Set(
                    pixeldist_to_angle_poly_list[3]
                )
                camera_prim.CreateAttribute("fthetaPolyE", Sdf.ValueTypeNames.Float).Set(
                    pixeldist_to_angle_poly_list[4]
                )
                camera_prim.CreateAttribute("fthetaPolyF", Sdf.ValueTypeNames.Float).Set(
                    pixeldist_to_angle_poly_list[5]
                )
                # max angle
                camera_prim.CreateAttribute("fthetaMaxFov", Sdf.ValueTypeNames.Float).Set(2.0 * np.rad2deg(max_angle))

            case _:
                # unsupported camera model : fallback to default
                camera_prim.GetFocalLengthAttr().Set(value=24.0)

        camera_transform = UsdGeom.Xformable(camera_prim).AddTransformOp()
        camera_transform.Set(params["extrinsics"])

        # Add neutral exposure attributes when force_no_exposure is True
        if force_no_exposure:
            _add_neutral_exposure_attributes(camera_prim)

        if add_default_cameras:
            add_default_opencv_pinhole_camera(
                stage,
                f"/World/{params['rig_name']}",
                f"{name}_ocv_pinhole",
                params["extrinsics"],
            )
            add_default_opencv_fisheye_camera(
                stage,
                f"/World/{params['rig_name']}",
                f"{name}_ocv_fisheye",
                params["extrinsics"],
            )

        # Hide camera from view since it will be composed with the NeRF by default
        imageable = UsdGeom.Imageable(camera_prim)
        imageable.CreateVisibilityAttr().Set("invisible")

    return stage


def serialize_rig_trajectories(
    rig_trajectories: RigTrajectories,
    filename: str = "rig_trajectories",
    formats: list[str] = ["json", "usda"],
    usd_timestamp_offset: int = 0,
    add_default_cameras: bool = False,
    force_no_exposure: bool = False,
) -> ArtifactContents:
    res: ArtifactContents = []
    for file_format in formats:
        filename_with_suffix = filename + "." + file_format
        match file_format:
            case "json":
                res.append(
                    NamedSerialized(
                        filename=filename_with_suffix, serialized=json.dumps(rig_trajectories.to_dict(), indent=4)
                    )
                )
            case "usd" | "usda":
                res.append(
                    NamedUSDStage(
                        filename=filename_with_suffix,
                        stage=serialize_rig_trajectories_usd(
                            rig_trajectories,
                            usd_timestamp_offset=usd_timestamp_offset,
                            add_default_cameras=add_default_cameras,
                            force_no_exposure=force_no_exposure,
                        ),
                    )
                )
            case _:
                raise ValueError(f"The following rig trajectory format is not supported: {file_format}")
    return res
