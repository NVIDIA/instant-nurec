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
import logging
import math
import os

from pathlib import Path
from typing import cast

import click
import numpy as np
import tqdm

from PIL import Image

import ncore.data

from internal.scripts.experimental.io import save_pc_dat
from ncore.impl.common.transformations import transform_point_cloud
from ncore.impl.data.util import padded_index_string
from nre.config.dataset import NCoreDatasetConfig
from nre.config.parse import parse_untyped_config
from nre.datasets.ncore import NCOREDataSource


# Rotation of NCORE camera frame to NGP camera frame
R_NCORE_NGP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


def pose_ncore_to_ngp(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3] @ R_NCORE_NGP
    T[:3, :3] = R
    return T


# Encodes the rolling shutter parameters [a,b,c] such y = a + b*x + c*y (Waymo rolling shutter is column wise)
RS_DIR_TO_NGP = {
    "ROLLING_TOP_TO_BOTTOM": np.array([0.0, 0.0, 1.0]),
    "ROLLING_LEFT_TO_RIGHT": np.array([0.0, 1.0, 0.0]),
    "ROLLING_BOTTOM_TO_TOP": np.array([1.0, 0.0, -1.0]),
    "ROLLING_RIGHT_TO_LEFT": np.array([1.0, -1.0, 0.0]),
}


def camera_model_parameters_to_ngp(
    camera_id: str, camera_model_parameters: ncore.data.ConcreteCameraModelParametersUnion
) -> dict:
    match camera_model_parameters:
        case ncore.data.FThetaCameraModelParameters(
            resolution=resolution,
            principal_point=principal_point,
            reference_poly=reference_poly,
            pixeldist_to_angle_poly=bw_poly,
            angle_to_pixeldist_poly=fw_poly,
            shutter_type=shutter_type,
        ):
            # This is not really the focal length, only a crude approximation, but NGP expects some value for fov (not used in training)
            focal_length = fw_poly[1]
            fov_angle_x = math.atan2(resolution[0], focal_length * 2) * 2

            assert len(bw_poly) == 6, (
                "Update polynomial export in case the internal order of distortion polynomials changed"
            )
            ret = {
                "w": int(resolution[0]),
                "h": int(resolution[1]),
                "cx": float(principal_point[0]),
                "cy": float(principal_point[1]),
                "ftheta_p0": float(bw_poly[0]),
                "ftheta_p1": float(bw_poly[1]),
                "ftheta_p2": float(bw_poly[2]),
                "ftheta_p3": float(bw_poly[3]),
                "ftheta_p4": float(bw_poly[4]),
                # Note: we are already outputting this higher-order coefficient although NGP might not use it yet
                "ftheta_p5": float(bw_poly[5]),
                "ftheta_f0": float(fw_poly[0]),
                "ftheta_f1": float(fw_poly[1]),
                "ftheta_f2": float(fw_poly[2]),
                "ftheta_f3": float(fw_poly[3]),
                "ftheta_f4": float(fw_poly[4]),
                "ftheta_f5": float(fw_poly[5]),
                "reference_poly": reference_poly.name,
                "camera_angle_x": fov_angle_x,
            }

            # Extract the rolling shutter parameters representing y = a + b*x + c*y (skip for GLOBAL shutter)
            if shutter_type.name in RS_DIR_TO_NGP:
                ret |= {"rolling_shutter": RS_DIR_TO_NGP[shutter_type.name].tolist()}

            return ret

        case ncore.data.OpenCVPinholeCameraModelParameters() as pinhole:
            # Get the focal length and compute the angular field of view
            fov_angle_x = math.atan(pinhole.resolution[0] / (pinhole.focal_length[0] * 2)) * 2
            fov_angle_y = math.atan(pinhole.resolution[1] / (pinhole.focal_length[1] * 2)) * 2

            if not np.isclose(pinhole.radial_coeffs[2], 0).all():
                logging.warn(
                    f"Pinhole camera model of {camera_id} has non-zero radial distortion coefficient k3, "
                    "which might not be supported by NGP yet - exporting anyway"
                )

            if not np.isclose(pinhole.radial_coeffs[3:], 0).all():
                logging.warn(
                    f"Pinhole camera model of {camera_id} has non-zero rational radial distortion coefficients [k4,k5,k6], "
                    "which might not be supported by NGP yet - exporting anyway"
                )

            if not np.isclose(pinhole.thin_prism_coeffs, 0).all():
                logging.warn(
                    f"Pinhole camera model of {camera_id} has non-zero thin-prism distortion coefficients [s1,s2,s3,s4], "
                    "which might not be supported by NGP yet - exporting anyway"
                )

            ret = {
                "w": int(pinhole.resolution[0]),
                "h": int(pinhole.resolution[1]),
                "cx": float(pinhole.principal_point[0]),
                "cy": float(pinhole.principal_point[1]),
                "k1": float(pinhole.radial_coeffs[0]),
                "k2": float(pinhole.radial_coeffs[1]),
                "p1": float(pinhole.tangential_coeffs[0]),
                "p2": float(pinhole.tangential_coeffs[1]),
                # Note: we are already outputting these higher-order / rational radial distortion and thin-prism coefficients, although NGP might not use it yet
                "k3": float(pinhole.radial_coeffs[2]),
                "k4": float(pinhole.radial_coeffs[3]),
                "k5": float(pinhole.radial_coeffs[4]),
                "k6": float(pinhole.radial_coeffs[5]),
                "s1": float(pinhole.thin_prism_coeffs[0]),
                "s2": float(pinhole.thin_prism_coeffs[1]),
                "s3": float(pinhole.thin_prism_coeffs[2]),
                "s4": float(pinhole.thin_prism_coeffs[3]),
                "camera_angle_x": fov_angle_x,
                "camera_angle_y": fov_angle_y,
            }

            # Extract the rolling shutter parameters representing y = a + b*x + c*y (skip for GLOBAL shutter)
            if pinhole.shutter_type.name in RS_DIR_TO_NGP:
                ret |= {"rolling_shutter": RS_DIR_TO_NGP[pinhole.shutter_type.name].tolist()}

            return ret

        case ncore.data.OpenCVFisheyeCameraModelParameters() as fisheye:
            # Get the focal length and compute the angular field of view
            fov_angle_x = math.atan(fisheye.resolution[0] / (fisheye.focal_length[0] * 2)) * 2
            fov_angle_y = math.atan(fisheye.resolution[1] / (fisheye.focal_length[1] * 2)) * 2

            ret = {
                "w": int(fisheye.resolution[0]),
                "h": int(fisheye.resolution[1]),
                "cx": float(fisheye.principal_point[0]),
                "cy": float(fisheye.principal_point[1]),
                "k1": float(fisheye.radial_coeffs[0]),
                "k2": float(fisheye.radial_coeffs[1]),
                "k3": float(fisheye.radial_coeffs[2]),
                "k4": float(fisheye.radial_coeffs[3]),
                "is_fisheye": True,
                "camera_angle_x": fov_angle_x,
                "camera_angle_y": fov_angle_y,
            }

            # Extract the rolling shutter parameters representing y = a + b*x + c*y (skip for GLOBAL shutter)
            if fisheye.shutter_type.name in RS_DIR_TO_NGP:
                ret |= {"rolling_shutter": RS_DIR_TO_NGP[fisheye.shutter_type.name].tolist()}

            return ret

        case _:
            raise TypeError(
                f"unsupported camera model type {type(camera_model_parameters)}, currently supporting Ftheta/OpenCV-Pinhole/OpenCV-Fisheye only"
            )


@click.command()
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    default="tests/ncore_ds",
    required=True,
)
@click.option(
    "--experiment-name",
    type=str,
    help="NGP experiment name",
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory",
    required=True,
)
@click.option(
    "--frame-id-format",
    type=click.Choice(["null-index", "frame-index", "timestamp"], case_sensitive=False),
    help="Frame id enumeration format (absolute 'timestamp', 'null-index' unconditionally starting at 0, or the source sensor's 'frame-index')",
    default="timestamp",
)
@click.argument("hydra-args", nargs=-1)
def ncore_to_ngp(
    config_name: str,
    experiment_name: str,
    output_dir: str,
    frame_id_format: str,
    hydra_args: list[str],
) -> None:
    """Exports a NCore dataset to NGP training format"""
    config = parse_untyped_config(config_name=config_name, hydra_args=hydra_args)

    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    ncore_datasource = NCOREDataSource(cast(NCoreDatasetConfig, config.dataset), config.sensor.lidar_models)

    # Create the output dir if it doesn't exist
    (output_dir_path := Path(output_dir)).mkdir(parents=True, exist_ok=True)

    (output_configs_dir_path := output_dir_path / "ngp_configs" / experiment_name).mkdir(parents=True, exist_ok=True)

    # Generate table of 256 random RBG colors
    np.random.seed(0)
    colors_256 = np.random.choice(np.array(range(256), dtype=np.uint8), size=3 * 256).reshape(256, 3)

    def get_frame_id(i: int, frame_export: NCOREDataSource.CameraFrameExport | NCOREDataSource.LidarFrameExport) -> str:
        frame_id: str
        match frame_id_format:
            case "timestamp":
                # Use unique timestamp as frame-id
                frame_id = str(frame_export.timestamp_end_us)
            case "null-index":
                # Use enumerated index as frame-id
                frame_id = padded_index_string(i)
            case "frame-index":
                # Use frame index as frame-id
                frame_id = padded_index_string(frame_export.frame_idx)
        return frame_id

    for camera_idx, camera_id in enumerate(ncore_datasource.camera_ids):
        logging.info(f"Processing camera '{camera_id}'")

        out_train = {
            "n_extra_learnable_dims": 32,
            "up": [0, 0, 1],
            "offset": (
                -ncore_datasource.world_to_nre.target_scale * ncore_datasource.world_to_nre.target_origin + 0.5
            ).tolist(),  # The 0.5 needs to be added to the offset as INGP centers their scenes at 0.5
            "scale": ncore_datasource.world_to_nre.target_scale,
            "max_bound": ncore_datasource.max_dist_m,
            "frames": [],
        }

        for i, frame_export in tqdm.tqdm(
            enumerate(ncore_datasource.export_camera_frames(camera_id)), desc=f"camera {camera_id}"
        ):
            (output_camera_path := output_dir_path / frame_export.sequence_id / camera_id).mkdir(
                parents=True, exist_ok=True
            )

            frame_id = get_frame_id(i, frame_export)

            # Output frame image
            frame_path = output_camera_path / Path(frame_id).with_suffix(
                f".{frame_export.image_data.get_encoded_image_format()}"
            )

            with open(frame_path, "wb") as frame_file:
                frame_file.write(frame_export.image_data.get_encoded_image_data())

            # Output frame dynamic_mask
            mask_path = output_camera_path / f"dynamic_mask_{frame_id}.png"
            Image.fromarray(np.logical_not(frame_export.valid_pixels_mask)).save(mask_path, bits=1, optimize=True)

            ngp_camera_model_parameters = camera_model_parameters_to_ngp(
                camera_id, frame_export.camera_model_parameters
            )

            out_frame = {
                "file_path": os.path.relpath(frame_path, output_configs_dir_path),
                "transform_matrix_start": pose_ncore_to_ngp(frame_export.T_sensor_to_world_start).tolist(),
                "transform_matrix_end": pose_ncore_to_ngp(frame_export.T_sensor_to_world_end).tolist(),
            } | ngp_camera_model_parameters

            out_train["frames"].append(out_frame)

            # Output colored track idx images, if enabled
            if (frame_track_idxs := frame_export.frame_track_idxs) is not None:
                tracks_path = output_camera_path / f"{frame_id}_tracks.png"

                frame_track_colors = np.zeros(frame_export.valid_pixels_mask.shape + (3,), dtype=np.uint8)
                frame_track_mask = frame_track_idxs >= 0
                frame_track_colors[frame_track_mask] = colors_256[
                    np.mod(frame_track_idxs[frame_track_mask], len(colors_256))
                ]

                Image.fromarray(frame_track_colors).save(tracks_path, optimize=True)

            # Output semantics, if enabled
            if (sem_seg_image := frame_export.sem_seg_image) is not None:
                sem_path = output_camera_path / f"{frame_id}_sem.png"
                sem_seg_image.save(sem_path)

                if "semantic" not in out_train:
                    # initialize semantic_meta with meta data of *first* frame, even though different sessions
                    # could have different meta data
                    assert (sem_seg_meta := frame_export.sem_seg_meta) is not None
                    out_train["semantic_meta"] = sem_seg_meta
                    out_train["semantic"] = []

                out_semantic = {
                    "file_path": os.path.relpath(sem_path, output_configs_dir_path),
                }

                out_train["semantic"].append(out_semantic)

            # Use the *first's* frames intrinsics as the "global" intrinsics for this camera, although the data could
            # come from different sessions / have different per-frame values
            if camera_idx == 0:
                out_train |= ngp_camera_model_parameters

        # Only a single camera configuration can contain lidar-data in NGP - export all lidar data along with first camera
        if camera_idx == 0:
            out_train["lidar"] = []

            for lidar_id in ncore_datasource.lidar_ids:
                logging.info(f"Processing lidar '{lidar_id}'")

                for i, frame_export in tqdm.tqdm(
                    enumerate(ncore_datasource.export_lidar_frames(lidar_id)), desc=f"lidar {lidar_id}"
                ):
                    (output_lidar_path := output_dir_path / frame_export.sequence_id / lidar_id).mkdir(
                        parents=True, exist_ok=True
                    )

                    frame_id = get_frame_id(i, frame_export)

                    T_sensor_to_world = frame_export.T_sensor_to_world_end

                    # Load relevant frame data for ray structure
                    xyz_s = transform_point_cloud(frame_export.xyz_s, T_sensor_to_world)
                    xyz_e = transform_point_cloud(frame_export.xyz_e, T_sensor_to_world)
                    dist = np.linalg.norm(xyz_s - xyz_e, axis=1)  # N x 1
                    intensity = frame_export.intensity
                    dynamic_flag = frame_export.dynamic_flag
                    if dynamic_flag is None:
                        # dynamic flag data is not available (indicated by -1 values)
                        dynamic_flag = np.full_like(intensity, -1, dtype=np.int8)

                    # Assemble full point-cloud ray structure
                    point_cloud = np.column_stack((xyz_s, xyz_e, dist, intensity, dynamic_flag))

                    # Serialize point cloud
                    frame_path = output_lidar_path / Path(frame_id).with_suffix(".dat")
                    save_pc_dat(str(frame_path), point_cloud)

                    out_train["lidar"].append({"file_path": os.path.relpath(frame_path, output_configs_dir_path)})

        output_configs_path = output_configs_dir_path / f"{camera_id}_train.json"
        with open(output_configs_path, "w") as outfile:
            json.dump(out_train, outfile, indent=4, sort_keys=True)


if __name__ == "__main__":
    ncore_to_ngp(show_default=True)
