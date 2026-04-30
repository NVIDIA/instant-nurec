# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import copy
import io
import logging

from dataclasses import replace
from pathlib import Path

import click
import hydra
import numpy as np
import torch

from PIL import Image
from tqdm import trange

from ncore.sensors import CameraModel
from ncore_internal.data.v3 import ShardDataWriter
from nre.artifact.artifact import Artifact
from nre.difix.model import DifixModel, DifixModelFactory
from nre.render import RenderableModel
from nre.render.render import PoseRange
from nre.utils.files import local_temp_file, parse_universal_path
from nre.utils.geometry import pose_offsets_to_se3, tquat_to_se3_matrix
from nre.utils.io.export.render import SensorCalibProviderFromRigTrajectories
from nre.utils.misc import unpack_optional
from nre.utils.types import RigTrajectories
from nre.utils.visualize import save_video
from nre.viewer.viewpoint import to_simple_pinhole


logger = logging.getLogger(__name__)


def alter_rig_trajectories(
    rig_trajectories: RigTrajectories, rig_offset_se3: np.ndarray, camera_offsets_se3: dict[str, np.ndarray]
) -> RigTrajectories:
    """Compute a new rig trajectory with the given offsets"""

    new_rig_trajectories = copy.deepcopy(rig_trajectories)
    cameras_to_alter: set[str] = set(camera_offsets_se3.keys())

    for unique_camera_id in new_rig_trajectories.camera_calibrations.keys():
        camera_id, _ = unique_camera_id.split("@")
        if camera_id in camera_offsets_se3:
            new_rig_trajectories.camera_calibrations[unique_camera_id].T_sensor_rig = (
                new_rig_trajectories.camera_calibrations[unique_camera_id].T_sensor_rig
                @ torch.from_numpy(camera_offsets_se3[camera_id])
            )
            cameras_to_alter.remove(camera_id)

    if len(cameras_to_alter) > 0:
        raise ValueError(f"Some cameras were not found in the data rig trajectories: {cameras_to_alter}")

    for trajectory in new_rig_trajectories.rig_trajectories:
        trajectory.T_rig_worlds = trajectory.T_rig_worlds @ torch.from_numpy(rig_offset_se3).double()
        trajectory.cameras_frame_T_rig_worlds = None  # For consistency

    return new_rig_trajectories


@click.command()
@click.option(
    "--artifact-path",
    type=str,
    help="Path to an USDZ artifact file (local or S3, e.g. last.usdz or s3://bucket/key.usdz)",
    default=None,
    required=True,
)
@click.option(
    "--output-path",
    type=str,
    help="Path to the output NCore data directory",
    default=None,
    required=True,
)
@click.option(
    "--rig-translation-offset",
    nargs=3,
    type=float,
    help="Translation offsets (front,left,up) in meters in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--rig-rotation-offset",
    nargs=3,
    type=float,
    help="Rotation offsets (yaw, -roll, -pitch) in degrees in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--camera-translation-offset",
    "camera_translation_offsets",
    type=(str, float, float, float),
    multiple=True,
    help="Camera translation offsets (camera_id, right, down, forward) in meters in camera space to be applied to the camera prior to rendering.",
    default=[],
)
@click.option(
    "--camera-rotation-offset",
    "camera_rotation_offsets",
    type=(str, float, float, float, bool),
    multiple=True,
    help="Camera rotation offsets (camera_id, roll, pitch[-up,+down], -yaw[+right,-left], rotation_first) in degrees in camera space to be applied to the camera prior to rendering.",
    default=[],
)
@click.option(
    "--camera-force-pinhole",
    "camera_force_pinholes",
    type=(str, bool),
    multiple=True,
    help="Camera force pinhole (camera_id, force_pinhole) to be applied to the camera prior to rendering.",
    default=[],
)
@click.option(
    "--difix/--no-difix",
    "difix_enabled",
    help="Apply difix to the rendered frames",
    default=True,
)
@click.option(
    "--difix-config",
    type=str,
    default="configs/difix/cosmos_difix.yaml",
    help="Path to the difix config file. If empty no difix will be applied.",
    required=False,
)
@click.option(
    "--visualize/--no-visualize",
    help="Visualize the rendered frames",
    default=False,
)
def nvs_ncore(
    artifact_path: str,
    output_path: str,
    rig_translation_offset: tuple[float, float, float],
    rig_rotation_offset: tuple[float, float, float],
    camera_translation_offsets: list[tuple[str, float, float, float]],
    camera_rotation_offsets: list[tuple[str, float, float, float, bool]],
    camera_force_pinholes: list[tuple[str, bool]],
    difix_enabled: bool,
    difix_config: str,
    visualize: bool,
) -> None:
    """Generate NCore data from an USDZ artifact file exported from training. artifact-path may be local or S3 (e.g. s3://bucket/key.usdz)."""
    logging.basicConfig(level=logging.INFO)

    if not output_path.endswith(".zarr.itar"):
        raise AssertionError(f"The script only supports itar-based zarr format as outputs, but got {output_path}")

    # Parse and load the difix model
    difix_model: DifixModel | None = None
    if difix_enabled:
        difix_config_path = Path(difix_config).resolve()
        with hydra.initialize_config_dir(config_dir=str(difix_config_path.parent), version_base=None):
            difix_config_dict = hydra.compose(config_name=difix_config_path.name)
        difix_model = DifixModelFactory.get(
            difix_config_dict.model_url,
            difix_config_dict.cache_dir,
            difix_config_dict.model_filename,
            tuple(difix_config_dict.model_resolution),
        )

    upath = parse_universal_path(artifact_path)
    with local_temp_file(upath) as path:
        _run_nvs_ncore(
            artifact_path=path,
            output_path=output_path,
            rig_translation_offset=rig_translation_offset,
            rig_rotation_offset=rig_rotation_offset,
            camera_translation_offsets=camera_translation_offsets,
            camera_rotation_offsets=camera_rotation_offsets,
            camera_force_pinholes=camera_force_pinholes,
            difix_model=difix_model,
            visualize=visualize,
        )


def _run_nvs_ncore(
    artifact_path: Path,
    output_path: str,
    rig_translation_offset: tuple[float, float, float],
    rig_rotation_offset: tuple[float, float, float],
    camera_translation_offsets: list[tuple[str, float, float, float]],
    camera_rotation_offsets: list[tuple[str, float, float, float, bool]],
    camera_force_pinholes: list[tuple[str, bool]],
    difix_model: DifixModel | None,
    visualize: bool,
) -> None:
    """Generate NCore data from a local USDZ artifact path."""
    artifact = Artifact(artifact_path)

    logger.info(f"Loaded artifact with scene id '{artifact.scene_id}', now constructing renderable model")
    renderable = RenderableModel.load_from_artifact(artifact, enable_nrend=False)

    source_rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

    rig_offset_se3 = pose_offsets_to_se3(rig_translation_offset, rig_rotation_offset)
    camera_translation_offsets_map = {camera_id: (tx, ty, tz) for camera_id, tx, ty, tz in camera_translation_offsets}
    camera_rotation_offsets_map = {
        camera_id: (yaw, roll, pitch, rotation_first)
        for camera_id, yaw, roll, pitch, rotation_first in camera_rotation_offsets
    }
    camera_offsets_se3: dict[str, np.ndarray] = {}
    for camera_id in set(camera_translation_offsets_map.keys()) | set(camera_rotation_offsets_map.keys()):
        t_offset = camera_translation_offsets_map.get(camera_id, (0.0, 0.0, 0.0))
        r_offset = camera_rotation_offsets_map.get(camera_id, (0.0, 0.0, 0.0, False))
        camera_offsets_se3[camera_id] = pose_offsets_to_se3(t_offset, r_offset[:3], rotation_first=r_offset[3])
    force_pinhole_cameras: set[str] = set(
        camera_id for camera_id, force_pinhole in camera_force_pinholes if force_pinhole
    )

    logger.info(f"Applying sensor offsets to cameras: {camera_offsets_se3.keys()}")
    target_rig_trajectories = alter_rig_trajectories(source_rig_trajectories, rig_offset_se3, camera_offsets_se3)

    calib_provider = SensorCalibProviderFromRigTrajectories(
        target_rig_trajectories,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        interpolate_poses=True,  # Use interpolated rig for sensors
        rolling_shutter_duration=None,
    )

    # Query cameras from the scene and select an existing trajectory and camera.
    target_camera_ids = calib_provider.get_available_camera_ids()
    target_camera_ids = [camera_id for camera_id in target_camera_ids if camera_id in camera_offsets_se3]
    if len(target_camera_ids) == 0:
        raise ValueError("No output cameras is specified")
    logger.info("Will render cameras:\n    " + "\n    ".join(target_camera_ids))

    # Initiate ncore writer (we only compute the "core" data)
    output_path_base = Path(output_path.replace(".zarr.itar", ""))
    ncore_writer = ShardDataWriter(
        output_path_base.parent,
        output_path_base.name,
        target_camera_ids,
        [],
        [],
        "nvs-ncore-calibration-type",
        "nvs-ncore-egomotion-type",
        artifact.data_info["sequence_id"],
        {},
        0,
        1,
        False,
    )
    ncore_writer.store_poses(calib_provider.get_ncore_v3_poses())

    for camera_id in target_camera_ids:
        camera_calib = calib_provider.get_camera_calibration(camera_id)
        if camera_id in force_pinhole_cameras:
            camera_calib = replace(
                camera_calib,
                camera_model_parameters=to_simple_pinhole(
                    CameraModel.from_parameters(camera_calib.camera_model_parameters)
                ).get_parameters(),
            )
        camera_width, camera_height = camera_calib.camera_model_parameters.resolution.astype(int).tolist()

        num_frames = calib_provider.get_num_camera_frames(camera_id)
        camera_pose_ranges: list[PoseRange] = []
        for frame_idx in range(num_frames):
            camera_pose_ranges.append(calib_provider.get_camera_view_pose(camera_id, frame_idx))

        T_camera_rig = camera_calib.T_sensor_rig.cpu().numpy()
        T_rig_camera = np.linalg.inv(T_camera_rig)
        ncore_writer.store_camera_meta(
            camera_id=camera_id,
            frame_timestamps_us=np.array(
                [pose_range.end_timestamp_us for pose_range in camera_pose_ranges], dtype=np.uint64
            ),
            T_sensor_rig=T_camera_rig,
            camera_model_parameters=camera_calib.camera_model_parameters,
            mask_image=None,  # Rendered frames are not masked
            generic_meta_data={},
        )

        rendered_frames: list[np.ndarray] = []
        for frame_idx in trange(num_frames, desc=f"Rendering camera {camera_id}"):
            camera_to_world: PoseRange = camera_pose_ranges[frame_idx]

            camera_frame = renderable.render_camera_frame(
                camera_intrinsics=camera_calib.camera_model_parameters,
                camera_to_world=camera_to_world,
                resolution=(camera_width, camera_height),
                unique_sensor_idx=camera_calib.unique_sensor_idx,
                unique_frame_idx=calib_provider.get_unique_frame_index(camera_id, frame_idx),
            )

            if difix_model is not None:
                camera_frame.color_image = difix_model.forward(
                    unpack_optional(camera_frame.color_image).reshape(-1, 3),
                    torch.Size([camera_height, camera_width]),
                    True,
                ).reshape(camera_height, camera_width, 3)

            assert camera_frame.color_image is not None
            # Convert to uint8 - render_camera_frame returns a float32 tensor
            camera_frame.color_image = (camera_frame.color_image * 255).clamp(0, 255).to(torch.uint8)
            rgb_image = Image.fromarray(camera_frame.color_image.cpu().numpy())
            rgb_image_data_fp = io.BytesIO()
            rgb_image.save(rgb_image_data_fp, format="jpeg", quality=93)
            rgb_image_data = rgb_image_data_fp.getvalue()

            T_rig_worlds_start = (
                tquat_to_se3_matrix(camera_to_world.start_pose_tquat_sensor_world).numpy() @ T_rig_camera
            )
            T_rig_worlds_end = tquat_to_se3_matrix(camera_to_world.end_pose_tquat_sensor_world).numpy() @ T_rig_camera

            ncore_writer.store_camera_frame(
                camera_id=camera_id,
                continuous_frame_index=frame_idx,
                image_file_binary_data=rgb_image_data,
                image_file_format="jpeg",
                T_rig_worlds=np.stack([T_rig_worlds_start, T_rig_worlds_end], axis=0),
                timestamps_us=np.array(
                    [camera_to_world.start_timestamp_us, camera_to_world.end_timestamp_us], dtype=np.uint64
                ),
                generic_data={},
                generic_meta_data={},
            )

            if visualize:
                rendered_frames.append(np.array(rgb_image))

        # Save the rendered frames if visualization is enabled
        if len(rendered_frames) > 0:
            visualize_video_path = output_path_base.parent / "visualization" / f"{camera_id}.mp4"
            visualize_video_path.parent.mkdir(parents=True, exist_ok=True)
            save_video(str(visualize_video_path), rendered_frames, fps=30)

    ncore_writer.finalize()


if __name__ == "__main__":
    nvs_ncore()
