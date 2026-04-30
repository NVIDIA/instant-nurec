# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import json
import logging
import os
import time

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import aiofiles
import click
import lietorch as lt
import numpy as np
import point_cloud_utils as pcu

from scipy.spatial.transform import Rotation as R

import ncore.impl.common.transformations as ncore_transformations
import nre.utils.cli as cli

from nre.artifact import Artifact
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.grpc.protos.common_pb2 import AABB
from nre.grpc.protos.common_pb2 import Empty as EmptyRequest
from nre.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableEgoMasksReturn,
    CameraSpec,
    DynamicObject,
    DynamicObjectTrack,
    EditAssetsRequest,
    EditAssetsResponse,
    EgoMaskId,
    ExternalAssetObjectsRequest,
    ExternalAssetObjectsReturn,
    ImageFormat,
    LidarDeviceType,
    LidarRenderFilter,
    LidarRenderRequest,
    LidarRenderReturn,
    LidarSpec,
    PosePair,
    ReplaceAssetAction,
    RestoreModelParametersRequest,
    RGBRenderRequest,
    RGBRenderReturn,
)
from nre.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub
from nre.grpc.serve import (
    actor_tracks_to_grpc,
    grpc,
    grpc_pose_to_torch_se3,
    se3_to_grpc_pose,
    tquat_to_grpc_pose,
)
from nre.render.actors import ActorTracks
from nre.utils.cli import SettingsCollector
from nre.utils.geometry import pose_offsets_to_se3, se3_matrix_to_se3, se3_matrix_to_tquat
from nre.utils.types import RigTrajectories
from nre.utils.visualize import scalar2rgb


log = logging.getLogger(__name__)

"""
This script is intended to demonstrates the ability to control dynamic actors using the gRPC RGBRenderRequest.

It requires a pretrained `GaussiansComposite` model with an NRE artifact, to train a suitable model make sure you train a dynamic reconstruction and enable saving `rig_trajectories` and `sequence_tracks`.

For example:
    ```
    python run.py --config-name apps/AV/NV/3dgut_dynamic.yaml out_dir=OUTDIR mode=train dataset.path=DATAPATH \
            checkpoint.artifact.enabled=true \
            checkpoint.artifact.rig_trajectories.enabled=true \
            checkpoint.artifact.sequence_tracks.enabled=true
    ```

Before running the script, make sure to start the grpc server, using the same `port` and `artifact-glob` that includes the artifact path. For example:

```
python run.py serve-grpc --artifact-glob ARTIFACT_GLOB
```
"""


def dynamic_poses_to_dynamic_objects(dynamic_poses: list) -> List[DynamicObject]:
    """Convert internal dynamic_poses to gRPC DynamicObject format."""
    return [
        DynamicObject(
            track_id=dynamic_pose["track_id"],
            pose_pair=PosePair(
                start_pose=tquat_to_grpc_pose(
                    dynamic_pose["frame_start_actor_to_world"].data,
                ),
                end_pose=tquat_to_grpc_pose(
                    dynamic_pose["frame_end_actor_to_world"].data,
                ),
            ),
        )
        for dynamic_pose in dynamic_poses
    ]


def generate_request(
    scene_id: str,
    camera_spec: CameraSpec,
    frame_start_camera_to_world: np.ndarray,  # 4x4 SE3 matrix
    frame_end_camera_to_world: np.ndarray,  # 4x4 SE3 matrix
    frame_start_timestamp_us: int,
    frame_end_timestamp_us: int,  # Pass frame-end timestamp +1 here (due to the #@$! half-closed-interval convention)
    height: int,
    dynamic_poses: list,
    format: ImageFormat,
    ego_mask_id: EgoMaskId | None,
) -> RGBRenderRequest:
    # gRPC uses half-closed-interval convention, so start and end timestamp can not be equal
    assert frame_start_timestamp_us < frame_end_timestamp_us
    return RGBRenderRequest(
        scene_id=scene_id,
        resolution_h=height,
        resolution_w=1,
        camera_intrinsics=camera_spec,
        frame_start_us=frame_start_timestamp_us,
        frame_end_us=frame_end_timestamp_us,
        sensor_pose=PosePair(
            start_pose=se3_to_grpc_pose(frame_start_camera_to_world),
            end_pose=se3_to_grpc_pose(frame_end_camera_to_world),
        ),
        dynamic_objects=dynamic_poses_to_dynamic_objects(dynamic_poses),
        image_format=format,
        image_quality=95,
        ego_mask_id=ego_mask_id,
        insert_ego_mask=ego_mask_id is not None,
    )


def generate_lidar_request(
    scene_id: str,
    lidar_spec: LidarSpec | None,
    poses,
    frame_start_timestamp_us: int,
    frame_end_timestamp_us: int,
    dynamic_poses: list,
    render_filter: LidarRenderFilter | None,
) -> LidarRenderRequest:
    return LidarRenderRequest(
        scene_id=scene_id,
        lidar_config=lidar_spec,
        frame_start_us=frame_start_timestamp_us,
        frame_end_us=frame_end_timestamp_us,
        sensor_pose=PosePair(
            start_pose=se3_to_grpc_pose(poses[0]),
            end_pose=se3_to_grpc_pose(poses[1]),
        ),
        dynamic_objects=dynamic_poses_to_dynamic_objects(dynamic_poses),
        render_filter=render_filter,
    )


async def save_request(fname: str, response: RGBRenderReturn) -> None:
    async with aiofiles.open(fname, "wb") as output_file:
        await output_file.write(response.image_bytes)


async def save_lidar_response(fname: str, response: LidarRenderReturn, format: str = "bin") -> None:
    """Save LiDAR response as raw binary or PLY.

    Args:
        fname: Output filename (extension will be added based on format)
        response: LiDAR render response from gRPC
        format: "bin" for raw binary, "ply" for PLY with intensity colors
    """
    if format == "ply":
        # Parse xyz and intensity from response
        if response.point_xyzs_buffer:
            xyz = np.frombuffer(response.point_xyzs_buffer, dtype=np.float32).reshape(-1, 3)
        else:
            xyz = np.array(response.point_xyzs, dtype=np.float32).reshape(-1, 3)

        # Create PLY mesh
        mesh = pcu.TriangleMesh()
        mesh.vertex_data.positions = xyz

        # Add intensity as colors if available
        if response.point_intensities_buffer:
            intensity = np.frombuffer(response.point_intensities_buffer, dtype=np.float32)
            mesh.vertex_data.colors = scalar2rgb(intensity)
        elif response.point_intensities:
            intensity = np.array(response.point_intensities, dtype=np.float32)
            mesh.vertex_data.colors = scalar2rgb(intensity)

        # Save synchronously (pcu doesn't support async)
        mesh.save(fname)
    else:
        # Raw binary format
        async with aiofiles.open(fname, "wb") as f:
            if response.point_xyzs_buffer:
                await f.write(response.point_xyzs_buffer)
            if response.point_intensities_buffer:
                await f.write(response.point_intensities_buffer)


@click.command("render-grpc")
@cli.scopedtimer_cli_options(print_func=log.info)
@click.option(
    "--artifact-path",
    type=str,
    help="Path to the NRE artifact `last.usdz`",
    default=None,
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output rendered image",
    required=True,
    default=None,
)
@click.option(
    "--host",
    type=str,
    help="GRPC server host",
    default="localhost",
    required=False,
)
@click.option(
    "--port",
    type=int,
    help="Port to run the gRPC server on",
    default=8080,
)
@click.option(
    "--height",
    type=int,
    help="Height of the image",
    default=300,
)
@click.option(
    "--camera-id",
    type=str,
    help="Camera ID",
    default="camera_front_wide_120fov",
)
@click.option(
    "--image-format",
    type=click.Choice(["png", "jpeg"]),
    help="png or jpeg",
    default="jpeg",
)
@click.option(
    "--lidar",
    is_flag=True,
    help="If set, call render_lidar instead of render_rgb and save point clouds.",
    default=False,
)
@click.option(
    "--lidar-id",
    type=str,
    help="Lidar ID to use for frame timestamps when --lidar is set.",
    default=None,
)
@click.option(
    "--lidar-format",
    type=click.Choice(["bin", "ply"]),
    help="Output format for lidar: 'bin' (raw binary) or 'ply' (with intensity colors, like validation).",
    default="bin",
)
@click.option(
    "--frame-step",
    type=int,
    help="Step size in frames",
    default=1,
)
@click.option(
    "--disable-rolling-shutter",
    is_flag=True,
    help="Disable rolling shutter by applying the frame-end timestamps to full frames (useful for debugging)",
    default=False,
)
@click.option(
    "--enable-editing-actors",
    is_flag=True,
    help="Enable sending dynamic actor updates in render requests.",
    default=False,
)
@click.option(
    "--demo-actor-transform",
    is_flag=True,
    help="Demo: apply a precomputed transformation to actor poses. Requires --enable-editing-actors.",
    default=False,
)
@click.option(
    "--shutdown-server-on-completion",
    is_flag=True,
    help="Shutdown the server on completion",
    default=False,
)
@click.option(
    "--frame-naming",
    type=click.Choice(["frame-end-timestamp", "contiguous-output-index"]),
    help=(
        "File naming scheme for exported frames: "
        "'frame-end-timestamp' - global absolute frame-end timestamp in microseconds, "
        "'contiguous-output-index' - frames always indexed 0,1,2,... per sensor, irrespective of --frame-step."
    ),
    default="contiguous-output-index",
)
@click.option(
    "--rig-name",
    type=str,
    default=None,
    help="Rig name for the inpainted ego hood (e.g. hyperion8.0 or hyperion8.1). Set to None to disable inpainting.",
)
@click.option(
    "--rig-translation-offset",
    nargs=3,
    type=float,
    help="Translation offsets (tx,ty,tz) in meters in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--rig-rotation-offset",
    nargs=3,
    type=float,
    # TODO: update this to (roll, pitch, yaw) once the axis permutation hack is removed from pose_offsets_to_se3().
    help="Rotation offsets (yaw, -roll, -pitch) in degrees in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--edit-assets",
    type=str,
    help="Path to JSON file containing edit assets configuration, see nre/utils/io/export/external_assets.py for more details + example",
    default=None,
    required=False,
)
@click.option(
    "--sequential",
    is_flag=True,
    help="Send gRPC requests one-by-one (no parallel batch after warmup)",
    default=False,
)
@click.option(
    "--lidar-raydrop-threshold",
    type=float,
    help="Raydrop probability threshold [0-1]. Rays with raydrop > threshold are dropped.",
    default=0.5,
)
@click.option(
    "--lidar-opacity-threshold",
    type=float,
    help="Opacity threshold [0-1]. Rays with opacity <= threshold are dropped. Set to 0.0 to disable.",
    default=0.0,
)
@click.option(
    "--lidar-distance-filter/--no-lidar-distance-filter",
    "lidar_enable_distance_filter",
    help="Enable/disable distance-based edge filtering to remove floating points. (default: use artifact config)",
    default=None,
)
@click.option(
    "--lidar-distance-filter-threshold",
    type=float,
    help="Distance filter threshold [0-1]. Higher = fewer points filtered. (default: use artifact config)",
    default=None,
)
@click.pass_context
def render_grpc(
    ctx: click.Context,
    artifact_path: str,
    output_dir: str,
    host: str,
    port: int,
    height: int,
    camera_id: str,
    image_format: str,
    frame_step: int,
    disable_rolling_shutter: bool,
    enable_editing_actors: bool,
    demo_actor_transform: bool,
    shutdown_server_on_completion: bool,
    rig_name: Optional[str],
    rig_translation_offset: Tuple[float, float, float],
    rig_rotation_offset: Tuple[float, float, float],
    edit_assets: Optional[str],
    lidar: bool,
    lidar_id: Optional[str],
    lidar_format: str,
    sequential: bool,
    lidar_raydrop_threshold: float,
    lidar_opacity_threshold: float,
    lidar_enable_distance_filter: Optional[bool],
    lidar_distance_filter_threshold: Optional[float],
    frame_naming: str,
) -> None:
    """gRPC demo client to render RGB or LiDAR along the training trajectory with optional actor editing"""

    os.makedirs(output_dir, exist_ok=True)

    if demo_actor_transform and not enable_editing_actors:
        raise ValueError("--demo-actor-transform requires --enable-editing-actors")

    # Capture, log, and save CLI settings
    collector = SettingsCollector.from_click_context(ctx, "render-grpc")
    collector.log_settings(log)
    collector.save_json(Path(output_dir) / "render_grpc_cli_args.json")

    format = {"png": ImageFormat.PNG, "jpeg": ImageFormat.JPEG}[image_format]

    asyncio.run(
        _render_grpc(
            artifact_path,
            output_dir,
            host,
            port,
            height,
            camera_id,
            format,
            frame_step,
            disable_rolling_shutter,
            enable_editing_actors,
            demo_actor_transform,
            shutdown_server_on_completion,
            rig_name,
            rig_translation_offset,
            rig_rotation_offset,
            edit_assets,
            lidar,
            lidar_id,
            lidar_format,
            sequential,
            lidar_raydrop_threshold,
            lidar_opacity_threshold,
            lidar_enable_distance_filter,
            lidar_distance_filter_threshold,
            frame_naming,
        )
    )


async def _render_grpc(
    artifact_path: str,
    output_dir: str,
    host: str,
    port: int,
    height: int,
    camera_id: str,
    format: ImageFormat,
    frame_step: int,
    disable_rolling_shutter: bool,
    enable_editing_actors: bool,
    demo_actor_transform: bool,
    shutdown_server_on_completion: bool,
    rig_name: Optional[str],
    rig_translation_offset: Tuple[float, float, float],
    rig_rotation_offset: Tuple[float, float, float],
    edit_assets: Optional[str],
    lidar: bool,
    lidar_id: Optional[str],
    lidar_format: str,
    sequential: bool,
    lidar_raydrop_threshold: float,
    lidar_opacity_threshold: float,
    lidar_enable_distance_filter: Optional[bool],
    lidar_distance_filter_threshold: Optional[float],
    frame_naming: str = "contiguous-output-index",
) -> None:
    artifacts_list = Artifact.discover_from_glob(artifact_path)
    assert len(artifacts_list) == 1
    format_ext = {ImageFormat.PNG: "png", ImageFormat.JPEG: "jpeg"}[format]
    artifact = artifacts_list[0]

    # Get test trajectory and poses
    scene_id = artifact.scene_id
    rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

    # Assume we only have a single trajectory
    assert len(rig_trajectories.rig_trajectories) == 1, (
        f"_render_grpc: expected a single rig_trajectory, got {len(rig_trajectories.rig_trajectories)}"
    )
    trajectory = rig_trajectories.rig_trajectories[0]

    rig_pose_interpolator = ncore_transformations.PoseInterpolator(
        poses=trajectory.T_rig_worlds.cpu(), timestamps=trajectory.T_rig_world_timestamps_us.cpu()
    )

    # Grab cuboid tracks, associated pose interpolators and store them in lists
    sequence_tracks = {k: CuboidTracks.from_dict(v) for k, v in artifact.sequence_tracks.items()}
    assert isinstance(sequence_tracks, dict) and len(sequence_tracks.values()) == 1
    cuboid_tracks = next(iter(sequence_tracks.values()))  # Assumes only one sequence
    controllable_cuboid_tracks = CuboidTracks.Ops.subset_from_mask(
        cuboid_tracks, cuboid_tracks.get_mask_flags_all(TrackFlags.CONTROLLABLE)
    )

    interpolator_list = []
    cuboid_tracks_list = []
    for track_idx in range(controllable_cuboid_tracks.n_tracks):
        cuboid_track = CuboidTracks.Ops.subset_from_indices(controllable_cuboid_tracks, [track_idx])
        cuboid_track = CuboidTracks.Ops.clone(cuboid_track)

        cuboid_tracks_list.append(cuboid_track)
        interpolator_list.append(
            ncore_transformations.PoseInterpolator(
                poses=cuboid_track.tracks_poses.matrix().cpu(), timestamps=cuboid_track.tracks_timestamps_us.cpu()
            )
        )

    # Configure channel with larger message size limits since lidar output can be large
    channel_options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
    ]
    async with grpc.aio.insecure_channel(f"{host}:{port}", options=channel_options) as channel:
        client_service = SensorsimServiceStub(channel)

        available_cameras: AvailableCamerasReturn = await client_service.get_available_cameras(
            AvailableCamerasRequest(scene_id=scene_id)
        )

        # Process edit assets configuration if provided
        remove = set()
        if edit_assets:
            with open(edit_assets, "r") as f:
                edit_assets_config = json.load(f)

            # Get available external assets from GRPC AssetBank
            external_assets_response: ExternalAssetObjectsReturn = await client_service.get_external_asset_objects(
                ExternalAssetObjectsRequest(scene_id=scene_id)
            )
            available_scene_track_ids = set(external_assets_response.track_ids)

            log.info(f"Available external asset track IDs: {available_scene_track_ids}")

            # track_ids to remove, used later when building RGBRenderRequest
            remove.update(edit_assets_config.get("remove", []))

            # Process tracks to replace
            replace_actions = []
            replace_list = edit_assets_config.get("replace", [])

            id_to_dims: dict = {}
            if replace_list:
                for asset_meta in edit_assets_config.get("metadata", {}).get("external_assets_metadata", []):
                    track_id = asset_meta.get("track_id")
                    cuboid_dims = asset_meta.get("cuboid_dims")
                    if track_id is not None and cuboid_dims is not None:
                        id_to_dims[track_id] = cuboid_dims

            for replacement_spec in replace_list:
                original_id = replacement_spec.get("original_id")
                replacement_id = replacement_spec.get("replacement_id")
                if not original_id or not replacement_id:
                    continue

                # Get specified dimensions or fallback to dimensions packaged in edit-assets.json
                object_size_list = replacement_spec.get("object_size", id_to_dims.get(replacement_id))
                if object_size_list is not None and len(object_size_list) == 0:
                    object_size_list = id_to_dims.get(replacement_id)

                assert object_size_list is not None and len(object_size_list) == 3, (
                    f"object_size must be list of 3 floats for replacing {original_id} --> {replacement_id}, got: {object_size_list}"
                )

                action = ReplaceAssetAction(
                    original_id=original_id,
                    replacement_id=replacement_id,
                    object_size=AABB(
                        size_x=float(object_size_list[0]),
                        size_y=float(object_size_list[1]),
                        size_z=float(object_size_list[2]),
                    ),
                )
                replace_actions.append(action)

            # Process tracks to insert
            insert: List[DynamicObjectTrack] = []
            actor_tracks = None
            track_to_asset = None
            if edit_assets_config["insert"]["data"]:
                inserted_track_ids = edit_assets_config["insert"]["data"]["tracks_data"]["tracks_id"]

                # Read asset_ids to decouple track identity from asset selection
                asset_ids = edit_assets_config["insert"]["asset_ids"]

                # Validate correspondence between tracks and assets
                assert len(asset_ids) == len(inserted_track_ids), (
                    f"asset_ids length mismatch {len(asset_ids)} with inserted track_ids {len(inserted_track_ids)}"
                )

                # Validate all required assets exist in AssetBank or on filesystem
                path_ids = {a for a in asset_ids if a not in available_scene_track_ids}

                missing_paths = {p for p in path_ids if not os.path.isfile(p)}
                assert not missing_paths, (
                    f"Missing assets (not in AssetBank or on disk): {missing_paths}. "
                    f"Available AssetBank assets: {sorted(available_scene_track_ids)}"
                )

                # Check for track ID conflicts with existing artifact tracks
                conflicting_ids = set(inserted_track_ids) & set(cuboid_tracks.tracks_id)
                assert not conflicting_ids, (
                    f"Conflicting track IDs between inserted and existing tracks: {sorted(conflicting_ids)}"
                )

                track_to_asset = dict(zip(inserted_track_ids, asset_ids))

                # Convert insert data to gRPC format
                insert_cuboid_tracks = CuboidTracks.from_dict(edit_assets_config["insert"]["data"])
                if insert_cuboid_tracks:
                    actor_tracks = ActorTracks._from_cuboid_tracks(insert_cuboid_tracks)
                    insert = actor_tracks_to_grpc(actor_tracks, track_to_asset=track_to_asset)

            if len(replace_actions) != 0 or len(insert) != 0:
                log.info(f"Sending EditAssetsRequest: {len(replace_actions)} replacements, {len(insert)} insertions")

                edit_request = EditAssetsRequest(
                    scene_id=scene_id,
                    replace=replace_actions,
                    insert=insert,
                )

                edit_response: EditAssetsResponse = await client_service.edit_assets(edit_request)

                if not edit_response.success:
                    raise RuntimeError(f"{edit_response.message}")

                # Add inserted cuboid_tracks to the artifact's cuboid_tracks
                if len(insert) != 0 and actor_tracks is not None:
                    cuboid_tracks_list.extend(actor_tracks._cuboid_tracks_list)
                    interpolator_list.extend(actor_tracks._interpolator_list)

        available_ego_masks: AvailableEgoMasksReturn = await client_service.get_available_ego_masks(EmptyRequest())
        available_rigs = {
            ego_mask.ego_mask_id.rig_config_id: ego_mask.ego_mask_id
            for ego_mask in available_ego_masks.ego_mask_metadata
        }
        print(f"Available rigs: {available_rigs.keys()}, choosing {rig_name=}.")
        if rig_name is not None:
            try:
                ego_mask_id = available_rigs[rig_name]
            except KeyError:
                raise KeyError(f"{rig_name=} not found in available rigs {available_rigs.keys()}")
        else:
            ego_mask_id = None

        # Validate camera and get timestamps based on rendering mode
        camera = None
        if lidar:
            # For lidar-only rendering, use lidar timestamps
            available_lidar_ids = list(trajectory.lidars_frame_timestamps_us.keys())
            if not available_lidar_ids:
                raise ValueError("No lidar timestamps available in artifact")

            # Use specified lidar_id or first available
            if lidar_id is not None:
                unique_lidar_id = lidar_id if "@" in lidar_id else f"{lidar_id}@{scene_id}"
                if unique_lidar_id not in trajectory.lidars_frame_timestamps_us:
                    raise ValueError(
                        f"{lidar_id=} not found in available lidars. "
                        f"Available: {[lid.split('@')[0] for lid in available_lidar_ids]}"
                    )
            else:
                unique_lidar_id = available_lidar_ids[0]
                log.info(f"Using first available lidar: {unique_lidar_id.split('@')[0]}")

            timestamps_us = trajectory.lidars_frame_timestamps_us[unique_lidar_id].cpu().numpy()
            num_frames = len(timestamps_us)
            log.info(f"Test trajectory contains poses for {num_frames} lidar frames")
        else:
            # For RGB rendering, validate camera and use camera timestamps
            assert camera_id in [
                available_camera.logical_id for available_camera in available_cameras.available_cameras
            ]
            for available_camera in available_cameras.available_cameras:
                if available_camera.logical_id == camera_id:
                    camera = available_camera
                    break
            if camera is None:
                raise ValueError(f"{camera_id=} not found in available cameras")

            unique_camera_id = camera_id + f"@{scene_id}"
            timestamps_us = trajectory.cameras_frame_timestamps_us[unique_camera_id].cpu().numpy()
            num_frames = len(timestamps_us)
            log.info(f"Test trajectory contains poses for {num_frames} video frames")

        assert timestamps_us.shape == (num_frames, 2)
        assert timestamps_us.dtype == np.int64
        frame_start_timestamps_us = timestamps_us[:, 0]
        frame_end_timestamps_us = timestamps_us[:, 1]

        # Make sure no frame-end timestamp is missing.
        if (num_missing_timestamps := np.sum(frame_end_timestamps_us < 0)) > 0:
            raise ValueError(
                f"Frame-end timestamps are required but {num_missing_timestamps} out of {num_frames} are missing"
            )

        if (num_missing_timestamps := np.sum(frame_start_timestamps_us < 0)) > 0:
            # Handle case when frame start timestamps stored inside the artifact are -1.
            # Happened when saving rig_trajectories with obsolete datasource.get_rig_trajectories(end_frame_timestamps_only=True) argument.
            log.warning(
                f"Rolling shutter effect disabled due to {num_missing_timestamps} "
                f"out of {num_frames} missing frame-start timestamps"
            )
            frame_start_timestamps_us = frame_end_timestamps_us
        elif disable_rolling_shutter:
            log.warning("Rolling shutter effect disabled from command-line. Only using frame-end timestamps.")
            frame_start_timestamps_us = frame_end_timestamps_us

        # spin objects to highlight controllability
        euler_angles = np.zeros((num_frames, 3))
        euler_angles[:, 2] = 40.0 * np.sin(np.linspace(0, 2 * np.pi, euler_angles.shape[0]))
        delta_matrices = np.eye(4)[None].repeat(euler_angles.shape[0], axis=0).astype(np.float32)
        delta_matrices[:, :3, :3] = R.from_euler("xyz", euler_angles, degrees=True).as_matrix()

        # Common: output frame timestamps
        output_frame_timestamps: list[dict[str, int | str]] = []
        frame_count = 0

        # Helper to build dynamic poses for RGB requests
        def _build_dynamic_poses(frame_start_us: int, frame_end_us: int, frame_idx: int) -> list[dict[str, Any]]:
            dynamic_poses: list[dict[str, Any]] = []
            for cuboid_track, interpolator in zip(cuboid_tracks_list, interpolator_list):
                track_id = cuboid_track.tracks_id[0]

                # remove defined as a list of track_ids(str) in edit-assets.json
                if track_id in remove:
                    continue

                # If the frame time range is within the cuboid track's range, the actor pose can not be interpolated.
                if (
                    frame_start_us < cuboid_track.tracks_timestamps_us.min()
                    or frame_end_us > cuboid_track.tracks_timestamps_us.max()
                ):
                    continue

                interpolated_world_poses = interpolator.interpolate_to_timestamps([frame_start_us, frame_end_us])
                assert interpolated_world_poses.shape == (2, 4, 4)
                assert interpolated_world_poses.dtype == np.float32

                frame_start_actor_to_world = lt.SE3(se3_matrix_to_tquat(interpolated_world_poses[0])).cuda()
                frame_end_actor_to_world = lt.SE3(se3_matrix_to_tquat(interpolated_world_poses[1])).cuda()

                if demo_actor_transform:
                    delta_matrix = se3_matrix_to_se3(delta_matrices[frame_idx]).cuda()
                    frame_start_actor_to_world = frame_start_actor_to_world * delta_matrix
                    frame_end_actor_to_world = frame_end_actor_to_world * delta_matrix

                dynamic_poses.append(
                    {
                        "track_id": track_id,
                        "frame_start_actor_to_world": frame_start_actor_to_world,
                        "frame_end_actor_to_world": frame_end_actor_to_world,
                    }
                )
            return dynamic_poses

        # Reusable executor with warmup for both RGB and LiDAR
        async def _exec(method: Callable, reqs: list[Any], sequential_exec: bool) -> list[Any]:
            """Execute requests with warmup: first request solo, rest either sequentially or in parallel."""
            if len(reqs) == 0:
                return []
            warmup, *rest = reqs
            results = [await method(warmup)]
            if rest:
                n_requests = len(rest)
                t0 = time.perf_counter()
                if sequential_exec:
                    results.extend([await method(r) for r in rest])
                else:
                    results.extend(await asyncio.gather(*[method(r) for r in rest]))
                elapsed = time.perf_counter() - t0
                s_per_request = elapsed / n_requests
                print(f"Served {n_requests=} in {elapsed:.3f}s, {s_per_request=:.3f}s")
            return results

        # prepare requests (RGB or LiDAR)
        rgb_requests: list[RGBRenderRequest] = []
        lidar_requests: list[LidarRenderRequest] = []
        lidar_spec = LidarSpec(lidar_type=LidarDeviceType.PANDAR128) if lidar else None

        # Build lidar render filter from CLI parameters
        lidar_render_filter = None
        if lidar:
            lidar_render_filter = LidarRenderFilter(
                raydrop_threshold=lidar_raydrop_threshold,
                opacity_threshold=lidar_opacity_threshold,
            )
            # Distance filter settings are optional (use artifact config if not specified)
            if lidar_enable_distance_filter is not None:
                lidar_render_filter.enable_distance_filter = lidar_enable_distance_filter
            if lidar_distance_filter_threshold is not None:
                lidar_render_filter.distance_filter_threshold = lidar_distance_filter_threshold

        for frame_idx in range(0, num_frames, frame_step):
            frame_start_us = int(frame_start_timestamps_us[frame_idx])
            frame_end_us = int(frame_end_timestamps_us[frame_idx])

            # gRPC time range is half-closed [start, end) (end exclusive).
            # For an instant shutter at t (start==end==t), encode as [t, t+1).
            frame_end_us_grpc = frame_end_us + 1 if frame_start_us == frame_end_us else frame_end_us

            dynamic_poses = (
                _build_dynamic_poses(frame_start_us, frame_end_us, frame_idx) if enable_editing_actors else []
            )

            if not lidar:
                # RGB: build dynamic poses and camera request
                assert camera is not None  # Set in else branch above when lidar=False
                # Dynamic actor updates are only sent when explicitly enabled.
                pose_rig_to_world = rig_pose_interpolator.interpolate_to_timestamps([frame_start_us, frame_end_us])
                assert pose_rig_to_world.shape == (2, 4, 4)
                pose_camera_to_rig = grpc_pose_to_torch_se3(camera.rig_to_camera).cpu().numpy()

                # Apply the transformation offset to the camera in the rig frame.
                # rig_offset_se3 @ pose_camera_to_rig can be considered as the modified camera-to-rig pose.
                # rig_offset_se3 should be identity when pose offsets are not specified (they should default to zero).
                rig_offset_se3 = pose_offsets_to_se3(rig_translation_offset, rig_rotation_offset)
                frame_start_camera_to_world_se3 = pose_rig_to_world[0] @ rig_offset_se3 @ pose_camera_to_rig
                frame_end_camera_to_world_se3 = pose_rig_to_world[1] @ rig_offset_se3 @ pose_camera_to_rig

                assert frame_start_camera_to_world_se3.shape == (4, 4)
                assert frame_end_camera_to_world_se3.shape == (4, 4)

                rgb_requests.append(
                    generate_request(
                        scene_id=scene_id,
                        camera_spec=camera.intrinsics,
                        frame_start_camera_to_world=frame_start_camera_to_world_se3,
                        frame_end_camera_to_world=frame_end_camera_to_world_se3,
                        frame_start_timestamp_us=frame_start_us,
                        frame_end_timestamp_us=frame_end_us_grpc,
                        height=height,
                        dynamic_poses=dynamic_poses,
                        format=format,
                        ego_mask_id=ego_mask_id,
                    )
                )
                match frame_naming:
                    case "frame-end-timestamp":
                        rgb_file_name = f"{frame_end_us}.{format_ext}"
                    case "contiguous-output-index":
                        rgb_file_name = f"{frame_count:06d}.{format_ext}"
                    case _:
                        raise ValueError(f"Invalid frame naming scheme: {frame_naming}")
                output_frame_timestamps.append(
                    {
                        "file_name": rgb_file_name,
                        "render_frame_idx": frame_count,
                        "frame_start_timestamp_us": frame_start_us,
                        "frame_end_timestamp_us": frame_end_us,
                    }
                )
            else:
                # LiDAR: build lidar request from rig pose
                poses = rig_pose_interpolator.interpolate_to_timestamps([frame_start_us, frame_end_us])
                lidar_requests.append(
                    generate_lidar_request(
                        scene_id=scene_id,
                        lidar_spec=lidar_spec,
                        poses=poses,
                        frame_start_timestamp_us=frame_start_us,
                        frame_end_timestamp_us=frame_end_us_grpc,
                        dynamic_poses=dynamic_poses,
                        render_filter=lidar_render_filter,
                    )
                )
                match frame_naming:
                    case "frame-end-timestamp":
                        lidar_file_name = f"{frame_end_us}.{lidar_format}"
                    case "contiguous-output-index":
                        lidar_file_name = f"{frame_count:06d}.{lidar_format}"
                    case _:
                        raise ValueError(f"Invalid frame naming scheme: {frame_naming}")
                output_frame_timestamps.append(
                    {
                        "file_name": lidar_file_name,
                        "render_frame_idx": frame_count,
                        "frame_start_timestamp_us": frame_start_us,
                        "frame_end_timestamp_us": frame_end_us,
                    }
                )
            frame_count += 1

        # Execute requests and save outputs
        if not lidar:
            # RGB path: render images
            responses: list[RGBRenderReturn] = await _exec(client_service.render_rgb, rgb_requests, sequential)
            if output_dir:
                await asyncio.gather(
                    *[
                        save_request(
                            os.path.join(output_dir, str(output_frame_timestamps[contiguous_frame_idx]["file_name"])),
                            response,
                        )
                        for contiguous_frame_idx, response in enumerate(responses)
                    ]
                )
        else:
            # LiDAR path: render point clouds
            lidar_responses: list[LidarRenderReturn] = await _exec(
                client_service.render_lidar, lidar_requests, sequential
            )
            if output_dir:
                lidar_ext = lidar_format
                await asyncio.gather(
                    *[
                        save_lidar_response(
                            os.path.join(output_dir, str(output_frame_timestamps[idx]["file_name"])),
                            response,
                            format=lidar_format,
                        )
                        for idx, response in enumerate(lidar_responses)
                    ]
                )

        # Write timestamps for both RGB and LiDAR paths
        if output_dir:
            with open(os.path.join(output_dir, "timestamps.json"), "w") as f:
                json.dump(output_frame_timestamps, f, indent=2)

        # Restore model parameters if assets were edited
        if edit_assets:
            restore_request = RestoreModelParametersRequest(scene_id=scene_id)
            await client_service.restore_model_parameters(restore_request)
            print(f"Restored model parameters for scene: {scene_id}")

        if shutdown_server_on_completion:
            await client_service.shut_down(EmptyRequest())
