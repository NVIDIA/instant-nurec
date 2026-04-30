# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import json
import logging
import os

from pathlib import Path

import click
import numpy as np

from scipy.spatial.transform import Rotation as R

from ncore.data import FrameTimepoint
from ncore.impl.common.transformations import PoseInterpolator, pose_bbox
from ncore_internal.data.v3 import CameraSensor, LidarSensor, Sensor, ShardDataLoader


# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_world_bbox_to_grpc_pose(bbox: np.ndarray) -> tuple:
    """
    Convert the bbox in world coordinates to log2sim format.
    """
    x, y, z = bbox[0], bbox[1], bbox[2]
    rotation_angles = bbox[6:9]
    quat = R.from_euler("xyz", rotation_angles, degrees=False).as_quat(canonical=False)

    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]), float(x), float(y), float(z)


def generate_sensor_tracks(
    sensor: Sensor,
    sensor_id: str,
    track_dict: dict,
    dynamic_tracks: dict,
    start_timestamp_us: int,
    end_timestamp_us: int,
):
    # Get camera frame range
    camera_timestamps = sensor.get_frames_timestamps_us()
    camera_frame_start_idx = np.where(camera_timestamps >= start_timestamp_us)[0][0]
    camera_frame_stop_idx = min(
        np.where(camera_timestamps < end_timestamp_us)[0][-1],
        sensor.get_frames_count(),
    )
    logger.info(
        f"Processing {camera_frame_stop_idx - camera_frame_start_idx + 1} camera frames: {camera_frame_start_idx} ~ {camera_frame_stop_idx}"
    )
    camera_frame_idx_list = sensor.get_frame_index_range(camera_frame_start_idx, camera_frame_stop_idx + 1)
    track_dict[sensor_id] = {}
    for idx, camera_frame_idx in enumerate(camera_frame_idx_list):
        # check which dynamic labels where observed at this frame's time + project
        frame_start_timestamp_us = sensor.get_frame_timestamp_us(camera_frame_idx, FrameTimepoint.START)
        frame_end_timestamp_us = sensor.get_frame_timestamp_us(camera_frame_idx, FrameTimepoint.END)
        frame_mid_timestamp = frame_start_timestamp_us + (frame_end_timestamp_us - frame_start_timestamp_us) // 2

        if isinstance(sensor, CameraSensor):
            track_dict[sensor_id][camera_frame_idx] = {
                "start_timestamp_us": int(frame_start_timestamp_us),
                "start_pose": list(
                    sensor.get_frame_T_sensor_world(camera_frame_idx, FrameTimepoint.START).astype(float).flatten()
                ),
                "end_timestamp_us": int(frame_end_timestamp_us),
                "end_pose": list(
                    sensor.get_frame_T_sensor_world(camera_frame_idx, FrameTimepoint.END).astype(float).flatten()
                ),
                "tracks": [],
            }
        elif isinstance(sensor, LidarSensor):
            track_dict[sensor_id][camera_frame_idx] = {
                "start_timestamp_us": int(frame_start_timestamp_us),
                "sensor2rig": list(sensor.get_T_sensor_rig().astype(float).flatten()),
                "start_pose": list(
                    sensor.get_frame_T_rig_world(camera_frame_idx, FrameTimepoint.START).astype(float).flatten()
                ),
                "end_timestamp_us": int(frame_end_timestamp_us),
                "end_pose": list(
                    sensor.get_frame_T_rig_world(camera_frame_idx, FrameTimepoint.END).astype(float).flatten()
                ),
                "tracks": [],
            }

        for track_id, dynamic_track in dynamic_tracks.items():
            if not (
                dynamic_track["timestamps_us"][0] <= frame_start_timestamp_us
                and frame_end_timestamp_us <= dynamic_track["timestamps_us"][-1]
            ):
                continue

            # interpolate track to camera start and end timestamp
            def get_object_meta(track_id, timestamp_us):
                bbox_pose = dynamic_track["pose_interpolator"].interpolate_to_timestamps(timestamp_us)[0]

                bbox = pose_bbox(bbox_pose, dynamic_track["dimension"])

                x, y, z = dynamic_track["dimension"]

                quat_x, quat_y, quat_z, quat_w, center_x, center_y, center_z = convert_world_bbox_to_grpc_pose(bbox)

                meta_list = [track_id, quat_x, quat_y, quat_z, quat_w, center_x, center_y, center_z, x, y, z]

                return meta_list

            track_dict[sensor_id][camera_frame_idx]["tracks"].append(
                (get_object_meta(track_id, frame_start_timestamp_us), get_object_meta(track_id, frame_end_timestamp_us))
            )


@click.command("export-ncore-tracks")
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option("--model-tracks-json", type=str, help="Model tracks to be loaded", required=True)
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--seek-offset-sec",
    type=click.IntRange(min=0, max_open=True),
    help="Initial pose timestamp offset in secondes to start export",
    default=None,
)
@click.option(
    "--duration-sec",
    type=click.FloatRange(min=-1, max_open=True),
    help="The duration seconds of the export sequence (-1 for all frames)",
    default=None,
)
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be used (multiple value option, front wide camera if not specified) "
    "- the first camera is considered to be the reference camera frames number parameters are relative to",
    default=["camera_front_wide_120fov"],
)
@click.option(
    "--lidar-id",
    "lidar_id",
    default=["lidar"],
    type=str,
    help="If provided, the lidar sensor to incorporate point clouds from",
)
@click.option(
    "--enable-lidars/--disable-lidars",
    default=True,
    help="Whether to include lidar data in the export",
)
def export_ncore_tracks(
    shard_file_pattern: str,
    model_tracks_json: str,
    output_dir: str,
    seek_offset_sec: float,
    duration_sec: float,
    camera_ids: list[str],
    lidar_id: str,
    enable_lidars: bool,
) -> None:
    """Extracts and exports sensor tracks to json file"""
    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)

    assert len(camera_ids), "Require at least a single camera sensor"

    pose_timestamps = loader.get_poses().T_rig_world_timestamps_us
    pose_end_timestamp_us = pose_timestamps[-1]
    start_timestamp_us = (
        min(pose_timestamps[0] + int(seek_offset_sec * 1e6), pose_end_timestamp_us)
        if seek_offset_sec
        else pose_timestamps[0]
    )
    end_timestamp_us = (
        min(start_timestamp_us + int(duration_sec * 1e6), pose_end_timestamp_us)
        if duration_sec
        else pose_end_timestamp_us
    )

    logger.info(f"Timestamp range that will be processed: [{start_timestamp_us}, {end_timestamp_us}]")

    # Create output paths
    output_path_data = Path(output_dir)
    output_path_data.mkdir(parents=True, exist_ok=True)

    # Load lidar time-range and labels
    dynamic_tracks: dict[str, dict] = {}
    lidar_sensor: LidarSensor
    if enable_lidars and lidar_id:
        logger.info(f"Preparing dynamic objects from '{lidar_id}'")

        # Load sensors
        lidar_sensor = loader.get_lidar_sensor(lidar_id)

        # Extract dynamic tracks from model tracks json
        def tquat_2_se3(tquat):
            rot = R.from_quat(tquat[3:]).as_matrix()
            x, y, z = tquat[0:3]
            mat = np.eye(4, dtype=np.float64)
            mat[0:3, 0:3] = rot
            mat[0, 3] = x
            mat[1, 3] = y
            mat[2, 3] = z
            return mat

        with open(model_tracks_json, "r") as f:
            tracks = json.load(f)["dummy_chunk_id"]
            tracks_id = tracks["tracks_data"]["tracks_id"]
            tracks_poses = tracks["tracks_data"]["tracks_poses"]
            tracks_timestamps = tracks["tracks_data"]["tracks_timestamps_us"]
            tracks_dimensions = tracks["cuboidtracks_data"]["cuboids_dims"]
            for idx, track_id in enumerate(tracks_id):
                track_id = str(track_id)
                track_meta = {}
                track_meta["timestamps_us"] = tracks_timestamps[idx]
                track_meta["dimension"] = np.array(tracks_dimensions[idx])
                track_poses = [tquat_2_se3(pose) for pose in tracks_poses[idx]]
                track_meta["pose_interpolator"] = PoseInterpolator(np.stack(track_poses), track_meta["timestamps_us"])
                dynamic_tracks[track_id] = track_meta

    json_file_path = os.path.join(output_dir, f"sensor_tracks_{start_timestamp_us}_{end_timestamp_us}.json")
    track_json = open(json_file_path, "w")

    track_dict: dict[str, dict] = {}

    # Generate render data for camera
    for camera_id in camera_ids:
        logger.info(f"Processing camera '{camera_id}'")

        assert isinstance(camera_sensor := loader.get_sensor(camera_id), CameraSensor)
        generate_sensor_tracks(
            camera_sensor, camera_id, track_dict, dynamic_tracks, start_timestamp_us, end_timestamp_us
        )

    # Generate render data for lidar
    if enable_lidars and lidar_id:
        logger.info(f"Processing lidar '{lidar_id}'")
        generate_sensor_tracks(lidar_sensor, lidar_id, track_dict, dynamic_tracks, start_timestamp_us, end_timestamp_us)

    json.dump(track_dict, track_json, indent=4)
    logger.info(f"Trajectories log has been written to {json_file_path}")
