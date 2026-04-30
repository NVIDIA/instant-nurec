# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, OrderedDict, Tuple

import click
import imageio.v3 as iio
import numpy as np
import torch

from PIL import Image

import ncore.impl.common.transformations as ncore_transformations
import ncore_internal.data.v3
import nre.utils.cli as cli

from nre.artifact import Artifact
from nre.render import ActorsSnapshot, ActorTracks, PoseRange, RenderableModel, SensorTrajectory
from nre.utils.cli import SettingsCollector
from nre.utils.geometry import pose_offsets_to_se3, se3_matrix_to_tquat
from nre.utils.types import RigTrajectories


log = logging.getLogger(__name__)


def export_video(
    image_dir: Path,
    video_path: Path,
    fps: float = 30.0,
    crf: int = 20,
    max_keyframe_interval_seconds: float = 0.5,
    image_extension: str = "png",
) -> bool:
    """Convert an image file sequence stored in a directory to an H264 MP4 video.

    Args:
        image_dir: Directory containing a sequence of images.
        video_path: Path to the output MP4 file.
        fps: Frame rate of the video to generate.
        crf: Video quality (Constant Rate Factor). Valid range is 0-51. Typical range is 18-28.
          Lower means higher quality.
        max_keyframe_interval_seconds: Maximum interval between keyframes in seconds.
        image_extension: Extension (without the dot) of the image files to find in the directory.

    Returns:
        True if successful, False if image files were not found in the directory.

    Raises if arguments are out of range or if video writer failed.
    """

    if fps <= 0.0:
        raise ValueError(f"fps ({fps}) is negative")
    if crf < 0 or crf > 51:
        raise ValueError(f"crf ({crf}) is outside of the allowed range (0, 51)")
    if max_keyframe_interval_seconds <= 0.0:
        raise ValueError(f"max_keyframe_interval_seconds ({max_keyframe_interval_seconds}) is negative")

    image_files = sorted(image_dir.glob("*." + image_extension))
    if not image_files:
        log.warning(f"No image files found in {image_dir}")
        return False

    log.info(f"Exporting {len(image_files)} frames to {video_path}")

    # Loads all frames into memory - poses a memory risk for long sequences and/or high resolution frames.
    frames = []
    for image_file in image_files:
        frame = iio.imread(image_file)
        frames.append(frame)

    max_keyframe_interval_frames = max(round(max_keyframe_interval_seconds * fps), 1)

    iio.imwrite(
        video_path,
        frames,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=None,  # Set to None to use CRF instead of variable bit rate
        # Params passed to ffmpeg.
        output_params=[
            "-crf",
            str(crf),
            # Guarantees a keyframe at least every second for an improved seeking experience.
            "-g",
            str(max_keyframe_interval_frames),
            # Web-friendly optimization, see: https://trac.ffmpeg.org/wiki/Encode/H.264#faststartforwebvideo
            "-movflags",
            "+faststart",
        ],
    )

    return True


def export_videos_and_json(
    output_dir: str,
    camera_ids: List[str],
    fps: float = 30.0,
    crf: int = 20,
    image_extension: str = "png",
) -> None:
    """Export frames from each camera subdirectory to an MP4 video and create a JSON index.

    Args:
        output_dir: Base output directory containing camera subdirectories
        camera_ids: List of camera IDs that were rendered
        fps: Frames per second for the output videos
        crf: Video quality (Constant Rate Factor). Typical range is 18-28. Lower means higher quality.
        image_extension: Extension (without the dot) of the image files to find in the directory.
    """
    output_path = Path(output_dir)
    video_paths = {}

    for camera_id in camera_ids:
        camera_dir = output_path / camera_id
        if not camera_dir.exists():
            log.warning(f"Camera directory not found: {camera_dir}")
            continue

        video_relative_path = f"{camera_id}.mp4"
        video_path = output_path / video_relative_path
        if export_video(camera_dir, video_path, fps=fps, crf=crf, image_extension=image_extension):
            video_paths[camera_id] = video_relative_path

    # Save JSON with relative paths
    json_path = output_path / "render_cli_videos.json"
    with open(json_path, "w") as f:
        json.dump(video_paths, f, indent=2)

    log.info(f"Exported {len(video_paths)} videos, JSON saved to {json_path}")


def _save_rendered_frame_to_disk(
    numpy_array: np.ndarray,
    image_path: str,
) -> None:
    """Convert a rendered frame to a PIL image and save it to disk.

    Intended to be run in a thread pool to avoid blocking the main rendering loop on disk I/O.

    Args:
        numpy_array: HxWx3 uint8 RGB image array.
        image_path: Full path to write the image file.
    """
    rgb_image = Image.fromarray(numpy_array)

    if image_path.lower().endswith((".jpg", ".jpeg")):
        rgb_image.save(image_path, quality=93)
    else:
        rgb_image.save(image_path)


def get_unique_frame_index(
    sensor_id: str,
    frame_idx: int,
    sensors_start_frame_indices: Optional[Dict[str, int]] = None,
    sequence_id: Optional[str] = None,
) -> Optional[int]:
    """Calculate the unique frame index for a camera frame used during training.

    Args:
        sensor_id: The sensor logical id, e.g. camera_front_wide_120fov.
        frame_idx: The 0-based index of the training frame from the selected sensor.
        sensors_start_frame_indices: Optional mapping of unique sensor identifiers to the linear unique frame index of
            the first training frame provided by the sensor.
        sequence_id: The sequence ID used to construct unique sensor identifiers.

    Returns:
        The unique frame index of the selected frame (unique within the sensor type),
        or None if cameras_start_frame_indices is not provided.
    """
    if sensors_start_frame_indices is None:
        return None  # Logged a warning in __init__() to avoid repeated logging per frame.
    unique_sensor_id = f"{sensor_id}@{sequence_id}"
    if unique_sensor_id not in sensors_start_frame_indices:
        raise ValueError(f"Sensor {unique_sensor_id} not found in sensors_start_frame_indices")
    return sensors_start_frame_indices[unique_sensor_id] + frame_idx


def manipulate_actor_poses(actors_snapshot: ActorsSnapshot) -> ActorsSnapshot:
    """Turns an ActorsSnapshot into another ActorsSnapshot with actors lifted from the ground along the world Z-axis."""
    poses = actors_snapshot.actor_poses  # [M,2,7], poses[i,j,:] ~ [tx,ty,tz,qx,qy,qz,qw] where j=0,1 (start,end pose)
    assert poses.shape == (len(actors_snapshot.actor_ids), 2, 7)  # Just info for the reader
    edited_poses = poses.clone()
    edited_poses[:, :, 2] += 1.0  # Translates all actors in world space along the Z axis
    return ActorsSnapshot(actor_ids=actors_snapshot.actor_ids, actor_poses=edited_poses)


def split_unique_sensor_ids(unique_sensor_ids: Iterable[str]) -> Tuple[List[str], Optional[str]]:
    """Split each unique sensor ID in a list into its constituents <sensor_id>@<sequence_id> and
    make sure that the sequence_id part is identical for all, otherwise raises an error.
    """
    sensor_ids = []
    sequence_ids = []
    for unique_sensor_id in unique_sensor_ids:
        parts = unique_sensor_id.split("@")
        sensor_ids.append(parts[0])
        sequence_ids.append(parts[1] if len(parts) > 1 else None)
    # Make sure scene_id is identical for all sensors
    unique_sequence_ids = set(sequence_ids)
    if len(unique_sequence_ids) > 1:
        raise ValueError(f"Sequence IDs are not identical for all provided sensors: {unique_sensor_ids}")
    return sensor_ids, unique_sequence_ids.pop() if len(unique_sequence_ids) > 0 else None


def fix_frame_timestamps(
    frame_timestamps: np.ndarray,
    min_timestamp: int,
    max_timestamp: int,
    rolling_shutter_duration: Optional[int] = None,
):
    """Validate a (n_frames, 2) array of frame start and end timestamps, fill in missing start timestamps or
    set the start timestamps to meet a fixed rolling shutter duration if specified.

    - Raise an error if frame end timestamps are invalid (e.g. -1).
    - Frame start timestamps can be missing (set to -1), this is not an error.
    - If rolling shutter duration is specified, frame start timestamps are overwritten to respect the duration,
      and then clamped to within the range (min_timestamp, max_timestamp).
    - If no rolling shutter duration is specified, but frame start timestamps are set to -1,
      they are clamped to the frame end timestamps, effectively disabling rolling shutter.
    - Frame start and end timestamps must be within the (min_timestamp, max_timestamp) range.

    Args:
        frame_timestamps: (n_frames, 2) array of frame start and end timestamps
        min_timestamp: Min allowed timestamp value
        max_timestamp: Max allowed timestamp value
        rolling_shutter_duration: Optional rolling shutter duration

    Returns: (n_frames, 2) array of clamped frame start and end timestamps.
    """

    num_frames = frame_timestamps.shape[0]
    assert frame_timestamps.shape == (num_frames, 2), "frame timestamps expected as an array of shape (n_frames, 2)"
    assert frame_timestamps.dtype == np.int64, "frame timestamps expected as an int64 array"

    frame_start_timestamps = frame_timestamps[:, 0]
    frame_end_timestamps = frame_timestamps[:, 1]

    # Make sure no frame-end timestamp is missing.
    if (count := np.sum(frame_end_timestamps < 0)) > 0:
        raise ValueError(f"Frame-end timestamps are required but {count} out of {num_frames} are missing")
    # Make sure all frame-end timestamps are within the trajectory timestamp range.
    if (count := np.sum((frame_end_timestamps < min_timestamp) | (frame_end_timestamps > max_timestamp))) > 0:
        raise ValueError(f"{count} of {num_frames} frame-end timestamps are out of bounds")

    if rolling_shutter_duration is not None:
        # Set frame start timestamps such that
        # frame_start_timestamps_us + rolling_shutter_duration = frame_end_timestamps_us,
        # and clamp within the overall time range of the trajectory.
        if min_timestamp + rolling_shutter_duration > max_timestamp:
            raise ValueError(
                f"The specified frame time ({rolling_shutter_duration} us) exceeds the overall trajectory length "
                f"({max_timestamp - min_timestamp} us)."
            )
        log.warning(
            "Overriding frame start timestamps to meet the specified "
            f"rolling shutter duration ({rolling_shutter_duration} us)"
        )
        frame_start_timestamps = (
            np.maximum(min_timestamp + rolling_shutter_duration, frame_end_timestamps) - rolling_shutter_duration
        )

        if (num_clamped := np.sum(frame_start_timestamps + rolling_shutter_duration < frame_end_timestamps)) > 0:
            log.warning(f"{num_clamped} start timestamps clamped within the trajectory timestamp range")

    elif (count := np.sum(frame_start_timestamps < 0)) > 0:
        # Handle case when frame start timestamps stored inside the artifact are -1.
        # Happened when saving rig_trajectories with obsolete datasource.get_rig_trajectories(end_frame_timestamps_only=True) argument.
        log.warning(
            f"Rolling shutter effect disabled due to {count} out of {num_frames} missing frame-start timestamps"
            "(consider the --rolling-shutter-duration option if you would like to simulate rolling shutter anyway)"
        )
        frame_start_timestamps = frame_end_timestamps

    fixed_timestamps = np.stack([frame_start_timestamps, frame_end_timestamps], axis=1)

    # Make sure all timestamps are within the allowed range.
    if (count := np.sum((fixed_timestamps < min_timestamp) | (fixed_timestamps > max_timestamp))) > 0:
        raise ValueError(f"{count} of {num_frames} frame start/end timestamps are out of bounds")

    assert fixed_timestamps.shape == (num_frames, 2)
    return fixed_timestamps


def get_trajectory_time_range(rig_trajectories: RigTrajectories) -> Tuple[int, int]:
    """Get the (start,end) timestamp of the rig trajectory in microseconds."""
    rig_trajectory = rig_trajectories.rig_trajectories[0]  # Assuming only one trajectory per clip.
    clip_start_us = int(rig_trajectory.T_rig_world_timestamps_us[0].item())
    clip_end_us = int(rig_trajectory.T_rig_world_timestamps_us[-1].item())
    assert clip_start_us >= 0
    assert clip_end_us >= 0
    return clip_start_us, clip_end_us


class SensorCalibProvider(ABC):
    """Base class for providing sensor intrinsics and poses in order to abstract away the calibration data source

    The data source may be all training view intrinsics and poses or a batch of novel view intrinsics and poses.
    """

    _camera_ids: List[str]
    _sequence_id: str | None

    def __init__(self, unique_sensor_ids: Iterable[str]):
        self._camera_ids, self._sequence_id = split_unique_sensor_ids(unique_sensor_ids)

    def get_unique_sensor_id(self, camera_id: str) -> str:
        if self._sequence_id is not None:
            return f"{camera_id}@{self._sequence_id}"
        else:
            return f"{camera_id}"

    @abstractmethod
    def get_available_camera_ids(self) -> List[str]:
        """Get the available camera ids that can be used to query poses"""
        pass

    @abstractmethod
    def get_num_camera_frames(self, camera_id: str) -> int:
        """Get the number of frames available to query poses for a camera"""
        pass

    @abstractmethod
    def get_camera_view_pose(self, camera_id: str, frame_idx: int) -> PoseRange:
        """Get the start/end poses and timestamps for a frame"""
        pass

    @abstractmethod
    def get_camera_calibration(self, camera_id: str) -> RigTrajectories.CameraCalibration:
        """Get the camera intrinsics and extrinsics (sensor-to-rig pose) for a selected camera."""
        pass

    def get_unique_frame_index(self, camera_id: str, frame_idx: int) -> Optional[int]:
        """Get the unique frame index for a given camera and frame index.

        Returns a unique frame index that can be passed to rendering call, e.g. to apply the ISP of a training view,
        or None if the object type can not provide this index.
        """
        log.warning(f"get_unique_frame_index() not supported by {self.__class__.__name__}")
        return None


class SensorCalibProviderFromRigTrajectories(SensorCalibProvider):
    """Provides sensor intrinsics and poses extracted from a RigTrajectories instance"""

    def __init__(
        self,
        rig_trajectories: RigTrajectories,
        rig_translation_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rig_rotation_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        interpolate_poses: bool = False,
        rolling_shutter_duration: Optional[int] = None,
    ):
        super().__init__(unique_sensor_ids=rig_trajectories.camera_calibrations.keys())

        self._T_rig_world_base = rig_trajectories.T_world_base
        self._camera_calibrations = rig_trajectories.camera_calibrations
        self._rig_trajectory = rig_trajectories.rig_trajectories[0]  # Assuming only one trajectory per clip
        self._rig_pose_interpolator: Optional[ncore_transformations.PoseInterpolator] = None

        if self._rig_trajectory.cameras_linear_start_frame_indices is None:
            # get_unique_frame_index() will return None. Warn here to avoid repeated warnings per frame.
            log.warning("Unique frame indices for training frames are not available")

        # Get the timestamp range of the rig trajectory for some checks below.
        trajectory_start_us = int(self._rig_trajectory.T_rig_world_timestamps_us[0].item())
        trajectory_end_us = int(self._rig_trajectory.T_rig_world_timestamps_us[-1].item())

        # Check and fill in missing camera timestamps, and support a manually specified rolling shutter duration.
        # Maps each camera_id to a (num_frames, 2) array of frame start and end timestamps in microseconds.
        self._camera_frame_timestamps_us: dict[str, np.ndarray] = {}
        for unique_sensor_id, frame_timestamps_us in self._rig_trajectory.cameras_frame_timestamps_us.items():
            self._camera_frame_timestamps_us[unique_sensor_id] = fix_frame_timestamps(
                frame_timestamps_us.cpu().numpy(), trajectory_start_us, trajectory_end_us, rolling_shutter_duration
            )

        # Precompute the rig pose offset transformation from its parameters.
        self._rig_offset_se3 = pose_offsets_to_se3(rig_translation_offset, rig_rotation_offset)

        if interpolate_poses:
            log.info(f"{self.__class__.__name__}: Pose interpolation ENABLED")
            self._rig_pose_interpolator = ncore_transformations.PoseInterpolator(
                poses=self._rig_trajectory.T_rig_worlds.cpu(),
                timestamps=self._rig_trajectory.T_rig_world_timestamps_us.cpu(),
            )
        else:
            log.info(f"{self.__class__.__name__}: Pose interpolation DISABLED")
            if self._rig_trajectory.cameras_frame_T_rig_worlds is None:
                raise ValueError(
                    "Per-frame rig-to-world poses are missing from input trajectory data but is "
                    "required when pose interpolation is disabled."
                )

    def get_available_camera_ids(self) -> List[str]:
        return list(self._camera_ids)

    def get_num_camera_frames(self, camera_id: str) -> int:
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        return len(self._rig_trajectory.cameras_frame_timestamps_us[unique_sensor_id])

    def get_camera_calibration(self, camera_id: str) -> RigTrajectories.CameraCalibration:
        """Get the camera intrinsics and extrinsics for a selected camera."""
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        return self._camera_calibrations[unique_sensor_id]

    def get_camera_view_pose(self, camera_id: str, frame_idx: int) -> PoseRange:
        """Get the start/end poses and timestamps for a frame"""
        # Access the frame start and end timestamps for the given camera and frame index.
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        camera_timestamps_us = self._camera_frame_timestamps_us[unique_sensor_id]
        assert camera_timestamps_us.dtype == np.int64
        frame_start_us = camera_timestamps_us[frame_idx, 0]
        frame_end_us = camera_timestamps_us[frame_idx, 1]

        # Get the camera-to-rig pose from the camera calibration.
        camera_to_rig = self.get_camera_calibration(camera_id).T_sensor_rig.cpu().numpy()

        # Modify sensor extrinsics by the rig offset. This effectively shifts the cameras in the rig frame.
        # rig_offset_se3 should be identity when pose offsets are not specified (they should default to zero).
        camera_to_rig = self._rig_offset_se3 @ camera_to_rig

        if self._rig_pose_interpolator is not None:
            # Interpolate the test trajectory at the frame (start, end) timestamps to obtain frame start and end
            # rig-to-world poses for the frame to be rendered.
            rig_to_world = self._rig_pose_interpolator.interpolate_to_timestamps([frame_start_us, frame_end_us]).astype(
                np.float32
            )
        else:
            # Directly fetch the per-frame pose from the trajectory data instead of interpolating.
            # Downstream code expects float32 dtype.
            assert self._rig_trajectory.cameras_frame_T_rig_worlds is not None
            rig_to_world = (
                self._rig_trajectory.cameras_frame_T_rig_worlds[unique_sensor_id][frame_idx].numpy().astype(np.float32)
            )

        assert rig_to_world.shape == (2, 4, 4)
        assert rig_to_world.dtype == np.float32

        camera_to_world = PoseRange(
            start_pose_tquat_sensor_world=se3_matrix_to_tquat(rig_to_world[0] @ camera_to_rig),
            end_pose_tquat_sensor_world=se3_matrix_to_tquat(rig_to_world[1] @ camera_to_rig),
            start_timestamp_us=int(frame_start_us),
            end_timestamp_us=int(frame_end_us),
        )
        return camera_to_world

    def get_unique_frame_index(self, camera_id: str, frame_idx: int) -> Optional[int]:
        """Get the unique frame index for a given camera and frame index."""
        return get_unique_frame_index(
            camera_id, frame_idx, self._rig_trajectory.cameras_linear_start_frame_indices, self._sequence_id
        )

    def get_ncore_v3_poses(self) -> ncore_internal.data.v3.Poses:
        """Get the NCore poses from the provided rig trajectory"""
        # Note since the rig offsets is applied directly on T_sensor_rig, we do not modify the
        # T_rig_worlds here for consistency
        return ncore_internal.data.v3.Poses(
            T_rig_world_base=self._T_rig_world_base.numpy(),
            T_rig_worlds=self._rig_trajectory.T_rig_worlds.numpy(),
            T_rig_world_timestamps_us=self._rig_trajectory.T_rig_world_timestamps_us.numpy().astype(np.uint64),
        )


class SensorCalibProviderFromSensorTrajectories(SensorCalibProvider):
    """Provides sensor poses from the output of a RenderableModel.get_camera_trajectories() call
    and intrinsics from a RigTrajectories.CameraCalibration object, passable to a render call.
    """

    def __init__(
        self,
        camera_calibrations: OrderedDict[str, RigTrajectories.CameraCalibration],
        camera_trajectories: Dict[str, SensorTrajectory],
        cameras_start_frame_indices: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            camera_calibrations: A dictionary of camera calibrations (unique_camera_id -> CameraCalibration).
            camera_trajectories: A dictionary of camera trajectories (unique_camera_id -> SensorTrajectory).
            cameras_start_frame_indices: An optional dictionary of linear start frame indices (unique_camera_id -> int),
                needed by get_unique_frame_index(). If not provided, get_unique_frame_index() will return None.
        """
        # Camera IDs in the camera_calibrations and camera_trajectories should be the same.
        # But being tolerant in anticipation of differences of (currently unknown) edge cases.
        unique_camera_ids_calib = set(camera_calibrations.keys())
        unique_camera_ids_traj = set(camera_trajectories.keys())
        if unique_camera_ids_calib != unique_camera_ids_traj:
            log.warning(
                f"{self.__class__.__name__}: "
                f"Camera IDs in camera_calibrations and those in camera_trajectories do not match: "
                f"{list(camera_calibrations.keys())} vs. {list(camera_trajectories.keys())}"
            )

        unique_camera_ids = unique_camera_ids_calib.intersection(unique_camera_ids_traj)

        if cameras_start_frame_indices is None:
            # get_unique_frame_index() will return None. Warn here to avoid repeated warnings per frame.
            log.warning("Unique frame indices for training frames are not available")

        # No overlap between keys means that this class is completely disfunctional, so raise an error.
        if len(unique_camera_ids) == 0:
            raise ValueError("Camera IDs in camera calibrations and those in camera_trajectories do not overlap")

        self._camera_calibrations = OrderedDict((k, camera_calibrations[k]) for k in unique_camera_ids)
        self._camera_trajectories = {k: camera_trajectories[k] for k in unique_camera_ids}
        self._cameras_start_frame_indices = cameras_start_frame_indices

        super().__init__(unique_sensor_ids=unique_camera_ids)

    def get_available_camera_ids(self) -> List[str]:
        """Get the available camera ids that can be used to query poses"""
        return self._camera_ids

    def get_num_camera_frames(self, camera_id: str) -> int:
        """Get the number of frames available to query poses for a camera"""
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        return len(self._camera_trajectories[unique_sensor_id])

    def get_camera_view_pose(self, camera_id: str, frame_idx: int) -> PoseRange:
        """Get the start/end poses and timestamps for a frame"""
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        camera_trajectory = self._camera_trajectories[unique_sensor_id]
        poses = camera_trajectory.poses_startend_sensor_world[frame_idx].cpu().numpy()
        timestamps = camera_trajectory.timestamps_startend_us[frame_idx].cpu().numpy()
        assert poses.shape == (2, 4, 4)
        assert timestamps.shape == (2,)
        return PoseRange(
            start_pose_tquat_sensor_world=se3_matrix_to_tquat(poses[0]),
            end_pose_tquat_sensor_world=se3_matrix_to_tquat(poses[1]),
            start_timestamp_us=int(timestamps[0]),
            end_timestamp_us=int(timestamps[1]),
        )

    def get_camera_calibration(self, camera_id: str) -> RigTrajectories.CameraCalibration:
        """Get the camera intrinsics and extrinsics for a selected camera."""
        unique_sensor_id = self.get_unique_sensor_id(camera_id)
        return self._camera_calibrations[unique_sensor_id]

    def get_unique_frame_index(self, camera_id: str, frame_idx: int) -> Optional[int]:
        """Get the unique frame index for a given camera and frame index."""
        return get_unique_frame_index(camera_id, frame_idx, self._cameras_start_frame_indices, self._sequence_id)


@click.command("render")
@click.option(
    "--artifact-path",
    type=str,
    help="Path to an USDZ artifact file exported from training, e.g. last.usdz",
    default=None,
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help=(
        "Path to an output directory to render frames to. Structured as "
        "<output_dir>/<sensor_id>/<frame_name>.<extension>)."
    ),
    required=True,
    default=None,
)
@click.option(
    "--camera-id",
    "camera_ids",  # Tuple[str]
    multiple=True,
    type=str,
    help="Selection of a camera used during training by id to render from. "
    "Specify multiple times to render from multiple cameras, e.g. --camera-id cam1 --camera-id cam2."
    "Not specifying any ID is an error but it can be used to list the available cameras.",
    required=False,
)
@click.option(
    "--height",  # Keeping this option as is for CLI backward-compatibility.
    type=int,
    help="Height of the rendered frames in pixels for all selected cameras."
    " Can not be used together with --image-scale but one of the two must be specified.",
    required=False,
    default=None,
)
@click.option(
    "--image-scale",  # Allows for easy scaling of the output resolution over multiple cameras.
    type=float,
    help="Specify the output image resolution in proportion of the resolution of each camera, in the range (0.0, 1.0]."
    "The output resolution is calculated as floor(input_image_resolution * image_scale). "
    "The minimum output resolution is 2x2 pixels."
    " Can not be used together with --height but one of the two must be specified.",
    required=False,
    default=None,
)
@click.option(
    "--frame-step",
    type=int,
    help="Frame step size",
    default=1,
)
@click.option(
    "--rolling-shutter-duration",
    type=int,
    help=(
        "Elapsed time between the start and end of each frame in integer microseconds. "
        "If set, it overrides the frame start timestamps in the USDZ input to the frame end timestamps "
        "minus the specified frame time. The timestamps are then clamped within the trajectory timestamp range."
    ),
    required=False,
    default=None,
)
@click.option(
    "--image-format",
    type=click.Choice(["png", "jpg", "jpeg"]),
    help="png or jpeg",
    default="png",
)
@click.option(
    "--export-video",
    is_flag=True,
    default=False,
    help="Export rendered frames as an MP4 H.264 video per camera after rendering completes."
    " All rendered frames are loaded back to memory per camera (requires sufficient memory).",
)
@click.option(
    "--video-fps",
    type=float,
    default=30.0,
    help="Frames per second for exported videos. Must be a positive float. Only used when --export-video is set.",
)
@click.option(
    "--video-crf",
    type=int,
    default=20,
    help="Desired quality (Constant Rate Factor) of the exported videos for --export-video. "
    "Valid range is 0-51. Typical range is 18-28. Lower means higher quality.",
)
@click.option(
    "--renderer",
    type=click.Choice(["default", "gsplat", "nrend"], case_sensitive=False),
    default="default",
    help="Renderer backend: 'default' uses the artifact's trained renderer (PyTorch forward pass), "
    "'gsplat' forces GSplatRenderer, 'nrend' uses the fast NRendWrapper (direct C++/CUDA JIT).",
)
@click.option(
    "--enable-nrend",
    is_flag=True,
    default=False,
    hidden=True,
    help="Deprecated: use --renderer instead.",
)
@click.option(
    "--enable-editing-actors",
    is_flag=True,
    help="Enable the actor editing API to allow modifications to dynamic actor poses.",
    default=False,
)
@click.option(
    "--demo-actor-transform",
    is_flag=True,
    help="Demo: apply a precomputed transformation to actor poses. Requires --enable-editing-actors.",
    default=False,
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
    help="Rotation offsets (yaw, -roll, -pitch) in degrees in rig space to be applied to the rig prior to rendering.",
    default=(0.0, 0.0, 0.0),
)
@click.option(
    "--replicate-training-views/--no-replicate-training-views",
    help="If set, replicate the exact training views (intrinsics, poses, and post-processing) for rendering."
    " Overrides --calib-source to 'training-sensor-poses-calib'.",
    default=True,
)
@click.option(
    "--calib-source",
    type=click.Choice(
        [
            "training-rig-poses",
            "training-rig-poses-per-frame",
            "training-sensor-poses-nocalib",
            "training-sensor-poses-calib",
        ]
    ),
    help=(
        "Specifies the input source and to obtain the sensor-to-world pose for each frame to be rendered:\n"
        "'training-rig-poses' = Interpolate the training rig trajectory from the USDZ to obtain the rig-to-world "
        "poses at the frame start/end timestamps, then compose the camera-to-rig pose on top. \n"
        "'training-rig-poses-per-frame' - Use the already interpolated per-frame rig-to-world poses from the USDZ "
        "and compose the camera-to-rig pose on top (should give the same results as 'training-rig-poses').\n"
        "'training-sensor-poses-nocalib' - Get the unoptimized sensor-to-world training poses from the calib module "
        "that is initialized from the USDZ as part of the model.\n"
        "'training-sensor-poses-calib' - Get the optimized sensor-to-world training poses from the calib module if "
        "calib was enabled during training, otherwise fall back to the uncalibrated sensor poses."
        "If the required data is missing from the USDZ, or if the required model does not support the calib module, "
        "an error is raised, respectively.\n"
    ),
    default="training-rig-poses",
)
@click.option(
    "--frame-naming",
    type=click.Choice(["frame-end-timestamp", "contiguous-output-index"]),
    help=(
        "File naming scheme for exported frames: "
        "'frame-end-timestamp' - global absolute frame-end timestamp in microseconds, "
        "'contiguous-output-index' - frames always indexed 0,1,2,... per sensor, irrespective of --frame-step."
        "Different sensors are not necessarily synchronized, so frames camera1/000035.png and camera2/000035.png "
        "are not necessarily acquired at the same time. This setting is not recommended for evaluation/benchmarking."
    ),
    default="contiguous-output-index",
)
@click.option(
    "--custom-rig-trajectory",
    type=str,
    help="Optional path to a custom rig trajectory JSON file. "
    "If set, the provided rig trajectory is used for rendering instead of the training rig trajectory. "
    "Each camera in the provided JSON should have its unique sensor index set to a valid training sensor index "
    "for ISP post-processing to pick up the correct trained parameters per sensor.",
    default=None,
)
@click.option(
    "--max-pending-save-tasks",
    type=int,
    help="Max number of save tasks allowed to be pending; render loop blocks until below this.",
    default=64,
)
@click.option(
    "--save-workers",
    type=int,
    help="Number of workers to use for saving frames to disk.",
    default=16,
)
@cli.scopedtimer_cli_options(print_func=log.info)
@click.argument("hydra-overrides", nargs=-1)
@click.pass_context
def render(
    ctx: click.Context,
    artifact_path: str,
    output_dir: str,
    height: Optional[int],
    image_scale: Optional[float],
    camera_ids: Optional[Tuple[str]],
    frame_step: int,
    rolling_shutter_duration: Optional[int],
    image_format: str,
    export_video: bool,
    video_fps: float,
    video_crf: int,
    renderer: str,
    enable_nrend: bool,  # Deprecated: ignored, use --renderer instead
    enable_editing_actors: bool,
    demo_actor_transform: bool,
    rig_translation_offset: Tuple[float, float, float],
    rig_rotation_offset: Tuple[float, float, float],
    replicate_training_views: bool,
    calib_source: str,
    frame_naming: str,
    custom_rig_trajectory: Optional[str],
    max_pending_save_tasks: int,
    save_workers: int,
    hydra_overrides: Tuple[str, ...],
) -> None:
    """Render the model along the training trajectory or a custom rig trajectory for novel view rendering,
    with optional actor editing."""

    if enable_nrend:
        import warnings

        warnings.warn("--enable-nrend is deprecated, use --renderer nrend instead.", DeprecationWarning, stacklevel=2)
        if renderer == "default":
            renderer = "nrend"
    del enable_nrend  # Prevent accidental use — only --renderer controls renderer selection

    if video_fps <= 0.0:
        raise ValueError(f"--video-fps {video_fps} is not positive")
    if video_crf < 0 or video_crf > 51:
        raise ValueError(f"--video-crf {video_crf} is outside of the allowed range (0, 51)")
    if video_crf < 18 or video_crf > 28:
        log.warning(f"--video-crf {video_crf} is outside of the recommended range (18, 28).")

    if max_pending_save_tasks <= 0:
        raise ValueError(f"--max-pending-save-tasks {max_pending_save_tasks} is not greater than 0")

    if save_workers <= 0:
        raise ValueError(f"--save-workers {save_workers} is not greater than 0")

    os.makedirs(output_dir, exist_ok=True)

    # Capture, log, and save CLI settings
    collector = SettingsCollector.from_click_context(ctx, "render")
    collector.log_settings(log)
    collector.save_json(Path(output_dir) / "render_cli_args.json")

    if not Path(artifact_path).is_file():
        raise FileNotFoundError(artifact_path)

    if (image_scale is None and height is None) or (image_scale is not None and height is not None):
        raise ValueError("Exactly one of --image-scale or --height must be specified to define the output resolution.")

    if image_scale is not None and (image_scale <= 0.0 or image_scale > 1.0):
        raise ValueError(f"Image scale {image_scale} is outside of the allowed range (0.0, 1.0]")

    if height is not None and (height <= 0):
        raise ValueError(f"Height {height} is not positive")

    # Check if translation or rotation offsets are set
    has_pose_offset = any(x != 0.0 for x in rig_translation_offset) or any(x != 0.0 for x in rig_rotation_offset)

    if replicate_training_views:
        # Make sure that translation and rotation offsets are zero
        if has_pose_offset:
            raise ValueError(
                "--replicate-training-views is set but --rig-translation-offset or --rig-rotation-offset are non-zero. "
                "No pose offset is allowed when replicating training views."
            )
        # Check rolling shutter duration is not overridden
        if rolling_shutter_duration is not None:
            raise ValueError(
                "--replicate-training-views is set, but --rolling-shutter-duration is overridden. "
                "No override for rolling shutter is allowed when replicating training views."
            )
        # Check actors are not edited
        if enable_editing_actors or demo_actor_transform:
            raise ValueError(
                "--replicate-training-views is set, but --enable-editing-actors or --demo-actor-transform is also set. "
                "No actor editing is allowed when replicating training views."
            )
        # Check custom trajectory is not used
        if custom_rig_trajectory not in [None, "None", ""]:
            raise ValueError(
                "--replicate-training-views is set, but --custom-rig-trajectory is also set. "
                "Custom trajectory is not allowed when replicating training views."
            )

        # Training views can only be replicated with calibrated sensor poses if calibration was enabled during training,
        # otherwise rendered frames may not necessarily align with training frames
        # (depending on the amount of pose correction achieved during training).
        # If calib was disabled during training, this will fall back to uncalibrated sensor poses.
        calib_source = "training-sensor-poses-calib"
        log.warning(f"Overriding --calib-source to {calib_source} because --replicate-training-views is set.")

    if demo_actor_transform and not enable_editing_actors:
        raise ValueError("--demo-actor-transform requires --enable-editing-actors to be set")

    log.info(f"Loading artifact from {artifact_path}")
    artifact = Artifact(Path(artifact_path))

    # Get the rig trajectories of the training clip.
    training_clip_rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

    # Load custom rig trajectories if provided and set rendering rig trajectories accordingly
    if custom_rig_trajectory not in [None, "None", ""]:
        assert custom_rig_trajectory is not None  # Type narrowing for type checker
        log.info(f"Loading custom rig trajectories from {custom_rig_trajectory}")
        artifact.load_custom_rig_trajectories(custom_rig_trajectory)
        log.info("Using custom rig trajectories for rendering")
        rendering_rig_trajectories = RigTrajectories.from_dict(artifact.custom_rig_trajectories)
    else:
        log.info("Using training rig trajectories for rendering")
        rendering_rig_trajectories = training_clip_rig_trajectories

    log.info(f"Constructing a renderable model for scene id '{artifact.scene_id}' with renderer='{renderer}'")
    config_overrides_list = list(hydra_overrides)
    enable_nrend = False
    if renderer.lower() == "gsplat":
        config_overrides_list.append("model.renderer.name=3dgut-gsplat")
    elif renderer.lower() == "nrend":
        enable_nrend = True
    renderable = RenderableModel.load_from_artifact(
        artifact, enable_nrend=enable_nrend, config_overrides=tuple(config_overrides_list)
    )

    # Get the look-up table of unique frame indices that can be used to identify training frames.
    training_clip_cameras_start_frame_indices = training_clip_rig_trajectories.rig_trajectories[
        0
    ].cameras_linear_start_frame_indices

    # Get the timestamp range of the training clip (always needed for actor tracks).
    training_clip_start_us, training_clip_end_us = get_trajectory_time_range(training_clip_rig_trajectories)
    assert training_clip_start_us <= training_clip_end_us

    # Get the timestamp range for rendering (may differ from training if using custom trajectory).
    rendering_clip_start_us, rendering_clip_end_us = get_trajectory_time_range(rendering_rig_trajectories)
    assert rendering_clip_start_us <= rendering_clip_end_us

    # Instantiate a provider of sensor intrinsics and poses for rendering individual views.
    calib_provider: SensorCalibProvider
    if calib_source == "training-rig-poses" or calib_source == "training-rig-poses-per-frame":
        interpolate_poses = calib_source == "training-rig-poses"
        calib_provider = SensorCalibProviderFromRigTrajectories(
            rendering_rig_trajectories,
            rig_translation_offset,
            rig_rotation_offset,
            interpolate_poses,
            rolling_shutter_duration,
        )
    elif calib_source == "training-sensor-poses-nocalib" or calib_source == "training-sensor-poses-calib":
        calibrated = calib_source == "training-sensor-poses-calib"
        if has_pose_offset:
            # Do not allow rig pose offsets in this mode.
            # Pose offsets are supposed to shift all sensors consistently in the rig frame but the notion of a rigid
            # rig is lost when using sensor-to-world poses directly from the calib module because
            # camera views are optimized independently without any rigidity constraint ("free poses") currently.
            raise ValueError(
                "The rig pose offset set via --rig-translation-offset or --rig-rotation-offset is non-zero but "
                f"the selected calib source '{calib_source}' does not support it."
            )
        calib_provider = SensorCalibProviderFromSensorTrajectories(
            camera_calibrations=training_clip_rig_trajectories.camera_calibrations,
            camera_trajectories=renderable.get_camera_trajectories(calibrated=calibrated),
            cameras_start_frame_indices=training_clip_cameras_start_frame_indices,
        )
    else:
        raise ValueError(f"Unsupported pose input type '{calib_source}'")

    # Query cameras from the scene and select an existing trajectory and camera.
    available_camera_ids = calib_provider.get_available_camera_ids()
    if len(available_camera_ids) == 0:
        raise ValueError("No cameras found in the artifact")
    log.info("Available cameras:\n    " + "\n    ".join(available_camera_ids))

    if not camera_ids:
        render_camera_ids: List[str] = available_camera_ids
        log.info(f"No --camera-id specified, rendering all {len(render_camera_ids)} available cameras")
    else:
        render_camera_ids = list(camera_ids)

    # Make sure the requested cameras exist.
    for camera_id in render_camera_ids:
        if camera_id not in available_camera_ids:
            raise ValueError(f"Could not find the requested camera: '{camera_id}'. Only found {available_camera_ids}")

    actor_tracks: Optional[ActorTracks] = None
    if enable_editing_actors:
        actor_tracks = renderable.get_actor_tracks()
        log.info(f"Scene contains {actor_tracks.num_tracks()} (controllable) actor tracks")

    # Render frames per selected camera id or skip if no camera ids were specified.
    for camera_id in render_camera_ids:
        log.info(f"Selecting camera {camera_id}")
        if camera_id not in available_camera_ids:
            raise ValueError(f"Could not find the requested camera {camera_id}")

        requested_camera = calib_provider.get_camera_calibration(camera_id)

        camera_width, camera_height = requested_camera.camera_model_parameters.resolution
        log.info(f"Camera resolution: {camera_width}x{camera_height}")

        num_frames = calib_provider.get_num_camera_frames(camera_id)
        log.info(f"Test trajectory contains poses for {num_frames} video frames")

        cam_output_dir = os.path.join(output_dir, camera_id)
        os.makedirs(cam_output_dir, exist_ok=True)

        # Render frames along the test trajectory.
        # Post-render (PIL + disk I/O) runs in a thread pool to unblock the next render.
        contiguous_frame_output_index = 0
        save_futures: List[Any] = []
        output_frame_timestamps: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=save_workers) as executor:
            for frame_idx in range(0, num_frames, frame_step):
                # Get the camera-to-world pose of the current frame.
                camera_to_world = calib_provider.get_camera_view_pose(camera_id, frame_idx)

                frame_start_us = camera_to_world.start_timestamp_us
                frame_end_us = camera_to_world.end_timestamp_us
                assert frame_start_us >= 0 and frame_end_us >= 0

                log.info(
                    f"Frame {frame_idx} at timestamp {frame_start_us} with {frame_end_us - frame_start_us} us exposure"
                )

                # Test whether the provided frame capture timestamps are within the timestamp range of the rendering clip.
                # Actors do not render correctly beyond this timestamp range.
                if frame_start_us < rendering_clip_start_us or frame_end_us > rendering_clip_end_us:
                    raise ValueError(
                        f"Frame capture timestamp range"
                        f"({frame_start_us}, {frame_end_us}) is outside of rendering clip"
                        f"timestamp range ({rendering_clip_start_us}, {rendering_clip_end_us})."
                    )

                # Only update actors if enable_editing_actors is enabled
                # Otherwise, skip to ensure all objects are rendered without modification.
                actors_snapshot: Optional[ActorsSnapshot] = None
                if enable_editing_actors and renderable.supports_edit_actors():
                    assert actor_tracks is not None
                    actors_snapshot = actor_tracks.get_snapshot_at_frame(frame_start_us, frame_end_us)
                    log.info(f"{actors_snapshot.num_actors()} active actors")
                    if demo_actor_transform:
                        actors_snapshot = manipulate_actor_poses(actors_snapshot)

                start_time = time.perf_counter()

                if height is not None:
                    # Render frames with a fixed height in pixels for all cameras.
                    # Setting width to 1 will set it internally such that the aspect ratio is preserved.
                    # Keeping this for CLI backward-compatibility.
                    render_width = 1
                    render_height = height
                elif image_scale is not None:
                    # Calculate the output resolution based on the camera resolution and the scale factor.
                    render_width = int(camera_width * image_scale)
                    render_height = int(camera_height * image_scale)
                    if render_width < 2 or render_height < 2:
                        raise ValueError(
                            f"Output resolution {render_width}x{render_height} is too low. "
                            "The minimum output resolution is 2x2 pixels. Consider increasing the --image-scale option."
                        )
                else:
                    raise ValueError("Either --height or --image-scale must be specified.")

                unique_frame_idx = (
                    calib_provider.get_unique_frame_index(camera_id, frame_idx) if replicate_training_views else None
                )

                # Render the scene from the given viewpoint, using the given intrinsics.
                camera_frame = renderable.render_camera_frame(
                    camera_intrinsics=requested_camera.camera_model_parameters,
                    camera_to_world=camera_to_world,
                    resolution=(render_width, render_height),
                    unique_sensor_idx=requested_camera.unique_sensor_idx,
                    unique_frame_idx=unique_frame_idx,
                    actors_snapshot=actors_snapshot,
                    frame_start_us=frame_start_us,
                    frame_end_us=frame_end_us,
                )
                elapsed_time = time.perf_counter() - start_time

                if camera_frame.color_image is None:
                    log.warning(f"Renderer did not return a color image and took {elapsed_time * 1e3:.3f} ms")
                else:
                    result_height, result_width = camera_frame.color_image.shape[:2]
                    log.info(f"{result_width}x{result_height} camera frame rendered in {elapsed_time * 1e3:.3f} ms")

                    # Pop the oldest future if the max number of pending save tasks is reached.
                    # Ignore uneven latency for simplicity.
                    while len(save_futures) >= max_pending_save_tasks:
                        save_futures.pop(0).result()

                    color_image_u8 = (camera_frame.color_image * 255).clamp(0, 255).to(torch.uint8)
                    numpy_array = color_image_u8.cpu().numpy().copy()

                    match frame_naming:
                        case "frame-end-timestamp":
                            file_name = f"{frame_end_us}.{image_format}"
                        case "contiguous-output-index":
                            file_name = f"{contiguous_frame_output_index:06d}.{image_format}"
                        case _:
                            raise ValueError(f"Invalid frame naming scheme: {frame_naming}")

                    image_path = os.path.join(cam_output_dir, file_name)

                    log.info(f"Saving image to {image_path}")

                    output_frame_timestamps.append(
                        {
                            "file_name": file_name,
                            "render_frame_idx": contiguous_frame_output_index,
                            "frame_start_timestamp_us": frame_start_us,
                            "frame_end_timestamp_us": frame_end_us,
                        }
                    )

                    save_futures.append(
                        executor.submit(
                            _save_rendered_frame_to_disk,
                            numpy_array,
                            image_path,
                        )
                    )

                contiguous_frame_output_index += 1

        # Wait for all frames to be saved to disk.
        for future in save_futures:
            future.result()

        # Save the per-frame timestamps used to render the images to a JSON file.
        with open(os.path.join(cam_output_dir, "timestamps.json"), "w") as f:
            json.dump(output_frame_timestamps, f, indent=2)

    if export_video:
        log.info("Exporting videos...")
        export_videos_and_json(
            output_dir,
            render_camera_ids,
            fps=video_fps,
            crf=video_crf,
            image_extension=image_format.lower(),
        )

    log.info(f"Rendering complete. Output: {output_dir}")
