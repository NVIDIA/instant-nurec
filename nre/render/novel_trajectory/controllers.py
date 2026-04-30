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

import logging

from abc import ABC, abstractmethod
from typing import Any, Generator, TypeAlias

import lietorch as lt
import numpy as np

from scipy.spatial.transform import Rotation

from ncore.impl.common.transformations import PoseInterpolator
from nre.datasets.tracks import CuboidTracks
from nre.utils.geometry import se3_matrix_to_se3, se3_matrix_to_tquat
from nre.utils.types import RigTrajectories


logger = logging.getLogger(__file__)


Timestamp: TypeAlias = tuple[int, int]


class Controller(ABC):
    """Controller for ego and actor motion"""

    def __init__(self, start_timestamp_us: int) -> None:
        self.current_timestamp_us = start_timestamp_us

    @abstractmethod
    def timestamp_generator(self) -> Generator[Timestamp, None, None]:
        """Generate timestamps for ego pose"""
        ...

    def last_timestamp_us(self) -> int:
        """Get the last timestamp of the ego controller"""
        last_timestamp_us = self.current_timestamp_us
        for timestamp_us in self.timestamp_generator():
            last_timestamp_us = timestamp_us[0]
        return last_timestamp_us


class EgoController(Controller):
    """Controller for ego (rig) pose management"""

    @abstractmethod
    def rig_pose_interpolator(self, timestamp_us: Timestamp) -> np.ndarray:
        """Interpolate the rig pose at the given timestamps"""
        ...

    @abstractmethod
    def rig_offset_se3(self, frame_index: int | None = None, timestamp_us: Timestamp | None = None) -> np.ndarray:
        """Get the rig offset SE3 matrix"""
        ...


class ActorController(Controller):
    """Controller for dynamic actor management"""

    @abstractmethod
    def dynamic_actors_interpolator(self, frame_index: int, timestamp_us: Timestamp) -> list[dict[str, Any]]:
        """Interpolate the dynamic actors at the given timestamps"""
        ...

    @abstractmethod
    def dynamic_actors_offset_se3(self, frame_index: int, timestamp_us: Timestamp, cuboid_id: str) -> np.ndarray:
        """Get the offset SE3 matrix for the dynamic actors at the given timestamps"""
        ...


class ReplayEgoController(EgoController):
    """Replays ego motion from recorded trajectory"""

    def __init__(
        self, trajectory: RigTrajectories.RigTrajectory, camera_id: str, duration_s: float | None, start_us: int
    ) -> None:
        super().__init__(start_us)
        self.trajectory = trajectory
        self.camera_id = camera_id
        self.duration_us = duration_s * 1_000_000 if duration_s is not None else None

        self.pose_interpolator = PoseInterpolator(
            self.trajectory.T_rig_worlds.cpu().numpy(), self.trajectory.T_rig_world_timestamps_us.cpu().numpy()
        )

        self.camera_timestamps = self.trajectory.cameras_frame_timestamps_us[self.camera_id].cpu().numpy()

    def timestamp_generator(self) -> Generator[Timestamp, None, None]:
        for i in range(self.camera_timestamps.shape[0]):
            timestamp_us = self.camera_timestamps[i]
            if timestamp_us[0] < self.current_timestamp_us:
                continue
            if self.duration_us is not None and timestamp_us[1] > self.current_timestamp_us + self.duration_us:
                break
            yield (timestamp_us[0], timestamp_us[1])

    def rig_pose_interpolator(self, timestamp_us: Timestamp) -> np.ndarray:
        return self.pose_interpolator.interpolate_to_timestamps(timestamp_us)

    def rig_offset_se3(self, frame_index: int | None = None, timestamp_us: Timestamp | None = None) -> np.ndarray:
        return np.eye(4, dtype=np.float32)


class SpiralEgoController(ReplayEgoController):
    """Adds spiral motion to ego pose"""

    def __init__(
        self,
        trajectory: RigTrajectories.RigTrajectory,
        camera_id: str,
        duration_s: float,
        start_us: int,
        spiral_num_spirals: int,
        spiral_radius: float,
        spiral_x_scale: float,
        spiral_y_scale: float,
        spiral_z_scale: float,
        framerate: float = 30.0,
        stop_on_start_pose: bool = True,
    ) -> None:
        super().__init__(trajectory, camera_id, duration_s, start_us)

        self.framerate = framerate
        self.spiral_num_frames = int(duration_s * self.framerate)
        spiral_angles = np.linspace(0, 2 * np.pi * spiral_num_spirals, self.spiral_num_frames)

        self.spiral_offset = np.stack(
            [
                spiral_radius * np.cos(spiral_angles) * spiral_x_scale,
                spiral_radius * np.cos(spiral_angles) * spiral_y_scale - spiral_radius * spiral_y_scale,
                spiral_radius * np.sin(spiral_angles) * spiral_z_scale,
            ],
            axis=-1,
        )

        self.stop_on_start_pose = stop_on_start_pose

    def timestamp_generator(self) -> Generator[Timestamp, None, None]:
        # Calculate frame duration based on framerate
        frame_duration_us = int(1_000_000 / self.framerate)
        for i in range(self.spiral_num_frames):
            if self.stop_on_start_pose:
                yield (self.current_timestamp_us, self.current_timestamp_us + 1)
            else:
                frame_start_us = self.current_timestamp_us + i * frame_duration_us
                frame_end_us = frame_start_us + frame_duration_us
                yield (frame_start_us, frame_end_us)

    def rig_offset_se3(self, frame_index: int | None = None, timestamp_us: Timestamp | None = None) -> np.ndarray:
        offset = np.eye(4, dtype=np.float32)
        if frame_index is not None and frame_index < len(self.spiral_offset):
            offset[:3, 3] = self.spiral_offset[frame_index]
        return offset


class StopEgoController(ReplayEgoController):
    """Stops ego motion at a fixed pose"""

    def __init__(
        self,
        trajectory: RigTrajectories.RigTrajectory,
        camera_id: str,
        duration_s: float,
        start_us: int,
        framerate: float = 30.0,
    ) -> None:
        super().__init__(trajectory, camera_id, None, start_us)  # No duration limit for replay data

        self.framerate = framerate
        self.num_frames = int(duration_s * self.framerate)

    def timestamp_generator(self) -> Generator[Timestamp, None, None]:
        for _ in range(self.num_frames):
            yield (self.current_timestamp_us, self.current_timestamp_us + 1)


class ReplayActorController(ActorController):
    """Replays dynamic actor motion from recorded trajectory"""

    def __init__(
        self, cuboid_tracks: CuboidTracks, duration_s: float | None, start_us: int, camera_timestamps: np.ndarray
    ) -> None:
        super().__init__(start_us)
        self.duration_us = duration_s * 1_000_000 if duration_s is not None else None
        self.camera_timestamps = camera_timestamps

        interpolator_list = []
        cuboid_tracks_list = []
        for track_idx in range(cuboid_tracks.n_tracks):
            cuboid_track = CuboidTracks.Ops.subset_from_indices(cuboid_tracks, [track_idx])
            cuboid_track = CuboidTracks.Ops.clone(cuboid_track)

            cuboid_tracks_list.append(cuboid_track)
            interpolator_list.append(
                PoseInterpolator(
                    poses=cuboid_track.tracks_poses.matrix().cpu(), timestamps=cuboid_track.tracks_timestamps_us.cpu()
                )
            )

        self.cuboid_tracks_list = cuboid_tracks_list
        self.interpolator_list = interpolator_list

    def timestamp_generator(self) -> Generator[Timestamp, None, None]:
        for i in range(self.camera_timestamps.shape[0]):
            timestamp_us = self.camera_timestamps[i]
            if timestamp_us[0] < self.current_timestamp_us:
                continue
            if self.duration_us is not None and timestamp_us[1] > self.current_timestamp_us + self.duration_us:
                break
            yield (timestamp_us[0], timestamp_us[1])

    def dynamic_actors_interpolator(self, frame_index: int, timestamp_us: Timestamp) -> list[dict[str, Any]]:
        dynamic_actors = []
        for cuboid_index, (cuboid_track, interpolator) in enumerate(
            zip(self.cuboid_tracks_list, self.interpolator_list)
        ):
            if (
                timestamp_us[0] < cuboid_track.tracks_timestamps_us.min()
                or timestamp_us[1] > cuboid_track.tracks_timestamps_us.max()
            ):
                continue

            interpolated_world_poses = interpolator.interpolate_to_timestamps(timestamp_us)
            assert interpolated_world_poses.shape == (2, 4, 4)
            assert interpolated_world_poses.dtype == np.float32

            delta_matrix = se3_matrix_to_se3(
                self.dynamic_actors_offset_se3(frame_index, timestamp_us, cuboid_track.tracks_id[0])
            ).cuda()
            frame_start_actor_to_world = lt.SE3(se3_matrix_to_tquat(interpolated_world_poses[0])).cuda() * delta_matrix
            frame_end_actor_to_world = lt.SE3(se3_matrix_to_tquat(interpolated_world_poses[1])).cuda() * delta_matrix

            dynamic_actors.append(
                {
                    "track_id": cuboid_track.tracks_id[0],
                    "frame_start_actor_to_world": frame_start_actor_to_world,
                    "frame_end_actor_to_world": frame_end_actor_to_world,
                }
            )
        return dynamic_actors

    def dynamic_actors_offset_se3(self, frame_index: int, timestamp_us: Timestamp, cuboid_id: str) -> np.ndarray:
        return np.eye(4, dtype=np.float32)


class StopActorController(ReplayActorController):
    """Stops dynamic actors at their current positions"""

    def __init__(
        self,
        cuboid_tracks: CuboidTracks,
        duration_s: float,
        start_us: int,
        camera_timestamps: np.ndarray,
        framerate: float = 30.0,
    ) -> None:
        super().__init__(cuboid_tracks, duration_s, start_us, camera_timestamps)

        self.framerate = framerate
        self.num_frames = int(duration_s * self.framerate)

    def timestamp_generator(self) -> Generator[Timestamp, None, None]:
        for _ in range(self.num_frames):
            yield (self.current_timestamp_us, self.current_timestamp_us + 1)


class SpinningActorController(ReplayActorController):
    """Spins dynamic actors around a fixed point"""

    def __init__(
        self,
        cuboid_tracks: CuboidTracks,
        duration_s: float,
        start_us: int,
        camera_timestamps: np.ndarray,
        framerate: float = 30.0,
        n_spins: int = 2,
    ) -> None:
        super().__init__(cuboid_tracks, duration_s, start_us, camera_timestamps)

        self.framerate = framerate
        self.num_frames = int(duration_s * self.framerate)
        self.degrees_per_frame = n_spins * 360 / self.num_frames

        self.cuboid_id_to_index = {cuboid_id: i for i, cuboid_id in enumerate(cuboid_tracks.tracks_id)}

    def dynamic_actors_offset_se3(self, frame_index: int, timestamp_us: Timestamp, cuboid_id: str) -> np.ndarray:
        offset = np.eye(4, dtype=np.float32)
        offset[:3, :3] = Rotation.from_euler("z", self.degrees_per_frame * frame_index, degrees=True).as_matrix()
        return offset
