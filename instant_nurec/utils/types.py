# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional, OrderedDict, Self

from dataclasses import dataclass
from enum import IntFlag, auto

from instant_nurec.utils import se3 as lt
import numpy as np
import numpy.typing as npt
import torch

from ncore.data import ConcreteCameraModelParametersUnion



@dataclass(slots=True)
class HalfClosedInterval:
    """Represents a closed interval [start, end)"""

    start: int
    end: int

    def __post_init__(self) -> None:
        assert self.start <= self.end

    def intersection(self, other: HalfClosedInterval) -> Optional[HalfClosedInterval]:
        """Computes the intersection of two half-closed interval"""
        if other.start >= self.end or other.end <= self.start:
            return None

        return HalfClosedInterval(max(self.start, other.start), min(self.end, other.end))



@dataclass(slots=True, kw_only=True)
class FrameConversion:
    """Represents parameters and functions to convert frame-associated data between different (potentially uniformly scaled) canonical 3d frames"""

    #: Homogeneous source -> target transformation matrix; its dtype declares the output dtype of this conversion.
    #:
    #: ⎡ R  -o ⎤
    #: ⎣ 0 1/s ⎦
    #:
    #: with
    #: - R: source -> target frame orientation with det(R)=1 (3,3)
    #: - o: origin of the target frame in the source frame (in source-frame units) (3,1)
    #: - s: the source -> target scale
    matrix: npt.NDArray[np.floating]

    def __post_init__(self):
        assert self.matrix.shape == (4, 4)
        if not np.issubdtype(self.matrix.dtype, np.floating):
            raise TypeError(f"Expected floating point matrix dtype, but got {self.matrix.dtype}")
        assert self.matrix[3, 3] > 0.0
        assert np.isclose(np.linalg.det(self.matrix[:3, :3]), 1.0)

    @property
    def dtype(self) -> np.dtype:
        """Returns the declared output dtype of this conversion, taken from the underlying matrix."""
        return self.matrix.dtype

    @property
    def target_scale(self) -> float:
        """The uniform scale of the target frame relative to the source frame"""
        return 1 / self.matrix[3, 3]

    def get_transformation_matrices(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Returns scale-aware (4,4) matrices T / S, which can be used to transform

        - source *points* / *vectors* x_source to the target frame via

          x_target = T @ x_source

        - source *poses* P_source to the target frame via

          P_target = T @ P_source @ S

          Resulting poses have target frame scale when incorporating S, or source frame scale if omitting S

        Both returned matrices have dtype == self.dtype.
        """

        # T has the form
        # ⎡ s*R -s*o ⎤
        # ⎣ 0    1   ⎦
        T = self.matrix.copy()
        T *= self.target_scale

        # S has the form
        # ⎡ 1/s*I 0 ⎤
        # ⎣ 0     1 ⎦
        inv_s = self.matrix[3, 3]
        S = np.zeros((4, 4), dtype=self.dtype)
        np.fill_diagonal(S, [inv_s, inv_s, inv_s, 1.0])

        return (T, S)

    def transform_poses(
        self,
        T_poses_source: np.ndarray,
    ) -> np.ndarray:
        """Transforms poses in the source frame to corresponding poses in the target frame.

        Returned poses have target frame units and dtype == self.dtype. Inputs are cast to self.dtype
        before the matmul so the computation itself happens in the declared dtype (no silent numpy
        promotion, no post-hoc downcast).

        Supports both singular (4,4) and batched (N,4,4) input poses 'T_poses_source'
        """

        # Cast to self.dtype first so the matmul runs in the declared dtype.
        T_poses = T_poses_source.astype(self.dtype, copy=False).reshape((-1, 4, 4))  # (N,4,4)

        # apply transformation
        T, S = self.get_transformation_matrices()
        T_poses = T @ T_poses @ S

        # unbatch dimensions conditionally
        return T_poses.squeeze()  # (N,4,4) or (4,4)


@dataclass(slots=True, kw_only=True)
class RigTrajectories:
    """Represents a list of rig trajectories (using NCore frame conventions)"""

    # NCore world frame -> base frame rigid transformation (potentially geo-located)
    T_world_base: torch.Tensor

    # NCore world -> scene-frame conversion
    world_to_scene: FrameConversion

    @dataclass(slots=True, kw_only=True)
    class RigTrajectory:
        """Represents a single rig trajectory with associated sensor and frame timestamps"""

        sequence_id: str  # the source sequence id of the current trajectory (might be shared with other trajectories)

        cameras_frame_timestamps_us: dict[str, torch.Tensor]

        # Timestamped trajectory of the rig frame in NCore world coordinates
        T_rig_worlds: torch.Tensor
        T_rig_world_timestamps_us: torch.Tensor

        def __post_init__(self):
            assert self.T_rig_world_timestamps_us.ndim == 1, "T_rig_world_timestamps_us must be 1D"
            assert len(self.T_rig_worlds) == len(self.T_rig_world_timestamps_us)
            assert all(
                camera_frame_timestamps_us.shape[1:] == (2,)
                for camera_frame_timestamps_us in self.cameras_frame_timestamps_us.values()
            )

    rig_trajectories: list[RigTrajectory]  # indexed by trajectory index

    @dataclass(slots=True, kw_only=True)
    class SensorCalibration:
        """Represents a generic sensor-associated calibration"""

        sequence_id: str  # sequence id
        unique_sensor_idx: int  # unique sensor index (of this associated sensor type!)

        T_sensor_rig: torch.Tensor  # extrinsics 4x4

    @dataclass(slots=True, kw_only=True)
    class CameraCalibration(SensorCalibration):
        """Represents a camera-associated calibration"""

        camera_model_parameters: ConcreteCameraModelParametersUnion  # intrinsics [available unconditionally]

    camera_calibrations: OrderedDict[str, CameraCalibration]  # indexed by *unique* camera sensor ids

    def __post_init__(self):
        # make sure sensors referenced by trajectories are available
        for rig_trajectory in self.rig_trajectories:
            for camera_id in rig_trajectory.cameras_frame_timestamps_us.keys():
                assert camera_id in self.camera_calibrations, f"Missing camera {camera_id} in camera calibrations"



class TrackFlags(IntFlag):
    """Bitmask flags of per-track properties (note: limited to 32 variants)"""

    # Special value without any set flag
    NONE = 0

    # Dynamic flags in accordance with the dataset loader
    DYNAMIC = auto()


@dataclass(kw_only=True, slots=True)
class TracksData:
    """
    Data-components of instant_nurec.datasets.tracks.Tracks.

    Args:
        tracks_id: list[str]  - (N_tracks) string identifiers of each track
        max_track_n_poses: int  - maximum number of poses for an individual track among all tracks (used within kernels for shared memory allocations)
        tracks_packinfo: torch.Tensor  - (N_tracks x 2 containing) with [track_start_idx, N_track_poses] each
        tracks_poses: lt.SE3  - (N_total_poses, ) containing SE3 poses
        tracks_timestamps_us: torch.Tensor  - (N_total_poses, ) containing per-pose timestamps
        tracks_flags: torch.Tensor  # (N_tracks) containing per-track flags int32 values (see TrackFlags)
    """

    tracks_id: list[str]
    max_track_n_poses: int
    tracks_packinfo: torch.Tensor
    tracks_poses: lt.SE3
    tracks_timestamps_us: torch.Tensor
    tracks_flags: torch.Tensor

    def __post_init__(self):
        """Post-init validation of the tracks data"""

        if self.tracks_packinfo.ndim != 2:
            raise ValueError(
                f"Track packinfo must have shape (N_tracks, 2), but has shape {self.tracks_packinfo.shape}"
            )
        if self.tracks_packinfo.shape[0] != self.n_tracks:
            raise ValueError(
                f"Number of tracks ({self.n_tracks}) does not match number of track packinfo ({self.tracks_packinfo.shape[0]})"
            )
        if self.tracks_packinfo.shape[1] != 2:
            raise ValueError(
                f"Track packinfo must have shape (N_tracks, 2), but has shape {self.tracks_packinfo.shape}"
            )

        n_total_poses = self.tracks_poses.shape[0]
        if self.tracks_timestamps_us.shape != (n_total_poses,):
            raise ValueError(
                f"Number of total poses ({n_total_poses}) does not match number of track timestamps ({self.tracks_timestamps_us.shape})"
            )
        if self.tracks_flags.shape != (self.n_tracks,):
            raise ValueError(
                f"Number of tracks ({self.n_tracks}) does not match number of track flags ({self.tracks_flags.shape})"
            )
        if self.tracks_flags.dtype != torch.int32:
            raise ValueError(f"Track flags must be of type torch.int32, but is {self.tracks_flags.dtype}")

    @property
    def n_tracks(self) -> int:
        return len(self.tracks_id)

    def to_device(self, device: torch.device) -> Self:
        return self.__class__(
            tracks_id=self.tracks_id,
            max_track_n_poses=self.max_track_n_poses,
            tracks_packinfo=self.tracks_packinfo.to(device),
            tracks_poses=self.tracks_poses.to(device),
            tracks_timestamps_us=self.tracks_timestamps_us.to(device),
            tracks_flags=self.tracks_flags.to(device),
        )


@dataclass(kw_only=True, slots=True)
class CuboidTracksData:
    """
    Data-components of instant_nurec.datasets.tracks.CuboidTracks.

    Args:
        cuboids_dims: torch.Tensor  - (N_tracks, 3) containing per-track dimensions in local track-frame
    """

    cuboids_dims: torch.Tensor

    def __post_init__(self):
        """Post-init validation of the cuboid tracks data"""
        if self.cuboids_dims.ndim != 2 or self.cuboids_dims.shape[1] != 3:
            raise ValueError(f"Cuboids dims must have shape (N_tracks, 3), but has shape {self.cuboids_dims.shape}")
        if self.cuboids_dims.dtype != torch.float32:
            raise ValueError(f"Cuboids dims must be of type torch.float32, but is {self.cuboids_dims.dtype}")

    @property
    def n_tracks(self) -> int:
        return self.cuboids_dims.shape[0]

    def to_device(self, device: torch.device) -> Self:
        return self.__class__(
            cuboids_dims=self.cuboids_dims.to(device),
        )


@dataclass(kw_only=True, slots=True)
class CuboidTracksDataPack:
    """Aggregation of a TracksData and CuboidTracksData pair.

    Args:
        tracks_data: TracksData  - (N_tracks) containing per-track flags int32 values (see TrackFlags)
        cuboidtracks_data: CuboidTracksData  - (N_tracks, 3) containing per-track dimensions in local track-frame
    """

    tracks_data: TracksData
    cuboidtracks_data: CuboidTracksData

    def to_device(self, device: torch.device) -> Self:
        return self.__class__(
            tracks_data=self.tracks_data.to_device(device),
            cuboidtracks_data=self.cuboidtracks_data.to_device(device),
        )

    def __post_init__(self):
        """Post-init validation of the cuboid tracks data"""
        if self.tracks_data.n_tracks != self.cuboidtracks_data.n_tracks:
            raise ValueError(
                f"Number of tracks ({self.tracks_data.n_tracks}) does not match number of cuboid tracks ({self.cuboidtracks_data.n_tracks})"
            )
