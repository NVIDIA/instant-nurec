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

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from ncore.impl.common.transformations import PoseInterpolator
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.utils.geometry import se3_matrix_to_tquat


# TODO: Test on static scenes whether this works with zero tracks
@dataclass(frozen=True)  # Frozen to guarantee integrity via the __post_init_() check at all times
class ActorsSnapshot:
    """Class to represent the poses of each controllable actors present at the start and end of a single frame capture

    Each actor has two poses because of rolling shutter support.
    The contents are specific to the time range used to obtain this "snapshot", e.g. from ActorTracks.
    """

    # TODO: Switch to tensor of repeatable indices (~repeated actor instances) for a full vectorization.
    actor_ids: List[str]  # list of length M, Track ID of each actor present within the frame capture time interval
    actor_poses: torch.Tensor  # [M, 2, 7], start and end tquat pose of each actor's cuboid, [tx,ty,tz,qx,qy,qz,qw]
    # poses[i,0,:] is the start pose and poses[i,1,:] is the end pose of the actor with ID ids[i].

    @classmethod
    def empty(cls):
        return cls(actor_ids=[], actor_poses=torch.empty((0, 2, 7), dtype=torch.float32))

    def __post_init__(self):
        if len(self.actor_ids) > 0 or self.actor_poses.numel() > 0:
            assert self.actor_poses.shape == (len(self.actor_ids), 2, 7)

    def num_actors(self) -> int:
        """Number of actors contained in the snapshot"""
        return len(self.actor_ids)


@dataclass(frozen=True)
class ActorTracks:
    """Class representing controllable actors in the scene and their trajectories"""

    # These are not part of the public API, the user is not meant to access these directly.
    # TODO: vectorize these
    _cuboid_tracks_list: List[CuboidTracks] = field(default_factory=list)
    _interpolator_list: List[PoseInterpolator] = field(default_factory=list)

    # TODO: Test on static scenes whether this works with zero tracks
    @staticmethod
    def _from_cuboid_tracks(cuboid_tracks: CuboidTracks):
        """Private factory method, only use internally"""
        controllable_cuboid_tracks = CuboidTracks.Ops.subset_from_mask(
            cuboid_tracks, cuboid_tracks.get_mask_flags_all(TrackFlags.CONTROLLABLE)
        )
        interpolator_list: List[PoseInterpolator] = []
        cuboid_tracks_list: List[CuboidTracks] = []
        for track_idx in range(controllable_cuboid_tracks.n_tracks):
            cuboid_track = CuboidTracks.Ops.subset_from_indices(controllable_cuboid_tracks, [track_idx])
            cuboid_track = CuboidTracks.Ops.clone(cuboid_track)
            cuboid_tracks_list.append(cuboid_track)
            interpolator_list.append(
                PoseInterpolator(
                    poses=cuboid_track.tracks_poses.matrix().cpu(), timestamps=cuboid_track.tracks_timestamps_us.cpu()
                )
            )
        return ActorTracks(_cuboid_tracks_list=cuboid_tracks_list, _interpolator_list=interpolator_list)

    def num_tracks(self) -> int:
        """Get the number of tracks"""
        return len(self._cuboid_tracks_list)

    # TODO: Test on static scenes whether this works with zero tracks
    def get_snapshot_at_frame(self, frame_start_timestamp_us: int, frame_end_timestamp_us: int) -> ActorsSnapshot:
        """Return the interpolated poses of all actors that are present in the scene at a given timestamp"""
        assert len(self._cuboid_tracks_list) == len(self._interpolator_list)
        if frame_start_timestamp_us > frame_end_timestamp_us:
            raise ValueError(
                f"invalid timestamps, start time is after end time "
                f"({frame_start_timestamp_us}>{frame_end_timestamp_us})"
            )

        if len(self._cuboid_tracks_list) == 0:
            return ActorsSnapshot.empty()

        track_ids: List[str] = []
        poses_list = []  # List of [2,7] tensors containing start, end pose at start, end frame capture timestamp
        for cuboid_track, interpolator in zip(self._cuboid_tracks_list, self._interpolator_list):
            if (
                frame_start_timestamp_us >= cuboid_track.tracks_timestamps_us.min()
                and frame_end_timestamp_us <= cuboid_track.tracks_timestamps_us.max()
            ):
                world_poses = interpolator.interpolate_to_timestamps([frame_start_timestamp_us, frame_end_timestamp_us])
                assert isinstance(world_poses, np.ndarray)
                assert world_poses.dtype == np.float32
                assert world_poses.shape == (2, 4, 4)
                world_poses_tquat = se3_matrix_to_tquat(world_poses)
                assert world_poses_tquat.shape == (2, 7)

                track_ids.append(cuboid_track.tracks_id[0])
                poses_list.append(world_poses_tquat)

        return (
            ActorsSnapshot(actor_ids=track_ids, actor_poses=torch.stack(poses_list, dim=0))
            if len(poses_list) > 0  # avoid torch.stack([]) which fails with 'stack expects a non-empty TensorList'
            else ActorsSnapshot.empty()
        )
