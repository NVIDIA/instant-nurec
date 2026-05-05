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

from dataclasses import dataclass
from typing import Literal, Self, cast

import numpy as np
import torch

from instant_nurec.utils import se3 as lt

from instant_nurec.datasets.ray_intersections import (
    point_cuboidtracks_intersection_interpolate_pose as _point_cuboidtracks_intersection_interpolate_pose,
    ray_cuboidtracks_intersection as _ray_cuboidtracks_intersection,
)
from instant_nurec.utils.packed_ops import (
    packed_searchsorted_indexed_vals as _packed_searchsorted_indexed_vals,
)
from instant_nurec.utils.geometry import se3_matrix_to_tquat
from instant_nurec.utils.misc import get_pack_info_from_n
from instant_nurec.utils.packed_ops import linstep_interleave
from instant_nurec.utils.types import CuboidTracksData, CuboidTracksDataPack, TrackFlags, TracksData


@dataclass(kw_only=True, slots=True)
class Tracks:
    """Manages time-dependent poses and related interpolation / intersection GPU operations for tracks

    This class uses the :class:`instant_nurec.utils.types.TracksData` class for storing data.
    """

    tracks_data: TracksData

    @property
    def tracks_id(self) -> list[str]:
        return self.tracks_data.tracks_id

    @property
    def max_track_n_poses(self) -> int:
        return self.tracks_data.max_track_n_poses

    @property
    def tracks_packinfo(self) -> torch.Tensor:
        return self.tracks_data.tracks_packinfo

    @property
    def tracks_poses(self) -> lt.SE3:
        return self.tracks_data.tracks_poses

    @tracks_poses.setter
    def tracks_poses(self, value: lt.SE3):
        """Allow updating tracks poses with consistent values"""
        assert (
            self.tracks_data.tracks_poses.shape == value.shape
            and self.tracks_data.tracks_poses.device == value.device
            and self.tracks_data.tracks_poses.dtype == value.dtype
        )
        self.tracks_data.tracks_poses = value

    @property
    def tracks_timestamps_us(self) -> torch.Tensor:
        return self.tracks_data.tracks_timestamps_us

    @property
    def tracks_flags(self) -> torch.Tensor:
        return self.tracks_data.tracks_flags

    class Factory:
        @staticmethod
        def from_numpy(
            tracks_id: list[str],
            tracks_poses: list[np.ndarray],
            tracks_timestamps_us: list[np.ndarray],
            tracks_flags: list[TrackFlags],
            pose_format: Literal["matrix", "tquat"] = "matrix",
            device: torch.device = torch.device("cuda"),
        ) -> Tracks:
            """
            Inputs (N_tracks: number of different tracks, N_poses_i: number of poses of track i):
            - track_ids: string identifiers of each track, N_tracks [str]
            - tracks_poses: 4x4 matrix or 7-dim tquat vector track to world pose transformations, depending on 'pose_format', N_tracks x (N_poses_i x [4 x 4 | 7]) [float]
            - tracks_timestamps_us: pose timestamps, N_tracks x (N_poses_i, ) [int64]
            - tracks_flags: per-track flags, N_tracks [TrackFlags]
            - pose_format: the format of the poses (either 4x4 'matrix'es or 7-dim 'tquat' vectors)
            - device: the device to store the data on [torch.device]
            """

            # Convert to compressed pose representation
            tracks_poses_sizes = [len(track_poses) for track_poses in tracks_poses]

            assert len(tracks_id) == len(tracks_poses_sizes), "Tracks: inconsistent track_ids / pose"
            assert len(tracks_id) == len(tracks_flags), "Tracks: inconsistent track_ids / flags"
            assert tracks_poses_sizes == [len(track_timestamps_us) for track_timestamps_us in tracks_timestamps_us], (
                "Tracks: inconsistent pose / timestamp pairs"
            )

            assert all(
                [
                    (track_poses.shape[-2:] == (4, 4) if pose_format == "matrix" else track_poses.shape[-1:] == (7,))
                    and track_poses.dtype == np.float32
                    for track_poses in tracks_poses
                ]
            ), "Tracks: invalid poses inputs"
            assert all([len(track_poses) >= 2 for track_poses in tracks_poses]), (
                "Tracks: require at least two poses per track"
            )
            assert all(
                [
                    len(track_timestamps_us.shape) == 1 and track_timestamps_us.dtype == np.int64
                    for track_timestamps_us in tracks_timestamps_us
                ]
            ), "Tracks: invalid tracks_timestamps_us inputs"

            # create packed pose / timestamp representation and upload to GPU
            # (also handle special case of empty / non-existing tracks)
            n_poses = [track_poses.shape[0] for track_poses in tracks_poses]
            start_idxs = np.cumsum([0] + n_poses, dtype=np.int32)[:-1]

            max_track_n_poses = max(n_poses) if len(n_poses) else 0

            tracks_packinfo = torch.tensor(
                np.stack((start_idxs, np.array(n_poses, dtype=np.int32)), axis=1), device=device
            )

            N_total_poses = tracks_packinfo[:, 1].sum()

            tracks_poses_array = np.empty(
                (N_total_poses, 4, 4) if pose_format == "matrix" else (N_total_poses, 7), dtype=np.float32
            )
            tracks_timestamps_array_us = np.empty((N_total_poses,), dtype=np.int64)
            if N_total_poses > 0:
                np.concatenate(tracks_poses, out=tracks_poses_array)
                np.concatenate(tracks_timestamps_us, out=tracks_timestamps_array_us)

            tracks_poses_se3: lt.SE3
            if pose_format == "matrix":
                tracks_poses_se3 = lt.SE3(se3_matrix_to_tquat(tracks_poses_array).to(device))
            else:
                tracks_poses_se3 = lt.SE3(torch.from_numpy(tracks_poses_array).to(device))

            return Tracks(
                tracks_data=TracksData(
                    tracks_id=tracks_id,
                    tracks_packinfo=tracks_packinfo,
                    tracks_poses=tracks_poses_se3,
                    tracks_timestamps_us=torch.from_numpy(tracks_timestamps_array_us).to(device),
                    tracks_flags=torch.tensor(
                        [track_flags.value for track_flags in tracks_flags], dtype=torch.int32, device=device
                    ),
                    max_track_n_poses=max_track_n_poses,
                )
            )

    @property
    def device(self) -> torch.device:
        return self.tracks_poses.device

    def to_device(self, device: torch.device) -> Self:
        return cast(Self, self.__class__(tracks_data=self.tracks_data.to_device(device)))

    @property
    def n_tracks(self) -> int:
        return len(self.tracks_id)


@dataclass(kw_only=True, slots=True)
class CuboidTracks(Tracks):
    """Manages time-dependent cuboid tracks related GPU intersection operations for tracks

    This class uses the :class:`instant_nurec.utils.types.CuboidTracksData` class for storing data.

    """

    cuboidtracks_data: CuboidTracksData

    @property
    def cuboids_dims(self) -> torch.Tensor:
        return self.cuboidtracks_data.cuboids_dims

    class Factory:
        @staticmethod
        def from_numpy(
            tracks_id: list[str],
            tracks_poses: list[np.ndarray],
            tracks_timestamps_us: list[np.ndarray],
            tracks_flags: list[TrackFlags],
            cuboids_dims: list[np.ndarray],
            device: torch.device = torch.device("cuda"),
        ) -> CuboidTracks:
            """
            Inputs (N_tracks: number of different tracks, N_poses_i: number of poses of track i):
            - track_ids: string identifiers of each track, N_tracks [str]
            - tracks_poses: 4x4 track to world pose transformations, N_tracks x (N_poses_i x 4 x 4) [float]
            - tracks_timestamps_us: pose timestamps, N_tracks x (N_poses_i, ) [int64]
            - tracks_flags: per-track flags, N_tracks [TrackFlags]
            - cuboids_dims: cuboid x/y/z extents (in local track frame), N_tracks x 3 [float]
            - device: the device to store the data on [torch.device]
            """

            # construct base tracks part
            tracks = Tracks.Factory.from_numpy(
                tracks_id, tracks_poses, tracks_timestamps_us, tracks_flags, device=device
            )

            # construct cuboid-tracks-specific part
            if len(cuboids_dims_array := np.empty((len(tracks.tracks_id), 3), dtype=np.float32)):
                np.stack(cuboids_dims, out=cuboids_dims_array)

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=tracks.tracks_id,
                    tracks_packinfo=tracks.tracks_packinfo,
                    tracks_poses=tracks.tracks_poses,
                    tracks_timestamps_us=tracks.tracks_timestamps_us,
                    tracks_flags=tracks.tracks_flags,
                    max_track_n_poses=tracks.max_track_n_poses,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=torch.tensor(cuboids_dims_array, dtype=torch.float32, device=device)
                ),
            )

        @staticmethod
        def from_pack(pack: CuboidTracksDataPack) -> CuboidTracks:
            """
            Constructs cuboid tracks from packed data components
            """

            return CuboidTracks(tracks_data=pack.tracks_data, cuboidtracks_data=pack.cuboidtracks_data)

    class Ops:
        """
        Operations on CuboidTracks returning new CuboidTracks instances.
        Compared to classmethods, this forces the semantics to be NOT inplace.
        """

        @staticmethod
        def subset_from_indices(cuboid_tracks: CuboidTracks, indices: list[int] | torch.Tensor) -> CuboidTracks:
            """
            Subsets the cuboid tracks so that the new tracks_id matches argument indices
            This operation keeps the gradient flow.
            """
            if isinstance(indices, list):
                indices_list = indices
                indices_tensor = torch.tensor(indices, device=cuboid_tracks.device, dtype=torch.long)

            else:
                indices_list = indices.cpu().numpy().tolist()
                indices_tensor = indices

            track_starts_counts = cuboid_tracks.tracks_packinfo[indices_tensor]
            track_starts, track_counts = track_starts_counts[:, 0], track_starts_counts[:, 1]
            packed_attr_ind = (
                linstep_interleave(track_starts, track_counts, 1)
                if len(track_starts)
                else torch.empty(0, dtype=torch.long, device=cuboid_tracks.device)
            )
            new_tracks_packinfo = get_pack_info_from_n(track_counts)

            return CuboidTracks(
                # base tracks part
                tracks_data=TracksData(
                    tracks_id=[cuboid_tracks.tracks_id[i] for i in indices_list],
                    tracks_packinfo=new_tracks_packinfo,
                    tracks_poses=cuboid_tracks.tracks_poses[packed_attr_ind],
                    tracks_timestamps_us=cuboid_tracks.tracks_timestamps_us[packed_attr_ind],
                    tracks_flags=cuboid_tracks.tracks_flags[indices_tensor],
                    max_track_n_poses=cast(int, track_counts.max().item()) if len(track_counts) else 0,
                ),
                # cuboid-tracks-specific part
                cuboidtracks_data=CuboidTracksData(
                    cuboids_dims=cuboid_tracks.cuboids_dims[indices_tensor],
                ),
            )

        @staticmethod
        def subset_from_mask(cuboid_tracks: CuboidTracks, mask: torch.Tensor) -> CuboidTracks:
            """
            Subsets the cuboid tracks so that the new tracks_id matches the mask
            This operation keeps the gradient flow.
            """
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            return CuboidTracks.Ops.subset_from_indices(cuboid_tracks, indices)

    def to_device(self, device):
        return CuboidTracks(
            tracks_data=self.tracks_data.to_device(device),
            cuboidtracks_data=self.cuboidtracks_data.to_device(device),
        )

    @dataclass
    class RayIntersectionResult:
        intersections_cnt: torch.Tensor  # number of intersections of each ray, N_rays [int]
        intersections_tracks_idx: (
            torch.Tensor
        )  # for each intersection, the index of the intersected track, N_rays x max_intersections_per_ray [int]

    def ray_intersection(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_timestamps_us: torch.Tensor,
        max_intersections_per_ray: int = 32,
    ) -> CuboidTracks.RayIntersectionResult:
        """
        Computes the intersection of all cuboid tracks with timed world rays

        Inputs:
        - rays_o: ray origins / 3d world positions, N_rays x 3 [float]
        - rays_d: normalized 3d world directions, N_rays x 3 [float]
        - rays_timestamps_us: per ray timestamp, N_rays [int64]
        - max_intersections_per_ray: upper limit of intersections to return [int]

        Returns:
        - RayIntersectionResult: result of the ray intersection
        """
        intersection_result = _ray_cuboidtracks_intersection(
            rays_o,
            rays_d,
            rays_timestamps_us,
            self.tracks_packinfo,
            self.tracks_poses.data,
            self.tracks_timestamps_us,
            self.cuboids_dims,
            self.max_track_n_poses,
            max_intersections_per_ray,
            False,  # with_intersections_ts: predict path never reads .intersections_ts
        )

        return CuboidTracks.RayIntersectionResult(
            intersections_cnt=intersection_result[0],
            intersections_tracks_idx=intersection_result[1],
        )

    def point_intersection_interpolate_pose(
        self,
        points: torch.Tensor,
        points_timestamps_us: torch.Tensor,
        cuboids_dims_padding: torch.Tensor,
    ) -> tuple[lt.SE3, torch.Tensor]:
        """
        For each point, returns the interpolated pose of the tracks that it is inside, as well as the track idx.

        Inputs:
        - points: 3D points to check for inside check, [..., 3] [float]
        - points_timestamps_us: per point timestamp, [...] [int64]
        - cuboids_dims_padding: 3d padding to add to cuboids, broadcastable to N_tracks x 3 [float]

        Returns:
        - interpolated_poses: for each point, the pose of the cuboid it is inside, [...] [SE3]
        - interpolated_tracks_idx: for each point, the index of the intersected track (-1 if no intersection), [...] [int]
        """

        data_shape = points.shape[:-1]
        points = points.reshape(-1, 3).contiguous()
        points_timestamps_us = points_timestamps_us.reshape(-1).contiguous()

        interpolated_tracks_pose_data, interpolated_tracks_idx = (
            _point_cuboidtracks_intersection_interpolate_pose(
                points,
                points_timestamps_us,
                self.tracks_packinfo,
                self.tracks_poses.data,
                self.tracks_timestamps_us,
                self.cuboids_dims + cuboids_dims_padding,
                self.max_track_n_poses,
            )
        )
        interpolated_poses = lt.SE3(
            interpolated_tracks_pose_data.reshape(data_shape + (interpolated_tracks_pose_data.shape[-1],))
        )
        return interpolated_poses, interpolated_tracks_idx.reshape(data_shape)

    def interpolate_tracks_poses(
        self,
        timestamps_us: torch.Tensor,
        tracks_idx: torch.Tensor,
    ) -> lt.SE3:
        """Compute the interpolated pose of the tracks at the given timestamps.
        This will not check if the timestamps are within the range of the tracks!

        Inputs:
        - timestamps_us: timestamps to interpolate the pose at, N_data [int64]
        - tracks_idx: indices of the tracks to interpolate the pose for, N_data [int]
        """
        tidx_right = _packed_searchsorted_indexed_vals(
            self.tracks_timestamps_us,
            self.tracks_packinfo,
            timestamps_us,
            tracks_idx,
        )
        # Make sure right index is locally [1, N] indices
        # (usually this won't happen since we already filter in the filtering stage)
        packinfo_start = self.tracks_packinfo[tracks_idx, 0]
        packinfo_end = packinfo_start + self.tracks_packinfo[tracks_idx, 1]
        tidx_right[tidx_right == packinfo_start] += 1
        tidx_right[tidx_right == packinfo_end] -= 1

        tidx_left = tidx_right - 1
        time_diff = (self.tracks_timestamps_us[tidx_right] - self.tracks_timestamps_us[tidx_left]).to(torch.float32)
        alpha = (timestamps_us - self.tracks_timestamps_us[tidx_left]).to(torch.float32) / time_diff
        interpolated_mask = torch.logical_and(time_diff != 0, torch.logical_and(alpha >= 0, alpha <= 1))
        alpha = torch.where(interpolated_mask, alpha, torch.zeros_like(alpha))

        pose_start = self.tracks_poses[tidx_left]
        pose_end = self.tracks_poses[tidx_right]

        # Perform manifold-product interpolation (to be consistent with transform-filter)
        R_start = lt.SO3.InitFromVec(pose_start.vec()[:, 3:])
        R_end = lt.SO3.InitFromVec(pose_end.vec()[:, 3:])
        t_start, t_end = pose_start.translation(), pose_end.translation()

        R_alpha = R_start * lt.SO3.exp(alpha[:, None] * (R_start.inv() * R_end).log())
        t_alpha = t_start + alpha[:, None] * (t_end - t_start)

        return lt.SE3.InitFromVec(torch.cat([t_alpha[:, :3], R_alpha.vec()], dim=1))


