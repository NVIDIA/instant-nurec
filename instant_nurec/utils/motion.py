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
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from instant_nurec.datasets.tracks import CuboidTracks


@dataclass(kw_only=True, slots=True)
class TimeRemapping:
    """
    Map from start_timestamp_us to end_timestamp_us to 0-1.
    """

    start_timestamp_us: int
    end_timestamp_us: int
    frame_gap_timestamps_us: torch.Tensor

    @classmethod
    def from_timestamps_startend_us(
        cls, timestamps_startend_us_cpu: torch.Tensor, frames_camera_idxs: torch.Tensor
    ) -> TimeRemapping:
        assert timestamps_startend_us_cpu.shape[1] == 2, "Timestamps must be (V, 2)"
        return cls(
            # Already a CPU copy; use directly
            start_timestamp_us=int(timestamps_startend_us_cpu[:, 0].min().item()),
            end_timestamp_us=int(timestamps_startend_us_cpu[:, 1].max().item()),
            frame_gap_timestamps_us=cls._compute_frame_gap(timestamps_startend_us_cpu, frames_camera_idxs),
        )

    @staticmethod
    def _compute_frame_gap(frames_timestamps_us: torch.Tensor, frames_camera_idxs: torch.Tensor) -> torch.Tensor:
        """
        Compute the timestamp gap from each frame to its nearest prev/next neighbor (at the same camera).
        Input:
            frames_timestamps_us: (V, 2) of start/end timestamps
            frames_camera_idxs: (V,)
        Output:
            gap_timestamps_us: (V, 2). If either prev/next is missing, it's set to the existing one.
        will set to 0.5s if neither exists.
        """
        # Compute median timestamp for each frame
        frames_timestamps_us = (
            frames_timestamps_us[:, 0] + (frames_timestamps_us[:, 1] - frames_timestamps_us[:, 0]) // 2
        )

        prev_gap_timestamps_us = torch.zeros_like(frames_timestamps_us)
        next_gap_timestamps_us = torch.zeros_like(frames_timestamps_us)

        # Process for each camera
        num_cameras: int = frames_camera_idxs.unique().shape[0]
        for camera_idx in range(num_cameras):
            camera_mask = torch.where(frames_camera_idxs == camera_idx)[0]
            sorted_time, sorted_idx = torch.sort(frames_timestamps_us[camera_mask])
            sorted_time_gap = sorted_time[1:] - sorted_time[:-1]
            # For prev we miss the first one, for next we miss the last one
            prev_gap_timestamps_us[camera_mask[sorted_idx[1:]]] = sorted_time_gap
            next_gap_timestamps_us[camera_mask[sorted_idx[:-1]]] = sorted_time_gap

        # Fill in missing values
        prev_gap_timestamps_us = torch.where(
            prev_gap_timestamps_us == 0, next_gap_timestamps_us, prev_gap_timestamps_us
        )
        next_gap_timestamps_us = torch.where(
            next_gap_timestamps_us == 0, prev_gap_timestamps_us, next_gap_timestamps_us
        )
        gap_timestamps_us = torch.stack([prev_gap_timestamps_us, next_gap_timestamps_us], dim=-1)
        gap_timestamps_us[gap_timestamps_us == 0] = 500000

        return gap_timestamps_us

    def timestamps_us_to_continuous_times(self, timestamps_us: torch.Tensor) -> torch.Tensor:
        span = self.end_timestamp_us - self.start_timestamp_us
        if span == 0:
            return torch.zeros_like(timestamps_us, dtype=torch.float32)
        return (timestamps_us - self.start_timestamp_us) / span


def warp_points_with_cuboid_tracks(
    points: torch.Tensor,
    source_timestamps_us: torch.Tensor,
    target_timestamps_us_list: list[torch.Tensor],
    dynamic_tracks: CuboidTracks,
    aux_tracks_idx: torch.Tensor,
    cuboids_dims_padding: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Associate each point with a dynamic cuboid track at its source timestamp, then warp
    it to each given target timestamp using that track's interpolated pose.

    Association uses point-cuboid intersection. For points that fail this association,
    `aux_tracks_idx` (e.g. from a ray-cuboid intersection on the originating ray) is used
    as a fallback.

    Args:
        points: [..., 3] world points at the source timestamps.
        source_timestamps_us: [...] (or [..., 1]) source timestamps per point.
        target_timestamps_us_list: list of target timestamp tensors, each [...] (or [..., 1]).
        dynamic_tracks: dynamic CuboidTracks to associate against.
        aux_tracks_idx: [...] fallback track ids for points whose point-cuboid
            intersection returns -1; pass -1 for "no fallback" entries.
        cuboids_dims_padding: 3D padding broadcastable to N_tracks x 3.

    Returns:
        - dynamic_mask: [...] bool, True for points associated with a track.
        - warped_points_list: per target, a [..., 3] tensor where associated points are
          warped to the corresponding target timestamp and unassociated points are unchanged.
    """
    data_shape = points.shape[:-1]

    def _squeeze_trailing_one(t: torch.Tensor) -> torch.Tensor:
        # Accept either [...] or [..., 1] timestamp shapes for caller convenience.
        if t.ndim == len(data_shape) + 1 and t.shape[-1] == 1:
            return t.squeeze(-1)
        return t

    src_ts = _squeeze_trailing_one(source_timestamps_us)
    tgt_ts_list = [_squeeze_trailing_one(t) for t in target_timestamps_us_list]

    # Main association: point inside cuboid at source timestamp.
    _, tracks_idx = dynamic_tracks.point_intersection_interpolate_pose(points, src_ts, cuboids_dims_padding)  # [...]

    # Fallback association via aux idx (e.g. ray-cuboid for movable rays).
    unassoc = tracks_idx == -1
    tracks_idx[unassoc] = aux_tracks_idx[unassoc]

    dynamic_mask = tracks_idx != -1
    sel = torch.where(dynamic_mask)
    sel_tracks_idx = tracks_idx[sel]

    warped_points_list: list[torch.Tensor]

    # Shortcut if no dynamic points are found.
    if sel_tracks_idx.numel() == 0:
        warped_points_list = [points.clone() for _ in tgt_ts_list]
        return dynamic_mask, warped_points_list

    inv_current_pose = dynamic_tracks.interpolate_tracks_poses(
        timestamps_us=src_ts[sel],
        tracks_idx=sel_tracks_idx,
    ).inv()

    warped_points_list = []
    for tgt_ts in tgt_ts_list:
        target_pose = dynamic_tracks.interpolate_tracks_poses(
            timestamps_us=tgt_ts[sel],
            tracks_idx=sel_tracks_idx,
        )
        new_points = points.clone()
        new_points[sel] = (target_pose * inv_current_pose) * new_points[sel]  # type: ignore[operator]
        warped_points_list.append(new_points)

    return dynamic_mask, warped_points_list
