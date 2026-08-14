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

from instant_nurec.utils.misc import unpack_optional


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


def associate_points_with_cuboid_tracks(
    points: torch.Tensor,
    points_timestamps_us: torch.Tensor,
    points_dynamic_mask: torch.Tensor | None,
    cuboid_tracks: CuboidTracks,
    cuboids_dims_padding: torch.Tensor | None,
    second_pass_cuboids_dims_padding_scale: float = 6.0,
    second_pass_chunk_size: int = 65536,
) -> torch.Tensor:
    """Associate world points with cuboid tracks at their timestamps.

    The first pass uses the configured padding. Movable points that miss that
    pass are checked against wider cuboids and assigned to the hit nearest its
    original, unpadded bounding box.

    All per-point scalar tensors use a trailing singleton dimension.
    """
    data_shape = points.shape[:-1]
    expected_scalar_shape = data_shape + (1,)
    assert points_timestamps_us.shape == expected_scalar_shape
    if points_dynamic_mask is not None:
        assert points_dynamic_mask.shape == expected_scalar_shape

    if cuboid_tracks.n_tracks == 0:
        return torch.full(
            expected_scalar_shape,
            -1,
            device=points.device,
            dtype=cuboid_tracks.tracks_packinfo.dtype,
        )

    points_ts = points_timestamps_us.squeeze(-1)
    first_pass = cuboid_tracks.point_intersection_interpolate_pose(
        points,
        points_ts,
        cuboids_dims_padding,
        max_intersections_per_point=1,
    )
    tracks_idx = first_pass.intersections_tracks_idx

    if points_dynamic_mask is None:
        return tracks_idx

    second_pass_mask = (points_dynamic_mask & (tracks_idx == -1)).squeeze(-1)
    n_second_pass = int(second_pass_mask.sum().item())
    if n_second_pass == 0:
        return tracks_idx

    if cuboids_dims_padding is None:
        wide_cuboids_dims_padding = torch.full(
            (3,),
            second_pass_cuboids_dims_padding_scale,
            device=points.device,
            dtype=points.dtype,
        )
    else:
        wide_cuboids_dims_padding = (
            cuboids_dims_padding.to(device=points.device, dtype=points.dtype) * second_pass_cuboids_dims_padding_scale
        )

    cuboids_dims = cuboid_tracks.cuboids_dims.to(device=points.device, dtype=points.dtype)
    second_pass_points = points[second_pass_mask]
    second_pass_timestamps_us = points_ts[second_pass_mask]
    best_tracks_idx = torch.full((n_second_pass, 1), -1, device=points.device, dtype=tracks_idx.dtype)
    has_intersection = torch.zeros((n_second_pass, 1), device=points.device, dtype=torch.bool)
    chunk_size = max(1, second_pass_chunk_size)
    for start in range(0, n_second_pass, chunk_size):
        end = min(start + chunk_size, n_second_pass)
        second_pass = cuboid_tracks.point_intersection_interpolate_pose(
            points=second_pass_points[start:end],
            points_timestamps_us=second_pass_timestamps_us[start:end],
            cuboids_dims_padding=wide_cuboids_dims_padding,
            max_intersections_per_point=64,
            with_local_points=True,
        )
        wide_tracks_idx = second_pass.intersections_tracks_idx
        wide_points_local = unpack_optional(second_pass.intersections_points_local)
        wide_valid_mask = wide_tracks_idx != -1
        if not torch.any(wide_valid_mask):
            continue

        matched_dims = cuboids_dims[wide_tracks_idx.clamp_min(0).to(torch.long)]
        distance_to_bbox = torch.linalg.norm(
            (wide_points_local.abs() - matched_dims * 0.5).clamp_min(0.0),
            dim=-1,
        )
        distance_to_bbox = torch.where(
            wide_valid_mask,
            distance_to_bbox,
            torch.full_like(distance_to_bbox, torch.inf),
        )
        best_hit_idx = torch.argmin(distance_to_bbox, dim=-1, keepdim=True)
        best_tracks_idx[start:end] = torch.gather(wide_tracks_idx, dim=-1, index=best_hit_idx)
        has_intersection[start:end] = wide_valid_mask.any(dim=-1, keepdim=True)

    tracks_idx[second_pass_mask] = torch.where(
        has_intersection,
        best_tracks_idx,
        tracks_idx[second_pass_mask],
    )
    return tracks_idx


def warp_points_with_cuboid_tracks(
    points: torch.Tensor,
    source_timestamps_us: torch.Tensor,
    target_timestamps_us_list: list[torch.Tensor],
    cuboid_tracks: CuboidTracks,
    tracks_idx: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Warp track-associated points to each target timestamp.

    All per-point scalar tensors use a trailing singleton dimension. Track
    indices are expected to come from :func:`associate_points_with_cuboid_tracks`.
    """
    data_shape = points.shape[:-1]
    expected_scalar_shape = data_shape + (1,)
    assert source_timestamps_us.shape == expected_scalar_shape
    for target_timestamps_us in target_timestamps_us_list:
        assert target_timestamps_us.shape == expected_scalar_shape
    assert tracks_idx.shape == expected_scalar_shape

    source_timestamps_us = source_timestamps_us.squeeze(-1)
    target_timestamps_us_list = [timestamps.squeeze(-1) for timestamps in target_timestamps_us_list]
    tracks_idx = tracks_idx.squeeze(-1)

    dynamic_mask = tracks_idx != -1
    sel = torch.where(dynamic_mask)
    sel_tracks_idx = tracks_idx[sel].to(dtype=cuboid_tracks.tracks_packinfo.dtype)

    warped_points_list: list[torch.Tensor]

    if sel_tracks_idx.numel() == 0:
        warped_points_list = [points.clone() for _ in target_timestamps_us_list]
        return dynamic_mask.unsqueeze(-1), warped_points_list

    inv_current_pose = cuboid_tracks.interpolate_tracks_poses(
        timestamps_us=source_timestamps_us[sel],
        tracks_idx=sel_tracks_idx,
    ).inv()

    warped_points_list = []
    for target_timestamps_us in target_timestamps_us_list:
        target_pose = cuboid_tracks.interpolate_tracks_poses(
            timestamps_us=target_timestamps_us[sel],
            tracks_idx=sel_tracks_idx,
        )
        new_points = points.clone()
        new_points[sel] = (target_pose * inv_current_pose) * new_points[sel]  # type: ignore[operator]
        warped_points_list.append(new_points)

    return dynamic_mask.unsqueeze(-1), warped_points_list
