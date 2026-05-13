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

"""Ray/point-cuboid intersection helpers for ``CuboidTracks``.

* ``ray_cuboidtracks_intersection`` (used by
  ``CuboidTracks.ray_intersection``) — per-(ray, track) ray-AABB slab
  intersection in the cuboid's local frame at the ray's timestamp.
* ``point_cuboidtracks_intersection_interpolate_pose`` (used by
  ``CuboidTracks.point_intersection_interpolate_pose``) — per-(point, track)
  inside-cuboid test, returning the interpolated pose at the point's
  timestamp.
"""

from __future__ import annotations

import torch

from instant_nurec.utils.se3 import quat_xyzw_slerp, quat_xyzw_to_rotmat


def _slab_aabb_intersection(
    rays_o: torch.Tensor,  # (N, 3)
    rays_d: torch.Tensor,  # (N, 3)
    half_dims: torch.Tensor,  # (N, 3) per-ray half extents
) -> torch.Tensor:
    """Per-ray slab method ray-AABB intersection in the cuboid's local frame.

    Returns ``(N, 2)`` ``(t_near, t_far)``. On a miss (``t_near > t_far``)
    returns ``(-1, -1)``; the caller's ``t_far > 0`` check then correctly
    classifies misses as non-hits regardless of the unfiltered intermediate
    values.
    """
    inv_d = 1.0 / torch.where(rays_d.abs() > 0, rays_d, rays_d.new_tensor(1e-30))
    t_lo = (-half_dims - rays_o) * inv_d
    t_hi = (half_dims - rays_o) * inv_d
    t1 = torch.minimum(t_lo, t_hi)
    t2 = torch.maximum(t_lo, t_hi)
    t_near = t1.amax(dim=-1)
    t_far = t2.amin(dim=-1)
    miss = t_near > t_far
    neg = t_near.new_full(t_near.shape, -1.0)
    t_near = torch.where(miss, neg, t_near)
    t_far = torch.where(miss, neg, t_far)
    return torch.stack([t_near, t_far], dim=-1)


def ray_cuboidtracks_intersection(
    rays_o: torch.Tensor,                # (N_rays, 3)
    rays_d: torch.Tensor,                # (N_rays, 3)
    rays_timestamps_us: torch.Tensor,    # (N_rays,) int64
    tracks_packinfo: torch.Tensor,       # (N_tracks, 2) int32 [start, count]
    tracks_poses: torch.Tensor,          # (N_total_poses, 7) [tx ty tz qx qy qz qw]
    tracks_timestamps_us: torch.Tensor,  # (N_total_poses,) int64
    cuboids_dims: torch.Tensor,          # (N_tracks, 3) cuboid xyz extents
    max_track_n_poses: int,
    max_intersections_per_ray: int,
    with_intersections_ts: bool,
):
    """Per (ray, track) ray-AABB slab intersection at the ray's timestamp.

    Returns the kernel's tuple shape:
      * intersections_cnt: (N_rays,) int32
      * intersections_tracks_idx: (N_rays, max_intersections_per_ray) int32, padded with -1
      * intersections_ts (only if with_intersections_ts=True): (N_rays, max_intersections_per_ray, 2)
    """
    n_rays = rays_o.shape[0]
    n_tracks = tracks_packinfo.shape[0]
    device = rays_o.device
    dtype = rays_o.dtype

    intersections_cnt = torch.zeros(n_rays, dtype=torch.int32, device=device)
    intersections_tracks_idx = torch.full(
        (n_rays, max_intersections_per_ray), -1, dtype=torch.int32, device=device
    )
    intersections_ts = (
        torch.full(
            (n_rays, max_intersections_per_ray, 2), -1.0, dtype=dtype, device=device
        )
        if with_intersections_ts
        else None
    )

    if n_rays == 0 or max_track_n_poses == 0 or n_tracks == 0:
        if with_intersections_ts:
            return intersections_cnt, intersections_tracks_idx, intersections_ts
        return intersections_cnt, intersections_tracks_idx

    # Filter tracks with N_track_poses > 1 (otherwise the kernel skips).
    track_lengths = tracks_packinfo[:, 1].to(torch.int64)
    valid_tracks = track_lengths > 1
    if not bool(valid_tracks.any()):
        if with_intersections_ts:
            return intersections_cnt, intersections_tracks_idx, intersections_ts
        return intersections_cnt, intersections_tracks_idx

    # Iterate over tracks (typically O(10-100) tracks; per-track cost is
    # vectorised across all rays). Matches the kernel's grid layout where
    # each (track, ray) thread processes one pair.
    for track_id in range(n_tracks):
        if not bool(valid_tracks[track_id]):
            continue
        start = int(tracks_packinfo[track_id, 0].item())
        n_poses = int(tracks_packinfo[track_id, 1].item())
        track_ts = tracks_timestamps_us[start : start + n_poses]
        track_poses_slice = tracks_poses[start : start + n_poses]

        # Per-ray time-range gate.
        in_range = (rays_timestamps_us >= track_ts[0]) & (
            rays_timestamps_us <= track_ts[-1]
        )
        ray_idxs = in_range.nonzero(as_tuple=False).squeeze(-1)
        if ray_idxs.numel() == 0:
            continue

        # binary_search_interp: smallest j in [1, n_poses-1] with
        # track_ts[j] >= rays_timestamps_us[i] (effectively right-end of
        # interpolation interval). Use lower-bound and clamp into [1, n_poses-1].
        ts_q = rays_timestamps_us[ray_idxs]
        end_idx_local = torch.searchsorted(track_ts, ts_q).clamp(min=1, max=n_poses - 1)
        start_idx_local = end_idx_local - 1

        ts_start = track_ts[start_idx_local].to(dtype)
        ts_end = track_ts[end_idx_local].to(dtype)
        t_interp = (ts_q.to(dtype) - ts_start) / (ts_end - ts_start)

        pose_start = track_poses_slice[start_idx_local]
        pose_end = track_poses_slice[end_idx_local]
        c_start = pose_start[:, :3]
        c_end = pose_end[:, :3]
        q_start = pose_start[:, 3:]
        q_end = pose_end[:, 3:]

        c_interp = (1.0 - t_interp).unsqueeze(-1) * c_start + t_interp.unsqueeze(-1) * c_end
        q_interp = quat_xyzw_slerp(q_start, q_end, t_interp)

        # Local-frame ray.
        # The kernel does R^T (transpose) — world→local rotation.
        R_world_to_local = quat_xyzw_to_rotmat(q_interp).transpose(-1, -2)
        rays_o_local = torch.bmm(
            R_world_to_local, (rays_o[ray_idxs] - c_interp).unsqueeze(-1)
        ).squeeze(-1)
        rays_d_local = torch.bmm(R_world_to_local, rays_d[ray_idxs].unsqueeze(-1)).squeeze(
            -1
        )

        half_dims = (cuboids_dims[track_id] * 0.5).expand_as(rays_o_local)
        t_pair = _slab_aabb_intersection(rays_o_local, rays_d_local, half_dims)
        t_near = t_pair[:, 0]
        t_far = t_pair[:, 1]
        # Kernel records intersection iff t_far > 0 (regardless of t_near).
        hit_mask = t_far > 0
        hit_local = hit_mask.nonzero(as_tuple=False).squeeze(-1)
        if hit_local.numel() == 0:
            continue
        hit_rays = ray_idxs[hit_local]

        # Per-ray atomicAdd in the kernel: increment cnt, write track_idx if
        # cnt < max. Translate to torch by sequentially appending.
        for i in range(hit_rays.shape[0]):
            r_idx = int(hit_rays[i].item())
            cnt_now = int(intersections_cnt[r_idx].item())
            intersections_cnt[r_idx] = cnt_now + 1
            if cnt_now < max_intersections_per_ray:
                intersections_tracks_idx[r_idx, cnt_now] = track_id
                if with_intersections_ts:
                    intersections_ts[r_idx, cnt_now, 0] = max(
                        float(t_near[hit_local[i]].item()), 0.0
                    )
                    intersections_ts[r_idx, cnt_now, 1] = float(t_far[hit_local[i]].item())

    if with_intersections_ts:
        return intersections_cnt, intersections_tracks_idx, intersections_ts
    return intersections_cnt, intersections_tracks_idx


def point_cuboidtracks_intersection_interpolate_pose(
    points: torch.Tensor,                # (N_points, 3)
    points_timestamps_us: torch.Tensor,  # (N_points,) int64
    tracks_packinfo: torch.Tensor,       # (N_tracks, 2) int32 [start, count]
    tracks_poses: torch.Tensor,          # (N_total_poses, 7)
    tracks_timestamps_us: torch.Tensor,  # (N_total_poses,) int64
    cuboids_dims: torch.Tensor,          # (N_tracks, 3)
    max_track_n_poses: int,
):
    """Per (point, track) inside-cuboid test with interpolated pose.

    Returns:
      interpolated_tracks_pose: (N_points, 7)
      interpolated_tracks_idx: (N_points,) int32, -1 if no intersection.
    """
    n_points = points.shape[0]
    n_tracks = tracks_packinfo.shape[0]
    device = points.device
    dtype = points.dtype

    interpolated_tracks_pose = torch.zeros((n_points, 7), dtype=dtype, device=device)
    interpolated_tracks_idx = torch.full(
        (n_points,), -1, dtype=torch.int32, device=device
    )

    if n_points == 0 or max_track_n_poses == 0 or n_tracks == 0:
        return interpolated_tracks_pose, interpolated_tracks_idx

    track_lengths = tracks_packinfo[:, 1].to(torch.int64)

    for track_id in range(n_tracks):
        n_poses = int(track_lengths[track_id].item())
        if n_poses <= 1:
            continue
        start = int(tracks_packinfo[track_id, 0].item())
        track_ts = tracks_timestamps_us[start : start + n_poses]
        track_poses_slice = tracks_poses[start : start + n_poses]

        in_range = (points_timestamps_us >= track_ts[0]) & (
            points_timestamps_us <= track_ts[-1]
        )
        pt_idxs = in_range.nonzero(as_tuple=False).squeeze(-1)
        if pt_idxs.numel() == 0:
            continue

        ts_q = points_timestamps_us[pt_idxs]
        end_idx_local = torch.searchsorted(track_ts, ts_q).clamp(min=1, max=n_poses - 1)
        start_idx_local = end_idx_local - 1
        ts_start = track_ts[start_idx_local].to(dtype)
        ts_end = track_ts[end_idx_local].to(dtype)
        t_interp = (ts_q.to(dtype) - ts_start) / (ts_end - ts_start)

        pose_start = track_poses_slice[start_idx_local]
        pose_end = track_poses_slice[end_idx_local]
        c_start = pose_start[:, :3]
        c_end = pose_end[:, :3]
        q_start = pose_start[:, 3:]
        q_end = pose_end[:, 3:]
        c_pointtime = (1.0 - t_interp).unsqueeze(-1) * c_start + t_interp.unsqueeze(-1) * c_end
        q_pointtime = quat_xyzw_slerp(q_start, q_end, t_interp)

        R_world_to_local = quat_xyzw_to_rotmat(q_pointtime).transpose(-1, -2)
        point_local = torch.bmm(
            R_world_to_local, (points[pt_idxs] - c_pointtime).unsqueeze(-1)
        ).squeeze(-1)

        half_dim = cuboids_dims[track_id] * 0.5
        # Strict-inequality inside-cuboid check (matches the CUDA kernel:
        # ``point_bbox > -dim/2 && point_bbox < dim/2``).
        inside = (
            (point_local > -half_dim).all(dim=-1) & (point_local < half_dim).all(dim=-1)
        )
        hit_local = inside.nonzero(as_tuple=False).squeeze(-1)
        if hit_local.numel() == 0:
            continue
        hit_pts = pt_idxs[hit_local]

        # Last-write-wins: if multiple tracks hit the same point, the
        # CUDA kernel races (atomicAdd-like behaviour); empirically the
        # standalone uses a track-disjoint dataset, so we just overwrite.
        interpolated_tracks_pose[hit_pts, :3] = c_pointtime[hit_local]
        interpolated_tracks_pose[hit_pts, 3:] = q_pointtime[hit_local]
        interpolated_tracks_idx[hit_pts] = track_id

    return interpolated_tracks_pose, interpolated_tracks_idx
