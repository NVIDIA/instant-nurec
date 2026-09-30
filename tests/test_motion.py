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

"""Branch-coverage tests for instant_nurec.utils.motion.TimeRemapping.

The module is pure-torch + pure-python (the ``CuboidTracks`` import is
``TYPE_CHECKING``-only), so no stubs are needed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.utils.motion import TimeRemapping


# ---------------------------------------------------------------------------
# from_timestamps_startend_us
# ---------------------------------------------------------------------------


def test_from_timestamps_startend_us_basic_single_camera():
    ts = torch.tensor([[0, 100], [200, 300], [400, 500]])
    cam = torch.tensor([0, 0, 0])
    tr = TimeRemapping.from_timestamps_startend_us(ts, cam)
    assert tr.start_timestamp_us == 0
    assert tr.end_timestamp_us == 500
    assert tr.frame_gap_timestamps_us.shape == (3, 2)


def test_from_timestamps_startend_us_rejects_wrong_shape():
    """The classmethod asserts the trailing dim is exactly 2."""
    ts = torch.tensor([[0, 100, 200]])  # (V, 3) — wrong
    cam = torch.tensor([0])
    with pytest.raises(AssertionError):
        TimeRemapping.from_timestamps_startend_us(ts, cam)


# ---------------------------------------------------------------------------
# _compute_frame_gap
# ---------------------------------------------------------------------------


def test_compute_frame_gap_single_camera_three_frames():
    """Three evenly-spaced frames at one camera: prev/next gaps are 200us
    everywhere (the first frame's missing prev is backfilled from next, and
    vice versa for the last frame)."""
    ts = torch.tensor([[0, 100], [200, 300], [400, 500]])
    cam = torch.tensor([0, 0, 0])
    gap = TimeRemapping._compute_frame_gap(ts, cam)
    # all entries should be 200us (median spacing)
    assert torch.equal(gap, torch.full_like(gap, 200))


def test_compute_frame_gap_two_cameras_independent():
    """Two cameras, each with two frames. Each camera's frames should pair
    up among themselves — gaps are not crossed between cameras."""
    ts = torch.tensor(
        [
            [0, 0],  # cam 0
            [1000, 1000],  # cam 0
            [50, 50],  # cam 1
            [9999, 9999],  # cam 1
        ]
    )
    cam = torch.tensor([0, 0, 1, 1])
    gap = TimeRemapping._compute_frame_gap(ts, cam)
    # cam 0 gap = 1000us
    assert gap[0, 0].item() == 1000 and gap[0, 1].item() == 1000
    assert gap[1, 0].item() == 1000 and gap[1, 1].item() == 1000
    # cam 1 gap = 9949us
    assert gap[2, 0].item() == 9949 and gap[2, 1].item() == 9949
    assert gap[3, 0].item() == 9949 and gap[3, 1].item() == 9949


def test_compute_frame_gap_single_frame_per_camera_falls_back_to_500000():
    """When a camera has only one frame, both prev and next are missing,
    triggering the 500000us fallback (0.5s default per the docstring)."""
    ts = torch.tensor([[0, 0], [1000, 1000]])
    cam = torch.tensor([0, 1])  # one frame per camera
    gap = TimeRemapping._compute_frame_gap(ts, cam)
    # both entries should be the 500000us fallback
    assert torch.equal(gap, torch.full_like(gap, 500000))


def test_compute_frame_gap_first_frame_backfilled_from_next():
    """First frame of a camera has no prev — its prev gap is filled from
    its next gap."""
    ts = torch.tensor([[0, 0], [100, 100], [500, 500]])  # asymmetric spacing
    cam = torch.tensor([0, 0, 0])
    gap = TimeRemapping._compute_frame_gap(ts, cam)
    # frame 0 (sorted-first): no prev, gets backfilled from its next gap (100us)
    assert gap[0, 0].item() == 100  # prev (backfilled)
    assert gap[0, 1].item() == 100  # next
    # frame 1 (middle): prev = 100us (from f0), next = 400us (to f2)
    assert gap[1, 0].item() == 100
    assert gap[1, 1].item() == 400
    # frame 2 (sorted-last): no next, gets backfilled from its prev (400us)
    assert gap[2, 0].item() == 400
    assert gap[2, 1].item() == 400


# ---------------------------------------------------------------------------
# timestamps_us_to_continuous_times
# ---------------------------------------------------------------------------


def test_timestamps_us_to_continuous_times_linear_map():
    tr = TimeRemapping(
        start_timestamp_us=0,
        end_timestamp_us=1000,
        frame_gap_timestamps_us=torch.empty(0, 2),
    )
    out = tr.timestamps_us_to_continuous_times(torch.tensor([0.0, 500.0, 1000.0]))
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]))


def test_timestamps_us_to_continuous_times_zero_span_returns_zeros():
    """The span==0 branch must return zeros (not divide by zero)."""
    tr = TimeRemapping(
        start_timestamp_us=42,
        end_timestamp_us=42,  # zero span
        frame_gap_timestamps_us=torch.empty(0, 2),
    )
    out = tr.timestamps_us_to_continuous_times(torch.tensor([42.0, 42.0]))
    assert torch.equal(out, torch.zeros(2))
    assert out.dtype == torch.float32


def test_timestamps_us_to_continuous_times_outside_range_extrapolates():
    """Inputs outside [start, end) extrapolate linearly — the function
    does not clamp."""
    tr = TimeRemapping(
        start_timestamp_us=0,
        end_timestamp_us=100,
        frame_gap_timestamps_us=torch.empty(0, 2),
    )
    out = tr.timestamps_us_to_continuous_times(torch.tensor([-50.0, 150.0]))
    assert torch.allclose(out, torch.tensor([-0.5, 1.5]))


# ---------------------------------------------------------------------------
# Cuboid association and warping
# ---------------------------------------------------------------------------


class _IntersectionResult:
    def __init__(self, tracks_idx: torch.Tensor, points_local: torch.Tensor | None = None):
        self.intersections_tracks_idx = tracks_idx
        self.intersections_points_local = points_local


class _FakeAssociationTracks:
    def __init__(self, first_pass_tracks_idx: torch.Tensor):
        self.n_tracks = 2
        self.tracks_packinfo = torch.tensor([[0, 2], [2, 2]], dtype=torch.int32)
        self.cuboids_dims = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        self.first_pass_tracks_idx = first_pass_tracks_idx
        self.calls: list[dict] = []

    def point_intersection_interpolate_pose(self, points, points_timestamps_us, cuboids_dims_padding, **kwargs):
        self.calls.append(
            {
                "points": points.clone(),
                "timestamps": points_timestamps_us.clone(),
                "padding": None if cuboids_dims_padding is None else cuboids_dims_padding.clone(),
                **kwargs,
            }
        )
        if kwargs["max_intersections_per_point"] == 1:
            return _IntersectionResult(self.first_pass_tracks_idx.clone())

        tracks_idx = torch.full((len(points), 64), -1, dtype=torch.int32)
        points_local = torch.zeros(len(points), 64, 3)
        for row, point in enumerate(points):
            tracks_idx[row, :2] = torch.tensor([0, 1], dtype=torch.int32)
            if point[0] == 0:
                # Track 1 is nearer its original bbox: distances are 1.0 and 0.1.
                points_local[row, 0, 0] = 2.0
                points_local[row, 1, 0] = 2.1
            else:
                # Track 0 is nearer its original bbox: distances are 0.2 and 3.0.
                points_local[row, 0, 0] = 1.2
                points_local[row, 1, 0] = 5.0
        return _IntersectionResult(tracks_idx, points_local)


def test_associate_points_first_pass_preserves_trailing_scalar_shape():
    from instant_nurec.utils.motion import associate_points_with_cuboid_tracks

    tracks = _FakeAssociationTracks(torch.tensor([[1], [-1]], dtype=torch.int32))
    points = torch.zeros(2, 3)
    timestamps = torch.tensor([[10], [20]], dtype=torch.int64)

    result = associate_points_with_cuboid_tracks(points, timestamps, None, tracks, None)

    assert torch.equal(result, torch.tensor([[1], [-1]], dtype=torch.int32))
    assert tracks.calls[0]["max_intersections_per_point"] == 1
    assert tracks.calls[0]["timestamps"].shape == (2,)


def test_associate_points_empty_tracks_returns_typed_minus_one():
    from instant_nurec.utils.motion import associate_points_with_cuboid_tracks

    class _EmptyTracks:
        n_tracks = 0
        tracks_packinfo = torch.empty((0, 2), dtype=torch.int32)

    points = torch.zeros(2, 3, 4, 3)
    timestamps = torch.zeros(2, 3, 4, 1, dtype=torch.int64)
    result = associate_points_with_cuboid_tracks(points, timestamps, None, _EmptyTracks(), None)
    assert result.shape == (2, 3, 4, 1)
    assert result.dtype == torch.int32
    assert torch.all(result == -1)


def test_associate_points_widened_second_pass_selects_nearest_original_bbox():
    from instant_nurec.utils.motion import associate_points_with_cuboid_tracks

    tracks = _FakeAssociationTracks(torch.full((2, 1), -1, dtype=torch.int32))
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    timestamps = torch.zeros(2, 1, dtype=torch.int64)
    movable = torch.ones(2, 1, dtype=torch.bool)
    padding = torch.tensor([0.2, 0.1, 0.05])

    result = associate_points_with_cuboid_tracks(
        points,
        timestamps,
        movable,
        tracks,
        padding,
        second_pass_chunk_size=1,
    )

    assert torch.equal(result, torch.tensor([[1], [0]], dtype=torch.int32))
    assert len(tracks.calls) == 3
    for call in tracks.calls[1:]:
        assert call["max_intersections_per_point"] == 64
        assert call["with_local_points"] is True
        assert torch.allclose(call["padding"], padding * 6.0)


def test_associate_points_second_pass_only_checks_movable_misses():
    from instant_nurec.utils.motion import associate_points_with_cuboid_tracks

    tracks = _FakeAssociationTracks(torch.full((2, 1), -1, dtype=torch.int32))
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    timestamps = torch.zeros(2, 1, dtype=torch.int64)
    movable = torch.tensor([[True], [False]])

    result = associate_points_with_cuboid_tracks(points, timestamps, movable, tracks, None)

    assert torch.equal(result, torch.tensor([[1], [-1]], dtype=torch.int32))
    assert len(tracks.calls) == 2
    assert torch.equal(tracks.calls[1]["points"], points[:1])
    assert torch.equal(tracks.calls[1]["padding"], torch.full((3,), 6.0))


class _FakePose:
    def __init__(self, offset: torch.Tensor | None = None):
        self.offset = offset if offset is not None else torch.zeros(3)

    def inv(self) -> _FakePose:
        return _FakePose(offset=-self.offset)

    def __mul__(self, other):
        if isinstance(other, _FakePose):
            return _FakePose(offset=self.offset + other.offset)
        return other + self.offset


class _FakeWarpTracks:
    def __init__(self, target_offsets: list[torch.Tensor] | None = None):
        self.tracks_packinfo = torch.tensor([[0, 2]], dtype=torch.int32)
        self.target_offsets = target_offsets or []
        self.call_count = 0

    def interpolate_tracks_poses(self, timestamps_us, tracks_idx):
        assert tracks_idx.dtype == self.tracks_packinfo.dtype
        if self.call_count == 0:
            self.call_count += 1
            return _FakePose()
        offset = self.target_offsets[self.call_count - 1]
        self.call_count += 1
        return _FakePose(offset)


def test_warp_points_no_associations_short_circuits_with_clones():
    from instant_nurec.utils.motion import warp_points_with_cuboid_tracks

    points = torch.zeros(4, 3)
    source = torch.zeros(4, 1, dtype=torch.int64)
    targets = [torch.zeros(4, 1, dtype=torch.int64), torch.ones(4, 1, dtype=torch.int64)]
    tracks_idx = torch.full((4, 1), -1, dtype=torch.int32)

    dynamic_mask, warped = warp_points_with_cuboid_tracks(points, source, targets, _FakeWarpTracks(), tracks_idx)

    assert dynamic_mask.shape == (4, 1)
    assert not dynamic_mask.any()
    assert len(warped) == 2
    for result in warped:
        assert torch.equal(result, points)
        assert result.data_ptr() != points.data_ptr()


def test_warp_points_moves_only_assigned_points():
    from instant_nurec.utils.motion import warp_points_with_cuboid_tracks

    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    source = torch.zeros(2, 1, dtype=torch.int64)
    targets = [torch.ones(2, 1, dtype=torch.int64)]
    tracks_idx = torch.tensor([[0], [-1]], dtype=torch.int64)
    tracks = _FakeWarpTracks([torch.tensor([10.0, 0.0, 0.0])])

    dynamic_mask, warped = warp_points_with_cuboid_tracks(points, source, targets, tracks, tracks_idx)

    assert torch.equal(dynamic_mask, torch.tensor([[True], [False]]))
    assert torch.equal(warped[0], torch.tensor([[10.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))


def test_warp_points_requires_trailing_singleton_scalar_shapes():
    from instant_nurec.utils.motion import warp_points_with_cuboid_tracks

    points = torch.zeros(3, 3)
    source = torch.zeros(3, dtype=torch.int64)
    target = torch.zeros(3, 1, dtype=torch.int64)
    tracks_idx = torch.full((3, 1), -1, dtype=torch.int32)
    with pytest.raises(AssertionError):
        warp_points_with_cuboid_tracks(points, source, [target], _FakeWarpTracks(), tracks_idx)
