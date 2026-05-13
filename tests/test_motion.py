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
            [0, 0],       # cam 0
            [1000, 1000], # cam 0
            [50, 50],     # cam 1
            [9999, 9999], # cam 1
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
# warp_points_with_cuboid_tracks
# ---------------------------------------------------------------------------


class _FakePose:
    """Stand-in for an SE3 pose object — supports .inv() and __mul__ returning
    self-or-other in such a way that the warp formula leaves points unchanged.

    We construct two flavors:
      - identity-like (any * pose == pose, pose * point == point)
      - shifting (pose * point == point + offset)
    """

    def __init__(self, offset: torch.Tensor | None = None):
        self.offset = offset if offset is not None else torch.zeros(3)

    def inv(self) -> _FakePose:
        return _FakePose(offset=-self.offset)

    def __mul__(self, other):
        if isinstance(other, _FakePose):
            return _FakePose(offset=self.offset + other.offset)
        # apply to points (broadcast)
        return other + self.offset


class _FakeCuboidTracks:
    """Duck-typed stand-in for CuboidTracks with the methods used by
    ``warp_points_with_cuboid_tracks``."""

    def __init__(self, tracks_idx: torch.Tensor, target_offsets: list[torch.Tensor] | None = None):
        self._tracks_idx = tracks_idx
        # Per-target offset vectors used in interpolate_tracks_poses for target ts.
        # Source poses are always identity.
        self._target_offsets = target_offsets or []
        self._call_count = 0

    def point_intersection_interpolate_pose(self, points, src_ts, padding):
        # ignore points/src_ts; return precanned tracks_idx
        return None, self._tracks_idx.clone()

    def interpolate_tracks_poses(self, timestamps_us, tracks_idx):
        if self._call_count == 0:
            # First call is the source (inv called outside) — return identity
            self._call_count += 1
            return _FakePose(offset=torch.zeros(3))
        # Subsequent calls are target poses — return offset
        offset = self._target_offsets[(self._call_count - 1) % max(1, len(self._target_offsets))]
        self._call_count += 1
        return _FakePose(offset=offset)


def test_warp_points_no_dynamic_short_circuits_with_clones():
    """When tracks_idx is all -1 (no associations), the warp should just
    return identical clones of the input for every target."""
    from instant_nurec.utils.motion import warp_points_with_cuboid_tracks

    points = torch.zeros(4, 3)
    src_ts = torch.zeros(4, dtype=torch.int64)
    tgt_ts = [torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)]
    aux = torch.full((4,), -1, dtype=torch.int64)  # no fallback

    fake_tracks = _FakeCuboidTracks(tracks_idx=torch.full((4,), -1, dtype=torch.int64))
    cuboids_dims_padding = torch.zeros(1, 3)

    dynamic_mask, warped = warp_points_with_cuboid_tracks(
        points, src_ts, tgt_ts, fake_tracks, aux, cuboids_dims_padding
    )
    assert dynamic_mask.shape == (4,)
    assert not dynamic_mask.any()
    assert len(warped) == 2
    for w, ts in zip(warped, tgt_ts):
        # Identical to the input for unassociated points
        assert torch.equal(w, points)


def test_warp_points_falls_back_to_aux_when_main_returns_minus_one():
    """If point_intersection_interpolate_pose returns -1 for an entry,
    the function should fall back to aux_tracks_idx (and only -1 there
    means no association)."""
    from instant_nurec.utils.motion import warp_points_with_cuboid_tracks

    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    src_ts = torch.zeros(2, dtype=torch.int64)
    tgt_ts = [torch.zeros(2, dtype=torch.int64)]
    aux = torch.tensor([0, -1], dtype=torch.int64)  # first falls back, second stays unassociated

    fake_tracks = _FakeCuboidTracks(
        tracks_idx=torch.tensor([-1, -1], dtype=torch.int64),
        target_offsets=[torch.tensor([10.0, 0.0, 0.0])],
    )
    cuboids_dims_padding = torch.zeros(1, 3)

    dynamic_mask, warped = warp_points_with_cuboid_tracks(
        points, src_ts, tgt_ts, fake_tracks, aux, cuboids_dims_padding
    )
    # First point gets associated (via aux), second stays unassociated.
    assert dynamic_mask.tolist() == [True, False]
    # The warped points: first point was at origin, gets pose (target * inv(source)) applied.
    # source pose = identity (offset=0); target pose offset=10 → net offset = 10
    # So warped[0] should be (10, 0, 0); warped[1] should still be (1, 1, 1).
    assert torch.allclose(warped[0][0], torch.tensor([10.0, 0.0, 0.0]))
    assert torch.equal(warped[0][1], points[1])


def test_warp_points_squeeze_trailing_one_branch():
    """Source/target timestamps with shape [..., 1] are squeezed to [...]."""
    from instant_nurec.utils.motion import warp_points_with_cuboid_tracks

    points = torch.zeros(3, 3)
    # Squeezable timestamp shapes (..., 1)
    src_ts = torch.zeros(3, 1, dtype=torch.int64)
    tgt_ts = [torch.zeros(3, 1, dtype=torch.int64)]
    aux = torch.full((3,), -1, dtype=torch.int64)

    fake_tracks = _FakeCuboidTracks(tracks_idx=torch.full((3,), -1, dtype=torch.int64))
    cuboids_dims_padding = torch.zeros(1, 3)

    dynamic_mask, warped = warp_points_with_cuboid_tracks(
        points, src_ts, tgt_ts, fake_tracks, aux, cuboids_dims_padding
    )
    assert dynamic_mask.shape == (3,)


def test_warp_points_already_squeezed_timestamps_path():
    """Plain (...) shape (no trailing 1) goes through the no-squeeze branch."""
    from instant_nurec.utils.motion import warp_points_with_cuboid_tracks

    points = torch.zeros(3, 3)
    src_ts = torch.zeros(3, dtype=torch.int64)
    tgt_ts = [torch.zeros(3, dtype=torch.int64)]
    aux = torch.full((3,), -1, dtype=torch.int64)

    fake_tracks = _FakeCuboidTracks(tracks_idx=torch.full((3,), -1, dtype=torch.int64))
    cuboids_dims_padding = torch.zeros(1, 3)

    dynamic_mask, warped = warp_points_with_cuboid_tracks(
        points, src_ts, tgt_ts, fake_tracks, aux, cuboids_dims_padding
    )
    assert dynamic_mask.shape == (3,)
