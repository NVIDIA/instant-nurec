"""Branch-coverage tests for nre.nrm.utils.motion.TimeRemapping.

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

from nre.nrm.utils.motion import TimeRemapping


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
