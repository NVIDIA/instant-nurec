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

"""Branch-coverage tests for ``instant_nurec.datasets.samplers``.

The module imports from ``instant_nurec.utils.types`` which pulls in
``ncore.data``. We stub it via ``sys.modules``. Phase B replaced the

is needed.
"""

from __future__ import annotations

import sys
import types as _typesmod
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _stub_compiled_imports(monkeypatch: pytest.MonkeyPatch):
    ncore_mod = _typesmod.ModuleType("ncore")
    ncore_data_mod = _typesmod.ModuleType("ncore.data")
    ncore_data_mod.ConcreteCameraModelParametersUnion = type(  # type: ignore[attr-defined]
        "X", (), {}
    )
    monkeypatch.setitem(sys.modules, "ncore", ncore_mod)
    monkeypatch.setitem(sys.modules, "ncore.data", ncore_data_mod)

    # Force fresh import of samplers and types
    sys.modules.pop("instant_nurec.datasets.samplers", None)
    sys.modules.pop("instant_nurec.utils.types", None)


# ---------------------------------------------------------------------------
# get_closest_frame_index
# ---------------------------------------------------------------------------


def test_get_closest_frame_index_exact_match_returns_that_index():
    from instant_nurec.datasets.samplers import get_closest_frame_index

    ts = np.array([0, 100, 200, 300])
    assert get_closest_frame_index(ts, 200) == 2


def test_get_closest_frame_index_picks_nearest_neighbor():
    from instant_nurec.datasets.samplers import get_closest_frame_index

    ts = np.array([0, 100, 200, 300])
    assert get_closest_frame_index(ts, 110) == 1  # closer to 100 than 200
    assert get_closest_frame_index(ts, 250) == 2  # tied to 200/300; argmin picks first


def test_get_closest_frame_index_target_below_min():
    from instant_nurec.datasets.samplers import get_closest_frame_index

    ts = np.array([100, 200, 300])
    assert get_closest_frame_index(ts, -50) == 0


def test_get_closest_frame_index_target_above_max():
    from instant_nurec.datasets.samplers import get_closest_frame_index

    ts = np.array([100, 200, 300])
    assert get_closest_frame_index(ts, 99999) == 2


def test_get_closest_frame_index_returns_python_int():
    from instant_nurec.datasets.samplers import get_closest_frame_index

    ts = np.array([0, 100], dtype=np.uint64)
    out = get_closest_frame_index(ts, 50)
    assert isinstance(out, int) and not isinstance(out, np.integer)


# ---------------------------------------------------------------------------
# AdaptiveSequentialFrameBatchSampler
# ---------------------------------------------------------------------------


def _make_sampler(*, n_frames_per_sample: int = 4, **cfg_overrides):
    """``n_frames_per_sample`` is a constructor arg; tests pass it directly."""
    from instant_nurec.config_schema.dataset import AdaptiveSequentialFrameBatchSamplerConfig
    from instant_nurec.datasets.samplers import AdaptiveSequentialFrameBatchSampler

    base = dict(
        n_samples_per_sequence=8,
        max_frame_gap_timestamp_us=200_000,
    )
    base.update(cfg_overrides)
    cfg = AdaptiveSequentialFrameBatchSamplerConfig(**base)
    return AdaptiveSequentialFrameBatchSampler(cfg, n_frames_per_sample=n_frames_per_sample)


def test_sampler_constructor_rejects_zero_frames_per_sample():
    with pytest.raises(AssertionError, match="n_frames_per_sample"):
        _make_sampler(n_frames_per_sample=0)


def test_sampler_constructor_rejects_zero_samples_per_sequence():
    with pytest.raises(AssertionError, match="n_samples_per_sequence"):
        _make_sampler(n_samples_per_sequence=0)


def test_sampler_constructor_rejects_zero_max_frame_gap():
    with pytest.raises(AssertionError, match="max_frame_gap_timestamp_us"):
        _make_sampler(max_frame_gap_timestamp_us=0)


def test_sample_frame_batch_returns_correct_indices_per_camera():
    from instant_nurec.utils.types import HalfClosedInterval

    sampler = _make_sampler(
        n_frames_per_sample=4,
        n_samples_per_sequence=2,
        max_frame_gap_timestamp_us=200_000,
    )
    cam_a_ts = np.array([0, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000])
    cam_b_ts = np.array([50_000, 150_000, 250_000, 350_000, 450_000, 550_000, 650_000, 750_000])
    intervals = [HalfClosedInterval(0, 800_000)]
    out = sampler.sample_frame_batch(
        sample_idx=0,
        camera_frame_timestamps_us={"cam_a": cam_a_ts, "cam_b": cam_b_ts},
        time_intervals=intervals,
    )
    assert set(out.keys()) == {"cam_a", "cam_b"}
    assert len(out["cam_a"]) == 4
    assert len(out["cam_b"]) == 4


def test_sample_frame_batch_rejects_empty_camera_dict():
    from instant_nurec.utils.types import HalfClosedInterval

    sampler = _make_sampler()
    with pytest.raises(AssertionError, match="No camera timestamps"):
        sampler.sample_frame_batch(
            sample_idx=0, camera_frame_timestamps_us={}, time_intervals=[HalfClosedInterval(0, 1)]
        )


def test_sample_frame_batch_rejects_out_of_range_sample_idx():
    from instant_nurec.utils.types import HalfClosedInterval

    sampler = _make_sampler(n_samples_per_sequence=2)
    cams = {"a": np.array([0, 100])}
    intervals = [HalfClosedInterval(0, 100)]
    with pytest.raises(AssertionError, match="Sample index out of bounds"):
        sampler.sample_frame_batch(
            sample_idx=2, camera_frame_timestamps_us=cams, time_intervals=intervals
        )
    with pytest.raises(AssertionError, match="Sample index out of bounds"):
        sampler.sample_frame_batch(
            sample_idx=-1, camera_frame_timestamps_us=cams, time_intervals=intervals
        )


def test_sample_frame_batch_rejects_empty_time_intervals():
    sampler = _make_sampler()
    with pytest.raises(AssertionError, match="No time intervals"):
        sampler.sample_frame_batch(
            sample_idx=0, camera_frame_timestamps_us={"a": np.array([0])}, time_intervals=[]
        )


def test_sample_frame_batch_returns_empty_when_chunk_count_under_sample_idx():
    """When the sequence is short enough that fewer chunks are needed than
    sample_idx, the method returns an empty dict."""
    from instant_nurec.utils.types import HalfClosedInterval

    sampler = _make_sampler(
        n_frames_per_sample=4,
        n_samples_per_sequence=8,
        max_frame_gap_timestamp_us=10_000_000,
    )
    cams = {"a": np.array([0, 100_000])}
    # 100us span with 40,000,000us max_chunk_timespan → only 1 chunk needed
    intervals = [HalfClosedInterval(0, 100_000)]
    # sample_idx=0 returns; sample_idx >=1 returns {}
    out_zero = sampler.sample_frame_batch(
        sample_idx=0, camera_frame_timestamps_us=cams, time_intervals=intervals
    )
    assert "a" in out_zero
    out_one = sampler.sample_frame_batch(
        sample_idx=1, camera_frame_timestamps_us=cams, time_intervals=intervals
    )
    assert out_one == {}


def test_sample_frame_batch_warns_when_clip_exceeds_max_chunks(caplog):
    """When the natural chunk count exceeds n_samples_per_sequence, the sampler
    emits a single warning naming the dropped chunk count and the suggested
    --max-chunks value."""
    import logging

    from instant_nurec.utils.types import HalfClosedInterval

    # 4 frames/sample * 1_000_000us gap = 4_000_000us per chunk.
    # 20_000_000us clip -> ceil(20/4) = 5 natural chunks, but cap is 2.
    sampler = _make_sampler(
        n_frames_per_sample=4,
        n_samples_per_sequence=2,
        max_frame_gap_timestamp_us=1_000_000,
    )
    cams = {"a": np.arange(0, 20_000_001, 500_000)}
    intervals = [HalfClosedInterval(0, 20_000_000)]

    with caplog.at_level(logging.WARNING, logger="instant_nurec.datasets.samplers"):
        sampler.sample_frame_batch(
            sample_idx=0, camera_frame_timestamps_us=cams, time_intervals=intervals
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "3 chunk(s) will be silently dropped" in msg
    assert "--max-chunks to 5" in msg


def test_sample_frame_batch_no_warning_when_within_max_chunks(caplog):
    """No warning when the natural chunk count fits within n_samples_per_sequence."""
    import logging

    from instant_nurec.utils.types import HalfClosedInterval

    # Clip is short enough that only 1 chunk is needed; cap of 8 is plenty.
    sampler = _make_sampler(
        n_frames_per_sample=4,
        n_samples_per_sequence=8,
        max_frame_gap_timestamp_us=10_000_000,
    )
    cams = {"a": np.array([0, 100_000])}
    intervals = [HalfClosedInterval(0, 100_000)]

    with caplog.at_level(logging.WARNING, logger="instant_nurec.datasets.samplers"):
        sampler.sample_frame_batch(
            sample_idx=0, camera_frame_timestamps_us=cams, time_intervals=intervals
        )

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_sample_frame_batch_truncation_warning_fires_only_at_sample_idx_zero(caplog):
    """The truncation warning fires once per clip (at sample_idx=0), not on
    subsequent sample_idx calls for the same oversized clip."""
    import logging

    from instant_nurec.utils.types import HalfClosedInterval

    sampler = _make_sampler(
        n_frames_per_sample=4,
        n_samples_per_sequence=2,
        max_frame_gap_timestamp_us=1_000_000,
    )
    cams = {"a": np.arange(0, 20_000_001, 500_000)}
    intervals = [HalfClosedInterval(0, 20_000_000)]

    with caplog.at_level(logging.WARNING, logger="instant_nurec.datasets.samplers"):
        sampler.sample_frame_batch(
            sample_idx=1, camera_frame_timestamps_us=cams, time_intervals=intervals
        )

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_sample_frame_batch_multi_interval_uses_min_start_max_end():
    """When multiple intervals are provided, the sampled span is min(start)
    to max(end)."""
    from instant_nurec.utils.types import HalfClosedInterval

    sampler = _make_sampler(
        n_frames_per_sample=2,
        n_samples_per_sequence=4,
        max_frame_gap_timestamp_us=500_000,
    )
    cams = {"a": np.array([0, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000])}
    # Two intervals with a gap → min start 0, max end 800_000
    intervals = [HalfClosedInterval(0, 200_000), HalfClosedInterval(600_000, 800_000)]
    out = sampler.sample_frame_batch(
        sample_idx=0, camera_frame_timestamps_us=cams, time_intervals=intervals
    )
    assert "a" in out
    assert len(out["a"]) == 2
