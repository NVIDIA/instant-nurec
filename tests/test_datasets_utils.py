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

"""Branch-coverage tests for instant_nurec.datasets.utils.

The module orchestrates ncore-side cuboid-track ingest into pandas DataFrames
and per-track dicts. The dependencies (`ncore.data`, `ncore.impl.common.
transformations`) are compiled extensions we don't ship in the test venv;
we stub them via ``sys.modules`` and feed lightweight ``CuboidTrackObservation``
dataclass instances through the real function bodies.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# ncore stubs that work for both functions in datasets/utils.py.
# Installed once per test via the fixture below.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _BBox3:
    """Minimal `BBox3`-like: stores 6 floats and exposes `to_array()`."""

    cx: float = 0.0
    cy: float = 0.0
    cz: float = 0.0
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0
    yaw: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.cx, self.cy, self.cz, self.length, self.width, self.height, self.yaw],
            dtype=np.float64,
        )


@dataclasses.dataclass
class _Source:
    name: str


@dataclasses.dataclass
class _CuboidTrackObservation:
    """Stand-in for ncore.data.CuboidTrackObservation. The real type has more
    fields; these are the ones the SUT touches."""

    track_id: str
    timestamp_us: int
    reference_frame_id: str
    reference_frame_timestamp_us: int
    bbox3: _BBox3
    source: _Source
    source_version: str | None
    class_id: int

    @classmethod
    def from_dict(cls, d: dict) -> "_CuboidTrackObservation":
        # Mimics the serialized → typed conversion. Reconstruct nested fields.
        bb = d["bbox3"]
        if isinstance(bb, dict):
            bb = _BBox3(**bb)
        src = d["source"]
        if isinstance(src, dict):
            src = _Source(**src)
        # The dataframe `row.to_dict()` form puts every column into the dict;
        # keep the keys we know about and drop any extras.
        kwargs = dict(d)
        kwargs["bbox3"] = bb
        kwargs["source"] = src
        return cls(**kwargs)


def _ncore_transformations_module():
    """Build a fake ``ncore.impl.common.transformations`` module."""

    @dataclasses.dataclass
    class HalfClosedInterval:
        start: int
        stop: int

    def transform_bbox(bbox_array: np.ndarray, T: np.ndarray) -> np.ndarray:
        # Apply the translation part of T to the centroid; leave the rest alone.
        out = np.array(bbox_array, dtype=np.float64, copy=True)
        out[:3] = T[:3, :3] @ out[:3] + T[:3, 3]
        return out

    def bbox_pose(bbox_array: np.ndarray) -> np.ndarray:
        # Build a 4x4 pose where the translation column is the bbox centroid.
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = bbox_array[:3]
        return T

    mod = types.ModuleType("ncore.impl.common.transformations")
    mod.HalfClosedInterval = HalfClosedInterval
    mod.transform_bbox = transform_bbox
    mod.bbox_pose = bbox_pose
    return mod


@pytest.fixture
def stubbed_datasets_utils(monkeypatch):
    ncore_mod = types.ModuleType("ncore")
    data_mod = types.ModuleType("ncore.data")
    data_mod.CuboidTrackObservation = _CuboidTrackObservation

    class _SequenceLoaderProtocol:
        pass

    data_mod.SequenceLoaderProtocol = _SequenceLoaderProtocol
    # instant_nurec.utils.types pulls this union in too — shape doesn't matter, we
    # just need the names to resolve at import time.
    data_mod.ConcreteCameraModelParametersUnion = object
    ncore_mod.data = data_mod

    impl_mod = types.ModuleType("ncore.impl")
    common_mod = types.ModuleType("ncore.impl.common")
    transformations_mod = _ncore_transformations_module()
    common_mod.transformations = transformations_mod
    impl_mod.common = common_mod
    ncore_mod.impl = impl_mod

    for name, mod in [
        ("ncore", ncore_mod),
        ("ncore.data", data_mod),
        ("ncore.impl", impl_mod),
        ("ncore.impl.common", common_mod),
        ("ncore.impl.common.transformations", transformations_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    # Drop cached imports so the new stubs take effect.
    for cached in ("instant_nurec.datasets.utils", "instant_nurec.utils.types"):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    import importlib

    mod = importlib.import_module("instant_nurec.datasets.utils")
    return mod, _CuboidTrackObservation, _BBox3, _Source, transformations_mod


class _FakePoseGraph:
    """Stand-in for sequence_loader.pose_graph with the methods the SUT calls."""

    def __init__(self, world_poses=None, rig_poses=None, missing_rig_for=None):
        # world_poses / rig_poses: dict[(frame_id, ts) → np.ndarray | None]
        self._world = world_poses or {}
        self._rig = rig_poses or {}
        self._missing_rig = set(missing_rig_for or [])

    def evaluate_poses(self, source_frame, target_frame, ts_array):
        ts = int(ts_array.item()) if hasattr(ts_array, "item") else int(ts_array)
        key = (source_frame, ts)
        if target_frame == "world":
            return self._world.get(key, np.eye(4))
        if target_frame == "rig":
            if key in self._missing_rig:
                raise KeyError(key)
            return self._rig.get(key, np.eye(4))
        raise ValueError(target_frame)


class _FakeSequenceLoader:
    """Carries .pose_graph + .get_cuboid_track_observations()."""

    def __init__(self, observations, pose_graph=None):
        self._observations = list(observations)
        self.pose_graph = pose_graph or _FakePoseGraph()

    def get_cuboid_track_observations(self, *, timestamp_interval_us):
        # Real loader filters by interval; we just hand back what was preloaded.
        return list(self._observations)


# ---------------------------------------------------------------------------
# compute_cuboid_df
# ---------------------------------------------------------------------------


def _make_obs(track_id="t1", ts=100, **overrides):
    base = dict(
        track_id=track_id,
        timestamp_us=ts,
        reference_frame_id="rig",
        reference_frame_timestamp_us=ts,
        bbox3=_BBox3(cx=1.0, cy=2.0, cz=0.0),
        source=_Source(name="manual"),
        source_version="v1",
        class_id=0,
    )
    base.update(overrides)
    return _CuboidTrackObservation(**base)


def test_compute_cuboid_df_returns_sorted_dataframe(stubbed_datasets_utils):
    mod, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    obs = [
        _make_obs(track_id="b", ts=2),
        _make_obs(track_id="a", ts=1),
        _make_obs(track_id="a", ts=2),
    ]
    loader = _FakeSequenceLoader(obs)
    df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=10))
    assert list(df["track_id"]) == ["a", "a", "b"]
    assert list(df["timestamp_us"]) == [1, 2, 2]


def test_compute_cuboid_df_empty_observations_returns_typed_empty_frame(
    stubbed_datasets_utils,
):
    """The empty-list branch must initialise an empty DataFrame whose columns
    match the dataclass fields of CuboidTrackObservation."""
    mod, _Obs, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    loader = _FakeSequenceLoader([])
    df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=10))
    expected_cols = {f.name for f in dataclasses.fields(_Obs)}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 0


# ---------------------------------------------------------------------------
# consolidate_cuboid_tracks — happy path + branch coverage
# ---------------------------------------------------------------------------


def test_consolidate_creates_new_track_and_records_pose(stubbed_datasets_utils):
    mod, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    obs = [_make_obs(track_id="A", ts=10)]
    loader = _FakeSequenceLoader(obs)
    cuboids_df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=100))

    out = mod.consolidate_cuboid_tracks(
        cuboids_df,
        loader,
        track_label_sources=["manual@v1"],
        track_min_centroid_rig_dist_m=0.0,
        T_world_world_base=np.eye(4),
    )
    assert "A" in out
    track = out["A"]
    assert len(track["poses"]) == 1
    assert len(track["timestamps_us"]) == 1
    assert track["timestamps_us"] == [10]
    assert track["dimension"].shape == (3,)
    assert track["label_class"] == 0


def test_consolidate_filters_observations_too_close_to_rig(stubbed_datasets_utils):
    """Observations within `track_min_centroid_rig_dist_m` from the rig
    centre are dropped (self-classification filter)."""
    mod, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    # bbox_rig = T_reference_rig @ bbox; with eye(4), bbox_rig[:3] == centroid.
    # Centroid (0.5, 0, 0) → norm 0.5, threshold 1.0 → drop.
    near_obs = _make_obs(bbox3=_BBox3(cx=0.5, cy=0.0, cz=0.0))
    far_obs = _make_obs(bbox3=_BBox3(cx=10.0, cy=0.0, cz=0.0))
    loader = _FakeSequenceLoader([near_obs, far_obs])

    df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=100))
    out = mod.consolidate_cuboid_tracks(
        df,
        loader,
        track_label_sources=["manual@any"],
        track_min_centroid_rig_dist_m=1.0,
        T_world_world_base=np.eye(4),
    )
    # The near observation was filtered, the far one survives.
    assert len(out["t1"]["poses"]) == 1


def test_consolidate_skips_observations_with_unmatched_label_source(
    stubbed_datasets_utils,
):
    """Label-source mismatch (versioned filter doesn't include the obs) → skip."""
    mod, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    matching = _make_obs(track_id="ok", source=_Source(name="manual"), source_version="v1")
    nonmatching = _make_obs(track_id="bad", source=_Source(name="manual"), source_version="v2")
    loader = _FakeSequenceLoader([matching, nonmatching])
    df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=100))

    out = mod.consolidate_cuboid_tracks(
        df,
        loader,
        track_label_sources=["manual@v1"],  # only v1 matches
        track_min_centroid_rig_dist_m=0.0,
        T_world_world_base=np.eye(4),
    )
    assert "ok" in out and "bad" not in out


def test_consolidate_unversioned_label_source_matches_any_version(stubbed_datasets_utils):
    """A track_label_source string without `@version` matches any version
    (i.e. gets the `@any` synthetic appended)."""
    mod, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    o1 = _make_obs(track_id="t1", source=_Source(name="auto"), source_version="v1")
    o2 = _make_obs(track_id="t2", source=_Source(name="auto"), source_version="v9")
    loader = _FakeSequenceLoader([o1, o2])
    df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=100))

    out = mod.consolidate_cuboid_tracks(
        df,
        loader,
        track_label_sources=["auto"],  # no @version → matches any
        track_min_centroid_rig_dist_m=0.0,
        T_world_world_base=np.eye(4),
    )
    assert {"t1", "t2"}.issubset(out.keys())


def test_consolidate_skips_duplicate_timestamp_within_track(stubbed_datasets_utils):
    """Two observations of the same track at the same timestamp → second is
    dropped by the dedup check on `track['timestamps_us'][-2:]`."""
    mod, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    o1 = _make_obs(track_id="A", ts=10)
    o2 = _make_obs(track_id="A", ts=10)
    o3 = _make_obs(track_id="A", ts=11)
    loader = _FakeSequenceLoader([o1, o2, o3])
    df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=100))
    out = mod.consolidate_cuboid_tracks(
        df,
        loader,
        track_label_sources=["manual@any"],
        track_min_centroid_rig_dist_m=0.0,
        T_world_world_base=np.eye(4),
    )
    assert out["A"]["timestamps_us"] == [10, 11]


def test_consolidate_handles_missing_rig_pose(stubbed_datasets_utils):
    """When pose_graph.evaluate_poses(..., 'rig', ...) raises KeyError, the
    rig-distance filter is skipped (T_reference_rig is None branch)."""
    mod, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    loader = _FakeSequenceLoader(
        [_make_obs(track_id="A", ts=10, bbox3=_BBox3(cx=0.0, cy=0.0, cz=0.0))],
        pose_graph=_FakePoseGraph(missing_rig_for=[("rig", 10)]),
    )
    df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=100))
    out = mod.consolidate_cuboid_tracks(
        df,
        loader,
        track_label_sources=["manual@any"],
        track_min_centroid_rig_dist_m=999.0,  # would normally drop everything
        T_world_world_base=np.eye(4),
    )
    # Despite the huge threshold, the observation passes because rig pose is
    # missing → the distance check is skipped entirely.
    assert "A" in out
    assert len(out["A"]["poses"]) == 1


def test_consolidate_serialized_bbox3_is_reconstructed_via_from_dict(
    stubbed_datasets_utils,
):
    """If the row's `bbox3` is already a dict (serialized form), the SUT goes
    through the `from_dict` reconstruction branch instead of plain init."""
    mod, *_ = stubbed_datasets_utils
    from instant_nurec.utils.types import HalfClosedInterval

    # Build the DataFrame with serialized bbox3 (dict instead of _BBox3).
    obs = _make_obs(track_id="A", ts=10)
    loader = _FakeSequenceLoader([obs])
    df = mod.compute_cuboid_df(loader, HalfClosedInterval(start=0, end=100))
    # Replace bbox3 cells with dicts to force the from_dict branch.
    df["bbox3"] = [dataclasses.asdict(b) for b in df["bbox3"]]
    df["source"] = [dataclasses.asdict(s) for s in df["source"]]

    out = mod.consolidate_cuboid_tracks(
        df,
        loader,
        track_label_sources=["manual@any"],
        track_min_centroid_rig_dist_m=0.0,
        T_world_world_base=np.eye(4),
    )
    assert "A" in out
