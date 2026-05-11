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

"""Branch-coverage tests for ``instant_nurec.datasets.tracks``.

Tests use
the in-tree ``se3`` shim, the only thing this fixture still has
to stub is ``ncore.data`` (transitively imported by
``instant_nurec.utils.types``). The heavier interface-equivalent
``_FakeSE3`` / ``_FakeSO3`` classes below remain as test stand-ins
because they only support the subset of the SE3/SO3 surface ``tracks.py``
exercises (``__getitem__`` / ``.shape`` / ``.dtype`` / ``.device``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Stubbed lt.SE3: enough to satisfy `__getitem__`, `.device`, `.shape`,
# `.dtype`, and being passed back into TracksData via the setter.
# ---------------------------------------------------------------------------


class _FakeSO3:
    """Tiny SO3 stand-in with the algebraic surface used by interpolate_tracks_poses."""

    def __init__(self, vec: torch.Tensor):
        self._vec = vec

    @classmethod
    def InitFromVec(cls, vec):
        return cls(vec)

    @classmethod
    def exp(cls, omega):
        # Fake: return identity quaternions with leading zero rotation vector,
        # but shaped as 4-vec to match the rotation-vec slot.
        n = omega.shape[0]
        out = torch.zeros(n, 4, device=omega.device)
        out[:, 3] = 1.0
        return cls(out)

    def vec(self):
        return self._vec

    def inv(self):
        return _FakeSO3(self._vec)

    def log(self):
        # Map back to a 3-vec (skipping the w component).
        return self._vec[:, :3]

    def __mul__(self, other):
        return _FakeSO3(self._vec)


class _FakeSE3:
    """Tensor-shaped wrapper exposing the lt.SE3 surface tracks.py touches."""

    def __init__(self, data: torch.Tensor):
        self._data = data

    @classmethod
    def InitFromVec(cls, data):
        return cls(data)

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def device(self):
        return self._data.device

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, idx):
        return _FakeSE3(self._data[idx])

    def to(self, device):
        return _FakeSE3(self._data.to(device))

    def vec(self):
        # tracks.py treats vec() as a (B, 7) tensor with [:, :3] = translation
        # and [:, 3:] = rotation (quaternion). Our _data is already that shape.
        return self._data

    def translation(self):
        return self._data[:, :3]


@pytest.fixture
def stubbed_tracks(monkeypatch):
    """Install ncore stubs and reload ``instant_nurec.datasets.tracks``."""
    ncore_mod = types.ModuleType("ncore")
    ncore_data_mod = types.ModuleType("ncore.data")
    ncore_data_mod.ConcreteCameraModelParametersUnion = object
    ncore_mod.data = ncore_data_mod
    monkeypatch.setitem(sys.modules, "ncore", ncore_mod)
    monkeypatch.setitem(sys.modules, "ncore.data", ncore_data_mod)

    # Drop cached modules so the new stubs take effect.
    for cached in (
        "instant_nurec.datasets.tracks",
        "instant_nurec.utils.types",
        "instant_nurec.utils.packed_ops",
    ):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    import importlib

    mod = importlib.import_module("instant_nurec.datasets.tracks")
    types_mod = importlib.import_module("instant_nurec.utils.types")
    return mod, types_mod


def _make_tracks_data(types_mod, *, n_tracks=2, total_poses=5, max_n=3):
    """Build a TracksData with shape-consistent torch tensors + fake SE3."""
    import torch

    return types_mod.TracksData(
        tracks_id=[f"t{i}" for i in range(n_tracks)],
        tracks_packinfo=torch.tensor(
            [[0, 2], [2, 3]],
            dtype=torch.int32,
        )[:n_tracks],
        tracks_poses=_FakeSE3(torch.zeros(total_poses, 7)),
        tracks_timestamps_us=torch.arange(total_poses, dtype=torch.int64),
        tracks_flags=torch.zeros(n_tracks, dtype=torch.int32),
        max_track_n_poses=max_n,
    )


def _make_cuboid_tracks_data(types_mod, *, n_tracks=2):
    return types_mod.CuboidTracksData(cuboids_dims=torch.ones(n_tracks, 3))


# ---------------------------------------------------------------------------
# Tracks property accessors
# ---------------------------------------------------------------------------


def test_tracks_property_accessors_passthrough(stubbed_tracks):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod)
    t = mod.Tracks(tracks_data=td)
    assert t.tracks_id == td.tracks_id
    assert t.max_track_n_poses == td.max_track_n_poses
    assert torch.equal(t.tracks_packinfo, td.tracks_packinfo)
    assert t.tracks_poses is td.tracks_poses
    assert torch.equal(t.tracks_timestamps_us, td.tracks_timestamps_us)
    assert torch.equal(t.tracks_flags, td.tracks_flags)


def test_tracks_poses_setter_replaces_with_consistent_shape(stubbed_tracks):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod)
    t = mod.Tracks(tracks_data=td)
    new_poses = _FakeSE3(torch.ones_like(td.tracks_poses.data))
    t.tracks_poses = new_poses
    assert t.tracks_poses is new_poses


def test_tracks_poses_setter_rejects_shape_mismatch(stubbed_tracks):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod)
    t = mod.Tracks(tracks_data=td)
    bad_poses = _FakeSE3(torch.zeros(td.tracks_poses.data.shape[0] + 3, 7))
    with pytest.raises(AssertionError):
        t.tracks_poses = bad_poses


def test_tracks_device_returns_pose_device(stubbed_tracks):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod)
    t = mod.Tracks(tracks_data=td)
    assert t.device == td.tracks_poses.device


def test_tracks_n_tracks_returns_id_length(stubbed_tracks):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod, n_tracks=2)
    t = mod.Tracks(tracks_data=td)
    assert t.n_tracks == 2


def test_tracks_to_device_returns_new_instance(stubbed_tracks, monkeypatch):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod)
    moved = _make_tracks_data(types_mod)
    # Force TracksData.to_device to return the prebuilt 'moved' so we can
    # assert identity rather than re-implementing CPU/GPU shuffles.
    monkeypatch.setattr(types_mod.TracksData, "to_device", lambda self, _device: moved)

    t = mod.Tracks(tracks_data=td)
    out = t.to_device(torch.device("cpu"))
    assert isinstance(out, mod.Tracks)
    assert out.tracks_data is moved


# ---------------------------------------------------------------------------
# CuboidTracks accessors and to_device
# ---------------------------------------------------------------------------


def test_cuboid_tracks_cuboids_dims_passthrough(stubbed_tracks):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod)
    ctd = _make_cuboid_tracks_data(types_mod)
    ct = mod.CuboidTracks(tracks_data=td, cuboidtracks_data=ctd)
    assert torch.equal(ct.cuboids_dims, ctd.cuboids_dims)


def test_cuboid_tracks_to_device_chains_data_to_device(
    stubbed_tracks, monkeypatch
):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod)
    ctd = _make_cuboid_tracks_data(types_mod)
    moved_td = _make_tracks_data(types_mod)
    moved_ctd = _make_cuboid_tracks_data(types_mod)
    monkeypatch.setattr(types_mod.TracksData, "to_device", lambda self, _d: moved_td)
    monkeypatch.setattr(types_mod.CuboidTracksData, "to_device", lambda self, _d: moved_ctd)

    ct = mod.CuboidTracks(tracks_data=td, cuboidtracks_data=ctd)
    out = ct.to_device(torch.device("cpu"))
    assert isinstance(out, mod.CuboidTracks)
    assert out.tracks_data is moved_td
    assert out.cuboidtracks_data is moved_ctd


# ---------------------------------------------------------------------------
# Factory.from_pack
# ---------------------------------------------------------------------------


def test_cuboid_tracks_factory_from_pack_passthrough(stubbed_tracks):
    mod, types_mod = stubbed_tracks
    td = _make_tracks_data(types_mod)
    ctd = _make_cuboid_tracks_data(types_mod)
    pack = types_mod.CuboidTracksDataPack(tracks_data=td, cuboidtracks_data=ctd)
    ct = mod.CuboidTracks.Factory.from_pack(pack)
    assert ct.tracks_data is td
    assert ct.cuboidtracks_data is ctd


# ---------------------------------------------------------------------------
# CuboidTracks.Ops.subset_from_indices and subset_from_mask
# ---------------------------------------------------------------------------


def _make_cuboid_tracks_with_two(stubbed_tracks):
    """Helper to build a CuboidTracks with exactly two tracks for subset tests."""
    mod, types_mod = stubbed_tracks
    td = types_mod.TracksData(
        tracks_id=["t0", "t1"],
        tracks_packinfo=torch.tensor([[0, 2], [2, 3]], dtype=torch.int32),
        tracks_poses=_FakeSE3(torch.arange(5 * 7).reshape(5, 7).float()),
        tracks_timestamps_us=torch.arange(5, dtype=torch.int64),
        tracks_flags=torch.tensor([10, 20], dtype=torch.int32),
        max_track_n_poses=3,
    )
    ctd = types_mod.CuboidTracksData(cuboids_dims=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    return mod.CuboidTracks(tracks_data=td, cuboidtracks_data=ctd)


def test_subset_from_indices_with_list_input(stubbed_tracks):
    mod, _ = stubbed_tracks
    ct = _make_cuboid_tracks_with_two(stubbed_tracks)
    out = mod.CuboidTracks.Ops.subset_from_indices(ct, [1])
    assert out.tracks_id == ["t1"]
    # Cuboid dims for t1 (index 1) → row [4, 5, 6].
    assert torch.equal(out.cuboids_dims, torch.tensor([[4.0, 5.0, 6.0]]))
    # Flags → only index 1's value (20).
    assert torch.equal(out.tracks_flags, torch.tensor([20], dtype=torch.int32))


def test_subset_from_indices_with_tensor_input(stubbed_tracks):
    mod, _ = stubbed_tracks
    ct = _make_cuboid_tracks_with_two(stubbed_tracks)
    out = mod.CuboidTracks.Ops.subset_from_indices(ct, torch.tensor([0, 1]))
    assert out.tracks_id == ["t0", "t1"]
    assert out.cuboids_dims.shape == (2, 3)


def test_subset_from_indices_empty_selection_returns_empty_tracks(stubbed_tracks):
    """An empty index list bypasses the linstep_interleave call — uses the
    empty-tensor short-circuit branch."""
    mod, _ = stubbed_tracks
    ct = _make_cuboid_tracks_with_two(stubbed_tracks)
    out = mod.CuboidTracks.Ops.subset_from_indices(ct, [])
    assert out.tracks_id == []
    assert out.cuboids_dims.shape == (0, 3)
    # max_track_n_poses falls back to 0 when track_counts is empty.
    assert out.max_track_n_poses == 0


def test_subset_from_mask_delegates_to_indices(stubbed_tracks, monkeypatch):
    """subset_from_mask is a thin nonzero+squeeze wrapper around subset_from_indices."""
    mod, _ = stubbed_tracks
    ct = _make_cuboid_tracks_with_two(stubbed_tracks)
    captured = {}

    real_impl = mod.CuboidTracks.Ops.subset_from_indices

    def _capture(ct_arg, indices):
        captured["indices"] = indices
        return real_impl(ct_arg, indices)

    monkeypatch.setattr(mod.CuboidTracks.Ops, "subset_from_indices", _capture)

    mask = torch.tensor([False, True])
    out = mod.CuboidTracks.Ops.subset_from_mask(ct, mask)
    assert torch.equal(captured["indices"], torch.tensor([1]))
    assert out.tracks_id == ["t1"]


# ---------------------------------------------------------------------------
# CuboidTracks.ray_intersection — wraps vren.ray_cuboidtracks_intersection
# ---------------------------------------------------------------------------


def test_ray_intersection_calls_vren_and_packs_result(stubbed_tracks, monkeypatch):
    mod, _ = stubbed_tracks
    ct = _make_cuboid_tracks_with_two(stubbed_tracks)

    captured = {}

    def _fake(rays_o, rays_d, ts, packinfo, poses_data, ts_us, dims, max_n, max_per_ray, with_ts):
        captured["max_n"] = max_n
        captured["max_per_ray"] = max_per_ray
        captured["with_ts"] = with_ts
        return (
            torch.tensor([1, 0, 2], dtype=torch.int32),
            torch.zeros(3, max_per_ray, dtype=torch.int32),
        )

    monkeypatch.setattr(mod, "_ray_cuboidtracks_intersection", _fake)

    rays_o = torch.zeros(3, 3)
    rays_d = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    ts = torch.tensor([100, 200, 300], dtype=torch.int64)
    out = ct.ray_intersection(rays_o, rays_d, ts, max_intersections_per_ray=8)

    assert torch.equal(out.intersections_cnt, torch.tensor([1, 0, 2], dtype=torch.int32))
    assert out.intersections_tracks_idx.shape == (3, 8)
    assert captured["max_n"] == ct.max_track_n_poses
    assert captured["max_per_ray"] == 8
    assert captured["with_ts"] is False  # predict path never reads .intersections_ts


# ---------------------------------------------------------------------------
# CuboidTracks.point_intersection_interpolate_pose — vren wrapper that
# also reshapes data and rebuilds an SE3.
# ---------------------------------------------------------------------------


def test_point_intersection_interpolate_pose_reshapes_and_rebuilds_se3(
    stubbed_tracks, monkeypatch
):
    mod, _ = stubbed_tracks
    ct = _make_cuboid_tracks_with_two(stubbed_tracks)

    captured = {}

    def _fake(points, points_ts, packinfo, poses_data, ts_us, padded_dims, max_n):
        captured["points_shape"] = points.shape
        captured["padded_dims"] = padded_dims
        return (
            torch.zeros(points.shape[0], 7),
            torch.zeros(points.shape[0], dtype=torch.long),
        )

    monkeypatch.setattr(
        mod, "_point_cuboidtracks_intersection_interpolate_pose", _fake
    )

    # Multi-dim input: (2, 3, 3)
    points = torch.zeros(2, 3, 3)
    points_ts = torch.zeros(2, 3, dtype=torch.int64)
    padding = torch.zeros(3)

    poses, idx = ct.point_intersection_interpolate_pose(points, points_ts, padding)
    # Reshape preserves leading dims.
    assert idx.shape == (2, 3)
    # Returned SE3 wraps a tensor of shape (2, 3, 7).
    assert poses.data.shape == (2, 3, 7)
    # SUT flattens points to (N, 3) before passing to vren.
    assert captured["points_shape"] == (6, 3)
    # SUT adds cuboids_dims + padding before calling vren.
    expected_padded = ct.cuboids_dims + padding
    assert torch.equal(captured["padded_dims"], expected_padded)


# ---------------------------------------------------------------------------
# Tracks.Factory.from_numpy — pose-format & assertion branches
# ---------------------------------------------------------------------------


import numpy as np  # noqa: E402  (after the heavy fixture imports)


def _matrix_pose(n: int) -> np.ndarray:
    """Return n random-but-valid 4x4 float32 SE(3) matrices (identity here)."""
    out = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    return out


def _tquat_pose(n: int) -> np.ndarray:
    """Return n random-but-valid 7-vector tquat float32 poses (identity)."""
    out = np.zeros((n, 7), dtype=np.float32)
    out[:, 6] = 1.0  # w of unit quaternion
    return out


def test_factory_from_numpy_matrix_format(stubbed_tracks):
    mod, _ = stubbed_tracks
    track = mod.Tracks.Factory.from_numpy(
        tracks_id=["a", "b"],
        tracks_poses=[_matrix_pose(2), _matrix_pose(3)],
        tracks_timestamps_us=[
            np.arange(2, dtype=np.int64),
            np.arange(3, dtype=np.int64),
        ],
        tracks_flags=[
            types.SimpleNamespace(value=1),
            types.SimpleNamespace(value=2),
        ],
        pose_format="matrix",
        device=torch.device("cpu"),
    )
    assert track.tracks_id == ["a", "b"]
    assert track.max_track_n_poses == 3
    # tracks_packinfo is a 2-row [start, count] tensor.
    assert track.tracks_packinfo.shape == (2, 2)
    assert int(track.tracks_packinfo[1, 1].item()) == 3
    # Flags get pulled from the .value attribute.
    assert track.tracks_flags.tolist() == [1, 2]


def test_factory_from_numpy_tquat_format(stubbed_tracks):
    mod, _ = stubbed_tracks
    track = mod.Tracks.Factory.from_numpy(
        tracks_id=["a"],
        tracks_poses=[_tquat_pose(2)],
        tracks_timestamps_us=[np.arange(2, dtype=np.int64)],
        tracks_flags=[types.SimpleNamespace(value=0)],
        pose_format="tquat",
        device=torch.device("cpu"),
    )
    assert track.tracks_id == ["a"]
    assert track.max_track_n_poses == 2


def test_factory_from_numpy_rejects_id_pose_count_mismatch(stubbed_tracks):
    mod, _ = stubbed_tracks
    with pytest.raises(AssertionError, match="track_ids / pose"):
        mod.Tracks.Factory.from_numpy(
            tracks_id=["a", "b"],
            tracks_poses=[_matrix_pose(2)],  # 1 entry, but 2 ids
            tracks_timestamps_us=[np.arange(2, dtype=np.int64)],
            tracks_flags=[types.SimpleNamespace(value=0)],
        )


def test_factory_from_numpy_rejects_pose_timestamp_mismatch(stubbed_tracks):
    mod, _ = stubbed_tracks
    with pytest.raises(AssertionError, match="pose / timestamp"):
        mod.Tracks.Factory.from_numpy(
            tracks_id=["a"],
            tracks_poses=[_matrix_pose(2)],
            tracks_timestamps_us=[np.arange(3, dtype=np.int64)],  # wrong length
            tracks_flags=[types.SimpleNamespace(value=0)],
        )


def test_factory_from_numpy_rejects_short_track(stubbed_tracks):
    mod, _ = stubbed_tracks
    with pytest.raises(AssertionError, match="at least two poses"):
        mod.Tracks.Factory.from_numpy(
            tracks_id=["a"],
            tracks_poses=[_matrix_pose(1)],
            tracks_timestamps_us=[np.arange(1, dtype=np.int64)],
            tracks_flags=[types.SimpleNamespace(value=0)],
        )


def test_factory_from_numpy_rejects_invalid_pose_dtype(stubbed_tracks):
    mod, _ = stubbed_tracks
    bad_pose = _matrix_pose(2).astype(np.float64)  # not float32
    with pytest.raises(AssertionError, match="invalid poses inputs"):
        mod.Tracks.Factory.from_numpy(
            tracks_id=["a"],
            tracks_poses=[bad_pose],
            tracks_timestamps_us=[np.arange(2, dtype=np.int64)],
            tracks_flags=[types.SimpleNamespace(value=0)],
        )


def test_factory_from_numpy_rejects_id_flags_mismatch(stubbed_tracks):
    mod, _ = stubbed_tracks
    with pytest.raises(AssertionError, match="track_ids / flags"):
        mod.Tracks.Factory.from_numpy(
            tracks_id=["a", "b"],
            tracks_poses=[_matrix_pose(2), _matrix_pose(2)],
            tracks_timestamps_us=[
                np.arange(2, dtype=np.int64),
                np.arange(2, dtype=np.int64),
            ],
            tracks_flags=[types.SimpleNamespace(value=0)],  # 1 flag, 2 ids
        )


def test_factory_from_numpy_rejects_invalid_timestamp_dtype(stubbed_tracks):
    mod, _ = stubbed_tracks
    bad_ts = np.arange(2, dtype=np.int32)  # not int64
    with pytest.raises(AssertionError, match="invalid tracks_timestamps_us"):
        mod.Tracks.Factory.from_numpy(
            tracks_id=["a"],
            tracks_poses=[_matrix_pose(2)],
            tracks_timestamps_us=[bad_ts],
            tracks_flags=[types.SimpleNamespace(value=0)],
        )


# ---------------------------------------------------------------------------
# CuboidTracks.Factory.from_numpy — wraps Tracks.Factory plus dims
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CuboidTracks.interpolate_tracks_poses — needs full lt.SE3 + lt.SO3 stubs
# ---------------------------------------------------------------------------


def test_interpolate_tracks_poses_returns_se3_with_right_shape(
    stubbed_tracks, monkeypatch
):
    mod, _ = stubbed_tracks
    ct = _make_cuboid_tracks_with_two(stubbed_tracks)

    # Override the imported torch helper to return a controllable index
    # (1 for everything → tidx_left=0, tidx_right=1)
    # replaced the kernel call with a module-level binding in tracks.py.
    def _fake_searchsorted(ts_us, packinfo, query_ts, tracks_idx):
        return torch.full((query_ts.shape[0],), 1, dtype=torch.long)

    monkeypatch.setattr(
        mod, "_packed_searchsorted_indexed_vals", _fake_searchsorted
    )

    timestamps_us = torch.tensor([5, 10], dtype=torch.int64)
    tracks_idx = torch.tensor([0, 0], dtype=torch.long)
    out = ct.interpolate_tracks_poses(timestamps_us, tracks_idx)
    # SUT returns lt.SE3.InitFromVec(torch.cat([t, R], dim=1)).
    # ``lt`` is ``se3.SE3`` (in-tree).
    assert hasattr(out, "data")
    assert out.data.shape == (2, 7)


def test_cuboid_factory_from_numpy_includes_dims(stubbed_tracks):
    mod, _ = stubbed_tracks
    ct = mod.CuboidTracks.Factory.from_numpy(
        tracks_id=["a", "b"],
        tracks_poses=[_matrix_pose(2), _matrix_pose(2)],
        tracks_timestamps_us=[
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
        ],
        tracks_flags=[
            types.SimpleNamespace(value=0),
            types.SimpleNamespace(value=1),
        ],
        cuboids_dims=[
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ],
        device=torch.device("cpu"),
    )
    assert ct.tracks_id == ["a", "b"]
    assert torch.equal(
        ct.cuboids_dims,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
