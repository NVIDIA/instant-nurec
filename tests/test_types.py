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

"""Branch-coverage tests for ``instant_nurec.utils.types``.

The module imports ``ncore.data`` at module load 
replaced by the in-tree ``se3`` shim in Phase B and no longer
needs stubbing). The two pure-python / pure-numpy types we exercise
(``HalfClosedInterval`` and ``FrameConversion``) don't actually use the
stubbed names at runtime.
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
    """Provide minimal sys.modules stubs so ``import instant_nurec.utils.types``
    succeeds without ncore."""
    ncore_mod = _typesmod.ModuleType("ncore")
    ncore_data_mod = _typesmod.ModuleType("ncore.data")
    ncore_data_mod.ConcreteCameraModelParametersUnion = type(  # type: ignore[attr-defined]
        "CCMP", (), {}
    )
    monkeypatch.setitem(sys.modules, "ncore", ncore_mod)
    monkeypatch.setitem(sys.modules, "ncore.data", ncore_data_mod)

    # Force a fresh import (drop any prior cached version).
    sys.modules.pop("instant_nurec.utils.types", None)


# ---------------------------------------------------------------------------
# HalfClosedInterval
# ---------------------------------------------------------------------------


def test_halfclosed_post_init_accepts_valid_interval():
    from instant_nurec.utils.types import HalfClosedInterval

    h = HalfClosedInterval(0, 10)
    assert h.start == 0
    assert h.end == 10


def test_halfclosed_post_init_accepts_empty_interval():
    """start == end is valid (the interval is just empty)."""
    from instant_nurec.utils.types import HalfClosedInterval

    h = HalfClosedInterval(5, 5)
    assert h.start == 5 and h.end == 5


def test_halfclosed_post_init_rejects_inverted_interval():
    from instant_nurec.utils.types import HalfClosedInterval

    with pytest.raises(AssertionError):
        HalfClosedInterval(10, 5)


def test_halfclosed_intersection_overlapping():
    from instant_nurec.utils.types import HalfClosedInterval

    a = HalfClosedInterval(0, 10)
    b = HalfClosedInterval(5, 15)
    out = a.intersection(b)
    assert out is not None
    assert out.start == 5
    assert out.end == 10


def test_halfclosed_intersection_subset():
    from instant_nurec.utils.types import HalfClosedInterval

    a = HalfClosedInterval(0, 100)
    b = HalfClosedInterval(20, 30)
    out = a.intersection(b)
    assert out is not None
    assert out.start == 20 and out.end == 30


def test_halfclosed_intersection_disjoint_other_to_the_right():
    """First branch: other.start >= self.end."""
    from instant_nurec.utils.types import HalfClosedInterval

    a = HalfClosedInterval(0, 10)
    b = HalfClosedInterval(10, 20)
    assert a.intersection(b) is None  # touching at the half-open end → empty


def test_halfclosed_intersection_disjoint_other_to_the_left():
    """Second branch: other.end <= self.start."""
    from instant_nurec.utils.types import HalfClosedInterval

    a = HalfClosedInterval(10, 20)
    b = HalfClosedInterval(0, 10)
    assert a.intersection(b) is None  # touching at the closed start → empty


# ---------------------------------------------------------------------------
# FrameConversion
# ---------------------------------------------------------------------------


def test_frameconversion_post_init_accepts_identity():
    from instant_nurec.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float64))
    assert fc.target_scale == 1.0
    assert fc.dtype == np.float64


def test_frameconversion_post_init_rejects_wrong_shape():
    from instant_nurec.utils.types import FrameConversion

    with pytest.raises(AssertionError):
        FrameConversion(matrix=np.eye(3, dtype=np.float64))


def test_frameconversion_post_init_rejects_non_floating_dtype():
    from instant_nurec.utils.types import FrameConversion

    with pytest.raises(TypeError, match="floating point"):
        FrameConversion(matrix=np.eye(4, dtype=np.int32))


def test_frameconversion_post_init_rejects_non_positive_scale_entry():
    from instant_nurec.utils.types import FrameConversion

    bad = np.eye(4, dtype=np.float64)
    bad[3, 3] = 0.0
    with pytest.raises(AssertionError):
        FrameConversion(matrix=bad)


def test_frameconversion_post_init_rejects_non_rotation_3x3():
    """The (3,3) block must be a rotation (det == 1)."""
    from instant_nurec.utils.types import FrameConversion

    bad = np.eye(4, dtype=np.float64)
    bad[0, 0] = 2.0  # det = 2
    with pytest.raises(AssertionError):
        FrameConversion(matrix=bad)


def test_frameconversion_target_scale_inverse_of_bottomright():
    """target_scale = 1 / matrix[3,3]."""
    from instant_nurec.utils.types import FrameConversion

    m = np.eye(4, dtype=np.float64)
    m[3, 3] = 0.5  # i.e. source -> target scale = 2.0
    fc = FrameConversion(matrix=m)
    assert fc.target_scale == 2.0


def test_frameconversion_get_transformation_matrices_identity():
    from instant_nurec.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float32))
    T, S = fc.get_transformation_matrices()
    assert T.shape == (4, 4) and S.shape == (4, 4)
    assert T.dtype == np.float32 and S.dtype == np.float32
    np.testing.assert_allclose(T, np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(S, np.eye(4, dtype=np.float32))


def test_frameconversion_get_transformation_matrices_with_scale():
    """target_scale=2 means T scales the rotation block by 2 and S has 0.5
    on the diagonal of the leading 3x3 (1/s with s=2)."""
    from instant_nurec.utils.types import FrameConversion

    m = np.eye(4, dtype=np.float64)
    m[3, 3] = 0.5  # target_scale = 2
    fc = FrameConversion(matrix=m)
    T, S = fc.get_transformation_matrices()
    # T = m * target_scale = m * 2 → leading 3x3 is 2*I, but bottom-right is 1
    # (since m[3,3] * 2 = 1).
    assert T[0, 0] == 2.0 and T[1, 1] == 2.0 and T[2, 2] == 2.0
    assert T[3, 3] == 1.0
    # S has 1/s = 0.5 on first 3 diagonal entries, 1.0 on the 4th.
    assert S[0, 0] == 0.5 and S[1, 1] == 0.5 and S[2, 2] == 0.5
    assert S[3, 3] == 1.0


def test_frameconversion_transform_poses_singular_input():
    """Input is a single (4,4) pose; output is also (4,4)."""
    from instant_nurec.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float64))
    p = np.eye(4, dtype=np.float64)
    out = fc.transform_poses(p)
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out, p)


def test_frameconversion_transform_poses_batched_input():
    """Input is (N,4,4); output is (N,4,4)."""
    from instant_nurec.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float64))
    p = np.tile(np.eye(4, dtype=np.float64)[None], (3, 1, 1))
    out = fc.transform_poses(p)
    assert out.shape == (3, 4, 4)
    np.testing.assert_allclose(out, p)


def test_frameconversion_transform_poses_casts_to_declared_dtype():
    """If conversion dtype is float32 and input is float64, output is float32."""
    from instant_nurec.utils.types import FrameConversion

    fc = FrameConversion(matrix=np.eye(4, dtype=np.float32))
    p = np.eye(4, dtype=np.float64)
    out = fc.transform_poses(p)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# RigTrajectories.RigTrajectory.__post_init__
# ---------------------------------------------------------------------------


def test_rigtrajectory_post_init_accepts_valid_inputs():
    import torch
    from instant_nurec.utils.types import RigTrajectories

    rt = RigTrajectories.RigTrajectory(
        sequence_id="seq",
        cameras_frame_timestamps_us={"cam_a": torch.tensor([[0, 100], [200, 300]])},
        T_rig_worlds=torch.eye(4)[None].repeat(2, 1, 1).double(),
        T_rig_world_timestamps_us=torch.tensor([0, 100]),
    )
    assert rt.sequence_id == "seq"


def test_rigtrajectory_post_init_rejects_2d_timestamps():
    import torch
    from instant_nurec.utils.types import RigTrajectories

    with pytest.raises(AssertionError, match="must be 1D"):
        RigTrajectories.RigTrajectory(
            sequence_id="s",
            cameras_frame_timestamps_us={},
            T_rig_worlds=torch.eye(4)[None].double(),
            T_rig_world_timestamps_us=torch.tensor([[0]]),  # 2D
        )


def test_rigtrajectory_post_init_rejects_pose_count_mismatch():
    import torch
    from instant_nurec.utils.types import RigTrajectories

    with pytest.raises(AssertionError):
        RigTrajectories.RigTrajectory(
            sequence_id="s",
            cameras_frame_timestamps_us={},
            T_rig_worlds=torch.eye(4)[None].repeat(2, 1, 1).double(),  # 2 poses
            T_rig_world_timestamps_us=torch.tensor([0]),  # 1 timestamp
        )


def test_rigtrajectory_post_init_rejects_wrong_camera_ts_shape():
    import torch
    from instant_nurec.utils.types import RigTrajectories

    with pytest.raises(AssertionError):
        RigTrajectories.RigTrajectory(
            sequence_id="s",
            cameras_frame_timestamps_us={"cam": torch.tensor([[0, 1, 2]])},  # 3 not 2
            T_rig_worlds=torch.eye(4)[None].double(),
            T_rig_world_timestamps_us=torch.tensor([0]),
        )


# ---------------------------------------------------------------------------
# RigTrajectories.__post_init__
# ---------------------------------------------------------------------------


def _make_rig_trajectories(camera_ids, traj_camera_ids=None):
    """Helper to build a minimal RigTrajectories instance for the post_init checks."""
    from collections import OrderedDict
    import torch
    from instant_nurec.utils.types import FrameConversion, RigTrajectories

    traj_camera_ids = traj_camera_ids if traj_camera_ids is not None else camera_ids

    rt = RigTrajectories.RigTrajectory(
        sequence_id="s",
        cameras_frame_timestamps_us={cid: torch.tensor([[0, 1]]) for cid in traj_camera_ids},
        T_rig_worlds=torch.eye(4)[None].double(),
        T_rig_world_timestamps_us=torch.tensor([0]),
    )

    cam_calibs = OrderedDict(
        (cid, RigTrajectories.CameraCalibration(
            sequence_id="s",
            unique_sensor_idx=i,
            T_sensor_rig=torch.eye(4),
            camera_model_parameters=object(),  # placeholder
        ))
        for i, cid in enumerate(camera_ids)
    )
    return RigTrajectories(
        T_world_base=torch.eye(4),
        world_to_scene=FrameConversion(matrix=np.eye(4, dtype=np.float64)),
        rig_trajectories=[rt],
        camera_calibrations=cam_calibs,
    )


def test_rig_trajectories_post_init_accepts_consistent_calibrations():
    rt = _make_rig_trajectories(["cam_a"])
    assert "cam_a" in rt.camera_calibrations


def test_rig_trajectories_post_init_rejects_missing_camera():
    """Trajectory references a camera_id that's not in camera_calibrations."""
    with pytest.raises(AssertionError, match="Missing camera"):
        _make_rig_trajectories(
            camera_ids=["cam_a"],
            traj_camera_ids=["cam_X"],  # not in calibrations
        )


# ---------------------------------------------------------------------------
# TracksData.__post_init__
# ---------------------------------------------------------------------------


class _FakePose:
    """Fake lt.SE3 stand-in providing only the .shape used by __post_init__ +
    a ``.to(device)`` for to_device."""

    def __init__(self, n_poses):
        self._n = n_poses

    @property
    def shape(self):
        return (self._n,)

    def to(self, device):
        return self


def _make_tracks_data(n_tracks=2, n_poses_per_track=3, **overrides):
    import torch
    from instant_nurec.utils.types import TracksData

    n_total = n_tracks * n_poses_per_track
    base = dict(
        tracks_id=[f"t{i}" for i in range(n_tracks)],
        max_track_n_poses=n_poses_per_track,
        tracks_packinfo=torch.tensor(
            [[i * n_poses_per_track, n_poses_per_track] for i in range(n_tracks)],
            dtype=torch.int64,
        ),
        tracks_poses=_FakePose(n_total),
        tracks_timestamps_us=torch.zeros(n_total, dtype=torch.int64),
        tracks_flags=torch.zeros(n_tracks, dtype=torch.int32),
    )
    base.update(overrides)
    return TracksData(**base)


def test_tracks_data_post_init_accepts_valid_inputs():
    td = _make_tracks_data()
    assert td.n_tracks == 2


def test_tracks_data_post_init_rejects_packinfo_wrong_ndim():
    import torch
    with pytest.raises(ValueError, match="N_tracks, 2"):
        _make_tracks_data(tracks_packinfo=torch.zeros(2, dtype=torch.int64))


def test_tracks_data_post_init_rejects_packinfo_wrong_n_tracks():
    import torch
    with pytest.raises(ValueError, match="number of track packinfo"):
        _make_tracks_data(tracks_packinfo=torch.zeros((3, 2), dtype=torch.int64))


def test_tracks_data_post_init_rejects_packinfo_trailing_3():
    import torch
    with pytest.raises(ValueError, match="N_tracks, 2"):
        _make_tracks_data(tracks_packinfo=torch.zeros((2, 3), dtype=torch.int64))


def test_tracks_data_post_init_rejects_timestamp_count_mismatch():
    import torch
    with pytest.raises(ValueError, match="track timestamps"):
        _make_tracks_data(tracks_timestamps_us=torch.zeros(99, dtype=torch.int64))


def test_tracks_data_post_init_rejects_flags_count_mismatch():
    import torch
    with pytest.raises(ValueError, match="track flags"):
        _make_tracks_data(tracks_flags=torch.zeros(7, dtype=torch.int32))


def test_tracks_data_post_init_rejects_flags_wrong_dtype():
    import torch
    with pytest.raises(ValueError, match="torch.int32"):
        _make_tracks_data(tracks_flags=torch.zeros(2, dtype=torch.int64))


def test_tracks_data_to_device_returns_new_instance():
    import torch
    td = _make_tracks_data()
    td2 = td.to_device(torch.device("cpu"))
    assert td2 is not td
    assert td2.n_tracks == td.n_tracks


# ---------------------------------------------------------------------------
# CuboidTracksData
# ---------------------------------------------------------------------------


def test_cuboid_tracks_data_post_init_accepts_valid_inputs():
    import torch
    from instant_nurec.utils.types import CuboidTracksData

    cd = CuboidTracksData(cuboids_dims=torch.zeros(3, 3, dtype=torch.float32))
    assert cd.n_tracks == 3


def test_cuboid_tracks_data_post_init_rejects_wrong_ndim():
    import torch
    from instant_nurec.utils.types import CuboidTracksData

    with pytest.raises(ValueError, match="N_tracks, 3"):
        CuboidTracksData(cuboids_dims=torch.zeros(3, dtype=torch.float32))


def test_cuboid_tracks_data_post_init_rejects_wrong_trailing_dim():
    import torch
    from instant_nurec.utils.types import CuboidTracksData

    with pytest.raises(ValueError, match="N_tracks, 3"):
        CuboidTracksData(cuboids_dims=torch.zeros(3, 4, dtype=torch.float32))


def test_cuboid_tracks_data_post_init_rejects_wrong_dtype():
    import torch
    from instant_nurec.utils.types import CuboidTracksData

    with pytest.raises(ValueError, match="torch.float32"):
        CuboidTracksData(cuboids_dims=torch.zeros(3, 3, dtype=torch.float64))


def test_cuboid_tracks_data_to_device():
    import torch
    from instant_nurec.utils.types import CuboidTracksData

    cd = CuboidTracksData(cuboids_dims=torch.zeros(2, 3, dtype=torch.float32))
    cd2 = cd.to_device(torch.device("cpu"))
    assert cd2 is not cd


# ---------------------------------------------------------------------------
# CuboidTracksDataPack
# ---------------------------------------------------------------------------


def test_cuboid_tracks_data_pack_post_init_accepts_matching_n_tracks():
    import torch
    from instant_nurec.utils.types import CuboidTracksData, CuboidTracksDataPack

    td = _make_tracks_data(n_tracks=3, n_poses_per_track=2)
    cd = CuboidTracksData(cuboids_dims=torch.zeros(3, 3, dtype=torch.float32))
    pack = CuboidTracksDataPack(tracks_data=td, cuboidtracks_data=cd)
    assert pack.tracks_data.n_tracks == pack.cuboidtracks_data.n_tracks


def test_cuboid_tracks_data_pack_post_init_rejects_n_tracks_mismatch():
    import torch
    from instant_nurec.utils.types import CuboidTracksData, CuboidTracksDataPack

    td = _make_tracks_data(n_tracks=3, n_poses_per_track=2)
    cd = CuboidTracksData(cuboids_dims=torch.zeros(7, 3, dtype=torch.float32))
    with pytest.raises(ValueError, match="cuboid tracks"):
        CuboidTracksDataPack(tracks_data=td, cuboidtracks_data=cd)


def test_cuboid_tracks_data_pack_to_device():
    import torch
    from instant_nurec.utils.types import CuboidTracksData, CuboidTracksDataPack

    td = _make_tracks_data(n_tracks=2, n_poses_per_track=2)
    cd = CuboidTracksData(cuboids_dims=torch.zeros(2, 3, dtype=torch.float32))
    pack = CuboidTracksDataPack(tracks_data=td, cuboidtracks_data=cd)
    pack2 = pack.to_device(torch.device("cpu"))
    assert pack2 is not pack


def test_track_flags_none_is_zero():
    from instant_nurec.utils.types import TrackFlags

    assert TrackFlags.NONE.value == 0
    assert TrackFlags.DYNAMIC.value > 0
