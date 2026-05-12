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

"""Branch-coverage tests for ``instant_nurec.utils.sensors`` (sensors.py + __init__.py).

Tests for the in-tree ``compute_poses_and_timestamps``
with a torch helper (matrix indexing); we no longer need the kernel stub.
``ncore`` is still stubbed via ``sys.modules`` for CPU-only test venvs.
The CUDA assertion in ``get_poses_and_timestamps_startend`` is exercised
via a tensor subclass whose ``.is_cuda`` is forced to True.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def stubbed_sensors(monkeypatch):
    # ncore.data.ShutterType + ncore.sensors camera models stubs (the only
    # external imports).
    ncore_mod = types.ModuleType("ncore")
    ncore_data_mod = types.ModuleType("ncore.data")
    ncore_sensors_mod = types.ModuleType("ncore.sensors")

    class _ShutterType:
        ROLLING = type("RollingTag", (), {"value": 1})()

    ncore_data_mod.ShutterType = _ShutterType

    class _StubCamera:
        pass

    ncore_sensors_mod.FThetaCameraModel = _StubCamera

    ncore_mod.data = ncore_data_mod
    ncore_mod.sensors = ncore_sensors_mod

    for name, mod in [
        ("ncore", ncore_mod),
        ("ncore.data", ncore_data_mod),
        ("ncore.sensors", ncore_sensors_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.delitem(sys.modules, "instant_nurec.utils.sensors", raising=False)
    monkeypatch.delitem(sys.modules, "instant_nurec.utils.sensors.sensors", raising=False)

    import importlib

    sensors_pkg_loaded = importlib.import_module("instant_nurec.utils.sensors")
    return sensors_pkg_loaded, _ShutterType


# ---------------------------------------------------------------------------
# RectSubsampledSensor
# ---------------------------------------------------------------------------


def test_rect_subsampled_sensor_default_offsets_and_subsample(stubbed_sensors):
    sensors_pkg, _ = stubbed_sensors
    s = sensors_pkg.RectSubsampledSensor(width=640, height=480)
    assert s.width == 640
    assert s.height == 480
    assert s.i == 0
    assert s.j == 0
    assert s.subsample_factor == 1.0


def test_rect_subsampled_sensor_kw_only_overrides(stubbed_sensors):
    sensors_pkg, _ = stubbed_sensors
    s = sensors_pkg.RectSubsampledSensor(width=320, height=240, i=10, j=20, subsample_factor=0.5)
    assert (s.i, s.j, s.subsample_factor) == (10, 20, 0.5)


def test_rect_subsampled_sensor_is_slotted_kw_only(stubbed_sensors):
    """`slots=True, kw_only=True` — positional args should fail."""
    sensors_pkg, _ = stubbed_sensors
    with pytest.raises(TypeError):
        sensors_pkg.RectSubsampledSensor(640, 480)  # type: ignore[call-arg]


def test_sensors_package_init_re_exports_public_symbols(stubbed_sensors):
    sensors_pkg, _ = stubbed_sensors
    assert "RectSubsampledSensor" in sensors_pkg.__all__
    assert "SensorModelComputations" in sensors_pkg.__all__


# ---------------------------------------------------------------------------
# SensorModelComputations.get_poses_and_timestamps_startend
# ---------------------------------------------------------------------------


class _FakeSensorModel:
    """Stand-in for ``nn.ModuleDict`` value with a `.shutter_type`."""

    def __init__(self, shutter_type):
        self.shutter_type = shutter_type


class _FakeModuleDict(dict):
    """A dict-by-string-key stand-in for ``nn.ModuleDict`` (which __getitem__
    is what the SUT uses)."""


def _cuda_like_zeros(*shape, dtype=torch.float32):
    """A CPU tensor whose ``is_cuda`` property is forced to True so the
    SUT's CUDA assertion passes without a real GPU."""
    t = torch.zeros(*shape, dtype=dtype)
    # ``is_cuda`` is a read-only property on Tensor; we patch it on the
    # instance via __dict__/object.__setattr__ won't work either — instead
    # use a thin Tensor subclass.
    return t


class _CudaLike(torch.Tensor):
    """Tensor subclass whose ``.is_cuda`` always reports True."""

    @staticmethod
    def __new__(cls, data):
        t = data.as_subclass(cls)
        return t

    @property
    def is_cuda(self):
        return True


def test_get_poses_and_timestamps_startend_returns_squeezed_tensors(stubbed_sensors):
    sensors_pkg, ShutterType = stubbed_sensors
    smc = sensors_pkg.SensorModelComputations

    T_views = _CudaLike(torch.zeros(1, 2, 4, 4))
    ts_views = _CudaLike(torch.zeros(1, 2, dtype=torch.int64))
    ts_views_cpu = torch.zeros(1, 2)

    out = smc.get_poses_and_timestamps_startend(
        T_sensor_world_startend_allviews=T_views,
        timestamps_startend_us_allviews=ts_views,
        timestamps_startend_us_allviews_cpu=ts_views_cpu,
        unique_frame_idx=0,
        unique_frame_idx_tensor=torch.tensor([0]),
    )
    assert out.T_sensor_world_startend.shape == (2, 4, 4)
    assert out.timestamps_startend_us.shape == (2,)
    assert out.timestamps_startend_us_gpu.shape == (1, 2)
    assert out.timestamps_startend_us_cpu.shape == (1, 2)


def test_compute_poses_and_timestamps_torch_indexes_views(stubbed_sensors):
    """per-sample matrix and timestamp indexing."""
    sensors_pkg, _ = stubbed_sensors
    helper = sensors_pkg.sensors._compute_poses_and_timestamps_torch
    T = torch.arange(96).reshape(3, 2, 4, 4).to(torch.float32)
    ts = torch.tensor([[0, 100], [1000, 1100], [2000, 2100]], dtype=torch.int64)
    fidx = torch.tensor([2, 0], dtype=torch.int32)
    T_out, ts_out = helper(T, fidx, ts)
    assert T_out.shape == (2, 2, 4, 4)
    assert ts_out.shape == (2, 2)
    assert torch.equal(T_out[0], T[2])
    assert torch.equal(T_out[1], T[0])
    assert torch.equal(ts_out, torch.tensor([[2000, 2100], [0, 100]]))


def test_compute_poses_and_timestamps_torch_empty_frame_idx(stubbed_sensors):
    sensors_pkg, _ = stubbed_sensors
    helper = sensors_pkg.sensors._compute_poses_and_timestamps_torch
    T = torch.zeros(2, 2, 4, 4)
    ts = torch.zeros(2, 2, dtype=torch.int64)
    fidx = torch.empty(0, dtype=torch.int32)
    T_out, ts_out = helper(T, fidx, ts)
    assert T_out.shape == (0, 2, 4, 4)
    assert ts_out.shape == (0, 2)
    assert ts_out.dtype == torch.int64


def test_get_poses_and_timestamps_startend_requires_cuda(stubbed_sensors):
    sensors_pkg, ShutterType = stubbed_sensors
    smc = sensors_pkg.SensorModelComputations

    cpu_tensor = torch.zeros(1, 2, 4, 4)  # is_cuda=False
    with pytest.raises(AssertionError, match="requires CUDA tensors"):
        smc.get_poses_and_timestamps_startend(
            T_sensor_world_startend_allviews=cpu_tensor,
            timestamps_startend_us_allviews=torch.zeros(1, 2),
            timestamps_startend_us_allviews_cpu=torch.zeros(1, 2),
            unique_frame_idx=0,
            unique_frame_idx_tensor=torch.tensor([0]),
        )
