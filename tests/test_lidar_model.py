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

"""Branch-coverage tests for instant_nurec.utils.lidar_model.

The module is six lines: an ``isinstance`` guard against ncorev4's
``SequenceLoaderV4.LidarSensor`` and a passthrough of ``model_parameters``.
We stub ``ncore.data`` / ``ncore.data.v4`` via ``sys.modules`` so the module
can be imported in a CPU-only test venv (the real ncore package is a
compiled extension we don't ship).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def stubbed_ncore(monkeypatch):
    """Install minimal ``ncore.data`` / ``ncore.data.v4`` stubs and reload
    ``instant_nurec.utils.lidar_model`` so it picks them up."""
    ncore_mod = types.ModuleType("ncore")
    data_mod = types.ModuleType("ncore.data")
    v4_mod = types.ModuleType("ncore.data.v4")

    class _LidarSensor:
        def __init__(self, model_parameters):
            self.model_parameters = model_parameters

    class _SequenceLoaderV4:
        LidarSensor = _LidarSensor

    v4_mod.SequenceLoaderV4 = _SequenceLoaderV4
    data_mod.v4 = v4_mod

    # The module also references ncore.data.LidarSensorProtocol /
    # ncore.data.ConcreteLidarModelParametersUnion in the type annotation, but
    # those are only consumed at static-typing time so plain attributes suffice.
    data_mod.LidarSensorProtocol = object
    data_mod.ConcreteLidarModelParametersUnion = object

    # Wire up attribute access (ncore.data, ncore.data.v4) so lookups don't
    # need to round-trip through sys.modules at attribute time.
    ncore_mod.data = data_mod
    monkeypatch.setitem(sys.modules, "ncore", ncore_mod)
    monkeypatch.setitem(sys.modules, "ncore.data", data_mod)
    monkeypatch.setitem(sys.modules, "ncore.data.v4", v4_mod)

    # Drop any previously-cached lidar_model so the import picks up our stubs.
    monkeypatch.delitem(sys.modules, "instant_nurec.utils.lidar_model", raising=False)

    import importlib

    return importlib.import_module("instant_nurec.utils.lidar_model"), _LidarSensor


def test_get_lidar_model_parameters_returns_attribute(stubbed_ncore):
    mod, LidarSensor = stubbed_ncore
    sensor = LidarSensor(model_parameters="some-params")
    assert mod.get_lidar_model_parameters(sensor) == "some-params"


def test_get_lidar_model_parameters_passes_through_none(stubbed_ncore):
    """A V4 sensor with model_parameters=None returns None."""
    mod, LidarSensor = stubbed_ncore
    sensor = LidarSensor(model_parameters=None)
    assert mod.get_lidar_model_parameters(sensor) is None


def test_get_lidar_model_parameters_rejects_unsupported_sensor_type(stubbed_ncore):
    """Anything that's not the V4 LidarSensor → AssertionError."""
    mod, _ = stubbed_ncore

    class _NotV4:
        model_parameters = "ignored"

    with pytest.raises(AssertionError, match="Unsupported lidar sensor type"):
        mod.get_lidar_model_parameters(_NotV4())
