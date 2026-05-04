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

"""Branch-coverage tests for ``instant_nurec.model._resolve_full_pt_path``.

The full ``make()`` body GPU-loads a real GaussiansInstantNuRecSystem so we
don't exercise it in the cpu-only test venv. The thin ``_resolve_full_pt_path``
delegator is pure Python and worth its own focused branch tests.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec import pretrained  # noqa: E402
from instant_nurec.model import _resolve_full_pt_path  # noqa: E402


def test_resolve_returns_path_when_download_succeeds(monkeypatch, tmp_path):
    fake_pt = tmp_path / "kelvin_full.pt"
    fake_pt.write_bytes(b"")
    monkeypatch.setattr(pretrained, "download_kelvin_full_pt", lambda **kw: str(fake_pt))
    assert _resolve_full_pt_path() == str(fake_pt)


def test_resolve_returns_none_when_download_raises(monkeypatch):
    def _raise(**kwargs):
        raise pretrained.PretrainedModelError("offline")

    monkeypatch.setattr(pretrained, "download_kelvin_full_pt", _raise)
    monkeypatch.delenv("INSTANT_NUREC_FULL_PT", raising=False)
    assert _resolve_full_pt_path() is None


def test_resolve_only_calls_downloader_once_per_invocation(monkeypatch):
    calls = {"n": 0}

    def _fake(**kwargs):
        calls["n"] += 1
        return "/tmp/something.pt"

    monkeypatch.setattr(pretrained, "download_kelvin_full_pt", _fake)
    _resolve_full_pt_path()
    assert calls["n"] == 1
