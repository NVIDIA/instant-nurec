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

"""Branch-coverage tests for ``instant_nurec.model._resolve_model_pt_path`` and
``instant_nurec.model._validate_camera_ids``.

The full ``make()`` body GPU-loads a real GaussiansInstantNuRecSystem so we
don't exercise it in the cpu-only test venv. The thin pure-Python helpers
are worth their own focused branch tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec import pretrained  # noqa: E402
from instant_nurec.model import _resolve_model_pt_path, _validate_camera_ids  # noqa: E402


def test_resolve_returns_path_when_download_succeeds(monkeypatch, tmp_path):
    fake_pt = tmp_path / "instant_nurec.pt"
    fake_pt.write_bytes(b"")
    monkeypatch.setattr(pretrained, "download_instant_nurec_pt", lambda **kw: str(fake_pt))
    assert _resolve_model_pt_path() == str(fake_pt)


def test_resolve_returns_none_when_download_raises(monkeypatch):
    def _raise(**kwargs):
        raise pretrained.PretrainedModelError("offline")

    monkeypatch.setattr(pretrained, "download_instant_nurec_pt", _raise)
    monkeypatch.delenv("INSTANT_NUREC_FULL_PT", raising=False)
    assert _resolve_model_pt_path() is None


def test_resolve_only_calls_downloader_once_per_invocation(monkeypatch):
    calls = {"n": 0}

    def _fake(**kwargs):
        calls["n"] += 1
        return "/tmp/something.pt"

    monkeypatch.setattr(pretrained, "download_instant_nurec_pt", _fake)
    _resolve_model_pt_path()
    assert calls["n"] == 1


# ---------- _validate_camera_ids ----------


_BASE_CAMERAS = ["cam_a", "cam_b", "cam_c"]


def _validate(**overrides):
    kwargs = dict(
        context_camera_ids=["cam_a"],
        supervision_camera_ids=["cam_a"],
        available_cameras=list(_BASE_CAMERAS),
        sequence_path="/seq/x.json",
    )
    kwargs.update(overrides)
    _validate_camera_ids(**kwargs)


def test_validate_camera_ids_happy_path_does_not_raise():
    _validate()


def test_validate_camera_ids_rejects_missing_context_camera():
    with pytest.raises(ValueError, match="camera_id 'cam_zzz' not found"):
        _validate(context_camera_ids=["cam_zzz"])


def test_validate_camera_ids_rejects_missing_supervision_camera():
    with pytest.raises(ValueError, match="camera_id 'cam_zzz' not found"):
        _validate(supervision_camera_ids=["cam_a", "cam_zzz"])


def test_validate_camera_ids_error_message_lists_available_cameras_sorted():
    with pytest.raises(ValueError) as exc:
        _validate(
            context_camera_ids=["cam_zzz"],
            available_cameras=["cam_b", "cam_a"],  # unsorted
        )
    # Ids in the message are sorted for readability.
    assert "['cam_a', 'cam_b']" in str(exc.value)


def test_validate_camera_ids_error_message_anchors_to_sequence_path():
    with pytest.raises(ValueError, match="/some/seq.json"):
        _validate(
            context_camera_ids=["cam_zzz"],
            sequence_path="/some/seq.json",
        )
