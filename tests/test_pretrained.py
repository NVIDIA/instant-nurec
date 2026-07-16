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

"""Branch-coverage tests for ``instant_nurec.pretrained``."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec import pretrained  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Each test starts with a clean env state for the resolver-relevant vars."""
    monkeypatch.delenv("INSTANT_NUREC_FULL_PT", raising=False)


def test_env_var_override_short_circuits_download(monkeypatch, tmp_path):
    """Setting INSTANT_NUREC_FULL_PT to an existing file returns it directly,
    without consulting huggingface_hub."""
    fake_pt = tmp_path / "local-instant_nurec_weights.pt"
    fake_pt.write_bytes(b"")
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(fake_pt))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: pytest.fail("must not call HF when env override resolves")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    assert pretrained.download_instant_nurec_pt() == str(fake_pt)


def test_env_var_pointing_at_missing_file_falls_through(monkeypatch, tmp_path):
    """If INSTANT_NUREC_FULL_PT points nowhere, we still reach hf_hub_download."""
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(tmp_path / "nope.pt"))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: f"DOWNLOADED:{kw['repo_id']}/{kw['filename']}"
    fake_hf.try_to_load_from_cache = lambda **kw: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    out = pretrained.download_instant_nurec_pt()
    assert out == f"DOWNLOADED:{pretrained.MODEL_REPO_ID}/{pretrained.MODEL_FILENAME}"


def test_download_forwards_explicit_cache_dir_kwarg(monkeypatch, tmp_path):
    """An explicit ``cache_dir`` is forwarded to ``hf_hub_download``."""
    captured: dict = {}

    def _fake_dl(**kw):
        captured.update(kw)
        return "/some/cached/path/instant_nurec_weights.pt"

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = _fake_dl
    fake_hf.try_to_load_from_cache = lambda **kw: None  # uncached
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    out = pretrained.download_instant_nurec_pt(cache_dir=tmp_path / "explicit")
    assert out == "/some/cached/path/instant_nurec_weights.pt"
    assert captured["repo_id"] == pretrained.MODEL_REPO_ID
    assert captured["filename"] == pretrained.MODEL_FILENAME
    assert captured["cache_dir"] == str(tmp_path / "explicit")


def test_download_without_cache_dir_omits_cache_dir_kwarg(monkeypatch):
    """With no kwarg, ``cache_dir`` is not forwarded so HF uses its default."""
    captured: dict = {}

    def _fake_dl(**kw):
        captured.update(kw)
        return "/anywhere/instant_nurec_weights.pt"

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = _fake_dl
    fake_hf.try_to_load_from_cache = lambda **kw: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    pretrained.download_instant_nurec_pt()
    assert "cache_dir" not in captured


def test_cached_copy_short_circuits_download(monkeypatch):
    """If the file is in the HF cache, return it without calling hf_hub_download."""
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.try_to_load_from_cache = lambda **kw: "/cached/instant_nurec_weights.pt"
    fake_hf.hf_hub_download = lambda **kw: pytest.fail(
        "hf_hub_download must not be called when the file is already cached"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    assert pretrained.download_instant_nurec_pt() == "/cached/instant_nurec_weights.pt"


def test_download_failure_raises_pretrained_model_error(monkeypatch):
    def _fail(**kw):
        raise OSError("network down")

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = _fail
    fake_hf.try_to_load_from_cache = lambda **kw: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(pretrained.PretrainedModelError, match="network down"):
        pretrained.download_instant_nurec_pt()


def test_missing_huggingface_hub_raises_actionable_error(monkeypatch):
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def _block_hf(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _block_hf)

    with pytest.raises(pretrained.PretrainedModelError, match="huggingface_hub is required"):
        pretrained.download_instant_nurec_pt()
