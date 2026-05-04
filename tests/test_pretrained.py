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
    monkeypatch.delenv("INSTANT_NUREC_HF_CACHE_DIR", raising=False)


def test_default_cache_dir_is_under_huggingface_namespace():
    parts = pretrained.DEFAULT_CACHE_DIR.parts[-5:]
    assert parts == (".cache", "huggingface", "nvidia", "instant_nurec", "kelvin")


def test_cache_dir_overridden_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "custom"))
    assert pretrained._cache_dir() == tmp_path / "custom"


def test_env_var_override_short_circuits_download(monkeypatch, tmp_path):
    """Setting INSTANT_NUREC_FULL_PT to an existing file returns it directly,
    without consulting huggingface_hub."""
    fake_pt = tmp_path / "local-kelvin.pt"
    fake_pt.write_bytes(b"")
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(fake_pt))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: pytest.fail("must not call HF when env override resolves")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    assert pretrained.download_kelvin_full_pt() == str(fake_pt)


def test_env_var_pointing_at_missing_file_falls_through(monkeypatch, tmp_path):
    """If INSTANT_NUREC_FULL_PT points nowhere, we still reach hf_hub_download."""
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(tmp_path / "nope.pt"))
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "cache"))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: f"DOWNLOADED:{kw['repo_id']}/{kw['filename']}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    out = pretrained.download_kelvin_full_pt()
    assert out == f"DOWNLOADED:{pretrained.KELVIN_REPO_ID}/{pretrained.KELVIN_FILENAME}"


def test_download_returns_path_from_hf_hub_download(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "cache"))

    captured: dict = {}

    def _fake_dl(**kw):
        captured.update(kw)
        return "/some/cached/path/kelvin_full.pt"

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = _fake_dl
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    out = pretrained.download_kelvin_full_pt()
    assert out == "/some/cached/path/kelvin_full.pt"
    assert captured["repo_id"] == pretrained.KELVIN_REPO_ID
    assert captured["filename"] == pretrained.KELVIN_FILENAME
    assert captured["cache_dir"] == str(tmp_path / "cache")


def test_explicit_cache_dir_kwarg_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "from-env"))

    captured: dict = {}

    def _fake_dl(**kw):
        captured.update(kw)
        return "/x"

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = _fake_dl
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    pretrained.download_kelvin_full_pt(cache_dir=tmp_path / "explicit")
    assert captured["cache_dir"] == str(tmp_path / "explicit")


def test_download_failure_raises_pretrained_model_error(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "cache"))

    def _fail(**kw):
        raise OSError("network down")

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = _fail
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(pretrained.PretrainedModelError, match="network down"):
        pretrained.download_kelvin_full_pt()


def test_missing_huggingface_hub_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "cache"))

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def _block_hf(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _block_hf)

    with pytest.raises(pretrained.PretrainedModelError, match="huggingface_hub is required"):
        pretrained.download_kelvin_full_pt()
