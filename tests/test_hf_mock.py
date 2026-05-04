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

"""Branch-coverage tests for instant_nurec._hf_mock.

The resolver tries (1) ``INSTANT_NUREC_FULL_PT`` env override → cache
copy, (2) cached copy, (3) ``huggingface_hub.hf_hub_download``. We
exercise each branch.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec import _hf_mock  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Each test starts with a clean env state for the resolver-relevant vars."""
    monkeypatch.delenv("INSTANT_NUREC_FULL_PT", raising=False)
    monkeypatch.delenv("INSTANT_NUREC_HF_CACHE_DIR", raising=False)


# ---------------------------------------------------------------------------
# Cache directory resolution
# ---------------------------------------------------------------------------


def test_cache_dir_defaults_to_user_home():
    cache_dir = _hf_mock._cache_dir()
    assert cache_dir == _hf_mock.DEFAULT_CACHE_DIR
    assert cache_dir.parts[-2:] == (".cache", "instant_nurec")


def test_cache_dir_overridden_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "custom"))
    assert _hf_mock._cache_dir() == tmp_path / "custom"


# ---------------------------------------------------------------------------
# Branch 1: INSTANT_NUREC_FULL_PT seeds the cache
# ---------------------------------------------------------------------------


def test_env_var_seeds_cache_when_pointing_at_existing_file(monkeypatch, tmp_path):
    src = tmp_path / "elsewhere" / "kelvin_full.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"fake-pickle-bytes")

    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(src))
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    out = _hf_mock.get_full_model_path()
    assert Path(out) == cache / "kelvin_full.pt"
    assert (cache / "kelvin_full.pt").read_bytes() == b"fake-pickle-bytes"


def test_env_var_pointing_at_missing_path_falls_through(monkeypatch, tmp_path):
    """If the env var points at a non-existent path, the resolver moves on
    to the cache / HF download branches (rather than failing immediately)."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kelvin_full.pt").write_bytes(b"already-cached")
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(tmp_path / "nope.pt"))
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    out = _hf_mock.get_full_model_path()
    assert Path(out) == cache / "kelvin_full.pt"


def test_env_var_skips_recopy_when_already_cached(monkeypatch, tmp_path):
    src = tmp_path / "elsewhere" / "kelvin_full.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x" * 32)

    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(src))
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    _hf_mock.get_full_model_path()
    cached = cache / "kelvin_full.pt"
    first_mtime = cached.stat().st_mtime_ns

    _hf_mock.get_full_model_path()
    assert cached.stat().st_mtime_ns == first_mtime


def test_env_var_recopies_when_size_mismatches(monkeypatch, tmp_path):
    src = tmp_path / "elsewhere" / "kelvin_full.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x" * 32)

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kelvin_full.pt").write_bytes(b"stale")
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(src))
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    _hf_mock.get_full_model_path()
    assert (cache / "kelvin_full.pt").read_bytes() == b"x" * 32


# ---------------------------------------------------------------------------
# Branch 2: cached copy
# ---------------------------------------------------------------------------


def test_cached_copy_returned_when_present(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kelvin_full.pt").write_bytes(b"some-bytes")
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    out = _hf_mock.get_full_model_path()
    assert Path(out) == cache / "kelvin_full.pt"


# ---------------------------------------------------------------------------
# Branch 3: huggingface_hub download
# ---------------------------------------------------------------------------


def test_auto_download_called_when_cache_empty(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kw: f"DOWNLOADED:{kw['repo_id']}/{kw['filename']}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    out = _hf_mock.get_full_model_path()
    assert out == f"DOWNLOADED:{_hf_mock.PLACEHOLDER_REPO_ID}/kelvin_full.pt"


def test_auto_download_failure_raises_hfmock_error(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    fake_hf = types.ModuleType("huggingface_hub")

    def _fail(**kw):
        raise OSError("network down")

    fake_hf.hf_hub_download = _fail
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    with pytest.raises(_hf_mock.HFMockError, match="network down"):
        _hf_mock.get_full_model_path()


def test_missing_huggingface_hub_raises_actionable_error(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _block_hf(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _block_hf)

    with pytest.raises(_hf_mock.HFMockError, match="huggingface_hub is required"):
        _hf_mock.get_full_model_path()


def test_explicit_cache_dir_kwarg_overrides_env(monkeypatch, tmp_path):
    other = tmp_path / "other-cache"
    other.mkdir()
    (other / "kelvin_full.pt").write_bytes(b"y")
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "different"))

    out = _hf_mock.get_full_model_path(cache_dir=other)
    assert Path(out) == other / "kelvin_full.pt"
