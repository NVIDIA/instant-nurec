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

The mock stands in for ``huggingface_hub.snapshot_download`` /
``hf_hub_download`` until the corp publishes the real
``nvidia/instant-nurec-kelvin`` repo. We exercise:

  * default-on env-var toggle (``INSTANT_NUREC_HF_MOCK``)
  * placeholder-repo-id resolution
  * cache seeding from ``INSTANT_NUREC_FULL_PT``
  * file-missing error path
  * env-var-off forwards to the real ``huggingface_hub`` import
  * convenience helpers ``get_full_model_path`` / ``get_sample_data_path``
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
    """Each test starts with a clean env state for the mock-relevant vars."""
    monkeypatch.delenv("INSTANT_NUREC_HF_MOCK", raising=False)
    monkeypatch.delenv("INSTANT_NUREC_FULL_PT", raising=False)
    monkeypatch.delenv("INSTANT_NUREC_HF_CACHE_DIR", raising=False)


# ---------------------------------------------------------------------------
# Mock toggle
# ---------------------------------------------------------------------------


def test_is_mock_enabled_default_true():
    assert _hf_mock._is_mock_enabled() is True


def test_is_mock_enabled_off_when_env_zero(monkeypatch):
    monkeypatch.setenv("INSTANT_NUREC_HF_MOCK", "0")
    assert _hf_mock._is_mock_enabled() is False


def test_is_mock_enabled_on_when_env_one(monkeypatch):
    monkeypatch.setenv("INSTANT_NUREC_HF_MOCK", "1")
    assert _hf_mock._is_mock_enabled() is True


# ---------------------------------------------------------------------------
# Cache directory resolution
# ---------------------------------------------------------------------------


def test_cache_dir_defaults_to_user_home():
    cache_dir = _hf_mock._cache_dir()
    assert cache_dir == _hf_mock.DEFAULT_CACHE_DIR
    # Default is under ~/.cache/instant_nurec.
    assert cache_dir.parts[-2:] == (".cache", "instant_nurec")


def test_cache_dir_overridden_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "custom"))
    assert _hf_mock._cache_dir() == tmp_path / "custom"


# ---------------------------------------------------------------------------
# snapshot_download — placeholder repo
# ---------------------------------------------------------------------------


def test_snapshot_download_placeholder_creates_cache_dir(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))
    out = _hf_mock.snapshot_download(_hf_mock.PLACEHOLDER_REPO_ID)
    assert Path(out) == cache
    assert cache.exists()


def test_snapshot_download_explicit_cache_dir_kwarg(monkeypatch, tmp_path):
    cache = tmp_path / "explicit"
    out = _hf_mock.snapshot_download(_hf_mock.PLACEHOLDER_REPO_ID, cache_dir=cache)
    assert Path(out) == cache
    assert cache.exists()


def test_snapshot_download_unknown_repo_raises(monkeypatch):
    with pytest.raises(_hf_mock.HFMockError, match="only knows"):
        _hf_mock.snapshot_download("some/other-repo")


def test_snapshot_download_seeds_cache_from_full_pt(monkeypatch, tmp_path):
    """If INSTANT_NUREC_FULL_PT points at an existing file, the mock copies
    it into the cache as ``kelvin_full.pt``."""
    src = tmp_path / "elsewhere" / "kelvin_full.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"fake-pickle-bytes")

    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(src))
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    out = _hf_mock.snapshot_download(_hf_mock.PLACEHOLDER_REPO_ID)
    cached = Path(out) / "kelvin_full.pt"
    assert cached.exists()
    assert cached.read_bytes() == b"fake-pickle-bytes"


def test_snapshot_download_skips_seed_when_full_pt_missing(monkeypatch, tmp_path):
    """If INSTANT_NUREC_FULL_PT points somewhere that doesn't exist, the
    mock silently skips the copy step (file just won't be in the cache)."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", "/nope/missing.pt")
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    out = _hf_mock.snapshot_download(_hf_mock.PLACEHOLDER_REPO_ID)
    assert Path(out) == cache
    assert not (cache / "kelvin_full.pt").exists()


def test_snapshot_download_does_not_re_copy_when_already_cached(
    monkeypatch, tmp_path
):
    """A second snapshot_download call with the same source file size
    should leave the cached copy alone (not re-copy)."""
    src = tmp_path / "elsewhere" / "kelvin_full.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x" * 32)

    cache = tmp_path / "cache"
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(src))
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    _hf_mock.snapshot_download(_hf_mock.PLACEHOLDER_REPO_ID)
    cached = cache / "kelvin_full.pt"
    first_mtime = cached.stat().st_mtime_ns

    _hf_mock.snapshot_download(_hf_mock.PLACEHOLDER_REPO_ID)
    second_mtime = cached.stat().st_mtime_ns
    assert second_mtime == first_mtime  # not re-copied


def test_snapshot_download_re_copies_when_size_mismatches(monkeypatch, tmp_path):
    """If the cache holds a stale file (different size), it gets refreshed."""
    src = tmp_path / "elsewhere" / "kelvin_full.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x" * 32)

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kelvin_full.pt").write_bytes(b"stale")

    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(src))
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    _hf_mock.snapshot_download(_hf_mock.PLACEHOLDER_REPO_ID)
    assert (cache / "kelvin_full.pt").read_bytes() == b"x" * 32


# ---------------------------------------------------------------------------
# snapshot_download — env-off forwards to real huggingface_hub
# ---------------------------------------------------------------------------


def test_snapshot_download_env_off_forwards_to_real_hf(monkeypatch):
    """When the env var is `0`, the call must go through to
    ``huggingface_hub.snapshot_download``. We stub the real module at
    sys.modules level."""
    monkeypatch.setenv("INSTANT_NUREC_HF_MOCK", "0")

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda repo_id, **kw: f"REAL:{repo_id}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    out = _hf_mock.snapshot_download("some/other-repo")
    assert out == "REAL:some/other-repo"


# ---------------------------------------------------------------------------
# hf_hub_download
# ---------------------------------------------------------------------------


def test_hf_hub_download_returns_existing_file(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kelvin_full.pt").write_bytes(b"some-bytes")
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    out = _hf_mock.hf_hub_download(_hf_mock.PLACEHOLDER_REPO_ID, "kelvin_full.pt")
    assert Path(out) == cache / "kelvin_full.pt"


def test_hf_hub_download_raises_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(_hf_mock.HFMockError, match="not found"):
        _hf_mock.hf_hub_download(_hf_mock.PLACEHOLDER_REPO_ID, "kelvin_full.pt")


def test_hf_hub_download_env_off_forwards_to_real(monkeypatch):
    monkeypatch.setenv("INSTANT_NUREC_HF_MOCK", "0")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda repo_id, filename, **kw: f"REAL:{repo_id}/{filename}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    out = _hf_mock.hf_hub_download("a/b", "f.pt")
    assert out == "REAL:a/b/f.pt"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def test_get_full_model_path(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kelvin_full.pt").write_bytes(b"x")
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    out = _hf_mock.get_full_model_path()
    assert Path(out) == cache / "kelvin_full.pt"


def test_get_sample_data_path_returns_dir_when_present(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    sample = cache / "ncorev4_sample"
    sample.mkdir(parents=True)
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    out = _hf_mock.get_sample_data_path()
    assert Path(out) == sample


def test_get_sample_data_path_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(_hf_mock.HFMockError, match="sample data"):
        _hf_mock.get_sample_data_path()


def test_get_sample_data_path_raises_when_path_is_a_file(monkeypatch, tmp_path):
    """If ``ncorev4_sample`` exists but is a file (not a dir), the mock
    rejects it explicitly rather than silently returning the file path."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ncorev4_sample").write_text("not a directory")
    monkeypatch.setenv("INSTANT_NUREC_HF_CACHE_DIR", str(cache))

    with pytest.raises(_hf_mock.HFMockError, match="sample data"):
        _hf_mock.get_sample_data_path()
