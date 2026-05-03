"""Branch-coverage tests for ``instant_nurec.nrm.systems._resolve_full_pt_path``.

The full ``make()`` body GPU-loads a real GaussiansNRMSystem so we don't
exercise it in the cpu-only test venv. The new ``_resolve_full_pt_path``
helper that wires the HF mock is pure Python and worth its own focused
branch tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec import _hf_mock  # noqa: E402
from instant_nurec.nrm.systems import _resolve_full_pt_path  # noqa: E402


def test_resolve_returns_hf_mock_path_when_mock_succeeds(monkeypatch, tmp_path):
    """Happy path: the HF mock resolves the .pt; resolver returns its path."""
    fake_pt = tmp_path / "kelvin_full.pt"
    fake_pt.write_bytes(b"")  # presence is what matters here

    def _fake_get(**kwargs):
        return str(fake_pt)

    monkeypatch.setattr(_hf_mock, "get_full_model_path", _fake_get)
    # Env var is irrelevant when the mock succeeds.
    monkeypatch.delenv("INSTANT_NUREC_FULL_PT", raising=False)

    assert _resolve_full_pt_path() == str(fake_pt)


def test_resolve_falls_back_to_env_var_when_mock_raises(monkeypatch, tmp_path):
    """HF mock raises HFMockError → resolver returns INSTANT_NUREC_FULL_PT."""

    def _raise(**kwargs):
        raise _hf_mock.HFMockError("not in cache")

    monkeypatch.setattr(_hf_mock, "get_full_model_path", _raise)
    fallback = tmp_path / "fallback.pt"
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", str(fallback))

    assert _resolve_full_pt_path() == str(fallback)


def test_resolve_returns_none_when_mock_raises_and_env_unset(monkeypatch):
    """No HF cache + no env var → None (tells make() to take the slow path)."""

    def _raise(**kwargs):
        raise _hf_mock.HFMockError("not in cache")

    monkeypatch.setattr(_hf_mock, "get_full_model_path", _raise)
    monkeypatch.delenv("INSTANT_NUREC_FULL_PT", raising=False)

    assert _resolve_full_pt_path() is None


def test_resolve_returns_none_when_mock_raises_and_env_is_empty_string(monkeypatch):
    """Empty env var should be treated the same as unset."""

    def _raise(**kwargs):
        raise _hf_mock.HFMockError("not in cache")

    monkeypatch.setattr(_hf_mock, "get_full_model_path", _raise)
    monkeypatch.setenv("INSTANT_NUREC_FULL_PT", "")

    assert _resolve_full_pt_path() is None


def test_resolve_only_calls_mock_once_per_invocation(monkeypatch):
    """The resolver shouldn't re-enter the mock if it already returned a path."""
    calls = {"n": 0}

    def _fake_get(**kwargs):
        calls["n"] += 1
        return "/tmp/something.pt"

    monkeypatch.setattr(_hf_mock, "get_full_model_path", _fake_get)
    _resolve_full_pt_path()
    assert calls["n"] == 1
