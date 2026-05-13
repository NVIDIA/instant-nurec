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

"""Branch-coverage tests for ``internal/scripts/dataset_load_scan.py``.

Uses synthetic dataset/config factories so we don't pay the import cost
of the real ``instant_nurec.datasets`` package and we can control every
branch of the smoke test deterministically.
"""

from __future__ import annotations

import importlib.util
import json
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def scan_mod():
    mod_name = "dataset_load_scan_module_under_test"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(REPO_ROOT / "internal" / "scripts" / "dataset_load_scan.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # dataclasses needs this for frozen field resolution
    spec.loader.exec_module(mod)
    return mod


# ---------- synthetic factories used by every test ----------


class _FakeDataset:
    """Stand-in for NCoreInstantNuRecDataset. Behavior is parameterized."""

    def __init__(
        self,
        *,
        config,
        frame_width,
        frame_height,
        n_frames_per_sample,
        length=1,
        raise_on_init=None,
        raise_on_getitem=None,
        raise_on_getitem_after=None,
    ):
        if raise_on_init is not None:
            raise raise_on_init
        self.config = config
        self._length = length
        self._raise_on_getitem = raise_on_getitem
        self._raise_after = raise_on_getitem_after
        self._items_returned = 0

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._raise_on_getitem is not None:
            raise self._raise_on_getitem
        if self._raise_after is not None and self._items_returned >= self._raise_after:
            raise RuntimeError("after-N failure")
        self._items_returned += 1
        return SimpleNamespace(idx=idx)


def _make_factories(
    *,
    length=1,
    raise_on_init=None,
    raise_on_getitem=None,
    raise_on_config=None,
    raise_on_getitem_after=None,
):
    def config_factory(**kwargs):
        if raise_on_config is not None:
            raise raise_on_config
        return SimpleNamespace(**kwargs)

    def dataset_factory(**kwargs):
        return _FakeDataset(
            length=length,
            raise_on_init=raise_on_init,
            raise_on_getitem=raise_on_getitem,
            raise_on_getitem_after=raise_on_getitem_after,
            **kwargs,
        )

    return dataset_factory, config_factory


def _stub_json(parent: Path, uuid: str) -> Path:
    d = parent / uuid
    d.mkdir()
    p = d / f"pai_{uuid}.json"
    p.write_text("{}")  # contents don't matter; factories are stubbed
    return p


# ---------- iter_clip_jsons ----------


def test_iter_clip_jsons_orders_and_filters(scan_mod, tmp_path):
    _stub_json(tmp_path, "b")
    _stub_json(tmp_path, "a")
    (tmp_path / "loose.json").write_text("{}")
    sub = tmp_path / "no-pai"
    sub.mkdir()
    (sub / "x.json").write_text("{}")
    files = list(scan_mod.iter_clip_jsons(tmp_path))
    assert [f.parent.name for f in files] == ["a", "b"]


def test_iter_clip_jsons_empty(scan_mod, tmp_path):
    assert list(scan_mod.iter_clip_jsons(tmp_path)) == []


# ---------- smoke_test_one_clip ----------


def _run(scan_mod, json_path, *, dataset_factory, config_factory, items_per_clip=1):
    return scan_mod.smoke_test_one_clip(
        json_path,
        camera_id="cam",
        max_chunks=8,
        frame_width=784,
        frame_height=448,
        n_frames_per_sample=18,
        items_per_clip=items_per_clip,
        dataset_factory=dataset_factory,
        config_factory=config_factory,
    )


def test_smoke_happy_path(scan_mod, tmp_path):
    p = _stub_json(tmp_path, "ok")
    df, cf = _make_factories(length=4)
    v = _run(scan_mod, p, dataset_factory=df, config_factory=cf)
    assert v.passed is True
    assert v.stage == "ok"
    assert v.exception_type is None
    assert v.n_items_fetched == 1
    assert v.elapsed_s >= 0.0


def test_smoke_config_factory_raises(scan_mod, tmp_path):
    p = _stub_json(tmp_path, "bad-config")
    df, cf = _make_factories(raise_on_config=ValueError("config blew up"))
    v = _run(scan_mod, p, dataset_factory=df, config_factory=cf)
    assert v.passed is False
    assert v.stage == "config"
    assert v.exception_type == "ValueError"
    assert "config blew up" in v.exception_message
    assert v.traceback_tail  # not empty


def test_smoke_init_raises(scan_mod, tmp_path):
    p = _stub_json(tmp_path, "bad-init")
    df, cf = _make_factories(raise_on_init=KeyError("missing thing"))
    v = _run(scan_mod, p, dataset_factory=df, config_factory=cf)
    assert v.passed is False
    assert v.stage == "init"
    assert v.exception_type == "KeyError"
    assert v.n_items_fetched == 0


def test_smoke_getitem_raises_first(scan_mod, tmp_path):
    p = _stub_json(tmp_path, "bad-get")
    df, cf = _make_factories(raise_on_getitem=RuntimeError("oops"))
    v = _run(scan_mod, p, dataset_factory=df, config_factory=cf)
    assert v.passed is False
    assert v.stage == "getitem"
    assert v.exception_type == "RuntimeError"
    assert v.n_items_fetched == 0


def test_smoke_getitem_raises_after_first(scan_mod, tmp_path):
    p = _stub_json(tmp_path, "partial")
    df, cf = _make_factories(length=3, raise_on_getitem_after=2)
    v = _run(scan_mod, p, dataset_factory=df, config_factory=cf, items_per_clip=3)
    assert v.passed is False
    assert v.stage == "getitem"
    assert v.n_items_fetched == 2


def test_smoke_items_per_clip_capped_by_len(scan_mod, tmp_path):
    """items_per_clip larger than len(ds) is silently clipped."""
    p = _stub_json(tmp_path, "short")
    df, cf = _make_factories(length=2)
    v = _run(scan_mod, p, dataset_factory=df, config_factory=cf, items_per_clip=10)
    assert v.passed is True
    assert v.n_items_fetched == 2


def test_smoke_exception_message_truncated(scan_mod, tmp_path):
    p = _stub_json(tmp_path, "long")
    df, cf = _make_factories(raise_on_init=RuntimeError("x" * 1000))
    v = _run(scan_mod, p, dataset_factory=df, config_factory=cf)
    assert len(v.exception_message) == 500


# ---------- _format_traceback_tail ----------


def test_format_traceback_tail_returns_last_n(scan_mod):
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        tail = scan_mod._format_traceback_tail(exc, n_lines=3)
    assert isinstance(tail, list)
    assert 1 <= len(tail) <= 3
    assert any("RuntimeError" in line for line in tail)


def test_format_traceback_tail_default_5(scan_mod):
    try:
        raise ValueError("vv")
    except ValueError as exc:
        tail = scan_mod._format_traceback_tail(exc)
    assert len(tail) <= 5


# ---------- build_report ----------


def _verdict(scan_mod, **kwargs):
    defaults = {
        "uuid": "u",
        "json_path": Path("/x"),
        "passed": True,
        "stage": "ok",
        "exception_type": None,
        "exception_message": "",
        "traceback_tail": [],
        "n_items_fetched": 1,
        "elapsed_s": 0.5,
    }
    defaults.update(kwargs)
    return scan_mod.ClipVerdict(**defaults)


def test_build_report_all_pass(scan_mod, tmp_path):
    v = [
        _verdict(scan_mod, uuid="a"),
        _verdict(scan_mod, uuid="b"),
    ]
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        verdicts=v,
        camera_id="cam",
        max_chunks=8,
        items_per_clip=1,
    )
    assert rep["summary"]["n_passed"] == 2
    assert rep["summary"]["n_failed"] == 0
    assert rep["summary"]["failures_by_stage"] == {}
    assert rep["summary"]["failures_by_exception"] == {}
    assert rep["failed"] == []
    assert rep["passed_uuids"] == ["a", "b"]


def test_build_report_mixed(scan_mod, tmp_path):
    v = [
        _verdict(scan_mod, uuid="a"),
        _verdict(
            scan_mod,
            uuid="b",
            passed=False,
            stage="getitem",
            exception_type="RuntimeError",
            exception_message="boom",
        ),
        _verdict(
            scan_mod,
            uuid="c",
            passed=False,
            stage="init",
            exception_type="KeyError",
            exception_message="nope",
        ),
        _verdict(
            scan_mod,
            uuid="d",
            passed=False,
            stage="getitem",
            exception_type="RuntimeError",
            exception_message="boom2",
        ),
    ]
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        verdicts=v,
        camera_id="cam",
        max_chunks=8,
        items_per_clip=1,
    )
    assert rep["summary"]["n_passed"] == 1
    assert rep["summary"]["n_failed"] == 3
    assert rep["summary"]["failures_by_stage"] == {"getitem": 2, "init": 1}
    assert rep["summary"]["failures_by_exception"] == {"RuntimeError": 2, "KeyError": 1}
    assert rep["passed_uuids"] == ["a"]
    assert {f["uuid"] for f in rep["failed"]} == {"b", "c", "d"}


def test_build_report_failure_without_exception_type(scan_mod, tmp_path):
    """A verdict failed but exception_type=None should still be tallied by stage."""
    v = [_verdict(scan_mod, uuid="x", passed=False, stage="init", exception_type=None)]
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        verdicts=v,
        camera_id="cam",
        max_chunks=8,
        items_per_clip=1,
    )
    assert rep["summary"]["failures_by_stage"] == {"init": 1}
    assert rep["summary"]["failures_by_exception"] == {}


# ---------- _print_progress ----------


def test_print_progress_ok(scan_mod, capsys):
    v = _verdict(scan_mod, uuid="u-a")
    scan_mod._print_progress(1, 10, v)
    out = capsys.readouterr().out
    assert "1/   10" in out
    assert "u-a" in out
    assert "ok" in out


def test_print_progress_fail(scan_mod, capsys):
    v = _verdict(scan_mod, uuid="u-b", passed=False, stage="getitem")
    scan_mod._print_progress(2, 10, v)
    out = capsys.readouterr().out
    assert "FAIL@getitem" in out


# ---------- main ----------


def test_main_clips_dir_missing(scan_mod, tmp_path):
    rc = scan_mod.main(
        [
            "--clips-dir",
            str(tmp_path / "no-such-dir"),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2


def test_main_empty_clips_dir(scan_mod, tmp_path):
    cdir = tmp_path / "cdir"
    cdir.mkdir()
    rc = scan_mod.main(
        ["--clips-dir", str(cdir), "--output", str(tmp_path / "out.json")]
    )
    assert rc == 1


def test_main_runs_with_injected_factories(scan_mod, monkeypatch, tmp_path):
    """Drive main() end-to-end with stubbed factories so we don't touch the
    real dataset import."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _stub_json(clips_dir, "ua")
    _stub_json(clips_dir, "ub")
    df, cf = _make_factories(length=1)
    monkeypatch.setattr(scan_mod, "_default_dataset_factory", lambda: df)
    monkeypatch.setattr(scan_mod, "_default_config_factory", lambda: cf)

    out = tmp_path / "out.json"
    rc = scan_mod.main(
        [
            "--clips-dir",
            str(clips_dir),
            "--output",
            str(out),
            "--camera-id",
            "cam",
            "--items-per-clip",
            "1",
        ]
    )
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["summary"]["n_total"] == 2
    assert rep["summary"]["n_passed"] == 2
    assert rep["config"]["camera_id"] == "cam"


def test_main_records_failures(scan_mod, monkeypatch, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _stub_json(clips_dir, "good")
    _stub_json(clips_dir, "bad")

    df_bad, cf = _make_factories(raise_on_init=ValueError("nope"))
    df_good, _ = _make_factories(length=1)

    def df(*, config, **kwargs):
        # Use the first __init__ argument to pick which dataset
        ncore_paths = config.ncore_json_paths
        if "bad" in ncore_paths[0]:
            return df_bad(config=config, **kwargs)
        return df_good(config=config, **kwargs)

    monkeypatch.setattr(scan_mod, "_default_dataset_factory", lambda: df)
    monkeypatch.setattr(scan_mod, "_default_config_factory", lambda: cf)

    out = tmp_path / "out.json"
    rc = scan_mod.main(
        [
            "--clips-dir",
            str(clips_dir),
            "--output",
            str(out),
            "--camera-id",
            "cam",
        ]
    )
    assert rc == 1
    rep = json.loads(out.read_text())
    assert rep["summary"]["n_passed"] == 1
    assert rep["summary"]["n_failed"] == 1
    assert rep["failed"][0]["uuid"] == "bad"


def test_main_max_clips_truncates(scan_mod, monkeypatch, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for i in range(5):
        _stub_json(clips_dir, f"u-{i}")
    df, cf = _make_factories(length=1)
    monkeypatch.setattr(scan_mod, "_default_dataset_factory", lambda: df)
    monkeypatch.setattr(scan_mod, "_default_config_factory", lambda: cf)
    out = tmp_path / "out.json"
    rc = scan_mod.main(
        [
            "--clips-dir",
            str(clips_dir),
            "--output",
            str(out),
            "--max-clips",
            "2",
        ]
    )
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["summary"]["n_total"] == 2
