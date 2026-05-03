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

"""Tests for benchmark/validate_parity.py.

The parity tool is the contract that gates every commit on this branch, so the
suite verifies branch coverage of every code path: identical files, count
mismatch, schema mismatch, dtype mismatch, per-property tolerance breach (above
and below the bar), default tolerance fallback, file-pairing under different
naming conventions, and missing-input errors.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "benchmark" / "validate_parity.py"

BASELINE_SCHEMA: list[tuple[str, str]] = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
    ("opacity", "f4"),
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ("road_mask", "u1"),
    ("sky_mask", "f4"),
]


def _make_ply(
    path: Path,
    n: int = 16,
    *,
    schema: list[tuple[str, str]] = BASELINE_SCHEMA,
    seed: int = 0,
    perturb: dict[str, float] | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    arr = np.zeros(n, dtype=schema)
    for name, dt in schema:
        if dt.startswith("f"):
            arr[name] = rng.standard_normal(n).astype(dt)
        else:
            arr[name] = rng.integers(0, 255, n, endpoint=False).astype(dt)
    if perturb:
        for name, delta in perturb.items():
            arr[name] = (arr[name].astype(np.float64) + delta).astype(arr[name].dtype)
    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))


def _empty_tol(tmp: Path) -> Path:
    p = tmp / "_empty_tol.json"
    p.write_text("{}")
    return p


def _run(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
    )


# ---------- merge mode ----------


def test_merge_identical_files_pass(tmp_path: Path) -> None:
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _make_ply(a, 32, seed=1)
    _make_ply(b, 32, seed=1)
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "merge", a, b)
    assert proc.returncode == 0, proc.stderr


def test_merge_missing_baseline_fail(tmp_path: Path) -> None:
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "merge", tmp_path / "nope.ply", tmp_path / "x.ply")
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr


def test_merge_missing_proposed_fail(tmp_path: Path) -> None:
    a = tmp_path / "a.ply"
    _make_ply(a, 8)
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "merge", a, tmp_path / "nope.ply")
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr


def test_merge_vertex_count_mismatch_fail(tmp_path: Path) -> None:
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _make_ply(a, 32, seed=2)
    _make_ply(b, 31, seed=2)
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "merge", a, b)
    assert proc.returncode == 1
    assert "vertex count mismatch" in proc.stderr


def test_merge_property_name_mismatch_fail(tmp_path: Path) -> None:
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    alt_schema = [(name if name != "sky_mask" else "skies", dt) for name, dt in BASELINE_SCHEMA]
    _make_ply(a, 16, seed=3)
    _make_ply(b, 16, seed=3, schema=alt_schema)
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "merge", a, b)
    assert proc.returncode == 1
    assert "property name mismatch" in proc.stderr


def test_merge_dtype_mismatch_fail(tmp_path: Path) -> None:
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    alt_schema = [(name, "f8" if name == "x" else dt) for name, dt in BASELINE_SCHEMA]
    _make_ply(a, 16, seed=4)
    _make_ply(b, 16, seed=4, schema=alt_schema)
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "merge", a, b)
    assert proc.returncode == 1
    assert "dtype mismatch" in proc.stderr


def test_merge_diff_above_tolerance_fail(tmp_path: Path) -> None:
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _make_ply(a, 16, seed=5)
    _make_ply(b, 16, seed=5, perturb={"x": 0.5})
    tol = tmp_path / "tol.json"
    tol.write_text(json.dumps({"x": 1e-3}))
    proc = _run("--tolerance-json", tol, "merge", a, b)
    assert proc.returncode == 1
    assert "property 'x'" in proc.stderr and "exceeds tolerance" in proc.stderr


def test_merge_diff_within_tolerance_pass(tmp_path: Path) -> None:
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _make_ply(a, 16, seed=6)
    _make_ply(b, 16, seed=6, perturb={"x": 1e-5})
    tol = tmp_path / "tol.json"
    tol.write_text(json.dumps({"x": 1e-3}))
    proc = _run("--tolerance-json", tol, "merge", a, b)
    assert proc.returncode == 0, proc.stderr


def test_merge_default_tolerance_used_when_property_absent_in_json(tmp_path: Path) -> None:
    """Property not in tolerance.json falls back to --default-tolerance."""
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _make_ply(a, 16, seed=7)
    _make_ply(b, 16, seed=7, perturb={"x": 1.0})
    tol = tmp_path / "tol.json"
    tol.write_text(json.dumps({"y": 100.0}))  # x not present
    proc = _run("--tolerance-json", tol, "--default-tolerance", "1e-3", "merge", a, b)
    assert proc.returncode == 1
    assert "property 'x'" in proc.stderr


def test_merge_uchar_diff_respects_tolerance(tmp_path: Path) -> None:
    """u1 properties (red, green, blue, road_mask) compare via int diff up to tol."""
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _make_ply(a, 16, seed=8)
    _make_ply(b, 16, seed=8, perturb={"red": 2})  # exact int diff of 2
    tol = tmp_path / "tol.json"
    tol.write_text(json.dumps({"red": 1.0}))
    proc = _run("--tolerance-json", tol, "merge", a, b)
    assert proc.returncode == 1
    assert "property 'red'" in proc.stderr


# ---------- no_merge mode ----------


def test_no_merge_two_pairs_pass(tmp_path: Path) -> None:
    base, prop = tmp_path / "base", tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    _make_ply(base / "chunk0.ply", 8, seed=10)
    _make_ply(base / "chunk1.ply", 9, seed=11)
    _make_ply(prop / "chunk0.ply", 8, seed=10)
    _make_ply(prop / "chunk1.ply", 9, seed=11)
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "no_merge", base, prop)
    assert proc.returncode == 0, proc.stderr


def test_no_merge_pairs_by_sorted_name_across_naming_schemes(tmp_path: Path) -> None:
    """Baseline uses chunk0/1.ply, proposed uses chunk_0000/0001.ply: sorted pairing
    must align them positionally."""
    base, prop = tmp_path / "base", tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    _make_ply(base / "pai_uuid_chunk0.ply", 12, seed=20)
    _make_ply(base / "pai_uuid_chunk1.ply", 13, seed=21)
    _make_ply(prop / "chunk_0000.ply", 12, seed=20)
    _make_ply(prop / "chunk_0001.ply", 13, seed=21)
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "no_merge", base, prop)
    assert proc.returncode == 0, proc.stderr


def test_no_merge_file_count_mismatch_fail(tmp_path: Path) -> None:
    base, prop = tmp_path / "base", tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    _make_ply(base / "0.ply", 4)
    _make_ply(base / "1.ply", 4)
    _make_ply(prop / "0.ply", 4)
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "no_merge", base, prop)
    assert proc.returncode == 1
    assert "file count mismatch" in proc.stderr


def test_no_merge_empty_dirs_fail(tmp_path: Path) -> None:
    base, prop = tmp_path / "base", tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "no_merge", base, prop)
    assert proc.returncode == 1
    assert "no PLY files" in proc.stderr or "no ply files" in proc.stderr


def test_no_merge_missing_baseline_dir_fail(tmp_path: Path) -> None:
    prop = tmp_path / "prop"
    prop.mkdir()
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "no_merge", tmp_path / "nope", prop)
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr


def test_no_merge_missing_proposed_dir_fail(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    proc = _run("--tolerance-json", _empty_tol(tmp_path), "no_merge", base, tmp_path / "nope")
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr


def test_no_merge_per_pair_diff_above_tolerance_fail(tmp_path: Path) -> None:
    base, prop = tmp_path / "base", tmp_path / "prop"
    base.mkdir()
    prop.mkdir()
    _make_ply(base / "0.ply", 8, seed=30)
    _make_ply(prop / "0.ply", 8, seed=30, perturb={"x": 1.0})
    tol = tmp_path / "tol.json"
    tol.write_text(json.dumps({"x": 1e-3}))
    proc = _run("--tolerance-json", tol, "no_merge", base, prop)
    assert proc.returncode == 1
    assert "exceeds tolerance" in proc.stderr


# ---------- argparse / discovery ----------


def test_subcommand_required(tmp_path: Path) -> None:
    proc = _run()
    assert proc.returncode != 0


def test_unknown_subcommand_rejected(tmp_path: Path) -> None:
    proc = _run("frobnicate")
    assert proc.returncode != 0


# ---------- self-test against the real baselines ----------
# These touch the actual 100-200 MB PLY files in baselines/original_baseline.
# They confirm the script can read the real schema and that the baseline
# compares equal to itself. Skipped if the baselines are not on disk.

_BASELINE_MERGE_PLY = next(
    iter(REPO_ROOT.glob("baselines/original_baseline/merge/*/ply/*/*.ply")),
    None,
)
_BASELINE_NO_MERGE_DIR = next(
    iter(REPO_ROOT.glob("baselines/original_baseline/no_merge/*/ply/*/")),
    None,
)


@pytest.mark.skipif(_BASELINE_MERGE_PLY is None, reason="baselines/ not on disk")
def test_self_compare_merge_baseline_passes(tmp_path: Path) -> None:
    proc = _run(
        "--tolerance-json", _empty_tol(tmp_path),
        "merge", _BASELINE_MERGE_PLY, _BASELINE_MERGE_PLY,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(_BASELINE_NO_MERGE_DIR is None, reason="baselines/ not on disk")
def test_self_compare_no_merge_baseline_passes(tmp_path: Path) -> None:
    proc = _run(
        "--tolerance-json", _empty_tol(tmp_path),
        "no_merge", _BASELINE_NO_MERGE_DIR, _BASELINE_NO_MERGE_DIR,
    )
    assert proc.returncode == 0, proc.stderr
