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

#!/usr/bin/env python3
"""Validate PLY parity against a baseline.

Every commit that touches runtime code must keep this script green against
`baselines/original_baseline` within the tolerances recorded in
`internal/tests/tolerance.json` (derived from the run-to-run noise floor across the
5 reruns in `baselines/more_baselines/run_{1..5}`).

Usage::

    validate_parity.py merge <baseline_ply> <proposed_ply>
    validate_parity.py no_merge <baseline_dir> <proposed_dir>

In `no_merge` mode the two directories are recursively globbed for `*.ply` and
paired in sorted order, so the script accommodates both the baseline naming
(``pai_<UUID>_chunk0.ply``) and the standalone naming (``chunk_0000.ply``).

Per-property diffs are computed in torch on whichever device torch is configured
to use (typically CPU here — PLYs sit easily in RAM and the inference step that
produces them already burned the GPU). Per-property tolerances live in a JSON
file keyed by property name; properties absent from the file fall back to
``--default-tolerance`` (1e-3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from plyfile import PlyData


def _load_tolerance(path: Path) -> Mapping[str, float]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _list_plys(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.ply"))


def _to_tensor(arr: np.ndarray, dtype: str) -> torch.Tensor:
    contig = np.ascontiguousarray(arr)
    if dtype.startswith("f"):
        return torch.from_numpy(contig.astype(np.float32))
    if dtype == "u1":
        return torch.from_numpy(contig.astype(np.int32))
    if dtype.startswith(("u", "i")):
        return torch.from_numpy(contig.astype(np.int64))
    raise ValueError(f"Unsupported PLY property dtype: {dtype}")


def _read_ply(path: Path) -> tuple[int, dict[str, tuple[str, torch.Tensor]]]:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY file has no 'vertex' element: {path}")
    vert = ply["vertex"]
    props: dict[str, tuple[str, torch.Tensor]] = {}
    for prop in vert.properties:
        name = prop.name
        dtype = prop.val_dtype
        props[name] = (dtype, _to_tensor(vert[name], dtype))
    return int(vert.count), props


def _compare_pair(
    a: Path,
    b: Path,
    tol: Mapping[str, float],
    default_tol: float,
    vertex_count_delta: int,
) -> list[str]:
    errors: list[str] = []
    count_a, props_a = _read_ply(a)
    count_b, props_b = _read_ply(b)

    abs_delta = abs(count_a - count_b)
    if abs_delta > vertex_count_delta:
        errors.append(
            f"vertex count mismatch: {a.name}={count_a} vs {b.name}={count_b} "
            f"(delta={abs_delta} > tolerance={vertex_count_delta})"
        )
        return errors

    if set(props_a) != set(props_b):
        only_a = sorted(set(props_a) - set(props_b))
        only_b = sorted(set(props_b) - set(props_a))
        errors.append(
            f"property name mismatch in {a.name}: only_in_baseline={only_a} "
            f"only_in_proposed={only_b}"
        )
        return errors

    if count_a != count_b:
        # Counts differ but within tolerance — skip per-property maxabs since
        # there's no canonical 1:1 vertex correspondence between point clouds
        # of different sizes. The count-delta check above is the only signal.
        return errors

    for name in sorted(props_a):
        dt_a, t_a = props_a[name]
        dt_b, t_b = props_b[name]
        if dt_a != dt_b:
            errors.append(
                f"dtype mismatch for property '{name}' in {a.name}: {dt_a} vs {dt_b}"
            )
            continue
        diff = (t_a.float() - t_b.float()).abs().max().item()
        prop_tol = float(tol.get(name, default_tol))
        if diff > prop_tol:
            errors.append(
                f"property '{name}' diff exceeds tolerance: "
                f"max|a-b|={diff:.6e} > tol={prop_tol:.6e} (file={a.name})"
            )
    return errors


def _vertex_count_delta(tol: Mapping[str, float], cli_override: int | None) -> int:
    """Resolve the vertex-count delta tolerance.

    Resolution order:
      1. CLI ``--vertex-count-delta`` if provided (highest priority).
      2. ``_vertex_count_delta`` key in ``internal/tests/tolerance.json``.
      3. Default 0 (exact-match).
    """
    if cli_override is not None:
        return int(cli_override)
    return int(tol.get("_vertex_count_delta", 0))


def cmd_merge(
    baseline: Path,
    proposed: Path,
    tol_path: Path,
    default_tol: float,
    vertex_count_delta: int | None,
) -> int:
    if not baseline.is_file():
        print(f"FAIL: baseline file does not exist: {baseline}", file=sys.stderr)
        return 1
    if not proposed.is_file():
        print(f"FAIL: proposed file does not exist: {proposed}", file=sys.stderr)
        return 1
    tol = _load_tolerance(tol_path)
    delta = _vertex_count_delta(tol, vertex_count_delta)
    errors = _compare_pair(baseline, proposed, tol, default_tol, delta)
    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"PASS: {baseline.name} matches {proposed.name} within tolerance.")
    return 0


def cmd_no_merge(
    baseline_dir: Path,
    proposed_dir: Path,
    tol_path: Path,
    default_tol: float,
    vertex_count_delta: int | None,
) -> int:
    if not baseline_dir.is_dir():
        print(f"FAIL: baseline dir does not exist: {baseline_dir}", file=sys.stderr)
        return 1
    if not proposed_dir.is_dir():
        print(f"FAIL: proposed dir does not exist: {proposed_dir}", file=sys.stderr)
        return 1
    a_files = _list_plys(baseline_dir)
    b_files = _list_plys(proposed_dir)
    if len(a_files) != len(b_files):
        print(
            f"FAIL: file count mismatch: baseline={len(a_files)} "
            f"proposed={len(b_files)}",
            file=sys.stderr,
        )
        print(f"  baseline files: {[f.name for f in a_files]}", file=sys.stderr)
        print(f"  proposed files: {[f.name for f in b_files]}", file=sys.stderr)
        return 1
    if not a_files:
        print(
            f"FAIL: no PLY files found under {baseline_dir} / {proposed_dir}",
            file=sys.stderr,
        )
        return 1
    tol = _load_tolerance(tol_path)
    delta = _vertex_count_delta(tol, vertex_count_delta)
    all_errors: list[str] = []
    for a, b in zip(a_files, b_files):
        all_errors.extend(_compare_pair(a, b, tol, default_tol, delta))
    if all_errors:
        for e in all_errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"PASS: {len(a_files)} PLY pair(s) match within tolerance.")
    return 0


def _default_tolerance_path() -> Path:
    return Path(__file__).resolve().parent.parent / "tests" / "tolerance.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate PLY parity against a baseline."
    )
    parser.add_argument(
        "--tolerance-json",
        type=Path,
        default=_default_tolerance_path(),
        help="Per-property tolerance file (JSON object keyed by property name).",
    )
    parser.add_argument(
        "--default-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for properties absent from --tolerance-json.",
    )
    parser.add_argument(
        "--vertex-count-delta",
        type=int,
        default=None,
        help=(
            "Maximum allowed |count_a - count_b| per file. Overrides the "
            "``_vertex_count_delta`` key in tolerance.json. Default 0 "
            "(exact-match) when not set anywhere."
        ),
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    merge = sub.add_parser("merge", help="Compare a single PLY file pair.")
    merge.add_argument("baseline", type=Path)
    merge.add_argument("proposed", type=Path)

    nm = sub.add_parser(
        "no_merge",
        help="Compare two directories of PLY files (paired by sorted filename).",
    )
    nm.add_argument("baseline", type=Path)
    nm.add_argument("proposed", type=Path)

    args = parser.parse_args(argv)
    if args.mode == "merge":
        return cmd_merge(
            args.baseline, args.proposed, args.tolerance_json,
            args.default_tolerance, args.vertex_count_delta,
        )
    return cmd_no_merge(
        args.baseline, args.proposed, args.tolerance_json,
        args.default_tolerance, args.vertex_count_delta,
    )


if __name__ == "__main__":
    sys.exit(main())
