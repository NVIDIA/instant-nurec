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
"""Derive per-property determinism tolerance from baselines/more_baselines.

The 5 reruns in `baselines/more_baselines/run_{1..5}` are independent invocations
of the same NRE script on the same input — any per-property diff between two of
them is run-to-run noise of the *original* pipeline. We take the max of those
diffs across all C(5,2)=10 pairs (per chunk index, per mode) and emit it as
`tests/tolerance.json`. Subsequent commits on this branch must keep
`benchmark/validate_parity.py` green within these tolerances; only Phase 2 CUDA →
torch swaps are allowed to ratchet a property's tolerance upwards.

Idempotent. Run from the repo root:

    .venv/bin/python benchmark/derive_determinism_tolerance.py

Writes to `tests/tolerance.json`.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = sorted(REPO_ROOT.glob("baselines/more_baselines/run_*/"))
OUT = REPO_ROOT / "tests" / "tolerance.json"


def _read_props(path: Path) -> dict[str, tuple[str, torch.Tensor]]:
    ply = PlyData.read(str(path))
    vert = ply["vertex"]
    props: dict[str, tuple[str, torch.Tensor]] = {}
    for prop in vert.properties:
        name = prop.name
        dtype = prop.val_dtype
        contig = np.ascontiguousarray(vert[name])
        if dtype.startswith("f"):
            t = torch.from_numpy(contig.astype(np.float32))
        elif dtype == "u1":
            t = torch.from_numpy(contig.astype(np.int32))
        elif dtype.startswith(("u", "i")):
            t = torch.from_numpy(contig.astype(np.int64))
        else:
            raise ValueError(f"Unsupported dtype {dtype} in {path}")
        props[name] = (dtype, t)
    return props


def _glob_one(run_dir: Path, mode: str, *, chunk_index: int | None = None) -> Path:
    """Return the merge PLY (chunk_index=None) or no_merge chunkN PLY for a run."""
    if mode == "merge":
        candidates = sorted(run_dir.glob("merge/*/ply/*/*.ply"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected 1 merge PLY in {run_dir}, got {candidates}")
        return candidates[0]
    if mode == "no_merge":
        all_files = sorted(run_dir.glob("no_merge/*/ply/*/*.ply"))
        # Sorted gives chunk0 first, chunk1 second.
        if chunk_index is None or chunk_index >= len(all_files):
            raise RuntimeError(
                f"expected chunk index {chunk_index} in {run_dir}, found {len(all_files)} files"
            )
        return all_files[chunk_index]
    raise ValueError(mode)


def _max_pair_diff(a_props: dict, b_props: dict, prev: dict[str, float]) -> dict[str, float]:
    """Update `prev` (dict prop -> max-so-far) with this pair's diffs.

    Every property in the union of `a_props` and `b_props` is touched exactly
    once (creating a 0.0 entry if no diff has ever been observed for it), so the
    final dict has full coverage of the schema even when the pipeline is
    bit-identical across runs.
    """
    if set(a_props) != set(b_props):
        raise RuntimeError(
            f"property sets differ across runs: {sorted(set(a_props) ^ set(b_props))}"
        )
    out = dict(prev)
    for name, (dt_a, t_a) in a_props.items():
        dt_b, t_b = b_props[name]
        if dt_a != dt_b:
            raise RuntimeError(f"dtype changed across runs for '{name}': {dt_a} vs {dt_b}")
        if t_a.shape != t_b.shape:
            raise RuntimeError(
                f"shape changed across runs for '{name}': {tuple(t_a.shape)} vs {tuple(t_b.shape)}"
            )
        diff = (t_a.float() - t_b.float()).abs().max().item()
        out[name] = max(out.get(name, 0.0), float(diff))
    return out


def main() -> None:
    if len(RUNS) < 2:
        raise SystemExit(
            f"need at least 2 runs in baselines/more_baselines, found {len(RUNS)}"
        )
    print(f"Found {len(RUNS)} runs:")
    for r in RUNS:
        print(f"  {r.relative_to(REPO_ROOT)}")
    print()

    tolerance: dict[str, float] = {}

    # merge: 1 PLY per run, all pairs.
    print("merge: pairwise diffs across runs")
    merge_props_per_run = [_read_props(_glob_one(r, "merge")) for r in RUNS]
    for i, j in combinations(range(len(RUNS)), 2):
        before = dict(tolerance)
        tolerance = _max_pair_diff(merge_props_per_run[i], merge_props_per_run[j], tolerance)
        bumped = {k: tolerance[k] for k in tolerance if tolerance[k] != before.get(k, 0.0)}
        if bumped:
            print(f"  run_{i+1} vs run_{j+1}: bumped {len(bumped)} props (showing top 3 by diff)")
            for k, v in sorted(bumped.items(), key=lambda kv: -kv[1])[:3]:
                print(f"    {k}: {v:.6e}")

    # no_merge: 2 chunks per run; pair within each chunk index.
    n_chunks = len(sorted(RUNS[0].glob("no_merge/*/ply/*/*.ply")))
    print(f"\nno_merge: {n_chunks} chunks per run, pairwise diffs per chunk index")
    for chunk_idx in range(n_chunks):
        per_run = [_read_props(_glob_one(r, "no_merge", chunk_index=chunk_idx)) for r in RUNS]
        for i, j in combinations(range(len(RUNS)), 2):
            before = dict(tolerance)
            tolerance = _max_pair_diff(per_run[i], per_run[j], tolerance)
            bumped = {k: tolerance[k] for k in tolerance if tolerance[k] != before.get(k, 0.0)}
            if bumped:
                print(
                    f"  chunk{chunk_idx} run_{i+1} vs run_{j+1}: bumped {len(bumped)} props (top 3)"
                )
                for k, v in sorted(bumped.items(), key=lambda kv: -kv[1])[:3]:
                    print(f"    {k}: {v:.6e}")

    print()
    print("Final per-property determinism tolerance (max |a-b| across all run pairs):")
    for k in sorted(tolerance):
        print(f"  {k:>14}: {tolerance[k]:.6e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(tolerance, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
