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
"""Attributed point-cloud similarity for Kelvin Gaussian PLY outputs.

Where ``validate_parity.py`` requires same-cardinality clouds and compares
property-by-property, this tool measures *how similar* two clouds are when
their vertex counts differ. It computes three classes of metrics:

1. **Bidirectional Chamfer distance** on positions only — answers "are the
   two geometric structures the same?". O((N+M)·log(N+M)) via KD-tree.
2. **Nearest-neighbor attribute correspondence** — for each point in B, find
   its nearest neighbor in A in xyz, then compare attributes (density,
   color, opacity, rotation, scale, masks). Bidirectional A↔B. Reports
   per-attribute RMSE, mean abs error, and median abs error.
3. **Marginal Wasserstein-1** per attribute — the empirical attribute
   distribution of cloud A vs cloud B as 1-D probability measures.
   Insensitive to vertex count.

Usage:

    compare_clouds.py <ply_a> <ply_b>                  # single-file
    compare_clouds.py --no-merge <dir_a> <dir_b>       # directory pairing

Prints a structured table; exits 0 always (this is a measurement tool, not
a parity gate). Use ``validate_parity.py`` if you want the parity gate.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance


@dataclass
class CloudMetrics:
    n_a: int
    n_b: int
    cd_a_to_b: float
    cd_b_to_a: float
    chamfer: float
    hausdorff: float
    fscore_at_001: float
    fscore_at_01: float
    nn_attr_rmse: dict[str, float]    # B→A nearest-neighbor attribute residual RMSE per property
    nn_attr_mae: dict[str, float]
    nn_attr_p95: dict[str, float]
    wasserstein: dict[str, float]     # marginal Wasserstein-1 per property


_POSITION_COLS = ("x", "y", "z")


def _load_ply(path: Path) -> dict[str, np.ndarray]:
    """Read a PLY into a dict keyed by property name."""
    ply = PlyData.read(str(path))
    el = ply["vertex"]
    return {name: np.asarray(el[name]) for name in el.data.dtype.names}


def _positions(props: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([props[c].astype(np.float64) for c in _POSITION_COLS], axis=1)


def _bidirectional_chamfer(pa: np.ndarray, pb: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (cd_a_to_b, cd_b_to_a, chamfer, hausdorff). All in PLY units."""
    tree_a = cKDTree(pa)
    tree_b = cKDTree(pb)
    d_a_to_b, _ = tree_b.query(pa, k=1)
    d_b_to_a, _ = tree_a.query(pb, k=1)
    cd_a_to_b = float(np.mean(d_a_to_b))
    cd_b_to_a = float(np.mean(d_b_to_a))
    chamfer = 0.5 * (cd_a_to_b + cd_b_to_a)
    hausdorff = max(float(d_a_to_b.max()), float(d_b_to_a.max()))
    return cd_a_to_b, cd_b_to_a, chamfer, hausdorff


def _fscore(pa: np.ndarray, pb: np.ndarray, tau: float) -> float:
    """F-score @ τ — precision/recall at distance threshold tau (in PLY units)."""
    tree_a = cKDTree(pa)
    tree_b = cKDTree(pb)
    d_a_to_b, _ = tree_b.query(pa, k=1)
    d_b_to_a, _ = tree_a.query(pb, k=1)
    precision = float(np.mean(d_a_to_b < tau))
    recall = float(np.mean(d_b_to_a < tau))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _nn_attribute_residuals(
    props_a: dict[str, np.ndarray],
    props_b: dict[str, np.ndarray],
    pa: np.ndarray,
    pb: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """For each b ∈ B, find its nearest neighbor a* ∈ A in xyz; compare
    attributes (a*, b). Returns (rmse, mae, p95) dicts keyed by property
    name. Position columns are excluded (they're the matching key).
    """
    tree_a = cKDTree(pa)
    _, nn_idx = tree_a.query(pb, k=1)

    rmse: dict[str, float] = {}
    mae: dict[str, float] = {}
    p95: dict[str, float] = {}
    for name in props_a:
        if name in _POSITION_COLS:
            continue
        if name not in props_b:
            continue
        va = props_a[name].astype(np.float64)
        vb = props_b[name].astype(np.float64)
        diff = vb - va[nn_idx]
        diff_abs = np.abs(diff)
        rmse[name] = float(np.sqrt(np.mean(diff * diff)))
        mae[name] = float(np.mean(diff_abs))
        p95[name] = float(np.percentile(diff_abs, 95))
    return rmse, mae, p95


def _marginal_wasserstein(
    props_a: dict[str, np.ndarray], props_b: dict[str, np.ndarray]
) -> dict[str, float]:
    """1-D Wasserstein per (non-position) attribute. Distribution-level
    similarity, no point-by-point matching.
    """
    out: dict[str, float] = {}
    for name in props_a:
        if name in _POSITION_COLS or name not in props_b:
            continue
        va = props_a[name].astype(np.float64).ravel()
        vb = props_b[name].astype(np.float64).ravel()
        out[name] = float(wasserstein_distance(va, vb))
    return out


def compute_metrics(path_a: Path, path_b: Path) -> CloudMetrics:
    a = _load_ply(path_a)
    b = _load_ply(path_b)
    pa = _positions(a)
    pb = _positions(b)
    cd_ab, cd_ba, ch, ha = _bidirectional_chamfer(pa, pb)
    fs_001 = _fscore(pa, pb, tau=0.01)
    fs_01 = _fscore(pa, pb, tau=0.1)
    rmse, mae, p95 = _nn_attribute_residuals(a, b, pa, pb)
    wass = _marginal_wasserstein(a, b)
    return CloudMetrics(
        n_a=len(pa), n_b=len(pb),
        cd_a_to_b=cd_ab, cd_b_to_a=cd_ba,
        chamfer=ch, hausdorff=ha,
        fscore_at_001=fs_001, fscore_at_01=fs_01,
        nn_attr_rmse=rmse, nn_attr_mae=mae, nn_attr_p95=p95,
        wasserstein=wass,
    )


def _print_metrics(label: str, m: CloudMetrics) -> None:
    print(f"\n=== {label} ===")
    print(f"  vertex counts:  A={m.n_a:,}  B={m.n_b:,}  Δ={m.n_b - m.n_a:+,}")
    print("  Chamfer (PLY units, scene-rescaled by 0.15):")
    print(f"    A→B mean NN dist:   {m.cd_a_to_b:.6e}")
    print(f"    B→A mean NN dist:   {m.cd_b_to_a:.6e}")
    print(f"    Chamfer (½ sum):    {m.chamfer:.6e}")
    print(f"    Hausdorff (max NN): {m.hausdorff:.6e}")
    print("  F-score @ τ:")
    print(f"    τ=0.01:             {m.fscore_at_001:.6f}")
    print(f"    τ=0.10:             {m.fscore_at_01:.6f}")
    print("  Nearest-neighbor attribute residuals (B→A pairs):")
    print(f"    {'attr':<14s}  {'RMSE':>14s}  {'MAE':>14s}  {'p95(|err|)':>14s}")
    for k in sorted(m.nn_attr_rmse):
        print(f"    {k:<14s}  {m.nn_attr_rmse[k]:>14.6e}  {m.nn_attr_mae[k]:>14.6e}  {m.nn_attr_p95[k]:>14.6e}")
    print("  Marginal Wasserstein-1 per attribute:")
    for k in sorted(m.wasserstein):
        print(f"    {k:<14s}  {m.wasserstein[k]:>14.6e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", type=Path, help="cloud A (the 'reference')")
    parser.add_argument("b", type=Path, help="cloud B (the 'proposed')")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="treat A and B as directories of PLYs, pair by sorted basename",
    )
    args = parser.parse_args(argv)

    if args.no_merge:
        files_a = sorted(args.a.glob("*.ply"))
        files_b = sorted(args.b.glob("*.ply"))
        if len(files_a) != len(files_b):
            print(
                f"file count mismatch: A has {len(files_a)} PLYs, B has {len(files_b)}",
                file=sys.stderr,
            )
            return 1
        for fa, fb in zip(files_a, files_b):
            m = compute_metrics(fa, fb)
            _print_metrics(f"{fa.name} ↔ {fb.name}", m)
    else:
        m = compute_metrics(args.a, args.b)
        _print_metrics(f"{args.a.name} ↔ {args.b.name}", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
