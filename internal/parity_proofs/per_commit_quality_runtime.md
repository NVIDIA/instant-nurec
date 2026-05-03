<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Per-commit quality + runtime regression report

Reference: `baselines/original_baseline/` — the NRE@`a54a6af` bazel run output.
Last commit with bit-exact parity (every property max-abs = 0, vertex count
identical): `2b48686` (Phase A.4 `packed_ops` swap, last commit before
`fc23075` introduced the first kernel-precision drift).

All distance metrics are in PLY units (scene-rescaled by 0.15 from the
original-frame meters; multiply by 6.667 to get back to meters).

## How to read the table

* **wall (no_merge / merge)** — wall-clock seconds for `python run_inference.py`
  end-to-end, per mode. For pre-A.8 commits (still bazel-driven), this includes
  the bazel server + first-build cost (~14 s). For commits ≤ `fc23075` the cold
  path also pays the `nrm-kelvin-pa_1.0.0-front.ckpt` → `state_dict` →
  `kelvin_full.pt` save (`+ ~50–80 s`) on the first run; subsequent runs in the
  same epoch are warm.
* **Δ chunk0 / chunk1 / merge** — vertex count delta (proposed − baseline).
  Positive = extra Gaussians, negative = fewer.
* **Chamfer** — bidirectional Chamfer distance in PLY units (½ ⋅ (mean NN A→B + mean NN B→A)).
  Bit-exact = 0; smaller is better.
* **F1@0.01** — F-score at τ=0.01 PLY units (nearest-neighbor coverage; 1.0 = perfect).

## Sweep results

| commit  | label                          | wall (no_merge / merge) | Δ chunk0 | Δ chunk1 | Δ merge | Chamfer (chunk0 / chunk1 / merge) | F1@0.01 (chunk0 / chunk1 / merge) | Source     |
|---------|--------------------------------|-------------------------|----------|----------|---------|-----------------------------------|-----------------------------------|------------|
| —       | baseline (bazel reference)     | 19.6 / 21.1 (log.txt)   |    —     |    —     |    —    | 0 / 0 / 0                         | 1.000 / 1.000 / 1.000             | log.txt    |
| 2b48686 | A.4 packed_ops                 | 108.7 / 36.5 (cold)     |    0     |    0     |    0    | 0 / 0 / 0                         | 1.000 / 1.000 / 1.000             | sweep ✓    |
| fc23075 | A.1 se3pose_from_matrix        | 120.0 / 37.2 (cold)     |   +8     |   +5     |  −25    | 4.78e-3 / 6.11e-3 / 4.38e-3       | 0.902 / 0.860 / 0.909             | sweep ✓    |
| 6b32da4 | A.5 pose_calib                 | 39.8 / 36.4             |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             | sweep ✓    |
| 037ed34 | A.2+A.3+A.6 bundle             | 121.5 / 37.7            |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             | sweep ✓    |
| efa4cc9 | A.7 vren                       | 63.6 / 52.2             |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             | sweep ✓    |
| 882c0d0 | A.8 drop libs/                 | 128.1 / 49.2            |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             | sweep ✓    |
| c144996 | lietorch → _se3_torch shim     | 135.4 / 50.7            |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             | sweep ✓    |
| e177e4c | Phase C basic flatten          | 132.8 / 49.2            |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             | sweep ✓    |
| b33892f | drop torchvision (B prep)      | 130.5 / 51.3            |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             | sweep ✓    |
| 07c8b20 | drop bazel + cu128 wheel       | 32.5 / 33.5             |   −2     |   +7     |  −21    | 5.09e-3 / 6.26e-3 / 4.61e-3       | 0.903 / 0.851 / 0.907             | sweep ✓    |
| 7ea7a65 | single-.pt artifact            | 29.1 / 33.1             |   −2     |   +7     |  −21    | 5.09e-3 / 6.26e-3 / 4.61e-3       | 0.903 / 0.851 / 0.907             | sweep ✓    |
| e58736e | NRM → InstantNuRec rename      | 28.5 / 33.1             |   −2     |   +7     |  −21    | 5.09e-3 / 6.26e-3 / 4.61e-3       | 0.903 / 0.851 / 0.907             | smoke ✓    |
| HEAD    | (current)                      | 28.5 / 33.1             |   −2     |   +7     |  −21    | 5.09e-3 / 6.26e-3 / 4.61e-3       | 0.903 / 0.851 / 0.907             | sweep ✓    |

## Sweep coverage notes

The first sweep crashed mid-run when bazel's per-target cache exploded
to **297 GB** (each commit's bazel build re-resolves BUILD targets
without deduping across commits). After cleaning the cache and adding
per-commit `bazel clean --expunge` to the sweep script, a second sweep
filled in every previously-missing commit. The data above is fully
empirical except `e58736e` (NRM → InstantNuRec rename), which was
verified via a single-commit smoke run rather than a full sweep — the
rename is provably math-unchanging (textual identifier swap + a one-off
.pt re-pickle that round-trips the same tensor data through new
qualnames; `validate_parity.py` green for both modes against the
baseline).

The 3 commits between A.8 and 07c8b20 (`c144996` lietorch shim,
`e177e4c` Phase C basic flatten, `b33892f` torchvision drop) reproduce
the A.8 drift numbers exactly — confirming they are math-unchanging:
lietorch `SE3`/`SO3` ops were replaced by algebraically-identical
pure-torch ops; Phase C flatten is layout-only; torchvision's
`Normalize` was an affine transform replicated bit-for-bit by
`_RGBNormalize`, and `transforms.functional.resize(antialias=True)`
maps to `torch.nn.functional.interpolate(antialias=True)` (same kernel).

## Where parity changed (single source of truth)

* **`fc23075` Phase A.1** — *first commit to break bit-exact parity*.
  Replaced `libs.geometry.kernels.pose.se3pose_from_matrix` with a
  pure-torch f64-internal Shepperd's method. Slang and torch lower to
  different SASS instruction sequences on CUDA (verified by side-by-side
  ULP measurement); torch f32 result differs by 0–3 ULP per quaternion
  component. With baseline density-prune threshold this flips ~5–30
  cull-boundary Gaussians per chunk; per-property maxabs across the
  remaining points stays inside the existing tolerance bounds. Commit
  added `_vertex_count_delta=50` to `tests/tolerance.json`.
* **`6b32da4` Phase A.5** — `compute_poses_and_timestamps`. Slang did a
  pointless SE3 round-trip (4×4 → quat,t → 4×4) inside the kernel even
  with `enable_calib=False`; torch skips it. No-op modulo ULPs but
  redistributes the cull-boundary count.
* **`037ed34` Phase A.2+A.3+A.6 bundle** — `image_points_to_world_rays_shutter_pose`
  + the supporting dataclasses. Torch rolling-shutter pose-interp
  matches ncore's pure-python reference more faithfully than the slang
  kernel did → chunk0 drift improved (+8 → +5).
* **`07c8b20` Phase B.2** — *cuda runtime swap, not a math change*.
  Same Phase A torch impls, run via the public `torch==2.7.0+cu128`
  wheel instead of NRE's internal `torch==2.7.0+cu128.gitc41f6e01287`
  build. Different TF32/cuDNN heuristics → drift redistribution from
  +5/−4/−30 to −2/+7/−21. Same 50-vertex band held.

## Where runtime changed

The wall-time gap of ~12 s no_merge and ~11 s merge between baseline
(19.6 / 21.1) and HEAD (28.5 / 33.1) is the *cumulative cost of the
slang/CUDA → torch swaps*. Unlike the parity gap (concentrated in 3
commits), the runtime cost is distributed across all of A.1, A.5,
A.2+A.3+A.6, A.7 and the cu128 wheel swap. Per the commit-body audit
the dominant contributors are:

* **`efa4cc9` A.7 vren** — `ray_cuboidtracks_intersection` and
  `point_cuboidtracks_intersection_interpolate_pose`. The torch impl
  iterates per-track in Python (~50–100 tracks per frame) but is fully
  vectorised across rays per iteration. Slang did this in a single GPU
  grid; the per-track loop is the dominant runtime add.
* **`037ed34` A.2+A.3+A.6 bundle** — FTheta inverse projection +
  rolling-shutter pose interp in pure torch. Slang fused these into a
  single kernel; torch path makes 5–6 separate kernel launches per
  pixel batch.
* **`fc23075` A.1 se3pose** — f64-internal Shepperd's method dominates
  the camera-pose-encode portion of the predict loop. ~0.5 s addend per
  predict step.

## Reproducing the metrics

The sweep tool lives at `benchmark/compare_clouds.py`. To re-run on a
single PLY pair:

```
python benchmark/compare_clouds.py \
  baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_*/pai_*.ply \
  /tmp/out/merge/*/ply/*/pai_*.ply
```

The full sweep script is at `internal/parity_proofs/run_sweep.sh`.
Re-running the full sweep requires ~50 GB free `/` disk (we hit 297 GB
of bazel cache during the first attempt; `bazel clean --expunge` between
commits keeps it sane).

## Tolerance bumps

`tests/tolerance.json` was modified by exactly one commit:

* **`9bc5bd4`** — initial creation. Per-property tolerances derived from
  10 pairwise comparisons across `baselines/more_baselines/run_{1..5}`
  (the run-to-run noise floor of the original NRE pipeline).
* **`fc23075`** — added `_vertex_count_delta=50` to absorb the
  cull-boundary count drift introduced by Phase A.1's slang→torch swap.
  No per-property tolerance was bumped.

Every subsequent commit kept the same tolerance.json values; the
post-A.1 drift moves all stayed inside the +50/−50 band on vertex count
AND inside the per-property bounds.
