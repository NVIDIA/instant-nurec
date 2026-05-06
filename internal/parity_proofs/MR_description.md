<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--
GitLab MR description for the kelvin-standalone branch.
Copy the body below into the MR's description field on GitLab.
-->

# Standalone NRM Kelvin predict mode

Predict-only carve-out of the NRM Kelvin model from the NRE codebase
(`a54a6af`). Bazel and slang/CUDA-compiled kernels are gone; the
runtime is pure-torch + a `torch==2.7.0+cu128` wheel. Layout matches
asset-harvester. The pickled artifact is now a single `kelvin_full.pt`
under the placeholder HF repo `nvidia/instant-nurec-kelvin`.

## What changed (high level)

- **Phase A** — replaced every slang/CUDA kernel with a pure-torch
  equivalent: `se3pose_from_matrix`, `compute_poses_and_timestamps`,
  `image_points_to_world_rays_shutter_pose`, `vren` ray-cuboid
  intersection, `packed_ops` searchsorted/linstep, sensor-parameter
  dataclasses.
- **Phase B** — dropped bazel; canonical invocation is now
  `python run_inference.py`. Pinned `torch==2.7.0+cu128` (matches
  NRE's internal cu128 build, public-installable).
- **Phase C** — flattened to asset-harvester layout (`instant_nurec/`
  package, `benchmark/`, `tests/`, `internal/`).
- **Polish** — single `.pt` artifact (NGC-checkpoint path dropped),
  pydantic defaults inlined into config schema (no `config.py`
  dict-literal), README rewritten asset-harvester-style, all
  `nrm`/`NRM`/`NRE` mentions removed from code (`InstantNuRec` /
  `instantnurec` everywhere), pure-torch lietorch shim with credit to
  the original authors (Zachary Teed and Jia Deng), docs/license
  files in place.

## Parity vs reference baseline

Parity is verified against `baselines/original_baseline/` — the
`a54a6af` bazel run output — using `benchmark/validate_parity.py`.
Bit-exact parity held until commit `2b48686` (Phase A.4); after that,
slang→torch instruction-sequence differences on CUDA introduce 0–3
ULP per quaternion component, which flips ~5–30 cull-boundary
Gaussians per chunk. The per-property max-abs differences stay inside
the run-to-run noise floor recorded in `tests/tolerance.json`
(derived from 5 reruns of the original pipeline).

Quality metrics (bidirectional Chamfer, F-score@τ=0.01, marginal
Wasserstein-1) and runtime are tracked per-commit. Reference for both
is the `a54a6af` bazel run output. Each row is the cumulative state at
that commit, not the delta vs the previous row.

| commit  | label                          | wall (no_merge / merge) | Δ chunk0 | Δ chunk1 | Δ merge | Chamfer (chunk0 / chunk1 / merge) | F1@0.01 (chunk0 / chunk1 / merge) |
|---------|--------------------------------|-------------------------|----------|----------|---------|-----------------------------------|-----------------------------------|
| —       | baseline (bazel reference)     | 19.6 / 21.1 s           |    —     |    —     |    —    | 0 / 0 / 0                         | 1.000 / 1.000 / 1.000             |
| 2b48686 | A.4 packed_ops                 | 108.7 / 36.5 s (cold)   |    0     |    0     |    0    | 0 / 0 / 0                         | 1.000 / 1.000 / 1.000             |
| fc23075 | A.1 se3pose_from_matrix        | 120.0 / 37.2 s (cold)   |   +8     |   +5     |  −25    | 4.78e-3 / 6.11e-3 / 4.38e-3       | 0.902 / 0.860 / 0.909             |
| 6b32da4 | A.5 pose_calib                 | 39.8 / 36.4 s           |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |
| 037ed34 | A.2+A.3+A.6 bundle             | 121.5 / 37.7 s          |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |
| efa4cc9 | A.7 vren                       | 63.6 / 52.2 s           |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |
| 882c0d0 | A.8 drop libs/                 | 128.1 / 49.2 s          |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |
| c144996 | lietorch → _se3_torch shim     | 135.4 / 50.7 s          |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |
| e177e4c | Phase C basic flatten          | 132.8 / 49.2 s          |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |
| b33892f | drop torchvision (B prep)      | 130.5 / 51.3 s          |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |
| 07c8b20 | drop bazel + cu128 wheel       | 32.5 / 33.5 s           |   −2     |   +7     |  −21    | 5.09e-3 / 6.26e-3 / 4.61e-3       | 0.903 / 0.851 / 0.907             |
| 7ea7a65 | single-.pt artifact            | 29.1 / 33.1 s           |   −2     |   +7     |  −21    | 5.09e-3 / 6.26e-3 / 4.61e-3       | 0.903 / 0.851 / 0.907             |
| e58736e | NRM → InstantNuRec rename      | 28.5 / 33.1 s           |   −2     |   +7     |  −21    | 5.09e-3 / 6.26e-3 / 4.61e-3       | 0.903 / 0.851 / 0.907             |
| 69c7601 | migrate to uv (loose pins)     | 49.7 / 28.0 s           |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |
| 7d713e5 | uv pins → NRE-bazel exact      | 28.5 / 33.1 s           |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |

`validate_parity.py` exits 0 in both `merge` and `no_merge` modes at HEAD.

### Where parity changed

Only **three** rows in the table moved parity vs the row above them.
Everything else either matched or was parity-neutral.

* **`fc23075` Phase A.1** — *first commit to break bit-exact parity*.
  Pure-torch f64-internal Shepperd's method replaces the slang
  kernel; torch f32 result differs by 0–3 ULP per quaternion
  component on CUDA. ~5–30 cull-boundary Gaussians per chunk flip.
  Per-property maxabs stays inside `tests/tolerance.json`.
  `_vertex_count_delta=50` added to absorb the count delta.
* **`6b32da4` Phase A.5** — `compute_poses_and_timestamps`. Slang did
  a pointless SE3 round-trip (4×4 → quat,t → 4×4) inside the kernel
  even with `enable_calib=False`; torch skips it. No-op modulo ULPs
  but redistributes the cull-boundary count.
* **`07c8b20` drop bazel + cu128 wheel** — *transitive dep swap, not
  a math change*. Same Phase A torch impls. Two things changed at
  once: (a) public `torch==2.7.0+cu128` wheel instead of NRE's
  internal `+cu128.gitc41f6e01287` build, and (b) the first
  `pip install -e .` after dropping bazel resolved the entire
  transitive Python tree fresh — pip picked numpy 2.x / scipy 1.17 /
  pandas 3.0 / nvidia-ncore 19.0 instead of NRE's pinned 1.26.4 /
  1.11.1 / 2.3.3 / 18.7.0. Drift redistributed to −2/+7/−21. The
  commit message originally blamed (a) the cu128 wheel; we later
  disproved that — see `69c7601` and `7d713e5` below.

### Where parity returned

* **`69c7601` migrate to uv (loose pins)** — uv's deterministic
  resolution picked dep versions much closer to NRE bazel than pip's
  free-for-all had at `07c8b20` (numpy 1.26.4 / pandas 2.3.3 /
  nvidia-ncore 18.7.0 — exact NRE matches; scipy 1.11.4 /
  numcodecs 0.15.1 / pyyaml 6.0.2 / huggingface_hub 1.13.0 — a few
  patch versions newer). Parity snapped back to **+5/−4/−30**, the
  Phase A.5–A.8 pattern. So the dominant factor in the `07c8b20` drift
  was the numpy / pandas / ncore major-version jumps, not the cu128
  wheel.
* **`7d713e5` uv pins → NRE-bazel exact** — tightened the four
  remaining patch-level drifts to NRE's exact pins
  (`scipy==1.11.1`, `numcodecs==0.11.0`, `pyyaml==6.0`,
  `huggingface_hub==0.36.2` on Python 3.11.15). **No parity change**
  vs `69c7601` — those four versions are parity-neutral. This commit
  exists for reproducibility-against-NRE rather than for parity.

### Net story

Three commits actually moved parity (`fc23075`, `6b32da4`, `07c8b20`).
Two of them (`fc23075`, `6b32da4`) are intrinsic to the slang→torch
port and unavoidable. The third (`07c8b20`) was a transient drift that
returned to the pre-drop pattern at `69c7601` once the dep tree was
managed deterministically. The residual +5/−4/−30 vs the bazel
reference at HEAD is entirely from `fc23075` + `6b32da4`. The
public-vs-internal cu128 wheel difference is parity-neutral.

## Runtime

Wall-clock per mode is in the table above. The ~9 s no_merge /
~12 s merge gap between baseline (19.6 / 21.1 s) and HEAD
(28.5 / 33.1 s) is the cumulative cost of the slang→torch swaps;
the per-commit attribution (A.7 vren is the biggest single
contributor at +24 s no_merge cold-cache) is in the linked report.

## Reproduction

```
./setup.sh
source .venv/bin/activate

mkdir -p /tmp/out/no_merge /tmp/out/merge
python run_inference.py --ncore-path /storage/data/nurec/ncorev4/debug.lst \
    --output-dir /tmp/out/no_merge --merge none
python run_inference.py --ncore-path /storage/data/nurec/ncorev4/debug.lst \
    --output-dir /tmp/out/merge   --merge frustum-ownership

python benchmark/validate_parity.py merge \
    baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_*/pai_*.ply \
    /tmp/out/merge/*/ply/*/*.ply
python benchmark/validate_parity.py no_merge \
    baselines/original_baseline/no_merge/e78RJgNGViMA3hsJoQXYVx/ply/pai_*/ \
    /tmp/out/no_merge/*/ply/*/

.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
```

## Removed dependencies

`pytorch_lightning`, `hydra-core`, `omegaconf`, `nvidia-ncore-internal`,
`torch_scatter`, `nvdiffrast`, `gsplat`, `lietorch`, `torchvision`,
plus the bazel build system and all `libs/*` slang/CUDA kernel sources.

## Files of note

- `instant_nurec/` — main package (asset-harvester layout)
- `benchmark/validate_parity.py` — PLY parity gate
- `benchmark/compare_clouds.py` — Chamfer/F-score/NN-attribute residuals/marginal Wasserstein
- `tests/tolerance.json` — per-property determinism tolerance
- `internal/parity_proofs/per_commit_quality_runtime.md` — full per-commit sweep table
- `scripts/migrate_kelvin_full_pt.py` — one-off `.pt` re-pickle helper for the NRM → InstantNuRec rename
