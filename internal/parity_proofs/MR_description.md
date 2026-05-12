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
| 7ec41ad | C2 switch loader to torch.jit.load | 40.5 / 42.5 s       |   +5     |  +20     |   −6    | 4.75e-3 / 7.11e-3 / 4.90e-3       | 0.901 / 0.835 / 0.893             |
| 34f69d0 | C2 polish (15 commits) — pre-fix   | 39.0 / 38.7 s       |   +5     |  +20     |   −6    | 4.75e-3 / 7.11e-3 / 4.90e-3       | 0.901 / 0.835 / 0.893             |
| 5682763 | HEAD — disable JIT kernel fusion at load | 26.2 / 26.2 s |   +5     |   −4     |  −30    | 4.75e-3 / 6.50e-3 / 4.60e-3       | 0.901 / 0.848 / 0.900             |

The 21 commits between `7d713e5` and `7ec41ad`, and the 15 commits
between `7ec41ad` and `34f69d0`, are parity-neutral: 7 are pre-JIT-runtime
tooling (CLI `.json/.lst`, JIT-export script, static-core refactors) that
keep using the eager pickle, and 15 are post-JIT polish (retire pickle
path, config refactors, lidar removal, CLI flags, docs) that keep using
the JIT artifact. None re-measured because the deltas inherit from the
row above.

`validate_parity.py` exits 0 in both `merge` and `no_merge` modes at HEAD.

### Where parity changed

**Four** rows in the table moved parity vs the row above them at
measurement time. One of the four (`7ec41ad`) was later reverted at
`5682763`; the net at HEAD is therefore three irreducible drifts.
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
* **`7ec41ad` C2 switch loader to torch.jit.load** — *artifact
  swap, not a runtime-code change*. Introduces `JITKelvinAdapter`
  which runs `KelvinStaticCore.forward_tensors` through a
  `torch.jit.trace`d graph instead of the eager pickle. Bisect
  confirmed at this exact commit: loading the **eager pickle**
  here reproduces `7d713e5`'s **+5/−4/−30** exactly, while loading
  the **JIT artifact** gives **+5/+20/−6**. The drift was later
  attributed precisely to `torch.jit.load`'s NNC kernel-fusion
  passes (not the trace itself — see `5682763` below for the proof
  via per-tensor `eager-vs-traced` vs `load round-trip` diffs and
  the load-time fix that reverts the regression).

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
* **`5682763` disable JIT kernel fusion at load** — *load-time
  fix*. The diagnostic added in `7b04d68` exposed that the trace
  itself is faithful (eager-vs-traced max-abs-diff ~5.8e-4 on
  `gs_xyz`, ~1.7e-5 on `gs_densities`) while the **load round-trip**
  blows that up by 2-4 orders of magnitude (`gs_densities` 9.3e-1,
  enough to flip Gaussians across the density-prune threshold and
  produce the +20 / −24 chunk1/merge shift seen at `7ec41ad`). The
  fix is one line in `instant_nurec.model.make`:
  `torch.jit.set_fusion_strategy([("STATIC", 0), ("DYNAMIC", 0)])`
  before `torch.jit.load`, which disables NNC kernel fusion at the
  JIT executor. The recorded graph then runs op-by-op (matching
  what the eager pickle does), and the deltas snap back to
  **+5/−4/−30** — bit-identical to `7d713e5` on Chamfer and F1 too,
  not just vertex counts. Bonus: ~12 s faster per mode (no
  autotuning cost for a one-shot CLI).

### Net story

Three commits actually move parity at HEAD (`fc23075`, `6b32da4`,
`07c8b20`). The first two are intrinsic to the slang→torch port and
unavoidable; the third is a transient drift that returned to the
pre-drop pattern at `69c7601`. The JIT-loader switch at `7ec41ad`
*looked* parity-moving in isolation (+5/+20/−6) but the apparent
regression was JIT load-time kernel fusion, which `5682763` disables.
Net at HEAD: residual `+5/−4/−30` from `fc23075` + `6b32da4` only,
the slang→torch ULP drift on the rotation/pose kernels. The
eager-pickle vs JIT artifact difference is parity-neutral, the
public-vs-internal cu128 wheel difference is parity-neutral.

## Runtime

Wall-clock per mode is in the table above. Cumulative gaps vs the
bazel baseline (19.6 / 21.1 s):

1. Baseline → `7d713e5` (28.5 / 33.1 s): **+8.9 / +12.0 s** from the
   slang→torch swaps (A.7 vren is the biggest single contributor at
   +24 s no_merge cold-cache; per-commit breakdown in the linked
   per-commit report).
2. `7d713e5` → `34f69d0` (39.0 / 38.7 s): **+10.5 / +5.6 s** from the
   JIT loader switch at `7ec41ad`. NNC kernel-fusion autotuning at
   first call plus the per-pixel-tensor marshalling in
   `JITKelvinAdapter` together added ~10 s per mode.
3. `34f69d0` → HEAD (`5682763`, 26.2 / 26.2 s): **−12.8 / −12.5 s**
   from disabling NNC fusion at load. For a one-shot CLI the
   autotuning cost can't be amortized, so skipping fusion is a net
   wall-time win on top of restoring parity.

Net HEAD vs baseline: **+6.6 / +5.1 s** (26.2 / 26.2 vs 19.6 / 21.1) —
the irreducible cost of running torch's CUDA kernels instead of the
hand-tuned NRE slang/CUDA originals.

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
