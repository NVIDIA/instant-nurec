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
Wasserstein-1) are summarized in `internal/parity_proofs/per_commit_quality_runtime.md`.
HEAD numbers vs baseline:

| mode      | Δ vertices | Chamfer (½ sum, PLY units) | F1@0.01 |
|-----------|------------|----------------------------|---------|
| chunk0    | +5         | 4.75e-3                    | 0.901   |
| chunk1    | −4         | 6.50e-3                    | 0.848   |
| merge     | −30        | 4.60e-3                    | 0.900   |

`validate_parity.py` exits 0 in both `merge` and `no_merge` modes.

### Note on intermediate parity transitions

Three commits in the chain showed parity moving and then returning:

* `07c8b20` (drop bazel) shifted the deltas from +5/−4/−30 to
  −2/+7/−21. The commit body originally attributed this to the public
  `torch==2.7.0+cu128` wheel replacing NRE's internal cu128 build.
  This was wrong: dropping bazel also reset the entire transitive
  Python dep tree (numpy, scipy, pandas, zarr, numcodecs, nvidia-ncore
  jumped past NRE's bazel-pinned versions because we hadn't pinned
  them).
* `69c7601` migrated to uv with loose pins. Parity stayed near
  −2/+7/−21.
* `7d713e5` tightened uv pins to NRE-bazel exact (`scipy==1.11.1`,
  `numcodecs==0.11.0`, `pyyaml==6.0`, `huggingface_hub==0.36.2`,
  `numpy==1.26.4`, `nvidia-ncore==18.7.0`, `pandas==2.3.3` on
  Python 3.11.15). Parity returned to +5/−4/−30, identical to the
  pre-bazel-drop A.5–A.8 pattern.

So the residual drift vs the bazel reference is +5/−4/−30, attributable
to the slang→torch swaps in Phase A. The public-vs-internal cu128
wheel difference is effectively parity-neutral when the rest of the
deps match NRE.

## Runtime

| commit                       | wall (no_merge / merge) |
|------------------------------|-------------------------|
| baseline (bazel reference)   | 19.6 s / 21.1 s         |
| HEAD (pure torch + cu128)    | 28.5 s / 33.1 s         |

The ~12 s no_merge / ~11 s merge gap is the cumulative cost of the
slang→torch swaps; the per-commit attribution (A.7 vren is the
biggest single contributor) is in the linked report.

## Reproduction

```
./setup.sh
source .venv/bin/activate

mkdir -p /tmp/out/no_merge /tmp/out/merge
python run_inference.py --ncore-path /storage/data/nurec/ncorev4 \
    --output-dir /tmp/out/no_merge --merge none
python run_inference.py --ncore-path /storage/data/nurec/ncorev4 \
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
