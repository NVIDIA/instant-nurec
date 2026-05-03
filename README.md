<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# instant_nurec

Standalone reference implementation of the NRM Kelvin predict pipeline:
ncorev4 ingest → frame batch prep → Kelvin predict → 3D-Gaussian PLY export.

This is the predict-only carve-out of the NRM Kelvin model from the NRE
codebase (commit `a54a6af`); training, validation, and the rendering /
loss / supervision paths are not included.

## What it does

Given a directory of ncorev4 sequences and a PLY output directory, the
pipeline runs Kelvin inference and writes either:

- one merged PLY per sequence (`--merge frustum-ownership`), or
- one PLY per chunk (`--merge none`).

Both modes are bit-for-bit (within the determinism tolerance in
`tests/tolerance.json`) reproducible against the reference baselines under
`baselines/original_baseline/`.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a Python venv, installs the pinned CUDA torch build
(`torch==2.7.0+cu128` from `https://download.pytorch.org/whl/cu128`),
then `pip install -e .`. All other runtime deps are pure Python / torch
wheels — no bazel, no compiled kernels, no `nvdiffrast` / `gsplat` /
`torch_scatter`.

The `torch` pin matches the version used by the NRE bazel build at
commit `a54a6af` (NRE pinned `torch==2.7.0+cu128.gitc41f6e01287` via the
NVIDIA-internal index; `2.7.0+cu128` from the public PyTorch index is
the closest publicly-installable equivalent). This keeps the
parity-vs-baseline drift attributable to a known torch+CUDA version.
Updating the pin is a deliberate, parity-gated change.

## Pretrained model

The pipeline loads a single artifact: a torch-pickled
`GaussiansNRMSystem` saved as `kelvin_full.pt`. There are exactly two
ways to make it available:

1. **HuggingFace cache** (eventual default): the placeholder repo id is
   `nvidia/instant-nurec-kelvin`; the in-tree mock at
   `instant_nurec/_hf_mock.py` resolves it to
   `~/.cache/instant_nurec/kelvin_full.pt`. When the corp publishes the
   real repo, set `INSTANT_NUREC_HF_MOCK=0` and `huggingface_hub`
   downloads it for you.

2. **Manual override**: set `INSTANT_NUREC_FULL_PT` to a local path. The
   first time the pipeline runs with that env var set, the file is
   copied into the HF cache so subsequent runs find it through path
   (1) automatically.

If neither path resolves a `.pt` file, the pipeline raises
`FullModelNotFoundError` with a clear message rather than attempting any
other download.

## Quickstart

The two canonical invocations:

```bash
# No-merge mode: writes per-chunk PLYs.
./run.sh \
    --ncore-path /path/to/ncorev4 \
    --output-dir /tmp/out/no_merge \
    --merge none

# Merge mode: writes a single merged PLY per sequence.
./run.sh \
    --ncore-path /path/to/ncorev4 \
    --output-dir /tmp/out/merge \
    --merge frustum-ownership
```

`run.sh` validates the inputs and execs `python run_inference.py`. You
can also call the CLI directly:

```bash
python run_inference.py \
    --ncore-path /path/to/ncorev4 \
    --output-dir /tmp/out \
    --merge none
```

## CLI reference

| flag | purpose |
| --- | --- |
| `--ncore-path` | ncorev4 dataset root containing `debug.lst`. Required. |
| `--output-dir` | Directory the pipeline writes PLYs (and the resolved config) into. Required. |
| `--merge` | `none` (default) for per-chunk PLYs, `frustum-ownership` for a single merged PLY. |
| `--log-level` | `DEBUG` / `INFO` (default) / `WARNING` / `ERROR` / `CRITICAL`. |

## Environment variables

| variable | purpose |
| --- | --- |
| `INSTANT_NUREC_FULL_PT` | Absolute path to a local `kelvin_full.pt`. Takes priority over the HF cache. |
| `INSTANT_NUREC_HF_MOCK` | `1` (default) selects the in-tree placeholder mock; `0` forwards through to real `huggingface_hub`. |

## Validating parity

Round-trip output against the reference baselines:

```bash
python benchmark/validate_parity.py merge \
    baselines/original_baseline/merge/*/ply/*/pai_*.ply \
    /tmp/out/merge/*/ply/*/*.ply

python benchmark/validate_parity.py no_merge \
    baselines/original_baseline/no_merge/*/ply/*/ \
    /tmp/out/no_merge/*/ply/*/
```

Both must exit 0 within `tests/tolerance.json`. The tolerance file is
derived from the run-to-run noise floor of the original NRE pipeline
(C(5,2)=10 pairwise comparisons across `baselines/more_baselines/run_{1..5}`).

## Repository layout

```
instant_nurec/                  # standalone package
    __init__.py
    cli.py                      # argparse entrypoint
    config.py                   # static NRMConfig literal + load_predict_config
    _hf_mock.py                 # placeholder HF resolver (Phase 4 step 9)
    config_schema/              # pydantic schemas for NRMConfig
    datasets/                   # ncorev4 ingest + cuboid-track helpers
    model/                      # GaussiansNRMSystem + KelvinNRM + blocks/backbone
    predict/                    # predict loop + PLY export + frustum-ownership merge
    primitives/                 # KelvinNRMPrimitive
    utils/                      # batch / geometry / sensors / nn-extensions
benchmark/
    validate_parity.py          # torch-backed PLY parity check
    derive_determinism_tolerance.py
tests/                          # branch-coverage tests
    tolerance.json
baselines/                      # reference PLYs + parsed configs (not modified)
data_samples/                   # ncorev4 fixture placeholder (HF mock target)
internal/                       # migration scaffolding — runtime is decoupled
    plans/                      # plan.md + plan2.md (project history)
    parity_proofs/              # parity-proof notes
run_inference.py                # canonical Python entrypoint
run.sh                          # input-validation wrapper
setup.sh                        # venv bootstrap
pyproject.toml                  # setuptools build + ruff config
```

## Development

```bash
# Run the unit tests (CPU-only, no GPU required for the suite — runtime parity
# tests are GPU-only and live in the iter loop).
.venv/bin/python -m pytest tests/ -q

# Lint:
.venv/bin/ruff check .
```

## Provenance

Code adapted from the NRE repository at commit `a54a6af`. All compiled
CUDA/slang kernels from that source tree (`libs/geometry/`,
`libs/sensors/`, `libs/vren/`, `libs/packed_ops/`) were replaced with
pure-torch equivalents (Phase A); bazel was dropped (Phase B); the
package was flattened to asset-harvester shape (Phase C); the
NGC-checkpoint download path was replaced by the single-`.pt` HF flow
described above (final dead-code pass).
