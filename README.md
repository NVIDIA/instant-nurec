<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: LicenseRef-NvidiaProprietary -->

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

`setup.sh` creates a Python venv and installs `numpy`/`torch`/`plyfile`
and the rest of the runtime deps. The compiled CUDA/slang kernels under
`libs/` are still built via bazel during the Phase 3 transition; the
pure-`python run_inference.py` form will be self-contained once Phase 2
finishes replacing them with torch-native equivalents.

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

`run.sh` validates the inputs and execs `python run_inference.py`. You can
also call the CLI directly:

```bash
python run_inference.py \
    --ncore-path /path/to/ncorev4 \
    --output-dir /tmp/out \
    --merge none
```

While the bazel transition is in flight the equivalent legacy launcher is:

```bash
bazel run //instant_nurec:run -- \
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
| `INSTANT_NUREC_FULL_PT` | When set to a path that exists, the pipeline torch-loads the full pickled `GaussiansNRMSystem` from there and skips the constructor + checkpoint state-dict load. When the path does not exist, the pipeline builds via the constructor and writes the system to that path so the next run takes the fast path. |
| `INSTANT_NUREC_HF_MOCK` | (Phase 4 placeholder) When set to `1` (default), `_hf_mock` resolves the HF placeholder repo `nvidia/instant-nurec-kelvin` to a local cached path. |

## Validating parity

Round-trip output against the reference baselines:

```bash
python scripts/validate_parity.py merge \
    baselines/original_baseline/merge/*/ply/*/pai_*.ply \
    /tmp/out/merge/*/ply/*/*.ply

python scripts/validate_parity.py no_merge \
    baselines/original_baseline/no_merge/*/ply/*/ \
    /tmp/out/no_merge/*/ply/*/
```

Both must exit 0 within `tests/tolerance.json`. The tolerance file is
derived from the run-to-run noise floor of the original NRE pipeline
(C(5,2)=10 pairwise comparisons across `baselines/more_baselines/run_{1..5}`).

## Repository layout

```
instant_nurec/                  # standalone package
    cli.py                      # argparse entrypoint
    config.py                   # static NRMConfig literal + load_predict_config
    _pkg/                       # ported Kelvin pipeline (Phase 1.5 rename target: flatten)
        nrm/                    # Kelvin model + datasets + predict driver
        utils/                  # batch / geometry / sensors / model registry / ncore helpers
        datasets/               # cuboid track helpers
        models/                 # PLY writer + nn extensions
        config/                 # base pydantic schema
libs/                           # compiled CUDA/slang kernels (bazel-built; Phase 3 transition)
scripts/
    validate_parity.py          # torch-backed PLY parity check
    derive_determinism_tolerance.py
tests/                          # branch-coverage tests (96% line coverage)
    tolerance.json
baselines/                      # reference PLYs + parsed configs (not modified)
run_inference.py                # canonical Python entrypoint
run.sh                          # input-validation wrapper
setup.sh                        # venv bootstrap
pyproject.toml                  # setuptools build + ruff/pyright config
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

Code adapted from the NRE repository at commit `a54a6af`. The Kelvin model
weights, the rolling-shutter sensor kernels (`libs/sensors/`), the
ray-cuboid intersection (`libs/vren/`), and the SE(3) pose helpers
(`libs/geometry/`) all originate there.
