<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contributing to instant_nurec

This is the standalone reference implementation of the NRM Kelvin predict
pipeline. Most contributions fix parity regressions, simplify the predict
path, or extend the test surface.

## Local setup

```bash
./setup.sh                # creates .venv and installs runtime + dev deps
source .venv/bin/activate
```

`setup.sh` uses `pip install -e .`; the build backend is plain
`setuptools`. There is no separate build step — every kernel is pure
torch, no compiled extensions to build.

## Running tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Branch coverage is the bar for new functions. Please add a test (or set
of tests) covering each branch of any new code.

## Linting

```bash
.venv/bin/ruff check .
```

The `ruff` config lives in `pyproject.toml` (`[tool.ruff]`). Please keep
lint clean before opening an MR.

## Parity verification

Any change that touches the predict path must keep parity against the
baselines in `baselines/original_baseline/`. Both modes must hold:

```bash
# unsandboxed (GPU required) — produces fresh PLYs
mkdir -p /tmp/parity/no_merge /tmp/parity/merge
python run_inference.py --ncore-path /storage/data/nurec/ncorev4 --output-dir /tmp/parity/no_merge --merge none
python run_inference.py --ncore-path /storage/data/nurec/ncorev4 --output-dir /tmp/parity/merge --merge frustum-ownership

# sandboxed
python scripts/validate_parity.py merge \
    baselines/original_baseline/merge/oEvmtCL5U5aiZZrLcLgmBm/ply/pai_*/pai_*.ply \
    /tmp/parity/merge/*/ply/*/*.ply
python scripts/validate_parity.py no_merge \
    baselines/original_baseline/no_merge/e78RJgNGViMA3hsJoQXYVx/ply/pai_*/ \
    /tmp/parity/no_merge/*/ply/*/
```

Both must exit 0 within `tests/tolerance.json`. The tolerance file is
the run-to-run noise floor of the original NRE pipeline (C(5,2)=10
pairwise comparisons across `baselines/more_baselines/run_{1..5}`).
Tolerance can ratchet upward (per-property), never downward, and only for
documented CUDA→torch swaps. Bumps must be justified in the commit body.

## Commit hygiene

- One logical change per commit.
- Subject line: `<type>(<area>): <imperative one-liner>`. Types in use:
  `feat`, `fix`, `refactor`, `chore`, `test`, `docs`.
- For changes ported from NRE, reference the source. For self-invented
  fixes (no NRE equivalent), include `(self-invented: <reason>)`.
- Pre-commit hooks (`.pre-commit-config.yaml`) run `ruff` and basic
  whitespace checks; please don't bypass them.

## What stays out of the repo

- `kelvin_full.pt` (large pickle; produced via `INSTANT_NUREC_FULL_PT`).
- `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage` (build artifacts).
- IDE state (`.idea/`, `.vscode/`).
- Container shell config (`.bashrc`, `.zshrc`, …).

These are all in `.gitignore`.
