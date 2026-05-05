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

This is the reference implementation of the InstantNuRec Kelvin predict
pipeline. Most contributions fix bugs, simplify the predict path, or
extend the test surface.

## Local setup

```bash
./setup.sh                # uv sync --frozen → .venv/
source .venv/bin/activate
```

`setup.sh` calls `uv sync --frozen`, which installs the locked
dependency tree from `uv.lock`. The build backend is plain `setuptools`
and every kernel is pure torch, so there are no compiled extensions to
build.

To bump a dependency, edit `pyproject.toml` and run `uv lock` to
regenerate the lockfile, then commit both files together.

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

## Commit hygiene

- One logical change per commit.
- Subject line: `<type>(<area>): <imperative one-liner>`. Types in use:
  `feat`, `fix`, `refactor`, `chore`, `test`, `docs`.
