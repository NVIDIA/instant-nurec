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

#### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

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

## End-to-end inference / parity

Final correctness checks require a GPU and a real ncorev4 clip.

## Commit hygiene

- One logical change per commit.
- Subject line: `<type>(<area>): <imperative one-liner>`. Types in use:
  `feat`, `fix`, `refactor`, `chore`, `test`, `docs`.
- Use `git commit --fixup=<SHA>` to amend an earlier commit on the same
  branch; otherwise a fresh commit per logical change.
