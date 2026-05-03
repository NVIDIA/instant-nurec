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

"""``docs/internal/`` is migration scaffolding (parity proofs, NRE provenance);
plan.md §8.3 says ``rm -rf`` of that tree must not break the runtime
package. This test renames the directory aside, drops cached imports,
imports ``instant_nurec.cli`` + the ``run_predict`` entrypoint, and
asserts both succeed. The tree is restored on teardown.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


_INTERNAL_DIR = REPO_ROOT / "docs" / "internal"


@pytest.fixture
def internal_renamed_aside(tmp_path: Path):
    """Move ``docs/internal/`` to a tmp location for the duration of the test;
    restore it on teardown regardless of test outcome."""
    if not _INTERNAL_DIR.exists():
        # nothing to rename — the test is trivially satisfied because the
        # runtime already imports without docs/internal/.
        yield
        return

    backup = tmp_path / "internal_backup"
    shutil.move(str(_INTERNAL_DIR), str(backup))
    try:
        yield
    finally:
        # Restore — `move` (not `copy`) avoided pulling parity_proofs through
        # tmp_path's filesystem if it sat on a different device.
        if _INTERNAL_DIR.exists():
            shutil.rmtree(_INTERNAL_DIR)
        shutil.move(str(backup), str(_INTERNAL_DIR))


@pytest.fixture
def isolated_instant_nurec_modules():
    """Snapshot ``sys.modules`` for everything under ``instant_nurec`` and
    drop them so the test gets a fresh import; restore the snapshot on
    teardown so other tests in the suite see the same cached objects they
    were holding references to."""
    saved = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "instant_nurec" or name.startswith("instant_nurec.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        # Drop whatever the test imported and restore the original cache so
        # later tests (e.g. test_model_registry) keep their captured
        # ``isinstance`` / class identities.
        for name in list(sys.modules):
            if name == "instant_nurec" or name.startswith("instant_nurec."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_runtime_imports_without_docs_internal(
    internal_renamed_aside, isolated_instant_nurec_modules
):
    """Importing the runtime CLI surface succeeds with ``docs/internal/``
    renamed away."""
    cli_mod = importlib.import_module("instant_nurec.cli")
    assert callable(cli_mod.main), "instant_nurec.cli.main should still be a callable entrypoint"

    # The argparse parser must build cleanly — this exercises the lazy-import
    # path where importing the package alone isn't sufficient if a runtime
    # dep accidentally pulled docs/internal/ at module load time.
    parser = cli_mod.make_parser()
    args = parser.parse_args(["--ncore-path", "/tmp/none", "--output-dir", "/tmp/none"])
    assert args.merge == "none"  # default

    # ``run_predict`` is only loaded inside ``main()``; force-import it here
    # so we catch any docs/internal/ dependency on that side too.
    run_mod = importlib.import_module("instant_nurec.predict.run")
    assert callable(run_mod.run_predict)


def test_internal_dir_is_documented_only_under_docs():
    """Sanity: there is no top-level ``internal/`` package; the migration
    scaffolding lives under ``docs/internal/`` exactly per plan §D.5."""
    top_level_internal = REPO_ROOT / "internal"
    assert not top_level_internal.exists(), (
        f"Top-level internal/ must not be present (D.5). Found: {top_level_internal}"
    )
