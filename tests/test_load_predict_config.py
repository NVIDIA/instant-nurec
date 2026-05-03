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

"""Tests for ``instant_nurec.config.load_predict_config``.

After the final dead-code pass the loader is purely about taking the
inline ``_PREDICT_CONFIG`` dict, patching the CLI-derived fields, and
validating the result against ``NRMConfig``. Pretrained-weights
resolution moved out of this module entirely (now lives in
``model.make()`` via the HF mock), so there are no
``_resolve_pretrained_checkpoint`` / ``create_model_registry`` paths
left to test here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


pytest.importorskip("yaml")
pytest.importorskip("pydantic")


@pytest.fixture
def fake_ncore_root(tmp_path: Path) -> Path:
    """Create an ncore root with a debug.lst (the loader stats both)."""
    ncore = tmp_path / "ncore"
    ncore.mkdir()
    (ncore / "debug.lst").write_text("seq_a/meta.json\n")
    return ncore


def test_merge_enabled_propagates_to_predict_config(
    fake_ncore_root: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=True,
    )
    assert cfg.predict.primitive_merge.enabled is True


def test_merge_disabled_propagates_to_predict_config(
    fake_ncore_root: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    assert cfg.predict.primitive_merge.enabled is False


def test_ncore_path_maps_to_base_and_list_paths(
    fake_ncore_root: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    assert cfg.dataset.predict is not None
    assert cfg.dataset.predict.ncore_json_base_path == str(fake_ncore_root)
    assert cfg.dataset.predict.ncore_json_list_path == str(
        fake_ncore_root / "debug.lst"
    )


def test_output_dir_maps_to_out_dir(fake_ncore_root: Path, tmp_path: Path):
    from instant_nurec.config import load_predict_config

    out = tmp_path / "out"
    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=out,
        merge_enabled=False,
    )
    assert cfg.out_dir == str(out)


def test_config_dir_is_auto_derived_from_out_dir_and_run_id(
    fake_ncore_root: Path, tmp_path: Path
):
    from instant_nurec.config import load_predict_config

    out = tmp_path / "out"
    cfg = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=out,
        merge_enabled=False,
    )
    assert cfg.config_dir == str(out / cfg.run_id / "config")


def test_run_ids_are_unique_per_invocation(fake_ncore_root: Path, tmp_path: Path):
    """Each load_predict_config() must mint a fresh run_id (shortuuid default_factory)."""
    from instant_nurec.config import load_predict_config

    cfg1 = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    cfg2 = load_predict_config(
        ncore_path=fake_ncore_root,
        output_dir=tmp_path / "out",
        merge_enabled=False,
    )
    assert cfg1.run_id != cfg2.run_id
