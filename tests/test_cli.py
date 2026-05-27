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

"""Tests for ``instant_nurec.cli``.

The CLI builds an ``InstantNuRecConfig`` directly from the pydantic schemas
in ``config_schema/`` and hands it to ``predict.run.run_predict``. We stub
``predict.run`` so the test doesn't need GPU, then inspect the constructed
``InstantNuRecConfig``.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _install_runtime_stubs(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub ``instant_nurec.predict.run`` so cli.main()'s lazy import
    resolves without pulling in the full predict path."""
    run_mod = types.ModuleType("instant_nurec.predict.run")
    fake_run_predict = MagicMock(return_value=None)
    run_mod.run_predict = fake_run_predict
    monkeypatch.setitem(sys.modules, "instant_nurec.predict.run", run_mod)
    return fake_run_predict


# ---------- argparse surface ----------


def test_parser_default_merge_is_false() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(["--ncore-path", "/x", "--output-dir", "/y"])
    assert args.merge is False


def test_parser_default_n_gaussians() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(["--ncore-path", "/x", "--output-dir", "/y"])
    assert args.n_gaussians == 2_000_000
    # The voxel-size and voxelization flags are no longer part of the CLI surface.
    assert not hasattr(args, "voxel_size")
    assert not hasattr(args, "voxelization")


def test_parser_accepts_explicit_n_gaussians() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(
        ["--ncore-path", "/x", "--output-dir", "/y", "--n-gaussians", "500000"]
    )
    assert args.n_gaussians == 500000


def test_parser_rejects_non_int_n_gaussians() -> None:
    from instant_nurec.cli import make_parser
    with pytest.raises(SystemExit):
        make_parser().parse_args(
            ["--ncore-path", "/x", "--output-dir", "/y", "--n-gaussians", "many"]
        )


def test_parser_no_longer_accepts_voxel_size() -> None:
    """The old --voxel-size flag must error so we don't silently ignore it."""
    from instant_nurec.cli import make_parser
    with pytest.raises(SystemExit):
        make_parser().parse_args(
            ["--ncore-path", "/x", "--output-dir", "/y", "--voxel-size", "0.25"]
        )


def test_parser_default_log_level_is_info() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(["--ncore-path", "/x", "--output-dir", "/y"])
    assert args.log_level == "INFO"


def test_parser_merge_flag_sets_true() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(
        ["--ncore-path", "/x", "--output-dir", "/y", "--merge"]
    )
    assert args.merge is True


def test_parser_merge_no_longer_takes_choice_argument() -> None:
    """The old `--merge {none, frustum-ownership}` form must error so we
    don't silently treat 'frustum-ownership' as a positional argument."""
    from instant_nurec.cli import make_parser
    with pytest.raises(SystemExit):
        make_parser().parse_args(
            ["--ncore-path", "/x", "--output-dir", "/y", "--merge", "frustum-ownership"]
        )


def test_parser_accepts_explicit_log_level() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(
        ["--ncore-path", "/x", "--output-dir", "/y", "--log-level", "DEBUG"]
    )
    assert args.log_level == "DEBUG"


def test_parser_rejects_unknown_log_level() -> None:
    from instant_nurec.cli import make_parser
    with pytest.raises(SystemExit):
        make_parser().parse_args(
            ["--ncore-path", "/x", "--output-dir", "/y", "--log-level", "TRACE"]
        )


def test_parser_requires_ncore_path() -> None:
    from instant_nurec.cli import make_parser
    with pytest.raises(SystemExit):
        make_parser().parse_args(["--output-dir", "/y"])


def test_parser_requires_output_dir() -> None:
    from instant_nurec.cli import make_parser
    with pytest.raises(SystemExit):
        make_parser().parse_args(["--ncore-path", "/x"])


# ---------- end-to-end main() with runtime stubbed ----------


def _make_json_path(tmp_path: Path) -> Path:
    p = tmp_path / "seq.json"
    p.write_text("{}")
    return p


def test_main_no_merge_constructs_config_with_disabled_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", str(json_path), "--output-dir", "/o"])
    assert rc == 0
    fake_run_predict.assert_called_once()
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.out_dir == "/o"
    assert cfg.dataset.predict.ncore_json_paths == [str(json_path.resolve())]
    assert cfg.predict.primitive_merge.enabled is False


def test_main_lst_path_resolves_each_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A ``.lst`` input is resolved into a list of JSON paths in the config."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("{}")
    b.write_text("{}")
    lst = tmp_path / "all.lst"
    lst.write_text(f"{a}\nb.json\n")  # one absolute, one relative-to-lst-dir

    fake_run_predict = _install_runtime_stubs(monkeypatch)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", str(lst), "--output-dir", "/o"])
    assert rc == 0
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.dataset.predict.ncore_json_paths == [
        str(a.resolve()),
        str(b.resolve()),
    ]


def test_main_merge_flag_constructs_config_with_enabled_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", str(json_path), "--output-dir", "/o", "--merge"])
    assert rc == 0
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.predict.primitive_merge.enabled is True


def test_main_no_merge_disables_voxelization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", str(json_path), "--output-dir", "/o"])
    assert rc == 0
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.predict.primitive_merge.enable_voxelization is False
    # Default target carries through even when voxelization is disabled.
    assert cfg.predict.primitive_merge.target_n_gaussians == 2_000_000


def test_main_merge_enables_voxelization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """--merge always enables voxelization (bundled).

    --n-gaussians propagates to ``target_n_gaussians``; the initial
    ``voxel_size`` stays at its config default (0.1) since the iteration
    discovers the converged value.
    """
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    rc = main([
        "--ncore-path", str(json_path),
        "--output-dir", "/o",
        "--merge",
        "--n-gaussians", "500000",
    ])
    assert rc == 0
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.predict.primitive_merge.enabled is True
    assert cfg.predict.primitive_merge.enable_voxelization is True
    assert cfg.predict.primitive_merge.target_n_gaussians == 500000
    assert cfg.predict.primitive_merge.voxel_size == 0.1


def test_main_configures_log_level(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    captured: dict[str, object] = {}
    real_basic_config = logging.basicConfig

    def fake_basic_config(**kwargs: object) -> None:
        captured.update(kwargs)
        real_basic_config()

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    from instant_nurec.cli import main
    main(["--ncore-path", str(json_path), "--output-dir", "/o", "--log-level", "DEBUG"])
    assert captured.get("level") == logging.DEBUG


def test_main_returns_zero_on_clean_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", str(json_path), "--output-dir", "/o"])
    assert rc == 0


def test_main_prints_refine_link_when_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", str(json_path), "--output-dir", "/o", "--merge"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Next: refine into USDZ with NuRec" in out
    assert "https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html" in out
    assert "SuperSplat" not in out


def test_main_prints_viewer_hint_when_no_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", str(json_path), "--output-dir", "/o"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Next: view your 3DGS PLY with SuperSplat" in out
    assert "https://playcanvas.com/supersplat/editor" in out
    assert "ply_viewer (NuRec container)" in out
    assert "refine into USDZ" not in out


def test_main_unrecognised_suffix_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_runtime_stubs(monkeypatch)
    bogus = tmp_path / "data.yaml"
    bogus.write_text("{}")
    from instant_nurec.cli import main
    with pytest.raises(ValueError, match="must end in .json or .lst"):
        main(["--ncore-path", str(bogus), "--output-dir", "/o"])
