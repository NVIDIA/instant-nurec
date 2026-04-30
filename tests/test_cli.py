"""Tests for instant_nurec.cli.

The CLI is the new flag surface that replaces NRE's `bazel run //nre/nrm:run --
--config-name=... +<hydra-overrides>` invocation. Phase 1 Step 3 only introduces
the argparse layer; under the hood we still delegate to NRE's `nre.nrm.run.main`
click command, so subsequent strips (Phase 1.4 onwards) can excise NRE
dependencies without rewriting the user-facing surface.

NRE imports are stubbed via sys.modules so this suite does not require NRE's
runtime deps to be installed in the test venv.
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


def _install_nre_stub(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``nre.nrm.run`` so cli.main()'s lazy import resolves."""
    nre_mod = types.ModuleType("nre")
    nrm_mod = types.ModuleType("nre.nrm")
    run_mod = types.ModuleType("nre.nrm.run")
    fake_callback = MagicMock(return_value=None)
    fake_main = MagicMock()
    fake_main.callback = fake_callback
    run_mod.main = fake_main
    monkeypatch.setitem(sys.modules, "nre", nre_mod)
    monkeypatch.setitem(sys.modules, "nre.nrm", nrm_mod)
    monkeypatch.setitem(sys.modules, "nre.nrm.run", run_mod)
    return fake_callback


# ---------- argparse surface ----------


def test_parser_default_merge_is_none() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(["--ncore-path", "/x", "--output-dir", "/y"])
    assert args.merge == "none"


def test_parser_default_log_level_is_info() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(["--ncore-path", "/x", "--output-dir", "/y"])
    assert args.log_level == "INFO"


def test_parser_accepts_merge_none() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(
        ["--ncore-path", "/x", "--output-dir", "/y", "--merge", "none"]
    )
    assert args.merge == "none"


def test_parser_accepts_merge_frustum_ownership() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(
        ["--ncore-path", "/x", "--output-dir", "/y", "--merge", "frustum-ownership"]
    )
    assert args.merge == "frustum-ownership"


def test_parser_rejects_unknown_merge() -> None:
    from instant_nurec.cli import make_parser
    with pytest.raises(SystemExit):
        make_parser().parse_args(
            ["--ncore-path", "/x", "--output-dir", "/y", "--merge", "frobnicate"]
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


# ---------- Hydra-override mapping ----------


def test_overrides_no_merge_disables_primitive_merge() -> None:
    from instant_nurec.cli import hydra_overrides, make_parser
    args = make_parser().parse_args(
        ["--ncore-path", "/data", "--output-dir", "/out", "--merge", "none"]
    )
    overrides = hydra_overrides(args)
    assert "predict.primitive_merge.enabled=false" in overrides
    assert not any(
        "predict.primitive_merge.enabled=true" in o for o in overrides
    )
    assert not any(
        "predict.primitive_merge.overlap_strategy=" in o for o in overrides
    )


def test_overrides_frustum_ownership_enables_merge_with_strategy() -> None:
    from instant_nurec.cli import hydra_overrides, make_parser
    args = make_parser().parse_args(
        ["--ncore-path", "/d", "--output-dir", "/o", "--merge", "frustum-ownership"]
    )
    overrides = hydra_overrides(args)
    assert "predict.primitive_merge.enabled=true" in overrides
    assert "predict.primitive_merge.overlap_strategy=frustum_ownership" in overrides


def test_overrides_paths_and_constants() -> None:
    from instant_nurec.cli import hydra_overrides, make_parser
    args = make_parser().parse_args(
        ["--ncore-path", "/data", "--output-dir", "/out"]
    )
    overrides = hydra_overrides(args)
    assert "dataset.predict.ncore_json_base_path=/data" in overrides
    assert "dataset.predict.ncore_json_list_path=/data/debug.lst" in overrides
    assert "out_dir=/out" in overrides
    assert "predict.render_video.enabled=false" in overrides
    assert "+nrm/apps/options=_kelvin_predict" in overrides
    assert "dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov" in overrides


# ---------- end-to-end main() with NRE stubbed ----------


def test_main_calls_nre_main_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = _install_nre_stub(monkeypatch)
    from instant_nurec.cli import CONFIG_NAME, main
    rc = main(["--ncore-path", "/d", "--output-dir", "/o"])
    assert rc == 0
    callback.assert_called_once()
    kwargs = callback.call_args.kwargs
    assert kwargs["config_name"] == CONFIG_NAME
    assert "predict.primitive_merge.enabled=false" in kwargs["hydra_args"]


def test_main_passes_frustum_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = _install_nre_stub(monkeypatch)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", "/d", "--output-dir", "/o", "--merge", "frustum-ownership"])
    assert rc == 0
    args = callback.call_args.kwargs["hydra_args"]
    assert "predict.primitive_merge.enabled=true" in args
    assert "predict.primitive_merge.overlap_strategy=frustum_ownership" in args


def test_main_configures_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nre_stub(monkeypatch)
    captured = {}
    real_basic_config = logging.basicConfig

    def fake_basic_config(**kwargs: object) -> None:
        captured.update(kwargs)
        real_basic_config()

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    from instant_nurec.cli import main
    main(["--ncore-path", "/d", "--output-dir", "/o", "--log-level", "DEBUG"])
    assert captured.get("level") == logging.DEBUG


def test_main_returns_zero_on_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nre_stub(monkeypatch)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", "/d", "--output-dir", "/o"])
    assert rc == 0
