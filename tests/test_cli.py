"""Tests for instant_nurec.cli.

The CLI is the user-facing flag surface for the standalone Kelvin predict
pipeline. After the Phase 1 step 4.4 hydra strip, ``main`` constructs an
:class:`NRMConfig` directly via ``instant_nurec.config.load_predict_config``
and hands it to ``nre.nrm.run.run_predict`` -- no Hydra overrides involved.

The lazy imports inside ``main`` are stubbed via ``sys.modules`` so this
suite does not require NRE's runtime deps to be installed in the test venv.
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


def _install_runtime_stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Inject fake ``instant_nurec.config`` + ``nre.nrm.run`` modules so
    cli.main()'s lazy imports resolve without pulling in NRE/torch."""
    config_mod = types.ModuleType("instant_nurec.config")
    fake_config = MagicMock(name="NRMConfig")
    fake_load = MagicMock(return_value=fake_config)
    config_mod.load_predict_config = fake_load

    nre_mod = types.ModuleType("nre")
    nrm_mod = types.ModuleType("instant_nurec._pkg.nrm")
    run_mod = types.ModuleType("instant_nurec._pkg.nrm.run")
    fake_run_predict = MagicMock(return_value=None)
    run_mod.run_predict = fake_run_predict

    monkeypatch.setitem(sys.modules, "instant_nurec.config", config_mod)
    monkeypatch.setitem(sys.modules, "nre", nre_mod)
    monkeypatch.setitem(sys.modules, "instant_nurec._pkg.nrm", nrm_mod)
    monkeypatch.setitem(sys.modules, "instant_nurec._pkg.nrm.run", run_mod)
    return fake_load, fake_run_predict


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


# ---------- end-to-end main() with runtime stubbed ----------


def test_main_no_merge_passes_disabled_to_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_load, fake_run_predict = _install_runtime_stubs(monkeypatch)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", "/d", "--output-dir", "/o"])
    assert rc == 0
    fake_load.assert_called_once()
    kwargs = fake_load.call_args.kwargs
    assert kwargs["ncore_path"] == Path("/d")
    assert kwargs["output_dir"] == Path("/o")
    assert kwargs["merge_enabled"] is False
    fake_run_predict.assert_called_once_with(fake_load.return_value)


def test_main_frustum_ownership_passes_enabled_to_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_load, fake_run_predict = _install_runtime_stubs(monkeypatch)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", "/d", "--output-dir", "/o", "--merge", "frustum-ownership"])
    assert rc == 0
    assert fake_load.call_args.kwargs["merge_enabled"] is True
    fake_run_predict.assert_called_once_with(fake_load.return_value)


def test_main_configures_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_runtime_stubs(monkeypatch)
    captured: dict[str, object] = {}
    real_basic_config = logging.basicConfig

    def fake_basic_config(**kwargs: object) -> None:
        captured.update(kwargs)
        real_basic_config()

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    from instant_nurec.cli import main
    main(["--ncore-path", "/d", "--output-dir", "/o", "--log-level", "DEBUG"])
    assert captured.get("level") == logging.DEBUG


def test_main_returns_zero_on_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_runtime_stubs(monkeypatch)
    from instant_nurec.cli import main
    rc = main(["--ncore-path", "/d", "--output-dir", "/o"])
    assert rc == 0
