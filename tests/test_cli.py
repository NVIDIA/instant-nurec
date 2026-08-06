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


def test_parser_defaults_to_pa_front() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args(["--ncore-path", "/x", "--output-dir", "/y"])
    assert args.model == "pa-front"
    assert args.camera_ids is None


def test_parser_accepts_multiview_and_repeated_camera_ids() -> None:
    from instant_nurec.cli import make_parser
    args = make_parser().parse_args([
        "--model", "pa-multiview",
        "--ncore-path", "/x",
        "--output-dir", "/y",
        "--camera-id", "front",
        "--camera-id", "left",
        "--camera-id", "right",
    ])
    assert args.model == "pa-multiview"
    assert args.camera_ids == ["front", "left", "right"]


def test_parser_accepts_point_query_profile() -> None:
    from instant_nurec.cli import make_parser

    args = make_parser().parse_args(
        ["--model", "pq-front", "--ncore-path", "/x", "--output-dir", "/y"]
    )
    assert args.model == "pq-front"


def test_parser_rejects_unknown_model() -> None:
    from instant_nurec.cli import make_parser
    with pytest.raises(SystemExit):
        make_parser().parse_args([
            "--model", "unknown",
            "--ncore-path", "/x",
            "--output-dir", "/y",
        ])


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


def test_parser_render_preview_flag_sets_true() -> None:
    from instant_nurec.cli import make_parser

    args = make_parser().parse_args(
        ["--ncore-path", "/x", "--output-dir", "/y", "--render-preview"]
    )
    assert args.render_preview is True


def test_parser_render_video_flag_sets_true() -> None:
    from instant_nurec.cli import make_parser

    args = make_parser().parse_args(
        ["--ncore-path", "/x", "--output-dir", "/y", "--render-video"]
    )
    assert args.render_video is True


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
    assert cfg.release_profile == "pa-front"
    assert cfg.dataset.predict.context_camera_ids == ["camera_front_wide_120fov"]
    assert cfg.dataset.predict.camera_subsampler.frame_width == 784
    assert cfg.dataset.predict.camera_subsampler.frame_height == 448


def test_main_render_preview_preflights_dependency_and_sets_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    from instant_nurec.predict import render_preview

    preflight = MagicMock(return_value=object())
    monkeypatch.setattr(render_preview, "require_gsplat", preflight)

    assert main(
        [
            "--ncore-path",
            str(json_path),
            "--output-dir",
            "/o",
            "--render-preview",
        ]
    ) == 0

    preflight.assert_called_once_with()
    assert fake_run_predict.call_args.args[0].predict.render_preview is True


def test_main_render_preview_reports_missing_dependency_before_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    from instant_nurec.predict import render_preview

    def fail_preflight():
        raise RuntimeError("install the render extra")

    monkeypatch.setattr(render_preview, "require_gsplat", fail_preflight)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--ncore-path",
                str(json_path),
                "--output-dir",
                "/o",
                "--render-preview",
            ]
        )

    assert "install the render extra" in capsys.readouterr().err
    fake_run_predict.assert_not_called()


def test_main_render_video_requires_merge_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--ncore-path",
                str(json_path),
                "--output-dir",
                "/o",
                "--render-video",
            ]
        )

    assert "--render-video requires --merge" in capsys.readouterr().err
    fake_run_predict.assert_not_called()


def test_main_render_video_preflights_dependencies_and_sets_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    from instant_nurec.predict import render_preview, render_video

    gsplat_preflight = MagicMock(return_value=object())
    ffmpeg_preflight = MagicMock(return_value="/usr/bin/ffmpeg")
    monkeypatch.setattr(render_preview, "require_gsplat", gsplat_preflight)
    monkeypatch.setattr(render_video, "require_ffmpeg", ffmpeg_preflight)

    assert main(
        [
            "--ncore-path",
            str(json_path),
            "--output-dir",
            "/o",
            "--merge",
            "--render-video",
            "--max-chunks",
            "12",
        ]
    ) == 0

    gsplat_preflight.assert_called_once_with()
    ffmpeg_preflight.assert_called_once_with()
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.predict.render_video is True
    assert cfg.predict.primitive_merge.enabled is True
    assert cfg.system.predict_batch_size == 12
    assert cfg.dataset.predict.frame_batch_sampler.n_samples_per_sequence == 12


def test_main_render_video_rejects_multiple_resolved_sequences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    first = _make_json_path(tmp_path)
    second = tmp_path / "second.json"
    second.write_text("{}")
    sequence_list = tmp_path / "sequences.lst"
    sequence_list.write_text(f"{first}\n{second}\n")
    from instant_nurec.cli import main
    from instant_nurec.predict import render_preview, render_video

    monkeypatch.setattr(render_preview, "require_gsplat", lambda: object())
    monkeypatch.setattr(render_video, "require_ffmpeg", lambda: "/usr/bin/ffmpeg")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--ncore-path",
                str(sequence_list),
                "--output-dir",
                "/o",
                "--merge",
                "--render-video",
            ]
        )

    assert "requires exactly one resolved NCore sequence" in capsys.readouterr().err
    fake_run_predict.assert_not_called()


def test_main_render_video_reports_missing_ffmpeg_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    from instant_nurec.predict import render_preview, render_video

    monkeypatch.setattr(render_preview, "require_gsplat", lambda: object())

    def fail_ffmpeg():
        raise RuntimeError("install ffmpeg")

    monkeypatch.setattr(render_video, "require_ffmpeg", fail_ffmpeg)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--ncore-path",
                str(json_path),
                "--output-dir",
                "/o",
                "--merge",
                "--render-video",
            ]
        )

    assert "install ffmpeg" in capsys.readouterr().err
    fake_run_predict.assert_not_called()


def test_main_multiview_uses_release_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main

    rc = main([
        "--model", "pa-multiview",
        "--ncore-path", str(json_path),
        "--output-dir", "/o",
    ])

    assert rc == 0
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.release_profile == "pa-multiview"
    assert cfg.dataset.predict.context_camera_ids == [
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
    ]
    assert cfg.dataset.predict.supervision_camera_ids == cfg.dataset.predict.context_camera_ids
    assert cfg.dataset.predict.camera_subsampler.frame_width == 504
    assert cfg.dataset.predict.camera_subsampler.frame_height == 280
    assert cfg.dataset.predict.frame_batch_sampler.n_frames_per_sample == 18
    assert cfg.model.decoder.motion_depth == 1


def test_main_point_query_uses_point_query_decoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    from instant_nurec.config_schema.models import KelvinPointQueryCADecoderConfig

    assert main(["--model", "pq-front", "--ncore-path", str(json_path), "--output-dir", "/o"]) == 0
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.release_profile == "pq-front"
    assert cfg.dataset.predict.context_camera_ids == ["camera_front_wide_120fov"]
    assert (
        cfg.dataset.predict.camera_subsampler.frame_width,
        cfg.dataset.predict.camera_subsampler.frame_height,
    ) == (784, 448)
    assert cfg.dataset.predict.frame_batch_sampler.n_frames_per_sample == 18
    assert isinstance(cfg.model.decoder, KelvinPointQueryCADecoderConfig)


def test_main_point_query_rejects_non_front_camera(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main

    with pytest.raises(ValueError, match="requires context camera"):
        main(
            [
                "--model",
                "pq-front",
                "--ncore-path",
                str(json_path),
                "--output-dir",
                "/o",
                "--camera-id",
                "camera_cross_left_120fov",
            ]
        )


def test_main_multiview_accepts_five_camera_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake_run_predict = _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main
    cameras = ["front", "cross-left", "cross-right", "rear-left", "rear-right"]
    argv = [
        "--model", "pa-multiview",
        "--ncore-path", str(json_path),
        "--output-dir", "/o",
    ]
    for camera in cameras:
        argv.extend(["--camera-id", camera])

    assert main(argv) == 0
    cfg = fake_run_predict.call_args.args[0]
    assert cfg.dataset.predict.context_camera_ids == cameras


def test_main_rejects_unsupported_multiview_camera_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _install_runtime_stubs(monkeypatch)
    json_path = _make_json_path(tmp_path)
    from instant_nurec.cli import main

    with pytest.raises(ValueError, match="expects 1, 3, 5 context camera"):
        main([
            "--model", "pa-multiview",
            "--ncore-path", str(json_path),
            "--output-dir", "/o",
            "--camera-id", "front",
            "--camera-id", "left",
        ])


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
