# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Branch-coverage for the stdout completion banner emitted by
``GaussiansInstantNuRecSystem.on_predict_batch_end`` after each PLY write.

``export_ply`` is stubbed so the test doesn't touch disk or GPU.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec.model import system as system_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_sky_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_mod,
        "export_sky",
        lambda primitive, rig, path: (
            path.with_suffix(".sky.npz"),
            path.with_suffix(".sky.png"),
        ),
    )


def _make_primitive(n_gaussians: int) -> MagicMock:
    primitive = MagicMock()
    primitive.static_layer.densities = torch.zeros(n_gaussians)
    return primitive


def _make_system(
    out_dir: Path,
    run_id: str,
    merge_enabled: bool,
    *,
    render_video: bool = False,
) -> system_mod.GaussiansInstantNuRecSystem:
    inst = system_mod.GaussiansInstantNuRecSystem.__new__(system_mod.GaussiansInstantNuRecSystem)
    inst.out_dir = str(out_dir)
    inst.run_id = run_id
    inst.predict_config = types.SimpleNamespace(
        primitive_merge=types.SimpleNamespace(enabled=merge_enabled),
        render_preview=False,
        render_video=render_video,
    )
    return inst


def _make_outputs(primitives: list[MagicMock], sequence_id: str) -> tuple[dict, MagicMock]:
    batch = MagicMock()
    batch.meta = [{"sequence_id": sequence_id} for _ in primitives]
    batch.context_rig = [MagicMock() for _ in primitives]
    batch.__len__.return_value = len(primitives)
    outputs = {"primitives": primitives, "batch": batch}
    return outputs, batch


def test_banner_fires_per_chunk_with_count_and_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(system_mod, "export_ply", lambda **kwargs: None)
    primitives = [_make_primitive(123), _make_primitive(456)]
    outputs, batch = _make_outputs(primitives, sequence_id="seq_x")
    inst = _make_system(tmp_path, run_id="run0", merge_enabled=False)

    inst.on_predict_batch_end(outputs, batch)

    out = capsys.readouterr().out
    expected_chunk0 = (tmp_path / "run0" / "ply" / "seq_x" / "seq_x_chunk0.ply").resolve()
    expected_chunk1 = (tmp_path / "run0" / "ply" / "seq_x" / "seq_x_chunk1.ply").resolve()
    assert f"Wrote 3DGS PLY (123 gaussians): {expected_chunk0}" in out
    assert f"Wrote 3DGS PLY (456 gaussians): {expected_chunk1}" in out


def test_banner_uses_no_chunk_suffix_when_merge_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(system_mod, "export_ply", lambda **kwargs: None)
    primitives = [_make_primitive(1_883_483)]
    outputs, batch = _make_outputs(primitives, sequence_id="seq_y")
    inst = _make_system(tmp_path, run_id="run1", merge_enabled=True)

    inst.on_predict_batch_end(outputs, batch)

    out = capsys.readouterr().out
    expected = (tmp_path / "run1" / "ply" / "seq_y" / "seq_y.ply").resolve()
    assert f"Wrote 3DGS PLY (1,883,483 gaussians): {expected}" in out
    assert f"Wrote sky sidecar: {expected.with_suffix('.sky.npz')}" in out
    assert f"Wrote sky preview: {expected.with_suffix('.sky.png')}" in out


def test_banner_count_uses_thousands_separator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(system_mod, "export_ply", lambda **kwargs: None)
    primitives = [_make_primitive(1_000_000)]
    outputs, batch = _make_outputs(primitives, sequence_id="seq_z")
    inst = _make_system(tmp_path, run_id="run2", merge_enabled=True)

    inst.on_predict_batch_end(outputs, batch)

    assert "(1,000,000 gaussians)" in capsys.readouterr().out


def test_full_video_uses_source_rig_and_prints_verified_timeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(system_mod, "export_ply", lambda **kwargs: None)
    video_stats = types.SimpleNamespace(
        path=tmp_path / "run3" / "ply" / "seq_video" / "seq_video.render.mp4",
        frame_count=599,
        fps=29.9994,
        duration_seconds=19.9667,
        background_fraction=0.42,
        sky_contribution_mean=0.1234,
    )
    render_video = MagicMock(return_value=video_stats)
    monkeypatch.setattr(system_mod, "render_reference_video", render_video)

    primitive = _make_primitive(1_918_461)
    outputs, batch = _make_outputs([primitive], sequence_id="seq_video")
    ncore_path = tmp_path / "sequence.json"
    batch.meta[0]["ncore_json_path"] = ncore_path
    inst = _make_system(tmp_path, run_id="run3", merge_enabled=True, render_video=True)
    full_rig = MagicMock(name="full_rig")
    dataset = MagicMock()
    dataset.config.context_camera_ids = ["front"]
    dataset.load_full_camera_rig.return_value = full_rig
    inst.datamodule = types.SimpleNamespace(predict_dataset=dataset)

    inst.on_predict_batch_end(outputs, batch)

    dataset.load_full_camera_rig.assert_called_once_with(
        ncore_path,
        batch.context_rig[0],
        camera_id="front",
    )
    expected_path = tmp_path / "run3" / "ply" / "seq_video" / "seq_video.render.mp4"
    render_video.assert_called_once_with(primitive, full_rig, expected_path)
    out = capsys.readouterr().out
    assert "599 frames, 29.999 fps, 19.97s" in out
    assert f"{expected_path.resolve()}" in out


def test_video_failure_happens_after_primary_exports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fake_export_ply(**kwargs) -> None:
        del kwargs
        calls.append("ply")

    def fake_export_sky(primitive, rig, path):
        del primitive, rig
        calls.append("sky")
        return path.with_suffix(".sky.npz"), path.with_suffix(".sky.png")

    def fail_video(*args, **kwargs):
        del args, kwargs
        calls.append("video")
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(system_mod, "export_ply", fake_export_ply)
    monkeypatch.setattr(system_mod, "export_sky", fake_export_sky)
    monkeypatch.setattr(system_mod, "render_reference_video", fail_video)

    primitive = _make_primitive(3)
    outputs, batch = _make_outputs([primitive], sequence_id="seq_failure")
    batch.meta[0]["ncore_json_path"] = tmp_path / "sequence.json"
    inst = _make_system(tmp_path, run_id="run4", merge_enabled=True, render_video=True)
    dataset = MagicMock()
    dataset.config.context_camera_ids = ["front"]
    dataset.load_full_camera_rig.return_value = MagicMock(name="full_rig")
    inst.datamodule = types.SimpleNamespace(predict_dataset=dataset)

    with pytest.raises(RuntimeError, match="encoder failed"):
        inst.on_predict_batch_end(outputs, batch)

    assert calls == ["ply", "sky", "video"]
