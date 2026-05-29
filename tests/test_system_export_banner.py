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


def _make_primitive(n_gaussians: int) -> MagicMock:
    primitive = MagicMock()
    primitive.static_layer.densities = torch.zeros(n_gaussians)
    return primitive


def _make_system(out_dir: Path, run_id: str, merge_enabled: bool) -> system_mod.GaussiansInstantNuRecSystem:
    inst = system_mod.GaussiansInstantNuRecSystem.__new__(system_mod.GaussiansInstantNuRecSystem)
    inst.out_dir = str(out_dir)
    inst.run_id = run_id
    inst.predict_config = types.SimpleNamespace(
        primitive_merge=types.SimpleNamespace(enabled=merge_enabled),
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


def test_banner_count_uses_thousands_separator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(system_mod, "export_ply", lambda **kwargs: None)
    primitives = [_make_primitive(1_000_000)]
    outputs, batch = _make_outputs(primitives, sequence_id="seq_z")
    inst = _make_system(tmp_path, run_id="run2", merge_enabled=True)

    inst.on_predict_batch_end(outputs, batch)

    assert "(1,000,000 gaussians)" in capsys.readouterr().out
