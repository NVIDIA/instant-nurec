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

"""Branch-coverage tests for ``instant_nurec.model._resolve_model_pt_path`` and
``instant_nurec.model._validate_camera_ids``.

The full ``make()`` body GPU-loads a real GaussiansInstantNuRecSystem so we
don't exercise it in the cpu-only test venv. The thin pure-Python helpers
are worth their own focused branch tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec import pretrained  # noqa: E402
from instant_nurec import model as model_mod  # noqa: E402
from instant_nurec.model import (  # noqa: E402
    ModelCheckpointError,
    ModelNotFoundError,
    _load_model_checkpoint,
    _resolve_model_pt_path,
    _validate_camera_ids,
)


def test_resolve_returns_path_when_download_succeeds(monkeypatch, tmp_path):
    fake_pt = tmp_path / "instant_nurec_weights.pt"
    fake_pt.write_bytes(b"")
    monkeypatch.setattr(pretrained, "download_instant_nurec_pt", lambda **kw: str(fake_pt))
    assert _resolve_model_pt_path() == str(fake_pt)


def test_resolve_returns_none_when_download_raises(monkeypatch):
    def _raise(**kwargs):
        raise pretrained.PretrainedModelError("offline")

    monkeypatch.setattr(pretrained, "download_instant_nurec_pt", _raise)
    monkeypatch.delenv("INSTANT_NUREC_FULL_PT", raising=False)
    assert _resolve_model_pt_path() is None


def test_resolve_only_calls_downloader_once_per_invocation(monkeypatch):
    calls = {"n": 0}

    def _fake(**kwargs):
        calls["n"] += 1
        return "/tmp/something.pt"

    monkeypatch.setattr(pretrained, "download_instant_nurec_pt", _fake)
    _resolve_model_pt_path()
    assert calls["n"] == 1


# ---------- _validate_camera_ids ----------


_BASE_CAMERAS = ["cam_a", "cam_b", "cam_c"]


def _validate(**overrides):
    kwargs = dict(
        context_camera_ids=["cam_a"],
        supervision_camera_ids=["cam_a"],
        available_cameras=list(_BASE_CAMERAS),
        sequence_path="/seq/x.json",
    )
    kwargs.update(overrides)
    _validate_camera_ids(**kwargs)


def test_validate_camera_ids_happy_path_does_not_raise():
    _validate()


def test_validate_camera_ids_rejects_missing_context_camera():
    with pytest.raises(ValueError, match="camera_id 'cam_zzz' not found"):
        _validate(context_camera_ids=["cam_zzz"])


def test_validate_camera_ids_rejects_missing_supervision_camera():
    with pytest.raises(ValueError, match="camera_id 'cam_zzz' not found"):
        _validate(supervision_camera_ids=["cam_a", "cam_zzz"])


def test_validate_camera_ids_error_message_lists_available_cameras_sorted():
    with pytest.raises(ValueError) as exc:
        _validate(
            context_camera_ids=["cam_zzz"],
            available_cameras=["cam_b", "cam_a"],  # unsorted
        )
    # Ids in the message are sorted for readability.
    assert "['cam_a', 'cam_b']" in str(exc.value)


def test_validate_camera_ids_error_message_anchors_to_sequence_path():
    with pytest.raises(ValueError, match="/some/seq.json"):
        _validate(
            context_camera_ids=["cam_zzz"],
            sequence_path="/some/seq.json",
        )


# ---------- _load_model_checkpoint ----------


def test_load_model_checkpoint_extracts_shape_metadata(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    expected = {
        "expected_v": torch.tensor(18),
        "expected_h": torch.tensor(448),
        "expected_w": torch.tensor(784),
        "layer.weight": torch.arange(3),
    }
    torch.save(expected, checkpoint)

    state_dict, input_shape = _load_model_checkpoint(str(checkpoint))

    assert state_dict.keys() == {"layer.weight"}
    assert torch.equal(state_dict["layer.weight"], expected["layer.weight"])
    assert input_shape == {"expected_v": 18, "expected_h": 448, "expected_w": 784}


@pytest.mark.parametrize("payload", [[torch.ones(1)], {"layer.weight": "not-a-tensor"}])
def test_load_model_checkpoint_rejects_non_tensor_mapping(tmp_path, payload):
    checkpoint = tmp_path / "invalid.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ModelCheckpointError, match="plain tensor mapping"):
        _load_model_checkpoint(str(checkpoint))


def test_load_model_checkpoint_requires_shape_metadata(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"layer.weight": torch.ones(1)}, checkpoint)

    with pytest.raises(ModelCheckpointError, match="missing checkpoint-backed"):
        _load_model_checkpoint(str(checkpoint))


@pytest.mark.parametrize(
    "key,value",
    [
        ("expected_v", torch.tensor([18, 18])),
        ("expected_h", torch.tensor(448.0)),
        ("expected_w", torch.tensor(0)),
    ],
)
def test_load_model_checkpoint_rejects_invalid_shape_metadata(tmp_path, key, value):
    checkpoint = tmp_path / "invalid.pt"
    payload = {
        "expected_v": torch.tensor(18),
        "expected_h": torch.tensor(448),
        "expected_w": torch.tensor(784),
        "layer.weight": torch.ones(1),
    }
    payload[key] = value
    torch.save(payload, checkpoint)

    with pytest.raises(ModelCheckpointError, match=key):
        _load_model_checkpoint(str(checkpoint))


def test_load_model_checkpoint_rejects_legacy_traced_archive_before_torch_load(
    monkeypatch, tmp_path
):
    import zipfile

    checkpoint = tmp_path / "legacy.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("legacy/constants.pkl", b"")

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("legacy archive must be rejected before torch.load"),
    )
    with pytest.raises(ModelCheckpointError, match="legacy traced-model archive"):
        _load_model_checkpoint(str(checkpoint))


# ---------- make ----------


def test_make_raises_when_checkpoint_cannot_be_resolved(monkeypatch):
    monkeypatch.setattr(model_mod, "_resolve_model_pt_path", lambda: None)

    with pytest.raises(ModelNotFoundError, match=pretrained.MODEL_FILENAME):
        model_mod.make(SimpleNamespace())


def test_make_builds_source_core_and_loads_weights(monkeypatch, tmp_path):
    checkpoint = tmp_path / pretrained.MODEL_FILENAME
    checkpoint.write_bytes(b"placeholder")
    state_dict = {"encoder.weight": torch.ones(1)}
    input_shape = {"expected_v": 18, "expected_h": 448, "expected_w": 784}
    calls = {}

    class FakeCore:
        def __init__(self, config):
            calls["core_config"] = config

        def load_state_dict(self, loaded, *, strict):
            calls["loaded"] = loaded
            calls["strict"] = strict

    class FakeInference:
        def __init__(self, core, **kwargs):
            calls["inference_core"] = core
            calls["inference_kwargs"] = kwargs

    sentinel_system = object()
    model_config = SimpleNamespace(scene_rescale=0.15)
    dataset_config = SimpleNamespace(context_camera_ids=["front", "left"])
    config = SimpleNamespace(
        model=model_config,
        dataset=SimpleNamespace(predict=dataset_config),
    )

    monkeypatch.setattr(model_mod, "_resolve_model_pt_path", lambda: str(checkpoint))
    monkeypatch.setattr(model_mod, "_preflight_validate_camera_ids", lambda cfg: calls.setdefault("preflight", cfg))
    monkeypatch.setattr(
        model_mod, "_load_model_checkpoint", lambda path: (state_dict, input_shape)
    )
    monkeypatch.setattr(model_mod, "KelvinStaticCore", FakeCore)
    monkeypatch.setattr(model_mod, "KelvinInferenceModel", FakeInference)
    monkeypatch.setattr(
        model_mod,
        "GaussiansInstantNuRecSystem",
        lambda cfg, model: calls.update(system_args=(cfg, model)) or sentinel_system,
    )

    result = model_mod.make(config)

    assert result is sentinel_system
    assert calls["preflight"] is config
    assert calls["core_config"] is model_config
    assert calls["loaded"] is state_dict
    assert calls["strict"] is True
    assert calls["inference_kwargs"] == {
        "scene_rescale": 0.15,
        "expected_frames": 18,
        "expected_height": 448,
        "expected_width": 784,
    }


def _system_config(context_camera_ids):
    return SimpleNamespace(
        out_dir="/tmp/out",
        run_id="test",
        system=SimpleNamespace(),
        predict=SimpleNamespace(),
        model=SimpleNamespace(export_preprocess=SimpleNamespace()),
        dataset=SimpleNamespace(
            predict=SimpleNamespace(context_camera_ids=context_camera_ids)
        ),
    )


def test_system_configures_dataset_from_checkpoint_backed_model_shape(monkeypatch):
    from instant_nurec.model import system as system_mod

    calls = {}

    class FakeDataModule:
        def __init__(self, config, **kwargs):
            calls["config"] = config
            calls["kwargs"] = kwargs

    model = torch.nn.Module()
    model.expected_v = 18
    model.expected_h = 448
    model.expected_w = 784
    config = _system_config(["front", "left"])
    monkeypatch.setattr(system_mod, "InstantNuRecDataModule", FakeDataModule)

    system_mod.GaussiansInstantNuRecSystem(config, model)

    assert calls == {
        "config": config,
        "kwargs": {
            "frame_width": 784,
            "frame_height": 448,
            "n_frames_per_sample": 9,
        },
    }


def test_system_rejects_camera_count_that_does_not_divide_expected_v():
    from instant_nurec.model.system import GaussiansInstantNuRecSystem

    model = torch.nn.Module()
    model.expected_v = 18
    model.expected_h = 448
    model.expected_w = 784

    with pytest.raises(ValueError, match="doesn't divide"):
        GaussiansInstantNuRecSystem(_system_config(["front"] * 4), model)
