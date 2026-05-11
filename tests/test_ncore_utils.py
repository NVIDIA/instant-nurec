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

"""Branch-coverage tests for instant_nurec.utils.ncore_utils."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Stub fixture — installs ncore.* unions, then loads
# ``instant_nurec.utils.ncore_utils``. 
# removals dropped the prior stubs from this fixture.
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed_ncore_utils(monkeypatch):
    # ncore packages
    ncore_mod = types.ModuleType("ncore")
    data_mod = types.ModuleType("ncore.data")
    v4_mod = types.ModuleType("ncore.data.v4")
    impl_mod = types.ModuleType("ncore.impl")
    impl_data_mod = types.ModuleType("ncore.impl.data")
    stores_mod = types.ModuleType("ncore.impl.data.stores")

    class _SequenceLoaderProtocol:
        pass

    class _CameraSensorProtocol:
        pass

    data_mod.SequenceLoaderProtocol = _SequenceLoaderProtocol
    data_mod.CameraSensorProtocol = _CameraSensorProtocol
    data_mod.ConcreteCameraModelParametersUnion = object

    captured_loader_args = {}

    class _FakeReader:
        def __init__(self, dataset_paths, open_consolidated):
            captured_loader_args["dataset_paths"] = dataset_paths
            captured_loader_args["open_consolidated"] = open_consolidated

    class _FakeSequenceLoaderV4:
        def __init__(
            self,
            reader,
            *,
            poses_component_group_name,
            intrinsics_component_group_name,
            masks_component_group_name,
            cuboids_component_group_name,
        ):
            captured_loader_args["reader"] = reader
            captured_loader_args["poses"] = poses_component_group_name
            captured_loader_args["intrinsics"] = intrinsics_component_group_name
            captured_loader_args["masks"] = masks_component_group_name
            captured_loader_args["cuboids"] = cuboids_component_group_name

    v4_mod.SequenceComponentGroupsReader = _FakeReader
    v4_mod.SequenceLoaderV4 = _FakeSequenceLoaderV4
    data_mod.v4 = v4_mod
    ncore_mod.data = data_mod

    class _IndexedTarStore:
        def __init__(self, path, *, mode):
            pass

    def _open_compressed_consolidated(*, store, mode):
        return store  # not exercised in our tests

    stores_mod.IndexedTarStore = _IndexedTarStore
    stores_mod.open_compressed_consolidated = _open_compressed_consolidated
    impl_data_mod.stores = stores_mod
    impl_mod.data = impl_data_mod
    ncore_mod.impl = impl_mod

    for name, mod in [
        ("ncore", ncore_mod),
        ("ncore.data", data_mod),
        ("ncore.data.v4", v4_mod),
        ("ncore.impl", impl_mod),
        ("ncore.impl.data", impl_data_mod),
        ("ncore.impl.data.stores", stores_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    for cached in ("instant_nurec.utils.ncore_utils", "instant_nurec.utils.types"):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    import importlib

    mod = importlib.import_module("instant_nurec.utils.ncore_utils")
    return mod, captured_loader_args


# ---------------------------------------------------------------------------
# parse_sequence_meta_file
# ---------------------------------------------------------------------------


def _write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _valid_meta_payload():
    return {
        "version": "v4.0.0",
        "sequence_id": "seq-X",
        "sequence_timestamp_interval_us": {"start": 1000, "stop": 2000},
        "component_stores": [{"path": "shard0.zarr"}, {"path": "shard1.zarr"}],
    }


def test_parse_sequence_meta_returns_id_interval_and_paths(
    stubbed_ncore_utils, tmp_path
):
    mod, _ = stubbed_ncore_utils
    from upath import UPath

    meta = tmp_path / "meta.json"
    _write_meta(meta, _valid_meta_payload())

    seq_id, interval, paths = mod.parse_sequence_meta_file(UPath(meta))
    assert seq_id == "seq-X"
    assert interval.start == 1000
    assert interval.end == 2000
    assert [Path(p).name for p in paths] == ["shard0.zarr", "shard1.zarr"]


def test_parse_sequence_meta_rejects_non_file(stubbed_ncore_utils, tmp_path):
    mod, _ = stubbed_ncore_utils
    from upath import UPath

    with pytest.raises(AssertionError, match="not a file"):
        mod.parse_sequence_meta_file(UPath(tmp_path / "missing.json"))


def test_parse_sequence_meta_rejects_invalid_json(stubbed_ncore_utils, tmp_path):
    mod, _ = stubbed_ncore_utils
    from upath import UPath

    bad = tmp_path / "bad.json"
    bad.write_text("not valid json {")
    with pytest.raises(ValueError, match="not a json file"):
        mod.parse_sequence_meta_file(UPath(bad))


def test_parse_sequence_meta_rejects_non_v4_version(stubbed_ncore_utils, tmp_path):
    mod, _ = stubbed_ncore_utils
    from upath import UPath

    payload = _valid_meta_payload()
    payload["version"] = "v3.5.0"
    p = tmp_path / "v3.json"
    _write_meta(p, payload)
    with pytest.raises(AssertionError, match="not a NCore V4 single-sequence file"):
        mod.parse_sequence_meta_file(UPath(p))


def test_parse_sequence_meta_rejects_payload_with_missing_keys(
    stubbed_ncore_utils, tmp_path
):
    mod, _ = stubbed_ncore_utils
    from upath import UPath

    payload = _valid_meta_payload()
    del payload["sequence_id"]  # missing required key
    p = tmp_path / "broken.json"
    _write_meta(p, payload)
    with pytest.raises(AssertionError, match="not a NCore V4 single-sequence file"):
        mod.parse_sequence_meta_file(UPath(p))


def test_parse_sequence_meta_rejects_missing_version_field(
    stubbed_ncore_utils, tmp_path
):
    mod, _ = stubbed_ncore_utils
    from upath import UPath

    payload = _valid_meta_payload()
    del payload["version"]
    p = tmp_path / "no-version.json"
    _write_meta(p, payload)
    with pytest.raises(AssertionError, match="not a NCore V4 single-sequence file"):
        mod.parse_sequence_meta_file(UPath(p))


# ---------------------------------------------------------------------------
# create_sequence_loader
# ---------------------------------------------------------------------------


def test_create_sequence_loader_passes_args_to_v4_loader(stubbed_ncore_utils, tmp_path):
    mod, captured = stubbed_ncore_utils
    from upath import UPath

    paths = [UPath(tmp_path / "a"), UPath(tmp_path / "b")]
    mod.create_sequence_loader(
        paths,
        open_consolidated=False,
        v4_poses_component_group="poses",
        v4_intrinsics_component_group="intr",
        v4_masks_component_group="masks",
        v4_cuboids_component_group="cubs",
    )
    assert captured["dataset_paths"] == paths
    assert captured["open_consolidated"] is False
    assert captured["poses"] == "poses"
    assert captured["intrinsics"] == "intr"
    assert captured["masks"] == "masks"
    assert captured["cuboids"] == "cubs"


# ---------------------------------------------------------------------------
# get_mask_image
# ---------------------------------------------------------------------------


def _make_pil(mode, size, fill):
    from PIL import Image

    img = Image.new(mode, size, fill)
    return img


def test_get_mask_image_returns_none_when_input_is_none(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    assert mod.get_mask_image(None, target_mask_size=(640, 480)) is None


def test_get_mask_image_no_resize_when_size_matches(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    img = _make_pil("RGB", (10, 8), (255, 255, 255))
    out = mod.get_mask_image(img, target_mask_size=(10, 8))
    # All ones → mask True everywhere.
    assert out.shape == (8, 10)  # numpy is (H, W)
    assert out.all()


def test_get_mask_image_resizes_when_aspect_matches(stubbed_ncore_utils, caplog):
    """A larger mask with the same aspect ratio is resized down."""
    import logging as logging_mod

    mod, _ = stubbed_ncore_utils
    img = _make_pil("L", (20, 16), 255)  # 20:16 = 5:4
    with caplog.at_level(logging_mod.INFO):
        out = mod.get_mask_image(img, target_mask_size=(10, 8))  # 10:8 = 5:4
    assert out.shape == (8, 10)
    assert "Resizing camera mask" in caplog.text


def test_get_mask_image_raises_on_aspect_ratio_mismatch(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    img = _make_pil("L", (20, 5), 255)  # aspect 4
    with pytest.raises(AssertionError, match="aspect ratio"):
        mod.get_mask_image(img, target_mask_size=(10, 8))  # aspect 1.25


def test_get_mask_image_zero_pixels_become_false(stubbed_ncore_utils):
    """Mask True iff pixel != 0."""
    mod, _ = stubbed_ncore_utils
    img = _make_pil("L", (4, 4), 0)
    out = mod.get_mask_image(img, target_mask_size=(4, 4))
    assert not out.any()


# ---------------------------------------------------------------------------
# get_camera_sensor_mask
# ---------------------------------------------------------------------------


def test_get_camera_sensor_mask_uses_ego_mask(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    img = _make_pil("L", (4, 4), 255)

    class _ModelParams:
        resolution = (4, 4)

    class _Sensor:
        model_parameters = _ModelParams()

        def get_mask_images(self):
            return {"ego": img, "other": _make_pil("L", (8, 8), 0)}

    out = mod.get_camera_sensor_mask(_Sensor())
    assert out.shape == (4, 4)
    assert out.all()


def test_get_camera_sensor_mask_returns_none_when_no_ego(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils

    class _ModelParams:
        resolution = (4, 4)

    class _Sensor:
        model_parameters = _ModelParams()

        def get_mask_images(self):
            return {}  # no 'ego' entry

    assert mod.get_camera_sensor_mask(_Sensor()) is None
