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

"""Branch-coverage tests for instant_nurec.utils.ncore_utils.

The module imports a substantial dependency graph (ncore, zarr, PIL,
torchvision). We stub the ncore parts via ``sys.modules``; the others are
real, since they're now installed in the test venv.

We don't drive ``AuxShardDataLoader.__init__`` end-to-end (it walks a real
zarr store on disk). Instead we instantiate a "headless" loader by setting
``base_groups`` / ``_sequence_id`` / ``_shard_count`` directly on a bare
instance, then exercise the ``has_*``/``get_*`` query surface.
"""

from __future__ import annotations

import io
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
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
    data_mod.ConcreteLidarModelParametersUnion = object

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


# ---------------------------------------------------------------------------
# AuxShardDataLoader query methods (no __init__ — set state directly)
# ---------------------------------------------------------------------------


def _png_bytes(arr: np.ndarray) -> bytes:
    """Encode a uint8 (H, W) array as PNG and return the bytes."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="png")
    return buf.getvalue()


class _FakeDataset:
    """Drop-in replacement for ``zarr.Array`` (used in shard[][k] indexing)."""

    def __init__(self, payload_bytes, attrs=None):
        self._bytes = payload_bytes
        self.attrs = attrs or {}

    def __getitem__(self, _idx):
        return self._bytes


class _FakeBaseGroup(dict):
    """Mimics a zarr.Group enough that `cam_id in group` and indexing work."""

    def __init__(self, attrs=None, **inner):
        super().__init__(inner)
        self.attrs = attrs or {}


def _new_aux_loader(mod, base_groups: dict, sequence_id="seq-X", shard_count=1):
    loader = mod.AuxShardDataLoader.__new__(mod.AuxShardDataLoader)
    loader.base_groups = defaultdict(list, base_groups)
    loader._sequence_id = sequence_id
    loader._shard_count = shard_count
    loader.aux_shard_stores = []
    return loader


def test_aux_loader_has_base_group_no_sensor(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    loader = _new_aux_loader(mod, {"semantic_segmentation": [{}]})
    assert loader._has_base_group("semantic_segmentation") is True
    assert loader._has_base_group("missing") is False


def test_aux_loader_has_base_group_with_sensor(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    bg = _FakeBaseGroup(camera_front=_FakeBaseGroup(), other=_FakeBaseGroup())
    loader = _new_aux_loader(mod, {"semantic_segmentation": [bg]})
    assert loader._has_base_group("semantic_segmentation", "camera_front") is True
    assert loader._has_base_group("semantic_segmentation", "missing_camera") is False


def test_aux_loader_has_semantic_segmentation_proxies(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    loader = _new_aux_loader(mod, {"semantic_segmentation": [_FakeBaseGroup(cam=_FakeBaseGroup())]})
    assert loader.has_semantic_segmentation() is True
    assert loader.has_semantic_segmentation("cam") is True
    assert loader.has_semantic_segmentation("nope") is False


def test_aux_loader_get_semantic_segmentation_meta_returns_attrs(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    cam_group = _FakeBaseGroup(attrs={"format": "png", "classes": 21})
    bg = _FakeBaseGroup(cam=cam_group)
    loader = _new_aux_loader(mod, {"semantic_segmentation": [bg]})
    out = loader.get_semantic_segmentation_meta("cam")
    assert out == {"format": "png", "classes": 21}


def test_aux_loader_get_semantic_segmentation_meta_raises_when_missing(
    stubbed_ncore_utils,
):
    mod, _ = stubbed_ncore_utils
    loader = _new_aux_loader(mod, {})
    with pytest.raises(KeyError, match="No semantic segmentation"):
        loader.get_semantic_segmentation_meta("cam")


def test_aux_loader_get_semantic_segmentation_decodes_png(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    pixels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    ds = _FakeDataset(_png_bytes(pixels), attrs={"format": "png"})
    bg = _FakeBaseGroup(cam=_FakeBaseGroup(**{"100": ds}))
    loader = _new_aux_loader(mod, {"semantic_segmentation": [bg]})
    img = loader.get_semantic_segmentation("cam", 100)
    assert img.size == (2, 2)


def test_aux_loader_get_semantic_segmentation_walks_shards(stubbed_ncore_utils):
    """When the first shard doesn't have the timestamp, the loader continues
    to the next shard."""
    mod, _ = stubbed_ncore_utils
    pixels = np.zeros((2, 2), dtype=np.uint8)
    ds = _FakeDataset(_png_bytes(pixels), attrs={"format": "png"})
    bg1 = _FakeBaseGroup(cam=_FakeBaseGroup())  # no timestamp 100
    bg2 = _FakeBaseGroup(cam=_FakeBaseGroup(**{"100": ds}))
    loader = _new_aux_loader(mod, {"semantic_segmentation": [bg1, bg2]})
    img = loader.get_semantic_segmentation("cam", 100)
    assert img.size == (2, 2)


def test_aux_loader_get_semantic_segmentation_raises_when_signal_absent(
    stubbed_ncore_utils,
):
    mod, _ = stubbed_ncore_utils
    loader = _new_aux_loader(mod, {})  # no semantic-seg signal at all
    with pytest.raises(KeyError, match="no semantic segmentation"):
        loader.get_semantic_segmentation("cam", 100)


def test_aux_loader_get_semantic_segmentation_raises_when_timestamp_missing_in_all_shards(
    stubbed_ncore_utils,
):
    mod, _ = stubbed_ncore_utils
    bg = _FakeBaseGroup(cam=_FakeBaseGroup())  # no timestamps at all
    loader = _new_aux_loader(mod, {"semantic_segmentation": [bg]})
    with pytest.raises(KeyError, match="semantic segmentation not found"):
        loader.get_semantic_segmentation("cam", 100)


def test_aux_loader_has_depth_proxies(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    bg = _FakeBaseGroup(cam=_FakeBaseGroup())
    loader = _new_aux_loader(mod, {"depth": [bg]})
    assert loader.has_depth() is True
    assert loader.has_depth("cam") is True
    assert loader.has_depth("nope") is False


def test_aux_loader_has_egomask_proxies(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    bg = _FakeBaseGroup(cam=_FakeBaseGroup())
    loader = _new_aux_loader(mod, {"egomask": [bg]})
    assert loader.has_egomask() is True
    assert loader.has_egomask("cam") is True


def test_aux_loader_get_egomask_returns_super_mask_for_zero_timestamp(
    stubbed_ncore_utils,
):
    """When frame_timestamps_us=0, the aggregated super ego-mask at frame_key=0
    is returned without a closest-timestamp search."""
    mod, _ = stubbed_ncore_utils
    pixels = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    ds = _FakeDataset(_png_bytes(pixels), attrs={"format": "png"})
    cam = _FakeBaseGroup()
    cam[0] = ds  # the SUT indexes with the int 0
    bg = _FakeBaseGroup(cam=cam)
    loader = _new_aux_loader(mod, {"egomask": [bg]})
    out = loader.get_egomask("cam", 0)
    assert out.shape == (2, 2)
    assert out.dtype == bool


def test_aux_loader_get_egomask_picks_closest_frame_for_nonzero(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    pixels = np.array([[0, 255]], dtype=np.uint8)
    ds_at_50 = _FakeDataset(_png_bytes(pixels), attrs={"format": "png"})
    ds_at_200 = _FakeDataset(_png_bytes(pixels), attrs={"format": "png"})
    cam = _FakeBaseGroup(**{"50": ds_at_50, "200": ds_at_200})
    bg = _FakeBaseGroup(cam=cam)
    loader = _new_aux_loader(mod, {"egomask": [bg]})
    # 100 → closest is "50".
    out = loader.get_egomask("cam", 100)
    assert out.shape == (1, 2)


def test_aux_loader_get_egomask_skips_shards_missing_camera(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    pixels = np.zeros((1, 2), dtype=np.uint8)
    ds = _FakeDataset(_png_bytes(pixels), attrs={"format": "png"})
    bg1 = _FakeBaseGroup()  # camera not present
    cam2 = _FakeBaseGroup()
    cam2[0] = ds
    bg2 = _FakeBaseGroup(cam=cam2)
    loader = _new_aux_loader(mod, {"egomask": [bg1, bg2]})
    out = loader.get_egomask("cam", 0)
    assert out.shape == (1, 2)


def test_aux_loader_get_egomask_raises_when_signal_absent(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    loader = _new_aux_loader(mod, {})
    with pytest.raises(KeyError, match="no ego-mask data loaded"):
        loader.get_egomask("cam", 0)


def test_aux_loader_get_egomask_raises_when_no_timestamps_in_camera(
    stubbed_ncore_utils,
):
    """For a non-zero target timestamp, an empty camera dict raises rather
    than falling silently to the next shard."""
    mod, _ = stubbed_ncore_utils
    bg = _FakeBaseGroup(cam=_FakeBaseGroup())
    loader = _new_aux_loader(mod, {"egomask": [bg]})
    with pytest.raises(KeyError, match="No ego-masks found"):
        loader.get_egomask("cam", 100)


def test_aux_loader_get_egomask_raises_when_camera_in_no_shard(stubbed_ncore_utils):
    """If no shard contains the camera, the final 'no mask found' branch fires."""
    mod, _ = stubbed_ncore_utils
    bg = _FakeBaseGroup()  # no cameras at all
    loader = _new_aux_loader(mod, {"egomask": [bg]})
    with pytest.raises(KeyError, match="No ego-mask found"):
        loader.get_egomask("cam", 0)


def test_aux_loader_get_egomask_zero_ts_skips_shards_without_super_mask(
    stubbed_ncore_utils,
):
    """frame_timestamps_us=0 picks frame_key=0; if a shard has the camera
    but no key 0, the inner ``except KeyError: continue`` branch fires and
    we move on to the next shard (or raise at the end if none works).

    This covers the second of the two `try/except: continue` blocks at
    lines 276-277 — a different path from the camera-not-present branch."""
    mod, _ = stubbed_ncore_utils
    bg1 = _FakeBaseGroup(cam=_FakeBaseGroup())  # camera present, no "0" entry
    bg2 = _FakeBaseGroup(cam=_FakeBaseGroup())  # also no "0" entry
    loader = _new_aux_loader(mod, {"egomask": [bg1, bg2]})
    with pytest.raises(KeyError, match="No ego-mask found"):
        loader.get_egomask("cam", 0)


def test_aux_loader_get_depth_decodes_png_storage(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    pixels = np.array([[0, 200]], dtype=np.uint8)
    ds = _FakeDataset(_png_bytes(pixels), attrs={"format": "png"})
    cam = _FakeBaseGroup(attrs={"store_depth_as_png": True, "max_depth_m": 100.0}, **{"50": ds})
    bg = _FakeBaseGroup(cam=cam)
    loader = _new_aux_loader(mod, {"depth": [bg]})
    # Override get_depth_meta to avoid ncore-side meta lookup we haven't stubbed.
    loader.get_depth_meta = lambda _camera_id: {"store_depth_as_png": True, "max_depth_m": 100.0}
    out = loader.get_depth("cam", 50)
    assert out.dtype == np.float32
    assert out.shape == (1, 2)


def test_aux_loader_get_depth_returns_raw_when_not_stored_as_png(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils

    class _RawDataset:
        attrs = {"format": "raw"}

        def __array__(self, dtype=None):
            arr = np.array([[1.5, 2.5, 3.5]], dtype=np.float32)
            return arr.astype(dtype) if dtype is not None else arr

        def __getitem__(self, _idx):
            return np.array([[1.5, 2.5, 3.5]], dtype=np.float32)

    cam = _FakeBaseGroup(**{"50": _RawDataset()})
    bg = _FakeBaseGroup(cam=cam)
    loader = _new_aux_loader(mod, {"depth": [bg]})
    loader.get_depth_meta = lambda _camera_id: {"store_depth_as_png": False}
    out = loader.get_depth("cam", 50)
    assert np.allclose(out, [[1.5, 2.5, 3.5]])


def test_aux_loader_get_depth_resizes_to_target_when_provided(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    pixels = np.array([[10, 20]], dtype=np.uint8)
    ds = _FakeDataset(_png_bytes(pixels), attrs={"format": "png"})
    cam = _FakeBaseGroup(**{"50": ds})
    bg = _FakeBaseGroup(cam=cam)
    loader = _new_aux_loader(mod, {"depth": [bg]})
    loader.get_depth_meta = lambda _camera_id: {"store_depth_as_png": True, "max_depth_m": 1.0}
    out = loader.get_depth("cam", 50, target_width_height=(4, 2))
    # Resized to 4x2 (h=2, w=4 numpy convention)
    assert out.shape == (2, 4)


def test_aux_loader_get_depth_raises_when_signal_absent(stubbed_ncore_utils):
    mod, _ = stubbed_ncore_utils
    loader = _new_aux_loader(mod, {})
    with pytest.raises(KeyError, match="no depth data loaded"):
        loader.get_depth("cam", 50)


def test_aux_loader_get_depth_raises_when_camera_or_timestamp_missing(
    stubbed_ncore_utils,
):
    mod, _ = stubbed_ncore_utils
    bg = _FakeBaseGroup(cam=_FakeBaseGroup())  # no timestamps for cam
    loader = _new_aux_loader(mod, {"depth": [bg]})
    loader.get_depth_meta = lambda _camera_id: {"store_depth_as_png": False}
    with pytest.raises(KeyError, match="depth not found"):
        loader.get_depth("cam", 50)


# ---------------------------------------------------------------------------
# AuxShardDataLoader.__init__ — real filesystem walk + stubbed zarr open
# ---------------------------------------------------------------------------


class _FakeRootGroup:
    """Stand-in for an opened zarr root with `.attrs` and item iteration."""

    def __init__(self, attrs, root_subgroups):
        # attrs supports `.get(key, default)` via dict.
        self.attrs = attrs
        # The SUT does ``aux_shard_root[aux_root_group_name].items()``; we
        # expose only the top-level subgroup name.
        self._root_subgroups = root_subgroups

    def __getitem__(self, name):
        return self._root_subgroups[name]


class _FakeRootSubgroup:
    """Stand-in for ``aux_shard_root[<aux_root_group_name>]`` — has .items()."""

    def __init__(self, mapping):
        self._mapping = mapping

    def items(self):
        return self._mapping.items()


@pytest.fixture
def stubbed_zarr_open(monkeypatch, stubbed_ncore_utils):
    """Patch zarr.open / zarr.storage.DirectoryStore /
    ncore_data_stores.IndexedTarStore on the SUT module to return our fake
    root groups, keyed by the store path basename."""
    mod, _ = stubbed_ncore_utils

    # Mapping `Path → FakeRootGroup` filled in per-test.
    registry: dict = {}

    class _FakeStore:
        def __init__(self, path):
            self.path = path

    def _fake_directory_store(path):
        return _FakeStore(path)

    def _fake_indexed_tar_store(path, *, mode):
        return _FakeStore(path)

    def _fake_zarr_open(*, store, mode):
        # Look up the prebuilt root group by store.path.name.
        return registry[Path(str(store.path)).name]

    # Patch on the SUT module — the SUT references zarr.storage.DirectoryStore
    # and zarr.open via the imported ``zarr`` name.
    monkeypatch.setattr(mod.zarr.storage, "DirectoryStore", _fake_directory_store, raising=False)
    monkeypatch.setattr(mod.zarr, "open", _fake_zarr_open, raising=False)
    monkeypatch.setattr(
        mod.ncore_data_stores, "IndexedTarStore", _fake_indexed_tar_store, raising=False
    )
    # Also redirect open_compressed_consolidated so the True-flag branch works.
    monkeypatch.setattr(
        mod.ncore_data_stores,
        "open_compressed_consolidated",
        lambda *, store, mode: registry[Path(str(store.path)).name],
        raising=False,
    )
    # Make the SUT's `isinstance(base_group, zarr.Group)` accept our fakes.
    monkeypatch.setattr(mod.zarr, "Group", _FakeRootSubgroup, raising=False)

    return mod, registry


def _seed_aux_dir(tmp_path: Path, *, dataset_basename: str, aux_signal_dirnames=()):
    """Create a synthetic dataset shard layout so the SUT's iterdir() walk has
    something to find."""
    parent = tmp_path / "shards"
    parent.mkdir(parents=True, exist_ok=True)
    # The dataset shard itself; the SUT's stem.split('.')[0] gives the basename.
    (parent / f"{dataset_basename}.something.zarr").mkdir()
    # Add aux .zarr directories the SUT should pick up.
    for name in aux_signal_dirnames:
        (parent / name).mkdir()
    # Also add a non-matching directory (must be ignored).
    (parent / "unrelated.dir").mkdir()
    # Also add a non-matching file (must be ignored).
    (parent / "junk.txt").write_text("x")
    return parent / f"{dataset_basename}.something.zarr"


def test_aux_loader_init_picks_up_directory_stores(stubbed_zarr_open, tmp_path):
    """A *.zarr directory matching ``<base>.aux.*`` is registered."""
    mod, registry = stubbed_zarr_open

    dataset_path = _seed_aux_dir(
        tmp_path,
        dataset_basename="seqA-shard0",
        aux_signal_dirnames=["seqA-shard0.aux.semseg.zarr"],
    )
    registry["seqA-shard0.aux.semseg.zarr"] = _FakeRootGroup(
        attrs={
            "sequence_id": "seqA",
            "shard_id": 0,
            "shard_count": 1,
            "aux_root_group_name": "annotations",
        },
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )

    loader = mod.AuxShardDataLoader(sequence_id="seqA", dataset_paths=[dataset_path])
    assert "depth" in loader.base_groups
    assert loader._sequence_id == "seqA"
    assert loader._shard_count == 1


def test_aux_loader_init_picks_up_itar_files(stubbed_zarr_open, tmp_path):
    """A ``<base>.aux.<signal>.zarr.itar`` file is registered (file branch)."""
    mod, registry = stubbed_zarr_open

    parent = tmp_path / "shards"
    parent.mkdir()
    (parent / "seqB-shard0.something.zarr").mkdir()  # the dataset itself
    itar = parent / "seqB-shard0.aux.depth.zarr.itar"
    itar.write_bytes(b"")
    registry["seqB-shard0.aux.depth.zarr.itar"] = _FakeRootGroup(
        attrs={"sequence_id": "seqB", "shard_id": 0, "shard_count": 1},
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )

    loader = mod.AuxShardDataLoader(
        sequence_id="seqB", dataset_paths=[parent / "seqB-shard0.something.zarr"]
    )
    assert "depth" in loader.base_groups


def test_aux_loader_init_picks_up_legacy_annotations_files(
    stubbed_zarr_open, tmp_path
):
    """The backwards-compatibility ``<base>-annotations*.zarr.itar`` pattern
    is recognised (the SUT's ``or`` branch)."""
    mod, registry = stubbed_zarr_open

    parent = tmp_path / "shards"
    parent.mkdir()
    (parent / "seqC-shard0.something.zarr").mkdir()
    legacy = parent / "seqC-shard0-annotations.zarr.itar"
    legacy.write_bytes(b"")
    registry["seqC-shard0-annotations.zarr.itar"] = _FakeRootGroup(
        attrs={"sequence_id": "seqC", "shard_id": 0, "shard_count": 1},
        root_subgroups={"annotations": _FakeRootSubgroup({"egomask": _FakeRootSubgroup({})})},
    )

    loader = mod.AuxShardDataLoader(
        sequence_id="seqC", dataset_paths=[parent / "seqC-shard0.something.zarr"]
    )
    assert "egomask" in loader.base_groups


def test_aux_loader_init_skips_non_matching_files_and_dirs(stubbed_zarr_open, tmp_path):
    """Files without ``.zarr.itar`` / dirs without ``.zarr`` / mismatched
    base names are silently ignored."""
    mod, registry = stubbed_zarr_open

    parent = tmp_path / "shards"
    parent.mkdir()
    (parent / "seqD-shard0.something.zarr").mkdir()
    # Non-matching bits to skip.
    (parent / "seqD-shard0.txt").write_text("x")  # not .zarr.itar
    (parent / "OTHER-shard0.aux.depth.zarr.itar").write_bytes(b"")  # base mismatch
    (parent / "seqD-shard0.aux.depth.bin").write_bytes(b"")  # extension mismatch
    (parent / "OTHER-shard0.aux.depth.zarr").mkdir()  # base mismatch
    (parent / "seqD-shard0.aux.depth.zarr").mkdir()  # this one matches

    registry["seqD-shard0.aux.depth.zarr"] = _FakeRootGroup(
        attrs={"sequence_id": "seqD", "shard_id": 0, "shard_count": 1},
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )

    loader = mod.AuxShardDataLoader(
        sequence_id="seqD", dataset_paths=[parent / "seqD-shard0.something.zarr"]
    )
    assert "depth" in loader.base_groups
    # Only the one matching store should have been loaded.
    assert len(loader.aux_shard_stores) == 1


def test_aux_loader_init_uses_default_aux_root_group_name(stubbed_zarr_open, tmp_path):
    """If ``aux_root_group_name`` attr is absent, the SUT falls back to
    ``"annotations"``."""
    mod, registry = stubbed_zarr_open

    dp = _seed_aux_dir(
        tmp_path,
        dataset_basename="seqE-shard0",
        aux_signal_dirnames=["seqE-shard0.aux.semseg.zarr"],
    )
    registry["seqE-shard0.aux.semseg.zarr"] = _FakeRootGroup(
        # NO `aux_root_group_name` attr → must fall back to 'annotations'.
        attrs={"sequence_id": "seqE", "shard_id": 0, "shard_count": 1},
        root_subgroups={"annotations": _FakeRootSubgroup({"semseg": _FakeRootSubgroup({})})},
    )

    loader = mod.AuxShardDataLoader(sequence_id="seqE", dataset_paths=[dp])
    assert "semseg" in loader.base_groups


def test_aux_loader_init_open_consolidated_branch(stubbed_zarr_open, tmp_path):
    """When ``open_consolidated=True`` (default), the SUT uses
    ``ncore_data_stores.open_compressed_consolidated`` instead of zarr.open."""
    mod, registry = stubbed_zarr_open

    dp = _seed_aux_dir(
        tmp_path,
        dataset_basename="seqF-shard0",
        aux_signal_dirnames=["seqF-shard0.aux.depth.zarr"],
    )
    registry["seqF-shard0.aux.depth.zarr"] = _FakeRootGroup(
        attrs={"sequence_id": "seqF", "shard_id": 0, "shard_count": 1},
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )

    loader = mod.AuxShardDataLoader(
        sequence_id="seqF", dataset_paths=[dp], open_consolidated=True
    )
    assert "depth" in loader.base_groups


def test_aux_loader_init_skips_non_group_entries(stubbed_zarr_open, tmp_path):
    """Items in the root group that aren't `zarr.Group` instances (i.e. raw
    datasets) are skipped, not registered as base_groups."""
    mod, registry = stubbed_zarr_open

    dp = _seed_aux_dir(
        tmp_path,
        dataset_basename="seqG-shard0",
        aux_signal_dirnames=["seqG-shard0.aux.semseg.zarr"],
    )
    # Provide one Group (proper subgroup) and one non-Group (a plain dict).
    not_a_group = object()
    registry["seqG-shard0.aux.semseg.zarr"] = _FakeRootGroup(
        attrs={"sequence_id": "seqG", "shard_id": 0, "shard_count": 1},
        root_subgroups={
            "annotations": _FakeRootSubgroup({
                "semseg": _FakeRootSubgroup({}),
                "raw_dataset": not_a_group,
            })
        },
    )

    loader = mod.AuxShardDataLoader(sequence_id="seqG", dataset_paths=[dp])
    assert "semseg" in loader.base_groups
    assert "raw_dataset" not in loader.base_groups


def test_aux_loader_init_rejects_mismatched_sequence_id_constructor_arg(
    stubbed_zarr_open, tmp_path
):
    """If the constructor's ``sequence_id`` doesn't match the loaded store's
    sequence_id attr, a ValueError is raised."""
    mod, registry = stubbed_zarr_open

    dp = _seed_aux_dir(
        tmp_path,
        dataset_basename="seqH-shard0",
        aux_signal_dirnames=["seqH-shard0.aux.depth.zarr"],
    )
    registry["seqH-shard0.aux.depth.zarr"] = _FakeRootGroup(
        attrs={"sequence_id": "WRONG", "shard_id": 0, "shard_count": 1},
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )

    with pytest.raises(ValueError, match="not compatible with source sequence"):
        mod.AuxShardDataLoader(sequence_id="seqH", dataset_paths=[dp])


def test_aux_loader_init_rejects_mismatched_sequence_across_stores(
    stubbed_zarr_open, tmp_path
):
    """Two stores with different sequence_id attrs → ValueError."""
    mod, registry = stubbed_zarr_open

    dp = _seed_aux_dir(
        tmp_path,
        dataset_basename="seqI-shard0",
        aux_signal_dirnames=[
            "seqI-shard0.aux.depth.zarr",
            "seqI-shard0.aux.semseg.zarr",
        ],
    )
    registry["seqI-shard0.aux.depth.zarr"] = _FakeRootGroup(
        attrs={"sequence_id": "seqI", "shard_id": 0, "shard_count": 1},
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )
    registry["seqI-shard0.aux.semseg.zarr"] = _FakeRootGroup(
        attrs={"sequence_id": "OTHER_SEQ", "shard_id": 0, "shard_count": 1},
        root_subgroups={"annotations": _FakeRootSubgroup({"semseg": _FakeRootSubgroup({})})},
    )

    with pytest.raises(ValueError, match="different sequences|not compatible"):
        mod.AuxShardDataLoader(sequence_id="seqI", dataset_paths=[dp])


def test_aux_loader_init_rejects_mismatched_shard_count(stubbed_zarr_open, tmp_path):
    mod, registry = stubbed_zarr_open

    dp = _seed_aux_dir(
        tmp_path,
        dataset_basename="seqJ-shard0",
        aux_signal_dirnames=[
            "seqJ-shard0.aux.depth.zarr",
            "seqJ-shard0.aux.semseg.zarr",
        ],
    )
    registry["seqJ-shard0.aux.depth.zarr"] = _FakeRootGroup(
        attrs={"sequence_id": "seqJ", "shard_id": 0, "shard_count": 2},
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )
    registry["seqJ-shard0.aux.semseg.zarr"] = _FakeRootGroup(
        attrs={"sequence_id": "seqJ", "shard_id": 0, "shard_count": 5},
        root_subgroups={"annotations": _FakeRootSubgroup({"semseg": _FakeRootSubgroup({})})},
    )

    with pytest.raises(ValueError, match="different sequence subdivisions"):
        mod.AuxShardDataLoader(sequence_id="seqJ", dataset_paths=[dp])


def test_aux_loader_init_rejects_duplicate_base_group_for_shard(
    stubbed_zarr_open, tmp_path
):
    """A given base_group_name+shard_id pair must appear at most once across
    stores (the loaded_base_groups sanity check)."""
    mod, registry = stubbed_zarr_open

    dp = _seed_aux_dir(
        tmp_path,
        dataset_basename="seqK-shard0",
        aux_signal_dirnames=[
            "seqK-shard0.aux.depth.zarr",
            "seqK-shard0.aux.depth-second.zarr",
        ],
    )
    # Two stores, same shard_id, both expose a "depth" base group.
    common_attrs = {"sequence_id": "seqK", "shard_id": 0, "shard_count": 1}
    registry["seqK-shard0.aux.depth.zarr"] = _FakeRootGroup(
        attrs=common_attrs,
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )
    registry["seqK-shard0.aux.depth-second.zarr"] = _FakeRootGroup(
        attrs=common_attrs,
        root_subgroups={"annotations": _FakeRootSubgroup({"depth": _FakeRootSubgroup({})})},
    )

    with pytest.raises(ValueError, match="loaded multiple times for shard ID"):
        mod.AuxShardDataLoader(sequence_id="seqK", dataset_paths=[dp])
