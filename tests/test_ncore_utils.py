"""Branch-coverage tests for nre.utils.ncore_utils.

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
# Stub fixture — installs ncore.* + lietorch + ncore.data unions, then loads
# nre.utils.ncore_utils.
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed_ncore_utils(monkeypatch):
    # lietorch (transitively imported via nre.utils.types)
    lietorch_mod = types.ModuleType("lietorch")

    class _FakeSE3:
        pass

    lietorch_mod.SE3 = _FakeSE3
    monkeypatch.setitem(sys.modules, "lietorch", lietorch_mod)

    # torchvision binary in this venv is incompatible with the cpu torch
    # build (op-registration conflict on torchvision::nms). Stub the only
    # surface ncore_utils touches: the resize transform.
    torchvision_mod = types.ModuleType("torchvision")
    transforms_mod = types.ModuleType("torchvision.transforms")
    functional_mod = types.ModuleType("torchvision.transforms.functional")

    def _fake_resize(tensor, target_hw, antialias=True):
        # Naive nearest-neighbour resize using torch.nn.functional.interpolate.
        import torch.nn.functional as F

        return F.interpolate(tensor, size=tuple(target_hw), mode="nearest")

    functional_mod.resize = _fake_resize
    transforms_mod.functional = functional_mod
    torchvision_mod.transforms = transforms_mod
    monkeypatch.setitem(sys.modules, "torchvision", torchvision_mod)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms_mod)
    monkeypatch.setitem(sys.modules, "torchvision.transforms.functional", functional_mod)

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
    for cached in ("nre.utils.ncore_utils", "nre.utils.types"):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    import importlib

    mod = importlib.import_module("nre.utils.ncore_utils")
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
