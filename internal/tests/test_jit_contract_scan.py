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

"""Branch-coverage tests for ``internal/scripts/jit_contract_scan.py``.

Pure-stdlib script; tests use only synthetic mini JSONs under tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import sys

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def scan_mod():
    sys.path.insert(0, str(REPO_ROOT))
    mod_name = "jit_contract_scan_module_under_test"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(REPO_ROOT / "internal" / "scripts" / "jit_contract_scan.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # dataclasses needs this for frozen field resolution
    spec.loader.exec_module(mod)
    return mod


def _write_clip(
    parent: Path,
    uuid: str,
    duration_us: int,
    cameras: list[str],
    *,
    other_sensors: dict[str, list[str]] | None = None,
    raw: str | None = None,
) -> Path:
    """Build a synthetic ncorev4-shaped JSON header under <parent>/<uuid>/."""
    clip_dir = parent / uuid
    clip_dir.mkdir()
    out = clip_dir / f"pai_{uuid}.json"
    if raw is not None:
        out.write_text(raw)
        return out

    stores: list[dict] = []
    if cameras:
        stores.append(
            {
                "path": f"pai_{uuid}.ncore4-cameras.zarr.itar",
                "md5": "deadbeef",
                "components": {
                    "cameras": {
                        cam: {"version": "v1", "generic_meta_data": {}}
                        for cam in cameras
                    }
                },
            }
        )
    if other_sensors:
        for kind, ids in other_sensors.items():
            stores.append(
                {
                    "path": f"pai_{uuid}.ncore4-{kind}.zarr.itar",
                    "md5": "deadbeef",
                    "components": {
                        kind: {sid: {"version": "v1", "generic_meta_data": {}} for sid in ids}
                    },
                }
            )

    doc = {
        "sequence_id": uuid,
        "sequence_timestamp_interval_us": {"start": 0, "stop": duration_us},
        "version": "v1",
        "component_stores": stores,
    }
    out.write_text(json.dumps(doc))
    return out


# ---------- parse_clip_metadata ----------


def test_parse_metadata_happy_path(scan_mod, tmp_path):
    p = _write_clip(
        tmp_path,
        "uuid-a",
        duration_us=20_000_000,
        cameras=["camera_front_wide_120fov", "camera_rear_tele_30fov"],
        other_sensors={"lidars": ["lidar_top_360fov"], "cuboids": ["default"]},
    )
    info = scan_mod.parse_clip_metadata(p)
    assert info.uuid == "uuid-a"
    assert info.duration_s == 20.0
    assert info.cameras == frozenset(
        {"camera_front_wide_120fov", "camera_rear_tele_30fov"}
    )
    assert info.other_sensors["lidars"] == frozenset({"lidar_top_360fov"})
    assert info.other_sensors["cuboids"] == frozenset({"default"})
    assert info.parse_error is None


def test_parse_metadata_cameras_dedup_across_stores(scan_mod, tmp_path):
    """Same camera id appearing in multiple stores should dedupe."""
    clip_dir = tmp_path / "uuid-b"
    clip_dir.mkdir()
    doc = {
        "sequence_timestamp_interval_us": {"start": 0, "stop": 10_000_000},
        "component_stores": [
            {
                "path": "a.zarr.itar",
                "md5": "x",
                "components": {"cameras": {"camera_front_wide_120fov": {}}},
            },
            {
                "path": "b.zarr.itar",
                "md5": "y",
                "components": {"cameras": {"camera_front_wide_120fov": {}}},
            },
        ],
    }
    p = clip_dir / "pai_uuid-b.json"
    p.write_text(json.dumps(doc))
    info = scan_mod.parse_clip_metadata(p)
    assert info.cameras == frozenset({"camera_front_wide_120fov"})


def test_parse_metadata_invalid_json(scan_mod, tmp_path):
    p = _write_clip(tmp_path, "uuid-c", 0, [], raw="{not valid json")
    info = scan_mod.parse_clip_metadata(p)
    assert info.parse_error is not None
    assert "JSONDecodeError" in info.parse_error
    assert info.duration_s == 0.0
    assert info.cameras == frozenset()


def test_parse_metadata_missing_interval(scan_mod, tmp_path):
    raw = json.dumps({"version": "v1", "component_stores": []})
    p = _write_clip(tmp_path, "uuid-d", 0, [], raw=raw)
    info = scan_mod.parse_clip_metadata(p)
    assert info.parse_error is not None
    assert "sequence_timestamp_interval_us" in info.parse_error


def test_parse_metadata_non_list_component_stores(scan_mod, tmp_path):
    raw = json.dumps(
        {
            "sequence_timestamp_interval_us": {"start": 0, "stop": 5_000_000},
            "component_stores": {"oops": "dict not list"},
        }
    )
    p = _write_clip(tmp_path, "uuid-e", 0, [], raw=raw)
    info = scan_mod.parse_clip_metadata(p)
    assert info.parse_error is not None
    assert "component_stores" in info.parse_error
    assert info.duration_s == 5.0  # still extracted before failure


def test_parse_metadata_malformed_store_entries_skipped(scan_mod, tmp_path):
    """Non-dict store entries and non-dict components should be ignored, not crash."""
    raw = json.dumps(
        {
            "sequence_timestamp_interval_us": {"start": 0, "stop": 20_000_000},
            "component_stores": [
                "not-a-dict",
                {"path": "x", "components": "not-a-dict"},
                {
                    "path": "y",
                    "components": {
                        "cameras": "not-a-dict",
                        "lidars": {"lidar_top_360fov": {}},
                    },
                },
            ],
        }
    )
    p = _write_clip(tmp_path, "uuid-f", 0, [], raw=raw)
    info = scan_mod.parse_clip_metadata(p)
    assert info.parse_error is None
    assert info.cameras == frozenset()
    assert info.other_sensors["lidars"] == frozenset({"lidar_top_360fov"})


def test_parse_metadata_invalid_interval_values(scan_mod, tmp_path):
    raw = json.dumps(
        {
            "sequence_timestamp_interval_us": {"start": "not-an-int", "stop": 5},
            "component_stores": [],
        }
    )
    p = _write_clip(tmp_path, "uuid-g", 0, [], raw=raw)
    info = scan_mod.parse_clip_metadata(p)
    assert info.parse_error is not None


# ---------- check_camera_support ----------


def _info(scan_mod, *, cameras: list[str], duration_s: float = 20.0, parse_error=None):
    return scan_mod.ClipInfo(
        uuid="uuid",
        json_path=Path("/dev/null"),
        duration_s=duration_s,
        cameras=frozenset(cameras),
        other_sensors={},
        parse_error=parse_error,
    )


def test_check_camera_support_happy(scan_mod):
    info = _info(scan_mod, cameras=["camera_front_wide_120fov"], duration_s=20.0)
    v = scan_mod.check_camera_support(info, "camera_front_wide_120fov")
    assert v.supported is True
    assert v.reasons == ()
    assert v.estimated_frames == 200  # 20s * 10fps default


def test_check_camera_support_missing_camera(scan_mod):
    info = _info(scan_mod, cameras=["camera_rear_tele_30fov"], duration_s=20.0)
    v = scan_mod.check_camera_support(info, "camera_front_wide_120fov")
    assert v.supported is False
    assert any("not in clip" in r for r in v.reasons)


def test_check_camera_support_duration_too_short(scan_mod):
    info = _info(scan_mod, cameras=["camera_front_wide_120fov"], duration_s=1.0)
    v = scan_mod.check_camera_support(info, "camera_front_wide_120fov")
    assert v.supported is False
    assert any("estimated frames" in r for r in v.reasons)


def test_check_camera_support_propagates_parse_error(scan_mod):
    info = _info(scan_mod, cameras=[], duration_s=0.0, parse_error="bad json")
    v = scan_mod.check_camera_support(info, "camera_front_wide_120fov")
    assert v.supported is False
    assert any("parse_error" in r for r in v.reasons)


def test_check_camera_support_boundary_exact_frame_count(scan_mod):
    """duration * fps exactly equal to min_frames passes (>= boundary)."""
    info = _info(scan_mod, cameras=["cam"], duration_s=1.8)  # 1.8 * 10 = 18
    v = scan_mod.check_camera_support(info, "cam", min_frames=18, nominal_fps=10.0)
    assert v.supported is True


def test_check_camera_support_boundary_just_below(scan_mod):
    info = _info(scan_mod, cameras=["cam"], duration_s=1.79)  # 1.79 * 10 = 17.9 -> int 17
    v = scan_mod.check_camera_support(info, "cam", min_frames=18, nominal_fps=10.0)
    assert v.supported is False


def test_check_camera_support_accumulates_multiple_reasons(scan_mod):
    info = _info(scan_mod, cameras=["other_cam"], duration_s=0.5)
    v = scan_mod.check_camera_support(info, "camera_front_wide_120fov")
    assert v.supported is False
    assert len(v.reasons) == 2


# ---------- iter_clip_jsons ----------


def test_iter_clip_jsons_finds_matches(scan_mod, tmp_path):
    _write_clip(tmp_path, "u-a", 10_000_000, ["cam_a"])
    _write_clip(tmp_path, "u-b", 10_000_000, ["cam_b"])
    files = list(scan_mod.iter_clip_jsons(tmp_path))
    assert [f.parent.name for f in files] == ["u-a", "u-b"]
    assert all(f.name.startswith("pai_") and f.name.endswith(".json") for f in files)


def test_iter_clip_jsons_skips_non_dirs_and_non_pai_files(scan_mod, tmp_path):
    (tmp_path / "loose_file.json").write_text("{}")
    _write_clip(tmp_path, "real", 10_000_000, ["cam"])
    sub = tmp_path / "no-pai"
    sub.mkdir()
    (sub / "something_else.json").write_text("{}")
    files = list(scan_mod.iter_clip_jsons(tmp_path))
    assert [f.parent.name for f in files] == ["real"]


def test_iter_clip_jsons_empty_dir(scan_mod, tmp_path):
    assert list(scan_mod.iter_clip_jsons(tmp_path)) == []


# ---------- _camera_universe ----------


def test_camera_universe_dedupes_and_sorts(scan_mod):
    a = _info(scan_mod, cameras=["b", "a"])
    b = _info(scan_mod, cameras=["c", "a"])
    assert scan_mod._camera_universe([a, b]) == ["a", "b", "c"]


def test_camera_universe_empty(scan_mod):
    assert scan_mod._camera_universe([]) == []


# ---------- build_report ----------


def test_build_report_aggregates_per_camera(scan_mod, tmp_path):
    clips = [
        _info(scan_mod, cameras=["cam_a", "cam_b"], duration_s=20.0),
        _info(scan_mod, cameras=["cam_a"], duration_s=20.0),  # missing cam_b
        _info(scan_mod, cameras=["cam_a", "cam_b"], duration_s=0.5),  # too short
    ]
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        clips=clips,
        camera_ids=["cam_a", "cam_b"],
        min_frames=18,
        nominal_fps=10.0,
        max_chunks=8,
        include_clip_details=False,
    )
    assert rep["per_camera"]["cam_a"]["n_supported"] == 2
    assert rep["per_camera"]["cam_a"]["n_unsupported"] == 1
    assert rep["per_camera"]["cam_b"]["n_supported"] == 1
    assert rep["per_camera"]["cam_b"]["n_unsupported"] == 2
    assert rep["summary"]["n_clips_total"] == 3
    assert rep["summary"]["n_unique_camera_sets"] == 2  # {a,b} and {a}
    assert "clips" not in rep


def test_build_report_truncation_flag(scan_mod, tmp_path):
    clips = [
        _info(scan_mod, cameras=["c"], duration_s=20.0),
        _info(scan_mod, cameras=["c"], duration_s=200.0),  # > 8 * 13.5 = 108s
    ]
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        clips=clips,
        camera_ids=["c"],
        min_frames=18,
        nominal_fps=10.0,
        max_chunks=8,
        include_clip_details=False,
    )
    assert rep["max_chunks_coverage"]["n_clips_truncated"] == 1


def test_build_report_records_parse_errors(scan_mod, tmp_path):
    bad = _info(scan_mod, cameras=[], duration_s=0.0, parse_error="boom")
    good = _info(scan_mod, cameras=["c"], duration_s=20.0)
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        clips=[bad, good],
        camera_ids=["c"],
        min_frames=18,
        nominal_fps=10.0,
        max_chunks=8,
        include_clip_details=False,
    )
    assert rep["summary"]["n_clips_parse_error"] == 1
    assert rep["summary"]["n_clips_parseable"] == 1
    assert rep["parse_errors"][0]["uuid"] == "uuid"
    # Parse-error clips never reach per_camera tallies.
    assert rep["per_camera"]["c"]["n_supported"] == 1
    assert rep["per_camera"]["c"]["n_unsupported"] == 0


def test_build_report_include_clip_details(scan_mod, tmp_path):
    clips = [_info(scan_mod, cameras=["x"], duration_s=20.0)]
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        clips=clips,
        camera_ids=["x"],
        min_frames=18,
        nominal_fps=10.0,
        max_chunks=8,
        include_clip_details=True,
    )
    assert "clips" in rep
    assert rep["clips"][0]["cameras"] == ["x"]
    assert rep["clips"][0]["duration_s"] == 20.0


# ---------- main / CLI ----------


def test_main_full_run(scan_mod, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(
        clips_dir,
        "u-good",
        duration_us=20_000_000,
        cameras=["camera_front_wide_120fov"],
        other_sensors={"lidars": ["lidar_top_360fov"]},
    )
    _write_clip(
        clips_dir,
        "u-bad",
        duration_us=500_000,  # 0.5s
        cameras=["camera_front_wide_120fov"],
    )
    _write_clip(
        clips_dir,
        "u-missing",
        duration_us=20_000_000,
        cameras=["camera_rear_tele_30fov"],
    )
    out = tmp_path / "report.json"
    rc = scan_mod.main(
        [
            "--clips-dir",
            str(clips_dir),
            "--output",
            str(out),
            "--camera-id",
            "camera_front_wide_120fov",
            "--include-clip-details",
        ]
    )
    assert rc == 0
    rep = json.loads(out.read_text())
    cam = rep["per_camera"]["camera_front_wide_120fov"]
    assert cam["n_supported"] == 1
    assert cam["n_unsupported"] == 2
    unsupp_uuids = {u["uuid"] for u in cam["unsupported"]}
    assert unsupp_uuids == {"u-bad", "u-missing"}
    assert "clips" in rep


def test_main_default_camera_id_matrix(scan_mod, tmp_path):
    """No --camera-id => evaluate every camera id ever seen."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir, "ua", 20_000_000, ["cam_a"])
    _write_clip(clips_dir, "ub", 20_000_000, ["cam_b"])
    out = tmp_path / "r.json"
    rc = scan_mod.main(["--clips-dir", str(clips_dir), "--output", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert set(rep["per_camera"].keys()) == {"cam_a", "cam_b"}
    assert rep["per_camera"]["cam_a"]["n_supported"] == 1
    assert rep["per_camera"]["cam_b"]["n_supported"] == 1


def test_main_clips_dir_does_not_exist(scan_mod, tmp_path):
    out = tmp_path / "r.json"
    rc = scan_mod.main(
        ["--clips-dir", str(tmp_path / "nope"), "--output", str(out)]
    )
    assert rc == 2


def test_main_empty_clips_dir(scan_mod, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    out = tmp_path / "r.json"
    rc = scan_mod.main(["--clips-dir", str(clips_dir), "--output", str(out)])
    assert rc == 1


def test_main_dedupes_repeated_camera_id(scan_mod, tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir, "ua", 20_000_000, ["cam_a"])
    out = tmp_path / "r.json"
    rc = scan_mod.main(
        [
            "--clips-dir",
            str(clips_dir),
            "--output",
            str(out),
            "--camera-id",
            "cam_a",
            "--camera-id",
            "cam_a",
        ]
    )
    assert rc == 0
    rep = json.loads(out.read_text())
    assert list(rep["per_camera"].keys()) == ["cam_a"]


def test_print_stdout_summary_with_truncation(scan_mod, capsys, tmp_path):
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        clips=[_info(scan_mod, cameras=["c"], duration_s=200.0)],
        camera_ids=["c"],
        min_frames=18,
        nominal_fps=10.0,
        max_chunks=8,
        include_clip_details=False,
    )
    scan_mod._print_stdout_summary(rep)
    out = capsys.readouterr().out
    assert "scanned 1 clip" in out
    assert "exceed --max-chunks" in out
    assert "supported" in out


def test_print_stdout_summary_no_truncation(scan_mod, capsys, tmp_path):
    rep = scan_mod.build_report(
        clips_dir=tmp_path,
        clips=[_info(scan_mod, cameras=["c"], duration_s=20.0)],
        camera_ids=["c"],
        min_frames=18,
        nominal_fps=10.0,
        max_chunks=8,
        include_clip_details=False,
    )
    scan_mod._print_stdout_summary(rep)
    out = capsys.readouterr().out
    assert "exceed --max-chunks" not in out
