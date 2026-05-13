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

#!/usr/bin/env python3
"""Scan an ncorev4 ``clips/`` tree for the per-clip external-distortion type.

Companion to ``jit_contract_scan.py``: that one reads the JSON header
(no zarr open) and tells us which clips satisfy the JIT input contract.
This one opens the zarr archive's small calibration record (no image
chunks touched) and tells us which clips carry a
``BivariateWindshieldModelParameters`` external distortion on the
selected camera id.

The standalone's ray-gen path
(``instant_nurec/utils/sensors/ray_gen.py``) currently raises
``NotImplementedError`` for anything other than ``NoExternalDistortion``,
so the list of windshield-carrying clips is exactly the set that the
standalone *cannot* process today.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CAMERA_ID: str = "camera_front_wide_120fov"


@dataclass(frozen=True)
class ClipScan:
    uuid: str
    json_path: Path
    camera_id: str
    has_windshield: bool | None  # None if scan errored
    external_distortion_type: str | None
    error: str | None = None
    elapsed_s: float = 0.0


def iter_clip_jsons(clips_dir: Path) -> Iterator[Path]:
    for child in sorted(clips_dir.iterdir()):
        if not child.is_dir():
            continue
        for json_path in sorted(child.glob("pai_*.json")):
            yield json_path


def scan_one_clip(json_path: Path, camera_id: str) -> ClipScan:
    """Open the clip's calibration record (zarr metadata only) and read
    the selected camera's external_distortion_parameters type."""
    from instant_nurec.utils.ncore_utils import create_sequence_loader
    from instant_nurec.utils.files import parse_universal_path

    uuid = json_path.parent.name
    t0 = time.monotonic()
    try:
        dataset_path = parse_universal_path(str(json_path))
        seq = create_sequence_loader(
            dataset_paths=[dataset_path],
            open_consolidated=False,
            v4_poses_component_group="default",
            v4_intrinsics_component_group="default",
            v4_masks_component_group="default",
            v4_cuboids_component_group="default",
        )
        if camera_id not in seq.camera_ids:
            return ClipScan(
                uuid=uuid,
                json_path=json_path,
                camera_id=camera_id,
                has_windshield=None,
                external_distortion_type=None,
                error=f"camera_id '{camera_id}' not in sequence's camera_ids {sorted(seq.camera_ids)}",
                elapsed_s=time.monotonic() - t0,
            )
        sensor = seq.get_camera_sensor(camera_id)
        params = sensor.model_parameters
        external = getattr(params, "external_distortion_parameters", None)
        ext_type = type(external).__name__ if external is not None else "None"
        has_bw = ext_type != "None"  # any non-None is windshield-like; refine if needed
        return ClipScan(
            uuid=uuid,
            json_path=json_path,
            camera_id=camera_id,
            has_windshield=has_bw,
            external_distortion_type=ext_type,
            error=None,
            elapsed_s=time.monotonic() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return ClipScan(
            uuid=uuid,
            json_path=json_path,
            camera_id=camera_id,
            has_windshield=None,
            external_distortion_type=None,
            error=f"{type(exc).__name__}: {exc}"[:400],
            elapsed_s=time.monotonic() - t0,
        )


def build_report(
    clips_dir: Path, scans: list[ClipScan], camera_id: str
) -> dict:
    with_bw = [s for s in scans if s.has_windshield is True]
    without_bw = [s for s in scans if s.has_windshield is False]
    errored = [s for s in scans if s.has_windshield is None]

    types_count: dict[str, int] = {}
    for s in scans:
        if s.external_distortion_type is not None:
            types_count[s.external_distortion_type] = types_count.get(
                s.external_distortion_type, 0
            ) + 1

    return {
        "version": 1,
        "scan_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "clips_dir": str(clips_dir),
        "camera_id": camera_id,
        "summary": {
            "n_total": len(scans),
            "n_with_windshield": len(with_bw),
            "n_without_windshield": len(without_bw),
            "n_errored": len(errored),
            "distortion_type_counts": types_count,
        },
        "windshield_uuids": sorted(s.uuid for s in with_bw),
        "no_windshield_uuids": sorted(s.uuid for s in without_bw),
        "errors": [
            {"uuid": s.uuid, "error": s.error, "elapsed_s": s.elapsed_s}
            for s in errored
        ],
    }


def _print_progress(idx: int, total: int, scan: ClipScan) -> None:
    if scan.error is not None:
        tag = f"ERROR ({scan.error[:40]}...)"
    elif scan.has_windshield:
        tag = f"WINDSHIELD ({scan.external_distortion_type})"
    else:
        tag = "no-windshield"
    print(
        f"[{idx:5d}/{total:5d}] {scan.uuid}  {tag:40s}  "
        f"{scan.elapsed_s:6.2f}s",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an ncorev4 clips/ tree for clips whose selected camera "
            "carries an external-distortion model (e.g. BivariateWindshield) "
            "the standalone's torch ray-gen path does not implement."
        )
    )
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--camera-id",
        type=str,
        default=DEFAULT_CAMERA_ID,
        help=f"Camera id to probe (default: {DEFAULT_CAMERA_ID}).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Stop after N clips (debug / smoke).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.clips_dir.is_dir():
        print(f"error: --clips-dir '{args.clips_dir}' is not a directory", file=sys.stderr)
        return 2

    jsons = list(iter_clip_jsons(args.clips_dir))
    if not jsons:
        print(f"error: no pai_*.json under {args.clips_dir}", file=sys.stderr)
        return 1
    if args.max_clips is not None:
        jsons = jsons[: args.max_clips]

    scans: list[ClipScan] = []
    t_start = time.monotonic()
    for i, p in enumerate(jsons, 1):
        s = scan_one_clip(p, args.camera_id)
        scans.append(s)
        _print_progress(i, len(jsons), s)

    report = build_report(args.clips_dir, scans, args.camera_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    elapsed = time.monotonic() - t_start
    print(
        f"\n=== windshield: {s['n_with_windshield']} / {s['n_total']}  "
        f"(no_wind: {s['n_without_windshield']}, err: {s['n_errored']})"
        f"  total {elapsed:.1f}s"
    )
    print(f"  distortion type counts: {s['distortion_type_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
