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
"""Scan an ncorev4 ``clips/`` tree against the JIT model's input contract.

The shipped ``kelvin_jit.pt`` is a TorchScript-traced module with four
shape constants baked in as persistent buffers (see
``internal/scripts/export_kelvin_jit.py``)::

    expected_b = 1     # batch
    expected_v = 18    # n_frames_per_sample (1 context-camera * 18 frames)
    expected_h = 448   # rectified image height
    expected_w = 784   # rectified image width

Rectified ``H x W`` is normalized by the dataset's frame-batch sampler
regardless of each camera's native resolution, so the only per-clip
facts that can violate the contract for a chosen ``--camera-id`` are:

    1. Does the clip's metadata header list that camera at all?
    2. Does the clip have enough frames for an 18-frame window?

Frame counts proper live inside the zarr-itar archives; this scan
deliberately does **not** open them. Duration is read from the JSON
header and converted to an estimated frame count using
``--nominal-fps``. The result is a fast (seconds for ~1000 clips)
support matrix per ``--camera-id``; clips flagged "duration too short"
should be re-verified by Stage B (dataset-load smoke) before being
declared unsupported.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


CHUNK_SECONDS: float = 13.5  # one Kelvin chunk window (matches CLI default)
DEFAULT_EXPECTED_V: int = 18
DEFAULT_NOMINAL_FPS: float = 10.0


@dataclass(frozen=True)
class ClipInfo:
    """Per-clip facts extracted from the ncorev4 sequence JSON header."""

    uuid: str
    json_path: Path
    duration_s: float
    cameras: frozenset[str]
    other_sensors: dict[str, frozenset[str]]
    parse_error: str | None = None


@dataclass(frozen=True)
class CameraSupport:
    """Per-(clip, camera) JIT-contract verdict."""

    uuid: str
    camera_id: str
    supported: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    duration_s: float = 0.0
    estimated_frames: int = 0


def parse_clip_metadata(json_path: Path) -> ClipInfo:
    """Read one ``pai_<uuid>.json`` header and extract clip facts.

    The metadata format observed (ncorev4) has::

        {
          "sequence_id": "...",
          "sequence_timestamp_interval_us": {"start": int, "stop": int},
          "component_stores": [
            {"path": "...", "md5": "...", "components": {
              "<kind>": {"<sensor_id>": {...}, ...},
              ...
            }},
            ...
          ],
          ...
        }

    ``<kind>`` is one of ``cameras``, ``lidars``, ``cuboids``,
    ``intrinsics``, ``masks``, ``poses``, ... A single store may carry
    multiple sensors and the same sensor kind may appear across stores.
    """
    uuid = json_path.parent.name
    try:
        with json_path.open() as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return ClipInfo(
            uuid=uuid,
            json_path=json_path,
            duration_s=0.0,
            cameras=frozenset(),
            other_sensors={},
            parse_error=f"{type(exc).__name__}: {exc}",
        )

    try:
        interval = doc["sequence_timestamp_interval_us"]
        start_us = int(interval["start"])
        stop_us = int(interval["stop"])
    except (KeyError, TypeError, ValueError) as exc:
        return ClipInfo(
            uuid=uuid,
            json_path=json_path,
            duration_s=0.0,
            cameras=frozenset(),
            other_sensors={},
            parse_error=f"missing/invalid sequence_timestamp_interval_us: {exc}",
        )

    duration_s = (stop_us - start_us) / 1_000_000.0

    stores = doc.get("component_stores")
    if not isinstance(stores, list):
        return ClipInfo(
            uuid=uuid,
            json_path=json_path,
            duration_s=duration_s,
            cameras=frozenset(),
            other_sensors={},
            parse_error="missing or non-list component_stores",
        )

    cameras: set[str] = set()
    others: dict[str, set[str]] = {}
    for store in stores:
        if not isinstance(store, dict):
            continue
        comps = store.get("components")
        if not isinstance(comps, dict):
            continue
        for kind, kdict in comps.items():
            if not isinstance(kdict, dict):
                continue
            ids = {sid for sid in kdict.keys()}
            if kind == "cameras":
                cameras.update(ids)
            else:
                others.setdefault(kind, set()).update(ids)

    return ClipInfo(
        uuid=uuid,
        json_path=json_path,
        duration_s=duration_s,
        cameras=frozenset(cameras),
        other_sensors={k: frozenset(v) for k, v in others.items()},
        parse_error=None,
    )


def check_camera_support(
    clip: ClipInfo,
    camera_id: str,
    min_frames: int = DEFAULT_EXPECTED_V,
    nominal_fps: float = DEFAULT_NOMINAL_FPS,
) -> CameraSupport:
    """Verify ``clip`` satisfies the JIT input contract for ``camera_id``.

    Two gates, in order:

    1. The clip's ``component_stores`` must list ``camera_id`` under
       ``cameras``. A missing camera is a definitive contract violation
       — the JIT graph has no fallback path that can synthesize a
       camera that doesn't exist in the source data.
    2. ``duration_s * nominal_fps`` must reach ``min_frames``. This is
       a heuristic: the true frame count lives inside the zarr archive,
       which we deliberately do not open. False positives here (clip
       flagged as too-short but actually has enough frames) are caught
       by Stage B (dataset-load smoke).
    """
    reasons: list[str] = []
    if clip.parse_error is not None:
        return CameraSupport(
            uuid=clip.uuid,
            camera_id=camera_id,
            supported=False,
            reasons=(f"parse_error: {clip.parse_error}",),
            duration_s=clip.duration_s,
            estimated_frames=0,
        )

    if camera_id not in clip.cameras:
        reasons.append(f"camera '{camera_id}' not in clip")

    estimated_frames = int(clip.duration_s * nominal_fps)
    if estimated_frames < min_frames:
        reasons.append(
            f"duration {clip.duration_s:.2f}s * {nominal_fps:g}fps "
            f"= {estimated_frames} estimated frames < required {min_frames} "
            f"(heuristic; verify with Stage B)"
        )

    return CameraSupport(
        uuid=clip.uuid,
        camera_id=camera_id,
        supported=not reasons,
        reasons=tuple(reasons),
        duration_s=clip.duration_s,
        estimated_frames=estimated_frames,
    )


def iter_clip_jsons(clips_dir: Path) -> Iterator[Path]:
    """Yield ``pai_<uuid>.json`` files directly under ``<clips_dir>/<uuid>/``.

    Subdirectories whose contents do not match ``pai_*.json`` are skipped
    silently. Returns paths sorted by uuid for deterministic output.
    """
    for child in sorted(clips_dir.iterdir()):
        if not child.is_dir():
            continue
        for json_path in sorted(child.glob("pai_*.json")):
            yield json_path


def _camera_universe(clips: Iterable[ClipInfo]) -> list[str]:
    seen: set[str] = set()
    for clip in clips:
        seen.update(clip.cameras)
    return sorted(seen)


def build_report(
    clips_dir: Path,
    clips: list[ClipInfo],
    camera_ids: list[str],
    min_frames: int,
    nominal_fps: float,
    max_chunks: int,
    include_clip_details: bool,
) -> dict:
    """Aggregate per-clip facts and per-camera support into the report dict."""
    parse_errors = [c for c in clips if c.parse_error is not None]
    parseable = [c for c in clips if c.parse_error is None]

    per_camera: dict[str, dict] = {}
    for cam in camera_ids:
        supported_uuids: list[str] = []
        unsupported: list[dict] = []
        for clip in parseable:
            verdict = check_camera_support(
                clip, cam, min_frames=min_frames, nominal_fps=nominal_fps
            )
            if verdict.supported:
                supported_uuids.append(verdict.uuid)
            else:
                unsupported.append(
                    {
                        "uuid": verdict.uuid,
                        "reasons": list(verdict.reasons),
                        "duration_s": verdict.duration_s,
                        "estimated_frames": verdict.estimated_frames,
                    }
                )
        per_camera[cam] = {
            "n_supported": len(supported_uuids),
            "n_unsupported": len(unsupported),
            "supported_uuids": supported_uuids,
            "unsupported": unsupported,
        }

    camera_set_counts: dict[str, int] = {}
    for clip in parseable:
        key = ",".join(sorted(clip.cameras)) or "<empty>"
        camera_set_counts[key] = camera_set_counts.get(key, 0) + 1

    truncation_seconds = max_chunks * CHUNK_SECONDS
    coverage_truncated_uuids = [
        clip.uuid for clip in parseable if clip.duration_s > truncation_seconds
    ]

    report: dict = {
        "version": 1,
        "scan_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "clips_dir": str(clips_dir),
        "jit_contract": {
            "expected_v": min_frames,
            "min_frames_per_camera": min_frames,
            "nominal_fps_assumed": nominal_fps,
        },
        "summary": {
            "n_clips_total": len(clips),
            "n_clips_parseable": len(parseable),
            "n_clips_parse_error": len(parse_errors),
            "n_unique_camera_sets": len(camera_set_counts),
            "camera_set_counts": camera_set_counts,
        },
        "max_chunks_coverage": {
            "max_chunks": max_chunks,
            "chunk_seconds": CHUNK_SECONDS,
            "max_runtime_seconds": truncation_seconds,
            "n_clips_truncated": len(coverage_truncated_uuids),
            "truncated_uuids": coverage_truncated_uuids,
        },
        "per_camera": per_camera,
    }
    if parse_errors:
        report["parse_errors"] = [
            {"uuid": c.uuid, "path": str(c.json_path), "error": c.parse_error}
            for c in parse_errors
        ]
    if include_clip_details:
        report["clips"] = [
            {
                "uuid": c.uuid,
                "duration_s": c.duration_s,
                "cameras": sorted(c.cameras),
                "other_sensors": {k: sorted(v) for k, v in c.other_sensors.items()},
                "parse_error": c.parse_error,
            }
            for c in clips
        ]
    return report


def _print_stdout_summary(report: dict) -> None:
    s = report["summary"]
    print(
        f"scanned {s['n_clips_total']} clip(s) "
        f"({s['n_clips_parse_error']} parse error(s), "
        f"{s['n_unique_camera_sets']} unique camera-set(s))"
    )
    mc = report["max_chunks_coverage"]
    if mc["n_clips_truncated"] > 0:
        print(
            f"  {mc['n_clips_truncated']} clip(s) exceed "
            f"--max-chunks={mc['max_chunks']} window "
            f"({mc['max_runtime_seconds']:.1f}s) and would be truncated"
        )
    print("per camera-id:")
    for cam, info in report["per_camera"].items():
        total = info["n_supported"] + info["n_unsupported"]
        print(f"  {cam:40s} {info['n_supported']:5d}/{total:5d} supported")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an ncorev4 clips/ tree against the JIT model's "
            "(camera-id, frame-count) input contract."
        )
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        required=True,
        help="Directory containing <uuid>/pai_<uuid>.json subdirs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--camera-id",
        action="append",
        default=None,
        help=(
            "Camera id to evaluate. Repeat to produce a matrix. "
            "Default: every camera id seen anywhere in the clips tree."
        ),
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_EXPECTED_V,
        help=f"Minimum frames per camera required (default: {DEFAULT_EXPECTED_V}).",
    )
    parser.add_argument(
        "--nominal-fps",
        type=float,
        default=DEFAULT_NOMINAL_FPS,
        help=(
            f"Assumed camera frame rate for the duration -> frame-count "
            f"heuristic (default: {DEFAULT_NOMINAL_FPS}). The actual frame "
            f"count is only knowable from the zarr archive, which this "
            f"scan does not open."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=8,
        help=(
            "Match the standalone CLI --max-chunks. Used to flag clips "
            "whose duration exceeds --max-chunks * 13.5s and would be "
            "silently truncated at inference (informational; not a "
            "support gate)."
        ),
    )
    parser.add_argument(
        "--include-clip-details",
        action="store_true",
        help="Embed per-clip facts (cameras, sensors, duration) in the report.",
    )

    args = parser.parse_args(argv)

    if not args.clips_dir.is_dir():
        print(f"error: --clips-dir '{args.clips_dir}' is not a directory", file=sys.stderr)
        return 2

    clips: list[ClipInfo] = []
    for json_path in iter_clip_jsons(args.clips_dir):
        clips.append(parse_clip_metadata(json_path))

    if not clips:
        print(f"error: no pai_*.json files found under {args.clips_dir}", file=sys.stderr)
        return 1

    if args.camera_id:
        camera_ids = list(dict.fromkeys(args.camera_id))
    else:
        camera_ids = _camera_universe(clips)

    report = build_report(
        clips_dir=args.clips_dir,
        clips=clips,
        camera_ids=camera_ids,
        min_frames=args.min_frames,
        nominal_fps=args.nominal_fps,
        max_chunks=args.max_chunks,
        include_clip_details=args.include_clip_details,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2)

    _print_stdout_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
