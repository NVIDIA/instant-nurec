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
"""Dataset-load smoke pass over an ncorev4 ``clips/`` tree.

Complements ``internal/scripts/jit_contract_scan.py``. That one checks
the JIT input contract from the JSON header alone (no zarr reads); this
one constructs ``NCoreInstantNuRecDataset`` for each clip and pulls one
sample, surfacing reader-level failures that aren't visible in the
header:

* sensor calibration fields missing or malformed in the zarr archive,
* cuboid-track schema variations,
* timestamp ranges that the frame batcher rejects,
* anything that throws inside ``__init__`` or ``__getitem__``.

Inference is **not** invoked — no GPU, no model load. Per-clip cost is
dominated by network reads of the sensor zarr archive (typically
seconds), so a full sweep over a 1000+ clip dataset takes tens of
minutes wall time on networked storage.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CAMERA_ID: str = "camera_front_wide_120fov"
DEFAULT_FRAME_WIDTH: int = 784
DEFAULT_FRAME_HEIGHT: int = 448
DEFAULT_N_FRAMES_PER_SAMPLE: int = 18
DEFAULT_MAX_CHUNKS: int = 8


@dataclass(frozen=True)
class ClipVerdict:
    """Outcome of the smoke pass against a single clip."""

    uuid: str
    json_path: Path
    passed: bool
    stage: str = "init"  # "init", "getitem", or "ok"
    exception_type: str | None = None
    exception_message: str = ""
    traceback_tail: list[str] = field(default_factory=list)
    n_items_fetched: int = 0
    elapsed_s: float = 0.0


def iter_clip_jsons(clips_dir: Path) -> Iterator[Path]:
    """Yield ``<uuid>/pai_<uuid>.json`` files in deterministic order."""
    for child in sorted(clips_dir.iterdir()):
        if not child.is_dir():
            continue
        for json_path in sorted(child.glob("pai_*.json")):
            yield json_path


def _format_traceback_tail(exc: BaseException, n_lines: int = 5) -> list[str]:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    full = "".join(tb).splitlines()
    return full[-n_lines:]


def smoke_test_one_clip(
    json_path: Path,
    *,
    camera_id: str,
    max_chunks: int,
    frame_width: int,
    frame_height: int,
    n_frames_per_sample: int,
    items_per_clip: int,
    dataset_factory,
    config_factory,
) -> ClipVerdict:
    """Run the construct → ``__getitem__`` smoke against one clip.

    ``dataset_factory`` and ``config_factory`` are injected so tests can
    swap in lightweight stand-ins without paying the real-import cost.
    They take the same arguments as
    ``NCoreInstantNuRecDataset(NCoreInstantNuRecDatasetConfig(...), ...)``
    in ``instant_nurec.cli``.
    """
    uuid = json_path.parent.name
    t0 = time.monotonic()
    try:
        config = config_factory(
            ncore_json_paths=[str(json_path)],
            context_camera_ids=[camera_id],
            supervision_camera_ids=[camera_id],
            n_samples_per_sequence=max_chunks,
        )
    except Exception as exc:  # noqa: BLE001 — we surface everything
        return ClipVerdict(
            uuid=uuid,
            json_path=json_path,
            passed=False,
            stage="config",
            exception_type=type(exc).__name__,
            exception_message=str(exc)[:500],
            traceback_tail=_format_traceback_tail(exc),
            n_items_fetched=0,
            elapsed_s=time.monotonic() - t0,
        )

    try:
        ds = dataset_factory(
            config=config,
            frame_width=frame_width,
            frame_height=frame_height,
            n_frames_per_sample=n_frames_per_sample,
        )
    except Exception as exc:  # noqa: BLE001
        return ClipVerdict(
            uuid=uuid,
            json_path=json_path,
            passed=False,
            stage="init",
            exception_type=type(exc).__name__,
            exception_message=str(exc)[:500],
            traceback_tail=_format_traceback_tail(exc),
            n_items_fetched=0,
            elapsed_s=time.monotonic() - t0,
        )

    n_fetched = 0
    try:
        n = min(items_per_clip, len(ds))
        for idx in range(n):
            _ = ds[idx]
            n_fetched += 1
    except Exception as exc:  # noqa: BLE001
        return ClipVerdict(
            uuid=uuid,
            json_path=json_path,
            passed=False,
            stage="getitem",
            exception_type=type(exc).__name__,
            exception_message=str(exc)[:500],
            traceback_tail=_format_traceback_tail(exc),
            n_items_fetched=n_fetched,
            elapsed_s=time.monotonic() - t0,
        )

    return ClipVerdict(
        uuid=uuid,
        json_path=json_path,
        passed=True,
        stage="ok",
        n_items_fetched=n_fetched,
        elapsed_s=time.monotonic() - t0,
    )


def _default_dataset_factory():
    """Real factory: imports and constructs the standalone dataset class."""
    from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset

    def make(*, config, frame_width, frame_height, n_frames_per_sample):
        return NCoreInstantNuRecDataset(
            config=config,
            frame_width=frame_width,
            frame_height=frame_height,
            n_frames_per_sample=n_frames_per_sample,
        )

    return make


def _default_config_factory():
    """Real factory: builds an ``NCoreInstantNuRecDatasetConfig`` instance."""
    from instant_nurec.config_schema.dataset import (
        AdaptiveSequentialFrameBatchSamplerConfig,
        NCoreInstantNuRecDatasetConfig,
    )

    def make(*, ncore_json_paths, context_camera_ids, supervision_camera_ids,
             n_samples_per_sequence):
        return NCoreInstantNuRecDatasetConfig(
            ncore_json_paths=ncore_json_paths,
            context_camera_ids=context_camera_ids,
            supervision_camera_ids=supervision_camera_ids,
            frame_batch_sampler=AdaptiveSequentialFrameBatchSamplerConfig(
                n_samples_per_sequence=n_samples_per_sequence,
            ),
        )

    return make


def build_report(
    clips_dir: Path,
    verdicts: list[ClipVerdict],
    camera_id: str,
    max_chunks: int,
    items_per_clip: int,
) -> dict:
    passed = [v for v in verdicts if v.passed]
    failed = [v for v in verdicts if not v.passed]

    by_stage: dict[str, int] = {}
    by_exception: dict[str, int] = {}
    for v in failed:
        by_stage[v.stage] = by_stage.get(v.stage, 0) + 1
        if v.exception_type is not None:
            by_exception[v.exception_type] = by_exception.get(v.exception_type, 0) + 1

    return {
        "version": 1,
        "scan_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "clips_dir": str(clips_dir),
        "config": {
            "camera_id": camera_id,
            "max_chunks": max_chunks,
            "items_per_clip": items_per_clip,
        },
        "summary": {
            "n_total": len(verdicts),
            "n_passed": len(passed),
            "n_failed": len(failed),
            "failures_by_stage": by_stage,
            "failures_by_exception": by_exception,
        },
        "passed_uuids": [v.uuid for v in passed],
        "failed": [
            {
                "uuid": v.uuid,
                "json_path": str(v.json_path),
                "stage": v.stage,
                "exception_type": v.exception_type,
                "exception_message": v.exception_message,
                "traceback_tail": v.traceback_tail,
                "n_items_fetched": v.n_items_fetched,
                "elapsed_s": v.elapsed_s,
            }
            for v in failed
        ],
    }


def _print_progress(idx: int, total: int, verdict: ClipVerdict) -> None:
    status = "ok" if verdict.passed else f"FAIL@{verdict.stage}"
    print(
        f"[{idx:5d}/{total:5d}] {verdict.uuid}  {status:18s}  "
        f"{verdict.elapsed_s:6.2f}s",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-load smoke pass over an ncorev4 clips/ tree. "
            "Builds NCoreInstantNuRecDataset for each clip and fetches one "
            "sample. No model load, no GPU."
        )
    )
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--camera-id",
        type=str,
        default=DEFAULT_CAMERA_ID,
        help=f"Single context+supervision camera id (default: {DEFAULT_CAMERA_ID}).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help="frame_batch_sampler.n_samples_per_sequence (matches CLI --max-chunks).",
    )
    parser.add_argument(
        "--items-per-clip",
        type=int,
        default=1,
        help=(
            "Number of __getitem__ indices to exercise per clip. "
            "Default 1; raise to exercise more chunks at the cost of "
            "longer wall time."
        ),
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="Rectification target width (matches JIT contract).",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help="Rectification target height (matches JIT contract).",
    )
    parser.add_argument(
        "--n-frames-per-sample",
        type=int,
        default=DEFAULT_N_FRAMES_PER_SAMPLE,
        help="Frame batcher window size (matches JIT contract).",
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
        help="Standalone logger level. WARNING by default to keep progress readable.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.clips_dir.is_dir():
        print(f"error: --clips-dir '{args.clips_dir}' is not a directory", file=sys.stderr)
        return 2

    jsons = list(iter_clip_jsons(args.clips_dir))
    if not jsons:
        print(f"error: no pai_*.json files under {args.clips_dir}", file=sys.stderr)
        return 1
    if args.max_clips is not None:
        jsons = jsons[: args.max_clips]

    dataset_factory = _default_dataset_factory()
    config_factory = _default_config_factory()

    verdicts: list[ClipVerdict] = []
    t_start = time.monotonic()
    for i, json_path in enumerate(jsons, 1):
        v = smoke_test_one_clip(
            json_path,
            camera_id=args.camera_id,
            max_chunks=args.max_chunks,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            n_frames_per_sample=args.n_frames_per_sample,
            items_per_clip=args.items_per_clip,
            dataset_factory=dataset_factory,
            config_factory=config_factory,
        )
        verdicts.append(v)
        _print_progress(i, len(jsons), v)

    report = build_report(
        clips_dir=args.clips_dir,
        verdicts=verdicts,
        camera_id=args.camera_id,
        max_chunks=args.max_chunks,
        items_per_clip=args.items_per_clip,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    elapsed = time.monotonic() - t_start
    print(
        f"\n=== {s['n_passed']}/{s['n_total']} passed  "
        f"({s['n_failed']} failed)  total {elapsed:.1f}s"
    )
    if s["failures_by_stage"]:
        print(f"  by stage: {s['failures_by_stage']}")
    if s["failures_by_exception"]:
        print(f"  by exception: {s['failures_by_exception']}")

    return 0 if s["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
