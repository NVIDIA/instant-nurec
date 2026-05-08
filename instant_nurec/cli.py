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

"""Standalone argparse CLI.

``--ncore-path`` accepts either a single ``.json`` ncorev4 sequence
metadata file (NuRec-aligned) or a ``.lst`` manifest listing one JSON
path per line (each absolute or relative-to-the-LST-file's directory).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence


_DEFAULT_CAMERA_ID = "camera_front_wide_120fov"
_DEFAULT_LIDAR_ID = "lidar_top_360fov"
_DEFAULT_MAX_CHUNKS = 8


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instant_nurec",
        description="Standalone Kelvin predict-mode CLI.",
    )
    parser.add_argument(
        "--ncore-path",
        type=Path,
        required=True,
        help=(
            "ncorev4 input. Either a single sequence ``.json`` (NuRec-aligned) "
            "or a ``.lst`` manifest with one JSON path per line "
            "(absolute or relative-to-the-LST-file's directory; "
            "``#``-prefixed and blank lines skipped)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PLY output.",
    )
    parser.add_argument(
        "--merge",
        choices=["none", "frustum-ownership"],
        default="none",
        help=(
            "Primitive merge strategy. 'none' writes per-chunk PLYs; "
            "'frustum-ownership' writes a single merged PLY."
        ),
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=_DEFAULT_CAMERA_ID,
        help=(
            f"ncorev4 context-camera id used as model input "
            f"(default: '{_DEFAULT_CAMERA_ID}'). Wires both "
            f"context_camera_ids and supervision_camera_ids to [CAMERA_ID]. "
            f"Exactly one camera is required."
        ),
    )
    parser.add_argument(
        "--lidar-id",
        type=str,
        default=_DEFAULT_LIDAR_ID,
        help=(
            f"ncorev4 LiDAR sensor id used to source cuboid tracks for "
            f"dynamic-mask refinement (default: '{_DEFAULT_LIDAR_ID}'). "
            f"Must exist in the sequence's lidar_sensors."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=_DEFAULT_MAX_CHUNKS,
        help=(
            f"Maximum number of time-chunks processed per clip "
            f"(default: {_DEFAULT_MAX_CHUNKS}). One chunk spans up to 13.5 s, "
            f"so the default covers 8 * 13.5 = 108 s. Clips longer than that "
            f"are silently truncated unless this is increased -- bump it to "
            f"ceil(clip_seconds / 13.5) for longer clips."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level forwarded to logging.basicConfig.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # Lazy imports keep argparse-only invocations (e.g. --help) cheap.
    from instant_nurec.config_schema.dataset import (
        AdaptiveSequentialFrameBatchSamplerConfig,
        NCoreInstantNuRecCuboidTracksParamsConfig,
        NCoreInstantNuRecDatasetConfig,
        InstantNuRecSplitsConfig,
    )
    from instant_nurec.config_schema.instantnurec import InstantNuRecConfig
    from instant_nurec.config_schema.predict import PredictConfig, PrimitiveMergeConfig
    from instant_nurec.ncore_input import resolve_ncore_paths
    from instant_nurec.predict.run import run_predict

    json_paths = resolve_ncore_paths(args.ncore_path)

    config = InstantNuRecConfig(
        out_dir=str(args.output_dir),
        dataset=InstantNuRecSplitsConfig(
            predict=NCoreInstantNuRecDatasetConfig(
                ncore_json_paths=[str(p) for p in json_paths],
                context_camera_ids=[args.camera_id],
                supervision_camera_ids=[args.camera_id],
                cuboid_tracks_params=NCoreInstantNuRecCuboidTracksParamsConfig(
                    lidar_id=args.lidar_id,
                ),
                frame_batch_sampler=AdaptiveSequentialFrameBatchSamplerConfig(
                    n_samples_per_sequence=args.max_chunks,
                ),
            ),
        ),
        predict=PredictConfig(
            primitive_merge=PrimitiveMergeConfig(
                enabled=(args.merge == "frustum-ownership"),
            ),
        ),
    )
    run_predict(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
