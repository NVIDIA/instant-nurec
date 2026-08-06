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

from instant_nurec.pretrained import (
    DEFAULT_MODEL_VARIANT,
    MODEL_VARIANTS,
    get_model_profile,
)

_DEFAULT_MAX_CHUNKS = 8
_DEFAULT_N_GAUSSIANS = 2_000_000


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instant_nurec",
        description="Standalone Kelvin predict-mode CLI.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_VARIANTS,
        default=DEFAULT_MODEL_VARIANT,
        help=(
            "Released model profile. pa-front uses one 784x448 camera with 18 frames; "
            "pa-multiview uses three 504x280 cameras with 18 frames each by default; "
            "pq-front uses the selective point-query decoder with 18 front-camera "
            "frames. "
            f"Default: {DEFAULT_MODEL_VARIANT}."
        ),
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
        help="Directory for PLY and sky-output bundles.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "If set, merge per-chunk primitives into a single "
            "frustum-ownership PLY per sequence and run kl-optimal "
            "voxelization (target count controlled by --n-gaussians). "
            "Default (flag absent) writes per-chunk PLYs and skips "
            "voxelization entirely."
        ),
    )
    parser.add_argument(
        "--n-gaussians",
        type=int,
        default=_DEFAULT_N_GAUSSIANS,
        help=(
            f"Target number of static Gaussians after kl-optimal voxelization "
            f"(default: {_DEFAULT_N_GAUSSIANS}). The voxel size is searched "
            f"iteratively via bracketed binary search (starting at 0.1) until "
            f"the count lands in [0.9 * target, target]. Only consulted when "
            f"--merge is set (voxelization is bundled with merge); must be > 0."
        ),
    )
    parser.add_argument(
        "--camera-id",
        dest="camera_ids",
        metavar="CAMERA_ID",
        type=str,
        action="append",
        default=None,
        help=(
            "Override a profile's ncorev4 context camera. Repeat the flag to "
            "provide multiple cameras in canonical order. The selected profile "
            "validates its camera contract. pq-front is fixed to "
            "camera_front_wide_120fov. If omitted, the profile's default camera "
            "set is used."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=_DEFAULT_MAX_CHUNKS,
        help=(
            f"Maximum number of time-chunks processed per clip "
            f"(default: {_DEFAULT_MAX_CHUNKS}). One chunk spans up to 13.5 s. "
            f"Longer clips are "
            f"truncated and a WARNING is logged naming the dropped chunk "
            f"count and the --max-chunks value needed to cover the full clip."
        ),
    )
    parser.add_argument(
        "--render-preview",
        action="store_true",
        help=(
            "Render the first context frame of each output chunk as a PNG, "
            "compositing the observation-derived sky behind the Gaussians. "
            "Install with `uv sync --extra render` first."
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
    parser = make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.render_preview:
        from instant_nurec.predict.render_preview import require_gsplat

        try:
            require_gsplat()
        except RuntimeError as exc:
            parser.error(str(exc))

    # Lazy imports keep argparse-only invocations (e.g. --help) cheap.
    from instant_nurec.config_schema.dataset import (
        AdaptiveSequentialFrameBatchSamplerConfig,
        NCoreInstantNuRecDatasetConfig,
        InstantNuRecSplitsConfig,
    )
    from instant_nurec.config_schema.instantnurec import InstantNuRecConfig
    from instant_nurec.config_schema.models import (
        KelvinDPTDecoderConfig,
        KelvinModelConfig,
        KelvinPointQueryCADecoderConfig,
    )
    from instant_nurec.config_schema.predict import PredictConfig, PrimitiveMergeConfig
    from instant_nurec.ncore_input import resolve_ncore_paths
    from instant_nurec.predict.run import run_predict

    json_paths = resolve_ncore_paths(args.ncore_path)
    profile = get_model_profile(args.model)
    camera_ids = list(args.camera_ids or profile.context_camera_ids)
    allowed_camera_counts = profile.supported_camera_counts
    if len(camera_ids) not in allowed_camera_counts:
        counts = ", ".join(str(count) for count in allowed_camera_counts)
        raise ValueError(
            f"{args.model} expects {counts} context camera(s); got {len(camera_ids)}. "
            "Repeat --camera-id once per camera."
        )

    decoder_config = (
        KelvinPointQueryCADecoderConfig(motion_depth=profile.motion_depth)
        if profile.decoder_kind == "point-query"
        else KelvinDPTDecoderConfig(motion_depth=profile.motion_depth)
    )

    config = InstantNuRecConfig(
        out_dir=str(args.output_dir),
        release_profile=args.model,
        model=KelvinModelConfig(
            decoder=decoder_config,
        ),
        dataset=InstantNuRecSplitsConfig(
            predict=NCoreInstantNuRecDatasetConfig(
                ncore_json_paths=[str(p) for p in json_paths],
                camera_subsampler={
                    "frame_width": profile.frame_width,
                    "frame_height": profile.frame_height,
                },
                context_camera_ids=camera_ids,
                supervision_camera_ids=camera_ids,
                frame_batch_sampler=AdaptiveSequentialFrameBatchSamplerConfig(
                    n_frames_per_sample=profile.n_frames_per_sample,
                    n_samples_per_sequence=args.max_chunks,
                ),
            ),
        ),
        predict=PredictConfig(
            primitive_merge=PrimitiveMergeConfig(
                enabled=args.merge,
                enable_voxelization=args.merge,
                target_n_gaussians=args.n_gaussians,
            ),
            render_preview=args.render_preview,
        ),
    )
    run_predict(config)
    if args.merge:
        print(
            "Next: refine into USDZ with NuRec — "
            "https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html",
            flush=True,
        )
    else:
        print(
            "Next: view your 3DGS PLY with SuperSplat "
            "(https://playcanvas.com/supersplat/editor) "
            "or ply_viewer (NuRec container).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
