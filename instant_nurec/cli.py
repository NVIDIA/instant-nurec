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
        help="Directory for PLY output and parsed config.",
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
