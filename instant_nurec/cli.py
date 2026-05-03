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

"""Standalone argparse CLI for the NRM Kelvin predict pipeline.

Phase 1 step 3 introduced the user-facing flag surface; step 4.4 dropped the
Hydra/OmegaConf composition path entirely — ``instant_nurec.config`` now
holds the static config inline as a Python dict and validates against
:class:`NRMConfig` at load time. No yaml/hydra/omegaconf imports.

Phase B dropped bazel; the canonical invocation is now
``python run_inference.py --ncore-path <path> --output-dir <path>
--merge {none,frustum-ownership}``.
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
        description="Standalone NRM Kelvin predict-mode CLI.",
    )
    parser.add_argument(
        "--ncore-path",
        type=Path,
        required=True,
        help="ncorev4 dataset root (must contain debug.lst).",
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
    from instant_nurec.config import load_predict_config
    from instant_nurec.predict.run import run_predict

    config = load_predict_config(
        ncore_path=args.ncore_path,
        output_dir=args.output_dir,
        merge_enabled=(args.merge == "frustum-ownership"),
    )
    run_predict(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
