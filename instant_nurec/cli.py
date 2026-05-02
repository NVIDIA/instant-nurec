"""Standalone argparse CLI for the NRM Kelvin predict pipeline.

Phase 1 step 3 introduced the user-facing flag surface; step 4.4 dropped the
Hydra/OmegaConf composition path entirely — ``instant_nurec.config`` now
holds the static config inline as a Python dict and validates against
:class:`NRMConfig` at load time. No yaml/hydra/omegaconf imports.

Until Phase 3 swaps the build system, this module is launched via
``bazel run //instant_nurec:run -- ...`` so that bazel-compiled slang/CUDA
artifacts stay deterministic across the strip iterations.
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
    from instant_nurec.nre.nrm.run import run_predict

    config = load_predict_config(
        ncore_path=args.ncore_path,
        output_dir=args.output_dir,
        merge_enabled=(args.merge == "frustum-ownership"),
    )
    run_predict(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
