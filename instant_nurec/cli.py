"""Standalone argparse CLI for the NRM Kelvin predict pipeline.

Phase 1 step 3 introduced the user-facing flag surface; step 4.4 swaps the
Hydra/OmegaConf composition path for a yaml.safe_load + pydantic validation
loop (``instant_nurec.config.load_predict_config``). The CLI now constructs
the typed :class:`NRMConfig` directly and hands it to the predict driver,
which keeps `nre/` purely as a runtime library (no hydra imports needed).

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
    from nre.nrm.run import run_predict

    config = load_predict_config(
        ncore_path=args.ncore_path,
        output_dir=args.output_dir,
        merge_enabled=(args.merge == "frustum-ownership"),
    )
    run_predict(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
