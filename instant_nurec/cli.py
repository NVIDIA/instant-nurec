"""Standalone argparse CLI for the NRM Kelvin predict pipeline.

Phase 1 step 3: introduces the user-facing flag surface (`--ncore-path`,
`--output-dir`, `--merge {none,frustum-ownership}`, `--log-level`) on top of
the verbatim NRE copy. Until Phase 3 swaps the build system, this module is
launched via `bazel run //instant_nurec:run -- ...` so that bazel-compiled
slang/CUDA artifacts stay deterministic across the strip iterations.

The CLI translates its flags into the Hydra overrides that NRE's
`nre.nrm.run.main` click command expects, then invokes that command's underlying
callback directly. Subsequent Phase 1 strips (config inlining, lightning
removal, NRE rename) progressively replace the delegation target without
disturbing the user-facing flag surface.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence


CONFIG_NAME: str = "configs/nrm/apps/pretrained/ngc_kelvin_pa_front.yaml"

# Overrides that are constant across both --merge modes. They mirror the
# defaults baked into nre_example_call.sh so that this CLI produces parity-
# matching output against baselines/original_baseline.
PRESET_OVERRIDES: tuple[str, ...] = (
    "+nrm/apps/options=_kelvin_predict",
    "dataset.predict.cuboid_tracks_params.lidar_id=lidar_top_360fov",
    "predict.render_video.enabled=false",
)


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


def hydra_overrides(args: argparse.Namespace) -> list[str]:
    """Translate parsed CLI args into Hydra-style override strings."""
    overrides: list[str] = list(PRESET_OVERRIDES)
    overrides.append(f"dataset.predict.ncore_json_base_path={args.ncore_path}")
    overrides.append(f"dataset.predict.ncore_json_list_path={args.ncore_path}/debug.lst")
    overrides.append(f"out_dir={args.output_dir}")
    if args.merge == "none":
        overrides.append("predict.primitive_merge.enabled=false")
    else:  # frustum-ownership
        overrides.append("predict.primitive_merge.enabled=true")
        overrides.append("predict.primitive_merge.overlap_strategy=frustum_ownership")
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    overrides = hydra_overrides(args)

    # Lazy import keeps the CLI surface unit-testable without NRE deps.
    from nre.nrm.run import main as nre_main

    nre_main.callback(config_name=CONFIG_NAME, hydra_args=tuple(overrides))
    return 0


if __name__ == "__main__":
    sys.exit(main())
