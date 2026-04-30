# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import argparse
import math

from pathlib import Path

import torch

from nre.models.gaussians.utils import PLYGaussianLoader, write_ply_3dgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate Gaussian PLYs and write new files without overwriting the originals."
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="One or more PLY files to rotate.",
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="x",
        help="Rotation axis. Default is x, which is the usual correction for upside-down assets.",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=180.0,
        help="Rotation angle in degrees. Default is 180.",
    )
    parser.add_argument(
        "--suffix",
        default="upright",
        help="Suffix appended to each output filename stem. Default: upright.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit output file path. Only valid when rotating a single input PLY.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for load/transform/write. Default: cpu.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def get_input_paths(input_paths: list[Path]) -> list[Path]:
    return [path.resolve() for path in input_paths]


def build_rotation_matrix(axis: str, degrees: float, device: str) -> torch.Tensor:
    radians = math.radians(degrees)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    transform = torch.eye(4, dtype=torch.float32, device=device)

    # The samples are currently upside down in the edit-assets path, so default to a 180 deg X rotation.
    if axis == "x":
        transform[:3, :3] = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos_theta, -sin_theta], [0.0, sin_theta, cos_theta]],
            dtype=torch.float32,
            device=device,
        )
    elif axis == "y":
        transform[:3, :3] = torch.tensor(
            [[cos_theta, 0.0, sin_theta], [0.0, 1.0, 0.0], [-sin_theta, 0.0, cos_theta]],
            dtype=torch.float32,
            device=device,
        )
    else:
        transform[:3, :3] = torch.tensor(
            [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

    return transform


def build_output_path(input_path: Path, output_file: Path | None, suffix: str) -> Path:
    if output_file is not None:
        return output_file.resolve()
    return input_path.parent / f"{input_path.stem}-{suffix}{input_path.suffix}"


def rotate_ply(input_path: Path, output_path: Path, transform: torch.Tensor, device: str, force: bool) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    loader = PLYGaussianLoader(input_path, device=device)
    loader.transform(transform)

    custom_attributes: dict[str, torch.Tensor] = {}
    if loader.road_mask is not None:
        custom_attributes["road_mask"] = loader.road_mask
    if loader.sky_mask is not None:
        custom_attributes["sky_mask"] = loader.sky_mask

    write_ply_3dgs(
        path=output_path,
        positions=loader.positions,
        rotations=loader.rotations,
        scales=loader.scales,
        densities=loader.densities,
        features_albedo=loader.features_albedo,
        features_specular=loader.features_specular,
        custom_attributes=custom_attributes,
    )


def main() -> None:
    args = parse_args()
    input_paths = get_input_paths(args.input_paths)
    if args.output_file is not None and len(input_paths) != 1:
        raise ValueError("--output-file can only be used with exactly one input PLY")

    transform = build_rotation_matrix(args.axis, args.degrees, args.device)

    for input_path in input_paths:
        output_path = build_output_path(input_path, args.output_file, args.suffix)
        rotate_ply(input_path, output_path, transform, args.device, args.force)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
