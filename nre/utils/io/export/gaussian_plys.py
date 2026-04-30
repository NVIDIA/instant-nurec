# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path
from typing import Optional

import click
import torch

import nre.systems

from nre.config.parse import parse_typed_config
from nre.models.gaussians.gaussians_model import GaussianExportFormat
from nre.systems.gaussians import GaussiansSystem


@click.command("export-gaussian-plys")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.option(
    "--checkpoint-name",
    type=str,
    help="Checkpoint file name to load",
    default="last.ckpt",
    required=False,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory",
    required=False,
)
@click.option(
    "--format",
    type=click.Choice([format for format in GaussianExportFormat]),
    help="""Format of the mesh files.
        3dgs: Exports plys in the original 3DGS format.
            The format should be compatible with the original 3DGS implementation but differences
            between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
            3rd-party 3DGS viewers.
        3dgrt: Exports plys in the 3DGRT format.
            This format is mostly for visualization purposes, as Gaussians are visualized as octahedrons,
            shifted, rotated, and scaled according to the Gaussian's parameters.""",
    required=False,
    default=GaussianExportFormat._3DGS,
)
@click.option(
    "--percentage-gaussians",
    type=float,
    help="Percentage of gaussians to export. Only used for 3DGRT format. Range (0, 100].",
    default=100,
)
@click.argument("hydra-args", nargs=-1)
@torch.inference_mode()
def export_gaussian_plys(
    config_name: str,
    checkpoint_name: str,
    output_dir: Optional[str],
    hydra_args: list[str],
    format: GaussianExportFormat,
    percentage_gaussians: float,
):
    config = parse_typed_config(config_name=config_name, hydra_args=hydra_args)

    # Load last checkpoint.
    checkpoint_path: Path = Path(config.ckpt_dir) / checkpoint_name
    config.mode = "val"
    if config.resume is None:
        config.resume = str(checkpoint_path)
    system = nre.systems.make(config.system.name, config, load_from_checkpoint=str(checkpoint_path))

    assert isinstance(system, GaussiansSystem), "Only GaussiansSystem supported"

    # If no separate out directory is specified, re-use the checkpoints directory.
    if output_dir is None:
        output_path = Path(config.ckpt_dir).parent / "plys"
    else:
        output_path = Path(output_dir)

    system.model.export_plys(output_path, format=format, percentage_gaussians=percentage_gaussians)
