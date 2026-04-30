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
from typing import Any, Dict, Optional

import click
import torch

import nre.systems

from nre.config.parse import parse_typed_config
from nre.systems.base import BaseSystemSO
from nre.utils.types import Checkpoint


@click.command("export-usdz-artifact")
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
    required=True,
    default="last.ckpt",
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory",
    required=False,
)
@click.argument("hydra-args", nargs=-1)
def export_usdz_artifact(
    config_name: str,
    checkpoint_name: str,
    output_dir: Optional[str],
    hydra_args: list[str],
):
    """Entry point to export NRE checkpoints to USD"""
    config = parse_typed_config(config_name=config_name, hydra_args=hydra_args)

    # Load last checkpoint.
    checkpoint_path: Path = Path(config.ckpt_dir) / checkpoint_name
    if (resume := config.resume) is None:  # FIXME: this is a hack to avoid reinitializing the model in the system init
        config.resume = str(checkpoint_path)

    system: BaseSystemSO = nre.systems.make(config.system.name, config, load_from_checkpoint=str(checkpoint_path))
    if not isinstance(system, BaseSystemSO):
        raise ValueError("Exporting USDZ artifact is only supported for systems that inherit from BaseSystemSO")
    # FIXME: this is a hack to avoid reinitializing the model in the system init
    system.resume = resume
    config.resume = resume

    # Checkpoint is effectively a dict[str, Any]
    checkpoint: Checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)

    # If no separate out directory is specified, re-use the checkpoints directory.
    if output_dir is None:
        if config.out_dir is None:
            config.out_dir = config.ckpt_dir
    else:
        config.out_dir = output_dir
    output_path = Path(config.out_dir)

    # Populate the artifacts cache. This generates all files in memory according to the config.
    system.populate_artifact_cache(checkpoint)

    # Save.
    system.artifact_cache.write_to_usdz(file_path=output_path / "last.usdz")
