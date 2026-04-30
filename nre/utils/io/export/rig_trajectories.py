# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from pathlib import Path

import click

from nre.config.parse import parse_untyped_config
from nre.datasets import make as make_dataset
from nre.datasets.base import RigTrajectoriesProvider
from nre.utils.io.rig_trajectories import rig_trajectories_time_range, serialize_rig_trajectories


@click.command("export-rig-trajectories")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    default="tests/ncore_ds",
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory",
    required=True,
)
@click.argument("hydra-args", nargs=-1)
def export_rig_trajectories(
    config_name: str,
    output_dir: str,
    hydra_args: list[str],
) -> None:
    """Extracts and exports the rig trajetories for a given dataset"""
    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    config = parse_untyped_config(config_name=config_name, hydra_args=hydra_args)

    datasource = make_dataset(config.dataset.name, config, split="train").get_datasource()
    if not isinstance(datasource, RigTrajectoriesProvider):
        raise NotImplementedError("Rig trajectory export requires a rig trajectory provider")

    # Load full rig trajectories (V4 compatibility layer makes this efficient)
    rig_trajectories = datasource.get_rig_trajectories()
    rig_timerange = rig_trajectories_time_range(rig_trajectories)

    data = serialize_rig_trajectories(rig_trajectories, usd_timestamp_offset=rig_timerange.start)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    for item in data:
        item.save(output_dir_path)
