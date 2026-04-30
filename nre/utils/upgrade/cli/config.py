# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from omegaconf import OmegaConf

from nre.artifact.artifact import Artifact
from nre.config.parse import parse_untyped_config
from nre.config.version import Version
from nre.utils.upgrade import upgrade_config as do_upgrade_config


@click.command("upgrade-config")
@click.option(
    "-i",
    "--input",
    "artifact_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the artifact file.",
)
@click.option("-c", "--config-name", type=str, help="Path to the config file.")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Output file path for the upgraded config.",
)
@click.option(
    "-t",
    "--target-version",
    type=str,
    default=None,
    help="Version to upgrade to in major.minor[.patch] format. Defaults to the current software version.",
)
@click.option(
    # It is important to sort the keys when the goal is to compare two configs.
    "--sort-keys/--no-sort-keys",
    type=bool,
    help="Sort the keys before saving.",
    default=True,
)
@click.argument("hydra_args", nargs=-1)
def upgrade_config(
    artifact_path: Optional[Path],
    config_name: Optional[str],
    output: Path,
    target_version: Optional[str],
    sort_keys: bool,
    hydra_args: list[str],
) -> None:
    """
    Upgrade a YAML config file or artifact to a specified version.

    This command can process either a USDZ artifact file or a YAML config file,
    upgrade the configuration to the specified target version (or current version),
    and save the result to an output file.

    \b
    Examples:
    \b
      # From artifact to current version
      upgrade-config --input artifact.usdz --output upgraded.yaml
    \b
      # From YAML to specific version
      upgrade-config --config-name config.yaml --target-version 1.2.3 --output out.yaml
    \b
      # Disable key sorting
      upgrade-config --config-name config.yaml --output out.yaml --no-sort-keys
    \b
      # With Hydra arguments
      upgrade-config --config-name config.yaml --output out.yaml -- dataset.batch_size=32
    """
    if bool(artifact_path) == bool(config_name):
        raise click.UsageError(
            "Please provide either --input (for an usdz artifact) or --config-name (for yaml), but not both."
        )

    if artifact_path:
        if hydra_args:
            click.echo("Warning: Hydra arguments are ignored when processing an artifact file.", err=True)
        artifact = Artifact(artifact_path)
        config = OmegaConf.create(artifact.parsed_config)
    else:
        assert config_name is not None
        config = parse_untyped_config(config_name, hydra_args)

    # Parse target version if provided
    version_target = None
    if target_version:
        version_parts = target_version.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])
        if len(version_parts) > 2:
            patch = int(version_parts[2])
        else:
            patch = 99999
        version_target = Version.from_components(major, minor, patch)

    # Upgrade config
    click.echo("Upgrading configuration...")
    upgraded_config = do_upgrade_config(config, version_target)

    # Save upgraded config
    yaml_config = OmegaConf.to_yaml(upgraded_config, sort_keys=sort_keys)
    with open(output, "w") as f:
        f.write(yaml_config)

    click.echo(f"Upgraded config written to {output}")
