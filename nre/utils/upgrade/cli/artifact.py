# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import shutil
import zipfile

from pathlib import Path
from typing import Optional

import click
import torch
import yaml

from omegaconf import OmegaConf

from nre.artifact import Artifact
from nre.config.version import Version, get_version
from nre.datasets.summary import DataSourceSummary
from nre.utils.upgrade import upgrade_config, upgrade_model


@click.command("upgrade-artifact")
@click.option(
    "-i",
    "--input",
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the input USDZ artifact file.",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to the output USDZ artifact file.",
)
@click.option(
    "-t",
    "--target-version",
    type=str,
    default=None,
    help="Version to upgrade to in major.minor[.patch] format. Defaults to the current software version.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging.",
)
def upgrade_artifact(
    input_file: Path,
    output_file: Path,
    target_version: Optional[str],
    debug: bool,
) -> None:
    """
    Upgrade a USDZ artifact file to a specified version.

    This command takes a USDZ artifact file, upgrades both its configuration
    and model checkpoint to the specified target version (or current version),
    and saves the result to a new USDZ file.

    \b
    Examples:
    \b
      # Upgrade to current version
      upgrade-artifact --input artifact.usdz --output upgraded.usdz
    \b
      # Upgrade to specific version
      upgrade-artifact --input artifact.usdz --output upgraded.usdz --target-version 1.2.3
    \b
      # With debug logging
      upgrade-artifact --input artifact.usdz --output upgraded.usdz --debug
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    if output_file.exists():
        raise click.ClickException(f"Output file {output_file} already exists.")

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
    else:
        version_target = get_version()
        if version_target is None:
            # In certain environments (e.g. CI test sandboxes), version information might not be
            # available. In such cases, the target is the latest compatible version that we can upgrade to.
            logging.warning("Current version (upgrade target) not available, assuming latest version.")

    click.echo(f"Loading artifact from {input_file}...")
    artifact = Artifact(input_file)

    # Upgrade config
    click.echo("Upgrading configuration...")
    orig_config = OmegaConf.create(artifact.parsed_config)
    upgraded_config = upgrade_config(orig_config, version_target)

    # Upgrade model
    click.echo("Upgrading model checkpoint...")
    checkpoint = torch.load(artifact.checkpoint, weights_only=False, map_location="cpu")
    model_state_dict = {
        k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
    }

    # update the temporal appearance V1 to V2
    datasource_summary = DataSourceSummary.from_dict(
        artifact.datasource_summary,
        infer_missing=True,  # For better backward-compatibility.
    )

    # Pass datasource_summary for temporal appearance upgrade
    upgraded_model_state_dict = upgrade_model(
        model_state_dict, orig_config, version_target, datasource_summary=datasource_summary
    )

    # Update checkpoint with upgraded model
    # Important: rebuild the entire `model.*` subtree to drop any obsolete keys from the original checkpoint
    original_state_dict = checkpoint["state_dict"]
    # Keep all non-model keys
    new_state_dict = {k: v for k, v in original_state_dict.items() if not k.startswith("model.")}
    # Add upgraded model keys only
    new_state_dict.update({f"model.{k}": v for k, v in upgraded_model_state_dict.items()})
    checkpoint["state_dict"] = new_state_dict

    # Write new usdz file
    click.echo(f"Writing upgraded artifact to {output_file}...")
    with zipfile.ZipFile(input_file, "r") as zin:
        with zipfile.ZipFile(output_file, "w") as zout:
            for item in zin.infolist():
                if item.filename == "parsed_config.yaml":
                    zout.writestr(item, yaml.dump(OmegaConf.to_container(upgraded_config, resolve=True)))
                elif item.filename == "checkpoint.ckpt":
                    with zout.open(item, "w") as checkpoint_file:
                        torch.save(checkpoint, checkpoint_file)
                else:
                    with zin.open(item) as src, zout.open(item, "w") as dst:
                        shutil.copyfileobj(src, dst)

    click.echo(f"Upgraded artifact written to {output_file}")
    click.echo("Upgrade complete.")
