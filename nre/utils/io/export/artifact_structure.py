# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json

from pathlib import Path
from typing import Any, MutableMapping, Optional

import click
import torch

from nre.artifact.artifact import Artifact


def analyze_state_dict(state_dict: MutableMapping[str, Any]) -> dict[str, Any]:
    """
    Analyzes a model's state_dict, filtering for tensors, and returns a
    hierarchical JSON-like structure with metadata for each leaf.
    """
    nested_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        parts = key.split(".")
        d = nested_dict
        for part in parts[:-1]:
            d = d.setdefault(part, {})

        leaf_info = {"shape": str(list(value.shape)), "dtype": str(value.dtype), "device": str(value.device)}

        d[parts[-1]] = leaf_info
    return nested_dict


@click.command("export-artifact-structure")
@click.option(
    "-i",
    "--input",
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the .usdz artifact or PyTorch checkpoint file.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output JSON file path. If not provided, prints to stdout.",
)
def export_artifact_structure(input_file: Path, output: Optional[Path]) -> None:
    """
    Exports the structure of a model's state_dict from a .usdz artifact
    or a PyTorch checkpoint file (.ckpt) as a JSON object.

    This command inspects the model checkpoint inside a USDZ artifact or a raw
    PyTorch checkpoint file and outputs the structure of all tensors as a
    hierarchical JSON object. For each tensor, the JSON output includes its
    shape and data type (dtype).

    This is useful for creating model upgrade functions, as it allows you to
    compare the checkpoint structure between two different NRE versions and
    identify what has changed (e.g., renamed layers, modified tensor shapes).

    \b
    Examples:
    \b
      # Export from artifact to stdout
      export-artifact-structure --input artifact.usdz
    \b
      # Export from checkpoint to file
      export-artifact-structure --input checkpoint.ckpt --output structure.json
    \b
      # Save structure for comparison
      export-artifact-structure --input my_artifact.usdz --output structure.json
    """

    if input_file.suffix == ".usdz":
        artifact = Artifact(source=input_file)
        checkpoint = torch.load(artifact.checkpoint, map_location=torch.device("cpu"), weights_only=False)
    elif input_file.suffix == ".ckpt":
        checkpoint = torch.load(input_file, map_location=torch.device("cpu"), weights_only=False)
    else:
        raise click.BadParameter(f"Input file must have a .usdz or .ckpt extension, but got {input_file.suffix!r}")

    # The actual model parameters are in the 'state_dict' key
    state_dict = checkpoint.get("state_dict", {})

    structure = analyze_state_dict(state_dict)

    json_output = json.dumps(structure, indent=4, sort_keys=True)

    if output:
        with open(output, "w") as f:
            f.write(json_output)
        click.echo(f"Model structure written to {output}")
    else:
        click.echo(json_output)
