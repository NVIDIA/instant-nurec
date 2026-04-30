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
import logging
import pprint

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import click

from omegaconf import DictConfig, OmegaConf

from nre.config.parse import parse_untyped_config


log = logging.getLogger(__file__)


def compute_diff(
    config_one: DictConfig, config_two: DictConfig, config_one_name: str, config_two_name: str, include_names: bool
) -> Dict[str, Any]:
    config_one_dict = OmegaConf.to_container(config_one)
    config_two_dict = OmegaConf.to_container(config_two)
    assert isinstance(config_one_dict, dict)
    assert isinstance(config_two_dict, dict)

    _ONLY_IN_CONFIG_ONE = f"only_in_{config_one_name if include_names else 'config_one'}"
    _ONLY_IN_CONFIG_TWO = f"only_in_{config_two_name if include_names else 'config_two'}"
    _ITERABLES_IN_CONFIG_ONE = f"iterables_only_in_{config_one_name if include_names else 'config_one'}"
    _ITERABLES_IN_CONFIG_TWO = f"iterables_only_in_{config_two_name if include_names else 'config_two'}"
    _TYPE_DIFFERENCES = "type_differences"
    _VALUE_DIFFERENCES = "value_differences"
    _CONFIG_ONE_REF = config_one_name.split("/")[-1] if include_names else "config_one"
    _CONFIG_TWO_REF = config_two_name.split("/")[-1] if include_names else "config_two"

    def dict_diff(dict_one: Dict[Any, Any], dict_two: Dict[Any, Any], path: str = "config", layer=0) -> Dict[str, Any]:
        """Recursively finds differences between two dictionaries"""
        diff: Dict[str, Any] = {
            _ONLY_IN_CONFIG_ONE: {},
            _ONLY_IN_CONFIG_TWO: {},
            _ITERABLES_IN_CONFIG_ONE: {},
            _ITERABLES_IN_CONFIG_TWO: {},
            _TYPE_DIFFERENCES: {},
            _VALUE_DIFFERENCES: {},
        }

        all_keys = set(dict_one.keys()).union(set(dict_two.keys()))

        for key in all_keys:
            key_path = f"{path}.{key}"

            # Find items not in the other dictionary
            if key not in dict_one:
                diff[_ONLY_IN_CONFIG_TWO][key_path] = dict_two[key]
            elif key not in dict_two:
                diff[_ONLY_IN_CONFIG_ONE][key_path] = dict_one[key]

            # In both dictionaries
            else:
                value_one, value_two = dict_one[key], dict_two[key]

                # If both are dictionaries, recursively compute
                if isinstance(value_one, Mapping) and isinstance(value_two, Mapping) and layer < 5:
                    nested_diff = dict_diff(dict(value_one), dict(value_two), key_path, layer + 1)
                    for section in diff:
                        for sub_key, sub_value in nested_diff[section].items():
                            diff[section][sub_key] = sub_value
                # If both are iterables and one has missing values
                elif (
                    isinstance(value_one, Iterable)
                    and not isinstance(value_one, str)
                    and isinstance(value_two, Iterable)
                    and not isinstance(value_two, str)
                ):
                    value_one, value_two = list(value_one), list(value_two)
                    for i, item_one in enumerate(value_one):
                        if i >= len(value_two) or item_one != value_two[i]:
                            diff[_ITERABLES_IN_CONFIG_ONE][f"{key_path}[{i}]"] = item_one
                    for i, item_two in enumerate(value_two):
                        if i >= len(value_one) or item_two != value_one[i]:
                            diff[_ITERABLES_IN_CONFIG_TWO][f"{key_path}[{i}]"] = item_two

                else:
                    if type(value_one) is not type(value_two):
                        diff[_TYPE_DIFFERENCES][key_path] = {
                            _CONFIG_ONE_REF: type(value_one).__name__,
                            _CONFIG_TWO_REF: type(value_two).__name__,
                        }
                    elif value_one != value_two:
                        diff[_VALUE_DIFFERENCES][key_path] = {_CONFIG_ONE_REF: value_one, _CONFIG_TWO_REF: value_two}

        return diff

    return dict_diff(config_one_dict, config_two_dict)


def output_diff(config_diff: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "config_diff.json"

    with open(str(filepath), "w") as f:
        json.dump(config_diff, f, indent=4, sort_keys=True, ensure_ascii=True)


@click.command()
@click.option(
    "--config-one-name",
    type=str,
    help="First Hydra config to compare",
    required=True,
)
@click.option(
    "--config-two-name",
    type=str,
    help="Second Hydra config to compare",
    required=True,
)
@click.option(
    "-d",
    "--dump",
    is_flag=True,
    help="Flag for dumping diff to output file",
    default=False,
)
@click.option(
    "-i",
    "--include-config-names",
    is_flag=True,
    help="Explicitly name configs based on file name rather than [config_one] and [config_two]",
    default=False,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output directory where the config diff will be dumped",
    default=None,
    required=False,
)
@click.argument("hydra-args", nargs=-1)
def export_config_diff(
    config_one_name: str,
    config_two_name: str,
    dump: bool,
    include_config_names: bool,
    output_dir: Optional[str],
    hydra_args: list[str],
) -> None:
    """Dumps the diff between [config_one] and [config_two] with hydra args"""
    # Load config files and cast them to NREConfig
    config_one = parse_untyped_config(config_one_name, hydra_args)
    config_two = parse_untyped_config(config_two_name, hydra_args)

    # Use DeepDiff to compute difference
    config_diff = compute_diff(config_one, config_two, config_one_name, config_two_name, include_config_names)

    # Make output readable and pretty
    log.info(f"Difference between config {config_one_name} and {config_two_name}:")
    log.info(pprint.pformat(config_diff, indent=4, width=50))

    # Dump file if specified
    if dump:
        if output_dir is None:
            raise ValueError("[--dump] flag was set to True but no [--output-dir] was provided.")

        output_diff(config_diff, Path(output_dir))
        log.info(f"Dumped diff between {config_one_name} and {config_two_name} to {output_dir}/config_diff.json")


if __name__ == "__main__":
    export_config_diff(show_default=True)
