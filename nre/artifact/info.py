# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import sys

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from nre.artifact.artifact import Artifact


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> OrderedDict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return OrderedDict(items)


def get_flattened_metadata_with_file_path(artifact: Artifact):
    metadata = flatten_dict(artifact.metadata)
    return OrderedDict([("file_path", artifact.source)] + list(metadata.items()))


def get_max_width(items: List[str]) -> int:
    return max(len(str(item)) for item in items)


def print_single(metadata: OrderedDict[str, Any], no_header: bool = False, keys: List[str] = None):
    if keys:
        metadata = OrderedDict((k, metadata[k]) for k in keys if k in metadata)

    if no_header:
        for value in metadata.values():
            print(value)
    else:
        max_key_length = max(len(key) for key in metadata.keys())
        for key, value in metadata.items():
            print(f"{key:<{max_key_length}} : {value}")


def print_tabular(artifacts: List[Artifact], keys: List[str] = None):
    # Concatenate artifact file path with metadata to disambiguate potentially identical artifacts
    all_metadata = []
    for artifact in artifacts:
        all_metadata.append(get_flattened_metadata_with_file_path(artifact))

    if not keys:
        keys = list(all_metadata[0].keys())

    col_widths = [get_max_width([k] + [str(m.get(k, "")) for m in all_metadata]) for k in keys]

    # Print header
    header_format = " | ".join(f"{{:<{w}}}" for w in col_widths)
    print(header_format.format(*keys))
    print("-" * (sum(col_widths) + 3 * (len(keys) - 1)))

    # Print rows
    for metadata in all_metadata:
        row = [str(metadata.get(k, "")) for k in keys]
        print(header_format.format(*row))


def process_artifacts(glob_query: str, keys: List[str] = None, no_header: bool = False):
    artifacts = Artifact.discover_from_glob(glob_query)

    if not artifacts:
        print(f"No artifact(s) found for {glob_query}")
        return

    if len(artifacts) == 1:
        print_single(get_flattened_metadata_with_file_path(artifacts[0]), no_header, keys)
    else:
        print_tabular(artifacts, keys)


def main():
    parser = argparse.ArgumentParser(description="Artifact info tool")
    parser.add_argument(
        "path", help="Path to the artifact or directory (*.usdz will be appended if path is a directory)"
    )
    parser.add_argument("--keys", nargs="+", help="Specific metadata keys to display")
    parser.add_argument("--no-header", action="store_true", help="Print only values without keys")

    args = parser.parse_args()

    path = Path(args.path)
    glob_query = str(path / "*.usdz") if path.is_dir() else args.path
    process_artifacts(glob_query, args.keys, args.no_header)


if __name__ == "__main__":
    main()
