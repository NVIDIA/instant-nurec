#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Simple script to load a msgpack file, update the version, and save it back.
"""

import argparse
import gzip
import io
import logging
import sys

from pathlib import Path

import msgpack

from nre.config.version import get_version


def load_msgpack_file(file_path: Path):
    """Load a msgpack file (handles both gzipped and non-gzipped)."""
    with open(file_path, "rb") as f:
        # Check if the file is gzipped
        magic = f.read(2)
        f.seek(0)

        if magic.startswith(b"\x1f\x8b"):
            # File is gzipped
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                packed_data = gz.read()
        else:
            # File is not gzipped
            packed_data = f.read()

    return msgpack.unpackb(packed_data, raw=False)


def save_msgpack_file(data, file_path: Path, compress: bool = True):
    """Save data as msgpack file (optionally gzipped)."""
    packed = msgpack.packb(data)

    if compress:
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
            f.write(packed)
        with open(file_path, "wb") as f:
            f.write(buffer.getvalue())
    else:
        with open(file_path, "wb") as f:
            f.write(packed)


# patch gaussian_primitives default node name
def patch_gaussian_primitives_node(version: str, data: dict):
    """Patch legacy model data to be compatible with current version."""

    if "nre_data" not in data:
        return

    data_version = data["nre_data"]["version"]
    try:
        data_major, _, _ = map(int, data_version.split("."))
    except ValueError:
        logging.warning(f"Could not parse data version string: {data_version}")
        return

    # check data version : data_version major >= 25
    if data_major < 25:
        return

    # Parse version string into components
    try:
        major, minor, patch = map(int, version.split("."))
    except ValueError:
        logging.warning(f"Could not parse version string: {version}")
        return

    # Check if target version is <= 0.2.715
    affected_version = (major < 0 and minor < 2) or (major == 0 and minor == 2 and patch <= 715)
    if not affected_version:
        return

    if "model" not in data["nre_data"] or data["nre_data"]["model"] != "nre":
        return

    config = data["nre_data"]["config"]
    if "name" not in config or config["name"] != "gaussians_primitive":
        return

    if "layers" not in config or "gaussians" in config["layers"]:
        return

    # Rename background layer to gaussians
    config["layers"]["gaussians"] = config["layers"].pop("background")

    # Rename all state_dict keys from .gaussians_nodes.background to .gaussians_nodes.gaussians
    if "state_dict" not in data["nre_data"]:
        return

    state_dict = data["nre_data"]["state_dict"]
    # Get list of keys to rename
    old_keys = [k for k in state_dict.keys() if ".gaussians_nodes.background" in k]

    # Create mapping of old keys to new keys
    key_mapping = {
        old_key: old_key.replace(".gaussians_nodes.background", ".gaussians_nodes.gaussians") for old_key in old_keys
    }

    # Rename keys
    for old_key, new_key in key_mapping.items():
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)

    logging.info(f"Patched gaussians_primitive default layer {data_version} to {version}")


def main():
    parser = argparse.ArgumentParser(description="Update version in msgpack file")
    parser.add_argument("input_file", type=Path, help="Input msgpack file")
    parser.add_argument("output_file", type=Path, help="Output msgpack file")
    parser.add_argument("--version", type=str, help="Explicit version to set (e.g., '1.2.3')")
    parser.add_argument("--no-compress", action="store_true", help="Don't compress output")

    args = parser.parse_args()

    # Load the msgpack file
    print(f"Loading {args.input_file}...")
    data = load_msgpack_file(args.input_file)

    # Determine version to use
    if args.version:
        target_version = args.version
        print(f"Using specified version: {target_version}")
    else:
        current_version = get_version()
        if current_version is None:
            print("Error: Could not determine current version and no version specified")
            sys.exit(1)
        target_version = current_version.semantic_string()
        print(f"Using current version: {target_version}")

    # patch gaussian_primitives default node name
    patch_gaussian_primitives_node(target_version, data)

    # Update version
    if "nre_data" in data and "version" in data["nre_data"]:
        old_version = data["nre_data"]["version"]
        data["nre_data"]["version"] = target_version
        print(f"Updated version: {old_version} -> {target_version}")
    else:
        print("Warning: No version found in nre_data")

    # Save the updated file
    print(f"Saving to {args.output_file}...")
    save_msgpack_file(data, args.output_file, compress=not args.no_compress)
    print("Done!")


if __name__ == "__main__":
    main()
