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
SBOM Release Script

This script compares two SBOM (Software Bill of Materials) files in CycloneDX format
and generates a CSV file containing information about new packages found in the modified SBOM.

Usage:
    python release.py <base_sbom.json> <modified_sbom.json> <output.csv>

The script extracts package information from PURL (Package URL) format and outputs
the following columns:
- Package / Component Name
- Version
- License (empty for now)
- Link to Component's License (empty for now)
- Method of Distribution (empty for now)
- Usage Method with NV proprietary code (empty for now)
- Comments (empty for now)
- Location where component was downloaded from (empty for now)
- Link to internal IT Controlled Repository (empty for now)
- OSRB Bug ID (empty for now)
"""

import argparse
import csv
import json
import re
import sys

from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse


def extract_package_info_from_purl(purl: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract package name and version from a PURL (Package URL).

    Args:
        purl: Package URL string (e.g., "pkg:pypi/aiofiles@24.1.0")

    Returns:
        Tuple of (package_name, version) or (None, None) if parsing fails
    """
    if not purl:
        return None, None

    try:
        # Parse the PURL format: pkg:type/namespace/name@version?qualifiers#subpath
        # We're interested in the name@version part
        if not purl.startswith("pkg:"):
            return None, None

        # Remove the pkg: prefix and split by /
        parts = purl[4:].split("/")
        if len(parts) < 2:
            return None, None

        # The last part contains name@version
        name_version_part = parts[-1]

        # Split by @ to separate name and version
        if "@" in name_version_part:
            name_part, version_part = name_version_part.split("@", 1)
            # Remove any query parameters or fragments from version
            version = version_part.split("?")[0].split("#")[0]
            return name_part, version
        else:
            # No version specified
            return name_version_part.split("?")[0].split("#")[0], None

    except Exception as e:
        print(f"Error parsing PURL '{purl}': {e}")
        return None, None


def load_sbom(file_path: str) -> Dict:
    """Load and parse an SBOM JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading SBOM file '{file_path}': {e}")
        sys.exit(1)


def extract_components(sbom: Dict) -> Set[Tuple[str, str]]:
    """
    Extract components from an SBOM file.

    Args:
        sbom: Parsed SBOM dictionary

    Returns:
        Set of (package_name, version) tuples for components with valid PURLs
    """
    components = set()

    if "components" not in sbom:
        print("Warning: No 'components' section found in SBOM")
        return components

    for component in sbom["components"]:
        purl = component.get("purl")
        if purl:
            name, version = extract_package_info_from_purl(purl)
            if name and version:
                components.add((name, version))

    return components


def extract_components_by_name(sbom: Dict) -> Dict[str, str]:
    """
    Extract components from an SBOM file, indexed by package name.

    Args:
        sbom: Parsed SBOM dictionary

    Returns:
        Dict mapping package_name to version for components with valid PURLs
    """
    components = {}

    if "components" not in sbom:
        print("Warning: No 'components' section found in SBOM")
        return components

    for component in sbom["components"]:
        purl = component.get("purl")
        if purl:
            name, version = extract_package_info_from_purl(purl)
            if name and version:
                components[name] = version

    return components


def find_new_components(
    base_components: Set[Tuple[str, str]], modified_components: Set[Tuple[str, str]]
) -> Set[Tuple[str, str]]:
    """Find components that are in modified but not in base."""
    return modified_components - base_components


def find_new_packages_by_name(
    base_components: Dict[str, str], modified_components: Dict[str, str]
) -> Set[Tuple[str, str]]:
    """Find packages that are in modified but not in base (by package name only)."""
    base_names = set(base_components.keys())
    modified_names = set(modified_components.keys())
    new_package_names = modified_names - base_names

    return {(name, modified_components[name]) for name in new_package_names}


def find_version_changes(
    base_components: Dict[str, str], modified_components: Dict[str, str]
) -> Set[Tuple[str, str, str]]:
    """Find packages that exist in both but have different versions."""
    version_changes = set()

    for name in base_components:
        if name in modified_components:
            base_version = base_components[name]
            modified_version = modified_components[name]
            if base_version != modified_version:
                version_changes.add((name, base_version, modified_version))

    return version_changes


def write_csv_report(new_packages: Set[Tuple[str, str]], version_changes: Set[Tuple[str, str, str]], output_file: str):
    """Write new packages and version changes to a CSV file with the required columns."""

    # Define the CSV headers as specified
    headers = [
        "Package / Component Name",
        "Version",
        "License",
        "Link to Component's License",
        "Method of Distribution",
        "Usage Method with NV proprietary code",
        "Comments",
        "Location where component was downloaded from",
        "Link to internal IT Controlled Repository",
        "OSRB Bug ID",
    ]

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(headers)

            # Write new packages (sorted for consistent output)
            for name, version in sorted(new_packages):
                row = [
                    name,
                    version,
                    "",  # License - empty for now
                    "",  # Link to Component's License - empty for now
                    "",  # Method of Distribution - empty for now
                    "",  # Usage Method with NV proprietary code - empty for now
                    "",  # Comments - empty for now
                    "",  # Location where component was downloaded from - empty for now
                    "",  # Link to internal IT Controlled Repository - empty for now
                    "",  # OSRB Bug ID - empty for now
                ]
                writer.writerow(row)

            # If there are version changes, add an empty row and then the version changes
            if version_changes:
                # Write empty row (just commas)
                writer.writerow([""] * len(headers))

                # Write version changes (sorted for consistent output)
                for name, old_version, new_version in sorted(version_changes):
                    row = [
                        name,
                        f"{old_version} -> {new_version}",  # Show version change
                        "",  # License - empty for now
                        "",  # Link to Component's License - empty for now
                        "",  # Method of Distribution - empty for now
                        "",  # Usage Method with NV proprietary code - empty for now
                        "Version change",  # Comments - indicate this is a version change
                        "",  # Location where component was downloaded from - empty for now
                        "",  # Link to internal IT Controlled Repository - empty for now
                        "",  # OSRB Bug ID - empty for now
                    ]
                    writer.writerow(row)

        total_entries = len(new_packages) + len(version_changes)
        print(
            f"Successfully wrote {len(new_packages)} new packages and {len(version_changes)} version changes ({total_entries} total entries) to '{output_file}'"
        )

    except Exception as e:
        print(f"Error writing CSV file '{output_file}': {e}")
        sys.exit(1)


def main():
    """Main function to orchestrate the SBOM comparison and CSV generation."""
    parser = argparse.ArgumentParser(
        description="Compare two SBOM files and generate a CSV report of new packages and version changes"
    )
    parser.add_argument("base_sbom", help="Path to the base SBOM file")
    parser.add_argument("modified_sbom", help="Path to the modified SBOM file")
    parser.add_argument("output_csv", help="Path for the output CSV file")
    parser.add_argument(
        "--identify-version-changes", action="store_true", help="Include version changes in the output CSV"
    )

    args = parser.parse_args()

    print(f"Loading base SBOM: {args.base_sbom}")
    base_sbom = load_sbom(args.base_sbom)

    print(f"Loading modified SBOM: {args.modified_sbom}")
    modified_sbom = load_sbom(args.modified_sbom)

    print("Extracting components by name from base SBOM...")
    base_components_by_name = extract_components_by_name(base_sbom)
    print(f"Found {len(base_components_by_name)} unique package names in base SBOM")

    print("Extracting components by name from modified SBOM...")
    modified_components_by_name = extract_components_by_name(modified_sbom)
    print(f"Found {len(modified_components_by_name)} unique package names in modified SBOM")

    print("Finding new packages (by name only)...")
    new_packages = find_new_packages_by_name(base_components_by_name, modified_components_by_name)
    print(f"Found {len(new_packages)} new packages")

    version_changes = set()
    if args.identify_version_changes:
        print("Finding version changes...")
        version_changes = find_version_changes(base_components_by_name, modified_components_by_name)
        print(f"Found {len(version_changes)} version changes")

    print(f"Writing CSV report to: {args.output_csv}")
    write_csv_report(new_packages, version_changes, args.output_csv)

    if new_packages:
        print("\nNew packages found:")
        for name, version in sorted(new_packages):
            print(f"  - {name}@{version}")

    if version_changes:
        print("\nVersion changes found:")
        for name, old_version, new_version in sorted(version_changes):
            print(f"  - {name}: {old_version} -> {new_version}")

    if not new_packages and not version_changes:
        print("No new packages or version changes found.")


if __name__ == "__main__":
    main()
