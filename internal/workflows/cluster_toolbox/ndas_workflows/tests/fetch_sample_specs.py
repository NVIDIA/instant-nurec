# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Fetches sample workflow specs from Maglev for use in integration tests.

This script scans all ndas_workflows config files to find *source_wf references
(source_wf, shim_source_wf, etc.), deduplicates them, and fetches the workflow
specs using the maglev CLI. Output files are named after the workflow
(e.g., "josliu-car2sim-template/latest" -> "josliu-car2sim-template.yaml").

Usage:
    python fetch_sample_specs.py
    # Or via bazel:
    bazel run //internal/workflows/cluster_toolbox/ndas_workflows/tests:fetch_sample_specs
"""

import re
import subprocess
import sys

from pathlib import Path


def find_config_dir() -> Path:
    """Find the ndas_workflows config directory (bazel only)."""
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent.parent / "cluster_configs" / "ndas_workflows"
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found at {config_dir}. "
            "This script must be run via bazel: bazel run //...tests:fetch_sample_specs"
        )
    return config_dir


def find_output_dir() -> Path:
    """Find or create the sample_test_data output directory, clearing any existing files."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "sample_test_data"

    # Clear existing files to avoid stale specs
    if output_dir.exists():
        for file in output_dir.glob("*.yaml"):
            file.unlink()
    else:
        output_dir.mkdir()

    return output_dir


def extract_source_wfs(config_dir: Path) -> dict[str, str]:
    """Extract all unique source_wf values from config files.

    Scans all config files and extracts workflow paths, using the workflow name
    (part before the /) as the output filename. This makes it easy to map
    workflow paths to cached specs in mock testing.

    Returns:
        Dict mapping output filename to workflow path
        (e.g., "josliu-car2sim-template.yaml" -> "josliu-car2sim-template/latest")
    """
    source_wfs: dict[str, str] = {}

    # Pattern to match any config ending with source_wf (e.g., source_wf, shim_source_wf, etc.)
    source_wf_pattern = re.compile(r"^\s*\w*source_wf:\s*(.+)$")

    for config_file in config_dir.glob("*.yaml"):
        with open(config_file) as f:
            for line in f:
                wf_match = source_wf_pattern.match(line)
                if wf_match:
                    wf_path = wf_match.group(1).strip()

                    # Skip variable references like ${car2sim.source_wf}
                    if wf_path.startswith("${"):
                        continue

                    # Extract workflow name from path (e.g., "user-workflow/latest" -> "user-workflow")
                    wf_name = wf_path.split("/")[0]
                    output_filename = f"{wf_name}.yaml"

                    # Deduplicate by workflow name (same workflow may appear in multiple configs)
                    if output_filename not in source_wfs:
                        source_wfs[output_filename] = wf_path

    return source_wfs


def fetch_workflow_spec(wf_path: str, output_file: Path) -> bool:
    """Fetch a workflow spec using maglev CLI.

    Args:
        wf_path: The workflow path (e.g., "user-workflow/latest-success")
        output_file: Path to write the spec to

    Returns:
        True if successful, False otherwise
    """
    print(f"Fetching {wf_path} -> {output_file.name}")

    try:
        result = subprocess.run(
            ["maglev", "workflows2", "get", wf_path, "--spec"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
            return False

        # Write the spec, filtering out CLI warning lines
        with open(output_file, "w") as f:
            for line in result.stdout.splitlines(keepends=True):
                if not line.startswith("Your CLI version might be outdated"):
                    f.write(line)

        print(f"  OK ({output_file.stat().st_size} bytes)")
        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: Timeout")
        return False
    except FileNotFoundError:
        print("  ERROR: maglev CLI not found")
        return False


def main() -> int:
    """Main entry point."""
    print("Finding config directory...")
    try:
        config_dir = find_config_dir()
        print(f"  Found: {config_dir}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return 1

    print("\nFinding output directory...")
    output_dir = find_output_dir()
    print(f"  Output: {output_dir}")

    print("\nExtracting source_wf references from configs...")
    source_wfs = extract_source_wfs(config_dir)

    if not source_wfs:
        print("  No source_wf references found!")
        return 1

    print(f"  Found {len(source_wfs)} unique workflows:")
    for filename, wf_path in sorted(source_wfs.items()):
        print(f"    {filename}: {wf_path}")

    print("\nFetching workflow specs...")
    success_count = 0
    failure_count = 0

    for filename, wf_path in sorted(source_wfs.items()):
        output_file = output_dir / filename
        if fetch_workflow_spec(wf_path, output_file):
            success_count += 1
        else:
            failure_count += 1

    print(f"\nDone: {success_count} succeeded, {failure_count} failed")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
