# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Automatically synchronizes Python dependency lists in BUILD.bazel files.

This script queries Bazel to find all transitive Python dependencies for specific targets
and updates the corresponding *_PIP_PKGS variables in BUILD.bazel. It handles conditional
dependencies that are only included when certain build flags are enabled.

Usage:
    bazel run //internal/scripts/pycena/runtime:update_deps -- //:run
    bazel run //internal/scripts/pycena/runtime:update_deps -- //apps:nre_tools
    bazel run //internal/scripts/pycena/runtime:update_deps -- --validate //:run
    bazel run //internal/scripts/pycena/runtime:update_deps -- --validate //apps:nre_tools
"""

import argparse
import difflib
import os
import re
import subprocess
import sys

from pathlib import Path


# Detect which bazel command to use, this is necessary because bazelisk is used in the CI
def _detect_bazel_command():
    """Detect whether to use 'bazel' or 'bazelisk' command."""
    for cmd in ["bazelisk", "bazel"]:
        try:
            result = subprocess.run([cmd, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    print("Error: Neither 'bazel' nor 'bazelisk' command found in PATH")
    sys.exit(1)


BAZEL_EXE = _detect_bazel_command()


# Configuration mapping targets to their dependency variables and conditional build flags.
# Each target can have multiple flags that enable additional dependencies based on build configuration.

# cquery_args are arguments to the bazel cquery command, affects what packages may be returned
# the condition is the bazel select() condition that is used to separate dependencies in the final *_PIP_PKGS list
CONFIG = {
    "//:run": {
        "pip_pkgs_var": "RUN_PIP_PKGS",  # Variable name in internal/scripts/pycena/runtime/BUILD.bazel to update
        "flags": {
            "internal": {
                "cquery_arg": "",  # No extra args needed - uses query_type instead
                "condition": "//bazel/conditions:internal",  # Bazel select() condition for internal builds
                "query_type": "internal",  # Special query that filters for _internal_ packages
            },
        },
    },
    "//apps:nre_tools": {
        "pip_pkgs_var": "NRE_TOOLS_PIP_PKGS",
        "flags": {
            "ncore_aux_data": {
                "cquery_arg": "--//bazel/flags:ncore_aux_data=True",
                "condition": "//bazel/conditions:ncore_aux_data",
            },
            "mask_annotator": {
                "cquery_arg": "--//bazel/flags:mask_annotator=True",
                "condition": "//bazel/conditions:mask_annotator",
            },
            "asset_harvester": {
                "cquery_arg": "--//bazel/flags:asset_harvester=True",
                "condition": "//bazel/conditions:asset_harvester",
            },
        },
        # Combinations of flags that need special handling to avoid duplicate deps
        # When both flags in a combination are True, the combination condition is more specific
        "combinations": {
            "ncore_aux_data_and_asset_harvester": {
                "flags": ["ncore_aux_data", "asset_harvester"],
                "condition": "//bazel/conditions:ncore_aux_data_and_asset_harvester",
            },
            "asset_harvester_and_mask_annotator": {
                "flags": ["asset_harvester", "mask_annotator"],
                "condition": "//bazel/conditions:asset_harvester_and_mask_annotator",
            },
            "ncore_aux_data_and_mask_annotator": {
                "flags": ["ncore_aux_data", "mask_annotator"],
                "condition": "//bazel/conditions:ncore_aux_data_and_mask_annotator",
            },
        },
    },
    "//nre/nrm:run": {
        "pip_pkgs_var": "NRM_PIP_PKGS",
        "flags": {
            "internal": {
                "cquery_arg": "",
                "condition": "//bazel/conditions:internal",
                "query_type": "internal",
            },
        },
    },
}


def get_workspace_root():
    """Get the workspace root directory."""
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        return Path(os.environ["BUILD_WORKSPACE_DIRECTORY"])
    else:
        return Path(__file__).parent.parent.parent.parent


def get_deps(target, flags=None, internal=False, disable_flags=None):
    """
    Query Bazel for Python library dependencies of a target.

    Args:
        target: Bazel target to query (e.g. "//:run", "//apps:nre_tools")
        flags: List of build flags to enable during query
        internal: Whether to query for internal-only dependencies
        disable_flags: List of flag names to explicitly disable (set to False/default)

    Returns:
        List of Bazel package targets or None on error
    """
    internal_flag = "true" if internal else "false"
    bazel_cmd = f"{BAZEL_EXE} cquery 'kind(\"py_library\", deps({target}))' --notool_deps --//bazel/flags:internal={internal_flag}"

    # Explicitly disable flags if requested (to override .bazelrc.user settings)
    if disable_flags and target in CONFIG:
        target_flags = CONFIG[target]["flags"]
        for flag_name in disable_flags:
            if flag_name in target_flags:
                cquery_arg = target_flags[flag_name].get("cquery_arg", "")
                if cquery_arg:
                    # Convert the flag to its disabled form
                    # e.g., "--//bazel/flags:asset_harvester=True" -> "--//bazel/flags:asset_harvester=False"
                    disabled_arg = cquery_arg.replace("=True", "=False")
                    bazel_cmd += f" {disabled_arg}"

    # Add flag arguments if provided
    if flags:
        # Get flags for this specific target
        if target not in CONFIG:
            print(f"Warning: Unknown target '{target}', cannot validate flags", file=sys.stderr)
            return None

        target_flags = CONFIG[target]["flags"]

        for flag in flags:
            if flag in target_flags:
                bazel_cmd += f" {target_flags[flag]['cquery_arg']}"
            else:
                print(f"Warning: Unknown flag '{flag}' for target '{target}', skipping", file=sys.stderr)

    # Print the command being used
    print(f"Running command: {bazel_cmd}")

    # Change to workspace root before running bazel command
    workspace_root = get_workspace_root()

    try:
        # Run cquery and capture output
        result = subprocess.run(bazel_cmd, shell=True, capture_output=True, text=True, check=True, cwd=workspace_root)

        # Filter and sort
        raw_lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip() and "pkg" in line]
        if internal:
            raw_lines = [line for line in raw_lines if "_internal_" in line]
        raw_lines = sorted(raw_lines)

        # Split lines and clean up the output
        lines = []
        for line in raw_lines:
            line = line.strip()
            if line:
                # Remove any extra information after the :pkg part
                # e.g., "@nre_pip_deps_package//:pkg (8a33404)" -> "@nre_pip_deps_package//:pkg"
                if ":pkg" in line:
                    clean_line = line.split(":pkg")[0] + ":pkg"

                    # Post-process with regex to extract package names
                    # Determine if this is an internal or regular dependency
                    is_internal = "nre_pip_deps_internal_311" in clean_line

                    if is_internal or "nre_pip_deps_311" in clean_line:
                        # Extract package name from bazel target path
                        # e.g., "@@rules_python++pip+nre_pip_deps_311_antlr4_python3_runtime_sdist_f224469b//:pkg" -> "antlr4_python3_runtime"
                        # e.g., "@@rules_python++pip+nre_pip_deps_internal_311_numpy_cp311_cp311_manylinux_2_17_x86_64_666dbfb6//:pkg" -> "numpy"
                        prefix = "nre_pip_deps_internal_311_" if is_internal else "nre_pip_deps_311_"
                        func_name = "pip_requirement_internal" if is_internal else "pip_requirement"

                        # Try to match with platform suffixes first (py2/py3/cp311/sdist/wheel/linux)
                        match = re.search(rf"{re.escape(prefix)}(.+?)_(?:py[23]|cp\d+|sdist|wheel|linux)", clean_line)
                        if match:
                            package_name = match.group(1)
                            clean_line = f'{func_name}("{package_name}")'
                        else:
                            # Fallback for packages without platform suffixes (e.g., apex, mmcv)
                            match = re.search(rf"{re.escape(prefix)}([^/]+?)/", clean_line)
                            if match:
                                package_name = match.group(1)
                                clean_line = f'{func_name}("{package_name}")'

                    lines.append(clean_line)

        # Remove duplicates while preserving order
        unique_lines = []
        seen = set()
        for line in lines:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)

        return unique_lines

    except subprocess.CalledProcessError as e:
        if e.stdout:
            print("cquery stdout:", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print("cquery stderr:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        print(f"Error: cquery failed with exit code {e.returncode}", file=sys.stderr)

        return None
    except OSError as e:
        print(f"Error: cquery failed for {target}: {e}", file=sys.stderr)
        return None


def get_all_dependencies(target, flags=None):
    """
    Get complete dependency list including base and tool-specific dependencies.

    Uses intersection-based logic:
    1. Query each flag individually (one flag ON, others OFF)
    2. Base = intersection of all flag deps (common to all)
    3. Tool-specific = flag_deps - base (complete list per tool, overlap is OK)

    For flags with special query types (e.g., 'internal'), the appropriate query method is used.

    Args:
        target: Bazel target to query for dependencies
        flags: List of flags to use defined in CONFIG[target]["flags"]

    Returns:
        Dict with 'base', 'tool_specific', and 'combinations' dependencies, or None on error
    """
    if target not in CONFIG:
        print(f"Warning: Unknown target '{target}', cannot process flags")
        return None

    target_config = CONFIG[target]
    target_flags = target_config["flags"]
    combinations_config = target_config.get("combinations", {})

    # If no flags provided, query with all flags disabled
    if not flags:
        print(f"Querying base dependencies from {target}")
        base_deps = get_deps(target, disable_flags=[])
        if base_deps is None:
            print("Failed to get base dependencies", file=sys.stderr)
            print("Aborting: cquery failed.", file=sys.stderr)
            return None
        # Add hardcoded dependencies
        base_deps.insert(0, '"@rules_python//python/runfiles"')
        return {"base": base_deps, "tool_specific": {}, "combinations": combinations_config}

    # Step 1: Query each flag individually
    flag_deps = {}
    valid_flags = []

    for flag in flags:
        if flag not in target_flags:
            print(f"Warning: Unknown flag '{flag}' for target '{target}', skipping")
            continue

        valid_flags.append(flag)
        flag_info = target_flags[flag]
        other_flags = [f for f in flags if f != flag]

        # Check if this flag has a special query type
        query_type = flag_info.get("query_type")

        if query_type == "internal":
            # Special handling for internal dependencies (queries _internal_ packages)
            print(f"Querying dependencies for {flag} (internal packages)")
            deps = get_deps(target, internal=True)
        else:
            # Standard cquery with flag enabled, others disabled
            print(f"Querying dependencies for {flag} (others disabled)")
            deps = get_deps(target, [flag], disable_flags=other_flags)

        if deps is not None:
            flag_deps[flag] = set(deps)
            print(f"  Found {len(deps)} dependencies for {flag}")
        else:
            print(f"Aborting: failed to get dependencies for {flag}", file=sys.stderr)
            return None

    if not flag_deps:
        print("No flag dependencies found")
        return {"base": [], "tool_specific": {}, "combinations": combinations_config}

    # Step 2: Find base dependencies
    # For flags with query_type="internal", we need to query without that flag to get the base
    # For regular flags, base = intersection of all flags
    internal_flags = [f for f in valid_flags if target_flags[f].get("query_type") == "internal"]
    regular_flags = [f for f in valid_flags if f not in internal_flags]

    if regular_flags:
        # Base = intersection of all regular flags
        regular_dep_sets = [flag_deps[f] for f in regular_flags]
        base_deps_set = set.intersection(*regular_dep_sets)
    else:
        # Only internal flags - query without internal to get base
        print(f"Querying base dependencies from {target} (no flags)")
        base_deps = get_deps(target, disable_flags=valid_flags)
        if base_deps is None:
            print("Aborting: cquery failed.", file=sys.stderr)
            return None
        base_deps_set = set(base_deps)

    # Add hardcoded dependencies
    base_deps_set.add('"@rules_python//python/runfiles"')

    print(f"Base (intersection): {len(base_deps_set)} common dependencies")

    # Step 3: Find per-tool deps beyond base (diff from base = flag_deps - base)
    # This is the NEW logic: each tool gets its complete dependency list (minus base)
    # Overlap between tools is OK - deduplication happens in Starlark
    tool_specific_deps = {}

    for flag in valid_flags:
        # Per-tool deps beyond base = this flag's deps - base (diff from base; overlap OK)
        specific_deps_set = flag_deps[flag] - base_deps_set
        specific_deps = sorted(specific_deps_set)

        if specific_deps:
            tool_specific_deps[flag] = {
                "condition": target_flags[flag]["condition"],
                "deps": specific_deps,
            }
            print(f"  {flag}: {len(specific_deps)} tool-specific dependencies")
        else:
            print(f"  {flag}: no tool-specific dependencies")

    base_deps = sorted(base_deps_set)
    return {"base": base_deps, "tool_specific": tool_specific_deps, "combinations": combinations_config}


def _flag_to_var_name(flag_name):
    """Convert flag name to variable name suffix (e.g., 'ncore_aux_data' -> 'NCORE_AUX_DATA')."""
    return flag_name.upper()


def generate_bazel_dependency_list(all_deps, dest_var_name, target=None):
    """
    Generate Bazel BUILD.bazel syntax for dependency lists with conditional select() statements.

    Args:
        all_deps: Dict with 'base' and 'tool_specific' dependency lists
        dest_var_name: Variable name to assign (e.g. "RUN_PIP_PKGS")
        target: Target name for special handling (e.g. "//apps:nre_tools")

    Returns:
        String containing BUILD.bazel variable assignment with select() statements
    """
    if not all_deps or not all_deps["base"]:
        return f"{dest_var_name} = []"

    # Special handling for NRE_TOOLS_PIP_PKGS - generate structured format with _DIFF variables
    if target == "//apps:nre_tools":
        # First generate the BASE variable with all base dependencies
        # Base contains deps common to ALL tools (intersection)
        base_var_name = f"{dest_var_name}_BASE"
        result = f"{base_var_name} = [\n"
        for dep in sorted(all_deps["base"]):
            result += f"    {dep},\n"
        result += "]\n\n"

        # Generate _DIFF variables for each tool
        tool_specific = all_deps.get("tool_specific", {})
        combinations_config = all_deps.get("combinations", {})

        # Define the order of tools for consistent output
        tool_order = ["ncore_aux_data", "mask_annotator", "asset_harvester"]

        for flag_name in tool_order:
            if flag_name in tool_specific:
                flag_info = tool_specific[flag_name]
                var_suffix = _flag_to_var_name(flag_name)
                result += f"NRE_TOOLS_PIP_PKGS_{var_suffix}_DIFF = [\n"
                for dep in sorted(flag_info["deps"]):
                    result += f"    {dep},\n"
                result += "]\n\n"

        # NRE_TOOLS_PIP_PKGS_ALL_TOOLS_DEDUPED = union of all tools (deduplicated)
        tool_vars = [f"NRE_TOOLS_PIP_PKGS_{_flag_to_var_name(f)}_DIFF" for f in tool_order if f in tool_specific]
        if tool_vars:
            result += "NRE_TOOLS_PIP_PKGS_ALL_TOOLS_DEDUPED = list({p: None for p in (\n"
            result += f"    {' + '.join(tool_vars)}\n"
            result += ")}.keys())\n\n"

        # Generate combination unions (e.g., ncore_aux_data + asset_harvester)
        for combo_name, combo_info in combinations_config.items():
            combo_flags = combo_info["flags"]
            combo_vars = [f"NRE_TOOLS_PIP_PKGS_{_flag_to_var_name(f)}_DIFF" for f in combo_flags if f in tool_specific]
            if combo_vars:
                combo_var_name = f"NRE_TOOLS_PIP_PKGS_{_flag_to_var_name(combo_name)}_DEDUPED"
                result += f"{combo_var_name} = list({{p: None for p in (\n"
                result += f"    {' + '.join(combo_vars)}\n"
                result += ")}.keys())\n\n"

        # Generate final NRE_TOOLS_PIP_PKGS with single select
        result += f"{dest_var_name} = {base_var_name} + select({{\n"
        result += f'    "//bazel/conditions:all_tools": NRE_TOOLS_PIP_PKGS_ALL_TOOLS_DEDUPED,\n'

        # Add combination conditions
        for combo_name, combo_info in combinations_config.items():
            combo_var_name = f"NRE_TOOLS_PIP_PKGS_{_flag_to_var_name(combo_name)}_DEDUPED"
            result += f'    "{combo_info["condition"]}": {combo_var_name},\n'

        # Add individual tool conditions
        for flag_name in tool_order:
            if flag_name in tool_specific:
                var_suffix = _flag_to_var_name(flag_name)
                condition = tool_specific[flag_name]["condition"]
                result += f'    "{condition}": NRE_TOOLS_PIP_PKGS_{var_suffix}_DIFF,\n'

        result += '    "//conditions:default": [],\n'
        result += "})\n\n"

        return result

    # Default handling for other targets (e.g., //:run)
    result = f"{dest_var_name} = [\n"

    # Add base dependencies
    for dep in sorted(all_deps["base"]):
        result += f"    {dep},\n"

    result += "]"

    # Add each conditional flag as a separate select() statement
    if all_deps.get("tool_specific"):
        for flag_name, flag_info in all_deps["tool_specific"].items():
            result += " + select({\n"
            result += f'    "{flag_info["condition"]}": [\n'
            for dep in sorted(flag_info["deps"]):
                result += f"        {dep},\n"
            result += "    ],\n"
            result += '    "//conditions:default": [],\n'
            result += "})"

    return result


def _print_validation_diff(dest_var_name, old_content, new_content, target):
    """Print unified diff between current and expected BUILD.bazel content."""
    print(f"✗ {dest_var_name} is out of sync\n")

    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile="BUILD.bazel (current)",
        tofile="BUILD.bazel (expected)",
        n=3,
        lineterm="",
    )

    print("".join(diff))
    print(f"\nRun 'bazel run //internal/scripts/pycena/runtime:update_deps -- {target}' to update")


def update_or_validate_build_file(target, flags, dest_var_name, validate_only=False):
    """
    Update or validate BUILD.bazel file dependency variable.

    Args:
        target: Bazel target to query dependencies for
        flags: List of flags to process defined in CONFIG[target]["flags"]
        dest_var_name: Variable name to update in BUILD.bazel
        validate_only: If True, check for differences without updating

    Returns:
        True if up-to-date (or update succeeded), False if out-of-sync (or error)
    """
    workspace_root = get_workspace_root()
    build_file = workspace_root / "internal" / "scripts" / "pycena" / "runtime" / "BUILD.bazel"

    if not build_file.exists():
        print(f"BUILD.bazel not found at {build_file}")
        return False

    action = "Validating" if validate_only else "Updating"
    print(f"{action} {dest_var_name} in BUILD.bazel")

    # Get all dependencies with the specified flags
    all_deps = get_all_dependencies(target, flags)
    if not all_deps:
        print("Dependency query failed.", file=sys.stderr)
        return False

    # Read current BUILD file
    with open(build_file, "r") as f:
        content = f.read()

    # Generate new dependency list
    new_list = generate_bazel_dependency_list(all_deps, dest_var_name, target)

    # Special handling for NRE_TOOLS_PIP_PKGS which has BASE + _DIFF variables + unions + main variable
    if target == "//apps:nre_tools":
        base_var_name = f"{dest_var_name}_BASE"
        # Pattern to match the entire NRE_TOOLS_PIP_PKGS section.
        #
        # This includes:
        #   - BASE variable
        #   - _*_DIFF variables
        #   - Computed unions (_ALL_TOOLS_DEPS_DEDUPED, etc.)
        #   - Final NRE_TOOLS_PIP_PKGS = ... + select({...})
        # Pattern consumes trailing \n\n to ensure idempotent replacement
        pattern = rf"{re.escape(base_var_name)} = \[.*?{re.escape(dest_var_name)} = {re.escape(base_var_name)} \+ select\(\{{.*?\}}\)\s*\n\n"

        if not re.search(pattern, content, re.DOTALL):
            # Debug: print what we're looking for
            print(f"Looking for pattern starting with: {base_var_name} = [")
            print(f"Variable names: {base_var_name} and {dest_var_name}")
            print(f"Could not find {base_var_name} and {dest_var_name} in {build_file}")
            # Print first few lines of content to help debug
            lines = content.split("\n")
            for i, line in enumerate(lines[300:350], start=301):
                if base_var_name in line or dest_var_name in line:
                    print(f"Line {i}: {line[:80]}")
            return False

        new_content = re.sub(pattern, new_list, content, flags=re.DOTALL)
    else:
        # Standard handling for other variables
        # Use a comprehensive regex that handles both formats:
        # 1. VARIABLE = [...] + select({...})
        # 2. VARIABLE = select({...}) + select({...})
        pattern = rf"{re.escape(dest_var_name)} = (?:\[.*?\](?:\s*\+\s*select\s*\(\{{.*?\}}\))*|select\s*\(\{{.*?\}}\)(?:\s*\+\s*select\s*\(\{{.*?\}}\))*)"

        if not re.search(pattern, content, re.DOTALL):
            print(f"Could not find {dest_var_name} in {build_file}")
            return False

        new_content = re.sub(pattern, new_list, content, flags=re.DOTALL)

    if new_content == content:
        print(f"{dest_var_name} is up to date in {build_file}")
        return True

    # If validating, report the mismatch with diff and return False
    if validate_only:
        _print_validation_diff(dest_var_name, content, new_content, target)
        return False

    # Write back the updated content
    with open(build_file, "w") as f:
        f.write(new_content)

    print(f"Updated {dest_var_name} in {build_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update pip dependency lists in BUILD.bazel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  bazel run //internal/scripts/pycena/runtime:update_deps -- //:run
  bazel run //internal/scripts/pycena/runtime:update_deps -- //apps:nre_tools
  bazel run //internal/scripts/pycena/runtime:update_deps -- --validate //:run
  bazel run //internal/scripts/pycena/runtime:update_deps -- --validate //apps:nre_tools
        """,
    )

    parser.add_argument("target", help="Bazel target to query for dependencies")
    parser.add_argument(
        "--validate", action="store_true", help="Validate dependencies without updating (exit 1 if out of sync)"
    )

    args = parser.parse_args()

    # Validate bazel target
    if args.target not in CONFIG:
        print(f"Error: Unknown target '{args.target}'")
        print(f"Available targets: {', '.join(CONFIG.keys())}")
        return 1

    # Get destination *_PIP_PKGS variable from target config
    dest_var_name = CONFIG[args.target]["pip_pkgs_var"]

    # Use all flags defined for this target
    target_flags = CONFIG[args.target]["flags"]
    flags = list(target_flags.keys())

    if flags:
        print(f"Using flags for {args.target}: {', '.join(flags)}")
    else:
        print(f"No conditional flags defined for {args.target}")

    # Update or validate based on flag
    success = update_or_validate_build_file(args.target, flags, dest_var_name, validate_only=args.validate)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
