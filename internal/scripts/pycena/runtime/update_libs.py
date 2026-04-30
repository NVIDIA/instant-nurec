# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Automatically update library dependency lists in BUILD.bazel files."""

import argparse
import os
import re
import subprocess
import sys

from pathlib import Path


# Config for updating library dependencies of targets
CONFIG = {
    "//:run": {
        "libs_var": "RUN_LIBS",
        "tools": {
            "run": {
                "target": "//:run",
                "condition": None,  # This is the base/default case
            },
        },
        # Additional libs that don't match pylib_cc/cclib pattern but are required
        "additional_libs": [
            "//libs/slang_gaussians:pylib",
            "//nre/models/post_processings/ppisp/slang:pylib",
        ],
    },
    "//apps:nre_tools": {
        "libs_var": "NRE_TOOLS_LIBS",
        "tools": {
            "ncore_aux_data": {
                "target": "//apps/aux_gen:ncore_aux_data",
                "condition": "//bazel/conditions:ncore_aux_data",
            },
            "asset_harvester": {
                "target": "//apps/asset_harvester:asset_harvester",
                "condition": "//bazel/conditions:asset_harvester",
            },
            "mask_annotator": {
                "target": "//apps/avmask_annotator:mask_annotator",
                "condition": "//bazel/conditions:mask_annotator",
            },
        },
        # Combinations of tools that need special handling to avoid duplicate deps
        "combinations": {
            "ncore_aux_data_and_asset_harvester": {
                "tools": ["ncore_aux_data", "asset_harvester"],
                "condition": "//bazel/conditions:ncore_aux_data_and_asset_harvester",
            },
            "asset_harvester_and_mask_annotator": {
                "tools": ["asset_harvester", "mask_annotator"],
                "condition": "//bazel/conditions:asset_harvester_and_mask_annotator",
            },
            "ncore_aux_data_and_mask_annotator": {
                "tools": ["ncore_aux_data", "mask_annotator"],
                "condition": "//bazel/conditions:ncore_aux_data_and_mask_annotator",
            },
        },
    },
    "//nre/nrm:run": {
        "libs_var": "NRM_LIBS",
        "tools": {
            "run": {
                "target": "//nre/nrm:run",
                "condition": None,
            },
        },
        "additional_libs": [
            "//libs/slang_gaussians:pylib",
            "//nre/models/post_processings/ppisp/slang:pylib",
        ],
    },
}

# External gsplat labels to exclude when the in-repo wrapper is present.
# The wrapper //nre/models/gaussians:pylib_cc_gsplat is the only label that should appear
# in RUN_LIBS (and others) so the pycena runtime never depends on the external repo.
GSPLAT_REPO_LIBS = (
    "@gsplat_repo//:pylib_cc",
    "@gsplat_repo_local//:pylib_cc",
)
GSPLAT_WRAPPER = "//nre/models/gaussians:pylib_cc_gsplat"


def get_workspace_root():
    """Get the workspace root directory."""
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        return Path(os.environ["BUILD_WORKSPACE_DIRECTORY"])
    else:
        return Path(__file__).parent.parent.parent.parent


def get_pylib_cc_deps(target):
    """Get pylib_cc and cclib dependencies for a target."""
    bazel_cmd = f"bazel cquery --noimplicit_deps 'deps({target})'"

    # Print the command being used
    print(f"Running command: {bazel_cmd}")

    # Change to workspace root before running bazel command
    workspace_root = get_workspace_root()

    try:
        result = subprocess.run(bazel_cmd, shell=True, capture_output=True, text=True, check=True, cwd=workspace_root)
        if result.stderr:
            print("cquery stderr:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

        # Split lines and clean up the output
        lines = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and ("pylib_cc" in line or "cclib" in line):
                # Extract the target label (e.g., "//libs/vren:pylib_cc" or "//libs/something:cclib")
                # Remove any extra information after the target
                if " " in line:
                    clean_line = line.split(" ")[0]
                else:
                    clean_line = line
                lines.append(clean_line)
        return sorted(lines)

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


def get_all_library_dependencies(config_name):
    """Get library dependencies using intersection-based logic.

    1. Query each tool individually
    2. Base = intersection of ALL tools (deps common to all)
    3. Combo-shared = intersection of combo tools - base
    4. Tool-unique = tool - other_combo_tools - base

    For single-tool configs, intersection of one set = that set (all deps are base).
    """
    if config_name not in CONFIG:
        print(f"Error: Unknown config '{config_name}'")
        return None

    config = CONFIG[config_name]
    tools_config = config["tools"]
    combinations_config = config.get("combinations", {})

    # Step 1: Query each tool individually
    tool_deps = {}
    for tool_name, tool_info in tools_config.items():
        print(f"Querying dependencies for {tool_name} from {tool_info['target']}")
        deps = get_pylib_cc_deps(tool_info["target"])
        if deps is not None:
            tool_deps[tool_name] = set(deps)
            # Use the in-repo gsplat wrapper only; do not list external gsplat repo labels.
            if GSPLAT_WRAPPER in tool_deps[tool_name]:
                tool_deps[tool_name] -= set(GSPLAT_REPO_LIBS)
            print(f"  Found {len(deps)} dependencies")
        else:
            print(f"  Failed to get dependencies for {tool_name}", file=sys.stderr)
            return None

    if not tool_deps or all(len(deps) == 0 for deps in tool_deps.values()):
        print("No tool dependencies found")
        return {"base": [], "conditional": {}, "combinations": {}}

    # Step 2: Base = intersection of ALL tools
    all_dep_sets = [deps for deps in tool_deps.values() if deps]
    if all_dep_sets:
        base_deps_set = set.intersection(*all_dep_sets)
    else:
        base_deps_set = set()

    # Add additional_libs to base deps
    additional_libs = config.get("additional_libs", [])
    if additional_libs:
        base_deps_set.update(additional_libs)
        print(f"Added {len(additional_libs)} additional libs to base")

    print(f"Base (intersection of all tools): {len(base_deps_set)} common dependencies")

    # Step 3: Compute combination-shared deps
    combination_deps = {}
    for combo_name, combo_info in combinations_config.items():
        combo_tools = combo_info["tools"]
        if all(t in tool_deps for t in combo_tools):
            combo_dep_sets = [tool_deps[t] for t in combo_tools]
            combo_intersection = set.intersection(*combo_dep_sets)
            combo_shared = combo_intersection - base_deps_set
            if combo_shared:
                combination_deps[combo_name] = {
                    "condition": combo_info["condition"],
                    "tools": combo_tools,
                    "deps": sorted(combo_shared),
                }
                print(f"  {combo_name} (shared): {len(combo_shared)} dependencies")
            else:
                print(f"  {combo_name} (shared): no shared dependencies")

    # Step 4: Tool-unique deps = tool - other_combo_tools - base
    # Only compute for tools that have a condition (skip base-only tools)
    conditional_deps = {}
    for tool_name, tool_info in tools_config.items():
        # Skip tools without conditions (they only contribute to base)
        if tool_info.get("condition") is None:
            continue

        # Find other tools in same combinations
        other_combo_tools = set()
        for combo_info in combinations_config.values():
            if tool_name in combo_info["tools"]:
                other_combo_tools.update(t for t in combo_info["tools"] if t != tool_name)

        # Unique = this tool - other combo tools - base
        unique_deps_set = tool_deps[tool_name].copy()
        for other_tool in other_combo_tools:
            if other_tool in tool_deps:
                unique_deps_set -= tool_deps[other_tool]
        unique_deps_set -= base_deps_set

        if unique_deps_set:
            conditional_deps[tool_name] = {
                "condition": tool_info["condition"],
                "deps": sorted(unique_deps_set),
            }
            print(f"  {tool_name}: {len(unique_deps_set)} unique dependencies")
        else:
            print(f"  {tool_name}: no unique dependencies")

    return {
        "base": sorted(base_deps_set),
        "conditional": conditional_deps,
        "combinations": combination_deps,
    }


def _flag_to_var_name(flag_name):
    """Convert flag name to variable name suffix (e.g., 'ncore_aux_data' -> 'NCORE_AUX_DATA')."""
    return flag_name.upper()


def generate_libs_list(all_deps, libs_var_name, config_name=None):
    """Generate the library dependency list with conditionals using deduplicated unions."""
    if not all_deps:
        return f"{libs_var_name} = []"

    # For nre_tools, use the new structured format with _DIFF variables and unions
    if config_name == "//apps:nre_tools":
        combinations_config = CONFIG[config_name].get("combinations", {})
        tools_config = CONFIG[config_name]["tools"]

        # First generate the BASE variable
        base_var_name = f"{libs_var_name}_BASE"
        result = f"{base_var_name} = [\n"
        for dep in sorted(all_deps.get("base", [])):
            result += f'    "{dep}",\n'
        result += "]\n\n"

        # Define the order of tools for consistent output
        tool_order = ["ncore_aux_data", "mask_annotator", "asset_harvester"]

        # Generate NRE_TOOLS_LIBS_*_DIFF variables for each tool
        for tool_name in tool_order:
            var_suffix = _flag_to_var_name(tool_name)
            result += f"NRE_TOOLS_LIBS_{var_suffix}_DIFF = [\n"
            if tool_name in all_deps.get("conditional", {}):
                for dep in sorted(all_deps["conditional"][tool_name]["deps"]):
                    result += f'    "{dep}",\n'
            result += "]\n\n"

        # NRE_TOOLS_LIBS_ALL_TOOLS_DEDUPED = union of all tools (deduplicated)
        tool_vars = [f"NRE_TOOLS_LIBS_{_flag_to_var_name(t)}_DIFF" for t in tool_order]
        result += "NRE_TOOLS_LIBS_ALL_TOOLS_DEDUPED = list({p: None for p in (\n"
        result += f"    {' + '.join(tool_vars)}\n"
        result += ")}.keys())\n\n"

        # Generate combination unions
        for combo_name, combo_info in combinations_config.items():
            combo_tools = combo_info["tools"]
            combo_vars = [f"NRE_TOOLS_LIBS_{_flag_to_var_name(t)}_DIFF" for t in combo_tools]
            combo_var_name = f"NRE_TOOLS_LIBS_{_flag_to_var_name(combo_name)}_DEDUPED"
            result += f"{combo_var_name} = list({{p: None for p in (\n"
            result += f"    {' + '.join(combo_vars)}\n"
            result += ")}.keys())\n\n"

        # Generate final NRE_TOOLS_LIBS with single select
        result += f"{libs_var_name} = {base_var_name} + select({{\n"
        result += f'    "//bazel/conditions:all_tools": NRE_TOOLS_LIBS_ALL_TOOLS_DEDUPED,\n'

        # Add combination conditions
        for combo_name, combo_info in combinations_config.items():
            combo_var_name = f"NRE_TOOLS_LIBS_{_flag_to_var_name(combo_name)}_DEDUPED"
            result += f'    "{combo_info["condition"]}": {combo_var_name},\n'

        # Add individual tool conditions
        for tool_name in tool_order:
            var_suffix = _flag_to_var_name(tool_name)
            condition = tools_config[tool_name]["condition"]
            result += f'    "{condition}": NRE_TOOLS_LIBS_{var_suffix}_DIFF,\n'

        result += '    "//conditions:default": [],\n'
        result += "})\n\n"

        return result

    # Default handling for other targets (e.g., //:run)
    result = f"{libs_var_name} = [\n"

    # Add base dependencies
    for dep in all_deps.get("base", []):
        result += f'    "{dep}",\n'

    result += "]"

    # Add each conditional flag as a separate select() statement
    if all_deps.get("conditional"):
        for tool_name, tool_info in all_deps["conditional"].items():
            result += " + select({\n"
            result += f'    "{tool_info["condition"]}": [\n'
            for dep in sorted(tool_info["deps"]):
                result += f'        "{dep}",\n'
            result += "    ],\n"
            result += '    "//conditions:default": [],\n'
            result += "})"

    return result


def update_build_file(config_name):
    """Update the BUILD.bazel file with current library dependencies."""
    if config_name not in CONFIG:
        print(f"Error: Unknown config '{config_name}'")
        return False

    workspace_root = get_workspace_root()
    build_file = workspace_root / "internal" / "scripts" / "pycena" / "runtime" / "BUILD.bazel"

    if not build_file.exists():
        print(f"BUILD.bazel not found at {build_file}")
        return False

    libs_var_name = CONFIG[config_name]["libs_var"]
    print(f"Updating {libs_var_name} in BUILD.bazel")

    # Get all library dependencies
    all_deps = get_all_library_dependencies(config_name)
    if not all_deps:
        return False

    # Read current BUILD file
    with open(build_file, "r") as f:
        content = f.read()

    # Generate new library list
    new_list = generate_libs_list(all_deps, libs_var_name, config_name)

    # For NRE_TOOLS_LIBS, match the entire structured section
    if config_name == "//apps:nre_tools":
        base_var_name = f"{libs_var_name}_BASE"
        # Pattern to match the entire NRE_TOOLS_LIBS section.
        # We match from the BASE variable definition through the final:
        #   NRE_TOOLS_LIBS = NRE_TOOLS_LIBS_BASE + select({...})
        # Also swallow any legacy/duplicated generated header comment lines that may exist
        # immediately before the BASE variable.
        legacy_header = r"(?:# (?:Library dependencies of NRE tools|Base libs required by all tools)\n)*"
        pattern = (
            rf"{legacy_header}{re.escape(base_var_name)} = \[.*?"
            rf"{re.escape(libs_var_name)} = {re.escape(base_var_name)} \+ select\(\{{.*?\}}\)\s*\n"
        )

        if re.search(pattern, content, re.DOTALL):
            new_content = re.sub(pattern, new_list, content, flags=re.DOTALL)
        else:
            # Try old format pattern
            old_pattern = rf"{re.escape(libs_var_name)} = \[.*?\](?:\s*\+\s*select\s*\(\{{.*?\}}\))*"
            if re.search(old_pattern, content, re.DOTALL):
                new_content = re.sub(old_pattern, new_list, content, flags=re.DOTALL)
            else:
                print(f"Could not find {libs_var_name} in {build_file}")
                return False
    else:
        # Standard handling for other variables
        pattern = rf"{re.escape(libs_var_name)} = (?:\[.*?\](?:\s*\+\s*select\s*\(\{{.*?\}}\))*)"

        if re.search(pattern, content, re.DOTALL):
            new_content = re.sub(pattern, new_list, content, flags=re.DOTALL)
        else:
            # Variable doesn't exist, add it after NRE_TOOLS_PIP_PKGS
            nre_tools_pip_pattern = r"(NRE_TOOLS_PIP_PKGS = select\(\{[^}]+\}\)(?:\s*\+\s*select\s*\(\{[^}]+\}\))*)"

            match = re.search(nre_tools_pip_pattern, content, re.DOTALL)
            if match:
                insert_pos = match.end()
                new_content = content[:insert_pos] + "\n\n" + new_list + content[insert_pos:]
            else:
                print("Could not find NRE_TOOLS_PIP_PKGS in BUILD.bazel")
                return False

    if new_content == content:
        print(f"{libs_var_name} is up to date in {build_file}")
        return True

    # Write back the updated content
    with open(build_file, "w") as f:
        f.write(new_content)

    print(f"Updated {libs_var_name} in {build_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update library dependency lists in BUILD.bazel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  bazel run //internal/scripts/pycena/runtime:update_libs -- //:run
  bazel run //internal/scripts/pycena/runtime:update_libs -- //apps:nre_tools
        """,
    )

    parser.add_argument("config", help="Config name to use for updating dependencies (e.g., //:run, //apps:nre_tools)")

    args = parser.parse_args()

    # Validate config
    if args.config not in CONFIG:
        print(f"Error: Unknown config '{args.config}'")
        print(f"Available configs: {', '.join(CONFIG.keys())}")
        return 1

    success = update_build_file(args.config)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
