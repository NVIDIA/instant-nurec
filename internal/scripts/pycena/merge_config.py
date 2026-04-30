# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import logging
import sys

from typing import Any, Dict, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file and return parsed content."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_enabled_tools(tools_config: Dict[str, Any], enabled_tool_names: List[str]) -> List[Dict[str, Any]]:
    """Get list of enabled tools based on tool names."""
    all_tools = tools_config.get("tools", [])
    enabled_tools_set = set(enabled_tool_names)

    enabled_tools = []
    for tool in all_tools:
        if tool["name"] in enabled_tools_set:
            enabled_tools.append(tool)

    return enabled_tools


def merge_obfuscation_modules(
    base_modules: List[Dict[str, Any]], tool_modules: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge obfuscation modules, avoiding duplicates based on mod name."""
    seen_mods = set()
    result = []

    # Add base modules first
    for module in base_modules:
        mod_name = module.get("mod")
        if mod_name and mod_name not in seen_mods:
            seen_mods.add(mod_name)
            result.append(module)

    # Add tool modules, avoiding duplicates
    for module in tool_modules:
        mod_name = module.get("mod")
        if mod_name and mod_name not in seen_mods:
            seen_mods.add(mod_name)
            result.append(module)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Merge obfuscation modules from tools into base config.")
    parser.add_argument("--base-config", required=True, help="Path to base nre_tools.yaml config")
    parser.add_argument("--tools-config", required=True, help="Path to tools.yaml config")
    parser.add_argument("--output", required=True, help="Path to output merged config")
    parser.add_argument("--enabled-tools", nargs="*", default=[], help="List of enabled tools by name")
    parser.add_argument("--gen-entrypoint-location", help="Location of the gen_entrypoint script to update in config")

    args = parser.parse_args()

    try:
        # Load configurations
        base_config = load_yaml(args.base_config)
        tools_config = load_yaml(args.tools_config)

        # Get enabled tools
        enabled_tools = get_enabled_tools(tools_config, args.enabled_tools)

        logging.info(f"Enabled tools: {[tool['name'] for tool in enabled_tools]}")

        # Collect obfuscation modules from enabled tools
        tool_obfuscation_modules = []
        for tool in enabled_tools:
            tool_modules = tool.get("obfuscation_modules", [])
            tool_obfuscation_modules.extend(tool_modules)
            if tool_modules:
                logging.info(f"Tool '{tool['name']}' contributes {len(tool_modules)} obfuscation modules")

        # Merge with base obfuscation modules
        base_modules = base_config.get("obfuscation_modules", [])
        merged_modules = merge_obfuscation_modules(base_modules, tool_obfuscation_modules)

        logging.info(
            f"Base config had {len(base_modules)} modules, "
            f"tools contribute {len(tool_obfuscation_modules)} modules, "
            f"final config has {len(merged_modules)} modules"
        )

        # Update config with merged modules
        base_config["obfuscation_modules"] = merged_modules

        # Update script path if gen_entrypoint_location is provided
        if args.gen_entrypoint_location:
            base_config["script"] = args.gen_entrypoint_location
            logging.info(f"Updated script path to: {args.gen_entrypoint_location}")

        # Write output
        with open(args.output, "w") as f:
            yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)

        logging.info(f"Merged config written to {args.output}")

    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
