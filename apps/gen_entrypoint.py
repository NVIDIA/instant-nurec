# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import yaml

from jinja2 import Environment, FileSystemLoader


def main():
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate entrypoint script from a template.")
    parser.add_argument("--config", required=True, help="Path to the tools YAML config file.")
    parser.add_argument("--template", required=True, help="Path to the entrypoint template file.")
    parser.add_argument("--output", required=True, help="Path to the output entrypoint script.")
    parser.add_argument("--enabled-tools", nargs="*", default=[], help="List of enabled conditional tools by name.")
    args = parser.parse_args()

    logging.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    all_tools = config.get("tools", [])
    enabled_tools_set = set(args.enabled_tools)
    tool_names = [tool["name"] for tool in all_tools]
    logging.info(f"Tools present = {', '.join(tool_names) if tool_names else 'none'}")
    logging.info(f"Tools enabled = {', '.join(enabled_tools_set) if enabled_tools_set else 'none'}")

    final_tools = []
    for tool in all_tools:
        if tool["name"] in enabled_tools_set:
            final_tools.append(tool)

    # The template path from bazel is relative to the execroot, so we can use it to find the template dir
    template_dir = os.path.dirname(args.template)
    template_name = os.path.basename(args.template)
    env = Environment(loader=FileSystemLoader(template_dir if template_dir else "."))
    template = env.get_template(template_name)

    output_content = template.render(tools=final_tools)

    tool_names = [tool["name"] for tool in final_tools]
    logging.info(f"Writing tools: {', '.join(tool_names) if tool_names else 'none'} to entrypoint to {args.output}")
    with open(args.output, "w") as f:
        f.write(output_content)

    logging.info("Entrypoint generation completed successfully")


if __name__ == "__main__":
    main()
