# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Update the LITE_TEST_CASES variable in defs.bzl with dynamically generated lite test cases,
# and update the command documentation section in README.md.
# In verify mode, check that both are in sync.

import os
import re
import sys

from pathlib import Path

import click

from internal.sqa.test_cases.artifacts import Artifacts, ArtifactsConfig, sqa_test_artifacts
from internal.sqa.test_cases.commands import Command, CommandGroup
from internal.sqa.test_cases.datasets import Dataset, DatasetConfig, sqa_test_datasets
from internal.sqa.test_cases.test_cases import TestCase, TestCaseConfig, generate_test_cases


def get_dataset_bazel_target(dataset):
    """Get the Bazel target for a dataset.

    Args:
        dataset: Dataset object or None

    Returns:
        str: Bazel target for the dataset, or None if not available or dataset is None
    """
    if dataset is not None and dataset.bazel_target is not None:
        return dataset.bazel_target["target"]
    return None


def get_artifacts_bazel_target(artifact_source, all_artifacts):
    """Get the Bazel target for an artifact source.

    Args:
        artifact_source: Artifact source name from test case
        all_artifacts: Dictionary of available artifacts

    Returns:
        str: Bazel target for the artifacts, or None if not available
    """
    if not artifact_source or artifact_source == "train_val":
        return None

    if artifact_source in all_artifacts:
        artifacts_obj = all_artifacts[artifact_source]
        if artifacts_obj.bazel_target is not None:
            return artifacts_obj.bazel_target["target"]
    return None


def get_file_path(workspace_relative_path: str, runfiles_relative_path: str) -> str:
    """Get the path to a file, handling both bazel run and bazel test contexts.

    Args:
        workspace_relative_path: Path relative to workspace root (used for bazel run)
        runfiles_relative_path: Path relative to script directory in runfiles (used for bazel test)

    Returns:
        str: Resolved path to the file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if "runfiles" in script_dir:
        workspace_dir = os.getenv("BUILD_WORKSPACE_DIRECTORY")
        if workspace_dir:
            # Running via bazel run with BUILD_WORKSPACE_DIRECTORY set
            return os.path.join(workspace_dir, workspace_relative_path)
        else:
            # Running via bazel test - look for the data file in runfiles
            return os.path.join(script_dir, runfiles_relative_path)
    else:
        # Running directly (not via bazel)
        return os.path.join(script_dir, runfiles_relative_path)


def get_all_command_executables(cmd):
    """Extract all bazel executables from a command, handling CommandGroups."""
    if hasattr(cmd, "commands") and cmd.commands:
        # It's a CommandGroup, get executables from all commands in the group
        return [sub_cmd.bazel_executable for sub_cmd in cmd.commands]
    else:
        # It's a regular Command
        return [cmd.bazel_executable]


def format_command_for_doc(cmd: Command, test_name: str, indent: str = "") -> str:
    """Format a single command for documentation with line continuation.

    Replaces variable paths with placeholders suitable for documentation.
    Focuses on Docker execution mode. Long commands are split across multiple
    lines using bash line continuation (backslash).

    Args:
        cmd: Command object to format
        test_name: Name of the test case (replaced with <TEST_NAME> placeholder)
        indent: Indentation prefix for continuation lines

    Returns:
        str: Formatted command line string (may be multi-line)
    """
    script = cmd.script
    args = list(cmd.args)

    # Replace placeholders for Docker mode documentation
    formatted_args = []
    for arg in args:
        # Replace $(EXTRA_PARAMS) with Docker tag placeholder
        if arg == "$(EXTRA_PARAMS)":
            formatted_args.append("--tag <DOCKER_TAG>")
        # Replace $(NRE_OUTPUT_DIR) with placeholder (this is the random subdir created by training)
        elif "$(NRE_OUTPUT_DIR)" in arg:
            formatted_args.append(arg.replace("$(NRE_OUTPUT_DIR)", "<TRAIN_OUTPUT_SUBDIR>"))
        # Replace $(GIF_TOOL) with placeholder
        elif "$(GIF_TOOL)" in arg:
            formatted_args.append(arg.replace("$(GIF_TOOL)", "<GIF_TOOL_PATH>"))
        else:
            # Replace test name with placeholder in paths
            formatted_args.append(arg.replace(test_name, "<TEST_NAME>"))

    # Group arguments: each --flag with its value(s) until the next --flag
    arg_groups: list[str] = []
    current_group: list[str] = []
    for arg in formatted_args:
        if arg.startswith("--") and current_group:
            arg_groups.append(" ".join(current_group))
            current_group = [arg]
        else:
            current_group.append(arg)
    if current_group:
        arg_groups.append(" ".join(current_group))

    # Build command with line continuation
    if not arg_groups:
        return f"{indent}./{script}"

    # First line: script with line continuation
    lines = [f"{indent}./{script} \\"]
    continuation_indent = indent + "  "

    for i, group in enumerate(arg_groups):
        is_last = i == len(arg_groups) - 1
        if is_last:
            lines.append(f"{continuation_indent}{group}")
        else:
            lines.append(f"{continuation_indent}{group} \\")

    return "\n".join(lines)


def format_command_group_for_doc(
    cmd_group: CommandGroup, test_name: str, step_num: int, is_last_step: bool
) -> list[str]:
    """Format a command group with markdown list structure and indented bash blocks.

    Args:
        cmd_group: CommandGroup object to format
        test_name: Name of the test case (replaced with <TEST_NAME> placeholder)
        step_num: Step number for this group
        is_last_step: Whether this is the last step in the sequence

    Returns:
        list[str]: List of formatted markdown lines
    """
    lines = []

    lines.append(f"- **Step {step_num}** (parallel):")
    lines.append("")

    for i, cmd in enumerate(cmd_group.commands):
        wait_time = cmd_group.wait_before_commands_s[i] if i < len(cmd_group.wait_before_commands_s) else 0
        run_bg = cmd_group.run_in_background[i] if i < len(cmd_group.run_in_background) else False

        # Build annotation
        if run_bg:
            annotation = "Background"
        elif wait_time > 0:
            annotation = f"Foreground after {int(wait_time)}s"
        else:
            annotation = "Foreground"

        # Nested list item for parallel sub-step (2-space indent for nested list)
        lines.append(f"  - **[{annotation}]**:")

        # Code block indented under nested list item (4 spaces)
        lines.append("    ```bash")
        lines.append(format_command_for_doc(cmd, test_name, indent="    "))
        lines.append("    ```")

    lines.append("")
    return lines


def format_single_command_for_doc(
    cmd: Command, test_name: str, step_num: int, is_last_step: bool, total_steps: int
) -> list[str]:
    """Format a single command with markdown list structure and indented bash block.

    Args:
        cmd: Command object to format
        test_name: Name of the test case (replaced with <TEST_NAME> placeholder)
        step_num: Step number for this command
        is_last_step: Whether this is the last step in the sequence
        total_steps: Total number of steps in the test case

    Returns:
        list[str]: Formatted markdown lines
    """
    lines = []

    if total_steps == 1:
        lines.append("- **Single command**:")
    else:
        lines.append(f"- **Step {step_num}**:")
        lines.append("")
    # Code block indented under list item (2 spaces)
    lines.append("  ```bash")
    lines.append(format_command_for_doc(cmd, test_name, indent="  "))
    lines.append("  ```")
    lines.append("")

    return lines


def generate_command_documentation(
    test_cases: list[TestCase],
    datasets: dict[str, Dataset],
    artifacts: dict[str, Artifacts],
) -> str:
    """Generate markdown documentation for all test case command lines.

    Args:
        test_cases: List of test cases
        datasets: Dictionary of available datasets
        artifacts: Dictionary of available artifacts

    Returns:
        str: Markdown formatted documentation
    """
    doc_lines = []
    doc_lines.append("<!-- BEGIN AUTO-GENERATED TEST PLAN DOCUMENTATION -->")
    doc_lines.append("<!-- DO NOT EDIT THIS SECTION MANUALLY -->")
    doc_lines.append("<!-- Run: bazel run //internal/sqa/test_cases:sync_test_plan -->")
    doc_lines.append("")

    # Resources section
    doc_lines.append("### Resources")
    doc_lines.append("")

    def format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
        """Format a markdown table with proper column alignment."""
        # Calculate column widths (including header)
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # Build table lines
        lines = []
        # Header row
        header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(headers)]
        lines.append("| " + " | ".join(header_cells) + " |")
        # Separator row
        separator_cells = ["-" * col_widths[i] for i in range(len(headers))]
        lines.append("| " + " | ".join(separator_cells) + " |")
        # Data rows
        for row in rows:
            data_cells = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
            lines.append("| " + " | ".join(data_cells) + " |")
        return lines

    # Datasets subsection
    doc_lines.append("#### Datasets")
    doc_lines.append("")
    dataset_rows = []
    for name, ds in sorted(datasets.items()):
        local_path = str(ds.local_path).replace("<DATASET_DIR>/", "")
        if ds.bazel_target:
            source = f"`{ds.bazel_target['target']}`"
        elif ds.remote_path:
            source = f"`{ds.remote_path}`"
        else:
            source = "N/A"
        dataset_rows.append([f"`{name}`", source, f"`{local_path}`"])
    doc_lines.extend(format_table(["Name", "Source", "Local Path"], dataset_rows))
    doc_lines.append("")

    # Artifacts subsection
    doc_lines.append("#### Artifacts")
    doc_lines.append("")
    artifact_rows = []
    for name, art in sorted(artifacts.items()):
        local_path = str(art.local_path).replace("<ARTIFACTS_DIR>/", "")
        if art.bazel_target:
            source = f"`{art.bazel_target['target']}`"
        elif art.remote_path:
            source = f"`{art.remote_path}`"
        else:
            source = "N/A"
        artifact_rows.append([f"`{name}`", source, f"`{local_path}`"])
    doc_lines.extend(format_table(["Name", "Source", "Local Path"], artifact_rows))
    doc_lines.append("")

    # Split test cases by mode (lite vs full), then group by test type
    lite_cases: list[TestCase] = []
    full_cases: list[TestCase] = []
    for tc in test_cases:
        if "--mode-lite--" in tc.name:
            lite_cases.append(tc)
        else:
            full_cases.append(tc)

    def group_by_test_type(cases: list[TestCase]) -> dict[str, list[TestCase]]:
        """Group test cases by test type (everything before the first '--')."""
        test_types: dict[str, list[TestCase]] = {}
        for tc in cases:
            test_type = tc.name.split("--")[0] if "--" in tc.name else tc.name
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(tc)
        return test_types

    def append_test_cases(cases_by_type: dict[str, list[TestCase]]) -> None:
        """Append formatted test cases to doc_lines."""
        for test_type, cases in cases_by_type.items():
            doc_lines.append(f"#### {test_type}")
            doc_lines.append("")

            for tc in cases:
                doc_lines.append(f"##### Test identifier `{tc.name}`")
                doc_lines.append("")

                if tc.owner:
                    doc_lines.append(f"**Owner:** `{tc.owner}`")
                    doc_lines.append("")
                if tc.description:
                    doc_lines.append(tc.description.strip())
                    doc_lines.append("")
                if tc.manual_validation:
                    doc_lines.append("**Manual validation required:** " + tc.manual_validation)
                    doc_lines.append("")

                num_steps = len(tc.commands)
                for i, cmd_or_group in enumerate(tc.commands):
                    step_num = i + 1
                    is_last_step = i == num_steps - 1

                    if isinstance(cmd_or_group, CommandGroup):
                        doc_lines.extend(format_command_group_for_doc(cmd_or_group, tc.name, step_num, is_last_step))
                    else:
                        doc_lines.extend(
                            format_single_command_for_doc(cmd_or_group, tc.name, step_num, is_last_step, num_steps)
                        )

    # SQA Lite section
    doc_lines.append("### SQA Lite")
    doc_lines.append("")
    append_test_cases(group_by_test_type(lite_cases))

    # Full SQA section
    doc_lines.append("### Full SQA")
    doc_lines.append("")
    append_test_cases(group_by_test_type(full_cases))

    doc_lines.append("<!-- END AUTO-GENERATED TEST PLAN DOCUMENTATION -->")

    return "\n".join(doc_lines)


def update_readme_documentation(readme_path: str, documentation: str) -> tuple[str, str]:
    """Update the README.md file with the generated documentation.

    Args:
        readme_path: Path to the README.md file
        documentation: Generated documentation content

    Returns:
        tuple[str, str]: (existing_content, updated_content)
    """
    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README.md not found at {readme_path}")

    with open(readme_path, "r") as f:
        existing_content = f.read()

    # Pattern to match the auto-generated section
    pattern = (
        r"<!-- BEGIN AUTO-GENERATED TEST PLAN DOCUMENTATION -->.*?<!-- END AUTO-GENERATED TEST PLAN DOCUMENTATION -->"
    )

    if re.search(pattern, existing_content, re.DOTALL):
        # Replace existing section
        updated_content = re.sub(pattern, documentation, existing_content, flags=re.DOTALL)
    else:
        # Append new section at the end
        updated_content = existing_content.rstrip() + "\n\n" + documentation + "\n"

    return existing_content, updated_content


@click.command()
@click.option("--verify", is_flag=True, help="Verify that defs.bzl and README.md are in sync instead of updating")
def sync_test_plan(verify):
    """Update the LITE_TEST_CASES variable in defs.bzl and command documentation in README.md."""
    # Get output file paths
    bzl_file = get_file_path("internal/sqa/test_cases/defs.bzl", "defs.bzl")
    readme_file = get_file_path("internal/sqa/README.md", "../README.md")

    # Generate test cases
    config = TestCaseConfig(
        results_base="<OUTPUT_DIR>",
        dataset_config=DatasetConfig(local_path=Path("<DATASET_DIR>")),
        artifacts_config=ArtifactsConfig(local_path=Path("<ARTIFACTS_DIR>")),
        force_obfuscation=None,
        grpc_port_base=8000,
    )
    test_cases = generate_test_cases(config)

    # Load datasets and artifacts from config
    datasets = sqa_test_datasets(config.dataset_config)
    artifacts = sqa_test_artifacts(config.artifacts_config)

    # Extract lite tests with dataset targets and executable info
    lite_tests = []

    for tc in test_cases:
        if tc.mode == "lite":
            # Get artifact target if test uses artifacts
            artifacts_target = get_artifacts_bazel_target(tc.artifact_source, artifacts)

            # Get dataset target
            dataset_target = get_dataset_bazel_target(tc.dataset)
            # Use empty string for dataset target if not available (e.g., for run_example test type)
            if not dataset_target:
                dataset_target = ""

            # Get bazel executable from commands and validate they're all the same
            # Get all bazel executables from all commands (flattened)
            all_executables = []
            for cmd in tc.commands:
                all_executables.extend(get_all_command_executables(cmd))

            # Validate that all commands use the same executable
            first_executable = all_executables[0]
            for i, executable in enumerate(all_executables[1:], 1):
                if executable != first_executable:
                    print(f"Error: Test case '{tc.name}' has mixed bazel executables:")
                    print(f"  Executable 0: {first_executable}")
                    print(f"  Executable {i}: {executable}")
                    print("Mixed executables per test case are not supported yet.")
                    sys.exit(1)

            executable_target = first_executable.get("target")
            if not executable_target:
                print(f"Error: Lite test '{tc.name}' does not have a Bazel executable target")
                sys.exit(1)

            # Build resources list (datasets and artifacts)
            resources = []
            if dataset_target:
                resources.append(dataset_target)
            if artifacts_target:
                resources.append(artifacts_target)

            test_entry = {
                "name": tc.name,
                "executable_target": executable_target,
                "resources": resources,
                "parallel_execution": tc.parallel_execution,
            }

            lite_tests.append(test_entry)

    # Sort the list
    lite_tests.sort(key=lambda x: x["name"])

    # Read existing bzl file content
    if not os.path.exists(bzl_file):
        print(f"Error: {bzl_file} does not exist. Please create it first.")
        sys.exit(1)

    with open(bzl_file, "r") as f:
        bzl_existing_content = f.read()

    # Generate new LITE_TEST_CASES content
    lite_tests_content = "LITE_TEST_CASES = [\n"
    for test in lite_tests:
        # Build dict in lexicographical order: executable, name, parallel_execution, resources
        fields = []
        fields.append(f'"executable": "{test["executable_target"]}"')
        fields.append(f'"name": "{test["name"]}"')
        fields.append(f'"parallel_execution": {test["parallel_execution"]}')
        resources_list = ", ".join([f'"{r}"' for r in test["resources"]])
        fields.append(f'"resources": [{resources_list}]')

        entry = "{" + ", ".join(fields) + "}"
        lite_tests_content += f"    {entry},\n"
    lite_tests_content += "]"

    # Replace the LITE_TEST_CASES variable in the existing content
    # Use greedy matching to capture the entire list including nested brackets
    pattern = r"LITE_TEST_CASES\s*=\s*\[.*\]"

    if not re.search(pattern, bzl_existing_content, re.DOTALL):
        print(f"Error: Could not find LITE_TEST_CASES variable in {bzl_file}")
        sys.exit(1)

    bzl_updated_content = re.sub(pattern, lite_tests_content, bzl_existing_content, flags=re.DOTALL)

    # Generate command documentation for README
    command_documentation = generate_command_documentation(test_cases, datasets, artifacts)
    readme_existing_content, readme_updated_content = update_readme_documentation(readme_file, command_documentation)

    if verify:
        # Verification mode - compare generated content with existing content
        bzl_in_sync = bzl_updated_content == bzl_existing_content
        readme_in_sync = readme_updated_content == readme_existing_content

        all_in_sync = True

        if bzl_in_sync:
            print("LITE_TEST_CASES in internal/sqa/test_cases/defs.bzl is in sync")
        else:
            print("LITE_TEST_CASES in internal/sqa/test_cases/defs.bzl is out of sync!")
            all_in_sync = False

        if readme_in_sync:
            print("Command documentation in internal/sqa/README.md is in sync")
        else:
            print("Command documentation in internal/sqa/README.md is out of sync!")
            all_in_sync = False

        if all_in_sync:
            sys.exit(0)
        else:
            print("To fix this, run: bazel run //internal/sqa/test_cases:sync_test_plan")
            sys.exit(1)
    else:
        # Update mode - write the updated files
        with open(bzl_file, "w") as f:
            f.write(bzl_updated_content)
        print(f"Updated LITE_TEST_CASES in {bzl_file} successfully with {len(lite_tests)} entries")

        with open(readme_file, "w") as f:
            f.write(readme_updated_content)
        print(f"Updated command documentation in {readme_file} successfully with {len(test_cases)} test cases")


if __name__ == "__main__":
    sync_test_plan(show_default=True)
