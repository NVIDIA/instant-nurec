# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import unittest

from pathlib import Path

from internal.sqa.test_cases.artifacts import ArtifactsConfig, sqa_test_artifacts
from internal.sqa.test_cases.commands import Command, CommandGroup
from internal.sqa.test_cases.test_cases import DatasetConfig, TestCaseConfig, generate_test_cases, parse_yaml_test_entry


class TestCasesIntegrityTest(unittest.TestCase):
    """Test to verify the integrity of the test_plan.yaml file."""

    def setUp(self):
        """Set up the test by generating the test cases from test_plan.yaml."""
        self.test_cases = generate_test_cases(
            TestCaseConfig(
                results_base="test",
                dataset_config=DatasetConfig(
                    local_path=Path("test"),
                ),
                artifacts_config=ArtifactsConfig(
                    local_path=Path("test"),
                ),
                force_obfuscation=None,
                grpc_port_base=0,
            )
        )

    def test_config_files_referenced_in_yaml(self):
        """Test that all config files referenced in the test_plan.yaml file are valid references."""
        configs = [
            test_case.train_val_config.config for test_case in self.test_cases if test_case.train_val_config is not None
        ]
        self.assertTrue(
            all(os.path.exists(config) for config in configs),
            "Some config files referenced in the test_cases.yaml file do not exist",
        )

    def test_scripts_referenced_in_yaml(self):
        """Test that all scripts referenced in the run_commands section are valid references."""
        for test_case in self.test_cases:
            for command in test_case.commands:
                if isinstance(command, CommandGroup):
                    for command in command.commands:
                        self.assertTrue(
                            os.path.exists(command.script),
                            f"Script '{command.script}' referenced in test case '{test_case.name}' does not exist",
                        )
                elif isinstance(command, Command):
                    self.assertTrue(
                        os.path.exists(command.script),
                        f"Script '{command.script}' referenced in test case '{test_case.name}' does not exist",
                    )
                else:
                    raise ValueError(f"Test case '{test_case.name}' has an invalid command type: {type(command)}")

    def test_lite_tests_use_bazel_datasets(self):
        """Test that lite tests use Bazel datasets, except for test types that don't require datasets."""
        for test_case in self.test_cases:
            if test_case.mode == "lite" and test_case.dataset is not None and test_case.dataset.bazel_target is None:
                self.fail(
                    f"Test '{test_case.name}' is a lite test with a dataset and thus it must use a Bazel dataset. "
                    f"Dataset '{test_case.dataset.name}' has no bazel_target defined."
                )

    def test_lite_tests_use_bazel_artifacts(self):
        """Test that lite tests with pre-built artifacts use Bazel artifacts."""
        artifacts = sqa_test_artifacts(ArtifactsConfig(local_path=Path("test")))
        for test_case in self.test_cases:
            if test_case.mode == "lite" and test_case.artifact_source and test_case.artifact_source != "train_val":
                # This is a lite test using pre-built artifacts - check if it has a Bazel target
                artifact = artifacts.get(test_case.artifact_source)
                if artifact is None:
                    self.fail(
                        f"Test '{test_case.name}' references unknown artifact_source '{test_case.artifact_source}'."
                    )
                if artifact.bazel_target is None:
                    self.fail(
                        f"Test '{test_case.name}' is a lite test and must use a Bazel artifact. "
                        f"Artifact '{artifact.name}' has no bazel_target defined."
                    )

    def test_parameter_order_consistency(self):
        """Test that parameters appear in consistent order across test cases."""
        # Expected parameter order
        expected_order = [
            "mode",
            "obfuscation",
            "dataset",
            "config",
            "artifact_source",
            "test_control_actor",
            "edit_assets_scenario",
            "use_gsplat",
            "script_filename",
        ]

        for test_case in self.test_cases:
            with self.subTest(test_case=test_case.name):
                _, params = parse_yaml_test_entry(test_case.name)

                # Skip if no parameters or only one parameter (no ordering to validate)
                if len(params) <= 1:
                    continue

                # Get the parameter names in order
                param_names = list(params.keys())

                # Check that parameters appear in the expected order
                # (not all parameters need to be present, but those that are should be in order)
                last_expected_index = -1
                for param_name in param_names:
                    # Note: Unknown parameters are checked in test_no_unexpected_parameters
                    if param_name in expected_order:
                        current_expected_index = expected_order.index(param_name)
                        self.assertGreater(
                            current_expected_index,
                            last_expected_index,
                            f"Parameter '{param_name}' appears out of order in test case '{test_case.name}'. "
                            f"Expected order: {expected_order}",
                        )
                        last_expected_index = current_expected_index

    def test_no_unexpected_parameters(self):
        """Test that test case names only contain expected parameters."""
        # Define all known/expected parameters
        expected_parameters = {
            "mode",
            "obfuscation",
            "dataset",
            "config",
            "artifact_source",
            "test_control_actor",
            "edit_assets_scenario",
            "use_gsplat",
            "script_filename",
        }

        for test_case in self.test_cases:
            with self.subTest(test_case=test_case.name):
                _, params = parse_yaml_test_entry(test_case.name)
                param_names = list(params.keys())

                # Check that all parameters are expected
                for param_name in param_names:
                    self.assertIn(
                        param_name,
                        expected_parameters,
                        f"Unexpected parameter '{param_name}' found in test case '{test_case.name}'. "
                        f"Expected parameters: {sorted(expected_parameters)}",
                    )

    def test_name_parameter_consistency(self):
        """Test that parameter values in the name match the TestCase object fields."""
        for test_case in self.test_cases:
            with self.subTest(test_case=test_case.name):
                test_type, params = parse_yaml_test_entry(test_case.name)

                name_mode = params.get("mode")
                if name_mode:
                    self.assertEqual(
                        name_mode,
                        test_case.mode,
                        f"Mode in name ('{name_mode}') doesn't match TestCase.mode ('{test_case.mode}') "
                        f"for test case '{test_case.name}'",
                    )

                # Extract obfuscation parameter from name and compare with TestCase.obfuscation
                name_obfuscation = params.get("obfuscation")
                if name_obfuscation:
                    self.assertEqual(
                        name_obfuscation,
                        test_case.obfuscation,
                        f"Obfuscation in name ('{name_obfuscation}') doesn't match TestCase.obfuscation "
                        f"('{test_case.obfuscation}') for test case '{test_case.name}'",
                    )

                # Extract dataset parameter from name and compare with TestCase.dataset.name
                name_dataset = params.get("dataset")
                if name_dataset:
                    if test_case.dataset is not None:
                        self.assertEqual(
                            name_dataset,
                            test_case.dataset.name,
                            f"Dataset in name ('{name_dataset}') doesn't match TestCase.dataset.name "
                            f"('{test_case.dataset.name}') for test case '{test_case.name}'",
                        )
                    else:
                        self.fail(
                            f"Dataset parameter '{name_dataset}' found in name but TestCase.dataset is None "
                            f"for test case '{test_case.name}'"
                        )

                # Extract config parameter from name and compare with TestCase.train_val_config
                name_config = params.get("config")
                if name_config:
                    if test_case.train_val_config is not None:
                        # Extract just the filename from the config path for comparison
                        config_filename = test_case.train_val_config.name
                        self.assertEqual(
                            name_config,
                            config_filename,
                            f"Config in name ('{name_config}') doesn't match TestCase.train_val_config.name "
                            f"('{config_filename}') for test case '{test_case.name}'",
                        )
                    else:
                        self.fail(
                            f"Config parameter '{name_config}' found in name but TestCase.train_val_config is None "
                            f"for test case '{test_case.name}'"
                        )
                elif test_case.train_val_config is not None:
                    self.fail(
                        f"TestCase.train_val_config is not None but no config parameter found in name "
                        f"for test case '{test_case.name}'"
                    )

                # Extract artifact_source parameter from name and compare with TestCase.artifact_source
                name_artifact_source = params.get("artifact_source")
                if name_artifact_source:
                    self.assertEqual(
                        name_artifact_source,
                        test_case.artifact_source,
                        f"Artifact source in name ('{name_artifact_source}') doesn't match TestCase.artifact_source "
                        f"('{test_case.artifact_source}') for test case '{test_case.name}'",
                    )
                elif test_case.artifact_source:
                    self.fail(
                        f"TestCase.artifact_source is not empty but no artifact_source parameter found in name "
                        f"for test case '{test_case.name}'"
                    )

                # Extract test_control_actor parameter from name and compare with TestCase.test_control_actor
                name_test_control_actor = params.get("test_control_actor")
                if name_test_control_actor:
                    self.assertEqual(
                        name_test_control_actor,
                        test_case.test_control_actor,
                        f"Test control actor in name ('{name_test_control_actor}') doesn't match TestCase.test_control_actor "
                        f"('{test_case.test_control_actor}') for test case '{test_case.name}'",
                    )
                elif test_case.test_control_actor:
                    self.fail(
                        f"TestCase.test_control_actor is not empty but no test_control_actor parameter found in name "
                        f"for test case '{test_case.name}'"
                    )

                name_edit_assets_scenario = params.get("edit_assets_scenario")
                if name_edit_assets_scenario:
                    self.assertEqual(
                        name_edit_assets_scenario,
                        test_case.edit_assets_scenario,
                        f"Edit-assets scenario in name ('{name_edit_assets_scenario}') doesn't match "
                        f"TestCase.edit_assets_scenario ('{test_case.edit_assets_scenario}') "
                        f"for test case '{test_case.name}'",
                    )
                elif test_case.edit_assets_scenario:
                    self.fail(
                        f"TestCase.edit_assets_scenario is not empty but no edit_assets_scenario parameter found "
                        f"in name for test case '{test_case.name}'"
                    )

                # Extract script_filename parameter from name and validate based on test type
                script_filename = params.get("script_filename")

                if test_type == "run_example":
                    # For run_example test type, script_filename must be present
                    if not script_filename:
                        self.fail(
                            f"Test case '{test_case.name}' is of type 'run_example' but has no script_filename parameter"
                        )
                else:
                    # For other test types, script_filename must not be present
                    if script_filename:
                        self.fail(
                            f"Test case '{test_case.name}' is of type '{test_type}' but has script_filename parameter. "
                            f"script_filename is only valid for 'run_example' test type"
                        )

    def test_render_tests_single_camera(self):
        """Test that render tests (nre_image_render, nre_render_grpc) have exactly 1 camera ID."""
        for test_case in self.test_cases:
            if "nre_image_render" in test_case.name or "nre_render_grpc" in test_case.name:
                with self.subTest(test_case=test_case.name):
                    num_cameras = len(test_case.dataset.sensors.camera_ids)
                    self.assertEqual(
                        num_cameras,
                        1,
                        f"Render test '{test_case.name}' has {num_cameras} camera IDs, but must have exactly 1. "
                        f"Camera IDs: {test_case.dataset.sensors.camera_ids}",
                    )

    def test_ci_runtime_limits_require_no_parallel_execution(self):
        """Test that tests with ci_runtime_limit_* must have parallel_execution=false.

        Runtime limits are only meaningful when the test has exclusive access to the machine, otherwise concurrent tests
        could affect timing measurements.
        """
        for test_case in self.test_cases:
            if test_case.ci_runtime_limits:
                with self.subTest(test_case=test_case.name):
                    self.assertFalse(
                        test_case.parallel_execution,
                        f"Test '{test_case.name}' has ci_runtime_limit_* defined but allows parallel execution. "
                        f"Tests with runtime limits must have 'parallel_execution: false' to ensure accurate timing measurements.",
                    )

    def test_parallel_execution_false_only_for_lite_tests(self):
        """Test that 'parallel_execution: false' is only used for lite tests.

        The 'parallel_execution' flag controls Bazel test execution parallelism. Full tests are never run through
        'bazel test', so the flag would be meaningless.
        """
        for test_case in self.test_cases:
            if not test_case.parallel_execution and test_case.mode != "lite":
                with self.subTest(test_case=test_case.name):
                    self.fail(
                        f"Test '{test_case.name}' has 'parallel_execution: false' but is not a lite test (mode={test_case.mode}). "
                        f"The 'parallel_execution' flag only affects Bazel test execution and is meaningless for full tests."
                    )

    def test_description_markdown_links_are_valid(self):
        """Test that relative markdown links in description and manual_validation fields point to existing files and anchors.

        Description and manual validation links are relative to internal/sqa/ (the README.md location).
        External URLs (containing '://') are skipped.
        For .md files that are reachable, section anchors are also validated.
        """
        import re

        # Links in descriptions and manual validation fields are resolved relative to internal/sqa/ (the README location)
        sqa_dir = Path(__file__).parent.parent
        readme_path = sqa_dir / "README.md"

        link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
        heading_pattern = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)

        def heading_to_anchor(heading_text: str) -> str:
            """Convert a markdown heading to a GitHub-style anchor slug."""
            # Strip inline code and common markdown formatting characters
            text = re.sub(r"`[^`]*`", lambda m: m.group(0)[1:-1], heading_text)
            text = re.sub(r"[*_\[\]()!]", "", text)
            text = text.lower()
            # Keep only alphanumerics, spaces, and hyphens; collapse spaces to a single hyphen
            text = re.sub(r"[^\w\s-]", "", text)
            text = re.sub(r"\s+", "-", text.strip())
            return text

        def get_anchors(filepath: Path) -> set[str]:
            """Return the set of heading anchors defined in a markdown file."""
            with open(filepath, "r") as f:
                content = f.read()
            return {heading_to_anchor(m.group(1).strip()) for m in heading_pattern.finditer(content)}

        for test_case in self.test_cases:
            if not test_case.description and not test_case.manual_validation:
                continue
            with self.subTest(test_case=test_case.name):
                description_links = link_pattern.findall(test_case.description or "")
                manual_validation_links = link_pattern.findall(test_case.manual_validation or "")
                for link_text, link_target in description_links + manual_validation_links:
                    link_source = (
                        "description" if (link_text, link_target) in description_links else "manual_validation"
                    )

                    # Skip external URLs
                    if "://" in link_target:
                        continue

                    # Split into file path and optional anchor
                    if "#" in link_target:
                        file_part, anchor = link_target.split("#", 1)
                    else:
                        file_part, anchor = link_target, ""

                    # Resolve the target file (empty file_part means the README itself)
                    if file_part:
                        target_file = sqa_dir / file_part
                        self.assertTrue(
                            target_file.exists(),
                            f"Broken link '[{link_text}]({link_target})' in {link_source} of '{test_case.name}': "
                            f"'{file_part}' not found. Either the path is wrong, or the file is not reachable "
                            f"in the test sandbox — links must point to .md files under internal/sqa/ that are "
                            f"declared as data deps in BUILD.bazel (resolved path: '{target_file}')",
                        )
                    else:
                        target_file = readme_path
                        self.assertTrue(
                            readme_path.exists(),
                            f"README.md not found at '{readme_path}' — ensure //internal/sqa:README.md is "
                            f"listed in the data deps in BUILD.bazel",
                        )

                    # Validate the anchor against headings of the target file.
                    # Anchors on non-.md files are never valid; no suffix guard — fail explicitly.
                    if anchor:
                        self.assertEqual(
                            target_file.suffix,
                            ".md",
                            f"Anchor '#{anchor}' in '[{link_text}]({link_target})' in {link_source} of '{test_case.name}': "
                            f"anchors are only valid for .md files, not '{target_file.suffix}' files",
                        )
                        available_anchors = get_anchors(target_file)
                        self.assertIn(
                            anchor,
                            available_anchors,
                            f"Broken anchor '[{link_text}]({link_target})' in {link_source} of '{test_case.name}': "
                            f"'#{anchor}' not found in '{target_file.relative_to(sqa_dir.parent.parent)}'. "
                            f"Available anchors: {sorted(available_anchors)}",
                        )


if __name__ == "__main__":
    unittest.main()
