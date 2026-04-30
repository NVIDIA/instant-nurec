# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Bazel macros to generate targets for individual SQA test cases."""

# nre:allow-direct-py-test - uses py_test directly due to dynamic test generation with custom args
load("@aspect_rules_py//py:defs.bzl", "py_test")

def test_plan_targets(lite_test_cases, srcs, deps, **kwargs):
    """Generate test targets for SQA test plan cases. For each case, generate a runfiles-based and a Docker-based test.

    Args:
        lite_test_cases: List of test case dictionaries with 'name', 'resources', and 'executable' keys
        srcs: Source files for the test targets
        deps: Dependencies for the test targets
        **kwargs: Additional arguments passed to py_test
    """
    for test_case in lite_test_cases:
        # Build tags list based on test case properties
        tags = ["manual"]

        # For documentation purposes and a bit too wide for now since only a subset of test cases use multiple GPUs.
        # To revisit once it will have a functional impact, ex. with Jira NRE-3033.
        tags.append("multi-gpu")
        if not test_case.get("parallel_execution", True):
            tags.append("exclusive")

        # Base configuration shared by both types of tests
        base_config = {
            "args": ["--test-identifiers", test_case["name"]],
            "deps": deps,
            "size": "large",
            "srcs": srcs,
            "tags": tags,
        }
        base_config.update(kwargs)

        # Runfiles-based test, benefits from Bazel caching and suitable for pre-merge CI checks
        runfiles_config = dict(base_config)
        runfiles_config.update({
            # Runfiles tests use gRPC ports 9000+test_id
            "args": base_config["args"] + ["--runfiles", "--grpc-port-base", "9000"],
            "data": test_case.get("resources", []) + [test_case["executable"]],
            # Required by the scripts, some download models from NGC at runtime and this won't be functional yet.
            # To revisit once we want to introduce such tests.
            "env": {"NGC_API_KEY": "dummy"},
            "name": "sqa_test--" + test_case["name"],
        })
        py_test(**runfiles_config)

        # Docker-based test, requires Docker image to be downloaded and does not benefit from Bazel caching,
        # suitable for periodic CI checks
        docker_config = dict(base_config)
        docker_config.update({
            # Docker tests use gRPC ports 8000+test_id (matching 'bazel run' without explicit parameter)
            "args": base_config["args"] + ["--grpc-port-base", "8000"],
            "data": test_case.get("resources", []),
            "name": "sqa_docker_test--" + test_case["name"],
            "tags": base_config["tags"] + ["external"],  # Mark "external" due to Docker dependency, no caching
        })
        py_test(**docker_config)
