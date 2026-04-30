# Copyright (c) 2023-2026 NVIDIA CORPORATION.  All rights reserved.

""" Wrap py_test with a common pytest wrapper """

# nre:allow-direct-py-test - this is the pytest_test wrapper implementation
load("@aspect_rules_py//py:defs.bzl", "py_test")
load("@gpu_info//:defs.bzl", "GPU_COUNT")
load("@mypy_integration//:mypy.bzl", "mypy_test")

def _compute_test_name_hash(name):
    """Compute a deterministic hash from test name.

    Args:
        name: The test target name

    Returns:
        A positive integer hash value
    """

    # Use Starlark's built-in hash function
    h = hash(name)

    # hash() can return negative values, so we need to make it positive
    if h < 0:
        h = -h

    return h

def pytest_test(name, srcs, mypy = True, deps = [], args = [], tags = [], env = {}, **kwargs):
    """Call pytest using a common wrapper script and register mypy test (optionally, default=True).

    GPU Load Balancing:
    On multi-GPU systems, a specific GPU is exposed to each test via CUDA_VISIBLE_DEVICES based on a
    hash of the test name. The number of GPUs is auto-detected at Bazel fetch time using 'nvidia-smi'.
    This helps spread single-GPU tests across multiple GPUs for better utilization.

    Args:
        name: Name of the test target
        srcs: Source files for the test
        mypy: Whether to run mypy type checking (default True)
        deps: Dependencies for the test
        args: Additional arguments to pass to pytest
        tags: Tags for the test. Use "multi-gpu" tag to prevent GPU assignment
              (for tests that natively use multiple GPUs)
        env: Additional environment variables for the test
        **kwargs: Additional arguments passed to py_test
    """

    # Compute hash from test name for GPU load balancing
    test_name_hash = _compute_test_name_hash(name)

    # Build environment with GPU load balancing info
    # The pytest plugin will use these to compute CUDA_VISIBLE_DEVICES at runtime
    test_env = dict(env)
    test_env["TEST_NAME_HASH"] = str(test_name_hash)
    test_env["TEST_TOTAL_GPUS"] = str(GPU_COUNT)
    if "multi-gpu" in tags:
        test_env["TEST_IS_MULTI_GPU"] = "1"

    py_test(
        name = name,
        srcs = srcs,
        pytest_main = "//bazel/pytest:pytest_wrapper",
        args = [
            "--capture=no",
            # Load GPU selection plugin for load balancing across GPUs
            "-p",
            "bazel.pytest.gpu_selection_plugin",
        ] + args + ["$(location :%s)" % x for x in srcs],
        deps = deps + [
            "//bazel/pytest:pytest_wrapper",
            "//bazel/pytest:gpu_selection_plugin",
        ],
        tags = tags,
        env = test_env,
        **kwargs
    )

    if mypy:
        mypy_test(
            name = name + "_mypy",
            timeout = "long",
            deps = [
                ":" + name,
            ],
        )
