# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

load("@rules_python//python:defs.bzl", "py_library")
load("@nre_pip_deps//:requirements.bzl", pip_requirement = "requirement")

py_library(
    name = "cosmos_predict1",
    srcs = glob(
        ["**/*.py"],
        exclude = [
            "tests/**",
            "test/**",
            "**/test_*.py",
            "**/*_test.py",
            "setup.py",
        ],
    ),
    data = glob(
        [
            "**/*.yaml",
            "**/*.yml",
            "**/*.json",
        ],
        allow_empty = True,
    ),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        pip_requirement("torch"),
        pip_requirement("einops"),
        pip_requirement("numpy"),
        pip_requirement("omegaconf"),
        pip_requirement("pillow"),
        pip_requirement("scipy"),
        pip_requirement("timm"),
        pip_requirement("transformers"),
    ],
)

