# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "pybind11",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@rules_python//python/cc:current_py_cc_libs",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)
