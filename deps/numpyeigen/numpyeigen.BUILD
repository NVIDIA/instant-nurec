# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["@nre_repo//libs/ray_utils:__pkg__"]) 

SRCS = [
    "src/npe_typedefs.cpp",
]

HDRS = [
    "src/npe.h",
    "src/npe_dense_array.h",
    "src/npe_dtype.h",
    "src/npe_sparse_array.h",
    "src/npe_typedefs.h",
    "src/npe_utils.h",
]

cc_library(
    name = "npe",
    srcs = SRCS,
    hdrs = HDRS,
    includes = ["src"],
    deps = [
        "@eigen",
        "@numpyeigen_pybind11//:pybind11",
        "@nre_pip_deps//numpy:numpy_headers",
    ],
)

py_binary(
    name = "codegen_function",
    srcs = ["src/codegen_function.py"],
)

py_binary(
    name = "codegen_module",
    srcs = ["src/codegen_module.py"],
)
