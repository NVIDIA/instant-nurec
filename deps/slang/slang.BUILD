# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Platform-agnostic slang targets imported from external repo
# (supported by both x86_64 and aarch64 builds, aliased in BUILD.bazel)

cc_library(
    name = 'slang_headers',
    includes = ['include'],
    hdrs = glob(['include/*.h']),
)

cc_import(
    name = 'lib_slang_compiler',
    shared_library = "lib/libslang-compiler.so.0.2025.23.2",
)

cc_import(
    name = 'lib_slang_rtc',
    shared_library = "lib/libslang-rt.so.0.2025.23.2",
)

filegroup(
    name = 'slangc',
    srcs = ['bin/slangc'],
)
