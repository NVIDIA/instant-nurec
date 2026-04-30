# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

load("@rules_cc//cc:defs.bzl", "cc_binary")

exports_files([
    "src/obfuscation/obfuscation.cpp",
    "src/obfuscation/obfuscation.h",
])

cc_binary(
    name = "obfuscator_cc",
    srcs = [
        "src/obfuscation/obfuscation.cpp",
        "src/obfuscation/obfuscation.h",
    ],
    visibility = ["@nre_repo//libs/nrend:__pkg__"],
    linkopts = [
        "-Wl,--no-undefined",
        "-Wl,--as-needed",
        "-lstdc++fs",
    ],
)
