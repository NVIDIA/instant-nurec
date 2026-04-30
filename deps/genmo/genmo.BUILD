# Copyright (c) 2025-2026 NVIDIA CORPORATION.  All rights reserved.

load("@aspect_rules_py//py:defs.bzl", "py_library")
load("@nre_pip_deps//:requirements.bzl", pip_requirement = "requirement")
load("@nre_pip_deps_internal//:requirements.bzl", pip_requirement_internal = "requirement")

genrule(
    name = "gen_nvhuman_layer_init",
    outs = ["hmr4d/utils/body_model/nvhuman_layer/__init__.py"],
    cmd = "echo 'from .nvhuman import NVHumanLayer' > $@",
)

py_library(
    name = "nvhuman_layer",
    srcs = glob(["hmr4d/utils/body_model/nvhuman_layer/*.py"]) + [":gen_nvhuman_layer_init"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        pip_requirement("torch"),
        pip_requirement("numpy"),
        pip_requirement("scipy"),
    ],
)

py_library(
    name = "hmr4d",
    srcs = glob(
        ["hmr4d/**/*.py"],
        exclude = [
            "hmr4d/utils/body_model/nvhuman_layer/*.py",
        ],
    ),
    data = glob([
        "hmr4d/**/*.yaml",
        "hmr4d/**/*.json",
        "hmr4d/**/*.npz",
    ]),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":nvhuman_layer",
        pip_requirement("torch"),
        pip_requirement("numpy"),
        pip_requirement("opencv-python"),
        pip_requirement("hydra-core"),
        pip_requirement("tqdm"),
        pip_requirement_internal("av"),
        pip_requirement_internal("smplx"),
        pip_requirement_internal("hydra-zen"),
        pip_requirement_internal("colorlog"),
        pip_requirement_internal("lapx"),
        pip_requirement_internal("ultralytics"),
        pip_requirement_internal("ffmpeg-python"),
        pip_requirement_internal("yacs"),
    ],
)

py_library(
    name = "motiondiff",
    srcs = glob(
        ["motiondiff/**/*.py"],
        exclude = [
            "motiondiff/**/test_*.py",
            "motiondiff/**/*_test.py",
        ],
    ),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        pip_requirement("torch"),
        pip_requirement("numpy"),
        pip_requirement_internal("pytorch3d"),
    ],
)

