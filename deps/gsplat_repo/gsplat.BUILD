# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_python//python:defs.bzl", "py_library")
load("@nre_pip_deps//:requirements.bzl", pip_requirement = "requirement")

DEFAULT_VISIBILITY = ["@nre_repo//:core_package_group"]

package(default_visibility = DEFAULT_VISIBILITY)

# Version script to export ONLY PyInit_csrc symbol
genrule(
    name = "gen_version_script",
    outs = ["version_script.lds"],
    cmd = "echo '{ global: PyInit_csrc; local: *; };' > $@",
)

# Compile CUDA kernels and C++ wrappers
cuda_library(
    name = "gsplat_cuda",
    srcs = glob([
        "gsplat/cuda/csrc/*.cu",
        "gsplat/cuda/csrc/*.cpp",
    ]),
    hdrs = glob([
        "gsplat/cuda/csrc/*.h",
        "gsplat/cuda/csrc/*.cuh",
        "gsplat/cuda/include/*.h",
        "gsplat/cuda/include/*.cuh",
    ]) + glob([
        "gsplat/cuda/csrc/third_party/glm/**/*.h",
        "gsplat/cuda/csrc/third_party/glm/**/*.hpp",
        "gsplat/cuda/csrc/third_party/glm/**/*.inl",
    ]),
    copts = [
        # Optimization
        "--use_fast_math",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        # Supress warnings
        "-diag-suppress=20012,186,177",
        # PyTorch CUDA compatibility
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        # Other flags
        "-Xcompiler=-Wno-sign-compare",
        "-Xcompiler=-fvisibility=hidden",
        "-DGSPLAT_BUILD_3DGUT=1",
        "-DGSPLAT_BUILD_3DGS=1",
        "-DGSPLAT_NUM_CHANNELS=1\\,3\\,4\\,8\\,20\\,24\\,32",
    ],
    host_copts = ["-std=c++20"],
    includes = [
        "gsplat/cuda/include",
        "gsplat/cuda/csrc/third_party/glm",
    ],
    deps = [
        "@nre_pip_deps//torch:torch_headers",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

# Create csrc.so inside gsplat/ directory for Python import
cc_binary(
    name = "gsplat/csrc.so",
    srcs = ["gsplat/cuda/ext.cpp"],
    copts = [
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=csrc",
        "-fvisibility=hidden",
        "-DGSPLAT_BUILD_3DGUT=1",
        "-DGSPLAT_BUILD_3DGS=1",
    ],
    linkopts = [
        "-Wl,--no-undefined",
        "-Wl,--as-needed",
        "-Wl,--exclude-libs,ALL",
        "-Wl,--version-script=$(location :gen_version_script)",
    ],
    additional_linker_inputs = [":gen_version_script"],
    linkshared = True,
    deps = [
        ":gsplat_cuda",
        "@nre_pip_deps//torch:libc10",
        "@nre_pip_deps//torch:libc10_cuda",
        "@nre_pip_deps//torch:libtorch_cpu",
        "@nre_pip_deps//torch:libtorch_python",
        "@nre_pip_deps//torch:torch_headers",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

# Create py.typed marker for PEP 561 compliance
genrule(
    name = "gen_py_typed",
    outs = ["gsplat/py.typed"],
    cmd = "touch $@",
)

# Python library, visible only to nre/models/gaussians,
# Use the wrapper: nre/models/gaussians:pylib_cc_gsplat instead
py_library(
    name = "pylib_cc",
    srcs = glob(
        ["gsplat/**/*.py"],
        exclude = [
            "gsplat/**/*_test.py",
            "gsplat/**/test_*.py",
        ],
    ),
    visibility = [
        "@nre_repo//nre/models/gaussians:__pkg__",
        "@nre_repo//:mypy_typing_group",
    ],
    data = [
        "gsplat/csrc.so",  # .so in gsplat directory for import
        "gsplat/py.typed",  # PEP 561 marker for type checking
    ],
    imports = ["."],
    deps = [
        pip_requirement("numpy"),
        pip_requirement("rich"),
        pip_requirement("scipy"),
        pip_requirement("torch"),
    ],
)