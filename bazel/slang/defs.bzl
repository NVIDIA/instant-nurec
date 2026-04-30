# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

""" Precompile a Slang module into a torch C++ CUDA extension """

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

def slangtorch_library(name, srcs, extra_cmd_args = "", deps = [], use_fast_math = True, **kwargs):
    """
    Precompile a Slang module into a torch C++ CUDA extension

    Args:
        name: The name of the target.
        srcs: The list of slang source files to compile. If not provided, the slang module name is used.
        extra_cmd_args: Additional arguments to pass to the slangc command.
        deps: The list of dependencies to add to the target additionaly to the srcs.
        use_fast_math: Whether to use fast math for the slang module.
        **kwargs: Additional arguments to pass to the genrule and cc_binary targets.
    """

    # CUDA module source generation
    module_src_str = " ".join(["$(location " + s + ")" for s in srcs])
    slangc_cmd_base = "$(location //deps/slang:slangc) " + extra_cmd_args + " " + module_src_str
    slangc_cmd_coverage = slangc_cmd_base + " -g -line-directive-mode standard -O0 -o $@"
    slangc_cmd_prod = slangc_cmd_base + " -O3 -o $@"
    combined_deps = deps + srcs

    prefix_name = name
    if name.endswith("_cc"):
        prefix_name = name[:-3]

    # CUDA module
    native.genrule(
        name = prefix_name + "_cuda_src",
        srcs = combined_deps,
        outs = [prefix_name + "_cuda.cu"],
        cmd = select({
            "//bazel/conditions:coverage_mode": slangc_cmd_coverage + " -target cuda",
            "//conditions:default": slangc_cmd_prod + " -target cuda",
        }),
        tools = ["//deps/slang:slangc"],
        **kwargs
    )

    cuda_library(
        name = prefix_name + "_cuda",
        srcs = [":" + prefix_name + "_cuda_src"],
        copts = [
            # Flags to be consistent with torch-cuda-extensions
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-w",
            "--generate-line-info",
        ] + (["--use_fast_math"] if use_fast_math else []),
        deps = [
            "@nre_pip_deps//torch:torch_headers",
            "@rules_python//python/cc:current_py_cc_headers",
        ],
    )

    # Torch C++ binding
    native.genrule(
        name = prefix_name + "_src",
        srcs = combined_deps,
        outs = [prefix_name + ".cpp"],
        cmd = select({
            "//bazel/conditions:coverage_mode": slangc_cmd_coverage + " -target torch-binding",
            "//conditions:default": slangc_cmd_prod + " -target torch-binding",
        }),
        tools = ["//deps/slang:slangc"],
        **kwargs
    )

    cc_binary(
        name = name,
        srcs = [":" + prefix_name + "_src"],
        copts = [
            # Flags to be consistent with torch-cuda-extensions and to build correct torch-extension bindings
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            "-DTORCH_EXTENSION_NAME=lib" + name,
            "-w",
        ] + select({
            # Slang generates `typedef _Float16 half;` which requires gcc-13+ on aarch64.
            # Map _Float16 to the supported __fp16 type for older compilers.
            "@platforms//cpu:aarch64": ["-D_Float16=__fp16"],
            "//conditions:default": [],
        }),
        linkopts = [
            "-Wl,--no-undefined",
            "-Wl,--as-needed",
        ],
        linkshared = True,
        deps = [
            ":" + prefix_name + "_cuda",
            "@nre_pip_deps//torch:libc10",
            "@nre_pip_deps//torch:libc10_cuda",
            "@nre_pip_deps//torch:libtorch_cpu",
            "@nre_pip_deps//torch:libtorch_python",
            "@nre_pip_deps//torch:torch_headers",
            "@rules_python//python/cc:current_py_cc_libs",
        ],
    )
