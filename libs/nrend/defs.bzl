# Copyright (c) 2024-2025 NVIDIA CORPORATION.  All rights reserved.

""" Instantiates nrend cuda + c++ binding libraries for different architectures """

load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

def obfuscate_jit_headers(headers):
    for header in headers:
        run_binary(
            name = "obfuscated_" + header,
            srcs = [header],
            outs = ["obfuscated/" + header],
            args = ["$(location obfuscated/%s)" % header] + ["$(location :%s)" % header],
            tool = "@tiny_cuda_nn//:obfuscator_cc",
        )

def nrend_libs(architectures):
    """Generate a single NRend CUDA library and Python extension using all TinyCUDA-NN architectures.

    Args:
      architectures: list of CUDA compute-capability strings (e.g. "80", "86") to include TinyCUDA-NN libraries for.
    """

    # Single NRend library targeting all architectures for tin ycudann deps
    # Build the GPU-code library
    aarch_deps = []
    x86_deps = []
    for arch in architectures:
        aarch_deps.append("@nre_pip_deps//tinycudann:lib_tinycudann_%s_aarch64" % arch)
        x86_deps.append("@nre_pip_deps//tinycudann:lib_tinycudann_%s_x86_64" % arch)
    cuda_library(
        name = "nrend_cuda",
        srcs = [
            "src/ngpOccupancyGrid.cu",
            "src/gutRenderer.cu",
            "src/ngpRenderer.cu",
            "src/nreShGaussianModel.cu",
            "src/nreGaussianCompositeModel.cu",
            "src/nreDynamicShGaussianModel.cu",
            "src/nreModel.cu",
            "src/nreLegacyRenderer.cu",
        ],
        hdrs = native.glob(["include/**/*.h", "include/**/*.cuh", "src/obfuscation/*.h"]),
        copts = [
            "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
        ],
        defines = ["TCNN_MIN_GPU_ARCH=70"],
        includes = ["include", "src/obfuscation"],
        deps = [
            "@local_cuda//:nvrtc_so",
            "@optix_dev//:optix_headers",
            "@nre_pip_deps//tinycudann:tinycudann_headers",
            "@nre_pip_deps//torch:libc10",
            "@nre_pip_deps//torch:libc10_cuda",
            "@nre_pip_deps//torch:libtorch_cpu",
            "@nre_pip_deps//torch:libtorch_cuda",
            "@nre_pip_deps//torch:libtorch_python",
            "@nre_pip_deps//torch:torch_headers",
            "@rules_python//python/cc:current_py_cc_libs",
            "@rules_python//python/cc:current_py_cc_headers",
            "//deps/slang:slang_headers",
            "//deps/slang:lib_slang_compiler",
            "//deps/slang:lib_slang_rtc",
        ] + select({
            "@platforms//cpu:aarch64": aarch_deps,
            "//conditions:default": x86_deps,
        }),
    )

    # Build the Python extension binary
    cc_binary(
        name = "nrend_cc",
        srcs = [
            "bindings.cpp",
            "src/cudaBuffer.cpp",
            "src/cudaTexture.cpp",
            "src/cudaRtcKernel.cpp",
            "src/grtOptixRenderer.cpp",
            "src/grutRenderer.cpp",
            "src/optixRtcPipeline.cpp",
            "src/optixAccelerationStructure.cpp",
            "src/rtcKernelConfig.cpp",
            "src/slangRtcKernel.cpp",
            "src/cudaKernelResources.cpp",
            "src/renderer.cpp",
            "src/nreRenderer.cpp",
            "src/status.cpp",
            "src/obfuscation/obfuscation.cpp",
            "src/iNRenderer.cpp",
        ],
        copts = [
            # Flags to be consistent with torch-cuda-extensions and to build correct torch-extension bindings
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            "-DTORCH_EXTENSION_NAME=libnrend_cc",
        ] + select({
            "@//bazel/conditions:internal": ["-DNREND_MAX_LOG_LEVEL=255"],  # allow debug levels for internal builds
            "//conditions:default": ["-DNREND_MAX_LOG_LEVEL=2"],  # allow up to warnings for no-internal builds
        }),
        defines = ["TCNN_MIN_GPU_ARCH=70"],
        linkopts = ["-Wl,--no-undefined", "-Wl,--as-needed"],
        linkshared = True,
        deps = [
            ":nrend_cuda",
            "//libs/vren:vren_headers",
            "@local_cuda//:cuda_so",
            "@nre_pip_deps//torch:libc10",
            "@nre_pip_deps//torch:libc10_cuda",
            "@nre_pip_deps//torch:libtorch_cpu",
            "@nre_pip_deps//torch:libtorch_python",
            "@nre_pip_deps//torch:torch_headers",
            "@rules_python//python/cc:current_py_cc_libs",
        ],
    )
