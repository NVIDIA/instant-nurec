# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Slang Kernel Loading and Caching for Gaussian Parameter Collection.

This module manages the loading, compilation, and caching of Slang GPU kernels
used for Gaussian parameter collection. It provides a two-tier caching strategy:

1. Pre-compiled kernels: Common layer configurations have pre-compiled kernels
   that are loaded at startup for instant availability.

2. Runtime compilation: If a requested configuration is not pre-compiled, the
   kernel is generated and compiled on-the-fly, then cached for future use.

Key Components:
---------------
- CollectorKernel: Dataclass holding a compiled Slang kernel and its module
- get_slang_kernels(): Main entry point for kernel retrieval/compilation
- _cached_kernels: Global cache mapping configurations to compiled kernels
- load_prebuilt_configs(): Loads pre-compiled kernels from disk

Caching Strategy:
-----------------
The cache is populated lazily:
1. On first use, pre-compiled kernels are loaded from disk
2. Requested kernels are looked up in the cache
3. Cache misses trigger runtime compilation via code generation
4. Newly compiled kernels are added to the cache for reuse

Thread Safety:
--------------
Prebuilt kernel loading and runtime cache updates are serialized with
``threading`` locks so concurrent callers do not populate ``_cached_kernels``
twice.
Kernel compilation uses file locking (fcntl) to prevent race conditions when
multiple processes compile the same configuration simultaneously.

Performance:
------------
- Pre-compiled kernels: Instant loading (no compilation overhead)
- Runtime compilation: ~10-15 seconds per unique compilation
- Cache persists across collector instances but not across process restarts
"""

import fcntl
import hashlib
import json
import os
import tempfile
import threading
import types

from dataclasses import dataclass
from typing import Any, Dict, List

import slangtorch

from libs.slang_gaussians.collector.codegen import CollectorConfiguration, generate_collector_code
from libs.slang_utils.utils import add_ninja_to_path, logger


@dataclass(slots=True, frozen=True)
class CollectorKernel:
    """Container for a compiled Slang kernel and its parent module.

    Attributes:
        slang_module: The SlangTorch module containing the compiled kernel
        kernel: The specific kernel function callable from the module
    """

    slang_module: Any
    kernel: Any


def get_slang_module_path():
    """Get the file system path to the base collector.slang-module.

    Returns:
        Path to the collector.slang-module file used as the base for all kernels
    """
    from python.runfiles import runfiles

    r = runfiles.Create()
    path = r.Rlocation("nre_repo/libs/slang_gaussians/collector/collector.slang-module")
    return path


_cached_kernels: Dict[CollectorConfiguration, CollectorKernel] = {}

# Serialize lazy prebuilt load and JIT compilation/cache insertion for concurrent callers.
_prebuilt_load_lock = threading.Lock()
_slang_kernel_cache_lock = threading.Lock()


def load_prebuilt_configs(slang_extension: types.ModuleType, json_path: str) -> None:
    """Load pre-compiled kernels from a Slang extension module into the cache.

    Reads a JSON mapping file that associates layer configurations with kernel names
    in the pre-compiled extension, then populates the global kernel cache.

    Args:
        slang_extension: Pre-compiled Slang extension module containing kernels
        json_path: Path to JSON file mapping configurations to kernel names

    Raises:
        ValueError: If a configuration already exists in the cache
    """
    # Get the association between the configurations and the kernel names.
    with open(json_path, "r") as f:
        configurations_and_kernel_names = json.load(f)

    # Fill in the cached kernels.
    slang_module = slangtorch.util.wrapModule(slang_extension)
    for configuration_and_kernel_name in configurations_and_kernel_names:
        configuration_parameters = configuration_and_kernel_name["configuration"]
        kernel_name = configuration_and_kernel_name["kernel_name"]
        configuration = CollectorConfiguration(parameters=tuple(configuration_parameters))
        if configuration in _cached_kernels:
            raise ValueError(f"Configuration {configuration} already exists in the cached kernels")
        kernel = getattr(slang_module, kernel_name)
        collector_kernel = CollectorKernel(slang_module=slang_module, kernel=kernel)
        _cached_kernels[configuration] = collector_kernel


_prebuilt_configs_loaded = False


def _load_prebuilt_configs_if_not_loaded() -> None:
    """Lazily load pre-compiled kernels on first use.

    This function is called automatically by get_slang_kernels() to ensure
    pre-compiled kernels are loaded before checking the cache. Uses a global
    flag under a lock so only one thread loads per process.
    """
    global _prebuilt_configs_loaded
    if _prebuilt_configs_loaded:
        return
    with _prebuilt_load_lock:
        if _prebuilt_configs_loaded:
            return

        # Get the path with the json association between the configurations and the kernel names.
        from python.runfiles import runfiles

        r = runfiles.Create()
        json_path = r.Rlocation("nre_repo/libs/slang_gaussians/collector-default-configs.json")

        # Import the pre-built extension.
        from libs.slang_gaussians.interface import slang_collector  # type: ignore

        load_prebuilt_configs(slang_collector, json_path)

        _prebuilt_configs_loaded = True


def get_slang_kernels(configurations: List[Any]) -> List[CollectorKernel]:
    """Get compiled Slang kernels for the requested layer configurations.

    This is the main entry point for kernel retrieval. It implements a two-tier
    caching strategy:

    1. Pre-compiled kernels: Checks the cache for pre-compiled kernels (loaded
       from disk on first call). Most common configurations are pre-compiled.

    2. Runtime compilation: If a configuration is not cached, generates Slang
       code for it, compiles it, and adds it to the cache for future use.

    The function handles multiple configurations efficiently by batch-compiling
    all missing kernels together in a single Slang module.

    Args:
        configurations: List of CollectorConfiguration objects specifying the
            layer configurations (e.g., activation types, dimensions)

    Returns:
        List of CollectorKernel objects in the same order as input configurations.
        Each contains the compiled kernel and its parent Slang module.

    Side Effects:
        - Loads pre-compiled kernels on first call (one-time setup)
        - Writes generated Slang code to temp files when compiling
        - Optionally outputs configurations to file if OUTPUT_CONFIGURATIONS_PATH
          environment variable is set (used for generating pre-built configs)

    Performance:
        - Pre-compiled kernels: ~0ms (cache lookup)
        - Runtime compilation: ~10-15 seconds per batch of unique configurations
        - Compiled kernels are cached for the lifetime of the process
    """
    _load_prebuilt_configs_if_not_loaded()

    # We have the possibility to output the configurations to a file, to be able to
    # pre-build the kernels and not have to compile them at runtime.
    output_configurations_path = os.environ.get("OUTPUT_CONFIGURATIONS_PATH", None)
    if output_configurations_path:
        output_configurations = [configuration.parameters for configuration in configurations]
        with open(output_configurations_path, "w") as f:
            json.dump(output_configurations, f, indent=4)

    with _slang_kernel_cache_lock:
        to_build = []
        # Keep the order of provided configurations, but only build unique ones.
        for configuration in configurations:
            if configuration not in _cached_kernels and configuration not in to_build:
                to_build.append(configuration)

        # We can build the kernels in one go to group compilation together.
        if len(to_build) > 0:
            kernel_code = generate_collector_code(to_build)

            # Using hashlib, because hash() is not deterministic across sessions.
            code_hash = hashlib.sha256(kernel_code.code.encode()).hexdigest()[:8]
            slang_file_path = os.path.join(tempfile.gettempdir(), f"collector_{code_hash}.slang")
            with open(slang_file_path, "w") as tf:
                try:
                    fcntl.flock(tf.fileno(), fcntl.LOCK_EX)
                    tf.write(kernel_code.code)
                finally:
                    fcntl.flock(tf.fileno(), fcntl.LOCK_UN)

            logger.info(f"Loading slang module: {slang_file_path} for configurations: {to_build}")

            add_ninja_to_path()
            # Find the pre-compiled collector module.
            includePaths = [os.path.realpath(os.path.dirname(get_slang_module_path()))]

            # Coverage mode: persist source to cache for NCU source correlation
            if os.environ.get("CUDA_COVERAGE_MODE") == "1":
                cache_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    code_hash = hashlib.sha256(kernel_code.code.encode("utf-8")).hexdigest()[:8]
                    slang_file_path = os.path.join(cache_dir, f"collector_{code_hash}.slang")
                    if not os.path.exists(slang_file_path):
                        with open(slang_file_path, "w") as f:
                            f.write(kernel_code.code)
                    logger.info(f"Loading slang module: {slang_file_path} for configurations: {to_build}")

            slang_module = slangtorch.loadModule(slang_file_path, verbose=False, includePaths=includePaths)

            assert len(kernel_code.kernel_names) == len(to_build)
            for name, configuration in zip(kernel_code.kernel_names, to_build):
                _cached_kernels[configuration] = CollectorKernel(
                    slang_module=slang_module, kernel=getattr(slang_module, name)
                )

    return [_cached_kernels[configuration] for configuration in configurations]
