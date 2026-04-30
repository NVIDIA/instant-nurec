# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Slang Code Generation for Gaussian Parameter Collection Kernels.

This module generates Slang GPU kernel code for Gaussian parameter collection
based on layer configurations. It is used in two contexts:

1. Pre-build time: Generate kernels for common configurations to be compiled
   ahead of time and shipped with the library.

2. Runtime: Generate and compile kernels on-the-fly for custom configurations
   not available in the pre-compiled set.

Architecture:
-------------
The code generator creates Slang kernel wrappers that call the generic
run_tasks() function from collector.slang with specific task types. Each
configuration maps to a unique combination of task types (collectors, activations).

Generated Code Structure:
--------------------------
For each configuration, generates a kernel like:

    [CUDAKernel]
    [Differentiable]
    [AutoPyBindCUDA]
    void collect_parameters_0(
        no_diff uint out_offset,
        no_diff uint count,
        Collector_Copy<3> arg0,
        RotationsCollector_Normalize arg1,
        ...
    ) {
        run_tasks(out_offset, count, arg0, arg1, ...);
    }

The run_tasks() function (defined in collector.slang) handles the actual
parallel processing, and the type parameters determine which operations to perform.

Key Components:
---------------
- CollectorConfiguration: Immutable configuration specifying task types
- CollectorKernelCode: Generated Slang code with kernel names
- generate_collector_code(): Main code generation function
- generate_collector_prebuilt_configs(): Batch generation for pre-compilation

Pre-build Workflow:
-------------------
1. Collect configurations from training runs (OUTPUT_CONFIGURATIONS_PATH)
2. Run this script to generate .slang file with all kernel variants
3. Compile .slang file to native extension module
4. Ship extension module with library for instant loading

Runtime Workflow:
-----------------
1. Request kernel for custom configuration
2. Generate Slang code for that configuration
3. Compile code via SlangTorch
4. Cache compiled kernel for reuse
"""

import argparse
import json

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass(slots=True, frozen=True)
class CollectorConfiguration:
    """Configuration for a Gaussian parameter collection kernel.

    Specifies the sequence of task types (collectors) to be applied to Gaussians.
    Each parameter is a Slang type name that implements the task interface.

    Attributes:
        parameters: Tuple of Slang task type names, e.g.:
            ("Collector_Copy<3>", "RotationsCollector_Normalize",
             "ScalesCollector_Exp", "DensitiesCollector_Sigmoid")
    """

    parameters: Tuple[str, ...]


@dataclass(slots=True, frozen=True)
class CollectorKernelCode:
    """Generated Slang code and associated kernel names.

    Result of code generation containing the complete Slang source code and
    the names of the generated kernel functions.

    Attributes:
        code: Complete Slang source code ready for compilation
        kernel_names: Tuple of kernel function names in the generated code,
            corresponding to the input configurations
    """

    code: str
    kernel_names: Tuple[str, ...]


_COLLECTOR_KERNEL_PREAMBLE = """
//
// Explicit instantiations.  We want to do this at runtime.
//

import collector;
"""

_COLLECTOR_KERNEL_NAME = "collect_parameters_{identifier}"
_COLLECTOR_KERNEL_TEMPLATE = """
[CUDAKernel]
[Differentiable]
[AutoPyBindCUDA]
void {kernel_name}(
    no_diff uint out_offset,
    no_diff uint count,
{parameters_declaration}
) {{
    run_tasks(
        out_offset,
        count,
{parameters_invocation}
    );
}}
"""


def generate_collector_code(configurations: List[Any]) -> CollectorKernelCode:
    """Generate Slang kernel code for the given collector configurations.

    Creates a complete Slang source file with kernel wrappers for each
    configuration. Each kernel is a thin wrapper that calls run_tasks() with
    the appropriate task types from the configuration.

    The generated code includes:
    - Import statement for the base collector module
    - One kernel function per configuration
    - CUDA kernel attributes for compilation
    - Differentiable markers for automatic differentiation
    - AutoPyBindCUDA for Python bindings

    Args:
        configurations: List of CollectorConfiguration objects specifying
            the task types for each kernel variant

    Returns:
        CollectorKernelCode containing the generated Slang source code and
        the list of kernel function names

    Raises:
        ValueError: If an unsupported configuration type is provided
    """
    names = []
    code = _COLLECTOR_KERNEL_PREAMBLE
    for identifier, configuration in enumerate(configurations):
        if type(configuration) == CollectorConfiguration:
            kernel_name = _COLLECTOR_KERNEL_NAME.format(identifier=identifier)
            parameters_declaration = "\n".join(
                [f"    {configuration.parameters[i]} arg{i}," for i in range(len(configuration.parameters))]
            )
            parameters_invocation = "\n".join([f"        arg{i}," for i in range(len(configuration.parameters))])
            code += _COLLECTOR_KERNEL_TEMPLATE.format(
                kernel_name=kernel_name,
                identifier=identifier,
                parameters_declaration=parameters_declaration,
                parameters_invocation=parameters_invocation,
            )
            names.append(kernel_name)
        else:
            raise ValueError(f"Unsupported configuration: {configuration}")

    return CollectorKernelCode(code=code, kernel_names=tuple(names))


def get_collector_configurations(input_json_paths: List[str]) -> List[CollectorConfiguration]:
    """Load and deduplicate collector configurations from JSON files.

    Reads one or more JSON files containing configuration parameter lists and
    combines them into a unique set of configurations. Used during pre-build
    to collect all configurations observed during training/testing.

    Args:
        input_json_paths: List of paths to JSON files, each containing a list
            of parameter tuples representing configurations

    Returns:
        List of unique CollectorConfiguration objects (duplicates removed)
    """
    configurations = []

    for input_json_path in input_json_paths:
        with open(input_json_path, "r") as f:
            configurations_from_json = json.load(f)
        for json_configuration in configurations_from_json:
            collector_configuration = CollectorConfiguration(parameters=tuple(json_configuration))
            if collector_configuration not in configurations:
                configurations.append(collector_configuration)

    return configurations


def generate_collector_prebuilt_configs(input_json_paths: List[str], output_slang_path: str, output_json_path: str):
    """Generate pre-compiled kernel artifacts from configuration files.

    Main function for the pre-build workflow. Takes configurations collected
    from training runs and generates:
    1. A .slang file with all kernel variants (to be compiled to extension)
    2. A .json file mapping configurations to kernel names (for runtime lookup)

    The generated files are used together: the .slang file is compiled into a
    native extension module, and the .json file allows the runtime to find the
    correct kernel for each configuration.

    Args:
        input_json_paths: Paths to JSON files containing configuration lists
        output_slang_path: Path where generated Slang source code will be written
        output_json_path: Path where configuration-to-kernel mapping will be written

    Output Files:
        output_slang_path: Contains Slang code with all kernel functions
        output_json_path: Contains JSON array of {configuration, kernel_name} objects
    """
    configurations = get_collector_configurations(input_json_paths)
    kernel_code = generate_collector_code(configurations)

    # Generate a slang file with the code
    with open(output_slang_path, "w") as f:
        f.write(kernel_code.code)

    # Generate a JSON file with the association between the configurations and the kernel names.
    configurations_and_kernel_names = []
    for configuration, kernel_name in zip(configurations, kernel_code.kernel_names):
        configurations_and_kernel_names.append({"configuration": configuration.parameters, "kernel_name": kernel_name})
    with open(output_json_path, "w") as f:
        json.dump(configurations_and_kernel_names, f, indent=4)


def main():
    """Command-line interface for generating pre-compiled kernel configurations.
    
    This script is used during the build process to generate Slang kernel code
    and mapping files from collected configurations. It reads JSON files
    containing configurations observed during training/testing and produces
    the artifacts needed for pre-compilation.
    
    Usage:
        python codegen.py --input-json configs/*.json \\
                          --output-slang kernels.slang \\
                          --output-json mappings.json
    
    The generated files are then used in the build system:
    1. kernels.slang is compiled to a native extension module
    2. mappings.json is packaged with the library for runtime kernel lookup
    """
    parser = argparse.ArgumentParser(description="Collect configuration prepopulation")
    parser.add_argument(
        "--input-json", type=str, nargs="*", default=[], help="Path(s) to input JSON configuration files"
    )
    parser.add_argument("--output-slang", type=str, required=True, help="Path to the output Slang file")
    parser.add_argument("--output-json", type=str, required=True, help="Path to the output JSON file")

    args = parser.parse_args()

    generate_collector_prebuilt_configs(args.input_json, args.output_slang, args.output_json)


if __name__ == "__main__":
    main()
