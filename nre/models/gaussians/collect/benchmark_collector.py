# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Standalone benchmark for the Gaussian parameter collector kernels.

Runs all layers together in a single collect() call, matching the real
training pattern.

Usage:
    bazel run //nre/models/gaussians/collect:benchmark_collector

Profile with ncu:
    ncu --target-processes all --kernel-name regex:collect_parameters_2 \
        --launch-skip 10 --launch-count 2 --set detailed \
        -o collector_ncu -f \
        bazel run //nre/models/gaussians/collect:benchmark_collector
"""

import os

from dataclasses import fields, is_dataclass

import torch

from nre.models.gaussians.collect import CreateGaussianParameterCollector


try:
    FIXTURE_PATH = os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], "collector_fixture.pt")
except KeyError:
    raise RuntimeError("BUILD_WORKSPACE_DIRECTORY not set. Run via: bazel run ...")

"""
How to generate the fixture:
in SlangGaussianParameterCollector.collect()
torch.save(
    {
        'layers_config': self.layers_config,
        'layers_data': layers_data,
        'layer_indices': layer_indices,
        'result': result,
    },
    'collector_fixture.pt',
)
"""
NB_WARMUP = 10
NB_MEASURE = 10


def zero_grad(obj):
    """Recursively zero gradients on tensors in a dataclass tree."""
    if isinstance(obj, torch.Tensor):
        obj.grad = None
    elif is_dataclass(obj):
        for field in fields(obj):
            zero_grad(getattr(obj, field.name))
    elif isinstance(obj, list):
        for item in obj:
            zero_grad(item)


def main():
    print("Loading fixture...")
    fixture = torch.load(FIXTURE_PATH, map_location="cuda", weights_only=False)
    layers_config = fixture["layers_config"]
    layers_data = fixture["layers_data"]

    print("Creating collector...")
    collector = CreateGaussianParameterCollector(layers_config)
    num_layers = len(layers_data.layers)
    layer_indices = list(range(num_layers))

    print(f"Benchmarking all {num_layers} layers together ({NB_WARMUP} warmup + {NB_MEASURE} measured)\n")

    # Warmup
    for _ in range(NB_WARMUP):
        zero_grad(layers_data)
        result = collector.collect(layers_data, layer_indices=layer_indices)
        buffers = [getattr(result, field.name) for field in fields(result)]
        gradients = [torch.randn_like(buffer) for buffer in buffers]
        torch.autograd.backward(buffers, gradients)
    torch.cuda.synchronize()

    # Measured iterations with NVTX ranges
    for i in range(NB_MEASURE):
        zero_grad(layers_data)
        with torch.cuda.nvtx.range("forward"):
            result = collector.collect(layers_data, layer_indices=layer_indices)
        with torch.cuda.nvtx.range("backward"):
            buffers = [getattr(result, field.name) for field in fields(result)]
            gradients = [torch.randn_like(buffer) for buffer in buffers]
            torch.autograd.backward(buffers, gradients)
    torch.cuda.synchronize()

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
