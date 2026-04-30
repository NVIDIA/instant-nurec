# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Pytest plugin for GPU load balancing, allowing improved GPU utilization when running parallel tests on multi-GPU
systems.

This module is registered as a pytest plugin and runs GPU assignment logic in pytest_configure, which executes before
test collection and module imports. This ensures CUDA_VISIBLE_DEVICES is set before any CUDA/PyTorch imports.

Environment variables (set by the pytest_test macro):
    TEST_TOTAL_GPUS: Number of available GPUs
    TEST_NAME_HASH: Hash of the test target name
    TEST_IS_MULTI_GPU: If set, the test uses multiple GPUs and needs all available GPUs exposed
"""

import os

import pytest


def _setup_gpu_visibility() -> None:
    """Expose a specific GPU to this test based on environment variables set by the pytest_test macro."""
    total_gpus: str | None = os.environ.get("TEST_TOTAL_GPUS")
    name_hash: str | None = os.environ.get("TEST_NAME_HASH")
    is_multi_gpu: str | None = os.environ.get("TEST_IS_MULTI_GPU")

    # No GPUs detected: nothing to do
    num_gpus: int = int(total_gpus) if total_gpus else 0
    if num_gpus == 0:
        print("[GPU Selection] No GPUs detected, skipping")
        return

    # Multi-GPU test: expose all GPUs
    if is_multi_gpu:
        print("[GPU Selection] Multi-GPU test, exposing all available GPUs")
        return

    if not name_hash:
        raise RuntimeError("[GPU Selection] TEST_NAME_HASH not set - this should be set by 'pytest_test' macro")

    test_hash: int = int(name_hash)
    gpu_id: int = test_hash % num_gpus if num_gpus > 1 else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU Selection] Exposing GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id}), out of {num_gpus} total GPUs")


def pytest_configure(config: pytest.Config) -> None:
    """Pytest hook that runs before test collection and module imports."""
    _setup_gpu_visibility()
