# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import importlib

from typing import Any

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch


# Import the C++ compiled modules
mcmc_slang: Any = importlib.import_module("libs.slang_gaussians.libmcmc_slang_cc")
slang_collector: Any = importlib.import_module("libs.slang_gaussians.libcollector_slang_cc")
gsplat_strategy_cuda: Any = importlib.import_module("libs.slang_gaussians.libgsplat_strategy_cuda_cc")

# Lazy import for Slang GSplat strategy (test-only; avoids SlangTorch build dep at runtime)
_gsplat_strategy_slang: Any = None

# Import the Python wrapper module that uses the C++ kernels
mcmc: Any = importlib.import_module("libs.slang_gaussians.mcmc.mcmc")
gsplat_strategy: Any = importlib.import_module("libs.slang_gaussians.strategy.gsplat")


def __getattr__(name: str) -> Any:
    global _gsplat_strategy_slang
    if name == "gsplat_strategy_slang":
        if _gsplat_strategy_slang is None:
            _gsplat_strategy_slang = importlib.import_module("libs.slang_gaussians.libgsplat_strategy_slang_cc")
        return _gsplat_strategy_slang
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
