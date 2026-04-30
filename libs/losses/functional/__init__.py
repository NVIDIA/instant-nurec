# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Layer 1: Autograd Function - PyTorch autograd bridge for losses."""

from libs.losses.functional.cuda_losses_function import CudaLossesFunction, RoadGaussiansFunction


def __getattr__(name: str):
    """Lazy-load SlangLossesFunction (only available when Slang kernels are built)."""
    if name == "SlangLossesFunction":
        from libs.losses.functional.slang_losses_function import SlangLossesFunction

        globals()["SlangLossesFunction"] = SlangLossesFunction
        return SlangLossesFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CudaLossesFunction", "SlangLossesFunction", "RoadGaussiansFunction"]
