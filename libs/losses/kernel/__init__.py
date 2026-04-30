# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch

import libs.losses.kernel.liblosses_cuda_cc as cuda_losses  # type: ignore # pycena: skip


def __getattr__(name: str):
    """Lazy-load Slang losses kernel (only built for tests)."""
    if name == "slang_losses":
        import libs.losses.kernel.liblosses_slang_cc as slang_losses  # type: ignore # pycena: skip

        globals()["slang_losses"] = slang_losses
        return slang_losses
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cuda_losses", "slang_losses"]
