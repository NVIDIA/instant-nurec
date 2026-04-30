# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Pose calibration kernels package - Layer 0 GPU operations for pose calibration."""

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch  # noqa: F401

from libs.sensors.kernels.pose_calib.bindings import compute_poses_and_timestamps  # pycena: skip


def __getattr__(name: str):
    """Lazy-load CUDA and Slang pose_calib modules."""
    if name == "pose_calib_cuda":
        import libs.sensors.libpose_calib_cuda_cc as pose_calib_cuda  # type: ignore # pycena: skip

        globals()["pose_calib_cuda"] = pose_calib_cuda
        return pose_calib_cuda
    if name == "pose_calib_slang":
        import libs.sensors.libpose_calib_slang_cc as pose_calib_slang  # type: ignore # pycena: skip

        globals()["pose_calib_slang"] = pose_calib_slang
        return pose_calib_slang
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # CUDA module (production, lazy-loaded)
    "pose_calib_cuda",
    # Kernel functions
    "compute_poses_and_timestamps",
]
