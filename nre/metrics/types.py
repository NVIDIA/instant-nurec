# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from enum import Enum, auto


class MetricType(Enum):
    """Enum for all available metric types."""

    PSNR = auto()
    CPSNR = auto()
    LIDAR_COMMON = auto()
    PERCEPTUAL = auto()
    FID = auto()
    TEMPORAL_COHERENCE = auto()
    FEATURE_DRIFT = auto()
    NTD = auto()
    D_SKEW = auto()
    D_KURT = auto()
    FCS_ADAPTIVE = auto()
    OBJECT_LEVEL_SEMANTIC = auto()
    OBJECT_LEVEL_PERCEPTUAL = auto()
    SSIM = auto()
    LPIPS = auto()
