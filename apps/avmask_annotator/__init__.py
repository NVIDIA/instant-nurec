# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""AV Mask Annotator module for camera frame mask editing and annotation.

This package provides tools for annotating and editing masks in camera frames.
"""

from apps.avmask_annotator.mask_annotator import MaskAnnotator, run_mask_annotator


__all__ = ["MaskAnnotator", "run_mask_annotator"]
