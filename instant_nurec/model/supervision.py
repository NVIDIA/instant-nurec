# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable predictions consumed by the training losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True, slots=True)
class MotionSupervision:
    context_flow: torch.Tensor
    source_timestamps_us: torch.Tensor
    target_timestamps_us: torch.Tensor
    reference_flow: torch.Tensor | None = None


@dataclass(kw_only=True, slots=True)
class SupervisionPack:
    """Differentiable supervision produced by the full model."""

    context_depth: torch.Tensor | None = None
    context_distance_confidence: torch.Tensor | None = None
    context_rgb: torch.Tensor | None = None
    context_world_normal: torch.Tensor | None = None
    context_semantic_logits: torch.Tensor | None = None
    motion_supervisions: list[MotionSupervision] | None = None
    predicted_sky_cubemap: torch.Tensor | None = None
    reference_sky_cubemap: torch.Tensor | None = None
    reference_sky_cubemap_mask: torch.Tensor | None = None
