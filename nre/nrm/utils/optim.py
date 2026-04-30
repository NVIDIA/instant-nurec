# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone keeps just
# mark_parameter_no_weight_decay (mamba_scan tags its own params at
# construction time). Optimizer/scheduler factories and the progress-
# based schedulers are training-only and removed.

from __future__ import annotations

import torch


def mark_parameter_no_weight_decay(param: torch.nn.Parameter) -> None:
    setattr(param, "_nrm_no_weight_decay", True)
