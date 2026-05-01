# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os

from typing import TYPE_CHECKING, Optional

import torch

from nre.nrm.systems.gaussians_nrm import GaussiansNRMSystem


if TYPE_CHECKING:
    from nre.nrm.config.nrm import NRMConfig


logger = logging.getLogger(__name__)


def make(config: "NRMConfig", load_from_checkpoint: Optional[str] = None) -> GaussiansNRMSystem:
    """Predict-only standalone: NRE supported both Lightning checkpoint loading
    and weights-only loading; only the latter survives (the predict config
    always set resume_weights_only=true).

    Phase 1 step 5: when ``INSTANT_NUREC_FULL_PT`` is set and the path exists,
    skip construction + state_dict-load and torch.load the entire system.
    Otherwise, follow the build-from-config path and (if the env var is set
    but the file is missing) write the system to that path so the next run
    can use the fast path. This preserves parity bit-for-bit while exposing
    the load shortcut that step 5 deletion will rely on.
    """
    full_pt_path = os.environ.get("INSTANT_NUREC_FULL_PT")

    if full_pt_path and os.path.exists(full_pt_path):
        logger.info("Loading full system from %s (bypassing construct+state_dict path).", full_pt_path)
        loaded = torch.load(full_pt_path, map_location="cpu", weights_only=False)
        assert isinstance(loaded, GaussiansNRMSystem), (
            f"Expected GaussiansNRMSystem from {full_pt_path}, got {type(loaded).__name__}"
        )
        return loaded

    system = GaussiansNRMSystem(config)
    if load_from_checkpoint is None:
        return system

    checkpoint = torch.load(load_from_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    system.load_state_dict(state_dict, strict=True)

    if full_pt_path:
        logger.info("Writing full system to %s for the next run's fast path.", full_pt_path)
        os.makedirs(os.path.dirname(full_pt_path) or ".", exist_ok=True)
        torch.save(system, full_pt_path)

    return system
