# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from typing import TYPE_CHECKING, Optional

import torch

from nre.nrm.systems.base import BaseNRMSystem
from nre.nrm.systems.gaussians_nrm import GaussiansNRMSystem


if TYPE_CHECKING:
    from nre.nrm.config.nrm import NRMConfig

logger = logging.getLogger(__name__)


def make(name: str, config: "NRMConfig", load_from_checkpoint: Optional[str] = None) -> "BaseNRMSystem":
    """Predict-only standalone: NRE handled both Lightning checkpoint loading
    and weights-only loading; the pretrained config always sets
    resume_weights_only=true so we keep just that branch."""
    if name != "base-nrm-system":
        raise ValueError(f"Unknown NRM system name: {name}")

    system = GaussiansNRMSystem(config)
    if load_from_checkpoint is None:
        return system

    if not config.resume_weights_only:
        raise NotImplementedError("resume_weights_only=False was a Lightning-only path; not supported.")

    checkpoint = torch.load(load_from_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    system.load_state_dict(state_dict, strict=True)
    return system


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseNRMSystem",
    "GaussiansNRMSystem",
]
