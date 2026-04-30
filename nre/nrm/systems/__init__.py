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
    if name == "base-nrm-system":
        system_cls = GaussiansNRMSystem
    else:
        raise ValueError(f"Unknown NRM system name: {name}")

    if load_from_checkpoint is not None:
        # Load checkpoint to check its format
        try:
            checkpoint = torch.load(load_from_checkpoint, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {load_from_checkpoint}: {e}")

        # Check if this is a PyTorch Lightning checkpoint
        if config.resume_weights_only:
            # Create system first, then manually load the state dict
            system = system_cls(config)
            if "state_dict" in checkpoint:
                # Standard Lightning checkpoint format but missing version info
                system.load_state_dict(checkpoint["state_dict"], strict=True)
            else:
                # Assume the entire checkpoint is the state dict
                system.load_state_dict(checkpoint, strict=True)
            return system
        else:
            # Standard Lightning checkpoint loading
            return system_cls.load_from_checkpoint(load_from_checkpoint, strict=True, config=config)

    return system_cls(config)


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseNRMSystem",
    "GaussiansNRMSystem",
]
