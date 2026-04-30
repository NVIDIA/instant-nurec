# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Layer 2: Loss Modules - Neural network modules for losses."""

from libs.losses.models.base_losses import (
    BaseLoss,
    BaseLossWithSemanticWeights,
    BasePrimitiveLoss,
    BaseRenderLoss,
    SlangBaseLoss,
)
from libs.losses.models.cuda_losses_module import RoadGaussiansLossCUDA
from libs.losses.models.losses_module import ModuleLosses


__all__ = [
    "BaseLoss",
    "BaseRenderLoss",
    "BasePrimitiveLoss",
    "BaseLossWithSemanticWeights",
    "SlangBaseLoss",
    "ModuleLosses",
    "RoadGaussiansLossCUDA",
]
