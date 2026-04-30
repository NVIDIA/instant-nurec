# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import Dict

import torch

from asset_harvester.tokengs.gs import GaussianRenderer


@dataclass
class RenderConfig:
    output_size: int = 512
    znear: float = 0.1
    zfar: float = 500.0
    fov: float = 70
    deferred_bp: bool = False


def render(
    opt: RenderConfig, gaussians: torch.Tensor, cam_view: torch.Tensor, intrinsics: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Render gaussians using TokenGS gsplat renderer, matching the old NRend interface."""
    renderer = GaussianRenderer(opt)
    result = renderer.render(
        gaussians,
        cam_view,
        intrinsics=intrinsics,
        output_size=(opt.output_size, opt.output_size),
    )
    return {"image": result["images_pred"], "alpha": result["alphas_pred"]}
