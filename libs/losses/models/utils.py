# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import ast
import logging
import math
import operator

from typing import List, Type

import torch

from torch.nn import functional as F

from nre.models.base import BaseModel
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.models.gaussians.gaussians_model import RigidGaussianModel
from nre.models.post_processing import (
    BilateralGridT,
)
from nre.nrm.models.base import BaseNRM
from nre.utils.batch import CameraFrameLabels, LidarFrameLabels
from nre.utils.types import GaussiansCompositeReturn, RayFlags


def _get_bilateral_grids(
    model: BaseModel | BaseNRM, bilateral_grid_class: Type[BilateralGridT]
) -> List[BilateralGridT]:
    grids: List[BilateralGridT] = []
    match model:
        # Match any model that supports post processing model
        case GaussiansComposite():
            for pp in model.post_processings:
                # Test if post-processing module belongs to requested class.
                if isinstance(pp, bilateral_grid_class):
                    grids.append(pp)
        case BaseNRM():
            # NRM uses different mechanism for post-processing, hence returning empty list.
            pass
        case _:
            raise RuntimeError(f"{_get_bilateral_grids.__qualname__} got unsupported model type {type(model)}")

    return grids


def _get_out_of_bound_gaussian_nodes(model: BaseModel) -> list[RigidGaussianModel]:
    if not isinstance(model, GaussiansComposite):
        return []
    gaussian_nodes = []
    for node_id in model.get_gaussians_node_ids():
        node = model.gaussians_nodes[node_id]
        if not isinstance(node, RigidGaussianModel):
            continue
        positions = node.get_positions()
        if positions.numel() == 0:
            continue
        gaussian_nodes.append(node)
    return gaussian_nodes


def get_rendered_visibility_mask(
    results: GaussiansCompositeReturn,
    element_count: int,
    device: torch.device,
    dtype: torch.dtype,
    visibility_filter: bool,
    occlusion_aware: bool = False,
) -> torch.Tensor:
    """
    Build visibility mask for gaussian regularization losses.

    If visibility_filter is False, returns an all-ones mask.
    If True, combines available camera/lidar visibility by union.
    """
    visibility = torch.ones(element_count, device=device, dtype=dtype)
    if not visibility_filter:
        return visibility

    visibility_sources: list[torch.Tensor] = []
    rendered_cam = results.rendered_cam
    if rendered_cam is not None:
        if occlusion_aware and rendered_cam.cumulated_weights is not None:
            visibility_sources.append(rendered_cam.cumulated_weights)
        elif (not occlusion_aware) and rendered_cam.visibility is not None:
            visibility_sources.append(rendered_cam.visibility)
    rendered_lidar = results.rendered_lidar
    if rendered_lidar is not None:
        if occlusion_aware and rendered_lidar.cumulated_weights is not None:
            visibility_sources.append(rendered_lidar.cumulated_weights)
        elif (not occlusion_aware) and rendered_lidar.visibility is not None:
            visibility_sources.append(rendered_lidar.visibility)

    if len(visibility_sources) == 0:
        if occlusion_aware:
            raise ValueError(
                "get_rendered_visibility_mask requires rendered_cam or rendered_lidar cumulated_weights when visibility_filter=True and occlusion_aware=True"
            )
        else:
            raise ValueError(
                "get_rendered_visibility_mask requires rendered_cam or rendered_lidar visibility when visibility_filter=True"
            )

    visibility = visibility_sources[0]
    for visibility_source in visibility_sources[1:]:
        visibility = torch.maximum(visibility, visibility_source)
    if occlusion_aware:
        visibility = (visibility > 0.0).float()

    if visibility.numel() != element_count:
        logging.warning(
            f"get_rendered_visibility_mask: visibility shape mismatch (got {visibility.numel()}, expected {element_count}). "
            f"Falling back to all-ones mask (visibility filter will be ignored)."
        )
        return torch.ones(element_count, device=device, dtype=dtype)

    return visibility


def _maybe_update_mcmc_visibility_counters(
    *,
    model: BaseModel,
    results: GaussiansCompositeReturn,
    visibility: torch.Tensor,
) -> None:
    if not isinstance(model, GaussiansComposite):
        return
    strategy = getattr(model, "gaussians_strategy", None)
    update_fn = getattr(strategy, "update_visibility_counters", None)
    if update_fn is None:
        return
    update_fn(
        results=results,
        visibility=visibility.detach().view(-1),
        gaussians_nodes=model.gaussians_nodes,
    )


def gaussian(window_size: int, sigma: float, device: torch.device = torch.device("cuda")) -> torch.Tensor:
    x = torch.arange(window_size, device=device, dtype=torch.float32)
    gauss = torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, device: torch.device = torch.device("cuda")) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5, device=device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # C, B, H, W
    return window


def torch_ssim(
    img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int = 11, channel: int = 3
) -> torch.Tensor:
    """The reference implementation of the SSIM loss as proposed in
    Zhou Wang, et al. (2024): Image quality assessment: from error visibility to structural similarity. In: IEEE Transactions on Image Processing.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


def get_mask_semantic(rays_meta: LidarFrameLabels | CameraFrameLabels, semantic_expr: str) -> torch.Tensor:
    """Get mask given semantic class specified using a string.

    Supports logical combinations using & (AND), | (OR), ~/! (NOT) operators.
    Examples:
        - "valid" - single semantic class
        - "valid&synthetic" - AND combination
        - "valid|synthetic" - OR combination
        - "valid&synthetic|~road" - mixed operations
    """
    _UNARY_OPS = {
        ast.Invert: operator.invert,
        ast.Not: operator.invert,
    }
    _BIN_OPS = {
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
    }

    def eval_expr(node: ast.AST) -> torch.Tensor:
        if isinstance(node, ast.Name):
            match node.id.strip():
                case "valid":
                    return rays_meta.get_mask_flags_none(RayFlags.INVALID)
                case "synthetic":
                    return rays_meta.get_mask_flags_all(RayFlags.SYNTHETIC)
                case "road":
                    return rays_meta.get_mask_flags_all(RayFlags.ROAD_SEMANTIC)
                case "sky":
                    return rays_meta.get_mask_flags_all(RayFlags.SKY_SEMANTIC)
                case "vehicle":
                    return rays_meta.get_mask_flags_all(RayFlags.VEHICLE_SEMANTIC)
                case "ego":
                    return rays_meta.get_mask_flags_all(RayFlags.EGO_SEMANTIC)
                case _:
                    raise RuntimeError(f"Got unsupported semantic class {node.id}.")

        elif isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in _UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
            operand = eval_expr(node.operand)
            return _UNARY_OPS[unary_op_type](operand)

        elif isinstance(node, ast.BinOp):
            bin_op_type = type(node.op)
            if bin_op_type not in _BIN_OPS:
                raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")
            left = eval_expr(node.left)
            right = eval_expr(node.right)
            return _BIN_OPS[bin_op_type](left, right)

        elif isinstance(node, ast.Expr):
            return eval_expr(node.value)

        else:
            raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

    parsed = ast.parse(semantic_expr, mode="eval")
    return eval_expr(parsed.body)
