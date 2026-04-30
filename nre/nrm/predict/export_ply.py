# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path

import torch

from nre.models.gaussians.utils import RGB2SH, write_ply_3dgs
from nre.models.utils import get_activation
from nre.nrm.config.predict import PrimitivePLYExportConfig
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.nrm.primitives.kelvin_primitive import KelvinNRMPrimitive, KelvinSemanticClass
from nre.utils.types import RigTrajectories


logger = logging.getLogger(__name__)


@dataclass
class PLYExportGaussians:
    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    rgb: torch.Tensor
    road_mask: torch.Tensor | None = None
    sky_mask: torch.Tensor | None = None
    normals: torch.Tensor | None = None

    def export(self, config: PrimitivePLYExportConfig, output_path: Path):
        """
        Export the gaussians to the PLY file used in SO, and values are exported as NuRec expects them.
        Colors are exported in SH, while scale and density are exported preactivated.
        """
        scales = self.scales.float()
        if config.scale_activation is not None:
            scale_activation_inv = get_activation(config.scale_activation, inverse=True)
            scales = scale_activation_inv(scales)

        densities = self.densities.float()
        if config.density_activation is not None:
            density_activation_inv = get_activation(config.density_activation, inverse=True)
            densities = density_activation_inv(densities)

        rgb = self.rgb.float()
        features_albedo = rgb if config.color_mode == "rgb" else RGB2SH(rgb)

        custom_attributes = {}
        if self.road_mask is not None:
            custom_attributes["road_mask"] = self.road_mask
        if self.sky_mask is not None:
            custom_attributes["sky_mask"] = self.sky_mask

        write_ply_3dgs(
            path=output_path,
            positions=self.positions.float(),
            rotations=self.rotations.float(),
            scales=scales,
            densities=densities,
            features_albedo=features_albedo,
            color=rgb,
            normals=self.normals.float() if self.normals is not None else None,
            custom_attributes=custom_attributes,
        )


def get_gaussian_shape_pruning_mask(
    config: PrimitivePLYExportConfig, densities: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """
    Get the gaussian pruning mask for the given primitive, based on the shape of the gaussians.
    Returns [num_gaussians, 1] mask tensor.
    """

    mask = torch.isfinite(densities)
    logger.info(f"Removed {(~mask).sum().item():,} non-finite density gaussians.")
    logger.info(f"Starting to filter {mask.sum().item():,} finite gaussians.")

    if config.minimum_density is not None:
        logger.info(f"Filter (density) removed {torch.sum(mask & (densities < config.minimum_density))} gaussians")
        mask = mask & (densities >= config.minimum_density)

    if config.minimum_scale is not None:
        logger.info(
            f"Filter (scale) removed {torch.sum(mask & (torch.min(scales, dim=1).values < config.minimum_scale).unsqueeze(1))} gaussians"
        )
        mask = mask & (torch.min(scales, dim=1).values >= config.minimum_scale).unsqueeze(1)

    if config.minimum_surface_area is not None:
        logger.info(
            f"Filter (surface area) removed {torch.sum(mask & (scales.square().sum(dim=1) < config.minimum_surface_area).unsqueeze(1))} gaussians"
        )
        mask = mask & (scales.square().sum(dim=1) >= config.minimum_surface_area).unsqueeze(1)

    return mask


def export_kelvin_ply(config: PrimitivePLYExportConfig, primitives: KelvinNRMPrimitive) -> PLYExportGaussians:
    """Export the ply file, static layer only."""
    static_layer = primitives.static_layer

    mask = get_gaussian_shape_pruning_mask(config, static_layer.densities, static_layer.scales).squeeze(-1)
    logger.info(f"After filtering, {torch.sum(mask)} gaussians remaining.")

    static_layer = static_layer.mask(mask)

    # Derive road/sky masks from per-gaussian semantic class
    road_mask: torch.Tensor | None = None
    sky_mask: torch.Tensor | None = None
    if static_layer.semantic_class is not None:
        road_mask = (static_layer.semantic_class == KelvinSemanticClass.ROAD).squeeze(-1).to(dtype=torch.uint8)
        sky_mask = (static_layer.semantic_class == KelvinSemanticClass.SKY).squeeze(-1).float()

    return PLYExportGaussians(
        positions=static_layer.positions,
        rotations=static_layer.rotations,
        scales=static_layer.scales,
        densities=static_layer.densities,
        rgb=static_layer.rgb,
        road_mask=road_mask,
        sky_mask=sky_mask,
        normals=static_layer.normals,
    )


def export_ply(
    config: PrimitivePLYExportConfig, primitives: BaseNRMPrimitive, rig_trajectories: RigTrajectories, path: Path
) -> None:
    """Export the NRM Primitives as a ply file after transforming to world space and applying some filtering.
    This ply export is intended to be used as an initialization for NuRec SO.
    """
    # First transform the primitives to the world frame
    primitives = primitives.rigid_transform(rig_trajectories.T_world_base.to(dtype=torch.float32))

    # Then compute the gaussians to be exported
    if isinstance(primitives, KelvinNRMPrimitive):
        gaussians_ply = export_kelvin_ply(config, primitives)
    else:
        raise ValueError(f"Unsupported primitive type: {type(primitives)}")

    # Then export the ply
    gaussians_ply.export(config, path)
