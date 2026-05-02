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

from instant_nurec._pkg.models.gaussians.utils import RGB2SH, write_ply_3dgs
from instant_nurec._pkg.nrm.primitives.kelvin_primitive import KelvinNRMPrimitive, KelvinSemanticClass
from instant_nurec._pkg.utils.types import RigTrajectories


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

    def export(self, output_path: Path):
        """
        Export the gaussians to the PLY file used in SO, and values are exported as NuRec expects them.
        Colors are exported in SH, while scale and density are exported preactivated.
        Hardcoded inverse activations: scale uses exp (inverse: log), density
        uses sigmoid (inverse: log(x/(1-x))).
        """
        scales = torch.log(self.scales.float())
        densities = torch.log(self.densities.float() / (1.0 - self.densities.float()))

        rgb = self.rgb.float()

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
            features_albedo=RGB2SH(rgb),
            color=rgb,
            normals=self.normals.float() if self.normals is not None else None,
            custom_attributes=custom_attributes,
        )


def export_kelvin_ply(primitives: KelvinNRMPrimitive) -> PLYExportGaussians:
    """Export the ply file, static layer only."""
    static_layer = primitives.static_layer

    finite_mask = torch.isfinite(static_layer.densities).squeeze(-1)
    logger.info(f"Removed {(~finite_mask).sum().item():,} non-finite density gaussians.")
    static_layer = static_layer.mask(finite_mask)
    logger.info(f"Exporting {static_layer.densities.numel():,} gaussians.")

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


def export_ply(primitives: KelvinNRMPrimitive, rig_trajectories: RigTrajectories, path: Path) -> None:
    """Export the NRM Primitives as a ply file after transforming to world space and applying some filtering.
    This ply export is intended to be used as an initialization for NuRec SO.
    """
    # First transform the primitives to the world frame.
    # Self-invented: NRE relied on Lightning's batch transfer to move
    # rig_trajectories.T_world_base onto the primitive's device; the standalone
    # predict loop moves it explicitly here.
    primitives = primitives.rigid_transform(
        rig_trajectories.T_world_base.to(device=primitives.device(), dtype=torch.float32)
    )

    gaussians_ply = export_kelvin_ply(primitives)
    gaussians_ply.export(path)
