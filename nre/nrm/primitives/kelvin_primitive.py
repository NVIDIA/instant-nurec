# Copyright (c) 2024-2026 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging

from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from typing import Self, Sequence

import torch


from nre.models.gaussians.renderers import BaseGaussianRenderer
from nre.nrm.config.models import PrimitiveExportPreprocessConfig
from nre.nrm.primitives.base import BaseGaussiansNRMPrimitive
from nre.nrm.utils.cubemap import rotate_sky_cubemap
from nre.utils.batch import CameraFrameLabels, DataAndRenderingBatch, LidarFrameLabels, RenderingData
from nre.utils.geometry import quat_mult_xyzw, so3_matrix_to_quat
from nre.utils.types import RayFlags, RigTrajectories


logger = logging.getLogger(__name__)


class KelvinSemanticClass(IntEnum):
    """Per-Gaussian semantic class IDs for Kelvin primitives.

    Channel index equals enum value. Targets come from ray flags via
    ``get_target_from_frame_labels``; predicted classes use argmax over logits in the decoder.
    """

    OTHERS = 0
    EGO = 1
    SKY = 2
    ROAD = 3
    MOVABLE = 4

    @classmethod
    def get_target_from_frame_labels(cls, rays_meta: CameraFrameLabels | LidarFrameLabels) -> torch.Tensor:
        """Compute per-pixel semantic class from frame label flags.

        Args:
            rays_meta: Labels with mask flags of shape (..., 1).

        Returns:
            Tensor of shape (..., 1) with dtype uint8 containing class IDs.
            Priority (last write wins): ROAD < SKY < MOVABLE < EGO.
        """
        mask_ego = rays_meta.get_mask_flags_all(RayFlags.EGO_SEMANTIC)
        mask_sky = rays_meta.get_mask_flags_all(RayFlags.SKY_SEMANTIC)
        mask_road = rays_meta.get_mask_flags_all(RayFlags.ROAD_SEMANTIC)
        mask_movable = rays_meta.get_mask_flags_all(RayFlags.VEHICLE_SEMANTIC)
        # (B, H, W, 1) uint8 class index; later overwrites take precedence
        out = torch.zeros((*mask_ego.shape[:-1], 1), dtype=torch.uint8, device=mask_ego.device)
        out[mask_road] = cls.ROAD
        out[mask_sky] = cls.SKY
        out[mask_movable] = cls.MOVABLE
        out[mask_ego] = cls.EGO
        return out

    @classmethod
    def opacity_mask_from_semantic_probs(cls, semantic_probs: torch.Tensor) -> torch.Tensor:
        """
        Opacity mask as intersection of non-ego and non-sky.
        semantic_probs: (..., C), C = number of semantic classes. Should be softmaxed.
        Returns (..., 1) in [0, 1] to multiply with gs_opacity.
        """
        ego = semantic_probs[..., cls.EGO : cls.EGO + 1]
        sky = semantic_probs[..., cls.SKY : cls.SKY + 1]
        return 1.0 - ego - sky

    @classmethod
    def semantics_to_rgb(cls, semantics: torch.Tensor) -> torch.Tensor:
        """
        Convert semantics (..., 1) uint8 class indices to RGB (..., 3). Colors from _VIS_COLORS.
        """
        assert semantics.shape[-1] == 1, f"semantics must have last dim 1, got shape {semantics.shape}"
        assert semantics.dtype == torch.uint8, f"semantics must be uint8, got {semantics.dtype}"
        colors = torch.tensor(_VIS_COLORS, dtype=torch.uint8, device=semantics.device)
        idx = semantics.squeeze(-1).long().clamp(0, len(colors) - 1)
        return colors[idx]


# RGB colors for KelvinSemanticClass.semantics_to_rgb (R, G, B) in [0, 255]; index = enum value
_VIS_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),  # OTHERS: black
    (255, 0, 0),  # EGO: red
    (0, 0, 255),  # SKY: blue
    (128, 128, 128),  # ROAD: gray
    (0, 255, 0),  # MOVABLE: green
)


@dataclass(kw_only=True)
class KelvinLayer:
    """
    Base class for all Gaussian layers that contains the following attribute:
        - rotations:            Rotation of each Gaussian represented as a unit quaternion         [n_gaussians, 4]
                                Note that this follows nrend's convention of wxyz format.
        - scales:               XYZ scale of each planar Gaussian                                  [n_gaussians, 3]
        - rgb:                  RGB color of each Gaussian                                         [n_gaussians, 3]
    """

    rotations: torch.Tensor
    scales: torch.Tensor
    rgb: torch.Tensor

    def __post_init__(self):
        assert self.rotations.shape == (len(self), 4), "Rotations must have shape (n_gaussians, 4)"
        assert self.scales.shape == (len(self), 3), "Scales must have shape (n_gaussians, 3)"
        assert self.rgb.shape == (len(self), 3), "RGB must have shape (n_gaussians, 3)"

    def device(self) -> torch.device:
        return self.rotations.device

    def __len__(self) -> int:
        return self.rotations.shape[0]

    @staticmethod
    def _validate_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
        quat_norm_invalid = torch.norm(quaternion, dim=-1) < 1.0e-6
        if (n_invalid := quat_norm_invalid.sum()) > 0:
            logger.warning(f"Found {n_invalid} invalid quaternions, setting them to identity.")
            quaternion = quaternion.clone()
            quaternion[quat_norm_invalid] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quaternion.device)
        return quaternion

    @torch.autocast(device_type="cuda", enabled=False)
    def rigid_transform(self, T_new: torch.Tensor) -> Self:
        assert (T_new := T_new.float()).shape == (4, 4), "Transform must have shape (4, 4)"
        q_new = so3_matrix_to_quat(T_new[:3, :3], unbatch=False)
        # When the rotations are invalid (e.g. all zeros), rotation will take no effect.
        rotations = self._validate_quaternion(self.rotations)
        rotations = quat_mult_xyzw(q_new.repeat(len(self), 1), rotations[:, [1, 2, 3, 0]])[:, [3, 0, 1, 2]]
        return replace(self, rotations=rotations)

    @classmethod
    def _concatenate_base(cls, layers: Sequence[KelvinLayer]) -> KelvinLayer:
        return cls(
            rotations=torch.cat([layer.rotations for layer in layers], dim=0),
            scales=torch.cat([layer.scales for layer in layers], dim=0),
            rgb=torch.cat([layer.rgb for layer in layers], dim=0),
        )


@dataclass(kw_only=True)
class KelvinStaticLayer(KelvinLayer):
    """
    Static layer that models the positions as fixed array:
        - positions:            Positions of the 3D Gaussians (x, y, z)                            [n_gaussians, 3]
        - densities:            Density of each Gaussian                                           [n_gaussians, 1]
        - semantic_class:       Per-Gaussian semantic class ID (see KelvinSemanticClass)            [n_gaussians, 1] uint8, optional
        - normals:              Per-Gaussian world-space surface normal (unit vector)              [n_gaussians, 3], optional
    """

    positions: torch.Tensor
    densities: torch.Tensor
    semantic_class: torch.Tensor | None = None
    normals: torch.Tensor | None = None

    def __post_init__(self):
        super().__post_init__()
        assert self.positions.shape == (len(self), 3), "Positions must have shape (n_gaussians, 3)"
        assert self.densities.shape == (len(self), 1), "Densities must have shape (n_gaussians, 1)"
        if self.semantic_class is not None:
            assert self.semantic_class.shape == (len(self), 1), "semantic_class must have shape (n_gaussians, 1)"
            assert self.semantic_class.dtype == torch.uint8, "semantic_class must be uint8"
        if self.normals is not None:
            assert self.normals.shape == (len(self), 3), "normals must have shape (n_gaussians, 3)"

    def __repr__(self) -> str:
        return f"StaticLayer(#GS={len(self) / 1e6:.2f}M)"

    @torch.autocast(device_type="cuda", enabled=False)
    def rigid_transform(self, T_new: torch.Tensor) -> Self:
        assert (T_new := T_new.float()).shape == (4, 4), "Transform must have shape (4, 4)"
        positions = (self.positions @ T_new[:3, :3].T) + T_new[:3, 3]
        normals = self.normals @ T_new[:3, :3].T if self.normals is not None else None
        return replace(super().rigid_transform(T_new), positions=positions, normals=normals)

    def mask(self, mask: torch.Tensor) -> Self:
        return replace(
            self,
            positions=self.positions[mask],
            densities=self.densities[mask],
            rotations=self.rotations[mask],
            scales=self.scales[mask],
            rgb=self.rgb[mask],
            semantic_class=self.semantic_class[mask] if self.semantic_class is not None else None,
            normals=self.normals[mask] if self.normals is not None else None,
        )

    @classmethod
    def concatenate(cls, layers: Sequence[Self]) -> Self:
        has_semantic = any(layer.semantic_class is not None for layer in layers)
        if has_semantic:
            semantic_class = torch.cat(
                [
                    layer.semantic_class
                    if layer.semantic_class is not None
                    else torch.zeros(len(layer), 1, dtype=torch.uint8, device=layer.device())
                    for layer in layers
                ],
                dim=0,
            )
        else:
            semantic_class = None
        has_normals = any(layer.normals is not None for layer in layers)
        if has_normals:
            normals = torch.cat(
                [
                    layer.normals if layer.normals is not None else torch.zeros(len(layer), 3, device=layer.device())
                    for layer in layers
                ],
                dim=0,
            )
        else:
            normals = None
        return cls(
            positions=torch.cat([layer.positions for layer in layers], dim=0),
            densities=torch.cat([layer.densities for layer in layers], dim=0),
            semantic_class=semantic_class,
            normals=normals,
            **asdict(KelvinLayer._concatenate_base(layers)),
        )

@dataclass(kw_only=True)
class KelvinDynamicLayer(KelvinLayer):
    """
    Dynamic layer that contains the following additional attribute that models the motion in a piecewise-linear fashion.
        - max_densities:            Maximum densities of each Gaussian                                    [n_gaussians, 1]
        - keyframe_positions:      Timed positions of the Gaussians (x, y, z)                         [n_gaussians, T, 3]
        - keyframe_timestamps_us:  Timestamps of each keyframe position (must be sorted)               [n_gaussians, T]
    Note that each layer can either represent a single actor or multiple actors.
    The Gaussian will additionally fade in/out in the first and last segments if falloff is True.
    """

    max_densities: torch.Tensor
    keyframe_positions: torch.Tensor
    keyframe_timestamps_us: torch.Tensor
    falloff: bool = True

    # Exponential factor for generalized gaussian
    # https://www.desmos.com/calculator/u1cbfaqj7g
    FALLOFF_GAUSSIAN_EXP = 10.0

    def __post_init__(self):
        super().__post_init__()
        assert self.n_keyframes > 1, "At least 2 keyframes are required"
        assert self.max_densities.shape == (len(self), 1), "Max densities must have shape (n_gaussians, 1)"
        assert self.keyframe_positions.shape == (len(self), self.n_keyframes, 3), (
            "Keyframe positions must have shape (n_gaussians, T, 3)"
        )
        assert self.keyframe_timestamps_us.shape == (len(self), self.n_keyframes), (
            "Keyframe timestamps must have shape (n_gaussians, T)"
        )
        assert not self.keyframe_timestamps_us.requires_grad, "Keyframe timestamps must not require gradients"

    @property
    def n_keyframes(self) -> int:
        return self.keyframe_positions.shape[1]


    def __repr__(self) -> str:
        return f"DynamicLayer(#GS={len(self) / 1e6:.2f}M, #KF={self.n_keyframes})"

    @torch.autocast(device_type="cuda", enabled=False)
    def rigid_transform(self, T_new: torch.Tensor) -> Self:
        assert (T_new := T_new.float()).shape == (4, 4), "Transform must have shape (4, 4)"
        keyframe_positions = (self.keyframe_positions @ T_new[:3, :3].T) + T_new[:3, 3]
        return replace(super().rigid_transform(T_new), keyframe_positions=keyframe_positions)

    def mask(self, mask: torch.Tensor) -> Self:
        return replace(
            self,
            rotations=self.rotations[mask],
            scales=self.scales[mask],
            rgb=self.rgb[mask],
            max_densities=self.max_densities[mask],
            keyframe_positions=self.keyframe_positions[mask],
            keyframe_timestamps_us=self.keyframe_timestamps_us[mask],
        )

    def ensure_minimum_density(self, minimum_density: float) -> Self:
        return replace(
            self,
            max_densities=torch.clamp(self.max_densities, min=minimum_density),
        )

    @classmethod
    def concatenate(cls, layers: Sequence[Self]) -> Self:
        return cls(
            max_densities=torch.cat([layer.max_densities for layer in layers], dim=0),
            keyframe_positions=torch.cat([layer.keyframe_positions for layer in layers], dim=0),
            keyframe_timestamps_us=torch.cat([layer.keyframe_timestamps_us for layer in layers], dim=0),
            falloff=layers[0].falloff,
            **asdict(KelvinLayer._concatenate_base(layers)),
        )

    def interpolate(self, timestamp_us: int) -> KelvinStaticLayer:
        assert timestamp_us >= 0, "Timestamp must be non-negative"

        # Obtain interpolation alpha (for edge segments alpha can be <0 or >1)
        query_timestamps = torch.tensor([timestamp_us], device=self.device())
        query_timestamps = query_timestamps.expand(len(self))[..., None]
        inds = torch.searchsorted(self.keyframe_timestamps_us, query_timestamps).clamp(1, self.n_keyframes - 1)
        left_timestamps = torch.gather(self.keyframe_timestamps_us, 1, inds - 1)
        right_timestamps = torch.gather(self.keyframe_timestamps_us, 1, inds)
        alpha = (query_timestamps - left_timestamps) / (right_timestamps - left_timestamps)

        # Interpolate positions
        left_positions = torch.gather(self.keyframe_positions, 1, (inds - 1).expand(-1, 3)[:, None])[:, 0]
        right_positions = torch.gather(self.keyframe_positions, 1, inds.expand(-1, 3)[:, None])[:, 0]
        positions = torch.lerp(left_positions, right_positions, alpha)

        # Obtain densities
        densities = self.max_densities
        if self.falloff:
            # Falloffs are only applied on edge segments.
            leftmost_inds = torch.where(inds == 1)[0]
            rightmost_inds = torch.where(inds == self.n_keyframes - 1)[0]
            falloff_factor = torch.ones_like(densities)
            # The *= operator is useful when we only have 1 segment.
            falloff_factor[leftmost_inds] *= 1 - torch.exp(
                -((1 + torch.clamp(alpha[leftmost_inds], min=-1.0)) ** self.FALLOFF_GAUSSIAN_EXP)
            )
            falloff_factor[rightmost_inds] *= torch.exp(-(alpha[rightmost_inds] ** self.FALLOFF_GAUSSIAN_EXP))
            densities = densities * falloff_factor

        return KelvinStaticLayer(
            positions=positions,
            densities=densities,
            rotations=self.rotations,
            scales=self.scales,
            rgb=self.rgb,
        )


class KelvinNRMPrimitive(BaseGaussiansNRMPrimitive):
    """
    Kelvin NRM primitive containing static and dynamic layers, with the following additional attributes:
        - sky_cubemap:         The sky cubemap for rendering the sky                            [6, height, width, 3]
                               Image order is (Right, Left, Top, Bottom, Front, Back) within NRM coordinate system.
        - affine_matrix:        The affine transform matrix                                        [n_cameras, 3, 4]
    """

    # Foregrounds
    static_layer: KelvinStaticLayer
    dynamic_layers: list[KelvinDynamicLayer]

    # Sky attributes
    sky_cubemap: torch.Tensor

    # Post-processing attributes
    affine_matrix: torch.Tensor

    def __init__(
        self,
        static_layer: KelvinStaticLayer,
        dynamic_layers: list[KelvinDynamicLayer],
        sky_cubemap: torch.Tensor,
        affine_matrix: torch.Tensor,
        use_2dgs: bool,
        gaussians_renderer: BaseGaussianRenderer | None,
    ):
        super().__init__(
            gaussians_renderer=gaussians_renderer,
            checkpointing="all",  # Maximum memory savings.
            shared_gaussian_parameters=len(dynamic_layers) == 0,
        )
        self.static_layer = static_layer
        self.dynamic_layers = dynamic_layers
        self.sky_cubemap = sky_cubemap
        self.affine_matrix = affine_matrix
        self.use_2dgs = use_2dgs
        self._post_init_validation()

    def _post_init_validation(self):
        assert self.sky_cubemap.ndim == 4, "Sky cubemap must have shape (6, height, width, 3)"
        cubemap_size = self.sky_cubemap.shape[1]
        assert self.sky_cubemap.shape == (6, cubemap_size, cubemap_size, 3), (
            "Sky cubemap must have shape (6, height, width, 3)"
        )
        assert self.affine_matrix.ndim == 3, "Affine matrix must have shape (n_cameras, 3, 4)"
        assert self.affine_matrix.shape[1:] == (3, 4), "Affine matrix must have shape (n_cameras, 3, 4)"

    def device(self) -> torch.device:
        return self.static_layer.device()

    def __len__(self) -> int:
        return len(self.static_layer) + sum(len(layer) for layer in self.dynamic_layers)

    def __repr__(self) -> str:
        return f"KelvinNRMPrimitive({repr(self.static_layer)}, {repr(self.dynamic_layers)})"

    @torch.autocast(device_type="cuda", enabled=False)
    def preprocess_for_export(
        self,
        context_batch: DataAndRenderingBatch,
        config: PrimitiveExportPreprocessConfig,
        context_rig: RigTrajectories | None = None,
    ) -> Self:
        """Filter static and dynamic layers by density threshold; do not apply rigid transform (merge does that)."""
        if getattr(config, "project_to_z_offset", False):
            raise NotImplementedError(
                "project_to_z_offset is not implemented for KelvinNRMPrimitive; use Celsius or set to false."
            )
        static_mask = self.static_layer.densities[:, 0] > config.density_prune_threshold
        new_static_layer = self.static_layer.mask(static_mask)
        new_dynamic_layers: list[KelvinDynamicLayer] = []
        for dynamic_layer in self.dynamic_layers:
            dynamic_mask = dynamic_layer.max_densities[:, 0] > config.density_prune_threshold
            new_dynamic_layers.append(dynamic_layer.mask(dynamic_mask))
        return self.__class__(
            static_layer=new_static_layer,
            dynamic_layers=new_dynamic_layers,
            sky_cubemap=self.sky_cubemap,
            affine_matrix=self.affine_matrix,
            use_2dgs=self.use_2dgs,
            gaussians_renderer=self.gaussians_renderer,
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def rigid_transform(self, T_new: torch.Tensor) -> Self:
        return self.__class__(
            static_layer=self.static_layer.rigid_transform(T_new),
            dynamic_layers=[layer.rigid_transform(T_new) for layer in self.dynamic_layers],
            sky_cubemap=rotate_sky_cubemap(self.sky_cubemap, T_new[:3, :3]),
            affine_matrix=self.affine_matrix,
            use_2dgs=self.use_2dgs,
            gaussians_renderer=self.gaussians_renderer,
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def color_transform(self, y: torch.Tensor) -> None:
        # Perform color transform in place
        assert y.shape == (3, 4), "Y must have shape (3, 4)"
        self.static_layer.rgb = self.static_layer.rgb @ y[:3, :3].T + y[:3, 3]
        for layer in self.dynamic_layers:
            layer.rgb = layer.rgb @ y[:3, :3].T + y[:3, 3]
        self.sky_cubemap = self.sky_cubemap @ y[:3, :3].T + y[:3, 3]

