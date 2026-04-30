# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Literal, Optional, Tuple

import torch

from einops import rearrange, repeat
from torchvision.transforms.functional import gaussian_blur

from libs.losses.models.base_losses import BaseLossWithSemanticWeights, BasePrimitiveLoss
from libs.losses.models.registry import register_loss, register_loss_fn
from libs.losses.models.utils import get_mask_semantic
from libs.losses.orchestration.config import LossItemConfig, LossReturn
from nre.models.base import BaseModel
from nre.nrm.models.base import BaseNRMSupervisionPack
from nre.nrm.models.celsius_model import CelsiusNRM, CelsiusNRMSupervisionPack
from nre.nrm.models.kelvin_backbone.base import KelvinNRMSupervisionPack
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.nrm.primitives.celsius_primitive import CelsiusNRMPrimitive
from nre.nrm.primitives.kelvin_primitive import KelvinNRMPrimitive, KelvinSemanticClass
from nre.utils.batch import CameraFrameLabels, DataAndRenderingBatch
from nre.utils.misc import unpack_optional
from nre.utils.trainer import TrainerConfig
from nre.utils.types import RayFlags


class PrimitiveGeometryBaseLoss(BasePrimitiveLoss, BaseLossWithSemanticWeights):
    """
    Future design ideas:
    - Add a "loss context" if we want to avoid re-computation of the point maps etc.
    - Add a "scaling" parameter to apply_loss_fn if we want the loss's denominator to be full pixels instead of masked ones.
    """

    @staticmethod
    def _get_distance_to_depth_scale(context: DataAndRenderingBatch) -> torch.Tensor:
        return unpack_optional(unpack_optional(context.rendering).camera).distance_to_depth_scale

    def __init__(
        self,
        loss_name: str,
        config: LossItemConfig,
        trainer_config: TrainerConfig,
        geometry_target: Literal["distance", "depth", "pointmap"],
        **kwargs,
    ) -> None:
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.geometry_target = geometry_target
        self.min_distance: float = unpack_optional(config.min_distance)
        self.max_distance: float = unpack_optional(config.max_distance)
        self.sky_distance: float | None = config.sky_distance

    def _compute_geometry(
        self,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        assert (camera := context.data.camera) is not None, "Camera data is required for primitive distance loss"

        sky_mask: torch.Tensor | None = None
        if self.sky_distance is not None:
            sky_mask = camera.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC)

        rays = unpack_optional(unpack_optional(context.rendering).camera).rays

        gt_value = gt_distance = unpack_optional(camera.labels.metric_distance)
        if sky_mask is not None and self.sky_distance is not None:
            gt_value = gt_value.masked_fill(sky_mask, self.sky_distance)
        if self.geometry_target == "depth":
            gt_value = gt_value * PrimitiveGeometryBaseLoss._get_distance_to_depth_scale(context)
        elif self.geometry_target == "pointmap":
            gt_value = rays[..., :3] + gt_value * rays[..., 3:]

        pred_conf: torch.Tensor | None = None
        match supervision_pack:
            case CelsiusNRMSupervisionPack():
                pred_value = unpack_optional(supervision_pack.context_distance)
                if self.geometry_target == "depth":
                    pred_value = pred_value * PrimitiveGeometryBaseLoss._get_distance_to_depth_scale(context)
                elif self.geometry_target == "pointmap":
                    pred_value = rays[..., :3] + pred_value * rays[..., 3:]
            case KelvinNRMSupervisionPack():
                if self.geometry_target == "pointmap":
                    pred_value = unpack_optional(supervision_pack.context_xyz)
                else:
                    # For geometry_target == depth or distance...
                    pred_value = unpack_optional(supervision_pack.context_depth)
                    pred_conf = unpack_optional(supervision_pack.context_depth_conf)
                    if self.geometry_target != "depth":
                        pred_value = pred_value / PrimitiveGeometryBaseLoss._get_distance_to_depth_scale(context)
                    # NB[JH]: Create new loss class if we want pred_pointmap to be bound to context_depth.
            case _:
                raise TypeError(
                    f"Unsupported supervision pack type {type(supervision_pack).__name__} for context distance calculation"
                )

        mask_valid = (gt_distance > self.min_distance) & (gt_distance < self.max_distance)
        if sky_mask is not None:
            mask_valid = mask_valid | sky_mask
        mask_valid &= camera.labels.get_mask_flags_none(RayFlags.INVALID)  # [batch, height, width, 1]

        return pred_value, pred_conf, gt_value, mask_valid

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        pred_value, pred_conf, gt_value, mask_valid = self._compute_geometry(supervision_pack, context)

        if not mask_valid.any():
            return None

        return self.apply_loss_fn(
            pred_value[mask_valid[..., 0]],
            gt_value[mask_valid[..., 0]],
            eps=self.min_distance,
            confidence=pred_conf[mask_valid] if pred_conf is not None else None,
            frame_labels=unpack_optional(context.data.camera).labels,
            frame_labels_mask=mask_valid,
        )


class PrimitiveGeometryGradientLoss(PrimitiveGeometryBaseLoss):
    def __init__(
        self,
        loss_name: str,
        config: LossItemConfig,
        trainer_config: TrainerConfig,
        geometry_target: Literal["distance", "depth", "pointmap"],
        **kwargs,
    ) -> None:
        super().__init__(loss_name, config, trainer_config, geometry_target, **kwargs)
        self.scale_steps: list[int] = unpack_optional(config.scale_steps)
        self.scale_lambdas: list[float] = unpack_optional(config.scale_lambdas)
        assert len(self.scale_steps) == len(self.scale_lambdas), (
            "scale_steps and scale_lambdas must have the same length"
        )

    def _get_gradient_residual(
        self,
        pred_value: torch.Tensor,
        gt_value: torch.Tensor,
        gt_mask: torch.Tensor,
        clamp_value: float = 100.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the gradient residual of the predicted value and the ground truth value:

        Args:
          - pred_value: (..., H, W, C)
          - gt_value: (..., H, W, C)
          - gt_mask: (..., H, W, 1)

        Returns:
          - grad_u: (..., H, W, C), where grad_u = (pred_u - pred_u-) - (gt_u - gt_u-)
          - grad_v: (..., H, W, C), where grad_v = (pred_v - pred_v-) - (gt_v - gt_v-)
          - mask_u: (..., H, W), where mask_u = gt_mask[..., :, 1:] & gt_mask[..., :, :-1]
          - mask_v: (..., H, W), where mask_v = gt_mask[..., 1:, :] & gt_mask[..., :-1, :]
        """
        diff = pred_value - gt_value
        grad_u = torch.diff(diff, dim=-2, prepend=diff[..., :, :1, :]).clamp(min=-clamp_value, max=clamp_value)
        grad_v = torch.diff(diff, dim=-3, prepend=diff[..., :1, :, :]).clamp(min=-clamp_value, max=clamp_value)

        gt_mask = gt_mask.squeeze(-1)
        mask_u = torch.mul(gt_mask[..., :, 1:], gt_mask[..., :, :-1])
        mask_u = torch.nn.functional.pad(mask_u, (1, 0, 0, 0), value=False)
        mask_v = torch.mul(gt_mask[..., 1:, :], gt_mask[..., :-1, :])
        mask_v = torch.nn.functional.pad(mask_v, (0, 0, 1, 0), value=False)

        return grad_u, grad_v, mask_u, mask_v

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        pred_value, pred_conf, gt_value, mask_valid = self._compute_geometry(supervision_pack, context)

        if not mask_valid.any():
            return None

        # Compute gradient loss at multiple scales
        grad_diffs: list[torch.Tensor] = []
        grad_confs: list[torch.Tensor] = []

        for scale_step, scale_lambda in zip(self.scale_steps, self.scale_lambdas):
            pred_value_scaled = pred_value[..., ::scale_step, ::scale_step, :]
            gt_value_scaled = gt_value[..., ::scale_step, ::scale_step, :]
            mask_valid_scaled = mask_valid[..., ::scale_step, ::scale_step, :]
            grad_u, grad_v, mask_u, mask_v = self._get_gradient_residual(
                pred_value_scaled, gt_value_scaled, mask_valid_scaled
            )

            # Perform reduction here for better efficiency
            inds_u = torch.where(mask_u)
            inds_v = torch.where(mask_v)
            grad_diffs.extend([grad_u[inds_u] * scale_lambda, grad_v[inds_v] * scale_lambda])
            if pred_conf is not None:
                pred_conf_scaled = pred_conf[..., ::scale_step, ::scale_step, :]
                grad_confs.extend([pred_conf_scaled[inds_u], pred_conf_scaled[inds_v]])

        return self.apply_loss_fn(
            torch.cat(grad_diffs, dim=0),
            confidence=torch.cat(grad_confs, dim=0).squeeze(-1) if pred_conf is not None else None,
            frame_labels=unpack_optional(context.data.camera).labels,
            # TODO: In case where this is needed, it should be torch.cat(inds_us, inds_vs)
            frame_labels_mask=None,
        )


@register_loss("primitive_distance")
class PrimitiveDistanceLoss(PrimitiveGeometryBaseLoss):
    """
    Loss that compares distance directly regressed from the model and the ground truth
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        self.use_z_depth: bool = config.use_z_depth
        super().__init__(
            "primitive_distance", config, trainer_config, "depth" if self.use_z_depth else "distance", **kwargs
        )


@register_loss("primitive_distance_gradient")
class PrimitiveDistanceGradientLoss(PrimitiveGeometryGradientLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        self.use_z_depth: bool = config.use_z_depth
        super().__init__(
            "primitive_distance_gradient", config, trainer_config, "depth" if self.use_z_depth else "distance", **kwargs
        )


@register_loss("primitive_pointmap")
class PrimitivePointmapLoss(PrimitiveGeometryBaseLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_pointmap", config, trainer_config, "pointmap", **kwargs)


@register_loss("primitive_pointmap_gradient")
class PrimitivePointmapGradientLoss(PrimitiveGeometryGradientLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_pointmap_gradient", config, trainer_config, "pointmap", **kwargs)


@register_loss("primitive_rgb")
class PrimitiveRgbLoss(BasePrimitiveLoss, BaseLossWithSemanticWeights):
    """
    This directly supervises the output from the rgb branches with the gt rgb.
    For flexibility, this loss would not limit the region to valid pixels by default.
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_rgb", config, trainer_config, **kwargs)

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := context.data.camera) is None:
            return None

        labels = camera.labels
        context_rgb: torch.Tensor | None = None
        match (primitive, supervision_pack):
            case (CelsiusNRMPrimitive(), CelsiusNRMSupervisionPack()):
                context_rgb = primitive.rgb.reshape(
                    unpack_optional(labels.b), unpack_optional(labels.h), unpack_optional(labels.w), 3
                )
            case (_, KelvinNRMSupervisionPack()):
                if (context_rgb := supervision_pack.context_rgb) is None:
                    return None
            case _:
                raise TypeError(f"Unsupported primitive type {type(primitive).__name__} for context rgb calculation")

        return self.apply_loss_fn(
            context_rgb,
            unpack_optional(labels.rgb),
            frame_labels=labels,
            frame_labels_mask=None,
        )


@register_loss("primitive_velocity")
class PrimitiveVelocityLoss(BasePrimitiveLoss, BaseLossWithSemanticWeights):
    """
    Celsius: predicted context velocity vs labels.velocity.
    Kelvin: predicted context_flow vs cuboid-warp reference_flow (full grid, all rays); raises if reference is missing.
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_velocity", config, trainer_config, **kwargs)
        self.velocity_mask_threshold: float | None = config.velocity_mask_threshold

    @staticmethod
    def _aggregate_velocity_mask(velocity_mask_list: list[torch.Tensor]) -> torch.Tensor:
        assert len(velocity_mask_list) > 0, "Velocity mask list is empty"
        velocity_mask = torch.zeros_like(velocity_mask_list[0])
        for mask in velocity_mask_list:
            velocity_mask |= mask
        return velocity_mask

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := context.data.camera) is None:
            return None
        labels = camera.labels

        pred_velocity: torch.Tensor | None = None
        gt_velocity: torch.Tensor | None = None
        velocity_mask: torch.Tensor | None = None
        match supervision_pack:
            case CelsiusNRMSupervisionPack():
                if labels is None or labels.velocity is None or supervision_pack.context_velocity is None:
                    return None

                gt_velocity = labels.velocity
                pred_velocity = supervision_pack.context_velocity.reshape_as(gt_velocity)
                assert self.velocity_mask_threshold is None, (
                    "Velocity mask threshold is not supported for Celsius model"
                )

            case KelvinNRMSupervisionPack():
                # Skip loss if supervision information is not provided.
                if len(supervision_pack.motion_supervisions) == 0:
                    return None

                # Tensor to compare will be of dim [B, H, W, n_ms * 3]
                gt_velocity = torch.cat(
                    [unpack_optional(ms.reference_flow) for ms in supervision_pack.motion_supervisions], dim=-1
                )
                pred_velocity = torch.cat([ms.context_flow for ms in supervision_pack.motion_supervisions], dim=-1)
                if self.velocity_mask_threshold is not None:
                    velocity_mask = self._aggregate_velocity_mask(
                        [
                            torch.linalg.norm(unpack_optional(ms.reference_flow), dim=-1) > self.velocity_mask_threshold
                            for ms in supervision_pack.motion_supervisions
                        ]
                    )
                    # Early return to avoid pure static scenes.
                    if not velocity_mask.any():
                        return None
            case _:
                raise TypeError(
                    f"Unsupported supervision pack {type(supervision_pack).__name__} for primitive_velocity"
                )

        if velocity_mask is not None:
            pred_velocity = pred_velocity[velocity_mask]
            gt_velocity = gt_velocity[velocity_mask]

        return self.apply_loss_fn(pred_velocity, gt_velocity, frame_labels=labels, frame_labels_mask=velocity_mask)


@register_loss("primitive_sky_distance")
class PrimitiveSkyDistanceLoss(BasePrimitiveLoss):
    """This directly supervises the output from the sky branches with the gt sky distance."""

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        loss_name = "primitive_sky_distance"
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.sky_distance = unpack_optional(config.sky_distance)

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        # Get ground-truth
        assert (camera := context.data.camera) is not None, "Camera data is required for primitive distance loss"

        # Get prediction [batch, height, width, 1]
        match supervision_pack:
            case CelsiusNRMSupervisionPack():
                pred_distance = unpack_optional(supervision_pack.context_distance)
            case _:
                raise TypeError(
                    f"Unsupported supervision pack type {type(supervision_pack).__name__} for context distance calculation"
                )

        sky_loss_mask = camera.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC) & camera.labels.get_mask_flags_none(
            RayFlags.INVALID
        )  # [batch, height, width, 1]

        if not sky_loss_mask.any():
            return None

        pred_value = pred_distance[sky_loss_mask]
        return self.apply_loss_fn(
            pred_value,
            torch.full_like(pred_value, self.sky_distance),
            eps=1.0,
        )


@register_loss("primitive_mask")
class PrimitiveMaskLoss(BasePrimitiveLoss):
    """Supervise so that a specific output from the primitive matches gt mask"""

    def __init__(
        self,
        config: LossItemConfig,
        trainer_config: TrainerConfig,
        name_override: str | None = None,
        semantic_class_override: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__("primitive_mask" if name_override is None else name_override, config, trainer_config, **kwargs)
        self.semantic_class = (
            semantic_class_override if semantic_class_override is not None else unpack_optional(config.semantic_class)
        )

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        target_mask = get_mask_semantic(unpack_optional(context.data.camera).labels, self.semantic_class)

        match (primitive, supervision_pack):
            case (CelsiusNRMPrimitive(), CelsiusNRMSupervisionPack()):
                if (pred_mask := primitive.sky_mask) is None:
                    return None
                pred_mask = pred_mask.view(-1)
            case _:
                raise TypeError(
                    f"{self.__class__.__name__} Unsupported primitive type {type(primitive).__name__} & supervision pack {type(supervision_pack).__name__}"
                )

        return self.apply_loss_fn(pred_mask.reshape_as(target_mask), target_mask)


@register_loss("primitive_semantics")
class PrimitiveSemanticsLoss(BasePrimitiveLoss):
    """Supervise the semantic prediction with cross-entropy."""

    def __init__(
        self,
        config: LossItemConfig,
        trainer_config: TrainerConfig,
        **kwargs,
    ) -> None:
        super().__init__("primitive_semantics", config, trainer_config, **kwargs)

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if not isinstance(supervision_pack, KelvinNRMSupervisionPack):
            return None
        if (pred_semantic_logits := supervision_pack.context_semantic_logits) is None:
            return None

        camera = unpack_optional(context.data.camera)
        target_semantics = KelvinSemanticClass.get_target_from_frame_labels(camera.labels)

        return self.apply_loss_fn(
            pred_semantic_logits.moveaxis(-1, 1),  # [B, H, W, C] -> [B, C, H, W]
            target_semantics.squeeze(-1).long(),  # [B, H, W, 1] -> [B, H, W]
        )


@register_loss("primitive_normal")
class PrimitiveNormalLoss(BasePrimitiveLoss, BaseLossWithSemanticWeights):
    """Supervise the predicted context world-normal against ground-truth normals.

    Pixels are masked using `RayFlags.VALID_NORMAL` and exclude `INVALID` and `SKY_SEMANTIC`.
    Both predictions (already unit-normalized in the decoder) and targets are expected to be
    in world space.
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_normal", config, trainer_config, **kwargs)
        self.allow_missing_supervision: bool = config.allow_missing_supervision or False

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if not isinstance(supervision_pack, KelvinNRMSupervisionPack):
            return None
        if (pred_normal := supervision_pack.context_world_normal) is None:
            return None
        if (camera := context.data.camera) is None:
            return None
        if (gt_normal := camera.labels.normals) is None:
            if self.allow_missing_supervision:
                return None
            raise ValueError(f"[{self.__class__.__name__}] target labels should contain normal labels, but got None.")

        normal_loss_mask = (
            camera.labels.get_mask_flags_none(RayFlags.INVALID)
            & ~camera.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC)
            & camera.labels.get_mask_flags_all(RayFlags.VALID_NORMAL)
        ).reshape(-1)

        # normal_cosine collapses the channel dim, so value is [N] and mask must match.
        return self.apply_loss_fn(
            pred_normal.reshape(-1, 3),
            gt_normal.reshape(-1, 3),
            frame_labels=camera.labels,
            reduce_mask=normal_loss_mask.float(),
        )


@register_loss("primitive_sky_mask")
class PrimitiveSkyMaskLoss(PrimitiveMaskLoss):
    """This directly supervises the output from the sky branches with the gt sky mask."""

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__(
            config, trainer_config, name_override="primitive_sky_mask", semantic_class_override="sky", **kwargs
        )


@register_loss("primitive_sky_cubemap")
class PrimitiveSkyCubemapLoss(BasePrimitiveLoss):
    """This directly supervises the output from the sky branches with the gt sky cubemap."""

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_sky_cubemap", config, trainer_config, **kwargs)
        self.masked_region = config.masked_region

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if not isinstance(supervision_pack, KelvinNRMSupervisionPack):
            return None

        # Fill in empty pixels with the average color
        pd_cubemap = unpack_optional(supervision_pack.predicted_sky_cubemap)
        gt_cubemap = unpack_optional(supervision_pack.reference_sky_cubemap).clone()
        gt_cubemap_mask = unpack_optional(supervision_pack.reference_sky_cubemap_mask)[..., 0]

        if self.masked_region == "skip":
            # Directly apply mask to the loss, so we don't supervise that region.
            return self.apply_loss_fn(pd_cubemap, gt_cubemap, reduce_mask=gt_cubemap_mask)

        elif self.masked_region == "average":
            # For non-sky regions, use the average color of the sky region.
            # For scenes such as tunnel, mask is all empty, no need to apply this loss.
            if not gt_cubemap_mask.any():
                return None
            gt_cubemap[~gt_cubemap_mask] = gt_cubemap[gt_cubemap_mask].mean(dim=0)

        elif self.masked_region == "smooth":
            # For non-sky regions, try to make it as smooth as possible.
            # Note we cannot detach here as masked region will become black.
            pd_cubemap_smoothed = rearrange(pd_cubemap, "b h w c -> b c h w")
            pd_cubemap_smoothed = gaussian_blur(pd_cubemap_smoothed, kernel_size=[29, 29])
            pd_cubemap_smoothed = rearrange(pd_cubemap_smoothed, "b c h w -> b h w c")
            gt_cubemap[~gt_cubemap_mask] = pd_cubemap_smoothed[~gt_cubemap_mask]

        else:
            raise ValueError(f"Unsupported masked region method: {self.masked_region}")

        return self.apply_loss_fn(pd_cubemap, gt_cubemap)


@register_loss("primitive_flow_regularization")
class PrimitiveFlowRegularizationLoss(BasePrimitiveLoss):
    """This urges the flow to be zero and sparse.
    Ref: STORM (Yang et al. 2025)"""

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_flow_regularization", config, trainer_config, **kwargs)

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        match supervision_pack:
            case CelsiusNRMSupervisionPack():
                if (context_velocity := supervision_pack.context_velocity) is None:
                    return None
            case _:
                raise TypeError(
                    f"{self.__class__.__name__} Unsupported supervision pack type {type(supervision_pack).__name__}"
                )

        return self.apply_loss_fn(context_velocity, torch.zeros_like(context_velocity))


@register_loss("primitive_sigma_regularization")
class PrimitiveSigmaRegularizationLoss(BasePrimitiveLoss, BaseLossWithSemanticWeights):
    """This is used to make the falloff sigma as big as possible to avoid local minima."""

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_sigma_regularization", config, trainer_config, **kwargs)
        assert config.fn == "exp_growth", f"[{self.__class__.__name__}] is not compatible with other loss functions"

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        frame_labels: CameraFrameLabels | None = None
        if (camera := context.data.camera) is not None:
            frame_labels = camera.labels

        match supervision_pack:
            case CelsiusNRMSupervisionPack():
                if (falloff_sigma := supervision_pack.context_unscaled_falloff_sigma) is None:
                    return None
            case _:
                raise TypeError(
                    f"{self.__class__.__name__} Unsupported supervision pack type {type(supervision_pack).__name__}"
                )

        return self.apply_loss_fn(falloff_sigma.reshape(-1), frame_labels=frame_labels, frame_labels_mask=None)


@register_loss_fn("exp_growth")
def exp_growth_loss(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x)


@register_loss("primitive_sigma")
class PrimitiveSigmaLoss(BasePrimitiveLoss, BaseLossWithSemanticWeights):
    """This is used to make the falloff sigma essentially a classification of whether it's dynamic or not."""

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_sigma", config, trainer_config, **kwargs)

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := context.data.camera) is None:
            return None

        labels = camera.labels
        if (gt_velocity := labels.velocity) is None:
            return None

        match supervision_pack:
            case CelsiusNRMSupervisionPack():
                if (falloff_sigma := supervision_pack.context_unscaled_falloff_sigma) is None:
                    return None
            case _:
                raise TypeError(
                    f"{self.__class__.__name__} Unsupported supervision pack type {type(supervision_pack).__name__}"
                )

        return self.apply_loss_fn(
            falloff_sigma.reshape(-1),
            torch.all(gt_velocity.reshape(-1, gt_velocity.shape[-1]) < 0.1, dim=-1),
            frame_labels=labels,
            frame_labels_mask=None,
        )


@register_loss("primitive_visibility_regularization")
class PrimitiveVisibilityRegularizationLoss(BasePrimitiveLoss):
    """This is used to force the visibility for xyz prediction models."""

    needs_target = True

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("primitive_visibility_regularization", config, trainer_config, **kwargs)

    @staticmethod
    def _get_corner_rays(rays_cam_o: torch.Tensor, rays_cam_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the corner rays of the image.
        Args:
            rays_cam_o: (V, H, W, 3)
            rays_cam_d: (V, H, W, 3)
        Returns:
            corner_rays_cam_o: (V, 4, 3)
            corner_rays_cam_d: (V, 4, 3)
        """
        assert rays_cam_o.ndim == rays_cam_d.ndim
        assert rays_cam_o.shape == rays_cam_d.shape
        assert rays_cam_o.shape[-1] == 3
        # The corner rays are in clockwise order.
        corner_rays_cam_o = torch.stack(
            [rays_cam_o[:, 0, 0, :], rays_cam_o[:, 0, -1, :], rays_cam_o[:, -1, -1, :], rays_cam_o[:, -1, 0, :]], dim=1
        )
        corner_rays_cam_d = torch.stack(
            [rays_cam_d[:, 0, 0, :], rays_cam_d[:, 0, -1, :], rays_cam_d[:, -1, -1, :], rays_cam_d[:, -1, 0, :]], dim=1
        )
        return corner_rays_cam_o, corner_rays_cam_d

    @staticmethod
    def _get_clockwise_ray_pairs(
        corner_rays_cam_o: torch.Tensor, corner_rays_cam_d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the clockwise ray pairs for plane normal test.
        Args:
            corner_rays_cam_o: (V, 4, 3)
            corner_rays_cam_d: (V, 4, 3)
        Returns:
            ray_pairs_cam_o: (V, 4, 2, 3)
            ray_pairs_cam_d: (V, 4, 2, 3)
        """
        assert corner_rays_cam_o.ndim == corner_rays_cam_d.ndim
        assert corner_rays_cam_o.shape == corner_rays_cam_d.shape
        assert corner_rays_cam_o.shape[-2] == 4
        # Get the clockwise ray pairs of the image.
        # The ray pairs are in clockwise order.
        ray_pairs_cam_o = torch.stack([corner_rays_cam_o, corner_rays_cam_o.roll(shifts=-1, dims=1)], dim=2)
        ray_pairs_cam_d = torch.stack([corner_rays_cam_d, corner_rays_cam_d.roll(shifts=-1, dims=1)], dim=2)
        return ray_pairs_cam_o, ray_pairs_cam_d

    @staticmethod
    def _compute_dot_product_to_plane_normal(
        primitive_positions: torch.Tensor, ray_pairs_cam_o: torch.Tensor, ray_pairs_cam_d: torch.Tensor
    ) -> torch.Tensor:
        """Compute the dot product of the primitive positions to the plane normals.
        Args:
            primitive_positions: (N, 3)
            ray_pairs_cam_o: (V, 4, 2, 3)
            ray_pairs_cam_d: (V, 4, 2, 3)
        Returns:
            dot_products: (N, V, 4, 2)
        """
        assert ray_pairs_cam_o.ndim == ray_pairs_cam_d.ndim
        assert ray_pairs_cam_o.shape == ray_pairs_cam_d.shape
        assert ray_pairs_cam_o.shape[-2] == 2
        # Compute the normal of the plane.
        # The plane is formed by clockwise ray pairs.
        # The direction of the normal is thus inward.
        plane_normals = torch.cross(ray_pairs_cam_d[:, :, 0, :], ray_pairs_cam_d[:, :, 1, :], dim=-1)  # (V, 4, 3)
        # normalize the plane normals
        plane_normals = torch.nn.functional.normalize(plane_normals, dim=-1)
        # compute the vector from the primitive position to the first ray's origin
        V = ray_pairs_cam_o.shape[0]
        vectors = repeat(primitive_positions, "n three -> n v 4 three", v=V) - rearrange(
            ray_pairs_cam_o[:, :, 0], "v four three -> 1 v four three"
        )
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        # compute the dot product
        dot_products = torch.einsum("nvft,vft->nvf", (vectors, plane_normals))  # (N, V, 4)
        return dot_products

    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
        target: Optional[DataAndRenderingBatch] = None,
    ) -> Optional[LossReturn]:
        match primitive:
            case CelsiusNRMPrimitive():
                if isinstance(model, CelsiusNRM):
                    assert model.config.centroid_prediction == "xyz", (
                        f"{self.__class__.__name__} is only designed for centroid_prediction='xyz', "
                        f"but got centroid_prediction='{model.config.centroid_prediction}'"
                    )
                primitive_positions = primitive.positions
            case KelvinNRMPrimitive():
                rendering = unpack_optional(context.rendering)
                camera = unpack_optional(rendering.camera)
                timestamps_us = camera.timestamps_startend_us_cpu[0:1] if len(primitive.dynamic_layers) > 0 else None
                primitive_positions = primitive.get_gaussian_parameters(timestamps_us)["positions"]
            case _:
                raise TypeError(f"{self.__class__.__name__} Unsupported results type {type(primitive).__name__}")

        # Use supervision (target) rays when available so the frustum check covers all
        # rendered views, matching legacy tokengs which checks all V supervision views.
        rays_source = target if target is not None and target.rendering is not None else context
        rays_cam = unpack_optional(unpack_optional(rays_source.rendering).camera).rays
        rays_cam_o, rays_cam_d = rays_cam[..., :3], rays_cam[..., 3:]
        corner_rays_cam_o, corner_rays_cam_d = self._get_corner_rays(rays_cam_o, rays_cam_d)
        ray_pairs_cam_o, ray_pairs_cam_d = self._get_clockwise_ray_pairs(corner_rays_cam_o, corner_rays_cam_d)
        dot_products = self._compute_dot_product_to_plane_normal(primitive_positions, ray_pairs_cam_o, ray_pairs_cam_d)

        neg_dot_products = torch.relu(-dot_products)  # (N, V, 4)
        neg_dot_products = neg_dot_products.sum(dim=-1)  # (N, V)
        neg_dot_products = neg_dot_products.min(dim=-1).values

        return self.apply_loss_fn(
            neg_dot_products,
            torch.zeros_like(neg_dot_products),
        )
