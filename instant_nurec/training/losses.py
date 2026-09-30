# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from instant_nurec.config_schema.train import TrainingLossConfig
from instant_nurec.model.supervision import SupervisionPack
from instant_nurec.primitives.kelvin_primitive import KelvinSemanticClass
from instant_nurec.training.renderer import RenderOutput
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.types import RayFlags


@dataclass(kw_only=True, slots=True)
class LossReturn:
    total_value: torch.Tensor
    values: dict[str, torch.Tensor]
    skipped: tuple[str, ...]


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def _weighted_masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Zero weighted-out values while retaining them in the denominator."""

    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    expanded_mask = mask.expand_as(value)
    expanded_weights = weights.expand_as(value)
    return (value * expanded_mask * expanded_weights).sum() / expanded_mask.sum().clamp_min(1)


def _quantile_mean(value: torch.Tensor, quantile: float = 0.98) -> torch.Tensor:
    flat = value.flatten()
    keep = int(flat.numel() * quantile)
    if keep == 0:
        return value.new_tensor(0.0)
    return flat[flat.argsort()[:keep]].mean()


def _semantic_targets(labels) -> torch.Tensor:
    flags = unpack_optional(labels.flags)
    targets = torch.full(flags.shape[:-1], int(KelvinSemanticClass.OTHERS), dtype=torch.long, device=flags.device)
    # Reference priority (last write wins): ROAD < SKY < MOVABLE < EGO.
    targets[labels.get_mask_flags_all(RayFlags.ROAD_SEMANTIC).squeeze(-1)] = int(KelvinSemanticClass.ROAD)
    targets[labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC).squeeze(-1)] = int(KelvinSemanticClass.SKY)
    targets[labels.get_mask_flags_all(RayFlags.VEHICLE_SEMANTIC).squeeze(-1)] = int(KelvinSemanticClass.MOVABLE)
    targets[labels.get_mask_flags_all(RayFlags.EGO_SEMANTIC).squeeze(-1)] = int(KelvinSemanticClass.EGO)
    return targets


def _required(value, name: str):
    if value is None:
        raise RuntimeError(f"Configured training loss requires {name}, but it is unavailable")
    return value


class TrainingLosses(torch.nn.Module):
    """Loss profile for the two-phase full-model training recipe."""

    def __init__(self, config: TrainingLossConfig):
        super().__init__()
        self.config = config
        self._lpips = None
        if config.lpips > 0:
            try:
                import lpips
            except ImportError as exc:
                raise RuntimeError("LPIPS render loss requires the training extra") from exc
            self._lpips = lpips.LPIPS(net="vgg")
            self._lpips.eval()
            for parameter in self._lpips.parameters():
                parameter.requires_grad = False

    def _context_losses(
        self,
        pack: SupervisionPack,
        context: DataAndRenderingBatch,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        labels = unpack_optional(context.data.camera).labels
        rendering = unpack_optional(unpack_optional(context.rendering).camera)
        values: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        if self.config.primitive_rgb > 0:
            context_rgb = _required(pack.context_rgb, "model context RGB")
            reference_rgb = _required(labels.rgb, "context RGB labels")
            values["primitive_rgb"] = (context_rgb - reference_rgb).abs().mean()

        geometry_enabled = self.config.primitive_distance > 0 or self.config.primitive_distance_gradient > 0
        if geometry_enabled:
            gt_distance = _required(labels.metric_distance, "context metric-distance labels")
            context_depth = _required(pack.context_depth, "model context depth")
            pred_distance = context_depth / rendering.distance_to_depth_scale
            valid = (
                (gt_distance > self.config.distance_min_m)
                & (gt_distance < self.config.primitive_distance_max_m)
                & labels.get_mask_flags_none(RayFlags.INVALID)
            )
            if valid.any():
                if self.config.primitive_distance > 0:
                    values["primitive_distance"] = _quantile_mean((pred_distance - gt_distance).abs()[valid])
                if self.config.primitive_distance_gradient > 0:
                    gradient_valid = valid & (
                        gt_distance < self.config.primitive_distance_gradient_max_m
                    )
                    gradient_terms = []
                    for stride in (1, 2, 4, 8):
                        pred = pred_distance[:, ::stride, ::stride]
                        gt = gt_distance[:, ::stride, ::stride]
                        mask = gradient_valid[:, ::stride, ::stride]
                        residual = pred - gt
                        du = torch.diff(residual, dim=2, prepend=residual[:, :, :1]).clamp(-100, 100)
                        dv = torch.diff(residual, dim=1, prepend=residual[:, :1]).clamp(-100, 100)
                        mask_u = mask & torch.roll(mask, 1, dims=2)
                        mask_v = mask & torch.roll(mask, 1, dims=1)
                        mask_u[:, :, 0] = False
                        mask_v[:, 0] = False
                        if mask_u.any():
                            gradient_terms.append(du[mask_u.expand_as(du)].abs())
                        if mask_v.any():
                            gradient_terms.append(dv[mask_v.expand_as(dv)].abs())
                    if gradient_terms:
                        values["primitive_distance_gradient"] = torch.cat(gradient_terms).mean()
            else:
                skipped.extend(
                    name
                    for name, weight in (
                        ("primitive_distance", self.config.primitive_distance),
                        ("primitive_distance_gradient", self.config.primitive_distance_gradient),
                    )
                    if weight > 0
                )
        if self.config.primitive_semantics > 0:
            semantic_logits = _required(pack.context_semantic_logits, "model context semantic logits")
            _required(labels.flags, "context semantic flags")
            targets = _semantic_targets(labels)
            values["primitive_semantics"] = F.cross_entropy(
                semantic_logits.moveaxis(-1, 1),
                targets,
            )

        if self.config.primitive_normal > 0 and pack.context_world_normal is not None and labels.normals is not None:
            valid = labels.get_mask_flags_all(RayFlags.VALID_NORMAL)
            valid &= labels.get_mask_flags_none(RayFlags.SKY_SEMANTIC | RayFlags.INVALID)
            if valid.any():
                cosine = (pack.context_world_normal * labels.normals).sum(dim=-1)
                values["primitive_normal"] = (1.0 - cosine)[valid.squeeze(-1)].mean()
            else:
                skipped.append("primitive_normal")
        elif self.config.primitive_normal > 0:
            skipped.append("primitive_normal")

        if self.config.primitive_velocity > 0 and pack.motion_supervisions:
            valid_motion = [motion for motion in pack.motion_supervisions if motion.reference_flow is not None]
            if valid_motion:
                pred = torch.cat([motion.context_flow for motion in valid_motion], dim=-1)
                target = torch.cat([unpack_optional(motion.reference_flow) for motion in valid_motion], dim=-1)
                moving = torch.linalg.vector_norm(target.reshape(*target.shape[:-1], -1, 3), dim=-1).amax(dim=-1) > 0.01
                if moving.any():
                    vehicle = labels.get_mask_flags_all(RayFlags.VEHICLE_SEMANTIC).squeeze(-1)
                    weights = torch.where(vehicle, 1.0, 0.1)
                    values["primitive_velocity"] = _weighted_masked_mean(
                        (pred - target).abs(),
                        moving,
                        weights,
                    )
                else:
                    skipped.append("primitive_velocity")
            else:
                skipped.append("primitive_velocity")
        elif self.config.primitive_velocity > 0:
            skipped.append("primitive_velocity")

        if self.config.primitive_sky_cubemap > 0:
            predicted_sky = _required(pack.predicted_sky_cubemap, "model sky cubemap")
            reference_sky = _required(pack.reference_sky_cubemap, "reference sky cubemap")
            reference_sky_mask = _required(pack.reference_sky_cubemap_mask, "reference sky cubemap mask")
            predicted_smoothed = gaussian_blur(
                predicted_sky.permute(0, 3, 1, 2),
                kernel_size=[29, 29],
            ).permute(0, 2, 3, 1)
            target = reference_sky.clone()
            target_mask = reference_sky_mask[..., 0].bool()
            target[~target_mask] = predicted_smoothed[~target_mask]
            values["primitive_sky_cubemap"] = (predicted_sky - target).abs().mean()
        return values, skipped

    def _render_losses(
        self,
        output: RenderOutput | None,
        supervision: DataAndRenderingBatch | None,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        values: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        enabled = {
            "rgb": self.config.rgb,
            "lpips": self.config.lpips,
            "distance": self.config.distance,
            "background": self.config.background,
        }
        if output is None or supervision is None:
            configured = [name for name, weight in enabled.items() if weight > 0]
            if configured:
                raise RuntimeError(
                    "Configured render losses require rendered supervision; "
                    f"missing output for: {', '.join(configured)}"
                )
            return values, skipped
        labels = unpack_optional(supervision.data.camera).labels
        valid = labels.get_mask_flags_all(RayFlags.RGB_LABEL) & labels.get_mask_flags_none(RayFlags.INVALID)
        synthetic = labels.get_mask_flags_all(RayFlags.SYNTHETIC)
        if self.config.rgb > 0:
            reference_rgb = _required(labels.rgb, "supervision RGB labels")
            valid_compact = valid.squeeze(-1)
            synthetic_compact = synthetic.squeeze(-1)[valid_compact]
            predicted_rgb = output.rgb.reshape_as(reference_rgb)
            squared_error = F.mse_loss(
                predicted_rgb[valid_compact],
                reference_rgb[valid_compact],
                reduction="none",
            )
            nonsynthetic_term = squared_error * (~synthetic_compact)[:, None]
            synthetic_term = 0.25 * squared_error * synthetic_compact[:, None]
            values["rgb"] = (nonsynthetic_term + synthetic_term).mean()

        if self.config.lpips > 0:
            reference_rgb = _required(labels.rgb, "supervision RGB labels")
            pred = torch.where(valid.expand_as(output.rgb), output.rgb, reference_rgb)
            pred = pred.permute(0, 3, 1, 2) * 2 - 1
            target = reference_rgb.permute(0, 3, 1, 2) * 2 - 1
            longest = max(pred.shape[-2:])
            if longest > 512:
                scale = 512 / longest
                shape = (int(pred.shape[-2] * scale), int(pred.shape[-1] * scale))
                pred = F.interpolate(pred, shape, mode="bilinear", align_corners=False)
                target = F.interpolate(target, shape, mode="bilinear", align_corners=False)
            values["lpips"] = self._lpips(pred, target).mean()

        if self.config.distance > 0 and labels.metric_distance is not None:
            gt = labels.metric_distance
            distance_valid = (
                (gt > self.config.distance_min_m)
                & (gt < self.config.render_distance_max_m)
                & labels.get_mask_flags_none(RayFlags.INVALID | RayFlags.HARMONIZED)
            )
            if distance_valid.any():
                expected = output.distance / output.opacity.clamp_min(1.0e-6)
                error = (
                    expected.clamp_min(self.config.distance_min_m).reciprocal()
                    - gt.clamp_min(self.config.distance_min_m).reciprocal()
                ).square()
                values["distance"] = _weighted_masked_mean(
                    error,
                    distance_valid,
                    ~synthetic,
                )
            else:
                skipped.append("distance")
        elif self.config.distance > 0:
            # Render distance is one of the official allow-missing losses.
            skipped.append("distance")

        if self.config.background > 0:
            sky = labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC)
            bg_valid = labels.get_mask_flags_none(
                RayFlags.INVALID | RayFlags.HARMONIZED | RayFlags.SYNTHETIC
            )
            if bg_valid.any():
                foreground_target = (~sky).float()
                values["background"] = _masked_mean(
                    (output.opacity.clamp(0, 1) - foreground_target).square(), bg_valid
                )
            else:
                skipped.append("background")
        return values, skipped

    def forward(
        self,
        pack: SupervisionPack,
        context: DataAndRenderingBatch,
        output: RenderOutput | None = None,
        supervision: DataAndRenderingBatch | None = None,
    ) -> LossReturn:
        context_values, context_skipped = self._context_losses(pack, context)
        render_values, render_skipped = self._render_losses(output, supervision)
        values = context_values | render_values
        weights = self.config.model_dump()
        if not values:
            raise RuntimeError("No training loss is active for this batch; check labels and loss weights")
        total = sum(value * float(weights[name]) for name, value in values.items())
        return LossReturn(total_value=total, values=values, skipped=tuple(context_skipped + render_skipped))
