# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math

from pathlib import Path
from typing import Literal

import torch

from pytorch_lightning import LightningModule
from torchmetrics.image import PeakSignalNoiseRatio

from instant_nurec.config_schema.train import KelvinTrainConfig
from instant_nurec.datasets.tracks import CuboidTracks, TrackFlags
from instant_nurec.model.kelvin import KelvinInstantNuRec
from instant_nurec.model.blocks.dav3 import convert_dav3_state_dict
from instant_nurec.training.losses import KelvinLossReturn, KelvinLosses
from instant_nurec.training.optim import CosineWithWarmupPBScheduler, make_kelvin_optimizer
from instant_nurec.training.renderer import render_kelvin_supervision
from instant_nurec.utils.batch import InstantNuRecDataBatch
from instant_nurec.utils.cubemap import unproject_to_sky_cubemap
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.motion import (
    associate_points_with_cuboid_tracks,
    warp_points_with_cuboid_tracks,
)
from instant_nurec.utils.types import RayFlags


logger = logging.getLogger(__name__)


class BroadcastExceptions:
    """Coordinate rank-local exceptions before DDP synchronization points."""

    def __init__(self, trainer):
        self.trainer = trainer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, traceback
        failed = self.trainer.strategy.reduce_boolean_decision(exc_value is not None, all=False)
        if failed:
            self.trainer.should_stop = True
            if exc_value is not None:
                return False
            raise RuntimeError(
                f"[rank{self.trainer.global_rank}] aborting training because another rank raised an exception"
            )
        return False


class KelvinTrainingSystem(LightningModule):
    """Lightning manual-optimization port of the Bazel Kelvin system."""

    automatic_optimization = False

    def __init__(self, config: KelvinTrainConfig):
        super().__init__()
        self.config = config
        self.model = KelvinInstantNuRec(config.model)
        self.loss = KelvinLosses(unpack_optional(config.loss))
        self.optimizer_implementation = "unconfigured"
        self._weights_initialized = False
        self._warned_skipped_losses: set[str] = set()
        self.save_hyperparameters(config.model_dump(mode="json"))

    def setup(self, stage: str) -> None:
        del stage
        # Keep independent state so Lightning can reset/synchronize train and
        # validation metrics with the same lifecycle as the Bazel system.
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.validation_psnr = PeakSignalNoiseRatio(data_range=1)

    def configure_optimizers(self):
        optimizer, implementation = make_kelvin_optimizer(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            self.config.system.optimizer,
        )
        self.optimizer_implementation = implementation
        scheduler = CosineWithWarmupPBScheduler(optimizer, self.config.system.scheduler)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @staticmethod
    def _read_raw_state_dict(path: Path) -> dict[str, torch.Tensor]:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint.get("state_dict", checkpoint)

    @classmethod
    def _read_full_state_dict(cls, path: Path) -> dict[str, torch.Tensor]:
        return {key.removeprefix("model."): value for key, value in cls._read_raw_state_dict(path).items()}

    def _initialize_from_dav3(self, path: Path) -> None:
        source = self._read_raw_state_dict(path)
        encoder_state = convert_dav3_state_dict(
            source,
            patch_embed_name="patch_embed_img",
            backbone_name="vit",
            dpt_reassemble_name=None,
            dpt_depth_head_name=None,
            dpt_rays_head_name=None,
            camera_encoder_name="embed_camera",
        )
        encoder_state.pop("vit.default_global_cls_tokens", None)
        self.model.encoder.load_state_dict(encoder_state, strict=True)

        decoder_state = convert_dav3_state_dict(
            source,
            patch_embed_name=None,
            backbone_name=None,
            dpt_reassemble_name="depth_head.reassemble",
            dpt_depth_head_name="depth_head.fusion_head",
            dpt_rays_head_name=None,
            camera_encoder_name=None,
        )
        context_init = [0.0] * 3 + [0.0, -1.0, 0.0] + [math.nan] * self.model.decoder.n_semantic_classes
        self.model.decoder.context_head.zero_init(context_init)
        self.model.decoder.gaussians_head.zero_init([math.nan] * 8)
        current_decoder_state = self.model.decoder.state_dict()
        current_decoder_state.update(decoder_state)
        self.model.decoder.load_state_dict(current_decoder_state, strict=True)
        logger.info("Initialized Kelvin encoder and depth head from DAv3 checkpoint %s.", path)

    def initialize_weights(self) -> None:
        paths = self.config.model.init_weights_paths
        if not paths:
            logger.warning("No initialization checkpoint configured; Kelvin is training from random weights.")
            if self.model.post_processing is not None:
                self.model.post_processing.zero_init()
            self._weights_initialized = True
            return
        full_key = next((key for key in ("full", "tokengs") if key in paths), None)
        if full_key is not None:
            state_dict = self._read_full_state_dict(Path(paths[full_key]))
            # Bazel always reinitializes the dense Gaussian head for full-model
            # and current-format TokenGS initialization, in either phase.
            state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith("decoder.gaussians_head.")
            }
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Initialized Kelvin from %s (%d missing, %d unexpected keys).",
                paths[full_key],
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        else:
            unsupported = set(paths) - {"dav3"}
            if unsupported:
                raise ValueError(f"Unsupported component initialization keys: {sorted(unsupported)}")
            if "dav3" not in paths:
                raise ValueError("Component initialization requires init_weights_paths.dav3")
            self._initialize_from_dav3(Path(paths["dav3"]))
        if self.model.post_processing is not None:
            self.model.post_processing.zero_init()
        self._weights_initialized = True

    def on_fit_start(self) -> None:
        # Lightning's sanity validation runs before ``on_train_start``.  Fresh
        # phase-one and phase-two weights must therefore be initialized here,
        # while restored runs keep the state loaded from their checkpoint.
        if not self._weights_initialized and not self.trainer.ckpt_path:
            self.initialize_weights()

    def on_train_batch_start(self, batch: InstantNuRecDataBatch, batch_idx: int) -> None:
        del batch, batch_idx
        self.model.update_step_train_batch_start(self.global_step)

    @staticmethod
    def _prepare_sky_reference(pack, supervision) -> None:
        camera = unpack_optional(supervision.data.camera)
        rendering = unpack_optional(unpack_optional(supervision.rendering).camera)
        if camera.labels.rgb is None or camera.labels.flags is None:
            return
        feature_mask = camera.labels.get_mask_flags_none(RayFlags.INVALID) & camera.labels.get_mask_flags_all(
            RayFlags.SKY_SEMANTIC
        )
        reference, mask = unproject_to_sky_cubemap(
            unpack_optional(pack.predicted_sky_cubemap).shape[1],
            tquat_to_se3_matrix(rendering.poses_tquat_startend[:, 1], unbatch=False)[:, :3, :3],
            rendering.sensor_model_parameters,
            camera.labels.rgb,
            feature_mask,
        )
        pack.reference_sky_cubemap = reference
        pack.reference_sky_cubemap_mask = mask

    def _prepare_motion_reference(self, pack, context, tracks: CuboidTracks | None) -> None:
        if tracks is None or not pack.motion_supervisions:
            pack.motion_supervisions = []
            return
        camera = unpack_optional(context.data.camera)
        rendering = unpack_optional(unpack_optional(context.rendering).camera)
        if camera.labels.metric_distance is None:
            pack.motion_supervisions = []
            return
        dynamic = CuboidTracks.Ops.subset_from_mask(tracks, tracks.tracks_flags & TrackFlags.DYNAMIC != 0)
        points = rendering.rays[..., :3] + camera.labels.metric_distance * rendering.rays[..., 3:]
        for motion in pack.motion_supervisions:
            tracks_idx = associate_points_with_cuboid_tracks(
                points=points,
                points_timestamps_us=motion.source_timestamps_us,
                points_dynamic_mask=None,
                cuboid_tracks=dynamic,
                cuboids_dims_padding=self.model.cuboids_dims_padding,
            )
            _, warped = warp_points_with_cuboid_tracks(
                points=points,
                source_timestamps_us=motion.source_timestamps_us,
                target_timestamps_us_list=[motion.target_timestamps_us],
                cuboid_tracks=dynamic,
                tracks_idx=tracks_idx,
            )
            motion.reference_flow = warped[0] - points

    @staticmethod
    def _render_psnr_inputs(render_output, supervision) -> tuple[torch.Tensor, torch.Tensor] | None:
        if render_output is None or supervision is None or supervision.data.camera is None:
            return None
        labels = supervision.data.camera.labels
        if labels.rgb is None:
            return None
        rgb_ray_mask = (
            labels.get_mask_flags_all(RayFlags.RGB_LABEL) & labels.get_mask_flags_none(RayFlags.INVALID)
        ).squeeze(-1)
        return render_output.rgb[rgb_ray_mask], labels.rgb[rgb_ray_mask]

    def _log_psnr(
        self,
        predicted_rgbs: list[torch.Tensor],
        ground_truth_rgbs: list[torch.Tensor],
        mode: Literal["train", "val"],
    ) -> None:
        if predicted_rgbs and ground_truth_rgbs:
            psnr_metric = self.train_psnr if mode == "train" else self.validation_psnr
            psnr_metric(torch.cat(predicted_rgbs, dim=0), torch.cat(ground_truth_rgbs, dim=0))
            self.log(f"{mode}/psnr", psnr_metric, prog_bar=True)
        elif mode == "val":
            # Phase one does not render.  Match Bazel's increasing placeholder
            # so val/psnr remains available to ModelCheckpoint.
            self.log("val/psnr", 0.1 * self.current_epoch, prog_bar=True)

    def forward_losses(self, batch: InstantNuRecDataBatch, mode: Literal["train", "val"]) -> KelvinLossReturn:
        batch.maybe_compute_rendering_data(device=self.device)
        tracks = (
            [CuboidTracks.Factory.from_pack(pack) for pack in batch.cuboid_tracks]
            if batch.cuboid_tracks is not None
            else None
        )
        batch.context = self.model.prepare_context(batch.context)
        primitives, packs = self.model.reconstruct_with_supervision(batch.context, tracks)
        per_item = []
        component_values: dict[str, list[torch.Tensor]] = {}
        skipped: list[str] = []
        predicted_rgbs: list[torch.Tensor] = []
        ground_truth_rgbs: list[torch.Tensor] = []
        for index, (primitive, pack, context) in enumerate(zip(primitives, packs, batch.context)):
            supervision = batch.supervision[index] if batch.supervision is not None else None
            if supervision is not None:
                self._prepare_sky_reference(pack, supervision)
            self._prepare_motion_reference(pack, context, tracks[index] if tracks is not None else None)
            render_output = None
            if self.global_step >= self.config.system.enable_render_global_step:
                if supervision is None:
                    raise RuntimeError("Render-stage training requires independently sampled supervision frames")
                render_output = render_kelvin_supervision(primitive, supervision)
            with torch.no_grad():
                psnr_inputs = self._render_psnr_inputs(render_output, supervision)
                if psnr_inputs is not None:
                    predicted_rgb, ground_truth_rgb = psnr_inputs
                    predicted_rgbs.append(predicted_rgb)
                    ground_truth_rgbs.append(ground_truth_rgb)
            result = self.loss(pack, context, render_output, supervision)
            per_item.append(result.total_value)
            skipped.extend(result.skipped)
            for name, value in result.values.items():
                component_values.setdefault(name, []).append(value)
        self._log_psnr(predicted_rgbs, ground_truth_rgbs, mode)
        # Official aggregation is a mean over batch elements, not a sum.
        total = torch.stack(per_item).mean()
        skipped_names = set(skipped)
        new_skips = skipped_names - self._warned_skipped_losses
        if new_skips:
            logger.warning(
                "Skipping Kelvin losses with unavailable labels/predictions: %s",
                ", ".join(sorted(new_skips)),
            )
            self._warned_skipped_losses.update(new_skips)
        return KelvinLossReturn(
            total_value=total,
            values={name: torch.stack(values).mean() for name, values in component_values.items()},
            skipped=tuple(sorted(skipped_names)),
        )

    def training_step(self, batch: InstantNuRecDataBatch, batch_idx: int):
        # Forward failures are reduced across ranks before any rank enters the
        # gradient-synchronizing manual_backward barrier.
        with BroadcastExceptions(self.trainer):
            optimizer = self.optimizers(use_pl_optimizer=True)
            optimizer.zero_grad()
            result = self.forward_losses(batch, "train")

        with BroadcastExceptions(self.trainer):
            self.manual_backward(result.total_value)
            raw_optimizer = optimizer.optimizer
            if any(parameter.grad is not None for group in raw_optimizer.param_groups for parameter in group["params"]):
                optimizer.step()
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, CosineWithWarmupPBScheduler):
                scheduler.set_progress(
                    self.current_epoch,
                    self.config.system.max_epochs,
                    batch_idx,
                    int(self.trainer.num_training_batches),
                )
                scheduler.step()
            self.log("train/loss", result.total_value, prog_bar=True, on_step=True, sync_dist=False)
            for name, value in result.values.items():
                self.log(f"train/{name}", value, on_step=True, sync_dist=False)
            self.log("train/lr", raw_optimizer.param_groups[0]["lr"], on_step=True)
        return result.total_value.detach()

    def validation_step(self, batch: InstantNuRecDataBatch, batch_idx: int):
        del batch_idx
        result = self.forward_losses(batch, "val")
        self.log("val/loss", result.total_value, prog_bar=True, on_epoch=True, sync_dist=True)
        return result.total_value

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["kelvin_training_contract"] = {
            "bazel_reference": self.config.bazel_reference,
            "phase": self.config.phase,
            "optimizer_implementation": self.optimizer_implementation,
            "world_size": self.trainer.world_size,
        }
