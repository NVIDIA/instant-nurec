# Copyright (c) 2024-2026 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import warnings

from dataclasses import replace
from itertools import chain
from typing import Any, Iterator, Optional, cast

import numpy as np
import torch

from einops import rearrange
from safetensors.torch import load_file

from ncore.data import ConcreteCameraModelParametersUnion
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.models.gaussians.renderers import BaseGaussianRenderer
from nre.nrm.config.models import KelvinModelConfig
from nre.nrm.models.base import BaseNRM
from nre.nrm.models.kelvin_backbone.base import KelvinNRMSupervisionPack
from nre.nrm.models.kelvin_backbone.decoders import KelvinDPTDecoder, make_decoder
from nre.nrm.models.kelvin_backbone.encoders import make_encoder
from nre.nrm.models.kelvin_backbone.sky import make_sky
from nre.nrm.models.post_processing import PerCameraAffinePostProcessing
from nre.nrm.primitives.kelvin_primitive import KelvinNRMPrimitive
from nre.nrm.utils.cubemap import layout_sky_cubemap, unproject_to_sky_cubemap
from nre.nrm.utils.motion import TimeRemapping
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.files import local_temp_file, parse_universal_path
from nre.utils.geometry import tquat_to_se3_matrix
from nre.utils.log import BatchMediaLogger
from nre.utils.misc import unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.types import RayFlags
from nre.utils.visualize import make_image_grid


logger = logging.getLogger(__name__)


class KelvinNRM(BaseNRM[KelvinNRMPrimitive, KelvinNRMSupervisionPack]):
    """
    Please refer to the [Kelvin Model](../docs/KELVIN_MODEL.md) for more details.
    """

    config: KelvinModelConfig

    def __init__(self, config: KelvinModelConfig):
        super().__init__(config)
        self.encoder = make_encoder(config)
        self.decoder = make_decoder(config)
        self.sky = make_sky(config)
        self.sky_cubemap_size = self.config.sky.cubemap_size
        self.post_processing: Optional[PerCameraAffinePostProcessing] = None
        if config.post_processing.enabled:
            self.post_processing = PerCameraAffinePostProcessing(
                embed_dim=config.encoder.embed_dim, init_token_scale=0.02, cross_attend=True, kv_norm=True
            )
        self.scene_rescale = self.config.scene_rescale
        self.cuboids_dims_padding = torch.nn.Buffer(torch.tensor(self.config.track_padding_m, dtype=torch.float32))

        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Prepare gaussian renderer
        self.use_2dgs = self.config.use_2dgs
        self.gaussians_renderer = BaseGaussianRenderer.factory(
            self.config.renderer.name,
            self.config.renderer,
            self,  # type: ignore
        )

    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        # nrend init just need the particle configuration part.
        return KelvinNRMPrimitive.random(
            1,
            use_2dgs=self.use_2dgs,
            device=torch.device("cpu"),
        ).serialize_to_json_dict(with_state_dict=with_state_dict)

    @staticmethod
    def _maybe_derive_normals_from_distance(batch: DataAndRenderingBatch) -> DataAndRenderingBatch:
        """Compute world-space normals + VALID_NORMAL flag from metric_distance via
        cross-product when the batch doesn't already carry `labels.normals`. Returns
        the batch unchanged when normals are preloaded or metric_distance is missing.
        """
        assert (camera_data := batch.data.camera) is not None
        assert (rendering_data := batch.rendering) is not None and (
            rendering_camera_data := rendering_data.camera
        ) is not None
        if camera_data.labels.normals is not None:
            return batch
        if (metric_distance := camera_data.labels.metric_distance) is None:
            return batch

        new_flags = unpack_optional(camera_data.labels.flags)
        batch_rays = rendering_camera_data.rays
        world_points = batch_rays[..., :3] + metric_distance * batch_rays[..., 3:]  # (B, H, W, 3)
        # Normals are valid only for interior pixels.
        new_normals = torch.zeros_like(world_points)
        new_normals[:, 1:-1, 1:-1] = torch.nn.functional.normalize(
            torch.cross(
                world_points[:, 2:, 1:-1] - world_points[:, :-2, 1:-1],
                world_points[:, 1:-1, 2:] - world_points[:, 1:-1, :-2],
            ),
            dim=-1,
        )
        distance_valid_mask = (
            (metric_distance[:, 2:, 1:-1] > 0.0)
            & (metric_distance[:, 1:-1, 2:] > 0.0)
            & (metric_distance[:, 1:-1, :-2] > 0.0)
            & (metric_distance[:, :-2, 1:-1] > 0.0)
        )
        new_normals[:, 1:-1, 1:-1][~distance_valid_mask.squeeze(-1)] = 0.0
        new_flags[:, 1:-1, 1:-1][distance_valid_mask] |= RayFlags.VALID_NORMAL  # Loss depends on this flag.
        return replace(
            batch,
            data=replace(
                batch.data,
                camera=replace(camera_data, labels=replace(camera_data.labels, normals=new_normals, flags=new_flags)),
            ),
        )

    @ScopedTimer("KelvinModel.prepare_supervision")
    def prepare_supervision(
        self,
        context: list[DataAndRenderingBatch],
        supervision: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        supervision_packs: list[KelvinNRMSupervisionPack],
        media_logger: BatchMediaLogger | None,
    ) -> tuple[list[DataAndRenderingBatch], list[KelvinNRMSupervisionPack]]:
        """
        Note we might have to make dynamic objects invalid here.
        """
        prepared_supervision: list[DataAndRenderingBatch] = []

        for bidx, batch in enumerate(supervision):
            assert (camera_data := batch.data.camera) is not None
            assert (rendering_data := batch.rendering) is not None and (
                rendering_camera_data := rendering_data.camera
            ) is not None

            if camera_data.labels.normals is None:
                batch = self._maybe_derive_normals_from_distance(batch)
                camera_data = unpack_optional(batch.data.camera)
            else:
                warnings.warn(
                    "Preload normals found (most likely from AUX files). Make sure they are in world space.",
                    stacklevel=2,
                )

            # Prepare sky mask for gradient detachment during sky compositing.
            rays_is_sky = camera_data.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC)
            rendering_camera_data = replace(rendering_camera_data, _rays_is_sky=rays_is_sky)
            batch = replace(batch, rendering=replace(rendering_data, camera=rendering_camera_data))

            prepared_supervision.append(batch)

            # Compute supervision pack (reference cubemap)
            gt_cubemap, gt_cubemap_mask = unproject_to_sky_cubemap(
                self.sky_cubemap_size,
                tquat_to_se3_matrix(rendering_camera_data.poses_tquat_startend[:, 1], unbatch=False)[:, :3, :3],
                [
                    cast(ConcreteCameraModelParametersUnion, sensor_model)
                    for sensor_model in rendering_camera_data.sensor_model_parameters
                ],
                unpack_optional(camera_data.labels.rgb),
                camera_data.labels.get_mask_flags_none(RayFlags.INVALID)
                & camera_data.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC),
            )
            supervision_packs[bidx].reference_sky_cubemap = gt_cubemap
            supervision_packs[bidx].reference_sky_cubemap_mask = gt_cubemap_mask

            if bidx == 0 and (media_logger is not None and media_logger.should_log_media):
                pd_cubemap = supervision_packs[bidx].predicted_sky_cubemap
                if pd_cubemap is not None:
                    pd_cubemap_np = layout_sky_cubemap(pd_cubemap).detach().cpu().numpy()
                    gt_cubemap_np = layout_sky_cubemap(gt_cubemap).detach().cpu().numpy()
                    media_logger.log_image(
                        "Predicted Sky Cubemap",
                        np.concatenate([pd_cubemap_np, gt_cubemap_np], axis=0),  # 4x3 grid
                    )

        # Motion supervision: reference displacement from cuboid warp on metric world points (Celsius-style padding).
        for bidx, batch in enumerate(context):
            if not supervision_packs[bidx].motion_supervisions:
                continue
            # Reference flow requires both cuboid tracks and metric distance (depth).
            # When either is unavailable, clear motion_supervisions so the loss doesn't
            # attempt to unpack a None reference_flow.
            if cuboid_tracks is None:
                supervision_packs[bidx].motion_supervisions = []
                continue
            data_camera = unpack_optional(batch.data.camera)
            metric_distance = data_camera.labels.metric_distance
            if metric_distance is None:
                supervision_packs[bidx].motion_supervisions = []
                continue

            rendering_camera = unpack_optional(unpack_optional(batch.rendering).camera)
            cuboid_track_bidx = cuboid_tracks[bidx]
            dynamic_track = CuboidTracks.Ops.subset_from_mask(
                cuboid_track_bidx, cuboid_track_bidx.tracks_flags & TrackFlags.DYNAMIC != 0
            )
            rays = rendering_camera.rays
            world_points = rays[..., :3] + metric_distance * rays[..., 3:]
            for motion_supervision in supervision_packs[bidx].motion_supervisions:
                if motion_supervision.reference_flow is not None:
                    continue
                motion_supervision.reference_flow = (
                    dynamic_track.warp_world_points_to_timestamps(
                        world_points,
                        motion_supervision.source_timestamps_us,
                        motion_supervision.target_timestamps_us,
                        self.cuboids_dims_padding,
                    )
                    - world_points
                )

        return prepared_supervision, supervision_packs

    @ScopedTimer("KelvinModel.prepare_context")
    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[DataAndRenderingBatch]:
        return [self._maybe_derive_normals_from_distance(batch) for batch in context]

    def on_train_from_scratch_start(self, system, **kwargs):
        # Full-model path: init_weights_paths contains a "full" or "tokengs" entry. Any additional
        # keys are ignored when either is present. Kept in lockstep with the has_full_init gate in
        # nre/nrm/run.py; "full" wins over "tokengs" if both are provided.
        full_model_path = None
        for key in ("full", "tokengs"):
            if key in self.config.init_weights_paths:
                full_model_path = self.config.init_weights_paths[key]
                break
        if full_model_path is not None:
            with local_temp_file(parse_universal_path(full_model_path, s3_block_size_mb=256)) as local_path:
                if str(local_path).endswith(".safetensors"):
                    init_state_dict = load_file(local_path)
                else:
                    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
                    init_state_dict = ckpt.get("state_dict", ckpt)
            init_state_dict = {k.replace("model.", ""): v for k, v in init_state_dict.items()}
            # Always re-init GS head.
            if isinstance(self.decoder, KelvinDPTDecoder):
                init_state_dict = {
                    k: v for k, v in init_state_dict.items() if not k.startswith("decoder.gaussians_head.")
                }
            model_sd = self.state_dict()
            # Keep only model keys; use converted where present, else current model init (e.g. sky, post_processing).
            missing_in_ckpt = [k for k in model_sd if k not in init_state_dict]
            if missing_in_ckpt:
                logger.info(
                    "Model parameters not found in checkpoint (using current init): %s",
                    missing_in_ckpt,
                )
            init_state_dict = {k: init_state_dict.get(k, model_sd[k].clone()) for k in model_sd}
            self.load_state_dict(init_state_dict, strict=True)
            if self.post_processing is not None:
                self.post_processing.zero_init()
            return

        loaded_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        for name, weights_path in self.config.init_weights_paths.items():
            with local_temp_file(parse_universal_path(weights_path, s3_block_size_mb=256)) as local_model_path:
                loaded_state_dicts[name] = load_file(local_model_path)

        self.encoder.initialize_weights(loaded_state_dicts)
        self.decoder.initialize_weights(loaded_state_dicts)
        self.sky.initialize_weights(loaded_state_dicts)
        if self.post_processing is not None:
            self.post_processing.zero_init()

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if self.post_processing is not None:
            self.post_processing.set_detach_linear_grad(
                global_step < self.config.post_processing.optimization_start_global_step
            )
        self.encoder.update_step_train_batch_start(epoch, global_step, system, **kwargs)
        self.decoder.update_step_train_batch_start(epoch, global_step, system, **kwargs)
        return {}

    def get_potential_unused_parameters(self) -> Iterator[torch.nn.Parameter]:
        parts = [
            self.encoder.get_potential_unused_parameters(),
            self.decoder.get_potential_unused_parameters(),
        ]
        if self.post_processing is not None:
            parts.append(self.post_processing.parameters())
        return chain(*parts)

    @staticmethod
    def _grab_metainfo(
        context: list[DataAndRenderingBatch],
    ) -> tuple[int, int, torch.Tensor, list[TimeRemapping]]:
        first_context_data = unpack_optional(context[0].data.camera)
        num_imgs = first_context_data.b
        num_views = len(set([meta.unique_sensor_idx for meta in first_context_data.meta]))

        batch_camera_idxs: list[torch.Tensor] = []
        time_remappings: list[TimeRemapping] = []
        for bidx, batch in enumerate(context):
            context_data = unpack_optional(batch.data.camera)
            unique_sensor_idx = torch.tensor([meta.unique_sensor_idx for meta in context_data.meta], dtype=torch.int64)
            num_views_bidx = len(unique_sensor_idx.unique())
            assert context_data.b == num_imgs, "All context batches must have the same number of images"
            assert num_views_bidx == num_views, "All context batches must have the same number of views"
            batch_camera_idxs.append(unique_sensor_idx)

            rendering = unpack_optional(batch.rendering)
            camera = unpack_optional(rendering.camera)
            time_remappings.append(
                TimeRemapping.from_timestamps_startend_us(camera.timestamps_startend_us_cpu, unique_sensor_idx)
            )

        return num_imgs, num_views, torch.stack(batch_camera_idxs, dim=0), time_remappings

    @ScopedTimer("KelvinModel.reconstruct")
    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        media_logger: BatchMediaLogger | None,
        compute_supervision_pack: bool = False,
    ) -> tuple[list[KelvinNRMPrimitive], list[KelvinNRMSupervisionPack] | None]:
        # Add assertions about input context -- num_images and num_views should match
        num_imgs, num_views, camera_idxs, time_remappings = self._grab_metainfo(context)

        # Log input images (first batch only)
        if media_logger is not None and media_logger.should_log_media:
            first_rgb = unpack_optional(context[0].data.camera).labels.rgb
            media_logger.log_image(
                "Input RGB",
                make_image_grid(
                    [(rgb_i * 255).astype(np.uint8) for rgb_i in unpack_optional(first_rgb).cpu().numpy()],
                    grid_width=num_imgs // num_views,
                ),
            )

        # Encode the inputs
        encoded_latent = self.encoder.encode(context, time_remappings, self.scene_rescale, media_logger)

        # Forward the decoder (Compute supervision packs along the way)
        decoder_returns = self.decoder.decode(
            encoded_latent,
            context,
            cuboid_tracks,
            time_remappings,
            self.scene_rescale,
            media_logger,
        )
        static_layers = [return_value.static_layer for return_value in decoder_returns]
        dynamic_layers = [return_value.dynamic_layers for return_value in decoder_returns]
        supervision_packs = [return_value.supervision_pack for return_value in decoder_returns]

        # Forward sky
        sky_cubemaps = self.sky.decode(encoded_latent, context).contiguous()

        # Per-camera affine RGB (optional)
        if self.post_processing is not None:
            _, affine_latents = self.post_processing.transform_tokens(
                rearrange(encoded_latent.deepest, "B V h w C -> B (V h w) C"), camera_idxs
            )
            affine_matrix_3, affine_bias = self.post_processing.decode_affine(affine_latents)
            affine_matrix = torch.cat([affine_matrix_3, affine_bias[..., None]], dim=-1)
        else:
            device = encoded_latent.deepest.device
            dtype = encoded_latent.deepest.dtype
            affine_matrix = torch.zeros(len(context), num_views, 3, 4, device=device, dtype=dtype)
            affine_matrix[..., :3, :3] = torch.eye(3, device=device, dtype=dtype)

        # Build the primitives
        primitives: list[KelvinNRMPrimitive] = []
        for bidx in range(len(context)):
            sky_cubemap_bidx = sky_cubemaps[bidx]
            supervision_packs[bidx].predicted_sky_cubemap = sky_cubemap_bidx

            primitive = KelvinNRMPrimitive(
                static_layer=unpack_optional(static_layers[bidx]),
                dynamic_layers=dynamic_layers[bidx],
                sky_cubemap=sky_cubemap_bidx,
                affine_matrix=affine_matrix[bidx],
                use_2dgs=self.use_2dgs,
                gaussians_renderer=self.gaussians_renderer,
            )
            primitives.append(primitive)

        return primitives, supervision_packs if compute_supervision_pack else None
