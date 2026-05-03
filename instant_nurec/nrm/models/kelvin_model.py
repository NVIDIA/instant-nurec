# Copyright (c) 2024-2026 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging

from dataclasses import replace

import torch
import torch.nn as nn

from einops import rearrange

from instant_nurec.datasets.tracks import CuboidTracks
from instant_nurec.nrm.config.models import KelvinModelConfig
from instant_nurec.nrm.models.kelvin_backbone.decoders import KelvinDPTDecoder
from instant_nurec.nrm.models.kelvin_backbone.encoders import KelvinDAv3Encoder
from instant_nurec.nrm.models.kelvin_backbone.sky import CubemapDecoderSky
from instant_nurec.nrm.models.post_processing import PerCameraAffinePostProcessing
from instant_nurec.nrm.primitives.kelvin_primitive import KelvinNRMPrimitive
from instant_nurec.nrm.utils.motion import TimeRemapping
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.types import RayFlags


logger = logging.getLogger(__name__)


class KelvinNRM(nn.Module):
    """
    Please refer to the [Kelvin Model](../docs/KELVIN_MODEL.md) for more details.
    """

    config: KelvinModelConfig

    def __init__(self, config: KelvinModelConfig):
        super().__init__()
        self.config = config
        self.encoder = KelvinDAv3Encoder(config.encoder, config)
        self.decoder = KelvinDPTDecoder(config.decoder, config)
        self.sky = CubemapDecoderSky(config.sky, config)
        self.post_processing = PerCameraAffinePostProcessing(
            embed_dim=config.encoder.embed_dim, init_token_scale=0.02
        )
        self.scene_rescale = self.config.scene_rescale
        self.cuboids_dims_padding = torch.nn.Buffer(torch.tensor(self.config.track_padding_m, dtype=torch.float32))

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

    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
    ) -> list[DataAndRenderingBatch]:
        return [self._maybe_derive_normals_from_distance(batch) for batch in context]

    @staticmethod
    def _grab_metainfo(
        context: list[DataAndRenderingBatch],
    ) -> tuple[int, int, torch.Tensor, list[TimeRemapping]]:
        first_context_data = unpack_optional(context[0].data.camera)
        num_imgs = first_context_data.b
        num_views = len(set([meta.unique_sensor_idx for meta in first_context_data.meta]))

        batch_camera_idxs: list[torch.Tensor] = []
        time_remappings: list[TimeRemapping] = []
        for batch in context:
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

    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[KelvinNRMPrimitive]:
        # Add assertions about input context -- num_images and num_views should match
        num_imgs, num_views, camera_idxs, time_remappings = self._grab_metainfo(context)

        # Encode the inputs
        encoded_latent = self.encoder.encode(context, self.scene_rescale)

        # Forward the decoder
        decoder_returns = self.decoder.decode(
            encoded_latent,
            context,
            cuboid_tracks,
            time_remappings,
            self.scene_rescale,
        )
        static_layers = [return_value.static_layer for return_value in decoder_returns]
        dynamic_layers = [return_value.dynamic_layers for return_value in decoder_returns]

        # Forward sky
        sky_cubemaps = self.sky.decode(encoded_latent, context).contiguous()

        # Per-camera affine RGB
        _, affine_latents = self.post_processing.transform_tokens(
            rearrange(encoded_latent.deepest, "B V h w C -> B (V h w) C"), camera_idxs
        )
        affine_matrix_3, affine_bias = self.post_processing.decode_affine(affine_latents)
        affine_matrix = torch.cat([affine_matrix_3, affine_bias[..., None]], dim=-1)

        # Build the primitives
        primitives: list[KelvinNRMPrimitive] = []
        for bidx in range(len(context)):
            primitive = KelvinNRMPrimitive(
                static_layer=unpack_optional(static_layers[bidx]),
                dynamic_layers=dynamic_layers[bidx],
                sky_cubemap=sky_cubemaps[bidx],
                affine_matrix=affine_matrix[bidx],
            )
            primitives.append(primitive)

        return primitives
