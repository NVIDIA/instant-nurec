# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2024-2026 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from einops import rearrange

from instant_nurec.datasets.tracks import CuboidTracks
from instant_nurec.config_schema.models import KelvinModelConfig, KelvinPointQueryCADecoderConfig
from instant_nurec.model.backbone.decoders import KelvinDPTDecoder
from instant_nurec.model.backbone.encoders import KelvinDAv3Encoder
from instant_nurec.model.backbone.sky import CubemapDecoderSky
from instant_nurec.model.post_processing import PerCameraAffinePostProcessing
from instant_nurec.model.supervision import SupervisionPack
from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.motion import TimeRemapping
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.types import RayFlags


logger = logging.getLogger(__name__)


class KelvinInstantNuRec(nn.Module):
    """
    Please refer to the [Kelvin Model](../docs/KELVIN_MODEL.md) for more details.
    """

    config: KelvinModelConfig

    def __init__(self, config: KelvinModelConfig):
        super().__init__()
        if isinstance(config.decoder, KelvinPointQueryCADecoderConfig):
            raise TypeError(
                "KelvinInstantNuRec full-model composition only supports the dense DPT decoder; "
                "use KelvinStaticCore for point-query inference"
            )
        self.config = config
        self.encoder = KelvinDAv3Encoder(config.encoder, config)
        self.decoder = KelvinDPTDecoder(config.decoder, config)
        self.sky = CubemapDecoderSky(config.sky, config)
        self.post_processing = (
            PerCameraAffinePostProcessing(embed_dim=config.encoder.embed_dim, init_token_scale=0.02)
            if config.post_processing.enabled
            else None
        )
        self.scene_rescale = self.config.scene_rescale
        self.cuboids_dims_padding = torch.nn.Buffer(torch.tensor(self.config.track_padding_m, dtype=torch.float32))
        if config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
    ) -> list[DataAndRenderingBatch]:
        for batch in context:
            camera = batch.data.camera
            rendering = batch.rendering.camera if batch.rendering is not None else None
            if camera is None or rendering is None:
                continue
            labels = camera.labels
            if labels.normals is not None or labels.metric_distance is None:
                continue
            points = rendering.rays[..., :3] + labels.metric_distance * rendering.rays[..., 3:]
            normals = torch.zeros_like(points)
            normals[:, 1:-1, 1:-1] = torch.nn.functional.normalize(
                torch.cross(
                    points[:, 2:, 1:-1] - points[:, :-2, 1:-1],
                    points[:, 1:-1, 2:] - points[:, 1:-1, :-2],
                    dim=-1,
                ),
                dim=-1,
            )
            valid = (
                (labels.metric_distance[:, 2:, 1:-1] > 0)
                & (labels.metric_distance[:, :-2, 1:-1] > 0)
                & (labels.metric_distance[:, 1:-1, 2:] > 0)
                & (labels.metric_distance[:, 1:-1, :-2] > 0)
            )
            normals[:, 1:-1, 1:-1][~valid.squeeze(-1)] = 0
            labels.normals = normals
            if labels.flags is not None:
                labels.flags[:, 1:-1, 1:-1][valid] |= int(RayFlags.VALID_NORMAL)
        return context

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

    def _compute_affine_matrix(
        self,
        encoded_latent,
        camera_idxs: torch.Tensor,
    ) -> torch.Tensor:
        # This affine transform is also used by the static PLY inference path.
        if self.post_processing is None:
            B = encoded_latent.deepest.shape[0]
            n_cameras = torch.unique(camera_idxs[0]).numel()
            affine = torch.zeros(B, n_cameras, 3, 4, device=encoded_latent.deepest.device, dtype=torch.float32)
            affine[..., :3, :3] = torch.eye(3, device=affine.device)
            return affine
        _, affine_latents = self.post_processing.transform_tokens(
            rearrange(encoded_latent.deepest, "B V h w C -> B (V h w) C"), camera_idxs
        )
        affine_matrix_3, affine_bias = self.post_processing.decode_affine(affine_latents)
        return torch.cat([affine_matrix_3, affine_bias[..., None]], dim=-1)

    @staticmethod
    def _build_primitives(
        context: list[DataAndRenderingBatch],
        decoder_returns,
        sky_cubemaps: torch.Tensor,
        affine_matrix: torch.Tensor,
    ) -> list[KelvinInstantNuRecPrimitive]:
        primitives: list[KelvinInstantNuRecPrimitive] = []
        for bidx in range(len(context)):
            primitive = KelvinInstantNuRecPrimitive(
                static_layer=unpack_optional(decoder_returns[bidx].static_layer),
                dynamic_layers=decoder_returns[bidx].dynamic_layers,
                sky_cubemap=sky_cubemaps[bidx],
                affine_matrix=affine_matrix[bidx],
            )
            primitives.append(primitive)
        return primitives

    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[KelvinInstantNuRecPrimitive]:
        primitives, _ = self.reconstruct_with_supervision(context, cuboid_tracks)
        return primitives

    def reconstruct_with_supervision(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> tuple[list[KelvinInstantNuRecPrimitive], list[SupervisionPack]]:
        """Reconstruct primitives and retain the dense training predictions."""

        num_imgs, num_views, camera_idxs, time_remappings = self._grab_metainfo(context)

        encoded_latent = self.encoder.encode(context, self.scene_rescale)

        decoder_returns = self.decoder.decode(
            encoded_latent,
            context,
            cuboid_tracks,
            time_remappings,
            self.scene_rescale,
        )

        # Decode the non-static sky head for full-model callers.
        sky_cubemaps = self.sky.decode(encoded_latent, context).contiguous()

        affine_matrix = self._compute_affine_matrix(encoded_latent, camera_idxs)
        packs = [decoder_return.supervision_pack for decoder_return in decoder_returns]
        for pack, sky_cubemap in zip(packs, sky_cubemaps):
            pack.predicted_sky_cubemap = sky_cubemap
        return self._build_primitives(context, decoder_returns, sky_cubemaps, affine_matrix), packs

    def update_step_train_batch_start(self, global_step: int) -> None:
        if self.post_processing is not None:
            self.post_processing.set_detach_linear_grad(
                global_step < self.config.post_processing.optimization_start_global_step
            )
