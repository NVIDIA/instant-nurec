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

"""``KelvinStaticCore``: the static-only path carved out for JIT export.

The shipped Kelvin artifact only feeds ``static_layer`` + ``affine_matrix`` into
the PLY export -- ``sky_cubemap`` and ``dynamic_layers`` are computed today but
never reach disk (see ``predict/export_ply.py:75`` and the commit-7 contract).
``KelvinStaticCore`` packages the encoder + (static parts of the) decoder +
per-camera affine post-processing as a single ``nn.Module`` so the next commit
can hand it to ``torch.jit.trace`` and persist the result as ``kelvin_jit.pt``.

The class is deliberately split out under ``internal/`` because it is only
used by the artifact-export pipeline; the public package never imports it
directly. Once commit 7 swaps the loader to ``torch.jit.load`` the shipped
runtime no longer needs ``KelvinStaticCore`` either.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from einops import rearrange
from torch import nn

from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.misc import unpack_optional
from instant_nurec_internal.model.backbone.decoders import KelvinDPTDecoder
from instant_nurec_internal.model.backbone.encoders import KelvinDAv3Encoder
from instant_nurec_internal.model.post_processing import PerCameraAffinePostProcessing


@dataclass
class StaticLayerTensors:
    """Per-batch tensor bundle for the static-layer fields the PLY exporter reads.

    Mirrors the ``KelvinStaticLayer`` dataclass but as a flat tensor tuple, which
    is what ``torch.jit.trace`` can persist: dataclasses do not survive scripting,
    so the JIT module returns these tensors directly and the Python-side adapter
    repackages them into ``KelvinStaticLayer`` after load.
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    rgb: torch.Tensor
    semantic_class: torch.Tensor
    normals: torch.Tensor


class KelvinStaticCore(nn.Module):
    """Static-only Kelvin forward suitable for JIT export.

    Parameter ownership: the constructor *takes* references to the existing
    encoder / decoder / post_processing modules. PyTorch will register them
    as submodules of this instance, so callers must transfer ownership rather
    than share references with another parent (otherwise ``state_dict()``
    surfaces duplicate keys for the same parameters). The artifact-export
    script in commit 5 instantiates fresh submodules and copies their
    ``state_dict``s out of the loaded ``kelvin_full.pt``.
    """

    def __init__(
        self,
        encoder: KelvinDAv3Encoder,
        decoder: KelvinDPTDecoder,
        post_processing: PerCameraAffinePostProcessing,
        scene_rescale: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.post_processing = post_processing
        self.scene_rescale = scene_rescale

    @staticmethod
    def _grab_camera_idxs(context: list[DataAndRenderingBatch]) -> torch.Tensor:
        """Same metainfo extraction as ``KelvinInstantNuRec._grab_metainfo`` but
        narrowed to the only output the static path actually consumes
        (``camera_idxs``)."""
        batch_camera_idxs: list[torch.Tensor] = []
        for batch in context:
            data = unpack_optional(batch.data.camera)
            unique_sensor_idx = torch.tensor(
                [meta.unique_sensor_idx for meta in data.meta], dtype=torch.int64
            )
            batch_camera_idxs.append(unique_sensor_idx)
        return torch.stack(batch_camera_idxs, dim=0)

    def _compute_affine_matrix(
        self, encoded_latent, camera_idxs: torch.Tensor
    ) -> torch.Tensor:
        _, affine_latents = self.post_processing.transform_tokens(
            rearrange(encoded_latent.deepest, "B V h w C -> B (V h w) C"), camera_idxs
        )
        affine_matrix_3, affine_bias = self.post_processing.decode_affine(affine_latents)
        return torch.cat([affine_matrix_3, affine_bias[..., None]], dim=-1)

    def forward(
        self, context: list[DataAndRenderingBatch]
    ) -> tuple[list[StaticLayerTensors], torch.Tensor]:
        """Run the static-only Kelvin pipeline.

        Returns one ``StaticLayerTensors`` per input batch plus a single
        ``affine_matrix`` (shape: ``(B, n_cameras, 3, 4)``). The dynamic layer
        and sky cubemap are not produced here -- the static-only output
        contract makes them unnecessary for PLY export.
        """
        encoded_latent = self.encoder.encode(context, self.scene_rescale)

        # ``cuboid_tracks=None`` skips the cuboid-track branch but the motion
        # head still runs (its outputs feed dynamic_layers, which we then
        # discard). Commit 5 introduces a ``decode_static_only`` path on the
        # decoder that bypasses the motion head entirely; until then the motion
        # head's compute is kept so the parity gate can compare bitwise against
        # the eager static_layer output.
        time_remappings = self._build_time_remappings(context)
        decoder_returns = self.decoder.decode(
            encoded_latent,
            context,
            None,  # cuboid_tracks
            time_remappings,
            self.scene_rescale,
        )

        camera_idxs = self._grab_camera_idxs(context).to(encoded_latent.deepest.device)
        affine_matrix = self._compute_affine_matrix(encoded_latent, camera_idxs)

        static_bundles: list[StaticLayerTensors] = []
        for ret in decoder_returns:
            sl = unpack_optional(ret.static_layer)
            static_bundles.append(
                StaticLayerTensors(
                    positions=sl.positions,
                    rotations=sl.rotations,
                    scales=sl.scales,
                    densities=sl.densities,
                    rgb=sl.rgb,
                    semantic_class=sl.semantic_class
                    if sl.semantic_class is not None
                    else torch.zeros(len(sl), 1, dtype=torch.uint8, device=sl.device()),
                    normals=sl.normals
                    if sl.normals is not None
                    else torch.zeros(len(sl), 3, device=sl.device()),
                )
            )

        return static_bundles, affine_matrix

    @staticmethod
    def _build_time_remappings(context: list[DataAndRenderingBatch]):
        """Mirror of ``KelvinInstantNuRec._grab_metainfo``'s time-remapping
        construction, kept independent so this module doesn't import the
        soon-to-be-thin ``KelvinInstantNuRec`` class. Commit 5's
        ``decode_static_only`` bypasses the motion-head call entirely and
        this helper goes away with it.
        """
        from instant_nurec.utils.motion import TimeRemapping

        time_remappings = []
        for batch in context:
            data = unpack_optional(batch.data.camera)
            unique_sensor_idx = torch.tensor(
                [meta.unique_sensor_idx for meta in data.meta], dtype=torch.int64
            )
            rendering = unpack_optional(unpack_optional(batch.rendering).camera)
            time_remappings.append(
                TimeRemapping.from_timestamps_startend_us(
                    rendering.timestamps_startend_us_cpu, unique_sensor_idx
                )
            )
        return time_remappings
