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

Two entry points:

- ``forward(context)``: dataclass-input convenience that runs ``encoder.encode``
  / ``decoder.decode`` (existing eager APIs). Used as the parity-reference
  during artifact export.
- ``forward_tensors(...)``: pure-tensor entry point that reimplements the
  static path using the encoder/decoder/post_processing submodules directly,
  so trace can cross the boundary. The artifact-export script in commit 5
  traces ``forward_tensors``.

Submodule ownership: ``__init__`` registers the encoder/decoder/post_processing
as submodules of this instance, so callers must transfer ownership rather than
share references with another parent (otherwise ``state_dict()`` surfaces
duplicate keys for the same parameters). The export script in commit 5
constructs fresh submodules and copies ``state_dict``s out of the loaded
``kelvin_full.pt``.

B=1 assumption: ``forward_tensors`` is shaped for ``predict_config.chunk_size=1``
(the only value the predict driver uses). A single per-batch loop iteration of
``decoder.decode``'s static path is unrolled.
"""

from __future__ import annotations

import math

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
    is what ``torch.jit.trace`` can persist: dataclasses do not survive
    serialization, so the JIT module returns these tensors directly and the
    Python-side adapter repackages them into ``KelvinStaticLayer`` after load.
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    rgb: torch.Tensor
    semantic_class: torch.Tensor
    normals: torch.Tensor


class KelvinStaticCore(nn.Module):
    """Static-only Kelvin forward suitable for JIT export."""

    # Class index for KelvinSemanticClass.MOVABLE -- pulled in as a constant
    # to keep ``forward_tensors`` free of imports that re-trigger module
    # initialization on the JIT-load side.
    _SEMANTIC_MOVABLE: int = 4
    _SEMANTIC_EGO: int = 1
    _SEMANTIC_SKY: int = 2

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

    # ------------------------------------------------------------------
    # forward(context) -- eager dataclass-input entry, used as parity ref
    # ------------------------------------------------------------------

    @staticmethod
    def _grab_camera_idxs(context: list[DataAndRenderingBatch]) -> torch.Tensor:
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
        """Run the static-only Kelvin pipeline via the dataclass entry points
        (encoder.encode + decoder.decode). The result matches
        ``forward_tensors``'s output to within numerical noise; the export
        script verifies this on a real batch.
        """
        encoded_latent = self.encoder.encode(context, self.scene_rescale)

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

    # ------------------------------------------------------------------
    # forward_tensors(...) -- pure-tensor entry, JIT-traceable
    # ------------------------------------------------------------------

    def forward_tensors(
        self,
        rgb: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        rays: torch.Tensor,
        distance_to_depth_scale: torch.Tensor,
        camera_idxs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,  # positions   (N_static, 3)
        torch.Tensor,  # rotations   (N_static, 4)
        torch.Tensor,  # scales      (N_static, 3)
        torch.Tensor,  # densities   (N_static, 1)
        torch.Tensor,  # rgb         (N_static, 3)
        torch.Tensor,  # semantic    (N_static, 1) uint8
        torch.Tensor,  # normals     (N_static, 3)
        torch.Tensor,  # affine      (1, n_affine_tokens, 3, 4)
    ]:
        """Static-only forward consuming pre-extracted tensors.

        Inputs (B=1):
            rgb:                     (1, V, H, W, 3)
            c2w:                     (1, V, 4, 4) -- end-of-frame, scene-rescaled
            fov:                     (1, V, 2)    -- (fov_w, fov_h) in radians
            rays:                    (1, V, H, W, 6) -- ``[origin (3), dir (3)]``
            distance_to_depth_scale: (1, V, H, W, 1)
            camera_idxs:             (1, V) int64

        Returns the static-layer tensor fields (positions/rotations/scales/
        densities/rgb/semantic_class/normals) followed by the affine_matrix.
        Output tuple-of-tensors mirrors what ``torch.jit.trace`` can persist;
        the loader-side adapter (commit 7) repackages it back into a
        ``KelvinStaticLayer`` + ``KelvinInstantNuRecPrimitive``.
        """
        scene_rescale = self.scene_rescale

        # ----- Encoder -----
        # Mirror of KelvinDAv3Encoder.encode (lines 144-161 of encoders.py)
        # but consuming the pre-stacked tensors directly.
        B, V, H, W, _ = rgb.shape
        x = self.encoder.patch_embed_img(
            self.encoder.rgb_normalize(rearrange(rgb, "B V H W C -> (B V) C H W"))
        )
        _, h, w, _ = x.shape
        x = rearrange(x, "(B V) h w C -> B V h w C", B=B, V=V)
        camera_encodings = self.encoder.embed_camera.forward(c2w, fov)
        with torch.autocast("cuda", enabled=True):
            img_feats, _ = self.encoder.vit.get_intermediate_features(
                x,
                block_indices=self.encoder.take_block_indices,
                global_cls_token=camera_encodings.unsqueeze(2),
            )
        # Mirror of KelvinMultiscaleFeaturesLatent.deepest:
        encoded_deepest = img_feats[-1]

        # ----- Decoder static path -----
        # Mirror of KelvinDPTDecoder.decode (lines 296-456 of decoders.py),
        # static-only: no motion head, no dynamic-layer construction.
        img_feats_flat = [rearrange(feat, "B V h w C -> (B V) h w C") for feat in img_feats]
        chunk_size = self.decoder.config.dpt_chunk_size

        # Depth
        depth_and_dconf = self.decoder.depth_head(
            img_feats_flat, output_shape=(H, W), chunk_size=chunk_size
        )
        depth_and_dconf = rearrange(depth_and_dconf, "(B V) C H W -> B V C H W", B=B, V=V)
        pred_depth = torch.exp(
            depth_and_dconf[:, :, 0].unsqueeze(-1) - math.log(scene_rescale)
        )  # (B, V, H, W, 1)

        # Context head -- inference-only path: never checkpointed.
        rgb_in_flat = rearrange(rgb, "B V H W C -> (B V) C H W")
        rgb_fusion_features = self.decoder.rgb_fusion(rgb_in_flat)
        context_features_tensor = self.decoder.context_head(
            img_feats_flat,
            output_shape=(H, W),
            fusion_features=rgb_fusion_features,
            chunk_size=chunk_size,
        )
        context_features_tensor = rearrange(
            context_features_tensor, "(B V) C H W -> B V H W C", B=B, V=V
        )
        n_semantic = self.decoder.n_semantic_classes
        context_rgb, context_world_normal, context_semantic_logits = (
            context_features_tensor.split([3, 3, n_semantic], dim=-1)
        )
        context_rgb = self.decoder.gaussian_activations.rgb(context_rgb)
        context_world_normal = torch.nn.functional.normalize(context_world_normal, dim=-1)

        context_dynamic_mask = (
            torch.argmax(context_semantic_logits, dim=-1) == self._SEMANTIC_MOVABLE
        )  # (B, V, H, W)

        # Gaussian heads
        gs_params_tensor = self.decoder.gaussians_head(
            img_feats_flat, output_shape=(H, W), fusion_features=None, chunk_size=chunk_size
        )
        gs_params_tensor = rearrange(gs_params_tensor, "(B V) C H W -> B V H W C", B=B, V=V)
        gs_scale, gs_world_quaternion, gs_opacity = gs_params_tensor.split([3, 4, 1], dim=-1)
        gs_distance = pred_depth / distance_to_depth_scale  # (B, V, H, W, 1)

        gs_scale = self.decoder.gaussian_activations.scale(gs_scale, scene_rescale=scene_rescale)
        # Mirror of KelvinSemanticClass.opacity_mask_from_semantic_probs (excludes ego + sky)
        semantic_probs = torch.softmax(context_semantic_logits, dim=-1)
        ego = semantic_probs[..., self._SEMANTIC_EGO : self._SEMANTIC_EGO + 1]
        sky = semantic_probs[..., self._SEMANTIC_SKY : self._SEMANTIC_SKY + 1]
        gs_valid_mask = 1.0 - ego - sky
        gs_opacity = (
            self.decoder.gaussian_activations.opacity(gs_opacity)
            * (gs_valid_mask > 0.5).float().detach()
        )
        gs_world_quaternion = self.decoder.gaussian_activations.rotation(gs_world_quaternion)
        gs_xyz = rays[..., :3] + rays[..., 3:] * gs_distance  # (B, V, H, W, 3)

        # Per-batch (B=1) static-layer extraction -- unrolled for trace.
        # ``[0]`` indices match ``for bidx in range(1)`` with ``bidx=0``.
        gs_xyz_flat = gs_xyz[0].reshape(-1, 3)
        gs_rotation_flat = gs_world_quaternion[0].reshape(-1, 4)
        gs_scale_flat = gs_scale[0].reshape(-1, 3)
        gs_opacity_flat = gs_opacity[0].reshape(-1, 1)
        gs_rgb_flat = context_rgb[0].reshape(-1, 3)

        dynamic_mask_flat = context_dynamic_mask[0].reshape(-1)
        static_mask = torch.where(~dynamic_mask_flat)[0]

        sem_class_flat = torch.argmax(context_semantic_logits[0], dim=-1).reshape(-1)
        semantic_class_static = sem_class_flat[static_mask].unsqueeze(-1).to(torch.uint8)
        normals_static = context_world_normal[0].reshape(-1, 3)[static_mask]

        positions = gs_xyz_flat[static_mask]
        rotations = gs_rotation_flat[static_mask]
        scales = gs_scale_flat[static_mask]
        densities = gs_opacity_flat[static_mask]
        out_rgb = gs_rgb_flat[static_mask]

        # ----- Per-camera affine post-processing -----
        # Mirror of KelvinInstantNuRec._compute_affine_matrix.
        encoded_deepest_tokens = rearrange(encoded_deepest, "B V h w C -> B (V h w) C")
        _, affine_latents = self.post_processing.transform_tokens(
            encoded_deepest_tokens, camera_idxs
        )
        affine_matrix_3, affine_bias = self.post_processing.decode_affine(affine_latents)
        affine_matrix = torch.cat([affine_matrix_3, affine_bias[..., None]], dim=-1)

        return (
            positions,
            rotations,
            scales,
            densities,
            out_rgb,
            semantic_class_static,
            normals_static,
            affine_matrix,
        )


class TraceableStaticCore(nn.Module):
    """Thin trace-target wrapper exposing only ``KelvinStaticCore.forward_tensors``.

    Trace records the module's ``forward``; this class exists so the traced
    artifact's call signature is ``(rgb, c2w, fov, rays, distance_to_depth_scale,
    camera_idxs) -> tuple[Tensor, ...]`` -- matching what the loader-side
    adapter (commit 7) constructs from a ``DataAndRenderingBatch``.
    """

    def __init__(self, static_core: KelvinStaticCore):
        super().__init__()
        self.static_core = static_core

    def forward(
        self,
        rgb: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        rays: torch.Tensor,
        distance_to_depth_scale: torch.Tensor,
        camera_idxs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self.static_core.forward_tensors(
            rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs
        )
