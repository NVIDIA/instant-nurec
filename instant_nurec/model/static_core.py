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

"""Eager source implementation of the pretrained static reconstruction core.

The PLY exporter consumes the static Gaussian layer and per-camera affine
matrix. This module therefore runs the released encoder, the static decoder
heads, and affine post-processing directly from Python source. Variable-length
cuboid-track masking remains in :mod:`instant_nurec.model.inference`.
"""

from __future__ import annotations

import math

import torch

from einops import rearrange
from torch import nn

from instant_nurec.model.backbone.decoders import KelvinDPTDecoder
from instant_nurec.model.backbone.encoders import KelvinDAv3Encoder
from instant_nurec.model.post_processing import PerCameraAffinePostProcessing
from instant_nurec.config_schema.models import KelvinModelConfig


class KelvinStaticCore(nn.Module):
    """Static Gaussian reconstruction heads used by public inference."""

    # Class indices for KelvinSemanticClass.{EGO,SKY,MOVABLE}.
    _SEMANTIC_EGO: int = 1
    _SEMANTIC_SKY: int = 2
    _SEMANTIC_MOVABLE: int = 4

    def __init__(self, config: KelvinModelConfig) -> None:
        super().__init__()
        inference_config = config.model_copy(deep=True)
        inference_config.encoder.checkpointing = "none"
        inference_config.decoder.checkpointing = False
        self.encoder = KelvinDAv3Encoder(inference_config.encoder, inference_config)
        self.decoder = KelvinDPTDecoder(inference_config.decoder, inference_config)
        self.post_processing = PerCameraAffinePostProcessing(
            embed_dim=inference_config.encoder.embed_dim,
            init_token_scale=0.02,
        )
        self.scene_rescale = inference_config.scene_rescale

    def forward(
        self,
        rgb: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        rays: torch.Tensor,
        distance_to_depth_scale: torch.Tensor,
        camera_idxs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,  # gs_xyz             (B, V, H, W, 3)
        torch.Tensor,  # gs_rotations       (B, V, H, W, 4)
        torch.Tensor,  # gs_scales          (B, V, H, W, 3)
        torch.Tensor,  # gs_densities       (B, V, H, W, 1)
        torch.Tensor,  # gs_rgb             (B, V, H, W, 3)
        torch.Tensor,  # semantic_argmax    (B, V, H, W)   int64
        torch.Tensor,  # normals            (B, V, H, W, 3)
        torch.Tensor,  # affine             (B, n_affine_tokens, 3, 4)
    ]:
        """Run the static model heads on pre-extracted tensors.

        Inputs (B=1):
            rgb:                     (1, V, H, W, 3)
            c2w:                     (1, V, 4, 4) -- end-of-frame, scene-rescaled
            fov:                     (1, V, 2)    -- (fov_w, fov_h) in radians
            rays:                    (1, V, H, W, 6) -- ``[origin (3), dir (3)]``
            distance_to_depth_scale: (1, V, H, W, 1)
            camera_idxs:             (1, V) int64

        Returns the *unmasked* per-pixel gaussian fields plus the affine
        matrix. Static/dynamic split, cuboid-track-based mask refinement,
        and final flatten + gather happen in ``KelvinInferenceModel``.
        """
        scene_rescale = self.scene_rescale

        # ----- Encoder -----
        # Mirror KelvinDAv3Encoder.encode while consuming stacked tensors.
        B, V, H, W, _ = rgb.shape
        x = self.encoder.patch_embed_img(
            self.encoder.rgb_normalize(rearrange(rgb, "B V H W C -> (B V) C H W"))
        )
        x = rearrange(x, "(B V) h w C -> B V h w C", B=B, V=V)
        camera_encodings = self.encoder.embed_camera.forward(c2w, fov)
        with torch.autocast("cuda", enabled=True):
            img_feats, _ = self.encoder.vit.get_intermediate_features(
                x,
                block_indices=self.encoder.take_block_indices,
                global_cls_token=camera_encodings.unsqueeze(2),
            )
        encoded_deepest = img_feats[-1]

        # ----- Decoder static path -----
        # Run the decoder heads that feed the exported static layer. Per-pixel
        # masking happens in KelvinInferenceModel so it can use cuboid tracks.
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

        # Context head
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
        semantic_argmax = torch.argmax(context_semantic_logits, dim=-1)  # (B, V, H, W)

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

        # ----- Per-camera affine post-processing -----
        encoded_deepest_tokens = rearrange(encoded_deepest, "B V h w C -> B (V h w) C")
        _, affine_latents = self.post_processing.transform_tokens(
            encoded_deepest_tokens, camera_idxs
        )
        affine_matrix_3, affine_bias = self.post_processing.decode_affine(affine_latents)
        affine_matrix = torch.cat([affine_matrix_3, affine_bias[..., None]], dim=-1)

        return (
            gs_xyz,
            gs_world_quaternion,
            gs_scale,
            gs_opacity,
            context_rgb,
            semantic_argmax,
            context_world_normal,
            affine_matrix,
        )
