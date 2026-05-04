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

import logging

from typing import cast

import numpy as np
import torch
import torch.nn as _nn

from einops import rearrange
from torch import nn

from ncore.data import ConcreteCameraModelParametersUnion, OpenCVPinholeCameraModelParameters
from instant_nurec.config_schema.models import (
    KelvinDAv3EncoderConfig,
    KelvinModelConfig,
)
from instant_nurec.model.blocks.aa_vit import AlternateAttentionVisionTransformer
from instant_nurec.model.blocks.dav3 import CameraEncoder
from instant_nurec.model.blocks.embeds import PatchEmbed
from instant_nurec.model.backbone.base import (
    KelvinLatent,
    KelvinMultiscaleFeaturesLatent,
)
from instant_nurec.utils.sensor import to_simple_pinhole_model_parameters
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional


class _RGBNormalize(_nn.Module):
    """Wraps ``torchvision.transforms.v2.Normalize`` for tensors of shape
    ``(..., C, H, W)``. The mean/std are stored as non-persistent buffers
    so they don't appear in ``state_dict()``.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("_mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        return TF.normalize(
            x,
            mean=self._mean.flatten().tolist(),
            std=self._std.flatten().tolist(),
        )


logger = logging.getLogger(__name__)


class KelvinDAv3Encoder(nn.Module):
    def __init__(self, config: KelvinDAv3EncoderConfig, model_config: KelvinModelConfig):
        super().__init__()
        self.rgb_normalize = _RGBNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        embed_dim = config.embed_dim // 2
        patch_shape = model_config.patch_shape

        self.patch_embed_img = PatchEmbed(
            patch_shape=patch_shape,
            input_dim=3,
            embed_dim=embed_dim,
            norm=False,
        )
        self.embed_camera = CameraEncoder(input_dim=9, output_dim=embed_dim)

        self.vit = AlternateAttentionVisionTransformer(
            depth=config.depth,
            embed_dim=embed_dim,
            n_heads=config.n_heads,
            mlp_ratio=4.0,
            aa_start_block_idx=config.aa_start_block_idx,
            img_pos_embed_shape=518 // patch_shape[0],
            n_cls_tokens=1,
            with_default_global_cls_tokens=False,
            rope_frequency=100.0,
            checkpointing=config.checkpointing,
        )
        self.take_block_indices = config.take_block_indices

    @staticmethod
    def _fov_wh_from_pinhole(pinhole_parameters: OpenCVPinholeCameraModelParameters) -> torch.Tensor:
        """
        Computes the fov and width/height from the pinhole parameters.
        """
        fov_w = 2 * np.arctan2(pinhole_parameters.resolution[0] / 2, pinhole_parameters.focal_length[0])
        fov_h = 2 * np.arctan2(pinhole_parameters.resolution[1] / 2, pinhole_parameters.focal_length[1])
        return torch.tensor([fov_w, fov_h]).float()

    @torch.autocast("cuda", enabled=False)
    def encode(
        self,
        batches: list[DataAndRenderingBatch],
        scene_rescale: float = 1.0,
    ) -> KelvinLatent:
        batch_rgbs: list[torch.Tensor] = []
        batch_c2ws: list[torch.Tensor] = []
        batch_fovs: list[torch.Tensor] = []

        for batch in batches:
            data = unpack_optional(batch.data.camera)
            rendering = unpack_optional(unpack_optional(batch.rendering).camera)

            rgb = unpack_optional(data.labels.rgb)
            num_imgs, _, _ = rgb.shape[:3]
            batch_rgbs.append(rgb)

            # Use end of frame pose for c2w approximation
            c2w_frame_end = tquat_to_se3_matrix(rendering.poses_tquat_startend[:, 1, :], unbatch=False)
            c2w_frame_end[:, :3, 3] *= scene_rescale
            batch_c2ws.append(c2w_frame_end)

            # Use simple pinhole model for fov approximation (TODO: Investigate?)
            # Since prediction is depth so we should probably hint rays as accurate as possible.
            pinhole_parameters = [
                to_simple_pinhole_model_parameters(
                    cast(ConcreteCameraModelParametersUnion, rendering.sensor_model_parameters[vidx]),
                )
                for vidx in range(num_imgs)
            ]
            fov_wh = torch.stack([self._fov_wh_from_pinhole(pinhole_parameters[vidx]) for vidx in range(num_imgs)]).to(
                rgb.device
            )
            batch_fovs.append(fov_wh)

        # Assertions about shapes should come with the stack function
        rgbs_in = torch.stack(batch_rgbs, dim=0)
        B, V, H, W, _ = rgbs_in.shape

        x = self.patch_embed_img(self.rgb_normalize(rearrange(rgbs_in, "B V H W C -> (B V) C H W")))
        _, h, w, _ = x.shape  # h and w is the number of patches
        x = rearrange(x, "(B V) h w C -> B V h w C", B=B, V=V)

        # Compute camera encoding
        c2w_in, fov_in = torch.stack(batch_c2ws, dim=0), torch.stack(batch_fovs, dim=0)
        camera_encodings = self.embed_camera.forward(c2w_in, fov_in)

        with torch.autocast("cuda", enabled=True):
            img_feats, cls_tokens = self.vit.get_intermediate_features(
                x, block_indices=self.take_block_indices, global_cls_token=camera_encodings.unsqueeze(2)
            )

        return KelvinMultiscaleFeaturesLatent(features=img_feats, cls_tokens=cls_tokens)
