# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import torch
import torchvision.transforms as transforms

from einops import rearrange
from torch import nn

from ncore.data import ConcreteCameraModelParametersUnion, OpenCVPinholeCameraModelParameters
from nre.models.nn_extensions import TypedModuleList
from nre.nrm.config.models import (
    KelvinDAv3EncoderConfig,
    KelvinModelConfig,
)
from nre.nrm.models.blocks.aa_vit import AlternateAttentionVisionTransformer
from nre.nrm.models.blocks.dav3 import CameraEncoder, convert_dav3_state_dict_to_nrm
from nre.nrm.models.blocks.embeds import PatchEmbed
from nre.nrm.models.kelvin_backbone.base import (
    KelvinLatent,
    KelvinMultiscaleFeaturesLatent,
)
from nre.nrm.utils.motion import TimeRemapping
from nre.nrm.utils.sensor import to_simple_pinhole_model_parameters
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.geometry import tquat_to_se3_matrix
from nre.utils.misc import unpack_optional
from nre.utils.profiling import ScopedTimer


logger = logging.getLogger(__name__)


class KelvinEncoderBase(nn.Module, ABC):
    @abstractmethod
    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        """
        Initialize the weights of the model from the loaded state dicts.
        """

    @abstractmethod
    def encode(
        self,
        batches: list[DataAndRenderingBatch],
        time_remappings: list[TimeRemapping],
        scene_rescale: float = 1.0,
    ) -> KelvinLatent:
        """
        Encode the input batch into a latent representation.
        """

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please call encode() method directly.")

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs):
        # Do nothing by default
        pass


class KelvinDAv3Encoder(KelvinEncoderBase):
    def __init__(self, config: KelvinDAv3EncoderConfig, model_config: KelvinModelConfig):
        super().__init__()
        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)

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
            ffn_type=config.ffn_type,
            checkpointing=config.checkpointing,
        )
        self.take_block_indices = config.take_block_indices

    def initialize_weights(self, loaded_state_dicts: dict[str, dict[str, torch.Tensor]]):
        if "dav3" not in loaded_state_dicts:
            logger.warning("[KelvinDAv3Encoder] No dav3 state dict found, skipping weight initialization.")
            return

        state_dict = convert_dav3_state_dict_to_nrm(
            loaded_state_dicts["dav3"],
            patch_embed_name="patch_embed_img",
            backbone_name="vit",
            dpt_reassemble_name=None,
            dpt_depth_head_name=None,
            dpt_rays_head_name=None,
            camera_encoder_name="embed_camera",
        )
        del state_dict["vit.default_global_cls_tokens"]
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _fov_wh_from_pinhole(pinhole_parameters: OpenCVPinholeCameraModelParameters) -> torch.Tensor:
        """
        Computes the fov and width/height from the pinhole parameters.
        """
        fov_w = 2 * np.arctan2(pinhole_parameters.resolution[0] / 2, pinhole_parameters.focal_length[0])
        fov_h = 2 * np.arctan2(pinhole_parameters.resolution[1] / 2, pinhole_parameters.focal_length[1])
        return torch.tensor([fov_w, fov_h]).float()

    @ScopedTimer("KelvinDAv3Encoder.encode")
    @torch.autocast("cuda", enabled=False)
    def encode(
        self,
        batches: list[DataAndRenderingBatch],
        time_remappings: list[TimeRemapping],
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
                    method="horizontal",
                    reduce="min",
                    percentile=1.0,
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


def make_encoder(config: KelvinModelConfig) -> KelvinEncoderBase:
    if isinstance(config.encoder, KelvinDAv3EncoderConfig):
        return KelvinDAv3Encoder(config.encoder, config)
    raise ValueError(f"Unsupported encoder config: {config.encoder}")
