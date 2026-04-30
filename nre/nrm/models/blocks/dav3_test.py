# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from depth_anything_3.api import DepthAnything3  # type: ignore[import-not-found]
from depth_anything_3.model.utils.transform import mat_to_quat as dav3_mat_to_quat  # type: ignore[import-not-found]
from depth_anything_3.model.utils.transform import standardize_quaternion  # type: ignore[import-not-found]
from einops import rearrange
from parameterized import parameterized
from scipy.spatial.transform import Rotation as R

from nre.nrm.models.blocks.aa_vit import AlternateAttentionVisionTransformer
from nre.nrm.models.blocks.dav3 import DAV3_CONFIGS, CameraEncoder, DAv3Config, convert_dav3_state_dict_to_nrm
from nre.nrm.models.blocks.dpt import DPTFusionHead, DPTReassembleBlock
from nre.nrm.models.blocks.embeds import PatchEmbed, PositionalEmbed
from nre.utils.geometry import so3_matrix_to_quat


class TestDepthAnythingV3Reproduction(unittest.TestCase):
    """Integration test to make sure DAv3 results can be reproduced."""

    class DepthAnythingV3Adapter(nn.Module):
        """Adapter module that supports loading the full state dict of DAv3 pretrained model"""

        def __init__(self, config: DAv3Config):
            super().__init__()

            self.backbone = AlternateAttentionVisionTransformer(
                depth=config.depth,
                embed_dim=config.embed_dim,
                n_heads=config.n_heads,
                mlp_ratio=4.0,
                aa_start_block_idx=config.aa_start_block_idx,
                img_pos_embed_shape=37,  # 518 / 14
                n_cls_tokens=1,
                with_default_global_cls_tokens=True,
                rope_frequency=100.0,
                ffn_type=config.ffn_type,
            )
            self.patch_embed = PatchEmbed(patch_shape=(14, 14), input_dim=3, embed_dim=config.embed_dim, norm=False)
            self.out_layers = config.take_block_indices

            self.dpt_reassemble = DPTReassembleBlock(
                input_dim=config.embed_dim * 2,
                output_dim=config.dpt_dim,
                n_blocks=len(self.out_layers),
                hidden_dims=tuple(config.dpt_reassemble_hidden_dims),
                pos_embed_strength=0.1,
            )
            self.dpt_depth_head = DPTFusionHead(
                input_dim=config.dpt_dim,
                output_dim=1 + 1,
                n_blocks=len(self.out_layers),
                before_conv="1-layer",
                after_conv="2-layers",
                after_conv_dim=32,
                pos_embed_strength=0.1,
            )
            self.dpt_rays_head = DPTFusionHead(
                input_dim=config.dpt_dim,
                output_dim=6 + 1,
                n_blocks=len(self.out_layers),
                before_conv="5-layers",
                after_conv="2-layers-w-norm",
                after_conv_dim=32,
                pos_embed_strength=0.1,
            )
            self.camera_encoder = CameraEncoder(input_dim=9, output_dim=config.embed_dim)

    def setUp(self):
        self.device = torch.device("cuda")
        self.batch_size = 2
        self.n_views = 3

    @parameterized.expand([("small",), ("base",), ("large",), ("giant",)])
    @torch.no_grad()
    def test_forward_identity(self, model_size: Literal["small", "base", "large", "giant"]):
        dav3_api = DepthAnything3.from_pretrained(f"depth-anything/DA3-{model_size.upper()}")
        dav3_api.to(self.device).eval()

        # Adapt the state dict to the internal model
        our_model = self.DepthAnythingV3Adapter(config=DAV3_CONFIGS[model_size])
        our_model_state_dict = convert_dav3_state_dict_to_nrm(dav3_api.state_dict())
        our_model.load_state_dict(our_model_state_dict)
        our_model.to(self.device).eval()

        # Create a random batch of images and random camera poses
        torch.random.manual_seed(0)
        input_images = torch.randn(self.batch_size, self.n_views, 3, 308, 504, device=self.device)
        input_c2w = torch.eye(4, device=self.device).repeat(self.batch_size * self.n_views, 1, 1)
        input_c2w[:, :3, :3] = torch.from_numpy(
            R.random(self.batch_size * self.n_views, random_state=np.random.RandomState(0)).as_matrix()
        ).to(self.device)
        input_c2w[:, :3, 3] = torch.randn(self.batch_size * self.n_views, 3, device=self.device)
        input_c2w = input_c2w.reshape(self.batch_size, self.n_views, 4, 4)
        input_fovs = torch.rand(self.batch_size, self.n_views, 2, device=self.device) * np.pi / 2.0

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.autocast(device_type=input_images.device.type, dtype=autocast_dtype):
            x = input_images
            pd_camera_tokens = our_model.camera_encoder.forward(input_c2w, input_fovs)

            B, V, C, H, W = x.shape
            x = rearrange(x, "B V C H W -> (B V) C H W")
            x = our_model.patch_embed(x)
            x = rearrange(x, "(B V) h w C -> B V h w C", B=B, V=V)
            pd_img_feats, pd_cls_feats = our_model.backbone.get_intermediate_features(
                x, block_indices=our_model.out_layers, global_cls_token=pd_camera_tokens.unsqueeze(2)
            )
            pd_img_feats = [rearrange(feat, "B V h w C -> (B V) h w C") for feat in pd_img_feats]

            # In DAv3, the freq embedding is implemented with pow and reciprocal.
            # While mathematically equivalent, there might be a very small numerical difference.
            # For full reproducibility, we monkey patch the same embedding here.
            ours_sincos_pos_embed = PositionalEmbed.get_1d_sincos_pos_embed

            def dav3_sincos_pos_embed(x: torch.Tensor, embed_dim: int, T: float) -> torch.Tensor:
                omega = torch.arange(embed_dim // 2, dtype=x.dtype, device=x.device) / (embed_dim // 2)
                omega = 1.0 / (T**omega)
                x = x[..., None] * omega  # (..., D//2)
                x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (..., D)
                return x

            PositionalEmbed.get_1d_sincos_pos_embed = dav3_sincos_pos_embed

            with torch.autocast(device_type=x.device.type, enabled=False):
                pd_dpt_feats = our_model.dpt_reassemble(pd_img_feats)
                pd_pred_depth = our_model.dpt_depth_head(pd_dpt_feats, output_shape=(H, W))
                pd_pred_rays = our_model.dpt_rays_head(pd_dpt_feats)

            PositionalEmbed.get_1d_sincos_pos_embed = ours_sincos_pos_embed

        # Forward using DAv3 model (we bypass the API to avoid pre-processing)
        # Logic following src/depth_anything_3/model/dav3.py:forward()
        with torch.autocast(device_type=input_images.device.type, dtype=autocast_dtype):
            x = input_images
            extrinsics = torch.linalg.inv(input_c2w)
            intrinsics = torch.eye(3, device=self.device).repeat(self.batch_size, self.n_views, 1, 1)
            intrinsics[..., 0, 0] = x.shape[-1] / (2.0 * torch.tan(input_fovs[..., 0] / 2.0))
            intrinsics[..., 1, 1] = x.shape[-2] / (2.0 * torch.tan(input_fovs[..., 1] / 2.0))
            with torch.autocast(device_type=x.device.type, enabled=False):
                gt_camera_tokens = dav3_api.model.cam_enc(extrinsics, intrinsics, x.shape[-2:])

            # Use pd_camera_tokens since little difference (caused mainly by mat/quat conversion)
            # can result in large diff in output
            gt_feats, _ = dav3_api.model.backbone(
                x, cam_token=pd_camera_tokens, export_feat_layers=our_model.out_layers
            )

            H, W = x.shape[-2], x.shape[-1]
            with torch.autocast(device_type=x.device.type, enabled=False):
                gt_output = dav3_api.model._process_depth_head(gt_feats, H, W)

        # Compare camera tokens
        self.assertTrue(torch.allclose(pd_camera_tokens, gt_camera_tokens, atol=1e-6))

        # Compare image and cls tokens
        gt_img_feats = [rearrange(t[0], "B V hw C -> (B V) hw C") for t in gt_feats]
        pd_img_feats = [rearrange(t, "BV h w C -> BV (h w) C") for t in pd_img_feats]

        for gt_feat, pd_feat in zip(gt_img_feats, pd_img_feats):
            self.assertTrue(torch.allclose(gt_feat, pd_feat, atol=1e-6))

        gt_cls_feats = [t[1] for t in gt_feats]
        pd_cls_feats = [t.squeeze(2) for t in pd_cls_feats]

        for gt_feat, pd_feat in zip(gt_cls_feats, pd_cls_feats):
            self.assertTrue(torch.allclose(gt_feat, pd_feat, atol=1e-6))

        # Compare final depth and ray predictions
        pd_depth = rearrange(torch.exp(pd_pred_depth[:, 0]), "(B V) H W -> B V H W", B=B, V=V)
        pd_depth_conf = rearrange(torch.exp(pd_pred_depth[:, 1]) + 1, "(B V) H W -> B V H W", B=B, V=V)
        gt_depth = gt_output.depth
        gt_depth_conf = gt_output.depth_conf

        pd_rays = rearrange(pd_pred_rays[:, :6], "(B V) C H W -> B V H W C", B=B, V=V, C=6)
        pd_rays_conf = rearrange(torch.exp(pd_pred_rays[:, 6]) + 1, "(B V) H W -> B V H W", B=B, V=V)
        gt_rays = gt_output.ray
        gt_rays_conf = gt_output.ray_conf

        self.assertTrue(torch.allclose(pd_depth, gt_depth, atol=1e-6))
        self.assertTrue(torch.allclose(pd_depth_conf, gt_depth_conf, atol=1e-6))

        self.assertTrue(torch.allclose(pd_rays, gt_rays, atol=1e-5))
        self.assertTrue(torch.allclose(pd_rays_conf, gt_rays_conf, atol=1e-5))

    def test_mat_to_quat_identity(self):
        # Generate random rotation matrices
        random_so3 = torch.from_numpy(R.random(100, random_state=np.random.RandomState(0)).as_matrix()).to(self.device)

        # Convert to quaternion (note that DAv3 requires standardized quaternions i.e. last dim is positive)
        nre_quat = standardize_quaternion(so3_matrix_to_quat(random_so3))
        dav3_quat = standardize_quaternion(dav3_mat_to_quat(random_so3))
        self.assertTrue(torch.allclose(nre_quat, dav3_quat, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
