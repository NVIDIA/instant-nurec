# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import re

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn

from nre.nrm.models.blocks.attention import AttentionBlock
from nre.nrm.models.blocks.layers import FeedForwardMLP
from nre.utils.geometry import so3_matrix_to_quat


@dataclass(kw_only=True)
class DAv3Config:
    depth: int
    n_heads: int
    take_block_indices: list[int]
    aa_start_block_idx: int
    dpt_reassemble_hidden_dims: list[int]
    embed_dim: int
    dpt_dim: int
    ffn_type: Literal["mlp", "swiglu"]


DAV3_CONFIGS: dict[str, DAv3Config] = {
    "small": DAv3Config(
        depth=12,
        n_heads=6,
        take_block_indices=[5, 7, 9, 11],
        aa_start_block_idx=4,
        dpt_reassemble_hidden_dims=[48, 96, 192, 384],
        embed_dim=384,
        dpt_dim=64,
        ffn_type="mlp",
    ),
    "base": DAv3Config(
        depth=12,
        n_heads=12,
        take_block_indices=[5, 7, 9, 11],
        aa_start_block_idx=4,
        dpt_reassemble_hidden_dims=[96, 192, 384, 768],
        embed_dim=768,
        dpt_dim=128,
        ffn_type="mlp",
    ),
    "large": DAv3Config(
        depth=24,
        n_heads=16,
        take_block_indices=[11, 15, 19, 23],
        aa_start_block_idx=8,
        dpt_reassemble_hidden_dims=[256, 512, 1024, 1024],
        embed_dim=1024,
        dpt_dim=256,
        ffn_type="mlp",
    ),
    "giant": DAv3Config(
        depth=40,
        n_heads=24,
        take_block_indices=[19, 27, 33, 39],
        aa_start_block_idx=13,
        dpt_reassemble_hidden_dims=[256, 512, 1024, 1024],
        embed_dim=1536,
        dpt_dim=256,
        ffn_type="swiglu",
    ),
}

_DAV3_WEIGHT_MAPPING_RULES: list[tuple[str, str | Callable | None]] = [
    (r"model\.backbone\.pretrained\.patch_embed\.(.*)", r"PATCH_EMBED.\1"),
    # Encoder backbone
    (r"model\.backbone\.pretrained\.blocks\.(.*)", r"BACKBONE.blocks.\1"),
    (r"model\.backbone\.pretrained\.norm\.(.*)", r"BACKBONE.norm_layer.\1"),
    # Heads
    (r"model\.head\.projects\.(\d+)\.(.*)", r"DPT_REASSEMBLE.proj_layers.\1.\2"),
    (r"model\.head\.resize_layers\.(\d+)\.(.*)", r"DPT_REASSEMBLE.resize_layers.\1.\2"),
    (r"model\.head\.norm\.(.*)", r"DPT_REASSEMBLE.norm_layer.\1"),
    (
        r"model\.head\.scratch\.layer(\d+)_rn\.(.*)",
        lambda m: f"DPT_REASSEMBLE.output_layers.{int(m.group(1)) - 1}.{m.group(2)}",
    ),
    # Matches: model.head.scratch.refinenet1.resConfUnit1.conv1.weight
    (
        r"model\.head\.scratch\.refinenet(\d+)\.(resConfUnit[12])\.(conv[12])\.(.*)",
        lambda m: f"DPT_DEPTH_HEAD.refinement_blocks.{int(m.group(1)) - 1}.{'res_block' if 'Unit1' in m.group(2) else 'main_block'}"
        + f".{'1' if 'conv1' in m.group(3) else '3'}.{m.group(4)}",
    ),
    (
        r"model\.head\.scratch\.refinenet(\d+)\.out_conv\.(.*)",
        lambda m: f"DPT_DEPTH_HEAD.refinement_blocks.{int(m.group(1)) - 1}.out_conv.{m.group(2)}",
    ),
    (
        r"model\.head\.scratch\.refinenet(\d+)_aux\.(resConfUnit[12])\.(conv[12])\.(.*)",
        lambda m: f"DPT_RAYS_HEAD.refinement_blocks.{int(m.group(1)) - 1}.{'res_block' if 'Unit1' in m.group(2) else 'main_block'}"
        + f".{'1' if 'conv1' in m.group(3) else '3'}.{m.group(4)}",
    ),
    (
        r"model\.head\.scratch\.refinenet(\d+)_aux\.out_conv\.(.*)",
        lambda m: f"DPT_RAYS_HEAD.refinement_blocks.{int(m.group(1)) - 1}.out_conv.{m.group(2)}",
    ),
    (r"model\.head\.scratch\.output_conv1\.(.*)", r"DPT_DEPTH_HEAD.before_conv.\1"),
    (r"model\.head\.scratch\.output_conv2\.(.*)", r"DPT_DEPTH_HEAD.after_conv.\1"),
    (r"model\.head\.scratch\.output_conv1_aux\.3\.(.*)", r"DPT_RAYS_HEAD.before_conv.\1"),
    (
        r"model\.head\.scratch\.output_conv2_aux\.3\.(\d+)\.(.*)",
        lambda m: f"DPT_RAYS_HEAD.after_conv.{ {0: 0, 2: 1, 5: 3}[int(m.group(1))] }.{m.group(2)}",
    ),
    (r"model\.head\.scratch\.output_conv1_aux\.(.*)", None),
    (r"model\.head\.scratch\.output_conv2_aux\.(.*)", None),
    # Others
    (r"model\.cam_enc\.(.*)", r"CAMERA_ENCODER.\1"),
    (r"model\.cam_dec\.(.*)", None),
    (r"model\.gs_head\.(.*)", None),
]


def convert_dav3_state_dict_to_nrm(
    dav3_state_dict: dict[str, torch.Tensor],
    patch_embed_name: str | None = "patch_embed",
    backbone_name: str | None = "backbone",
    dpt_reassemble_name: str | None = "dpt_reassemble",
    dpt_depth_head_name: str | None = "dpt_depth_head",
    dpt_rays_head_name: str | None = "dpt_rays_head",
    camera_encoder_name: str | None = "camera_encoder",
) -> dict[str, torch.Tensor]:
    """
    Converts a state dict from the DAv3 model to a state dict for the NRM model.
    """
    custom_name_mapping_dict: dict[str, str | None] = {
        "PATCH_EMBED": patch_embed_name,
        "BACKBONE": backbone_name,
        "DPT_REASSEMBLE": dpt_reassemble_name,
        "DPT_DEPTH_HEAD": dpt_depth_head_name,
        "DPT_RAYS_HEAD": dpt_rays_head_name,
        "CAMERA_ENCODER": camera_encoder_name,
    }
    compiled_rules = [(re.compile(pattern), target) for pattern, target in _DAV3_WEIGHT_MAPPING_RULES]

    new_state_dict: dict[str, torch.Tensor] = {}
    for k, v in dav3_state_dict.items():
        parsed_dict: dict[str, torch.Tensor] | None = None
        for pattern, target_template in compiled_rules:
            match = pattern.fullmatch(k)
            if match:
                if target_template is None:
                    parsed_dict = {}
                if isinstance(target_template, str):
                    parsed_dict = {match.expand(target_template): v}
                elif callable(target_template):
                    parsed_dict = {target_template(match): v}
                break
        if parsed_dict is None:
            if k.startswith(backbone_prefix := "model.backbone.pretrained."):
                internal_key = k.replace(backbone_prefix, "")
                if internal_key.startswith("cls_token"):
                    parsed_dict = {"BACKBONE.cls_tokens": v.squeeze(1)}
                elif internal_key.startswith("pos_embed"):
                    cls_pos = v[:, 0:1, :]
                    img_pos = v[:, 1:, :]
                    grid_size = int(math.sqrt(img_pos.shape[1]))
                    parsed_dict = {
                        "BACKBONE.cls_pos_embed": cls_pos.squeeze(1),
                        "BACKBONE.img_pos_embed": img_pos.squeeze(0).view(grid_size, grid_size, img_pos.shape[2]),
                    }
                elif internal_key == "camera_token":
                    parsed_dict = {"BACKBONE.default_global_cls_tokens": v.moveaxis(0, 1)}

        assert parsed_dict is not None, f"Unknown key in DAv3 state dict: {k}."

        for new_k, new_v in parsed_dict.items():
            # Check if we need to map the final key to a custom name specified by the user kwargs.
            # If user provides None, then we should not include it in the new state dict.
            for custom_name, custom_value in custom_name_mapping_dict.items():
                if custom_name in new_k:
                    if custom_value is not None:
                        new_k = new_k.replace(custom_name, custom_value)
                        new_state_dict[new_k] = new_v
                    break

    return new_state_dict


class CameraEncoder(nn.Module):
    """
    Encode extrinsics and intrinsics to pose encoding (to be used as CLS tokens)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 4,
        n_heads: int = 16,
        mlp_ratio: float = 4.0,
        layer_scale_init_values: float = 0.01,
    ):
        super().__init__()
        self.pose_branch = FeedForwardMLP(
            input_dim=input_dim,
            hidden_dim=output_dim // 2,
            output_dim=output_dim,
        )
        self.token_norm = nn.LayerNorm([output_dim])
        self.trunk = nn.Sequential(
            *[
                AttentionBlock(
                    input_dim=output_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    layer_scale_init_values=layer_scale_init_values,
                )
                for _ in range(depth)
            ]
        )
        self.trunk_norm = nn.LayerNorm([output_dim])

    # Always operate in high-precision mode
    @torch.autocast("cuda", enabled=False)
    def forward(self, T_camera_world: torch.Tensor, fov_wh: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T_camera_world: (B, V, 4, 4) camera-to-world transformation matrices
            fov_wh: (B, V, 2) field of view (fov_w, fov_h)

        Returns:
            pose_tokens: (B, V, D) pose tokens
        """
        B, V, _, _ = T_camera_world.shape
        quaternion = so3_matrix_to_quat(T_camera_world[..., :3, :3].float()).reshape(B, V, 4)
        quaternion = torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)
        translation = T_camera_world[..., :3, 3].float()
        pose_encoding = torch.cat([translation, quaternion, fov_wh[..., [1, 0]].float()], dim=-1)  # (B, V, 9)
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens
