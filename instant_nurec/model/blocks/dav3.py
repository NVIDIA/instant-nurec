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

import math
import re

from collections.abc import Callable

import torch
import torch.nn as nn

from instant_nurec.model.blocks.attention import AttentionBlock
from instant_nurec.model.blocks.layers import FeedForwardMLP
from instant_nurec.utils.geometry import so3_matrix_to_quat


_WeightTarget = str | Callable[[re.Match[str]], str] | None
_DAV3_WEIGHT_RULES: tuple[tuple[str, _WeightTarget], ...] = (
    (r"model\.backbone\.pretrained\.patch_embed\.(.*)", r"PATCH_EMBED.\1"),
    (r"model\.backbone\.pretrained\.blocks\.(.*)", r"BACKBONE.blocks.\1"),
    (r"model\.backbone\.pretrained\.norm\.(.*)", r"BACKBONE.norm_layer.\1"),
    (r"model\.head\.projects\.(\d+)\.(.*)", r"DPT_REASSEMBLE.proj_layers.\1.\2"),
    (r"model\.head\.resize_layers\.(\d+)\.(.*)", r"DPT_REASSEMBLE.resize_layers.\1.\2"),
    (r"model\.head\.norm\.(.*)", r"DPT_REASSEMBLE.norm_layer.\1"),
    (
        r"model\.head\.scratch\.layer(\d+)_rn\.(.*)",
        lambda match: f"DPT_REASSEMBLE.output_layers.{int(match.group(1)) - 1}.{match.group(2)}",
    ),
    (
        r"model\.head\.scratch\.refinenet(\d+)\.(resConfUnit[12])\.(conv[12])\.(.*)",
        lambda match: (
            f"DPT_DEPTH_HEAD.refinement_blocks.{int(match.group(1)) - 1}."
            f"{'res_block' if match.group(2) == 'resConfUnit1' else 'main_block'}."
            f"{'1' if match.group(3) == 'conv1' else '3'}.{match.group(4)}"
        ),
    ),
    (
        r"model\.head\.scratch\.refinenet(\d+)\.out_conv\.(.*)",
        lambda match: f"DPT_DEPTH_HEAD.refinement_blocks.{int(match.group(1)) - 1}.out_conv.{match.group(2)}",
    ),
    (
        r"model\.head\.scratch\.refinenet(\d+)_aux\.(resConfUnit[12])\.(conv[12])\.(.*)",
        lambda match: (
            f"DPT_RAYS_HEAD.refinement_blocks.{int(match.group(1)) - 1}."
            f"{'res_block' if match.group(2) == 'resConfUnit1' else 'main_block'}."
            f"{'1' if match.group(3) == 'conv1' else '3'}.{match.group(4)}"
        ),
    ),
    (
        r"model\.head\.scratch\.refinenet(\d+)_aux\.out_conv\.(.*)",
        lambda match: f"DPT_RAYS_HEAD.refinement_blocks.{int(match.group(1)) - 1}.out_conv.{match.group(2)}",
    ),
    (r"model\.head\.scratch\.output_conv1\.(.*)", r"DPT_DEPTH_HEAD.before_conv.\1"),
    (r"model\.head\.scratch\.output_conv2\.(.*)", r"DPT_DEPTH_HEAD.after_conv.\1"),
    (r"model\.head\.scratch\.output_conv1_aux\.3\.(.*)", r"DPT_RAYS_HEAD.before_conv.\1"),
    (
        r"model\.head\.scratch\.output_conv2_aux\.3\.(\d+)\.(.*)",
        lambda match: (
            f"DPT_RAYS_HEAD.after_conv.{ {0: 0, 2: 1, 5: 3}[int(match.group(1))] }.{match.group(2)}"
        ),
    ),
    (r"model\.head\.scratch\.output_conv1_aux\.(.*)", None),
    (r"model\.head\.scratch\.output_conv2_aux\.(.*)", None),
    (r"model\.cam_enc\.(.*)", r"CAMERA_ENCODER.\1"),
    (r"model\.cam_dec\.(.*)", None),
    (r"model\.gs_head\.(.*)", None),
)


def convert_dav3_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    patch_embed_name: str | None,
    backbone_name: str | None,
    dpt_reassemble_name: str | None,
    dpt_depth_head_name: str | None,
    dpt_rays_head_name: str | None,
    camera_encoder_name: str | None,
) -> dict[str, torch.Tensor]:
    """Map an official Depth Anything V3 checkpoint into Instant NuRec modules."""

    names = {
        "PATCH_EMBED": patch_embed_name,
        "BACKBONE": backbone_name,
        "DPT_REASSEMBLE": dpt_reassemble_name,
        "DPT_DEPTH_HEAD": dpt_depth_head_name,
        "DPT_RAYS_HEAD": dpt_rays_head_name,
        "CAMERA_ENCODER": camera_encoder_name,
    }
    compiled_rules = tuple((re.compile(pattern), target) for pattern, target in _DAV3_WEIGHT_RULES)
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        mapped: dict[str, torch.Tensor] | None = None
        for pattern, target in compiled_rules:
            match = pattern.fullmatch(key)
            if match is None:
                continue
            if target is None:
                mapped = {}
            elif isinstance(target, str):
                mapped = {match.expand(target): value}
            else:
                mapped = {target(match): value}
            break

        if mapped is None and key.startswith("model.backbone.pretrained."):
            source_key = key.removeprefix("model.backbone.pretrained.")
            if source_key == "cls_token":
                mapped = {"BACKBONE.cls_tokens": value.squeeze(1)}
            elif source_key == "pos_embed":
                class_position = value[:, :1]
                image_position = value[:, 1:]
                grid_size = math.isqrt(image_position.shape[1])
                if grid_size * grid_size != image_position.shape[1]:
                    raise ValueError("DAv3 positional embedding is not a square image grid")
                mapped = {
                    "BACKBONE.cls_pos_embed": class_position.squeeze(1),
                    "BACKBONE.img_pos_embed": image_position.squeeze(0).reshape(
                        grid_size, grid_size, image_position.shape[2]
                    ),
                }
            elif source_key == "camera_token":
                mapped = {"BACKBONE.default_global_cls_tokens": value.moveaxis(0, 1)}
        if mapped is None:
            raise ValueError(f"Unknown DAv3 checkpoint key: {key}")

        for mapped_key, mapped_value in mapped.items():
            for placeholder, replacement in names.items():
                if placeholder not in mapped_key:
                    continue
                if replacement is not None:
                    converted[mapped_key.replace(placeholder, replacement)] = mapped_value
                break
    return converted


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
