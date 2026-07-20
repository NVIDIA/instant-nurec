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

import torch
import torch.nn as nn

from instant_nurec.model.blocks.attention import AttentionBlock
from instant_nurec.model.blocks.layers import FeedForwardMLP
from instant_nurec.utils.geometry import so3_matrix_to_quat


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
