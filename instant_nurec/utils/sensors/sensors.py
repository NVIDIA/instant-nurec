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

# Most of the code here is copied from NCore in ncore/impl/sensors/ and adapted to use only torch (and not numpy).

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from ncore.sensors import FThetaCameraModel


def _compute_poses_and_timestamps_torch(
    T_sensor_world_startend_allviews: torch.Tensor,
    frame_idx: torch.Tensor,
    timestamps_startend_us_allviews: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame pose + timestamp indexing for the predict pipeline.

    Predict pins ``enable_calib=False``, ``rect_points_lb=None``,
    ``resolution=None``, ``embed_weights=None``, which reduces the kernel
    to ``T_out = T_in[frame_idx]`` and ``ts_out = ts_in[frame_idx]``.

    Args:
        T_sensor_world_startend_allviews: (V, 2, 4, 4) start/end SE(3) per view.
        frame_idx: (N,) int32 indices into V.
        timestamps_startend_us_allviews: (V, 2) int64 timestamps per view.

    Returns:
        (T_out, ts_out) of shapes (N, 2, 4, 4) and (N, 2).
    """
    if frame_idx.shape[0] == 0:
        return (
            torch.empty(
                (0, 2, 4, 4),
                device=T_sensor_world_startend_allviews.device,
                dtype=torch.float32,
            ),
            torch.empty(
                (0, 2),
                device=T_sensor_world_startend_allviews.device,
                dtype=torch.int64,
            ),
        )
    fidx = frame_idx.to(torch.int64)
    return T_sensor_world_startend_allviews[fidx], timestamps_startend_us_allviews[fidx]


ConcreteCameraModelsUnion: TypeAlias = FThetaCameraModel


@dataclass(kw_only=True, slots=True)
class RectSubsampledSensor:
    """
    Subsampled rectangular pixel region with offset i/j and dimension
    height/width.

    Note that the offset i/j and dimension height/width are relative to the
    scaled pixel domain. I.e., subsampling is applied first, then cropping.
    """

    width: int
    height: int
    i: int = 0
    j: int = 0
    subsample_factor: float = 1.0


class SensorModelComputations:
    @dataclass
    class PosesAndTimestampsStartendReturn:
        T_sensor_world_startend: torch.Tensor
        timestamps_startend_us: torch.Tensor  # (2,)
        timestamps_startend_us_gpu: torch.Tensor  # (1, 2)
        timestamps_startend_us_cpu: torch.Tensor  # (1, 2)

    @staticmethod
    def get_poses_and_timestamps_startend(
        T_sensor_world_startend_allviews: torch.Tensor,
        timestamps_startend_us_allviews: torch.Tensor,
        timestamps_startend_us_allviews_cpu: torch.Tensor,
        unique_frame_idx: int,
        unique_frame_idx_tensor: torch.Tensor,
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """GPU rolling-shutter pose + timestamp lookup.

        Predict requires CUDA tensors and pins ``FrameMeta.subsample=None``.
        """
        assert T_sensor_world_startend_allviews.is_cuda, (
            "get_poses_and_timestamps_startend requires CUDA tensors."
        )

        T_sensor_world_startend_batch, timestamps_startend_us_batch = (
            _compute_poses_and_timestamps_torch(
                T_sensor_world_startend_allviews,
                unique_frame_idx_tensor,
                timestamps_startend_us_allviews,
            )
        )

        # Squeeze batch dimension (batch_size=1) to get single frame result
        T_sensor_world_startend = T_sensor_world_startend_batch.squeeze(0)  # (2, 4, 4)
        timestamps_startend_us = timestamps_startend_us_batch.squeeze(0)  # (2,)
        timestamps_startend_us_gpu = timestamps_startend_us.unsqueeze(0)
        timestamps_startend_us_cpu = timestamps_startend_us_allviews_cpu[unique_frame_idx].unsqueeze(0)

        return SensorModelComputations.PosesAndTimestampsStartendReturn(
            T_sensor_world_startend=T_sensor_world_startend,
            timestamps_startend_us=timestamps_startend_us,
            timestamps_startend_us_gpu=timestamps_startend_us_gpu,
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
        )


