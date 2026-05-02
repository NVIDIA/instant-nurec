# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Most of the code here is copied from NCore in ncore/impl/sensors/ and adapted to use only torch (and not numpy).

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, Union, cast

import torch

from libs.sensors.kernels.pose_calib import compute_poses_and_timestamps
from ncore.data import ShutterType
from ncore.sensors import (
    FThetaCameraModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
)


ConcreteCameraModelsUnion: TypeAlias = Union[FThetaCameraModel, OpenCVFisheyeCameraModel, OpenCVPinholeCameraModel]


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
        sensor_models: torch.nn.ModuleDict,
        unique_frame_idx: int,
        unique_frame_idx_tensor: torch.Tensor,
        unique_sensor_idx_str: str,
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """GPU rolling-shutter interpolation via the Slang kernel.

        Standalone predict requires CUDA tensors and pins
        `FrameMeta.subsample = None`; the NRE-side compiled CPU fallback and
        the per-pixel-rect `interpolate_rect_timestamps_cpu` path were
        unreachable and dropped in Phase 1 step 4.3.
        """
        assert T_sensor_world_startend_allviews.is_cuda, (
            "get_poses_and_timestamps_startend requires CUDA tensors in the standalone predict pipeline."
        )
        shutter_type = cast(ShutterType, sensor_models[unique_sensor_idx_str].shutter_type)

        T_sensor_world_startend_batch, timestamps_startend_us_batch = compute_poses_and_timestamps(
            T_sensor_world_startend_allviews,
            None,  # embed_weights: standalone predict has no per-frame calibration deltas
            unique_frame_idx_tensor,
            None,  # subsample_rect_points_lb: FrameMeta.subsample is always None
            None,  # subsample_resolution: FrameMeta.subsample is always None
            timestamps_startend_us_allviews,
            shutter_type.value,
            False,  # enable_calib: dead in standalone predict (no calibration learning)
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


