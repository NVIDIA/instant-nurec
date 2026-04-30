# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import omegaconf

from nre.datasets.samplers.base import (
    BaseLidarPointSampler,
    LidarPointSamplerReturn,
)
from nre.utils.profiling import ScopedTimer, TimingTag


if TYPE_CHECKING:
    from nre.datasets.ncore import NCORETrainDataset


# TODO[zg/jme]: this sampler is not implementing cropping functionality yet - will be added along with ray-drop supervision
class LidarPointCloudCropSampler(BaseLidarPointSampler):
    """Sampler for LiDAR frames that can return full point cloud frames or subsample them randomly"""

    def __init__(
        self,
        config: omegaconf.dictconfig.DictConfig,
        dataset: NCORETrainDataset,
    ) -> None:
        self.subsample: int = config.subsample  # Subsampling factor, if 1 all points are kept

        assert isinstance(self.subsample, int) and self.subsample >= 1, (
            f"{self.__class__.__name__}: Subsample factor must be an integer >= 1, got {self.subsample}"
        )

    @ScopedTimer("LidarFrameSampler.sample_lidar_points", TimingTag.DATALOADER)
    def sample_lidar_points(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_point_samples: int,  # will be ignored in this sampler
        frame_valid_points_mask: npt.NDArray,
        unique_lidar_id: str,
        lidar_frame_idx: int,
    ) -> LidarPointSamplerReturn:
        """Samples full point cloud or randomly subsampled set of valid points"""

        # We currently sample from *valid* points only (this is different to current camera image-crop sampler)
        sampled_point_idxs = np.where(frame_valid_points_mask)[0]
        # If subsample is greater than 1, subsample the points
        if self.subsample > 1:
            # Calculate number of points to sample
            n_samples = int(len(sampled_point_idxs) / self.subsample)
            # Randomly sample indices without replacement
            sampled_point_idxs = rng.choice(sampled_point_idxs, size=n_samples, replace=False)

        return LidarPointSamplerReturn(sampled_point_idxs=sampled_point_idxs)


BaseLidarPointSampler.register_to_lidar_point_sampler_factory("lidar-crop", LidarPointCloudCropSampler)
