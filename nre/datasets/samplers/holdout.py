# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from omegaconf import DictConfig

from nre.utils.misc import unpack_optional


if TYPE_CHECKING:
    from nre.datasets.ncore import NCORETrainDataset  # pycena: skip
from nre.datasets.samplers.base import (
    BaseFrameSampler,
    FrameSamplerReturn,
)


class HoldOutFrameSampler(BaseFrameSampler):
    """Implements a hold-out sampling of sensor frames for training (every N frames will be excluded)"""

    include_frames_start: int | None
    exclude_frames_start: int | None
    include_every_n_frames: int | None
    exclude_every_n_frames: int | None

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        super().__init__(config, dataset)

        # Initialization to work even if one of these keys are missing from the config (backward compatibility).
        # It is not allowed to use HoldOutFrameSampler if both keys are missing from the config.
        self.include_every_n_frames = config.get("include_every_n_frames", None)
        self.exclude_every_n_frames = config.get("exclude_every_n_frames", None)

        # Initialization to work even if the keys are missing from the config (backward compatibility).
        # include_frames_start was introduced in the config after include_every_n_frames.
        if "include_frames_start" in config:
            self.include_frames_start = config.include_frames_start
        elif self.include_every_n_frames is None:
            self.include_frames_start = None
        else:
            self.include_frames_start = 0

        # Initialization to work even if the keys are missing from the config (backward compatibility).
        # include_frames_start was introduced in the config after include_every_n_frames.
        if "exclude_frames_start" in config:
            self.exclude_frames_start = config.exclude_frames_start
        elif self.exclude_every_n_frames is None:
            self.exclude_frames_start = None
        else:
            self.exclude_frames_start = 0

        assert (
            # Specifying frames to exclude
            self.include_frames_start is None
            and self.include_every_n_frames is None
            and self.exclude_frames_start is not None
            and self.exclude_every_n_frames is not None
        ) or (
            self.include_frames_start is not None
            and self.include_every_n_frames is not None
            and self.exclude_frames_start is None
            and self.exclude_every_n_frames is None
        ), (
            f"{self.__class__.__name__} Either specify "
            "frames to include (via include_frames_start and include_every_n_frames) or "
            "frames to exclude (via exclude_frames_start and exclude_every_n_frames) but not both"
        )

    def sample_frame(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        unique_sensor_id: str,
    ) -> FrameSamplerReturn:
        """Implementation of 'FrameSampler' protocol sampling from frame_range except every Nth frame"""

        if self.include_every_n_frames is not None:
            assert self.include_frames_start is not None
            p = np.zeros((len(frame_range),))
            p[self.include_frames_start :: self.include_every_n_frames] = 1
        else:
            assert self.exclude_frames_start is not None and self.exclude_every_n_frames is not None
            p = np.ones((len(frame_range),))
            p[self.exclude_frames_start :: self.exclude_every_n_frames] = 0

        p = p / p.sum()
        return FrameSamplerReturn(
            sampled_frame_idx=rng.choice(frame_range, size=1, replace=False, shuffle=False, p=p).item()
        )


BaseFrameSampler.register_to_frame_sampler_factory("holdout", HoldOutFrameSampler)
