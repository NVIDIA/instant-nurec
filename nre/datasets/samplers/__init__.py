# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import nre.datasets.samplers.base as base
import nre.datasets.samplers.holdout as holdout
import nre.datasets.samplers.image_crop as image_crop
import nre.datasets.samplers.lidar_crop as lidar_crop
import nre.datasets.samplers.semantic as semantic
import nre.datasets.samplers.timestamp as timestamp
import nre.datasets.samplers.uniform as uniform
import nre.datasets.samplers.weighted as weighted

from nre.datasets.samplers.base import (
    BaseBatchSampler,
    BaseCameraPixelSampler,
    BaseFrameSampler,
    BaseLidarPointSampler,
    BaseSensorSampler,
    DefaultBatchSampler,
    SkipCameraPixelSampler,
    SkipFrameSampler,
    SkipLidarPointSampler,
    SkipSensorSampler,
)
from nre.datasets.samplers.holdout import HoldOutFrameSampler
from nre.datasets.samplers.image_crop import ImageCropCameraPixelSampler
from nre.datasets.samplers.lidar_crop import LidarPointCloudCropSampler
from nre.datasets.samplers.semantic import SemanticLidarPointSampler
from nre.datasets.samplers.timestamp import TimestampFrameSampler
from nre.datasets.samplers.uniform import (
    UniformFrameSampler,
    UniformLidarPointSampler,
    UniformSensorSampler,
)
from nre.datasets.samplers.weighted import WeightedSensorSampler


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseSensorSampler",
    "BaseFrameSampler",
    "BaseCameraPixelSampler",
    "BaseLidarPointSampler",
    "SkipSensorSampler",
    "SkipFrameSampler",
    "SkipCameraPixelSampler",
    "SkipLidarPointSampler",
    "BaseBatchSampler",
    "DefaultBatchSampler",
    "ImageCropCameraPixelSampler",
    "SemanticLidarPointSampler",
    "TimestampFrameSampler",
    "UniformSensorSampler",
    "UniformFrameSampler",
    "UniformLidarPointSampler",
    "WeightedSensorSampler",
    "HoldOutFrameSampler",
    "LidarPointCloudCropSampler",
]
