# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nre.config.base_schema import BaseConfigSchema
from nre.config.checkpoint import ArtifactConfig, CheckpointConfig
from nre.config.datamodule import BaseDatamoduleConfig, SODatamoduleConfig
from nre.config.dataset import (
    CuboidTracksConfig,
    LidarDynamicPointsConfig,
    NCoreDatasetConfig,
    ValidLidarPointsCuboidTrackConfig,
    ValidPixelsCuboidTrackConfig,
    ValidPixelsFrameMaskConfig,
    ValidPixelsSceneFlowConfig,
    ValidPixelsTrafficLightConfig,
)
from nre.config.logger import DummyLoggerConfig, TensorboardLoggerConfig, WandbLoggerConfig
from nre.config.prober import ProberConfig
from nre.config.scopedtimer import ScopedTimerConfig, VerbosityLevel
from nre.config.sensor import LidarModelsConfig, SensorConfig
from nre.config.systems import (
    GaussiansSystemConfig,
    NRendTestGaussiansSystemConfig,
)
from nre.config.trainer import TrainerConfig
from nre.config.version import Version
from nre.config.viewer import ViewerConfig


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "ArtifactConfig",
    "BaseConfigSchema",
    "BaseDatamoduleConfig",
    "CheckpointConfig",
    "CuboidTracksConfig",
    "DummyLoggerConfig",
    "GaussiansSystemConfig",
    "LidarDynamicPointsConfig",
    "LidarModelsConfig",
    "NCoreDatasetConfig",
    "NRendTestGaussiansSystemConfig",
    "ProberConfig",
    "ScopedTimerConfig",
    "SensorConfig",
    "SODatamoduleConfig",
    "TensorboardLoggerConfig",
    "TrainerConfig",
    "ValidLidarPointsCuboidTrackConfig",
    "ValidPixelsCuboidTrackConfig",
    "ValidPixelsFrameMaskConfig",
    "ValidPixelsSceneFlowConfig",
    "ValidPixelsTrafficLightConfig",
    "VerbosityLevel",
    "Version",
    "ViewerConfig",
    "WandbLoggerConfig",
]
