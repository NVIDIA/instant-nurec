# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import nre.datasets.ncore as ncore
import nre.datasets.samplers as samplers

from nre.datasets.base import BaseDataset, BaseDataSource, RigTrajectoriesProvider
from nre.datasets.ncore import NCOREDataset, NCOREDataSource, NCORESequentialDataset, NCORETrainDataset
from nre.datasets.registry import make
from nre.datasets.tracks import CuboidTracks, RayIntersectionTransformFilter, TrackFlags


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "NCOREDataset",
    "NCOREDataSource",
    "NCORESequentialDataset",
    "NCORETrainDataset",
    "TrackFlags",
    "RayIntersectionTransformFilter",
    "CuboidTracks",
    "BaseDataset",
    "BaseDataSource",
    "RigTrajectoriesProvider",
    "CameraDataset",
]
