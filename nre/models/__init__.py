# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import nre.models.background as background
import nre.models.calib as calib
import nre.models.feature_volume as feature_volume
import nre.models.object_feature_volume as object_feature_volume
import nre.models.tracks_calib as tracks_calib

from nre.models.background import (
    BackgroundColor,
    BaseBackground,
    ExtraSignalSkyMlpBackground,
    SkipBackground,
    SkyMlpBackground,
)
from nre.models.composite import CompositeModel
from nre.models.custom_modules import (
    AlphaCompositing,
    RayAABBIntersector,
    RaySphereIntersector,
    WeightsFromAlphas,
    _TruncBCE,
    _TruncExp,
)
from nre.models.feature_volume import BaseFeatureVolume, TcnnMultiLevelEncoding
from nre.models.input_embedding import (
    BaseInputEmbedding,
    HolisticRemapTimeInputEmbedding,
    HolisticStepTimeInputEmbedding,
    IndividualRemapTimeInputEmbedding,
    IndividualStepTimeInputEmbedding,
    SkipInputEmbedding,
    WeightedInstanceInputEmbedding,
)
from nre.models.nn_extensions import LieGroupParameter
from nre.models.nrenderable import NRenderableModel
from nre.models.object_feature_volume import (
    BaseObjectFeatureVolume,
    HashGridObjectFeatureVolume,
)
from nre.models.post_processing import (
    BasePostProcessing,
    BilateralGridPerCamera,
    BilateralGridPerFrame,
    DinoV2PEPostProcessing,
    PPISPPostProcessing,
)
from nre.models.tracks_calib import (
    BaseTracksCalib,
    DirectTracksCalib,
    SkipTracksCalib,
    StaticTracksCalib,
    UnicycleTracksCalib,
)


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseBackground",
    "SkyMlpBackground",
    "ExtraSignalSkyMlpBackground",
    "BackgroundColor",
    "SkipBackground",
    "BilateralGridPerFrame",
    "BilateralGridPerCamera",
    "PPISPPostProcessing",
    "LieGroupParameter",
    "RayAABBIntersector",
    "RaySphereIntersector",
    "_TruncExp",
    "_TruncBCE",
    "AlphaCompositing",
    "WeightsFromAlphas",
    "TcnnMultiLevelEncoding",
    "BaseFeatureVolume",
    "BaseInputEmbedding",
    "SkipInputEmbedding",
    "WeightedInstanceInputEmbedding",
    "HolisticRemapTimeInputEmbedding",
    "HolisticStepTimeInputEmbedding",
    "IndividualRemapTimeInputEmbedding",
    "IndividualStepTimeInputEmbedding",
    "BaseObjectFeatureVolume",
    "HashGridObjectFeatureVolume",
    "BaseTracksCalib",
    "DirectTracksCalib",
    "StaticTracksCalib",
    "UnicycleTracksCalib",
    "SkipTracksCalib",
    "BasePostProcessing",
    "DinoV2PEPostProcessing",
    "NRenderableModel",
    "CompositeModel",
]
