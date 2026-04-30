# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nre.viewer.abstract_viewer import RasterArtifacts, RenderAction, RenderStateMachine
from nre.viewer.dataset_interface import CameraTrajectoryData, CameraTrajectoryId, ViewerDatasetInterface
from nre.viewer.lightning_viewer import LightningNRMViewer, LightningSOViewer, ObserverCallback
from nre.viewer.ncore_dataset_interface import ViewerNCOREInterface
from nre.viewer.viewpoint import LookAtPose, Viewpoint


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "CameraTrajectoryId",
    "CameraTrajectoryData",
    "ViewerDatasetInterface",
    "ViewerNCOREInterface",
    "ObserverCallback",
    "RenderAction",
    "RasterArtifacts",
    "RenderStateMachine",
    "LightningSOViewer",
    "LightningNRMViewer",
    "LookAtPose",
    "Viewpoint",
]
