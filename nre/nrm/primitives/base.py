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

from abc import abstractmethod
from typing import Literal, Self, TypeVar

import torch

from nre.models.gaussians.renderers import BaseGaussianRenderer
from nre.models.nrenderable import NRenderableModel
from nre.nrm.config.models import PrimitiveExportPreprocessConfig
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.types import RigTrajectories


class BaseNRMPrimitive(NRenderableModel):
    """
    Base class for all renderable primitives reconstructed by an NRM.
    """

    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def rigid_transform(self, T_new: torch.Tensor) -> Self: ...

    @abstractmethod
    def preprocess_for_export(
        self,
        context_batch: DataAndRenderingBatch,
        config: PrimitiveExportPreprocessConfig,
        context_rig: RigTrajectories | None = None,
    ) -> Self:
        """
        Filter and preprocess the primitive for export (e.g. density/sky/road masking).
        Called per chunk after forward; when merging is enabled, merge will then apply
        rigid_transform to align chunks. Implementations must not apply rigid_transform.
        context_rig is optional; Celsius may require it when project_to_z_offset is True.
        """

    @abstractmethod
    def __len__(self) -> int: ...


NRMPrimitiveType = TypeVar("NRMPrimitiveType", bound=BaseNRMPrimitive)


class BaseGaussiansNRMPrimitive(BaseNRMPrimitive):
    """
    Predict-only carrier for Gaussian primitives. The renderer is unused but the
    field is retained so downstream type annotations and merge helpers can keep
    propagating it as None.
    """

    gaussians_renderer: BaseGaussianRenderer | None

    def __init__(
        self,
        gaussians_renderer: BaseGaussianRenderer | None,
        checkpointing: Literal["render", "all", "none"] = "render",
        shared_gaussian_parameters: bool = False,
    ):
        self.gaussians_renderer = gaussians_renderer
        self.checkpointing = checkpointing
        self.shared_gaussian_parameters = shared_gaussian_parameters
