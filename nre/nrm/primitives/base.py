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
from typing import Self, TypeVar

import torch

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
        """

    @abstractmethod
    def __len__(self) -> int: ...


NRMPrimitiveType = TypeVar("NRMPrimitiveType", bound=BaseNRMPrimitive)


class BaseGaussiansNRMPrimitive(BaseNRMPrimitive):
    """Marker base class for Gaussian-primitive NRMs. The NRE-side renderer
    plumbing (gaussians_renderer, checkpointing, shared_gaussian_parameters)
    was removed in Phase 1 step 4.3 -- predict never invokes
    `primitive.forward()` / `render()`, so the renderer was always None and
    the checkpointing/shared-param flags went unread."""
