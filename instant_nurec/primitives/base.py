# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import abstractmethod
from typing import Self

import torch

from instant_nurec.config_schema.models import PrimitiveExportPreprocessConfig
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.types import RigTrajectories


class BaseInstantNuRecPrimitive:
    """
    Base class for all renderable primitives reconstructed by an InstantNuRec.
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
