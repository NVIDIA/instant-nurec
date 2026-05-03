# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(kw_only=True, slots=True)
class KelvinLatent(ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """
        Get the batch size of the latent.
        """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get the device of the latent.
        """

    @property
    @abstractmethod
    def deepest(self) -> torch.Tensor:
        """
        Get the deepest feature of the latent.
        Note this is not necessarily normalized via layer norm.
        Size will be (B, V, h, w, C)
        """


@dataclass(kw_only=True, slots=True)
class KelvinMultiscaleFeaturesLatent(KelvinLatent):
    """
    Features means transformed queries (i.e. output of the attention block).
    """

    # (B, V, h, w, C)
    features: list[torch.Tensor]

    # (B, V, n_cls_tokens, C)
    cls_tokens: list[torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        return self.features[0].shape[0]

    @property
    def device(self) -> torch.device:
        return self.features[0].device

    @property
    def deepest(self) -> torch.Tensor:
        return self.features[-1]
