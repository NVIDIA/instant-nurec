# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3 stub: real implementation lives in NRE training; this
# standalone predict pipeline does not run any losses, so the surface is
# reduced to what the few remaining `from libs.losses.orchestration.config
# import ...` statements need to resolve. Pydantic LossConfig accepts the raw
# dict from parsed.yaml; the rest are empty dataclasses used only as type
# annotations.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TypeVar

from pydantic import RootModel


class LossItemConfig(RootModel[Dict[str, Any]]):
    """Permissive: accepts any dict for a single loss item from parsed.yaml."""


class LossConfig(RootModel[Dict[str, LossItemConfig]]):
    """Permissive: accepts the entire `loss:` block from parsed.yaml."""

    def __getitem__(self, key: str) -> LossItemConfig:
        return self.root[key]

    def items(self):
        return self.root.items()

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def get(self, key: str, default=None):
        return self.root.get(key, default)


@dataclass
class LossReturn:
    """Stub: never instantiated in predict mode."""

    name: str = ""


@dataclass
class LossAggregatorReturn:
    loss_returns: Dict[str, LossReturn] = field(default_factory=dict)


@dataclass
class LossAggregatorBatchReturn:
    batch_loss_returns: List[LossAggregatorReturn] = field(default_factory=list)
    extra_fields: Dict[str, Any] = field(default_factory=dict)


LossReturnType = TypeVar("LossReturnType", LossAggregatorReturn, LossAggregatorBatchReturn)
