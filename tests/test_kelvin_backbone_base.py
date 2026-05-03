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

"""Branch-coverage tests for instant_nurec.model.backbone.base.KelvinMultiscaleFeaturesLatent.

The abstract ``KelvinLatent`` base class only contains abstract properties;
the concrete ``KelvinMultiscaleFeaturesLatent`` is the only implementation
in the predict-only standalone.

Pure torch / dataclass — no compiled-lib stubs needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.model.backbone.base import (
    KelvinLatent,
    KelvinMultiscaleFeaturesLatent,
)


def test_kelvin_latent_is_abstract():
    """Direct instantiation should be impossible — KelvinLatent has abstract
    properties batch_size/device/deepest."""
    with pytest.raises(TypeError, match="abstract"):
        KelvinLatent()  # type: ignore[abstract]


def test_kelvin_multiscale_features_latent_batch_size():
    feats = [torch.randn(4, 2, 8, 8, 16), torch.randn(4, 2, 4, 4, 32)]
    latent = KelvinMultiscaleFeaturesLatent(features=feats)
    assert latent.batch_size == 4


def test_kelvin_multiscale_features_latent_device_cpu():
    feats = [torch.zeros(2, 1, 4, 4, 8)]
    latent = KelvinMultiscaleFeaturesLatent(features=feats)
    assert latent.device == torch.device("cpu")


def test_kelvin_multiscale_features_latent_deepest_is_last_feature():
    f0 = torch.zeros(1, 1, 8, 8, 4)
    f1 = torch.ones(1, 1, 4, 4, 8)
    f2 = torch.full((1, 1, 2, 2, 16), 0.5)
    latent = KelvinMultiscaleFeaturesLatent(features=[f0, f1, f2])
    assert latent.deepest is f2


def test_kelvin_multiscale_features_latent_cls_tokens_default_none():
    feats = [torch.zeros(1, 1, 4, 4, 8)]
    latent = KelvinMultiscaleFeaturesLatent(features=feats)
    assert latent.cls_tokens is None


def test_kelvin_multiscale_features_latent_cls_tokens_provided():
    feats = [torch.zeros(2, 3, 4, 4, 8)]
    cls_tokens = [torch.randn(2, 3, 5, 8)]
    latent = KelvinMultiscaleFeaturesLatent(features=feats, cls_tokens=cls_tokens)
    assert latent.cls_tokens is cls_tokens


def test_kelvin_multiscale_features_latent_single_feature():
    """One feature in the list — ``deepest`` returns it; batch_size still works."""
    f = torch.zeros(7, 1, 4, 4, 8)
    latent = KelvinMultiscaleFeaturesLatent(features=[f])
    assert latent.batch_size == 7
    assert latent.deepest is f
