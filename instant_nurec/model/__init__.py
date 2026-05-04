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

import logging
import os

from typing import TYPE_CHECKING, Optional

import torch

from instant_nurec import _hf_mock
from instant_nurec.model.system import GaussiansInstantNuRecSystem


if TYPE_CHECKING:
    from instant_nurec.config_schema.instantnurec import InstantNuRecConfig


logger = logging.getLogger(__name__)


class FullModelNotFoundError(RuntimeError):
    """The pretrained ``kelvin_full.pt`` couldn't be resolved through any
    supported channel (HF cache or ``INSTANT_NUREC_FULL_PT``)."""


def _resolve_full_pt_path() -> Optional[str]:
    """Resolve the ``kelvin_full.pt`` path via the HF resolver."""
    try:
        return _hf_mock.get_full_model_path()
    except _hf_mock.HFMockError:
        return None


def make(config: "InstantNuRecConfig") -> GaussiansInstantNuRecSystem:
    """Load the pretrained ``kelvin_full.pt`` and patch in the per-invocation
    config (output dir / merge toggle / ncore paths).

    Single supported artifact: a torch-pickled ``GaussiansInstantNuRecSystem`` saved
    via ``torch.save(system, path)``. Resolved through the HF mock
    (placeholder repo ``nvidia/instant-nurec-kelvin``) or the
    ``INSTANT_NUREC_FULL_PT`` env var.

    Raises ``FullModelNotFoundError`` if no .pt file can be resolved.
    """
    full_pt_path = _resolve_full_pt_path()
    if not full_pt_path or not os.path.exists(full_pt_path):
        raise FullModelNotFoundError(
            "kelvin_full.pt not found. Either set INSTANT_NUREC_FULL_PT to a "
            "local .pt path or wait for the corp-published HF repo "
            f"{_hf_mock.PLACEHOLDER_REPO_ID!r} to provide it."
        )

    logger.info("Loading full system from %s.", full_pt_path)
    loaded = torch.load(full_pt_path, map_location="cpu", weights_only=False)
    assert isinstance(loaded, GaussiansInstantNuRecSystem), (
        f"Expected GaussiansInstantNuRecSystem from {full_pt_path}, got {type(loaded).__name__}"
    )

    # Per-invocation overrides — the .pt only round-trips weight tensors;
    # out_dir / run_id / merge flag / ncore path change every run.
    from instant_nurec.datasets.datamodule import InstantNuRecDataModule  # local to avoid bootstrap loops

    loaded.out_dir = config.out_dir
    loaded.run_id = config.run_id
    loaded.config = config.system
    loaded.predict_config = config.predict
    loaded.export_preprocess = config.model.export_preprocess
    loaded.datamodule = InstantNuRecDataModule(config)
    return loaded
