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

from torch import nn

from instant_nurec import pretrained
from instant_nurec.model.jit_adapter import JITKelvinAdapter
from instant_nurec.model.system import GaussiansInstantNuRecSystem


if TYPE_CHECKING:
    from instant_nurec.config_schema.instantnurec import InstantNuRecConfig


logger = logging.getLogger(__name__)


class FullModelNotFoundError(RuntimeError):
    """``kelvin_jit.pt`` couldn't be resolved (no HF download, no env override)."""


def _resolve_full_pt_path() -> Optional[str]:
    """Return the local path to ``kelvin_jit.pt`` or ``None`` on failure."""
    try:
        return pretrained.download_kelvin_full_pt()
    except pretrained.PretrainedModelError:
        return None


def make(config: "InstantNuRecConfig") -> GaussiansInstantNuRecSystem:
    """Load ``kelvin_jit.pt`` and build a ``GaussiansInstantNuRecSystem``.

    Resolution: ``INSTANT_NUREC_FULL_PT`` env var takes priority; otherwise
    the artifact is fetched from Hugging Face. The artifact is a TorchScript
    archive of ``TraceableStaticCore`` (see
    ``internal/scripts/export_kelvin_jit.py``); the legacy pickled-system
    path was retired once the JIT artifact reached PLY parity against the
    eager baseline.

    The full ``GaussiansInstantNuRecSystem.__init__`` is bypassed via
    ``__new__`` + manual attribute assignment because the system class no
    longer constructs an in-process model -- it just orchestrates a
    ``JITKelvinAdapter`` that wraps the loaded TorchScript module.
    """
    full_pt_path = _resolve_full_pt_path()
    if not full_pt_path or not os.path.exists(full_pt_path):
        raise FullModelNotFoundError(
            f"kelvin_jit.pt not found. Either set INSTANT_NUREC_FULL_PT to a "
            f"local .pt path or ensure {pretrained.KELVIN_REPO_ID!r} is reachable."
        )

    from instant_nurec.datasets.datamodule import InstantNuRecDataModule

    logger.info("Loading JIT system from %s.", full_pt_path)
    jit_module = torch.jit.load(full_pt_path, map_location="cpu")

    system: GaussiansInstantNuRecSystem = GaussiansInstantNuRecSystem.__new__(
        GaussiansInstantNuRecSystem
    )
    nn.Module.__init__(system)
    system.out_dir = config.out_dir
    system.run_id = config.run_id
    system.config = config.system
    system.predict_config = config.predict
    system.export_preprocess = config.model.export_preprocess
    system.datamodule = InstantNuRecDataModule(config)
    # The adapter pulls ``scene_rescale`` and ``cuboids_dims_padding`` off
    # the loaded JIT module's preserved buffers; the public config doesn't
    # carry architecture-side fields anymore.
    system.model = JITKelvinAdapter(jit_module=jit_module)
    return system
