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
import zipfile

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
    """``kelvin_full.pt`` couldn't be resolved (no HF download, no env override)."""


def _resolve_full_pt_path() -> Optional[str]:
    """Return the local path to ``kelvin_full.pt`` or ``None`` on failure."""
    try:
        return pretrained.download_kelvin_full_pt()
    except pretrained.PretrainedModelError:
        return None


def _is_jit_archive(path: str) -> bool:
    """Return True if ``path`` is a TorchScript zip archive (has ``code/``
    entries -- the serialized IR sources) rather than a torch-pickle
    archive (which only has ``data.pkl`` + tensor blobs).

    Both formats nest entries under a top-level directory named after the
    archive's file stem (``kelvin_jit/...`` vs ``kelvin_full/...``), so
    matching ``/code/`` anywhere in the path covers either layout."""
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except (zipfile.BadZipFile, FileNotFoundError):
        return False
    return any("/code/" in n for n in names)


def _make_from_jit(jit_path: str, config: "InstantNuRecConfig") -> GaussiansInstantNuRecSystem:
    """Build a ``GaussiansInstantNuRecSystem``-shaped object whose ``model`` is a
    ``JITKelvinAdapter`` wrapping the loaded TorchScript artifact.

    The full ``GaussiansInstantNuRecSystem.__init__`` would instantiate a
    fresh ``KelvinInstantNuRec`` (encoder + decoder + sky + post_processing)
    purely to throw it away; ``__new__`` + manual attribute assignment
    bypasses that wasted construction.
    """
    from instant_nurec.datasets.datamodule import InstantNuRecDataModule

    jit_module = torch.jit.load(jit_path, map_location="cpu")
    # The JIT artifact preserves KelvinDPTDecoder.cuboids_dims_padding as a
    # buffer on the traced module; the adapter needs it for cuboid-track-based
    # dynamic-mask refinement.
    cuboids_dims_padding = jit_module.static_core.decoder.cuboids_dims_padding
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
    system.model = JITKelvinAdapter(
        jit_module=jit_module,
        scene_rescale=config.model.scene_rescale,
        cuboids_dims_padding=cuboids_dims_padding,
    )
    return system


def _make_from_pickle(
    pickle_path: str, config: "InstantNuRecConfig"
) -> GaussiansInstantNuRecSystem:
    """Legacy path: load the pickled ``GaussiansInstantNuRecSystem`` and patch
    in per-invocation config. Retired in commit 8 once the JIT artifact has
    fully replaced the pickle on the HF side."""
    from instant_nurec.datasets.datamodule import InstantNuRecDataModule

    loaded = torch.load(pickle_path, map_location="cpu", weights_only=False)
    assert isinstance(loaded, GaussiansInstantNuRecSystem), (
        f"Expected GaussiansInstantNuRecSystem from {pickle_path}, got {type(loaded).__name__}"
    )

    loaded.out_dir = config.out_dir
    loaded.run_id = config.run_id
    loaded.config = config.system
    loaded.predict_config = config.predict
    loaded.export_preprocess = config.model.export_preprocess
    loaded.datamodule = InstantNuRecDataModule(config)
    return loaded


def make(config: "InstantNuRecConfig") -> GaussiansInstantNuRecSystem:
    """Resolve the pretrained checkpoint and build a system.

    Auto-detects whether the checkpoint is a TorchScript artifact
    (``kelvin_jit.pt`` from ``internal/scripts/export_kelvin_jit.py``) or a
    pickled ``GaussiansInstantNuRecSystem`` (legacy ``kelvin_full.pt``);
    builds the system accordingly.

    Resolution: ``INSTANT_NUREC_FULL_PT`` env var, then HF download.
    Both env-var override and HF artifact use the same name -- file format
    is auto-detected.
    """
    full_pt_path = _resolve_full_pt_path()
    if not full_pt_path or not os.path.exists(full_pt_path):
        raise FullModelNotFoundError(
            f"kelvin_full.pt not found. Either set INSTANT_NUREC_FULL_PT to a "
            f"local .pt path or ensure {pretrained.KELVIN_REPO_ID!r} is reachable."
        )

    if _is_jit_archive(full_pt_path):
        logger.info("Loading JIT system from %s.", full_pt_path)
        return _make_from_jit(full_pt_path, config)

    logger.info("Loading pickled system from %s.", full_pt_path)
    return _make_from_pickle(full_pt_path, config)
