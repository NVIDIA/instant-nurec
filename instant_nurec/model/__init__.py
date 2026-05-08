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


def _validate_sensor_ids(
    *,
    context_camera_ids: list[str],
    supervision_camera_ids: list[str],
    lidar_id: str,
    available_cameras: list[str],
    available_lidars: list[str],
    sequence_path: str,
) -> None:
    """Verify configured camera + lidar ids exist in the sequence.

    ``lidar_id`` may carry semicolon-separated alternatives (matching the
    runtime's ``lidar_id_candidates=lidar_id.split(';')`` behavior); the
    check passes if at least one candidate is present.

    Raises ``ValueError`` with a sequence-path-anchored message that lists
    the ids actually available in the bag, so a typo can be fixed without
    waiting for the dataloader to fail at first ``__getitem__``.
    """
    avail_cameras_sorted = sorted(available_cameras)
    avail_lidars_sorted = sorted(available_lidars)

    for cam in list(context_camera_ids) + list(supervision_camera_ids):
        if cam not in available_cameras:
            raise ValueError(
                f"camera_id {cam!r} not found in sequence {sequence_path}. "
                f"Available cameras: {avail_cameras_sorted}."
            )

    candidates = lidar_id.split(";")
    if not any(c in available_lidars for c in candidates):
        raise ValueError(
            f"lidar_id {lidar_id!r} not found in sequence {sequence_path}. "
            f"Available lidars: {avail_lidars_sorted}."
        )


def _preflight_validate_sensor_ids(config: "InstantNuRecConfig") -> None:
    """Open the first ncorev4 sequence, enumerate its cameras + lidars,
    and run ``_validate_sensor_ids``. Cheap relative to a full inference
    run; runs once in the main process before workers spin up."""
    dataset_cfg = config.dataset.predict
    if dataset_cfg is None or not dataset_cfg.ncore_json_paths:
        return

    from upath import UPath

    from instant_nurec.utils.ncore_utils import (
        create_sequence_loader,
        parse_sequence_meta_file,
    )

    first_path = UPath(dataset_cfg.ncore_json_paths[0])
    _, _, dataset_paths = parse_sequence_meta_file(first_path)

    # Quiet ncore's INFO chatter during the lightweight enumeration.
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    try:
        seq = create_sequence_loader(
            dataset_paths=dataset_paths,
            open_consolidated=dataset_cfg.open_consolidated,
            v4_poses_component_group="default",
            v4_intrinsics_component_group="default",
            v4_masks_component_group="default",
            v4_cuboids_component_group="default",
        )
    finally:
        root_logger.setLevel(prev_level)

    _validate_sensor_ids(
        context_camera_ids=list(dataset_cfg.context_camera_ids),
        supervision_camera_ids=list(dataset_cfg.supervision_camera_ids),
        lidar_id=dataset_cfg.cuboid_tracks_params.lidar_id,
        available_cameras=list(seq.camera_ids),
        available_lidars=list(seq.lidar_ids),
        sequence_path=str(first_path),
    )


def make(config: "InstantNuRecConfig") -> GaussiansInstantNuRecSystem:
    """Load ``kelvin_jit.pt`` and build a ``GaussiansInstantNuRecSystem``.

    Resolution: ``INSTANT_NUREC_FULL_PT`` env var takes priority; otherwise
    the artifact is fetched from Hugging Face.

    The full ``GaussiansInstantNuRecSystem.__init__`` is bypassed via
    ``__new__`` + manual attribute assignment.
    """
    full_pt_path = _resolve_full_pt_path()
    if not full_pt_path or not os.path.exists(full_pt_path):
        raise FullModelNotFoundError(
            f"kelvin_jit.pt not found. Either set INSTANT_NUREC_FULL_PT to a "
            f"local .pt path or ensure {pretrained.KELVIN_REPO_ID!r} is reachable."
        )

    from instant_nurec.datasets.datamodule import InstantNuRecDataModule

    _preflight_validate_sensor_ids(config)

    logger.info("Loading JIT system from %s.", full_pt_path)
    jit_module = torch.jit.load(full_pt_path, map_location="cpu")
    adapter = JITKelvinAdapter(jit_module=jit_module)

    n_context_cams = len(config.dataset.predict.context_camera_ids)
    if adapter.expected_v % n_context_cams != 0:
        raise FullModelNotFoundError(
            f"Model expects {adapter.expected_v} input frames; "
            f"len(context_camera_ids)={n_context_cams} doesn't divide it. "
            f"Update context_camera_ids so its length divides "
            f"{adapter.expected_v}."
        )
    n_frames_per_sample = adapter.expected_v // n_context_cams

    system: GaussiansInstantNuRecSystem = GaussiansInstantNuRecSystem.__new__(
        GaussiansInstantNuRecSystem
    )
    nn.Module.__init__(system)
    system.out_dir = config.out_dir
    system.run_id = config.run_id
    system.config = config.system
    system.predict_config = config.predict
    system.export_preprocess = config.model.export_preprocess
    system.datamodule = InstantNuRecDataModule(
        config,
        frame_width=adapter.expected_w,
        frame_height=adapter.expected_h,
        n_frames_per_sample=n_frames_per_sample,
    )
    system.model = adapter
    return system
