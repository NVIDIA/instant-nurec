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

from instant_nurec import pretrained
from instant_nurec.model.inference import KelvinInferenceModel
from instant_nurec.model.static_core import KelvinStaticCore
from instant_nurec.model.system import GaussiansInstantNuRecSystem


if TYPE_CHECKING:
    from instant_nurec.config_schema.instantnurec import InstantNuRecConfig


logger = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    """The pretrained weight checkpoint could not be resolved."""


class ModelCheckpointError(RuntimeError):
    """The resolved file is not a source-model state dictionary."""


def _resolve_model_pt_path() -> Optional[str]:
    """Return the local path to the weight checkpoint or ``None`` on failure."""
    try:
        return pretrained.download_instant_nurec_pt()
    except pretrained.PretrainedModelError:
        return None


def _validate_camera_ids(
    *,
    context_camera_ids: list[str],
    supervision_camera_ids: list[str],
    available_cameras: list[str],
    sequence_path: str,
) -> None:
    """Verify configured camera ids exist in the sequence.

    Raises ``ValueError`` with a sequence-path-anchored message that lists
    the ids actually available in the bag, so a typo can be fixed without
    waiting for the dataloader to fail at first ``__getitem__``.
    """
    avail_cameras_sorted = sorted(available_cameras)

    for cam in list(context_camera_ids) + list(supervision_camera_ids):
        if cam not in available_cameras:
            raise ValueError(
                f"camera_id {cam!r} not found in sequence {sequence_path}. "
                f"Available cameras: {avail_cameras_sorted}."
            )


def _preflight_validate_camera_ids(config: "InstantNuRecConfig") -> None:
    """Open the first ncorev4 sequence, enumerate its cameras,
    and run ``_validate_camera_ids``. Cheap relative to a full inference
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

    _validate_camera_ids(
        context_camera_ids=list(dataset_cfg.context_camera_ids),
        supervision_camera_ids=list(dataset_cfg.supervision_camera_ids),
        available_cameras=list(seq.camera_ids),
        sequence_path=str(first_path),
    )


def _load_model_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a weights-only checkpoint and reject legacy traced archives."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if any(name.endswith("/constants.pkl") for name in archive.namelist()):
                raise ModelCheckpointError(
                    f"{path} is a legacy traced-model archive. Download "
                    f"{pretrained.MODEL_FILENAME} or point INSTANT_NUREC_FULL_PT "
                    "at the released weights-only checkpoint."
                )

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ModelCheckpointError(
            f"{path} must contain a plain state dictionary of string keys and tensors."
        )
    return state_dict


def make(config: "InstantNuRecConfig") -> GaussiansInstantNuRecSystem:
    """Build the eager source model and load its pretrained weights.

    Resolution: ``INSTANT_NUREC_FULL_PT`` env var takes priority; otherwise
    the artifact is fetched from Hugging Face.
    """
    full_pt_path = _resolve_model_pt_path()
    if not full_pt_path or not os.path.exists(full_pt_path):
        raise ModelNotFoundError(
            f"{pretrained.MODEL_FILENAME} not found. Either set INSTANT_NUREC_FULL_PT to a "
            f"local .pt path or ensure {pretrained.MODEL_REPO_ID!r} is reachable."
        )

    _preflight_validate_camera_ids(config)
    logger.info("Loading source-model weights from %s.", full_pt_path)
    static_core = KelvinStaticCore(config.model)
    static_core.load_state_dict(_load_model_state_dict(full_pt_path), strict=True)

    dataset_config = config.dataset.predict
    assert dataset_config is not None, "dataset.predict must be configured for inference"
    model = KelvinInferenceModel(
        static_core,
        scene_rescale=config.model.scene_rescale,
        expected_frames=(
            len(dataset_config.context_camera_ids)
            * dataset_config.frame_batch_sampler.n_frames_per_sample
        ),
        expected_height=dataset_config.camera_subsampler.frame_height,
        expected_width=dataset_config.camera_subsampler.frame_width,
    )
    return GaussiansInstantNuRecSystem(config, model)
