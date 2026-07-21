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

"""Resolve released InstantNuRec weight checkpoints.

``download_instant_nurec_pt()`` downloads the selected release profile
from Hugging Face into the standard hub cache. Setting
``INSTANT_NUREC_FULL_PT`` to an existing local path short-circuits the
download (useful for offline use); the override must match the selected
profile.
"""

from __future__ import annotations

import logging
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, get_args


logger = logging.getLogger(__name__)


MODEL_REPO_ID = "nvidia/instant-nurec"

ModelVariant = Literal["pa-front", "pa-multiview", "pq-front"]
MODEL_VARIANTS: tuple[ModelVariant, ...] = get_args(ModelVariant)
DEFAULT_MODEL_VARIANT: ModelVariant = "pa-front"


@dataclass(frozen=True)
class ModelProfile:
    """Inference contract associated with a released checkpoint."""

    filename: str
    context_camera_ids: tuple[str, ...]
    frame_width: int
    frame_height: int
    n_frames_per_sample: int
    motion_depth: int
    supported_camera_counts: tuple[int, ...]
    decoder_kind: Literal["pixel-aligned", "point-query"]


MODEL_PROFILES: dict[ModelVariant, ModelProfile] = {
    "pa-front": ModelProfile(
        filename="pth/instant_nurec_pa_front_1.1.0.pth",
        context_camera_ids=("camera_front_wide_120fov",),
        frame_width=784,
        frame_height=448,
        n_frames_per_sample=18,
        motion_depth=1,
        supported_camera_counts=(1,),
        decoder_kind="pixel-aligned",
    ),
    "pa-multiview": ModelProfile(
        filename="pth/instant_nurec_pa_multiview_1.1.0.pth",
        context_camera_ids=(
            "camera_front_wide_120fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
        ),
        frame_width=504,
        frame_height=280,
        n_frames_per_sample=18,
        motion_depth=1,
        supported_camera_counts=(1, 3, 5),
        decoder_kind="pixel-aligned",
    ),
    "pq-front": ModelProfile(
        filename="pth/instant_nurec_pq_road_1.0.0.pth",
        context_camera_ids=("camera_front_wide_120fov",),
        frame_width=784,
        frame_height=448,
        n_frames_per_sample=18,
        motion_depth=1,
        supported_camera_counts=(1,),
        decoder_kind="point-query",
    ),
}

# Backward-compatible alias for callers that imported the original single
# checkpoint constant. PA-front remains the default release profile.
MODEL_FILENAME = MODEL_PROFILES[DEFAULT_MODEL_VARIANT].filename


def get_model_profile(model_variant: ModelVariant | str) -> ModelProfile:
    """Return the release profile for ``model_variant``."""
    try:
        return MODEL_PROFILES[model_variant]  # type: ignore[index]
    except KeyError as e:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown model variant {model_variant!r}; choose one of: {choices}") from e


class PretrainedModelError(RuntimeError):
    """Raised when an Instant NuRec checkpoint cannot be resolved."""


def download_instant_nurec_pt(
    model_variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    *,
    cache_dir: Optional[str | Path] = None,
) -> str:
    """Return the local path to the selected release checkpoint.

    Resolution order:

    1. ``INSTANT_NUREC_FULL_PT`` env var, if set and pointing at an
       existing file — used verbatim (offline override).
    2. ``huggingface_hub.hf_hub_download`` from :data:`MODEL_REPO_ID`.
       When ``cache_dir`` is passed it is forwarded verbatim; otherwise
       ``huggingface_hub`` uses its standard hub cache.
    """

    profile = get_model_profile(model_variant)

    env_path = os.environ.get("INSTANT_NUREC_FULL_PT")
    if env_path and Path(env_path).exists():
        return env_path

    if cache_dir is not None:
        target: Optional[Path] = Path(cache_dir)
        target.mkdir(parents=True, exist_ok=True)
    else:
        target = None

    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError as e:
        raise PretrainedModelError(
            "huggingface_hub is required to download InstantNuRec: "
            "pip install huggingface_hub"
        ) from e

    hf_kwargs: dict = {"repo_id": MODEL_REPO_ID, "filename": profile.filename}
    if target is not None:
        hf_kwargs["cache_dir"] = str(target)

    # If a cached copy exists, use it as-is -- no network roundtrip.
    cached = try_to_load_from_cache(**hf_kwargs)
    if isinstance(cached, str):
        return cached

    try:
        logger.info("Downloading %s/%s ...", MODEL_REPO_ID, profile.filename)
        return hf_hub_download(**hf_kwargs)
    except Exception as e:
        raise PretrainedModelError(
            f"Could not download {MODEL_REPO_ID}/{profile.filename}. "
            f"Set INSTANT_NUREC_FULL_PT to a local copy of "
            f"{profile.filename} as a fallback. Underlying error: {e}"
        ) from e
