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

"""Resolve the pretrained InstantNuRec model artifact.

``download_instant_nurec_pt()`` returns the local path to
``instant_nurec.pt``, downloading from Hugging Face on first use into
the standard HF hub cache. Setting ``INSTANT_NUREC_FULL_PT`` to an
existing local path short-circuits the download (useful for offline
use).
"""

from __future__ import annotations

import logging
import os

from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


MODEL_REPO_ID = "nvidia/instant-nurec"
MODEL_FILENAME = "instant_nurec.pt"


class PretrainedModelError(RuntimeError):
    """Raised when ``instant_nurec.pt`` cannot be resolved."""


def download_instant_nurec_pt(*, cache_dir: Optional[str | Path] = None) -> str:
    """Return the local path to ``instant_nurec.pt``.

    Resolution order:

    1. ``INSTANT_NUREC_FULL_PT`` env var, if set and pointing at an
       existing file — used verbatim (offline override).
    2. ``huggingface_hub.hf_hub_download`` from :data:`MODEL_REPO_ID`.
       When ``cache_dir`` is passed it is forwarded verbatim; otherwise
       ``huggingface_hub`` uses its standard hub cache.
    """

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

    hf_kwargs: dict = {"repo_id": MODEL_REPO_ID, "filename": MODEL_FILENAME}
    if target is not None:
        hf_kwargs["cache_dir"] = str(target)

    # If a cached copy exists, use it as-is -- no network roundtrip.
    cached = try_to_load_from_cache(**hf_kwargs)
    if isinstance(cached, str):
        return cached

    try:
        logger.info("Downloading %s/%s ...", MODEL_REPO_ID, MODEL_FILENAME)
        return hf_hub_download(**hf_kwargs)
    except Exception as e:
        raise PretrainedModelError(
            f"Could not download {MODEL_REPO_ID}/{MODEL_FILENAME}. "
            f"Set INSTANT_NUREC_FULL_PT to a local copy of "
            f"{MODEL_FILENAME} as a fallback. Underlying error: {e}"
        ) from e
