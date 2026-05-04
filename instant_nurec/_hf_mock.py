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

"""HuggingFace resolver for ``kelvin_full.pt``.

Resolves the pretrained model on first inference run, in this order:

1. If ``INSTANT_NUREC_FULL_PT`` points at an existing file, copy it into
   the cache and use it. Useful for offline use.
2. If a copy already lives in the cache, return it.
3. Otherwise auto-download from ``nvidia/instant-nurec-kelvin`` via
   ``huggingface_hub.hf_hub_download`` into the cache.

The HF repo id is currently a placeholder; until it's published, the
auto-download fails with an actionable error message and step (1) is the
expected escape hatch.
"""

from __future__ import annotations

import logging
import os
import shutil

from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


# The HF repo id; until it's published, auto-download fails and the user
# must point at a local copy via ``INSTANT_NUREC_FULL_PT``.
PLACEHOLDER_REPO_ID = "nvidia/instant-nurec-kelvin"

# Default cache root. Mirrors HF's own convention but namespaced under
# ``instant_nurec`` so we don't collide with anything else on the user's
# machine.
DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.cache/instant_nurec"))

_FULL_MODEL_FILENAME = "kelvin_full.pt"


class HFMockError(RuntimeError):
    """Raised when the resolver cannot satisfy a request."""


def _cache_dir() -> Path:
    return Path(os.environ.get("INSTANT_NUREC_HF_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


def _seed_cache_from_env(cache_root: Path) -> Optional[Path]:
    """If ``INSTANT_NUREC_FULL_PT`` points at an existing file, copy it
    into the cache when the cache is missing/stale and return the cache
    path. Returns ``None`` if the env var isn't set or doesn't resolve.
    """
    src = os.environ.get("INSTANT_NUREC_FULL_PT")
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    cache_root.mkdir(parents=True, exist_ok=True)
    dst = cache_root / _FULL_MODEL_FILENAME
    src_stat = src_path.stat()
    needs_copy = (
        not dst.exists()
        or dst.stat().st_size != src_stat.st_size
        or dst.stat().st_mtime < src_stat.st_mtime
    )
    if needs_copy:
        logger.info("Seeding cache from %s.", src_path)
        shutil.copyfile(src_path, dst)
    return dst


def get_full_model_path(*, cache_dir: Optional[str | Path] = None) -> str:
    """Resolve ``kelvin_full.pt``: env var → cache → HF auto-download."""

    cache_root = Path(cache_dir) if cache_dir else _cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)

    seeded = _seed_cache_from_env(cache_root)
    if seeded is not None:
        return str(seeded)

    cached = cache_root / _FULL_MODEL_FILENAME
    if cached.exists():
        return str(cached)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise HFMockError(
            f"huggingface_hub is required to auto-download "
            f"{PLACEHOLDER_REPO_ID!r}: pip install huggingface_hub"
        ) from e

    try:
        logger.info("Downloading %s/%s ...", PLACEHOLDER_REPO_ID, _FULL_MODEL_FILENAME)
        return hf_hub_download(
            repo_id=PLACEHOLDER_REPO_ID,
            filename=_FULL_MODEL_FILENAME,
            cache_dir=str(cache_root),
        )
    except Exception as e:
        raise HFMockError(
            f"Could not resolve {_FULL_MODEL_FILENAME}. The HF repo "
            f"{PLACEHOLDER_REPO_ID!r} download failed and no local copy was "
            f"found at {cached} or via INSTANT_NUREC_FULL_PT. Underlying "
            f"error: {e}"
        ) from e
