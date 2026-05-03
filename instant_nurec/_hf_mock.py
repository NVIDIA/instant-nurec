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

"""HuggingFace placeholder mock.

The standalone targets the (eventually-public) HF repo
``nvidia/instant-nurec-kelvin`` for both the pickled Kelvin system
(``kelvin_full.pt``) and an ncorev4 sample fixture (``ncorev4_sample/``).
Until the corp publishes that repo we keep the production import surface
identical to the real-HF path and stub the resolver in this module:

* ``snapshot_download(repo_id, ...)`` returns a local cache directory
  populated with whatever the user has already produced (Phase 1 step 5
  writes ``kelvin_full.pt`` via the ``INSTANT_NUREC_FULL_PT`` env var).
* ``hf_hub_download(repo_id, filename, ...)`` returns the absolute path
  to a single cached artifact.

The mock is selected via the env var ``INSTANT_NUREC_HF_MOCK`` (default
``1``). Setting it to ``0`` forwards the call through to
``huggingface_hub`` proper if it's installed.

Self-invented: NRE has no HF equivalent (it pulls from NGC via
``nre.utils.model_registry``).
"""

from __future__ import annotations

import os
import shutil

from pathlib import Path
from typing import Optional


# The placeholder HF repo id; the corp will replace this when the real
# upload happens. Until then the mock resolves it locally.
PLACEHOLDER_REPO_ID = "nvidia/instant-nurec-kelvin"

# Default cache root. Mirrors HF's own convention but namespaced under
# ``instant_nurec`` so we don't collide with anything else on the user's
# machine.
DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.cache/instant_nurec"))

# Names the standalone treats as ``downloadable`` from the placeholder repo.
_FULL_MODEL_FILENAME = "kelvin_full.pt"
_SAMPLE_DIR_NAME = "ncorev4_sample"


class HFMockError(RuntimeError):
    """Raised when the mock cannot satisfy a request (file missing, etc.)."""


def _is_mock_enabled() -> bool:
    """Default on; set ``INSTANT_NUREC_HF_MOCK=0`` to defer to real HF."""
    return os.environ.get("INSTANT_NUREC_HF_MOCK", "1") != "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("INSTANT_NUREC_HF_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


def _seed_cache_from_full_pt(cache_dir: Path) -> Path:
    """If ``INSTANT_NUREC_FULL_PT`` points at an existing file and the cache
    doesn't already have ``kelvin_full.pt``, copy it in. Returns the cache dir.

    This bridges Phase 1 step 5 (the user's existing pickled system) to the
    Phase 4 HF mock without requiring them to manually move files around.
    """
    src = os.environ.get("INSTANT_NUREC_FULL_PT")
    if not src:
        return cache_dir
    src_path = Path(src)
    if not src_path.exists():
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / _FULL_MODEL_FILENAME
    if not dst.exists() or dst.stat().st_size != src_path.stat().st_size:
        shutil.copyfile(src_path, dst)
    return cache_dir


def snapshot_download(
    repo_id: str,
    *,
    cache_dir: Optional[str | Path] = None,
    revision: Optional[str] = None,  # noqa: ARG001  (unused; kept for API compat)
    **kwargs,  # noqa: ARG001
) -> str:
    """Resolve ``repo_id`` to a local snapshot directory.

    The mock only knows how to satisfy ``PLACEHOLDER_REPO_ID``; any other
    repo id raises so the caller can fall through to real HF.
    """
    if not _is_mock_enabled():
        from huggingface_hub import snapshot_download as _real

        return _real(repo_id, cache_dir=cache_dir, revision=revision, **kwargs)

    if repo_id != PLACEHOLDER_REPO_ID:
        raise HFMockError(
            f"HF mock only knows {PLACEHOLDER_REPO_ID!r}; got {repo_id!r}. "
            f"Set INSTANT_NUREC_HF_MOCK=0 to defer to real huggingface_hub."
        )

    target = Path(cache_dir) if cache_dir else _cache_dir()
    target = _seed_cache_from_full_pt(target)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    cache_dir: Optional[str | Path] = None,
    revision: Optional[str] = None,  # noqa: ARG001
    **kwargs,  # noqa: ARG001
) -> str:
    """Resolve ``repo_id``/``filename`` to a local file path.

    Raises ``HFMockError`` if the file isn't present in the cache.
    """
    if not _is_mock_enabled():
        from huggingface_hub import hf_hub_download as _real

        return _real(repo_id, filename, cache_dir=cache_dir, revision=revision, **kwargs)

    snapshot_root = Path(snapshot_download(repo_id, cache_dir=cache_dir))
    file_path = snapshot_root / filename
    if not file_path.exists():
        raise HFMockError(
            f"HF mock: file {filename!r} not found in {snapshot_root}. "
            f"Phase 1 step 5 produces ``{_FULL_MODEL_FILENAME}`` via the "
            f"``INSTANT_NUREC_FULL_PT`` env var; either run with that var set "
            f"first or copy the file into the cache directory manually."
        )
    return str(file_path)


def get_full_model_path(*, cache_dir: Optional[str | Path] = None) -> str:
    """Convenience: resolve the canonical ``kelvin_full.pt`` from the
    placeholder repo. Equivalent to
    ``hf_hub_download(PLACEHOLDER_REPO_ID, "kelvin_full.pt")``."""
    return hf_hub_download(PLACEHOLDER_REPO_ID, _FULL_MODEL_FILENAME, cache_dir=cache_dir)


def get_sample_data_path(*, cache_dir: Optional[str | Path] = None) -> str:
    """Convenience: resolve the canonical ``ncorev4_sample/`` directory.

    Returns the directory path. Raises ``HFMockError`` if the directory
    isn't present (Phase 4 step 10 expects the sample fixture to be
    distributed via HF; until then users supply their own ``--ncore-path``).
    """
    snapshot_root = Path(snapshot_download(PLACEHOLDER_REPO_ID, cache_dir=cache_dir))
    sample_dir = snapshot_root / _SAMPLE_DIR_NAME
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise HFMockError(
            f"HF mock: sample data {_SAMPLE_DIR_NAME!r} not found at {sample_dir}. "
            f"Use --ncore-path to point at your own ncorev4 dataset."
        )
    return str(sample_dir)
