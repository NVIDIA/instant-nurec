# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import shutil
import threading
import time

from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple


log = logging.getLogger(__name__)


class SceneCacheValue(NamedTuple):
    """Cache value containing scene path and timestamp."""

    path: Path
    """Path to the cached scene."""

    timestamp: float
    """Timestamp of the scene."""


class SceneCache:
    """Manages an on-disk LRU cache of downloaded scenes."""

    def __init__(self, cache_dir: Path, max_size: int):
        """
        Initialize the scene cache.

        Args:
            cache_dir: Path to the cache directory
            max_size: Maximum number of scenes to keep in the cache
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self._cache: OrderedDict[str, SceneCacheValue] = OrderedDict()
        self._lock = threading.Lock()

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Scene cache initialized in {self.cache_dir} with max size {self.max_size}")

    def _enforce_max_size(self):
        """Remove oldest scenes if cache exceeds max size."""
        while len(self._cache) > self.max_size:
            scene_id, (path, _) = self._cache.popitem(last=False)  # Remove oldest
            try:
                os.remove(path)
                log.info(f"Removed {scene_id} from cache to enforce max size")
            except OSError as e:
                log.error(f"Failed to remove {path}: {e}")

    def has_scene(self, scene_id: str) -> bool:
        """Check if a scene is in the cache."""
        return scene_id in self._cache

    def get_scene_path(self, scene_id: str, update_access_time: bool = True) -> Path:
        """Get the path to a cached scene with proper locking."""
        with self._lock:
            if scene_id not in self._cache:
                raise KeyError(f"Scene {scene_id} not in cache")

            path, _ = self._cache[scene_id]
            if update_access_time:
                self._cache[scene_id] = SceneCacheValue(path, time.time())
            return path

    def add_scene(self, scene_id: str, temp_path: Path) -> Path:
        """
        Add a scene to the cache with proper locking.
        Adding a new scene will raise an error if the scene is already in the cache.
        """
        # Check again under lock if the scene was added while we were downloading
        with self._lock:
            if self.has_scene(scene_id):
                raise ValueError(f"Scene {scene_id} already in cache")

            # Generate destination path
            dest_path = self.cache_dir / f"{scene_id}.usdz"

            # Move the file to the cache directory
            shutil.move(temp_path, dest_path)

            # Update LRU with current timestamp
            now = time.time()
            self._cache[scene_id] = SceneCacheValue(dest_path, now)

            # Enforce max size
            self._enforce_max_size()

            return dest_path

    @contextlib.contextmanager
    def lock_scene_path(self, scene_id: str):
        """
        Context manager to lock the downloading of a scene_id to avoid concurrent downloads.

        This is a simplified version of the filelock library.

        - Open file for writing
        - Lock the file
        - Write the pid of the current process to the file
        - Yield the lock file to do the download under the lock
        - Unlock the file
        - Close the file

        Do not remove the lockfile:
        * https://github.com/tox-dev/py-filelock/issues/31
        * https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
        """

        assert self.cache_dir is not None
        assert self.cache_dir.exists()

        lock_file = self.cache_dir / f"{scene_id}.lock"

        with open(lock_file, "w") as f:
            log.info(f"Locking scene {scene_id} for download")

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(f"{os.getpid()}\r\n")
            yield lock_file
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            log.info(f"Unlocked scene {scene_id} for download")
