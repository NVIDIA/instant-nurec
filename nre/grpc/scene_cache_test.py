# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import fcntl
import shutil
import tempfile
import time
import uuid

from pathlib import Path

import pytest

from nre.grpc.scene_cache import SceneCache, SceneCacheValue


@pytest.fixture
def cache_dir():
    """
    Generate a temporary directory path (but not a directory)
    and delete it after the test.

    Note: don't use TemporaryDirectory() because we want to  check if the folder is created by the cache
    """

    temp_path = Path(tempfile.gettempdir()) / f"test-scene-cache-{uuid.uuid4()}"

    yield temp_path

    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def scene_cache(cache_dir: Path):
    assert not cache_dir.exists()
    cache = SceneCache(cache_dir=cache_dir, max_size=10)
    yield cache


@pytest.fixture
def make_temp_file():
    """
    Fixture for a file creation function with a given content.

    Returns:
        Callable[[str], Path]: function that creates a temporary file with the given content
    """

    def _create_temp_file(content: str):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(content.encode("utf-8"))
        temp_file.close()

        return Path(temp_file.name)

    return _create_temp_file


@pytest.fixture
def temp_scene_1(make_temp_file):
    """
    A shortcut fixuture to provide a dummy scene #1.
    """
    temp_file = make_temp_file("x-test-scene-1")

    yield temp_file

    # cleanup
    if Path(temp_file.name).exists():
        Path(temp_file.name).unlink()


@pytest.fixture
def temp_scene_2(make_temp_file):
    """
    A shortcut fixuture to provide a dummy scene #2.
    """
    temp_file = make_temp_file("x-test-scene-2")
    yield temp_file

    # cleanup
    if Path(temp_file.name).exists():
        Path(temp_file.name).unlink()


class TestSceneCache:
    """Test suite for SceneCache class functionality.

    Tests the initialization, scene management, and cache size enforcement
    of the SceneCache implementation.
    """

    def test_create_cache_dir(self, cache_dir: Path):
        """Test SceneCache initialization creates the cache directory correctly."""
        assert not cache_dir.exists()

        cache = SceneCache(cache_dir=cache_dir, max_size=10)

        # check that the cache directory exists
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_new_cache_has_no_scene(self, scene_cache):
        """Test that a newly initialized cache correctly reports no scenes present."""
        assert not scene_cache.has_scene("x-test-scene-1")

    def test_basic_add_scene(self, scene_cache: SceneCache, temp_scene_1: Path):
        """Basic test adding a new scene to the cache.

        Verifies that:
        - Scene is correctly added to cache
        - Original file is removed
        - Scene can be retrieved
        - Content matches original
        """
        expected_path = scene_cache.cache_dir / "x-test-scene-1.usdz"

        assert scene_cache.add_scene("x-test-scene-1", temp_scene_1) == expected_path
        assert not temp_scene_1.exists()  # the original file should be deleted

        assert scene_cache.has_scene("x-test-scene-1")
        assert scene_cache.get_scene_path("x-test-scene-1") == expected_path
        assert expected_path.read_bytes() == b"x-test-scene-1"

        # get a non-existent scene should raise an error
        with pytest.raises(KeyError) as e:
            scene_cache.get_scene_path("x-test-missing-scene")

        assert str(e.value) == "'Scene x-test-missing-scene not in cache'"

    def test_add_scene_with_same_id_raises_error(self, scene_cache: SceneCache, temp_scene_1: Path, temp_scene_2: Path):
        """Test handling of attempts to add duplicate scenes.

        Verifies that:
        - Adding a scene with existing ID raises ValueError
        - Original cached content remains unchanged
        """
        expected_path = scene_cache.cache_dir / "x-test-scene-1.usdz"

        assert scene_cache.add_scene("x-test-scene-1", temp_scene_1) == expected_path

        # Adding another scene with the same scene id should raise an error
        with pytest.raises(ValueError) as e:
            scene_cache.add_scene("x-test-scene-1", temp_scene_2)

        assert str(e.value) == "Scene x-test-scene-1 already in cache"

        # Make sure the original file is not deleted and the content is the same
        assert expected_path.exists()
        assert expected_path.read_bytes() == b"x-test-scene-1"

    def test_get_scene_path_not_found(self, scene_cache: SceneCache):
        """Test appropriate error handling when retrieving non-existent scene from the empty cache."""
        with pytest.raises(KeyError) as e:
            scene_cache.get_scene_path("x-test-scene-1")

        assert str(e.value) == "'Scene x-test-scene-1 not in cache'"

    def test_enforce_max_size(self, scene_cache: SceneCache, make_temp_file):
        """Test cache size limit enforcement.

        Verifies that:
        - Cache maintains maximum size limit
        - Oldest entries are removed when limit is reached
        - New entries can be added after limit is reached while old ones are removed
        """
        max_size = scene_cache.max_size

        for i in range(max_size * 2):
            temp_file = make_temp_file(f"x-test-scene-{i}")
            expected_path = scene_cache.cache_dir / f"x-test-scene-{i}.usdz"
            assert scene_cache.add_scene(f"x-test-scene-{i}", temp_file) == expected_path
            assert expected_path.exists()

            # the cache should have at most max_size files
            assert len(scene_cache._cache) <= max_size

            # List files in the cache directory
            files = list(scene_cache.cache_dir.glob("*.usdz"))
            assert len(files) == min(i + 1, max_size)


class TestSceneCacheValue:
    """Test suite for SceneCacheValue named tuple implementation."""

    def test_scene_cache_value_named_tuple(self):
        """Test SceneCacheValue named tuple creation and attribute access."""
        path = Path("/test/path")
        timestamp = time.time()

        value = SceneCacheValue(path, timestamp)

        assert value.path == path
        assert value.timestamp == timestamp
        assert isinstance(value, tuple)  # namedtuple is still a tuple


class TestSceneCacheLock:
    """Test suite for SceneCache lock functionality."""

    def test_lock_scene_path_keeps_lock_file(self, scene_cache: SceneCache):
        """
        Simple test to check that the lock file is created and locked

        Verifies that:
        - Lock file is created
        - Lock file is locked
        - Lock file is not deleted internally
        - Lock file is unlocked when the context manager is exited
        """

        expected_lock_file = scene_cache.cache_dir / "x-test-scene-1.lock"

        with scene_cache.lock_scene_path("x-test-scene-1"):
            assert expected_lock_file.exists()

            # Try to acquire low-level lock on the same file
            with open(expected_lock_file, "w") as f:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Lock file is NOT deleted internally
        assert expected_lock_file.exists()

        # Try to acquire low-level lock on the same file again
        with open(expected_lock_file, "w") as f:
            # should not raise an error
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_sequential_lock_file(self, scene_cache: SceneCache):
        """Test that the lock file can be acquired sequentially multiple times."""
        for _ in range(10):
            with scene_cache.lock_scene_path("x-test-scene-1"):
                pass
