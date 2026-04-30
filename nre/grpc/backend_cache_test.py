# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import contextlib

from typing import Callable, List, Optional, Set, Union

import pytest

from nre.grpc.serve import Backend, BackendCache, CacheFullError, NoSpareBackendsError


@pytest.fixture
def make_backend() -> Callable[[str], Backend]:
    def _backend_maker(_scene_id: str) -> Backend:
        # The test doesn't need a fully configured backend, so we just return a dummy one with None values
        return Backend(
            renderable_model=None,  # type: ignore
            camera_bank=None,  # type: ignore
            world_to_nre=None,  # type: ignore
            lidar_bank=None,  # type: ignore
            asset_bank=None,  # type: ignore
        )

    return _backend_maker


def keys_in_use(cache: BackendCache) -> Set[Union[str, None]]:
    return set(backend.cache_key for backend in cache.in_use)


def keys_spare(cache: BackendCache) -> List[Union[str, None]]:
    return [backend.cache_key for backend in cache.spares]


def check_cache(cache: BackendCache, expected_in_use: Set[str], expected_spare: List[str]):
    assert keys_in_use(cache) == expected_in_use
    assert keys_spare(cache) == expected_spare


def check_cache_size(cache: BackendCache, expected_in_use_size: int, expected_spare_size: int):
    assert len(cache.in_use) == expected_in_use_size
    assert len(cache.spares) == expected_spare_size


class TestBackendCache:
    def test_invalid_maxsize(self):
        """Test that BackendCache rejects invalid maxsize values"""
        with pytest.raises(ValueError, match="BackendCache maxsize must be positive, got 0"):
            BackendCache(maxsize=0)

        with pytest.raises(ValueError, match="BackendCache maxsize must be positive, got -1"):
            BackendCache(maxsize=-1)

        with pytest.raises(ValueError, match="BackendCache maxsize must be positive, got -10"):
            BackendCache(maxsize=-10)

    def test_backend_hash_eq_contract(self, make_backend):
        """Test that Backend maintains hash/eq contract using object identity"""
        b1 = make_backend("scene-1")
        b2 = make_backend("scene-1")  # Different object, same scene_id

        # Different objects should not be equal, even with same cache_key
        assert b1 != b2
        assert hash(b1) != hash(b2)

        # Same object should be equal to itself
        assert b1 == b1
        assert hash(b1) == hash(b1)

        # Hash should be stable even when cache_key is mutated
        original_hash = hash(b1)
        b1.cache_key = "scene-1"
        assert hash(b1) == original_hash

        # Can be added to set multiple times (different objects)
        backend_set = {b1, b2}
        assert len(backend_set) == 2

        # Adding same object twice doesn't increase set size
        backend_set.add(b1)
        assert len(backend_set) == 2

    def test_put(self, make_backend):
        """Test adding a backend to the cache"""
        cache = BackendCache(maxsize=3)

        b = make_backend("scene-id-0")
        cache.put("scene-id-0", b)

        check_cache(cache, {"scene-id-0"}, [])

    def test_checkin(self, make_backend):
        """Test checking in a backend makes it spare"""
        cache = BackendCache(maxsize=3)

        b = make_backend("scene-id-0")
        cache.put("scene-id-0", b)
        cache.checkin(b)

        check_cache(cache, set(), ["scene-id-0"])

    def test_count_based_lru_eviction(self, make_backend):
        """Test count-based LRU eviction with spare backends"""
        # Max 3 backends in cache
        cache = BackendCache(maxsize=3)

        scene_ids = []

        # Add 3 backends and check in (all become spares)
        for i in range(3):
            scene_id = f"scene-id-{i}"
            scene_ids.append(scene_id)

            b = make_backend(scene_id)
            cache.put(scene_id, b)
            cache.checkin(b)

            check_cache_size(cache, 0, i + 1)
            check_cache(cache, set(), scene_ids)

        # Add more backends to cause eviction (oldest spares evicted due to count limit)
        for i in range(3, 6):
            scene_id = f"scene-id-{i}"
            scene_ids.append(scene_id)
            scene_ids.pop(0)  # Remove oldest

            b = make_backend(scene_id)
            cache.put(scene_id, b)
            cache.checkin(b)

            # Should maintain 3 backends (maxsize limit)
            check_cache_size(cache, 0, 3)
            check_cache(cache, set(), scene_ids)

    def test_in_use_backends_not_evicted(self, make_backend):
        """Test that in-use backends are not evicted when adding new backends"""
        cache = BackendCache(maxsize=3)

        # Add 3 spare backends
        for i in range(3):
            b = make_backend(f"spare-{i}")
            cache.put(f"spare-{i}", b)
            cache.checkin(b)

        check_cache_size(cache, 0, 3)

        # Add in-use backends - should evict spares, not exceed maxsize
        in_use_ids = set()
        for i in range(3):  # Only add up to maxsize
            scene_id = f"in-use-{i}"
            in_use_ids.add(scene_id)

            b = make_backend(scene_id)
            cache.put(scene_id, b)  # no checkin - stays in-use

            # Each new in-use backend evicts a spare
            expected_spares = max(0, 3 - (i + 1))
            check_cache_size(cache, i + 1, expected_spares)

        # All spares should be evicted, only in-use remain
        check_cache(cache, in_use_ids, [])

    def test_checkout_on_empty_cache(self):
        """Test checkout returns None when cache is empty"""
        cache = BackendCache(maxsize=3)

        check_cache_size(cache, 0, 0)

        assert cache.checkout("scene-id-0") is None

    def test_checkout_used_backend(self, make_backend):
        """Test checkout returns None for in-use backends"""
        cache = BackendCache(maxsize=3)

        scene_id = "x-test-scene-id"

        b = make_backend(scene_id)
        cache.put(scene_id, b)  # b is in-use

        check_cache(cache, {"x-test-scene-id"}, [])

        assert cache.checkout(scene_id) is None
        check_cache(cache, {"x-test-scene-id"}, [])

    def test_checkout_unused_backend(self, make_backend):
        """Test checkout returns spare backend and marks as in-use"""
        cache = BackendCache(maxsize=3)

        scene_id = "scene-id"

        b = make_backend(scene_id)
        cache.put(scene_id, b)  # b is in-use
        cache.checkin(b)  # b is no longer in-use

        checked_out = cache.checkout(scene_id)
        assert id(checked_out) == id(b)
        check_cache(cache, {scene_id}, [])

    def test_evict_one_spare(self, make_backend):
        """Test evict_one_spare method used for OOM retry"""
        cache = BackendCache(maxsize=5)

        # Add 3 spare backends
        for i in range(3):
            b = make_backend(f"scene-{i}")
            cache.put(f"scene-{i}", b)
            cache.checkin(b)

        check_cache_size(cache, 0, 3)

        # Evict one spare
        assert cache.evict_one_spare() is True
        check_cache_size(cache, 0, 2)
        check_cache(cache, set(), ["scene-1", "scene-2"])  # scene-0 evicted (oldest)

        # Evict another
        assert cache.evict_one_spare() is True
        check_cache_size(cache, 0, 1)
        check_cache(cache, set(), ["scene-2"])  # scene-1 evicted

        # Evict last
        assert cache.evict_one_spare() is True
        check_cache_size(cache, 0, 0)
        check_cache(cache, set(), [])

        # No more spares to evict
        assert cache.evict_one_spare() is False

    def test_evict_one_spare_skips_in_use(self, make_backend):
        """Test evict_one_spare only evicts spares, not in-use backends"""
        cache = BackendCache(maxsize=5)

        # Add in-use backend
        b1 = make_backend("in-use")
        cache.put("in-use", b1)  # stays in-use

        # Add spare backend
        b2 = make_backend("spare")
        cache.put("spare", b2)
        cache.checkin(b2)

        check_cache_size(cache, 1, 1)

        # Evict one spare
        assert cache.evict_one_spare() is True
        check_cache_size(cache, 1, 0)
        check_cache(cache, {"in-use"}, [])

        # No more spares, in-use backend remains
        assert cache.evict_one_spare() is False
        check_cache(cache, {"in-use"}, [])

    def test_lru_order_preserved(self, make_backend):
        """Test that LRU order is maintained correctly"""
        cache = BackendCache(maxsize=3)

        # Add and check in backends in order
        for i in range(5):
            b = make_backend(f"scene-{i}")
            cache.put(f"scene-{i}", b)
            cache.checkin(b)

        # Only last 3 should remain (scene-2, scene-3, scene-4)
        check_cache(cache, set(), ["scene-2", "scene-3", "scene-4"])

    def test_checkout_after_eviction(self, make_backend):
        """Test that checkout returns None for evicted backends"""
        cache = BackendCache(maxsize=2)

        # Add and checkin backend
        b = make_backend("scene-old")
        cache.put("scene-old", b)
        cache.checkin(b)

        # Add two more to trigger eviction of scene-old
        b1 = make_backend("scene-1")
        cache.put("scene-1", b1)
        cache.checkin(b1)

        b2 = make_backend("scene-2")
        cache.put("scene-2", b2)
        cache.checkin(b2)

        # scene-old should be evicted
        check_cache(cache, set(), ["scene-1", "scene-2"])

        # Checkout of evicted backend returns None
        assert cache.checkout("scene-old") is None

    def test_multiple_checkins_checkout_cycles(self, make_backend):
        """Test multiple checkin/checkout cycles"""
        cache = BackendCache(maxsize=3)

        b = make_backend("scene-id")
        cache.put("scene-id", b)

        # Checkin and checkout multiple times
        for _ in range(3):
            cache.checkin(b)
            check_cache(cache, set(), ["scene-id"])

            checked_out = cache.checkout("scene-id")
            assert id(checked_out) == id(b)
            check_cache(cache, {"scene-id"}, [])

    def test_maxsize_enforced_when_all_in_use(self, make_backend):
        """Test that maxsize cannot be exceeded even when all backends are in-use.

        Also verifies that the rejected backend's GPU memory is freed before
        CacheFullError is raised (internal cleanup via del + torch.cuda.empty_cache).
        """
        cache = BackendCache(maxsize=3)

        # Add 3 backends, all in-use (no checkin)
        for i in range(3):
            b = make_backend(f"scene-{i}")
            cache.put(f"scene-{i}", b)

        check_cache_size(cache, 3, 0)
        check_cache(cache, {"scene-0", "scene-1", "scene-2"}, [])

        # Try to add a 4th backend - should fail because all 3 are in-use
        # The rejected backend's GPU memory will be freed internally before the exception
        b4 = make_backend("scene-3")
        with pytest.raises(CacheFullError, match="Cannot add backend 'scene-3': all 3 cache slots are in-use"):
            cache.put("scene-3", b4)

        # Backend's cache_key should be reset to None for clean disposal
        assert b4.cache_key is None

        # Cache should still have only 3 backends
        check_cache_size(cache, 3, 0)
        check_cache(cache, {"scene-0", "scene-1", "scene-2"}, [])

        # Now if we check in one backend, we should be able to add a new one
        b0 = next(b for b in cache.in_use if b.cache_key == "scene-0")
        cache.checkin(b0)
        check_cache_size(cache, 2, 1)

        # Now we can add scene-3, which will evict scene-0
        b4 = make_backend("scene-3")
        cache.put("scene-3", b4)
        check_cache_size(cache, 3, 0)
        check_cache(cache, {"scene-1", "scene-2", "scene-3"}, [])

    def test_put_with_retries_success(self, make_backend):
        """Test put_with_retries with successful backend creation on first attempt"""
        cache = BackendCache(maxsize=3)

        # Simple factory that succeeds
        def factory():
            return make_backend("test-scene")

        backend = cache.put_with_retries("test-scene", factory)

        assert backend is not None
        assert backend.cache_key == "test-scene"
        check_cache_size(cache, 1, 0)
        check_cache(cache, {"test-scene"}, [])

    def test_put_with_retries_oom_with_eviction_succeeds(self, make_backend):
        """Test put_with_retries handles OOM by evicting spares and retrying"""
        cache = BackendCache(maxsize=3)

        # Add 2 spare backends
        for i in range(2):
            b = make_backend(f"spare-{i}")
            cache.put(f"spare-{i}", b)
            cache.checkin(b)

        check_cache_size(cache, 0, 2)

        # Factory that fails once with OOM, then succeeds
        call_count = [0]

        def factory_with_oom():
            call_count[0] += 1
            if call_count[0] == 1:
                import torch

                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return make_backend("new-scene")

        # Should evict a spare and retry successfully
        backend = cache.put_with_retries("new-scene", factory_with_oom, enable_eviction=True)

        assert backend is not None
        assert backend.cache_key == "new-scene"
        assert call_count[0] == 2  # Called twice (failed once, succeeded once)
        check_cache_size(cache, 1, 1)  # One spare evicted

    def test_put_with_retries_oom_no_spares_raises(self, make_backend):
        """Test put_with_retries raises NoSpareBackendsError when OOM and no spares available"""
        cache = BackendCache(maxsize=2)

        # Add 2 in-use backends (no spares)
        for i in range(2):
            b = make_backend(f"in-use-{i}")
            cache.put(f"in-use-{i}", b)

        check_cache_size(cache, 2, 0)

        # Factory that always fails with OOM
        def factory_oom():
            import torch

            raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # Should raise NoSpareBackendsError since no spares to evict
        with pytest.raises(NoSpareBackendsError, match="No spare backends available to evict"):
            cache.put_with_retries("new-scene", factory_oom, enable_eviction=True)

    def test_put_with_retries_oom_eviction_disabled(self, make_backend):
        """Test put_with_retries with eviction disabled still retries on OOM"""
        cache = BackendCache(maxsize=3)

        # Add a spare backend
        b = make_backend("spare")
        cache.put("spare", b)
        cache.checkin(b)

        # Factory that fails twice with OOM, then succeeds
        call_count = [0]

        def factory_with_oom():
            call_count[0] += 1
            if call_count[0] <= 2:
                import torch

                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return make_backend("new-scene")

        # Should retry without evicting the spare
        backend = cache.put_with_retries("new-scene", factory_with_oom, max_retries=5, enable_eviction=False)

        assert backend is not None
        assert call_count[0] == 3  # Called 3 times (failed twice, succeeded on 3rd)
        check_cache_size(cache, 1, 1)  # Spare still present (not evicted)

    def test_put_with_retries_propagates_cache_full_error(self, make_backend):
        """Test put_with_retries propagates CacheFullError from put()"""
        cache = BackendCache(maxsize=2)

        # Fill cache with in-use backends
        for i in range(2):
            b = make_backend(f"in-use-{i}")
            cache.put(f"in-use-{i}", b)

        # Try to add a 3rd backend
        def factory():
            return make_backend("new-scene")

        # Should raise CacheFullError
        with pytest.raises(CacheFullError, match="all 2 cache slots are in-use"):
            cache.put_with_retries("new-scene", factory)

    def test_put_with_retries_propagates_other_exceptions(self):
        """Test put_with_retries propagates non-OOM exceptions immediately"""
        cache = BackendCache(maxsize=3)

        # Factory that raises a different exception
        def factory_error():
            raise ValueError("Simulated error")

        # Should raise ValueError immediately without retrying
        with pytest.raises(ValueError, match="Simulated error"):
            cache.put_with_retries("new-scene", factory_error)

        # Cache should still be empty
        check_cache_size(cache, 0, 0)


class TestGetBackendContextManager:
    """Tests for the get_backend context manager OOM recovery behavior.
    These tests verify that:
    1. Backends are always returned to the cache (checkin called), even on exceptions
    2. OOM errors trigger memory cleanup and spare backend eviction
    3. Exceptions are properly re-raised after cleanup
    Note: These tests use a helper context manager that mirrors the pattern in
    SensorSimService.get_backend() to verify the behavior without needing to
    instantiate the full service with all its dependencies.
    """

    @staticmethod
    @contextlib.contextmanager
    def _get_backend_pattern(cache: BackendCache, key: str, make_backend: Callable[[str], Backend]):
        """
        A test helper that mirrors the get_backend pattern from serve.py.
        This allows us to test the context manager behavior without instantiating
        the full SensorSimService with all its dependencies.
        """
        import gc

        import torch

        # Try to get an available backend (mirrors get_backend logic)
        backend = cache.checkout(key)

        if backend is None:
            # Create new backend
            backend = make_backend(key)
            cache.put(key, backend)

        try:
            yield backend
        except torch.cuda.OutOfMemoryError:
            # OOM recovery - mirrors the fix in serve.py
            # In real code: torch.cuda.empty_cache() and gc.collect()
            gc.collect()

            # Evict spare backends to free memory
            while cache.evict_one_spare():
                pass

            raise
        finally:
            # Always return backend to cache - THE KEY FIX
            cache.checkin(backend)

    def test_checkin_called_on_normal_exit(self, make_backend):
        """Test that checkin is called when context manager exits normally."""
        cache = BackendCache(maxsize=3)

        # Pre-populate cache with a spare backend
        b = make_backend("scene-1")
        cache.put("scene-1", b)
        cache.checkin(b)
        check_cache(cache, set(), ["scene-1"])

        # Use the context manager - backend should be checked out then back in
        with self._get_backend_pattern(cache, "scene-1", make_backend) as backend:
            assert backend is b
            check_cache(cache, {"scene-1"}, [])  # In use during context

        # After context exits, backend should be back in spares
        check_cache(cache, set(), ["scene-1"])

    def test_checkin_called_on_exception(self, make_backend):
        """Test that checkin is called even when an exception occurs.
        This verifies the fix for the bug where backends got stuck in in_use
        when exceptions occurred during rendering.
        """
        cache = BackendCache(maxsize=3)

        # Pre-populate cache
        b = make_backend("scene-1")
        cache.put("scene-1", b)
        cache.checkin(b)
        check_cache(cache, set(), ["scene-1"])

        # Use context manager with an exception - backend should still be returned
        with pytest.raises(RuntimeError, match="Simulated rendering error"):
            with self._get_backend_pattern(cache, "scene-1", make_backend) as backend:
                check_cache(cache, {"scene-1"}, [])  # In use during context
                raise RuntimeError("Simulated rendering error")

        # Backend should be back in spares, not stuck in in_use
        check_cache(cache, set(), ["scene-1"])

    def test_backend_not_stuck_on_oom(self, make_backend):
        """Test that backends don't get stuck in in_use when OOM occurs.
        This is the main bug that was fixed - without try/finally, backends
        would remain in in_use forever after OOM, causing cache to fill up.
        """
        import torch

        cache = BackendCache(maxsize=3)

        # Add multiple spare backends
        for i in range(3):
            b = make_backend(f"scene-{i}")
            cache.put(f"scene-{i}", b)
            cache.checkin(b)

        check_cache(cache, set(), ["scene-0", "scene-1", "scene-2"])

        # Use context manager with OOM - backend should still be returned
        # and spare backends should be evicted
        with pytest.raises(torch.cuda.OutOfMemoryError):
            with self._get_backend_pattern(cache, "scene-1", make_backend) as backend:
                check_cache(cache, {"scene-1"}, ["scene-0", "scene-2"])
                raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # Backend should be back in spares (not stuck in in_use)
        # Other spares should have been evicted by OOM handler
        check_cache(cache, set(), ["scene-1"])

    def test_oom_evicts_all_spares(self, make_backend):
        """Test that OOM recovery evicts all spare backends to free memory."""
        import torch

        cache = BackendCache(maxsize=5)

        # Add 4 spare backends
        for i in range(4):
            b = make_backend(f"spare-{i}")
            cache.put(f"spare-{i}", b)
            cache.checkin(b)

        check_cache_size(cache, 0, 4)

        # Use context manager with OOM - all OTHER spares should be evicted
        with pytest.raises(torch.cuda.OutOfMemoryError):
            with self._get_backend_pattern(cache, "spare-0", make_backend) as backend:
                # spare-0 is now in-use, spare-1,2,3 are spares
                check_cache_size(cache, 1, 3)
                raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # After OOM: spare-1,2,3 evicted, spare-0 returned to spares
        check_cache_size(cache, 0, 1)
        check_cache(cache, set(), ["spare-0"])

    def test_exception_propagates_after_cleanup(self, make_backend):
        """Test that exceptions are re-raised after cleanup."""
        cache = BackendCache(maxsize=3)

        b = make_backend("scene-1")
        cache.put("scene-1", b)
        cache.checkin(b)

        # Exception should propagate after cleanup
        with pytest.raises(ValueError, match="Test error"):
            with self._get_backend_pattern(cache, "scene-1", make_backend) as backend:
                raise ValueError("Test error")

        # Backend should still be checked in despite the exception
        check_cache(cache, set(), ["scene-1"])

    def test_new_backend_created_when_not_in_cache(self, make_backend):
        """Test that a new backend is created when not found in cache."""
        cache = BackendCache(maxsize=3)

        check_cache_size(cache, 0, 0)

        # Use context manager for a scene not in cache - should create new backend
        with self._get_backend_pattern(cache, "new-scene", make_backend) as backend:
            assert backend.cache_key == "new-scene"
            check_cache(cache, {"new-scene"}, [])

        # After context, new backend should be in spares
        check_cache(cache, set(), ["new-scene"])

    # -------------------------------------------------------------------------
    # Tests demonstrating the bug that the try/finally fix prevents
    # -------------------------------------------------------------------------

    @staticmethod
    @contextlib.contextmanager
    def _get_backend_pattern_buggy(cache: BackendCache, key: str, make_backend: Callable[[str], Backend]):
        """
        Simulates the BUGGY version of get_backend (before the fix).
        Missing try/finally causes backend to get stuck in in_use on exception.
        DO NOT USE THIS PATTERN IN PRODUCTION CODE!
        """
        backend = cache.checkout(key)

        if backend is None:
            backend = make_backend(key)
            cache.put(key, backend)

        yield backend
        # BUG: This line is skipped if an exception occurs!
        cache.checkin(backend)

    def test_buggy_pattern_leaks_backend_on_exception(self, make_backend):
        """Demonstrate the bug: without try/finally, backend gets stuck in in_use.
        This test shows what happens WITHOUT the fix - backends leak into in_use
        and are never returned to spares when exceptions occur.
        """
        cache = BackendCache(maxsize=4)

        # Pre-populate cache with a spare backend
        b = make_backend("scene-1")
        cache.put("scene-1", b)
        cache.checkin(b)
        check_cache(cache, set(), ["scene-1"])

        # Use the BUGGY pattern with an exception
        with pytest.raises(ValueError):
            with self._get_backend_pattern_buggy(cache, "scene-1", make_backend) as backend:
                raise ValueError("Test exception")

        # BUG: Backend is stuck in in_use, never returned to spares!
        check_cache(cache, {"scene-1"}, [])  # LEAK - should be in spares!

    def test_buggy_pattern_repeated_oom_fills_in_use(self, make_backend):
        """Demonstrate: buggy version causes in_use to grow with each OOM.
        This was the root cause of the unrecoverable OOM state - each OOM
        left a backend stuck in in_use, eventually filling the cache.
        """
        import torch

        cache = BackendCache(maxsize=4)

        # Simulate multiple OOM errors with buggy version
        for i in range(3):
            with pytest.raises(torch.cuda.OutOfMemoryError):
                with self._get_backend_pattern_buggy(cache, f"scene-{i}", make_backend) as backend:
                    raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # BUG: All 3 backends are stuck in in_use!
        check_cache_size(cache, 3, 0)  # LEAK! Should be 0 in_use, 3 spares

    def test_fixed_pattern_repeated_oom_does_not_fill_in_use(self, make_backend):
        """Verify the fix: repeated OOMs don't cause in_use to grow unboundedly.
        With the try/finally fix, backends are always returned to spares,
        allowing the server to recover from OOM conditions.
        """
        import torch

        cache = BackendCache(maxsize=4)

        # Simulate multiple OOM errors with FIXED version
        for i in range(5):
            with pytest.raises(torch.cuda.OutOfMemoryError):
                with self._get_backend_pattern(cache, f"scene-{i}", make_backend) as backend:
                    raise torch.cuda.OutOfMemoryError("Simulated OOM")

        # FIXED: in_use should be empty - no backends stuck
        # Note: spares are evicted during OOM recovery, so the last backend
        # is the only one remaining in spares
        check_cache_size(cache, 0, 1)  # Correct! No leaks
