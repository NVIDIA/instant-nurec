# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Test to prove that ScopedTimer has minimal execution overhead.

This test creates functions with multiple ScopedTimer decorations and measures
the actual number of executed instructions at runtime to prove
that ScopedTimer has minimal execution overhead.
"""

import sys
import time

from typing import Callable

import nre.utils.profiling as profile

from nre.config.scopedtimer import ProfilerBackend, ScopedTimerConfig, VerbosityLevel
from nre.utils.profiling import ProfileColor, ScopedTimer, TimingTag


# We'll test Tracy through the profile interface instead of importing directly


def undecorated_func() -> None:
    pass


def count_executed_instructions(func: Callable[[], None]) -> int:
    """Count the actual number of instructions executed when running a function."""
    instruction_count = 0

    def trace_calls(frame, event, arg):
        nonlocal instruction_count
        if event == "line":
            instruction_count += 1
        return trace_calls

    # Set the trace function
    old_trace = sys.gettrace()
    sys.settrace(trace_calls)

    try:
        # Execute the function
        func()
    finally:
        # Restore the old trace function
        sys.settrace(old_trace)

    return instruction_count


def create_decorated_function(num_decorations: int = 1000) -> Callable[[], None]:
    """Create a function with many ScopedTimer decorations that use global config."""

    # Start with the base function
    def base_function():
        undecorated_func()

    # Apply multiple ScopedTimer decorations that will use the global config
    decorated_func = base_function
    for i in range(num_decorations):
        # Don't pass config - let it use the global config
        timer = ScopedTimer(f"timer_{i}")
        decorated_func = timer(decorated_func)

    return decorated_func


class TestScopedTimerOverhead:
    """Test class to prove ScopedTimer has minimal execution overhead when verbosity is NONE."""

    def test_scoped_timer_execution_overhead(self):
        """Test that ScopedTimer executes fewer instructions when disabled vs enabled."""
        num_decorations = 64

        results = {}

        for enabled in [True, False]:
            ScopedTimer.set_global_config(ScopedTimerConfig(enabled=enabled, verbosity=VerbosityLevel.NONE))

            # Count executed instructions for undecorated function
            undecorated_executed = count_executed_instructions(undecorated_func)

            # Count executed instructions for decorated function
            decorated_func = create_decorated_function(num_decorations=num_decorations)
            decorated_executed = count_executed_instructions(decorated_func)

            # Calculate overhead per decoration
            overhead_per_decoration = (decorated_executed - undecorated_executed) / num_decorations
            print(
                f"\nEnabled={enabled}: undecorated={undecorated_executed}, decorated={decorated_executed}, overhead={overhead_per_decoration:.2f} instructions/decoration"
            )

            results[enabled] = {
                "undecorated": undecorated_executed,
                "decorated": decorated_executed,
                "overhead": overhead_per_decoration,
            }

        # The key test: disabled should execute fewer instructions than enabled
        disabled_overhead = results[False]["overhead"]
        enabled_overhead = results[True]["overhead"]

        print(f"\nCRITERION CHECK: disabled_overhead={disabled_overhead}  < enabled_overhead={enabled_overhead}")

        assert disabled_overhead < enabled_overhead, (
            f"FAILED: Disabled mode should execute fewer instructions than enabled mode. "
            f"Disabled: {disabled_overhead:.2f}, Enabled: {enabled_overhead:.2f}"
        )

        disabled_overhead_threshold = 10
        assert disabled_overhead < disabled_overhead_threshold, (
            f"FAILED: Disabled mode should execute fewer instructions (ccu) mode. "
            f"Disabled: {disabled_overhead:.2f}, Threshold: {disabled_overhead_threshold:.2f}"
        )

        print(
            f"✓ SUCCESS: Disabled mode executes {enabled_overhead - disabled_overhead:.2f} fewer instructions per decoration than enabled mode"
        )

    def test_profiling_backend_initialization(self):
        """Test that profiling backends can be initialized without errors."""
        # Test different backend values
        backends_to_test = [ProfilerBackend.NONE, ProfilerBackend.TRACY, ProfilerBackend.NVTX]

        for backend in backends_to_test:
            print(f"\nTesting profiling backend: {backend.value}")

            # This should not raise any exceptions
            try:
                profile.initialize(profiling_backend=backend)
                print(f"Successfully initialized {backend.value} backend")
            except (ImportError, ModuleNotFoundError, RuntimeError) as e:
                # For optional backends like Tracy/NVTX, initialization might fail
                # if the dependencies aren't available, but it shouldn't crash
                print(f"Backend {backend.value} initialization returned: {e}")
                # We allow this to pass since Tracy/NVTX might not be available in test environment

    def test_backend_switching(self):
        """Test that ScopedTimer works correctly when switching between backends."""
        print("\nTesting backend switching...")

        # Test with none backend
        profile.initialize(profiling_backend=ProfilerBackend.NONE)

        with ScopedTimer("test_none_backend"):
            time.sleep(0.001)

        print("✓ None backend timer completed")

        # Test tracy backend if available
        try:
            profile.initialize(profiling_backend=ProfilerBackend.TRACY)
            with ScopedTimer("test_tracy_backend"):
                time.sleep(0.001)
            print("✓ Tracy backend timer completed")
        except (ImportError, RuntimeError):
            print("✓ Tracy backend not available, skipping")

        # Test nvtx backend
        profile.initialize(profiling_backend=ProfilerBackend.NVTX)

        with ScopedTimer("test_nvtx_backend"):
            time.sleep(0.001)

        print("✓ NVTX backend timer completed")

        # Switch back to none
        profile.initialize(profiling_backend=ProfilerBackend.NONE)

        with ScopedTimer("test_none_again"):
            time.sleep(0.001)

        print("✓ None backend timer completed after switch")

        print("Backend switching tests passed!")

    def test_parameter_order(self):
        """Test that ScopedTimer parameters work correctly: name, tag, color, config."""
        print("\nTesting parameter order...")

        # Test positional arguments
        with ScopedTimer("test_positional"):
            time.sleep(0.001)
        print("✓ Positional name argument works")

        # Test positional name and tag
        with ScopedTimer("test_positional_tag", TimingTag.DATALOADER):
            time.sleep(0.001)
        print("✓ Positional name and tag arguments work")

        # Test keyword arguments
        with ScopedTimer(name="test_keyword"):
            time.sleep(0.001)
        print("✓ Keyword name argument works")

        # Test with color (keyword-only)
        with ScopedTimer("test_color", color=ProfileColor.GREEN):
            time.sleep(0.001)
        print("✓ Color keyword argument works")

        # Test with config (keyword-only)
        test_config = ScopedTimerConfig(enabled=True, verbosity=VerbosityLevel.BASIC)
        with ScopedTimer("test_config", config=test_config):
            time.sleep(0.001)
        print("✓ Config keyword argument works")

        # Test all arguments together
        with ScopedTimer("test_all", TimingTag.DATALOADER, color=ProfileColor.BLUE, config=test_config):
            time.sleep(0.001)
        print("✓ All arguments together work")

        print("Parameter order tests passed!")


class TestScopedTimerThreading:
    """Test ScopedTimer behavior with multiple threads."""

    def test_thread_local_storage_independence(self):
        """Test that timer measurements are independent across threads using TLS."""
        import threading

        print("\nTesting thread-local storage...")

        # Configure with BASIC verbosity to enable timer state tracking
        # Without proper TLS, this should cause "Timer already running" errors
        test_config = ScopedTimerConfig(
            enabled=True,
            verbosity=VerbosityLevel.BASIC,
            profiling_backend=ProfilerBackend.NONE,  # Use basic timer (time.perf_counter_ns)
        )

        # Create a single shared timer instance to stress-test TLS
        timer_name = "shared_timer_name"
        shared_timer = ScopedTimer(timer_name, config=test_config)

        results = {}
        lock = threading.Lock()

        def tls_worker(thread_id: int, barrier: threading.Barrier, expected_duration: float):
            """Worker that measures its own duration and stores the result."""
            try:
                # Synchronize start to maximize overlap and stress-test TLS
                barrier.wait()

                # All threads use the same timer instance with BASIC verbosity
                # Without TLS, this will throw "Timer already running" RuntimeError
                with shared_timer:
                    time.sleep(expected_duration)

                # Get the actual elapsed time from the timer itself (returns ms)
                actual_duration = shared_timer._bp_elapsed[timer_name] / 1000

                with lock:
                    results[thread_id] = {
                        "expected": expected_duration,
                        "actual": actual_duration,
                    }

            except Exception as e:
                with lock:
                    results[thread_id] = {"error": str(e)}

        # Create threads with different durations to verify independence
        num_threads = 5
        expected_durations = [0.01 * (i + 1) for i in range(num_threads)]
        barrier = threading.Barrier(num_threads)
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=tls_worker, args=(i, barrier, expected_durations[i]))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all threads completed successfully
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"

        # Verify no errors occurred
        for thread_id, result in results.items():
            assert "error" not in result, f"Thread {thread_id} failed: {result['error']}"

        # Print the timer values
        for thread_id, result in results.items():
            expected = result["expected"]
            actual = result["actual"]

            print(f"  Thread {thread_id}: expected {expected:.4f}s, got {actual:.4f}s")
