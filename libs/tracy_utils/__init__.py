# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tracy profiler Python integration for NRE."""

import functools
import logging

from contextlib import contextmanager
from typing import Any, Callable, Optional


_debug_log = logging.getLogger(__name__ + ".debug")


from libs.tracy_utils.interface import TRACY_AVAILABLE, PlotType, tracy_utils_py  # type: ignore


class TracyProfiler:
    """Python wrapper for Tracy profiler."""

    def __init__(self):
        # No need to store profiler instance, we use module functions directly
        pass

    @staticmethod
    def get_instance():
        """Get singleton instance of TracyProfiler."""
        return _profiler_instance

    def initialize(self, enabled: bool = False):
        """Initialize Tracy profiler."""
        if TRACY_AVAILABLE:
            tracy_utils_py.initialize(enabled)

    def mark_frame(self, name: Optional[str] = None):
        """Mark frame boundary and send plot data."""
        if TRACY_AVAILABLE:
            # Call with no arguments if name is None to use C++ default
            if name is None:
                tracy_utils_py.mark_frame()
            else:
                tracy_utils_py.mark_frame(name)

            # Send plot data at frame boundaries (once per frame)
            self._send_frame_plots()

    def message(self, text: str):
        """Send message to Tracy."""
        if TRACY_AVAILABLE:
            tracy_utils_py.message(text)

    def plot(self, plot_type: "PlotType", value: float):
        """Plot a value in Tracy."""
        if TRACY_AVAILABLE:
            tracy_utils_py.plot(plot_type, value)

    def _send_frame_plots(self):
        """Send memory and performance plots at frame boundaries.

        This is called automatically by mark_frame() to send plot data
        once per frame instead of spamming Tracy with updates.
        """
        if not TRACY_AVAILABLE:
            return

        try:
            # Send GPU memory data if CUDA is available
            self._send_gpu_memory_plots()

            # Send CPU memory data
            self._send_cpu_memory_plots()

        except Exception as e:
            _debug_log.debug(f"Failed to send Tracy frame plots: {e}")

    def _send_gpu_memory_plots(self):
        """Send GPU memory usage data as Tracy plots."""
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Get GPU memory statistics (in bytes, convert to MB)
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)

                # Send GPU memory plots
                self.plot(PlotType.GPU_MEM_ALLOCATED_MB, allocated)
                self.plot(PlotType.GPU_MEM_RESERVED_MB, reserved)

        except Exception as e:
            _debug_log.debug(f"Failed to send GPU memory plots: {e}")

    def _send_cpu_memory_plots(self):
        """Send CPU memory usage data as Tracy plots."""
        try:
            # Try to get CPU memory usage
            # First try psutil if available
            try:
                import psutil

                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.plot(PlotType.CPU_MEMORY_MB, memory_mb)
            except ImportError:
                # Fallback to basic resource module if psutil not available
                import resource

                memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # On Linux, ru_maxrss is in KB, on macOS it's in bytes
                import sys

                if sys.platform == "darwin":
                    memory_mb = memory_kb / (1024 * 1024)
                else:
                    memory_mb = memory_kb / 1024
                self.plot(PlotType.CPU_MEMORY_MB, memory_mb)

        except Exception as e:
            _debug_log.debug(f"Failed to send CPU memory plots: {e}")


# Singleton instance
_profiler_instance = TracyProfiler()


# Colors for zones
class Colors:
    """Tracy zone colors."""

    RED = tracy_utils_py.colors.RED if TRACY_AVAILABLE else 0xFF0000
    GREEN = tracy_utils_py.colors.GREEN if TRACY_AVAILABLE else 0x00FF00
    BLUE = tracy_utils_py.colors.BLUE if TRACY_AVAILABLE else 0x0000FF
    YELLOW = tracy_utils_py.colors.YELLOW if TRACY_AVAILABLE else 0xFFFF00
    MAGENTA = tracy_utils_py.colors.MAGENTA if TRACY_AVAILABLE else 0xFF00FF
    CYAN = tracy_utils_py.colors.CYAN if TRACY_AVAILABLE else 0x00FFFF
    ORANGE = tracy_utils_py.colors.ORANGE if TRACY_AVAILABLE else 0xFF8800
    PURPLE = tracy_utils_py.colors.PURPLE if TRACY_AVAILABLE else 0x8800FF


@contextmanager
def zone(name: str, color: int = 0):
    """Context manager for Tracy profiling zones.

    Args:
        name: Name of the zone
        color: Optional color for the zone (use Colors.* constants)

    Example:
        with tracy_utils.zone("render_frame", tracy_utils.Colors.GREEN):
            # Code to profile
            pass
    """
    if TRACY_AVAILABLE:
        zone_obj = tracy_utils_py.ScopedTracyZone(name, color)
        zone_obj.__enter__()
        try:
            yield zone_obj
        finally:
            zone_obj.__exit__(None, None, None)
    else:
        yield None


class TracyZone:
    """A cleaner wrapper for Tracy zones that handles all edge cases."""

    def __init__(self, name: str, color: int = 0, enabled: Optional[bool] = None):
        """Initialize a Tracy zone.

        Args:
            name: Name of the zone
            color: Optional color for the zone
            enabled: Override global enable state (None = use global state)
        """
        self.name = name
        self.color = color
        self.zone_obj = None

        # Determine if we should create a zone
        if enabled is None:
            self.enabled = TRACY_AVAILABLE
        else:
            self.enabled = TRACY_AVAILABLE and enabled

    def __enter__(self):
        """Enter the zone."""
        if self.enabled:
            try:
                self.zone_obj = tracy_utils_py.ScopedTracyZone(self.name, self.color)
                self.zone_obj.__enter__()
            except Exception as e:
                _debug_log.debug(f"Failed to enter Tracy zone '{self.name}': {e}")
                # Silently fail if Tracy has issues
                self.zone_obj = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the zone."""
        if self.zone_obj is not None:
            try:
                self.zone_obj.__exit__(None, None, None)
            except Exception as e:
                _debug_log.debug(f"Failed to exit Tracy zone '{self.name}': {e}")
                # Silently fail on exit
                pass
        return False  # Don't suppress exceptions

    def set_text(self, text: str):
        """Set zone text (if zone is active)."""
        if self.zone_obj is not None:
            try:
                self.zone_obj.set_text(text)
            except Exception as e:
                _debug_log.debug(f"Failed to set Tracy zone text '{text}': {e}")
                pass

    def set_name(self, name: str):
        """Set zone name (if zone is active)."""
        if self.zone_obj is not None:
            try:
                self.zone_obj.set_name(name)
            except Exception as e:
                _debug_log.debug(f"Failed to set Tracy zone name '{name}': {e}")
                pass


def auto_zone(name: Optional[str] = None, color: int = 0):
    """Decorator that automatically creates a Tracy zone for a function.

    This is a cleaner alternative to manual zone management.

    Args:
        name: Zone name (defaults to function name)
        color: Zone color

    Example:
        @tracy_utils.auto_zone("database_query", tracy_utils.Colors.BLUE)
        def query_database():
            ...
    """

    def decorator(func: Callable) -> Callable:
        zone_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with TracyZone(zone_name, color):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def plot_value(plot_type: "PlotType", value: float):
    """Plot a value in Tracy (if enabled).

    This is a convenience function that handles all error cases.
    """
    if TRACY_AVAILABLE:
        try:
            _profiler_instance.plot(plot_type, value)
        except Exception as e:
            _debug_log.debug(f"Failed to plot Tracy value {plot_type}={value}: {e}")
            pass


def mark_frame_boundary(name: Optional[str] = None):
    """Mark a frame boundary in Tracy (if enabled).

    This is a convenience function that handles all error cases.
    """
    if TRACY_AVAILABLE:
        try:
            _profiler_instance.mark_frame(name)
        except Exception as e:
            _debug_log.debug(f"Failed to mark Tracy frame boundary '{name}': {e}")
            pass


def message(text: str):
    """Send a message to Tracy (if enabled).

    This is a convenience function that handles all error cases.
    """
    if TRACY_AVAILABLE:
        try:
            _profiler_instance.message(text)
        except Exception as e:
            _debug_log.debug(f"Failed to send Tracy message '{text}': {e}")
            pass


def profile(name: Optional[str] = None, color: int = 0):
    """Decorator for profiling functions with Tracy.

    Args:
        name: Optional name for the zone (defaults to function name)
        color: Optional color for the zone

    Example:
        @tracy_utils.profile("render", tracy_utils.Colors.GREEN)
        def render_frame():
            pass
    """

    def decorator(func: Callable) -> Callable:
        zone_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with zone(zone_name, color):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# GPU profiling functions
def initialize_gpu_context(name: str = "CUDA", stream: int = 0):
    """Initialize global GPU context for Tracy.

    Creates and initializes a Tracy CUDA context for GPU profiling.

    Args:
        name: Name for the GPU context (default: "CUDA")
        stream: CUDA stream handle. Use 0 for the default stream, or pass
                a valid CUDA stream handle obtained from cudaStreamCreate().

    Note:
        Stream handles must be valid CUDA stream objects. Passing invalid
        stream handles may cause crashes or undefined behavior.
    """
    if TRACY_AVAILABLE:
        tracy_utils_py.initialize_gpu_context(name, stream)


def destroy_gpu_context():
    """Destroy global GPU context.

    Stops GPU profiling and destroys the Tracy CUDA context.
    """
    if TRACY_AVAILABLE:
        tracy_utils_py.destroy_gpu_context()


def collect_gpu(stream: Optional[int] = None):
    """Collect GPU profiling data.

    Forces collection of GPU events. In Tracy 0.12.2, this is usually
    handled automatically by a background thread, but can be called
    manually for immediate collection.

    Args:
        stream: Optional CUDA stream handle. If None, collects from default stream.
                If specified, must be a valid CUDA stream handle obtained from
                cudaStreamCreate().

    Note:
        Stream handles must be valid CUDA stream objects. Passing invalid
        stream handles may cause crashes or undefined behavior.
    """
    if TRACY_AVAILABLE:
        if stream is None:
            tracy_utils_py.collect_gpu()
        else:
            tracy_utils_py.collect_gpu_stream(stream)


def collect_all_gpu():
    """Collect GPU profiling data from all registered streams.

    This ensures GPU events from all streams are flushed to Tracy.
    """
    if TRACY_AVAILABLE:
        tracy_utils_py.collect_all_gpu()


def is_gpu_profiling_available() -> bool:
    """Check if GPU profiling support was compiled in.

    Returns:
        True if Tracy was built with GPU profiling support
    """
    if TRACY_AVAILABLE:
        return tracy_utils_py.is_gpu_profiling_available()
    return False


# Convenience functions
def get_profiler() -> TracyProfiler:
    """Get Tracy profiler instance."""
    return _profiler_instance


def is_available() -> bool:
    """Check if Tracy is available."""
    return TRACY_AVAILABLE


def is_connected() -> bool:
    """Check if Tracy is connected to a profiler."""
    if TRACY_AVAILABLE:
        return tracy_utils_py.is_connected()
    return False


__all__ = [
    "auto_zone",
    "collect_all_gpu",
    "collect_gpu",
    "Colors",
    "destroy_gpu_context",
    "get_profiler",
    "initialize_gpu_context",
    "is_available",
    "is_connected",
    "is_gpu_profiling_available",
    "mark_frame_boundary",
    "message",
    "plot_value",
    "PlotType",
    "profile",
    "TracyProfiler",
    "TracyZone",
    "zone",
]
