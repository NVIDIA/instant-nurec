# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utilities to help profile parts of the codebase."""

import atexit
import cProfile
import functools
import logging
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from types import TracebackType

# Import profiler backends
from typing import TYPE_CHECKING, Any, Callable, Final, Optional, Self, TextIO, Type, TypeVar, cast

import torch

from omegaconf import DictConfig
from pytorch_lightning.profilers import PassThroughProfiler, PyTorchProfiler

from nre.config.scopedtimer import ProfilerBackend, ScopedTimerConfig, VerbosityLevel, VerbosityLiteral
from nre.utils.trainer import adjust_step_for_world_size


if TYPE_CHECKING:
    import libs.tracy_utils as tracy  # type: ignore[import-not-found, import-untyped]
else:
    tracy = None

TRACY_AVAILABLE = False

# Check for Tracy availability
if os.getenv("TRACY_ENABLE") == "1":
    try:
        import libs.tracy_utils as tracy  # type: ignore[no-redef]

        TRACY_AVAILABLE = tracy.is_available()
    except ImportError:
        TRACY_AVAILABLE = False


_TRACY_UNAVAILABLE_MSG = (
    "Tracy backend requires TRACY_ENABLE=1. Run with internal/scripts/profilers/tracy/run_with_tracy.sh"
)


log: Final = logging.getLogger(__name__)


class ProfilingException(Exception):
    """Custom exception to raise in case of mis-configured CPU/GPU profiling."""


class ProfileColor(Enum):
    """Unified color constants that work across all profiler backends.

    Colors are automatically converted to backend-specific formats:
    - Tracy: RGB hex integer values for rich timeline visualization
    - NVTX: Color name strings (infrastructure only, PyTorch NVTX doesn't support colors)

    Note:
        Use these colors with ScopedTimer for consistent profiling zone appearance
        across different profiling tools and backends.
    """

    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    MAGENTA = 5
    CYAN = 6
    ORANGE = 7
    PURPLE = 8
    LIME = 9
    TEAL = 10
    PINK = 11
    INDIGO = 12
    LIGHT_RED = 13
    LIGHT_GREEN = 14
    LIGHT_BLUE = 15
    DARK_RED = 16
    DARK_GREEN = 17
    DARK_BLUE = 18
    DARK_PURPLE = 19
    GOLD = 20
    SILVER = 21
    WHITE = 22
    BLACK = 23
    GRAY = 24
    BROWN = 25


# Pre-built color mapping dictionaries to avoid reconstruction on every call
_TRACY_COLOR_MAP = {
    ProfileColor.RED: 0xFF0000,
    ProfileColor.GREEN: 0x00FF00,
    ProfileColor.BLUE: 0x0000FF,
    ProfileColor.YELLOW: 0xFFFF00,
    ProfileColor.MAGENTA: 0xFF00FF,
    ProfileColor.CYAN: 0x00FFFF,
    ProfileColor.ORANGE: 0xFF8800,
    ProfileColor.PURPLE: 0x8800FF,
    ProfileColor.LIME: 0x32CD32,
    ProfileColor.TEAL: 0x008080,
    ProfileColor.PINK: 0xFFC0CB,
    ProfileColor.INDIGO: 0x4B0082,
    ProfileColor.LIGHT_RED: 0xFFB6C1,
    ProfileColor.LIGHT_GREEN: 0x90EE90,
    ProfileColor.LIGHT_BLUE: 0x87CEEB,
    ProfileColor.DARK_RED: 0x8B0000,
    ProfileColor.DARK_GREEN: 0x006400,
    ProfileColor.DARK_BLUE: 0x00008B,
    ProfileColor.DARK_PURPLE: 0x301934,
    ProfileColor.GOLD: 0xFFD700,
    ProfileColor.SILVER: 0xC0C0C0,
    ProfileColor.WHITE: 0xFFFFFF,
    ProfileColor.BLACK: 0x000000,
    ProfileColor.GRAY: 0x808080,
    ProfileColor.BROWN: 0x8B4513,
}

_NVTX_COLOR_MAP = {
    ProfileColor.RED: "red",
    ProfileColor.GREEN: "green",
    ProfileColor.BLUE: "blue",
    ProfileColor.YELLOW: "yellow",
    ProfileColor.MAGENTA: "magenta",
    ProfileColor.CYAN: "cyan",
    ProfileColor.ORANGE: "orange",
    ProfileColor.PURPLE: "purple",
    ProfileColor.LIME: "lime",
    ProfileColor.TEAL: "teal",
    ProfileColor.PINK: "pink",
    ProfileColor.INDIGO: "indigo",
    ProfileColor.LIGHT_RED: "lightcoral",
    ProfileColor.LIGHT_GREEN: "lightgreen",
    ProfileColor.LIGHT_BLUE: "lightblue",
    ProfileColor.DARK_RED: "darkred",
    ProfileColor.DARK_GREEN: "darkgreen",
    ProfileColor.DARK_BLUE: "darkblue",
    ProfileColor.DARK_PURPLE: "darkviolet",
    ProfileColor.GOLD: "gold",
    ProfileColor.SILVER: "silver",
    ProfileColor.WHITE: "white",
    ProfileColor.BLACK: "black",
    ProfileColor.GRAY: "gray",
    ProfileColor.BROWN: "brown",
}


def _validate_backend_availability(backend: ProfilerBackend) -> None:
    """Validate that the requested backend is available.

    Args:
        backend: ProfilerBackend enum value to validate

    Raises:
        RuntimeError: If Tracy backend is requested but not available
    """
    if backend == ProfilerBackend.TRACY and not TRACY_AVAILABLE:
        raise RuntimeError(_TRACY_UNAVAILABLE_MSG)


def _convert_color_for_backend(color: Optional[ProfileColor], backend: ProfilerBackend):
    """Convert ProfileColor enum to backend-specific color format.

    Args:
        color: ProfileColor enum value or None
        backend: Target profiler backend

    Returns:
        Backend-specific color value (int for Tracy, str for NVTX, None if not supported)
    """
    if color is None:
        return None

    if backend == ProfilerBackend.TRACY:
        # Tracy uses int colors (RGB hex values)
        return _TRACY_COLOR_MAP.get(color, 0)
    elif backend == ProfilerBackend.NVTX:
        # NVTX uses string colors
        return _NVTX_COLOR_MAP.get(color, None)

    return None


def create_profiler(config: DictConfig) -> PyTorchProfiler:
    """Construct a Pytorch Lightning profiling object, which wraps Pytorch's profiler.

    This setup will produce three output:

    1. A file containing a table with the most time consuming cuda calls: fit-profiling_summary.txt
    2. A trace file to be opened in Google Chrome (URL: chrome://tracing): *.pt.trace.json
    3. A Tensorboard events files containing pytorch_profiler data: events.out.tfevents.*
        NOTE: to use this you must `pip install torch_tb_profiler` - recommended to do so in a new conda env.
        tensorboard --logdir=<out_dir>
    """
    params = config.profiling.params
    log.info("[Profiling] Enabled with configuration: %s", params)

    if params.activities.cpu is False and params.activities.cuda is False:
        raise ProfilingException("Profiling is enable, but neither CPU nor GPU activities are enabled for recording")
    if params.start_step < 0:
        raise ProfilingException(f"Got config.profiling.{params.start_step=}, but must be >= 0")
    if params.num_steps < 1:
        raise ProfilingException(f"Got config.profiling.{params.num_steps=}, but must be > 0")
    if params.start_step + params.num_steps > config.dataset.n_samples_per_epoch * config.trainer.max_epochs:
        msg = f"Configured profiling range is [{params.start_step}, {params.start_step + params.num_steps}], but "
        msg += f"training configured with {config.dataset.n_samples_per_epoch * config.trainer.max_epochs=} total steps"
        raise ProfilingException(f"Profiling step range out of bounds for training run! {msg}")

    if params.emit_nvtx:
        log.warning("[Profiling] Running with autograd NVTX enabled - be sure to run your command under `nsys profile`")
    else:
        log.info("[Profiling] Not emitting NVTX information")

    params.start_step = adjust_step_for_world_size(config.trainer, params.start_step)
    params.num_steps = adjust_step_for_world_size(config.trainer, params.num_steps)

    log.info(f"[Profiling] start_step={params.start_step} num_steps={params.num_steps}")

    prof_dir = f"{config.out_dir}/{config.run_id}/{params.output_dir}"
    summary_file = "profiling_summary"  # saved in prof_dir with automagically-appended suffix: ".txt"

    # Define the scheduler: it is a good idea to profile after a short burn-in period e.g. due to loading/caching data.
    # NOTE: profiling for many steps e.g. 100 can cause training to crash as results aggregation takes a very long time.
    schedule = torch.profiler.schedule(
        skip_first=max(0, params.start_step - 1),
        wait=1,
        warmup=1,
        active=params.num_steps,
    )

    # Experimental configuration is required to record GPU trace. See: https://github.com/pytorch/pytorch/issues/100253
    experimental_config = torch._C._profiler._ExperimentalConfig(verbose=True)  # type: ignore[attr-defined]

    activities = []
    if params.activities.cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if params.activities.cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Construct the full kwargs to pass through to pytorch.profiler.profile, via the lightning wrapper.
    torch_profiler_kwargs = dict(
        activities=activities,
        record_shapes=params.record_shapes,
        profile_memory=params.profile_memory,
        with_flops=params.with_flops,
        with_stack=params.with_stack,
        with_modules=params.with_modules,
        experimental_config=experimental_config,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=prof_dir),
        schedule=schedule,
        emit_nvtx=params.emit_nvtx,
        export_to_chrome=params.export_to_chrome,
    )

    del config.profiling.enabled  # remove attribute, not to pass it as an arg below.

    return PyTorchProfiler(
        dirpath=prof_dir,
        filename=summary_file,
        sort_by_key="cuda_time_total",
        row_limit=200,
        **config.profiling,
        **torch_profiler_kwargs,
    )


C = TypeVar("C", bound=Callable)


class TimingTag(IntEnum):
    DEFAULT = auto()
    DATALOADER = auto()


class BackgroundLogger:
    """
    A background process to write timing results, preserving multiprocessing correctness.
    """

    @staticmethod
    def background(q, summary_func: Callable, details_func: Callable):
        """Background process worker for logging timing results.

        Args:
            q: Multiprocessing queue for receiving timing data
            summary_func: Function to call for final summary
            details_func: Function to call for immediate reporting
        """
        # Results are managed and accessed by the background process
        results: defaultdict[str, list[float]] = defaultdict(list)
        # Loop until a None item is received
        while True:
            try:
                item = q.get(timeout=1.0)  # Add timeout to prevent hanging
            except KeyboardInterrupt:
                # Handle KeyboardInterrupt - break out of loop to shut down cleanly
                break
            except queue.Empty:
                # Handle queue timeout - continue waiting
                continue
            except (OSError, ValueError) as e:
                # Handle queue-related exceptions (closed queue, invalid state)
                # Log at debug level for diagnostics
                log.debug(f"BackgroundLogger queue.get exception: {e}")
                continue

            if item is None:
                break

            try:
                action, pid, depth, duration, report_immediately = item
                results[action].append(duration)
                if report_immediately:
                    details_func(pid, depth, action, duration)
            except Exception as e:
                # Handle malformed items gracefully
                log.debug(f"BackgroundLogger malformed item: {e}")
                continue

        # Summary the results - wrap in try/except for robustness
        try:
            summary_func(results)
        except Exception as e:
            # If summary fails, don't crash the background process
            log.debug(f"BackgroundLogger summary failed: {e}")
            pass

    def __init__(self, summary_func: Callable, details_func: Callable):
        """Initialize background logger with summary and details functions.

        Args:
            summary_func: Function to call for final summary
            details_func: Function to call for individual timing details
        """
        self.queue = multiprocessing.Queue()  # type: ignore
        self.background_process = multiprocessing.Process(
            target=BackgroundLogger.background, args=(self.queue, summary_func, details_func)
        )
        self.background_process.start()
        self._shutdown = False

    def __call__(self, action: str, pid: int, depth: int, duration: float, report_immediately: bool) -> None:
        """Log timing data via background process.

        Args:
            action: Name of the timed action
            pid: Process ID
            depth: Nesting depth for indentation
            duration: Duration in seconds
            report_immediately: Whether to report immediately or buffer
        """
        if not self._shutdown:
            try:
                self.queue.put((action, pid, depth, duration, report_immediately))
            except Exception:
                # Queue might be closed during shutdown
                pass

    def summary(self) -> None:
        """Gracefully shut down the background process."""
        if self._shutdown:
            return

        self._shutdown = True
        try:
            self.queue.put(None)  # Signal shutdown
        except Exception:
            pass

        # Wait for background process with timeout
        if self.background_process.is_alive():
            self.background_process.join(timeout=2.0)

        # Force terminate if still alive
        if self.background_process.is_alive():
            self.background_process.terminate()
            self.background_process.join(timeout=1.0)

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        # Avoid cleanup during interpreter shutdown
        if sys is not None and hasattr(self, "_shutdown"):
            self.summary()


@dataclass
class ScopedTimerWindow:
    start_step: int
    num_steps: int
    repeat_interval: int  # 0 means non-repeating


@dataclass
class CaptureState:
    active: bool = False
    current_step: int = 0


class ScopedTimer(PassThroughProfiler):
    """
    Timer utils based on https://gitlab-master.nvidia.com/omniverse/warp/-/blob/main/warp/utils.py#L662
    """

    _global_config: ScopedTimerConfig = ScopedTimerConfig()
    _global_print_func: Optional[Callable] = None
    _global_logfile: Optional[TextIO] = None
    _global_backend: ProfilerBackend = ProfilerBackend.NONE
    _step_window: Optional[ScopedTimerWindow] = None
    _capture_state: CaptureState = CaptureState()

    @staticmethod
    def set_global_config(config: ScopedTimerConfig) -> None:
        """Set global configuration for all ScopedTimer instances.

        Args:
            config: Global configuration to apply
        """
        ScopedTimer._global_config = config
        ScopedTimer._global_backend = config.profiling_backend
        ScopedTimer.configure_step_windows(config.emit_start_step, config.emit_num_steps, config.emit_repeat_interval)

    @staticmethod
    def set_global_print_func(print_func: Optional[Callable]) -> None:
        """Set global print function for timing output.

        Args:
            print_func: Function to use for printing timing results
        """
        ScopedTimer._global_print_func = print_func

    @staticmethod
    def get_global_print_func() -> Callable:
        """Get the current global print function.

        Returns:
            Print function to use for timing output
        """
        return ScopedTimer._global_print_func or log.info

    @staticmethod
    def set_global_logfile(filename: str) -> None:
        """Set global logfile for timing output.

        Args:
            filename: Path to logfile for timing results

        Note:
            Redirects all timing output to the specified file instead of console.
            File is opened in write mode and will overwrite existing content.
            Closes any previously opened logfile to prevent resource leaks.
        """
        if ScopedTimer._global_logfile is not None:
            try:
                ScopedTimer._global_logfile.close()
            except Exception:
                pass
            ScopedTimer._global_logfile = None

        try:
            logfile = open(filename, "w")
            ScopedTimer._global_logfile = logfile

            def write_to_logfile(msg: str) -> None:
                """Write message to the global logfile."""
                # No lock needed for writing - file handle is thread-safe for writes
                # and we only set _global_logfile once at startup
                if ScopedTimer._global_logfile is not None:
                    ScopedTimer._global_logfile.write(msg + "\n")
                    ScopedTimer._global_logfile.flush()

            ScopedTimer._global_print_func = write_to_logfile
        except IOError:
            log.exception(f"Failed to open logfile {filename}")
            ScopedTimer._global_print_func = log.info
            raise

    @staticmethod
    def maybe_close_global_logfile() -> None:
        """Close global logfile if one is open.

        Note:
            Safely closes the logfile handle and resets to None.
            Called during graceful shutdown to ensure data is flushed.
        """
        if ScopedTimer._global_logfile is not None:
            ScopedTimer._global_logfile.close()
        ScopedTimer._global_logfile = None

    @property
    def config(self) -> ScopedTimerConfig:
        """Get effective config (instance config if set, otherwise global config)."""
        return self._config or ScopedTimer._global_config

    @property
    def print_func(self) -> Callable:
        """Get the current global print function."""
        return ScopedTimer.get_global_print_func()

    @dataclass
    class ProcessLocalStorage:
        thread_lock: threading.Lock = threading.Lock()  # Make sure it is also thread safe
        indent: int = -1

    _local: ProcessLocalStorage = ProcessLocalStorage()

    @property
    def indent(self) -> int:
        """Get the current thread's indent level."""
        with ScopedTimer._local.thread_lock:
            return ScopedTimer._local.indent

    @staticmethod
    def configure_step_windows(
        start_step: Optional[int] = None, num_steps: Optional[int] = None, repeat_interval: Optional[int] = None
    ) -> None:
        """Configure optional step windows that gate backend emission.

        Note: Window configuration cannot be changed once capture has started.
        """
        # Prevent reconfiguration while capture is active
        if ScopedTimer._capture_state.active:
            raise RuntimeError(
                "Cannot reconfigure emit windows while capture is active. "
                "Window configuration must be set before training starts."
            )

        if start_step is None or num_steps is None:
            ScopedTimer._step_window = None
            return

        if start_step < 0 or num_steps <= 0:
            raise ValueError("ScopedTimer step window expects start_step >= 0 and num_steps > 0.")
        if repeat_interval is not None and repeat_interval <= 0:
            raise ValueError("ScopedTimer repeat_interval must be > 0 if set.")
        if repeat_interval is not None and repeat_interval < num_steps:
            raise ValueError(f"ScopedTimer repeat_interval ({repeat_interval}) must be >= num_steps ({num_steps}).")

        ScopedTimer._step_window = ScopedTimerWindow(start_step, num_steps, repeat_interval or 0)

    @staticmethod
    def set_step(step: int) -> None:
        """Set the current global step for backend gating."""
        ScopedTimer._capture_state.current_step = step
        ScopedTimer._update_capture_range()

    @classmethod
    def _backend_active_for_step(cls) -> bool:
        """Check whether backend emission should be active for the current step."""
        if not cls._step_window:
            return True
        return cls._in_window(cls._capture_state.current_step, cls._step_window)

    @classmethod
    def _start_capture(cls) -> None:
        """Start capture for the configured profiler backend.

        Note: Only called when emit windows are configured (emit_start_step and emit_num_steps).
        Without emit window config, this is never called and profiling behavior is unchanged.

        For NVTX backend: calls cudaProfilerStart() for nsys --capture-range=cudaProfilerApi
        """
        cls._capture_state.active = True
        if cls._global_backend == ProfilerBackend.NVTX:
            torch.cuda.cudart().cudaProfilerStart()

    @classmethod
    def _stop_capture(cls) -> None:
        """Stop capture for the configured profiler backend.

        Note: Only called when emit windows are configured (emit_start_step and emit_num_steps).
        Without emit window config, this is never called and profiling behavior is unchanged.

        For NVTX backend: calls cudaProfilerStop() for nsys --capture-range=cudaProfilerApi
        """
        cls._capture_state.active = False
        if cls._global_backend == ProfilerBackend.NVTX:
            torch.cuda.cudart().cudaProfilerStop()

    @classmethod
    def _update_capture_range(cls) -> None:
        """Start/stop capture based on emit windows.

        Note: This only has effect when emit windows are configured via:
        - scopedtimer.emit_start_step
        - scopedtimer.emit_num_steps

        For NVTX backend: calls cudaProfilerStart/Stop for --capture-range=cudaProfilerApi
        """
        window = cls._step_window
        state = cls._capture_state

        # Nothing to do if window is not configured
        if not window:
            return

        # Only emit if a backend is active
        if cls._global_backend == ProfilerBackend.NONE:
            return

        in_window = cls._in_window(state.current_step, window)

        if in_window and not state.active:
            cls._start_capture()
        elif not in_window and state.active:
            cls._stop_capture()

    @staticmethod
    def _in_window(step: int, window: ScopedTimerWindow) -> bool:
        """Return True if step is inside the emit window."""
        offset = step - window.start_step
        if offset < 0:
            return False
        if window.repeat_interval <= 0:
            return offset < window.num_steps
        return (offset % window.repeat_interval) < window.num_steps

    @staticmethod
    def increment_indent() -> None:
        """Increment the current thread's indent level for nested timing output."""
        with ScopedTimer._local.thread_lock:
            ScopedTimer._local.indent += 1

    @staticmethod
    def decrement_indent() -> None:
        """Decrement the current thread's indent level for nested timing output."""
        with ScopedTimer._local.thread_lock:
            ScopedTimer._local.indent -= 1

    _logger: BackgroundLogger | None = None
    _cleanup_registered: bool = False
    _cleanup_lock: threading.Lock = threading.Lock()

    @property
    def logger(self) -> BackgroundLogger:
        """Get or create the background logger instance."""
        if ScopedTimer._logger is None:
            ScopedTimer._logger = BackgroundLogger(ScopedTimer._final_summary, ScopedTimer._local_summary)
            # Register cleanup handlers when logger is first created (thread-safe)
            with ScopedTimer._cleanup_lock:
                if not ScopedTimer._cleanup_registered:
                    ScopedTimer._register_cleanup()
                    ScopedTimer._cleanup_registered = True
        return ScopedTimer._logger

    def __init__(
        self,
        name: Optional[str] = None,
        tag: TimingTag = TimingTag.DEFAULT,
        *,
        color: Optional[ProfileColor] = None,
        config: Optional[ScopedTimerConfig] = None,
        deep: bool = False,
    ) -> None:
        """Context manager object for a timer
        Parameters:
            name (str): Name of timer. Decorator default: func.__name__. Context manager: mandatory
            tag (TimingTag): Category of the timing range, defaults to TimingTag.DEFAULT
            color (ProfileColor): Optional color for profiler zones. Default: None
            config (ScopedTimerConfig): Override global config for this instance. Default: None (use global)
            deep (bool): If True, add nvtx annotations for all subfunctions recursively. Default: False
        """
        super().__init__()
        self._config = config
        self.name = name
        self.tag = tag
        self.color = color
        self.deep = deep

        # A basic profiler using time.perf_counter_ns()
        # The python C-profiler
        self._cp: Optional[cProfile.Profile] = None

        # Profiler backend state (use global backend)
        self.backend_context: list[Any] = []

        self.__profiler_current_zone_name: Optional[str] = None

    @property
    def _bp_start(self) -> dict[str, float]:
        """Thread-local timer start dictionary."""
        if not hasattr(ScopedTimer._tls, "bp_start"):
            ScopedTimer._tls.bp_start = defaultdict(float)
        return ScopedTimer._tls.bp_start

    @property
    def _bp_elapsed(self) -> dict[str, float]:
        """Thread-local timer elapsed dictionary."""
        if not hasattr(ScopedTimer._tls, "bp_elapsed"):
            ScopedTimer._tls.bp_elapsed = defaultdict(float)
        return ScopedTimer._tls.bp_elapsed

    @property
    def cp(self) -> Optional[cProfile.Profile]:
        """Get cProfile.Profile instance for detailed profiling, if enabled."""
        if self.config.verbosity != VerbosityLevel.DETAILS:
            return None
        if self._cp is None:
            self._cp = cProfile.Profile()
        return self._cp

    @property
    def profiler_backend(self) -> ProfilerBackend:
        """Get the active profiler backend for this timer.

        Returns:
            ProfilerBackend enum indicating which profiling system is active

        Note:
            Uses global backend set once during configuration.
        """
        return ScopedTimer._global_backend

    @staticmethod
    def print_summary() -> None:
        """Print final timing summary and shut down background logger."""
        if ScopedTimer._logger is not None:
            ScopedTimer._logger.summary()
        # Close the global logfile
        ScopedTimer.maybe_close_global_logfile()

    @staticmethod
    def _cleanup_on_exit() -> None:
        """Cleanup function called on program exit or signal."""
        try:
            ScopedTimer.print_summary()
        except Exception:
            # Ignore exceptions during cleanup to avoid masking original errors
            pass

    @staticmethod
    def _register_cleanup() -> None:
        """Register cleanup handlers for graceful shutdown."""
        # Register atexit handler (works from any thread)
        atexit.register(ScopedTimer._cleanup_on_exit)

        # Only register signal handlers from the main thread
        try:
            # Check if we're in the main thread
            if threading.current_thread() is threading.main_thread():

                def signal_handler(signum, frame):
                    """Handle termination signals by cleaning up resources."""
                    ScopedTimer._cleanup_on_exit()
                    # Re-raise KeyboardInterrupt for normal handling
                    if signum == signal.SIGINT:
                        raise KeyboardInterrupt()

                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
        except (ValueError, OSError):
            # Signal registration failed (not main thread or not supported)
            # This is fine - atexit handler will still work
            pass

    @staticmethod
    def _final_summary(results: dict[str, list[float]]) -> None:
        """Print final summary of all timing results.

        Args:
            results: Dictionary mapping action names to lists of timing durations
        """
        print_func = ScopedTimer.get_global_print_func()
        print_func("--- Timings Summary: ---")
        for key, value in results.items():
            if len(value) > 1:
                print_func(f" - {key} {sum(value[1:]) / len(value[1:])} (ms)")
            else:  # len(value)<=1 (actually ==1)
                print_func(f" - {key} {value[0]} (ms)")

    @staticmethod
    def _local_summary(pid: int, depth: int, action: str, duration: float) -> None:
        """Print immediate timing result for a single action.

        Args:
            pid: Process ID for identification
            depth: Nesting depth for indentation
            action: Name of the action that was timed
            duration: Duration in milliseconds
        """
        print_func = ScopedTimer.get_global_print_func()
        print_func(f"[{pid}]{'  ' * depth}{action} took {duration:.2f} ms")

    def _post_processing(self, action_name: str) -> None:
        """Handle post-processing after timing stops (logging, profiling output).

        Args:
            action_name: Name of the action that was timed
        """
        verbosity = self.config.verbosity

        # -- Basic Profiler --
        if verbosity != VerbosityLevel.NONE:
            should_print = verbosity == VerbosityLevel.BASIC or verbosity == VerbosityLevel.DETAILS
            self.logger(action_name, os.getpid(), self.indent, self._bp_elapsed[action_name], should_print)

        # -- CProfile --
        if self.cp is not None:
            self.cp.print_stats(sort="tottime")

    def __enter__(self) -> Self:
        """for using the with statement"""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """for using the with statement"""
        self.stop()

    def is_running(self, action_name: str) -> bool:
        """Check if a timer with the given name is currently running.

        Args:
            action_name: Name of the timer to check

        Returns:
            True if the timer is running, False otherwise
        """
        return action_name in self._bp_start and self._bp_start[action_name] > 0.0

    def start(self, action_name: str | None = None, *, force: bool = False) -> None:
        """Start timing for the given action.

        Args:
            action_name: Name of the action to time (uses instance name if None)
            force: If True, stop any existing timer with the same name first
        """
        # Property accessor called once
        config = self.config
        if not config.enabled:
            return

        backend_active = ScopedTimer._backend_active_for_step()

        # -- Set the action name --
        action_name = action_name or self.name
        assert action_name is not None, "Timer name is required"

        # -- Increment the indent for local summary --
        ScopedTimer.increment_indent()

        # -- Check if the timer is already running --
        if config.verbosity != VerbosityLevel.NONE and self.is_running(action_name):
            if force:
                self.stop(action_name, force=True)
            else:
                raise RuntimeError(f"Timer {action_name} is already running")

        # 2. Maybe synchronize the GPU
        if config.synchronize and self.tag != TimingTag.DATALOADER:
            torch.cuda.synchronize()

        # 3. Start the different profilers

        # -- Basic Profiler --
        if config.verbosity != VerbosityLevel.NONE:
            self._bp_start[action_name] = time.perf_counter_ns()

        # -- CProfile --
        if self.cp is not None:
            # mypy complains about clear() not being present in cProfile.Profile
            # but it works as intended
            self.cp.clear()  # type: ignore
            self.cp.enable()

        if backend_active and self.profiler_backend != ProfilerBackend.NONE:
            # -- Profiler Zones (Tracy/NVTX), ignored if already withing a deep trace block --
            if not self.deep_trace.refcount > 0:
                self._start_profiler_zone(action_name)

            # -- Deep Tracing (Tracy/NVTX) --
            if self.deep:
                self._start_deep_trace()

    def stop(self, action_name: str | None = None, *, force: bool = False) -> None:
        """Stop timing for the given action.

        Args:
            action_name: Name of the action to stop (uses instance name if None)
            force: If True, don't error if timer is not running
        """
        # Property accessor called once
        config = self.config
        if not config.enabled:
            return

        backend_active = ScopedTimer._backend_active_for_step()

        # -- Set the action name --
        action_name = action_name or self.name
        assert action_name is not None, "action_name cannot be None"

        # -- Check if the timer is running --
        if config.verbosity != VerbosityLevel.NONE and not self.is_running(action_name):
            if force:
                return
            else:
                raise RuntimeError(f"Timer {action_name} is not running")

        # 2. Maybe synchronize the GPU
        if config.synchronize and self.tag != TimingTag.DATALOADER:
            torch.cuda.synchronize()

        # 3. Stop different timers (in reverse order)

        if backend_active and self.profiler_backend != ProfilerBackend.NONE:
            # -- Deep Tracing (Tracy/NVTX) --
            if self.deep:
                self._stop_deep_trace()

            # -- Profiler Zones (Tracy/NVTX), ignored if already withing a deep trace block --
            if not self.deep_trace.refcount > 0:
                self._stop_profiler_zone()

        # -- CProfile --
        if self.cp is not None:
            self.cp.disable()

        # -- Basic Profiler --
        if config.verbosity != VerbosityLevel.NONE:
            elapsed = (time.perf_counter_ns() - self._bp_start[action_name]) / 1000000.0
            self._bp_start[action_name] = 0.0  # --> This marks the timer as stopped
            self._bp_elapsed[action_name] = elapsed

        # 3. Post-processing
        self._post_processing(action_name)
        ScopedTimer.decrement_indent()

    def __call__(self, func: C) -> C:
        """for using the the class as a decorator"""

        # Set the timer name to the function's name if no name was provided
        if self.name is None:
            self.name = func.__name__
            # We also keep a reference to the function to make the dependency explicit for pycena
            self.func = func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function that executes the decorated function with timing."""
            # Short circuit if timing is disabled (12 instructions => 3 instructions)
            if not self.config.enabled:
                return func(*args, **kwargs)

            # Always use context manager for profiling, even if timing is disabled
            # The context manager will handle profiling vs timing separately
            with self:
                return func(*args, **kwargs)

        return cast(C, wrapper)  # we just forward the function call in the wrapper

    _tls = threading.local()

    @dataclass
    class DeepTraceStorage:
        refcount: int = 0
        previous_trace_func: Optional[Callable] = None
        stack: list[bool] = field(default_factory=list)

    @property
    def deep_trace(self) -> DeepTraceStorage:
        if not hasattr(ScopedTimer._tls, "deep_trace"):
            ScopedTimer._tls.deep_trace = ScopedTimer.DeepTraceStorage()
        return ScopedTimer._tls.deep_trace

    def _create_deep_trace_func(self, previous_trace_func: Optional[Callable]) -> Callable:
        """Create a trace function for deep profiling that annotates all function calls.

        Returns:
            A trace function that can be passed to sys.settrace()

        Note:
            Only activates when deep=True and profiling backend supports it (currently NVTX only).
            Automatically pushes/pops NVTX ranges for each function call/return.
        """

        def trace_func(frame, event, arg):
            """Trace function that adds NVTX annotations for all function calls."""
            if previous_trace_func is not None:
                previous_trace_func(frame, event, arg)

            # Only trace function calls and returns
            if event == "call":
                # Get function name with module context
                func_name = frame.f_code.co_name
                module_name = frame.f_globals.get("__name__", "")

                # Filter out internal functions and modules we don't want to trace
                if module_name and not any(
                    skip in module_name for skip in ["importlib", "abc", "typing", "dataclasses", "enum", "contextlib"]
                ):
                    full_name = f"{module_name}.{func_name}" if module_name else func_name

                    torch.cuda.nvtx.range_push(full_name)
                    self.deep_trace.stack.insert(0, True)
                else:
                    self.deep_trace.stack.insert(0, False)

            elif event == "return":
                # Pop the NVTX range when function returns
                if len(self.deep_trace.stack) > 0 and self.deep_trace.stack.pop():
                    torch.cuda.nvtx.range_pop()

            return trace_func

        return trace_func

    def _start_deep_trace(self) -> None:
        """Start deep tracing to annotate all function calls recursively.

        Note:
            Only activates when deep=True and using NVTX backend.
            Installs a trace function that will annotate every function call.
            Uses reference counting to handle nested deep timers correctly.
        """
        # Increment reference count for this thread
        self.deep_trace.refcount += 1

        # Only install trace function on the first deep timer (refcount == 1)
        if self.deep_trace.refcount == 1:
            # Save the current trace function and install our trace function
            self.deep_trace.previous_trace_func = sys.gettrace()
            sys.settrace(self._create_deep_trace_func(self.deep_trace.previous_trace_func))

    def _stop_deep_trace(self) -> None:
        """Stop deep tracing and clean up any remaining NVTX ranges.

        Note:
            Restores the previous trace function and pops any unclosed NVTX ranges.
            Uses thread-local storage to ensure thread-safety.
            Uses reference counting to handle nested deep timers correctly.
        """
        # Decrement reference count for this thread
        self.deep_trace.refcount -= 1

        # Only restore trace function when all deep timers have stopped (refcount == 0)
        if self.deep_trace.refcount == 0:
            previous_trace_func = self.deep_trace.previous_trace_func
            sys.settrace(previous_trace_func)
            self.deep_trace.previous_trace_func = None

        # Clean up any remaining ranges in the stack (thread-local)
        while len(self.deep_trace.stack) > 0:
            if self.deep_trace.stack.pop():
                torch.cuda.nvtx.range_pop()

    def _start_profiler_zone(self, action_name: str) -> None:
        """Start profiler zone based on configured backend.

        Args:
            action_name: Name of the profiling zone to start

        Note:
            Creates Tracy or NVTX profiling zones independent of timing settings.
            Zone colors are converted to backend-specific formats automatically.
        """
        if not action_name:
            return

        backend = self.profiler_backend
        color = _convert_color_for_backend(self.color, backend)

        if backend == ProfilerBackend.TRACY and TRACY_AVAILABLE:
            try:
                if color is not None and isinstance(color, int):
                    tracy_zone = tracy.TracyZone(action_name, color)
                else:
                    tracy_zone = tracy.TracyZone(action_name)

                self.backend_context.append(tracy_zone)
                tracy_zone.__enter__()
            except (AttributeError, RuntimeError, TypeError) as e:
                # Log but don't crash if Tracy fails (initialization, API, or parameter issues)
                log.debug(f"Failed to start Tracy zone: {e}")

        elif backend == ProfilerBackend.NVTX:
            # Note: torch.cuda.nvtx.range_push does not support color parameter,
            # but full NVTX library would support: nvtx.start_range(name, color=color)
            # kept for future compatibility
            torch.cuda.nvtx.range_push(action_name)
            self.backend_context.append(True)  # Just mark that a zone was started

    def _stop_profiler_zone(self) -> None:
        """Stop profiler zone based on configured backend.

        Note:
            Safely stops Tracy or NVTX profiling zones and cleans up context.
            Handles cases where no zone was started gracefully.
        """
        if len(self.backend_context) == 0:
            return

        backend = self.profiler_backend
        if backend == ProfilerBackend.TRACY and TRACY_AVAILABLE:
            self.backend_context.pop().__exit__(None, None, None)
        elif backend == ProfilerBackend.NVTX:
            torch.cuda.nvtx.range_pop()
            self.backend_context.pop()

        self.backend_context = []


def mark_frame_boundary(name: Optional[str] = None):
    """Mark a frame boundary for Tracy profiling.

    Args:
        name: Optional name for the frame boundary

    Note:
        This sends memory plots to Tracy at frame boundaries.
        Only active when Tracy backend is enabled.
    """
    if TRACY_AVAILABLE:
        try:
            tracy.mark_frame_boundary(name)
        except Exception:
            # Don't let Tracy failures affect the main application
            pass


def initialize(profiling_backend: ProfilerBackend):
    """Initialize the profiling system with specified backend.

    Args:
        profiling_backend: Backend to use (ProfilerBackend enum)

    Note:
        For Tracy backend, initializes GPU context and starts capture.
    """
    _validate_backend_availability(profiling_backend)
    backend = profiling_backend

    if backend == ProfilerBackend.TRACY and TRACY_AVAILABLE:
        tracy_no_gpu = os.getenv("TRACY_NO_GPU", "0")

        profiler = tracy.get_profiler()
        profiler.initialize(True)

        # Only initialize GPU context if GPU profiling is not disabled
        if tracy_no_gpu != "1":
            tracy.initialize_gpu_context("CUDA", 0)

    # Log profiler status if enabled
    if backend != ProfilerBackend.NONE:
        log.info(f"Profiler backend: {backend.value}")

        if backend == ProfilerBackend.TRACY and TRACY_AVAILABLE:
            log.info("Tracy profiler initialized and enabled")

            # Create a dummy zone to trigger profiler initialization
            with ScopedTimer("profiler_init", color=ProfileColor.PURPLE):
                pass

            # Check connection status
            try:
                connected = tracy.is_connected()
                log.info(f"Tracy connection status: {'CONNECTED' if connected else 'NOT CONNECTED'}")
                log.info("Tracy should be listening on port 8086")
            except Exception:
                pass


def configure_scopedtimer_from_cli(
    *,
    enable_timing: bool,
    timing_verbosity: VerbosityLiteral,
    timing_logfile: Optional[str],
    timing_synchronize: bool,
    profiling_backend: ProfilerBackend,
    print_func: Optional[Callable[[str], None]] = None,
) -> ScopedTimerConfig:
    """Configure ScopedTimer similarly to CLI-driven services like serve_grpc."""
    config = ScopedTimerConfig(
        enabled=enable_timing,
        verbosity=timing_verbosity,
        profiling_backend=profiling_backend,
        synchronize=timing_synchronize,
        logfile=timing_logfile,
    )
    ScopedTimer.set_global_config(config)

    if timing_logfile:
        log.info(f"Writing timing results to: {timing_logfile}")
        ScopedTimer.set_global_logfile(timing_logfile)
    else:
        ScopedTimer.set_global_print_func(print_func or log.info)

    initialize(profiling_backend=config.profiling_backend)
    return config
