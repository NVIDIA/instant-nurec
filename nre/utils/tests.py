# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import atexit
import gc
import os
import sys
import time
import unittest

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch


def register_cuda_shutdown_cleanup() -> None:
    """
    Fix "free(): invalid pointer" error for CUDA tests with sys.exit().

    Patches sys.exit() to skip Python finalization for successful tests,
    avoiding race condition between PyTorch CUDA cleanup and extension destructors.

    Usage:
        from nre.utils.tests import register_cuda_shutdown_cleanup
        register_cuda_shutdown_cleanup()
    """

    def force_cuda_cleanup():
        """CUDA cleanup before finalization."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

    atexit.register(force_cuda_cleanup)
    original_exit = sys.exit

    def safe_exit(code: str | int | None = 0) -> NoReturn:
        """Skip finalization on successful pytest runs to avoid double-free."""
        if code == 0 and "pytest" in sys.modules:
            force_cuda_cleanup()
            os._exit(0)
        original_exit(code)

    sys.exit = safe_exit


@dataclass
class WithTolerance:
    """
    A wrapper class to specify a custom tolerance for an object.
    """

    object: Any
    atol: float
    rtol: float

    def clone(self) -> "WithTolerance":
        """Clones the underlying object if it is a torch.Tensor"""
        if isinstance(self.object, torch.Tensor):
            return WithTolerance(self.object.clone(), self.atol, self.rtol)
        else:
            raise NotImplementedError("Supported only for torch.Tensor and dataclasses")

    def detach(self) -> "WithTolerance":
        """Detaches the underlying object if it is a torch.Tensor"""
        if isinstance(self.object, torch.Tensor):
            return WithTolerance(self.object.detach(), self.atol, self.rtol)
        else:
            raise NotImplementedError("Supported only for torch.Tensor and dataclasses")


def assert_objects_allclose(a: Any, b: Any, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    """Compare two objects recursively.

    This function will compare all the fields of the two objects recursively. The supported types of the fields are:

    * torch.Tensor
    * np.ndarray
    * list/tuple/dict
    * int/float/bool/str/None
    * WithTolerance to specify a custom tolerance for an object

    Note the private fields with name starting with `_` are skipped.

    Args:
        a: The first object to compare.
        b: The second object to compare.
        atol: The absolute tolerance for the comparison.
        rtol: The relative tolerance for the comparison.

    Returns:
        None.

    Raises:
        AssertionError: If the two objects are not equal.
    """
    assert_function_supported_types: dict[Callable[[Any, Any], None], list[type]] = {
        partial(torch.testing.assert_close, atol=atol, rtol=rtol): [torch.Tensor],
        partial(np.testing.assert_allclose, atol=atol, rtol=rtol): [np.ndarray, float],
        np.testing.assert_equal: [int, bool, str, type(None)],
    }
    all_supported_types: tuple[type, ...] = tuple(sum(assert_function_supported_types.values(), []))

    if type(a) != type(b):
        raise AssertionError(f"Type mismatch: {type(a)} != {type(b)}")

    if isinstance(a, WithTolerance) and isinstance(b, WithTolerance):
        assert_objects_allclose(a.object, b.object, atol=a.atol, rtol=a.rtol)
        return

    # if a and b are custom classes, we will compare their fields recursively.
    if isinstance(a, type(b)) and is_dataclass(a) and is_dataclass(b):
        # first make sure they have the same set of fields
        a_fields = set([f.name for f in fields(a)])
        b_fields = set([f.name for f in fields(b)])
        if a_fields != b_fields:
            raise AssertionError(f"Fields mismatch: {a_fields} != {b_fields}")

        # then compare the fields recursively
        for k in a_fields:
            if k.startswith("_"):
                # skip private fields
                continue
            assert_objects_allclose(getattr(a, k), getattr(b, k), atol=atol, rtol=rtol)
        return

    # We also support list, tuple, and dict
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(len(a)):
            assert_objects_allclose(a[i], b[i], atol=atol, rtol=rtol)
        return

    elif isinstance(a, tuple) and isinstance(b, tuple):
        for i in range(len(a)):
            assert_objects_allclose(a[i], b[i], atol=atol, rtol=rtol)
        return

    elif isinstance(a, dict) and isinstance(b, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if a_keys != b_keys:
            raise AssertionError(f"Keys mismatch: {a_keys} != {b_keys}")
        for k in a_keys:
            assert_objects_allclose(a[k], b[k], atol=atol, rtol=rtol)
        return

    # if a or b are not supported types, we raise an error.
    if not isinstance(a, all_supported_types) or not isinstance(b, all_supported_types):
        raise AssertionError(f"Unsupported type: {type(a)} or {type(b)}")

    # Then compare the values
    for assert_function, _supported_types in assert_function_supported_types.items():
        supported_types = tuple(_supported_types)
        if isinstance(a, supported_types) and isinstance(b, supported_types):
            assert_function(a, b)
            return


class CommonTestCase(unittest.TestCase):
    def _compareTensor(
        self,
        a_in: np.ndarray | torch.Tensor,
        b_in: np.ndarray | torch.Tensor,
        decimal=6,
    ):
        np.testing.assert_array_almost_equal(
            a_in.cpu().numpy() if isinstance(a_in, torch.Tensor) else a_in,
            b_in.cpu().numpy() if isinstance(b_in, torch.Tensor) else b_in,
            decimal=decimal,
        )


class NonDeterministicTestCase(unittest.TestCase):
    # In non-deterministic test cases we additionally allow for some failure cases due
    # to the non-deterministic nature of the test
    def _compareTensor(
        self,
        a_in: np.ndarray | torch.Tensor,
        b_in: np.ndarray | torch.Tensor,
        absolute_decimal=6,
        relative_decimal=6,
        ratio_of_permitted_failures=0.02,
    ):
        a = a_in.cpu().numpy() if isinstance(a_in, torch.Tensor) else a_in
        b = b_in.cpu().numpy() if isinstance(b_in, torch.Tensor) else b_in

        self.assertEqual(a.shape, b.shape)
        assert isinstance(absolute_decimal, int), "Absolute decimal precision needs to be an integer value"
        assert isinstance(relative_decimal, int), "Relative decimal precision needs to be an integer value"

        a = np.expand_dims(a.flatten(), axis=1)
        b = np.expand_dims(b.flatten(), axis=1)

        # Check if nans exist and enforce they are at the same locations. Then replace them for further checks
        self.assertTrue(np.all(np.isnan(a) == np.isnan(b)))
        a = np.nan_to_num(a)
        b = np.nan_to_num(b)

        n_elements = len(a)
        absolute_diff = np.abs(a - b)
        if absolute_decimal > 0:
            n_above_max_abs = np.where(absolute_diff > 1.5 * 10 ** (-absolute_decimal))[0].shape[0]

            if n_above_max_abs / n_elements > ratio_of_permitted_failures:
                max_abs_diff = np.max(absolute_diff)
                raise AssertionError(
                    f"More than {ratio_of_permitted_failures}% of cases failed the absolute difference check, with the largest being {max_abs_diff}"
                )

        if relative_decimal > 0:
            is_zero = b == 0
            relative_diff = absolute_diff[~is_zero] / np.abs(b[~is_zero])
            n_above_max_rel = np.where(relative_diff > 1.5 * 10 ** (-relative_decimal))[0].shape[0]

            if n_above_max_rel / np.sum(~is_zero) > ratio_of_permitted_failures:
                max_rel_diff = np.max(relative_diff)
                raise AssertionError(
                    f"More than {ratio_of_permitted_failures * 100}% of cases failed the relative difference check, with the largest being {max_rel_diff}"
                )


_is_perf_test_mode: Optional[bool] = None


@dataclass
class TimingStats:
    label: str = ""
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0

    def __str__(self) -> str:
        return self.label


@dataclass
class TimingComparisonResult(TimingStats):
    speedup: float = 0.0
    quantities: Optional[Dict[str, int | float]] = None


def is_perf_test_mode() -> bool:
    """Check if perf test mode is enabled."""
    global _is_perf_test_mode
    if _is_perf_test_mode is None:
        _is_perf_test_mode = os.environ.get("RUN_PERF_TESTS", "").lower() in ["1", "true", "yes"]
    return _is_perf_test_mode


def set_perf_test_mode(value: bool) -> None:
    """Set the perf test mode."""
    global _is_perf_test_mode
    os.environ["RUN_PERF_TESTS"] = "1" if value else "0"
    _is_perf_test_mode = value


def perf_test_print(*args, **kwargs) -> None:
    """
    Print something only if perf test is enabled.

    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments to print

    Returns:
        None
    """
    if is_perf_test_mode():
        print(*args, **kwargs)


def to_unit(value: float, unit: str) -> str:
    """
    Convert a value to a string with a unit.

    Args:
        value: Value to convert
        unit: Unit to use

    Returns:
        String with the value and unit
    """
    prefixes = ["", "k", "M", "G", "T", "P", "E", "Z", "Y"]
    for prefix in prefixes:
        if value < 1e3:
            return f"{value:.3f} {prefix}{unit}"
        value /= 1e3
    return f"{value:.3f} {unit}"


def to_time_unit(value: float) -> str:
    """
    Convert a time value to a string with a unit (ns, us, ms, s).

    Args:
        value: Time value to convert

    Returns:
        String with the value and unit
    """
    prefixes = ["ns", "us", "ms", "s"]
    value = value * 1e9
    for prefix in prefixes:
        if value < 1e3:
            return f"{value:.3f} {prefix}"
        value /= 1000
    return f"{value:.3f} {prefix}"


def perf_test_format_timings(
    timings: Sequence[float],
    labels: Sequence[str],
    extra_info: str = "",
    quantities: Optional[Dict[str, int | float]] = None,
    total_time: Optional[float] = None,
) -> str:
    """
    Format a benchmark result.
    """

    max_label_length = max(len(label) for label in labels)

    def results_per_quantities_str(i: int) -> str:
        if quantities is None:
            return ""
        quantities_per_timing = {}
        for quantity_name, quantity_value in quantities.items():
            quantities_per_timing[quantity_name] = [quantity_value / timings[i] for i in range(len(timings))]
        return "\t".join(
            [
                f"{to_unit(quantitiy_per_timing[i], quantity_name)}/s"
                for quantity_name, quantitiy_per_timing in quantities_per_timing.items()
            ]
        )

    results_str = f"[BENCHMARK] " + (f"[INFO] {extra_info}" if len(extra_info) > 0 else "") + "\n"
    results_str += "\n".join(
        [
            f"[BENCHMARK]    {labels[i]:<{max_label_length}}: {to_time_unit(timings[i])} "
            + (f"\t[SPEEDUP {timings[0] / timings[i]:.2f}x] " if i > 0 else "\t[REFERENCE]       ")
            + f"\t[PERF] {results_per_quantities_str(i)}"
            for i in range(len(timings))
        ]
    )
    if total_time:
        results_str += f"\n[BENCHMARK] Total time: {to_time_unit(total_time)}"

    return results_str


def perf_test_benchmark_result(
    timings: Sequence[float | TimingStats],
    labels: Optional[Sequence[str]] = None,
    extra_info: str = "",
    quantities: Optional[Dict[str, int | float]] = None,
    total_time: Optional[float] = None,
) -> Sequence[float]:
    """
    Print a benchmark result.

    Args:
        timings: List of timings, can be a list of floats or TimingStats in that case the min timing is used
        labels: List of labels, can be None if the label are to be deduced from the TimingStats
        extra_info: Extra information to print
        quantities: Quantities for performance reporting, e.g. {"num_gaussians": 1000} will report the performance in Gaussians/s knowning that 1000 gaussians were processed
        total_time: Total time for the benchmark

    Returns:
        None
    """

    def get_time(time: float | TimingStats) -> float:
        """
        Get the time from a float or TimingStats.

        Args:
            time: Time to get, can be a float or TimingStats. In that case the min timing is used.

        Returns:
            Time
        """
        if isinstance(time, float):
            return time
        elif isinstance(time, TimingStats):
            return time.min
        else:
            raise ValueError(f"Invalid time type: {type(time)}")

    timings_values = [get_time(timing) for timing in timings]
    if labels is None:
        if not all(isinstance(timing, TimingStats) for timing in timings):
            raise ValueError("Labels are not provided and timings are not TimingStats")
        labels = [str(timing) for timing in timings]
    perf_test_print(perf_test_format_timings(timings_values, labels, extra_info, quantities, total_time))

    return timings_values


def perf_test_deduce_backward_result(
    timings: Sequence[float],
    forward_timings: Sequence[float],
    labels: Sequence[str],
    extra_info: str = "",
    quantities: Optional[Dict[str, int | float]] = None,
    total_time: Optional[float] = None,
) -> Sequence[float]:
    """
    Deduce the backward result from the forward result.
    """

    backward_timings = [timings[i] - forward_timings[i] for i in range(len(timings))]
    perf_test_print(perf_test_format_timings(backward_timings, labels, extra_info, quantities, total_time))

    return backward_timings


def assert_tests_allclose(
    impl_funcs: Iterable[Callable[[], Iterable[Any]]], *args, rtol: float = 1e-4, atol: float = 1e-4, **kwargs
) -> list[Any]:
    """
    Run a multiple tests and assert their results are all close by comparing them to the first one with assert_objects_allclose.

    Args:
        impl_funcs: Iterable of implementation test functions that all return the same type of results being an iterable of object supported by assert_objects_allclose
        *args: Arguments to pass to the test functions
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        **kwargs: Keyword arguments to pass to the test functions

    Returns:
        Array of results

    Raises:
        AssertionError: If results don't match within tolerance
    """
    results = []

    for impl_func in impl_funcs:
        result = impl_func(*args, **kwargs)
        results.append(result)
        if len(results) > 1:
            assert_objects_allclose(result, results[0], rtol=rtol, atol=atol)

    return results


class TimingCudaSyncMode(Enum):
    NO_CUDA_SYNC = auto()
    CUDA_SYNC_EACH_CALL = auto()
    CUDA_SYNC_EACH_RUN = auto()


def time_function(
    func: Callable,
    *args,
    num_per_runs: int = 1,
    cuda_sync_mode: TimingCudaSyncMode = TimingCudaSyncMode.CUDA_SYNC_EACH_CALL,
    **kwargs,
) -> float:
    """
    Time a function execution and return the timing.

    Args:
        func: Function to time
        *args: Arguments to pass to the function
        num_per_runs: Number of times to run the function
        cuda_sync_mode: CUDA synchronization mode
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Elapsed time
    """
    start_time = time.perf_counter()
    for _ in range(num_per_runs):
        func(*args, **kwargs)
        if cuda_sync_mode == TimingCudaSyncMode.CUDA_SYNC_EACH_CALL:
            torch.cuda.synchronize()

    if cuda_sync_mode == TimingCudaSyncMode.CUDA_SYNC_EACH_RUN:
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    return (end_time - start_time) / num_per_runs


def run_timing_benchmark(
    func: Callable,
    *args,
    num_warmup: int = 3,
    run_count: int = 10,
    num_per_runs: int = 50,
    cuda_sync_mode: TimingCudaSyncMode = TimingCudaSyncMode.CUDA_SYNC_EACH_CALL,
    **kwargs,
) -> TimingStats:
    """
    Do statistics over many timing runs.

    Args:
        func: Function to time
        *args: Arguments to pass to the function
        num_warmup: Number of warmup runs
        run_count: Number of timing runs
        num_per_runs: Number of times to run the function
        cuda_sync_mode: CUDA synchronization mode
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Dictionary with timing statistics
        - min: Minimum timing
        - max: Maximum timing
        - mean: Mean timing
    """
    result = None
    # Warmup runs
    for _ in range(num_warmup):
        result = func(*args, **kwargs)

    # Timing runs
    timings = []
    for _ in range(run_count):
        elapsed_time = time_function(func, *args, num_per_runs=num_per_runs, cuda_sync_mode=cuda_sync_mode, **kwargs)
        timings.append(elapsed_time)

    return TimingStats(
        label=str(result) if result is not None else "",
        min=min(timings),
        max=max(timings),
        mean=sum(timings) / len(timings),
    )


def run_timing_comparison_test(
    impl_funcs: Iterable[Callable], *args, quantities: Optional[Dict[str, int | float]] = None, **kwargs
) -> Tuple[float, list[TimingComparisonResult]]:
    """
    Run timing comparison between multiple implementations.

    Args:
        impl_funcs: Iterable of implementation functions, outputs are ignored, so the same group of functions can be also used with assert_tests_allclose
        *args: Arguments to pass to the functions
        quantities: Quantities for performance reporting, e.g. {"num_gaussians": 1000} will report the performance in Gaussians/s knowning that 1000 gaussians were processed
        **kwargs: Keyword arguments to pass to the functions

    Returns:
        Tuple with:
        - total time
        - Array with timing statistics and speedup information for each implementation vs the first implementation
            - min: Minimum timing
            - max: Maximum timing
            - mean: Mean timing
            - speedup: Speedup compared to the first implementation (min / min)
            - gaussians_per_sec: Performance in Gaussians/sec for the implementation
    """

    total_time_begin = time.perf_counter()
    # Benchmark all implementations
    impl_stats = []
    for impl_func in impl_funcs:
        impl_stats.append(run_timing_benchmark(impl_func, *args, **kwargs))

    # Calculate performance improvement
    results = []
    for i in range(len(impl_stats)):
        results.append(
            TimingComparisonResult(
                label=impl_stats[i].label,
                min=impl_stats[i].min,
                max=impl_stats[i].max,
                mean=impl_stats[i].mean,
                speedup=impl_stats[i].min / impl_stats[0].min,
                quantities=quantities,
            )
        )

    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_begin

    return total_time, results
