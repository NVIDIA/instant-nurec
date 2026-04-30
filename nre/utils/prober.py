# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import os

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, cast

import torch

from nre.config.prober import ProberConfig
from nre.utils.tests import (
    WithTolerance,
    assert_tests_allclose,
    is_perf_test_mode,
    perf_test_benchmark_result,
    perf_test_print,
    run_timing_comparison_test,
)


logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_test_data_dir(test_data_dir: Optional[str] = None) -> str:
    """Get the test data directory from the environment variable or the default.

    Args:
        test_data_dir: Optional test data directory.

    Returns in order of availability:
        - test_data_dir if provided
        - NRE_PROBER_DIR environment variable if set
        - test_data_prober_generated/test_data if available in the bazel runfiles
    """
    if test_data_dir is None:
        # Get test data directory from environment variable, fallback to default
        default_test_data_dir = "test_data"
        try:
            from python.runfiles import runfiles

            RUNFILES = runfiles.Create()
            anchor_path = RUNFILES.Rlocation("test_data_prober_generated/test_data/.anchor")
            if anchor_path:
                default_test_data_dir = str(Path(anchor_path).parent)
        except Exception:
            pass
        test_data_dir = os.environ.get("NRE_PROBER_DIR", default_test_data_dir)
    return test_data_dir


class TensorProber:
    """
    Utility class for saving tensors for debugging and testing purposes.
    In most case you should get the global prober instance using get_global_prober().
    """

    def __init__(self, test_data_dir: Optional[str] = None) -> None:
        """Setup the prober for being called inside a test.

        Args:
            test_data_dir: Optional test data directory. See get_test_data_dir for more details.
        """
        self._context: List[str] = []
        self._enabled = False
        self._test_data_dir = get_test_data_dir(test_data_dir)
        self._every_n_steps = 0
        self._batch_limit = 0
        self._saved_combinations: set[str] = set()  # Set of full file paths that have been saved

    @property
    def enabled(self) -> bool:
        return self._enabled

    def push_context(self, context: str) -> None:
        self._context.append(context)

    def pop_context(self) -> None:
        self._context.pop()

    def configure(self, prober_config: ProberConfig | None) -> None:
        """Configure the prober from a config object."""
        if prober_config is None:
            return

        # Handle both typed ProberConfig and legacy DictConfig
        if isinstance(prober_config, ProberConfig):
            self._enabled = prober_config.enabled
            self._test_data_dir = prober_config.test_data_dir
            self._every_n_steps = prober_config.every_n_steps
            self._batch_limit = prober_config.batch_limit

        logger.info(
            f"TensorProber configured with enabled: {self._enabled}, test_data_dir: {self._test_data_dir}, every_n_steps: {self._every_n_steps}, batch_limit: {self._batch_limit}"
        )

    def save_tensor(self, tensor: torch.Tensor, name: str, step: int) -> None:
        """
        Save a tensor to the test data directory.

        Args:
            tensor: Tensor to save
            name: Name for the tensor file
            step: Step number
        """
        details = ""
        # Create the full path to the test data directory
        context_path = os.path.join(self._test_data_dir, "-".join(self._context), f"step={step}")
        filename = os.path.join(context_path, f"{name}.pth")

        # Skip if already saved
        if filename in self._saved_combinations:
            return

        if self._batch_limit > 0 and tensor.shape[0] > self._batch_limit:
            n_steps = (tensor.shape[0] + self._batch_limit - 1) // self._batch_limit
            tensor = tensor[::n_steps]
            details = f" (truncated to {tensor.shape[0]} elements)"
        logger.info(f"Saving {name} to {os.path.abspath(filename)}{details}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename):
            os.remove(filename)
        print(f"Saving {name} to {os.path.abspath(filename)}(shape {tensor.shape})")

        torch.save(tensor.clone().detach(), filename)

        # Mark as saved
        self._saved_combinations.add(filename)

    def __call__(self, step: int, name: str, **kwargs) -> Optional[Tuple[Any, ...]]:
        """
        Convenience method for saving tensors given as kwargs if probing is enabled.
        If the key ends with "_grad", it is the gradient of the passed tensor that will be probed during the backward pass.
        In that case the new tensor must be connected to the rest of the computing graph, so it will be returned by the prober in the same order as the kwargs.

        Args:
            step: The step number
            name: The name of the tensor
            kwargs: The tensors to save, if key ends with "_grad", it is the gradient that will be probed during the backward pass.

        Returns:
            The tuple of the new tensors that need to be connected ("_grad" keys) if probing this step, otherwise None.

        Example:
        if (prober_result := get_global_prober()(step=0, name="test_case", input=input_rgb, other_input=other_input, output_grad=output_rgb, other_output_grad=other_output)) is not None:
            # Connect the gradient probing to the rest of the computation graph
            (output_rgb, other_output) = prober_result
        """
        results: Optional[Tuple[Any, ...]] = None
        if self._enabled and step % self._every_n_steps == 0:
            self._context.append(name)
            try:
                for key, value in kwargs.items():
                    suffix = "_grad"
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(f"Value {key} is not a tensor")

                    if key.endswith(suffix) and value.requires_grad:
                        real_key = key[: -len(suffix)]
                        (new_value,) = GradientProber.apply(self, step, name, [real_key], value)
                        results = (new_value,) if results is None else results + (new_value,)
                    else:
                        self.save_tensor(value, key, step)
            finally:
                self._context.pop()

        return results


# Global prober instance for backward compatibility
_global_prober: Optional[TensorProber] = None


def get_global_prober() -> TensorProber:
    """Get the global prober instance, creating it if necessary."""
    global _global_prober
    if _global_prober is None:
        _global_prober = TensorProber()
    return _global_prober


class GradientProber(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prober: TensorProber, step: int, name: str, keys: List[str], *args) -> tuple[Any, ...]:
        ctx.prober = prober
        ctx.step = step
        ctx.name = name
        ctx.keys = keys
        # Even if we don't use the saved tensors in the backward pass, we need to save them for the backward pass to be connected correctly.
        ctx.save_for_backward(*args)
        return tuple(args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        prober = ctx.prober
        step = ctx.step
        name = ctx.name
        keys = ctx.keys

        assert len(grad_outputs) == len(keys), "Number of saved tensors and grad outputs must match"

        for i, key in enumerate(keys):
            prober.push_context(name)
            try:
                if grad_outputs[i] is not None:
                    prober.save_tensor(grad_outputs[i], key + "_grad", step)
            finally:
                prober.pop_context()

        return None, None, None, None, *grad_outputs


class ProberDataSet:
    """
    Utility wrapping a dictionary of tensors to replicate test data results.
    """

    def __init__(self, name: str, tensors: Dict[str, torch.Tensor]):
        """
        Args:
            name: Name of the test data set
            tensors: Dictionary of tensors to save
        """
        self._name = name
        self._tensors = tensors

    @property
    def name(self) -> str:
        return self._name

    def clear_gradients(self) -> None:
        new_tensors = {}
        for key, tensor in self._tensors.items():
            new_tensor = tensor.clone().detach()
            if new_tensor.dtype.is_floating_point:
                new_tensor.requires_grad_(True)
            new_tensors[key] = new_tensor
        self._tensors = new_tensors

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._tensors[key]

    def __str__(self) -> str:
        return f"ProberDataSet(name={self._name}, tensors={self._tensors})"

    def __repr__(self) -> str:
        return f"ProberDataSet(name={self._name}, tensors={self._tensors})"


@dataclass
class ProberInjectedTensor:
    tensor: torch.Tensor
    name: str
    snapshot_set_name: str
    step: int


class ProberDataExplorer:
    """
    Prober data explorer, that can be used to load tensors from the test data directory or inject synthetic tensors.
    """

    _test_data_dir: str

    @lru_cache(maxsize=1)
    def __get_test_data_dir(self) -> str:
        test_data_dir = get_test_data_dir(self._test_data_dir)
        logger.info(f"Using test data directory: {test_data_dir}")
        return test_data_dir or "<injected>"

    def __init__(
        self, test_data_dir: Optional[str] = None, injected_tensors: Optional[Iterable[ProberInjectedTensor]] = None
    ):
        """
        Constructor.

        Args:
            test_data_dir: Optional test data directory. See get_test_data_dir for more details.
            injected_tensors: Optional list of tensors to inject into the test data directory.
        """

        if test_data_dir is None:
            # Get test data directory from environment variable, fallback to default
            default_test_data_dir = "test_data"
            try:
                from python.runfiles import runfiles

                RUNFILES = runfiles.Create()
                anchor_path = RUNFILES.Rlocation("test_data_prober_generated/test_data/.anchor")
                if anchor_path:
                    default_test_data_dir = str(Path(anchor_path).parent)
            except Exception:
                pass

            test_data_dir = os.environ.get("NRE_PROBER_DIR", default_test_data_dir)

        logger.info(f"Using test data directory: {test_data_dir}")
        self._test_data_dir = test_data_dir
        self._injected_tensors: Dict[Path, ProberInjectedTensor] = {}
        for tensor in injected_tensors or []:
            self.inject_tensor(tensor.tensor, tensor.name, tensor.snapshot_set_name, tensor.step)

    def inject_tensor(self, tensor: torch.Tensor, name: str, snapshot_set_name: str, step: int) -> torch.Tensor:
        """Inject a tensor into the test data.

        Args:
            tensor: Tensor to inject
            name: Name of the tensor
            snapshot_set_name: Name of the snapshot set
            step: Step number

        Returns:
            The injected tensor
        """
        self._injected_tensors[
            Path(self.__get_test_data_dir()) / snapshot_set_name / f"step={step}" / f"{name}.pth"
        ] = ProberInjectedTensor(tensor, name, snapshot_set_name, step)
        return tensor

    def load_tensor(
        self, tensor_name: str, context: str, device: str = DEFAULT_DEVICE, subdir: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """
        Load a tensor from a saved .pth file.

        Args:
            tensor_name: Name of the tensor file (e.g., 'rgb', 'pixel_idxs')
            context: Context directory (e.g., 'bilateral_grid_per_camera_input')
            device: Device to load tensor on
            subdir: Optional subdirectory within the context (e.g., 'step=1000')

        Returns:
            Loaded tensor or None if file doesn't exist
        """
        file_path = Path(self.__get_test_data_dir()) / context
        if subdir:
            file_path = file_path / subdir
        file_path = file_path / f"{tensor_name}.pth"

        if file_path in self._injected_tensors:
            return self._injected_tensors[file_path].tensor

        try:
            tensor = torch.load(file_path, map_location=device)
            return tensor
        except Exception as e:
            raise ValueError(f"Failed to load tensor {file_path}: {e}")

    def enumerate_snapshots(self, snapshot_set_name: str) -> Iterator[str]:
        """
        Enumerate all available data sets in a context directory.
        A data set is defined as a subdirectory containing tensor files.

        Args:
            snapshot_set_name: Name of the snapshot set

        Yields:
            Subdirectory names that contain tensor data
        """
        if snapshot_set_name == "<none>":
            yield "<none>"

        if self._test_data_dir is not None:
            snapshot_set_dir = Path(self._test_data_dir) / snapshot_set_name
            if snapshot_set_dir.exists():
                snapshots = [subdir.name for subdir in snapshot_set_dir.iterdir() if subdir.is_dir()]

                # Look for subdirectories that contain .pth files
                for snapshot in snapshots:
                    # Check if this subdirectory contains any .pth files
                    pth_files = list((snapshot_set_dir / snapshot).glob("*.pth"))
                    if pth_files:
                        yield snapshot

        injected_snapshots = set(
            [
                Path(injected_tensor.snapshot_set_name) / f"step={injected_tensor.step}"
                for injected_tensor in self._injected_tensors.values()
            ]
        )

        for injected_snapshot in injected_snapshots:
            injected_snapshot_name = injected_snapshot.name
            if injected_snapshot_name not in injected_snapshots:
                yield injected_snapshot_name

    def load_tensor_dict(
        self, snapshot_set_name: str, device: str = DEFAULT_DEVICE, snapshot_name: Optional[str] = None
    ) -> ProberDataSet:
        """
        Load all available tensors from a specific context directory.

        Args:
            snapshot_set_name: Snapshot set name
            device: Device to load tensors on
            snapshot_name: Optional snapshot name

        Returns:
            Dictionary of tensor name to tensor mapping
        """
        snapshot_dir = Path(self.__get_test_data_dir()) / snapshot_set_name
        if snapshot_name:
            snapshot_dir = snapshot_dir / snapshot_name

        data = ProberDataSet(snapshot_set_name, {})
        if snapshot_dir.exists():
            for pth_file in snapshot_dir.glob("*.pth"):
                tensor_name = pth_file.stem
                tensor = self.load_tensor(tensor_name, snapshot_set_name, device, snapshot_name)
                if tensor is not None:
                    data._tensors[tensor_name] = tensor

        for injected_tensor_path, injected_tensor in self._injected_tensors.items():
            if injected_tensor_path.parent == snapshot_dir:
                data._tensors[injected_tensor.name] = injected_tensor.tensor

        return data

    def run_with_snapshots(self, context: str, device: str, test_func: Callable[[str, ProberDataSet], None]) -> None:
        """
        Run a test function with all available snapshots for a context.
        Asserts fail if no snapshots are found.

        Args:
            context: Context directory name
            test_func: Function to run with each snapshot (snapshot_name, data_dict)
        """
        at_least_one_snapshot = False
        for snapshot_name in self.enumerate_snapshots(context):
            at_least_one_snapshot = True
            data = self.load_tensor_dict(context, device=device, snapshot_name=snapshot_name)
            test_func(snapshot_name, data)

        if not at_least_one_snapshot:
            raise AssertionError(f"No snapshots found for context: {context}")


class ProberTestResult:
    """
    Expected result of test decorated with prober_test_decorator.
    """

    def __init__(
        self, label: str, to_compare: WithTolerance | torch.Tensor | Iterable[WithTolerance | torch.Tensor | None]
    ):
        self.label = label
        self.to_compare = tuple((to_compare,) if isinstance(to_compare, (torch.Tensor, WithTolerance)) else to_compare)
        self.to_compare = tuple(result.clone().detach() if result is not None else None for result in self.to_compare)

    def __str__(self) -> str:
        return self.label


def prober_test_decorator(
    snapshot_set_name: Optional[Union[str, list[str]]] = None,
    test_args_combinations: Optional[Iterable[Sequence[Any]]] = None,
    perf_test_args_combinations: Optional[Iterable[Sequence[Any]]] = None,
    quantities_getter: Optional[Callable[[ProberDataSet], Dict[str, int | float]]] = None,
    explorer: Optional[ProberDataExplorer] = None,
    device: str = DEFAULT_DEVICE,
):
    """
    Decorator that factors out common test logic.

    Args:
        snapshot_set_name: Name of the snapshot set to test with
        extra_args: Extra arguments to pass to the test function
        test_args_combinations: Arguments for validation testing (forward/backward comparison)
        perf_test_args_combinations: Arguments for performance testing (timing benchmarks)
    """
    explorer = explorer or ProberDataExplorer()

    def decorator(func: Callable[..., ProberTestResult]):
        def _call_func_for_args(
            data: ProberDataSet, args_combinations: Optional[Iterable[Sequence[Any]]]
        ) -> Iterable[Callable[[], ProberTestResult]]:
            def clear_grad_and_call_functor(args_combination: Sequence[Any]) -> Callable[[], ProberTestResult]:
                def f():
                    data.clear_gradients()
                    return func(data, *args_combination)

                return f

            return (
                [clear_grad_and_call_functor(args_combination) for args_combination in args_combinations]
                if args_combinations is not None
                else []
            )

        def wrapper():
            def run_snapshot_test(snapshot_name: str, data: ProberDataSet):
                """Run test for a single snapshot."""
                perf_test_print(f"Testing with snapshot: {snapshot_name}")

                # Validation testing
                perf_test_print("Running validation tests...")
                funcs = list(_call_func_for_args(data, test_args_combinations))
                results = []
                for i, func in enumerate(funcs):
                    # Create a closure that captures the current function
                    def make_closure(f):
                        return lambda: f().to_compare

                    results.append(make_closure(func))
                if len(results) > 1:
                    assert_tests_allclose(results)
                if len(results) == 0:
                    logger.warning("No validation tests were run for snapshot: %s, is this intended?", snapshot_name)

                # Need to locally bind perf_test_args_combinations to avoid closure issues
                perf_args_combinations = perf_test_args_combinations
                if perf_args_combinations is None:
                    perf_args_combinations = test_args_combinations

                # Performance testing
                if is_perf_test_mode():
                    perf_test_print("[BENCHMARK] Running timing benchmarks...")

                    quantities = None
                    if quantities_getter is not None:
                        quantities = quantities_getter(data)

                    total_time, timing_results = run_timing_comparison_test(
                        _call_func_for_args(data, perf_args_combinations),
                        quantities=quantities,
                    )

                    description = (
                        (
                            "Dataset: "
                            + " ".join(
                                [
                                    f"{quantity_name}: {quantity_value}"
                                    for quantity_name, quantity_value in quantities.items()
                                ]
                            )
                        )
                        if quantities is not None
                        else "Dataset"
                    )

                    perf_test_benchmark_result(
                        timing_results,
                        extra_info=description,
                        quantities=quantities,
                        total_time=total_time,
                    )

            # Run test with all available snapshots
            for snapshot_name in (
                snapshot_set_name
                if isinstance(snapshot_set_name, list)
                else [snapshot_set_name]
                if snapshot_set_name is not None
                else ["<none>"]
            ):
                explorer.run_with_snapshots(snapshot_name, device, run_snapshot_test)

        return wrapper

    return decorator


# Test argument combinations
FALSE_TRUE = [(False,), (True,)]
FALSE_TRUE_SQ = [(False, False), (True, False), (False, True), (True, True)]
