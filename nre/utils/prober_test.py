# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest
import torch

from nre.utils.prober import ProberDataExplorer, ProberInjectedTensor, ProberTestResult, prober_test_decorator
from nre.utils.tests import is_perf_test_mode, set_perf_test_mode


_called_flags = set()
_called_perf_flags = set()

ARGS_COMBINATIONS = [(0,), (2,), (3,), (4,)]
PERF_ARGS_COMBINATIONS = [(0,), (1,), (2,), (3,), (4,)]

_mock_explorer = None


def mock_explorer():
    global _mock_explorer
    if _mock_explorer is None:
        _mock_explorer = ProberDataExplorer(
            injected_tensors=[ProberInjectedTensor(torch.randn(2, 3), "input", "test_snapshot", 0)]
        )
    return _mock_explorer


@pytest.mark.dependency(name="test_func")
@prober_test_decorator(
    snapshot_set_name="test_snapshot",
    test_args_combinations=ARGS_COMBINATIONS,
    explorer=mock_explorer(),
)
def test_func(data, flag):
    _called_flags.add(flag)
    return ProberTestResult("result", data["input"])


@pytest.mark.dependency(depends=["test_func"])
def test_check_all_args_called():
    assert len(_called_flags) == len(ARGS_COMBINATIONS)


_previous_perf_test_mode = is_perf_test_mode()


@pytest.mark.dependency(name="test_set_perf_test_mode")
def test_set_perf_test_mode():
    set_perf_test_mode(True)


@pytest.mark.dependency(name="test_func_perf", depends=["test_set_perf_test_mode"])
@prober_test_decorator(
    snapshot_set_name="test_snapshot",
    test_args_combinations=None,
    perf_test_args_combinations=PERF_ARGS_COMBINATIONS,
    explorer=mock_explorer(),
)
def test_func_perf(data, flag):
    _called_perf_flags.add(flag)
    assert is_perf_test_mode()
    return ProberTestResult("result", data["input"])


@pytest.mark.dependency(depends=["test_func_perf"])
def test_check_all_perf_args_called():
    assert len(_called_perf_flags) == len(PERF_ARGS_COMBINATIONS)
    set_perf_test_mode(_previous_perf_test_mode)


def test_injected_tensor():
    explorer = ProberDataExplorer(
        injected_tensors=[ProberInjectedTensor(torch.randn(2, 3), "input", "test_snapshot", 0)]
    )
    for snapshot in explorer.enumerate_snapshots("test_snapshot"):
        assert explorer.load_tensor_dict("test_snapshot", snapshot_name=snapshot)["input"].shape == (2, 3)


@prober_test_decorator(
    snapshot_set_name=None,
    test_args_combinations=[(0,)],
)
def test_injected_tensor_decorator(data, flag) -> ProberTestResult:
    assert flag == 0
    t = torch.randn(2, 3)
    return ProberTestResult("result", t)
