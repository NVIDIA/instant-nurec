# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict

import torch

from libs.geometry.kernels.pose import se3pose_to_inverse_matrix
from nre.utils.geometry import tquat_to_se3_matrix
from nre.utils.prober import ProberDataSet, ProberTestResult, prober_test_decorator


def get_dataset_quantities(data: ProberDataSet) -> Dict[str, int | float]:
    return {"Poses": data["poses_tquat"].shape[0]}


_TestCombinations = [
    (False,),  # Old implementation (tquat_to_se3_matrix + torch.inverse)
    (True,),  # New implementation (se3pose_to_inverse_matrix slang)
]


def _label(use_slang: bool) -> str:
    return "SlangImplementation" if use_slang else "OldImplementation"


@prober_test_decorator(
    snapshot_set_name="gsplat_viewmat",
    test_args_combinations=_TestCombinations,
    perf_test_args_combinations=_TestCombinations,
    quantities_getter=get_dataset_quantities,
)
def test_viewmat_computation(data: ProberDataSet, use_slang: bool):
    """
    Test comparing old and new implementations of viewmat computation for GSplat rendering.

    The old implementation used tquat_to_se3_matrix() followed by torch.inverse().
    The new implementation uses se3pose_to_inverse_matrix() from Slang geometry kernels.

    This test ensures both implementations produce the same results with proper gradient support.
    """
    poses_tquat = data["poses_tquat"]  # (N, 7) - translation (3) + quaternion wxyz (4)
    viewmat_grad = data["viewmat_grad"]  # (N, 4, 4) - gradient for backward pass

    N = poses_tquat.shape[0]

    # Make poses require gradients
    poses_tquat = poses_tquat.clone().requires_grad_(True)

    if use_slang:
        # New implementation: Direct inverse matrix computation using Slang
        viewmat = se3pose_to_inverse_matrix(poses_tquat[..., :3], poses_tquat[..., 3:])  # (N, 4, 4)
    else:
        # Old implementation: Convert to matrix then invert
        viewmat = torch.empty((N, 4, 4), device=poses_tquat.device, dtype=poses_tquat.dtype)
        for i in range(N):
            T_sensor_nre = tquat_to_se3_matrix(poses_tquat[i])  # (4, 4)
            viewmat[i] = torch.inverse(T_sensor_nre)

    # Backward pass
    viewmat.backward(viewmat_grad)

    return ProberTestResult(_label(use_slang), (viewmat, poses_tquat.grad))
