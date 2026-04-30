# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from libs.losses.kernel.constants import GRID_NUM_CHANNELS
from nre.models.post_processings.bilateral_grid.bilateral_grid import BilateralGrid
from nre.models.post_processings.bilateral_grid.bilateral_grid_cuda import BilateralGridCUDA
from nre.utils.prober import (
    DEFAULT_DEVICE,
    FALSE_TRUE,
    FALSE_TRUE_SQ,
    ProberDataSet,
    ProberTestResult,
    prober_test_decorator,
)


def get_dataset_quantities(data: ProberDataSet) -> Dict[str, int | float]:
    return {"Pixels": data["rgb"].shape[0] * data["rgb"].shape[1]}


def create_bilarf(use_cuda_impl: bool, *args, **kwargs):
    if use_cuda_impl:
        return BilateralGridCUDA(*args, **kwargs).to(DEFAULT_DEVICE)
    else:
        return BilateralGrid(*args, **kwargs).to(DEFAULT_DEVICE)


@prober_test_decorator(
    snapshot_set_name=["bilateral_grid_per_camera"],
    test_args_combinations=FALSE_TRUE,
    quantities_getter=get_dataset_quantities,
)
def test_bilateral_grid_forward_backward(data: ProberDataSet, use_cuda_impl: bool):
    rgb_input = data["rgb"]
    coords_xy = data["coords_xy"]
    grid_idcs = data["grid_idcs"]
    output_rgb_grad = data["output_rgb_grad"]
    bilateral_grid = data["bilateral_grid"]

    num_grids, C, depth, height, width = bilateral_grid.shape

    assert C == GRID_NUM_CHANNELS

    bilarf = create_bilarf(use_cuda_impl, num_grids, width, height, depth)
    bilarf.grid.data.copy_(bilateral_grid)
    rgb_output: torch.Tensor = bilarf(rgb_input, coords_xy, grid_idcs).contiguous()
    rgb_output.backward(output_rgb_grad)

    return ProberTestResult(
        f"BilateralGrid {'(Pytorch)' if not use_cuda_impl else '(CUDA)'}",
        (rgb_input, rgb_input.grad, rgb_output, bilarf.grid, bilarf.grid.grad),
    )


@prober_test_decorator(
    snapshot_set_name=["bilateral_grid_per_camera"],
    test_args_combinations=FALSE_TRUE,
    quantities_getter=get_dataset_quantities,
)
def test_bilateral_grid_forward_backward_gridsize1(data: ProberDataSet, use_cuda_impl: bool):
    rgb_input = data["rgb"]
    coords_xy = data["coords_xy"]
    grid_idcs = data["grid_idcs"]
    output_rgb_grad = data["output_rgb_grad"]
    bilateral_grid = data["bilateral_grid"]

    bilateral_grid = bilateral_grid.mean(dim=[2, 3, 4]).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    num_grids, C, depth, height, width = bilateral_grid.shape

    assert C == GRID_NUM_CHANNELS

    bilarf = create_bilarf(use_cuda_impl, num_grids, width, height, depth)
    bilarf.grid.data.copy_(bilateral_grid)
    rgb_output: torch.Tensor = bilarf(rgb_input, coords_xy, grid_idcs).contiguous()
    rgb_output.backward(output_rgb_grad)

    return ProberTestResult(
        f"BilateralGrid {'(Pytorch)' if not use_cuda_impl else '(CUDA)'}",
        (rgb_input, rgb_input.grad, rgb_output, bilarf.grid, bilarf.grid.grad),
    )


def test_grid_idcs_neg1_should_skip_pixels():
    """
    Test that bilarf now properly handles grid_idcs = -1 to skip pixel processing.
    Pixels with grid_idcs = -1 should remain unchanged while others are transformed.
    """
    # Create test data
    n_points = 100
    rgb_input = torch.rand(n_points, 3, device="cuda")
    pixel_idxs = torch.randint(0, 100, (n_points, 2), dtype=torch.int16, device="cuda")
    image_res = torch.tensor([[100, 100]], dtype=torch.int16, device="cuda")
    coords_xy = pixel_idxs / (image_res - 1.0)

    # Create grid_idcs tensor with some -1 values (should be skipped)
    grid_idcs = torch.randint(-1, 2, (n_points,), dtype=torch.int32, device="cuda")
    grid_idcs[0] = -1  # Ensure we have at least one -1

    # Test both implementations
    for use_cuda_impl in [False, True]:
        # Use a more realistic grid size to ensure transformations happen
        bilarf = create_bilarf(use_cuda_impl, 2, 8, 8, 4)

        # Process all pixels including those with grid_index = -1
        rgb_output = bilarf(rgb_input, coords_xy, grid_idcs)

        # Verify output shape
        assert rgb_output.shape == rgb_input.shape

        # Check that pixels with grid_index = -1 remain unchanged
        neg1_mask = grid_idcs == -1
        assert neg1_mask.any(), "No -1 grid_index found"
        assert torch.allclose(rgb_output[neg1_mask], rgb_input[neg1_mask])
