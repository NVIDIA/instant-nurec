# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from nre.utils.tests import register_cuda_shutdown_cleanup


# C++ / CUDA libs
try:
    from nre.models.post_processings.bilateral_grid.cuda import libbilateral_grid_cuda_cc  # type: ignore
except ImportError:
    import libbilateral_grid_cuda_cc  # type: ignore

register_cuda_shutdown_cleanup()


def test_gridsize1_optimization_equivalence():
    # Create a 1x1x1 grid
    n_grids = 10
    n_points = 10000
    grid = torch.randn(n_grids, 12, 1, 1, 1, device="cuda")
    pixel_idxs = torch.randint(-5, 105, (n_points, 2), dtype=torch.int16, device="cuda")
    image_res = torch.tensor([[100, 100]], dtype=torch.int16, device="cuda")
    coords_xy = pixel_idxs / (image_res - 1.0)
    rgb = torch.rand(n_points, 3, device="cuda")
    grid_idcs = torch.randint(0, n_grids, (n_points,), dtype=torch.int32, device="cuda")

    # Run forward pass with optimization disabled
    output_no_opt = torch.empty_like(rgb)
    libbilateral_grid_cuda_cc.bilateral_grid_forward(grid, coords_xy, rgb, grid_idcs, output_no_opt, False)

    # Run forward pass with optimization enabled
    output_with_opt = torch.empty_like(rgb)
    libbilateral_grid_cuda_cc.bilateral_grid_forward(grid, coords_xy, rgb, grid_idcs, output_with_opt, True)

    # Verify forward pass results are identical
    torch.testing.assert_close(output_no_opt, output_with_opt, rtol=1e-5, atol=1e-5)

    # Create gradient output
    grad_output = torch.rand_like(rgb)

    # Run backward pass with optimization disabled
    grad_grid_no_opt = torch.zeros_like(grid)
    grad_rgb_no_opt = torch.zeros_like(rgb)
    libbilateral_grid_cuda_cc.bilateral_grid_backward(
        grid, coords_xy, rgb, grid_idcs, grad_output, grad_grid_no_opt, grad_rgb_no_opt, False
    )

    # Run backward pass with optimization enabled
    grad_grid_with_opt = torch.zeros_like(grid)
    grad_rgb_with_opt = torch.zeros_like(rgb)
    libbilateral_grid_cuda_cc.bilateral_grid_backward(
        grid, coords_xy, rgb, grid_idcs, grad_output, grad_grid_with_opt, grad_rgb_with_opt, True
    )

    # Verify backward pass results are identical
    torch.testing.assert_close(grad_grid_no_opt, grad_grid_with_opt)
    torch.testing.assert_close(grad_rgb_no_opt, grad_rgb_with_opt)


def test_gridsize1_shared_memory_reduction_bug():
    """
    Test that exposes the bug in shared memory gradient reduction.

    The bug is that when enable_gridsize1_optimization=True, the shared memory
    reduction code is in the wrong branch and never executes, causing incorrect
    gradient accumulation when multiple threads in the same block reference the same grid.
    """
    # Create a 1x1x1 grid
    n_grids = 1
    # Use 512 points to ensure multiple CUDA blocks and threads per block
    n_points = 512
    grid = torch.ones(n_grids, 12, 1, 1, 1, device="cuda")

    # Set up a simple identity transformation in the grid
    grid[:, 0, :, :, :] = 1.0  # R channel scale
    grid[:, 5, :, :, :] = 1.0  # G channel scale
    grid[:, 10, :, :, :] = 1.0  # B channel scale
    # All other coefficients remain 0

    pixel_idxs = torch.zeros(n_points, 2, dtype=torch.int16, device="cuda")
    image_res = torch.tensor([[2, 2]], dtype=torch.int16, device="cuda")
    coords_xy = pixel_idxs / (image_res - 1.0)
    rgb = torch.ones(n_points, 3, device="cuda") * 0.5

    # All points reference the same grid (grid 0) - this is key for the bug
    grid_idcs = torch.zeros(n_points, dtype=torch.int32, device="cuda")

    # Create gradient output where each sample contributes the same gradient
    grad_output = torch.ones_like(rgb) * 0.1

    # Run backward pass with optimization disabled (this should work correctly)
    grid_clone_no_opt = grid.clone()
    grad_grid_no_opt = torch.zeros_like(grid)
    grad_rgb_no_opt = torch.zeros_like(rgb)
    libbilateral_grid_cuda_cc.bilateral_grid_backward(
        grid_clone_no_opt, coords_xy, rgb, grid_idcs, grad_output, grad_grid_no_opt, grad_rgb_no_opt, False
    )

    # Run backward pass with optimization enabled (this has the bug)
    grid_clone_with_opt = grid.clone()
    grad_grid_with_opt = torch.zeros_like(grid)
    grad_rgb_with_opt = torch.zeros_like(rgb)
    libbilateral_grid_cuda_cc.bilateral_grid_backward(
        grid_clone_with_opt,
        coords_xy,
        rgb,
        grid_idcs,
        grad_output,
        grad_grid_with_opt,
        grad_rgb_with_opt,
        True,
    )

    print(f"No optimization - Grid gradient sum: {grad_grid_no_opt.sum().item()}")
    print(f"With optimization - Grid gradient sum: {grad_grid_with_opt.sum().item()}")
    print(f"Expected gradient contribution per sample: {grad_output[0].sum().item()}")
    print(f"Total expected grid gradient: {grad_output.sum().item()}")

    # The bug: when optimization is enabled, shared memory reduction doesn't happen
    # This means gradients from threads in the same block that reference the same grid
    # are not properly accumulated, leading to incorrect gradient values

    # This assertion should fail due to the bug
    torch.testing.assert_close(
        grad_grid_no_opt,
        grad_grid_with_opt,
        rtol=1e-5,
        atol=1e-5,
        msg="Grid gradients should be identical between optimized and non-optimized paths",
    )


if __name__ == "__main__":
    test_gridsize1_optimization_equivalence()
    test_gridsize1_shared_memory_reduction_bug()
    print("All tests passed!")
