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

from nre.models.post_processings.ppisp import PPISP, PPISPSlang
from nre.utils.prober import DEFAULT_DEVICE, FALSE_TRUE, ProberDataSet, ProberTestResult, prober_test_decorator


def get_dataset_quantities(data: ProberDataSet) -> Dict[str, int | float]:
    return {
        "Pixels": data["rgb"].shape[0] * data["rgb"].shape[1],
        "Frames": data["frame_idcs"].max().item() + 1,
        "Cameras": data["unique_camera_idcs"].max().item() + 1,
    }


@prober_test_decorator(
    snapshot_set_name="ppisp",
    test_args_combinations=FALSE_TRUE,
    quantities_getter=get_dataset_quantities,
)
def test_ppisp_slang_vs_pytorch(data: ProberDataSet, use_slang: bool):
    """
    Comprehensive test for PPISP slang vs PyTorch implementations.
    """
    # Extract data
    rgb = data["rgb"]
    coords_xy = data["coords_xy"]
    unique_camera_idcs = data["unique_camera_idcs"]
    frame_idcs = data["frame_idcs"]
    output_rgb_grad = data["output_rgb_grad"]

    # Determine number of cameras and frames from the data
    num_cameras = int(unique_camera_idcs.max().item()) + 1
    num_frames = max(1, int(frame_idcs.max().item()) + 1)
    n_frames_per_camera = [num_frames] * num_cameras

    ppisp = (PPISPSlang if use_slang else PPISP)(DEFAULT_DEVICE, n_frames_per_camera, num_cameras, num_frames)

    # Forward pass
    input_rgb = rgb.clone().detach().requires_grad_(True)
    output_rgb = ppisp(input_rgb, coords_xy, unique_camera_idcs, frame_idcs)

    # Backward pass by propagating the gradient
    output_rgb.backward(output_rgb_grad)

    return ProberTestResult(
        f"PPISP {'(Pytorch)' if not use_slang else '(Slang)'}", (input_rgb, output_rgb, input_rgb.grad)
    )


def test_camera_id_neg1_should_skip_pixels():
    """
    Test that PPISP now properly handles camera_id = -1 to skip pixel processing.
    Pixels with camera_id = -1 should remain unchanged while others are transformed.
    """
    # Create test data
    n_points = 100
    input_rgb = torch.rand(n_points, 3, device="cuda")
    pixel_idxs = torch.randint(0, 100, (n_points, 2), dtype=torch.int16, device="cuda")
    image_res = torch.tensor([[100, 100]], dtype=torch.int16, device="cuda")
    coords_xy = pixel_idxs / (image_res - 1.0)

    frame_idcs = torch.randint(0, 100, (n_points,), dtype=torch.int32, device="cuda")
    unique_camera_idcs = torch.randint(0, 100, (n_points,), dtype=torch.int16, device="cuda")
    num_frames = int(frame_idcs.max().item()) + 1
    num_cameras = int(unique_camera_idcs.max().item()) + 1
    n_frames_per_camera = [num_frames] * num_cameras

    both_minus_one = torch.randint(0, n_points, (10,))
    frame_idcs[both_minus_one] = -1
    unique_camera_idcs[both_minus_one] = -1

    # Test both implementations
    for use_slang in [False, True]:
        # Use a more realistic grid size to ensure transformations happen
        PPISPType = PPISPSlang if use_slang else PPISP
        ppisp = PPISPType(DEFAULT_DEVICE, n_frames_per_camera, num_cameras, num_frames)

        # Process all pixels including those with sensor_id = -1
        output_rgb = ppisp(input_rgb, coords_xy, unique_camera_idcs, frame_idcs)

        # Verify output shape
        assert output_rgb.shape == input_rgb.shape

        # Check that pixels with both ids -1 remain unchanged
        neg_mask = torch.logical_and(unique_camera_idcs == -1, frame_idcs == -1)
        assert neg_mask.any(), "No -1 sensor_id found"

        gamma_corrected_input_rgb = torch.pow(torch.clamp(input_rgb, 0.0, 1.0), 1.0 / 2.2)
        assert torch.allclose(gamma_corrected_input_rgb[neg_mask], output_rgb[neg_mask])
