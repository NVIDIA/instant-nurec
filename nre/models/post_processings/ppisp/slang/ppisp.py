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

from libs.slang_utils.utils import div_up
from nre.models.post_processings.ppisp.slang import libppisp_slang_cc as ppisp_slang  # type: ignore # pycena: skip
from nre.utils.profiling import ScopedTimer


class PPISPSlangFunction(torch.autograd.Function):
    @staticmethod
    @ScopedTimer("PPISPSlangFunction.forward")
    def forward(
        ctx,
        batch_size: int,
        num_cameras: int,
        num_frames: int,
        exposure_params: torch.Tensor,
        vignetting_params: torch.Tensor,
        color_params: torch.Tensor,
        crf_params: torch.Tensor,
        rgb: torch.Tensor,
        coords_xy: torch.Tensor,
        camera_idcs: torch.Tensor,
        frame_idcs: torch.Tensor,
    ):
        # Add shape assertions
        assert exposure_params.shape == (num_frames,), (
            f"Expected exposure_params shape (num_frames,), got {exposure_params.shape}"
        )
        assert vignetting_params.shape == (
            num_cameras,
            3,
            5,
        ), f"Expected vignetting_params shape (num_cameras, 3, 5), got {vignetting_params.shape}"
        assert color_params.shape == (
            num_frames,
            8,
        ), f"Expected color_params shape (num_frames, 8), got {color_params.shape}"
        assert crf_params.shape == (
            num_cameras,
            3,
            7,
        ), f"Expected crf_params shape (num_cameras, 3, 7), got {crf_params.shape}"
        assert rgb.shape[1] == 3, f"Expected rgb to have 3 channels, got {rgb.shape[1]}"
        assert coords_xy.shape[1] == 2, f"Expected coords_xy to have 2 dimensions (x,y), got {coords_xy.shape[1]}"
        assert camera_idcs.shape == (batch_size,), f"Expected camera_idcs shape (batch_size,), got {camera_idcs.shape}"
        assert frame_idcs.shape == (batch_size,), f"Expected frame_idcs shape (batch_size,), got {frame_idcs.shape}"
        assert camera_idcs.dtype == torch.int16, f"Expected camera_idcs to be of type int16, got {camera_idcs.dtype}"
        assert frame_idcs.dtype == torch.int32, f"Expected frame_idcs to be of type int32, got {frame_idcs.dtype}"

        rgb_out = torch.empty_like(rgb)

        ppisp_slang.ppisp(
            (32, 1, 1),
            (div_up(batch_size, 32), 1, 1),
            batch_size,
            num_cameras,
            num_frames,
            (exposure_params, (exposure_params,)),
            (vignetting_params, (vignetting_params,)),
            (color_params, (color_params,)),
            (crf_params, (crf_params,)),
            (rgb, (rgb,)),
            (rgb_out, (rgb_out,)),
            coords_xy,
            camera_idcs,
            frame_idcs,
        )

        ctx.save_for_backward(exposure_params, vignetting_params, color_params, crf_params, rgb, rgb_out)
        ctx.batch_size = batch_size
        ctx.num_cameras = num_cameras
        ctx.num_frames = num_frames
        ctx.coords_xy = coords_xy
        ctx.camera_idcs = camera_idcs
        ctx.frame_idcs = frame_idcs

        return rgb_out

    @staticmethod
    @ScopedTimer("PPISPSlangFunction.backward")
    def backward(ctx, grad_output):
        (exposure_params, vignetting_params, color_params, crf_params, rgb, rgb_out) = ctx.saved_tensors

        batch_size = ctx.batch_size
        num_cameras = ctx.num_cameras
        num_frames = ctx.num_frames
        coords_xy = ctx.coords_xy
        camera_idcs = ctx.camera_idcs
        frame_idcs = ctx.frame_idcs

        # Initialize gradient tensors
        grad_exposure_params = torch.zeros_like(exposure_params)
        grad_vignetting_params = torch.zeros_like(vignetting_params)
        grad_color_params = torch.zeros_like(color_params)
        grad_crf_params = torch.zeros_like(crf_params)

        # Initialize gradient tensor for rgb_in with empty since it's read with loadOnce
        grad_rgb_in = torch.empty_like(rgb)

        # Ensure contiguous tensors for CUDA
        grad_output = grad_output.contiguous()

        ppisp_slang.ppisp_bwd_diff(
            (32, 1, 1),
            (div_up(batch_size, 32), 1, 1),
            batch_size,
            num_cameras,
            num_frames,
            (exposure_params, (grad_exposure_params,)),
            (vignetting_params, (grad_vignetting_params,)),
            (color_params, (grad_color_params,)),
            (crf_params, (grad_crf_params,)),
            (rgb, (grad_rgb_in,)),
            (rgb_out, (grad_output,)),
            coords_xy,
            camera_idcs,
            frame_idcs,
        )

        return (
            None,  # batch_size
            None,  # num_cameras
            None,  # num_frames
            grad_exposure_params,
            grad_vignetting_params,
            grad_color_params,
            grad_crf_params,
            grad_rgb_in,
            None,  # coords_xy
            None,  # unique_camera_idcs
            None,  # frame_idcs
        )
