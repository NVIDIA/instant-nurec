# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Numerical equivalence test: CudaLossesFunction vs SlangLossesFunction.

Verifies that the hand-written CUDA kernels produce identical forward outputs
and backward gradients compared to the Slang auto-differentiated kernels.
"""

import unittest

import torch

from libs.losses.functional.cuda_losses_function import CudaLossesFunction
from libs.losses.functional.slang_losses_function import SlangLossesFunction
from libs.losses.kernel.constants import GRID_NUM_CHANNELS
from nre.utils.types import RayFlags


class TestCudaSlangEquivalence(unittest.TestCase):
    def _make_inputs(self, device):
        """Create deterministic test inputs matching both function signatures."""
        torch.manual_seed(42)

        B, H, W = 2, 8, 8

        # RGB
        rgb_pred = torch.randn(B, H, W, 3, device=device, dtype=torch.float32)
        rgb_gt = torch.randn(B, H, W, 3, device=device, dtype=torch.float32)
        rgb_flags = torch.full((B, H, W, 1), RayFlags.RGB_LABEL.value, dtype=torch.int32, device=device)
        # Mark some pixels invalid/sky
        rgb_flags[0, 0, 0, 0] |= RayFlags.INVALID.value
        rgb_flags[0, 1, 1, 0] |= RayFlags.SKY_SEMANTIC.value
        rgb_flags[1, 0, 0, 0] |= RayFlags.DIFIXED.value
        rgb_flags[1, 2, 2, 0] |= RayFlags.SYNTHETIC.value

        rgb_valid = ((rgb_flags & RayFlags.RGB_LABEL.value) != 0) & ((rgb_flags & RayFlags.INVALID.value) == 0)
        n_valid_rgb = rgb_valid.sum().item()
        rgb_factor = 1.0 / n_valid_rgb if n_valid_rgb > 0 else 0.0

        # Background
        n_rays = B * H * W
        bg_pred = torch.rand(n_rays, device=device, dtype=torch.float32)
        rgb_flags_flat = rgb_flags.view(-1)
        bg_valid = (
            ((rgb_flags_flat & RayFlags.INVALID.value) == 0)
            & ((rgb_flags_flat & RayFlags.DIFIXED.value) == 0)
            & ((rgb_flags_flat & RayFlags.SYNTHETIC.value) == 0)
        )
        bg_factor = 1.0 / bg_valid.sum().item() if bg_valid.sum().item() > 0 else 0.0

        # LiDAR
        lidar_pred = torch.randn(B, H, W, 1, device=device, dtype=torch.float32)
        lidar_gt = torch.randn(B, H, W, 1, device=device, dtype=torch.float32)
        lidar_flags = torch.zeros((B, H, W, 1), dtype=torch.int32, device=device)
        lidar_flags[0, 0, 0, 0] |= RayFlags.INVALID.value
        lidar_flags[0, 3, 3, 0] |= RayFlags.DROPPED.value
        lidar_flags[1, 1, 1, 0] |= RayFlags.SKY_SEMANTIC.value

        lidar_valid = ((lidar_flags & RayFlags.INVALID.value) == 0) & ((lidar_flags & RayFlags.DROPPED.value) == 0)
        n_valid_lidar = lidar_valid.sum().item()
        lidar_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        bg_lidar_pred = torch.rand(n_rays, device=device, dtype=torch.float32)
        lidar_flags_flat = lidar_flags.view(-1)
        bg_lidar_valid = ((lidar_flags_flat & RayFlags.INVALID.value) == 0) & (
            (lidar_flags_flat & RayFlags.DROPPED.value) == 0
        )
        bg_lidar_factor = 1.0 / bg_lidar_valid.sum().item() if bg_lidar_valid.sum().item() > 0 else 0.0

        # Intensity
        intensity_pred = torch.randn(B, H, W, 1, device=device, dtype=torch.float32)
        intensity_gt = torch.randn(B, H, W, 1, device=device, dtype=torch.float32)
        intensity_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Raydrop
        raydrop_pred = torch.randn(B, H, W, 1, device=device, dtype=torch.float32)
        raydrop_gt = torch.randn(B, H, W, 1, device=device, dtype=torch.float32)
        raydrop_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Bilateral grids
        grid_per_camera = torch.randn(2, GRID_NUM_CHANNELS, 3, 4, 4, device=device, dtype=torch.float32)
        grid_per_frame = torch.randn(3, GRID_NUM_CHANNELS, 3, 4, 4, device=device, dtype=torch.float32)
        grids_cam = grid_per_camera.view(
            grid_per_camera.shape[0] * grid_per_camera.shape[1],
            grid_per_camera.shape[2],
            grid_per_camera.shape[3],
            grid_per_camera.shape[4],
        )
        grids_frame = grid_per_frame.view(
            grid_per_frame.shape[0] * grid_per_frame.shape[1],
            grid_per_frame.shape[2],
            grid_per_frame.shape[3],
            grid_per_frame.shape[4],
        )
        num_cam = (
            grid_per_camera.shape[0] * grid_per_camera.shape[2] * grid_per_camera.shape[3] * grid_per_camera.shape[4]
        )
        num_frame = (
            grid_per_frame.shape[0] * grid_per_frame.shape[2] * grid_per_frame.shape[3] * grid_per_frame.shape[4]
        )
        grid_drift_per_camera_factor = 1.0 / num_cam
        grid_drift_per_frame_factor = 1.0 / num_frame
        grid_camera_spatial_tv_factor = 1.0 / num_cam
        grid_frame_spatial_tv_factor = 1.0 / num_frame

        # Gaussian scales
        N_scales = 100
        gaussian_scales = torch.randn(N_scales, 3, device=device) - 3.0
        scale_factor = 1.0 / (N_scales * 3)

        # Background texture (cubemap)
        bg_tex = torch.randn(1, 6, 4, 4, 3, device=device, dtype=torch.float32)
        bg_tex_factor = 1.0

        # Gaussian densities
        N_densities = 120
        gaussian_densities = torch.randn(N_densities, device=device, dtype=torch.float32)
        density_factor = 1.0 / N_densities

        # Gaussian visibility mask
        N_gaussian_vis = max(N_scales, N_densities)
        gaussian_visibility = (torch.rand(N_gaussian_vis, device=device) > 0.3).float()

        # Out-of-bound
        out_of_bound_positions = torch.randn(20, 3, device=device, dtype=torch.float32) * 0.5
        out_of_bound_cuboid_dims = torch.rand(20, 3, device=device, dtype=torch.float32) + 0.1
        out_of_bound_factor = 1.0 / (20 * 3)

        # Z-scales
        N_z_scales = 80
        gaussian_z_scales = torch.randn(N_z_scales, 3, device=device) - 3.0
        z_scale_threshold = 0.01
        z_scale_factor = 1.0 / N_z_scales

        return dict(
            rgb_flags=rgb_flags,
            rgb_pred=rgb_pred,
            rgb_gt=rgb_gt,
            rgb_factor=rgb_factor,
            lidar_flags=lidar_flags,
            lidar_pred=lidar_pred,
            lidar_gt=lidar_gt,
            lidar_factor=lidar_factor,
            intensity_pred=intensity_pred,
            intensity_gt=intensity_gt,
            intensity_factor=intensity_factor,
            raydrop_pred=raydrop_pred,
            raydrop_gt=raydrop_gt,
            raydrop_factor=raydrop_factor,
            bg_pred=bg_pred,
            bg_factor=bg_factor,
            bg_lidar_pred=bg_lidar_pred,
            bg_lidar_factor=bg_lidar_factor,
            grids_per_camera=grids_cam,
            grids_per_frame=grids_frame,
            grid_drift_per_camera_factor=grid_drift_per_camera_factor,
            grid_drift_per_frame_factor=grid_drift_per_frame_factor,
            grid_camera_spatial_tv_factor=grid_camera_spatial_tv_factor,
            grid_frame_spatial_tv_factor=grid_frame_spatial_tv_factor,
            gaussian_scales=gaussian_scales,
            scale_factor=scale_factor,
            bg_tex=bg_tex,
            bg_tex_factor=bg_tex_factor,
            gaussian_densities=gaussian_densities,
            density_factor=density_factor,
            gaussian_visibility=gaussian_visibility,
            out_of_bound_positions=out_of_bound_positions,
            out_of_bound_cuboid_dims=out_of_bound_cuboid_dims,
            out_of_bound_factor=out_of_bound_factor,
            gaussian_z_scales=gaussian_z_scales,
            z_scale_threshold=z_scale_threshold,
            z_scale_factor=z_scale_factor,
        )

    def _run_function(self, fn_cls, inputs):
        """Run a LossesFunction with requires_grad on differentiable tensors, return outputs and grads."""
        diff_keys = [
            "rgb_pred",
            "lidar_pred",
            "intensity_pred",
            "raydrop_pred",
            "bg_pred",
            "bg_lidar_pred",
            "grids_per_camera",
            "grids_per_frame",
            "gaussian_scales",
            "bg_tex",
            "gaussian_densities",
            "out_of_bound_positions",
            "gaussian_z_scales",
        ]
        # Clone inputs and set requires_grad
        inp = {}
        for k, v in inputs.items():
            if k in diff_keys:
                inp[k] = v.clone().detach().requires_grad_(True)
            elif isinstance(v, torch.Tensor):
                inp[k] = v.clone().detach()
            else:
                inp[k] = v

        args = [
            inp["rgb_flags"],
            inp["rgb_pred"],
            inp["rgb_gt"],
            inp["rgb_factor"],
            inp["lidar_flags"],
            inp["lidar_pred"],
            inp["lidar_gt"],
            inp["lidar_factor"],
            inp["intensity_pred"],
            inp["intensity_gt"],
            inp["intensity_factor"],
            inp["raydrop_pred"],
            inp["raydrop_gt"],
            inp["raydrop_factor"],
            inp["bg_pred"],
            inp["bg_factor"],
            inp["bg_lidar_pred"],
            inp["bg_lidar_factor"],
            inp["grids_per_camera"],
            inp["grids_per_frame"],
            inp["grid_drift_per_camera_factor"],
            inp["grid_drift_per_frame_factor"],
            inp["grid_camera_spatial_tv_factor"],
            inp["grid_frame_spatial_tv_factor"],
            inp["gaussian_scales"],
            inp["scale_factor"],
            inp["bg_tex"],
            inp["bg_tex_factor"],
            inp["gaussian_densities"],
            inp["density_factor"],
            inp["gaussian_visibility"],
            inp["out_of_bound_positions"],
            inp["out_of_bound_cuboid_dims"],
            inp["out_of_bound_factor"],
            inp["gaussian_z_scales"],
            inp["z_scale_threshold"],
            inp["z_scale_factor"],
        ]

        outputs = fn_cls.apply(*args)

        # Backward: sum all outputs
        total = sum(o.sum() for o in outputs)
        total.backward()

        grads = {k: inp[k].grad for k in diff_keys}
        return outputs, grads

    def test_forward_equivalence(self):
        """Verify forward outputs match between CUDA and Slang implementations."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        inputs = self._make_inputs(device)

        output_names = [
            "rgb_loss",
            "lidar_loss",
            "bg_loss",
            "bg_lidar_loss",
            "grids_drift_loss",
            "grid_camera_spatial_tv_loss",
            "grid_frame_spatial_tv_loss",
            "scale_loss",
            "bg_tex_loss",
            "density_loss",
            "out_of_bound_loss",
            "z_scale_loss",
            "intensity_loss",
            "raydrop_loss",
        ]

        cuda_outputs, _ = self._run_function(CudaLossesFunction, inputs)
        slang_outputs, _ = self._run_function(SlangLossesFunction, inputs)

        for i, name in enumerate(output_names):
            with self.subTest(loss=name):
                torch.testing.assert_close(
                    cuda_outputs[i],
                    slang_outputs[i],
                    atol=1e-5,
                    rtol=1e-5,
                    msg=f"Forward mismatch for {name}",
                )

    def test_backward_equivalence(self):
        """Verify backward gradients match between CUDA and Slang implementations."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        inputs = self._make_inputs(device)

        _, cuda_grads = self._run_function(CudaLossesFunction, inputs)
        _, slang_grads = self._run_function(SlangLossesFunction, inputs)

        for key in cuda_grads:
            with self.subTest(grad=key):
                self.assertIsNotNone(cuda_grads[key], f"CUDA grad for {key} is None")
                self.assertIsNotNone(slang_grads[key], f"Slang grad for {key} is None")
                torch.testing.assert_close(
                    cuda_grads[key],
                    slang_grads[key],
                    atol=1e-4,
                    rtol=1e-4,
                    msg=f"Backward mismatch for grad_{key}",
                )


if __name__ == "__main__":
    unittest.main()
