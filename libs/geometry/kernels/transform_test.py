# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import importlib
import unittest

import lietorch as lt
import slangtorch
import torch


device = torch.device("cuda")
# Deterministic random for reproducibility
torch.manual_seed(123)


class TestSlangTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the pre-compiled SlangTorch library
        transform_test_slang_raw = importlib.import_module("libs.geometry.kernels.libtransform_test_slang_cc")

        # Wrap the raw module to get the friendly API
        cls.slang_module = slangtorch.util.wrapModule(transform_test_slang_raw)
        return super().setUpClass()

    def test_compose_rotations(self):
        NB_VALUES = 100000

        a = torch.nn.functional.normalize(torch.randn(NB_VALUES, 4, device=device))
        b = torch.nn.functional.normalize(torch.randn(NB_VALUES, 4, device=device))

        a.requires_grad = True
        b.requires_grad = True

        # Compute the ground truth.
        ground_truth = (lt.SO3.InitFromVec(a) * lt.SO3.InitFromVec(b)).vec()
        output_grad = torch.randn_like(ground_truth)
        ground_truth.backward(output_grad)
        ground_truth_a_grad = a.grad
        ground_truth_b_grad = b.grad

        # Compute the Slang output.
        count = NB_VALUES
        threads_per_block = 256
        blocks_per_grid = (count + threads_per_block - 1) // threads_per_block

        output = torch.full_like(ground_truth, -2)
        self.slang_module.compose_rotations(
            count=count,
            a=a,
            b=b,
            output=output,
        ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))

        self.assertTrue(torch.allclose(output, ground_truth, atol=1e-6))

        a_grad = torch.full_like(a, -2)
        b_grad = torch.full_like(b, -2)

        self.slang_module.compose_rotations.bwd(
            count=count,
            a=(a, a_grad),
            b=(b, b_grad),
            output=(output, output_grad),
        ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))

        # Since we don't implement tangent space propagation yet, to match lietorch's
        # results, we need to remove the parts of the gradient orthogonal to the tangent space.
        # This is done by projecting the gradient onto the tangent space.
        a_grad = a_grad - (a_grad * a).sum(dim=1, keepdim=True) * a
        b_grad = b_grad - (b_grad * b).sum(dim=1, keepdim=True) * b

        self.assertTrue(torch.allclose(a_grad, ground_truth_a_grad, atol=1e-5))
        self.assertTrue(torch.allclose(b_grad, ground_truth_b_grad, atol=1e-5))

    def test_interpolation(self):
        NB_VALUES = 100000

        axis = torch.randn(NB_VALUES, 3, device=device)
        source_angle = torch.rand(NB_VALUES, device=device) * 2 - 0.5
        target_angle = torch.rand(NB_VALUES, device=device) * 2 - 0.5
        source_points = torch.randn(NB_VALUES, 3, device=device)
        target_points = torch.randn(NB_VALUES, 3, device=device)
        alphas = torch.rand(NB_VALUES, device=device)

        # Choose rotations interpolation as rotation around the same axis but different angle,
        # so it's easy to have the ground truth.
        def compute_quaternion(axis, angle):
            axis_normalized = axis / axis.norm(dim=1, keepdim=True)
            half_angle = angle * 0.5
            sin_half_angle = torch.sin(half_angle)
            cos_half_angle = torch.cos(half_angle)
            quat_xyz = axis_normalized * sin_half_angle.unsqueeze(1)
            quat_w = cos_half_angle.unsqueeze(1)
            return torch.cat([quat_xyz, quat_w], dim=1)

        source_quaternion = compute_quaternion(axis, source_angle)
        target_quaternion = compute_quaternion(axis, target_angle)
        output_quaternion = compute_quaternion(axis, (1 - alphas) * source_angle + alphas * target_angle)
        output_points = (1 - alphas).unsqueeze(-1) * source_points + alphas.unsqueeze(-1) * target_points
        ground_truth = torch.cat([output_points, output_quaternion], dim=1)

        # Compute the Slang output.
        count = NB_VALUES
        threads_per_block = 256
        blocks_per_grid = (count + threads_per_block - 1) // threads_per_block

        a = torch.cat([source_points, source_quaternion], dim=1)
        b = torch.cat([target_points, target_quaternion], dim=1)
        x = alphas
        output = torch.full_like(ground_truth, -2)

        self.slang_module.interpolate(
            count=count,
            a=a,
            b=b,
            x=x,
            output=output,
        ).launchRaw(blockSize=(threads_per_block, 1, 1), gridSize=(blocks_per_grid, 1, 1))

        self.assertTrue(torch.allclose(output, ground_truth, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
