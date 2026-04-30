# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

import torch

# Python wrapper (strategy/gsplat.py) -- calls CUDA kernel internally
from libs.slang_gaussians.interface import gsplat_strategy as gsplat_cuda  # type: ignore
from libs.slang_gaussians.interface import (
    gsplat_strategy_cuda,  # CUDA C++ binding (direct kernel access)
    gsplat_strategy_slang,  # Slang C++ binding (test-only, equivalence verification)
)


def _slang_update_gradient_buffers(
    positions: torch.Tensor,
    params_grad: torch.Tensor,
    ray_origin: torch.Tensor,
    grad_norm_accum: torch.Tensor,
    grad_norm_denom: torch.Tensor,
) -> None:
    """Call the Slang kernel via its raw C++ binding (for equivalence testing)."""
    n_gaussians = positions.size(0)
    if n_gaussians == 0:
        return
    threads_per_block = 256
    blocks = (n_gaussians + threads_per_block - 1) // threads_per_block
    gsplat_strategy_slang.update_gradient_buffers_kernel(
        (threads_per_block, 1, 1),
        (blocks, 1, 1),
        positions.contiguous(),
        params_grad.contiguous(),
        ray_origin.contiguous(),
        grad_norm_accum.contiguous(),
        grad_norm_denom.contiguous(),
    )


def pytorch_update_gradient_buffers(
    positions: torch.Tensor,
    params_grad: torch.Tensor,
    ray_origin: torch.Tensor,
    grad_norm_accum: torch.Tensor,
    grad_norm_denom: torch.Tensor,
) -> None:
    """PyTorch reference implementation of update_gradient_buffers."""
    # Compute mask for non-zero gradients
    mask = (params_grad != 0).max(dim=1)[0]

    # Compute distance to camera
    distance_to_camera = (positions[mask] - ray_origin).norm(dim=1, keepdim=True)

    # Accumulate scaled gradient norms
    grad_norm_accum[mask] += torch.norm(params_grad[mask] * distance_to_camera, dim=-1, keepdim=True) / 2
    grad_norm_denom[mask] += 1


class TestGSplatCUDAVsPyTorch(unittest.TestCase):
    """Test that CUDA GSplat kernels match PyTorch reference implementation."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.num_gaussians = 10000

        # Generate random test data
        self.positions = torch.randn(self.num_gaussians, 3, device=self.device, dtype=torch.float32)
        self.params_grad = torch.randn(self.num_gaussians, 3, device=self.device, dtype=torch.float32)
        # Zero out some gradients to simulate non-hit Gaussians
        self.params_grad[torch.rand(self.num_gaussians, device=self.device) < 0.3] = 0.0

        self.ray_origin = torch.randn(3, device=self.device, dtype=torch.float32)

    def test_update_gradient_buffers(self):
        """Test update_gradient_buffers kernel matches PyTorch."""
        # CUDA version
        grad_norm_accum_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        gsplat_cuda.update_gradient_buffers(
            positions=self.positions,
            params_grad=self.params_grad,
            ray_origin=self.ray_origin,
            grad_norm_accum=grad_norm_accum_cuda,
            grad_norm_denom=grad_norm_denom_cuda,
        )
        torch.cuda.synchronize()

        # PyTorch reference
        grad_norm_accum_pytorch = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_pytorch = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        pytorch_update_gradient_buffers(
            positions=self.positions,
            params_grad=self.params_grad,
            ray_origin=self.ray_origin,
            grad_norm_accum=grad_norm_accum_pytorch,
            grad_norm_denom=grad_norm_denom_pytorch,
        )

        self.assertTrue(
            torch.allclose(grad_norm_accum_cuda, grad_norm_accum_pytorch, atol=1e-5, rtol=1e-5),
            f"Accum max diff: {(grad_norm_accum_cuda - grad_norm_accum_pytorch).abs().max().item()}",
        )
        self.assertTrue(
            torch.equal(grad_norm_denom_cuda, grad_norm_denom_pytorch),
            f"Denom mismatch",
        )

    def test_update_gradient_buffers_all_hit(self):
        """Test with all Gaussians having non-zero gradients."""
        # All non-zero gradients
        params_grad = torch.randn(self.num_gaussians, 3, device=self.device, dtype=torch.float32)
        params_grad = torch.clamp(params_grad, min=0.01)  # Ensure non-zero

        grad_norm_accum_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        gsplat_cuda.update_gradient_buffers(
            positions=self.positions,
            params_grad=params_grad,
            ray_origin=self.ray_origin,
            grad_norm_accum=grad_norm_accum_cuda,
            grad_norm_denom=grad_norm_denom_cuda,
        )
        torch.cuda.synchronize()

        grad_norm_accum_pytorch = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_pytorch = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        pytorch_update_gradient_buffers(
            positions=self.positions,
            params_grad=params_grad,
            ray_origin=self.ray_origin,
            grad_norm_accum=grad_norm_accum_pytorch,
            grad_norm_denom=grad_norm_denom_pytorch,
        )

        self.assertTrue(
            torch.allclose(grad_norm_accum_cuda, grad_norm_accum_pytorch, atol=1e-5, rtol=1e-5),
            f"Accum max diff: {(grad_norm_accum_cuda - grad_norm_accum_pytorch).abs().max().item()}",
        )
        self.assertTrue(torch.equal(grad_norm_denom_cuda, grad_norm_denom_pytorch))

    def test_update_gradient_buffers_none_hit(self):
        """Test with no Gaussians having non-zero gradients."""
        params_grad = torch.zeros(self.num_gaussians, 3, device=self.device, dtype=torch.float32)

        grad_norm_accum_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        gsplat_cuda.update_gradient_buffers(
            positions=self.positions,
            params_grad=params_grad,
            ray_origin=self.ray_origin,
            grad_norm_accum=grad_norm_accum_cuda,
            grad_norm_denom=grad_norm_denom_cuda,
        )
        torch.cuda.synchronize()

        # Should remain zeros
        self.assertTrue(torch.all(grad_norm_accum_cuda == 0))
        self.assertTrue(torch.all(grad_norm_denom_cuda == 0))

    def test_update_gradient_buffers_accumulation(self):
        """Test that multiple calls accumulate correctly."""
        grad_norm_accum_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        grad_norm_accum_pytorch = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_pytorch = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        # Call multiple times
        for _ in range(5):
            gsplat_cuda.update_gradient_buffers(
                positions=self.positions,
                params_grad=self.params_grad,
                ray_origin=self.ray_origin,
                grad_norm_accum=grad_norm_accum_cuda,
                grad_norm_denom=grad_norm_denom_cuda,
            )
            pytorch_update_gradient_buffers(
                positions=self.positions,
                params_grad=self.params_grad,
                ray_origin=self.ray_origin,
                grad_norm_accum=grad_norm_accum_pytorch,
                grad_norm_denom=grad_norm_denom_pytorch,
            )

        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(grad_norm_accum_cuda, grad_norm_accum_pytorch, atol=1e-5, rtol=1e-5),
            f"Accum max diff: {(grad_norm_accum_cuda - grad_norm_accum_pytorch).abs().max().item()}",
        )
        self.assertTrue(torch.equal(grad_norm_denom_cuda, grad_norm_denom_pytorch))

    def test_empty_tensors(self):
        """Test that kernel handles empty tensors gracefully."""
        empty_positions = torch.empty(0, 3, device=self.device, dtype=torch.float32)
        empty_grad = torch.empty(0, 3, device=self.device, dtype=torch.float32)
        empty_accum = torch.empty(0, 1, device=self.device, dtype=torch.float32)
        empty_denom = torch.empty(0, 1, device=self.device, dtype=torch.int32)

        # Should not raise any errors
        gsplat_cuda.update_gradient_buffers(
            positions=empty_positions,
            params_grad=empty_grad,
            ray_origin=self.ray_origin,
            grad_norm_accum=empty_accum,
            grad_norm_denom=empty_denom,
        )

    def test_cuda_kernel_directly(self):
        """Test the CUDA C++ binding directly (not through Python wrapper)."""
        from libs.slang_utils.utils import div_up

        grad_norm_accum = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        # Call CUDA C++ binding directly
        gsplat_strategy_cuda.update_gradient_buffers(
            self.positions.contiguous(),
            self.params_grad.contiguous(),
            self.ray_origin.contiguous(),
            grad_norm_accum,
            grad_norm_denom,
            256,  # threads_per_block
        )
        torch.cuda.synchronize()

        # Compare against PyTorch reference
        grad_norm_accum_ref = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_ref = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)
        pytorch_update_gradient_buffers(
            self.positions, self.params_grad, self.ray_origin, grad_norm_accum_ref, grad_norm_denom_ref
        )

        self.assertTrue(
            torch.allclose(grad_norm_accum, grad_norm_accum_ref, atol=1e-5),
            f"CUDA C++ binding vs PyTorch max diff: {(grad_norm_accum - grad_norm_accum_ref).abs().max().item()}",
        )
        self.assertTrue(torch.equal(grad_norm_denom, grad_norm_denom_ref))

    def test_cuda_vs_slang_equivalence(self):
        """Test that CUDA and Slang implementations produce identical results."""
        # CUDA version
        grad_norm_accum_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_cuda = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        gsplat_cuda.update_gradient_buffers(
            positions=self.positions,
            params_grad=self.params_grad,
            ray_origin=self.ray_origin,
            grad_norm_accum=grad_norm_accum_cuda,
            grad_norm_denom=grad_norm_denom_cuda,
        )
        torch.cuda.synchronize()

        # Slang version
        grad_norm_accum_slang = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.float32)
        grad_norm_denom_slang = torch.zeros(self.num_gaussians, 1, device=self.device, dtype=torch.int32)

        _slang_update_gradient_buffers(
            positions=self.positions,
            params_grad=self.params_grad,
            ray_origin=self.ray_origin,
            grad_norm_accum=grad_norm_accum_slang,
            grad_norm_denom=grad_norm_denom_slang,
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(grad_norm_accum_cuda, grad_norm_accum_slang, atol=1e-6),
            f"CUDA vs Slang accum max diff: {(grad_norm_accum_cuda - grad_norm_accum_slang).abs().max().item()}",
        )
        self.assertTrue(
            torch.equal(grad_norm_denom_cuda, grad_norm_denom_slang),
            "CUDA vs Slang denom mismatch",
        )


if __name__ == "__main__":
    unittest.main()
