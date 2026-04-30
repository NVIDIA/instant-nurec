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

from nre.nrm.models.blocks import AttentionBlock, Mamba2Block
from nre.nrm.models.blocks.attention import CrossAttention, CrossAttentionBlock, CrossAttentionWithKVProjector


class TestAttentionBlock(unittest.TestCase):
    """Smoke tests for AttentionBlock"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 128
        self.n_heads = 8
        self.batch_size = 4
        self.seq_len = 64
        self.device = torch.device("cuda")

        self.block = AttentionBlock(input_dim=self.input_dim, n_heads=self.n_heads).to(self.device)

    def test_forward_shape_preservation(self):
        """Test that forward pass preserves input shape"""
        self.block.eval()  # Set to eval mode to avoid dropout randomness

        # Test different input shapes
        test_cases = [
            (1, 16, self.input_dim),  # Single sample, short sequence
            (4, 64, self.input_dim),  # Small batch, medium sequence
            (8, 256, self.input_dim),  # Larger batch, longer sequence
        ]

        for batch_size, seq_len, dim in test_cases:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                x = torch.randn(batch_size, seq_len, dim, device=self.device)
                output = self.block(x)

                # Check output shape matches input shape
                self.assertEqual(output.shape, x.shape)

                # Check output is not NaN or Inf
                self.assertTrue(torch.isfinite(output).all())

    def test_forward_backward_pass(self):
        """Test forward and backward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim, requires_grad=True, device=self.device)

        # Forward pass
        self.block.train()
        output = self.block(x)

        # Compute loss and backward pass
        loss = output.mean()
        loss.backward()

        # Check that input gradients exist and are finite
        self.assertIsNotNone(x.grad)
        if x.grad is not None:
            self.assertTrue(torch.isfinite(x.grad).all())

        # Check that model parameters have gradients
        params_with_grad = 0
        for name, param in self.block.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has NaN or Inf gradients")
                params_with_grad += 1

        self.assertGreater(params_with_grad, 0, "No parameters received gradients")


def _mamba2_triton_unsupported() -> bool:
    """True if current GPU is not supported by mamba_ssm Triton kernels (e.g. sm_120)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    # Triton AccelerateMatmul does not support sm_120 (Blackwell) yet
    return cap[0] >= 12


@unittest.skipIf(_mamba2_triton_unsupported(), "Mamba2 Triton kernels do not support this GPU (e.g. sm_120)")
class TestMamba2Block(unittest.TestCase):
    """Smoke tests for Mamba2Block"""

    def setUp(self):
        """Set up test fixtures"""
        self.d_model = 128
        self.batch_size = 2
        self.seq_len = 64
        self.device = torch.device("cuda")

        self.block = Mamba2Block(d_model=self.d_model, scan_type="bi").to(self.device)

    def test_forward_shape_preservation(self):
        """Test that forward pass preserves input shape"""
        self.block.eval()

        # Test different input shapes
        test_cases = [
            (1, 16, self.d_model),  # Single sample, short sequence
            (2, 32, self.d_model),  # Small batch, medium sequence
            (4, 64, self.d_model),  # Larger batch, longer sequence
        ]

        for batch_size, seq_len, dim in test_cases:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                x = torch.randn(batch_size, seq_len, dim, device=self.device)

                # Capture output to avoid debug prints in test output
                output = self.block(x)

                # Check output shape matches input shape
                self.assertEqual(output.shape, x.shape)

                # Check output is not NaN or Inf
                self.assertTrue(torch.isfinite(output).all())

    def test_forward_backward_pass(self):
        """Test forward and backward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True, device=self.device)

        # Forward pass
        self.block.train()
        output = self.block(x)

        # Compute loss and backward pass
        loss = output.mean()
        loss.backward()

        # Check that input gradients exist and are finite
        self.assertIsNotNone(x.grad)
        if x.grad is not None:
            self.assertTrue(torch.isfinite(x.grad).all())

        # Check that at least some model parameters have gradients
        params_with_grad = 0
        for name, param in self.block.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has NaN or Inf gradients")
                params_with_grad += 1

        self.assertGreater(params_with_grad, 0, "No model parameters received gradients")


class TestCrossAttention(unittest.TestCase):
    """Smoke tests for CrossAttention with both kv_projection=False and kv_projection=True"""

    def setUp(self):
        """Set up test fixtures"""
        self.dim = 128
        self.n_heads = 8
        self.batch_size = 4
        self.q_seq_len = 16  # Number of query tokens
        self.kv_seq_len = 64  # Number of key/value tokens
        self.device = torch.device("cuda")

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape for both kv_projection variants"""
        # Test different input shapes
        test_cases = [
            (2, 8, 32),  # Small batch, few query tokens, short kv sequence
            (4, 16, 64),  # Medium batch and sequences
            (8, 32, 128),  # Larger batch and sequences
        ]

        # Test both kv_projection modes
        for kv_projection in [False, True]:
            with self.subTest(kv_projection=kv_projection):
                # Create module with appropriate configuration
                if kv_projection:
                    module = CrossAttentionWithKVProjector(
                        dim=self.dim,
                        n_heads=self.n_heads,
                        bias=True,
                        norm=True,
                    ).to(self.device)
                else:
                    module = CrossAttention(
                        dim=self.dim,
                        n_heads=self.n_heads,
                        q_bias=True,
                        qk_norm=True,
                    ).to(self.device)

                module.eval()

                for batch_size, q_len, kv_len in test_cases:
                    with self.subTest(batch_size=batch_size, q_len=q_len, kv_len=kv_len):
                        q_tokens = torch.randn(batch_size, q_len, self.dim, device=self.device)

                        if kv_projection:
                            # 3D tensors (B, M, dim) for CrossAttentionWithKVProjector
                            k = torch.randn(batch_size, kv_len, self.dim, device=self.device)
                            v = torch.randn(batch_size, kv_len, self.dim, device=self.device)
                        else:
                            # Pre-projected 4D tensors for CrossAttention (no kv_projector)
                            head_dim = self.dim // self.n_heads
                            k = torch.randn(batch_size, self.n_heads, kv_len, head_dim, device=self.device)
                            v = torch.randn(batch_size, self.n_heads, kv_len, head_dim, device=self.device)

                        output = module(q_tokens, k, v)

                        # Check output shape: should match query tokens shape
                        expected_shape = (batch_size, q_len, self.dim)
                        self.assertEqual(output.shape, expected_shape)

                        # Check output is not NaN or Inf
                        self.assertTrue(torch.isfinite(output).all())

    def test_forward_backward_pass(self):
        """Test forward and backward pass for both kv_projection variants"""
        # Test both kv_projection modes
        for kv_projection in [False, True]:
            with self.subTest(kv_projection=kv_projection):
                # Create module with appropriate configuration
                if kv_projection:
                    module = CrossAttentionWithKVProjector(
                        dim=self.dim,
                        n_heads=self.n_heads,
                        bias=True,
                        norm=True,
                    ).to(self.device)
                else:
                    module = CrossAttention(
                        dim=self.dim,
                        n_heads=self.n_heads,
                        q_bias=True,
                        qk_norm=True,
                    ).to(self.device)

                q_tokens = torch.randn(
                    self.batch_size, self.q_seq_len, self.dim, requires_grad=True, device=self.device
                )

                if kv_projection:
                    # 3D tensors (B, M, dim) for CrossAttentionWithKVProjector
                    k = torch.randn(self.batch_size, self.kv_seq_len, self.dim, requires_grad=True, device=self.device)
                    v = torch.randn(self.batch_size, self.kv_seq_len, self.dim, requires_grad=True, device=self.device)
                else:
                    # Pre-projected 4D tensors for kv_projection=False
                    head_dim = self.dim // self.n_heads
                    k = torch.randn(
                        self.batch_size, self.n_heads, self.kv_seq_len, head_dim, requires_grad=True, device=self.device
                    )
                    v = torch.randn(
                        self.batch_size, self.n_heads, self.kv_seq_len, head_dim, requires_grad=True, device=self.device
                    )

                # Forward pass
                module.train()
                output = module(q_tokens, k, v)

                # Compute loss and backward pass
                loss = output.mean()
                loss.backward()

                # Check that input gradients exist and are finite
                self.assertIsNotNone(q_tokens.grad)
                self.assertIsNotNone(k.grad)
                self.assertIsNotNone(v.grad)
                if q_tokens.grad is not None:
                    self.assertTrue(torch.isfinite(q_tokens.grad).all())
                if k.grad is not None:
                    self.assertTrue(torch.isfinite(k.grad).all())
                if v.grad is not None:
                    self.assertTrue(torch.isfinite(v.grad).all())

                # Check that model parameters have gradients
                params_with_grad = 0
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
                        if param.grad is not None:
                            self.assertTrue(
                                torch.isfinite(param.grad).all(), f"Parameter {name} has NaN or Inf gradients"
                            )
                        params_with_grad += 1

                self.assertGreater(params_with_grad, 0, "No parameters received gradients")


class TestCrossAttentionBlock(unittest.TestCase):
    """Smoke tests for CrossAttentionBlock"""

    def setUp(self):
        """Set up test fixtures"""
        self.dim = 128
        self.n_heads = 8
        self.batch_size = 4
        self.q_seq_len = 16  # Number of query tokens
        self.kv_seq_len = 64  # Number of key/value tokens
        self.device = torch.device("cuda")

        self.block = CrossAttentionBlock(dim=self.dim, n_heads=self.n_heads, qkv_bias=True, qk_norm=True).to(
            self.device
        )

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape"""
        self.block.eval()

        # Test different input shapes
        test_cases = [
            (2, 8, 32),  # Small batch, few query tokens, short kv sequence
            (4, 16, 64),  # Medium batch and sequences
            (8, 32, 128),  # Larger batch and sequences
        ]

        for batch_size, q_len, kv_len in test_cases:
            with self.subTest(batch_size=batch_size, q_len=q_len, kv_len=kv_len):
                head_dim = self.dim // self.n_heads
                q_tokens = torch.randn(batch_size, q_len, self.dim, device=self.device)
                k = torch.randn(batch_size, self.n_heads, kv_len, head_dim, device=self.device)
                v = torch.randn(batch_size, self.n_heads, kv_len, head_dim, device=self.device)

                output = self.block(q_tokens, k, v)

                # Check output shape: should match query tokens shape
                expected_shape = (batch_size, q_len, self.dim)
                self.assertEqual(output.shape, expected_shape)

                # Check output is not NaN or Inf
                self.assertTrue(torch.isfinite(output).all())

    def test_forward_backward_pass(self):
        """Test forward and backward pass"""
        head_dim = self.dim // self.n_heads
        q_tokens = torch.randn(self.batch_size, self.q_seq_len, self.dim, requires_grad=True, device=self.device)
        k = torch.randn(
            self.batch_size, self.n_heads, self.kv_seq_len, head_dim, requires_grad=True, device=self.device
        )
        v = torch.randn(
            self.batch_size, self.n_heads, self.kv_seq_len, head_dim, requires_grad=True, device=self.device
        )

        # Forward pass
        self.block.train()
        output = self.block(q_tokens, k, v)

        # Compute loss and backward pass
        loss = output.mean()
        loss.backward()

        # Check that input gradients exist and are finite
        self.assertIsNotNone(q_tokens.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        if q_tokens.grad is not None:
            self.assertTrue(torch.isfinite(q_tokens.grad).all())
        if k.grad is not None:
            self.assertTrue(torch.isfinite(k.grad).all())
        if v.grad is not None:
            self.assertTrue(torch.isfinite(v.grad).all())

        # Check that model parameters have gradients
        params_with_grad = 0
        for name, param in self.block.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has NaN or Inf gradients")
                params_with_grad += 1

        self.assertGreater(params_with_grad, 0, "No parameters received gradients")


if __name__ == "__main__":
    unittest.main()
