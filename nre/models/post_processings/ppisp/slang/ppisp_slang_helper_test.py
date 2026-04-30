# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nre.models.post_processings.ppisp import PiecewisePowerFunction
from nre.models.post_processings.ppisp.ppisp import ColorCorrection
from nre.models.post_processings.ppisp.slang import (  # type: ignore # pycena: skip
    libppisp_slang_helper_test_cc as ppisp_test_slang,
)


def test_crf_curve_points_matches_python():
    torch.manual_seed(0)
    crf_params = torch.randn(1, 3, 7, device="cuda")
    camera_idx = 0
    channel_idx = 1
    slang_out = _call_slang_crf_curve_points(crf_params, camera_idx, channel_idx)
    raw_params = PiecewisePowerFunction.RawParams(crf_params[camera_idx, channel_idx])
    curve_points = PiecewisePowerFunction.crf_curve_points(raw_params)
    torch_out = torch.stack(
        [
            curve_points.x0,
            curve_points.y0,
            curve_points.slope_p0,
            curve_points.y0_pre_gamma,
            curve_points.slope_line,
            curve_points.gamma,
            curve_points.x1,
            curve_points.y1,
            curve_points.slope_p1,
            curve_points.shoulder_x,
            curve_points.shoulder_y,
        ]
    )
    torch.testing.assert_close(slang_out, torch_out, rtol=1e-5, atol=1e-6)


# =============================================================================
# Tests for apply_color_correction_rg helper function
# Uses ppisp_slang_helper.slang which exposes the helper for testing
# =============================================================================


def _call_slang_apply_color_correction_rg_single(H: torch.Tensor, rg: torch.Tensor) -> torch.Tensor:
    """
    Call the Slang test kernel for apply_color_correction_rg on a single (H, rg) pair.

    Same interface as the Slang helper: (float3x3 H, float2 rg) -> float2

    Args:
        H: Homography matrix, shape (3, 3), float32
        rg: Input RG chromaticity, shape (2,), float32

    Returns:
        Output RG chromaticity, shape (2,)
    """
    assert H.shape == (3, 3), f"Expected H shape (3, 3), got {H.shape}"
    assert rg.shape == (2,), f"Expected rg shape (2,), got {rg.shape}"

    H = H.contiguous().float().cuda()
    rg = rg.contiguous().float().cuda()
    out_rg = torch.empty(2, dtype=torch.float32, device="cuda")

    ppisp_test_slang.ppisp_test_apply_color_correction_rg(
        (1, 1, 1),
        (1, 1, 1),
        (H, (H,)),
        (rg, (rg,)),
        (out_rg, (out_rg,)),
    )

    return out_rg


def _call_slang_apply_color_correction_rg_single_bwd(
    H: torch.Tensor,
    rg: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call the backward pass of the Slang test kernel for a single (H, rg) pair.

    Args:
        H: Homography matrix, shape (3, 3), float32
        rg: Input RG chromaticity, shape (2,), float32
        grad_output: Gradient w.r.t. output, shape (2,)

    Returns:
        Tuple of (grad_H, grad_rg) with shapes (3, 3) and (2,)
    """
    H = H.contiguous().float().cuda()
    rg = rg.contiguous().float().cuda()
    grad_output = grad_output.contiguous().float().cuda()
    out_rg = torch.empty(2, dtype=torch.float32, device="cuda")

    # Forward pass to populate out_rg
    ppisp_test_slang.ppisp_test_apply_color_correction_rg(
        (1, 1, 1),
        (1, 1, 1),
        (H, (H,)),
        (rg, (rg,)),
        (out_rg, (out_rg,)),
    )

    # Allocate gradient tensors
    grad_H = torch.zeros(3, 3, dtype=torch.float32, device="cuda")
    grad_rg = torch.zeros(2, dtype=torch.float32, device="cuda")

    # Backward pass
    ppisp_test_slang.ppisp_test_apply_color_correction_rg_bwd_diff(
        (1, 1, 1),
        (1, 1, 1),
        (H, (grad_H,)),
        (rg, (grad_rg,)),
        (out_rg, (grad_output,)),
    )

    return grad_H, grad_rg


def _call_slang_apply_color_correction_rg(h: torch.Tensor, rg: torch.Tensor) -> torch.Tensor:
    """
    Call the Slang test kernel for apply_color_correction_rg on batched inputs.

    Loops over all (batch, point) pairs calling the single-pair kernel.

    Args:
        h: Homography matrices, shape (B, 3, 3), float32
        rg: Source chromaticities, shape (N, 2), float32

    Returns:
        Color corrected chromaticities, shape (B, N, 2)
    """
    assert h.ndim == 3 and h.shape[1] == 3 and h.shape[2] == 3, f"Expected h shape (B, 3, 3), got {h.shape}"
    assert rg.ndim == 2 and rg.shape[1] == 2, f"Expected rg shape (N, 2), got {rg.shape}"

    num_batches = h.shape[0]
    num_points = rg.shape[0]

    out_rg = torch.empty(num_batches, num_points, 2, dtype=torch.float32, device="cuda")

    for b in range(num_batches):
        for n in range(num_points):
            out_rg[b, n] = _call_slang_apply_color_correction_rg_single(h[b], rg[n])

    return out_rg


def _call_slang_apply_color_correction_rg_bwd(
    h: torch.Tensor,
    rg: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call the backward pass of the Slang test kernel on batched inputs.

    Args:
        h: Homography matrices, shape (B, 3, 3), float32
        rg: Source chromaticities, shape (N, 2), float32
        grad_output: Gradient w.r.t. output, shape (B, N, 2)

    Returns:
        Tuple of (grad_h, grad_rg) with shapes (B, 3, 3) and (N, 2)
    """
    num_batches = h.shape[0]
    num_points = rg.shape[0]

    grad_h = torch.zeros_like(h)
    grad_rg = torch.zeros_like(rg)

    for b in range(num_batches):
        for n in range(num_points):
            g_h, g_rg = _call_slang_apply_color_correction_rg_single_bwd(h[b], rg[n], grad_output[b, n])
            grad_h[b] += g_h
            grad_rg[n] += g_rg

    return grad_h, grad_rg


def _call_slang_crf_curve_points(
    crf_params: torch.Tensor,
    camera_idx: int,
    channel_idx: int,
) -> torch.Tensor:
    crf_params = crf_params.contiguous().float().cuda()
    out = torch.empty(11, dtype=torch.float32, device="cuda")
    ppisp_test_slang.ppisp_test_crf_curve_points(
        (1, 1, 1),
        (1, 1, 1),
        camera_idx,
        channel_idx,
        (crf_params, (crf_params,)),
        (out, (out,)),
    )
    return out


def _call_slang_compute_curve_points(raw_params: torch.Tensor) -> torch.Tensor:
    assert raw_params.shape == (7,), f"Expected raw_params shape (7,), got {raw_params.shape}"
    raw_params = raw_params.contiguous().float().cuda()
    out_curve = torch.empty(11, dtype=torch.float32, device="cuda")
    ppisp_test_slang.ppisp_test_compute_curve_points(
        (1, 1, 1),
        (1, 1, 1),
        (raw_params, (raw_params,)),
        (out_curve, (out_curve,)),
    )
    return out_curve


def _call_slang_compute_curve_points_bwd(raw_params: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    raw_params = raw_params.contiguous().float().cuda()
    grad_output = grad_output.contiguous().float().cuda()
    out_curve = torch.empty(11, dtype=torch.float32, device="cuda")
    grad_raw_params = torch.zeros_like(raw_params)
    ppisp_test_slang.ppisp_test_compute_curve_points(
        (1, 1, 1),
        (1, 1, 1),
        (raw_params, (raw_params,)),
        (out_curve, (out_curve,)),
    )
    ppisp_test_slang.ppisp_test_compute_curve_points_bwd_diff(
        (1, 1, 1),
        (1, 1, 1),
        (raw_params, (grad_raw_params,)),
        (out_curve, (grad_output,)),
    )
    return grad_raw_params


def _call_slang_crf_curve_points_bwd(
    crf_params: torch.Tensor,
    camera_idx: int,
    channel_idx: int,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    crf_params = crf_params.contiguous().float().cuda()
    grad_output = grad_output.contiguous().float().cuda()
    out = torch.empty(11, dtype=torch.float32, device="cuda")
    grad_crf_params = torch.zeros_like(crf_params)
    ppisp_test_slang.ppisp_test_crf_curve_points(
        (1, 1, 1),
        (1, 1, 1),
        camera_idx,
        channel_idx,
        (crf_params, (crf_params,)),
        (out, (out,)),
    )
    ppisp_test_slang.ppisp_test_crf_curve_points_bwd_diff(
        (1, 1, 1),
        (1, 1, 1),
        camera_idx,
        channel_idx,
        (crf_params, (grad_crf_params,)),
        (out, (grad_output,)),
    )
    return grad_crf_params


# =============================================================================
# Tests for PiecewisePowerFunction helper (inverse_ppf)
# Uses ppisp_slang_helper.slang which exposes the helper for testing
# =============================================================================


def _python_inverse_ppf(raw_params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    raw_params_accessor = PiecewisePowerFunction.RawParams(raw_params)
    curve_points = PiecewisePowerFunction.crf_curve_points(raw_params_accessor)
    return PiecewisePowerFunction.inverse(curve_points, y)


def _call_slang_inverse_ppf_single(raw_params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert raw_params.shape == (7,), f"Expected raw_params shape (7,), got {raw_params.shape}"
    assert y.shape == (1,), f"Expected y shape (1,), got {y.shape}"

    raw_params = raw_params.contiguous().float().cuda()
    y = y.contiguous().float().cuda()
    out_x = torch.empty(1, dtype=torch.float32, device="cuda")

    ppisp_test_slang.ppisp_test_inverse_ppf(
        (1, 1, 1),
        (1, 1, 1),
        (raw_params, (raw_params,)),
        (y, (y,)),
        (out_x, (out_x,)),
    )

    return out_x


def _call_slang_inverse_ppf_single_bwd(
    raw_params: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_params = raw_params.contiguous().float().cuda()
    y = y.contiguous().float().cuda()
    grad_output = grad_output.contiguous().float().cuda()
    out_x = torch.empty(1, dtype=torch.float32, device="cuda")

    ppisp_test_slang.ppisp_test_inverse_ppf(
        (1, 1, 1),
        (1, 1, 1),
        (raw_params, (raw_params,)),
        (y, (y,)),
        (out_x, (out_x,)),
    )

    grad_raw_params = torch.zeros_like(raw_params)
    grad_y = torch.zeros_like(y)

    ppisp_test_slang.ppisp_test_inverse_ppf_bwd_diff(
        (1, 1, 1),
        (1, 1, 1),
        (raw_params, (grad_raw_params,)),
        (y, (grad_y,)),
        (out_x, (grad_output,)),
    )

    return grad_raw_params, grad_y


class TestApplyColorCorrectionRg:
    """
    Comprehensive tests for apply_color_correction_rg helper function.

    Tests the Slang helper via ppisp_test_apply_color_correction_rg kernel
    against Python ColorCorrection.apply_color_correction_rg.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"
        torch.manual_seed(42)

    # =========================================================================
    # Basic correctness tests vs Python reference
    # =========================================================================

    def test_matches_python_reference(self):
        """Test that Slang helper matches Python implementation."""
        num_batches = 16
        num_points = 4

        # Random homographies close to identity
        identity = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).clone()
        perturbation = (torch.rand(num_batches, 3, 3, device=self.device) - 0.5) * 0.2
        h = identity + perturbation

        rg = ColorCorrection.get_default_source_chroms(self.device)

        # Python reference
        expected = ColorCorrection.apply_color_correction_rg(rg, h)

        # Slang kernel
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_random_homographies_various_sizes(self):
        """Test with various random homographies and sizes."""
        for seed in range(5):
            torch.manual_seed(seed)
            num_batches = torch.randint(1, 50, (1,)).item()
            num_points = torch.randint(1, 20, (1,)).item()

            identity = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).clone()
            perturbation = (torch.rand(num_batches, 3, 3, device=self.device) - 0.5) * 0.3
            h = identity + perturbation

            rg = torch.rand(num_points, 2, device=self.device)

            expected = ColorCorrection.apply_color_correction_rg(rg, h)
            result = _call_slang_apply_color_correction_rg(h, rg)

            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Identity and special transformations
    # =========================================================================

    def test_identity_homography(self):
        """Test that identity homography returns unchanged chromaticities."""
        num_batches = 4
        h = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).contiguous()
        rg = ColorCorrection.get_default_source_chroms(self.device)

        result = _call_slang_apply_color_correction_rg(h, rg)

        for b in range(num_batches):
            torch.testing.assert_close(result[b], rg, rtol=1e-5, atol=1e-5)

    def test_translation_homography(self):
        """Test translation via homography."""
        h = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # identity
                [[1, 0, 0.1], [0, 1, -0.05], [0, 0, 1]],  # translate
            ],
            dtype=torch.float32,
            device=self.device,
        )

        rg = ColorCorrection.get_default_source_chroms(self.device)

        result = _call_slang_apply_color_correction_rg(h, rg)

        # Frame 0: unchanged
        torch.testing.assert_close(result[0], rg, rtol=1e-5, atol=1e-5)

        # Frame 1: translated
        expected_translated = rg.clone()
        expected_translated[:, 0] += 0.1
        expected_translated[:, 1] -= 0.05
        torch.testing.assert_close(result[1], expected_translated, rtol=1e-5, atol=1e-5)

    def test_scaling_homography(self):
        """Test scaling via homography."""
        h = torch.tensor(
            [
                [[1.1, 0, 0], [0, 1, 0], [0, 0, 1]],  # scale x by 1.1
                [[1, 0, 0], [0, 0.9, 0], [0, 0, 1]],  # scale y by 0.9
                [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1]],  # scale both by 1.2
            ],
            dtype=torch.float32,
            device=self.device,
        )

        rg = torch.tensor([[0.3, 0.3], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6]], dtype=torch.float32, device=self.device)

        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result[0, :, 0], rg[:, 0] * 1.1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result[0, :, 1], rg[:, 1], rtol=1e-5, atol=1e-5)

        torch.testing.assert_close(result[1, :, 0], rg[:, 0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result[1, :, 1], rg[:, 1] * 0.9, rtol=1e-5, atol=1e-5)

        torch.testing.assert_close(result[2], rg * 1.2, rtol=1e-5, atol=1e-5)

    def test_homography_layout_correctness(self):
        """Test that homography layout [row0, row1, row2] is correct."""
        h = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 1]]], dtype=torch.float32, device=self.device)
        rg = torch.tensor([[0.1, 0.2]], device=self.device)

        result = _call_slang_apply_color_correction_rg(h, rg)

        # Manual: p=[0.1,0.2,1], t=H@p=[3.5,7.4,3.3], out=t[:2]/(t[2]+eps)
        t = torch.tensor([3.5, 7.4, 3.3], device=self.device)
        expected = t[:2] / (t[2] + 1e-5)

        torch.testing.assert_close(result[0, 0], expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_single_batch(self):
        """Test with single batch."""
        h = torch.tensor([[[1.1, 0.05, 0.02], [-0.03, 0.95, -0.01], [0.01, -0.02, 1]]], device=self.device)
        rg = ColorCorrection.get_default_source_chroms(self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_single_point(self):
        """Test with single rg point."""
        num_batches = 10
        identity = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).clone()
        perturbation = (torch.rand(num_batches, 3, 3, device=self.device) - 0.5) * 0.2
        h = identity + perturbation

        rg = torch.tensor([[0.5, 0.5]], device=self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_large_batch(self):
        """Test with large batch size."""
        num_batches = 2048
        identity = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).clone()
        perturbation = (torch.rand(num_batches, 3, 3, device=self.device) - 0.5) * 0.2
        h = identity + perturbation

        rg = ColorCorrection.get_default_source_chroms(self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_many_points(self):
        """Test with many rg points."""
        num_batches = 10
        num_points = 100

        identity = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).clone()
        perturbation = (torch.rand(num_batches, 3, 3, device=self.device) - 0.5) * 0.2
        h = identity + perturbation

        rg = torch.rand(num_points, 2, device=self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    # =========================================================================
    # Corner chromaticities (default source points)
    # =========================================================================

    def test_blue_chromaticity(self):
        """Test with blue chromaticity (0, 0)."""
        h = torch.eye(3, device=self.device).unsqueeze(0) + (torch.rand(1, 3, 3, device=self.device) - 0.5) * 0.2
        rg = torch.tensor([[0.0, 0.0]], device=self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_red_chromaticity(self):
        """Test with red chromaticity (1, 0)."""
        h = torch.eye(3, device=self.device).unsqueeze(0) + (torch.rand(1, 3, 3, device=self.device) - 0.5) * 0.2
        rg = torch.tensor([[1.0, 0.0]], device=self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_green_chromaticity(self):
        """Test with green chromaticity (0, 1)."""
        h = torch.eye(3, device=self.device).unsqueeze(0) + (torch.rand(1, 3, 3, device=self.device) - 0.5) * 0.2
        rg = torch.tensor([[0.0, 1.0]], device=self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_gray_chromaticity(self):
        """Test with neutral gray chromaticity (1/3, 1/3)."""
        h = torch.eye(3, device=self.device).unsqueeze(0) + (torch.rand(1, 3, 3, device=self.device) - 0.5) * 0.2
        rg = torch.tensor([[1 / 3, 1 / 3]], device=self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    # =========================================================================
    # Numerical stability tests
    # =========================================================================

    def test_near_singular_homography(self):
        """Test with near-singular homography (small projective component)."""
        h = torch.tensor([[[1, 0, 0], [0, 1, 0], [0.01, 0.01, 1]]], device=self.device)
        rg = torch.rand(4, 2, device=self.device)

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_negative_z_coordinate(self):
        """Test when t.z < 0 (point projected 'behind' camera)."""
        rg = ColorCorrection.get_default_source_chroms(self.device)

        h = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [-3.0, 0, 1]],  # t.z < 0 for red (1,0)
                [[1, 0, 0], [0, 1, 0], [0, -3.0, 1]],  # t.z < 0 for green (0,1)
                [[1, 0, 0], [0, 1, 0], [-2.0, -2.0, 1]],  # t.z < 0 for gray
            ],
            device=self.device,
        )

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_near_zero_z_coordinate(self):
        """Test when t.z ≈ 0 (epsilon dominates division)."""
        rg = ColorCorrection.get_default_source_chroms(self.device)

        h = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [-1.0, 0, 1]],  # t.z = 0 for red (1,0)
                [[1, 0, 0], [0, 1, 0], [0, -1.0, 1]],  # t.z = 0 for green (0,1)
                [[1, 0, 0], [0, 1, 0], [-1.5, -1.5, 1]],  # t.z ≈ 0 for gray
            ],
            device=self.device,
        )

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        # Relaxed tolerance since epsilon dominates
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        assert torch.isfinite(result).all(), "Result contains inf or nan"

    def test_extreme_values(self):
        """Test with extreme homography values."""
        h = torch.tensor(
            [
                [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]],  # 2x scaling
                [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],  # 0.5x scaling
                [[1, 0.5, 0], [-0.5, 1, 0], [0, 0, 1]],  # rotation-like
            ],
            device=self.device,
        )

        rg = torch.rand(4, 2, device=self.device) * 0.5 + 0.25

        expected = ColorCorrection.apply_color_correction_rg(rg, h)
        result = _call_slang_apply_color_correction_rg(h, rg)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestApplyColorCorrectionRgBackward:
    """
    Backward pass tests for apply_color_correction_rg helper function.

    Tests gradients computed by Slang autodiff match PyTorch autograd.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"
        torch.manual_seed(42)

    def _compare_gradients(self, h: torch.Tensor, rg: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4):
        """Helper to compare Slang and Python gradients."""
        # Python forward + backward
        h_py = h.clone().detach().requires_grad_(True)
        rg_py = rg.clone().detach().requires_grad_(True)

        out_py = ColorCorrection.apply_color_correction_rg(rg_py, h_py)
        loss_py = out_py.sum()
        loss_py.backward()

        grad_h_py = h_py.grad
        grad_rg_py = rg_py.grad

        # Slang backward
        grad_output = torch.ones_like(out_py)
        grad_h_slang, grad_rg_slang = _call_slang_apply_color_correction_rg_bwd(h, rg, grad_output)

        torch.testing.assert_close(grad_h_slang, grad_h_py, rtol=rtol, atol=atol, msg="h gradients mismatch")
        torch.testing.assert_close(grad_rg_slang, grad_rg_py, rtol=rtol, atol=atol, msg="rg gradients mismatch")

        return grad_h_slang, grad_rg_slang

    # =========================================================================
    # Basic backward tests
    # =========================================================================

    def test_backward_random_homography(self):
        """Test backward with random homography."""
        num_batches = 32
        identity = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).clone()
        perturbation = (torch.rand(num_batches, 3, 3, device=self.device) - 0.5) * 0.2
        h = identity + perturbation

        rg = ColorCorrection.get_default_source_chroms(self.device)

        self._compare_gradients(h, rg)

    def test_backward_identity_homography(self):
        """Test backward with identity homography."""
        num_batches = 4
        h = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).contiguous()
        rg = ColorCorrection.get_default_source_chroms(self.device)

        self._compare_gradients(h, rg)

    def test_backward_translation_homography(self):
        """Test backward with translation homography."""
        h = torch.tensor(
            [
                [[1, 0, 0.1], [0, 1, 0.05], [0, 0, 1]],
                [[1, 0, -0.1], [0, 1, -0.05], [0, 0, 1]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        rg = ColorCorrection.get_default_source_chroms(self.device)

        self._compare_gradients(h, rg)

    def test_backward_scaling_homography(self):
        """Test backward with scaling homography."""
        h = torch.tensor(
            [
                [[1.1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 0.9, 0], [0, 0, 1]],
                [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        rg = ColorCorrection.get_default_source_chroms(self.device)

        self._compare_gradients(h, rg)

    # =========================================================================
    # Large batch backward tests
    # =========================================================================

    def test_backward_large_batch(self):
        """Test backward with large batch."""
        num_batches = 2048
        identity = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).clone()
        perturbation = (torch.rand(num_batches, 3, 3, device=self.device) - 0.5) * 0.2
        h = identity + perturbation

        rg = ColorCorrection.get_default_source_chroms(self.device)

        self._compare_gradients(h, rg)

    # =========================================================================
    # Corner chromaticities backward tests
    # =========================================================================

    def test_backward_blue_chromaticity(self):
        """Test backward with blue chromaticity (0, 0)."""
        h = torch.eye(3, device=self.device).unsqueeze(0) + (torch.rand(1, 3, 3, device=self.device) - 0.5) * 0.2
        rg = torch.tensor([[0.0, 0.0]], device=self.device)

        self._compare_gradients(h, rg)

    def test_backward_red_chromaticity(self):
        """Test backward with red chromaticity (1, 0)."""
        h = torch.eye(3, device=self.device).unsqueeze(0) + (torch.rand(1, 3, 3, device=self.device) - 0.5) * 0.2
        rg = torch.tensor([[1.0, 0.0]], device=self.device)

        self._compare_gradients(h, rg)

    def test_backward_green_chromaticity(self):
        """Test backward with green chromaticity (0, 1)."""
        h = torch.eye(3, device=self.device).unsqueeze(0) + (torch.rand(1, 3, 3, device=self.device) - 0.5) * 0.2
        rg = torch.tensor([[0.0, 1.0]], device=self.device)

        self._compare_gradients(h, rg)

    def test_backward_gray_chromaticity(self):
        """Test backward with gray chromaticity (1/3, 1/3)."""
        h = torch.eye(3, device=self.device).unsqueeze(0) + (torch.rand(1, 3, 3, device=self.device) - 0.5) * 0.2
        rg = torch.tensor([[1 / 3, 1 / 3]], device=self.device)

        self._compare_gradients(h, rg)

    # =========================================================================
    # Numerical stability backward tests
    # =========================================================================

    def test_backward_near_singular_homography(self):
        """Test backward with near-singular homography."""
        h = torch.tensor([[[1, 0, 0], [0, 1, 0], [0.01, 0.01, 1]]], device=self.device)
        rg = torch.rand(4, 2, device=self.device)

        self._compare_gradients(h, rg)

    def test_backward_negative_z_coordinate(self):
        """Test backward when t.z < 0."""
        rg = ColorCorrection.get_default_source_chroms(self.device)

        h = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [-3.0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, -3.0, 1]],
                [[1, 0, 0], [0, 1, 0], [-2.0, -2.0, 1]],
            ],
            device=self.device,
        )

        self._compare_gradients(h, rg, rtol=1e-3, atol=1e-3)

    def test_backward_near_zero_z_coordinate(self):
        """Test backward when t.z ≈ 0."""
        rg = ColorCorrection.get_default_source_chroms(self.device)

        h = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [-1.0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, -1.0, 1]],
                [[1, 0, 0], [0, 1, 0], [-1.5, -1.5, 1]],
            ],
            device=self.device,
        )

        self._compare_gradients(h, rg, rtol=1e-3, atol=1e-3)

        # Verify gradients are finite
        grad_h, grad_rg = _call_slang_apply_color_correction_rg_bwd(h, rg, torch.ones(3, 4, 2, device=self.device))
        assert torch.isfinite(grad_h).all(), "grad_h contains inf or nan"
        assert torch.isfinite(grad_rg).all(), "grad_rg contains inf or nan"

    def test_backward_extreme_values(self):
        """Test backward with extreme homography values."""
        h = torch.tensor(
            [
                [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]],
                [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]],
                [[1, 0.5, 0], [-0.5, 1, 0], [0, 0, 1]],
            ],
            device=self.device,
        )

        rg = torch.rand(4, 2, device=self.device) * 0.5 + 0.25

        self._compare_gradients(h, rg)

    # =========================================================================
    # Gradient correctness tests
    # =========================================================================

    def test_backward_gradient_flow_to_h(self):
        """Test that gradients flow correctly to h."""
        h = torch.tensor([[[1.1, 0.05, 0.02], [-0.03, 0.95, -0.01], [0.01, -0.02, 1]]], device=self.device)
        rg = ColorCorrection.get_default_source_chroms(self.device)

        grad_output = torch.ones(1, 4, 2, device=self.device)
        grad_h, _ = _call_slang_apply_color_correction_rg_bwd(h, rg, grad_output)

        assert (grad_h != 0).any(), "Expected non-zero gradients for h"

    def test_backward_gradient_flow_to_rg(self):
        """Test that gradients flow correctly to rg."""
        h = torch.tensor([[[1.1, 0.05, 0.02], [-0.03, 0.95, -0.01], [0.01, -0.02, 1]]], device=self.device)
        rg = ColorCorrection.get_default_source_chroms(self.device)

        grad_output = torch.ones(1, 4, 2, device=self.device)
        _, grad_rg = _call_slang_apply_color_correction_rg_bwd(h, rg, grad_output)

        assert (grad_rg != 0).any(), "Expected non-zero gradients for rg"

    def test_backward_zero_grad_output(self):
        """Test backward with zero gradient output."""
        h = torch.tensor([[[1.1, 0.05, 0.02], [-0.03, 0.95, -0.01], [0.01, -0.02, 1]]], device=self.device)
        rg = ColorCorrection.get_default_source_chroms(self.device)

        grad_output = torch.zeros(1, 4, 2, device=self.device)
        grad_h, grad_rg = _call_slang_apply_color_correction_rg_bwd(h, rg, grad_output)

        torch.testing.assert_close(grad_h, torch.zeros_like(grad_h))
        torch.testing.assert_close(grad_rg, torch.zeros_like(grad_rg))

    def test_backward_multiple_random_cases(self):
        """Test backward with multiple random cases."""
        for seed in range(10):
            torch.manual_seed(seed)
            num_batches = torch.randint(1, 20, (1,)).item()
            num_points = torch.randint(1, 10, (1,)).item()

            identity = torch.eye(3, device=self.device).unsqueeze(0).expand(num_batches, -1, -1).clone()
            perturbation = (torch.rand(num_batches, 3, 3, device=self.device) - 0.5) * 0.3
            h = identity + perturbation

            rg = torch.rand(num_points, 2, device=self.device)

            self._compare_gradients(h, rg)


class TestPiecewisePowerFunctionInverseForward:
    """
    Forward tests for inverse_ppf helper.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"
        torch.manual_seed(42)

    def test_inverse_ppf_matches_python_random(self):
        """Random raw params and y values should match Python reference."""
        for seed in range(5):
            torch.manual_seed(seed + 10)
            raw_params = torch.randn(7, device=self.device) * 0.5
            y = torch.rand(1, device=self.device) * 1.5 - 0.2

            expected = _python_inverse_ppf(raw_params, y)
            result = _call_slang_inverse_ppf_single(raw_params, y)

            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_inverse_ppf_region_boundaries(self):
        """Check inverse_ppf behavior across curve regions."""
        raw_params = PiecewisePowerFunction.get_crf_raw_param_values().data.to(self.device)
        curve_points = PiecewisePowerFunction.crf_curve_points(PiecewisePowerFunction.RawParams(raw_params))

        y_values = torch.tensor(
            [
                -0.1,
                0.5 * curve_points.y0.item(),
                0.5 * (curve_points.y0 + curve_points.y1).item(),
                0.5 * (curve_points.y1 + 1.0),
                1.1,
            ],
            device=self.device,
        )

        for y in y_values:
            y_val = y.view(1)
            expected = _python_inverse_ppf(raw_params, y_val)
            result = _call_slang_inverse_ppf_single(raw_params, y_val)
            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_inverse_ppf_roundtrip_with_python_forward(self):
        """inverse_ppf should match Python inverse for Python forward outputs."""
        for seed in range(5):
            torch.manual_seed(seed + 100)
            raw_params = torch.randn(7, device=self.device) * 0.5
            curve_points = PiecewisePowerFunction.crf_curve_points(PiecewisePowerFunction.RawParams(raw_params))
            max_x = (curve_points.shoulder_x * 1.1).clamp(min=0.1).item()
            x = torch.rand(1, device=self.device) * max_x - 0.1

            y = PiecewisePowerFunction.apply_ppf(curve_points, x)
            result = _call_slang_inverse_ppf_single(raw_params, y)
            expected = _python_inverse_ppf(raw_params, y)
            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestPiecewisePowerFunctionInverseBackward:
    """
    Backward tests for inverse_ppf helper.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"
        torch.manual_seed(42)

    def _compare_inverse_ppf_gradients(self, raw_params: torch.Tensor, y: torch.Tensor, rtol=1e-4, atol=1e-4):
        raw_params_py = raw_params.clone().detach().requires_grad_(True)
        y_py = y.clone().detach().requires_grad_(True)

        out_py = _python_inverse_ppf(raw_params_py, y_py)
        out_py.sum().backward()

        grad_raw_py = raw_params_py.grad
        grad_y_py = y_py.grad

        grad_output = torch.ones_like(out_py)
        grad_raw_slang, grad_y_slang = _call_slang_inverse_ppf_single_bwd(raw_params, y, grad_output)

        torch.testing.assert_close(
            grad_raw_slang, grad_raw_py, rtol=rtol, atol=atol, msg="raw params gradients mismatch"
        )
        torch.testing.assert_close(grad_y_slang, grad_y_py, rtol=rtol, atol=atol, msg="y gradients mismatch")

    def test_inverse_ppf_backward_matches_python_random(self):
        """Random gradients should match Python reference for inverse_ppf."""
        for seed in range(5):
            torch.manual_seed(seed + 10)
            raw_params = torch.randn(7, device=self.device) * 0.5
            y = torch.rand(1, device=self.device) * 1.5 - 0.2
            self._compare_inverse_ppf_gradients(raw_params, y)

    def test_backward_zero_grad_output_inverse(self):
        """Zero grad output should produce zero gradients for inverse_ppf."""
        raw_params = torch.randn(7, device=self.device) * 0.5
        y = torch.rand(1, device=self.device)

        grad_output = torch.zeros(1, device=self.device)
        grad_raw, grad_y = _call_slang_inverse_ppf_single_bwd(raw_params, y, grad_output)

        torch.testing.assert_close(grad_raw, torch.zeros_like(grad_raw))
        torch.testing.assert_close(grad_y, torch.zeros_like(grad_y))

    def test_backward_gradient_flow_inverse(self):
        """Gradients should flow to both raw params and y for inverse_ppf."""
        raw_params = torch.randn(7, device=self.device) * 0.5
        y = torch.tensor([0.5], device=self.device)

        grad_output = torch.ones(1, device=self.device)
        grad_raw, grad_y = _call_slang_inverse_ppf_single_bwd(raw_params, y, grad_output)

        assert (grad_raw != 0).any(), "Expected non-zero gradients for raw_params"
        assert (grad_y != 0).any(), "Expected non-zero gradients for y"

    def test_backward_finite_gradients_boundary_values(self):
        """Gradients should remain finite near key y boundaries."""
        raw_params = PiecewisePowerFunction.get_crf_raw_param_values().data.to(self.device)
        curve_points = PiecewisePowerFunction.crf_curve_points(PiecewisePowerFunction.RawParams(raw_params))

        y_values = torch.tensor(
            [
                curve_points.y0 * 0.999,
                curve_points.y0 * 1.001,
                curve_points.y1 * 0.999,
                curve_points.y1 * 1.001,
                torch.tensor(1.0, device=self.device),
            ],
            device=self.device,
        )

        for y in y_values:
            grad_output = torch.ones(1, device=self.device)
            grad_raw, grad_y = _call_slang_inverse_ppf_single_bwd(raw_params, y.view(1), grad_output)
            assert torch.isfinite(grad_raw).all(), "grad_raw contains inf or nan"
            assert torch.isfinite(grad_y).all(), "grad_y contains inf or nan"

    def test_backward_negative_y_zero_grad(self):
        """Negative y should clamp to 0 branch with zero gradient."""
        raw_params = torch.randn(7, device=self.device) * 0.5
        y = torch.tensor([-0.5], device=self.device)

        grad_output = torch.ones(1, device=self.device)
        grad_raw, grad_y = _call_slang_inverse_ppf_single_bwd(raw_params, y, grad_output)

        torch.testing.assert_close(grad_raw, torch.zeros_like(grad_raw))
        torch.testing.assert_close(grad_y, torch.zeros_like(grad_y))


class TestCurvePoints:
    """
    Forward and backward tests for compute_curve_points and crf_curve_points helpers.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = "cuda"
        torch.manual_seed(123)

    @staticmethod
    def _stack_curve_points(curve_points: PiecewisePowerFunction.CurvePoints) -> torch.Tensor:
        return torch.stack(
            [
                curve_points.x0,
                curve_points.y0,
                curve_points.slope_p0,
                curve_points.y0_pre_gamma,
                curve_points.slope_line,
                curve_points.gamma,
                curve_points.x1,
                curve_points.y1,
                curve_points.slope_p1,
                curve_points.shoulder_x,
                curve_points.shoulder_y,
            ]
        )

    def test_compute_curve_points_forward_matches_python(self):
        raw_params = torch.randn(7, device=self.device) * 0.5
        slang_out = _call_slang_compute_curve_points(raw_params)
        curve_points = PiecewisePowerFunction.crf_curve_points(PiecewisePowerFunction.RawParams(raw_params))
        torch_out = self._stack_curve_points(curve_points)
        torch.testing.assert_close(slang_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_compute_curve_points_forward_multiple_random(self):
        for seed in range(5):
            torch.manual_seed(seed + 1)
            raw_params = torch.randn(7, device=self.device) * 0.5
            slang_out = _call_slang_compute_curve_points(raw_params)
            curve_points = PiecewisePowerFunction.crf_curve_points(PiecewisePowerFunction.RawParams(raw_params))
            torch_out = self._stack_curve_points(curve_points)
            torch.testing.assert_close(slang_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_compute_curve_points_backward_matches_python(self):
        raw_params = (torch.randn(7, device=self.device) * 0.5).requires_grad_(True)
        curve_points = PiecewisePowerFunction.crf_curve_points(PiecewisePowerFunction.RawParams(raw_params))
        torch_out = self._stack_curve_points(curve_points)
        grad_output = torch.randn_like(torch_out)
        (torch_grad,) = torch.autograd.grad(torch_out, raw_params, grad_output)
        slang_grad = _call_slang_compute_curve_points_bwd(raw_params.detach(), grad_output)
        torch.testing.assert_close(slang_grad, torch_grad, rtol=1e-5, atol=1e-6)

    def test_compute_curve_points_backward_zero_grad_output(self):
        raw_params = torch.randn(7, device=self.device) * 0.5
        grad_output = torch.zeros(11, device=self.device)
        slang_grad = _call_slang_compute_curve_points_bwd(raw_params, grad_output)
        torch.testing.assert_close(slang_grad, torch.zeros_like(slang_grad))

    def test_crf_curve_points_forward_matches_python(self):
        crf_params = torch.randn(2, 3, 7, device=self.device) * 0.5
        camera_idx = 1
        channel_idx = 0
        slang_out = _call_slang_crf_curve_points(crf_params, camera_idx, channel_idx)
        raw_params = PiecewisePowerFunction.RawParams(crf_params[camera_idx, channel_idx])
        curve_points = PiecewisePowerFunction.crf_curve_points(raw_params)
        torch_out = self._stack_curve_points(curve_points)
        torch.testing.assert_close(slang_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_crf_curve_points_forward_multiple_random(self):
        for seed in range(5):
            torch.manual_seed(seed + 2)
            crf_params = torch.randn(2, 3, 7, device=self.device) * 0.5
            camera_idx = torch.randint(0, 2, (1,)).item()
            channel_idx = torch.randint(0, 3, (1,)).item()
            slang_out = _call_slang_crf_curve_points(crf_params, camera_idx, channel_idx)
            raw_params = PiecewisePowerFunction.RawParams(crf_params[camera_idx, channel_idx])
            curve_points = PiecewisePowerFunction.crf_curve_points(raw_params)
            torch_out = self._stack_curve_points(curve_points)
            torch.testing.assert_close(slang_out, torch_out, rtol=1e-5, atol=1e-6)

    def test_crf_curve_points_backward_matches_python(self):
        crf_params = (torch.randn(2, 3, 7, device=self.device) * 0.5).requires_grad_(True)
        camera_idx = 0
        channel_idx = 2
        raw_params = PiecewisePowerFunction.RawParams(crf_params[camera_idx, channel_idx])
        curve_points = PiecewisePowerFunction.crf_curve_points(raw_params)
        torch_out = self._stack_curve_points(curve_points)
        grad_output = torch.randn_like(torch_out)
        (torch_grad,) = torch.autograd.grad(torch_out, crf_params, grad_output)
        slang_grad = _call_slang_crf_curve_points_bwd(crf_params.detach(), camera_idx, channel_idx, grad_output)
        torch.testing.assert_close(slang_grad, torch_grad, rtol=1e-5, atol=1e-6)

    def test_crf_curve_points_backward_zero_grad_output(self):
        crf_params = torch.randn(2, 3, 7, device=self.device) * 0.5
        camera_idx = 1
        channel_idx = 1
        grad_output = torch.zeros(11, device=self.device)
        slang_grad = _call_slang_crf_curve_points_bwd(crf_params, camera_idx, channel_idx, grad_output)
        torch.testing.assert_close(slang_grad, torch.zeros_like(slang_grad))
