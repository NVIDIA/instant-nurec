# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for pose calibration kernels (CUDA and Slang)."""

import unittest

from typing import Tuple

import torch
import torch.nn as nn

from scipy.spatial.transform import Rotation

from libs.sensors.kernels.pose_calib import compute_poses_and_timestamps
from ncore.impl.data.types import ShutterType
from nre.utils.sensors.sensors import SensorModelComputations


def compute_reference_poses_and_timestamps(
    embedding: nn.Embedding,
    T_sensor_world_startend_allviews: torch.Tensor,
    timestamps_startend_us_allviews: torch.Tensor,
    frame_idx: torch.Tensor,
    rect_points_lb: torch.Tensor,
    resolution: torch.Tensor,
    shutter_type: ShutterType,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute reference poses and timestamps using sensors.py PyTorch implementation."""
    batch_size = frame_idx.shape[0]
    T_out_ref_list = []
    timestamps_out_ref_list = []
    for i in range(batch_size):
        fidx = int(frame_idx[i].item())
        result = SensorModelComputations._get_poses_and_timestamps_startend_compiled(
            subsample_rect_points_lb=rect_points_lb[i],
            subsample_resolution=resolution,
            embeds=embedding,
            T_offset_nre_startend=None,
            T_sensor_world_startend_allviews=T_sensor_world_startend_allviews,
            timestamps_startend_us_allviews=timestamps_startend_us_allviews,
            sensor_model_shutter_type_if_not_lidar=shutter_type,
            unique_frame_idx=fidx,
            unique_frame_idx_tensor=frame_idx[i : i + 1],
            enable_calib=True,
            is_lidar=False,
        )
        T_out_ref_list.append(result.T_sensor_world_startend)
        timestamps_out_ref_list.append(result.timestamps_startend_us)

    return torch.stack(T_out_ref_list, dim=0), torch.stack(timestamps_out_ref_list, dim=0)


def make_random_startend_poses(n_frames: int, device: torch.device) -> torch.Tensor:
    """Create n_frames * 2 valid random 4x4 pose matrices, shaped (n_frames, 2, 4, 4)."""
    n = n_frames * 2
    translation = torch.randn(n, 3, device=device, dtype=torch.float32)
    rotation_matrices = torch.from_numpy(Rotation.random(n).as_matrix()).to(device=device, dtype=torch.float32)

    T = torch.zeros(n, 4, 4, device=device, dtype=torch.float32)
    T[:, :3, :3] = rotation_matrices
    T[:, :3, 3] = translation
    T[:, 3, 3] = 1.0

    return T.reshape(n_frames, 2, 4, 4)


class TestPoseCalibKernels(unittest.TestCase):
    """Test cases for pose calibration CUDA kernels."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def test_compute_poses_and_timestamps(self):
        """Test forward and backward pass with rolling shutter and subsampling.

        Compares the CUDA kernel (compute_poses_and_timestamps) against
        the PyTorch reference implementation in sensors.py.
        """
        N_frames = 5
        batch_size = 3
        width, height = 1920, 1080
        shutter_type = ShutterType.ROLLING_TOP_TO_BOTTOM

        T_sensor_world_startend_allviews = make_random_startend_poses(N_frames, self.device)

        embedding = nn.Embedding(N_frames, 9, device=self.device)
        embed_weights = embedding.weight

        frame_idx = torch.randint(0, N_frames, (batch_size,), device=self.device, dtype=torch.int32)
        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(1, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_startend_us_allviews = torch.stack([timestamps_start, timestamps_end], dim=1)

        rect_lt = torch.rand(batch_size, 2, device=self.device, dtype=torch.float32) * 100
        rect_rb = rect_lt + torch.rand(batch_size, 2, device=self.device, dtype=torch.float32) * 400 + 100
        rect_points_lb = torch.stack([rect_lt, rect_rb], dim=1)
        resolution = torch.tensor([width, height], device=self.device, dtype=torch.float32)

        self.assertTrue(
            (rect_points_lb[:, :, 0] >= 0).all() and (rect_points_lb[:, :, 0] < width).all(),
            "x coordinates must be in [0, width)",
        )
        self.assertTrue(
            (rect_points_lb[:, :, 1] >= 0).all() and (rect_points_lb[:, :, 1] < height).all(),
            "y coordinates must be in [0, height)",
        )

        resolution_batched = resolution.unsqueeze(0).expand(batch_size, 2).contiguous()
        T_out, timestamps_out = compute_poses_and_timestamps(
            T_sensor_world_startend_allviews,
            embed_weights,
            frame_idx,
            rect_points_lb,
            resolution_batched,
            timestamps_startend_us_allviews,
            int(shutter_type),
            True,
        )

        T_out_ref, timestamps_out_ref = compute_reference_poses_and_timestamps(
            embedding,
            T_sensor_world_startend_allviews,
            timestamps_startend_us_allviews,
            frame_idx,
            rect_points_lb,
            resolution,
            shutter_type,
        )

        # float32 matrix-quat roundtrip introduces ~1e-6 error
        torch.testing.assert_close(T_out, T_out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(timestamps_out, timestamps_out_ref)

        T_out_grad = torch.randn_like(T_out)

        embedding.weight.grad = None
        T_out.backward(T_out_grad)
        cuda_grad = embed_weights.grad.clone()

        embedding.weight.grad = None
        T_out_ref.backward(T_out_grad)
        ref_grad = embed_weights.grad

        # CUDA backward uses analytic Jacobians; reference uses PyTorch autograd.
        # Both produce exact gradients but via different op orderings, causing
        # float32 rounding differences.
        torch.testing.assert_close(cuda_grad, ref_grad, atol=1e-4, rtol=1e-4)

    def test_no_calib_no_subsampling(self):
        """Test the simplest path: no calibration, no subsampling."""
        N_frames = 3
        batch_size = 4

        T_allviews = make_random_startend_poses(N_frames, self.device)
        frame_idx = torch.randint(0, N_frames, (batch_size,), device=self.device, dtype=torch.int32)
        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(1, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_startend = torch.stack([timestamps_start, timestamps_end], dim=1)

        T_out, ts_out = compute_poses_and_timestamps(
            T_allviews,
            None,
            frame_idx,
            None,
            None,
            timestamps_startend,
            int(ShutterType.GLOBAL),
            False,
        )

        self.assertEqual(T_out.shape, (batch_size, 2, 4, 4))
        self.assertEqual(ts_out.shape, (batch_size, 2))

        for i in range(batch_size):
            fidx = frame_idx[i].item()
            # no-calib is a direct copy, expect near-exact match
            torch.testing.assert_close(T_out[i], T_allviews[fidx], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(ts_out[i], timestamps_startend[fidx])

    def test_empty_batch(self):
        """batch_size=0 should return empty tensors."""
        N_frames = 3
        T_allviews = make_random_startend_poses(N_frames, self.device)
        frame_idx = torch.empty(0, device=self.device, dtype=torch.int32)
        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(1, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_startend = torch.stack([timestamps_start, timestamps_end], dim=1)

        T_out, ts_out = compute_poses_and_timestamps(
            T_allviews,
            None,
            frame_idx,
            None,
            None,
            timestamps_startend,
            int(ShutterType.GLOBAL),
            False,
        )

        self.assertEqual(T_out.shape[0], 0)
        self.assertEqual(ts_out.shape[0], 0)

    def test_shared_frame_idx_backward(self):
        """All batch elements reference same frame -- stress test atomicAdd accumulation."""
        N_frames = 3
        batch_size = 16

        T_allviews = make_random_startend_poses(N_frames, self.device)
        embedding = nn.Embedding(N_frames, 9, device=self.device)
        embed_weights = embedding.weight
        frame_idx = torch.zeros(batch_size, device=self.device, dtype=torch.int32)
        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(1, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_startend = torch.stack([timestamps_start, timestamps_end], dim=1)

        T_out, _ = compute_poses_and_timestamps(
            T_allviews,
            embed_weights,
            frame_idx,
            None,
            None,
            timestamps_startend,
            int(ShutterType.GLOBAL),
            True,
        )

        T_out_grad = torch.randn_like(T_out)
        T_out.backward(T_out_grad)

        self.assertIsNotNone(embed_weights.grad)
        self.assertTrue(torch.isfinite(embed_weights.grad).all(), "gradients must be finite")
        self.assertTrue((embed_weights.grad[0].abs() > 0).any(), "frame 0 gradient should be non-zero")

    def test_identity_calibration(self):
        """Zero embed_weights should produce identity calibration (output == input poses)."""
        N_frames = 3
        batch_size = 4

        T_allviews = make_random_startend_poses(N_frames, self.device)
        embed_weights = torch.zeros(N_frames, 9, device=self.device)
        frame_idx = torch.arange(min(batch_size, N_frames), device=self.device, dtype=torch.int32)
        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(1, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_startend = torch.stack([timestamps_start, timestamps_end], dim=1)

        T_out, _ = compute_poses_and_timestamps(
            T_allviews,
            embed_weights,
            frame_idx,
            None,
            None,
            timestamps_startend,
            int(ShutterType.GLOBAL),
            True,
        )

        # With zero embed_weights, rotation6d_to_matrix produces identity rotation
        # and dx=0 translation, so output should match input
        for i in range(min(batch_size, N_frames)):
            fidx = frame_idx[i].item()
            torch.testing.assert_close(T_out[i], T_allviews[fidx], atol=1e-5, rtol=1e-5)

    def test_gradcheck_calib(self):
        """Verify analytic gradients via finite-difference gradcheck.

        The CUDA kernel only supports float32, so we run gradcheck at float32
        with larger eps and atol to accommodate reduced numerical precision.
        """
        torch.manual_seed(42)
        N_frames = 3
        batch_size = 4

        T_allviews = make_random_startend_poses(N_frames, self.device)
        embed_weights = torch.randn(N_frames, 9, device=self.device, dtype=torch.float32, requires_grad=True)
        frame_idx = torch.randint(0, N_frames, (batch_size,), device=self.device, dtype=torch.int32)
        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(
            500000, 1000000, (N_frames,), device=self.device, dtype=torch.int64
        )
        timestamps_startend = torch.stack([timestamps_start, timestamps_end], dim=1)

        def func(ew):
            T_out, _ = compute_poses_and_timestamps(
                T_allviews,
                ew,
                frame_idx,
                None,
                None,
                timestamps_startend,
                int(ShutterType.GLOBAL),
                True,
            )
            return T_out

        # float32 requires larger eps and atol for finite-difference stability
        torch.autograd.gradcheck(
            func,
            (embed_weights,),
            eps=1e-3,
            atol=1e-2,
            raise_exception=True,
        )

    def test_gradcheck_calib_with_subsampling(self):
        """Finite-difference verification of analytic backward with rolling shutter subsampling."""
        torch.manual_seed(42)
        N_frames = 3
        batch_size = 4
        width, height = 1920, 1080

        T_allviews = make_random_startend_poses(N_frames, self.device)
        embed_weights = torch.randn(N_frames, 9, device=self.device, dtype=torch.float32, requires_grad=True) * 0.01
        frame_idx = torch.randint(0, N_frames, (batch_size,), device=self.device, dtype=torch.int32)

        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(
            500000, 1000000, (N_frames,), device=self.device, dtype=torch.int64
        )
        timestamps_startend = torch.stack([timestamps_start, timestamps_end], dim=1)

        rect_lt = torch.rand(batch_size, 2, device=self.device) * 100
        rect_rb = rect_lt + torch.rand(batch_size, 2, device=self.device) * 400 + 100
        rect_points_lb = torch.stack([rect_lt, rect_rb], dim=1)
        resolution = (
            torch.tensor([width, height], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, 2)
            .contiguous()
        )

        def func(ew):
            T_out, _ = compute_poses_and_timestamps(
                T_allviews,
                ew,
                frame_idx,
                rect_points_lb,
                resolution,
                timestamps_startend,
                int(ShutterType.ROLLING_TOP_TO_BOTTOM),
                True,
            )
            return T_out

        torch.autograd.gradcheck(func, (embed_weights,), eps=1e-3, atol=1e-2, raise_exception=True)


class TestPoseCalibCudaVsSlang(unittest.TestCase):
    """A/B equivalence test: CUDA kernels vs Slang kernels."""

    def setUp(self):
        self.device = torch.device("cuda")

    def _run_slang_forward(
        self,
        T_allviews,
        embed_weights,
        frame_idx,
        rect_points_lb,
        resolution,
        timestamps_startend,
        shutter_type,
        enable_calib,
        has_subsampling,
    ):
        """Run the Slang kernel forward pass directly."""
        from libs.sensors.kernels.pose_calib import pose_calib_slang
        from libs.slang_utils.utils import div_up

        batch_size = frame_idx.shape[0]
        device = T_allviews.device
        T_out = torch.empty((batch_size, 2, 4, 4), device=device, dtype=torch.float32)
        timestamps_out = torch.empty((batch_size, 2), device=device, dtype=torch.int64)

        threads = 256
        blocks = div_up(batch_size, threads)

        if has_subsampling:
            if enable_calib:
                pose_calib_slang.compute_poses_and_timestamps_calib_subsample_kernel(
                    (threads, 1, 1),
                    (blocks, 1, 1),
                    batch_size,
                    T_allviews,
                    (embed_weights, (embed_weights,)),
                    frame_idx,
                    rect_points_lb,
                    resolution,
                    timestamps_startend,
                    shutter_type,
                    (T_out, (T_out,)),
                    timestamps_out,
                )
            else:
                pose_calib_slang.compute_poses_and_timestamps_subsample_kernel(
                    (threads, 1, 1),
                    (blocks, 1, 1),
                    batch_size,
                    T_allviews,
                    frame_idx,
                    rect_points_lb,
                    resolution,
                    timestamps_startend,
                    shutter_type,
                    (T_out, (T_out,)),
                    timestamps_out,
                )
        else:
            if enable_calib:
                pose_calib_slang.compute_poses_and_timestamps_calib_kernel(
                    (threads, 1, 1),
                    (blocks, 1, 1),
                    batch_size,
                    T_allviews,
                    (embed_weights, (embed_weights,)),
                    frame_idx,
                    timestamps_startend,
                    (T_out, (T_out,)),
                    timestamps_out,
                )
            else:
                pose_calib_slang.compute_poses_and_timestamps_kernel(
                    (threads, 1, 1),
                    (blocks, 1, 1),
                    batch_size,
                    T_allviews,
                    frame_idx,
                    timestamps_startend,
                    (T_out, (T_out,)),
                    timestamps_out,
                )

        return T_out, timestamps_out

    def test_cuda_vs_slang_all_variants(self):
        """Test all 4 kernel variants: CUDA output matches Slang output."""
        N_frames = 5
        batch_size = 8
        width, height = 1920, 1080
        torch.manual_seed(42)

        T_allviews = make_random_startend_poses(N_frames, self.device)
        embedding = nn.Embedding(N_frames, 9, device=self.device)
        embed_weights = embedding.weight
        frame_idx = torch.randint(0, N_frames, (batch_size,), device=self.device, dtype=torch.int32)
        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(1, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_startend = torch.stack([timestamps_start, timestamps_end], dim=1)

        rect_lt = torch.rand(batch_size, 2, device=self.device, dtype=torch.float32) * 100
        rect_rb = rect_lt + torch.rand(batch_size, 2, device=self.device, dtype=torch.float32) * 400 + 100
        rect_points_lb = torch.stack([rect_lt, rect_rb], dim=1)
        resolution_batched = (
            torch.tensor([width, height], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, 2)
            .contiguous()
        )

        variants = [
            ("no_calib_no_sub", False, False),
            ("no_calib_sub", False, True),
            ("calib_no_sub", True, False),
            ("calib_sub", True, True),
        ]

        for name, enable_calib, has_sub in variants:
            with self.subTest(variant=name):
                rp = rect_points_lb if has_sub else None
                res = resolution_batched if has_sub else None
                ew = embed_weights if enable_calib else None
                st = int(ShutterType.ROLLING_TOP_TO_BOTTOM) if has_sub else int(ShutterType.GLOBAL)

                # CUDA path
                T_cuda, ts_cuda = compute_poses_and_timestamps(
                    T_allviews,
                    ew,
                    frame_idx,
                    rp,
                    res,
                    timestamps_startend,
                    st,
                    enable_calib,
                )

                # Slang path
                T_slang, ts_slang = self._run_slang_forward(
                    T_allviews.contiguous(),
                    embed_weights.contiguous() if enable_calib else embed_weights,
                    frame_idx.contiguous(),
                    rp.contiguous() if rp is not None else None,
                    res.contiguous() if res is not None else None,
                    timestamps_startend.contiguous(),
                    st,
                    enable_calib,
                    has_sub,
                )

                # float32 matrix-quat roundtrip introduces ~1e-6 error
                torch.testing.assert_close(T_cuda, T_slang, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(ts_cuda, ts_slang)

    def _run_slang_backward(
        self,
        T_allviews,
        embed_weights,
        frame_idx,
        rect_points_lb,
        resolution,
        timestamps_startend,
        shutter_type,
        has_subsampling,
        T_out,
        timestamps_out,
        grad_T_out,
    ):
        """Run the Slang kernel backward pass directly, returning embed_weights gradient."""
        from libs.sensors.kernels.pose_calib import pose_calib_slang
        from libs.slang_utils.utils import div_up

        batch_size = frame_idx.shape[0]
        threads = 256
        blocks = div_up(batch_size, threads)

        grad_embed_weights = torch.zeros_like(embed_weights)

        if has_subsampling:
            pose_calib_slang.compute_poses_and_timestamps_calib_subsample_kernel_bwd_diff(
                (threads, 1, 1),
                (blocks, 1, 1),
                batch_size,
                T_allviews,
                (embed_weights, (grad_embed_weights,)),
                frame_idx,
                rect_points_lb,
                resolution,
                timestamps_startend,
                shutter_type,
                (T_out, (grad_T_out,)),
                timestamps_out,
            )
        else:
            pose_calib_slang.compute_poses_and_timestamps_calib_kernel_bwd_diff(
                (threads, 1, 1),
                (blocks, 1, 1),
                batch_size,
                T_allviews,
                (embed_weights, (grad_embed_weights,)),
                frame_idx,
                timestamps_startend,
                (T_out, (grad_T_out,)),
                timestamps_out,
            )

        return grad_embed_weights

    def test_cuda_vs_slang_backward_equivalence(self):
        """Compare CUDA backward gradients against Slang auto-diff backward."""
        N_frames = 5
        batch_size = 8
        width, height = 1920, 1080
        torch.manual_seed(42)

        T_allviews = make_random_startend_poses(N_frames, self.device)
        embed_weights = torch.randn(N_frames, 9, device=self.device, dtype=torch.float32, requires_grad=True)
        frame_idx = torch.randint(0, N_frames, (batch_size,), device=self.device, dtype=torch.int32)
        timestamps_start = torch.randint(0, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_end = timestamps_start + torch.randint(1, 500000, (N_frames,), device=self.device, dtype=torch.int64)
        timestamps_startend = torch.stack([timestamps_start, timestamps_end], dim=1)

        rect_lt = torch.rand(batch_size, 2, device=self.device, dtype=torch.float32) * 100
        rect_rb = rect_lt + torch.rand(batch_size, 2, device=self.device, dtype=torch.float32) * 400 + 100
        rect_points_lb = torch.stack([rect_lt, rect_rb], dim=1)
        resolution_batched = (
            torch.tensor([width, height], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, 2)
            .contiguous()
        )

        calib_variants = [
            ("calib_no_sub", False),
            ("calib_sub", True),
        ]

        for name, has_sub in calib_variants:
            with self.subTest(variant=name):
                rp = rect_points_lb if has_sub else None
                res = resolution_batched if has_sub else None
                st = int(ShutterType.ROLLING_TOP_TO_BOTTOM) if has_sub else int(ShutterType.GLOBAL)

                # CUDA forward + backward
                ew_cuda = embed_weights.detach().clone().requires_grad_(True)
                T_cuda, _ = compute_poses_and_timestamps(
                    T_allviews,
                    ew_cuda,
                    frame_idx,
                    rp,
                    res,
                    timestamps_startend,
                    st,
                    True,
                )

                grad_T_out = torch.randn_like(T_cuda)
                T_cuda.backward(grad_T_out)
                cuda_grad = ew_cuda.grad.clone()

                # Slang forward (to get T_out / timestamps_out for backward)
                ew_slang = embed_weights.detach().clone().contiguous()
                T_slang, ts_slang = self._run_slang_forward(
                    T_allviews.contiguous(),
                    ew_slang,
                    frame_idx.contiguous(),
                    rp.contiguous() if rp is not None else None,
                    res.contiguous() if res is not None else None,
                    timestamps_startend.contiguous(),
                    st,
                    True,
                    has_sub,
                )

                # Slang backward
                slang_grad = self._run_slang_backward(
                    T_allviews.contiguous(),
                    ew_slang.contiguous(),
                    frame_idx.contiguous(),
                    rp.contiguous() if rp is not None else None,
                    res.contiguous() if res is not None else None,
                    timestamps_startend.contiguous(),
                    st,
                    has_sub,
                    T_slang.contiguous(),
                    ts_slang.contiguous(),
                    grad_T_out.contiguous(),
                )

                # CUDA backward is PyTorch autograd; Slang backward is auto-diff.
                # Float32 rounding order differences produce gradient deltas within 1e-4.
                torch.testing.assert_close(cuda_grad, slang_grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
