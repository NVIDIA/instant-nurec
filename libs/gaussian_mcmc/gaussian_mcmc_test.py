# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import time
import unittest

import parameterized
import torch

from libs.gaussian_mcmc.interface import gaussian_mcmc  # type: ignore
from nre.utils.geometry import quat_to_so3_matrix
from nre.utils.tests import CommonTestCase, is_perf_test_mode


class QuatScaleToCovarianceTest(CommonTestCase):
    def setUp(self):
        # Sample random rotations and quaternions
        self.device = torch.device("cuda")
        self.num_gaussians = 1000000
        self.quats = torch.randn(self.num_gaussians, 4, device=self.device)
        self.scales = torch.randn(self.num_gaussians, 3, device=self.device)

        # Normalize the quaternions
        self.quats = torch.nn.functional.normalize(self.quats, dim=1)
        self.n_runs = 10

    def reference_implementation(self, dtype: torch.dtype = torch.float32):
        S = torch.zeros((self.num_gaussians, 3, 3), dtype=dtype, device=self.device)
        R = quat_to_so3_matrix(self.quats)

        S[:, 0, 0] = self.scales[:, 0]
        S[:, 1, 1] = self.scales[:, 1]
        S[:, 2, 2] = self.scales[:, 2]

        return R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)

    @parameterized.parameterized.expand(
        [
            (torch.float32,),
            (torch.float64,),
        ]
    )
    def test_quat_scale_to_covariance(self, dtype):
        # Call your CUDA extension
        # Output should be (N, 3, 3)
        self.quats = self.quats.to(dtype)
        self.scales = self.scales.to(dtype)
        covars = gaussian_mcmc.quat_scale_to_covariance(self.quats, self.scales)
        covars_ref = self.reference_implementation(dtype)

        # Check shape
        self.assertEqual(covars.shape, (self.num_gaussians, 3, 3))
        self.assertEqual(covars_ref.shape, (self.num_gaussians, 3, 3))

        # Check that the two implementations are close
        self._compareTensor(covars, covars_ref, decimal=4)

    @parameterized.parameterized.expand(
        [
            (torch.float32,),
            (torch.float64,),
        ]
    )
    def test_xyzw_format(self, dtype):
        # Call your CUDA extension
        # Output should be (N, 3, 3)
        self.quats = self.quats.to(dtype)
        self.scales = self.scales.to(dtype)

        # Compute covariances using both conventions
        covars_xyzw = gaussian_mcmc.quat_scale_to_covariance(self.quats, self.scales, "xyzw")
        covars_wxyz = gaussian_mcmc.quat_scale_to_covariance(self.quats[:, [3, 0, 1, 2]], self.scales, "wxyz")

        # They should be (almost) identical
        self._compareTensor(covars_xyzw, covars_wxyz, decimal=5)

    @unittest.skipUnless(
        is_perf_test_mode(),
        "Performance tests are skipped by default. Set RUN_PERF_TESTS=1 to run.",
    )
    def test_performance(self):
        # Time CUDA kernel
        cuda_times = []
        ref_times = []

        # Warm up
        for _ in range(5):
            self.reference_implementation()
        torch.cuda.synchronize()

        for _ in range(self.n_runs):
            # CUDA kernel timing
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = gaussian_mcmc.quat_scale_to_covariance(self.quats, self.scales)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            cuda_times.append(t1 - t0)

            # Reference timing
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = self.reference_implementation()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ref_times.append(t1 - t0)

        cuda_times = torch.tensor(cuda_times)
        ref_times = torch.tensor(ref_times)

        print(
            f"CUDA kernel: mean={cuda_times.mean() * 1000:.3f}ms, std={cuda_times.std() * 1000:.3f}ms, min={cuda_times.min() * 1000:.3f}ms, max={cuda_times.max() * 1000:.3f}ms"
        )
        print(
            f"Reference:  mean={ref_times.mean() * 1000:.3f}ms, std={ref_times.std() * 1000:.3f}ms, min={ref_times.min() * 1000:.3f}ms, max={ref_times.max() * 1000:.3f}ms"
        )
        print(f"Speedup: {ref_times.mean() / cuda_times.mean():.2f}x")
