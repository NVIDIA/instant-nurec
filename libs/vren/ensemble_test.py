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

import numpy as np
import torch

from libs.vren.ensemble import ensemble_cuda, ensemble_numba


class TestEnsemble(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def test_ensemble_consistency(self):
        """Test consistency between CUDA and Python implementations of ensemble function"""
        # Generate test data
        num_points = 100
        num_cameras = 7
        num_classes = 20
        ignore_label = 255

        # Create test cases with different scenarios
        test_cases = [
            # Case 1: Completely random labels
            np.random.randint(0, num_classes, size=(num_points, num_cameras), dtype=np.uint8),
            # Case 2: Mixed data with ignore_label
            np.random.choice([0, 1, 2, ignore_label], size=(num_points, num_cameras)).astype(np.uint8),
            # Case 3: All ignore_label
            np.full((num_points, num_cameras), ignore_label, dtype=np.uint8),
            # Case 4: Same label for each point across cameras
            np.broadcast_to(np.random.randint(0, num_classes, size=(num_points, 1)), (num_points, num_cameras)).astype(
                np.uint8
            ),
        ]

        for i, test_data in enumerate(test_cases):
            # Run CUDA version
            cuda_result = ensemble_cuda(test_data, self.device, ignore_label)

            # Run Python version
            python_result = ensemble_numba(test_data, ignore_label)

            # Verify results
            np.testing.assert_array_equal(
                cuda_result.cpu().numpy(),
                python_result,
                err_msg=f"Test case {i + 1} failed: CUDA and Python outputs do not match. CUDA: {cuda_result}, Python: {python_result}",
            )

    def test_edge_cases(self):
        """Test edge cases for both implementations"""
        ignore_label = 255

        # Test empty array
        empty_data = np.zeros((0, 4), dtype=np.uint8)
        cuda_result = ensemble_cuda(empty_data, self.device, ignore_label)
        python_result = ensemble_numba(empty_data, ignore_label)
        np.testing.assert_array_equal(
            cuda_result.cpu().numpy(),
            python_result,
            err_msg="Test empty array failed: CUDA and Python outputs do not match. CUDA: {cuda_result}, Python: {python_result}",
        )

        # Test single camera
        single_camera = np.random.randint(0, 20, size=(100, 1), dtype=np.uint8)
        cuda_result = ensemble_cuda(single_camera, self.device, ignore_label)
        python_result = ensemble_numba(single_camera, ignore_label)
        np.testing.assert_array_equal(
            cuda_result.cpu().numpy(),
            python_result,
            err_msg="Test single camera failed: CUDA and Python outputs do not match. CUDA: {cuda_result}, Python: {python_result}",
        )

        # Test single point
        single_point = np.random.randint(0, 20, size=(1, 100), dtype=np.uint8)
        cuda_result = ensemble_cuda(single_point, self.device, ignore_label)
        python_result = ensemble_numba(single_point, ignore_label)
        np.testing.assert_array_equal(
            cuda_result.cpu().numpy(),
            python_result,
            err_msg="Test single point failed: CUDA and Python outputs do not match. CUDA: {cuda_result}, Python: {python_result}",
        )

    def test_data_type(self):
        num_points = 100
        num_cameras = 2
        num_classes = 20
        ignore_label = 255
        test_case = [
            # Case 1: numpy array
            np.random.randint(0, num_classes, size=(num_points, num_cameras), dtype=np.uint8),
            # Case 2: torch tensor
            torch.randint(0, num_classes, size=(num_points, num_cameras), dtype=torch.uint8),
            # Case 3: All ignore_label, numpy array
            np.full((num_points, num_cameras), ignore_label, dtype=np.uint8),
            # Case 4: All ignore_label, torch tensor
            torch.full((num_points, num_cameras), ignore_label, dtype=torch.uint8),
            # Case 5: Empty data, numpy array
            np.zeros((0, num_cameras), dtype=np.uint8),
            # Case 6: Empty data, torch tensor
            torch.zeros((0, num_cameras), dtype=torch.uint8),
        ]

        for i, test_data in enumerate(test_case):
            cuda_result = ensemble_cuda(test_data, self.device, ignore_label)
            # assert input and output have the same data type
            if isinstance(test_data, np.ndarray):
                test_data_dtype = test_data.dtype
            else:
                test_data_dtype = test_data.cpu().numpy().dtype

            self.assertEqual(
                test_data_dtype,
                cuda_result.cpu().numpy().dtype,
                msg=f"Test case {i + 1} failed: input and output types do not match. input: {test_data}, output: {cuda_result}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
