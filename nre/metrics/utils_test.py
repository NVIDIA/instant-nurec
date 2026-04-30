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

from nre.metrics.utils import AggregationMethod, aggregate_tensors


class TestUtils(unittest.TestCase):
    def test_aggregation_method_enum_values(self):
        """Test that AggregationMethod enum has the expected values."""
        self.assertEqual(AggregationMethod.MEAN.name.lower(), "mean")
        self.assertEqual(AggregationMethod.SUM.name.lower(), "sum")
        self.assertEqual(AggregationMethod.MIN.name.lower(), "min")
        self.assertEqual(AggregationMethod.MAX.name.lower(), "max")
        self.assertEqual(AggregationMethod.WEIGHTED_MEAN.name.lower(), "weighted_mean")

    def test_aggregation_method_enum_members(self):
        """Test that AggregationMethod enum has the expected members."""
        expected_members = {"MEAN", "SUM", "MIN", "MAX", "WEIGHTED_MEAN"}
        actual_members = {member.name for member in AggregationMethod}
        self.assertEqual(actual_members, expected_members)

    def test_aggregate_tensors_mean(self):
        """Test mean aggregation of tensors."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        result = aggregate_tensors(tensors, method=AggregationMethod.MEAN)
        expected = torch.tensor([3.0, 4.0])  # mean of [1,3,5] and [2,4,6]
        torch.testing.assert_close(result, expected)

    def test_aggregate_tensors_sum(self):
        """Test sum aggregation of tensors."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        result = aggregate_tensors(tensors, method=AggregationMethod.SUM)
        expected = torch.tensor([9.0, 12.0])  # sum of [1,3,5] and [2,4,6]
        torch.testing.assert_close(result, expected)

    def test_aggregate_tensors_min(self):
        """Test min aggregation of tensors."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        result = aggregate_tensors(tensors, method=AggregationMethod.MIN)
        expected = torch.tensor([1.0, 2.0])  # min of [1,3,5] and [2,4,6]
        torch.testing.assert_close(result, expected)

    def test_aggregate_tensors_max(self):
        """Test max aggregation of tensors."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        result = aggregate_tensors(tensors, method=AggregationMethod.MAX)
        expected = torch.tensor([5.0, 6.0])  # max of [1,3,5] and [2,4,6]
        torch.testing.assert_close(result, expected)

    def test_aggregate_tensors_weighted_mean(self):
        """Test weighted mean aggregation of tensors."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        weights = [torch.tensor([2.0, 1.0]), torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0])]
        result = aggregate_tensors(tensors, weights=weights, method=AggregationMethod.WEIGHTED_MEAN)
        # Weighted mean: (1*2 + 3*1 + 5*1)/(2+1+1) = 10/4 = 2.5, (2*1 + 4*2 + 6*1)/(1+2+1) = 16/4 = 4.0
        expected = torch.tensor([2.5, 4.0])
        torch.testing.assert_close(result, expected)

    def test_aggregate_tensors_weighted_mean_equal_weights(self):
        """Test weighted mean with equal weights (should be same as regular mean)."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        weights = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
        result = aggregate_tensors(tensors, weights=weights, method=AggregationMethod.WEIGHTED_MEAN)
        expected = torch.tensor([3.0, 4.0])  # same as regular mean
        torch.testing.assert_close(result, expected)

    def test_aggregate_tensors_weighted_mean_zero_weights(self):
        """Test weighted mean with zero weights (should handle division by zero)."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        weights = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        result = aggregate_tensors(tensors, weights=weights, method=AggregationMethod.WEIGHTED_MEAN)
        # When all weights are zero, the weighted mean is undefined (division by zero), so result should be NaN
        expected = torch.tensor([float("nan"), float("nan")])
        torch.testing.assert_close(result, expected, equal_nan=True)

    def test_aggregate_tensors_weighted_mean_mixed_zero_weights(self):
        """Test weighted mean with some zero weights."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        weights = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])]
        result = aggregate_tensors(tensors, weights=weights, method=AggregationMethod.WEIGHTED_MEAN)
        # For first element: (1*0 + 3*1 + 5*1)/(0+1+1) = 8/2 = 4.0
        # For second element: (2*1 + 4*0 + 6*1)/(1+0+1) = 8/2 = 4.0
        expected = torch.tensor([4.0, 4.0])
        torch.testing.assert_close(result, expected)

    def test_aggregate_tensors_weighted_mean_scalar_tensors(self):
        """Test weighted mean with scalar tensors."""
        tensors = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        weights = [torch.tensor(2.0), torch.tensor(1.0), torch.tensor(1.0)]
        result = aggregate_tensors(tensors, weights=weights, method=AggregationMethod.WEIGHTED_MEAN)
        # Weighted mean: (1*2 + 2*1 + 3*1)/(2+1+1) = 7/4 = 1.75
        self.assertEqual(result.item(), 1.75)

    def test_aggregate_tensors_weighted_mean_3d_tensors(self):
        """Test weighted mean with 3D tensors."""
        tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])]
        weights = [torch.tensor([[[2.0, 1.0], [1.0, 2.0]]]), torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])]
        result = aggregate_tensors(tensors, weights=weights, method=AggregationMethod.WEIGHTED_MEAN)
        # Weighted mean calculation for each element
        expected = torch.tensor(
            [
                [
                    [(1 * 2 + 5 * 1) / (2 + 1), (2 * 1 + 6 * 2) / (1 + 2)],
                    [(3 * 1 + 7 * 2) / (1 + 2), (4 * 2 + 8 * 1) / (2 + 1)],
                ]
            ]
        )
        torch.testing.assert_close(result, expected)

    def test_aggregate_tensors_weighted_mean_missing_weights(self):
        """Test weighted mean without providing weights (should raise error)."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        with self.assertRaises(ValueError) as context:
            aggregate_tensors(tensors, method=AggregationMethod.WEIGHTED_MEAN)
        self.assertIn("Weights must be provided for WEIGHTED_MEAN", str(context.exception))

    def test_aggregate_tensors_weighted_mean_mismatched_lengths(self):
        """Test weighted mean with mismatched tensor and weight lengths."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        weights = [torch.tensor([1.0, 1.0])]  # Only one weight for two tensors
        with self.assertRaises(ValueError) as context:
            aggregate_tensors(tensors, weights=weights, method=AggregationMethod.WEIGHTED_MEAN)
        self.assertIn("Values and weights must have the same length", str(context.exception))

    def test_aggregate_tensors_weighted_mean_empty_list(self):
        """Test weighted mean with empty tensor list."""
        tensors = []
        weights = []
        with self.assertRaises(ValueError) as context:
            aggregate_tensors(tensors, weights=weights, method=AggregationMethod.WEIGHTED_MEAN)
        self.assertIn("Cannot aggregate empty list", str(context.exception))

    def test_aggregate_tensors_single_tensor(self):
        """Test aggregation with a single tensor."""
        tensors = [torch.tensor([1.0, 2.0, 3.0])]

        result_mean = aggregate_tensors(tensors, method=AggregationMethod.MEAN)
        result_sum = aggregate_tensors(tensors, method=AggregationMethod.SUM)
        result_min = aggregate_tensors(tensors, method=AggregationMethod.MIN)
        result_max = aggregate_tensors(tensors, method=AggregationMethod.MAX)

        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result_mean, expected)
        torch.testing.assert_close(result_sum, expected)
        torch.testing.assert_close(result_min, expected)
        torch.testing.assert_close(result_max, expected)

    def test_aggregate_tensors_scalar_tensors(self):
        """Test aggregation of scalar tensors."""
        tensors = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]

        result_mean = aggregate_tensors(tensors, method=AggregationMethod.MEAN)
        result_sum = aggregate_tensors(tensors, method=AggregationMethod.SUM)
        result_min = aggregate_tensors(tensors, method=AggregationMethod.MIN)
        result_max = aggregate_tensors(tensors, method=AggregationMethod.MAX)

        self.assertEqual(result_mean.item(), 2.0)
        self.assertEqual(result_sum.item(), 6.0)
        self.assertEqual(result_min.item(), 1.0)
        self.assertEqual(result_max.item(), 3.0)

    def test_aggregate_tensors_negative_values(self):
        """Test aggregation with negative values."""
        tensors = [torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -4.0]), torch.tensor([-5.0, -6.0])]

        result_mean = aggregate_tensors(tensors, method=AggregationMethod.MEAN)
        result_sum = aggregate_tensors(tensors, method=AggregationMethod.SUM)
        result_min = aggregate_tensors(tensors, method=AggregationMethod.MIN)
        result_max = aggregate_tensors(tensors, method=AggregationMethod.MAX)

        torch.testing.assert_close(result_mean, torch.tensor([-3.0, -4.0]))
        torch.testing.assert_close(result_sum, torch.tensor([-9.0, -12.0]))
        torch.testing.assert_close(result_min, torch.tensor([-5.0, -6.0]))
        torch.testing.assert_close(result_max, torch.tensor([-1.0, -2.0]))

    def test_aggregate_tensors_mixed_shapes(self):
        """Test aggregation with tensors of different shapes."""
        tensors = [torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]

        # This should raise an error due to different shapes
        with self.assertRaises(RuntimeError):
            aggregate_tensors(tensors, method=AggregationMethod.MEAN)

    def test_aggregate_tensors_empty_list(self):
        """Test aggregation with empty tensor list."""
        tensors = []

        with self.assertRaises(RuntimeError):
            aggregate_tensors(tensors, method=AggregationMethod.MEAN)

    def test_aggregate_tensors_different_dtypes(self):
        """Test aggregation with tensors of different dtypes."""
        tensors = [torch.tensor([1, 2], dtype=torch.int32), torch.tensor([3, 4], dtype=torch.float32)]

        result = aggregate_tensors(tensors, method=AggregationMethod.MEAN)
        # Should work and result in float dtype
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.tensor([2.0, 3.0]))

    def test_aggregate_tensors_3d_tensors(self):
        """Test aggregation with 3D tensors."""
        tensors = [torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])]

        result_mean = aggregate_tensors(tensors, method=AggregationMethod.MEAN)
        result_sum = aggregate_tensors(tensors, method=AggregationMethod.SUM)

        expected_mean = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
        expected_sum = torch.tensor([[[6.0, 8.0], [10.0, 12.0]]])

        torch.testing.assert_close(result_mean, expected_mean)
        torch.testing.assert_close(result_sum, expected_sum)
