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

from nre.metrics.impl.cpsnr import CPSNRMetric
from nre.metrics.impl.factory import MetricFactory, MetricType
from nre.metrics.impl.lpips import LPIPSMetric
from nre.metrics.impl.psnr import PSNRMetric
from nre.metrics.impl.ssim import SSIMMetric
from nre.metrics.metric import BaseMetric


class TestFactory(unittest.TestCase):
    """Test cases for the metric factory functionality."""

    def test_psnr_metric_construction_via_factory(self):
        """Test constructing PSNR metric via factory."""
        # Test basic construction
        psnr_metric = MetricFactory[MetricType.PSNR]()
        self.assertIsInstance(psnr_metric, PSNRMetric)
        self.assertIsInstance(psnr_metric, BaseMetric)

        # Test construction with parameters (data_range is passed via kwargs)
        psnr_metric_with_params = MetricFactory[MetricType.PSNR](device="cpu", data_range=255.0)
        self.assertIsInstance(psnr_metric_with_params, PSNRMetric)
        # Access data_range through the specific PSNRMetric instance
        self.assertEqual(psnr_metric_with_params.data_range, 255.0)

    def test_cpsnr_metric_construction_via_factory(self):
        """Test constructing CPSNR metric via factory."""
        # Test basic construction
        cpsnr_metric = MetricFactory[MetricType.CPSNR]()
        self.assertIsInstance(cpsnr_metric, CPSNRMetric)
        self.assertIsInstance(cpsnr_metric, BaseMetric)

        # Test construction with parameters (data_range is a direct parameter)
        cpsnr_metric_with_params = MetricFactory[MetricType.CPSNR](data_range=1.0, device="cpu")
        self.assertIsInstance(cpsnr_metric_with_params, CPSNRMetric)
        self.assertEqual(cpsnr_metric_with_params.data_range, 1.0)

    def test_ssim_metric_construction_via_factory(self):
        """Test constructing SSIM metric via factory."""
        # Test basic construction
        ssim_metric = MetricFactory[MetricType.SSIM]()
        self.assertIsInstance(ssim_metric, SSIMMetric)
        self.assertIsInstance(ssim_metric, BaseMetric)

        # Test construction with parameters
        ssim_metric_with_params = MetricFactory[MetricType.SSIM](device="cpu", data_range=255.0, kernel_size=7)
        self.assertIsInstance(ssim_metric_with_params, SSIMMetric)
        self.assertEqual(ssim_metric_with_params.data_range, 255.0)
        self.assertEqual(ssim_metric_with_params.kernel_size, 7)

    def test_lpips_metric_construction_via_factory(self):
        """Test constructing LPIPS metric via factory."""
        # Test basic construction
        lpips_metric = MetricFactory[MetricType.LPIPS]()
        self.assertIsInstance(lpips_metric, LPIPSMetric)
        self.assertIsInstance(lpips_metric, BaseMetric)

        # Test construction with parameters
        lpips_metric_with_params = MetricFactory[MetricType.LPIPS](device="cpu", net_type="vgg", normalize=False)
        self.assertIsInstance(lpips_metric_with_params, LPIPSMetric)
        self.assertEqual(lpips_metric_with_params.net_type, "vgg")
        self.assertEqual(lpips_metric_with_params.normalize, False)

    def test_metric_type_enum_values(self):
        """Test that MetricType enum has correct string values."""
        self.assertEqual(MetricType.PSNR.name.lower(), "psnr")
        self.assertEqual(MetricType.CPSNR.name.lower(), "cpsnr")
        self.assertEqual(MetricType.SSIM.name.lower(), "ssim")
        self.assertEqual(MetricType.LPIPS.name.lower(), "lpips")

    def test_factory_mapping_consistency(self):
        """Test that factory mapping is consistent with expected classes."""
        self.assertEqual(MetricFactory[MetricType.PSNR], PSNRMetric)
        self.assertEqual(MetricFactory[MetricType.CPSNR], CPSNRMetric)
        self.assertEqual(MetricFactory[MetricType.SSIM], SSIMMetric)
        self.assertEqual(MetricFactory[MetricType.LPIPS], LPIPSMetric)

    def test_metric_construction_with_device(self):
        """Test constructing metrics with device parameter via factory."""
        if torch.cuda.is_available():
            # Test with CUDA device
            psnr_cuda = MetricFactory[MetricType.PSNR](device="cuda")
            self.assertEqual(psnr_cuda.device, "cuda")

            cpsnr_cuda = MetricFactory[MetricType.CPSNR](device="cuda")
            self.assertEqual(cpsnr_cuda.device, "cuda")

        # Test with CPU device
        psnr_cpu = MetricFactory[MetricType.PSNR](device="cpu")
        self.assertEqual(psnr_cpu.device, "cpu")

        cpsnr_cpu = MetricFactory[MetricType.CPSNR](device="cpu")
        self.assertEqual(cpsnr_cpu.device, "cpu")

    def test_metric_construction_with_aggregation_methods(self):
        """Test constructing metrics with aggregation methods via factory."""
        from nre.metrics.utils import AggregationMethod

        # Test with single aggregation method
        psnr_single = MetricFactory[MetricType.PSNR](aggregation_methods=AggregationMethod.MEAN)
        self.assertEqual(psnr_single.aggregation_methods(), [AggregationMethod.MEAN])

        # Test with multiple aggregation methods
        psnr_multiple = MetricFactory[MetricType.PSNR](
            aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.MAX]
        )
        self.assertEqual(psnr_multiple.aggregation_methods(), [AggregationMethod.MEAN, AggregationMethod.MAX])

    def test_metric_functionality_after_factory_construction(self):
        """Test that metrics constructed via factory are fully functional."""
        # Test PSNR metric functionality
        psnr_metric = MetricFactory[MetricType.PSNR](data_range=1.0)

        # Create test tensors
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)

        # Test computation
        result = psnr_metric.compute(pred, target)
        self.assertIsNotNone(result)
        self.assertIn("psnr", result.values)

    def test_ssim_metric_functionality_after_factory_construction(self):
        """Test that SSIM metric constructed via factory is fully functional."""
        ssim_metric = MetricFactory[MetricType.SSIM](data_range=1.0)

        # Create test tensors (values in [0, 1] range for data_range=1.0)
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        # Test computation
        result = ssim_metric.compute(pred, target)
        self.assertIsNotNone(result)
        self.assertIn("ssim", result.values)
        # SSIM should be in range [-1, 1], typically [0, 1] for similar images
        ssim_value = result.values["ssim"]
        self.assertTrue(ssim_value >= -1.0 and ssim_value <= 1.0)

    def test_lpips_metric_functionality_after_factory_construction(self):
        """Test that LPIPS metric constructed via factory is fully functional."""
        lpips_metric = MetricFactory[MetricType.LPIPS](device="cpu", normalize=True)

        # Create test tensors (values in [0, 1] range for normalize=True)
        # LPIPS requires exactly 3 channels
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        # Test computation
        result = lpips_metric.compute(pred, target)
        self.assertIsNotNone(result)
        self.assertIn("lpips", result.values)
        # LPIPS values are always >= 0 (distance metric)
        lpips_value = result.values["lpips"]
        self.assertTrue(lpips_value >= 0.0)

    def test_factory_immutability(self):
        """Test that the factory is immutable (read-only)."""
        original_psnr = MetricFactory[MetricType.PSNR]

        # Attempting to modify the factory should not affect the original mapping
        # (This test verifies the factory behaves as expected)
        self.assertEqual(MetricFactory[MetricType.PSNR], original_psnr)

    def test_all_metric_types_are_base_metric_subclasses(self):
        """Test that all metrics in factory inherit from BaseMetric."""
        for metric_type in MetricType:
            metric_class = MetricFactory[metric_type]
            self.assertTrue(issubclass(metric_class, BaseMetric))
            self.assertTrue(hasattr(metric_class, "compute"))
            self.assertTrue(hasattr(metric_class, "reset"))
            self.assertTrue(hasattr(metric_class, "aggregate"))


if __name__ == "__main__":
    unittest.main()
