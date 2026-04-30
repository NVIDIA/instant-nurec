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
import tempfile
import unittest

import numpy as np
import torch
import yaml

from nre.metrics.impl.lpips import LPIPSMetric
from nre.metrics.impl.psnr import PSNRMetric
from nre.metrics.impl.ssim import SSIMMetric
from nre.metrics.metric import ComputeEntry, MetricManager, MetricResult, MetricStorage
from nre.metrics.utils import AggregationMethod
from nre.utils.batch import FrameMeta


class TestMetricResult(unittest.TestCase):
    def test_metric_result_get_value(self):
        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})
        self.assertEqual(metric_result.get_value("psnr"), torch.tensor(25.5))

    def test_metric_result_get_value_nonexistent(self):
        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})
        with self.assertRaises(KeyError):
            metric_result.get_value("nonexistent")

    def test_metric_result_get_available_values(self):
        metric_result = MetricResult(values={"psnr": torch.tensor(25.5), "ssim": torch.tensor(0.8)})
        available_values = metric_result.get_available_values()
        self.assertIn("psnr", available_values)
        self.assertIn("ssim", available_values)
        self.assertEqual(len(available_values), 2)

    def test_metric_result_dict_access(self):
        """Test dictionary-like access to metric values."""
        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})
        self.assertEqual(metric_result["psnr"], torch.tensor(25.5))
        self.assertTrue("psnr" in metric_result)

    def test_metric_result_to_dict(self):
        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})
        result_dict = metric_result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["psnr"], torch.tensor(25.5))

    def test_to_serializable_dict_numpy_single_element(self):
        """Test _to_primitive with single element numpy array."""
        metric_result = MetricResult(values={"score": np.array(25.5)})
        result = metric_result.to_serializable_dict()
        self.assertEqual(result["values"]["score"], 25.5)
        self.assertIsInstance(result["values"]["score"], float)

    def test_to_serializable_dict_numpy_multi_element(self):
        """Test _to_primitive with multi-element numpy array."""
        metric_result = MetricResult(values={"scores": np.array([1.0, 2.0, 3.0])})
        result = metric_result.to_serializable_dict()
        self.assertEqual(result["values"]["scores"], [1.0, 2.0, 3.0])
        self.assertIsInstance(result["values"]["scores"], list)

    def test_to_serializable_dict_numpy_2d_array(self):
        """Test _to_primitive with 2D numpy array."""
        metric_result = MetricResult(values={"matrix": np.array([[1, 2], [3, 4]])})
        result = metric_result.to_serializable_dict()
        self.assertEqual(result["values"]["matrix"], [[1, 2], [3, 4]])
        self.assertIsInstance(result["values"]["matrix"], list)

    def test_to_serializable_dict_torch_single_element(self):
        """Test _to_primitive with single element torch tensor."""
        metric_result = MetricResult(values={"score": torch.tensor(25.5)})
        result = metric_result.to_serializable_dict()
        self.assertEqual(result["values"]["score"], 25.5)
        self.assertIsInstance(result["values"]["score"], float)

    def test_to_serializable_dict_torch_multi_element(self):
        """Test _to_primitive with multi-element torch tensor."""
        metric_result = MetricResult(values={"scores": torch.tensor([1.0, 2.0, 3.0])})
        result = metric_result.to_serializable_dict()
        self.assertEqual(result["values"]["scores"], [1.0, 2.0, 3.0])
        self.assertIsInstance(result["values"]["scores"], list)

    def test_to_serializable_dict_torch_2d_tensor(self):
        """Test _to_primitive with 2D torch tensor."""
        metric_result = MetricResult(values={"matrix": torch.tensor([[1, 2], [3, 4]])})
        result = metric_result.to_serializable_dict()
        self.assertEqual(result["values"]["matrix"], [[1, 2], [3, 4]])
        self.assertIsInstance(result["values"]["matrix"], list)

    def test_to_serializable_dict_unsupported_type(self):
        """Test _to_primitive with unsupported type raises ValueError."""

        class UnsupportedType:
            pass

        metric_result = MetricResult(values={"unsupported": UnsupportedType()})
        with self.assertRaises(ValueError) as context:
            metric_result.to_serializable_dict()
        self.assertIn("Unsupported type", str(context.exception))

    def test_to_serializable_dict_lists_tuples_sets(self):
        """Test _to_primitive with lists, tuples, and sets."""
        metric_result = MetricResult(
            values={
                "list_val": [torch.tensor(1.0), np.array(2.0), 3],
                "tuple_val": (torch.tensor(4.0), np.array(5.0), 6),
                "set_val": {7, 8, 9},  # Use hashable types for set
            }
        )
        result = metric_result.to_serializable_dict()

        # Lists should preserve order and convert elements
        self.assertEqual(result["values"]["list_val"], [1.0, 2.0, 3])
        self.assertIsInstance(result["values"]["list_val"], list)

        # Tuples should become lists
        self.assertEqual(result["values"]["tuple_val"], [4.0, 5.0, 6])
        self.assertIsInstance(result["values"]["tuple_val"], list)

        # Sets should become lists (order may vary)
        set_result = set(result["values"]["set_val"])
        self.assertEqual(set_result, {7, 8, 9})
        self.assertIsInstance(result["values"]["set_val"], list)

    def test_to_serializable_dict_unsupported_type_for_metadata(self):
        """Test _to_primitive with unsupported type raises ValueError."""

        class UnsupportedType:
            pass

        metric_result = MetricResult(metadata={"unsupported": UnsupportedType()})
        with self.assertRaises(ValueError) as context:
            metric_result.to_serializable_dict(include_metadata=True)
        self.assertIn("Unsupported type", str(context.exception))

    def test_to_serializable_dict_empty_values(self):
        """Test _to_primitive with empty values."""
        metric_result = MetricResult(values={})
        result = metric_result.to_serializable_dict()
        self.assertEqual(result["values"], {})
        self.assertNotIn("metadata", result)

    def test_to_serializable_dict_none_values(self):
        """Test _to_primitive with None values."""
        metric_result = MetricResult(
            values={"none_val": None, "mixed": {"none_in_dict": None, "tensor": torch.tensor(42.0)}}
        )
        result = metric_result.to_serializable_dict()
        self.assertIsNone(result["values"]["none_val"])
        self.assertIsNone(result["values"]["mixed"]["none_in_dict"])
        self.assertEqual(result["values"]["mixed"]["tensor"], 42.0)

    def test_to_serializable_dict_with_metadata_false(self):
        """Test to_serializable_dict with include_metadata=False."""
        metric_result = MetricResult(
            values={"score": torch.tensor(25.5)}, metadata={"data_range": 255.0, "input_shape": [1024, 1024]}
        )
        result = metric_result.to_serializable_dict(include_metadata=False)
        self.assertIn("values", result)
        self.assertNotIn("metadata", result)
        self.assertEqual(result["values"]["score"], 25.5)

    def test_to_serializable_dict_with_metadata_true(self):
        """Test to_serializable_dict with include_metadata=True."""
        metric_result = MetricResult(
            values={"score": torch.tensor(25.5)},
            metadata={
                "data_range": 255.0,
                "input_shape": [1024, 1024],
                "nested": {"tensor": torch.tensor(42.0), "array": np.array([1, 2, 3])},
            },
        )
        result = metric_result.to_serializable_dict(include_metadata=True)
        self.assertIn("values", result)
        self.assertIn("metadata", result)
        self.assertEqual(result["values"]["score"], 25.5)
        self.assertEqual(result["metadata"]["data_range"], 255.0)
        self.assertEqual(result["metadata"]["input_shape"], [1024, 1024])
        self.assertEqual(result["metadata"]["nested"]["tensor"], 42.0)
        self.assertEqual(result["metadata"]["nested"]["array"], [1, 2, 3])

    def test_to_serializable_dict_metadata_with_tensors(self):
        """Test to_serializable_dict with metadata containing tensors and arrays."""
        metric_result = MetricResult(
            values={"psnr": torch.tensor(25.5)},
            metadata={
                "tensor_metadata": torch.tensor([1.0, 2.0, 3.0]),
                "array_metadata": np.array([4.0, 5.0, 6.0]),
                "nested_tensors": {"inner_tensor": torch.tensor(7.0), "inner_array": np.array(8.0)},
            },
        )
        result = metric_result.to_serializable_dict(include_metadata=True)

        self.assertEqual(result["metadata"]["tensor_metadata"], [1.0, 2.0, 3.0])
        self.assertEqual(result["metadata"]["array_metadata"], [4.0, 5.0, 6.0])
        self.assertEqual(result["metadata"]["nested_tensors"]["inner_tensor"], 7.0)
        self.assertEqual(result["metadata"]["nested_tensors"]["inner_array"], 8.0)

    def test_to_serializable_dict_metadata_with_collections(self):
        """Test to_serializable_dict with metadata containing collections."""
        metric_result = MetricResult(
            values={"ssim": torch.tensor(0.8)},
            metadata={
                "list_metadata": [torch.tensor(1.0), np.array(2.0), 3],
                "tuple_metadata": (torch.tensor(4.0), np.array(5.0), 6),
                "set_metadata": {7, 8, 9},  # Use frozenset instead of set with unhashable types
                "dict_metadata": {
                    "list": [torch.tensor(10.0), np.array(11.0)],
                    "tuple": (torch.tensor(12.0), np.array(13.0)),
                },
            },
        )
        result = metric_result.to_serializable_dict(include_metadata=True)

        self.assertEqual(result["metadata"]["list_metadata"], [1.0, 2.0, 3])
        self.assertEqual(result["metadata"]["tuple_metadata"], [4.0, 5.0, 6])
        self.assertEqual(result["metadata"]["set_metadata"], list({7, 8, 9}))
        self.assertEqual(result["metadata"]["dict_metadata"]["list"], [10.0, 11.0])
        self.assertEqual(result["metadata"]["dict_metadata"]["tuple"], [12.0, 13.0])

    def test_to_serializable_dict_metadata_with_primitives(self):
        """Test to_serializable_dict with metadata containing only primitives."""
        metric_result = MetricResult(
            values={"lpips": torch.tensor(0.15)},
            metadata={
                "string_meta": "test_string",
                "int_meta": 42,
                "float_meta": 3.14159,
                "bool_meta": True,
                "none_meta": None,
            },
        )
        result = metric_result.to_serializable_dict(include_metadata=True)

        self.assertEqual(result["metadata"]["string_meta"], "test_string")
        self.assertEqual(result["metadata"]["int_meta"], 42)
        self.assertEqual(result["metadata"]["float_meta"], 3.14159)
        self.assertEqual(result["metadata"]["bool_meta"], True)
        self.assertIsNone(result["metadata"]["none_meta"])

    def test_to_serializable_dict_empty_metadata(self):
        """Test to_serializable_dict with empty metadata."""
        metric_result = MetricResult(values={"score": torch.tensor(25.5)}, metadata={})
        result = metric_result.to_serializable_dict(include_metadata=True)
        self.assertEqual(result["metadata"], {})

    def test_to_serializable_dict_deep_nesting(self):
        """Test to_serializable_dict with deeply nested structures."""
        metric_result = MetricResult(
            values={
                "deep_nested": {
                    "level1": {
                        "level2": {
                            "level3": {
                                "tensor": torch.tensor(42.0),
                                "array": np.array([1, 2, 3]),
                                "list": [torch.tensor(4.0), np.array(5.0)],
                                "dict": {"final": torch.tensor(6.0)},
                            }
                        }
                    }
                }
            }
        )
        result = metric_result.to_serializable_dict()

        expected = {
            "deep_nested": {
                "level1": {
                    "level2": {
                        "level3": {"tensor": 42.0, "array": [1, 2, 3], "list": [4.0, 5.0], "dict": {"final": 6.0}}
                    }
                }
            }
        }
        self.assertEqual(result["values"], expected)

    def test_to_serializable_dict_mixed_types_in_collections(self):
        """Test to_serializable_dict with mixed types in collections."""
        metric_result = MetricResult(
            values={
                "mixed_list": [
                    torch.tensor(1.0),
                    np.array(2.0),
                    3,
                    4.5,
                    "string",
                    True,
                    [torch.tensor(6.0), np.array(7.0)],
                    {"nested": torch.tensor(8.0)},
                ]
            }
        )
        result = metric_result.to_serializable_dict()

        expected = [1.0, 2.0, 3, 4.5, "string", True, [6.0, 7.0], {"nested": 8.0}]
        self.assertEqual(result["values"]["mixed_list"], expected)

    def test_to_serializable_dict_torch_tensor_different_dtypes(self):
        """Test _to_primitive with torch tensors of different dtypes."""
        metric_result = MetricResult(
            values={
                "float32": torch.tensor(25.5, dtype=torch.float32),
                "float64": torch.tensor(25.5, dtype=torch.float64),
                "int32": torch.tensor(42, dtype=torch.int32),
                "int64": torch.tensor(42, dtype=torch.int64),
                "bool": torch.tensor(True, dtype=torch.bool),
            }
        )
        result = metric_result.to_serializable_dict()

        self.assertEqual(result["values"]["float32"], 25.5)
        self.assertEqual(result["values"]["float64"], 25.5)
        self.assertEqual(result["values"]["int32"], 42)
        self.assertEqual(result["values"]["int64"], 42)
        self.assertEqual(result["values"]["bool"], True)

    def test_to_serializable_dict_numpy_array_different_dtypes(self):
        """Test _to_primitive with numpy arrays of different dtypes."""
        metric_result = MetricResult(
            values={
                "float32": np.array(25.5, dtype=np.float32),
                "float64": np.array(25.5, dtype=np.float64),
                "int32": np.array(42, dtype=np.int32),
                "int64": np.array(42, dtype=np.int64),
                "bool": np.array(True, dtype=np.bool_),
            }
        )
        result = metric_result.to_serializable_dict()

        self.assertEqual(result["values"]["float32"], 25.5)
        self.assertEqual(result["values"]["float64"], 25.5)
        self.assertEqual(result["values"]["int32"], 42)
        self.assertEqual(result["values"]["int64"], 42)
        self.assertEqual(result["values"]["bool"], True)


class TestMetricManager(unittest.TestCase):
    def test_metric_manager_register_metric(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)
        self.assertIn("psnr", metric_manager.list_metrics())

    def test_metric_manager_register_runtime_metric(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)
        self.assertEqual(metric_manager.get_metric("psnr"), psnr_metric)

    def test_device_handling(self):
        metric_manager = MetricManager(device=torch.device("cuda"))
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)
        self.assertEqual(psnr_metric.device, torch.device("cuda"))

    def test_metric_manager_list_metrics(self):
        metric_manager = MetricManager()
        psnr_metric1 = PSNRMetric(data_range=1.0)
        psnr_metric2 = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr1", psnr_metric1)
        metric_manager.register_metric("psnr2", psnr_metric2)
        metrics_list = metric_manager.list_metrics()
        self.assertIn("psnr1", metrics_list)
        self.assertIn("psnr2", metrics_list)
        self.assertEqual(len(metrics_list), 2)

    def test_metric_manager_aggregate_all(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(30.0, dtype=torch.float32)}))
        metric_manager.register_metric("psnr", psnr_metric)
        aggregated = metric_manager.aggregate()

        self.assertIn("psnr", aggregated)
        self.assertIn("psnr", aggregated["psnr"][AggregationMethod.MEAN].values)

    def test_metric_manager_aggregate_multiple_metrics(self):
        metric_manager = MetricManager()
        psnr_metric1 = PSNRMetric(data_range=1.0)
        psnr_metric2 = PSNRMetric(data_range=1.0)

        psnr_metric1.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))
        psnr_metric2.append(MetricResult(values={"psnr": torch.tensor(30.0, dtype=torch.float32)}))

        metric_manager.register_metric("psnr1", psnr_metric1)
        metric_manager.register_metric("psnr2", psnr_metric2)

        aggregated = metric_manager.aggregate(["psnr1", "psnr2"])

        self.assertIn("psnr1", aggregated)
        self.assertIn("psnr2", aggregated)
        self.assertIn("psnr", aggregated["psnr1"][AggregationMethod.MEAN].values)
        self.assertIn("psnr", aggregated["psnr2"][AggregationMethod.MEAN].values)

    def test_metric_manager_aggregate_nonexistent_metric(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        with self.assertRaises(KeyError):
            metric_manager.aggregate("nonexistent")

    def test_metric_manager_aggregate_multiple_methods(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0, aggregation_methods=[AggregationMethod.MEAN, AggregationMethod.SUM])
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(30.0, dtype=torch.float32)}))

        metric_manager.register_metric("psnr", psnr_metric)
        aggregated = metric_manager.aggregate()

        self.assertIn("psnr", aggregated)
        self.assertIn("psnr", aggregated["psnr"][AggregationMethod.MEAN].values)
        self.assertIn("psnr", aggregated["psnr"][AggregationMethod.SUM].values)

    def test_metric_manager_register_ssim_metric(self):
        """Test registering and using SSIM metric with MetricManager."""
        metric_manager = MetricManager()
        ssim_metric = SSIMMetric(data_range=1.0)
        metric_manager.register_metric("ssim", ssim_metric)
        self.assertIn("ssim", metric_manager.list_metrics())
        self.assertEqual(metric_manager.get_metric("ssim"), ssim_metric)

    def test_metric_manager_compute_ssim(self):
        """Test computing SSIM metric via MetricManager."""
        metric_manager = MetricManager()
        ssim_metric = SSIMMetric(data_range=1.0)
        metric_manager.register_metric("ssim", ssim_metric)

        # Create test tensors (values in [0, 1] range)
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)

        metric_manager.compute("ssim", pred, target)
        last_metric = metric_manager.get_last("ssim")

        self.assertIsNotNone(last_metric)
        if last_metric is not None:
            self.assertIn("ssim", last_metric.values)

    def test_metric_manager_aggregate_ssim(self):
        """Test aggregating SSIM metric results."""
        metric_manager = MetricManager()
        ssim_metric = SSIMMetric(data_range=1.0)
        ssim_metric.append(MetricResult(values={"ssim": torch.tensor(0.85, dtype=torch.float32)}))
        ssim_metric.append(MetricResult(values={"ssim": torch.tensor(0.90, dtype=torch.float32)}))

        metric_manager.register_metric("ssim", ssim_metric)
        aggregated = metric_manager.aggregate()

        self.assertIn("ssim", aggregated)
        self.assertIn("ssim", aggregated["ssim"][AggregationMethod.MEAN].values)

    def test_metric_manager_register_lpips_metric(self):
        """Test registering and using LPIPS metric with MetricManager."""
        metric_manager = MetricManager()
        lpips_metric = LPIPSMetric(device="cpu", normalize=True)
        metric_manager.register_metric("lpips", lpips_metric)
        self.assertIn("lpips", metric_manager.list_metrics())
        self.assertEqual(metric_manager.get_metric("lpips"), lpips_metric)

    def test_metric_manager_compute_lpips(self):
        """Test computing LPIPS metric via MetricManager."""
        metric_manager = MetricManager()
        lpips_metric = LPIPSMetric(device="cpu", normalize=True)
        metric_manager.register_metric("lpips", lpips_metric)

        # Create test tensors (values in [0, 1] range, 3 channels required)
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)

        metric_manager.compute("lpips", pred, target)
        last_metric = metric_manager.get_last("lpips")

        self.assertIsNotNone(last_metric)
        if last_metric is not None:
            self.assertIn("lpips", last_metric.values)

    def test_metric_manager_aggregate_lpips(self):
        """Test aggregating LPIPS metric results."""
        metric_manager = MetricManager()
        lpips_metric = LPIPSMetric(device="cpu", normalize=True)
        lpips_metric.append(MetricResult(values={"lpips": torch.tensor(0.15, dtype=torch.float32)}))
        lpips_metric.append(MetricResult(values={"lpips": torch.tensor(0.20, dtype=torch.float32)}))

        metric_manager.register_metric("lpips", lpips_metric)
        aggregated = metric_manager.aggregate()

        self.assertIn("lpips", aggregated)
        self.assertIn("lpips", aggregated["lpips"][AggregationMethod.MEAN].values)

    def test_metric_manager_multiple_metric_types(self):
        """Test registering and computing multiple metric types (PSNR, SSIM, LPIPS)."""
        metric_manager = MetricManager()

        psnr_metric = PSNRMetric(data_range=1.0)
        ssim_metric = SSIMMetric(data_range=1.0)
        lpips_metric = LPIPSMetric(device="cpu", normalize=True)

        metric_manager.register_metric("psnr", psnr_metric)
        metric_manager.register_metric("ssim", ssim_metric)
        metric_manager.register_metric("lpips", lpips_metric)

        # Create test tensors
        pred = torch.rand(1, 3, 64, 64)
        target = torch.rand(1, 3, 64, 64)

        # Compute all metrics
        metric_manager.compute("psnr", pred, target)
        metric_manager.compute("ssim", pred, target)
        metric_manager.compute("lpips", pred, target)

        # Verify all metrics are computed
        metrics_list = metric_manager.list_metrics()
        self.assertIn("psnr", metrics_list)
        self.assertIn("ssim", metrics_list)
        self.assertIn("lpips", metrics_list)

        # Verify last results
        psnr_result = metric_manager.get_last("psnr")
        ssim_result = metric_manager.get_last("ssim")
        lpips_result = metric_manager.get_last("lpips")

        self.assertIsNotNone(psnr_result)
        self.assertIsNotNone(ssim_result)
        self.assertIsNotNone(lpips_result)

    def test_metric_manager_reset_all(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))

        metric_manager.register_metric("psnr", psnr_metric)
        metric_manager.reset()

        self.assertEqual(len(psnr_metric), 1)

    def test_metric_manager_reset_single_metric(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))

        metric_manager.register_metric("psnr", psnr_metric)
        metric_manager.reset("psnr")

        self.assertEqual(len(psnr_metric), 1)

    def test_metric_manager_reset_multiple_metrics(self):
        metric_manager = MetricManager()
        psnr_metric1 = PSNRMetric(data_range=1.0)
        psnr_metric2 = PSNRMetric(data_range=1.0)

        psnr_metric1.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))
        psnr_metric2.append(MetricResult(values={"psnr": torch.tensor(30.0, dtype=torch.float32)}))

        metric_manager.register_metric("psnr1", psnr_metric1)
        metric_manager.register_metric("psnr2", psnr_metric2)

        metric_manager.reset(["psnr1", "psnr2"])
        self.assertEqual(len(psnr_metric1), 1)
        self.assertEqual(len(psnr_metric2), 1)

    def test_metric_manager_reset_nonexistent_metric(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        with self.assertRaises(KeyError):
            metric_manager.reset("nonexistent")

    def test_metric_manager_clear_all(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))

        metric_manager.register_metric("psnr", psnr_metric)
        metric_manager.clear()

        self.assertEqual(len(psnr_metric), 0)

    def test_metric_manager_clear_single_metric(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))

        metric_manager.register_metric("psnr", psnr_metric)
        metric_manager.clear("psnr")

        self.assertEqual(len(psnr_metric), 0)

    def test_metric_manager_clear_multiple_metrics(self):
        metric_manager = MetricManager()
        psnr_metric1 = PSNRMetric(data_range=1.0)
        psnr_metric2 = PSNRMetric(data_range=1.0)

        psnr_metric1.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))
        psnr_metric2.append(MetricResult(values={"psnr": torch.tensor(30.0, dtype=torch.float32)}))

        metric_manager.register_metric("psnr1", psnr_metric1)
        metric_manager.register_metric("psnr2", psnr_metric2)

        # Test clear specific metrics by list
        metric_manager.clear(["psnr1", "psnr2"])
        self.assertEqual(len(psnr_metric1), 0)
        self.assertEqual(len(psnr_metric2), 0)

    def test_metric_manager_clear_nonexistent_metric(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Test clear non-existent metric
        with self.assertRaises(KeyError):
            metric_manager.clear("nonexistent")
        with self.assertRaises(KeyError):
            metric_manager.clear(["psnr", "nonexistent"])

    def test_metric_manager_get_last(self):
        def dummy_function():
            # Create dummy image data for PSNR computation
            pred = torch.randn(1, 3, 64, 64)
            target = torch.randn(1, 3, 64, 64)
            return pred, target

        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)
        metric_manager.compute("psnr", torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)
        if last_metric is not None:
            assert last_metric.values["psnr"] is not None

    def test_metric_manager_get_all(self):
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(25.5, dtype=torch.float32)}))
        psnr_metric.append(MetricResult(values={"psnr": torch.tensor(30.0, dtype=torch.float32)}))

        all_metrics = metric_manager.get_all("psnr")
        self.assertEqual(len(all_metrics), 2)
        self.assertAlmostEqual(all_metrics[0]["psnr"].item(), 25.5)
        self.assertAlmostEqual(all_metrics[1]["psnr"].item(), 30.0)

    def test_metric_manager_compute_with_compute_entry_basic(self):
        """Test compute method with ComputeEntry using basic parameters."""
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Create ComputeEntry with basic parameters
        compute_entry = ComputeEntry(name="psnr")
        metric_manager.compute(compute_entry, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)
        if last_metric is not None:
            self.assertIn("psnr", last_metric.values)

    def test_metric_manager_compute_with_compute_entry_with_metadata(self):
        """Test compute method with ComputeEntry including metadata."""
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Create ComputeEntry with metadata using the new generic approach
        compute_entry = ComputeEntry(
            name="psnr",
            metadata={"frame_meta": FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)},
            include_metadata=True,
        )
        metric_manager.compute(compute_entry, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)

    def test_metric_manager_compute_with_compute_entry_with_sequence_id(self):
        """Test compute method with ComputeEntry including sequence_id."""
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Create ComputeEntry with sequence_id using the new generic approach
        compute_entry = ComputeEntry(name="psnr", metadata={"sequence_id": "test_sequence_001"})
        metric_manager.compute(compute_entry, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)

    def test_metric_manager_compute_with_compute_entry_with_sequence_id_list(self):
        """Test compute method with ComputeEntry including sequence_id as list."""
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Create ComputeEntry with sequence_id as list using the new generic approach
        compute_entry = ComputeEntry(name="psnr", metadata={"sequence_id": ["seq1", "seq2", "seq3"]})
        metric_manager.compute(compute_entry, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)

    def test_metric_manager_compute_with_compute_entry_with_datasource(self):
        """Test compute method with ComputeEntry including datasource."""
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Create ComputeEntry with datasource using the new generic approach
        compute_entry = ComputeEntry(name="psnr", metadata={"datasource": None})
        metric_manager.compute(compute_entry, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)

    def test_metric_manager_compute_with_compute_entry_complete(self):
        """Test compute method with ComputeEntry including all parameters."""
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Create ComputeEntry with all parameters using the new generic approach
        compute_entry = ComputeEntry(
            name="psnr",
            metadata={
                "datasource": None,  # None for testing
                "frame_meta": FrameMeta(unique_sensor_idx=0, unique_frame_idx=0),
                "sequence_id": "test_sequence_001",
            },
            include_metadata=True,
        )
        metric_manager.compute(compute_entry, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)

    def test_metric_manager_compute_with_compute_entry_include_metadata_false(self):
        """Test compute method with ComputeEntry with include_metadata=False."""
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Create ComputeEntry with include_metadata=False
        compute_entry = ComputeEntry(name="psnr", include_metadata=False)
        metric_manager.compute(compute_entry, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)

    def test_metric_manager_compute_with_compute_entry_nonexistent_metric(self):
        """Test compute method with ComputeEntry for nonexistent metric raises error."""
        metric_manager = MetricManager()

        # Create ComputeEntry for nonexistent metric
        compute_entry = ComputeEntry(name="nonexistent_metric")

        with self.assertRaises(KeyError):
            metric_manager.compute(compute_entry, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

    def test_metric_manager_compute_backward_compatibility_string(self):
        """Test that compute method still works with string parameter (backward compatibility)."""
        metric_manager = MetricManager()
        psnr_metric = PSNRMetric(data_range=1.0)
        metric_manager.register_metric("psnr", psnr_metric)

        # Use string parameter (old way)
        metric_manager.compute("psnr", torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        last_metric = metric_manager.get_last("psnr")
        self.assertIsNotNone(last_metric)
        if last_metric is not None:
            self.assertIn("psnr", last_metric.values)

    def test_metric_manager_compute_compute_entry_vs_string_equivalence(self):
        """Test that compute with ComputeEntry and string produce equivalent results."""
        metric_manager1 = MetricManager()
        metric_manager2 = MetricManager()

        psnr_metric1 = PSNRMetric(data_range=1.0)
        psnr_metric2 = PSNRMetric(data_range=1.0)

        metric_manager1.register_metric("psnr", psnr_metric1)
        metric_manager2.register_metric("psnr", psnr_metric2)

        # Create test data
        pred = torch.randn(1, 3, 64, 64)
        target = torch.randn(1, 3, 64, 64)

        # Compute using string
        metric_manager1.compute("psnr", pred, target)

        # Compute using ComputeEntry
        compute_entry = ComputeEntry(name="psnr")
        metric_manager2.compute(compute_entry, pred, target)

        # Both should have results
        result1 = metric_manager1.get_last("psnr")
        result2 = metric_manager2.get_last("psnr")

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        if result1 is not None and result2 is not None:
            self.assertIn("psnr", result1.values)
            self.assertIn("psnr", result2.values)

    def test_compute_entry_generic_metadata_capabilities(self):
        """Test the new generic metadata capabilities of ComputeEntry."""
        # Test with custom metadata keys
        compute_entry = ComputeEntry(
            name="test_metric",
            metadata={
                "custom_key": "custom_value",
                "numeric_value": 42,
                "nested_dict": {"inner_key": "inner_value"},
                "list_value": [1, 2, 3],
            },
        )

        # Test custom metadata access
        self.assertEqual(compute_entry.get("custom_key"), "custom_value")
        self.assertEqual(compute_entry.get("numeric_value"), 42)
        self.assertEqual(compute_entry.get("nested_dict"), {"inner_key": "inner_value"})
        self.assertEqual(compute_entry.get("list_value"), [1, 2, 3])
        self.assertIsNone(compute_entry.get("nonexistent_key"))
        self.assertEqual(compute_entry.get("nonexistent_key", "default"), "default")

        # Test setting new metadata
        compute_entry.set("new_key", "new_value")
        self.assertEqual(compute_entry.get("new_key"), "new_value")

        # Test backward compatibility properties
        self.assertIsNone(compute_entry.get("datasource"))
        self.assertIsNone(compute_entry.get("rays_metadata"))
        self.assertIsNone(compute_entry.get("sequence_id"))

    def test_compute_entry_empty_metadata_initialization(self):
        """Test ComputeEntry initialization with empty metadata."""
        compute_entry = ComputeEntry(name="test_metric")

        # Should be None metadata dict
        self.assertEqual(compute_entry.metadata, None)

        # Properties should return None
        self.assertIsNone(compute_entry.get("datasource"))
        self.assertIsNone(compute_entry.get("rays_metadata"))
        self.assertIsNone(compute_entry.get("sequence_id"))

        # Should be able to add metadata
        compute_entry.set("key", "value")
        self.assertEqual(compute_entry.get("key"), "value")

    def test_compute_entry_metadata_persistence(self):
        """Test that metadata persists through operations."""
        compute_entry = ComputeEntry(name="test_metric")

        # Add some metadata
        compute_entry.set("key1", "value1")
        compute_entry.set("key2", "value2")

        # Verify persistence
        self.assertEqual(compute_entry.get("key1"), "value1")
        self.assertEqual(compute_entry.get("key2"), "value2")

        # Update existing metadata
        compute_entry.set("key1", "updated_value")
        self.assertEqual(compute_entry.get("key1"), "updated_value")
        self.assertEqual(compute_entry.get("key2"), "value2")  # Should be unchanged


class TestMetricStorage(unittest.TestCase):
    """Test the MetricStorage class functionality."""

    def setUp(self):
        self.storage = MetricStorage()

    def test_metric_storage_initialization(self):
        """Test MetricStorage initialization."""
        self.assertIsInstance(self.storage.metrics_storage, dict)
        self.assertIn("metadata", self.storage.metrics_storage)
        self.assertIn("metrics", self.storage.metrics_storage)
        self.assertIn("program_version", self.storage.metrics_storage["metadata"])
        self.assertIn("run_info", self.storage.metrics_storage["metadata"])

    def test_initialize_metric_entry(self):
        """Test adding metric entries to storage."""
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {"data_range": 255.0})

        self.assertIn("test_metric", self.storage.metrics_storage["metrics"])
        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(entry["metric"], "psnr")
        self.assertEqual(entry["metadata"]["data_range"], 255.0)
        self.assertEqual(entry["aggregated_results"], [])
        self.assertEqual(entry["metric_results"], [])

    def test_add_aggregated_entry(self):
        """Test adding aggregated entries to storage."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})

        self.storage.add_aggregated_entry("test_metric", {"psnr": 25.5}, AggregationMethod.MEAN)

        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(entry["aggregated_results"][0]["result"]["psnr"], 25.5)
        self.assertEqual(entry["aggregated_results"][0]["method"], "mean")

    def test_add_aggregated_entry_multiple_methods(self):
        """Test adding multiple aggregated entries with different methods."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})

        self.storage.add_aggregated_entry("test_metric", {"psnr": 25.5}, AggregationMethod.MEAN)
        self.storage.add_aggregated_entry("test_metric", {"psnr": 30.0}, AggregationMethod.SUM)

        # Should have both entries
        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(len(entry["aggregated_results"]), 2)
        self.assertEqual(entry["aggregated_results"][0]["result"]["psnr"], 25.5)
        self.assertEqual(entry["aggregated_results"][0]["method"], "mean")
        self.assertEqual(entry["aggregated_results"][1]["result"]["psnr"], 30.0)
        self.assertEqual(entry["aggregated_results"][1]["method"], "sum")

    def test_process_sequence_id_none(self):
        """Test processing None sequence ID."""
        result = self.storage._process_sequence_id(None)
        self.assertEqual(result, "")

    def test_process_sequence_id_string(self):
        """Test processing string sequence ID."""
        result = self.storage._process_sequence_id("test_sequence")
        self.assertEqual(result, "test_sequence")

    def test_process_sequence_id_list(self):
        """Test processing list sequence ID."""
        result = self.storage._process_sequence_id(["seq1", "seq2", "seq3"])
        self.assertEqual(result, "seq1+seq2+seq3")

    def test_process_unique_sensor_id_none(self):
        """Test processing None unique sensor ID."""
        result = self.storage._process_unique_sensor_id(None, "test_sequence")
        self.assertEqual(result, "")

    def test_process_unique_sensor_id_no_sequence(self):
        """Test processing unique sensor ID without sequence ID."""
        result = self.storage._process_unique_sensor_id("camera_front", None)
        self.assertEqual(result, "camera_front")

    def test_process_unique_sensor_id_with_sequence_removal(self):
        """Test processing unique sensor ID with sequence ID removal."""
        unique_id = "camera_front@test_sequence"
        sequence_id = "test_sequence"
        result = self.storage._process_unique_sensor_id(unique_id, sequence_id)
        self.assertEqual(result, "camera_front")

    def test_process_unique_sensor_id_with_sequence_no_removal(self):
        """Test processing unique sensor ID where sequence ID doesn't match suffix."""
        unique_id = "camera_front"
        sequence_id = "test_sequence"
        result = self.storage._process_unique_sensor_id(unique_id, sequence_id)
        self.assertEqual(result, "camera_front")

    def test_add_entry_general(self):
        """Test adding general entry without metadata or sequence ID."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})

        self.storage.add_entry("test_metric", {"psnr": 25.5})

        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(entry["metric_results"][0]["psnr"], 25.5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "")
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["unique_sensor_id"], "")

    def test_add_entry_with_metadata(self):
        """Test adding entry with metadata."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})

        self.storage.add_entry(
            "test_metric", {"psnr": 25.5}, frame_meta=FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)
        )

        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(entry["metric_results"][0]["psnr"], 25.5)
        # Note: timestamp fields are only added if timestamp_us is not None
        # The mock metadata has timestamp_us=None, so these fields won't be present
        self.assertIn("unique_frame_idx", entry["metric_results"][0]["sensor_data"])

    def test_add_entry_with_sequence_id(self):
        """Test adding entry with sequence ID."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})

        self.storage.add_entry("test_metric", {"psnr": 25.5}, sequence_id="test_sequence")

        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(entry["metric_results"][0]["psnr"], 25.5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "test_sequence")
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["unique_sensor_id"], "")

    def test_add_entry_with_unique_sensor_id(self):
        """Test adding entry with unique sensor ID."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})

        self.storage.add_entry("test_metric", {"psnr": 25.5}, unique_sensor_id="camera_front")

        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(entry["metric_results"][0]["psnr"], 25.5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "")
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["unique_sensor_id"], "camera_front")

    def test_add_entry_with_sequence_and_sensor_id(self):
        """Test adding entry with both sequence ID and unique sensor ID."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})

        self.storage.add_entry(
            "test_metric", {"psnr": 25.5}, sequence_id="test_sequence", unique_sensor_id="camera_front"
        )

        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(entry["metric_results"][0]["psnr"], 25.5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "test_sequence")
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["unique_sensor_id"], "camera_front")

    def test_add_entry_with_metadata_and_sequence(self):
        """Test adding entry with metadata and sequence ID."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})

        self.storage.add_entry(
            "test_metric",
            {"psnr": 25.5},
            frame_meta=FrameMeta(unique_sensor_idx=0, unique_frame_idx=0),
            sequence_id="test_sequence",
        )

        entry = self.storage.metrics_storage["metrics"]["test_metric"]
        self.assertEqual(entry["metric_results"][0]["psnr"], 25.5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "test_sequence")
        # The unique_sensor_id should be empty string when not provided
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["unique_sensor_id"], "")
        # Note: timestamp fields are only added if timestamp_us is not None
        self.assertIn("unique_frame_idx", entry["metric_results"][0]["sensor_data"])

    def test_add_entry_nonexistent_metric(self):
        """Test adding entry to non-existent metric raises error."""
        with self.assertRaises(KeyError):
            self.storage.add_entry("nonexistent_metric", {"psnr": 25.5})

    def test_write_metrics_to_yaml(self):
        """Test writing metrics to YAML file."""
        # Add some test data
        from nre.metrics.types import MetricType

        self.storage.initialize_metric("test_metric", MetricType.PSNR, {})
        self.storage.add_aggregated_entry("test_metric", {"psnr": 25.5}, AggregationMethod.MEAN)
        self.storage.add_entry("test_metric", {"psnr": 25.5}, sequence_id="test_sequence")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.storage.write_metrics(temp_dir, ext="yaml")

            metrics_file = os.path.join(temp_dir, "metrics.yaml")
            self.assertTrue(os.path.exists(metrics_file))

            with open(metrics_file, "r") as f:
                content = yaml.safe_load(f)

                # Check structure
                self.assertIn("metrics", content)
                self.assertIn("metadata", content)
                self.assertIn("program_version", content["metadata"])
                self.assertIn("run_info", content["metadata"])


class TestMetricManagerCollectMethods(unittest.TestCase):
    """Test the collect_metric and related methods in MetricManager."""

    def setUp(self):
        self.metric_manager = MetricManager()
        self.mock_frame_meta = FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)

    def test_collect_metric_basic(self):
        """Test basic metric collection without metadata."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/psnr", MetricType.PSNR, {})

        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})

        self.metric_manager.collect_metric("test/psnr", metric_result)

        # Verify the metric was collected by checking storage
        # collect_metric creates extended names like "test/psnr/psnr"
        self.assertIn("test/psnr", self.metric_manager._storage.metrics_storage["metrics"])

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/psnr"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertEqual(entry["metric_results"][0]["values"]["psnr"], 25.5)

    def test_collect_metric_with_metadata(self):
        """Test collecting metrics with metadata."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/ssim", MetricType.PSNR, {})

        metric_result = MetricResult(values={"ssim": torch.tensor(0.8)})

        self.metric_manager.collect_metric(
            "test/ssim", metric_result, frame_meta=FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)
        )

        # Should be stored under the metric name with sensor data
        self.assertIn("test/ssim", self.metric_manager._storage.metrics_storage["metrics"])

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/ssim"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertAlmostEqual(entry["metric_results"][0]["values"]["ssim"], 0.8, places=5)
        self.assertIn("unique_frame_idx", entry["metric_results"][0]["sensor_data"])

    def test_collect_metric_with_sequence_id(self):
        """Test collecting metrics with sequence ID."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/lpips", MetricType.PSNR, {})

        metric_result = MetricResult(values={"lpips": torch.tensor(0.15)})

        self.metric_manager.collect_metric("test/lpips", metric_result, sequence_id="test_sequence")

        self.assertIn("test/lpips", self.metric_manager._storage.metrics_storage["metrics"])

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/lpips"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertAlmostEqual(entry["metric_results"][0]["values"]["lpips"], 0.15, places=5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "test_sequence")

    def test_collect_metric_with_metadata_and_sequence(self):
        """Test collecting metrics with metadata and sequence ID."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/accuracy", MetricType.PSNR, {})

        metric_result = MetricResult(values={"accuracy": torch.tensor(0.95)})

        self.metric_manager.collect_metric(
            "test/accuracy", metric_result, frame_meta=self.mock_frame_meta, sequence_id="test_sequence"
        )

        self.assertIn("test/accuracy", self.metric_manager._storage.metrics_storage["metrics"])

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/accuracy"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertAlmostEqual(entry["metric_results"][0]["values"]["accuracy"], 0.95, places=5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "test_sequence")

    def test_collect_metric_with_lidar_metadata(self):
        """Test collecting metrics with LiDAR metadata."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/distance_error", MetricType.PSNR, {})

        metric_result = MetricResult(values={"distance_error": torch.tensor(0.1)})

        self.metric_manager.collect_metric(
            "test/distance_error",
            metric_result,
            is_lidar=True,
            frame_meta=self.mock_frame_meta,
            sequence_id="lidar_sequence",
        )

        self.assertIn("test/distance_error", self.metric_manager._storage.metrics_storage["metrics"])

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/distance_error"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertAlmostEqual(entry["metric_results"][0]["values"]["distance_error"], 0.1, places=5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "lidar_sequence")

    def test_collect_metric_empty_result(self):
        """Test that collecting empty MetricResult raises error."""
        empty_result = MetricResult()

        with self.assertRaises(ValueError) as context:
            self.metric_manager.collect_metric("test/metric", empty_result)

        self.assertIn("MetricResult has no values to collect", str(context.exception))

    def test_collect_metric_multiple_values(self):
        """Test collecting MetricResult with multiple values."""
        # First register the metrics
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/quality", MetricType.PSNR, {})

        metric_result = MetricResult(
            values={"psnr": torch.tensor(25.5), "ssim": torch.tensor(0.8), "lpips": torch.tensor(0.15)}
        )

        self.metric_manager.collect_metric("test/quality", metric_result)

        # Should create separate entries for each value
        self.assertIn("test/quality", self.metric_manager._storage.metrics_storage["metrics"])
        self.assertIn("test/quality", self.metric_manager._storage.metrics_storage["metrics"])
        self.assertIn("test/quality", self.metric_manager._storage.metrics_storage["metrics"])

        # Check values
        self.assertEqual(
            self.metric_manager._storage.metrics_storage["metrics"]["test/quality"]["metric_results"][0]["values"][
                "psnr"
            ],
            25.5,
        )
        self.assertAlmostEqual(
            self.metric_manager._storage.metrics_storage["metrics"]["test/quality"]["metric_results"][0]["values"][
                "ssim"
            ],
            0.8,
            places=5,
        )
        self.assertAlmostEqual(
            self.metric_manager._storage.metrics_storage["metrics"]["test/quality"]["metric_results"][0]["values"][
                "lpips"
            ],
            0.15,
            places=5,
        )

    def test_collect_aggregated_metric(self):
        """Test collecting aggregated metrics."""
        metric_result = MetricResult(values={"mean": torch.tensor(25.5)})

        self.metric_manager.register_metric("test/psnr", PSNRMetric(data_range=1.0))

        self.metric_manager.collect_aggregated_metric("test/psnr", metric_result, AggregationMethod.MEAN)

        # The aggregated metrics are stored in the metrics_storage structure
        # We need to check if the metric was registered and aggregated results were added
        self.assertIn("test/psnr", self.metric_manager._storage.metrics_storage["metrics"])
        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/psnr"]
        self.assertEqual(len(entry["aggregated_results"]), 1)
        self.assertEqual(entry["aggregated_results"][0]["result"]["values"]["mean"], 25.5)
        self.assertEqual(entry["aggregated_results"][0]["method"], "mean")

    def test_collect_aggregated_metric_multiple_values(self):
        """Test collecting aggregated MetricResult with multiple values."""
        # First register the metrics
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/psnr", MetricType.PSNR, {})

        metric_result = MetricResult(
            values={
                "mean": torch.tensor(25.5),
                "std": torch.tensor(2.1),
                "min": torch.tensor(20.0),
                "max": torch.tensor(30.0),
            }
        )

        self.metric_manager.collect_aggregated_metric("test/psnr", metric_result, AggregationMethod.MEAN)

        # Should create separate entries for each value
        self.assertIn("test/psnr", self.metric_manager._storage.metrics_storage["metrics"])

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/psnr"]
        self.assertEqual(len(entry["aggregated_results"]), 1)
        self.assertEqual(entry["aggregated_results"][0]["method"], "mean")

    def test_collect_metric_ssim(self):
        """Test collecting SSIM metric results."""
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/ssim", MetricType.SSIM, {})

        metric_result = MetricResult(values={"ssim": torch.tensor(0.85)})

        self.metric_manager.collect_metric("test/ssim", metric_result)

        self.assertIn("test/ssim", self.metric_manager._storage.metrics_storage["metrics"])

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/ssim"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertAlmostEqual(entry["metric_results"][0]["values"]["ssim"], 0.85, places=5)

    def test_collect_metric_ssim_with_metadata(self):
        """Test collecting SSIM metric results with metadata."""
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/ssim", MetricType.SSIM, {})

        metric_result = MetricResult(values={"ssim": torch.tensor(0.90)})

        self.metric_manager.collect_metric(
            "test/ssim", metric_result, frame_meta=self.mock_frame_meta, sequence_id="test_sequence"
        )

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/ssim"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertAlmostEqual(entry["metric_results"][0]["values"]["ssim"], 0.90, places=5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "test_sequence")

    def test_collect_metric_lpips(self):
        """Test collecting LPIPS metric results."""
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/lpips", MetricType.LPIPS, {})

        metric_result = MetricResult(values={"lpips": torch.tensor(0.15)})

        self.metric_manager.collect_metric("test/lpips", metric_result)

        self.assertIn("test/lpips", self.metric_manager._storage.metrics_storage["metrics"])

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/lpips"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertAlmostEqual(entry["metric_results"][0]["values"]["lpips"], 0.15, places=5)

    def test_collect_metric_lpips_with_metadata(self):
        """Test collecting LPIPS metric results with metadata."""
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/lpips", MetricType.LPIPS, {})

        metric_result = MetricResult(values={"lpips": torch.tensor(0.20)})

        self.metric_manager.collect_metric(
            "test/lpips", metric_result, frame_meta=self.mock_frame_meta, sequence_id="test_sequence"
        )

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/lpips"]
        self.assertEqual(len(entry["metric_results"]), 1)
        self.assertAlmostEqual(entry["metric_results"][0]["values"]["lpips"], 0.20, places=5)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "test_sequence")

    def test_collect_aggregated_metric_ssim(self):
        """Test collecting aggregated SSIM metrics."""
        metric_result = MetricResult(values={"mean": torch.tensor(0.87)})

        self.metric_manager.register_metric("test/ssim", SSIMMetric(data_range=1.0))

        self.metric_manager.collect_aggregated_metric("test/ssim", metric_result, AggregationMethod.MEAN)

        self.assertIn("test/ssim", self.metric_manager._storage.metrics_storage["metrics"])
        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/ssim"]
        self.assertEqual(len(entry["aggregated_results"]), 1)
        self.assertAlmostEqual(entry["aggregated_results"][0]["result"]["values"]["mean"], 0.87, places=5)
        self.assertEqual(entry["aggregated_results"][0]["method"], "mean")

    def test_collect_aggregated_metric_lpips(self):
        """Test collecting aggregated LPIPS metrics."""
        metric_result = MetricResult(values={"mean": torch.tensor(0.18)})

        self.metric_manager.register_metric("test/lpips", LPIPSMetric(device="cpu", normalize=True))

        self.metric_manager.collect_aggregated_metric("test/lpips", metric_result, AggregationMethod.MEAN)

        self.assertIn("test/lpips", self.metric_manager._storage.metrics_storage["metrics"])
        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/lpips"]
        self.assertEqual(len(entry["aggregated_results"]), 1)
        self.assertAlmostEqual(entry["aggregated_results"][0]["result"]["values"]["mean"], 0.18, places=5)
        self.assertEqual(entry["aggregated_results"][0]["method"], "mean")

    def test_write_metrics_to_yaml_basic(self):
        """Test writing metrics to YAML file."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/psnr", MetricType.PSNR, {})

        # Collect some test metrics
        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})
        self.metric_manager.collect_metric("test/psnr", metric_result)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.metric_manager.write_metrics(temp_dir)

            metrics_file = os.path.join(temp_dir, "metrics.yaml")
            self.assertTrue(os.path.exists(metrics_file))

            with open(metrics_file, "r") as f:
                content = yaml.safe_load(f)

                # Check structure
                self.assertIn("metrics", content)
                self.assertIn("metadata", content)
                self.assertIn("program_version", content["metadata"])
                self.assertIn("run_info", content["metadata"])

                # Check that program_version is a dictionary (from Pydantic BaseModel)
                self.assertIsInstance(content["metadata"]["program_version"], dict)

                # Check metrics
                metrics = content["metrics"]
                self.assertIn("test/psnr", metrics)

    def test_write_metrics_to_yaml_with_aggregated_metrics(self):
        """Test writing metrics to YAML with aggregated metrics computation."""
        # Register a metric and compute it
        psnr_metric = PSNRMetric(data_range=1.0)
        self.metric_manager.register_metric("psnr", psnr_metric)

        def dummy_function():
            # Create dummy image data for PSNR computation
            pred = torch.randn(1, 3, 64, 64)
            target = torch.randn(1, 3, 64, 64)
            return pred, target

        # Compute the metric
        self.metric_manager.compute("psnr", torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))

        with tempfile.TemporaryDirectory() as temp_dir:
            self.metric_manager.write_metrics(temp_dir, aggregate_metrics=True)

            metrics_file = os.path.join(temp_dir, "metrics.yaml")
            self.assertTrue(os.path.exists(metrics_file))

            with open(metrics_file, "r") as f:
                content = yaml.safe_load(f)

                # Should have metrics with aggregated results
                metrics = content["metrics"]
                self.assertIn("psnr", metrics)
                self.assertIn("aggregated_results", metrics["psnr"])

    def test_write_metrics_to_yaml_without_aggregated_metrics(self):
        """Test writing metrics to YAML without computing aggregated metrics."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/psnr", MetricType.PSNR, {})

        # Collect some test metrics
        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})
        self.metric_manager.collect_metric("test/psnr", metric_result)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.metric_manager.write_metrics(temp_dir, aggregate_metrics=False)

            metrics_file = os.path.join(temp_dir, "metrics.yaml")
            self.assertTrue(os.path.exists(metrics_file))

            with open(metrics_file, "r") as f:
                content = yaml.safe_load(f)

                # Should still have the collected metrics
                metrics = content["metrics"]
                self.assertIn("test/psnr", metrics)

    def test_collect_metric_with_sequence_id_list(self):
        """Test collecting metrics with list sequence ID."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/psnr", MetricType.PSNR, {})

        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})

        self.metric_manager.collect_metric("test/psnr", metric_result, sequence_id=["seq1", "seq2"])

        # Should join with "+"
        self.assertIn("test/psnr", self.metric_manager._storage.metrics_storage["metrics"])
        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/psnr"]
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["sequence_id"], "seq1+seq2")

    def test_collect_metric_with_complex_metadata(self):
        """Test collecting metrics with complex metadata handling."""
        # First register the metric
        from nre.metrics.types import MetricType

        self.metric_manager._storage.initialize_metric("test/psnr", MetricType.PSNR, {})

        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})

        frame_meta = FrameMeta(unique_sensor_idx=0, unique_frame_idx=42)
        timestamps_startend_us = torch.tensor([[1000000, 2000000]])

        self.metric_manager.collect_metric(
            "test/psnr",
            metric_result,
            frame_meta=frame_meta,
            timestamps_startend_us=timestamps_startend_us,
            sequence_id="test_sequence",
        )

        entry = self.metric_manager._storage.metrics_storage["metrics"]["test/psnr"]
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["timestamp_us_begin"], 1000000)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["timestamp_us_end"], 2000000)
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["unique_frame_idx"], 42)

    def test_collect_metric_with_datasource_integration(self):
        """Test collecting metrics with datasource integration."""
        metric_manager_with_datasource = MetricManager()  # Remove datasource parameter

        # First register the metric
        from nre.metrics.types import MetricType

        metric_manager_with_datasource._storage.initialize_metric("test/psnr", MetricType.PSNR, {})
        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})
        frame_meta = FrameMeta(unique_sensor_idx=1, unique_frame_idx=0)

        metric_manager_with_datasource.collect_metric(
            "test/psnr", metric_result, frame_meta=frame_meta, sequence_id="test_sequence"
        )

        # Should use the datasource to get sensor ID
        self.assertIn("test/psnr", metric_manager_with_datasource._storage.metrics_storage["metrics"])
        entry = metric_manager_with_datasource._storage.metrics_storage["metrics"]["test/psnr"]
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["unique_sensor_id"], "camera_1")

    def test_collect_metric_datasource_fallback(self):
        """Test collecting metrics when datasource methods fail."""

        # Create a mock datasource that raises exceptions
        class MockDataSource:
            def get_camera_sensor_ids(self, unique_sensors=True):
                raise AttributeError("Method not available")

            def get_lidar_sensor_ids(self, unique_sensors=True):
                raise IndexError("Index out of range")

        metric_manager_with_datasource = MetricManager()

        # First register the metric
        from nre.metrics.types import MetricType

        metric_manager_with_datasource._storage.initialize_metric("test/psnr", MetricType.PSNR, {})

        metric_result = MetricResult(values={"psnr": torch.tensor(25.5)})

        # Should not raise exception, should fall back to default naming
        metric_manager_with_datasource.collect_metric(
            "test/psnr", metric_result, frame_meta=self.mock_frame_meta, sequence_id="test_sequence"
        )

        # Should use fallback naming
        self.assertIn("test/psnr", metric_manager_with_datasource._storage.metrics_storage["metrics"])
        entry = metric_manager_with_datasource._storage.metrics_storage["metrics"]["test/psnr"]
        self.assertEqual(entry["metric_results"][0]["sensor_data"]["unique_sensor_id"], "camera_0")


if __name__ == "__main__":
    unittest.main()
