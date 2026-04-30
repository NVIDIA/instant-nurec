# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from torch._prims_common import DeviceLikeType

from nre.config.version import Version, get_version
from nre.datasets.base import BaseDataSource
from nre.metrics.types import MetricType
from nre.metrics.utils import AggregationMethod
from nre.utils.batch import FrameMeta
from nre.utils.misc import unpack_optional


@dataclass
class MetricResult:
    """Unified result structure for all metrics.

    This class provides a unified result structure for all metrics.
    It contains the values and metadata for a metric.

    Args:
        values: Dictionary of metric values in the form of torch tensors.
        metadata: Dictionary of metric metadata.

    Example:
        values = {"metric_result": torch.tensor(25.0)}
        metadata = {"data_range": 1.0, "input_shape": [1024, 1024]}
        metric_result = MetricResult(values=values, metadata=metadata)
    """

    values: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_value(self, name: str) -> torch.Tensor:
        """Get a specific metric value by name."""
        if name in self:
            return self.values[name]
        else:
            raise KeyError(f"Metric value '{name}' not found")

    def get_available_values(self) -> list[str]:
        """Get list of available metric value names."""
        return list(self.values.keys())

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary format for compatibility."""
        return self.values

    def to(self, device: DeviceLikeType) -> MetricResult:
        """Move the metric result values to a specific device. Metadata is not moved."""
        for value in self.values.values():
            value.to(device)
        return self

    def to_serializable_dict(self, include_metadata: bool = False) -> dict[str, Any]:
        """Convert to a dictionary with Python primitives for JSON/YAML serialization.

        Args:
            include_metadata: Whether to include the metadata in the serialized dictionary. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary with metric values and (optionally) metadata.

        Converts torch tensors to Python primitives (float, int, list, etc.)
        and handles nested dictionaries and lists.
        """

        def _to_primitive(obj):
            """Convert to Python primitives."""
            if isinstance(obj, np.ndarray):
                if obj.size == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: _to_primitive(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [_to_primitive(item) for item in obj]
            else:
                raise ValueError(f"Unsupported type: {type(obj)}")

        serializable_dict = {"values": _to_primitive(self.values)}
        if include_metadata:
            serializable_dict["metadata"] = _to_primitive(self.metadata)
        return serializable_dict

    def __getitem__(self, name: str) -> torch.Tensor:
        """Dictionary-like access to metric values."""
        return self.get_value(name)

    def __contains__(self, name: str) -> bool:
        """Check if a metric value exists."""
        return name in self.values


class BaseMetric(ABC):
    """Abstract base class for all metrics in NRE.

    This class provides a common interface for all metrics, including image metrics,
    point cloud metrics, and performance metrics.
    """

    def __init__(
        self,
        device: DeviceLikeType | None = None,
        aggregation_methods: list[AggregationMethod] | AggregationMethod = AggregationMethod.MEAN,
    ):
        self._values: list[MetricResult] = []
        self.device = device
        if isinstance(aggregation_methods, AggregationMethod):
            self._aggregation_methods = [aggregation_methods]
        else:
            self._aggregation_methods = aggregation_methods

    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> None:
        """Validate the inputs to the metric."""
        pass

    @abstractmethod
    def _compute(self, *args, **kwargs) -> MetricResult:
        """Compute the metric value from inputs.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            MetricResult: Structured result containing metric values and metadata.
                For simple metrics, a single value will be populated.
                For complex metrics, additional values and metadata may be included.
        """
        pass

    def compute(self, *args, **kwargs) -> MetricResult:
        """Compute the metric value from inputs with validation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            MetricResult: Structured result containing metric values and metadata.
                For simple metrics, a single value will be populated.
                For complex metrics, additional values and metadata may be included.
        """
        self.validate_inputs(*args, **kwargs)
        return self._compute(*args, **kwargs)

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric state."""
        pass

    def append(self, value: MetricResult) -> None:
        """Store a computed metric value.

        Args:
            value: The computed metric value to store.
            metadata: Optional metadata to associate with the value.
        """
        self._values.append(value)

    @abstractmethod
    def aggregate(self) -> dict[AggregationMethod, MetricResult]:
        """Aggregate stored values using the specified methods."""
        pass

    def clear(self) -> None:
        """Clear all stored values."""
        self._values.clear()

    def to(self, device: DeviceLikeType) -> BaseMetric:
        """Move metric to specified device including all stored values.

        Args:
            device: The device to move the metric to.

        Returns:
            BaseMetric: The metric instance with the device set.
        """
        self.device = device
        self._values = [value.to(device) for value in self._values]
        return self

    def values(self) -> list[MetricResult]:
        """Return the list of stored values."""
        return self._values

    def aggregation_methods(self) -> list[AggregationMethod]:
        """Return the aggregation methods for the metric."""
        return self._aggregation_methods

    @abstractmethod
    def type(self) -> MetricType:
        """Return the type of the metric."""
        pass

    def metadata(self) -> dict[str, Any]:
        """Return the metadata for the metric.

        Subclasses can override this method to provide custom metadata.
        They can also call super().metadata() to extend the base metadata.
        Default implementation returns an empty dictionary.
        """
        return {}

    def __len__(self) -> int:
        """Return the number of stored values."""
        return len(self._values)


class MetricStorage:
    """Storage for metrics."""

    def __init__(
        self,
        train_config_name: str = "unknown",
        mode: Literal["train", "val", "grpc_server", "unknown"] = "unknown",
        run_id: str = "unknown",
        version: Version | None = None,
    ) -> None:
        self.metrics_storage: dict[str, Any] = {}
        self.metrics_storage["metrics"] = {}
        self.metrics_storage["metadata"] = {}

        self.metrics_storage["metadata"]["run_info"] = {}
        self.metrics_storage["metadata"]["run_info"]["train_config_name"] = train_config_name
        self.metrics_storage["metadata"]["run_info"]["mode"] = mode
        self.metrics_storage["metadata"]["run_info"]["run_id"] = run_id

        self.metrics_storage["metadata"]["schema_version"] = 3

        # TODO: update this when the schema changes in the future
        version = (
            version
            if version is not None
            else unpack_optional(
                get_version(
                    # allow empty as we require a concrete version number to store along with the metrics
                    allow_empty=True
                )
            )
        )
        # Convert Version object to dictionary for YAML serialization
        self.metrics_storage["metadata"]["program_version"] = (
            version.model_dump() if version is not None else {"version": "unknown"}
        )

    def _process_sequence_id(self, sequence_id: list[str] | str | None) -> str:
        """Process the sequence id."""
        if sequence_id is None:
            return ""
        elif isinstance(sequence_id, list):
            return str.join("+", sequence_id)
        else:
            return sequence_id

    def _process_unique_sensor_id(self, unique_sensor_id: str | None, sequence_id: str | None) -> str:
        unique_sensor_id = unique_sensor_id or ""
        # Remove the sequence_id from the unique_sensor_id if present
        # e.g:
        # unique_sensor_id = camera_front_wide_120fov@3c291d58-15ad-11ed-b911-00044bf65f0e@1659807950800159-1659807956200057
        # sequence_id = 3c291d58-15ad-11ed-b911-00044bf65f0e@1659807950800159-1659807956200057
        # -> unique_sensor_id = camera_front_wide_120fov
        if (
            sequence_id is not None
            and unique_sensor_id != ""
            and unique_sensor_id[-len(sequence_id) :] == sequence_id
            and unique_sensor_id[-len(sequence_id) - 1] in ["@", "+", "-", "_", "|"]
        ):
            unique_sensor_id = unique_sensor_id[: -len(sequence_id) - 1]
        return unique_sensor_id

    def initialize_metric(
        self,
        name: str,
        metric_type: MetricType,
        metric_metadata: dict[str, Any],
    ) -> None:
        """Register a new metric entry to the storage.

        Args:
            name: The name of the metric.
            metric: The metric to add.

        """
        self.metrics_storage["metrics"][name] = {
            "metric": metric_type.name.lower(),
            "metadata": metric_metadata,
            "aggregated_results": [],
            "metric_results": [],
        }

    def add_entry(
        self,
        name: str,
        metric_result: dict[str, Any],
        frame_meta: FrameMeta | None = None,
        timestamps_startend_us: torch.Tensor | None = None,
        sequence_id: list[str] | str | None = None,
        unique_sensor_id: str | None = None,
    ) -> None:
        """Add a metric result entry to the storage.

        Args:
            name: The name of the metric.
            metric_result: The metric result to add.
            frame_meta: Optional frame meta data with unique frame and sensor indices.
            timestamps_startend_us: Optional start and end timestamps for this frame.
            sequence_id: Optional sequence identifier.
            unique_sensor_id: Optional unique sensor identifier.
        """
        if name not in self.metrics_storage["metrics"]:
            raise KeyError(f"Metric '{name}' not found. Found only {self.metrics_storage['metrics'].keys()}")

        sequence_id = self._process_sequence_id(sequence_id)
        unique_sensor_id = self._process_unique_sensor_id(unique_sensor_id, sequence_id)

        # Add sequence_id and unique_sensor_id to the metric_result
        metric_result["sensor_data"] = {}
        metric_result["sensor_data"]["sequence_id"] = sequence_id
        metric_result["sensor_data"]["unique_sensor_id"] = unique_sensor_id

        if timestamps_startend_us is not None:
            # Already a CPU tensor copy; just index directly
            ts = timestamps_startend_us[0]
            metric_result["sensor_data"]["timestamp_us_begin"] = int(ts[0].item())
            metric_result["sensor_data"]["timestamp_us_end"] = int(ts[1].item())
        if frame_meta is not None:
            metric_result["sensor_data"]["unique_frame_idx"] = frame_meta.unique_frame_idx

        self.metrics_storage["metrics"][name]["metric_results"].append(metric_result)

    def add_aggregated_entry(
        self,
        name: str,
        aggregated_result: dict[str, Any],
        aggregation_method: AggregationMethod,
    ) -> None:
        """Add an aggregated metric result entry to the storage.

        Args:
            name: The name of the metric.
            aggregated_result: The aggregated metric result to add.
            aggregation_method: The aggregation method to use.
        """
        if name not in self.metrics_storage["metrics"]:
            raise KeyError(f"Metric '{name}' not found. Found only {self.metrics_storage['metrics'].keys()}")

        self.metrics_storage["metrics"][name]["aggregated_results"].append(
            {
                "method": aggregation_method.name.lower(),
                "result": aggregated_result,
            }
        )

    def write_metrics(self, output_dir: str, ext: str) -> None:
        """Write the metrics to a file given the extension.

        Args:
            output_dir: Directory to save the metrics file.
            ext: Extension of the file to save the metrics.
        """
        match ext:
            case "yaml":
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                with open(Path(output_dir) / f"metrics.yaml", "w") as f:
                    yaml.dump(self.metrics_storage, f)
            case _:
                raise ValueError(f"Unsupported extension: {ext}")


@dataclass
class ComputeEntry:
    """Generic entry for metric computation with flexible metadata.

    This class provides a flexible way to specify metadata for metric computation
    while maintaining backward compatibility with the old specific metadata approach.

    Args:
        name: Name of the metric in the MetricManager
        metadata: Generic metadata dictionary that can contain any serializable data.
                 This replaces the old specific fields for backward compatibility.
                 Can include keys like 'datasource', 'sequence_id', etc.
        include_metadata: Whether to include metadata in the serialized output
    """

    name: str  # Name of the metric in the MetricManager
    metadata: dict[str, Any] | None = None
    include_metadata: bool = True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata with optional default."""
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in metadata."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value


class MetricManager:
    """Central manager for all metrics in a system.

    This class provides a central point for managing all metrics in a system.
    It allows for the registration of metrics, the computation of metrics,
    the aggregation of metrics, and the resetting of metrics.
    """

    def __init__(
        self,
        train_config_name: str = "unknown",
        mode: Literal["train", "val", "grpc_server", "unknown"] = "unknown",
        run_id: str = "unknown",
        version: Version | None = None,
        device: DeviceLikeType | None = None,
    ):
        self.device = device
        self._metrics: dict[str, BaseMetric] = {}
        self._storage = MetricStorage(train_config_name, mode, run_id, version)

    def register_metric(
        self,
        name: str,
        metric: BaseMetric,
        overwrite_existing: bool = False,
    ) -> None:
        """Register a metric with the manager.

        Args:
            metric: The metric to register.

        Raises:
            KeyError: If the metric name is already registered.
        """
        if name in self._metrics and not overwrite_existing:
            raise KeyError(f"Metric '{name}' already registered")
        if self.device is not None:
            metric.to(self.device)
        self._metrics[name] = metric

        # Add the metric to the storage
        self._storage.initialize_metric(name, metric.type(), metric.metadata())

    def remove_metric(self, name: str) -> None:
        """Remove a registered metric by name.

        Args:
            name: The name of the metric to remove.
        """
        if not self.has_metric(name):
            raise KeyError(f"Metric '{name}' not found. Found only {self.list_metrics()}")
        del self._metrics[name]

    def get_metric(self, name: str) -> BaseMetric:
        """Get a registered metric by name.

        Args:
            name: The name of the metric to get.

        Returns:
            BaseMetric: The registered metric.
        """
        if not self.has_metric(name):
            raise KeyError(f"Metric '{name}' not found. Found only {self.list_metrics()}")
        return self._metrics[name]

    def get_last(self, name: str) -> MetricResult | None:
        """Get the last computed metric result for a given metric name.

        Args:
            name: The name of the metric to get the last result for.

        Returns:
            MetricResult | None: The last computed metric result.
        """
        if not self.has_metric(name):
            raise KeyError(f"Metric '{name}' not found. Found only {self.list_metrics()}")
        if len(self._metrics[name].values()) == 0:
            return None
        return self._metrics[name].values()[-1]

    def get_all(self, name: str) -> list[MetricResult]:
        """Get all computed metric results for a given metric name.

        Args:
            name: The name of the metric to get the results for.

        Returns:
            list[MetricResult]: A list of all computed metric results.
        """
        if not self.has_metric(name):
            raise KeyError(f"Metric '{name}' not found. Found only {self.list_metrics()}")
        return self._metrics[name].values()

    def compute(self, entry: str | ComputeEntry, *args, **kwargs) -> None:
        """Compute a metric, append the result, and collect the result in the metric manager.

        Args:
            entry: The name of the metric to compute or a ComputeEntry object which contains data for collecting the metric result.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # Determine input arguments
        if isinstance(entry, ComputeEntry):
            name = entry.name
            datasource = entry.get("datasource")
            is_lidar = entry.get("is_lidar")
            frame_meta = entry.get("frame_meta")
            timestamps_startend_us = entry.get("timestamps_startend_us")
            sequence_id = entry.get("sequence_id")
            include_metadata = entry.include_metadata
        else:
            name = entry
            datasource = None
            is_lidar = False
            frame_meta = None
            timestamps_startend_us = None
            sequence_id = None
            include_metadata = False

        # Compute the metric
        metric = self.get_metric(name)
        value = metric.compute(*args, **kwargs)
        metric.append(value)

        # TODO: Not all metrics metric need collection.
        # Would be nice to have a way to disable collection for certain metrics.
        # Collect the metric result
        self.collect_metric(
            name=name,
            metric_result=value,
            datasource=datasource,
            is_lidar=is_lidar,
            frame_meta=frame_meta,
            timestamps_startend_us=timestamps_startend_us,
            sequence_id=sequence_id,
            include_metadata=include_metadata,
        )

    def aggregate(self, name: list[str] | str | None = None) -> dict[str, dict[AggregationMethod, MetricResult]]:
        """Aggregate registered metrics.

        Args:
            name: The name(s) of the metric(s) to aggregate. If None, all metrics are aggregated. Default is None.

        Returns:
            dict[str, dict[AggregationMethod, MetricResult]]: A dictionary of metric names and their aggregated values.
        """
        aggregated_metrics = {}
        if name is None:
            for name, metric in self._metrics.items():
                aggregated_metrics[name] = metric.aggregate()
        else:
            if isinstance(name, str):
                name = [name]
            for metric_name in name:
                if not self.has_metric(metric_name):
                    raise KeyError(f"Metric '{metric_name}' not found. Found only {self.list_metrics()}")
                aggregated_metrics[metric_name] = self._metrics[metric_name].aggregate()
        return aggregated_metrics

    def reset(self, name: list[str] | str | None = None) -> None:
        """Reset the state of registered metrics.

        Args:
            name: The name(s) of the metric(s) to reset. If None, all metrics are reset. Default is None.
        """

        if name is None:
            for metric in self._metrics.values():
                metric.reset()
        else:
            if isinstance(name, str):
                name = [name]
            for metric_name in name:
                if not self.has_metric(metric_name):
                    raise KeyError(f"Metric '{metric_name}' not found. Found only {self.list_metrics()}")
                self._metrics[metric_name].reset()

    def clear(self, name: list[str] | str | None = None) -> None:
        """Clear stored values from metrics.

        Args:
            name: The name(s) of the metric(s) to clear. If None, all metrics are cleared. Default is None.
        """
        if name is None:
            for metric in self._metrics.values():
                metric.clear()
        else:
            if isinstance(name, str):
                name = [name]
            for metric_name in name:
                if not self.has_metric(metric_name):
                    raise KeyError(f"Metric '{metric_name}' not found. Found only {self.list_metrics()}")
                self._metrics[metric_name].clear()

    def to(self, device: DeviceLikeType) -> MetricManager:
        """Move the metric manager to a specific device including all stored values."""
        self.device = device
        for metric in self._metrics:
            self._metrics[metric] = self._metrics[metric].to(device)
        return self

    def list_metrics(self) -> list[str]:
        """List all registered metric names.

        Returns:
            List[str]: A list of all registered metric names.
        """
        return list(self._metrics.keys())

    def has_metric(self, name: str) -> bool:
        """Check if a metric is registered.

        Args:
            name: The name of the metric to check.

        Returns:
            bool: True if the metric is registered, False otherwise.
        """
        return name in self._metrics

    def collect_metric(
        self,
        name: str,
        metric_result: MetricResult,
        datasource: BaseDataSource | None = None,
        is_lidar: bool = False,
        frame_meta: FrameMeta | None = None,
        timestamps_startend_us: torch.Tensor | None = None,
        sequence_id: list[str] | str | None = None,
        include_metadata: bool = True,
    ) -> None:
        """Collect a MetricResult with metadata for later aggregation and export.

        This method allows collecting MetricResult objects directly into the MetricStorage.

        Args:
            name: The name of the metric.
            metric_result: The MetricResult to collect.
            datasource: Optional datasource used to get sensor id.
            is_lidar: True if the sensor is lidar, False for a camera.
            frame_meta: Optional frame meta data with unique frame and sensor indices.
            timestamps_startend_us: Optional start and end timestamps for this frame.
            sequence_id: Optional sequence identifier.
        """

        if not metric_result.values:
            raise ValueError("MetricResult has no values to collect")

        # Convert metric_result to a dictionary
        metric_result_dict = metric_result.to_serializable_dict(include_metadata=include_metadata)

        unique_sensor_id = None

        if frame_meta is not None:
            unique_sensor_idx = frame_meta.unique_sensor_idx

            if datasource is not None:
                try:
                    unique_sensor_id = (
                        datasource.get_camera_sensor_ids(unique_sensors=True)[unique_sensor_idx]
                        if not is_lidar
                        else datasource.get_lidar_sensor_ids(unique_sensors=True)[unique_sensor_idx]
                    )
                except (IndexError, AttributeError):
                    # Fallback if datasource doesn't have the required methods
                    unique_sensor_id = f"{'lidar' if is_lidar else 'camera'}_{unique_sensor_idx}"
            else:
                # Fallback for multiple sensors or missing datasource
                unique_sensor_id = f"{'lidar' if is_lidar else 'camera'}_{unique_sensor_idx}"

        self._storage.add_entry(
            name=name,
            metric_result=metric_result_dict,
            frame_meta=frame_meta,
            timestamps_startend_us=timestamps_startend_us,
            sequence_id=sequence_id,
            unique_sensor_id=unique_sensor_id,
        )

    def collect_aggregated_metric(
        self,
        name: str,
        result: MetricResult,
        aggregation_method: AggregationMethod,
    ) -> None:
        """Collect an aggregated metric result.

        Args:
            name: The name of the metric.
            result: The MetricResult to collect.
            aggregation_method: The aggregation method to use.
        """
        aggregate_dict = result.to_serializable_dict(include_metadata=True)
        self._storage.add_aggregated_entry(name, aggregate_dict, aggregation_method)

    def write_metrics(self, output_dir: str, aggregate_metrics: bool = True, ext: str = "yaml") -> None:
        """Write collected metrics to a YAML file.

        Args:
            output_dir: Directory to save the metrics YAML file.
        """
        if aggregate_metrics:
            aggregated_metrics = self.aggregate()
            for name, aggregated_metric in aggregated_metrics.items():
                for method, value in aggregated_metric.items():
                    self.collect_aggregated_metric(name, value, method)

        self._storage.write_metrics(output_dir, ext)
