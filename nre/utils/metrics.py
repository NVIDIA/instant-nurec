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

import logging
import os

from itertools import chain
from typing import Callable, Dict, List, Literal, Optional, Union

import torch
import yaml

from pydantic import BaseModel, Field

from nre.config.version import Version, get_version
from nre.utils.batch import FrameMeta
from nre.utils.misc import distributed_all_gather_nested, rank_zero_only, unpack_optional


logger = logging.getLogger(__name__)


class MetricsRunConfig(BaseModel):
    """Minimal config class for metrics collection, avoiding circular dependency."""

    version: Version
    train_config_name: str
    mode: str
    run_id: str


class MetricSample(BaseModel):
    """A class to represent a metric sample."""

    name: Optional[str] = Field(default=None, description="The name of the metric")
    value: float = Field(description="The value of the metric")

    @classmethod
    def from_value(cls, name: str, value: float) -> MetricSample:
        """Create a metric sample from a value."""

        return cls(name=name, value=value)


class StepwiseMetricSample(MetricSample):
    """A class to represent a stepwise metric sample."""

    timestamp_us_begin: int | None = Field(description="The optional start timestamp of the metric")
    timestamp_us_end: int | None = Field(description="The optional end timestamp of the metric")
    unique_frame_idx: int = Field(description="The unique frame index of the metric")

    @classmethod
    def from_value_and_metadata(
        cls,
        name: str,
        value: float,
        frame_meta: FrameMeta,
        timestamps_startend_us: torch.Tensor | None = None,
    ) -> StepwiseMetricSample:
        """Create a stepwise metric sample from a value and metadata."""

        timestamp_us_begin = None
        timestamp_us_end = None
        if timestamps_startend_us is not None:
            # timestamps_startend_us is (1,2) tensor; index directly
            timestamp_us_begin = int(timestamps_startend_us[0, 0].item())
            timestamp_us_end = int(timestamps_startend_us[0, 1].item())

        return cls(
            name=name,
            value=value,
            timestamp_us_begin=timestamp_us_begin,
            timestamp_us_end=timestamp_us_end,
            unique_frame_idx=frame_meta.unique_frame_idx,
        )


GenericMetricSample = Union[MetricSample, StepwiseMetricSample]


def create_metric_sample(
    name: str,
    value: float,
    frame_meta: FrameMeta | None = None,
    timestamps_startend_us: torch.Tensor | None = None,
) -> GenericMetricSample:
    """Create a metric sample from a value and metadata."""

    if frame_meta is not None:
        return StepwiseMetricSample.from_value_and_metadata(name, value, frame_meta, timestamps_startend_us)
    else:
        return MetricSample.from_value(name, value)


class MetricAggregate(BaseModel):
    """A class to represent a metric aggregate."""

    value: float = Field(description="The value of the metric")
    aggregation_method: str = Field(description="The aggregation method of the metric")


class MetricsFileRunInfo(BaseModel):
    """A class to represent the run informations."""

    train_config_name: str = Field(description="The name of the train config")
    mode: str = Field(description="The mode of the run")
    run_id: str = Field(description="The id of the run")


# Metrics grouped by sequence, sensor, metric
MetricsGeneralSampleDict = Dict[str, List[GenericMetricSample]]
MetricsPerSensorSampleDict = Dict[str, Dict[str, List[GenericMetricSample]]]
MetricsAggregateDict = Dict[str, MetricAggregate]


class MetricsSamplesPerSequencePerSensor(BaseModel):
    """A class to represent a metric samples grouped per sensor (camera or lidar)."""

    per_camera: MetricsPerSensorSampleDict = Field(description="Per camera metrics samples")
    per_lidar: MetricsPerSensorSampleDict = Field(description="Per lidar metrics samples")


MetricsSamplesPerSequenceDict = Dict[str, MetricsSamplesPerSequencePerSensor]


class MetricsSamples(BaseModel):
    """A class to represent a single value of a named performance or quality metric."""

    general: MetricsGeneralSampleDict = Field(description="General metrics samples")
    per_sequence: MetricsSamplesPerSequenceDict = Field(description="Per sequence metrics samples")


class MetricsFileContent(BaseModel):
    """A class to represent the content of a metrics file that will be saved to a file."""

    aggregated_metrics: MetricsAggregateDict = Field(description="Aggregated metrics")
    metrics: MetricsSamples = Field(description="List of metrics samples")
    program_version: Version = Field(description="Version informations of the program")
    run_info: MetricsFileRunInfo = Field(description="Run informations")
    # Update the schema version here when the metrics file format changes
    schema_version: int = Field(description="Schema version of the metrics file", default=2)


class MetricsCollector:
    """A class to collect and save metrics to YAML files."""

    def __init__(
        self,
        train_config_name: str = "unknown",
        mode: Literal["train", "val", "grpc_server", "unknown"] = "unknown",
        run_id: str = "unknown",
        version: Optional[Version] = None,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            train_config_name: Name of the training configuration
            mode: Mode of the run (e.g., "train", "val", "grpc_server")
            run_id: Unique identifier for the run
            version: Version information, if None will use get_version() (uses a surrogate version in actual runtime version is not available)
        """
        self._train_config_name = train_config_name
        self._mode = mode
        self._run_id = run_id
        self._version = (
            version
            if version is not None
            else unpack_optional(
                get_version(
                    # allow empty as we require a concrete version number to store along with the metrics
                    allow_empty=True
                )
            )
        )
        self._samples: MetricsSamples = MetricsSamples(general={}, per_sequence={})
        self._reduce_functions: Dict[str, Union[str, Callable[[torch.Tensor], torch.Tensor]]] = {}

    def reset(self) -> None:
        """Reset the metrics collector by clearing all collected metrics."""
        self._samples.general.clear()
        self._samples.per_sequence.clear()

    def save_metric(
        self,
        metric_sample: GenericMetricSample,
        unique_sensor_id: Optional[str] = None,
        sequence_id: Union[list[str], str, None] = None,
        reduce_fx: Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
        is_lidar: bool = False,
    ) -> None:
        """Save a metric value for later aggregation.

        Args:
            name: Name of the metric
            value: Value of the metric
            metadata: Metadata of the metric
            sequence_id: Sequence identifier
            reduce_fx: Reduction function to apply to the metric values
        """
        if metric_sample.name is None:
            raise ValueError("Metric sample name is None")

        if sequence_id is None and unique_sensor_id is None:
            if metric_sample.name not in self._samples.general:
                self._samples.general[metric_sample.name] = []
            self._samples.general[metric_sample.name].append(metric_sample)
        else:
            if sequence_id is None:
                sequence_id = ""
            if isinstance(sequence_id, list):
                sequence_id = str.join("+", sequence_id)
            # sequence_id: str

            if sequence_id not in self._samples.per_sequence:
                self._samples.per_sequence[sequence_id] = MetricsSamplesPerSequencePerSensor(
                    per_camera={}, per_lidar={}
                )
            sensor_dict = (
                self._samples.per_sequence[sequence_id].per_camera
                if not is_lidar
                else self._samples.per_sequence[sequence_id].per_lidar
            )

            unique_sensor_id = unique_sensor_id or ""
            # Remove the sequence_id from the unique_sensor_id if present
            # e.g:
            # unique_sensor_id = camera_front_wide_120fov@3c291d58-15ad-11ed-b911-00044bf65f0e@1659807950800159-1659807956200057
            # sequence_id = 3c291d58-15ad-11ed-b911-00044bf65f0e@1659807950800159-1659807956200057
            # -> unique_sensor_id = camera_front_wide_120fov
            if (
                sequence_id is not None
                and unique_sensor_id[-len(sequence_id) :] == sequence_id
                and unique_sensor_id[-len(sequence_id) - 1] in ["@", "+", "-", "_", "|"]
            ):
                unique_sensor_id = unique_sensor_id[: -len(sequence_id) - 1]

            if unique_sensor_id not in sensor_dict:
                sensor_dict[unique_sensor_id] = {}
            if metric_sample.name not in sensor_dict[unique_sensor_id]:
                sensor_dict[unique_sensor_id][metric_sample.name] = []
            sensor_dict[unique_sensor_id][metric_sample.name].append(metric_sample)

        if not metric_sample.name in self._reduce_functions:
            self._reduce_functions[metric_sample.name] = torch.mean

        if reduce_fx is not None:
            if metric_sample.name in self._reduce_functions and self._reduce_functions[metric_sample.name] != reduce_fx:
                logger.warning(
                    f"Metric {metric_sample.name} already has a different reduce function: {self._reduce_functions[metric_sample.name]} -> {reduce_fx}, overriding"
                )
            self._reduce_functions[metric_sample.name] = reduce_fx

    def combine_samples_across_ranks(self, rank_samples: MetricsSamples) -> MetricsSamples:
        if torch.distributed.is_initialized():
            # Use distributed_all_gather_nested to directly handle the nested structure
            # only_on_rank_zero=False because we need the combined data for sorting
            combined_samples = distributed_all_gather_nested(rank_samples, only_on_rank_zero=False)

            # Sort metrics by timestamp_us_begin
            for sequence_data in combined_samples.per_sequence.values():
                for sensor_type in ["per_camera", "per_lidar"]:
                    for sensor_data in getattr(sequence_data, sensor_type).values():
                        for metric_name, metric_values in sensor_data.items():
                            if metric_values and isinstance(metric_values[0], StepwiseMetricSample):
                                sensor_data[metric_name] = sorted(metric_values, key=lambda x: x.unique_frame_idx)

            return combined_samples
        else:
            return rank_samples

    @rank_zero_only
    def aggregate_metrics(self, combined_samples: MetricsSamples) -> MetricsAggregateDict:
        # Aggregate the combined metrics
        aggregated_metrics: MetricsAggregateDict = {}

        # We allow strings or torch functions as reduce_fx to remain compatible with LightningModule.log() arguments
        resolved_reduce_functions: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for metric_name, reduce_fx in self._reduce_functions.items():
            resolved_reduce_fx: Callable[[torch.Tensor], torch.Tensor]
            if isinstance(reduce_fx, str):
                match reduce_fx:
                    case "min":
                        resolved_reduce_fx = torch.min
                    case "max":
                        resolved_reduce_fx = torch.max
                    case "mean":
                        resolved_reduce_fx = torch.mean
                    case "sum":
                        resolved_reduce_fx = torch.sum
                    case _:
                        raise ValueError(f"Unknown reduction function: {reduce_fx}")
            elif reduce_fx is not None:
                resolved_reduce_fx = reduce_fx
            else:
                resolved_reduce_fx = torch.mean

            resolved_reduce_functions[metric_name] = resolved_reduce_fx

        aggregated_metrics_values: Dict[str, List[float]] = {}

        for _, values_for_sequence in combined_samples.per_sequence.items():
            for _, values_for_sensor in chain(
                values_for_sequence.per_camera.items(), values_for_sequence.per_lidar.items()
            ):
                for metric_name, values_for_metric in values_for_sensor.items():
                    if metric_name not in aggregated_metrics_values:
                        aggregated_metrics_values[metric_name] = []
                    aggregated_metrics_values[metric_name] += [x.value for x in values_for_metric]

        for metric_name in combined_samples.general:
            if metric_name not in aggregated_metrics_values:
                aggregated_metrics_values[metric_name] = []
            aggregated_metrics_values[metric_name] += [x.value for x in combined_samples.general[metric_name]]

        for metric_name, values in aggregated_metrics_values.items():
            if not values:
                continue

            values_tensor = torch.tensor(values, device="cuda")

            # Default aggregate using configured reduce function (backwards compatible)
            reduce_fx = (
                resolved_reduce_functions[metric_name] if metric_name in resolved_reduce_functions else torch.mean
            )
            aggregated_metrics[metric_name] = MetricAggregate(
                value=reduce_fx(values_tensor).item(),
                aggregation_method=reduce_fx.__name__,
            )

            # Additional aggregates: min, max, std, var (population stats for stability)
            aggregated_metrics[f"{metric_name}_min"] = MetricAggregate(
                value=torch.min(values_tensor).item(),
                aggregation_method="min",
            )
            aggregated_metrics[f"{metric_name}_max"] = MetricAggregate(
                value=torch.max(values_tensor).item(),
                aggregation_method="max",
            )
            aggregated_metrics[f"{metric_name}_std"] = MetricAggregate(
                value=torch.std(values_tensor, unbiased=False).item(),
                aggregation_method="std",
            )
            aggregated_metrics[f"{metric_name}_var"] = MetricAggregate(
                value=torch.var(values_tensor, unbiased=False).item(),
                aggregation_method="var",
            )

        return aggregated_metrics

    @rank_zero_only
    def write_aggregated_metrics_to_yaml(
        self, combined_samples: MetricsSamples, aggregated_metrics: MetricsAggregateDict, val_dir: str
    ) -> None:
        content = MetricsFileContent(
            aggregated_metrics=aggregated_metrics,
            metrics=combined_samples,
            program_version=self._version,
            run_info=MetricsFileRunInfo(
                train_config_name=self._train_config_name,
                mode=self._mode,
                run_id=self._run_id,
            ),
        )
        # Write metrics to YAML file
        filename = os.path.join(val_dir, "metrics.yaml")
        with open(filename, "w") as f:
            yaml.dump(content.model_dump(), f)

        logger.info(f"Metrics saved to {filename}")

    def write_metrics_to_yaml(self, val_dir: str) -> None:
        """Write collected metrics to a YAML file.

        Args:
            val_dir: Directory to save validation metrics
        """

        combined_samples = self.combine_samples_across_ranks(self._samples)
        aggregated_metrics = self.aggregate_metrics(combined_samples)
        self.write_aggregated_metrics_to_yaml(combined_samples, aggregated_metrics, val_dir)

        self._samples = combined_samples
