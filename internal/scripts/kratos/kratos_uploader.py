#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
NRE Metrics Library - Core functionality for uploading metrics to Kratos Telemetry.
"""

import datetime
import json
import logging
import math
import os
import time

from dataclasses import dataclass
from typing import List, Optional, Union

from nre.config.version import Version
from nre.utils.metrics import (
    GenericMetricSample,
    MetricsFileContent,
    MetricsFileRunInfo,
    MetricsGeneralSampleDict,
    MetricsPerSensorSampleDict,
    MetricsSamplesPerSequencePerSensor,
    StepwiseMetricSample,
)


@dataclass
class MetricRow:
    """Data model representing a single metric row."""

    sequence_id: Optional[str]
    camera_id: Optional[str]
    lidar_id: Optional[str]
    metric_name: str
    timestamp_us_begin: Optional[int]
    timestamp_us_end: Optional[int]
    unique_frame_idx: Optional[int]
    value: Optional[Union[float, int]]
    git_commit_date: Optional[str]
    git_commit_sha_short: Optional[str]
    git_tree_dirty: Optional[bool]
    version_major: Optional[int]
    version_minor: Optional[int]
    version_patch: Optional[int]
    version_string: Optional[str]
    mode: Optional[str]
    run_id: Optional[str]
    train_config_name: Optional[str]
    reporting_date: Optional[str]

    @staticmethod
    def from_sample_seq_sensor(
        metric_name: str,
        sequence_id: Optional[str],
        camera_id: Optional[str],
        lidar_id: Optional[str],
        sample: GenericMetricSample,
        version_info: Version,
        run_info: MetricsFileRunInfo,
        reporting_date: str,
    ) -> "MetricRow":
        """Create a MetricRow from a GenericMetricSample."""

        return MetricRow(
            metric_name=metric_name,
            sequence_id=sequence_id,
            camera_id=camera_id,
            lidar_id=lidar_id,
            timestamp_us_begin=sample.timestamp_us_begin if isinstance(sample, StepwiseMetricSample) else None,
            timestamp_us_end=sample.timestamp_us_end if isinstance(sample, StepwiseMetricSample) else None,
            unique_frame_idx=sample.unique_frame_idx if isinstance(sample, StepwiseMetricSample) else None,
            value=sample.value,
            git_commit_date=version_info.git_commit_date.isoformat(),
            git_commit_sha_short=version_info.git_commit_sha_short,
            git_tree_dirty=version_info.git_tree_dirty,
            version_major=version_info.version_major,
            version_minor=version_info.version_minor,
            version_patch=version_info.version_patch,
            version_string=version_info.version_string,
            mode=run_info.mode,
            run_id=run_info.run_id,
            train_config_name=run_info.train_config_name,
            reporting_date=reporting_date,
        )

    @staticmethod
    def from_sample(
        metric_name: str,
        sample: GenericMetricSample,
        version_info: Version,
        run_info: MetricsFileRunInfo,
        reporting_date: str,
    ) -> "MetricRow":
        """Create a MetricRow from a GenericMetricSample."""

        return MetricRow.from_sample_seq_sensor(
            metric_name=metric_name,
            sequence_id=None,
            camera_id=None,
            lidar_id=None,
            sample=sample,
            version_info=version_info,
            run_info=run_info,
            reporting_date=reporting_date,
        )

    def to_dict(self) -> dict:
        """Convert the metric row to a dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "camera_id": self.camera_id,
            "lidar_id": self.lidar_id,
            "metric_name": self.metric_name,
            "timestamp_us_begin": self.timestamp_us_begin,
            "timestamp_us_end": self.timestamp_us_end,
            "unique_frame_idx": self.unique_frame_idx,
            "value": self.value,
            "git_commit_date": self.git_commit_date,
            "git_commit_sha_short": self.git_commit_sha_short,
            "git_tree_dirty": self.git_tree_dirty,
            "version_major": self.version_major,
            "version_minor": self.version_minor,
            "version_patch": self.version_patch,
            "version_string": self.version_string,
            "mode": self.mode,
            "run_id": self.run_id,
            "train_config_name": self.train_config_name,
            "reporting_date": self.reporting_date,
        }


class Validator:
    """Utility class for validating numeric values."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate_numeric_value(
        self, value: Optional[Union[float, int]], metric_name: str
    ) -> Optional[Union[float, int]]:
        """Validate and log numeric values for PostgreSQL compatibility.

        Args:
            value: The numeric value to validate
            metric_name: Name of the metric for logging purposes

        Returns:
            The validated value or None if invalid
        """
        if value is None:
            self.logger.debug(f"None value found for {metric_name}")
            return None
        if isinstance(value, float):
            if math.isnan(value):
                self.logger.warning(f"NaN value found for {metric_name}, converting to NULL")
                return None
            if math.isinf(value):
                self.logger.warning(f"Infinity value found for {metric_name}, converting to NULL")
                return None
        return value


class Parser:
    """Utility class for parsing metrics into metric rows."""

    def __init__(self, value_validator: Validator, reporting_date: datetime.datetime):
        self.value_validator = value_validator
        self.reporting_date_str = reporting_date.isoformat()

    def parse_metrics(self, metrics_content: MetricsFileContent) -> List[MetricRow]:
        """Parse metrics content into metric rows.

        Args:
            metrics_content: MetricsFileContent object containing the metrics

        Returns:
            List of MetricRow objects

        Raises:
            ValueError: If metrics_content is None or invalid
        """

        result_rows = []
        version_info = metrics_content.program_version
        run_info = metrics_content.run_info

        if metrics_content.metrics.general is not None:
            # Process general metrics
            result_rows.extend(self._process_general_metrics(metrics_content.metrics.general, version_info, run_info))

        if metrics_content.metrics.per_sequence:
            # Process sensor-specific metrics
            for sequence_id, sensors in metrics_content.metrics.per_sequence.items():
                result_rows.extend(self._process_sensor_metrics(sequence_id, sensors, version_info, run_info))

        return result_rows

    def _process_general_metrics(
        self, general_metrics: MetricsGeneralSampleDict, version_info: Version, run_info: MetricsFileRunInfo
    ) -> List[MetricRow]:
        """Process metrics from the general section."""
        rows = []
        for metric_name, values in general_metrics.items():
            if not values or not isinstance(values, list):
                continue

            for value_entry in values:
                try:
                    validated_value = self.value_validator.validate_numeric_value(value_entry.value, metric_name)
                    row = MetricRow.from_sample(
                        metric_name, value_entry, version_info, run_info, self.reporting_date_str
                    )
                    row.value = validated_value
                    rows.append(row)
                except (AttributeError, ValueError) as e:
                    # Log error but continue processing other metrics
                    logging.error(f"Error processing general metric {metric_name}: {str(e)}")
                    continue
        return rows

    def _process_sensor_metrics(
        self,
        sequence_id: str,
        sensors: MetricsSamplesPerSequencePerSensor,
        version_info: Version,
        run_info: MetricsFileRunInfo,
    ) -> List[MetricRow]:
        """Process metrics from sensor-specific sections."""
        rows = []
        for is_lidar, sensor_items in enumerate([sensors.per_camera.items(), sensors.per_lidar.items()]):
            for sensor_label, metrics in sensor_items:
                for metric_name, values in metrics.items():
                    for value_entry in values:
                        try:
                            validated_value = self.value_validator.validate_numeric_value(
                                value_entry.value, metric_name
                            )
                            row = MetricRow.from_sample_seq_sensor(
                                metric_name=metric_name,
                                sequence_id=sequence_id,
                                camera_id=sensor_label if 0 == is_lidar else None,
                                lidar_id=sensor_label if 1 == is_lidar else None,
                                sample=value_entry,
                                version_info=version_info,
                                run_info=run_info,
                                reporting_date=self.reporting_date_str,
                            )
                            row.value = validated_value
                            rows.append(row)
                        except (AttributeError, ValueError) as e:
                            # Log error but continue processing other metrics
                            logging.error(
                                f"Error processing sensor metric {metric_name} for sensor {sensor_label}: {str(e)}"
                            )
                            continue
        return rows


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NaN values."""

    def default(self, o: object):
        if isinstance(o, float) and math.isnan(o):
            return None
        return super().default(o)


class MetricsUploader:
    """Uploads metrics to Kratos Telemetry."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # Validate required environment variables
        required_env_vars = [
            "NRE_SQA_SSA_URL",
            "NRE_SQA_SSA_CLIENT_ID",
            "NRE_SQA_SSA_CLIENT_SECRET",
            "NRE_SQA_KRATOS_SCHEMAID",
            "NRE_SQA_KRATOS_TELEMETRY_ENDPOINT",
        ]
        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        try:
            from kratos_pycloudevents.client import TelemetryClient

            # Initialize the Kratos Telemetry client with increased timeouts
            self.telemetry_client = TelemetryClient(
                ssaUrl=os.environ["NRE_SQA_SSA_URL"],
                ssaClientId=os.environ["NRE_SQA_SSA_CLIENT_ID"],
                ssaClientSecret=os.environ["NRE_SQA_SSA_CLIENT_SECRET"],
                telemetryUrl="https://prod.analytics.nvidiagrid.net",
                telemetryConnectTimeout=30,  # Increased from 10
                telemetryReadTimeout=120,  # Increased from 100
            )
        except Exception as e:
            self.logger.error(f"Error initializing Kratos Telemetry client: {str(e)}")
            raise e

    def upload_data(self, metric_rows: List[MetricRow]) -> bool:
        """Upload metric rows to Kratos Telemetry.

        Args:
            metric_rows: List of MetricRow objects to upload

        Returns:
            bool: True if upload was successful, False otherwise

        Raises:
            ValueError: If metric_rows is None or empty
        """
        if not metric_rows:
            self.logger.warning("No metric rows to upload")
            return True

        self.logger.info(f"Preparing to upload {len(metric_rows)} rows to Kratos Telemetry...")

        # Split large batches to avoid SSL issues with big payloads
        batch_size = 50  # Smaller batches to avoid SSL EOF errors
        total_batches = math.ceil(len(metric_rows) / batch_size)

        attributes = {"type": "workflow", "source": "gitlab", "schemaid": os.environ["NRE_SQA_KRATOS_SCHEMAID"]}
        from cloudevents.http import CloudEvent

        failed_batches = []

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(metric_rows))
            batch_rows = metric_rows[start_idx:end_idx]

            self.logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_rows)} rows)")

            # Convert batch to CloudEvents
            cloudevents = [CloudEvent(attributes=attributes, data=row.to_dict()) for row in batch_rows]

            # Retry logic for each batch
            max_retries = 3
            retry_delay = 2  # seconds
            batch_success = False

            for attempt in range(max_retries):
                try:
                    response = self.telemetry_client.send(
                        collectorId=os.environ["NRE_SQA_KRATOS_TELEMETRY_ENDPOINT"], cloudevents=cloudevents
                    )

                    if response.status_code == 200:
                        self.logger.info(f"Successfully uploaded batch {batch_num + 1}/{total_batches}")
                        batch_success = True
                        break
                    else:
                        self.logger.error(f"Failed to upload batch {batch_num + 1} - Status: {response.status_code}")
                        try:
                            self.logger.error(f"Response: {response.json()}")
                        except json.JSONDecodeError:
                            self.logger.error(f"Response: {response.text}")

                        # Don't retry on client errors
                        if response.status_code < 500:
                            break

                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(
                        f"Exception uploading batch {batch_num + 1} (attempt {attempt + 1}/{max_retries}): {error_msg}"
                    )

                    # Check if it's an SSL error
                    if "SSL" in error_msg or "EOF occurred" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2**attempt)  # Exponential backoff
                            self.logger.info(f"SSL error detected. Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)

                            # Re-initialize client on SSL errors
                            if attempt == 1:
                                try:
                                    self.logger.info("Reinitializing TelemetryClient due to SSL errors...")
                                    self.__init__(self.logger)
                                except Exception as reinit_error:
                                    self.logger.error(f"Failed to reinitialize client: {reinit_error}")
                            continue

                    # For non-SSL errors or last attempt, mark as failed
                    if attempt == max_retries - 1:
                        break

            if not batch_success:
                failed_batches.append(batch_num + 1)

            # Small delay between batches to avoid overwhelming the server
            if batch_num < total_batches - 1:
                time.sleep(0.5)

        if failed_batches:
            self.logger.error(f"Failed to upload {len(failed_batches)} batches: {failed_batches}")
            return False

        self.logger.info("Successfully uploaded all metrics to Kratos Telemetry.")
        return True


def upload_metrics(
    metrics_content: MetricsFileContent,
    reporting_date: Optional[datetime.datetime] = None,
    train_time: Optional[float] = None,
) -> bool:
    """Upload metrics from a MetricsFileContent object.

    Args:
        metrics_content: MetricsFileContent object containing the metrics to upload
        reporting_date: Optional datetime to use for reporting. If None, current UTC time is used.
        train_time: Optional elapsed time for train run in seconds to be added as a metric.

    Returns:
        bool: True if upload was successful, False otherwise

    Raises:
        ValueError: If metrics_content is None or invalid
    """
    logger = logging.getLogger(__name__)

    if reporting_date is None:
        reporting_date = datetime.datetime.now(datetime.timezone.utc)

    if not metrics_content:
        raise ValueError("metrics_content cannot be None")

    try:
        # Parse metrics and get metric rows
        metric_rows = Parser(Validator(logger), reporting_date).parse_metrics(metrics_content)

        if train_time is not None:
            version_info = metrics_content.program_version
            run_info = metrics_content.run_info

            # Create a metric row for the train duration
            train_row = MetricRow(
                sequence_id=None,
                camera_id=None,
                lidar_id=None,
                metric_name="elapsed_time",
                timestamp_us_begin=None,
                timestamp_us_end=None,
                unique_frame_idx=None,
                value=train_time,
                git_commit_date=version_info.git_commit_date.isoformat(),
                git_commit_sha_short=version_info.git_commit_sha_short,
                git_tree_dirty=version_info.git_tree_dirty,
                version_major=version_info.version_major,
                version_minor=version_info.version_minor,
                version_patch=version_info.version_patch,
                version_string=version_info.version_string,
                mode="train",  # Explicitly set to train
                run_id=run_info.run_id,
                train_config_name=run_info.train_config_name,
                reporting_date=reporting_date.isoformat(),
            )
            metric_rows.append(train_row)

        # Convert rows to dictionaries for JSON output
        result_dicts = [row.to_dict() for row in metric_rows]

        logger.info(json.dumps(result_dicts, cls=CustomJSONEncoder, indent=2))

        # Upload the data
        return MetricsUploader(logger).upload_data(metric_rows)
    except Exception as e:
        logger.error(f"Error during metrics upload: {str(e)}")
        return False
