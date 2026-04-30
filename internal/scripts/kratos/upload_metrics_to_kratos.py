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

import datetime
import logging
import os
import subprocess
import sys
import tempfile

from pathlib import Path
from typing import Optional

import click
import yaml

from internal.scripts.kratos.kratos_uploader import upload_metrics
from nre.utils.metrics import MetricsFileContent


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_s3_url(value: str) -> bool:
    return value.startswith("s3://")


def download_with_s3cmd(s3_url: str, dest_path: Path) -> None:
    cmd = [sys.executable, "-m", "s3cmd", "get", s3_url, str(dest_path)]
    logger.info("Downloading %s to %s", s3_url, dest_path)
    subprocess.check_call(cmd)


def parse_timings_file(file_path: Path) -> Optional[float]:
    """Parses timings.txt and returns train elapsed time in seconds."""
    try:
        if not file_path.exists():
            logger.warning(f"Timings file not found: {file_path}")
            return None

        with open(file_path, "r") as f:
            for line in f:
                if "Elapsed time for train run:" in line:
                    # Format: [timestamp] Elapsed time for train run: HH:MM:SS.mmm
                    parts = line.split("Elapsed time for train run:")
                    if len(parts) < 2:
                        continue
                    time_str = parts[1].strip()
                    try:
                        h, m, s = time_str.split(":")
                        return float(h) * 3600 + float(m) * 60 + float(s)
                    except ValueError:
                        logger.warning(f"Could not parse time string: {time_str}")
                        return None
    except Exception as e:
        logger.warning(f"Failed to parse timings file {file_path}: {e}")
    return None


def load_and_upload(file_path: Path, train_time: Optional[float] = None) -> bool:
    reporting_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path), tz=datetime.timezone.utc)
    with open(file_path, "r") as f:
        metrics_data = yaml.safe_load(f)
        metrics_content = MetricsFileContent.model_validate(metrics_data)
        return upload_metrics(metrics_content, reporting_date, train_time=train_time)


@click.command(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.argument("source")
@click.option(
    "--timings",
    type=click.Path(exists=True),
    help="Path to timings.txt file containing train run elapsed time",
    required=False,
)
def upload_metrics_to_kratos(source: str, timings: Optional[str]) -> int:
    """Upload a metrics.yaml to Kratos Telemetry. Accepts a local path or an s3:// URL."""

    train_time = None
    if timings:
        train_time = parse_timings_file(Path(timings))
        if train_time:
            logger.info(f"Parsed train duration: {train_time} seconds")

    # Local path
    if not is_s3_url(source):
        try:
            success = load_and_upload(Path(source), train_time)
            raise SystemExit(0 if success else 1)
        except Exception as e:
            logger.error("Error loading metrics file: %s", e)
            raise SystemExit(1)

    # S3 URL: download to temp then upload
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "metrics.yaml"
        try:
            download_with_s3cmd(source, tmp_path)
            success = load_and_upload(tmp_path, train_time)
            raise SystemExit(0 if success else 1)
        except subprocess.CalledProcessError as e:
            logger.error("s3cmd failed (exit %s)", e.returncode)
            raise SystemExit(1)
        except Exception as e:
            logger.error("Error processing metrics file: %s", e)
            raise SystemExit(1)


if __name__ == "__main__":
    upload_metrics_to_kratos()
