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
Automated metrics.yaml monitor and uploader using S3Cmd.
Monitors S3 bucket for new metrics.yaml files and uploads them to Kratos.
"""

import json
import logging
import os
import sys

from pathlib import Path
from typing import Dict

import click
import yaml

from internal.scripts.kratos.kratos_uploader import upload_metrics
from internal.scripts.kratos.s3cmd_monitor import S3CmdMonitor, S3Object
from nre.utils.metrics import MetricsFileContent


# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/metrics_monitor.log")],
)
logger = logging.getLogger(__name__)


class MetricsMonitor:
    """Orchestrates scanning and uploading using an injected S3 monitor."""

    def __init__(
        self,
        s3_monitor: S3CmdMonitor,
        bazel_target: str = "//internal/scripts/kratos:upload_metrics_to_kratos",
        dry_run: bool = False,
    ):
        self.s3_monitor = s3_monitor
        self.bazel_target = bazel_target
        self.dry_run = dry_run

        logger.info("Initialized S3Cmd monitor for bucket: %s", self.s3_monitor.bucket)
        logger.info("State file: %s", self.s3_monitor.state_path)
        logger.info("Temp directory: %s", self.s3_monitor.temp_dir)
        logger.info("Dry run mode: %s", self.dry_run)

    def _upload_metrics(self, file_path: Path) -> bool:
        """Upload metrics by parsing YAML and calling the uploader directly."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would upload: {file_path}")
            return True

        try:
            # Parse YAML and upload directly
            reporting_date = None
            try:
                mtime = os.path.getmtime(file_path)
                reporting_date = __import__("datetime").datetime.fromtimestamp(  # defer import for speed
                    mtime, tz=__import__("datetime").timezone.utc
                )
            except Exception:
                reporting_date = None

            with open(file_path, "r") as f:
                metrics_data = yaml.safe_load(f)
                metrics_content = MetricsFileContent.model_validate(metrics_data)

            success = upload_metrics(metrics_content, reporting_date)
            if success:
                logger.info("Successfully uploaded %s", file_path)
                return True
            logger.error("Uploader reported failure for %s", file_path)
            return False

        except Exception as e:
            logger.error(f"Exception during upload of {file_path}: {e}")
            return False

    def scan_and_process(self) -> Dict[str, bool]:
        """Monitor S3 bucket and process new metrics files."""
        logger.info("Starting S3 bucket scan for new metrics files...")

        # Define upload callback
        def upload_callback(file_path: str) -> bool:
            return self._upload_metrics(Path(file_path))

        # Run S3 monitor
        results = self.s3_monitor.monitor_once(upload_callback=upload_callback, dry_run=self.dry_run)

        # Summary with per-file detail
        successful_files = [k for k, v in results.items() if v]
        failed_files = [k for k, v in results.items() if not v]

        if results:
            if failed_files:
                logger.error(
                    "Failed uploads (%d): %s",
                    len(failed_files),
                    ", ".join(sorted(failed_files)),
                )
            if successful_files:
                logger.info(
                    "Successful uploads (%d): %s",
                    len(successful_files),
                    ", ".join(sorted(successful_files)),
                )
        else:
            logger.info("No new metrics files found")

        # If any upload failed, raise early with details
        if any(not ok for ok in results.values()):
            failed_files = [k for k, v in results.items() if not v]
            raise RuntimeError(f"One or more uploads failed: {', '.join(sorted(failed_files))}")

        return results

    def reset_state(self):
        """Reset the processed files state and persist an empty state file."""
        self.s3_monitor.reset_and_save_state()
        logger.info("Reset processed files state")

    def get_statistics(self) -> Dict:
        """Get statistics about processed files."""
        stats = self.s3_monitor.get_statistics()
        stats["bucket"] = self.s3_monitor.bucket
        stats["temp_directory"] = str(self.s3_monitor.temp_dir)
        return stats

    def scan_and_mark_processed(self) -> Dict[str, bool]:
        """Scan S3 bucket and mark all metrics files as processed without uploading."""
        logger.info("Starting scan-only mode (mark as processed without uploading)...")

        # Use the S3 monitor's scan_only mode
        results = self.s3_monitor.scan_and_mark_all()

        logger.info(f"Marked {len(results)} files as processed (without uploading)")
        return results


@click.command(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option("bucket", "--bucket", required=False, help="S3 bucket to monitor for metrics.yaml files")
@click.option(
    "state_file",
    "--state-file",
    default="/tmp/metrics_monitor_state.json",
    show_default=True,
    help="File to store processed file state",
)
@click.option(
    "temp_dir",
    "--temp-dir",
    default="/tmp/metrics_download",
    show_default=True,
    help="Temporary directory for downloaded files",
)
@click.option("reset_state", "--reset-state", is_flag=True, help="Reset the processed files state and exit")
@click.option("stats", "--stats", is_flag=True, help="Show statistics and exit")
@click.option("dry_run", "--dry-run", is_flag=True, help="Don't actually upload, just show what would be done")
@click.option(
    "scan_only",
    "--scan-only",
    is_flag=True,
    help="Scan and mark files as processed without uploading to Kratos",
)
@click.option(
    "log_level",
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Logging level (overrides default INFO)",
)
@click.option("verbose", "--verbose", is_flag=True, help="Enable verbose logging (equivalent to DEBUG)")
@click.option("object_", "--object", help="Process a single s3 object (s3://bucket/key) and exit")
def metrics_monitor(
    bucket: str | None,
    state_file: str,
    temp_dir: str,
    reset_state: bool,
    stats: bool,
    dry_run: bool,
    scan_only: bool,
    verbose: bool,
    log_level: str | None,
    object_: str | None,
):
    """Monitor an S3 bucket for metrics.yaml files and upload to Kratos."""
    # Set logging level (verbose forces DEBUG unless --log-level provided)
    effective_level_name = (log_level or ("DEBUG" if verbose else "INFO")).upper()
    effective_level = getattr(logging, effective_level_name, logging.INFO)
    logging.getLogger().setLevel(effective_level)

    # Handle one-off object processing (no state scan)
    if object_:
        if not object_.startswith("s3://"):
            logger.error("--object must be an s3:// URL")
            raise SystemExit(2)
        # Extract bucket and key
        try:
            _, _, rest = object_.partition("s3://")
            bucket_val, _, key = rest.partition("/")
            if not bucket_val or not key:
                raise ValueError("Invalid s3 URL")
        except Exception:
            logger.error("Invalid --object value: %s", object_)
            raise SystemExit(2)

        # Download then upload using same codepath as monitor
        s3mon = S3CmdMonitor(bucket=bucket_val, state_file=state_file, temp_dir=temp_dir)
        try:
            tmp_path = s3mon.download(S3Object(key=key, size=0, last_modified=""))
        except Exception as e:
            logger.error("Failed to download object: %s", e)
            raise SystemExit(1)

        # Use uploader directly
        monitor = MetricsMonitor(s3_monitor=s3mon, dry_run=dry_run)
        success = monitor._upload_metrics(tmp_path)
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(0 if success else 1)

    if not bucket:
        logger.error("--bucket is required unless --object is provided")
        raise SystemExit(2)

    # Create monitor instance (inject S3 monitor)
    s3_monitor = S3CmdMonitor(bucket=bucket, state_file=state_file, temp_dir=temp_dir)
    monitor = MetricsMonitor(
        s3_monitor=s3_monitor,
        dry_run=dry_run or scan_only,  # scan-only implies dry-run for uploads
    )

    # Handle special operations
    # If --stats is provided, print stats; if --reset-state also provided, ensure reset runs via finally
    try:
        if stats:
            print(json.dumps(monitor.get_statistics(), indent=2))
            raise SystemExit(0)
    finally:
        if reset_state:
            monitor.reset_state()
            print("State reset successfully")
            raise SystemExit(0)

    # Run monitoring
    if scan_only:
        # Special scan-only mode: mark files as processed without uploading
        results = monitor.scan_and_mark_processed()
        print(f"Scanned and marked {len(results)} files as processed (without uploading)")
        raise SystemExit(0)
    else:
        try:
            _ = monitor.scan_and_process()
            raise SystemExit(0)
        except RuntimeError as err:
            logger.error(str(err))
            raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(metrics_monitor())
