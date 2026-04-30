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

"""Python CLI for metrics monitor helper commands.

Replaces the bash script with a Bazel-friendly Python entrypoint that reuses
the existing monitor/uploader implementation.
"""

from __future__ import annotations

import json
import logging
import os

from pathlib import Path

import click

from internal.scripts.kratos.metrics_monitor import MetricsMonitor
from internal.scripts.kratos.s3cmd_monitor import S3CmdMonitor, S3Object


def _check_prerequisites() -> int:
    logging.info("Checking prerequisites...")

    # s3cmd Python module (invoked as `python -m s3cmd`)
    try:
        import importlib.util

        if importlib.util.find_spec("s3cmd") is None:
            logging.error("s3cmd Python module not found. Install with: bazel run @nre_pip_deps//:pip -- install s3cmd")
            return 1
    except Exception:
        logging.exception("Error while checking for s3cmd Python module")
        return 1

    # s3cmd configuration
    if not Path.home().joinpath(".s3cfg").exists():
        logging.error("s3cmd is not configured. Run: python3 -m s3cmd --configure")
        return 1

    # temp directory
    temp_dir = Path("/tmp/metrics_download")
    if not temp_dir.exists():
        logging.info("Creating temporary directory %s", temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    # Kratos env vars
    required_envs = [
        "NRE_SQA_SSA_URL",
        "NRE_SQA_SSA_CLIENT_ID",
        "NRE_SQA_SSA_CLIENT_SECRET",
        "NRE_SQA_KRATOS_TELEMETRY_ENDPOINT",
        "NRE_SQA_KRATOS_SCHEMAID",
    ]
    missing = [e for e in required_envs if not os.environ.get(e)]
    if missing:
        logging.warning("The following Kratos env vars are not set:")
        for m in missing:
            logging.warning("  - %s", m)
    else:
        logging.info("All required Kratos environment variables are set")

    return 0


def _require_bucket(bucket: str | None) -> None:
    if not bucket:
        raise SystemExit("--bucket is required for this command")


def cmd_scan(bucket: str, state_file: str, temp_dir: str) -> int:
    s3_monitor = S3CmdMonitor(bucket=bucket, state_file=state_file, temp_dir=temp_dir)
    monitor = MetricsMonitor(s3_monitor=s3_monitor)
    try:
        monitor.scan_and_process()
        return 0
    except RuntimeError as err:
        logging.error(str(err))
        return 1


def cmd_scan_only(bucket: str, state_file: str, temp_dir: str) -> int:
    s3_monitor = S3CmdMonitor(bucket=bucket, state_file=state_file, temp_dir=temp_dir)
    monitor = MetricsMonitor(s3_monitor=s3_monitor, dry_run=True)
    monitor.scan_and_mark_processed()
    return 0


def cmd_dry_run(bucket: str, state_file: str, temp_dir: str) -> int:
    s3_monitor = S3CmdMonitor(bucket=bucket, state_file=state_file, temp_dir=temp_dir)
    monitor = MetricsMonitor(s3_monitor=s3_monitor, dry_run=True)
    monitor.scan_and_process()
    return 0


def cmd_stats(bucket: str, state_file: str, temp_dir: str) -> int:
    # Stats are derived from monitor's s3 monitor state
    s3_monitor = S3CmdMonitor(bucket=bucket, state_file=state_file, temp_dir=temp_dir)
    monitor = MetricsMonitor(s3_monitor=s3_monitor, dry_run=True)
    stats = monitor.get_statistics()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_reset_state(bucket: str, state_file: str, temp_dir: str) -> int:
    s3_monitor = S3CmdMonitor(bucket=bucket, state_file=state_file, temp_dir=temp_dir)
    monitor = MetricsMonitor(s3_monitor=s3_monitor, dry_run=True)
    monitor.reset_state()
    print("State reset successfully")
    return 0


def cmd_one(s3_uri: str, state_file: str, temp_dir: str) -> int:
    if not s3_uri.startswith("s3://"):
        logging.error("Provide an S3 URI like s3://bucket/path/to/metrics.yaml")
        return 2

    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        logging.error("Invalid S3 URI: %s", s3_uri)
        return 2

    s3mon = S3CmdMonitor(bucket=bucket, state_file=state_file, temp_dir=temp_dir)
    try:
        tmp_path = s3mon.download(S3Object(key=key, size=0, last_modified=""))
    except Exception:
        logging.exception("Failed to download object: %s", s3_uri)
        return 1

    try:
        monitor = MetricsMonitor(s3_monitor=s3mon)
        ok = monitor._upload_metrics(tmp_path)
        return 0 if ok else 1
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@click.group(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.option(
    "log_level",
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging level",
)
@click.option("verbose", "--verbose", is_flag=True, help="Enable verbose logging (equivalent to DEBUG)")
def cli(log_level: str, verbose: bool):
    effective_level_name = ("DEBUG" if verbose else log_level).upper()
    logging.basicConfig(
        level=getattr(logging, effective_level_name, logging.INFO), format="%(levelname)s - %(message)s"
    )


@cli.command("check")
def check_cmd():
    raise SystemExit(_check_prerequisites())


@cli.command("scan")
@click.option("bucket", "--bucket", required=True)
@click.option("state_file", "--state-file", default="/tmp/metrics_monitor_state.json", show_default=True)
@click.option("temp_dir", "--temp-dir", default="/tmp/metrics_download", show_default=True)
def scan_cmd(bucket: str, state_file: str, temp_dir: str):
    _require_bucket(bucket)
    raise SystemExit(cmd_scan(bucket, state_file, temp_dir))


@cli.command("scan-only")
@click.option("bucket", "--bucket", required=True)
@click.option("state_file", "--state-file", default="/tmp/metrics_monitor_state.json", show_default=True)
@click.option("temp_dir", "--temp-dir", default="/tmp/metrics_download", show_default=True)
def scan_only_cmd(bucket: str, state_file: str, temp_dir: str):
    _require_bucket(bucket)
    raise SystemExit(cmd_scan_only(bucket, state_file, temp_dir))


@cli.command("dry-run")
@click.option("bucket", "--bucket", required=True)
@click.option("state_file", "--state-file", default="/tmp/metrics_monitor_state.json", show_default=True)
@click.option("temp_dir", "--temp-dir", default="/tmp/metrics_download", show_default=True)
def dry_run_cmd(bucket: str, state_file: str, temp_dir: str):
    _require_bucket(bucket)
    raise SystemExit(cmd_dry_run(bucket, state_file, temp_dir))


@cli.command("stats")
@click.option("bucket", "--bucket", required=True)
@click.option("state_file", "--state-file", default="/tmp/metrics_monitor_state.json", show_default=True)
@click.option("temp_dir", "--temp-dir", default="/tmp/metrics_download", show_default=True)
def stats_cmd(bucket: str, state_file: str, temp_dir: str):
    _require_bucket(bucket)
    raise SystemExit(cmd_stats(bucket, state_file, temp_dir))


@cli.command("reset-state")
@click.option("bucket", "--bucket", required=True)
@click.option("state_file", "--state-file", default="/tmp/metrics_monitor_state.json", show_default=True)
@click.option("temp_dir", "--temp-dir", default="/tmp/metrics_download", show_default=True)
def reset_state_cmd(bucket: str, state_file: str, temp_dir: str):
    _require_bucket(bucket)
    raise SystemExit(cmd_reset_state(bucket, state_file, temp_dir))


@cli.command("one")
@click.argument("s3_uri")
@click.option("state_file", "--state-file", default="/tmp/metrics_monitor_state.json", show_default=True)
@click.option("temp_dir", "--temp-dir", default="/tmp/metrics_download", show_default=True)
def one_cmd(s3_uri: str, state_file: str, temp_dir: str):
    raise SystemExit(cmd_one(s3_uri, state_file, temp_dir))


if __name__ == "__main__":
    cli()
