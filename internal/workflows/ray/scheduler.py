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
import threading
import time

from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import click
import ray
import ray.scripts.scripts as ray_scripts
import yaml

from click.testing import CliRunner


logger = logging.getLogger(__name__)


class JobStatusLoggingActor:
    """
    A ray remote actor that is used to gather job status logs.
    It will be polled by the main process to get logs from the jobs.
    """

    def __init__(self):
        self.lines = []
        # Prevent read/write race conditions
        self.cv = threading.Condition()

    def log(self, line: str):
        with self.cv:
            self.lines.append(line)
            self.cv.notify_all()

    def read_and_clear(self, timespan_s: float = 0.5) -> List[str]:
        """
        Block for timespan_s seconds and gather all new lines logged since the last read.
        Returns the new lines and clear the buffer
        """
        deadline = time.time() + timespan_s
        with self.cv:
            while len(self.lines) == 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return []
                self.cv.wait(timeout=remaining)
            lines = self.lines[:]
            self.lines.clear()
            return lines


def generate_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate parameter combinations from sweep config."""
    method = config.get("method", "grid")
    parameters = config.get("parameters", {})

    if method != "grid":
        raise ValueError(f"Only grid search supported, got: {method}")

    param_names = []
    param_values = []

    for param_name, param_config in parameters.items():
        param_names.append(param_name)
        if "values" in param_config:
            param_values.append(param_config["values"])
        elif "min" in param_config and "max" in param_config:
            step = param_config.get("step", 1)
            values = list(range(param_config["min"], param_config["max"] + 1, step))
            param_values.append(values)
        else:
            raise ValueError(f"Unsupported parameter config for {param_name}")

    combinations = []
    for combination in product(*param_values):
        param_dict = dict(zip(param_names, combination))
        combinations.append(param_dict)

    return combinations


def parse_command(command_template: List[str], parameters: Dict[str, Any]) -> List[str]:
    """Substitute parameters in command template."""
    substituted_command = []

    parameter_used: bool = False
    for cmd_part in command_template:
        # Handle ${args_no_hyphens} special case
        if cmd_part == "${args_no_hyphens}":
            # Add all parameters as separate arguments
            for param_name, param_value in parameters.items():
                substituted_command.append(f"{param_name}={param_value}")
            parameter_used = True
        else:
            # Note that we disable direct substitution now inline with wandb sweep behaviour
            substituted_command.append(cmd_part)

    if not parameter_used:
        raise ValueError("No parameters used in command template. Currently only `${args_no_hyphens}` is supported.")

    return substituted_command


def run_job(
    job_id: int,
    command_template: List[str],
    parameters: Dict[str, Any],
    working_dir: Optional[str] = None,
    log_file_dir: Optional[str] = None,
    status_logging_actor: Optional[JobStatusLoggingActor] = None,
) -> Tuple[int, str, str]:
    """
    Run a single job with given parameters.

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    # Get command template from object store and parse with parameters
    command = parse_command(command_template, parameters)

    # Set environment variables for parameters
    env = os.environ.copy()
    for key, value in parameters.items():
        env[key] = str(value)

    slurm_unique_job_id = os.environ.get("UNIQUE_JOB_ID", "N/A")
    if status_logging_actor is not None:
        status_logging_actor.log.remote(  # type: ignore
            f"Job {job_id} started with parameters: {parameters} (UNIQUE_JOB_ID: {slurm_unique_job_id})"
        )

    stdout_fp: Optional[TextIO] = None
    stderr_fp: Optional[TextIO] = None

    try:
        # Write bash outputs to files if possible
        if log_file_dir is not None:
            stdout_fp = open(os.path.join(log_file_dir, f"job-{job_id}.stdout"), "w")
            stderr_fp = open(os.path.join(log_file_dir, f"job-{job_id}.stderr"), "w")

            stdout_fp.write(f"-- Working directory: {working_dir}\n")
            stdout_fp.write(f"-- Running job {job_id} with parameters: {parameters}\n")
            stdout_fp.write(f"-- UNIQUE_JOB_ID: {slurm_unique_job_id}\n")
            stdout_fp.write(f"-- Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            stdout_fp.flush()

        # Run the command
        result = subprocess.run(command, env=env, cwd=working_dir, stdout=stdout_fp, stderr=stderr_fp)
        return result.returncode, "Output redirected to worker logs", "Err redirected to worker logs"

    except Exception as e:
        return -1, "", str(e)

    finally:
        if stdout_fp is not None:
            stdout_fp.close()
        if stderr_fp is not None:
            stderr_fp.close()


def run_sweep(config_path: str, max_retries: int = 3, log_file_dir: str | None = None):
    """Run the complete parameter sweep."""
    # Load sweep configuration
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Generate parameter combinations
    combinations = generate_combinations(sweep_config)
    command_template = sweep_config.get("command", [])

    if not command_template:
        raise ValueError("No command specified in sweep config")

    logger.info(f"Generated {len(combinations)} parameter combinations")

    ray_resource_request = sweep_config.get("ray_resource_request", {})
    num_cpus = ray_resource_request.get("num_cpus", 0)
    num_gpus = ray_resource_request.get("num_gpus", 1)
    logger.info(f"Using {num_cpus} CPUs and {num_gpus} GPUs per job")

    run_job_remote = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(run_job).remote

    # Upload command template once to object store
    command_template_ref = ray.put(command_template)
    log_file_dir_ref = ray.put(log_file_dir)

    logger.info("Submitting all jobs to Ray cluster...")

    # Submit ALL jobs at once - use single command template reference
    job_metadata = {}  # future -> (job_id, parameters, retry_count)
    futures = []

    # Status logging actor has to be persistent, hence we force it on the head node
    head_node_id = ray.get_runtime_context().get_node_id()
    status_logging_actor = ray.remote(num_cpus=0, num_gpus=0, label_selector={"ray.io/node-id": head_node_id})(
        JobStatusLoggingActor
    ).remote()

    for job_id, parameters in enumerate(combinations):
        future = run_job_remote(
            job_id=job_id,
            command_template=command_template_ref,
            parameters=parameters,
            log_file_dir=log_file_dir_ref,
            status_logging_actor=status_logging_actor,
        )
        job_metadata[future] = (job_id, parameters, 0)
        futures.append(future)

    logger.info(f"Submitted {len(futures)} jobs to Ray cluster")

    # Job tracking
    completed_jobs = []
    failed_jobs = []
    total_jobs = len(combinations)

    status_logging_actor_ref = status_logging_actor.read_and_clear.remote(timespan_s=0.5)
    futures.append(status_logging_actor_ref)

    # Monitor job completion using ray.wait()
    while futures:
        # Wait for at least one job to complete
        done_futures, futures = ray.wait(futures, num_returns=1)

        if len(done_futures) == 0:
            continue

        future = done_futures[0]

        # Every fixed interval this future will be polled to get the logs from the jobs.
        if future == status_logging_actor_ref:
            logs = ray.get(status_logging_actor_ref)
            for log in logs:
                logger.info(log)
            status_logging_actor_ref = status_logging_actor.read_and_clear.remote(timespan_s=0.5)

            # If jobs are all completed, no need to keep polling
            if len(futures) > 0:
                futures.append(status_logging_actor_ref)

            continue

        job_id, parameters, retry_count = job_metadata.pop(future)

        try:
            exit_code, stdout, stderr = ray.get(future)
            is_preemption = False

        except Exception as e:
            # This is usually RayTask error since the node is being preempted.
            # Under such case we schedule retrial without incrementing the retry count.
            exit_code, stdout, stderr = -1, "", str(e)
            is_preemption = True

        if exit_code == 0:
            completed_jobs.append((job_id, parameters))
            logger.info(f"Job {job_id} ({parameters}) completed successfully [{len(completed_jobs)}/{total_jobs}]")
            continue

        # Job failed, check if we should retry
        if retry_count < max_retries:
            warning_message = (
                f"Job {job_id} ({parameters}) failed (exit: {exit_code}), retrying... ({retry_count + 1}/{max_retries})"
                if not is_preemption
                else f"Job {job_id} ({parameters}) preempted (retry count: {retry_count}), rescheduling..."
            )
            logger.warning(warning_message)
            # Resubmit job
            new_future = run_job_remote(
                job_id=job_id,
                command_template=command_template_ref,
                parameters=parameters,
                log_file_dir=log_file_dir_ref,
                status_logging_actor=status_logging_actor,
            )
            job_metadata[new_future] = (job_id, parameters, retry_count + 1 if not is_preemption else retry_count)
            futures.append(new_future)
        else:
            failed_jobs.append((job_id, parameters, exit_code, stderr))
            logger.error(f"Job {job_id} ({parameters}) failed permanently after {max_retries} retries")

    if failed_jobs:
        logger.error("FAILED JOBS:")
        for job_id, parameters, exit_code, error in failed_jobs:
            logger.error(f"  Job {job_id}: {parameters} (exit: {exit_code})")


@click.group()
def cli():
    """Ray scheduler for running parameter sweeps."""
    pass


@click.command()
def start_head_node():
    """Initialize Ray cluster."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting ray head node on this cluster...")
    runner = CliRunner()
    if runner.invoke(ray_scripts.status).exit_code != 0:
        logger.info("Ray head node not running, starting it...")
        runner.invoke(ray_scripts.stop)
        start_result = runner.invoke(
            ray_scripts.start, ["--head", "--num-cpus", "0", "--num-gpus", "0", "--disable-usage-stats"]
        )
        if start_result.exit_code != 0:
            raise RuntimeError(f"Failed to start Ray head node: {start_result.output}")
    else:
        logger.info("Ray head node already running. Status query successful.")

    logger.info("Testing connection to Ray cluster...")
    try:
        ray.init(address="127.0.0.1:6379", ignore_reinit_error=True)
        logger.info("Ray cluster is up and ready to use.")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Ray cluster: {e}")
    finally:
        ray.shutdown()


@click.command()
def check_ray_status():
    """Check Ray cluster status."""
    logging.basicConfig(level=logging.INFO)
    runner = CliRunner()
    result = runner.invoke(ray_scripts.status)
    logger.info(f"Ray status command result (exit code: {result.exit_code}): \n{result.output}")


@click.command()
@click.option("--force", "-f", is_flag=True, help="Force stop Ray cluster")
def stop_head_node(force: bool):
    """Stop Ray cluster."""
    logging.basicConfig(level=logging.INFO)

    runner = CliRunner()
    result = runner.invoke(ray_scripts.stop, ["--force"] if force else [])
    logger.info(f"Ray stop command result (exit code: {result.exit_code}): \n{result.output}")


@click.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option("--max-retries", "-r", default=3, help="Maximum number of retries for failed jobs")
@click.option("--ray-address", "-a", default=None, help="Ray cluster address (if connecting to existing cluster)")
@click.option("--log-file-dir", "-l", default=None, help="Log file directory")
def run_sweep_cmd(config: Path, max_retries: int, ray_address: Optional[str], log_file_dir: str | None):
    """Run parameter sweep on existing Ray cluster."""
    # Configure logging
    if log_file_dir is not None:
        os.makedirs(log_file_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        filename=os.path.join(log_file_dir, "scheduler.log") if log_file_dir is not None else None,
        force=True,
    )

    try:
        # Connect to existing Ray cluster
        ray.init(address=ray_address)
        logger.info(f"Connected to Ray cluster at {ray_address}")

        # Run the sweep
        run_sweep(config_path=str(config), max_retries=max_retries, log_file_dir=log_file_dir)

    except Exception as e:
        logger.error(f"Failed to connect to Ray cluster: {e}")
        logger.error("Make sure Ray is initialized first using 'start-head-node' command")
        raise
    finally:
        # Cleanup
        ray.shutdown()


# Add commands to group
cli.add_command(start_head_node, name="start-head-node")
cli.add_command(check_ray_status, name="check-ray-status")
cli.add_command(stop_head_node, name="stop-head-node")
cli.add_command(run_sweep_cmd, name="run-sweep")


if __name__ == "__main__":
    cli()
