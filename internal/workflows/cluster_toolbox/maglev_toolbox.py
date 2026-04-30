# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import os
import re
import subprocess
import tempfile

import yaml

from omegaconf import DictConfig
from python.runfiles import runfiles

from internal.workflows import __reporoot__
from internal.workflows.cluster_toolbox.base_toolbox import ClusterToolbox
from internal.workflows.cluster_toolbox.utils import (
    SubprocessError,
    get_formatted_datetime,
    run_command_as_subprocess,
)


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


class MaglevToolbox(ClusterToolbox):
    job_template_path: str = "internal/workflows/cluster_toolbox/job_templates/maglev/workflow_template.yaml"
    sweep_job_template_path: str = (
        "internal/workflows/cluster_toolbox/job_templates/maglev/sweep_workflow_template.yaml"
    )

    def __init__(self, config_name: str, hydra_args: list[str] = []) -> None:
        super().__init__(config_name, hydra_args)
        if self.config.maglev_cli is None:
            # Pick up Maglev CLI exe path from Bazel environment if it exists, otherwise default to just "maglev".
            # The Bazel rule of script needs to have a data dependency on rule "@maglev_cli//file:maglev" for
            # the executable to be available at this path.
            try:
                bazel_maglev_cli = self._get_bazel_maglev_cli()
            except FileNotFoundError:
                log.warning("Maglev CLI executable not found at Bazel path, using 'maglev' instead")
                self.cli = "maglev"
            else:
                log.info(f"Using Bazel Maglev CLI executable at {bazel_maglev_cli}")
                self.cli = bazel_maglev_cli

        if not self.config.exec_mode == "docker":
            raise ValueError("MaglevToolbox only supports execution in Docker mode")

        self._check_image_exists(self.docker_image)
        self._check_cli()

        # Log in to the Maglev service if an API key has been provided. Otherwise before-script login is assumed.
        if self.config.maglev_api_key is not None:
            self._login(self.config.maglev_api_key)
        elif "MAGLEV_API_KEY" in os.environ:
            self._login(os.environ["MAGLEV_API_KEY"])

    @staticmethod
    def _get_bazel_maglev_cli() -> str:
        """Get the path to the Maglev CLI executable from Bazel if it exists, otherwise default to just "maglev".

        Raises:
            FileNotFoundError: If the Maglev CLI executable is not found at the Bazel path.
        """
        RUNFILES = runfiles.Create()
        maglev_cli_path = RUNFILES.Rlocation("maglev_cli/file/maglev")
        if not os.path.isfile(maglev_cli_path):
            raise FileNotFoundError(f"Maglev CLI executable not found at {maglev_cli_path}")
        return maglev_cli_path

    def _check_cli(self) -> None:
        """Make sure the Maglev CLI executable is found."""
        command = [self.cli, "version"]
        log.info(" ".join(command))
        try:
            run_command_as_subprocess(command)
        except SubprocessError as e:
            e.add_note("Maglev CLI test failed")
            raise e

    def _login(self, api_key: str) -> None:
        """Login to Maglev

        Args:
            api_key (str): maglev API key
        """
        command = [self.cli, "login", "maglev.nvda.ai", api_key]
        try:
            run_command_as_subprocess(command)
        except SubprocessError as e:
            e.add_note("Maglev login failed")
            raise e

    def _submit_workflow(self, name: str, spec_path: str, dry_run: bool = False) -> None:
        command = [self.cli, "workflows2", "run", "-f", spec_path, "--name", name]
        log.info(f"Launching Maglev workflow '{name}' using configuration {spec_path}")
        log.info(" ".join(command))
        if not dry_run:
            try:
                run_command_as_subprocess(command)
            except SubprocessError as e:
                e.add_note("Maglev workflow launch failed")
                raise e
        else:
            log.warning(
                "This was a dry run (simulation). No Maglev workflow, Docker image or W&B sweep have been created."
            )

    def _start_wandb_sweep_agents(
        self,
        sweep_name: str,
        num_agents: int,
        wandb_sweep_string: str,
        job_template_path: str | None,
        env_vars: dict[str, str],
        dry_run: bool,
        verbose: bool,
    ) -> None:
        """Launch a job on the cluster, with each worker running a wandb sweep agent to execute the runs of a wandb sweep"""
        self.config.num_workers = num_agents  # FIXME

        self.submit_job(
            job_name=self._get_sweep_job_name(sweep_name=sweep_name, sweep_string=wandb_sweep_string),
            job_template_path=(
                job_template_path
                if job_template_path is not None
                else os.path.join(__reporoot__, self.sweep_job_template_path)
            ),
            env_vars={"WANDB_SWEEP_STRING": wandb_sweep_string, "DATETIME": get_formatted_datetime()} | env_vars,
            dry_run=dry_run,
            verbose=verbose,
        )

    def _run_with_stdout(self, command: str) -> tuple[int, str]:
        """
        Run a command and return the stdout decoded as ascii
        """
        cmd_arr = ["/bin/bash", "-c", command]
        proc = subprocess.run(
            cmd_arr,
            encoding="utf-8",
            capture_output=True,
        )
        return proc.returncode, proc.stdout

    def get_wf_spec(self, wf_name: str) -> dict:
        """Get the Maglev workflow specification for a given workflow name"""
        cmd = " ".join([self.cli, "workflows2", "get", wf_name, "--spec"])
        exit_code, stdout = self._run_with_stdout(cmd)
        if exit_code != 0:
            raise SubprocessError(f"Maglev getting workflow spec failed: {stdout}")
        spec = yaml.safe_load(stdout)
        return spec

    def get_available_secret_names(self) -> list[str]:
        """Get the list of available secret names"""
        cmd = f"{self.cli} secrets list | cut -f1"
        exit_code, stdout = self._run_with_stdout(cmd)
        if exit_code != 0:
            raise SubprocessError(f"Maglev getting available secret names failed: {stdout}")
        secret_names = stdout.splitlines()
        return [name.strip() for name in secret_names]

    def get_latest_run(self, workflow_name: str) -> str:
        """Get the latest run for a given workflow"""
        cmd = f"{self.cli} workflows2 list {workflow_name} --output json | jq -r '.[0].id'"
        exit_code, stdout = self._run_with_stdout(cmd)
        if exit_code != 0:
            raise SubprocessError(f"Maglev getting latest run failed: {stdout}")
        return stdout.strip()

    def task_expired(self, workflow_name: str, run: str, task: str) -> bool:
        """Check the retention policy for a given workflow task. Returns True if the task is expired."""
        if run in ["latest-success", "latest"]:
            run = self.get_latest_run(workflow_name)

        cmd = f"{self.cli} workflows2 retention-policy get {workflow_name}/{run}/{task} --output json"
        _, stdout = self._run_with_stdout(cmd)

        # Handle empty or invalid JSON response
        if not stdout or not stdout.strip():
            log.warning(f"Empty response from retention policy check for {workflow_name}/{run}/{task}")
            return False

        try:
            retention_policy = json.loads(stdout)
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse retention policy JSON for {workflow_name}/{run}/{task}: {e}")
            log.debug(f"Raw stdout: {stdout}")
            return False

        if "expired" in retention_policy and retention_policy["expired"]:
            return True

        return False

    def submit_job(
        self,
        job_name: str,
        command: str | None = None,
        job_template_path: str | None = None,
        env_vars: dict[str, str] = {},
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        # Maglev only allows alphanumeric characters and dashes (-) in workflow names
        job_name = re.sub("[^a-zA-Z0-9]", "-", job_name)

        # Load selected Maglev workflow or template if not provided
        load_path = (
            job_template_path if job_template_path is not None else os.path.join(__reporoot__, self.job_template_path)
        )

        log.info(f"Loading Maglev workflow configuration from {load_path}")
        with open(load_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)

        # Modify Maglev workflow configuration according to the args.
        assert isinstance(self.config, DictConfig)
        tasks = conf["tasks"]
        worker_pools = conf["workerPools"]
        worker_pools[0]["workers"] = self.config.num_workers
        worker_pools[0]["resourceShare"] = self.config.resource_share
        worker_pools[0]["nodeConstraints"]["required"]["nodeType"] = self.config.node_type
        tasks[0]["replicas"] = worker_pools[0]["workers"]  # As many workers as job replicas
        tasks[0]["image"] = self.docker_image
        # Merge default env vars from config with passed env vars (passed ones take precedence)
        default_env_vars = self.config.get("default_env_vars", {})
        tasks[0]["env"] = {**default_env_vars, **env_vars}  # CLI env_vars override config ones

        if command is not None:
            # if a command is passed we insert it in the task definition
            tasks[0]["args"][1] = command

        if verbose:
            log.info(yaml.safe_dump(conf, sort_keys=False))

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf8", prefix="maglev_workflow_", suffix=".yaml", delete=False
        ) as tmp_file:
            # Save modified Maglev workflow configuration into a temporary file for the maglev CLI command.
            log.info(f"Saving {tmp_file.name}")
            yaml.safe_dump(conf, tmp_file)
            self._submit_workflow(name=job_name, spec_path=tmp_file.name, dry_run=dry_run)
