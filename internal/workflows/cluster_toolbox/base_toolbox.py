# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import getpass
import inspect
import logging
import os
import subprocess
import time

from pathlib import Path

import hydra
import omegaconf
import wandb
import yaml

from internal.workflows.cluster_toolbox.utils import (
    ConfError,
    SubprocessError,
    parse_image_name,
    run_command_as_subprocess,
    search_and_replace,
)


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


class ClusterToolbox:
    config: omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig
    config_path: str
    docker_image: str
    username: str

    def __init__(self, config_name: str, hydra_args: list[str] = []) -> None:
        """Initialize the toolbox from a given config file

        Args:
            config_name (str): path to config file
        """

        # get absolute config path
        calling_file = inspect.stack()[1].filename
        abs_base_dir = os.path.realpath(os.path.dirname(calling_file))
        self.config_path = os.path.join(abs_base_dir, "cluster_configs")

        self.username = getpass.getuser()
        self.config = self._load_config(config_name, hydra_args)
        if self.config.docker.build_push:
            self.docker_image = self._build_push_docker()
        else:
            self.docker_image = self.config.docker.image

        log.info(f"Using Docker image {self.docker_image}")

    def _load_config(
        self, config_name: str, hydra_args: list[str]
    ) -> omegaconf.dictconfig.DictConfig | omegaconf.listconfig.ListConfig:
        """
        Loads a hydra configuration with a schema specific to this cluster toolbox, e.g.
        which docker image to use for a submitted job, wandb settings, and cluster-specific settings
        """
        # Hydra expects relative paths
        if os.path.isabs(config_name):
            config_name = os.path.relpath(config_name, self.config_path)

        with hydra.initialize_config_dir(config_dir=self.config_path, version_base=None):
            print(config_name, hydra_args)
            config = hydra.compose(config_name=config_name, overrides=hydra_args)

        return config

    def _check_commit_exists_on_remote(self, commit: str) -> None:
        """Check if a git commit exists on the remote repository.

        Args:
            commit (str): Git commit hash to check

        Raises:
            SubprocessError: If the commit doesn't exist on the remote repository
        """
        try:
            # Get the default remote (usually 'origin')
            remote = (
                subprocess.check_output(["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL)
                .decode("ascii")
                .strip()
            )

            # Check if the commit exists on the remote
            subprocess.check_output(["git", "ls-remote", remote, commit], stderr=subprocess.DEVNULL)
            log.info(f"Commit {commit} exists on remote repository")

        except subprocess.CalledProcessError:
            log.error(
                f"Commit {commit} does not exist on remote repository. "
                f"This will cause the job to fail on the cluster. "
                f"Please push your changes to the remote repository first."
            )
            raise SubprocessError(
                f"Commit {commit} not found on remote repository. Push your changes before submitting the job."
            )
        except Exception as e:
            log.warning(f"Could not verify if commit {commit} exists on remote repository: {e}")
            log.warning("This may cause the job to fail on the cluster if the commit is not pushed.")

    def _check_image_exists(self, docker_image: str) -> None:
        """Check if a docker image exists in the remote registry.

        Args:
            docker_image (str): Full docker image path including registry, name and tag
        """
        try:
            run_command_as_subprocess(["docker", "manifest", "inspect", docker_image])

        except SubprocessError as e:
            e.add_note(f"Docker image {docker_image} does not exist")
            raise e

    def _build_push_docker(self) -> str:
        """Build, tag and push a docker image."""
        username = getpass.getuser()
        timestamp = time.strftime("%Y%m%d%H%M%S")

        registry, path, name, tag, digest = parse_image_name(self.config.docker.image)
        image = f"{path}/{name}"
        if tag is None:
            tag = f"{username}-{timestamp}"

        docker_image = f"{registry}/{image}:{tag}"

        log.info(f"Building and pushing Docker image {docker_image}")

        cwd = (
            Path(__file__).parent.parent.parent.resolve()
            if not "BUILD_WORKING_DIRECTORY" in os.environ
            else Path(os.environ["BUILD_WORKING_DIRECTORY"])
        )

        command = [
            "bazel",
            "run",
            "//:push_run_image_oci",
            "--",
            f"--repository={registry}/{image}",
            f"--tag={tag}",
        ]

        log.info("Running command: " + " ".join(command))
        try:
            run_command_as_subprocess(command, cwd=cwd)
        except SubprocessError as e:
            e.add_note("bazel process failed")

            raise e

        return docker_image

    def submit_job(
        self,
        job_name: str,
        command: str | None = None,
        job_template_path: str | None = None,
        env_vars: dict[str, str] = {},
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        """Submits job to the cluster."""
        raise NotImplementedError()

    def _search_and_replace_wandb_markers(
        self, sweep_conf, wandb_entity: str, wandb_project: str, wandb_tags: list[str]
    ):
        """Search and replace wandb markers in the sweep configuration"""
        if "command" in sweep_conf:
            comma_separated_tags = ",".join(wandb_tags) if wandb_tags is not None else ""
            sweep_conf["command"] = search_and_replace(
                sweep_conf["command"],
                {
                    "<<wandb_entity>>": wandb_entity,
                    "<<wandb_project>>": wandb_project,
                    "<<wandb_tags>>": "\\[" + comma_separated_tags + "\\]",
                },
            )
        return sweep_conf

    def _start_wandb_sweep(
        self,
        sweep_name: str,
        sweep_conf_path: str,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: list[str],
        dry_run: bool = False,
        verify_sweep: bool = False,
        verbose: bool = False,
    ) -> str:
        """Start a wandb sweep and returns the wandb_sweep_string that identifies it"""

        # Log in to the wandb server if an API key has been provided. Otherwise before-script login is assumed.
        if not dry_run and self.config.wandb.api_key is not None:
            log.info("Logging into wandb...")
            wandb.login(key=self.config.wandb.api_key)

        log.info("wandb")
        log.info(f"  sweep name: {sweep_name}")
        log.info(f"  sweep conf: {sweep_conf_path}")
        log.info(f"  project: {wandb_project}")
        log.info(f"  entity: {wandb_entity}")
        log.info(f"  tags: {wandb_tags}")

        log.info(f"Loading sweep configuration from {sweep_conf_path}")

        with open(sweep_conf_path, "r", encoding="utf-8") as f:
            sweep_conf = yaml.safe_load(f)

        if verify_sweep:
            log.info("Checking sweep configuration")
            try:
                for benchmark_conf_file in sweep_conf["parameters"]["benchmark_config"]["values"]:
                    if os.path.isfile(benchmark_conf_file) or os.path.isfile(
                        os.path.join("./configs", benchmark_conf_file)
                    ):
                        log.info(f"  Found {benchmark_conf_file}")
                    else:
                        raise ConfError(f"{benchmark_conf_file} not found")
            except KeyError as exc:
                raise ConfError("parameters.benchmark_config.values not found in sweep configuration") from exc

        if sweep_name is not None:
            # Overrides the sweep name specified in the sweep configuration file
            # If the sweep name remains unspecified, wandb.sweep() will generate a random unique name for the sweep
            sweep_conf["name"] = sweep_name

        # Substitutions in the sweep command if any
        self._search_and_replace_wandb_markers(sweep_conf, wandb_entity, wandb_project, wandb_tags)

        def str_presenter(dumper, data) -> str:
            """configures yaml for dumping multiline strings properly
            Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
            if data.count("\n") > 0:  # check for multiline string
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        if verbose:
            # Workaround to prevent multiline strings from getting dumped ugly in yaml.dump() and safe_dump()
            # (https://github.com/yaml/pyyaml/issues/240).
            yaml.add_representer(str, str_presenter)
            yaml.representer.SafeRepresenter.add_representer(str, str_presenter)  # to use with yaml.safe_dump

            print(yaml.safe_dump(sweep_conf, sort_keys=False))

        # Creates an empty W&B Sweep on the W&B server, which will wait for connected agents (workers) to execute runs.
        # Agents will connect to the Sweep Controller from Maglev workers.
        log.info(f"Launching wandb sweep '{sweep_name}'")
        for handler in log.handlers:
            handler.flush()
        sweep_id = wandb.sweep(sweep_conf, wandb_entity, wandb_project) if not dry_run else "<sweep_id>"

        wandb_sweep_string = f"{wandb_entity}/{wandb_project}/{sweep_id}"
        return wandb_sweep_string

    def submit_wandb_sweep_job(
        self,
        sweep_name: str,
        sweep_conf_path: str | None = None,
        wandb_sweep_string: str | None = None,
        num_agents: int = 4,
        job_template_path: str | None = None,
        env_vars: dict[str, str] = {},
        dry_run: bool = False,
        verbose: bool = False,
        verify_sweep: bool = False,
    ) -> None:
        """Start a wandb sweep and launch a job to start agents to execute this sweep on the cluster"""

        if sweep_conf_path is not None:
            if wandb_sweep_string is not None:
                raise ValueError("Cannot provide wandb_sweep_string when sweep_conf_path is provided")
            wandb_sweep_string = self._start_wandb_sweep(
                sweep_name=sweep_name,
                sweep_conf_path=sweep_conf_path,
                wandb_entity=self.config.wandb.entity,
                wandb_project=self.config.wandb.project,
                wandb_tags=self.config.wandb.tags,
                dry_run=dry_run,
                verify_sweep=verify_sweep,
                verbose=verbose,
            )
        else:
            assert wandb_sweep_string is not None, "wandb_sweep_string is required when sweep_conf_path is not provided"

        # Submit job to starts wandb agents on cluster
        self._start_wandb_sweep_agents(
            sweep_name=sweep_name,
            num_agents=num_agents,
            wandb_sweep_string=wandb_sweep_string,
            job_template_path=job_template_path,
            env_vars=env_vars,
            dry_run=dry_run,
            verbose=verbose,
        )

    def submit_ray_sweep_job(
        self,
        job_name: str,
        head_node_address: str,
        sweep_conf_path: str | None = None,
        num_workers: int = 4,
        env_vars: dict[str, str] = {},
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        """Start a ray sweep and launch a job to start workers to execute this sweep on the cluster"""
        raise NotImplementedError()

    def _start_wandb_sweep_agents(
        self,
        sweep_name: str,
        num_agents: int,
        wandb_sweep_string: str,
        job_template_path: str | None,
        env_vars: dict[str, str],
        dry_run: bool,
        verbose: bool,
    ):
        """Launch a job on the cluster, with each worker running a wandb sweep agent to execute the runs of a wandb sweep"""
        raise NotImplementedError()

    def _start_ray_workers(
        self,
        job_name: str,
        num_workers: int,
        head_node_address: str,
        env_vars: dict[str, str],
        dry_run: bool,
        verbose: bool,
    ) -> None:
        """Launch a job on cluster, running ray workers that will be used by ray head nodes"""
        raise NotImplementedError()

    def _get_sweep_job_name(self, sweep_name: str, sweep_string: str) -> str:
        return f"{self.username}-nre-sweep-{sweep_name}"

    def _get_ray_worker_job_name(self, job_name: str) -> str:
        return f"{self.username}-nre-ray-workers-{job_name}"
