# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import subprocess
import tempfile

from dataclasses import dataclass
from datetime import datetime

import yaml

from omegaconf import DictConfig

from internal.workflows.cluster_toolbox.base_toolbox import ClusterToolbox
from internal.workflows.cluster_toolbox.job_templates.ord.ord_template import (
    GITLAB_MASTER_URL,
    ORD_DEFS,
    ORD_JOBINFO,
    ORD_LAUNCH_PRE_COMMANDS_BAZEL,
    ORD_LAUNCH_PRE_COMMANDS_DOCKER,
    ORD_LAUNCH_PRE_COMMANDS_DOCKER_PYTHON,
    ORD_PREPARE_PRIVATE_WORKSPACE,
    ORD_PREPARE_SHARED_WORKSPACE,
    ORD_TEMPLATE_BAZEL,
    ORD_TEMPLATE_DOCKER,
    ORD_TEMPLATE_DOCKER_PYTHON,
)
from internal.workflows.cluster_toolbox.utils import (
    convert_time_to_timeout,
    generate_runstamp,
    get_formatted_datetime,
    mount_dict_to_string,
    run_command_as_subprocess,
    write_run_bash,
)
from nre.config.version import get_version  # type: ignore
from nre.utils.misc import unpack_optional  # type: ignore


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


class ORDToolbox(ClusterToolbox):
    def __init__(self, config_name: str, hydra_args: list[str] = []) -> None:
        super().__init__(config_name, hydra_args)

        # Get the git commit hash
        if self.config.git_commit is not None:
            # NOTE: Certain git operations (e.g., git fetch) will only work if the commit is a long hash
            self.commit = subprocess.check_output(["git", "rev-parse", self.config.git_commit]).decode("ascii").strip()
        else:
            # NOTE: What if the git repo is dirty? This will ignore dirty states
            self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
            # Check if the git repo is dirty
            if subprocess.check_output(["git", "status", "--porcelain"]):
                log.warning("Git repository is dirty. Uncommitted changes will not be included in the job.")

        log.info(f"Commit: {self.commit}")

        match self.config.exec_mode:
            case "docker":
                self._check_image_exists(self.docker_image)

            case "bazel" | "docker_python":
                self._check_commit_exists_on_remote(self.commit)

            case _:
                raise ValueError(f"Invalid execution mode: {self.config.exec_mode}")

    def submit_job(
        self,
        job_name: str,
        command: str | None = None,
        job_template_path: str | None = None,
        env_vars: dict[str, str] = {},
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        if not self.config.no_resume:
            raise NotImplementedError(f"{self.__class__.__name__}: we currently do not support resuming training runs")

        if command is None:
            raise NotImplementedError(f"{self.__class__.__name__}: `command` needs to be set")

        @dataclass
        class LocalRemoteFilePair:
            filename: str
            local_dir: str
            remote_dir: str

            @property
            def remote_path(self) -> str:
                return os.path.join(self.remote_dir, self.filename)

            @property
            def local_path(self) -> str:
                return os.path.join(self.local_dir, self.filename)

        with tempfile.TemporaryDirectory() as tmp_dir:
            assert isinstance(self.config, DictConfig)
            include_config_name = self.config.exec_path.get("include_config_name", False)
            include_dataset_name = self.config.exec_path.get("include_dataset_name", False)
            run_stamp, run_id = generate_runstamp(command, include_config_name, include_dataset_name)
            run_remote_dir = os.path.join(self.config.exec_path.remote, job_name, run_stamp)
            run_remote_log = os.path.join(run_remote_dir, "log")

            # Merge default env vars from config with passed env vars (passed ones take precedence)
            default_env_vars = self.config.get("default_env_vars", {})
            env_vars = {**default_env_vars, **env_vars}  # CLI env_vars override config ones
            env_vars["NRE_ENV_RUN_ID"] = run_id

            # Write bash script to be excecuted on the node (contains actual training command)
            ord_launch_job = LocalRemoteFilePair(filename="launch.sh", local_dir=tmp_dir, remote_dir=run_remote_dir)
            self._write_run_job_script(
                ord_launch_job.local_path,
                job_name,
                command,
                env_vars,
            )

            # Write bash script to be excecuted on ORD to submit the job (contains `srun` command)
            ord_submit_job = LocalRemoteFilePair(filename="submit.sh", local_dir=tmp_dir, remote_dir=run_remote_dir)
            self._write_submit_job_script(
                ord_submit_job.local_path,
                job_name,
                run_remote_dir,
                run_remote_log,
                launch_script=ord_launch_job.remote_path,
            )

            # Run local commands to copy files and submit job to ORD
            uname = self.config.user
            server = self.config.server

            system_cmd = [
                f'ssh {uname}@{server} "mkdir -p {run_remote_dir}"',
                f"scp -q {ord_launch_job.local_path} {uname}@{server}:{ord_launch_job.remote_path}",
                f"scp -q {ord_submit_job.local_path} {uname}@{server}:{ord_submit_job.remote_path}",
            ]

            log.info(f"Job name {job_name}")
            log.info(f"Find log results on {uname}@{server}:{run_remote_dir}")
            if not dry_run:
                try:
                    for cmd in system_cmd:
                        if verbose:
                            log.info(cmd)
                        os.system(cmd)
                    # Submit job to ORD
                    # We pass in --requeue flag explicitly because some clusters by default do not
                    # let jobs requeue.
                    output_submit = subprocess.check_output(
                        f'ssh {uname}@{server} "sbatch --requeue {ord_submit_job.remote_path}"', shell=True
                    )
                    if "Submitted batch job" in output_submit.decode("utf-8"):
                        job_id = output_submit.decode("utf-8").split(" ")[-1].rstrip()
                    else:
                        raise ValueError("Unknown SLURM Job ID")
                    log.info(f"Run the following command to print outputs for the first task:")
                    log.info(f"\tssh {uname}@{server} tail -f {run_remote_log}_{job_id}_1.log")
                    if (ntasks := self.config.num_gpus * self.config.num_nodes) > 1:
                        log.info(
                            f"Distributed training detected with ntasks={ntasks}. "
                            f"The default log file only contains rank 0 outputs. "
                            f"Additional outputs are available at:"
                        )
                        log.info(f"\t{run_remote_log}_{job_id}_1_r1.log")
                        log.info(f"\t⋮")
                        log.info(f"\t{run_remote_log}_{job_id}_1_r{ntasks - 1}.log")
                    log.info("Job submitted to ORD")
                except Exception as e:
                    e.add_note("ORD job launch failed")
                    raise e
                return
            else:
                log.warning("This was a dry run (simulation). No ORD, Docker image or W&B sweep have been created.")

    def _write_submit_job_script(
        self, filename: str, job_name: str, remote_dir: str, log_file_name: str, launch_script: str
    ) -> None:
        if segment := getattr(self.config, "segment", ""):
            segment_string = f"#SBATCH --segment={segment}"
        else:
            segment_string = ""

        ord_env_vars_definition_cmd = ORD_DEFS.format(
            job_name=job_name,
            job_info=ORD_JOBINFO.replace("{remote_dir}", remote_dir),
            # -- ORD setup --
            account=self.config.team,
            partition=",".join(self.config.partition),
            num_nodes=self.config.num_nodes,
            num_gpus=self.config.num_gpus,
            num_jobs=self.config.num_workers,  # number of sweep agents / array jobs
            time=self.config.time,
            timeout=convert_time_to_timeout(self.config.time),
            # -- launch methods --
            git_commit=self.commit,
            docker_link=self.docker_image,
            docker_sqsh=os.path.join(
                self.config.docker.image_cache_path, os.path.basename(self.docker_image) + ".sqsh"
            ),
            docker_python_sqsh=self.config.docker_python.dev_image_path,
            bazel_dev_image=self.config.bazel.dev_image_path,
            bazel_dev_cache=self.config.bazel.dev_cache_path,
            bazel_tmp_cache=f"{remote_dir}/runcache"
            if self.config.keep_runcache
            else f"{self.config.scratch_prefix}/cache_$UNIQUE_JOB_ID",
            scratch_prefix=self.config.scratch_prefix,
            scratch_in_container="1" if self.config.scratch_in_container else "0",
            bazel_cache_prefill=f"{self.config.bazel.cache_prefill}",
            # -- commands --
            launch_script=launch_script,
            # -- other --
            log=log_file_name,
            mount=mount_dict_to_string(self.config.mounts),
            segment_string=segment_string,
        )

        ord_cmd_list = []

        if self.config.keep_workspace:
            ord_cmd_list.append(ORD_PREPARE_SHARED_WORKSPACE)
        else:
            ord_cmd_list.append(ORD_PREPARE_PRIVATE_WORKSPACE)

        match self.config.exec_mode:
            case "docker":
                ord_cmd_list.append(ORD_TEMPLATE_DOCKER)

            case "bazel":
                ord_cmd_list.append(ORD_TEMPLATE_BAZEL)

            case "docker_python":
                ord_cmd_list.append(ORD_TEMPLATE_DOCKER_PYTHON)

            case _:
                raise ValueError(f"Invalid execution mode: {self.config.exec_mode}")

        write_run_bash(filename=filename, exp_cmds=ord_cmd_list, pre_cmds=[ord_env_vars_definition_cmd])

    def _write_run_job_script(self, filename: str, job_name: str, command: str, env_vars: dict[str, str] = {}) -> None:
        # Check if the command is valid
        if "/usr/local/bin/bazelisk" in command and os.environ.get("WORKFLOW_SKIP_BAZELISK_EXECUTABLE_CHECK") != "1":
            raise ValueError(
                "Invalid command detected: '/usr/local/bin/bazelisk' is being used to invoke Bazel. This can lead to "
                "toolbox malfunctions. To avoid issues, please use `bazelisk` or `bazel` directly instead. "
                "If you intentionally want to bypass this check, set the environment variable "
                "`WORKFLOW_SKIP_BAZELISK_EXECUTABLE_CHECK=1`."
            )

        # Setup common environment variables
        pre_cmds = []
        pre_cmds.append(f"#!/bin/bash")
        pre_cmds.append(f"set -eu -o pipefail")

        pre_cmds.append("")
        pre_cmds.append(f'export JOB_NAME="{job_name}"')
        pre_cmds.append(f'export RUN_NAME="{job_name}"')
        pre_cmds.append(f'export ACC_NAME="{self.config.user}"')

        # Redirect non-master rank outputs to files
        pre_cmds.append("")
        pre_cmds.append("# Redirect non-master rank outputs to files")
        pre_cmds.append('if [ "${SLURM_PROCID:-}" != "0" ]; then')
        pre_cmds.append("  exec > ${SRUN_LOG}_${UNIQUE_JOB_ID}_r${SLURM_PROCID}.log 2>&1")
        pre_cmds.append("fi")

        ## Pip wheel cache
        # FIXME: `pip install` is not thread safe, setting PIP_CACHE_DIR causes issues for multi-gpu training
        pre_cmds.append("")
        pre_cmds.append("# Pip Wheel Cache")
        # Make sure `pip` downloads wheels to the common persistent mount point '/keepcache' within all containers
        # Note: to fully make use of bazel repo caching for the downloaded and extracted wheels we'll
        #       have to switch from WORKSPACE to more modern bzlmod-based bazel external repository management, see
        #       https://rules-python.readthedocs.io/en/latest/pypi-dependencies.html#bazel-downloader-and-multi-platform-wheel-hub-repository
        private_pip_cache = "/scratch/pip_cache_$SLURM_LOCALID"
        pre_cmds.append(f"export PIP_CACHE_DIR={private_pip_cache}")
        # Don't run `pip`` within rules_python in isolated mode
        # - https://rules-python.readthedocs.io/en/latest/environment-variables.html#envvar-RULES_PYTHON_PIP_ISOLATED
        # - https://pip.pypa.io/en/stable/cli/pip/#cmdoption-isolated
        # to respect the globally set PIP_CACHE_DIR environment variable
        pre_cmds.append("export RULES_PYTHON_PIP_ISOLATED=0")
        # Now we can prefill the pip cache with a recent heuristics
        pre_cmds.append(f"if [ -f '{self.config.pip_cache}' ]; then")
        pre_cmds.append(f"  mkdir -p $PIP_CACHE_DIR")
        pre_cmds.append(f"  tar --strip-components=2 -xf {self.config.pip_cache} -C $PIP_CACHE_DIR")
        pre_cmds.append(f"else")
        pre_cmds.append(f"  echo 'PIP cache file ({self.config.pip_cache}) does not exist, skipping it!!!'")
        pre_cmds.append(f"fi")

        # Environment variable required to allow DDP to execute correctly
        pre_cmds.append("")
        pre_cmds.append("# PyTorch DDP Variables")
        pre_cmds.append("export LOCAL_RANK=$SLURM_LOCALID")
        pre_cmds.append("export NODE_RANK=$SLURM_NODEID")

        # Set the API key for wandb
        if self.config.wandb.api_key is not None:
            pre_cmds.append("")
            pre_cmds.append("# WandB API Key")
            pre_cmds.append(f'export WANDB_API_KEY="{self.config.wandb.api_key}"')

        # Add additional the environment variables
        pre_cmds.append("")
        pre_cmds.append("# Additional Environment Variables")
        for key, value in env_vars.items():
            pre_cmds.append(f'export {key}="{value}"')

        # Add the pre-launch
        pre_cmds.append("")
        pre_cmds.append("# ----------------------------------------------------")
        pre_cmds.append("# Pre-launch commands")
        match self.config.exec_mode:
            case "docker":
                pre_cmds.append(ORD_LAUNCH_PRE_COMMANDS_DOCKER)
            case "docker_python":
                pre_cmds.append(ORD_LAUNCH_PRE_COMMANDS_DOCKER_PYTHON)
            case "bazel":
                pre_cmds.append(ORD_LAUNCH_PRE_COMMANDS_BAZEL)
            case _:
                raise ValueError(f"Invalid execution mode: {self.config.exec_mode}")
        pre_cmds.append("# End of pre-launch commands")
        pre_cmds.append("# ----------------------------------------------------")

        # Generate the run bash script
        write_run_bash(filename, exp_cmds=[command], pre_cmds=pre_cmds)

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

        wandb_agent_args = ""
        if self.config.wandb.max_job_per_agent is not None:
            wandb_agent_args = f"--count {self.config.wandb.max_job_per_agent}"

        # We now supports running wandb CLI through bazel, but we need to be careful about the working directory. By default,
        # bazel starts its target under its own BUILD_WORKING_DIRECTORY directory. There is no way to properly configure this
        # according to https://github.com/bazelbuild/bazel/issues/2579#issuecomment-626840214. Therefore we need to achieve this
        # with --run_under.
        wandb = f"bazel run --run_under='cd /workspace &&' //internal/workflows/cluster_toolbox:wandb --"

        commands = [
            # Unset NRE_ENV_RUN_ID to prevent wandb agents from inheriting the parent job's run ID.
            # NRE_ENV_RUN_ID is normally used to provide consistent run IDs across distributed jobs,
            # but for wandb sweeps, each agent should generate unique run IDs for individual runs.
            "unset NRE_ENV_RUN_ID",
            "pip install shortuuid",
            "nvidia-smi",
            f"{wandb} sweep {wandb_sweep_string} --resume",
            f"{wandb} agent {wandb_agent_args} {wandb_sweep_string}",
        ]

        # Call wandb sweep --resume before potentially resuming a preempted run because it will otherwise error if the sweep is marked as finished
        # (which grid sweep gets marked as after all runs are started)
        # Also attempt to retry the wandb install since it sometimes flakes out
        self.submit_job(
            job_name=self._get_sweep_job_name(sweep_name=sweep_name, sweep_string=wandb_sweep_string),
            command=" && ".join(commands),
            env_vars={"WANDB_SWEEP_STRING": wandb_sweep_string, "DATETIME": get_formatted_datetime()} | env_vars,
            dry_run=dry_run,
            verbose=verbose,
        )

    @staticmethod
    def _resolve_ssh_hostname(address: str) -> str:
        """Resolve an SSH config alias to its real HostName. Returns the address unchanged if it's not an alias."""
        try:
            output = subprocess.check_output(["ssh", "-G", address], stderr=subprocess.DEVNULL).decode("ascii")
            for line in output.splitlines():
                if line.startswith("hostname "):
                    resolved = line.split(None, 1)[1]
                    if resolved != address:
                        log.info(f"Resolved SSH alias '{address}' -> '{resolved}'")
                    return resolved
        except subprocess.CalledProcessError:
            pass
        return address

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

        # Deploy the scheduler on the head node
        if sweep_conf_path is not None and not dry_run:
            # Ensure the ray head binary exists on head_node_address, if not git clone the repository there
            # and build the binary.
            head_node_full_address = f"{self.config.user}@{head_node_address}"
            head_node_code_exists = (
                os.system(f"ssh {head_node_full_address} 'test -d {self.config.exec_path.remote_ray_head}/code'") == 0
            )

            if not head_node_code_exists:
                log.info(f"Ray head code does not exist on {head_node_full_address}. Cloning and building...")
                clone_commands = [
                    f"mkdir -p {self.config.exec_path.remote_ray_head}",
                    f"cd {self.config.exec_path.remote_ray_head}",
                    # SSH Agent is required to clone the repository
                    'eval "$(ssh-agent -s)" && for k in ~/.ssh/*.pub; do [[ -f "${k%.pub}" ]] && ssh-add "${k%.pub}"; done',
                    f"git clone {GITLAB_MASTER_URL} code",
                    "cd code",
                    f"git checkout {self.commit}",
                ]
                run_command_as_subprocess(["ssh", head_node_full_address, " && ".join(clone_commands)])

            # Make sure that the bazelisk binary is available
            bazel_binary_path = os.path.join(self.config.exec_path.remote_ray_head, "bazel")
            bazelisk_available = os.system(f"ssh {head_node_full_address} 'test -f {bazel_binary_path}'") == 0
            if not bazelisk_available:
                log.info(f"Bazel not found on {head_node_full_address}. Installing...")
                install_bazel_commands = [
                    f"cd {self.config.exec_path.remote_ray_head}",
                    f"wget -O {bazel_binary_path} https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64",
                    f"chmod +x {bazel_binary_path}",
                ]
                run_command_as_subprocess(["ssh", head_node_full_address, " && ".join(install_bazel_commands)])

            # Make sure no other scheduler is running (both bazel and the python script under it)
            scheduler_pattern = "ray:(scheduler)|ray/scheduler\.py"
            scheduler_running = os.system(f"ssh {head_node_full_address} 'pgrep -f \"{scheduler_pattern}\"'") == 0
            if scheduler_running:
                raise RuntimeError(
                    f"Scheduler is already running on {head_node_full_address}. "
                    f"Please stop it first with `ssh {head_node_full_address} 'pkill -9 -f \"{scheduler_pattern}\"'`"
                )

            # job identifier used for scheduler log and temporary file on remote
            job_identifier = f"{job_name}-{datetime.now().strftime('%m%d%H%M%S')}"

            # Copy the sweep configuration to temporary folder on head node
            with tempfile.TemporaryDirectory() as temp_dir:
                with open(sweep_conf_path, "r") as f:
                    sweep_conf = yaml.safe_load(f)
                self._search_and_replace_wandb_markers(
                    sweep_conf, self.config.wandb.entity, self.config.wandb.project, self.config.wandb.tags
                )
                temp_sweep_conf_path = os.path.join(temp_dir, "sweep.yaml")
                with open(temp_sweep_conf_path, "w") as f:
                    yaml.safe_dump(sweep_conf, f)
                remote_temp_sweep_conf_path = f"/tmp/{job_identifier}.yaml"
                run_command_as_subprocess(
                    ["scp", temp_sweep_conf_path, f"{head_node_full_address}:{remote_temp_sweep_conf_path}"]
                )

            # Launch the scheduler on the head node (split into build and run steps to check build status since run step is non-blocking)
            scheduler_log_path = f"{self.config.exec_path.remote_ray_head}/job-{job_identifier}"
            launch_scheduler_commands = [
                f"cd {self.config.exec_path.remote_ray_head}/code",
                # Pass 1: Build the libray and up the head node
                f"{bazel_binary_path} run //internal/workflows/ray:scheduler -- start-head-node",
                "if [ $? -ne 0 ]; then exit 1; fi",
                # Pass 2: nohup start the sweep
                f"nohup ./bazel-bin/internal/workflows/ray/scheduler -- run-sweep {remote_temp_sweep_conf_path} --log-file-dir "
                f"{scheduler_log_path} > /dev/null 2>&1 < /dev/null &",
            ]
            run_command_as_subprocess(["ssh", head_node_full_address, "; ".join(launch_scheduler_commands)])

        # Submit jobs to cluster to start ray workers
        self._start_ray_workers(
            job_name=job_name,
            num_workers=num_workers,
            head_node_address=head_node_address,
            env_vars=env_vars,
            dry_run=dry_run,
            verbose=verbose,
        )

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
        head_node_address = self._resolve_ssh_hostname(head_node_address)

        if num_workers == 0:
            log.warning(f"No ray workers requested for job {job_name}. Skipping...")
            return

        self.config.num_workers = num_workers

        ray_worker_prefix = "bazel run --run_under='cd /workspace &&' //internal/workflows/ray:worker -- "
        ray_worker_start_command = ray_worker_prefix + f"start --address={head_node_address}:6379"

        commands = [
            # Unset NRE_ENV_RUN_ID to prevent wandb agents from inheriting the parent job's run ID.
            # NRE_ENV_RUN_ID is normally used to provide consistent run IDs across distributed jobs,
            # but for wandb sweeps, each agent should generate unique run IDs for individual runs.
            "unset NRE_ENV_RUN_ID",
            "nvidia-smi",
            "export RAY_SHOULD_STOP_PATH=/scratch/ray_should_stop",
            # Due to a ray bug: https://github.com/ray-project/ray/issues/46453, calling ray start
            # might result in random failure due to port conflicts. We workaround this temporarily by
            # trying to start multiple times before giving up.
            # Note that the start command is non-blocking after the process is launched.
            f"({' || '.join([ray_worker_start_command] * 3)})",
            # Wait forever until (or the job gets killed due to timeout)
            "while [ ! -f $RAY_SHOULD_STOP_PATH ]; do sleep 1; done",
            # About to stop the job, but before that provides a graceful period for the ray workers to finish their work.
            "sleep 20",
            # Remove the file
            "rm $RAY_SHOULD_STOP_PATH",
            # NB [JH]: It seems that if multiple jobs are running on the same node, the stop command
            # might kill other ray workers as well. Therefore we just let the script finish and have
            # slurm to kill workers related only to this job.
            # ray_worker_prefix + "stop --force",
        ]

        self.submit_job(
            job_name=self._get_ray_worker_job_name(job_name=job_name),
            command=" && ".join(commands),
            env_vars={"DATETIME": get_formatted_datetime()} | env_vars,
            dry_run=dry_run,
            verbose=verbose,
        )
