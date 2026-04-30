# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass

import click

from internal.workflows.cluster_toolbox.base_toolbox import ClusterToolbox
from internal.workflows.cluster_toolbox.maglev_toolbox import MaglevToolbox
from internal.workflows.cluster_toolbox.ngc_toolbox import NGCToolbox
from internal.workflows.cluster_toolbox.ord_toolbox import ORDToolbox


@dataclass(kw_only=True, slots=True, frozen=True)
class ClusterToolboxCLIBaseParams:
    """Parameters passed to non-command-based CLI part"""

    cluster_name: str
    config_name: str


@click.group(invoke_without_command=True)
@click.option(
    "--cluster-name",
    type=str,
    help="Cluster to run job on [maglev, ord, ngc]",
    required=True,
)
@click.option(
    "--config-name",
    type=str,
    help="Config to load, contains ie. docker image, wandb setting and additional cluster-specific parameters",
    required=True,
)
@click.pass_context
def cluster_toolbox_cli(
    ctx,
    cluster_name: str,
    config_name: str,
) -> None:
    """Main entry point for the cluster toolbox"""
    ctx.obj = ClusterToolboxCLIBaseParams(cluster_name=cluster_name, config_name=config_name)


def make_cluster_toolbox(
    cluster_toolbox_base_cli_params: ClusterToolboxCLIBaseParams, hydra_args: list[str] = []
) -> ClusterToolbox:
    """Creates a toolbox based on the selected parameters"""
    toolbox: None | ClusterToolbox = None

    match cluster_toolbox_base_cli_params.cluster_name:
        case "maglev":
            toolbox = MaglevToolbox(cluster_toolbox_base_cli_params.config_name, hydra_args)
        case "ord":
            toolbox = ORDToolbox(cluster_toolbox_base_cli_params.config_name, hydra_args)
        case "ord":
            toolbox = NGCToolbox(cluster_toolbox_base_cli_params.config_name, hydra_args)
        case _:
            raise ValueError(
                f"{cluster_toolbox_base_cli_params.cluster_name} is an unsupported cluster, available options are [maglev, ord, ngc]"
            )

    return toolbox


pass_toolbox_params = click.make_pass_decorator(ClusterToolboxCLIBaseParams, ensure=True)


@cluster_toolbox_cli.command("submit-job")
@click.option("--job-name", type=str, help="Job name", required=True)
@click.option(
    "--command",
    type=str,
    help="Command to run on cluster",
)
@click.option(
    "--job-template-path",
    type=str,
    help="Job template to use on cluster, for most use-cases default should be sufficient (Maglev-specific)",
)
@click.option("--env-var", "env_vars", type=(str, str), help="Environment variables to pass", multiple=True)
@click.option(
    "--dry-run",
    type=bool,
    is_flag=True,
    help="Do not effectively submit a job and allocate resources, only simulate it",
    default=False,
)
@click.option("--verbose", is_flag=True, type=bool, help="Output more info", default=False)
@click.argument("hydra-args", nargs=-1)
@pass_toolbox_params
def submit_job(
    toolbox_params: ClusterToolboxCLIBaseParams,
    job_name: str,
    command: str | None,
    job_template_path: str | None,
    env_vars: tuple[tuple[str, str]],
    dry_run: bool,
    verbose: bool,
    hydra_args: tuple[str],
) -> None:
    """Launch a job on the selected cluster"""
    toolbox = make_cluster_toolbox(toolbox_params, list(hydra_args))
    toolbox.submit_job(
        job_name=job_name,
        command=command,
        job_template_path=job_template_path,
        env_vars=dict(env_vars),
        dry_run=dry_run,
        verbose=verbose,
    )


@cluster_toolbox_cli.command("submit-wandb-sweep-job")
@click.option("--sweep-name", type=str, help="Wandb sweep name", required=True)
@click.option(
    "--sweep-conf-path",
    type=str,
    help="Wandb sweep configuration path. If not provided, will just launch the agents based on --wandb-sweep-string.",
)
@click.option("--wandb-sweep-string", type=str, help="Wandb sweep string.")
@click.option("--num-agents", type=int, help="Number of agents to run the sweep on", default=4)
@click.option(
    "--job-template-path",
    type=str,
    help="Job template to use on cluster, for most use-cases default should be sufficient (Maglev-specific)",
)
@click.option("--env-var", "env_vars", type=(str, str), help="Environment variables to pass", multiple=True)
@click.option("--dry-run", is_flag=True, help="Flag to perform a dry run", default=False)
@click.option("--verbose", is_flag=True, type=bool, help="Output more info", default=False)
@click.argument("hydra-args", nargs=-1)
@pass_toolbox_params
def submit_wandb_sweep_job(
    toolbox_params: ClusterToolboxCLIBaseParams,
    sweep_name: str,
    sweep_conf_path: str | None,
    wandb_sweep_string: str | None,
    num_agents: int,
    job_template_path: str | None,
    env_vars: tuple[tuple[str, str]],
    dry_run: bool,
    verbose: bool,
    hydra_args: tuple[str],
) -> None:
    """Start a wandb sweep and launch a job to start agents to execute this sweep on the selected cluster"""
    toolbox = make_cluster_toolbox(toolbox_params, list(hydra_args))
    toolbox.submit_wandb_sweep_job(
        sweep_name=sweep_name,
        sweep_conf_path=sweep_conf_path,
        wandb_sweep_string=wandb_sweep_string,
        num_agents=num_agents,
        job_template_path=job_template_path,
        env_vars=dict(env_vars),
        dry_run=dry_run,
        verbose=verbose,
    )


@cluster_toolbox_cli.command("submit-ray-sweep-job")
@click.option("--job-name", type=str, help="Wandb sweep name", required=True)
@click.option("--head-node-address", type=str, help="Address of the head node", required=True)
@click.option(
    "--sweep-conf-path",
    type=str,
    help="Wandb-like sweep configuration path. If not provided, will just launch the agents.",
    default=None,
    required=False,
)
@click.option("--num-workers", type=int, help="Number of workers to run the sweep on", default=4)
@click.option("--env-var", "env_vars", type=(str, str), help="Environment variables to pass", multiple=True)
@click.option("--dry-run", is_flag=True, help="Flag to perform a dry run", default=False)
@click.option("--verbose", is_flag=True, type=bool, help="Output more info", default=False)
@click.argument("hydra-args", nargs=-1)
@pass_toolbox_params
def submit_ray_sweep_job(
    toolbox_params: ClusterToolboxCLIBaseParams,
    job_name: str,
    head_node_address: str,
    sweep_conf_path: str | None,
    num_workers: int,
    env_vars: tuple[tuple[str, str]],
    dry_run: bool,
    verbose: bool,
    hydra_args: tuple[str],
) -> None:
    """Start a wandb sweep and launch a job to start agents to execute this sweep on the selected cluster"""
    toolbox = make_cluster_toolbox(toolbox_params, list(hydra_args))
    toolbox.submit_ray_sweep_job(
        job_name=job_name,
        head_node_address=head_node_address,
        sweep_conf_path=sweep_conf_path,
        num_workers=num_workers,
        env_vars=dict(env_vars),
        dry_run=dry_run,
        verbose=verbose,
    )


if __name__ == "__main__":
    cluster_toolbox_cli(show_default=True)
