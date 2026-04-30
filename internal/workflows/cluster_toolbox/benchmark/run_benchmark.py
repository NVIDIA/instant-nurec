# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import click

from internal.workflows.cluster_toolbox.benchmark.benchmark_toolbox import BenchmarkToolbox


@click.command("launch_benchmark")
@click.option("--num-agents", type=int, help="Number of agents to run the sweep on", default=4)
@click.option("--dry-run", is_flag=True, help="Flag to perform a dry run", default=False)
@click.option("--verbose", is_flag=True, type=bool, help="Output more info", default=False)
@click.argument("hydra-args", nargs=-1)
def launch_benchmark(
    num_agents: int,
    dry_run: bool,
    verbose: bool,
    hydra_args: list[str] = [],
) -> None:
    """Entrypoint to launch benchmark sweep"""
    benchmark_toolbox = BenchmarkToolbox(config_name="benchmark.yaml", hydra_args=hydra_args)
    benchmark_toolbox.launch_benchmark(num_agents=num_agents, dry_run=dry_run, verbose=verbose)


if __name__ == "__main__":
    launch_benchmark()
