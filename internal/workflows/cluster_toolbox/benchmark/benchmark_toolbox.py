# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from internal.workflows.cluster_toolbox.maglev_toolbox import MaglevToolbox
from internal.workflows.cluster_toolbox.utils import (
    parse_image_name,
    parse_wandb_sweep_string,
)


class BenchmarkToolbox(MaglevToolbox):
    sweep_job_template_path: str = "internal/workflows/cluster_toolbox/job_templates/maglev/benchmark_workflow.yaml"
    sweep_template_path: str = "internal/workflows/cluster_toolbox/wandb_sweep_configs/maglev/benchmark_sweep.yaml"

    def __init__(self, config_name: str, hydra_args: list[str]) -> None:
        super().__init__(config_name, hydra_args)

        if "benchmark" not in self.config.wandb.tags:
            self.config.wandb.tags.append("benchmark")

    def launch_benchmark(
        self,
        num_agents: int = 4,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        """Start a benchmark sweep and launch a job to start agents to execute this sweep on Maglev"""

        _, _, image_name, image_version, _ = parse_image_name(self.docker_image)

        sweep_name = f"benchmark:{image_name}:{image_version}"

        self.submit_wandb_sweep_job(
            sweep_name=sweep_name,
            sweep_conf_path=self.sweep_template_path,
            wandb_sweep_string=None,
            dry_run=dry_run,
            num_agents=num_agents,
            verbose=verbose,
            verify_sweep=True,
        )

    def _get_sweep_job_name(self, sweep_name: str, sweep_string: str) -> str:
        registry, path, name, tag, digest = parse_image_name(self.docker_image)
        entity, project, sweep_id = parse_wandb_sweep_string(sweep_string)
        return f"benchmark-{name}-{tag}-sweep-{sweep_id}"
