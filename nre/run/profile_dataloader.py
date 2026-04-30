# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import cProfile
import logging
import time

from typing import Optional

import click
import numpy as np

import nre.systems

from nre.config.parse import parse_typed_config
from nre.run.main import setup_environment_and_logger


@click.command("profile-dataloader")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    required=True,
)
@click.option(
    "--cprofiler-output",
    type=str,
    help="If set, cProfile output will be written to the specified file",
    required=False,
    default=None,
)
@click.argument("hydra-args", nargs=-1)
def profile_dataloader(
    config_name: str,
    cprofiler_output: Optional[str],
    hydra_args: list[str],
) -> None:
    """Entry point to profile dataloader - pulls training batches in isolation and reports associated timings"""

    config = parse_typed_config(config_name=config_name, hydra_args=hydra_args)

    setup_environment_and_logger(config)
    logging.getLogger(__name__).info("RUN 🆔: %s", config.run_id)

    # Make the system
    system = nre.systems.make(config.system.name, config)
    system.setup("train")  # setup is needed to initialize model parameters from datasource

    # Pull batches to estimate performance of dataloader and print timings
    start_time_sec = time.time()
    sample_n_iterations = 100
    samples: list[float] = []

    # Init cProfile conditionally
    profiler = cProfile.Profile()
    if cprofiler_output is None:
        profiler.disable()
    else:
        profiler.enable()

    for batch_idx, _ in enumerate(system.datamodule.train_dataloader()):
        if (batch_idx + 1) % sample_n_iterations == 0:
            samples.append(1 / ((time.time() - start_time_sec) / sample_n_iterations))

            print(f"{sample_n_iterations} iter average: {samples[-1]} it/s")

            start_time_sec = time.time()

    if cprofiler_output is not None:
        profiler.dump_stats(cprofiler_output)
        print(f"cProfile output written to {cprofiler_output}")

    print(f"median {sample_n_iterations} iter average {np.median(samples)} it/s")
