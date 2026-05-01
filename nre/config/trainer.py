# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class SlurmEnvironment:
    """Dataclass to hold SLURM environment variables for distributed training."""

    num_nodes: int
    num_tasks_per_node: int


def infer_slurm_environment() -> Optional[SlurmEnvironment]:
    """Infers the SLURM environment variables if available.
    On SLURM, distributed training must be configured according to specific rules. If
    certain parameters are set to their default values, they will be overwritten to
    ensure compliance with these rules.
    Reference https://github.com/Lightning-AI/pytorch-lightning/blob/2.5.2/src/lightning/fabric/plugins/environments/slurm.py#L158-L172

    Returns:
        Optional[SlurmEnvironment]: A dataclass containing SLURM environment variables or None if not available.
    """
    if "SLURM_NTASKS_PER_NODE" in os.environ and "SLURM_NNODES" in os.environ:
        return SlurmEnvironment(
            num_nodes=int(os.environ["SLURM_NNODES"]),
            num_tasks_per_node=int(os.environ["SLURM_NTASKS_PER_NODE"]),
        )
    return None
