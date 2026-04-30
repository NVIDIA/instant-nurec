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

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from nre.config.base_schema import BaseConfigSchema, Field


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


class TrainerConfig(BaseConfigSchema):
    """
    The trainer config parameters are passed to the pl.Trainer. See
    here: https://lightning.ai/docs/pytorch/stable/common/trainer.html#init for more details.
    """

    # Number of parallel processes used to train the model.
    # Usually == the number of GPUs used.
    world_size: int = Field(
        default=1,
        description="Number of parallel processes used to train the model. Usually == the number of GPUs used. Value 0 means using all available GPUs.",
    )

    # Number of nodes used for training.
    num_nodes: int = Field(
        default=1,
        description="Number of nodes used for training. Value 0 means using all available nodes.",
    )

    # Number of devices used during training.
    # By default it's equal to world_size.
    # Might be < world_size if distributing on multiple nodes
    device_count: Optional[int] = Field(
        default=None,
        description="Number of devices used during training. By default it's equal to world_size. "
        "This value might be smaller than world_size if distributing on multiple nodes",
    )

    # If true, the effective learning rates depend on the world_size.
    #   The learning rates defined in the config considers world_size==1 (one GPU),
    #   and is internally scaled so that the resulting model, when trained with world_size>1,
    #   is similar to the one that would have been obtained when world_size==1.
    #   See https://lightning.ai/docs/pytorch/stable/accelerators/gpu_faq.html
    # If false, the learning rates specified in the config are used unmodified.
    relative_lr: bool = Field(
        default=True,
        description="If true, the effective learning rates depend on the world_size. "
        "The learning rates defined in the config considers world_size==1 (one GPU), "
        "and is internally scaled so that the resulting model, when trained with world_size>1, "
        "is similar to the one that would have been obtained when world_size==1. "
        "If false, the learning rates specified in the config are used unmodified.",
    )

    # If true, the effective training schedule is dependent on the world_size.
    #   The training schedule defined in the config considers world_size==1, and
    #   is internally scaled to compensate from the fact that pytorch lightning divides
    #   the number of steps per epoch by the world_size.
    # If false, the training schedule is not modified.
    relative_schedule: bool = Field(
        default=True,
        description="If true, the effective training schedule is dependent on the world_size. "
        "The training schedule defined in the config considers world_size==1, and is internally "
        "scaled to compensate from the fact that pytorch lightning divides the number of steps "
        "per epoch by the world_size. If false, the training schedule is not modified.",
    )

    # If true, the number of workers created depends on the world_size.
    #   This makes the total number of workers, across all DDP subprocesses not dependent
    #   on the world_size.
    # If false, the number of workers is used as is.
    #   This might overload the system if not used with care, as the effective number of
    #   workers across all DDP subprocesses is multiplied by world_size.
    relative_num_workers: bool = Field(
        default=True,
        description="If true, the number of workers created depends on the world_size. This "
        "makes the total number of workers, across all DDP subprocesses not dependent on the "
        "world_size. If false, the number of workers is used as is. This might overload the "
        "system if not used with care, as the effective number of workers across all DDP "
        "subprocesses is multiplied by world_size.",
    )

    # This is a scaling factor that is useful for tuning the effective batch size for distributed training.
    batch_size_scaling_factor: float = Field(
        default=1.0,
        description="This is a scaling factor that is useful for tuning the effective batch size for distributed training.",
    )

    # This is a scaling factor that is useful for tuning the effective number of training steps for distributed training.
    training_step_scaling_factor: float = Field(
        default=1.0,
        description="This is a scaling factor that is useful for tuning the effective number of training steps for distributed training.",
    )

    max_epochs: int
    check_val_every_n_epoch: int
    val_check_interval: Optional[Union[int, float]] = Field(
        default=None,
        description="Run validation every N training steps (int) or every fraction of the training epoch (float in 0..1). "
        "If None, validates at epoch end (controlled by check_val_every_n_epoch).",
    )
    log_every_n_steps: int
    enable_progress_bar: bool
    num_sanity_val_steps: int
    plugins: Optional[list[Any]] = Field(default=None)

    # Lightning supports doing floating point operations in 64-bit precision (“double”), 32-bit precision (“full”),
    # or 16-bit (“half”) with both regular and bfloat16). This selected precision will have a direct impact in the
    # performance and memory usage based on your hardware. Automatic mixed precision settings are denoted by a
    # "-mixed" suffix, while “true” precision settings have a "-true" suffix:
    precision: int | str = Field(
        description="https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision",
    )

    # Enable anomalies detection in PyTorch Lightning Trainer
    # This controls whether the trainer should detect and handle training anomalies
    # like NaN/Inf values during training
    detect_anomaly: bool = Field(
        default=False,
        description="Enable PyTorch Lightning's built-in anomalies detection. "
        "When enabled, the trainer will detect and handle training anomalies "
        "like NaN/Inf values during training. See: "
        "https://lightning.ai/docs/pytorch/stable/common/trainer.html#detect-anomalies",
    )

    # NCCL timeout for distributed training collective operations (in minutes)
    # This timeout applies to all NCCL collective operations like allreduce, allgather, etc.
    # Default is 30 minutes. Increase if training long sequences or large models.
    nccl_timeout_minutes: int = Field(
        default=30,
        description="Timeout for NCCL collective operations in minutes. "
        "Increase this value if you encounter timeout errors during distributed training "
        "with large models or slow collective operations. Default is 30 minutes.",
    )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)

        # Auto-detect the number of devices per node for multi-node training
        if (slurm_environment := infer_slurm_environment()) is not None:
            if self.device_count is None:
                self.device_count = slurm_environment.num_tasks_per_node
                logging.getLogger(__name__).info(
                    "Auto-detected SLURM environment. Overriding default device_count to %d", self.device_count
                )
            if self.num_nodes == 0:  # The default value of num_nodes is 1
                self.num_nodes = slurm_environment.num_nodes
                logging.getLogger(__name__).info(
                    "Auto-detected SLURM environment. Overriding default num_nodes to %d", self.num_nodes
                )
            if self.world_size == 0:  # The default value of world_size is 1
                self.world_size = self.num_nodes * self.device_count
                logging.getLogger(__name__).info(
                    "Auto-detected SLURM environment. Overriding default world_size to %d", self.world_size
                )
            if self.relative_num_workers:
                # In SLURM environments, CPU cores are allocated proportionally to the number of GPUs, so reducing num_workers
                # by the number of GPUs was incorrect. Now, num_workers is set appropriately (to be a fixed value per instance)
                # when SLURM environment is detected.
                logging.getLogger(__name__).warning("Relative num workers should not be used with SLURM environment.")
                self.relative_num_workers = False

        # TODO: Add support for other distributed resource management systems based
        # on customer needs.
        else:
            if self.world_size == 0:
                self.world_size = torch.cuda.device_count()
            if self.num_nodes == 0:
                self.num_nodes = 1
            if self.device_count is None:
                self.device_count = self.world_size

        assert self.world_size == self.num_nodes * self.device_count, (
            f"The parameter 'trainer.world_size' ({self.world_size}) is not equal to the product of "
            f"'trainer.num_nodes' ({self.num_nodes}) and 'trainer.device_count' ({self.device_count}). "
            f"Please set 'trainer.world_size' correctly to the number of total GPUs across all nodes to use for training."
        )

        # Log as many times as single rank would log.
        if self.log_every_n_steps > 1:
            self.log_every_n_steps = (self.log_every_n_steps + (self.world_size - 1)) // self.world_size

        logging.getLogger(__name__).info("World Size: %d", self.world_size)
        logging.getLogger(__name__).info("Node Count: %d", self.num_nodes)
        logging.getLogger(__name__).info("Device Count per Node: %d", self.device_count)
        logging.getLogger(__name__).info("Log every %d steps", self.log_every_n_steps)
