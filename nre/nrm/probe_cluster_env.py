# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Cluster environment probe: prints system info and runs a minimal distributed training loop.

Usage:
    bazel run //nre/nrm:probe_cluster_env

This tool is designed to diagnose cluster environments by:
  1. Inferring SLURM environment via infer_slurm_environment().
  2. Printing ulimit, all environment variables, and GPU topology.
  3. Running a trivial PyTorch Lightning training loop (~50 iterations) with NCCL
     debug logging enabled so that collective-communication topology is visible.
"""

import logging
import os
import platform
import resource
import socket
import subprocess
import sys

from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

from nre.config.trainer import infer_slurm_environment


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

SEPARATOR = "=" * 80
NUM_TRAIN_ITERS = 50
HIDDEN_DIM = 64


# ---------------------------------------------------------------------------
# System information helpers
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_basic_system_info() -> None:
    _print_header("BASIC SYSTEM INFO")
    print(f"  Hostname        : {socket.gethostname()}")
    print(f"  FQDN            : {socket.getfqdn()}")
    print(f"  Platform        : {platform.platform()}")
    print(f"  Python          : {sys.version}")
    print(f"  PyTorch         : {torch.__version__}")
    print(f"  Lightning       : {pl.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  cuDNN version   : {torch.backends.cudnn.version()}")
        print(f"  GPU count       : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}           : {props.name}  ({props.total_memory / (1024**3):.1f} GiB)")
    print(f"  Timestamp       : {datetime.now().isoformat()}")


def print_ulimit_info() -> None:
    _print_header("ULIMIT / RESOURCE LIMITS")
    resource_names = [
        ("RLIMIT_NOFILE", resource.RLIMIT_NOFILE, "Max open files"),
        ("RLIMIT_NPROC", resource.RLIMIT_NPROC, "Max user processes"),
        ("RLIMIT_STACK", resource.RLIMIT_STACK, "Stack size (bytes)"),
        ("RLIMIT_MEMLOCK", resource.RLIMIT_MEMLOCK, "Max locked memory (bytes)"),
        ("RLIMIT_AS", resource.RLIMIT_AS, "Max address space (bytes)"),
        ("RLIMIT_DATA", resource.RLIMIT_DATA, "Max data segment (bytes)"),
        ("RLIMIT_FSIZE", resource.RLIMIT_FSIZE, "Max file size (bytes)"),
        ("RLIMIT_CORE", resource.RLIMIT_CORE, "Max core file size (bytes)"),
        ("RLIMIT_CPU", resource.RLIMIT_CPU, "Max CPU time (seconds)"),
    ]
    for name, res, desc in resource_names:
        soft, hard = resource.getrlimit(res)
        soft_str = "unlimited" if soft == resource.RLIM_INFINITY else str(soft)
        hard_str = "unlimited" if hard == resource.RLIM_INFINITY else str(hard)
        print(f"  {name:<20s} ({desc}): soft={soft_str}, hard={hard_str}")


def print_env_variables() -> None:
    _print_header("ENVIRONMENT VARIABLES")
    for key in sorted(os.environ):
        print(f"  {key}={os.environ[key]}")


def print_slurm_info() -> None:
    _print_header("SLURM ENVIRONMENT")
    slurm_env = infer_slurm_environment()
    if slurm_env is not None:
        print(f"  Detected SLURM environment:")
        print(f"    num_nodes          = {slurm_env.num_nodes}")
        print(f"    num_tasks_per_node = {slurm_env.num_tasks_per_node}")
        slurm_keys = sorted(k for k in os.environ if k.startswith("SLURM"))
        if slurm_keys:
            print(f"  All SLURM variables ({len(slurm_keys)}):")
            for k in slurm_keys:
                print(f"    {k}={os.environ[k]}")
    else:
        print("  No SLURM environment detected (SLURM_NTASKS_PER_NODE / SLURM_NNODES not set).")


def print_nvidia_topology() -> None:
    _print_header("NVIDIA GPU TOPOLOGY (nvidia-smi topo -m)")
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(result.stdout if result.returncode == 0 else f"  nvidia-smi failed: {result.stderr}")
    except FileNotFoundError:
        print("  nvidia-smi not found on PATH.")
    except subprocess.TimeoutExpired:
        print("  nvidia-smi timed out.")


# ---------------------------------------------------------------------------
# Minimal Lightning module & dataset
# ---------------------------------------------------------------------------


class DummyDataset(Dataset):
    """Generates random (input, target) pairs for a regression task."""

    def __init__(self, size: int = NUM_TRAIN_ITERS * 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(HIDDEN_DIM)
        y = torch.randn(HIDDEN_DIM)
        return x, y


class DummyModel(pl.LightningModule):
    """A trivial 2-layer MLP trained with MSE loss — used only to exercise the
    distributed runtime (NCCL collectives, GPU memory, etc.)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _configure_nccl_debug() -> None:
    """Enable NCCL debug logging so topology / ring info is printed."""
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,GRAPH,NET"


def main() -> None:
    _configure_nccl_debug()

    print_basic_system_info()
    print_ulimit_info()
    print_slurm_info()
    print_nvidia_topology()
    print_env_variables()

    # Determine distributed strategy
    slurm_env = infer_slurm_environment()
    num_devices = 1
    num_nodes = 1
    strategy: str | DDPStrategy = "auto"

    if slurm_env is not None:
        num_devices = slurm_env.num_tasks_per_node
        num_nodes = slurm_env.num_nodes
        strategy = DDPStrategy(find_unused_parameters=False)
        log.info("Using DDP strategy with %d devices across %d nodes (SLURM).", num_devices, num_nodes)
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_devices = torch.cuda.device_count()
        strategy = DDPStrategy(find_unused_parameters=False)
        log.info("Using DDP strategy with %d local GPUs.", num_devices)
    elif torch.cuda.is_available():
        num_devices = 1
        log.info("Using single-GPU strategy.")
    else:
        num_devices = 1
        log.info("No GPU detected — running on CPU.")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    _print_header(f"STARTING TRAINING PROBE ({NUM_TRAIN_ITERS} iterations)")
    print(f"  accelerator = {accelerator}")
    print(f"  devices     = {num_devices}")
    print(f"  num_nodes   = {num_nodes}")
    print(f"  strategy    = {strategy}")

    model = DummyModel()
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_devices,
        num_nodes=num_nodes,
        strategy=strategy,
        max_steps=NUM_TRAIN_ITERS,
        max_epochs=100,
        enable_checkpointing=False,
        enable_model_summary=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        logger=False,
    )

    trainer.fit(model, train_dataloaders=dataloader)

    _print_header("TRAINING PROBE COMPLETED SUCCESSFULLY")
    print(f"  Global steps executed: {trainer.global_step}")
    print(f"  Final train loss    : {trainer.callback_metrics.get('train_loss', 'N/A')}")


if __name__ == "__main__":
    main()
