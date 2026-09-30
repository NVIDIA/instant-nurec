# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import os

from pathlib import Path
from typing import Sequence

import yaml
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from instant_nurec.config_schema.train import TrainingConfig
from instant_nurec.training.callbacks import AtomicCheckpointAndExitOnSignalCallback, PreemptionInterrupt
from instant_nurec.training.data import ResumableDataModuleCallback, TrainingDataModule
from instant_nurec.training.system import TrainingSystem


logger = logging.getLogger(__name__)


def _is_distributed_run(config: TrainingConfig) -> bool:
    if config.system.num_nodes > 1:
        return True
    devices = config.system.devices
    if isinstance(devices, int):
        if devices == -1:
            return torch.cuda.device_count() > 1
        return devices > 1
    if devices == "auto":
        if config.system.accelerator == "cpu":
            return False
        return torch.cuda.device_count() > 1
    if devices == "-1":
        return torch.cuda.device_count() > 1
    return len([item for item in devices.split(",") if item.strip()]) > 1


def resolve_training_strategy(config: TrainingConfig) -> str:
    """Apply the reference find-unused requirement to distributed runs."""

    strategy = config.system.strategy
    if _is_distributed_run(config) and strategy in {
        "auto",
        "ddp",
        "ddp_find_unused_parameters_false",
    }:
        return "ddp_find_unused_parameters_true"
    return strategy


def load_training_config(path: Path) -> TrainingConfig:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")
    return TrainingConfig.model_validate(payload)


def make_training_logger(config: TrainingConfig, out_dir: Path):
    if config.logger.name == "csv":
        return CSVLogger(save_dir=out_dir, name="logs")
    return WandbLogger(
        name=config.logger.run_name,
        save_dir=out_dir,
        offline=config.logger.offline,
        id=config.run_id,
        project=config.logger.project,
        entity=config.logger.entity or None,
        log_model=config.logger.log_model,
        group=config.logger.group or None,
        tags=config.logger.tags,
        job_type=config.logger.job_type or None,
        resume="allow",
    )


def initialize_resume_logger_before_distributed_setup(config: TrainingConfig, training_logger) -> bool:
    """Initialize a resumed W&B run before Lightning enters DDP setup.

    Lightning lazily creates logger experiments inside its setup hook, between
    distributed collectives.  If a resumed W&B client blocks there, rank zero
    never enters the matching barrier while all other ranks wait in NCCL.
    External launchers already expose the global rank, so initialize W&B on
    rank zero before ``Trainer.fit`` creates the process group instead.

    Returns whether this process initialized the logger, which also keeps the
    rank policy directly testable without starting a distributed job.
    """

    if config.logger.name != "wandb" or config.resume_from_checkpoint is None:
        return False

    external_rank = next(
        (
            int(value)
            for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK")
            if (value := os.environ.get(name)) is not None
        ),
        None,
    )
    if external_rank not in {None, 0}:
        return False
    if external_rank is None and _is_distributed_run(config):
        logger.warning(
            "Cannot identify the external global rank before distributed setup; "
            "leaving resumed W&B initialization to Lightning."
        )
        return False

    logger.info("Initializing resumed W&B run before distributed setup on global rank zero.")
    _ = training_logger.experiment
    logger.info("Resumed W&B run initialized before distributed setup.")
    return True


def make_checkpoint_callback(config: TrainingConfig, out_dir: Path) -> ModelCheckpoint:
    if config.system.save_every_n_train_steps is not None:
        return ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="epoch={epoch:02d}-step={step}",
            save_last=True,
            save_top_k=-1,
            every_n_train_steps=config.system.save_every_n_train_steps,
            # The signal callback also publishes last.ckpt. Keep a single
            # rolling resume target instead of creating last-v1.ckpt.
            enable_version_counter=False,
        )
    return ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        filename="epoch={epoch:02d}-psnr={val/psnr:.2f}",
        save_last=True,
        save_top_k=config.system.save_top_k,
        monitor=config.system.checkpoint_monitor,
        mode=config.system.checkpoint_mode,
        every_n_epochs=1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )


def make_training_callbacks(
    config: TrainingConfig,
    out_dir: Path,
    datamodule: TrainingDataModule,
) -> list[Callback]:
    """Build callbacks in checkpoint-safe order.

    The data cursor must advance before either the signal callback or regular
    ModelCheckpoint observes the DataModule state.
    """

    callbacks: list[Callback] = [ResumableDataModuleCallback(datamodule)]
    if config.system.save_on_preemption:
        callbacks.append(AtomicCheckpointAndExitOnSignalCallback(out_dir / "checkpoints" / "last.ckpt"))
    callbacks.append(make_checkpoint_callback(config, out_dir))
    return callbacks


def run_training(config: TrainingConfig) -> Path:
    seed_everything(config.seed, workers=True)
    effective_strategy = resolve_training_strategy(config)
    if effective_strategy != config.system.strategy:
        logger.info(
            "Using %s instead of %s because the model has conditionally unused parameters in distributed training.",
            effective_strategy,
            config.system.strategy,
        )
        config.system.strategy = effective_strategy
    out_dir = Path(config.out_dir) / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    datamodule = TrainingDataModule(config)
    training_logger = make_training_logger(config, out_dir)
    initialize_resume_logger_before_distributed_setup(config, training_logger)
    trainer = Trainer(
        default_root_dir=out_dir,
        accelerator=config.system.accelerator,
        devices=config.system.devices,
        num_nodes=config.system.num_nodes,
        strategy=config.system.strategy,
        precision=config.system.precision,
        max_epochs=config.system.max_epochs,
        deterministic=config.system.deterministic,
        logger=training_logger,
        callbacks=make_training_callbacks(config, out_dir, datamodule),
        # A resumed W&B run already has its config. Avoid a second rank-zero
        # config update between distributed collectives after setup.
        enable_autolog_hparams=config.resume_from_checkpoint is None,
        log_every_n_steps=config.system.log_every_n_steps,
        limit_train_batches=config.system.limit_train_batches,
        limit_val_batches=config.system.limit_val_batches,
        use_distributed_sampler=True,
        # Required for the official epoch/item/global-seed dataset RNG.
        reload_dataloaders_every_n_epochs=1,
    )
    system = TrainingSystem(config)
    trainer.fit(
        system,
        datamodule=datamodule,
        ckpt_path=str(config.resume_from_checkpoint) if config.resume_from_checkpoint else None,
    )
    return out_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Instant NuRec models from NCore V4 data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    config = load_training_config(args.config)
    try:
        output = run_training(config)
    except PreemptionInterrupt as error:
        logger.warning("Training preempted cleanly: %s", error)
        return 1
    logger.info("Training completed: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
