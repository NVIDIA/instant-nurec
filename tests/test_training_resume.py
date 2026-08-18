# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning_fabric.utilities.data import _replace_dunder_methods, _update_dataloader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate

import instant_nurec.training.data as training_data_module
from instant_nurec.config_schema.train import KelvinSchedulerConfig
from instant_nurec.training.callbacks import AtomicCheckpointAndExitOnSignalCallback, PreemptionInterrupt
from instant_nurec.training.data import KelvinTrainingDataModule, ResumableDataModuleCallback, SkipBatchSampler
from instant_nurec.training.optim import CosineWithWarmupPBScheduler
from instant_nurec.training.run import make_training_callbacks


class _ToyDataset(Dataset):
    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.rng_epoch = -1
        self.epoch = -1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(float(index))

    def set_rng_epoch(self, epoch: int) -> None:
        self.rng_epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _toy_config() -> SimpleNamespace:
    return SimpleNamespace(
        seed=38,
        dataset=SimpleNamespace(train=object(), val=None),
        system=SimpleNamespace(
            train_batch_size=1,
            train_num_workers=0,
            val_batch_size=1,
            val_num_workers=0,
        ),
    )


class _ToyKelvinDataModule(KelvinTrainingDataModule):
    def _make_dataset(self, dataset_config):
        del dataset_config
        return _ToyDataset()


class _ResumeProbe(LightningModule):
    automatic_optimization = False

    def __init__(self, records: list[dict[str, float | int]]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.records = records

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([self.weight], lr=1.0)
        scheduler = CosineWithWarmupPBScheduler(
            optimizer,
            KelvinSchedulerConfig(warmup_factor=0.1, warmup_steps=2, cosine_factor=0.2),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        global_step_before = int(self.global_step)
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        loss = (self.weight - batch.float().mean()).square()
        self.manual_backward(loss)
        optimizer.step()

        scheduler = self.lr_schedulers()
        assert isinstance(scheduler, CosineWithWarmupPBScheduler)
        scheduler.set_progress(
            epoch=self.current_epoch,
            total_epochs=1,
            local_step=batch_idx,
            total_local_steps=int(self.trainer.num_training_batches),
        )
        scheduler.step()
        self.records.append(
            {
                "sample": int(batch.item()),
                "batch_idx": batch_idx,
                "global_step": global_step_before,
                "lr": optimizer.optimizer.param_groups[0]["lr"],
            }
        )
        return loss.detach()


class _TriggerPreemption(Callback):
    def __init__(self, target: AtomicCheckpointAndExitOnSignalCallback, batch_idx: int) -> None:
        self.target = target
        self.batch_idx = batch_idx

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        del trainer, pl_module, outputs, batch
        if batch_idx == self.batch_idx:
            self.target.preempting = True


def _trainer(callbacks: list[Callback]) -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        deterministic=True,
    )


def test_skip_batch_sampler_preserves_full_epoch_length() -> None:
    sampler = SkipBatchSampler(SequentialSampler(range(10)), batch_size=2, first_batch_idx=2)

    assert len(sampler) == 5
    assert list(sampler) == [[4, 5], [6, 7], [8, 9]]


def test_lightning_injects_distributed_sampler_without_losing_resume_cursor() -> None:
    dataset = _ToyDataset(size=10)
    generator = torch.Generator().manual_seed(38)
    with _replace_dunder_methods(DataLoader, "dataset"), _replace_dunder_methods(BatchSampler):
        dataloader = DataLoader(
            dataset,
            batch_sampler=SkipBatchSampler(
                RandomSampler(dataset, generator=generator),
                batch_size=2,
                first_batch_idx=2,
            ),
        )

    distributed_sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=38)
    injected = _update_dataloader(dataloader, distributed_sampler)

    assert isinstance(injected.batch_sampler, SkipBatchSampler)
    assert injected.batch_sampler.sampler is distributed_sampler
    assert injected.batch_sampler.first_batch_idx == 2
    assert len(injected.batch_sampler) == 3


def test_resumable_callback_precedes_all_checkpoint_callbacks(tmp_path) -> None:
    config = SimpleNamespace(
        system=SimpleNamespace(
            save_on_preemption=True,
            save_every_n_train_steps=1,
        )
    )
    datamodule = _ToyKelvinDataModule(_toy_config())

    callbacks = make_training_callbacks(config, tmp_path, datamodule)

    assert isinstance(callbacks[0], ResumableDataModuleCallback)
    assert isinstance(callbacks[1], AtomicCheckpointAndExitOnSignalCallback)
    assert isinstance(callbacks[2], ModelCheckpoint)
    callbacks[1].teardown(None, None, "fit")


def test_mid_epoch_checkpoint_resume_has_no_replay_and_preserves_schedule(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        training_data_module,
        "InstantNuRecDataBatch",
        SimpleNamespace(collate_fn=default_collate),
    )
    checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"

    interrupted_records: list[dict[str, float | int]] = []
    interrupted_datamodule = _ToyKelvinDataModule(_toy_config())
    resumable = ResumableDataModuleCallback(interrupted_datamodule)
    preemption = AtomicCheckpointAndExitOnSignalCallback(checkpoint_path, signal_type=None)
    trigger = _TriggerPreemption(preemption, batch_idx=2)
    with pytest.raises(PreemptionInterrupt, match="checkpoint saved"):
        _trainer([resumable, trigger, preemption]).fit(
            _ResumeProbe(interrupted_records),
            datamodule=interrupted_datamodule,
        )

    assert checkpoint_path.is_file()
    assert not Path(f"{checkpoint_path}.tmp").exists()
    assert interrupted_datamodule.state_dict() == {"next_train_batch_idx": 3}
    assert [record["batch_idx"] for record in interrupted_records] == [0, 1, 2]
    assert [record["global_step"] for record in interrupted_records] == [0, 1, 2]

    resumed_records: list[dict[str, float | int]] = []
    resumed_datamodule = _ToyKelvinDataModule(_toy_config())
    _trainer([ResumableDataModuleCallback(resumed_datamodule)]).fit(
        _ResumeProbe(resumed_records),
        datamodule=resumed_datamodule,
        ckpt_path=checkpoint_path,
    )

    uninterrupted_records: list[dict[str, float | int]] = []
    uninterrupted_datamodule = _ToyKelvinDataModule(_toy_config())
    _trainer([ResumableDataModuleCallback(uninterrupted_datamodule)]).fit(
        _ResumeProbe(uninterrupted_records),
        datamodule=uninterrupted_datamodule,
    )

    combined = interrupted_records + resumed_records
    assert [record["sample"] for record in combined] == [record["sample"] for record in uninterrupted_records]
    assert len({record["sample"] for record in combined}) == len(combined) == 8
    assert [record["batch_idx"] for record in resumed_records] == [3, 4, 5, 6, 7]
    assert [record["global_step"] for record in resumed_records] == [3, 4, 5, 6, 7]
    assert [record["lr"] for record in combined] == pytest.approx([record["lr"] for record in uninterrupted_records])
