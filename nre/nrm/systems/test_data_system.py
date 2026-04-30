# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import time

import torch

from torchmetrics.aggregation import SumMetric

from libs.losses.orchestration.config import (
    LossAggregatorBatchReturn,
    LossAggregatorReturn,
    LossReturn,
)
from nre.datasets.tracks import CuboidTracks
from nre.nrm.config.models import BaseModelConfig
from nre.nrm.config.nrm import NRMConfig
from nre.nrm.models.base import BaseNRM, BaseNRMSupervisionPack
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.nrm.systems.base import BaseNRMSystem
from nre.utils.batch import DataAndRenderingBatch, NRMDataBatch
from nre.utils.log import BatchMediaLogger


logger = logging.getLogger(__name__)


class DummyTestModel(BaseNRM[BaseNRMPrimitive, BaseNRMSupervisionPack]):
    """Minimal model with one parameter so the training loop and optimizer still run."""

    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__(config)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        media_logger: BatchMediaLogger | None,
        compute_supervision_pack: bool = False,
    ) -> tuple[list[BaseNRMPrimitive], list[BaseNRMSupervisionPack] | None]:
        return ([], None)

    def prepare_supervision(
        self,
        context: list[DataAndRenderingBatch],
        supervision: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
        supervision_packs: list[BaseNRMSupervisionPack],
        media_logger: BatchMediaLogger | None,
    ) -> tuple[list[DataAndRenderingBatch], list[BaseNRMSupervisionPack]]:
        return (supervision, supervision_packs)

    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None,
    ) -> list[DataAndRenderingBatch]:
        return context


class TestDataSystemNRMSystem(BaseNRMSystem):
    """
    Minimal NRM training system for testing data loading. Uses TestIndexNRMDataset
    and prints the batch indices seen each training epoch (from batch.meta["index"]).
    Also verifies that torch metrics are restored correctly after preemption by
    accumulating the sum of meta indices over train/val steps and asserting it
    matches the expected total at epoch end.
    """

    def __init__(self, config: NRMConfig) -> None:
        super().__init__(config)
        self.model = DummyTestModel(config.model)

    def setup(self, stage: str) -> None:
        """Create SumMetrics so we can verify they are checkpointed/restored correctly after preemption."""
        self.val_step_sum = SumMetric()

    def _meta_indices(self, batch: NRMDataBatch) -> list[int]:
        if batch.meta is None:
            return []
        return [int(meta["index"]) for meta in batch.meta if isinstance(meta, dict) and "index" in meta]

    def validation_step(self, batch: NRMDataBatch, batch_local_idx: int) -> None:
        """Minimal validation step so the validation loop runs (logging happens in on_validation_batch_start)."""
        meta_idx = self._meta_indices(batch)
        logger.info(
            f"[rank {self.trainer.global_rank}] val epoch={self.current_epoch} batch_idx={batch_local_idx} meta_index={meta_idx}",
        )

        # Simulate some processing time
        time.sleep(1)

        # Update metric with sum of meta indices so we can verify aggregated sum at epoch end (including after preemption)
        # We need to log the metric object itself so that no need to reset metric at epoch end.
        self.val_step_sum(float(sum(meta_idx)) if meta_idx else 0.0)
        self.log("val/step_num", self.val_step_sum, batch_size=1)

        self.log("val/psnr", 0.0, batch_size=1)

    def training_losses(self, batch: NRMDataBatch, batch_local_idx: int) -> LossAggregatorBatchReturn:
        meta_idx = self._meta_indices(batch)
        logger.info(
            f"[rank {self.trainer.global_rank}] train epoch={self.current_epoch} batch_idx={batch_local_idx} meta_index={meta_idx}",
        )

        # Simulate some processing time
        time.sleep(1)

        # Return a dummy loss so the training step can backward and optimizer can step
        assert isinstance(self.model, DummyTestModel)
        dummy_value = 0.0 * self.model.dummy
        loss_return = LossReturn(
            name="test_dummy",
            lambda_=1.0,
            value=dummy_value,
            reduce_fn=lambda value, **kwargs: value.mean() if value.numel() > 0 else value,
        )
        agg_return = LossAggregatorReturn(loss_returns={"test_dummy": loss_return})
        return LossAggregatorBatchReturn(batch_loss_returns=[agg_return])

    def on_validation_epoch_end(self) -> None:
        """Verify that the validation-step sum of meta indices matches expected (checks metric recovery after preemption)."""
        super().on_validation_epoch_end()
        val_loader = self.datamodule.val_dataloader()
        n_batches = len(val_loader) if val_loader is not None else 0
        expected_sum = n_batches * (n_batches - 1) // 2 if n_batches else 0
        actual_sum = self.val_step_sum.compute()
        actual_sum_val = float(actual_sum.item() if torch.is_tensor(actual_sum) else actual_sum)
        logger.info(
            f"[rank {self.trainer.global_rank}] val epoch={self.current_epoch} metric sum: {expected_sum} = {actual_sum_val}.",
        )
