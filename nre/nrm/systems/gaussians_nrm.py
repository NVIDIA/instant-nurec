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

import gc
import logging
import os

from pathlib import Path

import torch

from torch import nn
from tqdm import tqdm

from nre.datasets.tracks import CuboidTracks
from nre.nrm.config.nrm import GaussiansNRMSystemConfig, NRMConfig
from nre.nrm.datasets.datamodule import NRMDataModule
from nre.nrm.models.kelvin_model import KelvinNRM
from nre.nrm.predict.export_ply import export_ply
from nre.nrm.predict.primitive_merge import KelvinPrimitiveMerge
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.utils.batch import NRMDataBatch
from nre.utils.types import RigTrajectories


logger = logging.getLogger(__name__)


class GaussiansNRMSystem(nn.Module):
    """Predict-only system. Self-invented: NRE inherits LightningModule for the
    Trainer.fit/validate/test surfaces; we keep just nn.Module since the
    predict driver invokes hooks directly."""

    config: GaussiansNRMSystemConfig
    model: KelvinNRM
    datamodule: NRMDataModule

    def __init__(self, config: NRMConfig) -> None:
        super().__init__()

        self.out_dir = config.out_dir
        self.run_id = config.run_id
        self.config = config.system
        self.predict_config = config.predict
        self.export_preprocess = config.model.export_preprocess

        self.datamodule = NRMDataModule(config)
        self.model = KelvinNRM(config.model)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: NRMDataBatch) -> list[BaseNRMPrimitive]:
        cuboid_tracks = None
        if batch.cuboid_tracks is not None:
            cuboid_tracks = [CuboidTracks.Factory.from_pack(ct) for ct in batch.cuboid_tracks]

        batch.context = self.model.prepare_context(batch.context, cuboid_tracks)
        return self.model.reconstruct(batch.context, cuboid_tracks)

    def predict_step(self, batch: NRMDataBatch) -> dict[str, list[BaseNRMPrimitive] | NRMDataBatch]:
        # In the future maybe rendering data is not required any more for model forwarding.
        batch.maybe_compute_rendering_data(device=self.device)

        # For large batch sizes, process in chunks
        primitives_list: list[BaseNRMPrimitive] = []

        inner_batch_idx: int = 0
        progress_bar = tqdm(total=len(batch), desc="Predicting in chunks")
        while inner_batch_idx < len(batch):
            batch_chunk = batch[inner_batch_idx : inner_batch_idx + self.predict_config.chunk_size]
            primitives_chunk_list = self.forward(batch_chunk)
            context_rig_list = batch_chunk.context_rig if batch_chunk.context_rig is not None else None
            for i in range(len(primitives_chunk_list)):
                context_rig_i = context_rig_list[i] if context_rig_list is not None else None
                primitives_chunk_list[i] = primitives_chunk_list[i].preprocess_for_export(
                    batch_chunk.context[i], self.export_preprocess, context_rig=context_rig_i
                )
            primitives_list.extend(primitives_chunk_list)
            inner_batch_idx += self.predict_config.chunk_size
            progress_bar.update(self.predict_config.chunk_size)
        progress_bar.close()

        # Merge the primitives if enabled
        if self.predict_config.primitive_merge.enabled:
            primitive_merge = KelvinPrimitiveMerge(self.predict_config.primitive_merge)
            merged_primitive, batch = primitive_merge.merge_primitives_and_batch(primitives_list, batch)
            primitives_list = [merged_primitive]

        # Release memory if possible
        gc.collect()
        torch.cuda.empty_cache()

        return {"primitives": primitives_list, "batch": batch}

    def on_predict_batch_end(self, outputs, batch) -> None:
        # Ensure outputs are not None and contain the required keys
        assert outputs is not None and "primitives" in outputs and "batch" in outputs

        out_batch: NRMDataBatch = outputs["batch"]
        primitives_list: list[BaseNRMPrimitive] = outputs["primitives"]
        n_chunks = len(primitives_list)
        assert len(out_batch) == n_chunks, "batch context length must match number of primitives"

        if out_batch.meta is None or out_batch.context_rig is None:
            return

        # Helper to export PLY for one chunk. Standalone build does not export
        # USDZ artifacts and does not render videos (those code paths were
        # removed in Phase 1 step 4.3).
        def export_chunk(primitive: BaseNRMPrimitive, rig: RigTrajectories, meta: dict, chunk_suffix: str) -> None:
            path = os.path.join(
                self.out_dir,
                self.run_id,
                "ply",
                meta["sequence_id"],
                meta["sequence_id"] + chunk_suffix + ".ply",
            )
            export_ply(
                primitives=primitive,
                rig_trajectories=rig,
                path=Path(path),
            )

        for chunk_idx in range(n_chunks):
            meta = out_batch.meta[chunk_idx]
            assert "sequence_id" in meta, f"sequence_id key must be provided, only got {meta.keys()}"
            chunk_suffix = "" if self.predict_config.primitive_merge.enabled else f"_chunk{chunk_idx}"
            export_chunk(
                primitives_list[chunk_idx],
                out_batch.context_rig[chunk_idx],
                meta,
                chunk_suffix,
            )

