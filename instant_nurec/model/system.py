# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import logging
import os

from pathlib import Path

import torch

from torch import nn
from tqdm import tqdm

from instant_nurec.datasets.tracks import CuboidTracks
from instant_nurec.config_schema.instantnurec import GaussiansInstantNuRecSystemConfig
from instant_nurec.datasets.datamodule import InstantNuRecDataModule
from instant_nurec.model.inference import KelvinInferenceModel
from instant_nurec.predict.export_ply import export_ply
from instant_nurec.predict.primitive_merge import KelvinPrimitiveMerge
from instant_nurec.primitives.base import BaseInstantNuRecPrimitive
from instant_nurec.utils.batch import InstantNuRecDataBatch
from instant_nurec.utils.types import RigTrajectories


logger = logging.getLogger(__name__)


class GaussiansInstantNuRecSystem(nn.Module):
    """Predict-only system; the predict driver invokes hooks directly."""

    config: GaussiansInstantNuRecSystemConfig
    model: KelvinInferenceModel
    datamodule: InstantNuRecDataModule

    def __init__(self, config, model: KelvinInferenceModel) -> None:
        super().__init__()
        self.out_dir = config.out_dir
        self.run_id = config.run_id
        self.config = config.system
        self.predict_config = config.predict
        self.export_preprocess = config.model.export_preprocess
        self.datamodule = InstantNuRecDataModule(config)
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: InstantNuRecDataBatch) -> list[BaseInstantNuRecPrimitive]:
        cuboid_tracks = None
        if batch.cuboid_tracks is not None:
            cuboid_tracks = [CuboidTracks.Factory.from_pack(ct) for ct in batch.cuboid_tracks]

        batch.context = self.model.prepare_context(batch.context)
        return self.model.reconstruct(batch.context, cuboid_tracks)

    def predict_step(self, batch: InstantNuRecDataBatch) -> dict[str, list[BaseInstantNuRecPrimitive] | InstantNuRecDataBatch]:
        # In the future maybe rendering data is not required any more for model forwarding.
        batch.maybe_compute_rendering_data(device=self.device)

        _CHUNK_SIZE = 1  # not a config knob; the model expects B=1.
        primitives_list: list[BaseInstantNuRecPrimitive] = []

        inner_batch_idx: int = 0
        progress_bar = tqdm(total=len(batch), desc="Predicting in chunks")
        while inner_batch_idx < len(batch):
            batch_chunk = batch[inner_batch_idx : inner_batch_idx + _CHUNK_SIZE]
            primitives_chunk_list = self.forward(batch_chunk)
            context_rig_list = batch_chunk.context_rig if batch_chunk.context_rig is not None else None
            for i in range(len(primitives_chunk_list)):
                context_rig_i = context_rig_list[i] if context_rig_list is not None else None
                primitives_chunk_list[i] = primitives_chunk_list[i].preprocess_for_export(
                    batch_chunk.context[i], self.export_preprocess, context_rig=context_rig_i
                )
            primitives_list.extend(primitives_chunk_list)
            inner_batch_idx += _CHUNK_SIZE
            progress_bar.update(_CHUNK_SIZE)
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

        out_batch: InstantNuRecDataBatch = outputs["batch"]
        primitives_list: list[BaseInstantNuRecPrimitive] = outputs["primitives"]
        n_chunks = len(primitives_list)
        assert len(out_batch) == n_chunks, "batch context length must match number of primitives"

        if out_batch.meta is None or out_batch.context_rig is None:
            return

        def export_chunk(primitive: BaseInstantNuRecPrimitive, rig: RigTrajectories, meta: dict, chunk_suffix: str) -> None:
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
            n = primitive.static_layer.densities.numel()
            print(f"Wrote 3DGS PLY ({n:,} gaussians): {Path(path).resolve()}", flush=True)

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
