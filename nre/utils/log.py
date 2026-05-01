# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import weakref

from typing import Any

import numpy as np
import point_cloud_utils as pcu

from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage

from nre.config.logger import BatchMediaLoggerConfigMixin


class BatchMediaLogger:
    """
    A helper class to log media (images, videos) at the same step.
    TODO [JH]: Support other loggers (e.g. TensorBoard) if needed.

    The original wandb logger does not support logging multiple images at the same step, it also does not support
    *checking* if the current step should log media (so that we can skip generating the images). This class helps
    solve these problems while pack the functions into a class so they can be passed around easily.
    """

    def __init__(self, system: LightningModule, config: BatchMediaLoggerConfigMixin) -> None:
        self._system = weakref.ref(system)
        self._image_log_step_cache: dict[str, np.ndarray] = {}
        self._ply_log_step_cache: dict[str, pcu.TriangleMesh] = {}
        self.log_media_every_n_steps = int(config.log_media_every_n_steps)
        self.log_media_every_n_steps_val = int(config.log_media_every_n_steps_val)
        self.log_media_subsample = int(config.log_media_subsample)

    @property
    def system(self) -> LightningModule:
        result = self._system()
        if result is None:
            raise RuntimeError("System reference has been garbage collected")
        return result

    def log(self, *args: Any, **kwargs: Any) -> None:
        self.system.log(*args, **kwargs)

    def log_image(self, caption: str, data: np.ndarray) -> None:
        if self.should_log_media:
            self._image_log_step_cache[caption] = data

    def log_ply_point_cloud(
        self,
        name: str,
        xyz: np.ndarray,
        color: np.ndarray | None = None,
        other_attributes: dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Cache point cloud as pcu.TriangleMesh for writing as PLY in flush_logged_media.
        Only during validation and when should_log_media; overwrites any previous cache for this name.
        """
        if self.system.trainer.state.stage != RunningStage.VALIDATING:
            return
        if not self.should_log_media:
            return
        mesh = pcu.TriangleMesh()
        mesh.vertex_data.positions = np.asarray(xyz, dtype=np.float64)
        if color is not None:
            mesh.vertex_data.colors = np.asarray(color)
        for key, value in (other_attributes or {}).items():
            arr = np.asarray(value)
            mesh.vertex_data.custom_attributes[key] = arr.astype(np.float64) if arr.dtype.kind == "f" else arr
        self._ply_log_step_cache[name] = mesh

    @property
    def should_log_media(self) -> bool:
        match self.system.trainer.state.stage:
            case RunningStage.TRAINING:
                if self.log_media_every_n_steps == 0:
                    return False
                # For training, log according to batches stepped
                batches_stepped: int = self.system.trainer.fit_loop.epoch_loop.total_batch_idx
                return (batches_stepped + 1) % self.log_media_every_n_steps == 0

            case RunningStage.VALIDATING:
                if self.log_media_every_n_steps_val == 0:
                    return False
                # For validation, log according to the current step within the batch
                # within _evaluation_loop is a nice dispatch function that determines loop based on
                # trainer.fn (i.e. whether this is called from trainer.fit() or trainer.validate())
                local_batches_stepped: int = self.system.trainer._evaluation_loop.batch_progress.current.ready - 1
                return (local_batches_stepped + 1) % self.log_media_every_n_steps_val == 0

            case RunningStage.TESTING:
                return True

            case _:
                # For predicting / sanity-checking, don't log
                return False

