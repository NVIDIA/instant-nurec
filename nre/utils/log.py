# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import io
import logging
import os
import weakref

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import wandb


import point_cloud_utils as pcu

from PIL import Image as _PILImage
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.states import RunningStage

from nre.config.logger import BatchMediaLoggerConfigMixin


_JPEG_QUALITY = 93


def _ndarray_to_pil(arr: np.ndarray) -> _PILImage.Image:
    arr = arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return _PILImage.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return _PILImage.fromarray(arr.squeeze(-1), mode="L")
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return _PILImage.fromarray(arr, mode="RGBA").convert("RGB")
    return _PILImage.fromarray(arr)


class BatchMediaLogger:
    """
    A helper class to log media (images, videos) at the same step.
    TODO [JH]: Support other loggers (e.g. TensorBoard) if needed.

    The original wandb logger does not support logging multiple images at the same step, it also does not support
    *checking* if the current step should log media (so that we can skip generating the images). This class helps
    solve these problems while pack the functions into a class so they can be passed around easily.
    """

    @dataclass
    class EncodedVideo:
        data: io.BytesIO
        format: str

        def get_bytes_io(self) -> io.BytesIO:
            self.data.seek(0)
            return self.data

    def __init__(self, system: LightningModule, config: BatchMediaLoggerConfigMixin) -> None:
        self._system = weakref.ref(system)
        self._image_log_step_cache: dict[str, np.ndarray] = {}
        self._video_log_step_cache: dict[str, BatchMediaLogger.EncodedVideo] = {}
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

    def log_video(self, caption: str, data: np.ndarray, fps: int, format: str = "mp4") -> None:
        if self.should_log_media:
            video_file = io.BytesIO()
            imageio.v2.mimwrite(video_file, data, fps=fps, macro_block_size=1, format=format)  # type: ignore
            self._video_log_step_cache[caption] = self.EncodedVideo(data=video_file, format=format)

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

    def flush_logged_media(self, prefix: str, media_step: int | None = None) -> None:
        match self.system.logger:
            case WandbLogger():
                # system.logger.experiment is DummyExperiment on non-rank-zero.
                experiment = self.system.logger.experiment
                logged: bool = False

                if len(self._image_log_step_cache) > 0:
                    # Disable logging.warning about 'Images sizes do not match'
                    (root_logger := logging.getLogger()).setLevel(logging.ERROR)
                    # Pre-encode each image to JPEG at a chosen quality in-memory so we bypass
                    # wandb.Image's default PIL quality=75 when given a numpy array.
                    wandb_images = []
                    for caption, image in self._image_log_step_cache.items():
                        buf = io.BytesIO()
                        _ndarray_to_pil(image).save(buf, format="JPEG", quality=_JPEG_QUALITY)
                        buf.seek(0)
                        wandb_images.append(wandb.Image(_PILImage.open(buf), caption=caption))
                    experiment.log(
                        {f"{prefix}/images": wandb_images},
                        commit=media_step is None,
                    )
                    root_logger.setLevel(logging.INFO)
                    logged = True

                if len(self._video_log_step_cache) > 0:
                    experiment.log(
                        {
                            f"{prefix}/videos/{caption}": wandb.Video(data.get_bytes_io(), format=data.format)
                            for caption, data in self._video_log_step_cache.items()
                        },
                        commit=media_step is None,
                    )
                    logged = True

                # Log the media_step so it could be used as the x-axis in wandb UI.
                if media_step is not None and logged:
                    experiment.log({f"{prefix}/media_step": media_step}, commit=True)

            case DummyLogger():
                pass
            case _:
                raise ValueError(f"Unsupported logger type: {self.system.logger.__class__.__name__}")

        self._image_log_step_cache = {}
        self._video_log_step_cache = {}
        self._ply_log_step_cache = {}

    def save_logged_videos(self, save_dir: str, prefix: str) -> None:
        """
        Save logged videos to the specified directory.
        Note that this will take effect on all ranks.
        """
        if not self._video_log_step_cache:
            return

        os.makedirs(save_dir, exist_ok=True)

        for caption, video in self._video_log_step_cache.items():
            video_file = os.path.join(save_dir, f"{prefix}-{caption}.mp4")
            with open(video_file, "wb") as f:
                f.write(video.get_bytes_io().getbuffer())

    def save_logged_ply_point_clouds(self, save_dir: str, prefix: str) -> None:
        """Write cached PLY point clouds to save_dir (one .ply file per cache name).

        The cache itself is not cleared here; it is cleared later by ``flush_logged_media``.
        """
        if not self._ply_log_step_cache:
            return

        path_dir = Path(save_dir)
        path_dir.mkdir(parents=True, exist_ok=True)

        for name, mesh in self._ply_log_step_cache.items():
            path = path_dir / f"{prefix}-{name}.ply"
            mesh.save(str(path))
