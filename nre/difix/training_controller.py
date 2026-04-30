# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ncore.data import (
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
)
from nre.config.difix import TrainingDifixConfig
from nre.utils.batch import DataAndRenderingBatch, DataBatch, batch_collate_fn
from nre.utils.misc import compute_process_local_rng_seed, unpack_optional
from nre.utils.schedulers import MultiStepScheduler
from nre.utils.types import GaussiansCompositeReturn, NovelViewOverrides, RayFlags
from nre.utils.visualize import save_im


if TYPE_CHECKING:
    from nre.difix.model import DifixModel


log = logging.getLogger(__name__)


class TrainingDifixController:
    """Training-time Difix controller for enhancing novel view quality during training.

    Responsibilities:
    - Reads `difix.training` config and maintains an RNG-driven switch controlling when training-time Difix is active
    - Builds a sequential dataloader that yields novel-view batches aligned with the current training frame
    - Runs the Difix model on the current step's dirty render and replaces RGB labels in-place
    - Sets appropriate `RayFlags` to mark that labels came from Difix
    """

    # Intermediate function to allow pycena to track correctly the dependency between the datamodule and the collate_fn passed to the dataloader
    @staticmethod
    def __batch_collate_fn(*args, **kwargs) -> DataBatch | DataAndRenderingBatch:
        """Intermediate function to allow pycena to track correctly the dependency between the datamodule and the collate_fn passed to the dataloader"""
        return batch_collate_fn(*args, **kwargs)

    def __init__(self, system) -> None:
        """Create the controller.

        Args:
            system: Owning `GaussiansSystem` instance. Used for config access, device, logging and step indices.

        Reads config keys from the strongly-typed `difix.training` config:
        - `p_scheduler`: REQUIRED probability scheduler to use a novel view
        - `start_step`: step at which training-time Difix starts
        - `use_color_transfer`: whether to run the color transfer post-pass in Difix
        - `novel_view_poses`: list of translation/rotation overrides applied progressively within the window
        - `shuffle_novel_views`: whether to randomly shuffle novel view samples (default: False for sequential)
        - `log_images`, `log_every_n_steps`, `log_buffer_images`, `debug_dir`: logging controls

        Note: `full_novel_view_by_step` is automatically set to `max_train_steps` (total training steps).
        """
        self._system = system
        cfg: TrainingDifixConfig = system.difix_config.training
        assert isinstance(cfg, TrainingDifixConfig), f"Expected TrainingDifixConfig, got {type(cfg)}"
        self._enabled: bool = cfg.enabled

        # Initialize RNG first
        self._init_rng()

        # Read config values (they have sensible defaults from the schema)
        self._start_step = cfg.start_step
        self._full_step = self._system.max_train_steps
        self._use_color_transfer = cfg.use_color_transfer
        self._num_workers = cfg.num_workers
        self._novel_view_poses = cfg.novel_view_poses
        self._shuffle_novel_views = cfg.shuffle_novel_views
        self._log_images_enabled = cfg.log_images
        self._log_every_n_steps = cfg.log_every_n_steps
        self._save_debug_images = cfg.log_buffer_images
        self._debug_dir = cfg.debug_dir

        # Scheduler is only initialized when enabled
        self._p_scheduler = (
            MultiStepScheduler(cfg.p_scheduler.to_dictconfig(), self._system.trainer_config, cfg.p_scheduler.p_init)
            if self._enabled
            else None
        )

        self._novel_loader = None
        self._novel_iter = None
        self._difix_model: Optional["DifixModel"] = None

    def _init_rng(self) -> None:
        """Initialize the controller RNG deterministically using the global process seed."""
        try:
            seed = compute_process_local_rng_seed()
        except Exception as e:
            log.debug(f"TrainingDifixController: compute_process_local_rng_seed failed; fallback to 0: {e}")
            seed = 0
        self._rng = np.random.default_rng(seed=seed)

    def maybe_init_difix_model(self) -> None:
        """Initialize the Difix model when training-time Difix is enabled.

        Lazily imports and obtains a process-wide instance from `DifixModelFactory` using
        the model URL, local cache path, checkpoint filename and resolution specified in config.
        Does nothing if training-time Difix is disabled or a model is already initialized.
        """
        if not self._enabled or self._difix_model is not None:
            return
        from nre.difix.model import DifixModelFactory

        self._difix_model = DifixModelFactory.get(
            self._system.difix_config.model_url,
            self._system.difix_config.cache_dir,
            self._system.difix_config.model_filename,
            self._system.difix_config.model_resolution,
        )

    def should_use_novel_view(self) -> bool:
        """Return True when this training step should use a novel-view batch.

        Behavior:
        - Returns False before `start_step` and after `full_novel_view_by_step` (which equals max_train_steps).
        - Otherwise draws a Bernoulli random variable with probability from the scheduler per step.
        """
        if not self._enabled:
            return False
        if self._system.global_step < self._start_step:
            return False
        if self._system.global_step > self._full_step:
            return False

        # Get probability from scheduler (required)
        assert self._p_scheduler is not None, "p_scheduler must be initialized when enabled"
        try:
            cur_p = float(self._p_scheduler(epoch=self._system.current_epoch, global_step=self._system.global_step))
        except Exception as e:
            log.error(f"TrainingDifixController: p_scheduler evaluation failed: {e}")
            raise

        cur_p = max(0.0, min(1.0, cur_p))
        return self._rng.random() < cur_p

    def get_next_novel_batch(
        self, current_batch: Optional[DataAndRenderingBatch] = None
    ) -> Optional[DataAndRenderingBatch]:
        """Return a batch sampled at the same timestep but with current novel-view overrides applied.

        If `current_batch` is provided and contains an index, the sampler is nudged to yield the
        matching dataset frame once. The returned batch is moved to the system device.
        """
        loader = self._get_novel_view_dataloader()
        sampler = loader.sampler
        if hasattr(sampler, "novel_view_overrides"):
            setattr(sampler, "novel_view_overrides", self._compute_step_overrides())
        if current_batch is not None and current_batch.data.idx is not None:
            try:
                # Current training batch index can exceed dataset length. Map it into the
                # sequential dataloader domain.
                cur_idx = int(unpack_optional(current_batch.data.idx))
                if hasattr(sampler, "indices"):
                    indices = list(getattr(sampler, "indices"))
                    if len(indices) > 0:
                        slot = cur_idx % len(indices)
                        forced_dataset_idx = int(indices[slot])
                    else:
                        forced_dataset_idx = cur_idx % len(loader.dataset)
                else:
                    forced_dataset_idx = cur_idx % len(loader.dataset)

                if hasattr(sampler, "forced_index"):
                    setattr(sampler, "forced_index", forced_dataset_idx)
                    self._novel_iter = iter(unpack_optional(self._novel_loader))
            except Exception as e:
                log.debug(f"TrainingDifixController: failed to compute/assign forced index: {e}")

        try:
            assert self._novel_iter is not None
            b = next(self._novel_iter)
        except StopIteration:
            assert self._novel_loader is not None
            self._novel_iter = iter(self._novel_loader)
            b = next(self._novel_iter)

        b = b.to(self._system.device)
        return b

    def apply_difix_as_target(self, batch_for_render: DataAndRenderingBatch, out: GaussiansCompositeReturn) -> None:
        """Compute Difix target from the current dirty render and replace RGB labels in-place.

        Effects:
        - Computes `(H, W)` from batch metadata and calls the Difix model with optional color transfer
        - Sets `batch_for_render.data.camera.labels.rgb` to the Difix output and marks `RayFlags.RGB_LABEL|DIFIXED`
        - Clears `RayFlags.INVALID` for affected rays
        - Optionally logs before/after images and saves local debug PNGs when enabled
        """
        assert out.rendered_cam is not None and out.rendered_cam.rgb is not None

        # Infer H,W
        assert batch_for_render.data.camera is not None
        camera_data = batch_for_render.data.camera

        if camera_data.h is not None and camera_data.w is not None:
            h = int(unpack_optional(camera_data.h))
            w = int(unpack_optional(camera_data.w))
        else:
            # Fall back to getting resolution from sensor model parameters in rendering batch
            assert batch_for_render.rendering is not None and batch_for_render.rendering.camera is not None
            params = batch_for_render.rendering.camera.sensor_model_parameters[0]
            # Type narrowing: Difix only works with camera sensors, not lidar
            assert isinstance(
                params,
                (FThetaCameraModelParameters, OpenCVPinholeCameraModelParameters, OpenCVFisheyeCameraModelParameters),
            ), f"Expected camera parameters, got {type(params)}"
            w = int(params.resolution[0].item())
            h = int(params.resolution[1].item())

        dirty = out.rendered_cam.rgb

        # Get unique sensor idx from the first meta element
        unique_sensor_idx = camera_data.meta[0].unique_sensor_idx

        target = self._get_or_compute_difix(dirty, torch.Size([h, w]))
        log_camera_name = f"cam_{unique_sensor_idx:02d}"

        # Reshape target from (H*W, 3) to (B, H, W, 3) to match CameraFrameLabels format
        b = camera_data.b
        target = target.view(b, h, w, 3)

        # Update flags in the labels
        assert camera_data.labels.flags is not None
        flags = camera_data.labels.flags  # Store reference for clarity
        flags |= RayFlags.RGB_LABEL | RayFlags.DIFIXED
        flags &= ~RayFlags.INVALID
        camera_data.labels.rgb = target

        try:
            step = self._system.global_step
            should_log = self._log_images_enabled and (step % self._log_every_n_steps == 0)
            if should_log:
                debug_batch_i = 0  # Right now we support only bs of 1
                dirty_chw = (dirty.view(h, w, 3).permute(2, 0, 1).clamp(0, 1) * 255).byte().cpu()  # CHW
                fixed_chw = (target[debug_batch_i].permute(2, 0, 1).clamp(0, 1) * 255).byte().cpu()
                self._system.log_image(dirty_chw, f"difix_training/before/{log_camera_name}", step)
                self._system.log_image(fixed_chw, f"difix_training/after/{log_camera_name}", step)

                if self._save_debug_images:
                    debug_dir = self._debug_dir or os.path.join(
                        self._system.out_dir, self._system.run_id, "difix_training_debug"
                    )
                    os.makedirs(camera_dir := os.path.join(debug_dir, log_camera_name), exist_ok=True)
                    os.makedirs(before_dir := os.path.join(camera_dir, "before"), exist_ok=True)
                    os.makedirs(after_dir := os.path.join(camera_dir, "after"), exist_ok=True)
                    save_im(dirty.detach(), h, os.path.join(before_dir, f"difix_{step}.png"))
                    save_im(
                        target[debug_batch_i].view(h * w, 3).detach(), h, os.path.join(after_dir, f"difix_{step}.png")
                    )
        except Exception as e:
            log.debug(f"TrainingDifixController: image logging failed: {e}")

    def _compute_step_overrides(self) -> Optional[NovelViewOverrides]:
        """Compute per-step novel-view pose overrides.

        Linearly scales the configured translation/rotation deltas from 0 to their full values
        across the interval `[start_step, full_novel_view_by_step]`. Returns None if no poses are configured.
        """
        poses = self._novel_view_poses
        if not poses:
            return None
        idx = self._system.global_step % len(self._novel_view_poses)
        pose_cfg = poses[idx]

        start_step = self._start_step
        full_step = max(self._full_step, start_step + 1)
        ratio = max(min((self._system.global_step - start_step) / (full_step - start_step), 1.0), 0.0)

        return NovelViewOverrides(
            transl_delta_m=np.array(pose_cfg.translation, dtype=np.float32) * ratio
            if pose_cfg.translation is not None
            else None,
            rot_delta_deg=np.array(pose_cfg.rotation, dtype=np.float32) * ratio
            if pose_cfg.rotation is not None
            else None,
        )

    def _get_novel_view_dataloader(self):
        """Create the persistent dataloader used for novel-view sampling.

        The loader iterates the training dataset with `ParameterizedSequentialSampler`.
        It is constructed once and reused (with an internal iterator).
        Samples can be returned sequentially or randomly based on the `shuffle_novel_views` config.
        """
        if self._novel_loader is None:
            from torch.utils.data import DataLoader

            from nre.datasets.registry import make as make_dataset
            from nre.difix.utils import ParameterizedSequentialSampler

            dataset = make_dataset(self._system.dataset_name, self._system.cached_config, split="train-sequential")
            sampler = ParameterizedSequentialSampler(data_source=dataset, shuffle=self._shuffle_novel_views)
            self._novel_loader = DataLoader(
                dataset,
                num_workers=self._num_workers,
                persistent_workers=True if self._num_workers > 0 else False,
                batch_size=1,
                collate_fn=self.__batch_collate_fn,
                pin_memory=True,
                sampler=sampler,
            )
            self._novel_iter = iter(self._novel_loader)
        return self._novel_loader

    @torch.no_grad()
    def _get_or_compute_difix(self, dirty_rgb: torch.Tensor, img_size: torch.Size) -> torch.Tensor:
        """Run Difix on a flattened RGB image of shape `(H*W, 3)`.

        Args:
            dirty_rgb: flattened RGB in [0,1]
            img_size: `(H, W)` used for reshaping and internal resampling

        Returns:
            Flattened `(H*W, 3)` RGB tensor in [0,1].
        """
        assert self._difix_model is not None, "Difix model not initialized"
        return self._difix_model(dirty_rgb, img_size, self._use_color_transfer)
