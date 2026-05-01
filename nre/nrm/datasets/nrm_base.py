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

import hashlib
import logging
import os

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Sized, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import ncore.data

from nre.nrm.config.dataset import (
    CameraSubsamplerConfig,
)
from nre.utils.batch import NRMDataBatch, RectSubsampled


logger = logging.getLogger(__name__)


class CameraSubsampler:
    """
    Dedicated class to subsample camera parameters or the images (currently center crop is used).
    This could be later extended to include more complex subsampling strategies (e.g. for progressive training).
    """

    def __init__(self, config: CameraSubsamplerConfig):
        self.frame_width = config.frame_width
        self.frame_height = config.frame_height

    def _compute_pixel_rect(self, original_width: int, original_height: int) -> Tuple[RectSubsampled, Tuple[int, int]]:
        scale_factor = max(self.frame_width / original_width, self.frame_height / original_height)
        scaled_w = round(original_width * scale_factor)
        scaled_h = round(original_height * scale_factor)
        offset_w = (scaled_w - self.frame_width) // 2
        offset_h = (scaled_h - self.frame_height) // 2
        return RectSubsampled(
            subsample_factor=1.0 / scale_factor,
            i=offset_w,
            j=offset_h,
            width=self.frame_width,
            height=self.frame_height,
            original_width=original_width,
            original_height=original_height,
        ), (scaled_w, scaled_h)

    def apply_camera_parameters(
        self, camera_parameters: ncore.data.ConcreteCameraModelParametersUnion
    ) -> ncore.data.ConcreteCameraModelParametersUnion:
        original_width = camera_parameters.resolution[0].item()
        original_height = camera_parameters.resolution[1].item()
        pixel_rect, _ = self._compute_pixel_rect(original_width, original_height)
        return camera_parameters.transform(
            image_domain_scale=1.0 / pixel_rect.subsample_factor,
            image_domain_offset=(pixel_rect.i, pixel_rect.j),
            new_resolution=(pixel_rect.width, pixel_rect.height),
        )

    def apply_depth_data(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Apply reshape where depth_data is (H, W). Uses nearest-foreground downsampling
        via adaptive max-pool on negated depths to avoid sky bleed-in.
        """
        original_height, original_width = depth_data.shape[:2]
        pixel_rect, (scaled_w, scaled_h) = self._compute_pixel_rect(original_width, original_height)

        depth_data_pth = torch.from_numpy(depth_data).float().clone()
        depth_data_pth[depth_data_pth == 0] = float("inf")
        depth_data_pth = -F.adaptive_max_pool2d(-depth_data_pth[None, None], (scaled_h, scaled_w))
        depth_data_pth[depth_data_pth == float("inf")] = 0.0

        depth_data = depth_data_pth[0, 0].numpy()
        return depth_data[
            pixel_rect.j : pixel_rect.j + pixel_rect.height, pixel_rect.i : pixel_rect.i + pixel_rect.width
        ]

    def apply_frame_data(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Apply reshape where frame_data is (H, W, C) or (H, W)
        If frame_data is float then we do bilinear interpolation, otherwise we do nearest neighbor.
        """
        if not frame_data.flags.writeable:
            frame_data = frame_data.copy()

        if frame_data.ndim == 2:
            frame_data = frame_data[..., None]
            batch_dim = False
        else:
            batch_dim = True

        assert frame_data.ndim == 3
        is_floating_point = frame_data.dtype in [np.float32, np.float64]

        original_height, original_width = frame_data.shape[:2]
        pixel_rect, (scaled_w, scaled_h) = self._compute_pixel_rect(original_width, original_height)
        frame_data_pth = torch.from_numpy(frame_data).moveaxis(-1, 0)

        if is_floating_point:
            frame_data_pth = torch.nn.functional.interpolate(
                frame_data_pth[None],
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=True,
                antialias=True,
            )
        else:
            frame_data_pth = torch.nn.functional.interpolate(
                frame_data_pth[None].float(),
                size=(scaled_h, scaled_w),
                mode="nearest",
            )

        frame_data = frame_data_pth[0].moveaxis(0, -1).numpy().astype(frame_data.dtype)
        frame_data = frame_data[
            pixel_rect.j : pixel_rect.j + pixel_rect.height, pixel_rect.i : pixel_rect.i + pixel_rect.width
        ]
        return frame_data if batch_dim else frame_data[..., 0]

class NRMDataError(Exception):
    """
    Exception raised when an error occurs while loading NRM data.
    This is used to handle errors in a way that allows the dataset to continue loading other samples.
    """

    def __init__(self, message: str = "An error occurred while loading NRM data"):
        super().__init__(message)
        self.message = message


class BaseNRMDataset(torch.utils.data.Dataset[NRMDataBatch], ABC):
    """Base class for the NRM dataset; provides per-batch deterministic RNGs."""

    def _get_rng(self, batch_idx: int) -> np.random.Generator:
        """Hash (PL_GLOBAL_SEED, batch_idx) to get a per-batch deterministic generator."""
        assert "PL_GLOBAL_SEED" in os.environ, (
            "PL_GLOBAL_SEED environment variable is not set."
        )
        global_seed: int = int(os.environ["PL_GLOBAL_SEED"])
        digest = hashlib.sha256(f"{batch_idx}_{global_seed}".encode()).digest()
        return np.random.default_rng(seed=int.from_bytes(digest[:8], "big"))


class BaseNRMIndexableDataset(BaseNRMDataset, Sized):
    """
    Base class for the NRM dataset where the dataset is indexable via __getitem__.
    """

    def __getitem__(self, batch_idx: int) -> NRMDataBatch:
        current_batch_idx = batch_idx
        failed_batch_indices: list[int] = []

        # We allow up to 10 attempts.
        while len(failed_batch_indices) < 10:
            rng = self._get_rng(current_batch_idx)
            try:
                return self.getitem_allow_exceptions(current_batch_idx)
            except Exception as e:
                failed_batch_indices.append(current_batch_idx)
                # Iterate until we find a new valid batch index
                while (new_batch_idx := rng.integers(0, len(self))) in failed_batch_indices:
                    pass

                if isinstance(e, NRMDataError):
                    # Don't print the full exception traceback because this error is known.
                    logger.warning(
                        f"Known NRMDataError occurred while getting item {current_batch_idx} in {self.__class__.__name__}. "
                        f"Reason: {e.message}. Switching to a random index {new_batch_idx}."
                    )
                else:
                    logger.error(
                        f"Unexpected error occurred while getting item {current_batch_idx} in {self.__class__.__name__}. "
                        f"Switching to a random index {new_batch_idx}."
                    )
                    # Print the full exception traceback because this is unexpected.
                    logger.exception(e)
                    # Still allow continue execution since we don't want training to fail due to outliers.

                current_batch_idx = new_batch_idx

        # This definitely means that the dataset is broken, and we cannot recover.
        raise NRMDataError(
            f"{self.__class__.__name__} tried out {len(failed_batch_indices)} attempts = {failed_batch_indices} and none of them worked. "
            f"Please check the dataset integrity and ensure that the data is not corrupted."
        )

    @abstractmethod
    def __len__(self) -> int:
        """Returns number of items in the dataset"""
        ...

    @abstractmethod
    def getitem_allow_exceptions(self, batch_idx: int) -> NRMDataBatch: ...
