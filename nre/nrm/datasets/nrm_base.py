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
from typing import Iterator, Literal, Sized, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import ncore.data
import ncore.impl.common.transformations as ncore_transformations

from ncore.sensors import CameraModel
from nre.nrm.config.dataset import (
    CameraSubsamplerConfig,
    DummyNRMDatasetConfig,
    NRMMixedDatasetConfig,
    SingleNRMDatasetConfig,
    TestIndexNRMDatasetConfig,
)
from nre.nrm.datasets.registry import make as make_dataset
from nre.nrm.datasets.registry import register as register_dataset
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

    def apply_depth_data(
        self, depth_data: np.ndarray, mode: Literal["bilinear", "nearest-min"] = "bilinear"
    ) -> np.ndarray:
        """
        Apply reshape where depth_data is (H, W).
        This handles the case where depth is 0.

        Args:
            mode: If "nearest-min", use nearest-foreground downsampling via adaptive
                max-pool on negated depths. Otherwise use masked bilinear interpolation.
        """
        original_height, original_width = depth_data.shape[:2]
        pixel_rect, (scaled_w, scaled_h) = self._compute_pixel_rect(original_width, original_height)
        depth_data_pth = torch.from_numpy(depth_data).float()

        if mode == "nearest-min":
            depth_data_pth = depth_data_pth.clone()
            depth_data_pth[depth_data_pth == 0] = float("inf")
            depth_data_pth = -F.adaptive_max_pool2d(-depth_data_pth[None, None], (scaled_h, scaled_w))
            depth_data_pth[depth_data_pth == float("inf")] = 0.0
        else:
            depth_data_mask = (depth_data_pth > 0.0).float()
            depth_data_pth = F.interpolate(
                depth_data_pth[None, None],
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=True,
            )
            depth_data_mask = F.interpolate(
                depth_data_mask[None, None],
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=True,
            )
            depth_data_pth = depth_data_pth / (depth_data_mask + 1e-6)

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

    def apply_T_sensor_startend_timestamps(
        self,
        T_sensor_startend: np.ndarray,
        frame_start_timestamp_us: int,
        frame_end_timestamp_us: int,
        camera_parameters: ncore.data.ConcreteCameraModelParametersUnion,
    ) -> tuple[np.ndarray, int, int]:
        """
        Apply cropping to the start/end poses and timestamps.
        T_sensor_startend represents start/end poses relative to nre frame,
        which are interpolated to potentially _restricted_ local relative frame
        times due to cropping
        """
        original_width = camera_parameters.resolution[0].item()
        original_height = camera_parameters.resolution[1].item()
        pixel_rect, _ = self._compute_pixel_rect(original_width, original_height)

        rect_image_points_lt_rb = 0.5 + pixel_rect.subsample_factor * np.array(
            [
                [pixel_rect.i, pixel_rect.j],
                [
                    pixel_rect.i + (pixel_rect.width - 1.0 / pixel_rect.subsample_factor),
                    pixel_rect.j + (pixel_rect.height - 1.0 / pixel_rect.subsample_factor),
                ],
            ],
            dtype=np.float32,
        )

        camera_model = CameraModel.from_parameters(camera_parameters, device="cpu", dtype=torch.float32)
        # Compute relative frame times for the cropped region
        rect_relative_frame_times = np.sort(camera_model.image_points_relative_frame_times(rect_image_points_lt_rb))

        # Interpolate sensor poses to the cropped region's relative frame times
        T_sensor_startend_rect = ncore_transformations.PoseInterpolator(
            T_sensor_startend,
            [0.0, 1.0],
        ).interpolate_to_timestamps(rect_relative_frame_times)

        # Convert relative times to absolute timestamps
        camera_frame_duration_us = frame_end_timestamp_us - frame_start_timestamp_us

        frame_start_timestamp_us_rect = frame_start_timestamp_us + int(
            rect_relative_frame_times[0] * camera_frame_duration_us
        )
        frame_end_timestamp_us_rect = frame_start_timestamp_us + int(
            rect_relative_frame_times[1] * camera_frame_duration_us
        )

        return T_sensor_startend_rect, frame_start_timestamp_us_rect, frame_end_timestamp_us_rect


class NRMDataError(Exception):
    """
    Exception raised when an error occurs while loading NRM data.
    This is used to handle errors in a way that allows the dataset to continue loading other samples.
    """

    def __init__(self, message: str = "An error occurred while loading NRM data"):
        super().__init__(message)
        self.message = message


class BaseNRMDataset(torch.utils.data.Dataset[NRMDataBatch], ABC):
    """
    Base class for the NRM dataset where simple random number functions are provided.
    """

    # -1 (default) indicates a non-specified state.
    # This is commonly used for validation and testing where we don't want RNG to be dependent on epoch.
    _rng_epoch: int = -1

    # The actual epoch number, mainly used by augmentations to sub-select augmentation values.
    _epoch: int = -1

    def set_rng_epoch(self, rng_epoch: int) -> None:
        self._rng_epoch = rng_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def _get_rng(self, batch_idx: int) -> np.random.Generator:
        """
        rng is re-initialized for each data, and is uniquely determined by:
        global seed set via pl.seed_everything, epoch number, and data index.
        """
        assert "PL_GLOBAL_SEED" in os.environ, (
            "PL_GLOBAL_SEED environment variable is not set. Please call pl.seed_everything()."
        )
        global_seed: int = int(os.environ["PL_GLOBAL_SEED"])

        # nre.utils.misc.compute_process_local_rng_seed requires too many advancements to the RNG.
        # we use a simpler hashing approach here.
        # Collision rate: SHA256-64bit << SHA256-32bit ~= CRC32 << ADLER32 (very high collision rate)
        hased_value = hashlib.sha256(f"{self._rng_epoch}_{batch_idx}_{global_seed}".encode()).digest()
        hashed_int = int.from_bytes(hased_value[:8], "big")
        return np.random.default_rng(seed=hashed_int)


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
                return self.getitem_allow_exceptions(current_batch_idx, rng)
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
    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> NRMDataBatch: ...


class BaseNRMIterableDataset(torch.utils.data.IterableDataset[NRMDataBatch], BaseNRMDataset):
    """
    Base class for the NRM dataset where the dataset is iterable via __iter__.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[NRMDataBatch]:
        """Returns an iterator over the dataset"""
        ...


@register_dataset("nrm-dummy")
class DummyNRMDataset(BaseNRMIndexableDataset):
    """
    Dummy dataset as a placeholder for testing-only configs, or to override a mixture to empty.
    """

    def __init__(self, config: DummyNRMDatasetConfig, split: str = "train"):
        pass

    def __len__(self) -> int:
        """Returns number of items in the dataset"""
        return 0

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> NRMDataBatch:
        raise NRMDataError("Should not sample from dummy dataset.")


@register_dataset("nrm-test-index")
class TestIndexNRMDataset(BaseNRMIndexableDataset):
    """
    TestIndexNRMDataset is a dataset class that returns a batch of data at the given index.
    """

    def __init__(self, config: TestIndexNRMDatasetConfig, split: str = "train"):
        self.size = config.size

    def __len__(self) -> int:
        return self.size

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> NRMDataBatch:
        return NRMDataBatch(
            context=[],
            meta=[{"index": batch_idx}],
        )


@register_dataset("nrm-mixed")
class MixedNRMDataset(BaseNRMIndexableDataset):
    """
    MixedNRMDataset is a dataset class that combines multiple NRM datasets into a single dataset.
    """

    @dataclass
    class SubDataset:
        name: str
        config: SingleNRMDatasetConfig
        dataset: BaseNRMIndexableDataset
        sample_ratio: float

        full_length: int = field(init=False)
        sampled_length: int = field(init=False)

        def __post_init__(self):
            self.full_length = len(self.dataset)
            assert self.full_length > 0, f"Dataset {self.name} is empty."
            self.sampled_length = int(self.full_length * self.sample_ratio)
            logger.info(f"Dataset {self.name} has {self.full_length} samples, sampling {self.sampled_length} samples.")

        def sample_at(self, idx: int, rng: np.random.Generator) -> NRMDataBatch:
            """
            Sample a batch from the dataset at the given index.
            """
            assert 0 <= idx < self.sampled_length, f"Sub-index {idx} out of range for dataset {self.name}."

            # Always ensure all data is iterated over if over/full-sampling
            if self.sampled_length >= self.full_length and idx < self.full_length:
                return self.dataset[idx]

            # Otherwise we just do random sampling
            idx = rng.integers(self.full_length)
            return self.dataset[idx]

    def __init__(self, config: NRMMixedDatasetConfig, split: str = "train"):
        self.datasets: list[MixedNRMDataset.SubDataset] = []
        self.split = split

        for sub_config_name, sub_config in config.mixture.items():
            # Avoid creating empty datasets.
            if sub_config.sample_ratio == 0:
                continue
            sub_dataset = make_dataset(sub_config.config.name, sub_config.config, split=split)
            assert isinstance(sub_dataset, BaseNRMIndexableDataset), (
                f"Dataset {sub_config_name} instantiated is not a BaseNRMIndexableDataset, found {sub_dataset.__class__.__name__} instead."
            )
            self.datasets.append(
                MixedNRMDataset.SubDataset(
                    name=sub_config_name,
                    dataset=sub_dataset,
                    config=sub_config.config,
                    sample_ratio=sub_config.sample_ratio,
                )
            )

    def __len__(self) -> int:
        return sum(sub_dataset.sampled_length for sub_dataset in self.datasets)

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> NRMDataBatch:
        for sub_dataset in self.datasets:
            if batch_idx < sub_dataset.sampled_length:
                return sub_dataset.sample_at(batch_idx, rng)
            batch_idx -= sub_dataset.sampled_length
        raise IndexError(f"Batch index {batch_idx} out of range for mixed dataset.")

    def set_rng_epoch(self, rng_epoch):
        super().set_rng_epoch(rng_epoch)
        for sub_dataset in self.datasets:
            sub_dataset.dataset.set_rng_epoch(rng_epoch)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        for sub_dataset in self.datasets:
            sub_dataset.dataset.set_epoch(epoch)
