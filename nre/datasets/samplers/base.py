# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, Union

import numpy as np
import numpy.typing as npt

from omegaconf import DictConfig

from nre.utils.batch import DataAndRenderingBatch, RectSubsampled
from nre.utils.profiling import ScopedTimer, TimingTag


if TYPE_CHECKING:
    from nre.datasets.ncore import NCORETrainDataset  # pycena: skip


@dataclass(kw_only=True)
class SensorSamplerReturn:
    sampled_sensor_ids: list[str]


class BaseSensorSampler(ABC):
    """Base sensor sampler class used to sample a single or multiple sensors from which a given batch is formed.
    These can be sampled randomly, or biased based on some probability distribution (e.g. error maps)"""

    SENSOR_SAMPLER_VARIANTS: dict[str, Type[BaseSensorSampler]] = {}

    @staticmethod
    def register_to_sensor_sampler_factory(name: str, cls: Type[BaseSensorSampler]) -> None:
        if name in BaseSensorSampler.SENSOR_SAMPLER_VARIANTS:
            raise KeyError(f"{name=} already in SENSOR_SAMPLER_VARIANTS.")
        BaseSensorSampler.SENSOR_SAMPLER_VARIANTS[name] = cls

    @staticmethod
    def sensor_sampler_factory(name: str, config: DictConfig, dataset: NCORETrainDataset) -> BaseSensorSampler:
        return BaseSensorSampler.SENSOR_SAMPLER_VARIANTS[name](config, dataset)

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        self.sample_all_sensors = config.sample_all_sensors

    @abstractmethod
    def sample_sensor(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        sensor_ids: list[str],
    ) -> SensorSamplerReturn:
        """Sample sensor from `sensor_ids` argument."""
        ...

    def update_epoch(self, epoch: int, system, **kwargs) -> None:
        pass


class SkipSensorSampler(BaseSensorSampler):
    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        pass

    def sample_sensor(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        sensor_ids: list[str],
    ) -> SensorSamplerReturn:
        """Sample sensor from `sensor_ids` argument."""
        return SensorSamplerReturn(sampled_sensor_ids=[])


@dataclass(kw_only=True)
class FrameSamplerReturn:
    """Return type of FrameSampler"""

    sampled_frame_idx: int


class BaseFrameSampler(ABC):
    """Base sensor frame sampler used to sample a frame from which a given batch is formed.
    These can be sampled randomly, or biased based on some probability distribution (e.g. error maps)"""

    FRAME_SAMPLER_VARIANTS: dict[str, Type[BaseFrameSampler]] = {}

    @staticmethod
    def register_to_frame_sampler_factory(name: str, cls: Type[BaseFrameSampler]) -> None:
        if name in BaseFrameSampler.FRAME_SAMPLER_VARIANTS:
            raise KeyError(f"{name=} already in FRAME_SAMPLER_VARIANTS.")
        BaseFrameSampler.FRAME_SAMPLER_VARIANTS[name] = cls

    @staticmethod
    def frame_sampler_factory(name: str, config: DictConfig, dataset: NCORETrainDataset) -> BaseFrameSampler:
        return BaseFrameSampler.FRAME_SAMPLER_VARIANTS[name](config, dataset)

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        pass

    @abstractmethod
    def sample_frame(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        unique_sensor_id: str,
    ) -> FrameSamplerReturn:
        """Sample frame from `frame_range` argument.

        Note on indexing: The returned 'camera_frame_idx' is in the provided 'frame_range' and is
                        *not* required to be zero-based
        """
        ...

    def update_epoch(self, epoch: int, system, **kwargs) -> None:
        pass


class SkipFrameSampler(BaseFrameSampler):
    def sample_frame(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        unique_sensor_id: str,
    ) -> FrameSamplerReturn:
        """Sample frame from `frame_range` argument.

        Note on indexing: The returned 'camera_frame_idx' is in the provided 'frame_range' and is
                        *not* required to be zero-based
        """
        ...
        return FrameSamplerReturn(sampled_frame_idx=-1)


@dataclass(kw_only=True)
class LidarPointSamplerReturn:
    """Return type of LidarPointSampler"""

    sampled_point_idxs: npt.NDArray


class BaseLidarPointSampler(ABC):
    """Base lidar point sampler class used to sample lidar points for a given batch.
    These can be sampled randomly, or biased based on some probability distribution (e.g. error maps)"""

    LIDAR_POINT_SAMPLER_VARIANTS: dict[str, Type[BaseLidarPointSampler]] = {}

    @staticmethod
    def register_to_lidar_point_sampler_factory(name: str, cls: Type[BaseLidarPointSampler]) -> None:
        if name in BaseLidarPointSampler.LIDAR_POINT_SAMPLER_VARIANTS:
            raise KeyError(f"{name=} already in LIDAR_POINT_SAMPLER_VARIANTS.")
        BaseLidarPointSampler.LIDAR_POINT_SAMPLER_VARIANTS[name] = cls

    @staticmethod
    def lidar_point_sampler_factory(name: str, config: DictConfig, dataset: NCORETrainDataset) -> BaseLidarPointSampler:
        return BaseLidarPointSampler.LIDAR_POINT_SAMPLER_VARIANTS[name](config, dataset)

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        pass

    @abstractmethod
    def sample_lidar_points(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_point_samples: int,
        frame_valid_points_mask: npt.NDArray,
        unique_lidar_id: str,
        lidar_frame_idx: int,
    ) -> LidarPointSamplerReturn:
        """Returns a sample set of lidar point

        Note on indexing: The provided 'lidar_frame_idx' has to be in the passed 'frame_range' and is *not*
                        required to be zero-based (the frame range can be used to, e.g., compute a corresponding
                        zero-based index)
        """
        ...

    def update_epoch(self, epoch: int, system, **kwargs) -> None:
        pass


class SkipLidarPointSampler(BaseLidarPointSampler):
    def sample_lidar_points(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_point_samples: int,
        frame_valid_points_mask: npt.NDArray,
        unique_lidar_id: str,
        lidar_frame_idx: int,
    ) -> LidarPointSamplerReturn:
        """Samples no lidar points / always returns empty sample point set

        Note on indexing: The provided 'lidar_frame_idx' has to be in the passed 'frame_range' and is *not*
                        required to be zero-based (the frame range can be used to, e.g., compute a corresponding
                        zero-based index)
        """
        return LidarPointSamplerReturn(sampled_point_idxs=np.empty(0, dtype=np.int32))


@dataclass(kw_only=True)
class CameraPixelSamplerReturn:
    """Return type of CameraPixelSamplers

    sampled_pixels is either a free set of pixel indices or `RectSubsampled`, in which case sampled pixels represent a contiguous set of
    indices that can be reshaped into an image/patch"""

    sampled_pixels: np.ndarray | RectSubsampled

    def __post_init__(self):
        assert isinstance(self.sampled_pixels, np.ndarray | RectSubsampled), "Invalid sampled_pixels type"


class BaseCameraPixelSampler(ABC):
    """Base camera pixel sampler class used to sample camera pixels for a given batch.
    These can be sampled randomly, or biased based on some probability distribution (e.g. error maps)"""

    CAMERA_PIXEL_SAMPLER_VARIANTS: dict[str, Type[BaseCameraPixelSampler]] = {}

    @staticmethod
    def register_to_camera_pixel_sampler_factory(name: str, cls: Type[BaseCameraPixelSampler]) -> None:
        if name in BaseCameraPixelSampler.CAMERA_PIXEL_SAMPLER_VARIANTS:
            raise KeyError(f"{name=} already in CAMERA_PIXEL_SAMPLER_VARIANTS.")
        BaseCameraPixelSampler.CAMERA_PIXEL_SAMPLER_VARIANTS[name] = cls

    @staticmethod
    def camera_pixel_sampler_factory(
        name: str, config: DictConfig, dataset: NCORETrainDataset
    ) -> BaseCameraPixelSampler:
        return BaseCameraPixelSampler.CAMERA_PIXEL_SAMPLER_VARIANTS[name](config, dataset)

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        pass

    @abstractmethod
    def sample_camera_pixels(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_pixel_samples: int,
        frame_all_pixels: npt.NDArray,
        frame_valid_pixels_mask: npt.NDArray,
        unique_camera_id: str,
        camera_frame_idx: int,
    ) -> CameraPixelSamplerReturn:
        """Sample pixels from `frame_all_pixels` argument

        Note on indexing: The provided 'camera_frame_idx' has to be in the passed 'frame_range' and is *not*
                        required to be zero-based (the frame range can be used to, e.g., compute a corresponding
                        zero-based index)
        """
        ...

    def update_epoch(self, epoch: int, system, **kwargs) -> None:
        pass


class SkipCameraPixelSampler(BaseCameraPixelSampler):
    def sample_camera_pixels(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_pixel_samples: int,
        frame_all_pixels: npt.NDArray,
        frame_valid_pixels_mask: npt.NDArray,
        unique_camera_id: str,
        camera_frame_idx: int,
    ) -> CameraPixelSamplerReturn:
        """Sample pixels from `frame_all_pixels` argument

        Note on indexing: The provided 'camera_frame_idx' has to be in the passed 'frame_range' and is *not*
                        required to be zero-based (the frame range can be used to, e.g., compute a corresponding
                        zero-based index)
        """
        return CameraPixelSamplerReturn(sampled_pixels=np.empty(0, dtype=np.int32))


# Register the skip modules to the factory
BaseSensorSampler.register_to_sensor_sampler_factory("skip", SkipSensorSampler)
BaseFrameSampler.register_to_frame_sampler_factory("skip", SkipFrameSampler)
BaseCameraPixelSampler.register_to_camera_pixel_sampler_factory("skip", SkipCameraPixelSampler)
BaseLidarPointSampler.register_to_lidar_point_sampler_factory("skip", SkipLidarPointSampler)


SamplerType = Union[BaseSensorSampler, BaseFrameSampler, BaseCameraPixelSampler, BaseLidarPointSampler]


class BaseBatchSampler(ABC):
    """Base camera pixel sampler class used to sample camera pixels for a given batch.
    These can be sampled randomly, or biased based on some probability distribution (e.g. error maps)"""

    BATCH_SAMPLER_VARIANTS: dict[str, Type[BaseBatchSampler]] = {}

    @staticmethod
    def register_to_batch_sampler_factory(name: str, cls: Type[BaseBatchSampler]) -> None:
        if name in BaseBatchSampler.BATCH_SAMPLER_VARIANTS:
            raise KeyError(f"{name=} already in BATCH_SAMPLER_VARIANTS.")
        BaseBatchSampler.BATCH_SAMPLER_VARIANTS[name] = cls

    @staticmethod
    def batch_sampler_factory(name: str, config: DictConfig, dataset: NCORETrainDataset) -> BaseBatchSampler:
        return BaseBatchSampler.BATCH_SAMPLER_VARIANTS[name](config, dataset)

    camera_sensor_sampler: BaseSensorSampler
    lidar_sensor_sampler: BaseSensorSampler
    camera_frame_sampler: BaseFrameSampler
    lidar_frame_sampler: BaseFrameSampler
    camera_pixel_sampler: BaseCameraPixelSampler
    lidar_point_sampler: BaseLidarPointSampler

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        self.logger = logging.getLogger(__name__)
        self.update_n_epochs = config.update_n_epochs

        # We round the number up so that we produce at least the dataset.n_train_sample_camera_rays and dataset.n_train_sample_lidar_rays across all samplers.
        # TODO: this used to be important with RollingBufferDataloader but is not used anymore.
        self.n_train_sample_camera_rays = math.ceil(config.ratio_camera_samples * dataset.n_train_sample_camera_rays)
        self.n_train_sample_lidar_rays = math.ceil(config.ratio_lidar_samples * dataset.n_train_sample_lidar_rays)

        if config.ratio_camera_samples == 0 and config.ratio_lidar_samples == 0:
            self.logger.warning(
                f"{type(self).__name__}: No camera or lidar samples used, as ratio_camera_samples = ratio_lidar_samples = {config.ratio_camera_samples}!"
            )

        # Check sampler compatibility
        if config.camera_frame_sampler.name == "timestamp" or config.lidar_frame_sampler.name == "timestamp":
            assert config.lidar_frame_sampler.name == "timestamp" and config.camera_frame_sampler.name == "timestamp", (
                f"{self.__class__.__name__} if one of the frame_samplers is timestamp based, all of the have to be timestamp based."
            )

        self.camera_sensor_sampler = (
            BaseSensorSampler.sensor_sampler_factory(
                config.camera_sensor_sampler.name, config.camera_sensor_sampler, dataset
            )
            if config.ratio_camera_samples > 0
            else SkipSensorSampler(config.camera_sensor_sampler, dataset)
        )

        self.lidar_sensor_sampler = (
            BaseSensorSampler.sensor_sampler_factory(
                config.lidar_sensor_sampler.name, config.lidar_sensor_sampler, dataset
            )
            if config.ratio_lidar_samples > 0
            else SkipSensorSampler(config.lidar_sensor_sampler, dataset)
        )

        self.camera_frame_sampler = (
            BaseFrameSampler.frame_sampler_factory(
                config.camera_frame_sampler.name, config.camera_frame_sampler, dataset
            )
            if config.ratio_camera_samples > 0
            else SkipFrameSampler(config.camera_frame_sampler, dataset)
        )

        # If the selected camera frame sampler is timestamp based the Lidar one has to be the same object, if lidar samples > 0
        if config.camera_frame_sampler.name == "timestamp":
            self.lidar_frame_sampler = (
                self.camera_frame_sampler
                if config.ratio_lidar_samples > 0
                else SkipFrameSampler(config.lidar_frame_sampler, dataset)
            )
        else:
            self.lidar_frame_sampler = (
                BaseFrameSampler.frame_sampler_factory(
                    config.lidar_frame_sampler.name, config.lidar_frame_sampler, dataset
                )
                if config.ratio_lidar_samples > 0
                else SkipFrameSampler(config.lidar_frame_sampler, dataset)
            )

        # If both the camera frame sampler and pixel sampler are error based they should share the object
        if config.camera_frame_sampler.name == "error" and config.camera_pixel_sampler.name == "error":
            assert config.camera_frame_sampler == config.camera_pixel_sampler, (
                f"{self.__class__.__name__} if both camera frame and pixel sampler are error based their configs should be the same"
            )
            assert isinstance(self.camera_frame_sampler, BaseCameraPixelSampler)
            self.camera_pixel_sampler = self.camera_frame_sampler

        else:
            self.camera_pixel_sampler = (
                BaseCameraPixelSampler.camera_pixel_sampler_factory(
                    config.camera_pixel_sampler.name, config.camera_pixel_sampler, dataset
                )
                if config.ratio_camera_samples > 0
                else SkipCameraPixelSampler(config.camera_pixel_sampler, dataset)
            )

        self.lidar_point_sampler = (
            BaseLidarPointSampler.lidar_point_sampler_factory(
                config.lidar_point_sampler.name, config.lidar_point_sampler, dataset
            )
            if config.ratio_lidar_samples > 0
            else SkipLidarPointSampler(config.lidar_point_sampler, dataset)
        )

        self.samplers: set[SamplerType] = set(
            [
                self.lidar_sensor_sampler,
                self.camera_sensor_sampler,
                self.lidar_frame_sampler,
                self.camera_frame_sampler,
                self.camera_pixel_sampler,
                self.lidar_point_sampler,
            ]
        )

    @abstractmethod
    def get_batch(self, batch_idx: int, dataset: NCORETrainDataset) -> DataAndRenderingBatch: ...

    def _is_current_epoch_update_epoch(self, epoch: int) -> bool:
        return self.update_n_epochs != 0 and (epoch + 1) % self.update_n_epochs == 0

    def update_epoch(self, epoch: int, system, **kwargs) -> None:
        if self._is_current_epoch_update_epoch(epoch):
            for sampler in self.samplers:
                sampler.update_epoch(epoch, system, **kwargs)

    def get_max_rays_num(self) -> int:
        return self.n_train_sample_camera_rays + self.n_train_sample_lidar_rays


class DefaultBatchSampler(BaseBatchSampler):
    @ScopedTimer("DefaultBatchSampler.get_batch", TimingTag.DATALOADER)
    def get_batch(self, batch_idx: int, dataset: NCORETrainDataset) -> DataAndRenderingBatch:
        return dataset.get_train_batch(
            batch_idx=batch_idx,
            n_train_sample_camera_rays=self.n_train_sample_camera_rays,
            camera_sensor_sampler=self.camera_sensor_sampler,
            camera_frame_sampler=self.camera_frame_sampler,
            camera_pixel_sampler=self.camera_pixel_sampler,
            n_train_sample_lidar_rays=self.n_train_sample_lidar_rays,
            lidar_sensor_sampler=self.lidar_sensor_sampler,
            lidar_frame_sampler=self.lidar_frame_sampler,
            lidar_point_sampler=self.lidar_point_sampler,
        )


BaseBatchSampler.register_to_batch_sampler_factory("default", DefaultBatchSampler)
