# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import omegaconf

from nre.datasets.samplers.base import (
    BaseCameraPixelSampler,
    CameraPixelSamplerReturn,
)
from nre.utils.batch import RectSubsampled
from nre.utils.profiling import ScopedTimer, TimingTag


if TYPE_CHECKING:
    from nre.datasets.ncore import NCORETrainDataset


class ImageCropCameraPixelSampler(BaseCameraPixelSampler):
    """Sampler for camera batches, samples pixels from the whole image or from a crop of the image, and uniform random frames"""

    def __init__(
        self,
        config: omegaconf.dictconfig.DictConfig,
        dataset: NCORETrainDataset,
    ) -> None:
        self.crop_type = config.crop_type
        self.subsample = config.subsample
        self.roi_lt = config.roi_lt
        self.roi_rb = config.roi_rb
        self.crop_ratio = config.crop_ratio

        match self.crop_type:
            case "preset_roi":
                assert (
                    self.roi_rb is not None
                    and len(self.roi_rb) == 2
                    and all([x >= 0.0 and x <= 1.0 for x in self.roi_rb])
                )
                assert (
                    self.roi_lt is not None
                    and len(self.roi_lt) == 2
                    and all([x >= 0.0 and x <= 1.0 for x in self.roi_lt])
                )

                self.crop_width_ratio = self.roi_rb[0] - self.roi_lt[0]
                self.crop_height_ratio = self.roi_rb[1] - self.roi_lt[1]
            case "random_crop":  # random_crop
                assert self.crop_ratio is not None and self.crop_ratio >= 0.0 and self.crop_ratio <= 1.0
                self.crop_width_ratio = self.crop_height_ratio = self.crop_ratio
            case "full_image":
                self.crop_width_ratio = self.crop_height_ratio = 1.0
            case _:
                raise ValueError(f"Unsupported crop type {self.crop_type}")

    @ScopedTimer("ImageCropCameraPixelSampler.sample_camera_pixels", TimingTag.DATALOADER)
    def sample_camera_pixels(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_pixel_samples: int,  # will be ignored in this sampler
        frame_all_pixels: npt.NDArray,
        frame_valid_pixels_mask: npt.NDArray,
        unique_camera_id: str,
        camera_frame_idx: int,
    ) -> CameraPixelSamplerReturn:
        # Note that in ImageCropPixelSampler we can sample any pixels regardless
        # if they are considered valid or not, we just use the frame_valid_pixels_mask
        # to extract the image dimensions
        original_height = frame_valid_pixels_mask.shape[0]
        original_width = frame_valid_pixels_mask.shape[1]

        # Assert that the subsampling factor is valid
        assert original_width % self.subsample == 0 and original_height % self.subsample == 0, (
            f"Subsample factor {self.subsample} invalid, resolution is {original_width}x{original_height}"
        )

        img_width = original_width // self.subsample
        img_height = original_height // self.subsample

        # Compute crop dimensions after subsampling
        crop_height = int(img_height * self.crop_height_ratio)
        crop_width = int(img_width * self.crop_width_ratio)

        if (crop_type := self.crop_type) == "full_image":
            crop_img_lt_x = crop_img_lt_y = 0
        elif crop_type == "preset_roi":
            crop_img_lt_y = int(self.roi_lt[1] * img_height)
            crop_img_lt_x = int(self.roi_lt[0] * img_width)
            crop_img_rb_y = int(self.roi_rb[1] * img_height)
            crop_img_rb_x = int(self.roi_rb[0] * img_width)

            # Adjust to account for rounding up
            crop_width = crop_img_rb_x - crop_img_lt_x
            crop_height = crop_img_rb_y - crop_img_lt_y
        elif crop_type == "random_crop":
            # Determine the crop size
            crop_img_height = int(img_height * self.crop_height_ratio)
            crop_img_width = int(img_width * self.crop_width_ratio)

            # Determine the ROI's left top point
            crop_img_lt_y = int(rng.random() * (img_height - crop_img_height))
            crop_img_lt_x = int(rng.random() * (img_width - crop_img_width))
        else:
            raise ValueError(f"Unsupported crop type {crop_type}")

        # return the selected image rectangle
        return CameraPixelSamplerReturn(
            sampled_pixels=RectSubsampled(
                i=crop_img_lt_x,
                j=crop_img_lt_y,
                height=crop_height,
                width=crop_width,
                subsample_factor=float(self.subsample),
                original_width=original_width,
                original_height=original_height,
            ),
        )


BaseCameraPixelSampler.register_to_camera_pixel_sampler_factory("image-crop", ImageCropCameraPixelSampler)
