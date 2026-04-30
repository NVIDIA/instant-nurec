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
import os

from dataclasses import dataclass
from typing import Optional, Self

import imageio as iio
import torch

from einops import rearrange
from torchvision.transforms.functional import InterpolationMode, resize

from nre.grpc.protos.sensorsim_pb2 import AvailableEgoMasksReturn, EgoMaskId, RGBRenderRequest


logger = logging.getLogger("nre.grpc.serve")


@dataclass
class EgocarHood:
    """
    Holds an image of the egocar hood (visible in video but not in NRE renders)
    along with a segmentation mask to paste it in at the end of view generation.

    Since most renders are at the same resolution and the resizing operator is somewhat
    expensive, we cache the most recently used resolution of the mask to provide
    a fast path in the usual case.
    """

    camera_logical_id: str
    rig_id: str
    _hood_rgba: torch.Tensor  # C H W, RGBA, floating point in [0, 1]
    _resize_cache: torch.Tensor  # C H W, RGBA, floating point in [0, 1]

    @classmethod
    def load_from_file(cls, camera_logical_id: str, rig_id: str, path: str) -> Self:
        """
        Loads an egocar hood from a file.

        Assumes an RGBA image which is transparent outside of the car hood.

        Args:
            camera_logical_id: The logical ID of the camera (used as metadata)
            rig_id: The ID of the rig (used as metadata)
            path: The path to an RGBA png file

        Returns:
            An egocar hood object.
        """
        hood_rgba = iio.imread(path)
        assert hood_rgba.shape[-1] == 4, f"Image at {path} is not RGBA"

        hood_torch = torch.from_numpy(hood_rgba)

        hood_f32 = rearrange(hood_torch, "h w c -> c h w").float() / 255.0
        return cls(camera_logical_id=camera_logical_id, rig_id=rig_id, _hood_rgba=hood_f32, _resize_cache=hood_f32)

    def __post_init__(self):
        assert self._hood_rgba.ndim == 3
        assert self._hood_rgba.shape[0] == 4

    def overlay_on_image(self, rendered_image: torch.Tensor) -> torch.Tensor:
        """
        Overlay the egocar hood on a rendered image.

        Args:
            rendered_image: (C, H, W), RGB, floating point in [0, 1]

        Returns:
            (C, H, W), RGB, floating point in [0, 1]
        """
        if rendered_image.shape[0:2] != self._resize_cache.shape[1:3]:
            logger.info(
                f"EgocarHood: resize cache miss (need {rendered_image.shape}, have {self._resize_cache.shape})."
            )
            self._resize_cache = resize(
                self._hood_rgba,
                list(rendered_image.shape[0:2]),
                interpolation=InterpolationMode.BILINEAR,
            )

        self._resize_cache = self._resize_cache.to(rendered_image.device)

        hood = rearrange(self._resize_cache[:3], "c h w -> h w c")
        mask = self._resize_cache[3].unsqueeze(-1)  # (h, w, 1)

        return rendered_image * (1 - mask) + hood * mask

    @property
    def metadata(self) -> AvailableEgoMasksReturn.EgoMaskMetadata:
        return AvailableEgoMasksReturn.EgoMaskMetadata(
            ego_mask_id=EgoMaskId(camera_logical_id=self.camera_logical_id, rig_config_id=self.rig_id),
        )


@dataclass
class EgocarRig:
    """
    Holds a collection of EgocarHoods, keyed by camera logical ID.
    """

    rig_id: str
    _hoods: dict[str, EgocarHood]

    def get(self, camera_logical_id: str) -> Optional[EgocarHood]:
        return self._hoods.get(camera_logical_id, None)

    @classmethod
    def load_from_dir(cls, rig_id: str, path: str) -> Self:
        """Loads all ego hood images from <path>/<camera_logical_id>.png"""
        SUFFIX = ".png"

        hoods = {}
        for filename in os.listdir(path):
            if not filename.endswith(SUFFIX):
                continue
            camera_logical_id = filename.removesuffix(SUFFIX)
            hoods[camera_logical_id] = EgocarHood.load_from_file(
                camera_logical_id, rig_id, os.path.join(path, filename)
            )

        return cls(rig_id, hoods)

    def __repr__(self) -> str:
        return f"EgocarRig({', '.join(self._hoods.keys())})"

    def available_metadata(self) -> list[AvailableEgoMasksReturn.EgoMaskMetadata]:
        return [hood.metadata for hood in self._hoods.values()]


@dataclass
class EgocarRigBank:
    """
    Holds EgocarRigs corresponding to different rig configurations.
    """

    _rigs: dict[str, EgocarRig]

    @classmethod
    def load_from_dir(cls, path: str) -> Self:
        """Loads all egocar rigs from <path>/<rig_id>/<camera_logical_id>.png"""
        rigs = {}
        for rig_id in sorted(os.listdir(path)):
            rigs[rig_id] = EgocarRig.load_from_dir(rig_id, os.path.join(path, rig_id))

        return cls(rigs)

    @classmethod
    def empty(cls) -> Self:
        return cls({})

    def __repr__(self) -> str:
        return f"EgocarRigBank({', '.join(self._rigs.keys())})"

    def available_metadata(self) -> list[AvailableEgoMasksReturn.EgoMaskMetadata]:
        return [metadata for rig in self._rigs.values() for metadata in rig.available_metadata()]

    def select_from_request(self, request: RGBRenderRequest) -> Optional[EgocarHood]:
        if not request.insert_ego_mask:
            return None

        if request.ego_mask_id.rig_config_id not in self._rigs:
            raise KeyError(f"No egocar rig found for config ID {request.ego_mask_id.rig_config_id=}")

        return self._rigs[request.ego_mask_id.rig_config_id].get(request.ego_mask_id.camera_logical_id)
