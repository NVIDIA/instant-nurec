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

import functools

from abc import abstractmethod
from typing import Any, Callable, List, Literal, Optional, Self, TypeVar

import torch
import torch.utils.checkpoint

from omegaconf import DictConfig, OmegaConf

from nre.config.version import get_version
from nre.models.gaussians.renderers import BaseGaussianRenderer, Gaussian3DNRenderer
from nre.models.nrenderable import NRenderableModel
from nre.nrm.config.models import PrimitiveExportPreprocessConfig
from nre.utils.batch import DataAndRenderingBatch, FrameMeta, RenderingData
from nre.utils.misc import map_optional, strip_none_from_config
from nre.utils.profiling import ScopedTimer
from nre.utils.types import (
    Checkpoint,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    RigTrajectories,
)


class BaseNRMPrimitive(NRenderableModel):
    """
    Base class for all renderable primitives reconstructed by an NRM.
    """

    @abstractmethod
    def forward(
        self,
        rendering_cam_data: Optional[RenderingData] = None,
        frames_cam_meta: Optional[List[FrameMeta]] = None,
        rendering_lidar_data: Optional[RenderingData] = None,
        frames_lidar_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansCompositeReturn:
        pass

    @abstractmethod
    def state_dict_and_config(self) -> tuple[dict[str, Any], DictConfig]:
        """Used for serialization to be used in nrend."""

    @abstractmethod
    def get_checkpoint(self) -> Checkpoint:
        """Used for serialization to be used in render."""

    @torch.no_grad()
    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        """Used only during initialization time, and/or test time with nrend."""

        state_dict, config = self.state_dict_and_config()
        config_dict = strip_none_from_config(OmegaConf.to_container(config))
        json_dict = {
            "nre_data": {
                "version": map_optional(get_version(), lambda v: v.semantic_string()),
                "model": "nre",
                # Empty to be filled by the derived class
                "config": config_dict,
                "state_dict": {f".{key}": value for key, value in state_dict.items()} if with_state_dict else {},
            }
        }

        if with_state_dict:
            assert isinstance(json_dict["nre_data"]["state_dict"], dict)
            # add shape entries for every tensor
            json_dict["nre_data"]["state_dict"].update(
                {
                    key + ".shape": list(value.size())
                    for key, value in json_dict["nre_data"]["state_dict"].items()
                    if isinstance(value, torch.Tensor) and isinstance(key, str)
                }
            )

            # convert tensor to bytes
            def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
                # default conversion to half for single precision tensor
                tensor = tensor.to(dtype=torch.float16) if tensor.dtype == torch.float32 else tensor
                return tensor.flatten().cpu().numpy().tobytes()

            json_dict["nre_data"]["state_dict"] = {
                key: tensor_to_bytes(value) if isinstance(value, torch.Tensor) else value
                for key, value in json_dict["nre_data"]["state_dict"].items()
            }

        return json_dict

    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def detach(self) -> BaseNRMPrimitive: ...

    @abstractmethod
    def rigid_transform(self, T_new: torch.Tensor) -> Self: ...

    @abstractmethod
    def preprocess_for_export(
        self,
        context_batch: DataAndRenderingBatch,
        config: PrimitiveExportPreprocessConfig,
        context_rig: RigTrajectories | None = None,
    ) -> Self:
        """
        Filter and preprocess the primitive for export (e.g. density/sky/road masking).
        Called per chunk after forward; when merging is enabled, merge will then apply
        rigid_transform to align chunks. Implementations must not apply rigid_transform.
        context_rig is optional; Celsius may require it when project_to_z_offset is True.
        """

    @abstractmethod
    def __len__(self) -> int: ...


NRMPrimitiveType = TypeVar("NRMPrimitiveType", bound=BaseNRMPrimitive)


class BaseGaussiansNRMPrimitive(BaseNRMPrimitive):
    """
    NRM primitives that is rendered using Gaussians and potentially postprocessings (e.g. sky and PPISPs).
    """

    # Here we allow the gaussians_renderer to be None to just carry the data.
    # The main use case is when we want to pass in this model for nrend initialization (to break circular dependency).
    gaussians_renderer: BaseGaussianRenderer | None

    def __init__(
        self,
        gaussians_renderer: BaseGaussianRenderer | None,
        checkpointing: Literal["render", "all", "none"] = "render",
        shared_gaussian_parameters: bool = False,
    ):
        self.gaussians_renderer = gaussians_renderer
        self.checkpointing = checkpointing
        if isinstance(gaussians_renderer, Gaussian3DNRenderer) and self.checkpointing == "all":
            gaussians_renderer.config.checkpoint_friendly_backward = True
        # If True, when rendering multiple frames within a single forward call, the first frame's FramesMeta will be used to get the gaussian parameters for all frames.
        # Useful to save memory when rendering multiple frames.
        self.shared_gaussian_parameters = shared_gaussian_parameters
        # TODO[JW]: This is a hack to allow the model to be used in the viewer.
        self._nrend_renderer = None

    @abstractmethod
    def get_gaussian_parameters(self, timestamps_us: torch.Tensor | None) -> dict[str, torch.Tensor]: ...

    def get_extra_ray_signal_infos(self) -> tuple[list[str], list[int], list[Callable]]:
        return ([], [], [])

    def postprocess_rendering(
        self, out: GaussiansRenderReturn, rendering_data: RenderingData, unique_sensor_idx: int | None
    ) -> GaussiansRenderReturn:
        """This is supposed to be called one image at a time."""
        return out

    @ScopedTimer("BaseGaussiansNRMPrimitive.forward")
    def forward(
        self,
        rendering_cam_data: Optional[RenderingData] = None,
        frames_cam_meta: Optional[List[FrameMeta]] = None,
        rendering_lidar_data: Optional[RenderingData] = None,
        frames_lidar_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansCompositeReturn:
        # Rendering logic should resemble GaussiansComposite.forward() but conducted in batch.

        # Precompute the gaussian parameters for all frames.
        shared_gaussian_parameters: dict[str, torch.Tensor] = {}
        if self.shared_gaussian_parameters:
            if rendering_cam_data is not None:
                # Use CPU copy to avoid GPU->CPU sync when .item() is called inside get_gaussian_parameters
                shared_gaussian_parameters = self.get_gaussian_parameters(
                    rendering_cam_data.timestamps_startend_us_cpu[0:1]
                )
            elif rendering_lidar_data is not None:
                # Use CPU copy to avoid GPU->CPU sync when .item() is called inside get_gaussian_parameters
                shared_gaussian_parameters = self.get_gaussian_parameters(
                    rendering_lidar_data.timestamps_startend_us_cpu[0:1]
                )

        # Render the images
        out_cam_list: list[GaussiansRenderReturn] = []
        if frames_cam_meta is not None:
            assert rendering_cam_data is not None, (
                f"{self.__class__.__name__} All camera information has to be provided for rendering."
            )
            out_cam_list = self._render_cameras(rendering_cam_data, frames_cam_meta, shared_gaussian_parameters)

        # Render the lidar
        out_lidar_list: list[GaussiansRenderReturn] = []
        if frames_lidar_meta is not None:
            assert rendering_lidar_data is not None, (
                f"{self.__class__.__name__} All lidar information has to be provided for rendering."
            )
            out_lidar_list = self._render_lidars(rendering_lidar_data, frames_lidar_meta, shared_gaussian_parameters)

        return GaussiansCompositeReturn(
            rendered_cam=GaussiansRenderReturn.collate_fn(out_cam_list, device=self.device()) if out_cam_list else None,
            rendered_lidar=GaussiansRenderReturn.collate_fn(out_lidar_list, device=self.device())
            if out_lidar_list
            else None,
        )

    def _render_cameras(
        self,
        rendering_data: RenderingData,
        frame_meta: List[FrameMeta],
        shared_gaussian_parameters: dict[str, torch.Tensor],
    ) -> list[GaussiansRenderReturn]:
        """Render camera frames."""
        assert rendering_data.b == len(frame_meta), (
            f"Expected batch size to be {len(frame_meta)}, but got {rendering_data.b}"
        )

        render_all_fn = (
            functools.partial(torch.utils.checkpoint.checkpoint, self._render_single_camera_frame, use_reentrant=False)
            if self.checkpointing == "all"
            else self._render_single_camera_frame
        )

        B = rendering_data.b
        out_cam_list: list[GaussiansRenderReturn] = []
        for batch_idx in range(B):
            out_cam = render_all_fn(rendering_data[batch_idx], frame_meta[batch_idx], shared_gaussian_parameters)
            out_cam_list.append(out_cam)

        return out_cam_list

    def _render_single_camera_frame(
        self,
        rendering_data: RenderingData,
        frame_meta: FrameMeta,
        shared_gaussian_parameters: dict[str, torch.Tensor],
    ) -> GaussiansRenderReturn:
        """Render a single camera frame."""
        assert self.gaussians_renderer is not None, f"{self.__class__.__name__}: gaussians_renderer has to be provided."
        assert rendering_data.b == 1, f"Expected batch size to be 1, but got {rendering_data.b}"

        gaussian_parameters = (
            shared_gaussian_parameters
            if self.shared_gaussian_parameters
            else self.get_gaussian_parameters(rendering_data.timestamps_startend_us)
        )
        render_fn = (
            self.gaussians_renderer.render_with_deferred_bp
            if self.checkpointing == "render"
            else self.gaussians_renderer.render
        )

        out = render_fn(
            rendering_data=rendering_data,
            gaussian_parameters=gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=self.get_extra_ray_signal_infos(),
            frame_meta=[frame_meta],
        )
        out = self.postprocess_rendering(out, rendering_data, frame_meta.unique_sensor_idx)

        return out

    def _render_lidars(
        self,
        rendering_data: RenderingData,
        frame_meta: List[FrameMeta],
        shared_gaussian_parameters: dict[str, torch.Tensor],
    ) -> list[GaussiansRenderReturn]:
        """Render lidar frames."""
        assert rendering_data.b == len(frame_meta), (
            f"Expected batch size to be {len(frame_meta)}, but got {rendering_data.b}"
        )

        render_all_fn = (
            functools.partial(torch.utils.checkpoint.checkpoint, self._render_single_lidar_frame, use_reentrant=False)
            if self.checkpointing == "all"
            else self._render_single_lidar_frame
        )

        B = rendering_data.b
        out_lidar_list: list[GaussiansRenderReturn] = []
        for batch_idx in range(B):
            out_lidar = render_all_fn(rendering_data[batch_idx], frame_meta[batch_idx], shared_gaussian_parameters)
            out_lidar_list.append(out_lidar)
        return out_lidar_list

    def _render_single_lidar_frame(
        self, rendering_data: RenderingData, frame_meta: FrameMeta, shared_gaussian_parameters: dict[str, torch.Tensor]
    ) -> GaussiansRenderReturn:
        """Render a single lidar frame."""
        assert self.gaussians_renderer is not None, f"{self.__class__.__name__}: gaussians_renderer has to be provided."
        assert rendering_data.b == 1, f"Expected batch size to be 1, but got {rendering_data.b}"

        gaussian_parameters = (
            shared_gaussian_parameters
            if self.shared_gaussian_parameters
            else self.get_gaussian_parameters(rendering_data.timestamps_startend_us)
        )
        render_fn = (
            self.gaussians_renderer.render_with_deferred_bp
            if self.checkpointing == "render"
            else self.gaussians_renderer.render
        )
        out = render_fn(
            rendering_data=rendering_data,
            gaussian_parameters=gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=self.get_extra_ray_signal_infos(),
            frame_meta=[frame_meta],
        )
        return self.postprocess_rendering(out, rendering_data, frame_meta.unique_sensor_idx)
