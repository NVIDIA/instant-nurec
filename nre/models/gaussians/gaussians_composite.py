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

import copy
import functools
import logging
import operator

from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, cast

import lietorch as lt
import torch

from omegaconf import ListConfig

import nre.models.gaussians.collect as collect

from nre.config.model import ModelConfig
from nre.config.trainer import TrainerConfig
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks
from nre.models.background import BaseBackground
from nre.models.base import BaseModel
from nre.models.calib import BaseCalib
from nre.models.composite import CompositeModel, LayerTrackIds
from nre.models.gaussians.gaussians_model import (
    BaseGaussianModel,
    DeformableGaussianModel,
    GaussianExportFormat,
    RigidGaussianModel,
    SHGaussianModel,
    distributed_all_gather_gaussian_parameters,
)
from nre.models.gaussians.initializations import BaseInitialization, NoPointsFoundException
from nre.models.gaussians.renderers import BaseGaussianRenderer
from nre.models.gaussians.strategies import (
    BaseGaussianStrategy,  # Note: this has to be imported from strategies/__init__.py, not base.py to force imports of all strategies which is required for pycena.
)
from nre.models.gaussians.utils import SH2RGB, Asset
from nre.models.nn_extensions import TypedModuleDict, TypedModuleList
from nre.models.nrenderable import NRenderableModel
from nre.models.post_processing import BasePostProcessing
from nre.models.utils import model_config_compatibility_check, update_module_step
from nre.utils.batch import (
    DataAndRenderingBatch,
    FrameMeta,
    RectSubsampled,
    RenderingData,
)
from nre.utils.geometry import se3_matrix_to_tquat
from nre.utils.misc import all_gather_int32, unpack_optional
from nre.utils.optim import (
    OptimizerLRSchedulerConfig,
)
from nre.utils.profiling import ScopedTimer
from nre.utils.types import (
    AABB3D,
    ExtraSignal,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    SceneContractor,
)
from nre.visualdebugger import get_visualdebugger


log = logging.getLogger(__name__)


class GaussiansComposite(BaseModel, CompositeModel, NRenderableModel):
    config: ModelConfig  # type: ignore[assignment] # Override type annotation from BaseModel

    calib: BaseCalib
    background: BaseBackground
    gaussians_renderer_lidar: BaseGaussianRenderer
    gaussians_renderer_camera: BaseGaussianRenderer
    gaussians_nodes: TypedModuleDict[BaseGaussianModel]
    post_processings: TypedModuleList[BasePostProcessing]

    _scene_contractor: SceneContractor
    scene_extent: float

    last_step: int = 0
    debug_viz: bool
    saturate_radiance: bool

    def __init__(
        self,
        config: ModelConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
    ):
        BaseModel.__init__(self, config.to_dictconfig())
        NRenderableModel.__init__(self)
        self.config = config  # type: ignore[assignment] # Override with typed config
        self.trainer_config = trainer_config

        log.info(f"{self.__class__.__name__}/__init__")

        # Check that the selected modules are compatible before building a model
        model_config_compatibility_check(self.config)

        self.scene_extent = datasource.get_aabb().get_extent().max().item()

        calib = unpack_optional(self.config.calib)
        self.calib = BaseCalib.factory(calib.name, calib, trainer_config, datasource)

        background = unpack_optional(self.config.background)
        self.background = BaseBackground.factory(background.name, background, trainer_config)

        # initialize cuboid tracks
        if (cuboid_tracks := datasource.get_cuboid_tracks(dynamic_only=False)) is not None:
            # Clean track IDs to remove "@<source>" suffixes for consistency
            self.cuboid_tracks = CuboidTracks.Ops.clean_track_ids(cuboid_tracks, DataSourceSummary._clean_track_id_str)
        else:
            self.cuboid_tracks = CuboidTracks.Factory.empty()

        gaussians_nodes: dict[str, BaseGaussianModel] = {}
        gaussian_template_node_model: Optional[BaseGaussianModel] = None
        self.obj_track_ids: dict[str, LayerTrackIds] = {}

        rig_trajectories = datasource.get_rig_trajectories()

        for layer_id, layer_config in self.config.layers.items():
            obj_track_id = LayerTrackIds(config=(tracks_config := layer_config.to_dictconfig().get("tracks", {})))

            obj_track_id.initialize_from_tracks(self.cuboid_tracks)
            self.obj_track_ids[layer_id] = obj_track_id

            if obj_track_id.is_empty() and tracks_config:
                continue

            cuboid_subset = obj_track_id.get_layer_tracks(self.cuboid_tracks)

            # Check if the class specific labels are specified or if there is an intradependency on other layers
            labels_to_ignore = []
            labels_to_use = self.config.layers[layer_id].class_labels or []
            if layers_to_ignore := (self.config.layers[layer_id].ignore_classes_from_layers or []):
                labels_to_ignore.extend(
                    [
                        label
                        for layer in layers_to_ignore
                        for label in unpack_optional(self.config.layers[layer].class_labels)
                    ]
                )

            # Set start and end timestamp for SHGaussianModel with temporal appearance
            if unpack_optional(layer_config.fourier_features_dim) > 1:
                assert rig_trajectories is not None, (
                    "rig_trajectories is required for SHGaussianModel with temporal appearance"
                )
                assert len(rig_trajectories.rig_trajectories), (
                    f"{self.__class__.__name__}: expected at least a single rig_trajectory, got {len(rig_trajectories.rig_trajectories)}"
                )
                start_end_timestamp_us = (
                    int(rig_trajectories.rig_trajectories[0].T_rig_world_timestamps_us[0].item()),
                    int(rig_trajectories.rig_trajectories[-1].T_rig_world_timestamps_us[-1].item()),
                )
            else:
                start_end_timestamp_us = None

            try:
                initializer = BaseInitialization.factory(
                    layer_config.initialization.name,
                    layer_config.initialization,
                    layer_config,
                    trainer_config,
                    labels_to_ignore=labels_to_ignore,
                    labels_to_use=labels_to_use,
                )

                gaussians_nodes[layer_id] = BaseGaussianModel.factory(
                    layer_config.name,
                    layer_config,
                    trainer_config,
                    datasource,
                    init_from_datasource,
                    initializer,
                    layer_config.precision,
                    cuboid_subset,
                    start_end_timestamp_us,
                )

                if gaussian_template_node_model is None:
                    gaussian_template_node_model = gaussians_nodes[layer_id]

            except Exception as e:
                match e:
                    case NoPointsFoundException():
                        continue
                    case _:
                        raise RuntimeError(f"Failed to initialize {layer_config.name} with error: {e}") from e

        layers_camera_extra_ray_signal_infos = [
            layer.camera_extra_ray_signal_infos for layer in gaussians_nodes.values()
        ]
        self.camera_extra_ray_signal_infos = layers_camera_extra_ray_signal_infos[0]
        for camera_extra_ray_signal_infos in layers_camera_extra_ray_signal_infos[1:]:
            assert (
                self.camera_extra_ray_signal_infos[0] == camera_extra_ray_signal_infos[0]
                and self.camera_extra_ray_signal_infos[1] == camera_extra_ray_signal_infos[1]
            ), "camera_extra_ray_signal_infos must be the same for all layers"

        layers_lidar_extra_ray_signal_infos = [layer.lidar_extra_ray_signal_infos for layer in gaussians_nodes.values()]
        self.lidar_extra_ray_signal_infos = layers_lidar_extra_ray_signal_infos[0]
        for lidar_extra_ray_signal_infos in layers_lidar_extra_ray_signal_infos[1:]:
            assert (
                self.lidar_extra_ray_signal_infos[0] == lidar_extra_ray_signal_infos[0]
                and self.lidar_extra_ray_signal_infos[1] == lidar_extra_ray_signal_infos[1]
            ), "lidar_extra_ray_signal_infos must be the same for all layers"

        post_processings: list[BasePostProcessing] = []
        post_processings_config = self.config.post_processing or {}
        for config_key in sorted(post_processings_config.keys()):
            post_processing = BasePostProcessing.factory(
                post_processings_config[config_key].name,
                post_processings_config[config_key],
                trainer_config,
                n_frames_per_camera=datasource.get_n_frames_per_camera().tolist(),
                n_frames_per_lidar=[],
            )
            post_processings.append(post_processing)

        self.gaussians_nodes = TypedModuleDict(gaussians_nodes)
        self.gaussians_strategy = BaseGaussianStrategy.factory(
            self.config.strategy.name,
            self.config.strategy,
            trainer_config,
            init_from_datasource,
            self.gaussians_nodes,
        )

        self.post_processings = TypedModuleList(post_processings)

        self.debug_viz = self.config.debug_viz
        self.overriden_state: dict[str, Any] = {}

        # Create separate renderer instances for camera and lidar to enable thread-safe parallel rendering.
        # Each renderer maintains internal state (via update_model_parameters) that is not thread-safe.
        self._camera_renderer = BaseGaussianRenderer.factory(
            self.config.renderer.name, self.config.renderer, unpack_optional(gaussian_template_node_model)
        )
        self._lidar_renderer = BaseGaussianRenderer.factory(
            self.config.renderer.name, self.config.renderer, unpack_optional(gaussian_template_node_model)
        )
        # Keep backward compatibility aliases
        self.gaussians_renderer = self._camera_renderer
        self.gaussians_renderer_camera = self._camera_renderer
        self.gaussians_renderer_lidar = self._lidar_renderer

        # Dedicated CUDA stream for lidar rendering (camera uses default stream)
        self._lidar_stream: torch.cuda.Stream = torch.cuda.Stream()

        # Multi-GPU: index mapping for slicing renderer scene_data to local rank (set in collect_gaussian_parameters)
        self._visibility_local_indices: Optional[torch.Tensor] = None

        # We can have different renderers for camera and lidar frames.
        # They can be overriden from the command line.
        # Ex: to override the renderer camera
        # model.renderer.use_gsplat_for_camera_rendering=True
        if getattr(self.config.renderer, "use_gsplat_for_camera_rendering", False):
            config_gsplat = copy.deepcopy(self.config.renderer)
            config_gsplat.name = "3dgut-gsplat"
            # Instantiate the renderer used for camera frames (replaces _camera_renderer)
            self._camera_renderer = BaseGaussianRenderer.factory(
                config_gsplat.name, config_gsplat, unpack_optional(gaussian_template_node_model)
            )

        self.saturate_radiance = getattr(self.config, "saturate_radiance", True)

        # Create the gaussian parameter collector.
        layers_config = self._get_layers_config()
        self.collector = collect.CreateGaussianParameterCollector(layers_config)

        # Copy the collector to each node, so that each node can access the "global" functions.
        for node in self.gaussians_nodes.values():
            node = cast(SHGaussianModel, node)
            node.collector = self.collector

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        # The sensor models in calib aren't stored in the state_dict, let's keep on using the one that
        # we have created in the constructor.

        # TODO: ideally this code should be in the load_state_dict of the model that
        # contains the sensor models (CameraFreePoseViewGeometry or LidarFreePoseViewGeometry)
        # and somehow called from here (through BaseCalib)

        calib_sensor_models = {f"calib.{k}": v for k, v in self.calib.state_dict().items() if ".sensor_models." in k}

        updated_state_dict = dict(state_dict)
        updated_state_dict.update(calib_sensor_models)
        return super().load_state_dict(updated_state_dict, *args, **kwargs)

    def configure_sharded_params_and_buffers(self) -> list[str]:
        ret = []
        for name, node in self.gaussians_nodes.items():
            per_node = node.configure_sharded_params_and_buffers()
            ret += [f"gaussians_nodes.{name}.{p}" for p in per_node]
        return ret

    def _initialize_scene_contractor(self, aabb: AABB3D) -> None:
        self._scene_contractor = SceneContractor(
            None,
            aabb,
            is_merf=False,
        )

    def initialize_strategy_and_maybe_gaussians(self) -> None:
        """
        Actually initialize Gaussians Models and Strategies, potentially in a distributed environment.
        """
        for name, node in self.gaussians_nodes.items():
            log.info(f"GaussiansComposite/maybe_initialize_gaussians: node={name}")
            node.maybe_initialize_gaussians()
        self.gaussians_strategy.maybe_initialize_buffers(self.gaussians_nodes)

    def get_scene_contractor(self) -> SceneContractor:
        return self._scene_contractor

    def get_gaussians_node_ids(self, non_empty_only: bool = True) -> list[str]:
        keys = list(self.gaussians_nodes.keys())
        if non_empty_only:
            keys = [k for k in keys if not self.obj_track_ids[k].is_empty()]
        return keys

    def _collect_sensor(
        self,
        rendering_data: RenderingData,
        is_training_batch: bool,
        gaussian_nodes: Optional[list[str]],
        track_transform_ids: Optional[List[str]],
        track_transforms: Optional[torch.Tensor],
        edited_cuboid_tracks: Optional[CuboidTracks],
        sensor_name: str,
    ) -> dict[str, torch.Tensor]:
        """
        Collect gaussian parameters for a single sensor.

        This is the first phase of rendering - gathering all gaussian data
        needed for the actual render call.
        """
        with ScopedTimer(f"GaussiansComposite/collect_gaussian_parameters_{sensor_name}"):
            return self.collect_gaussian_parameters(
                rendering_data,
                is_training_batch,
                gaussian_nodes,
                track_transform_ids,
                track_transforms,
                edited_cuboid_tracks,
            )

    def _render_collected(
        self,
        rendering_data: RenderingData,
        frame_meta: List[FrameMeta],
        gaussian_parameters: dict[str, torch.Tensor],
        extra_ray_signal_infos: tuple,
        renderer: BaseGaussianRenderer,
        is_training_batch: bool,
        sensor_name: str,
    ) -> GaussiansRenderReturn:
        """
        Render using pre-collected gaussian parameters.

        This is the second phase of rendering - the actual GPU render call.
        """
        with ScopedTimer(f"GaussiansComposite/render_{sensor_name}"):
            out = renderer.render(
                rendering_data=rendering_data,
                gaussian_parameters=gaussian_parameters,
                n_active_features=self.get_n_active_features(),
                extra_ray_signal_infos=extra_ray_signal_infos,
                frame_meta=frame_meta,
            )

        if not is_training_batch:
            # Normalize the normal to unit length for inference
            out.normal = torch.nn.functional.normalize(out.normal, dim=-1) if out.normal is not None else None

        return out

    def _slice_scene_data_to_local(self, out: Optional[GaussiansRenderReturn]) -> None:
        """Slice scene_data fields (visibility, cumulated_weights) to the local rank's portion.

        In multi-GPU training, the renderer operates on all-gathered gaussians and produces
        per-gaussian scene_data of global size (total across all ranks). This method slices
        those fields down to the local rank's shard using the index mapping computed during
        collect_gaussian_parameters.
        """
        if out is None or self._visibility_local_indices is None:
            return
        if out.visibility is not None:
            out.visibility = out.visibility[self._visibility_local_indices]
        if out.cumulated_weights is not None:
            out.cumulated_weights = out.cumulated_weights[self._visibility_local_indices]

    @ScopedTimer("GaussiansComposite/forward_gaussians")
    def forward_gaussians(
        self,
        rendering_data_cam: Optional[RenderingData] = None,
        rendering_data_lidar: Optional[RenderingData] = None,
        frame_meta_cam: Optional[List[FrameMeta]] = None,  # List is for batch size
        frame_meta_lidar: Optional[List[FrameMeta]] = None,  # List is for batch size
        is_training_batch: bool = False,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
        gaussian_nodes: Optional[list[str]] = None,
    ) -> GaussiansCompositeReturn:
        assert rendering_data_cam is not None or rendering_data_lidar is not None, (
            f"{self.__class__.__name__} Either camera or lidar must be provided"
        )

        # Clear cached indices so they are recomputed with current gaussian counts.
        # Counts may change between steps due to densification/pruning strategies.
        self._visibility_local_indices = None

        out_cam: Optional[GaussiansRenderReturn] = None
        out_lidar: Optional[GaussiansRenderReturn] = None
        gaussian_cam_parameters: Optional[dict[str, torch.Tensor]] = None
        gaussian_lidar_parameters: Optional[dict[str, torch.Tensor]] = None

        has_cam = rendering_data_cam is not None
        has_lidar = rendering_data_lidar is not None

        if has_cam:
            assert frame_meta_cam is not None and len(frame_meta_cam) == 1, "Only single-frame batch is supported"
        if has_lidar:
            assert frame_meta_lidar is not None and len(frame_meta_lidar) == 1, "Only single-frame batch is supported"

        # Flow: collect_lidar (dedicated stream) -> collect_cam (default stream)
        #       -> render_lidar (dedicated stream) -> render_cam (default stream)
        # All operations on main thread, lidar uses dedicated CUDA stream for GPU parallelism.

        if has_lidar:
            self._lidar_stream.wait_stream(
                torch.cuda.current_stream()
            )  # lidar stream depends on inputs computed on the default stream
            rd_lidar = cast(RenderingData, rendering_data_lidar)
            # Collect lidar on dedicated stream
            with torch.cuda.stream(self._lidar_stream):
                gaussian_lidar_parameters = self._collect_sensor(
                    rd_lidar,
                    is_training_batch,
                    gaussian_nodes,
                    track_transform_ids,
                    track_transforms,
                    edited_cuboid_tracks,
                    "lidar",
                )
                # record stream so tensors created outside of lidar stream are not deallocated before lidar stream is done
                # skip rd_lidar because it will be recorded in the _render_collected call below.
                # Resolve to all nodes when None, matching collect_gaussian_parameters behavior.
                used_gaussian_nodes = gaussian_nodes if gaussian_nodes is not None else self.get_gaussians_node_ids()
                for node_id in used_gaussian_nodes:
                    gaussians = self.gaussians_nodes[node_id]
                    gaussians.record_stream(self._lidar_stream)
                if track_transforms is not None:
                    track_transforms.record_stream(self._lidar_stream)

        if has_cam:
            rd_cam = cast(RenderingData, rendering_data_cam)
            # Collect cam on default stream
            gaussian_cam_parameters = self._collect_sensor(
                rd_cam,
                is_training_batch,
                gaussian_nodes,
                track_transform_ids,
                track_transforms,
                edited_cuboid_tracks,
                "cam",
            )

        if has_lidar:
            rd_lidar = cast(RenderingData, rendering_data_lidar)
            # Render lidar on dedicated stream
            with torch.cuda.stream(self._lidar_stream):
                out_lidar = self._render_collected(
                    rd_lidar,
                    cast(List[FrameMeta], frame_meta_lidar),
                    cast(dict[str, torch.Tensor], gaussian_lidar_parameters),
                    self.lidar_extra_ray_signal_infos,
                    self._lidar_renderer,
                    is_training_batch,
                    "lidar",
                )
                # record stream so tensors created outside of lidar stream are not deallocated before lidar stream is done
                if rd_lidar is not None:
                    rd_lidar.record_stream(self._lidar_stream)
                # skip gaussian_lidar_parameters because it is generated within the context manager of self._lidar_stream. So it will automatically be recorded.
                if frame_meta_lidar is not None:
                    for frame_meta in frame_meta_lidar:
                        frame_meta.record_stream(self._lidar_stream)
                # skip self.lidar_extra_ray_signal_infos. Assumption: it doesn't contain any tensors used by any kernel to compute out_lidar.
                # skip self._lidar_renderer. Assumption: it doesn't contain any tensors used by any kernel to compute out_lidar.

        if has_cam:
            rd_cam = cast(RenderingData, rendering_data_cam)
            # Render cam on default stream
            out_cam = self._render_collected(
                rd_cam,
                cast(List[FrameMeta], frame_meta_cam),
                cast(dict[str, torch.Tensor], gaussian_cam_parameters),
                self.camera_extra_ray_signal_infos,
                self._camera_renderer,
                is_training_batch,
                "cam",
            )

        # Synchronize the lidar stream with the default stream to ensure lidar results are ready
        # before they are accessed by the caller (e.g., for opacity thresholding in render.py)
        if has_lidar:
            torch.cuda.current_stream().wait_stream(self._lidar_stream)

        # Multi-GPU: slice scene_data (visibility, cumulated_weights) from global to local rank size.
        # The renderer operates on all-gathered gaussians, so these tensors have length = total gaussians
        # across all ranks. The losses expect them to match the local (sharded) parameter count.
        if self._visibility_local_indices is not None:
            self._slice_scene_data_to_local(out_cam)
            self._slice_scene_data_to_local(out_lidar)

        # Combine deformation smoothness from both modalities
        deform_smoothness = deform_smoothness_camera = deform_smoothness_lidar = None
        deform_smoothness_mask = deform_smoothness_mask_camera = deform_smoothness_mask_lidar = None
        if gaussian_cam_parameters is not None:
            deform_smoothness_camera = gaussian_cam_parameters.get("deform_smoothness", None)
            deform_smoothness_mask_camera = gaussian_cam_parameters.get("deform_smoothness_mask", None)
        if gaussian_lidar_parameters is not None:
            deform_smoothness_lidar = gaussian_lidar_parameters.get("deform_smoothness", None)
            deform_smoothness_mask_lidar = gaussian_lidar_parameters.get("deform_smoothness_mask", None)

        if deform_smoothness_camera is not None and deform_smoothness_lidar is not None:
            deform_smoothness = torch.cat([deform_smoothness_camera, deform_smoothness_lidar], dim=0)
        elif deform_smoothness_camera is not None:
            deform_smoothness = deform_smoothness_camera
        elif deform_smoothness_lidar is not None:
            deform_smoothness = deform_smoothness_lidar

        if deform_smoothness_mask_camera is not None and deform_smoothness_mask_lidar is not None:
            deform_smoothness_mask = torch.cat([deform_smoothness_mask_camera, deform_smoothness_mask_lidar], dim=0)
        elif deform_smoothness_mask_camera is not None:
            deform_smoothness_mask = deform_smoothness_mask_camera
        elif deform_smoothness_mask_lidar is not None:
            deform_smoothness_mask = deform_smoothness_mask_lidar

        return GaussiansCompositeReturn(
            rendered_cam=out_cam,
            rendered_lidar=out_lidar,
            deform_smoothness=deform_smoothness,
            deform_smoothness_mask=deform_smoothness_mask,
        )

    def get_n_active_features(self) -> int:
        """Return the first nodes's number of active features"""
        for gaussians in self.gaussians_nodes.values():
            if hasattr(gaussians, "n_active_features"):
                n_active_features = getattr(gaussians, "n_active_features")
                if isinstance(n_active_features, int):
                    return n_active_features
        return 0

    @ScopedTimer("GaussiansComposite/forward")
    def forward(
        self,
        batch: DataAndRenderingBatch,
        *,
        global_step: int = 0,
        do_render: bool = True,
        enable_calib: bool = True,
        enable_pp: bool = True,
        is_training_batch: bool = False,
        gaussian_nodes: Optional[list[str]] = None,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
        render_background: bool = True,
        **extra_rays_data,
    ) -> GaussiansCompositeReturn:
        """Main forward pass logic"""

        # Evaluate calibration and update batch.rendering (inplace).
        # This makes sure after this call, batch.rendering is not None.
        if batch.rendering is None:
            with ScopedTimer(f"GaussiansComposite/calib/{self.calib.__class__.__name__}"):
                batch.rendering = self.calib(
                    batch.data, skip_calib=not enable_calib, global_step_for_prober=global_step
                )

        # Gaussians forward
        out = self.forward_gaussians(
            rendering_data_cam=batch.rendering.camera,
            rendering_data_lidar=batch.rendering.lidar,
            frame_meta_cam=batch.data.camera.meta if batch.data.camera is not None else None,
            frame_meta_lidar=batch.data.lidar.meta if batch.data.lidar is not None else None,
            is_training_batch=is_training_batch,
            gaussian_nodes=gaussian_nodes,
            track_transform_ids=track_transform_ids,
            track_transforms=track_transforms,
            edited_cuboid_tracks=edited_cuboid_tracks,
            **extra_rays_data,
        )

        # Apply the background if selected
        if render_background and batch.rendering.camera is not None:
            with ScopedTimer(f"GaussiansComposite/background/{self.background.__class__.__name__}"):
                self.background(batch.rendering.camera.rays[..., 3:].reshape(-1, 3), out, is_training_batch)

        # Apply post-processing. We assume single image for post-processing. And only for camera for now.
        if (
            enable_pp
            and len(self.post_processings) > 0
            and batch.rendering.camera is not None
            and batch.data.camera is not None
        ):
            with ScopedTimer("GaussiansComposite/post_processing/subsample"):
                assert batch.rendering.camera.b == 1, "Post-processing only supports batch size 1."
                subsample = batch.data.camera.meta[0].subsample
                if subsample is None:
                    subsample = RectSubsampled(
                        original_width=batch.rendering.camera.w,
                        original_height=batch.rendering.camera.h,
                        width=batch.rendering.camera.w,
                        height=batch.rendering.camera.h,
                    )
                coords_xy = subsample.coordinates_in_original_sensor(
                    normalized=True, device=batch.rendering.camera.rays.device
                )
            for post_processing in self.post_processings:
                with ScopedTimer(
                    f"GaussiansComposite/post_processing/post_processing/{post_processing.__class__.__name__}"
                ):
                    post_processing(
                        results=out,
                        rays=batch.rendering.camera.rays.reshape(-1, 6),
                        coords_xy=coords_xy.reshape(-1, 2),
                        unique_frame_idx=batch.data.camera.meta[0].unique_frame_idx,
                        unique_sensor_idx=batch.data.camera.meta[0].unique_sensor_idx,
                        global_step=global_step,
                    )

        # RGB values can saturate, clamp to [0, 1]
        if out.rendered_cam and (out.rendered_cam.rgb is not None) and self.saturate_radiance:
            out.rendered_cam.rgb = out.rendered_cam.rgb.clamp(0.0, 1.0)

        return out

    @ScopedTimer("GaussiansComposite/update_step_train_batch_start")
    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        additional_parameters: dict[str, torch.Tensor] = {}
        additional_parameters |= update_module_step(self.calib, epoch, global_step, system)
        for gaussians in self.gaussians_nodes.values():
            additional_parameters |= update_module_step(gaussians, epoch, global_step, system)
        additional_parameters |= update_module_step(self.background, epoch, global_step, system)
        for post_processing in self.post_processings:
            additional_parameters |= update_module_step(post_processing, epoch, global_step, system)

        return additional_parameters

    @ScopedTimer("GaussiansComposite/update_step_train_batch_end")
    def update_step_train_batch_end(
        self, epoch: int, global_step: int, batch: DataAndRenderingBatch, system, **kwargs
    ) -> None:
        for child in self.get_base_model_children():
            with ScopedTimer(f"{child.__class__.__name__}/update_step_train_batch_end"):
                child.update_step_train_batch_end(epoch, global_step, batch, system, **kwargs)
        for name, node in self.gaussians_nodes.items():
            with ScopedTimer(f"GaussiansComposite/GaussiansNode/{name}/update_step_train_batch_end"):
                node.update_step_train_batch_end(epoch, global_step, batch, system, **kwargs)
        with ScopedTimer(f"{self.gaussians_strategy.__class__.__name__}/update_step_train_batch_end"):
            self.gaussians_strategy.update_step_train_batch_end(
                epoch, global_step, batch, system, self.gaussians_nodes, **kwargs
            )

        total_gaussians = 0
        for name, node in self.gaussians_nodes.items():
            count = node.get_num_gaussians()
            total_gaussians += count
            system.log(
                f"gaussians/{name}_num_gaussians",
                count * self.trainer_config.world_size,
                prog_bar=False,
            )

        system.log(
            "gaussians/num_gaussians",
            total_gaussians * self.trainer_config.world_size,
            prog_bar=self.config.log_to_prog_bar,
        )

    def get_extra_state(self) -> dict[str, Any] | None:
        if self.obj_track_ids is None:
            assert self.obj_track_ids is None, "State requires obj_track_ids"
            return None
        else:
            return {
                "obj_track_ids": {
                    key: value.track_ids for key, value in self.obj_track_ids.items() if not value.is_empty()
                }
            }

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if (state is None) or "obj_track_ids" not in state:
            assert self.obj_track_ids is None, "State contains no obj_track_ids which the model requires"
            return

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        # collect all optimizers of model components in a flat list, if present
        submodule_optims: list[OptimizerLRSchedulerConfig] = functools.reduce(
            operator.iconcat,
            map(
                lambda model: model.configure_optimizers(name_prefix) if model is not None else [],
                (
                    self.calib,
                    self.background,
                    *self.post_processings,
                ),
            ),
            [],
        )

        per_node_optims: list[OptimizerLRSchedulerConfig] = []
        for node_id in self.get_gaussians_node_ids():
            gaussians_node = self.gaussians_nodes[node_id]
            per_node_optims.extend(gaussians_node.configure_optimizers(name_prefix=node_id))

        return submodule_optims + per_node_optims

    def _get_layers_config(self) -> collect.LayersConfig:
        layers_config = []
        for gaussians in self.gaussians_nodes.values():
            layers_config.append(gaussians.get_layer_config())

        def get_unique_value(getter: Callable[[BaseGaussianModel], int], name: str) -> int:
            values = [getter(gaussians) for gaussians in self.gaussians_nodes.values()]
            if len(set(values)) > 1:
                raise ValueError(f"All values in {name} must be the same, got {values}")
            return values[0]

        extra_signal_dim = get_unique_value(
            lambda gaussians: gaussians.config.particle.extra_signal_dim, "extra_signal_dim"
        )
        camera_extra_signal_dim = get_unique_value(
            lambda gaussians: gaussians.config.particle.camera_extra_signal_dim, "camera_extra_signal_dim"
        )
        lidar_extra_signal_dim = get_unique_value(
            lambda gaussians: gaussians.config.particle.lidar_extra_signal_dim, "lidar_extra_signal_dim"
        )

        albedo_sh_dim = get_unique_value(
            lambda gaussians: cast(SHGaussianModel, gaussians).get_albedo_sh_dim(), "features_albedo"
        )
        specular_sh_dim = get_unique_value(
            lambda gaussians: cast(SHGaussianModel, gaussians).get_specular_sh_dim(), "features_specular"
        )

        return collect.LayersConfig(
            layers=layers_config,
            extra_signal_dim=extra_signal_dim,
            camera_extra_signal_dim=camera_extra_signal_dim,
            lidar_extra_signal_dim=lidar_extra_signal_dim,
            albedo_dim=albedo_sh_dim,
            specular_dim=specular_sh_dim,
        )

    def collect_gaussian_parameters(
        self,
        rendering_data: RenderingData,
        is_training_batch: bool = False,
        gaussian_nodes: Optional[list[str]] = None,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> dict[str, torch.Tensor]:
        """Collects gaussian parameters from all nodes"""
        if self.debug_viz:
            visualizer = get_visualdebugger()
            visualizer.set_properties(up="z_up", front="neg_x_front")
        else:
            visualizer = None

        all_node_ids = list(self.gaussians_nodes.keys())
        if gaussian_nodes is None:
            gaussian_nodes = self.get_gaussians_node_ids()
            layer_indices = None
        else:
            existing_node_ids = set(self.get_gaussians_node_ids())
            for gaussian_node in gaussian_nodes:
                assert gaussian_node in existing_node_ids, f"{gaussian_node} not in {existing_node_ids}"
            layer_indices = [all_node_ids.index(node_id) for node_id in gaussian_nodes]

        # Multi-GPU: determine distributed context for visibility slicing
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        visibility_local_indices_parts: list[torch.Tensor] = []

        layers_data = collect.LayersData(
            layers=[],
            frame_timestamp_us=BaseGaussianModel.get_frame_timestamp(rendering_data),
        )
        offsets = []
        offset = 0
        additional_gaussian_parameters = defaultdict(list)
        for node_id in gaussian_nodes:
            gaussians = self.gaussians_nodes[node_id]

            tracks_edit = None
            if isinstance(gaussians, RigidGaussianModel):
                if edited_cuboid_tracks is not None:
                    tracks_edit = gaussians.create_tracks_edit_from_cuboid_tracks(edited_cuboid_tracks)
                if track_transform_ids is not None:
                    cuboid_tracks = (
                        tracks_edit.cuboid_tracks
                        if tracks_edit is not None
                        else gaussians.tracks_calib(gaussians.cuboid_tracks)
                    )
                    mapping = (
                        tracks_edit.mapping
                        if tracks_edit is not None
                        else torch.arange(gaussians.cuboid_tracks.n_tracks, device=gaussians.cuboid_tracks.device)
                    )
                    pose_deltas = se3_matrix_to_tquat(
                        torch.eye(4, device=gaussians.cuboid_tracks.device)
                        .unsqueeze(0)
                        .expand(cuboid_tracks.tracks_data.tracks_poses.shape[0], -1, -1)
                    )
                    for track_transform_id, track_transform in zip(
                        track_transform_ids, unpack_optional(track_transforms)
                    ):
                        if track_transform_id in cuboid_tracks.tracks_id:
                            pack_info = cuboid_tracks.tracks_data.tracks_packinfo[
                                cuboid_tracks.tracks_id.index(track_transform_id)
                            ]
                            pose_deltas[int(pack_info[0]) : int(pack_info[0] + pack_info[1])] = track_transform

                    adjusted_tracks = CuboidTracks.Ops.transform_with_delta_poses(
                        cuboid_tracks, lt.SE3.InitFromVec(pose_deltas), left_multiply=False
                    )
                    tracks_edit = LayerTrackIds.Edit(cuboid_tracks=adjusted_tracks, mapping=mapping)

            with ScopedTimer(f"GaussiansComposite/get_layer_data[{node_id}]"):
                params = gaussians.get_parameters()
                local_count = next(iter(params.values())).shape[0]
                params = distributed_all_gather_gaussian_parameters(params)
                context = BaseGaussianModel.CollectionContext(
                    rendering_data=rendering_data, is_training_batch=is_training_batch, tracks_edit=tracks_edit
                )
                layer_data = gaussians.get_layer_data(context, params)
                layers_data.layers.append(layer_data)

                offsets.append(offset)
                layer_total = layer_data.positions.shape[0]

                # Multi-GPU: compute this rank's indices within the all-gathered layer.
                # evenly_divisible_all_gather produces contiguous blocks [rank0 | rank1 | ...],
                # so rank R's portion starts at sum(counts[:R]) within the layer.
                # Skip if already computed (indices are identical across camera/lidar collections).
                if world_size > 1 and self._visibility_local_indices is None:
                    device = layer_data.positions.device
                    all_counts = all_gather_int32(world_size, local_count, device=device)
                    rank_start = sum(all_counts[:world_rank])
                    visibility_local_indices_parts.append(
                        torch.arange(
                            offset + rank_start,
                            offset + rank_start + local_count,
                            device=device,
                        )
                    )

                offset += layer_total

                for name, value in context.additional_data.items():
                    if value[1]:
                        additional_gaussian_parameters[name].append(value[0])
            offsets.append(offset)

        # Store local-to-global index mapping for multi-GPU visibility/cumulated_weights slicing.
        # This is used by forward_gaussians to slice renderer scene_data back to local rank size.
        # Only set on the first call; subsequent calls (e.g., lidar after camera) reuse the existing mapping.
        if self._visibility_local_indices is None and visibility_local_indices_parts:
            self._visibility_local_indices = torch.cat(visibility_local_indices_parts)

        with ScopedTimer("GaussiansComposite/collect"):
            collector_result = self.collector.collect(layers_data, layer_indices=layer_indices)

        for i, node_id in enumerate(gaussian_nodes):
            gaussians = self.gaussians_nodes[node_id]

            with ScopedTimer(f"GaussiansComposite/post_process[{node_id}]"):
                gaussians.post_process(offsets[i], collector_result, layers_data.layers[i])

        results = {field.name: getattr(collector_result, field.name) for field in fields(collector_result)}
        additional_results = {k: torch.cat(v, 0) for k, v in additional_gaussian_parameters.items()}

        if visualizer is not None:
            for i, node_id in enumerate(gaussian_nodes):
                positions = results["positions"][offsets[i] : offsets[i + 1]]
                features = results["features"][offsets[i] : offsets[i + 1]]
                visualizer.add_point_cloud(
                    f"{node_id}_positions",
                    points=positions.detach().cpu().numpy(),
                    radius=0.0005,
                    colors_quantities={
                        f"{node_id}_color": SH2RGB(features[:, :3].detach().cpu().numpy()),
                    },
                )

            visualizer.show()

        return results | additional_results

    def get_all_gaussian_positions(self) -> torch.Tensor | None:
        """
        Collect all Gaussian center positions from all Gaussian nodes if possible.

        Returns:
            torch.Tensor | None: Concatenated positions of all Gaussians from all nodes, shape (N, 3), or None if not available
        """
        all_positions: list[torch.Tensor] = []
        for gaussian_node in self.gaussians_nodes.values():
            positions = gaussian_node.get_positions()
            all_positions.append(positions.data)

        if not all_positions:
            return None

        return torch.cat(all_positions, dim=0)

    def track_ply_override(
        self,
        track_id: str,
        asset: Asset,
        transform: torch.Tensor,
        dims_offset: torch.Tensor,
    ) -> bool:
        """
        Overrides the gaussians for a track id with gaussians from an asset.
        Note that the override is only for validation and should be reverted by calling
        revert_track_ply_override before resuming training.
        """
        replaced = False
        for gaussian_node in self.gaussians_nodes.values():
            if isinstance(gaussian_node, RigidGaussianModel) or isinstance(gaussian_node, DeformableGaussianModel):
                replaced = (
                    gaussian_node.track_ply_override(
                        track_id, asset=asset, transform=transform, dims_offset=dims_offset
                    )
                    or replaced
                )

        return replaced

    def insert_asset(self, asset: Asset, inserted_cuboid_tracks: CuboidTracks, dims_offset: torch.Tensor) -> bool:
        """
        Insert gaussians associated with a new cuboid track into a RigidGaussianModel.

        Args:
            asset: Asset containing gaussian parameters
            inserted_cuboid_tracks: CuboidTracks data for inserted track, derived from sensorsim proto's EditAssetsRequest.insert
            dims_offset: Offset to apply to cuboid dimensions
        """
        track_inserted = False
        (track_id,) = inserted_cuboid_tracks.tracks_data.tracks_id
        concatenated_cuboid_tracks = CuboidTracks.Ops.concatenate([self.cuboid_tracks, inserted_cuboid_tracks])

        for layer_id, gaussian_node in self.gaussians_nodes.items():
            layer_config = self.obj_track_ids[layer_id].config
            if layer_config.get("is_dynamic", None) is not True:
                continue
            if not isinstance(gaussian_node, RigidGaussianModel):
                continue

            inserted_label_class = inserted_cuboid_tracks.tracks_data.tracks_label_class[0]
            # Accept insert if the layer's config allows this label_class (e.g. "vehicle"), or if
            # the scene already has a track with this label (e.g. "automobile" from dataset).
            label_allowed = False
            if "label_classes" in layer_config and isinstance(layer_config.label_classes, ListConfig):
                label_allowed = inserted_label_class in list(layer_config.label_classes)
            if not label_allowed:
                label_allowed = inserted_label_class in gaussian_node.cuboid_tracks.tracks_data.tracks_label_class
            if not label_allowed:
                log.warning(
                    f"Insertion of {inserted_label_class} for {layer_id} is not allowed. "
                    f"Allowed label classes: {layer_config.get('label_classes', 'N/A')}"
                )
                continue

            new_obj_track_ids = LayerTrackIds(config=self.obj_track_ids[layer_id].config)
            new_obj_track_ids.initialize_from_tracks(concatenated_cuboid_tracks)

            layer_cuboid_tracks = new_obj_track_ids.get_layer_tracks(concatenated_cuboid_tracks)

            if track_id not in layer_cuboid_tracks.tracks_id:
                log.info(f"insert_asset: skipping layer={layer_id} - track {track_id} excluded by layer filter")
                continue

            self.obj_track_ids[layer_id] = new_obj_track_ids
            gaussian_node.update_tracks_calib_and_time_embed(layer_cuboid_tracks, inserted_track_ids={track_id})
            gaussian_node.cuboid_tracks = layer_cuboid_tracks
            track_inserted = (
                gaussian_node.insert_asset(track_id, asset, inserted_cuboid_tracks, dims_offset) or track_inserted
            )

        if track_inserted:
            self.cuboid_tracks = concatenated_cuboid_tracks
            return True

        return False

    @ScopedTimer("GaussiansComposite/remove_background_gaussians")
    def remove_background_gaussians(
        self, track_ids: List[str], track_shift_m: torch.Tensor, track_padding_m: torch.Tensor
    ) -> None:
        """
        Removes background gaussians that fall inside the specified cuboid tracks. This ensure that
        objects do not overlap with the background when modifying the trajectories of these objects.
        """

        trajectories_to_remove: List[torch.Tensor] = []
        dims_to_remove: List[torch.Tensor] = []
        for track_id in track_ids:
            for gaussian_node in self.gaussians_nodes.values():
                if isinstance(gaussian_node, RigidGaussianModel):
                    track_poses_and_dims = gaussian_node.get_track_poses_and_dims(track_id)
                    if track_poses_and_dims is not None:
                        trajectory = track_poses_and_dims[0].matrix()
                        trajectory[:, :3, 3] += track_shift_m

                        cuboid_dims = track_poses_and_dims[1]
                        cuboid_dims_padded = cuboid_dims + track_padding_m

                        trajectories_to_remove.append(trajectory)
                        dims_to_remove.append(cuboid_dims_padded)

        for gaussian_node in self.gaussians_nodes.values():
            if not isinstance(gaussian_node, RigidGaussianModel):
                gaussian_node.remove_gaussians_in_trajectories(trajectories_to_remove, dims_to_remove)

    def save_training_parameters(self) -> None:
        if len(self.overriden_state) == 0:
            self.overriden_state["cuboid_tracks"] = self.cuboid_tracks
            self.overriden_state["obj_track_ids"] = copy.copy(self.obj_track_ids)

        for gaussian_node in self.gaussians_nodes.values():
            gaussian_node.save_training_parameters()

    @ScopedTimer("GaussiansComposite/restore_training_parameters")
    def restore_training_parameters(self) -> None:
        """
        Should be called at the end of the validation phase to revert any changes made by
        track_ply_override, remove_background_gaussians, and grpc asset edits.
        """
        if len(self.overriden_state) > 0:
            self.cuboid_tracks = self.overriden_state["cuboid_tracks"]
            self.obj_track_ids = self.overriden_state["obj_track_ids"]

        for gaussian_node in self.gaussians_nodes.values():
            gaussian_node.restore_training_parameters()

        self.overriden_state = {}

    def export_plys(
        self,
        export_dir: Path,
        format: GaussianExportFormat = GaussianExportFormat._3DGS,
        percentage_gaussians: float = 100,
    ) -> None:
        """
        Exports point clouds (PLY files) in the original 3DGS format or in the 3DGRT format for each node.

        For 3DGS format, it should be compatible with the original 3DGS implementation but differences
        between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
        3rd-party 3DGS viewers.

        Args:
            export_dir (Path): The directory where the exported PLY files will be saved.
            format (GaussianExportFormat, optional): The export format to use.
                - GaussianExportFormat._3DGS: Exports in the original 3D Gaussian Splatting format.
                - GaussianExportFormat._3DGRT: Exports in the 3D Gaussian Ray Tracing format.
                Default is GaussianExportFormat._3DGS.
            percentage_gaussians (float, optional): The percentage (0, 100] of gaussians to export.
                Allows for subsampling the exported gaussians. Default is 100 (all gaussians).

        Returns:
            None

        """
        for key, gaussian_node in self.gaussians_nodes.items():
            if isinstance(gaussian_node, SHGaussianModel):
                gaussian_node.export_ply(export_dir / key, format=format, percentage_gaussians=percentage_gaussians)

    @ScopedTimer("GaussiansComposite/get_updated_cuboid_tracks")
    @torch.no_grad()
    def get_updated_cuboid_tracks(self) -> CuboidTracks:
        """return the updated tracks of every primitives"""
        updated_cuboid_tracks = []
        for key, gaussian_node in self.gaussians_nodes.items():
            if isinstance(gaussian_node, RigidGaussianModel) and not gaussian_node.is_static:
                cuboid_subset = self.obj_track_ids[key].get_layer_tracks(unpack_optional(self.cuboid_tracks))
                updated_cuboid_tracks.append(gaussian_node.tracks_calib(cuboid_subset))

        return (
            CuboidTracks.Ops.concatenate(updated_cuboid_tracks)
            if len(updated_cuboid_tracks) > 0
            else CuboidTracks.Factory.empty()
        )

    def get_number_of_gaussians_per_track(self) -> dict[str, dict[str, int]]:
        """
        Returns the number of gaussians for each node per track.
        """
        return {
            key: gaussian_node.get_number_of_gaussians_per_track()
            for key, gaussian_node in self.gaussians_nodes.items()
        }

    def parse_extra_ray_signals(self, extra_ray_signals: torch.Tensor, lidar_extra_ray_signals: bool) -> ExtraSignal:
        """
        Parse extra ray signals from the renderer output into an ExtraSignal object.

        Args:
            extra_ray_signals: Tensor containing extra signals from renderer, shape [..., total_signal_dims]
            lidar_extra_ray_signals: whether the extra ray signals are for lidar rays

        Returns:
            ExtraSignal: Object containing parsed signals as attributes
        """
        if extra_ray_signals is None or extra_ray_signals.shape[-1] == 0:
            return ExtraSignal()
        else:
            return ExtraSignal.from_packed_tensor(
                extra_ray_signals,
                self.lidar_extra_ray_signal_infos if lidar_extra_ray_signals else self.camera_extra_ray_signal_infos,
            )

    def collect_renderer_profilings(self) -> dict[str, float]:
        """
        Returns:
            dict: A dictionary containing the profiling information from both camera and lidar renderers.
        """
        if self.gaussians_renderer_camera == self.gaussians_renderer_lidar:
            return self.gaussians_renderer_camera.collect_profilings()
        else:
            profiling_camera = self.gaussians_renderer_camera.collect_profilings()
            profiling_lidar = self.gaussians_renderer_lidar.collect_profilings()
            return dict(Counter(profiling_camera) + Counter(profiling_lidar))
