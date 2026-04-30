# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import gc
import glob
import logging
import os

from pathlib import Path
from typing import Any, List, Mapping, Optional, cast

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from einops import rearrange
from point_cloud_utils import TriangleMesh, chamfer_distance
from pytorch_lightning.loggers.wandb import WandbLogger
from scipy.spatial.transform import Rotation
from torchmetrics import Metric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from libs.losses.orchestration.config import LossAggregatorReturn
from libs.nrend.renderer import Renderer  # type: ignore
from ncore.data import ConcreteLidarModelParametersUnion, FrameTimepoint
from nre.config.nre import NREConfig
from nre.config.systems import (
    GaussiansSystemConfig,
    NRendTestGaussiansSystemConfig,
)
from nre.datasets.ncore import NCOREDataSource, NCORESequentialDataset
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks
from nre.difix.training_controller import TrainingDifixController
from nre.metrics import AggregationMethod, MetricFactory, MetricManager, MetricType
from nre.models.background import SkyEnvMapBackground
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.models.gaussians.utils import Asset
from nre.systems.base import BaseSystemSO
from nre.systems.registry import register as register_system
from nre.utils.batch import DataAndRenderingBatch, DataBatch, RenderingData, generate_grid_2d_indices
from nre.utils.geometry import se3_matrix_inverse, se3_matrix_to_tquat
from nre.utils.io.checkpoint import reduce_precision_to_fp16, serialize_checkpoint, strip_optimizer_state
from nre.utils.lidar_post_processing import distance_based_filter
from nre.utils.misc import crop_mask_border, rank_zero_only, to_numpy, to_torch, unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.trainer import adjust_step_for_world_size
from nre.utils.types import (
    Checkpoint,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    NovelViewOverrides,
    PointCloud,
    RayFlags,
)
from nre.utils.visualize import DEFAULT_PS_VIEW_JSON_STR, draw_text_overlay, save_video, scalar2img, scalar2rgb, sem2img
from nre.visualdebugger import VisualDebugger, get_visualdebugger


log = logging.getLogger(__name__)


@register_system("gaussians-system")
class GaussiansSystem(BaseSystemSO):
    config: GaussiansSystemConfig | NRendTestGaussiansSystemConfig
    model: GaussiansComposite
    test_out_types: list[str]

    def __init__(self, config: NREConfig) -> None:
        super().__init__(config)

        self.config.record_timings.train_interval = adjust_step_for_world_size(
            config.trainer, self.config.record_timings.train_interval
        )
        self.config.record_timings.val_interval = adjust_step_for_world_size(
            config.trainer, self.config.record_timings.val_interval
        )
        self.config.record_quality_metrics.psnr_interval = adjust_step_for_world_size(
            config.trainer, self.config.record_quality_metrics.psnr_interval
        )

        log.info(
            f"GaussiansSystem/record_timings: train_interval={self.config.record_timings.train_interval} val_interval={self.config.record_timings.val_interval}"
        )
        log.info(
            f"GaussiansSystem/record_quality_metrics: psnr_interval={self.config.record_quality_metrics.psnr_interval}"
        )

        datasource_summary = DataSourceSummary.from_datasource(self.datamodule.get_datasource())

        self.model = GaussiansComposite(
            config.model,
            config.trainer,
            datasource_summary,
            init_from_datasource="train" in config.mode and not config.resume,
        )
        # Check if we are in the training mode
        is_train = "train" in config.mode

        if is_train:
            # Initialize the loss, semantic loss requires the semantic_classes_map
            self.loss.initialize(self.datamodule.train_dataset)

        # TODO: Clean this up. This is kind of a messy workaround needed for max extent below
        self.max_dist_m: float | None = None
        if config.dataset.name == "ncore":
            self.max_dist_m = config.dataset.max_dist_m

        self._training_difix = TrainingDifixController(self)

        self.track_padding_m = config.dataset.valid_pixels_cuboid_track_params.track_padding_m

    def setup(self, stage: str) -> None:
        # The `setup` function is called by the `pl.Trainer` fit/validate/test functions. At this point,
        # the DDP environment is already initialized, unlike in the `__init__` function. Therefore,
        # we move the initialization of the model to the `setup` function.
        # NOTE: If `init_from_datasource` is not used, this function can be called during construction.
        #       This is done automatically in the `GaussiansComposite` model.
        super().setup(stage)

        self.model.initialize_strategy_and_maybe_gaussians()

        # TODO: Move the metric managers to BaseSystemSO
        # Initialize train metrics
        self.train_metric_manager = MetricManager(device=torch.device("cuda"))
        self.train_metric_manager.register_metric("psnr", MetricFactory[MetricType.PSNR](data_range=1))

        # Initialize validation metrics
        self.val_metric_manager = MetricManager(device=torch.device("cuda"))
        self.val_metric_manager.register_metric("psnr", MetricFactory[MetricType.PSNR](data_range=1))
        self.val_metric_manager.register_metric(
            "cpsnr", MetricFactory[MetricType.CPSNR](data_range=1, aggregation_methods=AggregationMethod.WEIGHTED_MEAN)
        )
        self.val_metric_manager.register_metric("lidar_common", MetricFactory[MetricType.LIDAR_COMMON]())

        self.criterions: dict[str, Metric] = {
            "ssim": StructuralSimilarityIndexMeasure().to("cuda"),
            "lpips": LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda"),
        }

        self._training_difix.maybe_init_difix_model()

    def get_ssim_criterion(self) -> StructuralSimilarityIndexMeasure:
        criterion = self.criterions["ssim"]
        assert isinstance(criterion, StructuralSimilarityIndexMeasure)
        return criterion

    def get_lpips_criterion(self) -> LearnedPerceptualImagePatchSimilarity:
        criterion = self.criterions["lpips"]
        assert isinstance(criterion, LearnedPerceptualImagePatchSimilarity)
        return criterion

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.resume:
            return
        self.model.on_train_from_scratch_start(self)

    def on_train_batch_end(
        self,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_local_idx: int,
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_local_idx)
        with ScopedTimer("GaussiansSystem/model/update_step_train_batch_end"):
            self.model.update_step_train_batch_end(self.current_epoch, self.global_step, batch, self)
        # Memory-based garbage collection
        if self._should_collect_garbage():
            with ScopedTimer("GaussiansSystem/_should_collect_garbage"):
                gc.collect()
                torch.cuda.empty_cache()

    def _should_collect_garbage(self) -> bool:
        """Check if garbage collection should be triggered based on memory usage"""

        if not hasattr(self.config, "collect_garbage_mem_usage") or self.config.collect_garbage_mem_usage is None:
            return False

        # Only check memory every N steps to reduce overhead
        check_interval = getattr(self.config, "collect_garbage_check_interval", 250)
        if self.global_step % check_interval != 0:
            return False

        # Get memory information using PyTorch
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        usage_ratio = used_mem / total_mem

        # Log GPU memory info when we check
        used_gb = used_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        usage_percent = usage_ratio * 100
        log.info(f"GPU Memory: {usage_percent:.1f}% ({used_gb:.1f}GB / {total_gb:.1f}GB)")

        return usage_ratio > self.config.collect_garbage_mem_usage

    def forward(
        self,
        batch: DataAndRenderingBatch,
        is_training_batch: bool = False,
        gaussian_nodes: Optional[list[str]] = None,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        render_background: bool = True,
    ) -> GaussiansCompositeReturn:
        return self.model(
            batch,
            is_training_batch=is_training_batch,
            gaussian_nodes=gaussian_nodes,
            track_transform_ids=track_transform_ids,
            track_transforms=track_transforms,
            render_background=render_background,
            global_step=self.global_step,
        )

    @ScopedTimer("GaussiansSystem/training_losses")
    def training_losses(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> LossAggregatorReturn:
        # Training-time Difix controller
        use_training_difix = self._training_difix.should_use_novel_view()
        maybe_training_batch = None
        if use_training_difix:
            maybe_training_batch = self._training_difix.get_next_novel_batch(batch)
            if maybe_training_batch is not None:
                batch = maybe_training_batch

        # Call the forward pass on all rays and maybe time call
        @ScopedTimer("GaussiansSystem/training_losses/forward_call")
        def forward_call() -> GaussiansCompositeReturn:
            return self(batch, is_training_batch=True)

        out = forward_call()

        # If training-time Difix is enabled and used this step, replace gt with difix target computed from dirty render
        if use_training_difix and maybe_training_batch is not None:
            self._training_difix.apply_difix_as_target(batch, out)

        loss_return = self._training_losses(batch, out, batch_local_idx)

        return loss_return

    @ScopedTimer("GaussiansSystem/training_losses/_training_losses")
    def _training_losses(
        self, batch: DataAndRenderingBatch, out: GaussiansCompositeReturn, batch_local_idx: int
    ) -> LossAggregatorReturn:
        self.apply_alpha(batch, out)

        # Compute the loss
        with ScopedTimer("_training_losses/loss_call"):
            loss_return = self.loss(results=out, target=batch, model=self.model, step=self.global_step)

        with ScopedTimer("_training_losses/loss_logging"):
            loss_return.log(logging_fn=self.log)

        timer = ScopedTimer("_training_losses/misc_logging")
        timer.start()

        data_camera = batch.data.camera
        if data_camera is not None:
            if self.global_step % self.config.record_quality_metrics.psnr_interval == 0:
                with torch.no_grad():
                    rgb_ray_mask = torch.logical_and(
                        data_camera.labels.get_mask_flags_all(RayFlags.RGB_LABEL),
                        data_camera.labels.get_mask_flags_none(RayFlags.INVALID),
                    ).reshape(-1)
                    rendered_rgb = unpack_optional(unpack_optional(out.rendered_cam).rgb)
                    self.train_metric_manager.compute(
                        "psnr",
                        rendered_rgb[rgb_ray_mask],
                        unpack_optional(data_camera.labels.rgb).reshape(rendered_rgb.shape)[
                            torch.argwhere(rgb_ray_mask).squeeze(-1)
                        ],
                    )
                    psnr = self.train_metric_manager.get_last("psnr")
                    if psnr is not None:
                        self.log("train/psnr", psnr["psnr"].cpu(), prog_bar=True)

        assert batch.rendering is not None, "Rendering should not be None at this stage"
        rendering_camera = batch.rendering.camera
        rendering_lidar = batch.rendering.lidar
        total_number_of_rays = (
            rendering_camera.b * rendering_camera.h * rendering_camera.w if rendering_camera is not None else 0
        ) + (rendering_lidar.b * rendering_lidar.h * rendering_lidar.w if rendering_lidar is not None else 0)
        self.log("train/nrays", float(total_number_of_rays), prog_bar=True)

        timer.stop()

        return loss_return

    def get_output_types(self) -> list[str]:
        # Configuration defines what image sequence types (modalities) will be output during validation and test
        out_types = []
        if self.config.test.save_results:
            out_types.extend(["pred_rgb", "pred_distance", "pred_normal", "pred_opacity"])
        if self.config.test.save_inputs:
            out_types.extend(
                [
                    "input_rgb",
                    "input_distance",
                    "input_normal",
                    "input_valid_mask",
                    "input_sky_mask",
                    "input_road_mask",
                    "input_semantic_mask",
                ]
            )
        if self.config.test.save_extra_signals:
            out_types.extend(["pred_semantic", "bg_rgb", "rgb_before_post_processing"])
        return out_types

    def get_max_dist_m(self):
        if self.max_dist_m is not None:
            return self.max_dist_m
        else:
            return self.model.scene_extent

    def get_ncore_semantic_classes_map(self) -> Optional[dict[str, int]]:
        if hasattr(self.datamodule, "val_dataset") and isinstance(self.datamodule.val_dataset, NCORESequentialDataset):
            return self.datamodule.val_dataset.get_datasource().get_semantic_classes_map(
                camera_semantics=True, lidar_semantics=False
            )
        return None

    def on_validation_start(self) -> None:
        super().on_validation_start()

        self.val_suffix = []
        if isinstance(self.datamodule.val_dataset, NCORESequentialDataset):
            if self.datamodule.val_dataset.val_sensor_transl_delta_m is not None:
                self.val_suffix.append(
                    "transl_{}".format(
                        "-".join([str(x) for x in self.datamodule.val_dataset.val_sensor_transl_delta_m])
                    )
                )
            if self.datamodule.val_dataset.val_sensor_rot_delta_deg is not None:
                self.val_suffix.append(
                    "rot_{}".format("-".join([str(x) for x in self.datamodule.val_dataset.val_sensor_rot_delta_deg]))
                )

        self.val_out_types = self.get_output_types()

        self.maybe_override_track_plys()
        self.maybe_remove_background_gaussians()
        self.maybe_setup_visualdebugger()

    def maybe_setup_visualdebugger(self) -> None:
        """
        Setup the visualdebugger for pointcloud renders.
        """
        if self.config.test.lidar.save_renders.enabled:
            self.visualdebugger = get_visualdebugger()
            self.visualdebugger.set_properties(
                program_name="Lidar Points",
                up="z_up",
                front="x_front",
                ground_plane_mode="none",
                length_scale=1.0,
                automatically_compute_scene_extents=False,
                window_size=(1920, 1080),
                window_resizable=False,
                view_projection_mode="perspective",
                json_str=DEFAULT_PS_VIEW_JSON_STR,
            )
            self.visualdebugger.set_camera_extrinsics(
                VisualDebugger.RootLookUp(
                    root=self.config.test.lidar.save_renders.root,
                    look_dir=self.config.test.lidar.save_renders.look_dir,
                    up_dir=self.config.test.lidar.save_renders.up_dir,
                )
            )
            self.visualdebugger.clear()

            os.makedirs(os.path.join(self.val_dir, "pc_renders", "pred"), exist_ok=True)
            os.makedirs(os.path.join(self.val_dir, "pc_renders", "gt"), exist_ok=True)
            self.val_out_types += ["pc_renders"]

        # setup the model's renderer when testing
        if self.config.test.nrend.enabled:
            assert len(self.config.test.val_render_selected_nodes) == 0, (
                "Selective node rendering not supported in NRend"
            )
            self.calibrated_cuboid_tracks = self.model.get_updated_cuboid_tracks()
            self.model.setup_nrend(
                track_ids=self.calibrated_cuboid_tracks.tracks_id,
                rendered_model_json_path=self.config.test.nrend.rendered_model_json_path,
                renderer_hint=Renderer.Hint(self.config.test.nrend.renderer_hint),
                renderer_settings_json_path=self.config.test.nrend.renderer_settings_json_path,
                log_level=Renderer.LogLevel(self.config.test.nrend.log_level),
                profiling_frequency=self.config.test.nrend.profiling_frequency,
            )

        if isinstance(self.model.background, SkyEnvMapBackground):
            self.log_and_inpaint_env_map(self.model.background)

        self.maybe_render_track_orbits()

    def maybe_override_track_plys(self) -> None:
        """
        Overrides the gaussians for specified tracks with gaussians from ply files
        """
        if len(self.config.test.track_ply.overrides) == 0:
            return

        transform = torch.FloatTensor(self.config.test.track_ply.transform).to(self.device)
        for track_id, ply_path in self.config.test.track_ply.overrides.items():
            self.model.track_ply_override(
                track_id,
                Asset(Path(ply_path)),
                transform=transform,
                dims_offset=torch.FloatTensor(self.track_padding_m).to(self.device),
            )

    def maybe_remove_background_gaussians(self) -> None:
        """
        Removes background gaussians that fall inside the specified cuboid tracks. This ensure that
        objects do not overlap with the background when modifying the trajectories of these objects.
        """
        if len(self.config.test.background_removal.track_ids) == 0:
            return

        track_shift_m = torch.FloatTensor(self.config.test.background_removal.track_shift_m).to(self.device)
        track_padding_m = torch.FloatTensor(self.config.test.background_removal.track_padding_m).to(self.device)

        self.model.remove_background_gaussians(
            self.config.test.background_removal.track_ids, track_shift_m, track_padding_m
        )

    def maybe_render_track_orbits(self) -> None:
        """
        Renders around specified tracks in an orbit trajectory
        """

        track_orbit_ids = set(self.config.test.track_orbit.track_ids)
        if len(track_orbit_ids) == 0:
            return

        render_width = self.config.test.track_orbit.render_width
        render_height = self.config.test.track_orbit.render_height
        assert render_width % 2 == 0, (
            "width should be multiple of 2"
        )  # otherwise videos might not render properly in ffmpeg
        assert render_height % 2 == 0, (
            "weight should be multiple of 2"
        )  # otherwise videos might not render properly in ffmpeg
        distance = self.config.test.track_orbit.distance
        n_frames = self.config.test.track_orbit.n_frames

        tracks = self.model.cuboid_tracks
        for track_idx in range(tracks.n_tracks):
            if tracks.tracks_id[track_idx] in track_orbit_ids:
                cur_track = CuboidTracks.Ops.subset_from_tracks_id(tracks, [tracks.tracks_id[track_idx]])

                for view_idx, view in enumerate(
                    self.get_track_orbit_views(
                        cur_track,
                        distance,
                        render_width,
                        render_height,
                        n_frames,
                    )
                ):
                    if len(self.val_out_types):
                        self.save_images(
                            self.val_dir,
                            None,
                            unpack_optional(self.forward(view.to(self.device)).rendered_cam),
                            view_idx,
                            h=self.config.test.track_orbit.render_height,
                            out_types=self.val_out_types,
                            suffix_list=self.val_suffix + [tracks.tracks_id[track_idx]],
                        )

                if self.config.test.save_videos and len(self.val_out_types):
                    self.save_videos(
                        self.val_dir,
                        self.val_out_types,
                        suffix_list=self.val_suffix + [tracks.tracks_id[track_idx]],
                    )

    def validation_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        track_transform_ids: Optional[list[str]] = None
        track_transforms: Optional[torch.Tensor] = None

        batch_idx = unpack_optional(batch.data.idx)
        if len(self.config.test.track_rotation.track_ids) > 0:
            track_transform_ids = self.config.test.track_rotation.track_ids
            track_transforms = (
                torch.eye(4, device=self.device)
                .unsqueeze(0)
                .repeat(len(self.config.test.track_rotation.track_ids), 1, 1)
            )
            track_transforms[:, :3, :3] = torch.FloatTensor(
                Rotation.from_euler(
                    "z",
                    self.config.test.track_rotation.angle_per_frame * batch_idx
                    + np.array(self.config.test.track_rotation.angle_offsets),
                    degrees=True,
                ).as_matrix()
            ).to(self.device)
            track_transforms = se3_matrix_to_tquat(track_transforms)
        else:
            track_transform_ids = None
            track_transforms = None

        if (camera := batch.data.camera) is not None:
            assert camera.b == 1, f"{self.__class__.__name__}: only supports batch size one in validation mode"

        results: GaussiansCompositeReturn
        data_batch_sensor: DataBatch.Camera | DataBatch.Lidar

        if batch.data.camera is not None:
            assert batch.data.lidar is None, "Validation batch should not have both rays_cam and rays_lidar"
            data_batch_sensor = unpack_optional(batch.data.camera)
        else:
            assert batch.data.camera is None, "Validation batch should not have both rays_cam and rays_lidar"
            data_batch_sensor = unpack_optional(batch.data.lidar)

        if self.model.has_nrend():
            assert track_transform_ids is None and track_transforms is None, "Track rotation not supported in NRend"
            rendering_data_sensor: RenderingData

            assert batch.rendering is not None
            if batch.rendering.camera is not None:
                assert batch.rendering.lidar is None, "Validation batch should not have both rays_cam and rays_lidar"
                rendering_data_sensor = batch.rendering.camera
            else:
                assert batch.rendering.camera is None, "Validation batch should not have both rays_cam and rays_lidar"
                rendering_data_sensor = unpack_optional(batch.rendering.lidar)

            # Render all cam rays using NRend and maybe time call
            def render_nrend_call() -> GaussiansCompositeReturn:
                nrend_results = self.model.render_nrend_sensor_rays(
                    batch_idx,
                    rendering_data_sensor,
                    data_batch_sensor.meta[0],
                    self.calibrated_cuboid_tracks,
                )

                return GaussiansCompositeReturn(rendered_cam=nrend_results)

            results = render_nrend_call()

        else:
            # Render all rays using forward and maybe time call
            def forward_call() -> GaussiansCompositeReturn:
                return self(
                    batch,
                    track_transform_ids=track_transform_ids,
                    track_transforms=track_transforms,
                )

            results = forward_call()

        # Apply diffusion fixer
        if self.difix_config.inference.enabled and results.rendered_cam is not None:
            if unpack_optional(results.rendered_cam).rgb is not None:
                camera = unpack_optional(batch.data.camera)
                w, h = unpack_optional(camera.w), unpack_optional(camera.h)
                unpack_optional(results.rendered_cam).rgb = self.difix_model(
                    unpack_optional(results.rendered_cam).rgb,
                    torch.Size([h, w]),
                    self.difix_config.inference.use_color_transfer,
                )
            else:
                raise ValueError("Cannot apply Difix to an empty render.")

        # Update batch depending on model settings (apply alpha to rgb etc.)
        self.apply_alpha(batch, results)
        if results.rendered_cam is not None:
            self.validation_camera_step(batch, results.rendered_cam)
        if results.rendered_lidar is not None:
            self.validation_lidar_step(batch, results.rendered_lidar)

    def validation_camera_step(self, batch: DataAndRenderingBatch, rendered: GaussiansRenderReturn) -> None:
        if (camera := batch.data.camera) is None:
            log.warning(f"No camera data for {unpack_optional(batch.data.idx)}, skipping validation metrics")
            return

        w, h = unpack_optional(camera.w), unpack_optional(camera.h)
        rgb_gt = unpack_optional(camera.labels.rgb).squeeze(0)  # (h, w, 3)
        batch_idx = unpack_optional(batch.data.idx)

        # Used for metrics
        timestamps_startend_us = None
        if batch.rendering is not None and batch.rendering.camera is not None:
            timestamps_startend_us = batch.rendering.camera.timestamps_startend_us

        # only validate valid rays (rays that are *not* marked as invalid)
        valid_rays_mask = camera.labels.get_mask_flags_none(RayFlags.INVALID).squeeze()  # h, w
        if valid_rays_mask.any():
            rendered_rgb = rearrange(rendered.rgb, "(h w) c -> h w c", h=h)
            valid_rgb = rendered_rgb[valid_rays_mask]
            valid_rgb_gt = rgb_gt[valid_rays_mask]

            self.val_metric_manager.compute("psnr", valid_rgb, valid_rgb_gt)
            psnr = self.val_metric_manager.get_last("psnr")
            if psnr is not None:
                self.collect_metric(
                    "test/psnr",
                    psnr["psnr"].cpu(),
                    is_lidar=False,
                    frame_meta=camera.meta[0],
                    timestamps_startend_us=timestamps_startend_us,
                    sequence_id=batch.data.sequence_id,
                )
                self.val_metric_manager.reset("psnr")

            if (
                self.config.test.metrics.cpsnr.enabled
                and (semantic_classes_map := self.get_ncore_semantic_classes_map()) is not None
            ):
                semantic = camera.labels.semantic
                if semantic is not None:
                    self.val_metric_manager.compute(
                        "cpsnr",
                        rendered_rgb,
                        rgb_gt,
                        valid_mask=valid_rays_mask.squeeze(),
                        segmentation_frame=semantic.squeeze(),
                        color_dict=semantic_classes_map,
                        include_categories=self.config.test.metrics.cpsnr.classes,
                    )
                    cpsnr = self.val_metric_manager.get_last("cpsnr")
                    if cpsnr is not None:
                        for class_name, value in cpsnr.values.items():
                            pixel_count = cpsnr.metadata["pixel_counts"][class_name]
                            if pixel_count > 0:
                                self.collect_metric(
                                    f"test/cpsnr/{class_name}",
                                    value,
                                    is_lidar=False,
                                    frame_meta=camera.meta[0],
                                    timestamps_startend_us=timestamps_startend_us,
                                    sequence_id=batch.data.sequence_id,
                                )
                        self.val_metric_manager.reset("cpsnr")

            if self.config.test.metrics.ssim.enabled or self.config.test.metrics.lpips.enabled:
                reshaped_rgb = rearrange(
                    unpack_optional(rendered.rgb).clone(), "(h w) c -> h w c", h=h, w=w
                )  # (h, w, 3)
                reshaped_rgb_gt = rgb_gt.clone()  # (h, w, 3)
                reshaped_rgb_gt = rearrange(reshaped_rgb_gt, "h w c -> 1 c h w", h=h, w=w)
                reshaped_rgb = rearrange(reshaped_rgb, "h w c -> 1 c h w", h=h, w=w)

                datasource = self.datamodule.get_datasource()
                camera_mask_border: list[int] | None = None
                if isinstance(datasource, NCOREDataSource) and datasource.camera_mask_border is not None:
                    camera_id = datasource.camera_ids[camera.meta[0].unique_sensor_idx]
                    camera_mask_border = datasource.camera_mask_border.get(camera_id, None)

                if camera_mask_border is not None:
                    # When a camera_mask_border is defined, the valid part of the rgb and rgb_gt images still form a rectangle, so we can crop the border from the images
                    reshaped_rgb = crop_mask_border(reshaped_rgb, camera_mask_border)
                    reshaped_rgb_gt = crop_mask_border(reshaped_rgb_gt, camera_mask_border)
                else:
                    reshaped_rgb[:, :, ~valid_rays_mask] = reshaped_rgb_gt[:, :, ~valid_rays_mask]

                if self.config.test.metrics.ssim.enabled:
                    ssim = self.get_ssim_criterion()(reshaped_rgb, reshaped_rgb_gt)
                    self.validation_step_outputs["ssim"].append(ssim)
                    self.collect_metric(
                        "test/ssim",
                        ssim,
                        is_lidar=False,
                        frame_meta=camera.meta[0],
                        timestamps_startend_us=timestamps_startend_us,
                        sequence_id=batch.data.sequence_id,
                    )
                    self.get_ssim_criterion().reset()

                if self.config.test.metrics.lpips.enabled:
                    lpips = self.get_lpips_criterion()(reshaped_rgb, reshaped_rgb_gt)
                    self.validation_step_outputs["lpips"].append(lpips)
                    self.collect_metric(
                        "test/lpips",
                        lpips,
                        is_lidar=False,
                        frame_meta=camera.meta[0],
                        timestamps_startend_us=timestamps_startend_us,
                        sequence_id=batch.data.sequence_id,
                    )
                    self.get_lpips_criterion().reset()
        else:
            log.warning(f"No valid rays for {batch_idx}, skipping validation metrics")

        if len(self.val_out_types):
            # save image to disk
            self.save_images(
                self.val_dir,
                batch,
                rendered,
                batch_idx,
                h=h,
                out_types=self.val_out_types,
                suffix_list=self.val_suffix,
            )

            for suffix, gaussian_nodes in self.config.test.val_render_selected_nodes.items():
                node_results: GaussiansCompositeReturn = self(
                    batch,
                    gaussian_nodes=gaussian_nodes,
                    render_background=False,
                )

                # Apply diffusion fixer
                if self.difix_config.inference.enabled:
                    assert node_results.rendered_cam is not None
                    if node_results.rendered_cam.rgb is not None:
                        node_results.rendered_cam.rgb = self.difix_model(
                            node_results.rendered_cam.rgb,
                            torch.Size([h, w]),
                            self.difix_config.inference.use_color_transfer,
                        )
                    else:
                        raise ValueError("Cannot apply Difix to an empty render.")

                self.save_images(
                    self.val_dir,
                    batch,
                    unpack_optional(node_results.rendered_cam),
                    batch_idx,
                    h=h,
                    out_types=self.val_out_types,
                    suffix_list=self.val_suffix + [suffix],
                )

    def validation_lidar_step(
        self,
        batch: DataAndRenderingBatch,
        rendered: GaussiansRenderReturn,
    ) -> None:
        if (data_lidar := batch.data.lidar) is None:
            log.warning(f"No lidar data for {unpack_optional(batch.data.idx)}, skipping validation metrics")
            return
        assert batch.rendering is not None and (rendering_lidar := batch.rendering.lidar) is not None

        # flags are shaped (b, h, w, 1)
        is_valid = data_lidar.labels.get_mask_flags_none(RayFlags.INVALID).squeeze(-1)

        # rendering_lidar has shape (b, h, w, 6)
        rays_lidar = rendering_lidar.rays[is_valid]
        rendered_valid = rendered[is_valid.reshape(-1)]
        height, width = is_valid.shape[1], is_valid.shape[2]
        lidar_elements = generate_grid_2d_indices((width, height), order="yx", device=is_valid.device)
        lidar_valid_elements = lidar_elements[is_valid.reshape(-1)].to(dtype=torch.int32)

        # filter by predicted raydrop mask
        if rendered_valid.extra_ray_signals is not None and rendered_valid.extra_ray_signals.raydrop is not None:
            save_raydrop = True
            did_return_pred = (
                rendered_valid.extra_ray_signals.raydrop.reshape(-1) < self.config.test.lidar.raydrop_threshold
            )
        else:
            save_raydrop = False
            did_return_pred = torch.ones_like(
                unpack_optional(data_lidar.labels.flags)[is_valid], dtype=torch.bool
            ).reshape(-1)

        if self.config.test.lidar.save_filtered_pc.enabled:
            filter_threshold = self.config.test.lidar.save_filtered_pc.filter_threshold
            # remove points if filter_mask equal to True
            filter_mask = distance_based_filter(
                rendered_valid, lidar_valid_elements, did_return_pred, filter_threshold, height, width
            )
            filter_mask_valid = filter_mask[is_valid.reshape(-1)]
            # did_return_pred[filter_mask_valid] = False
            did_return_pred = torch.logical_and(did_return_pred, ~filter_mask_valid)

        # Used for metrics
        timestamps_startend_us = None
        if batch.rendering is not None and batch.rendering.lidar is not None:
            timestamps_startend_us = batch.rendering.lidar.timestamps_startend_us

        self.val_metric_manager.compute(
            "lidar_common", data_lidar, rendering_lidar, rendered, did_return_pred, self.config
        )
        lidar_common = self.val_metric_manager.get_last("lidar_common")

        dist_gt = unpack_optional(data_lidar.labels.distance)[is_valid].reshape(-1)
        did_return_gt = data_lidar.labels.get_mask_flags_none(RayFlags.DROPPED)[is_valid].reshape(-1)
        if self.config.test.lidar.ROI.min_m is not None:
            did_return_gt = torch.logical_and(did_return_gt, dist_gt >= self.config.test.lidar.ROI.min_m)
        if self.config.test.lidar.ROI.max_m is not None:
            did_return_gt = torch.logical_and(did_return_gt, dist_gt <= self.config.test.lidar.ROI.max_m)

        if lidar_common is not None and "raydrop_accuracy" in lidar_common:
            self.collect_metric(
                "test/raydrop_accuracy",
                lidar_common["raydrop_accuracy"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "raydrop_precision" in lidar_common:
            self.collect_metric(
                "test/raydrop_precision",
                lidar_common["raydrop_precision"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "raydrop_recall" in lidar_common:
            self.collect_metric(
                "test/raydrop_recall",
                lidar_common["raydrop_recall"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "raydrop_IoU" in lidar_common:
            self.collect_metric(
                "test/raydrop_IoU",
                lidar_common["raydrop_IoU"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )

        dist_pred = unpack_optional(rendered_valid.distance)

        if lidar_common is not None and "depth_median_l2" in lidar_common:
            self.collect_metric(
                "test/depth_median_l2",
                lidar_common["depth_median_l2"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_mean_rel_l2" in lidar_common:
            self.collect_metric(
                "test/depth_mean_rel_l2",
                lidar_common["depth_mean_rel_l2"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_rmse" in lidar_common:
            self.collect_metric(
                "test/depth_rmse",
                lidar_common["depth_rmse"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_mae" in lidar_common:
            self.collect_metric(
                "test/depth_mae",
                lidar_common["depth_mae"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_medae" in lidar_common:
            self.collect_metric(
                "test/depth_medae",
                lidar_common["depth_medae"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_recall50" in lidar_common:
            self.collect_metric(
                "test/depth_recall50",
                lidar_common["depth_recall50"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )

        xyz_end_pred = torch.addcmul(rays_lidar[:, 0:3], rays_lidar[:, 3:6], dist_pred.unsqueeze(1)).clone()
        xyz_end_gt = torch.addcmul(rays_lidar[:, 0:3], rays_lidar[:, 3:6], dist_gt.unsqueeze(1)).clone()

        if lidar_common is not None and "chamfer_distance" in lidar_common:
            self.collect_metric(
                "test/chamfer_distance",
                lidar_common["chamfer_distance"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )

        intensity_gt = (
            data_lidar.labels.intensity[is_valid].reshape(-1) if data_lidar.labels.intensity is not None else None
        )
        if rendered_valid.extra_ray_signals is not None and rendered_valid.extra_ray_signals.intensity is not None:
            intensity_pred = rendered_valid.extra_ray_signals.intensity.squeeze()

            if lidar_common is not None and "intensity_mae" in lidar_common:
                self.collect_metric(
                    "test/intensity_mae",
                    lidar_common["intensity_mae"].cpu(),
                    is_lidar=True,
                    frame_meta=data_lidar.meta[0],
                    timestamps_startend_us=timestamps_startend_us,
                    sequence_id=batch.data.sequence_id,
                )

            if lidar_common is not None and "intensity_rmse" in lidar_common:
                self.collect_metric(
                    "test/intensity_rmse",
                    lidar_common["intensity_rmse"].cpu(),
                    is_lidar=True,
                    frame_meta=data_lidar.meta[0],
                    timestamps_startend_us=timestamps_startend_us,
                    sequence_id=batch.data.sequence_id,
                )
        else:
            intensity_pred = None

        datasource = self.datamodule.get_datasource()
        assert isinstance(datasource, NCOREDataSource), "Only NCOREDataSource supported."
        lidar_sensor = datasource.lidar_sensors[datasource.lidar_ids[data_lidar.meta[0].unique_sensor_idx]]

        # Note (RDL): This feels a bit hacky as it's likely wrong if we were to have multiple lidars?
        # guess we don't really want unique frame idxs here but per sensor idx
        lidar_frame_index = data_lidar.meta[0].unique_frame_idx

        T_sensor_nre = to_torch(
            datasource.world_to_nre.transform_poses(
                lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index)
            ),
            device=self.device,
        )
        T_nre_sensor = se3_matrix_inverse(T_sensor_nre)

        xyz_start_sensor = (T_nre_sensor[:3, :3] @ rays_lidar[:, 0:3].T + T_nre_sensor[:3, 3:4]).T
        xyz_end_pred_sensor = (T_nre_sensor[:3, :3] @ xyz_end_pred.T + T_nre_sensor[:3, 3:4]).T
        xyz_end_gt_sensor = (T_nre_sensor[:3, :3] @ xyz_end_gt.T + T_nre_sensor[:3, 3:4]).T

        flags = data_lidar.labels.flags.clone()[is_valid].reshape(-1) if data_lidar.labels.flags is not None else None
        pc_gt = PointCloud(
            xyz_start=xyz_start_sensor,
            xyz_end=xyz_end_gt_sensor,
            intensity=intensity_gt,
            flags=flags,
        )
        unpack_optional(pc_gt.flags)[~did_return_gt] |= RayFlags.DROPPED
        pc_pred = PointCloud(
            xyz_start=xyz_start_sensor,
            xyz_end=xyz_end_pred_sensor,
            intensity=intensity_pred,
            flags=flags,
        )
        unpack_optional(pc_pred.flags)[~did_return_pred] |= RayFlags.DROPPED
        batch_idx = unpack_optional(batch.data.idx)
        lidar_model_parameters = cast(ConcreteLidarModelParametersUnion, rendering_lidar.sensor_model_parameters[0])

        self.save_pointcloud(batch_idx, pc_pred, pc_gt)

        self.save_rangeview(batch_idx, pc_pred, pc_gt, lidar_valid_elements, lidar_model_parameters)
        if save_raydrop:
            self.save_raydrop(batch_idx, did_return_pred, did_return_gt, lidar_valid_elements, lidar_model_parameters)
        if self.config.test.lidar.save_renders.enabled:
            self.save_pointcloud_renders(batch, pc_pred, pc_gt)

    def save_pointcloud_renders(self, batch: DataAndRenderingBatch, pc_pred: PointCloud, pc_gt: PointCloud) -> None:
        """
        Save the pointcloud renders to the visualdebugger and save the images to disk.
        """
        idx = unpack_optional(batch.data.idx)

        xyz_end_pred_sensor = pc_pred.xyz_end
        xyz_end_gt_sensor = pc_gt.xyz_end

        self.visualdebugger.clear()
        self.visualdebugger.add_point_cloud(
            "Pred",
            to_numpy(xyz_end_pred_sensor),
            radius=self.config.test.lidar.save_renders.radius,
            color=(0, 0, 1),
        )
        image_name = os.path.join(self.val_dir, "pc_renders", "pred", f"{idx:06d}.png")
        pred_im = self.visualdebugger.screenshot_to_buffer()
        imageio.v2.imsave(image_name, pred_im)

        self.visualdebugger.clear()
        self.visualdebugger.add_point_cloud(
            "GT", to_numpy(xyz_end_gt_sensor), radius=self.config.test.lidar.save_renders.radius, color=(1, 0, 0)
        )
        image_name = os.path.join(self.val_dir, "pc_renders", "gt", f"{idx:06d}.png")
        gt_im = self.visualdebugger.screenshot_to_buffer()
        imageio.v2.imsave(image_name, gt_im)

        if self.config.save_logger:
            for image_name, image_im in [("pred", pred_im), ("gt", gt_im)]:
                image_group = f"pc_renders-{image_name}-rank_{self.global_rank}"
                self.log_image(image_im, image_group, idx)

    def save_pointcloud(self, batch_idx: int, pc_pred: PointCloud, pc_gt: PointCloud) -> None:
        image_name = f"{batch_idx:06d}"
        os.makedirs(out_dir := os.path.join(self.val_dir, "pred_pc"), exist_ok=True)
        image_name = os.path.join(out_dir, image_name)

        pcd_pred = TriangleMesh()
        valid_mask_pred = torch.bitwise_and(unpack_optional(pc_pred.flags), RayFlags.DROPPED).eq(0)
        pcd_pred.vertex_data.positions = pc_pred.xyz_end[valid_mask_pred].cpu().numpy()
        if pc_pred.intensity is not None:
            pcd_pred.vertex_data.colors = scalar2rgb(pc_pred.intensity[valid_mask_pred].cpu().numpy().squeeze())
        pcd_pred.save(image_name + "output.ply")

        pcd_gt = TriangleMesh()
        valid_mask_gt = torch.bitwise_and(unpack_optional(pc_gt.flags), RayFlags.DROPPED).eq(0)
        pcd_gt.vertex_data.positions = pc_gt.xyz_end[valid_mask_gt].cpu().numpy()
        if pc_gt.intensity is not None:
            pcd_gt.vertex_data.colors = scalar2rgb(pc_gt.intensity[valid_mask_gt].cpu().numpy())
        pcd_gt.save(image_name + "output_gt.ply")

    def save_rangeview(
        self,
        batch_idx: int,
        pc_gt: PointCloud,
        pc_pred: PointCloud,
        model_element: torch.Tensor,
        lidar_parameters: ConcreteLidarModelParametersUnion,
    ) -> None:
        n_vertical_bins = lidar_parameters.n_rows
        n_horizontal_bins = lidar_parameters.n_columns

        rangeview_range_gt = torch.full((n_vertical_bins, n_horizontal_bins), torch.inf, device=self.device)
        rangeview_range_pred = torch.full((n_vertical_bins, n_horizontal_bins), torch.inf, device=self.device)

        pc_lidar_gt = pc_gt.xyz_end - pc_gt.xyz_start
        r_gt = pc_lidar_gt.norm(dim=1)
        order = torch.argsort(-r_gt)
        r_gt = r_gt[order]
        pc_lidar_gt = pc_lidar_gt[order]

        pc_lidar_pred = pc_pred.xyz_end - pc_pred.xyz_start
        r_pred = pc_lidar_pred.norm(dim=1)
        valid_mask_pred = torch.bitwise_and(unpack_optional(pc_pred.flags), RayFlags.DROPPED).eq(0)
        r_pred[~valid_mask_pred] = 0

        valid_mask_gt = torch.bitwise_and(unpack_optional(pc_gt.flags), RayFlags.DROPPED).eq(0)
        rangeview_range_gt[model_element[..., 0][valid_mask_gt], model_element[..., 1][valid_mask_gt]] = r_gt[
            valid_mask_gt
        ]

        rangeview_range_pred[model_element[..., 0][valid_mask_pred], model_element[..., 1][valid_mask_pred]] = r_pred[
            valid_mask_pred
        ]

        rangeview_range_pred[rangeview_range_gt == torch.inf] = 0
        rangeview_range_gt[rangeview_range_gt == torch.inf] = 0

        image_name = f"{unpack_optional(batch_idx):06d}"
        os.makedirs(out_dir_rangeview := os.path.join(self.val_dir, "pred_pc"), exist_ok=True)
        image_name = os.path.join(out_dir_rangeview, image_name)

        rangeview_range_gt_numpy = rangeview_range_gt.cpu().numpy()
        im_gt = scalar2img(rangeview_range_gt_numpy, vmin=0.0, vmax=self.get_max_dist_m())
        im_gt[rangeview_range_gt_numpy == 0] = 0
        imageio.v2.imsave(image_name + "_gt.png", im_gt)

        rangeview_range_pred_numpy = rangeview_range_pred.cpu().numpy()
        im_pred = scalar2img(rangeview_range_pred_numpy, vmin=0.0, vmax=self.get_max_dist_m())
        im_pred[rangeview_range_pred_numpy == 0] = 0
        imageio.v2.imsave(image_name + ".png", im_pred)

    def save_raydrop(
        self,
        batch_idx: int,
        raydrop_pred: torch.Tensor,
        raydrop_gt: torch.Tensor,
        model_element: torch.Tensor,
        lidar_parameters: ConcreteLidarModelParametersUnion,
    ) -> None:
        n_vertical_bins = lidar_parameters.n_rows
        n_horizontal_bins = lidar_parameters.n_columns

        rangeview_dropped_gt = torch.ones((n_vertical_bins, n_horizontal_bins), device=self.device, dtype=torch.bool)
        rangeview_dropped_gt[model_element[..., 0][~raydrop_gt], model_element[..., 1][~raydrop_gt]] = False
        rangeview_dropped_pred = torch.ones((n_vertical_bins, n_horizontal_bins), device=self.device, dtype=torch.bool)
        rangeview_dropped_pred[model_element[..., 0][~raydrop_pred], model_element[..., 1][~raydrop_pred]] = False

        os.makedirs(out_dir := os.path.join(self.val_dir, "ray_drop"), exist_ok=True)
        plt.imsave(
            f"{out_dir}/{batch_idx}_gt.png",
            rangeview_dropped_gt.cpu().numpy(),
            cmap="gray",
        )
        plt.imsave(
            f"{out_dir}/{batch_idx}.png",
            rangeview_dropped_pred.cpu().numpy(),
            cmap="gray",
        )

    def on_validation_epoch_end(self) -> None:
        if self.config.test.save_videos and len(self.val_out_types):
            self.save_videos(self.val_dir, self.val_out_types, suffix_list=self.val_suffix)
            for suffix in self.config.test.val_render_selected_nodes:
                self.save_videos(self.val_dir, self.val_out_types, suffix_list=self.val_suffix + [suffix])

        for key in self.validation_step_outputs:
            if len(vals := self.validation_step_outputs[key]) > 0:
                self.log(f"test/{key}", torch.stack(vals).mean(), prog_bar=True)
                vals.clear()  # free memory

        if len(self.val_metric_manager.get_metric("psnr")) > 0:
            aggregated_psnr = self.val_metric_manager.aggregate("psnr")["psnr"]
            for method in aggregated_psnr:
                self.log(f"test/psnr/{method.name.lower()}", aggregated_psnr[method]["psnr"], prog_bar=True)
            self.val_metric_manager.clear("psnr")  # free memory

        if (
            self.config.test.metrics.cpsnr.enabled
            and (semantic_classes_map := self.get_ncore_semantic_classes_map()) is not None
        ):
            aggregated_cpsnr = self.val_metric_manager.aggregate("cpsnr")["cpsnr"]
            for method in aggregated_cpsnr:
                for class_name in semantic_classes_map.keys():
                    if class_name in aggregated_cpsnr[method].values:
                        value = aggregated_cpsnr[method][class_name]
                        self.log(
                            f"test/cpsnr/{method.name.lower()}/{class_name}",
                            value,
                            prog_bar=True,
                        )
            self.val_metric_manager.clear("cpsnr")  # free memory

        if isinstance(self.model.background, SkyEnvMapBackground):
            self.model.background.restore_original_textures()

        self.model.restore_training_parameters()

    def on_test_start(self) -> None:
        super().on_test_start()

        if isinstance(self.model.background, SkyEnvMapBackground):
            self.log_and_inpaint_env_map(self.model.background)

    def test_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        if len(self.test_out_types):
            batch_idx = unpack_optional(batch.data.idx)

            results: GaussiansCompositeReturn = self(batch)

            if batch.data.camera is not None:
                self.save_images(
                    self.test_dir,
                    batch,
                    unpack_optional(results.rendered_cam),
                    batch_idx,
                    h=unpack_optional(batch.data.camera.h),
                    out_types=self.test_out_types,
                    suffix_list=self.val_suffix,
                )

    def on_test_epoch_end(self):
        if isinstance(self.model.background, SkyEnvMapBackground):
            self.model.background.restore_original_textures()

    def serialize_artifact_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Set the precision and loading type for the artifacts checkpoint.

        For Gaussian systems we:
            - do not cast the checkpoint to fp16
            - set the load_in_place flag to false to enable enable dealing with dynamic number of Gaussians.
        """

        # Most checkpoints can be loaded in place, but if the dimensions of the buffers are not know in advance, they need to be assigned to the uninitialized tesnors instead.
        checkpoint["load_in_place"] = False

        ckpt = (
            strip_optimizer_state(checkpoint)
            if self.artifact_cache.artifact_config.checkpoint.strip_optimizer
            else checkpoint
        )
        if self.artifact_cache.artifact_config.checkpoint.fp16:
            ckpt = reduce_precision_to_fp16(ckpt)
        self.artifact_cache.checkpoint = serialize_checkpoint(ckpt)  # TODO update when is_resumable is available

    def get_camera_and_frame_export_names(
        self, camera: DataBatch.Camera, datasource: NCOREDataSource, batch_idx: int
    ) -> tuple[str, str]:
        """Compute the output directory name and frame filename stem for a validation/test frame.

        Returns (cam_dir, image_name) based on the configured naming scheme.
        Only called when camera data is available (not for synthetic views like track_orbit).
        """
        unique_sensor_idx = camera.meta[0].unique_sensor_idx
        camera_id = datasource.all_camera_ids[unique_sensor_idx]
        cam_dir = camera_id if self.config.test.use_camera_name_dirs else f"cam_{unique_sensor_idx:02d}"
        match self.config.test.frame_naming:
            case "frame-end-timestamp":
                unique_frame_idx = camera.meta[0].unique_frame_idx
                sensor_frame_idx = (
                    unique_frame_idx
                    - datasource.camera_linear_start_frame_indices[camera_id]
                    + datasource.camera_frame_ranges[camera_id].start
                )
                frame_end_ts = datasource.camera_sensors[camera_id].get_frame_timestamp_us(
                    sensor_frame_idx, frame_timepoint=FrameTimepoint.END
                )
                image_name = f"{frame_end_ts}"
            case "batch-index":
                image_name = f"{batch_idx:06d}"
            case _:
                raise ValueError(f"Invalid frame naming scheme: {self.config.test.frame_naming}")
        return cam_dir, image_name

    def save_images(
        self,
        save_dir: str,
        batch: Optional[DataAndRenderingBatch],
        rendered: GaussiansRenderReturn,
        batch_idx: int,
        h: int,
        out_types: list[str],
        suffix_list: list[str],
        overlay_text: Optional[str] = None,
    ) -> None:
        camera = batch.data.camera if batch is not None else None
        datasource = self.datamodule.get_datasource()
        if camera is not None:
            assert isinstance(datasource, NCOREDataSource), "Only NCOREDataSource supported."
            cam_dir, image_name = self.get_camera_and_frame_export_names(camera, datasource, batch_idx)
        else:
            # Fallback for synthetic views (e.g. track_orbit) where batch/camera are not available.
            # Hardcode cam_00 and use a sequential index.
            cam_dir = "cam_00"
            image_name = f"{batch_idx:06d}"

        for out_type in out_types:
            im = None

            match out_type:
                case "pred_rgb":
                    if rendered.rgb is not None:
                        im = rearrange(rendered.rgb.clamp(0, 1).cpu().numpy(), "(h w) c -> h w c", h=h)
                        im = (im * 255).astype(np.uint8)
                        if overlay_text:
                            draw_text_overlay(im, overlay_text)

                case "pred_distance":
                    im = scalar2img(
                        rearrange(rendered.distance.cpu().numpy(), "(h w) -> h w", h=h),
                        vmin=0.0,
                        vmax=self.get_max_dist_m(),
                    )

                case "pred_normal":
                    if rendered.normal is not None:
                        im = rearrange(rendered.normal.cpu().numpy(), "(h w) c -> h w c", h=h)
                        im = (im + 1) / 2 * 255
                        im = im.astype(np.uint8)

                case "pred_opacity":
                    im = rendered.opacity.cpu().numpy().reshape((h, -1))
                    im = (im * 255).astype(np.uint8)

                case "pred_semantic":
                    if (
                        rendered.extra_ray_signals is not None
                        and rendered.extra_ray_signals.semantic_logits is not None
                    ):
                        dim = rendered.extra_ray_signals.semantic_logits.shape[-1]

                        im = rearrange(
                            rendered.extra_ray_signals.semantic_logits.reshape((-1, dim)).argmax(1).cpu().numpy(),
                            "(h w) -> h w",
                            h=h,
                        )

                        im = sem2img(
                            im,
                            color_remap=self.datamodule.val_dataset.get_datasource().get_semantic_colormap(
                                camera_semantics=True, lidar_semantics=False
                            ),
                        )

                case "input_rgb":
                    if camera is not None:
                        im = unpack_optional(camera.labels.rgb).cpu().numpy().squeeze()
                        im = (im * 255).astype(np.uint8)

                case "input_distance":
                    if camera is not None and camera.labels.metric_distance is not None:
                        im = scalar2img(
                            camera.labels.metric_distance.cpu().numpy().squeeze(),
                            vmin=0.0,
                            vmax=self.get_max_dist_m(),
                        )

                case "input_normal":
                    if camera is not None and camera.labels.normals is not None:
                        im = camera.labels.normals.cpu().numpy().squeeze()
                        im = (im + 1) / 2 * 255
                        im = im.astype(np.uint8)

                case "input_valid_mask":
                    if camera is not None:
                        im = camera.labels.get_mask_flags_none(RayFlags.INVALID).cpu().numpy().reshape((h, -1))
                        im = (im * 255).astype(np.uint8)

                case "input_sky_mask":
                    if camera is not None and isinstance(datasource, NCOREDataSource) and datasource.aux_data:
                        im = camera.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC).cpu().numpy().reshape((h, -1))
                        im = (im * 255).astype(np.uint8)

                case "input_road_mask":
                    if camera is not None and isinstance(datasource, NCOREDataSource) and datasource.aux_data:
                        im = camera.labels.get_mask_flags_all(RayFlags.ROAD_SEMANTIC).cpu().numpy().reshape((h, -1))
                        im = (im * 255).astype(np.uint8)

                case "input_semantic_mask":
                    if camera is not None:
                        semantic_colormap = datasource.get_semantic_colormap(
                            camera_semantics=True, lidar_semantics=False
                        )
                        semantic = camera.labels.semantic
                        if semantic_colormap is not None and semantic is not None:
                            semantic_class_id = semantic.cpu().numpy().reshape((h, -1))
                            im = semantic_colormap[semantic_class_id]

                case "bg_rgb":
                    if rendered.extra_ray_signals is not None and rendered.extra_ray_signals.rgb_background is not None:
                        rgb_background = rendered.extra_ray_signals.rgb_background * (1.0 - rendered.opacity[..., None])
                        im = rearrange(rgb_background.cpu().numpy(), "(h w) c -> h w c", h=h)
                        im = (im * 255).astype(np.uint8)

                case "rgb_before_post_processing":
                    if (
                        rendered.extra_ray_signals is not None
                        and rendered.extra_ray_signals.rgb_before_post_processing is not None
                    ):
                        im = rearrange(
                            rendered.extra_ray_signals.rgb_before_post_processing.cpu().numpy(), "(h w) c -> h w c", h=h
                        )
                        im = (im * 255).astype(np.uint8)

                case "pc_renders":
                    # saved in save_pointcloud_renders
                    continue

                case _:
                    raise ValueError(f"{self.__class__.__name__}: Unsupported output name")

            if not isinstance(im, np.ndarray):
                continue

            out_type_name = out_type
            if len(suffix_list) > 0:
                out_type_name = out_type_name + "-" + "-".join(suffix_list)

            os.makedirs(out_dir := os.path.join(save_dir, out_type_name, cam_dir), exist_ok=True)
            imageio.v2.imsave(os.path.join(out_dir, image_name + ".png"), im)

            if self.config.save_logger:
                # log image sequences as batch-numerated '<image-type>(-suffix)-cam_<cam-idx>-step_<global-step>-rank_<rank>' metrics in UIs
                image_group = (
                    out_type_name
                    + "-cam_"
                    + cam_dir
                    + "-step_"
                    + str(self.global_step)
                    + "-rank_"
                    + str(self.global_rank)
                )

                self.log_image(im, image_group, batch_idx)

    @rank_zero_only
    def save_videos(self, save_dir: str, out_seq_types: list[str], suffix_list: list[str]) -> None:
        # Only rank 0 should create videos in distributed training/validation
        for out_seq_type in out_seq_types:
            out_type_name = out_seq_type
            if len(suffix_list) > 0:
                out_type_name = out_type_name + "-" + "-".join(suffix_list)

            if not os.path.exists(out_type_dir := os.path.join(save_dir, out_type_name)):
                continue
            for cam_dir in os.listdir(out_type_dir):
                video_file = os.path.join(save_dir, out_type_name + "-" + cam_dir + ".mp4")
                im_files = sorted(glob.glob(os.path.join(save_dir, out_type_name, cam_dir, "*.png")))

                # Loads all images at once - can run out of memory if too many frames
                ims: list[np.ndarray] = [imageio.v2.imread(f) for f in im_files]

                save_video(video_file, ims, fps=self.config.test.video_fps)

                if self.config.save_logger and isinstance(self.logger, WandbLogger):
                    # log video sequences as '<video-type>(-suffix)_video-cam_<cam-idx>-step_<global-step>-rank_<rank>' metrics in UIs
                    self.logger.experiment.log(
                        {
                            out_type_name
                            + "_video-"
                            + cam_dir
                            + "-step_"
                            + str(self.global_step)
                            + "-rank_"
                            + str(self.global_rank): wandb.Video(video_file)
                        }
                    )

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        # The sensor models in calib aren't stored in the state_dict, let's keep on using the one that
        # we have created in the constructor.

        # TODO: ideally this code should be in the load_state_dict of the model that
        # contains the sensor models (CameraFreePoseViewGeometry or LidarFreePoseViewGeometry)
        # and somehow called from here (through GaussiansComposite and BaseCalib)

        updated_state_dict = dict(state_dict)

        if hasattr(self.model, "calib"):
            calib_sensor_models = {
                f"model.calib.{k}": v for k, v in self.model.calib.state_dict().items() if ".sensor_models." in k
            }
            updated_state_dict.update(calib_sensor_models)

        return super().load_state_dict(updated_state_dict, *args, **kwargs)
