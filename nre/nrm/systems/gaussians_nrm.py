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

import gc
import logging
import os

from pathlib import Path
from typing import Literal, cast

import imageio
import numpy as np
import torch

from einops import rearrange
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm

import ncore.impl.common.transformations as ncore_transformations

from libs.losses.orchestration.config import LossAggregatorBatchReturn, LossAggregatorReturn
from nre.datasets.tracks import CuboidTracks
from nre.nrm.config.nrm import NRMConfig
from nre.nrm.models.base import BaseNRMSupervisionPack
from nre.nrm.models.kelvin_backbone.base import KelvinNRMSupervisionPack
from nre.nrm.models.kelvin_model import KelvinNRM
from nre.nrm.predict.export_ply import export_ply
from nre.nrm.predict.primitive_merge import make as make_primitive_merge
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.nrm.systems.base import BaseNRMSystem
from nre.nrm.utils.trajectory import SensorOverride, pad_rig_timestamps
from nre.render.render import PoseRange, RenderableModel
from nre.utils.batch import DataAndRenderingBatch, NRMDataBatch
from nre.utils.custom_metrics import AbsRelError, Delta1Accuracy
from nre.utils.geometry import se3_matrix_to_tquat
from nre.utils.misc import unpack_optional
from nre.utils.types import (
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    RayFlags,
    RigTrajectories,
)
from nre.utils.visualize import flow2img, make_image_grid, scalar2img


logger = logging.getLogger(__name__)


class GaussiansNRMSystem(BaseNRMSystem):
    def __init__(self, config: NRMConfig) -> None:
        super().__init__(config)

        if config.model.name == "kelvin":
            self.model = KelvinNRM(config.model)
        else:
            raise ValueError(f"Unknown config name {config.model.name}.")

    def setup(self, stage: str) -> None:
        # We need separate Metrics class for proper reset/sync logic.
        # Use dist_sync_on_step=True for per-step logging across all devices
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.validation_psnr = PeakSignalNoiseRatio(data_range=1)
        self.train_distance_delta1 = Delta1Accuracy()
        self.validation_distance_delta1 = Delta1Accuracy()
        self.train_distance_absrel = AbsRelError(max_rel_error=5.0)
        self.validation_distance_absrel = AbsRelError(max_rel_error=5.0)
        self.train_context_distance_delta1 = Delta1Accuracy()
        self.validation_context_distance_delta1 = Delta1Accuracy()
        self.train_context_distance_absrel = AbsRelError(max_rel_error=5.0)
        self.validation_context_distance_absrel = AbsRelError(max_rel_error=5.0)
        self.train_context_flow_epe = MeanMetric()
        self.validation_context_flow_epe = MeanMetric()

    def on_train_start(self) -> None:
        if self.resume:
            return
        self.model.on_train_from_scratch_start(self)

    def forward(
        self, batch: NRMDataBatch
    ) -> tuple[list[BaseNRMPrimitive], list[BaseNRMSupervisionPack], list[GaussiansCompositeReturn]]:
        cuboid_tracks = None
        if batch.cuboid_tracks is not None:
            cuboid_tracks = [CuboidTracks.Factory.from_pack(ct) for ct in batch.cuboid_tracks]

        # Prepare context for model inference
        batch.context = self.model.prepare_context(batch.context, cuboid_tracks)

        # Reconstruct the primitives
        primitives, supervision_packs = self.model.reconstruct(
            batch.context,
            cuboid_tracks,
            media_logger=self.media_logger,
            compute_supervision_pack=batch.supervision is not None,
        )

        # If no supervision is provided, we don't need to render at those poses.
        if batch.supervision is None:
            return primitives, [], []

        # Prepare supervision signals
        batch.supervision, supervision_packs = self.model.prepare_supervision(
            batch.context, batch.supervision, cuboid_tracks, unpack_optional(supervision_packs), self.media_logger
        )

        # If no rendering is needed, we don't need to forward the primitives.
        if self.global_step < self.config.enable_render_global_step:
            return primitives, supervision_packs, [GaussiansCompositeReturn() for _ in primitives]

        # Forward the primitives
        traced_returns: list[GaussiansCompositeReturn] = []
        for primitive, supervision in zip(primitives, unpack_optional(batch.supervision)):
            assert supervision.rendering is not None, "Rendering must be provided"
            assert supervision.data is not None, "Data must be provided"
            traced_returns.append(
                primitive.forward(
                    rendering_cam_data=supervision.rendering.camera,
                    frames_cam_meta=supervision.data.camera.meta if supervision.data.camera is not None else None,
                    rendering_lidar_data=supervision.rendering.lidar,
                    frames_lidar_meta=supervision.data.lidar.meta if supervision.data.lidar is not None else None,
                )
            )

        return primitives, supervision_packs, traced_returns

    @staticmethod
    def _get_context_distance_pred_gt(
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Return predicted and GT context distance (metric distance) and valid mask.
        For Celsius: pred is context_distance (already distance).
        For Kelvin: pred is context_depth / scale (depth -> distance).
        GT is camera_labels.metric_distance.
        Returns (pred_distance, gt_distance, mask) full-frame, or (None, None, None).
        """
        if context.data is None or context.data.camera is None or context.rendering is None:
            return None, None, None
        camera_labels = context.data.camera.labels
        gt_distance = camera_labels.metric_distance if camera_labels is not None else None
        if gt_distance is None:
            return None, None, None
        pred_distance: torch.Tensor | None = None
        if isinstance(supervision_pack, KelvinNRMSupervisionPack):
            if supervision_pack.context_depth is not None and context.rendering is not None:
                scale = unpack_optional(context.rendering.camera).distance_to_depth_scale
                pred_distance = supervision_pack.context_depth / scale
        if pred_distance is None:
            return None, None, None
        mask = (gt_distance > 0) & (gt_distance < 500.0) & camera_labels.get_mask_flags_none(RayFlags.INVALID)
        if not mask.any():
            return None, None, None
        return pred_distance, gt_distance, mask

    @staticmethod
    def _get_context_flow_pred_gt(
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Extract predicted and GT context flow for all motion targets.
        """
        if not isinstance(supervision_pack, KelvinNRMSupervisionPack):
            return [], []
        # `motion_supervisions` is introduced by the motion head in a later MR in
        # this split stack; tolerate its absence so validation logging stays a no-op
        # until that MR lands.
        motion_supervisions = getattr(supervision_pack, "motion_supervisions", None)
        if not motion_supervisions:
            return [], []
        return [ms.context_flow for ms in motion_supervisions], [
            unpack_optional(ms.reference_flow) for ms in motion_supervisions
        ]

    @staticmethod
    def _rendered_to_distance_img(
        rendered_distance: torch.Tensor | np.ndarray,
        rendered_opacity: torch.Tensor | np.ndarray,
        img_width: int,
        img_height: int,
        start_n_rays: int = 0,
    ) -> np.ndarray:
        img_n_pixels = img_width * img_height
        # Expected distance with opacity mask
        rendered_distance = rendered_distance[start_n_rays : start_n_rays + img_n_pixels]
        rendered_opacity = rendered_opacity[start_n_rays : start_n_rays + img_n_pixels]
        if isinstance(rendered_distance, torch.Tensor):
            rendered_distance = rendered_distance.cpu().numpy()
        if isinstance(rendered_opacity, torch.Tensor):
            rendered_opacity = rendered_opacity.cpu().numpy()
        # Compute inverse expected distance, weighted by opacity (ZipNeRF visualization formulation)
        expected_distance = (rendered_opacity / np.clip(rendered_distance, a_min=1.0e-6, a_max=None)) * rendered_opacity
        return scalar2img(rearrange(expected_distance, "(h w) -> h w", h=img_height), vmin=0.0, vmax=0.5)

    def _extract_render_timestamps_us(self, rig_trajectories: RigTrajectories) -> tuple[int, int]:
        min_timestamps_us: int = 2**63 - 1
        max_timestamps_us: int = 0
        for context_single_rig in rig_trajectories.rig_trajectories:
            for camera_timestamps_us in context_single_rig.cameras_frame_timestamps_us.values():
                min_timestamps_us = min(min_timestamps_us, int(camera_timestamps_us.min().item()))
                max_timestamps_us = max(max_timestamps_us, int(camera_timestamps_us.max().item()))

        return min_timestamps_us, max_timestamps_us

    def render_rig_trajectories_video(
        self,
        camera_unique_id: str,
        primitive: BaseNRMPrimitive,
        supervision_rig: RigTrajectories,
        timestamps_us_range: tuple[int, int],
        fps: int,
        sensor_override: SensorOverride | None = None,
    ):
        # Extend rig trajectory to cover the render timestamps (use constant padding).
        supervision_rig = pad_rig_timestamps(supervision_rig, *timestamps_us_range)

        # Prepare the renderable model.
        renderable_model = RenderableModel.load_from_nrm_primitive(primitive, supervision_rig.world_to_nre)

        # Prepare rendering timestamps
        render_timestamps_us = np.arange(*timestamps_us_range, 1_000_000 // fps)
        # If too few frames (e.g. timestamps_us is actually img index), just sample a 1s video
        if len(render_timestamps_us) < fps:
            render_timestamps_us = np.linspace(*timestamps_us_range, num=fps, dtype=np.int64)

        camera_calibration = supervision_rig.camera_calibrations[camera_unique_id]
        matched_trajectories = [
            traj for traj in supervision_rig.rig_trajectories if camera_calibration.sequence_id == traj.sequence_id
        ]
        assert len(matched_trajectories) == 1, "Expected exactly one trajectory for each camera calibration."
        trajectory = matched_trajectories[0]

        # Interpolate the rig trajectory to get poses from the rendering timestamps.
        interpolator = ncore_transformations.PoseInterpolator(
            trajectory.T_rig_worlds.cpu().numpy(), trajectory.T_rig_world_timestamps_us.cpu().numpy().astype(np.int64)
        )
        T_rig_world = (
            torch.from_numpy(interpolator.interpolate_to_timestamps(render_timestamps_us)).float().to(self.device)
        )  # [t, 4, 4]
        with torch.autocast(device_type="cuda", enabled=False):
            T_sensor_world = T_rig_world @ camera_calibration.T_sensor_rig
            if sensor_override is not None:
                T_sensor_world = sensor_override.apply_sensor_to_world(T_sensor_world)
            camera_world_tquat = se3_matrix_to_tquat(T_sensor_world)

        parameters = camera_calibration.camera_model_parameters
        if sensor_override is not None:
            parameters = sensor_override.apply_camera_model_parameters(parameters)
        img_width, img_height = parameters.resolution.tolist()

        # Yield the rendered frames when ready
        for frame_tquat, render_timestamp_us in zip(camera_world_tquat, render_timestamps_us):
            with torch.inference_mode():
                yield renderable_model.render_camera_frame(
                    parameters,
                    PoseRange(
                        frame_tquat,
                        frame_tquat,
                        render_timestamp_us,
                        render_timestamp_us,
                    ),
                    (img_width, img_height),
                    camera_calibration.unique_sensor_idx,
                    unique_frame_idx=0,  # Dummy data since we don't have this
                    fields=["color_image", "distance_image", "opacity_image"],
                )

    def log_camera_video(
        self,
        prefix: str,
        camera_unique_id: str,
        primitive: BaseNRMPrimitive,
        rig: RigTrajectories,
        timestamps_us_range: tuple[int, int],
        fps: int,
        sensor_override: SensorOverride | None = None,
    ) -> None:
        ims: list[np.ndarray] = []
        for rendered_camera_frame in self.render_rig_trajectories_video(
            camera_unique_id, primitive, rig, timestamps_us_range, fps, sensor_override
        ):
            im_pd_rgb = rendered_camera_frame.color_image.cpu().numpy()
            im_pd_rgb = np.clip(im_pd_rgb * 255, 0, 255).astype(np.uint8)
            im_pd_distance = self._rendered_to_distance_img(
                rendered_camera_frame.distance_image.reshape(-1),
                rendered_camera_frame.opacity_image.reshape(-1),
                im_pd_rgb.shape[1],
                im_pd_rgb.shape[0],
            )
            ims.append(np.concatenate([im_pd_rgb, im_pd_distance], axis=0))
        self.media_logger.log_video(f"{prefix}-{camera_unique_id}", np.stack(ims, axis=0), fps)

    @torch.no_grad()
    def log_prediction_and_groundtruth(
        self, prediction: GaussiansCompositeReturn, groundtruth: DataAndRenderingBatch, n_views: int = 3
    ):
        if prediction.rendered_cam is None:
            return

        running_n_rays: int = 0
        im_pd_rgbs: list[np.ndarray] = []
        im_pd_distances: list[np.ndarray] = []
        im_pd_opacities: list[np.ndarray] = []
        im_pd_velocities: list[np.ndarray] = []
        im_pd_normals: list[np.ndarray] = []
        im_gt_rgbs: list[np.ndarray] = []
        im_gt_distances: list[np.ndarray] = []
        im_gt_velocities: list[np.ndarray] = []
        im_gt_normals: list[np.ndarray] = []

        assert groundtruth.rendering is not None and groundtruth.rendering.camera is not None, (
            "Rendering must be provided"
        )
        img_height = unpack_optional(groundtruth.rendering.camera.h)
        img_width = unpack_optional(groundtruth.rendering.camera.w)
        for batch_idx in range(groundtruth.rendering.camera.b):
            img_n_pixels = img_height * img_width
            assert prediction.rendered_cam.rgb is not None
            im_pd_rgb = rearrange(
                prediction.rendered_cam.rgb[running_n_rays : running_n_rays + img_n_pixels].float().cpu().numpy(),
                "(h w) c -> h w c",
                h=img_height,
            )
            im_pd_rgb = (im_pd_rgb * 255).astype(np.uint8)
            im_pd_rgbs.append(im_pd_rgb)

            rendered_opacity = (
                prediction.rendered_cam.opacity[running_n_rays : running_n_rays + img_n_pixels].float().cpu().numpy()
            )
            im_pd_opacity = rendered_opacity.reshape((img_height, img_width))
            im_pd_opacity = (im_pd_opacity * 255).astype(np.uint8)
            im_pd_opacities.append(im_pd_opacity)

            im_pd_distances.append(
                self._rendered_to_distance_img(
                    prediction.rendered_cam.distance,
                    prediction.rendered_cam.opacity,
                    img_width,
                    img_height,
                    start_n_rays=running_n_rays,
                )
            )

            if (extra_signal := prediction.rendered_cam.extra_ray_signals) is not None and (
                rendered_velocity := extra_signal.velocity
            ) is not None:
                # Take only X,Z direction (Y is up/down so don't care)
                im_pd_velocities.append(
                    rearrange(
                        rendered_velocity[running_n_rays : running_n_rays + img_n_pixels].float().cpu().numpy(),
                        "(h w) c -> h w c",
                        h=img_height,
                    )[..., [0, 2]]
                )

            if (rendered_normals := prediction.rendered_cam.normal) is not None:
                im_pd_normal = rearrange(
                    rendered_normals[running_n_rays : running_n_rays + img_n_pixels].float().cpu().numpy(),
                    "(h w) c -> h w c",
                    h=img_height,
                )
                im_pd_normals.append(((im_pd_normal + 1) / 2 * 255).astype(np.uint8))

            assert groundtruth.data is not None and groundtruth.data.camera is not None, "Data must be provided"
            gt_camera_img = groundtruth.data.camera.labels[batch_idx]
            im_gt_rgb = unpack_optional(gt_camera_img.rgb).cpu().numpy()
            im_gt_mask = gt_camera_img.get_mask_flags_none(RayFlags.INVALID).squeeze(-1).cpu().numpy()
            im_gt_sky_mask = gt_camera_img.get_mask_flags_all(RayFlags.SKY_SEMANTIC).squeeze(-1).cpu().numpy()
            im_gt_vehicle_mask = gt_camera_img.get_mask_flags_all(RayFlags.VEHICLE_SEMANTIC).squeeze(-1).cpu().numpy()
            im_gt_rgb_vis = im_gt_rgb.copy()
            im_gt_rgb_vis[~im_gt_mask] *= 0.2
            im_gt_rgb_vis[im_gt_sky_mask, 2] = 0.8 + 0.2 * im_gt_rgb_vis[im_gt_sky_mask, 2]
            im_gt_rgb_vis[im_gt_vehicle_mask] = 0.8 + 0.2 * im_gt_rgb_vis[im_gt_vehicle_mask]
            im_gt_rgb_vis = im_gt_rgb_vis.squeeze(0)  # [height, width, D]
            im_gt_rgbs.append((im_gt_rgb_vis * 255).astype(np.uint8))

            if (gt_metric_distance := gt_camera_img.metric_distance) is not None:
                metric_distance = gt_metric_distance.squeeze(0).cpu().numpy()  # [height, width, 1]
                metric_distance[metric_distance > 0.0] = 1.0 / metric_distance[metric_distance > 0.0]
                im_gt_distance = scalar2img(metric_distance.squeeze(-1), vmin=0.0, vmax=0.5)
                im_gt_distances.append(im_gt_distance // 2 + (im_gt_rgb.squeeze(0) * 127).astype(np.uint8))

            if (gt_velocity := gt_camera_img.velocity) is not None:
                im_gt_velocities.append(gt_velocity.squeeze(0).cpu().numpy()[..., [0, 2]])

            if (gt_normals := gt_camera_img.normals) is not None:
                im_gt_normals.append(((gt_normals.squeeze(0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8))

            running_n_rays += img_n_pixels

        n_times = len(im_pd_rgbs) // n_views
        subsample = self.media_logger.log_media_subsample
        self.media_logger.log_image(
            "Rendered RGB",
            make_image_grid(im_gt_rgbs + im_pd_rgbs, grid_width=n_times, subsample=subsample),
        )
        self.media_logger.log_image(
            "Rendered Distance",
            make_image_grid(im_gt_distances + im_pd_distances, grid_width=n_times, subsample=subsample),
        )

        im_pd_alpha: np.ndarray = make_image_grid(im_pd_opacities, grid_width=n_times, subsample=subsample)
        self.media_logger.log_image("Rendered Alpha", im_pd_alpha)
        if len(im_velocities := (im_pd_velocities + im_gt_velocities)) > 0:
            self.media_logger.log_image(
                "Rendered Velocity",
                flow2img(make_image_grid(im_velocities, grid_width=n_times, subsample=subsample), rad_max=None),
            )

        if len(im_normals := (im_pd_normals + im_gt_normals)) > 0:
            self.media_logger.log_image(
                "Rendered World Normals",
                make_image_grid(im_normals, grid_width=n_times, subsample=subsample),
            )

    @torch.no_grad()
    def log_context_distance_visualization(
        self,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        n_views: int,
    ) -> None:
        """Log context distance (GT, prediction) and absolute relative error."""
        pred_distance, gt_distance, mask = self._get_context_distance_pred_gt(supervision_pack, context)
        if pred_distance is None or gt_distance is None or mask is None:
            return
        gt_dist_np = gt_distance.float().cpu().numpy()
        pred_dist_np = pred_distance.float().cpu().numpy()
        valid = mask.cpu().numpy()
        im_gt_list = [
            scalar2img(1.0 / gt_dist_np[b].squeeze(-1), vmin=0.0, vmax=0.5) for b in range(gt_dist_np.shape[0])
        ]
        im_pd_list = [
            scalar2img(1.0 / pred_dist_np[b].squeeze(-1), vmin=0.0, vmax=0.5) for b in range(pred_dist_np.shape[0])
        ]
        absrel = np.zeros_like(gt_dist_np)
        absrel[valid] = np.abs(pred_dist_np[valid] - gt_dist_np[valid]) / np.clip(gt_dist_np[valid], 1e-6, None)
        im_err_list = [
            scalar2img(absrel[b].squeeze(-1), vmin=0.0, vmax=0.5, cmap="hot") for b in range(absrel.shape[0])
        ]
        n_times = gt_dist_np.shape[0] // n_views
        subsample = self.media_logger.log_media_subsample
        self.media_logger.log_image(
            "Context Distance",
            make_image_grid(im_gt_list + im_pd_list + im_err_list, grid_width=n_times, subsample=subsample),
        )

    @torch.no_grad()
    def log_context_flow_visualization(
        self,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        n_views: int,
    ) -> None:
        """Log context flow: GT ref, prediction (XZ color), and endpoint-error heatmap per motion target."""
        pred_flows, gt_flows = self._get_context_flow_pred_gt(supervision_pack, context)
        if not pred_flows:
            return
        subsample = self.media_logger.log_media_subsample
        for flow_idx, (pred_flow, gt_flow) in enumerate(zip(pred_flows, gt_flows)):
            pred_np, ref_np = pred_flow.float().cpu().numpy(), gt_flow.float().cpu().numpy()
            pred_np_masked = np.copy(pred_np)
            pred_np_masked[ref_np == 0] = 0
            n_times = ref_np.shape[0] // n_views
            self.media_logger.log_image(
                f"Context Flow {flow_idx}",
                np.concatenate(
                    [
                        flow2img(
                            make_image_grid(
                                [ref_np[i][..., [0, 2]] for i in range(ref_np.shape[0])]
                                + [pred_np[i][..., [0, 2]] for i in range(pred_np.shape[0])],
                                grid_width=n_times,
                                subsample=subsample,
                            ),
                            rad_max=None,
                        ),
                        scalar2img(
                            make_image_grid(
                                list(np.linalg.norm(pred_np_masked - ref_np, axis=-1)),
                                grid_width=n_times,
                                subsample=subsample,
                            ),
                            vmin=0.0,
                            vmax=2.0,
                            cmap="hot",
                        ),
                    ],
                    axis=0,
                ),
            )

    def system_forward_step(
        self, batch: NRMDataBatch, batch_local_idx: int, mode: Literal["train", "val", "test"]
    ) -> LossAggregatorBatchReturn:
        # Compute rendering data if not done in the dataloader.
        batch.maybe_compute_rendering_data(device=self.device)

        # Forward pass (handles both chunked and non-chunked inference)
        primitives_list, supervision_packs_list, out_list = self.forward(batch)

        # Compute the loss & metric for each primitive
        loss_returns: list[LossAggregatorReturn] = []
        gt_rgbs: list[torch.Tensor] = []
        pd_rgbs: list[torch.Tensor] = []
        gt_distances: list[torch.Tensor] = []
        pd_distances: list[torch.Tensor] = []
        gt_context_distances: list[torch.Tensor] = []
        pd_context_distances: list[torch.Tensor] = []
        context_flow_epes: list[torch.Tensor] = []
        flow_valid_masks: list[torch.Tensor] = []

        out: GaussiansCompositeReturn
        primitive: BaseNRMPrimitive
        supervision: DataAndRenderingBatch
        context: DataAndRenderingBatch
        supervision_rig: RigTrajectories
        context_rig: RigTrajectories
        for bidx, (out, primitive, supervision_pack, supervision, context, supervision_rig, context_rig) in enumerate(
            zip(
                out_list,
                primitives_list,
                supervision_packs_list,
                unpack_optional(batch.supervision),
                unpack_optional(batch.context),
                unpack_optional(batch.supervision_rig),
                unpack_optional(batch.context_rig),
            )
        ):
            loss_return = self.loss(
                step=self.global_step,
                model=self.model,  # type: ignore[arg-type]
                results=out,
                target=supervision,
                primitive=primitive,
                context=context,
                supervision_pack=supervision_pack,
            )
            loss_returns.append(loss_return)

            if bidx == 0 and self.media_logger.should_log_media:
                # Log image only for the first batch element in the first rank (img grid width should be num_views)
                num_supervision_views = len(supervision_rig.camera_calibrations)
                num_context_views = len(context_rig.camera_calibrations)
                self.log_prediction_and_groundtruth(out, supervision, num_supervision_views)
                self.log_context_distance_visualization(supervision_pack, context, num_context_views)
                self.log_context_flow_visualization(supervision_pack, context, num_context_views)
                if mode != "train":
                    timestamps_us_range = self._extract_render_timestamps_us(context_rig)
                    base_rig = (
                        context_rig
                        if self.config.log_rig_trajectories_video == "context"
                        else supervision_rig
                        if self.config.log_rig_trajectories_video == "supervision"
                        else None
                    )
                    if base_rig is not None:
                        for camera_unique_id in base_rig.camera_calibrations.keys():
                            self.log_camera_video("pd", camera_unique_id, primitive, base_rig, timestamps_us_range, 30)
                    for nv_idx, override_config in enumerate(self.config.log_novel_view_overrides):
                        if override_config.sensor_id not in (base_rig or context_rig).camera_calibrations:
                            logger.warning(
                                "Camera %s from log_novel_view_overrides not found in rig, skipping",
                                override_config.sensor_id,
                            )
                            continue
                        sensor_override = SensorOverride.from_config(override_config)
                        self.log_camera_video(
                            f"nv-{nv_idx}",
                            override_config.sensor_id,
                            primitive,
                            base_rig or context_rig,
                            timestamps_us_range,
                            30,
                            sensor_override,
                        )

            if out.rendered_cam is not None:
                with torch.no_grad():
                    assert out.rendered_cam.rgb is not None
                    assert supervision.data is not None and supervision.data.camera is not None, "Data must be provided"
                    assert supervision.rendering is not None and supervision.rendering.camera is not None, (
                        "Rendering must be provided"
                    )
                    if supervision.data.camera.labels is not None and supervision.rendering.camera.rays is not None:
                        rgb_ray_mask = torch.logical_and(
                            supervision.data.camera.labels.get_mask_flags_all(RayFlags.RGB_LABEL),
                            supervision.data.camera.labels.get_mask_flags_none(RayFlags.INVALID),
                        ).squeeze(-1)  # [batch, height, width]
                        pd_rgbs.append(out.rendered_cam.rgb[rgb_ray_mask.flatten()])
                        gt_rgbs.append(unpack_optional(supervision.data.camera.labels.rgb)[rgb_ray_mask])

                        # Add distance metric calculation
                        assert supervision.data.camera.labels is not None, "Camera labels must be provided"
                        if supervision.data.camera.labels.metric_distance is not None:
                            gt_distance = unpack_optional(supervision.data.camera.labels.metric_distance)
                            pred_distance = out.rendered_cam.distance
                            pred_opacity = out.rendered_cam.opacity
                            pred_distance = pred_distance / pred_opacity.clamp(min=1.0e-6)
                            distance_ray_mask = (
                                (gt_distance > 0)
                                & (gt_distance < 500)
                                & supervision.data.camera.labels.get_mask_flags_none(RayFlags.INVALID)
                            )
                            pd_distances.append(pred_distance[distance_ray_mask.flatten()])
                            gt_distances.append(gt_distance[distance_ray_mask])

            pred_dist, gt_dist, mask = self._get_context_distance_pred_gt(supervision_pack, context)
            if pred_dist is not None and gt_dist is not None and mask is not None:
                pd_context_distances.append(pred_dist[mask].flatten())
                gt_context_distances.append(gt_dist[mask].flatten())

            pred_flows, gt_flows = self._get_context_flow_pred_gt(supervision_pack, context)
            for pd_flow, gt_flow in zip(pred_flows, gt_flows):
                context_flow_epes.append(torch.linalg.norm(pd_flow - gt_flow, dim=-1).flatten())
                flow_valid_masks.append(torch.linalg.norm(gt_flow, dim=-1).flatten() > 0)

        if pd_rgbs and gt_rgbs:
            psnr_metric = self.train_psnr if mode == "train" else self.validation_psnr
            psnr_metric(torch.cat(pd_rgbs, dim=0), torch.cat(gt_rgbs, dim=0))
            # This would not trigger sync_dist of the metric during training (when on_step=True), PL's log function will
            # reuse the _forward_cache that is available on this rank.
            # During validation, the metric will sync automatically (when compute() is called).
            self.log(f"{mode}/psnr", psnr_metric, prog_bar=True)
        elif mode == "val":
            # When enable_render_global_step has not been reached, no rendering is done so PSNR is not computed.
            # Log a placeholder so ModelCheckpoint(monitor='val/psnr') can find the key and avoid MisconfigurationException.
            # This placeholder is a small value that increases with epoch to ensure last ckpt is saved.
            self.log(f"{mode}/psnr", 0.1 * self.trainer.current_epoch, prog_bar=True)

        if pd_distances and gt_distances:
            # Concatenate all distance predictions and ground truth
            pred_distances_tensor = torch.cat(pd_distances, dim=0)
            gt_distances_tensor = torch.cat(gt_distances, dim=0)

            # Compute delta1 metric using the Delta1Accuracy class
            delta1_metric = self.train_distance_delta1 if mode == "train" else self.validation_distance_delta1
            delta1_metric(pred_distances_tensor, gt_distances_tensor)
            self.log(f"{mode}/distance_delta1", delta1_metric, prog_bar=True)

            # Compute absolute relative error
            absrel_metric = self.train_distance_absrel if mode == "train" else self.validation_distance_absrel
            absrel_metric(pred_distances_tensor, gt_distances_tensor)
            self.log(f"{mode}/distance_absrel", absrel_metric, prog_bar=True)

        if pd_context_distances and gt_context_distances:
            pred_ctx_tensor = torch.cat(pd_context_distances, dim=0)
            gt_ctx_tensor = torch.cat(gt_context_distances, dim=0)
            ctx_delta1_metric = (
                self.train_context_distance_delta1 if mode == "train" else self.validation_context_distance_delta1
            )
            ctx_delta1_metric(pred_ctx_tensor, gt_ctx_tensor)
            self.log(f"{mode}/context_distance_delta1", ctx_delta1_metric, prog_bar=True)
            ctx_absrel_metric = (
                self.train_context_distance_absrel if mode == "train" else self.validation_context_distance_absrel
            )
            ctx_absrel_metric(pred_ctx_tensor, gt_ctx_tensor)
            self.log(f"{mode}/context_distance_absrel", ctx_absrel_metric, prog_bar=True)

        if context_flow_epes:
            flow_epe_tensor = torch.cat(context_flow_epes, dim=0)
            flow_epe_tensor = flow_epe_tensor[torch.cat(flow_valid_masks, dim=0) > 0]
            if flow_epe_tensor.numel() > 0:
                flow_epe_metric = self.train_context_flow_epe if mode == "train" else self.validation_context_flow_epe
                flow_epe_metric(flow_epe_tensor)
                self.log(f"{mode}/context_flow_masked_epe", flow_epe_metric, prog_bar=True)

        batch_loss_return = LossAggregatorBatchReturn(batch_loss_returns=loss_returns)
        # For validation mode where we want a metric across all ranks we need to sync for every step.
        batch_loss_return.log(logging_fn=self.log, sync_dist=mode != "train")

        # Also return the reconstructed model (e.g. to be used in a viewer callback)
        batch_loss_return.extra_fields["primitives"] = primitives_list

        return batch_loss_return

    def training_losses(self, batch: NRMDataBatch, batch_local_idx: int) -> LossAggregatorBatchReturn:
        return self.system_forward_step(batch, batch_local_idx, mode="train")

    # Return values packed as dict to support visualization + compatible with base pl.LightningModule

    def validation_step(self, batch: NRMDataBatch, batch_local_idx: int) -> dict[str, LossAggregatorBatchReturn]:
        return {"result": self.system_forward_step(batch, batch_local_idx, mode="val")}

    def test_step(self, batch: NRMDataBatch, batch_local_idx: int) -> dict[str, LossAggregatorBatchReturn]:
        return {"result": self.system_forward_step(batch, batch_local_idx, mode="test")}

    def predict_step(
        self, batch: NRMDataBatch, batch_local_idx: int
    ) -> dict[str, list[BaseNRMPrimitive] | NRMDataBatch]:
        # In the future maybe rendering data is not required any more for model forwarding.
        batch.maybe_compute_rendering_data(device=self.device)

        # Note that we don't guarantee that batch.supervision is not None here.

        # For large batch sizes, process in chunks
        primitives_list: list[BaseNRMPrimitive] = []

        inner_batch_idx: int = 0
        progress_bar = tqdm(total=len(batch), desc="Predicting in chunks")
        while inner_batch_idx < len(batch):
            batch_chunk = batch[inner_batch_idx : inner_batch_idx + self.predict_config.chunk_size]
            primitives_chunk_list, _, _ = self.forward(batch_chunk)
            export_preprocess = self.cached_config.model.export_preprocess
            if export_preprocess.enabled:
                context_rig_list = batch_chunk.context_rig if batch_chunk.context_rig is not None else None
                for i in range(len(primitives_chunk_list)):
                    context_rig_i = context_rig_list[i] if context_rig_list is not None else None
                    primitives_chunk_list[i] = primitives_chunk_list[i].preprocess_for_export(
                        batch_chunk.context[i], export_preprocess, context_rig=context_rig_i
                    )
            primitives_list.extend(primitives_chunk_list)
            inner_batch_idx += self.predict_config.chunk_size
            progress_bar.update(self.predict_config.chunk_size)
        progress_bar.close()

        # Merge the primitives if enabled
        if self.predict_config.primitive_merge.enabled:
            primitive_merge = make_primitive_merge(
                type(primitives_list[0]),
                self.predict_config.primitive_merge,
                self.cached_config.model.export_preprocess,
            )
            merged_primitive, batch = primitive_merge.merge_primitives_and_batch(primitives_list, batch)
            primitives_list = [merged_primitive]

        # Release memory if possible
        gc.collect()
        torch.cuda.empty_cache()

        return {"primitives": primitives_list, "batch": batch}

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)

        # Ensure outputs are not None and contain the required keys
        assert outputs is not None and "primitives" in outputs and "batch" in outputs

        out_batch: NRMDataBatch = outputs["batch"]
        primitives_list: list[BaseNRMPrimitive] = outputs["primitives"]
        n_chunks = len(primitives_list)
        assert len(out_batch) == n_chunks, "batch context length must match number of primitives"

        if out_batch.meta is None or out_batch.context_rig is None:
            return

        # Helper to export PLY for one chunk. Standalone build does not export
        # USDZ artifacts and does not render videos (those code paths were
        # removed in Phase 1 step 4.3).
        def export_chunk(primitive: BaseNRMPrimitive, rig: RigTrajectories, meta: dict, chunk_suffix: str) -> None:
            if self.predict_config.export_ply.enabled:
                path = os.path.join(
                    self.out_dir,
                    self.run_id,
                    "ply",
                    meta["sequence_id"],
                    meta["sequence_id"] + chunk_suffix + ".ply",
                )
                export_ply(
                    config=self.predict_config.export_ply,
                    primitives=primitive,
                    rig_trajectories=rig,
                    path=Path(path),
                )

        for chunk_idx in range(n_chunks):
            meta = out_batch.meta[chunk_idx]
            assert "sequence_id" in meta, f"sequence_id key must be provided, only got {meta.keys()}"
            chunk_suffix = "" if self.predict_config.primitive_merge.enabled else f"_chunk{chunk_idx}"
            export_chunk(
                primitives_list[chunk_idx],
                out_batch.context_rig[chunk_idx],
                meta,
                chunk_suffix,
            )

