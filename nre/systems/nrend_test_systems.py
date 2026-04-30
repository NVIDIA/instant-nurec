# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import warnings

from typing import Any, Mapping, Optional, cast

import torch

from libs.nrend.renderer_test_case import RendererTestCase  # type: ignore
from nre.config.systems import NRendTestGaussiansSystemConfig
from nre.config.version import get_version
from nre.datasets.tracks import CuboidTracks
from nre.models.base import BaseModel
from nre.models.nrenderable import NRenderableModel
from nre.systems.gaussians import GaussiansSystem
from nre.systems.registry import register as register_system
from nre.utils.batch import DataAndRenderingBatch, FrameMeta, RenderingData
from nre.utils.misc import unpack_optional
from nre.utils.types import (
    ConcreteCameraModelParametersUnion,
    ExtraSignal,
    GaussiansCompositeReturn,
    RayFlags,
)


class NRendTestBase:
    def setup(self, validation_step_outputs: dict):
        self.prof_start = torch.cuda.Event(enable_timing=True)
        self.prof_end = torch.cuda.Event(enable_timing=True)
        self.profilings: dict[str, list[float]] = {"total_inference": []}

    def nrend_render_test_case(
        self,
        model: BaseModel,
        default_frame_idx: int,
        rendering_data: RenderingData,
        frame_meta: FrameMeta,
        tracks: Optional[CuboidTracks],
        rays_radiance: Optional[torch.Tensor],
        rays_density: torch.Tensor,
        rays_distance: torch.Tensor,
        extra_ray_signals: Optional[ExtraSignal],
        save_dir: str,
        lidar: bool,
        update: bool = False,
    ):
        assert isinstance(model, NRenderableModel) and model.has_nrend()
        device = rendering_data.rays.device

        # Ideally the calibration should not be part of the model / factored out (see #176): input rays should be provided already with calibration applied
        # TODO: skip for now until we add back the calibration
        # if (calib := getattr(model, "calib")) is not None:
        #     assert isinstance(calib, BaseCalib)
        #     rays_cam_calib, rays_cam_meta_calib, frames_cam_meta_calib, _, _, _ = calib(
        #         rays_cam, rays_cam_meta, frames_cam_meta, None, None, None
        #     )

        #     # The results are not none because the inputs aren't. Need to assert so mypy doesn't complain.
        #     assert rays_cam_calib is not None
        #     assert rays_cam_meta_calib is not None
        #     rays_cam = rays_cam_calib
        #     rays_cam_meta = rays_cam_meta_calib
        #     frames_cam_meta = frames_cam_meta_calib
        frame_idx = default_frame_idx  # alternatively : rays_cam_meta.unique_frame_idx
        ray_origins, ray_directions = torch.split(rendering_data.rays[0], [3, 3], dim=-1)
        ray_origins = ray_origins.contiguous()
        ray_directions = ray_directions.contiguous()
        if rendering_data._rays_footprints is not None:
            ray_directions = ray_directions * rendering_data._rays_footprints[0]
        rays_timestamps_us = (
            rendering_data.rays_timestamps_us[0] if rendering_data.rays_timestamps_us is not None else None
        )
        frame_start_timestamp = frame_idx if rays_timestamps_us is None else int(torch.min(rays_timestamps_us))
        frame_end_timestamp = frame_idx if rays_timestamps_us is None else int(torch.max(rays_timestamps_us))

        if (rays_timestamps_us is not None) and (tracks is not None):
            track_instances_uid = tracks.tracks_id
            (
                num_active_track_instances,
                active_track_instances_ids,
                active_track_instances_start_pose,
                active_track_instances_end_pose,
            ) = tracks.frame_poses_interpolation(
                torch.tensor([frame_start_timestamp, frame_end_timestamp], device=device, dtype=torch.int64)
            )
        else:
            track_instances_uid = []
            num_active_track_instances = torch.zeros((1,), dtype=torch.int32)
            active_track_instances_ids = torch.empty((0,), dtype=torch.int32)
            active_track_instances_start_pose = torch.empty((0, 7), dtype=torch.float32)
            active_track_instances_end_pose = torch.empty((0, 7), dtype=torch.float32)

        if rays_radiance is None:
            warnings.warn("rays_radiance is None.")
            rays_radiance = torch.zeros(
                (rays_density.shape[0], 3), device=rays_density.device, dtype=rays_density.dtype
            )

        assert isinstance(model, NRenderableModel), "NRend support only NRenderable models"
        test_case = RendererTestCase(
            model=model.serialize_to_json_dict(),
            renderer=model.renderer_settings(),
            track_instances_uid=track_instances_uid,
            frame_id=frame_idx,
            frame_width=rendering_data.w,
            frame_height=rendering_data.h,
            frame_start_timestamp=frame_start_timestamp,
            frame_end_timestamp=frame_end_timestamp,
            rays_origin=ray_origins,
            rays_direction=ray_directions,
            rays_timestamp=rays_timestamps_us,
            frames_sensor_model=cast(ConcreteCameraModelParametersUnion, rendering_data.sensor_model_parameters[0]),
            frames_sensor_ids=torch.stack(
                [
                    torch.tensor(frame_meta.unique_sensor_idx, device=device),
                    torch.tensor(frame_meta.unique_frame_idx, device=device),
                ],
                dim=0,
            ),
            frames_sensor_start_pose=rendering_data.poses_tquat_startend[0][0, :]
            if rendering_data.poses_tquat_startend[0] is not None
            else None,
            frames_sensor_end_pose=rendering_data.poses_tquat_startend[0][1, :]
            if rendering_data.poses_tquat_startend[0] is not None
            else None,
            num_active_track_instances=int(num_active_track_instances[0]),
            active_track_instances_ids=active_track_instances_ids,
            active_track_instances_start_pose=active_track_instances_start_pose,
            active_track_instances_end_pose=active_track_instances_end_pose,
            rays_radiance_density=torch.concat([rays_radiance, torch.unsqueeze(rays_density, 1)], dim=1),
            rays_hit_distance=rays_distance,
            rays_hit_normal=None,
            extra_ray_signals=extra_ray_signals.concatenated() if extra_ray_signals is not None else None,
            device=ray_origins.device,
        )

        sensor_idx: int = frame_meta.unique_sensor_idx
        sensor_dir = f"{'lidar' if lidar else 'cam'}_{sensor_idx:02d}"
        os.makedirs(out_dir := os.path.join(save_dir, "nrend_test_cases", sensor_dir), exist_ok=True)

        image_name = f"{frame_idx:06d}"
        test_case_path = os.path.join(
            out_dir, image_name + "." + unpack_optional(get_version()).semantic_string() + ".msgpack"
        )

        if update:
            test_case.update(test_case_path)
            test_case.run()
        else:
            test_case.write(test_case_path)


@register_system("nrend-test-gaussians-system")
class NRendTestGaussiansSystem(GaussiansSystem, NRendTestBase):
    config: NRendTestGaussiansSystemConfig

    def setup(self, stage):
        GaussiansSystem.setup(self, stage)
        NRendTestBase.setup(self, self.validation_step_outputs)

    def on_train_batch_end(self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int):
        super().on_train_batch_end(outputs, batch, batch_idx)
        profilings = self.model.collect_renderer_profilings()
        for name, value in profilings.items():
            self.log(name=f"train/profiling/{name}", value=value, prog_bar=False)

    def on_validation_start(self) -> None:
        super().on_validation_start()

    def validation_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        if batch.rendering is None:
            assert hasattr(self.model, "calib"), "Model must have .calib to create rendering data"
            batch.rendering = self.model.calib(batch.data, skip_calib=True)

        if batch.data.camera is not None:
            self.validation_step_camera(batch)
        if batch.data.lidar is not None:
            self.validation_step_lidar(batch)

    def validation_step_camera(self, batch: DataAndRenderingBatch) -> None:
        assert unpack_optional(batch.data.camera).b == 1, (
            f"{self.__class__.__name__}: only supports batch size one in validation mode"
        )

        batch_idx = unpack_optional(batch.data.idx)
        # Use calibrated cuboid tracks if available, else None
        tracks_for_render = getattr(self, "calibrated_cuboid_tracks", None)

        # if supported, render the batch using the model renderer
        results: GaussiansCompositeReturn
        if isinstance(self.model, NRenderableModel) and self.model.has_nrend():
            assert batch.rendering is not None, "Camera validation batch should have rendering data"
            assert batch.rendering.camera is not None, "Camera validation batch should have camera rendering data"

            self.prof_start.record()
            nrend_results = self.model.render_nrend_sensor_rays(
                batch_idx,
                batch.rendering.camera,
                unpack_optional(batch.data.camera).meta[0],
                tracks_for_render,
            )
            results = GaussiansCompositeReturn(rendered_cam=nrend_results)
            self.prof_end.record()
        else:
            self.prof_start.record()
            results = self(batch)
            self.prof_end.record()

        self.prof_end.synchronize()
        self.profilings["total_inference"].append(self.prof_start.elapsed_time(self.prof_end))

        # collecting nrend profilings
        if (
            isinstance(self.model, NRenderableModel)
            and self.model.has_nrend()
            and (self.config.test.nrend.profiling_frequency > 0.0)
        ):
            nrend_profilings = self.model.nrend_profilings()
            for tag, ms in nrend_profilings.items():
                tag_key = f"nrend{tag}"
                if tag_key not in self.profilings:
                    self.profilings[tag_key] = []
                self.profilings[tag_key].append(ms)
        else:
            nrend_profilings = self.model.collect_renderer_profilings()
            for tag, ms in nrend_profilings.items():
                tag_key = f"nrend{tag}"
                if tag_key not in self.profilings:
                    self.profilings[tag_key] = []
                self.profilings[tag_key].append(ms)

        if self.config.test.nrend.create_test_case:
            valid_rendering_results = unpack_optional(results.rendered_cam)
            self.nrend_render_test_case(
                self.model,
                batch_idx,
                unpack_optional(unpack_optional(batch.rendering).camera),
                unpack_optional(batch.data.camera).meta[0],
                tracks_for_render,
                valid_rendering_results.rgb,
                valid_rendering_results.opacity,
                valid_rendering_results.distance,
                valid_rendering_results.extra_ray_signals,
                self.val_dir,
                lidar=False,
                update=self.config.test.nrend.create_test_case_update,
            )

        self.apply_alpha(batch, results)

        rgb_gt = unpack_optional(
            unpack_optional(batch.data.camera).labels.rgb
        )  # we expect rgb reference values to be available

        # only validate valid rays (rays that are *not* marked as invalid)
        valid_rays_mask = torch.bitwise_and(
            unpack_optional(unpack_optional(batch.data.camera).labels.flags), RayFlags.INVALID
        ).eq(0)
        valid_rgb = unpack_optional(results.rendered_cam).rgb
        if valid_rgb is not None:
            valid_rgb_gt = rgb_gt.view(valid_rgb.shape)[valid_rays_mask.flatten()]
            valid_rgb = valid_rgb[valid_rays_mask.flatten()]

            self.val_metric_manager.compute("psnr", valid_rgb, valid_rgb_gt)

        if len(self.val_out_types):
            # save image to disk

            render_time: Optional[float] = None
            if self.config.test.nrend.overlay_render_time:
                render_time = self.profilings["total_inference"][-1]
                render_keys = [k for k in self.profilings.keys() if k.endswith("render")]
                if render_keys:
                    render_time = self.profilings[min(render_keys, key=len)][-1]

            self.save_images(
                self.val_dir,
                batch,
                unpack_optional(results.rendered_cam),
                batch_idx,
                h=unpack_optional(unpack_optional(batch.rendering).camera).h,
                out_types=self.val_out_types,
                suffix_list=[],
                overlay_text=f"rendering FPS : {1000 / render_time:.2f}" if render_time else None,
            )

    def validation_step_lidar(self, batch: DataAndRenderingBatch) -> None:
        assert batch.rendering is not None, "Validation batch should have rendering data"
        assert batch.rendering.lidar is not None, "Validation batch should have lidar rendering data"

        batch_idx = unpack_optional(batch.data.idx)

        tracks_for_render = getattr(self, "calibrated_cuboid_tracks", None)

        # if supported, render the batch using the model renderer
        results: GaussiansCompositeReturn
        if isinstance(self.model, NRenderableModel) and self.model.has_nrend():
            self.prof_start.record()

            nrend_results_lidar = self.model.render_nrend_sensor_rays(
                batch_idx,
                batch.rendering.lidar,
                unpack_optional(batch.data.lidar).meta[0],
                tracks_for_render,
            )

            results = GaussiansCompositeReturn(rendered_cam=None, rendered_lidar=nrend_results_lidar)
            self.prof_end.record()
        else:
            if self.config.test.nrend.lidar_only:
                raise ValueError("NRendTestGaussiansSystem does not support lidar only mode when nrend is not enabled")
            self.prof_start.record()
            results = self(batch)
            self.prof_end.record()

        self.prof_end.synchronize()
        self.profilings["total_inference"].append(self.prof_start.elapsed_time(self.prof_end))

        # collecting nrend profilings
        if (
            isinstance(self.model, NRenderableModel)
            and self.model.has_nrend()
            and (self.config.test.nrend.profiling_frequency > 0.0)
        ):
            nrend_profilings = self.model.nrend_profilings()
            for tag, ms in nrend_profilings.items():
                tag_key = f"nrend{tag}"
                if tag_key not in self.profilings:
                    self.profilings[tag_key] = []
                self.profilings[tag_key].append(ms)
        else:
            nrend_profilings = self.model.collect_renderer_profilings()
            for tag, ms in nrend_profilings.items():
                tag_key = f"nrend{tag}"
                if tag_key not in self.profilings:
                    self.profilings[tag_key] = []
                self.profilings[tag_key].append(ms)

        # FIXME : add support for lidar test case
        if self.config.test.nrend.create_test_case and batch.rendering.lidar is not None:
            valid_rendering_results = unpack_optional(results.rendered_lidar)
            self.nrend_render_test_case(
                self.model,
                batch_idx,
                unpack_optional(unpack_optional(batch.rendering).lidar),
                unpack_optional(batch.data.lidar).meta[0],
                tracks_for_render,
                valid_rendering_results.rgb,
                valid_rendering_results.opacity,
                valid_rendering_results.distance,
                valid_rendering_results.extra_ray_signals,
                self.val_dir,
                lidar=True,
                update=self.config.test.nrend.create_test_case_update,
            )

        self.validation_lidar_step(batch, unpack_optional(results.rendered_lidar))

    def on_validation_epoch_end(self):
        # collecting nrend profilings
        if self.profilings:
            for tag, times in self.profilings.items():
                mean_time = torch.tensor(times[1:]).mean()
                self.log(f"test/profiling/{tag}", mean_time, prog_bar=True)
                times.clear()

        super().on_validation_epoch_end()

    def test_step(self, batch: DataAndRenderingBatch, batch_local_idx: int):
        raise NotImplementedError(f"[NRendTestGaussiansSystem] does not implement test mode. (use validation instead)")
