# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public eager-mode InstantNuRec inference model.

Applies semantic + optional cuboid-track-based dynamic-mask refinement
to the per-pixel outputs and packages them into
``KelvinInstantNuRecPrimitive``. Released checkpoints omit the learned
sky head, so :class:`KelvinStaticCore` derives a cubemap from canonical RGB
and predicted SKY pixels. The default static PLY path keeps empty dynamic-layer
slots; opt-in dynamic inference retains the temporal decoder outputs with
learned or track motion.
"""

from __future__ import annotations

import math

from typing import Literal

import torch

from torch import nn

from instant_nurec.datasets.tracks import CuboidTracks
from instant_nurec.model.static_core import KelvinStaticCore, KelvinStaticOutput
from instant_nurec.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinInstantNuRecPrimitive,
    KelvinSemanticClass,
    KelvinStaticLayer,
)
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.motion import (
    TimeRemapping,
    associate_points_with_cuboid_tracks,
    cuboid_track_time_bounds_us,
    warp_points_with_cuboid_tracks,
)
from instant_nurec.utils.sensor import to_simple_pinhole_model_parameters
from instant_nurec.utils.types import TrackFlags


class KelvinInferenceModel(nn.Module):
    """Wrap the eager model core and package its tensor outputs.

    Args:
        static_core: Source-defined encoder, decoder, and post-processing core.
        scene_rescale: Scale factor used by the released model.
        expected_frames: Number of input frames per inference sample.
        expected_height: Input image height after preprocessing.
        expected_width: Input image width after preprocessing.
        include_dynamic: Retain temporal Gaussian layers for dense PA profiles.
        motion_source: Use NCore tracks or the decoder's learned motion.
    """

    def __init__(
        self,
        static_core: KelvinStaticCore,
        *,
        scene_rescale: float,
        expected_frames: int,
        expected_height: int,
        expected_width: int,
        include_dynamic: bool = False,
        motion_source: Literal["tracks", "learned"] = "tracks",
    ) -> None:
        super().__init__()
        self.static_core = static_core
        self.scene_rescale = scene_rescale
        self.include_dynamic = include_dynamic
        self.motion_source = motion_source
        self.expected_b = 1
        self.expected_v = expected_frames
        self.expected_h = expected_height
        self.expected_w = expected_width
        self.register_buffer(
            "cuboids_dims_padding",
            static_core.decoder.cuboids_dims_padding.detach().clone(),
            persistent=False,
        )

    def _validate_input_shape(self, rgb: torch.Tensor) -> None:
        b, v, h, w, c = rgb.shape
        expected = (self.expected_b, self.expected_v, self.expected_h, self.expected_w, 3)
        if (b, v, h, w, c) != expected:
            raise ValueError(
                f"Input shape mismatch: got rgb {tuple(rgb.shape)}, "
                f"expected {expected}. Model expects {self.expected_v} "
                f"input frames at {self.expected_h}x{self.expected_w}; "
                f"check that ``len(context_camera_ids) * n_frames_per_sample`` "
                f"equals {self.expected_v}."
            )

    def prepare_context(
        self,
        context: list[DataAndRenderingBatch],
    ) -> list[DataAndRenderingBatch]:
        return context

    # ------------------------------------------------------------------
    # reconstruct: tensor extraction and primitive packaging
    # ------------------------------------------------------------------

    def _extract_tensors(self, batch: DataAndRenderingBatch):
        """Extract the model's input tensors from a single
        ``DataAndRenderingBatch`` (chunk_size=1)."""
        data = unpack_optional(batch.data.camera)
        rendering = unpack_optional(unpack_optional(batch.rendering).camera)

        rgb = unpack_optional(data.labels.rgb).unsqueeze(0)  # (1, V, H, W, 3)
        rays = rendering.rays.unsqueeze(0)  # (1, V, H, W, 6)
        distance_to_depth_scale = rendering.distance_to_depth_scale.unsqueeze(0)

        c2w = tquat_to_se3_matrix(rendering.poses_tquat_startend[:, 1, :], unbatch=False)
        c2w = c2w.clone()
        c2w[:, :3, 3] *= self.scene_rescale
        c2w = c2w.unsqueeze(0)  # (1, V, 4, 4)

        pinhole_parameters = [
            to_simple_pinhole_model_parameters(rendering.sensor_model_parameters[vidx]) for vidx in range(data.b)
        ]
        fov_list = []
        for p in pinhole_parameters:
            fov_w = 2 * math.atan2(p.resolution[0] / 2, p.focal_length[0])
            fov_h = 2 * math.atan2(p.resolution[1] / 2, p.focal_length[1])
            fov_list.append([fov_w, fov_h])
        fov = torch.tensor(fov_list, dtype=torch.float32, device=rgb.device).unsqueeze(0)

        camera_idxs = (
            torch.tensor([meta.unique_sensor_idx for meta in data.meta], dtype=torch.int64).to(rgb.device).unsqueeze(0)
        )

        return rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs

    def _compute_motion(
        self,
        gs_xyz: torch.Tensor,
        semantic_argmax: torch.Tensor,
        rendering_camera,
        cuboid_tracks_b: CuboidTracks | None,
        source_indices: torch.Tensor | None = None,
        target_timestamps_us_list: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
        """Compute the shared mask, optional warped positions and actor bounds.

        Bounds have shape (N_dynamic, 2), in the dynamic mask's flattened order.

        Without tracks (or with an empty track set): dynamic_mask is the semantic
        argmax-equals-MOVABLE.

        With ``cuboid_tracks_b``: associate points using the same two-pass
        cuboid contract as training, then keep only dynamic-track assignments.
        """
        # Single-batch slice: dense outputs have shapes (V, H, W, 3) and
        # (V, H, W); point-query outputs have shapes (N, 3) and (N,).
        gs_xyz_v = gs_xyz[0]
        semantic_v = semantic_argmax[0]

        if cuboid_tracks_b is None or cuboid_tracks_b.n_tracks == 0:
            return semantic_v == self._semantic_movable_value(), None, None

        movable_mask = semantic_v == self._semantic_movable_value()
        ray_ts = unpack_optional(rendering_camera.rays_timestamps_us)  # (V, H, W, 1)
        if source_indices is not None:
            # The sparse decoder carries each Gaussian's flattened source-pixel
            # index so cuboid-track association uses the aligned time.
            source_indices_v = source_indices[0].long()
            ray_ts = ray_ts.reshape(-1, 1)[source_indices_v]

        tracks_idx = associate_points_with_cuboid_tracks(
            points=gs_xyz_v,
            points_timestamps_us=ray_ts,
            points_dynamic_mask=movable_mask.unsqueeze(-1),
            cuboid_tracks=cuboid_tracks_b,
            cuboids_dims_padding=self.cuboids_dims_padding,
        )
        dynamic_track_mask = (cuboid_tracks_b.tracks_flags & TrackFlags.DYNAMIC) != 0
        if dynamic_track_mask.numel() > 0:
            is_dynamic_track = dynamic_track_mask[tracks_idx.clamp_min(0).to(torch.long)]
            tracks_idx.masked_fill_(~is_dynamic_track, -1)

        dynamic_mask, warped_points = warp_points_with_cuboid_tracks(
            points=gs_xyz_v,
            source_timestamps_us=ray_ts,
            target_timestamps_us_list=target_timestamps_us_list if target_timestamps_us_list is not None else [ray_ts],
            cuboid_tracks=cuboid_tracks_b,
            tracks_idx=tracks_idx,
        )
        dynamic_mask = dynamic_mask.squeeze(-1)
        time_bounds = None
        if target_timestamps_us_list is not None:
            # Use exactly the same flattened selection as dynamic-layer packing.
            time_bounds = cuboid_track_time_bounds_us(cuboid_tracks_b, tracks_idx.squeeze(-1)[dynamic_mask])
        return dynamic_mask, warped_points, time_bounds

    @staticmethod
    def _semantic_movable_value() -> int:
        return KelvinSemanticClass.MOVABLE.value

    def _empty_dynamic_layer(self, device: torch.device) -> KelvinDynamicLayer:
        """Zero-gaussian KelvinDynamicLayer placeholder. Required because
        ``KelvinPrimitiveMerge`` asserts ``len(primitive.dynamic_layers) == 1``."""
        return KelvinDynamicLayer(
            max_densities=torch.zeros(0, 1, device=device),
            keyframe_positions=torch.zeros(0, 3, 3, device=device),
            keyframe_timestamps_us=torch.zeros(0, 3, dtype=torch.int64, device=device),
            rotations=torch.zeros(0, 4, device=device),
            scales=torch.zeros(0, 3, device=device),
            rgb=torch.zeros(0, 3, device=device),
        )

    @torch.autocast("cuda", enabled=False)
    def _decode_dynamic_layer(
        self,
        output: KelvinStaticOutput,
        dynamic_mask: torch.Tensor,
        timestamps: list[torch.Tensor],
        time_remapping: TimeRemapping,
        warped_points: list[torch.Tensor] | None,
        motion_time_bounds_us: torch.Tensor | None,
    ) -> KelvinDynamicLayer:
        points = output.positions[0]
        if warped_points is None:
            prev_flow, next_flow = self.static_core.decoder.context_motion_head(
                unpack_optional(output.encoded_latent),
                output_shape=points.shape[1:3],
                fusion_features=None,
                chunk_size=self.static_core.decoder.config.dpt_chunk_size,
                time_remappings=[time_remapping],
                source_timestamps_us=timestamps[1].unsqueeze(0),
                prev_target_timestamps_us=timestamps[0].unsqueeze(0),
                next_target_timestamps_us=timestamps[2].unsqueeze(0),
            )
            prev_points = points + prev_flow[0] / self.scene_rescale
            next_points = points + next_flow[0] / self.scene_rescale
        else:
            prev_points, next_points = warped_points
        indices = torch.where(dynamic_mask.reshape(-1))[0]
        keyframe_timestamps = torch.stack([t.reshape(-1)[indices] for t in timestamps], dim=1)
        if motion_time_bounds_us is not None:
            # Positions were queried at these effective times, not the requested
            # times. Keep endpoint timing in the stored motion representation.
            keyframe_timestamps = keyframe_timestamps.clamp(
                min=motion_time_bounds_us[:, :1], max=motion_time_bounds_us[:, 1:]
            )
        layer = KelvinDynamicLayer(
            keyframe_positions=torch.stack(
                [p.reshape(-1, 3)[indices] for p in (prev_points, points, next_points)], dim=1
            ),
            keyframe_timestamps_us=keyframe_timestamps,
            motion_time_bounds_us=motion_time_bounds_us,
            rotations=output.rotations[0].reshape(-1, 4)[indices],
            scales=output.scales[0].reshape(-1, 3)[indices],
            max_densities=output.densities[0].reshape(-1, 1)[indices],
            rgb=output.rgb[0].reshape(-1, 3)[indices],
        )
        return layer.ensure_minimum_density(0.75)

    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None = None,
    ) -> list[KelvinInstantNuRecPrimitive]:
        if self.motion_source == "learned":
            cuboid_tracks = None
        primitives: list[KelvinInstantNuRecPrimitive] = []
        for bidx, batch in enumerate(context):
            tensors = self._extract_tensors(batch)
            self._validate_input_shape(tensors[0])
            tracks = cuboid_tracks[bidx] if cuboid_tracks is not None else None
            learned_motion = self.include_dynamic and (tracks is None or tracks.n_tracks == 0)
            output = self.static_core(*tensors, retain_features=learned_motion)
            rendering = unpack_optional(unpack_optional(batch.rendering).camera)
            target_timestamps = None
            if self.include_dynamic:
                source_timestamps = unpack_optional(rendering.rays_timestamps_us)
                time_remapping = TimeRemapping.from_timestamps_startend_us(
                    rendering.timestamps_startend_us_cpu, tensors[-1][0].cpu()
                )
                gaps = time_remapping.frame_gap_timestamps_us.to(source_timestamps.device)
                prev_timestamps = source_timestamps - gaps[:, 0, None, None, None]
                next_timestamps = source_timestamps + gaps[:, 1, None, None, None]
                target_timestamps = [prev_timestamps, next_timestamps]
            dynamic_mask, warped_points, motion_time_bounds = self._compute_motion(
                output.positions, output.semantic_class, rendering, tracks, output.source_indices, target_timestamps
            )
            static_idx = torch.where(~dynamic_mask.reshape(-1))[0]
            static_layer = KelvinStaticLayer(
                positions=output.positions[0].reshape(-1, 3)[static_idx],
                rotations=output.rotations[0].reshape(-1, 4)[static_idx],
                scales=output.scales[0].reshape(-1, 3)[static_idx],
                densities=output.densities[0].reshape(-1, 1)[static_idx],
                rgb=output.rgb[0].reshape(-1, 3)[static_idx],
                semantic_class=output.semantic_class[0].reshape(-1)[static_idx].unsqueeze(-1).to(torch.uint8),
                normals=output.normals[0].reshape(-1, 3)[static_idx],
            )
            dynamic_layer = self._empty_dynamic_layer(static_layer.positions.device)
            if self.include_dynamic:
                dynamic_layer = self._decode_dynamic_layer(
                    output, dynamic_mask, [prev_timestamps, source_timestamps, next_timestamps],
                    time_remapping, warped_points, motion_time_bounds,
                )
            primitives.append(
                KelvinInstantNuRecPrimitive(
                    static_layer=static_layer,
                    dynamic_layers=[dynamic_layer],
                    sky_cubemap=output.sky_cubemap[0],
                    sky_cubemap_mask=output.sky_cubemap_mask[0],
                    affine_matrix=output.affine_matrix[0],
                )
            )
        return primitives
