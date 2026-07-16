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
``KelvinInstantNuRecPrimitive``. Sky cubemap and dynamic-layer slots
are filled with zero placeholders to satisfy the primitive invariants
and the chunk-merge code path; ``export_ply.py`` reads only
``static_layer``.
"""

from __future__ import annotations

import math

import torch

from torch import nn

from instant_nurec.datasets.tracks import CuboidTracks
from instant_nurec.model.static_core import KelvinStaticCore
from instant_nurec.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinInstantNuRecPrimitive,
    KelvinSemanticClass,
    KelvinStaticLayer,
)
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.motion import warp_points_with_cuboid_tracks
from instant_nurec.utils.sensor import to_simple_pinhole_model_parameters
from instant_nurec.utils.types import TrackFlags


# Cubemap placeholder size: small enough to keep memory negligible, large
# enough that the merge code's per-pixel arithmetic produces well-defined
# results. The merged cubemap is computed but never written -- ``export_ply.py``
# only reads ``static_layer``.
_PLACEHOLDER_SKY_CUBEMAP_SIZE = 16


class KelvinInferenceModel(nn.Module):
    """Wrap the eager model core and package its tensor outputs.

    Args:
        static_core: Source-defined encoder, decoder, and post-processing core.
        scene_rescale: Scale factor used by the released model.
        expected_frames: Number of input frames per inference sample.
        expected_height: Input image height after preprocessing.
        expected_width: Input image width after preprocessing.
    """

    def __init__(
        self,
        static_core: KelvinStaticCore,
        *,
        scene_rescale: float,
        expected_frames: int,
        expected_height: int,
        expected_width: int,
    ) -> None:
        super().__init__()
        self.static_core = static_core
        self.scene_rescale = scene_rescale
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
            to_simple_pinhole_model_parameters(rendering.sensor_model_parameters[vidx])
            for vidx in range(data.b)
        ]
        fov_list = []
        for p in pinhole_parameters:
            fov_w = 2 * math.atan2(p.resolution[0] / 2, p.focal_length[0])
            fov_h = 2 * math.atan2(p.resolution[1] / 2, p.focal_length[1])
            fov_list.append([fov_w, fov_h])
        fov = torch.tensor(fov_list, dtype=torch.float32, device=rgb.device).unsqueeze(0)

        camera_idxs = (
            torch.tensor([meta.unique_sensor_idx for meta in data.meta], dtype=torch.int64)
            .to(rgb.device)
            .unsqueeze(0)
        )

        return rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs

    def _compute_dynamic_mask(
        self,
        gs_xyz: torch.Tensor,
        semantic_argmax: torch.Tensor,
        rendering_camera,
        cuboid_tracks_b: CuboidTracks | None,
    ) -> torch.Tensor:
        """Compute the dynamic mask.

        Without ``cuboid_tracks_b``: dynamic_mask is purely the semantic
        argmax-equals-MOVABLE.

        With ``cuboid_tracks_b``: refine via point-cuboid intersection
        (and fallback ray-cuboid intersection on movable rays).
        """
        # Single-batch slice: shapes (V, H, W, 3) and (V, H, W).
        gs_xyz_v = gs_xyz[0]
        semantic_v = semantic_argmax[0]

        if cuboid_tracks_b is None:
            return semantic_v == self._semantic_movable_value()

        dynamic_track = CuboidTracks.Ops.subset_from_mask(
            cuboid_tracks_b, cuboid_tracks_b.tracks_flags & TrackFlags.DYNAMIC != 0
        )

        movable_mask = semantic_v == self._semantic_movable_value()
        rays = rendering_camera.rays  # (V, H, W, 6)
        ray_ts = unpack_optional(rendering_camera.rays_timestamps_us)  # (V, H, W, 1)

        aux_ray_intersection_result = dynamic_track.ray_intersection(
            rays[..., :3][movable_mask],
            rays[..., 3:][movable_mask],
            ray_ts[..., 0][movable_mask],
            max_intersections_per_ray=2,
        )
        aux_movable_tracks_idx = aux_ray_intersection_result.intersections_tracks_idx[..., 0]
        aux_movable_tracks_idx[aux_ray_intersection_result.intersections_cnt != 1] = -1
        aux_tracks_idx = torch.full_like(movable_mask, -1, dtype=aux_movable_tracks_idx.dtype)
        aux_tracks_idx[movable_mask] = aux_movable_tracks_idx

        prev_target_ts = ray_ts.clone()  # placeholder -- only dynamic_mask is used downstream
        next_target_ts = ray_ts.clone()
        dynamic_mask, _ = warp_points_with_cuboid_tracks(
            points=gs_xyz_v,
            source_timestamps_us=ray_ts,
            target_timestamps_us_list=[prev_target_ts, next_target_ts],
            dynamic_tracks=dynamic_track,
            aux_tracks_idx=aux_tracks_idx,
            cuboids_dims_padding=self.cuboids_dims_padding,
        )
        return dynamic_mask

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

    def _placeholder_sky_cubemap(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        s = _PLACEHOLDER_SKY_CUBEMAP_SIZE
        return torch.zeros(6, s, s, 3, device=device, dtype=dtype)

    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None = None,
    ) -> list[KelvinInstantNuRecPrimitive]:
        primitives: list[KelvinInstantNuRecPrimitive] = []
        for bidx, batch in enumerate(context):
            tensors = self._extract_tensors(batch)
            self._validate_input_shape(tensors[0])
            (
                gs_xyz,
                gs_rotations,
                gs_scales,
                gs_densities,
                gs_rgb,
                semantic_argmax,
                normals,
                affine,
            ) = self.static_core(*tensors)

            rendering_camera = unpack_optional(unpack_optional(batch.rendering).camera)
            cuboid_tracks_b = cuboid_tracks[bidx] if cuboid_tracks is not None else None
            dynamic_mask = self._compute_dynamic_mask(
                gs_xyz, semantic_argmax, rendering_camera, cuboid_tracks_b
            )

            # Flatten and gather static-only.
            dynamic_mask_flat = dynamic_mask.reshape(-1)
            static_idx = torch.where(~dynamic_mask_flat)[0]
            static_layer = KelvinStaticLayer(
                positions=gs_xyz[0].reshape(-1, 3)[static_idx],
                rotations=gs_rotations[0].reshape(-1, 4)[static_idx],
                scales=gs_scales[0].reshape(-1, 3)[static_idx],
                densities=gs_densities[0].reshape(-1, 1)[static_idx],
                rgb=gs_rgb[0].reshape(-1, 3)[static_idx],
                semantic_class=semantic_argmax[0].reshape(-1)[static_idx].unsqueeze(-1).to(torch.uint8),
                normals=normals[0].reshape(-1, 3)[static_idx],
            )

            primitives.append(
                KelvinInstantNuRecPrimitive(
                    static_layer=static_layer,
                    dynamic_layers=[self._empty_dynamic_layer(static_layer.positions.device)],
                    sky_cubemap=self._placeholder_sky_cubemap(
                        static_layer.positions.device, static_layer.positions.dtype
                    ),
                    affine_matrix=affine[0],  # (1, n_cams, 3, 4) -> (n_cams, 3, 4)
                )
            )
        return primitives
