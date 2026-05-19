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

from __future__ import annotations

import logging
import math

from dataclasses import dataclass
from typing import cast

import torch  # type: ignore

from ncore.data import ConcreteCameraModelParametersUnion  # type: ignore
from ncore.sensors import CameraModel  # type: ignore
from instant_nurec.config_schema.predict import PrimitiveMergeConfig
from instant_nurec.primitives.kelvin_primitive import KelvinDynamicLayer, KelvinInstantNuRecPrimitive, KelvinStaticLayer
from instant_nurec.utils.trajectory import merge_rig_trajectories, transform_rig_trajectories
from instant_nurec.utils.batch import CameraFreePoseViewGeometry, DataAndRenderingBatch, DataBatch, InstantNuRecDataBatch, RenderingBatch
from instant_nurec.utils.geometry import se3_matrix_inverse, tquat_to_se3_matrix
from instant_nurec.utils.misc import list_of_dicts_to_singleton_dict, unpack_optional
from instant_nurec.utils.types import RigTrajectories


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class CameraFrustum:
    """
    Represents a camera frustum that are mainly used for observability check.
    """

    camera_model: CameraModel
    poses_T_startend: torch.Tensor

    def in_frustum(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Approximated check if the given positions are within the camera frustum.
        """
        w, h = self.camera_model.resolution.tolist()
        T_world_sensor = se3_matrix_inverse(self.poses_T_startend[1], unbatch=True)
        local_positions = positions @ T_world_sensor[:3, :3].T + T_world_sensor[:3, 3]
        local_positions /= local_positions[:, 2:].abs()

        local_rays = self.camera_model.pixels_to_camera_rays(
            torch.tensor([[0, 0], [w, h]], dtype=torch.int32, device=self.camera_model.device),
        )
        local_rays /= local_rays[:, 2:]

        return (
            (local_positions[:, 2] > 0)
            & torch.all(local_positions[:, :2] >= local_rays[0, :2], dim=1)
            & torch.all(local_positions[:, :2] <= local_rays[1, :2], dim=1)
        )

    def distance_to_center(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance to the camera for the given positions.
        """
        camera_center = self.poses_T_startend[1][:3, 3]
        return torch.norm(positions - camera_center, dim=1)



def merge_context_batch(
    context_batches: list[DataAndRenderingBatch],
    context_frame_mapping: dict[tuple[int, int], int],
) -> DataBatch:
    """
    Collate a new data batch with correct unique frame idx mapping
    """
    collated_context_batch = DataBatch.collate_fn([cb.data for cb in context_batches])
    batch_indices = sum([[bidx] * unpack_optional(cb.data.camera).b for bidx, cb in enumerate(context_batches)], [])
    for idx, frame_meta in zip(batch_indices, unpack_optional(collated_context_batch.camera).meta):
        new_idx = context_frame_mapping[(idx, frame_meta.unique_frame_idx)]
        frame_meta.unique_frame_idx = new_idx
        # FrameMeta.__post_init__ caches `unique_frame_idx_tensor` at construction time based on
        # the original chunk-local index -- need to rebuild.
        if frame_meta.unique_frame_idx_tensor is not None:
            frame_meta.unique_frame_idx_tensor = torch.tensor(
                [new_idx],
                dtype=frame_meta.unique_frame_idx_tensor.dtype,
                device=frame_meta.unique_frame_idx_tensor.device,
            )
    return collated_context_batch


def build_world_camera_frustums(
    batch: InstantNuRecDataBatch,
    batch_rig_transforms: list[torch.Tensor],
) -> list[list[CameraFrustum]]:
    """
    Build camera frustums per chunk in world space for distance checks and merging strategies.
    """
    batch_camera_frustums: list[list[CameraFrustum]] = []
    for b_idx, batch_data in enumerate(batch.context):
        rendering_data = unpack_optional(unpack_optional(batch_data.rendering).camera)
        camera_frustums: list[CameraFrustum] = []
        rig_T = batch_rig_transforms[b_idx].to(
            device=rendering_data.poses_tquat_startend.device, dtype=torch.float32
        )
        for frame_idx in range(rendering_data.b):
            global_T_sensor = rig_T @ tquat_to_se3_matrix(rendering_data.poses_tquat_startend[frame_idx])
            camera_model_parameters = cast(
                ConcreteCameraModelParametersUnion, rendering_data.sensor_model_parameters[frame_idx]
            )
            camera_frustums.append(
                CameraFrustum(
                    camera_model=CameraModel.from_parameters(camera_model_parameters),
                    poses_T_startend=global_T_sensor,
                )
            )
        batch_camera_frustums.append(camera_frustums)
    return batch_camera_frustums


def compute_frustum_ownership_mask(
    batch_idx: int, positions: torch.Tensor, batch_camera_frustums: list[list[CameraFrustum]], max_diff_m: float
) -> torch.Tensor:
    """
    Compute the mask of the Gaussians that is guaranteed to be owned by the current chunk, so that we can
    drop the others. The idea is that if a Gaussian comes from chunk i, but it affects chunk j more than chunk i,
    then this means there must be a Gaussian actually coming from chunk j that covers the same geometry.

    Args:
        batch_idx: The index of the current chunk
        positions: The positions of the Gaussians
        batch_camera_frustums: The camera frustums of all chunks
        max_diff_m: The maximum distance in meters between the distances from one GS to non-owned chunks and owned chunks

    Returns:
        The mask of the Gaussians that is guaranteed to be owned by the current chunk
    """
    all_distances: list[torch.Tensor] = []
    for b_jidx, camera_frustums_j in enumerate(batch_camera_frustums):
        # Prefer the current chunk if no chunk see this Gaussian (which is less likely but might happen)
        # oob -> out of bounds
        oob_distance = 1.0e6 if b_jidx != batch_idx else 1.0e5

        distance_j = torch.full_like(positions[:, 0], oob_distance)
        for camera_frustum in camera_frustums_j:
            center_dist = camera_frustum.distance_to_center(positions)
            in_frustum = camera_frustum.in_frustum(positions)
            # NB [JH]: Add normal check to make sure backward facing Gaussians are eliminated to oob.
            center_dist[~in_frustum] = oob_distance
            distance_j = torch.minimum(distance_j, center_dist)

        all_distances.append(distance_j)

    # Compute the closest two chunks for each Gaussian
    dists, inds = torch.topk(torch.stack(all_distances, dim=0), k=2, dim=0, largest=False)
    # Keep the gaussian if it's owned by current chunk, or the closest non-owned chunk is close enough to this one.
    keep_mask = (inds[0] == batch_idx) | ((inds[1] == batch_idx) & (dists[1] - dists[0] < max_diff_m))
    return keep_mask


class KelvinPrimitiveMerge:
    """
    Merge Kelvin primitives from non-overlapping chunks into a single primitive.
    """

    def __init__(self, config: PrimitiveMergeConfig):
        self.config = config

    def _maybe_voxelize_static_layer(
        self, merged_primitive: KelvinInstantNuRecPrimitive
    ) -> KelvinInstantNuRecPrimitive:
        """Bracketed binary search over voxel size to land the static-layer
        count in ``[0.9 * target_n_gaussians, target_n_gaussians]``.

        Starting from ``self.config.voxel_size`` (default 0.1), each
        iteration re-voxelizes the *original* merged static layer at a
        candidate voxel size:

          * count > target → current voxel_size is too small (too many
            Gaussians). Record it as the lower bracket; next candidate is
            the midpoint of the bracket if both ends are known, else
            double (search for the upper bracket).
          * count < 0.9 * target → current voxel_size is too large (too
            few Gaussians). Record it as the upper bracket; next
            candidate is the midpoint if bracketed, else halve.
          * count in band → return.

        The midpoint step replaces the previous blind doubling/halving
        and converges for any monotone count-vs-size relationship,
        including targets that the coarse 2× step would have oscillated
        across (e.g. count drops 8× per doubling but the acceptance band
        is only 10% wide).

        No-op when ``self.config.enable_voxelization`` is False so
        default predict runs stay byte-identical to the pre-voxelization
        path.

        The voxel count function is discrete (Gaussians bucket via
        ``round(positions / voxel_size)``), so for some target values
        the band falls *between* two adjacent count plateaus and the
        bracket shrinks without ever entering the band; the loop exits
        with a WARNING when the bracket width falls below
        ``BRACKET_TOL`` relative to its upper end, or when
        ``max_voxelization_iterations`` is reached.
        """
        if not self.config.enable_voxelization:
            return merged_primitive

        original_static = merged_primitive.static_layer
        n_before = len(original_static)
        target = self.config.target_n_gaussians
        lower_bound = 0.9 * target
        voxel_size = self.config.voxel_size
        max_iter = self.config.max_voxelization_iterations
        # Bracket invariant: target voxel size lies in (v_low, v_high).
        v_low = 0.0
        v_high = math.inf
        BRACKET_TOL = 1e-3  # stop when (v_high - v_low) / v_high < BRACKET_TOL

        voxelized: KelvinStaticLayer | None = None
        chosen_voxel_size = voxel_size
        converged = False
        bracket_collapsed = False
        iteration = 0
        for iteration in range(1, max_iter + 1):
            chosen_voxel_size = voxel_size
            voxelized = original_static.voxelize(voxel_size)
            n = len(voxelized)
            if n > target:
                # voxel_size too small → tighten lower bracket
                v_low = max(v_low, voxel_size)
            elif n < lower_bound:
                # voxel_size too large → tighten upper bracket
                v_high = min(v_high, voxel_size)
            else:
                converged = True
                break
            if math.isfinite(v_high) and v_low > 0.0:
                if (v_high - v_low) / v_high < BRACKET_TOL:
                    bracket_collapsed = True
                    break
                voxel_size = 0.5 * (v_low + v_high)
            elif n > target:
                voxel_size *= 2.0
            else:  # n < lower_bound
                voxel_size /= 2.0

        assert voxelized is not None  # max_iter > 0 enforced by config validation
        merged_primitive.static_layer = voxelized
        n_after = len(voxelized)
        reduction_pct = 100.0 * (1.0 - n_after / n_before) if n_before > 0 else 0.0
        if converged:
            logger.info(
                "Voxelization converged in %d iter (voxel_size=%.4g, target=%d): "
                "%d -> %d static Gaussians (%.1f%% reduction)",
                iteration, chosen_voxel_size, target, n_before, n_after, reduction_pct,
            )
        elif bracket_collapsed:
            logger.warning(
                "Voxelization search bracket collapsed after %d iter without "
                "entering the band (voxel_size=%.4g, target=%d, last count=%d); "
                "the discrete voxel count function jumps over [0.9*target, target]. "
                "Returning last result.",
                iteration, chosen_voxel_size, target, n_after,
            )
        else:
            logger.warning(
                "Voxelization did not converge after %d iter (voxel_size=%.4g, "
                "target=%d, last count=%d); returning last result.",
                max_iter, chosen_voxel_size, target, n_after,
            )
        return merged_primitive

    @torch.autocast(device_type="cuda", enabled=False)
    def merge_primitives_and_batch(
        self,
        primitives_list: list[KelvinInstantNuRecPrimitive],
        batch: InstantNuRecDataBatch,
    ) -> tuple[KelvinInstantNuRecPrimitive, InstantNuRecDataBatch]:
        """
        Merge primitives from non-overlapping chunks into a single primitive.

        Stage 1 transforms each primitive into the reference frame (first chunk) so they can be
        concatenated; stage 2 dispatches to ``merge_processed_primitives``; stage 3 stitches the
        per-chunk batches into a single merged batch.
        """
        assert len(primitives_list) > 0, "No primitives to merge"
        logger.info(f"Merging {len(primitives_list)} chunks ({sum(len(p) for p in primitives_list)} Gaussians)")

        batch_context_rig: list[RigTrajectories] = unpack_optional(batch.context_rig)
        T_world_ref: torch.Tensor = se3_matrix_inverse(batch_context_rig[0].T_world_base)
        batch_rig_transforms: list[torch.Tensor] = [T_world_ref @ cr.T_world_base for cr in batch_context_rig]
        for b_idx, primitive in enumerate(primitives_list):
            primitives_list[b_idx] = primitive.rigid_transform(
                batch_rig_transforms[b_idx].to(device=primitive.device(), dtype=torch.float32)
            )

        merged_primitive = self.merge_processed_primitives(primitives_list, batch_rig_transforms, batch)
        merged_primitive = self._maybe_voxelize_static_layer(merged_primitive)

        logger.info(f"Merged {len(primitives_list)} primitives into {repr(merged_primitive)}")

        if len(batch.context) == 1:
            merged_batch = batch
        else:
            merged_context_rig, context_frame_mapping = merge_rig_trajectories(
                [
                    transform_rig_trajectories(rig_trajectories, left_transform=rig_transform)
                    for rig_trajectories, rig_transform in zip(batch_context_rig, batch_rig_transforms)
                ]
            )
            merged_context_data = merge_context_batch(batch.context, context_frame_mapping)
            device = merged_primitive.device()
            merged_context_rendering = (
                CameraFreePoseViewGeometry.from_rig_trajectories(merged_context_rig)
                .to(device=device)
                .to_rendering_data(unpack_optional(merged_context_data.camera).to(device))
            )
            merged_context_batch = DataAndRenderingBatch(
                data=merged_context_data, rendering=RenderingBatch(camera=merged_context_rendering)
            )
            merged_meta = None if batch.meta is None else list_of_dicts_to_singleton_dict(batch.meta)
            merged_batch = InstantNuRecDataBatch(
                context=[merged_context_batch],
                context_rig=[merged_context_rig],
                cuboid_tracks=None,
                meta=[merged_meta] if merged_meta is not None else None,
            )

        return merged_primitive, merged_batch

    def _merge_sky_cubemaps(
        self,
        sky_cubemaps: list[torch.Tensor],
        batch: InstantNuRecDataBatch,
        batch_rig_transforms: list[torch.Tensor],
    ) -> torch.Tensor:
        del batch, batch_rig_transforms
        return torch.stack(sky_cubemaps, dim=0).mean(dim=0)

    @torch.autocast(device_type="cuda", enabled=False)
    def _merge_affine_matrices(self, affine_matrices: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Merge affine matrices from multiple primitives.
        This is achieved by computing the best affine matrix Y_i for each chunk i (where we fix Y_1 = identity)
        So that Aff_i @ Y_i ~= Aff_1 @ Y_1 (in Frobenius norm) for all i.
        Returns merged affine matrices, and Y_i_inv to be applied to the RGB values of the primitive.
        """
        ref_matrices: torch.Tensor = affine_matrices[0].float()
        y_inv_matrices: list[torch.Tensor] = []

        for bidx in range(len(affine_matrices)):
            affine_matrix_bidx = affine_matrices[bidx].float()
            lhs = affine_matrix_bidx[:, :3, :3].reshape(-1, 3)
            rhs = torch.cat(
                [
                    ref_matrices[:, :3, :3].reshape(-1, 3),
                    (ref_matrices[:, :3, 3] - affine_matrix_bidx[:, :3, 3]).reshape(-1, 1),
                ],
                dim=1,
            )
            y_matrix = torch.linalg.lstsq(lhs, rhs).solution
            y_rot_inv = torch.linalg.inv(y_matrix[:3, :3])
            y_inv_matrices.append(torch.cat([y_rot_inv, -y_rot_inv @ y_matrix[:3, 3:4]], dim=1))

        return ref_matrices, y_inv_matrices

    def merge_processed_primitives(
        self, all_primitives: list[KelvinInstantNuRecPrimitive], batch_rig_transforms: list[torch.Tensor], batch: InstantNuRecDataBatch
    ) -> KelvinInstantNuRecPrimitive:
        if len(all_primitives) == 1:
            return all_primitives[0]

        batch_camera_frustums = build_world_camera_frustums(batch, batch_rig_transforms)

        all_static_layers: list[KelvinStaticLayer] = []
        all_dynamic_layers: list[KelvinDynamicLayer] = []
        for b_idx, primitive in enumerate(all_primitives):
            static_layer = primitive.static_layer
            static_mask = compute_frustum_ownership_mask(
                b_idx, static_layer.positions, batch_camera_frustums, self.config.frustum_ownership_max_diff_m
            )
            all_static_layers.append(static_layer.mask(static_mask))

            assert len(primitive.dynamic_layers) == 1, "Dynamic layer association is not supported for now."
            dynamic_layer = primitive.dynamic_layers[0]
            all_dynamic_layers.append(dynamic_layer)

        # Compute the best affine matrix
        merged_affine_matrix, y_inv_matrices = self._merge_affine_matrices([p.affine_matrix for p in all_primitives])
        for b_idx, y_inv_matrix in enumerate(y_inv_matrices):
            all_primitives[b_idx].color_transform(y_inv_matrix)

        merged_sky_cubemap = self._merge_sky_cubemaps(
            [p.sky_cubemap for p in all_primitives], batch, batch_rig_transforms
        )

        merged_primitive = KelvinInstantNuRecPrimitive(
            static_layer=KelvinStaticLayer.concatenate(all_static_layers),
            dynamic_layers=[KelvinDynamicLayer.concatenate(all_dynamic_layers)],
            sky_cubemap=merged_sky_cubemap,
            affine_matrix=merged_affine_matrix,
        )
        return merged_primitive
