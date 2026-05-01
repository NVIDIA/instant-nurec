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

from dataclasses import dataclass
from typing import cast

import torch  # type: ignore

from ncore.data import ConcreteCameraModelParametersUnion  # type: ignore
from ncore.sensors import CameraModel  # type: ignore
from nre.nrm.config.models import PrimitiveExportPreprocessConfig
from nre.nrm.config.predict import PrimitiveMergeConfig
from nre.nrm.primitives.kelvin_primitive import KelvinDynamicLayer, KelvinNRMPrimitive, KelvinStaticLayer
from nre.nrm.utils.cubemap import unproject_to_sky_cubemap
from nre.nrm.utils.trajectory import merge_rig_trajectories, transform_rig_trajectories
from nre.utils.batch import CameraFreePoseViewGeometry, DataAndRenderingBatch, DataBatch, NRMDataBatch, RenderingBatch
from nre.utils.geometry import se3_matrix_inverse, tquat_to_se3_matrix
from nre.utils.misc import list_of_dicts_to_dict_of_lists, unpack_optional
from nre.utils.types import RayFlags, RigTrajectories


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
    batch: NRMDataBatch,
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

    def __init__(
        self,
        config: PrimitiveMergeConfig,
        export_preprocess_config: PrimitiveExportPreprocessConfig | None = None,
    ):
        self.config = config
        self.export_preprocess_config = export_preprocess_config

    @torch.autocast(device_type="cuda", enabled=False)
    def merge_primitives_and_batch(
        self,
        primitives_list: list[KelvinNRMPrimitive],
        batch: NRMDataBatch,
    ) -> tuple[KelvinNRMPrimitive, NRMDataBatch]:
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
                .to_rendering_data(unpack_optional(merged_context_data.camera).to(device), cache_sensor_params=True)
            )
            merged_context_batch = DataAndRenderingBatch(
                data=merged_context_data, rendering=RenderingBatch(camera=merged_context_rendering)
            )
            merged_meta = None if batch.meta is None else list_of_dicts_to_dict_of_lists(batch.meta, singleton=True)
            merged_batch = NRMDataBatch(
                context=[merged_context_batch],
                context_rig=[merged_context_rig],
                supervision=None,
                cuboid_tracks=None,
                supervision_rig=None,
                meta=[merged_meta] if merged_meta is not None else None,
            )

        return merged_primitive, merged_batch

    @staticmethod
    def _soften_cubemap_mask(
        cubemap_mask: torch.Tensor,
        dilate_kernel_size: int,
        blur_sigma: float,
    ) -> torch.Tensor:
        """
        Soften a per-face cubemap mask via morphological dilation followed by a Gaussian blur,
        producing smooth weights in [0, 1] that fade out around the original mask edges.

        Each cubemap face is processed independently; smoothing does not cross face seams.
        Args:
            cubemap_mask: (6, H, W, 1) boolean or float mask.
            dilate_kernel_size: Square structuring element side length (in pixels) for dilation.
                Use <= 1 to skip dilation.
            blur_sigma: Standard deviation (in pixels) of the separable Gaussian blur.
                Use <= 0 to skip blurring.
        Returns:
            (6, H, W, 1) float tensor with values in [0, 1].
        """
        x = cubemap_mask.float().permute(0, 3, 1, 2)  # (6, 1, H, W)
        if dilate_kernel_size > 1:
            pad = dilate_kernel_size // 2
            x = torch.nn.functional.max_pool2d(x, kernel_size=dilate_kernel_size, stride=1, padding=pad)
            # max_pool2d with even kernels can shift the spatial size; crop back to original H, W.
            x = x[..., : cubemap_mask.shape[1], : cubemap_mask.shape[2]]
        if blur_sigma > 0:
            kernel_size = 2 * int(round(3 * blur_sigma)) + 1
            coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
            kernel_1d = torch.exp(-(coords**2) / (2 * blur_sigma**2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            pad = kernel_size // 2
            # Use 'replicate' padding (not conv2d's default zero-padding) so that the Gaussian
            # blur does not attenuate the mask near face boundaries -- zero-padding would create
            # a dark seam of width ~3*sigma around each cubemap face edge. This is not a true
            # cubemap-aware pad across neighboring faces, but it avoids the artifact.
            x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="replicate")
            x = torch.nn.functional.conv2d(x, kernel_1d.view(1, 1, 1, -1))
            x = torch.nn.functional.pad(x, (0, 0, pad, pad), mode="replicate")
            x = torch.nn.functional.conv2d(x, kernel_1d.view(1, 1, -1, 1))
        return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    def _merge_sky_cubemaps(
        self,
        sky_cubemaps: list[torch.Tensor],
        batch: NRMDataBatch,
        batch_rig_transforms: list[torch.Tensor],
        dilate_kernel_size: int = 31,
        blur_sigma: float = 8.0,
        floor_weight: float = 0.01,
    ) -> torch.Tensor:
        """
        Weighted average of multiple sky cubemaps by sky-mask visibility (determined by per-image sky labels).

        Each per-chunk sky mask is unprojected onto the cubemap and then *softened* by a morphological
        dilation followed by a Gaussian blur. The resulting soft mask (in [0, 1]) is used as the
        per-pixel blending weight, plus a small ``floor_weight`` so that directions without any sky
        observation still fall back to (an average of) the per-chunk sky cubemap predictions.
        """
        merged_sky_cubemap = torch.zeros_like(sky_cubemaps[0])
        merged_sky_cubemap_weights = torch.zeros_like(merged_sky_cubemap)

        for b_idx, (sky_cubemap, batch_rig_transform) in enumerate(
            zip(sky_cubemaps, batch_rig_transforms, strict=True)
        ):
            camera_data = unpack_optional(batch.context[b_idx].data.camera)
            if camera_data.labels is None or camera_data.labels.flags is None:
                merged_sky_cubemap += sky_cubemap
                merged_sky_cubemap_weights += 1.0
                continue

            sky_mask = camera_data.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC).float()
            rendering_data = unpack_optional(unpack_optional(batch.context[b_idx].rendering).camera)
            rig_T = batch_rig_transform.to(
                device=rendering_data.poses_tquat_startend.device, dtype=torch.float32
            )
            global_R_sensors = torch.stack(
                [
                    (rig_T @ tquat_to_se3_matrix(rendering_data.poses_tquat_startend[f, 1, :].unsqueeze(0)))[:3, :3]
                    for f in range(rendering_data.b)
                ],
                dim=0,
            )
            _, cubemap_sky_mask = unproject_to_sky_cubemap(
                merged_sky_cubemap.shape[-2],
                global_R_sensors,
                [
                    cast(ConcreteCameraModelParametersUnion, sensor_model)
                    for sensor_model in rendering_data.sensor_model_parameters
                ],
                sky_mask,
                feature_mask=sky_mask,
            )

            # Soft per-pixel weight in [0, 1], plus a small floor to avoid hard zeros where no
            # chunk observes sky (the per-chunk MLP still predicts a cubemap value there).
            soft_mask = self._soften_cubemap_mask(cubemap_sky_mask, dilate_kernel_size, blur_sigma)
            sky_cubemap_weights = soft_mask + floor_weight
            merged_sky_cubemap += sky_cubemap * sky_cubemap_weights
            merged_sky_cubemap_weights += sky_cubemap_weights

        return merged_sky_cubemap / (merged_sky_cubemap_weights + 1e-8)

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
        self, all_primitives: list[KelvinNRMPrimitive], batch_rig_transforms: list[torch.Tensor], batch: NRMDataBatch
    ) -> KelvinNRMPrimitive:
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

        first_primitive = all_primitives[0]
        merged_primitive = KelvinNRMPrimitive(
            static_layer=KelvinStaticLayer.concatenate(all_static_layers),
            dynamic_layers=[KelvinDynamicLayer.concatenate(all_dynamic_layers)],
            sky_cubemap=merged_sky_cubemap,
            affine_matrix=merged_affine_matrix,
            use_2dgs=first_primitive.use_2dgs,
            gaussians_renderer=first_primitive.gaussians_renderer,
        )
        return merged_primitive

