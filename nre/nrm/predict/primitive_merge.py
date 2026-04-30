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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Type, cast

import torch  # type: ignore
import torch_scatter  # type: ignore

from ncore.data import ConcreteCameraModelParametersUnion  # type: ignore
from ncore.sensors import CameraModel  # type: ignore
from nre.nrm.config.models import PrimitiveExportPreprocessConfig
from nre.nrm.config.predict import PrimitiveMergeConfig
from nre.nrm.primitives.base import BaseNRMPrimitive, NRMPrimitiveType
from nre.nrm.primitives.celsius_primitive import CelsiusNRMPrimitive
from nre.nrm.primitives.kelvin_primitive import KelvinDynamicLayer, KelvinNRMPrimitive, KelvinStaticLayer
from nre.nrm.utils.covariance import merge_covariances_kl_optimal
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
    timestamps_startend_us: torch.Tensor
    # CPU copy to avoid GPU->CPU sync when calling .item()
    timestamps_startend_us_cpu: torch.Tensor = field(init=False)

    def __post_init__(self):
        # Store CPU copy after initialization
        object.__setattr__(self, "timestamps_startend_us_cpu", self.timestamps_startend_us.cpu())

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

    @property
    def end_timestamp_us(self) -> int:
        # Use stored CPU copy to avoid GPU->CPU sync
        return int(self.timestamps_startend_us_cpu[1].item())


def voxelize_with_fusion(
    pts3d: torch.Tensor,
    features: dict[str, torch.Tensor],
    voxel_size: float,
    conf: torch.Tensor | None = None,
    fusion_mode: str = "average",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Voxelize points and features using confidence-weighted fusion.

    Args:
        pts3d: Point positions [N, 3]
        features: Dictionary of feature vectors [N, F]
        voxel_size: Size of voxels
        conf: Confidence scores [N, 1] or None
        fusion_mode: 'average' for weighted averaging of all attributes (existing behavior),
            'kl_optimal' for moment-matching of position/rotation/scale.

    Returns:
        voxel_pts: Voxelized positions [M, 3]
        voxel_feats: Dictionary of voxelized features [M, F]
    """
    if conf is None:
        conf = torch.ones(pts3d.shape[0], 1, device=pts3d.device, dtype=pts3d.dtype)

    # Drop Gaussians with NaN/inf in positions, scales, or rotations.
    valid = torch.isfinite(pts3d).all(dim=1) & torch.isfinite(conf).all(dim=1)
    for feat in features.values():
        if feat is not None:
            valid = valid & torch.isfinite(feat).all(dim=1)
    n_bad = (~valid).sum().item()
    if n_bad > 0:
        import logging

        logging.getLogger(__name__).warning(
            "[voxelize_with_fusion] %d/%d Gaussians have NaN/inf; dropping.", n_bad, pts3d.shape[0]
        )
        pts3d = pts3d[valid]
        conf = conf[valid]
        features = {k: v[valid] if v is not None else None for k, v in features.items()}

    # Compute voxel indices
    voxel_indices = (pts3d / voxel_size).round().int()  # [N, 3]
    unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)

    # Flatten confidence scores
    conf_flat = conf.flatten()  # [N]

    # Compute softmax weights per voxel
    conf_voxel_max, _ = torch_scatter.scatter_max(conf_flat, inverse_indices, dim=0)
    conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
    voxel_weights = torch_scatter.scatter_add(conf_exp, inverse_indices, dim=0)  # [num_unique_voxels]
    weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)  # [N, 1]

    # Compute weighted average of positions
    voxel_pts = torch_scatter.scatter_add(pts3d * weights, inverse_indices, dim=0)  # [num_unique_voxels, 3]

    if fusion_mode == "kl_optimal" and "rotations" in features and "scales" in features:
        # KL-optimal merge for spatial parameters
        rotations_wxyz = features["rotations"]
        scales = features["scales"]
        rotations_merged, scales_merged = merge_covariances_kl_optimal(
            pts3d, rotations_wxyz, scales, weights, inverse_indices, voxel_pts
        )
        # Average remaining features
        voxel_feats: dict[str, torch.Tensor] = {}
        for feat_name, feat in features.items():
            if feat_name in ("rotations", "scales"):
                continue
            if feat is None:
                voxel_feats[feat_name] = None  # type: ignore[assignment]
                continue
            voxel_feats[feat_name] = torch_scatter.scatter_add(feat * weights, inverse_indices, dim=0)
        voxel_feats["rotations"] = rotations_merged
        voxel_feats["scales"] = scales_merged
    else:
        # Average all features
        voxel_feats = {}
        for feat_name, feat in features.items():
            if feat is None:
                voxel_feats[feat_name] = None  # type: ignore[assignment]
                continue
            voxel_feats[feat_name] = torch_scatter.scatter_add(
                feat * weights, inverse_indices, dim=0
            )  # [num_unique_voxels, feat_dim]

    if "rotations" in voxel_feats:
        voxel_feats["rotations"] = torch.nn.functional.normalize(voxel_feats["rotations"], dim=1)

    return voxel_pts, voxel_feats


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
        for frame_idx in range(rendering_data.b):
            global_T_sensor = batch_rig_transforms[b_idx].float() @ tquat_to_se3_matrix(
                rendering_data.poses_tquat_startend[frame_idx]
            )
            camera_model_parameters = cast(
                ConcreteCameraModelParametersUnion, rendering_data.sensor_model_parameters[frame_idx]
            )
            camera_frustums.append(
                CameraFrustum(
                    camera_model=CameraModel.from_parameters(camera_model_parameters),
                    poses_T_startend=global_T_sensor,
                    timestamps_startend_us=rendering_data.timestamps_startend_us[frame_idx],
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


class PrimitiveMerge(ABC, Generic[NRMPrimitiveType]):
    """
    Merge primitives from non-overlapping chunks into a single primitive.
    """

    def __init__(
        self,
        config: PrimitiveMergeConfig,
        export_preprocess_config: PrimitiveExportPreprocessConfig | None = None,
    ):
        self.config = config
        self.export_preprocess_config = export_preprocess_config

    @abstractmethod
    def merge_processed_primitives(
        self, all_primitives: list[NRMPrimitiveType], batch_rig_transforms: list[torch.Tensor], batch: NRMDataBatch
    ) -> NRMPrimitiveType:
        """
        Merge the processed primitives into a single primitive. Allows to modify all_primitives in place.
        all_primitives are already transformed by batch_rig_transforms, but batch is not.
        """

    @abstractmethod
    def postprocess_merged_primitive(self, merged_primitive: NRMPrimitiveType) -> None:
        """
        Postprocess the merged primitive such as voxelization.
        """

    @torch.autocast(device_type="cuda", enabled=False)
    def merge_primitives_and_batch(
        self,
        primitives_list: list[NRMPrimitiveType],
        batch: NRMDataBatch,
    ) -> tuple[NRMPrimitiveType, NRMDataBatch]:
        """
        Merge primitives from non-overlapping chunks into a single primitive.

        Args:
            chunked_primitives: List of primitive lists from each chunk
            batch: NRM data batch containing context information

        Returns:
            A single merged primitive and batch
        """
        assert len(primitives_list) > 0, "No primitives to merge"
        logger.info(f"Merging {len(primitives_list)} chunks ({sum(len(p) for p in primitives_list)} Gaussians)")

        # Stage 1: Transform each primitive into the reference frame (first chunk) so they can be concatenated.
        # Rigid transform is only needed when merging: per-chunk preprocess does not change frame; merging requires a common frame.
        batch_context_rig: list[RigTrajectories] = unpack_optional(batch.context_rig)
        T_world_ref: torch.Tensor = se3_matrix_inverse(batch_context_rig[0].T_world_base)
        batch_rig_transforms: list[torch.Tensor] = [T_world_ref @ cr.T_world_base for cr in batch_context_rig]
        for b_idx, primitive in enumerate(primitives_list):
            primitives_list[b_idx] = primitive.rigid_transform(batch_rig_transforms[b_idx].float())

        # Step 2: Merge the processed primitives into a single primitive
        merged_primitive = self.merge_processed_primitives(primitives_list, batch_rig_transforms, batch)

        # Step 3: Postprocess the merged primitive
        self.postprocess_merged_primitive(merged_primitive)

        logger.info(f"Merged {len(primitives_list)} primitives into {repr(merged_primitive)}")

        # Step 4: Merge batch and meta
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

            # process meta data from all the chunks
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


primitive_mergers: dict[Type[BaseNRMPrimitive], Type[PrimitiveMerge[BaseNRMPrimitive]]] = {}


def register(primitive_type: Type[BaseNRMPrimitive]):
    def decorator(cls):
        primitive_mergers[primitive_type] = cls
        return cls

    return decorator


def make(
    primitive_type: Type[BaseNRMPrimitive],
    config: PrimitiveMergeConfig,
    export_preprocess_config: PrimitiveExportPreprocessConfig | None = None,
) -> PrimitiveMerge[BaseNRMPrimitive]:
    return primitive_mergers[primitive_type](config, export_preprocess_config)


@register(CelsiusNRMPrimitive)
class CelsiusPrimitiveMerge(PrimitiveMerge[CelsiusNRMPrimitive]):
    """
    Merge Celsius primitives from non-overlapping chunks into a single primitive.
    """

    def _probe_first_in_frustum_timestamp(
        self,
        positions: torch.Tensor,
        current_timestamps_us: torch.Tensor,
        init_inds: torch.Tensor,
        camera_frustums: list[CameraFrustum],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions[init_inds]
        for camera_frustum in camera_frustums:
            if init_inds.numel() == 0:
                break

            # NB [JH]: Here we use the simple in_frustum check to see if a Gaussian
            # badly benefit the other chunk (so this is a bit too conservative).
            # In the future we should think about better ways such as reprojection loss.
            in_mask = camera_frustum.in_frustum(positions)
            current_timestamps_us[init_inds[in_mask]] = camera_frustum.end_timestamp_us
            init_inds = init_inds[~in_mask]
            positions = positions[~in_mask]

        return current_timestamps_us, init_inds

    def merge_processed_primitives(
        self, all_primitives: list[CelsiusNRMPrimitive], batch_rig_transforms: list[torch.Tensor], batch: NRMDataBatch
    ) -> CelsiusNRMPrimitive:
        # Merged output must have sky: either render via token or keep sky Gaussians.
        assert self.export_preprocess_config is not None, (
            "Celsius merge requires export_preprocess_config (model.export_preprocess) to be passed to make()."
        )
        keep_sky = getattr(self.export_preprocess_config, "keep_sky_gaussians", False)
        assert self.config.enable_sky_mask or keep_sky, (
            "When merging Celsius primitives, either primitive_merge.enable_sky_mask or "
            "model.export_preprocess.keep_sky_gaussians must be True (merged output must have sky)."
        )
        # Simplified case: no overlap strategy or only one primitive
        if self.config.overlap_strategy == "none" or len(all_primitives) == 1:
            return all_primitives[0]

        # The strategies will keep the Gaussians static.
        if self.config.overlap_strategy in ["depth_truncation", "frustum_ownership"]:
            for b_idx, primitive in enumerate(all_primitives):
                if primitive.falloff_sigma is None:  # Model has no motion module
                    dynamic_mask = torch.zeros_like(
                        unpack_optional(primitive.positions)[:, 0],
                        dtype=torch.bool,
                        device=unpack_optional(primitive.positions).device,
                    )
                else:
                    dynamic_mask = primitive.falloff_sigma[:, 0] < self.config.dynamic_sigma_threshold
                    # Elongate timespan for static GS
                    primitive.falloff_sigma[~dynamic_mask] = 1e5

                # Use depth truncation to avoid overlap (in the primitive-local space) for all but last primitive
                if self.config.overlap_strategy == "depth_truncation":
                    # Always keep dynamic GS
                    if b_idx != len(all_primitives) - 1:
                        gaussians_mask = torch.logical_or(
                            dynamic_mask, primitive.positions[:, 2] < self.config.depth_truncation_threshold
                        )
                        all_primitives[b_idx] = primitive.mask(gaussians_mask)

                elif self.config.overlap_strategy == "frustum_ownership":
                    batch_camera_frustums = build_world_camera_frustums(batch, batch_rig_transforms)
                    gaussians_mask = torch.logical_or(
                        dynamic_mask,
                        compute_frustum_ownership_mask(
                            b_idx, primitive.positions, batch_camera_frustums, self.config.frustum_ownership_max_diff_m
                        ),
                    )
                    all_primitives[b_idx] = primitive.mask(gaussians_mask)

        elif self.config.overlap_strategy == "two_sigma":
            # Determine an updated version of sigma_falloff (2-sides) based on its observability in other frustums
            # so we make sure that the Gaussians from one chunk are never visible in other chunks.
            batch_camera_frustums = build_world_camera_frustums(batch, batch_rig_transforms)

            # Then iterate through all the primitives, trying to extend the falloff sigma if observability check passes
            for b_idx, primitive in enumerate(all_primitives):
                assert primitive.falloff_sigma is not None, "Falloff sigma must be present"

                gaussians_timestamps_us = primitive.timestamps_us[:, 0]
                init_falloff_sigma = primitive.falloff_sigma[:, 0]
                assert primitive.falloff_sigma.shape[1] == 1, "The process can only handle 1-side falloff sigma"
                init_inds = torch.where(init_falloff_sigma > 1.0)[0]  # Those that are considered as static.

                # Probe frames that are after this batch
                gaussians_next_fade = gaussians_timestamps_us + (init_falloff_sigma * 1e6).long()
                gaussians_next_fade, forever_inds = self._probe_first_in_frustum_timestamp(
                    primitive.positions,
                    gaussians_next_fade,
                    init_inds,
                    sorted(sum(batch_camera_frustums[b_idx + 1 :], []), key=lambda x: x.end_timestamp_us),
                )
                gaussians_next_fade[forever_inds] += int(20 * 1e6)  # Extend 20s so it always exists

                # Probe frames that are before this batch
                gaussians_prev_fade = gaussians_timestamps_us - (init_falloff_sigma * 1e6).long()
                gaussians_prev_fade, forever_inds = self._probe_first_in_frustum_timestamp(
                    primitive.positions,
                    gaussians_prev_fade,
                    init_inds,
                    sorted(sum(batch_camera_frustums[:b_idx], []), key=lambda x: x.end_timestamp_us, reverse=True),
                )
                gaussians_prev_fade[forever_inds] -= int(20 * 1e6)  # Extend 20s so it always exists

                primitive.falloff_sigma = (
                    torch.stack(
                        [
                            gaussians_next_fade - gaussians_timestamps_us,
                            gaussians_timestamps_us - gaussians_prev_fade,
                        ],
                        dim=1,
                    ).float()
                    / 1e6
                )

        first_primitive = all_primitives[0]
        merged_primitive = CelsiusNRMPrimitive.concatenate_gaussians(all_primitives)

        # Concatenate the GS will remove other attributes as well, add them using the first primitive
        for attr_name in first_primitive.OTHER_ATTRIBUTES_NAMES:
            setattr(merged_primitive, attr_name, getattr(first_primitive, attr_name))
        merged_primitive.set_sky_mask_enabled(self.config.enable_sky_mask)

        return merged_primitive

    @staticmethod
    def _dense_merging_postprocessing(
        merged_primitive: BaseNRMPrimitive, n_primitives: int, mode: str, speed_threshold_mps: float
    ) -> None:
        """
        Postprocess merged primitive in place when one wants to merge primitives densely.
        This is currently not used in any places, but kept for future reference.
        """
        assert isinstance(merged_primitive, CelsiusNRMPrimitive), "Merged primitive must be a CelsiusNRMPrimitive"

        match mode:
            case "mean_densities":
                if merged_primitive.densities is not None:
                    merged_primitive.densities = merged_primitive.densities / n_primitives

            case "faster_falloff":
                if merged_primitive.falloff_sigma is not None:
                    merged_primitive.falloff_sigma = merged_primitive.falloff_sigma / n_primitives

            case "hybrid":
                # Implement hybrid strategy based on forward_speed_mps
                if merged_primitive.forward_speed_mps is not None:
                    # Calculate speed magnitude for each Gaussian
                    speed_magnitude = torch.norm(
                        merged_primitive.forward_speed_mps, dim=1, keepdim=True
                    )  # [n_gaussians, 1]

                    # Create masks for static and dynamic Gaussians
                    static_mask = speed_magnitude <= speed_threshold_mps  # [n_gaussians, 1]
                    dynamic_mask = speed_magnitude > speed_threshold_mps  # [n_gaussians, 1]

                    # Apply mean_densities strategy to static Gaussians
                    if torch.any(static_mask):
                        merged_primitive.densities[static_mask] = merged_primitive.densities[static_mask] / n_primitives

                    # Apply faster_falloff strategy to dynamic Gaussians
                    if torch.any(dynamic_mask) and merged_primitive.falloff_sigma is not None:
                        merged_primitive.falloff_sigma[dynamic_mask] = (
                            merged_primitive.falloff_sigma[dynamic_mask] / n_primitives
                        )
                else:
                    # If no forward_speed_mps, treat all as static and use mean_densities
                    merged_primitive.densities = merged_primitive.densities / n_primitives

            case "none":
                pass

            case _:
                raise ValueError(f"Invalid mode: {mode}")

    def postprocess_merged_primitive(self, merged_primitive: CelsiusNRMPrimitive) -> None:
        # Optionally voxelize the attributes
        if self.config.enable_voxelization:
            merged_gaussian_positions = merged_primitive.positions
            merged_gaussian_features = {
                feat_name: getattr(merged_primitive, feat_name)
                for feat_name in merged_primitive.GAUSSIAN_ATTRIBUTES_NAMES
                if feat_name != "positions"
            }
            merged_gaussian_positions, merged_gaussian_features = voxelize_with_fusion(
                merged_gaussian_positions,
                merged_gaussian_features,
                self.config.voxel_size,
                None,
                fusion_mode=self.config.voxel_fusion_mode,
            )
            merged_primitive.positions = merged_gaussian_positions
            for feat_name, feat in merged_gaussian_features.items():
                setattr(merged_primitive, feat_name, feat)

            # Re-booleanize mask attributes that were turned to float by weighted averaging.
            for mask_name in ("dynamic_bbox_mask", "road_mask"):
                mask_val = getattr(merged_primitive, mask_name, None)
                if mask_val is not None and mask_val.is_floating_point():
                    setattr(merged_primitive, mask_name, mask_val > 0.5)


@register(KelvinNRMPrimitive)
class KelvinPrimitiveMerge(PrimitiveMerge[KelvinNRMPrimitive]):
    """
    Merge Kelvin primitives from non-overlapping chunks into a single primitive.
    """

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
            global_R_sensors = torch.stack(
                [
                    (
                        batch_rig_transform.float()
                        @ tquat_to_se3_matrix(rendering_data.poses_tquat_startend[f, 1, :].unsqueeze(0))
                    )[:3, :3]
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
        # Merged output must have sky (Kelvin has no keep_sky_gaussians; require enable_sky_mask).
        assert self.config.enable_sky_mask, (
            "When merging Kelvin primitives, primitive_merge.enable_sky_mask must be True (merged output must have sky)."
        )
        # Simplified case: no overlap strategy or only one primitive
        if self.config.overlap_strategy == "none" or len(all_primitives) == 1:
            return all_primitives[0]

        assert self.config.overlap_strategy == "frustum_ownership", (
            "Only frustum ownership strategy is supported for Kelvin model"
        )

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

    def postprocess_merged_primitive(self, merged_primitive: KelvinNRMPrimitive) -> None:
        # Voxelization is applied on static layer only, with equal confidence for now.
        if self.config.enable_voxelization:
            merged_primitive.static_layer = merged_primitive.static_layer.voxelize(
                self.config.voxel_size, None, fusion_mode=self.config.voxel_fusion_mode
            )
