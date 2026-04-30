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

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace

import torch

import ncore.impl.data.types as ncore_datatypes

from nre.nrm.config.predict import SensorOverrideConfig
from nre.nrm.utils.sensor import to_simple_pinhole_model_parameters
from nre.utils.geometry import pose_offsets_to_se3, se3_matrix_inverse
from nre.utils.types import RigTrajectories


@torch.autocast(device_type="cuda", enabled=False)
def transform_rig_trajectories(
    rig_trajectories: RigTrajectories,
    left_transform: torch.Tensor | None = None,
    right_transform: torch.Tensor | None = None,
) -> RigTrajectories:
    """
    Transform a rig trajectory by applying a left (global) or right (local purturbation) transform.
    Note that when left_transform is applied, the T_world_base is updated inversely so that the geo-positioning is not affected.
    When right_transform is applied, we don't (and cannot) update T_world_base since this is considered as a local perturbation.
    """
    rig_trajectories_device = rig_trajectories.T_world_base.device

    if left_transform is not None:
        assert (
            left_transform.shape == (4, 4)
            and left_transform.dtype in [torch.float32, torch.float64]
            and left_transform.device == rig_trajectories_device
        ), f"Left transform must have shape (4, 4), torch.float32/64 and device {rig_trajectories_device}"
    else:
        left_transform = torch.eye(4, dtype=torch.float64, device=rig_trajectories_device)

    if right_transform is not None:
        assert (
            right_transform.shape == (4, 4)
            and right_transform.dtype in [torch.float32, torch.float64]
            and right_transform.device == rig_trajectories_device
        ), f"Right transform must have shape (4, 4), torch.float32/64 and device {rig_trajectories_device}"
    else:
        right_transform = torch.eye(4, dtype=torch.float64, device=rig_trajectories_device)

    left_transform, right_transform = left_transform.double(), right_transform.double()
    new_rig_trajectories_list = [
        replace(rig_trajectory, T_rig_worlds=left_transform @ rig_trajectory.T_rig_worlds.double() @ right_transform)
        for rig_trajectory in rig_trajectories.rig_trajectories
    ]
    new_T_world_base = rig_trajectories.T_world_base @ se3_matrix_inverse(left_transform)

    return replace(rig_trajectories, rig_trajectories=new_rig_trajectories_list, T_world_base=new_T_world_base)


@torch.autocast(device_type="cuda", enabled=False)
def merge_rig_trajectories(
    rig_trajectories_list: list[RigTrajectories],
) -> tuple[RigTrajectories, dict[tuple[int, int], int]]:
    """
    Merge rig trajectories from multiple chunks into a single long trajectory, also compute the mapping of index
    from multiple data batches into a single data batch.

    Args:
        rig_trajectories_list: List of rig trajectories to merge

    Returns:
        merged_rig_trajectories: Merged rig trajectories
        old_idx_to_new_idx: Mapping of unique_frame_idx. A dictionary mapping from
            (index in input list, input unique_frame_idx) to the new unique_frame_idx in merged trajectory.
    """
    assert len(rig_trajectories_list) > 1, "Fewer than 2 rig trajectories to merge"

    first_trajectories: RigTrajectories = rig_trajectories_list[0]
    target_device = first_trajectories.T_world_base.device
    merged_rig_trajectories: list[RigTrajectories.RigTrajectory] = []

    # Merge the T_rig_worlds measurements
    for traj_idx, rig_trajectory in enumerate(first_trajectories.rig_trajectories):
        time_t_tuple_list: list[tuple[int, torch.Tensor]] = []
        cameras_frame_timestamps_us_list: dict[str, list[torch.Tensor]] = defaultdict(list)
        lidars_frame_timestamps_us_list: dict[str, list[torch.Tensor]] = defaultdict(list)

        # Iterate over all the chunks
        for bidx, other_rig_trajectories in enumerate(rig_trajectories_list):
            other_rig_trajectory = other_rig_trajectories.rig_trajectories[traj_idx]
            assert other_rig_trajectory.sequence_id == rig_trajectory.sequence_id

            # Since rig_trajectory might already contains the full sequence trajectory,
            # We just need to add the missing timestamps.
            T_rig_world_timestamps_us_set = set([t for t, _ in time_t_tuple_list])
            missing_time_T_tuple = [
                (int(t), T)
                for t, T in zip(
                    other_rig_trajectory.T_rig_world_timestamps_us.cpu().numpy().tolist(),
                    other_rig_trajectory.T_rig_worlds,
                )
                if t not in T_rig_world_timestamps_us_set
            ]
            time_t_tuple_list.extend(missing_time_T_tuple)

            # Get related camera and lidar ids for this rig trajectory
            for camera_id, frame_timestamps_us in other_rig_trajectory.cameras_frame_timestamps_us.items():
                cameras_frame_timestamps_us_list[camera_id].append(frame_timestamps_us)
            for lidar_id, frame_timestamps_us in other_rig_trajectory.lidars_frame_timestamps_us.items():
                lidars_frame_timestamps_us_list[lidar_id].append(frame_timestamps_us)

        time_t_tuple_list.sort(key=lambda x: x[0])
        merged_sequence_id = rig_trajectory.sequence_id
        merged_T_rig_worlds = torch.stack([T for _, T in time_t_tuple_list], dim=0).to(torch.float64)
        merged_T_rig_world_timestamps_us = torch.tensor([t for t, _ in time_t_tuple_list], device=target_device).to(
            torch.int64
        )
        merged_cameras_frame_timestamps_us = {
            camera_id: torch.cat(frame_timestamps_us_list, dim=0)
            for camera_id, frame_timestamps_us_list in cameras_frame_timestamps_us_list.items()
        }
        merged_lidars_frame_timestamps_us = {
            lidar_id: torch.cat(frame_timestamps_us_list, dim=0)
            for lidar_id, frame_timestamps_us_list in lidars_frame_timestamps_us_list.items()
        }

        merged_rig_trajectories.append(
            RigTrajectories.RigTrajectory(
                sequence_id=merged_sequence_id,
                rig_bbox=rig_trajectory.rig_bbox,
                cameras_frame_timestamps_us=merged_cameras_frame_timestamps_us,
                lidars_frame_timestamps_us=merged_lidars_frame_timestamps_us,
                T_rig_worlds=merged_T_rig_worlds,
                T_rig_world_timestamps_us=merged_T_rig_world_timestamps_us,
            )
        )

    merged_camera_calibrations = first_trajectories.camera_calibrations
    for other_rig_trajectories in rig_trajectories_list:
        ref_camera_keys = list(merged_camera_calibrations.keys())
        other_camera_keys = list(other_rig_trajectories.camera_calibrations.keys())
        assert ref_camera_keys == other_camera_keys, "Reference camera keys must match"

    merged_T_world_base = first_trajectories.T_world_base
    merged_world_to_nre = first_trajectories.world_to_nre

    # Make sure that T_world_base is the same for all rig trajectories
    for rig_trajectories in rig_trajectories_list:
        assert rig_trajectories.T_world_base.allclose(merged_T_world_base, atol=1e-3), (
            "T_world_base must be the same for all rig trajectories to be merged."
        )

    final_rig_trajectories = RigTrajectories(
        T_world_base=merged_T_world_base,
        world_to_nre=merged_world_to_nre,
        rig_trajectories=merged_rig_trajectories,
        camera_calibrations=merged_camera_calibrations,
        lidar_calibrations=first_trajectories.lidar_calibrations,
    )

    # Compute a mapping of unique_frame_idx
    @dataclass(kw_only=True, frozen=True)
    class UniqueFrameId:
        camera_id: str
        frame_end_timestamp_us: int

    # First iterate through merged trajectory
    # NB [JH]: This has to follow the same order as in CameraFreePoseViewGeometry.from_rig_trajectories
    current_unique_frame_idx: int = 0
    unique_frame_id_to_idx_mapping: dict[UniqueFrameId, int] = {}
    old_idx_to_new_idx: dict[tuple[int, int], int] = {}
    for camera_id in merged_camera_calibrations.keys():
        sequence_id = merged_camera_calibrations[camera_id].sequence_id
        rig_trajectory = [r for r in final_rig_trajectories.rig_trajectories if r.sequence_id == sequence_id][0]
        for frame_end_timestamp_us in rig_trajectory.cameras_frame_timestamps_us[camera_id][:, 1].tolist():
            unique_frame_id_to_idx_mapping[
                UniqueFrameId(camera_id=camera_id, frame_end_timestamp_us=frame_end_timestamp_us)
            ] = current_unique_frame_idx
            current_unique_frame_idx += 1

    # Then iterate through each chunk
    for bidx, other_rig_trajectories in enumerate(rig_trajectories_list):
        current_unique_frame_idx = 0
        for camera_id in other_rig_trajectories.camera_calibrations.keys():
            sequence_id = other_rig_trajectories.camera_calibrations[camera_id].sequence_id
            rig_trajectory = [r for r in other_rig_trajectories.rig_trajectories if r.sequence_id == sequence_id][0]
            for frame_end_timestamp_us in rig_trajectory.cameras_frame_timestamps_us[camera_id][:, 1].tolist():
                old_idx_to_new_idx[(bidx, current_unique_frame_idx)] = unique_frame_id_to_idx_mapping[
                    UniqueFrameId(camera_id=camera_id, frame_end_timestamp_us=frame_end_timestamp_us)
                ]
                current_unique_frame_idx += 1

    return final_rig_trajectories, old_idx_to_new_idx


def keep_only_first_camera(rig_trajectories: RigTrajectories) -> RigTrajectories:
    """
    Keep only the first camera in the rig trajectory (mainly used for visualization purposes).
    """
    main_camera_idx, main_camera_calibration = sorted(
        rig_trajectories.camera_calibrations.items(), key=lambda x: x[1].unique_sensor_idx
    )[0]
    return replace(
        rig_trajectories,
        rig_trajectories=[
            replace(
                rig_trajectory,
                cameras_frame_timestamps_us={
                    main_camera_idx: rig_trajectory.cameras_frame_timestamps_us[main_camera_idx]
                },
            )
            for rig_trajectory in rig_trajectories.rig_trajectories
        ],
        camera_calibrations=OrderedDict([(main_camera_idx, main_camera_calibration)]),
    )


def pad_rig_timestamps(
    rig_trajectories: RigTrajectories, start_timestamp_us: int, end_timestamp_us: int
) -> RigTrajectories:
    """
    Constant padding the timestamps of each rig trajectory (if applicable):
        - T_rig_world_timestamps_us (T_rig_worlds will be constantly padded)
        - {cameras/lidars}_frame_timestamps_us
    """
    new_rig_trajectories: list[RigTrajectories.RigTrajectory] = []
    for rig_trajectory in rig_trajectories.rig_trajectories:
        T_rig_worlds = rig_trajectory.T_rig_worlds
        T_rig_world_timestamps_us = rig_trajectory.T_rig_world_timestamps_us

        # Pad the T_rig_worlds timestamps
        if T_rig_world_timestamps_us[0].item() > start_timestamp_us:
            T_rig_worlds = torch.cat([T_rig_worlds[:1], T_rig_worlds], dim=0)
            T_rig_world_timestamps_us = torch.nn.functional.pad(
                T_rig_world_timestamps_us, (1, 0), mode="constant", value=start_timestamp_us
            )
        if T_rig_world_timestamps_us[-1].item() < end_timestamp_us:
            T_rig_worlds = torch.cat([T_rig_worlds, T_rig_worlds[-1:]], dim=0)
            T_rig_world_timestamps_us = torch.nn.functional.pad(
                T_rig_world_timestamps_us, (0, 1), mode="constant", value=end_timestamp_us
            )

        # Pad the camera timestamps
        cameras_frame_timestamps_us: dict[str, torch.Tensor] = {}
        for camera_id, frame_timestamps_us in rig_trajectory.cameras_frame_timestamps_us.items():
            if frame_timestamps_us[0, 0].item() > start_timestamp_us:
                frame_timestamps_us = torch.nn.functional.pad(
                    frame_timestamps_us, (0, 0, 1, 0), mode="constant", value=start_timestamp_us
                )
            if frame_timestamps_us[-1, -1].item() < end_timestamp_us:
                frame_timestamps_us = torch.nn.functional.pad(
                    frame_timestamps_us, (0, 0, 0, 1), mode="constant", value=end_timestamp_us
                )
            cameras_frame_timestamps_us[camera_id] = frame_timestamps_us

        # Pad the lidar timestamps
        lidars_frame_timestamps_us: dict[str, torch.Tensor] = {}
        for lidar_id, frame_timestamps_us in rig_trajectory.lidars_frame_timestamps_us.items():
            if frame_timestamps_us[0, 0].item() > start_timestamp_us:
                frame_timestamps_us = torch.nn.functional.pad(
                    frame_timestamps_us, (0, 0, 1, 0), mode="constant", value=start_timestamp_us
                )
            if frame_timestamps_us[-1, -1].item() < end_timestamp_us:
                frame_timestamps_us = torch.nn.functional.pad(
                    frame_timestamps_us, (0, 0, 0, 1), mode="constant", value=end_timestamp_us
                )
            lidars_frame_timestamps_us[lidar_id] = frame_timestamps_us

        new_rig_trajectories.append(
            replace(
                rig_trajectory,
                T_rig_worlds=T_rig_worlds,
                T_rig_world_timestamps_us=T_rig_world_timestamps_us,
                cameras_frame_timestamps_us=cameras_frame_timestamps_us,
                lidars_frame_timestamps_us=lidars_frame_timestamps_us,
                # For sanity
                cameras_frame_T_rig_worlds=None,
            )
        )
    return replace(rig_trajectories, rig_trajectories=new_rig_trajectories)


@dataclass(kw_only=True, frozen=True)
class SensorOverride:
    sensor_id: str
    height: int | None
    translation_offset: tuple[float, float, float]
    rotation_offset: tuple[float, float, float]
    rotation_first: bool
    force_pinhole: bool

    @staticmethod
    def from_config(config: SensorOverrideConfig) -> SensorOverride:
        rotation_offset = config.rotation_offset or (0.0, 0.0, 0.0, False)
        return SensorOverride(
            sensor_id=config.sensor_id,
            height=config.height,
            translation_offset=config.translation_offset or (0.0, 0.0, 0.0),
            rotation_offset=rotation_offset[:3],
            rotation_first=rotation_offset[3],
            force_pinhole=config.force_pinhole,
        )

    def apply_camera_model_parameters(
        self, camera_model_parameters: ncore_datatypes.ConcreteCameraModelParametersUnion
    ) -> ncore_datatypes.ConcreteCameraModelParametersUnion:
        if self.height is not None:
            original_width, original_height = camera_model_parameters.resolution.tolist()
            image_scale = self.height / original_height
            image_width = int(original_width * image_scale)
            camera_model_parameters = camera_model_parameters.transform(
                image_domain_scale=image_scale,
                new_resolution=(image_width, self.height),
            )
        if self.force_pinhole:
            camera_model_parameters = to_simple_pinhole_model_parameters(
                camera_model_parameters,
                method="corner",
                reduce="min",
                # Mimic the behavior of nre.viewer.viewpoint.to_simple_pinhole
                percentile=0.01,
            )
        return camera_model_parameters

    def apply_sensor_to_world(self, sensor_to_world: torch.Tensor) -> torch.Tensor:
        device, dtype = sensor_to_world.device, sensor_to_world.dtype
        offset_se3 = pose_offsets_to_se3(
            self.translation_offset,
            self.rotation_offset,
            self.rotation_first,
        )
        return sensor_to_world @ torch.from_numpy(offset_se3).to(device, dtype)
