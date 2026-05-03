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

from collections import defaultdict
from dataclasses import dataclass, replace

import torch

from instant_nurec.utils.geometry import se3_matrix_inverse
from instant_nurec.utils.types import RigTrajectories


@torch.autocast(device_type="cuda", enabled=False)
def transform_rig_trajectories(
    rig_trajectories: RigTrajectories,
    *,
    left_transform: torch.Tensor,
) -> RigTrajectories:
    """Apply a left (global) transform to a rig trajectory; T_world_base is updated inversely so
    geo-positioning is unaffected."""
    rig_trajectories_device = rig_trajectories.T_world_base.device
    assert (
        left_transform.shape == (4, 4)
        and left_transform.dtype in [torch.float32, torch.float64]
        and left_transform.device == rig_trajectories_device
    ), f"Left transform must have shape (4, 4), torch.float32/64 and device {rig_trajectories_device}"

    left_transform = left_transform.double()
    new_rig_trajectories_list = [
        replace(rig_trajectory, T_rig_worlds=left_transform @ rig_trajectory.T_rig_worlds.double())
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
        for other_rig_trajectories in rig_trajectories_list:
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

