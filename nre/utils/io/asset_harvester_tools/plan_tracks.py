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

import json
import logging
import math
import os

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger(__name__)

import json as jsonlib
import re

import numpy as np
import torch

from omegaconf import OmegaConf

from nre.config.asset_harvest import AssetHarvestingConfig
from nre.config.dataset import NCoreDatasetConfig
from nre.config.parse import parse_untyped_config
from nre.datasets.ncore import NCOREDataSource


@dataclass
class TrackPlan:
    track_id: str
    label_source: str
    scene_id: str
    label_class: str
    min_distance_m: float
    min_distance_ego_timestamp_us: int
    min_distance_ego_pose_idx: int
    min_distance_camera_id: str | None
    min_distance_camera_frame_idx: int | None


@dataclass
class ScenePlan:
    scene_id: str
    shard: str
    tracks: list[TrackPlan]


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _min_distance_over_time(ego_positions_m: np.ndarray, obj_positions_m: np.ndarray) -> float:
    # ego_positions_m: (T, 3), obj_positions_m: (K, 3) with matched timestamps. For simplicity, use nearest neighbors over time.
    if ego_positions_m.size == 0 or obj_positions_m.size == 0:
        return float("inf")
    # Use broadcast to compute pairwise distances and take min along both axes
    diff = ego_positions_m[:, None, :] - obj_positions_m[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return float(dists.min())


def _get_ego_xyz_world(rig_trajectories) -> np.ndarray:
    # rig_trajectories: RigTrajectories for a single sequence; use its single RigTrajectory
    assert len(rig_trajectories.rig_trajectories) == 1, "Expected single-sequence dataset"
    T = rig_trajectories.rig_trajectories[0].T_rig_worlds.numpy()  # (N, 4, 4)
    return T[:, :3, 3]  # (N, 3)


def _get_ego_timestamps_us(rig_trajectories) -> np.ndarray:
    return rig_trajectories.rig_trajectories[0].T_rig_world_timestamps_us.cpu().numpy().astype(np.int64)


def _get_tracks_world_positions_from_datasource(datasource) -> tuple[list[str], list[np.ndarray]]:
    # World-frame cuboid tracks
    cuboids = datasource.get_cuboid_tracks(dynamic_only=False, world_frame=True)
    track_ids = cuboids.tracks_id
    # Extract per-track world positions at each stored pose timestamp
    poses_se3 = cuboids.tracks_poses.matrix().cpu().numpy()  # (N_total_poses, 4, 4)
    packinfo = cuboids.tracks_packinfo.cpu().numpy()  # (N_tracks, 2)
    xyz_tracks: list[np.ndarray] = []
    for start, count in packinfo:
        if count <= 0:
            xyz_tracks.append(np.zeros((0, 3), dtype=np.float32))
            continue
        P = poses_se3[start : start + count]
        xyz_tracks.append(P[:, :3, 3].astype(np.float32))
    return track_ids, xyz_tracks


def _get_tracks_timestamps_us_from_datasource(datasource) -> tuple[list[str], list[np.ndarray]]:
    cuboids = datasource.get_cuboid_tracks(dynamic_only=False, world_frame=True)
    track_ids = cuboids.tracks_id
    ts = cuboids.tracks_timestamps_us.cpu().numpy().astype(np.int64)
    packinfo = cuboids.tracks_packinfo.cpu().numpy()
    ts_tracks: list[np.ndarray] = []
    for start, count in packinfo:
        if count <= 0:
            ts_tracks.append(np.zeros((0,), dtype=np.int64))
            continue
        ts_tracks.append(ts[start : start + count])
    return track_ids, ts_tracks


def _min_distance_all_pairs(ego_xyz: np.ndarray, track_xyz: np.ndarray) -> tuple[float, int, int]:
    # Returns: (min_d, ego_idx, track_idx) using full pairwise scan
    if ego_xyz.size == 0 or track_xyz.size == 0:
        return float("inf"), -1, -1
    # Efficient pairwise distance min: split to blocks to avoid huge memory, but here arrays are moderate; do simple loop
    min_d = float("inf")
    best_e_idx = -1
    best_t_idx = -1
    for i in range(track_xyz.shape[0]):
        diff = ego_xyz - track_xyz[i][None, :]
        dists = np.linalg.norm(diff, axis=1)
        j = int(dists.argmin())
        d = float(dists[j])
        if d < min_d:
            min_d = d
            best_e_idx = j
            best_t_idx = i
    return min_d, best_e_idx, best_t_idx


def _find_camera_frame_for_ego_timestamp(rig_trajectories, ego_ts_us: int) -> tuple[str | None, int | None]:
    rt = rig_trajectories.rig_trajectories[0]
    cam_frames = rt.cameras_frame_timestamps_us
    for cam_id, arr in cam_frames.items():
        # arr shape (N,2): start,end
        starts = arr[:, 0]
        ends = arr[:, 1]
        # Handle case where starts may be -1 if end_frame_timestamps_only was used
        mask = (starts <= ego_ts_us) & (ego_ts_us <= ends)
        idx = np.where(mask)[0]
        if idx.size > 0:
            return cam_id, int(idx[0])
    return None, None


def generate_harvest_plan(
    *,
    dataset_path: str,
    camera_ids: list[str],
    train_camera_ids: list[str],
    lidar_ids: list[str],
    radius_m: float = 3.0,
    min_visible_frames: int = 5,
    label_source: str = "scene:obstacles:autolabels:v2",
    harvest_config_name: str = "configs/experimental/asset_harvesting/harvest.yaml",
    dataset_config_name: str = "configs/dataset/ncore.yaml",
    output_dir: Optional[str] | None = None,
    write_output: bool = True,
) -> tuple[list[ScenePlan], Optional[str]]:
    """Generate a harvest plan programmatically.

    Returns the in-memory plan and, if write_output is True, the path to the
    written JSON file.
    """
    # Parse harvest config to get the list of harvestable classes
    harvest_hydra_overrides = [
        "+logger.save_dir=.",
        "+save_dir=.",
        "+ckpt_dir=.",
        "+config_dir=.",
    ]

    harvest_untyped = parse_untyped_config(config_name=harvest_config_name, hydra_args=harvest_hydra_overrides)
    harvest_config = AssetHarvestingConfig.model_validate(harvest_untyped)
    harvestable_classes = None

    logger.info(f"No class filtering applied (crop_labels removed in GA migration)")

    # Use the base ncore dataset config and override specific fields
    dataset_hydra_overrides = [
        "+logger.save_dir=.",
        "+save_dir=.",
        "+ckpt_dir=.",
        "+config_dir=.",
        f"dataset.path={dataset_path}",
        f"dataset.camera_ids=[{','.join(camera_ids)}]",
        f"dataset.train_camera_ids=[{','.join(train_camera_ids)}]",
        f"dataset.val_camera_ids=[{','.join(camera_ids)}]",
        f"dataset.lidar_ids=[{','.join(lidar_ids)}]",
        f"dataset.train_lidar_ids=[{','.join(lidar_ids)}]",
        f"dataset.val_lidar_ids=[{','.join(lidar_ids)}]",
        f"dataset.val_lidar={len(lidar_ids) > 0}",
        f"dataset.train_sequential_lidar={len(lidar_ids) > 0}",
        "dataset.generate_traffic_light_cuboid_tracks.enabled=false",
        "dataset.valid_measurements_method=EGO",
    ]

    untyped = parse_untyped_config(config_name=dataset_config_name, hydra_args=dataset_hydra_overrides)
    dataset_dict = OmegaConf.to_container(untyped.dataset, resolve=True)
    dataset_cfg = NCoreDatasetConfig.model_validate(dataset_dict)
    datasource = NCOREDataSource(dataset_cfg, lidar_model_config=None)

    # Use harvestable classes from harvest config instead of all dynamic classes
    classes = harvestable_classes
    logger.info(f"Filtering tracks by harvestable classes: {classes}")

    ds_path = dataset_cfg.path
    ds_dir = os.path.dirname(ds_path)
    out_dir = output_dir if output_dir else ds_dir
    logger.info(f"Dataset path: {ds_path}")
    logger.info(f"Output dir: {out_dir}")
    logger.info("Selecting tracks by radius and visibility...")

    plan = plan_from_datasource(
        datasource,
        radius_m=radius_m,
        include_classes=classes,
        min_visible_frames=min_visible_frames,
        label_source=label_source,
        dataset_path=ds_path,
    )

    scene_id = plan[0].scene_id if len(plan) else "scene"
    plan_json_out = os.path.join(out_dir, f"harvest_plan_{scene_id}.json")

    if write_output:
        Path(os.path.dirname(plan_json_out)).mkdir(parents=True, exist_ok=True)
        with open(plan_json_out, "w") as f:
            json.dump([asdict(sp) for sp in plan], f, indent=2)
        logger.info(f"Plan written to {plan_json_out}")

    return plan, (plan_json_out if write_output else None)


def plan_from_datasource(
    datasource,
    *,
    radius_m: float,
    include_classes: list[str] | None,
    min_visible_frames: int,
    label_source: str,
    dataset_path: str,
) -> list[ScenePlan]:
    scene_id = datasource.sequence_id
    # Get full frame ranges to enable camera frame lookup
    rig_traj = datasource.get_rig_trajectories()
    ego_xyz_w = _get_ego_xyz_world(rig_traj)
    ego_ts = _get_ego_timestamps_us(rig_traj)

    track_ids, xyz_tracks = _get_tracks_world_positions_from_datasource(datasource)

    # Get all cuboid tracks (not just dynamic) since we'll filter by harvestable classes
    cuboids = datasource.get_cuboid_tracks(dynamic_only=False, world_frame=True)
    label_classes = cuboids.tracks_label_class
    # Try to build a class name -> id map (use any available consistent semantic mapping)
    try:
        datasource.get_semantic_classes_map(camera_semantics=True, lidar_semantics=False)
    except (AttributeError, KeyError) as e:
        logger.debug(f"Camera semantic map not available: {e}")

    selected: list[tuple[str, float, int, int | None, str | None]] = []
    for tid, xyz, cls in zip(track_ids, xyz_tracks, label_classes):
        if include_classes is not None and cls.lower() not in include_classes:
            continue
        if len(xyz) < min_visible_frames:
            continue
        d, eidx, _ = _min_distance_all_pairs(ego_xyz_w, xyz)
        ets = int(ego_ts[eidx]) if eidx >= 0 else -1
        cam_id, cam_fidx = _find_camera_frame_for_ego_timestamp(rig_traj, ets) if ets >= 0 else (None, None)
        if d <= radius_m:
            selected.append((str(tid), d, ets, cam_fidx if cam_fidx is not None else -1, cam_id))

    tracks_plans: list[TrackPlan] = []
    for tid, dmin, ets, cam_fidx, cam_id in selected:
        idx = track_ids.index(tid)
        cls_name = label_classes[idx] if idx < len(label_classes) else ""
        tracks_plans.append(
            TrackPlan(
                track_id=str(tid),
                label_source=label_source,
                scene_id=scene_id,
                label_class=cls_name,
                min_distance_m=float(dmin),
                min_distance_ego_timestamp_us=int(ets),
                min_distance_ego_pose_idx=int(np.searchsorted(ego_ts, ets)) if ets >= 0 else -1,
                min_distance_camera_id=cam_id,
                min_distance_camera_frame_idx=int(cam_fidx) if cam_fidx is not None and cam_fidx >= 0 else None,
            )
        )

    return [ScenePlan(scene_id=scene_id, shard=str(dataset_path), tracks=tracks_plans)]
