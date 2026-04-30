# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Object data loader for object-level metrics evaluation.

This module provides the ObjectDataLoader class for loading and managing
data required for object-level metrics computation.
"""

import glob
import json
import logging
import os

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from nre.benchmark.utils.bbox_projector import BBoxProjector
from nre.benchmark.utils.shard_data_manager import ShardDataManager


log = logging.getLogger(__name__)


def load_timestamps_file(filepath: str) -> List[int]:
    """Load timestamps from file (supports 'frame_idx timestamp' or 'timestamp' format).

    Args:
        filepath: Path to timestamps file.

    Returns:
        List of timestamps.
    """
    timestamps: List[int] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    try:
                        timestamps.append(int(parts[-1]))
                    except ValueError:
                        log.warning("Skipping invalid timestamp: %s", line.strip())
    except OSError as e:
        log.warning("Could not read timestamps file %s: %s", filepath, e)
    return timestamps


def normalize_track_id(track_id: Any) -> str:
    """Normalize track ID to a consistent string representation.

    Handles multiple formats:
    - Numeric: 35382.0 -> "35382", 35382 -> "35382"
    - String floats: "35382.0" -> "35382"
    - Autolabel format: "255@scene:obstacles:..." -> "255"
    - Other strings: kept as-is

    Args:
        track_id: Track ID in any format (float, int, str).

    Returns:
        Normalized string representation (integer if possible).
    """
    # Handle numeric types
    if isinstance(track_id, float) and track_id.is_integer():
        return str(int(track_id))
    if isinstance(track_id, (int, float)):
        return str(track_id)

    # Convert to string for pattern matching
    str_id = str(track_id)

    # Handle string floats like "35518.0" -> "35518"
    if str_id.endswith(".0"):
        try:
            return str(int(float(str_id)))
        except ValueError:
            pass

    # Handle "255@scene:obstacles:autolabels:v2:" -> "255"
    if "@" in str_id:
        prefix = str_id.split("@")[0]
        if prefix.isdigit():
            return prefix

    return str_id


def load_camera_offset_json(filepath: str) -> Tuple[float, float, float]:
    """Load camera offset from NDAS JSON file.

    Args:
        filepath: Path to JSON file with keys tx_m, ty_m, tz_m.

    Returns:
        Tuple of (x_offset, y_offset, z_offset) in meters.

    Raises:
        ValueError: On file/parse/key errors.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support nested structure (e.g., data["offset"]) or flat
        if "offset" in data:
            data = data["offset"]
        # Require keys to exist
        return (float(data["tx_m"]), float(data["ty_m"]), float(data["tz_m"]))
    except (OSError, json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to load camera offset from {filepath}: {e}") from e


class ObjectDataLoader:
    """Loads and manages data for object-level metrics evaluation.

    This class is responsible for:
    1. Loading shard data (3D labels, camera/LiDAR sensors)
    2. Loading rendered video frames
    3. Pre-loading GT frames from shards
    4. Extracting object crops from frames
    5. Providing data access methods for metric computation

    It does NOT compute metrics - that's handled by MetricManager.
    """

    def __init__(
        self,
        shard_pattern: str,
        camera_id: str,
        device: str = "cuda",
        crop_padding_ratio: float = 0.0,
        min_bbox_size: int = 50,
    ):
        """Initialize object data loader.

        Args:
            shard_pattern: Glob pattern for shard files
            camera_id: Camera ID to process
            device: Compute device ('cuda' or 'cpu')
            crop_padding_ratio: Padding ratio around bounding boxes
            min_bbox_size: Minimum bbox size in pixels
        """
        self.device = device
        self.crop_padding_ratio = crop_padding_ratio
        self.min_bbox_size = min_bbox_size
        self.camera_id = camera_id

        # Data containers
        self.shard_manager: Optional[ShardDataManager] = None
        self.labels_3d: Optional[Dict] = None
        self._rendered_video_path: Optional[str] = None
        self._num_frames: int = 0
        self._bbox_projector: Optional[BBoxProjector] = None
        self._x_offset: float = 0.0
        self._y_offset: float = 0.0
        self._z_offset: float = 0.0
        self._max_frames: Optional[int] = None
        self._gt_subsample_frames: int = 1
        self._rendered_timestamps: Optional[List[int]] = None
        self._video_cap: Optional[cv2.VideoCapture] = None

        # Load shard data
        log.info("Initializing ObjectDataLoader")
        log.info("  Camera: %s", camera_id)
        log.info("  Crop padding ratio: %f", crop_padding_ratio)
        log.info("  Min bbox size: %d", min_bbox_size)

        self._load_shard(shard_pattern)

    def _load_shard(self, shard_pattern: str) -> None:
        """Load shard data and compile 3D labels.

        Args:
            shard_pattern: Glob pattern for shard files
        """
        log.info("Loading shard data...")
        log.info("  Shard pattern: %s", shard_pattern)

        shard_files = glob.glob(shard_pattern)
        if not shard_files:
            raise ValueError(f"No shard files found: {shard_pattern}")

        # Initialize shard manager
        self.shard_manager = ShardDataManager(shard_pattern, self.camera_id)

        # Compile 3D labels
        log.info("Compiling 3D labels...")
        self.labels_3d = self.shard_manager.compile_3d_labels()
        log.info("  Compiled labels for %d frames", len(self.labels_3d))

        # Create bbox projector
        self._bbox_projector = BBoxProjector(
            self.shard_manager.camera_model,
            self.shard_manager.camera_sensor,
        )

        log.info("Shard data loaded successfully")

    def load_video(
        self,
        rendered_video_path: str,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        max_frames: Optional[int] = None,
        gt_subsample_frames: int = 1,
        rendered_timestamps_file: Optional[str] = None,
    ) -> None:
        """Load rendered video and GT frames.

        Args:
            rendered_video_path: Path to rendered video file.
            x_offset: Camera X offset in meters.
            y_offset: Camera Y offset in meters.
            z_offset: Camera Z offset in meters.
            max_frames: Maximum frames to process (None = all).
            gt_subsample_frames: GT frame stride for alignment. Ignored if
                rendered_timestamps_file is provided.
            rendered_timestamps_file: Optional path to timestamps file containing frame indices and timestamps.
                When provided, enables timestamp-based frame matching instead of gt_subsample_frames.
        """
        if not rendered_video_path or not os.path.isfile(rendered_video_path):
            raise ValueError(f"Invalid video path: {rendered_video_path}")

        log.info("Loading video metadata...")
        log.info("  Video: %s", rendered_video_path)

        # Open video to get metadata (don't load frames yet)
        cap = cv2.VideoCapture(rendered_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {rendered_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Determine number of frames to process
        self._num_frames = min(total_frames, max_frames) if max_frames else total_frames

        log.info("  Video info: %d frames @ %.1f FPS, %dx%d", total_frames, fps, width, height)
        log.info("  Will process: %d frames", self._num_frames)

        # Store parameters (no pre-loading!)
        self._rendered_video_path = rendered_video_path
        self._x_offset = x_offset
        self._y_offset = y_offset
        self._z_offset = z_offset
        self._max_frames = max_frames
        self._gt_subsample_frames = gt_subsample_frames

        # Load timestamps from file if provided
        if rendered_timestamps_file is not None:
            loaded_timestamps = load_timestamps_file(rendered_timestamps_file)

            if loaded_timestamps:
                self._rendered_timestamps = loaded_timestamps
                log.info("  Matching rendered frames to GT using %d timestamps", len(loaded_timestamps))
            else:
                self._rendered_timestamps = None
                log.warning(
                    "  Timestamps file empty/invalid, using GT subsample matching (stride=%d)", gt_subsample_frames
                )
        else:
            self._rendered_timestamps = None
            log.info("  Matching rendered frames to GT using subsample (stride=%d)", gt_subsample_frames)

        log.info("Video metadata loaded (frames will be loaded on-demand)")
        log.info("  Offsets: X=%.2fm, Y=%.2fm, Z=%.2fm", x_offset, y_offset, z_offset)

    def _load_rendered_frame(self, frame_idx: int) -> np.ndarray:
        """Load a single rendered frame on-demand.

        Args:
            frame_idx: Frame index to load

        Returns:
            Frame as numpy array (RGB, uint8)
        """
        if not self._rendered_video_path:
            raise ValueError("No video path set")

        if frame_idx >= self._num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range (0-{self._num_frames - 1})")

        # Reuse VideoCapture for efficiency
        if self._video_cap is None or not self._video_cap.isOpened():
            self._video_cap = cv2.VideoCapture(self._rendered_video_path)

        assert self._video_cap is not None  # For type checker
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._video_cap.read()

        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        """Release video capture resources."""
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    @property
    def num_frames(self) -> int:
        """Get number of rendered frames."""
        return self._num_frames

    def get_frame_crops_with_metadata(self, rendered_video_idx: int) -> Optional[Dict[str, Any]]:
        """Extract object crops from a frame with full metadata.

        Args:
            rendered_video_idx: Rendered video frame index.

        Returns:
            Dictionary containing crops and metadata, or None if no valid crops.
        """
        if not self._rendered_video_path or self._bbox_projector is None:
            raise ValueError("Data not loaded. Call load_video() first.")

        assert self.shard_manager is not None
        assert self.labels_3d is not None

        camera_timestamps = self._bbox_projector.camera_sensor.get_frames_timestamps_us()
        num_camera_frames = len(self._bbox_projector.camera_sensor.get_frame_index_range())

        # Determine GT camera frame index based on matching mode
        if self._rendered_timestamps is not None:
            # Timestamp-based matching: find closest GT camera frame
            if rendered_video_idx >= len(self._rendered_timestamps):
                log.warning(
                    "Frame %d: No timestamp available (only %d)", rendered_video_idx, len(self._rendered_timestamps)
                )
                return None

            rendered_timestamp_us = self._rendered_timestamps[rendered_video_idx]

            # Find closest camera frame by timestamp using vectorized numpy ops
            if num_camera_frames == 0:
                gt_camera_idx = None
            else:
                camera_ts_array = np.asarray(camera_timestamps[:num_camera_frames])
                gt_camera_idx = int(np.argmin(np.abs(camera_ts_array - rendered_timestamp_us)))

            if gt_camera_idx is None:
                log.warning(
                    "Frame %d: No matching GT frame for timestamp %d", rendered_video_idx, rendered_timestamp_us
                )
                return None
        else:
            # Subsample-based matching (original logic)
            gt_camera_idx = rendered_video_idx * self._gt_subsample_frames

            if gt_camera_idx >= num_camera_frames:
                log.info(
                    "Frame %d: GT frame %d out of range (%d frames)",
                    rendered_video_idx,
                    gt_camera_idx,
                    num_camera_frames,
                )
                return None

        # Get GT camera timestamp for LiDAR matching
        gt_camera_timestamp_us = camera_timestamps[gt_camera_idx]

        # Find closest LiDAR frame to this GT camera timestamp
        lidar_sensor = self.shard_manager.lidar_sensor
        lidar_timestamps = lidar_sensor.get_frames_timestamps_us()
        lidar_frame_range = lidar_sensor.get_frame_index_range()

        # Find closest LiDAR timestamp using vectorized numpy ops
        lidar_indices = np.asarray(list(lidar_frame_range))
        if len(lidar_indices) == 0:
            closest_lidar_idx = None
        else:
            lidar_ts_array = np.asarray([lidar_timestamps[idx] for idx in lidar_indices])
            closest_lidar_idx = int(lidar_indices[np.argmin(np.abs(lidar_ts_array - gt_camera_timestamp_us))])

        if closest_lidar_idx is None:
            log.info("Frame %d: No LiDAR frame found", rendered_video_idx)
            return None

        lidar_frame_idx = closest_lidar_idx

        # Get 3D boxes for this LiDAR frame
        if lidar_frame_idx not in self.labels_3d:
            log.info("Frame %d: No labels for lidar frame %d", rendered_video_idx, lidar_frame_idx)
            return None

        frame_boxes = self.labels_3d[lidar_frame_idx]
        if not frame_boxes:
            log.info("Frame %d: Empty frame_boxes for lidar frame %d", rendered_video_idx, lidar_frame_idx)
            return None

        # Use the GT camera frame index we calculated
        camera_frame_idx = gt_camera_idx

        # Load frames on-demand (memory efficient!)
        try:
            rendered_frame = self._load_rendered_frame(rendered_video_idx)
            # Load GT frame directly from camera sensor
            gt_frame = self._bbox_projector.camera_sensor.get_frame_image_array(gt_camera_idx)
        except RuntimeError as e:
            log.warning("Frame %d: Could not load frames: %s", rendered_video_idx, e)
            return None

        # Get resolution info
        video_height, video_width = rendered_frame.shape[:2]
        gt_height, gt_width = gt_frame.shape[:2]
        scale_x = video_width / gt_width
        scale_y = video_height / gt_height

        # Get rendered timestamp if available (for detailed metrics)
        rendered_timestamp: int | None = None
        if self._rendered_timestamps is not None:
            rendered_timestamp = self._rendered_timestamps[rendered_video_idx]

        # Extract crops
        crops_data: Dict[str, Any] = {
            "pred_crops": [],
            "gt_crops": [],
            "track_ids": [],
            "class_names": [],
            "bboxes_gt": [],
            "bboxes_rendered": [],
            "camera_frame_idx": camera_frame_idx,
            "rendered_video_idx": rendered_video_idx,
            "lidar_frame_idx": lidar_frame_idx,
            "rendered_timestamp": rendered_timestamp,
        }

        # Batch projection: collect all object corners for vectorized processing
        track_ids = []
        class_names = []
        all_corners = []

        for track_id, box_data in frame_boxes.items():
            # Normalize track_id to consistent string format
            track_ids.append(normalize_track_id(track_id))
            class_names.append(box_data.get("class", "unknown"))
            corners = box_data["vertices"]
            if isinstance(corners, np.ndarray):
                corners = torch.from_numpy(corners).float()
            all_corners.append(corners)

        if not all_corners:
            return None

        # Stack all corners into single tensor [num_objects * 8, 3]
        all_corners_tensor = torch.cat(all_corners, dim=0)

        # Get camera transformations (same for all objects)
        t_world_sensor_original_start, t_world_sensor_original_end = (
            self._bbox_projector.get_shifted_camera_transformations(
                camera_frame_idx,
                0.0,  # No offset for GT
                0.0,
                0.0,
            )
        )

        t_world_sensor_shifted_start, t_world_sensor_shifted_end = (
            self._bbox_projector.get_shifted_camera_transformations(
                camera_frame_idx,
                self._x_offset,
                self._y_offset,
                self._z_offset,
            )
        )

        # Batch project all corners at once (2 calls vs 2*N calls)
        proj_original_batch = self._bbox_projector.camera_model.world_points_to_image_points_shutter_pose(
            all_corners_tensor,
            t_world_sensor_original_start,
            t_world_sensor_original_end,
            return_valid_indices=True,
            return_all_projections=True,
        )

        proj_shifted_batch = self._bbox_projector.camera_model.world_points_to_image_points_shutter_pose(
            all_corners_tensor,
            t_world_sensor_shifted_start,
            t_world_sensor_shifted_end,
            return_valid_indices=True,
            return_all_projections=True,
        )

        # Extract crops for each object using batched projection results
        num_corners_per_obj = 8

        for obj_idx, (track_id, class_name) in enumerate(zip(track_ids, class_names)):
            try:
                # Extract this object's corners from batch
                start_idx = obj_idx * num_corners_per_obj
                end_idx = start_idx + num_corners_per_obj

                proj_original_points = proj_original_batch.image_points[start_idx:end_idx]
                proj_original_valid = proj_original_batch.valid_indices
                orig_valid_mask = (proj_original_valid >= start_idx) & (proj_original_valid < end_idx)
                orig_sum = orig_valid_mask.sum().item()

                proj_shifted_points = proj_shifted_batch.image_points[start_idx:end_idx]
                proj_shifted_valid = proj_shifted_batch.valid_indices
                shifted_valid_mask = (proj_shifted_valid >= start_idx) & (proj_shifted_valid < end_idx)
                shifted_sum = shifted_valid_mask.sum().item()

                # Require at least 4 visible corners in both views
                if orig_sum < 4 or shifted_sum < 4:
                    continue

                # Get 2D bboxes from projected corners
                proj_original_points_np = proj_original_points.numpy()
                bbox_original = self._bbox_projector.get_2d_bbox_from_projection(proj_original_points_np)

                proj_shifted_points_np = proj_shifted_points.numpy()

                # Scale if resolution differs
                if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                    proj_shifted_points_scaled = proj_shifted_points_np.copy()
                    proj_shifted_points_scaled[:, 0] *= scale_x
                    proj_shifted_points_scaled[:, 1] *= scale_y
                    bbox_shifted = self._bbox_projector.get_2d_bbox_from_projection(proj_shifted_points_scaled)
                else:
                    bbox_shifted = self._bbox_projector.get_2d_bbox_from_projection(proj_shifted_points_np)

                # Extract object crops
                gt_crop = self._bbox_projector.crop_object(
                    gt_frame,
                    bbox_original,
                    min_size=self.min_bbox_size,
                )

                if gt_crop is None or gt_crop.size == 0:
                    continue

                # Scale min_size for rendered video
                scaled_min_size = self._bbox_projector.compute_scaled_min_size(
                    reference_shape=gt_frame.shape,
                    target_shape=rendered_frame.shape,
                    reference_min_size=self.min_bbox_size,
                )

                # Crop rendered object using bbox_shifted
                rendered_crop = self._bbox_projector.crop_object(
                    rendered_frame,
                    bbox_shifted,
                    min_size=scaled_min_size,
                )

                if rendered_crop is None or rendered_crop.size == 0:
                    continue

                # Store crops and metadata
                crops_data["pred_crops"].append(rendered_crop)
                crops_data["gt_crops"].append(gt_crop)
                crops_data["track_ids"].append(track_id)
                crops_data["class_names"].append(class_name)
                crops_data["bboxes_gt"].append(bbox_original)  # bbox from GT projection
                crops_data["bboxes_rendered"].append(bbox_shifted)  # bbox from rendered projection

            except (ValueError, RuntimeError) as e:
                log.debug("Failed to process box for track %s at frame %d: %s", track_id, rendered_video_idx, e)
                continue

        if not crops_data["pred_crops"]:
            return None

        return crops_data
