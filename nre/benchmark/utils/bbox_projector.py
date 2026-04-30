# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""BBox Projector utility for 3D to 2D projection and bounding box operations."""

from typing import Optional, Tuple

import numpy as np
import torch

from ncore.data import FrameTimepoint


class BBoxProjector:
    """Handles 3D to 2D projection, transformations, and bounding box operations.

    This class manages all operations related to projecting 3D bounding boxes
    to 2D image space, applying camera transformations, and extracting crops.
    """

    def __init__(self, camera_model, camera_sensor):
        """Initialize bounding box projector.

        Args:
            camera_model: Camera model for projection.
            camera_sensor: Camera sensor for transformations.
        """
        self.camera_model = camera_model
        self.camera_sensor = camera_sensor

    def update_sensors(self, camera_model, camera_sensor) -> None:
        """Update camera model and sensor (for batch processing after reload).

        Args:
            camera_model: New camera model.
            camera_sensor: New camera sensor.
        """
        self.camera_model = camera_model
        self.camera_sensor = camera_sensor

    def get_shifted_camera_transformations(
        self,
        frame_idx: int,
        x_offset: float,
        y_offset: float,
        z_offset: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get camera transformations with applied offset.

        Args:
            frame_idx: Frame index.
            x_offset: X-axis offset in meters.
            y_offset: Y-axis offset in meters.
            z_offset: Z-axis offset in meters.

        Returns:
            Tuple of (t_world_sensor_start, t_world_sensor_end)
            transformation tensors.
        """
        # Get rig-to-world transformations for this frame
        t_rig_world_start = self.camera_sensor.get_frame_T_rig_world(frame_idx, FrameTimepoint.START)
        t_rig_world_end = self.camera_sensor.get_frame_T_rig_world(frame_idx, FrameTimepoint.END)

        # Get sensor-to-rig transformation (camera mounting position)
        t_sensor_rig = self.camera_sensor.get_T_sensor_rig()

        # Convert to numpy and copy
        if isinstance(t_sensor_rig, torch.Tensor):
            t_sensor_rig_shifted = t_sensor_rig.cpu().numpy().copy()
        else:
            t_sensor_rig_shifted = t_sensor_rig.copy()

        # Apply offsets to the sensor position relative to the rig
        t_sensor_rig_shifted[0, 3] += x_offset  # X offset (forward/backward)
        t_sensor_rig_shifted[1, 3] += y_offset  # Y offset (left/right)
        t_sensor_rig_shifted[2, 3] += z_offset  # Z offset (up/down)

        # Convert rig-to-world to numpy
        if isinstance(t_rig_world_start, torch.Tensor):
            t_rig_world_start_np = t_rig_world_start.cpu().numpy()
        else:
            t_rig_world_start_np = t_rig_world_start

        if isinstance(t_rig_world_end, torch.Tensor):
            t_rig_world_end_np = t_rig_world_end.cpu().numpy()
        else:
            t_rig_world_end_np = t_rig_world_end

        # Compute sensor-to-world transformations
        t_sensor_world_start = t_rig_world_start_np @ t_sensor_rig_shifted
        t_sensor_world_end = t_rig_world_end_np @ t_sensor_rig_shifted

        # Invert to get world-to-sensor transformations for projection
        t_world_sensor_start = np.linalg.inv(t_sensor_world_start)
        t_world_sensor_end = np.linalg.inv(t_sensor_world_end)

        return (
            torch.from_numpy(t_world_sensor_start),
            torch.from_numpy(t_world_sensor_end),
        )

    @staticmethod
    def get_2d_bbox_from_projection(
        image_points: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """Convert projected 2D points to bounding box.

        Args:
            image_points: Array of 2D projected points.

        Returns:
            Tuple of (x, y, width, height) for the bounding box.
        """
        if image_points.ndim == 3:
            image_points = image_points[0]  # Remove batch dimension

        x_coords = image_points[:, 0]
        y_coords = image_points[:, 1]

        x_min = int(np.floor(np.min(x_coords)))
        y_min = int(np.floor(np.min(y_coords)))
        x_max = int(np.ceil(np.max(x_coords)))
        y_max = int(np.ceil(np.max(y_coords)))

        width = x_max - x_min
        height = y_max - y_min

        return (x_min, y_min, width, height)

    @staticmethod
    def compute_scaled_min_size(
        reference_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...],
        reference_min_size: int = 20,
    ) -> int:
        """Compute scaled minimum size based on resolution difference.

        Scales the minimum size threshold proportionally to the resolution
        difference between reference (e.g., ground truth) and target
        (e.g., rendered) images.

        Args:
            reference_shape: Reference image shape (height, width, ...).
            target_shape: Target image shape (height, width, ...).
            reference_min_size: Min size for reference resolution.

        Returns:
            Scaled minimum size for target resolution.
        """
        ref_height, ref_width = reference_shape[:2]
        tgt_height, tgt_width = target_shape[:2]

        # Use average of height and width scaling factors
        scale_h = tgt_height / ref_height
        scale_w = tgt_width / ref_width
        scale_factor = (scale_h + scale_w) / 2.0

        # Scale the minimum size and ensure it's at least 1 pixel
        scaled_min_size = max(1, int(reference_min_size * scale_factor))

        return scaled_min_size

    @staticmethod
    def scale_bbox_around_center(
        bbox: Tuple[int, int, int, int],
        scale: float,
        image_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """Scale bbox around its center, clamped to image bounds.

        Args:
            bbox: Tuple of (x, y, width, height).
            scale: Scale factor (>1.0 expands, <1.0 crops/zooms in).
            image_shape: Image shape (height, width, ...).

        Returns:
            Scaled bbox (x, y, width, height) clamped to image.
        """
        x, y, width, height = bbox
        img_height, img_width = image_shape[:2]

        # Find center
        cx = x + width / 2.0
        cy = y + height / 2.0

        # Scale dimensions
        new_width = max(1.0, width * scale)
        new_height = max(1.0, height * scale)

        # Compute new bbox
        x0 = max(0, int(cx - new_width / 2.0))
        y0 = max(0, int(cy - new_height / 2.0))
        x1 = min(img_width, int(cx + new_width / 2.0))
        y1 = min(img_height, int(cy + new_height / 2.0))

        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    @staticmethod
    def is_bbox_valid(
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, ...],
        min_size: int = 20,
    ) -> bool:
        """Check if bbox is within image bounds and meets minimum size.

        Args:
            bbox: Tuple of (x, y, width, height).
            image_shape: Shape of the image (height, width, ...).
            min_size: Minimum size in pixels for valid bbox (default: 20).

        Returns:
            True if bbox is valid, False otherwise.
        """
        x, y, width, height = bbox
        img_height, img_width = image_shape[:2]

        # Check if bbox is within image bounds
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            return False

        # Check minimum size requirement
        if width < min_size or height < min_size:
            return False

        return True

    @staticmethod
    def crop_object(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding_ratio: float = 0.0,
        min_size: int = 50,
    ) -> Optional[np.ndarray]:
        """Extract object crop with padding around bounding box.

        Args:
            image: Input image array.
            bbox: Tuple of (x, y, width, height).
            padding_ratio: Ratio of padding to add around the bbox.
            min_size: Minimum size in pixels for valid bbox (default: 50).

        Returns:
            Cropped image array or None if invalid.
        """
        # check if bbox passes minimum size requirement
        if not BBoxProjector.is_bbox_valid(bbox, image.shape, min_size=min_size):
            return None

        x, y, width, height = bbox
        img_height, img_width = image.shape[:2]

        # Add padding based on largest dimension
        padding = int(max(width, height) * padding_ratio)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img_width, x + width + padding)
        y_end = min(img_height, y + height + padding)

        # Extract crop
        crop = image[y_start:y_end, x_start:x_end]

        # Ensure valid crop
        if crop.size == 0:
            return None

        return crop
