# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import time

from pathlib import Path
from typing import Any, Dict, Optional

import click
import cv2
import numpy as np
import point_cloud_utils as pcu
import tqdm
import yaml

from PIL import Image, ImageDraw, ImageFont
from upath import UPath

import ncore.data
import nre.utils.ncore_utils as ncore_utils

from ncore.data import (
    FrameTimepoint,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
)
from ncore.impl.common.transformations import transform_point_cloud
from ncore_internal.data.v3 import ShardDataLoader


class VideoWriter:
    def __init__(self, filepath: str, fps: int, resolution: tuple[int, int]):
        fc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(filepath, fc, fps, resolution)

    def write(self, image: Image.Image) -> None:
        self.writer.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


def export_segmentation_legend(
    file_path: str, semantic_meta: dict, legend_image_size: tuple[int, int] = (400, 600)
) -> None:
    img = Image.new("RGB", legend_image_size, (230, 230, 230))
    draw = ImageDraw.Draw(img)

    # Create a padding
    y_padding = 5
    x_padding = 100
    point_size = 30
    y_offset = y_padding
    x_offset = y_padding
    max_height = legend_image_size[1] - y_padding
    font = ImageFont.load_default()
    font_size = 20

    # Add label maps
    for idx in range(len(semantic_meta["stuff_colors"])):
        if y_offset + point_size + y_padding > max_height:
            y_offset = y_padding
            x_offset += point_size + x_padding

        draw.rectangle(
            (x_offset, y_offset, x_offset + point_size, y_offset + point_size),
            fill=tuple(semantic_meta["stuff_colors"][idx]),
        )
        draw.text(
            (x_offset + point_size + y_padding, y_offset + (point_size - font_size) // 2),
            semantic_meta["stuff_classes"][idx],
            fill=(0, 0, 0),
            font=font,
        )
        y_offset += point_size + y_padding

    img.save(file_path)


def get_lidar_aux_meta(aux_loader: ncore_utils.AuxShardDataLoader, lidar_id: str) -> Dict[str, Any]:
    aux_meta: Dict[str, Any] = {}
    if aux_loader.has_lidar_camera_visibility():
        aux_meta["lidar_camera_visibility"] = aux_loader.get_lidar_camera_visibility_meta(lidar_id)
    if aux_loader.has_lidar_semantic_segmentation():
        aux_meta["lidar_semantic_segmentation"] = aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)
    return aux_meta


def get_camera_aux_meta(aux_loader: ncore_utils.AuxShardDataLoader, camera_id: str) -> Dict[str, Any]:
    aux_meta: Dict[str, Any] = {}
    if aux_loader.has_depth(camera_id):
        aux_meta["depth"] = aux_loader.get_depth_meta(camera_id)
    if aux_loader.has_instance_segmentation(camera_id):
        aux_meta["instance_segmentation"] = aux_loader.get_instance_segmentation_meta(camera_id)
    if aux_loader.has_normal(camera_id):
        aux_meta["normal"] = aux_loader.get_normal_meta(camera_id)
    if aux_loader.has_optical_flow(camera_id):
        aux_meta["optical_flow"] = aux_loader.get_optical_flow_meta(camera_id)
    if aux_loader.has_scene_flow(camera_id):
        aux_meta["scene_flow"] = aux_loader.get_scene_flow_meta(camera_id)
    if aux_loader.has_semantic_logits(camera_id):
        aux_meta["semantic_logits"] = aux_loader.get_semantic_logits_meta(camera_id)
    if aux_loader.has_semantic_segmentation(camera_id):
        aux_meta["semantic_segmentation"] = aux_loader.get_semantic_segmentation_meta(camera_id)

    return aux_meta


def get_aux_overview(aux_loader: ncore_utils.AuxShardDataLoader) -> Dict[str, Any]:
    aux_meta: Dict[str, Any] = {}
    aux_meta["has_depth"] = aux_loader.has_depth()
    aux_meta["has_instance_segmentation"] = aux_loader.has_instance_segmentation()
    aux_meta["has_normal"] = aux_loader.has_normal()
    aux_meta["has_optical_flow"] = aux_loader.has_optical_flow()
    aux_meta["has_scene_flow"] = aux_loader.has_scene_flow()
    aux_meta["has_semantic_logits"] = aux_loader.has_semantic_logits()
    aux_meta["has_semantic_segmentation"] = aux_loader.has_semantic_segmentation()
    aux_meta["has_lidar_semantic_segmentation"] = aux_loader.has_lidar_semantic_segmentation()
    return aux_meta


def get_camera_model(camera: ncore.data.CameraSensorProtocol, detailed: bool = True) -> Dict[str, Any]:
    params = camera.model_parameters
    params_dict: Dict[str, Any] = {
        "resolution": params.resolution.tolist(),
        "shutter_type": params.shutter_type.name,
        "model_type": params.type(),
    }
    if detailed:
        if isinstance(params, OpenCVPinholeCameraModelParameters):
            params_dict["model_params"] = {
                "focal_length": params.focal_length.tolist(),
                "principal_point": params.principal_point.tolist(),
                "radial_coeffs": params.radial_coeffs.tolist(),
                "tangential_coeffs": params.tangential_coeffs.tolist(),
                "thin_prism_coeffs": params.thin_prism_coeffs.tolist(),
            }
        elif isinstance(params, OpenCVFisheyeCameraModelParameters):
            params_dict["model_params"] = {
                "focal_length": params.focal_length.tolist(),
                "principal_point": params.principal_point.tolist(),
                "radial_coeffs": params.radial_coeffs.tolist(),
                "max_angle": params.max_angle,
            }
        elif isinstance(params, FThetaCameraModelParameters):
            params_dict["model_params"] = {
                "principal_point": params.principal_point.tolist(),
                "reference_poly": params.reference_poly.name,
                "POLYNOMIAL_DEGREE": params.POLYNOMIAL_DEGREE,
                "max_angle": params.max_angle,
                "pixeldist_to_angle_poly": params.pixeldist_to_angle_poly.tolist(),
                "angle_to_pixeldist_poly": params.angle_to_pixeldist_poly.tolist(),
                "bw_poly": params.bw_poly.tolist(),
            }

    return params_dict


def ncore_get_metadata(
    loader: ncore.data.SequenceLoaderProtocol, aux_loader: ncore_utils.AuxShardDataLoader, detailed: bool = True
) -> Dict[str, Any]:
    """Return ncore metadata in a dictionary"""

    rig_world_edge = loader.pose_graph.get_edge("rig", "world")

    meta: Dict[str, Any] = {
        "sequence_id": loader.sequence_id,
        "shard_ids": list(range(len(loader.sequence_paths))),
        "shard_paths": [str(path) for path in loader.sequence_paths],
        "calibration_type": "unknown",
        "egomotion_type": "unknown",
        "camera_ids": list(loader.camera_ids),
        "lidar_ids": list(loader.lidar_ids),
        "poses": {
            "count": rig_world_edge.timestamps_us.size
            if rig_world_edge is not None and rig_world_edge.timestamps_us is not None
            else 0,
        },
    }

    if detailed and rig_world_edge is not None:
        meta["poses"]["T_rig_to_world_base"] = rig_world_edge.T_source_target[0].tolist()

    camera_sensors = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in loader.camera_ids}
    lidar_sensors = {lidar_id: loader.get_lidar_sensor(lidar_id) for lidar_id in loader.lidar_ids}

    # Add generic sensor metadata
    meta["sensors"] = {}
    for sensor in [*camera_sensors.values(), *lidar_sensors.values()]:
        frame_timestamps = sensor.frames_timestamps_us
        start_timestamp = int(frame_timestamps[0, 0])
        end_timestamp = int(frame_timestamps[-1, 1])
        frame_count = sensor.frames_count
        duration_sec = float(end_timestamp - start_timestamp) * 1e-6
        meta["sensors"][sensor.sensor_id] = {
            "frames_count": frame_count,
            "duration_sec": duration_sec,
            "overall_fps": (frame_count - 1) / duration_sec if duration_sec > 0 else 0,
        }
        if detailed and sensor.T_sensor_rig is not None:
            T_sensor_rig = np.asarray(sensor.T_sensor_rig)
            meta["sensors"][sensor.sensor_id]["T_sensor_rig"] = T_sensor_rig.tolist()
            meta["sensors"][sensor.sensor_id]["T_rig_sensor"] = np.linalg.inv(T_sensor_rig).tolist()

    # Add camera-specific metadata
    for camera_id, camera in camera_sensors.items():
        meta["sensors"][camera_id]["camera_model"] = get_camera_model(camera, detailed=detailed)
        if detailed:
            meta["sensors"][camera_id]["aux_meta"] = get_camera_aux_meta(aux_loader, camera_id)

    # Add lidar-specific metadata
    for lidar_id in lidar_sensors:
        if detailed:
            meta["sensors"][lidar_id]["aux_meta"] = get_lidar_aux_meta(aux_loader, lidar_id)

    meta["aux_meta"] = get_aux_overview(aux_loader)

    return meta


def get_frame_id(
    sensor: ncore.data.CameraSensorProtocol | ncore.data.LidarSensorProtocol, frame_index: int, name_by_timestamp: bool
):
    return str(int(sensor.frames_timestamps_us[frame_index, 1])) if name_by_timestamp else str(frame_index).zfill(6)


def ncore_export_camera_images(
    loader: ncore.data.SequenceLoaderProtocol,
    camera_ids: list[str],
    output_dir: Path,
    frame_step_camera: int,
    export_images: bool,
    export_video: bool,
    fps: int,
    name_by_timestamp: bool,
) -> None:
    if not camera_ids or (not export_images and not export_video):
        return

    if export_video:
        output_dir.mkdir(parents=True, exist_ok=True)

    for camera_id in camera_ids:
        if export_images:
            camera_output_dir = output_dir / camera_id
            camera_output_dir.mkdir(parents=True, exist_ok=True)

        camera = loader.get_camera_sensor(camera_id)
        frame_count = camera.frames_count
        indices = list(range(0, frame_count, frame_step_camera))
        if len(indices) == 0:
            continue

        video = None
        try:
            if export_video:
                video_name = f"{camera_id}_input.mp4"
                width, height = camera.get_frame_image(indices[0]).size
                video = VideoWriter(str(output_dir / video_name), fps, (width, height))

            for frame_index in tqdm.tqdm(indices, desc=f"{camera.sensor_id}"):
                image = camera.get_frame_image(frame_index)
                if export_images:
                    frame_id = get_frame_id(camera, frame_index, name_by_timestamp)
                    image.save(camera_output_dir / f"{frame_id}.jpg", quality=85)
                if video:
                    video.write(image)

        finally:
            if video:
                video.close()


def _save_points_ply(filename: str, positions: np.ndarray, colors: Optional[np.ndarray]) -> None:
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = positions
    if colors is not None:
        mesh.vertex_data.colors = colors
    mesh.save(filename)


def ncore_export_lidar_points(
    loader: ncore.data.SequenceLoaderProtocol,
    aux_loader: ncore_utils.AuxShardDataLoader,
    lidar_ids: list[str],
    output_dir: Path,
    frame_step_lidar: int,
    export_per_frame: bool,
    export_fused: bool,
    name_by_timestamp: bool,
) -> None:
    if not lidar_ids or (not export_per_frame and not export_fused):
        return

    # Apply semantic color map if aux data is provided
    has_semantics = aux_loader.has_lidar_semantic_segmentation()

    for lidar_id in lidar_ids:
        lidar_output_dir = output_dir / lidar_id
        lidar_output_dir.mkdir(parents=True, exist_ok=True)

        sensor = loader.get_lidar_sensor(lidar_id)
        frame_count = sensor.frames_count
        frame_indices = list(range(0, frame_count, frame_step_lidar))

        if has_semantics:
            semantic_meta = aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)
            color_map = np.array(semantic_meta["stuff_colors"], dtype=np.uint8)
            ignore_label = semantic_meta["ignore_label"]

            export_segmentation_legend(str(lidar_output_dir / "semantic_classes.png"), semantic_meta)
        else:
            ignore_label = None
            color_map = None

        fused_positions = []
        fused_colors = []

        for frame_index in tqdm.tqdm(frame_indices, desc=lidar_id):
            T_sensor_target = sensor.get_frames_T_sensor_target(
                target_node="world", frame_indices=frame_index, frame_timepoint=FrameTimepoint.END
            )

            pc = sensor.get_frame_point_cloud(frame_index, motion_compensation=True, with_start_points=False)
            point_positions = transform_point_cloud(pc.xyz_m_end, T_sensor_target)
            if export_fused:
                fused_positions.append(point_positions)

            # Export points with semantic class colors when available
            if color_map is not None:
                timestamp = int(sensor.frames_timestamps_us[frame_index, 1])
                semantic_segmentation = aux_loader.get_lidar_semantic_segmentation(lidar_id, timestamp)
                # Points with "ignore" label associated are points that could not receive a valid semantic label,
                # e.g. because they are not visible in any camera frames. We export these points in black.
                valid_labels = semantic_segmentation != ignore_label
                point_colors = np.empty((semantic_segmentation.shape[0], color_map.shape[1]), dtype=np.uint8)
                point_colors[valid_labels] = color_map[semantic_segmentation[valid_labels]]
                point_colors[~valid_labels] = np.array([0, 0, 0], dtype=np.uint8)
                if export_fused:
                    fused_colors.append(point_colors)
            else:
                point_colors = None

            if export_per_frame:
                frame_id = get_frame_id(sensor, frame_index, name_by_timestamp)
                filepath = lidar_output_dir / f"world_point_cloud_{frame_id}.ply"
                _save_points_ply(str(filepath), point_positions, point_colors)

        if export_fused:
            filepath = lidar_output_dir / "world_point_cloud_fused.ply"
            print(f"Exporting {filepath}")
            _save_points_ply(
                str(filepath), np.concatenate(fused_positions), np.concatenate(fused_colors) if has_semantics else None
            )


def _get_camera_semantic_labelmap(
    aux_loader: ncore_utils.AuxShardDataLoader,
    camera: ncore.data.CameraSensorProtocol,
    color_map: np.ndarray,
    frame_index: int,
) -> Image.Image:
    timestamp = int(camera.frames_timestamps_us[frame_index, 1])
    segmentation = aux_loader.get_semantic_segmentation(camera.sensor_id, timestamp)
    segmentation_array = np.asarray(segmentation)
    rgb_array = color_map[segmentation_array]
    segmentation_labelmap = Image.fromarray(rgb_array)
    return segmentation_labelmap


def ncore_export_camera_semantic_labelmaps(
    loader: ncore.data.SequenceLoaderProtocol,
    aux_loader: ncore_utils.AuxShardDataLoader,
    camera_ids: list[str],
    output_dir: Path,
    frame_step: int,
    export_images: bool,
    export_video: bool,
    fps: int,
    name_by_timestamp: bool,
):
    print(f"Exporting camera semantic labelmaps to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    aux_loader = ncore_utils.AuxShardDataLoader.from_sequence_loader(loader)
    for camera_id in camera_ids:
        camera = loader.get_camera_sensor(camera_id)
        if not aux_loader.has_semantic_segmentation(camera_id):
            print(f"Camera {camera_id} has no semantic segmentation, skipping")
            continue

        sensor_dir = output_dir / camera_id
        sensor_dir.mkdir(parents=True, exist_ok=True)

        segmentation_metadata = aux_loader.get_semantic_segmentation_meta(camera_id)
        color_map = np.array(segmentation_metadata["stuff_colors"], dtype=np.uint8)
        indices = list(range(0, camera.frames_count, frame_step))

        export_segmentation_legend(str(output_dir / f"{camera_id}_semantic_classes.png"), segmentation_metadata)

        video = None
        try:
            if export_video:
                video_name = f"{camera_id}_semantic_labelmaps.mp4"
                width, height = _get_camera_semantic_labelmap(aux_loader, camera, color_map, indices[0]).size
                video = VideoWriter(str(output_dir / video_name), fps, (width, height))

            for i, frame_index in enumerate(tqdm.tqdm(indices, desc=camera_id)):
                labelmap = _get_camera_semantic_labelmap(aux_loader, camera, color_map, frame_index)
                if export_images:
                    frame_id = get_frame_id(camera, frame_index, name_by_timestamp)
                    labelmap.save(sensor_dir / f"{frame_id}.png", compress_level=1)
                if video:
                    video.write(labelmap)

        finally:
            if video:
                video.close()


def ncore_export_camera_semantic_overlays(
    loader: ncore.data.SequenceLoaderProtocol,
    aux_loader: ncore_utils.AuxShardDataLoader,
    camera_ids: list[str],
    output_dir: Path,
    frame_step: int,
    export_images: bool,
    export_video: bool,
    fps: int,
    name_by_timestamp: bool,
):
    print(f"Exporting semantic labelmaps overlaid on camera images {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    aux_loader = ncore_utils.AuxShardDataLoader.from_sequence_loader(loader)
    for camera_id in camera_ids:
        if not aux_loader.has_semantic_segmentation(camera_id):
            print(f"Camera {camera_id} has no semantic segmentation, skipping")
            continue

        sensor_dir = output_dir / camera_id
        sensor_dir.mkdir(parents=True, exist_ok=True)

        camera = loader.get_camera_sensor(camera_id)
        segmentation_metadata = aux_loader.get_semantic_segmentation_meta(camera_id)
        color_map = np.array(segmentation_metadata["stuff_colors"], dtype=np.uint8)
        indices = list(range(0, camera.frames_count, frame_step))

        export_segmentation_legend(str(output_dir / f"{camera_id}_semantic_classes.png"), segmentation_metadata)

        video = None
        try:
            if export_video:
                video_name = f"{camera_id}_semantic_overlays.mp4"
                width, height = _get_camera_semantic_labelmap(aux_loader, camera, color_map, indices[0]).size
                video = VideoWriter(str(output_dir / video_name), fps, (width, height))

            for i, frame_index in enumerate(tqdm.tqdm(indices, desc=camera_id)):
                labelmap = _get_camera_semantic_labelmap(aux_loader, camera, color_map, frame_index)
                overlay = camera.get_frame_image(frame_index)
                labelmap.putalpha(128)
                overlay.paste(labelmap, (0, 0), labelmap)

                if export_images:
                    frame_id = get_frame_id(camera, frame_index, name_by_timestamp)
                    overlay.save(sensor_dir / f"{frame_id}.jpg", quality=85)
                if video:
                    video.write(overlay)

        finally:
            if video:
                video.close()


def _get_ego_mask_image(camera: ncore.data.CameraSensorProtocol, invert_mask: bool) -> Optional[Image.Image]:
    """Get ego mask as a PIL RGB Image, using the V3/V4-aware utility."""
    mask_array = ncore_utils.get_camera_sensor_mask(camera)
    if mask_array is None:
        return None
    if invert_mask:
        mask_array = ~mask_array
    return Image.fromarray((mask_array.astype(np.uint8) * 255)).convert("RGB")


def ncore_export_camera_ego_masks(
    loader: ncore.data.SequenceLoaderProtocol,
    camera_ids: list[str],
    output_dir: Path,
    invert_mask: bool,
) -> None:
    """Export per-camera static masks"""
    print(f"Exporting ego masks to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera_id in camera_ids:
        camera = loader.get_camera_sensor(camera_id)
        camera_mask = _get_ego_mask_image(camera, invert_mask)
        if camera_mask is not None:
            camera_mask.save(output_dir / f"{camera_id}.png")


def ncore_export_camera_ego_mask_overlays(
    loader: ncore.data.SequenceLoaderProtocol,
    camera_ids: list[str],
    output_dir: Path,
    frame_step: int,
    export_images: bool,
    export_video: bool,
    fps: int,
    name_by_timestamp: bool,
    invert_mask: bool,
) -> None:
    print(f"Exporting ego masks overlaid on camera images {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for camera_id in camera_ids:
        sensor_dir = output_dir / camera_id
        sensor_dir.mkdir(parents=True, exist_ok=True)

        camera = loader.get_camera_sensor(camera_id)
        resolution = camera.model_parameters.resolution
        width, height = int(resolution[0]), int(resolution[1])

        ego_car_mask = _get_ego_mask_image(camera, invert_mask)
        if ego_car_mask is not None:
            ego_car_mask.putalpha(128)

        indices = list(range(0, camera.frames_count, frame_step))
        video = None
        try:
            if export_video:
                video_name = f"{camera_id}_ego_mask_overlay.mp4"
                video = VideoWriter(str(output_dir / video_name), fps, (width, height))

            for i, frame_index in enumerate(tqdm.tqdm(indices, desc=camera_id)):
                overlay = camera.get_frame_image(frame_index)
                if ego_car_mask is not None:
                    overlay.paste(ego_car_mask, (0, 0), ego_car_mask)

                if export_images:
                    frame_id = get_frame_id(camera, frame_index, name_by_timestamp)
                    overlay.save(sensor_dir / f"{frame_id}.jpg", quality=85)
                if video:
                    video.write(overlay)

        finally:
            if video:
                video.close()


@click.command("export-ncore-diagnostic")
@click.option(
    "--shard-file-pattern",
    type=str,
    help=(
        "Path to the input .zarr.itar file(s). The extension can be omitted. "
        "Expands /path/file-[1-3] to [/path/file-1, /path/file-2, /path/file-3]"
    ),
    required=False,
    deprecated="Please use --dataset-path instead",
)
@click.option(
    "--dataset-path",
    type=str,
    help="Path to a NCore V3/V4 sequence meta-file",
    required=False,
)
@click.option("--poses-component-group", type=str, help="V4 component group for 'poses'", default="default")
@click.option("--intrinsics-component-group", type=str, help="V4 component group for 'intrinsics'", default="default")
@click.option("--masks-component-group", type=str, help="V4 component group for 'masks'", default="default")
@click.option("--cuboids-component-group", type=str, help="V4 component group for 'cuboids'", default="default")
@click.option(
    "--output-dir",
    type=str,
    help="Path to an output directory to export the requested data to",
    required=False,
)
@click.option(
    "--frame-step-camera",
    type=int,
    help="Frame step to use when exporting camera RGB frames / images (>1 means frame skipping)",
    default=50,
)
@click.option(
    "--frame-step-lidar",
    type=int,
    help="Frame step to use when exporting Lidar frames / spins (>1 means frame skipping)",
    default=50,
)
@click.option(
    "--frame-naming",
    type=click.Choice(["index", "timestamp"]),
    help=(
        "File naming scheme for exported frames: 'index' - frame indexing starting from zero per sensor and "
        "reflecting frame steps/skips, 'timestamp' - global absolute timestamp in microseconds "
    ),
    default="timestamp",
)
@click.option(
    "--video-fps",
    type=int,
    help="fps of the exported videos",
    default=15,
)
@click.option(
    "--meta",
    "export_meta",
    is_flag=True,
    help="Export NCORE metadata into a YAML file",
    default=False,
)
@click.option(
    "--camera-images",
    "export_camera_images",
    is_flag=True,
    help="Export camera RGB frames per camera",
    default=False,
)
@click.option(
    "--lidar-points",
    "export_lidar_points",
    is_flag=True,
    help="Export LIDAR points per frame (spin) into separate PLY files, colorized per semantic class when available",
    default=False,
)
@click.option(
    "--lidar-points-fused",
    "export_lidar_points_fused",
    is_flag=True,
    help="Export all LIDAR points into a single PLY per sensor, colorized per semantic class when available",
    default=False,
)
@click.option(
    "--semantic-labelmaps",
    "export_semantic_labelmaps",
    is_flag=True,
    help="Export semantic labelmaps per camera when available",
    default=False,
)
@click.option(
    "--semantic-overlays",
    "export_semantic_overlays",
    is_flag=True,
    help="If available, export semantic labelmaps overlaid on the corresponding input RGB images per camera",
    default=False,
)
@click.option(
    "--ego-masks",
    "export_ego_masks",
    is_flag=True,
    help="If available, export ego car masks per camera",
    default=False,
)
@click.option(
    "--ego-mask-overlays",
    "export_ego_mask_overlays",
    is_flag=True,
    help="If available, export ego car masks overlaid on the corresponding input RGB images per camera",
    default=False,
)
@click.option(
    "--invert-mask/--no-invert-mask",
    default=False,
    help="Whether to show the masked region (--no-invert-mask) or show the unmasked region (--invert-mask)",
)
@click.option(
    "--format",
    type=click.Choice(["image", "video", "image+video"]),
    help="Controls the type of export file produced",
    default="image",
)
@click.option(
    "--all",
    "export_all",
    is_flag=True,
    help="Sets all export flags to true",
    default=False,
)
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be used (multiple value option, all if not specified)",
)
@click.option(
    "--lidar-id",
    "lidar_ids",
    multiple=True,
    type=str,
    help="Lidars to be used (multiple value option, all if not specified)",
)
def export_ncore_diagnostic(
    shard_file_pattern: Optional[str],
    dataset_path: Optional[str],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
    output_dir: str,
    frame_step_camera: int,
    frame_step_lidar: int,
    frame_naming: str,
    video_fps: int,
    export_meta: bool,
    export_camera_images: bool,
    export_lidar_points: bool,
    export_lidar_points_fused: bool,
    export_semantic_labelmaps: bool,
    export_semantic_overlays: bool,
    export_ego_masks: bool,
    export_ego_mask_overlays: bool,
    invert_mask: bool,
    format: str,
    export_all: bool,
    camera_ids: tuple[str, ...],
    lidar_ids: tuple[str, ...],
) -> None:
    """Exports low-level ncore / aux data for diagnostics or benchmarking"""
    # Forward the call to a regular function for reusability.
    export_ncore_diagnostic_func(
        shard_file_pattern=shard_file_pattern,
        dataset_path=dataset_path,
        poses_component_group=poses_component_group,
        intrinsics_component_group=intrinsics_component_group,
        masks_component_group=masks_component_group,
        cuboids_component_group=cuboids_component_group,
        output_dir=output_dir,
        frame_step_camera=frame_step_camera,
        frame_step_lidar=frame_step_lidar,
        frame_naming=frame_naming,
        video_fps=video_fps,
        export_meta=export_meta,
        export_camera_images=export_camera_images,
        export_lidar_points=export_lidar_points,
        export_lidar_points_fused=export_lidar_points_fused,
        export_semantic_labelmaps=export_semantic_labelmaps,
        export_semantic_overlays=export_semantic_overlays,
        export_ego_masks=export_ego_masks,
        export_ego_mask_overlays=export_ego_mask_overlays,
        invert_mask=invert_mask,
        format=format,
        export_all=export_all,
        camera_ids=camera_ids,
        lidar_ids=lidar_ids,
    )


def export_ncore_diagnostic_func(
    shard_file_pattern: Optional[str],
    dataset_path: Optional[str],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
    output_dir: Optional[str],
    frame_step_camera: int,
    frame_step_lidar: int,
    frame_naming: str,
    video_fps: int,
    export_meta: bool,
    export_camera_images: bool,
    export_lidar_points: bool,
    export_lidar_points_fused: bool,
    export_semantic_labelmaps: bool,
    export_semantic_overlays: bool,
    export_ego_masks: bool,
    export_ego_mask_overlays: bool,
    invert_mask: bool,
    format: str,
    export_all: bool,
    camera_ids: tuple[str, ...] = (),
    lidar_ids: tuple[str, ...] = (),
) -> None:
    """Export low-level ncore / aux data for diagnostics or benchmarking"""
    if export_all:
        export_meta = True
        export_camera_images = True
        export_lidar_points = True
        export_lidar_points_fused = True
        export_semantic_labelmaps = True
        export_semantic_overlays = True
        export_ego_masks = True
        export_ego_mask_overlays = True

    export_any = (
        export_meta
        or export_camera_images
        or export_lidar_points
        or export_lidar_points_fused
        or export_semantic_labelmaps
        or export_semantic_overlays
        or export_ego_masks
        or export_ego_mask_overlays
    )

    if output_dir is None and export_any:
        raise ValueError("Please specify [--output-dir]")

    if bool(dataset_path) == bool(shard_file_pattern):
        raise click.ClickException("Exactly one of --shard-file-pattern or --dataset-path must be provided")

    if shard_file_pattern is not None:
        logging.warning(
            "`--shard-file-pattern` is deprecated and will be removed in future releases. "
            "Please use `--dataset-path` instead."
        )
        if len(Path(shard_file_pattern).suffixes) == 0:
            shard_file_pattern += ".zarr.itar"
        shard_files = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
        data_format = ncore_utils.NCoreDataFormat.V3
        resolved_dataset_paths = [UPath(p) for p in shard_files]
    else:
        assert dataset_path is not None
        data_format, _, _, resolved_dataset_paths = ncore_utils.parse_sequence_meta_file(UPath(dataset_path))

    print("Loading dataset...")
    for dataset_file in resolved_dataset_paths:
        print("  " + str(dataset_file))

    start_timestamp = time.perf_counter()
    loader = ncore_utils.create_sequence_loader(
        data_format=data_format,
        dataset_paths=resolved_dataset_paths,
        open_consolidated=True,
        v3_cuboid_loading_max_workers=None,
        v4_poses_component_group=poses_component_group,
        v4_intrinsics_component_group=intrinsics_component_group,
        v4_masks_component_group=masks_component_group,
        v4_cuboids_component_group=cuboids_component_group,
    )
    aux_loader = ncore_utils.AuxShardDataLoader.from_sequence_loader(loader)
    print(f"{len(resolved_dataset_paths)} dataset files loaded in {time.perf_counter() - start_timestamp:.3f} seconds")

    camera_ids_list = list(loader.camera_ids) if not camera_ids else list(camera_ids)
    lidar_ids_list = list(loader.lidar_ids) if not lidar_ids else list(lidar_ids)
    export_image_format = format in ("image", "image+video") or export_all
    export_video_format = format in ("video", "image+video") or export_all

    name_by_timestamp = frame_naming == "timestamp"

    if output_dir is not None:
        print(f"Creating directory {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_data_dir = Path(output_dir)  # type: ignore

        if export_meta:
            output_yaml_path = output_data_dir / "meta_overview.yaml"
            print(f"Exporting overview to {output_yaml_path}")
            with open(output_yaml_path, "w", encoding="utf-8") as outfile:
                meta = ncore_get_metadata(loader, aux_loader, detailed=False)
                yaml.safe_dump(meta, outfile, default_flow_style=False, sort_keys=False)

            output_yaml_path = output_data_dir / "meta.yaml"
            print(f"Exporting metadata to {output_yaml_path}")
            with open(output_yaml_path, "w", encoding="utf-8") as outfile:
                meta = ncore_get_metadata(loader, aux_loader, detailed=True)
                yaml.safe_dump(meta, outfile, default_flow_style=False, sort_keys=False)

        if export_camera_images:
            output_images_path = output_data_dir / "camera_images"
            print(f"Exporting camera images to {output_images_path}")
            ncore_export_camera_images(
                loader=loader,
                camera_ids=camera_ids_list,
                output_dir=output_images_path,
                frame_step_camera=frame_step_camera,
                export_images=export_image_format,
                export_video=export_video_format,
                fps=video_fps,
                name_by_timestamp=name_by_timestamp,
            )

        if export_lidar_points or export_lidar_points_fused:
            output_pc_path = output_data_dir / "lidar_point_clouds"
            print(f"Exporting lidar point clouds to {output_pc_path}")
            ncore_export_lidar_points(
                loader=loader,
                aux_loader=aux_loader,
                lidar_ids=lidar_ids_list,
                output_dir=output_pc_path,
                frame_step_lidar=frame_step_lidar,
                export_per_frame=export_lidar_points,
                export_fused=export_lidar_points_fused,
                name_by_timestamp=name_by_timestamp,
            )

        if export_semantic_labelmaps:
            if aux_loader.has_semantic_segmentation():
                output_labelmap_path = output_data_dir / "camera_semantic_labelmaps"
                ncore_export_camera_semantic_labelmaps(
                    loader=loader,
                    aux_loader=aux_loader,
                    camera_ids=camera_ids_list,
                    output_dir=output_labelmap_path,
                    frame_step=frame_step_camera,
                    export_images=export_image_format,
                    export_video=export_video_format,
                    fps=video_fps,
                    name_by_timestamp=name_by_timestamp,
                )
            else:
                print("Dataset provided does not contain image semantic segmentation")

        if export_semantic_overlays:
            if aux_loader.has_semantic_segmentation():
                output_overlay_path = output_data_dir / "camera_semantic_overlays"
                ncore_export_camera_semantic_overlays(
                    loader=loader,
                    aux_loader=aux_loader,
                    camera_ids=camera_ids_list,
                    output_dir=output_overlay_path,
                    frame_step=frame_step_camera,
                    export_images=export_image_format,
                    export_video=export_video_format,
                    fps=video_fps,
                    name_by_timestamp=name_by_timestamp,
                )
            else:
                print("Dataset provided does not contain image semantic segmentation")

        if export_ego_masks:
            output_ego_mask_path = output_data_dir / "camera_ego_masks"
            ncore_export_camera_ego_masks(
                loader=loader,
                camera_ids=camera_ids_list,
                output_dir=output_ego_mask_path,
                invert_mask=invert_mask,
            )

        if export_ego_mask_overlays:
            output_overlay_path = output_data_dir / "camera_ego_mask_overlays"
            ncore_export_camera_ego_mask_overlays(
                loader=loader,
                camera_ids=camera_ids_list,
                output_dir=output_overlay_path,
                frame_step=frame_step_camera,
                export_images=export_image_format,
                export_video=export_video_format,
                fps=video_fps,
                name_by_timestamp=name_by_timestamp,
                invert_mask=invert_mask,
            )

    if not export_any:
        print(yaml.safe_dump(ncore_get_metadata(loader, aux_loader, detailed=False), sort_keys=False))
