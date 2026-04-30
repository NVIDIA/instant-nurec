# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import multiprocessing
import multiprocessing.synchronize
import os

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Tuple, cast

import click
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as PILImage
import torch
import tqdm

from upath import UPath

import ncore.data
import ncore.data.v4
import ncore.impl.data.stores as ncore_stores
import ncore_internal.data.v3
import nre.utils.cli as nre_utils_cli

from apps.aux_gen.estimators import (
    DepthAnythingV2Estimator,
    DINOv2Estimator,
    LidarSegmentationByProjectEstimator,
    Mask2FormerSegmentationEstimator,
    MetricDepthAnythingV2Estimator,
    PromptlessSAM2EgoMaskEstimator,
)
from ncore.data import ConcreteCameraModelParametersUnion
from ncore.impl.common.transformations import time_bounds
from nre.utils.debug.remote_debug import breakpoint_env
from nre.utils.files import parse_universal_path
from nre.utils.ncore_utils import (
    PCA,
    AuxDataCameraSemSegProvider,
    AuxDataWriter,
    AuxShardDataLoader,
    CamAuxDataLoader,
    Feature2ColorTransform,
    get_camera_sensor_mask,
    parse_sequence_meta_file,
)
from nre.utils.profiling import ScopedTimer, TimingTag
from nre.utils.types import HalfClosedInterval


# Conditionally activate remote debugging based on environment variables
breakpoint_env()


@dataclass(kw_only=True, slots=True, frozen=True)
class SensorRanges:
    camera_frame_ranges: dict[str, range]
    lidar_frame_ranges: dict[str, range]


SensorRangeCallback = Callable[
    [ncore.data.SequenceLoaderProtocol], SensorRanges
]  # represents frame range for each enabled camera / lidar sensor


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part"""

    # V3 low-level shard-based entry point [deprecated]
    shard_file_pattern: Optional[str]

    # V3/V4 meta-file entry point
    dataset_path: Optional[str]  # path to a NCore meta-file

    # V4 components considered for data loading
    poses_component_group: str
    intrinsics_component_group: str
    masks_component_group: str
    cuboids_component_group: str

    output_dir: str
    camera_ids: list[str]
    lidar_ids: list[str]
    segmentation_backend: str
    seg_logits: bool
    enable_trt: bool
    dinov2_backend: str
    dinov2_pca_dim: int
    dinov2_width: int
    lidar_seg_camvis: bool
    lidar_seg_ensemble_cuda: bool
    depth_backend: str
    relative_depth: bool
    max_depth_m: float
    depth_input_resolution: int
    store_depth_as_png: bool
    zarr_store_type: str
    open_consolidated: bool
    num_threads: str
    debug: bool
    visualize: bool
    store_meta: bool
    ego_mask: bool
    ego_mask_samples_per_second: float
    ego_mask_aggregation_method: Literal["majority"]
    ego_mask_camera_ids: tuple[str, ...] | None = None
    parallel_mode: bool
    workers_per_gpu: int


def _run_ego_mask_for_camera(
    ego_mask: bool,
    ego_mask_camera_ids: Optional[Sequence[str]],
    camera_id: str,
) -> bool:
    """Return whether ego-mask estimation should run for the given camera.

    Used in the per-camera loop to respect --ego-mask and --ego-mask-camera-id:
    when ego_mask_camera_ids is None or empty, all cameras get ego mask (when
    ego_mask is True); otherwise only cameras listed in ego_mask_camera_ids do.

    Args:
        ego_mask: Whether ego-mask is enabled (e.g. from --ego-mask).
        ego_mask_camera_ids: Optional tuple of camera IDs from Click (or any Sequence) to
            restrict ego-mask to. None or empty means "all cameras".
        camera_id: The current camera ID.

    Returns:
        True if ego-mask estimation should run for this camera, False otherwise.
    """
    return bool(ego_mask and (not ego_mask_camera_ids or camera_id in ego_mask_camera_ids))


def mask2former_worker(
    device_id: int,
    gpu_sem: multiprocessing.synchronize.BoundedSemaphore,
    encoded_sequence_loader: str,
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
    zarr_store_type: str,
    open_consolidated: bool,
    output_path: Path,
    camera_id: str,
    camera_frame_range: range,
    seg_logits: bool,
    enable_trt: bool,
    visualize: bool,
    has_ego_mask: bool = False,
):
    """Worker: mask2former."""
    with gpu_sem:
        torch.cuda.set_device(device_id)
        logging.info(
            "\n*********** Device Sanity Check ************ \n"
            f"visible device counts: {torch.cuda.device_count()} \n"
            f"current device: {torch.cuda.current_device()} \n"
            f"device name: {torch.cuda.get_device_name(0)} \n"
            "*********** End of Device Sanity Check ************"
        )

        # construct the current loader and writer for each process
        sequence_loader = decode_sequence_loader_json(
            encoded_sequence_loader,
            poses_component_group,
            intrinsics_component_group,
            masks_component_group,
            cuboids_component_group,
            open_consolidated,
        )

        base_name, shard_id, shard_count = sequence_loader_shard_info(sequence_loader)

        # Need to be aware of the race condition
        writer = AuxDataWriter(
            output_path,
            base_name,
            sequence_loader.sequence_id,
            shard_id,
            shard_count,
            zarr_store_type,
        )

        camera_sensor = sequence_loader.get_camera_sensor(camera_id)
        ts = list(
            zip(
                camera_sensor.get_frames_timestamps_us(),
                [camera_sensor.get_frame_handle(i) for i in range(camera_sensor.frames_count)],
            )
        )[camera_frame_range.start : camera_frame_range.stop]

        cam_params = camera_sensor.model_parameters

        logging.info(f"[GPU{device_id}] camera {camera_id}: {len(ts)} frames, TRT={enable_trt}, logits={seg_logits}")
        estimator = Mask2FormerSegmentationEstimator(
            cam_params.resolution.tolist(),
            estimate_logits=seg_logits,
            enable_trt=enable_trt,
        )

        run_mask2former_segmentation(
            estimator, writer, camera_id, cam_params, camera_sensor, ts, visualize=visualize, has_ego_mask=has_ego_mask
        )

        del estimator
        torch.cuda.empty_cache()

        # properly close permanent annotation shard
        writer.finalize()


def merge_zarr_itar(
    root_dir: Path,
    pattern: str,
    output_dir: Path,
):
    """Merged the zarr itar files together."""
    itar_files = list(root_dir.rglob(pattern))
    if not itar_files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {root_dir}")
    output_fp = output_dir / itar_files[0].name

    zarr_store_list = []
    for path in itar_files:
        zarr_store = ncore_stores.IndexedTarStore(path, mode="r")
        zarr_store_list.append(zarr_store)

    with ncore_stores.IndexedTarStore(output_fp, mode="w") as s_itar_out:
        for store in zarr_store_list:
            for k in store.keys():
                if k.startswith(".zmetadata"):
                    continue

                if k not in s_itar_out:
                    s_itar_out[k] = store[k]

        ncore_stores.consolidate_compressed_metadata(s_itar_out)

    return output_fp


def encode_sequence_loader_json(
    sequence_loader: ncore.data.SequenceLoaderProtocol,
) -> str:
    """Encode sequence loader to a JSON string that can be decoded into an instance in another process."""

    match sequence_loader:
        case ncore.data.v4.SequenceLoaderV4():
            dataset_format = "V4"
        case ncore_internal.data.v3.SequenceLoaderV3():
            dataset_format = "V3"
        case _:
            raise ValueError(f"Unsupported sequence loader type: {type(sequence_loader)}")

    # encode dataset format and sequence paths
    return json.dumps(
        {
            "format": dataset_format,
            "sequence_paths": [str(p) for p in sequence_loader.sequence_paths],
        }
    )


def sequence_loader_shard_info(sequence_loader: ncore.data.SequenceLoaderProtocol) -> Tuple[str, int, int]:
    """Get shard info for V3/V4 sequence loader.

    Returns:
        base_name: str - base name of the sequence
        shard_id: int - shard id
        shard_count: int - shard count
    """

    # usual default cases for shard id / count
    if isinstance(sequence_loader, ncore_internal.data.v3.SequenceLoaderV3):
        # Infer V3 shard id / counts if required (a bit hacky - we know that we only load a single shard, so this ID is valid)
        base_name = sequence_loader.sequence_paths[0].stem.split(".")[0]
        shard_id = sequence_loader._loader._shard_ids[0]
        shard_count = sequence_loader._loader._shard_count
    else:
        # V4 is always single-shard
        base_name = sequence_loader.sequence_id
        shard_id = 0
        shard_count = 1

    return base_name, shard_id, shard_count


def decode_sequence_loader_json(
    encoded_sequence_loader_json: str,
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
    open_consolidated: bool,
) -> ncore.data.SequenceLoaderProtocol:
    """Decode sequence loader from a JSON string."""

    decoded = json.loads(encoded_sequence_loader_json)
    dataset_format = decoded["format"]
    sequence_paths = [UPath(p) for p in decoded["sequence_paths"]]

    match dataset_format:
        case "V4":
            reader = ncore.data.v4.SequenceComponentGroupsReader(sequence_paths, open_consolidated=open_consolidated)
            return ncore.data.v4.SequenceLoaderV4(
                reader,
                poses_component_group_name=poses_component_group,
                intrinsics_component_group_name=intrinsics_component_group,
                masks_component_group_name=masks_component_group,
                cuboids_component_group_name=cuboids_component_group,
            )
        case "V3":
            return ncore_internal.data.v3.SequenceLoaderV3(
                ncore_internal.data.v3.ShardDataLoader(sequence_paths, open_consolidated=open_consolidated)
            )
        case _:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")


def run_mask2former_parallel(
    sensor_ranges: SensorRanges,
    sequence_loader: ncore.data.SequenceLoaderProtocol,
    params: CLIBaseParams,
    output_path: Path,
):
    """Run run_mask2former task in parallel mode."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA device available")

    ctx = multiprocessing.get_context("spawn")
    gpu_sems = [ctx.BoundedSemaphore(params.workers_per_gpu) for _ in range(num_gpus)]
    procs = []

    for idx, (camera_id, camera_frame_range) in enumerate(sensor_ranges.camera_frame_ranges.items()):
        device_id = idx % num_gpus
        logging.info(f"Assign camera {camera_id} -> GPU {device_id}")
        cam_output_path = output_path / "cam" / camera_id
        p = ctx.Process(
            target=mask2former_worker,
            args=(
                device_id,
                gpu_sems[device_id],
                encode_sequence_loader_json(sequence_loader),
                params.poses_component_group,
                params.intrinsics_component_group,
                params.masks_component_group,
                params.cuboids_component_group,
                params.zarr_store_type,
                params.open_consolidated,
                cam_output_path,
                camera_id,
                camera_frame_range,
                params.seg_logits,
                params.enable_trt,
                params.visualize,
                False,  # has_ego_mask: parallel seg runs before ego mask in main loop
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    final_output_fp = merge_zarr_itar(
        root_dir=Path(output_path / "cam"),
        pattern="*.aux.sseg.zarr.itar",
        output_dir=output_path,
    )

    return final_output_fp


def lidar_segmentation_worker(
    device_id: int,
    gpu_sem: multiprocessing.synchronize.BoundedSemaphore,
    lidar_id: str,
    camera_ids: list,
    encoded_sequence_loader: str,
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
    zarr_store_type: str,
    open_consolidated: bool,
    lidar_seg_ensemble_cuda: bool,
    cam_seg_fp: Path,
    output_path: Path,
    frame_timestamps_us: list[int],
    visualize: bool,
):
    """Worker: Lidar Segmentation."""
    with gpu_sem:
        torch.cuda.set_device(device_id)
        logging.info(
            "\n*********** Device Sanity Check ************\n"
            f"visible device counts: {torch.cuda.device_count()} \n"
            f"current device: {torch.cuda.current_device()} \n"
            f"device name: {torch.cuda.get_device_name(0)} \n"
            "*********** End of Device Sanity Check ************"
        )

        # Init the cam seg aux loader
        cam_seg_aux_loader = CamAuxDataLoader(cam_seg_fp)

        # construct the current loader and writer for each process
        sequence_loader = decode_sequence_loader_json(
            encoded_sequence_loader,
            poses_component_group,
            intrinsics_component_group,
            masks_component_group,
            cuboids_component_group,
            open_consolidated,
        )

        base_name, shard_id, shard_count = sequence_loader_shard_info(sequence_loader)

        estimator = LidarSegmentationByProjectEstimator(
            sequence_loader,
            cam_seg_aux_loader,
            camera_ids=camera_ids,
            ensemble_cuda=lidar_seg_ensemble_cuda,
        )

        # init the writer
        writer = AuxDataWriter(
            output_path,
            base_name,
            sequence_loader.sequence_id,
            shard_id,
            shard_count,
            zarr_store_type,
        )

        run_lidar_segmentation(
            estimator,
            sequence_loader,
            writer,
            lidar_id,
            frame_timestamps_us,
            visualize,
        )

        del estimator
        torch.cuda.empty_cache()

        # properly close permanent annotation shard
        writer.finalize()


def run_lidar_segmentation_parallel(
    sequence_loader: ncore.data.SequenceLoaderProtocol,
    params: CLIBaseParams,
    output_path: Path,
    cam_seg_fp: Path,
    lidar_id: str,
    camera_ids: list[str],
    frame_timestamps_us: list[int],
    num_threads: int,
):
    """Split the jobs based on the number of available GPU"""

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA device available")

    ctx = multiprocessing.get_context("spawn")
    gpu_sems = [ctx.BoundedSemaphore(int(np.ceil(num_threads / num_gpus))) for _ in range(num_gpus)]
    procs = []
    output_path_list = []

    def split_chunks(lst, n):
        """Split list into n roughly equal parts."""
        return [list(chunk) for chunk in np.array_split(lst, n)]

    frame_timestamp_chunks = split_chunks(frame_timestamps_us, num_threads)

    for idx, ts_list in enumerate(frame_timestamp_chunks):
        device_id = idx % num_gpus

        logging.info(f"Assign chunk {idx} -> GPU {device_id}")
        lidar_output_path = output_path / "lidar_seg" / f"chunk_{idx}"
        output_path_list.append(lidar_output_path)
        p = ctx.Process(
            target=lidar_segmentation_worker,
            args=(
                device_id,
                gpu_sems[device_id],
                lidar_id,
                camera_ids,
                encode_sequence_loader_json(sequence_loader),
                params.poses_component_group,
                params.intrinsics_component_group,
                params.masks_component_group,
                params.cuboids_component_group,
                params.zarr_store_type,
                params.open_consolidated,
                params.lidar_seg_ensemble_cuda,
                cam_seg_fp,
                lidar_output_path,
                ts_list,
                params.visualize,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    final_output_fp = merge_zarr_itar(
        root_dir=Path(output_path / "lidar_seg"),
        pattern="*.aux.lidar-sseg.zarr.itar",
        output_dir=output_path,
    )

    return final_output_fp


def run_mask2former_segmentation(
    segmentation_estimator: Mask2FormerSegmentationEstimator,
    writer: AuxDataWriter,
    camera_id: str,
    camera_model_parameters: ConcreteCameraModelParametersUnion,
    camera_sensor: ncore.data.CameraSensorProtocol,
    timestamped_image_frame_handles: list[Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]],
    visualize: bool = False,
    has_ego_mask: bool = False,
) -> None:
    """Run Mask2Former semantic segmentation for a camera's image frames.

    Predicts per-pixel semantic labels (and optionally logits) for each frame,
    stores metadata and results via the writer. When no camera sensor mask is
    available and has_ego_mask is True, uses the writer's aggregated egomask.

    Args:
        segmentation_estimator: Mask2Former model used for prediction.
        writer: Aux data writer for storing semantic meta, segmentations, and logits.
        camera_id: Identifier of the camera (used for writer keys and paths).
        camera_model_parameters: Camera intrinsics/resolution for the estimator.
        camera_sensor: NCore camera sensor; used to obtain sensor mask if available.
        timestamped_image_frame_handles: List of (timestamp_us, image_handle) to process.
        visualize: If True, write segmentation logits for debugging visualization.
        has_ego_mask: If True and no camera sensor mask is available, use the writer's
            aggregated egomask (writer.get_egomask). If False, do not request an
            egomask, avoiding creation of an empty egomask store (e.g. with --no-ego-mask).

    Returns:
        None.
    """
    logger = logging.getLogger(__name__)
    segmentation_estimator.set_resolution(camera_model_parameters.resolution.tolist())

    logger.debug(f"Storing semantic meta {(camera_id, *segmentation_estimator.get_semantic_metadata().values())}")
    writer.store_semantic_meta(camera_id, **segmentation_estimator.get_semantic_metadata())

    camera_mask = get_camera_sensor_mask(camera_sensor)
    if camera_mask is None and has_ego_mask:
        logger.info("Using aux generated ego-mask (aggregated) from writer.")
        camera_mask = writer.get_egomask(camera_id, 0)

    for frame_timestamp_us, frame_image_handle in tqdm.tqdm(timestamped_image_frame_handles):
        encoded_img_data = frame_image_handle.get_data()
        img = encoded_img_data.get_decoded_image()
        if visualize:
            vis_path = (
                writer.output_dir_path / "seg_logits" / camera_id / writer.store_base_name / str(frame_timestamp_us)
            )
        else:
            vis_path = None
        segmentation = segmentation_estimator.predict(img, vis_path=vis_path, ego_mask=camera_mask)

        semantic_seg = segmentation.semantic_seg
        semantic_seg_logits = segmentation.semantic_seg_logits

        assert semantic_seg is not None

        # writing results to shard
        logger.debug(f"Storing semantic {(camera_id, frame_timestamp_us, semantic_seg, 'png')}")
        writer.store_semantic_segmentation(camera_id, frame_timestamp_us, semantic_seg, "png")

        if semantic_seg_logits is not None:
            logger.debug(f"Storing semantic logits {(camera_id, frame_timestamp_us)}")
            writer.store_semantic_logits(camera_id, frame_timestamp_us, semantic_seg_logits)


def run_dinov2_feature_precomputation(
    dinov2_backend: str,
    dinov2_width: int,
    dinov2_pca_dim: int,
    loader: ncore.data.SequenceLoaderProtocol,
    sensor_ranges: SensorRanges,
) -> tuple[Optional[PCA], Optional[Feature2ColorTransform]]:
    dinov2_pca: Optional[PCA] = None
    dinov2_color_transform: Optional[Feature2ColorTransform] = None
    logger = logging.getLogger(__name__)

    match dinov2_backend:
        case "none":
            pass

        case _:
            dinov2_features_all = []

            for camera_id, camera_frame_range in sensor_ranges.camera_frame_ranges.items():
                camera_sensor = loader.get_camera_sensor(camera_id)

                # Collect all image handles with timestamps and restrict to active frames
                # Sample every 15 frames here to reduce computation burden.
                timestamped_image_frame_handles_subset: list[
                    Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]
                ] = list(
                    zip(
                        camera_sensor.get_frames_timestamps_us(),
                        [camera_sensor.get_frame_handle(i) for i in range(camera_sensor.frames_count)],
                    )
                )[camera_frame_range.start : camera_frame_range.stop : 15]

                camera_model_parameters = camera_sensor.model_parameters
                dinov2_estimator = DINOv2Estimator(
                    DINOv2Estimator.resolution_from_dino_width(
                        camera_model_parameters.resolution.tolist(), dinov2_width
                    ),
                    backend=dinov2_backend,
                )

                logger.debug(f"Precompute DINOv2 for {(camera_id, *dinov2_estimator.get_metadata().values())}")

                camera_mask: Optional[np.ndarray] = get_camera_sensor_mask(camera_sensor)

                dinov2_feats_list: list[np.ndarray] = []

                for _, frame_image_handle in tqdm.tqdm(timestamped_image_frame_handles_subset):
                    encoded_img_data = frame_image_handle.get_data()
                    img = encoded_img_data.get_decoded_image()

                    dinov2_feats, dinov2_mask = dinov2_estimator.predict(img, camera_mask)
                    if dinov2_mask is not None:
                        dinov2_feats = dinov2_feats[~dinov2_mask]
                    dinov2_feats_list.append(dinov2_feats)

                dinov2_features_all.extend(dinov2_feats_list)

                del dinov2_estimator
                torch.cuda.empty_cache()

            dinov2_features_sample = np.concatenate(dinov2_features_all)

            if dinov2_pca_dim > 0:
                # Parameter following EmerNeRF practice (n_iter = 20, no additional centralizing)
                dinov2_pca = PCA.from_data(dinov2_features_sample, q=dinov2_pca_dim, n_iter=20, data_centralized=True)
                dinov2_features_sample = dinov2_pca.transform(dinov2_features_sample)

            assert isinstance(dinov2_features_sample, np.ndarray)
            dinov2_color_transform = Feature2ColorTransform.from_feature(dinov2_features_sample)

    return dinov2_pca, dinov2_color_transform


def run_dinov2_feature_extraction(
    dinov2_estimator: DINOv2Estimator,
    writer: AuxDataWriter,
    camera_id: str,
    camera_sensor: ncore.data.CameraSensorProtocol,
    timestamped_image_frame_handles: list[Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]],
    pca: Optional[PCA],
    color_transform: Feature2ColorTransform,
    visualize: bool = False,
) -> None:
    logger = logging.getLogger(__name__)
    logger.debug(f"Storing DINOv2 feature for {(camera_id, *dinov2_estimator.get_metadata().values())}")

    writer.store_dinov2_meta(
        camera_id, dinov2_estimator.get_metadata(), color_transform.to_dict(), pca.to_dict() if pca else None
    )

    # Ego mask is used to determine if DINOv2 feature is valid for reconstruction.
    camera_mask: Optional[np.ndarray] = get_camera_sensor_mask(camera_sensor)

    for frame_timestamp_us, frame_image_handle in tqdm.tqdm(timestamped_image_frame_handles):
        encoded_img_data = frame_image_handle.get_data()
        img = encoded_img_data.get_decoded_image()
        dinov2_feats, dino_ego_mask = dinov2_estimator.predict(img, ego_mask=camera_mask)

        if pca is not None:
            dinov2_feats = cast(np.ndarray, pca.transform(dinov2_feats))

        dino_valid_mask = None
        if dino_ego_mask is not None:
            dino_valid_mask = ~dino_ego_mask

        logger.debug(f"Storing semantic dinov2 feats {(camera_id, frame_timestamp_us)}")
        writer.store_dinov2(camera_id, frame_timestamp_us, dinov2_feats, dino_valid_mask)

        if visualize:
            vis_path = writer.output_dir_path / "dinov2_feats" / camera_id / writer.store_base_name
            os.makedirs(vis_path, exist_ok=True)
            dinov2_feats_rgb = color_transform.transform(dinov2_feats)
            plt.imsave(str(vis_path / f"{frame_timestamp_us}.jpg"), dinov2_feats_rgb)


def run_depth_estimation(
    depth_estimator: DepthAnythingV2Estimator | MetricDepthAnythingV2Estimator,
    writer: AuxDataWriter,
    store_depth_as_png: bool,
    camera_id: str,
    timestamped_image_frame_handles: list[Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]],
    visualize: bool = False,
) -> None:
    logger = logging.getLogger(__name__)

    logger.debug(f"Storing depth estimation meta")

    # Get the colormap in case we use the visualization
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")  # type: ignore

    for frame_idx, (frame_timestamp_us, frame_image_handle) in enumerate(tqdm.tqdm(timestamped_image_frame_handles)):
        # Invert the image channels as DepthAnything expects a BGR image
        img = np.asarray(frame_image_handle.get_data().get_decoded_image())[:, :, ::-1]
        depth = depth_estimator.predict(img).squeeze(axis=0)

        if frame_idx == 0:
            # Store the depth metadata
            writer.store_depth_meta(
                camera_id,
                resolution=list([depth.shape[1], depth.shape[0]]),
                store_depth_as_png=store_depth_as_png,
                max_depth_m=depth_estimator.max_depth_m,
                method=depth_estimator.method,
            )

        if visualize:
            vis_folder = writer.output_dir_path / "depth" / camera_id / writer.store_base_name
            os.makedirs(str(vis_folder), exist_ok=True)

            # Convert the predicted depth to color and store it
            vis_depth = np.copy(depth)
            vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min()) * 255.0
            vis_depth = vis_depth.astype(np.uint8)
            vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            cv2.imwrite(str((vis_folder / (str(frame_timestamp_us) + ".jpeg"))), vis_depth)

        # writing results to the shard
        logger.debug(
            f"Storing depth estimation {(camera_id, frame_timestamp_us, depth, 'png' if store_depth_as_png else 'npy')}"
        )
        writer.store_depth(camera_id, frame_timestamp_us, depth)


def run_lidar_segmentation(
    estimator: LidarSegmentationByProjectEstimator,
    loader: ncore.data.SequenceLoaderProtocol,
    writer: AuxDataWriter,
    lidar_id: str,
    frame_timestamps_us: list[int],
    visualize: bool = False,
) -> None:
    logger = logging.getLogger(__name__)

    lidar_sensor = loader.get_lidar_sensor(lidar_id)

    estimator.set_meta()
    logger.debug(f"Storing semantic meta {(lidar_id, *estimator.get_semantic_metadata().values())}")
    writer.store_lidar_semantic_meta(lidar_id, **estimator.get_semantic_metadata())
    logger.debug(f"Storing lidar-camera-visibility meta {(lidar_id, estimator.get_visibility_metadata())}")
    writer.store_lidar_camera_visibility_meta(lidar_id, **estimator.get_visibility_metadata())

    if visualize:
        vis_path = writer.output_dir_path / "lidar_seg" / lidar_id / writer.store_base_name
        os.makedirs(vis_path, exist_ok=True)
        for camera_id in estimator.camera_ids:
            os.makedirs(vis_path / camera_id, exist_ok=True)
    else:
        vis_path = None

    if not estimator.ensemble_cuda:
        logging.info(
            f"Start lidar segmentation. If you find the speed is too slow (< 1it/s), please switch to cuda version, lower "
            f" the number of numba threads (using cli --numba-num-threads), or stop other tasks with high CPU usage."
        )
    for frame_timestamp_us in tqdm.tqdm(frame_timestamps_us):
        lidar_seg, visibility_mask = estimator.predict(
            lidar_sensor=lidar_sensor,
            frame_timestamp_us=frame_timestamp_us,
            vis_path=vis_path,
        )
        # writing results to shard
        logger.debug(f"Storing semantic {(lidar_id, frame_timestamp_us, lidar_seg)}")
        writer.store_lidar_semantic_segmentation(lidar_id, frame_timestamp_us, lidar_seg, "png")
        writer.store_lidar_camera_visibility(lidar_id, frame_timestamp_us, visibility_mask)


def visualize_egomask_overlay(
    ego_mask: np.ndarray,
    frame_np: np.ndarray,
    vis_folder: Path,
    frame_timestamp_us: int,
    vis_name: str,
) -> None:
    color = np.array([0, 255, 0], dtype=np.uint8)
    mask_color = np.where(ego_mask[..., None], color, frame_np)
    # blend with original frame using alpha blending
    out_frame = cv2.addWeighted(frame_np, 0.8, mask_color, 0.2, 0)
    # save overlay image
    PILImage.fromarray(out_frame).save(vis_folder / f"{frame_timestamp_us}_{vis_name}.png", optimize=True)


def run_egomask_estimation(
    estimator: PromptlessSAM2EgoMaskEstimator,
    writer: AuxDataWriter,
    camera_id: str,
    camera_model_parameters: ConcreteCameraModelParametersUnion,
    timestamped_image_frame_handles: list[Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]],
    aggregation_method: Literal["majority"] = "majority",
    visualize: bool = False,
) -> None:
    """Generate and store ego-mask instance-segmentation with finetuned promptless SAM2 estimator"""
    resolution = camera_model_parameters.resolution

    # store ego-mask estimator metadata
    writer.store_egomask_meta(
        camera_id,
        resolution=resolution.tolist(),
        dataset_name=estimator.dataset_name,
        method=estimator.method,
        pretrained_checkpoint=estimator.pretrained_checkpoint,
        aggregation_method=aggregation_method,
    )

    vis_folder = None
    if visualize:
        # overlay mask on original frame
        vis_folder = writer.output_dir_path / "ego_mask" / camera_id / writer.store_base_name
        os.makedirs(vis_folder, exist_ok=True)

    ego_mask_bin = np.zeros((int(resolution[1]), int(resolution[0])), dtype=np.uint64)

    # store ego-mask for each frame
    for _, (frame_timestamp_us, frame_image_handle) in enumerate(tqdm.tqdm(timestamped_image_frame_handles)):
        img = frame_image_handle.get_data().get_decoded_image()
        ego_mask = estimator.predict(img)  # binary mask

        # store ego-mask for current frame
        writer.store_egomask(
            camera_id,
            frame_timestamp_us,
            ego_mask,
            image_file_format="png",
        )

        if aggregation_method == "majority":
            # accumulate binary masks
            ego_mask_bin += ego_mask

        # visualize ego-mask overlayed on original frame
        if vis_folder:
            visualize_egomask_overlay(ego_mask, np.array(img), vis_folder, frame_timestamp_us, "raw_ego")

    if aggregation_method == "majority":
        # majority voting
        num_masks = len(timestamped_image_frame_handles)
        ego_mask = (ego_mask_bin >= num_masks / 2).astype(np.uint8)

        # store aggregated ego-mask as super ego-mask
        writer.store_egomask(
            camera_id,
            0,  # dummy frame timestamp for super ego-mask
            ego_mask,
            image_file_format="png",
        )

    # visualize super ego-mask overlayed on all frames
    # loop over all samples again instead of storing raw frames
    if vis_folder:
        for frame_timestamp_us, frame_image_handle in tqdm.tqdm(timestamped_image_frame_handles):
            frame_np = np.array(frame_image_handle.get_data().get_decoded_image())
            visualize_egomask_overlay(ego_mask, frame_np, vis_folder, frame_timestamp_us, "agg_ego")


@click.group(invoke_without_command=True)
# Deprecated low-level V3 entry point
@click.option(
    "--shard-file-pattern",
    type=str,
    help="V3 entry point for data shard pattern to load (supports range expansion)",
    required=False,
    deprecated="Please use --dataset-path instead",
)
# V3/V4 entry point
@click.option("--dataset-path", type=str, help="Path to a NCore V3/V4 sequence meta-file", required=False)
@click.option("--poses-component-group", type=str, help="V4 component group for 'poses'", default="default")
@click.option("--intrinsics-component-group", type=str, help="V4 component group for 'intrinsics'", default="default")
@click.option("--masks-component-group", type=str, help="V4 component group for 'masks'", default="default")
@click.option("--cuboids-component-group", type=str, help="V4 component group for 'cuboids'", default="default")
# Common options
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be used (multiple value option, all if not specified)",
    default=None,
)
@click.option(
    "--lidar-id",
    "lidar_ids",
    multiple=True,
    type=str,
    help="Lidars to be used (multiple value option, all if not specified)",
    default=None,
)
@click.option(
    "--segmentation-backend",
    default="mask2former",
    type=click.Choice(["none", "mask2former"], case_sensitive=False),
    help="Perform segmentation, please choose a backend. \
        (0)None, not to perform segmentation.\
        (1)Mask2Former, supporting semantic segmentation and logits saving via --seg-logits (optional).",
)
@click.option(
    "--seg-logits/--no-seg-logits", is_flag=True, default=False, help="Perform semantic segmentation and save logits"
)
@click.option("--enable-trt/--disable-trt", is_flag=True, default=True, help="Enable running TRT optimized models")
@click.option(
    "--dinov2-backend",
    default="none",
    type=click.Choice(
        ["none", "nv_dinov2", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"], case_sensitive=False
    ),
    help="DINOv2 backbone to be used for feature extraction",
)
@click.option(
    "--dinov2-pca-dim",
    type=int,
    default=-1,
    help="PCA dimension for the features to be extracted (-1 means not to apply PCA).",
)
@click.option(
    "--dinov2-width",
    type=int,
    default=256,
    help="DINOv2 feature width (default 256)",
)
@click.option(
    "--lidar-seg-camvis/--no-lidar-seg-camvis",
    is_flag=True,
    default=True,
    help="Perform lidar segmentation and point-in-cameras visibility determination",
)
@click.option(
    "--lidar-seg-ensemble-cuda/--no-lidar-seg-ensemble-cuda",
    is_flag=True,
    default=True,
    help="Wether to use cuda-based ensemble function for lidar segmentation",
)
@click.option(
    "--depth-backend",
    default="none",
    type=click.Choice(["none", "depthanythingv2"], case_sensitive=False),
    help="Perform depth estimation, please choose a backend. \
        (0) None, not to perform depth estimation.\
        (1) Depthanythingv2 using small model.",
)
@click.option(
    "--relative-depth",
    is_flag=True,
    default=False,
    help="Estimate the relative depth (as opposed to metric)",
)
@click.option(
    "--max-depth-m",
    type=float,
    default=80.0,
    help="The maximum metric depth predicted by the metric depth estimation network. For relative depth this parameter has no effect and the values will be normalized to the range [0, 1] \
         More information on selecting the max depth https://github.com/DepthAnything/Depth-Anything-V2/issues/147 the default for outdoor is 80.0",
)
@click.option(
    "--depth-input-resolution",
    type=int,
    default=1036,
    help="The resolution of the inputs to the depth estimation network",
)
@click.option(
    "--store-depth-as-png",
    is_flag=True,
    default=False,
    help="Store depth in a quantized form (as PNG)",
)
@click.option("--ego-mask/--no-ego-mask", is_flag=True, default=True, help="Perform automatic ego-mask estimation")
@click.option(
    "--ego-mask-samples-per-second",
    type=click.FloatRange(min=0.0002, max=30.0),
    default=0.2,
    help="Number of frames to sample for ego-mask estimation per second (default: 0.2).",
)
@click.option(
    "--ego-mask-aggregation-method",
    type=click.Choice(["majority"], case_sensitive=False),
    default="majority",
    help="Aggregation method for ego-mask estimation when using multiple samples.",
)
@click.option(
    "--ego-mask-camera-id",
    "ego_mask_camera_ids",
    multiple=True,
    type=str,
    default=None,
    help="Restrict ego-mask estimation to these camera IDs (default: all cameras in the run). Can be passed multiple times.",
)
@click.option(
    "--zarr-store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    help="Zarr store type to store the aux data in",
)
@click.option(
    "--open-consolidated/--no-open-consolidated", is_flag=True, default=True, help="Open shards consolidated meta-data"
)
@click.option(
    "--num-threads",
    type=str,
    help="Number of threads to use (use 'auto' to determine number of threads from current cpu count)",
    is_flag=False,
    flag_value="auto",
    default="8",
)
@click.option(
    "--parallel-mode/--no-parallel-mode",
    is_flag=True,
    default=False,
    help="Enable parallel processing mode to run multiple lidar or camera segmentation tasks concurrently",
)
@click.option(
    "--workers-per-gpu",
    type=int,
    default=6,
    help="Number of parallel workers to launch per GPU device.",
)
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging outputs")
@click.option("--visualize", is_flag=True, default=False, help="Enable outputting visualization results")
@click.option(
    "--store-meta",
    is_flag=True,
    default=False,
    help="Store meta-file per shard with CLI arguments and maglev runtime logging (if available)",
)
@nre_utils_cli.scopedtimer_cli_options(print_func=logging.info)
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """Processes inferred aux-data signals for NCore shards"""

    params = CLIBaseParams(**kwargs)

    # Initialize basic top-level logger configuration
    logging.basicConfig(
        level=logging.DEBUG if params.debug else logging.INFO,
        format="<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s",
    )

    ctx.obj = params

    if ctx.invoked_subcommand is None:
        # No subcommand was called - run default processing without range restriction for all enabled sensors

        def range_callback(sequence_loader: ncore.data.SequenceLoaderProtocol):
            camera_ids = params.camera_ids if params.camera_ids else sequence_loader.camera_ids
            lidar_ids = params.lidar_ids if params.lidar_ids else sequence_loader.lidar_ids

            return SensorRanges(
                camera_frame_ranges={
                    camera_id: range(sequence_loader.get_camera_sensor(camera_id).frames_count)
                    for camera_id in camera_ids
                },
                lidar_frame_ranges={
                    lidar_id: range(sequence_loader.get_lidar_sensor(lidar_id).frames_count) for lidar_id in lidar_ids
                },
            )

        try:
            ncore_aux_data(params, range_callback)
        finally:
            ScopedTimer.print_summary()


@cli.command(name="offset")
@click.option(
    "--sequence-seek-sec",
    type=click.FloatRange(min=0.0, max_open=True),
    default=None,
    help="Time to skip for processing each *individual* sequence (in seconds)",
)
@click.option(
    "--sequence-duration-sec",
    type=click.FloatRange(min=0.0, max_open=True),
    default=None,
    help="Restrict total duration of each *individual* sequence (in seconds)",
)
@click.pass_context
def offset(ctx, sequence_seek_sec: float | None, sequence_duration_sec: float | None) -> None:
    """Perform per-sequence time range offset to all sensor"""
    params: CLIBaseParams = ctx.obj

    def range_callback(loader: ncore.data.SequenceLoaderProtocol):
        camera_ids = params.camera_ids if params.camera_ids else loader.camera_ids
        lidar_ids = params.lidar_ids if params.lidar_ids else loader.lidar_ids

        start_timestamp_us, end_timestamp_us = time_bounds(
            [loader.sequence_timestamp_interval_us.start, loader.sequence_timestamp_interval_us.stop + 1],
            sequence_seek_sec,
            sequence_duration_sec,
        )

        shard_restriction_interval_us = HalfClosedInterval(
            start_timestamp_us, end_timestamp_us + 1
        )  # make sure to include end-timestamp in interval

        return SensorRanges(
            camera_frame_ranges={
                camera_id: shard_restriction_interval_us.cover_range(
                    loader.get_camera_sensor(camera_id).get_frames_timestamps_us()
                )
                for camera_id in camera_ids
            },
            lidar_frame_ranges={
                lidar_id: shard_restriction_interval_us.cover_range(
                    loader.get_lidar_sensor(lidar_id).get_frames_timestamps_us()
                )
                for lidar_id in lidar_ids
            },
        )

    ncore_aux_data(params, range_callback)
    ScopedTimer.print_summary()


@cli.command(name="sensor-frames")
@click.option(
    "--main-sensor-id",
    "main_sensor_ids",
    multiple=True,
    type=str,
    help="Main sensors to be used (multiple value option, requires at least one)",
    default=None,
)
@click.option(
    "--start-frame",
    type=click.IntRange(min=0, max_open=True),
    help="If provided, the initial frame index of *all* main sensors to be exported",
    default=None,
)
@click.option(
    "--stop-frame",
    type=click.IntRange(min=0, max_open=True),
    help="If provided, the past-the-end frame index of *all* main sensors to be exported",
    default=None,
)
@click.pass_context
def sensor_frames(ctx, main_sensor_ids: list[str], start_frame: int | None, stop_frame: int | None) -> None:
    """Perform per-shard time range restriction based on sensor frame numbers of main sensors

    Minimum / maximum time ranges will be inferred if multiple main sensors are specified

    """

    assert len(main_sensor_ids), "At least a single main sensor needs to be provided"

    params: CLIBaseParams = ctx.obj

    def range_callback(sequence_loader: ncore.data.SequenceLoaderProtocol):
        def sensor_timestamps(sensor_id: str) -> np.ndarray:
            if sensor_id in sequence_loader.camera_ids:
                return sequence_loader.get_camera_sensor(sensor_id).get_frames_timestamps_us()
            elif sensor_id in sequence_loader.lidar_ids:
                return sequence_loader.get_lidar_sensor(sensor_id).get_frames_timestamps_us()
            elif sensor_id in sequence_loader.radar_ids:
                return sequence_loader.get_radar_sensor(sensor_id).get_frames_timestamps_us()
            else:
                raise ValueError(f"Unknown main sensor id {sensor_id}")

        # determine combined ranges of all main time-sensors
        main_sensors_timestamps_us = [
            sensor_timestamps(sensor_id)[start_frame:stop_frame] for sensor_id in main_sensor_ids
        ]
        start_timestamp_us = min(
            [main_sensor_timestamps_us.min() for main_sensor_timestamps_us in main_sensors_timestamps_us]
        )
        end_timestamp_us = max(
            [main_sensor_timestamps_us.max() for main_sensor_timestamps_us in main_sensors_timestamps_us]
        )

        shard_restriction_interval_us = HalfClosedInterval(
            start_timestamp_us, end_timestamp_us + 1
        )  # make sure to include end-timestamp in interval

        camera_ids = params.camera_ids if params.camera_ids else sequence_loader.camera_ids
        lidar_ids = params.lidar_ids if params.lidar_ids else sequence_loader.lidar_ids

        return SensorRanges(
            camera_frame_ranges={
                camera_id: shard_restriction_interval_us.cover_range(
                    sequence_loader.get_camera_sensor(camera_id).get_frames_timestamps_us()
                )
                for camera_id in camera_ids
            },
            lidar_frame_ranges={
                lidar_id: shard_restriction_interval_us.cover_range(
                    sequence_loader.get_lidar_sensor(lidar_id).get_frames_timestamps_us()
                )
                for lidar_id in lidar_ids
            },
        )

    ncore_aux_data(params, range_callback)
    ScopedTimer.print_summary()


def ncore_aux_data(params: CLIBaseParams, range_callback: SensorRangeCallback):
    # Determine and set number of numba threads to use
    num_threads = (
        multiprocessing.cpu_count() if (num_threads_param := params.num_threads) == "auto" else int(num_threads_param)
    )
    num_threads_str = str(num_threads)
    logging.info(f"Setting number of threads to {num_threads_str}")

    ## Initialize the sequence data loaders
    sequence_loaders: list[ncore.data.SequenceLoaderProtocol] = []
    assert bool(params.dataset_path) != bool(params.shard_file_pattern), (
        "Either shard-file-pattern or dataset-path need to be provided, but not both"
    )
    if params.shard_file_pattern is not None:
        logging.warning(
            "`--shard-file-pattern` is deprecated and will be removed in future releases. Please use `--dataset-path` instead."
        )
        # Treat each V3 shard as *individual* self-contained sequence
        sequence_loaders = [
            ncore_internal.data.v3.SequenceLoaderV3(
                ncore_internal.data.v3.ShardDataLoader(
                    shard_paths=[shard_input_path],
                    open_consolidated=params.open_consolidated,
                )
            )
            for shard_input_path in ncore_internal.data.v3.ShardDataLoader.evaluate_shard_file_pattern(
                params.shard_file_pattern
            )
        ]
    if params.dataset_path is not None:
        # Load V3 / V4 dataset meta-file and treat as single sequence
        (
            data_format,
            _,
            _,
            # contains either V3 zarr.itar shards, or V4 zarr.itar archives / zarr directories
            dataset_paths,
        ) = parse_sequence_meta_file(parse_universal_path(params.dataset_path, s3_cache_type="blockcache"))

        if data_format.name == "V3":
            # V3 single-sequence meta file
            sequence_loaders = [
                ncore_internal.data.v3.SequenceLoaderV3(
                    ncore_internal.data.v3.ShardDataLoader(
                        shard_paths=[str(data_path) for data_path in dataset_paths],
                        open_consolidated=params.open_consolidated,
                    )
                )
            ]
        else:
            sequence_loaders = [
                ncore.data.v4.SequenceLoaderV4(
                    ncore.data.v4.SequenceComponentGroupsReader(
                        dataset_paths, open_consolidated=params.open_consolidated
                    ),
                    poses_component_group_name=params.poses_component_group,
                    intrinsics_component_group_name=params.intrinsics_component_group,
                    masks_component_group_name=params.masks_component_group,
                    cuboids_component_group_name=params.cuboids_component_group,
                )
            ]

    # Set up output path
    (output_path := Path(params.output_dir)).mkdir(parents=True, exist_ok=True)

    # Check estimation dependencies
    if params.segmentation_backend != "mask2former" and params.seg_logits:
        raise ValueError("segmentation logits requires 'mask2former' segmentation backend")

    # Process each input sequence individually
    for sequence_loader in sequence_loaders:
        sensor_ranges = range_callback(sequence_loader)

        base_name, shard_id, shard_count = sequence_loader_shard_info(sequence_loader)

        # Initialize the output writer for current annotation container
        writer = AuxDataWriter(
            output_path,
            base_name,
            sequence_loader.sequence_id,
            shard_id,
            shard_count,
            params.zarr_store_type,
        )

        logging.info(
            f"Processing aux data for sequence {sequence_loader.sequence_id} ({sequence_loader.sequence_paths}) on cameras {sensor_ranges.camera_frame_ranges.keys()}"
        )

        # If PCA is needed for DINOv2 feature extraction, pre-compute the transformation with **all cameras** jointly.
        #   TODO [JH]: In the future, we might want to perform PCA on all shards provided.
        with ScopedTimer("aux_gen/dinov2_precomputation", tag=TimingTag.DEFAULT):
            dinov2_pca, dinov2_color_transform = run_dinov2_feature_precomputation(
                params.dinov2_backend, params.dinov2_width, params.dinov2_pca_dim, sequence_loader, sensor_ranges
            )

        cam_seg_aux_loader = None
        cam_seg_output_fp = None
        if params.parallel_mode and params.segmentation_backend == "mask2former":
            logging.info(
                f"Running semantic segmentation using mask2former for {sensor_ranges.camera_frame_ranges.keys()}"
            )
            with ScopedTimer("aux_gen/mask2former_parallel", tag=TimingTag.DEFAULT):
                cam_seg_output_fp = run_mask2former_parallel(
                    sensor_ranges=sensor_ranges,
                    sequence_loader=sequence_loader,
                    params=params,
                    output_path=output_path,
                )
            cam_seg_aux_loader = CamAuxDataLoader(cam_seg_output_fp)

        for camera_id, camera_frame_range in sensor_ranges.camera_frame_ranges.items():
            camera_sensor = sequence_loader.get_camera_sensor(camera_id)

            # Collect all image handles with timestamps and restrict to active frames
            timestamped_image_frame_handles: list[
                Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]
            ] = list(
                zip(
                    camera_sensor.get_frames_timestamps_us(),
                    [camera_sensor.get_frame_handle(i) for i in range(camera_sensor.frames_count)],
                )
            )[camera_frame_range.start : camera_frame_range.stop]

            camera_model_parameters = camera_sensor.model_parameters

            run_ego_mask = _run_ego_mask_for_camera(params.ego_mask, params.ego_mask_camera_ids, camera_id)
            if run_ego_mask:
                # Determine handles based on number of samples
                total_frames = len(timestamped_image_frame_handles)
                seq_duration_sec = (
                    timestamped_image_frame_handles[-1][0] - timestamped_image_frame_handles[0][0]
                ) / 1e6

                num_samples = max(1, int(params.ego_mask_samples_per_second * seq_duration_sec))

                if total_frames == 0:
                    raise ValueError(f"No frames available for ego-mask estimation on camera {camera_id}")
                if num_samples >= total_frames:
                    # Use all available frames if requested samples exceed total
                    sampled_handles = timestamped_image_frame_handles
                    logging.warning(
                        f"Requested {num_samples} samples but only {total_frames} frames available for camera {camera_id}, using all frames"
                    )
                else:
                    # Split frames into num_samples+1 chunks and sample the last frame of first num_samples chunks
                    # Avoids picking first and last frames and keeps a good buffer.
                    chunk_size = total_frames // (num_samples + 1)
                    sampled_handles = []
                    for i in range(num_samples):
                        # sample the last frame of each chunk
                        sample_idx = (i + 1) * chunk_size - 1
                        sampled_handles.append(timestamped_image_frame_handles[sample_idx])

                ego_mask_estimator = PromptlessSAM2EgoMaskEstimator()
                logging.info(
                    f"Running automatic ego-mask estimation (samples={len(sampled_handles)}, aggregation={params.ego_mask_aggregation_method}) on camera {camera_id}"
                )
                with ScopedTimer(
                    f"aux_gen/ego_mask_estimation/{camera_id}(samples={len(sampled_handles)})",
                    tag=TimingTag.DEFAULT,
                ):
                    run_egomask_estimation(
                        ego_mask_estimator,
                        writer,
                        camera_id,
                        camera_model_parameters,
                        sampled_handles,
                        aggregation_method=params.ego_mask_aggregation_method,
                        visualize=params.visualize,
                    )
                del ego_mask_estimator
                torch.cuda.empty_cache()

            match params.segmentation_backend:
                case "mask2former":
                    if not params.parallel_mode:
                        logging.info(
                            f"Running semantic segmentation using mask2former on {len(timestamped_image_frame_handles)} images of camera {camera_id}"
                        )
                        mask2former_segmentation_estimator = Mask2FormerSegmentationEstimator(
                            camera_model_parameters.resolution.tolist(),
                            estimate_logits=params.seg_logits,
                            enable_trt=params.enable_trt,
                        )
                        with ScopedTimer(
                            f"aux_gen/mask2former_segmentation/{camera_id}(n={len(timestamped_image_frame_handles)})",
                            tag=TimingTag.DEFAULT,
                        ):
                            run_mask2former_segmentation(
                                mask2former_segmentation_estimator,
                                writer,
                                camera_id,
                                camera_model_parameters,
                                camera_sensor,
                                timestamped_image_frame_handles,
                                visualize=params.visualize,
                                has_ego_mask=run_ego_mask,
                            )
                        del mask2former_segmentation_estimator
                        torch.cuda.empty_cache()

                case "none":
                    pass

                case _:
                    raise ValueError("depth_backend need to be set")

            match params.depth_backend:
                case "depthanythingv2":
                    logging.info(
                        f"Running {'relative' if params.relative_depth else 'metric'} depth estimation using DepthAnythingV2 on {len(timestamped_image_frame_handles)} images of camera {camera_id}"
                    )
                    depthanythingv2_depth_estimator: DepthAnythingV2Estimator | MetricDepthAnythingV2Estimator = (
                        DepthAnythingV2Estimator(max_depth_m=1.0, input_resolution=params.depth_input_resolution)
                        if params.relative_depth
                        else MetricDepthAnythingV2Estimator(
                            max_depth_m=params.max_depth_m, input_resolution=params.depth_input_resolution
                        )
                    )

                    with ScopedTimer(
                        f"aux_gen/depth_estimation/{camera_id}(n={len(timestamped_image_frame_handles)})",
                        tag=TimingTag.DEFAULT,
                    ):
                        run_depth_estimation(
                            depthanythingv2_depth_estimator,
                            writer,
                            params.store_depth_as_png,
                            camera_id,
                            timestamped_image_frame_handles,
                            visualize=params.visualize,
                        )

                    del depthanythingv2_depth_estimator
                    torch.cuda.empty_cache()

                case "none":
                    pass

                case _:
                    raise ValueError("depth_backend need to be set")

            match params.dinov2_backend:
                case "none":
                    pass

                case _:
                    dinov2_estimator = DINOv2Estimator(
                        DINOv2Estimator.resolution_from_dino_width(
                            camera_model_parameters.resolution.tolist(), params.dinov2_width
                        ),
                        backend=params.dinov2_backend,
                    )
                    logging.info(
                        f"Running DINOv2 feature extraction on {len(timestamped_image_frame_handles)} images of camera {camera_id}"
                    )
                    assert dinov2_color_transform is not None
                    with ScopedTimer(
                        f"aux_gen/dinov2_feature_extraction/{camera_id}(n={len(timestamped_image_frame_handles)})",
                        tag=TimingTag.DEFAULT,
                    ):
                        run_dinov2_feature_extraction(
                            dinov2_estimator,
                            writer,
                            camera_id,
                            camera_sensor,
                            timestamped_image_frame_handles,
                            pca=dinov2_pca,
                            color_transform=dinov2_color_transform,
                            visualize=params.visualize,
                        )
                    del dinov2_estimator
                    torch.cuda.empty_cache()

        for lidar_id, lidar_frame_range in sensor_ranges.lidar_frame_ranges.items():
            lidar_sensor = sequence_loader.get_lidar_sensor(lidar_id)

            # Collect timestamps of all frames to process
            frame_timestamps_us = list(
                lidar_sensor.get_frames_timestamps_us()[lidar_frame_range.start : lidar_frame_range.stop]
            )

            if params.lidar_seg_camvis:
                aux_loader: AuxDataCameraSemSegProvider
                if params.segmentation_backend != "none":
                    aux_loader = writer
                    if params.parallel_mode:
                        if cam_seg_aux_loader is None:
                            raise RuntimeError("cam_seg_aux_loader not initialized for parallel mode")
                        aux_loader = cam_seg_aux_loader
                else:
                    aux_loader = AuxShardDataLoader.from_sequence_loader(sequence_loader)
                    # prerequisite for lidar seg: check that all enabled cameras have semantic segmentation
                    if camera_ids_without_semantic_segmentation := [
                        camera_id
                        for camera_id in list(sensor_ranges.camera_frame_ranges.keys())
                        if not aux_loader.has_semantic_segmentation(camera_id)
                    ]:
                        raise RuntimeError(
                            f"Semantic segmentation aux_data not found for the following camera_id: {', '.join(camera_ids_without_semantic_segmentation)}"
                        )

                if params.parallel_mode and cam_seg_output_fp is not None:
                    cam_ids = list(sensor_ranges.camera_frame_ranges.keys())
                    with ScopedTimer(
                        f"aux_gen/lidar_segmentation/{lidar_id}(n={len(frame_timestamps_us)})",
                        tag=TimingTag.DEFAULT,
                    ):
                        run_lidar_segmentation_parallel(
                            sequence_loader=sequence_loader,
                            params=params,
                            output_path=output_path,
                            cam_seg_fp=cam_seg_output_fp,
                            lidar_id=lidar_id,
                            camera_ids=cam_ids,
                            frame_timestamps_us=frame_timestamps_us,
                            num_threads=num_threads,
                        )

                else:
                    lidar_seg_estimator = LidarSegmentationByProjectEstimator(
                        sequence_loader,
                        aux_loader,
                        camera_ids=list(sensor_ranges.camera_frame_ranges.keys()),
                        ensemble_cuda=params.lidar_seg_ensemble_cuda,
                    )
                    logging.info(
                        f"Running lidar semantic segmentation and point-in-cameras visibility on {len(frame_timestamps_us)} frames of point cloud {lidar_id}"
                    )
                    with ScopedTimer(
                        f"aux_gen/lidar_segmentation/{lidar_id}(n={len(frame_timestamps_us)})",
                        tag=TimingTag.DEFAULT,
                    ):
                        run_lidar_segmentation(
                            lidar_seg_estimator,
                            sequence_loader,
                            writer,
                            lidar_id=lidar_id,
                            frame_timestamps_us=frame_timestamps_us,
                            visualize=params.visualize,
                        )
                    del lidar_seg_estimator
                    torch.cuda.empty_cache()

        # properly close permanent annotation shard
        writer.finalize()

        # store meta-data of invocation arguments / cluster runtime information (if available)
        if params.store_meta:
            with open(writer.output_dir_path / f"{writer.store_base_name}.aux-meta.json", "w") as meta_json:
                meta = {"cli": asdict(params)}
                # store workflow runtime information if available
                if (workflow_identifier := os.environ.get("WORKFLOW_IDENTIFIER")) is not None:
                    meta["workflow"] = {
                        "id": workflow_identifier,
                        "date-time": os.environ["WORKFLOW_DATE_TIME"],
                    }
                json.dump(meta, meta_json, indent=4, sort_keys=True)


if __name__ == "__main__":
    cli(show_default=True)
