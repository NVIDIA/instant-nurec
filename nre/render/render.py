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
import os

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Self, Tuple, cast

import lietorch
import torch

from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter

from libs.sensors.kernels.cameras.bindings import generate_image_points
from libs.vren.interface import (  # type: ignore
    image_points_to_world_rays_shutter_pose,
    vren,
)
from libs.vren.lidars import (  # type: ignore
    elements_to_world_rays_shutter_pose,
)
from ncore.data import ConcreteCameraModelParametersUnion
from ncore.impl.common.transformations import transform_point_cloud
from ncore.sensors import CameraModel
from nre.artifact import Artifact
from nre.config.model import ModelConfig, RendererConfigType
from nre.config.trainer import TrainerConfig
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks
from nre.models.calib import FreePoseCalib
from nre.models.composite import CompositeModel
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.models.gaussians.renderers import BaseGaussianRenderer
from nre.models.gaussians.utils import Asset
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.nrm.primitives.kelvin_primitive import KelvinNRMPrimitive
from nre.render.actors import ActorsSnapshot, ActorTracks
from nre.render.utils import camera_model_to_parameters, frame_transform_poses, transform_intrinsics_to_resolution
from nre.utils.batch import (
    CameraFrameLabels,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    LidarFrameLabels,
    RenderingBatch,
    RenderingData,
    generate_grid_2d_indices,
)
from nre.utils.geometry import se3_matrix_inverse, tquat_to_se3_matrix
from nre.utils.lidar_model import LidarModelBundle
from nre.utils.lidar_post_processing import distance_based_filter
from nre.utils.misc import map_optional, to_float_device, tree_map, unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.types import (
    Checkpoint,
    FrameConversion,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    RayFlags,
    RigTrajectories,
    TrackFlags,
)
from nre.utils.upgrade import upgrade_config, upgrade_model


log = logging.getLogger(__name__)


@dataclass
class NRendWrapper:
    """Wrapper for invoking NRend from the Python rendering API internally."""

    # This wrapper is not meant to be used from client code, use the Python rendering API instead.

    tracks: Optional[CuboidTracks] = None

    @dataclass
    class NRendPoses:
        num_active_tracks: int = 0
        active_tracks_global_idx: Optional[torch.Tensor] = None
        start_track_poses: Optional[torch.Tensor] = None
        end_track_poses: Optional[torch.Tensor] = None

        @classmethod
        def from_actors_snapshot(
            cls,
            actors_snapshot: ActorsSnapshot,
            track_poses_nre: torch.Tensor,
            active_track_ids: list[str],
            device: torch.device,
        ) -> NRendWrapper.NRendPoses:
            """Create NRend pose data from actor snapshot for rendering.

            Transform actor poses in an ActorsSnapshot to an NRend-compatible pose representation.

            Args:
                actors_snapshot: Snapshot containing actor pose and metadata. Not modified.
                track_poses_nre: track poses in NRE coordinates. Not modified. [N, 2, 7]
                active_track_ids: List of track IDs from the model's cuboid tracks.
                device: Target torch device for tensor allocation.

            Returns:
                NRendPoses object containing transformed pose data.
            """
            if actors_snapshot.num_actors() == 0:
                return NRendWrapper.NRendPoses()

            track_global_idx = []
            for track_id in actors_snapshot.actor_ids:
                if track_id not in active_track_ids:
                    raise ValueError(
                        f"Trying to update actor with track_id {track_id!r} that is not in active_track_ids: {active_track_ids}"
                    )
                global_idx = active_track_ids.index(track_id)
                track_global_idx.append([global_idx, global_idx])

            return cls(
                num_active_tracks=len(track_global_idx),
                active_tracks_global_idx=torch.tensor(track_global_idx, dtype=torch.int32, device=device),
                start_track_poses=track_poses_nre[:, 0],  # [N, 7]
                end_track_poses=track_poses_nre[:, 1],  # [N, 7]
            )

    @staticmethod
    def supports_model(model: Any) -> bool:
        return isinstance(model, GaussiansComposite)

    def __call__(
        self,
        model: GaussiansComposite,
        rendering_data: RenderingData,
        frame_meta: FrameMeta,
        nrend_actor_poses: Optional[NRendWrapper.NRendPoses] = None,
    ) -> GaussiansCompositeReturn:
        # Commenting out to not alter behavior for gRPC but it should be enabled in the long run
        # if not NRendWrapper.supports_model(model):
        #     raise TypeError(f"NRend call does not support models of type {type(model)}")
        if nrend_actor_poses is not None:
            results = model.render_nrend_sensor_rays_with_poses(
                0,  # frame sequential id, used as random seed
                rendering_data,
                frame_meta,
                nrend_actor_poses.num_active_tracks,
                nrend_actor_poses.active_tracks_global_idx,
                nrend_actor_poses.start_track_poses,
                nrend_actor_poses.end_track_poses,
            )
        else:
            results = model.render_nrend_sensor_rays(
                0,  # frame sequential id, used as random seed
                rendering_data,
                frame_meta,
                tracks=self.tracks,
            )

        return GaussiansCompositeReturn(rendered_cam=results)


@dataclass
class CameraFrame:
    """Rendered framebuffers of shape (height, width, channels) returned from rendering functions.

    Fields are optional to enable the rendering function to only return a subset based on rendering options.

    Fields:
      color_image: RGB image of shape (height, width, 3) as torch.float32 with values in [0, 1]
      distance_image: Distance image of shape (height, width) and type torch.float32
      opacity_image: Opacity image of shape (height, width) and type torch.float32
    """

    color_image: Optional[torch.Tensor] = None
    distance_image: Optional[torch.Tensor] = None
    opacity_image: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.color_image is not None:
            assert len(self.color_image.shape) == 3, "Color image tensor does not have 3 dimensions"
            assert self.color_image.shape[2] == 3, "Color image does not have 3 channels"

        if self.distance_image is not None:
            assert len(self.distance_image.shape) == 2, "Depth image tensor does not have 2 dimensions"

        if self.opacity_image is not None:
            assert len(self.opacity_image.shape) == 2, "Opacity image tensor does not have 2 dimensions"


CameraFrameFields = Literal["color_image", "distance_image", "opacity_image"]


@dataclass
class LidarFrame:
    """A simulated LiDAR "frame", containing a point cloud typically from a single revolution
    (revolving LiDAR) or a single shot (solid-state LiDAR)

    Fields:
      point_positions: float32 tensor of shape (n, 3), position of each scanned point
      point_intensities: optional float32 tensor of shape (n,), intensity of each point
    """

    point_positions: torch.Tensor
    point_intensities: Optional[torch.Tensor] = None

    def __post_init__(self):
        assert len(self.point_positions.shape) == 2, "Point positions tensor does not have 2 dimensions"
        assert self.point_positions.shape[1] == 3, "Point positions tensor does not have 3 dimensions"
        if self.point_intensities is not None:
            assert len(self.point_intensities.shape) == 1, "Intensity tensor does not have 1 dimension"
            assert self.point_intensities.shape[0] == self.point_positions.shape[0], (
                "Intensity tensor does not have the same number of points as point positions tensor"
            )


@dataclass
class LidarRenderingSettings:
    """Settings for lidar rendering."""

    lidar_raydrop_threshold: float = 0.0
    lidar_opacity_threshold: float = 0.8
    lidar_post_filter_threshold: float = 0.02
    enable_lidar_post_filter: bool = False

    def __post_init__(self):
        assert self.lidar_raydrop_threshold >= 0.0 and self.lidar_raydrop_threshold <= 1.0
        assert self.lidar_opacity_threshold >= 0.0 and self.lidar_opacity_threshold <= 1.0
        assert self.lidar_post_filter_threshold >= 0.0 and self.lidar_post_filter_threshold <= 1.0


@dataclass
class PoseRange:
    start_pose_tquat_sensor_world: torch.Tensor  # 7D translation-quaternion Tensor of shape (7,)
    end_pose_tquat_sensor_world: torch.Tensor  # 7D translation-quaternion Tensor of shape (7,)
    start_timestamp_us: int  # Frame-start timestamp
    end_timestamp_us: int  # Frame-end timestamp

    def __post_init__(self):
        assert self.start_pose_tquat_sensor_world.shape == (7,)
        assert self.end_pose_tquat_sensor_world.shape == (7,)
        assert self.start_timestamp_us <= self.end_timestamp_us


@dataclass
class SensorTrajectory:
    """Frame acquisition start/end timestamps and sensor-to-world poses for a sequence of frames from a sensor"""

    # TODO: Turn into tquat (n_frames, 2, 7) for consistency with PoseRange and to restrict the degrees of freedom.
    poses_startend_sensor_world: torch.Tensor  # (n_frames, 2, 4, 4)
    timestamps_startend_us: torch.Tensor  # (n_frames, 2)

    def __post_init__(self):
        assert self.timestamps_startend_us.ndim == 2
        assert self.poses_startend_sensor_world.ndim == 4
        n_frames = self.timestamps_startend_us.shape[0]
        assert self.poses_startend_sensor_world.shape == (n_frames, 2, 4, 4)
        assert self.timestamps_startend_us.shape == (n_frames, 2)

    def __len__(self):
        return self.timestamps_startend_us.shape[0]


@dataclass
class RayBundle:
    """Rays sampled over a raster from a single camera frame."""

    rendering_data: RenderingData
    frame_meta: FrameMeta

    @classmethod
    @torch.autocast(device_type="cuda", enabled=False)
    def build(
        cls,
        camera_model_parameters: ConcreteCameraModelParametersUnion,
        camera_to_world: PoseRange,
        world_to_nre: FrameConversion,
        unique_sensor_idx: Optional[int] = None,
        unique_frame_idx: Optional[int] = None,
    ) -> RayBundle:
        """Generate a ray bundle in NRE model space from a single camera view defined by its camera intrinsics and pose
        with optional rolling shutter support.

        Args:
          - camera_model: Lens model of the camera view.
          - camera_to_world: Frame start and end poses (camera-to-world transforms) and corresponding timestamps for
            the camera view with rolling shutter (simply feed the same start/end poses/timestamps for global shutter).
          - world_to_nre: Transformation from world space to NRE model space.
          - unique_sensor_idx: Optional unique index of a camera used during training if the intrinsic parameters of
              the view to generate rays is from such a camera (accessible via SceneInfo.get_camera()).
          - unique_frame_idx: Optional unique index of a camera frame if the rays are to be generated from a view that
              matches or should be associated with a training view (accessible via SceneInfo.get_camera()).
        """
        with ScopedTimer("RayBundle.build"):
            device = torch.device("cuda")
            width, height = camera_model_parameters.resolution.tolist()

            # Generate a grid in 2D (float) pixel coordinates such that grid nodes fall in the center of the output pixels.
            with ScopedTimer("constructing_image_points"):
                image_points = generate_image_points(resolution=(width, height), device=device)

            # 4x4 transformation matrices from camera to world frame at the start- and end-time of the frame.
            with ScopedTimer("preparing_T_sensor_worlds"):
                poses_tquat_camera_world = torch.stack(
                    [camera_to_world.start_pose_tquat_sensor_world, camera_to_world.end_pose_tquat_sensor_world],
                    dim=0,
                )
                if poses_tquat_camera_world.device.type == "cpu":
                    poses_tquat_camera_world = poses_tquat_camera_world.pin_memory().to(
                        device=device, non_blocking=True
                    )
                else:
                    poses_tquat_camera_world = poses_tquat_camera_world.to(device=device)
                assert poses_tquat_camera_world.shape == (2, 7)

                # Apply extra transformation from the metric world to NRE model space to both start and end poses.
                poses_tquat_camera_nre = frame_transform_poses(world_to_nre, poses_tquat_camera_world, is_tquat=True)

            # Generate rays in the NRE model space through the sampled grid points in the image.
            with ScopedTimer("image_points_to_world_rays_shutter_pose"):
                timestamps_us = torch.tensor(
                    [camera_to_world.start_timestamp_us, camera_to_world.end_timestamp_us], device=device
                )
                world_rays = image_points_to_world_rays_shutter_pose(
                    camera_model_parameters,
                    image_points,
                    T_sensor_worlds=poses_tquat_camera_nre,
                    timestamps_us=timestamps_us,
                )

            with ScopedTimer("RenderingData::__init__"):
                # Rendering from any viewpoint should not require referring to a training sensor/view/trajectory.
                # Allow for missing camera / trajectory / frame index (i.e. training information) in general.
                # These are missing when we have a free (e.g. interactive) camera, completely independent of training views.
                # Workaround: selects a default training sensor, trajectory, frame, even for an independent/free viewpoint,
                # but this may undesirable side effects.
                # Keep GPU version for GPU operations, create CPU copy to avoid .item() calls
                timestamps_us_gpu = timestamps_us.unsqueeze(0)
                timestamps_us_cpu = torch.tensor(
                    [camera_to_world.start_timestamp_us, camera_to_world.end_timestamp_us],
                    device="cpu",
                    dtype=torch.int64,
                ).unsqueeze(0)
                rendering_data = RenderingData(
                    rays=world_rays.world_rays.reshape(1, height, width, 6),
                    sensor_model_parameters=[camera_model_parameters],
                    poses_tquat_startend=poses_tquat_camera_nre.unsqueeze(0),  # [1, 2, 7]
                    timestamps_startend_us=timestamps_us_gpu,  # [1, 2] - on GPU
                    rays_timestamps_us=map_optional(world_rays.timestamps_us, lambda x: x.reshape(1, height, width, 1)),
                    timestamps_startend_us_cpu=timestamps_us_cpu,  # [1, 2] - on CPU
                )
                frame_meta = FrameMeta(
                    unique_sensor_idx=unique_sensor_idx if unique_sensor_idx is not None else -1,
                    unique_frame_idx=unique_frame_idx if unique_frame_idx is not None else -1,
                )

            return cls(rendering_data=rendering_data, frame_meta=frame_meta)


@dataclass
class LidarRayBundle:
    """Rays sampled for a single lidar frame."""

    rendering_data: RenderingData
    frame_meta: FrameMeta

    # TODO: remove it in favor of rendering_data.poses_tquat_startend
    T_nre_sensor_end: torch.Tensor
    model_elements: torch.Tensor

    @classmethod
    def build(
        cls,
        lidar_model: LidarModelBundle,
        lidar_to_world: PoseRange,
        world_to_nre: FrameConversion,
        unique_sensor_idx: Optional[int] = None,
        unique_frame_idx: Optional[int] = None,
    ) -> LidarRayBundle:
        device = torch.device("cuda")
        start_pose_tquat_sensor_world = lidar_to_world.start_pose_tquat_sensor_world
        if start_pose_tquat_sensor_world.device.type == "cpu":
            start_pose_tquat_sensor_world = start_pose_tquat_sensor_world.pin_memory().to(
                device=device, non_blocking=True
            )
        else:
            start_pose_tquat_sensor_world = start_pose_tquat_sensor_world.to(device=device)
        end_pose_tquat_sensor_world = lidar_to_world.end_pose_tquat_sensor_world
        if end_pose_tquat_sensor_world.device.type == "cpu":
            end_pose_tquat_sensor_world = end_pose_tquat_sensor_world.pin_memory().to(device=device, non_blocking=True)
        else:
            end_pose_tquat_sensor_world = end_pose_tquat_sensor_world.to(device=device)
        tquat_sensor_nre_start = frame_transform_poses(
            world_to_nre,
            start_pose_tquat_sensor_world.reshape(-1, 7),
            is_tquat=True,
        )
        tquat_sensor_nre_end = frame_transform_poses(
            world_to_nre,
            end_pose_tquat_sensor_world.reshape(-1, 7),
            is_tquat=True,
        )
        T_nre_sensor_end = se3_matrix_inverse(tquat_to_se3_matrix(tquat_sensor_nre_end))
        virtual_model_elements = lidar_model.elements

        height = cast(int, lidar_model.lidar_model.n_rows)
        width = cast(int, lidar_model.lidar_model.n_columns)

        nre_rays = elements_to_world_rays_shutter_pose(
            vren_lidar_model_parameters=lidar_model.vren_lidar.parameters,
            element=torch.tensor(virtual_model_elements, dtype=torch.int32),
            T_sensor_worlds=torch.cat([tquat_sensor_nre_start, tquat_sensor_nre_end], dim=0),
            timestamps_us=torch.tensor(
                [lidar_to_world.start_timestamp_us, lidar_to_world.end_timestamp_us], dtype=torch.int64
            ),
        )

        rays = nre_rays.world_rays
        timestamps_us = nre_rays.timestamps_us

        # TODO: Make indices optional.
        if unique_sensor_idx is None:
            unique_sensor_idx = -1

        if unique_frame_idx is None:
            unique_frame_idx = -1

        frame_timestamps_startend_us = (lidar_to_world.start_timestamp_us, lidar_to_world.end_timestamp_us)

        # Keep GPU version for GPU operations, create CPU copy to avoid .item() calls
        device = rays.device
        timestamps_tensor_gpu = torch.tensor(frame_timestamps_startend_us, dtype=torch.int64, device=device).view(1, 2)
        timestamps_tensor_cpu = torch.tensor(frame_timestamps_startend_us, dtype=torch.int64, device="cpu").view(1, 2)
        rendering_data = RenderingData(
            rays=rays.reshape(1, height, width, 6),
            sensor_model_parameters=[lidar_model.lidar_parameters],
            poses_tquat_startend=torch.cat([tquat_sensor_nre_start, tquat_sensor_nre_end], dim=0).unsqueeze(
                0
            ),  # (1, 2, 7)
            timestamps_startend_us=timestamps_tensor_gpu,  # on GPU
            rays_timestamps_us=map_optional(timestamps_us, lambda x: x.reshape(1, height, width, 1)),
            timestamps_startend_us_cpu=timestamps_tensor_cpu,  # on CPU
        )

        frame_meta = FrameMeta(unique_sensor_idx=unique_sensor_idx, unique_frame_idx=unique_frame_idx)

        return cls(
            rendering_data=rendering_data,
            frame_meta=frame_meta,
            T_nre_sensor_end=T_nre_sensor_end,
            model_elements=torch.from_numpy(lidar_model.elements).to(T_nre_sensor_end.device),
        )


@dataclass
class RenderableModel:
    """High-level API unified across model types for rendering camera frames off a model."""

    _model: GaussiansComposite | BaseNRMPrimitive
    _nrend: NRendWrapper = field(default_factory=NRendWrapper)
    _autocast: Optional[torch.autocast] = None

    # The subset of methods that require it will raise if not provided.
    _world_to_nre: Optional[FrameConversion] = None

    _lidar_rendering_settings: LidarRenderingSettings = field(default_factory=LidarRenderingSettings)

    # Inference entry points (render CLI, serve-grpc, USDZ viewer, NRM NVS) never
    # surface the rasterized extra-signal tensors through their return types. Default
    # them off so the gsplat rasterizer instantiates at CDIM=4 (RGB+D) instead of
    # CDIM=24 (RGB + 20 semantic_logits + D) for artifacts trained with a semantic
    # head. Training uses a different entry point and is unaffected. Callers that
    # genuinely need the splatted extra-signal map can opt back in by passing the
    # matching override in `config_overrides` — later entries win the merge.
    _INFERENCE_DEFAULT_OVERRIDES: ClassVar[Tuple[str, ...]] = (
        "model.renderer.outputs.camera.enable_extended_features=False",
    )

    @classmethod
    def load_from_artifact(
        cls,
        artifact: Artifact,
        enable_nrend: bool,
        config_overrides: Tuple[str, ...] = (),
    ) -> Self:
        orig_untyped_config = OmegaConf.create(artifact.parsed_config)

        # Upgrade the config to the current version of the software, should the artifact be from an earlier version.
        untyped_config = upgrade_config(orig_untyped_config)

        # Prepend inference defaults so any caller-supplied override of the same
        # key wins (OmegaConf.from_dotlist applies entries left-to-right; the last
        # assignment to a key is the one that takes effect).
        config_overrides = cls._INFERENCE_DEFAULT_OVERRIDES + tuple(config_overrides)

        # Apply Hydra-style config overrides (e.g., model.renderer.name=gsplat)
        # NOTE: We're forced to apply the overrides here because load_from_artifact is actually doing two things:
        # loading the artifact from disk, and creating the RenderableModel instance. We can't apply the override
        # after the instance is created, not before the artifact is loaded. A better design would split these two steps,
        # allowing the override to be applied by the caller in between these operations
        if config_overrides:
            log.info(f"Applying {len(config_overrides)} Hydra-style override(s)")
            for override in config_overrides:
                log.info(f"  Override: {override}")

            # OmegaConf.from_dotlist handles format validation, parsing, type conversion, and nested key creation
            untyped_config = cast(
                DictConfig,
                OmegaConf.unsafe_merge(
                    untyped_config, cast(DictConfig, OmegaConf.from_dotlist(list(config_overrides)))
                ),
            )

        lidar_raydrop_threshold = OmegaConf.select(untyped_config, "system.test.lidar.raydrop_threshold", default=0.0)
        lidar_opacity_threshold = OmegaConf.select(untyped_config, "system.test.lidar.opacity_threshold", default=0.8)
        lidar_post_filter_threshold = OmegaConf.select(
            untyped_config, "system.test.lidar.save_filtered_pc.filter_threshold", default=0.02
        )
        enable_lidar_post_filter = OmegaConf.select(
            untyped_config, "system.test.lidar.save_filtered_pc.enabled", default=False
        )

        OmegaConf.resolve(untyped_config)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # If set, configure torch's default allocation block size - appending to any existing key:value configs.
        if (
            "max_split_size_mb" in untyped_config.system
            and (split_size := untyped_config.system.max_split_size_mb) is not None
        ):
            new_entry = f"max_split_size_mb:{split_size}"
            conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_entry if conf is None else ",".join([conf, new_entry])

        model: GaussiansComposite | BaseNRMPrimitive = cls._load_model_and_upgrade(
            untyped_config, artifact, orig_untyped_config
        )

        # Setup the fast CUDA renderer (falls back to PyTorch renderer).
        nrend = NRendWrapper()
        if enable_nrend:
            if not NRendWrapper.supports_model(model):
                raise TypeError(f"NRend does not support models of type {type(model)}")
            nrend.tracks = model.get_updated_cuboid_tracks() if isinstance(model, CompositeModel) else None
            model.setup_nrend(track_ids=nrend.tracks.tracks_id if nrend.tracks else [])
            assert model.has_nrend()

        world_to_nre = RigTrajectories.from_dict(artifact.rig_trajectories).world_to_nre

        return cls(
            model,
            nrend,
            torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16),
            world_to_nre,
            LidarRenderingSettings(
                lidar_raydrop_threshold, lidar_opacity_threshold, lidar_post_filter_threshold, enable_lidar_post_filter
            ),
        )

    @classmethod
    def _load_model_and_upgrade(
        cls, untyped_config: DictConfig, artifact: Artifact, orig_untyped_config: DictConfig
    ) -> GaussiansComposite | BaseNRMPrimitive:
        match untyped_config.model.name:
            case "gaussians_primitive" | "gaussians-composite":
                return cls._load_model_gaussians_composite_and_upgrade(untyped_config, artifact, orig_untyped_config)
            case "kelvin":
                return cls._load_model_nrm_and_upgrade(untyped_config, artifact, orig_untyped_config)
            case _:
                raise TypeError(f"Unsupported model name '{untyped_config.model.name}'")

    @classmethod
    def _load_model_gaussians_composite_and_upgrade(
        cls, untyped_config: DictConfig, artifact: Artifact, orig_untyped_config: DictConfig
    ) -> GaussiansComposite:
        model_config = ModelConfig.model_validate(untyped_config.model)

        # Log which renderer will be used
        renderer_name = model_config.renderer.name
        log.info(f"Using renderer: {renderer_name}")

        datasource_summary = DataSourceSummary.from_dict(
            artifact.datasource_summary,
            infer_missing=True,  # For better backward-compatibility.
        )
        trainer_config = TrainerConfig.model_validate(untyped_config.trainer)
        model = GaussiansComposite(
            model_config, trainer_config, datasource=datasource_summary, init_from_datasource=False
        )

        checkpoint = torch.load(artifact.checkpoint, weights_only=False)
        model_state_dict = {
            k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
        }
        # Upgrade the state_dict to the current version of the software, should the artifact be from an earlier version.
        upgraded_model_state_dict = upgrade_model(
            model_state_dict, orig_untyped_config, datasource_summary=datasource_summary
        )

        model.load_state_dict(upgraded_model_state_dict, assign=not checkpoint.get("load_in_place", True))
        model.to("cuda")
        model.eval()

        return model

    @classmethod
    def _load_model_nrm_and_upgrade(
        cls, untyped_config: DictConfig, artifact: Artifact, orig_untyped_config: DictConfig
    ) -> BaseNRMPrimitive:
        model_name = untyped_config.model.name
        checkpoint: Checkpoint = torch.load(artifact.checkpoint, weights_only=False)
        primitive: KelvinNRMPrimitive
        if model_name == "kelvin":
            primitive = KelvinNRMPrimitive.load_from_checkpoint(checkpoint)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        renderer_config: RendererConfigType = TypeAdapter(RendererConfigType).validate_python(
            untyped_config.model.renderer
        )
        gaussians_renderer = BaseGaussianRenderer.factory(
            renderer_config.name,
            renderer_config,
            primitive,  # type: ignore
        )
        primitive.gaussians_renderer = gaussians_renderer
        return primitive

    @classmethod
    def load_from_nrm_primitive(cls, nrm_primitive: BaseNRMPrimitive, world_to_nre: FrameConversion) -> Self:
        return cls(nrm_primitive, NRendWrapper(), None, world_to_nre)

    @property
    def device(self) -> torch.device:
        if isinstance(self._model, BaseNRMPrimitive):
            return self._model.device()
        return next(iter(self._model.parameters())).device

    @property
    def autocast(self):
        return self._autocast if self._autocast is not None else nullcontext()

    def get_all_gaussian_positions(self) -> torch.Tensor | None:
        """
        Get all Gaussian center positions from the underlying model.

        Returns:
            Optional[torch.Tensor]: Positions of all Gaussians in NRE frame, shape (N, 3),
                                   or None if not available
        """
        return self._model.get_all_gaussian_positions()

    def get_camera_trajectories(self, calibrated: bool = False) -> Dict[str, SensorTrajectory]:
        """Get all camera training view start/end poses and timestamps from the calib module and return them per camera.

        Args:
            calibrated: returns the calibrated poses if available when True, or the original input poses otherwise.

        Returns:
            A dictionary that maps unique camera ids (in the form "camera_id@sequence_id") to the training trajectory
            of the camera. Each trajectory consists of a frame start/end pose, as well as start/end timestamps
            for each training view of the camera, in increasing order of frame-end timestamps.
            Each pose is a 4x4 SE(3) matrix transforming points from sensor space to world space.
        """

        if not isinstance(self._model, GaussiansComposite):
            raise TypeError("Only GaussiansComposite is supported for now")
        # FreePoseCalib is always instantiated internally currently, even if calib is skipped or disabled.
        if not isinstance(self._model.calib, FreePoseCalib):
            raise TypeError("Only FreePoseCalib is supported for now")
        if self._model.calib.camera_view_geometry is None:
            raise ValueError("Camera view geometry is not enabled")
        if self._world_to_nre is None:
            raise ValueError("World-to-NRE transformation is required but not initialized")

        with torch.inference_mode():
            # Get the start/end timestamps and poses for all training frames/views vectorized over all cameras.
            timestamps_startend_us_allviews = self._model.calib.camera_view_geometry.get_timestamps()
            assert timestamps_startend_us_allviews.ndim == 2
            assert timestamps_startend_us_allviews.shape[1] == 2
            T_sensor_world_startend_allviews = self._model.calib.camera_view_geometry.get_poses(
                skip_calib=not calibrated
            )
            assert T_sensor_world_startend_allviews.ndim == 4
            assert T_sensor_world_startend_allviews.shape[1:] == (2, 4, 4)
            assert T_sensor_world_startend_allviews.shape[0] == timestamps_startend_us_allviews.shape[0]

        # Poses from the calib module are camera-to-nre space transforms, convert them to camera-to-world transforms.
        # The shape is (N, 2, 4, 4) but the transformation is only applicable to (M, 4, 4) tensors.
        num_frames = T_sensor_world_startend_allviews.shape[0]
        T_reshaped = T_sensor_world_startend_allviews.reshape(2 * num_frames, 4, 4)
        T_transformed = frame_transform_poses(self._world_to_nre.inverse(), T_reshaped, is_tquat=False)
        T_sensor_world_startend_allviews = T_transformed.reshape(num_frames, 2, 4, 4)

        # Extract the trajectory of each camera from the vectorized poses and timestamps.
        camera_trajectories: Dict[str, SensorTrajectory] = {}
        frame_ranges = self._model.calib.camera_view_geometry.get_frame_ranges_per_sensor()
        for unique_sensor_id, frame_range in frame_ranges.items():
            camera_trajectories[unique_sensor_id] = SensorTrajectory(
                poses_startend_sensor_world=T_sensor_world_startend_allviews[frame_range],
                timestamps_startend_us=timestamps_startend_us_allviews[frame_range],
            )

        return camera_trajectories

    def supports_edit_actors(self) -> bool:
        return (
            isinstance(self._model, CompositeModel)
            and isinstance(self._model.cuboid_tracks, CuboidTracks)
            and self._world_to_nre is not None
        )

    def get_actor_tracks(self) -> ActorTracks:
        """
        Returns the actor tracks in the world frame after applying the learned track calibration (delta poses)
        if the tracks_calib model was enabled during training.
        """
        if self._world_to_nre is None:
            raise ValueError("World-to-NRE transformation is required but not initialized")

        if not self.supports_edit_actors():
            raise ValueError(f"Actor tracks not supported by models of type {self._model.__class__.__name__}")

        assert isinstance(self._model, CompositeModel)
        cuboid_tracks = self._model.get_updated_cuboid_tracks()
        cuboid_tracks = CuboidTracks.Ops.transform_with_frame_conversion(
            cuboid_tracks, self._world_to_nre.inverse(), None
        )
        cuboid_tracks.tracks_data.tracks_flags |= TrackFlags.CONTROLLABLE
        return ActorTracks._from_cuboid_tracks(cuboid_tracks)

    def save_training_parameters(self) -> None:
        """
        Save the original pre-edit state of self._model that was loaded from the usdz.
        This includes both Gaussian parameters and structural track state required to undo grpc asset edits.
        """
        if isinstance(self._model, GaussiansComposite):
            self._model.save_training_parameters()
        else:
            raise NotImplementedError(f"Save training parameters not supported for model type {type(self._model)}")

    def restore_training_parameters(self) -> None:
        """
        Restore the original pre-edit state of self._model that was loaded from the usdz.
        This undoes both Gaussian parameter edits and structural grpc asset edits.
        """
        if isinstance(self._model, GaussiansComposite):
            self._model.restore_training_parameters()
        else:
            raise NotImplementedError(f"Restore training parameters not supported for model type {type(self._model)}")

    def replace_asset(self, track_id: str, asset: Asset, dims_offset: torch.Tensor) -> bool:
        """
        Replace the gaussians of a target track_id with gaussians from an Asset
        The Asset is the same as PlyGaussianLoader(nre/models/gaussians/utils.py) class

        Args:
            track_id: The track ID to override
            asset: Asset to use (preferred)
        """
        # Hard coded transform, we are replacing only the gaussians of an asset, not applying modifications to its pose
        transform = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cuda"
        )

        if isinstance(self._model, GaussiansComposite):
            try:
                return self._model.track_ply_override(
                    track_id, asset=asset, transform=transform, dims_offset=dims_offset
                )
            except Exception:
                log.exception(f"Failed to apply track PLY override for {track_id}")
                raise
        else:
            raise NotImplementedError(f"Track PLY override not supported for model type {type(self._model)}")

    def insert_asset(self, asset: Asset, cuboid_tracks: CuboidTracks, dims_offset: torch.Tensor) -> bool:
        """
        Insert track gaussians with associated CuboidTracks.

        Args:
            asset: Asset containing gaussian parameters
            cuboid_tracks: CuboidTracks data for the track (should be in NRE coordinates)
        """
        if not self.supports_edit_actors():
            raise RuntimeError("Model does not support actor updates")

        if not isinstance(self._model, GaussiansComposite):
            raise RuntimeError("Track insertion only supported for GaussiansComposite models")

        assert cuboid_tracks.n_tracks == 1, "Expected a single cuboid track for inserting a single asset"
        # Delegate to underlying model - cuboid_tracks should already be in NRE frame
        return self._model.insert_asset(asset, cuboid_tracks, dims_offset)

    @ScopedTimer()
    def _edit_actors(
        self,
        actors_snapshot: ActorsSnapshot,
        frame_start_us: int,
        frame_end_us: int,
    ) -> Tuple[Optional[NRendWrapper.NRendPoses], Optional[CuboidTracks]]:
        """
        Updates the cuboid poses of the controllable actors in the scene's model.

        Effectively this works by `freezing` the dynamic objects at the requested poses for all times.

        Args:
            actors_snapshot: start and end poses of all actors present at the start and end time of the frame capture.
            frame_start_us: start timestamp (in microseconds) of the frame captured.
            frame_end_us: end timestamp (in microseconds) of the frame captured.

        Returns:
            Tuple of (nrend_actor_poses, edited_cuboid_tracks). Either value may be None
            depending on renderer/model support and actor availability.
        """
        if not self.supports_edit_actors():
            raise NotImplementedError("Actor track updates not supported by model")

        if actors_snapshot.num_actors() == 0:
            return None, None

        nrend_enabled = self._model.has_nrend()

        # Transform the poses to the NRE frame
        # Supports_edit_actors() called above must handle this case.
        if self._world_to_nre is None:
            raise ValueError("World-to-NRE transformation is required but not initialized")

        track_poses_nre = frame_transform_poses(
            self._world_to_nre, actors_snapshot.actor_poses.to(self.device).reshape(-1, 7), is_tquat=True
        ).reshape(-1, 2, 7)  # Reshape needed in case track_poses is empty

        nrend_actor_poses: Optional[NRendWrapper.NRendPoses] = None
        if nrend_enabled:
            assert self._nrend.tracks is not None
            nrend_actor_poses = NRendWrapper.NRendPoses.from_actors_snapshot(
                actors_snapshot=actors_snapshot,
                track_poses_nre=track_poses_nre,
                active_track_ids=self._nrend.tracks.tracks_id,
                device=self.device,
            )

        assert isinstance(self._model, CompositeModel)  # Type must have a cuboid_tracks member used below.

        # This is still required even with NRend in order to support lidar rendering.
        cuboid_tracks = self._model.cuboid_tracks
        cuboid_tracks = CuboidTracks.Ops.subset_from_tracks_id(
            cuboid_tracks=cuboid_tracks, tracks_id=actors_snapshot.actor_ids
        )
        cuboid_tracks = CuboidTracks.Ops.freeze(
            cuboid_tracks=cuboid_tracks,
            min_timestamps_us=torch.full((cuboid_tracks.n_tracks,), frame_start_us, dtype=torch.int64).cuda(),
            max_timestamps_us=torch.full((cuboid_tracks.n_tracks,), frame_end_us, dtype=torch.int64).cuda(),
            tracks_poses_start=lietorch.SE3(data=track_poses_nre[:, 0]),
            tracks_poses_end=lietorch.SE3(data=track_poses_nre[:, 1]),
        )
        return nrend_actor_poses, cuboid_tracks

    @ScopedTimer()
    def _render_volume_from_ray_bundle(
        self,
        ray_bundle: RayBundle,
        actors_snapshot: Optional[ActorsSnapshot] = None,
        frame_start_us: int = 0,
        frame_end_us: int = 0,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> GaussiansRenderReturn:
        """Render a model given a ray bundle generated from a single camera view and return a GaussiansRenderReturn.

        Uses NRend automatically as the rendering engine if NRend is enabled in the model.
        Use RayBundle.build() to generate rays from a camera view.

        Args:
            ray_bundle: rays in NRE model space with metadata frame indices
            actors_snapshot: Optional ActorsSnapshot containing the actors ids and poses
            frame_start_us: start timestamp (in microseconds) of the frame captured
            frame_end_us: end timestamp (in microseconds) of the frame captured
            edited_cuboid_tracks: Optional edited tracks to use for stateless actor edits

        Returns:
            GaussiansRenderReturn containing rendered outputs
        """

        # This function is not supposed to be used directly.
        # TODO: Merge this function into render_camera_frame_from_ray_bundle() once all use-cases are eliminated.

        # NOTE: _edit_actors creates tensors through its use of frame_transform_poses.
        # If it is placed below in the autocast section, the input tensors will end up
        # being converted to the autocasted dtype (e.g. float16) and we have asserts in
        # interpolate_tracks_poses that expect tracks_interpolation_data.tracks_poses.dtype
        # (originating from the cuboid_tracks dtype in _edit_actors) to be float32. So
        # it is crucial to place it here before the autocast section.
        nrend_actor_poses: Optional[NRendWrapper.NRendPoses] = None
        if actors_snapshot is not None:
            nrend_actor_poses, actor_edited_cuboid_tracks = self._edit_actors(
                actors_snapshot=actors_snapshot, frame_start_us=frame_start_us, frame_end_us=frame_end_us
            )
            if edited_cuboid_tracks is None:
                edited_cuboid_tracks = actor_edited_cuboid_tracks

        with self.autocast, torch.no_grad(), ScopedTimer("RenderableModel._render_volume_from_ray_bundle"):
            # Copy all tensors with the ray bundle to GPU
            ray_bundle = cast(RayBundle, tree_map(ray_bundle, to_float_device(self.device)))

            # Call the model. Different models behave slightly differently.
            # TODO: replace model type branching with an abstract base class like CameraRenderable
            model_return: GaussiansCompositeReturn
            model = self._model

            if model.has_nrend():
                assert isinstance(model, GaussiansComposite)
                model_return = self._nrend(
                    model=model,
                    rendering_data=ray_bundle.rendering_data,
                    frame_meta=ray_bundle.frame_meta,
                    nrend_actor_poses=nrend_actor_poses,
                )
            elif isinstance(model, GaussiansComposite):
                # TODO: directly passing in RenderingBatch instead of having to create a DataAndRenderingBatch
                batch = DataAndRenderingBatch(
                    data=DataBatch(
                        camera=DataBatch.Camera(
                            meta=[ray_bundle.frame_meta],
                            labels=CameraFrameLabels(),  # empty labels
                        ),
                    ),
                    rendering=RenderingBatch(camera=ray_bundle.rendering_data),
                )
                model_return = model(
                    batch=batch, enable_calib=False, enable_pp=True, edited_cuboid_tracks=edited_cuboid_tracks
                )
            elif isinstance(model, BaseNRMPrimitive):
                model_return = model.forward(
                    rendering_cam_data=ray_bundle.rendering_data, frames_cam_meta=[ray_bundle.frame_meta]
                )
            else:
                raise TypeError(f"Unsupported model type {type(model)}.")

            return unpack_optional(model_return.rendered_cam)

    def render_camera_frame_from_ray_bundle(
        self,
        ray_bundle: RayBundle,
        fields: Optional[List[CameraFrameFields]] = None,
        actors_snapshot: Optional[ActorsSnapshot] = None,
        frame_start_us: int = 0,
        frame_end_us: int = 0,
    ) -> CameraFrame:
        """Render frame buffers of a model given a ray bundle generated from a camera view.

        Use RayBundle.build() to generate rays from a camera view.

        Args:
            ray_bundle: Ray bundle containing rays and metadata for rendering
            fields: Optional list of fields to render ("color_image", "distance_image", "opacity_image")
            actors_snapshot: Optional ActorsSnapshot containing the actors ids and poses
            frame_start_us: start timestamp (in microseconds) of the frame captured
            frame_end_us: end timestamp (in microseconds) of the frame captured

        Returns:
            CameraFrame with color_image as (h, w, 3) float32 tensor with values in [0, 1]
        """
        if fields is None:
            fields = ["color_image"]

        assert len(fields) > 0, "At least one output field must be provided"
        rendered = self._render_volume_from_ray_bundle(
            ray_bundle=ray_bundle,
            actors_snapshot=actors_snapshot,
            frame_start_us=frame_start_us,
            frame_end_us=frame_end_us,
        )

        raster_width = ray_bundle.rendering_data.w
        raster_height = ray_bundle.rendering_data.h

        rgb_image_f32: Optional[torch.Tensor] = None
        distance_image_f32: Optional[torch.Tensor] = None
        opacity_image_f32: Optional[torch.Tensor] = None

        if "color_image" in fields:
            rgb_image_f32 = unpack_optional(rendered.rgb)  # [(hw), c]
            rgb_image_f32 = rearrange(rgb_image_f32, "(h w) d -> h w d", h=raster_height, w=raster_width).float()

        if "distance_image" in fields:
            distance_image_f32 = rendered.distance  # [(hw),]
            distance_image_f32 = rearrange(distance_image_f32, "(h w) -> h w", h=raster_height, w=raster_width)

        if "opacity_image" in fields:
            opacity_image_f32 = rendered.opacity  # [(hw),]
            opacity_image_f32 = rearrange(opacity_image_f32, "(h w) -> h w", h=raster_height, w=raster_width)

        return CameraFrame(
            color_image=rgb_image_f32, distance_image=distance_image_f32, opacity_image=opacity_image_f32
        )

    def render_camera_frame(
        self,
        camera_intrinsics: ConcreteCameraModelParametersUnion,
        camera_to_world: PoseRange,
        resolution: Tuple[int, int],  # (width, height)
        unique_sensor_idx: Optional[int] = None,
        unique_frame_idx: Optional[int] = None,
        fields: Optional[List[CameraFrameFields]] = None,
        actors_snapshot: Optional[ActorsSnapshot] = None,
        frame_start_us: int = 0,
        frame_end_us: int = 0,
    ) -> CameraFrame:
        """Render a model at a specified resolution from a camera view defined by its camera intrinsics and pose,
        with the option to provide training camera and trajectory information, should the model require it.

        Allows to render the model in the following use-cases:
        (a) view that was part of the training data (training camera intrinsics and pose), or
        (b) view with intrinsics from a camera available during training but an arbitrary camera pose, or
        (c) view with intrinsics obtained by modifying those of a training camera, and an arbitrary pose, or
        (d) free viewpoint, a novel view with intrinsics and pose completely independent from training cameras or poses.

        Some stages in inference may not be independent of viewpoint, but specific to a training frame
        (e.g. post-processing learned per frame), or training camera (e.g. post-processing learned per camera).
        If such stages are enabled in the model, a frame index and/or camera and trajectory index must be provided.
        As a rule of thumb, whenever the provided intrinsics originate from a training camera on a training trajectory
        make sure that the camera and trajectory index are provided.

        Args:
          - camera_intrinsics: Intrinsic parameters of camera view to render from.
          - camera_to_world: Frame start and end poses (camera-to-world transforms) and corresponding timestamps for
            the camera view with rolling shutter (simply feed the same start/end poses/timestamps for global shutter).
          - resolution: The desired width and height (width, height) of the frame to be rendered.
          - unique_sensor_idx: Optional unique index of a camera used during training if the intrinsic parameters of
              the view to generate rays is from such a camera (obtain via SceneInfo.get_camera()).
          - unique_frame_idx: Optional unique index of a camera frame if the rays are to be generated from a view that
              matches or should be associated with a training view (obtain via SceneInfo.get_camera()).
          - fields: Optional list of fields to render ("color_image", "distance_image", "opacity_image").
          - actors_snapshot: Optional ActorsSnapshot containing the actors ids and poses.
          - frame_start_us: start timestamp (in microseconds) of the frame captured.
          - frame_end_us: end timestamp (in microseconds) of the frame captured.

        Returns:
          - CameraFrame.color_image: Predicted RGB image as f32 torch tensor of shape (height, width, channels).
        """

        # TODO: Disable inference stages that require any of the optional inputs above that are not provided.

        if self._world_to_nre is None:
            raise ValueError("World-to-NRE transformation is required but not initialized")

        camera_intrinsics = transform_intrinsics_to_resolution(camera_intrinsics, resolution)

        ray_bundle = RayBundle.build(
            camera_intrinsics,
            camera_to_world,
            self._world_to_nre,
            unique_sensor_idx,
            unique_frame_idx,
        )

        return self.render_camera_frame_from_ray_bundle(
            ray_bundle,
            fields,
            actors_snapshot=actors_snapshot,
            frame_start_us=frame_start_us,
            frame_end_us=frame_end_us,
        )

    def _lidar_rays_inference(
        self,
        ray_bundle: LidarRayBundle,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> GaussiansCompositeReturn:
        # TODO: This function is too low-level to be used from client code (e.g. gRPC service).
        #       Expose a high-level function instead that computes the tensors provided here from LiDAR intrinsics
        #       and pose information, similar to RenderableModel.render_camera_frame().
        if isinstance(self._model, GaussiansComposite):
            with self.autocast, torch.inference_mode():
                batch = DataAndRenderingBatch(
                    data=DataBatch(
                        lidar=DataBatch.Lidar(
                            meta=[ray_bundle.frame_meta],
                            labels=LidarFrameLabels(),  # empty labels
                        ),
                    ),
                    rendering=RenderingBatch(lidar=ray_bundle.rendering_data),
                )
                return self._model(
                    batch=batch,
                    enable_calib=False,
                    enable_pp=True,
                    edited_cuboid_tracks=edited_cuboid_tracks,
                )
        else:
            raise TypeError(f"Lidar rendering does not support models of type {type(self._model)}.")

    def render_lidar_frame_from_ray_bundle(
        self,
        ray_bundle: LidarRayBundle,
        raydrop_threshold: Optional[float] = None,
        opacity_threshold: Optional[float] = None,
        enable_distance_filter: Optional[bool] = None,
        distance_filter_threshold: Optional[float] = None,
        actors_snapshot: Optional[ActorsSnapshot] = None,
        frame_start_us: int = 0,
        frame_end_us: int = 0,
    ) -> LidarFrame:
        """Render a model given a ray bundle generated from a single lidar view and return a LidarFrame.

        Args:
            ray_bundle: The lidar ray bundle to render.
            raydrop_threshold: Override for raydrop probability threshold [0-1].
                Rays with raydrop > threshold are dropped. If None, uses artifact config default.
            opacity_threshold: Override for opacity threshold [0-1].
                Rays with opacity <= threshold are dropped. Set to 0.0 to disable (matching validation).
                If None, uses artifact config default.
            enable_distance_filter: Override to enable/disable distance-based edge filtering.
                If None, uses artifact config default.
            distance_filter_threshold: Override for distance filter threshold [0-1].
                Higher = fewer points filtered. If None, uses artifact config default.
            actors_snapshot: Optional ActorsSnapshot containing the actors ids and poses
            frame_start_us: start timestamp (in microseconds) of the frame captured
            frame_end_us: end timestamp (in microseconds) of the frame captured

        Returns:
            LidarFrame containing the rendered point cloud.
        """
        # Use provided overrides or fall back to artifact config defaults
        effective_raydrop_threshold = (
            raydrop_threshold
            if raydrop_threshold is not None
            else self._lidar_rendering_settings.lidar_raydrop_threshold
        )
        effective_opacity_threshold = (
            opacity_threshold
            if opacity_threshold is not None
            else self._lidar_rendering_settings.lidar_opacity_threshold
        )
        effective_enable_distance_filter = (
            enable_distance_filter
            if enable_distance_filter is not None
            else self._lidar_rendering_settings.enable_lidar_post_filter
        )
        effective_distance_filter_threshold = (
            distance_filter_threshold
            if distance_filter_threshold is not None
            else self._lidar_rendering_settings.lidar_post_filter_threshold
        )

        edited_cuboid_tracks: Optional[CuboidTracks] = None
        if actors_snapshot is not None:
            _, edited_cuboid_tracks = self._edit_actors(
                actors_snapshot=actors_snapshot, frame_start_us=frame_start_us, frame_end_us=frame_end_us
            )

        # Copy all tensors with the ray bundle to GPU
        ray_bundle = cast(LidarRayBundle, tree_map(ray_bundle, to_float_device(self.device)))
        model_return = self._lidar_rays_inference(ray_bundle, edited_cuboid_tracks=edited_cuboid_tracks)

        rendered = unpack_optional(model_return.rendered_lidar)
        dist = unpack_optional(rendered.distance)
        rays = ray_bundle.rendering_data.rays.reshape(-1, 6)
        xyz_end = torch.addcmul(rays[:, 0:3], rays[:, 3:6], dist.unsqueeze(1))

        if rendered.extra_ray_signals is not None and rendered.extra_ray_signals.intensity is not None:
            intensity = rendered.extra_ray_signals.intensity.squeeze()
        else:
            intensity = None

        # filter by opacity mask
        opacity = unpack_optional(rendered.opacity)
        keep_mask = opacity > effective_opacity_threshold

        if rendered.extra_ray_signals is not None and rendered.extra_ray_signals.raydrop is not None:
            # filter by predicted raydrop mask
            raydrop = rendered.extra_ray_signals.raydrop.flatten()
            did_return_pred = raydrop < effective_raydrop_threshold
            keep_mask &= did_return_pred
        else:
            did_return_pred = keep_mask.clone()

        # distance-based filter
        if effective_enable_distance_filter:
            height = ray_bundle.rendering_data.rays.shape[1]
            width = ray_bundle.rendering_data.rays.shape[2]
            model_elements = ray_bundle.model_elements.to(dtype=torch.int32)
            # remove points if filter_mask equal to True
            filter_mask = distance_based_filter(
                rendered, model_elements, did_return_pred, effective_distance_filter_threshold, height, width
            )
            keep_mask &= ~filter_mask

        xyz_end = xyz_end[keep_mask]
        if intensity is not None:
            intensity = intensity[keep_mask]
            intensity = intensity.contiguous()

        pc_nre = xyz_end.contiguous().unsqueeze(0)  # (1, n, 3)

        # Transform the points from NRE space to Lidar space (the end timestamp)
        T_nre_sensor_end = ray_bundle.T_nre_sensor_end.unsqueeze(0)  # (1, 4, 4)
        pc_sensor = transform_point_cloud(pc_nre, T_nre_sensor_end).squeeze(0)  # (n, 3)

        return LidarFrame(point_positions=pc_sensor, point_intensities=intensity)
