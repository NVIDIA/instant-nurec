# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import socket
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Literal, Optional, cast, get_args

import numpy as np
import torch
import trimesh
import viser
import viser.transforms as vtf

from einops import rearrange
from PIL import Image as PILImage
from python.runfiles import runfiles

import ncore.impl.common.transformations as ncore_transformations

from nre.config.viewer import ViewerConfig
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.render import RenderableModel
from nre.utils.geometry import pose_offsets_to_se3, se3_matrix_to_tquat
from nre.utils.misc import tree_map, unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.types import GaussiansRenderReturn
from nre.utils.visualize import scalar2img
from nre.viewer.dataset_interface import CameraTrajectoryData, ViewerDatasetInterface
from nre.viewer.lock import LockLike, MockLock
from nre.viewer.viewpoint import ViewerCameraChoice, Viewpoint, pinhole_to_fov, to_simple_pinhole


logger = logging.getLogger(__name__)
RUNFILES = runfiles.Create()


def pad_to_aspect_ratio(input_image: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """
    Pads `input_image` with black stripes of equal width (up to 1px difference) to match
    the target `aspect_ratio` defined as width / height.

    Args:
        - input_image: (h, w, 3) image to be padded
        - aspect_ratio: aspect ratio (width / height) to be matched by padding with 0s

    Returns:
        A padded image of shape (h', w', 3)
    """
    h, w, _ = input_image.shape
    target_w = int(h * aspect_ratio)
    target_h = int(w / aspect_ratio)

    if target_w > w:
        pad_height = target_w - w
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = 0
        pad_right = 0
    else:
        pad_width = target_h - h
        pad_top = 0
        pad_bottom = 0
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

    # FIXME: in some rare cases this results in negative padding values. For now we clip it
    # potentially (briefly) breaking overlay geometry and issue a warning (otherwise the viewer will crash)
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        logger.warning(
            f"Viewer: padding {h=}, {w=} to {aspect_ratio=} resulted in negative "
            f"{pad_left=}, {pad_right=}, {pad_top=}, {pad_bottom=}. Please report "
            f"to mtyszkiewicz@nvidia.com. For now clipping."
        )
        pad_left = max(0, pad_left)
        pad_right = max(0, pad_right)
        pad_top = max(0, pad_top)
        pad_bottom = max(0, pad_bottom)

    return np.pad(
        input_image, ((pad_left, pad_right), (pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0
    )


def pil_image_to_bytes(image: PILImage.Image, format: str = "png") -> bytes:
    """
    Converts a PIL Image to bytes for sending over network.

    Args:
        - image: image to convert
        - format: compression format specifier

    Returns:
        Raw bytes encoding the image in the given format.
    """
    bytes_io = BytesIO()
    image.save(bytes_io, format=format)
    return bytes_io.getvalue()


RenderState = Literal[
    "empty",  # scene not yet loaded
    "low_move",  # low resolution after movement
    "low_static",  # low resolution after no movement
    "high",  # high resolution render
]

RenderActionLabel = Literal[
    "comparison",  # prepare a GT/render comparison at the closest frame
    "move",  # camera has smoothly moved
    "static",  # camera remains fixed
    "unload",  # unload the scene (set to None)
    "reload",  # reload the scene
    "render_video",  # render out a video
]


@dataclass
class RenderAction:
    label: RenderActionLabel


@dataclass
class RasterArtifacts:
    resolution: Literal["high", "low"]

    rgb: torch.Tensor  # (h, w, 3)
    distance: torch.Tensor  # (h, w, 1), world space scale

    def get_rgb_raster(self) -> np.ndarray:
        return (self.rgb * 255).to(torch.uint8).numpy()

    def get_distance_raster(self) -> np.ndarray:
        distance = self.distance.squeeze(-1).numpy()
        return scalar2img(distance, cmap="viridis")

    def keys(self) -> list[str]:
        keys = ["distance", "rgb"]

        return keys

    def get_modality(self, id: str) -> np.ndarray:
        match id:
            case "rgb":
                return self.get_rgb_raster()
            case "distance":
                return self.get_distance_raster()
            case _:
                raise KeyError(f"Key {id} not found in RasterArtifacts")


@dataclass
class BBoxHandle:
    box: viser.MeshHandle
    label: viser.LabelHandle
    color: np.ndarray
    node_name: str


class RenderStateMachine(Thread):
    """
    A thread responsible for handling single user's viewer connection.

    It implements a state machine for resolution-progressive rendering (to provide
    interactive FPS while browsing and high resolution when static).

    At a high level, the state machine keeps track of the current render state
    (low resolution render after move, low resolution after no move, high resolution)
    and is informed by RenderStateMachine.action(RenderAction) to possibly re-render
    the image.
    """

    state: RenderState
    # Allow current trajectory and viewpoint to be None for the empty state
    current_trajectory: CameraTrajectoryData | None
    viewpoint: Viewpoint | None
    # Scene (dataset_interface and model wrapper) memory address tuple used to identify
    # whether the viewer has changed its scene. (so that we can trigger proper unload/reload)
    current_scene_id: tuple[int, int]

    DEFAULT_LOOK_AT_DISTANCE: float = 1.0

    def __init__(
        self,
        viewer: AbstractViewer,
        client: viser.ClientHandle,
    ):
        super().__init__(name=f"RenderStateMachine({client.client_id=}")
        self.viewer = viewer
        self.client = client

        # The values are to be populated later in case right now the scene is not yet ready.
        self.current_trajectory = None
        self.viewpoint = None

        # set up a state machine to manage progressive low/high resolution rendering
        # while moving camera around
        self.transitions: dict[RenderState, dict[RenderActionLabel, RenderState]] = {
            s: {} for s in get_args(RenderState)
        }
        # by default, everything is a self-transition
        for a in get_args(RenderActionLabel):
            for s in get_args(RenderState):
                self.transitions[s][a] = s
        # then define the actions between states
        self.transitions["low_move"]["static"] = "low_static"
        self.transitions["low_static"]["static"] = "high"
        self.transitions["low_static"]["move"] = "low_move"
        self.transitions["low_static"]["comparison"] = "high"
        self.transitions["low_move"]["comparison"] = "high"
        self.transitions["high"]["move"] = "low_move"
        # unload and reload could happen at every state
        for s in get_args(RenderState):
            self.transitions[s]["unload"] = "empty"
            self.transitions[s]["reload"] = "low_static"

        self.action_lock = Lock()
        self.next_action: RenderAction | None = None
        self.render_trigger = Event()

        with self.viewer.scene_lock:
            self.current_scene_id = self.viewer.get_current_scene_id()
        if self.current_scene_id == (-1, -1):
            # Scene not ready -- go to empty state the hold
            self.state = "empty"
        else:
            # Scene is ready now, directly reload it.
            self.state = "low_static"
            self.action(RenderAction("reload"))

        self.current_raster_artifacts: RasterArtifacts | None = None

        @self.client.camera.on_update
        def _camera_on_update(_: viser.CameraHandle) -> None:
            if not self.viewer._ready:
                return
            self.action(RenderAction("move"))

        # Last pose of the sensor when _adjust_viewpoint is called
        # Used to compute the relative transform to the sensor when following the camera
        self.last_sensor_se3: vtf.SE3 | None = None
        self.force_relative_transform: vtf.SE3 | None = None

        self.rig_pose_interpolator: ncore_transformations.PoseInterpolator | None = None

        # GUI handles that are created dynamically
        self.time_slider: viser.GuiSliderHandle[int] | None = None
        self.time_absolute_label: viser.GuiInputHandle[str] | None = None
        self.follow_camera: viser.GuiCheckboxHandle | None = None
        self.ego_handle: viser.GlbHandle | None = None

        self.daemon = True
        self.running = True

    def _reload_scene(self) -> None:
        # Called when model or dataset changes are detected. Rebuild GUI.
        dataset_interface = self.viewer.get_dataset_interface()
        self.current_trajectory = dataset_interface.get_initial_trajectory_data()
        self.viewpoint = Viewpoint.create(
            self.current_trajectory,
            self.current_trajectory.time_range_us.start + self.current_trajectory.average_exposure_time_us,
            self.DEFAULT_LOOK_AT_DISTANCE,
        )
        self.force_relative_transform = None

        # Load rig trajectory for visualizing ego pose.
        if (rig_trajectories := dataset_interface.get_rig_trajectories()) is not None:
            rig_trajectory = rig_trajectories.rig_trajectories[0]
            self.rig_pose_interpolator = ncore_transformations.PoseInterpolator(
                poses=rig_trajectory.T_rig_worlds.cpu().numpy(),
                timestamps=rig_trajectory.T_rig_world_timestamps_us.cpu().numpy(),
            )
        else:
            self.rig_pose_interpolator = None

        self.ego_handle = None
        self.client.scene.reset()
        self.client.gui.reset()
        self._build_gui()
        self._set_client_camera()
        self.action(RenderAction("move"))

    def _unload_scene(self) -> None:
        self.client.scene.reset()
        self.client.gui.reset()
        self.client.gui.add_markdown("⏳ Waiting for the scene to become available")

    def _add_gaussian_centers_point_cloud(self) -> viser.SceneNodeHandle | None:
        """
        Extract Gaussian centers from the model and add them as a point cloud visualization.
        Returns the point cloud handle or None if Gaussians are not available.
        """
        model = self.viewer.model
        if model is None:
            return None

        # Use the clean getter method
        gaussian_positions_tensor = model.get_all_gaussian_positions()

        if gaussian_positions_tensor is None or len(gaussian_positions_tensor) == 0:
            logger.info("No Gaussian positions found in the model")
            return None

        # Convert to numpy
        gaussian_positions = gaussian_positions_tensor.detach().cpu().numpy()

        # Transform positions from NRE frame to world frame
        world_to_nre = self.viewer.get_dataset_interface().world_to_nre
        nre_to_world = world_to_nre.inverse()
        gaussian_positions_world = nre_to_world.transform_points(gaussian_positions)

        # Create point cloud visualization with a distinctive color (cyan/blue)
        point_cloud_handle = self.client.scene.add_point_cloud(
            "gaussian_centers",
            gaussian_positions_world,
            colors=np.array([0.0, 1.0, 1.0]),  # cyan color to distinguish from lidar
            point_size=0.01,
            point_shape="circle",
        )

        logger.info(f"Added {len(gaussian_positions)} Gaussian centers to the viewer")
        return point_cloud_handle

    def _adjust_viewpoint(self) -> None:
        # Don't adjust if trajectory is not yet populated
        if self.current_trajectory is None:
            return

        # Compute relative transform to the sensor when following the camera
        relative_transform = self.force_relative_transform or vtf.SE3.identity()
        look_at_distance = self.DEFAULT_LOOK_AT_DISTANCE

        if (
            self.follow_camera is not None
            and self.follow_camera.value
            and self.last_sensor_se3 is not None
            and self.force_relative_transform is None
        ):
            client_pose = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(wxyz=self.client.camera.wxyz),
                self.client.camera.position,
            )
            relative_transform = self.last_sensor_se3.inverse() @ client_pose
            look_at_distance = np.linalg.norm(self.client.camera.look_at - self.client.camera.position).item()

        # The flag should only affect once this function as above
        self.force_relative_transform = None

        self.viewpoint = Viewpoint.create(
            self.current_trajectory,
            self._get_time_us(),
            look_at_distance,
        )
        self.last_sensor_se3 = self.viewpoint.se3_world
        self.viewpoint.se3_world = self.viewpoint.se3_world @ relative_transform

        # If we don't follow the camera then viewpoint should use current client pose
        if self.follow_camera is not None and not self.follow_camera.value:
            self.viewpoint = self.viewpoint.update_to_client_pose(self.client)

        with self.client.atomic():
            self._set_client_camera()
            self._update_ego()
        # TODO: _set_client_camera possibly fires client.camera.on_update so we don't need to trigger move?
        self.action(RenderAction("move"))

    def _set_time(self, timestamp_us: int) -> None:
        # on client side we use time relative to the beginning of the sequences because javascript frontend
        # breaks with full precision unix timestamps
        assert self.time_slider is not None
        timestamp_relative_us = self.viewer.get_dataset_interface().get_time_converter_us().to_local(timestamp_us)

        if self.time_slider.value == timestamp_relative_us:
            return  # no change, don't trigger adjustments

        self.time_slider.value = timestamp_relative_us
        self._on_time_updated()

    def _get_time_us(self) -> int:
        # on client side we use time relative to the beginning of the sequences because javascript frontend
        # breaks with full precision unix timestamps
        assert self.current_trajectory is not None
        assert self.time_slider is not None
        timestamp_relative_us = self.time_slider.value
        time_converter = self.current_trajectory.time_range_us
        timestamp_global_us = time_converter.to_global(timestamp_relative_us)
        return timestamp_global_us

    def _on_time_updated(self) -> None:
        self._adjust_viewpoint()

    def _recreate_time_slider(self) -> None:
        assert self.current_trajectory is not None

        if self.time_slider is not None:
            self.time_slider.remove()
            self.time_slider = None
        if self.time_absolute_label is not None:
            self.time_absolute_label.remove()
            self.time_absolute_label = None
        if self.follow_camera is not None:
            self.follow_camera.remove()
            self.follow_camera = None

        time_range = self.current_trajectory.time_range_us

        time_slider = self.client.gui.add_slider(
            "Time offset (us)",
            min=self.current_trajectory.average_exposure_time_us,
            max=time_range.length - 1,
            initial_value=self.current_trajectory.average_exposure_time_us,
            step=1,
            marks=(0, -1),
            hint="Pick pose at which the model is evaluated",
            order=0,  # put at the top
        )

        self.time_absolute_label = self.client.gui.add_text(
            "Absolute time (us)",
            initial_value=str(self.current_trajectory.time_range_us.to_global(time_slider.value)),
            disabled=True,
            order=0.1,
        )

        @time_slider.on_update
        def _time_slider_on_update(_) -> None:
            assert self.current_trajectory is not None
            assert self.time_absolute_label is not None
            absolute_time = self.current_trajectory.time_range_us.to_global(time_slider.value)
            self.time_absolute_label.value = str(absolute_time)
            self._on_time_updated()

        self.time_slider = time_slider

        self.follow_camera = self.client.gui.add_checkbox("Follow camera", initial_value=True, order=0.2)
        # Clear follow-camera relative transform
        self.force_relative_transform = vtf.SE3.identity()

        # At the moment when the follow camera is toggled on, we snap to the original camera view
        @self.follow_camera.on_update
        def _follow_camera_on_update(_) -> None:
            assert self.follow_camera is not None
            if self.follow_camera.value:
                self.force_relative_transform = vtf.SE3.identity()
                self._adjust_viewpoint()

        self._on_time_updated()

    def _update_ego(self) -> None:
        """This method updates the ego visualization based on the current timestamp"""
        if self.rig_pose_interpolator is None:
            return

        if self.ego_handle is None:
            assert RUNFILES is not None, "RUNFILES is not set. Please check bazel environment setup."
            ego_mesh = trimesh.load(RUNFILES.Rlocation("nv_assets/nv-car-no-lidar-diy.glb"))
            self.ego_handle = self.client.scene.add_mesh_trimesh(
                name="ego",
                mesh=ego_mesh,
                visible=False,  # Initially hidden
            )

        pose = self.rig_pose_interpolator.interpolate_to_timestamps(self._get_time_us())
        tquat_pose = se3_matrix_to_tquat(pose)
        self.ego_handle.wxyz = np.roll(tquat_pose[3:], 1)
        self.ego_handle.position = tquat_pose[:3].numpy()

    def _update_bboxes(self) -> None:
        """This method updates the bbox visualization based on the current timestamp"""
        timestamp_us = self._get_time_us()
        if (cuboid_tracks := self.layer_tracks) is not None:
            for track_idx, track_id in enumerate(cuboid_tracks.tracks_id):
                selected_track = CuboidTracks.Ops.subset_from_tracks_id(cuboid_tracks, [track_id])

                if (
                    timestamp_us >= selected_track.tracks_timestamps_us.min()
                    and timestamp_us <= selected_track.tracks_timestamps_us.max()
                ):
                    interpolator = ncore_transformations.PoseInterpolator(
                        poses=selected_track.tracks_poses.matrix().cpu(),
                        timestamps=selected_track.tracks_timestamps_us.cpu(),
                    )

                    pose = interpolator.interpolate_to_timestamps(timestamp_us)
                    tquat_pose = se3_matrix_to_tquat(pose)

                    object_dim = cuboid_tracks.cuboids_dims[track_idx].cpu().numpy()

                    if track_id in self.bbox_handles:
                        color = self.bbox_handles[track_id].color
                        node_name = self.bbox_handles[track_id].node_name
                    else:
                        color = np.random.rand(3)
                        trackflags_str = TrackFlags(int(cuboid_tracks.tracks_flags[track_idx].item())).name
                        node_name = f"{track_id}-{trackflags_str}-{cuboid_tracks.tracks_label_class[track_idx]}"

                    bbox_handle = self.client.scene.add_box(
                        name=node_name,
                        color=color,
                        dimensions=object_dim,
                        wxyz=(quat := np.roll(tquat_pose[3:], 1)),  # rolling as tquat is [XYZ, XYZW]
                        position=(pos := tquat_pose[:3].numpy()),
                        visible=self.bboxes_visible,
                    )

                    bbox_label_handle = self.client.scene.add_label(
                        f"{track_id}_label", node_name, wxyz=quat, position=pos, visible=self.bboxes_visible
                    )
                    self.bbox_handles[track_id] = BBoxHandle(
                        box=bbox_handle,
                        label=bbox_label_handle,
                        color=color,
                        node_name=node_name,
                    )
                else:
                    if track_id in self.bbox_handles:
                        bbox_to_remove = self.bbox_handles.pop(track_id)
                        bbox_to_remove.box.visible = False
                        bbox_to_remove.label.visible = False

    @property
    def current_viewer_camera(self) -> ViewerCameraChoice:
        return cast(ViewerCameraChoice, self.camera_model_dropdown.value)

    @property
    def current_point_cloud_color_type(self) -> Literal["camera-rgb", "semantics"] | None:
        if self.point_cloud_color_type_dropdown.value == "uniform (red)":
            return None
        else:
            return cast(Literal["camera-rgb", "semantics"], self.point_cloud_color_type_dropdown.value)

    def _build_gui(self) -> None:
        """
        Creates all GUI widgets and associated callbacks.
        """
        self.bbox_handles: dict[str, BBoxHandle] = {}
        self.layer_tracks: Optional[CuboidTracks] = None
        self.bboxes_visible: bool = True

        # for models that don't have cuboid tracks we use the datasource's sequence tracks
        # and the layer id corresponds to the sequence id
        self.layer_tracks = self.viewer.get_dataset_interface().get_cuboid_tracks()

        ## slider for picking trajectory timestamp
        self._recreate_time_slider()

        trajectory_ids = self.viewer.get_dataset_interface().get_trajectory_ids()
        str_to_trajectory_id = {id.camera_sequence_str(): id for id in trajectory_ids}
        trajectory_choice = self.client.gui.add_dropdown(
            "Choose camera trajectory",
            options=list(str_to_trajectory_id.keys()),
            initial_value=list(str_to_trajectory_id.keys())[0],
        )

        @trajectory_choice.on_update
        def _trajectory_choice_updated(_) -> None:
            trajectory_id = str_to_trajectory_id[trajectory_choice.value]
            self.current_trajectory = self.viewer.get_dataset_interface().get_trajectory_data(trajectory_id)

            # recreate the time slider to respect the new time range
            self._recreate_time_slider()
            self.camera_rotation_offset.value = (0.0, 0.0, 0.0)
            self.camera_translation_offset.value = (0.0, 0.0, 0.0)

        ## dropdown for picking display modality (rgb/distance/etc)
        self.modality_dropdown = self.client.gui.add_dropdown(
            "Modality",
            options=("rgb", "distance"),
            initial_value="rgb",
            hint="Choose the modality to visualize",
        )

        @self.modality_dropdown.on_update
        def _modality_dropdown_updated(_) -> None:
            self._send_render_to_client()

        ## button to clear the current scene cached in the viewer.
        # (indicating that the main process might be able to populate something else for it)
        if self.viewer.supports_clear_current_scene():
            self.clear_model_cache_button = self.client.gui.add_button("Next Scene")

            @self.clear_model_cache_button.on_click
            def _clear_model_cache_button_clicked(_) -> None:
                self.viewer.clear_current_scene()

        ## dropdown for picking original/pinhole camera model
        self.camera_model_dropdown: viser.GuiDropdownHandle = self.client.gui.add_dropdown(
            "Camera model",
            options=("original", "pinhole"),
            initial_value="pinhole",  # self.current_viewer_camera,
            hint=(
                "Original: source camera parameters. Pinhole: closest distortion-free pinhole, "
                "enables 3D overlays (since the GUI doesn't support FTheta)."
            ),
        )

        ## an entire panel of controls specific to pinhole camera model
        pinhole_3d_items: list[viser.SceneNodeHandle] = []
        pinhole_gui_items: list[viser._gui_handles._GuiInputHandle[Any]] = []
        with self.client.gui.add_folder("Pinhole controls"):
            ## splines marking camera trajectories and toggle button
            trajectory_splines: list[viser.SceneNodeHandle] = []
            camera_frusta: list[viser.CameraFrustumHandle] = []
            num_poses: int = 0
            for trajectory_id in self.viewer.get_dataset_interface().get_trajectory_ids():
                trajectory_data = self.viewer.get_dataset_interface().get_trajectory_data(trajectory_id)

                trajectory_poses = trajectory_data.get_poses_world(
                    torch.linspace(  # a thousand equispaced trajectory knots
                        trajectory_data.time_range_us.start, trajectory_data.time_range_us.end, 1_000, dtype=torch.int64
                    )
                )
                trajectory_positions = trajectory_poses[..., :3, -1].numpy()

                if trajectory_positions.ndim == 1 or trajectory_positions.shape[0] == 1:
                    continue  # skip trajectories with only one pose (the case of PLY viewer)

                num_poses += len(trajectory_positions)
                spline = self.client.scene.add_spline_catmull_rom(
                    trajectory_id.camera_sequence_str(),
                    trajectory_positions,
                    line_width=3.0,
                    tension=0.0,
                    color=(0.0, 1.0, 0.0),  # green
                )

                trajectory_splines.append(spline)

                # Create camera frusta at frame timestamps
                frame_timestamps = trajectory_data.get_frame_timestamps()
                frustum_subsample_step = max(1, len(frame_timestamps) // self.viewer.config.max_n_visible_frusta)
                for frame_idx, end_ts in enumerate(frame_timestamps[::frustum_subsample_step, 1].tolist()):
                    # Get pose at the frame timestamp
                    frame_pose = trajectory_data.get_poses_world(torch.tensor([end_ts]))
                    frame_pose_tquat = se3_matrix_to_tquat(frame_pose)

                    # Get camera intrinsics
                    camera_model = trajectory_data.camera_model
                    w, h = camera_model.resolution.tolist()

                    # Try to get the image for this frame
                    if self.viewer.get_dataset_interface().supports_get_closest_frame_image():
                        _, frame_image = trajectory_data.get_closest_frame_image(end_ts)
                        image = frame_image.cpu().numpy()
                    else:
                        # If image is not available, create a placeholder
                        image = np.full((h, w, 3), fill_value=128, dtype=np.uint8)

                    position = frame_pose_tquat[:3].numpy()
                    wxyz = np.roll(frame_pose_tquat[3:], 1)

                    # Compute fov for the nearest pinhole camera model
                    fov = pinhole_to_fov(to_simple_pinhole(camera_model), viewer_aspect_ratio=1.0)

                    RED = (255, 0, 0)
                    BLACK = (0, 0, 0)
                    # hacky way to color the frusta based on the camera type in NRM
                    color = RED if ("supervision" in trajectory_id.unique_camera_id) else BLACK

                    # Create camera frustum
                    frustum = self.client.scene.add_camera_frustum(
                        f"{trajectory_id.camera_sequence_str()}_frustum_{frame_idx}",
                        fov=fov,
                        aspect=w / h,
                        scale=0.15,
                        image=image,
                        wxyz=wxyz,
                        position=position,
                        visible=True,
                        color=color,
                    )
                    camera_frusta.append(frustum)

            if trajectory_splines:
                spline_toggle_button = self.client.gui.add_button(
                    "Toggle trajectory overlay",
                )

                @spline_toggle_button.on_click
                def _spline_toggle_button_clicked(_) -> None:
                    for spline in trajectory_splines:
                        spline.visible = not spline.visible

                pinhole_3d_items.extend(trajectory_splines)
                pinhole_gui_items.append(spline_toggle_button)

            if camera_frusta:
                frusta_toggle_button = self.client.gui.add_button(
                    "Toggle camera frusta overlay",
                )

                @frusta_toggle_button.on_click
                def _frusta_toggle_button_clicked(_) -> None:
                    for frustum in camera_frusta:
                        frustum.visible = not frustum.visible

                pinhole_3d_items.extend(camera_frusta)
                pinhole_gui_items.append(frusta_toggle_button)

            step_frame = max(num_poses // self.viewer.config.n_lidar_frames_displayed, 1)
            self.point_cloud_color_type: str | None = None
            self.point_cloud = self.viewer.get_dataset_interface().get_point_cloud(
                step_frame=step_frame, color_type=self.point_cloud_color_type
            )

            ## an overlaid Gaussian centers point cloud and toggle button
            self.gaussian_centers_3d_handle = self._add_gaussian_centers_point_cloud()
            if self.gaussian_centers_3d_handle is not None:
                gaussian_centers_toggle_button = self.client.gui.add_button(
                    "Toggle Gaussian centers overlay",
                )

                @gaussian_centers_toggle_button.on_click
                def _gaussian_centers_toggle_button_clicked(_) -> None:
                    if self.gaussian_centers_3d_handle is not None:
                        self.gaussian_centers_3d_handle.visible = not self.gaussian_centers_3d_handle.visible

                pinhole_3d_items.append(self.gaussian_centers_3d_handle)
                pinhole_gui_items.append(gaussian_centers_toggle_button)

            ## an overlaid lidar point cloud and toggle button
            if self.point_cloud is not None:
                self.point_cloud_3d_handle = self.client.scene.add_point_cloud(
                    "lidar_point_cloud",
                    self.viewer.get_dataset_interface()
                    .world_to_nre.inverse()
                    .transform_points(self.point_cloud.xyz_end.cpu().numpy()),
                    colors=(
                        self.point_cloud.color.cpu().numpy()
                        if self.point_cloud.color is not None
                        else np.array([1.0, 0.0, 0.0])
                    ),
                    point_size=0.02,
                    point_shape="circle",
                )

                point_cloud_toggle_button = self.client.gui.add_button(
                    "Toggle point cloud overlay",
                )

                @point_cloud_toggle_button.on_click
                def _point_cloud_toggle_button_clicked(_) -> None:
                    self.point_cloud_3d_handle.visible = not self.point_cloud_3d_handle.visible

                # dropdown menu to choose color-type for point cloud
                self.point_cloud_color_type_dropdown: viser.GuiDropdownHandle = self.client.gui.add_dropdown(
                    "Point cloud color type",
                    options=("uniform (red)", "camera-rgb", "semantics"),
                    initial_value="uniform (red)",
                )

                @self.point_cloud_color_type_dropdown.on_update
                def _point_cloud_color_type_dropdown_updated(_) -> None:
                    self.point_cloud_color_type = self.current_point_cloud_color_type
                    self.point_cloud = self.viewer.get_dataset_interface().get_point_cloud(
                        step_frame=step_frame, color_type=self.point_cloud_color_type
                    )

                    if self.point_cloud_3d_handle in pinhole_3d_items:
                        pinhole_3d_items.remove(self.point_cloud_3d_handle)
                    if self.point_cloud is not None:
                        self.point_cloud_3d_handle = self.client.scene.add_point_cloud(
                            "lidar_point_cloud",
                            self.viewer.get_dataset_interface()
                            .world_to_nre.inverse()
                            .transform_points(self.point_cloud.xyz_end.cpu().numpy()),
                            colors=(
                                self.point_cloud.color.cpu().numpy()
                                if self.point_cloud.color is not None
                                else np.array([1.0, 0.0, 0.0])
                            ),
                            point_size=0.02,
                            point_shape="circle",
                        )
                        pinhole_3d_items.append(self.point_cloud_3d_handle)

                pinhole_3d_items.append(self.point_cloud_3d_handle)
                pinhole_gui_items.append(point_cloud_toggle_button)
                pinhole_gui_items.append(self.point_cloud_color_type_dropdown)

            ## setup bbox visualization
            self._update_bboxes()
            bbox_toggle_button = self.client.gui.add_button(
                "Toggle bbox overlay",
            )

            @bbox_toggle_button.on_click
            def _bbox_toggle_button_clicked(_) -> None:
                self.bboxes_visible = not self.bboxes_visible
                for bbox in self.bbox_handles.values():
                    bbox.box.visible = self.bboxes_visible
                    bbox.label.visible = self.bboxes_visible

            pinhole_gui_items.append(bbox_toggle_button)

            ## setup ego visualization
            self._update_ego()

            if self.ego_handle is not None:
                ego_toggle_button = self.client.gui.add_button(
                    "Toggle ego-car overlay",
                )

                @ego_toggle_button.on_click
                def _ego_toggle_button_clicked(_) -> None:
                    assert self.ego_handle is not None
                    self.ego_handle.visible = not self.ego_handle.visible

                pinhole_3d_items.append(self.ego_handle)
                pinhole_gui_items.append(ego_toggle_button)

        ## an entire panel of functions specific to FTheta camera model (original)
        ftheta_gui_items: list[viser._gui_handles._GuiInputHandle[Any]] = []

        if self.viewer.get_dataset_interface().supports_get_closest_frame_image():
            with self.client.gui.add_folder("FTheta controls"):
                ## button to snap to a close frame with available GT and download comparison bitmaps
                render_comparison_button = self.client.gui.add_button(
                    "Render GT comparison",
                    disabled=self.current_viewer_camera == "pinhole",  # disable if we're starting with pinhole
                )

                @render_comparison_button.on_click
                def _render_comparison_button_clicked(_) -> None:
                    self.action(RenderAction("comparison"))

                ftheta_gui_items.append(render_comparison_button)

        ## make the camera model dropdown disable/enable the respective pane
        @self.camera_model_dropdown.on_update
        def _camera_model_dropdown_updated(_) -> None:
            match self.current_viewer_camera:
                case "original":
                    for item_3d in pinhole_3d_items:
                        item_3d.visible = False

                    for gui_item in pinhole_gui_items:
                        gui_item.disabled = True

                    for gui_item in ftheta_gui_items:
                        gui_item.disabled = False

                    self.bboxes_visible = False

                case "pinhole":
                    for gui_item in pinhole_gui_items:
                        gui_item.disabled = False

                    for gui_item in ftheta_gui_items:
                        gui_item.disabled = True

            self.action(RenderAction("move"))

        ## Add control groups for video rendering
        with self.client.gui.add_folder("Demo Options"):
            self.camera_rotation_offset = self.client.gui.add_vector3(
                "Camera Rotation Offset",
                initial_value=(0.0, 0.0, 0.0),
                step=5.0,
            )
            self.camera_translation_offset = self.client.gui.add_vector3(
                "Camera Translation Offset",
                initial_value=(0.0, 0.0, 0.0),
                step=0.1,
            )
            self.render_video_out_path = self.client.gui.add_text(
                "Video Output",
                initial_value="/tmp/viewer_demo_video/",
            )
            render_demo_video_button = self.client.gui.add_button(
                "Render Video (Blocking)",
            )

            @render_demo_video_button.on_click
            def _render_demo_video_button_clicked(_) -> None:
                self.action(RenderAction("render_video"))

            def _set_relative_transform() -> None:
                # Not supporting rig offset here since we need to apply the adjoint of rig-sensor
                # which we don't have now.
                self.force_relative_transform = vtf.SE3.from_matrix(
                    pose_offsets_to_se3(
                        self.camera_translation_offset.value, self.camera_rotation_offset.value, rotation_first=True
                    )
                )
                if self.follow_camera is not None and not self.follow_camera.value:
                    logger.warning("Offsets won't take effect until follow camera is checked.")
                self._adjust_viewpoint()

            @self.camera_rotation_offset.on_update
            def _camera_rotation_offset_updated(_) -> None:
                _set_relative_transform()

            @self.camera_translation_offset.on_update
            def _camera_translation_offset_updated(_) -> None:
                _set_relative_transform()

    def _set_client_camera(self) -> None:
        """
        Set the client camera to match our current viewpoint. Used when the change comes from
        gui (camera dropdown, etc) rather than client viewport.
        """

        # Viser uses orbit controls be default (adjusting wxyz also changes position) so we need to
        # decompose Qt_world into `position`, `look_at` and `camera_up_direction` as these are the actual
        # viser primitives
        view_system = unpack_optional(self.viewpoint).lookat_world

        with self.client.atomic():
            # setting position modifies look_at so it needs to be done first
            self.client.camera.position = view_system.position

            self.client.camera.look_at = view_system.look_at
            self.client.camera.up_direction = view_system.up

            self._maybe_set_camera_fov()

    def _maybe_set_camera_fov(self) -> None:
        """
        If the current camera is a simple pinhole we adjust client camera's FOV to properly
        overlay point clouds and other 3d primitives on top of the renders.
        """
        client_aspect = self.client.camera.aspect
        new_fov = unpack_optional(self.viewpoint).viser_matching_fov(client_aspect)

        if abs(self.client.camera.fov - new_fov) < 1e-4:
            # setting the fov triggers client.camera.on_update which in turn
            # triggers this piece of code, creating an infinite loop. Therefore we only
            # make the update if there is a meaningful change
            return

        self.client.camera.fov = new_fov

    def _render_video(self) -> None:
        """
        Render a video of the current scene by stepping through the time slider.
        """
        if self.time_slider is None:
            return

        prev_state = self.state
        self.state = "high"

        image_width = self.client.camera.image_width
        image_height = self.client.camera.image_height

        save_folder = Path(self.render_video_out_path.value)
        save_folder.mkdir(parents=True, exist_ok=True)
        render_fps = 30

        for frame_idx, relative_timestamp_us in enumerate(
            np.arange(self.time_slider.min, self.time_slider.max, 1000000 // render_fps)
        ):
            self.time_slider.value = relative_timestamp_us
            # Wait for the other thread to update camera
            # There's currently no way of knowing whether the update has been applied.
            # so we sleep here
            time.sleep(0.05)

            self.current_raster_artifacts = self._render()
            self._send_render_to_client()
            self._update_bboxes()

            # Capture the screenshot (slow)
            snapshot_img = self.client.get_render(height=image_height, width=image_width, transport_format="png")
            img = PILImage.fromarray(snapshot_img)
            img.save(f"{save_folder}/frame_{frame_idx:06d}.png")

        self.state = prev_state

    def run(self):
        """
        Main loop for the render thread, stepping through the states according to actions
        """
        while self.running:
            if not self.viewer._ready:
                time.sleep(0.1)
                continue
            if not self.render_trigger.wait(0.2):
                # if we haven't received a trigger in a while, send a static action
                self.action(RenderAction("static"))

            # During state machine execution where we need to access/change the viewer's scene
            # Make sure their references are properly locked.
            with self.viewer.scene_lock:
                new_scene_id = self.viewer.get_current_scene_id()
                if self.current_scene_id != new_scene_id:
                    self.current_scene_id = new_scene_id
                    self.action(RenderAction("unload" if self.current_scene_id == (-1, -1) else "reload"))

                with self.action_lock:
                    action = self.next_action
                    self.render_trigger.clear()
                    if action is None:
                        continue
                    self.next_action = None
                if self.state == "high" and action.label == "static":
                    # if we are in high res and we get a static action, we don't need to do anything
                    continue

                if action.label == "reload":
                    self._reload_scene()

                if action.label == "unload":
                    self._unload_scene()

                if action.label == "comparison":
                    closest_frame_timestamp_us, gt_rgb = self.current_trajectory.get_closest_frame_image(
                        self._get_time_us()
                    )
                    self._set_time(closest_frame_timestamp_us)
                    self._adjust_viewpoint()

                self.state = self.transitions[self.state][action.label]
                if self.state == "empty":
                    continue

                self.current_raster_artifacts = self._render()
                self._send_render_to_client()
                self._update_bboxes()

                if action.label == "comparison":
                    self._send_comparison_to_client(closest_frame_timestamp_us, gt_rgb)

                if action.label == "render_video":
                    self._render_video()

    def action(self, action: RenderAction) -> None:
        """
        Used from the main thread to enqueue a renderer action and
        from RenderStateMachine.run for periodic updates.
        """
        # lock to avoid clashes between main thread callbacks and the RenderStateMachine thread
        with self.action_lock:
            if self.next_action is None or action.label in ("unload", "reload"):
                self.next_action = action
            elif self.next_action.label == "comparison":
                # never overwrite comparisons
                pass
            elif action.label == "static" and self.next_action == "move":
                # don't overwrite a move action with a static: static is always self-fired
                return
            else:
                #  minimal use case, just set the next action
                self.next_action = action

            self.render_trigger.set()

    def _render(self, high_res: bool = False, _is_a_retry: bool = False) -> RasterArtifacts:
        """
        Calculate the viewing rays and render the NeRF.
        """
        # update camera position
        self.viewpoint = unpack_optional(self.viewpoint).update_to_client_pose(self.client)

        if self.state == "high" or high_res:
            resolution_step = self.viewer.config.ray_subsampling_step.high_resolution
            is_high_resolution = True
        elif self.state in ("low_static", "low_move"):
            resolution_step = self.viewer.config.ray_subsampling_step.low_resolution
            is_high_resolution = False
        else:
            raise ValueError(f"Invalid state {self.state}")

        ray_bundle = self.viewpoint.get_ray_bundle(
            resolution_step,
            self.current_viewer_camera,
            self.viewer.get_dataset_interface().world_to_nre,
            self.current_trajectory.camera_unique_idx if self.current_trajectory is not None else None,
        )

        should_retry = False

        rendered: Optional[GaussiansRenderReturn] = None
        try:
            with self.viewer.system_lock:
                # TODO: Refactor to call the public RenderableModel.render_camera_frame_from_ray_bundle() instead.
                rendered = unpack_optional(self.viewer.model)._render_volume_from_ray_bundle(ray_bundle)
        except RuntimeError as error:
            error_message = str(error)
            if ("check failed params.device().is_cuda()" in error_message) or (  # TCNN error
                "Expected all tensors to be on the same device" in error_message  # regular torch error
            ):
                # We're likely at the end of the fit loop and the model is being torn down from
                # CUDA. Wait two seconds and retry (ObserverCallback.on_fit_end should put us back).
                # Check if we're already retrying: if yes, error out, otherwise move outside this
                # context manager (to drop the renderer lock) and retry.
                #
                # Note: the teardown is happening from a concurrent thread so non-synchronized
                # checks will not help. I tried locking self.viewer.system_lock between
                # ObserverCallback.teardown and ObserverCallback.on_fit_end to prevent the teardown
                # while render calls are ongoing but it does not seem to capture the right execution
                # segment
                if _is_a_retry:
                    raise RuntimeError(f"Failed due despite retry") from error
                else:
                    should_retry = True
            else:
                raise

        if should_retry:
            logger.warning("Model not on CUDA, retrying")
            time.sleep(2.0)
            return self._render(high_res=high_res, _is_a_retry=True)

        if rendered is None:
            raise RuntimeError("Model returned None")  # for mypy, it should have retried or raised anyway

        with ScopedTimer("viewpoint::postprocess"):
            raster_w, raster_h = ray_bundle.rendering_data.w, ray_bundle.rendering_data.h
            to_raster = lambda ray_values: rearrange(ray_values, "(h w) d -> h w d", h=raster_h, w=raster_w)

            rgb = to_raster(unpack_optional(rendered.rgb))
            distance_nre = to_raster(rendered.distance.unsqueeze(-1))
            distance_world = self.viewer.get_dataset_interface().distance_nre_to_distance_world(distance_nre)

            rasters = RasterArtifacts(
                resolution="high" if is_high_resolution else "low",
                rgb=rgb,
                distance=distance_world,
            )

            return tree_map(rasters, lambda x: x.cpu() if torch.is_tensor(x) else x)

    def _send_render_to_client(self) -> None:
        """
        Postprocesses the raw nerf outputs and sends them to the client
        """
        if self.current_raster_artifacts is None:
            raise TypeError("_send_to_client called before setting self.current_render_artifacts")

        client_aspect = self.client.camera.aspect
        image = self.current_raster_artifacts.get_modality(self.modality_dropdown.value)
        image_padded = pad_to_aspect_ratio(image, client_aspect)

        quality = 95 if self.state == "high" else 75
        with self.client.atomic():
            self.client.scene.set_background_image(
                image_padded,
                format="jpeg",
                jpeg_quality=quality,
                depth=None,
            )
            self._maybe_set_camera_fov()

    def _send_comparison_to_client(self, timestamp_us: int, gt_rgb: torch.Tensor) -> None:
        assert self.current_raster_artifacts is not None

        modalities: dict[str, PILImage.Image] = {}
        for key in self.current_raster_artifacts.keys():
            image = PILImage.fromarray(self.current_raster_artifacts.get_modality(key))
            modalities[key] = image

        # add the ground truth RGB image
        gt_rgb_pil = PILImage.fromarray(gt_rgb.cpu().numpy())
        # TODO: subselect the actual pixels corresponding to rays instead of interpolating
        rgb_image_pil_resized = gt_rgb_pil.resize(
            modalities["rgb"].size,
            resample=PILImage.Resampling.LANCZOS,
        )
        modalities["rgb_gt"] = rgb_image_pil_resized

        # send to user
        assert self.current_trajectory is not None
        fname_prefix = f"{self.current_trajectory.trajectory_id.camera_sequence_str()}_{timestamp_us}"
        for modality_name, modality_content_pil in modalities.items():
            self.client.send_file_download(
                f"{fname_prefix}__{modality_name}.png",
                pil_image_to_bytes(modality_content_pil),
            )


class AbstractViewer(ABC):
    config: ViewerConfig
    host: str
    port: int
    # Allow model to be empty (used during NRM inference when primitives are not yet available)
    model: RenderableModel | None
    system_lock: LockLike
    # Lock the references to model and datasource_interface
    scene_lock: LockLike

    _ready: bool
    _server: Optional[viser.ViserServer]
    _running_render_statemachines: dict[int, RenderStateMachine]

    def __init__(self) -> None:
        self._ready = False
        self._server = None
        self._running_render_statemachines = {}
        self.scene_lock = Lock() if self.supports_clear_current_scene() else MockLock()

    def get_host_ip(self) -> str:
        if self.host != "auto":
            return self.host

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                # Doesn't even have to be reachable
                s.connect(("8.8.8.8", 1))
                internal_ip = s.getsockname()[0]
            except Exception:
                internal_ip = "127.0.0.1"
        return internal_ip

    def start_server(self) -> None:
        if not self._ready:
            self._server = viser.ViserServer(
                host=self.get_host_ip(),
                port=self.port,
            )

            self._server.on_client_disconnect(self.disconnect_client)
            self._server.on_client_connect(self.connect_new_client)

            self._ready = True

    def stop_server(self) -> None:
        if self._server is not None:
            raise RuntimeError("Server is not running")

        for client_id in self._running_render_statemachines:
            self._disconnect_client_by_id(client_id)

    def _disconnect_client_by_id(self, client_id: int) -> None:
        if client_id not in self._running_render_statemachines:
            return
        self._running_render_statemachines[client_id].running = False
        self._running_render_statemachines.pop(client_id)

    def disconnect_client(self, client: viser.ClientHandle) -> None:
        """
        Removes a client connection from the pool. The thread will stop once it checks
        its `running` flag.
        """
        self._disconnect_client_by_id(client.client_id)

    def connect_new_client(self, client: viser.ClientHandle) -> None:
        """
        Adds a new connection to the pool.
        """
        render_state_machine = RenderStateMachine(self, client)
        self._running_render_statemachines[client.client_id] = render_state_machine
        self._running_render_statemachines[client.client_id].start()

    @abstractmethod
    def get_dataset_interface(self) -> ViewerDatasetInterface:
        raise NotImplementedError("get_datasource_interface must be implemented")

    def supports_clear_current_scene(self) -> bool:
        return False

    def clear_current_scene(self) -> None:
        raise NotImplementedError("clear_current_scene is not implemented for this class")

    def update_scene(
        self,
        renderable_model: RenderableModel,
        dataset_interface: ViewerDatasetInterface,
        block_until_empty_again: bool = False,
    ):
        """Sets a new scene in the viewer if not already set (currently used mainly in LightningNRMViewer)"""

    def get_current_scene_id(self) -> tuple[int, int]:
        """
        Scene ID represented by the memory address of the model and the dataset interface.
        This is used by the render state machine to detect scene changes in the viewer.
        Need to return -1, -1 if the scene is empty.
        """
        if self.model is None:
            return -1, -1
        return id(self.model), id(self.get_dataset_interface())
