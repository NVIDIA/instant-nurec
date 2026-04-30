# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import time

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import trimesh
import viser
import viser.extras

from scipy.spatial.transform import Rotation as R

import ncore.data

from internal.scripts.ncore_vis.lazy_loader import NCoreLazyLoader
from ncore.sensors import CameraModel
from nre.utils.geometry import se3_matrix_to_tquat


class NCoreViewer:
    """NCore Viewer that creates a viser server to allow clients to connect to the specified host and port for
    visualizing the NCore dataset provided.
    """

    def __init__(
        self, shard_file_pattern: str, host: str = "0.0.0.0", port: int = 8080, ply_path: Optional[str] = None
    ):
        self.shard_file_pattern = shard_file_pattern
        self.host = host
        self.port = port
        self.ply_path: Optional[str] = ply_path

    def start_server(self) -> None:
        """
        Starts the viser server which exposes the HTTP url and Websocket URL
        within the command line.
        """
        self.loader = NCoreLazyLoader(self.shard_file_pattern)
        self.server = viser.ViserServer(
            host=self.host,
            port=self.port,
        )

        self.running_renderers: dict[int, NCoreRenderer] = {}
        self.server.on_client_connect(self.connect_new_client)
        self.server.on_client_disconnect(self.disconnect_client)

        while True:
            time.sleep(1.0)

    def connect_new_client(self, client: viser.ClientHandle) -> None:
        """
        Runs whenever a new client is connected to the viser server.
        Attaches a new renderer to the client and keeps track of the
        client.

        Args:
            client (viser.ClientHandle): a new web client connected to
                                         the viser server.
        """
        renderer = NCoreRenderer(self, client)
        self.running_renderers[client.client_id] = renderer

    def disconnect_client(self, client: viser.ClientHandle) -> None:
        self.running_renderers.pop(client.client_id)


class NCoreRenderer:
    """A class responsible for rendering a single viewer's page in viser."""

    def __init__(
        self,
        viewer: NCoreViewer,
        client: viser.ClientHandle,
    ) -> None:
        self.viewer = viewer
        self.client = client

        self.lidar_colors = np.array([[0, 122, 0], [130, 0, 120], [20, 170, 140], [20, 10, 170]], dtype=np.uint8)
        self.cancel_record = False

        self._populate_server()

    def _create_handle_maps(self):
        """
        Creates client handles that are maintained throughout the connection
        and keep track of client specific settings set in the visualizer. This
        should be called before rendering any elements using the client.
        """
        # Gui handles
        self.sensor_frame_handles: dict[str, viser.GuiInputHandle] = {}

        self.camera_frusta: dict[str, viser.SceneNodeHandle] = {}
        self.camera_poses: dict[str, viser.FrameHandle] = {}
        self.camera_labels: dict[str, viser.LabelHandle] = {}

        self.camera_frames: dict[str, int] = {}
        self.camera_visible: dict[str, bool] = {}
        self.camera_image_option: dict[str, str] = {}
        self.camera_overlay_cuboids: dict[str, bool] = {}
        self.camera_cuboids_source: dict[str, str] = {}
        self.camera_show_cuboid_labels: dict[str, bool] = {}

        self.camera_labels_visible = True

        for camera_id in self.viewer.loader.get_camera_ids():
            self.camera_frames[camera_id] = 0
            self.camera_visible[camera_id] = True
            self.camera_image_option[camera_id] = "Image"
            self.camera_overlay_cuboids[camera_id] = False
            self.camera_cuboids_source[camera_id] = ncore.data.LabelSource._member_names_[0]
            self.camera_show_cuboid_labels[camera_id] = True

        self.lidar_point_clouds: dict[str, viser.PointCloudHandle] = {}
        self.lidar_cuboids: dict[str, list[viser.MeshHandle]] = {}
        self.lidar_cuboids_source: dict[str, str] = {}
        self.lidar_cuboid_labels: dict[str, list[viser.LabelHandle]] = {}
        self.lidar_ghosted_point_clouds: dict[str, list[viser.PointCloudHandle]] = {}

        self.lidar_frames: dict[str, int] = {}
        self.lidar_color_style: dict[str, str] = {}
        self.lidar_point_size: dict[str, float] = {}
        self.lidar_ghosting: dict[str, int] = {}
        self.lidar_is_fused: dict[str, bool] = {}
        self.lidar_show_point_cloud: dict[str, bool] = {}
        self.lidar_fused_frame_step: dict[str, int] = {}
        self.lidar_fused_range: dict[str, tuple[int, int]] = {}

        self.lidar_show_cuboids: dict[str, bool] = {}
        self.lidar_show_cuboid_labels: dict[str, bool] = {}
        self.lidar_cuboid_is_fused: dict[str, bool] = {}
        self.lidar_cuboid_fused_frame_step: dict[str, int] = {}
        self.lidar_cuboid_fused_range: dict[str, tuple[int, int]] = {}

        for lidar_id in self.viewer.loader.get_lidar_ids():
            self.lidar_frames[lidar_id] = 0
            self.lidar_color_style[lidar_id] = "Semantic"
            self.lidar_point_size[lidar_id] = 0.025
            self.lidar_ghosting[lidar_id] = 1

            self.lidar_show_point_cloud[lidar_id] = True
            self.lidar_is_fused[lidar_id] = False
            self.lidar_fused_frame_step[lidar_id] = 1

            self.lidar_cuboids_source[lidar_id] = ncore.data.LabelSource._member_names_[0]
            self.lidar_show_cuboids[lidar_id] = True
            self.lidar_show_cuboid_labels[lidar_id] = True
            self.lidar_cuboid_is_fused[lidar_id] = False
            self.lidar_cuboid_fused_frame_step[lidar_id] = 1

        self.show_trajectory = True

        # PLY point clouds
        self.ply_point_clouds: dict[str, viser.PointCloudHandle] = {}
        self.ply_points: dict[str, np.ndarray] = {}
        self.ply_colors_from_file: dict[str, Optional[np.ndarray]] = {}
        self.ply_visible: dict[str, bool] = {}
        self.ply_point_size: dict[str, float] = {}

        self._preload_ply_clouds()

    def _create_static_elements(self) -> None:
        """
        Creates the static elements within the visualizer such as the title
        and background.
        """
        self.client.gui.set_panel_label("NCORE Data Controls")
        image = np.full(shape=(1000, 1000, 3), fill_value=40, dtype=np.uint8)
        self.client.scene.set_background_image(image=image)

    def _load_ply(self, path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load a PLY file and return (points[N,3], colors[N,3] or None). Colors are uint8 in RGB.
        """

        tm = trimesh.load(path, process=False)
        if isinstance(tm, trimesh.points.PointCloud):
            points = np.asarray(tm.vertices, dtype=np.float32)
            cols = None
            if tm.colors is not None and len(tm.colors) == points.shape[0]:
                cols = np.asarray(tm.colors, dtype=np.uint8)
                if cols.shape[1] >= 3:
                    cols = cols[:, :3]
            print("trimesh: loaded as PointCloud")
            return points, cols
        elif isinstance(tm, trimesh.Trimesh):
            points = np.asarray(tm.vertices, dtype=np.float32)
            cols = None
            if tm.visual is not None and hasattr(tm.visual, "vertex_colors"):
                vc = np.asarray(tm.visual.vertex_colors)
                if vc.size > 0:
                    cols = vc.astype(np.uint8)
                    if cols.shape[1] >= 3:
                        cols = cols[:, :3]
            print("trimesh: loaded as Trimesh (using vertices as points)")
            return points, cols
        else:
            # Unknown object; try to grab vertices attribute generically
            points = np.asarray(getattr(tm, "vertices"), dtype=np.float32)
            cols = None
            if hasattr(tm, "colors"):
                c = np.asarray(getattr(tm, "colors"))
                if c.size > 0:
                    cols = c.astype(np.uint8)
                    if cols.shape[1] >= 3:
                        cols = cols[:, :3]
            print("trimesh: loaded generically")
            return points, cols

    def _preload_ply_clouds(self) -> None:
        """Preload the single PLY file provided to the viewer (if any)."""
        if self.viewer.ply_path is None:
            return
        ply_path = self.viewer.ply_path
        try:
            points, colors = self._load_ply(ply_path)
            ply_id = Path(ply_path).stem
            self.ply_points[ply_id] = points
            self.ply_colors_from_file[ply_id] = colors
            self.ply_visible[ply_id] = True
            self.ply_point_size[ply_id] = 0.025
            print(f"Loaded PLY '{ply_path}', num_points={points.shape[0]}")
        except Exception as e:
            print(f"Failed to load PLY '{ply_path}': {e}")

    def _change_camera_visibility(self, camera_id: str, visible: bool, update_labels: bool = True) -> None:
        self.camera_visible[camera_id] = visible
        if camera_id in self.camera_frusta:
            self.camera_frusta[camera_id].visible = visible
        if camera_id in self.camera_poses:
            self.camera_poses[camera_id].visible = visible
        if update_labels and camera_id in self.camera_labels:
            self.camera_labels[camera_id].visible = visible
            self.camera_labels[camera_id].wxyz = self.camera_poses[camera_id].wxyz

    def _on_camera_frame_change(
        self,
        camera_id: str,
        slider: viser.GuiInputHandle,
        dropdown: viser.GuiDropdownHandle,
        checkbox: viser.GuiInputHandle,
        go_to_frame: viser.GuiButtonHandle,
    ) -> None:
        """
        Defines all the behaviors for when the camera frame settings changes.

        Args:
            camera_id (str): the current camera being changes
            slider (viser.GuiInputHandle): a frame slider for the camera's frame
            dropdown (viser.GuiDropdownHandle): a dropdown for the image type that
                                                selects the type of image (e.g. semantic)
            checkbox (viser.GuiInputHandle): a checkbox to show the camera or not
            go_to_frame (viser.GuiButtonHandle): a button that jumps to the camera
                                                 frustum and image
        """

        @slider.on_update
        def _(_):
            self.camera_frames[camera_id] = slider.value
            self.update_camera_data(camera_id)

        @dropdown.on_update
        def _(_):
            self.camera_image_option[camera_id] = dropdown.value
            self.update_camera_data(camera_id)

        @checkbox.on_update
        def _(_):
            self._change_camera_visibility(camera_id, checkbox.value)

        @go_to_frame.on_click
        def _(_):
            self.client.camera.wxyz = self.camera_poses[camera_id].wxyz
            self.client.camera.position = self.camera_poses[camera_id].position

    def _create_camera_guis(self, tab: viser.GuiTabGroupHandle) -> None:
        """
        Creates the camera gui elements within the viser control tab. These
        are visualization controls such as frame, image_type, etc.

        Args:
            tab (viser.GuiTabGroupHandle): the gui tab group handle used to
                                           create the `Cameras` tab within
                                           the visualiser controls
        """
        with tab.add_tab("Cameras"):
            toggle_cameras = self.client.gui.add_checkbox("Toggle Cameras", initial_value=True)
            show_labels = self.client.gui.add_checkbox("Show labels", initial_value=True)

            for camera_id in self.viewer.loader.get_camera_ids():
                # Get camera indices
                camera_indices = self.viewer.loader.get_sensor_indices(camera_id)
                image_options = ["Image"]
                if self.viewer.loader.has_camera_semantic_segmentation(camera_id):
                    image_options.extend(["Semantic Segmentation", "Semantic Segmentation (Overlay)"])
                if self.viewer.loader.has_camera_depth(camera_id):
                    image_options.append("Depth")
                if self.viewer.loader.has_camera_normals(camera_id):
                    image_options.append("Normals")

                with self.client.gui.add_folder(camera_id):
                    camera_frame_slider = self.client.gui.add_slider(
                        "Frame",
                        min=camera_indices[0],
                        max=camera_indices[-1],
                        step=1,
                        initial_value=0,
                        disabled=False,
                    )
                    self.sensor_frame_handles[camera_id] = camera_frame_slider

                    image_option_control = self.client.gui.add_dropdown(
                        "Image Option",
                        options=image_options,
                        initial_value="Image",
                        hint="Image data type shown in camera frustums",
                    )

                    show_camera_checkbox = self.client.gui.add_checkbox("Show Camera", initial_value=True)

                    with self.client.gui.add_folder("Cuboid Overlay"):
                        cuboid_overlay_checkbox = self.client.gui.add_checkbox(
                            "Overlay Cuboids", initial_value=self.camera_overlay_cuboids.get(camera_id, False)
                        )
                        cuboid_source_dropdown = self.client.gui.add_dropdown(
                            "Cuboid Source",
                            ncore.data.LabelSource._member_names_,
                            self.camera_cuboids_source[camera_id],
                        )
                        label_checkbox = self.client.gui.add_checkbox(
                            "Labels", initial_value=self.camera_show_cuboid_labels.get(camera_id, True)
                        )

                    go_to_frame = self.client.gui.add_button(
                        "Go to Frame",
                    )

                    self._on_camera_frame_change(
                        camera_id, camera_frame_slider, image_option_control, show_camera_checkbox, go_to_frame
                    )

                    @cuboid_overlay_checkbox.on_update
                    def _(_, _camera_id=camera_id, _checkbox=cuboid_overlay_checkbox):
                        self.camera_overlay_cuboids[_camera_id] = _checkbox.value
                        self.update_camera_data(_camera_id)

                    @cuboid_source_dropdown.on_update
                    def _(_, _camera_id=camera_id, _dropdown=cuboid_source_dropdown):
                        self.camera_cuboids_source[_camera_id] = _dropdown.value
                        if self.camera_overlay_cuboids[_camera_id]:
                            self.update_camera_data(_camera_id)

                    @label_checkbox.on_update
                    def _(_, _camera_id=camera_id, _checkbox=label_checkbox):
                        self.camera_show_cuboid_labels[_camera_id] = _checkbox.value
                        if self.camera_overlay_cuboids[_camera_id]:
                            self.update_camera_data(_camera_id)

            @show_labels.on_update
            def _(_):
                self.camera_labels_visible = show_labels.value
                for camera_id in self.camera_labels:
                    self.camera_labels[camera_id].visible = show_labels.value
                    self.camera_labels[camera_id].wxyz = self.camera_poses[camera_id].wxyz

            @toggle_cameras.on_update
            def _(_):
                for camera_id in self.camera_labels:
                    self._change_camera_visibility(camera_id, toggle_cameras.value, update_labels=False)

    def _on_lidar_data_change(
        self,
        lidar_id: str,
        frame_slider: viser.GuiInputHandle,
        color_dropdown: viser.GuiDropdownHandle,
        point_size: viser.GuiInputHandle,
        ghosting: viser.GuiInputHandle,
        cuboid_checkbox: viser.GuiInputHandle,
        fused_checkbox: viser.GuiInputHandle,
        fused_range: viser.GuiInputHandle,
        point_cloud_checkbox: viser.GuiInputHandle,
        fused_frame_step: viser.GuiInputHandle,
        cuboid_source_dropdown: viser.GuiDropdownHandle,
        cuboid_fused_checkbox: viser.GuiInputHandle,
        cuboid_fused_frame_step: viser.GuiInputHandle,
        cuboid_fused_range: viser.GuiInputHandle,
        cuboid_label_checkbox: viser.GuiInputHandle,
        regenerate_button: viser.GuiButtonHandle,
    ) -> None:
        """
        Defines all the behaviors when the lidar settings change

        Args:
            lidar_id (str): lidar sensor that was changed
            frame_slider (viser.GuiInputHandle): a slider that controls the frame
            color_dropdown (viser.GuiDropdownHandle): a dropdown that controls the
                                                      the point cloud color type
            point_size (viser.GuiInputHandle): a slider that controls the point
                                               size for the point cloud
            ghosting (viser.GuiInputHandle): a slider that controls point cloud
                                             ghosting (show x point clouds before and
                                             after)
            cuboid_checkbox (viser.GuiInputHandle): a checkbox to show cuboids or not
            fused_checkbox (viser.GuiInputHandle): a checkbox to show fused point
                                                   cloud or not
            fused_range (viser.GuiInputHandle): a multi-slider that controls the frame
                                                range for the fused point cloud
            point_cloud_checkbox (viser.GuiInputHandle): a checkbox to show the point
                                                         cloud or not
            fused_frame_step (viser.GuiInputHandle): a slider to control the frame-step
                                                     of the point cloud
            cuboid_source_dropdown (viser.GuiDropdownHandle): a dropdown that controls the
                                                            source of the cuboids
            cuboid_fused_checkbox (viser.GuiInputHandle): a checkbox to show cuboid
                                                          fused data or not
            cuboid_fused_frame_step (viser.GuiInputHandle): a slider that controls
                                                            the frame step for cuboid
                                                            fused data
            cuboid_fused_range (viser.GuiInputHandle): a multi-slider that controls the
                                                       fused range of the fused cuboid
                                                       data
            cuboid_label_checkbox (viser.GuiInputHandle): a checkbox to show the cuboid
                                                          labels or not
            regenerate_button (viser.GuiButtonHandle): a button that regenerates data for
                                                       the given lidar sensor
        """

        @frame_slider.on_update
        def _(_):
            self.lidar_frames[lidar_id] = frame_slider.value
            if not self.lidar_is_fused[lidar_id] or not self.lidar_cuboid_is_fused[lidar_id]:
                self.update_lidar_data(lidar_id)

        @color_dropdown.on_update
        def _(_):
            self.lidar_color_style[lidar_id] = color_dropdown.value
            self.update_lidar_data(lidar_id)

        @point_size.on_update
        def _(_):
            self.lidar_point_size[lidar_id] = point_size.value / 1000
            self.update_lidar_data(lidar_id)

        @ghosting.on_update
        def _(_):
            self.lidar_ghosting[lidar_id] = ghosting.value
            if not self.lidar_is_fused[lidar_id]:
                self.update_lidar_data(lidar_id)

        @cuboid_checkbox.on_update
        def _(_):
            self.lidar_show_cuboids[lidar_id] = cuboid_checkbox.value
            if lidar_id in self.lidar_cuboids:
                for cuboid in self.lidar_cuboids[lidar_id]:
                    cuboid.visible = cuboid_checkbox.value

        @fused_checkbox.on_update
        def _(_):
            self.lidar_is_fused[lidar_id] = fused_checkbox.value
            self.update_lidar_data(lidar_id)

        @fused_range.on_update
        def _(_):
            self.lidar_fused_range[lidar_id] = fused_range.value
            if self.lidar_is_fused[lidar_id]:
                self.update_lidar_data(lidar_id)

        @point_cloud_checkbox.on_update
        def _(_):
            self.lidar_show_point_cloud[lidar_id] = point_cloud_checkbox.value
            if lidar_id in self.lidar_point_clouds:
                self.lidar_point_clouds[lidar_id].visible = point_cloud_checkbox.value
            if lidar_id in self.lidar_ghosted_point_clouds:
                for point_cloud in self.lidar_ghosted_point_clouds[lidar_id]:
                    point_cloud.visible = point_cloud_checkbox.value

        @fused_frame_step.on_update
        def _(_):
            self.lidar_fused_frame_step[lidar_id] = fused_frame_step.value
            if self.lidar_is_fused[lidar_id]:
                self.update_lidar_data(lidar_id)

        @cuboid_source_dropdown.on_update
        def _(_):
            self.lidar_cuboids_source[lidar_id] = cuboid_source_dropdown.value
            self.update_lidar_data(lidar_id)

        @cuboid_fused_checkbox.on_update
        def _(_):
            self.lidar_cuboid_is_fused[lidar_id] = cuboid_fused_checkbox.value
            self.update_lidar_data(lidar_id)

        @cuboid_fused_frame_step.on_update
        def _(_):
            self.lidar_cuboid_fused_frame_step[lidar_id] = cuboid_fused_frame_step.value
            if self.lidar_cuboid_is_fused[lidar_id]:
                self.update_lidar_data(lidar_id)

        @cuboid_fused_range.on_update
        def _(_):
            self.lidar_cuboid_fused_range[lidar_id] = cuboid_fused_range.value
            if self.lidar_cuboid_is_fused[lidar_id]:
                self.update_lidar_data(lidar_id)

        @cuboid_label_checkbox.on_update
        def _(_):
            self.lidar_show_cuboid_labels[lidar_id] = cuboid_label_checkbox.value
            if lidar_id in self.lidar_cuboid_labels:
                for i, label in enumerate(self.lidar_cuboid_labels[lidar_id]):
                    label.visible = cuboid_label_checkbox.value
                    label.wxyz = self.lidar_cuboids[lidar_id][0].wxyz

        @regenerate_button.on_click
        def _(_):
            self.update_lidar_data(lidar_id)

    def _create_lidar_guis(self, tab: viser.GuiTabGroupHandle) -> None:
        """
        Creates the lidar gui elements within the viser control tab. These
        are visualization controls such as frame, point clouds, fused data,
        cuboids, etc.

        Args:
            tab (viser.GuiTabGroupHandle): the gui tab group handle used to
                                           create the `Lidars` tab within
                                           the visualiser controls
        """
        with tab.add_tab("Lidars"):
            for lidar_id in self.viewer.loader.get_lidar_ids():
                # Get lidar indices
                lidar_indices = self.viewer.loader.get_sensor_indices(lidar_id)
                lidar_options = ["Semantic", "Intensity", "Intensity γ=1/4", "RGB"]
                cuboid_source_options = ncore.data.LabelSource._member_names_
                self.lidar_fused_range[lidar_id] = (lidar_indices[0], lidar_indices[-1])
                self.lidar_cuboid_fused_range[lidar_id] = (lidar_indices[0], lidar_indices[-1])

                with self.client.gui.add_folder(lidar_id):
                    frame_slider = self.client.gui.add_slider(
                        "Frame",
                        min=lidar_indices[0],
                        max=lidar_indices[-1],
                        step=1,
                        initial_value=0,
                    )
                    self.sensor_frame_handles[lidar_id] = frame_slider

                    with self.client.gui.add_folder("Point Cloud Settings"):
                        color_dropdown = self.client.gui.add_dropdown(
                            "Color Style",
                            lidar_options,
                            lidar_options[0],
                        )

                        point_size = self.client.gui.add_slider(
                            "Point Size Radius (cm)", min=0, max=50, step=1, initial_value=25
                        )

                        ghosting = self.client.gui.add_slider("Ghosting", min=1, max=5, step=1, initial_value=1)

                        point_cloud_checkbox = self.client.gui.add_checkbox("Visible", initial_value=True)

                        fused_checkbox = self.client.gui.add_checkbox("Fuse", initial_value=False)

                        frame_step_init = 40 if 40 <= lidar_indices[-1] else lidar_indices[-1]
                        self.lidar_fused_frame_step[lidar_id] = frame_step_init
                        fused_frame_step = self.client.gui.add_slider(
                            "Frame Step (Fused)",
                            min=min(1, len(lidar_indices) - 1),  # set min to 0 when there is only 1 lidar frame
                            max=lidar_indices[-1],
                            step=1,
                            initial_value=frame_step_init,
                        )

                        fused_range = self.client.gui.add_multi_slider(
                            "Fused Range",
                            min=lidar_indices[0],
                            max=lidar_indices[-1],
                            step=1,
                            initial_value=(lidar_indices[0], lidar_indices[-1]),
                        )

                    with self.client.gui.add_folder("Cuboid Settings"):
                        cuboid_source_dropdown = self.client.gui.add_dropdown(
                            "Cuboid Source", cuboid_source_options, cuboid_source_options[0]
                        )

                        cuboid_checkbox = self.client.gui.add_checkbox("Visible", initial_value=True)

                        cuboid_label_checkbox = self.client.gui.add_checkbox("Labels", initial_value=True)

                        cuboid_fused_checkbox = self.client.gui.add_checkbox("Fuse", initial_value=False)

                        self.lidar_fused_frame_step[lidar_id] = frame_step_init
                        cuboid_fused_frame_step = self.client.gui.add_slider(
                            "Frame Step (Fused)",
                            min=min(1, len(lidar_indices) - 1),  # set min to 0 when there is only 1 lidar frame
                            max=lidar_indices[-1],
                            step=1,
                            initial_value=frame_step_init,
                        )

                        cuboid_fused_range = self.client.gui.add_multi_slider(
                            "Fused Range",
                            min=lidar_indices[0],
                            max=lidar_indices[-1],
                            step=1,
                            initial_value=(lidar_indices[0], lidar_indices[-1]),
                        )

                    regenerate_button = self.client.gui.add_button(
                        "Regenerate Data", hint="Run this to regenerate existing data"
                    )

                    self._on_lidar_data_change(
                        lidar_id,
                        frame_slider,
                        color_dropdown,
                        point_size,
                        ghosting,
                        cuboid_checkbox,
                        fused_checkbox,
                        fused_range,
                        point_cloud_checkbox,
                        fused_frame_step,
                        cuboid_source_dropdown,
                        cuboid_fused_checkbox,
                        cuboid_fused_frame_step,
                        cuboid_fused_range,
                        cuboid_label_checkbox,
                        regenerate_button,
                    )

    def _update_all_sensors(self, ref_frame: int) -> None:
        with self.client.atomic():
            for sensor_id in self.sensor_frame_handles:
                frame = self.viewer.loader.get_closest_sensor_frame(ref_frame, self.reference_sensor, sensor_id)
                self.sensor_frame_handles[sensor_id].value = frame
        self.client.flush()

    def _on_reference_frame_update(self, frame_slider: viser.GuiInputHandle):
        """
        Handles the behavior when the reference frame within the scene is updated. This
        adjusts the frames of all existing sensors within the scene to match the reference
        frame as closely as possible.

        Args:
            frame_slider (viser.GuiInputHandle): a slider that controls the reference frame
        """

        @frame_slider.on_update
        def _(_):
            self._update_all_sensors(frame_slider.value)

    def update_ply_cloud(self, ply_id: str) -> None:
        with self.client.atomic():
            # Remove existing
            if ply_id in self.ply_point_clouds:
                self.ply_point_clouds[ply_id].remove()

            if ply_id not in self.ply_points:
                return

            points = self.ply_points[ply_id]

            colors_arr: Optional[np.ndarray] = self.ply_colors_from_file.get(ply_id)
            if colors_arr is None:
                # Fallback to a default solid color if PLY has no colors
                rgb = np.array([255, 0, 0], dtype=np.uint8)
                colors_arr = np.repeat(rgb[None, :], repeats=points.shape[0], axis=0)

            handle_name = f"/ply/{ply_id}"
            pc = self.client.scene.add_point_cloud(
                handle_name,
                points=points,
                colors=colors_arr,
                point_size=self.ply_point_size.get(ply_id, 0.025),
                point_shape="circle",
                visible=self.ply_visible.get(ply_id, True),
            )
            self.ply_point_clouds[ply_id] = pc
        self.client.flush()

    def _record_scene(
        self, start_frame: int, end_frame: int, file_name: str, width: int, height: int, fps: int
    ) -> None:
        """
        Creates a video recording of the frames from [start_frame] to [end_frame] from
        client camera's POV.

        Args:
            start_frame (int): frame to start the recording at
            end_frame (int): frame to end the recording at
        """

        # Initialize the video writer
        fc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        output_file = f"{file_name}.mp4"
        out = cv2.VideoWriter(output_file, fc, fps, (width, height))

        for frame in range(start_frame, end_frame + 1):
            if self.cancel_record:
                out.release()
                self.cancel_record = False
                return

            self._update_all_sensors(frame)
            # Pass the queried resolution to get_render()
            rendered_frame = self.client.camera.get_render(height=height, width=width)
            out.write(cv2.cvtColor(np.array(rendered_frame), cv2.COLOR_RGB2BGR))

        # Release the video writer
        out.release()
        if self.cancel_record:
            self.cancel_record = False
            return

        # Pass the video to the client for download
        with open(output_file, "rb") as f:
            file_data = f.read()
        self.client.send_file_download(output_file, file_data)

    def _create_scene_guis(self, tab: viser.GuiTabGroupHandle) -> None:
        """
        Creates GUIs that handle fundamental scene behaviors such as controlling the
        global scene frame with reference to a chosen sensor.

        Args:
            tab (viser.GuiTabGroupHandle): The GUI tab used to navigate to the
                                           scene control options.
        """

        with tab.add_tab("Scene"):
            gui_traj_checkbox = self.client.gui.add_checkbox("Rig Trajectory", self.show_trajectory)

            sensor_list = self.viewer.loader.get_camera_ids()
            sensor_list.extend(self.viewer.loader.get_lidar_ids())
            self.reference_sensor: str = sensor_list[0]

            gui_reference_dropdown = self.client.gui.add_dropdown(
                label="Reference Sensor", options=sensor_list, initial_value=sensor_list[0]
            )

            init_sensor_indices = self.viewer.loader.get_sensor_indices(sensor_list[0])
            self.reference_frame_slider = self.client.gui.add_slider(
                "Reference Frame", min=init_sensor_indices[0], max=init_sensor_indices[-1], initial_value=0, step=1
            )

            self._on_reference_frame_update(self.reference_frame_slider)

            init_range = (init_sensor_indices[0], init_sensor_indices[-1])
            self.recording_range = self.client.gui.add_multi_slider(
                "Recording Range", min=init_range[0], max=init_range[-1], step=1, initial_value=init_range
            )
            self.recording_file_name = self.client.gui.add_text(label="File Name", initial_value="recording")
            self.recording_fps = self.client.gui.add_number(label="FPS", initial_value=30, min=1, max=60, step=1)
            self.recording_width = self.client.gui.add_number(
                label="Width", initial_value=1920, min=1, max=3840, step=1
            )
            self.recording_height = self.client.gui.add_number(
                label="Height", initial_value=1280, min=1, max=2560, step=1
            )
            self.start_recording_button = self.client.gui.add_button("Start Recording")
            self.cancel_recording_button = self.client.gui.add_button("Cancel")

            @self.start_recording_button.on_click
            def _(_):
                self.start_recording_button.disabled = True
                self._record_scene(
                    start_frame=self.recording_range.value[0],
                    end_frame=self.recording_range.value[1],
                    file_name=self.recording_file_name.value,
                    width=self.recording_width.value,
                    height=self.recording_height.value,
                    fps=self.recording_fps.value,
                )
                self.start_recording_button.disabled = False

            @self.cancel_recording_button.on_click
            def _(_):
                self.cancel_record = True

            @gui_traj_checkbox.on_update
            def _(_):
                self.show_trajectory = gui_traj_checkbox.value
                with self.client.atomic():
                    for traj in self._rig_trajectories:
                        traj.visible = self.show_trajectory
                self.client.flush()

            @gui_reference_dropdown.on_update
            def _(_):
                self.reference_sensor = gui_reference_dropdown.value
                sensor_indices = self.viewer.loader.get_sensor_indices(gui_reference_dropdown.value)

                self.reference_frame_slider.remove()
                self.reference_frame_slider = self.client.gui.add_slider(
                    "Reference Frame", min=sensor_indices[0], max=sensor_indices[-1], initial_value=0, step=1
                )

                self.recording_range.remove()
                self.recording_file_name.remove()
                self.recording_fps.remove()
                self.recording_height.remove()
                self.recording_width.remove()
                self.start_recording_button.remove()
                self.cancel_recording_button.remove()

                self._on_reference_frame_update(self.reference_frame_slider)

                self.recording_range = self.client.gui.add_multi_slider(
                    "Recording Range",
                    min=sensor_indices[0],
                    max=sensor_indices[-1],
                    step=1,
                    initial_value=(sensor_indices[0], sensor_indices[-1]),
                )
                self.recording_file_name = self.client.gui.add_text(label="File Name", initial_value="recording")
                self.recording_fps = self.client.gui.add_number(label="FPS", initial_value=30, min=1, max=60, step=1)
                self.recording_width = self.client.gui.add_number(
                    label="Width", initial_value=1920, min=1, max=3840, step=1
                )
                self.recording_height = self.client.gui.add_number(
                    label="Height", initial_value=1280, min=1, max=2560, step=1
                )
                self.start_recording_button = self.client.gui.add_button("Start Recording")
                self.cancel_recording_button = self.client.gui.add_button("Cancel")

                @self.start_recording_button.on_click
                def _(_):
                    self.start_recording_button.disabled = True
                    self._record_scene(
                        start_frame=self.recording_range.value[0],
                        end_frame=self.recording_range.value[1],
                        file_name=self.recording_file_name.value,
                        width=self.recording_width.value,
                        height=self.recording_height.value,
                        fps=self.recording_fps.value,
                    )
                    self.start_recording_button.disabled = False

                @self.cancel_recording_button.on_click
                def _(_):
                    self.cancel_record = True

    def _create_ply_guis(self, tab: viser.GuiTabGroupHandle) -> None:
        if self.viewer.ply_path is None:
            return
        with tab.add_tab("PLY"):
            toggle_ply = self.client.gui.add_checkbox("Toggle PLY", initial_value=True)

            ply_id = Path(self.viewer.ply_path).stem
            with self.client.gui.add_folder(ply_id):
                visible_checkbox = self.client.gui.add_checkbox(
                    "Visible", initial_value=self.ply_visible.get(ply_id, True)
                )
                point_size = self.client.gui.add_slider(
                    "Point Size Radius (cm)",
                    min=0,
                    max=50,
                    step=1,
                    initial_value=int(self.ply_point_size.get(ply_id, 0.025) * 1000),
                )

                @visible_checkbox.on_update
                def _(_event, _ply_id=ply_id):
                    self.ply_visible[_ply_id] = visible_checkbox.value
                    if _ply_id in self.ply_point_clouds:
                        self.ply_point_clouds[_ply_id].visible = visible_checkbox.value

                @point_size.on_update
                def _(_event, _ply_id=ply_id):
                    self.ply_point_size[_ply_id] = point_size.value / 1000
                    self.update_ply_cloud(_ply_id)

            @toggle_ply.on_update
            def _(_):
                with self.client.atomic():
                    self.ply_visible[ply_id] = toggle_ply.value
                    if ply_id in self.ply_point_clouds:
                        self.ply_point_clouds[ply_id].visible = toggle_ply.value
                self.client.flush()

    def _populate_server(self) -> None:
        """Populates the viser server with NCORE data visualizations"""
        self._create_handle_maps()
        self._create_static_elements()
        self._create_trajectories()

        tab = self.client.gui.add_tab_group()
        self._create_camera_guis(tab)
        self._create_lidar_guis(tab)
        self._create_scene_guis(tab)
        self._create_ply_guis(tab)

        # Add camera frustums of existing cameras in the dataset
        for camera_id in self.viewer.loader.get_camera_ids():
            self.update_camera_data(camera_id)

        # Add lidar data for existing lidar sensors in the dataset
        for lidar_id in self.viewer.loader.get_lidar_ids():
            self.update_lidar_data(lidar_id)

        # Add PLY point clouds if any
        if self.viewer.ply_path is not None:
            ply_id = Path(self.viewer.ply_path).stem
            self.update_ply_cloud(ply_id)

    def _move_client_to_camera_frustum(self, frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
        """
        Handles the behavior when a camera frustum is clicked by the client. Current behavior is
        to move the client camera to the frustum clicked to visualize the 2d image rendered.

        Args:
            frustum (viser.CameraFrustumHandle): a frustum clicked within the scene
            frame (viser.FrameHandle): the frame associated with the frustum which, in viser,
                                       maintains the position and coordinates of the frustum
        """

        @frustum.on_click
        def _(_) -> None:
            self.client.camera.wxyz = frame.wxyz
            self.client.camera.position = frame.position

    def _create_trajectories(self):
        """
        Creates and renderes the trajectories of the rig provided in the NCORE dataset. This
        also renders little arrows along the trajectory to indicate the rig direction.
        """
        trajectories = self.viewer.loader.get_trajectories()
        self._rig_trajectories: list[viser.MeshHandle] = []

        rotation_left = R.from_euler("xyz", angles=[0, 0, 50], degrees=True)
        T_rotate_left = np.eye(4)
        T_rotate_left[:3, :3] = rotation_left.as_matrix()
        T_translate_left = np.identity(n=4)
        T_translate_left[1, 3] = 0.2

        rotation_right = R.from_euler("xyz", angles=[0, 0, -50], degrees=True)
        T_rotate_right = np.eye(4)
        T_rotate_right[:3, :3] = rotation_right.as_matrix()
        T_translate_right = np.identity(n=4)
        T_translate_right[1, 3] = -0.2

        for i in range(trajectories.shape[0]):
            T_trajectory = trajectories[i]
            tquat = se3_matrix_to_tquat(T_trajectory)

            trajectory = self.client.scene.add_box(
                name=f"rig_trajectory/{i}",
                color=(1.0, 0.0, 0.0),
                dimensions=(1.0, 0.2, 0.1),
                wxyz=np.roll(tquat[3:], 1),
                position=tquat[:3].numpy(),
                visible=self.show_trajectory,
            )
            self._rig_trajectories.append(trajectory)

            # Create arrows
            if i % 10 == 0:
                tquat_left = se3_matrix_to_tquat(T_trajectory @ T_translate_left @ T_rotate_left)
                arrow_left = self.client.scene.add_box(
                    name=f"rig_trajectory/{i}_left_arrow",
                    color=(1.0, 0.0, 0.0),
                    dimensions=(0.2, 0.7, 0.1),
                    wxyz=np.roll(tquat_left[3:], 1),
                    position=tquat_left[:3].numpy(),
                    visible=self.show_trajectory,
                )
                self._rig_trajectories.append(arrow_left)

                tquat_right = se3_matrix_to_tquat(T_trajectory @ T_translate_right @ T_rotate_right)
                arrow_right = self.client.scene.add_box(
                    name=f"rig_trajectory/{i}_right_arrow",
                    color=(1.0, 0.0, 0.0),
                    dimensions=(0.2, 0.7, 0.1),
                    wxyz=np.roll(tquat_right[3:], 1),
                    position=tquat_right[:3].numpy(),
                    visible=self.show_trajectory,
                )
                self._rig_trajectories.append(arrow_right)

    def update_lidar_data(self, lidar_id: str) -> None:
        """
        Renders all the lidar data given the current, internal, client-specific
        settings set.

        Args:
            lidar_id (str): id of the lidar sensor for which the data is rendered.
        """
        with self.client.atomic():
            if lidar_id in self.lidar_point_clouds:
                self.lidar_point_clouds[lidar_id].remove()
            if lidar_id in self.lidar_cuboids:
                for cuboid in self.lidar_cuboids[lidar_id]:
                    cuboid.remove()
            if lidar_id in self.lidar_cuboid_labels:
                for label in self.lidar_cuboid_labels[lidar_id]:
                    label.remove()
            if lidar_id in self.lidar_ghosted_point_clouds:
                for point_cloud in self.lidar_ghosted_point_clouds[lidar_id]:
                    point_cloud.remove()

            frame = self.lidar_frames[lidar_id]
            color_style = self.lidar_color_style[lidar_id]
            point_size = self.lidar_point_size[lidar_id]
            ghosting = self.lidar_ghosting[lidar_id]
            show_point_cloud = self.lidar_show_point_cloud[lidar_id]
            is_fused = self.lidar_is_fused[lidar_id]
            fused_range = self.lidar_fused_range[lidar_id]
            fused_frame_step = self.lidar_fused_frame_step[lidar_id]

            cuboid_source = self.lidar_cuboids_source[lidar_id]
            show_cuboids = self.lidar_show_cuboids[lidar_id]
            show_cuboid_labels = self.lidar_show_cuboid_labels[lidar_id]
            cuboid_is_fused = self.lidar_cuboid_is_fused[lidar_id]
            cuboid_fused_frame_step = self.lidar_cuboid_fused_frame_step[lidar_id]
            cuboid_fused_range = self.lidar_cuboid_fused_range[lidar_id]

            point_cloud = self._create_lidar_point_cloud(
                lidar_id=lidar_id,
                frame=frame,
                color_style=color_style,
                point_size=point_size,
                is_fused=is_fused,
                fused_range=fused_range,
                fused_frame_step=fused_frame_step,
                visible=show_point_cloud,
            )
            cuboids = self._create_lidar_cuboids(
                lidar_id=lidar_id,
                frame=frame,
                cuboid_source=cuboid_source,
                is_fused=cuboid_is_fused,
                fused_range=cuboid_fused_range,
                fused_frame_step=cuboid_fused_frame_step,
                visible=show_cuboids,
            )
            cuboid_labels = self._create_lidar_cuboids_labels(
                lidar_id=lidar_id,
                frame=frame,
                cuboid_source=cuboid_source,
                is_fused=cuboid_is_fused,
                fused_range=cuboid_fused_range,
                fused_frame_step=cuboid_fused_frame_step,
                visible=show_cuboid_labels,
            )

            ghosted_point_clouds = []
            if ghosting > 1 and not self.lidar_is_fused[lidar_id]:
                for i in range(1, ghosting):
                    if frame - i < 0 and frame + i >= len(self.viewer.loader.get_sensor_indices(lidar_id)):
                        break

                    if frame - i >= 0:
                        prev_pc = self._create_lidar_point_cloud(
                            lidar_id=lidar_id,
                            frame=frame - i,
                            color_style=color_style,
                            point_size=point_size,
                            is_fused=False,
                            fused_range=fused_range,
                            fused_frame_step=fused_frame_step,
                            visible=show_point_cloud,
                            lcolor=i,
                        )
                        ghosted_point_clouds.append(prev_pc)

                    if frame + i < len(self.viewer.loader.get_sensor_indices(lidar_id)):
                        next_pc = self._create_lidar_point_cloud(
                            lidar_id=lidar_id,
                            frame=frame + i,
                            color_style=color_style,
                            point_size=point_size,
                            is_fused=False,
                            fused_range=fused_range,
                            fused_frame_step=fused_frame_step,
                            visible=show_point_cloud,
                            lcolor=i,
                        )
                        ghosted_point_clouds.append(next_pc)

            self.lidar_point_clouds[lidar_id] = point_cloud
            self.lidar_cuboids[lidar_id] = cuboids
            self.lidar_cuboid_labels[lidar_id] = cuboid_labels
            self.lidar_ghosted_point_clouds[lidar_id] = ghosted_point_clouds
        self.client.flush()

    def _create_lidar_point_cloud(
        self,
        lidar_id: str,
        frame: int,
        color_style: str,
        point_size: float,
        is_fused: bool,
        fused_range: tuple[int, int],
        fused_frame_step: int,
        visible: bool,
        lcolor: int = 0,
    ) -> viser.PointCloudHandle:
        """
        Creates and renders a point cloud for a lidar

        Args:
            lidar_id (str): id of the lidar sensor for which the point cloud
                            is created.
            frame (int): the frame of the lidar at which to get data from
            color_style (str): the style of the point colors (e.g. semantic)
            point_size (float): size of the points (in cm) of the point cloud
            is_fused (bool): whether to create a fused (multi-frame) point cloud
            fused_range (tuple[int, int]): if fused, the range of frames to render
                                           the point cloud
            fused_frame_step (int): if fused, how many frames to skip per frame
                                    chosen
            visible (bool): whether the point cloud is visible within the scene
            lcolor (int, optional): lambda of the color (used by ghosting to more
                                    easily visualize overlapping point clouds). A
                                    value used to obtain a predefined color set
                                    for the points.

        Returns:
            viser.PointCloudHandle: a point cloud rendered handle in viser
        """
        handle_name = (
            f"/lidars/{lidar_id}/f{frame}/point_cloud"
            if not is_fused
            else f"/lidars/{lidar_id}/f{fused_range[0]}-f{fused_range[1]}/fused_point_cloud"
        )
        if is_fused:
            points = self.viewer.loader.get_fused_point_cloud(
                lidar_id, fused_range[0], fused_range[1], fused_frame_step
            )
            colors = self.viewer.loader.get_fused_point_cloud_color(
                lidar_id, fused_range[0], fused_range[1], fused_frame_step, type=color_style
            )
        else:
            points = self.viewer.loader.get_point_cloud(lidar_id, frame)
            colors = self.viewer.loader.get_point_cloud_color(lidar_id, frame, color_style)

        if lcolor >= 1 and lcolor <= self.lidar_colors.shape[0]:
            colors += self.lidar_colors[lcolor - 1]

        point_cloud = self.client.scene.add_point_cloud(
            handle_name, points=points, colors=colors, point_size=point_size, point_shape="circle", visible=visible
        )
        return point_cloud

    def _create_lidar_cuboids(
        self,
        lidar_id: str,
        frame: int,
        cuboid_source: str,
        is_fused: bool,
        fused_range: tuple[int, int],
        fused_frame_step: int,
        visible: bool,
        lcolor: int = 0,
    ) -> list[viser.MeshHandle]:
        """
        Creates and renders cuboid bounding boxes (bboxes) for a lidar

        Args:
            lidar_id (str): id of the lidar sensor for which the bboxes is created.
            frame (int): the frame of the lidar at which to get data from
            cuboid_soruce (str): the source of the cuboids (e.g. autolabels)
            is_fused (bool): whether to render fused (multi-frame) labels
            fused_range (tuple[int, int]): if fused, the range of frames to render
                                           the bboxes
            fused_frame_step (int): if fused, how frames to skip per frame chosen
            visible (bool): whether the bboxes are visible within the scene
            lcolor (int, optional): lambda of the color (used by ghosting to more
                                    easily visualize overlapping bboxes). A
                                    value used to obtain a predefined color set
                                    for the points.

        Returns:
            list[viser.MeshHandle]: a list of bounding box handles in viser
        """
        cuboids: list[viser.MeshHandle] = []
        if is_fused:
            cuboid_data = self.viewer.loader.get_fused_cuboid_data(
                lidar_id, fused_range[0], fused_range[1], fused_frame_step, cuboid_source
            )
        else:
            cuboid_data = self.viewer.loader.get_cuboid_data(lidar_id, frame, cuboid_source)

        if cuboid_data is None:
            return []

        for i in range(cuboid_data.shape[0]):
            handle_name = (
                f"/lidars/{lidar_id}/f{frame}/cuboid_{i}"
                if not is_fused
                else f"/lidars/{lidar_id}/f{fused_range[0]}-f{fused_range[1]}/cuboid_{i}"
            )
            pose = cuboid_data[i]["pose"][0]
            bbox = cuboid_data[i]["bbox"][0]
            label_class = cuboid_data[i]["class"][0]
            tquat = se3_matrix_to_tquat(pose)

            mesh = trimesh.creation.box(bbox[3:6])

            color = self.viewer.loader.get_cuboid_class_color(label_class)
            if lcolor >= 1 and lcolor <= self.lidar_colors.shape[0]:
                color += self.lidar_colors[lcolor - 1]

            bbox_handle = self.client.scene.add_mesh_simple(
                name=handle_name,
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=color,
                position=tquat[:3].numpy(),
                wxyz=np.roll(tquat[3:], 1),
                wireframe=True,
                visible=visible,
            )
            cuboids.append(bbox_handle)
        return cuboids

    def _create_lidar_cuboids_labels(
        self,
        lidar_id: str,
        frame: int,
        cuboid_source: str,
        is_fused: bool,
        fused_range: tuple[int, int],
        fused_frame_step: int,
        visible: bool,
    ) -> list[viser.LabelHandle]:
        """
        Creates and renders cuboid bounding boxe labels for a lidar

        Args:
            lidar_id (str): id of the lidar sensor for which the labels are created.
            frame (int): the frame of the lidar at which to get labels from
            cuboid_soruce (str): the source of the cuboids (e.g. autolabels)
            is_fused (bool): whether to render fused (multi-frame) label
            fused_range (tuple[int, int]): if fused, the range of frames to render
                                           the labels
            fused_frame_step (int): if fused, how frames to skip per frame chosen
            visible (bool): whether the labels are visible within the scene

        Returns:
            list[viser.LabelHandle]: a list of label handles in viser
        """
        labels: list[viser.LabelHandle] = []
        if is_fused:
            cuboid_data = self.viewer.loader.get_fused_cuboid_data(
                lidar_id, fused_range[0], fused_range[1], fused_frame_step, cuboid_source
            )
        else:
            cuboid_data = self.viewer.loader.get_cuboid_data(lidar_id, frame, cuboid_source)

        if cuboid_data is None:
            return []

        for i in range(cuboid_data.shape[0]):
            handle_name = (
                f"/labels/{lidar_id}/f{frame}/cuboid_{i}_label"
                if not is_fused
                else f"/labels/{lidar_id}/f{fused_range[0]}-f{fused_range[1]}/cuboid_{i}"
            )
            pose = cuboid_data[i]["pose"][0]
            label_class = cuboid_data[i]["class"][0]
            track_id = cuboid_data[i]["track_id"][0]
            node_name = f"{track_id}[{label_class}]"
            tquat = se3_matrix_to_tquat(pose)

            bbox_label_handle = self.client.scene.add_label(
                handle_name, node_name, wxyz=np.roll(tquat[3:], 1), position=tquat[:3].numpy(), visible=visible
            )
            labels.append(bbox_label_handle)
        return labels

    def update_camera_data(self, camera_id: str) -> None:
        """
        Renders all the camera data given the current, internal, client-specific
        settings set.

        Args:
            camera_id (str): id of the camera sensor for which the data is rendered.
        """
        with self.client.atomic():
            if camera_id in self.camera_frusta:
                self.camera_frusta[camera_id].remove()
            if camera_id in self.camera_poses:
                self.camera_poses[camera_id].remove()
            if camera_id in self.camera_labels:
                self.camera_labels[camera_id].remove()

            frame = self.camera_frames[camera_id]
            image_option = self.camera_image_option[camera_id]
            visible = self.camera_visible[camera_id]

            pose = self._create_camera_pose(camera_id, frame, visible)
            frustum = self._create_camera_frustum(camera_id, frame, image_option, visible)
            self._on_camera_click(frustum, pose)

            label = self._create_camera_label(camera_id, frame, self.camera_labels_visible)

            self.camera_frusta[camera_id] = frustum
            self.camera_poses[camera_id] = pose
            self.camera_labels[camera_id] = label
        self.client.flush()

    def _on_camera_click(self, frustum: viser.CameraFrustumHandle, pose: viser.FrameHandle):
        """
        What happens when the camera frustum is clicked

        Args:
            frustum (viser.CameraFrustumHandle): the camera frustum
            pose (viser.FrameHandle): the camera pose or viser frame (position and wxyz)
        """

        @frustum.on_click
        def _(_) -> None:
            self.client.camera.wxyz = pose.wxyz
            self.client.camera.position = pose.position

    def _create_camera_pose(self, camera_id: str, frame: int, visible: bool) -> viser.FrameHandle:
        """
        Creates the camera pose (viser position and wxyz) at a given frame

        Args:
            camera_id (str): id of the camera sensor
            frame (int): frame within the camera's frame range
            visible (bool): whether to show the frame/pose viser object or not

        Returns:
            viser.FrameHandle: A frame handle (viser position and wxyz)
        """
        camera = self.viewer.loader.get_camera(camera_id)
        camera_pose = self.client.scene.add_frame(
            f"/cameras/{camera_id}/f{frame}",
            wxyz=camera.wxyz_at_frame(frame),
            position=camera.position_at_frame(frame),
            axes_length=0.003,
            axes_radius=0.0005,
            visible=visible,
        )
        return camera_pose

    def _create_camera_frustum(
        self, camera_id: str, frame: int, image_option: str, visible: bool
    ) -> viser.CameraFrustumHandle:
        """
        Creates the camera frustum at the given frame.

        Args:
            camera_id (str): id of the camera sensor
            frame (int): frame within the camera's frame range
            image_option (str): type of data (e.g. semantic, image, etc.)
            visible (bool): whether to show the frustum in the scene or not

        Returns:
            viser.CameraFrustumHandle: a viser camera frustum handle
        """
        camera = self.viewer.loader.get_camera(camera_id)

        result: np.ndarray | None = None
        match image_option:
            case "Image":
                result = self.viewer.loader.get_camera_image(camera_id, frame)
            case "Semantic Segmentation":
                result = self.viewer.loader.get_camera_semantic_image(camera_id, frame)
            case "Semantic Segmentation (Overlay)":
                result = self.viewer.loader.get_camera_semantic_overlay_image(camera_id, frame)
            case "Depth":
                result = self.viewer.loader.get_camera_depth_image(camera_id, frame)
            case "Normals":
                result = self.viewer.loader.get_camera_normals_image(camera_id, frame)
            case _:
                raise ValueError(f"Invalid image option: {image_option}")

        if result is None:
            raise ValueError(f"Camera with id: {camera_id} does not contain {image_option} data")
        image = result

        # Optionally overlay cuboids on raw image view
        if image_option == "Image" and self.camera_overlay_cuboids.get(camera_id, False):
            try:
                image = self._overlay_cuboids_on_image(camera_id, frame, image)
            except Exception:
                # Avoid breaking the viewer due to overlay errors
                pass

        camera_frustum = self.client.scene.add_camera_frustum(
            f"/cameras/{camera_id}/f{frame}/{image_option}_frustum",
            fov=camera.get_fov(),
            aspect=camera.aspect,
            scale=camera.scale,
            image=image,
            visible=visible,
        )
        return camera_frustum

    def _overlay_cuboids_on_image(self, camera_id: str, frame: int, image: np.ndarray) -> np.ndarray:
        """
        Project and draw 3D cuboid edges onto a camera image.

        Args:
            camera_id: Target camera sensor id
            frame: Camera frame index
            image: RGB image array (H, W, 3), uint8

        Returns:
            A copy of the input image with visible cuboid edges and optional labels overlaid
        """
        # Camera model and rolling-shutter poses/timestamps
        camera_sensor = self.viewer.loader.get_camera(camera_id).sensor
        camera_model_params = camera_sensor.get_camera_model_parameters()
        camera_model = CameraModel.from_parameters(camera_model_params, device="cpu", dtype=torch.float32)

        T_world_sensor_start = camera_sensor.get_frame_T_world_sensor(frame, ncore.data.FrameTimepoint.START)
        T_world_sensor_end = camera_sensor.get_frame_T_world_sensor(frame, ncore.data.FrameTimepoint.END)
        timestamp_start_us = camera_sensor.get_frame_timestamp_us(frame, ncore.data.FrameTimepoint.START)
        timestamp_end_us = camera_sensor.get_frame_timestamp_us(frame, ncore.data.FrameTimepoint.END)

        # Source cuboids are gathered per-lidar at frames closest to the target camera frame
        lidar_ids = self.viewer.loader.get_lidar_ids()
        cuboid_source = self.camera_cuboids_source.get(camera_id, ncore.data.LabelSource._member_names_[0])

        # Work on a copy to avoid mutating inputs
        output_image = image.copy()
        image_height, image_width = output_image.shape[:2]
        image_rect = (0, 0, image_width, image_height)

        # Cuboid corner indexing and edges (0..7)
        cuboid_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom ring
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top ring
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # verticals
        ]

        def compute_box_corners_world(box_pose_world: np.ndarray, box_dimensions_lwh: np.ndarray) -> np.ndarray:
            """Return 8 world-space box corners given pose and LWH dimensions."""
            # local unit cube corners scaled by box dimensions
            local_corners = np.array(
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                ],
                dtype=np.float32,
            )
            local_corners *= box_dimensions_lwh.astype(np.float32)
            rotation_world = box_pose_world[:3, :3].astype(np.float32)
            translation_world = box_pose_world[:3, 3].astype(np.float32)
            return local_corners @ rotation_world.T + translation_world

        # Aggregate and render per lidar
        for lidar_id in lidar_ids:
            lidar_frame_idx = self.viewer.loader.get_closest_sensor_frame(frame, camera_id, lidar_id)
            cuboids = self.viewer.loader.get_cuboid_data(lidar_id, lidar_frame_idx, cuboid_source)
            if cuboids is None:
                continue

            for cuboid_idx in range(cuboids.shape[0]):
                box_pose_world = cuboids[cuboid_idx]["pose"][0]
                bbox_array = cuboids[cuboid_idx]["bbox"][0]
                class_label = cuboids[cuboid_idx]["class"][0]
                track_id = cuboids[cuboid_idx]["track_id"][0]

                # Compute 8 world corners and project them to image
                box_dimensions_lwh = np.asarray(bbox_array[3:6], dtype=np.float32)
                corners_world = compute_box_corners_world(box_pose_world, box_dimensions_lwh)

                projection = camera_model.world_points_to_image_points_shutter_pose(
                    torch.from_numpy(corners_world),
                    T_world_sensor_start,
                    T_world_sensor_end,
                    start_timestamp_us=int(timestamp_start_us),
                    end_timestamp_us=int(timestamp_end_us),
                    return_valid_indices=True,
                    return_all_projections=True,
                )

                if projection.valid_indices is None or projection.image_points.shape[0] == 0:
                    continue

                valid_corner_indices = projection.valid_indices.cpu().numpy().astype(np.int32)
                projected_points = projection.image_points.cpu().numpy().astype(np.float32)
                valid_corner_mask = np.zeros(projected_points.shape[0], dtype=bool)
                valid_corner_mask[valid_corner_indices] = True

                # Color by class when available, otherwise white
                line_color = (255, 255, 255)
                try:
                    class_color = self.viewer.loader.get_cuboid_class_color(class_label)
                    line_color = (int(class_color[0]), int(class_color[1]), int(class_color[2]))
                except Exception:
                    pass

                # Draw visible parts of edges (clip segments to image rectangle)
                for corner_a, corner_b in cuboid_edges:
                    if not (valid_corner_mask[corner_a] or valid_corner_mask[corner_b]):
                        continue
                    p1 = (int(round(projected_points[corner_a, 0])), int(round(projected_points[corner_a, 1])))
                    p2 = (int(round(projected_points[corner_b, 0])), int(round(projected_points[corner_b, 1])))
                    ok, clipped_p1, clipped_p2 = cv2.clipLine(image_rect, p1, p2)
                    if ok:
                        cv2.line(output_image, clipped_p1, clipped_p2, color=line_color, thickness=2)

                # Optional label: prefer projected box center; fallback to a visible corner
                if self.camera_show_cuboid_labels.get(camera_id, True):
                    # Try projecting the cuboid center (translation component of the pose)
                    label_px: int | None = None
                    label_py: int | None = None
                    center_world = box_pose_world[:3, 3].astype(np.float32)
                    center_proj = camera_model.world_points_to_image_points_shutter_pose(
                        torch.from_numpy(center_world[None, :]),
                        T_world_sensor_start,
                        T_world_sensor_end,
                        start_timestamp_us=int(timestamp_start_us),
                        end_timestamp_us=int(timestamp_end_us),
                        return_valid_indices=True,
                        return_all_projections=True,
                    )
                    if center_proj.valid_indices is not None and center_proj.valid_indices.numel() > 0:
                        cpt = center_proj.image_points.cpu().numpy().astype(np.float32)[0]
                        label_px = int(round(cpt[0]))
                        label_py = int(round(cpt[1]))
                    else:
                        # Fallback: use a visible corner (prefer front-top-right)
                        preferred_corner = 6
                        visible_corners = np.where(valid_corner_mask)[0]
                        if preferred_corner in visible_corners:
                            label_px = int(round(projected_points[preferred_corner, 0]))
                            label_py = int(round(projected_points[preferred_corner, 1]))
                        elif visible_corners.size > 0:
                            idx = int(visible_corners[0])
                            label_px = int(round(projected_points[idx, 0]))
                            label_py = int(round(projected_points[idx, 1]))

                    if (
                        label_px is not None
                        and label_py is not None
                        and 0 <= label_px < image_width
                        and 0 <= label_py < image_height
                    ):
                        text = f"{track_id}[{class_label}]"
                        # Simple contrasting outline for readability
                        cv2.putText(
                            output_image,
                            text,
                            (label_px + 4, label_py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            thickness=3,
                            lineType=cv2.LINE_AA,
                        )
                        cv2.putText(
                            output_image,
                            text,
                            (label_px + 4, label_py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            line_color,
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

        return output_image

    def _create_camera_label(self, camera_id: str, frame: int, visible: bool) -> viser.LabelHandle:
        """
        Creates a label with the camera name at the position of the camera at the given frame

        Args:
            camera_id (str): id of the camera sensor
            frame (int): frame within the camera's frame range
            visible (bool): whether to show the label in the scene or not

        Returns:
            viser.LabelHandle: a viser label handle
        """
        camera = self.viewer.loader.get_camera(camera_id)
        camera_label = self.client.scene.add_label(
            f"/labels/{camera_id}",
            camera_id,
            wxyz=camera.wxyz_at_frame(frame),
            position=camera.position_at_frame(frame),
            visible=visible,
        )
        return camera_label
