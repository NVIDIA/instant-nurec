# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Polyscope implementation of the visual debugger interface."""

import logging

from typing import Dict, Optional, Tuple

import numpy as np
import polyscope as ps

from nre.utils.misc import singleton_register
from nre.visualdebugger.interface import VisualDebugger


logger = logging.getLogger(__name__)


@singleton_register(VisualDebugger)
class PolyscopeVisualDebugger(VisualDebugger):
    """Polyscope implementation of the VisualDebugger interface.

    This class wraps Polyscope functionality to conform to the VisualDebugger interface.
    """

    def __init__(self):
        """Initialize the Polyscope visual debugger."""
        self._initialized = False

    def _ensure_initialized(self):
        """Initialize Polyscope if not already done."""
        if not self._initialized:
            logger.debug("Initializing PolyscopeVisualDebugger")
            ps.set_verbosity(0)
            ps.set_allow_headless_backends(True)
            ps.init()
            self._initialized = True

    def show(self) -> None:
        """Display the Polyscope visualization window."""
        self._ensure_initialized()
        ps.show()

    def clear(self) -> None:
        """Clear all objects from the visualization."""
        self._ensure_initialized()
        ps.remove_all_structures()

    def update(self) -> None:
        """Update the visualization when working with non blocking visualizations."""
        self._ensure_initialized()
        ps.frame_tick()

    def set_properties(
        self,
        program_name: Optional[str] = None,
        up: Optional[str] = None,
        front: Optional[str] = None,
        navigation_style: Optional[str] = None,
        ground_plane_mode: Optional[str] = None,
        length_scale: Optional[float] = None,
        automatically_compute_scene_extents: Optional[bool] = None,
        window_size: Optional[Tuple[int, int]] = None,
        view_projection_mode: Optional[str] = None,
        window_resizable: Optional[bool] = None,
        json_str: Optional[str] = None,
    ) -> None:
        """Set global properties of the visualization

        Args:
            program_name: Optional name of the program
            up: Optional up direction
            front: Optional front direction
            navigation_style: Optional navigation style
        """
        self._ensure_initialized()
        if program_name is not None:
            ps.set_program_name(program_name)
        if up is not None:
            ps.set_up_dir(up)
        if front is not None:
            ps.set_front_dir(front)
        if navigation_style is not None:
            ps.set_navigation_style(navigation_style)
        if ground_plane_mode is not None:
            ps.set_ground_plane_mode(ground_plane_mode)
        if length_scale is not None:
            ps.set_length_scale(length_scale)
        if automatically_compute_scene_extents is not None:
            ps.set_automatically_compute_scene_extents(automatically_compute_scene_extents)
        if window_resizable is not None:
            ps.set_window_resizable(window_resizable)
        if window_size is not None:
            ps.set_window_size(window_size[0], window_size[1])
        if view_projection_mode is not None:
            ps.set_view_projection_mode(view_projection_mode)
        if json_str is not None:
            ps.set_view_from_json(json_str)

    def _add_to_group(self, obj: ps.Structure, group: Optional[str] = None) -> None:
        """Add an object to a group."""
        if group is not None:
            if ps.has_group(group):
                ps_group = ps.get_group(group)
            else:
                ps_group = ps.create_group(group)
            if not obj.is_in_group(ps_group):
                obj.add_to_group(ps_group)
        pass

    def _color_to_float(self, color: np.ndarray) -> np.ndarray:
        """Convert a color to a float array."""
        if color.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            return color.astype(np.float32) / float(np.iinfo(color.dtype).max)
        return color

    def add_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        enabled: Optional[bool] = None,
        color: Optional[Tuple[float, float, float]] = None,
        radius: Optional[float] = None,
        transparency: Optional[float] = None,
        colors_quantities: Optional[Dict[str, np.ndarray]] = None,
        point_render_mode: Optional[str] = None,
        group: Optional[str] = None,
    ) -> None:
        """Add a point cloud to the visualization.

        Args:
            name: Unique identifier for this point cloud
            points: Array of 3D points with shape (N, 3)
            enabled: Whether the point cloud is enabled
            color: Optional array of RGB colors with shape (N, 3)
            radius: Optional radius of the point cloud
            transparency: Optional transparency of the point cloud
            colors_quantities: Optional dictionary of colors and quantities with shape (N, C)
            point_render_mode: Optional render mode of the point cloud
            group: Name of the group to add the point cloud to

        Returns:
            ID of the registered point cloud
        """
        self._ensure_initialized()

        if ps.has_point_cloud(name):
            point_cloud = ps.get_point_cloud(name)
            point_cloud.update_point_positions(points)

        else:
            # Register point cloud with Polyscope
            point_cloud = ps.register_point_cloud(name, points)

        if enabled is not None:
            point_cloud.set_enabled(enabled)
        if color is not None:
            point_cloud.set_color(color)
        if radius is not None:
            point_cloud.set_radius(radius)
        if transparency is not None:
            point_cloud.set_transparency(transparency)
        if point_render_mode is not None:
            point_cloud.set_point_render_mode(point_render_mode)
        if colors_quantities is not None:
            colors_quantities_enabled = True
            for quantity_name, quantity_values in colors_quantities.items():
                point_cloud.add_color_quantity(
                    quantity_name, self._color_to_float(quantity_values), enabled=colors_quantities_enabled
                )
                colors_quantities_enabled = False
        # Add to group if specified
        self._add_to_group(point_cloud, group)

    def add_surface_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        enabled: Optional[bool] = None,
        color: Optional[Tuple[float, float, float]] = None,
        edge_width: Optional[float] = None,
        edge_color: Optional[Tuple[float, float, float]] = None,
        transparency: Optional[float] = None,
        group: Optional[str] = None,
    ) -> None:
        """Add a triangle mesh to the visualization. If the mesh already exists, it will be updated with the new vertices.

        Args:
            name: Unique identifier for this mesh
            vertices: Array of 3D vertices with shape (V, 3)
            faces: Array of triangle indices with shape (F, 3)
            enabled: Whether the mesh is enabled
            color: Optional array of RGB colors with shape (3,)
            edge_width: Optional width of the edges
            edge_color: Optional color of the edges
            transparency: Optional transparency of the mesh
            group: Name of the group to add the mesh to

        Returns:
            ID of the registered mesh
        """
        self._ensure_initialized()

        # Register mesh with Polyscope
        if ps.has_surface_mesh(name):
            mesh = ps.get_surface_mesh(name)
            mesh.update_vertex_positions(vertices)
        else:
            mesh = ps.register_surface_mesh(name, vertices, faces)

        if enabled is not None:
            mesh.set_enabled(enabled)
        if edge_width is not None:
            mesh.set_edge_width(edge_width)
        if edge_color is not None:
            mesh.set_edge_color(edge_color)
        if transparency is not None:
            mesh.set_transparency(transparency)
        if color is not None:
            mesh.set_color(color)

        # Add to group if specified
        self._add_to_group(mesh, group)

    def add_curve_network(
        self,
        name: str,
        corners: np.ndarray,
        edges: VisualDebugger.CurveNetworkEdge,
        enabled: Optional[bool] = None,
        color: Optional[Tuple[float, float, float]] = None,
        radius: Optional[float] = None,
        transparency: Optional[float] = None,
        group: Optional[str] = None,
    ) -> None:
        """Add line segments to the visualization or update an existing one.

        Args:
            name: Unique identifier for these lines
            corners: Array of corners with shape (N, 3)
            edges: Array of edges with shape (N, 3)
            enabled: Whether the lines are enabled
            colors: Optional array of RGB colors with shape (N, 3)
            radius: Optional radius of the lines
            transparency: Optional transparency of the lines
            group: Name of the group to add the lines to

        Returns:
            ID of the registered lines
        """
        self._ensure_initialized()

        if ps.has_curve_network(name):
            lines = ps.get_curve_network(name)
            lines.update_node_positions(corners)
        else:
            lines = ps.register_curve_network(name, corners, edges)

        if enabled is not None:
            lines.set_enabled(enabled)
        if color is not None:
            lines.set_color(color)
        if radius is not None:
            lines.set_radius(radius)
        if transparency is not None:
            lines.set_transparency(transparency)

        self._add_to_group(lines, group)

    def set_camera_extrinsics(self, extrinsics: VisualDebugger.RootLookUp | np.ndarray) -> None:
        """Set the camera extrinsics. Can be provided as either as a RootLookDir dataclass or a 4x4 matrix."""
        self._ensure_initialized()
        if isinstance(extrinsics, VisualDebugger.RootLookUp):
            extrinsics = ps.CameraExtrinsics(
                root=extrinsics.root, look_dir=extrinsics.look_dir, up_dir=extrinsics.up_dir
            )
        else:
            extrinsics = ps.CameraExtrinsics(mat=extrinsics)
        intrinsics = ps.get_view_camera_parameters().get_intrinsics()
        ps.set_view_camera_parameters(ps.CameraParameters(intrinsics=intrinsics, extrinsics=extrinsics))

    def set_camera_intrinsics(
        self,
        fov_vertical_deg: float | None = None,
        fov_horizontal_deg: float | None = None,
        aspect: float | None = None,
    ) -> None:
        """Set the camera intrinsics."""
        self._ensure_initialized()
        extrinsics = ps.get_view_camera_parameters().get_extrinsics()
        ps.set_view_camera_parameters(
            ps.CameraParameters(
                intrinsics=ps.CameraIntrinsics(
                    fov_vertical_deg=fov_vertical_deg, fov_horizontal_deg=fov_horizontal_deg, aspect=aspect
                ),
                extrinsics=extrinsics,
            )
        )

    def screenshot(self, filename: str) -> None:
        """Take a screenshot of the visualization."""
        self._ensure_initialized()
        ps.screenshot(filename)

    def screenshot_to_buffer(self) -> np.ndarray:
        """Take a screenshot of the visualization and return it as a numpy array."""
        self._ensure_initialized()
        return ps.screenshot_to_buffer()
