# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""VisualDebugger interface for rendering 3D geometry and other visualizations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union, cast

import numpy as np

from nre.utils.misc import singleton_get_instance


class VisualDebugger(ABC):
    """Abstract interface for visual debuggers.

    This class defines the common operations that any visual debugger should support,
    including adding geometric primitives for visualization and control operations.
    """

    @abstractmethod
    def show(self) -> None:
        """Display the visualization window."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all objects from the visualization."""
        pass

    @abstractmethod
    def update(self) -> None:
        """Update the visualization when working with non blocking visualizations."""
        pass

    @abstractmethod
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
        """Set the direction of the camera."""
        pass

    @abstractmethod
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
        """Add a point cloud to the visualization or update an existing one.

        Args:
            name: Unique identifier for this point cloud
            points: Array of 3D points with shape (N, 3)
            enabled: Whether the point cloud is enabled
            color: Optional color of the point cloud
            radius: Optional radius of the point cloud
            transparency: Optional transparency of the point cloud
            colors_quantities: Optional dictionary of colors and quantities with shape (N, C)
            point_render_mode: Optional render mode of the point cloud
            group: Name of the group to add the point cloud to

        Returns:
            ID of the registered point cloud
        """
        pass

    @abstractmethod
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
        """Add a triangle mesh to the visualization or update an existing one.

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
        pass

    CurveNetworkEdge = Union[np.ndarray, Literal["line", "loop"]]

    @abstractmethod
    def add_curve_network(
        self,
        name: str,
        corners: np.ndarray,
        edges: CurveNetworkEdge,
        enabled: Optional[bool] = None,
        color: Optional[Tuple[float, float, float]] = None,
        radius: Optional[float] = None,
        transparency: Optional[float] = None,
        group: Optional[str] = None,
    ) -> None:
        """Add line segments to the visualization.

        Args:
            name: Unique identifier for these lines
            corners: Array of corner points with shape (N, 3)
            edges: Array of edge indices with shape (N, 2)
            enabled: Whether the lines are enabled
            color: Optional color of the lines
            radius: Optional radius of the lines
            transparency: Optional transparency of the lines
            group: Name of the group to add the lines to

        Returns:
            ID of the registered lines
        """
        pass

    @dataclass
    class RootLookUp:
        root: Tuple[float, float, float]
        look_dir: Tuple[float, float, float]
        up_dir: Tuple[float, float, float]

    @abstractmethod
    def set_camera_extrinsics(self, extrinsics: RootLookUp | np.ndarray) -> None:
        """Set the camera extrinsics. Can be provided as either as a RootLookDir dataclass or a 4x4 matrix."""
        pass

    @abstractmethod
    def set_camera_intrinsics(
        self,
        fov_vertical_deg: float | None = None,
        fov_horizontal_deg: float | None = None,
        aspect: float | None = None,
    ) -> None:
        """Set the camera intrinsics."""
        pass

    @abstractmethod
    def screenshot(self, filename: str) -> None:
        """Take a screenshot of the current visualization."""
        pass

    @abstractmethod
    def screenshot_to_buffer(self) -> np.ndarray:
        """Take a screenshot of the current visualization and return it as a numpy array."""
        pass


def get_visualdebugger() -> VisualDebugger:
    return cast(VisualDebugger, singleton_get_instance(VisualDebugger))
