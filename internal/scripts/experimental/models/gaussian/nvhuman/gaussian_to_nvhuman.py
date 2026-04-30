#!/usr/bin/env python3
import logging

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from scipy.spatial import cKDTree


logger = logging.getLogger(__name__)

# Initialize GenMO PROJ_ROOT before importing GenMO modules
import hmr4d


# Path constants (relative to hmr4d.PROJ_ROOT)
NVHUMAN_TEMPLATE_RELPATH = "inputs/nvhuman_data/nvHuman_shape_TPose.npz"

from internal.scripts.experimental.models.gaussian.genmo import genmo_init  # noqa: F401
from internal.scripts.experimental.models.gaussian.nvhuman.gaussian_nvhuman_layer import GaussianNVHumanLayer
from internal.scripts.experimental.models.gaussian.nvhuman.tools.utils import move_to_start_point_face_z
from nre.models.gaussians.utils import PLYGaussianLoader
from nre.utils.io.ply import save_ply


class GaussianToNVHumanConverter:
    def __init__(self):
        template_path = hmr4d.PROJ_ROOT / NVHUMAN_TEMPLATE_RELPATH
        logger.info(f"Auto-detected template: {template_path}")

        self.template_path = Path(template_path)
        self.template_nvhuman = None
        self._load_template()

    def _load_template(self) -> None:
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {self.template_path}")

        logger.info(f"Loading NVHuman template from: {self.template_path}")
        # Use GaussianNVHumanLayer which extends genmo's base NVHumanLayer
        self.template_nvhuman = GaussianNVHumanLayer(
            model_path=str(self.template_path), rest_type="T", dtype=torch.float32, batch_size=10000
        )
        vertices = self.template_nvhuman.v_template.cpu().numpy()
        faces = self.template_nvhuman.faces_tensor.cpu().numpy()
        logger.info(f"Loaded NVHuman template: {len(vertices)} vertices, {len(faces)} faces")
        logger.debug(f"Skeleton: {self.template_nvhuman.num_joints} joints")

    def extract_pose_from_file(self, pose_file: Path, frame_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Extract SMPL pose parameters from GenMO HMR4D results file.

        Args:
            pose_file: Path to the pose file (hmr4d_results.pt)
            frame_idx: Frame index to extract (default: 0 for reference T-pose)

        Returns:
            Dictionary with SMPL parameters for the specified frame
        """
        logger.debug(f"Loading pose from: {pose_file}, frame: {frame_idx}")
        pose_data = torch.load(pose_file, map_location="cpu")

        if "smpl_params_incam" not in pose_data:
            raise ValueError("Could not find 'smpl_params_incam' in pose file")

        pose_sequence = pose_data["smpl_params_incam"]

        # Extract single frame parameters
        reference_pose = {}
        for key, value in pose_sequence.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 1:
                    # Take the specified frame
                    reference_pose[key] = value[frame_idx : frame_idx + 1]
                else:
                    # Scalar or 1D tensor - use as is
                    reference_pose[key] = value

        # Print what we extracted
        logger.debug("Extracted reference pose parameters:")
        for key, value in reference_pose.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"  {key}: {value.shape}")

        return reference_pose

    def _scale_gaussians_to_template_height(
        self, gaussians: np.ndarray, template_vertices: np.ndarray, middle_percent: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        logger.info(
            f"Filtering Gaussians by height position (middle {middle_percent} of range), scaling to template..."
        )

        template_y = template_vertices[:, 1]
        template_height = np.max(template_y) - np.min(template_y)
        template_top_y = np.max(template_y)

        gaussian_y = gaussians[:, 1]
        gaussian_y_min = np.min(gaussian_y)
        gaussian_y_max = np.max(gaussian_y)
        gaussian_full_height = gaussian_y_max - gaussian_y_min

        height_margin = (1.0 - middle_percent) / 2.0
        gaussian_y_lower = gaussian_y_min + height_margin * gaussian_full_height
        gaussian_y_upper = gaussian_y_max - height_margin * gaussian_full_height

        gaussian_filtered_mask = (gaussian_y >= gaussian_y_lower) & (gaussian_y <= gaussian_y_upper)
        gaussians_kept = gaussians[gaussian_filtered_mask]
        gaussian_filtered_y = gaussians_kept[:, 1]
        gaussian_height = np.max(gaussian_filtered_y) - np.min(gaussian_filtered_y)

        logger.debug(f"Removed {len(gaussians) - len(gaussians_kept)} Gaussian points outside height range")
        logger.info(f"Remaining Gaussians: {len(gaussians_kept)}/{len(gaussians)} points, height={gaussian_height:.3f}")

        if gaussian_height > 1e-6:
            scale_factor = template_height / gaussian_height
        else:
            raise ValueError("Gaussian height is too small to scale")

        logger.debug(f"Template: {len(template_y)} total points, height={template_height:.3f}")
        logger.info(f"Scale factor: {scale_factor:.3f}")

        gaussian_center = np.mean(gaussians_kept, axis=0)
        scaled_gaussians = (gaussians_kept - gaussian_center) * scale_factor + gaussian_center

        scaled_top_y = np.max(scaled_gaussians[:, 1])
        y_offset = template_top_y - scaled_top_y
        scaled_gaussians[:, 1] += y_offset

        logger.debug(f"Applied Y offset: {y_offset:.3f}")
        logger.info(f"Scaled and aligned {len(scaled_gaussians)} Gaussians")

        return scaled_gaussians, gaussian_filtered_mask, scale_factor

    def _get_posed_template_vertices(
        self, reference_pose_params: Dict[str, torch.Tensor], timestamp: int = 0
    ) -> torch.Tensor:
        """Apply reference pose parameters to NVHuman template to get posed vertices.

        Args:
            reference_pose_params: Pose parameters from HMR4D (SMPL parameters)
            timestamp: Frame index to extract pose from (default 0 for initial/reference pose)
        """
        logger.debug(f"Applying pose parameters at timestamp {timestamp} to NVHuman template...")

        try:
            device = next(self.template_nvhuman.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        pose_params_t = {}
        for key, value in reference_pose_params.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 1 and value.shape[0] > timestamp:
                    pose_params_t[key] = value[timestamp : timestamp + 1].to(device)
                elif value.dim() > 1:
                    pose_params_t[key] = value[0:1].to(device)
                else:
                    pose_params_t[key] = value.to(device)
            else:
                pose_params_t[key] = value

        with torch.no_grad():
            output = self.template_nvhuman(**pose_params_t)
            posed_verts = output["vertices"][0].cpu().numpy()
            posed_joints = output["joints"][0].cpu().numpy() if "joints" in output else None
            posed_vertices = torch.from_numpy(posed_verts).unsqueeze(0)

        return posed_vertices

    def _compute_gaussian_face_binding(
        self, gaussian_positions: np.ndarray, template_vertices: np.ndarray
    ) -> np.ndarray:
        """Bind each Gaussian to the nearest face by computing distance to face centers.

        Args:
            gaussian_positions: [N, 3] - Gaussian positions
            template_vertices: [V, 3] - Template mesh vertices

        Returns:
            gaussian_prim_ind: [N] - Face index for each Gaussian
        """
        logger.debug(f"Computing primitive (face) binding for {len(gaussian_positions)} Gaussians...")
        template_faces = self.template_nvhuman.faces_tensor.cpu().numpy()

        # Compute face centers
        face_vertices = template_vertices[template_faces]  # [F, 3, 3]
        face_centers = face_vertices.mean(axis=1)  # [F, 3]

        # Build KDTree on face centers and find nearest face for each Gaussian
        tree = cKDTree(face_centers)
        distances, face_indices = tree.query(gaussian_positions)

        gaussian_prim_ind = face_indices.astype(np.int64)
        logger.debug(f"Unique faces used: {len(np.unique(gaussian_prim_ind))}/{len(template_faces)}")
        return gaussian_prim_ind

    def load_gaussian_ply(self, ply_path: Path) -> PLYGaussianLoader:
        logger.info(f"Loading Gaussian PLY from: {ply_path}")
        loaded_ply = PLYGaussianLoader(ply_path)
        logger.info(f"Loaded {loaded_ply.positions.shape[0]} Gaussians")
        return loaded_ply

    def create_nvhuman_model(
        self,
        gaussian_ply: PLYGaussianLoader,
        output_dir: Path,
        ply_filename: str,
        reference_pose_params: Optional[Dict[str, torch.Tensor]] = None,
        height_scale_percent: float = 0.8,
    ) -> Dict[str, Any]:
        gaussian_positions = gaussian_ply.positions.cpu().numpy()
        gaussian_scales = gaussian_ply.scales.cpu().numpy()
        gaussian_rotations = gaussian_ply.rotations.cpu().numpy()
        gaussian_opacities = gaussian_ply.densities.cpu().numpy()

        gaussian_shs_albedo = gaussian_ply.features_albedo.cpu().numpy()
        if gaussian_ply.features_specular is not None:
            gaussian_shs_specular = gaussian_ply.features_specular.cpu().numpy()
            gaussian_shs = np.concatenate([gaussian_shs_albedo, gaussian_shs_specular], axis=1)
        else:
            gaussian_shs = gaussian_shs_albedo

        logger.info(f"Creating NVHuman data from {len(gaussian_positions)} Gaussian points")

        logger.debug(f"Gaussian first 10 SHs: {gaussian_shs[:10]}")

        # Get T-pose mesh vertices (bind pose)
        tpose_vertices = self.template_nvhuman.v_template.cpu().numpy()

        # Get reference pose mesh vertices (initial frame from HMR4D at timestamp 0)
        logger.info("Creating reference pose mesh from HMR4D parameters at timestamp 0")
        reference_pose_vertices = self._get_posed_template_vertices(reference_pose_params, timestamp=0)
        reference_pose_vertices = move_to_start_point_face_z(reference_pose_vertices, self.template_nvhuman.J_regressor)
        reference_pose_vertices_np = reference_pose_vertices[0].cpu().numpy()
        logger.debug(f"Reference pose mesh: {len(reference_pose_vertices_np)} vertices")

        # Align and filter Gaussians to reference pose mesh
        aligned_gaussian_positions, filtered_indices, scale_factor = self._scale_gaussians_to_template_height(
            gaussian_positions, reference_pose_vertices_np, height_scale_percent
        )
        logger.debug(f"Scale factor: {scale_factor}")
        # Apply scale factor to Gaussian scales (positions were scaled, so scales must be too)
        # Scale factor should always be positive from _scale_gaussians_to_template_height
        assert scale_factor > 0, f"Unexpected negative scale factor: {scale_factor}"
        aligned_gaussian_scales = gaussian_scales[filtered_indices] + np.log(scale_factor)  # log scale factor
        aligned_gaussian_rotations = gaussian_rotations[filtered_indices]
        aligned_gaussian_opacities = gaussian_opacities[filtered_indices]
        aligned_gaussian_shs = gaussian_shs[filtered_indices]

        logger.info(f"Filtered {len(aligned_gaussian_positions)} Gaussians")

        assert len(aligned_gaussian_scales) == len(aligned_gaussian_positions)
        assert len(aligned_gaussian_rotations) == len(aligned_gaussian_positions)
        assert len(aligned_gaussian_opacities) == len(aligned_gaussian_positions)
        assert len(aligned_gaussian_shs) == len(aligned_gaussian_positions)

        # Debug: Save reference pose comparison (Gaussians + mesh)
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_combined_vertices = np.vstack([aligned_gaussian_positions, reference_pose_vertices_np])
        n_gaussians = len(aligned_gaussian_positions)
        n_vertices = len(reference_pose_vertices_np)
        gaussian_colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (n_gaussians, 1))  # Red: Gaussians
        mesh_colors = np.tile(np.array([0, 255, 0], dtype=np.uint8), (n_vertices, 1))  # Green: Mesh
        reference_combined_colors = np.vstack([gaussian_colors, mesh_colors])

        reference_comparison_path = output_dir / f"{ply_filename}_reference_comparison.ply"
        save_ply(
            filename=str(reference_comparison_path),
            vertices=reference_combined_vertices.astype(np.float32),
            colors=reference_combined_colors,
        )

        # Bind Gaussians to mesh faces (in reference pose where they align naturally)
        logger.info("Binding Gaussians to mesh faces...")
        gaussian_prim_ind = self._compute_gaussian_face_binding(aligned_gaussian_positions, reference_pose_vertices_np)

        # Apply reverse deformation: reference pose global → T-pose local coordinates
        # This uses NVHuman's inverse deformation (proper inverse of _deform_gaussians)
        logger.info("Applying reverse deformation: reference pose → T-pose (positions + rotations)...")
        result = self.template_nvhuman.reverse_deform_gaussians(
            gaussian_positions_global=torch.from_numpy(aligned_gaussian_positions).float(),
            gaussian_rotations_global=torch.from_numpy(aligned_gaussian_rotations).float(),
            gaussian_prim_ind=torch.from_numpy(gaussian_prim_ind).long(),
            reference_vertices=torch.from_numpy(reference_pose_vertices_np).float(),
            tpose_vertices=torch.from_numpy(tpose_vertices).float(),
        )

        local_gaussian_positions = result["local_positions"].cpu().numpy()  # [N, 3] - to store in .npz
        local_gaussian_rotations = result["local_rotations"].cpu().numpy()  # [N, 4] - to store in .npz (WXYZ)
        tpose_gaussian_positions = result["tpose_global_positions"].cpu().numpy()  # [N, 3] - for debug viz
        tpose_gaussian_rotations = result["tpose_global_rotations"].cpu().numpy()  # [N, 4] - for debug viz

        # Debug: Save T-pose Gaussians for visualization
        tpose_gaussian_colors = np.tile(np.array([0, 255, 0], dtype=np.uint8), (len(tpose_gaussian_positions), 1))
        tpose_gaussian_path = output_dir / f"{ply_filename}_tpose_gaussians.ply"
        save_ply(
            filename=str(tpose_gaussian_path),
            vertices=tpose_gaussian_positions.astype(np.float32),
            colors=tpose_gaussian_colors,
        )

        nvhuman_data = {
            "v_template": tpose_vertices.astype(np.float32),
            "rest_tpose_verts": tpose_vertices.astype(np.float32),
            "faces": self.template_nvhuman.faces_tensor.cpu().numpy().astype(np.int64),
            "J_regressor": self.template_nvhuman.J_regressor.cpu().numpy().astype(np.float32),
            "parents": self.template_nvhuman.parents.cpu().numpy().astype(np.int64),
            "lbs_weights": self.template_nvhuman.lbs_weights.cpu().numpy().astype(np.float32),
            "rig_joint_names": self.template_nvhuman.rig_joint_names,
            "num_joints": self.template_nvhuman.num_joints,
            "shapedirs": self.template_nvhuman.shapedirs.cpu().numpy().astype(np.float32),
            "gaussian_positions": local_gaussian_positions.astype(np.float32),  # Stored in local face coordinates
            "gaussian_scales": aligned_gaussian_scales.astype(np.float32),  # Stored in log space
            "gaussian_rotations": local_gaussian_rotations.astype(
                np.float32
            ),  # Stored in local face coordinates (WXYZ)
            "gaussian_opacities": aligned_gaussian_opacities.astype(np.float32),
            "gaussian_shs": aligned_gaussian_shs.astype(np.float32),
            "num_gaussians": np.int64(len(local_gaussian_positions)),
            "gaussian_prim_ind": gaussian_prim_ind.astype(np.int64),
        }

        # Debug visualization: T-pose Gaussians + T-pose template mesh
        combined_vertices = np.vstack([tpose_gaussian_positions, tpose_vertices])

        n_gaussians = len(tpose_gaussian_positions)
        n_template = len(tpose_vertices)
        gaussian_colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (n_gaussians, 1))  # Red: Gaussians
        template_colors = np.tile(np.array([0, 0, 255], dtype=np.uint8), (n_template, 1))  # Blue: T-pose mesh
        combined_colors = np.vstack([gaussian_colors, template_colors])

        comparison_debug_path = output_dir / f"{ply_filename}_tpose_comparison.ply"
        save_ply(
            filename=str(comparison_debug_path),
            vertices=combined_vertices.astype(np.float32),
            colors=combined_colors,
        )

        return nvhuman_data

    def save_nvhuman_model(self, nvhuman_data: Dict[str, Any], output_path: Path) -> None:
        logger.info(f"Saving NVHuman data to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **nvhuman_data)
        logger.info(f"Saved NVHuman data with {len(nvhuman_data)} fields")

    def load_reference_pose(self, hmr4d_path: Path) -> Dict[str, torch.Tensor]:
        """Load reference pose (initial pose at timestamp 0) from HMR4D motion capture data.

        The reference pose is used to align Gaussians to the initial frame of the animation.
        """
        logger.debug(f"Loading reference pose from HMR4D data: {hmr4d_path}")
        if not hmr4d_path.exists():
            raise FileNotFoundError(f"HMR4D results file not found: {hmr4d_path}")

        hmr4d_data = torch.load(hmr4d_path, map_location="cpu")
        if "smpl_params_incam" in hmr4d_data:
            reference_pose_params = hmr4d_data["smpl_params_incam"]
            for key, value in reference_pose_params.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"  {key}: {value.shape}")
            return reference_pose_params
        else:
            raise ValueError("Could not find 'smpl_params_incam' in HMR4D results file")
