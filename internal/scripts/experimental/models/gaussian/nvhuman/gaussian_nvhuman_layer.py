import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

from hmr4d.utils.body_model.nvhuman_layer import NVHumanLayer

from internal.scripts.experimental.models.gaussian.genmo import genmo_init  # noqa: F401
from nre.utils.geometry import quat_mult_xyzw, so3_matrix_to_quat


def to_tensor(array, dtype=torch.float32):
    """Convert numpy array to torch tensor."""
    if isinstance(array, torch.Tensor):
        return array
    return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    """Convert array to numpy, handling scipy sparse matrices."""
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class GaussianNVHumanLayer(NVHumanLayer):
    """
    Extended NVHumanLayer with 3D Gaussian Splatting support.

    Inherits from genmo's NVHumanLayer and adds Gaussian deformation capabilities.

    Args:
        model_path: Path to .npz file with NVHuman parameters and optionally Gaussian data
        rest_type: Rest pose type ("T" for T-pose)
        dtype: Data type for tensors
        batch_size: Batch size for operations (inherited from base class)

    Additional NPZ Keys (beyond base NVHumanLayer):
        gaussian_positions: [N, 3] - Gaussian centers in local face coordinates
        gaussian_scales: [N, 3] - Gaussian scale parameters (log space)
        gaussian_rotations: [N, 4] - Gaussian rotation quaternions (WXYZ)
        gaussian_opacities: [N, 1] - Gaussian opacity values
        gaussian_shs: [N, 48] - Spherical harmonics coefficients (16 bands * 3 channels)
        gaussian_prim_ind: [N] - Face index for each Gaussian (binding)
        num_gaussians: int - Total number of Gaussians
    """

    def __init__(self, model_path, rest_type="T", dtype=torch.float32, batch_size=10000):
        # Initialize base NVHumanLayer
        super().__init__(model_path, rest_type, dtype)

        # Store batch size for Gaussian operations
        self.batch_size = batch_size

        # Load Gaussian data if available
        npz_data = np.load(model_path)
        self.data = dict(npz_data)

        if "gaussian_positions" in self.data:
            # Register Gaussian parameters as buffers
            self.register_buffer("gaussian_positions", to_tensor(to_np(self.data["gaussian_positions"]), dtype=dtype))
            self.register_buffer("gaussian_scales", to_tensor(to_np(self.data["gaussian_scales"]), dtype=dtype))
            self.register_buffer("gaussian_rotations", to_tensor(to_np(self.data["gaussian_rotations"]), dtype=dtype))
            self.register_buffer("gaussian_opacities", to_tensor(to_np(self.data["gaussian_opacities"]), dtype=dtype))
            self.register_buffer("gaussian_shs", to_tensor(to_np(self.data["gaussian_shs"]), dtype=dtype))
            self.register_buffer(
                "gaussian_prim_ind", to_tensor(to_np(self.data["gaussian_prim_ind"]), dtype=torch.long)
            )

            self.num_gaussians = int(self.data["num_gaussians"])
            self.has_gaussians = True
            self.has_face_binding = True
            logger.info(f"Loaded {self.num_gaussians} Gaussians with face binding")
        else:
            self.has_gaussians = False
            self.has_face_binding = False
            logger.info("No Gaussian data found - mesh-only mode")

    def get_face_properties(self, verts, faces):
        """
        Compute face properties: center, orientation matrix, and scaling.
        Similar to Gaussian Avatar Haar implementation.

        Args:
            verts: [B, V, 3] - Mesh vertices
            faces: [F, 3] - Face indices

        Returns:
            face_center: [B, F, 3] - Center of each face
            face_orien_mat: [B, F, 3, 3] - Orientation matrix for each face
            face_scaling: [B, F, 1] - Scaling factor for each face
        """
        batch_size = verts.shape[0]
        device = verts.device

        # Get face vertices: [B, F, 3, 3]
        face_vertices = verts[:, faces]  # [B, F, 3 vertices, 3 coords]

        # Compute face centers: [B, F, 3]
        face_center = face_vertices.mean(dim=2)

        # Compute face normals and orientation
        v0, v1, v2 = face_vertices[:, :, 0], face_vertices[:, :, 1], face_vertices[:, :, 2]

        # Edge vectors
        e1 = v1 - v0  # [B, F, 3]
        e2 = v2 - v0  # [B, F, 3]

        # Normal vector
        normal = torch.cross(e1, e2, dim=2)  # [B, F, 3]
        normal = F.normalize(normal, p=2, dim=2)

        # Compute face scaling as the area of the triangle
        # Area = 0.5 * ||e1 x e2||
        face_area = torch.norm(torch.cross(e1, e2, dim=2), dim=2, keepdim=True) * 0.5  # [B, F, 1]

        # Reference area from template (compute once)
        if not hasattr(self, "_template_face_areas"):
            template_face_verts = self.v_template[faces]  # [F, 3, 3]
            t_v0, t_v1, t_v2 = template_face_verts[:, 0], template_face_verts[:, 1], template_face_verts[:, 2]
            t_e1 = t_v1 - t_v0
            t_e2 = t_v2 - t_v0
            template_areas = torch.norm(torch.cross(t_e1, t_e2, dim=1), dim=1, keepdim=True) * 0.5  # [F, 1]
            self._template_face_areas = template_areas

        # Scaling factor: sqrt(deformed_area / template_area)
        face_scaling = torch.sqrt(face_area / (self._template_face_areas.unsqueeze(0) + 1e-8))  # [B, F, 1]

        # Create orthonormal basis for face orientation
        # Z-axis: normal
        # X-axis: first edge (normalized)
        # Y-axis: Z x X
        x_axis = F.normalize(e1, p=2, dim=2)  # [B, F, 3]
        z_axis = normal  # [B, F, 3]
        y_axis = torch.cross(z_axis, x_axis, dim=2)  # [B, F, 3]
        y_axis = F.normalize(y_axis, p=2, dim=2)

        # Recompute x_axis to ensure orthogonality
        x_axis = torch.cross(y_axis, z_axis, dim=2)  # [B, F, 3]

        # Create orientation matrix: [B, F, 3, 3]
        face_orien_mat = torch.stack([x_axis, y_axis, z_axis], dim=3)  # [B, F, 3 axes, 3 coords]

        return face_center, face_orien_mat, face_scaling

    def forward(
        self,
        body_pose,
        betas=None,
        global_orient=None,
        transl=None,
        pose2rot=True,
    ):
        """
        Forward pass with Gaussian deformation.

        Extends base forward() to also deform Gaussians when available.
        Matches NVHumanLayer signature (no return_verts/return_full_pose).
        """
        # Call base class forward for mesh deformation
        # NVHumanLayer always returns vertices, doesn't have return_verts parameter
        output = super().forward(
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
            pose2rot=pose2rot,
        )

        # Deform Gaussians if available
        if self.has_gaussians:
            vertices = output["vertices"]

            # Use face-based Gaussian deformation
            gaussian_params = self._deform_gaussians(vertices)
            output["gaussian_params"] = gaussian_params

        return output

    def _deform_gaussians(self, deformed_vertices):
        """
        Deform Gaussians using face-based transformation.
        Similar to Gaussian Avatar Haar approach.

        Args:
            deformed_vertices: [B, V, 3] - Deformed mesh vertices

        Returns:
            deformed_gaussian_params: Dict with deformed Gaussian parameters
        """
        batch_size = deformed_vertices.shape[0]
        num_gaussians = self.gaussian_positions.shape[0]
        device = deformed_vertices.device

        # Get original Gaussian positions: [B, N, 3]
        gaussian_xyz = self.gaussian_positions.unsqueeze(0).expand(batch_size, -1, -1)

        logger.debug("Deforming Gaussians using face-based transformations")

        # Get face binding: [N] - face ID for each Gaussian (gaussian_prim_ind)
        face_ind = self.gaussian_prim_ind  # [N]

        # Compute face properties from deformed mesh
        face_center, face_orien_mat, face_scaling = self.get_face_properties(
            deformed_vertices, self.faces_tensor
        )  # [B, F, 3], [B, F, 3, 3], [B, F, 1]

        # Transform Gaussian positions using face properties
        # Step 1: Apply rotation - [B, F, 3, 3] @ [N, 3, 1] for bound faces
        # We need to select the face properties for each Gaussian's bound face

        # Get face properties for bound faces: [B, N, ...]
        gaussian_face_center = face_center[:, face_ind]  # [B, N, 3]
        gaussian_face_orien_mat = face_orien_mat[:, face_ind]  # [B, N, 3, 3]
        gaussian_face_scaling = face_scaling[:, face_ind]  # [B, N, 1]

        # Apply face orientation to Gaussian positions: [B, N, 3, 3] @ [B, N, 3, 1] -> [B, N, 3, 1]
        rotated_xyz = torch.bmm(
            gaussian_face_orien_mat.reshape(-1, 3, 3),  # [B*N, 3, 3]
            gaussian_xyz.reshape(-1, 3, 1),  # [B*N, 3, 1]
        ).reshape(batch_size, num_gaussians, 3)  # [B, N, 3]

        # Step 2: Apply scaling and translation
        deformed_gaussian_positions = (
            rotated_xyz * gaussian_face_scaling.expand(-1, -1, 3) + gaussian_face_center
        )  # [B, N, 3]

        # Transform Gaussian scales using face scaling
        # Scales are stored in log space, so we ADD log(face_scaling) instead of multiplying
        original_scales = self.gaussian_scales.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 3] in log space
        log_face_scaling = torch.log(gaussian_face_scaling + 1e-8)  # [B, N, 1] convert to log space
        scales = original_scales + log_face_scaling.expand(-1, -1, 3)  # [B, N, 3] add in log space

        # Transform Gaussian rotations from local face coordinates to global coordinates
        # Convert face orientation matrices to quaternions, multiply with local rotations
        face_rotation_quat_xyzw = so3_matrix_to_quat(gaussian_face_orien_mat.reshape(-1, 3, 3), unbatch=False).reshape(
            batch_size, num_gaussians, 4
        )

        # Convert stored local rotations (WXYZ) to XYZW for multiplication
        local_rotations_wxyz = self.gaussian_rotations.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, N, 4] WXYZ (local face coords)
        local_rotations_xyzw = torch.cat([local_rotations_wxyz[..., 1:], local_rotations_wxyz[..., 0:1]], dim=-1)

        # Multiply quaternions: global = face_orientation ⊗ local
        rotations_xyzw = quat_mult_xyzw(face_rotation_quat_xyzw, local_rotations_xyzw)

        # Convert final result back to WXYZ for output (expected by rendering code)
        rotations = torch.cat(
            [rotations_xyzw[..., 3:4], rotations_xyzw[..., :3]], dim=-1
        )  # [B, N, 4] WXYZ (global coords)

        # Expand other attributes to batch size (unchanged)
        opacities = self.gaussian_opacities.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 1]
        shs = self.gaussian_shs.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N_sh]

        logger.debug(f"Face-based transformation applied to {num_gaussians} Gaussians")
        logger.debug(
            f"Position change magnitude: {torch.norm(deformed_gaussian_positions - gaussian_xyz, dim=-1).mean():.3f}"
        )
        logger.debug(f"Unique faces used: {len(torch.unique(face_ind))}/{self.faces_tensor.shape[0]}")

        return {
            "positions": deformed_gaussian_positions,  # [B, N, 3]
            "scales": scales,  # [B, N, 3]
            "rotations": rotations,  # [B, N, 4]
            "opacities": opacities,  # [B, N, 1]
            "shs": shs,  # [B, N, N_sh]
        }

    def reverse_deform_gaussians(
        self,
        gaussian_positions_global,
        gaussian_rotations_global,
        gaussian_prim_ind,
        reference_vertices,
        tpose_vertices,
    ):
        """
        Reverse deformation: Convert Gaussians from reference pose global coordinates to local face coordinates,
        and compute their T-pose global positions.

        This is the inverse operation of _deform_gaussians.

        Args:
            gaussian_positions_global: [N, 3] - Gaussian positions in reference pose (global coordinates)
            gaussian_rotations_global: [N, 4] - Gaussian rotations in reference pose (global, WXYZ quaternions)
            gaussian_prim_ind: [N] - Face index for each Gaussian
            reference_vertices: [V, 3] - Mesh vertices in reference pose
            tpose_vertices: [V, 3] - Mesh vertices in T-pose

        Returns:
            dict with:
                'local_positions': [N, 3] - Gaussian positions in local face coordinates
                'local_rotations': [N, 4] - Gaussian rotations in local face coordinates (WXYZ quaternions)
                'tpose_global_positions': [N, 3] - Gaussian positions in T-pose (global coordinates)
                'tpose_global_rotations': [N, 4] - Gaussian rotations in T-pose (global, WXYZ quaternions, for debug)
        """
        logger.debug("Reverse deformation: reference pose → local coordinates → T-pose")

        num_gaussians = gaussian_positions_global.shape[0]
        device = gaussian_positions_global.device

        # Compute face properties in reference pose
        logger.debug("Computing reference pose face properties...")
        ref_face_v0 = reference_vertices[self.faces_tensor[:, 0]]  # [F, 3]
        ref_face_v1 = reference_vertices[self.faces_tensor[:, 1]]  # [F, 3]
        ref_face_v2 = reference_vertices[self.faces_tensor[:, 2]]  # [F, 3]

        ref_face_center = (ref_face_v0 + ref_face_v1 + ref_face_v2) / 3.0  # [F, 3]
        ref_face_e1 = ref_face_v1 - ref_face_v0  # [F, 3]
        ref_face_e2 = ref_face_v2 - ref_face_v0  # [F, 3]
        ref_face_normal = torch.cross(ref_face_e1, ref_face_e2, dim=-1)  # [F, 3]
        ref_face_normal = F.normalize(ref_face_normal, p=2, dim=-1)  # [F, 3]

        # Build reference face orientation matrices [F, 3, 3]
        ref_face_e1_norm = F.normalize(ref_face_e1, p=2, dim=-1)  # [F, 3]
        ref_face_e2_ortho = ref_face_e2 - (ref_face_e2 * ref_face_e1_norm).sum(dim=-1, keepdim=True) * ref_face_e1_norm
        ref_face_e2_norm = F.normalize(ref_face_e2_ortho, p=2, dim=-1)  # [F, 3]

        ref_face_orien_mat = torch.stack([ref_face_e1_norm, ref_face_e2_norm, ref_face_normal], dim=-1)  # [F, 3, 3]

        # Get face properties for each Gaussian (reference pose)
        gaussian_ref_center = ref_face_center[gaussian_prim_ind]  # [N, 3]
        gaussian_ref_orien = ref_face_orien_mat[gaussian_prim_ind]  # [N, 3, 3]

        # Inverse transform: global → local coordinates
        # local = R^T @ (global - center)
        logger.debug("Converting to local face coordinates...")
        relative_pos = gaussian_positions_global - gaussian_ref_center  # [N, 3]
        local_positions = torch.bmm(
            gaussian_ref_orien.transpose(1, 2),  # [N, 3, 3] - transpose to get inverse rotation
            relative_pos.unsqueeze(-1),  # [N, 3, 1]
        ).squeeze(-1)  # [N, 3]

        # Compute face properties in T-pose
        logger.debug("Computing T-pose face properties...")
        tpose_face_v0 = tpose_vertices[self.faces_tensor[:, 0]]  # [F, 3]
        tpose_face_v1 = tpose_vertices[self.faces_tensor[:, 1]]  # [F, 3]
        tpose_face_v2 = tpose_vertices[self.faces_tensor[:, 2]]  # [F, 3]

        tpose_face_center = (tpose_face_v0 + tpose_face_v1 + tpose_face_v2) / 3.0  # [F, 3]
        tpose_face_e1 = tpose_face_v1 - tpose_face_v0  # [F, 3]
        tpose_face_e2 = tpose_face_v2 - tpose_face_v0  # [F, 3]
        tpose_face_normal = torch.cross(tpose_face_e1, tpose_face_e2, dim=-1)  # [F, 3]
        tpose_face_normal = F.normalize(tpose_face_normal, p=2, dim=-1)  # [F, 3]

        # Build T-pose face orientation matrices [F, 3, 3]
        tpose_face_e1_norm = F.normalize(tpose_face_e1, p=2, dim=-1)  # [F, 3]
        tpose_face_e2_ortho = (
            tpose_face_e2 - (tpose_face_e2 * tpose_face_e1_norm).sum(dim=-1, keepdim=True) * tpose_face_e1_norm
        )
        tpose_face_e2_norm = F.normalize(tpose_face_e2_ortho, p=2, dim=-1)  # [F, 3]

        tpose_face_orien_mat = torch.stack(
            [tpose_face_e1_norm, tpose_face_e2_norm, tpose_face_normal], dim=-1
        )  # [F, 3, 3]

        # Get face properties for each Gaussian (T-pose)
        gaussian_tpose_center = tpose_face_center[gaussian_prim_ind]  # [N, 3]
        gaussian_tpose_orien = tpose_face_orien_mat[gaussian_prim_ind]  # [N, 3, 3]

        # Forward transform: local → T-pose global coordinates
        # global = R @ local + center
        logger.debug("Converting to T-pose global coordinates...")
        tpose_global_positions = (
            torch.bmm(
                gaussian_tpose_orien,  # [N, 3, 3]
                local_positions.unsqueeze(-1),  # [N, 3, 1]
            ).squeeze(-1)
            + gaussian_tpose_center
        )  # [N, 3]

        # Convert rotations: global (reference) → local face coordinates
        logger.debug("Converting rotations to local face coordinates...")
        # Convert reference face orientation to quaternions [N, 4] XYZW
        ref_face_quat_xyzw = so3_matrix_to_quat(gaussian_ref_orien, unbatch=False)  # [N, 4] XYZW

        # Convert input rotations from WXYZ to XYZW for quaternion operations
        global_rot_xyzw = torch.cat(
            [gaussian_rotations_global[..., 1:], gaussian_rotations_global[..., 0:1]], dim=-1
        )  # [N, 4] XYZW

        # Compute inverse: local_rot = ref_face_rot^-1 ⊗ global_rot
        # Quaternion inverse: q^-1 = [x, y, z, w]^-1 = [-x, -y, -z, w] (for unit quaternions)
        ref_face_quat_inv_xyzw = torch.cat(
            [-ref_face_quat_xyzw[..., :3], ref_face_quat_xyzw[..., 3:4]], dim=-1
        )  # [N, 4] XYZW

        local_rot_xyzw = quat_mult_xyzw(ref_face_quat_inv_xyzw, global_rot_xyzw)  # [N, 4] XYZW

        # Convert back to WXYZ for storage
        local_rotations = torch.cat([local_rot_xyzw[..., 3:4], local_rot_xyzw[..., :3]], dim=-1)  # [N, 4] WXYZ

        # Compute T-pose global rotations (for debug visualization)
        logger.debug("Computing T-pose global rotations...")
        tpose_face_quat_xyzw = so3_matrix_to_quat(gaussian_tpose_orien, unbatch=False)  # [N, 4] XYZW
        tpose_global_rot_xyzw = quat_mult_xyzw(tpose_face_quat_xyzw, local_rot_xyzw)  # [N, 4] XYZW
        tpose_global_rotations = torch.cat(
            [tpose_global_rot_xyzw[..., 3:4], tpose_global_rot_xyzw[..., :3]], dim=-1
        )  # [N, 4] WXYZ

        logger.debug(f"Converted {num_gaussians} Gaussians (positions + rotations)")
        logger.debug(f"Local position range: [{local_positions.min():.3f}, {local_positions.max():.3f}]")
        logger.debug(f"Local rotation range: [{local_rotations.min():.3f}, {local_rotations.max():.3f}]")

        return {
            "local_positions": local_positions,
            "local_rotations": local_rotations,
            "tpose_global_positions": tpose_global_positions,
            "tpose_global_rotations": tpose_global_rotations,
        }
