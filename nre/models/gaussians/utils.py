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
import math

from pathlib import Path
from typing import List, Optional, TypeAlias, TypeVar

import numpy as np
import point_cloud_utils as pcu
import torch

from torch import nn

from nre.utils.geometry import quat_to_so3_matrix, so3_matrix_to_quat
from nre.utils.types import PointCloud


log = logging.getLogger(__name__)

## NOTE: SPH code from gaussian-splatting, from plenoctree, from ???
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

T = TypeVar("T", np.ndarray, torch.Tensor)


def RGB2SH(rgb: T) -> T:
    return (rgb - 0.5) / C0


def SH2RGB(sh: T) -> T:
    return sh * C0 + 0.5


def sh_degree_to_specular_dim(degree: int) -> int:
    """Number of dimensions used by SH of deg [1..degree], inclusive"""
    return 3 * ((degree + 1) ** 2 - 1)


def sh_degree_to_num_features(degree: int) -> int:
    """Number of dimensions used by SH of deg [0..degree], inclusive"""
    return sh_degree_to_specular_dim(degree) + 3


def num_features_to_sh_degree(num_features: int) -> int:
    """
    Given num_features from sh_degree_to_num_features(d) = 3 * (d + 1)^2, compute the integer degree d.
    """
    # 1) Check that num_features is a multiple of 3
    assert num_features % 3 == 0, (
        f"num_features = {num_features} is not a multiple of 3, so it cannot match 3*(d+1)^2 for integer d"
    )

    # 2) Divide by 3
    squared_part = num_features // 3

    # 3) Check that squared_part is a perfect square:
    candidate = math.isqrt(squared_part)
    assert candidate * candidate == squared_part, (
        f"num_features = {num_features} implies {squared_part} is not a perfect square, so it cannot match (d+1)^2 for integer d"
    )

    # 4) Subtract 1 to get the degree
    degree = candidate - 1

    # 5) Optional: check for negative degree (if candidate == 0, that means no valid degree).
    assert degree >= 0, f"num_features = {num_features} is too small to represent degree 0 or higher"

    return degree


def cube_root(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3)


def spherical_to_cartesian(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def uniform_sample_sphere(
    num_samples: int, device: torch.device, inverse: bool = False, radius: float = 1.0
) -> torch.Tensor:
    """
    refer to https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    sample points uniformly inside a sphere
    """
    if not inverse:
        dist = torch.rand((num_samples,)).to(device)
        dist = cube_root(dist) * radius
    else:
        dist = torch.rand((num_samples,)).to(device)
        dist = 1 / dist.clamp_min(1 / radius)
    thetas = torch.arccos(2 * torch.rand((num_samples,)) - 1).to(device)
    phis = 2 * torch.pi * torch.rand((num_samples,)).to(device)
    pts = spherical_to_cartesian(dist, thetas, phis)
    return pts


def subsample_points(
    num_points: int,
    point_clouds: List[PointCloud],
    gaussian_cuboid_ids: List[torch.Tensor],
    gaussian_scales: List[torch.Tensor],
    observed_counts: List[int],
) -> tuple[List[PointCloud], List[torch.Tensor], List[torch.Tensor], List[int]]:
    assert num_points > 0, "num_points must be greater than 0"

    total_points = sum([pc.n_points for pc in point_clouds])
    if total_points > num_points:
        layer_sample_ratio = float(num_points) / total_points
        subsampled_point_clouds: List[PointCloud] = []
        subsampled_gaussian_cuboid_ids: List[torch.Tensor] = []
        subsampled_gaussian_scales: List[torch.Tensor] = []
        subsampled_observed_counts: List[int] = []

        for pc, gid, scale, observed_count in zip(point_clouds, gaussian_cuboid_ids, gaussian_scales, observed_counts):
            num_sampled_points = int(pc.n_points * layer_sample_ratio)
            if num_sampled_points == 0:
                continue
            random_idxs = torch.tensor(
                np.random.choice(pc.n_points, num_sampled_points, replace=False), device=pc.xyz_start.device
            )
            subsampled_point_clouds.append(pc[random_idxs])
            subsampled_gaussian_cuboid_ids.append(gid[random_idxs])
            subsampled_gaussian_scales.append(scale[random_idxs])
            subsampled_observed_counts.append(observed_count)

        point_clouds = subsampled_point_clouds
        gaussian_cuboid_ids = subsampled_gaussian_cuboid_ids
        gaussian_scales = subsampled_gaussian_scales
        observed_counts = subsampled_observed_counts

    return point_clouds, gaussian_cuboid_ids, gaussian_scales, observed_counts


def track_random_initialization(
    num_points: int, track_index: int, cuboid_extent: torch.Tensor, default_scale: float, device: torch.device
) -> tuple[PointCloud, torch.Tensor, torch.Tensor]:
    xyz_e_r_local = torch.rand((num_points, 3), device=device) * cuboid_extent - cuboid_extent / 2
    color_r = (torch.rand((num_points, 3), device=device) * 255.0).byte()
    # We don't currently use xyz_start for initialization so setting it to xyz_e_r_local, but if that changes
    # we need to determine what the right starting point for random initializations should be
    scale_r = torch.full((num_points,), default_scale, device=device)
    # we use scale_r as a point cloud argument before otherwise PointCloud.collate_fn will fail due to
    # inconsistent shapes
    point_cloud = PointCloud(
        xyz_start=xyz_e_r_local,
        xyz_end=xyz_e_r_local,
        color=color_r,
        camera_footprint_scale=scale_r,
    )
    gaussian_cuboid_id = torch.full(
        (num_points, 1),
        fill_value=track_index,
        dtype=torch.int32,
        device=device,
    )
    gaussian_scale = scale_r.unsqueeze(-1)
    return point_cloud, gaussian_cuboid_id, gaussian_scale


class PLYGaussianLoader:
    """
    Loads ply files generated by https://github.com/3DTopia/LGM/blob/main/core/gs.py#L101,
    which should be the format used by the original 3DGS implementation.

    This has been expanded to import additional properties exported by NRM, such as a road_mask
    (1 for each gaussian associated with the road) and sky_mask (float per gaussian, used for
    sky filtering). This class makes no assumptions about whether the gaussian properties
    (scale, density) are activated or not.
    """

    cuboids_dims: Optional[torch.Tensor]

    def __init__(self, path: Path, quaternion_format: str = "wxyz", device: str = "cuda") -> None:
        """
        quaternion_format (str): specifies the format of the rotation read from the PLY file.
            Must be either "wxyz" or "xyzw". This is used when transforming.
        """
        self.device = device

        assert (Path(path)).is_file(), f"{self.__class__.__name__} provided path {path} not a file"
        data = self._load_ply(path)

        self.positions = data["positions"]
        self.rotations = data["rotations"]
        self.densities = data["densities"]
        self.scales = data["scales"]
        self.features_albedo = data["features_albedo"]
        self.features_specular = data["features_specular"] if "features_specular" in data else None
        self.road_mask = data["road_mask"] if "road_mask" in data else None
        self.sky_mask = data["sky_mask"] if "sky_mask" in data else None
        self.cuboids_dims = None

        assert quaternion_format in ["xyzw", "wxyz"]
        self.quaternion_format = quaternion_format

    def transform(self, transform: torch.Tensor) -> None:
        """Applies a rigid transform to the gaussians in-place.

        transform (torch.Tensor): Expects the transform in the form of a homogeneous source -> target
            transformation:
            ⎡ R  -o ⎤
            ⎣ 0 1/s ⎦
        """
        self.positions = (self.positions @ transform[:3, :3].T) + transform[:3, 3]

        # quat_to_so3_matrix expects XYZW format
        if self.quaternion_format == "xyzw":
            quaternions = self.rotations
        else:
            quaternions = self.rotations[:, [1, 2, 3, 0]]

        rotations_matrix = quat_to_so3_matrix(quaternions, unbatch=False)
        quaternions = so3_matrix_to_quat(transform[:3, :3] @ rotations_matrix, unbatch=False)

        if self.quaternion_format == "xyzw":
            self.rotations = quaternions
        else:
            self.rotations = quaternions[:, [3, 0, 1, 2]]

    def scale(self, scale_factor: float, scale_activation: nn.Module, scale_activation_inv: nn.Module) -> None:
        """Scales the gaussians in-place"""
        self.positions *= scale_factor
        self.scales = scale_activation_inv(scale_activation(self.scales) * scale_factor)

    def _load_ply(self, path: Path) -> dict[str, torch.Tensor]:
        ply_data = pcu.load_triangle_mesh(path).vertex_data
        ply_metadata = ply_data.custom_attributes

        positions = torch.FloatTensor(ply_data.positions).to(self.device)
        densities = torch.FloatTensor(ply_metadata["opacity"]).unsqueeze(-1).to(self.device)

        features_albedo = torch.cat(
            [
                torch.FloatTensor(ply_metadata["f_dc_0"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["f_dc_1"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["f_dc_2"]).unsqueeze(-1),
            ],
            dim=1,
        ).to(self.device)

        scales = torch.cat(
            [
                torch.FloatTensor(ply_metadata["scale_0"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["scale_1"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["scale_2"]).unsqueeze(-1),
            ],
            dim=1,
        ).to(self.device)

        rotations = torch.cat(
            [
                torch.FloatTensor(ply_metadata["rot_0"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["rot_1"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["rot_2"]).unsqueeze(-1),
                torch.FloatTensor(ply_metadata["rot_3"]).unsqueeze(-1),
            ],
            dim=1,
        ).to(self.device)

        # Load all f_rest_x parameters into a buffer
        f_rest_keys = [key for key in ply_metadata.keys() if key.startswith("f_rest_")]
        f_rest_tensors = [torch.FloatTensor(ply_metadata[key]).unsqueeze(-1) for key in f_rest_keys]

        ret = {
            "positions": positions,
            "rotations": rotations,
            "densities": densities,
            "scales": scales,
            "features_albedo": features_albedo,
        }

        num_gaussians = positions.shape[0]
        num_speculars = len(f_rest_tensors) // 3
        features_specular: Optional[torch.Tensor] = None
        if f_rest_tensors:
            features_specular = (
                torch.cat(f_rest_tensors, dim=1)
                .to(self.device)
                .reshape(num_gaussians, 3, num_speculars)
                .transpose(2, 1)
                .reshape(num_gaussians, num_speculars * 3)
            )
            ret["features_specular"] = features_specular

        if "road_mask" in ply_metadata.keys():
            ret["road_mask"] = torch.BoolTensor(ply_metadata["road_mask"]).to(self.device)

        if "sky_mask" in ply_metadata.keys():  # "Mask" is a float per gaussian describing the sky probability
            sky = torch.FloatTensor(ply_metadata["sky_mask"]).to(self.device)
            ret["sky_mask"] = sky.unsqueeze(-1) if sky.dim() == 1 else sky

        return ret

    @classmethod
    def from_ply_bytes(
        cls, ply_bytes: bytes, device: str = "cuda", cuboids_dims: Optional[torch.Tensor] = None
    ) -> "PLYGaussianLoader":
        """
        Create loader from PLY bytes by writing to temporary file.

        Args:
            ply_bytes: PLY file contents as bytes
            device: Device to load tensors onto

        Returns:
            PLYGaussianLoader instance
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_file.write(ply_bytes)
            tmp_path = Path(tmp_file.name)

        try:
            loader = cls(path=tmp_path, device=device)
            loader.cuboids_dims = cuboids_dims
            return loader
        finally:
            tmp_path.unlink(missing_ok=True)


# Type alias for cleaner API
Asset: TypeAlias = PLYGaussianLoader


def write_ply_3dgs(
    path: Path,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    features_albedo: torch.Tensor,
    features_specular: torch.Tensor | None = None,
    color: torch.Tensor | None = None,
    normals: torch.Tensor | None = None,
    custom_attributes: dict[str, torch.Tensor] = {},
) -> None:
    """
    Writes a PLY file from the given tensors in the original 3DGS format.

    Note that the format should be compatible with the original 3DGS implementation but differences
    between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
    3rd-party 3DGS viewers.
    Note2: The given tensors should be the raw Gaussian parameters, not the activated ones (e.g., sigmoid, exp, relu, etc.).
    """
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = positions.cpu().numpy()

    if color is not None:
        mesh.vertex_data.colors = color.cpu().numpy()

    if normals is not None:
        assert normals.shape == positions.shape, "normals must have the same shape as positions"
        mesh.vertex_data.normals = normals.cpu().numpy()

    rotations_numpy = rotations.cpu().numpy()
    for attr_i in range(4):
        mesh.vertex_data.custom_attributes[f"rot_{attr_i}"] = rotations_numpy[..., attr_i]

    scales_numpy = scales.cpu().numpy()
    for attr_i in range(3):
        mesh.vertex_data.custom_attributes[f"scale_{attr_i}"] = scales_numpy[..., attr_i]

    mesh.vertex_data.custom_attributes["opacity"] = densities.cpu().numpy()

    features_albedo_numpy = features_albedo.cpu().numpy()
    for attr_i in range(3):
        mesh.vertex_data.custom_attributes[f"f_dc_{attr_i}"] = features_albedo_numpy[..., attr_i]

    num_gaussians = positions.shape[0]
    if features_specular is not None:
        num_speculars = features_specular.shape[-1] // 3
        features_specular_numpy = (
            features_specular.reshape((num_gaussians, num_speculars, 3))
            .transpose(2, 1)
            .reshape((num_gaussians, num_speculars * 3))
            .cpu()
            .numpy()
        )
        for attr_i in range(features_specular.shape[-1]):
            mesh.vertex_data.custom_attributes[f"f_rest_{attr_i}"] = features_specular_numpy[..., attr_i]

    for key, value in custom_attributes.items():
        mesh.vertex_data.custom_attributes[key] = value.cpu().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    log.info(f"Wrote {path.suffix}-file: {path.absolute()}")


def write_ply_3dgrt(
    path: Path,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    percentage_gaussians: float = 100,
) -> None:
    """
    Write a specified percentage of Gaussians into a PLY file, each Gaussian as a pseudo-colored mesh octahedron.

    Each Gaussian is represented as an octahedron transformed by the Gaussian parameters:
    gaussian = alpha * S @ R.t() @ v + mu, where S is the scale matrix, R is the rotation matrix, v are the canonical vertices of the octahedron, mu is the positions vector.
    alpha affects the size of each Gaussian as it's based on the density, alpha = sqrt(2 * log(density / 0.01)).
    The color of each Gaussian is based its direction, i.e. the sign of its own coordinate system:
    (+x, +y, +z, -x, -y, -z) = (red, green, blue, white, white, white)

    Note that the given tensors should be the Gaussian parameters after applying the activation function (e.g., sigmoid, exp, relu, etc.).

    Args:
        positions[torch.Tensor]: Positions of the 3D Gaussians (x, y, z) of shape [n_gaussians, 3]
        rotations[torch.Tensor]: Rotation of each Gaussian represented as a unit quaternion (xyzw) of shape [n_gaussians, 4]
        scales[torch.Tensor]: Anisotropic scale of each Gaussian of shape [n_gaussians, 3]
        densities[torch.Tensor]: Density of each Gaussian of shape [n_gaussians, 1]
        percentage_gaussians[float]: Use only n% of gaussians to reduce the number of vertices. Range (0, 100]. Default is 100.

    Returns:
        None
    """
    assert 0 < percentage_gaussians <= 100

    # Find invalid values (NaN, +-inf) in the three input tensors and remove them
    valid_mask = ~(
        ~torch.isfinite(positions).all(dim=1)
        | ~torch.isfinite(rotations).all(dim=1)
        | ~torch.isfinite(scales).all(dim=1)
        | ~torch.isfinite(densities).all(dim=1)
        | torch.isnan(positions).any(dim=1)
        | torch.isnan(rotations).any(dim=1)
        | torch.isnan(scales).any(dim=1)
        | torch.isnan(densities).any(dim=1)
    )

    n_gaussians_all = positions.shape[0]
    n_valid = int(torch.sum(valid_mask).item())
    if n_gaussians_all - n_valid > 0:
        log.info(
            f"Found {n_gaussians_all - n_valid} invalid gaussians out of {n_gaussians_all} total. Keeping {n_valid} valid gaussians"
        )
    if n_valid == 0:
        log.warning("No valid gaussians remaining after filtering! Exiting mesh construction early.")
        return

    positions = positions[valid_mask, ...]  # [n_valid, 3]
    rotations = rotations[valid_mask, ...]  # [n_valid, 4]
    scales = scales[valid_mask, ...]  # [n_valid, 3]
    densities = densities[valid_mask, ...]  # [n_valid, 1]

    # Remove Gaussians according to percentage_gaussians, but keep at least 1 Gaussian
    n_gaussians_valid = positions.shape[0]
    n_gaussians_to_keep = max(1, int(n_gaussians_valid * percentage_gaussians / 100))
    log.info(
        f"Using {n_gaussians_to_keep} gaussians out of {n_gaussians_valid} ({percentage_gaussians:.1f}%) via random sampling"
    )

    # Generate random indices for uniform sampling
    if n_gaussians_to_keep < n_gaussians_valid:
        random_indices = torch.randperm(n_gaussians_valid, device=positions.device)[:n_gaussians_to_keep]
        random_indices = torch.sort(random_indices)[0]  # Sort for better memory access patterns

        positions = positions[random_indices, :]  # [n_gaussians_to_keep, 3]
        rotations = rotations[random_indices, :]  # [n_gaussians_to_keep, 4]
        scales = scales[random_indices, :]  # [n_gaussians_to_keep, 3]
        densities = densities[random_indices, :]  # [n_gaussians_to_keep, 1]

    # Build the canonical octahedron
    octa_canonical = torch.cat(
        [torch.eye(3, device=scales.device), -torch.eye(3, device=scales.device)], dim=-1
    )  # [3, 6]
    n_vertices_per_gaussian = octa_canonical.shape[1]

    # Vertices of the mesh
    # Transform octahedron using alpha * S @ R.t() @ v + mu formula from 3DGRT paper
    alpha_min = 0.005  # default opacity threshold
    densities_clamped = densities.clamp_min(alpha_min)
    alpha = torch.sqrt(2 * torch.log(densities_clamped / alpha_min))  # [n_gaussians_to_keep, 1]
    rot_mats = quat_to_so3_matrix(rotations, unbatch=False)  # [n_gaussians_to_keep, 3, 3]
    scale_mats = torch.diag_embed(scales)  # [n_gaussians_to_keep, 3, 3]
    gaussians = (
        alpha[:, None] * scale_mats @ rot_mats.transpose(-1, -2) @ octa_canonical[None, :, :] + positions[:, :, None]
    )  # [n_gaussians_to_keep, 3, n_vertices_per_gaussian]
    gaussians = gaussians.transpose(dim0=-1, dim1=-2)  # [n_gaussians_to_keep, n_vertices_per_gaussian, 3]

    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = gaussians.reshape(-1, 3).cpu().numpy()  # [V, 3]

    # Faces of the mesh
    faces = generate_octahedron_faces(octa_canonical.transpose(1, 0)[None, :, :])  # [1, 8, 3]
    # Compute the vertices index offsets for each gaussian
    vertex_offsets = (
        torch.arange(gaussians.shape[0], device=faces.device) * n_vertices_per_gaussian
    )  # [n_gaussians_to_keep]
    faces = faces + vertex_offsets[:, None, None]  # [n_gaussians_to_keep, 8, 3]
    mesh.face_data.vertex_ids = faces.reshape(-1, 3).cpu().numpy()

    # Color (RGBA) the gaussians in the mesh (red is x+, green is y+, blue is z+, rest is white)
    colors = np.array(
        gaussians.shape[0]
        * [
            [1, 0, 0, 1],  # (1,0,0) x+
            [0, 1, 0, 1],  # (0,1,0) y+
            [0, 0, 1, 1],  # (0,0,1) z+
            [1, 1, 1, 1],  # (-1,0,0) x-
            [1, 1, 1, 1],  # (0,-1,0) y-
            [1, 1, 1, 1],  # (0,0,-1) z-
        ]
    )  # [n_gaussians_to_keep * 6, 4]
    mesh.vertex_data.colors = colors

    # Save the mesh
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    logging.info(f"Saved mesh file: {path}")


def generate_octahedron_faces(vertices: torch.Tensor) -> torch.Tensor:
    """
    Generating octahedron faces from batches of six 3D vertices.

    Args:
        vertices: torch tensor of shape (..., 6, 3) containing sets of six 3D points

    Returns:
        torch tensor of shape (..., 8, 3) containing face indices with correct winding
    """
    original_shape = vertices.shape
    assert original_shape[-2:] == (6, 3), "Last two dimensions must be (6, 3)"

    # Reshape to (N, 6, 3) for batch processing
    batch_shape = original_shape[:-2]
    n_octahedra = int(torch.prod(torch.tensor(batch_shape)).item()) if batch_shape else 1
    vertices_flat = vertices.reshape(n_octahedra, 6, 3)

    # Define the 8 triangular faces of the octahedron
    # Note: The faces have the correct winding order in the canonical octahedron: (+-1,0,0), (0,+-1,0), (0,0,+-1).
    face_templates = [
        # Top pyramid (around +z vertex, index 2)
        [0, 1, 2],  # +x, +y, +z
        [1, 3, 2],  # +y, -x, +z
        [3, 4, 2],  # -x, -y, +z
        [4, 0, 2],  # -y, +x, +z
        # Bottom pyramid (around -z vertex, index 5)
        [0, 4, 5],  # +x, -y, -z
        [4, 3, 5],  # -y, -x, -z
        [3, 1, 5],  # -x, +y, -z
        [1, 0, 5],  # +y, +x, -z
    ]

    # Compute face indices for given octahedra defined by vertices such that the normals point outward (correct winding order).
    # Initialize output array
    faces_flat = torch.zeros((n_octahedra, 8, 3), dtype=torch.long, device=vertices.device)

    # Process each face template
    for face_idx, face_template in enumerate(face_templates):
        v0, v1, v2 = face_template

        # Vectorized computation for all octahedra
        edge1 = vertices_flat[:, v1] - vertices_flat[:, v0]  # (N, 3)
        edge2 = vertices_flat[:, v2] - vertices_flat[:, v0]  # (N, 3)

        # Cross product for all octahedra
        normals = torch.cross(edge1, edge2)  # (N, 3)

        # Calculate face centers
        face_centers = (vertices_flat[:, v0] + vertices_flat[:, v1] + vertices_flat[:, v2]) / 3  # (N, 3)

        # Outward direction (normalized face center for octahedron)
        face_center_norms = torch.norm(face_centers, dim=1, keepdim=True)  # (N, 1)
        outward_directions = face_centers / (face_center_norms + 1e-8)  # (N, 3)

        # Dot product to check if normal points outward
        dot_products = torch.sum(normals * outward_directions, dim=1)  # (N,)

        # Set face indices based on dot product sign
        faces_flat[:, face_idx] = torch.where(
            dot_products[:, None] < 0,  # If normal points inward
            torch.tensor([v0, v2, v1], device=vertices.device),  # Reverse winding
            torch.tensor([v0, v1, v2], device=vertices.device),  # Keep original winding
        )

    # Reshape back to original batch shape
    output_shape = batch_shape + (8, 3)
    return faces_flat.reshape(output_shape)
