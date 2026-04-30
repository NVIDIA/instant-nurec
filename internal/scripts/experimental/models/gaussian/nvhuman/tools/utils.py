import logging

import torch

from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay


logger = logging.getLogger(__name__)


# ================== Geometric Transform Helper Functions ================== #


def move_to_start_point_face_z(verts, J_regressor):
    """
    XZ to origin, Start from the ground, Face-Z
    Args:
        verts: (L, V, 3) - vertices
        J_regressor: (J, V) - joint regressor matrix
    Returns:
        verts: (L, V, 3) - transformed vertices
    """
    # position
    verts = verts.clone()  # (L, V, 3)
    offset = torch.einsum("j v, v i -> j i", J_regressor, verts[0])[0]  # (3) - root joint position
    offset[1] = verts[:, :, [1]].min()  # set y to ground level
    verts = verts - offset

    # face direction - compute transformation to face -Z
    joints = torch.einsum("j v, l v i -> l j i", J_regressor, verts[[0]])  # (1, J, 3)
    T_ay2ayfz = compute_T_ayfz2ay(joints, inverse=True)
    verts = apply_T_on_points(verts, T_ay2ayfz)

    return verts
