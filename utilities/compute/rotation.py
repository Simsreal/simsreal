import math

import numpy as np
import torch


def rotation_matrix(axis: str, angle: float) -> torch.Tensor:
    """
    Returns a 4x4 homogeneous rotation matrix for a rotation about
    axis ('x', 'y', or 'z') by 'angle' radians.
    """
    R = torch.eye(4)
    c, s = torch.cos(angle), torch.sin(angle)
    if axis == "x":
        R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    elif axis == "y":
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
    elif axis == "z":
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def rotate_around_z(xmat: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotates a 3×3 orientation matrix xmat by angle_degrees around the z-axis.
    """
    angle = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    # Basic rotation around the z-axis
    rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    return xmat @ rot


def rotate_vector_around_z(vec: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotates a 3D vector by angle_degrees around the z-axis.
    """
    angle = math.radians(angle_degrees)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    return R @ vec
