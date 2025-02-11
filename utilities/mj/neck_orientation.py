import numpy as np
from scipy.spatial.transform import Rotation as R


def orientation_for_target_direction(
    desired_dir_world, forward_local=np.array([1, 0, 0])
):
    """
    Build a rotation R_head_world such that it rotates `forward_local`
    (in the head’s local space) to `desired_dir_world` (in world space).

    Parameters
    ----------
    desired_dir_world : (3,) array-like
        The desired heading direction in world coords (does not need to be normalized).
    forward_local : (3,) array-like, default = [1, 0, 0]
        Which local axis we consider 'forward' for the head.

    Returns
    -------
    rot_head_world : scipy.spatial.transform.Rotation
        The rotation in world space that aligns the local +X axis
        with the desired direction.
    """
    # Normalize the desired direction
    desired_dir_world = np.array(desired_dir_world, dtype=float)
    desired_dir_world /= np.linalg.norm(desired_dir_world)

    forward_local = np.array(forward_local, dtype=float)
    forward_local /= np.linalg.norm(forward_local)

    # Determine axis and angle via cross/dot for a single rotation
    axis = np.cross(forward_local, desired_dir_world)
    axis_norm = np.linalg.norm(axis)
    if (
        axis_norm < 1e-12
    ):  # forward_local is parallel or anti-parallel to desired_dir_world
        dot_val = np.dot(forward_local, desired_dir_world)
        # If dot < 0, they are pointing opposite (180 deg needed)
        if dot_val < -0.999999:
            # 180 deg rotation about any vector perpendicular to forward_local
            axis = np.array([0, 1, 0])  # pick an arbitrary perpendicular axis
            angle = np.pi
        else:
            # They are already aligned
            axis = np.array([0, 1, 0])
            angle = 0.0
    else:
        # They are not parallel or anti-parallel
        axis = axis / axis_norm
        dot_val = np.dot(forward_local, desired_dir_world)
        # clamp to avoid floating errors outside [-1,1]
        dot_val = max(-1.0, min(1.0, dot_val))
        angle = np.arccos(dot_val)

    rot_head_world = R.from_rotvec(axis * angle)
    return rot_head_world


def compute_neck_angles_to_face_direction(
    chest_xmat,
    desired_dir_world,
    euler_order="xyz",
    model=None,
):
    """
    Computes neck_x, neck_y, neck_z angles so that the head's local +X axis
    (assuming that is forward_local) aligns with the desired_dir_world.

    This function:
      1) Reads the chest rotation from chest_xmat.
      2) Builds a rotation that points local +X to desired_dir_world.
      3) Computes the relative rotation from chest to that target head orientation.
      4) Extracts Euler angles in the specified order (default 'xyz').

    Parameters
    ----------
    chest_xmat : (3, 3) array-like
        The chest rotation in world coords.
    desired_dir_world : (3,) array-like
        The desired heading direction in world coords.
    euler_order : str, default = 'xyz'
        The Euler angle order to match your MuJoCo joint axes.
    model : mjModel (optional, can be unused if you only need data)

    Returns
    -------
    (rx, ry, rz) : tuple of floats
        The Euler angles that, when assigned to your 3 hinge joints in sequence,
        orient the head so that local +X faces the desired direction in world space.
    """
    # 1) Chest orientation in world coords
    chest_R_world = chest_xmat.reshape(3, 3)
    rot_chest_world = R.from_matrix(chest_R_world)

    # 2) Rotation so that local +X aligns with desired_dir_world
    rot_head_world = orientation_for_target_direction(
        desired_dir_world, forward_local=[1, 0, 0]
    )

    # 3) chest->head = rot_chest_world.inv() * rot_head_world
    rot_chest_to_head = rot_chest_world.inv() * rot_head_world

    # 4) Extract Euler angles in 'xyz' order (or whatever you need)
    angles_xyz = rot_chest_to_head.as_euler(euler_order, degrees=False)

    return tuple(angles_xyz)
