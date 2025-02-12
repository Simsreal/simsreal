import numpy as np


def rotate_axis(
    xpos: np.ndarray,
    xmat: np.ndarray,
    angle_deg: float,
    axis: str = "x",
    swap_axes: tuple[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotates around specified axis by arbitrary angle and optionally swaps axes.

    Args:
        xpos: Position vector of shape (3,)
        xmat: Rotation matrix of shape (3, 3)
        angle_deg: Rotation angle in degrees
        axis: Rotation axis ('x', 'y', or 'z')
        swap_axes: Optional tuple of axes to swap ('x', 'y', 'z')

    Returns:
        tuple: (new_xpos, new_xmat)
    """
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    # Basic rotation matrices around principal axes
    if axis.lower() == "x":
        rot_matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis.lower() == "y":
        rot_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis.lower() == "z":
        rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Apply rotation
    new_xmat = rot_matrix @ xmat

    # Optionally swap axes
    if swap_axes:
        axis1, axis2 = swap_axes
        axis_map = {"x": 0, "y": 1, "z": 2}
        if not all(ax in axis_map for ax in (axis1, axis2)):
            raise ValueError("Invalid axes for swapping")

        swap_matrix = np.eye(3)
        i, j = axis_map[axis1], axis_map[axis2]
        swap_matrix[i, i], swap_matrix[j, j] = 0, 0
        swap_matrix[i, j], swap_matrix[j, i] = 1, 1

        new_xmat = swap_matrix @ new_xmat

    return xpos, new_xmat


def check_head_orientation(xmat, forward_axis: str = "x"):
    # Extract rotation matrix columns
    mapping = {"x": 0, "y": 1, "z": 2}
    forward = xmat[:, mapping[forward_axis]]

    # Project forward vector onto horizontal plane
    forward_horizontal = np.array([forward[0], forward[1], 0])
    norm = np.linalg.norm(forward_horizontal)
    if norm < 1e-6:
        return 0.0, "center"
    forward_horizontal = forward_horizontal / norm

    # Calculate yaw
    yaw = np.arctan2(-forward_horizontal[1], -forward_horizontal[0])

    # Determine turn direction
    turn_direction = "right" if yaw > 0 else "left"

    return abs(yaw), turn_direction


def check_chest_orientation(current_xmat, forward_axis: str = "x"):
    mapping = {"x": 0, "y": 1, "z": 2}
    forward = current_xmat[:, mapping[forward_axis]]

    # Project forward vector onto horizontal plane
    # Use X and Y components (not Z) for horizontal plane
    forward_horizontal = np.array([forward[0], forward[1], 0])  # Using Y instead of Z
    norm = np.linalg.norm(forward_horizontal)
    if norm < 1e-6:
        return 0.0, "center"
    forward_horizontal = forward_horizontal / norm

    # Calculate yaw
    yaw = np.arctan2(forward_horizontal[1], forward_horizontal[0])
    turn_direction = "left" if yaw > 0 else "right"
    return abs(yaw), turn_direction
