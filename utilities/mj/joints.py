import numpy as np


def pd_control_joints(
    current_positions: np.ndarray,
    current_velocities: np.ndarray,
    desired_positions: np.ndarray,
    kp=50.0,
    kd=5.0,
):
    """
    PD controller for N joints.
    """
    error_pos = desired_positions - current_positions
    error_vel = -current_velocities
    torques = kp * error_pos + kd * error_vel
    return torques
