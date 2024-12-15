from dataclasses import dataclass, field
from typing import List

import numpy as np
from geometry_msgs.msg import Quaternion, Vector3


@dataclass
class JointStateMessage:
    """
    ROS2 JointState message schema.
    """

    timestamp: float
    name: List[str] = field(
        default_factory=lambda: [
            "pelvis",
            "right_lower_arm",
            "left_lower_arm",
            "right_shin",
            "left_shin",
        ]
    )
    position: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    effort: List[float] = field(default_factory=list)


@dataclass
class ImageMessage:
    """
    ROS2 Image message schema.
    """

    timestamp: float
    height: int
    width: int
    encoding: str
    data: np.array


@dataclass
class ImuMessage:
    """
    ROS2 Imu message schema.
    """

    timestamp: float
    orientation: Quaternion
    orientation_covariance: np.array
    angular_velocity: Vector3
    angular_velocity_covariance: np.array
    linear_acceleration: Vector3
    linear_acceleration_covariance: np.array
